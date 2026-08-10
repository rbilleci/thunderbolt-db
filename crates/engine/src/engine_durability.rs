//! ADR-014 canonical WAL integration and recovery-lineage validation.

use super::engine_canonical_operation::{
    SealedCanonicalOperation, ENGINE_OPERATION_CODEC_RESOLVED_BINARY,
    ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, ENGINE_OPERATION_MAGIC,
};
use super::*;

mod canonical_envelope;
mod canonical_records;

const TRANSACTION_STATUS_MAGIC: &[u8; 12] = b"GPUDBSTATUS1";
const CANONICAL_TRANSACTION_CLAIM_STATUS_BYTES: usize = TRANSACTION_STATUS_MAGIC.len()
    + std::mem::size_of::<[u8; 16]>()
    + std::mem::size_of::<[u8; 16]>()
    + std::mem::size_of::<TxnId>()
    + std::mem::size_of::<gpu_db_wal::CanonicalDigest>()
    + std::mem::size_of::<u8>()
    + 7
    + std::mem::size_of::<u64>();

pub(crate) const fn canonical_transaction_claim_status_len() -> usize {
    CANONICAL_TRANSACTION_CLAIM_STATUS_BYTES
}
const ENGINE_OPERATION_CODEC_LEGACY_SQL: u8 = 1;
/// Historical canonical typed commands contain bare canonical JSON and replay with pre-PRODUCT-001
/// catalog semantics.
const ENGINE_OPERATION_CODEC_TYPED_COMMAND_V1: u8 = 3;
/// Additive discriminator for commands emitted after stable index OIDs/dependency policy landed.
const ENGINE_TYPED_COMMAND_TAG: u8 = 0xfe;
const ENGINE_TYPED_COMMAND_VERSION_LEGACY: u8 = 1;
const ENGINE_TYPED_COMMAND_VERSION_CURRENT: u8 = 2;
static CANONICAL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[allow(clippy::large_enum_variant)] // replay retains the validated S7 artifact without a second owner/allocation
enum PreparedDurableReplayRecord {
    EngineOperation(WalRecord),
    WriteAuthority,
    SemanticsV2TypedInsert {
        artifact: crate::typed_insert_aggregate::SemanticsV2ReplayArtifact,
        protocol: SemanticsV2ReplayProtocol,
    },
}

/// Bit 30 is an authenticated wire-level ownership discriminator, not a heuristic based on
/// whether a partial historical control prefix happened to be restored. The retired writer set
/// it on every production three-record terminal; the generic one-record writer clears it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticsV2ReplayProtocol {
    HistoricalWriteAuthority,
    GenericOneRecord,
}

/// Preflight the clear-bit codec-5 allocator authority before a recovery candidate can retain a
/// source, launch generation, or mutate a device plan.  Bit 30 selected this protocol already;
/// absence of markers is therefore a required fact, not a historical-chain inference.
fn validate_generic_codec5_replay_preflight(
    write_authority: &crate::engine_write_authority::DurableWriteAuthorityIndex,
    stable_transaction_id: TxnId,
    row_allocator_before: u64,
    row_allocator_high_water: u64,
    affected_rows: u64,
    simulated_allocator_high_water: u64,
) -> Result<u64, EngineError> {
    if write_authority.has_any_parent_marker(stable_transaction_id) {
        return Err(EngineError::Durability(format!(
            "generic one-record codec-5 transaction {stable_transaction_id} has retired claim or allocator markers",
        )));
    }
    if row_allocator_before != simulated_allocator_high_water
        || row_allocator_high_water
            != row_allocator_before
                .checked_add(affected_rows)
                .ok_or_else(|| {
                    EngineError::Durability(
                        "generic codec-5 replay allocator range overflows".to_string(),
                    )
                })?
    {
        return Err(EngineError::Durability(format!(
            "generic one-record codec-5 transaction {stable_transaction_id} allocator range is not the exact replay frontier",
        )));
    }
    Ok(row_allocator_high_water)
}

impl Engine {
    fn canonical_genesis_catalog_digest(
        identity: gpu_db_wal::CanonicalIdentity,
    ) -> gpu_db_wal::CanonicalDigest {
        let mut body = Vec::with_capacity(72);
        body.extend_from_slice(b"GPUDBCATALOGGENESIS1");
        body.extend_from_slice(&identity.database_id);
        body.extend_from_slice(&identity.cluster_id);
        body.extend_from_slice(&identity.timeline_id);
        body.extend_from_slice(&identity.format_epoch.to_le_bytes());
        gpu_db_wal::canonical_request_digest(&body)
    }

    pub(crate) fn canonical_catalog_transition(
        before: gpu_db_wal::CanonicalDigest,
        kind: gpu_db_wal::CanonicalFragmentKind,
        operation_body: &[u8],
    ) -> gpu_db_wal::CanonicalDigest {
        if kind != gpu_db_wal::CanonicalFragmentKind::CatalogMutation {
            return before;
        }
        let mut body = Vec::with_capacity(64 + operation_body.len());
        body.extend_from_slice(b"GPUDBCATALOGTRANSITION1");
        body.extend_from_slice(&before);
        body.extend_from_slice(operation_body);
        gpu_db_wal::canonical_request_digest(&body)
    }

    pub(crate) fn canonical_catalog_boundary(
        identity: gpu_db_wal::CanonicalIdentity,
        prior: Option<gpu_db_wal::CanonicalCatalogTail>,
    ) -> Result<(u64, gpu_db_wal::CanonicalDigest), EngineError> {
        let Some(prior) = prior else {
            return Ok((0, Self::canonical_genesis_catalog_digest(identity)));
        };
        if prior.identity != identity {
            return Err(EngineError::Durability(
                "canonical catalog boundary crosses database lineage".to_string(),
            ));
        }
        Ok((prior.catalog_after_epoch, prior.catalog_after_digest))
    }

    pub(crate) fn fresh_canonical_identity() -> gpu_db_wal::CanonicalIdentity {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let counter = CANONICAL_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let mut seed = Vec::with_capacity(40);
        seed.extend_from_slice(&now.to_le_bytes());
        seed.extend_from_slice(&counter.to_le_bytes());
        seed.extend_from_slice(&std::process::id().to_le_bytes());
        let derive = |domain: u8| {
            let mut material = seed.clone();
            material.push(domain);
            let digest = gpu_db_wal::canonical_request_digest(&material);
            let mut id = [0_u8; 16];
            id.copy_from_slice(&digest[..16]);
            id
        };
        gpu_db_wal::CanonicalIdentity {
            database_id: derive(1),
            cluster_id: derive(2),
            timeline_id: derive(3),
            format_epoch: 1,
        }
    }

    pub(crate) fn install_fresh_durable_identity(
        &self,
        base: &std::path::Path,
    ) -> Result<(), EngineError> {
        let identity = self.commit_state().canonical_identity;
        gpu_db_wal::write_durable_identity(base, identity)?;
        self.commit_state().canonical_lineage_bound = true;
        Ok(())
    }

    pub(crate) fn bind_durable_identity_for_recovery(
        &self,
        base: &std::path::Path,
        records: &[WalRecord],
    ) -> Result<(), EngineError> {
        self.prepare_legacy_index_oid_recovery(records)?;
        let mut record_identity = None;
        for record in records {
            let Some(envelope) = gpu_db_wal::decode_canonical_record_payload(&record.payload)?
            else {
                continue;
            };
            match record_identity {
                None => record_identity = Some(envelope.header.identity),
                Some(expected) if expected == envelope.header.identity => {}
                Some(_) => {
                    return Err(EngineError::Durability(
                        "canonical WAL lineage changes within the recovery source".to_string(),
                    ));
                }
            }
        }
        match (gpu_db_wal::read_durable_identity(base)?, record_identity) {
            (Some(anchor), Some(records)) if anchor != records => {
                Err(EngineError::Durability(format!(
                    "durable identity anchor beside {} does not match canonical WAL lineage",
                    base.display()
                )))
            }
            (Some(anchor), _) => {
                let mut commit = self.commit_state();
                commit.canonical_identity = anchor;
                commit.canonical_lineage_bound = true;
                Ok(())
            }
            (None, Some(_)) => Err(EngineError::Durability(format!(
                "canonical WAL exists beside {} but its durable identity anchor is missing",
                base.display()
            ))),
            (None, None) => self.install_fresh_durable_identity(base),
        }
    }

    fn recovery_scan_payload(record: &WalRecord) -> Result<Option<Arc<[u8]>>, EngineError> {
        if let Some(envelope) = gpu_db_wal::decode_canonical_record_payload(&record.payload)? {
            // Codec-5 has its own strict retained recovery owner. The historical index-OID
            // preflight must neither decode it as a legacy engine operation nor allocate a
            // throwaway typed image; the actual replay preparation below closes it once.
            if crate::engine_write_authority::decode_write_authority_envelope(&envelope)?.is_some()
                || Self::canonical_envelope_is_codec5(&envelope)
            {
                return Ok(None);
            }
            let operation = envelope.fragments.first().ok_or_else(|| {
                EngineError::Durability(format!(
                    "canonical WAL record {} has no operation fragment",
                    record.txn_id
                ))
            })?;
            return Self::decode_engine_operation(&operation.body).map(Some);
        }
        if record.payload.first() == Some(&WAL_BINARY_TAG)
            || record.payload.first() == Some(&ENGINE_TYPED_COMMAND_TAG)
        {
            return Ok(Some(Arc::clone(&record.payload)));
        }
        let command = Self::decode_engine_command(&record.payload)?.ok_or_else(|| {
            EngineError::Durability(format!(
                "legacy WAL record {} has no engine command",
                record.txn_id
            ))
        })?;
        Self::encode_replay_typed_command(&command, ENGINE_TYPED_COMMAND_VERSION_LEGACY).map(Some)
    }

    pub(crate) fn canonical_envelope_is_codec5(envelope: &gpu_db_wal::CanonicalEnvelope) -> bool {
        envelope
            .fragments
            .first()
            .is_some_and(|fragment| {
                fragment.kind == gpu_db_wal::CanonicalFragmentKind::RowMutation
                    && fragment.body.get(8)
                        == Some(&crate::typed_insert_aggregate::ENGINE_OPERATION_CODEC_TYPED_INSERT_AGGREGATE)
            })
    }

    fn legacy_catalog_oid_allocations(command: &Command) -> Result<u32, EngineError> {
        let count = match command {
            Command::CreateTable(create) => 1usize.saturating_add(
                create
                    .columns
                    .iter()
                    .filter(|column| {
                        matches!(
                            column.default,
                            Some(ColumnDefault::SequenceNextVal {
                                create_if_missing: true,
                                ..
                            })
                        )
                    })
                    .count(),
            ),
            Command::CreateView(_)
            | Command::CreateMaterializedView(_)
            | Command::CreateFunction(_)
            | Command::CreateSequence(_)
            | Command::CreateDomain(_)
            | Command::CreateDatabase(_)
            | Command::CreateTablespace(_)
            | Command::CreatePublication(_)
            | Command::CreateSubscription(_)
            | Command::CreateRole(_) => 1,
            _ => 0,
        };
        u32::try_from(count).map_err(|_| {
            EngineError::Durability(
                "historical catalog operation has too many implicit OID allocations".to_string(),
            )
        })
    }

    /// Inspect the complete historical prefix before replay can synthesize any index identity.
    /// Pre-PRODUCT-001 indexes did not consume `relational_next_oid`, so a one-pass replay could
    /// otherwise assign an index OID that a later old table record still owns.  This read-only pass
    /// establishes a conservative low-range high-water first; replay then uses a separate
    /// recovery-only cursor and folds it into the shared allocator at the first current epoch.
    pub(crate) fn prepare_legacy_index_oid_recovery(
        &self,
        records: &[WalRecord],
    ) -> Result<(), EngineError> {
        // File/lane recovery binds and scans the complete durable source first, then replays its
        // checkpoint/prefix and suffix in separate calls. Once an earlier replay chunk has crossed
        // the one-way index-identity boundary, a later chunk must inherit that fact: byte-stable
        // index-neutral legacy opcodes remain legal, but they must not be reclassified as an old
        // low-OID prefix. A fresh engine also starts `index_oid_epoch_current=true`, so only trust
        // the value after the complete-source floor has been prepared.
        let mut current_epoch_seen = {
            let catalog = self.ddl_catalog();
            catalog.legacy_recovery_floor_prepared && catalog.index_oid_epoch_current
        };
        let mut low_oid_high_water = FIRST_USER_RELATION_OID;
        let mut has_legacy_prefix = false;
        for record in records {
            let Some(payload) = Self::recovery_scan_payload(record)? else {
                continue;
            };
            if payload.first() == Some(&WAL_BINARY_TAG) {
                let BinaryWalRecord::Transaction(transaction) = decode_binary_record(&payload)?
                else {
                    continue;
                };
                if transaction.catalog_commands.is_empty() {
                    continue;
                }
                match transaction.catalog_epoch {
                    BinaryTransactionCatalogEpoch::IndexIdentityV1 => {
                        current_epoch_seen = true;
                    }
                    BinaryTransactionCatalogEpoch::Legacy => {
                        let requires_index_epoch =
                            transaction.catalog_commands.iter().any(|operation| {
                                command_requires_index_catalog_opcode(&operation.command)
                            });
                        if current_epoch_seen && requires_index_epoch {
                            return Err(EngineError::Durability(
                                "legacy transaction catalog record follows the PRODUCT-001 index identity boundary"
                                    .to_string(),
                            ));
                        }
                        if current_epoch_seen {
                            // Byte-stable index-neutral opcodes remain legal after the one-way
                            // migration. They execute against the already-current shared
                            // allocator and cannot synthesize another legacy index identity.
                            continue;
                        }
                        has_legacy_prefix |= requires_index_epoch;
                        if let Some(output) = &transaction.catalog_output {
                            low_oid_high_water = low_oid_high_water.max(output.relational_next_oid);
                        } else {
                            for operation in &transaction.catalog_commands {
                                low_oid_high_water = low_oid_high_water
                                    .checked_add(Self::legacy_catalog_oid_allocations(
                                        &operation.command,
                                    )?)
                                    .ok_or_else(|| {
                                        EngineError::Durability(
                                            "historical catalog OID prefix overflows".to_string(),
                                        )
                                    })?;
                            }
                        }
                    }
                }
                continue;
            }
            let command = Self::decode_engine_command(&payload)?.ok_or_else(|| {
                EngineError::Durability(format!(
                    "WAL record {} has no typed catalog command",
                    record.txn_id
                ))
            })?;
            if Self::engine_command_uses_current_index_semantics(&payload) {
                current_epoch_seen = true;
                continue;
            }
            if current_epoch_seen && command_requires_index_catalog_opcode(&command) {
                return Err(EngineError::Durability(
                    "legacy typed command follows the PRODUCT-001 index identity boundary"
                        .to_string(),
                ));
            }
            if current_epoch_seen {
                continue;
            }
            has_legacy_prefix |= command_requires_index_catalog_opcode(&command);
            low_oid_high_water = low_oid_high_water
                .checked_add(Self::legacy_catalog_oid_allocations(&command)?)
                .ok_or_else(|| {
                    EngineError::Durability("historical catalog OID prefix overflows".to_string())
                })?;
        }
        let mut catalog = self.ddl_catalog();
        catalog.prepare_legacy_index_oid_recovery_floor(
            low_oid_high_water.max(FIRST_LEGACY_RECOVERY_INDEX_OID),
            has_legacy_prefix,
        )
    }

    fn canonical_fragment_kind(
        payload: &[u8],
    ) -> Result<gpu_db_wal::CanonicalFragmentKind, EngineError> {
        if payload.first() == Some(&WAL_BINARY_TAG) {
            return Ok(match decode_binary_record(payload)? {
                crate::wal_binary::BinaryWalRecord::Transaction(record)
                    if !record.catalog_commands.is_empty() =>
                {
                    gpu_db_wal::CanonicalFragmentKind::CatalogMutation
                }
                crate::wal_binary::BinaryWalRecord::Transaction(record)
                    if !record.table_resets.is_empty() =>
                {
                    gpu_db_wal::CanonicalFragmentKind::TableReset
                }
                crate::wal_binary::BinaryWalRecord::SequenceValueTransition(_) => {
                    gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition
                }
                _ => gpu_db_wal::CanonicalFragmentKind::RowMutation,
            });
        }
        let command = Self::decode_engine_command(payload)?.ok_or_else(|| {
            EngineError::Durability(
                "canonical WAL operation is neither a resolved binary record nor a typed command"
                    .to_string(),
            )
        })?;
        Ok(match command {
            Command::TruncateTable(_) => gpu_db_wal::CanonicalFragmentKind::TableReset,
            Command::RefreshMaterializedView(_) => gpu_db_wal::CanonicalFragmentKind::TableRewrite,
            Command::SequenceNextVal(_) | Command::SequenceSetVal(_) => {
                gpu_db_wal::CanonicalFragmentKind::SequenceValueTransition
            }
            Command::SetKv { .. }
            | Command::DeleteKv { .. }
            | Command::Insert(_)
            | Command::Delete(_)
            | Command::Update(_) => gpu_db_wal::CanonicalFragmentKind::RowMutation,
            _ => gpu_db_wal::CanonicalFragmentKind::CatalogMutation,
        })
    }

    fn canonical_allocator_high_water(payload: &[u8]) -> Result<u64, EngineError> {
        let referenced_next = |row_id: u64| {
            row_id.checked_add(1).ok_or_else(|| {
                EngineError::Durability(
                    "canonical WAL references the reserved maximum row identity".to_string(),
                )
            })
        };
        if payload.first() != Some(&WAL_BINARY_TAG) {
            return Ok(0);
        }
        match decode_binary_record(payload)? {
            crate::wal_binary::BinaryWalRecord::Insert(record) => record
                .rows
                .iter()
                .map(|(row_id, _)| referenced_next(*row_id))
                .try_fold(0, |high, next| next.map(|next| high.max(next))),
            crate::wal_binary::BinaryWalRecord::DeleteByKey(_) => Ok(0),
            crate::wal_binary::BinaryWalRecord::UpdateByKey(record) => {
                referenced_next(record.new_row_id)
            }
            crate::wal_binary::BinaryWalRecord::Transaction(record) => record
                .mutations
                .iter()
                .map(|mutation| match mutation {
                    crate::wal_binary::BinaryTransactionMutation::Insert { row_id, .. }
                    | crate::wal_binary::BinaryTransactionMutation::Update { row_id, .. }
                    | crate::wal_binary::BinaryTransactionMutation::Delete { row_id, .. } => {
                        referenced_next(*row_id)
                    }
                })
                .try_fold(record.allocator_high_water, |high, next| {
                    next.map(|next| high.max(next))
                }),
            crate::wal_binary::BinaryWalRecord::SequenceValueTransition(_) => Ok(0),
        }
    }

    /// Decode the replay payload for a typed command. Live callers still hand the engine SQL text;
    /// canonical recovery hands it the versioned AST payload below. The latter contains no SQL
    /// source and therefore cannot invoke the SQL parser during replay.
    pub(crate) fn decode_engine_command(payload: &[u8]) -> Result<Option<Command>, EngineError> {
        if payload.first() == Some(&WAL_BINARY_TAG) {
            return Ok(None);
        }
        if payload.first() != Some(&ENGINE_TYPED_COMMAND_TAG) {
            let text = std::str::from_utf8(payload).map_err(|_| {
                EngineError::Durability("engine command payload is not UTF-8".to_string())
            })?;
            return parse_command(text).map(Some).map_err(|error| {
                EngineError::Durability(format!("engine command parse failed: {error}"))
            });
        }
        if payload.len() < 10
            || !matches!(
                payload[1],
                ENGINE_TYPED_COMMAND_VERSION_LEGACY | ENGINE_TYPED_COMMAND_VERSION_CURRENT
            )
        {
            return Err(EngineError::Durability(
                "typed engine command has an unsupported or truncated header".to_string(),
            ));
        }
        let len = u64::from_le_bytes(payload[2..10].try_into().expect("checked typed header"));
        let body = &payload[10..];
        if len != body.len() as u64 {
            return Err(EngineError::Durability(
                "typed engine command length does not match its body".to_string(),
            ));
        }
        serde_json::from_slice(body).map(Some).map_err(|error| {
            EngineError::Durability(format!("typed engine command decode failed: {error}"))
        })
    }

    pub(crate) fn engine_command_uses_current_index_semantics(payload: &[u8]) -> bool {
        payload.first() != Some(&ENGINE_TYPED_COMMAND_TAG)
            || payload.get(1) == Some(&ENGINE_TYPED_COMMAND_VERSION_CURRENT)
    }

    fn encode_replay_typed_command(
        command: &Command,
        version: u8,
    ) -> Result<Arc<[u8]>, EngineError> {
        let canonical = serde_json::to_vec(command).map_err(|error| {
            EngineError::Durability(format!("typed replay command encode failed: {error}"))
        })?;
        let mut replay = Vec::with_capacity(10 + canonical.len());
        replay.push(ENGINE_TYPED_COMMAND_TAG);
        replay.push(version);
        replay.extend_from_slice(&(canonical.len() as u64).to_le_bytes());
        replay.extend_from_slice(&canonical);
        Ok(Arc::from(replay))
    }

    #[cfg(test)]
    pub(crate) fn encode_legacy_replay_command_for_test(
        command: &Command,
    ) -> Result<Arc<[u8]>, EngineError> {
        Self::encode_replay_typed_command(command, ENGINE_TYPED_COMMAND_VERSION_LEGACY)
    }

    #[cfg(test)]
    pub(crate) fn canonical_legacy_wal_record_for_test(
        commit: &mut CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
    ) -> Result<WalRecord, EngineError> {
        if payload.get(..2)
            != Some(&[
                ENGINE_TYPED_COMMAND_TAG,
                ENGINE_TYPED_COMMAND_VERSION_LEGACY,
            ])
        {
            return Err(EngineError::Durability(
                "historical canonical test record requires a legacy typed command".to_string(),
            ));
        }
        let command = Self::decode_engine_command(payload)?.ok_or_else(|| {
            EngineError::Durability(
                "historical canonical test record has no typed command".to_string(),
            )
        })?;
        let canonical = serde_json::to_vec(&command).map_err(|error| {
            EngineError::Durability(format!(
                "historical canonical test command encode failed: {error}"
            ))
        })?;
        let current = Self::canonical_wal_record(commit, txn_id, commit_seq, lane_id, payload)?;
        let envelope =
            gpu_db_wal::decode_canonical_record_payload(&current.as_wal_record().payload)?
                .ok_or_else(|| {
                    EngineError::Durability(
                        "historical canonical test record did not produce an envelope".to_string(),
                    )
                })?;

        let mut operation = Vec::with_capacity(ENGINE_OPERATION_MAGIC.len() + 12 + canonical.len());
        operation.extend_from_slice(ENGINE_OPERATION_MAGIC);
        operation.push(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V1);
        operation.extend_from_slice(&[0; 3]);
        operation.extend_from_slice(&(canonical.len() as u64).to_le_bytes());
        operation.extend_from_slice(&canonical);

        let mut fragments = envelope.fragments;
        fragments[0].body = operation;
        let mut header = envelope.header;
        header.catalog_after_digest = Self::canonical_catalog_transition(
            header.catalog_before_digest,
            fragments[0].kind,
            &fragments[0].body,
        );
        let mut outcome = envelope.outcome;
        outcome.target_digest = gpu_db_wal::canonical_request_digest(&fragments[0].body);
        let encoded = gpu_db_wal::encode_canonical_envelope(
            envelope.physical,
            &header,
            &fragments,
            &outcome,
        )?;
        Ok(encoded.into_prepared_record(txn_id)?.into_wal_record())
    }

    fn encode_transaction_claim_status(
        identity: gpu_db_wal::CanonicalIdentity,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Vec<u8> {
        let mut body = Vec::with_capacity(canonical_transaction_claim_status_len());
        body.extend_from_slice(TRANSACTION_STATUS_MAGIC);
        body.extend_from_slice(&identity.database_id);
        body.extend_from_slice(&identity.timeline_id);
        body.extend_from_slice(&txn_id.to_le_bytes());
        body.extend_from_slice(&request_digest);
        body.push(1); // durable claim pending publication; the terminal marker carries outcome
        body.extend_from_slice(&[0; 7]);
        body.extend_from_slice(&u64::MAX.to_le_bytes()); // retained/unexpired in the current policy
        debug_assert_eq!(body.len(), canonical_transaction_claim_status_len());
        body
    }

    fn validate_transaction_claim_status(
        body: &[u8],
        identity: gpu_db_wal::CanonicalIdentity,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<(), EngineError> {
        if body.len() != canonical_transaction_claim_status_len()
            || &body[..12] != TRANSACTION_STATUS_MAGIC
        {
            return Err(EngineError::Durability(
                "canonical WAL transaction-status fragment is malformed".to_string(),
            ));
        }
        if body[12..28] != identity.database_id
            || body[28..44] != identity.timeline_id
            || u64::from_le_bytes(body[44..52].try_into().unwrap()) != txn_id
            || body[52..84] != request_digest
            || body[84] != 1
            || body[85..92] != [0; 7]
            || u64::from_le_bytes(body[92..100].try_into().unwrap()) != u64::MAX
        {
            return Err(EngineError::Durability(
                "canonical WAL transaction-status claim does not match its envelope".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn decode_engine_operation(body: &[u8]) -> Result<Arc<[u8]>, EngineError> {
        let prefix_len = ENGINE_OPERATION_MAGIC.len() + 12;
        if body.len() < prefix_len
            || &body[..ENGINE_OPERATION_MAGIC.len()] != ENGINE_OPERATION_MAGIC
        {
            return Err(EngineError::Durability(
                "canonical WAL fragment has an invalid engine operation header".to_string(),
            ));
        }
        let codec = body[ENGINE_OPERATION_MAGIC.len()];
        if !matches!(
            codec,
            ENGINE_OPERATION_CODEC_LEGACY_SQL
                | ENGINE_OPERATION_CODEC_RESOLVED_BINARY
                | ENGINE_OPERATION_CODEC_TYPED_COMMAND_V1
                | ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2
        ) || body[ENGINE_OPERATION_MAGIC.len() + 1..ENGINE_OPERATION_MAGIC.len() + 4] != [0; 3]
        {
            return Err(EngineError::Durability(
                "canonical WAL fragment has an unsupported engine operation codec".to_string(),
            ));
        }
        let len_offset = ENGINE_OPERATION_MAGIC.len() + 4;
        let len = u64::from_le_bytes(body[len_offset..len_offset + 8].try_into().unwrap());
        let payload = &body[prefix_len..];
        if len != payload.len() as u64 {
            return Err(EngineError::Durability(
                "canonical WAL fragment engine operation payload is inconsistent".to_string(),
            ));
        }
        match codec {
            ENGINE_OPERATION_CODEC_RESOLVED_BINARY => {
                if payload.first() != Some(&WAL_BINARY_TAG) {
                    return Err(EngineError::Durability(
                        "canonical resolved-binary operation has no binary tag".to_string(),
                    ));
                }
                decode_binary_record(payload)?;
                Ok(Arc::from(payload))
            }
            ENGINE_OPERATION_CODEC_LEGACY_SQL => {
                let text = std::str::from_utf8(payload).map_err(|_| {
                    EngineError::Durability(
                        "legacy canonical SQL operation is not UTF-8".to_string(),
                    )
                })?;
                let command = parse_command(text).map_err(|error| {
                    EngineError::Durability(format!(
                        "legacy canonical SQL operation is invalid: {error}"
                    ))
                })?;
                Self::encode_replay_typed_command(&command, ENGINE_TYPED_COMMAND_VERSION_LEGACY)
            }
            ENGINE_OPERATION_CODEC_TYPED_COMMAND_V1 | ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2 => {
                let command: Command = serde_json::from_slice(payload).map_err(|error| {
                    EngineError::Durability(format!(
                        "canonical typed command decode failed: {error}"
                    ))
                })?;
                let canonical = serde_json::to_vec(&command).map_err(|error| {
                    EngineError::Durability(format!(
                        "canonical typed command re-encode failed: {error}"
                    ))
                })?;
                if canonical != payload {
                    return Err(EngineError::Durability(
                        "canonical typed command uses a non-canonical serialization".to_string(),
                    ));
                }
                Self::encode_replay_typed_command(
                    &command,
                    if codec == ENGINE_OPERATION_CODEC_TYPED_COMMAND_V1 {
                        ENGINE_TYPED_COMMAND_VERSION_LEGACY
                    } else {
                        ENGINE_TYPED_COMMAND_VERSION_CURRENT
                    },
                )
            }
            _ => unreachable!("codec checked above"),
        }
    }

    /// Validate every canonical authority before replaying any mutation. Legacy records are
    /// accepted only as an upgrade prefix; the first canonical record binds database lineage.
    fn prepare_durable_records_for_replay(
        &self,
        records: &[WalRecord],
    ) -> Result<Vec<PreparedDurableReplayRecord>, EngineError> {
        let mut lineage = None;
        let mut commit = self.commit_state();
        let mut expected_commit_seq = commit.repl.peek_next_index();
        let mut expected_catalog = commit.wal.canonical_catalog_tail()?.map(|tail| {
            (
                tail.identity,
                tail.catalog_after_epoch,
                tail.catalog_after_digest,
            )
        });
        // A prior replay chunk installs every canonical terminal claim before the next chunk is
        // admitted. This makes the one-way migration barrier span checkpoint/serial/lane chunks,
        // without treating a lineage-only identity anchor as evidence that canonical WAL exists.
        let mut canonical_seen = commit.canonical_replay_seen;
        let mut write_authority = commit.write_authority.clone();
        let mut simulated_allocator_high_water = self.read_state.mvcc.current_row_id();
        let mut transaction_claims: HashMap<_, _> = commit
            .transaction_status
            .iter()
            .map(|(txn_id, status)| (*txn_id, status.request_digest))
            .collect();
        drop(commit);
        let mut replay = Vec::with_capacity(records.len());
        for record in records {
            let Some(envelope) = gpu_db_wal::decode_canonical_record_payload(&record.payload)?
            else {
                if canonical_seen {
                    return Err(EngineError::Durability(format!(
                        "legacy WAL record {} follows the canonical migration boundary",
                        record.txn_id
                    )));
                }
                let payload = if record.payload.first() == Some(&WAL_BINARY_TAG) {
                    // Binary opcodes carry their own durable catalog epoch.
                    decode_binary_record(&record.payload)?;
                    validate_sequence_envelope_transaction_id(&record.payload, record.txn_id)?;
                    Arc::clone(&record.payload)
                } else if record.payload.first() == Some(&ENGINE_TYPED_COMMAND_TAG) {
                    let command =
                        Self::decode_engine_command(&record.payload)?.ok_or_else(|| {
                            EngineError::Durability(
                                "typed WAL record has no engine command".to_string(),
                            )
                        })?;
                    let canonical = Self::encode_replay_typed_command(
                        &command,
                        if Self::engine_command_uses_current_index_semantics(&record.payload) {
                            ENGINE_TYPED_COMMAND_VERSION_CURRENT
                        } else {
                            ENGINE_TYPED_COMMAND_VERSION_LEGACY
                        },
                    )?;
                    if canonical.as_ref() != record.payload.as_ref() {
                        return Err(EngineError::Durability(
                            "typed WAL record uses a non-canonical serialization".to_string(),
                        ));
                    }
                    canonical
                } else {
                    let command =
                        Self::decode_engine_command(&record.payload)?.ok_or_else(|| {
                            EngineError::Durability(
                                "legacy WAL record has no typed engine command".to_string(),
                            )
                        })?;
                    Self::encode_replay_typed_command(
                        &command,
                        ENGINE_TYPED_COMMAND_VERSION_LEGACY,
                    )?
                };
                replay.push(PreparedDurableReplayRecord::EngineOperation(WalRecord {
                    txn_id: record.txn_id,
                    payload,
                }));
                expected_commit_seq = expected_commit_seq.checked_add(1).ok_or_else(|| {
                    EngineError::Durability(
                        "WAL commit sequence overflow during recovery".to_string(),
                    )
                })?;
                continue;
            };
            canonical_seen = true;
            if envelope.header.stable_transaction_id != record.txn_id {
                return Err(EngineError::Durability(format!(
                    "canonical WAL stable transaction {} does not match outer record {}",
                    envelope.header.stable_transaction_id, record.txn_id
                )));
            }
            match lineage {
                None => lineage = Some(envelope.header.identity),
                Some(expected) if expected == envelope.header.identity => {}
                Some(_) => {
                    return Err(EngineError::Durability(
                        "canonical WAL database/cluster/timeline lineage changed mid-log"
                            .to_string(),
                    ));
                }
            }
            if envelope.header.commit_seq != expected_commit_seq {
                return Err(EngineError::Durability(format!(
                    "canonical WAL commit sequence gap: expected {expected_commit_seq}, found {}",
                    envelope.header.commit_seq
                )));
            }
            expected_commit_seq = expected_commit_seq.checked_add(1).ok_or_else(|| {
                EngineError::Durability(
                    "canonical WAL commit sequence overflow during recovery".to_string(),
                )
            })?;
            if let Some(authority) =
                crate::engine_write_authority::decode_write_authority_envelope(&envelope)?
            {
                let (catalog_identity, catalog_before_epoch, catalog_before_digest) =
                    expected_catalog.unwrap_or_else(|| {
                        (
                            envelope.header.identity,
                            0,
                            Self::canonical_genesis_catalog_digest(envelope.header.identity),
                        )
                    });
                if catalog_identity != envelope.header.identity
                    || envelope.header.catalog_before_epoch != catalog_before_epoch
                    || envelope.header.catalog_before_digest != catalog_before_digest
                    || envelope.header.catalog_after_epoch != catalog_before_epoch
                    || envelope.header.catalog_after_digest != catalog_before_digest
                {
                    return Err(EngineError::Durability(format!(
                        "canonical write-authority catalog boundary mismatch for transaction {}",
                        record.txn_id
                    )));
                }
                if let Some(high_water) = write_authority.apply(
                    envelope.header.commit_seq,
                    authority,
                    simulated_allocator_high_water,
                )? {
                    simulated_allocator_high_water = high_water;
                }
                replay.push(PreparedDurableReplayRecord::WriteAuthority);
                continue;
            }
            if Self::canonical_envelope_is_codec5(&envelope) {
                let fragments = envelope
                    .fragments
                    .iter()
                    .map(|fragment| gpu_db_wal::CanonicalFragmentRef {
                        kind: fragment.kind,
                        body: &fragment.body,
                    })
                    .collect::<Vec<_>>();
                let artifact = crate::typed_insert_aggregate::decode_closed_semantics_v2_replay(
                    &envelope.header,
                    &envelope.outcome,
                    &fragments,
                )?
                .ok_or_else(|| {
                    EngineError::Durability(
                        "codec-5 canonical WAL does not select supported semantics-v2 replay"
                            .to_string(),
                    )
                })?;
                let metadata = artifact.metadata();
                let table_count = u32::try_from(artifact.table_count()).map_err(|_| {
                    EngineError::Durability("codec-5 replay table count exceeds u32".to_string())
                })?;
                let (catalog_identity, catalog_before_epoch, catalog_before_digest) =
                    expected_catalog.unwrap_or_else(|| {
                        (
                            envelope.header.identity,
                            0,
                            Self::canonical_genesis_catalog_digest(envelope.header.identity),
                        )
                    });
                let composition_changes_catalog = artifact
                    .catalog_composition()
                    .is_some_and(|composition| !composition.catalog_commands.is_empty());
                let catalog_after_epoch = if composition_changes_catalog {
                    catalog_before_epoch.checked_add(1).ok_or_else(|| {
                        EngineError::Durability(
                            "codec-5 catalog epoch overflow during recovery".to_string(),
                        )
                    })?
                } else {
                    catalog_before_epoch
                };
                let catalog_after_digest = if composition_changes_catalog {
                    envelope.header.catalog_after_digest
                } else {
                    catalog_before_digest
                };
                if metadata.canonical_identity != envelope.header.identity
                    || metadata.leader_epoch != envelope.header.leader_epoch
                    || metadata.stable_transaction_id != record.txn_id
                    || metadata.request_digest != envelope.header.request_digest
                    || metadata.commit_sequence != envelope.header.commit_seq
                    || metadata.catalog_epoch != catalog_before_epoch
                    || metadata.catalog_digest != catalog_before_digest
                    || catalog_identity != envelope.header.identity
                    || envelope.header.catalog_before_epoch != catalog_before_epoch
                    || envelope.header.catalog_before_digest != catalog_before_digest
                    || envelope.header.catalog_after_epoch != catalog_after_epoch
                    || envelope.header.catalog_after_digest != catalog_after_digest
                    || envelope.header.table_block_count != table_count
                    || envelope.header.allocator_high_water != 0
                    || envelope.outcome.affected_rows != metadata.affected_rows
                {
                    return Err(EngineError::Durability(format!(
                        "codec-5 canonical WAL catalog or generation identity mismatch for transaction {}",
                        record.txn_id
                    )));
                }
                if transaction_claims
                    .insert(record.txn_id, envelope.header.request_digest)
                    .is_some()
                {
                    return Err(EngineError::Durability(format!(
                        "canonical WAL repeats terminal transaction claim {}",
                        record.txn_id
                    )));
                }
                let protocol = if envelope.header.flags
                    & crate::typed_insert_aggregate::OUTER_FLAG_FIRST_TYPED_INSERT_WRITER_EPOCH
                    != 0
                {
                    if !metadata.autocommit || metadata.statement_count != 1 {
                        return Err(EngineError::Durability(format!(
                            "historical codec-5 writer record {} cannot claim explicit or composed transaction mode",
                            record.txn_id
                        )));
                    }
                    write_authority.validate_historical_codec5_parent(
                        record.txn_id,
                        metadata.request_digest,
                        metadata.commit_sequence,
                        metadata.typed_statement_digest,
                        metadata.row_allocator_before,
                        metadata.row_allocator_high_water,
                        metadata.affected_rows,
                    )?;
                    SemanticsV2ReplayProtocol::HistoricalWriteAuthority
                } else {
                    simulated_allocator_high_water = validate_generic_codec5_replay_preflight(
                        &write_authority,
                        record.txn_id,
                        metadata.row_allocator_before,
                        metadata.row_allocator_high_water,
                        metadata.affected_rows,
                        simulated_allocator_high_water,
                    )?;
                    SemanticsV2ReplayProtocol::GenericOneRecord
                };
                expected_catalog = Some((
                    envelope.header.identity,
                    catalog_after_epoch,
                    catalog_after_digest,
                ));
                replay.push(PreparedDurableReplayRecord::SemanticsV2TypedInsert {
                    artifact,
                    protocol,
                });
                continue;
            }
            if transaction_claims
                .insert(record.txn_id, envelope.header.request_digest)
                .is_some()
            {
                return Err(EngineError::Durability(format!(
                    "canonical WAL repeats terminal transaction claim {}",
                    record.txn_id
                )));
            }
            if !matches!(
                envelope.outcome.kind,
                gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
                    | gpu_db_wal::CanonicalOutcomeKind::CommitNoOp
            ) {
                return Err(EngineError::Durability(format!(
                    "canonical WAL contains abort outcome for replay slot {}",
                    envelope.header.commit_seq
                )));
            }
            if envelope.fragments.len() != 2 {
                return Err(EngineError::Durability(format!(
                    "engine canonical WAL record contains {} fragments; one operation and one status claim are required",
                    envelope.fragments.len()
                )));
            }
            let operation = &envelope.fragments[0];
            let status = &envelope.fragments[1];
            if status.kind != gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus {
                return Err(EngineError::Durability(
                    "canonical WAL terminal operation has no ordered transaction-status claim"
                        .to_string(),
                ));
            }
            Self::validate_transaction_claim_status(
                &status.body,
                envelope.header.identity,
                record.txn_id,
                envelope.header.request_digest,
            )?;
            let payload = Self::decode_engine_operation(&operation.body)?;
            validate_sequence_envelope_transaction_id(&payload, record.txn_id)?;
            let operation_kind = Self::canonical_fragment_kind(&payload)?;
            let referenced_allocator_high_water = Self::canonical_allocator_high_water(&payload)?;
            if gpu_db_wal::canonical_request_digest(&operation.body)
                != envelope.outcome.target_digest
                || operation_kind != operation.kind
            {
                return Err(EngineError::Durability(format!(
                    "canonical WAL request/fragment digest mismatch for transaction {}",
                    record.txn_id
                )));
            }
            let (catalog_identity, catalog_before_epoch, catalog_before_digest) = expected_catalog
                .unwrap_or_else(|| {
                    (
                        envelope.header.identity,
                        0,
                        Self::canonical_genesis_catalog_digest(envelope.header.identity),
                    )
                });
            if catalog_identity != envelope.header.identity
                || envelope.header.catalog_before_epoch != catalog_before_epoch
                || envelope.header.catalog_before_digest != catalog_before_digest
            {
                return Err(EngineError::Durability(format!(
                    "canonical WAL catalog-before boundary mismatch for transaction {}",
                    record.txn_id
                )));
            }
            let catalog_after_epoch =
                if operation_kind == gpu_db_wal::CanonicalFragmentKind::CatalogMutation {
                    catalog_before_epoch.checked_add(1).ok_or_else(|| {
                        EngineError::Durability(
                            "canonical catalog epoch overflow during recovery".to_string(),
                        )
                    })?
                } else {
                    catalog_before_epoch
                };
            let catalog_after_digest = Self::canonical_catalog_transition(
                catalog_before_digest,
                operation_kind,
                &operation.body,
            );
            let table_block_count = Self::canonical_table_block_count(&payload, operation_kind)?;
            if envelope.header.catalog_after_epoch != catalog_after_epoch
                || envelope.header.catalog_after_digest != catalog_after_digest
                || envelope.header.flags != u32::from(operation_kind as u16)
                || envelope.header.table_block_count != table_block_count
                || envelope.header.allocator_high_water < referenced_allocator_high_water
            {
                return Err(EngineError::Durability(format!(
                    "canonical WAL catalog-after boundary or operation metadata mismatch for transaction {}",
                    record.txn_id
                )));
            }
            expected_catalog = Some((
                envelope.header.identity,
                catalog_after_epoch,
                catalog_after_digest,
            ));
            replay.push(PreparedDurableReplayRecord::EngineOperation(WalRecord {
                txn_id: record.txn_id,
                payload,
            }));
        }
        if let Some(identity) = lineage {
            let mut commit = self.commit_state();
            if commit.canonical_lineage_bound && commit.canonical_identity != identity {
                return Err(EngineError::Durability(
                    "canonical WAL recovery source changed database lineage".to_string(),
                ));
            }
            commit.canonical_identity = identity;
            commit.canonical_lineage_bound = true;
            commit.canonical_replay_seen = true;
        }
        Ok(replay)
    }

    pub(crate) fn replay_durable_records(&self, records: &[WalRecord]) -> Result<(), EngineError> {
        // `bind_durable_identity_for_recovery` supplies the complete multi-lane/file source.
        // In-memory and test recovery enter here directly with their complete source.
        self.prepare_legacy_index_oid_recovery(records)?;
        let mut claims = Vec::new();
        let mut outcomes = Vec::with_capacity(records.len());
        for record in records {
            if let Some(envelope) = gpu_db_wal::decode_canonical_record_payload(&record.payload)? {
                if crate::engine_write_authority::decode_write_authority_envelope(&envelope)?
                    .is_some()
                {
                    outcomes.push(None);
                    continue;
                }
                if Self::canonical_envelope_is_codec5(&envelope) {
                    claims.push((
                        record.txn_id,
                        DurableTransactionStatus {
                            request_digest: envelope.header.request_digest,
                            outcome: DurableTransactionOutcome::Committed {
                                commit_seq: envelope.header.commit_seq,
                                affected_rows: envelope.outcome.affected_rows,
                            },
                        },
                    ));
                    // Preparation below owns the strict retained decode.  This pre-scan merely
                    // records the outer status without routing the aggregate through GPUDBOP1.
                    outcomes.push(Some((
                        envelope.header.commit_seq,
                        envelope.outcome,
                        false,
                        envelope.header.allocator_high_water,
                    )));
                    continue;
                }
                claims.push((
                    record.txn_id,
                    DurableTransactionStatus {
                        request_digest: envelope.header.request_digest,
                        outcome: DurableTransactionOutcome::Committed {
                            commit_seq: envelope.header.commit_seq,
                            affected_rows: envelope.outcome.affected_rows,
                        },
                    },
                ));
                let operation = Self::decode_engine_operation(&envelope.fragments[0].body)?;
                let marker_kind_from_rows = if operation.first() == Some(&WAL_BINARY_TAG) {
                    matches!(
                        decode_binary_record(&operation)?,
                        crate::wal_binary::BinaryWalRecord::Insert(_)
                            | crate::wal_binary::BinaryWalRecord::DeleteByKey(_)
                            | crate::wal_binary::BinaryWalRecord::UpdateByKey(_)
                    )
                } else {
                    false
                };
                outcomes.push(Some((
                    envelope.header.commit_seq,
                    envelope.outcome,
                    marker_kind_from_rows,
                    envelope.header.allocator_high_water,
                )));
            } else {
                outcomes.push(None);
            }
        }
        let replay = self.prepare_durable_records_for_replay(records)?;
        for ((durable_record, record), authority) in records.iter().zip(replay).zip(outcomes) {
            match record {
                PreparedDurableReplayRecord::EngineOperation(record) => {
                    self.replay_validated_durable_record(durable_record, record)?
                }
                PreparedDurableReplayRecord::WriteAuthority => {
                    self.replay_validated_write_authority_control(durable_record)?
                }
                PreparedDurableReplayRecord::SemanticsV2TypedInsert { artifact, protocol } => self
                    .replay_validated_semantics_v2_typed_insert(
                        durable_record,
                        artifact,
                        protocol,
                    )?,
            }
            let Some((commit_seq, expected, marker_kind_from_rows, allocator_high_water)) =
                authority
            else {
                continue;
            };
            let observed = self.commit_state().last_applied_outcome;
            if observed != Some((commit_seq, expected.affected_rows)) {
                return Err(EngineError::Durability(format!(
                    "canonical WAL outcome mismatch at commit {commit_seq}: marker affected {} row(s), replay produced {:?}",
                    expected.affected_rows,
                    observed.map(|(_, rows)| rows)
                )));
            }
            if marker_kind_from_rows {
                let observed_kind = if expected.affected_rows == 0 {
                    gpu_db_wal::CanonicalOutcomeKind::CommitNoOp
                } else {
                    gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
                };
                if expected.kind != observed_kind {
                    return Err(EngineError::Durability(format!(
                        "canonical WAL terminal kind mismatch at commit {commit_seq}: marker {:?}, replay {:?}",
                        expected.kind, observed_kind
                    )));
                }
            }
            self.read_state
                .mvcc
                .advance_row_id_to_at_least(allocator_high_water);
        }
        let mut commit = self.commit_state();
        for (txn_id, status) in claims {
            match commit.transaction_status.entry(txn_id) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(status);
                    commit.invalidate_transaction_status_reservations();
                }
                std::collections::hash_map::Entry::Occupied(_) => {
                    return Err(EngineError::Durability(format!(
                        "canonical WAL repeats terminal transaction claim {txn_id} across replay chunks"
                    )));
                }
            }
        }
        let next_txn_id = commit
            .transaction_status
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.transaction_id_allocator
            .fetch_max(next_txn_id, AtomicOrdering::Relaxed);
        Ok(())
    }

    pub(crate) fn install_reconciled_transaction_statuses(
        &self,
        base: &std::path::Path,
    ) -> Result<Vec<gpu_db_wal::ReconciledTransactionStatus>, EngineError> {
        let statuses = gpu_db_wal::read_reconciled_transaction_statuses(base)?;
        if statuses.is_empty() {
            return Ok(statuses);
        }
        let mut commit = self.commit_state();
        for status in &statuses {
            if commit.canonical_lineage_bound && commit.canonical_identity != status.identity {
                return Err(EngineError::Durability(format!(
                    "reconciled transaction {} belongs to another database/timeline",
                    status.txn_id
                )));
            }
            let recovered = DurableTransactionStatus {
                request_digest: status.request_digest,
                outcome: DurableTransactionOutcome::AbortedDiscardedOrphan,
            };
            match commit.transaction_status.get(&status.txn_id).copied() {
                Some(existing) if existing != recovered => {
                    return Err(EngineError::Durability(format!(
                        "reconciled transaction {} conflicts with terminal WAL status",
                        status.txn_id
                    )));
                }
                Some(_) => {}
                None => {
                    commit.transaction_status.insert(status.txn_id, recovered);
                    commit.invalidate_transaction_status_reservations();
                }
            }
        }
        Ok(statuses)
    }

    /// Replay one already-validated authority record without synthesizing a replacement WAL
    /// envelope. This is essential for typed command bodies: the replicator applies the decoded
    /// AST payload, while the in-memory continuation WAL retains the byte-identical physical and
    /// logical authority that recovery verified. Repeated recovery therefore never rewrites
    /// lineage, request digests, outcome markers, or physical-range mappings.
    fn replay_validated_durable_record(
        &self,
        durable_record: &WalRecord,
        apply_record: WalRecord,
    ) -> Result<(), EngineError> {
        let mut commit = self.commit_state();
        let expected = commit.repl.peek_next_index();
        let token = commit.repl.propose(apply_record.payload)?;
        if token.index != expected {
            return Err(EngineError::Durability(format!(
                "recovery sequencer assigned {}, expected {expected}",
                token.index
            )));
        }
        commit.wal.append(durable_record.clone());
        commit.wal.flush_all()?;
        commit
            .repl
            .wait_committed(token, Duration::from_millis(0))?;
        let timestamp_micros =
            current_timestamp_micros().max(commit.max_commit_timestamp_micros.saturating_add(1));
        commit.record_commit_timestamp(durable_record.txn_id, timestamp_micros);
        self.apply_and_publish_committed(&mut commit, durable_record.txn_id, token.index)?;
        self.metrics.inc_commit();
        Ok(())
    }

    fn replay_validated_write_authority_control(
        &self,
        durable_record: &WalRecord,
    ) -> Result<(), EngineError> {
        let mut commit = self.commit_state();
        let expected = commit.repl.peek_next_index();
        let token = commit.repl.propose(Arc::clone(&durable_record.payload))?;
        if token.index != expected {
            return Err(EngineError::Durability(format!(
                "write-authority recovery sequencer assigned {}, expected {expected}",
                token.index
            )));
        }
        commit.wal.append(durable_record.clone());
        commit.wal.flush_all()?;
        commit
            .repl
            .wait_committed(token, Duration::from_millis(0))?;
        self.apply_and_publish_write_authority_control(
            &mut commit,
            durable_record.txn_id,
            token.index,
        )?;
        Ok(())
    }

    /// Replay a closed plural codec-5 transaction through the same per-table generic GPU
    /// generator and shared physical reservation used by live commit, while retaining one WAL,
    /// allocator, apply, and root-publication boundary for the transaction.
    fn replay_validated_plural_semantics_v2_typed_insert(
        &self,
        durable_record: &WalRecord,
        artifact: crate::typed_insert_aggregate::SemanticsV2ReplayArtifact,
        protocol: SemanticsV2ReplayProtocol,
    ) -> Result<(), EngineError> {
        if protocol != SemanticsV2ReplayProtocol::GenericOneRecord {
            return Err(EngineError::Durability(
                "historical codec-5 replay cannot claim a plural transaction".to_string(),
            ));
        }
        let transaction_metadata = artifact.metadata();
        let row_count = u32::try_from(transaction_metadata.affected_rows).map_err(|_| {
            EngineError::Durability("plural codec-5 recovery rows exceed u32".to_string())
        })?;
        let proposed = crate::wal_binary::ProposedRowIdRange::new(
            transaction_metadata.row_allocator_before,
            row_count,
        )?;
        if proposed.allocator_high_water() != transaction_metadata.row_allocator_high_water
            || self.read_state.mvcc.current_row_id() != transaction_metadata.row_allocator_before
        {
            return Err(EngineError::Durability(
                "plural codec-5 recovery allocator predecessor or range drifted".to_string(),
            ));
        }

        struct PreparedPluralReplayTable {
            table: crate::RelationalTable,
            metadata: crate::typed_insert_aggregate::SemanticsV2ReplayMetadata,
            source: Option<crate::typed_insert_batch::PreparedResidentAppendSource>,
            generation: Option<crate::engine_transaction_delta::TypedInsertRuntimeGenerationOutput>,
            indexed: bool,
        }

        let catalog = self.catalog_snapshot();
        let original_roots = self.read_state.typed_generation_roots.load_full();
        let mut evolving_roots = Arc::clone(&original_roots);
        let mut root_publication: Option<crate::engine_commit::LiveTypedGenerationRootPublication> =
            None;
        let (table_artifacts, private_sequence_publications, catalog_composition) =
            artifact.into_parts();
        let mut projected_catalog = self.ddl_catalog().clone();
        if let Some(composition) = catalog_composition
            .as_ref()
            .filter(|composition| !composition.catalog_commands.is_empty())
        {
            self.apply_codec5_catalog_composition(
                &mut projected_catalog,
                transaction_metadata.commit_sequence,
                composition,
            )?;
        }
        let mut prior_sequence_oid = None;
        for publication in private_sequence_publications.iter() {
            if prior_sequence_oid.is_some_and(|prior| prior >= publication.sequence_oid) {
                return Err(EngineError::Durability(
                    "plural codec-5 recovery private sequence publications are not in unique stable-OID order"
                        .to_string(),
                ));
            }
            self.apply_codec5_private_sequence_name_binding(&mut projected_catalog, publication)?;
            prior_sequence_oid = Some(publication.sequence_oid);
        }
        let table_count = table_artifacts.len();
        let mut prepared_tables = Vec::with_capacity(table_count);
        for (table_ordinal, table_artifact) in table_artifacts.into_vec().into_iter().enumerate() {
            let table_ref = u32::try_from(table_ordinal).map_err(|_| {
                EngineError::Durability("plural codec-5 table ordinal exceeds u32".to_string())
            })?;
            if table_artifact.table_ref() != table_ref {
                return Err(EngineError::Durability(
                    "plural codec-5 recovery table order or index breadth drifted".to_string(),
                ));
            }
            let metadata = table_artifact.metadata();
            let replay_indexes = table_artifact.indexes().to_vec();
            let final_table = projected_catalog
                .relational_catalog
                .get(table_artifact.target_table_name())
                .cloned()
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "plural codec-5 recovery postimage lacks relation \"{}\"",
                        table_artifact.target_table_name()
                    ))
                })?;
            // S3 establishes the transaction-private catalog postimage before the sole
            // codec-5 terminal publishes it.  A transaction-created table consequently has
            // no public relational or GPU predecessor; its authenticated postimage supplies
            // the physical shape for the same GPU CREATE generation used by live commit.
            let public_table = catalog
                .relational_catalog
                .get(table_artifact.target_table_name())
                .cloned();
            let table = match (metadata.initial_table_absent, public_table) {
                (true, None) => final_table.clone(),
                (true, Some(_)) => {
                    return Err(EngineError::Durability(
                        "plural codec-5 recovery CREATE target already has a public predecessor"
                            .to_string(),
                    ));
                }
                (false, Some(table)) => table,
                (false, None) => {
                    return Err(EngineError::Durability(format!(
                        "plural codec-5 recovery targets unknown relation \"{}\"",
                        table_artifact.target_table_name()
                    )));
                }
            };
            let replay_row_sources = table_artifact.row_sources().to_vec();
            let source = table_artifact.into_recovery_source(&table, &final_table)?;
            let table_rows = u32::try_from(metadata.affected_rows).map_err(|_| {
                EngineError::Durability("plural codec-5 table row count exceeds u32".to_string())
            })?;
            let surviving_rows = u32::try_from(replay_row_sources.len()).map_err(|_| {
                EngineError::Durability(
                    "plural codec-5 table survivor count exceeds u32".to_string(),
                )
            })?;
            let table_range = crate::wal_binary::ProposedRowIdRange::new(
                metadata.row_allocator_before,
                table_rows,
            )?;
            if table_range.allocator_high_water() != metadata.row_allocator_high_water {
                return Err(EngineError::Durability(
                    "plural codec-5 table allocator interval drifted".to_string(),
                ));
            }
            let predecessor = match (
                metadata.initial_table_absent,
                evolving_roots.table(table.stable_table_id),
            ) {
                (true, None) => crate::engine_state::TypedTableGenerationRoot {
                    data_generation: 0,
                    table_root: [0; 32],
                    logical_row_count: 0,
                },
                (true, Some(_)) => {
                    return Err(EngineError::Durability(
                        "plural codec-5 recovery CREATE target already has a GPU predecessor"
                            .to_string(),
                    ));
                }
                (false, Some(predecessor)) => predecessor,
                (false, None) => {
                    return Err(EngineError::Durability(
                        "plural codec-5 recovery has no GPU-authenticated INSERT predecessor"
                            .to_string(),
                    ));
                }
            };
            let predecessor_database_root = evolving_roots.database_root;
            if predecessor.data_generation != metadata.data_generation_before
                || predecessor.table_root != metadata.initial_table_root
                || predecessor.logical_row_count != metadata.initial_logical_row_count
                || (!metadata.initial_table_absent && predecessor_database_root.is_none())
            {
                return Err(EngineError::Durability(
                    "plural codec-5 recovery table predecessor differs from durable S7".to_string(),
                ));
            }
            if source.is_none() {
                if surviving_rows != 0 || !replay_indexes.is_empty() {
                    return Err(EngineError::Durability(
                        "neutral plural codec-5 table retained physical generation work"
                            .to_string(),
                    ));
                }
                prepared_tables.push(PreparedPluralReplayTable {
                    table,
                    metadata,
                    source: None,
                    generation: None,
                    indexed: false,
                });
                continue;
            }
            let source = source.expect("nonneutral replay table has a resident source");
            let catalog_index_count = table.indexes.len();
            if catalog_index_count != replay_indexes.len() {
                return Err(EngineError::Durability(
                    "plural codec-5 recovery index inventory differs from the current catalog"
                        .to_string(),
                ));
            }
            let mut runtime_indexes = Vec::with_capacity(replay_indexes.len());
            let mut predecessor_index_roots = Vec::with_capacity(replay_indexes.len());
            let mut successor_index_roots = Vec::with_capacity(replay_indexes.len());
            let mut key_start = 0_u32;
            let mut effect_start = 0_u32;
            for retained in &replay_indexes {
                let catalog_index = table
                    .indexes
                    .iter()
                    .find(|index| u64::from(index.oid) == retained.stable_index_id)
                    .ok_or_else(|| {
                        EngineError::Durability(format!(
                            "plural codec-5 recovery index {} is absent from the current catalog",
                            retained.stable_index_id
                        ))
                    })?;
                let predecessor_index = match (
                    metadata.initial_table_absent,
                    evolving_roots
                        .table_index_root(metadata.stable_table_id, retained.stable_index_id),
                ) {
                    (true, None) => crate::engine_state::TypedIndexGenerationRoot {
                        stable_index_id: retained.stable_index_id,
                        index_generation: 0,
                        index_root: [0; 32],
                    },
                    (true, Some(_)) => {
                        return Err(EngineError::Durability(
                            "plural codec-5 recovery CREATE index already has a GPU predecessor"
                                .to_string(),
                        ));
                    }
                    (false, Some(predecessor)) => predecessor,
                    (false, None) => {
                        return Err(EngineError::Durability(format!(
                            "plural codec-5 recovery has no GPU-authenticated predecessor for index {}",
                            retained.stable_index_id
                        )));
                    }
                };
                if catalog_index.name != retained.name.as_ref()
                    || predecessor_index.index_generation != retained.base_generation
                    || predecessor_index.index_root != retained.base_root
                    || retained.final_generation != metadata.data_generation_after
                    || retained.final_root == [0; 32]
                    || retained.final_root == retained.base_root
                {
                    return Err(EngineError::Durability(
                        "plural codec-5 recovery durable index roots differ from the published predecessor"
                            .to_string(),
                    ));
                }
                let key_columns = retained
                    .key_columns
                    .iter()
                    .map(
                        |key| gpu_db_execution::RuntimeTypedInsertGenerationIndexKeyColumn {
                            key_ordinal: key.key_ordinal,
                            catalog_column_ordinal: key.catalog_column_ordinal,
                            stable_column_id: key.stable_column_id,
                            attnum: key.attnum,
                            storage: key.storage,
                            declared_type_oid: key.declared_type_oid,
                            signed_type_size: key.signed_type_size,
                            column_name_digest: key.column_name_digest,
                        },
                    )
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let key_count = u32::try_from(key_columns.len()).map_err(|_| {
                    EngineError::Durability(
                        "plural codec-5 recovery index key count exceeds u32".to_string(),
                    )
                })?;
                runtime_indexes.push(
                    crate::engine_transaction_delta::TypedInsertRuntimeGenerationIndexInput {
                        descriptor: gpu_db_execution::RuntimeTypedInsertGenerationIndex {
                            stable_index_id: retained.stable_index_id,
                            raw_catalog_index_ordinal: retained.raw_catalog_ordinal,
                            index_flags: retained.flags,
                            null_equality_policy: retained.null_equality_policy,
                            base_generation: retained.base_generation,
                            base_root: retained.base_root,
                            key_start,
                            key_count,
                            effect_start,
                            effect_count: surviving_rows,
                        },
                        key_columns,
                    },
                );
                key_start = key_start.checked_add(key_count).ok_or_else(|| {
                    EngineError::Durability(
                        "plural codec-5 recovery index key range overflows".to_string(),
                    )
                })?;
                effect_start = effect_start.checked_add(surviving_rows).ok_or_else(|| {
                    EngineError::Durability(
                        "plural codec-5 recovery index effect range overflows".to_string(),
                    )
                })?;
                if !metadata.initial_table_absent {
                    predecessor_index_roots.push(predecessor_index);
                }
                successor_index_roots.push(crate::engine_state::TypedIndexGenerationRoot {
                    stable_index_id: retained.stable_index_id,
                    index_generation: retained.final_generation,
                    index_root: retained.final_root,
                });
            }
            let table_map_predecessor = match predecessor_database_root {
                Some(initial_database_root) => {
                    evolving_roots
                        .retained_table_map_predecessor(table.stable_table_id, initial_database_root)?
                        .ok_or_else(|| {
                            EngineError::Durability(
                                "plural codec-5 recovery has no retained table-map witness"
                                    .to_string(),
                            )
                        })?
                }
                None if metadata.initial_table_absent => {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase
                }
                None => {
                    return Err(EngineError::Durability(
                        "plural codec-5 recovery has no GPU-authenticated database predecessor"
                            .to_string(),
                    ));
                }
            };
            let generation = self.run_typed_insert_runtime_generation(
                crate::engine_transaction_delta::TypedInsertRuntimeGenerationInput {
                    source: &source,
                    row_allocator_before: metadata.row_allocator_before,
                    first_row_id: metadata.row_allocator_before,
                    row_sources: &replay_row_sources,
                    database_id: transaction_metadata.canonical_identity.database_id,
                    catalog_epoch: transaction_metadata.catalog_epoch,
                    catalog_digest: transaction_metadata.catalog_digest,
                    stable_transaction_id: transaction_metadata.stable_transaction_id,
                    commit_sequence: transaction_metadata.commit_sequence,
                    typed_statement_digest: [0; 32],
                    action: if metadata.initial_table_absent {
                        gpu_db_execution::RuntimeTypedInsertGenerationTableAction::CreateWithRowSet
                    } else if metadata.resets_existing_rows {
                        gpu_db_execution::RuntimeTypedInsertGenerationTableAction::ResetThenRowSetInsert
                    } else {
                        gpu_db_execution::RuntimeTypedInsertGenerationTableAction::RowSetInsert
                    },
                    table_map_predecessor,
                    stable_table_id: metadata.stable_table_id,
                    write001_final_image_ref: table_ref,
                    // The GPU derives the first CREATE generation at commit sequence while S7
                    // retains the real absent predecessor (zero generation/root).
                    base_data_generation: if metadata.initial_table_absent {
                        transaction_metadata.commit_sequence
                    } else {
                        predecessor.data_generation
                    },
                    base_table_root: predecessor.table_root,
                    row_allocator_high_water: metadata.row_allocator_high_water,
                    initial_logical_row_count: predecessor.logical_row_count,
                    final_logical_row_count: metadata.final_logical_row_count,
                    image_layout_digest: metadata.image_layout_digest,
                    image_content_digest: metadata.image_content_digest,
                    indexes: &runtime_indexes,
                },
            )?;
            if generation.initial_table_root != metadata.initial_table_root
                || generation.final_table_root != metadata.final_table_root
                || predecessor_database_root
                    .is_some_and(|root| generation.initial_database_root != root)
            {
                return Err(EngineError::Durability(
                    "plural codec-5 recovery CUDA commitments differ from durable S7 roots"
                        .to_string(),
                ));
            }
            let mut generated_index_roots = vec![
                gpu_db_execution::RuntimeTypedInsertGenerationIndexRoot {
                    stable_index_id: 0,
                    initial_generation: 0,
                    initial_root: [0; 32],
                    final_generation: 0,
                    final_root: [0; 32],
                };
                replay_indexes.len()
            ];
            generation
                .logical_completion
                .copy_index_generation_roots_into(&mut generated_index_roots)
                .map_err(|_| {
                    EngineError::Durability(
                        "plural codec-5 recovery GPU index-root cardinality drifted".to_string(),
                    )
                })?;
            if generated_index_roots
                .iter()
                .zip(&replay_indexes)
                .any(|(generated, retained)| {
                    generated.stable_index_id != retained.stable_index_id
                        || generated.initial_generation != retained.base_generation
                        || generated.initial_root != retained.base_root
                        || generated.final_generation != retained.final_generation
                        || generated.final_root != retained.final_root
                })
            {
                return Err(EngineError::Durability(
                    "plural codec-5 recovery CUDA index commitments differ from durable S7 roots"
                        .to_string(),
                ));
            }
            let mut shape_roots = vec![[0; 32]; table.columns.len()];
            let mut column_roots = vec![[0; 32]; table.columns.len()];
            generation
                .logical_completion
                .copy_column_roots_into(&mut shape_roots, &mut column_roots)
                .map_err(|_| {
                    EngineError::Durability(
                        "plural codec-5 recovery GPU column-root cardinality drifted".to_string(),
                    )
                })?;
            let successor_columns = table
                .columns
                .iter()
                .enumerate()
                .map(|(ordinal, column)| {
                    Ok(crate::engine_state::TypedColumnGenerationRoot {
                        catalog_column_ordinal: u32::try_from(ordinal).map_err(|_| {
                            EngineError::Durability(
                                "plural codec-5 column ordinal exceeds u32".to_string(),
                            )
                        })?,
                        stable_column_id: column.id,
                        attnum: column.attnum,
                        column_shape_root: shape_roots[ordinal],
                        column_root: column_roots[ordinal],
                    })
                })
                .collect::<Result<Vec<_>, EngineError>>()?;
            root_publication = Some(match root_publication {
                None => crate::engine_commit::LiveTypedGenerationRootPublication::
                    from_exact_gpu_completed_table_map_predecessor(
                        Arc::clone(&original_roots),
                        table.stable_table_id,
                        (!metadata.initial_table_absent).then_some(predecessor),
                        metadata.resets_existing_rows,
                        predecessor_database_root,
                        crate::engine_state::TypedTableGenerationRoot {
                            data_generation: metadata.data_generation_after,
                            table_root: metadata.final_table_root,
                            logical_row_count: metadata.final_logical_row_count,
                        },
                        &successor_columns,
                        &predecessor_index_roots,
                        &successor_index_roots,
                        generation.final_database_root,
                        &generation.table_map_completion,
                    )?,
                Some(publication) => publication
                    .then_exact_gpu_completed_table_map_predecessor(
                        table.stable_table_id,
                        (!metadata.initial_table_absent).then_some(predecessor),
                        metadata.resets_existing_rows,
                        crate::engine_state::TypedTableGenerationRoot {
                            data_generation: metadata.data_generation_after,
                            table_root: metadata.final_table_root,
                            logical_row_count: metadata.final_logical_row_count,
                        },
                        &successor_columns,
                        &predecessor_index_roots,
                        &successor_index_roots,
                        generation.final_database_root,
                        &generation.table_map_completion,
                    )?,
            });
            evolving_roots = root_publication
                .as_ref()
                .expect("plural recovery publication was just built")
                .private_candidate();
            prepared_tables.push(PreparedPluralReplayTable {
                table,
                metadata,
                source: Some(source),
                generation: Some(generation),
                indexed: !replay_indexes.is_empty(),
            });
        }

        if evolving_roots.database_root != Some(transaction_metadata.final_database_root) {
            return Err(EngineError::Durability(
                "plural codec-5 recovery table-map successor differs from durable S7".to_string(),
            ));
        }

        let mut compile_order = prepared_tables
            .iter()
            .enumerate()
            .filter_map(|(index, prepared)| prepared.source.is_some().then_some(index))
            .collect::<Vec<_>>();
        compile_order.sort_by_key(|index| {
            let prepared = &prepared_tables[*index];
            let requires_rollover = !prepared.metadata.initial_table_absent
                && self.transaction_terminal_typed_insert_requires_rollover(
                    prepared
                        .source
                        .as_ref()
                        .expect("plural recovery source remains before plan compilation"),
                    prepared.metadata.resets_existing_rows,
                );
            (!prepared.indexed || !requires_rollover, !prepared.indexed)
        });
        let indexed_table_count = prepared_tables
            .iter()
            .filter(|prepared| prepared.indexed)
            .count();
        let mut remaining_indexed_rollovers = compile_order
            .iter()
            .filter(|index| {
                let prepared = &prepared_tables[**index];
                !prepared.metadata.initial_table_absent
                    && prepared.indexed
                    && self.transaction_terminal_typed_insert_requires_rollover(
                        prepared
                            .source
                            .as_ref()
                            .expect("plural recovery source remains before plan compilation"),
                        prepared.metadata.resets_existing_rows,
                    )
            })
            .count();
        let mut named_index_lifecycle = (indexed_table_count != 0).then(|| {
            self.read_state
                .residency
                .begin_transaction_named_index_publication(
                    prepared_tables
                        .iter()
                        .filter(|prepared| prepared.source.is_some())
                        .map(|prepared| prepared.table.name.clone())
                        .collect(),
                )
        });
        let plan_count = compile_order.len();
        let mut plans = Vec::with_capacity(plan_count);
        let mut shared_gate = None;
        let mut shared_budget_guard = None;
        let mut manifest_predecessor = None;
        let mut prior_reserved_bytes = 0_u64;
        let mut remaining_indexed_tables = indexed_table_count;
        for (position, table_index) in compile_order.into_iter().enumerate() {
            let prepared = &mut prepared_tables[table_index];
            let current_is_indexed_rollover = !prepared.metadata.initial_table_absent
                && prepared.indexed
                && self.transaction_terminal_typed_insert_requires_rollover(
                    prepared
                        .source
                        .as_ref()
                        .expect("plural recovery source remains before plan compilation"),
                    prepared.metadata.resets_existing_rows,
                );
            if prepared.indexed {
                remaining_indexed_tables = remaining_indexed_tables.saturating_sub(1);
            }
            let source = prepared
                .source
                .take()
                .expect("plural recovery source is consumed by one device plan");
            let surviving_rows = u64::try_from(source.row_count()).map_err(|_| {
                EngineError::Durability(
                    "plural codec-5 recovery source row count exceeds u64".to_string(),
                )
            })?;
            let surviving_high_water = prepared
                .metadata
                .row_allocator_before
                .checked_add(surviving_rows)
                .ok_or_else(|| {
                    EngineError::Durability(
                        "plural codec-5 recovery survivor row-id range overflows".to_string(),
                    )
                })?;
            let ids = (prepared.metadata.row_allocator_before..surviving_high_water)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let mut plan = if prepared.metadata.initial_table_absent {
                let lifecycle = prepared.indexed.then(|| {
                    named_index_lifecycle
                        .take()
                        .expect("indexed transaction-created plural replay retains one lifecycle")
                });
                if let Some(gate) = shared_gate.take() {
                    self.compile_transaction_created_table_typed_insert_device_plan_with_gate(
                        &prepared.table,
                        source,
                        crate::engine_residency::DeviceInsertRowIds::exact(ids),
                        transaction_metadata.commit_sequence,
                        lifecycle,
                        gate,
                        shared_budget_guard.take(),
                        prior_reserved_bytes,
                    )
                } else {
                    if shared_budget_guard.is_some() || prior_reserved_bytes != 0 {
                        return Err(EngineError::Durability(
                            "plural codec-5 recovery lost the shared reservation before a transaction-created table"
                                .to_string(),
                        ));
                    }
                    self.compile_transaction_created_table_typed_insert_device_plan(
                        &prepared.table,
                        source,
                        crate::engine_residency::DeviceInsertRowIds::exact(ids),
                        transaction_metadata.commit_sequence,
                        lifecycle,
                    )
                }
            } else if prepared.indexed {
                let lifecycle = named_index_lifecycle
                    .take()
                    .expect("indexed plural replay retains one transaction lifecycle");
                if let Some(gate) = shared_gate.take() {
                    self.compile_transaction_terminal_indexed_typed_insert_device_plan_with_gate(
                        &prepared.table,
                        source,
                        crate::engine_residency::DeviceInsertRowIds::exact(ids),
                        lifecycle,
                        transaction_metadata.commit_sequence,
                        prepared.metadata.resets_existing_rows,
                        gate,
                        shared_budget_guard.take(),
                        prior_reserved_bytes,
                        manifest_predecessor.take(),
                    )
                } else {
                    self.compile_transaction_terminal_indexed_typed_insert_device_plan(
                        &prepared.table,
                        source,
                        crate::engine_residency::DeviceInsertRowIds::exact(ids),
                        lifecycle,
                        transaction_metadata.commit_sequence,
                        prepared.metadata.resets_existing_rows,
                    )
                }
            } else if let Some(gate) = shared_gate.take() {
                self.compile_transaction_terminal_typed_insert_device_plan_with_gate(
                    source,
                    crate::engine_residency::DeviceInsertRowIds::exact(ids),
                    transaction_metadata.commit_sequence,
                    prepared.metadata.resets_existing_rows,
                    gate,
                    shared_budget_guard.take(),
                    prior_reserved_bytes,
                )
            } else {
                self.compile_transaction_terminal_typed_insert_device_plan(
                    source,
                    crate::engine_residency::DeviceInsertRowIds::exact(ids),
                    transaction_metadata.commit_sequence,
                    prepared.metadata.resets_existing_rows,
                )
            }
            .map_err(|error| {
                EngineError::Durability(format!(
                    "plural codec-5 recovery device plan compilation failed: {error:?}"
                ))
            })?;
            if current_is_indexed_rollover {
                remaining_indexed_rollovers = remaining_indexed_rollovers.saturating_sub(1);
                if remaining_indexed_rollovers != 0 {
                    manifest_predecessor = plan.indexed_rollover_manifest_successor_predecessor();
                    if manifest_predecessor.is_none() {
                        return Err(EngineError::Durability(
                            "plural codec-5 recovery rollover plan lost its prepared manifest successor"
                                .to_string(),
                        ));
                    }
                }
            }
            if prepared.indexed && remaining_indexed_tables != 0 {
                named_index_lifecycle = plan.take_named_index_publication_guard();
                if named_index_lifecycle.is_none() {
                    return Err(EngineError::Durability(
                        "plural codec-5 recovery plan lost the shared named-index lifecycle"
                            .to_string(),
                    ));
                }
            }
            if position + 1 != plan_count {
                shared_gate = plan.take_transaction_terminal_unindexed_device_apply_guard();
                if shared_gate.is_none() {
                    return Err(EngineError::Durability(
                        "plural codec-5 recovery plan lost the shared mutation gate".to_string(),
                    ));
                }
                if let Some((guard, reserved_bytes)) =
                    plan.take_transaction_terminal_unindexed_budget_guard()
                {
                    prior_reserved_bytes = prior_reserved_bytes
                        .checked_add(reserved_bytes)
                        .ok_or_else(|| {
                            EngineError::Durability(
                                "plural codec-5 recovery reserved-byte charge overflows"
                                    .to_string(),
                            )
                        })?;
                    shared_budget_guard = Some(guard);
                }
            }
            plans.push(plan);
        }

        let mut commit = self.commit_state();
        let expected_index = commit.repl.peek_next_index();
        let (catalog_epoch, catalog_digest) = Self::canonical_catalog_boundary(
            commit.canonical_identity,
            commit.wal.canonical_catalog_tail()?,
        )?;
        if durable_record.txn_id != transaction_metadata.stable_transaction_id
            || transaction_metadata.canonical_identity != commit.canonical_identity
            || transaction_metadata.commit_sequence != expected_index
            || transaction_metadata.catalog_epoch != catalog_epoch
            || transaction_metadata.catalog_digest != catalog_digest
            || transaction_metadata.request_digest == [0; 32]
        {
            return Err(EngineError::Durability(
                "plural codec-5 recovery parent identity differs from its durable predecessor"
                    .to_string(),
            ));
        }
        let token = commit.repl.propose(Arc::clone(&durable_record.payload))?;
        if token.index != expected_index {
            return Err(EngineError::Durability(format!(
                "plural codec-5 recovery sequencer assigned {}, expected {expected_index}",
                token.index
            )));
        }
        commit.wal.append(durable_record.clone());
        commit.wal.flush_all()?;
        commit
            .repl
            .wait_committed(token, Duration::from_millis(0))?;
        let mut named_index_publication = plans
            .iter_mut()
            .find_map(|plan| plan.take_named_index_publication_guard());
        if let Some(lifecycle) = named_index_publication.as_mut() {
            lifecycle.enter_final_publication();
        }
        let _publication_owner = named_index_publication
            .as_ref()
            .map(|_| crate::engine_state::TransactionNamedIndexPublicationOwnerGuard::enter());
        let timestamp_micros =
            current_timestamp_micros().max(commit.max_commit_timestamp_micros.saturating_add(1));
        commit.record_commit_timestamp(durable_record.txn_id, timestamp_micros);
        for plan in plans {
            let permit =
                crate::engine_dml_concurrent::issue_transaction_terminal_typed_insert_apply_permit(
                    token.index,
                );
            plan.apply_after_transaction_wal_claim(self, permit)
                .map_err(|error| {
                    EngineError::Durability(format!(
                        "plural codec-5 recovery device apply failed after durable WAL: {error:?}"
                    ))
                })?;
        }
        self.read_state
            .mvcc
            .consume_proposed_row_id_range(proposed)
            .map_err(|error| {
                EngineError::Durability(format!(
                    "plural codec-5 replay allocator consumption drifted after device apply: {error}"
                ))
            })?;
        let mut write_set = WriteSet::default();
        for prepared in &prepared_tables {
            write_set.add_table(&prepared.table);
        }
        let table_names = prepared_tables
            .iter()
            .map(|prepared| prepared.table.name.clone())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let witness = crate::engine_commit::LiveTypedTransactionApply {
            txn_id: transaction_metadata.stable_transaction_id,
            expected_index: token.index,
            payload_authority: crate::engine_commit::LiveTypedPayloadAuthority::Digest(
                gpu_db_wal::canonical_request_digest(&durable_record.payload),
            ),
            payload_len: durable_record.payload.len(),
            tables: table_names,
            allocator_high_water: transaction_metadata.row_allocator_high_water,
            allocator_already_consumed: true,
            affected_rows: transaction_metadata.affected_rows,
            write_set,
            private_sequence_publications,
            catalog_composition,
            parent_authority: None,
        };
        self.apply_and_publish_committed_with_recovered_semantics_v2_codec5_transaction(
            &mut commit,
            transaction_metadata.stable_transaction_id,
            token.index,
            witness,
            root_publication,
        )?;
        let _generation_authority = prepared_tables
            .into_iter()
            .filter_map(|prepared| prepared.generation)
            .collect::<Vec<_>>();
        self.metrics.inc_commit();
        Ok(())
    }

    /// Recover one fully closed codec-5 aggregate through the same typed resident source,
    /// generic CUDA generation, device append, and root publication cut as a live INSERT. Bit 30
    /// retains the historical claim/lease owner; clear bit runs the one-record allocator cut and
    /// has no write-authority parent. This method neither decodes a legacy GPUDBOP1 operation nor
    /// rebuilds host relational rows.
    fn replay_validated_semantics_v2_typed_insert(
        &self,
        durable_record: &WalRecord,
        artifact: crate::typed_insert_aggregate::SemanticsV2ReplayArtifact,
        protocol: SemanticsV2ReplayProtocol,
    ) -> Result<(), EngineError> {
        if artifact.table_count() > 1 {
            return self.replay_validated_plural_semantics_v2_typed_insert(
                durable_record,
                artifact,
                protocol,
            );
        }
        let metadata = artifact.metadata();
        let mut projected_catalog = self.ddl_catalog().clone();
        if let Some(composition) = artifact
            .catalog_composition()
            .filter(|composition| !composition.catalog_commands.is_empty())
        {
            self.apply_codec5_catalog_composition(
                &mut projected_catalog,
                metadata.commit_sequence,
                composition,
            )?;
        }
        let mut prior_sequence_oid = None;
        for publication in artifact.private_sequence_publications() {
            if prior_sequence_oid.is_some_and(|prior| prior >= publication.sequence_oid) {
                return Err(EngineError::Durability(
                    "codec-5 recovery private sequence publications are not in unique stable-OID order"
                        .to_string(),
                ));
            }
            self.apply_codec5_private_sequence_name_binding(&mut projected_catalog, publication)?;
            prior_sequence_oid = Some(publication.sequence_oid);
        }
        let final_table = projected_catalog
            .relational_catalog
            .get(artifact.target_table_name())
            .cloned()
            .ok_or_else(|| {
                EngineError::Durability(format!(
                    "codec-5 recovery postimage lacks relation \"{}\"",
                    artifact.target_table_name()
                ))
            })?;
        // S3 first establishes the transaction's final catalog image, but that image must not
        // become public before the shared codec-5 terminal.  A first table generation therefore
        // borrows this authenticated postimage as its physical shape while proving that no
        // public table/root predecessor exists.
        let public_table = self
            .catalog_snapshot()
            .relational_catalog
            .get(artifact.target_table_name())
            .cloned();
        let table = match (metadata.initial_table_absent, public_table) {
            (true, None) => final_table.clone(),
            (true, Some(_)) => {
                return Err(EngineError::Durability(
                    "codec-5 recovery CREATE target already has a public predecessor".to_string(),
                ));
            }
            (false, Some(table)) => table,
            (false, None) => {
                return Err(EngineError::Durability(format!(
                    "codec-5 recovery targets unknown relation \"{}\"",
                    artifact.target_table_name()
                )));
            }
        };
        let replay_indexes = artifact.indexes().to_vec();
        let replay_row_sources = artifact.row_sources().to_vec();
        let (source, private_sequence_publications, catalog_composition) =
            artifact.into_recovery_source(&table, &final_table)?;
        // A paired-zero descriptor over an already-published table is legal only for the
        // S3 CREATE INDEX closure decoded above.  Keep that identity set explicit through the
        // existing GPU generation/root/publication lifecycle; it is not a second recovery path.
        let created_index_ids = replay_indexes
            .iter()
            .filter(|index| {
                !metadata.initial_table_absent
                    && index.base_generation == 0
                    && index.base_root == [0; 32]
            })
            .map(|index| index.stable_index_id)
            .collect::<Vec<_>>();
        if created_index_ids.iter().enumerate().any(|(ordinal, id)| {
            *id == 0 || *id == u64::MAX || ordinal != 0 && created_index_ids[ordinal - 1] >= *id
        }) || (metadata.resets_existing_rows && !created_index_ids.is_empty())
        {
            return Err(EngineError::Durability(
                "codec-5 recovery S3-created index identities are not an exact append-only proof"
                    .to_string(),
            ));
        }
        let mut s3_retired_index_candidates = catalog_composition
            .as_ref()
            .into_iter()
            .flat_map(|record| record.index_lifecycle_operations.iter())
            .flat_map(|operation| operation.targets.iter())
            .filter_map(|target| {
                match (
                    target.table_before.as_ref(),
                    target.table_after.as_ref(),
                    target.index_before.as_ref(),
                    target.index_after.as_ref(),
                ) {
                    (Some(table_before), Some(table_after), Some(index_before), None)
                        if table_before.oid == metadata.display_oid
                            && table_after.oid == metadata.display_oid
                            && index_before.table_oid == metadata.display_oid
                            && table
                                .indexes
                                .iter()
                                .any(|index| index.oid == index_before.oid) =>
                    {
                        Some(u64::from(index_before.oid))
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        s3_retired_index_candidates.sort_unstable();
        if s3_retired_index_candidates
            .windows(2)
            .any(|pair| pair[0] == pair[1])
        {
            return Err(EngineError::Durability(
                "codec-5 recovery S3 repeats a retired index identity".to_string(),
            ));
        }
        let row_count = u32::try_from(metadata.affected_rows).map_err(|_| {
            EngineError::Durability("codec-5 recovery affected rows exceed u32".to_string())
        })?;
        let surviving_row_count = u32::try_from(replay_row_sources.len()).map_err(|_| {
            EngineError::Durability("codec-5 recovery survivor count exceeds u32".to_string())
        })?;
        let proposed =
            crate::wal_binary::ProposedRowIdRange::new(metadata.row_allocator_before, row_count)?;
        if proposed.allocator_high_water() != metadata.row_allocator_high_water {
            return Err(EngineError::Durability(
                "codec-5 recovery allocator range differs from the retained aggregate".to_string(),
            ));
        }
        let surviving_high_water = metadata
            .row_allocator_before
            .checked_add(u64::from(surviving_row_count))
            .ok_or_else(|| {
                EngineError::Durability(
                    "codec-5 recovery survivor row-id range overflows".to_string(),
                )
            })?;
        let exact_row_ids = (metadata.row_allocator_before..surviving_high_water)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let expected_roots = self.read_state.typed_generation_roots.load_full();
        let predecessor = match (
            metadata.initial_table_absent,
            expected_roots.table(table.stable_table_id),
        ) {
            (true, None) => crate::engine_state::TypedTableGenerationRoot {
                data_generation: 0,
                table_root: [0; 32],
                logical_row_count: 0,
            },
            (true, Some(_)) => {
                return Err(EngineError::Durability(
                    "codec-5 recovery CREATE target already has a GPU predecessor".to_string(),
                ));
            }
            (false, Some(predecessor)) => predecessor,
            (false, None) => {
                return Err(EngineError::Durability(
                    "codec-5 recovery has no GPU-authenticated INSERT predecessor".to_string(),
                ));
            }
        };
        let predecessor_database_root = expected_roots.database_root;
        if predecessor.data_generation != metadata.data_generation_before
            || predecessor.table_root != metadata.initial_table_root
            || predecessor.logical_row_count != metadata.initial_logical_row_count
            || predecessor_database_root.is_some_and(|root| root != metadata.initial_database_root)
            || (!metadata.initial_table_absent && predecessor_database_root.is_none())
        {
            return Err(EngineError::Durability(
                "codec-5 recovery durable predecessor roots differ from CREATE publication"
                    .to_string(),
            ));
        }
        let retired_predecessor_index_ids = expected_roots
            .table_index_roots(metadata.stable_table_id)
            .filter(|root| {
                final_table
                    .indexes
                    .iter()
                    .all(|index| u64::from(index.oid) != root.stable_index_id)
            })
            .map(|root| root.stable_index_id)
            .collect::<Vec<_>>();
        if retired_predecessor_index_ids
            .iter()
            .any(|id| s3_retired_index_candidates.binary_search(id).is_err())
            || (metadata.resets_existing_rows
                && (!created_index_ids.is_empty() || !retired_predecessor_index_ids.is_empty()))
        {
            return Err(EngineError::Durability(
                "codec-5 recovery index roots differ outside the S3 transition proof".to_string(),
            ));
        }
        let has_s3_index_transition =
            !created_index_ids.is_empty() || !s3_retired_index_candidates.is_empty();
        let generation_table = if has_s3_index_transition {
            &final_table
        } else {
            &table
        };
        if replay_indexes.len() != generation_table.indexes.len()
            || generation_table.indexes.iter().any(|catalog_index| {
                replay_indexes
                    .iter()
                    .filter(|retained| retained.stable_index_id == u64::from(catalog_index.oid))
                    .count()
                    != 1
            })
        {
            return Err(EngineError::Durability(
                "codec-5 recovery index inventory differs from the S3-authenticated catalog postimage"
                    .to_string(),
            ));
        }
        if protocol == SemanticsV2ReplayProtocol::GenericOneRecord
            && self.read_state.mvcc.current_row_id() != metadata.row_allocator_before
        {
            return Err(EngineError::Durability(
                "generic one-record codec-5 allocator predecessor differs from replay frontier"
                    .to_string(),
            ));
        }
        let (mut plan, root_publication, generation) = if let Some(source) = source {
            let table_map_predecessor = match predecessor_database_root {
                Some(initial_database_root) => {
                    expected_roots
                        .retained_table_map_predecessor(metadata.stable_table_id, initial_database_root)?
                        .ok_or_else(|| {
                            EngineError::Durability(
                                "codec-5 recovery has no retained table-map witness"
                                    .to_string(),
                            )
                        })?
                }
                None if metadata.initial_table_absent => {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableMapPredecessor::UninitializedEmptyDatabase
                }
                None => {
                    return Err(EngineError::Durability(
                        "codec-5 recovery has no GPU-authenticated database predecessor"
                            .to_string(),
                    ));
                }
            };
            let mut runtime_indexes = Vec::with_capacity(replay_indexes.len());
            let publication_predecessor_index_roots = if metadata.initial_table_absent {
                Vec::new()
            } else {
                expected_roots
                    .table_index_roots(metadata.stable_table_id)
                    .collect::<Vec<_>>()
            };
            let mut successor_index_roots = Vec::with_capacity(replay_indexes.len());
            let mut key_start = 0_u32;
            let mut effect_start = 0_u32;
            for retained in &replay_indexes {
                let (catalog_index_ordinal, catalog_index) = generation_table
                    .indexes
                    .iter()
                    .enumerate()
                    .find(|(_, index)| u64::from(index.oid) == retained.stable_index_id)
                    .ok_or_else(|| {
                        EngineError::Durability(format!(
                            "codec-5 recovery index {} is absent from the authenticated catalog postimage",
                            retained.stable_index_id
                        ))
                    })?;
                let catalog_flags = u32::from(catalog_index.unique)
                    | (u32::from(catalog_index.primary_key) << 1)
                    | (u32::from(catalog_index.unique_constraint) << 2)
                    | (1 << 3);
                let catalog_keys_match = catalog_index
                    .key_columns
                    .iter()
                    .enumerate()
                    .zip(retained.key_columns.iter())
                    .try_fold(true, |matches, ((key_ordinal, key_name), retained_key)| {
                        let Some((catalog_column_ordinal, column)) = generation_table
                            .columns
                            .iter()
                            .enumerate()
                            .find(|(_, column)| column.name == *key_name)
                        else {
                            return Ok(false);
                        };
                        let column_name_digest =
                            crate::typed_insert_aggregate::write001_identifier_digest(
                                &column.name,
                            )?;
                        Ok(matches
                            && retained_key.key_ordinal == key_ordinal as u32
                            && retained_key.catalog_column_ordinal == catalog_column_ordinal as u32
                            && retained_key.stable_column_id == column.id
                            && retained_key.attnum == column.attnum
                            && retained_key.storage
                                == crate::typed_insert_batch::typed_image_sql_storage(column.ty)
                            && retained_key.declared_type_oid == column.type_oid
                            && retained_key.signed_type_size == column.type_size
                            && retained_key.column_name_digest == column_name_digest)
                    })?;
                if retained.raw_catalog_ordinal != catalog_index_ordinal as u32
                    || catalog_index.name != retained.name.as_ref()
                    || catalog_flags != retained.flags
                    || catalog_index.key_columns.len() != retained.key_columns.len()
                    || !catalog_keys_match
                {
                    return Err(EngineError::Durability(
                        "codec-5 recovery retained index descriptor differs from the authenticated catalog postimage"
                            .to_string(),
                    ));
                }
                let created_on_existing_table = created_index_ids
                    .binary_search(&retained.stable_index_id)
                    .is_ok();
                let predecessor_index = match (
                    metadata.initial_table_absent,
                    expected_roots
                        .table_index_root(metadata.stable_table_id, retained.stable_index_id),
                ) {
                    (true, None) => crate::engine_state::TypedIndexGenerationRoot {
                        stable_index_id: retained.stable_index_id,
                        index_generation: 0,
                        index_root: [0; 32],
                    },
                    (true, Some(_)) => {
                        return Err(EngineError::Durability(
                            "codec-5 recovery CREATE index already has a GPU predecessor"
                                .to_string(),
                        ));
                    }
                    (false, Some(predecessor)) => predecessor,
                    (false, None) if created_on_existing_table => {
                        crate::engine_state::TypedIndexGenerationRoot {
                            stable_index_id: retained.stable_index_id,
                            index_generation: 0,
                            index_root: [0; 32],
                        }
                    }
                    (false, None) => {
                        return Err(EngineError::Durability(format!(
                            "codec-5 recovery has no GPU-authenticated predecessor for index {}",
                            retained.stable_index_id
                        )));
                    }
                };
                if predecessor_index.index_generation != retained.base_generation
                    || predecessor_index.index_root != retained.base_root
                    || retained.final_generation != metadata.data_generation_after
                    || retained.final_root == [0; 32]
                    || retained.final_root == retained.base_root
                {
                    return Err(EngineError::Durability(
                    "codec-5 recovery durable index roots differ from the published predecessor"
                        .to_string(),
                ));
                }
                let key_columns = retained
                    .key_columns
                    .iter()
                    .map(
                        |key| gpu_db_execution::RuntimeTypedInsertGenerationIndexKeyColumn {
                            key_ordinal: key.key_ordinal,
                            catalog_column_ordinal: key.catalog_column_ordinal,
                            stable_column_id: key.stable_column_id,
                            attnum: key.attnum,
                            storage: key.storage,
                            declared_type_oid: key.declared_type_oid,
                            signed_type_size: key.signed_type_size,
                            column_name_digest: key.column_name_digest,
                        },
                    )
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let key_count = u32::try_from(key_columns.len()).map_err(|_| {
                    EngineError::Durability(
                        "codec-5 recovery index key count exceeds u32".to_string(),
                    )
                })?;
                runtime_indexes.push(
                    crate::engine_transaction_delta::TypedInsertRuntimeGenerationIndexInput {
                        descriptor: gpu_db_execution::RuntimeTypedInsertGenerationIndex {
                            stable_index_id: retained.stable_index_id,
                            raw_catalog_index_ordinal: retained.raw_catalog_ordinal,
                            index_flags: retained.flags,
                            null_equality_policy: retained.null_equality_policy,
                            base_generation: retained.base_generation,
                            base_root: retained.base_root,
                            key_start,
                            key_count,
                            effect_start,
                            effect_count: surviving_row_count,
                        },
                        key_columns,
                    },
                );
                key_start = key_start.checked_add(key_count).ok_or_else(|| {
                    EngineError::Durability(
                        "codec-5 recovery index key range overflows".to_string(),
                    )
                })?;
                effect_start = effect_start
                    .checked_add(surviving_row_count)
                    .ok_or_else(|| {
                        EngineError::Durability(
                            "codec-5 recovery index effect range overflows".to_string(),
                        )
                    })?;
                successor_index_roots.push(crate::engine_state::TypedIndexGenerationRoot {
                    stable_index_id: retained.stable_index_id,
                    index_generation: retained.final_generation,
                    index_root: retained.final_root,
                });
            }
            let generation = self.run_typed_insert_runtime_generation(
            crate::engine_transaction_delta::TypedInsertRuntimeGenerationInput {
                source: &source,
                row_allocator_before: metadata.row_allocator_before,
                first_row_id: metadata.row_allocator_before,
                row_sources: &replay_row_sources,
                database_id: metadata.canonical_identity.database_id,
                catalog_epoch: metadata.catalog_epoch,
                catalog_digest: metadata.catalog_digest,
                stable_transaction_id: metadata.stable_transaction_id,
                commit_sequence: metadata.commit_sequence,
                typed_statement_digest: metadata.typed_statement_digest,
                action: if metadata.initial_table_absent {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableAction::CreateWithRowSet
                } else if metadata.resets_existing_rows {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableAction::ResetThenRowSetInsert
                } else if !created_index_ids.is_empty() {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableAction::CreateIndexThenRowSetInsert
                } else {
                    gpu_db_execution::RuntimeTypedInsertGenerationTableAction::RowSetInsert
                },
                table_map_predecessor,
                stable_table_id: metadata.stable_table_id,
                write001_final_image_ref: 0,
                // CUDA derives the first table root at commit sequence while S7 retains the
                // true zero predecessor for replay closure.
                base_data_generation: if metadata.initial_table_absent {
                    metadata.commit_sequence
                } else {
                    predecessor.data_generation
                },
                base_table_root: predecessor.table_root,
                row_allocator_high_water: metadata.row_allocator_high_water,
                initial_logical_row_count: predecessor.logical_row_count,
                final_logical_row_count: metadata.final_logical_row_count,
                image_layout_digest: metadata.image_layout_digest,
                image_content_digest: metadata.image_content_digest,
                indexes: &runtime_indexes,
            },
        )?;
            if generation.initial_table_root != metadata.initial_table_root
                || generation.final_table_root != metadata.final_table_root
                || generation.initial_database_root != metadata.initial_database_root
                || generation.final_database_root != metadata.final_database_root
            {
                return Err(EngineError::Durability(
                    "codec-5 recovery CUDA commitments differ from durable S7 roots".to_string(),
                ));
            }
            let mut generated_index_roots = vec![
            gpu_db_execution::RuntimeTypedInsertGenerationIndexRoot {
                stable_index_id: 0,
                initial_generation: 0,
                initial_root: [0; 32],
                final_generation: 0,
                final_root: [0; 32],
            };
            replay_indexes.len()
        ];
            generation
                .logical_completion
                .copy_index_generation_roots_into(&mut generated_index_roots)
                .map_err(|_| {
                    EngineError::Durability(
                        "codec-5 recovery GPU index-root cardinality drifted".to_string(),
                    )
                })?;
            if generated_index_roots
                .iter()
                .zip(&replay_indexes)
                .any(|(generated, retained)| {
                    generated.stable_index_id != retained.stable_index_id
                        || generated.initial_generation != retained.base_generation
                        || generated.initial_root != retained.base_root
                        || generated.final_generation != retained.final_generation
                        || generated.final_root != retained.final_root
                })
            {
                return Err(EngineError::Durability(
                    "codec-5 recovery CUDA index commitments differ from durable S7 roots"
                        .to_string(),
                ));
            }
            let mut shape_roots = vec![[0; 32]; table.columns.len()];
            let mut column_roots = vec![[0; 32]; table.columns.len()];
            generation
                .logical_completion
                .copy_column_roots_into(&mut shape_roots, &mut column_roots)
                .map_err(|_| {
                    EngineError::Durability(
                        "codec-5 recovery GPU column-root cardinality drifted".to_string(),
                    )
                })?;
            let successor_columns = table
                .columns
                .iter()
                .enumerate()
                .map(|(ordinal, column)| {
                    Ok(crate::engine_state::TypedColumnGenerationRoot {
                        catalog_column_ordinal: u32::try_from(ordinal).map_err(|_| {
                            EngineError::Durability(
                                "codec-5 recovery column ordinal exceeds u32".to_string(),
                            )
                        })?,
                        stable_column_id: column.id,
                        attnum: column.attnum,
                        column_shape_root: shape_roots[ordinal],
                        column_root: column_roots[ordinal],
                    })
                })
                .collect::<Result<Vec<_>, EngineError>>()?;
            let root_publication = crate::engine_commit::LiveTypedGenerationRootPublication::
            from_exact_gpu_completed_table_map_predecessor_with_created_indexes(
                expected_roots,
                metadata.stable_table_id,
                (!metadata.initial_table_absent).then_some(predecessor),
                metadata.resets_existing_rows,
                predecessor_database_root,
                crate::engine_state::TypedTableGenerationRoot {
                    data_generation: metadata.data_generation_after,
                    table_root: metadata.final_table_root,
                    logical_row_count: metadata.final_logical_row_count,
                },
                &successor_columns,
                &publication_predecessor_index_roots,
                &successor_index_roots,
                metadata.final_database_root,
                &generation.table_map_completion,
                &created_index_ids,
                &retired_predecessor_index_ids,
            )?;
            let plan = if metadata.initial_table_absent {
                self.compile_transaction_created_table_typed_insert_device_plan(
                    &table,
                    source,
                    crate::engine_residency::DeviceInsertRowIds::exact(exact_row_ids),
                    metadata.commit_sequence,
                    (!replay_indexes.is_empty()).then(|| {
                        self.read_state
                            .residency
                            .begin_transaction_named_index_publication(
                                std::collections::BTreeSet::from([table.name.clone()]),
                            )
                    }),
                )
            } else if replay_indexes.is_empty() {
                self.compile_transaction_terminal_typed_insert_device_plan(
                    source,
                    crate::engine_residency::DeviceInsertRowIds::exact(exact_row_ids),
                    metadata.commit_sequence,
                    metadata.resets_existing_rows,
                )
            } else {
                let lifecycle = self
                    .read_state
                    .residency
                    .begin_transaction_named_index_publication(std::collections::BTreeSet::from([
                        table.name.clone(),
                    ]));
                if !has_s3_index_transition {
                    self.compile_transaction_terminal_indexed_typed_insert_device_plan(
                        &table,
                        source,
                        crate::engine_residency::DeviceInsertRowIds::exact(exact_row_ids),
                        lifecycle,
                        metadata.commit_sequence,
                        metadata.resets_existing_rows,
                    )
                } else {
                    self.compile_transaction_terminal_s3_created_index_typed_insert_device_plan(
                        &final_table,
                        &table,
                        &created_index_ids,
                        &s3_retired_index_candidates,
                        source,
                        crate::engine_residency::DeviceInsertRowIds::exact(exact_row_ids),
                        lifecycle,
                        metadata.commit_sequence,
                    )
                }
            }
            .map_err(|error| {
                EngineError::Durability(format!(
                    "codec-5 recovery device plan compilation failed: {error:?}"
                ))
            })?;
            (Some(plan), Some(root_publication), Some(generation))
        } else {
            if !replay_indexes.is_empty() || !replay_row_sources.is_empty() {
                return Err(EngineError::Durability(
                    "neutral codec-5 recovery retained physical generation work".to_string(),
                ));
            }
            (None, None, None)
        };

        let mut commit = self.commit_state();
        let expected_index = commit.repl.peek_next_index();
        let (catalog_epoch, catalog_digest) = Self::canonical_catalog_boundary(
            commit.canonical_identity,
            commit.wal.canonical_catalog_tail()?,
        )?;
        if durable_record.txn_id != metadata.stable_transaction_id
            || metadata.canonical_identity != commit.canonical_identity
            || metadata.commit_sequence != expected_index
            || metadata.catalog_epoch != catalog_epoch
            || metadata.catalog_digest != catalog_digest
            || metadata.request_digest == [0; 32]
        {
            return Err(EngineError::Durability(
                "codec-5 recovery parent identity differs from its durable predecessor".to_string(),
            ));
        }
        let token = commit.repl.propose(Arc::clone(&durable_record.payload))?;
        if token.index != expected_index {
            return Err(EngineError::Durability(format!(
                "codec-5 recovery sequencer assigned {}, expected {expected_index}",
                token.index
            )));
        }
        commit.wal.append(durable_record.clone());
        commit.wal.flush_all()?;
        commit
            .repl
            .wait_committed(token, Duration::from_millis(0))?;
        let mut named_index_publication = plan
            .as_mut()
            .and_then(|plan| plan.take_named_index_publication_guard());
        if let Some(lifecycle) = named_index_publication.as_mut() {
            lifecycle.enter_final_publication();
        }
        let _publication_owner = named_index_publication
            .as_ref()
            .map(|_| crate::engine_state::TransactionNamedIndexPublicationOwnerGuard::enter());
        let timestamp_micros =
            current_timestamp_micros().max(commit.max_commit_timestamp_micros.saturating_add(1));
        commit.record_commit_timestamp(durable_record.txn_id, timestamp_micros);
        if let Some(plan) = plan {
            let permit =
                crate::engine_dml_concurrent::issue_transaction_terminal_typed_insert_apply_permit(
                    token.index,
                );
            plan.apply_after_transaction_wal_claim(self, permit)
                .map_err(|error| {
                    EngineError::Durability(format!(
                        "codec-5 recovery device apply failed after durable WAL: {error:?}"
                    ))
                })?;
        }
        let allocator_already_consumed = protocol == SemanticsV2ReplayProtocol::GenericOneRecord;
        if allocator_already_consumed {
            self.read_state
                .mvcc
                .consume_proposed_row_id_range(proposed)
                .map_err(|error| {
                    EngineError::Durability(format!(
                        "generic one-record codec-5 replay allocator consumption drifted after device apply: {error}"
                    ))
                })?;
        }
        let mut write_set = WriteSet::default();
        write_set.add_table(&table);
        let witness = crate::engine_commit::LiveTypedTransactionApply {
            txn_id: metadata.stable_transaction_id,
            expected_index: token.index,
            payload_authority: crate::engine_commit::LiveTypedPayloadAuthority::Digest(
                gpu_db_wal::canonical_request_digest(&durable_record.payload),
            ),
            payload_len: durable_record.payload.len(),
            tables: Box::new([table.name.clone()]),
            allocator_high_water: metadata.row_allocator_high_water,
            allocator_already_consumed,
            affected_rows: metadata.affected_rows,
            write_set,
            private_sequence_publications,
            catalog_composition,
            parent_authority: (protocol == SemanticsV2ReplayProtocol::HistoricalWriteAuthority)
                .then_some(crate::engine_commit::LiveTypedParentAuthority {
                    request_digest: metadata.request_digest,
                    autocommit: true,
                    typed_statement_digests: Box::new([metadata.typed_statement_digest]),
                }),
        };
        self.apply_and_publish_committed_with_recovered_semantics_v2_codec5_transaction(
            &mut commit,
            metadata.stable_transaction_id,
            token.index,
            witness,
            root_publication,
        )?;
        if let Some(lifecycle) = named_index_publication {
            lifecycle.complete();
        }
        let _generation_authority =
            generation.map(|generation| (generation.commitments, generation.logical_completion));
        self.metrics.inc_commit();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_transaction_reset::table_schema_digest;

    #[test]
    fn clear_bit_codec5_preflight_rejects_parent_markers_and_frontier_drift() {
        let identity = Engine::fresh_canonical_identity();
        let mut markers = crate::engine_write_authority::DurableWriteAuthorityIndex::default();
        markers
            .apply(
                7,
                crate::engine_write_authority::DecodedWriteAuthority::RetentionClaim(
                    crate::engine_write_authority::CanonicalRetentionClaim {
                        identity,
                        leader_epoch: 1,
                        parent_stable_transaction_id: 41,
                        parent_request_digest: [3; 32],
                        parent_autocommit: true,
                        statement_digests: Box::new([[4; 32]]),
                        eligible_statement_bits: Box::new([0]),
                        candidate_deadline: 0,
                    },
                ),
                0,
            )
            .expect("well-formed retained marker enters the recovery index");
        let marker = validate_generic_codec5_replay_preflight(&markers, 41, 0, 1, 1, 0)
            .expect_err("clear-bit codec-5 must reject any parent marker");
        assert!(marker
            .to_string()
            .contains("retired claim or allocator markers"));

        let frontier = validate_generic_codec5_replay_preflight(
            &crate::engine_write_authority::DurableWriteAuthorityIndex::default(),
            42,
            2,
            3,
            1,
            0,
        )
        .expect_err("clear-bit codec-5 must require the exact simulated allocator frontier");
        assert!(frontier
            .to_string()
            .contains("allocator range is not the exact replay frontier"));
    }

    #[test]
    fn transaction_claim_status_length_helper_matches_real_encoder() {
        let body = Engine::encode_transaction_claim_status(
            Engine::fresh_canonical_identity(),
            73,
            [0x5a; 32],
        );
        assert_eq!(canonical_transaction_claim_status_len(), 100);
        assert_eq!(body.len(), canonical_transaction_claim_status_len());
    }

    #[path = "product_001_tests.rs"]
    mod product_001_tests;

    #[test]
    fn pre_product_002_typed_dml_bodies_remain_canonical() {
        const OLD_BODIES: [&[u8]; 3] = [
            br#"{"Insert":{"table":"t","columns":["id"],"rows":[[{"Int4":1}]]}}"#,
            br#"{"Update":{"table":"t","assignments":[{"column":"v","value":{"Int8":1}}],"filter":null,"filters":[],"filter_groups":[]}}"#,
            br#"{"Delete":{"table":"t","filter":null,"filters":[],"filter_groups":[]}}"#,
        ];

        for old_body in OLD_BODIES {
            let mut operation =
                Vec::with_capacity(ENGINE_OPERATION_MAGIC.len() + 12 + old_body.len());
            operation.extend_from_slice(ENGINE_OPERATION_MAGIC);
            operation.push(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V1);
            operation.extend_from_slice(&[0; 3]);
            operation.extend_from_slice(&(old_body.len() as u64).to_le_bytes());
            operation.extend_from_slice(old_body);

            let replay = Engine::decode_engine_operation(&operation)
                .expect("pre-PRODUCT-002 canonical body must remain replayable");
            let command = Engine::decode_engine_command(&replay)
                .expect("typed replay body")
                .expect("typed command");
            assert_eq!(serde_json::to_vec(&command).unwrap(), old_body);
        }
    }

    #[test]
    fn pre_ordered_canonical_composite_envelopes_keep_historical_table_counts() {
        // Literal opcode-5 payload captured from the pre-ordered composite codec. The command
        // JSON, opcode, and fixed-width framing are deliberately not produced by today's encoder:
        // this fixture must remain recoverable with the historical catalog-only table count 0.
        const LEGACY_CREATE: &[u8] = br#"{"CreateTable":{"table":"composite_codec","columns":[{"name":"id","ty":"Int4","domain":null,"default":null}],"primary_key":null,"unique_constraints":[],"check_constraints":[]}}"#;
        assert_eq!(LEGACY_CREATE.len(), 176);
        let mut literal_opcode_5 = vec![
            255, 1, 5, // binary tag, version, legacy composite opcode
            1, 0, 0, 0, // one catalog command
            176, 0, 0, 0, // literal typed-command length
        ];
        literal_opcode_5.extend_from_slice(LEGACY_CREATE);
        literal_opcode_5.extend_from_slice(&8_u64.to_le_bytes());
        literal_opcode_5.extend_from_slice(&0_u32.to_le_bytes()); // no sequence advances
        literal_opcode_5.extend_from_slice(&0_u32.to_le_bytes()); // no row mutations
        let BinaryWalRecord::Transaction(decoded) =
            decode_binary_record(&literal_opcode_5).unwrap()
        else {
            panic!("literal opcode-5 fixture did not decode as a transaction");
        };
        assert!(decoded.operation_order.is_empty());
        assert_eq!(decoded.catalog_commands.len(), 1);

        let payload: Arc<[u8]> = Arc::from(literal_opcode_5);
        let identity = Engine::fresh_canonical_identity();
        let record = Engine::canonical_wal_record_with_boundary_and_request_digest(
            identity,
            0,
            Engine::canonical_genesis_catalog_digest(identity),
            6_901,
            1,
            0,
            &payload,
            gpu_db_wal::canonical_request_digest(&payload),
        )
        .unwrap();
        let envelope = gpu_db_wal::decode_canonical_record_payload(&record.as_wal_record().payload)
            .unwrap()
            .unwrap();
        assert_eq!(envelope.header.table_block_count, 0);
        let recovered = Engine::recover_from_durable_wal(&[record.into_wal_record()]).unwrap();
        assert!(recovered
            .catalog_snapshot()
            .relational_catalog
            .contains_key("composite_codec"));

        // Opcode 8 historically counted only its existing-table mutation output, not the table
        // created by its catalog prefix. Exercise that exact canonical recovery shape as well.
        let prefix_records = vec![WalRecord {
            txn_id: 6_902,
            payload: Arc::from(&b"CREATE TABLE legacy_identity_rows (id int4)"[..]),
        }];
        let prefix = Engine::recover_from_durable_wal(&prefix_records).unwrap();
        let table = prefix.catalog_snapshot().relational_catalog["legacy_identity_rows"].clone();
        let legacy_identity = BinaryTransactionRecord {
            catalog_epoch: BinaryTransactionCatalogEpoch::Legacy,
            allocator_high_water: 2,
            catalog_commands: vec![BinaryTransactionCatalogCommand {
                ordinal: 0,
                command: parse_command("CREATE TABLE legacy_identity_created (id int4)").unwrap(),
            }],
            created_table_identities: BTreeMap::new(),
            created_table_index_identities: BTreeMap::new(),
            catalog_output: None,
            view_operations: Vec::new(),
            view_lifecycle_operations: Vec::new(),
            index_lifecycle_operations: Vec::new(),
            sequence_lifecycle_operations: Vec::new(),
            sequence_reset_operations: Vec::new(),
            sequence_advances_by_oid: BTreeMap::new(),
            operation_order: Vec::new(),
            statement_digests: Vec::new(),
            sequence_input_oids: BTreeMap::new(),
            sequence_value_references: Vec::new(),
            table_resets: Vec::new(),
            sequence_advances: BTreeMap::new(),
            table_identities: BTreeMap::from([(
                table.name.clone(),
                BinaryTransactionTableIdentity {
                    table_oid: table.oid,
                    schema_digest: table_schema_digest(&table).unwrap(),
                },
            )]),
            mutations: vec![BinaryTransactionMutation::Insert {
                table: table.name.clone(),
                row_id: 1,
                row_encoded: encode_relational_row(&[SqlValue::Int4(7)]),
            }],
        };
        let legacy_payload: Arc<[u8]> =
            Arc::from(try_encode_binary_transaction(&legacy_identity).unwrap());
        assert_eq!(legacy_payload[..3], [255, 1, 8]);
        let legacy_identity_anchor = Engine::fresh_canonical_identity();
        let legacy_record = Engine::canonical_wal_record_with_boundary_and_request_digest(
            legacy_identity_anchor,
            0,
            Engine::canonical_genesis_catalog_digest(legacy_identity_anchor),
            6_903,
            2,
            0,
            &legacy_payload,
            gpu_db_wal::canonical_request_digest(&legacy_payload),
        )
        .unwrap();
        let legacy_envelope =
            gpu_db_wal::decode_canonical_record_payload(&legacy_record.as_wal_record().payload)
                .unwrap()
                .unwrap();
        assert_eq!(legacy_envelope.header.table_block_count, 1);
        let recovered = Engine::recover_from_durable_wal(&[
            prefix_records[0].clone(),
            legacy_record.into_wal_record(),
        ])
        .unwrap();
        assert!(recovered
            .catalog_snapshot()
            .relational_catalog
            .contains_key("legacy_identity_created"));
        assert_eq!(
            recovered
                .execute_relational_select_text("SELECT id FROM legacy_identity_rows")
                .unwrap()
                .rows,
            vec![vec![SqlValue::Int4(7)]]
        );
    }

    #[test]
    fn pre_public_only_view_bodies_remain_canonical_and_true_round_trips() {
        const OLD_BODIES: [&[u8]; 2] = [
            br#"{"CreateView":{"name":"legacy_v","query":{"table":"legacy_source","distinct":false,"projection":"All","group_by":null,"having_groups":[],"filter":null,"filters":[],"filter_groups":[],"order_by":[],"limit":null,"offset":null},"definition":"SELECT * FROM legacy_source","or_replace":false}}"#,
            br#"{"CreateMaterializedView":{"name":"legacy_mv","query":{"table":"legacy_source","distinct":false,"projection":"All","group_by":null,"having_groups":[],"filter":null,"filters":[],"filter_groups":[],"order_by":[],"limit":null,"offset":null},"definition":"SELECT * FROM legacy_source","with_data":false}}"#,
        ];

        for old_body in OLD_BODIES {
            let mut typed = Vec::with_capacity(10 + old_body.len());
            typed.push(ENGINE_TYPED_COMMAND_TAG);
            typed.push(ENGINE_TYPED_COMMAND_VERSION_LEGACY);
            typed.extend_from_slice(&(old_body.len() as u64).to_le_bytes());
            typed.extend_from_slice(old_body);
            let command = Engine::decode_engine_command(&typed)
                .expect("historical typed view command must decode")
                .expect("typed command");
            let historical_public_only = match &command {
                Command::CreateView(view) => view.query.public_only,
                Command::CreateMaterializedView(view) => view.query.public_only,
                other => panic!("unexpected historical command: {other:?}"),
            };
            assert!(!historical_public_only);
            assert_eq!(serde_json::to_vec(&command).unwrap(), old_body);

            let mut operation =
                Vec::with_capacity(ENGINE_OPERATION_MAGIC.len() + 12 + old_body.len());
            operation.extend_from_slice(ENGINE_OPERATION_MAGIC);
            operation.push(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V1);
            operation.extend_from_slice(&[0; 3]);
            operation.extend_from_slice(&(old_body.len() as u64).to_le_bytes());
            operation.extend_from_slice(old_body);
            let replay = Engine::decode_engine_operation(&operation)
                .expect("historical canonical view fragment must remain replayable");
            assert_eq!(
                Engine::decode_engine_command(&replay).unwrap(),
                Some(command)
            );
        }

        let current =
            parse_command("CREATE VIEW current_v AS SELECT * FROM public.current_source").unwrap();
        let Command::CreateView(view) = &current else {
            panic!("expected CREATE VIEW");
        };
        assert!(view.query.public_only);
        let current_body = serde_json::to_vec(&current).unwrap();
        let marker = b"\"public_only\":true";
        assert!(current_body
            .windows(marker.len())
            .any(|window| window == marker));
        let decoded: Command = serde_json::from_slice(&current_body).unwrap();
        assert_eq!(decoded, current);

        let mut current_operation =
            Vec::with_capacity(ENGINE_OPERATION_MAGIC.len() + 12 + current_body.len());
        current_operation.extend_from_slice(ENGINE_OPERATION_MAGIC);
        current_operation.push(ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2);
        current_operation.extend_from_slice(&[0; 3]);
        current_operation.extend_from_slice(&(current_body.len() as u64).to_le_bytes());
        current_operation.extend_from_slice(&current_body);
        let current_replay = Engine::decode_engine_operation(&current_operation)
            .expect("current canonical view fragment must remain replayable");
        assert_eq!(
            Engine::decode_engine_command(&current_replay).unwrap(),
            Some(current)
        );
    }

    #[test]
    fn canonical_record_is_active_and_same_id_retry_resolves_exactly() {
        let engine = Engine::new_local();
        let payload: Arc<[u8]> = Arc::from(&b"SET canonical=value"[..]);
        let first = engine.commit_mutation(7001, Arc::clone(&payload)).unwrap();
        let before = engine.durable_wal_records();
        let envelope = gpu_db_wal::decode_canonical_record_payload(&before[0].payload)
            .unwrap()
            .expect("new commits use canonical authority");
        assert_eq!(envelope.header.stable_transaction_id, 7001);
        assert_eq!(envelope.header.commit_seq, first.index);
        assert_eq!(envelope.fragments.len(), 2);
        assert_eq!(
            envelope.fragments[1].kind,
            gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus
        );
        assert_eq!(
            envelope.outcome.kind,
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
        );

        let retry = engine.commit_mutation(7001, Arc::clone(&payload)).unwrap();
        assert_eq!(retry, first);
        assert_eq!(engine.durable_wal_records(), before);
        let error = engine
            .commit_mutation(7001, Arc::from(&b"SET canonical=different"[..]))
            .unwrap_err();
        assert!(error.to_string().contains("claimed by a different request"));
    }

    #[test]
    fn typed_command_body_is_parse_free_and_replay_rejects_a_false_outcome() {
        let source = Engine::new_local();
        let ddl: Arc<[u8]> = Arc::from(&b"CREATE TABLE typed_outcome (id INT PRIMARY KEY)"[..]);
        source.commit_mutation(7051, Arc::clone(&ddl)).unwrap();
        let records = source.durable_wal_records();
        let ddl_envelope = gpu_db_wal::decode_canonical_record_payload(&records[0].payload)
            .unwrap()
            .unwrap();
        let replay_payload = Engine::decode_engine_operation(&ddl_envelope.fragments[0].body)
            .expect("typed command operation");
        assert_eq!(
            Engine::decode_engine_command(&replay_payload).unwrap(),
            Some(parse_command(std::str::from_utf8(&ddl).unwrap()).unwrap())
        );
        assert!(
            !ddl_envelope.fragments[0]
                .body
                .windows(ddl.len())
                .any(|window| window == ddl.as_ref()),
            "canonical command body must not retain SQL source text"
        );

        let insert: Arc<[u8]> = Arc::from(&b"INSERT INTO typed_outcome VALUES (1)"[..]);
        let valid_insert = Engine::canonical_wal_record_with_boundary_and_outcome(
            ddl_envelope.header.identity,
            ddl_envelope.header.catalog_after_epoch,
            ddl_envelope.header.catalog_after_digest,
            7052,
            2,
            0,
            &insert,
            gpu_db_wal::canonical_request_digest(&insert),
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            1,
        )
        .unwrap();
        let valid_envelope =
            gpu_db_wal::decode_canonical_record_payload(&valid_insert.as_wal_record().payload)
                .unwrap()
                .unwrap();
        let false_noop = gpu_db_wal::CanonicalOutcome {
            kind: gpu_db_wal::CanonicalOutcomeKind::CommitNoOp,
            affected_rows: 0,
            ..valid_envelope.outcome
        };
        let false_noop = gpu_db_wal::encode_canonical_envelope(
            valid_envelope.physical,
            &valid_envelope.header,
            &valid_envelope.fragments,
            &false_noop,
        )
        .unwrap()
        .into_prepared_record(7052)
        .unwrap();
        let error = match Engine::recover_from_durable_wal(&[
            records[0].clone(),
            false_noop.into_wal_record(),
        ]) {
            Ok(_) => panic!("false affected-row marker was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("outcome mismatch"), "{error}");
    }

    #[test]
    fn recovery_validates_all_canonical_bytes_and_lineage_before_apply() {
        let first = Engine::new_local();
        first
            .commit_mutation(7101, Arc::from(&b"SET a=1"[..]))
            .unwrap();
        first
            .commit_mutation(7102, Arc::from(&b"SET b=2"[..]))
            .unwrap();
        first
            .commit_mutation(7103, Arc::from(&b"SET c=3"[..]))
            .unwrap();
        let records = first.durable_wal_records();

        let mut corrupted = records.clone();
        let mut bytes = corrupted[1].payload.to_vec();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0x5a;
        corrupted[1].payload = Arc::from(bytes);
        assert!(Engine::recover_from_durable_wal(&corrupted).is_err());

        let gap = vec![records[0].clone(), records[2].clone()];
        let error = match Engine::recover_from_durable_wal(&gap) {
            Ok(_) => panic!("commit-sequence gap was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("commit sequence gap"));

        let foreign = Engine::new_local();
        foreign
            .commit_mutation(7201, Arc::from(&b"SET foreign=1"[..]))
            .unwrap();
        let mixed = vec![records[0].clone(), foreign.durable_wal_records()[0].clone()];
        let error = match Engine::recover_from_durable_wal(&mixed) {
            Ok(_) => panic!("foreign lineage was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("lineage changed"));

        let legacy_suffix = vec![
            records[0].clone(),
            WalRecord {
                txn_id: 7104,
                payload: Arc::from(&b"SET legacy=after-canonical"[..]),
            },
        ];
        let error = match Engine::recover_from_durable_wal(&legacy_suffix) {
            Ok(_) => panic!("legacy suffix crossed the canonical migration barrier"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("boundary"), "{error}");

        // Re-encode a self-consistent envelope with a forged logical catalog boundary. Byte-level
        // checksums alone cannot detect this; recovery must bind it to the preceding record.
        let envelope = gpu_db_wal::decode_canonical_record_payload(&records[1].payload)
            .unwrap()
            .unwrap();
        let mut header = envelope.header;
        header.catalog_before_digest[0] ^= 0x5a;
        header.catalog_after_digest = header.catalog_before_digest;
        let forged = gpu_db_wal::encode_canonical_envelope(
            envelope.physical,
            &header,
            &envelope.fragments,
            &envelope.outcome,
        )
        .unwrap();
        let mut forged_chain = records.clone();
        forged_chain[1].payload =
            Arc::from(gpu_db_wal::pack_canonical_record_payload(&forged).unwrap());
        let error = match Engine::recover_from_durable_wal(&forged_chain) {
            Ok(_) => panic!("forged catalog lineage was accepted"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("catalog-before boundary"),
            "{error}"
        );
    }

    #[test]
    fn canonical_allocator_high_water_is_checked_and_monotonic_on_replay() {
        let values = vec![SqlValue::Int4(7)];
        let payload: Arc<[u8]> = Arc::from(
            encode_historical_binary_insert_fixture(
                "allocator_t",
                &[(u64::MAX - 1, values.as_slice())],
            )
            .expect("encodable boundary row"),
        );
        let identity = Engine::fresh_canonical_identity();
        let record = Engine::canonical_wal_record_with_boundary_and_request_digest(
            identity,
            0,
            Engine::canonical_genesis_catalog_digest(identity),
            7251,
            1,
            0,
            &payload,
            gpu_db_wal::canonical_request_digest(&payload),
        )
        .unwrap();
        let envelope = gpu_db_wal::decode_canonical_record_payload(&record.as_wal_record().payload)
            .unwrap()
            .unwrap();
        assert_eq!(envelope.header.allocator_high_water, u64::MAX);

        let invalid: Arc<[u8]> = Arc::from(
            encode_historical_binary_insert_fixture(
                "allocator_t",
                &[(u64::MAX, values.as_slice())],
            )
            .expect("binary framing can represent the sentinel for rejection"),
        );
        let error = Engine::canonical_wal_record_with_boundary_and_request_digest(
            identity,
            0,
            Engine::canonical_genesis_catalog_digest(identity),
            7252,
            1,
            0,
            &invalid,
            gpu_db_wal::canonical_request_digest(&invalid),
        )
        .unwrap_err();
        assert!(error.to_string().contains("reserved maximum row identity"));
    }

    #[test]
    fn canonical_recovery_is_repeatable_in_fresh_engine_contexts() {
        let live = Engine::new_local();
        live.commit_mutation(7301, Arc::from(&b"SET repeat=1"[..]))
            .unwrap();
        live.commit_mutation(7302, Arc::from(&b"SET repeat=2"[..]))
            .unwrap();
        let first = Engine::recover_from_durable_wal(&live.durable_wal_records()).unwrap();
        let second = Engine::recover_from_durable_wal(&first.durable_wal_records()).unwrap();
        assert_eq!(second.wal_flushed_count(), 2);
        let retry = second
            .commit_mutation(7302, Arc::from(&b"SET repeat=2"[..]))
            .unwrap();
        assert_eq!(retry.index, 2);
        assert_eq!(second.wal_flushed_count(), 2);
    }

    #[test]
    fn recovery_retries_context_loss_once_from_immutable_authority() {
        let source = Engine::new_local_test_engine();
        source
            .commit_mutation(7091, Arc::from(&b"SET context_retry=durable"[..]))
            .unwrap();
        let records = source.durable_wal_records();

        Engine::inject_one_recovery_context_loss();
        let recovered = Engine::recover_from_durable_wal(&records)
            .expect("one fresh-context recovery retry must succeed");
        assert_eq!(Engine::recovery_attempt_count(), 2);
        assert_eq!(recovered.get("context_retry"), Some("durable".to_string()));
        assert_eq!(recovered.durable_wal_records(), records);
    }

    #[test]
    fn recovery_stays_unavailable_after_the_bounded_context_retry() {
        let source = Engine::new_local_test_engine();
        source
            .commit_mutation(7092, Arc::from(&b"SET context_retry=bounded"[..]))
            .unwrap();
        let records = source.durable_wal_records();

        Engine::inject_recovery_context_losses(2);
        let error = match Engine::recover_from_durable_wal(&records) {
            Ok(_) => panic!("a second poisoned context became available"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("launch failed: 719"), "{error}");
        assert_eq!(Engine::recovery_attempt_count(), 2);
    }

    #[test]
    fn file_recovery_requires_matching_durable_identity_anchor() {
        let source = Engine::new_local();
        source
            .commit_mutation(7401, Arc::from(&b"SET anchored=1"[..]))
            .unwrap();
        let records = source.durable_wal_records();
        let identity = gpu_db_wal::decode_canonical_record_payload(&records[0].payload)
            .unwrap()
            .unwrap()
            .header
            .identity;
        let path = std::env::temp_dir().join(format!(
            "gpu-db-canonical-anchor-{}-{:?}.wal",
            std::process::id(),
            std::thread::current().id()
        ));
        gpu_db_wal::write_wal_segment(&path, &records).unwrap();
        std::fs::remove_file(gpu_db_wal::durable_identity_path(&path)).unwrap();
        assert!(Engine::recover_from_durable_wal_file(&path).is_err());

        let mut foreign = identity;
        foreign.database_id[0] ^= 0x5a;
        gpu_db_wal::write_durable_identity(&path, foreign).unwrap();
        let error = match Engine::recover_from_durable_wal_file(&path) {
            Ok(_) => panic!("foreign identity anchor was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("does not match"));

        gpu_db_wal::write_durable_identity(&path, identity).unwrap();
        Engine::inject_one_recovery_context_loss();
        let recovered = Engine::recover_from_durable_wal_file(&path)
            .expect("file recovery must restart from the durable identity and WAL");
        assert_eq!(Engine::recovery_attempt_count(), 2);
        assert_eq!(recovered.get("anchored"), Some("1".to_string()));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(gpu_db_wal::durable_identity_path(&path));
    }
}
