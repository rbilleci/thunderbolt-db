//! ADR-014 canonical WAL integration and recovery-lineage validation.

use super::*;

const ENGINE_OPERATION_MAGIC: &[u8; 8] = b"GPUDBOP1";
const TRANSACTION_STATUS_MAGIC: &[u8; 12] = b"GPUDBSTATUS1";
const ENGINE_OPERATION_CODEC_LEGACY_SQL: u8 = 1;
const ENGINE_OPERATION_CODEC_RESOLVED_BINARY: u8 = 2;
/// Historical canonical typed commands contain bare canonical JSON and replay with pre-PRODUCT-001
/// catalog semantics.
const ENGINE_OPERATION_CODEC_TYPED_COMMAND_V1: u8 = 3;
/// Additive discriminator for commands emitted after stable index OIDs/dependency policy landed.
const ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2: u8 = 4;
const ENGINE_TYPED_COMMAND_TAG: u8 = 0xfe;
const ENGINE_TYPED_COMMAND_VERSION_LEGACY: u8 = 1;
const ENGINE_TYPED_COMMAND_VERSION_CURRENT: u8 = 2;
static CANONICAL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

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

    fn canonical_catalog_transition(
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
        prior: Option<&WalRecord>,
    ) -> Result<(u64, gpu_db_wal::CanonicalDigest), EngineError> {
        let Some(prior) = prior else {
            return Ok((0, Self::canonical_genesis_catalog_digest(identity)));
        };
        let Some(envelope) = gpu_db_wal::decode_canonical_record_payload(&prior.payload)? else {
            return Ok((0, Self::canonical_genesis_catalog_digest(identity)));
        };
        if envelope.header.identity != identity {
            return Err(EngineError::Durability(
                "canonical catalog boundary crosses database lineage".to_string(),
            ));
        }
        Ok((
            envelope.header.catalog_after_epoch,
            envelope.header.catalog_after_digest,
        ))
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
                    ))
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

    fn recovery_scan_payload(record: &WalRecord) -> Result<Arc<[u8]>, EngineError> {
        if let Some(envelope) = gpu_db_wal::decode_canonical_record_payload(&record.payload)? {
            let operation = envelope.fragments.first().ok_or_else(|| {
                EngineError::Durability(format!(
                    "canonical WAL record {} has no operation fragment",
                    record.txn_id
                ))
            })?;
            return Self::decode_engine_operation(&operation.body);
        }
        if record.payload.first() == Some(&WAL_BINARY_TAG)
            || record.payload.first() == Some(&ENGINE_TYPED_COMMAND_TAG)
        {
            return Ok(Arc::clone(&record.payload));
        }
        let command = Self::decode_engine_command(&record.payload)?.ok_or_else(|| {
            EngineError::Durability(format!(
                "legacy WAL record {} has no engine command",
                record.txn_id
            ))
        })?;
        Self::encode_replay_typed_command(&command, ENGINE_TYPED_COMMAND_VERSION_LEGACY)
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
            let payload = Self::recovery_scan_payload(record)?;
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
        }
    }

    fn encode_engine_operation(payload: &[u8]) -> Result<Vec<u8>, EngineError> {
        let (codec, canonical_payload) = if payload.first() == Some(&WAL_BINARY_TAG) {
            // Decode before persistence so a tagged-but-malformed binary record never reaches a
            // canonical fragment and becomes an unknown committed operation at restart.
            decode_binary_record(payload)?;
            (ENGINE_OPERATION_CODEC_RESOLVED_BINARY, payload.to_vec())
        } else {
            let command = Self::decode_engine_command(payload)?.ok_or_else(|| {
                EngineError::Durability(
                    "canonical WAL command payload cannot be decoded".to_string(),
                )
            })?;
            let bytes = serde_json::to_vec(&command).map_err(|error| {
                EngineError::Durability(format!(
                    "canonical WAL typed command encode failed: {error}"
                ))
            })?;
            (ENGINE_OPERATION_CODEC_TYPED_COMMAND_V2, bytes)
        };
        let len = u64::try_from(canonical_payload.len()).map_err(|_| {
            EngineError::Durability("canonical WAL operation length overflow".to_string())
        })?;
        let mut body =
            Vec::with_capacity(ENGINE_OPERATION_MAGIC.len() + 12 + canonical_payload.len());
        body.extend_from_slice(ENGINE_OPERATION_MAGIC);
        body.push(codec);
        body.extend_from_slice(&[0; 3]);
        body.extend_from_slice(&len.to_le_bytes());
        body.extend_from_slice(&canonical_payload);
        Ok(body)
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

    fn encode_transaction_claim_status(
        identity: gpu_db_wal::CanonicalIdentity,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Vec<u8> {
        let mut body = Vec::with_capacity(96);
        body.extend_from_slice(TRANSACTION_STATUS_MAGIC);
        body.extend_from_slice(&identity.database_id);
        body.extend_from_slice(&identity.timeline_id);
        body.extend_from_slice(&txn_id.to_le_bytes());
        body.extend_from_slice(&request_digest);
        body.push(1); // durable claim pending publication; the terminal marker carries outcome
        body.extend_from_slice(&[0; 7]);
        body.extend_from_slice(&u64::MAX.to_le_bytes()); // retained/unexpired in the current policy
        body
    }

    fn validate_transaction_claim_status(
        body: &[u8],
        identity: gpu_db_wal::CanonicalIdentity,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<(), EngineError> {
        if body.len() != 100 || &body[..12] != TRANSACTION_STATUS_MAGIC {
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

    pub(crate) fn canonical_wal_record(
        commit: &CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
    ) -> Result<WalRecord, EngineError> {
        Self::canonical_wal_record_with_commit_request_digest(
            commit,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            gpu_db_wal::canonical_request_digest(payload),
        )
    }

    pub(crate) fn canonical_wal_record_with_isolation(
        commit: &CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        isolation: gpu_db_wal::CanonicalIsolation,
    ) -> Result<WalRecord, EngineError> {
        Self::canonical_wal_record_with_isolation_and_request_digest(
            commit,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            isolation,
            gpu_db_wal::canonical_request_digest(payload),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_with_isolation_and_request_digest(
        commit: &CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        isolation: gpu_db_wal::CanonicalIsolation,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<WalRecord, EngineError> {
        let (catalog_epoch, catalog_digest) =
            Self::canonical_catalog_boundary(commit.canonical_identity, commit.wal.last_record())?;
        Self::canonical_wal_record_with_boundary_and_outcome_isolation(
            commit.canonical_identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            Self::canonical_affected_rows(payload)?,
            isolation,
        )
    }

    pub(crate) fn canonical_affected_rows(payload: &[u8]) -> Result<u64, EngineError> {
        if payload.first() == Some(&WAL_BINARY_TAG) {
            match decode_binary_record(payload)? {
                crate::wal_binary::BinaryWalRecord::Insert(record) => Ok(record.rows.len() as u64),
                crate::wal_binary::BinaryWalRecord::Transaction(record) => {
                    Ok(record.mutations.len() as u64)
                }
                crate::wal_binary::BinaryWalRecord::DeleteByKey(_)
                | crate::wal_binary::BinaryWalRecord::UpdateByKey(_) => {
                    Err(EngineError::Durability(
                        "unresolved by-key WAL requires an exact GPU outcome marker".to_string(),
                    ))
                }
            }
        } else {
            match Self::decode_engine_command(payload)?.ok_or_else(|| {
                EngineError::Durability("canonical WAL command has no typed operation".to_string())
            })? {
                Command::Insert(insert) => Ok(insert.rows.len() as u64),
                Command::Delete(_) | Command::Update(_) => Err(EngineError::Durability(
                    "unresolved text UPDATE/DELETE requires an exact outcome marker".to_string(),
                )),
                _ => Ok(0),
            }
        }
    }

    fn canonical_table_block_count(
        payload: &[u8],
        operation_kind: gpu_db_wal::CanonicalFragmentKind,
    ) -> Result<u32, EngineError> {
        if payload.first() == Some(&WAL_BINARY_TAG) {
            if let crate::wal_binary::BinaryWalRecord::Transaction(record) =
                decode_binary_record(payload)?
            {
                // Opcodes 4--9 predate the ordered statement vector and their acknowledged
                // canonical envelopes counted only reset/mutation output tables. Preserve that
                // exact header interpretation for upgrade replay. Opcodes 10/11 always decode a
                // non-empty operation order and additionally cover catalog-only/private tables.
                let ordered = !record.operation_order.is_empty();
                let tables = record
                    .catalog_commands
                    .iter()
                    .filter_map(|operation| match &operation.command {
                        Command::CreateTable(create) if ordered => Some(create.table.as_str()),
                        _ => None,
                    })
                    .chain(record.operation_order.iter().filter_map(|operation| {
                        if ordered {
                            operation.table()
                        } else {
                            None
                        }
                    }))
                    .chain(record.table_resets.iter().map(|reset| reset.table.as_str()))
                    .chain(record.mutations.iter().map(|mutation| match mutation {
                        crate::wal_binary::BinaryTransactionMutation::Insert { table, .. }
                        | crate::wal_binary::BinaryTransactionMutation::Update { table, .. }
                        | crate::wal_binary::BinaryTransactionMutation::Delete { table, .. } => {
                            table.as_str()
                        }
                    }))
                    .collect::<BTreeSet<_>>();
                return u32::try_from(tables.len()).map_err(|_| {
                    EngineError::Durability(
                        "canonical transaction table-block count exceeds u32".to_string(),
                    )
                });
            }
        }
        Ok(u32::from(matches!(
            operation_kind,
            gpu_db_wal::CanonicalFragmentKind::RowMutation
                | gpu_db_wal::CanonicalFragmentKind::TableReset
                | gpu_db_wal::CanonicalFragmentKind::TableRewrite
        )))
    }

    pub(crate) fn canonical_wal_record_with_commit_request_digest(
        commit: &CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<WalRecord, EngineError> {
        let (catalog_epoch, catalog_digest) =
            Self::canonical_catalog_boundary(commit.canonical_identity, commit.wal.last_record())?;
        Self::canonical_wal_record_with_boundary_and_request_digest(
            commit.canonical_identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_with_boundary_and_request_digest(
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
    ) -> Result<WalRecord, EngineError> {
        let affected_rows = Self::canonical_affected_rows(payload)?;
        Self::canonical_wal_record_with_boundary_and_outcome(
            identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
            gpu_db_wal::CanonicalOutcomeKind::CommitSuccess,
            affected_rows,
        )
    }

    /// Resolve the exact deterministic outcome for the serialized path while its commit lock is
    /// held and before sequence/WAL assignment. The returned delta is intentionally discarded:
    /// apply re-runs against the same protected committed boundary, and recovery compares that
    /// result with this durable marker. A route that cannot be resolved here is refused pre-WAL.
    pub(crate) fn canonical_serialized_outcome(
        &self,
        payload: &[u8],
        commit_seq: Index,
    ) -> Result<(gpu_db_wal::CanonicalOutcomeKind, u64), EngineError> {
        let rows = if payload.first() == Some(&WAL_BINARY_TAG) {
            match decode_binary_record(payload)? {
                crate::wal_binary::BinaryWalRecord::Insert(record) => record.rows.len() as u64,
                crate::wal_binary::BinaryWalRecord::Transaction(record) => {
                    record.mutations.len() as u64
                }
                crate::wal_binary::BinaryWalRecord::DeleteByKey(_)
                | crate::wal_binary::BinaryWalRecord::UpdateByKey(_) => {
                    return Err(EngineError::Durability(
                        "serialized by-key WAL requires an applied GPU outcome".to_string(),
                    ));
                }
            }
        } else {
            let Some(command) = Self::decode_engine_command(payload)? else {
                return Err(EngineError::Durability(
                    "serialized canonical operation is not decodable".to_string(),
                ));
            };
            let snapshot = DmlReadSnapshot {
                commit_seq,
                next_row_id: self.read_state.mvcc.current_row_id(),
            };
            match command {
                Command::Insert(insert) => self
                    .prepare_insert(&insert, snapshot, None, InsertPrepareValidation::Full)?
                    .rows_affected(),
                Command::Delete(delete) => self.prepare_delete(&delete, snapshot)?.rows_affected(),
                Command::Update(update) => self.prepare_update(&update, snapshot)?.rows_affected(),
                _ => 0,
            }
        };
        Ok((
            if rows == 0
                && matches!(
                    Self::decode_engine_command(payload)?,
                    Some(Command::Insert(_) | Command::Delete(_) | Command::Update(_))
                )
            {
                gpu_db_wal::CanonicalOutcomeKind::CommitNoOp
            } else {
                gpu_db_wal::CanonicalOutcomeKind::CommitSuccess
            },
            rows,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_with_commit_outcome(
        commit: &CommitState,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        affected_rows: u64,
    ) -> Result<WalRecord, EngineError> {
        let (catalog_epoch, catalog_digest) =
            Self::canonical_catalog_boundary(commit.canonical_identity, commit.wal.last_record())?;
        Self::canonical_wal_record_with_boundary_and_outcome(
            commit.canonical_identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
            outcome_kind,
            affected_rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_with_boundary_and_outcome(
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        affected_rows: u64,
    ) -> Result<WalRecord, EngineError> {
        Self::canonical_wal_record_with_boundary_and_outcome_isolation(
            identity,
            catalog_epoch,
            catalog_digest,
            txn_id,
            commit_seq,
            lane_id,
            payload,
            request_digest,
            outcome_kind,
            affected_rows,
            gpu_db_wal::CanonicalIsolation::ReadCommitted,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn canonical_wal_record_with_boundary_and_outcome_isolation(
        identity: gpu_db_wal::CanonicalIdentity,
        catalog_epoch: u64,
        catalog_digest: gpu_db_wal::CanonicalDigest,
        txn_id: TxnId,
        commit_seq: Index,
        lane_id: u32,
        payload: &Arc<[u8]>,
        request_digest: gpu_db_wal::CanonicalDigest,
        outcome_kind: gpu_db_wal::CanonicalOutcomeKind,
        affected_rows: u64,
        isolation: gpu_db_wal::CanonicalIsolation,
    ) -> Result<WalRecord, EngineError> {
        if outcome_kind == gpu_db_wal::CanonicalOutcomeKind::AbortError {
            return Err(EngineError::Durability(
                "committed engine WAL cannot be encoded with an abort outcome".to_string(),
            ));
        }
        let operation_body = Self::encode_engine_operation(payload)?;
        let allocator_high_water = Self::canonical_allocator_high_water(payload)?;
        let operation_digest = gpu_db_wal::canonical_request_digest(&operation_body);
        let operation_kind = Self::canonical_fragment_kind(payload)?;
        let operation_fragment = gpu_db_wal::CanonicalFragment {
            kind: operation_kind,
            body: operation_body,
        };
        let status_fragment = gpu_db_wal::CanonicalFragment {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: Self::encode_transaction_claim_status(identity, txn_id, request_digest),
        };
        let catalog_after_epoch =
            if operation_kind == gpu_db_wal::CanonicalFragmentKind::CatalogMutation {
                catalog_epoch.checked_add(1).ok_or_else(|| {
                    EngineError::Durability("canonical catalog epoch overflow".to_string())
                })?
            } else {
                catalog_epoch
            };
        let catalog_after_digest = Self::canonical_catalog_transition(
            catalog_digest,
            operation_kind,
            &operation_fragment.body,
        );
        let header = gpu_db_wal::CanonicalPreApplyHeader {
            identity,
            leader_epoch: 1,
            commit_seq,
            stable_transaction_id: txn_id,
            request_digest,
            isolation,
            flags: u32::from(operation_kind as u16),
            catalog_before_epoch: catalog_epoch,
            catalog_after_epoch,
            catalog_before_digest: catalog_digest,
            catalog_after_digest,
            operation_count: 2,
            table_block_count: Self::canonical_table_block_count(payload, operation_kind)?,
            allocator_high_water,
        };
        let outcome = gpu_db_wal::CanonicalOutcome {
            kind: outcome_kind,
            affected_rows,
            sqlstate: None,
            constraint_id: 0,
            target_digest: operation_digest,
            returning_digest: [0; 32],
        };
        let encoded = gpu_db_wal::encode_canonical_envelope(
            gpu_db_wal::CanonicalPhysicalRange {
                log_epoch: 1,
                lane_id,
                segment_id: commit_seq,
                first_frame_ordinal: 0,
            },
            &header,
            &[operation_fragment, status_fragment],
            &outcome,
        )?;
        Ok(WalRecord {
            txn_id,
            payload: Arc::from(gpu_db_wal::pack_canonical_record_payload(&encoded)?),
        })
    }

    /// Validate every canonical authority before replaying any mutation. Legacy records are
    /// accepted only as an upgrade prefix; the first canonical record binds database lineage.
    pub(crate) fn prepare_durable_records_for_replay(
        &self,
        records: &[WalRecord],
    ) -> Result<Vec<WalRecord>, EngineError> {
        let mut lineage = None;
        let commit = self.commit_state();
        let mut expected_commit_seq = commit.repl.peek_next_index();
        let mut expected_catalog = match commit.wal.last_record() {
            Some(record) => {
                gpu_db_wal::decode_canonical_record_payload(&record.payload)?.map(|envelope| {
                    (
                        envelope.header.identity,
                        envelope.header.catalog_after_epoch,
                        envelope.header.catalog_after_digest,
                    )
                })
            }
            None => None,
        };
        // A prior replay chunk installs every canonical terminal claim before the next chunk is
        // admitted. This makes the one-way migration barrier span checkpoint/serial/lane chunks,
        // without treating a lineage-only identity anchor as evidence that canonical WAL exists.
        let mut canonical_seen = commit.canonical_replay_seen;
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
                replay.push(WalRecord {
                    txn_id: record.txn_id,
                    payload,
                });
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
            replay.push(WalRecord {
                txn_id: record.txn_id,
                payload,
            });
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
            self.replay_validated_durable_record(durable_record, record)?;
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
                }
                std::collections::hash_map::Entry::Occupied(_) => {
                    return Err(EngineError::Durability(format!(
                        "canonical WAL repeats terminal transaction claim {txn_id} across replay chunks"
                    )));
                }
            }
        }
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
                    )))
                }
                Some(_) => {}
                None => {
                    commit.transaction_status.insert(status.txn_id, recovered);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_transaction_reset::table_schema_digest;

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
        let envelope = gpu_db_wal::decode_canonical_record_payload(&record.payload)
            .unwrap()
            .unwrap();
        assert_eq!(envelope.header.table_block_count, 0);
        let recovered = Engine::recover_from_durable_wal(&[record]).unwrap();
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
            operation_order: Vec::new(),
            statement_digests: Vec::new(),
            sequence_input_oids: BTreeMap::new(),
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
        let legacy_envelope = gpu_db_wal::decode_canonical_record_payload(&legacy_record.payload)
            .unwrap()
            .unwrap();
        assert_eq!(legacy_envelope.header.table_block_count, 1);
        let recovered =
            Engine::recover_from_durable_wal(&[prefix_records[0].clone(), legacy_record]).unwrap();
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
        let false_noop = Engine::canonical_wal_record_with_boundary_and_outcome(
            ddl_envelope.header.identity,
            ddl_envelope.header.catalog_after_epoch,
            ddl_envelope.header.catalog_after_digest,
            7052,
            2,
            0,
            &insert,
            gpu_db_wal::canonical_request_digest(&insert),
            gpu_db_wal::CanonicalOutcomeKind::CommitNoOp,
            0,
        )
        .unwrap();
        let error = match Engine::recover_from_durable_wal(&[records[0].clone(), false_noop]) {
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
            try_encode_binary_insert("allocator_t", &[(u64::MAX - 1, values.as_slice())])
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
        let envelope = gpu_db_wal::decode_canonical_record_payload(&record.payload)
            .unwrap()
            .unwrap();
        assert_eq!(envelope.header.allocator_high_water, u64::MAX);

        let invalid: Arc<[u8]> = Arc::from(
            try_encode_binary_insert("allocator_t", &[(u64::MAX, values.as_slice())])
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
