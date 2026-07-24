//! Transaction-private typed table reset.
//!
//! A reset is an ordered empty-root barrier, not a filterless row DELETE. Device preparation
//! resolves both the guarded published root used for durable/replay proof and the exact
//! transaction-visible root being cleared. Neither row set becomes DELETE bodies in the durable
//! transaction record.

mod access;
mod digest;
#[cfg(test)]
#[path = "engine_transaction_reset/regression_tests.rs"]
mod regression_tests;

use digest::{DeviceTableDigestSource, DeviceTableSourceDigest};

use super::*;
use crate::engine_transaction_delta::gpu_accounting::transaction_private_shard_bytes;
use crate::engine_transaction_delta::TransactionGpuReservation;
use crate::table_access::DirectTableReadLease;

#[derive(Clone, Copy)]
struct TableRootProofPolicy {
    cold_authoritative: bool,
    validate_global_cold_root: bool,
    allow_missing_empty_root: bool,
}

/// Result of resolving canonical mutation identity before claiming fresh table access. Exact
/// terminal retries carry their durable affected-row outcome; fresh work carries the requested
/// lease. The acquisition helpers re-resolve after a raced access failure, so another claimant
/// cannot commit the same request in the lookup-to-lease gap and make its peer report stale
/// relation or reset-conflict state.
pub(crate) enum StableRetryOr<T> {
    Terminal(u64),
    Fresh(T),
}

enum StableRetryState {
    Terminal(u64),
    Pending,
    Fresh,
}

impl StagedTableReset {
    pub(crate) fn to_binary(&self) -> BinaryTransactionTableReset {
        BinaryTransactionTableReset {
            ordinal: self.ordinal,
            table: self.table.clone(),
            table_oid: self.table_oid,
            schema_digest: self.schema_digest,
            source_commit_seq: self.source_commit_seq,
            before_digest: self.before_digest,
            expected_rows: self.expected_rows,
            after_empty_digest: self.after_empty_digest,
            dependency_identities: self.dependency_identities.clone(),
        }
    }
}

pub(crate) fn validate_binary_table_reset_source_roots(
    record: &BinaryTransactionRecord,
    ledger: &RecentCommitsLedger,
) -> Result<(), EngineError> {
    if let Some(reset) = record
        .table_resets
        .iter()
        .find(|reset| ledger.table_root_index(&reset.table) != reset.source_commit_seq)
    {
        return Err(EngineError::Durability(format!(
            "transaction table reset source-root index mismatch for \"{}\": WAL names {}, current root is {}",
            reset.table,
            reset.source_commit_seq,
            ledger.table_root_index(&reset.table)
        )));
    }
    Ok(())
}

pub(crate) fn validate_binary_table_reset_source_roots_payload(
    payload: &[u8],
    ledger: &RecentCommitsLedger,
) -> Result<(), EngineError> {
    if let Ok(BinaryWalRecord::Transaction(record)) = decode_binary_record(payload) {
        validate_binary_table_reset_source_roots(&record, ledger)?;
    }
    Ok(())
}

pub(crate) fn is_binary_transaction_payload(payload: &[u8]) -> bool {
    matches!(
        decode_binary_record(payload),
        Ok(BinaryWalRecord::Transaction(_))
    )
}

pub(crate) fn is_single_binary_transaction(entries: &[LogEntry]) -> bool {
    entries.len() == 1 && is_binary_transaction_payload(&entries[0].payload)
}

impl Engine {
    fn stable_retry_state_before_table_access(
        &self,
        txn_id: TxnId,
        request_digest: [u8; 32],
    ) -> Result<StableRetryState, ExecuteError> {
        let commit = self.commit_state();
        if let Some((token, affected_rows)) = commit
            .resolve_transaction_retry_digest_outcome(txn_id, request_digest)
            .map_err(ExecuteError::Engine)?
        {
            if self.committed_seq() < token.index {
                return Err(ExecuteError::Indeterminate(format!(
                    "transaction id {txn_id} has canonical commit sequence {} but publication has not reached it; retry after recovery/publication",
                    token.index
                )));
            }
            return Ok(StableRetryState::Terminal(affected_rows));
        }
        if self
            .resolve_pending_transaction_claim(txn_id, request_digest)
            .map_err(ExecuteError::Engine)?
        {
            return Ok(StableRetryState::Pending);
        }
        drop(commit);
        Ok(StableRetryState::Fresh)
    }

    pub(crate) fn resolve_stable_retry_before_table_access(
        &self,
        txn_id: TxnId,
        request_digest: [u8; 32],
    ) -> Result<Option<u64>, ExecuteError> {
        match self.stable_retry_state_before_table_access(txn_id, request_digest)? {
            StableRetryState::Terminal(affected_rows) => Ok(Some(affected_rows)),
            StableRetryState::Pending => Err(ExecuteError::Indeterminate(format!(
                "transaction id {txn_id} is pending in canonical mutation admission"
            ))),
            StableRetryState::Fresh => Ok(None),
        }
    }

    fn resolve_or_acquire_table_access<T>(
        &self,
        txn_id: TxnId,
        request_digest: [u8; 32],
        acquire: impl FnOnce() -> Result<T, ExecuteError>,
    ) -> Result<StableRetryOr<T>, ExecuteError> {
        if let Some(affected_rows) =
            self.resolve_stable_retry_before_table_access(txn_id, request_digest)?
        {
            return Ok(StableRetryOr::Terminal(affected_rows));
        }
        match acquire() {
            Ok(access) => Ok(StableRetryOr::Fresh(access)),
            Err(access_error) => {
                match self.resolve_stable_retry_before_table_access(txn_id, request_digest)? {
                    Some(affected_rows) => Ok(StableRetryOr::Terminal(affected_rows)),
                    None => Err(access_error),
                }
            }
        }
    }

    fn resolve_or_acquire_table_access_accept_pending<T>(
        &self,
        txn_id: TxnId,
        request_digest: [u8; 32],
        acquire: impl FnOnce() -> Result<T, ExecuteError>,
    ) -> Result<StableRetryOr<T>, ExecuteError> {
        match self.stable_retry_state_before_table_access(txn_id, request_digest)? {
            StableRetryState::Terminal(affected_rows) => {
                return Ok(StableRetryOr::Terminal(affected_rows));
            }
            StableRetryState::Pending => return Ok(StableRetryOr::Terminal(0)),
            StableRetryState::Fresh => {}
        }
        match acquire() {
            Ok(access) => Ok(StableRetryOr::Fresh(access)),
            Err(access_error) => {
                match self.stable_retry_state_before_table_access(txn_id, request_digest)? {
                    StableRetryState::Terminal(affected_rows) => {
                        Ok(StableRetryOr::Terminal(affected_rows))
                    }
                    StableRetryState::Pending => Ok(StableRetryOr::Terminal(0)),
                    StableRetryState::Fresh => Err(access_error),
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn sabotage_retry_access_race(
        &self,
        txn_id: TxnId,
        request_digest: [u8; 32],
        acquire: impl FnOnce() -> Result<(), ExecuteError>,
    ) -> Result<StableRetryOr<()>, ExecuteError> {
        self.resolve_or_acquire_table_access(txn_id, request_digest, acquire)
    }

    pub(crate) fn acquire_autocommit_table_access_after_retry(
        &self,
        table: &str,
        txn_id: TxnId,
        request_digest: [u8; 32],
    ) -> Result<StableRetryOr<Arc<TableAccessLease>>, ExecuteError> {
        self.resolve_or_acquire_table_access(txn_id, request_digest, || {
            self.acquire_autocommit_table_access(table)
        })
    }

    pub(crate) fn acquire_autocommit_command_table_access_after_retry(
        &self,
        command: &Command,
        txn_id: TxnId,
        request_digest: [u8; 32],
    ) -> Result<StableRetryOr<Option<Arc<TableAccessLease>>>, ExecuteError> {
        self.resolve_or_acquire_table_access(txn_id, request_digest, || {
            self.acquire_autocommit_command_table_access(command)
        })
    }

    pub(crate) fn acquire_autocommit_command_table_access_after_retry_or_pending(
        &self,
        command: &Command,
        txn_id: TxnId,
        request_digest: [u8; 32],
    ) -> Result<StableRetryOr<Option<Arc<TableAccessLease>>>, ExecuteError> {
        self.resolve_or_acquire_table_access_accept_pending(txn_id, request_digest, || {
            self.acquire_autocommit_command_table_access(command)
        })
    }

    pub(crate) fn acquire_transaction_table_access(
        &self,
        snapshot: &TransactionSnapshot,
        tables: impl IntoIterator<Item = String>,
    ) -> Result<(), ExecuteError> {
        let catalog = snapshot.transaction_catalog();
        let identities = tables
            .into_iter()
            .filter_map(|name| {
                catalog
                    .relational_catalog
                    .get(&name)
                    .map(|table| (name, table.oid))
            })
            .collect::<BTreeMap<_, _>>();
        self.acquire_transaction_table_access_identities(snapshot, &identities)
    }

    pub(crate) fn acquire_transaction_table_access_identities(
        &self,
        snapshot: &TransactionSnapshot,
        identities: &BTreeMap<String, u32>,
    ) -> Result<(), ExecuteError> {
        // Acquire the lifetime guard before sampling the current fence publication. A reset can
        // therefore occur either wholly before this access or wholly after its owner releases.
        // Deliberately do not compare against the latest catalog here: REPEATABLE READ owns its
        // pinned catalog and retained old OID/root, so later rename/drop/recreate metadata remains
        // invisible. A non-MVCC reset of that old OID is the exception and is detected by the
        // retained rewrite-fence map immediately after this acquisition.
        snapshot
            .table_access
            .acquire_shared(identities.values().copied())?;
        let fences = self.read_state.table_rewrite_fences.load();
        self.mark_transaction_rewrite_fenced_tables(
            snapshot,
            identities
                .iter()
                .filter(|(_, oid)| {
                    fences
                        .get(oid)
                        .is_some_and(|fence| snapshot.boundary < *fence)
                })
                .map(|(name, _)| name.clone()),
        )?;
        Ok(())
    }

    pub(crate) fn acquire_transaction_write_table_access_identities(
        &self,
        snapshot: &TransactionSnapshot,
        identities: &BTreeMap<String, u32>,
    ) -> Result<(), ExecuteError> {
        let latest = self.read_state.latest_catalog();
        if let Some((name, expected_oid)) = identities.iter().find(|(name, expected_oid)| {
            // A relation introduced only by this transaction's private catalog overlay has no
            // published identity to validate yet. Published bindings, including RR-held ones,
            // must still match latest before staging a write.
            snapshot
                .catalog
                .relational_catalog
                .get(name.as_str())
                .is_some_and(|table| table.oid == **expected_oid)
                && latest
                    .relational_catalog
                    .get(name.as_str())
                    .is_none_or(|table| table.oid != **expected_oid)
        }) {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{name}\" changed stable identity from {expected_oid} before transaction write staging"
            )));
        }
        self.acquire_transaction_table_access_identities(snapshot, identities)
    }

    fn mark_transaction_rewrite_fenced_tables(
        &self,
        snapshot: &TransactionSnapshot,
        tables: impl IntoIterator<Item = String>,
    ) -> Result<(), ExecuteError> {
        let tables = tables.into_iter().collect::<BTreeSet<_>>();
        let transaction_catalog = snapshot.transaction_catalog();
        let mut fenced = snapshot
            .rewrite_fenced_tables
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let newly_fenced = tables.difference(&fenced).cloned().collect::<BTreeSet<_>>();
        if newly_fenced.is_empty() {
            return Ok(());
        }
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(table) = delta.operations.iter().find_map(|operation| {
            let table = match operation {
                TransactionOperation::Catalog(_) => return None,
                TransactionOperation::Row(delta) => match &delta.mutation {
                    PreparedMutation::Insert { table, .. }
                    | PreparedMutation::Update { table, .. }
                    | PreparedMutation::Delete { table, .. } => table,
                },
                TransactionOperation::TableReset(reset) => &reset.table,
            };
            newly_fenced.contains(table).then_some(table)
        }) {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{table}\" acquired a rewrite fence after private transaction access"
            )));
        }

        let mut shards = (*delta.resident_shards).clone();
        let mut cold_chunks = (*delta.streaming_cold_chunks).clone();
        let mut gpu_reservation = TransactionGpuReservation::new(self);
        for table_name in &newly_fenced {
            let table = transaction_catalog
                .relational_catalog
                .get(table_name)
                .ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "rewrite-fenced relation \"{table_name}\" left the transaction catalog"
                    ))
                })?;
            shards.insert(table_name.clone(), Vec::new());
            cold_chunks.remove(table_name);
            self.append_transaction_empty_root(table, &mut shards, &mut gpu_reservation)?;
        }
        let next_private_gpu_bytes =
            transaction_private_shard_bytes(snapshot.resident_shards.as_ref(), &shards);
        gpu_reservation.ensure_replacement_admitted(
            &delta.private_gpu_bytes_by_gpu,
            &next_private_gpu_bytes,
        )?;
        delta.publish_resident_shards(Arc::new(shards));
        delta.publish_streaming_cold_chunks(Arc::new(cold_chunks));
        delta.generation = delta.generation.saturating_add(1);
        gpu_reservation
            .replace_charges(&mut delta.private_gpu_bytes_by_gpu, next_private_gpu_bytes);
        fenced.extend(newly_fenced);
        Ok(())
    }

    pub(crate) fn acquire_autocommit_table_access(
        &self,
        table_name: &str,
    ) -> Result<Arc<TableAccessLease>, ExecuteError> {
        self.acquire_autocommit_table_accesses([table_name.to_string()])
    }

    /// Retain one directly-read relation identity without rebuilding its FK dependency closure.
    /// The reset owner already holds every dependency OID exclusively, so a reader of any affected
    /// relation conflicts on that relation's own OID. Prepared point reads call this per batch; the
    /// O(catalog) closure walk belongs to mutations/reset preparation, not the GPU read hot path.
    pub(crate) fn acquire_autocommit_table_read_access(
        &self,
        table_name: &str,
        table_oid: u32,
        boundary: Index,
    ) -> Result<DirectTableReadLease, ExecuteError> {
        let lease = self.table_access.acquire_direct_shared(table_oid)?;
        // The usual path is one relaxed sequence read after binding. If an unrelated publication
        // raced the bind, validate just this stable identity and reset fence while retaining its
        // shared guard. This closes the bind-before-guard window without rebuilding an FK closure
        // or penalizing concurrent writes to other relations.
        if self.committed_seq() != boundary {
            let latest = self.read_state.latest_catalog();
            if latest
                .relational_catalog
                .get(table_name)
                .is_none_or(|table| table.oid != table_oid)
                || self
                    .read_state
                    .table_rewrite_fences
                    .load()
                    .get(&table_oid)
                    .is_some_and(|fence| boundary < *fence)
            {
                return Err(ExecuteError::Serialization(format!(
                    "relation \"{table_name}\" changed stable identity or reset root while its point read was binding"
                )));
            }
        }
        Ok(lease)
    }

    /// Atomically retain the dependency closure of several relation names under one owner. This
    /// is the submission/queue primitive: a deferred operation either owns every required shared
    /// guard or owns none, and dropping its one `Arc` releases the whole closure.
    pub(crate) fn acquire_autocommit_table_accesses(
        &self,
        table_names: impl IntoIterator<Item = String>,
    ) -> Result<Arc<TableAccessLease>, ExecuteError> {
        let table_names = table_names.into_iter().collect::<Vec<_>>();
        let scoped_snapshot = self.current_transaction_read_snapshot();
        let catalog = scoped_snapshot.as_ref().map_or_else(
            || self.read_state.latest_catalog(),
            |snapshot| snapshot.transaction_catalog(),
        );
        let mut identities = BTreeMap::new();
        for table_name in table_names {
            if let Some(table) = catalog.relational_catalog.get(&table_name) {
                identities.extend(table_access_dependency_identities(
                    &catalog.relational_catalog,
                    table,
                )?);
            }
        }
        if let Some(snapshot) = scoped_snapshot {
            // Public probes are also used below explicit/statement-scoped execution. They must be
            // same-owner reentrant there: a fresh autocommit owner would conflict with that
            // transaction's reset upgrade. Return a separately droppable shared subset so a
            // storable proof cannot retain unrelated identities or the parent's exclusive mode.
            self.acquire_transaction_table_access_identities(&snapshot, &identities)?;
            return snapshot
                .table_access
                .retain_shared(identities.values().copied());
        }
        let lease = self.table_access.lease();
        lease.acquire_shared(identities.values().copied())?;
        Ok(lease)
    }

    /// Conflict-history baseline for a table accessed through a historical transaction snapshot.
    /// A non-MVCC reset deliberately retires every version before its fence, so a transaction that
    /// first acquired this table after that fence validates new conflicts from the fence forward,
    /// not from its older unrelated snapshot boundary. Its retained shared lease prevents another
    /// reset from changing this high-water until terminal control.
    pub(crate) fn transaction_table_conflict_boundary(
        &self,
        snapshot: &TransactionSnapshot,
        table_name: &str,
        fallback: Index,
    ) -> Result<Index, ExecuteError> {
        if !snapshot.table_is_rewrite_fenced(table_name) {
            return Ok(fallback);
        }
        let table = snapshot
            .transaction_catalog()
            .relational_catalog
            .get(table_name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "rewrite-fenced relation \"{table_name}\" left the transaction catalog"
                ))
            })?;
        let fences = self.read_state.table_rewrite_fences.load();
        let fence = fences.get(&table.oid).copied().ok_or_else(|| {
            ExecuteError::Serialization(format!(
                "rewrite-fenced relation \"{table_name}\" lost its publication boundary"
            ))
        })?;
        Ok(fallback.max(fence))
    }

    pub(crate) fn command_claims_canonical_mutation(command: &Command) -> bool {
        matches!(
            command,
            Command::SetKv { .. }
                | Command::DeleteKv { .. }
                | Command::SequenceNextVal(_)
                | Command::SequenceSetVal(_)
                | Command::Insert(_)
                | Command::Update(_)
                | Command::Delete(_)
        ) || (Self::command_changes_catalog(command)
            && !matches!(
                command,
                Command::CreateExtension(_) | Command::DropExtension(_)
            ))
    }

    fn command_changes_catalog(command: &Command) -> bool {
        matches!(
            command,
            Command::CreateSchema(_)
                | Command::DropSchema(_)
                | Command::CreateDatabase(_)
                | Command::DropDatabase(_)
                | Command::RenameDatabase(_)
                | Command::CreateTablespace(_)
                | Command::DropTablespace(_)
                | Command::RenameTablespace(_)
                | Command::CreateTable(_)
                | Command::AddPrimaryKey(_)
                | Command::AddUniqueConstraint(_)
                | Command::AddCheckConstraint(_)
                | Command::AddForeignKey(_)
                | Command::AddColumn(_)
                | Command::RenameTable(_)
                | Command::RenameColumn(_)
                | Command::RenameConstraint(_)
                | Command::DropColumn(_)
                | Command::DropConstraint(_)
                | Command::CreateIndex(_)
                | Command::RenameIndex(_)
                | Command::CreateView(_)
                | Command::RenameView(_)
                | Command::CreateMaterializedView(_)
                | Command::RefreshMaterializedView(_)
                | Command::RenameMaterializedView(_)
                | Command::CreateFunction(_)
                | Command::RenameFunction(_)
                | Command::DropFunction(_)
                | Command::CreateExtension(_)
                | Command::DropExtension(_)
                | Command::CreateSequence(_)
                | Command::SequenceRestart(_)
                | Command::CreateDomain(_)
                | Command::RenameSequence(_)
                | Command::DropSequence(_)
                | Command::DropDomain(_)
                | Command::CreatePublication(_)
                | Command::DropPublication(_)
                | Command::CreateSubscription(_)
                | Command::DropSubscription(_)
                | Command::CreateRole(_)
                | Command::DropRole(_)
                | Command::RenameRole(_)
                | Command::GrantTable(_)
                | Command::RevokeTable(_)
                | Command::GrantDatabase(_)
                | Command::RevokeDatabase(_)
                | Command::GrantTablespace(_)
                | Command::RevokeTablespace(_)
                | Command::GrantFunction(_)
                | Command::RevokeFunction(_)
                | Command::GrantSchema(_)
                | Command::RevokeSchema(_)
                | Command::GrantDefaultTablePrivileges(_)
                | Command::RevokeDefaultTablePrivileges(_)
                | Command::DropTable(_)
                | Command::DropIndex(_)
                | Command::DropMaterializedView(_)
                | Command::DropView(_)
                | Command::AlterColumnDefault(_)
                | Command::CommentOn(_)
                | Command::AlterRoleLogin(_)
        )
    }

    /// Guard the low-level compatibility claimant, including binary covered-route records.
    /// Ordinary row bodies share the same dependency-closure discipline as SQL; catalog,
    /// sequence, and table-reset binary bodies are internal typed-transaction artifacts and are
    /// rejected from this independently callable raw claimant.
    pub(crate) fn acquire_raw_mutation_table_access(
        &self,
        payload: &[u8],
    ) -> Result<Option<Arc<TableAccessLease>>, EngineError> {
        if let Some(command) = Self::decode_engine_command(payload)? {
            if matches!(command, Command::TruncateTable(_)) {
                return Err(EngineError::ApplyFailed(
                    "live SQL TRUNCATE must enter typed transaction admission".to_string(),
                ));
            }
            // The crate-private group-commit seam deliberately admits statement-ordered catalog
            // dependencies before any entry is published (for example CREATE SEQUENCE followed by
            // nextval in the same durable flush group). There is no published OID to retain yet.
            // A standalone raw claimant still runs serialized preflight before WAL, while grouped
            // apply resolves the target from its evolving working catalog under the catalog latch.
            let sequence_value_name = match &command {
                Command::SequenceNextVal(nextval) => Some(nextval.name.as_str()),
                Command::SequenceSetVal(setval) => Some(setval.name.as_str()),
                _ => None,
            };
            if let Some(name) = sequence_value_name {
                if self
                    .read_state
                    .latest_catalog()
                    .pg_class_relation_kind(name)?
                    .is_none()
                {
                    return Ok(None);
                }
            }
            return self
                .acquire_autocommit_command_table_access(&command)
                .map_err(|error| EngineError::ApplyFailed(error.to_string()));
        }

        let record = decode_binary_record(payload)?;
        let catalog = self.read_state.latest_catalog();
        let lease = self.table_access.lease();
        let mut shared = BTreeMap::<String, u32>::new();
        let mut add_table = |table_name: &str| -> Result<(), EngineError> {
            if let Some(table) = catalog.relational_catalog.get(table_name) {
                shared.extend(
                    table_access_dependency_identities(&catalog.relational_catalog, table)
                        .map_err(|error| EngineError::ApplyFailed(error.to_string()))?,
                );
            }
            Ok(())
        };
        match record {
            BinaryWalRecord::Insert(record) => add_table(&record.table)?,
            BinaryWalRecord::DeleteByKey(record) => add_table(&record.table)?,
            BinaryWalRecord::UpdateByKey(record) => add_table(&record.table)?,
            BinaryWalRecord::Transaction(record) => {
                if !record.catalog_commands.is_empty()
                    || !record.table_resets.is_empty()
                    || !record.sequence_advances.is_empty()
                    || !record.sequence_advances_by_oid.is_empty()
                {
                    return Err(EngineError::ApplyFailed(
                        "raw binary transactions may contain resolved row mutations only; catalog, sequence, and table-reset effects require typed transaction admission"
                            .to_string(),
                    ));
                }
                for mutation in &record.mutations {
                    let table = match mutation {
                        BinaryTransactionMutation::Insert { table, .. }
                        | BinaryTransactionMutation::Update { table, .. }
                        | BinaryTransactionMutation::Delete { table, .. } => table,
                    };
                    add_table(table)?;
                }
            }
        }
        lease
            .acquire_shared(shared.values().copied())
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        Ok((!shared.is_empty()).then_some(lease))
    }

    pub(crate) fn validate_transaction_table_resets(
        &self,
        catalog: &CatalogSnapshot,
        resets: &[StagedTableReset],
        ledger: &RecentCommitsLedger,
    ) -> Result<(), ExecuteError> {
        for reset in resets {
            let table = catalog
                .relational_catalog
                .get(&reset.table)
                .ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "table reset target \"{}\" left the transaction catalog",
                        reset.table
                    ))
                })?;
            if table.oid != reset.table_oid || table_schema_digest(table)? != reset.schema_digest {
                return Err(ExecuteError::Serialization(format!(
                    "table reset target \"{}\" changed catalog identity",
                    reset.table
                )));
            }
            if table_access_dependency_identities(&catalog.relational_catalog, table)?
                != reset.dependency_identities
            {
                return Err(ExecuteError::Serialization(format!(
                    "table reset dependency closure for \"{}\" changed after snapshot {}",
                    reset.table, reset.read_snapshot
                )));
            }
            for (name, expected) in &reset.catalog_dependencies {
                if catalog.relational_catalog.get(name) != Some(expected) {
                    return Err(ExecuteError::Serialization(format!(
                        "table reset catalog dependency \"{name}\" changed after snapshot {}",
                        reset.read_snapshot
                    )));
                }
            }
            if let Some(name) = reset
                .dependency_identities
                .keys()
                .find(|name| ledger.table_changed_after(name, reset.read_snapshot))
            {
                return Err(ExecuteError::Serialization(format!(
                    "table reset dependency relation \"{name}\" changed after snapshot {}",
                    reset.read_snapshot
                )));
            }
            let current_root = ledger.table_root_index(&reset.table);
            if current_root != reset.source_commit_seq {
                return Err(ExecuteError::Serialization(format!(
                    "table reset source root for \"{}\" changed from {} to {}",
                    reset.table, reset.source_commit_seq, current_root
                )));
            }
            let (visible_rows, digest) = self.table_reset_device_root_proof(
                table,
                reset.source_commit_seq,
                self.committed_seq(),
            )?;
            if visible_rows != reset.expected_rows || digest != reset.before_digest {
                return Err(ExecuteError::Serialization(format!(
                    "table reset source root proof for \"{}\" changed before commit",
                    reset.table
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn execute_truncate_in_transaction(
        &self,
        txn_id: TxnId,
        truncate: TruncateTable,
        expected_catalog_version: Option<u64>,
    ) -> Result<(), ExecuteError> {
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        let snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, &snapshot)?;
        let catalog = snapshot.transaction_catalog();
        if let Some(expected) = expected_catalog_version {
            let actual_table = catalog.relational_catalog.get(&truncate.name);
            let transaction_private_target = !snapshot
                .catalog
                .relational_catalog
                .contains_key(&truncate.name)
                && snapshot
                    .delta
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .operations
                    .iter()
                    .any(|operation| {
                        matches!(
                            operation,
                            TransactionOperation::Catalog(staged)
                                if matches!(&staged.command, Command::CreateTable(create) if create.table == truncate.name)
                        )
                    });
            let expected_table = (!transaction_private_target)
                .then(|| self.read_state.catalog_as_of(expected))
                .and_then(|expected_catalog| {
                    expected_catalog
                        .relational_catalog
                        .get(&truncate.name)
                        .cloned()
                });
            if !transaction_private_target && expected_table.as_ref() != actual_table {
                return Err(ExecuteError::Unsupported(
                    "prepared TRUNCATE catalog identity changed at the transaction statement snapshot; re-Parse is required"
                        .to_string(),
                ));
            }
        }
        self.stage_table_reset_statement_locked(txn_id, &snapshot, truncate)
    }

    /// Autocommit TRUNCATE is a one-statement explicit transaction internally. This keeps the
    /// product/raw compatibility surfaces on the exact same typed root-reset WAL, lifetime guard,
    /// fence publication, and recovery path as user-declared transactions.
    pub(crate) fn execute_truncate_autocommit(
        &self,
        txn_id: TxnId,
        truncate: TruncateTable,
        expected_catalog_version: Option<u64>,
    ) -> Result<(), ExecuteError> {
        let request_digest = table_reset_request_digest(&truncate, expected_catalog_version);
        {
            let commit = self.commit_state();
            match commit.resolve_transaction_retry_digest_outcome(txn_id, request_digest) {
                Ok(Some((token, _))) if self.committed_seq() >= token.index => return Ok(()),
                Ok(Some((token, _))) => {
                    return Err(ExecuteError::Indeterminate(format!(
                        "TRUNCATE transaction {txn_id} has canonical commit sequence {} but publication has not reached it",
                        token.index
                    )))
                }
                Err(error) => return Err(ExecuteError::Engine(error)),
                Ok(None) => {}
            }
        }
        match self.reserve_pending_transaction_claim(txn_id, request_digest) {
            Ok(true) => {}
            Ok(false) => {
                return Err(ExecuteError::Indeterminate(format!(
                    "TRUNCATE transaction {txn_id} is pending in canonical mutation admission"
                )))
            }
            Err(error) => return Err(ExecuteError::Engine(error)),
        }
        if let Err(error) = self.begin_claimed_transaction_context(
            txn_id,
            TransactionCharacteristics::READ_COMMITTED_READ_WRITE,
            request_digest,
        ) {
            self.release_pending_transaction_claim(txn_id, request_digest);
            let commit = self.commit_state();
            return match commit.resolve_transaction_retry_digest_outcome(txn_id, request_digest) {
                Ok(Some((token, _))) if self.committed_seq() >= token.index => Ok(()),
                Ok(Some((token, _))) => Err(ExecuteError::Indeterminate(format!(
                    "TRUNCATE transaction {txn_id} has canonical commit sequence {} but publication has not reached it",
                    token.index
                ))),
                Err(retry_error) => Err(ExecuteError::Engine(retry_error)),
                Ok(None) => Err(error),
            };
        }
        if let Err(error) =
            self.execute_truncate_in_transaction(txn_id, truncate, expected_catalog_version)
        {
            let cleanup = self.cancel_internal_transaction_context(txn_id);
            self.release_pending_transaction_claim(txn_id, request_digest);
            if cleanup.is_err() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "autocommit TRUNCATE staging failed ({error}); its internal transaction could not be released"
                ))));
            }
            return Err(error);
        }
        match self.commit_claimed_transaction_delta(
            txn_id,
            request_digest,
            current_timestamp_micros(),
        ) {
            Ok(()) => Ok(()),
            Err(error) => {
                // Pre-durable rejection leaves the internal transaction active; release its
                // exclusive lease. A wedged/post-durable failure deliberately preserves state for
                // restart recovery and must not be rewritten into a rollback.
                if !self.is_commit_path_poisoned()
                    && self.transaction_snapshot_handle(txn_id).is_some()
                {
                    let _ = self.cancel_internal_transaction_context(txn_id);
                    self.release_pending_transaction_claim(txn_id, request_digest);
                }
                Err(error)
            }
        }
    }

    fn stage_table_reset_statement_locked(
        &self,
        txn_id: TxnId,
        snapshot: &Arc<TransactionSnapshot>,
        truncate: TruncateTable,
    ) -> Result<(), ExecuteError> {
        self.legacy_lane_history_write_guard()
            .map_err(ExecuteError::Engine)?;
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        if snapshot.characteristics.access == TransactionAccessMode::ReadOnly {
            return Err(ExecuteError::Unsupported(
                "cannot execute TRUNCATE in a READ ONLY transaction".to_string(),
            ));
        }
        let catalog = snapshot.transaction_catalog();
        if catalog.relational_views.contains_key(&truncate.name)
            || catalog
                .relational_materialized_views
                .contains_key(&truncate.name)
            || catalog.relational_sequences.contains_key(&truncate.name)
        {
            return Err(ExecuteError::Unsupported(format!(
                "relation \"{}\" is not a table",
                truncate.name
            )));
        }
        let table = catalog
            .relational_catalog
            .get(&truncate.name)
            .cloned()
            .ok_or_else(|| ExecuteError::UndefinedRelation(truncate.name.clone()))?;
        let transaction_private = !snapshot
            .catalog
            .relational_catalog
            .contains_key(&truncate.name);
        if transaction_private {
            let created_before_reset = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .operations
                .iter()
                .any(|operation| {
                    matches!(
                        operation,
                        TransactionOperation::Catalog(staged)
                            if matches!(&staged.command, Command::CreateTable(create) if create.table == truncate.name)
                    )
                });
            if !created_before_reset {
                return Err(ExecuteError::Serialization(format!(
                    "transaction-private reset target \"{}\" has no ordered CREATE operation",
                    truncate.name
                )));
            }
        }
        if let Some((child, foreign_key)) = catalog.relational_catalog.values().find_map(|child| {
            child
                .foreign_keys
                .iter()
                .find(|foreign_key| foreign_key.referenced_table == table.name)
                .map(|foreign_key| (child, foreign_key))
        }) {
            return Err(ExecuteError::Unsupported(format!(
                "cannot truncate relation \"{}\" because constraint \"{}\" on relation \"{}\" references it; multi-table TRUNCATE/CASCADE is not implemented",
                table.name, foreign_key.name, child.name
            )));
        }
        let dependency_identities =
            table_access_dependency_identities(&catalog.relational_catalog, &table)?;
        snapshot
            .table_access
            .acquire_exclusive(dependency_identities.values().copied())?;
        if truncate.restart_identity {
            let owned_sequence_oids = table
                .columns
                .iter()
                .filter_map(|column| match &column.default {
                    Some(ColumnDefault::SequenceNextVal {
                        sequence,
                        create_if_missing: true,
                    }) => catalog
                        .relational_sequences
                        .get(sequence)
                        .map(|sequence| sequence.oid),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            snapshot
                .table_access
                .acquire_exclusive(owned_sequence_oids)?;
        }

        // The durable proof describes the globally published root immediately before this reset,
        // not the transaction-private root. For INSERT;TRUNCATE, for example, the private INSERT
        // is shadowed and intentionally absent from WAL, so including it in the before proof would
        // make live apply and replay compare against rows that can never exist globally. The
        // exclusive stable-OID guard freezes this published root through transaction termination.
        let published_catalog = self.read_state.latest_catalog();
        if transaction_private {
            if published_catalog
                .relational_catalog
                .contains_key(&table.name)
            {
                return Err(ExecuteError::Serialization(format!(
                    "transaction-private reset target \"{}\" collided with a published relation",
                    table.name
                )));
            }
        } else {
            if published_catalog.relational_catalog.get(&table.name) != Some(&table) {
                return Err(ExecuteError::Serialization(format!(
                    "table reset target \"{}\" changed catalog identity before staging",
                    table.name
                )));
            }
            if table_access_dependency_identities(&published_catalog.relational_catalog, &table)?
                != dependency_identities
            {
                return Err(ExecuteError::Serialization(format!(
                    "table reset dependency closure for \"{}\" changed before staging",
                    table.name
                )));
            }
        }
        // The exclusive stable-OID lease freezes this globally published root while we sample its
        // canonical publication identity and GPU-visible cardinality. No row image or row key is
        // gathered to the host: the proof remains constant-size for every table cardinality.
        let (source_commit_seq, expected_rows, before_digest) = {
            let commit = self
                .commit_state_after_wave_quiescence()
                .map_err(ExecuteError::Engine)?;
            let source_commit_seq = if transaction_private {
                if commit.ledger.table_root_index(&table.name) != 0 {
                    return Err(ExecuteError::Serialization(format!(
                        "transaction-private reset target \"{}\" collided with prior table-root history",
                        table.name
                    )));
                }
                0
            } else {
                commit.ledger.table_root_index(&table.name)
            };
            let (expected_rows, before_digest) = self.table_reset_device_root_proof(
                &table,
                source_commit_seq,
                self.committed_seq(),
            )?;
            (source_commit_seq, expected_rows, before_digest)
        };

        let (generation, operation_count) = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (delta.generation, delta.operations.len())
        };
        let ordinal = u32::try_from(operation_count).map_err(|_| {
            ExecuteError::Unsupported(
                "transaction operation ordinal exceeds typed reset framing".to_string(),
            )
        })?;
        let (next_catalog_overlay, sequence_reset_identity) = if truncate.restart_identity {
            let (overlay, identity) =
                Self::transaction_sequence_reset_overlay(ordinal, &table.name, &catalog)?;
            (Some(overlay), Some(identity))
        } else {
            (None, None)
        };
        // Independently resolve the transaction-visible root. This proves the retained private GPU
        // generation can actually be reset, while its pre-reset private rows remain statement-local
        // and do not contaminate the durable published-root proof above.
        let current_shards = snapshot.transaction_shards();
        let current_cold_chunks = snapshot.transaction_cold_chunks();
        let _private_visible_rows = self
            .table_device_visible_source_from_roots(
                &table,
                snapshot.boundary,
                &current_shards,
                &current_cold_chunks,
                None,
                TableRootProofPolicy {
                    cold_authoritative: snapshot
                        .chunk_authoritative_tables
                        .contains_key(&table.name)
                        && !snapshot.table_is_rewrite_fenced(&table.name)
                        && !snapshot.transaction_table_is_reset(&table.name),
                    validate_global_cold_root: false,
                    allow_missing_empty_root: source_commit_seq == 0,
                },
            )?
            .visible_rows;
        let _scope = self.enter_transaction_read(Arc::clone(snapshot));
        self.validate_transaction_delta_residency(&table, !transaction_private)?;
        let schema_digest = table_schema_digest(&table)?;
        let after_empty_digest = table_reset_empty_digest(table.oid, schema_digest);

        let mut next_shards = (*current_shards).clone();
        next_shards.insert(table.name.clone(), Vec::new());
        let mut next_cold_chunks = (*current_cold_chunks).clone();
        next_cold_chunks.remove(&table.name);
        let mut gpu_reservation = TransactionGpuReservation::new(self);
        self.append_transaction_empty_root(&table, &mut next_shards, &mut gpu_reservation)?;
        let next_private_gpu_bytes =
            transaction_private_shard_bytes(snapshot.resident_shards.as_ref(), &next_shards);

        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if delta.generation != generation {
            return Err(ExecuteError::Serialization(
                "concurrent statements attempted to publish the same transaction reset".to_string(),
            ));
        }
        if delta.operations.len() != operation_count {
            return Err(ExecuteError::Serialization(
                "transaction operation order changed during table-reset staging".to_string(),
            ));
        }
        gpu_reservation.ensure_replacement_admitted(
            &delta.private_gpu_bytes_by_gpu,
            &next_private_gpu_bytes,
        )?;
        let statement_digest =
            transaction_statement_digest(&Command::TruncateTable(truncate.clone()))?;
        delta
            .operations
            .push(TransactionOperation::TableReset(Arc::new(
                StagedTableReset {
                    ordinal,
                    statement_digest,
                    table: table.name.clone(),
                    table_oid: table.oid,
                    schema_digest,
                    source_commit_seq,
                    before_digest,
                    after_empty_digest,
                    expected_rows,
                    read_snapshot: snapshot.boundary,
                    catalog_dependencies: dependency_identities
                        .keys()
                        .filter_map(|name| {
                            (if transaction_private {
                                catalog.as_ref()
                            } else {
                                published_catalog.as_ref()
                            })
                            .relational_catalog
                            .get(name)
                            .cloned()
                            .map(|dependency| (name.clone(), dependency))
                        })
                        .collect(),
                    foreign_key_dependencies: dependency_identities
                        .keys()
                        .filter(|name| *name != &table.name)
                        .cloned()
                        .collect(),
                    dependency_identities,
                    sequence_reset_identity: sequence_reset_identity.clone(),
                },
            )));
        if let Some(overlay) = next_catalog_overlay {
            if delta.catalog_base.is_none() {
                delta.catalog_base = Some(Arc::clone(&snapshot.catalog));
            }
            for target in &sequence_reset_identity
                .as_ref()
                .expect("restart overlay has a reset identity")
                .targets
            {
                let sequence_oid = target
                    .target_after
                    .as_ref()
                    .or(target.target_before.as_ref())
                    .expect("reset target retains one stable sequence identity")
                    .oid;
                delta
                    .sequence_state
                    .insert(target.before_name.clone(), (1, false));
                delta.sequence_state_by_oid.insert(sequence_oid, (1, false));
            }
            delta.catalog_overlay = Some(overlay);
        }
        delta.write_set = final_transaction_write_set(&delta.operations);
        delta.publish_resident_shards(Arc::new(next_shards));
        delta.publish_streaming_cold_chunks(Arc::new(next_cold_chunks));
        delta.generation = delta.generation.saturating_add(1);
        drop(current_shards);
        drop(current_cold_chunks);
        gpu_reservation
            .replace_charges(&mut delta.private_gpu_bytes_by_gpu, next_private_gpu_bytes);
        Ok(())
    }
}

impl Engine {
    /// Build the replay-stable proof for the current globally published root. The logical source
    /// identity is its last canonical table publication; cardinality is reduced on the GPU from
    /// the exact resident/cold generation and its MVCC sidecars.
    pub(crate) fn table_reset_device_root_proof(
        &self,
        table: &RelationalTable,
        source_commit_seq: Index,
        boundary: Index,
    ) -> Result<(u64, gpu_db_wal::CanonicalDigest), ExecuteError> {
        // This proof binds the globally published before-root named by `source_commit_seq`, never
        // the caller's transaction-private overlay. Contextual read helpers deliberately substitute
        // private reset/fence roots and would make commit validation prove the after-image instead.
        let shards = self.read_state.residency.shards.load_full();
        let cold_chunks = self.read_state.residency.streaming_cold_chunks.load_full();
        let single = self
            .read_state
            .residency
            .snapshots
            .load_full()
            .get(&table.name)
            .cloned();
        let chunk_authoritative = self
            .read_state
            .residency
            .chunk_authoritative_tables
            .load()
            .contains_key(&table.name);
        let device_authoritative = self
            .read_state
            .residency
            .device_authoritative_tables
            .load()
            .contains(&table.name);
        let has_hot_root = shards
            .get(&table.name)
            .is_some_and(|table_shards| !table_shards.is_empty())
            || single.is_some();
        let cold_authoritative = chunk_authoritative
            || (!device_authoritative && !has_hot_root && cold_chunks.contains_key(&table.name));
        let source = self.table_device_visible_source_from_roots(
            table,
            boundary,
            &shards,
            &cold_chunks,
            single.as_ref(),
            TableRootProofPolicy {
                cold_authoritative,
                validate_global_cold_root: true,
                allow_missing_empty_root: source_commit_seq == 0,
            },
        )?;
        let rows = source.visible_rows;
        let schema_digest = table_schema_digest(table)?;
        Ok((
            rows,
            table_reset_root_digest(
                table.oid,
                schema_digest,
                source_commit_seq,
                rows,
                source.lanes,
            ),
        ))
    }

    fn table_device_visible_source_from_roots(
        &self,
        table: &RelationalTable,
        boundary: Index,
        shards: &BTreeMap<String, Vec<RelationalResidentShard>>,
        cold_chunks: &BTreeMap<String, Arc<crate::engine_streaming_exec::ColdTableChunks>>,
        single: Option<&RelationalResidencyEntry>,
        policy: TableRootProofPolicy,
    ) -> Result<DeviceTableSourceDigest, ExecuteError> {
        let table_name = table.name.as_str();
        let boundary = i64::try_from(boundary).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "MVCC boundary exceeds the signed device visibility domain".to_string(),
            ))
        })?;
        let map_cuda = |error: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "device root proof for relation \"{table_name}\" failed: {error}"
            )))
        };

        if policy.cold_authoritative {
            let cold = cold_chunks.get(table_name).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "chunk-authoritative relation \"{table_name}\" has no cold device root for reset proof"
                )))
            })?;
            if policy.validate_global_cold_root
                && !self.global_cold_root_matches_reset_boundary(table, cold, boundary as u64)
            {
                return Err(ExecuteError::Serialization(format!(
                    "chunk-authoritative relation \"{table_name}\" has a stale cold root at reset boundary {boundary}"
                )));
            }
            let mut total = DeviceTableSourceDigest::default();
            for chunk in &cold.chunks {
                if chunk.payload_copin_s > boundary as u64 {
                    continue;
                }
                let (source, visibility) = self
                    .stage_cold_chunk(chunk, boundary as u64)
                    .and_then(|staged| staged.ready().map_err(|_| ()))
                    .map_err(|()| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "cold device root for relation \"{table_name}\" could not be staged"
                        )))
                    })?;
                let header_rows = source
                    .device_memory
                    .count_rows_from_header()
                    .map_err(map_cuda)?;
                if header_rows != source.row_count {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "cold device root header for relation \"{table_name}\" reports {header_rows} rows but its descriptor reports {}",
                        source.row_count
                    ))));
                }
                let deleted = visibility
                    .and_then(|value| value.deleted_by_offset)
                    .map(|offset| (source.device_memory.as_ref(), offset));
                let created = visibility
                    .and_then(|value| value.created_by_offset)
                    .map(|offset| (source.device_memory.as_ref(), offset));
                if chunk.entity_ids.len() != source.row_count as usize {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "cold device root for relation \"{table_name}\" has no exact stable row-identity vector"
                    ))));
                }
                let row_ids = (source.row_count != 0)
                    .then(|| {
                        self.upload_table_reset_entity_ids(
                            source.descriptor.gpu_id,
                            chunk.entity_ids.as_slice(),
                        )
                    })
                    .transpose()?;
                let digest = self.table_reset_digest_source(
                    table,
                    DeviceTableDigestSource {
                        snapshot: source.descriptor.as_ref(),
                        memory: source.device_memory.as_ref(),
                        row_count: source.row_count,
                        row_ids: row_ids.as_ref().map(|memory| (memory, 0)),
                        boundary,
                        deleted_by: deleted,
                        created_by: created,
                    },
                )?;
                total.combine(digest);
            }
            return Ok(total);
        }

        if let Some(table_shards) = shards.get(table_name) {
            if table_shards.is_empty() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table_name}\" has an unallocated empty shard placeholder, not an authoritative device root"
                ))));
            }
            let pressured = self.router.runtime().snapshot().memory_pressured_gpu_ids;
            let mut total = DeviceTableSourceDigest::default();
            for shard in table_shards {
                if !shard.is_valid(pressured.contains(&shard.gpu_id)) {
                    return Err(ExecuteError::Serialization(format!(
                        "resident root for relation \"{table_name}\" changed during reset proof"
                    )));
                }
                let memory = shard.device_memory.as_ref().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident root for relation \"{table_name}\" has no device allocation"
                    )))
                })?;
                let header_rows = memory.count_rows_from_header().map_err(map_cuda)?;
                if header_rows != shard.row_count as u64 {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident root header for relation \"{table_name}\" reports {header_rows} rows but its descriptor reports {}",
                        shard.row_count
                    ))));
                }
                let descriptor = self.resident_snapshot_for_shard(shard, table);
                let digest = self.table_reset_digest_source(
                    table,
                    DeviceTableDigestSource {
                        snapshot: &descriptor,
                        memory,
                        row_count: shard.row_count as u64,
                        row_ids: shard
                            .row_id_region
                            .as_ref()
                            .map(|region| (region.as_ref(), 0)),
                        boundary,
                        deleted_by: shard
                            .deleted_by_region
                            .as_ref()
                            .map(|region| (region.as_ref(), 0)),
                        created_by: shard
                            .created_by_region
                            .as_ref()
                            .map(|region| (region.as_ref(), 0)),
                    },
                )?;
                total.combine(digest);
            }
            return Ok(total);
        }

        if let Some(entry) = single {
            if !entry.descriptor.is_valid() {
                return Err(ExecuteError::Serialization(format!(
                    "single-buffer root for relation \"{table_name}\" changed during reset proof"
                )));
            }
            let memory = entry.device_memory.as_ref().ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "single-buffer root for relation \"{table_name}\" has no device allocation"
                )))
            })?;
            let header_rows = memory.count_rows_from_header().map_err(map_cuda)?;
            if header_rows != entry.descriptor.row_count as u64 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "single-buffer root header for relation \"{table_name}\" reports {header_rows} rows but its descriptor reports {}",
                    entry.descriptor.row_count
                ))));
            }
            let sidecar_key = (table_name.to_string(), 0);
            let row_ids = self
                .read_state
                .residency
                .shard_row_id_memory
                .get(&sidecar_key);
            if entry.descriptor.row_count != 0 && row_ids.is_none() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "single-buffer root for relation \"{table_name}\" has no device stable-row-identity region"
                ))));
            }
            let deleted_by = self
                .read_state
                .residency
                .shard_deleted_by_memory
                .get(&sidecar_key);
            let created_by = self
                .read_state
                .residency
                .shard_created_by_memory
                .get(&sidecar_key);
            return self.table_reset_digest_source(
                table,
                DeviceTableDigestSource {
                    snapshot: entry.descriptor.as_ref(),
                    memory,
                    row_count: entry.descriptor.row_count as u64,
                    row_ids: row_ids.as_ref().map(|region| (region.as_ref(), 0)),
                    boundary,
                    deleted_by: deleted_by.as_ref().map(|region| (region.as_ref(), 0)),
                    created_by: created_by.as_ref().map(|region| (region.as_ref(), 0)),
                },
            );
        }

        if policy.allow_missing_empty_root {
            let names = table
                .columns
                .iter()
                .map(|column| column.name.clone())
                .collect::<Vec<_>>();
            let types = table
                .columns
                .iter()
                .map(|column| column.ty)
                .collect::<Vec<_>>();
            let (payload, ..) =
                crate::engine_residency::build_relational_device_payload(&names, &types, &[])?;
            let memory = self
                .relational_residency_device_memory(self.planner.default_gpu_id(), &payload)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "zero-row GPU bootstrap for relation \"{table_name}\" failed"
                    )))
                })?;
            let header_rows = memory.count_rows_from_header().map_err(map_cuda)?;
            if header_rows != 0 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "zero-row GPU bootstrap for relation \"{table_name}\" reported {header_rows} rows"
                ))));
            }
            return Ok(DeviceTableSourceDigest::default());
        }

        Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "relation \"{table_name}\" has no authoritative device root for reset proof"
        ))))
    }
}

fn table_reset_request_digest(
    truncate: &TruncateTable,
    expected_catalog_version: Option<u64>,
) -> gpu_db_wal::CanonicalDigest {
    let mut body = Vec::with_capacity(48 + truncate.name.len());
    body.extend_from_slice(b"GPUDBTRUNCATEREQUEST1");
    body.extend_from_slice(&(truncate.name.len() as u64).to_le_bytes());
    body.extend_from_slice(truncate.name.as_bytes());
    body.push(u8::from(truncate.restart_identity));
    match expected_catalog_version {
        Some(version) => {
            body.push(1);
            body.extend_from_slice(&version.to_le_bytes());
        }
        None => body.push(0),
    }
    gpu_db_wal::canonical_request_digest(&body)
}

pub(crate) fn table_schema_digest(
    table: &RelationalTable,
) -> Result<gpu_db_wal::CanonicalDigest, ExecuteError> {
    // This is a durable v1 encoding, not a derived serializer. Field and variant tags stay
    // explicit so adding a Rust field or changing serde representation cannot silently make an
    // already-acknowledged reset unreplayable after an upgrade.
    let mut body = Vec::new();
    body.extend_from_slice(b"GPUDBTABLESCHEMA1");
    digest_push_string(&mut body, &table.schema)?;
    digest_push_string(&mut body, &table.name)?;
    body.extend_from_slice(&table.oid.to_le_bytes());
    digest_push_count(&mut body, table.columns.len())?;
    for column in &table.columns {
        body.extend_from_slice(&column.id.to_le_bytes());
        body.extend_from_slice(&column.table_oid.to_le_bytes());
        body.extend_from_slice(&column.attnum.to_le_bytes());
        digest_push_string(&mut body, &column.name)?;
        digest_push_sql_type(&mut body, column.ty);
        match &column.domain {
            Some(domain) => {
                body.push(1);
                digest_push_string(&mut body, domain)?;
            }
            None => body.push(0),
        }
        match &column.default {
            Some(ColumnDefault::Literal(value)) => {
                body.push(1);
                digest_push_value(&mut body, value)?;
            }
            Some(ColumnDefault::SequenceNextVal {
                sequence,
                create_if_missing,
            }) => {
                body.push(2);
                digest_push_string(&mut body, sequence)?;
                body.push(u8::from(*create_if_missing));
            }
            None => body.push(0),
        }
        body.extend_from_slice(&column.type_oid.to_le_bytes());
        body.extend_from_slice(&column.type_size.to_le_bytes());
    }
    digest_push_count(&mut body, table.indexes.len())?;
    for index in &table.indexes {
        digest_push_string(&mut body, &index.name)?;
        digest_push_string(&mut body, &index.table)?;
        digest_push_string(&mut body, &index.column)?;
        digest_push_count(&mut body, index.key_columns.len())?;
        for column in &index.key_columns {
            digest_push_string(&mut body, column)?;
        }
        body.extend_from_slice(&[
            u8::from(index.unique),
            u8::from(index.primary_key),
            u8::from(index.unique_constraint),
        ]);
    }
    digest_push_count(&mut body, table.check_constraints.len())?;
    for constraint in &table.check_constraints {
        digest_push_string(&mut body, &constraint.name)?;
        digest_push_string(&mut body, &constraint.column)?;
        body.push(match constraint.op {
            SelectFilterOp::Eq => 0,
            SelectFilterOp::Lt => 1,
            SelectFilterOp::Lte => 2,
            SelectFilterOp::Gt => 3,
            SelectFilterOp::Gte => 4,
            SelectFilterOp::LikePrefix => 5,
        });
        digest_push_value(&mut body, &constraint.value)?;
    }
    digest_push_count(&mut body, table.foreign_keys.len())?;
    for foreign_key in &table.foreign_keys {
        digest_push_string(&mut body, &foreign_key.name)?;
        digest_push_string(&mut body, &foreign_key.column)?;
        digest_push_string(&mut body, &foreign_key.referenced_table)?;
        digest_push_string(&mut body, &foreign_key.referenced_column)?;
    }
    digest_push_count(&mut body, table.acl.len())?;
    for (role, privileges) in &table.acl {
        digest_push_string(&mut body, role)?;
        digest_push_count(&mut body, privileges.len())?;
        for privilege in privileges {
            body.push(match privilege {
                TablePrivilege::Select => 0,
                TablePrivilege::Insert => 1,
                TablePrivilege::Update => 2,
                TablePrivilege::Delete => 3,
            });
        }
    }
    Ok(gpu_db_wal::canonical_request_digest(&body))
}

fn digest_push_count(body: &mut Vec<u8>, count: usize) -> Result<(), ExecuteError> {
    let count = u32::try_from(count).map_err(|_| {
        ExecuteError::Unsupported("table schema digest collection exceeds u32".to_string())
    })?;
    body.extend_from_slice(&count.to_le_bytes());
    Ok(())
}

fn digest_push_string(body: &mut Vec<u8>, value: &str) -> Result<(), ExecuteError> {
    digest_push_count(body, value.len())?;
    body.extend_from_slice(value.as_bytes());
    Ok(())
}

fn digest_push_value(body: &mut Vec<u8>, value: &SqlValue) -> Result<(), ExecuteError> {
    digest_push_string(body, &encode_relational_row(std::slice::from_ref(value)))
}

fn digest_push_sql_type(body: &mut Vec<u8>, ty: SqlType) {
    match ty {
        SqlType::Int2 => body.push(0),
        SqlType::Int4 => body.push(1),
        SqlType::Int8 => body.push(2),
        SqlType::Numeric { precision, scale } => {
            body.extend_from_slice(&[3, precision, scale]);
        }
        SqlType::Bool => body.push(4),
        SqlType::Text => body.push(5),
        SqlType::Date => body.push(6),
        SqlType::Timestamp => body.push(7),
        SqlType::Uuid => body.push(8),
    }
}

pub(crate) fn table_reset_root_digest(
    table_oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
    source_commit_seq: Index,
    visible_rows: u64,
    source_lanes: [u64; 4],
) -> gpu_db_wal::CanonicalDigest {
    let mut body = Vec::with_capacity(112);
    body.extend_from_slice(b"GPUDBTABLERESETROOT3");
    body.extend_from_slice(&table_oid.to_le_bytes());
    body.extend_from_slice(&schema_digest);
    body.extend_from_slice(&source_commit_seq.to_le_bytes());
    body.extend_from_slice(&visible_rows.to_le_bytes());
    for lane in source_lanes {
        body.extend_from_slice(&lane.to_le_bytes());
    }
    gpu_db_wal::canonical_request_digest(&body)
}

pub(crate) fn table_reset_empty_digest(
    table_oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
) -> gpu_db_wal::CanonicalDigest {
    let mut body = Vec::with_capacity(64);
    body.extend_from_slice(b"GPUDBTABLERESETEMPTY1");
    body.extend_from_slice(&table_oid.to_le_bytes());
    body.extend_from_slice(&schema_digest);
    gpu_db_wal::canonical_request_digest(&body)
}

pub(crate) fn table_access_dependency_identities(
    catalog: &BTreeMap<String, RelationalTable>,
    table: &RelationalTable,
) -> Result<BTreeMap<String, u32>, ExecuteError> {
    let mut names = BTreeSet::from([table.name.clone()]);
    names.extend(
        table
            .foreign_keys
            .iter()
            .map(|foreign_key| foreign_key.referenced_table.clone()),
    );
    for candidate in catalog.values() {
        if candidate
            .foreign_keys
            .iter()
            .any(|foreign_key| foreign_key.referenced_table == table.name)
        {
            names.insert(candidate.name.clone());
        }
    }
    names
        .into_iter()
        .map(|name| {
            catalog
                .get(&name)
                .map(|dependency| (name.clone(), dependency.oid))
                .ok_or(ExecuteError::UndefinedRelation(name))
        })
        .collect()
}

pub(crate) fn transaction_row_deltas(operations: &[TransactionOperation]) -> Vec<WriteDelta> {
    operations
        .iter()
        .filter_map(|operation| match operation {
            TransactionOperation::Row(staged) => Some(staged.delta.clone()),
            TransactionOperation::Catalog(_) | TransactionOperation::TableReset(_) => None,
        })
        .collect()
}

pub(crate) fn final_transaction_row_operations(
    operations: &[TransactionOperation],
) -> Vec<Arc<StagedRowOperation>> {
    let last_resets = operations
        .iter()
        .enumerate()
        .filter_map(|(ordinal, operation)| match operation {
            TransactionOperation::TableReset(reset) => Some((reset.table.clone(), ordinal)),
            TransactionOperation::Catalog(_) | TransactionOperation::Row(_) => None,
        })
        .collect::<BTreeMap<_, _>>();
    operations
        .iter()
        .enumerate()
        .filter_map(|(ordinal, operation)| match operation {
            TransactionOperation::Row(staged)
                if last_resets
                    .get(mutation_table(&staged.mutation))
                    .is_none_or(|reset| ordinal > *reset) =>
            {
                Some(Arc::clone(staged))
            }
            TransactionOperation::Catalog(_)
            | TransactionOperation::Row(_)
            | TransactionOperation::TableReset(_) => None,
        })
        .collect()
}

pub(crate) fn final_transaction_operations(
    operations: &[TransactionOperation],
) -> (Vec<WriteDelta>, Vec<StagedTableReset>) {
    let last_resets = operations
        .iter()
        .enumerate()
        .filter_map(|(ordinal, operation)| match operation {
            TransactionOperation::TableReset(reset) => Some((reset.table.clone(), ordinal)),
            TransactionOperation::Catalog(_) | TransactionOperation::Row(_) => None,
        })
        .collect::<BTreeMap<_, _>>();
    let rows = final_transaction_row_operations(operations)
        .into_iter()
        .map(|staged| staged.delta.clone())
        .collect();
    let mut resets = Vec::new();
    for (ordinal, operation) in operations.iter().enumerate() {
        match operation {
            TransactionOperation::Catalog(_) | TransactionOperation::Row(_) => {}
            TransactionOperation::TableReset(reset)
                if last_resets.get(&reset.table) == Some(&ordinal) =>
            {
                resets.push(reset.as_ref().clone());
            }
            TransactionOperation::TableReset(_) => {}
        }
    }
    (rows, resets)
}

pub(crate) fn final_transaction_write_set(operations: &[TransactionOperation]) -> WriteSet {
    let (rows, resets) = final_transaction_operations(operations);
    let mut write_set = WriteSet::default();
    for delta in rows {
        write_set.extend_deduplicated(&delta.write_set);
    }
    for reset in resets {
        write_set.tables.insert(reset.table);
        write_set.tables.extend(reset.foreign_key_dependencies);
    }
    write_set
}

pub(crate) fn mutation_table(mutation: &PreparedMutation) -> &str {
    match mutation {
        PreparedMutation::Insert { table, .. }
        | PreparedMutation::Update { table, .. }
        | PreparedMutation::Delete { table, .. } => table,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
        gpu_db_sql::ParsedCommand::parse(sql).unwrap()
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn transactional_reset_replaces_cold_authority_with_a_fresh_empty_device_root() {
        let mut engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine.set_relational_residency_budget_bytes(0, 8192);
        engine
            .submit_transaction(
                1,
                parsed("CREATE TABLE cold_reset_target (id int4 PRIMARY KEY, value int4)"),
            )
            .unwrap();
        engine
            .submit_transaction(2, parsed("INSERT INTO cold_reset_target VALUES (1, 9)"))
            .unwrap();
        engine
            .transition_device_table_to_streaming_repair_above("cold_reset_target", 1)
            .unwrap();
        assert_eq!(
            engine
                .execute_relational_select_text("SELECT id FROM cold_reset_target")
                .unwrap()
                .rows,
            vec![vec![SqlValue::Int4(1)]]
        );
        assert!(engine
            .table_chunk_authoritative("cold_reset_target")
            .is_some());

        engine.submit_transaction(3, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(3, parsed("TRUNCATE cold_reset_target"))
            .unwrap();
        assert!(engine
            .execute_relational_select_in_transaction(
                3,
                &match parse_command("SELECT id FROM cold_reset_target").unwrap() {
                    Command::Select(select) => select,
                    _ => unreachable!(),
                },
            )
            .unwrap()
            .rows
            .is_empty());
        engine.submit_transaction(3, parsed("COMMIT")).unwrap();

        assert!(engine
            .table_chunk_authoritative("cold_reset_target")
            .is_none());
        assert!(engine.table_device_authoritative("cold_reset_target"));
        let table = engine.catalog_snapshot().relational_catalog["cold_reset_target"].clone();
        assert_eq!(
            engine.zero_row_resident_generation_boundary(&table),
            Some(engine.committed_seq())
        );
        assert!(engine
            .execute_relational_select_text("SELECT id FROM cold_reset_target")
            .unwrap()
            .rows
            .is_empty());

        let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
        assert!(recovered
            .execute_relational_select_text("SELECT id FROM cold_reset_target")
            .unwrap()
            .rows
            .is_empty());
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn staged_reset_excludes_direct_concurrent_and_serialized_dml_entrypoints() {
        let mut engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine
            .submit_transaction(
                10,
                parsed("CREATE TABLE reset_entry_guard (id int4 PRIMARY KEY)"),
            )
            .unwrap();
        engine
            .submit_transaction(11, parsed("INSERT INTO reset_entry_guard VALUES (1)"))
            .unwrap();
        engine.submit_transaction(12, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(12, parsed("TRUNCATE reset_entry_guard"))
            .unwrap();
        let wal_before = engine.durable_wal_records().len();

        let concurrent = engine
            .execute_dml_concurrent(13, "INSERT INTO reset_entry_guard VALUES (2)")
            .unwrap_err();
        assert!(matches!(concurrent, ExecuteError::Serialization(_)));
        let serialized = engine
            .execute_text(14, "INSERT INTO reset_entry_guard VALUES (3)")
            .unwrap_err();
        assert!(matches!(serialized, ExecuteError::Serialization(_)));
        let queued = engine
            .enqueue_set_text(
                15,
                "INSERT INTO reset_entry_guard VALUES (4)",
                Instant::now(),
            )
            .unwrap_err();
        assert!(matches!(queued, ExecuteError::Serialization(_)));
        assert_eq!(engine.batcher().len(), 0);
        let copy = engine
            .relational_copy_target("reset_entry_guard")
            .unwrap_err();
        assert!(matches!(copy, ExecuteError::Serialization(_)));
        let retained = engine
            .prepare_relational_retained_read_job(&match parse_command(
                "SELECT id FROM reset_entry_guard WHERE id = 1",
            )
            .unwrap()
            {
                Command::Select(select) => select,
                _ => unreachable!(),
            })
            .unwrap_err();
        assert!(matches!(retained, ExecuteError::Serialization(_)));
        let prepared_intent = engine
            .prepare_covered_insert_route("reset_entry_guard")
            .unwrap_err();
        assert!(matches!(prepared_intent, ExecuteError::Serialization(_)));
        let ddl = engine
            .execute_text(16, "ALTER TABLE reset_entry_guard ADD COLUMN value INT")
            .unwrap_err();
        assert!(matches!(ddl, ExecuteError::Serialization(_)));
        let catalog_wide = engine
            .execute_text(17, "CREATE TABLE reset_entry_peer (id INT)")
            .unwrap_err();
        assert!(matches!(catalog_wide, ExecuteError::Serialization(_)));
        assert_eq!(engine.durable_wal_records().len(), wal_before);

        engine.submit_transaction(12, parsed("COMMIT")).unwrap();
        assert!(engine
            .execute_relational_select_text("SELECT id FROM reset_entry_guard")
            .unwrap()
            .rows
            .is_empty());
    }
}
