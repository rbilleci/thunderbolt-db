//! Sole typed facade-to-engine admission for SQL control and logical mutations.
//!
//! Preparation/apply strategies remain distinct for now, but client-facing dispatch no longer
//! hands loose SQL text to several product entries. The parsed command and exact request text stay
//! paired by `ParsedCommand`; this boundary selects the existing strategy before any sequence/WAL
//! claim. Read execution deliberately remains outside this module.

use super::*;
use crate::engine_dml_concurrent::command_has_returning;
use crate::engine_transaction_reset::final_transaction_operations;

pub struct MutationRequest {
    parsed: gpu_db_sql::ParsedCommand,
    expected_catalog_version: Option<u64>,
    on_prepared: Option<Box<dyn FnOnce() + Send + 'static>>,
}

struct MutationRequestParts {
    parsed: gpu_db_sql::ParsedCommand,
    expected_catalog_version: Option<u64>,
    on_prepared: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl MutationRequest {
    pub fn new(parsed: gpu_db_sql::ParsedCommand) -> Self {
        Self {
            parsed,
            expected_catalog_version: None,
            on_prepared: None,
        }
    }

    /// Require mutation publication to use the catalog generation that was revalidated for this
    /// prepared execution. The commit sequencer checks the stamp under its publication lock.
    pub fn with_expected_catalog_version(mut self, version: u64) -> Self {
        self.expected_catalog_version = Some(version);
        self
    }

    /// Test-only instrumentation carried through the canonical submission method.
    #[doc(hidden)]
    pub fn with_prepared_hook(mut self, on_prepared: impl FnOnce() + Send + 'static) -> Self {
        self.on_prepared = Some(Box::new(on_prepared));
        self
    }

    fn into_parts(self) -> MutationRequestParts {
        MutationRequestParts {
            parsed: self.parsed,
            expected_catalog_version: self.expected_catalog_version,
            on_prepared: self.on_prepared,
        }
    }
}

impl From<gpu_db_sql::ParsedCommand> for MutationRequest {
    fn from(parsed: gpu_db_sql::ParsedCommand) -> Self {
        Self::new(parsed)
    }
}

pub(crate) fn validate_prepared_catalog_version(
    expected: u64,
    actual: u64,
) -> Result<(), ExecuteError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ExecuteError::Unsupported(format!(
            "prepared command catalog changed before execution (expected generation {expected}, current generation {actual}); re-Parse is required"
        )))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CatalogVersionExpectation {
    Prepared(u64),
    SequenceRoute(u64),
}

impl CatalogVersionExpectation {
    fn version(self) -> u64 {
        match self {
            Self::Prepared(version) | Self::SequenceRoute(version) => version,
        }
    }

    fn changed(self, detail: &str) -> ExecuteError {
        match self {
            Self::Prepared(_) => ExecuteError::Unsupported(format!(
                "prepared command {detail}; re-Parse is required"
            )),
            Self::SequenceRoute(_) => ExecuteError::Serialization(format!(
                "sequence-default route {detail}; retry the statement"
            )),
        }
    }
}

pub(crate) fn validate_catalog_version_expectation(
    expectation: CatalogVersionExpectation,
    actual: u64,
) -> Result<(), ExecuteError> {
    let expected = expectation.version();
    if expected == actual {
        Ok(())
    } else {
        Err(expectation.changed(&format!(
            "catalog changed before execution (expected generation {expected}, current generation {actual})"
        )))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransactionResources {
    pub operations: u32,
    pub mutations: u32,
    pub post_image_and_wal_bytes: u64,
    pub maintained_index_fanout: u32,
    pub touched_tables: u32,
    pub cold_accesses: u32,
    pub result_bytes: u64,
}

impl TransactionResources {
    pub const W1_MAX: Self = Self {
        operations: 1,
        mutations: 1,
        post_image_and_wal_bytes: 512,
        maintained_index_fanout: 2,
        touched_tables: 1,
        cold_accesses: 0,
        result_bytes: 64,
    };
    pub const T8_MAX: Self = Self {
        operations: 8,
        mutations: 4,
        post_image_and_wal_bytes: 4_096,
        maintained_index_fanout: 8,
        touched_tables: 3,
        cold_accesses: 0,
        result_bytes: 2_048,
    };
    pub const T32_MAX: Self = Self {
        operations: 32,
        mutations: 16,
        post_image_and_wal_bytes: 16_384,
        maintained_index_fanout: 32,
        touched_tables: 3,
        cold_accesses: 0,
        result_bytes: 8_192,
    };

    pub(crate) fn fits_within(self, limit: Self) -> bool {
        self.operations <= limit.operations
            && self.mutations <= limit.mutations
            && self.post_image_and_wal_bytes <= limit.post_image_and_wal_bytes
            && self.maintained_index_fanout <= limit.maintained_index_fanout
            && self.touched_tables <= limit.touched_tables
            && self.cold_accesses <= limit.cold_accesses
            && self.result_bytes <= limit.result_bytes
    }

    fn first_excess(self, declared: Self) -> Option<&'static str> {
        [
            (self.operations > declared.operations, "operations"),
            (self.mutations > declared.mutations, "mutations"),
            (
                self.post_image_and_wal_bytes > declared.post_image_and_wal_bytes,
                "post-image plus logical-WAL bytes",
            ),
            (
                self.maintained_index_fanout > declared.maintained_index_fanout,
                "maintained-index fanout",
            ),
            (
                self.touched_tables > declared.touched_tables,
                "touched tables",
            ),
            (self.cold_accesses > declared.cold_accesses, "cold accesses"),
            (self.result_bytes > declared.result_bytes, "result bytes"),
        ]
        .into_iter()
        .find_map(|(excess, name)| excess.then_some(name))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionClass {
    W1,
    T8,
    T32,
    /// Semantically supported work outside the bounded low-latency envelopes. This class has no
    /// operation-count ceiling; interactive transactions likewise retain no T32-derived cap.
    General,
}

impl TransactionClass {
    pub(crate) fn derive(
        resources: TransactionResources,
        engine_prepared_fast_route: bool,
    ) -> Self {
        if !engine_prepared_fast_route {
            return Self::General;
        }
        if resources.operations == 1
            && resources.mutations == 1
            && resources.fits_within(TransactionResources::W1_MAX)
        {
            Self::W1
        } else if (2..=8).contains(&resources.operations)
            && (1..=4).contains(&resources.mutations)
            && resources.fits_within(TransactionResources::T8_MAX)
        {
            Self::T8
        } else if (9..=32).contains(&resources.operations)
            && (1..=16).contains(&resources.mutations)
            && resources.fits_within(TransactionResources::T32_MAX)
        {
            Self::T32
        } else {
            Self::General
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredeclaredTransaction {
    operations: Vec<gpu_db_sql::ParsedCommand>,
    declared: TransactionResources,
    characteristics: TransactionCharacteristics,
}

impl PredeclaredTransaction {
    pub fn new(
        operations: Vec<gpu_db_sql::ParsedCommand>,
        declared: TransactionResources,
        characteristics: TransactionCharacteristics,
    ) -> Self {
        Self {
            operations,
            declared,
            characteristics,
        }
    }
}

// Keep the single-statement fast path inline. Boxing it merely to shrink this transient dispatch
// enum would add a heap allocation to every W1 admission; predeclared programs already own a Vec.
#[allow(clippy::large_enum_variant)]
pub enum TransactionRequest {
    Statement(MutationRequest),
    Copy(CopyMutationRequest),
    Predeclared(PredeclaredTransaction),
    Prepared(BoundPreparedTransactionRoute),
}

/// One typed COPY FROM STDIN mutation admitted through [`Engine::submit_transaction`].
///
/// The pgwire adapter owns framing and row decoding; this request keeps the parsed COPY target and
/// typed cells together at the engine's sole product-facing mutation boundary. Autocommit COPY may
/// use the private current-apply strategy, while an explicit transaction stages the same rows in
/// its private GPU generation and publishes them only at COMMIT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyMutationRequest {
    copy: CopyFromStdin,
    rows: Vec<Vec<SqlValue>>,
    target: CopyTargetProof,
}

impl CopyMutationRequest {
    pub fn new(copy: CopyFromStdin, rows: Vec<Vec<SqlValue>>, target: CopyTargetProof) -> Self {
        Self { copy, rows, target }
    }

    fn into_parts(self) -> (CopyFromStdin, Vec<Vec<SqlValue>>, CopyTargetProof) {
        (self.copy, self.rows, self.target)
    }
}

impl From<MutationRequest> for TransactionRequest {
    fn from(request: MutationRequest) -> Self {
        Self::Statement(request)
    }
}

impl From<CopyMutationRequest> for TransactionRequest {
    fn from(request: CopyMutationRequest) -> Self {
        Self::Copy(request)
    }
}

impl From<gpu_db_sql::ParsedCommand> for TransactionRequest {
    fn from(parsed: gpu_db_sql::ParsedCommand) -> Self {
        Self::Statement(MutationRequest::new(parsed))
    }
}

impl From<PredeclaredTransaction> for TransactionRequest {
    fn from(transaction: PredeclaredTransaction) -> Self {
        Self::Predeclared(transaction)
    }
}

impl From<BoundPreparedTransactionRoute> for TransactionRequest {
    fn from(transaction: BoundPreparedTransactionRoute) -> Self {
        Self::Prepared(transaction)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredeclaredOperationResult {
    Read(RelationalSelectResult),
    Mutation(DmlExecutionResult),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredeclaredTransactionResult {
    pub class: TransactionClass,
    pub actual_resources: TransactionResources,
    pub operations: Vec<PredeclaredOperationResult>,
}

pub(crate) struct PredeclaredSubmissionContext<OnRegistered, OnStaged> {
    pub(crate) engine_prepared_route: bool,
    pub(crate) engine_prepared_fast_route: bool,
    pub(crate) prepared_proof: Option<Arc<crate::engine_prepared_transaction::PreparedRouteProof>>,
    pub(crate) prepared_pin_registry:
        Option<crate::engine_prepared_transaction::PreparedPinRegistry>,
    pub(crate) on_transaction_registered: OnRegistered,
    pub(crate) on_operation_staged: OnStaged,
}

struct PredeclaredStageContext<'a, OnStaged> {
    engine_prepared_fast_route: bool,
    prepared_proof: Option<&'a crate::engine_prepared_transaction::PreparedRouteProof>,
    prepared_pin_registry: Option<&'a crate::engine_prepared_transaction::PreparedPinRegistry>,
    on_operation_staged: &'a mut OnStaged,
}

#[derive(Debug)]
pub enum TransactionAdmissionResult {
    Command,
    Dml(DmlExecutionResult),
    SequenceValue(SequenceValueOutcome),
    Transaction(Option<TxnId>),
    Predeclared(PredeclaredTransactionResult),
}

impl Engine {
    /// Submit one typed statement/control request or one atomic predeclared transaction.
    ///
    /// This is the only product-facing mutation boundary used by `gpu_db_facade`. Existing engine
    /// text/covered/probe APIs remain compatibility or preparation seams while PRODUCT-001 migrates
    /// their consumers; none is called by the canonical facade after this slice.
    pub fn submit_transaction(
        &self,
        txn_id: TxnId,
        request: impl Into<TransactionRequest>,
    ) -> Result<TransactionAdmissionResult, ExecuteError> {
        self.observe_transaction_id(txn_id);
        match request.into() {
            TransactionRequest::Statement(request) => self.submit_statement(txn_id, request),
            TransactionRequest::Copy(request) => {
                self.reject_nonstatement_sequence_autocommit_parent(txn_id)?;
                self.submit_copy(txn_id, request)
            }
            TransactionRequest::Predeclared(transaction) => {
                self.reject_nonstatement_sequence_autocommit_parent(txn_id)?;
                self.submit_predeclared_transaction(txn_id, transaction)
                    .map(TransactionAdmissionResult::Predeclared)
            }
            TransactionRequest::Prepared(transaction) => {
                self.reject_nonstatement_sequence_autocommit_parent(txn_id)?;
                self.submit_bound_prepared_transaction(txn_id, transaction)
                    .map(TransactionAdmissionResult::Predeclared)
            }
        }
    }

    fn submit_copy(
        &self,
        txn_id: TxnId,
        request: CopyMutationRequest,
    ) -> Result<TransactionAdmissionResult, ExecuteError> {
        let (copy, rows, target) = request.into_parts();
        if self.transaction_snapshot_handle(txn_id).is_some() {
            let insert = self.relational_copy_insert(&copy, rows);
            let result =
                self.execute_copy_in_transaction_with_result(txn_id, insert, &copy, &target)?;
            return Ok(TransactionAdmissionResult::Dml(result));
        }
        let (rows_affected, _profile) =
            self.execute_relational_copy_rows_profiled_with_target(txn_id, &copy, rows, &target)?;
        Ok(TransactionAdmissionResult::Dml(DmlExecutionResult {
            rows_affected: rows_affected as u64,
            returning: None,
        }))
    }

    fn submit_statement(
        &self,
        txn_id: TxnId,
        request: MutationRequest,
    ) -> Result<TransactionAdmissionResult, ExecuteError> {
        let MutationRequestParts {
            parsed,
            expected_catalog_version,
            on_prepared,
        } = request.into_parts();
        let (command, source) = parsed.into_parts();
        self.validate_sequence_autocommit_statement_parent(txn_id, &command)?;
        match command {
            Command::Select(_)
            | Command::SelectFunction(_)
            | Command::SelectLiteral(_)
            | Command::ShowTransactionIsolation
            | Command::SequenceCurrVal(_)
            | Command::GetKv { .. } => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "read-only commands do not enter mutation admission".to_string(),
            ))),
            Command::Begin { characteristics } => {
                reject_prepared_hook(on_prepared)?;
                self.execute_parsed_text(txn_id, Command::Begin { characteristics }, &source)?;
                Ok(TransactionAdmissionResult::Transaction(Some(txn_id)))
            }
            command @ (Command::SequenceNextVal(_) | Command::SequenceSetVal(_)) => {
                reject_prepared_hook(on_prepared)?;
                self.execute_sequence_value_command(txn_id, &command)
                    .map(TransactionAdmissionResult::SequenceValue)
            }
            Command::Commit { chain } => {
                reject_prepared_hook(on_prepared)?;
                self.commit_explicit_transaction(txn_id, chain)
                    .map(TransactionAdmissionResult::Transaction)
            }
            Command::Rollback { chain } => {
                reject_prepared_hook(on_prepared)?;
                self.rollback_explicit_transaction(txn_id, chain)
                    .map(TransactionAdmissionResult::Transaction)
            }
            Command::SessionControl {
                transaction: Some(characteristics),
                access_share_relations,
            } => {
                reject_prepared_hook(on_prepared)?;
                if !access_share_relations.is_empty() {
                    return Err(ExecuteError::Unsupported(
                        "SET TRANSACTION cannot carry an access-share relation list".to_string(),
                    ));
                }
                self.set_empty_transaction_characteristics(txn_id, characteristics)?;
                Ok(TransactionAdmissionResult::Command)
            }
            Command::TruncateTable(truncate) => {
                reject_prepared_hook(on_prepared)?;
                if self.transaction_snapshot_handle(txn_id).is_some() {
                    self.execute_truncate_in_transaction(
                        txn_id,
                        truncate,
                        expected_catalog_version,
                    )?;
                } else {
                    self.execute_truncate_autocommit(txn_id, truncate, expected_catalog_version)?;
                }
                Ok(TransactionAdmissionResult::Command)
            }
            command @ (Command::Insert(_) | Command::Update(_) | Command::Delete(_)) => {
                let result = if self.transaction_snapshot_handle(txn_id).is_some() {
                    reject_prepared_hook(on_prepared)?;
                    self.execute_prepared_dml_in_transaction_with_result(
                        txn_id,
                        command,
                        expected_catalog_version.map(CatalogVersionExpectation::Prepared),
                        false,
                    )?
                } else {
                    let (omits_published_sequence_default, route_catalog_version) =
                        self.insert_sequence_default_route(&command);
                    let catalog_expectation = expected_catalog_version
                        .map(CatalogVersionExpectation::Prepared)
                        .or_else(|| {
                            route_catalog_version.map(CatalogVersionExpectation::SequenceRoute)
                        });
                    if omits_published_sequence_default {
                        reject_prepared_hook(on_prepared)?;
                        self.execute_sequence_default_autocommit(
                            txn_id,
                            command,
                            catalog_expectation,
                        )?
                    } else if self.is_concurrent_dml_command(&command) {
                        match on_prepared {
                            Some(on_prepared) => self
                                .execute_parsed_dml_concurrent_instrumented_with_catalog(
                                    txn_id,
                                    command,
                                    &source,
                                    catalog_expectation,
                                    on_prepared,
                                )?,
                            None => self.execute_parsed_dml_concurrent_with_catalog(
                                txn_id,
                                command,
                                &source,
                                catalog_expectation,
                            )?,
                        }
                    } else {
                        reject_prepared_hook(on_prepared)?;
                        if command_has_returning(&command) {
                            return Err(ExecuteError::Unsupported(
                                "DML RETURNING requires the GPU-native concurrent mutation path"
                                    .to_string(),
                            ));
                        }
                        self.execute_parsed_text_with_catalog(
                            txn_id,
                            command,
                            &source,
                            route_catalog_version,
                        )?;
                        return Ok(TransactionAdmissionResult::Command);
                    }
                };
                Ok(TransactionAdmissionResult::Dml(result))
            }
            command => {
                reject_prepared_hook(on_prepared)?;
                if self.transaction_snapshot_handle(txn_id).is_some() {
                    self.execute_catalog_in_transaction(
                        txn_id,
                        command,
                        source,
                        expected_catalog_version,
                    )?;
                    return Ok(TransactionAdmissionResult::Command);
                }
                self.execute_parsed_text(txn_id, command, &source)?;
                Ok(TransactionAdmissionResult::Command)
            }
        }
    }

    pub(crate) fn execute_prepared_dml_in_transaction_with_result(
        &self,
        txn_id: TxnId,
        command: Command,
        expectation: Option<CatalogVersionExpectation>,
        sequence_parent_autocommit: bool,
    ) -> Result<DmlExecutionResult, ExecuteError> {
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
        if let Some(expectation) = expectation {
            let transaction_catalog = snapshot.transaction_catalog();
            // A prepared command targeting this transaction's own CREATE TABLE cannot be
            // resolved in the published catalog generation carried by the protocol descriptor:
            // the relation deliberately does not exist there yet. The private overlay is the
            // only valid expected catalog for that exact target. All published-table targets
            // retain the existing generation-pinned dependency proof below.
            let expected_dependencies =
                if transaction_private_create_is_dml_target(&snapshot, &command) {
                    prepared_dml_catalog_dependencies(&transaction_catalog, &command)?
                } else {
                    let expected_catalog = self.read_catalog_as_of(expectation.version());
                    prepared_dml_catalog_dependencies(&expected_catalog, &command)?
                };
            let actual_dependencies =
                prepared_dml_catalog_dependencies(&transaction_catalog, &command)?;
            if expected_dependencies != actual_dependencies {
                return Err(expectation.changed(
                    "catalog dependencies changed at the transaction statement snapshot",
                ));
            }
        }
        self.execute_parsed_dml_in_transaction_statement_locked_with_sequence_parent(
            txn_id,
            command,
            &snapshot,
            sequence_parent_autocommit,
        )
    }

    fn submit_predeclared_transaction(
        &self,
        txn_id: TxnId,
        transaction: PredeclaredTransaction,
    ) -> Result<PredeclaredTransactionResult, ExecuteError> {
        self.submit_predeclared_transaction_with_hooks(
            txn_id,
            transaction,
            PredeclaredSubmissionContext {
                engine_prepared_route: false,
                engine_prepared_fast_route: false,
                prepared_proof: None,
                prepared_pin_registry: None,
                on_transaction_registered: || {},
                on_operation_staged: |_| {},
            },
        )
    }

    #[cfg(test)]
    fn submit_predeclared_transaction_instrumented(
        &self,
        txn_id: TxnId,
        transaction: PredeclaredTransaction,
        on_transaction_registered: impl FnOnce(),
        on_operation_staged: impl FnMut(usize),
    ) -> Result<PredeclaredTransactionResult, ExecuteError> {
        self.submit_predeclared_transaction_with_hooks(
            txn_id,
            transaction,
            PredeclaredSubmissionContext {
                engine_prepared_route: false,
                engine_prepared_fast_route: false,
                prepared_proof: None,
                prepared_pin_registry: None,
                on_transaction_registered,
                on_operation_staged,
            },
        )
    }

    pub(crate) fn submit_predeclared_transaction_with_hooks<OnRegistered, OnStaged>(
        &self,
        txn_id: TxnId,
        transaction: PredeclaredTransaction,
        context: PredeclaredSubmissionContext<OnRegistered, OnStaged>,
    ) -> Result<PredeclaredTransactionResult, ExecuteError>
    where
        OnRegistered: FnOnce(),
        OnStaged: FnMut(usize),
    {
        let PredeclaredSubmissionContext {
            engine_prepared_route,
            engine_prepared_fast_route,
            prepared_proof,
            prepared_pin_registry,
            on_transaction_registered,
            mut on_operation_staged,
        } = context;
        let PredeclaredTransaction {
            operations,
            declared,
            characteristics,
        } = transaction;
        validate_predeclared_characteristics(characteristics, &operations)?;
        let mutation_count = operations
            .iter()
            .filter(|operation| is_dml(operation.command()))
            .count();
        if mutation_count > 0 {
            self.legacy_lane_history_write_guard()
                .map_err(ExecuteError::Engine)?;
        }
        let operation_count = u32::try_from(operations.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "predeclared transaction operation count exceeds u32 framing".to_string(),
            )
        })?;
        let mutation_count = u32::try_from(mutation_count).map_err(|_| {
            ExecuteError::Unsupported(
                "predeclared transaction mutation count exceeds u32 framing".to_string(),
            )
        })?;
        if operations.is_empty() {
            return Err(ExecuteError::Unsupported(
                "predeclared transaction must contain at least one operation".to_string(),
            ));
        }
        if operation_count != declared.operations || mutation_count != declared.mutations {
            return Err(ExecuteError::Unsupported(format!(
                "predeclared transaction shape does not match its resource envelope: actual operations/mutations {operation_count}/{mutation_count}, declared {}/{}",
                declared.operations, declared.mutations
            )));
        }
        // General atomic programs still require statically bounded one-row operations. This keeps
        // result/work memory bounded before execution and expands every DML dependency into the
        // resource plan; arbitrary scans/broad DML remain available through the interactive path.
        let access_plan = predeclared_access_plan(&self.catalog_snapshot(), &operations)?;
        let touched_table_count =
            u32::try_from(access_plan.touched_tables.len()).map_err(|_| {
                ExecuteError::Unsupported(
                    "predeclared transaction touched-table count exceeds u32 framing".to_string(),
                )
            })?;
        if touched_table_count > declared.touched_tables {
            return Err(resource_excess("touched tables"));
        }
        // Parsed SQL is deliberately General. W1/T8/T32 become reachable only when an
        // engine-owned prepared route supplies the proof; a caller-provided manifest cannot
        // promote arbitrary scans or broad DML into a latency class.
        let class = TransactionClass::derive(declared, engine_prepared_fast_route);

        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.begin_predeclared_transaction_context(txn_id, characteristics)?;
        on_transaction_registered();
        let mut snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        debug_assert!(snapshot.program_owned.load(AtomicOrdering::Acquire));
        // One existing transaction statement guard owns the complete program lifecycle. Per-op
        // helpers below are explicitly the statement-locked variants, so no recursive mutex is
        // introduced and no same-id COMMIT/DML/SELECT can interleave a prefix.
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _program = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, &snapshot)?;
        if let Some(proof) = prepared_proof.as_deref() {
            let access = predeclared_access_plan(&snapshot.catalog, &operations)?;
            let expected = proof.table_names();
            if access.touched_tables != expected {
                let _ =
                    self.rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
                return Err(ExecuteError::Unsupported(
                    "prepared transaction catalog dependency closure changed; re-prepare required"
                        .to_string(),
                ));
            }
            let already_pinned = prepared_pin_registry.as_ref().is_some_and(|registry| {
                registry
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .iter()
                    .any(|pins| self.prepared_pins_cover_snapshot(proof, &snapshot, pins))
            });
            match already_pinned.then_some(()).map_or_else(
                || {
                    self.validate_prepared_route_snapshot(proof, &snapshot)
                        .map(Some)
                },
                |_| Ok(None),
            ) {
                Ok(Some(pins)) => {
                    prepared_pin_registry
                        .as_ref()
                        .expect("prepared proof carries its pin registry")
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(pins);
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = self
                        .rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
                    return Err(error);
                }
            }
        }
        let access_plan = match predeclared_access_plan(&snapshot.catalog, &operations) {
            Ok(plan) => plan,
            Err(error) => {
                let _ =
                    self.rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
                return Err(error);
            }
        };
        let touched_table_count = match u32::try_from(access_plan.touched_tables.len()) {
            Ok(value) => value,
            Err(_) => {
                let _ =
                    self.rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
                return Err(ExecuteError::Unsupported(
                    "predeclared transaction touched-table count exceeds u32 framing".to_string(),
                ));
            }
        };
        let cold_accesses = match predeclared_cold_accesses(&snapshot, &access_plan) {
            Ok(value) => value,
            Err(error) => {
                let _ =
                    self.rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
                return Err(error);
            }
        };
        if touched_table_count > declared.touched_tables {
            let _ = self.rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
            return Err(resource_excess("touched tables"));
        }
        if cold_accesses > declared.cold_accesses {
            let _ = self.rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
            return Err(resource_excess("cold accesses"));
        }
        let staged = self.stage_predeclared_operations(
            txn_id,
            &mut snapshot,
            operations,
            &mut PredeclaredStageContext {
                engine_prepared_fast_route,
                prepared_proof: prepared_proof.as_deref(),
                prepared_pin_registry: prepared_pin_registry.as_ref(),
                on_operation_staged: &mut on_operation_staged,
            },
        );
        let operation_results = match staged {
            Ok(results) => results,
            Err(error) => {
                let _ =
                    self.rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
                return Err(error);
            }
        };
        let actual = match self.predeclared_actual_resources(
            txn_id,
            &snapshot,
            operation_count,
            mutation_count,
            touched_table_count,
            cold_accesses,
            &operation_results,
        ) {
            Ok(actual) => actual,
            Err(error) => {
                let _ =
                    self.rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
                return Err(error);
            }
        };
        if let Some(dimension) = actual.first_excess(declared) {
            let _ = self.rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
            return Err(resource_excess(dimension));
        }
        if engine_prepared_route {
            self.record_prepared_transaction_admission(class);
        } else {
            self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
        }

        match self.commit_explicit_transaction_statement_locked(txn_id, false, &snapshot) {
            Ok(None) => Ok(PredeclaredTransactionResult {
                class,
                actual_resources: actual,
                operations: operation_results,
            }),
            Ok(Some(_)) => Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "predeclared transaction unexpectedly produced an AND CHAIN successor".to_string(),
            ))),
            Err(error) => {
                if !error.is_indeterminate() && !self.is_commit_path_poisoned() {
                    let _ = self
                        .rollback_explicit_transaction_statement_locked(txn_id, false, &snapshot);
                }
                Err(error)
            }
        }
    }

    fn stage_predeclared_operations<OnStaged>(
        &self,
        txn_id: TxnId,
        snapshot: &mut Arc<TransactionSnapshot>,
        operations: Vec<gpu_db_sql::ParsedCommand>,
        context: &mut PredeclaredStageContext<'_, OnStaged>,
    ) -> Result<Vec<PredeclaredOperationResult>, ExecuteError>
    where
        OnStaged: FnMut(usize),
    {
        let mut results = Vec::with_capacity(operations.len());
        for (operation_index, operation) in operations.into_iter().enumerate() {
            *snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, snapshot)?;
            if let Some(proof) = context.prepared_proof {
                let registry = context
                    .prepared_pin_registry
                    .expect("prepared proof carries its pin registry");
                let already_pinned = registry
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .iter()
                    .any(|pins| self.prepared_pins_cover_snapshot(proof, snapshot, pins));
                if !already_pinned {
                    let pins = self.validate_prepared_route_snapshot(proof, snapshot)?;
                    registry
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(pins);
                }
            }
            let (command, _) = operation.into_parts();
            match command {
                Command::Select(select) => {
                    let result = if context.engine_prepared_fast_route {
                        self.execute_prepared_index_select_in_transaction_statement_locked(
                            txn_id, snapshot, &select,
                        )?
                    } else {
                        self.execute_relational_select_in_transaction_statement_locked(
                            txn_id,
                            snapshot,
                            &select,
                            || {},
                        )?
                    };
                    results.push(PredeclaredOperationResult::Read(result));
                }
                command @ (Command::Insert(_) | Command::Update(_) | Command::Delete(_)) => {
                    results.push(PredeclaredOperationResult::Mutation(
                        self.execute_parsed_dml_in_transaction_statement_locked(
                            txn_id, command, snapshot,
                        )?,
                    ));
                }
                _ => {
                    return Err(ExecuteError::Unsupported(
                        "predeclared transactions currently accept relational SELECT and DML operations only"
                            .to_string(),
                    ));
                }
            }
            (context.on_operation_staged)(operation_index);
        }
        Ok(results)
    }

    #[allow(clippy::too_many_arguments)]
    fn predeclared_actual_resources(
        &self,
        txn_id: TxnId,
        snapshot: &Arc<TransactionSnapshot>,
        operations: u32,
        mutations: u32,
        touched_tables: u32,
        cold_accesses: u32,
        results: &[PredeclaredOperationResult],
    ) -> Result<TransactionResources, ExecuteError> {
        let staged_operations = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .operations
            .clone();
        let (deltas, table_resets) = final_transaction_operations(&staged_operations);
        let has_staged_deltas = !deltas.is_empty() || !table_resets.is_empty();
        let provisional = Self::transaction_insert_identities(&deltas)?;
        let provisional_count = u64::try_from(provisional.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "predeclared transaction provisional identity count exceeds u64 framing"
                    .to_string(),
            )
        })?;
        let allocator_high_water = 1u64.checked_add(provisional_count).ok_or_else(|| {
            ExecuteError::Unsupported(
                "predeclared transaction provisional identity count overflow".to_string(),
            )
        })?;
        let mut record = Self::resolved_transaction_record(
            &deltas,
            &deltas,
            &provisional,
            1,
            allocator_high_water,
            &[],
        )?;
        record.table_resets = table_resets
            .iter()
            .map(StagedTableReset::to_binary)
            .collect();
        let wal_bytes = if has_staged_deltas {
            let payload: Arc<[u8]> =
                Arc::from(try_encode_binary_transaction(&record).ok_or_else(|| {
                    ExecuteError::Unsupported(
                        "predeclared transaction logical WAL exceeds binary framing".to_string(),
                    )
                })?);
            // Constructing and decoding this synthetic canonical record is read-only: the sequence
            // has fixed-width encoding and no row-id, replication, or WAL claim occurs here. The
            // manifest helper counts typed logical fragments + outcome only; physical frame and
            // record authority is deliberately excluded and reported by durability separately.
            let commit = self.commit_state();
            let isolation = match snapshot.characteristics.isolation {
                TransactionIsolation::ReadUncommitted | TransactionIsolation::ReadCommitted => {
                    gpu_db_wal::CanonicalIsolation::ReadCommitted
                }
                TransactionIsolation::RepeatableRead => {
                    gpu_db_wal::CanonicalIsolation::RepeatableRead
                }
                TransactionIsolation::Serializable => {
                    return Err(ExecuteError::Unsupported(
                        "SERIALIZABLE transactions are not implemented".to_string(),
                    ));
                }
            };
            let canonical = Self::canonical_wal_record_with_isolation(
                &commit, txn_id, 1, 0, &payload, isolation,
            )
            .map_err(ExecuteError::Engine)?;
            let envelope = gpu_db_wal::decode_canonical_record_payload(&canonical.payload)
                .map_err(ExecuteError::Engine)?
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::Durability(
                        "predeclared WAL preflight did not produce a canonical envelope"
                            .to_string(),
                    ))
                })?;
            gpu_db_wal::canonical_logical_intent_outcome_bytes(
                &envelope.fragments,
                &envelope.outcome,
            )
            .map_err(ExecuteError::Engine)?
        } else {
            // Empty explicit transactions are closed without sequence/WAL publication.
            0
        };
        let mut post_image_bytes = 0u64;
        let mut maintained_index_fanout = 0u32;
        for mutation in &record.mutations {
            let (table, post_image_len) = match mutation {
                BinaryTransactionMutation::Insert {
                    table, row_encoded, ..
                }
                | BinaryTransactionMutation::Update {
                    table,
                    new_row_encoded: row_encoded,
                    ..
                } => (table, row_encoded.len()),
                BinaryTransactionMutation::Delete { table, .. } => (table, 0),
            };
            let post_image = u64::try_from(post_image_len).map_err(|_| {
                ExecuteError::Unsupported(
                    "predeclared transaction post-image length exceeds u64 framing".to_string(),
                )
            })?;
            post_image_bytes = post_image_bytes.checked_add(post_image).ok_or_else(|| {
                ExecuteError::Unsupported(
                    "predeclared transaction post-image byte count overflow".to_string(),
                )
            })?;
            let index_count = snapshot
                .catalog
                .relational_catalog
                .get(table)
                .map(|table| table.indexes.len())
                .unwrap_or(0);
            let index_count = u32::try_from(index_count).map_err(|_| {
                ExecuteError::Unsupported(
                    "predeclared transaction index count exceeds u32 framing".to_string(),
                )
            })?;
            maintained_index_fanout = maintained_index_fanout
                .checked_add(index_count)
                .ok_or_else(|| {
                    ExecuteError::Unsupported(
                        "predeclared transaction index-fanout count overflow".to_string(),
                    )
                })?;
        }
        let result_bytes = results.iter().try_fold(0u64, |total, result| {
            total
                .checked_add(predeclared_result_bytes(result)?)
                .ok_or_else(|| {
                    ExecuteError::Unsupported(
                        "predeclared transaction result-byte count overflow".to_string(),
                    )
                })
        })?;
        Ok(TransactionResources {
            operations,
            mutations,
            post_image_and_wal_bytes: post_image_bytes.checked_add(wal_bytes).ok_or_else(|| {
                ExecuteError::Unsupported(
                    "predeclared transaction post-image plus WAL byte count overflow".to_string(),
                )
            })?,
            maintained_index_fanout,
            touched_tables,
            cold_accesses,
            result_bytes,
        })
    }

    /// Execute the bounded legacy read commands outside mutation admission.
    ///
    /// These commands preserve the old leader/poison/metrics behavior, but unlike the historical
    /// generic `execute_text` route they cannot trigger representation repair and cannot claim a
    /// sequence or WAL record. Relational SELECT has its own result-bearing GPU read path.
    pub fn execute_parsed_compatibility_read(
        &self,
        parsed: gpu_db_sql::ParsedCommand,
    ) -> Result<(), ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let (command, _) = parsed.into_parts();
        match command {
            Command::GetKv { key } => {
                if self.repl_role() != Role::Leader {
                    return Err(ExecuteError::Engine(EngineError::NotLeader));
                }
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                if let Some(len) = self.commit_state().sm.kv.get(&key).map(|value| value.len()) {
                    self.metrics.observe_d2h_bytes(len as u64);
                }
                Ok(())
            }
            Command::SelectFunction(call) if call.name == "pg_advisory_unlock_all" => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                Ok(())
            }
            Command::SequenceCurrVal(_) => {
                self.metrics.inc_fallback(FallbackReason::NotGpuEligible);
                Ok(())
            }
            Command::SelectLiteral(_) => Err(ExecuteError::NonReadCommand(
                "literal projections use the relational GPU result boundary",
            )),
            _ => Err(ExecuteError::NonReadCommand(
                "parsed compatibility read boundary",
            )),
        }
    }
}

fn is_dml(command: &Command) -> bool {
    matches!(
        command,
        Command::Insert(_) | Command::Update(_) | Command::Delete(_)
    )
}

fn prepared_dml_catalog_dependencies(
    catalog: &CatalogSnapshot,
    command: &Command,
) -> Result<BTreeMap<String, RelationalTable>, ExecuteError> {
    let target_name = match command {
        Command::Insert(insert) => &insert.table,
        Command::Update(update) => &update.table,
        Command::Delete(delete) => &delete.table,
        _ => {
            return Err(ExecuteError::Unsupported(
                "prepared transaction catalog validation accepts DML only".to_string(),
            ));
        }
    };
    let target = catalog
        .relational_catalog
        .get(target_name)
        .ok_or_else(|| ExecuteError::UndefinedRelation(target_name.clone()))?;
    let mut names = BTreeSet::from([target_name.clone()]);
    names.extend(
        target
            .foreign_keys
            .iter()
            .map(|foreign_key| foreign_key.referenced_table.clone()),
    );
    names.extend(
        catalog
            .relational_catalog
            .iter()
            .filter(|(_, candidate)| {
                candidate
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == *target_name)
            })
            .map(|(name, _)| name.clone()),
    );
    names
        .into_iter()
        .map(|name| {
            catalog
                .relational_catalog
                .get(&name)
                .cloned()
                .map(|table| (name.clone(), table))
                .ok_or(ExecuteError::UndefinedRelation(name))
        })
        .collect()
}

fn transaction_private_create_is_dml_target(
    snapshot: &TransactionSnapshot,
    command: &Command,
) -> bool {
    let target_name = match command {
        Command::Insert(insert) => insert.table.as_str(),
        Command::Update(update) => update.table.as_str(),
        Command::Delete(delete) => delete.table.as_str(),
        _ => return false,
    };
    let delta = snapshot
        .delta
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    delta.operations.iter().any(|operation| {
        matches!(
            operation,
            TransactionOperation::Catalog(staged)
                if matches!(&staged.command, Command::CreateTable(create) if create.table == target_name)
        )
    })
}

fn validate_predeclared_characteristics(
    characteristics: TransactionCharacteristics,
    operations: &[gpu_db_sql::ParsedCommand],
) -> Result<(), ExecuteError> {
    validate_transaction_characteristics(characteristics)?;
    if characteristics.access == TransactionAccessMode::ReadOnly
        && operations
            .iter()
            .any(|operation| is_dml(operation.command()))
    {
        return Err(ExecuteError::Unsupported(
            "READ ONLY predeclared transaction contains DML; request rejected before BEGIN"
                .to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_transaction_characteristics(
    mut characteristics: TransactionCharacteristics,
) -> Result<TransactionCharacteristics, ExecuteError> {
    if characteristics.deferrable {
        return Err(ExecuteError::Unsupported(
            "DEFERRABLE transactions are not implemented; request rejected before BEGIN"
                .to_string(),
        ));
    }
    if characteristics.isolation == TransactionIsolation::Serializable {
        return Err(ExecuteError::Unsupported(
            "SERIALIZABLE transactions are not implemented; request rejected before BEGIN"
                .to_string(),
        ));
    }
    if characteristics.isolation == TransactionIsolation::ReadUncommitted {
        characteristics.isolation = TransactionIsolation::ReadCommitted;
    }
    Ok(characteristics)
}

pub(crate) struct PredeclaredAccessPlan {
    pub(crate) touched_tables: BTreeSet<String>,
    pub(crate) operation_accesses: Vec<BTreeSet<String>>,
}

pub(crate) fn predeclared_access_plan(
    catalog: &CatalogSnapshot,
    operations: &[gpu_db_sql::ParsedCommand],
) -> Result<PredeclaredAccessPlan, ExecuteError> {
    let mut touched_tables = BTreeSet::new();
    let mut operation_accesses = Vec::with_capacity(operations.len());
    for operation in operations {
        let (table_name, mutation, bounded) = match operation.command() {
            Command::Select(select) => {
                let table = catalog.relational_catalog.get(&select.table);
                (
                    select.table.as_str(),
                    false,
                    table.is_some_and(|table| {
                        filter_targets_unique_row(
                            table,
                            select.filter.as_ref(),
                            &select.filters,
                            &select.filter_groups,
                        )
                    }),
                )
            }
            Command::Insert(insert) => (insert.table.as_str(), true, insert.rows.len() == 1),
            Command::Update(update) => {
                let table = catalog.relational_catalog.get(&update.table);
                (
                    update.table.as_str(),
                    true,
                    table.is_some_and(|table| {
                        filter_targets_unique_row(
                            table,
                            update.filter.as_ref(),
                            &update.filters,
                            &update.filter_groups,
                        )
                    }),
                )
            }
            Command::Delete(delete) => {
                let table = catalog.relational_catalog.get(&delete.table);
                (
                    delete.table.as_str(),
                    true,
                    table.is_some_and(|table| {
                        filter_targets_unique_row(
                            table,
                            delete.filter.as_ref(),
                            &delete.filters,
                            &delete.filter_groups,
                        )
                    }),
                )
            }
            _ => {
                return Err(ExecuteError::Unsupported(
                    "predeclared transactions accept bounded relational SELECT and DML operations only; request rejected before BEGIN"
                        .to_string(),
                ));
            }
        };
        let table = catalog.relational_catalog.get(table_name).ok_or_else(|| {
            ExecuteError::Unsupported(format!(
                "predeclared transaction relation \"{table_name}\" is not a base table in the prepared catalog"
            ))
        })?;
        if !bounded {
            return Err(ExecuteError::Unsupported(format!(
                "predeclared transaction operation on \"{table_name}\" is not statically bounded to one unique-key row; use the unbounded interactive transaction path"
            )));
        }

        let mut accesses = BTreeSet::from([table_name.to_string()]);
        if mutation {
            accesses.extend(
                table
                    .foreign_keys
                    .iter()
                    .map(|foreign_key| foreign_key.referenced_table.clone()),
            );
            // UPDATE/DELETE constraint checks can consult referencing relations. Counting the
            // conservative reverse-FK closure prevents hidden table/cold work from escaping the
            // manifest even when the eventual target is absent.
            if matches!(operation.command(), Command::Update(_) | Command::Delete(_)) {
                for (candidate_name, candidate) in &catalog.relational_catalog {
                    if candidate
                        .foreign_keys
                        .iter()
                        .any(|foreign_key| foreign_key.referenced_table == table_name)
                    {
                        accesses.insert(candidate_name.clone());
                    }
                }
            }
        }
        touched_tables.extend(accesses.iter().cloned());
        operation_accesses.push(accesses);
    }
    Ok(PredeclaredAccessPlan {
        touched_tables,
        operation_accesses,
    })
}

fn filter_targets_unique_row(
    table: &RelationalTable,
    filter: Option<&gpu_db_sql::SelectFilter>,
    filters: &[gpu_db_sql::SelectFilter],
    filter_groups: &[Vec<gpu_db_sql::SelectFilter>],
) -> bool {
    if filter_groups.len() > 1 {
        return false;
    }
    let equality_columns = filter
        .into_iter()
        .chain(filters)
        .filter(|filter| filter.op == SelectFilterOp::Eq)
        .map(|filter| filter.column.as_str())
        .collect::<BTreeSet<_>>();
    table.indexes.iter().any(|index| {
        index.unique
            && !index.key_columns.is_empty()
            && index
                .key_columns
                .iter()
                .all(|column| equality_columns.contains(column.as_str()))
    })
}

fn predeclared_cold_accesses(
    snapshot: &Arc<TransactionSnapshot>,
    access_plan: &PredeclaredAccessPlan,
) -> Result<u32, ExecuteError> {
    access_plan
        .operation_accesses
        .iter()
        .try_fold(0u32, |total, accesses| {
            let cold = accesses.iter().try_fold(0u32, |count, table| {
                count
                    .checked_add(u32::from(
                        snapshot.chunk_authoritative_tables.contains_key(table),
                    ))
                    .ok_or_else(|| {
                        ExecuteError::Unsupported(
                            "predeclared transaction cold-access count overflow".to_string(),
                        )
                    })
            })?;
            total.checked_add(cold).ok_or_else(|| {
                ExecuteError::Unsupported(
                    "predeclared transaction cold-access count overflow".to_string(),
                )
            })
        })
}

fn predeclared_result_bytes(result: &PredeclaredOperationResult) -> Result<u64, ExecuteError> {
    let rows = match result {
        PredeclaredOperationResult::Read(result) => Some(&result.rows),
        PredeclaredOperationResult::Mutation(result) => {
            result.returning.as_ref().map(|returning| &returning.rows)
        }
    };
    rows.into_iter()
        .flat_map(|rows| rows.iter())
        .flat_map(|row| row.iter())
        .try_fold(0u64, |total, value| {
            total.checked_add(sql_value_bytes(value)?).ok_or_else(|| {
                ExecuteError::Unsupported(
                    "predeclared transaction result-byte count overflow".to_string(),
                )
            })
        })
}

fn sql_value_bytes(value: &SqlValue) -> Result<u64, ExecuteError> {
    let bytes = match value {
        SqlValue::Null => 0,
        SqlValue::Int2(_) => 2,
        SqlValue::Int4(_) | SqlValue::Date(_) => 4,
        SqlValue::Int8(_) | SqlValue::Timestamp(_) => 8,
        SqlValue::Numeric(_) | SqlValue::Uuid(_) => 16,
        SqlValue::Bool(_) => 1,
        SqlValue::Text(value) => u64::try_from(value.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "predeclared transaction text result exceeds u64 framing".to_string(),
            )
        })?,
        SqlValue::Parameter { .. } => {
            return Err(ExecuteError::Unsupported(
                "unbound prepared parameter reached transaction result accounting".to_string(),
            ));
        }
    };
    Ok(bytes)
}

fn resource_excess(dimension: &'static str) -> ExecuteError {
    ExecuteError::Unsupported(format!(
        "predeclared transaction exceeded its declared {dimension} before sequence/WAL claim"
    ))
}

fn reject_prepared_hook(
    on_prepared: Option<Box<dyn FnOnce() + Send + 'static>>,
) -> Result<(), ExecuteError> {
    if on_prepared.is_some() {
        return Err(ExecuteError::Unsupported(
            "prepared instrumentation requires eligible autocommit DML".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        transaction_private_create_is_dml_target, validate_transaction_characteristics, Engine,
        MutationRequest, PredeclaredOperationResult, PredeclaredTransaction,
        TransactionAdmissionResult, TransactionCharacteristics, TransactionClass,
        TransactionResources,
    };
    use gpu_db_sql::{ParsedCommand, SqlValue};
    use std::sync::{mpsc, Arc, Barrier};
    use std::time::Duration;

    fn parsed(sql: &str) -> ParsedCommand {
        ParsedCommand::parse(sql).unwrap()
    }

    fn resources(operations: u32, mutations: u32) -> TransactionResources {
        TransactionResources {
            operations,
            mutations,
            post_image_and_wal_bytes: u64::MAX,
            maintained_index_fanout: u32::MAX,
            touched_tables: u32::MAX,
            cold_accesses: u32::MAX,
            result_bytes: u64::MAX,
        }
    }

    #[test]
    fn only_engine_prepared_routes_derive_fast_classes_and_general_has_no_t32_cap() {
        assert_eq!(
            TransactionClass::derive(TransactionResources::W1_MAX, false),
            TransactionClass::General
        );
        assert_eq!(
            TransactionClass::derive(TransactionResources::W1_MAX, true),
            TransactionClass::W1
        );
        assert_eq!(
            TransactionClass::derive(TransactionResources::T8_MAX, true),
            TransactionClass::T8
        );
        assert_eq!(
            TransactionClass::derive(TransactionResources::T32_MAX, true),
            TransactionClass::T32
        );
        assert_eq!(
            TransactionClass::derive(resources(33, 17), true),
            TransactionClass::General
        );
        assert_eq!(
            TransactionClass::derive(resources(10_000, 5_000), true),
            TransactionClass::General
        );
    }

    #[test]
    fn resource_envelope_checks_every_manifest_dimension() {
        let declared = TransactionResources {
            operations: 1,
            mutations: 1,
            post_image_and_wal_bytes: 1,
            maintained_index_fanout: 1,
            touched_tables: 1,
            cold_accesses: 1,
            result_bytes: 1,
        };
        let cases = [
            (
                TransactionResources {
                    operations: 2,
                    ..declared
                },
                "operations",
            ),
            (
                TransactionResources {
                    mutations: 2,
                    ..declared
                },
                "mutations",
            ),
            (
                TransactionResources {
                    post_image_and_wal_bytes: 2,
                    ..declared
                },
                "post-image plus logical-WAL bytes",
            ),
            (
                TransactionResources {
                    maintained_index_fanout: 2,
                    ..declared
                },
                "maintained-index fanout",
            ),
            (
                TransactionResources {
                    touched_tables: 2,
                    ..declared
                },
                "touched tables",
            ),
            (
                TransactionResources {
                    cold_accesses: 2,
                    ..declared
                },
                "cold accesses",
            ),
            (
                TransactionResources {
                    result_bytes: 2,
                    ..declared
                },
                "result bytes",
            ),
        ];
        for (actual, expected) in cases {
            assert_eq!(actual.first_excess(declared), Some(expected));
        }
    }

    #[test]
    fn invalid_predeclared_shape_fails_before_transaction_sequence_or_wal_effects() {
        let engine = Engine::new_local();
        let visible_before = engine.committed_seq();
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let transaction = PredeclaredTransaction::new(
            vec![parsed("CREATE TABLE forbidden (id INT)")],
            resources(1, 0),
            TransactionCharacteristics::REPEATABLE_READ_WRITE,
        );

        let error = engine.submit_transaction(71, transaction).unwrap_err();

        assert!(matches!(error, crate::ExecuteError::Unsupported(_)));
        assert_eq!(engine.committed_seq(), visible_before);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert!(engine.transaction_snapshot_handle(71).is_none());
    }

    #[test]
    fn transaction_characteristics_support_rc_and_reject_unsupported_modes_before_begin() {
        for isolation in [
            super::TransactionIsolation::ReadCommitted,
            super::TransactionIsolation::ReadUncommitted,
        ] {
            let normalized =
                validate_transaction_characteristics(super::TransactionCharacteristics {
                    isolation,
                    access: super::TransactionAccessMode::ReadWrite,
                    deferrable: false,
                })
                .unwrap();
            assert_eq!(
                normalized.isolation,
                super::TransactionIsolation::ReadCommitted
            );
        }

        for (txn_id, characteristics, expected) in [
            (
                72,
                super::TransactionCharacteristics {
                    isolation: super::TransactionIsolation::Serializable,
                    access: super::TransactionAccessMode::ReadWrite,
                    deferrable: false,
                },
                "SERIALIZABLE",
            ),
            (
                73,
                super::TransactionCharacteristics {
                    isolation: super::TransactionIsolation::ReadCommitted,
                    access: super::TransactionAccessMode::ReadWrite,
                    deferrable: true,
                },
                "DEFERRABLE",
            ),
        ] {
            let engine = Engine::new_local();
            let visible_before = engine.committed_seq();
            let wal_before = engine.durable_wal_records().len();
            let transaction = PredeclaredTransaction::new(
                vec![parsed("SELECT id FROM t WHERE id = 1")],
                resources(1, 0),
                characteristics,
            );
            let error = engine.submit_transaction(txn_id, transaction).unwrap_err();
            assert!(matches!(error, crate::ExecuteError::Unsupported(_)));
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(engine.committed_seq(), visible_before);
            assert_eq!(engine.durable_wal_records().len(), wal_before);
            assert!(engine.transaction_snapshot_handle(txn_id).is_none());
        }
    }

    #[test]
    fn dependency_closure_is_preclaimed_before_begin() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(1, parsed("CREATE TABLE p (id INT PRIMARY KEY)"))
            .unwrap();
        engine
            .submit_transaction(2, parsed("CREATE TABLE c (id INT PRIMARY KEY, pid INT)"))
            .unwrap();
        engine
            .submit_transaction(
                3,
                parsed("ALTER TABLE ONLY c ADD CONSTRAINT c_fk FOREIGN KEY (pid) REFERENCES p(id)"),
            )
            .unwrap();
        let visible_before = engine.committed_seq();
        let wal_before = engine.durable_wal_records().len();
        let transaction = PredeclaredTransaction::new(
            vec![parsed("INSERT INTO c VALUES (1, 1)")],
            TransactionResources {
                operations: 1,
                mutations: 1,
                post_image_and_wal_bytes: u64::MAX,
                maintained_index_fanout: u32::MAX,
                touched_tables: 1,
                cold_accesses: u32::MAX,
                result_bytes: u64::MAX,
            },
            TransactionCharacteristics::REPEATABLE_READ_WRITE,
        );

        let error = engine.submit_transaction(4, transaction).unwrap_err();

        assert!(error.to_string().contains("touched tables"));
        assert_eq!(engine.committed_seq(), visible_before);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert!(engine.transaction_snapshot_handle(4).is_none());
    }

    #[test]
    fn parsed_mutation_admission_routes_generic_and_concurrent_dml() {
        let engine = Engine::new_local();
        assert!(matches!(
            engine
                .submit_transaction(
                    1,
                    MutationRequest::new(parsed("CREATE TABLE t (id INT, v INT)")),
                )
                .unwrap(),
            TransactionAdmissionResult::Command
        ));
        let TransactionAdmissionResult::Dml(result) = engine
            .submit_transaction(
                2,
                MutationRequest::new(parsed("INSERT INTO t VALUES (7, 9)")),
            )
            .unwrap()
        else {
            panic!("concurrent DML must retain its result through typed admission");
        };
        assert_eq!(result.rows_affected, 1);
        assert_eq!(engine.durable_wal_records().len(), 2);
    }

    #[test]
    fn private_catalog_expected_proof_is_scoped_to_the_exact_created_table() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(10, parsed("CREATE TABLE published_target (id int4)"))
            .unwrap();
        engine.submit_transaction(11, parsed("BEGIN")).unwrap();
        engine
            .submit_transaction(
                11,
                parsed("CREATE TABLE private_target (id int4, value int4)"),
            )
            .unwrap();
        let snapshot = engine.transaction_snapshot_handle(11).unwrap();

        assert!(transaction_private_create_is_dml_target(
            &snapshot,
            parsed("INSERT INTO private_target VALUES (1, 2)").command()
        ));
        assert!(transaction_private_create_is_dml_target(
            &snapshot,
            parsed("UPDATE private_target SET value = 3 WHERE id = 1").command()
        ));
        assert!(transaction_private_create_is_dml_target(
            &snapshot,
            parsed("DELETE FROM private_target WHERE id = 1").command()
        ));
        assert!(!transaction_private_create_is_dml_target(
            &snapshot,
            parsed("INSERT INTO published_target VALUES (1)").command()
        ));
        assert!(!transaction_private_create_is_dml_target(
            &snapshot,
            parsed("SELECT id FROM private_target").command()
        ));

        engine.submit_transaction(11, parsed("ROLLBACK")).unwrap();
    }

    #[test]
    fn read_only_command_is_rejected_without_sequence_or_wal_claim() {
        let engine = Engine::new_local();
        let visible_before = engine.committed_seq();
        let wal_before = engine.durable_wal_records().len();
        let error = engine
            .submit_transaction(1, MutationRequest::new(parsed("SELECT id FROM t")))
            .unwrap_err();
        assert!(error.to_string().contains("read-only commands"));
        assert_eq!(engine.committed_seq(), visible_before);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
    }

    #[test]
    fn compatibility_reads_are_unsequenced_and_do_not_enter_representation_repair() {
        let engine = Engine::new_local();
        engine.execute_text(1, "SET answer=forty-two").unwrap();
        let visible_before = engine.committed_seq();
        let wal_before = engine.durable_wal_records().len();
        let fallback_before = engine.metrics().snapshot().fallback_total;

        for sql in [
            "GET answer",
            "SELECT pg_advisory_unlock_all()",
            "SELECT currval('seq')",
        ] {
            engine
                .execute_parsed_compatibility_read(parsed(sql))
                .unwrap();
        }

        assert_eq!(engine.committed_seq(), visible_before);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(
            engine.metrics().snapshot().fallback_total,
            fallback_before + 3
        );
    }

    #[test]
    fn nonconcurrent_returning_rejects_before_sequence_or_wal_claim() {
        let engine = Engine::new_local();
        let visible_before = engine.committed_seq();
        let wal_before = engine.durable_wal_records().len();
        let error = engine
            .submit_transaction(
                1,
                MutationRequest::new(parsed("INSERT INTO missing VALUES (1) RETURNING id")),
            )
            .unwrap_err();
        assert!(matches!(error, crate::ExecuteError::Unsupported(_)));
        assert_eq!(engine.committed_seq(), visible_before);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn general_atomic_program_reads_its_writes_commits_once_and_recovers() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(
                1,
                parsed("CREATE TABLE t (id INT PRIMARY KEY, v INT UNIQUE)"),
            )
            .unwrap();
        let visible_before = engine.committed_seq();
        let wal_before = engine.durable_wal_records().len();
        let transaction = PredeclaredTransaction::new(
            [
                "INSERT INTO t VALUES (1, 10) RETURNING id",
                "SELECT id, v FROM t WHERE id = 1",
                "UPDATE t SET v = 11 WHERE id = 1 RETURNING v",
                "SELECT id, v FROM t WHERE id = 1",
                "INSERT INTO t VALUES (2, 20) RETURNING id",
                "SELECT id, v FROM t WHERE id = 2",
                "DELETE FROM t WHERE id = 2 RETURNING id",
                "SELECT id, v FROM t WHERE id = 2",
            ]
            .into_iter()
            .map(parsed)
            .collect(),
            TransactionResources::T8_MAX,
            TransactionCharacteristics::REPEATABLE_READ_WRITE,
        );

        let TransactionAdmissionResult::Predeclared(result) =
            engine.submit_transaction(2, transaction).unwrap()
        else {
            panic!("predeclared submission must retain its ordered results");
        };
        assert_eq!(result.class, TransactionClass::General);
        assert_eq!(result.actual_resources.operations, 8);
        assert_eq!(result.actual_resources.mutations, 4);
        let PredeclaredOperationResult::Read(first_read) = &result.operations[1] else {
            panic!("operation 2 must be the first read-own-write result");
        };
        assert_eq!(
            first_read.rows,
            vec![vec![SqlValue::Int4(1), SqlValue::Int4(10)]]
        );
        let PredeclaredOperationResult::Read(updated_read) = &result.operations[3] else {
            panic!("operation 4 must be the updated read-own-write result");
        };
        assert_eq!(
            updated_read.rows,
            vec![vec![SqlValue::Int4(1), SqlValue::Int4(11)]]
        );
        let PredeclaredOperationResult::Read(deleted_read) = &result.operations[7] else {
            panic!("operation 8 must read after delete");
        };
        assert!(deleted_read.rows.is_empty());
        assert_eq!(engine.committed_seq(), visible_before + 1);
        assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
        let records = engine.durable_wal_records();
        let envelope = gpu_db_wal::decode_canonical_record_payload(
            &records.last().expect("transaction WAL").payload,
        )
        .unwrap()
        .expect("canonical transaction WAL");
        assert_eq!(
            envelope.header.isolation,
            gpu_db_wal::CanonicalIsolation::RepeatableRead
        );

        let recovered = Engine::recover_from_durable_wal(&records).unwrap();
        let recovered_rows = recovered
            .execute_relational_select_text("SELECT id, v FROM t")
            .unwrap();
        assert_eq!(
            recovered_rows.rows,
            vec![vec![SqlValue::Int4(1), SqlValue::Int4(11)]]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn same_id_commit_and_dml_cannot_interleave_a_predeclared_prefix() {
        let engine = Arc::new(Engine::new_local());
        engine
            .submit_transaction(1, parsed("CREATE TABLE t (id INT PRIMARY KEY)"))
            .unwrap();
        let visible_before = engine.committed_seq();
        let wal_before = engine.durable_wal_records().len();
        let registered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let program_engine = Arc::clone(&engine);
        let program_registered = Arc::clone(&registered);
        let program_release = Arc::clone(&release);
        let program = std::thread::spawn(move || {
            program_engine.submit_predeclared_transaction_instrumented(
                90,
                PredeclaredTransaction::new(
                    vec![
                        parsed("INSERT INTO t VALUES (1)"),
                        parsed("INSERT INTO t VALUES (2)"),
                    ],
                    resources(2, 2),
                    TransactionCharacteristics::REPEATABLE_READ_WRITE,
                ),
                move || {
                    program_registered.wait();
                    program_release.wait();
                },
                |_| {},
            )
        });
        registered.wait();

        let (commit_tx, commit_rx) = mpsc::channel();
        let commit_engine = Arc::clone(&engine);
        let commit = std::thread::spawn(move || {
            let result = commit_engine.submit_transaction(90, parsed("COMMIT"));
            commit_tx.send(result).unwrap();
        });
        let (dml_tx, dml_rx) = mpsc::channel();
        let dml_engine = Arc::clone(&engine);
        let dml = std::thread::spawn(move || {
            let result = dml_engine.submit_transaction(90, parsed("INSERT INTO t VALUES (3)"));
            dml_tx.send(result).unwrap();
        });
        assert!(commit_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .is_err());
        assert!(dml_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .is_err());

        release.wait();
        let result = program.join().unwrap().unwrap();
        assert_eq!(result.operations.len(), 2);
        commit.join().unwrap();
        dml.join().unwrap();
        assert_eq!(engine.committed_seq(), visible_before + 1);
        assert_eq!(engine.durable_wal_records().len(), wal_before + 1);
        assert_eq!(
            engine
                .execute_relational_select_text("SELECT id FROM t")
                .unwrap()
                .rows,
            vec![vec![SqlValue::Int4(1)], vec![SqlValue::Int4(2)]]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn post_durable_failure_is_indeterminate_and_recovery_owns_the_commit() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(1, parsed("CREATE TABLE t (id INT PRIMARY KEY)"))
            .unwrap();
        engine.fail_next_transaction_post_durable_apply();
        let transaction = PredeclaredTransaction::new(
            vec![parsed("INSERT INTO t VALUES (1)")],
            resources(1, 1),
            TransactionCharacteristics::REPEATABLE_READ_WRITE,
        );

        let error = engine.submit_transaction(2, transaction).unwrap_err();

        assert!(error.is_indeterminate());
        assert!(engine.is_commit_path_poisoned());
        let recovered = Engine::recover_from_durable_wal(&engine.durable_wal_records()).unwrap();
        assert_eq!(
            recovered
                .execute_relational_select_text("SELECT id FROM t")
                .unwrap()
                .rows,
            vec![vec![SqlValue::Int4(1)]]
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_resource_excess_rolls_back_before_global_claims() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(1, parsed("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)"))
            .unwrap();
        let visible_before = engine.committed_seq();
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let transaction = PredeclaredTransaction::new(
            vec![parsed("INSERT INTO t VALUES (1, 'too large')")],
            TransactionResources {
                operations: 1,
                mutations: 1,
                post_image_and_wal_bytes: 0,
                maintained_index_fanout: 1,
                touched_tables: 1,
                cold_accesses: 0,
                result_bytes: 0,
            },
            TransactionCharacteristics::REPEATABLE_READ_WRITE,
        );

        let error = engine.submit_transaction(2, transaction).unwrap_err();

        assert!(error
            .to_string()
            .contains("post-image plus logical-WAL bytes"));
        assert_eq!(engine.committed_seq(), visible_before);
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert!(engine.transaction_snapshot_handle(2).is_none());
        assert!(engine
            .execute_relational_select_text("SELECT id FROM t")
            .unwrap()
            .rows
            .is_empty());
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn general_predeclared_and_interactive_transactions_exceed_thirty_two_operations() {
        let engine = Engine::new_local();
        engine
            .submit_transaction(1, parsed("CREATE TABLE t (id INT PRIMARY KEY)"))
            .unwrap();
        let reads = (0..40)
            .map(|_| parsed("SELECT id FROM t WHERE id = -1"))
            .collect();
        let TransactionAdmissionResult::Predeclared(result) = engine
            .submit_transaction(
                2,
                PredeclaredTransaction::new(
                    reads,
                    TransactionResources {
                        operations: 40,
                        mutations: 0,
                        post_image_and_wal_bytes: 0,
                        maintained_index_fanout: 0,
                        touched_tables: 1,
                        cold_accesses: 0,
                        result_bytes: 0,
                    },
                    TransactionCharacteristics::REPEATABLE_READ_WRITE,
                ),
            )
            .unwrap()
        else {
            panic!("forty-operation predeclared transaction must be supported");
        };
        assert_eq!(result.class, TransactionClass::General);
        assert_eq!(result.operations.len(), 40);

        engine.submit_transaction(3, parsed("BEGIN")).unwrap();
        for id in 0..40 {
            engine
                .submit_transaction(3, parsed(&format!("INSERT INTO t VALUES ({id})")))
                .unwrap();
        }
        engine.submit_transaction(3, parsed("COMMIT")).unwrap();
        assert_eq!(
            engine
                .execute_relational_select_text("SELECT id FROM t")
                .unwrap()
                .rows
                .len(),
            40
        );
    }
}
