//! Explicit-transaction private GPU generation: stage prepared DML into immutable shard overlays
//! without mutating the globally published relation. The transaction snapshot remains the owner;
//! SELECT and subsequent DML load the latest private generation through the ordinary residency
//! accessors.

use super::*;
use crate::engine_transaction_catalog::TransactionCatalogEnvelopeSlices;
use crate::engine_transaction_reset::{
    final_transaction_operations, final_transaction_row_operations, final_transaction_write_set,
    table_access_dependency_identities, table_schema_digest, transaction_row_deltas,
    FinalTransactionRowOperation,
};

mod codec5_operation;
pub(crate) use codec5_operation::indexed_generation_inputs;
pub(crate) mod gpu_accounting;
mod isolation;
mod ordered_wal;
mod record;
mod typed_insert;
use gpu_accounting::transaction_private_shard_bytes;
pub(crate) use gpu_accounting::TransactionGpuReservation;
pub(crate) use typed_insert::{StagedTypedInsert, TypedPrivateSequenceAdvance};

type TargetRow<'a> = (u64, &'a [SqlValue]);

/// One type-neutral CUDA generation invocation shared by live codec-5 INSERT and fresh replay.
/// The source stays columnar and move-only; this owner carries only scalar generation inputs and
/// opaque completion evidence, never a host row reconstruction or a type-selected write branch.
pub(crate) struct TypedInsertRuntimeGenerationInput<'a> {
    pub(crate) source: &'a crate::typed_insert_batch::PreparedResidentAppendSource,
    /// Exact global allocator metadata authenticated by this generation.  The row source below
    /// owns stable entity identities independently: full catalog rewrites retain existing,
    /// potentially sparse ids and therefore must not masquerade as allocator claims.
    pub(crate) row_allocator_before: u64,
    pub(crate) first_row_id: u64,
    /// Exact S7 source identity for every row in the combined final image. A one-statement
    /// INSERT is simply a slice whose statement ordinal is constant; recovery rebuilds this
    /// same slice from the retained transition directory.
    pub(crate) row_sources: &'a [TypedInsertRuntimeGenerationRowSource],
    pub(crate) database_id: [u8; 16],
    pub(crate) catalog_epoch: u64,
    pub(crate) catalog_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) stable_transaction_id: TxnId,
    pub(crate) commit_sequence: Index,
    pub(crate) typed_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) action: gpu_db_execution::RuntimeTypedInsertGenerationTableAction,
    pub(crate) table_map_predecessor:
        gpu_db_execution::RuntimeTypedInsertGenerationTableMapPredecessor,
    pub(crate) stable_table_id: u64,
    pub(crate) write001_final_image_ref: u32,
    pub(crate) base_data_generation: u64,
    pub(crate) base_table_root: gpu_db_wal::CanonicalDigest,
    pub(crate) row_allocator_high_water: u64,
    pub(crate) initial_logical_row_count: u64,
    pub(crate) final_logical_row_count: u64,
    pub(crate) image_layout_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) image_content_digest: gpu_db_wal::CanonicalDigest,
    /// Catalog-ordered index descriptors for the one generic device generation.  They are
    /// neutral, bounded facts only: the CUDA program derives every successor root and the
    /// physical plan remains the sole post-WAL maintainer.
    pub(crate) indexes: &'a [TypedInsertRuntimeGenerationIndexInput],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TypedInsertRuntimeGenerationRowSource {
    pub(crate) stable_row_id: u64,
    pub(crate) statement_ordinal: u32,
    pub(crate) source_row_ordinal: u32,
}

/// One catalog-ordered named-index input retained only through the device generation launch.
/// The descriptor owns no row values, index storage, publication capability, or host-derived
/// generation root; its components refer back to the sealed columnar source written above.
pub(crate) struct TypedInsertRuntimeGenerationIndexInput {
    pub(crate) descriptor: gpu_db_execution::RuntimeTypedInsertGenerationIndex,
    pub(crate) key_columns: Box<[gpu_db_execution::RuntimeTypedInsertGenerationIndexKeyColumn]>,
}

pub(crate) struct TypedInsertRuntimeGenerationOutput {
    /// Scalar geometry already validated by the one generic generation input.  S7 consumes this
    /// immutable proof rather than rescanning the same sealed columnar source merely to recover
    /// its row and cell counts.
    pub(crate) source_geometry: gpu_db_execution::RuntimeTypedInsertGenerationGeometry,
    pub(crate) initial_table_root: gpu_db_wal::CanonicalDigest,
    pub(crate) final_table_root: gpu_db_wal::CanonicalDigest,
    pub(crate) initial_database_root: gpu_db_wal::CanonicalDigest,
    pub(crate) final_database_root: gpu_db_wal::CanonicalDigest,
    /// Exact fixed-height table-map root transition copied from the same GPU completion as the
    /// table/database commitments. The root publication holder consumes it without a host hash.
    pub(crate) table_map_completion: TypedTableMapGpuCompletion,
    pub(crate) commitments: gpu_db_execution::RuntimeTypedInsertGenerationCommitments,
    pub(crate) logical_completion: gpu_db_execution::RuntimeTypedInsertGenerationLogicalCompletion,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TransactionTerminalPreDurableFault {
    ReplicationProposal,
    CanonicalWalAppend,
    WalFlush,
}

#[cfg(test)]
impl TransactionTerminalPreDurableFault {
    const fn code(self) -> u8 {
        match self {
            Self::ReplicationProposal => 1,
            Self::CanonicalWalAppend => 2,
            Self::WalFlush => 3,
        }
    }
}

#[cfg(test)]
thread_local! {
    static TRANSACTION_TERMINAL_PRE_DURABLE_FAULT: std::cell::Cell<u8> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
type TransactionTerminalPreCommitLockHook = (
    usize,
    std::sync::Arc<std::sync::Barrier>,
    std::sync::Arc<std::sync::Barrier>,
);

#[cfg(test)]
fn transaction_terminal_pre_commit_lock_hook(
) -> &'static std::sync::Mutex<Option<TransactionTerminalPreCommitLockHook>> {
    static HOOK: std::sync::OnceLock<
        std::sync::Mutex<Option<TransactionTerminalPreCommitLockHook>>,
    > = std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

impl Engine {
    /// Execute exactly one authenticated generic typed generation.  Both the live terminal and
    /// recovery bind their sealed column vectors through this method before the sole device
    /// append/publication path; callers cannot specialize by SQL type or fall back to host rows.
    pub(crate) fn run_typed_insert_runtime_generation(
        &self,
        input: TypedInsertRuntimeGenerationInput<'_>,
    ) -> Result<TypedInsertRuntimeGenerationOutput, EngineError> {
        #[cfg(feature = "probe-timing")]
        let probe_generation_started = std::time::Instant::now();
        let generation_view = input
            .source
            .prepare_runtime_generation_view(input.row_sources)?;
        #[cfg(feature = "probe-timing")]
        let probe_source_view_nanos = probe_generation_started.elapsed().as_nanos() as u64;
        if generation_view.first_row_id() != input.first_row_id {
            return Err(EngineError::Durability(
                "typed generation first row identity differs from its ordered source".to_string(),
            ));
        }
        let target = self
            .cuda_driver_probe_runtime()
            .runtime_typed_insert_generation_target(0)
            .map_err(|error| {
                EngineError::Durability(format!(
                    "codec-5 CUDA generation target declined: {error:?}"
                ))
            })?;
        let attempt =
            gpu_db_execution::RuntimeTypedInsertGenerationAttempt::new(input.commit_sequence)
                .map_err(|error| {
                    EngineError::Durability(format!(
                        "codec-5 generation attempt is invalid: {error:?}"
                    ))
                })?;
        let source_geometry = generation_view.geometry();
        let mut index_keys = 0_usize;
        let mut index_effects = 0_usize;
        let mut index_effect_components = 0_usize;
        let mut expected_key_start = 0_u32;
        let mut expected_effect_start = 0_u32;
        for index in input.indexes {
            let key_count = u32::try_from(index.key_columns.len()).map_err(|_| {
                EngineError::Durability(
                    "codec-5 generation index key count exceeds u32".to_string(),
                )
            })?;
            let effect_count = u32::try_from(source_geometry.rows).map_err(|_| {
                EngineError::Durability("codec-5 generation row count exceeds u32".to_string())
            })?;
            if index.descriptor.stable_index_id == 0
                || index.descriptor.key_start != expected_key_start
                || index.descriptor.key_count != key_count
                || index.descriptor.effect_start != expected_effect_start
                || index.descriptor.effect_count != effect_count
                || key_count == 0
            {
                return Err(EngineError::Durability(
                    "codec-5 generation index descriptor ranges are not canonical".to_string(),
                ));
            }
            expected_key_start = expected_key_start.checked_add(key_count).ok_or_else(|| {
                EngineError::Durability("codec-5 generation index key range overflows".to_string())
            })?;
            expected_effect_start =
                expected_effect_start
                    .checked_add(effect_count)
                    .ok_or_else(|| {
                        EngineError::Durability(
                            "codec-5 generation index effect range overflows".to_string(),
                        )
                    })?;
            index_keys = index_keys
                .checked_add(index.key_columns.len())
                .ok_or_else(|| {
                    EngineError::Durability(
                        "codec-5 generation index key geometry overflows".to_string(),
                    )
                })?;
            index_effects = index_effects
                .checked_add(source_geometry.rows)
                .ok_or_else(|| {
                    EngineError::Durability(
                        "codec-5 generation index effect geometry overflows".to_string(),
                    )
                })?;
            index_effect_components = index_effect_components
                .checked_add(
                    index
                        .key_columns
                        .len()
                        .checked_mul(source_geometry.rows)
                        .ok_or_else(|| {
                            EngineError::Durability(
                                "codec-5 generation index component geometry overflows".to_string(),
                            )
                        })?,
                )
                .ok_or_else(|| {
                    EngineError::Durability(
                        "codec-5 generation index component geometry overflows".to_string(),
                    )
                })?;
        }
        let generation_geometry = gpu_db_execution::RuntimeTypedInsertGenerationGeometry {
            rows: source_geometry.rows,
            cells: source_geometry.cells,
            value_bytes: source_geometry.value_bytes,
            indexes: input.indexes.len(),
            index_keys,
            index_effects,
            index_effect_components,
        };
        if u32::try_from(index_keys).is_err()
            || u32::try_from(index_effects).is_err()
            || u32::try_from(index_effect_components).is_err()
        {
            return Err(EngineError::Durability(
                "codec-5 generation index geometry exceeds the runtime ABI".to_string(),
            ));
        }
        #[cfg(feature = "probe-timing")]
        let probe_prepare_reserve_started = std::time::Instant::now();
        let prepared = gpu_db_execution::PreparedRuntimeTypedInsertGeneration::reserve(
            target,
            attempt,
            generation_geometry,
        )
        .map_err(|error| {
            EngineError::Durability(format!(
                "codec-5 CUDA generation reservation declined: {error:?}"
            ))
        })?;
        #[cfg(feature = "probe-timing")]
        let probe_prepare_reserve_nanos = probe_prepare_reserve_started.elapsed().as_nanos() as u64;
        let runtime_identity = gpu_db_execution::RuntimeTypedInsertGenerationIdentity {
            database_id: input.database_id,
            catalog_epoch: input.catalog_epoch,
            catalog_digest: input.catalog_digest,
            stable_transaction_id: input.stable_transaction_id,
            commit_sequence: input.commit_sequence,
            write001_typed_statement_digest: input.typed_statement_digest,
        };
        let runtime_table = gpu_db_execution::RuntimeTypedInsertGenerationTable {
            action: input.action,
            table_map_predecessor: input.table_map_predecessor,
            stable_table_id: input.stable_table_id,
            write001_final_image_ref: input.write001_final_image_ref,
            base_data_generation: input.base_data_generation,
            base_table_root: input.base_table_root,
            row_allocator_before: input.row_allocator_before,
            row_allocator_high_water: input.row_allocator_high_water,
            initial_logical_row_count: input.initial_logical_row_count,
            final_logical_row_count: input.final_logical_row_count,
            image_layout_digest: input.image_layout_digest,
            image_content_digest: input.image_content_digest,
        };
        #[cfg(feature = "probe-timing")]
        let probe_submit_complete_started = std::time::Instant::now();
        let completion = prepared
            .launch(|encoder| {
                encoder.write_identity(runtime_identity);
                encoder.write_table(runtime_table);
                generation_view.write_rows(encoder);
                for index in input.indexes {
                    encoder.write_index(index.descriptor);
                    for key in index.key_columns.iter().copied() {
                        encoder.write_index_key(key);
                    }
                }
                let mut effect_ref = 0_u32;
                let mut component_ref = 0_u32;
                // Effects are index-major because every descriptor owns one contiguous
                // `effect_start..effect_start + effect_count` range.  The runtime accepts
                // plural descriptors even though WRITE-001 selects one physical index today.
                for index in input.indexes {
                    for source_row_ordinal in 0..source_geometry.rows {
                        let stable_row_id = input.row_sources[source_row_ordinal].stable_row_id;
                        encoder.write_index_effect(
                            gpu_db_execution::RuntimeTypedInsertGenerationIndexEffect {
                                stable_table_id: input.stable_table_id,
                                stable_index_id: index.descriptor.stable_index_id,
                                stable_row_id,
                                source_catalog_ordinal: index.descriptor.raw_catalog_index_ordinal,
                                component_start: component_ref,
                                component_count: u32::try_from(index.key_columns.len())
                                    .expect("validated index key count"),
                            },
                        );
                        effect_ref = effect_ref
                            .checked_add(1)
                            .expect("validated index effect count");
                        for key in index.key_columns.iter() {
                            encoder.write_index_effect_component(
                                gpu_db_execution::RuntimeTypedInsertGenerationIndexEffectComponent {
                                    catalog_column_ordinal: key.catalog_column_ordinal,
                                    stable_column_id: key.stable_column_id,
                                },
                            );
                            component_ref = component_ref
                                .checked_add(1)
                                .expect("validated index component count");
                        }
                    }
                }
                debug_assert_eq!(effect_ref as usize, index_effects);
                debug_assert_eq!(component_ref as usize, index_effect_components);
            })
            .complete();
        #[cfg(feature = "probe-timing")]
        let probe_submit_complete_nanos = probe_submit_complete_started.elapsed().as_nanos() as u64;
        let proof = match completion {
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => proof,
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
                return Err(EngineError::Durability(format!(
                    "codec-5 CUDA generation failed before publication: {error}"
                )));
            }
            gpu_db_execution::RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(
                unknown,
            ) => {
                // A CUDA API failure after submission makes quiescence unknowable.  Retrying the
                // same poisoned context can spin forever; drop the owner for its one bounded
                // drain/park-or-quarantine attempt, then let fresh replay select a new context.
                let error = unknown.error().to_string();
                drop(unknown);
                return Err(EngineError::Durability(format!(
                    "codec-5 CUDA generation quiescence is unproven: {error}"
                )));
            }
        };
        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_typed_append_generation_nanos(
            probe_generation_started.elapsed().as_nanos() as u64,
            probe_source_view_nanos,
            probe_prepare_reserve_nanos,
            probe_submit_complete_nanos,
            proof.kernel_event_elapsed_nanos(),
            proof.kernel_phase_event_nanos(),
        );
        let mut initial_table_root = [0_u8; 32];
        let mut final_table_root = [0_u8; 32];
        let mut initial_database_root = [0_u8; 32];
        let mut final_database_root = [0_u8; 32];
        let (commitments, logical_completion) = proof.consume(|_attempt, commitments, logical| {
            commitments.copy_initial_table_root_into(&mut initial_table_root);
            commitments.copy_final_table_root_into(&mut final_table_root);
            commitments.copy_initial_database_root_into(&mut initial_database_root);
            commitments.copy_final_database_root_into(&mut final_database_root);
            (commitments, logical)
        });
        let initial_table_absent = matches!(
            input.action,
            gpu_db_execution::RuntimeTypedInsertGenerationTableAction::CreateWithRowSet
        );
        if (!initial_table_absent && initial_table_root == [0; 32])
            || (initial_table_absent && initial_table_root != [0; 32])
            || final_table_root == [0; 32]
            || initial_database_root == [0; 32]
            || final_database_root == [0; 32]
            || initial_table_root == final_table_root
            || initial_database_root == final_database_root
        {
            return Err(EngineError::Durability(
                "codec-5 CUDA generation returned invalid root commitments".to_string(),
            ));
        }
        let table_map_completion =
            TypedTableMapGpuCompletion::from_logical_completion(&logical_completion)?;
        Ok(TypedInsertRuntimeGenerationOutput {
            source_geometry,
            initial_table_root,
            final_table_root,
            initial_database_root,
            final_database_root,
            table_map_completion,
            commitments,
            logical_completion,
        })
    }

    #[cfg(test)]
    pub(crate) fn fail_next_transaction_terminal_pre_durable_at(
        &self,
        boundary: TransactionTerminalPreDurableFault,
    ) {
        TRANSACTION_TERMINAL_PRE_DURABLE_FAULT.with(|fault| fault.set(boundary.code()));
    }

    /// Pause the current transaction terminal just before it acquires the canonical commit cut.
    /// This test seam proves all transaction terminal callers reload the allocator only after the
    /// cut, including the autocommit overlay route that replaced the serial INSERT wave.
    #[cfg(test)]
    pub(crate) fn set_transaction_terminal_pre_commit_lock_hook(
        &self,
        reached: std::sync::Arc<std::sync::Barrier>,
        resume: std::sync::Arc<std::sync::Barrier>,
    ) {
        *transaction_terminal_pre_commit_lock_hook()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((self as *const Self as usize, reached, resume));
    }

    #[cfg(test)]
    fn run_transaction_terminal_pre_commit_lock_hook(&self) {
        let hook = {
            let mut hook = transaction_terminal_pre_commit_lock_hook()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            hook.as_ref()
                .is_some_and(|(engine, _, _)| *engine == self as *const Self as usize)
                .then(|| hook.take())
                .flatten()
        };
        if let Some((_, reached, resume)) = hook {
            reached.wait();
            resume.wait();
        }
    }

    #[cfg(test)]
    pub(crate) fn fail_next_transaction_post_durable_apply(&self) {
        self.fail_next_transaction_post_durable_apply
            .store(true, AtomicOrdering::Release);
    }

    /// The transaction terminal has one post-durable failure seam irrespective of the physical
    /// plan selected beneath it. A consumed injection always requires restart recovery; it is
    /// never a license to re-enter another INSERT strategy.
    #[cfg(test)]
    fn take_transaction_post_durable_apply_fault(&self) -> Option<EngineError> {
        self.fail_next_transaction_post_durable_apply
            .swap(false, AtomicOrdering::AcqRel)
            .then(|| {
                EngineError::ApplyFailed(
                    "injected post-durable transaction-terminal apply failure".to_string(),
                )
            })
    }

    /// Consume one test-only generic terminal seam before the durable boundary. The production
    /// path has no fault branch; all three locations use their ordinary rollback path.
    #[cfg(test)]
    fn fail_transaction_terminal_pre_durable_if_injected(
        &self,
        boundary: TransactionTerminalPreDurableFault,
    ) -> Result<(), ExecuteError> {
        let injected = TRANSACTION_TERMINAL_PRE_DURABLE_FAULT.with(|fault| {
            let injected = fault.get() == boundary.code();
            if injected {
                fault.set(0);
            }
            injected
        });
        if injected {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "injected transaction-terminal pre-durable failure at {boundary:?}"
            ))));
        }
        Ok(())
    }

    /// All terminal plan and publication reservations complete before WAL. A decline leaves the
    /// private overlay, allocator, WAL, status, and publication unchanged, so callers receive a
    /// retryable serialization result rather than an apply failure from a physical strategy.
    fn retryable_transaction_terminal_pre_wal_decline(
        detail: impl std::fmt::Display,
    ) -> ExecuteError {
        ExecuteError::Serialization(format!(
            "transaction terminal pre-WAL plan or reservation declined; retry the statement: {detail}"
        ))
    }

    #[cfg(test)]
    pub(crate) fn set_transaction_post_durable_hook(&self, hook: impl FnOnce() + Send + 'static) {
        *self
            .transaction_post_durable_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Box::new(hook));
    }

    /// Drive COMMIT and an optional successor as one engine-owned transaction-control operation.
    /// Returning the successor identity lets protocol façades adopt the transaction that was
    /// registered under the same commit/active-snapshot locks, preserving `AND CHAIN` atomicity.
    pub fn commit_explicit_transaction(
        &self,
        txn_id: TxnId,
        chain: bool,
    ) -> Result<Option<TxnId>, ExecuteError> {
        let txn_id = self.resolve_public_transaction_id(txn_id);
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if let Some(snapshot) = self.transaction_snapshot_handle(txn_id) {
            self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
            let _statement = snapshot
                .statement_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
            self.commit_explicit_transaction_statement_locked(txn_id, chain, &snapshot)
        } else {
            self.finish_transaction_context(txn_id, true, chain)
                .map_err(ExecuteError::Txn)
        }
    }

    pub(crate) fn commit_explicit_transaction_statement_locked(
        &self,
        txn_id: TxnId,
        chain: bool,
        snapshot: &Arc<TransactionSnapshot>,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        if !snapshot.transaction_delta_is_empty() {
            self.commit_transaction_delta(txn_id, chain, current_timestamp_micros())
        } else {
            self.finish_transaction_context(txn_id, true, chain)
                .map_err(ExecuteError::Txn)
        }
    }

    pub fn rollback_explicit_transaction(
        &self,
        txn_id: TxnId,
        chain: bool,
    ) -> Result<Option<TxnId>, ExecuteError> {
        let txn_id = self.resolve_public_transaction_id(txn_id);
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if let Some(snapshot) = self.transaction_snapshot_handle(txn_id) {
            self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
            let _statement = snapshot
                .statement_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
            self.rollback_explicit_transaction_statement_locked(txn_id, chain, &snapshot)
        } else {
            self.finish_transaction_context(txn_id, false, chain)
                .map_err(ExecuteError::Txn)
        }
    }

    pub(crate) fn rollback_explicit_transaction_statement_locked(
        &self,
        txn_id: TxnId,
        chain: bool,
        snapshot: &Arc<TransactionSnapshot>,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        self.finish_transaction_context(txn_id, false, chain)
            .map_err(ExecuteError::Txn)
    }

    /// Stage one DML statement in an explicit transaction. Preparation and every constraint/read
    /// bind to the retained generation plus prior private deltas. A successful statement publishes
    /// a new transaction-private shard map atomically; global WAL, MVCC, residency, and committed
    /// visibility remain untouched until COMMIT.
    pub fn execute_dml_in_transaction(
        &self,
        txn_id: TxnId,
        text: &str,
    ) -> Result<(), ExecuteError> {
        let txn_id = self.resolve_public_transaction_id(txn_id);
        let command = parse_command(text)?;
        if crate::engine_dml_concurrent::command_has_returning(&command) {
            return Err(crate::engine_dml_concurrent::discarded_returning_error());
        }
        self.execute_parsed_dml_in_transaction_with_result(txn_id, command)
            .map(|_| ())
    }

    pub fn execute_dml_in_transaction_with_result(
        &self,
        txn_id: TxnId,
        text: &str,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        let txn_id = self.resolve_public_transaction_id(txn_id);
        let command = parse_command(text)?;
        self.execute_parsed_dml_in_transaction_with_result(txn_id, command)
    }

    pub(crate) fn execute_parsed_dml_in_transaction_with_result(
        &self,
        txn_id: TxnId,
        command: Command,
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
        self.execute_parsed_dml_in_transaction_statement_locked(txn_id, command, &snapshot)
    }

    pub(crate) fn execute_parsed_dml_in_transaction_statement_locked(
        &self,
        txn_id: TxnId,
        command: Command,
        snapshot: &Arc<TransactionSnapshot>,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        self.execute_parsed_dml_in_transaction_statement_locked_with_sequence_parent(
            txn_id, command, snapshot, false,
        )
    }

    pub(crate) fn execute_parsed_dml_in_transaction_statement_locked_with_sequence_parent(
        &self,
        txn_id: TxnId,
        mut command: Command,
        snapshot: &Arc<TransactionSnapshot>,
        sequence_parent_autocommit: bool,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        self.legacy_lane_history_write_guard()
            .map_err(ExecuteError::Engine)?;
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        if !matches!(
            &command,
            Command::Insert(_) | Command::Update(_) | Command::Delete(_)
        ) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "explicit transaction staging accepts INSERT, UPDATE, or DELETE".to_string(),
            )));
        }
        if snapshot.characteristics.access == TransactionAccessMode::ReadOnly {
            return Err(ExecuteError::Unsupported(
                "cannot execute DML in a READ ONLY transaction".to_string(),
            ));
        }
        let statement_digest = transaction_statement_digest(&command)?;

        let table_name = match &command {
            Command::Insert(insert) => insert.table.clone(),
            Command::Update(update) => update.table.clone(),
            Command::Delete(delete) => delete.table.clone(),
            _ => unreachable!("DML shape checked above"),
        };
        let transaction_catalog = snapshot.transaction_catalog();
        let table = transaction_catalog
            .relational_catalog
            .get(&table_name)
            .cloned()
            .ok_or_else(|| ExecuteError::UndefinedRelation(table_name.clone()))?;
        let access_identities =
            table_access_dependency_identities(&transaction_catalog.relational_catalog, &table)?;
        // RR reads retain their historical catalog/OID binding, but writes may not target an
        // object that has since been renamed, dropped, or recreated under the same name.
        self.acquire_transaction_write_table_access_identities(snapshot, &access_identities)?;
        let sequence_access_identities = table
            .columns
            .iter()
            .filter_map(|column| match &column.default {
                Some(ColumnDefault::SequenceNextVal { sequence, .. }) => transaction_catalog
                    .relational_sequences
                    .get(sequence)
                    .map(|sequence| sequence.oid),
                None
                | Some(ColumnDefault::Literal(_))
                | Some(ColumnDefault::DeferredScalar { .. }) => None,
            })
            .collect::<BTreeSet<_>>();
        snapshot
            .table_access
            .acquire_shared(sequence_access_identities)?;

        let (
            generation,
            next_row_id,
            operation_ordinal,
            typed_statement_ordinal,
            expression_ordinal_base,
        ) = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let typed_statement_ordinal = u32::try_from(
                delta
                    .operations
                    .iter()
                    .filter(|operation| matches!(operation, TransactionOperation::TypedInsert(_)))
                    .count(),
            )
            .map_err(|_| {
                ExecuteError::Unsupported(
                    "transaction typed INSERT count exceeds typed WAL framing".to_string(),
                )
            })?;
            (
                delta.generation,
                delta.next_row_id,
                u32::try_from(delta.operations.len()).map_err(|_| {
                    ExecuteError::Unsupported(
                        "transaction operation count exceeds typed WAL framing".to_string(),
                    )
                })?,
                typed_statement_ordinal,
                u32::try_from(
                    delta
                        .sequence_value_references
                        .iter()
                        .filter(|reference| {
                            usize::try_from(reference.statement_ordinal).ok()
                                == usize::try_from(typed_statement_ordinal).ok()
                        })
                        .count(),
                )
                .map_err(|_| {
                    ExecuteError::Unsupported(
                        "transaction sequence expression count exceeds typed WAL framing"
                            .to_string(),
                    )
                })?,
            )
        };
        // Capture sequence-default requests from the original SQL shape before materialization.
        // Published transitions replace source cells below, but their receipts must still seal
        // against these immutable requests rather than making the typed carrier rediscover a
        // now-programmatic value.
        let prepared_typed_insert = if let Command::Insert(insert) = &command {
            let expected_input_width = if insert.columns.is_empty() {
                table.columns.len()
            } else {
                insert.columns.len()
            };
            let typed_private_shape = !insert.rows.is_empty()
                && table.columns.iter().all(|column| {
                    matches!(
                        column.ty,
                        SqlType::Int2
                            | SqlType::Int4
                            | SqlType::Date
                            | SqlType::Int8
                            | SqlType::Timestamp
                            | SqlType::Numeric { .. }
                            | SqlType::Uuid
                            | SqlType::Bool
                            | SqlType::Text
                    )
                })
                && insert
                    .rows
                    .iter()
                    .all(|row| row.len() == expected_input_width);
            if !typed_private_shape {
                return Err(ExecuteError::Unsupported(
                    "typed transaction INSERT cannot form a private device artifact; legacy WriteDelta fallback is removed"
                        .to_string(),
                ));
            }
            Some(
                crate::engine_insert_plan::PreparedInsertEffectPlan::prepare_explicit_on_locked_snapshot(
                    self,
                    txn_id,
                    Arc::clone(snapshot),
                    insert,
                    sequence_parent_autocommit,
                    None,
                )?
                .ok_or_else(|| {
                    ExecuteError::Unsupported(
                        "typed transaction INSERT semantic preparation declined; legacy WriteDelta fallback is removed"
                            .to_string(),
                    )
                })?,
            )
        } else {
            None
        };
        let typed_parent_request_digest = prepared_typed_insert
            .as_ref()
            .map(|prepared| prepared.parent_request_digest())
            .unwrap_or(statement_digest);

        if sequence_parent_autocommit {
            self.validate_sequence_default_parent_typed_identity(
                txn_id,
                typed_parent_request_digest,
            )?;
        }

        // Published sequence transitions remain the sole sequence authority. Materialize their
        // durable values after capture, then bind every resulting receipt to that same consumed
        // private artifact below; a typed stage never re-evaluates a default.
        let mut sequence_value_references = self.materialize_published_sequence_defaults(
            snapshot,
            &transaction_catalog,
            &table,
            &mut command,
            crate::engine_sequence_value::SequenceDefaultStatementIdentity {
                parent_txn_id: txn_id,
                parent_autocommit: sequence_parent_autocommit,
                statement_ordinal: typed_statement_ordinal,
                expression_ordinal_base,
                parent_request_digest: typed_parent_request_digest,
            },
        )?;

        // Once a fixed-width, BOOL, or TEXT relation enters this typed vertical, unsupported breadth
        // fails before private-generation side effects. It never converts the consumed typed
        // batch back into a legacy `WriteDelta`.
        if let Command::Insert(insert) = &command {
            if let Some(prepared) = prepared_typed_insert {
                let sealed = prepared.seal_live_explicit(&sequence_value_references)?;
                let batch = sealed.batch;
                let private_sequence_advances = sealed.private_sequence_advances;
                batch
                    .validate_live_required_input_policy()
                    .map_err(ExecuteError::Engine)?;
                if batch.supports_transaction_private_fixed_stage(&table)
                    && batch.domain_dependencies_match_current_catalog(&transaction_catalog)
                {
                    // Reuse the sole device-native local CHECK and same-batch key proof
                    // before any private GPU generation is constructed. Current-resident
                    // UNIQUE history remains revalidated from the final typed record at
                    // the canonical COMMIT cut below.
                    crate::engine_insert_plan::pre_wal_constraints::validate_before_transaction_typed_stage(
                                self,
                                &batch,
                                &transaction_catalog,
                            )
                            .map_err(ExecuteError::Engine)?;
                    let mut staged = batch.into_transaction_private_fixed_stage(
                        &table,
                        statement_digest,
                        snapshot.boundary,
                        next_row_id,
                    )?;
                    #[cfg(feature = "probe-timing")]
                    {
                        let [constructor_validate, unique_slots, payload_digest] =
                            staged.private_probe_nanos();
                        self.record_insert_probe_staged_payload_nanos([
                            constructor_validate,
                            unique_slots,
                            payload_digest,
                            0,
                            0,
                        ]);
                    }
                    #[cfg(feature = "probe-timing")]
                    {
                        let (record_nanos, final_image_nanos, resident_source_nanos) =
                            staged.codec5_sources.probe_seal_nanos();
                        self.record_insert_probe_codec5_seal_nanos(
                            record_nanos,
                            final_image_nanos,
                            resident_source_nanos,
                        );
                    }
                    // Bind the full parent/child catalog closure at the same statement
                    // snapshot as the consumed typed rows.  COMMIT uses these exact
                    // dependency stamps to reject FK provider/consumer races before its
                    // device verdict over the canonical final record.
                    staged.bind_foreign_key_dependencies(&transaction_catalog)?;
                    staged.bind_private_sequence_advances(&private_sequence_advances)?;
                    staged.bind_operation_ordinal(operation_ordinal);
                    self.bind_sequence_default_typed_insert_rows(
                        &table,
                        &staged,
                        &mut sequence_value_references,
                    )?;
                    let result = self.stage_typed_insert_in_transaction(
                        snapshot,
                        &table,
                        generation,
                        staged,
                        sequence_value_references,
                        private_sequence_advances,
                        &insert.returning,
                    );
                    return result;
                }
                if batch.supports_transaction_private_dense_variable_stage(&table)
                    && batch.domain_dependencies_match_current_catalog(&transaction_catalog)
                {
                    crate::engine_insert_plan::pre_wal_constraints::validate_before_transaction_typed_stage(
                                self,
                                &batch,
                                &transaction_catalog,
                            )
                            .map_err(ExecuteError::Engine)?;
                    let mut staged = batch.into_transaction_private_dense_variable_stage(
                        &table,
                        statement_digest,
                        snapshot.boundary,
                        next_row_id,
                    )?;
                    #[cfg(feature = "probe-timing")]
                    {
                        let [constructor_validate, unique_slots, payload_digest] =
                            staged.private_probe_nanos();
                        self.record_insert_probe_staged_payload_nanos([
                            constructor_validate,
                            unique_slots,
                            payload_digest,
                            0,
                            0,
                        ]);
                    }
                    #[cfg(feature = "probe-timing")]
                    {
                        let (record_nanos, final_image_nanos, resident_source_nanos) =
                            staged.codec5_sources.probe_seal_nanos();
                        self.record_insert_probe_codec5_seal_nanos(
                            record_nanos,
                            final_image_nanos,
                            resident_source_nanos,
                        );
                    }
                    staged.bind_foreign_key_dependencies(&transaction_catalog)?;
                    staged.bind_private_sequence_advances(&private_sequence_advances)?;
                    staged.bind_operation_ordinal(operation_ordinal);
                    self.bind_sequence_default_typed_insert_rows(
                        &table,
                        &staged,
                        &mut sequence_value_references,
                    )?;
                    let result = self.stage_typed_insert_in_transaction(
                        snapshot,
                        &table,
                        generation,
                        staged,
                        sequence_value_references,
                        private_sequence_advances,
                        &insert.returning,
                    );
                    return result;
                }
            }
            return Err(ExecuteError::Unsupported(
                "typed transaction INSERT could not form a private device artifact; legacy WriteDelta fallback is removed"
                    .to_string(),
            ));
        }
        let _scope = self.enter_transaction_read(Arc::clone(snapshot));
        let dml_snapshot = DmlReadSnapshot {
            commit_seq: snapshot.boundary,
            next_row_id,
        };
        let prepared = self.prepare_dml(&command, dml_snapshot)?;
        self.bind_sequence_default_insert_rows(&table, &prepared, &mut sequence_value_references)?;
        let prepared_sequence_state = match &prepared.mutation {
            PreparedMutation::Insert { seq_advances, .. } => seq_advances.clone(),
            PreparedMutation::Update { .. } | PreparedMutation::Delete { .. } => BTreeMap::new(),
        };
        let sequence_input_oids = prepared_sequence_state
            .keys()
            .map(|name| {
                transaction_catalog
                    .relational_sequences
                    .get(name)
                    .map(|sequence| (name.clone(), sequence.oid))
                    .ok_or_else(|| {
                        ExecuteError::Serialization(format!(
                            "transaction sequence input \"{name}\" left its private catalog generation"
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let prepared_sequence_state_by_oid = prepared_sequence_state
            .iter()
            .map(|(name, state)| {
                (
                    *sequence_input_oids
                        .get(name)
                        .expect("captured every prepared sequence input"),
                    *state,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let rows_affected = prepared.rows_affected();
        let returning = self.project_dml_returning(&command, &prepared, snapshot.boundary)?;

        self.validate_transaction_delta_residency(
            &table,
            snapshot
                .catalog
                .relational_catalog
                .contains_key(&table_name),
        )?;

        let current_shards = snapshot.transaction_shards();
        let current_cold_chunks = snapshot.transaction_cold_chunks();
        let mut next_shards = (*current_shards).clone();
        let mut next_cold_chunks = (*current_cold_chunks).clone();
        let mut gpu_reservation = TransactionGpuReservation::new(self);
        self.apply_transaction_private_delta(
            &table,
            &prepared,
            snapshot.boundary,
            &mut next_shards,
            &mut next_cold_chunks,
            &mut gpu_reservation,
        )?;
        let next_private_gpu_bytes =
            transaction_private_shard_bytes(snapshot.resident_shards.as_ref(), &next_shards);

        // Compare-and-publish under the transaction mutex. Connection/session execution is ordered,
        // but the generation check also makes accidental same-transaction concurrent use fail loud
        // instead of losing one private statement.
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if delta.generation != generation {
            return Err(ExecuteError::Serialization(
                "concurrent statements attempted to publish the same transaction delta".to_string(),
            ));
        }
        gpu_reservation.ensure_replacement_admitted(
            &delta.private_gpu_bytes_by_gpu,
            &next_private_gpu_bytes,
        )?;
        u32::try_from(delta.operations.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "transaction operation count exceeds typed WAL framing".to_string(),
            )
        })?;
        delta.next_row_id = delta.next_row_id.saturating_add(prepared.rows_consumed);
        delta.sequence_state.extend(prepared_sequence_state);
        delta
            .sequence_state_by_oid
            .extend(prepared_sequence_state_by_oid);
        delta
            .sequence_value_references
            .extend(sequence_value_references);
        delta
            .operations
            .push(TransactionOperation::Row(Arc::new(StagedRowOperation {
                statement_digest,
                sequence_input_oids,
                delta: prepared,
            })));
        delta.write_set = final_transaction_write_set(&delta.operations);
        delta.publish_resident_shards(Arc::new(next_shards));
        delta.publish_streaming_cold_chunks(Arc::new(next_cold_chunks));
        delta.generation = delta.generation.saturating_add(1);
        // The transaction statement lock excludes every other same-transaction reader or writer.
        // Release this statement's old-map pin before dropping superseded allocation charges, then
        // atomically convert the temporary reservations into the exact replacement-generation account.
        drop(current_shards);
        gpu_reservation
            .replace_charges(&mut delta.private_gpu_bytes_by_gpu, next_private_gpu_bytes);
        Ok(DmlExecutionResult {
            rows_affected,
            returning,
        })
    }

    /// Publish one immutable typed INSERT artifact into the transaction's private GPU generation.
    /// The global residency map remains untouched; Sync later binds its exact row images through
    /// the established transaction WAL/status/apply/publication terminal.
    #[allow(clippy::too_many_arguments)] // direct transaction state inputs avoid an ambient mutable context
    fn stage_typed_insert_in_transaction(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        table: &RelationalTable,
        generation: u64,
        mut staged: StagedTypedInsert,
        sequence_value_references: Vec<BinarySequenceValueReference>,
        private_sequence_advances: Vec<TypedPrivateSequenceAdvance>,
        returning: &[String],
    ) -> Result<DmlExecutionResult, ExecuteError> {
        staged.returning_requested = !returning.is_empty();
        let _scope = self.enter_transaction_read(Arc::clone(snapshot));
        if self.table_chunk_authoritative(&table.name).is_some() {
            return Err(ExecuteError::Unsupported(
                "typed transaction INSERT does not yet stage a cold-chunk private generation"
                    .to_string(),
            ));
        }
        self.validate_transaction_delta_residency(
            table,
            snapshot
                .catalog
                .relational_catalog
                .contains_key(&table.name),
        )?;
        let transaction_catalog = snapshot.transaction_catalog();
        for advance in &private_sequence_advances {
            let sequence = transaction_catalog
                .relational_sequences
                .get(&advance.sequence_name)
                .ok_or_else(|| {
                    ExecuteError::Serialization(
                        "typed private sequence advance left the transaction catalog".to_string(),
                    )
                })?;
            if sequence.oid != advance.sequence_oid {
                return Err(ExecuteError::Serialization(
                    "typed private sequence advance changed stable identity before staging"
                        .to_string(),
                ));
            }
        }
        let current_shards = snapshot.transaction_shards();
        let current_cold_chunks = snapshot.transaction_cold_chunks();
        let mut next_shards = (*current_shards).clone();
        let mut gpu_reservation = TransactionGpuReservation::new(self);
        #[cfg(feature = "probe-timing")]
        let probe_private_overlay_started = std::time::Instant::now();
        self.append_transaction_typed_insert_shard(
            table,
            &staged,
            &mut next_shards,
            &mut gpu_reservation,
        )?;
        let returning_shard = next_shards
            .get(&table.name)
            .and_then(|shards| shards.last())
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "typed transaction INSERT lost its private GPU shard before RETURNING"
                        .to_string(),
                ))
            })?;
        // Project directly from the just-reserved private shard. The general GPU executor owns
        // only its normal result buffers; no row image is decoded and no second source upload is
        // constructed. A projection failure drops this unpublished shard with its reservation.
        let returning = self.project_typed_insert_returning_from_private_shard(
            table,
            returning,
            returning_shard,
            staged.read_snapshot,
        )?;
        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_transaction_private_overlay_materialize_nanos(
            probe_private_overlay_started.elapsed().as_nanos() as u64,
        );
        let private_foreign_key_validation_required = !table.foreign_keys.is_empty();
        let next_shards = Arc::new(next_shards);
        let next_private_gpu_bytes = transaction_private_shard_bytes(
            snapshot.resident_shards.as_ref(),
            next_shards.as_ref(),
        );
        let rows_affected = staged.rows_consumed();
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if delta.generation != generation {
            return Err(ExecuteError::Serialization(
                "concurrent statements attempted to publish the same typed transaction generation"
                    .to_string(),
            ));
        }
        gpu_reservation.ensure_replacement_admitted(
            &delta.private_gpu_bytes_by_gpu,
            &next_private_gpu_bytes,
        )?;
        // The private suffix is immutable. The public prefix may share an open allocation with a
        // concurrent writer, so it remains subject to the final commit-cut revalidation below.
        let table_reset_in_private_generation = delta.operations.iter().any(|operation| {
            matches!(operation, TransactionOperation::TableReset(reset) if reset.table == table.name)
        });
        let private_shard_start = if table_reset_in_private_generation {
            0
        } else {
            snapshot
                .resident_shards
                .get(&table.name)
                .map_or(0, Vec::len)
        };
        let private_unique_shards = next_shards
            .get(&table.name)
            .and_then(|shards| shards.get(private_shard_start..))
            .unwrap_or_default()
            .to_vec();
        let has_unique_index = table.indexes.iter().any(|index| index.unique);
        // Preserve immediate duplicate errors against a relation that existed at this statement
        // snapshot. A newly-created or reset relation has no public rows participating in the
        // check; its private suffix is sufficient.
        let public_unique_validation_required = has_unique_index
            && !table_reset_in_private_generation
            && snapshot
                .catalog
                .relational_catalog
                .contains_key(&table.name);
        let private_unique_validation_required =
            has_unique_index && private_unique_shards.len() > 1;
        // A second typed INSERT into a private UNIQUE/PRIMARY relation must receive PostgreSQL's
        // statement-local 23505, and an immediate FK must receive 23503 when its own statement
        // ends, not later at the transaction terminal. The new immutable shard is temporarily the
        // transaction's sole proven generation while the existing GPU validators observe it. The
        // transaction statement lock excludes every other reader/writer of this overlay; a failed
        // proof restores the exact predecessor before returning, so it neither creates a
        // row/sequence effect nor reaches the codec-5 terminal.
        if public_unique_validation_required
            || private_unique_validation_required
            || private_foreign_key_validation_required
        {
            delta.publish_resident_shards(Arc::clone(&next_shards));
            drop(delta);
            let validation = (|| {
                if public_unique_validation_required {
                    let typed_tables = BTreeSet::from([table.name.clone()]);
                    match self.validate_transaction_final_typed_unique_indexes_device(
                        snapshot,
                        &transaction_catalog,
                        &typed_tables,
                    ) {
                        // This is the one legitimate pre-WAL race: an external in-place append
                        // advanced the inherited public allocation after READ COMMITTED capture.
                        // The terminal rebase validates that prefix under the canonical cut. A
                        // private-suffix mismatch and every other error remain fail-closed here.
                        Err(ExecuteError::Serialization(message))
                            if message == format!(
                                "relation \"{}\" has transaction-pinned GPU shard count-header drift",
                                table.name
                            ) => {}
                        other => other?,
                    }
                }
                if private_unique_validation_required {
                    self.validate_transaction_private_typed_unique_indexes_device(
                        snapshot,
                        table,
                        &private_unique_shards,
                    )?;
                }
                if private_foreign_key_validation_required {
                    self.validate_staged_typed_insert_foreign_keys_device(
                        snapshot,
                        &transaction_catalog,
                        &staged,
                    )?;
                }
                Ok(())
            })();
            if let Err(error) = validation {
                let mut delta = snapshot
                    .delta
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if delta.generation != generation {
                    delta.publish_resident_shards(current_shards);
                    return Err(ExecuteError::Serialization(
                        "concurrent statements changed the private generation during typed UNIQUE validation"
                            .to_string(),
                    ));
                }
                delta.publish_resident_shards(current_shards);
                return Err(error);
            }
            delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if delta.generation != generation {
                delta.publish_resident_shards(current_shards);
                return Err(ExecuteError::Serialization(
                    "concurrent statements changed the private generation during typed UNIQUE validation"
                        .to_string(),
                ));
            }
        }
        delta.next_row_id = delta
            .next_row_id
            .checked_add(rows_affected)
            .ok_or_else(|| {
                ExecuteError::Unsupported(
                    "transaction provisional row identity space exhausted".to_string(),
                )
            })?;
        delta
            .operations
            .push(TransactionOperation::TypedInsert(Arc::new(staged)));
        delta
            .sequence_value_references
            .extend(sequence_value_references);
        for advance in private_sequence_advances {
            delta
                .sequence_state
                .insert(advance.sequence_name, advance.next_state);
            delta
                .sequence_state_by_oid
                .insert(advance.sequence_oid, advance.next_state);
        }
        delta.write_set = final_transaction_write_set(&delta.operations);
        delta.publish_resident_shards(next_shards);
        // The typed scope leaves cold state unchanged, but publishes the paired witness with the
        // new shard generation so readers cannot observe a torn private generation.
        delta.publish_streaming_cold_chunks(current_cold_chunks);
        delta.generation = delta.generation.saturating_add(1);
        drop(current_shards);
        gpu_reservation
            .replace_charges(&mut delta.private_gpu_bytes_by_gpu, next_private_gpu_bytes);
        Ok(DmlExecutionResult {
            rows_affected,
            returning,
        })
    }

    /// Immediate foreign keys use the just-constructed private GPU generation at statement
    /// admission.  The terminal validator remains responsible for final-image and concurrent
    /// parent/child conflicts at COMMIT; this narrow pre-terminal verdict only preserves the
    /// PostgreSQL statement boundary for an INSERT that already has a device plan.
    fn validate_staged_typed_insert_foreign_keys_device(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        catalog: &CatalogSnapshot,
        staged: &StagedTypedInsert,
    ) -> Result<(), ExecuteError> {
        let child = catalog
            .relational_catalog
            .get(&staged.table)
            .ok_or_else(|| {
                ExecuteError::Serialization(
                    "typed transaction INSERT target left the private catalog before FK validation"
                        .to_string(),
                )
            })?;
        if child.oid != staged.table_oid
            || table_schema_digest(child)? != staged.table_schema_digest
        {
            return Err(ExecuteError::Serialization(
                "typed transaction INSERT target schema changed before FK validation".to_string(),
            ));
        }
        for foreign_key in &child.foreign_keys {
            let parent = catalog
                .relational_catalog
                .get(&foreign_key.referenced_table)
                .ok_or_else(|| {
                    ExecuteError::Serialization(
                        "typed transaction INSERT foreign-key provider left the private catalog"
                            .to_string(),
                    )
                })?;
            let child_idx =
                relational_column_index(child, &foreign_key.column).map_err(|error| {
                    ExecuteError::Engine(EngineError::ApplyFailed(error.to_string()))
                })?;
            let parent_idx = relational_column_index(parent, &foreign_key.referenced_column)
                .map_err(|error| {
                    ExecuteError::Engine(EngineError::ApplyFailed(error.to_string()))
                })?;
            for row_ordinal in 0..staged.rows_consumed() as usize {
                let row = staged.private_row_values(child, row_ordinal)?;
                let value = &row[child_idx];
                if matches!(value, SqlValue::Null) {
                    continue;
                }
                let provider = self
                    .device_visible_row_with_value(
                        parent,
                        StorageVisibility {
                            read_txn_id: snapshot.boundary,
                        },
                        parent_idx,
                        value,
                        None,
                    )
                    .ok_or_else(|| {
                        ExecuteError::Serialization(format!(
                            "device foreign-key provider verdict unavailable for relation \"{}\"",
                            parent.name
                        ))
                    })?;
                if !provider {
                    return Err(ExecuteError::Engine(EngineError::ForeignKeyViolation(
                        format!(
                            "insert or update on table \"{}\" violates foreign key constraint \"{}\"",
                            child.name, foreign_key.name
                        ),
                    )));
                }
            }
        }
        Ok(())
    }

    /// Durably publish all staged statements as ONE resolved WAL record and ONE MVCC generation.
    /// The commit lock closes catalog/conflict races, final insert identities are claimed once, and
    /// recovery applies the same ordered row mutations without re-evaluating predicates.
    pub(crate) fn commit_transaction_delta(
        &self,
        txn_id: TxnId,
        chain: bool,
        timestamp_micros: u64,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.commit_transaction_delta_inner(txn_id, chain, timestamp_micros, None)
    }

    pub(crate) fn commit_claimed_transaction_delta(
        &self,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
        timestamp_micros: u64,
    ) -> Result<(), ExecuteError> {
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        let _statement = snapshot
            .statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        self.commit_transaction_delta_inner(txn_id, false, timestamp_micros, Some(request_digest))?;
        Ok(())
    }

    fn commit_transaction_delta_inner(
        &self,
        txn_id: TxnId,
        chain: bool,
        timestamp_micros: u64,
        request_digest_override: Option<gpu_db_wal::CanonicalDigest>,
    ) -> Result<Option<TxnId>, ExecuteError> {
        #[cfg(feature = "probe-timing")]
        let probe_terminal_pre_wal_started = std::time::Instant::now();
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        // A classic wave installs its device generation before its durability tail publishes
        // `committed_seq`. Explicit-transaction row/unique/FK validation must not inspect that
        // applied-but-unpublished interval. Drain first, then prove under the commit lock that no
        // sequencer handed off a new tail in the acquisition gap; retry until the cut is settled.
        #[cfg(test)]
        self.run_transaction_terminal_pre_commit_lock_hook();
        let mut commit = self
            .commit_state_after_wave_quiescence()
            .map_err(ExecuteError::Engine)?;
        self.legacy_lane_history_write_guard()
            .map_err(ExecuteError::Engine)?;
        if commit.txn_manager.state(txn_id) != Some(TxnState::Active) {
            return Err(ExecuteError::Txn(TxnError::NotActive(txn_id)));
        }
        if commit.transaction_status.contains_key(&txn_id) {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "transaction id {txn_id} was already claimed by another write strategy while the explicit transaction was active"
            ))));
        }
        // The transaction can have staged against the prior immutable descriptor while another
        // commit subsequently appended in place to that descriptor's device allocation. Rebase
        // READ COMMITTED private state only after this lock has excluded that header/descriptor
        // publication window, so final validation never observes a count header newer than its
        // retained shard descriptor.
        let snapshot = self.refresh_transaction_snapshot_at_commit_boundary(txn_id, &snapshot)?;
        let (operations, catalog_base, sequence_value_references) = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                delta.operations.clone(),
                delta.catalog_base.clone(),
                delta.sequence_value_references.clone(),
            )
        };
        if operations.is_empty() && sequence_value_references.is_empty() {
            drop(commit);
            return self
                .finish_transaction_context(txn_id, true, chain)
                .map_err(ExecuteError::Txn);
        }
        for (ordinal, operation) in operations.iter().enumerate() {
            let stored = match operation {
                TransactionOperation::Catalog(staged) => Some(staged.ordinal),
                TransactionOperation::TableReset(reset) => Some(reset.ordinal),
                TransactionOperation::Row(_) => None,
                TransactionOperation::TypedInsert(staged) => Some(staged.operation_ordinal),
            };
            if stored.is_some_and(|stored| usize::try_from(stored).ok() != Some(ordinal)) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transaction operation lost its global statement ordinal".to_string(),
                )));
            }
        }
        let catalog_commands = operations
            .iter()
            .filter_map(|operation| match operation {
                TransactionOperation::Catalog(staged) => Some(staged.as_ref().clone()),
                TransactionOperation::Row(_)
                | TransactionOperation::TableReset(_)
                | TransactionOperation::TypedInsert(_) => None,
            })
            .collect::<Vec<_>>();
        let all_deltas = transaction_row_deltas(&operations);
        let (deltas, table_resets) = final_transaction_operations(&operations);
        let final_staged_rows = final_transaction_row_operations(&operations);
        // The ordinary autocommit INSERT path is a one-statement private overlay followed by
        // this terminal, rather than a wave item. Count it only after canonical apply/publish,
        // matching the successful-wave accounting boundary.
        #[cfg(feature = "probe-timing")]
        let typed_insert_rows = final_staged_rows
            .iter()
            .filter_map(|operation| match operation {
                FinalTransactionRowOperation::TypedInsert(staged) => Some(staged.rows_consumed()),
                FinalTransactionRowOperation::Legacy(_) => None,
            })
            .collect::<Vec<_>>();
        let has_sequence_resets = operations.iter().any(|operation| {
            matches!(
                operation,
                TransactionOperation::TableReset(reset)
                    if reset.sequence_reset_identity.is_some()
            )
        });
        if !catalog_commands.is_empty() || has_sequence_resets {
            let base = catalog_base.ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "transactional catalog operation envelope lost its base generation".to_string(),
                ))
            })?;
            if !Self::catalogs_same_ignoring_referenced_sequence_values(
                self.read_state.latest_catalog().as_ref(),
                base.as_ref(),
                &sequence_value_references,
            ) {
                return Err(ExecuteError::Serialization(
                    "catalog contents changed after transactional DDL staging".to_string(),
                ));
            }
            debug_assert!(catalog_commands.iter().all(|operation| {
                Self::transaction_catalog_command_is_supported(&operation.command)
            }));
        }
        let transaction_catalog = snapshot.transaction_catalog();
        for staged in &final_staged_rows {
            match staged {
                FinalTransactionRowOperation::Legacy(staged) => {
                    let delta = &staged.delta;
                    for (name, expected) in &delta.catalog_dependencies {
                        if transaction_catalog
                            .relational_catalog
                            .get(name)
                            .is_none_or(|observed| {
                                !same_transaction_row_catalog_dependency(
                                    expected,
                                    observed,
                                    &staged.sequence_input_oids,
                                    &transaction_catalog,
                                    &catalog_commands,
                                )
                            })
                        {
                            return Err(ExecuteError::Serialization(format!(
                                "catalog dependency \"{name}\" changed after transaction statement snapshot {}",
                                delta.read_snapshot
                            )));
                        }
                    }
                }
                FinalTransactionRowOperation::TypedInsert(staged) => {
                    if transaction_catalog.commit_seq != staged.prepared_catalog_seq {
                        return Err(ExecuteError::Serialization(
                            "typed INSERT prepared catalog generation changed before transaction commit"
                                .to_string(),
                        ));
                    }
                    for (name, expected) in &staged.catalog_dependencies {
                        if transaction_catalog
                            .relational_catalog
                            .get(name)
                            .is_none_or(|observed| {
                                !same_transaction_row_catalog_dependency(
                                    expected,
                                    observed,
                                    &BTreeMap::new(),
                                    &transaction_catalog,
                                    &catalog_commands,
                                )
                            })
                        {
                            return Err(ExecuteError::Serialization(format!(
                                "typed INSERT catalog dependency \"{name}\" changed after transaction statement snapshot {}",
                                staged.read_snapshot
                            )));
                        }
                    }
                }
            }
        }
        let published_catalog = self.read_state.latest_catalog();
        for delta in &deltas {
            if let Some((name, expected)) =
                delta.catalog_dependencies.iter().find(|(name, expected)| {
                    snapshot.catalog.relational_catalog.get(name.as_str()) == Some(*expected)
                        && published_catalog.relational_catalog.get(name.as_str())
                            != Some(*expected)
                })
            {
                return Err(ExecuteError::Serialization(format!(
                    "catalog dependency \"{name}\" changed stable identity or schema before transaction commit (expected OID {})",
                    expected.oid
                )));
            }
        }
        for staged in final_staged_rows
            .iter()
            .filter_map(|operation| match operation {
                FinalTransactionRowOperation::TypedInsert(staged) => Some(staged),
                FinalTransactionRowOperation::Legacy(_) => None,
            })
        {
            if let Some((name, expected)) =
                staged.catalog_dependencies.iter().find(|(name, expected)| {
                    snapshot.catalog.relational_catalog.get(name.as_str()) == Some(*expected)
                        && published_catalog.relational_catalog.get(name.as_str())
                            != Some(*expected)
                })
            {
                return Err(ExecuteError::Serialization(format!(
                    "typed INSERT catalog dependency \"{name}\" changed stable identity or schema before transaction commit (expected OID {})",
                    expected.oid
                )));
            }
        }
        let reset_catalog = if catalog_commands.is_empty() && !has_sequence_resets {
            published_catalog.as_ref()
        } else {
            transaction_catalog.as_ref()
        };
        self.validate_transaction_table_resets(reset_catalog, &table_resets, &commit.ledger)?;
        let staged_tables = final_staged_rows
            .iter()
            .map(|operation| match operation {
                FinalTransactionRowOperation::Legacy(staged) => match &staged.delta.mutation {
                    PreparedMutation::Insert { table, .. }
                    | PreparedMutation::Update { table, .. }
                    | PreparedMutation::Delete { table, .. } => table,
                },
                FinalTransactionRowOperation::TypedInsert(staged) => &staged.table,
            })
            .chain(table_resets.iter().map(|reset| &reset.table))
            .collect::<BTreeSet<_>>();
        if let Some(table) = staged_tables.iter().find(|table| {
            snapshot
                .catalog
                .relational_catalog
                .contains_key(table.as_str())
                && !snapshot
                    .chunk_authoritative_tables
                    .contains_key(table.as_str())
                && self.table_chunk_authoritative(table).is_some()
        }) {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{table}\" became cold-chunk authoritative after transaction staging"
            )));
        }
        if let Some(table) = staged_tables.iter().find(|table| {
            snapshot
                .catalog
                .relational_catalog
                .contains_key(table.as_str())
                && !self.table_has_live_dml_generation(table)
        }) {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{table}\" lost its globally authoritative GPU generation after transaction staging"
            )));
        }

        // A later statement may update/delete a row inserted earlier in this transaction. Its
        // provisional row key is not a real conflict point: nobody outside this transaction can
        // observe that entity, and COMMIT assigns it a fresh globally-claimed identity.
        let provisional_inserts = Self::transaction_operation_insert_identities(&operations)?;
        self.validate_transaction_device_row_conflicts(&snapshot, &deltas, &provisional_inserts)?;

        // The canonical commit mutex excludes every other row-id claimant. Use its current value as
        // the tentative record base, validate the complete unique/FK outcome, and let canonical
        // apply advance to the encoded high-water. A rejected pre-WAL transaction therefore cannot
        // leak process-wide row identities.
        let insert_count = provisional_inserts.len() as u64;
        let final_base = self.read_state.mvcc.current_row_id();
        let allocator_high_water = final_base.checked_add(insert_count).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "transaction row identity space exhausted".to_string(),
            ))
        })?;
        // Decide the single codec-5 terminal from immutable staged facts before constructing any
        // legacy resolved-record or binary-payload state. A selected typed transaction remains on
        // its existing overlay -> DeviceInsertPlan -> codec-5 lifecycle throughout this finalizer.
        let codec5_predecessor_roots = self.read_state.typed_generation_roots.load_full();
        #[cfg(feature = "probe-timing")]
        let probe_codec5_terminal_select_started = std::time::Instant::now();
        let codec5_staged = operations
            .iter()
            .any(|operation| matches!(operation, TransactionOperation::TypedInsert(_)))
            .then(|| {
                codec5_operation::compile_insert_bearing_codec5_staged(
                    &operations,
                    all_deltas.len(),
                    table_resets.len(),
                    &final_staged_rows,
                    &catalog_commands,
                    &transaction_catalog,
                    &sequence_value_references,
                    &codec5_predecessor_roots,
                    false,
                )
            })
            .transpose()?;
        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_codec5_materialization_nanos([
            0,
            0,
            probe_codec5_terminal_select_started.elapsed().as_nanos() as u64,
            0,
        ]);
        let codec5_sequence_value_references = codec5_staged
            .as_ref()
            .map(|staged| {
                let mut references = sequence_value_references.clone();
                staged
                    .bind_published_sequence_final_values(&mut references, &transaction_catalog)?;
                Self::resolved_sequence_value_references(
                    &provisional_inserts,
                    final_base,
                    &references,
                )
            })
            .transpose()?;
        let codec5_catalog_composition = codec5_staged
            .as_ref()
            .map(|staged| {
                codec5_operation::prepare_catalog_composition(
                    self,
                    txn_id,
                    staged,
                    &operations,
                    &final_staged_rows,
                    &catalog_commands,
                    &table_resets,
                    &transaction_catalog,
                    &provisional_inserts,
                    final_base,
                    codec5_sequence_value_references.as_deref().unwrap_or(&[]),
                )
            })
            .transpose()?
            .flatten();
        let mut legacy_record = if codec5_staged.is_none() {
            Some(Self::resolved_transaction_record(
                &final_staged_rows,
                &provisional_inserts,
                final_base,
                allocator_high_water,
                &sequence_value_references,
            )?)
        } else {
            None
        };
        let cold_index_candidate_tables = if let Some(record) = legacy_record.as_mut() {
            Self::bind_transaction_record_catalog_envelope(
                record,
                &operations,
                &catalog_commands,
                &table_resets,
                &transaction_catalog,
            )?;
            Self::bind_sequence_reference_final_dispositions(record, &transaction_catalog)?;
            let candidates = record
                .index_lifecycle_operations
                .iter()
                .flat_map(|operation| &operation.targets)
                .filter_map(|target| target.owner_name.clone())
                .collect::<BTreeSet<_>>();
            if !record.catalog_commands.is_empty() || !record.sequence_reset_operations.is_empty() {
                self.validate_transaction_catalog_before_wal(
                    record.catalog_epoch,
                    &record.catalog_commands,
                    TransactionCatalogEnvelopeSlices {
                        view_operations: &record.view_operations,
                        view_lifecycle_operations: &record.view_lifecycle_operations,
                        index_lifecycle_operations: &record.index_lifecycle_operations,
                        sequence_lifecycle_operations: &record.sequence_lifecycle_operations,
                        sequence_reset_operations: &record.sequence_reset_operations,
                        operation_order: &record.operation_order,
                        catalog_output: record.catalog_output.as_ref(),
                        sequence_input_oids: &record.sequence_input_oids,
                        sequence_value_references: &record.sequence_value_references,
                    },
                    &transaction_catalog,
                    false,
                )?;
            }
            self.validate_transaction_surviving_created_unique_indexes_device(
                &snapshot,
                &transaction_catalog,
                &record.catalog_commands,
                &record.index_lifecycle_operations,
            )?;
            candidates
        } else {
            BTreeSet::new()
        };
        let reset_tables = table_resets
            .iter()
            .map(|reset| reset.table.clone())
            .collect::<BTreeSet<_>>();
        let typed_tables = final_staged_rows
            .iter()
            .filter_map(|operation| match operation {
                FinalTransactionRowOperation::TypedInsert(staged) => Some(staged.table.clone()),
                FinalTransactionRowOperation::Legacy(_) => None,
            })
            .collect::<BTreeSet<_>>();
        if !typed_tables.is_empty() {
            self.validate_transaction_final_typed_unique_indexes_device(
                &snapshot,
                &transaction_catalog,
                &typed_tables,
            )?;
        }
        // First reject a duplicate that survives in this transaction's final GPU generation as
        // the user-visible 23505 violation. The subsequent history check still owns the distinct
        // first-committer-wins case: a key claimed and then released since this snapshot has no
        // final duplicate, so it remains a retryable serialization conflict rather than being
        // relabelled as a uniqueness error.
        self.validate_transaction_device_unique_conflicts(
            &snapshot,
            &deltas,
            &final_staged_rows,
            legacy_record.as_ref(),
            &reset_tables,
        )?;
        self.validate_transaction_device_foreign_key_conflicts(
            &snapshot,
            &deltas,
            &final_staged_rows,
            legacy_record.as_ref(),
            &provisional_inserts,
            final_base,
            &commit.ledger,
        )?;
        let isolation = match snapshot.characteristics.isolation {
            TransactionIsolation::ReadUncommitted | TransactionIsolation::ReadCommitted => {
                gpu_db_wal::CanonicalIsolation::ReadCommitted
            }
            TransactionIsolation::RepeatableRead => gpu_db_wal::CanonicalIsolation::RepeatableRead,
            TransactionIsolation::Serializable => {
                return Err(ExecuteError::Unsupported(
                    "SERIALIZABLE transactions are not implemented".to_string(),
                ));
            }
        };
        // Selection remains in the generic finalizer and happens only after every common
        // constraint/catalog validation above. Typed INSERT contributions may span multiple tables;
        // their stable-id-ordered CUDA generations remain private until one root-map successor is
        // published. Both a claimed
        // autocommit request and an explicit COMMIT bind their existing transaction identity to
        // the same writer; neither mode mints a statement-local retry authority. Its first
        // indexed cut accepts exactly one existing maintained named index through the same
        // finalizer. UNIQUE/PRIMARY KEY verdicts have already closed on the GPU above and are
        // serialized as S7 guards; they do not select another terminal. The binary transaction
        // body below is reserved for non-INSERT UPDATE/DELETE compatibility and historical WAL;
        // a live typed INSERT is either codec-5 or rejected before any payload is constructed.
        let expected_commit_seq = commit.repl.peek_next_index();
        let (catalog_epoch, catalog_digest) = Self::canonical_catalog_boundary(
            commit.canonical_identity,
            commit
                .wal
                .canonical_catalog_tail()
                .map_err(ExecuteError::Engine)?,
        )
        .map_err(ExecuteError::Engine)?;
        let payload = legacy_record
            .as_ref()
            .map(|record| {
                try_encode_binary_transaction(record)
                    .map(Arc::<[u8]>::from)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::Durability(
                            "resolved transaction WAL record exceeds binary framing limits"
                                .to_string(),
                        ))
                    })
            })
            .transpose()?;
        // Claimed autocommit keeps the caller's stable retry digest. Explicit codec-5 COMMIT uses
        // the aggregate's canonical ordered-statement digest directly; only the legacy terminal
        // derives identity from a resolved binary payload.
        let request_digest = match (request_digest_override, codec5_staged.as_ref()) {
            (Some(digest), _) => digest,
            (None, Some(staged)) => {
                let digests = staged
                    .iter()
                    .map(|contribution| contribution.typed_statement_digest)
                    .collect::<Vec<_>>();
                let final_writers = staged.final_writer_statement_digests();
                if let Some(composition) = codec5_catalog_composition.as_ref() {
                    crate::typed_insert_aggregate::live_explicit_request_digest_with_catalog(
                        &digests,
                        &final_writers,
                        &composition.operation_body,
                    )
                    .map_err(ExecuteError::Engine)?
                } else {
                    crate::typed_insert_aggregate::live_explicit_request_digest(
                        &digests,
                        &final_writers,
                    )
                    .map_err(ExecuteError::Engine)?
                }
            }
            (None, None) => gpu_db_wal::canonical_request_digest(
                payload
                    .as_deref()
                    .expect("legacy terminal owns its resolved binary payload"),
            ),
        };
        let codec5_mode = if request_digest_override.is_some() {
            crate::typed_insert_aggregate::LiveTypedInsertMode::Autocommit
        } else {
            crate::typed_insert_aggregate::LiveTypedInsertMode::Explicit
        };
        // Keep enrolled named-index coverage stable through canonical apply. Codec-5 derives the
        // set from its sealed typed contributions and performs its own exact DeviceInsertPlan
        // reservation; the generic resolved-record reservation remains legacy-only.
        let named_index_tables = if let Some(record) = legacy_record.as_ref() {
            record
                .mutations
                .iter()
                .map(|mutation| match mutation {
                    BinaryTransactionMutation::Insert { table, .. }
                    | BinaryTransactionMutation::Update { table, .. }
                    | BinaryTransactionMutation::Delete { table, .. } => table.clone(),
                })
                .chain(record.catalog_commands.iter().filter_map(|operation| {
                    match &operation.command {
                        Command::CreateTable(create) => Some(create.table.clone()),
                        _ => None,
                    }
                }))
                .chain(
                    record
                        .index_lifecycle_operations
                        .iter()
                        .flat_map(|operation| &operation.targets)
                        .filter_map(|target| target.owner_name.clone()),
                )
                .chain(record.table_resets.iter().map(|reset| reset.table.clone()))
                .collect::<BTreeSet<_>>()
        } else {
            codec5_staged
                .as_ref()
                .expect("non-legacy terminal owns typed contributions")
                .iter()
                .map(|contribution| contribution.table.clone())
                .collect()
        };
        let mut named_index_publication = Some(
            self.read_state
                .residency
                .begin_transaction_named_index_publication(named_index_tables),
        );
        if let Some(record) = legacy_record.as_ref() {
            self.reserve_transaction_canonical_publication(&snapshot, &transaction_catalog, record)
                .map_err(Self::retryable_transaction_terminal_pre_wal_decline)?;
        }
        let mut codec5_operation = if let Some(staged) = codec5_staged {
            let has_index = staged.has_surviving_indexed_table(&transaction_catalog);
            let indexed_lifecycle = has_index.then(|| {
                named_index_publication.take().expect(
                    "indexed codec-5 preflight must retain the common named-index lifecycle",
                )
            });
            Some(codec5_operation::prepare_generic_codec5_operation(
                self,
                txn_id,
                &staged,
                codec5_sequence_value_references
                    .as_deref()
                    .expect("codec-5 selection resolves its sequence references"),
                &provisional_inserts,
                &transaction_catalog.relational_catalog,
                &transaction_catalog,
                &catalog_commands,
                commit.canonical_identity,
                1,
                expected_commit_seq,
                catalog_epoch,
                catalog_digest,
                codec5_mode,
                request_digest,
                codec5_catalog_composition,
                isolation,
                final_base,
                allocator_high_water,
                indexed_lifecycle,
            )?)
        } else {
            None
        };
        // The resolved record remains owned here until its sole WAL encoding below. Its mutation
        // vector is the exact acknowledgement cardinality for the transaction opcode, so derive
        // it directly rather than decoding the newly encoded live payload through a historical
        // reader merely to recover a value this terminal already owns.
        let affected_rows = if let Some(operation) = codec5_operation.as_ref() {
            operation.affected_rows
        } else {
            u64::try_from(
                legacy_record
                    .as_ref()
                    .expect("legacy terminal owns its resolved record")
                    .mutations
                    .len(),
            )
            .map_err(|_| {
                ExecuteError::Engine(EngineError::Durability(
                    "transaction mutation count exceeds canonical affected-row framing".to_string(),
                ))
            })?
        };
        // Register the chained successor before the first durable effect. Allocator exhaustion is
        // therefore a clean pre-WAL error that leaves the parent transaction and its lifetime
        // snapshot active. Every later failure path cancels this provisional successor.
        let successor_id = if chain {
            Some(
                self.begin_unclaimed_transaction(&mut commit)
                    .map_err(ExecuteError::Txn)?,
            )
        } else {
            None
        };

        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_transaction_terminal_pre_wal_nanos(
            probe_terminal_pre_wal_started.elapsed().as_nanos() as u64,
        );
        #[cfg(feature = "probe-timing")]
        let probe_terminal_wal_status_started = std::time::Instant::now();
        let wal_len_before = commit.wal.len();
        let mut typed_control_plane = match codec5_operation.as_mut() {
            Some(operation) => Some(
                commit
                    .reserve_typed_canonical_control_plane(
                        txn_id,
                        expected_commit_seq,
                        operation
                            .record
                            .take()
                            .expect("generic codec-5 record is consumed only by the control plane"),
                    )
                    .map_err(ExecuteError::Engine)?,
            ),
            None => None,
        };
        #[cfg(test)]
        if let Err(error) = self.fail_transaction_terminal_pre_durable_if_injected(
            TransactionTerminalPreDurableFault::ReplicationProposal,
        ) {
            Self::cancel_chained_successor(&mut commit, successor_id);
            return Err(error);
        }
        let token_result = match typed_control_plane.as_mut() {
            Some(control_plane) => control_plane.propose(&mut commit),
            None => commit.repl.propose(Arc::clone(
                payload
                    .as_ref()
                    .expect("legacy proposal owns its resolved binary payload"),
            )),
        };
        let token = match token_result {
            Ok(token) => token,
            Err(err) => {
                Self::cancel_chained_successor(&mut commit, successor_id);
                return Err(ExecuteError::Engine(err));
            }
        };
        let canonical_record = if typed_control_plane.is_none() {
            match Self::canonical_wal_record_from_live_transaction(
                &mut commit,
                txn_id,
                token.index,
                0,
                payload
                    .as_ref()
                    .expect("legacy canonical record owns its binary payload"),
                legacy_record
                    .as_ref()
                    .expect("legacy canonical record owns its resolved record"),
                isolation,
                request_digest,
                affected_rows,
            ) {
                Ok(canonical_record) => Some(canonical_record),
                Err(err) => {
                    commit.repl.rollback_unapplied_from(token.index);
                    Self::cancel_chained_successor(&mut commit, successor_id);
                    return Err(ExecuteError::Engine(err));
                }
            }
        } else {
            None
        };
        #[cfg(test)]
        if let Err(error) = self.fail_transaction_terminal_pre_durable_if_injected(
            TransactionTerminalPreDurableFault::CanonicalWalAppend,
        ) {
            commit.repl.rollback_unapplied_from(token.index);
            Self::cancel_chained_successor(&mut commit, successor_id);
            return Err(error);
        }
        if let Some(control_plane) = typed_control_plane.as_mut() {
            if let Err(error) = control_plane.append_canonical(&mut commit) {
                commit.repl.rollback_unapplied_from(token.index);
                Self::cancel_chained_successor(&mut commit, successor_id);
                return Err(ExecuteError::Engine(error));
            }
        } else {
            commit
                .wal
                .append_canonical(canonical_record.expect("resolved canonical record exists"));
        }
        #[cfg(test)]
        if let Err(error) = self.fail_transaction_terminal_pre_durable_if_injected(
            TransactionTerminalPreDurableFault::WalFlush,
        ) {
            commit.repl.rollback_unapplied_from(token.index);
            if let Some(control_plane) = typed_control_plane.as_mut() {
                control_plane.rollback_tentative_wal_after_pre_durable_failure(&mut commit);
            } else {
                commit.wal.truncate(wal_len_before);
            }
            Self::cancel_chained_successor(&mut commit, successor_id);
            return Err(error);
        }
        if let Some(control_plane) = typed_control_plane.as_mut() {
            if let Err(error) =
                control_plane.record_transaction_status(&mut commit, request_digest, affected_rows)
            {
                control_plane.rollback_tentative_wal_after_pre_durable_failure(&mut commit);
                commit.repl.rollback_unapplied_from(token.index);
                Self::cancel_chained_successor(&mut commit, successor_id);
                return Err(ExecuteError::Engine(error));
            }
            if let Err(error) = control_plane.record_commit_timestamp(&mut commit, timestamp_micros)
            {
                control_plane.rollback_inserted_status_after_pre_durable_failure(
                    &mut commit,
                    request_digest,
                    affected_rows,
                );
                control_plane.rollback_tentative_wal_after_pre_durable_failure(&mut commit);
                commit.repl.rollback_unapplied_from(token.index);
                Self::cancel_chained_successor(&mut commit, successor_id);
                return Err(ExecuteError::Engine(error));
            }
            let _wal_position =
                control_plane.claim_tentative_wal_after_final_rollbackable_step(&mut commit);
        }
        if let Err(err) = commit.wal.flush_all() {
            Self::cancel_chained_successor(&mut commit, successor_id);
            if typed_control_plane.is_some() {
                self.wedge_commit_path();
                return Err(ExecuteError::Indeterminate(format!(
                    "explicit transaction {txn_id} claimed codec-5 durability facts but WAL flush failed: {err}; restart recovery must resolve it"
                )));
            }
            commit.repl.rollback_unapplied_from(token.index);
            commit.wal.truncate(wal_len_before);
            return Err(ExecuteError::Engine(err));
        }
        if let Err(error) = commit.repl.wait_committed(token, Duration::from_millis(0)) {
            Self::cancel_chained_successor(&mut commit, successor_id);
            self.wedge_commit_path();
            return Err(ExecuteError::Indeterminate(format!(
                "explicit transaction {txn_id} flushed WAL but commit confirmation failed: {error}; restart recovery must resolve it"
            )));
        }
        if typed_control_plane.is_none() {
            if let Err(error) = commit.record_transaction_status_digest_outcome(
                txn_id,
                request_digest,
                token.index,
                affected_rows,
            ) {
                Self::cancel_chained_successor(&mut commit, successor_id);
                self.wedge_commit_path();
                return Err(ExecuteError::Indeterminate(format!(
                    "explicit transaction {txn_id} became durable but terminal status installation failed: {error}; restart recovery must resolve it"
                )));
            }
        }
        if request_digest_override.is_some() {
            self.release_pending_transaction_claim(txn_id, request_digest);
        }
        if typed_control_plane.is_none() {
            commit.record_commit_timestamp(txn_id, timestamp_micros);
        }
        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_transaction_terminal_wal_status_nanos(
            probe_terminal_wal_status_started.elapsed().as_nanos() as u64,
        );
        // The durable resolved record is now the sole source of truth. Retire transaction-private
        // COW allocations before canonical apply constructs/publishes the global generation; this
        // avoids double-accounting the same logical rows against the GPU residency budget.
        let commit_gpu_credit = self.retire_transaction_private_generation_post_durable(&snapshot);
        #[cfg(test)]
        let post_durable_hook = self
            .transaction_post_durable_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        #[cfg(test)]
        if let Some(hook) = post_durable_hook {
            hook();
        }
        // The pre-WAL indexed physical reservation held the same common lifecycle owner while it
        // sealed cache/currentness evidence.  Move it back before the generic finalizer enters
        // final publication; no second lifecycle guard is minted for the device plan.
        if let Some(lifecycle) = codec5_operation.as_mut().and_then(|operation| {
            operation
                .plans
                .iter_mut()
                .find_map(|plan| plan.take_named_index_publication_guard())
        }) {
            named_index_publication = Some(lifecycle);
        }
        let mut named_index_publication = named_index_publication
            .expect("generic finalizer must retain one named-index lifecycle through publication");
        // From canonical apply's first allocation through acknowledgement, an external
        // pressure/retirement request must outlive the final index publication. Owner-local
        // replacement purges use the explicit during-transaction bypass and are not replayed.
        named_index_publication.enter_final_publication();
        let publication_owner = TransactionNamedIndexPublicationOwnerGuard::enter();
        #[cfg(feature = "probe-timing")]
        let probe_canonical_apply_started = std::time::Instant::now();
        #[cfg(test)]
        let apply_result = self.with_transaction_commit_gpu_credit(&commit_gpu_credit, || {
            if let Some(error) = self.take_transaction_post_durable_apply_fault() {
                return Err(error);
            }
            if let Some(operation) = codec5_operation.take() {
                for plan in operation.plans {
                    let permit = crate::engine_dml_concurrent::issue_transaction_terminal_typed_insert_apply_permit(
                        token.index,
                    );
                    plan.apply_after_transaction_wal_claim(self, permit)
                        .map_err(|error| {
                            EngineError::Durability(format!(
                                "generic codec-5 device apply failed after durable WAL: {error:?}"
                            ))
                        })?;
                }
                self.read_state
                    .mvcc
                    .consume_proposed_row_id_range(operation.proposed_range)
                    .map_err(|error| {
                        EngineError::Durability(format!(
                            "generic codec-5 allocator consumption drifted after device apply: {error}"
                        ))
                    })?;
                let payload_len = operation.payload_authority.len();
                let witness = crate::engine_commit::LiveTypedTransactionApply {
                    txn_id,
                    expected_index: token.index,
                    payload_authority: crate::engine_commit::LiveTypedPayloadAuthority::Exact(
                        operation.payload_authority,
                    ),
                    payload_len,
                    tables: operation.tables,
                    allocator_high_water: operation.allocator_high_water,
                    allocator_already_consumed: true,
                    affected_rows: operation.affected_rows,
                    write_set: operation.write_set,
                    private_sequence_publications: operation.private_sequence_publications,
                    catalog_composition: operation.catalog_composition,
                    parent_authority: None,
                };
                self.apply_and_publish_committed_with_live_semantics_v2_codec5_transaction(
                    &mut commit,
                    txn_id,
                    token.index,
                    witness,
                    operation.root_publication,
                )?;
                let _generation_authority = operation.generations;
                Ok(())
            } else {
                self.apply_and_publish_committed(&mut commit, txn_id, token.index)
            }
        });
        #[cfg(not(test))]
        let apply_result = self.with_transaction_commit_gpu_credit(&commit_gpu_credit, || {
            if let Some(operation) = codec5_operation.take() {
                for plan in operation.plans {
                    let permit = crate::engine_dml_concurrent::issue_transaction_terminal_typed_insert_apply_permit(
                        token.index,
                    );
                    plan.apply_after_transaction_wal_claim(self, permit)
                        .map_err(|error| {
                            EngineError::Durability(format!(
                                "generic codec-5 device apply failed after durable WAL: {error:?}"
                            ))
                        })?;
                }
                self.read_state
                    .mvcc
                    .consume_proposed_row_id_range(operation.proposed_range)
                    .map_err(|error| {
                        EngineError::Durability(format!(
                            "generic codec-5 allocator consumption drifted after device apply: {error}"
                        ))
                    })?;
                let payload_len = operation.payload_authority.len();
                let witness = crate::engine_commit::LiveTypedTransactionApply {
                    txn_id,
                    expected_index: token.index,
                    payload_authority: crate::engine_commit::LiveTypedPayloadAuthority::Exact(
                        operation.payload_authority,
                    ),
                    payload_len,
                    tables: operation.tables,
                    allocator_high_water: operation.allocator_high_water,
                    allocator_already_consumed: true,
                    affected_rows: operation.affected_rows,
                    write_set: operation.write_set,
                    private_sequence_publications: operation.private_sequence_publications,
                    catalog_composition: operation.catalog_composition,
                    parent_authority: None,
                };
                self.apply_and_publish_committed_with_live_semantics_v2_codec5_transaction(
                    &mut commit,
                    txn_id,
                    token.index,
                    witness,
                    operation.root_publication,
                )?;
                let _generation_authority = operation.generations;
                Ok(())
            } else {
                self.apply_and_publish_committed(&mut commit, txn_id, token.index)
            }
        });
        drop(publication_owner);
        // A post-durable failure can occur before the codec-5 operation is moved into apply.
        // Its indexed device plan still owns the pre-WAL GPU budget guard in that case. Drop the
        // unconsumed plan before releasing the transaction-private accounting credit, whose
        // cleanup acquires the same budget lock. Recovery remains the only outcome authority.
        drop(codec5_operation);
        self.release_transaction_commit_gpu_credit(&commit_gpu_credit);
        if let Err(err) = apply_result {
            Self::cancel_chained_successor(&mut commit, successor_id);
            self.wedge_commit_path();
            return Err(ExecuteError::Indeterminate(format!(
                "explicit transaction {txn_id} is durable but could not be fully installed: {err}; engine restart recovery required"
            )));
        }
        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_transaction_canonical_apply_publish_nanos(
            probe_canonical_apply_started.elapsed().as_nanos() as u64,
        );
        #[cfg(feature = "probe-timing")]
        let probe_terminal_post_canonical_finalize_started = std::time::Instant::now();
        #[cfg(feature = "probe-timing")]
        for rows in typed_insert_rows {
            self.record_insert_probe_success(rows);
        }
        #[cfg(test)]
        self.read_state
            .residency
            .run_transaction_named_index_post_apply_hook();
        named_index_publication.complete();

        commit
            .txn_manager
            .commit(txn_id)
            .expect("active transaction remained active under the commit lock");
        let successor = successor_id.map(|next_id| {
            (
                next_id,
                self.capture_transaction_snapshot(self.committed_seq(), snapshot.characteristics),
            )
        });
        // Retire only generations that are older than this transaction's own lifetime snapshot.
        // Deregistering first would make the just-published commit immediately eligible and erase
        // its exact created-by sidecar before its successful terminal returns.
        self.gc_transaction_created_by_regions();
        let mut aliases = self
            .public_transaction_aliases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut active = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed = active.deregister_transaction(txn_id);
        debug_assert!(removed.is_some(), "committed transaction lost its snapshot");
        let successor_id = successor.as_ref().map(|(next_id, _)| *next_id);
        if let Some((next_id, next_snapshot)) = successor {
            active.register_transaction(next_id, next_snapshot);
        }
        Self::complete_public_transaction_alias_in_map(&mut aliases, txn_id, successor_id, true);
        drop(active);
        drop(aliases);
        self.metrics.inc_commit();
        drop(commit);

        // Cold candidate directories are GPU structures, but spilled payload staging may issue
        // positional NVMe reads. Lifecycle publication therefore retires the obsolete directories
        // while holding the sole commit/publication lock, then rebuilds the final catalog's unique
        // key candidates only after releasing it. COMMIT does not return until this best-effort
        // priming pass finishes; a concurrent DML that reaches an unprimed spilled generation
        // during this narrow interval fails closed instead of consulting any host index.
        let cold_chunks = self.read_streaming_cold_chunks();
        for table_name in cold_index_candidate_tables {
            if let Some(entry) = cold_chunks.get(&table_name) {
                self.prime_chunk_key_candidates(&table_name, entry);
            }
        }
        #[cfg(feature = "probe-timing")]
        self.record_insert_probe_transaction_terminal_post_canonical_finalize_nanos(
            probe_terminal_post_canonical_finalize_started
                .elapsed()
                .as_nanos() as u64,
        );
        Ok(successor_id)
    }

    fn retire_transaction_private_generation_post_durable(
        &self,
        snapshot: &TransactionSnapshot,
    ) -> BTreeMap<u16, u64> {
        let (private_shards, private_cold_chunks, charges, commit_charges) = {
            let mut delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert!(
                Arc::ptr_eq(&delta.resident_shards, &delta.resident_shards_authority)
                    && Arc::ptr_eq(
                        &delta.streaming_cold_chunks,
                        &delta.streaming_cold_chunks_authority
                    ),
                "pre-WAL validation must seal the private GPU generation before retirement"
            );
            let private_shards = Arc::clone(&delta.resident_shards);
            let private_cold_chunks = Arc::clone(&delta.streaming_cold_chunks);
            delta.publish_resident_shards(Arc::clone(&snapshot.resident_shards));
            delta.publish_streaming_cold_chunks(Arc::clone(&snapshot.base_streaming_cold_chunks));
            let charges = std::mem::take(&mut delta.private_gpu_bytes_by_gpu);
            let commit_charges = std::mem::take(&mut delta.commit_gpu_bytes_by_gpu);
            (private_shards, private_cold_chunks, charges, commit_charges)
        };
        let mut credit = charges;
        for (gpu_id, bytes) in commit_charges {
            let slot = credit.entry(gpu_id).or_default();
            *slot = slot.saturating_add(bytes);
        }
        // Keep the exact account charge alive as a publication credit while the private Arcs are
        // dropped. Concurrent allocators continue to see it; only this apply thread subtracts it
        // from its own budget observations until global payload/index allocations replace it.
        drop(private_shards);
        drop(private_cold_chunks);
        credit
    }

    fn validate_transaction_device_row_conflicts(
        &self,
        snapshot: &TransactionSnapshot,
        deltas: &[WriteDelta],
        provisional_inserts: &BTreeSet<(String, u64)>,
    ) -> Result<(), ExecuteError> {
        debug_assert!(
            self.current_transaction_read_snapshot().is_none(),
            "COMMIT conflict validation must bind the current device generation"
        );
        let current_boundary = self.committed_seq();
        let mut checked = BTreeSet::new();
        for delta in deltas {
            let read_snapshot = delta.read_snapshot;
            let (table_name, targets): (&str, Vec<(&str, &[SqlValue])>) = match &delta.mutation {
                PreparedMutation::Insert { .. } => continue,
                PreparedMutation::Update {
                    table,
                    installs,
                    updated_old_rows,
                    ..
                } => (
                    table,
                    installs
                        .iter()
                        .zip(updated_old_rows)
                        .map(|((_, key, _), row)| (key.as_str(), row.as_slice()))
                        .collect(),
                ),
                PreparedMutation::Delete {
                    table,
                    deleted_rows,
                    ..
                } => (
                    table,
                    delta
                        .write_set
                        .rows
                        .iter()
                        .zip(deleted_rows)
                        .map(|(key, row)| (key.row_key.as_str(), row.as_slice()))
                        .collect(),
                ),
            };
            let catalog = snapshot.transaction_catalog();
            let table = catalog
                .relational_catalog
                .get(table_name)
                .expect("staged transaction table remains in its retained catalog");
            let prefix = relational_key_prefix(table_name);
            for (row_key, expected_row) in targets {
                let row_id = crate::engine_residency::parse_relational_row_id(row_key, &prefix)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction conflict target lost stable entity identity".to_string(),
                        ))
                    })?;
                let identity = (table_name.to_string(), row_id);
                if provisional_inserts.contains(&identity) || !checked.insert(identity) {
                    continue;
                }
                if snapshot.chunk_authoritative_tables.contains_key(table_name) {
                    let Some((observed, created_by, _, _)) =
                        self.class_row_by_entity_identity(table, row_id, current_boundary)
                    else {
                        return Err(ExecuteError::Serialization(format!(
                            "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                            read_snapshot
                        )));
                    };
                    if created_by > read_snapshot || observed.as_slice() != expected_row {
                        return Err(ExecuteError::Serialization(format!(
                            "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                            read_snapshot
                        )));
                    }
                    continue;
                }
                let filters = expected_row
                    .iter()
                    .enumerate()
                    .map(|(idx, value)| (idx, SelectFilterOp::Eq, value.clone()))
                    .collect::<Vec<_>>();
                let probe = self.dml_device_probe_key(table, &filters);
                let indexed = probe.and_then(|(key_id, needle)| {
                    self.locate_resident_pk_via_shard_index_detailed(table, key_id, needle)
                });
                let hits = match indexed {
                    Some(hits) if !hits.is_empty() => hits,
                    _ if crate::engine_prepared_transaction::prepared_index_route_required() => {
                        // The engine-owned proof promised an exact named-index route. Version-chain
                        // overflow or any physical decline is a retryable conflict, never permission
                        // to introduce an undeclared O(rows) commit-time scan.
                        return Err(ExecuteError::Serialization(format!(
                            "prepared device write-conflict index verdict unavailable for relation \"{table_name}\""
                        )));
                    }
                    _ => {
                        // Version churn can make a unique index decline because old and new
                        // physical slots share a key. Scan the same probe key structurally when
                        // possible; it may return multiple versions, which the identity, creation
                        // stamp, and full-row materialization checks below disambiguate exactly.
                        let key_cols = if let Some((key_id, _)) = probe {
                            let Some(value) = expected_row.get(key_id) else {
                                return Err(ExecuteError::Serialization(format!(
                                    "device write-conflict predicate unavailable for relation \"{table_name}\""
                                )));
                            };
                            vec![(key_id, value.clone())]
                        } else {
                            expected_row
                                .iter()
                                .enumerate()
                                .map(|(idx, value)| (idx, value.clone()))
                                .collect::<Vec<_>>()
                        };
                        let Some(predicates) =
                            crate::engine_dml_prepare::device_structural_tuple_predicates(
                                table, &key_cols,
                            )
                        else {
                            return Err(ExecuteError::Serialization(format!(
                                "device write-conflict predicate unavailable for relation \"{table_name}\""
                            )));
                        };
                        let Some(hits) = self
                            .locate_resident_conjunct_slots_detailed(table, &predicates)
                            .or_else(|| self.locate_resident_all_slots_detailed(table))
                        else {
                            return Err(ExecuteError::Serialization(format!(
                                "device write-conflict locate unavailable for relation \"{table_name}\""
                            )));
                        };
                        hits
                    }
                };
                let mut matched = 0usize;
                for hit in hits {
                    if self.hit_entity_id(&hit) != Some(row_id) {
                        continue;
                    }
                    let created_by = match &hit.created_by {
                        None => 0,
                        Some(region) => {
                            let halves = region
                                .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
                                .map_err(|err| {
                                    ExecuteError::Serialization(format!(
                                        "device write-conflict stamp read failed for relation \"{table_name}\": {err}"
                                    ))
                                })?;
                            (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32)
                        }
                    };
                    if created_by > read_snapshot {
                        return Err(ExecuteError::Serialization(format!(
                            "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                            read_snapshot
                        )));
                    }
                    match self.materialize_resident_row_via_hit(table, &hit, current_boundary) {
                        Some(Some(observed)) if observed.as_slice() == expected_row => matched += 1,
                        Some(Some(_)) | Some(None) => {}
                        None => {
                            return Err(ExecuteError::Serialization(format!(
                                "device write-conflict row verification failed for relation \"{table_name}\""
                            )));
                        }
                    }
                }
                if matched != 1 {
                    return Err(ExecuteError::Serialization(format!(
                        "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                        read_snapshot
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_transaction_device_unique_conflicts(
        &self,
        snapshot: &TransactionSnapshot,
        deltas: &[WriteDelta],
        final_staged_rows: &[FinalTransactionRowOperation],
        legacy_record: Option<&BinaryTransactionRecord>,
        reset_tables: &BTreeSet<String>,
    ) -> Result<(), ExecuteError> {
        // Typed INSERT has already closed row-local values before this terminal.  When the
        // transaction contains no legacy row mutation and every touched typed table has no
        // unique index, reconstructing every private row below cannot establish a new
        // first-committer-wins or current-key fact: there is no unique key to probe.  Keep the
        // exact existing validator for any legacy shape or indexed typed table; this is only the
        // semantic no-op case, not a relaxed constraint verdict.
        if deltas.is_empty() && legacy_record.is_none() {
            let catalog = snapshot.transaction_catalog();
            let typed_tables_have_no_unique_index = final_staged_rows
                .iter()
                .filter_map(|operation| match operation {
                    FinalTransactionRowOperation::TypedInsert(staged) => Some(staged),
                    FinalTransactionRowOperation::Legacy(_) => None,
                })
                .all(|staged| {
                    catalog
                        .relational_catalog
                        .get(&staged.table)
                        .is_some_and(|table| !table.indexes.iter().any(|index| index.unique))
                });
            let typed_only = final_staged_rows
                .iter()
                .all(|operation| matches!(operation, FinalTransactionRowOperation::TypedInsert(_)));
            if typed_only && typed_tables_have_no_unique_index {
                return Ok(());
            }
        }
        // Every UPDATE/DELETE identity retires its old unique-key ownership at this transaction's
        // single publish boundary. Candidate final rows (including INSERTs) must exclude the whole
        // retiring set, not merely an UPDATE's own identity: key release+reuse and key swaps are
        // valid when the transaction's final relation is unique.
        let mut retiring_ids = BTreeMap::<String, BTreeSet<u64>>::new();
        for mutation in legacy_record
            .into_iter()
            .flat_map(|record| &record.mutations)
        {
            match mutation {
                BinaryTransactionMutation::Update { table, row_id, .. }
                | BinaryTransactionMutation::Delete { table, row_id, .. } => {
                    retiring_ids
                        .entry(table.clone())
                        .or_default()
                        .insert(*row_id);
                }
                BinaryTransactionMutation::Insert { .. } => {}
            }
        }
        // First-committer-wins includes released keys, not only current duplicates. Scan every old
        // and new exact key against physical device version history before the current-row check
        // below; an insert+delete or key-away after BEGIN must still serialize this transaction.
        for delta in deltas {
            let (table_name, rows): (&str, Vec<&[SqlValue]>) = match &delta.mutation {
                PreparedMutation::Insert {
                    table,
                    inserted_rows,
                    ..
                } => (
                    table,
                    inserted_rows
                        .iter()
                        .map(|(_, row)| row.as_slice())
                        .collect(),
                ),
                PreparedMutation::Update {
                    table,
                    installs,
                    updated_old_rows,
                    ..
                } => (
                    table,
                    updated_old_rows
                        .iter()
                        .map(Vec::as_slice)
                        .chain(installs.iter().map(|(_, _, row)| row.as_slice()))
                        .collect(),
                ),
                PreparedMutation::Delete {
                    table,
                    deleted_rows,
                    ..
                } => (table, deleted_rows.iter().map(Vec::as_slice).collect()),
            };
            let catalog = snapshot.transaction_catalog();
            let table = catalog
                .relational_catalog
                .get(table_name)
                .expect("transaction record table remains in retained catalog");
            if reset_tables.contains(table_name) {
                continue;
            }
            if !table.indexes.iter().any(|index| index.unique)
                || !snapshot.catalog.relational_catalog.contains_key(table_name)
            {
                continue;
            }
            let conflict_boundary = self.transaction_table_conflict_boundary(
                snapshot,
                table_name,
                delta.read_snapshot,
            )?;
            match self.device_unique_rows_conflict(table, &rows, conflict_boundary) {
                Some(false) => {}
                Some(true) => {
                    return Err(ExecuteError::Serialization(format!(
                        "device unique conflict: write history changed in relation \"{table_name}\" after snapshot {}",
                        conflict_boundary
                    )));
                }
                None => {
                    return Err(ExecuteError::Serialization(format!(
                        "device unique-history verdict unavailable for relation \"{table_name}\""
                    )));
                }
            }
        }
        // Typed INSERT deliberately has no legacy `WriteDelta`, so derive its exact transient
        // control-plane rows from the immutable private payload at the same history-sensitive
        // device check. A current-row probe below is insufficient: a key claimed and released
        // after this stage's snapshot must still force first-committer-wins serialization failure.
        for staged in final_staged_rows
            .iter()
            .filter_map(|operation| match operation {
                FinalTransactionRowOperation::TypedInsert(staged) => Some(staged),
                FinalTransactionRowOperation::Legacy(_) => None,
            })
        {
            let table_name = staged.table.as_str();
            if reset_tables.contains(table_name) {
                continue;
            }
            let catalog = snapshot.transaction_catalog();
            let table = catalog
                .relational_catalog
                .get(table_name)
                .expect("typed transaction record table remains in retained catalog");
            if !table.indexes.iter().any(|index| index.unique)
                || !snapshot.catalog.relational_catalog.contains_key(table_name)
            {
                continue;
            }
            let decoded_rows = (0..staged.rows_consumed() as usize)
                .map(|row| staged.private_row_values(table, row))
                .collect::<Result<Vec<_>, _>>()?;
            let rows = decoded_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let conflict_boundary = self.transaction_table_conflict_boundary(
                snapshot,
                table_name,
                staged.read_snapshot,
            )?;
            match self.device_unique_rows_conflict(table, &rows, conflict_boundary) {
                Some(false) => {}
                Some(true) => {
                    return Err(ExecuteError::Serialization(format!(
                        "device unique conflict: write history changed in relation \"{table_name}\" after snapshot {}",
                        conflict_boundary
                    )));
                }
                None => {
                    return Err(ExecuteError::Serialization(format!(
                        "device unique-history verdict unavailable for relation \"{table_name}\""
                    )));
                }
            }
        }
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };
        let catalog = snapshot.transaction_catalog();
        let mut final_images = Vec::<(String, Vec<SqlValue>)>::new();
        if let Some(record) = legacy_record {
            for mutation in &record.mutations {
                match mutation {
                    BinaryTransactionMutation::Insert {
                        table, row_encoded, ..
                    } => {
                        let relation = catalog
                            .relational_catalog
                            .get(table)
                            .expect("transaction record table remains in retained catalog");
                        let row = decode_relational_row(row_encoded, &relation.columns).map_err(
                            |err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                    "transaction unique image decode failed for relation \"{table}\": {err}"
                                )))
                            },
                        )?;
                        final_images.push((table.clone(), row));
                    }
                    BinaryTransactionMutation::Update {
                        table,
                        new_row_encoded,
                        ..
                    } => {
                        let relation = catalog
                            .relational_catalog
                            .get(table)
                            .expect("transaction record table remains in retained catalog");
                        let row = decode_relational_row(new_row_encoded, &relation.columns)
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                    "transaction unique image decode failed for relation \"{table}\": {err}"
                                )))
                            })?;
                        final_images.push((table.clone(), row));
                    }
                    BinaryTransactionMutation::Delete { .. } => {}
                }
            }
        } else {
            for staged in final_staged_rows
                .iter()
                .filter_map(|operation| match operation {
                    FinalTransactionRowOperation::TypedInsert(staged) => Some(staged),
                    FinalTransactionRowOperation::Legacy(_) => None,
                })
            {
                let table = catalog
                    .relational_catalog
                    .get(&staged.table)
                    .expect("typed transaction record table remains in retained catalog");
                for row_ordinal in 0..staged.rows_consumed() as usize {
                    final_images.push((
                        staged.table.clone(),
                        staged.private_row_values(table, row_ordinal)?,
                    ));
                }
            }
        }
        for (table_name, row) in final_images {
            let table = catalog
                .relational_catalog
                .get(&table_name)
                .expect("transaction record table remains in retained catalog");
            if reset_tables.contains(&table_name) {
                continue;
            }
            if !snapshot
                .catalog
                .relational_catalog
                .contains_key(&table_name)
            {
                continue;
            }
            if !table.indexes.iter().any(|index| index.unique) {
                continue;
            }
            if self.table_chunk_authoritative(&table_name).is_some() {
                let exclusion = retiring_ids.get(&table_name).map(|ids| {
                    let mut coordinates = BTreeSet::new();
                    let mut epoch = None;
                    for retiring_id in ids {
                        let (_, _, coordinate, observed_epoch) = self
                            .class_row_by_entity_identity(
                                table,
                                *retiring_id,
                                visibility.read_txn_id,
                            )
                            .ok_or_else(|| {
                                ExecuteError::Serialization(format!(
                                    "device retiring-identity verdict unavailable for entity {retiring_id} in relation \"{table_name}\""
                                ))
                            })?;
                        if epoch.is_some_and(|current| current != observed_epoch) {
                            return Err(ExecuteError::Serialization(format!(
                                "device retiring identities crossed class generations in relation \"{table_name}\""
                            )));
                        }
                        epoch = Some(observed_epoch);
                        coordinates.insert(coordinate);
                    }
                    Ok((coordinates, epoch.unwrap_or(0)))
                }).transpose()?;
                let exclusion_ref = exclusion
                    .as_ref()
                    .map(|(coordinates, epoch)| (coordinates, *epoch));
                match self.validate_class_insert_uniqueness(
                    table,
                    std::slice::from_ref(&row),
                    visibility.read_txn_id,
                    exclusion_ref,
                ) {
                    Some(Ok(())) => {}
                    Some(Err(error)) => return Err(ExecuteError::Engine(error)),
                    None => {
                        return Err(ExecuteError::Serialization(format!(
                            "device unique-conflict verdict unavailable for relation \"{table_name}\""
                        )));
                    }
                }
                continue;
            }

            let excluded = retiring_ids.get(&table_name).map(|ids| {
                ids.iter()
                    .map(|retiring_id| relational_row_key(&table_name, *retiring_id))
                    .collect::<BTreeSet<_>>()
            });
            for (ordinal, index) in table
                .indexes
                .iter()
                .enumerate()
                .filter(|(_, index)| index.unique)
            {
                let positions = crate::engine_residency::index_key_column_positions(table, index)
                    .ok_or_else(|| {
                        ExecuteError::Serialization(format!(
                            "device unique-conflict key binding unavailable for relation \"{table_name}\""
                        ))
                    })?;
                // PostgreSQL UNIQUE uses NULLS DISTINCT by default. Primary-key NULLs have
                // already failed the ordered NOT NULL phase, so every remaining NULL-bearing
                // ordinary unique tuple has no equality probe and cannot conflict.
                if positions
                    .iter()
                    .any(|position| matches!(row[*position], SqlValue::Null))
                {
                    continue;
                }
                let conflict = if crate::engine_residency::index_uses_fingerprint(table, index) {
                    let key_id = crate::engine_residency::index_probe_key_id(table, index, ordinal)
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                            "device unique-conflict index unavailable for relation \"{table_name}\""
                        ))
                        })?;
                    let key_columns = positions
                        .iter()
                        .map(|position| (*position, row[*position].clone()))
                        .collect::<Vec<_>>();
                    match crate::engine_residency::compound_index_row_fingerprint(
                        table, index, &row,
                    ) {
                        Some(fingerprint) => self.device_visible_row_with_tuple(
                            table,
                            visibility,
                            key_id,
                            Some(fingerprint),
                            &key_columns,
                            excluded.as_ref(),
                        ),
                        None if positions.len() == 1 => self.device_visible_row_with_value(
                            table,
                            visibility,
                            positions[0],
                            &row[positions[0]],
                            excluded.as_ref(),
                        ),
                        None => self.device_visible_row_with_tuple(
                            table,
                            visibility,
                            key_id,
                            None,
                            &key_columns,
                            excluded.as_ref(),
                        ),
                    }
                } else {
                    self.device_visible_row_with_value(
                        table,
                        visibility,
                        positions[0],
                        &row[positions[0]],
                        excluded.as_ref(),
                    )
                };
                match conflict {
                    Some(false) => {}
                    Some(true) => {
                        return Err(ExecuteError::Engine(EngineError::UniqueViolation(format!(
                            "duplicate key value violates unique constraint \"{}\"",
                            index.name
                        ))));
                    }
                    None => {
                        return Err(ExecuteError::Serialization(format!(
                            "device unique-conflict verdict unavailable for relation \"{table_name}\""
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)] // every input participates in the transaction-scoped FK proof
    fn validate_transaction_device_foreign_key_conflicts(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        deltas: &[WriteDelta],
        final_staged_rows: &[FinalTransactionRowOperation],
        legacy_record: Option<&BinaryTransactionRecord>,
        provisional_inserts: &BTreeSet<(String, u64)>,
        final_base: u64,
        ledger: &RecentCommitsLedger,
    ) -> Result<(), ExecuteError> {
        // A typed-only INSERT cannot create an inbound FK obligation because it removes no
        // provider.  If each target has no outbound FK and its staged dependency proof names no
        // provider, the generic validator would only decode every already-sealed private row to
        // discover empty FK lists. Preserve the full path for legacy changes, outbound FKs, and
        // every nonempty staged provider dependency.
        if deltas.is_empty() && legacy_record.is_none() {
            let catalog = snapshot.transaction_catalog();
            let typed_tables_have_no_foreign_keys = final_staged_rows
                .iter()
                .filter_map(|operation| match operation {
                    FinalTransactionRowOperation::TypedInsert(staged) => Some(staged),
                    FinalTransactionRowOperation::Legacy(_) => None,
                })
                .all(|staged| {
                    staged.foreign_key_dependencies.is_empty()
                        && catalog
                            .relational_catalog
                            .get(&staged.table)
                            .is_some_and(|table| table.foreign_keys.is_empty())
                });
            let typed_only = final_staged_rows
                .iter()
                .all(|operation| matches!(operation, FinalTransactionRowOperation::TypedInsert(_)));
            if typed_only && typed_tables_have_no_foreign_keys {
                return Ok(());
            }
        }
        let catalog = snapshot.transaction_catalog();
        for delta in deltas {
            for table in &delta.foreign_key_dependencies {
                let conflict_boundary =
                    self.transaction_table_conflict_boundary(snapshot, table, delta.read_snapshot)?;
                let table_oid = catalog
                    .relational_catalog
                    .get(table)
                    .ok_or_else(|| {
                        ExecuteError::Serialization(format!(
                            "foreign-key dependency relation \"{table}\" left the transaction catalog"
                        ))
                    })?
                    .oid;
                if ledger.table_changed_after(table_oid, conflict_boundary) {
                    return Err(ExecuteError::Serialization(format!(
                        "foreign-key dependency relation \"{table}\" changed after transaction conflict boundary {conflict_boundary}"
                    )));
                }
            }
        }
        for staged in final_staged_rows
            .iter()
            .filter_map(|operation| match operation {
                FinalTransactionRowOperation::TypedInsert(staged) => Some(staged),
                FinalTransactionRowOperation::Legacy(_) => None,
            })
        {
            for table in &staged.foreign_key_dependencies {
                let conflict_boundary = self.transaction_table_conflict_boundary(
                    snapshot,
                    table,
                    staged.foreign_key_read_snapshot,
                )?;
                let table_oid = catalog
                    .relational_catalog
                    .get(table)
                    .ok_or_else(|| {
                        ExecuteError::Serialization(format!(
                            "foreign-key dependency relation \"{table}\" left the transaction catalog"
                        ))
                    })?
                    .oid;
                if ledger.table_changed_after(table_oid, conflict_boundary) {
                    return Err(ExecuteError::Serialization(format!(
                        "foreign-key dependency relation \"{table}\" changed after transaction conflict boundary {conflict_boundary}"
                    )));
                }
            }
        }
        type IdentityRows = BTreeMap<String, Vec<(u64, Vec<SqlValue>)>>;
        let mut final_rows = IdentityRows::new();
        let mut removed_rows = IdentityRows::new();
        let mut mutated_ids = BTreeMap::<String, BTreeSet<u64>>::new();
        if let Some(record) = legacy_record {
            for mutation in &record.mutations {
                let (table_name, row_id) = match mutation {
                    BinaryTransactionMutation::Insert { table, row_id, .. }
                    | BinaryTransactionMutation::Update { table, row_id, .. }
                    | BinaryTransactionMutation::Delete { table, row_id, .. } => (table, *row_id),
                };
                let table = catalog
                    .relational_catalog
                    .get(table_name)
                    .expect("transaction FK table remains in retained catalog");
                mutated_ids
                    .entry(table_name.clone())
                    .or_default()
                    .insert(row_id);
                match mutation {
                    BinaryTransactionMutation::Insert { row_encoded, .. } => {
                        let row = decode_relational_row(row_encoded, &table.columns).map_err(
                            |err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                    "transaction FK insert image decode failed for relation \"{table_name}\": {err}"
                                )))
                            },
                        )?;
                        final_rows
                            .entry(table_name.clone())
                            .or_default()
                            .push((row_id, row));
                    }
                    BinaryTransactionMutation::Update {
                        old_row_encoded,
                        new_row_encoded,
                        ..
                    } => {
                        let old = decode_relational_row(old_row_encoded, &table.columns).map_err(
                            |err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                    "transaction FK old image decode failed for relation \"{table_name}\": {err}"
                                )))
                            },
                        )?;
                        let new = decode_relational_row(new_row_encoded, &table.columns).map_err(
                            |err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                    "transaction FK new image decode failed for relation \"{table_name}\": {err}"
                                )))
                            },
                        )?;
                        removed_rows
                            .entry(table_name.clone())
                            .or_default()
                            .push((row_id, old));
                        final_rows
                            .entry(table_name.clone())
                            .or_default()
                            .push((row_id, new));
                    }
                    BinaryTransactionMutation::Delete {
                        old_row_encoded, ..
                    } => {
                        let old = decode_relational_row(old_row_encoded, &table.columns).map_err(
                            |err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                    "transaction FK delete image decode failed for relation \"{table_name}\": {err}"
                                )))
                            },
                        )?;
                        removed_rows
                            .entry(table_name.clone())
                            .or_default()
                            .push((row_id, old));
                    }
                }
            }
        } else {
            let final_ids = Self::resolved_insert_identities(provisional_inserts, final_base)?;
            for staged in final_staged_rows
                .iter()
                .filter_map(|operation| match operation {
                    FinalTransactionRowOperation::TypedInsert(staged) => Some(staged),
                    FinalTransactionRowOperation::Legacy(_) => None,
                })
            {
                let table = catalog
                    .relational_catalog
                    .get(&staged.table)
                    .expect("typed transaction FK table remains in retained catalog");
                for (row_ordinal, provisional) in
                    staged.provisional_row_ids.iter().copied().enumerate()
                {
                    let row_id = *final_ids
                        .get(&(staged.table.clone(), provisional))
                        .ok_or_else(|| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "typed transaction FK row lost provisional identity in relation \"{}\"",
                                staged.table
                            )))
                        })?;
                    let row = staged.private_row_values(table, row_ordinal)?;
                    mutated_ids
                        .entry(staged.table.clone())
                        .or_default()
                        .insert(row_id);
                    final_rows
                        .entry(staged.table.clone())
                        .or_default()
                        .push((row_id, row));
                }
            }
        }
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };

        // Outbound stamps: untouched current providers are checked against the current device
        // generation. Providers written by this transaction must exist in its final private
        // device generation; the final-row stamp prevents an old BEGIN version from satisfying it.
        for (child_name, rows) in &final_rows {
            let child = catalog
                .relational_catalog
                .get(child_name)
                .expect("transaction child remains cataloged");
            for foreign_key in &child.foreign_keys {
                let Some(parent) = catalog
                    .relational_catalog
                    .get(&foreign_key.referenced_table)
                else {
                    continue;
                };
                let child_idx =
                    relational_column_index(child, &foreign_key.column).map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                let parent_idx = relational_column_index(parent, &foreign_key.referenced_column)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                let parent_exclusions = mutated_ids.get(&parent.name).map(|ids| {
                    ids.iter()
                        .map(|row_id| relational_row_key(&parent.name, *row_id))
                        .collect::<BTreeSet<_>>()
                });
                // A provider relation created by this transaction has no public predecessor to
                // probe. Its exact final GPU generation is already the required provider
                // authority below; attempting the public probe first turns a valid private
                // parent/child transaction into a false serialization error. Existing parents
                // retain the current-generation verdict with transaction-mutated identities
                // excluded, so this is not a host fallback or a second FK authority.
                let parent_created_by_transaction = !snapshot
                    .catalog
                    .relational_catalog
                    .contains_key(&parent.name);
                for (_, row) in rows {
                    let value = &row[child_idx];
                    if matches!(value, SqlValue::Null) {
                        continue;
                    }
                    let current = if parent_created_by_transaction {
                        false
                    } else {
                        self.device_visible_row_with_value(
                            parent,
                            visibility,
                            parent_idx,
                            value,
                            parent_exclusions.as_ref(),
                        )
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "device foreign-key provider verdict unavailable for relation \"{}\"",
                                parent.name
                            ))
                        })?
                    };
                    if current {
                        continue;
                    }
                    let final_provider_stamp =
                        final_rows.get(&parent.name).is_some_and(|providers| {
                            providers
                                .iter()
                                .any(|(_, provider)| provider[parent_idx] == *value)
                        });
                    let private_provider = if final_provider_stamp {
                        let _scope = self.enter_transaction_read(Arc::clone(snapshot));
                        self.device_visible_row_with_value(
                            parent,
                            StorageVisibility {
                                read_txn_id: snapshot.boundary,
                            },
                            parent_idx,
                            value,
                            None,
                        )
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "private device foreign-key provider verdict unavailable for relation \"{}\"",
                                parent.name
                            ))
                        })?
                    } else {
                        false
                    };
                    if !private_provider {
                        return Err(ExecuteError::Engine(EngineError::ForeignKeyViolation(
                            format!(
                                "insert or update on table \"{}\" violates foreign key constraint \"{}\"",
                                child.name, foreign_key.name
                            ),
                        )));
                    }
                }
            }
        }

        // Inbound stamps: final private child images are explicit conflicts; current untouched
        // children are checked on-device with transaction-removed identities excluded.
        for (parent_name, rows) in &removed_rows {
            let parent = catalog
                .relational_catalog
                .get(parent_name)
                .expect("transaction parent remains cataloged");
            for child in catalog.relational_catalog.values() {
                for foreign_key in child
                    .foreign_keys
                    .iter()
                    .filter(|foreign_key| foreign_key.referenced_table == *parent_name)
                {
                    let parent_idx =
                        relational_column_index(parent, &foreign_key.referenced_column).map_err(
                            |err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())),
                        )?;
                    let child_idx =
                        relational_column_index(child, &foreign_key.column).map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    let child_exclusions = mutated_ids.get(&child.name).map(|ids| {
                        ids.iter()
                            .map(|row_id| relational_row_key(&child.name, *row_id))
                            .collect::<BTreeSet<_>>()
                    });
                    let final_children = final_rows.get(&child.name);
                    let final_providers = final_rows.get(parent_name);
                    let mut checked = BTreeSet::<SqlValue>::new();
                    for (_, old_parent) in rows {
                        let value = old_parent[parent_idx].clone();
                        if matches!(value, SqlValue::Null) || !checked.insert(value.clone()) {
                            continue;
                        }
                        if final_providers.is_some_and(|providers| {
                            providers
                                .iter()
                                .any(|(_, provider)| provider[parent_idx] == value)
                        }) {
                            continue;
                        }
                        if final_children.is_some_and(|children| {
                            children
                                .iter()
                                .any(|(_, child_row)| child_row[child_idx] == value)
                        }) {
                            return Err(ExecuteError::Engine(EngineError::ForeignKeyViolation(
                                format!(
                                    "insert or update on table \"{}\" violates foreign key constraint \"{}\"",
                                    child.name, foreign_key.name
                                ),
                            )));
                        }
                        let surviving_child = self
                            .device_visible_row_with_value(
                                child,
                                visibility,
                                child_idx,
                                &value,
                                child_exclusions.as_ref(),
                            )
                            .ok_or_else(|| {
                                ExecuteError::Serialization(format!(
                                    "device foreign-key child verdict unavailable for relation \"{}\"",
                                    child.name
                                ))
                            })?;
                        if surviving_child {
                            return Err(ExecuteError::Engine(EngineError::ForeignKeyViolation(
                                format!(
                                    "insert or update on table \"{}\" violates foreign key constraint \"{}\"",
                                    child.name, foreign_key.name
                                ),
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Keep BEGIN-generation preparation and its cheap conflict verdict available to the CPU
    /// parity oracle, but require an actual retained device generation before publishing a private
    /// delta. The CPU store is not a transaction-execution fallback for a conflict-free statement.
    pub(crate) fn validate_transaction_delta_residency(
        &self,
        table: &RelationalTable,
        existed_at_transaction_base: bool,
    ) -> Result<(), ExecuteError> {
        // A transaction-private CREATE owns an empty table generation before its first INSERT.
        // There cannot be a published shard for that relation yet; the private append below builds
        // its first device shard, and canonical apply/replay installs the final committed shard.
        if !existed_at_transaction_base {
            return Ok(());
        }
        if self
            .current_transaction_read_snapshot()
            .is_some_and(|snapshot| snapshot.table_has_typed_empty_root(&table.name))
        {
            return Ok(());
        }
        if self.table_chunk_authoritative(&table.name).is_some() {
            if self.read_streaming_cold_chunks().contains_key(&table.name) {
                return Ok(());
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no retained cold-chunk generation for transaction staging",
                table.name
            ))));
        }
        let shards = self.read_residency_shards();
        if shards
            .get(&table.name)
            .is_none_or(|table_shards| table_shards.is_empty())
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no retained GPU shard generation for transaction staging",
                table.name
            ))));
        }
        Ok(())
    }

    fn apply_transaction_private_delta(
        &self,
        table: &RelationalTable,
        delta: &WriteDelta,
        boundary: Index,
        shards: &mut BTreeMap<String, Vec<RelationalResidentShard>>,
        cold_chunks: &mut BTreeMap<String, Arc<crate::engine_streaming_exec::ColdTableChunks>>,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<(), ExecuteError> {
        if self.table_chunk_authoritative(&table.name).is_some() {
            let prefix = relational_key_prefix(&table.name);
            let entry = cold_chunks.get(&table.name).cloned().ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" lost its retained cold-chunk generation",
                    table.name
                )))
            })?;
            let parse_ids = |keys: Vec<&str>| -> Result<Vec<u64>, ExecuteError> {
                keys.into_iter()
                    .map(|key| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                            || {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction cold mutation lost stable entity identity"
                                        .to_string(),
                                ))
                            },
                        )
                    })
                    .collect()
            };
            let next = match &delta.mutation {
                PreparedMutation::Insert { inserted_rows, .. } => {
                    let rows = inserted_rows
                        .iter()
                        .map(|(_, row)| row.clone())
                        .collect::<Vec<_>>();
                    let row_ids =
                        parse_ids(inserted_rows.iter().map(|(key, _)| key.as_str()).collect())?;
                    self.append_transaction_cold_tail(table, &entry, &rows, &row_ids, boundary)
                }
                PreparedMutation::Update {
                    installs,
                    class_epoch: Some(epoch),
                    ..
                } => {
                    let coordinates = installs
                        .iter()
                        .map(|(coordinate, _, _)| *coordinate)
                        .collect::<Vec<_>>();
                    let row_ids =
                        parse_ids(installs.iter().map(|(_, key, _)| key.as_str()).collect())?;
                    let rows = installs
                        .iter()
                        .map(|(_, _, row)| row.clone())
                        .collect::<Vec<_>>();
                    self.stamp_transaction_cold_coordinates(&entry, &coordinates, *epoch, boundary)
                        .and_then(|stamped| {
                            self.append_transaction_cold_tail(
                                table, &stamped, &rows, &row_ids, boundary,
                            )
                        })
                }
                PreparedMutation::Delete {
                    tuple_ids,
                    class_epoch: Some(epoch),
                    ..
                } => self.stamp_transaction_cold_coordinates(&entry, tuple_ids, *epoch, boundary),
                PreparedMutation::Update {
                    class_epoch: None, ..
                }
                | PreparedMutation::Delete {
                    class_epoch: None, ..
                } => None,
            }
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" could not build a transaction-private cold delta",
                    table.name
                )))
            })?;
            cold_chunks.insert(table.name.clone(), next);
            return Ok(());
        }
        let prefix = relational_key_prefix(&table.name);
        match &delta.mutation {
            PreparedMutation::Insert { inserted_rows, .. } => {
                let row_ids = inserted_rows
                    .iter()
                    .map(|(key, _)| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                            || {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction INSERT produced a non-canonical entity identity"
                                        .to_string(),
                                ))
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let rows = inserted_rows
                    .iter()
                    .map(|(_, row)| row.clone())
                    .collect::<Vec<_>>();
                self.append_transaction_delta_shard(table, &rows, &row_ids, shards, gpu_reservation)
            }
            PreparedMutation::Update {
                installs,
                updated_old_rows,
                ..
            } => {
                let row_ids = installs
                    .iter()
                    .map(|(_, key, _)| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                            || {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction UPDATE lost stable entity identity".to_string(),
                                ))
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if row_ids.len() != updated_old_rows.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction UPDATE old/new identity vectors are not parallel".to_string(),
                    )));
                }
                let targets = row_ids
                    .iter()
                    .copied()
                    .zip(updated_old_rows.iter().map(Vec::as_slice))
                    .collect::<Vec<_>>();
                self.tombstone_transaction_rows(
                    table,
                    &targets,
                    boundary,
                    shards,
                    gpu_reservation,
                )?;
                let rows = installs
                    .iter()
                    .map(|(_, _, row)| row.clone())
                    .collect::<Vec<_>>();
                self.append_transaction_delta_shard(table, &rows, &row_ids, shards, gpu_reservation)
            }
            PreparedMutation::Delete { deleted_rows, .. } => {
                if deleted_rows.len() != delta.write_set.rows.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction DELETE row images lost their identity ordering".to_string(),
                    )));
                }
                let targets = delta
                    .write_set
                    .rows
                    .iter()
                    .zip(deleted_rows)
                    .map(|(key, row)| {
                        crate::engine_residency::parse_relational_row_id(&key.row_key, &prefix)
                            .map(|row_id| (row_id, row.as_slice()))
                            .ok_or_else(|| {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction DELETE lost stable entity identity".to_string(),
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.tombstone_transaction_rows(table, &targets, boundary, shards, gpu_reservation)
            }
        }
    }

    fn tombstone_transaction_rows(
        &self,
        table: &RelationalTable,
        targets: &[TargetRow<'_>],
        boundary: Index,
        shards: &mut BTreeMap<String, Vec<RelationalResidentShard>>,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<(), ExecuteError> {
        let mut slots_by_shard: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for &(row_id, old_row) in targets {
            let filters = old_row
                .iter()
                .enumerate()
                .map(|(idx, value)| (idx, SelectFilterOp::Eq, value.clone()))
                .collect::<Vec<_>>();
            let (key_id, needle) = self.dml_device_probe_key(table, &filters).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no device key usable by the transaction delta",
                    table.name
                )))
            })?;
            let hits = self
                .locate_resident_pk_via_shard_index_detailed(table, key_id, needle)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" transaction target locate declined",
                        table.name
                    )))
                })?;
            let mut visible = Vec::new();
            let mut observed = Vec::new();
            for hit in hits {
                let Some(identity) = self.hit_entity_id(&hit) else {
                    observed.push((hit.shard_id, hit.slot, None, None));
                    continue;
                };
                let materialized = self.materialize_resident_row_via_hit(table, &hit, boundary);
                observed.push((hit.shard_id, hit.slot, Some(identity), materialized.clone()));
                if identity != row_id {
                    continue;
                }
                if materialized.is_some_and(|row| row.as_deref() == Some(old_row)) {
                    visible.push((hit.shard_id, hit.slot));
                }
            }
            if visible.len() != 1 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction target entity {row_id} resolved {} visible versions; expected {old_row:?}, observed {observed:?}",
                    visible.len(),
                ))));
            }
            let (shard_id, slot) = visible[0];
            slots_by_shard.entry(shard_id).or_default().push(slot);
        }

        let table_shards = shards.get_mut(&table.name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" lost its transaction shard generation",
                table.name
            )))
        })?;
        for (shard_id, mut slots) in slots_by_shard {
            slots.sort_unstable();
            slots.dedup();
            let shard = table_shards
                .iter_mut()
                .find(|shard| shard.shard_id == shard_id)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "transaction target shard {shard_id} disappeared"
                    )))
                })?;
            let private = self.clone_transaction_deleted_region(shard, gpu_reservation)?;
            let stamps = vec![boundary; slots.len()];
            private.scatter_u64_slots(&slots, &stamps).map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction tombstone scatter failed: {err}"
                )))
            })?;
            shard.deleted_by_region = Some(private);
        }
        Ok(())
    }

    pub(crate) fn hit_entity_id(
        &self,
        hit: &crate::engine_retained_read::ShardPkHit,
    ) -> Option<u64> {
        let words = hit
            .row_id
            .as_ref()?
            .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
            .ok()?;
        let lo = *words.first()? as u32 as u64;
        let hi = *words.get(1)? as u32 as u64;
        Some(lo | (hi << 32))
    }
}

/// Row mutation decoding, FK closure, and typed value semantics are independent of the table's
/// index list. An ordered transaction may therefore stage DML on both sides of its own index
/// lifecycle commands. Those commands carry their own exact table/index before/after identities;
/// accepting only an index-list difference here avoids making that typed owner compete with a
/// second, stale full-table equality check.
fn same_transaction_row_catalog_dependency(
    expected: &RelationalTable,
    observed: &RelationalTable,
    sequence_input_oids: &BTreeMap<String, u32>,
    transaction_catalog: &CatalogSnapshot,
    catalog_commands: &[StagedCatalogCommand],
) -> bool {
    if expected == observed {
        return true;
    }
    let mut normalized = expected.clone();
    normalized.indexes = observed.indexes.clone();
    for column in &mut normalized.columns {
        let Some(ColumnDefault::SequenceNextVal { sequence, .. }) = &mut column.default else {
            continue;
        };
        let expected_oid = sequence_input_oids.get(sequence).copied().or_else(|| {
            catalog_commands
                .iter()
                .filter_map(|staged| staged.sequence_identity.as_ref())
                .flat_map(|identity| &identity.targets)
                .find_map(|target| {
                    (target.before_name == *sequence)
                        .then(|| target.target_before.as_ref().map(|identity| identity.oid))
                        .flatten()
                })
        });
        let Some(expected_oid) = expected_oid else {
            continue;
        };
        let Some(current_name) = transaction_catalog
            .relational_sequences
            .iter()
            .find_map(|(name, candidate)| (candidate.oid == expected_oid).then_some(name))
        else {
            return false;
        };
        *sequence = current_name.clone();
    }
    &normalized == observed
}
