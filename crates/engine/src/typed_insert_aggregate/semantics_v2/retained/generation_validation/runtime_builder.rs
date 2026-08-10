//! Production adapter from the sealed neutral generation input to the plural typed CUDA program.

use std::sync::{Arc, Mutex};

use gpu_db_execution::{
    OpaqueRuntimeTypedInsertGenerationProof, PreparedRuntimeTypedInsertGeneration,
    RuntimeTypedInsertGenerationAttempt, RuntimeTypedInsertGenerationCell,
    RuntimeTypedInsertGenerationCommitments, RuntimeTypedInsertGenerationCompletion,
    RuntimeTypedInsertGenerationGeometry, RuntimeTypedInsertGenerationIdentity,
    RuntimeTypedInsertGenerationLogicalCompletion, RuntimeTypedInsertGenerationRow,
    RuntimeTypedInsertGenerationSubmission, RuntimeTypedInsertGenerationTable,
    RuntimeTypedInsertGenerationTableAction, RuntimeTypedInsertGenerationTableMapPredecessor,
    RuntimeTypedInsertGenerationTarget, RuntimeTypedInsertGenerationUnknownQuiescence,
};

use super::{
    generation_error, input, sealed, DrainOutcome, LaunchedAttempt,
    SemanticsV2ReservedGenerationBuilder, SemanticsV2ReservedGenerationWork,
};
use crate::EngineError;

pub(in super::super) struct LiveTypedInsertGenerationBuilder {
    target: RuntimeTypedInsertGenerationTarget,
    attempt: RuntimeTypedInsertGenerationAttempt,
    table_map_predecessor: RuntimeTypedInsertGenerationTableMapPredecessor,
}

impl LiveTypedInsertGenerationBuilder {
    pub(in super::super) fn new(
        target: RuntimeTypedInsertGenerationTarget,
        attempt: RuntimeTypedInsertGenerationAttempt,
        table_map_predecessor: RuntimeTypedInsertGenerationTableMapPredecessor,
    ) -> Self {
        Self {
            target,
            attempt,
            table_map_predecessor,
        }
    }
}

struct SharedCompletion {
    proof: Option<OpaqueRuntimeTypedInsertGenerationProof>,
}

pub(in super::super) struct LiveTypedInsertGenerationCandidate {
    prepared: Option<PreparedRuntimeTypedInsertGeneration>,
    completion: Arc<Mutex<SharedCompletion>>,
    identity: Option<RuntimeTypedInsertGenerationIdentity>,
    table: Option<RuntimeTypedInsertGenerationTable>,
    commitments: Option<RuntimeTypedInsertGenerationCommitments>,
    logical: Option<RuntimeTypedInsertGenerationLogicalCompletion>,
}

impl LiveTypedInsertGenerationCandidate {
    pub(in super::super) fn take_authenticated_completion(
        &mut self,
    ) -> Option<(
        RuntimeTypedInsertGenerationCommitments,
        RuntimeTypedInsertGenerationLogicalCompletion,
    )> {
        Some((self.commitments.take()?, self.logical.take()?))
    }
}

enum WorkState {
    Submitted(RuntimeTypedInsertGenerationSubmission),
    Unknown(RuntimeTypedInsertGenerationUnknownQuiescence),
    Finished,
}

pub(in super::super) struct LiveTypedInsertGenerationWork {
    state: WorkState,
    completion: Arc<Mutex<SharedCompletion>>,
}

impl sealed::Builder for LiveTypedInsertGenerationBuilder {}
impl sealed::Candidate for LiveTypedInsertGenerationCandidate {}
impl sealed::Work for LiveTypedInsertGenerationWork {}

impl SemanticsV2ReservedGenerationBuilder for LiveTypedInsertGenerationBuilder {
    type Candidate = LiveTypedInsertGenerationCandidate;
    type Work = LiveTypedInsertGenerationWork;

    fn try_reserve_candidate(
        &self,
        reservation: input::GenerationBuilderReservation,
    ) -> Result<Self::Candidate, EngineError> {
        let geometry = RuntimeTypedInsertGenerationGeometry {
            rows: reservation.rows(),
            cells: reservation.cells(),
            value_bytes: reservation.value_bytes(),
            indexes: reservation.indexes(),
            index_keys: reservation.keys(),
            index_effects: reservation.effects(),
            index_effect_components: reservation.effect_values(),
        };
        if reservation.tables() != 1
            || reservation.keys() != 0
            || reservation.effects() != 0
            || reservation.effect_values() != 0
            || reservation.effect_value_bytes() != 0
            || reservation.table_outputs() != 1
            || reservation.index_outputs() != 0
        {
            return Err(generation_error(
                "first production typed generation vertical requires one table and zero indexes/effects",
            ));
        }
        let prepared = PreparedRuntimeTypedInsertGeneration::reserve(
            self.target.clone(),
            self.attempt,
            geometry,
        )
        .map_err(|error| {
            generation_error(&format!("CUDA generation reservation failed: {error:?}"))
        })?;
        Ok(LiveTypedInsertGenerationCandidate {
            prepared: Some(prepared),
            completion: Arc::new(Mutex::new(SharedCompletion { proof: None })),
            identity: None,
            table: None,
            commitments: None,
            logical: None,
        })
    }

    fn launch(
        self,
        mut launch: input::ReservedGenerationLaunch<Self::Candidate, Self::Work>,
    ) -> LaunchedAttempt<Self::Candidate, Self::Work> {
        let (identity, table) = {
            let neutral = launch.input().neutral_view();
            let identity = neutral.identity();
            let mut tables = neutral.tables();
            let table = tables
                .next()
                .expect("reserved one-table neutral generation has one table");
            debug_assert!(tables.next().is_none());
            (identity, table)
        };
        let runtime_identity = RuntimeTypedInsertGenerationIdentity {
            database_id: identity.database_id,
            catalog_epoch: identity.catalog_epoch,
            catalog_digest: identity.catalog_digest,
            stable_transaction_id: identity.stable_transaction_id,
            commit_sequence: identity.commit_sequence,
            write001_typed_statement_digest: [0; 32],
        };
        let runtime_table = RuntimeTypedInsertGenerationTable {
            action: if table.resets_existing_rows {
                RuntimeTypedInsertGenerationTableAction::ResetThenRowSetInsert
            } else {
                RuntimeTypedInsertGenerationTableAction::RowSetInsert
            },
            table_map_predecessor: self.table_map_predecessor,
            stable_table_id: table.stable_table_id,
            write001_final_image_ref: 0,
            base_data_generation: table.base_data_generation,
            base_table_root: table.base_table_root,
            row_allocator_before: table.row_allocator_before,
            row_allocator_high_water: table.row_allocator_high_water,
            initial_logical_row_count: table.initial_logical_row_count,
            final_logical_row_count: table.final_logical_row_count,
            image_layout_digest: table.image_layout_digest,
            image_content_digest: table.image_content_digest,
        };
        let candidate = launch.candidate_mut();
        candidate.identity = Some(runtime_identity);
        candidate.table = Some(runtime_table);
        let completion = Arc::clone(&candidate.completion);
        let prepared = candidate
            .prepared
            .take()
            .expect("reserved generation candidate owns one unlaunched CUDA program");
        let submission = {
            let neutral = launch.input().neutral_view();
            prepared.launch(|encoder| {
                encoder.write_identity(runtime_identity);
                encoder.write_table(runtime_table);
                for row in neutral.rows() {
                    encoder.write_row(RuntimeTypedInsertGenerationRow {
                        stable_table_id: row.stable_table_id,
                        stable_row_id: row.stable_row_id,
                        source_statement_ordinal: row.source_statement_ordinal,
                        source_row_ordinal: row.source_row_ordinal,
                        cell_count: row.cell_count,
                    });
                }
                for cell in neutral.cells() {
                    encoder.write_cell(RuntimeTypedInsertGenerationCell {
                        catalog_column_ordinal: cell.catalog_column_ordinal,
                        stable_column_id: cell.stable_column_id,
                        attnum: cell.attnum,
                        storage: cell.storage,
                        declared_type_oid: cell.declared_type_oid,
                        signed_type_size: cell.signed_type_size,
                        is_null: cell.is_null,
                        value: cell.value,
                    });
                }
            })
        };
        LaunchedAttempt::launched(
            launch,
            LiveTypedInsertGenerationWork {
                state: WorkState::Submitted(submission),
                completion,
            },
        )
    }

    fn finalize_quiesced(
        candidate: &mut Self::Candidate,
        outputs: &mut input::ReservedGenerationOutputs,
    ) -> Result<(), EngineError> {
        let proof = candidate
            .completion
            .lock()
            .map_err(|_| generation_error("CUDA generation completion rendezvous was poisoned"))?
            .proof
            .take()
            .ok_or_else(|| generation_error("quiesced CUDA generation produced no proof"))?;
        let identity = candidate
            .identity
            .ok_or_else(|| generation_error("generation candidate lost its identity"))?;
        let table = candidate
            .table
            .ok_or_else(|| generation_error("generation candidate lost its table"))?;
        let mut generation_input_digest = [0_u8; 32];
        let mut initial_database_root = [0_u8; 32];
        let mut final_table_root = [0_u8; 32];
        let mut final_database_root = [0_u8; 32];
        proof.consume(|_attempt, commitments, logical| {
            commitments.copy_generation_input_into(&mut generation_input_digest);
            commitments.copy_initial_database_root_into(&mut initial_database_root);
            commitments.copy_final_table_root_into(&mut final_table_root);
            commitments.copy_final_database_root_into(&mut final_database_root);
            candidate.commitments = Some(commitments);
            candidate.logical = Some(logical);
        });
        outputs.set_header(input::GenerationOutputHeader {
            filled: true,
            duplicate: false,
            root_descriptor_version: 1,
            database_id: identity.database_id,
            catalog_epoch: identity.catalog_epoch,
            catalog_digest: identity.catalog_digest,
            stable_transaction_id: identity.stable_transaction_id,
            commit_sequence: identity.commit_sequence,
            initial_database_root,
            generation_input_digest,
            final_database_root,
        });
        outputs.set_table(
            0,
            input::GenerationOutputTable {
                filled: true,
                duplicate: false,
                stable_table_id: table.stable_table_id,
                final_data_generation: identity.commit_sequence,
                final_table_root,
                final_logical_row_count: table.final_logical_row_count,
                index_start: 0,
                index_count: 0,
            },
        );
        Ok(())
    }
}

impl SemanticsV2ReservedGenerationWork for LiveTypedInsertGenerationWork {
    fn drain_once(&mut self) -> DrainOutcome {
        let state = std::mem::replace(&mut self.state, WorkState::Finished);
        let completion = match state {
            WorkState::Submitted(submission) => submission.complete(),
            WorkState::Unknown(unknown) => unknown.retry_complete(),
            WorkState::Finished => {
                return DrainOutcome::QuiescenceUnproven(generation_error(
                    "CUDA generation drain was already consumed",
                ));
            }
        };
        match completion {
            RuntimeTypedInsertGenerationCompletion::Quiesced(Ok(proof)) => {
                match self.completion.lock() {
                    Ok(mut completion) => completion.proof = Some(proof),
                    Err(_) => {
                        return DrainOutcome::Quiesced {
                            execution: Err(generation_error(
                                "CUDA generation completion rendezvous was poisoned",
                            )),
                        };
                    }
                }
                DrainOutcome::Quiesced { execution: Ok(()) }
            }
            RuntimeTypedInsertGenerationCompletion::Quiesced(Err(error)) => {
                DrainOutcome::Quiesced {
                    execution: Err(generation_error(&format!(
                        "CUDA generation execution failed: {error:?}"
                    ))),
                }
            }
            RuntimeTypedInsertGenerationCompletion::UnknownQuiescence(unknown) => {
                let error = generation_error(&format!(
                    "CUDA generation quiescence is unproven: {:?}",
                    unknown.error()
                ));
                self.state = WorkState::Unknown(unknown);
                DrainOutcome::QuiescenceUnproven(error)
            }
        }
    }
}
