//! Sealed engine-to-execution adapter for the first real bootstrap rebuild operator.
//!
//! It accepts only the root-free compiler capability, pins an actual primary CUDA target through
//! the execution type system, and pairs opaque completion slots with the independently compiled
//! proof layout. Expected roots remain comparator-only and never cross this module.

use gpu_db_execution::{
    OpaqueRuntimeGenerationRebuildProof, PreparedRuntimeGenerationRebuild,
    RuntimeGenerationRebuildAttempt, RuntimeGenerationRebuildCompletion,
    RuntimeGenerationRebuildError, RuntimeGenerationRebuildInput,
    RuntimeGenerationRebuildPrepareFailure, RuntimeGenerationRebuildRoleSpan,
    RuntimeGenerationRebuildShard, RuntimeGenerationRebuildShardRoles,
    RuntimeGenerationRebuildSource, RuntimeGenerationRebuildTarget,
    RuntimeGenerationRebuildUnknownQuiescence,
};
use gpu_db_sql::SqlType;

use super::{
    bootstrap_publication::{
        BootstrapPhysicalLayoutRoleKind, BootstrapRebuildRootFreeResource,
        BootstrapResourceLedgerKind, BootstrapResourceLedgerOwner,
    },
    bootstrap_rebuild::{
        BootstrapRebuildCompilerInput, BootstrapRebuildGpuParts,
        BootstrapRebuildPreparedProofLayout,
    },
    resources, DataGenerationError,
};

/// Prepared engine pairing for one actual V1 single-table INT4 rebuild. It has no comparator
/// roots, installation authority, or publication hook.
#[must_use = "a prepared bootstrap rebuild must be enqueued or intentionally dropped"]
pub(super) struct PreparedBootstrapRuntimeGenerationRebuild {
    proof: BootstrapRebuildPreparedProofLayout,
    prepared: PreparedRuntimeGenerationRebuild,
}

/// Completion pairs a drained opaque execution proof with exactly the sealed engine proof layout.
#[must_use = "an in-flight bootstrap rebuild must be completed or safely retained"]
pub(super) struct BootstrapRuntimeGenerationRebuildSubmission {
    proof: BootstrapRebuildPreparedProofLayout,
    submission: gpu_db_execution::RuntimeGenerationRebuildSubmission,
}

/// The adapter returns its compiler capability unchanged for a semantic/owner bridge rejection,
/// and otherwise leaves the execution input inside its own move-only preparation failure.
pub(super) enum BootstrapRuntimeGenerationRebuildPrepareFailure {
    Compiler {
        error: DataGenerationError,
        compiler: BootstrapRebuildCompilerInput,
    },
    Execution {
        proof: BootstrapRebuildPreparedProofLayout,
        failure: Box<RuntimeGenerationRebuildPrepareFailure>,
    },
}

/// A successful proof contains no expected root comparison. A later engine comparator can consume
/// it with independently retained expectation evidence; this adapter cannot manufacture that link.
pub(super) struct BootstrapRuntimeGenerationRebuildProof {
    proof: OpaqueRuntimeGenerationRebuildProof,
}

#[allow(
    clippy::large_enum_variant,
    reason = "boxing after the execution enqueue would allocate on the unknown-quiescence path"
)]
pub(super) enum BootstrapRuntimeGenerationRebuildCompletion {
    Quiesced(Result<BootstrapRuntimeGenerationRebuildProof, RuntimeGenerationRebuildError>),
    UnknownQuiescence(BootstrapRuntimeGenerationRebuildUnknownQuiescence),
}

pub(super) struct BootstrapRuntimeGenerationRebuildUnknownQuiescence {
    proof: BootstrapRebuildPreparedProofLayout,
    unknown: RuntimeGenerationRebuildUnknownQuiescence,
}

/// Consume one accepted root-free compiler input at the only engine-to-execution bridge. This
/// slice deliberately rejects every grammar it does not fully compute on GPU: all types other
/// than nullable INT4, indexes, predicates/sidecars, multiple tables, or non-V1 source facts.
pub(super) fn prepare_v1_single_table_int4_rebuild(
    compiler: BootstrapRebuildCompilerInput,
    target: RuntimeGenerationRebuildTarget,
    attempt: RuntimeGenerationRebuildAttempt,
) -> Result<
    PreparedBootstrapRuntimeGenerationRebuild,
    Box<BootstrapRuntimeGenerationRebuildPrepareFailure>,
> {
    let parts = compiler.into_gpu_parts();
    let descriptor = match compile_v1_single_table_int4_input(parts, target, attempt) {
        Ok(descriptor) => descriptor,
        Err(failure) => {
            let (error, parts) = *failure;
            return Err(Box::new(
                BootstrapRuntimeGenerationRebuildPrepareFailure::Compiler {
                    error,
                    compiler: parts.into_compiler_input(),
                },
            ));
        }
    };
    let proof = descriptor.proof;
    match PreparedRuntimeGenerationRebuild::prepare(descriptor.input) {
        Ok(prepared) => Ok(PreparedBootstrapRuntimeGenerationRebuild { proof, prepared }),
        Err(failure) => Err(Box::new(
            BootstrapRuntimeGenerationRebuildPrepareFailure::Execution { proof, failure },
        )),
    }
}

impl PreparedBootstrapRuntimeGenerationRebuild {
    pub(super) fn enqueue(self) -> BootstrapRuntimeGenerationRebuildSubmission {
        BootstrapRuntimeGenerationRebuildSubmission {
            proof: self.proof,
            submission: self.prepared.enqueue(),
        }
    }
}

impl BootstrapRuntimeGenerationRebuildSubmission {
    pub(super) fn complete(self) -> BootstrapRuntimeGenerationRebuildCompletion {
        complete_with_proof(self.proof, self.submission.complete())
    }
}

impl BootstrapRuntimeGenerationRebuildUnknownQuiescence {
    pub(super) fn error(&self) -> &RuntimeGenerationRebuildError {
        self.unknown.error()
    }

    pub(super) fn retry_complete(self) -> BootstrapRuntimeGenerationRebuildCompletion {
        complete_with_proof(self.proof, self.unknown.retry_complete())
    }
}

impl BootstrapRuntimeGenerationRebuildProof {
    pub(super) fn attempt(&self) -> u64 {
        self.proof.attempt().get()
    }

    pub(super) fn slot_count(&self) -> usize {
        self.proof.slot_count()
    }
}

fn complete_with_proof(
    proof: BootstrapRebuildPreparedProofLayout,
    completion: RuntimeGenerationRebuildCompletion,
) -> BootstrapRuntimeGenerationRebuildCompletion {
    match completion {
        RuntimeGenerationRebuildCompletion::Quiesced(result) => {
            BootstrapRuntimeGenerationRebuildCompletion::Quiesced(result.and_then(|opaque| {
                if opaque.root_format() != 1
                    || opaque.slot_count() != proof.exact_output_count() as usize
                    || !proof.is_v1_single_table_int4_contract()
                {
                    return Err(RuntimeGenerationRebuildError::DeviceRejected(8));
                }
                Ok(BootstrapRuntimeGenerationRebuildProof { proof: opaque })
            }))
        }
        RuntimeGenerationRebuildCompletion::UnknownQuiescence(unknown) => {
            BootstrapRuntimeGenerationRebuildCompletion::UnknownQuiescence(
                BootstrapRuntimeGenerationRebuildUnknownQuiescence { proof, unknown },
            )
        }
    }
}

struct CompiledInput {
    input: RuntimeGenerationRebuildInput,
    proof: BootstrapRebuildPreparedProofLayout,
}

fn compile_v1_single_table_int4_input(
    parts: BootstrapRebuildGpuParts,
    target: RuntimeGenerationRebuildTarget,
    attempt: RuntimeGenerationRebuildAttempt,
) -> Result<CompiledInput, Box<(DataGenerationError, BootstrapRebuildGpuParts)>> {
    let validation = validate_v1_single_table_int4_shape(&parts);
    if let Err(error) = validation {
        return Err(Box::new((error, parts)));
    }
    let sources = match resources::into_runtime_generation_rebuild_sources(parts.owners, &target) {
        Ok(sources) => sources,
        Err(failure) => {
            let (error, owners) = failure.into_parts();
            return Err(Box::new((
                error,
                BootstrapRebuildGpuParts {
                    owners,
                    source: parts.source,
                    proof: parts.proof,
                },
            )));
        }
    };
    let table = &parts.source.tables[0];
    let column = &table.columns[0];
    let shards = compile_shards(
        &parts.source.resources,
        sources,
        table.table_id.get(),
        column.column_id.get(),
    );
    Ok(CompiledInput {
        input: RuntimeGenerationRebuildInput::new(
            target,
            attempt,
            parts.source.database_id.bytes(),
            table.table_id.get(),
            table.data_generation.get(),
            parts.source.covered_through,
            table.logical_row_count,
            column.column_id.get(),
            column.attnum,
            column.declared_type_oid,
            column.signed_type_size,
            shards,
        ),
        proof: parts.proof,
    })
}

fn validate_v1_single_table_int4_shape(
    parts: &BootstrapRebuildGpuParts,
) -> Result<(), DataGenerationError> {
    let source = &parts.source;
    if source.root_format.get() != 1
        || source.tables.len() != 1
        || source.tables[0].logical_row_count == 0
        || source.tables[0].columns.len() != 1
        || !source.tables[0].indexes.is_empty()
        || source.covered_through == 0
        || !parts.proof.is_v1_single_table_int4_contract()
    {
        return Err(DataGenerationError::Invalid(
            "unsupported bootstrap V1 single-table INT4 rebuild shape",
        ));
    }
    let column = &source.tables[0].columns[0];
    if column.sql_type != SqlType::Int4
        || column.attnum <= 0
        || column.declared_type_oid != SqlType::Int4.postgres_oid()
        || column.signed_type_size != SqlType::Int4.type_size()
    {
        return Err(DataGenerationError::Invalid(
            "unsupported bootstrap V1 single-table INT4 column",
        ));
    }
    for resource in source.resources.iter() {
        match (resource.kind, resource.owner) {
            (
                BootstrapResourceLedgerKind::DatabaseManifest,
                BootstrapResourceLedgerOwner::Database,
            )
            | (BootstrapResourceLedgerKind::StatusView, BootstrapResourceLedgerOwner::Status) => {}
            (
                BootstrapResourceLedgerKind::TablePayload,
                BootstrapResourceLedgerOwner::Table(table_id),
            ) if table_id == source.tables[0].table_id => {}
            _ => {
                return Err(DataGenerationError::Invalid(
                    "unsupported bootstrap V1 rebuild resource",
                ))
            }
        }
        if resource.kind == BootstrapResourceLedgerKind::TablePayload {
            compile_roles(resource, column.column_id.get())?;
        }
    }
    Ok(())
}

fn compile_shards(
    resources: &[BootstrapRebuildRootFreeResource],
    sources: Box<[RuntimeGenerationRebuildSource]>,
    table_id: u64,
    column_id: u64,
) -> Box<[RuntimeGenerationRebuildShard]> {
    assert_eq!(
        resources.len(),
        sources.len(),
        "validated execution source coverage"
    );
    let mut shards = Vec::new();
    for (resource, source) in resources.iter().zip(sources.into_vec()) {
        if resource.kind != BootstrapResourceLedgerKind::TablePayload
            || resource.owner
                != BootstrapResourceLedgerOwner::Table(
                    super::digest::StableTableId::new(table_id)
                        .expect("validated nonzero table ID"),
                )
        {
            continue;
        }
        let roles = compile_roles(resource, column_id)
            .expect("validated root-free table payload roles remain compilable");
        shards.push(RuntimeGenerationRebuildShard::new(
            source,
            resource.layout.row_start,
            resource.layout.row_count,
            roles,
        ));
    }
    assert!(
        !shards.is_empty(),
        "validated root-free source has table payloads"
    );
    shards.into_boxed_slice()
}

fn compile_roles(
    resource: &BootstrapRebuildRootFreeResource,
    column_id: u64,
) -> Result<RuntimeGenerationRebuildShardRoles, DataGenerationError> {
    let mut stable = None;
    let mut validity = None;
    let mut values = None;
    let mut created = None;
    let mut deleted = None;
    for role in resource.layout.roles.iter() {
        let span = RuntimeGenerationRebuildRoleSpan {
            byte_offset: role.byte_offset,
            byte_len: role.byte_len,
        };
        match role.kind {
            BootstrapPhysicalLayoutRoleKind::StableRowId => assign_role(&mut stable, span)?,
            BootstrapPhysicalLayoutRoleKind::Validity
                if role.column_id.is_some_and(|id| id.get() == column_id)
                    && role.sql_type == Some(SqlType::Int4) =>
            {
                assign_role(&mut validity, span)?
            }
            BootstrapPhysicalLayoutRoleKind::Value
                if role.column_id.is_some_and(|id| id.get() == column_id)
                    && role.sql_type == Some(SqlType::Int4) =>
            {
                assign_role(&mut values, span)?
            }
            BootstrapPhysicalLayoutRoleKind::CreatedBy => assign_role(&mut created, span)?,
            BootstrapPhysicalLayoutRoleKind::DeletedBy => assign_role(&mut deleted, span)?,
            _ => {
                return Err(DataGenerationError::Invalid(
                    "unsupported bootstrap V1 rebuild physical role",
                ))
            }
        }
    }
    Ok(RuntimeGenerationRebuildShardRoles {
        stable_row_ids: required_role(stable, "bootstrap rebuild stable row IDs")?,
        validity: required_role(validity, "bootstrap rebuild validity")?,
        values: required_role(values, "bootstrap rebuild values")?,
        created_by: required_role(created, "bootstrap rebuild created-by")?,
        deleted_by: required_role(deleted, "bootstrap rebuild deleted-by")?,
    })
}

fn assign_role(
    slot: &mut Option<RuntimeGenerationRebuildRoleSpan>,
    role: RuntimeGenerationRebuildRoleSpan,
) -> Result<(), DataGenerationError> {
    if slot.replace(role).is_some() {
        return Err(DataGenerationError::Invalid(
            "duplicate bootstrap rebuild physical role",
        ));
    }
    Ok(())
}

fn required_role(
    role: Option<RuntimeGenerationRebuildRoleSpan>,
    label: &'static str,
) -> Result<RuntimeGenerationRebuildRoleSpan, DataGenerationError> {
    role.ok_or(DataGenerationError::Missing(label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_data_generation::{bootstrap_rebuild, resources};

    #[test]
    fn adapter_rejects_indexes_before_any_execution_owner_is_consumed() {
        let attached = resources::attached_resources_for_bootstrap_rebuild_test();
        let (compiler, _expectations) = bootstrap_rebuild::prepare_bootstrap_rebuild(attached)
            .unwrap_or_else(|_| panic!("prepared root-free compiler input"));
        let parts = compiler.into_gpu_parts();
        assert_eq!(
            validate_v1_single_table_int4_shape(&parts),
            Err(DataGenerationError::Invalid(
                "unsupported bootstrap V1 single-table INT4 rebuild shape"
            ))
        );
        let _compiler = parts.into_compiler_input();
    }
}
