//! Pre-queue GPU proof for typed INSERT row-local CHECK constraints.
//!
//! The prepared plan owns the resulting proof, while the short-lived source is only a physical
//! device operator input.  This module deliberately contains no append, allocator, WAL, or
//! publication operation.

use super::{CatalogSnapshot, Engine, EngineError, Index};
use crate::relational_model::RelationalTable;
use crate::typed_insert_batch::TypedInsertBatch;
use crate::typed_insert_batch::TypedInsertConstraintDeviceSource;
use crate::{CheckOperandIdentityVersion, ExecuteError, SelectFilterOp, SqlType, SqlValue};

/// A move-only catalog witness.  The fields remain private so only a successful device verdict
/// can create a checked proof and only this module can compare it at the commit gate.
pub(super) struct RowLocalConstraintProof {
    table_oid: u32,
    schema_digest: gpu_db_wal::CanonicalDigest,
    catalog_seq: Index,
    checks: Box<[RowLocalCheckBinding]>,
}

struct RowLocalCheckBinding {
    name: String,
    column_id: u32,
    op: SelectFilterOp,
    value: SqlValue,
    resolved_input_type: SqlType,
    identity_version: CheckOperandIdentityVersion,
}

impl RowLocalConstraintProof {
    pub(super) fn vacuous(batch: &TypedInsertBatch) -> Self {
        let (table_oid, schema_digest, catalog_seq) = batch.row_local_constraint_target();
        Self {
            table_oid,
            schema_digest,
            catalog_seq,
            checks: Box::default(),
        }
    }

    pub(super) fn checked(
        batch: &TypedInsertBatch,
        table: &RelationalTable,
    ) -> Result<Self, EngineError> {
        let (table_oid, schema_digest, catalog_seq) = batch.row_local_constraint_target();
        let checks = table
            .check_constraints
            .iter()
            .map(|constraint| {
                let column_id = table
                    .columns
                    .iter()
                    .find(|column| column.name == constraint.column)
                    .map(|column| column.id)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!(
                            "device CHECK column \"{}\" is absent from relation \"{}\"",
                            constraint.column, table.name
                        ))
                    })?;
                Ok(RowLocalCheckBinding {
                    name: constraint.name.clone(),
                    column_id,
                    op: constraint.op,
                    value: constraint.value.clone(),
                    resolved_input_type: constraint.resolved_input_type,
                    identity_version: constraint.identity_version,
                })
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        Ok(Self {
            table_oid,
            schema_digest,
            catalog_seq,
            checks: checks.into(),
        })
    }

    pub(super) fn matches_current_catalog(&self, catalog: &CatalogSnapshot) -> bool {
        catalog.commit_seq == self.catalog_seq && self.matches_catalog_binding(catalog)
    }

    /// The proof-only indexed resident pass may observe an unrelated committed DML generation
    /// after its batch-local CHECK pass.  The checked table and its ordered constraints must
    /// still match exactly; only the monotonic global catalog sequence may have advanced.
    #[cfg(test)]
    pub(super) fn matches_current_target_binding(&self, catalog: &CatalogSnapshot) -> bool {
        catalog.commit_seq >= self.catalog_seq && self.matches_catalog_binding(catalog)
    }

    fn matches_catalog_binding(&self, catalog: &CatalogSnapshot) -> bool {
        let Some(table) = catalog
            .relational_catalog
            .values()
            .find(|table| table.oid == self.table_oid)
        else {
            return false;
        };
        crate::engine_transaction_reset::table_schema_digest(table).ok() == Some(self.schema_digest)
            && table.check_constraints.len() == self.checks.len()
            && table
                .check_constraints
                .iter()
                .zip(self.checks.iter())
                .all(|(live, proof)| {
                    table
                        .columns
                        .iter()
                        .find(|column| column.name == live.column)
                        .is_some_and(|column| {
                            live.name == proof.name
                                && column.id == proof.column_id
                                && live.op == proof.op
                                && live.value == proof.value
                                && live.resolved_input_type == proof.resolved_input_type
                                && live.identity_version == proof.identity_version
                        })
                })
    }
}

/// CHECK-only compilation metadata.  The composite pre-WAL owner owns source lifetime,
/// allocation scope, scheduling, and cross-constraint arbitration; this leaf owns only
/// CHECK eligibility and the operator-specific setup.
pub(super) struct CompiledRowLocalChecks {
    check_ordinals: Box<[usize]>,
}

/// One bounded terminal returned by a CHECK operator.  It deliberately has no diagnostic
/// policy: the composite owner orders it against primary-key NULL and key-duplicate terminals.
pub(super) struct RowLocalCheckCandidate {
    pub(super) row: u32,
    pub(super) check_ordinal: usize,
}

/// Static eligibility remains deliberately narrow. Indexes and foreign keys need their own
/// global proofs; CHECK eligibility is exactly the catalog-resolved device violation compiler.
pub(crate) fn table_has_supported_row_local_checks(table: &RelationalTable) -> bool {
    table.indexes.is_empty() && table.foreign_keys.is_empty() && checks_are_device_supported(table)
}

/// Device CHECK eligibility without global-index or foreign-key policy.  The production builder
/// deliberately retains its narrower wrapper above; the proof-only test seam uses this exact
/// predicate alongside key-proof eligibility.
pub(crate) fn checks_are_device_supported(table: &RelationalTable) -> bool {
    table.check_constraints.iter().all(|constraint| {
        table
            .columns
            .iter()
            .position(|column| column.name == constraint.column)
            .is_some_and(|column| {
                crate::check_violation_expr::compile_check_violation(table, constraint, column)
                    .is_some()
            })
    })
}

pub(super) fn compile(
    batch: &TypedInsertBatch,
    table: &RelationalTable,
) -> Result<CompiledRowLocalChecks, EngineError> {
    if !checks_are_device_supported(table) {
        return Err(EngineError::ApplyFailed(format!(
            "device CHECK proof is unavailable for relation \"{}\"",
            table.name
        )));
    }
    for constraint in &table.check_constraints {
        crate::check_violation_expr::validate_check_literal_at_evaluation(constraint)?;
    }
    let _ = batch;
    Ok(CompiledRowLocalChecks {
        check_ordinals: (0..table.check_constraints.len()).collect(),
    })
}

/// Exact CHECK VM high-water excluding the caller-owned common device source.
pub(super) fn scratch_bytes(
    batch: &TypedInsertBatch,
    table: &RelationalTable,
) -> Result<u64, EngineError> {
    if table.check_constraints.is_empty() {
        return Ok(0);
    }
    let rows = usize::try_from(batch.binary_insert_template_row_count())
        .expect("u32 rows fit usize on supported hosts");
    let mask_bytes = pooled_bucket(rows.checked_mul(std::mem::size_of::<i32>()).ok_or_else(
        || EngineError::ApplyFailed("device CHECK mask extent overflows".to_string()),
    )?)?;
    let value_bytes = table
        .check_constraints
        .iter()
        .filter_map(|constraint| {
            table
                .columns
                .iter()
                .find(|column| column.name == constraint.column)
                .map(|column| match column.ty {
                    crate::SqlType::Numeric { .. } | crate::SqlType::Uuid => 16_usize,
                    crate::SqlType::Int8 | crate::SqlType::Timestamp => 8,
                    _ => 4,
                })
        })
        .max()
        .unwrap_or(4)
        .checked_mul(rows)
        .ok_or_else(|| {
            EngineError::ApplyFailed("device CHECK value extent overflows".to_string())
        })?;
    let vm_bucket = mask_bytes.max(pooled_bucket(value_bytes)?);
    let literal_bucket = table
        .check_constraints
        .iter()
        .filter_map(|constraint| match &constraint.value {
            SqlValue::Text(value) => Some(value.len().max(1)),
            SqlValue::Uuid(_) => Some(16),
            _ => None,
        })
        .max()
        .map(pooled_bucket)
        .transpose()?
        .unwrap_or(0);
    let counter_bytes = pooled_bucket(std::mem::size_of::<u32>())?;
    // One single-column CHECK compiles to either a value buffer + mask or a varlen mask, then a
    // nullable validity mask and a MaskBinary output.  Three `vm_bucket`s therefore bound every
    // live VM stack shape (including a 16-byte numeric input); `overflow` and the terminal flag
    // are independent u32 pooled allocations.  A text/uuid needle exists only beside one mask,
    // but is retained here as a conservative additive bound before source allocation.
    let vm_peak = vm_bucket
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(counter_bytes))
        .and_then(|bytes| bytes.checked_add(literal_bucket))
        .and_then(|bytes| bytes.checked_add(counter_bytes))
        .ok_or_else(|| EngineError::ApplyFailed("device CHECK VM peak overflows".to_string()))?;
    Ok(vm_peak)
}

/// Evaluate every CHECK through the caller-owned source.  No early return may skip a later
/// semantic operator: the composite owner arbitrates all bounded terminals after this returns.
pub(super) fn evaluate(
    engine: &Engine,
    batch: &TypedInsertBatch,
    table: &RelationalTable,
    source: &TypedInsertConstraintDeviceSource,
    compiled: &CompiledRowLocalChecks,
) -> Result<Box<[RowLocalCheckCandidate]>, EngineError> {
    let mut violations = Vec::new();
    for &check_ordinal in &compiled.check_ordinals {
        let constraint = &table.check_constraints[check_ordinal];
        let column = table
            .columns
            .iter()
            .position(|column| column.name == constraint.column)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "device CHECK column \"{}\" is absent from relation \"{}\"",
                    constraint.column, table.name
                ))
            })?;
        let violation =
            crate::check_violation_expr::compile_check_violation(table, constraint, column)
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "device CHECK violation unavailable for constraint \"{}\"",
                        constraint.name
                    ))
                })?;
        let mask = engine
            .resident_predicate_device_mask(
                Some(&violation),
                table,
                source.descriptor(),
                source.memory(),
                batch.binary_insert_template_row_count(),
                None,
            )
            .map_err(as_engine_error)?
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "device CHECK mask is unavailable for constraint \"{}\"",
                    constraint.name
                ))
            })?;
        #[cfg(feature = "probe-timing")]
        engine.record_insert_probe_row_local_check_launch();
        let first_row = mask.first_true_row().map_err(|error| {
            EngineError::ApplyFailed(format!(
                "device CHECK verdict failed for constraint \"{}\": {error}",
                constraint.name
            ))
        })?;
        #[cfg(feature = "probe-timing")]
        engine.record_insert_probe_row_local_check_verdict();
        if let Some(row) = first_row {
            violations.push(RowLocalCheckCandidate { row, check_ordinal });
        }
    }
    Ok(violations.into())
}

pub(super) fn seal_after_success(
    batch: &TypedInsertBatch,
    table: &RelationalTable,
    _compiled: &CompiledRowLocalChecks,
) -> Result<RowLocalConstraintProof, EngineError> {
    if table.check_constraints.is_empty() {
        Ok(RowLocalConstraintProof::vacuous(batch))
    } else {
        RowLocalConstraintProof::checked(batch, table)
    }
}

fn pooled_bucket(bytes: usize) -> Result<u64, EngineError> {
    bytes
        .max(256)
        .checked_next_power_of_two()
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            EngineError::ApplyFailed("device CHECK scratch bucket overflows".to_string())
        })
}

pub(super) fn as_engine_error(error: ExecuteError) -> EngineError {
    match error {
        ExecuteError::Engine(error) => error,
        other => EngineError::ApplyFailed(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_rejects_catalog_sequence_and_each_check_identity_component() {
        let engine = crate::Engine::new_local_test_engine();
        engine
            .execute_text(
                1,
                "CREATE TABLE checked_identity (id int4, other int4, \
                 CONSTRAINT checked_identity_rule CHECK (id > 0))",
            )
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let command =
            gpu_db_sql::parse_command("INSERT INTO checked_identity VALUES (1, 9)").unwrap();
        let batch = crate::typed_insert_batch::try_prepare_typed_insert_batch(
            &command,
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .expect("CHECK-only typed batch stays eligible before device proof");
        let build_proof = |catalog: &CatalogSnapshot| {
            let table = catalog.relational_catalog.get("checked_identity").unwrap();
            RowLocalConstraintProof::checked(&batch, table).unwrap()
        };
        assert!(build_proof(&catalog).matches_current_catalog(&catalog));

        let mut seq_drift = (*catalog).clone();
        seq_drift.commit_seq += 1;
        assert!(!build_proof(&catalog).matches_current_catalog(&seq_drift));

        for sabotage in ["name", "column", "op", "literal", "input_type", "identity"] {
            let mut drift = (*catalog).clone();
            let table = drift
                .relational_catalog
                .get_mut("checked_identity")
                .unwrap();
            let check = &mut table.check_constraints[0];
            match sabotage {
                "name" => check.name.push_str("_drift"),
                "column" => check.column = "other".to_string(),
                "op" => check.op = SelectFilterOp::Lte,
                "literal" => check.value = SqlValue::Int4(2),
                "input_type" => check.resolved_input_type = crate::SqlType::Int8,
                "identity" => check.identity_version = crate::CheckOperandIdentityVersion::LegacyV1,
                _ => unreachable!(),
            }
            let mut proof = build_proof(&catalog);
            proof.schema_digest = crate::engine_transaction_reset::table_schema_digest(table)
                .expect("sabotage table stays digestible");
            assert!(
                !proof.matches_current_catalog(&drift),
                "CHECK proof accepted {sabotage} drift"
            );
        }
    }

    #[test]
    fn check_proof_source_has_no_legacy_row_or_host_verdict_path() {
        let source = include_str!("row_local_constraints.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production CHECK proof precedes tests");
        assert!(!source.contains("build_transient_relation_residency"));
        assert!(!source.contains("lower_resident_predicate"));
        assert!(!source.contains("predicate_mask_indices_u32"));
        assert!(!source.contains("Vec<Vec<SqlValue>>"));
        assert!(!source.contains("mask.any_true()"));
        assert!(!source.contains("CudaAllocationScope"));
        assert!(!source.contains("row_local_constraint_device_source"));
        assert!(source.contains("mask.first_true_row()"));
        assert!(source.contains("RowLocalCheckCandidate"));
        assert!(source.contains("pub(super) fn evaluate"));
        assert!(source.contains("pub(super) fn seal_after_success"));
        assert!(source.contains("RowLocalConstraintProof"));
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_check_only_typed_insert_normalizes_cross_numeric_checks_pre_wal() {
        let engine = crate::Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(
                1,
                "CREATE TABLE checked_values (id int4, small_value smallint, fractional int4, \
                 amount numeric(8,2), score int4, timestamp_value timestamp, date_value date, \
                 CONSTRAINT checked_id_positive CHECK (id > 0), \
                 CONSTRAINT checked_id_wide CHECK (id < 2147483648), \
                 CONSTRAINT checked_small_wide CHECK (small_value < 32768), \
                 CONSTRAINT checked_fractional CHECK (fractional < 1.2), \
                 CONSTRAINT checked_amount_finer CHECK (amount < 12.345), \
                 CONSTRAINT checked_score_ceiling CHECK (score <= 10), \
                 CONSTRAINT checked_timestamp_before_date CHECK (timestamp_value < '2000-01-02'::date), \
                 CONSTRAINT checked_date_after_timestamp CHECK (date_value > '1999-12-31 23:59:59.999999'::timestamp))",
            )
            .unwrap();
        #[cfg(feature = "probe-timing")]
        let before = engine.insert_probe_snapshot();
        let mut first_id = 1_i32;
        for (request, rows) in [(2_u64, 31_i32), (3, 32), (4, 33)] {
            let values = (first_id..first_id + rows)
                .map(|id| {
                    if id == first_id + rows - 1 {
                        format!(
                            "({id}, 7, 1, 12.34, NULL, '2000-01-01 23:59:59.999999'::timestamp, NULL)"
                        )
                    } else {
                        format!(
                            "({id}, 7, 1, 12.34, 1, '2000-01-01 23:59:59.999999'::timestamp, '2000-01-01'::date)"
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            let insert = format!("INSERT INTO checked_values VALUES {values}");
            engine.execute_dml_concurrent(request, &insert).unwrap();
            first_id += rows;
        }
        #[cfg(feature = "probe-timing")]
        {
            let probe = engine.insert_probe_snapshot().delta_since(before);
            assert_eq!(probe.direct_fixed_insert_carriers, 3);
            assert_eq!(probe.fixed_insert_typed_commits, 3);
            assert_eq!(probe.legacy_insert_delta_builds, 0);
            assert_eq!(probe.predicted_row_keys_materialized, 0);
            assert_eq!(probe.fixed_insert_legacy_fallbacks, 0);
            assert_eq!(probe.row_local_check_launches, 24);
            assert_eq!(probe.row_local_check_verdicts, 24);
        }
        let readback = engine
            .execute_relational_select_text(
                "SELECT id, small_value, fractional, amount, score, timestamp_value, date_value FROM checked_values ORDER BY id",
            )
            .unwrap();
        assert_eq!(readback.executed_target, crate::DeviceTarget::Gpu(0));
        assert_eq!(readback.rows.len(), 96);
        assert_eq!(
            readback.rows[0],
            vec![
                SqlValue::Int4(1),
                SqlValue::Int2(7),
                SqlValue::Int4(1),
                SqlValue::Numeric(crate::Decimal128::new(1_234, 2)),
                SqlValue::Int4(1),
                SqlValue::Timestamp(86_399_999_999),
                SqlValue::Date(0),
            ]
        );
        assert_eq!(
            readback.rows[30],
            vec![
                SqlValue::Int4(31),
                SqlValue::Int2(7),
                SqlValue::Int4(1),
                SqlValue::Numeric(crate::Decimal128::new(1_234, 2)),
                SqlValue::Null,
                SqlValue::Timestamp(86_399_999_999),
                SqlValue::Null,
            ]
        );
        assert!(matches!(readback.rows[62][4], SqlValue::Null));
        assert!(matches!(readback.rows[95][4], SqlValue::Null));
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let error = engine
            .execute_dml_concurrent(5, "INSERT INTO checked_values VALUES (97, 7, 2, 12.34, 1, '2000-01-01'::timestamp, '2000-01-01'::date)")
            .unwrap_err();
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::CheckViolation(message))
                if message.contains("checked_fractional")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        let error = engine
            .execute_dml_concurrent(6, "INSERT INTO checked_values VALUES (97, 7, 1, 12.35, 1, '2000-01-01'::timestamp, '2000-01-01'::date)")
            .unwrap_err();
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::CheckViolation(message))
                if message.contains("checked_amount_finer")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        let error = engine
            .execute_dml_concurrent(7, "INSERT INTO checked_values VALUES (-1, 7, 1, 12.34, 1, '2000-01-01'::timestamp, '2000-01-01'::date)")
            .unwrap_err();
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::CheckViolation(message))
                if message.contains("checked_id_positive")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        let error = engine
            .execute_dml_concurrent(8, "INSERT INTO checked_values VALUES (97, 7, 1, 12.34, 1, '2000-01-02'::timestamp, '2000-01-01'::date)")
            .unwrap_err();
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::CheckViolation(message))
                if message.contains("checked_timestamp_before_date")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        let error = engine
            .execute_dml_concurrent(9, "INSERT INTO checked_values VALUES (97, 7, 1, 12.34, 1, '2000-01-01'::timestamp, '1999-12-31'::date)")
            .unwrap_err();
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::CheckViolation(message))
                if message.contains("checked_date_after_timestamp")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_check_verdict_is_row_major_then_relcache_name_order_before_wal() {
        let engine = crate::Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(64);
        engine
            .execute_text(
                1,
                "CREATE TABLE ranked_checks (a int4, b int4, \
                 CONSTRAINT z_rank CHECK (a > 0), \
                 CONSTRAINT a_rank CHECK (b > 0))",
            )
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let assert_check = |request,
                            table: &str,
                            sql: &str,
                            expected_constraint: &str,
                            wal_before,
                            row_id_before| {
            let error = engine.execute_dml_concurrent(request, sql).unwrap_err();
            assert!(matches!(
                error,
                ExecuteError::Engine(EngineError::CheckViolation(message))
                    if message == format!(
                        "new row for relation \"{table}\" violates check constraint \"{expected_constraint}\""
                    )
            ));
            assert_eq!(engine.durable_wal_records().len(), wal_before);
            assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        };

        // The lexically later z_rank fails row 0 while a_rank fails row 1, so row-major ordering
        // still selects z_rank. Repeat the same device work to prove the atomicMin terminal is
        // deterministic despite its parallel traversal.
        for request in 2..=4 {
            assert_check(
                request,
                "ranked_checks",
                "INSERT INTO ranked_checks VALUES (-1, 1), (1, -1)",
                "z_rank",
                wal_before,
                row_id_before,
            );
        }
        // Both checks fail row 0, so relcache's lexical constraint-name order chooses a_rank,
        // even though z_rank was declared first.
        assert_check(
            5,
            "ranked_checks",
            "INSERT INTO ranked_checks VALUES (-1, -1)",
            "a_rank",
            wal_before,
            row_id_before,
        );

        // ALTER appends a_rank after z_rank in the catalog vector too; the verdict still follows
        // relcache's canonical name order rather than declaration/ALTER append position.
        engine
            .execute_text(
                10,
                "CREATE TABLE altered_ranked_checks (a int4, b int4, \
                 CONSTRAINT z_alter CHECK (a > 0))",
            )
            .unwrap();
        engine
            .execute_text(
                11,
                "ALTER TABLE altered_ranked_checks ADD CONSTRAINT a_alter CHECK (b > 0)",
            )
            .unwrap();
        let alter_wal_before = engine.durable_wal_records().len();
        let alter_row_id_before = engine.read_state.mvcc.current_row_id();
        assert_check(
            12,
            "altered_ranked_checks",
            "INSERT INTO altered_ranked_checks VALUES (-1, -1)",
            "a_alter",
            alter_wal_before,
            alter_row_id_before,
        );
    }

    #[test]
    fn tight_pre_wal_budget_refuses_before_device_source_wal_or_row_id() {
        let mut engine = crate::Engine::new_local_test_engine();
        engine.set_binary_wal_records_enabled(true);
        engine
            .execute_text(
                1,
                "CREATE TABLE budget_checked (id int4, \
                 CONSTRAINT budget_checked_positive CHECK (id > 0))",
            )
            .unwrap();
        engine.set_relational_residency_budget_bytes(0, 0);
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let error = engine
            .execute_dml_concurrent(2, "INSERT INTO budget_checked VALUES (1)")
            .unwrap_err();
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::ApplyFailed(message))
                if message.contains("device pre-WAL allocation refused")
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        assert!(!engine.table_device_authoritative("budget_checked"));
    }
}
