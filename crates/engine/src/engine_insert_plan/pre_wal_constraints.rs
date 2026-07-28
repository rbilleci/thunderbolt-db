//! Composite pre-WAL INSERT constraint owner.
//!
//! This is the sole owner of the short-lived source, CUDA allocation scope, sequential operator
//! schedule, and cross-class SQL diagnostic arbitration.  It completes before a WAL template,
//! allocator identity, residency append, or publication can exist.

use super::{batch_key_constraints, row_local_constraints, CatalogSnapshot, Engine, EngineError};
use crate::relational_model::RelationalTable;
use crate::typed_insert_batch::TypedInsertBatch;
use gpu_db_execution::CudaAllocationScope;

/// Move-only composite catalog witness carried by the prepared INSERT plan.
pub(super) struct PreWalConstraintProof {
    checks: row_local_constraints::RowLocalConstraintProof,
    keys: batch_key_constraints::BatchKeyConstraintProof,
}

/// Complete local proof plus the one earliest SQL candidate.  The live route turns the candidate
/// into an error immediately; the indexed proof-only seam retains it only long enough to order it
/// against a current-resident verdict under the mutation gate.
pub(super) struct PreWalConstraintPreparation {
    pub(super) proof: PreWalConstraintProof,
    pub(super) candidate: Option<ConstraintCandidate>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ConstraintPhase {
    PrimaryKeyNull = 0,
    Check = 1,
    Duplicate = 2,
}

#[cfg_attr(test, derive(Clone))]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ConstraintOrdinal {
    Attnum(i16),
    Check(String, usize),
    Index(usize),
}

#[cfg_attr(test, derive(Clone))]
enum ConstraintDiagnostic {
    NotNull { table: String, column: String },
    Check { table: String, name: String },
    Unique { name: String },
}

/// A bounded terminal from a device operation.  Candidate order is exactly `(row, phase,
/// ordinal)`: CHECK ordinal is relcache lexical `(name, raw catalog ordinal)`, PRIMARY KEY NULL
/// uses attnum, and duplicate uses the raw index vector ordinal.
#[cfg_attr(test, derive(Clone))]
pub(super) struct ConstraintCandidate {
    row: u32,
    phase: ConstraintPhase,
    ordinal: ConstraintOrdinal,
    diagnostic: ConstraintDiagnostic,
}

impl PreWalConstraintProof {
    pub(super) fn matches_current_catalog(&self, catalog: &CatalogSnapshot) -> bool {
        self.checks.matches_current_catalog(catalog) && self.keys.matches_current_catalog(catalog)
    }

    #[cfg(test)]
    pub(super) fn batch_key_proof(&self) -> &batch_key_constraints::BatchKeyConstraintProof {
        &self.keys
    }

    /// Consume the complete pre-WAL witness only after both semantic classes have been
    /// revalidated against the held current catalog.  The cfg(test) resident seam receives the
    /// one sealed raw-index proof; it must not clone or reconstruct a second authority from the
    /// catalog.
    #[cfg(test)]
    pub(super) fn into_batch_key_proof_after_current_catalog(
        self,
        catalog: &CatalogSnapshot,
    ) -> Result<batch_key_constraints::BatchKeyConstraintProof, EngineError> {
        if !self.checks.matches_current_target_binding(catalog)
            || !self.keys.matches_current_target_binding(catalog)
        {
            return Err(EngineError::ApplyFailed(
                "device pre-WAL constraint witness drifted before resident proof".to_string(),
            ));
        }
        let Self { checks, keys } = self;
        drop(checks);
        Ok(keys)
    }
}

/// Complete every batch-local device semantic operation before a pre-WAL proof is sealed.
/// CUDA/runtime errors are fail-closed and never route through host constraint evaluation.
pub(super) fn validate_before_queue(
    engine: &Engine,
    batch: &TypedInsertBatch,
    catalog: &CatalogSnapshot,
) -> Result<PreWalConstraintProof, EngineError> {
    let prepared = prepare_before_queue(engine, batch, catalog)?;
    if let Some(candidate) = prepared.candidate {
        return Err(candidate.into_error());
    }
    Ok(prepared.proof)
}

/// Run all batch-local device checks and preserve their globally ordered minimum without deciding
/// whether a current-resident unique candidate exists.  This is intentionally not used by the
/// live route: retaining an indexed local candidate is authorization solely for the cfg(test)
/// proof seam, never a production eligibility lift.
pub(super) fn prepare_before_queue(
    engine: &Engine,
    batch: &TypedInsertBatch,
    catalog: &CatalogSnapshot,
) -> Result<PreWalConstraintPreparation, EngineError> {
    let table = bound_table(batch, catalog)?;
    if !table.foreign_keys.is_empty() {
        return Err(EngineError::ApplyFailed(format!(
            "device pre-WAL proof is unavailable for relation \"{}\" with foreign keys",
            table.name
        )));
    }
    let checks = row_local_constraints::compile(batch, table)?;
    let keys = batch_key_constraints::compile(batch, table)?;
    let check_scratch = row_local_constraints::scratch_bytes(batch, table)?;
    let key_scratch = batch_key_constraints::max_scratch_bytes(batch, &keys)?;
    let needs_source = !table.check_constraints.is_empty() || keys.requires_device_source();

    let mut candidates = Vec::new();
    if needs_source {
        let source_bytes = batch.row_local_constraint_device_payload_bytes()?;
        let operator_scratch = check_scratch.max(key_scratch);
        let peak_bytes = source_bytes.checked_add(operator_scratch).ok_or_else(|| {
            EngineError::ApplyFailed("device pre-WAL allocation peak overflows".to_string())
        })?;
        let gpu_id = engine.planner.default_gpu_id();
        let budget = match engine.relational_residency_budget_bytes(gpu_id) {
            Some(limit) => limit
                .checked_sub(engine.relational_resident_bytes_for_gpu(gpu_id))
                .ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "device pre-WAL allocation refused: resident data already exceeds its budget"
                            .to_string(),
                    )
                })?,
            None => peak_bytes,
        };
        let allocation_scope = CudaAllocationScope::with_budget(budget);
        CudaAllocationScope::ensure_available(peak_bytes).map_err(|error| {
            EngineError::ApplyFailed(format!("device pre-WAL allocation refused: {error}"))
        })?;
        let source = batch
            .row_local_constraint_device_source(engine, table)
            .map_err(row_local_constraints::as_engine_error)?;
        debug_assert_eq!(source.payload_bytes(), source_bytes);

        for candidate in row_local_constraints::evaluate(engine, batch, table, &source, &checks)? {
            let constraint = &table.check_constraints[candidate.check_ordinal];
            candidates.push(ConstraintCandidate {
                row: candidate.row,
                phase: ConstraintPhase::Check,
                ordinal: ConstraintOrdinal::Check(constraint.name.clone(), candidate.check_ordinal),
                diagnostic: ConstraintDiagnostic::Check {
                    table: table.name.clone(),
                    name: constraint.name.clone(),
                },
            });
        }
        for candidate in batch_key_constraints::evaluate(batch, &source, &keys)? {
            match candidate {
                batch_key_constraints::BatchKeyCandidate::PrimaryKeyNull { row, column } => {
                    candidates.push(ConstraintCandidate {
                        row,
                        phase: ConstraintPhase::PrimaryKeyNull,
                        ordinal: ConstraintOrdinal::Attnum(column.attnum),
                        diagnostic: ConstraintDiagnostic::NotNull {
                            table: table.name.clone(),
                            column: column.name,
                        },
                    })
                }
                batch_key_constraints::BatchKeyCandidate::Duplicate { row, index_ordinal } => {
                    candidates.push(ConstraintCandidate {
                        row,
                        phase: ConstraintPhase::Duplicate,
                        ordinal: ConstraintOrdinal::Index(index_ordinal),
                        diagnostic: ConstraintDiagnostic::Unique {
                            name: keys.index(index_ordinal).name.clone(),
                        },
                    });
                }
            }
        }
        // Source leases disappear before the scope.  On every error path lexical Drop preserves
        // the same order, so the common payload cannot outlive the accounting scope.
        drop(source);
        drop(allocation_scope);
    }

    Ok(PreWalConstraintPreparation {
        proof: PreWalConstraintProof {
            checks: row_local_constraints::seal_after_success(batch, table, &checks)?,
            keys: batch_key_constraints::seal_after_success(batch, &keys),
        },
        candidate: candidates.into_iter().min_by(|left, right| {
            (left.row, left.phase, &left.ordinal).cmp(&(right.row, right.phase, &right.ordinal))
        }),
    })
}

impl ConstraintCandidate {
    #[cfg(test)]
    pub(super) fn unique(row: u32, index_ordinal: usize, name: String) -> Self {
        Self {
            row,
            phase: ConstraintPhase::Duplicate,
            ordinal: ConstraintOrdinal::Index(index_ordinal),
            diagnostic: ConstraintDiagnostic::Unique { name },
        }
    }

    #[cfg(test)]
    pub(super) fn choose(left: Option<Self>, right: Option<Self>) -> Option<Self> {
        match (left, right) {
            (Some(left), Some(right)) => Some(
                [left, right]
                    .into_iter()
                    .min_by(|left, right| {
                        (left.row, left.phase, &left.ordinal).cmp(&(
                            right.row,
                            right.phase,
                            &right.ordinal,
                        ))
                    })
                    .expect("two candidates have a minimum"),
            ),
            (left, right) => left.or(right),
        }
    }

    pub(super) fn into_error(self) -> EngineError {
        match self.diagnostic {
            ConstraintDiagnostic::NotNull { table, column } => {
                EngineError::NotNullViolation(format!(
                    "null value in column \"{column}\" of relation \"{table}\" violates not-null constraint"
                ))
            }
            ConstraintDiagnostic::Check { table, name } => EngineError::CheckViolation(format!(
                "new row for relation \"{table}\" violates check constraint \"{name}\""
            )),
            ConstraintDiagnostic::Unique { name } => EngineError::UniqueViolation(format!(
                "duplicate key value violates unique constraint \"{name}\""
            )),
        }
    }
}

pub(super) fn bound_table<'a>(
    batch: &TypedInsertBatch,
    catalog: &'a CatalogSnapshot,
) -> Result<&'a RelationalTable, EngineError> {
    let (table_oid, schema_digest, catalog_seq) = batch.row_local_constraint_target();
    let table = catalog
        .relational_catalog
        .values()
        .find(|table| table.oid == table_oid)
        .ok_or_else(|| {
            EngineError::ApplyFailed("device pre-WAL target relation is absent".to_string())
        })?;
    if catalog.commit_seq != catalog_seq
        || crate::engine_transaction_reset::table_schema_digest(table).ok() != Some(schema_digest)
    {
        return Err(EngineError::ApplyFailed(
            "device pre-WAL target binding drifted before off-lock preparation".to_string(),
        ));
    }
    Ok(table)
}

/// Test-only counterpart for the resident key proof. Ordinary DML republishes the same catalog
/// payload at a newer commit sequence, so this permits only monotonic sequence advance while
/// preserving the target OID and schema digest exactly. Production pre-WAL binding remains
/// sequence-exact through [`bound_table`].
#[cfg(test)]
pub(super) fn bound_table_current_generation<'a>(
    batch: &TypedInsertBatch,
    catalog: &'a CatalogSnapshot,
) -> Result<&'a RelationalTable, EngineError> {
    let (table_oid, schema_digest, catalog_seq) = batch.row_local_constraint_target();
    let table = catalog
        .relational_catalog
        .values()
        .find(|table| table.oid == table_oid)
        .ok_or_else(|| {
            EngineError::ApplyFailed("device pre-WAL target relation is absent".to_string())
        })?;
    if catalog.commit_seq < catalog_seq
        || crate::engine_transaction_reset::table_schema_digest(table).ok() != Some(schema_digest)
    {
        return Err(EngineError::ApplyFailed(
            "device pre-WAL target binding drifted before current-generation proof".to_string(),
        ));
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proof_only_batch(
        catalog: &CatalogSnapshot,
        command: &crate::Command,
    ) -> crate::typed_insert_batch::TypedInsertBatch {
        crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
            command,
            catalog,
            catalog.commit_seq,
        )
        .expect("proof-only typed builder succeeds")
        .expect("indexed proof-only shape remains eligible")
    }

    fn all_types_command(table: &str, rows: Vec<Vec<crate::SqlValue>>) -> crate::Command {
        crate::Command::Insert(crate::Insert {
            table: table.to_string(),
            columns: Vec::new(),
            rows: gpu_db_sql::Insert::programmatic_rows(rows),
            returning: Vec::new(),
        })
    }

    struct AllTypesRow {
        small: i16,
        integer: i32,
        day: i32,
        big: i64,
        moment: i64,
        amount: i128,
        token: u8,
        flag: bool,
        note: crate::SqlValue,
    }

    fn all_types_row(row: AllTypesRow) -> Vec<crate::SqlValue> {
        vec![
            crate::SqlValue::Int2(row.small),
            crate::SqlValue::Int4(row.integer),
            crate::SqlValue::Date(row.day),
            crate::SqlValue::Int8(row.big),
            crate::SqlValue::Timestamp(row.moment),
            crate::SqlValue::Numeric(crate::Decimal128::new(row.amount, 2)),
            crate::SqlValue::Uuid([row.token; 16]),
            crate::SqlValue::Bool(row.flag),
            row.note,
        ]
    }

    fn add_all_types_unique_index(catalog: &mut CatalogSnapshot, table_name: &str, name: &str) {
        let table = catalog.relational_catalog.get_mut(table_name).unwrap();
        table
            .indexes
            .push(crate::relational_model::RelationalIndex {
                oid: 90_001,
                name: name.to_string(),
                table: table.name.clone(),
                column: "small".to_string(),
                key_columns: vec![
                    "small".to_string(),
                    "integer".to_string(),
                    "day".to_string(),
                    "big".to_string(),
                    "moment".to_string(),
                    "amount".to_string(),
                    "token".to_string(),
                    "flag".to_string(),
                    "note".to_string(),
                ],
                unique: true,
                primary_key: false,
                unique_constraint: false,
            });
    }

    fn proof_only_error(engine: &crate::Engine, sql: &str) -> EngineError {
        let catalog = engine.catalog_snapshot();
        let command = gpu_db_sql::parse_command(sql).expect("test INSERT parses");
        let batch = proof_only_batch(&catalog, &command);
        match validate_before_queue(engine, &batch, &catalog) {
            Ok(_) => panic!("the local device proof must reject this prepared batch"),
            Err(error) => error,
        }
    }

    #[test]
    fn proof_only_indexed_seam_does_not_widen_the_live_typed_route() {
        let engine = crate::Engine::new_local_test_engine();
        engine
            .execute_text(
                1,
                "CREATE TABLE proof_only_gate (id int4 PRIMARY KEY, code int4 UNIQUE)",
            )
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let command =
            gpu_db_sql::parse_command("INSERT INTO proof_only_gate VALUES (1, 7)").unwrap();
        assert!(crate::typed_insert_batch::try_prepare_typed_insert_batch(
            &command,
            &catalog,
            catalog.commit_seq,
            None,
        )
        .unwrap()
        .is_none());
        assert!(
            crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
                &command,
                &catalog,
                catalog.commit_seq,
            )
            .unwrap()
            .is_some()
        );

        let mut foreign_key_catalog = (*catalog).clone();
        foreign_key_catalog
            .relational_catalog
            .get_mut("proof_only_gate")
            .unwrap()
            .foreign_keys
            .push(crate::relational_model::RelationalForeignKey {
                name: "proof_only_gate_fk".to_string(),
                column: "code".to_string(),
                referenced_table: "parent".to_string(),
                referenced_column: "id".to_string(),
            });
        assert!(
            crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
                &command,
                &foreign_key_catalog,
                foreign_key_catalog.commit_seq,
            )
            .unwrap()
            .is_none()
        );

        let mut column_owner_drift = (*catalog).clone();
        let table = column_owner_drift
            .relational_catalog
            .get_mut("proof_only_gate")
            .unwrap();
        table.columns[0].table_oid = table.columns[0].table_oid.wrapping_add(1);
        assert!(
            crate::typed_insert_batch::try_prepare_typed_insert_batch_proof_only(
                &command,
                &column_owner_drift,
                column_owner_drift.commit_seq,
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn composite_owner_keeps_one_source_scope_order_and_live_index_gate() {
        let source = include_str!("pre_wal_constraints.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production composite owner precedes tests");
        assert_eq!(
            source
                .matches("row_local_constraint_device_source(")
                .count(),
            1
        );
        let ensure = source
            .find("CudaAllocationScope::ensure_available(peak_bytes)")
            .expect("composite owner admits the complete peak before allocation");
        let build = source
            .find(".row_local_constraint_device_source(")
            .expect("composite owner builds the one shared source");
        assert!(ensure < build);
        assert!(source.contains("drop(source);\n        drop(allocation_scope);"));

        let live_gate = include_str!("row_local_constraints.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("typed builder is production source");
        assert!(live_gate.contains("table.indexes.is_empty()"));
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_pre_wal_exact_budget_uses_one_source_for_check_and_wide_unique() {
        let mut engine = crate::Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine
            .execute_text(
                1,
                "CREATE TABLE proof_exact_budget (small int2, integer int4, day date, big int8, \
                 moment timestamp, amount numeric(10,2), token uuid, flag bool, note text, \
                 CONSTRAINT proof_exact_budget_check CHECK (integer > 0))",
            )
            .unwrap();
        let mut catalog = (*engine.catalog_snapshot()).clone();
        add_all_types_unique_index(
            &mut catalog,
            "proof_exact_budget",
            "proof_exact_budget_all_types_unique",
        );
        let command = all_types_command(
            "proof_exact_budget",
            vec![
                all_types_row(AllTypesRow {
                    small: 1,
                    integer: 1,
                    day: 1,
                    big: 1,
                    moment: 1,
                    amount: 1_234,
                    token: 1,
                    flag: true,
                    note: crate::SqlValue::Null,
                }),
                all_types_row(AllTypesRow {
                    small: 2,
                    integer: 2,
                    day: 2,
                    big: 2,
                    moment: 2,
                    amount: 2_345,
                    token: 2,
                    flag: false,
                    note: crate::SqlValue::Text("present".to_string()),
                }),
            ],
        );
        let batch = proof_only_batch(&catalog, &command);
        let source_bytes = batch.row_local_constraint_device_payload_bytes().unwrap();
        let rows = usize::try_from(batch.binary_insert_template_row_count()).unwrap();
        let key_scratch = gpu_db_execution::insert_batch_key_verdict_scratch_bytes(rows, 10)
            .expect("nine data descriptors plus one actual validity descriptor fit");
        let table = catalog
            .relational_catalog
            .get("proof_exact_budget")
            .unwrap();
        assert!(
            super::super::row_local_constraints::scratch_bytes(&batch, table).unwrap()
                <= key_scratch,
            "the wide exact-key primitive is the composite scratch high-water"
        );
        let peak = source_bytes.checked_add(key_scratch).unwrap();
        let gpu_id = engine.planner.default_gpu_id();
        let resident = engine.relational_resident_bytes_for_gpu(gpu_id);
        engine.set_relational_residency_budget_bytes(gpu_id, resident + peak - 1);
        crate::typed_insert_batch::reset_constraint_source_upload_count();
        let wal_before = engine.durable_wal_records().len();
        let row_id_before = engine.read_state.mvcc.current_row_id();
        let rejected =
            super::super::PreparedDeviceInsertPlan::from_typed_batch(batch, &engine, &catalog);
        assert!(matches!(
            rejected,
            Err(EngineError::ApplyFailed(message))
                if message.contains("device pre-WAL allocation refused")
        ));
        assert_eq!(
            crate::typed_insert_batch::constraint_source_upload_count(),
            0
        );
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);

        engine.set_relational_residency_budget_bytes(gpu_id, resident + peak);
        let fresh_batch = proof_only_batch(&catalog, &command);
        crate::typed_insert_batch::reset_constraint_source_upload_count();
        let plan = match super::super::PreparedDeviceInsertPlan::from_typed_batch(
            fresh_batch,
            &engine,
            &catalog,
        ) {
            Ok(plan) => plan,
            Err(error) => panic!("exact pre-WAL resource geometry unexpectedly failed: {error}"),
        };
        assert_eq!(
            crate::typed_insert_batch::constraint_source_upload_count(),
            1
        );
        assert_eq!(engine.durable_wal_records().len(), wal_before);
        assert_eq!(engine.read_state.mvcc.current_row_id(), row_id_before);
        drop(plan);
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_composite_key_cases_cover_all_types_nulls_and_repeated_suffixes() {
        let engine = crate::Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine
            .execute_text(
                1,
                "CREATE TABLE proof_all_types (small int2, integer int4, day date, big int8, \
                 moment timestamp, amount numeric(10,2), token uuid, flag bool, note text)",
            )
            .unwrap();
        let mut all_types_catalog = (*engine.catalog_snapshot()).clone();
        add_all_types_unique_index(
            &mut all_types_catalog,
            "proof_all_types",
            "proof_all_types_unique",
        );
        let duplicate = all_types_command(
            "proof_all_types",
            vec![
                all_types_row(AllTypesRow {
                    small: 1,
                    integer: 1,
                    day: 1,
                    big: 1,
                    moment: 1,
                    amount: 1_234,
                    token: 1,
                    flag: true,
                    note: crate::SqlValue::Text("same".to_string()),
                }),
                all_types_row(AllTypesRow {
                    small: 1,
                    integer: 1,
                    day: 1,
                    big: 1,
                    moment: 1,
                    amount: 1_234,
                    token: 1,
                    flag: true,
                    note: crate::SqlValue::Text("same".to_string()),
                }),
            ],
        );
        let error = match validate_before_queue(
            &engine,
            &proof_only_batch(&all_types_catalog, &duplicate),
            &all_types_catalog,
        ) {
            Ok(_) => panic!("exact all-types local duplicate must fail"),
            Err(error) => error,
        };
        assert!(
            matches!(error, EngineError::UniqueViolation(message) if message.contains("proof_all_types_unique"))
        );

        engine
            .execute_text(2, "CREATE TABLE proof_null_unique (code int4 UNIQUE)")
            .unwrap();
        let nulls =
            gpu_db_sql::parse_command("INSERT INTO proof_null_unique VALUES (NULL), (NULL)")
                .unwrap();
        let null_catalog = engine.catalog_snapshot();
        let null_plan = match super::super::PreparedDeviceInsertPlan::from_typed_batch(
            proof_only_batch(&null_catalog, &nulls),
            &engine,
            &null_catalog,
        ) {
            Ok(plan) => plan,
            Err(error) => panic!("ordinary UNIQUE NULLS DISTINCT unexpectedly failed: {error}"),
        };
        drop(null_plan);
        let error = proof_only_error(
            &engine,
            "INSERT INTO proof_null_unique VALUES (NULL), (NULL), (7), (7)",
        );
        assert!(
            matches!(error, EngineError::UniqueViolation(message) if message.contains("proof_null_unique_code_key"))
        );

        engine
            .execute_text(3, "CREATE TABLE proof_repeated_suffix (a int4)")
            .unwrap();
        let mut repeated_catalog = (*engine.catalog_snapshot()).clone();
        let table = repeated_catalog
            .relational_catalog
            .get_mut("proof_repeated_suffix")
            .unwrap();
        table
            .indexes
            .push(crate::relational_model::RelationalIndex {
                oid: 90_002,
                name: "proof_repeated_suffix_unique".to_string(),
                table: table.name.clone(),
                column: "a".to_string(),
                key_columns: vec!["a".to_string(), "a".to_string()],
                unique: true,
                primary_key: false,
                unique_constraint: false,
            });
        let repeated = all_types_command(
            "proof_repeated_suffix",
            vec![
                vec![crate::SqlValue::Null],
                vec![crate::SqlValue::Int4(7)],
                vec![crate::SqlValue::Int4(7)],
            ],
        );
        let error = match validate_before_queue(
            &engine,
            &proof_only_batch(&repeated_catalog, &repeated),
            &repeated_catalog,
        ) {
            Ok(_) => panic!("repeated descriptor local duplicate must fail"),
            Err(error) => error,
        };
        assert!(
            matches!(error, EngineError::UniqueViolation(message) if message.contains("proof_repeated_suffix_unique"))
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn actual_gpu_composite_proof_preserves_postgres_batch_precedence() {
        let engine = crate::Engine::new_local();
        let hardware = engine.cuda_driver_probe_runtime().snapshot();
        if !hardware.driver_available || hardware.device_count == 0 {
            return;
        }
        engine
            .execute_text(
                1,
                "CREATE TABLE proof_rank (id int4 PRIMARY KEY, u int4 UNIQUE, v int4, \
                 CONSTRAINT z_check CHECK (v > 0), CONSTRAINT a_check CHECK (v < 10))",
            )
            .unwrap();

        let error = proof_only_error(&engine, "INSERT INTO proof_rank VALUES (NULL, 1, 20)");
        assert!(matches!(error, EngineError::NotNullViolation(message)
            if message.contains("column \"id\"") && message.contains("proof_rank")));

        let error = proof_only_error(&engine, "INSERT INTO proof_rank VALUES (2, 1, 20)");
        assert!(
            matches!(error, EngineError::CheckViolation(message) if message.contains("a_check"))
        );

        let error = proof_only_error(
            &engine,
            "INSERT INTO proof_rank VALUES (1, 7, 1), (2, 7, 1), (NULL, 8, 1)",
        );
        assert!(
            matches!(error, EngineError::UniqueViolation(message) if message.contains("proof_rank_u_key"))
        );

        let error = proof_only_error(
            &engine,
            "INSERT INTO proof_rank VALUES (1, 1, 20), (2, 9, 1), (3, 9, 1)",
        );
        assert!(
            matches!(error, EngineError::CheckViolation(message) if message.contains("a_check"))
        );

        engine
            .execute_text(
                2,
                "CREATE TABLE proof_pk_order (a int4, b int4, PRIMARY KEY (b, a))",
            )
            .unwrap();
        let error = proof_only_error(&engine, "INSERT INTO proof_pk_order VALUES (NULL, NULL)");
        assert!(
            matches!(error, EngineError::NotNullViolation(message) if message.contains("column \"a\""))
        );

        engine
            .execute_text(
                3,
                "CREATE TABLE proof_check_order (a int4, v int4, CONSTRAINT z_rule CHECK (v > 0))",
            )
            .unwrap();
        engine
            .execute_text(
                4,
                "ALTER TABLE proof_check_order ADD CONSTRAINT a_rule CHECK (v < 10)",
            )
            .unwrap();
        let error = proof_only_error(&engine, "INSERT INTO proof_check_order VALUES (1, 20)");
        assert!(
            matches!(error, EngineError::CheckViolation(message) if message.contains("a_rule"))
        );

        engine
            .execute_text(
                5,
                "CREATE TABLE proof_index_order (a int4, CONSTRAINT z_unique UNIQUE (a))",
            )
            .unwrap();
        engine
            .execute_text(
                6,
                "CREATE UNIQUE INDEX a_standalone ON proof_index_order (a)",
            )
            .unwrap();
        let error = proof_only_error(&engine, "INSERT INTO proof_index_order VALUES (1), (1)");
        assert!(
            matches!(error, EngineError::UniqueViolation(message) if message.contains("z_unique"))
        );

        engine
            .execute_text(
                7,
                "CREATE TABLE proof_groups (a int4 UNIQUE, b int4 UNIQUE)",
            )
            .unwrap();
        let error = proof_only_error(
            &engine,
            "INSERT INTO proof_groups VALUES (1, 10), (2, 20), (1, 30), (3, 20)",
        );
        assert!(
            matches!(error, EngineError::UniqueViolation(message) if message.contains("proof_groups_a_key"))
        );
    }
}
