//! General GPU executor — the `Expr` IR and its device interpreter (Charter rule 2;
//! `docs/architecture/17-general-gpu-executor.md`). This is the GENERAL path that replaces the
//! enumerated `execute_relational_*_with_resident_device_memory_probe` shape methods: a query is an
//! expression tree, lowered to a pipeline of device primitives, NOT matched against a fixed catalog
//! of shapes.
//!
//! `ResidentExpr` is the general scalar IR (it can represent any int4 arithmetic/comparison/boolean
//! tree). The interpreter's *coverage* grows node-by-node; today `lower_resident_predicate` lowers
//! the prototype shape `Compare(arith(Column, Column), Int4Literal)` to the composed buffer->buffer
//! primitive (`expr_filter_two_col_compare_from_payload`) and materializes the projected int4
//! columns by gathering the surviving rows. Fuller trees (deeper arithmetic, AND/OR over masks,
//! column-vs-column compares) land as the device bytecode VM in §2.3 of the design doc — by
//! extending this interpreter, never by adding a new shape method.
//!
//! Forward-API module: the IR, the op-code maps, and `execute_resident_expr_select` are exercised by
//! the GPU parity tests today and wired to the parser/planner next; until then they are unused in a
//! non-test build, so the whole module allows dead code. Drop this allow once a production caller
//! (the SQL->Expr binding) lands.
#![allow(dead_code)]

use super::*;

/// A binary operator in the resident expression IR (arithmetic, comparison, or boolean). The
/// interpreter pattern-matches these; callers (tests now, the parser/planner later) construct them.
/// Not all variants have interpreter coverage yet (the device bytecode VM, design §2.3, adds
/// AND/OR/Ne/col-vs-col).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentBinaryOp {
    Add,
    Sub,
    Mul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// The general scalar-expression IR the GPU interpreter evaluates. Column fields are table column
/// indices; the interpreter resolves device byte-offsets. Grows by node (literals of other types,
/// casts, unary ops, `LIKE`, ...), never by enumerating whole query shapes.
#[derive(Debug, Clone)]
pub(crate) enum ResidentExpr {
    Column(usize),
    Int4Literal(i32),
    Binary {
        op: ResidentBinaryOp,
        lhs: Box<ResidentExpr>,
        rhs: Box<ResidentExpr>,
    },
}

/// Device op-code for an arithmetic binary op (matches `expr_proto.ptx`: 0=add, 1=sub, 2=mul), or
/// `None` if `op` is not arithmetic.
fn arith_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Add => Some(0),
        ResidentBinaryOp::Sub => Some(1),
        ResidentBinaryOp::Mul => Some(2),
        _ => None,
    }
}

/// Device comparison code for a comparison binary op (matches `expr_proto.ptx`: 0=eq, 1=lt, 2=le,
/// 3=gt, 4=ge), or `None` if `op` is not a kernel-supported comparison (`Ne` has no primitive yet).
fn compare_op_code(op: ResidentBinaryOp) -> Option<u32> {
    match op {
        ResidentBinaryOp::Eq => Some(0),
        ResidentBinaryOp::Lt => Some(1),
        ResidentBinaryOp::Le => Some(2),
        ResidentBinaryOp::Gt => Some(3),
        ResidentBinaryOp::Ge => Some(4),
        _ => None,
    }
}

impl Engine {
    /// General GPU executor entry (Charter rule 2): run `SELECT <int4 columns> FROM <table>` filtered
    /// by a general predicate [`ResidentExpr`], evaluating the predicate on the GPU via the device
    /// interpreter and materializing the surviving rows by gathering the projected columns. The
    /// predicate is supplied as an expression tree (the unit of execution is an expression, not a
    /// recognized shape); the `Select` carries the table + projection + MVCC binding. NOT routed
    /// through `resident_route_query_shape` — this is the general path, parallel to the (frozen)
    /// enumerated probe dispatch.
    pub(crate) fn execute_resident_expr_select(
        &self,
        select: &Select,
        predicate: &ResidentExpr,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if bound.selected_indexes.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident Expr select requires at least one projected column".to_string(),
            )));
        }
        for &col in &bound.selected_indexes {
            if table.columns[col].ty != SqlType::Int4 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident Expr select currently materializes only int4 projection columns"
                        .to_string(),
                )));
            }
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot_ref(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
        if snapshot.schema != table.schema || snapshot.table != table.name {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot no longer matches catalog table identity".to_string(),
            )));
        }
        if !snapshot.is_valid() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" resident snapshot is invalid",
                table.name
            ))));
        }
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range".to_string(),
            ))
        })?;

        // Evaluate the predicate on the GPU -> surviving row indices (ascending).
        let indices = self.lower_resident_predicate(
            predicate,
            &table,
            &snapshot,
            &device_memory,
            row_count,
        )?;
        let indices_u64: Vec<u64> = indices.iter().map(|&i| u64::from(i)).collect();

        // Materialize: gather each projected int4 column at the surviving row indices on the GPU.
        let mut projected_columns: Vec<Vec<i32>> = Vec::with_capacity(bound.selected_indexes.len());
        for &col in &bound.selected_indexes {
            let byte_offset = resident_device_int4_column_offset(&snapshot, &table, col)?;
            let values = device_memory
                .project_i32_rows_from_payload(byte_offset, &indices_u64)
                .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
            projected_columns.push(values);
        }
        let rows: Vec<Vec<SqlValue>> = (0..indices_u64.len())
            .map(|row| {
                projected_columns
                    .iter()
                    .map(|column| SqlValue::Int4(column[row]))
                    .collect()
            })
            .collect();

        Ok(RelationalSelectResult {
            columns: bound.selected_columns,
            rows,
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path,
        })
    }

    /// Lower a predicate [`ResidentExpr`] to the surviving row indices, evaluated on the GPU.
    ///
    /// Coverage (grows by extending this method, per the design): the prototype shape
    /// `Compare(arith(Column a, Column b), Int4Literal k)` lowers to the composed buffer->buffer
    /// primitive `expr_filter_two_col_compare_from_payload` (elementwise `a <arith> b` into an
    /// intermediate device buffer, then compare-to-row-indices). Anything else is rejected with a
    /// pointer to the device bytecode VM that generalizes it — NOT by falling back to a shape method
    /// or to the CPU.
    fn lower_resident_predicate(
        &self,
        predicate: &ResidentExpr,
        table: &RelationalTable,
        snapshot: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        row_count: u64,
    ) -> Result<Vec<u32>, ExecuteError> {
        if let ResidentExpr::Binary {
            op: compare,
            lhs,
            rhs,
        } = predicate
        {
            if let (Some(comparison), ResidentExpr::Int4Literal(needle)) =
                (compare_op_code(*compare), rhs.as_ref())
            {
                if let ResidentExpr::Binary {
                    op: arith,
                    lhs: a,
                    rhs: b,
                } = lhs.as_ref()
                {
                    if let (
                        Some(op_code),
                        ResidentExpr::Column(col_a),
                        ResidentExpr::Column(col_b),
                    ) = (arith_op_code(*arith), a.as_ref(), b.as_ref())
                    {
                        let a_offset = resident_device_int4_column_offset(snapshot, table, *col_a)?;
                        let b_offset = resident_device_int4_column_offset(snapshot, table, *col_b)?;
                        return device_memory
                            .expr_filter_two_col_compare_from_payload(
                                a_offset, b_offset, op_code, row_count, *needle, comparison,
                            )
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            });
                    }
                }
            }
        }
        Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident Expr interpreter currently lowers Compare(arith(col, col), int4_literal); \
             fuller trees (deeper arithmetic, AND/OR, col-vs-col) land via the device bytecode VM \
             (docs/architecture/17-general-gpu-executor.md §2.3)"
                .to_string(),
        )))
    }
}
