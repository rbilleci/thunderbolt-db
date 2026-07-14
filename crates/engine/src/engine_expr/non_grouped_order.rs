//! Non-grouped GPU ORDER BY key materialization, sorting, and post-sort windowing.

use super::predicate_compiler::{
    compile_arith_program, predicate_references_nullable_column, push_leaf_validity_and,
};
use super::predicate_operands::expr_mentions_int8;
use crate::engine_expr_ir::ResidentExpr;
use crate::rel_exec_helpers::relational_column_index;
use crate::relational_model::{
    resident_device_int4_column_offset, resident_device_int8_column_offset,
    resident_device_null_column_offset, resident_device_numeric_column_offset,
    resident_device_text_column_layout, RelationalResidencySnapshot, RelationalTable,
};
use crate::ExecuteError;
use gpu_db_execution::{CudaResidentDeviceMemory, ExprStep, ResidentElemType};
use gpu_db_sql::{Select, SqlType};
use gpu_db_types::EngineError;

#[allow(clippy::too_many_arguments)]
pub(super) fn order_and_window_indices(
    select: &Select,
    table: &RelationalTable,
    order_by_exprs: &[Option<ResidentExpr>],
    order_by_nulls_first: &[Option<bool>],
    snapshot: &RelationalResidencySnapshot,
    device_memory: &CudaResidentDeviceMemory,
    row_count: u64,
    indices_u64: Vec<u64>,
) -> Result<Vec<u64>, ExecuteError> {
    // Non-grouped ORDER BY: reorder the surviving indices by the order key(s) on the GPU (bitonic
    // sort) BEFORE gathering, so the projected rows come out sorted -- a charter-native GPU sort,
    // not a host/CPU sort. The routing only sends sorts whose keys are ALL i64-sortable int columns
    // (int2/int4/int8/date/timestamp) to this path -- one key OR several (`ORDER BY a ASC, b DESC`).
    // A single TEXT-key ORDER BY takes the varlen GPU sort path (the byte-wise comparator); all
    // other routed keys are i64-sortable ints and go through the key-matrix path below.
    // Classify the ORDER BY keys: k==1 text -> the varlen text sort; k>1 with ANY text key -> the
    // heterogeneous mixed (int+text) sort; all-int -> the i64 key matrix.
    // Materialize an INT-bearing ORDER BY key's i64 values for `indices`. A sort EXPRESSION
    // (`ORDER BY a+b`) is evaluated on-device into an i64 column (checked int4 overflow -> PG error,
    // never CPU); a plain int column is projected. Both sort arms below use this; an expression key
    // always counts as an int key. `indices` is a param (the else arm below moves `indices_u64`).
    let materialize_int_key_column =
        |ki: usize, indices: &[u64]| -> Result<Vec<i64>, ExecuteError> {
            let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
            };
            if let Some(Some(expr)) = order_by_exprs.get(ki) {
                let mut program = Vec::new();
                compile_arith_program(expr, table, snapshot, &mut program)?;
                let idx32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
                // int8 operands -> the i64 arith VM + i64 value width; else int4/I32. Reading an int8
                // expr as I32 would stride a BIGINT column by 4 bytes -> silently garbage sort keys.
                let elem = if expr_mentions_int8(expr, table) {
                    ResidentElemType::I64
                } else {
                    ResidentElemType::I32
                };
                // M3 (doc 21): a NULLABLE int4 expression blends the i64::MAX default-end sentinel
                // ON-DEVICE where any operand is NULL (a validity mask VM run + the on-device blend). int8
                // nullable + explicit NULLS FIRST/LAST are clean-errored at the routing loop below; a
                // non-nullable expression keeps the plain (no-validity) path.
                if elem == ResidentElemType::I32
                    && predicate_references_nullable_column(expr, table, snapshot)?
                {
                    let mut validity_program = vec![ExprStep::ConstMask { value: true }];
                    push_leaf_validity_and(&[expr], table, snapshot, &mut validity_program)?;
                    return device_memory
                        .arith_value_column_at_indices_nullable(
                            &program,
                            &validity_program,
                            row_count,
                            &idx32,
                        )
                        .map_err(map_err);
                }
                return device_memory
                    .arith_value_column_at_indices(&program, row_count, &idx32, elem)
                    .map_err(map_err);
            }
            let order = &select.order_by[ki];
            let order_idx = relational_column_index(table, &order.column)?;
            let keys: Vec<i64> = match table.columns[order_idx].ty {
                SqlType::Int4 | SqlType::Int2 | SqlType::Date => device_memory
                    .project_i32_rows_from_payload(
                        resident_device_int4_column_offset(snapshot, table, order_idx)?,
                        indices,
                    )
                    .map_err(map_err)?
                    .into_iter()
                    .map(i64::from)
                    .collect(),
                SqlType::Int8 | SqlType::Timestamp => device_memory
                    .project_i64_rows_from_payload(
                        resident_device_int8_column_offset(snapshot, table, order_idx)?,
                        indices,
                    )
                    .map_err(map_err)?,
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "ORDER BY key is not an i64-sortable int column".to_string(),
                    )))
                }
            };
            // M3 (doc 21): NULL placement is done ON-DEVICE by the sort comparator (it reads the per-key
            // validity bitmap and orders NULL keys to the PG-default end), so the key VALUES here are raw —
            // a NULL row's placeholder value is never compared. (A nullable key always routes to the
            // validity-aware hetero comparator below; this raw matrix only feeds that path or the non-null
            // pure-int path.) No host-side NULL overwrite.
            Ok(keys)
        };
    // A sort EXPRESSION is int-valued. has_text_key (TEXT only) gates the single-text fast path;
    // has_hetero_key (TEXT/NUMERIC/UUID -- the keys that can't live in the i64 matrix) routes to the
    // heterogeneous comparator.
    let mut has_text_key = false;
    let mut has_hetero_key = false;
    // M3 (doc 21): per-key NULL validity bitmap byte offset (sentinel u64::MAX = the key holds no NULL).
    // The hetero sort comparator reads this ON-DEVICE and orders NULL keys to the PG-default end (last
    // ASC / first DESC) — GPU-native, no host shard / sentinel. A nullable key of ANY type routes
    // to that comparator (below). A nullable sort EXPRESSION stays a clean-error follow-up (an
    // expression has no single column validity bitmap; a derived one is a follow-up).
    let mut key_null_offs: Vec<u64> = vec![u64::MAX; select.order_by.len()];
    for (ki, order) in select.order_by.iter().enumerate() {
        if let Some(Some(expr)) = order_by_exprs.get(ki) {
            // A NULLABLE int4 expression is supported via the on-device value-sentinel
            // (materialize_int_key_column): a NULL result becomes the i64::MAX default-end sentinel,
            // so key_null_offs stays u64::MAX (no validity bitmap -- the NULL is value-encoded). Two
            // cases still clean-error: an int8 expression (i64::MAX could collide with a real result)
            // and an explicit NULLS FIRST/LAST (the value-sentinel only realizes the PG default).
            if predicate_references_nullable_column(expr, table, snapshot)? {
                if expr_mentions_int8(expr, table) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "ORDER BY a nullable int8 expression is a follow-up (the i64::MAX NULL \
                     sentinel could collide with a real bigint result)"
                            .to_string(),
                    )));
                }
                if order_by_nulls_first.get(ki).copied().flatten().is_some() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "explicit NULLS FIRST/LAST on a nullable ORDER BY expression is a follow-up \
                     (the default NULL placement is supported)"
                        .to_string(),
                )));
                }
            }
            continue;
        }
        let order_idx = relational_column_index(table, &order.column)?;
        match table.columns[order_idx].ty {
            SqlType::Text => {
                has_text_key = true;
                has_hetero_key = true;
            }
            SqlType::Numeric { .. } | SqlType::Uuid => has_hetero_key = true,
            _ => {}
        }
        if let Some(off) = resident_device_null_column_offset(snapshot, table, order_idx)? {
            key_null_offs[ki] = off;
        }
    }
    // A nullable key (any type) must route to the validity-aware hetero comparator. A non-null single
    // text key keeps the dedicated text fast path; non-null int-only keys keep the int matrix/radix path.
    let any_nullable_key = key_null_offs.iter().any(|&o| o != u64::MAX);
    let single_text_key = select.order_by.len() == 1 && has_text_key && !any_nullable_key;
    let use_hetero = has_hetero_key || any_nullable_key;
    let indices_u64 = if select.order_by.is_empty() {
        indices_u64
    } else if single_text_key {
        // The varlen text key can't live in the i64 key matrix, so the kernel sorts the surviving
        // rows by reading each row's bytes from the resident text column via the indices indirection
        // (lexicographic, unsigned bytes, a prefix sorts smaller). Charter-native GPU sort.
        let order = &select.order_by[0];
        let order_idx = relational_column_index(table, &order.column)?;
        let layout = resident_device_text_column_layout(snapshot, table, order_idx)?;
        let perm = device_memory
            .bitonic_sort_text(
                &indices_u64,
                layout.offsets_byte_offset,
                layout.bytes_byte_offset,
                order.descending,
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        perm.iter().map(|&p| indices_u64[p as usize]).collect()
    } else if use_hetero {
        // A key tuple with a TEXT/NUMERIC/UUID key (`ORDER BY name /*text*/, age /*int*/`, or a
        // single numeric/uuid key): the heterogeneous GPU comparator dispatches each key to the s64
        // compare (int, from a row-major by-position matrix), the byte compare (text, in place), or
        // the 16-byte compare (numeric = signed-hi/unsigned-lo i128; uuid = big-endian unsigned),
        // reading numeric/uuid in place from the resident column. Charter-native GPU sort.
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let n = indices_u64.len();
        let k = select.order_by.len();
        if k > 64 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "ORDER BY supports at most 64 sort keys on the GPU sort path".to_string(),
            )));
        }
        // Walk the keys in order: each int key takes the next int-matrix column, each text key the
        // next text-column slot; build key_plan (bit31=is_text, low bits=slot) + desc_mask.
        let mut num_int = 0usize;
        let mut text_cols: Vec<(u64, u64)> = Vec::new();
        let mut b128_cols: Vec<u64> = Vec::new();
        let mut key_plan: Vec<u32> = Vec::with_capacity(k);
        let mut desc_mask: u64 = 0;
        // Per-key effective NULLS FIRST bit: the explicit override, else the key's DESC (PG default).
        // The comparator reads it to place NULLs ON-DEVICE, decoupled from the value-compare direction.
        let mut nulls_first_mask: u64 = 0;
        let mut int_key_slots: Vec<(usize, usize)> = Vec::new();
        for (ki, order) in select.order_by.iter().enumerate() {
            if order.descending {
                desc_mask |= 1u64 << ki;
            }
            if order_by_nulls_first
                .get(ki)
                .copied()
                .flatten()
                .unwrap_or(order.descending)
            {
                nulls_first_mask |= 1u64 << ki;
            }
            // A sort EXPRESSION is an int-valued key -> the next int slot (materialized below).
            if order_by_exprs.get(ki).is_some_and(|e| e.is_some()) {
                let int_slot = num_int;
                num_int += 1;
                key_plan.push(int_slot as u32);
                int_key_slots.push((ki, int_slot));
                continue;
            }
            let order_idx = relational_column_index(table, &order.column)?;
            match table.columns[order_idx].ty {
                SqlType::Text => {
                    let layout = resident_device_text_column_layout(snapshot, table, order_idx)?;
                    let text_slot = text_cols.len() as u32;
                    text_cols.push((layout.offsets_byte_offset, layout.bytes_byte_offset));
                    // kind 1 (key_plan bits 30-31 = 01) = text.
                    key_plan.push(0x4000_0000_u32 | text_slot);
                }
                SqlType::Numeric { .. } => {
                    let b128_slot = b128_cols.len() as u32;
                    b128_cols.push(resident_device_numeric_column_offset(
                        snapshot, table, order_idx,
                    )?);
                    // kind 2 (bits 30-31 = 10) = numeric (signed-hi/unsigned-lo i128).
                    key_plan.push(0x8000_0000_u32 | b128_slot);
                }
                SqlType::Uuid => {
                    let b128_slot = b128_cols.len() as u32;
                    b128_cols.push(resident_device_numeric_column_offset(
                        snapshot, table, order_idx,
                    )?);
                    // kind 3 (bits 30-31 = 11) = uuid (big-endian unsigned).
                    key_plan.push(0xC000_0000_u32 | b128_slot);
                }
                SqlType::Int4
                | SqlType::Int2
                | SqlType::Date
                | SqlType::Int8
                | SqlType::Timestamp => {
                    let int_slot = num_int;
                    num_int += 1;
                    key_plan.push(int_slot as u32);
                    int_key_slots.push((ki, int_slot));
                }
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "mixed-key ORDER BY on the Expr path supports int2 / int4 / int8 / date / \
                     timestamp / text / numeric / uuid columns (the GPU sort); other types \
                     are a follow-on"
                            .to_string(),
                    )));
                }
            }
        }
        // Materialize the int keys into a row-major n*num_int matrix by position (matching the
        // multikey kernel's layout; the text keys read in place via the indices indirection).
        let mut int_keys = vec![0i64; n * num_int];
        for (ki, int_slot) in int_key_slots {
            let col_keys = materialize_int_key_column(ki, &indices_u64)?;
            for (i, v) in col_keys.into_iter().enumerate() {
                int_keys[i * num_int + int_slot] = v;
            }
        }
        let perm = device_memory
            .bitonic_sort_hetero(
                &indices_u64,
                &int_keys,
                num_int,
                &text_cols,
                &b128_cols,
                &key_plan,
                desc_mask,
                &key_null_offs,
                nulls_first_mask,
            )
            .map_err(map_err)?;
        perm.iter().map(|&p| indices_u64[p as usize]).collect()
    } else {
        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let n = indices_u64.len();
        let k = select.order_by.len();
        if k > 64 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "ORDER BY supports at most 64 sort keys on the GPU sort path".to_string(),
            )));
        }
        // Materialize every ORDER BY key column's i64 value for the surviving rows into a row-major
        // key matrix (`keys[row * K + key]`, key 0 most significant) and the per-key direction mask
        // (bit k set => key k is DESC). int2/int4/date sign-extend through i32; int8/timestamp are
        // the full i64. The GPU multi-key comparator breaks ties on key 0 by key 1, then key 2, ...
        let mut key_matrix = vec![0i64; n * k];
        let mut desc_mask: u64 = 0;
        for (kk, order) in select.order_by.iter().enumerate() {
            if order.descending {
                desc_mask |= 1u64 << kk;
            }
            // Each key is a plain int column OR a sort expression (`a+b`); the helper materializes
            // the i64 value column either way (expression -> on-device eval with checked overflow).
            let col_keys = materialize_int_key_column(kk, &indices_u64)?;
            for (i, v) in col_keys.into_iter().enumerate() {
                key_matrix[i * k + kk] = v;
            }
        }
        // Single-key dispatches by size -- bitonic for small n, radix (O(n)) for large n -- via
        // order_by_sort_i64; multi-key uses the row-major bitonic comparator.
        let perm = if k == 1 {
            device_memory
                .order_by_sort_i64(&key_matrix, (desc_mask & 1) != 0)
                .map_err(map_err)?
        } else {
            device_memory
                .bitonic_sort_multikey(&key_matrix, n, k, desc_mask)
                .map_err(map_err)?
        };
        perm.iter().map(|&p| indices_u64[p as usize]).collect()
    };
    // LIMIT/OFFSET as control-plane WINDOWING of the device-ordered index vector: slice the surviving
    // indices to the [OFFSET, OFFSET+LIMIT) window BEFORE the column gather, so only the kept rows are
    // materialized from the device (we never gather rows that would then be dropped -- the real win).
    // SQL clause order: ORDER BY (the GPU sort above) -> OFFSET -> LIMIT. Slicing an index vector is
    // control-plane; the relational ordering+windowing decision rode the device sort.
    let indices_u64 = if select.offset.is_some() || select.limit.is_some() {
        let start = select.offset.unwrap_or(0).min(indices_u64.len());
        let end = select.limit.map_or(indices_u64.len(), |l| {
            start.saturating_add(l).min(indices_u64.len())
        });
        indices_u64[start..end].to_vec()
    } else {
        indices_u64
    };
    Ok(indices_u64)
}
