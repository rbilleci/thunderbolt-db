//! Grouped-value reconstruction and representative-row grouping helpers.

use crate::relational_model::{
    resident_device_bool_column_offset, resident_device_int4_column_offset,
    resident_device_int8_column_offset, resident_device_numeric_column_offset,
    resident_device_text_column_layout, RelationalResidencySnapshot, RelationalTable,
};
use crate::ExecuteError;
use gpu_db_sql::{Decimal128, SqlType, SqlValue};
use gpu_db_types::EngineError;

/// Narrow a grouped MIN/MAX or GROUP BY key, which the GPU kernel computes as an i64 (or, for
/// numeric, an i128 split into `lo`/`hi`), back to the column's own `SqlType`. int2/int4/date ride the
/// 4-byte read; int8/timestamp the 8-byte read; numeric reconstructs `hi:lo` at `scale`. `hi`/`scale`
/// are ignored for the non-numeric types.
pub(super) fn narrow_ordered_value(ty: SqlType, lo: i64, hi: i64, scale: u8) -> SqlValue {
    match ty {
        SqlType::Numeric { .. } => SqlValue::Numeric(Decimal128::new(
            (i128::from(hi) << 64) | i128::from(lo as u64),
            scale,
        )),
        SqlType::Int8 => SqlValue::Int8(lo),
        SqlType::Timestamp => SqlValue::Timestamp(lo),
        SqlType::Int2 => SqlValue::Int2(lo as i16),
        SqlType::Date => SqlValue::Date(lo as i32),
        // bool key / bool MIN/MAX value: the derived int4 column is 0/1 -> Bool.
        SqlType::Bool => SqlValue::Bool(lo != 0),
        _ => SqlValue::Int4(lo as i32),
    }
}

/// COUNT(DISTINCT v) on the non-plain-integer/composite group-key route: the building block of the
/// GROUP-BY-(g,v) reduction.
/// Runs a general composite GROUP BY COUNT(*) over `members` (any mix of fixed-width + text columns --
/// the wide-key buffer for the fixed members + a text descriptor for the text ones), OPTIONALLY
/// prefixed by a typed DERIVED member (the EXPRESSION group key, materialized into its own per-row
/// i32/i64 buffer), over the rows
/// in `indices`, and returns each distinct tuple's REPRESENTATIVE absolute row (the b128 slot's lo).
/// The caller selects the representative-row family for a composite/text/numeric/UUID/bool/expression
/// group key and appends the DISTINCT value member; it never routes the two-fixed packed case, which has
/// no representative row. All grouping is on the GPU; the rep rows are control-plane row indices (like
/// the WHERE survivors) -- NO host download of key values.
pub(super) fn composite_group_count_reps(
    snapshot: &RelationalResidencySnapshot,
    table: &RelationalTable,
    device_memory: &gpu_db_execution::CudaResidentDeviceMemory,
    members: &[(usize, SqlType)],
    indices: &[u32],
    row_count: u64,
    derived: Option<(gpu_db_execution::CudaGroupDeviceView<'_>, bool)>,
) -> Result<Vec<u32>, ExecuteError> {
    let map_err = |e: gpu_db_execution::CudaRuntimeProbeError| {
        ExecuteError::Engine(EngineError::ApplyFailed(e.to_string()))
    };
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    // Fixed members -> the comp_w wide-key buffer (int 8B, numeric/uuid 16B); text -> the descriptor.
    // A derived member (the expr key) is prefixed at dst 0, so column members start at dst 8.
    let mut descriptors: Vec<gpu_db_execution::CudaWideKeyDescriptor<'_>> = Vec::new();
    let mut dst_off: u64 = 0;
    if let Some((buffer, is_i64)) = derived {
        descriptors.push(gpu_db_execution::CudaWideKeyDescriptor {
            source: if is_i64 {
                gpu_db_execution::CudaWideKeySource::DerivedI64 { buffer }
            } else {
                gpu_db_execution::CudaWideKeySource::DerivedI32 { buffer }
            },
            destination_byte_offset: dst_off,
        });
        dst_off += 8;
    }
    for &(idx, ty) in members {
        match ty {
            SqlType::Numeric { .. } | SqlType::Uuid => {
                descriptors.push(gpu_db_execution::CudaWideKeyDescriptor {
                    source: gpu_db_execution::CudaWideKeySource::ResidentI128 {
                        byte_offset: resident_device_numeric_column_offset(snapshot, table, idx)?,
                    },
                    destination_byte_offset: dst_off,
                });
                dst_off += 16;
            }
            SqlType::Int8 | SqlType::Timestamp => {
                descriptors.push(gpu_db_execution::CudaWideKeyDescriptor {
                    source: gpu_db_execution::CudaWideKeySource::ResidentI64 {
                        byte_offset: resident_device_int8_column_offset(snapshot, table, idx)?,
                    },
                    destination_byte_offset: dst_off,
                });
                dst_off += 8;
            }
            SqlType::Bool => {
                descriptors.push(gpu_db_execution::CudaWideKeyDescriptor {
                    source: gpu_db_execution::CudaWideKeySource::ResidentBool {
                        bitmap_byte_offset: resident_device_bool_column_offset(
                            snapshot, table, idx,
                        )?,
                    },
                    destination_byte_offset: dst_off,
                });
                dst_off += 8;
            }
            SqlType::Text => {}
            _ => {
                descriptors.push(gpu_db_execution::CudaWideKeyDescriptor {
                    source: gpu_db_execution::CudaWideKeySource::ResidentI32 {
                        byte_offset: resident_device_int4_column_offset(snapshot, table, idx)?,
                    },
                    destination_byte_offset: dst_off,
                });
                dst_off += 8;
            }
        }
    }
    let comp_w = dst_off;
    let _kbuf = if comp_w > 0 {
        let buf = device_memory
            // COUNT(DISTINCT) reps are over non-NULL keys (a nullable group key with COUNT(DISTINCT)
            // clean-errors upstream), so no per-member validity here.
            .build_wide_key_device(&descriptors, comp_w, row_count, &[])
            .map_err(map_err)?;
        Some(buf)
    } else {
        None
    };
    let mut text_sources = Vec::new();
    for &(idx, ty) in members {
        if matches!(ty, SqlType::Text) {
            let layout = resident_device_text_column_layout(snapshot, table, idx)?;
            text_sources.push(gpu_db_execution::CudaGroupTextSource {
                offsets_byte_offset: layout.offsets_byte_offset,
                bytes_byte_offset: layout.bytes_byte_offset,
                bytes_len: layout.bytes_len,
                row_count,
            });
        }
    }
    let _tdesc = if text_sources.is_empty() {
        None
    } else {
        Some(
            device_memory
                .upload_group_text_descriptors(&text_sources)
                .map_err(map_err)?,
        )
    };
    let key = gpu_db_execution::CudaGroupKeySource::Composite {
        fixed: _kbuf
            .as_ref()
            .map(|buf| gpu_db_execution::CudaGroupWideSource {
                buffer: buf.group_view(),
                row_width: comp_w,
                row_count,
            }),
        text: _tdesc.as_ref().map(|buf| buf.descriptors()),
        row_count,
    };
    let groups = device_memory
        .group_by_i32_count_sum_minmax_from_payload(
            gpu_db_execution::CudaGroupByInput {
                key,
                value: gpu_db_execution::CudaGroupValueSource::Unused { row_count },
                key_validity_bitmap_offset: None,
                value_validity_bitmap_offset: None,
            },
            indices,
            // COUNT(DISTINCT) representative pass: reads only g.key_i128 (no stat field), so the mask is
            // immaterial -- pass ALL (no prune, fully behavior-preserving for this internal pass).
            gpu_db_execution::grouped_agg_mask::COUNT,
        )
        .map_err(map_err)?;
    Ok(groups.iter().map(|g| g.key_i128 as u64 as u32).collect())
}
