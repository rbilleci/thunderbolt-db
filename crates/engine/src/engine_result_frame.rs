//! Strict terminal decoding of one contiguous device-result frame.

use super::*;
use gpu_db_execution::{
    CudaMaterializedColumnKind, CudaMaterializedColumnLayout, CudaMaterializedResultFrame,
};

fn frame_error(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message.into()))
}

fn checked_extent(
    offset: u64,
    len: u64,
    frame_len: usize,
    label: &str,
) -> Result<usize, ExecuteError> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| frame_error(format!("materialized result {label} extent overflows")))?;
    if end > frame_len as u64 {
        return Err(frame_error(format!(
            "materialized result {label} extent exceeds its frame"
        )));
    }
    usize::try_from(offset).map_err(|_| {
        frame_error(format!(
            "materialized result {label} offset exceeds host size"
        ))
    })
}

fn fixed_width(ty: SqlType) -> Option<u8> {
    match ty {
        SqlType::Bool | SqlType::Int2 | SqlType::Int4 | SqlType::Date => Some(4),
        SqlType::Int8 | SqlType::Timestamp => Some(8),
        SqlType::Numeric { .. } | SqlType::Uuid => Some(16),
        SqlType::Text => None,
    }
}

fn validity_at(bytes: &[u8], bitmap: usize, row: usize) -> Result<bool, ExecuteError> {
    let word_offset =
        bitmap
            .checked_add((row / 32).checked_mul(4).ok_or_else(|| {
                frame_error("materialized result validity bitmap index overflows")
            })?)
            .ok_or_else(|| frame_error("materialized result validity bitmap offset overflows"))?;
    let word = bytes
        .get(word_offset..word_offset + 4)
        .ok_or_else(|| frame_error("materialized result validity bitmap is truncated"))?;
    Ok(u32::from_le_bytes(word.try_into().expect("four-byte slice")) & (1 << (row % 32)) != 0)
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, ExecuteError> {
    let raw = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| frame_error("materialized text offsets are truncated"))?;
    Ok(u64::from_le_bytes(
        raw.try_into().expect("eight-byte slice"),
    ))
}

fn decode_fixed(bytes: &[u8], ty: SqlType) -> Result<SqlValue, ExecuteError> {
    Ok(match ty {
        SqlType::Bool => SqlValue::Bool(
            i32::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| frame_error("invalid bool result width"))?,
            ) != 0,
        ),
        SqlType::Int2 => SqlValue::Int2(i32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| frame_error("invalid int2 result width"))?,
        ) as i16),
        SqlType::Int4 => SqlValue::Int4(i32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| frame_error("invalid int4 result width"))?,
        )),
        SqlType::Date => SqlValue::Date(i32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| frame_error("invalid date result width"))?,
        )),
        SqlType::Int8 => SqlValue::Int8(i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| frame_error("invalid int8 result width"))?,
        )),
        SqlType::Timestamp => SqlValue::Timestamp(i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| frame_error("invalid timestamp result width"))?,
        )),
        SqlType::Numeric { scale, .. } => SqlValue::Numeric(gpu_db_sql::Decimal128::new(
            i128::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| frame_error("invalid numeric result width"))?,
            ),
            scale,
        )),
        SqlType::Uuid => SqlValue::Uuid(
            bytes
                .try_into()
                .map_err(|_| frame_error("invalid uuid result width"))?,
        ),
        SqlType::Text => return Err(frame_error("text result used a fixed-width layout")),
    })
}

fn decode_result_parts(
    bytes: &[u8],
    layouts: &[CudaMaterializedColumnLayout],
    row_count: u32,
    types: &[SqlType],
) -> Result<Vec<Vec<SqlValue>>, ExecuteError> {
    if layouts.len() != types.len() {
        return Err(frame_error(
            "materialized result schema does not match its frame columns",
        ));
    }
    if row_count != 0 && layouts.is_empty() {
        return Err(frame_error(
            "materialized result has rows but no result columns",
        ));
    }
    let rows = usize::try_from(row_count)
        .map_err(|_| frame_error("materialized result row count exceeds host size"))?;
    let validity_len = u64::from(row_count)
        .checked_add(31)
        .ok_or_else(|| frame_error("materialized result validity size overflows"))?
        / 32
        * 4;

    // Validate every device-authored extent and text coordinate before allocating by `row_count`.
    // This keeps a corrupt frame a typed error instead of an attacker-controlled host allocation.
    for (layout, ty) in layouts.iter().zip(types) {
        let validity = checked_extent(
            layout.validity_bitmap_offset,
            validity_len,
            bytes.len(),
            "validity bitmap",
        )?;
        match layout.kind {
            CudaMaterializedColumnKind::Fixed { width } => {
                let expected = fixed_width(*ty).ok_or_else(|| {
                    frame_error("materialized text result used a fixed-width layout")
                })?;
                if width != expected {
                    return Err(frame_error(format!(
                        "materialized result width {width} does not match column type width {expected}"
                    )));
                }
                let value_len = u64::from(row_count)
                    .checked_mul(u64::from(width))
                    .ok_or_else(|| frame_error("materialized fixed result size overflows"))?;
                checked_extent(
                    layout.value_byte_offset,
                    value_len,
                    bytes.len(),
                    "fixed values",
                )?;
            }
            CudaMaterializedColumnKind::Text => {
                if *ty != SqlType::Text {
                    return Err(frame_error("materialized fixed result used a text layout"));
                }
                let offsets_len = u64::from(row_count)
                    .checked_add(1)
                    .and_then(|count| count.checked_mul(8))
                    .ok_or_else(|| frame_error("materialized text offset size overflows"))?;
                let offsets = checked_extent(
                    layout.value_byte_offset,
                    offsets_len,
                    bytes.len(),
                    "text offsets",
                )?;
                let blob_offset = layout.text_bytes_byte_offset.ok_or_else(|| {
                    frame_error("materialized text result is missing its byte payload")
                })?;
                let blob = checked_extent(
                    blob_offset,
                    layout.text_bytes_len,
                    bytes.len(),
                    "text bytes",
                )?;
                if u64_at(bytes, offsets)? != 0 {
                    return Err(frame_error(
                        "materialized text offsets do not start at zero",
                    ));
                }
                let mut previous = 0_u64;
                for row in 0..rows {
                    let start = u64_at(bytes, offsets + row * 8)?;
                    let end = u64_at(bytes, offsets + (row + 1) * 8)?;
                    if start != previous || end < start || end > layout.text_bytes_len {
                        return Err(frame_error("materialized text offsets are not contiguous"));
                    }
                    previous = end;
                    if validity_at(bytes, validity, row)? {
                        let start = blob
                            .checked_add(usize::try_from(start).map_err(|_| {
                                frame_error("materialized text start exceeds host size")
                            })?)
                            .ok_or_else(|| frame_error("materialized text start overflows"))?;
                        let end = blob
                            .checked_add(usize::try_from(end).map_err(|_| {
                                frame_error("materialized text end exceeds host size")
                            })?)
                            .ok_or_else(|| frame_error("materialized text end overflows"))?;
                        std::str::from_utf8(&bytes[start..end])
                            .map_err(|_| frame_error("materialized text result is not UTF-8"))?;
                    }
                }
                if previous != layout.text_bytes_len {
                    return Err(frame_error(
                        "materialized text offsets do not cover the byte payload",
                    ));
                }
            }
        }
    }

    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(rows)
        .map_err(|_| frame_error("materialized result row allocation is too large"))?;
    for _ in 0..rows {
        let mut row = Vec::new();
        row.try_reserve_exact(types.len())
            .map_err(|_| frame_error("materialized result column allocation is too large"))?;
        decoded.push(row);
    }

    for (layout, ty) in layouts.iter().zip(types) {
        let validity = checked_extent(
            layout.validity_bitmap_offset,
            validity_len,
            bytes.len(),
            "validity bitmap",
        )?;
        match layout.kind {
            CudaMaterializedColumnKind::Fixed { width } => {
                let expected = fixed_width(*ty).ok_or_else(|| {
                    frame_error("materialized text result used a fixed-width layout")
                })?;
                if width != expected {
                    return Err(frame_error(format!(
                        "materialized result width {width} does not match column type width {expected}"
                    )));
                }
                let value_len = u64::from(row_count)
                    .checked_mul(u64::from(width))
                    .ok_or_else(|| frame_error("materialized fixed result size overflows"))?;
                let values = checked_extent(
                    layout.value_byte_offset,
                    value_len,
                    bytes.len(),
                    "fixed values",
                )?;
                for (row, out) in decoded.iter_mut().enumerate() {
                    if !validity_at(bytes, validity, row)? {
                        out.push(SqlValue::Null);
                        continue;
                    }
                    let start = values
                        .checked_add(row.checked_mul(width as usize).ok_or_else(|| {
                            frame_error("materialized fixed result index overflows")
                        })?)
                        .ok_or_else(|| frame_error("materialized fixed result offset overflows"))?;
                    let end = start + width as usize;
                    out.push(decode_fixed(&bytes[start..end], *ty)?);
                }
            }
            CudaMaterializedColumnKind::Text => {
                if *ty != SqlType::Text {
                    return Err(frame_error("materialized fixed result used a text layout"));
                }
                let offsets_len = u64::from(row_count)
                    .checked_add(1)
                    .and_then(|count| count.checked_mul(8))
                    .ok_or_else(|| frame_error("materialized text offset size overflows"))?;
                let offsets = checked_extent(
                    layout.value_byte_offset,
                    offsets_len,
                    bytes.len(),
                    "text offsets",
                )?;
                let blob_offset = layout.text_bytes_byte_offset.ok_or_else(|| {
                    frame_error("materialized text result is missing its byte payload")
                })?;
                let blob = checked_extent(
                    blob_offset,
                    layout.text_bytes_len,
                    bytes.len(),
                    "text bytes",
                )?;
                if u64_at(bytes, offsets)? != 0 {
                    return Err(frame_error(
                        "materialized text offsets do not start at zero",
                    ));
                }
                let mut previous = 0_u64;
                for (row, out) in decoded.iter_mut().enumerate() {
                    let start = u64_at(bytes, offsets + row * 8)?;
                    let end = u64_at(bytes, offsets + (row + 1) * 8)?;
                    if start != previous || end < start || end > layout.text_bytes_len {
                        return Err(frame_error("materialized text offsets are not contiguous"));
                    }
                    previous = end;
                    if !validity_at(bytes, validity, row)? {
                        out.push(SqlValue::Null);
                        continue;
                    }
                    let start = blob
                        .checked_add(usize::try_from(start).map_err(|_| {
                            frame_error("materialized text start exceeds host size")
                        })?)
                        .ok_or_else(|| frame_error("materialized text start overflows"))?;
                    let end = blob
                        .checked_add(
                            usize::try_from(end).map_err(|_| {
                                frame_error("materialized text end exceeds host size")
                            })?,
                        )
                        .ok_or_else(|| frame_error("materialized text end overflows"))?;
                    let text = std::str::from_utf8(&bytes[start..end])
                        .map_err(|_| frame_error("materialized text result is not UTF-8"))?;
                    let mut owned = String::new();
                    owned
                        .try_reserve_exact(text.len())
                        .map_err(|_| frame_error("materialized text allocation is too large"))?;
                    owned.push_str(text);
                    out.push(SqlValue::Text(owned));
                }
                if previous != layout.text_bytes_len {
                    return Err(frame_error(
                        "materialized text offsets do not cover the byte payload",
                    ));
                }
            }
        }
    }
    Ok(decoded)
}

impl Engine {
    pub(crate) fn decode_materialized_result_frame(
        &self,
        frame: &CudaMaterializedResultFrame,
        columns: &[RelationalColumn],
    ) -> Result<Vec<Vec<SqlValue>>, ExecuteError> {
        let types = columns.iter().map(|column| column.ty).collect::<Vec<_>>();
        decode_result_parts(frame.bytes(), frame.columns(), frame.row_count(), &types)
    }

    pub(crate) fn decode_materialized_result_column_frame(
        &self,
        frame: &CudaMaterializedResultFrame,
        ty: SqlType,
    ) -> Result<Vec<SqlValue>, ExecuteError> {
        decode_result_parts(frame.bytes(), frame.columns(), frame.row_count(), &[ty])?
            .into_iter()
            .map(|mut row| {
                row.pop()
                    .ok_or_else(|| frame_error("materialized result column is missing"))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_layout(frame: &mut Vec<u8>, width: u8, first: &[u8]) -> CudaMaterializedColumnLayout {
        assert_eq!(first.len(), width as usize);
        let value_byte_offset = frame.len() as u64;
        frame.extend_from_slice(first);
        frame.resize(frame.len() + width as usize, 0);
        let validity_bitmap_offset = frame.len() as u64;
        frame.extend_from_slice(&1_u32.to_le_bytes());
        CudaMaterializedColumnLayout {
            kind: CudaMaterializedColumnKind::Fixed { width },
            value_byte_offset,
            text_bytes_byte_offset: None,
            text_bytes_len: 0,
            validity_bitmap_offset,
        }
    }

    #[test]
    fn terminal_frame_decodes_every_scalar_type_and_null() {
        let mut bytes = Vec::new();
        let mut layouts = Vec::new();
        let mut types = Vec::new();

        layouts.push(fixed_layout(&mut bytes, 4, &1_i32.to_le_bytes()));
        types.push(SqlType::Bool);
        layouts.push(fixed_layout(&mut bytes, 4, &(-7_i32).to_le_bytes()));
        types.push(SqlType::Int2);
        layouts.push(fixed_layout(&mut bytes, 4, &42_i32.to_le_bytes()));
        types.push(SqlType::Int4);
        layouts.push(fixed_layout(&mut bytes, 4, &20_000_i32.to_le_bytes()));
        types.push(SqlType::Date);
        layouts.push(fixed_layout(&mut bytes, 8, &99_i64.to_le_bytes()));
        types.push(SqlType::Int8);
        layouts.push(fixed_layout(&mut bytes, 8, &123_456_i64.to_le_bytes()));
        types.push(SqlType::Timestamp);
        layouts.push(fixed_layout(&mut bytes, 16, &12_345_i128.to_le_bytes()));
        types.push(SqlType::Numeric {
            precision: 10,
            scale: 2,
        });
        layouts.push(fixed_layout(&mut bytes, 16, &[0xabu8; 16]));
        types.push(SqlType::Uuid);

        let value_byte_offset = bytes.len() as u64;
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        let text_bytes_byte_offset = bytes.len() as u64;
        bytes.extend_from_slice(b"ok");
        let validity_bitmap_offset = bytes.len() as u64;
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        layouts.push(CudaMaterializedColumnLayout {
            kind: CudaMaterializedColumnKind::Text,
            value_byte_offset,
            text_bytes_byte_offset: Some(text_bytes_byte_offset),
            text_bytes_len: 2,
            validity_bitmap_offset,
        });
        types.push(SqlType::Text);

        let rows = decode_result_parts(&bytes, &layouts, 2, &types).expect("valid frame");
        assert_eq!(
            rows[0],
            vec![
                SqlValue::Bool(true),
                SqlValue::Int2(-7),
                SqlValue::Int4(42),
                SqlValue::Date(20_000),
                SqlValue::Int8(99),
                SqlValue::Timestamp(123_456),
                SqlValue::Numeric(gpu_db_sql::Decimal128::new(12_345, 2)),
                SqlValue::Uuid([0xabu8; 16]),
                SqlValue::Text("ok".to_string()),
            ]
        );
        assert!(rows[1].iter().all(|value| *value == SqlValue::Null));
    }

    #[test]
    fn terminal_frame_rejects_malformed_layouts_and_text() {
        let fixed = CudaMaterializedColumnLayout {
            kind: CudaMaterializedColumnKind::Fixed { width: 8 },
            value_byte_offset: 0,
            text_bytes_byte_offset: None,
            text_bytes_len: 0,
            validity_bitmap_offset: 8,
        };
        assert!(decode_result_parts(&[0; 12], &[fixed], 1, &[SqlType::Int4]).is_err());

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        let blob = bytes.len() as u64;
        bytes.extend_from_slice(&[0xff, 0xff]);
        let validity = bytes.len() as u64;
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        let text = CudaMaterializedColumnLayout {
            kind: CudaMaterializedColumnKind::Text,
            value_byte_offset: 0,
            text_bytes_byte_offset: Some(blob),
            text_bytes_len: 2,
            validity_bitmap_offset: validity,
        };
        assert!(decode_result_parts(&bytes, &[text], 2, &[SqlType::Text]).is_err());

        bytes[16..24].copy_from_slice(&2_u64.to_le_bytes());
        assert!(decode_result_parts(&bytes, &[text], 2, &[SqlType::Text]).is_err());
    }
}
