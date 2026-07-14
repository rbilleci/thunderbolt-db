//! Typed result-column decoding for streaming LAG and LEAD windows.

use super::*;

impl Engine {
    pub(crate) fn decode_materialized_column(
        &self,
        run: &gpu_db_execution::CudaMaterializedRelation,
        column_index: usize,
        ty: SqlType,
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
    ) -> Result<Vec<SqlValue>, ExecuteError> {
        use gpu_db_execution::CudaMaterializedColumnKind;

        let map_err = |err: gpu_db_execution::CudaRuntimeProbeError| {
            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
        };
        let layout = *run.columns().get(column_index).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "materialized column index is outside the run schema".to_string(),
            ))
        })?;
        Ok(match layout.kind {
            CudaMaterializedColumnKind::Text => run
                .memory()
                .project_text_from_join_coordinates(
                    coordinates,
                    0,
                    run.memory(),
                    layout.value_byte_offset,
                    layout.text_bytes_byte_offset.expect("text bytes layout"),
                    layout.text_bytes_len,
                    Some(layout.validity_bitmap_offset),
                )
                .map_err(map_err)?
                .into_iter()
                .map(|value| value.map_or(SqlValue::Null, SqlValue::Text))
                .collect(),
            CudaMaterializedColumnKind::Fixed { width } => {
                let (raw, valid) = run
                    .memory()
                    .project_fixed_from_join_coordinates(
                        coordinates,
                        0,
                        run.memory(),
                        layout.value_byte_offset,
                        Some(layout.validity_bitmap_offset),
                        width,
                    )
                    .map_err(map_err)?;
                raw.chunks_exact(width as usize)
                    .zip(valid)
                    .map(|(bytes, valid)| {
                        if !valid {
                            return SqlValue::Null;
                        }
                        match ty {
                            SqlType::Bool => SqlValue::Bool(
                                i32::from_le_bytes(bytes.try_into().expect("bool width")) != 0,
                            ),
                            SqlType::Int2 => SqlValue::Int2(i32::from_le_bytes(
                                bytes.try_into().expect("int2 width"),
                            ) as i16),
                            SqlType::Int4 => SqlValue::Int4(i32::from_le_bytes(
                                bytes.try_into().expect("int4 width"),
                            )),
                            SqlType::Date => SqlValue::Date(i32::from_le_bytes(
                                bytes.try_into().expect("date width"),
                            )),
                            SqlType::Int8 => SqlValue::Int8(i64::from_le_bytes(
                                bytes.try_into().expect("int8 width"),
                            )),
                            SqlType::Timestamp => SqlValue::Timestamp(i64::from_le_bytes(
                                bytes.try_into().expect("timestamp width"),
                            )),
                            SqlType::Numeric { scale, .. } => {
                                SqlValue::Numeric(gpu_db_sql::Decimal128::new(
                                    i128::from_le_bytes(bytes.try_into().expect("numeric width")),
                                    scale,
                                ))
                            }
                            SqlType::Uuid => SqlValue::Uuid(bytes.try_into().expect("uuid width")),
                            SqlType::Text => unreachable!(),
                        }
                    })
                    .collect()
            }
        })
    }
}
