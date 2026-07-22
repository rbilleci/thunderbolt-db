use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct DeviceTableSourceDigest {
    pub(super) visible_rows: u64,
    pub(super) lanes: [u64; 4],
}

impl DeviceTableSourceDigest {
    pub(super) fn combine(&mut self, next: Self) {
        self.visible_rows = self.visible_rows.wrapping_add(next.visible_rows);
        for (total, value) in self.lanes.iter_mut().zip(next.lanes) {
            *total = total.wrapping_add(value);
        }
    }
}

pub(super) struct DeviceTableDigestSource<'a> {
    pub(super) snapshot: &'a RelationalResidencySnapshot,
    pub(super) memory: &'a CudaResidentDeviceMemory,
    pub(super) row_count: u64,
    pub(super) row_ids: Option<(&'a CudaResidentDeviceMemory, u64)>,
    pub(super) boundary: i64,
    pub(super) deleted_by: Option<(&'a CudaResidentDeviceMemory, u64)>,
    pub(super) created_by: Option<(&'a CudaResidentDeviceMemory, u64)>,
}

impl Engine {
    fn table_reset_digest_columns(
        &self,
        snapshot: &RelationalResidencySnapshot,
        table: &RelationalTable,
    ) -> Result<Vec<gpu_db_execution::CudaVisibleDigestColumn>, ExecuteError> {
        table
            .columns
            .iter()
            .enumerate()
            .map(|(column_idx, column)| {
                let validity_byte_offset =
                    resident_device_null_column_offset(snapshot, table, column_idx)?;
                let descriptor = match column.ty {
                    SqlType::Int2 | SqlType::Int4 | SqlType::Date => {
                        gpu_db_execution::CudaVisibleDigestColumn::Fixed {
                            byte_offset: resident_device_int4_column_offset(
                                snapshot, table, column_idx,
                            )?,
                            width_bytes: 4,
                            validity_byte_offset,
                        }
                    }
                    SqlType::Int8 | SqlType::Timestamp => {
                        gpu_db_execution::CudaVisibleDigestColumn::Fixed {
                            byte_offset: resident_device_int8_column_offset(
                                snapshot, table, column_idx,
                            )?,
                            width_bytes: 8,
                            validity_byte_offset,
                        }
                    }
                    SqlType::Numeric { .. } | SqlType::Uuid => {
                        gpu_db_execution::CudaVisibleDigestColumn::Fixed {
                            byte_offset: resident_device_numeric_column_offset(
                                snapshot, table, column_idx,
                            )?,
                            width_bytes: 16,
                            validity_byte_offset,
                        }
                    }
                    SqlType::Bool => gpu_db_execution::CudaVisibleDigestColumn::Bool {
                        bitmap_byte_offset: resident_device_bool_column_offset(
                            snapshot, table, column_idx,
                        )?,
                        validity_byte_offset,
                    },
                    SqlType::Text => {
                        let layout =
                            resident_device_text_column_layout(snapshot, table, column_idx)?;
                        gpu_db_execution::CudaVisibleDigestColumn::Text {
                            offsets_byte_offset: layout.offsets_byte_offset,
                            bytes_byte_offset: layout.bytes_byte_offset,
                            bytes_len: layout.bytes_len,
                            validity_byte_offset,
                        }
                    }
                };
                Ok(descriptor)
            })
            .collect()
    }

    pub(super) fn table_reset_digest_source(
        &self,
        table: &RelationalTable,
        source: DeviceTableDigestSource<'_>,
    ) -> Result<DeviceTableSourceDigest, ExecuteError> {
        let columns = self.table_reset_digest_columns(source.snapshot, table)?;
        let digest = source
            .memory
            .digest_visible_source(
                source.row_count,
                &columns,
                source.row_ids,
                source.boundary,
                source.deleted_by,
                source.created_by,
            )
            .map_err(|error| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "device source digest for relation \"{}\" failed: {error}",
                    table.name
                )))
            })?;
        Ok(DeviceTableSourceDigest {
            visible_rows: digest.visible_rows,
            lanes: digest.lanes,
        })
    }

    pub(super) fn upload_table_reset_entity_ids(
        &self,
        gpu_id: u16,
        entity_ids: &[u64],
    ) -> Result<CudaResidentDeviceMemory, ExecuteError> {
        let mut bytes = Vec::with_capacity(entity_ids.len().saturating_mul(8));
        for entity_id in entity_ids {
            bytes.extend_from_slice(&entity_id.to_le_bytes());
        }
        self.relational_residency_device_memory(gpu_id, &bytes)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "stable row identities could not be staged for device source digest"
                        .to_string(),
                ))
            })
    }
}
