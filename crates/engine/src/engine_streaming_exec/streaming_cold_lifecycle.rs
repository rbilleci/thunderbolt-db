//! Streaming transient/cold staging, patch, load, spill, and publication lifecycle.

use super::*;

mod cold_index_validation_support;
use cold_index_validation_support::{
    cold_validation_descriptor_bytes, try_clone_cold_int4_stats, try_clone_cold_memory_proof,
    try_clone_cold_string, try_clone_cold_strings, validate_cold_text_offsets_bounded,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ColdIndexValidationWindow {
    pub(crate) chunk_ordinal: usize,
    pub(crate) row_start: usize,
    pub(crate) row_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct ColdFixedColumnWindow {
    source_capacity: usize,
    row_start: usize,
    row_count: usize,
}

struct ColdValidationCopy<'a> {
    source: &'a super::ColdPayload,
    table: &'a str,
    host_limit: usize,
    required: usize,
}

struct ColdIndexValidationPayload {
    resident_bytes: u64,
    bool_columns: Vec<ResidentDeviceBoolColumnLayout>,
    text_columns: Vec<ResidentDeviceTextColumnLayout>,
    null_columns: Vec<ResidentDeviceNullBitmapLayout>,
    payload: Vec<u8>,
    deleted_by: Option<Vec<u8>>,
    host_staging_bytes: usize,
}

pub(crate) const COLD_INDEX_MAX_HOST_STAGING_BYTES: usize = 256 * 1024 * 1024;
const COLD_BITMAP_SCAN_BLOCK_BYTES: usize = 64 * 1024;

fn cold_index_validation_error(message: impl Into<String>) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(message.into()))
}

fn cold_index_host_staging_exhausted(table: &str, required: usize, limit: usize) -> ExecuteError {
    ExecuteError::ResourceExhausted(format!(
        "transactional UNIQUE index validation for relation \"{table}\" requires {required} \
         bytes of cold host staging above its bounded {limit}-byte lease; raise the GPU residency \
         budget, compact/admit the relation, and retry"
    ))
}

fn map_cold_payload_range_error(
    table: &str,
    required: usize,
    limit: usize,
    error: super::ColdPayloadRangeError,
    truncated: &'static str,
) -> ExecuteError {
    match error {
        super::ColdPayloadRangeError::Allocation => {
            cold_index_host_staging_exhausted(table, required, limit)
        }
        super::ColdPayloadRangeError::OutOfBoundsOrIo => cold_index_validation_error(truncated),
    }
}

fn read_cold_payload_u64(
    payload: &super::ColdPayload,
    byte_offset: usize,
) -> Result<u64, ExecuteError> {
    payload
        .read_array(byte_offset)
        .map(u64::from_le_bytes)
        .map_err(|_| cold_index_validation_error("cold validation offset array is truncated"))
}

impl ColdValidationCopy<'_> {
    fn append_fixed_columns(
        &self,
        destination: &mut Vec<u8>,
        section_start: usize,
        column_count: usize,
        window: ColdFixedColumnWindow,
        width: usize,
    ) -> Result<usize, ExecuteError> {
        let column_stride = window
            .source_capacity
            .checked_mul(width)
            .ok_or_else(|| cold_index_validation_error("cold fixed-column stride overflowed"))?;
        let slice_start = window
            .row_start
            .checked_mul(width)
            .ok_or_else(|| cold_index_validation_error("cold fixed-column window overflowed"))?;
        let slice_len = window
            .row_count
            .checked_mul(width)
            .ok_or_else(|| cold_index_validation_error("cold fixed-column length overflowed"))?;
        for ordinal in 0..column_count {
            let start =
                section_start
                    .checked_add(ordinal.checked_mul(column_stride).ok_or_else(|| {
                        cold_index_validation_error("cold column offset overflowed")
                    })?)
                    .and_then(|offset| offset.checked_add(slice_start))
                    .ok_or_else(|| cold_index_validation_error("cold column offset overflowed"))?;
            let end = start
                .checked_add(slice_len)
                .ok_or_else(|| cold_index_validation_error("cold column extent overflowed"))?;
            self.source
                .append_range(start, end - start, destination)
                .map_err(|error| {
                    map_cold_payload_range_error(
                        self.table,
                        self.required,
                        self.host_limit,
                        error,
                        "cold fixed-column payload is truncated",
                    )
                })?;
        }
        section_start
            .checked_add(
                column_count
                    .checked_mul(column_stride)
                    .ok_or_else(|| cold_index_validation_error("cold section extent overflowed"))?,
            )
            .ok_or_else(|| cold_index_validation_error("cold section extent overflowed"))
    }

    fn append_bitmap(
        &self,
        destination: &mut Vec<u8>,
        source_offset: u64,
        source_capacity: usize,
        row_start: usize,
        row_count: usize,
    ) -> Result<(u64, u64, u64), ExecuteError> {
        let source_offset = usize::try_from(source_offset)
            .map_err(|_| cold_index_validation_error("cold bitmap offset exceeds host framing"))?;
        let source_bytes = source_capacity
            .div_ceil(32)
            .checked_mul(4)
            .ok_or_else(|| cold_index_validation_error("cold bitmap extent overflowed"))?;
        if source_offset
            .checked_add(source_bytes)
            .is_none_or(|end| end > self.source.len())
        {
            return Err(cold_index_validation_error(
                "cold bitmap payload is truncated",
            ));
        }
        let output_offset = u64::try_from(destination.len()).map_err(|_| {
            cold_index_validation_error("cold bitmap output exceeds device framing")
        })?;
        let output_words = row_count.div_ceil(32);
        destination
            .try_reserve_exact(output_words.saturating_mul(4))
            .map_err(|_| {
                cold_index_host_staging_exhausted(self.table, self.required, self.host_limit)
            })?;
        let first_source_word = row_start / 32;
        let source_word_count = row_count
            .checked_add(row_start % 32)
            .ok_or_else(|| cold_index_validation_error("cold bitmap window overflowed"))?
            .div_ceil(32);
        let mut block = [0u8; COLD_BITMAP_SCAN_BLOCK_BYTES];
        let mut loaded_word_start = usize::MAX;
        let mut loaded_word_count = 0usize;
        let mut spill_reads = 0u64;
        let mut spill_bytes = 0u64;
        for output_word in 0..output_words {
            let mut word = 0u32;
            let local_start = output_word * 32;
            let local_end = row_count.min(local_start + 32);
            for local in local_start..local_end {
                let source_row = row_start + local;
                let source_word_ordinal = source_row / 32;
                if source_word_ordinal < loaded_word_start
                    || source_word_ordinal >= loaded_word_start.saturating_add(loaded_word_count)
                {
                    let consumed_words = source_word_ordinal
                        .checked_sub(first_source_word)
                        .ok_or_else(|| {
                            cold_index_validation_error("cold bitmap window underflowed")
                        })?;
                    let remaining_words = source_word_count
                        .checked_sub(consumed_words)
                        .ok_or_else(|| {
                            cold_index_validation_error("cold bitmap window overflowed")
                        })?;
                    loaded_word_count = remaining_words
                        .min(COLD_BITMAP_SCAN_BLOCK_BYTES / std::mem::size_of::<u32>());
                    let read_len = loaded_word_count
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(|| {
                            cold_index_validation_error("cold bitmap range overflowed")
                        })?;
                    let read_offset = source_word_ordinal
                        .checked_mul(std::mem::size_of::<u32>())
                        .and_then(|offset| source_offset.checked_add(offset))
                        .ok_or_else(|| {
                            cold_index_validation_error("cold bitmap range overflowed")
                        })?;
                    self.source
                        .read_exact_range(read_offset, &mut block[..read_len])
                        .map_err(|_| {
                            cold_index_validation_error("cold bitmap payload is truncated")
                        })?;
                    if self.source.is_spilled() {
                        spill_reads = spill_reads.checked_add(1).ok_or_else(|| {
                            cold_index_validation_error("cold bitmap read count overflowed")
                        })?;
                        spill_bytes =
                            spill_bytes.checked_add(read_len as u64).ok_or_else(|| {
                                cold_index_validation_error("cold bitmap read bytes overflowed")
                            })?;
                    }
                    loaded_word_start = source_word_ordinal;
                }
                let block_offset = source_word_ordinal
                    .checked_sub(loaded_word_start)
                    .and_then(|word| word.checked_mul(std::mem::size_of::<u32>()))
                    .ok_or_else(|| cold_index_validation_error("cold bitmap block underflowed"))?;
                let source_word = u32::from_le_bytes(
                    block[block_offset..block_offset + 4]
                        .try_into()
                        .expect("four-byte bitmap word"),
                );
                if source_word & (1u32 << (source_row % 32)) != 0 {
                    word |= 1u32 << (local % 32);
                }
            }
            destination.extend_from_slice(&word.to_le_bytes());
        }
        Ok((output_offset, spill_reads, spill_bytes))
    }
}

/// Slice one immutable cold payload by row coordinates without decoding or evaluating relational
/// values. Fixed columns byte-copy, bool/validity metadata bit-repacks, and text offset metadata is
/// rebased around the selected blob span. The result is the same dense device format consumed by
/// the shared resident executor; all NULL filtering and key equality remain GPU operations.
fn slice_cold_index_validation_window(
    chunk: &ColdChunk,
    window: ColdIndexValidationWindow,
    host_limit: usize,
) -> Result<ColdIndexValidationPayload, ExecuteError> {
    let source = &chunk.payload;
    let table = chunk.snapshot.table.as_str();
    let source_rows = usize::try_from(chunk.row_count)
        .map_err(|_| cold_index_validation_error("cold row count exceeds host framing"))?;
    let row_end = window
        .row_start
        .checked_add(window.row_count)
        .ok_or_else(|| cold_index_validation_error("cold validation window overflowed"))?;
    if window.row_count == 0
        || row_end > source_rows
        || chunk.snapshot.row_count != source_rows
        || chunk.snapshot.capacity < source_rows
    {
        return Err(cold_index_validation_error(
            "cold validation window is outside its immutable payload",
        ));
    }
    if read_cold_payload_u64(source, 0)? != chunk.row_count {
        return Err(cold_index_validation_error(
            "cold validation payload header is torn",
        ));
    }

    let capacity = chunk.snapshot.capacity;
    let int4_count = chunk.snapshot.resident_device_int4_columns.len();
    let int8_count = chunk.snapshot.resident_device_int8_columns.len();
    let b128_count = chunk.snapshot.resident_device_numeric_columns.len();
    let fixed_bytes = window
        .row_count
        .checked_mul(
            int4_count
                .checked_mul(4)
                .and_then(|bytes| {
                    int8_count
                        .checked_mul(8)
                        .and_then(|int8| bytes.checked_add(int8))
                })
                .and_then(|bytes| {
                    b128_count
                        .checked_mul(16)
                        .and_then(|b128| bytes.checked_add(b128))
                })
                .ok_or_else(|| cold_index_validation_error("cold fixed geometry overflowed"))?,
        )
        .ok_or_else(|| cold_index_validation_error("cold fixed geometry overflowed"))?;
    let output_bitmap_bytes = window
        .row_count
        .div_ceil(32)
        .checked_mul(4)
        .ok_or_else(|| cold_index_validation_error("cold bitmap geometry overflowed"))?;
    let bitmap_count = chunk
        .snapshot
        .resident_device_bool_columns
        .len()
        .checked_add(chunk.snapshot.resident_device_null_columns.len())
        .ok_or_else(|| cold_index_validation_error("cold bitmap geometry overflowed"))?;
    let mut required_payload = 8usize
        .checked_add(fixed_bytes)
        .and_then(|bytes| {
            output_bitmap_bytes
                .checked_mul(bitmap_count)
                .and_then(|bitmaps| bytes.checked_add(bitmaps))
        })
        .ok_or_else(|| cold_index_validation_error("cold payload geometry overflowed"))?;
    for layout in &chunk.snapshot.resident_device_text_columns {
        required_payload = required_payload
            .checked_add((8usize.wrapping_sub(required_payload % 8)) % 8)
            .ok_or_else(|| cold_index_validation_error("cold text geometry overflowed"))?;
        let source_offsets = usize::try_from(layout.offsets_byte_offset)
            .map_err(|_| cold_index_validation_error("cold text offsets exceed host framing"))?;
        let base_offset = window
            .row_start
            .checked_mul(8)
            .and_then(|offset| source_offsets.checked_add(offset))
            .ok_or_else(|| cold_index_validation_error("cold text offset overflowed"))?;
        let limit_offset = row_end
            .checked_mul(8)
            .and_then(|offset| source_offsets.checked_add(offset))
            .ok_or_else(|| cold_index_validation_error("cold text offset overflowed"))?;
        let base = read_cold_payload_u64(source, base_offset)?;
        let limit = read_cold_payload_u64(source, limit_offset)?;
        if base > limit || limit > layout.bytes_len {
            return Err(cold_index_validation_error(
                "cold text offsets are non-monotonic or out of bounds",
            ));
        }
        let offsets_bytes = window
            .row_count
            .checked_add(1)
            .and_then(|entries| entries.checked_mul(8))
            .ok_or_else(|| cold_index_validation_error("cold text geometry overflowed"))?;
        let blob_bytes = usize::try_from(limit - base)
            .map_err(|_| cold_index_validation_error("cold text window exceeds host framing"))?;
        required_payload = required_payload
            .checked_add(offsets_bytes)
            .and_then(|bytes| bytes.checked_add(blob_bytes))
            .ok_or_else(|| cold_index_validation_error("cold text geometry overflowed"))?;
    }
    let sidecar_bytes = chunk
        .deleted_by
        .as_ref()
        .map_or(0, |_| window.row_count.saturating_mul(8));
    let host_staging_bytes = required_payload
        .checked_add(sidecar_bytes)
        .ok_or_else(|| cold_index_host_staging_exhausted(table, usize::MAX, host_limit))?;
    if host_staging_bytes > host_limit {
        return Err(cold_index_host_staging_exhausted(
            table,
            host_staging_bytes,
            host_limit,
        ));
    }
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(required_payload)
        .map_err(|_| cold_index_host_staging_exhausted(table, host_staging_bytes, host_limit))?;
    payload.extend_from_slice(&(window.row_count as u64).to_le_bytes());
    let fixed_window = ColdFixedColumnWindow {
        source_capacity: capacity,
        row_start: window.row_start,
        row_count: window.row_count,
    };
    let copy = ColdValidationCopy {
        source,
        table,
        host_limit,
        required: host_staging_bytes,
    };
    let mut section = 8usize;
    section = copy.append_fixed_columns(&mut payload, section, int4_count, fixed_window, 4)?;
    section = copy.append_fixed_columns(&mut payload, section, int8_count, fixed_window, 8)?;
    let _fixed_end =
        copy.append_fixed_columns(&mut payload, section, b128_count, fixed_window, 16)?;

    let mut bool_columns = Vec::new();
    bool_columns
        .try_reserve_exact(chunk.snapshot.resident_device_bool_columns.len())
        .map_err(|_| cold_index_host_staging_exhausted(table, host_staging_bytes, host_limit))?;
    for layout in &chunk.snapshot.resident_device_bool_columns {
        let mut name = String::new();
        name.try_reserve_exact(layout.name.len()).map_err(|_| {
            cold_index_host_staging_exhausted(table, host_staging_bytes, host_limit)
        })?;
        name.push_str(&layout.name);
        bool_columns.push(ResidentDeviceBoolColumnLayout {
            name,
            bitmap_byte_offset: copy
                .append_bitmap(
                    &mut payload,
                    layout.bitmap_byte_offset,
                    capacity,
                    window.row_start,
                    window.row_count,
                )?
                .0,
        });
    }
    let mut null_columns = Vec::new();
    null_columns
        .try_reserve_exact(chunk.snapshot.resident_device_null_columns.len())
        .map_err(|_| cold_index_host_staging_exhausted(table, host_staging_bytes, host_limit))?;
    for layout in &chunk.snapshot.resident_device_null_columns {
        let mut name = String::new();
        name.try_reserve_exact(layout.name.len()).map_err(|_| {
            cold_index_host_staging_exhausted(table, host_staging_bytes, host_limit)
        })?;
        name.push_str(&layout.name);
        null_columns.push(ResidentDeviceNullBitmapLayout {
            name,
            bitmap_byte_offset: copy
                .append_bitmap(
                    &mut payload,
                    layout.bitmap_byte_offset,
                    capacity,
                    window.row_start,
                    window.row_count,
                )?
                .0,
        });
    }
    let mut text_columns = Vec::new();
    text_columns
        .try_reserve_exact(chunk.snapshot.resident_device_text_columns.len())
        .map_err(|_| cold_index_host_staging_exhausted(table, host_staging_bytes, host_limit))?;
    for layout in &chunk.snapshot.resident_device_text_columns {
        while !payload.len().is_multiple_of(8) {
            payload.push(0);
        }
        let source_offsets = usize::try_from(layout.offsets_byte_offset)
            .map_err(|_| cold_index_validation_error("cold text offsets exceed host framing"))?;
        let source_bytes = usize::try_from(layout.bytes_byte_offset)
            .map_err(|_| cold_index_validation_error("cold text bytes exceed host framing"))?;
        let source_blob_len = usize::try_from(layout.bytes_len)
            .map_err(|_| cold_index_validation_error("cold text blob exceeds host framing"))?;
        let window_offsets_start = window
            .row_start
            .checked_mul(8)
            .and_then(|offset| source_offsets.checked_add(offset))
            .ok_or_else(|| cold_index_validation_error("cold text offset overflowed"))?;
        let window_offsets_len = window
            .row_count
            .checked_add(1)
            .and_then(|entries| entries.checked_mul(8))
            .ok_or_else(|| cold_index_validation_error("cold text offset overflowed"))?;
        let base = read_cold_payload_u64(source, window_offsets_start)?;
        let limit = read_cold_payload_u64(
            source,
            row_end
                .checked_mul(8)
                .and_then(|offset| source_offsets.checked_add(offset))
                .ok_or_else(|| cold_index_validation_error("cold text offset overflowed"))?,
        )?;
        if base > limit || limit > layout.bytes_len {
            return Err(cold_index_validation_error(
                "cold text offsets are non-monotonic or out of bounds",
            ));
        }
        let offsets_byte_offset = payload.len() as u64;
        let output_offsets_start = payload.len();
        source
            .append_range(window_offsets_start, window_offsets_len, &mut payload)
            .map_err(|error| {
                map_cold_payload_range_error(
                    table,
                    host_staging_bytes,
                    host_limit,
                    error,
                    "cold text offsets are truncated",
                )
            })?;
        let mut previous = base;
        for encoded in payload[output_offsets_start..].chunks_exact_mut(std::mem::size_of::<u64>())
        {
            let offset = u64::from_le_bytes(encoded.try_into().expect("eight-byte offset"));
            if offset < base || offset > limit {
                return Err(cold_index_validation_error(
                    "cold text window offsets are non-monotonic",
                ));
            }
            if offset < previous {
                return Err(cold_index_validation_error(
                    "cold text window offsets are non-monotonic",
                ));
            }
            previous = offset;
            encoded.copy_from_slice(&(offset - base).to_le_bytes());
        }
        let bytes_byte_offset = payload.len() as u64;
        let blob_start =
            source_bytes
                .checked_add(usize::try_from(base).map_err(|_| {
                    cold_index_validation_error("cold text base exceeds host framing")
                })?)
                .ok_or_else(|| cold_index_validation_error("cold text base overflowed"))?;
        let blob_end =
            source_bytes
                .checked_add(usize::try_from(limit).map_err(|_| {
                    cold_index_validation_error("cold text limit exceeds host framing")
                })?)
                .ok_or_else(|| cold_index_validation_error("cold text limit overflowed"))?;
        if source_bytes
            .checked_add(source_blob_len)
            .is_none_or(|end| end > source.len())
        {
            return Err(cold_index_validation_error("cold text blob is truncated"));
        }
        source
            .append_range(blob_start, blob_end - blob_start, &mut payload)
            .map_err(|error| {
                map_cold_payload_range_error(
                    table,
                    host_staging_bytes,
                    host_limit,
                    error,
                    "cold text window blob is truncated",
                )
            })?;
        let mut name = String::new();
        name.try_reserve_exact(layout.name.len()).map_err(|_| {
            cold_index_host_staging_exhausted(table, host_staging_bytes, host_limit)
        })?;
        name.push_str(&layout.name);
        text_columns.push(ResidentDeviceTextColumnLayout {
            name,
            offsets_byte_offset,
            bytes_byte_offset,
            bytes_len: limit - base,
        });
    }

    let deleted_by = chunk
        .deleted_by
        .as_ref()
        .map(|sidecar| {
            let start = window
                .row_start
                .checked_mul(8)
                .ok_or_else(|| cold_index_validation_error("cold tombstone offset overflowed"))?;
            let end = row_end
                .checked_mul(8)
                .ok_or_else(|| cold_index_validation_error("cold tombstone extent overflowed"))?;
            let source = sidecar.get(start..end).ok_or_else(|| {
                cold_index_validation_error("cold tombstone sidecar is truncated")
            })?;
            let mut window = Vec::new();
            window.try_reserve_exact(source.len()).map_err(|_| {
                cold_index_host_staging_exhausted(table, host_staging_bytes, host_limit)
            })?;
            window.extend_from_slice(source);
            Ok::<Vec<u8>, ExecuteError>(window)
        })
        .transpose()?;
    if payload.len() != required_payload {
        return Err(cold_index_validation_error(
            "cold validation payload geometry changed while slicing",
        ));
    }
    Ok(ColdIndexValidationPayload {
        resident_bytes: payload.len() as u64,
        bool_columns,
        text_columns,
        null_columns,
        payload,
        deleted_by,
        host_staging_bytes,
    })
}

impl Engine {
    #[cfg(test)]
    pub(crate) fn clone_cold_without_chunk_for_test(
        cold: &ColdTableChunks,
        remove_at: usize,
    ) -> Arc<ColdTableChunks> {
        let clone_payload = |payload: &super::ColdPayload| match payload {
            super::ColdPayload::Ram(bytes) => super::ColdPayload::Ram(Arc::clone(bytes)),
            super::ColdPayload::Spilled { file, offset, len } => super::ColdPayload::Spilled {
                file: Arc::clone(file),
                offset: *offset,
                len: *len,
            },
        };
        let mut chunks = cold
            .chunks
            .iter()
            .map(|chunk| super::ColdChunk {
                payload: clone_payload(&chunk.payload),
                snapshot: chunk.snapshot.clone(),
                row_count: chunk.row_count,
                entity_ids: Arc::clone(&chunk.entity_ids),
                chunk_id: chunk.chunk_id,
                tuple_range: chunk.tuple_range,
                payload_copin_s: chunk.payload_copin_s,
                deleted_by: chunk.deleted_by.as_ref().map(Arc::clone),
            })
            .collect::<Vec<_>>();
        chunks.remove(remove_at);
        let total_payload_bytes = chunks
            .iter()
            .map(|chunk| {
                chunk
                    .payload
                    .read()
                    .expect("test sabotage retains readable cold payloads")
                    .len() as u64
                    + chunk
                        .deleted_by
                        .as_ref()
                        .map_or(0, |sidecar| sidecar.len() as u64)
            })
            .sum();
        Arc::new(ColdTableChunks {
            generation: Arc::clone(&cold.generation),
            column_signature: cold.column_signature.clone(),
            build_copin_s: cold.build_copin_s,
            chunk_target_bytes: cold.chunk_target_bytes,
            total_payload_bytes,
            spilled: cold.spilled,
            entry_epoch: cold.entry_epoch,
            chunks,
        })
    }

    /// Prove that a transaction-selected cold entry is the complete immutable DEVICE-FORMAT
    /// authority for `table` before UNIQUE validation is allowed to use its row count. This is a
    /// structural proof only: SQL NULL/key semantics still execute exclusively through the GPU
    /// predicate + GROUP path. In particular, even a zero/one-row shortcut must reject a stale
    /// generation, a future-born chunk, or a torn payload instead of treating it as an empty table.
    pub(crate) fn validate_transaction_cold_index_authority(
        table: &RelationalTable,
        cold: &ColdTableChunks,
        expected_generation: &Arc<crate::resident_storage::TableVersionData>,
        boundary: Index,
        chunk_authority_floor: Option<Index>,
    ) -> Result<u64, ExecuteError> {
        let stale = || {
            ExecuteError::Serialization(format!(
                "cold relation \"{}\" changed before index validation",
                table.name
            ))
        };
        let expected_signature = table
            .columns
            .iter()
            .map(|column| (column.name.clone(), column.ty))
            .collect::<Vec<_>>();
        if !Arc::ptr_eq(&cold.generation, expected_generation)
            || cold.column_signature != expected_signature
            || cold.build_copin_s > boundary
            || chunk_authority_floor.is_some_and(|floor| boundary < floor)
        {
            return Err(stale());
        }

        let expected_int4 = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        let expected_int8 = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Int8 | SqlType::Timestamp))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        let expected_b128 = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        let expected_bool = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Bool))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        let expected_text = table
            .columns
            .iter()
            .filter(|column| matches!(column.ty, SqlType::Text))
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();
        let column_ordinals = table
            .columns
            .iter()
            .enumerate()
            .map(|(ordinal, column)| (column.name.as_str(), ordinal))
            .collect::<BTreeMap<_, _>>();
        let bitmap_bytes = |capacity: usize| {
            capacity
                .div_ceil(32)
                .checked_mul(std::mem::size_of::<u32>())
        };

        let mut total_rows = 0u64;
        let mut total_payload_bytes = 0u64;
        for chunk in &cold.chunks {
            let row_count = usize::try_from(chunk.row_count).map_err(|_| stale())?;
            if chunk.payload_copin_s > boundary
                || chunk.snapshot.schema != table.schema
                || chunk.snapshot.table != table.name
                || chunk.snapshot.row_count != row_count
                || chunk.snapshot.capacity != row_count
                || chunk.snapshot.column_count != table.columns.len()
                || chunk
                    .snapshot
                    .resident_device_int4_columns
                    .iter()
                    .map(String::as_str)
                    .ne(expected_int4.iter().copied())
                || chunk
                    .snapshot
                    .resident_device_int8_columns
                    .iter()
                    .map(String::as_str)
                    .ne(expected_int8.iter().copied())
                || chunk
                    .snapshot
                    .resident_device_numeric_columns
                    .iter()
                    .map(String::as_str)
                    .ne(expected_b128.iter().copied())
                || chunk
                    .snapshot
                    .resident_device_bool_columns
                    .iter()
                    .map(|layout| layout.name.as_str())
                    .ne(expected_bool.iter().copied())
                || chunk
                    .snapshot
                    .resident_device_text_columns
                    .iter()
                    .map(|layout| layout.name.as_str())
                    .ne(expected_text.iter().copied())
                || (chunk_authority_floor.is_some() && chunk.entity_ids.len() != row_count)
            {
                return Err(stale());
            }

            let mut last_null_ordinal = None;
            for layout in &chunk.snapshot.resident_device_null_columns {
                let ordinal = column_ordinals
                    .get(layout.name.as_str())
                    .copied()
                    .ok_or_else(stale)?;
                if last_null_ordinal.is_some_and(|last| ordinal <= last) {
                    return Err(stale());
                }
                last_null_ordinal = Some(ordinal);
            }

            if read_cold_payload_u64(&chunk.payload, 0).map_err(|_| stale())? != chunk.row_count
                || chunk.snapshot.resident_bytes != chunk.payload.len() as u64
            {
                return Err(stale());
            }
            let capacity = chunk.snapshot.capacity;
            let mut cursor = 8usize
                .checked_add(
                    capacity
                        .checked_mul(expected_int4.len())
                        .and_then(|slots| slots.checked_mul(4))
                        .ok_or_else(stale)?,
                )
                .and_then(|cursor| {
                    capacity
                        .checked_mul(expected_int8.len())
                        .and_then(|slots| slots.checked_mul(8))
                        .and_then(|bytes| cursor.checked_add(bytes))
                })
                .and_then(|cursor| {
                    capacity
                        .checked_mul(expected_b128.len())
                        .and_then(|slots| slots.checked_mul(16))
                        .and_then(|bytes| cursor.checked_add(bytes))
                })
                .ok_or_else(stale)?;
            let bitmap_bytes = bitmap_bytes(capacity).ok_or_else(stale)?;
            for layout in &chunk.snapshot.resident_device_bool_columns {
                if usize::try_from(layout.bitmap_byte_offset).ok() != Some(cursor) {
                    return Err(stale());
                }
                cursor = cursor.checked_add(bitmap_bytes).ok_or_else(stale)?;
            }
            for layout in &chunk.snapshot.resident_device_null_columns {
                if usize::try_from(layout.bitmap_byte_offset).ok() != Some(cursor) {
                    return Err(stale());
                }
                cursor = cursor.checked_add(bitmap_bytes).ok_or_else(stale)?;
            }
            for layout in &chunk.snapshot.resident_device_text_columns {
                cursor = cursor
                    .checked_add((8usize.wrapping_sub(cursor % 8)) % 8)
                    .ok_or_else(stale)?;
                if usize::try_from(layout.offsets_byte_offset).ok() != Some(cursor) {
                    return Err(stale());
                }
                let offsets_bytes = row_count
                    .checked_add(1)
                    .and_then(|entries| entries.checked_mul(8))
                    .ok_or_else(stale)?;
                cursor = cursor.checked_add(offsets_bytes).ok_or_else(stale)?;
                if usize::try_from(layout.bytes_byte_offset).ok() != Some(cursor) {
                    return Err(stale());
                }
                let offsets_valid = validate_cold_text_offsets_bounded(
                    &chunk.payload,
                    usize::try_from(layout.offsets_byte_offset).map_err(|_| stale())?,
                    row_count.checked_add(1).ok_or_else(stale)?,
                    layout.bytes_len,
                )
                .map_err(|_| stale())?
                .0;
                if !offsets_valid {
                    return Err(stale());
                }
                cursor = cursor
                    .checked_add(usize::try_from(layout.bytes_len).map_err(|_| stale())?)
                    .ok_or_else(stale)?;
            }
            if cursor != chunk.payload.len()
                || chunk
                    .deleted_by
                    .as_ref()
                    .is_some_and(|sidecar| sidecar.len() != row_count.saturating_mul(8))
            {
                return Err(stale());
            }
            total_rows = total_rows.checked_add(chunk.row_count).ok_or_else(stale)?;
            total_payload_bytes = total_payload_bytes
                .checked_add(chunk.payload.len() as u64)
                .and_then(|bytes| {
                    chunk.deleted_by.as_ref().map_or(Some(bytes), |sidecar| {
                        bytes.checked_add(sidecar.len() as u64)
                    })
                })
                .ok_or_else(stale)?;
        }
        if total_payload_bytes != cold.total_payload_bytes {
            return Err(stale());
        }
        Ok(total_rows)
    }

    /// Validate a cold root selected as global reset authority without patching or falling back to
    /// another representation. Durable reset proofs must never describe a stale cache entry.
    pub(crate) fn global_cold_root_matches_reset_boundary(
        &self,
        table: &RelationalTable,
        cold: &ColdTableChunks,
        boundary: Index,
    ) -> bool {
        let current = self
            .read_state
            .mvcc
            .table_rows(&table.name)
            .generation_payload();
        let signature = table
            .columns
            .iter()
            .map(|column| (column.name.clone(), column.ty))
            .collect::<Vec<_>>();
        Arc::ptr_eq(&cold.generation, &current)
            && cold.column_signature == signature
            && match self
                .read_state
                .residency
                .chunk_authoritative_tables
                .load()
                .get(&table.name)
                .copied()
            {
                Some(freeze) => boundary >= freeze,
                None => boundary >= cold.build_copin_s,
            }
    }

    /// STRATA S-E.5: stage one chunk — build its transient payload (host) and enqueue the upload on a
    /// private copy stream (async when the driver supports it). The caller computes the PREVIOUSLY
    /// staged chunk next, so this upload overlaps that compute and the subsequent host staging.
    pub(super) fn stage_streaming_chunk(
        &self,
        table: &RelationalTable,
        chunk_rows: &[Vec<SqlValue>],
        chunk_range: (u64, u64),
        capture: &mut Option<ColdCacheBuilder>,
        gpu_id: u16,
    ) -> Result<StagedChunk, ()> {
        let (snapshot, pending, payload) = self
            .build_transient_relation_residency_async(table, chunk_rows, gpu_id)
            .map_err(|_| ())?;
        // The out-of-core proof: the ACTUAL transient device bytes for this chunk (fetch_max monotonic).
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(snapshot.resident_bytes, Ordering::Relaxed);
        // S-E.6: capture the built payload bytes for the cold tier (the upload already staged them
        // into pinned memory; keeping the Vec is zero extra copies). Above the spill threshold the
        // builder streams them to the unlinked spill file instead of holding RAM (S-E.6b).
        if let Some(builder) = capture {
            builder.push(
                payload,
                snapshot.clone(),
                chunk_rows.len() as u64,
                chunk_range,
            );
        }
        Ok(StagedChunk {
            snapshot,
            pending,
            row_count: chunk_rows.len() as u64,
            visibility: None,
        })
    }

    /// S-E.6b (audit LOW): evict a table's cold entry after a replay failure — a bad spill file
    /// (disk fault) would otherwise defer-thrash every future streaming read on the table; dropping
    /// the entry lets the next read rebuild it.
    pub(super) fn evict_streaming_cold(&self, table_name: &str) {
        // Audit H2: a CHUNK-AUTHORITATIVE table's entry is the record-of-truth for post-freeze
        // writes — fold-failure eviction must never remove it. A read decline fails loudly, and the
        // explicit RETIRE-002 repair boundary still needs the entry to replay the delta.
        if self.table_chunk_authoritative(table_name).is_some() {
            return;
        }
        let residency = &self.read_state.residency;
        let _publish = residency
            .streaming_cold_lock
            .lock()
            .expect("streaming cold-tier lock poisoned");
        let mut map = std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
        if map.remove(table_name).is_some() {
            residency.streaming_cold_chunks.store(Arc::new(map));
            self.purge_chunk_key_indexes_for_table(table_name);
            self.purge_chunk_key_blooms_for_table(table_name);
        }
    }

    /// S-E.6: stage one COLD chunk — re-upload the cached device payload bytes (async copy stream),
    /// with a fresh proof stamped onto the cached descriptor template. No decode, no assembly.
    pub(crate) fn stage_cold_chunk(
        &self,
        chunk: &ColdChunk,
        reader_copin_s: Index,
    ) -> Result<StagedChunk, ()> {
        self.stage_cold_chunk_on_gpu(chunk, reader_copin_s, chunk.snapshot.gpu_id)
    }

    pub(super) fn stage_cold_chunk_on_gpu(
        &self,
        chunk: &ColdChunk,
        reader_copin_s: Index,
        gpu_id: u16,
    ) -> Result<StagedChunk, ()> {
        let runtime = self.cuda_driver_probe_runtime();
        // RAM chunks borrow; spilled chunks positional-read from the unlinked file (an IO error is
        // a defer, never a wrong answer).
        let payload = chunk.payload.read()?;
        // P2: a sidecar-bearing chunk uploads payload + 8-aligned deleted_by sidecar as ONE device
        // buffer (the transient source is one allocation; `ResidentVisibility` addresses the
        // sidecar by ABSOLUTE offset). The concat is one host memcpy paid ONLY by delete-bearing
        // chunks — delete-free chunks keep the zero-copy borrow. The mask (`deleted_by >
        // read_txn_id`, signed s64) is ANDed in-kernel by the executor's mask VM.
        let (bytes, visibility) = match &chunk.deleted_by {
            None => (payload, None),
            Some(sidecar) => {
                let padded = payload.len().next_multiple_of(8);
                let mut buf = Vec::with_capacity(padded + sidecar.len());
                buf.extend_from_slice(&payload);
                buf.resize(padded, 0);
                buf.extend_from_slice(sidecar);
                (
                    std::borrow::Cow::Owned(buf),
                    Some(crate::engine_expr::ResidentVisibility {
                        read_txn_id: reader_copin_s as i64,
                        deleted_by_offset: Some(padded as u64),
                        created_by_offset: None,
                    }),
                )
            }
        };
        let pending = runtime
            .retain_device_memory_copy_async(gpu_id, &bytes)
            .map_err(|_| ())?;
        self.read_state
            .residency
            .streaming_fold_peak_chunk_bytes
            .fetch_max(chunk.snapshot.resident_bytes, Ordering::Relaxed);
        let mut snapshot = chunk.snapshot.clone();
        snapshot.gpu_id = gpu_id;
        snapshot.device_memory_proof = Some(pending.metadata().clone());
        Ok(StagedChunk {
            snapshot,
            pending,
            row_count: chunk.row_count,
            visibility,
        })
    }

    /// Stage a bounded selection from one transaction-pinned cold relation as immutable shard
    /// descriptors for the shared D2D unified-source builder. Transactional UNIQUE validation calls
    /// this with either one chunk or one cross-chunk pair, so payload, tombstone, recompaction, and
    /// GROUP BY scratch stay O(bounded batch) rather than O(table). Row values remain
    /// device-resident and the ordinary NULL/text/MVCC kernels remain the sole relational
    /// execution path.
    pub(crate) fn stage_cold_index_validation_shards(
        &self,
        table: &RelationalTable,
        cold: &ColdTableChunks,
        boundary: Index,
        gpu_id: u16,
        windows: &[ColdIndexValidationWindow],
        host_staging_limit: usize,
    ) -> Result<Vec<RelationalResidentShard>, ExecuteError> {
        let expected_signature = table
            .columns
            .iter()
            .map(|column| (column.name.clone(), column.ty))
            .collect::<Vec<_>>();
        if cold.column_signature != expected_signature || cold.build_copin_s > boundary {
            return Err(ExecuteError::Serialization(format!(
                "cold relation \"{}\" changed before index validation",
                table.name
            )));
        }
        if windows.is_empty()
            || windows.len() > 4096
            || windows.windows(2).any(|pair| {
                let left = (pair[0].chunk_ordinal, pair[0].row_start);
                let right = (pair[1].chunk_ordinal, pair[1].row_start);
                left >= right
                    || (pair[0].chunk_ordinal == pair[1].chunk_ordinal
                        && pair[0].row_start.saturating_add(pair[0].row_count) > pair[1].row_start)
            })
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "cold index validation requires a nonempty bounded ordered window set".to_string(),
            )));
        }
        #[cfg(test)]
        self.read_state
            .residency
            .cold_index_validation_peak_staged_chunks
            .fetch_max(windows.len() as u64, Ordering::Relaxed);
        let descriptor_bytes = windows.iter().try_fold(0usize, |bytes, window| {
            let chunk = cold.chunks.get(window.chunk_ordinal).ok_or_else(|| {
                ExecuteError::Serialization(format!(
                    "cold relation \"{}\" lost validation chunk {}",
                    table.name, window.chunk_ordinal
                ))
            })?;
            let window_bytes = cold_validation_descriptor_bytes(table, &chunk.snapshot)
                .ok_or_else(|| {
                    cold_index_host_staging_exhausted(&table.name, usize::MAX, host_staging_limit)
                })?;
            bytes.checked_add(window_bytes).ok_or_else(|| {
                cold_index_host_staging_exhausted(&table.name, usize::MAX, host_staging_limit)
            })
        })?;
        if descriptor_bytes >= host_staging_limit {
            return Err(cold_index_host_staging_exhausted(
                &table.name,
                descriptor_bytes,
                host_staging_limit,
            ));
        }
        let payload_host_limit = host_staging_limit - descriptor_bytes;
        let runtime = self.cuda_driver_probe_runtime();
        let mut row_start = 0usize;
        let mut shards = Vec::new();
        shards.try_reserve_exact(windows.len()).map_err(|_| {
            cold_index_host_staging_exhausted(&table.name, descriptor_bytes, host_staging_limit)
        })?;
        let point_route_generation = Arc::new(());
        for (shard_ordinal, &window) in windows.iter().enumerate() {
            let chunk_ordinal = window.chunk_ordinal;
            let chunk = cold
                .chunks
                .get(chunk_ordinal)
                .filter(|chunk| chunk.row_count != 0)
                .ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "cold relation \"{}\" lost validation chunk {chunk_ordinal}",
                        table.name
                    ))
                })?;
            if chunk.payload_copin_s > boundary
                || chunk.snapshot.schema != table.schema
                || chunk.snapshot.table != table.name
                || chunk.snapshot.row_count != chunk.row_count as usize
                || chunk.snapshot.capacity < chunk.snapshot.row_count
            {
                return Err(ExecuteError::Serialization(format!(
                    "cold relation \"{}\" has a stale or torn chunk descriptor",
                    table.name
                )));
            }
            let ColdIndexValidationPayload {
                resident_bytes,
                bool_columns,
                text_columns,
                null_columns,
                payload,
                deleted_by,
                host_staging_bytes,
            } = slice_cold_index_validation_window(chunk, window, payload_host_limit)?;
            let stage_host_staging_bytes = descriptor_bytes
                .checked_add(host_staging_bytes)
                .ok_or_else(|| {
                    cold_index_host_staging_exhausted(&table.name, usize::MAX, host_staging_limit)
                })?;
            #[cfg(not(test))]
            let _ = stage_host_staging_bytes;
            #[cfg(test)]
            self.read_state
                .residency
                .cold_index_validation_peak_host_staging_bytes
                .fetch_max(
                    stage_host_staging_bytes as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            let memory = Arc::new(
                runtime
                    .retain_device_memory_copy_scoped(gpu_id, &payload)
                    .map_err(|error| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "cold relation \"{}\" GPU staging failed: {error}",
                            table.name
                        )))
                    })?,
            );
            let deleted_by_region = deleted_by
                .as_ref()
                .map(|sidecar| {
                    runtime
                        .retain_device_memory_copy_scoped(gpu_id, sidecar)
                        .map(Arc::new)
                        .map_err(|error| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "cold relation \"{}\" tombstone staging failed: {error}",
                                table.name
                            )))
                        })
                })
                .transpose()?;
            let row_count = window.row_count;
            let shard_id = u32::try_from(shard_ordinal).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "cold index-validation shard count exceeds device framing".to_string(),
                ))
            })?;
            let allocated_bytes = memory.metadata().allocated_bytes;
            let int4_stats = try_clone_cold_int4_stats(
                &chunk.snapshot.resident_device_int4_column_stats,
                &table.name,
                stage_host_staging_bytes,
                host_staging_limit,
            )?;
            let int4_columns = try_clone_cold_strings(
                &chunk.snapshot.resident_device_int4_columns,
                &table.name,
                stage_host_staging_bytes,
                host_staging_limit,
            )?;
            let int8_columns = try_clone_cold_strings(
                &chunk.snapshot.resident_device_int8_columns,
                &table.name,
                stage_host_staging_bytes,
                host_staging_limit,
            )?;
            let numeric_columns = try_clone_cold_strings(
                &chunk.snapshot.resident_device_numeric_columns,
                &table.name,
                stage_host_staging_bytes,
                host_staging_limit,
            )?;
            let schema = try_clone_cold_string(
                &table.schema,
                &table.name,
                stage_host_staging_bytes,
                host_staging_limit,
            )?;
            let table_name = try_clone_cold_string(
                &table.name,
                &table.name,
                stage_host_staging_bytes,
                host_staging_limit,
            )?;
            let device_memory_proof = Some(try_clone_cold_memory_proof(
                memory.metadata(),
                &table.name,
                stage_host_staging_bytes,
                host_staging_limit,
            )?);
            shards.push(RelationalResidentShard {
                shard_id,
                row_start,
                row_count,
                history_floor_index: 0,
                capacity: row_count,
                int4_appendable: false,
                resident_device_int4_column_stats: int4_stats,
                resident_bytes,
                allocated_bytes,
                count_header_byte_offset: 0,
                resident_device_int4_columns: int4_columns,
                resident_device_int8_columns: int8_columns,
                resident_device_numeric_columns: numeric_columns,
                resident_device_bool_columns: bool_columns,
                resident_device_text_columns: text_columns,
                resident_device_null_columns: null_columns,
                gpu_id,
                schema,
                table: table_name,
                point_route_generation: Arc::clone(&point_route_generation),
                device_memory_proof,
                invalidated_by_txn_id: None,
                invalidated_at_index: None,
                invalidated_by_memory_pressure: false,
                memory_pressure_active: false,
                device_memory: Some(memory),
                deleted_by_region,
                created_by_region: None,
                row_id_region: None,
                max_created_by: 0,
            });
            row_start = row_start.checked_add(row_count).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "cold index-validation row offset overflowed".to_string(),
                ))
            })?;
        }
        Ok(shards)
    }

    /// 6c-1: rebuild the visible rows of ONE effective TupleId range into cold chunks (payload +
    /// descriptor, NO upload — replays stamp a fresh proof). Splits at the chunk byte target. The
    /// decode/build here is the SAME staging the scan path performs, bounded to the dirty range —
    /// the O(delta) win (charter: the staging upload carve-out; the registered scan-build debt
    /// shrinks from O(table)/write to O(delta)/write).
    #[allow(clippy::too_many_arguments)]
    fn build_cold_chunks_for_range(
        &self,
        table: &RelationalTable,
        store: &crate::resident_storage::TableVersionData,
        copin_s: Index,
        eff_lo: u64,
        eff_hi: u64,
        chunk_target_bytes: u64,
        // F1 (6c-1 audit): rebuilt chunks accumulate through this SPILL-AWARE builder — a big
        // tail / wide dirty range streams to the unlinked spill file above the threshold instead
        // of materializing all payloads in host RAM (the same out-of-core bound the scan-build
        // has). The builder's chunks stay in ascending range order across calls.
        builder: &mut ColdCacheBuilder,
    ) -> Result<(), ()> {
        let visibility = StorageVisibility {
            read_txn_id: copin_s,
        };
        let versions = store
            .rows
            .visible_versions_in_range(visibility, eff_lo, eff_hi)
            .map_err(|_| ())?;
        let column_types: Vec<SqlType> = table.columns.iter().map(|column| column.ty).collect();
        let mut rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut bytes: u64 = 0;
        let mut range: Option<(u64, u64)> = None;
        let prefix = relational_key_prefix(&table.name);
        let flush = |rows: &mut Vec<Vec<SqlValue>>,
                     range: &mut Option<(u64, u64)>,
                     builder: &mut ColdCacheBuilder|
         -> Result<(), ()> {
            if rows.is_empty() {
                return Ok(());
            }
            let (snapshot, payload) = self.build_cold_payload(table, rows)?;
            builder.push(
                payload,
                snapshot,
                rows.len() as u64,
                range.take().expect("non-empty chunk has a range"),
            );
            rows.clear();
            Ok(())
        };
        for version in versions {
            if !version.key.starts_with(&prefix) {
                continue;
            }
            let decoded = decode_relational_row(&version.value, &table.columns).map_err(|_| ())?;
            bytes = bytes.saturating_add(chunk_row_device_bytes(&decoded, &column_types));
            range = Some(match range {
                None => (version.tuple_id, version.tuple_id),
                Some((lo, _)) => (lo, version.tuple_id),
            });
            rows.push(decoded);
            if bytes >= chunk_target_bytes {
                flush(&mut rows, &mut range, builder)?;
                bytes = 0;
            }
        }
        flush(&mut rows, &mut range, builder)?;
        if builder.poisoned {
            return Err(());
        }
        Ok(())
    }

    /// The payload + descriptor for a cold chunk WITHOUT uploading (proof = None; stage_cold_chunk
    /// stamps a fresh proof per replay). Mirrors `build_transient_relation_residency`'s descriptor.
    fn build_cold_payload(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
    ) -> Result<(RelationalResidencySnapshot, Vec<u8>), ()> {
        // 6c-3 (audit F4 adopted): payload + descriptor ONLY — no throwaway upload. The replay
        // (stage_cold_chunk) stamps a fresh proof when it actually uploads.
        self.build_transient_relation_payload_only(table, rows)
            .map_err(|_| ())
    }

    /// P2: classify one changed chain as a PURE DELETE of a payload-visible row — the only
    /// change tolerated by the sidecar STAMP downgrade. Returns the deleting commit seq when the
    /// old and new chains are identical EXCEPT exactly one version (a payload row: `created_by <=
    /// payload_copin_s`, previously live) gained a `deleted_by` stamp. Anything else — tail
    /// growth, value edits, same-id version appends, vanished chains, double deletes — returns
    /// `None` and the chunk keeps the 6c-1 REBUILD arm (correctness backstop; never a wrong
    /// answer). Control-plane version-METADATA comparison only (charter: no row values computed,
    /// the equality checks are structural).
    fn classify_pure_delete(
        old: &[gpu_db_storage::TupleVersion],
        new: &[gpu_db_storage::TupleVersion],
        payload_copin_s: Index,
    ) -> Option<Index> {
        if old.len() != new.len() {
            return None;
        }
        let mut stamp: Option<Index> = None;
        for (o, n) in old.iter().zip(new.iter()) {
            if o == n {
                continue;
            }
            if stamp.is_some() {
                return None; // more than one changed version
            }
            if o.tuple_id != n.tuple_id
                || o.key != n.key
                || o.value != n.value
                || o.created_by != n.created_by
            {
                return None;
            }
            if o.deleted_by.is_some() || n.deleted_by.is_none() {
                return None;
            }
            if o.created_by > payload_copin_s {
                return None; // not a payload row (defensive: interior inserts cannot happen)
            }
            stamp = n.deleted_by;
        }
        stamp
    }

    /// 6c-1 — CHUNK-GRANULAR DELTA PATCHING (deletes the whole-table invalidation): a stale cold
    /// entry (generation mismatch = a write happened) is PATCHED, not discarded. The changed
    /// TupleIds come from the O(delta) COW-chain diff (`changed_tuple_ids` — untouched subtrees are
    /// pointer-equal); each maps to its chunk through the EFFECTIVE range tiling (chunk i owns
    /// (prev.hi, hi]; ids beyond the last chunk are the TAIL — the rollover pattern). Untouched
    /// chunks REUSE their bytes verbatim (chain identity + the old entry's settled boundary make
    /// their visible sets boundary-invariant); dirty ranges + the tail REBUILD at the patching
    /// reader's boundary. The patched entry re-installs under the SAME settled-boundary commit-lock
    /// proof as a fresh build (S-E.6a). Returns the landed entry, or None (caller evicts + scans).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn patch_streaming_cold(
        &self,
        table_name: &str,
        table: &RelationalTable,
        stale: &Arc<ColdTableChunks>,
        current: &Arc<crate::resident_storage::TableVersionData>,
        copin_s: Index,
        chunk_target_bytes: u64,
        commit_lock_held: bool,
    ) -> Option<Arc<ColdTableChunks>> {
        // The ALTER guard: a shape-changing DDL republished the store too — cached payload layouts
        // would be reused with the WRONG column shape. Signature inequality -> evict.
        let signature: Vec<(String, SqlType)> = table
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.ty))
            .collect();
        if signature != stale.column_signature || stale.chunk_target_bytes != chunk_target_bytes {
            return None;
        }
        let changed = stale.generation.rows.changed_tuple_ids(&current.rows);
        // Map changed ids to dirty chunks via the effective tiling; ids past the last hi = tail.
        let mut dirty = vec![false; stale.chunks.len()];
        let his: Vec<u64> = stale.chunks.iter().map(|c| c.tuple_range.1).collect();
        let last_hi = his.last().copied().unwrap_or(0);
        let mut tail_dirty = stale.chunks.is_empty();
        let mut per_chunk_ids: Vec<Vec<u64>> = vec![Vec::new(); stale.chunks.len()];
        for id in &changed {
            if *id > last_hi {
                tail_dirty = true;
                continue;
            }
            let idx = his.partition_point(|hi| *hi < *id);
            dirty[idx] = true;
            per_chunk_ids[idx].push(*id);
        }
        // P2 — the SIDECAR STAMP DOWNGRADE: a dirty chunk whose every change is a PURE DELETE of
        // one of its payload rows keeps its bytes and gains tombstone stamps (an O(8B x rows)
        // sidecar COW) instead of the O(chunk) decode+rebuild. The row's SLOT is its rank among
        // the ids visible at the chunk's OWN payload boundary within the chunk's effective range
        // (scan order IS TupleId order; the walk anchors at `payload_copin_s`, never the entry
        // boundary — a stamped row stays IN the payload, masked in-kernel at replay). Any
        // classification failure keeps the rebuild arm.
        let mut stamps: Vec<Option<Vec<(usize, Index)>>> = vec![None; stale.chunks.len()];
        'downgrade: for i in 0..stale.chunks.len() {
            if !dirty[i] || per_chunk_ids[i].is_empty() {
                continue;
            }
            let chunk = &stale.chunks[i];
            if chunk.row_count == 0 {
                continue;
            }
            let eff_lo_i = if i == 0 {
                0
            } else {
                his[i - 1].saturating_add(1)
            };
            let payload_vis = StorageVisibility {
                read_txn_id: chunk.payload_copin_s,
            };
            let mut list: Vec<(usize, Index)> = Vec::with_capacity(per_chunk_ids[i].len());
            for id in &per_chunk_ids[i] {
                let (Some(old_chain), Some(new_chain)) =
                    (stale.generation.rows.chain(*id), current.rows.chain(*id))
                else {
                    continue 'downgrade;
                };
                let Some(stamp) =
                    Self::classify_pure_delete(old_chain, new_chain, chunk.payload_copin_s)
                else {
                    continue 'downgrade;
                };
                let Ok(slot) = stale.generation.rows.visible_count_in_range(
                    payload_vis,
                    eff_lo_i,
                    id.saturating_sub(1),
                ) else {
                    continue 'downgrade;
                };
                if slot >= chunk.row_count as usize {
                    continue 'downgrade; // rank disagrees with the payload — rebuild (defensive)
                }
                list.push((slot, stamp));
            }
            stamps[i] = Some(list);
            dirty[i] = false;
        }
        // F2 (6c-1 audit — fragmentation cap): when the tail grows, COALESCE a trailing RUNT chunk
        // (under half the target) into the tail rebuild — insert/read ping-pong would otherwise
        // accrete one tiny chunk per write, degrading every later replay. Each patch absorbs the
        // runt, so at most one lives at any time.
        if tail_dirty && !stale.chunks.is_empty() {
            let last = stale.chunks.len() - 1;
            let last_bytes = match &stale.chunks[last].payload {
                ColdPayload::Ram(bytes) => bytes.len() as u64,
                ColdPayload::Spilled { len, .. } => *len as u64,
            };
            if last_bytes < chunk_target_bytes / 2 {
                dirty[last] = true;
                stamps[last] = None; // the tail absorption needs the rebuild arm
            }
        }
        // Rebuild dirty ranges through ONE spill-aware builder (F1: rebuilt payloads stream to the
        // spill file above the threshold — never unbounded host RAM), then MERGE with the reused
        // chunks by ascending range (both sequences are ascending; control-plane assembly).
        let mut rebuild = ColdCacheBuilder {
            generation: Arc::clone(current),
            build_copin_s: copin_s,
            chunk_target_bytes,
            total_payload_bytes: 0,
            column_signature: signature.clone(),
            chunks: Vec::new(),
            spill: None,
            poisoned: false,
        };
        let mut reused: Vec<ColdChunk> = Vec::new();
        let mut stamped_rows: u64 = 0;
        let mut eff_lo: u64 = 0;
        for (i, chunk) in stale.chunks.iter().enumerate() {
            let eff_hi = chunk.tuple_range.1;
            // A dirty chunk's range REBUILDS; when the runt-coalesce marked the LAST chunk dirty,
            // extend its rebuild into the tail in one scan (eff_hi = MAX below handles it).
            let rebuild_hi = if dirty[i] && i == stale.chunks.len() - 1 && tail_dirty {
                u64::MAX
            } else {
                eff_hi
            };
            if dirty[i] {
                self.build_cold_chunks_for_range(
                    table,
                    current,
                    copin_s,
                    eff_lo,
                    rebuild_hi,
                    chunk_target_bytes,
                    &mut rebuild,
                )
                .ok()?;
                self.read_state
                    .residency
                    .streaming_cold_chunks_rebuilt
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                // P2: a stamp-downgraded chunk reuses its payload and COWs its sidecar (get-or-
                // materialize at the 0x7F live fill — a delete-free chunk pays only here, on its
                // FIRST delete); a plain reuse carries both through unchanged.
                let deleted_by = match &stamps[i] {
                    Some(list) if !list.is_empty() => {
                        let mut bytes = match &chunk.deleted_by {
                            Some(existing) => existing.as_ref().clone(),
                            None => {
                                vec![COLD_DELETED_BY_LIVE_FILL_BYTE; (chunk.row_count as usize) * 8]
                            }
                        };
                        for (slot, stamp) in list {
                            bytes[slot * 8..slot * 8 + 8].copy_from_slice(&stamp.to_le_bytes());
                        }
                        stamped_rows += list.len() as u64;
                        Some(Arc::new(bytes))
                    }
                    _ => chunk.deleted_by.as_ref().map(Arc::clone),
                };
                reused.push(ColdChunk {
                    chunk_id: chunk.chunk_id,
                    payload: match &chunk.payload {
                        ColdPayload::Ram(bytes) => ColdPayload::Ram(Arc::clone(bytes)),
                        ColdPayload::Spilled { file, offset, len } => ColdPayload::Spilled {
                            file: Arc::clone(file),
                            offset: *offset,
                            len: *len,
                        },
                    },
                    snapshot: chunk.snapshot.clone(),
                    row_count: chunk.row_count,
                    entity_ids: Arc::clone(&chunk.entity_ids),
                    tuple_range: chunk.tuple_range,
                    // P2: reuse preserves the payload's OWN boundary (stamps do NOT advance it).
                    payload_copin_s: chunk.payload_copin_s,
                    deleted_by,
                });
            }
            eff_lo = eff_hi.saturating_add(1);
        }
        // The tail (unless the runt-coalesce already extended the last rebuild through MAX).
        let tail_absorbed = tail_dirty && !stale.chunks.is_empty() && dirty[stale.chunks.len() - 1];
        if tail_dirty && !tail_absorbed {
            self.build_cold_chunks_for_range(
                table,
                current,
                copin_s,
                last_hi.saturating_add(1),
                u64::MAX,
                chunk_target_bytes,
                &mut rebuild,
            )
            .ok()?;
        }
        // Merge reused + rebuilt by ascending range start (both already ascending).
        let mut chunks: Vec<ColdChunk> = Vec::with_capacity(reused.len() + rebuild.chunks.len());
        {
            let mut a = reused.into_iter().peekable();
            let mut b = rebuild.chunks.into_iter().peekable();
            loop {
                match (a.peek(), b.peek()) {
                    (Some(x), Some(y)) => {
                        if x.tuple_range.0 <= y.tuple_range.0 {
                            chunks.push(a.next().expect("peeked"));
                        } else {
                            chunks.push(b.next().expect("peeked"));
                        }
                    }
                    (Some(_), None) => chunks.push(a.next().expect("peeked")),
                    (None, Some(_)) => chunks.push(b.next().expect("peeked")),
                    (None, None) => break,
                }
            }
        }
        let total_payload_bytes: u64 = chunks
            .iter()
            .map(|c| {
                let payload = match &c.payload {
                    ColdPayload::Ram(bytes) => bytes.len() as u64,
                    ColdPayload::Spilled { len, .. } => *len as u64,
                };
                // P2: sidecars count against the cap class too (they are held host bytes).
                payload + c.deleted_by.as_ref().map_or(0, |b| b.len() as u64)
            })
            .sum();
        let builder = ColdCacheBuilder {
            generation: Arc::clone(current),
            build_copin_s: copin_s,
            chunk_target_bytes,
            total_payload_bytes,
            column_signature: signature,
            chunks,
            spill: None,
            poisoned: false,
        };
        if !self.install_streaming_cold_inner(table_name, builder, true, commit_lock_held) {
            return None;
        }
        self.read_state
            .residency
            .streaming_cold_patches
            .fetch_add(1, Ordering::Relaxed);
        if stamped_rows > 0 {
            self.read_state
                .residency
                .streaming_cold_stamps
                .fetch_add(stamped_rows, Ordering::Relaxed);
        }
        self.read_state
            .residency
            .streaming_cold_chunks
            .load()
            .get(table_name)
            .cloned()
    }

    /// S-E.6: the table's valid cold-tier chunks, or `None` (miss -> the caller scans + captures).
    /// A hit requires the SAME tuple-store generation (pointer equality — see [`ColdTableChunks`]),
    /// the same chunk target, AND `copin_s >= build_copin_s` (the boundary-invariance condition:
    /// the install proved no stamp exceeds the build boundary, so every boundary at-or-above it
    /// sees the identical set — audit F1). A GENERATION-mismatched entry is EVICTED here (audit
    /// F3: a stale entry would otherwise pin the superseded TableVersionData until the next
    /// install). `streaming_cold_hits` counts validity-passed ATTEMPTS (the fold may still defer
    /// on a later chunk — audit F4).
    pub(super) fn load_streaming_cold(
        &self,
        table_name: &str,
        table: &RelationalTable,
        chunk_target_bytes: u64,
        copin_s: Index,
    ) -> Option<Arc<ColdTableChunks>> {
        let cold = self.read_streaming_cold_chunks().get(table_name).cloned()?;
        let transaction_scoped = self.current_transaction_read_snapshot().is_some();
        let current = if transaction_scoped {
            self.read_table_rows_at(table_name, copin_s)
                .generation_payload()
        } else {
            self.read_state
                .mvcc
                .table_rows(table_name)
                .generation_payload()
        };
        if !Arc::ptr_eq(&cold.generation, &current) {
            // A retained transaction may only consume its captured/private generation. Patching
            // would rebind it to current global state and mutate the global cold cache.
            if transaction_scoped {
                return None;
            }
            // 6c-1: the table was written — PATCH the entry (rebuild only the dirty chunks + tail,
            // O(delta)) instead of discarding it. A patch that cannot apply (ALTER'd shape, install
            // race, IO error) falls through to the evict arm; the next read scans + rebuilds.
            if let Some(patched) = self.patch_streaming_cold(
                table_name,
                table,
                &cold,
                &current,
                copin_s,
                chunk_target_bytes,
                false,
            ) {
                self.read_state
                    .residency
                    .streaming_cold_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Some(patched);
            }
            // The table was written: drop the stale entry (and its pinned old generation) now.
            let residency = &self.read_state.residency;
            let _publish = residency
                .streaming_cold_lock
                .lock()
                .expect("streaming cold-tier lock poisoned");
            let mut map =
                std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
            // Re-check under the lock (a concurrent rebuild may have installed a FRESH entry).
            if let Some(entry) = map.get(table_name) {
                if !Arc::ptr_eq(&entry.generation, &current) {
                    map.remove(table_name);
                    residency.streaming_cold_chunks.store(Arc::new(map));
                }
            }
            return None;
        }
        let class_freeze = self.table_chunk_authoritative(table_name);
        // Cache tiling follows the caller's current working-set target, but a chunk-authoritative
        // table's tiling is physical record-of-truth state. A later budget change cannot turn that
        // authority into a cache miss (the reclaimed tuple store has no alternate rows); explicit
        // class rechunking is the only operation allowed to replace its geometry.
        if class_freeze.is_none() && cold.chunk_target_bytes != chunk_target_bytes {
            return None;
        }
        // P4-3 — THE BORN GATE (design review C3): a CLASS table's entry boundary advances with
        // every tail append, so `copin_s >= build` would MISS any reader pinned below the latest
        // write and thrash de-auth. Class hits require only `copin_s >= the FREEZE boundary`
        // (below it the frozen chains serve exactly); the replay arms skip chunks BORN LATER
        // (`payload_copin_s > copin_s`) and the sidecar mask handles deletes — exact MVCC per
        // reader. Non-class entries keep the strict boundary rule (their chunks are rebuilt at
        // the entry boundary; no per-chunk born discipline exists for them).
        match class_freeze {
            Some(freeze) => {
                if copin_s < freeze {
                    return None;
                }
            }
            None => {
                if copin_s < cold.build_copin_s {
                    return None;
                }
            }
        }
        self.read_state
            .residency
            .streaming_cold_hits
            .fetch_add(1, Ordering::Relaxed);
        Some(cold)
    }

    /// S-E.6: install a completed scan's captured chunks under the COMMIT LOCK for serialized cache
    /// publication. Lock-free intent lanes may still advance `committed_seq`; the settled-boundary proof
    /// is strict generation identity plus `committed_seq() == build_copin_s`, so a racing frontier bump
    /// safely discards the install. A matching generation contains no stamp above the build boundary,
    /// making the captured set boundary-invariant for every reader at or above it. Any commit since the
    /// bind (even to another table) discards the install
    /// (conservative; caches build in the read-mostly phases they exist for). A mid-commit
    /// internal read skips installing entirely (the lock is already held by this thread — the
    /// `rehydrate_elided_serialized` pattern). CAP policy (audit F2): an entry alone over the cap
    /// never installs (rebuild-then-clear thrash); a combined breach evicts the OTHER entries.
    pub(super) fn install_streaming_cold(
        &self,
        table_name: &str,
        builder: ColdCacheBuilder,
    ) -> bool {
        self.install_streaming_cold_inner(table_name, builder, false, false)
    }

    /// `is_patch` keeps the BUILD counter honest (a patch re-install is not a fresh build — audit
    /// 6c-1 F3); everything else is identical.
    pub(super) fn install_streaming_cold_inner(
        &self,
        table_name: &str,
        builder: ColdCacheBuilder,
        is_patch: bool,
        // 6c-3: the caller IS the serialized committer (both engine_commit hook sites hold the
        // commit mutex — one with the internal-read flag UNSET, so inference would deadlock;
        // explicit beats inference). NOTE (audit): committed_seq is NOT frozen under this mutex —
        // intent lanes publish it LOCK-FREE off this path — the actual safety is (a) the STRICT
        // EQUALITY guard below (a concurrent bump FAILS the install — a safe miss, never a
        // higher-stamp pass), (b) generation ptr identity (every write COW-publishes a fresh Arc),
        // and (c) per-read visibility at replay. Never weaken the generation check on a
        // frozen-seq assumption.
        commit_lock_held: bool,
    ) -> bool {
        // A spill IO error poisoned the capture: the chunk list is incomplete — never install it.
        if builder.poisoned {
            return false;
        }
        // 6c-1: a PATCHED entry can mix reused Spilled chunks with rebuilt Ram ones — class by the
        // chunks themselves, not the builder's own spill stream.
        let spilled = builder.spill.is_some()
            || builder
                .chunks
                .iter()
                .any(|c| matches!(c.payload, ColdPayload::Spilled { .. }));
        let class_cap = if spilled {
            STREAMING_COLD_DISK_CAP_BYTES
        } else {
            STREAMING_COLD_CAP_BYTES
        };
        if builder.total_payload_bytes > class_cap {
            return false;
        }
        let _commit_guard = if commit_lock_held {
            None
        } else {
            if self.mvcc_read_skips_leader_check() {
                // Mid-commit INTERNAL READ (not our hook): acquiring the lock would self-deadlock
                // and the boundary is mid-mutation — skip installing (the read path rebuilds).
                return false;
            }
            Some(self.commit_state())
        };
        // A chunk-authoritative entry is the table's representation of record and carries stable
        // entity IDs plus post-freeze sidecars. A general scan capture (including one produced by
        // a transaction-private read) has cache-only descriptors and empty entity IDs; allowing it
        // to replace the class entry would create an ABA generation with the right payload pointer
        // but no DML identity authority. The commit lock above serializes this check with class
        // entry/exit. Class mutation and compaction publish through `install_streaming_cold_class`.
        if self.table_chunk_authoritative(table_name).is_some() {
            return false;
        }
        let current = self
            .read_state
            .mvcc
            .table_rows(table_name)
            .generation_payload();
        if !Arc::ptr_eq(&builder.generation, &current)
            || self.committed_seq() != builder.build_copin_s
        {
            return false;
        }
        let entry = Arc::new(ColdTableChunks {
            generation: builder.generation,
            column_signature: builder.column_signature,
            build_copin_s: builder.build_copin_s,
            chunk_target_bytes: builder.chunk_target_bytes,
            total_payload_bytes: builder.total_payload_bytes,
            spilled,
            entry_epoch: COLD_ENTRY_EPOCH.fetch_add(1, Ordering::Relaxed),
            chunks: builder.chunks,
        });
        let live_chunk_ids: std::collections::BTreeSet<u64> =
            entry.chunks.iter().map(|chunk| chunk.chunk_id).collect();
        let residency = &self.read_state.residency;
        let _publish = residency
            .streaming_cold_lock
            .lock()
            .expect("streaming cold-tier lock poisoned");
        let mut map = std::collections::BTreeMap::clone(&residency.streaming_cold_chunks.load());
        map.insert(table_name.to_string(), Arc::clone(&entry));
        // Per-class caps (RAM vs spilled/DISK): a breach evicts the OTHER entries of that class.
        let class_total: u64 = map
            .values()
            .filter(|c| c.spilled == spilled)
            .map(|c| c.total_payload_bytes)
            .sum();
        if class_total > class_cap {
            let kept = map.remove(table_name).expect("just inserted");
            // P4-2b: a CHUNK-AUTHORITATIVE table's entry is its representation-of-record — cap
            // pressure must never evict it (the frozen store lacks the post-freeze writes).
            let protected = self.read_state.residency.chunk_authoritative_tables.load();
            map.retain(|name, c| c.spilled != spilled || protected.contains_key(name));
            map.insert(table_name.to_string(), kept);
        }
        if !is_patch {
            residency
                .streaming_cold_builds
                .fetch_add(1, Ordering::Relaxed);
        }
        if spilled {
            residency
                .streaming_cold_spills
                .fetch_add(1, Ordering::Relaxed);
        }
        residency.streaming_cold_chunks.store(Arc::new(map));
        drop(_publish);
        self.purge_stale_chunk_key_candidates(table_name, &live_chunk_ids);
        drop(_commit_guard);
        // Key candidate structures are primed only after releasing the global commit mutex. This
        // is load-bearing for spilled captures: staging may perform positional NVMe reads, which
        // must never occur in the later class-entry hook under the commit lock.
        if !commit_lock_held {
            self.prime_chunk_key_candidates(table_name, &entry);
        }
        true
    }
}
