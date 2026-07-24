use super::*;

const COLD_INDEX_DESCRIPTOR_FIXED_OVERHEAD_BYTES: usize = 1024;
const COLD_INDEX_DESCRIPTOR_MAX_DEVICE_NAME_BYTES: usize = 512;
pub(super) const COLD_TEXT_OFFSET_SCAN_BLOCK_BYTES: usize = 64 * 1024;

pub(super) fn cold_validation_descriptor_bytes(
    table: &RelationalTable,
    snapshot: &RelationalResidencySnapshot,
) -> Option<usize> {
    // This fixed envelope covers the shared Arc control blocks, CUDA-memory/proof structs, and a
    // bounded runtime device name. Every variable-size Vec/String below is counted separately.
    let mut bytes = std::mem::size_of::<RelationalResidentShard>()
        .checked_add(COLD_INDEX_DESCRIPTOR_FIXED_OVERHEAD_BYTES)?;
    let mut add_strings = |values: &[String]| -> Option<()> {
        bytes = bytes.checked_add(values.len().checked_mul(std::mem::size_of::<String>())?)?;
        for value in values {
            bytes = bytes.checked_add(value.len())?;
        }
        Some(())
    };
    add_strings(&snapshot.resident_device_int4_columns)?;
    add_strings(&snapshot.resident_device_int8_columns)?;
    add_strings(&snapshot.resident_device_numeric_columns)?;
    bytes = bytes.checked_add(
        snapshot
            .resident_device_int4_column_stats
            .len()
            .checked_mul(std::mem::size_of::<ResidentDeviceInt4ColumnStats>())?,
    )?;
    for stats in &snapshot.resident_device_int4_column_stats {
        bytes = bytes.checked_add(stats.name.len())?;
    }
    bytes = bytes.checked_add(
        snapshot
            .resident_device_bool_columns
            .len()
            .checked_mul(std::mem::size_of::<ResidentDeviceBoolColumnLayout>())?,
    )?;
    for layout in &snapshot.resident_device_bool_columns {
        bytes = bytes.checked_add(layout.name.len())?;
    }
    bytes = bytes.checked_add(
        snapshot
            .resident_device_text_columns
            .len()
            .checked_mul(std::mem::size_of::<ResidentDeviceTextColumnLayout>())?,
    )?;
    for layout in &snapshot.resident_device_text_columns {
        bytes = bytes.checked_add(layout.name.len())?;
    }
    bytes = bytes.checked_add(
        snapshot
            .resident_device_null_columns
            .len()
            .checked_mul(std::mem::size_of::<ResidentDeviceNullBitmapLayout>())?,
    )?;
    for layout in &snapshot.resident_device_null_columns {
        bytes = bytes.checked_add(layout.name.len())?;
    }
    bytes
        .checked_add(table.schema.len())?
        .checked_add(table.name.len())
}

pub(super) fn try_clone_cold_string(
    value: &str,
    table: &str,
    required: usize,
    limit: usize,
) -> Result<String, ExecuteError> {
    let mut clone = String::new();
    clone
        .try_reserve_exact(value.len())
        .map_err(|_| cold_index_host_staging_exhausted(table, required, limit))?;
    clone.push_str(value);
    Ok(clone)
}

pub(super) fn try_clone_cold_strings(
    values: &[String],
    table: &str,
    required: usize,
    limit: usize,
) -> Result<Vec<String>, ExecuteError> {
    let mut clones = Vec::new();
    clones
        .try_reserve_exact(values.len())
        .map_err(|_| cold_index_host_staging_exhausted(table, required, limit))?;
    for value in values {
        clones.push(try_clone_cold_string(value, table, required, limit)?);
    }
    Ok(clones)
}

pub(super) fn try_clone_cold_int4_stats(
    values: &[ResidentDeviceInt4ColumnStats],
    table: &str,
    required: usize,
    limit: usize,
) -> Result<Vec<ResidentDeviceInt4ColumnStats>, ExecuteError> {
    let mut clones = Vec::new();
    clones
        .try_reserve_exact(values.len())
        .map_err(|_| cold_index_host_staging_exhausted(table, required, limit))?;
    for value in values {
        clones.push(ResidentDeviceInt4ColumnStats {
            name: try_clone_cold_string(&value.name, table, required, limit)?,
            min: value.min,
            max: value.max,
        });
    }
    Ok(clones)
}

pub(super) fn try_clone_cold_memory_proof(
    proof: &gpu_db_execution::CudaDeviceMemoryProof,
    table: &str,
    required: usize,
    limit: usize,
) -> Result<gpu_db_execution::CudaDeviceMemoryProof, ExecuteError> {
    if proof.device_name.len() > COLD_INDEX_DESCRIPTOR_MAX_DEVICE_NAME_BYTES {
        return Err(cold_index_host_staging_exhausted(
            table,
            required.saturating_add(
                proof
                    .device_name
                    .len()
                    .saturating_sub(COLD_INDEX_DESCRIPTOR_MAX_DEVICE_NAME_BYTES),
            ),
            limit,
        ));
    }
    Ok(gpu_db_execution::CudaDeviceMemoryProof {
        gpu_id: proof.gpu_id,
        device_name: try_clone_cold_string(&proof.device_name, table, required, limit)?,
        allocated_bytes: proof.allocated_bytes,
        copied_bytes: proof.copied_bytes,
        retained: proof.retained,
    })
}

/// Validate one complete text-offset array with at most one positional read per fixed 64-KiB
/// block. The returned read/byte counts cover spill I/O only and make the syscall bound directly
/// sabotage-testable. No relational value is decoded or compared on the host.
pub(super) fn validate_cold_text_offsets_bounded(
    payload: &super::super::ColdPayload,
    byte_offset: usize,
    entries: usize,
    bytes_len: u64,
) -> Result<(bool, u64, u64), ()> {
    let total_bytes = entries.checked_mul(std::mem::size_of::<u64>()).ok_or(())?;
    let end = byte_offset.checked_add(total_bytes).ok_or(())?;
    if entries == 0 || end > payload.len() {
        return Err(());
    }
    let mut block = [0u8; COLD_TEXT_OFFSET_SCAN_BLOCK_BYTES];
    let mut consumed = 0usize;
    let mut previous = 0u64;
    let mut spill_reads = 0u64;
    let mut spill_bytes = 0u64;
    while consumed < total_bytes {
        let len = (total_bytes - consumed).min(block.len());
        let start = byte_offset.checked_add(consumed).ok_or(())?;
        payload.read_exact_range(start, &mut block[..len])?;
        if payload.is_spilled() {
            spill_reads = spill_reads.checked_add(1).ok_or(())?;
            spill_bytes = spill_bytes
                .checked_add(u64::try_from(len).map_err(|_| ())?)
                .ok_or(())?;
        }
        for encoded in block[..len].chunks_exact(std::mem::size_of::<u64>()) {
            let offset = u64::from_le_bytes(encoded.try_into().map_err(|_| ())?);
            if offset < previous || offset > bytes_len {
                return Ok((false, spill_reads, spill_bytes));
            }
            previous = offset;
        }
        consumed = consumed.checked_add(len).ok_or(())?;
    }
    Ok((previous == bytes_len, spill_reads, spill_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::FileExt;

    #[test]
    fn spilled_text_offset_scan_is_block_bounded_and_fail_closed() {
        const ENTRIES: usize = 200_001;
        let mut encoded = Vec::new();
        encoded.try_reserve_exact(ENTRIES * 8).unwrap();
        for offset in 0..ENTRIES {
            encoded.extend_from_slice(&(offset as u64).to_le_bytes());
        }
        let file = super::super::unlinked_spill_file().unwrap();
        let mut writer = file.as_ref();
        writer.write_all(&encoded).unwrap();
        let payload = super::super::super::ColdPayload::Spilled {
            file: Arc::clone(&file),
            offset: 0,
            len: encoded.len(),
        };
        let (valid, reads, bytes) =
            validate_cold_text_offsets_bounded(&payload, 0, ENTRIES, (ENTRIES - 1) as u64).unwrap();
        assert!(valid);
        assert_eq!(bytes, encoded.len() as u64);
        assert_eq!(
            reads as usize,
            encoded.len().div_ceil(COLD_TEXT_OFFSET_SCAN_BLOCK_BYTES)
        );
        assert!(
            reads < 32,
            "large offset arrays must not issue per-entry reads"
        );

        file.write_all_at(&0u64.to_le_bytes(), 100_000 * 8).unwrap();
        let (valid, corrupt_reads, corrupt_bytes) =
            validate_cold_text_offsets_bounded(&payload, 0, ENTRIES, (ENTRIES - 1) as u64).unwrap();
        assert!(!valid);
        assert!(corrupt_reads <= reads);
        assert!(corrupt_bytes <= bytes);
    }

    #[test]
    fn spilled_bitmap_repack_is_bit_exact_block_bounded_and_truncation_closed() {
        const SOURCE_ROWS: usize = 8 * 1024 * 1024;
        let source_words = SOURCE_ROWS.div_ceil(32);
        let mut encoded = Vec::new();
        encoded.try_reserve_exact(source_words * 4).unwrap();
        for ordinal in 0..source_words {
            let word = (ordinal as u32)
                .wrapping_mul(0x9e37_79b9)
                .rotate_left((ordinal % 31) as u32);
            encoded.extend_from_slice(&word.to_le_bytes());
        }
        let file = super::super::unlinked_spill_file().unwrap();
        let mut writer = file.as_ref();
        writer.write_all(&encoded).unwrap();
        let payload = super::super::super::ColdPayload::Spilled {
            file: Arc::clone(&file),
            offset: 0,
            len: encoded.len(),
        };
        let source_bit = |row: usize| {
            let offset = (row / 32) * 4;
            let word = u32::from_le_bytes(encoded[offset..offset + 4].try_into().unwrap());
            word & (1 << (row % 32)) != 0
        };

        for start in [0usize, 1, 31, 32, 33] {
            let rows = 257;
            let mut output = Vec::new();
            let (_, reads, bytes) = super::super::ColdValidationCopy {
                source: &payload,
                table: "bitmap_sabotage",
                host_limit: usize::MAX,
                required: 0,
            }
            .append_bitmap(&mut output, 0, SOURCE_ROWS, start, rows)
            .unwrap();
            assert_eq!(reads, 1);
            assert_eq!(bytes as usize, (rows + start % 32).div_ceil(32) * 4);
            for local in 0..rows {
                let word = u32::from_le_bytes(
                    output[(local / 32) * 4..(local / 32 + 1) * 4]
                        .try_into()
                        .unwrap(),
                );
                assert_eq!(
                    word & (1 << (local % 32)) != 0,
                    source_bit(start + local),
                    "unaligned bitmap repack diverged at start={start}, local={local}"
                );
            }
        }

        let start = 33usize;
        let rows = SOURCE_ROWS - start;
        let expected_source_bytes = (rows + start % 32).div_ceil(32) * 4;
        let expected_reads =
            expected_source_bytes.div_ceil(super::super::COLD_BITMAP_SCAN_BLOCK_BYTES);
        let mut aggregate_reads = 0u64;
        let mut aggregate_bytes = 0u64;
        for _ in 0..32 {
            let mut output = Vec::new();
            let (_, reads, bytes) = super::super::ColdValidationCopy {
                source: &payload,
                table: "wide_bitmap_sabotage",
                host_limit: usize::MAX,
                required: 0,
            }
            .append_bitmap(&mut output, 0, SOURCE_ROWS, start, rows)
            .unwrap();
            assert_eq!(reads as usize, expected_reads);
            assert_eq!(bytes as usize, expected_source_bytes);
            aggregate_reads += reads;
            aggregate_bytes += bytes;
        }
        assert_eq!(aggregate_reads as usize, expected_reads * 32);
        assert_eq!(aggregate_bytes as usize, expected_source_bytes * 32);
        assert!(
            aggregate_reads < (SOURCE_ROWS / 32) as u64,
            "32 wide spill bitmaps must still issue block reads, never scalar-word reads"
        );

        let truncated = super::super::super::ColdPayload::Spilled {
            file,
            offset: encoded.len() as u64 - 2,
            len: encoded.len(),
        };
        let error = super::super::ColdValidationCopy {
            source: &truncated,
            table: "truncated_bitmap_sabotage",
            host_limit: usize::MAX,
            required: 0,
        }
        .append_bitmap(&mut Vec::new(), 0, SOURCE_ROWS, 33, 257)
        .expect_err("a short positional range must fail closed");
        assert!(error.to_string().contains("bitmap payload is truncated"));
    }
}
