use super::CudaRuntimeProbeError;

fn invalid_window(end: u64) -> CudaRuntimeProbeError {
    CudaRuntimeProbeError::InvalidInputLength(usize::try_from(end).unwrap_or(usize::MAX))
}

pub(super) fn validate_window(
    allocated_bytes: u64,
    byte_offset: u64,
    element_count: u64,
    element_width: u64,
) -> Result<(), CudaRuntimeProbeError> {
    let end = element_count
        .checked_mul(element_width)
        .and_then(|bytes| byte_offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    if end > allocated_bytes {
        return Err(invalid_window(end));
    }
    Ok(())
}

pub(super) fn validate_text_windows(
    allocated_bytes: u64,
    offsets_byte_offset: u64,
    bytes_byte_offset: u64,
    bytes_len: u64,
    row_count: u64,
) -> Result<u64, CudaRuntimeProbeError> {
    let offset_count = row_count
        .checked_add(1)
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    validate_window(
        allocated_bytes,
        offsets_byte_offset,
        offset_count,
        std::mem::size_of::<u64>() as u64,
    )?;
    validate_window(allocated_bytes, bytes_byte_offset, bytes_len, 1)?;
    Ok(bytes_len)
}
