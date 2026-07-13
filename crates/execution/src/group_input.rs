use super::derived_column::{CudaGroupDeviceView, DeviceArithBuffer};
use super::resident_window::{CudaGroupTextSource, validate_aligned_window, validate_text_windows};
use super::{CudaResidentDeviceMemory, CudaRuntimeProbeError};

#[derive(Debug, Clone, Copy)]
pub enum CudaGroupFixedSource<'a> {
    Resident {
        byte_offset: u64,
        width: u8,
        row_count: u64,
    },
    Derived {
        buffer: CudaGroupDeviceView<'a>,
        width: u8,
        row_count: u64,
    },
}
impl CudaGroupFixedSource<'_> {
    fn width(self) -> u64 {
        match self {
            Self::Resident { width, .. } | Self::Derived { width, .. } => u64::from(width),
        }
    }

    fn row_count(self) -> u64 {
        match self {
            Self::Resident { row_count, .. } | Self::Derived { row_count, .. } => row_count,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaGroupWideSource<'a> {
    pub buffer: CudaGroupDeviceView<'a>,
    pub row_width: u64,
    pub row_count: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaGroupTextDescriptors<'a> {
    pub(super) buffer: CudaGroupDeviceView<'a>,
    pub(super) sources: &'a [CudaGroupTextSource],
}

pub struct CudaGroupTextDescriptorBuffer<'a> {
    pub(super) buffer: DeviceArithBuffer<'a>,
    pub(super) sources: Vec<CudaGroupTextSource>,
}

impl CudaGroupTextDescriptorBuffer<'_> {
    pub fn descriptors(&self) -> CudaGroupTextDescriptors<'_> {
        CudaGroupTextDescriptors {
            buffer: self.buffer.group_view(),
            sources: &self.sources,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CudaGroupKeySource<'a> {
    Fixed(CudaGroupFixedSource<'a>),
    Text {
        text: CudaGroupTextSource,
        fixed_component: Option<CudaGroupFixedSource<'a>>,
    },
    Composite {
        fixed: Option<CudaGroupWideSource<'a>>,
        text: Option<CudaGroupTextDescriptors<'a>>,
        row_count: u64,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum CudaGroupValueSource<'a> {
    /// COUNT-only input. The current PTX performs one bounded, ignored int4 placeholder load from
    /// resident offset zero before the aggregate mask suppresses value updates; preflight covers it.
    Unused {
        row_count: u64,
    },
    Fixed(CudaGroupFixedSource<'a>),
    Numeric(CudaGroupFixedSource<'a>),
    Uuid(CudaGroupFixedSource<'a>),
    Text(CudaGroupTextSource),
}

#[derive(Debug, Clone, Copy)]
pub struct CudaGroupByInput<'a> {
    pub key: CudaGroupKeySource<'a>,
    pub value: CudaGroupValueSource<'a>,
    pub key_validity_bitmap_offset: Option<u64>,
    pub value_validity_bitmap_offset: Option<u64>,
}

impl CudaGroupByInput<'static> {
    pub fn resident_i32(key_byte_offset: u64, value_byte_offset: u64, row_count: u64) -> Self {
        Self {
            key: CudaGroupKeySource::Fixed(CudaGroupFixedSource::Resident {
                byte_offset: key_byte_offset,
                width: 4,
                row_count,
            }),
            value: CudaGroupValueSource::Fixed(CudaGroupFixedSource::Resident {
                byte_offset: value_byte_offset,
                width: 4,
                row_count,
            }),
            key_validity_bitmap_offset: None,
            value_validity_bitmap_offset: None,
        }
    }
}

pub(super) struct ValidatedGroupInput {
    pub key_byte_offset: u64,
    pub value_byte_offset: u64,
    pub value_is_int8: bool,
    pub key_is_int8: bool,
    pub value_is_numeric: bool,
    pub value_is_uuid: bool,
    pub key_is_i128: bool,
    pub key_is_text: bool,
    pub key_offsets_off: u64,
    pub key_bytes_off: u64,
    pub key_bytes_len: u64,
    pub value_is_text: bool,
    pub value_offsets_off: u64,
    pub value_bytes_off: u64,
    pub value_bytes_len: u64,
    pub key_base_override: u64,
    pub value_base_override: u64,
    pub comp_w: u64,
    pub n_text: u64,
    pub text_desc_ptr: u64,
    pub value_null_off: Option<u64>,
    pub key_null_off: Option<u64>,
}

fn invalid(end: u64) -> CudaRuntimeProbeError {
    CudaRuntimeProbeError::InvalidInputLength(usize::try_from(end).unwrap_or(usize::MAX))
}

fn checked_end(offset: u64, count: u64, width: u64) -> Result<u64, CudaRuntimeProbeError> {
    count
        .checked_mul(width)
        .and_then(|bytes| offset.checked_add(bytes))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))
}

fn validate_view(
    view: CudaGroupDeviceView<'_>,
    context_identity: usize,
    required_bytes: u64,
) -> Result<(), CudaRuntimeProbeError> {
    if view.context_identity != context_identity || required_bytes > view.initialized_bytes {
        return Err(invalid(required_bytes));
    }
    Ok(())
}

fn validate_fixed(
    resident_bytes: u64,
    context_identity: usize,
    source: CudaGroupFixedSource<'_>,
    max_index: u64,
) -> Result<(u64, u64, u64, u64), CudaRuntimeProbeError> {
    let width = source.width();
    if !matches!(width, 4 | 8 | 16) || max_index >= source.row_count() {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(max_index).unwrap_or(usize::MAX),
        ));
    }
    let rows_end = checked_end(0, source.row_count(), width)?;
    match source {
        CudaGroupFixedSource::Resident { byte_offset, .. } => {
            validate_aligned_window(resident_bytes, byte_offset, source.row_count(), width, 4)?;
            Ok((byte_offset, 0, width, source.row_count()))
        }
        CudaGroupFixedSource::Derived { buffer, .. } => {
            validate_view(buffer, context_identity, rows_end)?;
            Ok((0, buffer.ptr, width, source.row_count()))
        }
    }
}

fn validate_text(
    resident_bytes: u64,
    source: CudaGroupTextSource,
    max_index: u64,
) -> Result<(), CudaRuntimeProbeError> {
    if max_index >= source.row_count {
        return Err(CudaRuntimeProbeError::InvalidInputLength(
            usize::try_from(max_index).unwrap_or(usize::MAX),
        ));
    }
    validate_text_windows(
        resident_bytes,
        source.offsets_byte_offset,
        source.bytes_byte_offset,
        source.bytes_len,
        source.row_count,
    )?;
    Ok(())
}

fn validate_bitmap(
    resident_bytes: u64,
    offset: Option<u64>,
    max_index: u64,
) -> Result<(), CudaRuntimeProbeError> {
    let Some(offset) = offset else {
        return Ok(());
    };
    let words = max_index
        .checked_div(32)
        .and_then(|word| word.checked_add(1))
        .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
    validate_aligned_window(resident_bytes, offset, words, 4, 4)
}

pub(super) fn validate_group_input(
    resident: &CudaResidentDeviceMemory,
    input: CudaGroupByInput<'_>,
    indices: &[u32],
) -> Result<ValidatedGroupInput, CudaRuntimeProbeError> {
    validate_group_input_parts(
        resident.metadata().allocated_bytes,
        std::ptr::from_ref(resident.primary()).addr(),
        input,
        indices,
    )
}

fn validate_group_input_parts(
    resident_bytes: u64,
    context_identity: usize,
    input: CudaGroupByInput<'_>,
    indices: &[u32],
) -> Result<ValidatedGroupInput, CudaRuntimeProbeError> {
    let max_index = u64::from(
        indices
            .iter()
            .copied()
            .max()
            .ok_or(CudaRuntimeProbeError::InvalidInputLength(0))?,
    );
    let key_rows = match input.key {
        CudaGroupKeySource::Fixed(source) => source.row_count(),
        CudaGroupKeySource::Text { text, .. } => text.row_count,
        CudaGroupKeySource::Composite { row_count, .. } => row_count,
    };
    let value_rows = match input.value {
        CudaGroupValueSource::Unused { row_count } => row_count,
        CudaGroupValueSource::Fixed(source)
        | CudaGroupValueSource::Numeric(source)
        | CudaGroupValueSource::Uuid(source) => source.row_count(),
        CudaGroupValueSource::Text(text) => text.row_count,
    };
    if key_rows != value_rows {
        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
    }
    let mut out = ValidatedGroupInput {
        key_byte_offset: 0,
        value_byte_offset: 0,
        value_is_int8: false,
        key_is_int8: false,
        value_is_numeric: false,
        value_is_uuid: false,
        key_is_i128: false,
        key_is_text: false,
        key_offsets_off: 0,
        key_bytes_off: 0,
        key_bytes_len: 0,
        value_is_text: false,
        value_offsets_off: 0,
        value_bytes_off: 0,
        value_bytes_len: 0,
        key_base_override: 0,
        value_base_override: 0,
        comp_w: 0,
        n_text: 0,
        text_desc_ptr: 0,
        value_null_off: input.value_validity_bitmap_offset,
        key_null_off: input.key_validity_bitmap_offset,
    };

    match input.key {
        CudaGroupKeySource::Fixed(source) => {
            let (offset, base, width, _) =
                validate_fixed(resident_bytes, context_identity, source, max_index)?;
            out.key_byte_offset = offset;
            out.key_base_override = base;
            out.key_is_int8 = width == 8;
            out.key_is_i128 = width == 16;
        }
        CudaGroupKeySource::Text {
            text,
            fixed_component,
        } => {
            validate_text(resident_bytes, text, max_index)?;
            out.key_is_text = true;
            out.key_offsets_off = text.offsets_byte_offset;
            out.key_bytes_off = text.bytes_byte_offset;
            out.key_bytes_len = text.bytes_len;
            if let Some(fixed) = fixed_component {
                let (_, base, width, rows) =
                    validate_fixed(resident_bytes, context_identity, fixed, max_index)?;
                if base == 0 || width != 8 || rows != text.row_count {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
                }
                out.key_base_override = base;
            }
        }
        CudaGroupKeySource::Composite {
            fixed,
            text,
            row_count,
        } => {
            if max_index >= row_count || (fixed.is_none() && text.is_none()) {
                return Err(CudaRuntimeProbeError::InvalidInputLength(
                    usize::try_from(max_index).unwrap_or(usize::MAX),
                ));
            }
            if let Some(fixed) = fixed {
                if fixed.row_width == 0 || fixed.row_count != row_count {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
                }
                let end = checked_end(0, row_count, fixed.row_width)?;
                validate_view(fixed.buffer, context_identity, end)?;
                out.key_base_override = fixed.buffer.ptr;
                out.comp_w = fixed.row_width;
            }
            if let Some(text) = text {
                if text.sources.is_empty() {
                    return Err(CudaRuntimeProbeError::InvalidInputLength(0));
                }
                let desc_bytes = text
                    .sources
                    .len()
                    .checked_mul(3 * std::mem::size_of::<u64>())
                    .ok_or(CudaRuntimeProbeError::InvalidInputLength(usize::MAX))?;
                validate_view(text.buffer, context_identity, desc_bytes as u64)?;
                for &source in text.sources {
                    if source.row_count != row_count {
                        return Err(CudaRuntimeProbeError::InvalidInputLength(usize::MAX));
                    }
                    validate_text(resident_bytes, source, max_index)?;
                }
                out.n_text = text.sources.len() as u64;
                out.text_desc_ptr = text.buffer.ptr;
            }
        }
    }

    match input.value {
        CudaGroupValueSource::Unused { row_count } => {
            // The current PTX value dispatch loads an ignored int4 placeholder before the COUNT mask
            // suppresses value aggregation. Keep that load total without exposing a fake raw pointer.
            let end = checked_end(0, row_count, 4)?;
            if end > resident_bytes || input.value_validity_bitmap_offset.is_some() {
                return Err(invalid(end));
            }
        }
        CudaGroupValueSource::Fixed(source)
        | CudaGroupValueSource::Numeric(source)
        | CudaGroupValueSource::Uuid(source) => {
            let (offset, base, width, _) =
                validate_fixed(resident_bytes, context_identity, source, max_index)?;
            let numeric = matches!(input.value, CudaGroupValueSource::Numeric(_));
            let uuid = matches!(input.value, CudaGroupValueSource::Uuid(_));
            if (numeric || uuid) && width != 16 {
                return Err(CudaRuntimeProbeError::InvalidInputLength(width as usize));
            }
            if !numeric && !uuid && !matches!(width, 4 | 8) {
                return Err(CudaRuntimeProbeError::InvalidInputLength(width as usize));
            }
            out.value_byte_offset = offset;
            out.value_base_override = base;
            out.value_is_int8 = width == 8;
            out.value_is_numeric = numeric;
            out.value_is_uuid = uuid;
        }
        CudaGroupValueSource::Text(text) => {
            validate_text(resident_bytes, text, max_index)?;
            out.value_is_text = true;
            out.value_offsets_off = text.offsets_byte_offset;
            out.value_bytes_off = text.bytes_byte_offset;
            out.value_bytes_len = text.bytes_len;
        }
    }
    validate_bitmap(resident_bytes, input.key_validity_bitmap_offset, max_index)?;
    validate_bitmap(
        resident_bytes,
        input.value_validity_bitmap_offset,
        max_index,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        CudaGroupByInput, CudaGroupDeviceView, CudaGroupFixedSource, CudaGroupKeySource,
        CudaGroupTextDescriptors, CudaGroupTextSource, CudaGroupValueSource, CudaGroupWideSource,
        checked_end, validate_bitmap, validate_fixed, validate_group_input_parts, validate_text,
        validate_view,
    };

    #[test]
    fn grouped_checked_extent_accepts_boundary_and_rejects_overflow() {
        assert_eq!(checked_end(8, 6, 4).unwrap(), 32);
        assert!(checked_end(u64::MAX, 1, 4).is_err());
        assert!(checked_end(0, u64::MAX, 16).is_err());
    }

    #[test]
    fn grouped_bitmap_extent_is_max_index_bounded() {
        validate_bitmap(12, Some(4), 32).unwrap();
        assert!(validate_bitmap(11, Some(4), 32).is_err());
        assert!(validate_bitmap(12, Some(1), 32).is_err());
        validate_bitmap(0, None, u64::MAX).unwrap();
    }

    #[test]
    fn grouped_fixed_sources_check_rows_width_and_borrowed_capacity() {
        let view = CudaGroupDeviceView::new(0x1000, 32, 7);
        validate_fixed(
            0,
            7,
            CudaGroupFixedSource::Derived {
                buffer: view,
                width: 8,
                row_count: 4,
            },
            3,
        )
        .unwrap();
        assert!(
            validate_fixed(
                0,
                7,
                CudaGroupFixedSource::Derived {
                    buffer: view,
                    width: 8,
                    row_count: 5,
                },
                4,
            )
            .is_err()
        );
        assert!(
            validate_fixed(
                64,
                7,
                CudaGroupFixedSource::Resident {
                    byte_offset: 60,
                    width: 4,
                    row_count: 2,
                },
                1,
            )
            .is_err()
        );
        assert!(
            validate_fixed(
                64,
                7,
                CudaGroupFixedSource::Resident {
                    byte_offset: 1,
                    width: 4,
                    row_count: 2,
                },
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn grouped_device_views_enforce_origin_and_initialized_extent() {
        let view = CudaGroupDeviceView::new(0x1000, 4, 7);
        validate_view(view, 7, 4).unwrap();
        assert!(validate_view(view, 8, 4).is_err());
        assert!(validate_view(view, 7, 5).is_err());
    }

    #[test]
    fn grouped_text_sources_check_offsets_bytes_and_logical_rows() {
        let source = CudaGroupTextSource {
            offsets_byte_offset: 0,
            bytes_byte_offset: 24,
            bytes_len: 8,
            row_count: 2,
        };
        validate_text(32, source, 1).unwrap();
        assert!(validate_text(31, source, 1).is_err());
        assert!(validate_text(32, source, 2).is_err());
        assert!(
            validate_text(
                40,
                CudaGroupTextSource {
                    offsets_byte_offset: 4,
                    bytes_byte_offset: 28,
                    bytes_len: 8,
                    row_count: 2,
                },
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn grouped_mode_matrix_checks_exact_extents_and_incoherent_modes() {
        let fixed4 = CudaGroupFixedSource::Resident {
            byte_offset: 0,
            width: 4,
            row_count: 2,
        };
        let fixed16 = CudaGroupFixedSource::Resident {
            byte_offset: 16,
            width: 16,
            row_count: 2,
        };
        let text = CudaGroupTextSource {
            offsets_byte_offset: 48,
            bytes_byte_offset: 72,
            bytes_len: 8,
            row_count: 2,
        };
        let input = |key, value| CudaGroupByInput {
            key,
            value,
            key_validity_bitmap_offset: None,
            value_validity_bitmap_offset: None,
        };

        for value in [
            CudaGroupValueSource::Fixed(fixed4),
            CudaGroupValueSource::Numeric(fixed16),
            CudaGroupValueSource::Uuid(fixed16),
            CudaGroupValueSource::Text(text),
            CudaGroupValueSource::Unused { row_count: 2 },
        ] {
            validate_group_input_parts(
                80,
                7,
                input(CudaGroupKeySource::Fixed(fixed4), value),
                &[1],
            )
            .unwrap();
        }
        validate_group_input_parts(
            80,
            7,
            input(
                CudaGroupKeySource::Text {
                    text,
                    fixed_component: None,
                },
                CudaGroupValueSource::Fixed(fixed4),
            ),
            &[1],
        )
        .unwrap();

        let sources = [text];
        validate_group_input_parts(
            80,
            7,
            input(
                CudaGroupKeySource::Composite {
                    fixed: Some(CudaGroupWideSource {
                        buffer: CudaGroupDeviceView::new(0x1000, 16, 7),
                        row_width: 8,
                        row_count: 2,
                    }),
                    text: Some(CudaGroupTextDescriptors {
                        buffer: CudaGroupDeviceView::new(0x2000, 24, 7),
                        sources: &sources,
                    }),
                    row_count: 2,
                },
                CudaGroupValueSource::Fixed(fixed4),
            ),
            &[1],
        )
        .unwrap();

        assert!(
            validate_group_input_parts(
                80,
                7,
                input(
                    CudaGroupKeySource::Composite {
                        fixed: None,
                        text: None,
                        row_count: 2,
                    },
                    CudaGroupValueSource::Unused { row_count: 2 },
                ),
                &[1],
            )
            .is_err()
        );
        assert!(
            validate_group_input_parts(
                80,
                7,
                input(
                    CudaGroupKeySource::Fixed(fixed4),
                    CudaGroupValueSource::Fixed(fixed16),
                ),
                &[1],
            )
            .is_err()
        );
        assert!(
            validate_group_input_parts(
                80,
                7,
                input(
                    CudaGroupKeySource::Fixed(fixed4),
                    CudaGroupValueSource::Unused { row_count: 3 },
                ),
                &[1],
            )
            .is_err()
        );
        let mut nullable_unused = input(
            CudaGroupKeySource::Fixed(fixed4),
            CudaGroupValueSource::Unused { row_count: 2 },
        );
        nullable_unused.value_validity_bitmap_offset = Some(76);
        assert!(validate_group_input_parts(80, 7, nullable_unused, &[1]).is_err());
    }
}
