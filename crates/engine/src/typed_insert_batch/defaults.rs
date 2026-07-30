//! Deterministic scalar-default lowering for sealed typed INSERT vectors.
//!
//! Source-order coercion has already completed when this leaf runs. It walks rows first and
//! catalog columns second, rebuilding only vectors that contain an omitted or explicit DEFAULT
//! cell. The original state/provenance vectors remain untouched, while the physical vectors gain
//! a compact proof that every such cell was resolved before WAL templating or device planning.

use super::*;

pub(super) enum DefaultResolutionError {
    Engine(EngineError),
}

pub(super) fn resolve(
    columns: &mut [TypedInsertColumn],
    table: &RelationalTable,
    rows: usize,
) -> Result<(), DefaultResolutionError> {
    if columns.len() != table.columns.len()
        || columns.iter().zip(&table.columns).any(|(sealed, live)| {
            sealed.column_id != live.id || sealed.ty != live.ty || !sealed.values.rows_match(rows)
        })
    {
        return Err(DefaultResolutionError::Engine(EngineError::Durability(
            "typed INSERT default resolver lost catalog-order vector geometry".to_string(),
        )));
    }

    let mut rebuilt = columns
        .iter()
        .zip(&table.columns)
        .map(|(column, live)| {
            let needs_default = column.input_states.iter().any(|state| {
                matches!(
                    state,
                    TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
                )
            });
            // Stateful sequence cells remain visibly unresolved until their dedicated owner
            // returns exact per-row receipts.  They must not be rebuilt into scalar placeholders.
            (needs_default && !matches!(live.default, Some(ColumnDefault::SequenceNextVal { .. })))
                .then(|| RebuiltDefaultColumn::new(column.ty, rows))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(DefaultResolutionError::Engine)?;

    // Scalar defaults are deterministic and must be evaluated once per column before
    // the row loop.  The resulting value is broadcast into every omitted/DEFAULT slot;
    // sequences deliberately remain a deferred per-row route.
    let resolved_scalar_defaults = columns
        .iter()
        .zip(&table.columns)
        .map(|(column, live)| {
            let needs_default = column.input_states.iter().any(|state| {
                matches!(
                    state,
                    TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
                )
            });
            if !needs_default {
                return Ok(None);
            }
            match live.default.as_ref() {
                Some(ColumnDefault::SequenceNextVal { .. }) => Ok(None),
                Some(
                    default @ (ColumnDefault::Literal(_) | ColumnDefault::DeferredScalar { .. }),
                ) => evaluate_scalar(default, live.ty, &live.name)
                    .map(Some)
                    .map_err(DefaultResolutionError::Engine),
                None => Ok(None),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    for row in 0..rows {
        for (((column, live), replacement), scalar_default) in columns
            .iter()
            .zip(&table.columns)
            .zip(&mut rebuilt)
            .zip(&resolved_scalar_defaults)
        {
            let Some(output) = replacement.as_mut() else {
                continue;
            };
            match column.input_states[row] {
                TypedInsertInputState::Provided => output
                    .copy_direct(&column.values, column.validity.is_valid(row), row)
                    .map_err(DefaultResolutionError::Engine)?,
                TypedInsertInputState::ProvidedNull => {
                    output
                        .write_null(row)
                        .map_err(DefaultResolutionError::Engine)?;
                }
                TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault => {
                    output
                        .mark_defaulted(row)
                        .map_err(DefaultResolutionError::Engine)?;
                    match live.default.as_ref() {
                        Some(ColumnDefault::Literal(SqlValue::Null)) | None => {
                            output
                                .write_null(row)
                                .map_err(DefaultResolutionError::Engine)?;
                        }
                        Some(ColumnDefault::Literal(_))
                        | Some(ColumnDefault::DeferredScalar { .. }) => {
                            let value = scalar_default
                                .as_ref()
                                .expect("scalar default pre-evaluated");
                            if matches!(value, SqlValue::Null) {
                                output
                                    .write_null(row)
                                    .map_err(DefaultResolutionError::Engine)?;
                            } else {
                                output
                                    .write_default(value, row)
                                    .map_err(DefaultResolutionError::Engine)?;
                            }
                        }
                        Some(ColumnDefault::SequenceNextVal { .. }) => {
                            unreachable!("sequence defaults have no scalar replacement vector")
                        }
                    }
                }
            }
            output
                .finish_row(row)
                .map_err(DefaultResolutionError::Engine)?;
        }
    }

    for (column, replacement) in columns.iter_mut().zip(rebuilt) {
        let Some(replacement) = replacement else {
            column.default_resolution = TypedInsertDefaultResolution::AllDirect;
            continue;
        };
        let (values, validity, default_resolution) = replacement.finish();
        column.values = values;
        column.validity = validity;
        column.presence = TypedInsertColumnPresence::AllProvided;
        column.default_resolution = default_resolution;
    }
    Ok(())
}

struct RebuiltDefaultColumn {
    ty: SqlType,
    rows: usize,
    values: TypedInsertColumnValues,
    validity_words: Vec<u32>,
    defaulted_words: Vec<u32>,
    text_bytes: Option<Vec<u8>>,
}

impl RebuiltDefaultColumn {
    fn new(ty: SqlType, rows: usize) -> Result<Self, EngineError> {
        Ok(Self {
            ty,
            rows,
            values: TypedInsertColumnValues::zeroed(ty, rows)?,
            validity_words: vec![0; bitmap_words(rows)?],
            defaulted_words: vec![0; bitmap_words(rows)?],
            text_bytes: (ty == SqlType::Text).then(Vec::new),
        })
    }

    fn copy_direct(
        &mut self,
        source: &TypedInsertColumnValues,
        valid: bool,
        row: usize,
    ) -> Result<(), EngineError> {
        if !valid {
            return self.write_null(row);
        }
        set_bit(&mut self.validity_words, row)?;
        match (&mut self.values, source, self.ty) {
            (TypedInsertColumnValues::I32(output), TypedInsertColumnValues::I32(input), _) => {
                output[row] = input[row]
            }
            (TypedInsertColumnValues::I64(output), TypedInsertColumnValues::I64(input), _) => {
                output[row] = input[row]
            }
            (TypedInsertColumnValues::I128(output), TypedInsertColumnValues::I128(input), _) => {
                output[row] = input[row]
            }
            (
                TypedInsertColumnValues::Bytes16(output),
                TypedInsertColumnValues::Bytes16(input),
                _,
            ) => output[row] = input[row],
            (
                TypedInsertColumnValues::BoolBits(output),
                TypedInsertColumnValues::BoolBits(input),
                SqlType::Bool,
            ) => {
                if bit_is_set(input, row) {
                    set_bit(output, row)?;
                }
            }
            (
                TypedInsertColumnValues::Text { offsets, bytes },
                TypedInsertColumnValues::Text {
                    offsets: input_offsets,
                    bytes: input_bytes,
                },
                SqlType::Text,
            ) => {
                let start = usize::try_from(input_offsets[row]).map_err(|_| {
                    EngineError::Durability("typed INSERT text start offset overflows".to_string())
                })?;
                let end = usize::try_from(input_offsets[row + 1]).map_err(|_| {
                    EngineError::Durability("typed INSERT text end offset overflows".to_string())
                })?;
                let value = input_bytes.get(start..end).ok_or_else(|| {
                    EngineError::Durability("typed INSERT text offsets are invalid".to_string())
                })?;
                let output_bytes = self.text_bytes.as_mut().ok_or_else(|| {
                    EngineError::Durability(
                        "typed INSERT text resolver lost byte vector".to_string(),
                    )
                })?;
                output_bytes.extend_from_slice(value);
                // `finish_row` owns offsets so NULL and empty strings stay distinguishable.
                let _ = (offsets, bytes);
            }
            _ => {
                return Err(EngineError::Durability(
                    "typed INSERT direct default copy disagrees with its vector arm".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn write_default(&mut self, value: &SqlValue, row: usize) -> Result<(), EngineError> {
        if matches!(value, SqlValue::Null) {
            return self.write_null(row);
        }
        set_bit(&mut self.validity_words, row)?;
        match (&mut self.values, self.ty, value) {
            (TypedInsertColumnValues::I32(values), SqlType::Int2, SqlValue::Int2(value)) => {
                values[row] = i32::from(*value)
            }
            (TypedInsertColumnValues::I32(values), SqlType::Int4, SqlValue::Int4(value))
            | (TypedInsertColumnValues::I32(values), SqlType::Date, SqlValue::Date(value)) => {
                values[row] = *value
            }
            (TypedInsertColumnValues::I64(values), SqlType::Int8, SqlValue::Int8(value))
            | (
                TypedInsertColumnValues::I64(values),
                SqlType::Timestamp,
                SqlValue::Timestamp(value),
            ) => values[row] = *value,
            (
                TypedInsertColumnValues::I128(values),
                SqlType::Numeric { .. },
                SqlValue::Numeric(value),
            ) => values[row] = value.mantissa,
            (TypedInsertColumnValues::Bytes16(values), SqlType::Uuid, SqlValue::Uuid(value)) => {
                values[row] = *value
            }
            (TypedInsertColumnValues::BoolBits(words), SqlType::Bool, SqlValue::Bool(value)) => {
                if *value {
                    set_bit(words, row)?;
                }
            }
            (TypedInsertColumnValues::Text { .. }, SqlType::Text, SqlValue::Text(value)) => {
                let bytes = self.text_bytes.as_mut().ok_or_else(|| {
                    EngineError::Durability(
                        "typed INSERT text resolver lost byte vector".to_string(),
                    )
                })?;
                let next_len = bytes.len().checked_add(value.len()).ok_or_else(|| {
                    EngineError::Durability("typed INSERT text bytes overflow".to_string())
                })?;
                u64::try_from(next_len).map_err(|_| {
                    EngineError::Durability("typed INSERT text exceeds u64 offsets".to_string())
                })?;
                bytes.extend_from_slice(value.as_bytes());
            }
            _ => {
                return Err(EngineError::Durability(
                    "typed INSERT literal default does not match its column type".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn write_null(&mut self, _row: usize) -> Result<(), EngineError> {
        Ok(())
    }

    fn mark_defaulted(&mut self, row: usize) -> Result<(), EngineError> {
        set_bit(&mut self.defaulted_words, row)
    }

    fn finish_row(&mut self, row: usize) -> Result<(), EngineError> {
        if let TypedInsertColumnValues::Text { offsets, .. } = &mut self.values {
            let bytes = self.text_bytes.as_ref().ok_or_else(|| {
                EngineError::Durability("typed INSERT text resolver lost byte vector".to_string())
            })?;
            offsets[row.checked_add(1).ok_or_else(|| {
                EngineError::Durability("typed INSERT text row index overflows".to_string())
            })?] = u64::try_from(bytes.len()).map_err(|_| {
                EngineError::Durability("typed INSERT text exceeds u64 offsets".to_string())
            })?;
        }
        Ok(())
    }

    fn finish(
        mut self,
    ) -> (
        TypedInsertColumnValues,
        TypedInsertColumnValidity,
        TypedInsertDefaultResolution,
    ) {
        if let TypedInsertColumnValues::Text { bytes, .. } = &mut self.values {
            *bytes = self
                .text_bytes
                .take()
                .expect("text vector exists for a text column")
                .into();
        }
        (
            self.values,
            if bitmap_is_all_set(&self.validity_words, self.rows) {
                TypedInsertColumnValidity::AllValid
            } else {
                TypedInsertColumnValidity::Bitmap(self.validity_words.into())
            },
            TypedInsertDefaultResolution::Bitmap(self.defaulted_words.into()),
        )
    }
}
