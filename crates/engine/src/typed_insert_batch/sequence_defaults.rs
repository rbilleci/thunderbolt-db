//! Stateful sequence-default request binding and seal-time materialization.
//!
//! Discovery is pure catalog/semantic work. It never advances a sequence. A later owner must
//! return one move-only binding for each exact request before this module can materialize i32
//! output vectors.

use super::*;
use std::collections::BTreeSet;

#[cfg(test)]
std::thread_local! {
    static MATERIALIZATION_WRITE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_materialization_write_count() {
    MATERIALIZATION_WRITE_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn materialization_write_count() -> usize {
    MATERIALIZATION_WRITE_COUNT.with(std::cell::Cell::get)
}

pub(crate) mod effects;
pub(crate) use effects::{SequenceDefaultBinding, SequenceDefaultBindings};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SequenceDefaultRequest {
    pub(crate) target_table_oid: u32,
    pub(crate) row_ordinal: u32,
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) column_id: u32,
    pub(crate) sequence_oid: u32,
    sequence_source_name: Arc<str>,
    sequence_effective_name: Arc<str>,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) expression_ordinal: u32,
}

/// One ordered scalar sequence-default request view for a future effect owner. The source and
/// effective names remain names only: this view cannot expose or mutate the typed value vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Adopted by the next production-compiled effect-plan handoff.
pub(crate) struct SequenceDefaultRequestEffectShape<'a> {
    target_table_oid: u32,
    row_ordinal: u32,
    catalog_column_ordinal: u32,
    column_id: u32,
    sequence_oid: u32,
    sequence_source_name: &'a str,
    sequence_effective_name: &'a str,
    statement_ordinal: InsertStatementOrdinal,
    expression_ordinal: u32,
}

#[allow(dead_code)] // The current live route intentionally has no effect-plan consumer.
impl SequenceDefaultRequestEffectShape<'_> {
    pub(crate) const fn target_table_oid(self) -> u32 {
        self.target_table_oid
    }

    pub(crate) const fn row_ordinal(self) -> u32 {
        self.row_ordinal
    }

    pub(crate) const fn catalog_column_ordinal(self) -> u32 {
        self.catalog_column_ordinal
    }

    pub(crate) const fn column_id(self) -> u32 {
        self.column_id
    }

    pub(crate) const fn sequence_oid(self) -> u32 {
        self.sequence_oid
    }

    pub(crate) fn sequence_source_name(&self) -> &str {
        self.sequence_source_name
    }

    pub(crate) fn sequence_effective_name(&self) -> &str {
        self.sequence_effective_name
    }

    pub(crate) const fn statement_ordinal(self) -> InsertStatementOrdinal {
        self.statement_ordinal
    }

    pub(crate) const fn expression_ordinal(self) -> u32 {
        self.expression_ordinal
    }
}

pub(crate) struct SequenceDefaultRequests {
    requests: Box<[SequenceDefaultRequest]>,
}

impl SequenceDefaultRequests {
    pub(crate) fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    #[allow(dead_code)] // The current live route intentionally has no effect-plan consumer.
    pub(crate) fn effect_shapes(
        &self,
    ) -> impl ExactSizeIterator<Item = SequenceDefaultRequestEffectShape<'_>> + '_ {
        self.requests
            .iter()
            .map(|request| SequenceDefaultRequestEffectShape {
                target_table_oid: request.target_table_oid,
                row_ordinal: request.row_ordinal,
                catalog_column_ordinal: request.catalog_column_ordinal,
                column_id: request.column_id,
                sequence_oid: request.sequence_oid,
                sequence_source_name: request.sequence_source_name.as_ref(),
                sequence_effective_name: request.sequence_effective_name.as_ref(),
                statement_ordinal: request.statement_ordinal,
                expression_ordinal: request.expression_ordinal,
            })
    }

    #[cfg(test)]
    pub(crate) fn requests(&self) -> &[SequenceDefaultRequest] {
        &self.requests
    }
}

pub(super) fn discover(
    columns: &[TypedInsertColumn],
    table: &RelationalTable,
    catalog: &CatalogSnapshot,
    statement_ordinal: InsertStatementOrdinal,
    rows: usize,
) -> Result<SequenceDefaultRequests, ExecuteError> {
    if columns.len() != table.columns.len()
        || columns.iter().zip(&table.columns).any(|(sealed, live)| {
            sealed.column_id != live.id
                || sealed.ty != live.ty
                || sealed.input_states.len() != rows
                || sealed.input_provenance.len() != rows
        })
    {
        return Err(EngineError::Durability(
            "typed INSERT sequence discovery lost catalog-order vector geometry".to_string(),
        )
        .into());
    }

    // Match the existing sequence-value transition geometry: first select only sequence-backed
    // catalog columns with at least one omission/DEFAULT anywhere in this statement, then assign
    // row × active-column slots. A wholly supplied sequence column is not an active expression
    // and must not shift another column's ordinal or canonical input digest.
    let sequence_slots = table
        .columns
        .iter()
        .zip(columns)
        .enumerate()
        .filter_map(|(catalog_ordinal, (live, sealed))| {
            (matches!(live.default, Some(ColumnDefault::SequenceNextVal { .. }))
                && sealed.input_states.iter().any(|state| {
                    matches!(
                        state,
                        TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
                    )
                }))
            .then_some(catalog_ordinal)
        })
        .collect::<Vec<_>>();
    let sequence_slot_count = u32::try_from(sequence_slots.len()).map_err(|_| {
        EngineError::Durability("typed INSERT sequence slot count exceeds u32".to_string())
    })?;
    u32::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(sequence_slot_count))
        .ok_or_else(|| {
            EngineError::Durability(
                "typed INSERT sequence expression geometry exceeds u32".to_string(),
            )
        })?;
    let mut requests = Vec::new();
    for row in 0..rows {
        for (catalog_column_ordinal, (sealed, live)) in
            columns.iter().zip(&table.columns).enumerate()
        {
            let Some(ColumnDefault::SequenceNextVal { sequence, .. }) = live.default.as_ref()
            else {
                continue;
            };
            if !matches!(
                sealed.input_states[row],
                TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
            ) {
                continue;
            }
            let sequence_oid = catalog
                .relational_sequences
                .get(sequence)
                .map(|sequence| sequence.oid)
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "sequence \"{sequence}\" for relation \"{}\" does not exist",
                        table.name
                    ))
                })?;
            let sequence_slot = sequence_slots
                .iter()
                .position(|slot| *slot == catalog_column_ordinal)
                .and_then(|slot| u32::try_from(slot).ok())
                .ok_or_else(|| {
                    EngineError::Durability(
                        "typed INSERT sequence catalog slot is not stable".to_string(),
                    )
                })?;
            let expression_ordinal = u32::try_from(row)
                .ok()
                .and_then(|row| row.checked_mul(sequence_slot_count))
                .and_then(|row_base| row_base.checked_add(sequence_slot))
                .ok_or_else(|| {
                    EngineError::Durability(
                        "typed INSERT sequence expression ordinal exceeds u32".to_string(),
                    )
                })?;
            requests.push(SequenceDefaultRequest {
                target_table_oid: table.oid,
                row_ordinal: u32::try_from(row).map_err(|_| {
                    EngineError::Durability(
                        "typed INSERT sequence row ordinal exceeds u32".to_string(),
                    )
                })?,
                catalog_column_ordinal: u32::try_from(catalog_column_ordinal).map_err(|_| {
                    EngineError::Durability(
                        "typed INSERT sequence catalog column ordinal exceeds u32".to_string(),
                    )
                })?,
                column_id: live.id,
                sequence_oid,
                sequence_source_name: Arc::from(sequence.as_str()),
                sequence_effective_name: Arc::from(sequence.as_str()),
                statement_ordinal,
                expression_ordinal,
            });
        }
    }
    Ok(SequenceDefaultRequests {
        requests: requests.into(),
    })
}

pub(super) fn materialize(
    columns: &mut [TypedInsertColumn],
    statement_ordinal: InsertStatementOrdinal,
    rows: usize,
    requests: &SequenceDefaultRequests,
    bindings: SequenceDefaultBindings,
) -> Result<Box<[SequenceDefaultBinding]>, ExecuteError> {
    validate_bindings(columns, statement_ordinal, rows, requests, &bindings)?;

    let default_words = bitmap_words(rows)?;
    let mut defaulted: Vec<Vec<u32>> = columns.iter().map(|_| vec![0; default_words]).collect();
    for (request, binding) in requests.requests.iter().zip(&bindings.bindings) {
        let row = usize::try_from(request.row_ordinal)
            .expect("validated sequence-default row ordinal is addressable");
        let column_ordinal = usize::try_from(request.catalog_column_ordinal)
            .expect("validated sequence-default column ordinal is addressable");
        let column = columns
            .get_mut(column_ordinal)
            .expect("validated sequence-default column is present");
        let TypedInsertColumnValues::I32(values) = &mut column.values else {
            unreachable!("validated sequence-default output is i32")
        };
        #[cfg(test)]
        MATERIALIZATION_WRITE_COUNT.with(|count| count.set(count.get() + 1));
        values[row] =
            i32::try_from(binding.value).expect("validated sequence-default output fits i32");
        match &mut column.validity {
            TypedInsertColumnValidity::AllValid => {
                unreachable!("validated sequence-default target is initially invalid")
            }
            TypedInsertColumnValidity::Bitmap(words) => set_bit(words, row)
                .expect("validated sequence-default validity bitmap covers its row"),
        }
        set_bit(&mut defaulted[column_ordinal], row)
            .expect("validated sequence-default coverage bitmap covers its row");
    }
    for (column_ordinal, column) in columns.iter_mut().enumerate() {
        let has_request = requests
            .requests
            .iter()
            .any(|request| request.catalog_column_ordinal == column_ordinal as u32);
        if !has_request {
            continue;
        }
        let all_defaults_bound = column.input_states.iter().enumerate().all(|(row, state)| {
            !matches!(
                state,
                TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
            ) || bit_is_set(&defaulted[column_ordinal], row)
        });
        debug_assert!(
            all_defaults_bound,
            "preflight proved complete default coverage"
        );
        column.presence = TypedInsertColumnPresence::AllProvided;
        column.default_resolution = TypedInsertDefaultResolution::Bitmap(
            std::mem::take(&mut defaulted[column_ordinal]).into(),
        );
        if let TypedInsertColumnValidity::Bitmap(words) = &column.validity {
            if bitmap_is_all_set(words, rows) {
                column.validity = TypedInsertColumnValidity::AllValid;
            }
        }
    }
    Ok(bindings.bindings)
}

/// Validate the complete effect bundle before any typed value vector is materialized. The inert
/// WRITE-001 terminal calls this same pure gate before it consumes semantic vectors through
/// `PreparedTypedInsert::seal`, so every rejected bundle leaves those vectors untouched.
pub(super) fn validate_bindings(
    columns: &[TypedInsertColumn],
    statement_ordinal: InsertStatementOrdinal,
    rows: usize,
    requests: &SequenceDefaultRequests,
    bindings: &SequenceDefaultBindings,
) -> Result<(), ExecuteError> {
    if requests.requests.len() != bindings.bindings.len() {
        return Err(binding_error("missing or extra sequence-default binding"));
    }
    let parent = match (&bindings.parent, requests.requests.is_empty()) {
        (None, true) => return Ok(()),
        (Some(parent), false) => parent,
        _ => return Err(binding_error("sequence-default parent context drift")),
    };
    let default_words = bitmap_words(rows)?;
    let mut covered_slots = BTreeSet::new();
    let mut previous: Option<&SequenceDefaultRequest> = None;
    for (request, binding) in requests.requests.iter().zip(&bindings.bindings) {
        if request.statement_ordinal != statement_ordinal
            || !binding.matches_request(request, parent)
            || !request_order_is_exact(previous, request)
        {
            return Err(binding_error("sequence-default binding identity drift"));
        }
        let row = usize::try_from(request.row_ordinal).map_err(|_| {
            binding_error("sequence-default binding row ordinal is not addressable")
        })?;
        let column_ordinal = usize::try_from(request.catalog_column_ordinal).map_err(|_| {
            binding_error("sequence-default binding column ordinal is not addressable")
        })?;
        let Some(sealed) = columns.get(column_ordinal) else {
            return Err(binding_error("sequence-default binding column is absent"));
        };
        if row >= rows
            || sealed.column_id != request.column_id
            || sealed.ty != SqlType::Int4
            || !matches!(
                sealed.input_states.get(row),
                Some(TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault)
            )
            || i32::try_from(binding.value).is_err()
        {
            return Err(binding_error(
                "sequence-default binding value or target drift",
            ));
        }
        let TypedInsertColumnValidity::Bitmap(words) = &sealed.validity else {
            return Err(binding_error(
                "sequence-default binding overwrote an already-valid cell",
            ));
        };
        if words.len() != default_words
            || bit_is_set(words, row)
            || !covered_slots.insert((column_ordinal, row))
        {
            return Err(binding_error(
                "sequence-default binding validity or slot coverage drift",
            ));
        }
        previous = Some(request);
    }
    for (column_ordinal, sealed) in columns.iter().enumerate() {
        let has_request = requests
            .requests
            .iter()
            .any(|request| request.catalog_column_ordinal == column_ordinal as u32);
        if !has_request {
            continue;
        }
        if sealed.input_states.iter().enumerate().any(|(row, state)| {
            matches!(
                state,
                TypedInsertInputState::Omitted | TypedInsertInputState::ExplicitDefault
            ) && !covered_slots.contains(&(column_ordinal, row))
        }) {
            return Err(binding_error(
                "sequence-default binding omitted a requested cell",
            ));
        }
    }
    Ok(())
}

fn request_order_is_exact(
    previous: Option<&SequenceDefaultRequest>,
    request: &SequenceDefaultRequest,
) -> bool {
    previous.is_none_or(|previous| {
        (
            previous.row_ordinal,
            previous.catalog_column_ordinal,
            previous.expression_ordinal,
        ) < (
            request.row_ordinal,
            request.catalog_column_ordinal,
            request.expression_ordinal,
        )
    })
}

fn binding_error(message: &str) -> ExecuteError {
    EngineError::Durability(format!("typed INSERT {message}")).into()
}
