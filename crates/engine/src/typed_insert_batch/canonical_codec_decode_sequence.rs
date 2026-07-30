//! Sequence-section validation that needs the decoder's private parsed model.

use super::super::validation::{self, SequenceSectionEntry, SequenceSectionKind};
use super::*;

pub(super) fn validate_decoded_sequence_section(
    parent: DecodedParent,
    effects: &[DecodedEffect],
) -> Result<(), EngineError> {
    let entries = effects
        .iter()
        .map(|effect| {
            let kind = match effect.kind {
                DecodedEffectKind::Published {
                    transition_txn_id, ..
                } => SequenceSectionKind::Published { transition_txn_id },
                DecodedEffectKind::Private {
                    lifetime_origin,
                    owner,
                    ..
                } => SequenceSectionKind::Private {
                    lifetime_origin,
                    owner,
                },
            };
            SequenceSectionEntry {
                request: effect.request.view(),
                absolute_expression_ordinal: effect.request.absolute_expression_ordinal,
                kind,
            }
        })
        .collect::<Vec<_>>();
    validation::validate_sequence_section(parent.view(), &entries)
}

pub(super) fn validate_sequence_vector_target(
    request: &DecodedRequest,
    target: &Target,
    columns: &[DecodedColumn],
) -> Result<(), EngineError> {
    let column = columns
        .get(request.catalog_column_ordinal as usize)
        .ok_or_else(|| codec_error("sequence request column is absent"))?;
    let row =
        usize::try_from(request.row_ordinal).map_err(|_| codec_error("sequence row overflows"))?;
    let resolved_value = match (&column.values, column.ty) {
        (TypedInsertColumnValues::I32(values), SqlType::Int4) => values.get(row).copied(),
        _ => None,
    };
    validation::validate_sequence_target_cell(
        target.oid,
        target.rows,
        request.view(),
        column.column_id,
        column.ty,
        column.states.get(row).copied(),
        column.defaults.was_defaulted(row),
        column.validity.is_valid(row),
        resolved_value,
        request.value,
    )
}
