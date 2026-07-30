//! Decoder-only sequence validation without decoder-side maps or sets.
//!
//! The producer keeps its logarithmic map implementation.  The strict reader has already
//! retained the bounded effect vector, so it verifies the same identities with rescans rather
//! than allocating attacker-directed validation registries.

use super::super::validation;
use super::*;

pub(super) fn validate_decoded_sequence_section(
    parent: DecodedParent,
    effects: &[DecodedEffect],
) -> Result<(), EngineError> {
    if effects.is_empty() {
        return Err(codec_error("sequence section parent has no effects"));
    }
    if parent.autocommit
        && (parent.statement_ordinal != InsertStatementOrdinal::FIRST
            || parent.expression_base != 0)
    {
        return Err(codec_error(
            "autocommit sequence parent geometry is invalid",
        ));
    }

    let active_count = effects
        .iter()
        .enumerate()
        .filter(|(ordinal, effect)| {
            !effects[..*ordinal].iter().any(|prior| {
                prior.request.catalog_column_ordinal == effect.request.catalog_column_ordinal
            })
        })
        .count();
    let active_count = u32::try_from(active_count)
        .map_err(|_| codec_error("active sequence column count overflows"))?;
    if active_count == 0 {
        return Err(codec_error("sequence section has no active columns"));
    }

    let mut prior_key = None;
    let mut prior_transition = None;
    for (ordinal, effect) in effects.iter().enumerate() {
        let request = effect.request.view();
        if request.statement_ordinal != parent.statement_ordinal {
            return Err(codec_error("sequence request statement identity drifted"));
        }
        let key = (request.row_ordinal, request.catalog_column_ordinal);
        if prior_key.is_some_and(|prior| prior >= key) {
            return Err(codec_error(
                "sequence section is not row-major/catalog ordered",
            ));
        }
        prior_key = Some(key);

        let slot = effects
            .iter()
            .enumerate()
            .filter(|(candidate_ordinal, candidate)| {
                candidate.request.catalog_column_ordinal < request.catalog_column_ordinal
                    && !effects[..*candidate_ordinal].iter().any(|prior| {
                        prior.request.catalog_column_ordinal
                            == candidate.request.catalog_column_ordinal
                    })
            })
            .count();
        let slot =
            u32::try_from(slot).map_err(|_| codec_error("active sequence slot overflows"))?;
        let local = request
            .row_ordinal
            .checked_mul(active_count)
            .and_then(|base| base.checked_add(slot))
            .ok_or_else(|| codec_error("sequence local expression ordinal overflows"))?;
        let absolute = parent
            .expression_base
            .checked_add(local)
            .ok_or_else(|| codec_error("sequence absolute expression ordinal overflows"))?;
        if request.expression_ordinal != local
            || effect.request.absolute_expression_ordinal != absolute
        {
            return Err(codec_error("sequence expression geometry is noncanonical"));
        }

        for prior in &effects[..ordinal] {
            let previous = prior.request.view();
            if previous.catalog_column_ordinal == request.catalog_column_ordinal
                && (previous.column_id != request.column_id
                    || previous.sequence_oid != request.sequence_oid
                    || previous.sequence_source_name != request.sequence_source_name
                    || previous.sequence_effective_name != request.sequence_effective_name)
            {
                return Err(codec_error("sequence column identity drifted"));
            }
            if previous.sequence_oid == request.sequence_oid
                && !same_sequence_identity(prior, effect)
            {
                return Err(codec_error("sequence OID mode or identity drifted"));
            }
            if previous.sequence_effective_name == request.sequence_effective_name
                && previous.sequence_oid != request.sequence_oid
            {
                return Err(codec_error("sequence effective-name identity drifted"));
            }
        }

        match effect.kind {
            DecodedEffectKind::Published {
                transition_txn_id, ..
            } => {
                if transition_txn_id == 0
                    || transition_txn_id == parent.txn_id
                    || prior_transition.is_some_and(|prior| prior >= transition_txn_id)
                {
                    return Err(codec_error("published sequence transition order drifted"));
                }
                prior_transition = Some(transition_txn_id);
            }
            DecodedEffectKind::Private { owner, .. } => {
                if parent.autocommit {
                    return Err(codec_error(
                        "autocommit sequence section has private effect",
                    ));
                }
                if owner.statement_ordinal >= parent.statement_ordinal.as_u32() {
                    return Err(codec_error(
                        "private sequence owner is not before parent statement",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn same_sequence_identity(left: &DecodedEffect, right: &DecodedEffect) -> bool {
    if left.request.effective_name != right.request.effective_name {
        return false;
    }
    match (&left.kind, &right.kind) {
        (DecodedEffectKind::Published { .. }, DecodedEffectKind::Published { .. }) => true,
        (
            DecodedEffectKind::Private {
                lifetime_origin: left_origin,
                owner: left_owner,
                ..
            },
            DecodedEffectKind::Private {
                lifetime_origin: right_origin,
                owner: right_owner,
                ..
            },
        ) => left_origin == right_origin && left_owner == right_owner,
        _ => false,
    }
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
