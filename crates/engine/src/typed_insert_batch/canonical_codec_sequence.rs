//! Canonical sequence-effect evidence for the inert typed-INSERT record.
//!
//! This leaf owns no sequence state. It only validates and serializes immutable receipt/private
//! evidence that was supplied by the current sequence owner.

use super::validation::{self, SequenceSectionEntry, SequenceSectionKind};
use super::*;
use crate::typed_insert_batch::sequence_defaults::effects::{
    CanonicalSequenceEffectKindView, CanonicalSequenceEffectView, CanonicalSequenceParentView,
    CanonicalSequenceRequestView,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn encode_sequence_effects(
    bindings: &[sequence_defaults::SequenceDefaultBinding],
    expected_request_digest: gpu_db_wal::CanonicalDigest,
    expected_statement_ordinal: InsertStatementOrdinal,
    out: &mut Writer,
) -> Result<(), EngineError> {
    let views = bindings
        .iter()
        .map(|binding| binding.canonical_view())
        .collect::<Vec<_>>();
    let parent = sequence_parent(&views, expected_request_digest, expected_statement_ordinal)?;
    out.bool(parent.is_some())?;
    if let Some(parent) = parent {
        let entries = sequence_section_entries(&views, parent)?;
        validation::validate_sequence_section(parent, &entries)?;
        append_sequence_parent(out, parent)?;
    }
    out.u32(checked_u32(views.len(), "sequence effect count")?)?;
    let mut published_ids = BTreeSet::new();
    let mut private_chains = BTreeMap::new();
    let mut previous = None;
    for (ordinal, view) in views.iter().copied().enumerate() {
        out.u32(u32::try_from(ordinal).map_err(|_| codec_error("sequence ordinal"))?)?;
        append_sequence_effect(
            out,
            view,
            parent.expect("nonempty effect has parent"),
            &mut published_ids,
            &mut private_chains,
            previous,
        )?;
        previous = Some(view.request);
    }
    Ok(())
}

fn sequence_section_entries<'a>(
    views: &[CanonicalSequenceEffectView<'a>],
    parent: CanonicalSequenceParentView,
) -> Result<Vec<SequenceSectionEntry<'a>>, EngineError> {
    views
        .iter()
        .copied()
        .map(|view| {
            let absolute_expression_ordinal = parent
                .expression_ordinal_base
                .checked_add(view.request.expression_ordinal)
                .ok_or_else(|| codec_error("sequence absolute expression ordinal overflows"))?;
            let kind = match view.kind {
                CanonicalSequenceEffectKindView::Published {
                    transition_txn_id, ..
                } => SequenceSectionKind::Published { transition_txn_id },
                CanonicalSequenceEffectKindView::Private {
                    lifetime_origin,
                    owner_kind,
                    owner_statement_ordinal,
                    owner_statement_digest,
                    owner_creator_catalog_column_ordinal,
                    ..
                } => SequenceSectionKind::Private {
                    lifetime_origin,
                    owner: PrivateOwner {
                        kind: owner_kind,
                        statement_ordinal: owner_statement_ordinal,
                        statement_digest: owner_statement_digest,
                        creator_catalog_column_ordinal: owner_creator_catalog_column_ordinal,
                    },
                },
            };
            Ok(SequenceSectionEntry {
                request: view.request,
                absolute_expression_ordinal,
                kind,
            })
        })
        .collect()
}

fn append_sequence_effect(
    out: &mut Writer,
    view: CanonicalSequenceEffectView<'_>,
    parent: CanonicalSequenceParentView,
    published_ids: &mut BTreeSet<TxnId>,
    private_chains: &mut BTreeMap<u32, PrivateChain>,
    previous: Option<CanonicalSequenceRequestView<'_>>,
) -> Result<(), EngineError> {
    validate_sequence_request(view.request, parent, previous)?;
    append_sequence_request(
        out,
        view.request.target_table_oid,
        view.request.row_ordinal,
        view.request.catalog_column_ordinal,
        view.request.column_id,
        view.request.sequence_oid,
        view.request.sequence_source_name,
        view.request.sequence_effective_name,
        view.request.statement_ordinal,
        view.request.expression_ordinal,
    )?;
    let absolute_expression_ordinal = parent
        .expression_ordinal_base
        .checked_add(view.request.expression_ordinal)
        .ok_or_else(|| codec_error("sequence absolute expression ordinal overflows"))?;
    out.u32(absolute_expression_ordinal)?;
    let descriptor = crate::sequence_descriptor_digest(
        view.request.sequence_oid,
        view.request.sequence_effective_name,
    );
    out.digest(&descriptor)?;
    out.i64(view.value)?;
    match view.kind {
        CanonicalSequenceEffectKindView::Published {
            parent: effect_parent,
            transition_txn_id,
            input_digest,
            returned_value,
        } => {
            if effect_parent != parent
                || transition_txn_id == 0
                || !published_ids.insert(transition_txn_id)
                || returned_value != view.value
                || input_digest
                    != sequence_input_digest(parent, view.request, absolute_expression_ordinal)
            {
                return Err(codec_error("published sequence effect identity drifted"));
            }
            out.u8(1)?;
            out.u64(transition_txn_id)?;
            out.digest(&input_digest)?;
            out.i64(returned_value)?;
        }
        CanonicalSequenceEffectKindView::Private {
            parent: effect_parent,
            prior_last_value,
            prior_is_called,
            next_last_value,
            next_is_called,
            lifetime_origin,
            owner_kind,
            owner_statement_ordinal,
            owner_statement_digest,
            owner_creator_catalog_column_ordinal,
            predecessor_tag,
            predecessor_digest,
            input_digest,
            descriptor_digest,
            child_digest,
            outcome_digest,
        } => {
            let owner = PrivateOwner {
                kind: owner_kind,
                statement_ordinal: owner_statement_ordinal,
                statement_digest: owner_statement_digest,
                creator_catalog_column_ordinal: owner_creator_catalog_column_ordinal,
            };
            let predecessor =
                PrivatePredecessor::from_effect(predecessor_tag, predecessor_digest, owner)?;
            if effect_parent != parent {
                return Err(codec_error("private sequence effect parent drifted"));
            }
            let child = private_child_digest(
                parent,
                view.request,
                absolute_expression_ordinal,
                input_digest,
                descriptor,
                lifetime_origin,
                (prior_last_value, prior_is_called),
                predecessor,
            )?;
            let outcome = private_outcome_digest(
                child,
                owner,
                view.value,
                (next_last_value, next_is_called),
            )?;
            if child != child_digest || outcome != outcome_digest {
                return Err(codec_error("private sequence digest witness drifted"));
            }
            validate_private_effect(
                view,
                parent,
                absolute_expression_ordinal,
                descriptor,
                PrivateEffectEvidence {
                    prior_last_value,
                    prior_is_called,
                    next_last_value,
                    next_is_called,
                    lifetime_origin,
                    owner,
                    predecessor,
                    input_digest,
                    descriptor_digest,
                    child_digest,
                    outcome_digest,
                },
                private_chains,
            )?;
            out.u8(2)?;
            out.i64(prior_last_value)?;
            out.bool(prior_is_called)?;
            out.i64(next_last_value)?;
            out.bool(next_is_called)?;
            out.u8(lifetime_origin)?;
            append_private_owner(out, owner)?;
            append_private_predecessor(out, predecessor)?;
            out.digest(&input_digest)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct PrivateOwner {
    pub(super) kind: u8,
    pub(super) statement_ordinal: u32,
    pub(super) statement_digest: gpu_db_wal::CanonicalDigest,
    pub(super) creator_catalog_column_ordinal: Option<u32>,
}

#[derive(Clone, Copy)]
pub(super) enum PrivatePredecessor {
    Lifecycle(PrivateOwner),
    Outcome(gpu_db_wal::CanonicalDigest),
}

#[derive(Clone, Copy)]
pub(super) struct PrivateChain {
    state: (i64, bool),
    owner: PrivateOwner,
    outcome: gpu_db_wal::CanonicalDigest,
}

pub(super) struct PrivateEffectEvidence {
    pub(super) prior_last_value: i64,
    pub(super) prior_is_called: bool,
    pub(super) next_last_value: i64,
    pub(super) next_is_called: bool,
    pub(super) lifetime_origin: u8,
    pub(super) owner: PrivateOwner,
    pub(super) predecessor: PrivatePredecessor,
    pub(super) input_digest: gpu_db_wal::CanonicalDigest,
    pub(super) descriptor_digest: gpu_db_wal::CanonicalDigest,
    pub(super) child_digest: gpu_db_wal::CanonicalDigest,
    pub(super) outcome_digest: gpu_db_wal::CanonicalDigest,
}

impl PrivatePredecessor {
    pub(super) fn from_effect(
        tag: u8,
        digest: gpu_db_wal::CanonicalDigest,
        owner: PrivateOwner,
    ) -> Result<Self, EngineError> {
        match tag {
            1 if digest == owner.statement_digest => Ok(Self::Lifecycle(owner)),
            2 if !zero_digest(digest) => Ok(Self::Outcome(digest)),
            _ => Err(codec_error("private predecessor tag or digest is invalid")),
        }
    }
}

fn sequence_parent(
    views: &[CanonicalSequenceEffectView<'_>],
    expected_request_digest: gpu_db_wal::CanonicalDigest,
    expected_statement_ordinal: InsertStatementOrdinal,
) -> Result<Option<CanonicalSequenceParentView>, EngineError> {
    let Some(first) = views.first() else {
        return Ok(None);
    };
    let parent = match first.kind {
        CanonicalSequenceEffectKindView::Published { parent, .. }
        | CanonicalSequenceEffectKindView::Private { parent, .. } => parent,
    };
    validate_sequence_parent(parent, expected_request_digest, expected_statement_ordinal)?;
    if views.iter().any(|view| match view.kind {
        CanonicalSequenceEffectKindView::Published { parent: other, .. }
        | CanonicalSequenceEffectKindView::Private { parent: other, .. } => other != parent,
    }) {
        return Err(codec_error(
            "sequence effects do not share one admitted parent",
        ));
    }
    Ok(Some(parent))
}

pub(super) fn validate_sequence_parent(
    parent: CanonicalSequenceParentView,
    expected_request_digest: gpu_db_wal::CanonicalDigest,
    expected_statement_ordinal: InsertStatementOrdinal,
) -> Result<(), EngineError> {
    if parent.txn_id == 0
        || zero_digest(parent.request_digest)
        || parent.request_digest != expected_request_digest
        || parent.statement_ordinal != expected_statement_ordinal
    {
        return Err(codec_error(
            "sequence parent is not bound to this typed statement",
        ));
    }
    Ok(())
}

pub(super) fn validate_sequence_request(
    request: CanonicalSequenceRequestView<'_>,
    parent: CanonicalSequenceParentView,
    previous: Option<CanonicalSequenceRequestView<'_>>,
) -> Result<(), EngineError> {
    if request.target_table_oid == 0
        || request.column_id == 0
        || request.sequence_oid == 0
        || request.sequence_source_name.is_empty()
        || request.sequence_effective_name.is_empty()
        || request.statement_ordinal != parent.statement_ordinal
        || parent
            .expression_ordinal_base
            .checked_add(request.expression_ordinal)
            .is_none()
    {
        return Err(codec_error("sequence request identity is invalid"));
    }
    if let Some(previous) = previous {
        if (
            previous.row_ordinal,
            previous.catalog_column_ordinal,
            previous.expression_ordinal,
        ) >= (
            request.row_ordinal,
            request.catalog_column_ordinal,
            request.expression_ordinal,
        ) {
            return Err(codec_error(
                "sequence requests are not row-major/catalog ordered",
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_private_effect(
    view: CanonicalSequenceEffectView<'_>,
    parent: CanonicalSequenceParentView,
    absolute: u32,
    descriptor: gpu_db_wal::CanonicalDigest,
    evidence: PrivateEffectEvidence,
    chains: &mut BTreeMap<u32, PrivateChain>,
) -> Result<(), EngineError> {
    let expected = if evidence.prior_is_called {
        evidence.prior_last_value.checked_add(1)
    } else {
        Some(evidence.prior_last_value)
    }
    .ok_or_else(|| codec_error("private sequence advance overflows"))?;
    if expected != view.value
        || evidence.next_last_value != expected
        || !evidence.next_is_called
        || evidence.next_last_value != view.value
        || !matches!(evidence.lifetime_origin, 1 | 2)
        || !matches!(evidence.owner.kind, 1..=3)
        || zero_digest(evidence.owner.statement_digest)
        || !private_owner_compatible(evidence.lifetime_origin, evidence.owner)
        || evidence.input_digest != sequence_input_digest(parent, view.request, absolute)
        || evidence.descriptor_digest != descriptor
    {
        return Err(codec_error("private sequence evidence is inconsistent"));
    }
    if let Some(previous) = chains.get(&view.request.sequence_oid) {
        if (evidence.prior_last_value, evidence.prior_is_called) != previous.state
            || evidence.owner != previous.owner
            || !matches!(evidence.predecessor, PrivatePredecessor::Outcome(digest) if digest == previous.outcome)
        {
            return Err(codec_error("private sequence predecessor chain drifted"));
        }
    } else if !matches!(evidence.predecessor, PrivatePredecessor::Lifecycle(owner) if owner == evidence.owner)
    {
        return Err(codec_error(
            "first private sequence effect lacks lifecycle predecessor",
        ));
    }
    let child = private_child_digest(
        parent,
        view.request,
        absolute,
        evidence.input_digest,
        descriptor,
        evidence.lifetime_origin,
        (evidence.prior_last_value, evidence.prior_is_called),
        evidence.predecessor,
    )?;
    let outcome = private_outcome_digest(
        child,
        evidence.owner,
        view.value,
        (evidence.next_last_value, evidence.next_is_called),
    )?;
    if child != evidence.child_digest || outcome != evidence.outcome_digest {
        return Err(codec_error("private sequence digest witness drifted"));
    }
    chains.insert(
        view.request.sequence_oid,
        PrivateChain {
            state: (evidence.next_last_value, evidence.next_is_called),
            owner: evidence.owner,
            outcome,
        },
    );
    Ok(())
}

fn private_owner_compatible(origin: u8, owner: PrivateOwner) -> bool {
    !(matches!((origin, owner.kind), (1, 1))
        || matches!(owner.kind, 2 | 3) && owner.creator_catalog_column_ordinal.is_some())
}

pub(super) fn sequence_input_digest(
    parent: CanonicalSequenceParentView,
    request: CanonicalSequenceRequestView<'_>,
    absolute: u32,
) -> gpu_db_wal::CanonicalDigest {
    crate::sequence_value_input_digest(crate::SequenceValueInput {
        parent_txn_id: parent.txn_id,
        parent_autocommit: parent.autocommit,
        statement_ordinal: parent.statement_ordinal.as_u32(),
        expression_ordinal: absolute,
        parent_request_digest: parent.request_digest,
        source_name: request.sequence_source_name,
        operation: crate::BinarySequenceValueOperation::Default,
        set_value: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn private_child_digest(
    parent: CanonicalSequenceParentView,
    request: CanonicalSequenceRequestView<'_>,
    absolute: u32,
    input: gpu_db_wal::CanonicalDigest,
    descriptor: gpu_db_wal::CanonicalDigest,
    lifetime: u8,
    prior: (i64, bool),
    predecessor: PrivatePredecessor,
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let mut body = Writer::new(MAX_RECORD_BYTES);
    body.bytes(b"GPUDBPRIVATESEQCHILD1")?;
    append_sequence_parent(&mut body, parent)?;
    append_sequence_request(
        &mut body,
        request.target_table_oid,
        request.row_ordinal,
        request.catalog_column_ordinal,
        request.column_id,
        request.sequence_oid,
        request.sequence_source_name,
        request.sequence_effective_name,
        request.statement_ordinal,
        request.expression_ordinal,
    )?;
    body.u32(absolute)?;
    body.digest(&input)?;
    body.digest(&descriptor)?;
    body.u8(lifetime)?;
    body.i64(prior.0)?;
    body.bool(prior.1)?;
    append_private_predecessor(&mut body, predecessor)?;
    Ok(gpu_db_wal::canonical_request_digest(&body.finish()))
}

pub(super) fn private_outcome_digest(
    child: gpu_db_wal::CanonicalDigest,
    owner: PrivateOwner,
    output: i64,
    next: (i64, bool),
) -> Result<gpu_db_wal::CanonicalDigest, EngineError> {
    let mut body = Writer::new(MAX_RECORD_BYTES);
    body.bytes(b"GPUDBPRIVATESEQOUTCOME1")?;
    body.digest(&child)?;
    append_private_owner(&mut body, owner)?;
    body.i64(output)?;
    body.i64(next.0)?;
    body.bool(next.1)?;
    Ok(gpu_db_wal::canonical_request_digest(&body.finish()))
}

pub(super) fn append_private_owner(
    out: &mut Writer,
    owner: PrivateOwner,
) -> Result<(), EngineError> {
    out.u8(owner.kind)?;
    out.u32(owner.statement_ordinal)?;
    out.digest(&owner.statement_digest)?;
    out.option_u32(owner.creator_catalog_column_ordinal)
}

pub(super) fn append_private_predecessor(
    out: &mut Writer,
    predecessor: PrivatePredecessor,
) -> Result<(), EngineError> {
    match predecessor {
        PrivatePredecessor::Lifecycle(owner) => {
            out.u8(1)?;
            append_private_owner(out, owner)
        }
        PrivatePredecessor::Outcome(digest) => {
            out.u8(2)?;
            out.digest(&digest)
        }
    }
}
