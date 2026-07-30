//! Scalar-only views over a strictly validated decoded typed-INSERT model.
//!
//! This leaf deliberately exposes only the S2/S5 binding facts needed by a future aggregate
//! replay validator. It cannot expose decoded vectors, names, batches, plans, mutation state, or
//! any conversion into a live write path.

use super::*;

/// Target identity retained by one canonical typed-INSERT record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedTypedInsertTargetFacts {
    pub(crate) oid: u32,
    pub(crate) schema_digest: gpu_db_wal::CanonicalDigest,
}

/// RETURNING geometry retained by one canonical typed-INSERT record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedReturningLayoutFacts {
    pub(crate) digest: gpu_db_wal::CanonicalDigest,
    pub(crate) row_count: u32,
    pub(crate) column_count: u32,
    pub(crate) cell_count: u64,
}

/// Scalar facts which bind a decoded typed record to its aggregate statement directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedTypedInsertRecordFacts {
    pub(crate) typed_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) target: DecodedTypedInsertTargetFacts,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) row_count: u32,
    pub(crate) column_count: u32,
    pub(crate) returning: DecodedReturningLayoutFacts,
    pub(crate) sequence_effect_count: u32,
}

/// The one admitted sequence parent shared by a nonempty decoded effect vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedSequenceParentFacts {
    pub(crate) txn_id: TxnId,
    pub(crate) autocommit: bool,
    pub(crate) request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) expression_ordinal_base: u32,
}

/// Stable scalar identity of one canonical sequence-effect request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedSequenceRequestFacts {
    pub(crate) effect_ordinal: u32,
    pub(crate) target_table_oid: u32,
    pub(crate) row_ordinal: u32,
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) column_id: u32,
    pub(crate) sequence_oid: u32,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) expression_ordinal: u32,
    pub(crate) absolute_expression_ordinal: u32,
}

/// Read-only kind facts for one sequence effect. Published values are independently durable;
/// private state remains model-owned, so its view contains only the request-binding digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodedSequenceEffectKindFacts {
    Published {
        transition_txn_id: TxnId,
        input_digest: gpu_db_wal::CanonicalDigest,
        returned_value: i64,
    },
    Private {
        input_digest: gpu_db_wal::CanonicalDigest,
    },
}

/// Scalar-only record of one canonical sequence effect, in canonical effect order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodedSequenceEffectFacts {
    pub(crate) request: DecodedSequenceRequestFacts,
    /// The resolved typed-vector value. This equals `returned_value` for a published effect.
    pub(crate) resolved_value: i64,
    pub(crate) kind: DecodedSequenceEffectKindFacts,
}

pub(super) fn record_facts(model: &DecodedModel) -> DecodedTypedInsertRecordFacts {
    DecodedTypedInsertRecordFacts {
        typed_statement_digest: model.typed_statement_digest,
        target: DecodedTypedInsertTargetFacts {
            oid: model.target.oid,
            schema_digest: model.target.schema_digest,
        },
        statement_ordinal: model.target.statement_ordinal,
        row_count: model.target.rows,
        column_count: model.target.column_count,
        returning: DecodedReturningLayoutFacts {
            digest: model.returning_digest,
            row_count: model.returning.rows,
            column_count: model.returning.columns,
            cell_count: model.returning.cells,
        },
        sequence_effect_count: u32::try_from(model.effects.effects.len())
            .expect("validated decoded sequence effect count fits u32"),
    }
}

pub(super) fn sequence_parent_facts(model: &DecodedModel) -> Option<DecodedSequenceParentFacts> {
    model.effects.parent.map(parent_facts)
}

pub(super) fn sequence_effect_facts(
    model: &DecodedModel,
) -> impl ExactSizeIterator<Item = DecodedSequenceEffectFacts> + '_ {
    model.effects.effects.iter().map(effect_facts)
}

fn parent_facts(parent: DecodedParent) -> DecodedSequenceParentFacts {
    DecodedSequenceParentFacts {
        txn_id: parent.txn_id,
        autocommit: parent.autocommit,
        request_digest: parent.request_digest,
        statement_ordinal: parent.statement_ordinal,
        expression_ordinal_base: parent.expression_base,
    }
}

fn effect_facts(effect: &DecodedEffect) -> DecodedSequenceEffectFacts {
    let request = &effect.request;
    let kind = match &effect.kind {
        DecodedEffectKind::Published {
            transition_txn_id,
            input_digest,
            returned_value,
        } => DecodedSequenceEffectKindFacts::Published {
            transition_txn_id: *transition_txn_id,
            input_digest: *input_digest,
            returned_value: *returned_value,
        },
        DecodedEffectKind::Private { input_digest, .. } => {
            // The full private chain is retained by `DecodedModel`; S5 only needs the binding
            // digest while S2 still owns all private transition state.
            DecodedSequenceEffectKindFacts::Private {
                input_digest: *input_digest,
            }
        }
    };
    DecodedSequenceEffectFacts {
        request: DecodedSequenceRequestFacts {
            effect_ordinal: request.ordinal,
            target_table_oid: request.target_table_oid,
            row_ordinal: request.row_ordinal,
            catalog_column_ordinal: request.catalog_column_ordinal,
            column_id: request.column_id,
            sequence_oid: request.sequence_oid,
            statement_ordinal: request.statement_ordinal,
            expression_ordinal: request.expression_ordinal,
            absolute_expression_ordinal: request.absolute_expression_ordinal,
        },
        resolved_value: request.value,
        kind,
    }
}
