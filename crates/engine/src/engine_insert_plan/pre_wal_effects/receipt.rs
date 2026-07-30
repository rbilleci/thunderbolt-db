//! Move-only synthetic receipt inputs and compact inert seal evidence.
//!
//! WRITE-001 keeps these types production-compiled but only permits construction in tests. They
//! carry no live sequence transition, session, WAL, or result-route authority.

/// One caller-supplied stand-in for an already durable published sequence transition. The inert
/// terminal verifies every field against its classifier-planned request before it can bind a
/// typed default value. In tests this proves only structural binding, never durability itself.
#[allow(dead_code)]
pub(super) struct PublishedReceiptInput {
    pub(super) transition_txn_id: crate::TxnId,
    pub(super) returned_value: i64,
    pub(super) input_digest: gpu_db_wal::CanonicalDigest,
}

#[cfg(test)]
impl PublishedReceiptInput {
    pub(super) fn for_test(
        transition_txn_id: crate::TxnId,
        returned_value: i64,
        input_digest: gpu_db_wal::CanonicalDigest,
    ) -> Self {
        Self {
            transition_txn_id,
            returned_value,
            input_digest,
        }
    }
}

/// The caller owns the ordered published receipt list. Private entries never appear here: the
/// terminal derives their binding solely from the captured classifier plan.
#[allow(dead_code)]
pub(super) struct SequenceReceiptBundle {
    pub(super) published: Box<[PublishedReceiptInput]>,
}

#[cfg(test)]
impl SequenceReceiptBundle {
    pub(super) fn for_test(published: Vec<PublishedReceiptInput>) -> Self {
        Self {
            published: published.into(),
        }
    }
}

/// One compact scalar output in request order. It is intentionally not a session update and does
/// not contain a materialized value vector.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct SequenceOutputEvidence {
    pub(super) sequence_oid: u32,
    pub(super) row_ordinal: u32,
    pub(super) catalog_column_ordinal: u32,
    pub(super) column_id: u32,
    pub(super) local_expression_ordinal: u32,
    pub(super) absolute_expression_ordinal: u32,
    pub(super) input_digest: gpu_db_wal::CanonicalDigest,
    pub(super) descriptor_digest: gpu_db_wal::CanonicalDigest,
    pub(super) returned_value: i32,
    pub(super) transition: SequenceTransitionEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum SequenceTransitionEvidence {
    Published {
        transition_txn_id: crate::TxnId,
    },
    Private {
        lifetime_origin: u8,
        prior_state: (i64, bool),
        next_state: (i64, bool),
        owner: PrivateOwnerEvidence,
        predecessor: PrivatePredecessorEvidence,
        child_digest: gpu_db_wal::CanonicalDigest,
        outcome_digest: gpu_db_wal::CanonicalDigest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) struct PrivateOwnerEvidence {
    pub(super) kind: u8,
    pub(super) statement_ordinal: u32,
    pub(super) statement_digest: gpu_db_wal::CanonicalDigest,
    pub(super) creator_catalog_column_ordinal: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum PrivatePredecessorEvidence {
    Lifecycle(PrivateOwnerEvidence),
    PlannedOutcome(gpu_db_wal::CanonicalDigest),
}

/// The validated materialization input that remains entirely private to the terminal. The sealed
/// batch is never a field here; it is dropped inside the terminal after seal succeeds.
#[allow(dead_code)]
pub(super) struct SequenceSealBindingBundle {
    pub(super) bindings: crate::typed_insert_batch::sequence_defaults::SequenceDefaultBindings,
    pub(super) outputs: Box<[SequenceOutputEvidence]>,
}
