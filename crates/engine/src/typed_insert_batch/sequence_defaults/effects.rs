//! Move-only sequence-effect receipts checked against one admitted parent context.

use super::*;

/// The admitted parent statement identity a sequence owner must carry into seal. Semantic
/// preparation deliberately does not invent transaction authority; empty bindings are valid only
/// when no sequence request exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SequenceDefaultParentContext {
    parent_txn_id: TxnId,
    parent_autocommit: bool,
    parent_request_digest: gpu_db_wal::CanonicalDigest,
    statement_ordinal: InsertStatementOrdinal,
    expression_ordinal_base: u32,
}

impl SequenceDefaultParentContext {
    #[cfg(test)]
    pub(crate) fn for_test(
        parent_txn_id: TxnId,
        parent_autocommit: bool,
        parent_request_digest: gpu_db_wal::CanonicalDigest,
        statement_ordinal: InsertStatementOrdinal,
        expression_ordinal_base: u32,
    ) -> Self {
        Self {
            parent_txn_id,
            parent_autocommit,
            parent_request_digest,
            statement_ordinal,
            expression_ordinal_base,
        }
    }

    fn expression_ordinal(&self, request: &SequenceDefaultRequest) -> Option<u32> {
        self.expression_ordinal_base
            .checked_add(request.expression_ordinal)
    }

    fn admits(&self, request: &SequenceDefaultRequest) -> bool {
        self.parent_txn_id != 0
            && self.parent_request_digest != [0; 32]
            && self.statement_ordinal == request.statement_ordinal
            && self.expression_ordinal(request).is_some()
    }
}

/// A published transition is independently durable. A private state instead preserves the exact
/// next-state transition the future transaction envelope must own; neither effect is cloneable.
#[derive(Debug)]
pub(crate) struct PublishedSequenceReceipt {
    parent: SequenceDefaultParentContext,
    statement_ordinal: InsertStatementOrdinal,
    expression_ordinal: u32,
    target_table_oid: u32,
    target_column_id: u32,
    staging_row_ordinal: u32,
    transition_txn_id: TxnId,
    input_digest: gpu_db_wal::CanonicalDigest,
    sequence_oid: u32,
    returned_value: i64,
}

#[allow(dead_code)] // WRITE-001 retains this complete provenance until a live owner is designed.
#[derive(Debug)]
pub(crate) struct PrivateSequenceState {
    parent: SequenceDefaultParentContext,
    statement_ordinal: InsertStatementOrdinal,
    expression_ordinal: u32,
    target_table_oid: u32,
    target_column_id: u32,
    staging_row_ordinal: u32,
    sequence_oid: u32,
    prior_last_value: i64,
    prior_is_called: bool,
    next_last_value: i64,
    next_is_called: bool,
    /// Full classifier provenance retained through seal. The generic seal contract checks only
    /// local request/value geometry; WRITE-001 verifies this complete planning evidence before it
    /// ever calls that mutating contract.
    private_lifetime_origin: u8,
    private_owner_kind: u8,
    private_owner_statement_ordinal: u32,
    private_owner_statement_digest: gpu_db_wal::CanonicalDigest,
    private_owner_creator_catalog_column_ordinal: Option<u32>,
    private_predecessor_tag: u8,
    private_predecessor_digest: gpu_db_wal::CanonicalDigest,
    private_input_digest: gpu_db_wal::CanonicalDigest,
    private_descriptor_digest: gpu_db_wal::CanonicalDigest,
    private_child_digest: gpu_db_wal::CanonicalDigest,
    private_outcome_digest: gpu_db_wal::CanonicalDigest,
}

/// Complete private planning provenance retained by the inert effect terminal. It carries only
/// scalars and canonical digests; it is neither a sequence state mutator nor a durable record.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct PrivateSequencePlanningEvidence {
    lifetime_origin: u8,
    owner_kind: u8,
    owner_statement_ordinal: u32,
    owner_statement_digest: gpu_db_wal::CanonicalDigest,
    owner_creator_catalog_column_ordinal: Option<u32>,
    predecessor_tag: u8,
    predecessor_digest: gpu_db_wal::CanonicalDigest,
    input_digest: gpu_db_wal::CanonicalDigest,
    descriptor_digest: gpu_db_wal::CanonicalDigest,
    child_digest: gpu_db_wal::CanonicalDigest,
    outcome_digest: gpu_db_wal::CanonicalDigest,
}

/// Borrowed, scalar-only sequence evidence for the inert canonical typed-INSERT codec.  This is
/// deliberately a read-only view: it neither exposes the move-only binding fields for mutation
/// nor creates a second sequence-effect owner.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct CanonicalSequenceParentView {
    pub(crate) txn_id: TxnId,
    pub(crate) autocommit: bool,
    pub(crate) request_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) expression_ordinal_base: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct CanonicalSequenceRequestView<'a> {
    pub(crate) target_table_oid: u32,
    pub(crate) row_ordinal: u32,
    pub(crate) catalog_column_ordinal: u32,
    pub(crate) column_id: u32,
    pub(crate) sequence_oid: u32,
    pub(crate) sequence_source_name: &'a str,
    pub(crate) sequence_effective_name: &'a str,
    pub(crate) statement_ordinal: InsertStatementOrdinal,
    pub(crate) expression_ordinal: u32,
}

#[derive(Clone, Copy)]
pub(crate) enum CanonicalSequenceEffectKindView {
    Published {
        parent: CanonicalSequenceParentView,
        transition_txn_id: TxnId,
        input_digest: gpu_db_wal::CanonicalDigest,
        returned_value: i64,
    },
    Private {
        parent: CanonicalSequenceParentView,
        prior_last_value: i64,
        prior_is_called: bool,
        next_last_value: i64,
        next_is_called: bool,
        lifetime_origin: u8,
        owner_kind: u8,
        owner_statement_ordinal: u32,
        owner_statement_digest: gpu_db_wal::CanonicalDigest,
        owner_creator_catalog_column_ordinal: Option<u32>,
        predecessor_tag: u8,
        predecessor_digest: gpu_db_wal::CanonicalDigest,
        input_digest: gpu_db_wal::CanonicalDigest,
        descriptor_digest: gpu_db_wal::CanonicalDigest,
        child_digest: gpu_db_wal::CanonicalDigest,
        outcome_digest: gpu_db_wal::CanonicalDigest,
    },
}

#[derive(Clone, Copy)]
pub(crate) struct CanonicalSequenceEffectView<'a> {
    pub(crate) request: CanonicalSequenceRequestView<'a>,
    pub(crate) value: i64,
    pub(crate) kind: CanonicalSequenceEffectKindView,
}

#[cfg(test)]
impl PrivateSequencePlanningEvidence {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exact_for_test(
        lifetime_origin: u8,
        owner_kind: u8,
        owner_statement_ordinal: u32,
        owner_statement_digest: gpu_db_wal::CanonicalDigest,
        owner_creator_catalog_column_ordinal: Option<u32>,
        predecessor_tag: u8,
        predecessor_digest: gpu_db_wal::CanonicalDigest,
        input_digest: gpu_db_wal::CanonicalDigest,
        descriptor_digest: gpu_db_wal::CanonicalDigest,
        child_digest: gpu_db_wal::CanonicalDigest,
        outcome_digest: gpu_db_wal::CanonicalDigest,
    ) -> Self {
        Self {
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
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum SequenceDefaultEffect {
    Published(PublishedSequenceReceipt),
    Private(PrivateSequenceState),
}

#[derive(Debug)]
pub(crate) struct SequenceDefaultBinding {
    request: SequenceDefaultRequest,
    pub(super) value: i64,
    effect: SequenceDefaultEffect,
}

impl SequenceDefaultBinding {
    pub(crate) fn canonical_view(&self) -> CanonicalSequenceEffectView<'_> {
        let request = CanonicalSequenceRequestView {
            target_table_oid: self.request.target_table_oid,
            row_ordinal: self.request.row_ordinal,
            catalog_column_ordinal: self.request.catalog_column_ordinal,
            column_id: self.request.column_id,
            sequence_oid: self.request.sequence_oid,
            sequence_source_name: self.request.sequence_source_name.as_ref(),
            sequence_effective_name: self.request.sequence_effective_name.as_ref(),
            statement_ordinal: self.request.statement_ordinal,
            expression_ordinal: self.request.expression_ordinal,
        };
        let parent_view = |parent: &SequenceDefaultParentContext| CanonicalSequenceParentView {
            txn_id: parent.parent_txn_id,
            autocommit: parent.parent_autocommit,
            request_digest: parent.parent_request_digest,
            statement_ordinal: parent.statement_ordinal,
            expression_ordinal_base: parent.expression_ordinal_base,
        };
        let kind = match &self.effect {
            SequenceDefaultEffect::Published(receipt) => {
                CanonicalSequenceEffectKindView::Published {
                    parent: parent_view(&receipt.parent),
                    transition_txn_id: receipt.transition_txn_id,
                    input_digest: receipt.input_digest,
                    returned_value: receipt.returned_value,
                }
            }
            SequenceDefaultEffect::Private(state) => CanonicalSequenceEffectKindView::Private {
                parent: parent_view(&state.parent),
                prior_last_value: state.prior_last_value,
                prior_is_called: state.prior_is_called,
                next_last_value: state.next_last_value,
                next_is_called: state.next_is_called,
                lifetime_origin: state.private_lifetime_origin,
                owner_kind: state.private_owner_kind,
                owner_statement_ordinal: state.private_owner_statement_ordinal,
                owner_statement_digest: state.private_owner_statement_digest,
                owner_creator_catalog_column_ordinal: state
                    .private_owner_creator_catalog_column_ordinal,
                predecessor_tag: state.private_predecessor_tag,
                predecessor_digest: state.private_predecessor_digest,
                input_digest: state.private_input_digest,
                descriptor_digest: state.private_descriptor_digest,
                child_digest: state.private_child_digest,
                outcome_digest: state.private_outcome_digest,
            },
        };
        CanonicalSequenceEffectView {
            request,
            value: self.value,
            kind,
        }
    }

    /// Test-only exact construction for the inert WRITE-001 terminal. Unlike the older semantic
    /// fixtures below, this constructor does not derive an input digest or transition identity.
    #[cfg(test)]
    pub(crate) fn published_exact_for_test(
        request: SequenceDefaultRequest,
        parent: SequenceDefaultParentContext,
        value: i64,
        transition_txn_id: TxnId,
        input_digest: gpu_db_wal::CanonicalDigest,
    ) -> Self {
        let expression_ordinal = parent
            .expression_ordinal(&request)
            .expect("terminal parent expression ordinal is in range");
        Self {
            effect: SequenceDefaultEffect::Published(PublishedSequenceReceipt {
                parent: parent.clone(),
                statement_ordinal: request.statement_ordinal,
                expression_ordinal,
                target_table_oid: request.target_table_oid,
                target_column_id: request.column_id,
                staging_row_ordinal: request.row_ordinal,
                transition_txn_id,
                input_digest,
                sequence_oid: request.sequence_oid,
                returned_value: value,
            }),
            request,
            value,
        }
    }

    /// Test-only exact private construction for the inert WRITE-001 terminal. Every scalar comes
    /// from the stable-OID classifier; this constructor never synthesizes predecessor state.
    #[cfg(test)]
    pub(crate) fn private_exact_for_test(
        request: SequenceDefaultRequest,
        parent: SequenceDefaultParentContext,
        value: i64,
        prior_state: (i64, bool),
        next_state: (i64, bool),
        planning: PrivateSequencePlanningEvidence,
    ) -> Self {
        let expression_ordinal = parent
            .expression_ordinal(&request)
            .expect("terminal parent expression ordinal is in range");
        Self {
            effect: SequenceDefaultEffect::Private(PrivateSequenceState {
                parent,
                statement_ordinal: request.statement_ordinal,
                expression_ordinal,
                target_table_oid: request.target_table_oid,
                target_column_id: request.column_id,
                staging_row_ordinal: request.row_ordinal,
                sequence_oid: request.sequence_oid,
                prior_last_value: prior_state.0,
                prior_is_called: prior_state.1,
                next_last_value: next_state.0,
                next_is_called: next_state.1,
                private_lifetime_origin: planning.lifetime_origin,
                private_owner_kind: planning.owner_kind,
                private_owner_statement_ordinal: planning.owner_statement_ordinal,
                private_owner_statement_digest: planning.owner_statement_digest,
                private_owner_creator_catalog_column_ordinal: planning
                    .owner_creator_catalog_column_ordinal,
                private_predecessor_tag: planning.predecessor_tag,
                private_predecessor_digest: planning.predecessor_digest,
                private_input_digest: planning.input_digest,
                private_descriptor_digest: planning.descriptor_digest,
                private_child_digest: planning.child_digest,
                private_outcome_digest: planning.outcome_digest,
            }),
            request,
            value,
        }
    }

    #[cfg(test)]
    pub(crate) fn published(
        request: SequenceDefaultRequest,
        parent: SequenceDefaultParentContext,
        value: i64,
    ) -> Self {
        let expression_ordinal = parent
            .expression_ordinal(&request)
            .expect("test parent expression ordinal is in range");
        let input_digest = crate::sequence_value_input_digest(crate::SequenceValueInput {
            parent_txn_id: parent.parent_txn_id,
            parent_autocommit: parent.parent_autocommit,
            statement_ordinal: parent.statement_ordinal.as_u32(),
            expression_ordinal,
            parent_request_digest: parent.parent_request_digest,
            source_name: request.sequence_source_name.as_ref(),
            operation: crate::BinarySequenceValueOperation::Default,
            set_value: None,
        });
        Self {
            effect: SequenceDefaultEffect::Published(PublishedSequenceReceipt {
                parent: parent.clone(),
                statement_ordinal: request.statement_ordinal,
                expression_ordinal,
                target_table_oid: request.target_table_oid,
                target_column_id: request.column_id,
                staging_row_ordinal: request.row_ordinal,
                transition_txn_id: TxnId::from(request.expression_ordinal) + 1,
                input_digest,
                sequence_oid: request.sequence_oid,
                returned_value: value,
            }),
            request,
            value,
        }
    }

    #[cfg(test)]
    pub(crate) fn private(
        request: SequenceDefaultRequest,
        parent: SequenceDefaultParentContext,
        value: i64,
    ) -> Self {
        let expression_ordinal = parent
            .expression_ordinal(&request)
            .expect("test parent expression ordinal is in range");
        let input_digest = crate::sequence_value_input_digest(crate::SequenceValueInput {
            parent_txn_id: parent.parent_txn_id,
            parent_autocommit: parent.parent_autocommit,
            statement_ordinal: parent.statement_ordinal.as_u32(),
            expression_ordinal,
            parent_request_digest: parent.parent_request_digest,
            source_name: request.sequence_source_name.as_ref(),
            operation: crate::BinarySequenceValueOperation::Default,
            set_value: None,
        });
        Self {
            effect: SequenceDefaultEffect::Private(PrivateSequenceState {
                parent,
                statement_ordinal: request.statement_ordinal,
                expression_ordinal,
                target_table_oid: request.target_table_oid,
                target_column_id: request.column_id,
                staging_row_ordinal: request.row_ordinal,
                sequence_oid: request.sequence_oid,
                prior_last_value: value.saturating_sub(1),
                prior_is_called: true,
                next_last_value: value,
                next_is_called: true,
                private_lifetime_origin: 2,
                private_owner_kind: 1,
                private_owner_statement_ordinal: 0,
                private_owner_statement_digest: [1; 32],
                private_owner_creator_catalog_column_ordinal: None,
                private_predecessor_tag: 1,
                private_predecessor_digest: [1; 32],
                private_input_digest: input_digest,
                private_descriptor_digest: crate::sequence_descriptor_digest(
                    request.sequence_oid,
                    request.sequence_effective_name.as_ref(),
                ),
                private_child_digest: [1; 32],
                private_outcome_digest: [1; 32],
            }),
            request,
            value,
        }
    }

    #[cfg(test)]
    pub(crate) fn request_mut_for_test(&mut self) -> &mut SequenceDefaultRequest {
        &mut self.request
    }

    #[cfg(test)]
    pub(crate) fn set_effect_value_for_test(&mut self, value: i64) {
        match &mut self.effect {
            SequenceDefaultEffect::Published(receipt) => receipt.returned_value = value,
            SequenceDefaultEffect::Private(state) => state.next_last_value = value,
        }
    }

    #[cfg(test)]
    pub(crate) fn clear_published_receipt_identity_for_test(&mut self) {
        if let SequenceDefaultEffect::Published(receipt) = &mut self.effect {
            receipt.transition_txn_id = 0;
            receipt.input_digest = [0; 32];
        }
    }

    #[cfg(test)]
    pub(crate) fn published_input_identity_for_test(
        &self,
    ) -> Option<(u32, gpu_db_wal::CanonicalDigest)> {
        let SequenceDefaultEffect::Published(receipt) = &self.effect else {
            return None;
        };
        Some((receipt.expression_ordinal, receipt.input_digest))
    }

    pub(super) fn matches_request(
        &self,
        request: &SequenceDefaultRequest,
        parent: &SequenceDefaultParentContext,
    ) -> bool {
        self.request == *request
            && parent.admits(request)
            && match &self.effect {
                SequenceDefaultEffect::Published(receipt) => {
                    let Some(expression_ordinal) = parent.expression_ordinal(request) else {
                        return false;
                    };
                    receipt.parent == *parent
                        && receipt.statement_ordinal == request.statement_ordinal
                        && receipt.expression_ordinal == expression_ordinal
                        && receipt.target_table_oid == request.target_table_oid
                        && receipt.target_column_id == request.column_id
                        && receipt.staging_row_ordinal == request.row_ordinal
                        && receipt.transition_txn_id != 0
                        && receipt.input_digest != [0; 32]
                        && receipt.sequence_oid == request.sequence_oid
                        && receipt.returned_value == self.value
                        && receipt.input_digest
                            == crate::sequence_value_input_digest(crate::SequenceValueInput {
                                parent_txn_id: parent.parent_txn_id,
                                parent_autocommit: parent.parent_autocommit,
                                statement_ordinal: parent.statement_ordinal.as_u32(),
                                expression_ordinal,
                                parent_request_digest: parent.parent_request_digest,
                                source_name: request.sequence_source_name.as_ref(),
                                operation: crate::BinarySequenceValueOperation::Default,
                                set_value: None,
                            })
                }
                SequenceDefaultEffect::Private(state) => {
                    let Some(expression_ordinal) = parent.expression_ordinal(request) else {
                        return false;
                    };
                    state.parent == *parent
                        && state.statement_ordinal == request.statement_ordinal
                        && state.expression_ordinal == expression_ordinal
                        && state.target_table_oid == request.target_table_oid
                        && state.target_column_id == request.column_id
                        && state.staging_row_ordinal == request.row_ordinal
                        && state.sequence_oid == request.sequence_oid
                        && state.next_is_called
                        && state.next_last_value == self.value
                        && matches!(state.private_lifetime_origin, 1 | 2)
                        && matches!(state.private_owner_kind, 1..=3)
                        && state.private_owner_statement_digest != [0; 32]
                        && matches!(state.private_predecessor_tag, 1 | 2)
                        && state.private_predecessor_digest != [0; 32]
                        && state.private_input_digest
                            == crate::sequence_value_input_digest(crate::SequenceValueInput {
                                parent_txn_id: parent.parent_txn_id,
                                parent_autocommit: parent.parent_autocommit,
                                statement_ordinal: parent.statement_ordinal.as_u32(),
                                expression_ordinal,
                                parent_request_digest: parent.parent_request_digest,
                                source_name: request.sequence_source_name.as_ref(),
                                operation: crate::BinarySequenceValueOperation::Default,
                                set_value: None,
                            })
                        && state.private_descriptor_digest != [0; 32]
                        && state.private_descriptor_digest
                            == crate::sequence_descriptor_digest(
                                request.sequence_oid,
                                request.sequence_effective_name.as_ref(),
                            )
                        && state.private_child_digest != [0; 32]
                        && state.private_outcome_digest != [0; 32]
                        && if state.prior_is_called {
                            state.prior_last_value.checked_add(1) == Some(state.next_last_value)
                        } else {
                            state.prior_last_value == state.next_last_value
                        }
                }
            }
    }
}

/// A move-only bundle supplied by the sequence owner. Its parent context is separate from every
/// receipt, so a structurally valid effect from another admitted parent cannot be rebound here.
pub(crate) struct SequenceDefaultBindings {
    pub(super) parent: Option<SequenceDefaultParentContext>,
    pub(super) bindings: Box<[SequenceDefaultBinding]>,
}

impl SequenceDefaultBindings {
    pub(crate) fn empty() -> Self {
        Self {
            parent: None,
            bindings: Box::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_bindings(
        parent: SequenceDefaultParentContext,
        bindings: Vec<SequenceDefaultBinding>,
    ) -> Self {
        Self {
            parent: Some(parent),
            bindings: bindings.into(),
        }
    }
}
