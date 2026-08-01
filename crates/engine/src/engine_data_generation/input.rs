//! Root-free inputs. They bind ordered, GPU-observed facts without carrying a final table,
//! index, database, or publication root.

use super::digest::{
    CanonicalCatalogDigest, CatalogIdentity, ColumnShapeRoot, CommitSequence, CurrentRowLeafRoot,
    DatabaseId, DatabaseRoot, IndexEntryLeafRoot, IndexGeneration, IndexRoot, IndexShapeRoot,
    RequestDigest, ReturningDigest, StableColumnId, StableIndexId, StableRowId, StableTableId,
    StableTransactionId, TableRoot, TargetDigest, TerminalEnvelopeDigest, TypedValueRoot,
};
use super::DataGenerationError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TableDeltaKind {
    RowSet,
    CreateEmpty,
    Drop,
    ResetEmpty,
    Rebuild,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RowAction {
    Insert,
    Replace,
    Remove,
    RebuildCurrent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TerminalOutcomeKind {
    CommitSuccess,
    CommitNoOp,
    AbortError,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TerminalOutcome {
    pub(super) kind: TerminalOutcomeKind,
    pub(super) affected_rows: u64,
    pub(super) sqlstate: Option<[u8; 5]>,
    pub(super) constraint_id: u64,
}

impl TerminalOutcome {
    pub(super) fn validate(&self) -> Result<(), DataGenerationError> {
        match self.kind {
            TerminalOutcomeKind::CommitSuccess
                if self.sqlstate.is_none() && self.constraint_id == 0 =>
            {
                Ok(())
            }
            TerminalOutcomeKind::CommitNoOp
                if self.sqlstate.is_none()
                    && self.constraint_id == 0
                    && self.affected_rows == 0 =>
            {
                Ok(())
            }
            TerminalOutcomeKind::CommitSuccess | TerminalOutcomeKind::CommitNoOp => {
                Err(DataGenerationError::Invalid("commit outcome"))
            }
            TerminalOutcomeKind::AbortError => {
                let Some(sqlstate) = self.sqlstate else {
                    return Err(DataGenerationError::Invalid("rejected SQLSTATE"));
                };
                if sqlstate
                    .iter()
                    .any(|byte| !byte.is_ascii_uppercase() && !byte.is_ascii_digit())
                {
                    return Err(DataGenerationError::Invalid("SQLSTATE bytes"));
                }
                if self.affected_rows != 0 {
                    return Err(DataGenerationError::Invalid("abort affected rows"));
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TableBefore {
    pub(super) data_generation: super::digest::DataGeneration,
    pub(super) root: TableRoot,
    pub(super) logical_row_count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IndexBefore {
    pub(super) generation: IndexGeneration,
    pub(super) root: IndexRoot,
}

/// One ordered component of the root-free index grammar. The future GPU completion adapter must
/// bind this declared grammar to `FinalIndexShape::shape_root`; this foundation preserves both
/// facts without deriving a host commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IndexKeyDescriptor {
    pub(super) key_ordinal: u16,
    pub(super) column_id: StableColumnId,
    pub(super) shape_root: ColumnShapeRoot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FinalIndexShape {
    pub(super) index_id: StableIndexId,
    pub(super) shape_root: IndexShapeRoot,
    pub(super) key_descriptors: Vec<IndexKeyDescriptor>,
    pub(super) before: Option<IndexBefore>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ColumnInput {
    pub(super) column_id: StableColumnId,
    pub(super) shape_root: ColumnShapeRoot,
    pub(super) typed_value_root: TypedValueRoot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IndexKeyInput {
    /// Position in the index key grammar. This is independent from the stable column identity:
    /// a composite key such as `(b, a)` is represented as ordinals zero then one even when
    /// `b` has a larger stable ID than `a`.
    pub(super) key_ordinal: u16,
    pub(super) column_id: StableColumnId,
    pub(super) shape_root: ColumnShapeRoot,
    pub(super) typed_value_root: TypedValueRoot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct IndexMembershipInput {
    pub(super) index_id: StableIndexId,
    pub(super) before_entry_leaf: Option<IndexEntryLeafRoot>,
    pub(super) after_present: bool,
    pub(super) key_columns: Vec<IndexKeyInput>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AfterRow {
    pub(super) created_by: CommitSequence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RowInput {
    pub(super) row_id: StableRowId,
    pub(super) action: RowAction,
    pub(super) before_current_row_leaf: Option<CurrentRowLeafRoot>,
    pub(super) after: Option<AfterRow>,
    pub(super) columns: Vec<ColumnInput>,
    pub(super) index_memberships: Vec<IndexMembershipInput>,
}

impl RowInput {
    fn validate(
        &self,
        commit_sequence: CommitSequence,
        final_indexes: &[FinalIndexShape],
        table_kind: TableDeltaKind,
    ) -> Result<(), DataGenerationError> {
        if table_kind == TableDeltaKind::Rebuild && self.action != RowAction::RebuildCurrent {
            return Err(DataGenerationError::Invalid("Rebuild row action"));
        }
        match self.action {
            RowAction::Insert => {
                if self.before_current_row_leaf.is_some()
                    || self.after
                        != Some(AfterRow {
                            created_by: commit_sequence,
                        })
                {
                    return Err(DataGenerationError::Invalid("insert row input"));
                }
            }
            RowAction::Replace => {
                if self.before_current_row_leaf.is_none()
                    || self.after
                        != Some(AfterRow {
                            created_by: commit_sequence,
                        })
                {
                    return Err(DataGenerationError::Invalid("replace row input"));
                }
            }
            RowAction::Remove => {
                if self.before_current_row_leaf.is_none()
                    || self.after.is_some()
                    || !self.columns.is_empty()
                {
                    return Err(DataGenerationError::Invalid("remove row input"));
                }
            }
            RowAction::RebuildCurrent => {
                if table_kind != TableDeltaKind::Rebuild || self.after.is_none() {
                    return Err(DataGenerationError::Invalid("rebuild-current row input"));
                }
                if self.before_current_row_leaf.is_none()
                    && self.after
                        != Some(AfterRow {
                            created_by: commit_sequence,
                        })
                {
                    return Err(DataGenerationError::Invalid("new rebuild-current row"));
                }
            }
        }
        validate_strictly_ascending(
            self.columns.iter().map(|column| column.column_id),
            "row columns",
        )?;
        validate_strictly_ascending(
            self.index_memberships
                .iter()
                .map(|membership| membership.index_id),
            "row index memberships",
        )?;
        if self.index_memberships.len() != final_indexes.len() {
            return Err(DataGenerationError::Invalid(
                "row index membership coverage",
            ));
        }
        for (membership, final_index) in self.index_memberships.iter().zip(final_indexes) {
            if membership.index_id != final_index.index_id {
                return Err(DataGenerationError::Invalid(
                    "row index membership identity",
                ));
            }
            if self.before_current_row_leaf.is_none() && membership.before_entry_leaf.is_some() {
                return Err(DataGenerationError::Invalid(
                    "new row index membership predecessor",
                ));
            }
            validate_dense_key_ordinals(&membership.key_columns)?;
            if !membership.after_present && !membership.key_columns.is_empty() {
                return Err(DataGenerationError::Invalid("absent index membership keys"));
            }
            if membership.after_present {
                validate_membership_key_descriptors(membership, final_index)?;
            }
            if self.action == RowAction::Remove && membership.after_present {
                return Err(DataGenerationError::Invalid("removed row index membership"));
            }
        }
        Ok(())
    }
}

fn validate_dense_key_ordinals(keys: &[IndexKeyInput]) -> Result<(), DataGenerationError> {
    for (expected, key) in keys.iter().enumerate() {
        let expected = u16::try_from(expected)
            .map_err(|_| DataGenerationError::Invalid("index key ordinal range"))?;
        if key.key_ordinal != expected {
            return Err(DataGenerationError::Invalid("index key ordinal"));
        }
    }
    Ok(())
}

fn validate_membership_key_descriptors(
    membership: &IndexMembershipInput,
    final_index: &FinalIndexShape,
) -> Result<(), DataGenerationError> {
    if membership.key_columns.len() != final_index.key_descriptors.len() {
        return Err(DataGenerationError::Invalid(
            "index key descriptor coverage",
        ));
    }
    for (component, descriptor) in membership
        .key_columns
        .iter()
        .zip(&final_index.key_descriptors)
    {
        if component.key_ordinal != descriptor.key_ordinal {
            return Err(DataGenerationError::Invalid("index key descriptor ordinal"));
        }
        if component.column_id != descriptor.column_id {
            return Err(DataGenerationError::Invalid("index key descriptor column"));
        }
        if component.shape_root != descriptor.shape_root {
            return Err(DataGenerationError::Invalid("index key descriptor shape"));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RowSetInput {
    pub(super) table_id: StableTableId,
    pub(super) before: TableBefore,
    pub(super) final_indexes: Vec<FinalIndexShape>,
    pub(super) rows: Vec<RowInput>,
}

impl RowSetInput {
    fn validate(&self, commit_sequence: CommitSequence) -> Result<(), DataGenerationError> {
        if self.rows.is_empty() {
            return Err(DataGenerationError::Invalid("empty RowSet"));
        }
        validate_final_indexes(&self.final_indexes, IndexBeforeRule::AllPresent)?;
        validate_strictly_ascending(self.rows.iter().map(|row| row.row_id), "RowSet row ids")?;
        for row in &self.rows {
            row.validate(commit_sequence, &self.final_indexes, TableDeltaKind::RowSet)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CreateEmptyInput {
    pub(super) table_id: StableTableId,
    pub(super) final_indexes: Vec<FinalIndexShape>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DropInput {
    pub(super) table_id: StableTableId,
    pub(super) before: TableBefore,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ResetEmptyInput {
    pub(super) table_id: StableTableId,
    pub(super) before: TableBefore,
    pub(super) final_indexes: Vec<FinalIndexShape>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RebuildInput {
    pub(super) table_id: StableTableId,
    pub(super) before: Option<TableBefore>,
    pub(super) final_indexes: Vec<FinalIndexShape>,
    pub(super) rows: Vec<RowInput>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum TableDelta {
    RowSet(RowSetInput),
    CreateEmpty(CreateEmptyInput),
    Drop(DropInput),
    ResetEmpty(ResetEmptyInput),
    Rebuild(RebuildInput),
}

impl TableDelta {
    pub(super) fn kind(&self) -> TableDeltaKind {
        match self {
            Self::RowSet(_) => TableDeltaKind::RowSet,
            Self::CreateEmpty(_) => TableDeltaKind::CreateEmpty,
            Self::Drop(_) => TableDeltaKind::Drop,
            Self::ResetEmpty(_) => TableDeltaKind::ResetEmpty,
            Self::Rebuild(_) => TableDeltaKind::Rebuild,
        }
    }

    pub(super) fn table_id(&self) -> StableTableId {
        match self {
            Self::RowSet(input) => input.table_id,
            Self::CreateEmpty(input) => input.table_id,
            Self::Drop(input) => input.table_id,
            Self::ResetEmpty(input) => input.table_id,
            Self::Rebuild(input) => input.table_id,
        }
    }

    fn validate(&self, commit_sequence: CommitSequence) -> Result<(), DataGenerationError> {
        match self {
            Self::RowSet(input) => input.validate(commit_sequence),
            Self::CreateEmpty(input) => {
                validate_final_indexes(&input.final_indexes, IndexBeforeRule::AllAbsent)
            }
            Self::Drop(_) => Ok(()),
            Self::ResetEmpty(input) => {
                if input.before.logical_row_count == 0 {
                    return Err(DataGenerationError::Invalid("empty ResetEmpty predecessor"));
                }
                validate_final_indexes(&input.final_indexes, IndexBeforeRule::Either)
            }
            Self::Rebuild(input) => {
                if input.rows.is_empty() {
                    return Err(DataGenerationError::Invalid("empty Rebuild"));
                }
                validate_final_indexes(
                    &input.final_indexes,
                    if input.before.is_some() {
                        IndexBeforeRule::Either
                    } else {
                        IndexBeforeRule::AllAbsent
                    },
                )?;
                validate_strictly_ascending(
                    input.rows.iter().map(|row| row.row_id),
                    "Rebuild row ids",
                )?;
                for row in &input.rows {
                    row.validate(
                        commit_sequence,
                        &input.final_indexes,
                        TableDeltaKind::Rebuild,
                    )?;
                    if input.before.is_none() && row.before_current_row_leaf.is_some() {
                        return Err(DataGenerationError::Invalid("new Rebuild row predecessor"));
                    }
                    for (membership, final_index) in
                        row.index_memberships.iter().zip(&input.final_indexes)
                    {
                        if final_index.before.is_none() && membership.before_entry_leaf.is_some() {
                            return Err(DataGenerationError::Invalid(
                                "new Rebuild index membership predecessor",
                            ));
                        }
                    }
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StatusInput {
    pub(super) request_digest: RequestDigest,
    pub(super) target_digest: TargetDigest,
    pub(super) returning_digest: ReturningDigest,
    pub(super) terminal_envelope_digest: TerminalEnvelopeDigest,
}

/// All durable/recovery input facts before any final root or final generation is constructed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RootFreeGenerationInput {
    pub(super) database_id: DatabaseId,
    pub(super) initial_database_root: DatabaseRoot,
    pub(super) catalog_before: CatalogIdentity,
    pub(super) catalog_after: CatalogIdentity,
    pub(super) transaction_id: StableTransactionId,
    pub(super) commit_sequence: CommitSequence,
    pub(super) terminal_outcome: TerminalOutcome,
    pub(super) status: StatusInput,
    pub(super) table_deltas: Vec<TableDelta>,
}

impl RootFreeGenerationInput {
    pub(super) fn validate(&self) -> Result<(), DataGenerationError> {
        self.terminal_outcome.validate()?;
        validate_strictly_ascending(
            self.table_deltas.iter().map(TableDelta::table_id),
            "table deltas",
        )?;
        if self.terminal_outcome.kind == TerminalOutcomeKind::CommitNoOp
            && !self.table_deltas.is_empty()
        {
            return Err(DataGenerationError::Invalid(
                "commit-no-op generation mutations",
            ));
        }
        if self.terminal_outcome.kind == TerminalOutcomeKind::AbortError
            && !self.table_deltas.is_empty()
        {
            return Err(DataGenerationError::Invalid("abort generation mutations"));
        }
        if matches!(
            self.terminal_outcome.kind,
            TerminalOutcomeKind::CommitNoOp | TerminalOutcomeKind::AbortError
        ) && self.catalog_after != self.catalog_before
        {
            return Err(DataGenerationError::Invalid(
                "non-success catalog transition",
            ));
        }
        for delta in &self.table_deltas {
            delta.validate(self.commit_sequence)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum IndexBeforeRule {
    AllPresent,
    AllAbsent,
    Either,
}

fn validate_final_indexes(
    indexes: &[FinalIndexShape],
    before_rule: IndexBeforeRule,
) -> Result<(), DataGenerationError> {
    validate_strictly_ascending(indexes.iter().map(|index| index.index_id), "final indexes")?;
    for index in indexes {
        validate_dense_key_descriptor_ordinals(&index.key_descriptors)?;
    }
    match before_rule {
        IndexBeforeRule::AllPresent if indexes.iter().any(|index| index.before.is_none()) => {
            return Err(DataGenerationError::Invalid("final index predecessor"));
        }
        IndexBeforeRule::AllAbsent if indexes.iter().any(|index| index.before.is_some()) => {
            return Err(DataGenerationError::Invalid("new final index predecessor"));
        }
        _ => {}
    }
    Ok(())
}

fn validate_dense_key_descriptor_ordinals(
    descriptors: &[IndexKeyDescriptor],
) -> Result<(), DataGenerationError> {
    for (expected, descriptor) in descriptors.iter().enumerate() {
        let expected = u16::try_from(expected)
            .map_err(|_| DataGenerationError::Invalid("index key descriptor ordinal range"))?;
        if descriptor.key_ordinal != expected {
            return Err(DataGenerationError::Invalid("index key descriptor ordinal"));
        }
    }
    Ok(())
}

fn validate_strictly_ascending<T: Ord>(
    values: impl Iterator<Item = T>,
    label: &'static str,
) -> Result<(), DataGenerationError> {
    let mut previous = None;
    for value in values {
        if previous.as_ref().is_some_and(|prior| prior >= &value) {
            return Err(DataGenerationError::NonCanonicalOrder(label));
        }
        previous = Some(value);
    }
    Ok(())
}

#[allow(dead_code)]
fn _catalog_digest_is_opaque(_: CanonicalCatalogDigest) {}
