//! Positive read-only neutral views for the reserved generation builder.

use super::{
    CanonicalDigest, NeutralCell, NeutralEffect, NeutralIdentity, NeutralIndex, NeutralIndexKey,
    NeutralRow, NeutralTable, NeutralTypedValue, SealedGenerationInput,
};

/// Builder-visible neutral input.  The view has no graph/catalog/allocator reference and exposes
/// no private flat offsets or output-derived roots/generations.
pub(in super::super::super) struct NeutralGenerationInputView<'a> {
    input: &'a SealedGenerationInput,
}

#[derive(Clone, Copy)]
pub(in super::super::super) struct NeutralIdentityView {
    pub(in super::super::super) database_id: [u8; 16],
    pub(in super::super::super) catalog_epoch: u64,
    pub(in super::super::super) catalog_digest: CanonicalDigest,
    pub(in super::super::super) stable_transaction_id: u64,
    pub(in super::super::super) commit_sequence: u64,
    pub(in super::super::super) initial_database_root: CanonicalDigest,
}

#[derive(Clone, Copy)]
pub(in super::super::super) struct NeutralTableView {
    pub(in super::super::super) stable_table_id: u64,
    pub(in super::super::super) base_data_generation: u64,
    pub(in super::super::super) base_table_root: CanonicalDigest,
    pub(in super::super::super) row_allocator_before: u64,
    pub(in super::super::super) row_allocator_high_water: u64,
    pub(in super::super::super) initial_logical_row_count: u64,
    pub(in super::super::super) final_logical_row_count: u64,
    pub(in super::super::super) image_layout_digest: CanonicalDigest,
    pub(in super::super::super) image_content_digest: CanonicalDigest,
    pub(in super::super::super) row_count: u32,
    pub(in super::super::super) index_count: u32,
}

#[derive(Clone, Copy)]
pub(in super::super::super) struct NeutralRowView {
    pub(in super::super::super) stable_table_id: u64,
    pub(in super::super::super) stable_row_id: u64,
    pub(in super::super::super) source_statement_ordinal: u32,
    pub(in super::super::super) source_row_ordinal: u32,
    pub(in super::super::super) cell_count: u32,
}

pub(in super::super::super) struct NeutralCellView<'a> {
    pub(in super::super::super) catalog_column_ordinal: u32,
    pub(in super::super::super) stable_column_id: u32,
    pub(in super::super::super) attnum: i16,
    pub(in super::super::super) storage: [u8; 4],
    pub(in super::super::super) declared_type_oid: u32,
    pub(in super::super::super) signed_type_size: i16,
    pub(in super::super::super) is_null: bool,
    pub(in super::super::super) value: &'a [u8],
}

#[derive(Clone, Copy)]
pub(in super::super::super) struct NeutralIndexView {
    pub(in super::super::super) owner_stable_table_id: u64,
    pub(in super::super::super) stable_index_id: u64,
    pub(in super::super::super) flags: u32,
    pub(in super::super::super) null_equality_policy: u8,
    pub(in super::super::super) base_generation: u64,
    pub(in super::super::super) base_root: CanonicalDigest,
    pub(in super::super::super) key_count: u32,
    pub(in super::super::super) effect_count: u32,
}

#[derive(Clone, Copy)]
pub(in super::super::super) struct NeutralIndexKeyView {
    pub(in super::super::super) key_ordinal: u32,
    pub(in super::super::super) owner_catalog_column_ordinal: u32,
    pub(in super::super::super) stable_column_id: u32,
    pub(in super::super::super) attnum: i16,
    pub(in super::super::super) storage: [u8; 4],
    pub(in super::super::super) declared_type_oid: u32,
    pub(in super::super::super) signed_type_size: i16,
    pub(in super::super::super) column_name_digest: CanonicalDigest,
}

#[derive(Clone, Copy)]
pub(in super::super::super) struct NeutralEffectView {
    pub(in super::super::super) stable_table_id: u64,
    pub(in super::super::super) stable_index_id: u64,
    pub(in super::super::super) stable_row_id: u64,
    pub(in super::super::super) source_catalog_ordinal: u32,
    pub(in super::super::super) key_arity: u32,
    pub(in super::super::super) value_count: u32,
}

pub(in super::super::super) struct NeutralTypedValueView<'a> {
    pub(in super::super::super) storage: [u8; 4],
    pub(in super::super::super) declared_type_oid: u32,
    pub(in super::super::super) signed_type_size: i16,
    pub(in super::super::super) is_null: bool,
    pub(in super::super::super) value: &'a [u8],
}

impl SealedGenerationInput {
    pub(in super::super::super) fn neutral_view(&self) -> NeutralGenerationInputView<'_> {
        NeutralGenerationInputView { input: self }
    }
}

impl<'a> NeutralGenerationInputView<'a> {
    pub(in super::super::super) fn identity(&self) -> NeutralIdentityView {
        identity_view(self.input.identity)
    }

    pub(in super::super::super) fn tables(
        &self,
    ) -> impl ExactSizeIterator<Item = NeutralTableView> + '_ {
        self.input.tables.iter().copied().map(table_view)
    }

    pub(in super::super::super) fn rows(
        &self,
    ) -> impl ExactSizeIterator<Item = NeutralRowView> + '_ {
        self.input.rows.iter().copied().map(row_view)
    }

    pub(in super::super::super) fn cells(
        &self,
    ) -> impl ExactSizeIterator<Item = NeutralCellView<'a>> + '_ {
        self.input
            .cells
            .iter()
            .copied()
            .map(move |cell| cell_view(cell, &self.input.values))
    }

    pub(in super::super::super) fn indexes(
        &self,
    ) -> impl ExactSizeIterator<Item = NeutralIndexView> + '_ {
        self.input.indexes.iter().copied().map(index_view)
    }

    pub(in super::super::super) fn keys(
        &self,
    ) -> impl ExactSizeIterator<Item = NeutralIndexKeyView> + '_ {
        self.input.keys.iter().copied().map(key_view)
    }

    pub(in super::super::super) fn effects(
        &self,
    ) -> impl ExactSizeIterator<Item = NeutralEffectView> + '_ {
        self.input.effects.iter().copied().map(effect_view)
    }

    pub(in super::super::super) fn effect_values(
        &self,
    ) -> impl ExactSizeIterator<Item = NeutralTypedValueView<'a>> + '_ {
        self.input
            .effect_values
            .iter()
            .copied()
            .map(move |value| typed_value_view(value, &self.input.effect_value_bytes))
    }
}

fn identity_view(identity: NeutralIdentity) -> NeutralIdentityView {
    NeutralIdentityView {
        database_id: identity.database_id,
        catalog_epoch: identity.catalog_epoch,
        catalog_digest: identity.catalog_digest,
        stable_transaction_id: identity.stable_transaction_id,
        commit_sequence: identity.commit_sequence,
        initial_database_root: identity.initial_database_root,
    }
}

fn table_view(table: NeutralTable) -> NeutralTableView {
    NeutralTableView {
        stable_table_id: table.stable_table_id,
        base_data_generation: table.base_data_generation,
        base_table_root: table.base_table_root,
        row_allocator_before: table.row_allocator_before,
        row_allocator_high_water: table.row_allocator_high_water,
        initial_logical_row_count: table.initial_logical_row_count,
        final_logical_row_count: table.final_logical_row_count,
        image_layout_digest: table.image_layout_digest,
        image_content_digest: table.image_content_digest,
        row_count: table.row_count,
        index_count: table.index_count,
    }
}

fn row_view(row: NeutralRow) -> NeutralRowView {
    NeutralRowView {
        stable_table_id: row.stable_table_id,
        stable_row_id: row.stable_row_id,
        source_statement_ordinal: row.source_statement_ordinal,
        source_row_ordinal: row.source_row_ordinal,
        cell_count: row.cell_count,
    }
}

fn cell_view(cell: NeutralCell, values: &[u8]) -> NeutralCellView<'_> {
    NeutralCellView {
        catalog_column_ordinal: cell.catalog_column_ordinal,
        stable_column_id: cell.stable_column_id,
        attnum: cell.attnum,
        storage: cell.storage,
        declared_type_oid: cell.declared_type_oid,
        signed_type_size: cell.signed_type_size,
        is_null: cell.is_null,
        value: flat_bytes(values, cell.value_start, cell.value_count),
    }
}

fn index_view(index: NeutralIndex) -> NeutralIndexView {
    NeutralIndexView {
        owner_stable_table_id: index.owner_stable_table_id,
        stable_index_id: index.stable_index_id,
        flags: index.flags,
        null_equality_policy: index.null_equality_policy,
        base_generation: index.base_generation,
        base_root: index.base_root,
        key_count: index.key_count,
        effect_count: index.effect_count,
    }
}

fn key_view(key: NeutralIndexKey) -> NeutralIndexKeyView {
    NeutralIndexKeyView {
        key_ordinal: key.key_ordinal,
        owner_catalog_column_ordinal: key.owner_catalog_column_ordinal,
        stable_column_id: key.stable_column_id,
        attnum: key.attnum,
        storage: key.storage,
        declared_type_oid: key.declared_type_oid,
        signed_type_size: key.signed_type_size,
        column_name_digest: key.column_name_digest,
    }
}

fn effect_view(effect: NeutralEffect) -> NeutralEffectView {
    NeutralEffectView {
        stable_table_id: effect.stable_table_id,
        stable_index_id: effect.stable_index_id,
        stable_row_id: effect.stable_row_id,
        source_catalog_ordinal: effect.source_catalog_ordinal,
        key_arity: effect.key_arity,
        value_count: effect.value_count,
    }
}

fn typed_value_view(value: NeutralTypedValue, bytes: &[u8]) -> NeutralTypedValueView<'_> {
    NeutralTypedValueView {
        storage: value.storage,
        declared_type_oid: value.declared_type_oid,
        signed_type_size: value.signed_type_size,
        is_null: value.is_null,
        value: flat_bytes(bytes, value.value_start, value.value_count),
    }
}

fn flat_bytes(values: &[u8], start: u32, count: u32) -> &[u8] {
    let start = usize::try_from(start).expect("sealed flat value start fits usize");
    let count = usize::try_from(count).expect("sealed flat value count fits usize");
    values
        .get(
            start
                ..start
                    .checked_add(count)
                    .expect("sealed flat value range does not overflow"),
        )
        .expect("sealed flat value range was checked before launch")
}
