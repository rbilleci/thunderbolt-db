//! Closed host/device ABI for the first plural typed-INSERT generation program.
//!
//! The descriptor owns one table and any number of rows/cells.  Cell payloads are logical typed
//! bytes, not INT4-shaped vectors: fixed-width, BOOL, TEXT, UUID, NUMERIC, and NULL all use the
//! same descriptor.  The indexed canary adds one ordered descriptor/key/effect tail; the zero
//! index layout remains byte-for-byte identical to the original descriptor.

pub(super) const HEADER_BYTES: usize = 256;
/// One table descriptor plus the exact predecessor table-map sibling path.  A pinned retained
/// witness can additionally carry roots already authenticated by the immutable predecessor map;
/// it never carries a successor and the device still derives the only new COW path.
pub(super) const TABLE_BYTES: usize = 144
    + RADIX_DEPTH * DIGEST_BYTES
    + EMPTY_ROOTS * DIGEST_BYTES
    + DIGEST_BYTES
    + RADIX_DEPTH * DIGEST_BYTES;
pub(super) const ROW_BYTES: usize = 40;
pub(super) const CELL_BYTES: usize = 48;
/// The first indexed vertical is deliberately one non-unique compound index. The descriptor owns
/// the immutable predecessor generation/root; its two ordered key descriptors and one effect per
/// inserted row follow in their own fixed-width regions. Successive INSERTs advance that root by
/// a device-derived COW lineage transition without rebuilding predecessor entries.
pub(super) const INDEX_BYTES: usize = 80;
pub(super) const INDEX_KEY_BYTES: usize = 64;
pub(super) const INDEX_EFFECT_BYTES: usize = 40;
pub(super) const INDEX_EFFECT_COMPONENT_BYTES: usize = 8;
pub(super) const OUTPUT_HEADER_BYTES: usize = 32;
pub(super) const DIGEST_BYTES: usize = 32;
pub(super) const RADIX_DEPTH: usize = 64;
pub(super) const EMPTY_ROOTS: usize = RADIX_DEPTH + 1;
const V3_MAP_NODE_PREFIX_STATE_BYTES: usize = RADIX_DEPTH * 8 * std::mem::size_of::<u32>();

pub(super) const DATABASE_ID_OFFSET: usize = 0;
pub(super) const CATALOG_EPOCH_OFFSET: usize = 16;
pub(super) const CATALOG_DIGEST_OFFSET: usize = 24;
pub(super) const STABLE_TRANSACTION_ID_OFFSET: usize = 56;
pub(super) const COMMIT_SEQUENCE_OFFSET: usize = 64;
pub(super) const INITIAL_DATABASE_ROOT_OFFSET: usize = 72;
pub(super) const ROOT_FORMAT_OFFSET: usize = 104;
pub(super) const TABLE_ACTION_OFFSET: usize = 106;
pub(super) const TABLE_MAP_PREDECESSOR_OFFSET: usize = 107;
pub(super) const TABLE_COUNT_OFFSET: usize = 108;
pub(super) const ROW_COUNT_OFFSET: usize = 112;
pub(super) const CELL_COUNT_OFFSET: usize = 116;
pub(super) const VALUE_BYTES_OFFSET: usize = 120;
pub(super) const INDEX_COUNT_OFFSET: usize = 124;
pub(super) const HEADER_INDEX_KEY_COUNT_OFFSET: usize = 128;
pub(super) const HEADER_INDEX_EFFECT_COUNT_OFFSET: usize = 132;
pub(super) const HEADER_INDEX_EFFECT_COMPONENT_COUNT_OFFSET: usize = 136;
/// Canonical S7 final-image reference for this one-table generation invocation. The runtime
/// program remains one generic table generator; plural transactions invoke it once per table
/// and bind each device-produced final-row digest to that table's retained image directory slot.
pub(super) const WRITE001_FINAL_IMAGE_REF_OFFSET: usize = 140;
pub(super) const TABLE_OFFSET_OFFSET: usize = 144;
pub(super) const ROW_OFFSET_OFFSET: usize = 152;
pub(super) const CELL_OFFSET_OFFSET: usize = 160;
pub(super) const VALUE_OFFSET_OFFSET: usize = 168;
pub(super) const WORKSPACE_POINTER_OFFSET: usize = 176;
pub(super) const INDEX_OFFSET_OFFSET: usize = 184;
pub(super) const INDEX_KEY_OFFSET_OFFSET: usize = 192;
pub(super) const INDEX_EFFECT_OFFSET_OFFSET: usize = 200;
pub(super) const INDEX_EFFECT_COMPONENT_OFFSET_OFFSET: usize = 208;
/// Optional, named codec-5 transition input. General root builders leave this zero; the live and
/// replayed WRITE-001 typed-INSERT routes provide the exact S2 statement digest.
pub(super) const WRITE001_TYPED_STATEMENT_DIGEST_OFFSET: usize = 216;

pub(super) const TABLE_ID_OFFSET: usize = 0;
pub(super) const TABLE_BASE_GENERATION_OFFSET: usize = 8;
pub(super) const TABLE_BASE_ROOT_OFFSET: usize = 16;
pub(super) const TABLE_ALLOCATOR_BEFORE_OFFSET: usize = 48;
pub(super) const TABLE_ALLOCATOR_HIGH_WATER_OFFSET: usize = 56;
pub(super) const TABLE_INITIAL_ROW_COUNT_OFFSET: usize = 64;
pub(super) const TABLE_FINAL_ROW_COUNT_OFFSET: usize = 72;
pub(super) const TABLE_IMAGE_LAYOUT_DIGEST_OFFSET: usize = 80;
pub(super) const TABLE_IMAGE_CONTENT_DIGEST_OFFSET: usize = 112;
/// One sibling for every root-to-leaf table-map branch, in ascending depth order (0 through
/// 63). The leaf itself is derived on-device from the explicit action and table predecessor.
pub(super) const TABLE_MAP_SIBLINGS_OFFSET: usize = 144;
/// Canonical depth-zero through depth-64 empty roots retained by an already published map.
/// They are copied only for the physical retained-witness schedule and are rechecked by the
/// immutable-map publisher; they are never a host-derived successor.
pub(super) const TABLE_MAP_RETAINED_EMPTY_ROOTS_OFFSET: usize =
    TABLE_MAP_SIBLINGS_OFFSET + RADIX_DEPTH * DIGEST_BYTES;
pub(super) const TABLE_MAP_RETAINED_INITIAL_LEAF_OFFSET: usize =
    TABLE_MAP_RETAINED_EMPTY_ROOTS_OFFSET + EMPTY_ROOTS * DIGEST_BYTES;
pub(super) const TABLE_MAP_RETAINED_INITIAL_PATH_OFFSET: usize =
    TABLE_MAP_RETAINED_INITIAL_LEAF_OFFSET + DIGEST_BYTES;

pub(super) const ROW_TABLE_ID_OFFSET: usize = 0;
pub(super) const ROW_ID_OFFSET: usize = 8;
pub(super) const ROW_STATEMENT_ORDINAL_OFFSET: usize = 16;
pub(super) const ROW_SOURCE_ORDINAL_OFFSET: usize = 20;
pub(super) const ROW_CELL_START_OFFSET: usize = 24;
pub(super) const ROW_CELL_COUNT_OFFSET: usize = 28;

pub(super) const CELL_CATALOG_ORDINAL_OFFSET: usize = 0;
pub(super) const CELL_STABLE_COLUMN_ID_OFFSET: usize = 4;
pub(super) const CELL_ATTNUM_OFFSET: usize = 8;
pub(super) const CELL_STORAGE_OFFSET: usize = 12;
pub(super) const CELL_DECLARED_OID_OFFSET: usize = 16;
pub(super) const CELL_SIGNED_SIZE_OFFSET: usize = 20;
pub(super) const CELL_NULL_OFFSET: usize = 22;
pub(super) const CELL_VALUE_START_OFFSET: usize = 24;
pub(super) const CELL_VALUE_COUNT_OFFSET: usize = 28;

pub(super) const INDEX_STABLE_ID_OFFSET: usize = 0;
pub(super) const INDEX_RAW_CATALOG_ORDINAL_OFFSET: usize = 8;
pub(super) const INDEX_FLAGS_OFFSET: usize = 12;
pub(super) const INDEX_NULL_EQUALITY_POLICY_OFFSET: usize = 16;
pub(super) const INDEX_BASE_GENERATION_OFFSET: usize = 24;
pub(super) const INDEX_BASE_ROOT_OFFSET: usize = 32;
pub(super) const INDEX_KEY_START_OFFSET: usize = 64;
pub(super) const INDEX_KEY_COUNT_OFFSET: usize = 68;
pub(super) const INDEX_EFFECT_START_OFFSET: usize = 72;
pub(super) const INDEX_EFFECT_COUNT_OFFSET: usize = 76;

pub(super) const INDEX_KEY_ORDINAL_OFFSET: usize = 0;
pub(super) const INDEX_KEY_CATALOG_ORDINAL_OFFSET: usize = 4;
pub(super) const INDEX_KEY_STABLE_COLUMN_ID_OFFSET: usize = 8;
pub(super) const INDEX_KEY_ATTNUM_OFFSET: usize = 12;
pub(super) const INDEX_KEY_STORAGE_OFFSET: usize = 16;
pub(super) const INDEX_KEY_DECLARED_OID_OFFSET: usize = 20;
pub(super) const INDEX_KEY_SIGNED_SIZE_OFFSET: usize = 24;
pub(super) const INDEX_KEY_COLUMN_NAME_DIGEST_OFFSET: usize = 32;

pub(super) const INDEX_EFFECT_STABLE_TABLE_ID_OFFSET: usize = 0;
pub(super) const INDEX_EFFECT_STABLE_INDEX_ID_OFFSET: usize = 8;
pub(super) const INDEX_EFFECT_STABLE_ROW_ID_OFFSET: usize = 16;
pub(super) const INDEX_EFFECT_SOURCE_CATALOG_ORDINAL_OFFSET: usize = 24;
pub(super) const INDEX_EFFECT_COMPONENT_START_OFFSET: usize = 28;
pub(super) const INDEX_EFFECT_COMPONENT_COUNT_OFFSET: usize = 32;

pub(super) const INDEX_EFFECT_COMPONENT_CATALOG_ORDINAL_OFFSET: usize = 0;
pub(super) const INDEX_EFFECT_COMPONENT_STABLE_COLUMN_ID_OFFSET: usize = 4;

pub(super) const OUTPUT_STATUS_OFFSET: usize = 0;
pub(super) const OUTPUT_ACTIVE_ROW_NODES_OFFSET: usize = 4;
pub(super) const OUTPUT_ROW_COUNT_OFFSET: usize = 8;
pub(super) const OUTPUT_CELL_COUNT_OFFSET: usize = 12;
pub(super) const OUTPUT_DIGEST_COUNT_OFFSET: usize = 16;

pub(super) const SLOT_GENERATION_INPUT: usize = 0;
pub(super) const SLOT_INITIAL_TABLE_ROOT: usize = 1;
pub(super) const SLOT_FINAL_TABLE_ROOT: usize = 2;
pub(super) const SLOT_INITIAL_DATABASE_ROOT: usize = 5;
pub(super) const SLOT_FINAL_DATABASE_ROOT: usize = 6;
pub(super) const SLOT_ROW_EMPTY: usize = 7;

#[derive(Clone, Copy, Debug)]
pub(super) struct OutputLayout {
    pub(super) shape_roots: usize,
    pub(super) typed_roots: usize,
    pub(super) column_roots: usize,
    #[cfg(test)]
    pub(super) current_row_leaves: usize,
    /// Device-produced canonical S7 final-row digests. These are distinct from the runtime
    /// row-tree leaves: S7 binds the original typed cells and image order for WAL closure.
    pub(super) s7_final_row_digests: usize,
    /// Device-produced digest of each exact 192-byte S7 transition record.
    pub(super) s7_transition_digests: usize,
    #[cfg(test)]
    pub(super) row_nodes: usize,
    pub(super) row_node_capacity: usize,
    table_empty_slot: usize,
    initial_table_leaf_slot: usize,
    initial_table_path_slot: usize,
    final_table_leaf_slot: usize,
    final_table_path_slot: usize,
    index_initial_roots: usize,
    index_final_roots: usize,
    pub(super) digest_count: usize,
}

impl OutputLayout {
    pub(super) fn new(rows: usize, cells: usize, indexes: usize) -> Option<Self> {
        let shape_roots = SLOT_ROW_EMPTY.checked_add(EMPTY_ROOTS)?;
        let typed_roots = shape_roots.checked_add(cells)?;
        let column_roots = typed_roots.checked_add(cells)?;
        let current_row_leaves = column_roots.checked_add(cells)?;
        let s7_final_row_digests = current_row_leaves.checked_add(rows)?;
        let s7_transition_digests = s7_final_row_digests.checked_add(rows)?;
        let row_leaves = s7_transition_digests.checked_add(rows)?;
        let row_nodes = row_leaves.checked_add(rows)?;
        let row_node_capacity = compact_row_node_capacity(rows)?;
        let table_map_empty = row_nodes.checked_add(row_node_capacity)?;
        let initial_table_map_leaf = table_map_empty.checked_add(EMPTY_ROOTS)?;
        let initial_table_map_path = initial_table_map_leaf.checked_add(1)?;
        let final_table_map_leaf = initial_table_map_path.checked_add(RADIX_DEPTH)?;
        let final_table_map_path = final_table_map_leaf.checked_add(1)?;
        let index_initial_roots = final_table_map_path.checked_add(RADIX_DEPTH)?;
        let index_final_roots = index_initial_roots.checked_add(indexes)?;
        let digest_count = index_final_roots.checked_add(indexes)?;
        Some(Self {
            shape_roots,
            typed_roots,
            column_roots,
            #[cfg(test)]
            current_row_leaves,
            s7_final_row_digests,
            s7_transition_digests,
            #[cfg(test)]
            row_nodes,
            row_node_capacity,
            table_empty_slot: table_map_empty,
            initial_table_leaf_slot: initial_table_map_leaf,
            initial_table_path_slot: initial_table_map_path,
            final_table_leaf_slot: final_table_map_leaf,
            final_table_path_slot: final_table_map_path,
            index_initial_roots,
            index_final_roots,
            digest_count,
        })
    }

    pub(super) fn output_bytes(self) -> Option<usize> {
        OUTPUT_HEADER_BYTES.checked_add(self.digest_count.checked_mul(DIGEST_BYTES)?)
    }

    pub(super) fn table_map_empty_slot(self) -> usize {
        self.table_empty_slot
    }

    pub(super) fn initial_table_map_leaf_slot(self) -> usize {
        self.initial_table_leaf_slot
    }

    pub(super) fn initial_table_map_path_slot(self) -> usize {
        self.initial_table_path_slot
    }

    pub(super) fn final_table_map_leaf_slot(self) -> usize {
        self.final_table_leaf_slot
    }

    pub(super) fn final_table_map_path_slot(self) -> usize {
        self.final_table_path_slot
    }

    pub(super) fn index_initial_roots_slot(self) -> usize {
        self.index_initial_roots
    }

    pub(super) fn index_final_roots_slot(self) -> usize {
        self.index_final_roots
    }
}

/// Capacity for all active internal nodes of a radix tree over a contiguous, ascending ID range.
/// At a level whose nodes cover `span` IDs, an unaligned contiguous interval touches at most
/// `ceil(rows / span) + 1` nodes.  The kernel validates contiguity before relying on this bound.
pub(super) fn compact_row_node_capacity(rows: usize) -> Option<usize> {
    if rows == 0 {
        return Some(0);
    }
    let mut total = 0_usize;
    for shift in 1..=RADIX_DEPTH {
        let level_capacity = if shift == RADIX_DEPTH {
            1
        } else {
            let span = 1_usize << shift;
            let rounded = rows.checked_add(span.checked_sub(1)?)? >> shift;
            rounded.saturating_add(1).min(rows)
        };
        total = total.checked_add(level_capacity)?;
    }
    Some(total)
}

pub(super) fn input_bytes(
    rows: usize,
    cells: usize,
    value_bytes: usize,
    indexes: usize,
    index_keys: usize,
    index_effects: usize,
    index_effect_components: usize,
) -> Option<usize> {
    HEADER_BYTES
        .checked_add(TABLE_BYTES)?
        .checked_add(rows.checked_mul(ROW_BYTES)?)?
        .checked_add(cells.checked_mul(CELL_BYTES)?)?
        .checked_add(value_bytes)?
        .checked_add(indexes.checked_mul(INDEX_BYTES)?)?
        .checked_add(index_keys.checked_mul(INDEX_KEY_BYTES)?)?
        .checked_add(index_effects.checked_mul(INDEX_EFFECT_BYTES)?)?
        .checked_add(index_effect_components.checked_mul(INDEX_EFFECT_COMPONENT_BYTES)?)
}

pub(super) fn workspace_bytes(
    rows: usize,
    cells: usize,
    value_bytes: usize,
    indexes: usize,
    index_keys: usize,
    index_effects: usize,
    index_effect_components: usize,
) -> Option<usize> {
    // The original two input-sized scratch regions admit variable-width typed values without a
    // fixed TEXT ceiling. They are followed by two alternating radix-node arrays (48 bytes per
    // row), which remain the capacity needed by the older generation programs.
    let scratch = input_bytes(
        rows,
        cells,
        value_bytes,
        indexes,
        index_keys,
        index_effects,
        index_effect_components,
    )?
    .checked_add(512)?;
    let established_capacity = scratch
        .checked_mul(2)?
        .checked_add(16)?
        .checked_add(rows.checked_mul(48)?.checked_mul(2)?)?;

    // V3's parallel row phase materializes two disjoint, variable-width preimage arenas: the
    // existing current-row commitment and the codec-5 S7 final-row commitment consumed by the
    // live WAL writer.  They must not alias while rows execute concurrently. Express their
    // capacity in the generic row/cell/value geometry, then retain the larger of this and the
    // established program capacity. No SQL type gets a separate arena or route.
    let parallel_row_preimages = rows
        .checked_mul(96)?
        .checked_add(cells.checked_mul(72)?)?
        .checked_add(rows.checked_mul(67)?)?
        .checked_add(cells.checked_mul(25)?)?
        .checked_add(value_bytes)?;
    let v3_capacity = parallel_row_preimages
        .checked_add(512)?
        .checked_add(rows.checked_mul(48)?.checked_mul(2)?)?;
    // The table-map prefix states are transient finalizer scratch. They begin after *both* the
    // legacy/index effect arena and v3's row/tree arenas, so indexed finalization cannot clobber
    // a state another finalizer thread will consume after its block barrier.
    established_capacity
        .max(v3_capacity)
        .checked_add(3)
        .map(|bytes| bytes & !3)
        .and_then(|bytes| bytes.checked_add(V3_MAP_NODE_PREFIX_STATE_BYTES))
}

pub(super) fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn put_i16(bytes: &mut [u8], offset: usize, value: i16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed ABI range"),
    )
}
