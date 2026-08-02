//! Closed host/device ABI for the V1 single-table INT4 rebuild program.
//!
//! The descriptor is encoded explicitly as little-endian bytes. Keeping offsets here prevents
//! host Rust layout, CUDA ABI alignment, and user-provided layout metadata from becoming an
//! accidental logical-root input.

pub(super) const DESCRIPTOR_HEADER_BYTES: usize = 80;
pub(super) const SHARD_DESCRIPTOR_BYTES: usize = 56;
pub(super) const OUTPUT_STATUS_BYTES: usize = 4;
// Closed V1 output: table proof roots, 65 empty table-map roots, one table-map leaf, the
// 64-node one-entry COW path, then the database root. The map path is necessary to import the
// sole immutable one-table publication map; no status/index or multi-table completion exists.
pub(super) const PROOF_DIGEST_SLOTS: usize = 200;
pub(super) const OUTPUT_BYTES: usize = OUTPUT_STATUS_BYTES + PROOF_DIGEST_SLOTS * 32;

pub(super) const DIGEST_BYTES: u64 = 32;
const VECTOR_FRAME_BYTES: u64 = 16;
const ROW_NODE_BYTES: u64 = 48;
const ROW_MAP_SCRATCH_NODE_COUNT: u64 = 2;
pub(super) const TYPED_VECTOR_PROOF_DOMAIN_BYTES: u64 =
    b"gpu-db/runtime-generation/rebuild-proof/typed-vector/v1".len() as u64;
pub(super) const ROW_VECTOR_PROOF_DOMAIN_BYTES: u64 =
    b"gpu-db/runtime-generation/rebuild-proof/current-rows/v1".len() as u64;

pub(super) const DATABASE_ID_OFFSET: usize = 0;
pub(super) const TABLE_ID_OFFSET: usize = 16;
pub(super) const DATA_GENERATION_OFFSET: usize = 24;
pub(super) const ROW_COUNT_OFFSET: usize = 32;
pub(super) const VISIBILITY_CUT_OFFSET: usize = 40;
pub(super) const COLUMN_ID_OFFSET: usize = 48;
pub(super) const ATTNUM_OFFSET: usize = 56;
pub(super) const DECLARED_OID_OFFSET: usize = 60;
pub(super) const SIGNED_SIZE_OFFSET: usize = 64;
pub(super) const ROOT_FORMAT_OFFSET: usize = 66;
pub(super) const SHARD_COUNT_OFFSET: usize = 68;
pub(super) const WORKSPACE_POINTER_OFFSET: usize = 72;

pub(super) const SHARD_ROW_START_OFFSET: usize = 0;
pub(super) const SHARD_ROW_COUNT_OFFSET: usize = 8;
pub(super) const SHARD_ROW_ID_POINTER_OFFSET: usize = 16;
pub(super) const SHARD_VALIDITY_POINTER_OFFSET: usize = 24;
pub(super) const SHARD_VALUE_POINTER_OFFSET: usize = 32;
pub(super) const SHARD_CREATED_POINTER_OFFSET: usize = 40;
pub(super) const SHARD_DELETED_POINTER_OFFSET: usize = 48;

#[cfg(test)]
pub(super) const SLOT_COLUMN_SHAPE: usize = 0;
#[cfg(test)]
pub(super) const SLOT_TYPED_VECTOR: usize = 1;
#[cfg(test)]
pub(super) const SLOT_CURRENT_ROW_LEAVES: usize = 2;
#[cfg(test)]
pub(super) const SLOT_ROW_EMPTY: usize = 3;
pub(super) const SLOT_TABLE_ROOT: usize = 68;
pub(super) const SLOT_TABLE_MAP_EMPTY: usize = 69;
pub(super) const SLOT_TABLE_MAP_LEAF: usize = 134;
pub(super) const SLOT_TABLE_MAP_PATH: usize = 135;
pub(super) const SLOT_DATABASE_ROOT: usize = 199;

pub(super) fn descriptor_bytes(shards: usize) -> Option<usize> {
    DESCRIPTOR_HEADER_BYTES.checked_add(shards.checked_mul(SHARD_DESCRIPTOR_BYTES)?)
}

pub(super) struct WorkspaceLayout {
    pub(super) bytes: usize,
    pub(super) scratch_bytes: u64,
}

/// The two proof vectors share the scratch arena serially. Their preimages include an encoded
/// domain length, the domain bytes, and the encoded row count before the digest vector.
pub(super) fn proof_vector_preimage_bytes(rows: u64) -> Option<[u64; 2]> {
    let framed = |domain_bytes| {
        VECTOR_FRAME_BYTES
            .checked_add(domain_bytes)?
            .checked_add(rows.checked_mul(DIGEST_BYTES)?)
    };
    Some([
        framed(TYPED_VECTOR_PROOF_DOMAIN_BYTES)?,
        framed(ROW_VECTOR_PROOF_DOMAIN_BYTES)?,
    ])
}

#[cfg(test)]
pub(super) fn proof_vector_row_ceiling() -> u64 {
    let longest_domain = TYPED_VECTOR_PROOF_DOMAIN_BYTES.max(ROW_VECTOR_PROOF_DOMAIN_BYTES);
    (u64::from(u32::MAX) - VECTOR_FRAME_BYTES - longest_domain) / DIGEST_BYTES
}

pub(super) fn workspace_layout(rows: u64) -> Option<WorkspaceLayout> {
    // typed roots + current-row roots + shared vector/map scratch. The map needs two complete
    // radix levels; vector frames are reused before the map starts.
    let typed_and_leaves = rows.checked_mul(DIGEST_BYTES)?.checked_mul(2)?;
    let vector_scratch = proof_vector_preimage_bytes(rows)?.into_iter().max()?;
    let row_map_scratch = rows
        .checked_mul(ROW_NODE_BYTES)?
        .checked_mul(ROW_MAP_SCRATCH_NODE_COUNT)?;
    let scratch_bytes = vector_scratch.max(row_map_scratch);
    let bytes = typed_and_leaves
        .checked_add(scratch_bytes)?
        .checked_add(128)?;
    Some(WorkspaceLayout {
        bytes: usize::try_from(bytes).ok()?,
        scratch_bytes,
    })
}

pub(super) fn workspace_bytes(rows: u64) -> Option<usize> {
    workspace_layout(rows).map(|layout| layout.bytes)
}

pub(super) fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_proof_slots_cover_the_closed_v1_single_table_contract() {
        assert_eq!(SLOT_COLUMN_SHAPE, 0);
        assert_eq!(SLOT_TYPED_VECTOR, 1);
        assert_eq!(SLOT_CURRENT_ROW_LEAVES, 2);
        assert_eq!(SLOT_ROW_EMPTY + 64, SLOT_TABLE_ROOT - 1);
        assert_eq!(SLOT_TABLE_MAP_EMPTY + 64, SLOT_TABLE_MAP_LEAF - 1);
        assert_eq!(SLOT_TABLE_MAP_PATH + 63, SLOT_DATABASE_ROOT - 1);
        assert_eq!(SLOT_DATABASE_ROOT + 1, PROOF_DIGEST_SLOTS);
        assert_eq!(OUTPUT_BYTES, 4 + 200 * 32);
    }

    #[test]
    fn proof_vector_frames_include_both_prefixes_at_the_u32_cursor_boundary() {
        let former_vector_only_ceiling = u64::from(u32::MAX) / DIGEST_BYTES;
        let frames = proof_vector_preimage_bytes(former_vector_only_ceiling)
            .expect("former vector-only ceiling has representable u64 geometry");
        assert!(frames.iter().all(|bytes| *bytes > u64::from(u32::MAX)));

        let admitted = proof_vector_row_ceiling();
        assert_eq!(former_vector_only_ceiling, admitted + 2);
        let frames = proof_vector_preimage_bytes(admitted)
            .expect("near-boundary framed geometry remains representable");
        assert!(frames.iter().all(|bytes| *bytes <= u64::from(u32::MAX)));
        let layout = workspace_layout(admitted).expect("admitted workspace geometry");
        assert!(frames.iter().all(|bytes| *bytes <= layout.scratch_bytes));
    }
}
