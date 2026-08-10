//! S7 envelope/root framing checks independent of directory semantics.

use super::{
    begin_v2_digest, error, read_u64, DecodedAggregateFraming, EngineError, S7_HEADER_BYTES,
};
use sha2::Digest;

pub(super) fn validate_s7_header_roots(
    framing: &DecodedAggregateFraming<'_>,
    outer: &gpu_db_wal::CanonicalPreApplyHeader,
    header: &[u8; S7_HEADER_BYTES as usize],
) -> Result<(), EngineError> {
    let initial_database_root = &header[408..440];
    let final_database_root = &header[440..472];
    let initial_overlay_root = &header[472..504];
    let final_overlay_root = &header[504..536];
    let root_descriptor = &header[536..568];
    let payload_digest = &header[568..600];
    if initial_database_root == [0; 32]
        || final_database_root == [0; 32]
        || initial_overlay_root == [0; 32]
        || final_overlay_root == [0; 32]
        || root_descriptor == [0; 32]
        || payload_digest == [0; 32]
    {
        return Err(error("S7 root identity or digest is zero"));
    }
    let has_catalog =
        framing.header_scalars().flags & crate::typed_insert_aggregate::AGGREGATE_FLAG_CATALOG != 0;
    let catalog_boundary_is_exact = if has_catalog {
        outer.catalog_after_epoch == outer.catalog_before_epoch.checked_add(1).unwrap_or(0)
            && outer.catalog_after_digest != outer.catalog_before_digest
    } else {
        outer.catalog_after_epoch == outer.catalog_before_epoch
            && outer.catalog_after_digest == outer.catalog_before_digest
    };
    if read_u64(header, 328) != outer.catalog_before_epoch
        || read_u64(header, 336) != outer.catalog_after_epoch
        || header[344..376] != outer.catalog_before_digest
        || header[376..408] != outer.catalog_after_digest
        || !catalog_boundary_is_exact
        || outer.catalog_before_digest == [0; 32]
        || outer.catalog_after_digest == [0; 32]
    {
        return Err(error(
            "S7 catalog echoes do not equal the immutable outer header",
        ));
    }
    framing.with_section_reader(0, |s1| {
        let first = s1.exact::<144>()?;
        let mut last = first;
        while s1.remaining() != 0 {
            last = s1.exact::<144>()?;
        }
        if first[80..112] != *initial_overlay_root || last[112..144] != *final_overlay_root {
            return Err(error("S7 initial/final overlay roots do not close over S1"));
        }
        Ok(())
    })
}

/// `s7_payload_digest` uses the frozen v2 `D` primitive (domain length followed by exact
/// fields), not the v1 aggregate helper that length-prefixes every part.
pub(super) fn validate_s7_payload_digest(
    framing: &DecodedAggregateFraming<'_>,
    header: &[u8; S7_HEADER_BYTES as usize],
) -> Result<(), EngineError> {
    let declared = &header[568..600];
    let computed: [u8; 32] = framing.with_section_reader(6, |reader| {
        let observed_header = reader.exact::<{ S7_HEADER_BYTES as usize }>()?;
        if observed_header != *header {
            return Err(error("S7 header changed between pass-zero rescans"));
        }
        let mut digest = begin_v2_digest(b"gpu-db/write001/s7-payload/v2");
        digest.update(
            u64::from_le_bytes(header[32..40].try_into().expect("fixed S7 total")).to_le_bytes(),
        );
        digest.update(&header[..568]);
        digest.update([0; 32]);
        digest.update(&header[600..]);
        let mut scratch = [0_u8; 4096];
        while reader.remaining() != 0 {
            let take = usize::try_from(reader.remaining().min(scratch.len() as u64))
                .map_err(|_| error("S7 digest scratch length is not addressable"))?;
            reader.copy_exact(&mut scratch[..take])?;
            digest.update(&scratch[..take]);
        }
        Ok(digest.finalize().into())
    })?;
    if declared != computed {
        return Err(error("S7 payload digest is invalid"));
    }
    Ok(())
}
