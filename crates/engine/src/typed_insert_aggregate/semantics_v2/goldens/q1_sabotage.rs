//! Coherently rehashed Q1 hostile vectors.
//!
//! Each fixture retains valid wire framing and pass-zero hash closure, then asks the test-only
//! retained codec closure to reject one witness-free semantic lie.  None is a catalog, allocator,
//! generation, recovery, WAL, device, or publication test.

use super::*;
use sha2::{Digest, Sha256};

const ABSENT: u32 = u32::MAX;

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("Q1 u32 field"))
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("Q1 u64 field"))
}

fn directory(s7: &[u8], ordinal: usize) -> usize {
    u64_at(s7, 104 + ordinal * 16) as usize
}

fn exact(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    v2_digest(domain, fields)
}

fn list_root(domain: &[u8], table_id: u64, values: impl Iterator<Item = [u8; 32]>) -> [u8; 32] {
    let values: Vec<_> = values.collect();
    exact(
        domain,
        &[
            &table_id.to_le_bytes(),
            &(values.len() as u32).to_le_bytes(),
            &values.concat(),
        ],
    )
}

fn rehash_table_roots(s7: &mut [u8]) {
    let table_at = directory(s7, 0);
    let transition_at = directory(s7, 7);
    let effect_at = directory(s7, 8);
    for table_ref in 0..u32_at(s7, 40) as usize {
        let table = table_at + table_ref * 384;
        let table_id = u64_at(s7, table + 8);
        let transitions = (u32_at(s7, table + 88)..u32_at(s7, table + 88) + u32_at(s7, table + 92))
            .map(|ordinal| {
                s7[transition_at + ordinal as usize * 192 + 128
                    ..transition_at + ordinal as usize * 192 + 160]
                    .try_into()
                    .expect("Q1 transition digest")
            });
        let effects = (u32_at(s7, table + 104)..u32_at(s7, table + 104) + u32_at(s7, table + 108))
            .map(|ordinal| {
                s7[effect_at + ordinal as usize * 192 + 128
                    ..effect_at + ordinal as usize * 192 + 160]
                    .try_into()
                    .expect("Q1 effect digest")
            });
        let transition_root = list_root(
            b"gpu-db/write001/s7-table-transition-root/v2",
            table_id,
            transitions,
        );
        let effect_root = list_root(
            b"gpu-db/write001/s7-table-index-effect-root/v2",
            table_id,
            effects,
        );
        s7[table + 288..table + 320].copy_from_slice(&transition_root);
        s7[table + 320..table + 352].copy_from_slice(&effect_root);
    }
}

fn rehash_table_manifests(s7: &mut [u8]) {
    let table_at = directory(s7, 0);
    let disposition_at = directory(s7, 1);
    let dependency_at = directory(s7, 3);
    let index_at = directory(s7, 5);
    let transition_at = directory(s7, 7);
    let effect_at = directory(s7, 8);
    let image_descriptor_at = directory(s7, 11);
    for table_ref in 0..u32_at(s7, 40) as usize {
        let table = table_at + table_ref * 384;
        let target = u32_at(s7, table + 20) as usize;
        let disposition_start = u32_at(s7, table + 80) as usize;
        let disposition_count = u32_at(s7, table + 84) as usize;
        let index_start = u32_at(s7, table + 96) as usize;
        let index_count = u32_at(s7, table + 100) as usize;
        let transition_start = u32_at(s7, table + 88) as usize;
        let transition_count = u32_at(s7, table + 92) as usize;
        let effect_start = u32_at(s7, table + 104) as usize;
        let effect_count = u32_at(s7, table + 108) as usize;
        let image_ref = u32_at(s7, table + 112) as usize;
        let mut hash = Sha256::new();
        let domain = b"gpu-db/write001/s7-table-manifest/v2";
        hash.update((domain.len() as u64).to_le_bytes());
        hash.update(domain);
        hash.update(&s7[table..table + 352]);
        hash.update([0; 32]);
        hash.update(&s7[dependency_at + target * 224 + 192..dependency_at + target * 224 + 224]);
        for ordinal in disposition_start..disposition_start + disposition_count {
            hash.update(&s7[disposition_at + ordinal * 32..disposition_at + (ordinal + 1) * 32]);
        }
        for ordinal in index_start..index_start + index_count {
            hash.update(&s7[index_at + ordinal * 384 + 304..index_at + ordinal * 384 + 336]);
        }
        for ordinal in transition_start..transition_start + transition_count {
            hash.update(
                &s7[transition_at + ordinal * 192 + 128..transition_at + ordinal * 192 + 160],
            );
        }
        for ordinal in effect_start..effect_start + effect_count {
            hash.update(&s7[effect_at + ordinal * 192 + 128..effect_at + ordinal * 192 + 160]);
        }
        hash.update(
            &s7[image_descriptor_at + image_ref * 160 + 128
                ..image_descriptor_at + image_ref * 160 + 160],
        );
        let manifest: [u8; 32] = hash.finalize().into();
        s7[table + 352..table + 384].copy_from_slice(&manifest);
    }
}

fn rehash_root_descriptor(s7: &mut [u8]) {
    let table_at = directory(s7, 0);
    let mut hash = Sha256::new();
    let domain = b"gpu-db/write001/s7-root-descriptor/v2";
    hash.update((domain.len() as u64).to_le_bytes());
    hash.update(domain);
    hash.update(&s7[30..32]);
    hash.update(&s7[328..536]);
    hash.update(&s7[40..44]);
    for table_ref in 0..u32_at(s7, 40) as usize {
        let table = table_at + table_ref * 384;
        hash.update(&s7[table..table + 4]);
        hash.update(&s7[table + 4..table + 8]);
        hash.update(&s7[table + 8..table + 16]);
        hash.update(&s7[table + 32..table + 48]);
        hash.update(&s7[table + 160..table + 224]);
        hash.update(&s7[table + 352..table + 384]);
    }
    let root: [u8; 32] = hash.finalize().into();
    s7[536..568].copy_from_slice(&root);
}

fn rehash_payload(s7: &mut [u8]) -> [u8; 32] {
    let bytes = s7.len() as u64;
    let payload = exact(
        b"gpu-db/write001/s7-payload/v2",
        &[&bytes.to_le_bytes(), &s7[..568], &[0; 32], &s7[600..]],
    );
    s7[568..600].copy_from_slice(&payload);
    payload
}

fn aggregate_request_digest(s1: &[u8], s7: &[u8]) -> [u8; 32] {
    let statement_count = u32_at(s7, 44) as usize;
    let resolutions = directory(s7, 2);
    let projections = directory(s7, 10);
    let mut hash = Sha256::new();
    let domain = b"gpu-db/write001/aggregate-request/v2";
    hash.update((domain.len() as u64).to_le_bytes());
    hash.update(domain);
    hash.update([2]);
    hash.update((statement_count as u32).to_le_bytes());
    for statement in 0..statement_count {
        let resolution = resolutions + statement * 320;
        let projection_start = u32_at(s7, resolution + 48) as usize;
        let projection_count = u32_at(s7, resolution + 52) as usize;
        hash.update((statement as u32).to_le_bytes());
        hash.update(&s1[statement * 144 + 16..statement * 144 + 48]);
        hash.update((projection_count as u32).to_le_bytes());
        for projection in projection_start..projection_start + projection_count {
            hash.update(
                &s7[projections + projection * 128 + 38..projections + projection * 128 + 40],
            );
        }
    }
    hash.finalize().into()
}

fn reframed(
    base: &q1_vectors::Q1Fixture,
    sections: [Vec<u8>; AGGREGATE_SECTION_COUNT],
) -> q1_vectors::Q1Fixture {
    reframed_with_request(base, sections, base.outer.request_digest)
}

fn reframed_explicit_abort(
    base: &q1_vectors::Q1Fixture,
    mut sections: [Vec<u8>; AGGREGATE_SECTION_COUNT],
) -> q1_vectors::Q1Fixture {
    let (root, payload, manifest, overlay) = {
        let s7 = &mut sections[6];
        let payload = rehash_payload(s7);
        let table_at = directory(s7, 0);
        (
            s7[536..568].try_into().expect("Q1 root descriptor"),
            payload,
            s7[table_at + 352..table_at + 384]
                .try_into()
                .expect("Q1 explicit manifest"),
            s7[504..536].try_into().expect("Q1 overlay"),
        )
    };
    q1_vectors::finish_fixture(
        sections,
        base.outer.request_digest,
        1,
        2,
        0,
        2,
        [2, 2, 0, 2, 1, 2, 1, 0],
        base.outcome.clone(),
        root,
        payload,
        manifest,
        overlay,
    )
}

fn reframed_with_request(
    base: &q1_vectors::Q1Fixture,
    mut sections: [Vec<u8>; AGGREGATE_SECTION_COUNT],
    request_digest: [u8; 32],
) -> q1_vectors::Q1Fixture {
    let (root, payload, manifest, overlay) = {
        let s7 = &mut sections[6];
        let payload = rehash_payload(s7);
        let table_at = directory(s7, 0);
        (
            s7[536..568].try_into().expect("Q1 root descriptor"),
            payload,
            s7[table_at + 352..table_at + 384]
                .try_into()
                .expect("Q1 parent manifest"),
            s7[504..536].try_into().expect("Q1 overlay"),
        )
    };
    q1_vectors::finish_fixture(
        sections,
        request_digest,
        2,
        3,
        3,
        3,
        [3, 3, 0, 3, 1, 3, 1, 0],
        base.outcome.clone(),
        root,
        payload,
        manifest,
        overlay,
    )
}

fn finalise_table_chain(s7: &mut [u8]) {
    rehash_transition_digests(s7);
    rehash_table_roots(s7);
    rehash_table_manifests(s7);
    rehash_root_descriptor(s7);
}

fn rehash_transition_digests(s7: &mut [u8]) {
    let transitions = directory(s7, 7);
    let effects = directory(s7, 8);
    for ordinal in 0..u32_at(s7, 68) as usize {
        let transition = transitions + ordinal * 192;
        let effect_start = u32_at(s7, transition + 40) as usize;
        let effect_count = u32_at(s7, transition + 44) as usize;
        let effect_digests: Vec<u8> = (effect_start..effect_start + effect_count)
            .flat_map(|effect| {
                s7[effects + effect * 192 + 128..effects + effect * 192 + 160]
                    .iter()
                    .copied()
            })
            .collect();
        let digest = exact(
            b"gpu-db/write001/s7-transition/v2",
            &[
                &s7[transition..transition + 128],
                &[0; 32],
                &s7[transition + 160..transition + 192],
                &effect_digests,
            ],
        );
        s7[transition + 128..transition + 160].copy_from_slice(&digest);
    }
}

fn rehash_table_dependency_token(s7: &mut [u8], dependency_ref: usize) {
    let dependencies = directory(s7, 3);
    let token = dependencies + dependency_ref * 224;
    assert!(
        matches!(s7[token + 4], 1 | 2),
        "Q1 hostile table token has a table-object identity"
    );
    let identity = exact(
        b"gpu-db/write001/s7-table-object/v2",
        &[
            &s7[token + 4..token + 5],
            &s7[token + 8..token + 16],
            &s7[token + 16..token + 20],
            &s7[token + 48..token + 56],
            &s7[token + 24..token + 32],
            &s7[token + 64..token + 96],
            &s7[token + 96..token + 128],
            &s7[token + 128..token + 160],
        ],
    );
    s7[token + 160..token + 192].copy_from_slice(&identity);
    let digest = exact(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&s7[token..token + 192], &[0; 32], &[0; 32], &[0; 32]],
    );
    s7[token + 192..token + 224].copy_from_slice(&digest);
}

fn rehash_published_sequence_token(s5: &[u8], s7: &mut [u8], dependency_ref: usize) {
    let dependencies = directory(s7, 3);
    let token = dependencies + dependency_ref * 224;
    assert_eq!(s7[token + 4], 8, "Q1 hostile token is a published sequence");
    let body_bytes = u32_at(s5, 16) as usize;
    assert_eq!(
        body_bytes,
        crate::ENCODED_SEQUENCE_VALUE_REFERENCE_BYTES,
        "Q1 published sequence body stays exact"
    );
    let identity = exact(
        b"gpu-db/write001/s7-published-sequence/v2",
        &[
            &s7[token + 8..token + 16],
            &s7[token + 16..token + 20],
            &s7[token + 48..token + 56],
            &s7[token + 24..token + 32],
            &s7[token + 128..token + 160],
            &s7[token + 96..token + 128],
            &s5[52..52 + body_bytes],
        ],
    );
    s7[token + 160..token + 192].copy_from_slice(&identity);
    let digest = exact(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&s7[token..token + 192], &[0; 32], &[0; 32], &[0; 32]],
    );
    s7[token + 192..token + 224].copy_from_slice(&digest);
}

fn rehash_constraint_token(s7: &mut [u8], dependency_ref: usize) {
    let dependencies = directory(s7, 3);
    let tables = directory(s7, 0);
    let token = dependencies + dependency_ref * 224;
    assert!(
        matches!(s7[token + 4], 9..=11),
        "Q2 hostile token is a static catalog guard"
    );
    let table_ref = u32_at(s7, token + 20) as usize;
    let table = tables + table_ref * 384;
    let identity = exact(
        b"gpu-db/write001/s7-constraint-object/v2",
        &[
            &s7[token + 4..token + 5],
            &s7[token + 8..token + 16],
            &s7[token + 16..token + 20],
            &s7[table + 8..table + 16],
            &s7[token + 48..token + 56],
            &s7[token + 24..token + 32],
            &s7[token + 64..token + 96],
            &s7[token + 96..token + 128],
            &s7[token + 128..token + 160],
        ],
    );
    s7[token + 160..token + 192].copy_from_slice(&identity);
    let digest = exact(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&s7[token..token + 192], &[0; 32], &[0; 32], &[0; 32]],
    );
    s7[token + 192..token + 224].copy_from_slice(&digest);
}

/// A coherent Q1-valid substitution which swaps the two table-local static NOT NULL guard
/// tokens. The pinned catalog remains unchanged and must reject this at Q2 guard closure.
pub(super) fn q2_owner_swapped_guard_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    let s7 = &mut sections[6];
    let dependencies = directory(s7, 3);
    let uses = directory(s7, 4);
    let parent = dependencies + 14 * 224;
    let child = dependencies + 15 * 224;
    assert_eq!(s7[parent + 4], 9, "Q2 parent guard is NOT NULL");
    assert_eq!(s7[child + 4], 9, "Q2 child guard is NOT NULL");
    assert_eq!(u32_at(s7, parent + 20), 0, "Q2 parent guard target");
    assert_eq!(u32_at(s7, child + 20), 1, "Q2 child guard target");

    s7[parent + 20..parent + 24].copy_from_slice(&1_u32.to_le_bytes());
    s7[child + 20..child + 24].copy_from_slice(&0_u32.to_le_bytes());
    let parent_name: [u8; 32] = s7[parent + 128..parent + 160]
        .try_into()
        .expect("Q2 parent guard name");
    let child_name: [u8; 32] = s7[child + 128..child + 160]
        .try_into()
        .expect("Q2 child guard name");
    s7[parent + 128..parent + 160].copy_from_slice(&child_name);
    s7[child + 128..child + 160].copy_from_slice(&parent_name);
    rehash_constraint_token(s7, 14);
    rehash_constraint_token(s7, 15);

    for ordinal in 0..u32_at(s7, 56) as usize {
        let usage = uses + ordinal * 32;
        if u16::from_le_bytes(
            s7[usage + 8..usage + 10]
                .try_into()
                .expect("Q2 guard use role"),
        ) == 9
        {
            let dependency_ref = u32_at(s7, usage + 4);
            if dependency_ref == 14 {
                s7[usage + 4..usage + 8].copy_from_slice(&15_u32.to_le_bytes());
            } else if dependency_ref == 15 {
                s7[usage + 4..usage + 8].copy_from_slice(&14_u32.to_le_bytes());
            }
        }
    }
    rehash_overlay_chain(&mut sections);
    reframed(&base, sections)
}

/// A coherent wire-level root substitution.  Every affected S7 digest is repaired so codec
/// closure accepts it; Q2's unchanged independent builder must still reject the substituted
/// final roots against its fixed output authority.
pub(super) fn q2_repaired_root_substitution_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    let s7 = &mut sections[6];
    let parent = directory(s7, 0);
    s7[parent + 192] ^= 0x5a;
    s7[440] ^= 0xa5;
    rehash_table_manifests(s7);
    rehash_root_descriptor(s7);
    reframed(&base, sections)
}

/// Recreates the pre-repair shape where the first statement has no ordinary NOT NULL witness
/// for `required`. The dependency and its use are physically absent, leaving only the terminal
/// guard for the aborting statement. Wire framing and every affected root are rebuilt so codec
/// closure accepts the exact old shape; only the pinned Q2 catalog guard bijection may reject the
/// missing `required` witness.
pub(super) fn q2_explicit_missing_first_ordinary_guard_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::explicit_abort_fixture();
    let mut sections = base.sections.clone();
    {
        let s7 = &mut sections[6];
        let uses = directory(s7, 4);
        let dependencies = directory(s7, 3);
        assert_eq!(u32_at(s7, 52), 4, "Q2 repaired dependency count");
        assert_eq!(u32_at(s7, 56), 5, "Q2 repaired use count");
        assert_eq!(
            uses,
            dependencies + 4 * 224,
            "Q2 repaired dependency extent"
        );
        assert_eq!(u32_at(s7, uses + 32), 0, "Q2 first guard statement");
        assert_eq!(u32_at(s7, uses + 32 + 4), 2, "Q2 ordinary guard reference");
        assert_eq!(
            u16::from_le_bytes(
                s7[uses + 32 + 8..uses + 32 + 10]
                    .try_into()
                    .expect("Q2 ordinary guard role"),
            ),
            9,
            "Q2 ordinary guard role"
        );

        // Remove dependency #2 and its statement-zero use. The terminal guard becomes #2 and
        // the sections after the dependency/use arenas move left by their exact record widths.
        let terminal_token = dependencies + 3 * 224;
        let use_after_ordinary = uses + 2 * 32;
        let after_uses = uses + 5 * 32;
        let mut old_shape = Vec::with_capacity(s7.len() - 224 - 32);
        old_shape.extend_from_slice(&s7[..dependencies + 2 * 224]);
        old_shape.extend_from_slice(&s7[terminal_token..terminal_token + 224]);
        old_shape.extend_from_slice(&s7[uses..uses + 32]);
        old_shape.extend_from_slice(&s7[use_after_ordinary..after_uses]);
        old_shape.extend_from_slice(&s7[after_uses..]);
        *s7 = old_shape;

        write_u32(s7, 52, 3);
        write_u32(s7, 56, 4);
        let regions = [
            (640_u64, 384_u64),
            (1_024, 64),
            (1_088, 640),
            (1_728, 672),
            (2_400, 128),
            (2_528, 0),
            (2_528, 0),
            (2_528, 0),
            (2_528, 0),
            (2_528, 0),
            (2_528, 128),
            (2_656, 160),
            (2_816, 0),
            (2_816, u64_at(s7, 96)),
        ];
        for (ordinal, (offset, len)) in regions.iter().enumerate() {
            write_u64(s7, 104 + ordinal * 16, *offset);
            write_u64(s7, 112 + ordinal * 16, *len);
        }

        let resolutions = directory(s7, 2);
        let old_uses = directory(s7, 4);
        write_u32(s7, resolutions + 40, 0);
        write_u32(s7, resolutions + 44, 1);
        write_u32(s7, resolutions + 320 + 40, 1);
        write_u32(s7, resolutions + 320 + 44, 3);
        write_u32(s7, resolutions + 320 + 84, 2);

        let terminal = dependencies + 2 * 224;
        let terminal_use = old_uses + 3 * 32;
        assert_eq!(u32_at(s7, terminal), 3, "Q2 terminal guard old reference");
        assert_eq!(
            u32_at(s7, terminal_use + 4),
            3,
            "Q2 terminal use old reference"
        );
        write_u32(s7, terminal, 2);
        write_u32(s7, terminal_use + 4, 2);
        rehash_constraint_token(s7, 2);
        let s7_len = s7.len() as u64;
        write_u64(s7, 32, s7_len);
    }
    rehash_overlay_chain(&mut sections);
    reframed_explicit_abort(&base, sections)
}

fn s5_offset(s5: &[u8], ordinal: usize) -> usize {
    let mut offset = 0;
    for _ in 0..ordinal {
        let body_bytes = u32_at(s5, offset + 16) as usize;
        offset += 52 + body_bytes;
    }
    offset
}

fn rehash_overlay_chain(sections: &mut [Vec<u8>; AGGREGATE_SECTION_COUNT]) {
    let [s1, _s2, _s3, s4, s5, s6, s7, _s8] = sections;
    let resolutions = directory(s7, 2);
    let uses = directory(s7, 4);
    let projections = directory(s7, 10);
    let statement_count = u32_at(s7, 44) as usize;
    let mut prior: [u8; 32] = s7[472..504].try_into().expect("Q1 initial overlay root");
    for statement in 0..statement_count {
        let resolution = resolutions + statement * 320;
        let s4_start = u32_at(s7, resolution + 24) as usize;
        let s4_count = u32_at(s7, resolution + 28) as usize;
        let s5_start = u32_at(s7, resolution + 32) as usize;
        let s5_count = u32_at(s7, resolution + 36) as usize;
        let use_start = u32_at(s7, resolution + 40) as usize;
        let use_count = u32_at(s7, resolution + 44) as usize;
        let projection_start = u32_at(s7, resolution + 48) as usize;
        let projection_count = u32_at(s7, resolution + 52) as usize;

        let mut disposition_hash = Sha256::new();
        let disposition_domain = b"gpu-db/write001/s7-statement-dispositions/v2";
        disposition_hash.update((disposition_domain.len() as u64).to_le_bytes());
        disposition_hash.update(disposition_domain);
        disposition_hash.update((statement as u32).to_le_bytes());
        disposition_hash.update((s4_count as u32).to_le_bytes());
        disposition_hash.update(&s4[s4_start * 64..(s4_start + s4_count) * 64]);
        let disposition_root: [u8; 32] = disposition_hash.finalize().into();

        let mut sequence_hash = Sha256::new();
        let sequence_domain = b"gpu-db/write001/s7-statement-sequences/v2";
        sequence_hash.update((sequence_domain.len() as u64).to_le_bytes());
        sequence_hash.update(sequence_domain);
        sequence_hash.update((statement as u32).to_le_bytes());
        sequence_hash.update((s5_count as u32).to_le_bytes());
        for ordinal in s5_start..s5_start + s5_count {
            let offset = s5_offset(s5, ordinal);
            let body_bytes = u32_at(s5, offset + 16) as usize;
            sequence_hash.update(&s5[offset..offset + 52 + body_bytes]);
        }
        let sequence_root: [u8; 32] = sequence_hash.finalize().into();

        let mut dependency_hash = Sha256::new();
        let dependency_domain = b"gpu-db/write001/s7-statement-dependencies/v2";
        dependency_hash.update((dependency_domain.len() as u64).to_le_bytes());
        dependency_hash.update(dependency_domain);
        dependency_hash.update((statement as u32).to_le_bytes());
        dependency_hash.update((use_count as u32).to_le_bytes());
        for ordinal in use_start..use_start + use_count {
            let usage = uses + ordinal * 32;
            let dependency_ref = u32_at(s7, usage + 4) as usize;
            dependency_hash.update(&s7[usage..usage + 32]);
            let token = directory(s7, 3) + dependency_ref * 224;
            dependency_hash.update(&s7[token + 192..token + 224]);
        }
        let dependency_root: [u8; 32] = dependency_hash.finalize().into();

        let mut projection_hash = Sha256::new();
        let projection_domain = b"gpu-db/write001/s7-statement-projections/v2";
        projection_hash.update((projection_domain.len() as u64).to_le_bytes());
        projection_hash.update(projection_domain);
        projection_hash.update((statement as u32).to_le_bytes());
        projection_hash.update((projection_count as u32).to_le_bytes());
        for ordinal in projection_start..projection_start + projection_count {
            let projection = projections + ordinal * 128;
            projection_hash.update(&s7[projection + 96..projection + 128]);
        }
        let projection_root: [u8; 32] = projection_hash.finalize().into();

        let after = exact(
            b"gpu-db/write001/s7-statement-overlay-root/v2",
            &[
                &prior,
                &(statement as u32).to_le_bytes(),
                &s7[resolution + 128..resolution + 160],
                &s7[resolution + 160..resolution + 192],
                &disposition_root,
                &sequence_root,
                &dependency_root,
                &projection_root,
            ],
        );
        s1[statement * 144 + 80..statement * 144 + 112].copy_from_slice(&prior);
        s1[statement * 144 + 112..statement * 144 + 144].copy_from_slice(&after);
        s7[resolution + 224..resolution + 256].copy_from_slice(&prior);
        s7[resolution + 256..resolution + 288].copy_from_slice(&after);
        let s6_entry = statement * 136;
        let mut outcome =
            gpu_db_wal::decode_canonical_outcome_exact(&s6[s6_entry + 44..s6_entry + 136])
                .expect("Q1 S6 outcome decodes before rehash");
        outcome.target_digest = after;
        let mut encoded = [0; gpu_db_wal::CANONICAL_OUTCOME_BYTES];
        gpu_db_wal::encode_canonical_outcome_into_exact(&outcome, &mut encoded)
            .expect("Q1 S6 outcome reencodes");
        s6[s6_entry + 44..s6_entry + 136].copy_from_slice(&encoded);
        let outcome_digest = exact(
            b"gpu-db/write001/s7-s6-entry/v2",
            &[&s6[s6_entry..s6_entry + 136]],
        );
        s7[resolution + 288..resolution + 320].copy_from_slice(&outcome_digest);
        prior = after;
    }
    s7[504..536].copy_from_slice(&prior);
    rehash_root_descriptor(s7);
}

fn s4_stable_row_id_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    let first = u64_at(&sections[3], 8);
    let second = u64_at(&sections[3], 128 + 8);
    assert_eq!(first + 1, second, "Q1 parent S4 rows stay contiguous");
    sections[3][8..16].copy_from_slice(&(first + 1).to_le_bytes());
    sections[3][128 + 8..128 + 16].copy_from_slice(&(second + 1).to_le_bytes());
    let s7 = &mut sections[6];
    let tables = directory(s7, 0);
    let dispositions = directory(s7, 1);
    let transitions = directory(s7, 7);
    s7[tables + 48..tables + 56].copy_from_slice(&(first + 1).to_le_bytes());
    s7[tables + 56..tables + 64].copy_from_slice(&(second + 2).to_le_bytes());
    s7[dispositions + 8..dispositions + 16].copy_from_slice(&(first + 1).to_le_bytes());
    s7[dispositions + 32 + 8..dispositions + 32 + 16].copy_from_slice(&(second + 1).to_le_bytes());
    s7[transitions + 8..transitions + 16].copy_from_slice(&(first + 1).to_le_bytes());
    s7[transitions + 192 + 8..transitions + 192 + 16].copy_from_slice(&(second + 1).to_le_bytes());
    finalise_table_chain(s7);
    reframed(&base, sections)
}

fn projection_result_format_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    {
        let s7 = &mut sections[6];
        let projection = directory(s7, 10);
        assert_eq!(
            u16::from_le_bytes(
                s7[projection + 38..projection + 40]
                    .try_into()
                    .expect("Q1 projection format")
            ),
            0
        );
        s7[projection + 38..projection + 40].copy_from_slice(&1_u16.to_le_bytes());
        let digest = exact(
            b"gpu-db/write001/s7-projection/v2",
            &[&s7[projection..projection + 96], &[0; 32]],
        );
        s7[projection + 96..projection + 128].copy_from_slice(&digest);
    }
    let request_digest = aggregate_request_digest(&sections[0], &sections[6]);
    reframed_with_request(&base, sections, request_digest)
}

fn redirected_equality_use_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    let s7 = &mut sections[6];
    let uses = directory(s7, 4);
    let count = u32_at(s7, 56) as usize;
    let use_at = (0..count)
        .map(|ordinal| uses + ordinal * 32)
        .find(|offset| {
            u32_at(s7, *offset) == 2
                && u16::from_le_bytes(
                    s7[*offset + 8..*offset + 10]
                        .try_into()
                        .expect("Q1 use role"),
                ) == 3
        })
        .expect("Q1 statement two equality use");
    s7[use_at + 16..use_at + 20].copy_from_slice(&0_u32.to_le_bytes());
    reframed(&base, sections)
}

fn sequence_overwrite_flag_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    assert_eq!(sections[4][9], 1, "Q1 sequence starts as a live default");
    sections[4][9] = 3;
    reframed(&base, sections)
}

fn fk_parent_stable_owner_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    {
        let s7 = &mut sections[6];
        let dependencies = directory(s7, 3);
        let parent = dependencies + 2 * 224;
        assert_eq!(s7[parent + 4], 2, "Q1 FK parent token stays role four");
        let stable_owner = u64_at(s7, parent + 8);
        s7[parent + 8..parent + 16].copy_from_slice(&(stable_owner + 1).to_le_bytes());
        rehash_table_dependency_token(s7, 2);
    }
    rehash_overlay_chain(&mut sections);
    reframed(&base, sections)
}

fn published_sequence_name_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    {
        let [_, _, _, _, s5, _, s7, _] = &mut sections;
        let schema = b"public";
        let substituted = b"q1_substituted_sequence";
        let name_digest = exact(
            b"gpu-db/write001/s7-qualified-name/v2",
            &[
                &(schema.len() as u32).to_le_bytes(),
                schema,
                &(substituted.len() as u32).to_le_bytes(),
                substituted,
            ],
        );
        let token = directory(s7, 3) + 13 * 224;
        s7[token + 128..token + 160].copy_from_slice(&name_digest);
        rehash_published_sequence_token(s5, s7, 13);
    }
    rehash_overlay_chain(&mut sections);
    reframed(&base, sections)
}

fn changed_table_unchanged_database_root_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    let s7 = &mut sections[6];
    assert_ne!(
        &s7[408..440],
        &s7[440..472],
        "Q1 changed-table fixture starts with distinct database roots"
    );
    let initial_root: [u8; 32] = s7[408..440].try_into().expect("Q1 initial database root");
    s7[440..472].copy_from_slice(&initial_root);
    rehash_root_descriptor(s7);
    reframed(&base, sections)
}

fn manifest_root_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    let s7 = &mut sections[6];
    let table = directory(s7, 0);
    s7[table + 352] ^= 0x5a;
    rehash_root_descriptor(s7);
    reframed(&base, sections)
}

fn component_truth_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_fixture();
    let mut sections = base.sections.clone();
    let s7 = &mut sections[6];
    let keys = directory(s7, 6);
    let transitions = directory(s7, 7);
    let effects = directory(s7, 8);
    let components = directory(s7, 9);
    let component = components;
    let key_ref = u32_at(s7, component + 16) as usize;
    s7[component + 20..component + 24].copy_from_slice(&1_u32.to_le_bytes());
    let component_digest = exact(
        b"gpu-db/write001/s7-typed-key-component/v2",
        &[
            &s7[component..component + 80],
            &[0; 32],
            &s7[component + 112..component + 128],
            &s7[keys + key_ref * 112 + 72..keys + key_ref * 112 + 104],
        ],
    );
    s7[component + 80..component + 112].copy_from_slice(&component_digest);
    let effect_ref = u32_at(s7, component + 4) as usize;
    let effect = effects + effect_ref * 192;
    let component_start = u32_at(s7, effect + 28) as usize;
    let component_count = u32_at(s7, effect + 32) as usize;
    let component_digests: Vec<u8> = (component_start..component_start + component_count)
        .flat_map(|ordinal| {
            s7[components + ordinal * 128 + 80..components + ordinal * 128 + 112]
                .iter()
                .copied()
        })
        .collect();
    let typed_key = exact(
        b"gpu-db/write001/s7-typed-key/v2",
        &[
            &(effect_ref as u32).to_le_bytes(),
            &[2],
            &(component_count as u32).to_le_bytes(),
            &component_digests,
        ],
    );
    s7[effect + 96..effect + 128].copy_from_slice(&typed_key);
    let index_ref = u32_at(s7, effect + 12) as usize;
    let indexes = directory(s7, 5);
    let effect_digest = exact(
        b"gpu-db/write001/s7-key-effect/v2",
        &[
            &s7[effect..effect + 16],
            &ABSENT.to_le_bytes(),
            &s7[effect + 20..effect + 128],
            &[0; 32],
            &s7[effect + 160..effect + 192],
            &s7[indexes + index_ref * 384 + 304..indexes + index_ref * 384 + 336],
            &component_digests,
        ],
    );
    s7[effect + 128..effect + 160].copy_from_slice(&effect_digest);
    for ordinal in 0..u32_at(s7, 68) as usize {
        let transition = transitions + ordinal * 192;
        let start = u32_at(s7, transition + 40) as usize;
        let count = u32_at(s7, transition + 44) as usize;
        if (start..start + count).contains(&effect_ref) {
            let effect_digests: Vec<u8> = (start..start + count)
                .flat_map(|effect| {
                    s7[effects + effect * 192 + 128..effects + effect * 192 + 160]
                        .iter()
                        .copied()
                })
                .collect();
            let transition_digest = exact(
                b"gpu-db/write001/s7-transition/v2",
                &[
                    &s7[transition..transition + 128],
                    &[0; 32],
                    &s7[transition + 160..transition + 192],
                    &effect_digests,
                ],
            );
            s7[transition + 128..transition + 160].copy_from_slice(&transition_digest);
        }
    }
    finalise_table_chain(s7);
    reframed(&base, sections)
}

fn assert_codec_closure_rejection(fixture: q1_vectors::Q1Fixture, expected: &str) {
    assert_close_rejection(fixture, "codec closure", expected);
}

fn assert_close_rejection(fixture: q1_vectors::Q1Fixture, phase: &str, expected: &str) {
    let fragments = [
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::RowMutation,
            body: &fixture.fragment_body,
        },
        gpu_db_wal::CanonicalFragmentRef {
            kind: gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus,
            body: &fixture.status,
        },
    ];
    measure_canonical_semantics_v2(&fixture.outer, &fixture.outcome, &fragments)
        .expect("coherently rehashed hostile vector passes raw pass zero");
    let error = close_canonical_semantics_v2_for_test(&fixture.outer, &fixture.outcome, &fragments)
        .expect_err("codec closure rejects the hostile semantic substitution");
    let text = error.to_string();
    assert!(
        text.contains(phase),
        "rejection stays in its expected retained phase ({phase}): {text}"
    );
    assert!(
        text.contains(expected),
        "rejection names its hostile boundary: {text}"
    );
}

fn final_image_cell_fixture() -> q1_vectors::Q1Fixture {
    let base = q1_vectors::successful_a_b_a_with_rehashed_parent_image_cell();
    let mut sections = base.sections.clone();
    let s7 = &mut sections[6];
    rehash_table_manifests(s7);
    rehash_root_descriptor(s7);
    reframed(&base, sections)
}

#[test]
fn q1_rehashed_final_image_cell_substitution_reaches_codec_closure() {
    assert_codec_closure_rejection(final_image_cell_fixture(), "transition");
}

#[test]
fn q1_rehashed_s4_stable_row_id_reaches_codec_closure() {
    assert_codec_closure_rejection(s4_stable_row_id_fixture(), "transition");
}

#[test]
fn q1_rehashed_component_cell_truth_reaches_codec_closure() {
    assert_codec_closure_rejection(component_truth_fixture(), "typed key component");
}

#[test]
fn q1_rehashed_equality_use_redirect_reaches_codec_closure() {
    assert_codec_closure_rejection(redirected_equality_use_fixture(), "equality");
}

#[test]
fn q1_rehashed_published_sequence_overwrite_flag_reaches_codec_closure() {
    assert_codec_closure_rejection(sequence_overwrite_flag_fixture(), "S5 published sequence");
}

#[test]
fn q1_rehashed_fk_parent_stable_owner_substitution_reaches_codec_closure() {
    assert_codec_closure_rejection(
        fk_parent_stable_owner_fixture(),
        "FK descriptor and parent token",
    );
}

#[test]
fn q1_rehashed_published_sequence_name_substitution_reaches_codec_closure() {
    assert_codec_closure_rejection(
        published_sequence_name_fixture(),
        "published-sequence token name",
    );
}

#[test]
fn q1_rehashed_changed_table_unchanged_database_root_reaches_codec_closure() {
    assert_codec_closure_rejection(
        changed_table_unchanged_database_root_fixture(),
        "database root does not exactly reflect",
    );
}

#[test]
fn q1_rehashed_projection_result_format_reaches_codec_closure() {
    assert_codec_closure_rejection(
        projection_result_format_fixture(),
        "S6 logical RETURNING digest",
    );
}

#[test]
fn q1_rehashed_manifest_root_chain_reaches_codec_closure() {
    assert_codec_closure_rejection(manifest_root_fixture(), "manifest");
}
