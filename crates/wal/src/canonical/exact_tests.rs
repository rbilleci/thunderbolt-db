use super::*;

fn digest(byte: u8) -> CanonicalDigest {
    [byte; 32]
}

fn identity() -> CanonicalIdentity {
    CanonicalIdentity {
        database_id: [1; 16],
        cluster_id: [2; 16],
        timeline_id: [3; 16],
        format_epoch: 7,
    }
}

fn header(operation_count: u32) -> CanonicalPreApplyHeader {
    CanonicalPreApplyHeader {
        identity: identity(),
        leader_epoch: 11,
        commit_seq: 19,
        stable_transaction_id: 23,
        request_digest: digest(4),
        isolation: CanonicalIsolation::ReadCommitted,
        flags: 5,
        catalog_before_epoch: 29,
        catalog_after_epoch: 30,
        catalog_before_digest: digest(6),
        catalog_after_digest: digest(7),
        operation_count,
        table_block_count: 1,
        allocator_high_water: 31,
    }
}

fn physical() -> CanonicalPhysicalRange {
    CanonicalPhysicalRange {
        log_epoch: 11,
        lane_id: 2,
        segment_id: 41,
        first_frame_ordinal: 43,
    }
}

fn outcome() -> CanonicalOutcome {
    CanonicalOutcome {
        kind: CanonicalOutcomeKind::CommitSuccess,
        affected_rows: 1,
        sqlstate: None,
        constraint_id: 0,
        target_digest: digest(8),
        returning_digest: digest(9),
    }
}

fn fragments(fragment_count: usize) -> Vec<CanonicalFragment> {
    (0..fragment_count)
        .map(|index| CanonicalFragment {
            kind: match index {
                0 => CanonicalFragmentKind::CatalogMutation,
                1 => CanonicalFragmentKind::RowMutation,
                2 => CanonicalFragmentKind::SequenceValueTransition,
                3 => CanonicalFragmentKind::AllocatorLease,
                _ => CanonicalFragmentKind::TransactionClaimStatus,
            },
            body: (0..(index * 17 + 3))
                .map(|offset| u8::try_from(index * 19 + offset).expect("small fixture byte"))
                .collect(),
        })
        .collect()
}

fn assert_rejected_before_record_exposure(
    header: &CanonicalPreApplyHeader,
    fragments: &[CanonicalFragment],
    footprint: CanonicalWalFootprint,
) {
    let packed_len = usize::try_from(footprint.packed_record_bytes).expect("fixture fits usize");
    let serialized_len =
        usize::try_from(footprint.serialized_record_bytes).expect("fixture fits usize");
    for actual in [packed_len - 1, packed_len + 1] {
        let mut packed = vec![0xa5; actual];
        let mut serialized = vec![0x5a; serialized_len];
        let packed_before = packed.clone();
        let serialized_before = serialized.clone();
        assert!(encode_canonical_record_exact_into(
            header.stable_transaction_id,
            physical(),
            header,
            fragments,
            &outcome(),
            &mut packed,
            &mut serialized,
        )
        .is_err());
        assert_eq!(
            packed, packed_before,
            "invalid packed buffer remains untouched"
        );
        assert_eq!(
            serialized, serialized_before,
            "invalid packed buffer cannot expose a serialized record"
        );
    }
    for actual in [serialized_len - 1, serialized_len + 1] {
        let mut packed = vec![0xa5; packed_len];
        let mut serialized = vec![0x5a; actual];
        let packed_before = packed.clone();
        let serialized_before = serialized.clone();
        assert!(encode_canonical_record_exact_into(
            header.stable_transaction_id,
            physical(),
            header,
            fragments,
            &outcome(),
            &mut packed,
            &mut serialized,
        )
        .is_err());
        assert_eq!(
            packed, packed_before,
            "invalid serialized buffer cannot expose a packed record"
        );
        assert_eq!(
            serialized, serialized_before,
            "invalid serialized buffer remains untouched"
        );
    }
}

#[test]
fn exact_buffers_match_v1_and_reject_short_or_surplus_for_one_through_five_fragments() {
    for fragment_count in 1..=5 {
        let fragments = fragments(fragment_count);
        let header = header(u32::try_from(fragment_count).expect("fixture count fits u32"));
        let body_lengths = fragments
            .iter()
            .map(|fragment| u64::try_from(fragment.body.len()).expect("fixture length fits u64"))
            .collect::<Vec<_>>();
        let measured =
            measure_canonical_exact_buffers(physical(), &header, &body_lengths, &outcome())
                .expect("exact-buffer measurement accepts v1 fixture");
        assert_eq!(measured, canonical_wal_footprint(&body_lengths).unwrap());

        let legacy = encode_canonical_envelope(physical(), &header, &fragments, &outcome())
            .expect("v1 envelope encodes");
        let legacy_payload = pack_canonical_record_payload(&legacy).expect("v1 record packs");
        let mut legacy_serialized = Vec::new();
        crate::encode_wal_record_parts_into(
            &mut legacy_serialized,
            header.stable_transaction_id,
            &legacy_payload,
        );
        let mut packed = vec![0; usize::try_from(measured.packed_record_bytes).unwrap()];
        let mut serialized = vec![0; usize::try_from(measured.serialized_record_bytes).unwrap()];
        let exact = encode_canonical_record_exact_into(
            header.stable_transaction_id,
            physical(),
            &header,
            &fragments,
            &outcome(),
            &mut packed,
            &mut serialized,
        )
        .expect("exact caller buffers encode");

        assert_eq!(exact.footprint, measured);
        assert_eq!(exact.ordered_fragment_root, legacy.ordered_fragment_root);
        assert_eq!(exact.final_digest, legacy.final_digest);
        assert_eq!(packed, legacy_payload);
        assert_eq!(serialized, legacy_serialized);
        assert_eq!(packed.len() as u64, measured.packed_record_bytes);
        assert_eq!(serialized.len() as u64, measured.serialized_record_bytes);

        let borrowed = fragments
            .iter()
            .map(|fragment| CanonicalFragmentRef {
                kind: fragment.kind,
                body: &fragment.body,
            })
            .collect::<Vec<_>>();
        let mut borrowed_packed = vec![0; packed.len()];
        let mut borrowed_serialized = vec![0; serialized.len()];
        let borrowed_exact = encode_canonical_record_exact_from_borrowed(
            header.stable_transaction_id,
            physical(),
            &header,
            &borrowed,
            &outcome(),
            &mut borrowed_packed,
            &mut borrowed_serialized,
        )
        .expect("borrowed exact caller buffers encode");
        // Probe timings attribute separate wall-clock samples, so the two encodes share only
        // their semantic footprint and commitment.  Comparing the full encoding would make this
        // wire-contract test nondeterministic whenever `probe-timing` is enabled.
        assert_eq!(borrowed_exact.footprint, exact.footprint);
        assert_eq!(
            borrowed_exact.ordered_fragment_root,
            exact.ordered_fragment_root
        );
        assert_eq!(borrowed_exact.final_digest, exact.final_digest);
        assert_eq!(borrowed_packed, packed);
        assert_eq!(borrowed_serialized, serialized);
        assert_rejected_before_record_exposure(&header, &fragments, measured);
    }
}

#[test]
fn exact_buffer_measurement_keeps_the_codec_five_max_bound_shape_inclusive() {
    let fragment_limit = canonical_fragment_body_limit();
    let exact = [
        fragment_limit,
        fragment_limit,
        fragment_limit,
        fragment_limit - 240,
        0,
    ];
    let measured = measure_canonical_exact_buffers(physical(), &header(5), &exact, &outcome())
        .expect("five-fragment maximum pre-apply shape fits inclusively");
    assert_eq!(measured, canonical_wal_footprint(&exact).unwrap());
    assert_eq!(measured.fragment_count, 5);
    assert_eq!(measured.frame_count, 6);
    assert_eq!(measured.preapply_bytes, 64 * 1024 * 1024);

    let one_byte_over = [
        fragment_limit,
        fragment_limit,
        fragment_limit,
        fragment_limit - 239,
        0,
    ];
    assert!(
        measure_canonical_exact_buffers(physical(), &header(5), &one_byte_over, &outcome())
            .is_err()
    );
}

#[test]
fn exact_buffer_path_remains_collection_free_and_uses_shared_v1_layout_owners() {
    let source = include_str!("exact.rs");
    for forbidden in ["Vec", "BTree", "HashMap", "HashSet"] {
        assert!(
            !source.contains(forbidden),
            "exact-buffer path must not allocate {forbidden}"
        );
    }
    assert!(source.contains("canonical_wal_footprint_by_index"));
    assert!(source.contains(".encode_into"));
    assert!(source.contains("encode_frame_into"));
}
