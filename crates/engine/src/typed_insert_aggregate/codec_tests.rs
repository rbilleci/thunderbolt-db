use super::*;

const ENTRY_COUNTS: [u32; AGGREGATE_SECTION_COUNT] = [1, 1, 0, 1, 0, 1, 1, 0];

fn view<'a>(payloads: &'a [Vec<u8>; AGGREGATE_SECTION_COUNT]) -> TypedInsertAggregateView<'a> {
    TypedInsertAggregateView {
        semantics: TypedInsertAggregateSemantics::V1,
        flags: AGGREGATE_FLAG_AUTOCOMMIT,
        outer_flags: OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
        stable_transaction_id: 41,
        statement_count: 1,
        insert_statement_count: 1,
        original_inserted_row_count: 1,
        final_row_transition_count: 1,
        allocator_before: 99,
        allocator_high_water: 100,
        table_block_count: 1,
        sections: std::array::from_fn(|index| TypedInsertAggregateSectionView {
            entry_count: ENTRY_COUNTS[index],
            payload: &payloads[index],
        }),
    }
}

fn status(roots: TypedInsertAggregateStatusRoots) -> TypedInsertStatusV2 {
    TypedInsertStatusV2 {
        database_id: [1; 16],
        timeline_id: [2; 16],
        txn_id: 41,
        request_digest: [3; 32],
        isolation: 1,
        flags: 0,
        retention_deadline: 0,
        statement_count: 1,
        response_artifact_count: 0,
        statement_outcome_root: roots.statement_outcome_root,
        response_root: roots.response_root,
        aggregate_root: roots.aggregate_root,
    }
}

fn encode_fixture(
    payloads: &[Vec<u8>; AGGREGATE_SECTION_COUNT],
) -> EncodedTypedInsertAggregateBodies {
    let view = view(payloads);
    let layout = view.measure().expect("fixture layout");
    let prepared =
        prepare_typed_insert_aggregate_encoding(view, layout).expect("fixture aggregate proof");
    let roots = prepared.roots();
    let reserved = reserve_typed_insert_aggregate_bodies(layout).expect("reserve exact bodies");
    prepared
        .encode(&status(roots), reserved)
        .expect("encode exact bodies")
}

fn encoded_copies(encoded: &EncodedTypedInsertAggregateBodies) -> Vec<Vec<u8>> {
    encoded.fragment_bodies().map(<[u8]>::to_vec).collect()
}

fn body_refs(bodies: &[Vec<u8>]) -> Vec<&[u8]> {
    bodies.iter().map(Vec::as_slice).collect()
}

const SEMANTICS_V1_EXACT_BODY_GOLDEN_HEX: [&str; 2] = [
    "47505544424f50310501030024010000000000000000000001000000000000000000000024010000000000008c5b99aa6fbe9f3ec46bc18070691d913bf7c9d2e67df18470f2af3c4216b2aa475055444254584e414747310000000001000100010001000100000008000000a400000000000000290000000000000001000000010000000100000000000000010000000000000063000000000000006400000000000000010000000000000001000000010000000100000000000000310200000001000000020000000000000032320300000000000000030000000000000033333304000000010000000400000000000000343434340500000000000000050000000000000035353535350600000001000000060000000000000036363636363607000000010000000700000000000000373737373737370800000000000000080000000000000038383838383838388c5b99aa6fbe9f3ec46bc18070691d913bf7c9d2e67df18470f2af3c4216b2aa",
    "47505544425354415455533201010101010101010101010101010101020202020202020202020202020202022900000000000000030303030303030303030303030303030303030303030303030303030303030301010000000000000000000000000000010000000000000003a3a1ab320d7eccc1be21b1d6fa322a63a88dfdc5bd09b4a3665dc5a31bdb6900000000000000000000000000000000000000000000000000000000000000008c5b99aa6fbe9f3ec46bc18070691d913bf7c9d2e67df18470f2af3c4216b2aa",
];

fn decode_fixture_hex(encoded: &str) -> Vec<u8> {
    assert!(
        encoded.len().is_multiple_of(2),
        "fixture hex is byte-aligned"
    );
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digits = std::str::from_utf8(pair).expect("fixture hex is ASCII");
            u8::from_str_radix(digits, 16).expect("fixture hex digit")
        })
        .collect()
}

fn payloads_for_chunk_count(chunks: usize) -> [Vec<u8>; AGGREGATE_SECTION_COUNT] {
    assert!((1..=AGGREGATE_MAX_CHUNKS).contains(&chunks));
    let mut payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] = std::array::from_fn(|index| {
        if index == 0 {
            Vec::new()
        } else {
            vec![0x40 + index as u8]
        }
    });
    let base = view(&payloads).measure().expect("base layout").stream_bytes;
    let desired = if chunks == 1 {
        base + 97
    } else {
        AGGREGATE_CHUNK_PAYLOAD_BYTES * (chunks as u64 - 1) + 97
    };
    payloads[0].resize(
        usize::try_from(desired - base).expect("fixture payload is addressable"),
        0xa5,
    );
    assert_eq!(
        view(&payloads).measure().expect("sized layout").chunk_count as usize,
        chunks
    );
    payloads
}

#[test]
fn semantics_v1_complete_fragment_bodies_match_the_literal_wire_golden() {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![0x31 + index as u8; index + 1]);
    let actual = encoded_copies(&encode_fixture(&payloads));
    let expected: Vec<Vec<u8>> = SEMANTICS_V1_EXACT_BODY_GOLDEN_HEX
        .iter()
        .map(|encoded| decode_fixture_hex(encoded))
        .collect();
    assert_eq!(
        actual, expected,
        "semantics-v1 chunk/status bytes drifted from their literal golden"
    );
    let expected_refs = body_refs(&expected);
    let decoded = decode_typed_insert_aggregate_bodies(
        OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
        &expected_refs,
    )
    .expect("literal semantics-v1 golden remains strictly decodable");
    let expected_view = view(&payloads);
    let expected_layout = expected_view.measure().expect("golden fixture layout");
    let expected_roots = typed_insert_aggregate_status_roots(&expected_view, &expected_layout)
        .expect("golden fixture roots");
    assert_eq!(decoded.layout().measure, expected_layout.measure);
    assert_eq!(decoded.status(), &status(expected_roots));
}

#[test]
fn exact_body_roundtrip_preserves_each_section() {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8 + 1; index * 11 + 1]);
    let encoded = encode_fixture(&payloads);
    let canonical = encoded.canonical_fragment_refs();
    assert_eq!(canonical.as_slice().len(), encoded.fragment_count());
    assert!(
        canonical.as_slice()[..encoded.layout().chunk_count as usize]
            .iter()
            .all(|fragment| fragment.kind == gpu_db_wal::CanonicalFragmentKind::RowMutation)
    );
    assert_eq!(
        canonical.as_slice().last().unwrap().kind,
        gpu_db_wal::CanonicalFragmentKind::TransactionClaimStatus
    );
    let bodies: Vec<&[u8]> = encoded.fragment_bodies().collect();
    let decoded = decode_typed_insert_aggregate_bodies(
        OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
        &bodies,
    )
    .expect("decode exact bodies");
    assert_eq!(decoded.aggregate_root(), encoded.aggregate_root());
    assert_eq!(decoded.section_roots(), encoded.section_roots());
    assert_eq!(decoded.layout(), encoded.layout());
    let roots = TypedInsertAggregateStatusRoots {
        aggregate_root: encoded.aggregate_root(),
        statement_outcome_root: encoded.section_roots()[5],
        response_root: [0; 32],
    };
    assert_eq!(decoded.status(), &status(roots));
    for (index, expected) in payloads.iter().enumerate() {
        let mut actual = vec![0; expected.len()];
        decoded
            .copy_section_payload(index, &mut actual)
            .expect("copy decoded section");
        assert_eq!(&actual, expected);
    }
}

#[test]
fn returning_and_status_roots_are_closed_by_the_same_section_traversal() {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8 + 1; index + 2]);
    let mut returning = view(&payloads);
    returning.flags |= AGGREGATE_FLAG_RETURNING;
    returning.outer_flags |= OUTER_CONTENT_RETURNING;
    let layout = returning.measure().expect("RETURNING layout");
    let roots =
        typed_insert_aggregate_status_roots(&returning, &layout).expect("RETURNING status roots");
    assert_ne!(roots.aggregate_root, [0; 32]);
    assert_ne!(roots.statement_outcome_root, [0; 32]);
    assert_ne!(roots.response_root, [0; 32]);
    let reserved = reserve_typed_insert_aggregate_bodies(layout).unwrap();
    let encoded =
        encode_typed_insert_aggregate_bodies(&returning, &status(roots), reserved).unwrap();
    assert_eq!(encoded.section_roots()[5], roots.statement_outcome_root);
    let mut bodies = encoded_copies(&encoded);
    let status_index = bodies.len() - 1;
    for offset in [108_usize, 140, 172] {
        bodies[status_index][offset] ^= 1;
        assert!(
            decode_typed_insert_aggregate_bodies(returning.outer_flags, &body_refs(&bodies))
                .is_err()
        );
        bodies[status_index][offset] ^= 1;
    }
}

#[test]
fn one_through_four_chunk_encodings_are_canonical_and_exact() {
    for chunk_count in 1..=AGGREGATE_MAX_CHUNKS {
        let payloads = payloads_for_chunk_count(chunk_count);
        let encoded = encode_fixture(&payloads);
        assert_eq!(encoded.layout().chunk_count as usize, chunk_count);
        assert_eq!(encoded.fragment_count(), chunk_count + 1);
        for (index, (body, expected)) in encoded
            .fragment_bodies()
            .zip(encoded.layout().live_fragment_body_bytes())
            .enumerate()
        {
            assert_eq!(body.len() as u64, *expected, "body {index}");
            if index < chunk_count {
                let flags = u16::from_le_bytes(body[10..12].try_into().unwrap());
                assert_eq!(flags & AGGREGATE_CHUNK_FLAG_FIRST != 0, index == 0);
                assert_eq!(
                    flags & AGGREGATE_CHUNK_FLAG_LAST != 0,
                    index + 1 == chunk_count
                );
                assert_eq!(
                    u32::from_le_bytes(body[20..24].try_into().unwrap()) as usize,
                    index
                );
                assert_eq!(
                    u32::from_le_bytes(body[24..28].try_into().unwrap()) as usize,
                    chunk_count
                );
            }
        }
        let copies = encoded_copies(&encoded);
        let refs = body_refs(&copies);
        let decoded = decode_typed_insert_aggregate_bodies(
            OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
            &refs,
        )
        .expect("canonical chunk sequence decodes");
        let mut copied = vec![0; payloads[0].len()];
        decoded
            .copy_section_payload(0, &mut copied)
            .expect("section spanning chunks copies");
        assert_eq!(copied, payloads[0]);
    }
}

#[test]
fn every_chunk_header_byte_sabotage_is_rejected() {
    let payloads = payloads_for_chunk_count(2);
    let encoded = encode_fixture(&payloads);
    let mut bodies = encoded_copies(&encoded);
    for chunk in 0..2 {
        for byte in 0..AGGREGATE_CHUNK_HEADER_BYTES as usize {
            bodies[chunk][byte] ^= 0x80;
            let refs = body_refs(&bodies);
            assert!(
                decode_typed_insert_aggregate_bodies(
                    OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
                    &refs,
                )
                .is_err(),
                "accepted chunk {chunk} header byte {byte} sabotage"
            );
            bodies[chunk][byte] ^= 0x80;
        }
    }
}

#[test]
fn every_aggregate_stream_byte_sabotage_is_root_rejected() {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8 + 7; index + 1]);
    let encoded = encode_fixture(&payloads);
    assert_eq!(encoded.layout().chunk_count, 1);
    let mut bodies = encoded_copies(&encoded);
    let stream_start = AGGREGATE_CHUNK_HEADER_BYTES as usize;
    for byte in stream_start..bodies[0].len() {
        bodies[0][byte] ^= 1;
        let refs = body_refs(&bodies);
        assert!(
            decode_typed_insert_aggregate_bodies(
                OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
                &refs,
            )
            .is_err(),
            "accepted aggregate stream byte {} sabotage",
            byte - stream_start
        );
        bodies[0][byte] ^= 1;
    }
}

#[test]
fn exact_fragment_set_and_body_lengths_fail_closed() {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8 + 1; 3]);
    let encoded = encode_fixture(&payloads);
    let valid = encoded_copies(&encoded);

    let only_chunk = vec![valid[0].as_slice()];
    assert!(decode_typed_insert_aggregate_bodies(
        OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
        &only_chunk,
    )
    .is_err());

    let too_many_storage = [
        Vec::<u8>::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let too_many = body_refs(&too_many_storage);
    assert!(decode_typed_insert_aggregate_bodies(
        OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
        &too_many,
    )
    .is_err());

    for body_index in 0..valid.len() {
        let mut short = valid.clone();
        short[body_index].pop();
        assert!(decode_typed_insert_aggregate_bodies(
            OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
            &body_refs(&short),
        )
        .is_err());

        let mut surplus = valid.clone();
        surplus[body_index].push(0);
        assert!(decode_typed_insert_aggregate_bodies(
            OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW,
            &body_refs(&surplus),
        )
        .is_err());
    }
}

#[test]
fn outer_content_flag_sabotage_is_rejected() {
    let payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8; 1]);
    let encoded = encode_fixture(&payloads);
    let copies = encoded_copies(&encoded);
    let refs = body_refs(&copies);
    for flags in [
        OUTER_CONTENT_ROW,
        OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1,
        OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW | OUTER_CONTENT_CATALOG,
        OUTER_FLAG_TYPED_INSERT_AGGREGATE_V1 | OUTER_CONTENT_ROW | (1 << 29),
    ] {
        assert!(
            decode_typed_insert_aggregate_bodies(flags, &refs).is_err(),
            "accepted sabotaged outer flags {flags:#x}"
        );
    }
}

#[test]
fn status_v2_exact_roundtrip_and_structural_sabotage() {
    let valid = status(TypedInsertAggregateStatusRoots {
        aggregate_root: [9; 32],
        statement_outcome_root: [8; 32],
        response_root: [0; 32],
    });
    let mut bytes = [0_u8; AGGREGATE_STATUS_V2_BYTES as usize];
    encode_status_v2(&valid, &mut bytes).expect("encode status");
    assert_eq!(decode_status_v2(&bytes).expect("decode status"), valid);
    let status_len = bytes.len();
    assert!(encode_status_v2(&valid, &mut bytes[..status_len - 1]).is_err());
    let mut surplus = vec![0; bytes.len() + 1];
    assert!(encode_status_v2(&valid, &mut surplus).is_err());
    assert!(decode_status_v2(&bytes[..bytes.len() - 1]).is_err());
    surplus[..bytes.len()].copy_from_slice(&bytes);
    assert!(decode_status_v2(&surplus).is_err());

    for byte in [0_usize, 84, 85, 86, 87, 88, 89, 90, 91, 107] {
        let mut corrupt = bytes;
        corrupt[byte] ^= 0xff;
        assert!(
            decode_status_v2(&corrupt).is_err(),
            "accepted STATUS2 structural byte {byte} sabotage"
        );
    }
    for range in [12..28, 28..44, 44..52, 52..84, 100..104, 108..140, 172..204] {
        let mut corrupt = bytes;
        corrupt[range.clone()].fill(0);
        assert!(
            decode_status_v2(&corrupt).is_err(),
            "accepted zeroed STATUS2 field {range:?}"
        );
    }
}

#[test]
fn root_is_deterministic_and_binds_ordered_section_bytes() {
    let mut payloads: [Vec<u8>; AGGREGATE_SECTION_COUNT] =
        std::array::from_fn(|index| vec![index as u8 + 1; index + 1]);
    let first_view = view(&payloads);
    let first_layout = first_view.measure().unwrap();
    let first = typed_insert_aggregate_root(&first_view, &first_layout).unwrap();
    assert_eq!(
        typed_insert_aggregate_root(&first_view, &first_layout).unwrap(),
        first
    );

    payloads[5][0] ^= 1;
    let changed_view = view(&payloads);
    let changed_layout = changed_view.measure().unwrap();
    let changed = typed_insert_aggregate_root(&changed_view, &changed_layout).unwrap();
    assert_ne!(changed, first);
}

#[test]
fn decoder_source_has_no_aggregate_stream_allocation() {
    let source = include_str!("codec.rs");
    let decoder = source
        .split("pub(crate) fn decode_typed_insert_aggregate_bodies")
        .nth(1)
        .expect("decoder exists")
        .split("#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]")
        .next()
        .expect("decoder precedes decoded section type");
    for forbidden in ["Vec<", "Vec::", "Box<", "collect(", "to_vec("] {
        assert!(
            !decoder.contains(forbidden),
            "decoder assembled heap state through {forbidden}"
        );
    }
    assert!(decoder.contains("ChunkPayloadReader"));
    assert!(decoder.contains("[Option<&'a [u8]>; AGGREGATE_MAX_CHUNKS]"));
}
