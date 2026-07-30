use super::*;

struct Fixture {
    nullable: TypedInsertColumnValidity,
    all_valid: TypedInsertColumnValidity,
    int2: TypedInsertColumnValues,
    int4: TypedInsertColumnValues,
    int8: TypedInsertColumnValues,
    numeric: TypedInsertColumnValues,
    boolean: TypedInsertColumnValues,
    text: TypedInsertColumnValues,
    date: TypedInsertColumnValues,
    timestamp: TypedInsertColumnValues,
    uuid: TypedInsertColumnValues,
}

impl Fixture {
    fn new() -> Self {
        Self {
            nullable: TypedInsertColumnValidity::Bitmap(vec![0b101].into()),
            all_valid: TypedInsertColumnValidity::AllValid,
            int2: TypedInsertColumnValues::I32(vec![10, 0, 12].into()),
            int4: TypedInsertColumnValues::I32(vec![44, 45, 46].into()),
            int8: TypedInsertColumnValues::I64(vec![100, 101, 102].into()),
            numeric: TypedInsertColumnValues::I128(vec![12345, 0, -99].into()),
            boolean: TypedInsertColumnValues::BoolBits(vec![0b001].into()),
            text: TypedInsertColumnValues::Text {
                offsets: vec![0, 1, 1, 2].into(),
                bytes: b"az".to_vec().into(),
            },
            date: TypedInsertColumnValues::I32(vec![0, 1, 2].into()),
            timestamp: TypedInsertColumnValues::I64(vec![0, 1, 2].into()),
            uuid: TypedInsertColumnValues::Bytes16(vec![[1; 16], [2; 16], [3; 16]].into()),
        }
    }

    fn response_columns(&self) -> Vec<TypedImageColumnView<'_>> {
        let types = [
            SqlType::Int2,
            SqlType::Int4,
            SqlType::Int8,
            SqlType::Numeric {
                precision: 8,
                scale: 2,
            },
            SqlType::Bool,
            SqlType::Text,
            SqlType::Date,
            SqlType::Timestamp,
            SqlType::Uuid,
        ];
        let names = [
            "small", "regular", "large", "amount", "enabled", "label", "created", "changed",
            "identity",
        ];
        let catalog_ordinals = [8, 2, 5, 1, 9, 3, 7, 0, 4];
        let values = [
            &self.int2,
            &self.int4,
            &self.int8,
            &self.numeric,
            &self.boolean,
            &self.text,
            &self.date,
            &self.timestamp,
            &self.uuid,
        ];
        types
            .into_iter()
            .enumerate()
            .map(|(ordinal, ty)| TypedImageColumnView {
                catalog_column_ordinal: catalog_ordinals[ordinal],
                stable_column_id: 100 + ordinal as u32,
                table_ref: 4,
                attnum: ordinal as i16 + 1,
                ty,
                // A domain's declared OID intentionally differs from Numeric's storage tag.
                type_oid: if ordinal == 3 {
                    81_337
                } else {
                    ty.postgres_oid()
                },
                type_size: ty.type_size(),
                result_format: u16::from(ordinal % 2 == 0),
                name: names[ordinal],
                validity: if matches!(ordinal, 0 | 4 | 5) {
                    &self.nullable
                } else {
                    &self.all_valid
                },
                values: values[ordinal],
            })
            .collect()
    }

    fn final_columns(&self) -> [TypedImageColumnView<'_>; 2] {
        [
            TypedImageColumnView {
                catalog_column_ordinal: 0,
                stable_column_id: 100,
                table_ref: 4,
                attnum: 1,
                ty: SqlType::Int2,
                type_oid: SqlType::Int2.postgres_oid(),
                type_size: SqlType::Int2.type_size(),
                result_format: 0,
                name: "",
                validity: &self.nullable,
                values: &self.int2,
            },
            TypedImageColumnView {
                catalog_column_ordinal: 1,
                stable_column_id: 101,
                table_ref: 4,
                attnum: 2,
                ty: SqlType::Text,
                type_oid: SqlType::Text.postgres_oid(),
                type_size: SqlType::Text.type_size(),
                result_format: 0,
                name: "",
                validity: &self.nullable,
                values: &self.text,
            },
        ]
    }
}

fn response_bytes() -> Vec<u8> {
    let fixture = Fixture::new();
    let columns = fixture.response_columns();
    encode_typed_image(&TypedImageView {
        role: TypedImageRole::RetainedResponse,
        rows: 3,
        columns: &columns,
    })
    .expect("response fixture encodes")
}

fn vector_offset(bytes: &[u8], ordinal: usize) -> usize {
    let start = TYPED_IMAGE_HEADER_BYTES as usize + ordinal * TYPED_IMAGE_DESCRIPTOR_BYTES as usize;
    usize::try_from(u64::from_le_bytes(
        bytes[start + 48..start + 56]
            .try_into()
            .expect("vector offset field"),
    ))
    .expect("test vector offset fits usize")
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Deliberately exposes every byte as a separate source segment.  Thus every fixed image
/// header/descriptor/vector field crosses a `read_at` boundary instead of accidentally relying
/// on the slice adapter's contiguous reads.
struct BytewiseImageSource<'a> {
    bytes: &'a [u8],
}

impl TypedImageReadAt for BytewiseImageSource<'_> {
    fn len(&self) -> u64 {
        u64::try_from(self.bytes.len()).expect("test slice length fits u64")
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), EngineError> {
        let start = usize::try_from(offset).map_err(|_| image_error("test offset"))?;
        let end = start
            .checked_add(out.len())
            .ok_or_else(|| image_error("test range overflow"))?;
        let range = self
            .bytes
            .get(start..end)
            .ok_or_else(|| image_error("test range truncated"))?;
        for (destination, source) in out.iter_mut().zip(range) {
            *destination = *source;
        }
        Ok(())
    }
}

#[test]
fn typed_image_round_trips_all_physical_shapes_with_stable_layout_and_vector_digests() {
    let fixture = Fixture::new();
    let columns = fixture.response_columns();
    let view = TypedImageView {
        role: TypedImageRole::RetainedResponse,
        rows: 3,
        columns: &columns,
    };
    let measure = measure_typed_image(&view).expect("measure response");
    let bytes = encode_typed_image(&view).expect("encode response");
    assert_eq!(u64::try_from(bytes.len()).unwrap(), 1_377);
    assert_eq!(measure.encoded_bytes(), 1_377);
    assert_eq!(measure.descriptor_bytes(), 864);
    assert_eq!(measure.name_bytes(), 57);
    assert_eq!(measure.vector_bytes(), 344);
    assert_eq!(measure.decoded_owned_bytes(), 1_511);
    assert_eq!(measure.encode_maximum_scratch_bytes(), 864);
    assert_eq!(measure.decode_maximum_scratch_bytes(), 864);
    assert_eq!(measure.encoded_allocation_slots(), 1);
    assert_eq!(measure.encode_maximum_scratch_allocation_slots(), 1);
    assert_eq!(measure.decode_maximum_scratch_allocation_slots(), 1);
    assert_eq!(measure.decoded_persistent_allocation_slots(), 23);

    let decoded = decode_typed_image(&bytes).expect("decode response");
    assert_eq!(
        decoded.facts(),
        DecodedTypedImageFacts {
            role: TypedImageRole::RetainedResponse,
            rows: 3,
            columns: 9,
            layout_digest: bytes[64..96].try_into().expect("layout digest"),
        }
    );
    let facts: Vec<_> = decoded.columns().collect();
    assert_eq!(facts.len(), 9);
    assert_eq!(facts[0].catalog_column_ordinal, 8);
    assert_eq!(facts[3].type_oid, 81_337);
    assert_eq!(facts[5].name, "label");
    assert!(matches!(
        facts[4].values,
        TypedInsertColumnValues::BoolBits(_)
    ));
    assert!(matches!(
        facts[5].values,
        TypedInsertColumnValues::Text { .. }
    ));
    assert!(facts.iter().all(|fact| fact.vector_digest != [0; 32]));
    assert_eq!(
        decoded.facts().layout_digest,
        [
            0xe6, 0x78, 0x4b, 0x08, 0x4a, 0xa0, 0x77, 0x55, 0x06, 0x31, 0x88, 0x18, 0xba, 0x24,
            0xa5, 0x75, 0xd6, 0xda, 0xdd, 0x56, 0x27, 0xdb, 0xd7, 0xfe, 0xfb, 0x50, 0x70, 0x8f,
            0x97, 0x69, 0x43, 0xbc,
        ]
    );
    assert_eq!(
        facts[0].vector_digest,
        [
            0x5f, 0xfb, 0x2c, 0x0c, 0x4b, 0xc3, 0xe1, 0x00, 0x5e, 0xed, 0x2e, 0xf7, 0x07, 0xea,
            0xbc, 0xa7, 0x37, 0xe5, 0x2c, 0x04, 0xbc, 0x28, 0xd6, 0x08, 0x9f, 0xb3, 0xbf, 0x9e,
            0x40, 0x3a, 0x73, 0xb5,
        ]
    );

    // The decoder intentionally has no reencoder. Re-encoding the same sealed borrowed view is
    // nevertheless byte-for-byte stable, and the decoded owner accepts those exact bytes.
    assert_eq!(
        bytes,
        encode_typed_image(&view).expect("stable response bytes")
    );
}

#[test]
fn typed_image_enforces_final_and_response_descriptor_contracts() {
    let fixture = Fixture::new();
    let final_columns = fixture.final_columns();
    let final_view = TypedImageView {
        role: TypedImageRole::FinalTableImage,
        rows: 3,
        columns: &final_columns,
    };
    let final_bytes = encode_typed_image(&final_view).expect("final image encodes");
    assert_eq!(
        u32::from_le_bytes(final_bytes[20..24].try_into().unwrap()),
        FINAL_TABLE_IMAGE
    );
    assert_eq!(
        u64::from_le_bytes(final_bytes[48..56].try_into().unwrap()),
        0
    );
    assert!(decode_typed_image(&final_bytes).is_ok());

    let mut response_columns = fixture.response_columns();
    response_columns[0].catalog_column_ordinal = DERIVED_U32;
    response_columns[0].stable_column_id = 0;
    response_columns[0].table_ref = DERIVED_U32;
    response_columns[0].attnum = DERIVED_ATNUM;
    assert!(encode_typed_image(&TypedImageView {
        role: TypedImageRole::RetainedResponse,
        rows: 3,
        columns: &response_columns,
    })
    .is_ok());

    let mut malformed_final = fixture.final_columns();
    malformed_final[0].name = "forbidden";
    assert!(encode_typed_image(&TypedImageView {
        role: TypedImageRole::FinalTableImage,
        rows: 3,
        columns: &malformed_final,
    })
    .is_err());
}

#[test]
fn typed_image_zero_rows_are_canonical_for_every_vector_shape() {
    let types = [
        SqlType::Int2,
        SqlType::Int4,
        SqlType::Int8,
        SqlType::Numeric {
            precision: 8,
            scale: 2,
        },
        SqlType::Bool,
        SqlType::Text,
        SqlType::Date,
        SqlType::Timestamp,
        SqlType::Uuid,
    ];
    let values: Vec<_> = types
        .iter()
        .copied()
        .map(|ty| TypedInsertColumnValues::zeroed(ty, 0))
        .collect::<Result<_, _>>()
        .expect("zero vectors");
    let validity: Vec<_> = (0..types.len())
        .map(|_| TypedInsertColumnValidity::AllValid)
        .collect();
    let names: Vec<_> = (0..types.len())
        .map(|ordinal| format!("z{ordinal}"))
        .collect();
    let columns: Vec<_> = types
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, ty)| TypedImageColumnView {
            catalog_column_ordinal: ordinal as u32,
            stable_column_id: 200 + ordinal as u32,
            table_ref: 7,
            attnum: ordinal as i16 + 1,
            ty,
            type_oid: ty.postgres_oid(),
            type_size: ty.type_size(),
            result_format: 0,
            name: &names[ordinal],
            validity: &validity[ordinal],
            values: &values[ordinal],
        })
        .collect();
    let view = TypedImageView {
        role: TypedImageRole::RetainedResponse,
        rows: 0,
        columns: &columns,
    };
    let measure = measure_typed_image(&view).expect("zero rows measure");
    let bytes = encode_typed_image(&view).expect("zero rows encode");
    assert_eq!(u64::try_from(bytes.len()).unwrap(), measure.encoded_bytes());
    assert_eq!(measure.descriptor_bytes(), 9 * TYPED_IMAGE_DESCRIPTOR_BYTES);
    assert_eq!(measure.vector_bytes(), 110);
    assert!(measure.name_bytes() > 0);
    assert!(measure.decoded_owned_bytes() > measure.name_bytes());
    assert!(measure.encode_maximum_scratch_bytes() > 0);
    assert!(measure.decode_maximum_scratch_bytes() > 0);
    assert_eq!(measure.encoded_allocation_slots(), 1);
    assert_eq!(measure.encode_maximum_scratch_allocation_slots(), 1);
    assert_eq!(measure.decode_maximum_scratch_allocation_slots(), 1);
    assert_eq!(measure.decoded_persistent_allocation_slots(), 11);
    assert!(decode_typed_image(&bytes).is_ok());
    let bool_offset = vector_offset(&bytes, 4);
    assert_eq!(
        u32::from_le_bytes(
            bytes[bool_offset + 10..bool_offset + 14]
                .try_into()
                .unwrap()
        ),
        0
    );
    let text_offset = vector_offset(&bytes, 5);
    assert_eq!(
        u32::from_le_bytes(
            bytes[text_offset + 10..text_offset + 14]
                .try_into()
                .unwrap()
        ),
        1
    );
    assert_eq!(
        u32::from_le_bytes(
            bytes[text_offset + 22..text_offset + 26]
                .try_into()
                .unwrap()
        ),
        0
    );
}

#[test]
fn typed_image_rejects_header_directory_digest_and_length_sabotage_before_ownership() {
    let bytes = response_bytes();
    let descriptor = TYPED_IMAGE_HEADER_BYTES as usize;
    let vectors = vector_offset(&bytes, 0);
    let mut cases = Vec::new();
    for (offset, value) in [
        (0, 0x58),
        (16, 3),
        (18, 111),
        (20, 3),
        (96, 1),
        (descriptor, 4),
        (descriptor + 8, 0),
        (descriptor + 18, 1),
        (descriptor + 20, 0),
        (descriptor + 21, 255),
        (descriptor + 30, 2),
        (descriptor + 64, 0),
        (64, 0),
    ] {
        let mut mutated = bytes.clone();
        mutated[offset] = value;
        cases.push(mutated);
    }
    let mut huge_rows = bytes.clone();
    write_u32(&mut huge_rows, 24, u32::MAX);
    cases.push(huge_rows);
    // All scalar count fields agree, but the 412 GiB descriptor body is absent. This must fail
    // in the allocation-free header/length pass, before a directory reservation is attempted.
    let mut huge_columns = bytes.clone();
    write_u32(&mut huge_columns, 28, u32::MAX);
    write_u64(
        &mut huge_columns,
        32,
        3_u64.checked_mul(u64::from(u32::MAX)).unwrap(),
    );
    write_u64(&mut huge_columns, 40, u64::from(u32::MAX) * 96);
    write_u64(&mut huge_columns, 48, 0);
    write_u64(&mut huge_columns, 56, 0);
    let (huge_result, reservations) =
        observe_decode_reservations_for_test(|| decode_typed_image(&huge_columns));
    assert!(huge_result.is_err());
    assert_eq!(
        reservations, 0,
        "internally consistent hostile counts must fail before descriptor/value reservation"
    );
    cases.push(huge_columns);
    let mut name_overlap = bytes.clone();
    write_u64(
        &mut name_overlap,
        descriptor + TYPED_IMAGE_DESCRIPTOR_BYTES as usize + 32,
        1,
    );
    cases.push(name_overlap);
    let mut vector_overlap = bytes.clone();
    write_u64(
        &mut vector_overlap,
        descriptor + TYPED_IMAGE_DESCRIPTOR_BYTES as usize + 48,
        vectors as u64,
    );
    cases.push(vector_overlap);
    let mut vector_length = bytes.clone();
    write_u64(&mut vector_length, descriptor + 56, 1);
    cases.push(vector_length);
    let mut zero_type_oid = bytes.clone();
    write_u32(&mut zero_type_oid, descriptor + 24, 0);
    cases.push(zero_type_oid);
    let mut wrong_type_size = bytes.clone();
    wrong_type_size[descriptor + 28..descriptor + 30].copy_from_slice(&999_i16.to_le_bytes());
    cases.push(wrong_type_size);
    let mut nul_name = bytes.clone();
    let name_offset = usize::try_from(u64::from_le_bytes(
        nul_name[descriptor + 32..descriptor + 40]
            .try_into()
            .unwrap(),
    ))
    .unwrap();
    nul_name[name_offset] = 0;
    cases.push(nul_name);
    let mut trailing = bytes.clone();
    trailing.push(0);
    cases.push(trailing);
    assert!(cases.iter().all(|case| decode_typed_image(case).is_err()));
}

#[test]
fn typed_image_rejects_vector_canonicality_and_placeholder_sabotage() {
    let bytes = response_bytes();
    let int2 = vector_offset(&bytes, 0);
    let numeric = vector_offset(&bytes, 3);
    let boolean = vector_offset(&bytes, 4);
    let text = vector_offset(&bytes, 5);
    let date = vector_offset(&bytes, 6);
    let timestamp = vector_offset(&bytes, 7);
    let mut cases = Vec::new();

    let mut bitmap_words = bytes.clone();
    write_u32(&mut bitmap_words, int2 + 1, 0);
    cases.push(bitmap_words);
    let mut all_valid_bitmap = bytes.clone();
    write_u32(&mut all_valid_bitmap, int2 + 5, 0b111);
    cases.push(all_valid_bitmap);
    let mut bitmap_tail = bytes.clone();
    write_u32(&mut bitmap_tail, int2 + 5, 0b1000);
    cases.push(bitmap_tail);
    let mut invalid_placeholder = bytes.clone();
    write_u32(&mut invalid_placeholder, int2 + 22, 1);
    cases.push(invalid_placeholder);
    let mut int2_bounds = bytes.clone();
    write_u32(&mut int2_bounds, int2 + 18, i32::MAX as u32);
    cases.push(int2_bounds);
    let mut numeric_bounds = bytes.clone();
    numeric_bounds[numeric + 10..numeric + 26].copy_from_slice(&i128::MAX.to_le_bytes());
    cases.push(numeric_bounds);
    let mut date_bounds = bytes.clone();
    write_u32(&mut date_bounds, date + 10, i32::MAX as u32);
    cases.push(date_bounds);
    let mut timestamp_bounds = bytes.clone();
    timestamp_bounds[timestamp + 10..timestamp + 18].copy_from_slice(&i64::MAX.to_le_bytes());
    cases.push(timestamp_bounds);
    let mut bool_tail = bytes.clone();
    write_u32(&mut bool_tail, boolean + 22, 0b1000);
    cases.push(bool_tail);
    let mut text_order = bytes.clone();
    write_u64(&mut text_order, text + 30, 3);
    cases.push(text_order);
    let mut text_utf8 = bytes.clone();
    text_utf8[text + 58] = 0xff;
    cases.push(text_utf8);

    assert!(cases.iter().all(|case| decode_typed_image(case).is_err()));
}

#[test]
fn typed_image_rejects_zero_row_noncanonical_bool_and_text_payloads() {
    let types = [SqlType::Bool, SqlType::Text];
    let values: Vec<_> = types
        .iter()
        .copied()
        .map(|ty| TypedInsertColumnValues::zeroed(ty, 0))
        .collect::<Result<_, _>>()
        .expect("zero vectors");
    let validity = [
        TypedInsertColumnValidity::AllValid,
        TypedInsertColumnValidity::AllValid,
    ];
    let columns = [
        TypedImageColumnView {
            catalog_column_ordinal: 0,
            stable_column_id: 1,
            table_ref: 1,
            attnum: 1,
            ty: types[0],
            type_oid: types[0].postgres_oid(),
            type_size: types[0].type_size(),
            result_format: 0,
            name: "b",
            validity: &validity[0],
            values: &values[0],
        },
        TypedImageColumnView {
            catalog_column_ordinal: 1,
            stable_column_id: 2,
            table_ref: 1,
            attnum: 2,
            ty: types[1],
            type_oid: types[1].postgres_oid(),
            type_size: types[1].type_size(),
            result_format: 0,
            name: "t",
            validity: &validity[1],
            values: &values[1],
        },
    ];
    let bytes = encode_typed_image(&TypedImageView {
        role: TypedImageRole::RetainedResponse,
        rows: 0,
        columns: &columns,
    })
    .expect("zero fixture");
    let mut bool_payload = bytes.clone();
    write_u32(&mut bool_payload, vector_offset(&bytes, 0) + 10, 1);
    let mut text_payload = bytes.clone();
    write_u32(&mut text_payload, vector_offset(&bytes, 1) + 22, 1);
    assert!(decode_typed_image(&bool_payload).is_err());
    assert!(decode_typed_image(&text_payload).is_err());
}

#[test]
fn typed_image_raw_measure_reserves_every_owner_and_drains_on_failure() {
    let bytes = response_bytes();
    let measure = measure_decoded_typed_image(&bytes).expect("raw image pass succeeds");
    assert_eq!(measure.persistent_bytes(), 1_511);
    assert_eq!(measure.persistent_allocation_slots(), 23);
    assert_eq!(
        measure.maximum_scratch_bytes(),
        9 * u64::try_from(std::mem::size_of::<ImageDescriptor>()).unwrap()
    );
    assert_eq!(measure.maximum_scratch_allocation_slots(), 1);

    let (_, stats) = observe_decode_reservation_stats_for_test(|| {
        decode_typed_image_after_measure(&bytes, measure).expect("measured decode succeeds")
    });
    assert_eq!(stats.persistent_bytes, measure.persistent_bytes());
    assert_eq!(
        stats.persistent_slots,
        measure.persistent_allocation_slots()
    );
    assert_eq!(stats.scratch_bytes, measure.maximum_scratch_bytes());
    assert_eq!(
        stats.scratch_slots,
        measure.maximum_scratch_allocation_slots()
    );
    let owners = stats.attempts;

    for owner in 1..=owners {
        let failed = fail_decode_reservation_for_test(owner, None, None, || {
            decode_typed_image_after_measure(&bytes, measure)
        });
        assert!(failed.is_err(), "owner {owner} must fail closed");
        assert!(
            decode_typed_image_after_measure(&bytes, measure).is_ok(),
            "owner {owner} failure must drain before immediate retry"
        );
    }

    let below_persistent = measure.persistent_bytes().checked_sub(1).unwrap();
    let persistent_refusal =
        fail_decode_reservation_for_test(owners + 1, Some(below_persistent), None, || {
            decode_typed_image_after_measure(&bytes, measure)
        });
    assert!(persistent_refusal.is_err());
    assert!(decode_typed_image_after_measure(&bytes, measure).is_ok());

    let below_scratch = measure.maximum_scratch_bytes().checked_sub(1).unwrap();
    let scratch_refusal =
        fail_decode_reservation_for_test(owners + 1, None, Some(below_scratch), || {
            decode_typed_image_after_measure(&bytes, measure)
        });
    assert!(scratch_refusal.is_err());
    assert!(decode_typed_image_after_measure(&bytes, measure).is_ok());
}

#[test]
fn typed_image_read_at_raw_pass_crosses_every_fixed_boundary_without_reservation() {
    let bytes = response_bytes();
    let source = BytewiseImageSource { bytes: &bytes };
    let (measure, owners) =
        observe_decode_reservations_for_test(|| measure_decoded_typed_image_from_source(&source));
    let measure = measure.expect("bytewise source raw pass succeeds");
    assert_eq!(owners, 0, "raw source pass may not reserve decoded owners");
    assert_eq!(measure.image_bytes(), u64::try_from(bytes.len()).unwrap());
    assert_eq!(measure, measure_decoded_typed_image(&bytes).unwrap());

    let mut copied = Vec::new();
    copied
        .try_reserve_exact(bytes.len())
        .expect("test exact image copy reservation");
    copied.resize(bytes.len(), 0);
    copy_typed_image_after_measure(&source, measure, &mut copied).expect("copy after raw pass");
    assert_eq!(copied, bytes);

    let name_start = TYPED_IMAGE_HEADER_BYTES as usize + 9 * TYPED_IMAGE_DESCRIPTOR_BYTES as usize;
    let mut invalid_utf8 = bytes.clone();
    invalid_utf8[name_start] = 0xff;
    let (invalid_utf8, owners) = observe_decode_reservations_for_test(|| {
        measure_decoded_typed_image_from_source(&BytewiseImageSource {
            bytes: &invalid_utf8,
        })
    });
    assert!(invalid_utf8.is_err());
    assert_eq!(owners, 0, "UTF-8 sabotage must fail before any reservation");

    let mut nonzero_placeholder = bytes.clone();
    write_u32(&mut nonzero_placeholder, vector_offset(&bytes, 0) + 22, 1);
    let (nonzero_placeholder, owners) = observe_decode_reservations_for_test(|| {
        measure_decoded_typed_image_from_source(&BytewiseImageSource {
            bytes: &nonzero_placeholder,
        })
    });
    assert!(nonzero_placeholder.is_err());
    assert_eq!(
        owners, 0,
        "placeholder sabotage must fail before any reservation"
    );

    let mut forged_digest = bytes.clone();
    let first_descriptor_digest = TYPED_IMAGE_HEADER_BYTES as usize + 64;
    forged_digest[first_descriptor_digest] ^= 1;
    let (forged_digest, owners) = observe_decode_reservations_for_test(|| {
        measure_decoded_typed_image_from_source(&BytewiseImageSource {
            bytes: &forged_digest,
        })
    });
    assert!(forged_digest.is_err());
    assert_eq!(
        owners, 0,
        "digest sabotage must fail before any reservation"
    );
}

#[test]
fn typed_image_source_measure_rejects_valid_same_length_content_drift() {
    let original = response_bytes();
    let measure =
        measure_decoded_typed_image_from_source(&BytewiseImageSource { bytes: &original })
            .expect("original image source measures");

    let mut changed_fixture = Fixture::new();
    changed_fixture.int4 = TypedInsertColumnValues::I32(vec![44, 45, 47].into());
    let changed_columns = changed_fixture.response_columns();
    let changed = encode_typed_image(&TypedImageView {
        role: TypedImageRole::RetainedResponse,
        rows: 3,
        columns: &changed_columns,
    })
    .expect("changed image remains independently canonical");
    assert_eq!(
        changed.len(),
        original.len(),
        "content-drift fixture preserves raw geometry"
    );
    assert!(
        measure_decoded_typed_image_from_source(&BytewiseImageSource { bytes: &changed }).is_ok(),
        "changed source must remain a valid image rather than malformed sabotage"
    );

    let mut destination = vec![0; original.len()];
    assert!(
        copy_typed_image_after_measure(
            &BytewiseImageSource { bytes: &changed },
            measure,
            &mut destination,
        )
        .is_err(),
        "a raw measure must bind the exact source content, not only its geometry"
    );
}

#[test]
fn typed_image_source_guards_keep_one_inert_shared_vector_authority() {
    let source = include_str!("typed_image_codec.rs");
    let source_reader = include_str!("typed_image_codec/read_at.rs");
    let value_contract = include_str!("typed_image_codec_value_contract.rs");
    let encoder = include_str!("canonical_codec.rs");
    let decoder = include_str!("canonical_codec_decode.rs");
    assert!(source.contains("GPUDBTYPEDIMAGE2"));
    assert!(source.contains("gpu-db/write001/image-layout/v2"));
    assert!(source.contains("gpu-db/write001/typed-vector/v2"));
    assert!(source.contains("try_reserve_exact"));
    assert!(!source.contains("Vec::with_capacity"));
    assert!(!source.contains("vec!["));
    assert!(!source.contains("WalRecord"));
    assert!(!source.contains("WriteDelta"));
    assert!(!source.contains("fn reencode"));
    assert!(!source.contains("fn into_inner"));
    assert!(source.contains("typed_image_codec_value_contract"));
    assert!(source.contains("decode_reservation"));
    assert!(source.contains("measure_decoded_typed_image_from_source"));
    assert!(source_reader.contains("trait TypedImageReadAt"));
    assert!(!source_reader.contains("Vec<"));
    assert!(!value_contract.contains("Vec::with_capacity"));
    assert!(!value_contract.contains("vec!["));
    assert!(!value_contract.contains("append_typed_values"));
    assert!(!value_contract.contains("decode_typed_values"));
    assert!(encoder.contains("typed_image_codec::append_typed_values"));
    assert!(encoder.contains("typed_image_codec::append_typed_validity"));
    assert!(decoder.contains("typed_image_codec::decode_typed_values"));
    assert!(decoder.contains("typed_image_codec::decode_typed_validity"));
}
