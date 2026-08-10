//! Fixed-layout S7 grammar helpers for the live typed INSERT writer.
//!
//! This leaf owns only deterministic validation, bytes, and digests. It has no
//! caller-visible entry point and cannot reserve WAL, apply a device plan, or publish a root.

use super::*;

pub(super) fn validate_input(input: &LiveTypedInsertView<'_>) -> Result<(), EngineError> {
    let original_rows = input.final_writers.len();
    let surviving_rows = input
        .final_writers
        .iter()
        .filter(|writer| writer.survives)
        .count();
    let image_rows = read_u32(input.final_image, 24)? as usize;
    let image_columns = read_u32(input.final_image, 28)? as usize;
    let source_shape_is_exact = match (input.source_geometry, input.final_row_digests) {
        (Some(source_geometry), Some(_)) if surviving_rows != 0 => {
            source_geometry.rows == surviving_rows
                && source_geometry.cells == surviving_rows.saturating_mul(image_columns)
        }
        (None, None) if surviving_rows == 0 => true,
        _ => false,
    };
    let allocator_rows_are_exact = {
        let mut ids = input
            .final_writers
            .iter()
            .map(|writer| writer.stable_row_id)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.iter()
            .copied()
            .eq(input.table.row_allocator_before..input.table.row_allocator_high_water)
            && input
                .final_writers
                .iter()
                .filter(|writer| writer.survives)
                .map(|writer| writer.stable_row_id)
                .eq(input.table.row_allocator_before
                    ..input.table.row_allocator_before + surviving_rows as u64)
    };
    if original_rows == 0
        || image_columns == 0
        || input.table.schema.is_empty()
        || input.table.name.is_empty()
        || input.table.stable_table_id == 0
        || input.table.stable_table_id == u64::MAX
        || input.table.display_oid == 0
        || input.table.display_oid > 0x7fff_ffff
        || !source_shape_is_exact
        || !allocator_rows_are_exact
        || input.table.row_allocator_before == 0
        || input.table.row_allocator_high_water
            != input
                .table
                .row_allocator_before
                .checked_add(original_rows as u64)
                .ok_or_else(|| error("row allocator range overflows"))?
        || input.table.final_logical_row_count
            != if input.table.resets_existing_rows {
                surviving_rows as u64
            } else {
                input
                    .table
                    .initial_logical_row_count
                    .checked_add(surviving_rows as u64)
                    .ok_or_else(|| error("logical row count overflows"))?
            }
        || (!input.table.initial_table_absent && input.table.data_generation_before == 0)
        || (input.table.initial_table_absent
            && (input.table.resets_existing_rows
                || input.table.data_generation_before != 0
                || input.table.initial_logical_row_count != 0))
        || (surviving_rows != 0
            && input.table.data_generation_after <= input.table.data_generation_before)
        || (surviving_rows == 0
            && input.table.data_generation_after != input.table.data_generation_before)
        || input.table.schema_digest == [0; 32]
        || input.identity.catalog_digest == [0; 32]
        || (!input.table.initial_table_absent && input.table.initial_table_root == [0; 32])
        || (input.table.initial_table_absent && input.table.initial_table_root != [0; 32])
        || input.table.final_table_root == [0; 32]
        || (surviving_rows != 0 && input.table.initial_table_root == input.table.final_table_root)
        || (surviving_rows == 0 && input.table.initial_table_root != input.table.final_table_root)
        || input.table.initial_database_root == [0; 32]
        || input.table.final_database_root == [0; 32]
        || (surviving_rows != 0
            && input.table.initial_database_root == input.table.final_database_root)
        || (surviving_rows == 0
            && input.table.initial_database_root != input.table.final_database_root)
        || input.statements.is_empty()
        || input
            .statements
            .windows(2)
            .any(|pair| pair[0].statement_ordinal >= pair[1].statement_ordinal)
        || input.statements.iter().any(|statement| {
            statement.typed_statement_digest == [0; 32] || statement.record.is_empty()
        })
        || (!input.table.initial_table_absent && input.identity.catalog_epoch == 0)
        || (!input.table.initial_table_absent
            && (input.identity.dependency_validation_floor == 0
                || input.identity.dependency_validation_floor >= input.identity.commit_sequence))
        || (input.table.initial_table_absent
            && input.identity.dependency_validation_floor > input.identity.commit_sequence)
        || image_rows != surviving_rows
        || input.final_writers.iter().any(|writer| {
            writer.final_writer_statement_digest == [0; 32]
                && writer.final_writer_statement_ordinal != writer.source_statement_ordinal
                || writer.final_writer_statement_digest != [0; 32]
                    && writer.final_writer_statement_ordinal <= writer.source_statement_ordinal
                || !writer.survives && writer.final_writer_statement_digest == [0; 32]
        })
    {
        return Err(error(
            "live typed INSERT input geometry or authenticated roots are invalid",
        ));
    }
    Ok(())
}

pub(super) fn target_dependency(input: &LiveTypedInsertView<'_>, reference: u32) -> [u8; 224] {
    let name_digest = v2_digest(
        b"gpu-db/write001/s7-qualified-name/v2",
        &[
            &(input.table.schema.len() as u32).to_le_bytes(),
            input.table.schema.as_bytes(),
            &(input.table.name.len() as u32).to_le_bytes(),
            input.table.name.as_bytes(),
        ],
    );
    let identity_digest = v2_digest(
        b"gpu-db/write001/s7-table-object/v2",
        &[
            &[1],
            &input.table.stable_table_id.to_le_bytes(),
            &input.table.display_oid.to_le_bytes(),
            &input.identity.catalog_epoch.to_le_bytes(),
            &input.table.data_generation_before.to_le_bytes(),
            &input.table.schema_digest,
            &input.table.initial_table_root,
            &name_digest,
        ],
    );
    let mut raw = [0_u8; 224];
    put_u32(&mut raw, 0, reference);
    raw[4] = 1;
    raw[5] = 3;
    put_u64(&mut raw, 8, input.table.stable_table_id);
    put_u32(&mut raw, 16, input.table.display_oid);
    put_u32(&mut raw, 20, reference);
    put_u64(&mut raw, 24, input.table.data_generation_before);
    put_u64(&mut raw, 32, input.identity.dependency_validation_floor);
    put_u32(&mut raw, 40, ABSENT_U32);
    put_u32(&mut raw, 44, ABSENT_U32);
    put_u64(&mut raw, 48, input.identity.catalog_epoch);
    put_digest(&mut raw, 64, input.table.schema_digest);
    put_digest(&mut raw, 96, input.table.initial_table_root);
    put_digest(&mut raw, 128, name_digest);
    put_digest(&mut raw, 160, identity_digest);
    let token_digest = v2_digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&raw[..192], &[0; 32], &[0; 32], &[0; 32]],
    );
    put_digest(&mut raw, 192, token_digest);
    raw
}

pub(super) fn target_dependency_use(statement_ordinal: u32, dependency_ref: u32) -> [u8; 32] {
    let mut raw = [0_u8; 32];
    put_u32(&mut raw, 0, statement_ordinal);
    put_u32(&mut raw, 4, dependency_ref);
    put_u16(&mut raw, 8, 1);
    put_u32(&mut raw, 16, ABSENT_U32);
    put_u32(&mut raw, 20, ABSENT_U32);
    raw
}

#[allow(clippy::too_many_arguments)] // positional arguments mirror the fixed S4 binary layout
pub(super) fn append_s4(
    out: &mut Vec<u8>,
    statement: u32,
    source_row: u32,
    table_ref: u32,
    transition_ref: u32,
    stable_row_id: u64,
    typed_digest: [u8; 32],
    final_writer: &LiveTypedInsertFinalWriter,
) {
    let mut raw = [0_u8; S4_BYTES];
    put_u32(&mut raw, 0, statement);
    put_u32(&mut raw, 4, source_row);
    put_u64(&mut raw, 8, stable_row_id);
    raw[16] = if final_writer.survives { 1 } else { 2 };
    put_u32(&mut raw, 20, table_ref);
    put_u32(&mut raw, 24, transition_ref);
    if final_writer.survives {
        put_digest(&mut raw, 32, typed_digest);
    } else {
        raw[17] = 1;
        put_u32(&mut raw, 28, final_writer.final_writer_statement_ordinal);
        put_digest(&mut raw, 32, final_writer.final_writer_statement_digest);
    }
    out.extend_from_slice(&raw);
}

#[allow(clippy::too_many_arguments)] // positional arguments mirror the fixed S5 binary layout
pub(super) fn append_table_disposition(
    out: &mut Vec<u8>,
    table_ref: u32,
    s4_ref: u32,
    statement: u32,
    source_row: u32,
    stable_row_id: u64,
    disposition: u8,
) {
    let mut raw = [0_u8; 32];
    put_u32(&mut raw, 0, table_ref);
    put_u32(&mut raw, 4, s4_ref);
    put_u64(&mut raw, 8, stable_row_id);
    put_u32(&mut raw, 16, statement);
    put_u32(&mut raw, 20, source_row);
    raw[24] = disposition;
    out.extend_from_slice(&raw);
}

#[allow(clippy::too_many_arguments)] // positional arguments mirror the fixed S7 binary layout
pub(super) fn transition_bytes(
    reference: u32,
    table_ref: u32,
    stable_row_id: u64,
    source_s4: u32,
    source_statement: u32,
    source_row: u32,
    image_ref: u32,
    image_row: u32,
    effect_start: u32,
    effect_count: u32,
    final_writer_statement: u32,
    typed_digest: [u8; 32],
    final_row_digest: [u8; 32],
    transition_digest: [u8; 32],
    final_writer_statement_digest: [u8; 32],
) -> [u8; 192] {
    let mut raw = [0_u8; 192];
    put_u32(&mut raw, 0, reference);
    put_u32(&mut raw, 4, table_ref);
    put_u64(&mut raw, 8, stable_row_id);
    raw[16] = 1;
    put_u32(&mut raw, 20, source_s4);
    put_u32(&mut raw, 24, source_statement);
    put_u32(&mut raw, 28, source_row);
    put_u32(&mut raw, 32, image_ref);
    put_u32(&mut raw, 36, image_row);
    put_u32(&mut raw, 40, effect_start);
    put_u32(&mut raw, 44, effect_count);
    put_u32(&mut raw, 48, final_writer_statement);
    raw[17] = u8::from(final_writer_statement_digest != [0; 32]);
    put_digest(&mut raw, 64, typed_digest);
    put_digest(&mut raw, 96, final_row_digest);
    put_digest(&mut raw, 128, transition_digest);
    put_digest(&mut raw, 160, final_writer_statement_digest);
    raw
}

#[allow(clippy::too_many_arguments)] // positional arguments mirror the fixed S7 binary layout
pub(super) fn transition_digest(
    reference: u32,
    table_ref: u32,
    stable_row_id: u64,
    source_s4: u32,
    source_statement: u32,
    source_row: u32,
    image_ref: u32,
    image_row: u32,
    effect_start: u32,
    effect_count: u32,
    final_writer_statement: u32,
    typed_digest: [u8; 32],
    final_row_digest: [u8; 32],
    final_writer_statement_digest: [u8; 32],
) -> [u8; 32] {
    let transition = transition_bytes(
        reference,
        table_ref,
        stable_row_id,
        source_s4,
        source_statement,
        source_row,
        image_ref,
        image_row,
        effect_start,
        effect_count,
        final_writer_statement,
        typed_digest,
        final_row_digest,
        [0; 32],
        final_writer_statement_digest,
    );
    v2_digest(
        b"gpu-db/write001/s7-transition/v2",
        &[&transition[..128], &[0; 32], &transition[160..]],
    )
}

pub(super) fn image_descriptor(
    input: &LiveTypedInsertView<'_>,
    image_ref: u32,
    image_offset: u64,
    image_content_digest: gpu_db_wal::CanonicalDigest,
) -> Result<[u8; 160], EngineError> {
    let mut raw = [0_u8; 160];
    put_u32(&mut raw, 0, image_ref);
    put_u32(&mut raw, 4, image_ref);
    put_u32(&mut raw, 8, 1);
    put_u32(&mut raw, 16, read_u32(input.final_image, 24)?);
    put_u32(&mut raw, 20, read_u32(input.final_image, 28)?);
    put_u64(&mut raw, 24, image_offset);
    put_u64(&mut raw, 32, input.final_image.len() as u64);
    let layout: [u8; 32] = input
        .final_image
        .get(64..96)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| error("final image lacks its layout digest"))?;
    put_digest(&mut raw, 64, layout);
    put_digest(&mut raw, 96, image_content_digest);
    let descriptor = v2_digest(
        b"gpu-db/write001/s7-image-descriptor/v2",
        &[&raw[..128], &[0; 32]],
    );
    put_digest(&mut raw, 128, descriptor);
    Ok(raw)
}

#[allow(clippy::too_many_arguments)] // positional arguments mirror the fixed S7 binary layout
pub(super) fn table_block(
    input: &LiveTypedInsertView<'_>,
    table_ref: u32,
    target_dependency_ref: u32,
    disposition_count: u32,
    transition_count: u32,
    disposition_start: u32,
    transition_start: u32,
    owned_index_start: u32,
    owned_index_count: u32,
    key_effect_start: u32,
    key_effect_count: u32,
    transition_root: [u8; 32],
    index_effect_root: [u8; 32],
    image_content_digest: gpu_db_wal::CanonicalDigest,
) -> [u8; 384] {
    let mut raw = [0_u8; 384];
    put_u32(&mut raw, 0, table_ref);
    put_u32(
        &mut raw,
        4,
        (u32::from(input.table.resets_existing_rows) * S7_TABLE_FLAG_RESETS_EXISTING_ROWS)
            | (u32::from(input.table.initial_table_absent) * S7_TABLE_FLAG_INITIAL_TABLE_ABSENT),
    );
    put_u64(&mut raw, 8, input.table.stable_table_id);
    put_u32(&mut raw, 16, input.table.display_oid);
    put_u32(&mut raw, 20, target_dependency_ref);
    put_u64(&mut raw, 24, input.identity.catalog_epoch);
    put_u64(&mut raw, 32, input.table.data_generation_before);
    put_u64(&mut raw, 40, input.table.data_generation_after);
    put_u64(&mut raw, 48, input.table.row_allocator_before);
    put_u64(&mut raw, 56, input.table.row_allocator_high_water);
    put_u64(&mut raw, 64, input.table.initial_logical_row_count);
    put_u64(&mut raw, 72, input.table.final_logical_row_count);
    put_u32(&mut raw, 80, disposition_start);
    put_u32(&mut raw, 84, disposition_count);
    put_u32(&mut raw, 88, transition_start);
    put_u32(&mut raw, 92, transition_count);
    put_u32(&mut raw, 96, owned_index_start);
    put_u32(&mut raw, 100, owned_index_count);
    put_u32(&mut raw, 104, key_effect_start);
    put_u32(&mut raw, 108, key_effect_count);
    put_u32(&mut raw, 112, table_ref);
    put_u32(
        &mut raw,
        116,
        read_u32(input.final_image, 28).expect("validated image column count"),
    );
    put_digest(&mut raw, 128, input.table.schema_digest);
    put_digest(&mut raw, 160, input.table.initial_table_root);
    put_digest(&mut raw, 192, input.table.final_table_root);
    put_digest(
        &mut raw,
        224,
        input.final_image[64..96]
            .try_into()
            .expect("validated final image layout digest"),
    );
    put_digest(&mut raw, 256, image_content_digest);
    put_digest(&mut raw, 288, transition_root);
    put_digest(&mut raw, 320, index_effect_root);
    raw
}

pub(super) fn root_descriptor(
    identity: LiveTypedInsertIdentity,
    initial_database_root: [u8; 32],
    final_database_root: [u8; 32],
    initial_overlay: [u8; 32],
    final_overlay: [u8; 32],
    tables: &[[u8; 384]],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-root-descriptor/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(1_u16.to_le_bytes());
    digest.update(identity.catalog_epoch.to_le_bytes());
    digest.update(identity.catalog_after_epoch.to_le_bytes());
    digest.update(identity.catalog_digest);
    digest.update(identity.catalog_after_digest);
    digest.update(initial_database_root);
    digest.update(final_database_root);
    digest.update(initial_overlay);
    digest.update(final_overlay);
    digest.update(
        u32::try_from(tables.len())
            .expect("validated table count fits u32")
            .to_le_bytes(),
    );
    for table in tables {
        digest.update(&table[0..4]);
        digest.update(&table[4..8]);
        digest.update(&table[8..16]);
        digest.update(&table[32..48]);
        digest.update(&table[160..224]);
        digest.update(&table[352..384]);
    }
    digest.finalize().into()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_s7(
    identity: LiveTypedInsertIdentity,
    initial_database_root: [u8; 32],
    final_database_root: [u8; 32],
    counts: [u32; 12],
    root_descriptor: [u8; 32],
    initial_overlay: [u8; 32],
    final_overlay: [u8; 32],
    tables: &[[u8; 384]],
    table_dispositions: &[u8],
    resolution: &[u8],
    dependencies: &[[u8; 224]],
    dependency_uses: &[[u8; 32]],
    indexes: &[[u8; 384]],
    index_keys: &[[u8; 112]],
    transitions: &[[u8; 192]],
    effects: &[[u8; 192]],
    components: &[[u8; 128]],
    projections: &[[u8; 128]],
    values: &[u8],
    image_descriptors: &[[u8; 160]],
    final_images: &[&[u8]],
) -> Result<Vec<u8>, EngineError> {
    let mut directory_bytes = [0_u64; 14];
    for index in 0..12 {
        directory_bytes[index] = (counts[S7_COUNT_TO_DIRECTORY[index]] as u64)
            .checked_mul(S7_FIXED_WIDTHS[index] as u64)
            .ok_or_else(|| error("S7 fixed directory size overflows"))?;
    }
    directory_bytes[12] =
        u64::try_from(values.len()).map_err(|_| error("S7 value arena exceeds u64"))?;
    directory_bytes[13] = final_images.iter().try_fold(0_u64, |total, image| {
        total
            .checked_add(u64::try_from(image.len()).map_err(|_| error("S7 image exceeds u64"))?)
            .ok_or_else(|| error("S7 image arena exceeds u64"))
    })?;
    let mut offsets = [0_u64; 14];
    let mut total = S7_HEADER_BYTES as u64;
    for index in 0..14 {
        offsets[index] = total;
        total = total
            .checked_add(directory_bytes[index])
            .ok_or_else(|| error("S7 total bytes overflow"))?;
    }
    let capacity = usize::try_from(total).map_err(|_| error("S7 exceeds host addressability"))?;
    let mut out = vec![0_u8; S7_HEADER_BYTES];
    out[..16].copy_from_slice(S7_MAGIC);
    put_u16(&mut out, 16, 1);
    put_u16(&mut out, 18, 2);
    put_u32(&mut out, 20, S7_HEADER_BYTES as u32);
    put_u16(&mut out, 28, 14);
    put_u16(&mut out, 30, 1);
    put_u64(&mut out, 32, total);
    for (index, count) in counts.iter().copied().enumerate() {
        put_u32(&mut out, 40 + index * 4, count);
    }
    put_u64(&mut out, 88, directory_bytes[12]);
    put_u64(&mut out, 96, directory_bytes[13]);
    for index in 0..14 {
        put_u64(&mut out, 104 + index * 16, offsets[index]);
        put_u64(&mut out, 112 + index * 16, directory_bytes[index]);
    }
    put_u64(&mut out, 328, identity.catalog_epoch);
    put_u64(&mut out, 336, identity.catalog_after_epoch);
    put_digest(&mut out, 344, identity.catalog_digest);
    put_digest(&mut out, 376, identity.catalog_after_digest);
    put_digest(&mut out, 408, initial_database_root);
    put_digest(&mut out, 440, final_database_root);
    put_digest(&mut out, 472, initial_overlay);
    put_digest(&mut out, 504, final_overlay);
    put_digest(&mut out, 536, root_descriptor);
    out.reserve_exact(capacity - S7_HEADER_BYTES);
    for table in tables {
        out.extend_from_slice(table);
    }
    out.extend_from_slice(table_dispositions);
    out.extend_from_slice(resolution);
    for dependency in dependencies {
        out.extend_from_slice(dependency);
    }
    for dependency_use in dependency_uses {
        out.extend_from_slice(dependency_use);
    }
    for index in indexes {
        out.extend_from_slice(index);
    }
    for key in index_keys {
        out.extend_from_slice(key);
    }
    for transition in transitions {
        out.extend_from_slice(transition);
    }
    for effect in effects {
        out.extend_from_slice(effect);
    }
    for component in components {
        out.extend_from_slice(component);
    }
    for projection in projections {
        out.extend_from_slice(projection);
    }
    for image_descriptor in image_descriptors {
        out.extend_from_slice(image_descriptor);
    }
    out.extend_from_slice(values);
    for image in final_images {
        out.extend_from_slice(image);
    }
    if out.len() != capacity {
        return Err(error("S7 directory bytes do not exactly cover the payload"));
    }
    let payload_digest = v2_digest(
        b"gpu-db/write001/s7-payload/v2",
        &[&total.to_le_bytes(), &out[..568], &[0; 32], &out[600..]],
    );
    put_digest(&mut out, 568, payload_digest);
    Ok(out)
}

pub(super) fn initial_overlay_root(
    identity: LiveTypedInsertIdentity,
    initial_database_root: [u8; 32],
) -> [u8; 32] {
    v2_digest(
        b"gpu-db/write001/s7-initial-overlay-root/v2",
        &[
            &identity.canonical.database_id,
            &identity.catalog_epoch.to_le_bytes(),
            &identity.catalog_digest,
            &identity.stable_transaction_id.to_le_bytes(),
            &identity.commit_sequence.to_le_bytes(),
            &initial_database_root,
        ],
    )
}

/// Derive the frozen parent request identity before the physical envelope exists.  The live
/// terminal uses this narrow scalar authority to prepare independent pre-WAL control records in
/// parallel with envelope materialization, then proves that the completed envelope chose the
/// same identity before any record can commit.
pub(crate) fn live_autocommit_request_digest(typed_statement_digest: [u8; 32]) -> [u8; 32] {
    v2_digest(
        b"gpu-db/write001/aggregate-request/v2",
        &[
            &[1],
            &1_u32.to_le_bytes(),
            &0_u32.to_le_bytes(),
            &typed_statement_digest,
            &0_u32.to_le_bytes(),
        ],
    )
}

/// Derive the explicit-transaction request identity from the ordered typed statement identities
/// before the physical codec-5 envelope exists. WRITE-001 currently admits no RETURNING artifact,
/// so each canonical projection count is zero. The retained reader recomputes this exact framing
/// from S1/S7 and rejects any mismatch.
pub(crate) fn live_explicit_request_digest(
    typed_statement_digests: &[[u8; 32]],
    final_writer_statement_digests: &[(u32, [u8; 32])],
) -> Result<[u8; 32], EngineError> {
    Ok(
        live_explicit_request_hasher(typed_statement_digests, final_writer_statement_digests)?
            .finalize()
            .into(),
    )
}

/// Bind an independently ordered catalog envelope into the same explicit codec-5 request
/// identity. The S3 body is already canonical and aggregate-authenticated; this suffix prevents
/// two catalog programs with identical typed INSERT statements from sharing retry identity.
pub(crate) fn live_explicit_request_digest_with_catalog(
    typed_statement_digests: &[[u8; 32]],
    final_writer_statement_digests: &[(u32, [u8; 32])],
    catalog_operation_body: &[u8],
) -> Result<[u8; 32], EngineError> {
    if catalog_operation_body.is_empty() {
        return Err(error(
            "explicit catalog request has an empty S3 operation body",
        ));
    }
    let mut digest =
        live_explicit_request_hasher(typed_statement_digests, final_writer_statement_digests)?;
    digest.update(b"CATALOG1");
    digest.update(
        u64::try_from(catalog_operation_body.len())
            .map_err(|_| error("explicit catalog operation length exceeds u64"))?
            .to_le_bytes(),
    );
    digest.update(catalog_operation_body);
    Ok(digest.finalize().into())
}

pub(super) fn live_explicit_request_hasher(
    typed_statement_digests: &[[u8; 32]],
    final_writer_statement_digests: &[(u32, [u8; 32])],
) -> Result<Sha256, EngineError> {
    let statement_count = u32::try_from(typed_statement_digests.len())
        .map_err(|_| error("explicit typed statement count exceeds u32"))?;
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/aggregate-request/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update([2]);
    digest.update(statement_count.to_le_bytes());
    for (ordinal, statement_digest) in typed_statement_digests.iter().enumerate() {
        digest.update(
            u32::try_from(ordinal)
                .map_err(|_| error("explicit typed statement ordinal exceeds u32"))?
                .to_le_bytes(),
        );
        digest.update(statement_digest);
        digest.update(0_u32.to_le_bytes());
    }
    if !final_writer_statement_digests.is_empty() {
        let writer_count = u32::try_from(final_writer_statement_digests.len())
            .map_err(|_| error("explicit final-writer count exceeds u32"))?;
        digest.update(b"FINALWRITERS1");
        digest.update(writer_count.to_le_bytes());
        for (ordinal, statement_digest) in final_writer_statement_digests {
            if *statement_digest == [0; 32] {
                return Err(error(
                    "explicit final-writer binding has a zero statement digest",
                ));
            }
            digest.update(ordinal.to_le_bytes());
            digest.update(statement_digest);
        }
    }
    Ok(digest)
}

/// Canonical identifier commitment shared by the device-generation adapter and the S7 writer.
/// It authenticates a catalog name already carried by the sealed typed record; it is never a
/// substitute for a GPU-produced generation or publication root.
pub(crate) fn write001_identifier_digest(value: &str) -> Result<[u8; 32], EngineError> {
    let bytes = u32::try_from(value.len())
        .map_err(|_| error("identifier length exceeds the canonical domain"))?;
    Ok(v2_digest(
        b"gpu-db/write001/s7-identifier/v2",
        &[&bytes.to_le_bytes(), value.as_bytes()],
    ))
}

pub(super) fn digest_list(domain: &[u8], stable_table_id: u64, values: &[[u8; 32]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(stable_table_id.to_le_bytes());
    digest.update((values.len() as u32).to_le_bytes());
    for value in values {
        digest.update(value);
    }
    digest.finalize().into()
}

/// Hash the fixed transition-digest field directly out of the canonical transition directory.
///
/// The live generic writer already owns the complete fixed-width directory for S7 emission.  A
/// second `Vec<[u8; 32]>` would be a purely transient duplicate of the bytes at offsets 128..160
/// of each record.  Streaming those exact canonical fields keeps the wire root unchanged while
/// avoiding a per-row host allocation/copy for every typed INSERT shape.
pub(super) fn digest_transition_records(
    domain: &[u8],
    stable_table_id: u64,
    transitions: &[[u8; 192]],
) -> Result<[u8; 32], EngineError> {
    let count = u32::try_from(transitions.len())
        .map_err(|_| error("transition directory count exceeds u32"))?;
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(stable_table_id.to_le_bytes());
    digest.update(count.to_le_bytes());
    for transition in transitions {
        digest.update(&transition[128..160]);
    }
    Ok(digest.finalize().into())
}

/// Preserve the frozen S7 table-manifest byte order without collecting the interleaved
/// transition digests into another contiguous carrier.
pub(super) fn table_manifest_digest(
    table: &[u8; 384],
    target_digest: [u8; 32],
    table_dispositions: &[u8],
    index_digests: &[[u8; 32]],
    transitions: &[[u8; 192]],
    effect_digests: &[[u8; 32]],
    image_descriptor_digest: [u8; 32],
) -> Result<[u8; 32], EngineError> {
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-table-manifest/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(&table[..352]);
    digest.update([0; 32]);
    digest.update(target_digest);
    digest.update(table_dispositions);
    for index in index_digests {
        digest.update(index);
    }
    for transition in transitions {
        digest.update(&transition[128..160]);
    }
    for effect in effect_digests {
        digest.update(effect);
    }
    digest.update(image_descriptor_digest);
    Ok(digest.finalize().into())
}

pub(super) fn qualified_name_digest(schema: &str, name: &str) -> Result<[u8; 32], EngineError> {
    let schema_len = u32::try_from(schema.len()).map_err(|_| error("schema name exceeds u32"))?;
    let name_len = u32::try_from(name.len()).map_err(|_| error("object name exceeds u32"))?;
    Ok(v2_digest(
        b"gpu-db/write001/s7-qualified-name/v2",
        &[
            &schema_len.to_le_bytes(),
            schema.as_bytes(),
            &name_len.to_le_bytes(),
            name.as_bytes(),
        ],
    ))
}

pub(super) fn index_descriptor_digest(index: &[u8; 384], keys: &[IndexedS7Key]) -> [u8; 32] {
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-index-descriptor/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(&index[..304]);
    digest.update([0; 32]);
    digest.update(&index[336..]);
    for key in keys {
        digest.update(key.digest);
    }
    digest.finalize().into()
}

pub(super) fn typed_key_value_digest(
    valid: bool,
    value: DecodedTypedValueFacts<'_>,
    storage: [u8; 4],
    type_oid: u32,
    type_size: i16,
) -> ([u8; 32], Vec<u8>) {
    let bytes = if valid {
        match value {
            DecodedTypedValueFacts::I32(value) => value.to_le_bytes().to_vec(),
            DecodedTypedValueFacts::I64(value) => value.to_le_bytes().to_vec(),
            DecodedTypedValueFacts::I128(value) => value.to_le_bytes().to_vec(),
            DecodedTypedValueFacts::Uuid(value) => value.to_vec(),
            DecodedTypedValueFacts::Bool(value) => vec![u8::from(value)],
            DecodedTypedValueFacts::Text(value) => value.as_bytes().to_vec(),
        }
    } else {
        Vec::new()
    };
    let digest = typed_key_value_digest_from_bytes(valid, &bytes, storage, type_oid, type_size);
    (digest, bytes)
}

pub(super) fn typed_key_value_digest_from_bytes(
    valid: bool,
    bytes: &[u8],
    storage: [u8; 4],
    type_oid: u32,
    type_size: i16,
) -> [u8; 32] {
    let bytes_len = u32::try_from(bytes.len()).expect("typed-value bytes fit u32");
    v2_digest(
        b"gpu-db/write001/s7-typed-key-value/v2",
        &[
            &storage,
            &type_oid.to_le_bytes(),
            &type_size.to_le_bytes(),
            &[u8::from(!valid)],
            &bytes_len.to_le_bytes(),
            bytes,
        ],
    )
}

pub(super) fn logical_returning_digest(
    record: &DecodedTypedInsertRecord,
    statement_ordinal: u32,
    row_count: u32,
    projections: &[[u8; 128]],
) -> Result<[u8; 32], EngineError> {
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-statement-returning-result/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(statement_ordinal.to_le_bytes());
    digest.update(row_count.to_le_bytes());
    digest.update(
        u32::try_from(projections.len())
            .map_err(|_| error("RETURNING projection count exceeds u32"))?
            .to_le_bytes(),
    );
    for projection in projections {
        digest.update(&projection[96..128]);
    }
    for row in 0..row_count {
        digest.update(row.to_le_bytes());
        for projection in projections {
            let projection_ordinal =
                u32::from_le_bytes(projection[8..12].try_into().expect("fixed projection"));
            let source_catalog_ordinal =
                u32::from_le_bytes(projection[12..16].try_into().expect("fixed projection"));
            digest.update(projection_ordinal.to_le_bytes());
            digest.update(source_catalog_ordinal.to_le_bytes());
            digest.update(&projection[16..20]);
            digest.update(&projection[24..26]);
            digest.update(&projection[28..32]);
            digest.update(&projection[32..36]);
            digest.update(&projection[36..38]);
            digest.update(&projection[38..40]);
            let (valid, value) = record.column_value_at(source_catalog_ordinal, row)?;
            append_returning_value(&mut digest, valid, value);
        }
    }
    Ok(digest.finalize().into())
}

pub(super) fn append_returning_value(
    digest: &mut Sha256,
    valid: bool,
    value: DecodedTypedValueFacts<'_>,
) {
    digest.update([u8::from(!valid)]);
    if !valid {
        digest.update(0_u32.to_le_bytes());
        return;
    }
    match value {
        DecodedTypedValueFacts::I32(value) => {
            digest.update(4_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        DecodedTypedValueFacts::I64(value) => {
            digest.update(8_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        DecodedTypedValueFacts::I128(value) => {
            digest.update(16_u32.to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        DecodedTypedValueFacts::Uuid(value) => {
            digest.update(16_u32.to_le_bytes());
            digest.update(value);
        }
        DecodedTypedValueFacts::Bool(value) => {
            digest.update(1_u32.to_le_bytes());
            digest.update([u8::from(value)]);
        }
        DecodedTypedValueFacts::Text(value) => {
            digest.update((value.len() as u32).to_le_bytes());
            digest.update(value.as_bytes());
        }
    }
}

pub(super) fn typed_key_digest(effect_ref: u32, components: &[[u8; 32]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-typed-key/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(effect_ref.to_le_bytes());
    digest.update([2]);
    digest.update((components.len() as u32).to_le_bytes());
    for component in components {
        digest.update(component);
    }
    digest.finalize().into()
}

#[allow(clippy::too_many_arguments)] // positional arguments mirror the fixed S7 binary layout
pub(super) fn key_effect_digest(
    effect_ref: u32,
    role: u8,
    transition_ref: u32,
    index_ref: u32,
    source_catalog_ordinal: u32,
    component_start: u32,
    component_digests: &[[u8; 32]],
    index_descriptor_digest: [u8; 32],
    contains_null: bool,
    participates: bool,
) -> [u8; 32] {
    let mut raw = [0_u8; 192];
    put_u32(&mut raw, 0, effect_ref);
    raw[4] = role;
    raw[5] = role;
    put_u32(&mut raw, 8, transition_ref);
    put_u32(&mut raw, 12, index_ref);
    put_u32(&mut raw, 20, ABSENT_U32);
    put_u32(&mut raw, 28, component_start);
    put_u32(&mut raw, 32, component_digests.len() as u32);
    put_u32(&mut raw, 36, component_digests.len() as u32);
    raw[41] = 1;
    raw[42] = 1;
    raw[43] = u8::from(participates);
    raw[44] = u8::from(contains_null);
    put_u32(&mut raw, 48, source_catalog_ordinal);
    put_digest(
        &mut raw,
        96,
        typed_key_digest(effect_ref, component_digests),
    );
    let component_bytes = component_digests.concat();
    v2_digest(
        b"gpu-db/write001/s7-key-effect/v2",
        &[
            &raw[..16],
            &ABSENT_U32.to_le_bytes(),
            &raw[20..128],
            &[0; 32],
            &raw[160..],
            &index_descriptor_digest,
            &component_bytes,
        ],
    )
}

pub(super) fn index_dependency(
    reference: u32,
    kind: u8,
    index: &[u8; 384],
    descriptor_digest: [u8; 32],
    validation_floor: u64,
    key_effect_ref: u32,
    effect_digest: [u8; 32],
) -> [u8; 224] {
    let stable_index_id = u64::from_le_bytes(index[16..24].try_into().expect("fixed index id"));
    let display_oid = u32::from_le_bytes(index[24..28].try_into().expect("fixed index oid"));
    let base_generation =
        u64::from_le_bytes(index[352..360].try_into().expect("fixed index generation"));
    let catalog_epoch = u64::from_le_bytes(index[72..80].try_into().expect("fixed catalog epoch"));
    let owner_table_ref = u32::from_le_bytes(index[8..12].try_into().expect("fixed table ref"));
    let index_ref = u32::from_le_bytes(index[..4].try_into().expect("fixed index ref"));
    let live = key_effect_ref != ABSENT_U32;
    let mut raw = [0_u8; 224];
    put_u32(&mut raw, 0, reference);
    raw[4] = kind;
    raw[5] = if kind == 3 { 3 } else { 2 };
    put_u16(&mut raw, 6, u16::from(live));
    put_u64(&mut raw, 8, stable_index_id);
    put_u32(&mut raw, 16, display_oid);
    put_u32(&mut raw, 20, owner_table_ref);
    put_u64(&mut raw, 24, base_generation);
    put_u64(&mut raw, 32, validation_floor);
    put_u32(&mut raw, 40, key_effect_ref);
    put_u32(&mut raw, 44, index_ref);
    put_u64(&mut raw, 48, catalog_epoch);
    raw[64..96].copy_from_slice(&index[80..112]);
    raw[96..128].copy_from_slice(&index[240..272]);
    raw[128..160].copy_from_slice(&index[144..176]);
    let identity = v2_digest(
        b"gpu-db/write001/s7-index-object/v2",
        &[
            &[kind],
            &stable_index_id.to_le_bytes(),
            &display_oid.to_le_bytes(),
            &catalog_epoch.to_le_bytes(),
            &base_generation.to_le_bytes(),
            &raw[64..96],
            &raw[96..128],
            &raw[128..160],
            &descriptor_digest,
            &effect_digest,
        ],
    );
    put_digest(&mut raw, 160, identity);
    let digest = v2_digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&raw[..192], &[0; 32], &descriptor_digest, &effect_digest],
    );
    put_digest(&mut raw, 192, digest);
    raw
}

pub(super) fn published_sequence_dependency(
    reference: u32,
    catalog_epoch: u64,
    entry: &SequenceEntry,
) -> [u8; 224] {
    let SequenceEntryKind::Published {
        stable_sequence_id,
        sequence_oid,
        transition_txn_id,
        name_digest,
        reference_body_digest,
        body,
        final_value_overwritten: _,
    } = entry.kind
    else {
        unreachable!("only published entries own sequence dependencies")
    };
    let identity = v2_digest(
        b"gpu-db/write001/s7-published-sequence/v2",
        &[
            &stable_sequence_id.to_le_bytes(),
            &sequence_oid.to_le_bytes(),
            &catalog_epoch.to_le_bytes(),
            &transition_txn_id.to_le_bytes(),
            &name_digest,
            &reference_body_digest,
            &body,
        ],
    );
    let mut raw = [0_u8; 224];
    put_u32(&mut raw, 0, reference);
    raw[4] = 8;
    raw[5] = 4;
    put_u64(&mut raw, 8, stable_sequence_id);
    put_u32(&mut raw, 16, sequence_oid);
    put_u32(&mut raw, 20, entry.table_ref);
    put_u64(&mut raw, 24, transition_txn_id);
    put_u32(&mut raw, 40, ABSENT_U32);
    put_u32(&mut raw, 44, ABSENT_U32);
    put_u64(&mut raw, 48, catalog_epoch);
    put_digest(&mut raw, 96, reference_body_digest);
    put_digest(&mut raw, 128, name_digest);
    put_digest(&mut raw, 160, identity);
    let token_digest = v2_digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&raw[..192], &[0; 32], &[0; 32], &[0; 32]],
    );
    put_digest(&mut raw, 192, token_digest);
    raw
}

pub(super) fn dependency_use(
    statement_ordinal: u32,
    dependency_ref: u32,
    role: u16,
    source_ordinal: u32,
    transition_ref: u32,
    key_effect_ref: u32,
) -> [u8; 32] {
    let mut raw = [0_u8; 32];
    put_u32(&mut raw, 0, statement_ordinal);
    put_u32(&mut raw, 4, dependency_ref);
    put_u16(&mut raw, 8, role);
    put_u32(&mut raw, 12, source_ordinal);
    put_u32(&mut raw, 16, transition_ref);
    put_u32(&mut raw, 20, key_effect_ref);
    raw
}

/// S7 exposes one globally ordered dependency directory, although individual table closures
/// are assembled independently. Reindex the completed directory once, then carry that exact
/// map through the existing effects, statement uses, and table manifests. This is directory
/// normalization only: token identities, descriptors, effects, and semantic roles are not
/// changed or reinterpreted.
pub(super) fn canonicalize_dependency_references(
    dependencies: &mut Vec<[u8; 224]>,
    dependency_uses: &mut [[u8; 32]],
    effects: &mut [[u8; 192]],
    indexes: &[[u8; 384]],
) -> Result<Vec<u32>, EngineError> {
    let dependency_count = dependencies.len();
    let mut ordered = dependencies.drain(..).enumerate().collect::<Vec<_>>();
    ordered.sort_unstable_by(|(_, left), (_, right)| dependency_token_order(left, right));
    if ordered
        .windows(2)
        .any(|pair| dependency_token_order(&pair[0].1, &pair[1].1) != std::cmp::Ordering::Less)
    {
        return Err(error(
            "S7 dependency tokens are not uniquely identity-orderable",
        ));
    }

    let mut remap = vec![ABSENT_U32; dependency_count];
    let mut normalized = Vec::with_capacity(dependency_count);
    for (new_ordinal, (old_ordinal, mut token)) in ordered.into_iter().enumerate() {
        let new_ref =
            u32::try_from(new_ordinal).map_err(|_| error("S7 dependency count exceeds u32"))?;
        let old_ref = remap
            .get_mut(old_ordinal)
            .ok_or_else(|| error("S7 dependency remap ordinal is absent"))?;
        if *old_ref != ABSENT_U32 {
            return Err(error("S7 dependency remap contains a duplicate ordinal"));
        }
        *old_ref = new_ref;
        put_u32(&mut token, 0, new_ref);
        refresh_dependency_token_digest(&mut token, indexes, effects)?;
        normalized.push(token);
    }

    for usage in dependency_uses {
        remap_dependency_reference(&mut usage[4..8], &remap, "statement dependency use")?;
    }
    for effect in effects {
        if effect[16..20] != ABSENT_U32.to_le_bytes() {
            remap_dependency_reference(&mut effect[16..20], &remap, "key effect")?;
        }
    }
    *dependencies = normalized;
    Ok(remap)
}

pub(super) fn remap_dependency_reference(
    raw: &mut [u8],
    remap: &[u32],
    owner: &str,
) -> Result<(), EngineError> {
    let old_ref = u32::from_le_bytes(
        raw.try_into()
            .map_err(|_| error("S7 dependency reference has invalid width"))?,
    );
    let new_ref = remap
        .get(old_ref as usize)
        .copied()
        .filter(|reference| *reference != ABSENT_U32)
        .ok_or_else(|| error(&format!("S7 {owner} references an absent dependency")))?;
    raw.copy_from_slice(&new_ref.to_le_bytes());
    Ok(())
}

pub(super) fn refresh_dependency_token_digest(
    token: &mut [u8; 224],
    indexes: &[[u8; 384]],
    effects: &[[u8; 192]],
) -> Result<(), EngineError> {
    let descriptor_ref =
        u32::from_le_bytes(token[44..48].try_into().expect("fixed descriptor ref"));
    let effect_ref = u32::from_le_bytes(token[40..44].try_into().expect("fixed effect ref"));
    let descriptor_digest = if descriptor_ref == ABSENT_U32 {
        [0; 32]
    } else {
        indexes
            .get(descriptor_ref as usize)
            .map(|index| index[304..336].try_into().expect("fixed descriptor digest"))
            .ok_or_else(|| error("S7 dependency token references an absent index descriptor"))?
    };
    let effect_digest = if effect_ref == ABSENT_U32 {
        [0; 32]
    } else {
        effects
            .get(effect_ref as usize)
            .map(|effect| {
                effect[128..160]
                    .try_into()
                    .expect("fixed key-effect digest")
            })
            .ok_or_else(|| error("S7 dependency token references an absent key effect"))?
    };
    let token_digest = v2_digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&token[..192], &[0; 32], &descriptor_digest, &effect_digest],
    );
    put_digest(token, 192, token_digest);
    Ok(())
}

pub(super) fn dependency_token_order(left: &[u8; 224], right: &[u8; 224]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let terminal_left =
        u16::from_le_bytes(left[6..8].try_into().expect("fixed dependency flags")) & 2 != 0;
    let terminal_right =
        u16::from_le_bytes(right[6..8].try_into().expect("fixed dependency flags")) & 2 != 0;
    let read_u64 = |raw: &[u8; 224], offset: usize| {
        u64::from_le_bytes(
            raw[offset..offset + 8]
                .try_into()
                .expect("fixed dependency u64"),
        )
    };
    let read_u32 = |raw: &[u8; 224], offset: usize| {
        u32::from_le_bytes(
            raw[offset..offset + 4]
                .try_into()
                .expect("fixed dependency u32"),
        )
    };
    let scalars = [
        (u64::from(left[4]), u64::from(right[4])),
        (u64::from(terminal_left), u64::from(terminal_right)),
        (read_u64(left, 8), read_u64(right, 8)),
        (
            u64::from(read_u32(left, 16)),
            u64::from(read_u32(right, 16)),
        ),
        (
            u64::from(read_u32(left, 20)),
            u64::from(read_u32(right, 20)),
        ),
        (read_u64(left, 48), read_u64(right, 48)),
        (read_u64(left, 24), read_u64(right, 24)),
        (
            u64::from(read_u32(left, 40)),
            u64::from(read_u32(right, 40)),
        ),
        (
            u64::from(read_u32(left, 44)),
            u64::from(read_u32(right, 44)),
        ),
    ];
    for (left, right) in scalars {
        match left.cmp(&right) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    for (start, end) in [(64, 96), (96, 128), (128, 160), (160, 192)] {
        match left[start..end].cmp(&right[start..end]) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    Ordering::Equal
}

pub(super) fn statement_dependency_root(
    statement_ordinal: u32,
    dependencies: &[[u8; 224]],
    uses: &[[u8; 32]],
) -> Result<[u8; 32], EngineError> {
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-statement-dependencies/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(statement_ordinal.to_le_bytes());
    digest.update(
        u32::try_from(uses.len())
            .map_err(|_| error("S7 dependency-use count exceeds u32"))?
            .to_le_bytes(),
    );
    for usage in uses {
        let dependency_ref = u32::from_le_bytes(usage[4..8].try_into().expect("fixed use ref"));
        let token = dependencies
            .get(dependency_ref as usize)
            .ok_or_else(|| error("S7 dependency use references an absent token"))?;
        digest.update(usage);
        digest.update(&token[192..224]);
    }
    Ok(digest.finalize().into())
}

pub(super) fn statement_projection_root(
    statement_ordinal: u32,
    projections: &[[u8; 128]],
) -> Result<[u8; 32], EngineError> {
    let mut digest = Sha256::new();
    let domain = b"gpu-db/write001/s7-statement-projections/v2";
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(statement_ordinal.to_le_bytes());
    digest.update(
        u32::try_from(projections.len())
            .map_err(|_| error("RETURNING projection count exceeds u32"))?
            .to_le_bytes(),
    );
    for projection in projections {
        digest.update(&projection[96..128]);
    }
    Ok(digest.finalize().into())
}

pub(super) fn v2_digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    for field in fields {
        digest.update(field);
    }
    digest.finalize().into()
}

pub(super) fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, EngineError> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| error("fixed u32 source is truncated"))
}

pub(super) fn put_u16(raw: &mut [u8], offset: usize, value: u16) {
    raw[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn put_i16(raw: &mut [u8], offset: usize, value: i16) {
    raw[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn put_u32(raw: &mut [u8], offset: usize, value: u32) {
    raw[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn put_u32_vec(raw: &mut Vec<u8>, value: u32) {
    raw.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn put_u64(raw: &mut [u8], offset: usize, value: u64) {
    raw[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn put_digest(raw: &mut [u8], offset: usize, value: [u8; 32]) {
    raw[offset..offset + 32].copy_from_slice(&value);
}

pub(super) fn error(message: &str) -> EngineError {
    EngineError::Durability(format!(
        "typed INSERT aggregate semantics-v2 live writer: {message}"
    ))
}
