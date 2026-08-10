//! Typed INSERT transaction-overlay artifact.
//!
//! The statement-stage owner consumes `TypedInsertBatch` into this immutable, cloneable record.
//! It retains the already-columnar private payload plus the sealed codec-5 sources, never a
//! row-string duplicate, batch, prepared device plan, SQL command, or `WriteDelta`.

use super::*;

/// One exact private sequence advance consumed while sealing a typed INSERT.  It is scalar
/// transaction/WAL provenance only: row values remain owned by the immutable columnar payload.
/// Retaining the classifier evidence in the staged artifact lets later statements and rebase
/// continue the same private sequence chain without reconstructing SQL or a `WriteDelta`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TypedPrivateSequenceAdvance {
    pub(crate) sequence_name: String,
    pub(crate) sequence_oid: u32,
    pub(crate) next_state: (i64, bool),
    pub(crate) lifetime_origin: u8,
    pub(crate) owner_kind: u8,
    pub(crate) owner_statement_ordinal: u32,
    pub(crate) owner_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) owner_creator_catalog_column_ordinal: Option<u32>,
    pub(crate) predecessor_tag: u8,
    pub(crate) predecessor_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) child_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) outcome_digest: gpu_db_wal::CanonicalDigest,
}

/// One post-consumption typed INSERT staged in an explicit transaction.
///
/// Clones share immutable payload and codec-5 source allocations so transaction generation
/// rebases, reset folding, and WAL binding cannot recreate or reinterpret the consumed typed
/// batch.
#[derive(Clone)]
pub(crate) struct StagedTypedInsert {
    pub(crate) statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) typed_statement_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) statement_ordinal: u32,
    /// Position in the complete transaction operation program. Codec-5 statement ordinals count
    /// only typed INSERT statements; a preceding lifecycle operation (for example a private
    /// sequence RESTART) retains its own global position without punching a hole in S1/S2.
    pub(crate) operation_ordinal: u32,
    /// The first generic codec-5 vertical has no result artifact. This is bound at statement
    /// staging, before the immutable artifact enters the transaction operation list.
    pub(crate) returning_requested: bool,
    pub(crate) table: String,
    pub(crate) stable_table_id: u64,
    pub(crate) table_oid: u32,
    pub(crate) table_schema_digest: gpu_db_wal::CanonicalDigest,
    pub(crate) prepared_catalog_seq: Index,
    pub(crate) read_snapshot: Index,
    pub(crate) catalog_dependencies: BTreeMap<String, RelationalTable>,
    /// Parent and child tables whose mutation history must remain stable through the canonical
    /// final-record GPU foreign-key verdict at COMMIT.
    pub(crate) foreign_key_dependencies: BTreeSet<String>,
    /// The statement boundary at which the foreign-key relation closure was bound. A READ
    /// COMMITTED rebase refreshes the private shard's read snapshot, but it must not erase an
    /// intervening provider or child mutation from this final-record conflict floor.
    pub(crate) foreign_key_read_snapshot: Index,
    pub(crate) provisional_row_ids: Arc<[u64]>,
    pub(crate) write_set: WriteSet,
    /// Exact host upload for the private GPU shard: header plus type-grouped fixed-width vectors.
    /// These are private so the constructor proof remains the sole source of their value identity;
    /// downstream consumers receive only narrow immutable borrows.
    private_device_payload: Arc<[u8]>,
    private_payload_digest: gpu_db_wal::CanonicalDigest,
    private_int4_stats: Arc<[ResidentDeviceInt4ColumnStats]>,
    private_null_layouts: Arc<[ResidentDeviceNullBitmapLayout]>,
    private_bool_layouts: Arc<[ResidentDeviceBoolColumnLayout]>,
    private_text_layouts: Arc<[ResidentDeviceTextColumnLayout]>,
    /// Exact S2 and shared typed-image bytes produced once from `TypedInsertBatch` before the
    /// physical stage consumes its catalog-order vectors.
    pub(crate) codec5_sources: crate::typed_insert_batch::SealedTypedInsertCodec5Sources,
    /// Private CREATE/RESTART sequence advances in request order. Published ordinary sequence
    /// transitions instead retain their independent durable references on the transaction delta.
    pub(crate) private_sequence_advances: Arc<[TypedPrivateSequenceAdvance]>,
    /// Build-only construction attribution. It follows the immutable staged owner, so the
    /// statement executor can report constructor work without rebuilding or re-reading payloads.
    #[cfg(feature = "probe-timing")]
    private_probe_nanos: [u64; 3],
}

impl std::fmt::Debug for StagedTypedInsert {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StagedTypedInsert")
            .field("table", &self.table)
            .field("statement_ordinal", &self.statement_ordinal)
            .field("operation_ordinal", &self.operation_ordinal)
            .field("rows", &self.rows_consumed())
            .finish_non_exhaustive()
    }
}

fn fixed_stage_type(ty: SqlType) -> bool {
    matches!(
        ty,
        SqlType::Int2
            | SqlType::Int4
            | SqlType::Date
            | SqlType::Int8
            | SqlType::Timestamp
            | SqlType::Numeric { .. }
            | SqlType::Uuid
    )
}

fn private_stage_type(ty: SqlType) -> bool {
    fixed_stage_type(ty) || matches!(ty, SqlType::Bool | SqlType::Text)
}

/// Return the exact fixed-width private payload length, every catalog-column byte offset, and
/// the number of i32-section statistics. The layout deliberately matches
/// `build_relational_device_payload`: type-grouped, catalog order within each group.
fn private_fixed_payload_layout(
    table: &RelationalTable,
    rows: usize,
    bool_layouts: &[ResidentDeviceBoolColumnLayout],
    null_layouts: &[ResidentDeviceNullBitmapLayout],
    text_layouts: &[ResidentDeviceTextColumnLayout],
) -> Result<(usize, Vec<usize>, usize), &'static str> {
    if table.columns.is_empty()
        || !table
            .columns
            .iter()
            .all(|column| private_stage_type(column.ty))
    {
        return Err("typed transaction INSERT payload has unsupported column types");
    }
    let mut offsets = vec![0; table.columns.len()];
    let mut offset = std::mem::size_of::<u64>();
    let mut int4_columns = 0;
    for group in 0..3 {
        for (column_index, column) in table.columns.iter().enumerate() {
            let width = match group {
                0 if matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date) => {
                    std::mem::size_of::<i32>()
                }
                1 if matches!(column.ty, SqlType::Int8 | SqlType::Timestamp) => {
                    std::mem::size_of::<i64>()
                }
                2 if matches!(column.ty, SqlType::Numeric { .. } | SqlType::Uuid) => 16,
                _ => continue,
            };
            if width == 0 {
                continue;
            }
            offsets[column_index] = offset;
            offset = offset
                .checked_add(
                    rows.checked_mul(width)
                        .ok_or("typed transaction INSERT payload size overflow")?,
                )
                .ok_or("typed transaction INSERT payload size overflow")?;
            if width == std::mem::size_of::<i32>() {
                int4_columns += 1;
            }
        }
    }
    let bitmap_bytes = rows
        .checked_add(31)
        .map(|value| value / 32)
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
        .ok_or("typed transaction INSERT bitmap size overflow")?;
    let expected_bool_columns = table
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Bool)
        .count();
    if bool_layouts.len() != expected_bool_columns {
        return Err("typed transaction INSERT BOOL bitmap layout count drifted");
    }
    let mut previous_bool_column = None;
    for layout in bool_layouts {
        let column_index = table
            .columns
            .iter()
            .position(|column| column.name == layout.name && column.ty == SqlType::Bool)
            .ok_or("typed transaction INSERT BOOL bitmap references unknown column")?;
        if previous_bool_column.is_some_and(|previous| column_index <= previous)
            || usize::try_from(layout.bitmap_byte_offset).ok() != Some(offset)
        {
            return Err("typed transaction INSERT BOOL bitmap layout drifted");
        }
        previous_bool_column = Some(column_index);
        offset = offset
            .checked_add(bitmap_bytes)
            .ok_or("typed transaction INSERT BOOL bitmap size overflow")?;
    }
    let mut previous_column = None;
    for layout in null_layouts {
        let column_index = table
            .columns
            .iter()
            .position(|column| column.name == layout.name)
            .ok_or("typed transaction INSERT NULL bitmap references unknown column")?;
        if previous_column.is_some_and(|previous| column_index <= previous)
            || usize::try_from(layout.bitmap_byte_offset).ok() != Some(offset)
        {
            return Err("typed transaction INSERT NULL bitmap layout drifted");
        }
        previous_column = Some(column_index);
        offset = offset
            .checked_add(bitmap_bytes)
            .ok_or("typed transaction INSERT NULL bitmap size overflow")?;
    }
    let mut previous_text_column = None;
    let expected_text_columns = table
        .columns
        .iter()
        .filter(|column| column.ty == SqlType::Text)
        .count();
    if text_layouts.len() != expected_text_columns {
        return Err("typed transaction INSERT text layout count drifted");
    }
    for layout in text_layouts {
        let column_index = table
            .columns
            .iter()
            .position(|column| column.name == layout.name && column.ty == SqlType::Text)
            .ok_or("typed transaction INSERT text layout references unknown column")?;
        if previous_text_column.is_some_and(|previous| column_index <= previous) {
            return Err("typed transaction INSERT text layouts are out of catalog order");
        }
        previous_text_column = Some(column_index);
        while !offset.is_multiple_of(8) {
            offset = offset
                .checked_add(1)
                .ok_or("typed transaction INSERT text alignment overflow")?;
        }
        if usize::try_from(layout.offsets_byte_offset).ok() != Some(offset) {
            return Err("typed transaction INSERT text offsets layout drifted");
        }
        let offsets_bytes = rows
            .checked_add(1)
            .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
            .ok_or("typed transaction INSERT text offsets size overflow")?;
        offset = offset
            .checked_add(offsets_bytes)
            .ok_or("typed transaction INSERT text offsets size overflow")?;
        if usize::try_from(layout.bytes_byte_offset).ok() != Some(offset) {
            return Err("typed transaction INSERT text bytes layout drifted");
        }
        offset = offset
            .checked_add(
                usize::try_from(layout.bytes_len)
                    .map_err(|_| "typed transaction INSERT text bytes length overflow")?,
            )
            .ok_or("typed transaction INSERT text bytes size overflow")?;
    }
    Ok((offset, offsets, int4_columns))
}

fn private_null_layouts_are_canonical(
    payload: &[u8],
    rows: usize,
    layouts: &[ResidentDeviceNullBitmapLayout],
) -> bool {
    let words = rows.div_ceil(32);
    layouts.iter().all(|layout| {
        let Ok(offset) = usize::try_from(layout.bitmap_byte_offset) else {
            return false;
        };
        let Some(bytes) = payload.get(offset..offset.saturating_add(words * 4)) else {
            return false;
        };
        let values = bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("exact NULL bitmap word")))
            .collect::<Vec<_>>();
        let tail_is_canonical = rows.is_multiple_of(32)
            || values.last().is_some_and(|word| {
                let valid_mask = (1_u32 << (rows % 32)) - 1;
                word & !valid_mask == 0
            });
        let any_invalid = values.iter().take(rows / 32).any(|word| *word != u32::MAX)
            || (!rows.is_multiple_of(32)
                && values.last().is_some_and(|word| {
                    let valid_mask = (1_u32 << (rows % 32)) - 1;
                    word != &valid_mask
                }));
        values.len() == words && tail_is_canonical && any_invalid
    })
}

fn private_bool_layouts_are_canonical(
    payload: &[u8],
    rows: usize,
    layouts: &[ResidentDeviceBoolColumnLayout],
) -> bool {
    let words = rows.div_ceil(32);
    layouts.iter().all(|layout| {
        let Ok(offset) = usize::try_from(layout.bitmap_byte_offset) else {
            return false;
        };
        let Some(bytes) = payload.get(offset..offset.saturating_add(words * 4)) else {
            return false;
        };
        let values = bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("exact BOOL bitmap word")))
            .collect::<Vec<_>>();
        values.len() == words
            && (rows.is_multiple_of(32)
                || values.last().is_some_and(|word| {
                    let valid_mask = (1_u32 << (rows % 32)) - 1;
                    word & !valid_mask == 0
                }))
    })
}

fn private_row_is_valid(
    payload: &[u8],
    layouts: &[ResidentDeviceNullBitmapLayout],
    column: &str,
    row: usize,
) -> bool {
    let Some(layout) = layouts.iter().find(|layout| layout.name == column) else {
        return true;
    };
    let Ok(offset) = usize::try_from(layout.bitmap_byte_offset) else {
        return false;
    };
    let word_offset = offset.saturating_add((row / 32) * 4);
    payload
        .get(word_offset..word_offset.saturating_add(4))
        .and_then(|bytes| bytes.try_into().ok())
        .map(u32::from_le_bytes)
        .is_some_and(|word| word & (1_u32 << (row % 32)) != 0)
}

/// Reconstruct one transient control-plane row from the immutable private payload. This is the
/// only row-wise view retained by the transaction overlay: it is used for conflict/sequence/FK
/// coordination and is dropped by each caller. It must not become a row storage or WAL carrier.
fn decode_private_payload_row(
    table: &RelationalTable,
    payload: &[u8],
    bool_layouts: &[ResidentDeviceBoolColumnLayout],
    null_layouts: &[ResidentDeviceNullBitmapLayout],
    text_layouts: &[ResidentDeviceTextColumnLayout],
    row: usize,
) -> Result<Vec<SqlValue>, &'static str> {
    let rows = usize::try_from(u64::from_le_bytes(
        payload
            .get(..std::mem::size_of::<u64>())
            .ok_or("typed transaction INSERT payload header is truncated")?
            .try_into()
            .expect("checked private payload header width"),
    ))
    .map_err(|_| "typed transaction INSERT payload row count overflows")?;
    let (expected_bytes, offsets, _) =
        private_fixed_payload_layout(table, rows, bool_layouts, null_layouts, text_layouts)?;
    if row >= rows || payload.len() != expected_bytes {
        return Err("typed transaction INSERT payload row geometry drifted");
    }
    let mut values = Vec::with_capacity(table.columns.len());
    for (column_index, column) in table.columns.iter().enumerate() {
        if !private_row_is_valid(payload, null_layouts, &column.name, row) {
            values.push(SqlValue::Null);
            continue;
        }
        let offset = offsets[column_index];
        let value = match column.ty {
            SqlType::Int2 => {
                let raw = i32::from_le_bytes(
                    payload[offset + row * 4..offset + (row + 1) * 4]
                        .try_into()
                        .expect("checked int2 payload geometry"),
                );
                SqlValue::Int2(
                    i16::try_from(raw)
                        .map_err(|_| "typed transaction INSERT int2 payload drifted")?,
                )
            }
            SqlType::Int4 => SqlValue::Int4(i32::from_le_bytes(
                payload[offset + row * 4..offset + (row + 1) * 4]
                    .try_into()
                    .expect("checked int4 payload geometry"),
            )),
            SqlType::Date => SqlValue::Date(i32::from_le_bytes(
                payload[offset + row * 4..offset + (row + 1) * 4]
                    .try_into()
                    .expect("checked date payload geometry"),
            )),
            SqlType::Int8 => SqlValue::Int8(i64::from_le_bytes(
                payload[offset + row * 8..offset + (row + 1) * 8]
                    .try_into()
                    .expect("checked int8 payload geometry"),
            )),
            SqlType::Timestamp => SqlValue::Timestamp(i64::from_le_bytes(
                payload[offset + row * 8..offset + (row + 1) * 8]
                    .try_into()
                    .expect("checked timestamp payload geometry"),
            )),
            SqlType::Numeric { scale, .. } => SqlValue::Numeric(gpu_db_sql::Decimal128::new(
                i128::from_le_bytes(
                    payload[offset + row * 16..offset + (row + 1) * 16]
                        .try_into()
                        .expect("checked numeric payload geometry"),
                ),
                scale,
            )),
            SqlType::Uuid => SqlValue::Uuid(
                payload[offset + row * 16..offset + (row + 1) * 16]
                    .try_into()
                    .expect("checked UUID payload geometry"),
            ),
            SqlType::Bool => {
                let layout = bool_layouts
                    .iter()
                    .find(|layout| layout.name == column.name)
                    .ok_or("typed transaction INSERT BOOL bitmap binding drifted")?;
                let bitmap_base = usize::try_from(layout.bitmap_byte_offset)
                    .map_err(|_| "typed transaction INSERT BOOL bitmap offset overflow")?;
                let bitmap_offset = bitmap_base + (row / 32) * std::mem::size_of::<u32>();
                let word = u32::from_le_bytes(
                    payload[bitmap_offset..bitmap_offset + std::mem::size_of::<u32>()]
                        .try_into()
                        .expect("checked BOOL bitmap geometry"),
                );
                SqlValue::Bool(word & (1_u32 << (row % 32)) != 0)
            }
            SqlType::Text => {
                let layout = text_layouts
                    .iter()
                    .find(|layout| layout.name == column.name)
                    .ok_or("typed transaction INSERT text layout binding drifted")?;
                let offsets_base = usize::try_from(layout.offsets_byte_offset)
                    .map_err(|_| "typed transaction INSERT text offsets overflow")?;
                let bytes_base = usize::try_from(layout.bytes_byte_offset)
                    .map_err(|_| "typed transaction INSERT text bytes overflow")?;
                let offset_start = offsets_base + row * std::mem::size_of::<u64>();
                let start = usize::try_from(u64::from_le_bytes(
                    payload[offset_start..offset_start + std::mem::size_of::<u64>()]
                        .try_into()
                        .expect("checked text row start geometry"),
                ))
                .map_err(|_| "typed transaction INSERT text row start overflow")?;
                let end = usize::try_from(u64::from_le_bytes(
                    payload[offset_start + std::mem::size_of::<u64>()
                        ..offset_start + 2 * std::mem::size_of::<u64>()]
                        .try_into()
                        .expect("checked text row end geometry"),
                ))
                .map_err(|_| "typed transaction INSERT text row end overflow")?;
                let bytes = payload
                    .get(bytes_base + start..bytes_base + end)
                    .ok_or("typed transaction INSERT text row bytes drifted")?;
                SqlValue::Text(
                    std::str::from_utf8(bytes)
                        .map_err(|_| "typed transaction INSERT text row bytes drifted")?
                        .to_string(),
                )
            }
        };
        values.push(value);
    }
    Ok(values)
}

impl StagedTypedInsert {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        statement_digest: gpu_db_wal::CanonicalDigest,
        typed_statement_digest: gpu_db_wal::CanonicalDigest,
        statement_ordinal: u32,
        table: &RelationalTable,
        table_schema_digest: gpu_db_wal::CanonicalDigest,
        prepared_catalog_seq: Index,
        read_snapshot: Index,
        provisional_row_ids: Arc<[u64]>,
        private_device_payload: Arc<[u8]>,
        private_int4_stats: Arc<[ResidentDeviceInt4ColumnStats]>,
        private_null_layouts: Arc<[ResidentDeviceNullBitmapLayout]>,
        codec5_sources: crate::typed_insert_batch::SealedTypedInsertCodec5Sources,
    ) -> Result<Self, EngineError> {
        Self::new_dense(
            statement_digest,
            typed_statement_digest,
            statement_ordinal,
            table,
            table_schema_digest,
            prepared_catalog_seq,
            read_snapshot,
            provisional_row_ids,
            private_device_payload,
            private_int4_stats,
            private_null_layouts,
            Arc::from([]),
            Arc::from([]),
            codec5_sources,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_dense(
        statement_digest: gpu_db_wal::CanonicalDigest,
        typed_statement_digest: gpu_db_wal::CanonicalDigest,
        statement_ordinal: u32,
        table: &RelationalTable,
        table_schema_digest: gpu_db_wal::CanonicalDigest,
        prepared_catalog_seq: Index,
        read_snapshot: Index,
        provisional_row_ids: Arc<[u64]>,
        private_device_payload: Arc<[u8]>,
        private_int4_stats: Arc<[ResidentDeviceInt4ColumnStats]>,
        private_null_layouts: Arc<[ResidentDeviceNullBitmapLayout]>,
        private_bool_layouts: Arc<[ResidentDeviceBoolColumnLayout]>,
        private_text_layouts: Arc<[ResidentDeviceTextColumnLayout]>,
        codec5_sources: crate::typed_insert_batch::SealedTypedInsertCodec5Sources,
    ) -> Result<Self, EngineError> {
        #[cfg(feature = "probe-timing")]
        let probe_payload_validate_started = std::time::Instant::now();
        let row_count = provisional_row_ids.len();
        if row_count == 0
            || table.stable_table_id == 0
            || table.stable_table_id == u64::MAX
            || table.columns.is_empty()
            || !table
                .columns
                .iter()
                .all(|column| private_stage_type(column.ty))
        {
            return Err(EngineError::ApplyFailed(
                "typed transaction INSERT artifact has invalid int4 geometry".to_string(),
            ));
        }
        let expected_bytes = private_fixed_payload_layout(
            table,
            row_count,
            &private_bool_layouts,
            &private_null_layouts,
            &private_text_layouts,
        )
        .map_err(|reason| EngineError::ApplyFailed(reason.to_string()))?
        .0;
        if private_device_payload.len() != expected_bytes {
            return Err(EngineError::ApplyFailed(
                "typed transaction INSERT payload does not match row geometry".to_string(),
            ));
        }
        Self::validate_exact_fixed_payload(
            table,
            private_device_payload.as_ref(),
            private_int4_stats.as_ref(),
            private_bool_layouts.as_ref(),
            private_null_layouts.as_ref(),
            private_text_layouts.as_ref(),
        )
        .map_err(|reason| EngineError::ApplyFailed(reason.to_string()))?;
        #[cfg(feature = "probe-timing")]
        let probe_payload_validate_nanos =
            probe_payload_validate_started.elapsed().as_nanos() as u64;
        let mut write_set = WriteSet::default();
        write_set.add_table(table);
        // A table without a UNIQUE/primary-key index has no slot to claim. Decoding every
        // columnar row into transient `SqlValue`s in that case only rebuilds data the one typed
        // lifecycle has already validated and never consumes. Indexed tables retain the exact
        // existing row-wise proof below until their typed-column key path replaces it.
        let has_unique_slots = table.indexes.iter().any(|index| index.unique);
        #[cfg(feature = "probe-timing")]
        let mut probe_unique_slots_nanos = 0_u64;
        for row_ordinal in 0..provisional_row_ids.len() {
            if has_unique_slots {
                #[cfg(feature = "probe-timing")]
                let probe_unique_slot_started = std::time::Instant::now();
                // The typed overlay is the only live INSERT semantic carrier. Its transaction
                // write set therefore retains the same UNIQUE-slot claims as the retired
                // row-delta carrier, but derives them directly from the constructor-validated
                // columnar payload instead of materializing and reparsing a row-string duplicate.
                let values = decode_private_payload_row(
                    table,
                    private_device_payload.as_ref(),
                    private_bool_layouts.as_ref(),
                    private_null_layouts.as_ref(),
                    private_text_layouts.as_ref(),
                    row_ordinal,
                )
                .map_err(|reason| EngineError::ApplyFailed(reason.to_string()))?;
                write_set.add_unique_slots(table, &values);
                #[cfg(feature = "probe-timing")]
                {
                    probe_unique_slots_nanos = probe_unique_slots_nanos
                        .saturating_add(probe_unique_slot_started.elapsed().as_nanos() as u64);
                }
            }
        }
        #[cfg(feature = "probe-timing")]
        let probe_payload_digest_started = std::time::Instant::now();
        let private_payload_digest = gpu_db_wal::canonical_request_digest(&private_device_payload);
        #[cfg(feature = "probe-timing")]
        let probe_payload_digest_nanos = probe_payload_digest_started.elapsed().as_nanos() as u64;
        Ok(Self {
            statement_digest,
            typed_statement_digest,
            statement_ordinal,
            operation_ordinal: statement_ordinal,
            returning_requested: false,
            table: table.name.clone(),
            stable_table_id: table.stable_table_id,
            table_oid: table.oid,
            table_schema_digest,
            prepared_catalog_seq,
            read_snapshot,
            catalog_dependencies: BTreeMap::from([(table.name.clone(), table.clone())]),
            foreign_key_dependencies: BTreeSet::new(),
            foreign_key_read_snapshot: read_snapshot,
            provisional_row_ids,
            write_set,
            private_payload_digest,
            private_device_payload,
            private_int4_stats,
            private_null_layouts,
            private_bool_layouts,
            private_text_layouts,
            codec5_sources,
            private_sequence_advances: Arc::from([]),
            #[cfg(feature = "probe-timing")]
            private_probe_nanos: [
                probe_payload_validate_nanos,
                probe_unique_slots_nanos,
                probe_payload_digest_nanos,
            ],
        })
    }

    /// Bind classifier-derived private sequence advancement before this artifact joins the
    /// transaction generation.  The values have already been sealed into its payload; this
    /// preserves only the stable sequence/WAL chain needed by later statements and COMMIT.
    pub(crate) fn bind_private_sequence_advances(
        &mut self,
        advances: &[TypedPrivateSequenceAdvance],
    ) -> Result<(), ExecuteError> {
        let mut previous_by_oid = BTreeMap::<u32, (i64, bool)>::new();
        for advance in advances {
            if advance.sequence_name.is_empty()
                || advance.sequence_oid == 0
                || !matches!(advance.lifetime_origin, 1 | 2)
                || !matches!(advance.owner_kind, 1..=3)
                || advance.owner_statement_digest == [0; 32]
                || !matches!(advance.predecessor_tag, 1 | 2)
                || advance.predecessor_digest == [0; 32]
                || advance.child_digest == [0; 32]
                || advance.outcome_digest == [0; 32]
                || !advance.next_state.1
            {
                return Err(ExecuteError::Engine(EngineError::Durability(
                    "typed private sequence advance lost classifier provenance".to_string(),
                )));
            }
            if let Some((prior_value, prior_called)) =
                previous_by_oid.insert(advance.sequence_oid, advance.next_state)
            {
                let expected = if prior_called {
                    prior_value.checked_add(1)
                } else {
                    Some(prior_value)
                };
                if expected != Some(advance.next_state.0) {
                    return Err(ExecuteError::Engine(EngineError::Durability(
                        "typed private sequence advances are not a contiguous request chain"
                            .to_string(),
                    )));
                }
            }
        }
        self.private_sequence_advances = advances.to_vec().into();
        Ok(())
    }

    pub(crate) fn bind_operation_ordinal(&mut self, operation_ordinal: u32) {
        self.operation_ordinal = operation_ordinal;
    }

    pub(crate) fn rows_consumed(&self) -> u64 {
        u64::try_from(self.provisional_row_ids.len())
            .expect("typed transaction row count fits in u64")
    }

    /// Borrow the one constructor-validated private upload. The byte owner remains this staged
    /// artifact; callers cannot replace it or its binding digest.
    pub(crate) fn private_device_payload(&self) -> &[u8] {
        &self.private_device_payload
    }

    pub(crate) fn private_payload_digest(&self) -> gpu_db_wal::CanonicalDigest {
        self.private_payload_digest
    }

    #[cfg(feature = "probe-timing")]
    pub(crate) fn private_probe_nanos(&self) -> [u64; 3] {
        self.private_probe_nanos
    }

    pub(crate) fn private_int4_stats(&self) -> &[ResidentDeviceInt4ColumnStats] {
        &self.private_int4_stats
    }

    pub(crate) fn private_null_layouts(&self) -> &[ResidentDeviceNullBitmapLayout] {
        &self.private_null_layouts
    }

    pub(crate) fn private_bool_layouts(&self) -> &[ResidentDeviceBoolColumnLayout] {
        &self.private_bool_layouts
    }

    pub(crate) fn private_text_layouts(&self) -> &[ResidentDeviceTextColumnLayout] {
        &self.private_text_layouts
    }

    /// Recover one short-lived control-plane row from the immutable payload. This is not an
    /// alternate write representation: the caller owns the returned transient values and the
    /// staged artifact remains the unique payload/S2 source.
    pub(crate) fn private_row_values(
        &self,
        table: &RelationalTable,
        row: usize,
    ) -> Result<Vec<SqlValue>, ExecuteError> {
        decode_private_payload_row(
            table,
            &self.private_device_payload,
            &self.private_bool_layouts,
            &self.private_null_layouts,
            &self.private_text_layouts,
            row,
        )
        .map_err(|reason| ExecuteError::Serialization(reason.to_string()))
    }

    /// Capture exactly the transaction catalog relations that the final FK verdict may read.
    /// This is performed before the artifact becomes part of a private GPU generation, so a
    /// typed child cannot later observe an unbound parent or inbound child relation at COMMIT.
    pub(crate) fn bind_foreign_key_dependencies(
        &mut self,
        catalog: &CatalogSnapshot,
    ) -> Result<(), ExecuteError> {
        let table = catalog.relational_catalog.get(&self.table).ok_or_else(|| {
            ExecuteError::Serialization(
                "typed transaction INSERT target left the statement catalog".to_string(),
            )
        })?;
        if table.oid != self.table_oid
            || crate::engine_transaction_reset::table_schema_digest(table)?
                != self.table_schema_digest
        {
            return Err(ExecuteError::Serialization(
                "typed transaction INSERT target schema changed before FK dependency binding"
                    .to_string(),
            ));
        }
        let mut names = BTreeSet::from([table.name.clone()]);
        let mut foreign_key_dependencies = BTreeSet::new();
        for foreign_key in &table.foreign_keys {
            names.insert(foreign_key.referenced_table.clone());
            foreign_key_dependencies.insert(foreign_key.referenced_table.clone());
        }
        for candidate in catalog.relational_catalog.values() {
            if candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
            {
                names.insert(candidate.name.clone());
                foreign_key_dependencies.insert(candidate.name.clone());
            }
        }
        self.catalog_dependencies = names
            .into_iter()
            .map(|name| {
                catalog
                    .relational_catalog
                    .get(&name)
                    .cloned()
                    .map(|relation| (name, relation))
                    .ok_or_else(|| {
                        ExecuteError::Serialization(
                            "typed transaction INSERT FK dependency left the statement catalog"
                                .to_string(),
                        )
                    })
            })
            .collect::<Result<_, _>>()?;
        self.foreign_key_dependencies = foreign_key_dependencies;
        Ok(())
    }

    /// Validate the private columnar source before it becomes immutable transaction state.
    ///
    /// S2 and the final image are sealed from the same move-only `TypedInsertBatch` immediately
    /// before this payload is constructed. Keeping a row-string mirror solely to reparse it here
    /// would create a fourth live value representation. The payload stays private after this
    /// constructor, while later codec-5/replay boundaries strictly validate their own sealed
    /// bytes.
    fn validate_exact_fixed_payload(
        table: &RelationalTable,
        private_device_payload: &[u8],
        private_int4_stats: &[ResidentDeviceInt4ColumnStats],
        private_bool_layouts: &[ResidentDeviceBoolColumnLayout],
        private_null_layouts: &[ResidentDeviceNullBitmapLayout],
        private_text_layouts: &[ResidentDeviceTextColumnLayout],
    ) -> Result<(), &'static str> {
        let rows = usize::try_from(u64::from_le_bytes(
            private_device_payload
                .get(..std::mem::size_of::<u64>())
                .ok_or("typed transaction INSERT payload header is truncated")?
                .try_into()
                .expect("checked private payload header width"),
        ))
        .map_err(|_| "typed transaction INSERT payload row count overflows")?;
        let (expected_bytes, offsets, int4_columns) = private_fixed_payload_layout(
            table,
            rows,
            private_bool_layouts,
            private_null_layouts,
            private_text_layouts,
        )?;
        if rows == 0
            || private_device_payload.len() != expected_bytes
            || private_int4_stats.len() != int4_columns
            || private_device_payload[..std::mem::size_of::<u64>()]
                != u64::try_from(rows)
                    .expect("typed transaction rows fit u64")
                    .to_le_bytes()
        {
            return Err("typed transaction INSERT payload geometry drifted");
        }
        if !private_null_layouts_are_canonical(private_device_payload, rows, private_null_layouts) {
            return Err("typed transaction INSERT NULL bitmap payload drifted");
        }
        if !private_bool_layouts_are_canonical(private_device_payload, rows, private_bool_layouts) {
            return Err("typed transaction INSERT BOOL bitmap payload drifted");
        }
        for layout in private_text_layouts {
            let offsets_base = usize::try_from(layout.offsets_byte_offset)
                .map_err(|_| "typed transaction INSERT text offsets overflow")?;
            let bytes_base = usize::try_from(layout.bytes_byte_offset)
                .map_err(|_| "typed transaction INSERT text bytes overflow")?;
            let bytes_len = usize::try_from(layout.bytes_len)
                .map_err(|_| "typed transaction INSERT text bytes length overflow")?;
            let offsets_len = rows
                .checked_add(1)
                .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
                .ok_or("typed transaction INSERT text offsets size overflow")?;
            let offsets = private_device_payload
                .get(offsets_base..offsets_base.saturating_add(offsets_len))
                .ok_or("typed transaction INSERT text offsets exceed payload")?;
            let text_bytes = private_device_payload
                .get(bytes_base..bytes_base.saturating_add(bytes_len))
                .ok_or("typed transaction INSERT text bytes exceed payload")?;
            let mut previous = 0_u64;
            for index in 0..=rows {
                let start = index * std::mem::size_of::<u64>();
                let value = u64::from_le_bytes(
                    offsets[start..start + std::mem::size_of::<u64>()]
                        .try_into()
                        .expect("checked text offsets geometry"),
                );
                if value < previous || value > layout.bytes_len || (index == 0 && value != 0) {
                    return Err("typed transaction INSERT text offsets drifted");
                }
                previous = value;
            }
            if previous != layout.bytes_len || std::str::from_utf8(text_bytes).is_err() {
                return Err("typed transaction INSERT text bytes drifted");
            }
            for row in 0..rows {
                let start = usize::try_from(u64::from_le_bytes(
                    offsets
                        [row * std::mem::size_of::<u64>()..(row + 1) * std::mem::size_of::<u64>()]
                        .try_into()
                        .expect("checked text start geometry"),
                ))
                .map_err(|_| "typed transaction INSERT text row start overflow")?;
                let end = usize::try_from(u64::from_le_bytes(
                    offsets[(row + 1) * std::mem::size_of::<u64>()
                        ..(row + 2) * std::mem::size_of::<u64>()]
                        .try_into()
                        .expect("checked text end geometry"),
                ))
                .map_err(|_| "typed transaction INSERT text row end overflow")?;
                if std::str::from_utf8(
                    text_bytes
                        .get(start..end)
                        .ok_or("typed transaction INSERT text row bounds drifted")?,
                )
                .is_err()
                {
                    return Err("typed transaction INSERT text row bytes drifted");
                }
            }
        }

        let mut stats_index = 0;
        for (column_index, column) in table.columns.iter().enumerate() {
            if !matches!(column.ty, SqlType::Int2 | SqlType::Int4 | SqlType::Date) {
                continue;
            }
            let stats = &private_int4_stats[stats_index];
            stats_index += 1;
            if stats.name != column.name {
                return Err("typed transaction INSERT payload column binding drifted");
            }
            let mut min = i32::MAX;
            let mut max = i32::MIN;
            for row in 0..rows {
                if !private_row_is_valid(
                    private_device_payload,
                    private_null_layouts,
                    &column.name,
                    row,
                ) {
                    continue;
                }
                let offset = offsets[column_index] + row * std::mem::size_of::<i32>();
                let value = i32::from_le_bytes(
                    private_device_payload[offset..offset + std::mem::size_of::<i32>()]
                        .try_into()
                        .expect("checked int4 payload geometry"),
                );
                min = min.min(value);
                max = max.max(value);
            }
            if stats.min != min || stats.max != max {
                return Err("typed transaction INSERT payload statistics drifted");
            }
        }

        Ok(())
    }

    pub(crate) fn validate_private_payload(
        &self,
        table: &RelationalTable,
    ) -> Result<(), ExecuteError> {
        if self.table != table.name
            || self.table_oid != table.oid
            || self.typed_statement_digest == [0; 32]
            || self.table_schema_digest
                != crate::engine_transaction_reset::table_schema_digest(table)?
            || !table
                .columns
                .iter()
                .all(|column| private_stage_type(column.ty))
            || self.private_payload_digest
                != gpu_db_wal::canonical_request_digest(&self.private_device_payload)
        {
            return Err(ExecuteError::Serialization(
                "typed transaction INSERT artifact drifted before private GPU generation"
                    .to_string(),
            ));
        }
        let expected_rows = self.provisional_row_ids.len();
        let expected_bytes = private_fixed_payload_layout(
            table,
            expected_rows,
            self.private_bool_layouts.as_ref(),
            self.private_null_layouts.as_ref(),
            self.private_text_layouts.as_ref(),
        )
        .map_err(|reason| ExecuteError::Unsupported(reason.to_string()))?
        .0;
        if expected_rows == 0
            || self.private_device_payload.len() != expected_bytes
            || self.private_device_payload[..std::mem::size_of::<u64>()]
                != u64::try_from(expected_rows)
                    .expect("typed transaction rows fit in u64")
                    .to_le_bytes()
        {
            return Err(ExecuteError::Serialization(
                "typed transaction INSERT artifact lost exact private payload geometry".to_string(),
            ));
        }
        Self::validate_exact_fixed_payload(
            table,
            self.private_device_payload.as_ref(),
            self.private_int4_stats.as_ref(),
            self.private_bool_layouts.as_ref(),
            self.private_null_layouts.as_ref(),
            self.private_text_layouts.as_ref(),
        )
        .map_err(|reason| ExecuteError::Serialization(reason.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insert_semantic_ir::InsertStatementOrdinal;
    use crate::typed_insert_batch::{
        prepare_typed_insert_semantics_at, sequence_defaults::SequenceDefaultBindings,
    };
    use crate::{parse_command, Command, Engine};

    #[test]
    fn typed_private_artifact_derives_transient_rows_without_a_row_string_mirror() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog["accounts"].clone();
        let Command::Insert(insert) =
            parse_command("INSERT INTO accounts VALUES (1, 10), (2, 20)").unwrap()
        else {
            unreachable!("INSERT test SQL must parse")
        };
        let prepared = prepare_typed_insert_semantics_at(
            &insert,
            &catalog,
            catalog.commit_seq,
            None,
            InsertStatementOrdinal::from_u32(0),
        )
        .unwrap()
        .expect("default-free int4 INSERT prepares");
        let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
        let staged = batch
            .into_transaction_private_fixed_stage(
                &table,
                gpu_db_wal::canonical_request_digest(b"typed-private-artifact-test"),
                catalog.commit_seq,
                1,
            )
            .unwrap();
        assert_eq!(
            staged.private_row_values(&table, 1).unwrap(),
            vec![SqlValue::Int4(2), SqlValue::Int4(20)]
        );
    }

    #[test]
    fn null_free_int4_stages_through_the_generic_private_overlay() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE accounts (id int4, balance int4)")
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog["accounts"].clone();
        let Command::Insert(insert) =
            parse_command("INSERT INTO accounts VALUES (1, 10), (2, 20)").unwrap()
        else {
            unreachable!("INSERT test SQL must parse")
        };
        let prepared = prepare_typed_insert_semantics_at(
            &insert,
            &catalog,
            catalog.commit_seq,
            None,
            InsertStatementOrdinal::from_u32(0),
        )
        .unwrap()
        .expect("default-free int4 INSERT prepares");
        let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
        let staged = batch
            .into_transaction_private_fixed_stage(
                &table,
                gpu_db_wal::canonical_request_digest(b"typed-private-artifact-test"),
                catalog.commit_seq,
                1,
            )
            .unwrap();

        assert_eq!(
            staged.write_set.tables,
            BTreeSet::from(["accounts".to_string()])
        );
        assert_eq!(staged.write_set.table_oids, vec![table.oid]);
        assert!(
            staged.write_set.rows.is_empty() && staged.write_set.stable_rows.is_empty(),
            "fresh typed INSERT identities are allocator output, not legacy row-key conflict slots"
        );
        assert!(staged.write_set.unique_slots.is_empty());
        assert!(staged.write_set.unique_slots_i32.is_empty());
        #[cfg(feature = "probe-timing")]
        assert_eq!(
            staged.private_probe_nanos()[1],
            0,
            "an unindexed typed stage must not reconstruct rows solely for UNIQUE slots"
        );
        assert_eq!(staged.rows_consumed(), 2);
        assert!(!staged.codec5_sources.record().is_empty());
        assert!(!staged.codec5_sources.final_image().is_empty());
    }

    #[test]
    fn check_constrained_int4_retains_the_generic_private_overlay() {
        let engine = Engine::new_local();
        engine
            .execute_text(
                1,
                "CREATE TABLE accounts (id int4, balance int4, CHECK (balance > 0))",
            )
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog["accounts"].clone();
        let Command::Insert(insert) =
            parse_command("INSERT INTO accounts VALUES (1, 10), (2, 20)").unwrap()
        else {
            unreachable!("INSERT test SQL must parse")
        };
        let batch = prepare_typed_insert_semantics_at(
            &insert,
            &catalog,
            catalog.commit_seq,
            None,
            InsertStatementOrdinal::from_u32(0),
        )
        .unwrap()
        .expect("checked int4 INSERT prepares")
        .seal(SequenceDefaultBindings::empty())
        .unwrap();

        assert!(batch.supports_transaction_private_fixed_stage(&table));
    }

    #[test]
    fn typed_private_fixed_artifact_rejects_mixed_width_payload_drift() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE fixed (id int4, tally int8)")
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog["fixed"].clone();
        let Command::Insert(insert) =
            parse_command("INSERT INTO fixed VALUES (1, 7000000000), (2, 8000000000)").unwrap()
        else {
            unreachable!("INSERT test SQL must parse")
        };
        let prepared = prepare_typed_insert_semantics_at(
            &insert,
            &catalog,
            catalog.commit_seq,
            None,
            InsertStatementOrdinal::from_u32(0),
        )
        .unwrap()
        .expect("fixed-width INSERT prepares");
        let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
        let mut staged = batch
            .into_transaction_private_fixed_stage(
                &table,
                gpu_db_wal::canonical_request_digest(b"typed-private-fixed-artifact-test"),
                catalog.commit_seq,
                1,
            )
            .unwrap();
        let mut payload = staged.private_device_payload.as_ref().to_vec();
        // Header + two int4 values starts the int8 section. A private payload cannot be replaced
        // through the production API; this forged test mutation must fail its sealed digest.
        payload[std::mem::size_of::<u64>() + 2 * std::mem::size_of::<i32>()] ^= 1;
        staged.private_device_payload = Arc::from(payload);

        let error = staged.validate_private_payload(&table).unwrap_err();
        assert!(
            matches!(error, ExecuteError::Serialization(ref message) if message.contains("artifact drifted")),
            "{error:?}"
        );
    }

    #[test]
    fn typed_private_fixed_artifact_rejects_b128_payload_drift() {
        let engine = Engine::new_local();
        engine
            .execute_text(
                1,
                "CREATE TABLE fixed_all (small int2, day date, observed_at timestamp, amount numeric(12,2), ident uuid)",
            )
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog["fixed_all"].clone();
        let Command::Insert(insert) = parse_command(
            "INSERT INTO fixed_all VALUES \
             (-7, '2024-01-02', '2024-01-02 03:04:05.678901', 12.34, \
              '550e8400-e29b-41d4-a716-446655440000')",
        )
        .unwrap() else {
            unreachable!("INSERT test SQL must parse")
        };
        let prepared = prepare_typed_insert_semantics_at(
            &insert,
            &catalog,
            catalog.commit_seq,
            None,
            InsertStatementOrdinal::from_u32(0),
        )
        .unwrap()
        .expect("fixed-width INSERT prepares");
        let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
        let mut staged = batch
            .into_transaction_private_fixed_stage(
                &table,
                gpu_db_wal::canonical_request_digest(b"typed-private-b128-artifact-test"),
                catalog.commit_seq,
                1,
            )
            .unwrap();
        let mut payload = staged.private_device_payload.as_ref().to_vec();
        // Header + two i32 vectors + one i64 vector reaches the numeric b128 section.
        payload[std::mem::size_of::<u64>()
            + 2 * std::mem::size_of::<i32>()
            + std::mem::size_of::<i64>()] ^= 1;
        staged.private_device_payload = Arc::from(payload);

        let error = staged.validate_private_payload(&table).unwrap_err();
        assert!(
            matches!(error, ExecuteError::Serialization(ref message) if message.contains("artifact drifted")),
            "{error:?}"
        );
    }

    #[test]
    fn typed_private_fixed_artifact_rejects_validity_bitmap_drift() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE nullable_fixed (id int4, tally int8)")
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog["nullable_fixed"].clone();
        let Command::Insert(insert) =
            parse_command("INSERT INTO nullable_fixed VALUES (1, NULL), (NULL, 2)").unwrap()
        else {
            unreachable!("INSERT test SQL must parse")
        };
        let prepared = prepare_typed_insert_semantics_at(
            &insert,
            &catalog,
            catalog.commit_seq,
            None,
            InsertStatementOrdinal::from_u32(0),
        )
        .unwrap()
        .expect("nullable fixed-width INSERT prepares");
        let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
        let mut staged = batch
            .into_transaction_private_fixed_stage(
                &table,
                gpu_db_wal::canonical_request_digest(b"typed-private-validity-artifact-test"),
                catalog.commit_seq,
                1,
            )
            .unwrap();
        let mut payload = staged.private_device_payload.as_ref().to_vec();
        let bitmap_offset = usize::try_from(staged.private_null_layouts[0].bitmap_byte_offset)
            .expect("test bitmap offset fits usize");
        payload[bitmap_offset] ^= 1;
        staged.private_device_payload = Arc::from(payload);
        staged.private_payload_digest =
            gpu_db_wal::canonical_request_digest(&staged.private_device_payload);

        let error = staged.validate_private_payload(&table).unwrap_err();
        assert!(
            matches!(error, ExecuteError::Serialization(ref message) if message.contains("statistics drifted")),
            "{error:?}"
        );
    }

    #[test]
    fn typed_private_text_artifact_rejects_split_utf8_offset_before_publication() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE private_text (id int4, body text)")
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog["private_text"].clone();
        let Command::Insert(insert) =
            parse_command("INSERT INTO private_text VALUES (1, 'é'), (2, NULL)").unwrap()
        else {
            unreachable!("TEXT INSERT test SQL must parse")
        };
        let prepared = prepare_typed_insert_semantics_at(
            &insert,
            &catalog,
            catalog.commit_seq,
            None,
            InsertStatementOrdinal::from_u32(0),
        )
        .unwrap()
        .expect("TEXT INSERT prepares");
        let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
        let mut staged = batch
            .into_transaction_private_dense_variable_stage(
                &table,
                gpu_db_wal::canonical_request_digest(b"typed-private-text-artifact-test"),
                catalog.commit_seq,
                1,
            )
            .unwrap();
        let mut payload = staged.private_device_payload.as_ref().to_vec();
        let offsets_base = usize::try_from(staged.private_text_layouts[0].offsets_byte_offset)
            .expect("test text offsets fit usize");
        // The whole blob remains valid UTF-8, but row zero would end in the middle of `é`.
        // This must fail before the corrupt bytes reach private shard publication.
        payload[offsets_base + std::mem::size_of::<u64>()
            ..offsets_base + 2 * std::mem::size_of::<u64>()]
            .copy_from_slice(&1_u64.to_le_bytes());
        staged.private_device_payload = Arc::from(payload);
        staged.private_payload_digest =
            gpu_db_wal::canonical_request_digest(&staged.private_device_payload);

        let error = staged.validate_private_payload(&table).unwrap_err();
        assert!(
            matches!(error, ExecuteError::Serialization(ref message) if message.contains("text row bytes drifted")),
            "{error:?}"
        );
    }

    #[test]
    fn typed_private_bool_artifact_rejects_noncanonical_bitmap_tail_before_publication() {
        let engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE private_bool (id int4, enabled bool)")
            .unwrap();
        let catalog = engine.catalog_snapshot();
        let table = catalog.relational_catalog["private_bool"].clone();
        let Command::Insert(insert) =
            parse_command("INSERT INTO private_bool VALUES (1, true), (2, false), (3, NULL)")
                .unwrap()
        else {
            unreachable!("BOOL INSERT test SQL must parse")
        };
        let prepared = prepare_typed_insert_semantics_at(
            &insert,
            &catalog,
            catalog.commit_seq,
            None,
            InsertStatementOrdinal::from_u32(0),
        )
        .unwrap()
        .expect("BOOL INSERT prepares");
        let batch = prepared.seal(SequenceDefaultBindings::empty()).unwrap();
        let mut staged = batch
            .into_transaction_private_dense_variable_stage(
                &table,
                gpu_db_wal::canonical_request_digest(b"typed-private-bool-artifact-test"),
                catalog.commit_seq,
                1,
            )
            .unwrap();
        let mut payload = staged.private_device_payload.as_ref().to_vec();
        let bitmap_offset = usize::try_from(staged.private_bool_layouts[0].bitmap_byte_offset)
            .expect("test BOOL bitmap offset fits usize");
        // The first three rows are valid, but packed BOOL vectors must never carry state in
        // their unused tail bits. This is checked before a descriptor can reach GPU publication.
        payload[bitmap_offset + 3] |= 1 << 7;
        staged.private_device_payload = Arc::from(payload);
        staged.private_payload_digest =
            gpu_db_wal::canonical_request_digest(&staged.private_device_payload);

        let error = staged.validate_private_payload(&table).unwrap_err();
        assert!(
            matches!(error, ExecuteError::Serialization(ref message) if message.contains("BOOL bitmap payload drifted")),
            "{error:?}"
        );
    }
}
