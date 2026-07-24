//! Sharded unified-source layout and device-to-device recompaction. Query routing and point-result
//! framing remain with their execution owners; this leaf owns only source construction.

use super::execution_source::{ResidentExecSource, ResidentVisibility, ShardedUnifiedExecSource};
use super::shard_pruning::{mandatory_int4_equalities, shard_zone_map_excludes};
use crate::engine_expr_ir::ResidentExpr;
use crate::relational_model::RelationalTable;
use crate::resident_storage::RelationalResidentShard;
use crate::{Engine, ExecuteError};
use gpu_db_execution::{CudaResidentDeviceMemory, CudaSidecarSource, CudaTextOffsetSource};
use gpu_db_types::{EngineError, Index};
use std::sync::Arc;

/// One shard-to-unified TEXT offset rebase operation: destination offsets/base row/blob base,
/// source allocation/offsets/blob/length, and row count.
type TextRebaseOp = (u64, u32, u64, u64, u64, u64, u64, u32);

impl Engine {
    /// SLICE B: build the SHARDED UNIFIED EXEC SOURCE — recompact a shard-resident table's shards into ONE
    /// unified device buffer (8-byte header + int4 columns in catalog order + SV3b/SV6 version columns +
    /// M3 null-validity bitmaps) and return it as an injectable [`ResidentExecSource`] plus its MVCC
    /// [`ResidentVisibility`], so ANY general-executor caller runs the SAME on-device execution over
    /// sharded tables: the sharded shape BRIDGE below AND the SQL->Expr PG path (which serves `IS NULL` /
    /// `IS NOT NULL` and every other general shape — previously those ERRORED on a sharded-only table
    /// because the PG path only knew the single-buffer store). `predicate` drives the S-d3 zone-map prune
    /// (mandatory int4 equalities only; `None` / non-equality predicates gather every shard — sound, the
    /// device predicate filters). The unified SoA is laid out exactly as a whole-table single store, so
    /// the single-store offset helpers address it byte-identically; per-shard identity/validity prechecks
    /// mirror the retired probes. Extracted VERBATIM from `execute_resident_sharded_via_general` (which
    /// now calls it) — the recompaction logic exists ONCE.
    pub(crate) fn build_sharded_unified_exec_source(
        &self,
        table: &RelationalTable,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
    ) -> Result<ShardedUnifiedExecSource, ExecuteError> {
        // Load the table's resident shards in published order (sorted by (row_start, shard_id)).
        // Error text mirrors the retired probes.
        let shards = self
            .read_residency_shards()
            .get(&table.name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident shards",
                    table.name
                )))
            })?;
        self.build_sharded_unified_exec_source_from_shards(table, predicate, copin_s, shards)
    }

    /// Exact-source variant used by pre-WAL DDL validation after cold chunks have been staged as
    /// private immutable shards. It shares the complete D2D recompaction, NULL, text-offset, and
    /// MVCC visibility implementation with ordinary reads; callers cannot introduce a second
    /// relational execution path by assembling a parallel validator.
    pub(crate) fn build_sharded_unified_exec_source_from_shards(
        &self,
        table: &RelationalTable,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        shards: Vec<RelationalResidentShard>,
    ) -> Result<ShardedUnifiedExecSource, ExecuteError> {
        self.build_sharded_unified_exec_source_from_shards_inner(
            table, predicate, copin_s, shards, false,
        )
    }

    /// Transactional UNIQUE-validation form: source shard ownership is already explicitly scoped,
    /// and this routes the D2D unified allocation through the same scope before `cuMemAlloc`.
    pub(crate) fn build_scoped_sharded_unified_exec_source_from_shards(
        &self,
        table: &RelationalTable,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        shards: Vec<RelationalResidentShard>,
    ) -> Result<ShardedUnifiedExecSource, ExecuteError> {
        self.build_sharded_unified_exec_source_from_shards_inner(
            table, predicate, copin_s, shards, true,
        )
    }

    fn build_sharded_unified_exec_source_from_shards_inner(
        &self,
        table: &RelationalTable,
        predicate: Option<&ResidentExpr>,
        copin_s: Index,
        mut shards: Vec<RelationalResidentShard>,
        scoped_allocation: bool,
    ) -> Result<ShardedUnifiedExecSource, ExecuteError> {
        if shards.is_empty() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no resident shards",
                table.name
            ))));
        }
        // CREATE TABLE publishes a zero-row bootstrap generation so the first write never needs a
        // host authority hand-off. Once a non-empty shard exists, that descriptor contributes no
        // rows, visibility, or payload and must not force the general multi-shard recompactor: in
        // particular, doing so can replace a correctly padded fixed-width source with a synthetic
        // mixed-width layout. Retain one empty descriptor only for a genuinely empty table.
        if shards.iter().any(|shard| shard.row_count > 0) {
            shards.retain(|shard| shard.row_count > 0);
        }
        // S-d3 zone-map pruning: for a point-lookup shape (`col = needle`, ANDed at top level) drop every
        // shard whose min/max zone map for that column excludes the needle — it cannot hold a matching row,
        // so the recompaction never gathers it. This turns the sharded read from O(num_shards) toward O(1).
        // SOUNDNESS: prune ONLY on a shard that carries a zone-map stat for the column AND provably excludes
        // the needle; a shard with no stat (e.g. a benchmark install) or an unresolved column is always kept.
        // The constraint's `col` is a FULL-CATALOG column index (from `ResidentExpr::Column`, built via
        // `relational_column_index`), so we resolve it to the column's real NAME through `table.columns` and
        // match the zone-map stat by name — the SAME catalog->int4-ordinal translation the resident read
        // offsets perform (`resident_device_int4_column_offset`). Indexing the int4-ordinal-compacted stat
        // list by the raw catalog index would read the WRONG column on a mixed-type table (non-int4 column
        // before the filter column) and could wrongly prune a matching shard. If pruning would drop EVERY
        // shard (needle in no range), keep the first shard so the recompaction machinery stays well-formed
        // and the device predicate returns the correct empty set.
        if let Some(pred) = predicate {
            let mut constraints: Vec<(usize, i32)> = Vec::new();
            mandatory_int4_equalities(pred, &mut constraints);
            if !constraints.is_empty() {
                // Catalog column names in catalog order — the constraints' `col` indexes into THIS.
                let column_names: Vec<&str> =
                    table.columns.iter().map(|c| c.name.as_str()).collect();
                let kept: Vec<RelationalResidentShard> = shards
                    .iter()
                    .filter(|shard| {
                        // Keep the shard unless SOME mandatory equality's zone map provably excludes it.
                        !constraints.iter().any(|(col, needle)| {
                            shard_zone_map_excludes(
                                &column_names,
                                &shard.resident_device_int4_column_stats,
                                *col,
                                *needle,
                            )
                        })
                    })
                    .cloned()
                    .collect();
                shards = if kept.is_empty() {
                    vec![shards[0].clone()]
                } else {
                    kept
                };
            }
        }
        self.read_state
            .residency
            .sharded_shards_gathered
            .fetch_add(shards.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let gpu_id = shards[0].gpu_id;
        let runtime_snapshot = self.router.runtime().snapshot();

        // Validate each shard + resolve its pinned device memory (the per-shard identity + validity
        // prechecks mirror the probe's at engine_resident_probe.rs ~873), or return the error a probe did.
        let source_for =
            |shard: &RelationalResidentShard| -> Result<ResidentExecSource, ExecuteError> {
                if shard.schema != table.schema || shard.table != table.name {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident shard no longer matches catalog table identity".to_string(),
                    )));
                }
                let memory_pressure_active = runtime_snapshot
                    .memory_pressured_gpu_ids
                    .contains(&shard.gpu_id);
                if !shard.is_valid(memory_pressure_active) {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident shard {} is invalid",
                        shard.shard_id
                    ))));
                }
                // D4 (ADR-013 pre2): the buffer rides the loaded descriptor — the SAME generation
                // as the metadata by construction (no second map load to pair a stale descriptor
                // with a republished buffer).
                let device_memory = shard.device_memory.clone().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident shard {} has no retained device memory",
                        shard.shard_id
                    )))
                })?;
                Ok(ResidentExecSource {
                    descriptor: Arc::new(self.resident_snapshot_for_shard(shard, table)),
                    device_memory,
                    row_count: shard.row_count as u64,
                })
            };

        // FLIP slice — ZERO-COPY single-shard fast path (measured: the per-read recompaction DtoD copy
        // costs ~440-460us at 524k rows vs the single-buffer's 19us COUNT — ledger #4). When the
        // zone-map prune leaves EXACTLY ONE shard and that shard is VERSION-FREE (no deleted_by /
        // created_by region — a version region lives in a SEPARATE device buffer, so it cannot be
        // addressed by an absolute offset inside the shard's own buffer), serve the shard's OWN buffer
        // directly: `resident_snapshot_for_shard` already describes it (capacity-strided, its own null
        // bitmaps), and every consumer reads through the descriptor's offset helpers. No DtoD, no
        // allocation. A versioned or multi-shard survivor set takes the recompaction below unchanged.
        if shards.len() == 1 {
            let shard = &shards[0];
            // D4: the version-free check reads the SAME loaded descriptor the buffer came from — a
            // concurrent re-admit purging the side maps can no longer fake version-freeness for a
            // reader still holding the old (tombstone-bearing) generation (the resurrection race).
            // D3 hwm gate: a created_by-only shard whose stamps are ALL <= the reader's boundary
            // (s >= max_created_by) is EFFECTIVELY VERSION-FREE for this reader — every row is
            // born-visible at s, so serving the raw buffer is exact. Only a reader pinned inside
            // an append window (s < hwm) falls through to the gated recompaction.
            let version_free = shard.deleted_by_region.is_none()
                && (shard.created_by_region.is_none() || copin_s >= shard.max_created_by);
            if version_free {
                let src = source_for(shard)?;
                return Ok(ShardedUnifiedExecSource {
                    src,
                    visibility: None,
                    gpu_id,
                });
            }
        }

        // The unified int4-only buffer lays the table's int4 columns out in catalog order, each contiguous
        // over `total_row_count` rows after the 8-byte row-count header — exactly the whole-table single
        // store the offset helpers expect. The recompaction indexes source slices POSITIONALLY by int4
        // ordinal and the unified descriptor labels the buffer with this list, so every shard MUST
        // carry the SAME `resident_device_int4_columns` (same names, same order) as shard 0 — otherwise
        // a slice would land in the wrong column's slot (silent wrong data) or read past a too-short source
        // (the DtoD primitive bounds-checks only the destination). The sharded benchmark install runs no
        // layout validation, so we enforce uniformity per shard in the gather loop below (audit F1).
        let int4_columns = shards[0].resident_device_int4_columns.clone();
        let num_int4_cols = int4_columns.len();
        // TYPE-COVERAGE track 2 slice 2: the i64 SECTION (Int8/Timestamp) recompacts exactly like
        // the i32 sections — same builder layout (header + all i32 sections + all i64 sections,
        // capacity-strided, NO padding: the offset helper + the 2x-u32 load discipline own the
        // 4-mod-8 case), same per-shard uniformity guard, same positional-ordinal DtoD plan.
        let int8_columns = shards[0].resident_device_int8_columns.clone();
        let num_int8_cols = int8_columns.len();
        // TYPE-COVERAGE #14 (numeric): the b128 (Numeric/Uuid) 16-byte section, recompacted after
        // the i64 sections into the unified buffer (else a numeric column read on a MULTI-SHARD
        // table would miss its bytes — a correctness hole, not an optimization).
        let numeric_columns = shards[0].resident_device_numeric_columns.clone();
        let num_numeric_cols = numeric_columns.len();

        // Gather each shard's (device_ptr, row_count) by running the SAME identity/validity precheck the
        // probes did (via `source_for`), accumulate `total_row_count`, and build the device-to-device copy
        // plan: for each int4 column ordinal `c` and each shard `p`, copy `p`'s slice of column `c`
        // (`8 + c*p.row_count*4`, len `p.row_count*4`) into the unified slot
        // (`8 + c*total_row_count*4 + rows_before_p*4`). Empty shards contribute a zero-length segment
        // (skipped by the primitive). The host never touches the column bytes.
        let mut shard_ptrs: Vec<(u64, usize, usize)> = Vec::with_capacity(shards.len());
        let mut total_row_count = 0_usize;
        for shard in &shards {
            let src = source_for(shard)?;
            // Uniformity guard (audit F1): the positional ordinal gather + the unified descriptor both
            // assume every shard's int4 layout equals shard 0's. Reject a mismatch with a clean
            // error rather than recompact a slice into the wrong column (or read past a short source).
            if shard.resident_device_int4_columns != int4_columns {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident shard {} int4 column layout {:?} does not match shard 0 layout {:?}",
                    shard.shard_id, shard.resident_device_int4_columns, int4_columns
                ))));
            }
            if shard.resident_device_int8_columns != int8_columns {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident shard {} int8 column layout {:?} does not match shard 0 layout {:?}",
                    shard.shard_id, shard.resident_device_int8_columns, int8_columns
                ))));
            }
            if shard.resident_device_numeric_columns != numeric_columns {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident shard {} b128 column layout {:?} does not match shard 0 layout {:?}",
                    shard.shard_id, shard.resident_device_numeric_columns, numeric_columns
                ))));
            }
            shard_ptrs.push((
                src.device_memory.device_ptr(),
                shard.row_count,
                shard.capacity,
            ));
            total_row_count = total_row_count.saturating_add(shard.row_count);
        }

        let mut segments: Vec<gpu_db_execution::RecompactSegment> =
            Vec::with_capacity(num_int4_cols * shards.len());
        for ordinal in 0..num_int4_cols {
            let mut rows_before = 0_u64;
            for (device_ptr, row_count, capacity) in &shard_ptrs {
                let row_count = *row_count as u64;
                let capacity = *capacity as u64;
                let byte_len = row_count.saturating_mul(4);
                segments.push(gpu_db_execution::RecompactSegment {
                    src_device_ptr: *device_ptr,
                    // S-d2: a column's live rows sit at its CAPACITY-strided start in the (possibly padded)
                    // shard (`8 + ordinal*capacity*4`); copy only the `row_count` live rows into the dense
                    // unified buffer. capacity == row_count for a dense/sealed shard (unchanged there).
                    src_byte_offset: 8 + (ordinal as u64) * capacity * 4,
                    dst_byte_offset: 8
                        + (ordinal as u64) * (total_row_count as u64) * 4
                        + rows_before * 4,
                    byte_len,
                });
                rows_before = rows_before.saturating_add(row_count);
            }
        }
        let int4_bytes = 8 + (total_row_count as u64) * 4 * (num_int4_cols as u64);
        // The i64 sections sit immediately after the i32 sections in BOTH the shard payloads and
        // the unified buffer (the shared offset-helper formula; per-shard stride = capacity,
        // unified stride = total_row_count).
        for ordinal in 0..num_int8_cols {
            let mut rows_before = 0_u64;
            for (device_ptr, row_count, capacity) in &shard_ptrs {
                let row_count = *row_count as u64;
                let capacity = *capacity as u64;
                segments.push(gpu_db_execution::RecompactSegment {
                    src_device_ptr: *device_ptr,
                    src_byte_offset: 8
                        + (num_int4_cols as u64) * capacity * 4
                        + (ordinal as u64) * capacity * 8,
                    dst_byte_offset: int4_bytes
                        + (ordinal as u64) * (total_row_count as u64) * 8
                        + rows_before * 8,
                    byte_len: row_count.saturating_mul(8),
                });
                rows_before = rows_before.saturating_add(row_count);
            }
        }
        let int8_end_bytes = int4_bytes + (total_row_count as u64) * 8 * (num_int8_cols as u64);
        // TYPE-COVERAGE #14 (numeric): the b128 sections sit immediately after the i64 sections in
        // BOTH the shard payloads (per-shard stride = capacity*16) and the unified buffer (stride =
        // total_row_count*16). The per-shard section base = 8 + num_int4*cap*4 + num_int8*cap*8.
        for ordinal in 0..num_numeric_cols {
            let mut rows_before = 0_u64;
            for (device_ptr, row_count, capacity) in &shard_ptrs {
                let row_count = *row_count as u64;
                let capacity = *capacity as u64;
                segments.push(gpu_db_execution::RecompactSegment {
                    src_device_ptr: *device_ptr,
                    src_byte_offset: 8
                        + (num_int4_cols as u64) * capacity * 4
                        + (num_int8_cols as u64) * capacity * 8
                        + (ordinal as u64) * capacity * 16,
                    dst_byte_offset: int8_end_bytes
                        + (ordinal as u64) * (total_row_count as u64) * 16
                        + rows_before * 16,
                    byte_len: row_count.saturating_mul(16),
                });
                rows_before = rows_before.saturating_add(row_count);
            }
        }
        let fixed_section_bytes =
            int8_end_bytes + (total_row_count as u64) * 16 * (num_numeric_cols as u64);

        // SV3b/SV6 MVCC visibility gather: if ANY surviving shard is VERSIONED (carries an on-demand
        // `deleted_by` and/or `created_by` region), append the corresponding co-resident DENSE i64 column(s)
        // to the unified buffer right after the int4 columns. The predicate then ANDs
        // `deleted_by > read_txn_id` (hide tombstoned rows) and/or `created_by <= read_txn_id` (hide
        // versions appended by a commit newer than the read snapshot — the SV5 double-read flip-gate). The
        // int4 descriptor/offsets are untouched (the executor reads the version columns by ABSOLUTE byte
        // offset via LoadColumnI64). The un-versioned majority allocates zero extra bytes and takes the
        // `None` visibility path -- byte-identical to the pre-SV3b read.
        let mut fills: Vec<gpu_db_execution::RecompactFill> = Vec::new();
        let mut deleted_by_offset: Option<u64> = None;
        let mut created_by_offset: Option<u64> = None;
        let mut allocated_bytes = fixed_section_bytes;
        // The two version-metadata regions (`deleted_by` upper bound, SV3b; `created_by` lower bound, SV6)
        // are gathered INDEPENDENTLY — an UPDATE tombstones the old version in one shard and appends the
        // stamped new version into the OPEN shard, so a shard can carry either region without the other.
        // Each region mirrors the same fill+segment pattern: FILL the whole unified column with the
        // "visible" sentinel (covers un-versioned shards' rows + any tail), then one DtoD segment per
        // region-bearing shard overwrites its own live rows. `rows_before` walks shards in the SAME
        // published order the int4 gather used, so the version rows line up 1:1 with the int4 rows.
        // D4: regions come from the SAME loaded descriptors as the buffers/metadata — one snapshot.
        type RegionOf = fn(&RelationalResidentShard) -> Option<&Arc<CudaResidentDeviceMemory>>;
        // D3 hwm gate: the created_by AXIS is needed only when SOME surviving shard carries a
        // stamp ABOVE the reader's boundary (s < hwm — a reader pinned inside an append window).
        // When every stamp is <= s the conjunct `created_by <= s` is identically true — skipping
        // the axis is exact, keeps `visibility: None` for insert-only tables at the newest
        // boundary, and thereby keeps the reshaping (DISTINCT/GROUP BY/ORDER BY/JOIN) shapes
        // served (their guards fire on `visibility.is_some()`). The deleted_by axis has no such
        // shortcut (a tombstone hides rows at ANY later boundary).
        let created_axis_needed = shards
            .iter()
            .any(|shard| shard.created_by_region.is_some() && copin_s < shard.max_created_by);
        let region_axes: [(RegionOf, u8, &mut Option<u64>, bool); 2] = [
            (
                |shard| shard.deleted_by_region.as_ref(),
                crate::engine_residency::DELETED_BY_LIVE_FILL_BYTE,
                &mut deleted_by_offset,
                true,
            ),
            (
                |shard| shard.created_by_region.as_ref(),
                crate::engine_residency::CREATED_BY_VISIBLE_FILL_BYTE,
                &mut created_by_offset,
                created_axis_needed,
            ),
        ];
        for (region_of, fill_byte, offset_out, axis_needed) in region_axes {
            let has_region = shards.iter().any(|shard| region_of(shard).is_some());
            if !has_region || !axis_needed {
                continue;
            }
            let region_offset = allocated_bytes;
            let region_bytes = (total_row_count as u64) * 8;
            fills.push(gpu_db_execution::RecompactFill {
                byte_offset: region_offset,
                len: region_bytes,
                fill_byte,
            });
            let mut rows_before = 0_u64;
            for shard in &shards {
                let row_count = shard.row_count as u64;
                if let Some(region) = region_of(shard) {
                    segments.push(gpu_db_execution::RecompactSegment {
                        src_device_ptr: region.device_ptr(),
                        src_byte_offset: 0,
                        dst_byte_offset: region_offset + rows_before * 8,
                        byte_len: row_count * 8,
                    });
                }
                rows_before = rows_before.saturating_add(row_count);
            }
            *offset_out = Some(region_offset);
            allocated_bytes += region_bytes;
        }
        let visibility: Option<ResidentVisibility> =
            if deleted_by_offset.is_some() || created_by_offset.is_some() {
                Some(ResidentVisibility {
                    read_txn_id: copin_s as i64,
                    deleted_by_offset,
                    created_by_offset,
                })
            } else {
                None
            };

        // M3-for-shards: recompact each column's NULL VALIDITY BITMAP into the unified buffer. A column gets a
        // unified bitmap iff SOME surviving shard carries one; the region is 1 bit/row (u32 words, LSB-first,
        // 1 = valid / 0 = NULL), placed 4-aligned after the int4 (+ version) sections. The executor reads it
        // by ABSOLUTE offset (`resident_device_null_column_offset`) and materializes `SqlValue::Null` for a
        // 0 bit, so the sharded scan stops reading a NULL-stored-0 placeholder as `0`. ADR-006 (NULL coverage):
        // a null-bearing table is now MULTI-shard (a NULL insert rolls a dense shard), so the region is
        // PRE-FILLED 0xFF for shards that elide an all-valid bitmap; each null-bearing shard's bits are
        // explicitly set/cleared with the ALIGNMENT-FREE per-bit gather kernel — the same
        // strategy as the bool bitmaps, since a shard's `rows_before` is generally not 32-row aligned and a
        // byte-copy would land its bits in the wrong destination word. No null-bearing shard -> zero extra
        // bytes, byte-identical read.
        let mut unified_null_columns: Vec<crate::relational_model::ResidentDeviceNullBitmapLayout> =
            Vec::new();
        let sidecar_u32 = |value: u64, field: &str| {
            u32::try_from(value).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "sharded resident sidecar {field} exceeds the CUDA u32 geometry"
                )))
            })
        };
        let sidecar_add = |left: u64, right: u64, field: &str| {
            left.checked_add(right).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "sharded resident sidecar {field} overflow"
                )))
            })
        };
        // (dst_bitmap_offset, dst_base_row, src_device_ptr, src_bitmap_offset, count) per (column, shard).
        let mut null_gather_ops: Vec<(u64, u32, u64, u64, u32)> = Vec::new();
        {
            // Union of null-bearing column names across surviving shards, in catalog order (deterministic).
            let null_col_names: Vec<String> = table
                .columns
                .iter()
                .filter(|column| {
                    shards.iter().any(|s| {
                        s.resident_device_null_columns
                            .iter()
                            .any(|n| n.name == column.name)
                    })
                })
                .map(|column| column.name.clone())
                .collect();
            let words_per_col = (total_row_count as u64).div_ceil(32);
            let col_region_bytes = words_per_col * 4;
            for name in &null_col_names {
                let col_offset = allocated_bytes;
                // Born all-valid (0xFF): a row/shard without a bitmap for this column is non-NULL.
                fills.push(gpu_db_execution::RecompactFill {
                    byte_offset: col_offset,
                    len: col_region_bytes,
                    fill_byte: 0xFF,
                });
                let mut rows_before = 0_u64;
                for (shard, (device_ptr, row_count, _capacity)) in
                    shards.iter().zip(shard_ptrs.iter())
                {
                    let row_count = *row_count as u64;
                    if let Some(layout) = shard
                        .resident_device_null_columns
                        .iter()
                        .find(|n| &n.name == name)
                    {
                        // Alignment-free per-bit repack (like bool): the kernel places source local row `l`
                        // at unified row `rows_before + l`, writing both validity states — no 32-row-alignment
                        // requirement on `rows_before`, so a null-bearing shard at any offset is correct.
                        if row_count > 0 {
                            let dst_base = sidecar_u32(rows_before, "NULL destination base")?;
                            let count = sidecar_u32(row_count, "NULL row count")?;
                            null_gather_ops.push((
                                col_offset,
                                dst_base,
                                *device_ptr,
                                layout.bitmap_byte_offset,
                                count,
                            ));
                        }
                    }
                    rows_before = sidecar_add(rows_before, row_count, "NULL row base")?;
                }
                unified_null_columns.push(
                    crate::relational_model::ResidentDeviceNullBitmapLayout {
                        name: name.clone(),
                        bitmap_byte_offset: col_offset,
                    },
                );
                allocated_bytes = allocated_bytes.saturating_add(col_region_bytes);
            }
        }
        // TYPE-COVERAGE #14 (bool): recompact each bool column's 1-bit/row bitmap into the unified buffer.
        // Unlike the NULL bitmaps (sparse) or the fixed-width sections (byte-copyable at any boundary), a
        // bool bitmap CANNOT be byte-concatenated across shards: shards seal at ARBITRARY row counts (the
        // first dense admit seals at exactly its row count, e.g. 100), so a shard's `rows_before` is
        // generally not 32-row aligned and its bits would land in the wrong destination word. So the region
        // is defensively PRE-ZEROED here (RecompactFill 0x00); after DtoD recompaction each shard's bits are
        // repacked into place by a per-bit gather KERNEL (`gather_bool_bitmap_from_shard`) that reads source
        // bit `l` and atomically writes that state at `rows_before + l` — alignment-free. The executor reads the
        // region by ABSOLUTE offset (`resident_device_bool_column_offset`), like the NULL bitmaps.
        let mut unified_bool_columns: Vec<crate::relational_model::ResidentDeviceBoolColumnLayout> =
            Vec::new();
        // (dst_bitmap_offset, dst_base_row, src_device_ptr, src_bitmap_offset, count) per (column, shard).
        let mut bool_gather_ops: Vec<(u64, u32, u64, u64, u32)> = Vec::new();
        {
            let words_per_col = (total_row_count as u64).div_ceil(32);
            let col_region_bytes = words_per_col * 4;
            let bool_names: Vec<String> = shards[0]
                .resident_device_bool_columns
                .iter()
                .map(|b| b.name.clone())
                .collect();
            for name in &bool_names {
                let col_offset = allocated_bytes;
                fills.push(gpu_db_execution::RecompactFill {
                    byte_offset: col_offset,
                    len: col_region_bytes,
                    fill_byte: 0x00,
                });
                let mut rows_before = 0_u64;
                for (shard, (device_ptr, row_count, _capacity)) in
                    shards.iter().zip(shard_ptrs.iter())
                {
                    let row_count = *row_count as u64;
                    let Some(layout) = shard
                        .resident_device_bool_columns
                        .iter()
                        .find(|b| &b.name == name)
                    else {
                        // A shard missing a bool column its siblings carry => layout skew => decline.
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "sharded bool recompaction: shard {} lacks bool column \"{name}\"",
                            shard.shard_id
                        ))));
                    };
                    if row_count > 0 {
                        let dst_base = sidecar_u32(rows_before, "bool destination base")?;
                        let count = sidecar_u32(row_count, "bool row count")?;
                        bool_gather_ops.push((
                            col_offset,
                            dst_base,
                            *device_ptr,
                            layout.bitmap_byte_offset,
                            count,
                        ));
                    }
                    rows_before = sidecar_add(rows_before, row_count, "bool row base")?;
                }
                unified_bool_columns.push(
                    crate::relational_model::ResidentDeviceBoolColumnLayout {
                        name: name.clone(),
                        bitmap_byte_offset: col_offset,
                    },
                );
                allocated_bytes = allocated_bytes.saturating_add(col_region_bytes);
            }
        }
        // TYPE-COVERAGE #14 (text): recompact each TEXT column into the unified buffer. Text can't
        // byte-concatenate directly across shards: each shard's offsets are RELATIVE to its own bytes
        // blob. So the unified layout is, per column, an 8-aligned offsets section (total+1 i64) then the
        // concatenated bytes blob; the blobs byte-copy at a running blob_base (RecompactSegment) and a
        // per-element REBASE kernel adds each shard's blob_base to its offsets. Executor reads via the text
        // offset helper (offsets_byte_offset + bytes_byte_offset), same layout as the single buffer.
        let mut unified_text_columns: Vec<crate::relational_model::ResidentDeviceTextColumnLayout> =
            Vec::new();
        // (dst_offsets_byte_offset, dst_base_row, blob_base, src_ptr, src_offsets_byte_offset,
        //  src_bytes_byte_offset, src_blob_len, count).
        let mut text_rebase_ops: Vec<TextRebaseOp> = Vec::new();
        {
            let text_names: Vec<String> = shards[0]
                .resident_device_text_columns
                .iter()
                .map(|t| t.name.clone())
                .collect();
            for name in &text_names {
                while !allocated_bytes.is_multiple_of(8) {
                    allocated_bytes += 1;
                }
                let offsets_byte_offset = allocated_bytes;
                let offsets_bytes = (total_row_count as u64 + 1) * 8;
                allocated_bytes = allocated_bytes.saturating_add(offsets_bytes);
                let bytes_byte_offset = allocated_bytes;
                // Every offset entry IS written by the rebase (the shard ranges tile [0..total]); the
                // fill is a defensive pre-zero (a skipped/empty shard leaves no garbage gap).
                fills.push(gpu_db_execution::RecompactFill {
                    byte_offset: offsets_byte_offset,
                    len: offsets_bytes,
                    fill_byte: 0,
                });
                let mut rows_before = 0_u64;
                let mut blob_base = 0_u64;
                for (shard, (device_ptr, row_count, _capacity)) in
                    shards.iter().zip(shard_ptrs.iter())
                {
                    let row_count = *row_count as u64;
                    let Some(layout) = shard
                        .resident_device_text_columns
                        .iter()
                        .find(|t| &t.name == name)
                    else {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "sharded text recompaction: shard {} lacks text column \"{name}\"",
                            shard.shard_id
                        ))));
                    };
                    if row_count > 0 {
                        if layout.bytes_len > 0 {
                            segments.push(gpu_db_execution::RecompactSegment {
                                src_device_ptr: *device_ptr,
                                src_byte_offset: layout.bytes_byte_offset,
                                dst_byte_offset: bytes_byte_offset + blob_base,
                                byte_len: layout.bytes_len,
                            });
                        }
                        // Rebase this shard's (row_count+1) offsets into [rows_before .. +row_count].
                        let dst_base = sidecar_u32(rows_before, "text destination base")?;
                        let offset_count = sidecar_add(row_count, 1, "text offset count")?;
                        let count = sidecar_u32(offset_count, "text offset count")?;
                        text_rebase_ops.push((
                            offsets_byte_offset,
                            dst_base,
                            blob_base,
                            *device_ptr,
                            layout.offsets_byte_offset,
                            layout.bytes_byte_offset,
                            layout.bytes_len,
                            count,
                        ));
                    }
                    rows_before = sidecar_add(rows_before, row_count, "text row base")?;
                    blob_base = sidecar_add(blob_base, layout.bytes_len, "text blob base")?;
                }
                allocated_bytes = allocated_bytes.saturating_add(blob_base);
                unified_text_columns.push(
                    crate::relational_model::ResidentDeviceTextColumnLayout {
                        name: name.clone(),
                        offsets_byte_offset,
                        bytes_byte_offset,
                        bytes_len: blob_base,
                    },
                );
            }
        }
        let header = (total_row_count as u64).to_le_bytes();

        // Recompact ON-DEVICE into one unified buffer, then build the whole-table descriptor + injected
        // source the executor runs over ONCE.
        let runtime = self.cuda_driver_probe_runtime();
        let unified_mem = (if scoped_allocation {
            runtime.retain_device_memory_recompacted_scoped(
                gpu_id,
                allocated_bytes,
                &header,
                &fills,
                &segments,
            )
        } else {
            runtime.retain_device_memory_recompacted(
                gpu_id,
                allocated_bytes,
                &header,
                &fills,
                &segments,
            )
        })
        .map_err(|err| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "sharded resident recompaction into a unified device buffer failed: {err}"
            )))
        })?;
        let sidecar_source = |device_ptr: u64, byte_offset: u64| {
            shards
                .iter()
                .find_map(|shard| {
                    shard.device_memory.as_ref().and_then(|memory| {
                        (memory.device_ptr() == device_ptr).then(|| CudaSidecarSource {
                            memory: Arc::clone(memory),
                            byte_offset,
                        })
                    })
                })
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "sidecar source generation is no longer owned".to_string(),
                    ))
                })
        };
        // TYPE-COVERAGE #14 (bool): repack each shard's bool bits into the unified regions with
        // the alignment-free per-bit gather kernel (DtoD, no HtoD). Runs after the DtoD recompaction so the
        // unified buffer + shard buffers are both live; a failure declines the whole sharded read.
        for (dst_off, dst_base, src_ptr, src_off, count) in &bool_gather_ops {
            let source = sidecar_source(*src_ptr, *src_off)?;
            unified_mem
                .gather_bool_bitmap_from_shard(*dst_off, *dst_base, &source, *count)
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sharded bool bitmap gather kernel failed: {err}"
                    )))
                })?;
        }
        // ADR-006 (NULL coverage): repack each null-bearing shard's validity bits into the unified regions
        // with the alignment-free per-bit gather kernel (DtoD). The 0xFF fill covers shards that elide an
        // all-valid bitmap. A failure declines the read.
        for (dst_off, dst_base, src_ptr, src_off, count) in &null_gather_ops {
            let source = sidecar_source(*src_ptr, *src_off)?;
            unified_mem
                .gather_null_bitmap_from_shard(*dst_off, *dst_base, &source, *count)
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sharded null bitmap gather kernel failed: {err}"
                    )))
                })?;
        }
        // TYPE-COVERAGE #14 (text): rebase each shard's offsets into the unified offsets section (DtoD,
        // after the blob byte-copies above). A failure declines the whole sharded read.
        for (dst_off, dst_base, blob_base, src_ptr, src_off, src_bytes_off, src_blob_len, count) in
            &text_rebase_ops
        {
            let source = sidecar_source(*src_ptr, *src_off)?;
            let source = CudaTextOffsetSource {
                memory: source.memory,
                offsets_byte_offset: source.byte_offset,
                bytes_byte_offset: *src_bytes_off,
                bytes_len: *src_blob_len,
            };
            unified_mem
                .rebase_text_offsets_from_shard(*dst_off, *dst_base, *blob_base, &source, *count)
                .map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "sharded text offset rebase kernel failed: {err}"
                    )))
                })?;
        }
        let proof = unified_mem.metadata().clone();
        let snapshot = self.resident_snapshot_for_unified(
            table,
            crate::engine_residency::UnifiedResidentSnapshotParts {
                total_row_count,
                gpu_id,
                resident_bytes: allocated_bytes,
                proof,
                int4_columns,
                int8_columns,
                numeric_columns,
                bool_columns: unified_bool_columns,
                text_columns: unified_text_columns,
                null_columns: unified_null_columns,
            },
        );
        Ok(ShardedUnifiedExecSource {
            src: ResidentExecSource {
                descriptor: Arc::new(snapshot),
                device_memory: Arc::new(unified_mem),
                row_count: total_row_count as u64,
            },
            visibility,
            gpu_id,
        })
    }
}
