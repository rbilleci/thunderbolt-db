//! DML mutation prepare path (P0 §9.6 decomposition, behavior-preserving): a
//! focused `impl Engine` block that turns a parsed Insert/Delete/Update into a
//! prepared WriteDelta off-lock (prepare_insert / prepare_delete / prepare_update
//! against a dml_read_snapshot), plus the direct apply_insert helper. Pairs with
//! engine_write_apply (which installs the WriteDelta under the commit lock).

use super::*;
use crate::engine_expr_ir::{ResidentBinaryOp, ResidentExpr};

mod contracts;
pub(crate) use contracts::{device_structural_tuple_predicate, device_structural_tuple_predicates};
mod device_tuple;
mod unique_conflict;
#[cfg(test)]
pub(crate) use unique_conflict::RESIDENT_EXACT_KEY_FULL_SCAN_PROBES;

use contracts::device_eq_scan_literal;
pub(crate) use contracts::{
    dml_filter_groups_to_device_predicate, AppliedInsert, DmlResolvedMatch, DmlResolvedUpdate,
    InsertPrepareValidation,
};

impl Engine {
    pub(crate) fn apply_insert(
        &self,
        cat: &mut DdlCatalogState,
        insert: Insert,
        txn_id: TxnId,
    ) -> Result<Option<AppliedInsert>, EngineError> {
        self.apply_insert_with_profile(cat, insert, txn_id, None)
    }

    /// The off-lock read boundary a `prepare_*` runs against (write-half MVCC, Stage 2).
    ///
    /// Under serialization today `commit_seq` is the entry's commit `Index` and `next_row_id`
    /// is `relational_next_row_id` captured immediately before apply — so `prepare_*` reads
    /// exactly what the old direct apply read, and computes the identical row keys. When the
    /// commit lock is removed (Stage 4) this becomes a true snapshot taken at statement start.
    pub(crate) fn dml_read_snapshot(&self, commit_seq: TxnId) -> DmlReadSnapshot {
        DmlReadSnapshot {
            commit_seq,
            next_row_id: self.read_state.mvcc.current_row_id(),
        }
    }

    /// PURE preflight + encode for `INSERT` (write-half MVCC, Stage 2). Reads only from the
    /// `snapshot` (no `&mut self`, no engine mutation); runs the unique / check / FK preflight
    /// exactly as the old `apply_insert_with_profile`; encodes the new row versions and computes
    /// the write-set. The returned [`WriteDelta`] is what [`Engine::apply_delta`] installs.
    ///
    /// One deliberate refinement vs. the old in-line apply: the old code evaluated `nextval`
    /// column defaults (mutating the sequence) BEFORE the preflight, so a preflight FAILURE still
    /// advanced the sequence. Here the advance is deferred to `apply_delta`, so a prepare that
    /// fails preflight advances nothing — the Stage-4 abort-is-side-effect-free semantics. This is
    /// not observable on the live paths: `execute_text` / the COPY path run the same preflight
    /// BEFORE committing, so a constraint-violating INSERT never reaches apply in the first place.
    pub(crate) fn prepare_insert(
        &self,
        insert: &Insert,
        snapshot: DmlReadSnapshot,
        mut profile: Option<&mut RelationalCopyAdmissionProfile>,
        validation: InsertPrepareValidation,
    ) -> Result<WriteDelta, EngineError> {
        let txn_id = snapshot.commit_seq;
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): PIN ONE catalog generation and
        // BORROW the target out of it (no clone). This runs per item on the sequencer's under-lock
        // re-resolve (host-phase probe: ~2.6us/item, the 2nd-largest phase), where cloning the whole
        // RelationalTable (columns + indexes + constraints) per call was pure waste; the returned
        // WriteDelta owns only the table NAME + coerced rows, so `table` never needs to outlive this
        // pin. Reusing the ONE pin for the wave-deferred / index-validate checks below is also more
        // consistent than re-pinning (all reads see the same generation).
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(&insert.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", insert.table))
            })?;
        let column_indexes = if insert.columns.is_empty() {
            (0..table.columns.len()).collect::<Vec<_>>()
        } else {
            let mut indexes = Vec::with_capacity(insert.columns.len());
            let mut seen = BTreeSet::new();
            for column in &insert.columns {
                if !seen.insert(column) {
                    return Err(EngineError::DuplicateColumn(column.clone()));
                }
                let idx = table
                    .columns
                    .iter()
                    .position(|candidate| candidate.name == *column)
                    .ok_or_else(|| EngineError::UndefinedColumn(column.clone()))?;
                indexes.push(idx);
            }
            indexes
        };
        let row_prepare_started = Instant::now();
        let mut new_rows = Vec::with_capacity(insert.rows.len());
        // Sequence advancement scratch: keeps `prepare_insert` pure (no `&mut self`) while
        // evaluating `nextval` column defaults. Seeded lazily from the engine's sequence catalog,
        // advanced per row in source order (matching the old in-line apply), then installed by
        // `apply_delta`.
        let mut seq_state: BTreeMap<String, (i64, bool)> = BTreeMap::new();
        for row in &insert.rows {
            if row.len() != column_indexes.len() {
                return Err(EngineError::ApplyFailed(
                    "INSERT value count must match target columns".to_string(),
                ));
            }
            let mut values = vec![None; table.columns.len()];
            for (source_idx, target_idx) in column_indexes.iter().copied().enumerate() {
                let value = row[source_idx].clone();
                let expected_ty = table.columns[target_idx].ty;
                let coerced =
                    coerce_insert_value(value, expected_ty, &table.columns[target_idx].name)?;
                values[target_idx] = Some(coerced);
            }
            for (idx, value) in values.iter_mut().enumerate() {
                if value.is_none() {
                    if let Some(default) = table.columns[idx].default.clone() {
                        *value = Some(self.evaluate_column_default_pure(&default, &mut seq_state)?);
                    }
                }
            }
            if values.iter().any(Option::is_none) {
                return Err(EngineError::ApplyFailed(
                    "INSERT must provide every column without a default in the bootstrap relational subset"
                        .to_string(),
                ));
            }
            let values = values.into_iter().map(Option::unwrap).collect::<Vec<_>>();
            new_rows.push(values);
        }
        if let Some(profile) = profile.as_mut() {
            profile.row_prepare_micros += row_prepare_started.elapsed().as_micros();
        }

        // PG constraint order: not-null (23502) BEFORE unique — over the NEW rows only, O(new).
        Self::validate_primary_key_not_null(table, new_rows.iter().map(Vec::as_slice))?;
        // R3-004: INSERT constraints are decided by exact typed device probes over the current
        // generation; the host value-index and visible-row scan are not authorities.
        // This is THE measured PK'd-table collapse: `prepare_insert` is the concurrent path's
        // authoritative validation (P2 removed its duplicate preflight) AND re-runs under the
        // sequencer lock at re-resolve, so the scan cost 923 vs 102,045 sustained TPS @16w rode
        // on it twice per commit (oltp_commit_slo_benchmark, GPU_DB_BENCH_PK=1). Same eligibility
        // as the serialized write-apply Insert arm. Self-referencing providers are resolved from
        // the statement's new images before the device probe, so they need no host survivor scan.
        let has_constraints = table.indexes.iter().any(|index| index.unique)
            || !table.check_constraints.is_empty()
            || !table.foreign_keys.is_empty();
        // The under-lock re-resolve skips the redundant unique/CHECK pass on FK-free tables: exact
        // device version history plus wave-local arbitration is the commit-time guard (see
        // [`InsertPrepareValidation`] for the coverage proof).
        let device_covered = validation == InsertPrepareValidation::ReResolveDeviceCovered
            && table.foreign_keys.is_empty();
        // M1 design B (wave-time batched validation): eligible INSERTs DEFER the PK-unique check
        // to the wave sequencer (one batched device locate for the whole wave). The off-lock
        // prepare here skips it; the sequencer re-validates via `wave_batch_validate_unique`
        // (SHARED eligibility `insert_unique_wave_batchable` -> no bypass). PK not-null already
        // ran above; eligible tables have only unique indexes (no CHECK/FK), so the whole
        // constraint pass is deferred. Off the wave path (serialized apply, Full validation)
        // this is never set -> unchanged.
        // Scoped to SINGLE-ROW inserts: a multi-row insert's in-statement dup (two rows, same PK)
        // is caught by the off-lock `seen` set but NOT by a pre-wave device locate, so multi-row
        // keeps the off-lock path. The bench + common OLTP shape is single-row.
        let wave_deferred = validation == InsertPrepareValidation::WaveOffLock
            && insert.rows.len() == 1
            && self.insert_unique_wave_batchable(&catalog, table);
        // P5-2 (S-E.P5, the KEYED-CLASS lift): a CHUNK-AUTHORITATIVE keyed table validates
        // uniqueness ON-DEVICE. The reclaimed tuple store is not a relational authority, so a
        // device verdict is the sole admission decision: probe the per-chunk key indexes and
        // device-recheck each hit at this statement's snapshot. A decline fails closed before
        // durability; it never transfers validation to a host replay arm.
        // shape, any failure) DE-AUTHORITIZES so the ladder below validates against the
        // The caller reports the unavailable device verdict as an apply failure.
        if self.table_chunk_authoritative(&table.name).is_some()
            && table.indexes.iter().any(|index| index.unique)
        {
            match self.validate_class_insert_uniqueness(table, &new_rows, txn_id, None) {
                Some(verdict) => verdict?,
                None => {
                    return Err(EngineError::ApplyFailed(format!(
                        "device uniqueness verdict unavailable for cold relation \"{}\"",
                        table.name
                    )))
                }
            }
        }
        if device_covered || wave_deferred {
            // fall through to encode: PK not-null ran; uniqueness is device-history-covered or
            // deferred to the wave batch (B).
        } else {
            if has_constraints {
                let validate_started = Instant::now();
                self.validate_dml_constraints_via_device(
                    &catalog,
                    table,
                    &new_rows,
                    &[],
                    &BTreeSet::new(),
                    StorageVisibility {
                        read_txn_id: txn_id,
                    },
                )?;
                if let Some(profile) = profile.as_mut() {
                    // The index-driven pass validates all three dimensions in one call; its cost
                    // lands in the unique bucket (the first the scan path would have charged).
                    profile.unique_preflight_micros += validate_started.elapsed().as_micros();
                }
            }
        }

        // Encode the new versions against the snapshot's `next_row_id` base (pure: does not bump
        // `relational_next_row_id` — `apply_delta` advances it by `rows_consumed`). These row keys
        // are exactly what the old `apply_insert` assigned because, under the still-serialized
        // commit, the snapshot is taken immediately before apply.
        let rows_consumed = u64::try_from(new_rows.len()).map_err(|_| {
            EngineError::ApplyFailed(
                "INSERT row count exceeds relational row-id capacity".to_string(),
            )
        })?;
        snapshot
            .next_row_id
            .checked_add(rows_consumed)
            .ok_or_else(|| {
                EngineError::ApplyFailed("relational row-id allocator overflow".to_string())
            })?;
        let mut inserted_rows = Vec::with_capacity(new_rows.len());
        for (offset, values) in new_rows.into_iter().enumerate() {
            let row_id = snapshot
                .next_row_id
                .checked_add(u64::try_from(offset).expect("usize always fits u64"))
                .expect("prevalidated INSERT row-id range");
            let row_key = relational_row_key(&insert.table, row_id);
            inserted_rows.push((row_key, values));
        }
        let mut write_set = WriteSet::default();
        write_set.tables.insert(insert.table.clone());
        for (_row_key, values) in &inserted_rows {
            // An INSERT claims a FRESH, unique row id at install time (`apply_delta` reserves the
            // tuple id + advances `relational_next_row_id` under the commit lock), so its row slot
            // can never truly collide with another writer's — the predicted `row_key` here is only
            // a snapshot-relative label and is re-derived live on install. Putting it in the
            // conflict `write_set.rows` would make two concurrent disjoint inserts whose prepare
            // windows overlap (and therefore read the SAME off-lock `next_row_id`) predict the SAME
            // key and FALSELY conflict. Inserts conflict ONLY on the unique-index slots they
            // occupy (the genuine first-committer-wins point); the row slot is intentionally NOT a
            // conflict dimension for inserts.
            write_set.add_unique_slots(table, values);
        }

        let (catalog_dependencies, foreign_key_dependencies) =
            transaction_dml_dependencies(&catalog, table);
        Ok(WriteDelta {
            write_set,
            read_snapshot: snapshot.commit_seq,
            catalog_dependencies,
            foreign_key_dependencies,
            rows_consumed,
            mutation: PreparedMutation::Insert {
                table: insert.table.clone(),
                inserted_rows,
                seq_advances: seq_state,
            },
        })
    }

    /// PURE preflight + scan for `DELETE` (write-half MVCC, Stage 2). Resolves which existing
    /// versions match (against `snapshot`), runs the inbound-FK preflight as the old
    /// `apply_delete`, and records the tombstone write-set. No engine mutation.
    pub(crate) fn prepare_delete(
        &self,
        delete: &Delete,
        snapshot: DmlReadSnapshot,
    ) -> Result<WriteDelta, EngineError> {
        self.prepare_delete_inner(delete, snapshot)
    }

    fn prepare_delete_inner(
        &self,
        delete: &Delete,
        snapshot: DmlReadSnapshot,
    ) -> Result<WriteDelta, EngineError> {
        let txn_id = snapshot.commit_seq;
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): pin ONE catalog snapshot for both the
        // target bind and the inbound-FK-dependents scan below.
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(&delete.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", delete.table))
            })?;
        let filter_groups = bind_delete_filter_groups(table, delete)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let has_inbound_fks = catalog.relational_catalog.values().any(|candidate| {
            candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
        });
        // R3-004: resolve from the authoritative device generation. A cold keyed class remains a
        // device-native authority; every other table uses resident predicate scan/compaction.
        // A decline is an availability error, never permission to reconstruct or scan host tuples.
        let (deletes, class_epoch) = if self.table_chunk_authoritative(&table.name).is_some() {
            self.resolve_class_dml_matches(table, &filter_groups, visibility)
                .map(|(matches, epoch)| (matches, Some(epoch)))
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "device DML verdict unavailable for cold relation \"{}\"",
                        table.name
                    ))
                })?
        } else {
            let matches = self
                .resolve_dml_matches_via_device(table, &filter_groups, visibility)?
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "device DML verdict unavailable for relation \"{}\"",
                        table.name
                    ))
                })?;
            (matches, None)
        };

        if has_inbound_fks {
            let touched_keys: BTreeSet<String> =
                deletes.iter().map(|(_, key, _)| key.clone()).collect();
            let removed: Vec<Vec<SqlValue>> =
                deletes.iter().map(|(_, _, row)| row.clone()).collect();
            self.validate_dml_constraints_via_device(
                &catalog,
                table,
                &[],
                &removed,
                &touched_keys,
                visibility,
            )?;
        }

        let mut write_set = WriteSet::default();
        write_set.tables.insert(delete.table.clone());
        let mut tuple_ids = Vec::with_capacity(deletes.len());
        // SV4b: surface the resolved row images (catalog order) so the commit path can locate + tombstone
        // them on the resident GPU shard in place. Already decoded above for the filter/FK scan -- clone here.
        let mut deleted_rows = Vec::with_capacity(deletes.len());
        for (tuple_id, key, row) in &deletes {
            tuple_ids.push(*tuple_id);
            deleted_rows.push(row.clone());
            write_set.rows.push(RowWriteKey {
                table: delete.table.clone(),
                row_key: key.clone(),
            });
            // A delete releases the row's unique-index slots; record them as written so a
            // concurrent insert reusing the value conflicts (Stage 4 first-committer-wins).
            write_set.add_unique_slots(table, row);
        }

        let (catalog_dependencies, foreign_key_dependencies) =
            transaction_dml_dependencies(&catalog, table);
        Ok(WriteDelta {
            write_set,
            read_snapshot: snapshot.commit_seq,
            catalog_dependencies,
            foreign_key_dependencies,
            rows_consumed: 0,
            mutation: PreparedMutation::Delete {
                table: delete.table.clone(),
                tuple_ids,
                deleted_rows,
                class_epoch,
            },
        })
    }

    /// Pick the device index probe key used by transaction conflict/history bookkeeping for one
    /// Eq-predicate group. Mutation resolution itself uses the exact typed predicate scan below.
    /// Preference: a named unique index whose EVERY key column is Eq-covered, byte-matching the
    /// device-built index. Compatibility callers may finally use the first raw i32-section Eq as a
    /// scan-address hint; an engine-prepared route never may, because its proof pins named indexes
    /// only and must not manufacture an undeclared cache build from column order.
    pub(crate) fn dml_device_probe_key(
        &self,
        table: &RelationalTable,
        group: &[(usize, SelectFilterOp, SqlValue)],
    ) -> Option<(usize, i32)> {
        // Every Eq predicate in the group, by column (first occurrence wins).
        let mut eqs: Vec<(usize, &SqlValue)> = Vec::new();
        for (idx, op, value) in group {
            if *op == SelectFilterOp::Eq && !eqs.iter().any(|(existing, _)| existing == idx) {
                eqs.push((*idx, value));
            }
        }
        // Prefer a fingerprint-backed unique index whose EVERY key column is Eq-covered by a
        // foldable typed value. Fold in key-column order, byte-matching the device-built index.
        for (ord, index) in table.indexes.iter().enumerate().filter(|(_, index)| {
            index.unique && crate::engine_residency::index_uses_fingerprint(table, index)
        }) {
            let Some(positions) = crate::engine_residency::index_key_column_positions(table, index)
            else {
                continue;
            };
            let mut words: Vec<i32> = Vec::with_capacity(positions.len());
            if positions.iter().all(|p| {
                match eqs.iter().find(|(idx, _)| idx == p) {
                    Some((_, value)) => {
                        // Coerce the WHERE literal to the column type (Text -> Uuid, Numeric ->
                        // column scale) so the folded words match the stored b128 section bytes.
                        match table.columns.get(*p).and_then(|column| {
                            crate::rel_exec_helpers::coerce_insert_value(
                                (*value).clone(),
                                column.ty,
                                &column.name,
                            )
                            .ok()
                            .and_then(|v| {
                                crate::engine_residency::sql_value_key_words(column.ty, &v)
                            })
                        }) {
                            Some(column_words) => {
                                words.extend(column_words);
                                true
                            }
                            None => false,
                        }
                    }
                    None => false,
                }
            }) {
                let key_id = crate::engine_residency::index_probe_key_id(table, index, ord)?;
                return Some((
                    key_id,
                    crate::engine_residency::compound_key_fingerprint(&words),
                ));
            }
        }
        // A raw single-column named index stores its i32-section needle directly. Resolve this
        // before the compatibility fallback so `(payload INT, id INT PRIMARY KEY)` probes `id`,
        // regardless of physical column order.
        for (ord, index) in table.indexes.iter().enumerate().filter(|(_, index)| {
            index.unique && !crate::engine_residency::index_uses_fingerprint(table, index)
        }) {
            let Some(positions) = crate::engine_residency::index_key_column_positions(table, index)
            else {
                continue;
            };
            let [position] = positions.as_slice() else {
                continue;
            };
            let Some((_, value)) = eqs.iter().find(|(idx, _)| idx == position) else {
                continue;
            };
            let Some(needle) = table
                .columns
                .get(*position)
                .and_then(|column| crate::engine_residency::i32_section_needle(column.ty, value))
            else {
                continue;
            };
            let Some(key_id) = crate::engine_residency::index_probe_key_id(table, index, ord)
            else {
                continue;
            };
            return Some((key_id, needle));
        }
        if crate::engine_prepared_transaction::prepared_index_route_required() {
            return None;
        }
        // Single-column key fallback: the first i32-section Eq -> `(col_idx, raw i32 needle)` (a
        // compatibility scan hint stores the raw i32; byte-identical to the prior behavior).
        eqs.iter().find_map(|(idx, value)| {
            table
                .columns
                .get(*idx)
                .and_then(|column| crate::engine_residency::i32_section_needle(column.ty, value))
                .map(|needle| (*idx, needle))
        })
    }

    pub(crate) fn resolve_dml_matches_via_device(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        mut visibility: StorageVisibility,
    ) -> Result<Option<Vec<DmlResolvedMatch>>, EngineError> {
        if self.current_transaction_read_snapshot().is_none() {
            visibility.read_txn_id = visibility.read_txn_id.max(self.committed_seq());
        }
        self.try_resolve_dml_via_predicate_scan(table, filter_groups, visibility, None)
            .map(|resolved| resolved.map(|(matches, _)| matches))
    }

    pub(crate) fn resolve_update_matches_via_device(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        mut visibility: StorageVisibility,
        assignments: &[BoundUpdateAssignment],
    ) -> Result<Option<DmlResolvedUpdate>, EngineError> {
        if self.current_transaction_read_snapshot().is_none() {
            visibility.read_txn_id = visibility.read_txn_id.max(self.committed_seq());
        }
        self.try_resolve_dml_via_predicate_scan(table, filter_groups, visibility, Some(assignments))
    }

    /// Resolve DELETE/UPDATE matches with the exact typed predicate on every resident shard. Matching
    /// local slots are materialized from the same captured generation with SV3b/SV6 visibility applied.
    /// `Ok(None)` means the device could not provide an authoritative verdict; production callers
    /// convert that decline into a loud error rather than dispatching to host relational execution.
    fn try_resolve_dml_via_predicate_scan(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        visibility: StorageVisibility,
        assignments: Option<&[BoundUpdateAssignment]>,
    ) -> Result<Option<DmlResolvedUpdate>, EngineError> {
        // Evaluate the predicate ON-DEVICE per shard -> matching slots WITH each slot's generation-consistent
        // buffer + version regions + descriptor captured from ONE `shards.load()` (the W0-guarded detailed
        // locate), so slot + buffer + regions never straddle a concurrent re-admit (prepare runs off-lock).
        // The slots are visibility-BLIND physical positions; the materialize step applies SV3b/SV6 visibility.
        let require_index = crate::engine_prepared_transaction::prepared_index_route_required();
        let hits = if require_index {
            let [group] = filter_groups else {
                return Err(EngineError::ApplyFailed(format!(
                    "engine-prepared device index route for relation \"{}\" requires one unique-key predicate group",
                    table.name
                )));
            };
            self.prepared_exact_index_hits(table, group, visibility)?
        } else {
            // Lower the WHERE to the typed ResidentExpr DNF only for the general scan route.
            // Prepared equality with a typed NULL has an indexed empty verdict and deliberately
            // bypasses this lowering, whose scalar comparison vocabulary has no NULL needle.
            let predicate = if filter_groups.is_empty() {
                None
            } else {
                Some(
                    dml_filter_groups_to_device_predicate(table, filter_groups).ok_or_else(
                        || {
                            EngineError::ApplyFailed(format!(
                                "device DML predicate is unsupported for relation \"{}\"",
                                table.name
                            ))
                        },
                    )?,
                )
            };
            let hits = match predicate.as_ref() {
                Some(predicate) => self.locate_resident_delete_slots_detailed(table, predicate),
                None => self.locate_resident_all_slots_detailed(table),
            };
            let Some(hits) = hits else {
                return Ok(None);
            };
            hits
        };
        let mut matches: Vec<DmlResolvedMatch> = Vec::new();
        let mut old_rows: Vec<(u64, Vec<SqlValue>)> = Vec::new();
        for hit in &hits {
            // The stable entity key derives from the row-identity region at the local slot.
            let Some(region) = &hit.row_id else {
                return Ok(None); // identity-unknown lineage cannot authorize mutation
            };
            let Ok(halves) = region.read_resident_i32_column(u64::from(hit.slot) * 8, 2) else {
                return Ok(None);
            };
            let (Some(lo), Some(hi)) = (halves.first(), halves.get(1)) else {
                return Ok(None);
            };
            let row_id = (*lo as u32 as u64) | ((*hi as u32 as u64) << 32);
            if row_id == u64::MAX {
                return Ok(None); // unstamped slot: identity unknown
            }
            match self.materialize_resident_row_via_hit(table, hit, visibility.read_txn_id) {
                Some(Some(mut row)) => {
                    if let Some(assignments) = assignments {
                        old_rows.push((row_id, row.clone()));
                        self.apply_update_assignments_on_device(
                            table,
                            assignments,
                            &hit.descriptor,
                            &hit.device_memory,
                            &[hit.slot],
                            std::slice::from_mut(&mut row),
                        )?;
                    }
                    matches.push((row_id, relational_row_key(&table.name, row_id), row))
                }
                Some(None) => continue, // not visible at this snapshot (tombstoned / too-new version)
                None => return Ok(None), // materialization could not produce a device verdict
            }
        }
        // A logical row can hit in multiple shards (an SV5 update-append: tombstoned old slot + live new
        // slot), same row_id -> one match (parity with the point path's dedup).
        matches.sort_by_key(|(row_id, _, _)| *row_id);
        matches.dedup_by_key(|(row_id, _, _)| *row_id);
        old_rows.sort_by_key(|(row_id, _)| *row_id);
        old_rows.dedup_by_key(|(row_id, _)| *row_id);
        self.read_state
            .residency
            .dml_device_resolve_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some((
            matches,
            old_rows.into_iter().map(|(_, row)| row).collect(),
        )))
    }

    /// Resolve a named-index candidate set through an exact fixed-width GPU predicate and GPU MVCC
    /// visibility at those same coordinates. The host only maps device-approved coordinates back to
    /// generation-pinned hit descriptors; it never compares key values or decides visibility.
    pub(crate) fn prepared_exact_index_hits(
        &self,
        table: &RelationalTable,
        group: &[(usize, SelectFilterOp, SqlValue)],
        visibility: StorageVisibility,
    ) -> Result<Vec<crate::engine_retained_read::ShardPkHit>, EngineError> {
        let candidates = self.prepared_exact_index_candidates(table, group)?;
        let mut by_shard: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (position, hit) in candidates.iter().enumerate() {
            by_shard.entry(hit.shard_id).or_default().push(position);
        }
        let mut approved = BTreeSet::new();
        for (shard_id, positions) in by_shard {
            let first = &candidates[positions[0]];
            let mut slots = positions
                .iter()
                .map(|position| candidates[*position].slot)
                .collect::<Vec<_>>();
            for (region, cmp) in [(&first.deleted_by, 3_u32), (&first.created_by, 2_u32)] {
                let Some(region) = region else {
                    continue;
                };
                slots = region
                    .run_expr_predicate_filter_at_indices(
                        &[
                            gpu_db_execution::ExprStep::LoadColumnI64 { byte_offset: 0 },
                            gpu_db_execution::ExprStep::CompareScalarI64 {
                                cmp,
                                scalar: i64::try_from(visibility.read_txn_id).map_err(|_| {
                                    EngineError::ApplyFailed(
                                        "transaction visibility exceeds signed device sequence"
                                            .to_string(),
                                    )
                                })?,
                                scalar_on_left: false,
                            },
                        ],
                        u32::try_from(first.descriptor.row_count).map_err(|_| {
                            EngineError::ApplyFailed(
                                "resident candidate source exceeds u32 device coordinates"
                                    .to_string(),
                            )
                        })?,
                        &slots,
                        gpu_db_execution::ResidentElemType::I32,
                    )
                    .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
            }
            approved.extend(slots.into_iter().map(|slot| (shard_id, slot)));
        }
        let hits = candidates
            .into_iter()
            .filter(|hit| approved.contains(&(hit.shard_id, hit.slot)))
            .collect::<Vec<_>>();
        self.read_state
            .residency
            .dml_device_resolve_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(hits)
    }

    /// Exact-key candidates without a visibility filter, used by write-conflict history. The named
    /// index bounds the addresses and the typed GPU predicate rejects fingerprint collisions.
    pub(crate) fn prepared_exact_index_candidates(
        &self,
        table: &RelationalTable,
        group: &[(usize, SelectFilterOp, SqlValue)],
    ) -> Result<Vec<crate::engine_retained_read::ShardPkHit>, EngineError> {
        let (key_id, needle) = match crate::engine_prepared_transaction::prepared_index_probe(
            table, group,
        ) {
            Some(crate::engine_prepared_transaction::PreparedIndexProbe::Empty) => {
                return Ok(Vec::new());
            }
            Some(crate::engine_prepared_transaction::PreparedIndexProbe::Key {
                key_id,
                needle,
            }) => (key_id, needle),
            None => {
                return Err(EngineError::ApplyFailed(format!(
                    "engine-prepared device index route for relation \"{}\" has no exact named-index probe",
                    table.name
                )));
            }
        };
        let mut candidates = self
            .locate_resident_pk_via_shard_index_detailed(table, key_id, needle)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "engine-prepared device index route for relation \"{}\" declined",
                    table.name
                ))
            })?;
        // A transaction-held snapshot may safely reuse a newer superset index over the same pinned
        // source allocation. Remove coordinates beyond that snapshot's descriptor row count before
        // any typed load; this is address-range validation, while value equality and visibility
        // remain GPU verdicts below.
        candidates.retain(|hit| (hit.slot as usize) < hit.descriptor.row_count);
        let predicate = dml_filter_groups_to_device_predicate(table, &[group.to_vec()])
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "engine-prepared exact predicate for relation \"{}\" is unsupported",
                    table.name
                ))
            })?;
        let mut by_shard: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (position, hit) in candidates.iter().enumerate() {
            by_shard.entry(hit.shard_id).or_default().push(position);
        }
        let mut approved = BTreeSet::new();
        for (shard_id, positions) in by_shard {
            let first = &candidates[positions[0]];
            if positions.iter().any(|position| {
                let hit = &candidates[*position];
                hit.device_memory.device_ptr() != first.device_memory.device_ptr()
                    || hit.descriptor.row_count != first.descriptor.row_count
            }) {
                return Err(EngineError::ApplyFailed(format!(
                    "engine-prepared candidate generation for relation \"{}\" is torn",
                    table.name
                )));
            }
            let slots = positions
                .iter()
                .map(|position| candidates[*position].slot)
                .collect::<Vec<_>>();
            let slots = self
                .resident_predicate_device_filter_at_indices(
                    &predicate,
                    table,
                    &first.descriptor,
                    &first.device_memory,
                    &slots,
                )
                .map_err(prepared_candidate_execute_error)?;
            approved.extend(slots.into_iter().map(|slot| (shard_id, slot)));
        }
        let hits = candidates
            .into_iter()
            .filter(|hit| approved.contains(&(hit.shard_id, hit.slot)))
            .collect::<Vec<_>>();
        Ok(hits)
    }

    /// Apply a bound UPDATE assignment set against one pinned device source. Literal assignment
    /// remains control-plane row construction; same-column addition is evaluated by the checked
    /// arithmetic VM only at the device-approved coordinates. SQL assignments read the original
    /// row image simultaneously, so every expression is launched before its output is installed.
    pub(crate) fn apply_update_assignments_on_device(
        &self,
        table: &RelationalTable,
        assignments: &[BoundUpdateAssignment],
        descriptor: &RelationalResidencySnapshot,
        device_memory: &CudaResidentDeviceMemory,
        slots: &[u32],
        rows: &mut [Vec<SqlValue>],
    ) -> Result<(), EngineError> {
        if slots.len() != rows.len() {
            return Err(EngineError::ApplyFailed(
                "device UPDATE coordinate/image cardinality mismatch".to_string(),
            ));
        }
        let mut pending: Vec<(usize, Vec<SqlValue>)> = Vec::with_capacity(assignments.len());
        for assignment in assignments {
            let column = table.columns.get(assignment.column_idx).ok_or_else(|| {
                EngineError::ApplyFailed(
                    "UPDATE assignment column is outside the table".to_string(),
                )
            })?;
            match &assignment.value {
                BoundUpdateValue::Literal(value) => {
                    pending.push((assignment.column_idx, vec![value.clone(); rows.len()]));
                }
                BoundUpdateValue::AddSameColumn(delta) => {
                    if matches!(delta, SqlValue::Null) {
                        pending.push((assignment.column_idx, vec![SqlValue::Null; rows.len()]));
                        continue;
                    }
                    let mut active_slots = Vec::new();
                    let mut active_rows = Vec::new();
                    for (row_idx, (slot, row)) in slots.iter().zip(rows.iter()).enumerate() {
                        if !matches!(row.get(assignment.column_idx), Some(SqlValue::Null)) {
                            active_slots.push(*slot);
                            active_rows.push(row_idx);
                        }
                    }
                    let mut values = vec![SqlValue::Null; rows.len()];
                    if !active_slots.is_empty() {
                        let offset = crate::relational_model::resident_device_int_column_offset(
                            descriptor,
                            table,
                            assignment.column_idx,
                        )
                        .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
                        let (program, elem) = match (column.ty, delta) {
                            (SqlType::Int4, SqlValue::Int4(delta)) => (
                                vec![
                                    gpu_db_execution::ExprStep::LoadColumn {
                                        byte_offset: offset,
                                    },
                                    gpu_db_execution::ExprStep::ScalarBinary {
                                        op: 0,
                                        scalar: *delta,
                                        scalar_on_left: false,
                                    },
                                ],
                                ResidentElemType::I32,
                            ),
                            (SqlType::Int8, SqlValue::Int8(delta)) => (
                                vec![
                                    gpu_db_execution::ExprStep::LoadColumnI64 {
                                        byte_offset: offset,
                                    },
                                    gpu_db_execution::ExprStep::ScalarBinaryI64 {
                                        op: 0,
                                        scalar: *delta,
                                        scalar_on_left: false,
                                    },
                                ],
                                ResidentElemType::I64,
                            ),
                            _ => {
                                return Err(EngineError::ApplyFailed(format!(
                                    "checked UPDATE addition type mismatch for column \"{}\"",
                                    column.name
                                )));
                            }
                        };
                        let computed = device_memory
                            .arith_value_column_at_selected_indices(
                                &program,
                                descriptor.row_count as u64,
                                &active_slots,
                                elem,
                            )
                            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
                        if computed.len() != active_rows.len() {
                            return Err(EngineError::ApplyFailed(
                                "device UPDATE arithmetic cardinality mismatch".to_string(),
                            ));
                        }
                        for (row_idx, value) in active_rows.into_iter().zip(computed) {
                            values[row_idx] = match column.ty {
                                SqlType::Int4 => SqlValue::Int4(value as i32),
                                SqlType::Int8 => SqlValue::Int8(value),
                                _ => unreachable!("bound UPDATE arithmetic is int4/int8 only"),
                            };
                        }
                    }
                    pending.push((assignment.column_idx, values));
                }
            }
        }
        for (column_idx, values) in pending {
            for (row, value) in rows.iter_mut().zip(values) {
                row[column_idx] = value;
            }
        }
        Ok(())
    }

    /// RETIREMENT A4a: materialize a located row ENTIRELY FROM THE DEVICE — values gathered from
    /// the hit's pinned shard columns, visibility decided by the created_by/deleted_by regions —
    /// the replacement for the host `tuple_fetch_by_key` that the install elision (A4e) removes.
    /// Returns:
    ///   `Some(Some(row))` — the slot is VISIBLE at `read_txn_id`, row = the device column values;
    ///   `Some(None)`      — the slot is NOT visible at this snapshot (created after it, or
    ///                       tombstoned at-or-before it) — the device analog of a fetch miss;
    ///   `None`            — DECLINE: this primitive cannot answer (a null-bearing shard whose raw
    ///                       i32 read would alias NULL as 0, a non-int4 column, or a device-read
    ///                       failure) -> the caller must fail closed.
    /// Visibility semantics are SV3b/SV6's exactly: visible ⟺ `created_by <= read_txn_id <
    /// deleted_by`, with an ABSENT created_by region = born-visible (0) and an ABSENT deleted_by
    /// region = never-deleted (+inf).
    ///
    /// CONTRACT (load-bearing): `read_txn_id` must be >= the commit seq of every INSERT-appended
    /// slot in the hit's shard snapshot — i.e. the CURRENT published seq, which is what every
    /// serialized DML prepare/preflight passes. Only UPDATE-appended versions carry created_by
    /// stamps (SV6); plain INSERT appends are BORN-VISIBLE and are gated for concurrent READERS
    /// by the snapshot-pinned `row_count` instead — a mechanism a slot-addressed materializer
    /// cannot replicate. Historical time-travel below an unstamped insert is NOT this primitive's
    /// contract. WIRED by A4e: the elided resolve + probe materialize through this (proven
    /// first by the `a4a_device_materialization_matches_host_fetch` differential).
    pub(crate) fn materialize_resident_row_via_hit(
        &self,
        table: &RelationalTable,
        hit: &crate::engine_retained_read::ShardPkHit,
        read_txn_id: u64,
    ) -> Option<Option<Vec<SqlValue>>> {
        // ADR-006 (nullable-column DML): a NULL-bearing shard no longer declines wholesale — each
        // column's validity bitmap is read per-slot below (a 0 bit ⇒ SqlValue::Null), so a
        // DELETE/UPDATE whose PREDICATE is device-resolvable resolves on-device even when the table
        // has nullable columns (the located rows already excluded NULL predicate operands via the
        // device 3VL validity-AND; the recheck's `select_filter_matches` re-applies 3VL). Was: a
        // blanket decline because a raw i32 read would alias a stored NULL as 0.
        // Audit A4 F1, lifted by TYPE-COVERAGE track 2 (stages 1 + iii) + #14: every FIXED-WIDTH
        // section materializes with its CATALOG-derived variant — i32 via one u32/slot, i64 via two
        // (the 4-mod-8 discipline), b128 (Numeric/Uuid) via four. TEXT (variable-length) materializes
        // this slot's blob span from the shard's text section (offsets+blob) — the compound-text-key
        // recheck reads the resident key on-device without transferring relational authority.
        if table.columns.iter().any(|column| {
            !matches!(
                column.ty,
                crate::SqlType::Int4
                    | crate::SqlType::Date
                    | crate::SqlType::Int2
                    | crate::SqlType::Int8
                    | crate::SqlType::Timestamp
                    | crate::SqlType::Numeric { .. }
                    | crate::SqlType::Uuid
                    | crate::SqlType::Text
                    | crate::SqlType::Bool
            )
        }) {
            return None;
        }
        let slot = u64::from(hit.slot);
        let read_region_u64 = |region: &crate::CudaResidentDeviceMemory| -> Option<u64> {
            let halves = region.read_resident_i32_column(slot * 8, 2).ok()?;
            let (lo, hi) = (*halves.first()?, *halves.get(1)?);
            Some((lo as u32 as u64) | ((hi as u32 as u64) << 32))
        };
        let created_by = match &hit.created_by {
            Some(region) => read_region_u64(region)?,
            None => 0, // un-stamped shard: born-visible
        };
        let deleted_by = match &hit.deleted_by {
            Some(region) => read_region_u64(region)?,
            None => u64::MAX, // version-free shard: never deleted
        };
        if !(created_by <= read_txn_id && read_txn_id < deleted_by) {
            return Some(None);
        }
        let mut row = Vec::with_capacity(table.columns.len());
        for idx in 0..table.columns.len() {
            // ADR-006 (nullable-column DML): if this column has a validity bitmap and the slot's bit
            // is 0, the value is NULL (byte-identical to the gather Bool-bitmap addressing: word
            // `slot/32`, bit `slot%32`, LSB-first; 1 = present, 0 = NULL). Read ONE word for the slot.
            if let Some(layout) = hit
                .descriptor
                .resident_device_null_columns
                .iter()
                .find(|layout| layout.name == table.columns[idx].name)
            {
                let word = hit
                    .device_memory
                    .read_resident_i32_column(layout.bitmap_byte_offset + (slot / 32) * 4, 1)
                    .ok()?;
                if (*word.first()? as u32 >> (slot % 32)) & 1 == 0 {
                    row.push(SqlValue::Null);
                    continue;
                }
            }
            match table.columns[idx].ty {
                crate::SqlType::Numeric { .. } | crate::SqlType::Uuid => {
                    // b128 (16-byte) section: 4 LE i32 words per slot, reassembled byte-identically to
                    // the rehydration decode (`gather_resident_table_rows_from_device` B128 arm).
                    let base = crate::relational_model::resident_device_numeric_column_offset(
                        &hit.descriptor,
                        table,
                        idx,
                    )
                    .ok()?;
                    let words = hit
                        .device_memory
                        .read_resident_i32_column(base + slot * 16, 4)
                        .ok()?;
                    if words.len() != 4 {
                        return None;
                    }
                    let mut bytes = [0u8; 16];
                    for (w, word) in words.iter().enumerate() {
                        bytes[w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
                    }
                    row.push(match table.columns[idx].ty {
                        crate::SqlType::Numeric { scale, .. } => SqlValue::Numeric(
                            gpu_db_sql::Decimal128::new(i128::from_le_bytes(bytes), scale),
                        ),
                        crate::SqlType::Uuid => SqlValue::Uuid(bytes),
                        _ => return None,
                    });
                }
                crate::SqlType::Int8 | crate::SqlType::Timestamp => {
                    let base = crate::relational_model::resident_device_int8_column_offset(
                        &hit.descriptor,
                        table,
                        idx,
                    )
                    .ok()?;
                    let halves = hit
                        .device_memory
                        .read_resident_i32_column(base + slot * 8, 2)
                        .ok()?;
                    let lo = *halves.first()? as u32 as u64;
                    let hi = *halves.get(1)? as u32 as u64;
                    row.push(crate::engine_residency::sql_value_from_i64_section(
                        table.columns[idx].ty,
                        (lo | (hi << 32)) as i64,
                    )?);
                }
                crate::SqlType::Text => {
                    // TEXT section: read THIS slot's [start,end) offsets (2 consecutive u64) then the
                    // blob span, byte-identical to the rehydration decode
                    // (`gather_resident_table_rows_from_device` Text arm), but for one slot only.
                    let layout = crate::relational_model::resident_device_text_column_layout(
                        &hit.descriptor,
                        table,
                        idx,
                    )
                    .ok()?;
                    let bounds = hit
                        .device_memory
                        .read_resident_u64_column(layout.offsets_byte_offset + slot * 8, 2)
                        .ok()?;
                    let start = *bounds.first()?;
                    let end = *bounds.get(1)?;
                    if end < start {
                        return None;
                    }
                    let bytes = hit
                        .device_memory
                        .read_resident_bytes(
                            layout.bytes_byte_offset + start,
                            (end - start) as usize,
                        )
                        .ok()?;
                    let text = std::str::from_utf8(&bytes).ok()?.to_string();
                    row.push(SqlValue::Text(text));
                }
                crate::SqlType::Bool => {
                    // BOOL section: a 1-bit-per-row bitmap (LSB-first u32 words). Read the u32 word
                    // holding THIS slot and extract its bit, byte-identical to the rehydration decode
                    // (`gather_resident_table_rows_from_device` Bool arm), for one slot only.
                    let base = crate::relational_model::resident_device_bool_column_offset(
                        &hit.descriptor,
                        table,
                        idx,
                    )
                    .ok()?;
                    let word = hit
                        .device_memory
                        .read_resident_i32_column(base + (slot / 32) * 4, 1)
                        .ok()?;
                    let bit = (*word.first()? as u32 >> (slot % 32)) & 1;
                    row.push(SqlValue::Bool(bit == 1));
                }
                _ => {
                    let base = crate::relational_model::resident_device_int4_column_offset(
                        &hit.descriptor,
                        table,
                        idx,
                    )
                    .ok()?;
                    let values = hit
                        .device_memory
                        .read_resident_i32_column(base + slot * 4, 1)
                        .ok()?;
                    row.push(crate::engine_residency::sql_value_from_i32_section(
                        table.columns[idx].ty,
                        *values.first()?,
                    )?);
                }
            }
        }
        Some(Some(row))
    }

    /// Exact typed device constraint probe. `Some(bool)` is authoritative; `None` makes the caller
    /// fail loud. Physical hits are visibility-checked from the captured generation, and NULL uses
    /// the resident validity bitmap's structural `IS NULL` semantics.
    pub(crate) fn device_visible_row_with_value(
        &self,
        table: &RelationalTable,
        mut visibility: StorageVisibility,
        column_idx: usize,
        value: &SqlValue,
        exclude_keys: Option<&BTreeSet<String>>,
    ) -> Option<bool> {
        if let Some(snapshot) = self.current_transaction_read_snapshot() {
            if snapshot.table_has_typed_empty_root(&table.name)
                || (!snapshot
                    .catalog
                    .relational_catalog
                    .contains_key(&table.name)
                    && snapshot
                        .transaction_shards()
                        .get(&table.name)
                        .is_some_and(Vec::is_empty))
            {
                // A private CREATE owns an exact empty device relation before its first INSERT.
                // No published or host relation can contain a conflicting key, and later private
                // statements replace the empty vector with real GPU shards before probing here.
                return Some(false);
            }
        }
        if self.table_chunk_authoritative(&table.name).is_some() {
            if self.current_transaction_read_snapshot().is_none() {
                visibility.read_txn_id = visibility.read_txn_id.max(self.committed_seq());
            }
            return self.chunk_class_visible_row_with_value(
                table,
                visibility.read_txn_id,
                column_idx,
                value,
                exclude_keys,
            );
        }
        if crate::engine_prepared_transaction::prepared_index_route_required() {
            let hits = self
                .prepared_exact_index_hits(
                    table,
                    &[(column_idx, SelectFilterOp::Eq, value.clone())],
                    visibility,
                )
                .ok()?;
            return self.prepared_index_hit_exists(table, &hits, exclude_keys);
        }
        // R3-004: use the exact typed predicate for every resident probe. Fingerprint indexes are
        // addressing accelerators, not authorities; bypassing them here removes the host tuple
        // recheck that used to compensate for collisions and stale physical versions.
        let column_ty = table.columns.get(column_idx)?.ty;
        if self.current_transaction_read_snapshot().is_none() {
            visibility.read_txn_id = visibility.read_txn_id.max(self.committed_seq());
        }
        let predicate = if matches!(value, SqlValue::Null) {
            crate::engine_expr::ResidentExpr::IsNull {
                col: column_idx,
                is_not_null: false,
            }
        } else {
            let rhs = device_eq_scan_literal(column_ty, value)?;
            crate::engine_expr::ResidentExpr::Binary {
                op: crate::engine_expr::ResidentBinaryOp::Eq,
                lhs: Box::new(crate::engine_expr::ResidentExpr::Column(column_idx)),
                rhs: Box::new(rhs),
            }
        };
        let hits = self.locate_resident_delete_slots_detailed(table, &predicate)?;
        let mut answer = false;
        for hit in &hits {
            let region = hit.row_id.as_ref()?;
            // A device-read failure declines the whole probe (the A2 finding-2 discipline).
            let halves = match region.read_resident_i32_column(u64::from(hit.slot) * 8, 2) {
                Ok(halves) => halves,
                Err(_) => return None,
            };
            let (lo, hi) = (*halves.first()?, *halves.get(1)?);
            let row_id = (lo as u32 as u64) | ((hi as u32 as u64) << 32);
            if row_id == u64::MAX {
                return None; // unstamped slot cannot authorize a constraint verdict
            }
            let key = relational_row_key(&table.name, row_id);
            if exclude_keys.is_some_and(|excluded| excluded.contains(&key)) {
                continue; // exclude rows replaced by this statement
            }
            match self.materialize_resident_row_via_hit(table, hit, visibility.read_txn_id) {
                Some(Some(_)) => {
                    answer = true;
                    break;
                }
                Some(None) => continue,
                None => return None,
            }
        }
        self.read_state
            .residency
            .dml_device_validate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(answer)
    }

    pub(crate) fn prepared_index_hit_exists(
        &self,
        table: &RelationalTable,
        hits: &[crate::engine_retained_read::ShardPkHit],
        exclude_keys: Option<&BTreeSet<String>>,
    ) -> Option<bool> {
        for hit in hits {
            let region = hit.row_id.as_ref()?;
            let halves = region
                .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
                .ok()?;
            let (lo, hi) = (*halves.first()?, *halves.get(1)?);
            let row_id = (lo as u32 as u64) | ((hi as u32 as u64) << 32);
            if row_id == u64::MAX {
                return None;
            }
            let key = relational_row_key(&table.name, row_id);
            if !exclude_keys.is_some_and(|excluded| excluded.contains(&key)) {
                self.read_state
                    .residency
                    .dml_device_validate_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(true);
            }
        }
        self.read_state
            .residency
            .dml_device_validate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(false)
    }

    /// Device-authoritative structural equality probe. A decline fails loud; it never dispatches
    /// to a host value index or rehydrates a host relational generation.
    pub(crate) fn visible_row_with_value(
        &self,
        table: &RelationalTable,
        visibility: StorageVisibility,
        column_idx: usize,
        value: &SqlValue,
        exclude_keys: Option<&BTreeSet<String>>,
    ) -> Result<bool, EngineError> {
        self.device_visible_row_with_value(table, visibility, column_idx, value, exclude_keys)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "device constraint verdict unavailable for relation \"{}\"",
                    table.name
                ))
            })
    }

    /// Device-authoritative tuple generalization of [`Self::visible_row_with_value`]. A fingerprint
    /// may address candidates, but exact typed tuple equality is the verdict; any decline fails loud.
    pub(crate) fn visible_row_with_tuple(
        &self,
        table: &RelationalTable,
        visibility: StorageVisibility,
        key_id: usize,
        // `None` = the tuple has a non-i32-section key value (e.g. NULL), so the fingerprint probe
        // cannot run. The device tuple helper then builds an exact typed predicate (including
        // structural IS NULL leaves); an authoritative relation fails closed if that scan declines.
        fingerprint: Option<i32>,
        key_cols: &[(usize, SqlValue)],
        exclude_keys: Option<&BTreeSet<String>>,
    ) -> Result<bool, EngineError> {
        if fingerprint.is_some() {
            if let Some(answer) = self.device_visible_row_with_tuple(
                table,
                visibility,
                key_id,
                fingerprint,
                key_cols,
                exclude_keys,
            ) {
                return Ok(answer);
            }
        } else if let [(column_idx, value)] = key_cols {
            // A single nullable fingerprint key has no fingerprint word. Preserve structural
            // UNIQUE NULL semantics with the resident validity bitmap's IS NULL scan before the
            // tuple path; compound partial-NULL tuples use the tuple-aware predicate below.
            if let Some(answer) = self.device_visible_row_with_value(
                table,
                visibility,
                *column_idx,
                value,
                exclude_keys,
            ) {
                return Ok(answer);
            }
        } else if let Some(answer) = self.device_visible_row_with_tuple(
            table,
            visibility,
            key_id,
            None,
            key_cols,
            exclude_keys,
        ) {
            return Ok(answer);
        }
        Err(EngineError::ApplyFailed(format!(
            "device tuple-constraint verdict unavailable for relation \"{}\"",
            table.name
        )))
    }

    /// Evaluate row-local CHECK violations over a transient device relation. The predicate is the
    /// exact logical complement of the stored CHECK comparison; SQL NULLs satisfy CHECK because
    /// the device validity mask excludes them from both the comparison and its complement. The
    /// compacted candidate coordinates are only marshaled into the device threshold primitive;
    /// its one status bit is the constraint verdict.
    fn validate_check_constraints_on_device(
        &self,
        table: &RelationalTable,
        new_images: &[Vec<SqlValue>],
    ) -> Result<(), EngineError> {
        if new_images.is_empty() || table.check_constraints.is_empty() {
            return Ok(());
        }
        let (snapshot, memory) = self
            .build_transient_relation_residency(table, new_images)
            .map_err(|error| {
                EngineError::ApplyFailed(format!(
                    "device CHECK relation build failed for \"{}\": {error}",
                    table.name
                ))
            })?;
        for constraint in &table.check_constraints {
            let column = relational_column_index(table, &constraint.column)
                .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
            let predicate = dml_filter_groups_to_device_predicate(
                table,
                &[vec![(column, constraint.op, constraint.value.clone())]],
            )
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "device CHECK predicate unavailable for constraint \"{}\"",
                    constraint.name
                ))
            })?;
            let ResidentExpr::Binary { op, lhs, rhs } = predicate else {
                return Err(EngineError::ApplyFailed(format!(
                    "device CHECK predicate shape unavailable for constraint \"{}\"",
                    constraint.name
                )));
            };
            let violation_op = match op {
                ResidentBinaryOp::Eq => ResidentBinaryOp::Ne,
                ResidentBinaryOp::Lt => ResidentBinaryOp::Ge,
                ResidentBinaryOp::Le => ResidentBinaryOp::Gt,
                ResidentBinaryOp::Gt => ResidentBinaryOp::Le,
                ResidentBinaryOp::Ge => ResidentBinaryOp::Lt,
                _ => {
                    return Err(EngineError::ApplyFailed(format!(
                        "device CHECK operator unavailable for constraint \"{}\"",
                        constraint.name
                    )))
                }
            };
            let violation = ResidentExpr::Binary {
                op: violation_op,
                lhs,
                rhs,
            };
            let candidates = self
                .lower_resident_predicate(
                    &violation,
                    table,
                    &snapshot,
                    &memory,
                    new_images.len() as u64,
                    None,
                )
                .map_err(|error| {
                    EngineError::ApplyFailed(format!(
                        "device CHECK evaluation failed for constraint \"{}\": {error}",
                        constraint.name
                    ))
                })?
                .into_iter()
                .map(u64::from)
                .collect::<Vec<_>>();
            let violated = memory
                .unique_coordinate_threshold_reached(&candidates, &[], 1)
                .map_err(|error| {
                    EngineError::ApplyFailed(format!(
                        "device CHECK verdict failed for constraint \"{}\": {error}",
                        constraint.name
                    ))
                })?;
            if violated {
                return Err(EngineError::CheckViolation(format!(
                    "new row for relation \"{}\" violates check constraint \"{}\"",
                    table.name, constraint.name
                )));
            }
        }
        Ok(())
    }

    /// Device-native constraint validation restricted to what the statement can affect: untouched
    /// survivors were valid before it, so only new images (unique/CHECK/outbound FK) and removed
    /// provider values (inbound FK) need work. Exact typed device probes replace related-table
    /// survivor materializations. Validator order remains unique -> CHECK -> FK.
    ///
    /// DELETE passes empty `new_images` (unique/check/outbound sections no-op, exactly as the scan
    /// path never ran them for DELETE); UPDATE passes the post-assignment images.
    ///
    /// Each probe captures its own device generation and uses `visibility.read_txn_id`; no host view
    /// is threaded through the ladder.
    pub(crate) fn validate_dml_constraints_via_device(
        &self,
        catalog: &CatalogSnapshot,
        table: &RelationalTable,
        new_images: &[Vec<SqlValue>],
        removed_images: &[Vec<SqlValue>],
        touched_keys: &BTreeSet<String>,
        visibility: StorageVisibility,
    ) -> Result<(), EngineError> {
        // 0. PK NOT NULL (PG 23502): checked FIRST, before unique — byte-identical to the scan arm.
        Self::validate_primary_key_not_null(table, new_images.iter().map(Vec::as_slice))?;
        // 1. UNIQUE: the transient device relation owns the exact within-statement tuple verdict.
        // The ordinary/general path deliberately has no row-count compatibility cap; the bounded
        // chunk class applies its own latency-class cap before calling the same GPU primitive.
        match self.validate_new_rows_unique_on_device(table, new_images) {
            Some(result) => result?,
            None => {
                return Err(EngineError::ApplyFailed(format!(
                    "device in-batch unique verdict unavailable for relation \"{}\"",
                    table.name
                )))
            }
        }
        // Each new value is then checked against the UNTOUCHED visible device generation.
        for (ord, index) in table.indexes.iter().enumerate().filter(|(_, i)| i.unique) {
            // COMPOUND KEYS: validate the ORDERED key TUPLE (single-column keys resolve `[column_idx]`,
            // byte-identical to the prior path). In-batch tuple dedup + each new tuple vs untouched
            // visible rows are both decided from the authoritative device generation.
            let Some(positions) = crate::engine_residency::index_key_column_positions(table, index)
            else {
                continue;
            };
            let fingerprint_backed = crate::engine_residency::index_uses_fingerprint(table, index);
            let key_id = crate::engine_residency::index_probe_key_id(table, index, ord)
                .unwrap_or(positions[0]);
            for row in new_images {
                if positions
                    .iter()
                    .any(|&position| matches!(row[position], SqlValue::Null))
                {
                    continue;
                }
                let conflict = if fingerprint_backed {
                    let key_cols: Vec<(usize, SqlValue)> =
                        positions.iter().map(|&i| (i, row[i].clone())).collect();
                    let fingerprint =
                        crate::engine_residency::compound_index_row_fingerprint(table, index, row);
                    self.visible_row_with_tuple(
                        table,
                        visibility,
                        key_id,
                        fingerprint,
                        &key_cols,
                        Some(touched_keys),
                    )?
                } else {
                    self.visible_row_with_value(
                        table,
                        visibility,
                        key_id,
                        &row[positions[0]],
                        Some(touched_keys),
                    )?
                };
                if conflict {
                    return Err(EngineError::UniqueViolation(format!(
                        "duplicate key value violates unique index \"{}\"",
                        index.name
                    )));
                }
            }
        }
        // 2. CHECK: a transient device relation evaluates the complement predicate and the device
        // threshold bit decides whether any non-NULL row violates it. Survivors passed at their own
        // write; ADD CHECK validates existing rows at DDL time.
        self.validate_check_constraints_on_device(table, new_images)?;
        // 3. OUTBOUND FK (this table is the child): each new image's FK value must have a visible
        //    provider. For a self-FK, another new image in this statement may provide it.
        for foreign_key in &table.foreign_keys {
            let Some(parent) = catalog
                .relational_catalog
                .get(&foreign_key.referenced_table)
            else {
                continue;
            };
            let child_idx = relational_column_index(table, &foreign_key.column)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            let parent_idx = relational_column_index(parent, &foreign_key.referenced_column)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            for row in new_images {
                // PG 3VL (MATCH SIMPLE): a NULL fk value references nothing — no provider needed
                // (and a structural NULL==NULL index hit on a parent NULL must not "provide").
                if matches!(row[child_idx], SqlValue::Null) {
                    continue;
                }
                if parent.name == table.name
                    && new_images
                        .iter()
                        .any(|candidate| candidate[parent_idx] == row[child_idx])
                {
                    continue;
                }
                if !self.visible_row_with_value(
                    parent,
                    visibility,
                    parent_idx,
                    &row[child_idx],
                    None,
                )? {
                    return Err(EngineError::ForeignKeyViolation(format!(
                        "insert or update on table \"{}\" violates foreign key constraint \"{}\"",
                        table.name, foreign_key.name
                    )));
                }
            }
        }
        // 4. INBOUND FK (children referencing this table): a REMOVED provider value that a visible
        //    child row still references, with no surviving (or newly-installed) provider, is a
        //    violation. Restricted-to-removed-values is equivalent to the scan validator's full
        //    child-set check under the survivors-were-valid invariant.
        for child in catalog.relational_catalog.values() {
            for foreign_key in &child.foreign_keys {
                if foreign_key.referenced_table != table.name {
                    continue;
                }
                let parent_idx = relational_column_index(table, &foreign_key.referenced_column)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let child_idx = relational_column_index(child, &foreign_key.column)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                let new_provider_values: BTreeSet<&SqlValue> =
                    new_images.iter().map(|row| &row[parent_idx]).collect();
                let mut checked: BTreeSet<&SqlValue> = BTreeSet::new();
                for old in removed_images {
                    let value = &old[parent_idx];
                    // A removed NULL provider value cannot orphan anyone: a NULL fk passes
                    // regardless (PG MATCH SIMPLE) and NULL never "provides".
                    if matches!(value, SqlValue::Null) {
                        continue;
                    }
                    if !checked.insert(value) || new_provider_values.contains(value) {
                        continue;
                    }
                    // A surviving untouched provider keeps the value alive.
                    if self.visible_row_with_value(
                        table,
                        visibility,
                        parent_idx,
                        value,
                        Some(touched_keys),
                    )? {
                        continue;
                    }
                    // No provider left: any visible child row still referencing it = violation.
                    let child_exclusions = (child.name == table.name).then_some(touched_keys);
                    if self.visible_row_with_value(
                        child,
                        visibility,
                        child_idx,
                        value,
                        child_exclusions,
                    )? {
                        return Err(EngineError::ForeignKeyViolation(format!(
                            "insert or update on table \"{}\" violates foreign key constraint \"{}\"",
                            child.name, foreign_key.name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// PURE preflight + scan + encode for `UPDATE` (write-half MVCC, Stage 2). Resolves the
    /// matching versions, applies the assignments to encode the new row images, runs the unique /
    /// check / FK preflight as the old `apply_update`, and records the write-set (old slot
    /// tombstoned + new version + unique slots). No engine mutation.
    pub(crate) fn prepare_update(
        &self,
        update: &Update,
        snapshot: DmlReadSnapshot,
    ) -> Result<WriteDelta, EngineError> {
        let txn_id = snapshot.commit_seq;
        // Lock-free concurrent-DML path (Stage 2 — blocker #1): pin ONE catalog snapshot for the
        // target bind and the inbound-FK-dependents scan below.
        let catalog = self.catalog_snapshot();
        let table = catalog
            .relational_catalog
            .get(&update.table)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!("relation \"{}\" does not exist", update.table))
            })?;
        let assignments = bind_update_assignments(table, update)
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let filter_groups = bind_delete_filter_groups(
            table,
            &Delete {
                table: update.table.clone(),
                filter: update.filter.clone(),
                filters: update.filters.clone(),
                filter_groups: update.filter_groups.clone(),
                returning: Vec::new(),
            },
        )
        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let mut updates = Vec::new();
        // SV5: OLD images (catalog order) captured before the assignments, PARALLEL to `updates`.
        let mut updated_old_rows: Vec<Vec<SqlValue>> = Vec::new();
        // Unique slots the OLD images RELEASE (prereq #2, Stage-4 audit). An UPDATE that changes a
        // unique column frees its old `(table, column, value)` slot; record those freed slots in the
        // write-set so a CONCURRENT insert/update reusing the freed value conflicts under
        // first-committer-wins — matching the DELETE path, which already records the released slots.
        // This is the conservative choice: it never admits a phantom unique duplicate across a
        // concurrent free+reuse (a slot-release left unrecorded could). A no-op-on-the-unique-column
        // UPDATE records the same slot as both released (old) and claimed (new) — harmless (the
        // write-set dedups to one slot), so an idempotent rewrite does not self-conflict.
        let mut released_unique_slots: Vec<UniqueIndexSlotKey> = Vec::new();
        let constrained = table.indexes.iter().any(|index| index.unique)
            || !table.check_constraints.is_empty()
            || !table.foreign_keys.is_empty()
            || catalog.relational_catalog.values().any(|candidate| {
                candidate
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == table.name)
            });
        // Resolve from the authoritative device generation. Cold keyed classes stay on their
        // device-native coordinate/index path; resident tables use typed predicate scan/compaction.
        let (matches, resolved_old_rows, class_epoch) = if self
            .table_chunk_authoritative(&table.name)
            .is_some()
        {
            let (matches, old_rows, epoch) = self
                .resolve_class_update_matches(table, &filter_groups, visibility, &assignments)?
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "device DML verdict unavailable for cold relation \"{}\"",
                        table.name
                    ))
                })?;
            if table.indexes.iter().any(|index| index.unique) {
                let mut new_images: Vec<Vec<SqlValue>> = Vec::with_capacity(matches.len());
                for (_, _, row) in &matches {
                    new_images.push(row.clone());
                }
                let own: BTreeSet<u64> = matches.iter().map(|(id, _, _)| *id).collect();
                self.validate_class_insert_uniqueness(
                    table,
                    &new_images,
                    visibility.read_txn_id,
                    Some((&own, epoch)),
                )
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "device uniqueness verdict unavailable for cold relation \"{}\"",
                        table.name
                    ))
                })??;
            }
            (matches, old_rows, Some(epoch))
        } else {
            let (matches, old_rows) = self
                .resolve_update_matches_via_device(table, &filter_groups, visibility, &assignments)?
                .ok_or_else(|| {
                    EngineError::ApplyFailed(format!(
                        "device DML verdict unavailable for relation \"{}\"",
                        table.name
                    ))
                })?;
            (matches, old_rows, None)
        };
        if matches.len() != resolved_old_rows.len() {
            return Err(EngineError::ApplyFailed(
                "device UPDATE old/new image cardinality mismatch".to_string(),
            ));
        }
        for ((tuple_id, key, row), old_row) in matches.into_iter().zip(resolved_old_rows) {
            let mut old_slots = WriteSet::default();
            old_slots.add_unique_slots(table, &old_row);
            released_unique_slots.append(&mut old_slots.unique_slots);
            updated_old_rows.push(old_row);
            updates.push((tuple_id, key, row));
        }

        if constrained {
            let touched_keys: BTreeSet<String> =
                updates.iter().map(|(_, key, _)| key.clone()).collect();
            let new_images: Vec<Vec<SqlValue>> =
                updates.iter().map(|(_, _, row)| row.clone()).collect();
            self.validate_dml_constraints_via_device(
                &catalog,
                table,
                &new_images,
                &updated_old_rows,
                &touched_keys,
                visibility,
            )?;
        }

        let mut write_set = WriteSet::default();
        write_set.tables.insert(update.table.clone());
        for (_, key, row) in &updates {
            // An UPDATE tombstones the old version and installs a new one at the SAME row key,
            // so the row slot is written once.
            write_set.rows.push(RowWriteKey {
                table: update.table.clone(),
                row_key: key.clone(),
            });
            // The new image's unique-index slots are claimed by this txn.
            write_set.add_unique_slots(table, row);
        }
        // The old images' RELEASED unique slots are also conflict points (prereq #2). Dedup so a
        // value carried unchanged through the UPDATE (same slot released and re-claimed) is recorded
        // once and never self-conflicts.
        write_set.unique_slots.append(&mut released_unique_slots);
        write_set.unique_slots.sort();
        write_set.unique_slots.dedup();

        // `updates` is already `(tuple_id, row_key, new_values)` — exactly the install shape.
        let (catalog_dependencies, foreign_key_dependencies) =
            transaction_dml_dependencies(&catalog, table);
        Ok(WriteDelta {
            write_set,
            read_snapshot: snapshot.commit_seq,
            catalog_dependencies,
            foreign_key_dependencies,
            rows_consumed: 0,
            mutation: PreparedMutation::Update {
                table: update.table.clone(),
                installs: updates,
                updated_old_rows,
                class_epoch,
            },
        })
    }
}

fn transaction_dml_dependencies(
    catalog: &CatalogSnapshot,
    target: &RelationalTable,
) -> (BTreeMap<String, RelationalTable>, BTreeSet<String>) {
    let mut names = BTreeSet::from([target.name.clone()]);
    let mut foreign_key_dependencies = BTreeSet::new();
    for foreign_key in &target.foreign_keys {
        names.insert(foreign_key.referenced_table.clone());
        foreign_key_dependencies.insert(foreign_key.referenced_table.clone());
    }
    for candidate in catalog.relational_catalog.values() {
        if candidate
            .foreign_keys
            .iter()
            .any(|foreign_key| foreign_key.referenced_table == target.name)
        {
            names.insert(candidate.name.clone());
            foreign_key_dependencies.insert(candidate.name.clone());
        }
    }
    let catalog_dependencies = names
        .into_iter()
        .filter_map(|name| {
            catalog
                .relational_catalog
                .get(&name)
                .cloned()
                .map(|table| (name, table))
        })
        .collect();
    (catalog_dependencies, foreign_key_dependencies)
}

fn prepared_candidate_execute_error(error: ExecuteError) -> EngineError {
    match error {
        ExecuteError::Engine(error) => error,
        other => EngineError::ApplyFailed(other.to_string()),
    }
}
