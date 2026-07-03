//! DML mutation prepare path (P0 §9.6 decomposition, behavior-preserving): a
//! focused `impl Engine` block that turns a parsed Insert/Delete/Update into a
//! prepared WriteDelta off-lock (prepare_insert / prepare_delete / prepare_update
//! against a dml_read_snapshot), plus the direct apply_insert helper. Pairs with
//! engine_write_apply (which installs the WriteDelta under the commit lock).

use super::*;

/// PHASE C slice 1: one resolved DML match — `(tuple_id, row_key, decoded_row)`, exactly the triple
/// the seq_scan produced. `None` from the resolver = index-ineligible -> the caller scans.
pub(crate) type DmlResolvedMatch = (u64, String, Vec<SqlValue>);

/// Ledger #18: how much constraint validation `prepare_insert` runs. `Full` everywhere EXCEPT
/// the wave sequencer's under-lock RE-RESOLVE, where unique/CHECK re-validation of an FK-FREE
/// table is PROVABLY REDUNDANT — the coverage argument, verified against the sequencer:
///  - a dup committed at C <= S (the item's read snapshot): the OFF-LOCK prepare validated
///    against every row visible at S and errored the statement before it ever enqueued;
///  - a dup committed in (S, commit] — INCLUDING an earlier item of the SAME wave: the
///    ledger conflict check runs BEFORE the re-resolve (`conflicts` at the item loop head;
///    each item `record`s before later items validate) and aborts with a retryable
///    serialization conflict; the item's registered snapshot guard pins ledger pruning <= S,
///    so no entry it needs can vanish mid-flight;
///  - a WITHIN-STATEMENT dup (VALUES (1),(1)): deterministic on the statement text — the
///    off-lock prepare's in-batch check already rejected it;
///  - CHECK constraints are row-local and deterministic on the values: same verdict as the
///    off-lock pass.
/// FK re-validation is NOT covered (a parent provider deleted in (S, commit] writes the
/// PARENT's row keys into the ledger, which the CHILD's write-set never claims — no
/// conflict), so FK-bearing tables always validate fully. PK NOT-NULL (O(new), pure) runs
/// unconditionally as cheap defense.
///
/// PRECONDITION (audit 3b1be580): the skip is granted ONLY while the catalog generation
/// still matches the off-lock prepare's (`CommitWaveItem::prepared_catalog_seq`) — a
/// constraint-adding DDL (ADD UNIQUE/CHECK) committing in (S, wave] records NOTHING in the
/// conflict ledger and the item's write_set lacks slots for the new index, so an unguarded
/// skip silently bypassed it (sabotage-verified by
/// `wave_insert_prepared_before_add_check_is_revalidated`). Any DDL bumps the stamp -> Full.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum InsertPrepareValidation {
    Full,
    ReResolveLedgerCovered,
}

impl Engine {
    pub(crate) fn apply_insert(
        &self,
        cat: &mut DdlCatalogState,
        insert: Insert,
        txn_id: TxnId,
    ) -> Result<Option<(String, Vec<Vec<SqlValue>>, WriteSet, Vec<u64>)>, EngineError> {
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
            for column in &insert.columns {
                let idx = table
                    .columns
                    .iter()
                    .position(|candidate| candidate.name == *column)
                    .ok_or_else(|| {
                        EngineError::ApplyFailed(format!("column \"{}\" does not exist", column))
                    })?;
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
        Self::validate_primary_key_not_null(&table, new_rows.iter().map(Vec::as_slice))?;
        // TYPE-COVERAGE track 1 (ledger #17): INDEX-DRIVEN INSERT validation — O(new x constraints)
        // through the 1b-audited `validate_dml_constraints_via_index` (probe ladder: device index
        // first, value_index on decline), replacing the O(table) candidate materialization below.
        // This is THE measured PK'd-table collapse: `prepare_insert` is the concurrent path's
        // authoritative validation (P2 removed its duplicate preflight) AND re-runs under the
        // sequencer lock at re-resolve, so the scan cost 923 vs 102,045 sustained TPS @16w rode
        // on it twice per commit (oltp_commit_slo_benchmark, GPU_DB_BENCH_PK=1). Same eligibility
        // as the serialized write-apply Insert arm: self-referencing-FK tables keep the scan (a
        // new row may provide for another new row, which the parent's index cannot see
        // pre-install). Semantics + error text are byte-identical (the 1b contract).
        let self_referencing_fk = table
            .foreign_keys
            .iter()
            .any(|foreign_key| foreign_key.referenced_table == table.name);
        let has_constraints = table.indexes.iter().any(|index| index.unique)
            || !table.check_constraints.is_empty()
            || !table.foreign_keys.is_empty();
        // Ledger #18: the under-lock re-resolve skips the redundant unique/CHECK pass on
        // FK-free tables — the ledger conflict check IS the commit-time guard (see
        // [`InsertPrepareValidation`] for the coverage proof). Measured: the second validator
        // probe per commit was the PK'd arm's largest sequencer-side residual.
        let ledger_covered = validation == InsertPrepareValidation::ReResolveLedgerCovered
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
        let wave_deferred = validation == InsertPrepareValidation::Full
            && insert.rows.len() == 1
            && self.insert_unique_wave_batchable(&catalog, table);
        if ledger_covered || wave_deferred {
            // fall through to encode: PK not-null ran; unique covered by the ledger (#18) or
            // deferred to the wave batch (B).
        } else if self.dml_value_index_resolve_enabled() && !self_referencing_fk {
            if has_constraints {
                let validate_started = Instant::now();
                self.validate_dml_constraints_via_index(
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
        } else {
            // P2 (write-path assessment): ONE shared visible-row materialization for all three
            // validators — this used to be three separate O(table) scans (+ a `new_rows` clone
            // each) per prepare, i.e. per constraint dimension. The scan cost lands in the first
            // active validator's profile bucket (they used to pay one scan each); validation
            // semantics and errors are unchanged (`prepare_update` already shares its scan the
            // same way). Kept as the flag-off / self-referencing-FK oracle arm.
            let mut candidate_rows: Option<Vec<Vec<SqlValue>>> = None;
            let materialize_candidates = |engine: &Self| -> Result<Vec<Vec<SqlValue>>, EngineError> {
                let mut rows = engine.visible_relational_rows(
                    &table,
                    StorageVisibility {
                        read_txn_id: txn_id,
                    },
                )?;
                rows.extend(new_rows.clone());
                Ok(rows)
            };
            if table.indexes.iter().any(|index| index.unique) {
                let unique_preflight_started = Instant::now();
                if candidate_rows.is_none() {
                    candidate_rows = Some(materialize_candidates(self)?);
                }
                Self::validate_unique_indexes_for_rows(
                    &table,
                    candidate_rows.as_ref().expect("materialized above"),
                )?;
                if let Some(profile) = profile.as_mut() {
                    profile.unique_preflight_micros +=
                        unique_preflight_started.elapsed().as_micros();
                }
            }
            if !table.check_constraints.is_empty() {
                let check_preflight_started = Instant::now();
                if candidate_rows.is_none() {
                    candidate_rows = Some(materialize_candidates(self)?);
                }
                Self::validate_check_constraints_for_rows(
                    &table,
                    candidate_rows.as_ref().expect("materialized above"),
                )?;
                if let Some(profile) = profile.as_mut() {
                    profile.check_preflight_micros +=
                        check_preflight_started.elapsed().as_micros();
                }
            }
            if !table.foreign_keys.is_empty() {
                let foreign_key_preflight_started = Instant::now();
                if candidate_rows.is_none() {
                    candidate_rows = Some(materialize_candidates(self)?);
                }
                self.validate_foreign_keys_with_table_rows(
                    &table.name,
                    candidate_rows.as_ref().expect("materialized above"),
                    StorageVisibility {
                        read_txn_id: txn_id,
                    },
                )?;
                if let Some(profile) = profile.as_mut() {
                    profile.foreign_key_preflight_micros +=
                        foreign_key_preflight_started.elapsed().as_micros();
                }
            }
        }

        // Encode the new versions against the snapshot's `next_row_id` base (pure: does not bump
        // `relational_next_row_id` — `apply_delta` advances it by `rows_consumed`). These row keys
        // are exactly what the old `apply_insert` assigned because, under the still-serialized
        // commit, the snapshot is taken immediately before apply.
        let rows_consumed = new_rows.len() as u64;
        let mut inserted_rows = Vec::with_capacity(new_rows.len());
        for (offset, values) in new_rows.into_iter().enumerate() {
            let row_id = snapshot.next_row_id + offset as u64;
            let row_key = relational_row_key(&insert.table, row_id);
            inserted_rows.push((row_key, values));
        }
        // ELIDED-SKIP (lpb per-row-work cut): an elided (device-authoritative) table's apply
        // arm discards the host value-index entirely (engine_write_apply.rs), so computing the
        // per-row `ColumnValueKey`s here is pure waste on the sequencer's hot path — for a 64-row
        // batch this compute is ~40% of the per-row host cost. Skip it. SAFETY across a de-elision
        // race (elided at off-lock prepare, NOT elided by under-lock apply): both apply sites
        // recompute from the published catalog when they see an empty map for a non-empty insert
        // (`value_index_entries_for_deferred_apply`) — a real insert of >=1 row into a >=1-column
        // table always yields >=1 entry, so empty-and-non-empty-rows uniquely marks the deferral.
        let value_index_entries = if self.table_install_elided(&table.name) {
            BTreeMap::new()
        } else {
            relational_value_index_entries_for_rows(&table.columns, &inserted_rows)
        };

        let mut write_set = WriteSet::default();
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
            write_set.add_unique_slots(&table, values);
        }

        Ok(WriteDelta {
            write_set,
            rows_consumed,
            mutation: PreparedMutation::Insert {
                table: insert.table.clone(),
                inserted_rows,
                value_index_entries,
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
        let prefix = relational_key_prefix(&delete.table);
        // Resolve (tuple_id, key, row) for each matching version against this table's published
        // generation: tuple_id is what apply tombstones; key/row feed the write-set entries.
        let mut table_rows = self.read_state.mvcc.table_rows(&delete.table);
        let has_inbound_fks = catalog.relational_catalog.values().any(|candidate| {
            candidate
                .foreign_keys
                .iter()
                .any(|foreign_key| foreign_key.referenced_table == table.name)
        });
        // PHASE C slice 1 (ledger #1) + 1b: an Eq-bearing DELETE resolves its matches through the
        // VALUE INDEX — O(matches), not the O(table) seq_scan — and (1b) its inbound-FK validation
        // runs index-driven too. A SELF-REFERENCING FK falls back to the scan (its provider set
        // interleaves with the statement's own images). `None` (ineligible) -> the scan, unchanged.
        let self_referencing_fk = table
            .foreign_keys
            .iter()
            .any(|foreign_key| foreign_key.referenced_table == table.name);
        let index_resolved: Option<Vec<DmlResolvedMatch>> =
            if self_referencing_fk || !self.dml_value_index_resolve_enabled() {
                // A4e: the ladder is bypassed entirely -> an elided table must rehydrate before
                // the scan below reads the stale store.
                if self.table_install_elided(&table.name) {
                    // LOCK-AWARE + committed_seq stamps (audit f80f2350 FINDING B + the
                    // facade-seq poison find — see `visible_row_with_value`). Also closes the
                    // GAP-1 TOCTOU: a table eliding between the concurrent guard's check and
                    // this prepare now rehydrates under the commit lock, never a bare
                    // `with_table_mut` race.
                    self.rehydrate_elided_serialized(&table.name)?;
                    // A5 FLIP SI FIX: the scan below must read the FRESH generation.
                    table_rows = self.read_state.mvcc.table_rows(&table.name);
                }
                None
            } else {
                // RETIREMENT A2: the DEVICE resolve first (locate -> row-identity -> keyed fetch);
                // any decline falls to the value-index resolve (slice 1), then the scan below.
                match self.resolve_dml_matches_via_device(
                    &table,
                    &filter_groups,
                    visibility,
                    &table_rows,
                )? {
                    Some(matches) => Some(matches),
                    None => {
                        // A5 FLIP SI FIX (the SV6 elided-churn double-read): the device decline
                        // may have REHYDRATED — a COW publish of a FRESH host generation — and
                        // the view pinned above predates it. Falling back on the stale view
                        // resolves a STALE OLD IMAGE, whose visibility-blind tombstone locate
                        // then stamps an ALREADY-DEAD slot (exact-count 1 passes!) and leaves
                        // the truly-current version live forever; a stale-EMPTY view silently
                        // LOSES the update (0 matches). RE-PIN before every fallback.
                        table_rows = self.read_state.mvcc.table_rows(&table.name);
                        Self::resolve_dml_matches_via_value_index(
                            &table,
                            &table_rows,
                            &filter_groups,
                            visibility,
                            &prefix,
                        )?
                    }
                }
            };
        let index_arm = index_resolved.is_some();
        let deletes: Vec<DmlResolvedMatch> = match index_resolved {
            Some(matches) => matches,
            None => {
                let mut deletes: Vec<DmlResolvedMatch> = Vec::new();
                let mut cursor = table_rows
                    .store()
                    .seq_scan_open(visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;

                while let Some(tuple) = cursor.next() {
                    if !tuple.key.starts_with(&prefix) {
                        continue;
                    }
                    let row = decode_relational_row(&tuple.value, &table.columns)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    if filter_groups.iter().any(|filters| {
                        filters
                            .iter()
                            .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
                    }) {
                        deletes.push((tuple.tuple_id, tuple.key.clone(), row));
                    }
                }
                drop(cursor);
                deletes
            }
        };

        if has_inbound_fks {
            if index_arm {
                // PHASE C slice 1b: index-driven inbound-FK validation over the REMOVED provider
                // values only — O(deleted x FKs), replacing the O(table) survivor materialization
                // (and the validator's own O(all related tables) scans).
                let touched_keys: BTreeSet<String> =
                    deletes.iter().map(|(_, key, _)| key.clone()).collect();
                let removed: Vec<Vec<SqlValue>> =
                    deletes.iter().map(|(_, _, row)| row.clone()).collect();
                self.validate_dml_constraints_via_index(
                    &catalog,
                    &table,
                    &[],
                    &removed,
                    &touched_keys,
                    visibility,
                )?;
            } else {
                let deleted_ids = deletes
                    .iter()
                    .map(|(tuple_id, _, _)| *tuple_id)
                    .collect::<BTreeSet<_>>();
                let mut candidate_rows = Vec::new();
                let mut cursor = table_rows
                    .store()
                    .seq_scan_open(visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                while let Some(tuple) = cursor.next() {
                    if tuple.key.starts_with(&prefix) && !deleted_ids.contains(&tuple.tuple_id) {
                        candidate_rows.push(
                            decode_relational_row(&tuple.value, &table.columns)
                                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?,
                        );
                    }
                }
                drop(cursor);
                self.validate_foreign_keys_with_table_rows(
                    &table.name,
                    &candidate_rows,
                    visibility,
                )?;
            }
        }

        let mut write_set = WriteSet::default();
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
            write_set.add_unique_slots(&table, row);
        }

        Ok(WriteDelta {
            write_set,
            rows_consumed: 0,
            mutation: PreparedMutation::Delete {
                table: delete.table.clone(),
                tuple_ids,
                deleted_rows,
            },
        })
    }

    /// RETIREMENT A2: resolve a single-Eq DML statement's matches via the DEVICE — the cross-shard
    /// PK locate (hash+bloom over the resident shards, generation-consistent capture) -> the A1
    /// row-identity region (`row_id[slot]`) -> the derived host key -> ONE keyed fetch at the pinned
    /// visibility (tuple_id + the authoritative current version, until A4 retires the host chains)
    /// -> the FULL filter-group recheck. The host VALUE INDEX is not consulted — this is what lets
    /// A4 delete it. ELIGIBILITY (`None` -> the caller's fallback chain: value-index resolve, then
    /// the scan): flag ON; exactly ONE filter group with exactly one usable Int4 `Eq` (the locate is
    /// a single-needle unique-key probe); the locate must not decline (dup/oversize/invalid/absent
    /// shards); every hit must carry a STAMPED identity (sentinel/absent region = unknown lineage).
    /// The locate is PHYSICAL (a tombstoned row still hits): the keyed fetch at `visibility` is the
    /// authoritative filter — a host-invisible row resolves to no match, exactly as the scan would.
    pub(crate) fn resolve_dml_matches_via_device(
        &self,
        table: &RelationalTable,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        visibility: StorageVisibility,
        table_rows: &crate::resident_storage::TableRowsView,
    ) -> Result<Option<Vec<DmlResolvedMatch>>, EngineError> {
        let elided = self.table_install_elided(&table.name);
        // A4e: EVERY decline on an elided table must REHYDRATE first (sticky de-elision) — the
        // fallbacks below the ladder read the STALE store (elided commits never installed), so a
        // plain decline hands them wrong-empties or stale matches. This includes the SHAPE
        // early-exits (OR-groups / range-only / non-Int4): the stale value-index silently MISSES
        // elided-era rows.
        let rehydrate_if_elided = |engine: &Self| -> Result<(), EngineError> {
            if elided {
                // THE FACADE-SEQ POISON, final seam (found by the Date/Int2 gauntlet's
                // dup-date decline — the Int4-only ladders never lit this exit up): stamping
                // the reconcile at `visibility.read_txn_id` (the serialized path's FACADE txn
                // id, here observed 8 vs committed 5) made the reconciled elided-era rows
                // created_by=FUTURE -> invisible to the commit's own re-admit -> both rows
                // VANISHED from the device (k=5 bisect: 202 reconciled, store readable 200,
                // point500=0). `_serialized` stamps at the ENGINE's committed_seq and is
                // lock-aware, like every other prepare/probe seam post-audit.
                engine.rehydrate_elided_serialized(&table.name)?;
            }
            Ok(())
        };
        if !self.dml_device_resolve_enabled() {
            rehydrate_if_elided(self)?;
            return Ok(None);
        }
        // Single-needle shape: one group, one Eq on an Int4 column (the locate probes one key).
        let [group] = filter_groups else {
            rehydrate_if_elided(self)?;
            return Ok(None);
        };
        let mut eq: Option<(usize, i32)> = None;
        for (idx, op, value) in group {
            if *op == SelectFilterOp::Eq {
                // TYPE-COVERAGE track 2: any i32-section-typed Eq (Int4/Date/Int2, variant
                // agreeing with the column) can drive the locate.
                if let Some(needle) = table
                    .columns
                    .get(*idx)
                    .and_then(|column| crate::engine_residency::i32_section_needle(column.ty, value))
                {
                    eq = Some((*idx, needle));
                    break;
                }
            }
        }
        let Some((filter_idx, needle)) = eq else {
            rehydrate_if_elided(self)?;
            return Ok(None);
        };
        let Some(hits) =
            self.locate_resident_pk_via_shard_index_detailed(table, filter_idx, needle)
        else {
            rehydrate_if_elided(self)?;
            return Ok(None); // locate declined (dup / oversize / invalid / not resident)
        };
        let mut matches: Vec<DmlResolvedMatch> = Vec::new();
        let prefix = relational_key_prefix(&table.name);
        for hit in &hits {
            let Some(region) = &hit.row_id else {
                rehydrate_if_elided(self)?;
                return Ok(None); // identity-unknown lineage -> host path
            };
            // A device-read failure DECLINES to the host fallback (audit A2 finding 2) — the
            // value-index resolve never touches the device, so a transient CUDA error must not
            // fail a statement the fallback would serve; every sibling exit in this loop declines.
            let Ok(halves) = region.read_resident_i32_column(u64::from(hit.slot) * 8, 2) else {
                rehydrate_if_elided(self)?;
                return Ok(None);
            };
            let (Some(lo), Some(hi)) = (halves.first(), halves.get(1)) else {
                rehydrate_if_elided(self)?;
                return Ok(None);
            };
            let row_id = (*lo as u32 as u64) | ((*hi as u32 as u64) << 32);
            if row_id == u64::MAX {
                rehydrate_if_elided(self)?;
                return Ok(None);
            }
            let key = relational_row_key(&table.name, row_id);
            debug_assert!(key.starts_with(&prefix));
            // A4e: an ELIDED table's rows exist ONLY on the device — the host store is a stale
            // prefix. Materialize row + visibility from the hit (A4a); tuple_id is synthetic
            // (= row_id; apply skips host tombstoning for elided tables, nothing consumes it).
            if elided {
                match self.materialize_resident_row_via_hit(table, hit, visibility.read_txn_id) {
                    Some(Some(row)) => {
                        if filter_groups.iter().any(|filters| {
                            filters.iter().all(|(idx, op, value)| {
                                select_filter_matches(&row[*idx], *op, value)
                            })
                        }) {
                            matches.push((row_id, key, row));
                        }
                        continue;
                    }
                    Some(None) => continue, // not visible at this snapshot, like a fetch miss
                    None => {
                        rehydrate_if_elided(self)?;
                        return Ok(None);
                    }
                }
            }
            let fetched = table_rows
                .store()
                .tuple_fetch_by_key(&key, visibility)
                .map_err(|err: gpu_db_storage::StorageError| {
                    EngineError::ApplyFailed(err.to_string())
                })?;
            let Some(tuple) = fetched else {
                continue; // not visible at this snapshot (e.g. tombstoned): no match, like the scan
            };
            let row = decode_relational_row(&tuple.value, &table.columns)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            if filter_groups.iter().any(|filters| {
                filters
                    .iter()
                    .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
            }) {
                matches.push((tuple.tuple_id, key, row));
            }
        }
        matches.sort_by_key(|(tuple_id, _, _)| *tuple_id);
        // A LOGICAL row can hit in MULTIPLE shards: an SV5 update-append lands the new version in
        // the OPEN shard while the tombstoned old slot stays in its sealed shard — each shard's
        // hash is dup-free, so the visibility-blind locate returns BOTH slots. They carry the SAME
        // row_id -> same key -> same visible tuple; emitting it twice made prepare_update hand the
        // SV5 gate 2 matches for 1 slot -> commit fell back to invalidate+re-admit (caught by the
        // SV6 concurrent hammer). Version slots of one logical row are ONE match.
        matches.dedup_by_key(|(tuple_id, _, _)| *tuple_id);
        self.read_state
            .residency
            .dml_device_resolve_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(matches))
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
    ///                       failure) -> the caller must use the host fetch.
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
        if !hit.descriptor.resident_device_null_columns.is_empty() {
            return None; // raw i32 would read a stored NULL as 0 (the M3 decline discipline)
        }
        // Audit A4 F1, lifted by TYPE-COVERAGE track 2 (stages 1 + iii): every FIXED-WIDTH
        // section materializes with its CATALOG-derived variant — i32 via one u32/slot, i64 via
        // two (the 4-mod-8 discipline). Non-fixed-width types still decline to the host fetch.
        if table.columns.iter().any(|column| {
            !matches!(
                column.ty,
                crate::SqlType::Int4
                    | crate::SqlType::Date
                    | crate::SqlType::Int2
                    | crate::SqlType::Int8
                    | crate::SqlType::Timestamp
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
            match table.columns[idx].ty {
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

    /// RETIREMENT A3: the DEVICE-INDEX constraint probe — answers `any_visible_row_with_value`
    /// through the per-shard device PK-index locate + the A1 row-identity region instead of the
    /// host value_index. `Some(bool)` = authoritative answer; `None` = decline (the host ladder
    /// serves). COVERAGE argument (the FALSE answer is load-bearing — a missed row would wrongly
    /// PASS a unique/FK check): every visible row's CURRENT version occupies a live slot of some
    /// valid shard holding its current column value (residency is maintained or invalidated in the
    /// same serialized commit path), the locate probes EVERY shard's full-column hash (the bloom
    /// prune has no false negatives) and declines the WHOLE probe on any shard it cannot answer
    /// (dup/oversize/invalid/absent/unstamped) — so zero surviving hits proves no visible row
    /// carries the value. A physical hit whose visible version no longer matches (an SV5-tombstoned
    /// old slot) is neutralized by the fetch-at-visibility + structural recheck, exactly like the
    /// stale host-index entry it mirrors. NULL / non-Int4 values decline (host structural
    /// semantics, NULL == NULL, serve them).
    fn device_visible_row_with_value(
        &self,
        table: &RelationalTable,
        visibility: StorageVisibility,
        column_idx: usize,
        value: &SqlValue,
        exclude_keys: Option<&BTreeSet<String>>,
    ) -> Option<bool> {
        if !self.dml_device_validate_enabled() {
            return None;
        }
        // TYPE-COVERAGE track 2: i32-section needles (Int4/Date/Int2) probe with the exact
        // section encoding; anything else declines to the host ladder.
        let needle = crate::engine_residency::i32_section_needle(
            table.columns.get(column_idx)?.ty,
            value,
        )?;
        let hits = self.locate_resident_pk_via_shard_index_detailed(table, column_idx, needle)?;
        let mut answer = false;
        for hit in &hits {
            let region = hit.row_id.as_ref()?;
            // A device-read failure declines the whole probe (the A2 finding-2 discipline).
            let halves = region
                .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
                .ok()?;
            let (lo, hi) = (*halves.first()?, *halves.get(1)?);
            let row_id = (lo as u32 as u64) | ((hi as u32 as u64) << 32);
            if row_id == u64::MAX {
                return None; // unstamped slot: identity unknown -> host ladder
            }
            let key = relational_row_key(&table.name, row_id);
            if exclude_keys.is_some_and(|excluded| excluded.contains(&key)) {
                continue; // a row this statement touches: excluded, like the host probe
            }
            // RETIREMENT A4e: elided tables materialize from the device (the host store is
            // empty); visibility rides the regions (A4a).
            let row = if self.table_install_elided(&table.name) {
                match self.materialize_resident_row_via_hit(table, hit, visibility.read_txn_id) {
                    Some(Some(row)) => row,
                    Some(None) => continue,
                    None => return None,
                }
            } else {
                // Self-pinned view (the SI-fix discipline): loaded fresh per probe so a
                // mid-statement rehydration can never leave this fetch on a stale generation.
                let table_rows = self.read_state.mvcc.table_rows(&table.name);
                let fetched = table_rows
                    .store()
                    .tuple_fetch_by_key(&key, visibility)
                    .ok()?;
                let Some(tuple) = fetched else {
                    continue; // no visible version at this snapshot
                };
                decode_relational_row(&tuple.value, &table.columns).ok()?
            };
            if row[column_idx] == *value {
                answer = true;
                break;
            }
        }
        self.read_state
            .residency
            .dml_device_validate_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(answer)
    }

    /// RETIREMENT A3: the probe LADDER — device index first, host value_index on decline. Every
    /// validator probe goes through here; the ladder preserves slice-1b semantics exactly (the
    /// device arm answers only what it can prove, everything else falls through).
    ///
    /// SELF-PINNED VIEW (constrained-elision slice): the host arm loads the table's CURRENT
    /// generation itself — callers no longer thread a view. A probe earlier in the same statement
    /// may have rehydrated this (or another) table, COW-publishing a fresh generation; a
    /// caller-pinned view from before that publish is a stale prefix whose value_index would
    /// silently miss elided-era rows (the A5-flip SI-fix class, here a constraint bypass).
    /// Snapshot correctness is untouched: visibility rides `visibility.read_txn_id`.
    pub(crate) fn visible_row_with_value(
        &self,
        table: &RelationalTable,
        visibility: StorageVisibility,
        column_idx: usize,
        value: &SqlValue,
        exclude_keys: Option<&BTreeSet<String>>,
    ) -> Result<bool, EngineError> {
        if let Some(answer) =
            self.device_visible_row_with_value(table, visibility, column_idx, value, exclude_keys)
        {
            return Ok(answer);
        }
        // A4e + A5 FLIP SI FIX: a device decline on an ELIDED table must rehydrate BEFORE the
        // host probe (the stale value-index would answer from missing/old rows = a constraint
        // hole); the fresh pin below then reads the post-rehydration generation.
        //
        // LOCK DISCIPLINE (audit f80f2350 FINDING B): rehydration goes through the LOCK-AWARE
        // `rehydrate_elided_serialized` — this ladder runs OFF-LOCK in the concurrent INSERT
        // prepare (where the direct call raced `with_table_mut`'s clone-mutate-publish against
        // the sequencer: lost/torn generation publish) AND under the commit lock in the
        // serialized preflight / wave re-resolve (where the internal-read flag routes it to the
        // direct branch — the FINDING-A wraps). `_serialized` also stamps the reconcile at the
        // ENGINE's committed_seq, never the caller's visibility (the facade-seq poison find:
        // the preflight probes at the FACADE txn id; threading it into the reconcile stamped
        // store versions with future/foreign seqs -> "tuple not found" for later readers).
        let mut probe_visibility = visibility;
        if self.table_install_elided(&table.name) {
            self.rehydrate_elided_serialized(&table.name)?;
            // Audit FINDING C hardening: the reconcile stamps at committed_seq; a caller
            // boundary BELOW it (facade txn ids are decoupled from commit seqs) would read
            // `created_by > boundary` on the just-rehydrated committed rows = false MISS =
            // constraint bypass. Raise the probe boundary to cover the reconcile's stamps.
            probe_visibility.read_txn_id = probe_visibility.read_txn_id.max(self.committed_seq());
        }
        let fresh = self.read_state.mvcc.table_rows(&table.name);
        Self::any_visible_row_with_value(
            table,
            &fresh,
            probe_visibility,
            column_idx,
            value,
            exclude_keys,
        )
    }

    /// PHASE C slice 1b: does ANY VISIBLE row (optionally excluding `exclude_keys` — the rows this
    /// statement touches) carry `column_idx == value`? Resolves through the append-only value index
    /// (candidates) + the visibility fetch + a STRUCTURAL-equality recheck. Structural (`==`), NOT
    /// the 3VL matcher: the scan validators compare via `BTreeSet` membership, where NULL == NULL
    /// and same-column values share the column's coerced representation — this must match them.
    pub(crate) fn any_visible_row_with_value(
        table: &RelationalTable,
        table_rows: &crate::resident_storage::TableRowsView,
        visibility: StorageVisibility,
        column_idx: usize,
        value: &SqlValue,
        exclude_keys: Option<&BTreeSet<String>>,
    ) -> Result<bool, EngineError> {
        let mut keys = table_rows.index_keys(
            &table.columns[column_idx].name,
            &relational_index_value(value),
        );
        keys.sort();
        keys.dedup();
        for key in keys {
            if exclude_keys.is_some_and(|excluded| excluded.contains(&key)) {
                continue;
            }
            let fetched = table_rows
                .store()
                .tuple_fetch_by_key(&key, visibility)
                .map_err(|err: gpu_db_storage::StorageError| {
                    EngineError::ApplyFailed(err.to_string())
                })?;
            let Some(tuple) = fetched else {
                continue; // stale index entry: no visible version at this snapshot
            };
            let row = decode_relational_row(&tuple.value, &table.columns)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            if row[column_idx] == *value {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// PHASE C slice 1b: INDEX-DRIVEN constraint validation for an index-resolved DELETE/UPDATE —
    /// semantically identical to the scan validators (`validate_unique_indexes_for_rows` /
    /// `validate_check_constraints_for_rows` / `validate_foreign_keys_with_table_rows`) RESTRICTED
    /// to what the statement can affect: the untouched survivors were valid before it (every prior
    /// statement validated; ADD CHECK / ADD FK validate existing rows at DDL time), so only the NEW
    /// images (unique/check/outbound-FK) and the REMOVED provider values (inbound-FK) need work —
    /// O(rows touched x constraints) via the value indexes, replacing the validators' O(all related
    /// tables) survivor materializations. Validator ORDER mirrors the scan path (unique -> check ->
    /// FK) and the error messages are byte-identical. PRECONDITION (caller eligibility): `table` has
    /// NO self-referencing FK (its provider/consumer sets would interleave with the statement's own
    /// images — those tables fall back to the scan validators).
    ///
    /// DELETE passes empty `new_images` (unique/check/outbound sections no-op, exactly as the scan
    /// path never ran them for DELETE); UPDATE passes the post-assignment images.
    ///
    /// VIEW DISCIPLINE (constrained-elision slice, the A5-flip SI-fix class): the probes pin their
    /// OWN table view per probe (`visible_row_with_value` self-pins) — no caller-threaded view. A
    /// probe on an elided table may REHYDRATE (COW-publishing a fresh host generation); any view
    /// pinned before that publish is a stale prefix, and a later probe reading it would validate
    /// against MISSING elided-era rows (constraint bypass). MVCC makes the fresh pin sound: row
    /// visibility rides `visibility.read_txn_id`, not view recency.
    pub(crate) fn validate_dml_constraints_via_index(
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
        // 1. UNIQUE: in-batch duplicates among the new images (the scan validator's BTreeSet pass,
        //    NULLs collide) + each new value vs the UNTOUCHED visible rows via the index.
        for index in table.indexes.iter().filter(|index| index.unique) {
            let Some(column_idx) = table
                .columns
                .iter()
                .position(|column| column.name == index.column)
            else {
                continue;
            };
            let mut seen = BTreeSet::new();
            for row in new_images {
                if !seen.insert(row[column_idx].clone())
                    || self.visible_row_with_value(
                        table,
                        visibility,
                        column_idx,
                        &row[column_idx],
                        Some(touched_keys),
                    )?
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "duplicate key value violates unique index \"{}\"",
                        index.name
                    )));
                }
            }
        }
        // 2. CHECK: per-row on the new images (survivors passed at their own write; ADD CHECK
        //    validates existing rows at DDL time — the invariant the restriction rests on).
        Self::validate_check_constraints_for_rows(table, new_images)?;
        // 3. OUTBOUND FK (this table is the child): each new image's FK value must have a visible
        //    provider in the (untouched — no self-FK by precondition) parent table.
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
                if !self.visible_row_with_value(
                    parent,
                    visibility,
                    parent_idx,
                    &row[child_idx],
                    None,
                )? {
                    return Err(EngineError::ApplyFailed(format!(
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
            if child.name == table.name {
                continue; // self-FK excluded by the caller's eligibility
            }
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
                    if self.visible_row_with_value(
                        child,
                        visibility,
                        child_idx,
                        value,
                        None,
                    )? {
                        return Err(EngineError::ApplyFailed(format!(
                            "insert or update on table \"{}\" violates foreign key constraint \"{}\"",
                            child.name, foreign_key.name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// PHASE C slice 1 (ledger #1): resolve the rows a DELETE/UPDATE touches via the per-table
    /// equality VALUE-INDEX instead of the O(table) seq_scan + decode (MEASURED: single-row
    /// DELETE/UPDATE p50 80-88ms at 262k rows, LINEAR in table size — the write path's dominant
    /// cost; `examples/c1_prepare_split.rs`). ELIGIBILITY: every filter group carries at least one
    /// `Eq` filter, so the union over groups of `index_keys(column, value)` is a SUPERSET of the
    /// matching rows — the value-index is APPEND-ONLY (a stale entry names a row whose current
    /// visible version no longer matches), and staleness resolves exactly as the read-side equality
    /// fast-path resolves it: fetch each candidate key at the pinned `visibility`
    /// (`tuple_fetch_by_key`, O(log n + chain)) and RE-CHECK the FULL filter groups on the decoded
    /// row. Matches return sorted by `tuple_id` ascending — the seq_scan's iteration order
    /// (`versions.values()` is tuple_id-keyed) — so the produced WriteDelta is byte-identical to
    /// the scan path's. `None` = not eligible (no filters = full-table DML, or a range-only group)
    /// -> the caller runs the seq_scan (the oracle path, always correct).
    pub(crate) fn resolve_dml_matches_via_value_index(
        table: &RelationalTable,
        table_rows: &crate::resident_storage::TableRowsView,
        filter_groups: &[Vec<(usize, SelectFilterOp, SqlValue)>],
        visibility: StorageVisibility,
        prefix: &str,
    ) -> Result<Option<Vec<DmlResolvedMatch>>, EngineError> {
        if filter_groups.is_empty() {
            return Ok(None);
        }
        let mut candidate_keys: Vec<String> = Vec::new();
        for group in filter_groups {
            let Some((idx, _, value)) = group.iter().find(|(_, op, _)| *op == SelectFilterOp::Eq)
            else {
                return Ok(None); // a range-only group: the index cannot bound it -> scan
            };
            let column = &table.columns[*idx].name;
            candidate_keys.extend(table_rows.index_keys(column, &relational_index_value(value)));
        }
        // The append-only index records a key once per version that wrote the slot: dedup, and
        // keep only THIS table's keys (defensive — the per-table index is table-scoped already).
        candidate_keys.sort();
        candidate_keys.dedup();
        let mut matches: Vec<DmlResolvedMatch> = Vec::new();
        for key in candidate_keys {
            if !key.starts_with(prefix) {
                continue;
            }
            let fetched = table_rows
                .store()
                .tuple_fetch_by_key(&key, visibility)
                .map_err(|err: gpu_db_storage::StorageError| {
                    EngineError::ApplyFailed(err.to_string())
                })?;
            let Some(tuple) = fetched else {
                continue; // deleted / not visible at this snapshot (a stale index entry)
            };
            let row = decode_relational_row(&tuple.value, &table.columns)
                .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
            // The full predicate recheck: the candidate came from ONE Eq per group; the row must
            // satisfy SOME complete group (and a stale entry whose current version no longer
            // matches is excluded here).
            if filter_groups.iter().any(|filters| {
                filters
                    .iter()
                    .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
            }) {
                matches.push((tuple.tuple_id, key, row));
            }
        }
        // The seq_scan iterates tuple_id-ascending; match it so the delta bytes are identical.
        matches.sort_by_key(|(tuple_id, _, _)| *tuple_id);
        Ok(Some(matches))
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
            &table,
            &Delete {
                table: update.table.clone(),
                filter: update.filter.clone(),
                filters: update.filters.clone(),
                filter_groups: update.filter_groups.clone(),
            },
        )
        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let visibility = StorageVisibility {
            read_txn_id: txn_id,
        };
        let prefix = relational_key_prefix(&update.table);
        let mut updates = Vec::new();
        // SV5: OLD images (catalog order) captured before the assignments, PARALLEL to `updates`.
        let mut updated_old_rows: Vec<Vec<SqlValue>> = Vec::new();
        let mut candidate_rows = Vec::new();
        // Unique slots the OLD images RELEASE (prereq #2, Stage-4 audit). An UPDATE that changes a
        // unique column frees its old `(table, column, value)` slot; record those freed slots in the
        // write-set so a CONCURRENT insert/update reusing the freed value conflicts under
        // first-committer-wins — matching the DELETE path, which already records the released slots.
        // This is the conservative choice: it never admits a phantom unique duplicate across a
        // concurrent free+reuse (a slot-release left unrecorded could). A no-op-on-the-unique-column
        // UPDATE records the same slot as both released (old) and claimed (new) — harmless (the
        // write-set dedups to one slot), so an idempotent rewrite does not self-conflict.
        let mut released_unique_slots: Vec<UniqueIndexSlotKey> = Vec::new();
        let mut table_rows = self.read_state.mvcc.table_rows(&update.table);
        let constrained = table.indexes.iter().any(|index| index.unique)
            || !table.check_constraints.is_empty()
            || !table.foreign_keys.is_empty()
            || catalog.relational_catalog.values().any(|candidate| {
                candidate
                    .foreign_keys
                    .iter()
                    .any(|foreign_key| foreign_key.referenced_table == table.name)
            });
        // PHASE C slice 1 (ledger #1) + 1b: an Eq-bearing UPDATE resolves its matches through the
        // VALUE INDEX — O(matches), not the O(table) seq_scan — and (1b) a CONSTRAINED table's
        // validators run index-driven over the touched images (`validate_dml_constraints_via_index`)
        // instead of over the scan's survivor set. A SELF-REFERENCING FK falls back to the scan
        // (its provider set interleaves with the statement's own images).
        let self_referencing_fk = table
            .foreign_keys
            .iter()
            .any(|foreign_key| foreign_key.referenced_table == table.name);
        let index_resolved: Option<Vec<DmlResolvedMatch>> =
            if self_referencing_fk || !self.dml_value_index_resolve_enabled() {
                // A4e: the ladder is bypassed entirely -> an elided table must rehydrate before
                // the scan below reads the stale store.
                if self.table_install_elided(&table.name) {
                    // LOCK-AWARE + committed_seq stamps (audit f80f2350 FINDING B + the
                    // facade-seq poison find — see `visible_row_with_value`). Also closes the
                    // GAP-1 TOCTOU: a table eliding between the concurrent guard's check and
                    // this prepare now rehydrates under the commit lock, never a bare
                    // `with_table_mut` race.
                    self.rehydrate_elided_serialized(&table.name)?;
                    // A5 FLIP SI FIX: the scan below must read the FRESH generation.
                    table_rows = self.read_state.mvcc.table_rows(&table.name);
                }
                None
            } else {
                // RETIREMENT A2: the DEVICE resolve first (locate -> row-identity -> keyed fetch);
                // any decline falls to the value-index resolve (slice 1), then the scan below.
                match self.resolve_dml_matches_via_device(
                    &table,
                    &filter_groups,
                    visibility,
                    &table_rows,
                )? {
                    Some(matches) => Some(matches),
                    None => {
                        // A5 FLIP SI FIX (the SV6 elided-churn double-read): the device decline
                        // may have REHYDRATED — a COW publish of a FRESH host generation — and
                        // the view pinned above predates it. Falling back on the stale view
                        // resolves a STALE OLD IMAGE, whose visibility-blind tombstone locate
                        // then stamps an ALREADY-DEAD slot (exact-count 1 passes!) and leaves
                        // the truly-current version live forever; a stale-EMPTY view silently
                        // LOSES the update (0 matches). RE-PIN before every fallback.
                        table_rows = self.read_state.mvcc.table_rows(&table.name);
                        Self::resolve_dml_matches_via_value_index(
                            &table,
                            &table_rows,
                            &filter_groups,
                            visibility,
                            &prefix,
                        )?
                    }
                }
            };
        let index_arm = index_resolved.is_some();
        match index_resolved {
            Some(matches) => {
                for (tuple_id, key, mut row) in matches {
                    // Identical per-match processing to the scan arm below (old-image slots ->
                    // released; old image captured; assignments applied; install tuple pushed).
                    let mut old_slots = WriteSet::default();
                    old_slots.add_unique_slots(&table, &row);
                    released_unique_slots.append(&mut old_slots.unique_slots);
                    updated_old_rows.push(row.clone());
                    for (idx, value) in &assignments {
                        row[*idx] = value.clone();
                    }
                    updates.push((tuple_id, key, row));
                }
            }
            None => {
                let mut cursor = table_rows
                    .store()
                    .seq_scan_open(visibility)
                    .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;

                while let Some(tuple) = cursor.next() {
                    if !tuple.key.starts_with(&prefix) {
                        continue;
                    }
                    let mut row = decode_relational_row(&tuple.value, &table.columns)
                        .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
                    if filter_groups.iter().any(|filters| {
                        filters
                            .iter()
                            .all(|(idx, op, value)| select_filter_matches(&row[*idx], *op, value))
                    }) {
                        // Capture the old image's unique slots BEFORE the assignments overwrite them.
                        let mut old_slots = WriteSet::default();
                        old_slots.add_unique_slots(&table, &row);
                        released_unique_slots.append(&mut old_slots.unique_slots);
                        // SV5: capture the OLD image before the assignments overwrite it (parallel to
                        // `updates`).
                        updated_old_rows.push(row.clone());
                        for (idx, value) in &assignments {
                            row[*idx] = value.clone();
                        }
                        updates.push((tuple.tuple_id, tuple.key.clone(), row));
                    } else {
                        candidate_rows.push(row);
                    }
                }
                drop(cursor);
            }
        }

        if index_arm {
            // PHASE C slice 1b: index-driven validation over the touched images — O(touched x
            // constraints) via the value indexes, replacing the validators' survivor-set scans.
            // (`candidate_rows` is empty in this arm and unused.)
            if constrained {
                let touched_keys: BTreeSet<String> =
                    updates.iter().map(|(_, key, _)| key.clone()).collect();
                let new_images: Vec<Vec<SqlValue>> =
                    updates.iter().map(|(_, _, row)| row.clone()).collect();
                self.validate_dml_constraints_via_index(
                    &catalog,
                    &table,
                    &new_images,
                    &updated_old_rows,
                    &touched_keys,
                    visibility,
                )?;
            }
        } else {
            // PG constraint order: not-null (23502) BEFORE unique — over the post-assignment NEW
            // images only, O(touched). (The index arm gets the identical check inside
            // `validate_dml_constraints_via_index`.)
            Self::validate_primary_key_not_null(
                &table,
                updates.iter().map(|(_, _, row)| row.as_slice()),
            )?;
            if constrained {
                candidate_rows.extend(updates.iter().map(|(_, _, row)| row.clone()));
            }
            if table.indexes.iter().any(|index| index.unique) {
                Self::validate_unique_indexes_for_rows(&table, &candidate_rows)?;
            }
            if !table.check_constraints.is_empty() {
                Self::validate_check_constraints_for_rows(&table, &candidate_rows)?;
            }
            if !table.foreign_keys.is_empty()
                || catalog.relational_catalog.values().any(|candidate| {
                    candidate
                        .foreign_keys
                        .iter()
                        .any(|foreign_key| foreign_key.referenced_table == table.name)
                })
            {
                self.validate_foreign_keys_with_table_rows(
                    &table.name,
                    &candidate_rows,
                    visibility,
                )?;
            }
        }

        let updated_rows: Vec<(String, Vec<SqlValue>)> = updates
            .iter()
            .map(|(_, key, row)| (key.clone(), row.clone()))
            .collect();
        let value_index_entries =
            relational_value_index_entries_for_rows(&table.columns, &updated_rows);

        let mut write_set = WriteSet::default();
        for (_, key, row) in &updates {
            // An UPDATE tombstones the old version and installs a new one at the SAME row key,
            // so the row slot is written once.
            write_set.rows.push(RowWriteKey {
                table: update.table.clone(),
                row_key: key.clone(),
            });
            // The new image's unique-index slots are claimed by this txn.
            write_set.add_unique_slots(&table, row);
        }
        // The old images' RELEASED unique slots are also conflict points (prereq #2). Dedup so a
        // value carried unchanged through the UPDATE (same slot released and re-claimed) is recorded
        // once and never self-conflicts.
        write_set.unique_slots.append(&mut released_unique_slots);
        write_set.unique_slots.sort();
        write_set.unique_slots.dedup();

        // `updates` is already `(tuple_id, row_key, new_values)` — exactly the install shape.
        Ok(WriteDelta {
            write_set,
            rows_consumed: 0,
            mutation: PreparedMutation::Update {
                table: update.table.clone(),
                installs: updates,
                value_index_entries,
                updated_old_rows,
            },
        })
    }
}
