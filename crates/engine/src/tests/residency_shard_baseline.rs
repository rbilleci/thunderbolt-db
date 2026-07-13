/// Billions-of-rows S-d1: admitting a table as a (single dense) SEGMENTED shard — `shard_residency`
/// flag ON, routed through the sharded resident read path — must produce byte-identical reads to the
/// single capacity-padded unified buffer (flag OFF). NON-VACUITY: with the flag ON the table lands in
/// `residency.shards` (the sharded route), and NOT with the flag OFF (the single-buffer route).
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_residency_admit_reads_match_single_buffer() {
    let run = |shard: bool| -> (RowBlock, RowBlock, RowBlock, bool) {
        let mut e = Engine::new_local();
        // This gate compares two EXPLICIT post-load admissions. Keep fixture construction in the
        // host store so the production auto-admit/elision flip cannot make an earlier INSERT batch
        // device-authoritative before the comparison snapshot is requested.
        e.set_auto_admit_on_commit(false);
        e.set_shard_residency_enabled(shard);
        e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        let mut txn = 2u64;
        let mut id = 0i64;
        while id < 1000 {
            let mut vals = String::new();
            for _ in 0..200 {
                if id >= 1000 {
                    break;
                }
                if !vals.is_empty() {
                    vals.push(',');
                }
                vals.push_str(&format!("({id}, {})", id * 10));
                id += 1;
            }
            e.execute_text(
                txn,
                &format!("INSERT INTO accounts (id, balance) VALUES {vals}"),
            )
            .unwrap();
            txn += 1;
        }
        e.populate_relational_residency_snapshot("accounts")
            .unwrap();
        let in_shards = e
            .read_state
            .residency
            .shards
            .load()
            .get("accounts")
            .is_some_and(|s| !s.is_empty());
        let sel = |sql: &str| -> RowBlock { e.execute_relational_select_text(sql).unwrap().rows };
        (
            sel("SELECT id, balance FROM accounts WHERE id = 137"),
            sel("SELECT id, balance FROM accounts ORDER BY id"),
            sel("SELECT COUNT(*) FROM accounts"),
            in_shards,
        )
    };
    let (on_pt, on_scan, on_cnt, on_in_shards) = run(true);
    let (off_pt, off_scan, off_cnt, off_in_shards) = run(false);
    // NON-VACUITY: flag ON took the sharded route; flag OFF took the single buffer.
    assert!(
        on_in_shards,
        "flag ON must admit the table as a shard (the sharded read route)"
    );
    assert!(
        !off_in_shards,
        "flag OFF must use the single buffer (no shard)"
    );
    // point-lookup + COUNT(*) ARE served by the sharded route (the non-vacuous sharded gates); the
    // ORDER BY scan on a shard table takes the CPU fallback (a shard has no `snapshots` entry, so the
    // gpu-sortable gate declines) — kept as a correctness check (CPU shard path == GPU single buffer).
    assert_eq!(
        on_pt, off_pt,
        "sharded point lookup == single-buffer baseline"
    );
    assert_eq!(
        on_scan, off_scan,
        "scan (CPU fallback) == single-buffer baseline"
    );
    assert_eq!(
        on_cnt, off_cnt,
        "sharded COUNT(*) == single-buffer baseline"
    );
    assert_eq!(on_scan.len(), 1000, "all 1000 rows present");

    // S-d2a non-vacuity: the OPEN shard carries capacity HEADROOM (capacity > row_count), and the
    // capacity-aware sharded recompaction above addressed it correctly (the reads matched the
    // baseline — a dense-stride recompaction would have copied the wrong column slice and diverged).
    {
        let mut e = Engine::new_local();
        e.set_shard_residency_enabled(true);
        e.execute_text(1, "CREATE TABLE t (id INT, balance INT)")
            .unwrap();
        e.execute_text(2, "INSERT INTO t (id, balance) VALUES (1,10),(2,20),(3,30)")
            .unwrap();
        e.populate_relational_residency_snapshot("t").unwrap();
        let shards = e.read_state.residency.shards.load();
        let shard = &shards.get("t").expect("table admitted as a shard")[0];
        assert_eq!(shard.row_count, 3, "live row count");
        assert!(
            shard.capacity > shard.row_count,
            "open shard must carry headroom: capacity {} > row_count {}",
            shard.capacity,
            shard.row_count
        );
    }
}

/// Audit (S-d1) fix: a runtime `shard_residency` flag FLIP + re-admit must clear the OPPOSITE residency
/// representation, so a stale shard can't shadow a fresh snapshot (the read route checks shards FIRST →
/// it would serve wrong rows) and a stale snapshot can't shadow fresh shards. Verifies the map state
/// after each flip — WITHOUT the clears the stale cell persists, so the `!in_*` asserts fail.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_residency_flag_flip_clears_opposite_representation() {
    let mut e = Engine::new_local();
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    e.execute_text(
        2,
        "INSERT INTO accounts (id, balance) VALUES (1, 100), (2, 200), (3, 300)",
    )
    .unwrap();
    let in_shards = |e: &Engine| {
        e.read_state
            .residency
            .shards
            .load()
            .get("accounts")
            .is_some_and(|s| !s.is_empty())
    };
    let in_snaps = |e: &Engine| {
        e.read_state
            .residency
            .snapshots
            .load()
            .get("accounts")
            .is_some()
    };

    // OFF -> single buffer.
    e.set_shard_residency_enabled(false);
    e.populate_relational_residency_snapshot("accounts")
        .unwrap();
    assert!(
        in_snaps(&e) && !in_shards(&e),
        "OFF admits the single buffer"
    );
    // Flip ON -> shard; the stale snapshot must be cleared.
    e.set_shard_residency_enabled(true);
    e.populate_relational_residency_snapshot("accounts")
        .unwrap();
    assert!(
        in_shards(&e) && !in_snaps(&e),
        "OFF->ON re-admit must clear the stale snapshot"
    );
    // Flip OFF -> single buffer; the stale shard must be cleared (the wrong-rows footgun).
    e.set_shard_residency_enabled(false);
    e.populate_relational_residency_snapshot("accounts")
        .unwrap();
    assert!(
        in_snaps(&e) && !in_shards(&e),
        "ON->OFF re-admit must clear the stale shard (else it shadows the fresh snapshot)"
    );
    // Reads are correct after the flips.
    let r = e
        .execute_relational_select_text("SELECT id, balance FROM accounts WHERE id = 2")
        .unwrap()
        .rows;
    assert_eq!(r.len(), 1, "id=2 present");
    assert_eq!(r.row(0), &[SqlValue::Int4(2), SqlValue::Int4(200)]);
}

/// S-d2b: with the shard flag ON, a committed INSERT appends IN PLACE into the resident OPEN shard's
/// headroom (no whole-table re-admit) — the shard-path analog of 1b-ii-c/d. Across 50 in-headroom
/// commits the open_shard_append_hits counter advances by exactly 50, the table stays ONE shard (in
/// place, not rollover/re-admit), the open shard's row_count grows, and a point lookup + COUNT over the
/// sharded route return the appended rows. (Counter +50 vs 0 distinguishes append from re-admit.)
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_open_append_in_place_reads_correct() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    // Batch-admit 300 rows as ONE shard with headroom (default 4M target -> the admit builds a
    // capacity-1024 open shard holding 300 rows), so the next 50 single-row commits append IN PLACE
    // without a rollover (rollover is exercised by shard_open_append_rolls_over_and_reads_correct).
    let base: Vec<String> = (0..300_i64).map(|i| format!("({i}, {})", i * 10)).collect();
    e.execute_text(
        2,
        &format!(
            "INSERT INTO accounts (id, balance) VALUES {}",
            base.join(",")
        ),
    )
    .unwrap();
    let shard_state = |e: &Engine| -> (usize, usize) {
        let shards = e.read_state.residency.shards.load();
        let s = shards.get("accounts").expect("shard-resident");
        (s.len(), s.last().unwrap().row_count)
    };
    let count_before = shard_state(&e).0;
    let hits_before = e.open_shard_append_hits();
    for i in 300..350_i64 {
        e.execute_text(
            (i as u64) + 2,
            &format!(
                "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                i * 10
            ),
        )
        .unwrap();
    }
    let (count_after, open_row_count) = shard_state(&e);
    // NON-VACUITY: the 50 in-headroom commits appended IN PLACE to the open shard (counter +50, vs 0
    // for a re-admit), the shard count is unchanged (no rollover/re-admit), and row_count grew.
    assert_eq!(
        e.open_shard_append_hits() - hits_before,
        50,
        "all 50 in-headroom commits must append in place to the open shard (re-admit would give 0)"
    );
    assert_eq!(
        count_after, count_before,
        "no rollover/re-admit -> shard count unchanged"
    );
    assert_eq!(
        open_row_count, 350,
        "the open shard's row_count grew to 350 via in-place append"
    );

    // Reads over the sharded route include the in-place-appended rows.
    let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
    let pt = sel("SELECT id, balance FROM accounts WHERE id = 342");
    assert_eq!(
        pt.len(),
        1,
        "appended key 342 present via the sharded route"
    );
    assert_eq!(pt.row(0), &[SqlValue::Int4(342), SqlValue::Int4(3420)]);
    assert_eq!(
        sel("SELECT COUNT(*) FROM accounts").row(0),
        &[SqlValue::Int8(350)],
        "COUNT(*) sees the appended rows"
    );
}

/// S-d2c: with the shard flag ON and a small shard-size target, a table built via committed INSERTs
/// SEALS the full open shard and ROLLS OVER into a fresh one — growing as MULTIPLE bounded shards
/// instead of ever re-admitting the whole table (this is what removes the ~536M single-buffer cap). The
/// recompaction reads correctly across the sealed + open shards. NON-VACUITY: rollover produced multiple
/// shards (a broken rollover -> re-admit -> ONE dense shard -> the assert fails); the per-shard row_counts
/// sum to the table total; point lookups across different shards + COUNT(*) are correct.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_open_append_rolls_over_and_reads_correct() {
    let e = Engine::new_local();
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64); // small -> roll over every ~64 rows
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    for i in 0..200_i64 {
        e.execute_text(
            (i as u64) + 2,
            &format!(
                "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                i * 10
            ),
        )
        .unwrap();
    }
    let (shard_count, total_rows) = {
        let shards = e.read_state.residency.shards.load();
        let s = shards.get("accounts").expect("shard-resident");
        (s.len(), s.iter().map(|sh| sh.row_count).sum::<usize>())
    };
    // NON-VACUITY: rollover produced MULTIPLE bounded shards (200 rows / 64 target -> >= 3); a broken
    // rollover would re-admit into ONE dense shard.
    assert!(
        shard_count >= 3,
        "rollover must grow the table as multiple shards (got {shard_count})"
    );
    assert_eq!(
        total_rows, 200,
        "per-shard row_counts sum to the table total"
    );

    // Reads recompact across all shards: point lookups landing in different shards + COUNT(*).
    let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
    for id in [5_i64, 70, 137, 199] {
        let r = sel(&format!("SELECT id, balance FROM accounts WHERE id = {id}"));
        assert_eq!(r.len(), 1, "id={id} present across shards");
        assert_eq!(
            r.row(0),
            &[SqlValue::Int4(id as i32), SqlValue::Int4((id * 10) as i32)],
            "id={id} value correct across shards"
        );
    }
    assert_eq!(
        sel("SELECT COUNT(*) FROM accounts").row(0),
        &[SqlValue::Int8(200)],
        "COUNT(*) across all shards"
    );
    assert_eq!(
        sel("SELECT id, balance FROM accounts").len(),
        200,
        "scan recompacts all shards"
    );
}

/// S-d3: per-shard zone maps (min/max) PRUNE the sharded read. A table grown as several bounded shards
/// by ORDERED committed INSERTs gives each shard a disjoint key range, so a point lookup gathers ~1
/// shard instead of recompacting all of them — O(1), not O(num_shards).
///
/// NON-VACUITY / anti-sabotage, all keyed off the `sharded_shards_gathered` telemetry (output equality
/// alone can't see a skipped shard — a pruned shard holds no matching rows either way):
///  1. the table really is MULTIPLE shards (>= 3), so pruning to 1 is a real reduction;
///  2. a point lookup gathers EXACTLY 1 shard (a broken prune keeping all would read `shard_count`);
///  3. a full scan (no predicate) gathers ALL shards (proves the counter isn't hardwired to 1, and that
///     no-filter never prunes);
///  4. EVERY key across shard boundaries — including the MAX key, which was appended IN PLACE into the
///     open shard AFTER its rollover-admit — still reads its row (a broken append-time zone-map
///     extension would leave the open shard's max stale and WRONGLY prune the just-appended key -> 0 rows);
///  5. an absent key (beyond every range) prunes to the keep-one fallback (1 shard) and returns empty.
#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn shard_zone_map_prunes_point_lookup() {
    let e = Engine::new_local();
    // THE FLIP: this test exercises the re-admit/scan-layer semantics — pin the pre-flip
    // configuration it tests (each flag remains a supported kill switch).
    e.set_shard_index_probe_enabled(false);
    e.set_shard_batched_point_read_enabled(false);
    e.set_shard_residency_enabled(true);
    e.set_auto_admit_on_commit(true);
    e.set_shard_size_target(64); // small -> several shards with disjoint ascending key ranges
    e.execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
        .unwrap();
    for i in 0..200_i64 {
        e.execute_text(
            (i as u64) + 2,
            &format!(
                "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                i * 10
            ),
        )
        .unwrap();
    }
    let shard_count = {
        let shards = e.read_state.residency.shards.load();
        shards.get("accounts").expect("shard-resident").len()
    };
    // (1) multiple shards, so a prune to 1 is meaningful.
    assert!(
        shard_count >= 3,
        "need several shards for pruning to matter (got {shard_count})"
    );

    let sel = |sql: &str| e.execute_relational_select_text(sql).unwrap().rows;
    let gathered = |e: &Engine| e.sharded_shards_gathered();

    // (2) a point lookup landing inside one shard's range gathers EXACTLY one shard.
    let before = gathered(&e);
    let pt = sel("SELECT id, balance FROM accounts WHERE id = 137");
    let one = gathered(&e) - before;
    assert_eq!(pt.len(), 1, "id=137 present");
    assert_eq!(pt.row(0), &[SqlValue::Int4(137), SqlValue::Int4(1370)]);
    assert_eq!(
        one, 1,
        "zone-map prune must gather exactly ONE shard for a point lookup (got {one} of {shard_count})"
    );

    // (3) an unpredicated SUM (shape `sharded_int4_scalar_aggregate`) routes through the SAME
    //     recompaction function but carries NO predicate at all, so it never prunes and gathers ALL
    //     shards — proving the counter reaches `shard_count` (it is not pinned to 1) and that pruning
    //     is precisely what cut the equality lookup to one shard. (An unpredicated COUNT(*) no longer
    //     gathers anything: the FLIP metadata fast path answers it from `sum(shard.row_count)` on a
    //     version-free table — asserted as the 0-gather control below.)
    let before = gathered(&e);
    let total = sel("SELECT SUM(balance) FROM accounts");
    let scanned = gathered(&e) - before;
    assert_eq!(
        total.row(0),
        &[SqlValue::Int8((0..200_i64).map(|i| i * 10).sum())],
        "SUM(balance) across all shards"
    );
    assert_eq!(
        scanned, shard_count as u64,
        "an unpruned SUM must gather ALL shards, proving the counter isn't pinned to 1"
    );
    // FLIP metadata COUNT: version-free unpredicated COUNT(*) is served from shard metadata — exact
    // AND gather-free. Sabotage: route it through the recompaction instead and the 0 becomes
    // shard_count (or break row_count accounting and the value diverges).
    let before = gathered(&e);
    let cnt = sel("SELECT COUNT(*) FROM accounts");
    let scanned = gathered(&e) - before;
    assert_eq!(
        cnt.row(0),
        &[SqlValue::Int8(200)],
        "COUNT(*) across all shards"
    );
    assert_eq!(
        scanned, 0,
        "version-free COUNT(*) is metadata-served (no shard gathered)"
    );

    // (4) every boundary + the MAX key (appended in place into the open shard) still reads correctly,
    //     each gathering exactly one shard -> no key is ever wrongly pruned.
    for id in [0_i64, 63, 64, 127, 128, 191, 199] {
        let before = gathered(&e);
        let r = sel(&format!("SELECT id, balance FROM accounts WHERE id = {id}"));
        let g = gathered(&e) - before;
        assert_eq!(r.len(), 1, "id={id} must be found (never wrongly pruned)");
        assert_eq!(
            r.row(0),
            &[SqlValue::Int4(id as i32), SqlValue::Int4((id * 10) as i32)],
            "id={id} value correct after pruning"
        );
        assert_eq!(g, 1, "id={id} prunes to exactly one shard (got {g})");
    }

    // (5) a key beyond every shard's range prunes to the keep-one fallback and returns empty.
    let before = gathered(&e);
    let miss = sel("SELECT id, balance FROM accounts WHERE id = 100000");
    let g = gathered(&e) - before;
    assert_eq!(miss.len(), 0, "absent key returns no rows");
    assert_eq!(
        g, 1,
        "an out-of-range needle prunes to the single keep-one fallback shard"
    );
}
