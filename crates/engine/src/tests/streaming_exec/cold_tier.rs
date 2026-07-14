use super::{gpu_available, select, ClassEntryDisabled};
use crate::Engine;
use gpu_db_sql::SqlValue;

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_replay_probe() {
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT, b INT)")
        .unwrap();
    const N: i32 = 100_000;
    for batch in 0..10 {
        let mut values = String::new();
        for j in 0..(N / 10) {
            let i = batch * (N / 10) + j;
            if j > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i}, {})", i * 2));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO big (a, b) VALUES {values}"))
            .unwrap();
    }
    e.set_relational_residency_budget_bytes(0, 65536);
    // warm-up
    let _ = e
        .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
        .unwrap();
    for run in 0..3 {
        let t = std::time::Instant::now();
        let c = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap();
        let t1 = t.elapsed().as_micros();
        let t = std::time::Instant::now();
        let s = e
            .execute_relational_select(&select("SELECT SUM(a) FROM big"))
            .unwrap();
        let t2 = t.elapsed().as_micros();
        assert_eq!(
            c.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(100_000)]]
        );
        assert_eq!(
            s.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(4_999_950_000i64)]]
        );
        eprintln!("COLDPROBE run={run} count_us={t1} sum_us={t2}");
        assert!(
            e.streaming_cold_hits() >= 1,
            "cold tier served the repeat reads"
        );
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_invalidates_on_write() {
    // S-E.6 THE correctness gate: the cold tier serves BYTE-REPLAYS of a prior build, so a WRITE must
    // invalidate it (the tuple-store generation Arc changes on every COW publish) — a stale hit would
    // serve pre-write data to post-write readers. Build -> hit -> INSERT -> fresh result -> hit again
    // -> DELETE -> fresh result.
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    // Build (scan + capture), then a hit (byte replay), identical results.
    let count = |e: &Engine| {
        e.execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap()
            .rows
            .clone()
            .into_boxed()
    };
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    let builds_after_first = e.streaming_cold_builds();
    assert!(
        builds_after_first >= 1,
        "the first streaming read installs the cold tier"
    );
    let hits_before = e.streaming_cold_hits();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert!(
        e.streaming_cold_hits() > hits_before,
        "the repeat read must SERVE FROM the cold tier"
    );
    // A different fold shape hits the SAME cache (chunks are fold-agnostic).
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    let expected_sum: i64 = (0..i64::from(N)).sum();
    assert_eq!(
        sum.rows.clone().into_boxed(),
        vec![vec![SqlValue::Int8(expected_sum)]]
    );

    // INSERT -> the generation Arc changes -> MISS -> fresh scan sees N+1 (a stale hit would say N).
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a) VALUES (100000)")
        .unwrap();
    assert_eq!(
        count(&e),
        vec![vec![SqlValue::Int8(i64::from(N) + 1)]],
        "a write must invalidate the cold tier (stale replay would return the OLD count)"
    );
    // 6c-1: the post-write read PATCHES the entry (O(delta)) instead of rebuilding it.
    assert_eq!(
        e.streaming_cold_builds(),
        builds_after_first,
        "the post-write read PATCHES — no fresh build"
    );
    assert!(
        e.streaming_cold_patches() >= 1,
        "the write was served by a PATCH"
    );
    // The rebuilt cache serves hits again...
    let hits_before = e.streaming_cold_hits();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N) + 1)]]);
    assert!(
        e.streaming_cold_hits() > hits_before,
        "rebuilt cache hits again"
    );
    // ...and a DELETE invalidates again.
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 100000")
        .unwrap();
    assert_eq!(
        count(&e),
        vec![vec![SqlValue::Int8(i64::from(N))]],
        "a DELETE must invalidate the cold tier"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_spills_and_replays_from_disk() {
    // S-E.6b: over the spill threshold the cold tier lives in an UNLINKED temp file, not host RAM.
    // Force a tiny threshold (test-only override) so the capture spills, then prove: the spill
    // counter fired; replays (aggregates AND the exact-order projection — chunk offsets must map
    // back byte-exactly) match the scan build; a write invalidates; a rebuilt spill serves again.
    crate::engine_streaming_exec::STREAMING_COLD_SPILL_THRESHOLD_TEST
        .store(1024, std::sync::atomic::Ordering::Relaxed);
    let result = std::panic::catch_unwind(|| {
        let mut e = Engine::new_local_cpu_oracle();
        let mut seq = 0u64;
        if !gpu_available(&mut e, &mut seq) {
            return;
        }
        seq += 1;
        e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
        const N: i32 = 1500;
        let mut values = String::new();
        for i in 0..N {
            if i > 0 {
                values.push(',');
            }
            values.push_str(&format!("({i})"));
        }
        seq += 1;
        e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
            .unwrap();
        e.set_relational_residency_budget_bytes(0, 4096);

        // Build: 6000B of payloads > the 1KB forced threshold -> the capture SPILLS.
        let count = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap();
        assert_eq!(
            count.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(i64::from(N))]]
        );
        assert!(
            e.streaming_cold_spills() >= 1,
            "the capture must have SPILLED (threshold forced to 1KB)"
        );
        // Aggregate replay from disk.
        let hits_before = e.streaming_cold_hits();
        let sum = e
            .execute_relational_select(&select("SELECT SUM(a) FROM big"))
            .unwrap();
        let expected_sum: i64 = (0..i64::from(N)).sum();
        assert_eq!(
            sum.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(expected_sum)]]
        );
        assert!(
            e.streaming_cold_hits() > hits_before,
            "spilled replay served the SUM"
        );
        // EXACT-ORDER projection replay: chunk offsets must round-trip byte-exactly (a swapped or
        // misaligned positional read would reorder or corrupt rows).
        let rows = e
            .execute_relational_select(&select("SELECT a FROM big WHERE a >= 1000"))
            .unwrap();
        let expected: Vec<Vec<SqlValue>> = (1000..N).map(|i| vec![SqlValue::Int4(i)]).collect();
        assert_eq!(
            rows.rows.clone().into_boxed(),
            expected,
            "spilled projection byte-exact"
        );

        // A write invalidates the spilled entry (generation change), and the rebuild re-spills.
        let spills_before = e.streaming_cold_spills();
        seq += 1;
        e.execute_text(seq, "INSERT INTO big (a) VALUES (100000)")
            .unwrap();
        let count = e
            .execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap();
        assert_eq!(
            count.rows.clone().into_boxed(),
            vec![vec![SqlValue::Int8(i64::from(N) + 1)]],
            "post-write count fresh (stale spilled replay would say N)"
        );
        assert!(
            e.streaming_cold_spills() > spills_before,
            "the rebuild re-spilled"
        );
    });
    crate::engine_streaming_exec::STREAMING_COLD_SPILL_THRESHOLD_TEST
        .store(0, std::sync::atomic::Ordering::Relaxed);
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_patches_deltas_chunk_granular() {
    // These gates exercise the STORE-DRIVEN patch/stamp machinery (live for non-class tables);
    // without this the table class-enters mid-test and the semantics legitimately change.
    let _class_off = ClassEntryDisabled::new();
    // 6c-1: a write PATCHES the cold entry at CHUNK granularity instead of discarding it — the
    // O(delta) maintenance win. INSERT = a pure tail append (zero dirty chunks rebuilt); a one-row
    // DELETE rebuilds EXACTLY ONE dirty chunk (of several); every aggregate stays exact through
    // the patches. Composes: COW chain identity (imbl diff), effective-range tiling, the rollover
    // tail, the S-E.6a settled-boundary install.
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    let count = |e: &Engine| {
        e.execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap()
            .rows
            .clone()
            .into_boxed()
    };
    // Build (~3 chunks of 512 rows).
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert_eq!(
        e.streaming_cold_patches(),
        0,
        "first read is a build, not a patch"
    );

    // INSERT -> tail-append patch: zero dirty chunks rebuilt.
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a) VALUES (100000)")
        .unwrap();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N) + 1)]]);
    assert_eq!(e.streaming_cold_patches(), 1, "the write PATCHED the entry");
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        0,
        "an INSERT is a pure TAIL append — no existing chunk rebuilds"
    );

    // One-row DELETE inside the FIRST chunk -> P2: a SIDECAR STAMP, zero rebuilds (the chunk's
    // bytes stay; the tombstone masks the row in-kernel at replay).
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 3").unwrap();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert_eq!(e.streaming_cold_patches(), 2);
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        0,
        "a pure one-row DELETE STAMPS its chunk's sidecar — nothing rebuilds (P2)"
    );
    assert_eq!(
        e.streaming_cold_stamps(),
        1,
        "exactly the deleted row is tombstone-stamped"
    );

    // Aggregate exactness through the patched chunks (SUM over the survivors + the tail row).
    let expected_sum: i64 = (0..i64::from(N)).sum::<i64>() - 3 + 100000;
    let sum = e
        .execute_relational_select(&select("SELECT SUM(a) FROM big"))
        .unwrap();
    assert_eq!(
        sum.rows.clone().into_boxed(),
        vec![vec![SqlValue::Int8(expected_sum)]]
    );

    // The patched entry serves plain hits again (no further patches).
    let patches = e.streaming_cold_patches();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert_eq!(
        e.streaming_cold_patches(),
        patches,
        "clean hit after the patch"
    );

    // F6 (audit): the EMPTY-table sentinel -> INSERT patch path (the (1,0) sentinel chunk's hi=0
    // routes every new id to the tail; no panic, exact results).
    seq += 1;
    e.execute_text(seq, "CREATE TABLE hollow (a INT)").unwrap();
    let hollow_count = |e: &Engine| {
        e.execute_relational_select(&select("SELECT COUNT(*) FROM hollow"))
            .unwrap()
            .rows
            .clone()
            .into_boxed()
    };
    assert_eq!(
        hollow_count(&e),
        vec![vec![SqlValue::Int8(0)]],
        "empty build"
    );
    seq += 1;
    e.execute_text(seq, "INSERT INTO hollow (a) VALUES (1), (2)")
        .unwrap();
    assert_eq!(
        hollow_count(&e),
        vec![vec![SqlValue::Int8(2)]],
        "insert-into-empty patches (sentinel -> tail) without a panic"
    );
}

#[test]
#[ignore = "requires a local NVIDIA driver and GPU"]
fn gpu_streaming_cold_tier_eager_commit_maintenance() {
    // These gates exercise the STORE-DRIVEN patch/stamp machinery (live for non-class tables);
    // without this the table class-enters mid-test and the semantics legitimately change.
    let _class_off = ClassEntryDisabled::new();
    // 6c-3: a COMMIT eagerly patches the table's cold entry (best-effort, under the held commit
    // mutex, O(delta)) — the patch counter moves AT COMMIT TIME, before any read; the next read is
    // a CLEAN HIT (no read-time patch). Reads never pay the maintenance.
    let mut e = Engine::new_local_cpu_oracle();
    let mut seq = 0u64;
    if !gpu_available(&mut e, &mut seq) {
        return;
    }
    seq += 1;
    e.execute_text(seq, "CREATE TABLE big (a INT)").unwrap();
    const N: i32 = 1500;
    let mut values = String::new();
    for i in 0..N {
        if i > 0 {
            values.push(',');
        }
        values.push_str(&format!("({i})"));
    }
    seq += 1;
    e.execute_text(seq, &format!("INSERT INTO big (a) VALUES {values}"))
        .unwrap();
    e.set_relational_residency_budget_bytes(0, 4096);

    // First read builds the entry.
    let count = |e: &Engine| {
        e.execute_relational_select(&select("SELECT COUNT(*) FROM big"))
            .unwrap()
            .rows
            .clone()
            .into_boxed()
    };
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
    assert_eq!(e.streaming_cold_patches(), 0);

    // THE COMMIT ITSELF patches — no read in between.
    seq += 1;
    e.execute_text(seq, "INSERT INTO big (a) VALUES (100000)")
        .unwrap();
    assert_eq!(
        e.streaming_cold_patches(),
        1,
        "the COMMIT must have eagerly patched the cold entry (before any read)"
    );

    // The next read is a CLEAN HIT: correct result, no read-time patch.
    let hits_before = e.streaming_cold_hits();
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N) + 1)]]);
    assert_eq!(
        e.streaming_cold_patches(),
        1,
        "no read-time patch — the read was a clean hit"
    );
    assert!(e.streaming_cold_hits() > hits_before);

    // A DELETE commit patches eagerly too — P2: a SIDECAR STAMP at commit, zero rebuilds.
    let rebuilt_before = e.streaming_cold_chunks_rebuilt();
    let stamps_before = e.streaming_cold_stamps();
    seq += 1;
    e.execute_text(seq, "DELETE FROM big WHERE a = 3").unwrap();
    assert_eq!(
        e.streaming_cold_patches(),
        2,
        "the DELETE commit patched eagerly"
    );
    assert_eq!(
        e.streaming_cold_chunks_rebuilt(),
        rebuilt_before,
        "the eager DELETE patch STAMPS — no chunk rebuild at commit (P2)"
    );
    assert_eq!(
        e.streaming_cold_stamps(),
        stamps_before + 1,
        "the commit stamped exactly the deleted row"
    );
    assert_eq!(count(&e), vec![vec![SqlValue::Int8(i64::from(N))]]);
}
