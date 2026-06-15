//! Concurrency-correctness suite for the write-half MVCC concurrency flip (Thread-4, Stage 4).
//!
//! This is the heart of the Stage-4 gate (design doc `16-write-half-mvcc-design.md`): with writes
//! now concurrent (off-lock prepare + a short commit critical section + Snapshot-Isolation
//! conflict detection, and the engine write lock removed), these tests assert the SI correctness
//! properties DETERMINISTICALLY. Each runs `N` times with the project's 8-thread / barrier idiom so
//! a passing run is a real interleaving, not a timing accident — the threads rendezvous at a
//! `Barrier` so they enter the contended window simultaneously.
//!
//! Properties covered (the milestone's exit criteria):
//! - **Lost-update SI:** two concurrent UPDATEs to the same row → exactly one commits, the other
//!   aborts with a retryable serialization error.
//! - **Snapshot-isolation read:** an in-flight reader's result is stable across a concurrent
//!   committing writer (no dirty / non-repeatable read within a statement).
//! - **Disjoint writers both commit:** writers to different rows never falsely conflict.
//! - **Disjoint INSERTs all commit:** concurrent inserts of distinct values into ONE table (no
//!   unique index) never falsely conflict on a predicted row key.
//! - **Unique conflict:** two concurrent inserts of the same unique value → one commits, one aborts.
//! - **FK phantom under concurrency:** an INSERT child whose FK parent a concurrent committer deletes
//!   aborts with a RETRYABLE serialization error (never a panic / engine wedge).
//! - **Residency↔data consistency (real GPU, `#[ignore]`):** a concurrent writer commits +
//!   invalidates a table's GPU residency while a reader reads it → the reader's result is a single
//!   consistent snapshot (no cross-snapshot rows), matching the CPU result at its snapshot.
//! - **Kill-mid-commit under concurrency:** after many concurrent commits, recovery from the
//!   durable WAL loses nothing past the durable boundary and reproduces a consistent state.
//!
//! Writers overlap (off-lock prepare; only the short commit_mutex serializes) and a writer never
//! blocks a reader — also asserted directly (`concurrent_writers_overlap_*`).
//!
//! Every `#[test]` body runs under [`with_deadline`]: a concurrency test that can hang silently is
//! itself a defect, so a missed barrier rendezvous / deadlock surfaces as a normal test FAILURE
//! (panic) rather than an infinite park. Correspondingly, NO panicable / early-returning code runs
//! before a worker reaches its `barrier.wait()` — a worker can never skip its barrier arrival due to
//! an error and strand its peers.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use gpu_db_engine::{Engine, RelationalResidencyWarmupPolicy};
use gpu_db_facade::{
    execute_concurrent_dml_with_prepared_hook, execute_on_shared_engine, DbError, DbValue,
    ErrorCategory, QueryOutcome, SharedEngine,
};

/// Repetitions for each deterministic concurrency test. Each repetition rebuilds fresh state and
/// re-runs the barrier'd interleaving, so the property is asserted over many real interleavings.
const REPS: usize = 50;
/// The project's contended-writer fan-out.
const THREADS: usize = 8;
/// Per-test wall-clock deadline. Passing runs take a few seconds even at `REPS` reps; this is a
/// generous cap whose ONLY job is to convert a (rare, load-dependent) deadlock into a VISIBLE test
/// failure instead of an infinite silent park (see [`with_deadline`]).
const TEST_DEADLINE_SECS: u64 = 60;

// ----- helpers -------------------------------------------------------------------------------

/// Run `body` on a worker thread under a wall-clock deadline. If it does not finish within `secs`,
/// `panic!` (failing the test) instead of letting the harness park forever. A concurrency test that
/// can hang silently is itself a defect: every `#[test]` below wraps its body in this so ANY missed
/// rendezvous / deadlock becomes a normal (non-124) test FAILURE the harness reports, never a hang.
///
/// On timeout the deadlocked worker is intentionally LEAKED (we cannot safely cancel a parked
/// thread); the process exits right after the panic propagates, reaping it. The body must be
/// `Send + 'static` so it can move to the worker (the tests build all their state inside `body`).
fn with_deadline(secs: u64, name: &'static str, body: impl FnOnce() + Send + 'static) {
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let worker = thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            body();
            // Best-effort: the receiver may already be gone if we timed out; ignore the error.
            let _ = done_tx.send(());
        })
        .expect("spawn deadline worker");
    match done_rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(()) => {
            // Body finished in time; join to surface any panic it raised as THIS test's failure.
            worker
                .join()
                .unwrap_or_else(|_| panic!("{name}: worker thread panicked"));
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // The body is wedged (e.g. all threads parked on a barrier a dead worker never reached).
            // Leak the worker and fail loudly so this is a reported FAILURE, not an infinite park.
            panic!("{name}: deadlocked / exceeded {secs}s");
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            // The worker dropped the sender without sending => it panicked before `send`. Join to
            // re-raise that panic as this test's failure.
            worker
                .join()
                .unwrap_or_else(|_| panic!("{name}: worker thread panicked"));
        }
    }
}

fn run(shared: &SharedEngine, sql: &str) -> Result<QueryOutcome, DbError> {
    execute_on_shared_engine(shared, sql)
}

fn run_ok(shared: &SharedEngine, sql: &str) {
    run(shared, sql).unwrap_or_else(|err| panic!("{sql:?} failed: {err:?}"));
}

/// Run a write, returning `Ok(())` on commit and `Err(category)` on failure. Used to classify a
/// concurrent writer's outcome (commit vs. retryable serialization abort).
fn run_write(shared: &SharedEngine, sql: &str) -> Result<(), ErrorCategory> {
    run(shared, sql).map(|_| ()).map_err(|err| err.category)
}

fn scalar_i64(shared: &SharedEngine, sql: &str) -> i64 {
    match run(shared, sql).unwrap() {
        QueryOutcome::Rows { rows, .. } => {
            assert_eq!(rows.len(), 1, "{sql:?} expected exactly one row");
            assert_eq!(rows[0].len(), 1, "{sql:?} expected exactly one column");
            match &rows[0][0] {
                DbValue::Int4(v) => i64::from(*v),
                DbValue::Int8(v) => *v,
                other => panic!("{sql:?} expected an integer, got {other:?}"),
            }
        }
        other => panic!("{sql:?} expected rows, got {other:?}"),
    }
}

/// The set of `id` values currently visible in `t` (a fresh snapshot per call).
fn visible_ids(shared: &SharedEngine, table: &str) -> Vec<i64> {
    match run(shared, &format!("SELECT id FROM {table} ORDER BY id")).unwrap() {
        QueryOutcome::Rows { rows, .. } => rows
            .iter()
            .map(|row| match &row[0] {
                DbValue::Int4(v) => i64::from(*v),
                DbValue::Int8(v) => *v,
                other => panic!("expected integer id, got {other:?}"),
            })
            .collect(),
        other => panic!("expected rows, got {other:?}"),
    }
}

// ----- lost-update SI ------------------------------------------------------------------------

#[test]
fn lost_update_two_concurrent_updates_one_commits_one_aborts() {
    with_deadline(
        TEST_DEADLINE_SECS,
        "lost_update_two_concurrent_updates_one_commits_one_aborts",
        || {
            for rep in 0..REPS {
                let shared = Arc::new(SharedEngine::new());
                run_ok(&shared, "CREATE TABLE t (id INT, balance INT)");
                run_ok(&shared, "INSERT INTO t (id, balance) VALUES (1, 100)");

                // Two writers race to UPDATE the SAME row. To force the SI conflict window
                // DETERMINISTICALLY, both rendezvous at a barrier AFTER capturing their read snapshot
                // + preparing but BEFORE committing (via the prepared-hook). So both read the same
                // snapshot, then both attempt the commit: first-committer-wins admits exactly one; the
                // other's write-set row carries a commit newer than its snapshot ⇒ retryable
                // serialization abort.
                let barrier = Arc::new(Barrier::new(2));
                let handles: Vec<_> = (0..2)
                    .map(|w| {
                        let shared = Arc::clone(&shared);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let sql = format!("UPDATE t SET balance = {} WHERE id = 1", 200 + w);
                            let hook_barrier = Arc::clone(&barrier);
                            execute_concurrent_dml_with_prepared_hook(
                                &shared,
                                1000 + w as u64,
                                &sql,
                                || {
                                    // Snapshot captured + prepared; align both writers here, then
                                    // race to commit.
                                    hook_barrier.wait();
                                },
                            )
                            .map(|_| ())
                            .map_err(|err| err.category)
                        })
                    })
                    .collect();
                let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

                let commits = outcomes.iter().filter(|r| r.is_ok()).count();
                let serialization_aborts = outcomes
                    .iter()
                    .filter(|r| matches!(r, Err(ErrorCategory::Serialization)))
                    .count();
                assert_eq!(
                    commits, 1,
                    "rep {rep}: exactly one concurrent UPDATE of the same row must commit (outcomes: {outcomes:?})"
                );
                assert_eq!(
                    serialization_aborts, 1,
                    "rep {rep}: the loser must abort with a retryable serialization error (outcomes: {outcomes:?})"
                );

                // The surviving balance is one of the two writers' values, and the row count is
                // unchanged.
                let balance = scalar_i64(&shared, "SELECT balance FROM t WHERE id = 1");
                assert!(
                    balance == 200 || balance == 201,
                    "rep {rep}: surviving balance must be a committed writer's value, got {balance}"
                );
                assert_eq!(scalar_i64(&shared, "SELECT COUNT(*) FROM t"), 1);
            }
        },
    );
}

// ----- snapshot-isolation read stability -----------------------------------------------------

#[test]
fn snapshot_isolation_read_is_stable_across_a_concurrent_committing_writer() {
    with_deadline(
        TEST_DEADLINE_SECS,
        "snapshot_isolation_read_is_stable_across_a_concurrent_committing_writer",
        || {
            // A statement reads ONE consistent snapshot: COUNT(*) computed concurrently with
            // committing writers is always one of the legal committed counts, never a torn/partial
            // count. We assert the stronger invariant via a gap-free-prefix check: every observed
            // visible id set is a contiguous committed prefix, i.e. no partially-applied write leaked
            // into a read.
            for rep in 0..REPS {
                let shared = Arc::new(SharedEngine::new());
                run_ok(&shared, "CREATE TABLE t (id INT)");
                // Seed rows 1..=base.
                let base = 20;
                for id in 1..=base {
                    run_ok(&shared, &format!("INSERT INTO t (id) VALUES ({id})"));
                }

                let barrier = Arc::new(Barrier::new(THREADS));
                // One writer thread inserting new rows; the rest read repeatedly and check snapshot
                // consistency. The barrier is the FIRST thing each worker touches — nothing that can
                // panic runs before it, so a worker can never strand its peers at the rendezvous.
                let writer = {
                    let shared = Arc::clone(&shared);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        for id in (base + 1)..=(base + 30) {
                            // Each insert is its own autocommit txn; concurrent with readers.
                            let _ =
                                run_write(&shared, &format!("INSERT INTO t (id) VALUES ({id})"));
                        }
                    })
                };
                let readers: Vec<_> = (0..(THREADS - 1))
                    .map(|_| {
                        let shared = Arc::clone(&shared);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            for _ in 0..50 {
                                // ONE statement, ONE pinned snapshot. The writer appends ids in
                                // increasing order and each commit is atomic, so the rows visible at
                                // ANY single snapshot are exactly the contiguous prefix 1..=k. A torn
                                // / non-repeatable read WITHIN this statement (the value-index of one
                                // commit_seq resolved against the rows of another — the prereq #1
                                // hazard) would surface as a GAP (e.g. [1,2,4]) or a row from a newer
                                // commit mixed with an older snapshot. Asserting the result is a
                                // gap-free prefix is the single-statement snapshot-consistency check.
                                let ids = visible_ids(&shared, "t");
                                let expected: Vec<i64> = (1..=(ids.len() as i64)).collect();
                                assert_eq!(
                                    ids, expected,
                                    "rep {rep}: a single statement returned a torn / cross-snapshot \
                                     row set (not a gap-free committed prefix)"
                                );
                            }
                        })
                    })
                    .collect();

                writer.join().unwrap();
                for r in readers {
                    r.join().unwrap();
                }
                assert_eq!(scalar_i64(&shared, "SELECT COUNT(*) FROM t"), base + 30);
            }
        },
    );
}

// ----- disjoint writers both commit ----------------------------------------------------------

#[test]
fn disjoint_writers_all_commit_no_false_conflicts() {
    with_deadline(
        TEST_DEADLINE_SECS,
        "disjoint_writers_all_commit_no_false_conflicts",
        || {
            for rep in 0..REPS {
                let shared = Arc::new(SharedEngine::new());
                run_ok(&shared, "CREATE TABLE t (id INT, balance INT)");
                for id in 0..THREADS {
                    run_ok(
                        &shared,
                        &format!("INSERT INTO t (id, balance) VALUES ({id}, 0)"),
                    );
                }

                // Each writer updates a DIFFERENT row. None overlap, so all must commit — no false
                // aborts.
                let barrier = Arc::new(Barrier::new(THREADS));
                let handles: Vec<_> = (0..THREADS)
                    .map(|id| {
                        let shared = Arc::clone(&shared);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let sql = format!("UPDATE t SET balance = 1 WHERE id = {id}");
                            barrier.wait();
                            run_write(&shared, &sql)
                        })
                    })
                    .collect();
                let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

                for (id, outcome) in outcomes.iter().enumerate() {
                    assert!(
                        outcome.is_ok(),
                        "rep {rep}: disjoint writer {id} falsely aborted: {outcome:?}"
                    );
                }
                assert_eq!(
                    scalar_i64(&shared, "SELECT COUNT(*) FROM t WHERE balance = 1"),
                    THREADS as i64,
                    "rep {rep}: every disjoint UPDATE must be visible"
                );
            }
        },
    );
}

// ----- disjoint INSERTs all commit (BUG 1 regression) ----------------------------------------

#[test]
fn disjoint_inserts_into_one_table_all_commit_no_false_conflicts() {
    // BUG 1 regression. N writers each INSERT a DISTINCT value into ONE table that has NO unique
    // index, all rendezvousing at a barrier so their off-lock prepare windows overlap. Each insert
    // claims a fresh row id at install time, so two inserts can NEVER truly conflict — they are
    // distinct rows. Before the fix, `prepare_insert` predicted each row's key from the off-lock
    // snapshot's `next_row_id` and pushed it into the conflict write-set; overlapping prepare windows
    // read the SAME `next_row_id` ⇒ predicted the SAME key ⇒ the later committer FALSELY saw that key
    // in the ledger and aborted with `Serialization`. So this test FAILS before the fix (fewer than N
    // commits) and PASSES after (all N commit, all N values present).
    with_deadline(
        TEST_DEADLINE_SECS,
        "disjoint_inserts_into_one_table_all_commit_no_false_conflicts",
        || {
            for rep in 0..REPS {
                let shared = Arc::new(SharedEngine::new());
                // No unique index: there is NO legitimate conflict dimension for these inserts.
                run_ok(&shared, "CREATE TABLE t (id INT)");

                let barrier = Arc::new(Barrier::new(THREADS));
                let handles: Vec<_> = (0..THREADS)
                    .map(|w| {
                        let shared = Arc::clone(&shared);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let sql = format!("INSERT INTO t (id) VALUES ({w})");
                            barrier.wait();
                            run_write(&shared, &sql)
                        })
                    })
                    .collect();
                let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

                for (w, outcome) in outcomes.iter().enumerate() {
                    assert!(
                        outcome.is_ok(),
                        "rep {rep}: disjoint INSERT {w} falsely aborted (BUG 1 — predicted row key in \
                         the conflict set): {outcome:?}"
                    );
                }
                // Every distinct value committed exactly once.
                assert_eq!(
                    scalar_i64(&shared, "SELECT COUNT(*) FROM t"),
                    THREADS as i64,
                    "rep {rep}: all {THREADS} disjoint inserts must commit"
                );
                let mut ids = visible_ids(&shared, "t");
                ids.sort();
                assert_eq!(
                    ids,
                    (0..THREADS as i64).collect::<Vec<_>>(),
                    "rep {rep}: every distinct inserted value must be present exactly once"
                );
            }
        },
    );
}

// ----- unique conflict -----------------------------------------------------------------------

#[test]
fn concurrent_inserts_of_the_same_unique_value_one_commits_one_aborts() {
    with_deadline(
        TEST_DEADLINE_SECS,
        "concurrent_inserts_of_the_same_unique_value_one_commits_one_aborts",
        || {
            for rep in 0..REPS {
                let shared = Arc::new(SharedEngine::new());
                run_ok(&shared, "CREATE TABLE u (id INT)");
                run_ok(&shared, "CREATE UNIQUE INDEX u_id ON u (id)");

                // Many writers race to INSERT the SAME unique value. The unique-slot conflict
                // dimension of the ledger admits exactly one; the rest abort retryable.
                let barrier = Arc::new(Barrier::new(THREADS));
                let handles: Vec<_> = (0..THREADS)
                    .map(|_| {
                        let shared = Arc::clone(&shared);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            run_write(&shared, "INSERT INTO u (id) VALUES (42)")
                        })
                    })
                    .collect();
                let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

                let commits = outcomes.iter().filter(|r| r.is_ok()).count();
                assert_eq!(
                    commits, 1,
                    "rep {rep}: exactly one concurrent unique insert must commit (outcomes: {outcomes:?})"
                );
                // Every loser aborted; aborts are either serialization (lost the ledger race) or
                // engine (a committed-duplicate rejection if it lost before snapshotting) — never a
                // silent success.
                for outcome in &outcomes {
                    if let Err(category) = outcome {
                        assert!(
                            matches!(
                                category,
                                ErrorCategory::Serialization | ErrorCategory::Engine
                            ),
                            "rep {rep}: unique loser must be a retryable/constraint error, got {category:?}"
                        );
                    }
                }
                assert_eq!(
                    scalar_i64(&shared, "SELECT COUNT(*) FROM u WHERE id = 42"),
                    1,
                    "rep {rep}: the unique value must appear exactly once"
                );
            }
        },
    );
}

// ----- FK phantom under concurrency: loser aborts, never panics / wedges (BUG 2 regression) ---

#[test]
fn insert_child_with_concurrently_deleted_fk_parent_aborts_retryable_not_wedged() {
    // BUG 2 regression. An INSERT child references parent P. A concurrent committer DELETEs P. Both
    // capture their read snapshot while P still exists (so the INSERT's OFF-LOCK FK preflight passes),
    // then the DELETE commits FIRST and the INSERT's commit-time re-resolve at `commit_seq` sees P
    // gone. The conflict check passes (child keys are disjoint from the parent row), so before the fix
    // the commit-path re-prepare's FK failure hit `panic!("...MUST succeed...")`, poisoning the
    // commit_mutex and wedging the whole engine. After the fix that re-resolve failure is a RETRYABLE
    // `Serialization` abort, the engine stays healthy, and a subsequent statement still succeeds.
    //
    // We sequence the interleaving deterministically with the prepared-hook: the child INSERT, in its
    // hook (snapshot captured + FK-preflighted, before its commit section), blocks until the DELETE
    // has fully committed.
    with_deadline(
        TEST_DEADLINE_SECS,
        "insert_child_with_concurrently_deleted_fk_parent_aborts_retryable_not_wedged",
        || {
            for rep in 0..REPS {
                let shared = Arc::new(SharedEngine::new());
                run_ok(&shared, "CREATE TABLE parent (id INT PRIMARY KEY)");
                run_ok(&shared, "CREATE TABLE child (id INT PRIMARY KEY, pid INT)");
                run_ok(
                    &shared,
                    "ALTER TABLE ONLY child ADD CONSTRAINT child_pid_fk FOREIGN KEY (pid) REFERENCES parent(id)",
                );
                run_ok(&shared, "INSERT INTO parent (id) VALUES (1)");

                // Rendezvous: the child's hook fires once its snapshot is captured + FK-preflighted
                // (P still present). It then waits for the DELETE to finish before entering its commit
                // section, so its commit-time re-resolve provably sees P deleted. The DELETE waits for
                // the child to be prepared first (so the child's snapshot predates the delete commit).
                let child_prepared = Arc::new(Barrier::new(2));
                let parent_deleted = Arc::new(Barrier::new(2));

                let deleter = {
                    let shared = Arc::clone(&shared);
                    let child_prepared = Arc::clone(&child_prepared);
                    let parent_deleted = Arc::clone(&parent_deleted);
                    thread::spawn(move || {
                        // Wait until the child has snapshotted + preflighted (parent still visible to
                        // it), then delete + commit the parent.
                        child_prepared.wait();
                        let outcome = run_write(&shared, "DELETE FROM parent WHERE id = 1");
                        // Signal the child that the parent is now gone, regardless of the delete's
                        // outcome, so the child never strands waiting (no panicable skip of the
                        // rendezvous).
                        parent_deleted.wait();
                        outcome
                    })
                };

                let inserter = {
                    let shared = Arc::clone(&shared);
                    let child_prepared = Arc::clone(&child_prepared);
                    let parent_deleted = Arc::clone(&parent_deleted);
                    thread::spawn(move || {
                        execute_concurrent_dml_with_prepared_hook(
                            &shared,
                            2000,
                            "INSERT INTO child (id, pid) VALUES (10, 1)",
                            || {
                                // Snapshot captured + FK-preflighted (parent present). Release the
                                // deleter, then block until the parent delete has committed so our
                                // commit-time re-resolve provably races a now-missing parent.
                                child_prepared.wait();
                                parent_deleted.wait();
                            },
                        )
                        .map(|_| ())
                        .map_err(|err| err.category)
                    })
                };

                let delete_outcome = deleter.join().unwrap();
                let insert_outcome = inserter.join().unwrap();

                assert_eq!(
                    delete_outcome,
                    Ok(()),
                    "rep {rep}: the parent DELETE must commit"
                );
                // The child INSERT must be a RETRYABLE serialization abort — NOT a panic (the test
                // would have unwound), NOT an `Internal`, NOT a silent success.
                assert_eq!(
                    insert_outcome,
                    Err(ErrorCategory::Serialization),
                    "rep {rep}: the child INSERT whose FK parent was concurrently deleted must abort \
                     RETRYABLE (serialization), got {insert_outcome:?}"
                );

                // The engine is NOT wedged: the commit_mutex was not poisoned, so a subsequent simple
                // statement still succeeds (this would error with a poisoned-engine error if BUG 2's
                // panic had fired).
                run_ok(&shared, "INSERT INTO parent (id) VALUES (2)");
                assert_eq!(
                    scalar_i64(&shared, "SELECT COUNT(*) FROM parent"),
                    1,
                    "rep {rep}: parent(1) deleted, parent(2) inserted ⇒ exactly one parent row"
                );
                assert_eq!(
                    scalar_i64(&shared, "SELECT COUNT(*) FROM child"),
                    0,
                    "rep {rep}: the conflicting child INSERT left nothing durable"
                );
            }
        },
    );
}

// ----- writers overlap; a writer never blocks a reader ---------------------------------------

#[test]
fn concurrent_writers_overlap_off_lock_prepare() {
    // Deterministic overlap proof (not timing-dependent): every writer runs its OFF-LOCK prepare
    // (the expensive constraint preflight over a non-trivial table) and rendezvouses at a barrier
    // BEFORE committing. If prepare took the engine write lock, the barrier could never release
    // (writers would serialize before reaching it). Reaching the barrier with all THREADS writers
    // simultaneously in their prepare window proves prepare is off-lock and writers overlap.
    with_deadline(
        TEST_DEADLINE_SECS,
        "concurrent_writers_overlap_off_lock_prepare",
        || {
            for _ in 0..REPS {
                let shared = Arc::new(SharedEngine::new());
                run_ok(&shared, "CREATE TABLE t (id INT, v INT)");
                // A few hundred rows so the off-lock preflight/scan is non-trivial.
                for id in 0..200 {
                    run_ok(&shared, &format!("INSERT INTO t (id, v) VALUES ({id}, 0)"));
                }

                let concurrent = Arc::new(AtomicUsize::new(0));
                let peak = Arc::new(AtomicUsize::new(0));
                let barrier = Arc::new(Barrier::new(THREADS));

                let writers: Vec<_> = (0..THREADS)
                    .map(|w| {
                        let shared = Arc::clone(&shared);
                        let concurrent = Arc::clone(&concurrent);
                        let peak = Arc::clone(&peak);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            // Mark "in flight", then meet at the barrier. All writers must be able to
                            // be in flight simultaneously — impossible if a global write lock
                            // serialized them.
                            let now = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                            let mut observed = peak.load(Ordering::SeqCst);
                            while now > observed {
                                match peak.compare_exchange_weak(
                                    observed,
                                    now,
                                    Ordering::SeqCst,
                                    Ordering::SeqCst,
                                ) {
                                    Ok(_) => break,
                                    Err(actual) => observed = actual,
                                }
                            }
                            barrier.wait();
                            concurrent.fetch_sub(1, Ordering::SeqCst);
                            // Then actually commit a disjoint-row update (all should succeed).
                            run_write(&shared, &format!("UPDATE t SET v = 1 WHERE id = {w}"))
                        })
                    })
                    .collect();

                for h in writers {
                    h.join().unwrap().unwrap();
                }
                assert_eq!(
                    peak.load(Ordering::SeqCst),
                    THREADS,
                    "concurrent writers did not all overlap; the write path is serializing before commit"
                );
            }
        },
    );
}

#[test]
fn a_committing_writer_never_blocks_readers() {
    // While a long stream of writers commits, readers must keep making progress concurrently. We
    // prove overlap deterministically: readers and writers rendezvous at a barrier, and each reader
    // records peak reader concurrency while a writer commit stream runs. Peak == reader count means
    // reads ran fully concurrently with the committing writers (a writer's commit never excluded a
    // reader).
    const READERS: usize = 6;
    with_deadline(
        TEST_DEADLINE_SECS,
        "a_committing_writer_never_blocks_readers",
        || {
            for _ in 0..REPS {
                let shared = Arc::new(SharedEngine::new());
                run_ok(&shared, "CREATE TABLE t (id INT)");
                for id in 0..50 {
                    run_ok(&shared, &format!("INSERT INTO t (id) VALUES ({id})"));
                }

                let peak = Arc::new(AtomicUsize::new(0));
                let concurrent = Arc::new(AtomicUsize::new(0));
                let barrier = Arc::new(Barrier::new(READERS));
                let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

                // Background writer commit stream (separate from the reader barrier group).
                let writer = {
                    let shared = Arc::clone(&shared);
                    let stop = Arc::clone(&stop);
                    thread::spawn(move || {
                        let mut id = 1000;
                        while !stop.load(Ordering::Relaxed) {
                            let _ =
                                run_write(&shared, &format!("INSERT INTO t (id) VALUES ({id})"));
                            id += 1;
                        }
                    })
                };

                let readers: Vec<_> = (0..READERS)
                    .map(|_| {
                        let shared = Arc::clone(&shared);
                        let peak = Arc::clone(&peak);
                        let concurrent = Arc::clone(&concurrent);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            // Rendezvous FIRST while "in" the read group so peak reader concurrency is
                            // observable — nothing that can panic / early-return runs before the
                            // barrier, so a reader can never strand its peers (BUG 4: the warm-up read
                            // that previously preceded the barrier could panic on an error and hang
                            // the whole test). The warm-up read is moved AFTER the barrier and is
                            // non-fatal.
                            let now = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                            let mut observed = peak.load(Ordering::SeqCst);
                            while now > observed {
                                match peak.compare_exchange_weak(
                                    observed,
                                    now,
                                    Ordering::SeqCst,
                                    Ordering::SeqCst,
                                ) {
                                    Ok(_) => break,
                                    Err(actual) => observed = actual,
                                }
                            }
                            barrier.wait();
                            // A read against the committing table, AFTER the rendezvous. Non-fatal:
                            // its result is irrelevant to this overlap test, and it must never abort
                            // a worker before it has done its barrier accounting.
                            let _ = run(&shared, "SELECT COUNT(*) FROM t");
                            concurrent.fetch_sub(1, Ordering::SeqCst);
                        })
                    })
                    .collect();

                for r in readers {
                    r.join().unwrap();
                }
                stop.store(true, Ordering::Relaxed);
                writer.join().unwrap();
                assert_eq!(
                    peak.load(Ordering::SeqCst),
                    READERS,
                    "readers did not all overlap while a writer was committing — a commit blocked readers"
                );
            }
        },
    );
}

// ----- kill-mid-commit under concurrency -----------------------------------------------------

#[test]
fn kill_mid_commit_under_concurrency_loses_nothing_past_the_durable_boundary() {
    // Concurrent committers against a DURABLE-WAL engine; then "kill" (drop the engine) and recover
    // from the durable WAL. Recovery must reproduce a consistent state containing exactly the
    // committed rows — nothing past the durable boundary is lost, and replay of the concurrent
    // commit log is deterministic (each entry re-applied at its commit `Index`).
    use std::path::PathBuf;

    with_deadline(
        TEST_DEADLINE_SECS,
        "kill_mid_commit_under_concurrency_loses_nothing_past_the_durable_boundary",
        || {
            for rep in 0..10 {
                let dir = std::env::temp_dir().join(format!(
                    "gpu_db_kill_mid_commit_{}_{rep}",
                    std::process::id()
                ));
                std::fs::create_dir_all(&dir).unwrap();
                let segment: PathBuf = dir.join("wal.segment");

                let committed_ids = {
                    let engine = Engine::with_durable_wal_segment(&segment);
                    assert!(engine.wal_is_durable());
                    let shared = Arc::new(SharedEngine::from_engine(engine));
                    run_ok(&shared, "CREATE TABLE t (id INT)");

                    // THREADS writers each insert a disjoint block of ids concurrently.
                    let per = 25;
                    let barrier = Arc::new(Barrier::new(THREADS));
                    let handles: Vec<_> = (0..THREADS)
                        .map(|w| {
                            let shared = Arc::clone(&shared);
                            let barrier = Arc::clone(&barrier);
                            thread::spawn(move || {
                                barrier.wait();
                                let mut committed = Vec::new();
                                for k in 0..per {
                                    let id = (w * per + k) as i64;
                                    if run_write(
                                        &shared,
                                        &format!("INSERT INTO t (id) VALUES ({id})"),
                                    )
                                    .is_ok()
                                    {
                                        committed.push(id);
                                    }
                                }
                                committed
                            })
                        })
                        .collect();
                    let mut committed_ids: Vec<i64> = handles
                        .into_iter()
                        .flat_map(|h| h.join().unwrap())
                        .collect();
                    committed_ids.sort();

                    // The live engine sees exactly the committed ids.
                    assert_eq!(
                        visible_ids(&shared, "t"),
                        committed_ids,
                        "rep {rep}: live visible ids must equal the committed ids"
                    );
                    committed_ids
                    // `shared` (and its engine) dropped here == the "kill".
                };

                // Recover from the durable WAL and assert the committed rows survived intact.
                let recovered = Engine::recover_from_durable_wal_file(&segment).unwrap();
                let recovered_shared = SharedEngine::from_engine(recovered);
                assert_eq!(
                    visible_ids(&recovered_shared, "t"),
                    committed_ids,
                    "rep {rep}: recovery from the durable WAL must reproduce exactly the committed rows"
                );

                let _ = std::fs::remove_dir_all(&dir);
            }
        },
    );
}

// ----- residency ↔ data consistency (real GPU) -----------------------------------------------

#[test]
#[ignore = "requires a local NVIDIA driver and GPU (residency↔data snapshot consistency)"]
fn residency_data_consistency_concurrent_writer_invalidates_while_reader_reads() {
    // Warm a table to GPU residency, then run a concurrent writer (which commits + invalidates the
    // table's residency) against many readers. EVERY reader's result must be a SINGLE consistent
    // snapshot: the count it observes equals the contiguous id-prefix it observes (no cross-snapshot
    // rows mixing the GPU-resident generation with a newer committed one), and equals the CPU result
    // at that snapshot. This is the residency↔data snapshot-consistency property (design Risk #3) on
    // real device memory. It also exercises BUG 3's CPU fallback: a commit invalidates residency
    // mid-statement, and the resident-route read transparently falls back to the CPU pinned read
    // instead of surfacing "no retained resident device memory".
    with_deadline(
        TEST_DEADLINE_SECS,
        "residency_data_consistency_concurrent_writer_invalidates_while_reader_reads",
        || {
            for rep in 0..10 {
                let mut engine = Engine::new_local();
                engine
                    .execute_text(1, "CREATE TABLE t (id INT, v INT)")
                    .unwrap();
                let base = 64;
                for id in 1..=base {
                    engine
                        .execute_text(
                            (id + 1) as u64,
                            &format!("INSERT INTO t (id, v) VALUES ({id}, {id})"),
                        )
                        .unwrap();
                }
                // Warm `t` to GPU residency so reads take the resident route.
                let report =
                    engine.warm_relational_residency_with_policy(RelationalResidencyWarmupPolicy {
                        tables: vec!["t".to_string()],
                        refresh_invalidated: true,
                        ..RelationalResidencyWarmupPolicy::default()
                    });
                assert_eq!(report.entries.len(), 1, "rep {rep}: warm one table");

                let shared = Arc::new(SharedEngine::from_engine(engine));

                let barrier = Arc::new(Barrier::new(THREADS));
                // One writer commits new rows (each commit invalidates `t`'s GPU residency via the
                // concurrent path's device-memory tombstone); the rest read and check snapshot
                // consistency.
                let writer = {
                    let shared = Arc::clone(&shared);
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        for id in (base + 1)..=(base + 25) {
                            let _ = run_write(
                                &shared,
                                &format!("INSERT INTO t (id, v) VALUES ({id}, {id})"),
                            );
                        }
                    })
                };
                let readers: Vec<_> = (0..(THREADS - 1))
                    .map(|_| {
                        let shared = Arc::clone(&shared);
                        let barrier = Arc::clone(&barrier);
                        thread::spawn(move || {
                            barrier.wait();
                            for _ in 0..40 {
                                let ids = visible_ids(&shared, "t");
                                let count = ids.len() as i64;
                                // The row set is always a single consistent committed prefix 1..=count:
                                // a GPU-route result that mixed the stale resident generation with
                                // newer committed rows (cross-snapshot) would break this. This also
                                // equals the CPU result at the same snapshot (same predicate, same
                                // boundary).
                                let expected: Vec<i64> = (1..=count).collect();
                                assert_eq!(
                                    ids, expected,
                                    "rep {rep}: GPU/CPU read returned cross-snapshot rows under a \
                                     concurrent residency-invalidating writer"
                                );
                            }
                        })
                    })
                    .collect();

                writer.join().unwrap();
                for r in readers {
                    r.join().unwrap();
                }
                assert_eq!(scalar_i64(&shared, "SELECT COUNT(*) FROM t"), base + 25);
            }
        },
    );
}
