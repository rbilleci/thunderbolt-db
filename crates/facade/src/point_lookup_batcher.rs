//! Engine-side point-lookup batcher (Thread-3, Stage 1).
//!
//! Amortizes the per-call host-side CUDA driver-submit floor by coalescing many
//! concurrent equality point-lookups into ONE GPU submission. It owns a single
//! coalescer OS thread (Model 1: synchronous `complete` on that thread) and one
//! route class (`int4_equality_projection`). The async ingress, instead of a
//! `spawn_blocking(execute_on_shared_engine)` per query, hands a batchable
//! `SELECT` to [`PointLookupBatcher::enqueue`] and `await`s the returned
//! `oneshot` while holding no semaphore permit (it is parked, not running).
//!
//! ## Correctness invariants (the audit checks all of these)
//!
//! - **One read lock per batch.** The coalescer takes `engine.read()` ONCE and
//!   runs prepare→submit→complete for the whole drained batch under that single
//!   acquisition, so no writer interleaves a batch (writes take the write lock,
//!   which is mutually exclusive — plan §7).
//! - **Error fanout is total.** Any error from prepare/submit/complete is sent to
//!   *every* waiter whose request was in that submit group — never a hung
//!   connection. A drained batch is split into per-`route_id` groups; a failing
//!   group fails only its own waiters, the other groups still complete.
//! - **Stale generation is a clean per-waiter error.** The engine's per-job
//!   generation guard rejects a stale item with an `Err`; that `Err` is fanned
//!   out to the group's waiters (it does not wedge the coalescer).
//! - **Dropped receiver is harmless.** If a client disconnected while parked, its
//!   `oneshot` receiver is gone and the coalescer's `send` returns `Err(_)` which
//!   is ignored; the batch still completes for the other waiters.
//! - **Shutdown drains the queue.** When the [`PointLookupBatcher`] is dropped the
//!   request channel closes. `mpsc` delivers every already-buffered request before
//!   `recv` reports the close, so the coalescer runs each still-queued request for
//!   real (answering its `oneshot`) and only then exits — no waiter is left parked.
//! - **Request↔job order + needle dedup.** Within a `route_id` group, identical
//!   needles are submitted to the kernel once (the `equal_any` kernel scans every
//!   row against all N needles, so duplicates would be wasted work); each request
//!   is mapped back to its needle's sliced result, preserving per-request order.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use gpu_db_batching::{Batch, DualTriggerBatcher};
use gpu_db_engine::{Engine, ExecuteError, RelationalSelectResult};
use gpu_db_protocol::Select;
use tokio::sync::oneshot;

use crate::{map_column, map_value, DbError, ErrorCategory, QueryOutcome, SharedEngine};

/// Default flush triggers. At connection-count 1 a batch is size-1 (the time
/// trigger fires almost immediately), so there is no single-client regression;
/// at high concurrency the count trigger fires first and never waits for the
/// timer. These are the Stage-1 starting point — Stage 2 sweeps them.
const DEFAULT_MAX_ITEMS: usize = 32;
const DEFAULT_MAX_WAIT: Duration = Duration::from_micros(50);

/// One batchable point-lookup request: a parsed `SELECT`, its int4 needle, and
/// the `oneshot` sender the coalescer answers on. `route_id` is filled in by the
/// coalescer (it needs an `engine.read()` to compute), so requests of different
/// shapes/tables can share the queue and be grouped at flush time.
struct PointLookupRequest {
    select: Select,
    needle: i32,
    respond: oneshot::Sender<Result<QueryOutcome, DbError>>,
}

/// Handle to the running batcher. Dropping it closes the request channel, which
/// makes the coalescer drain every still-queued request (each is answered on its
/// `oneshot`) and then exit; the `Drop` impl joins the coalescer so all responses
/// are delivered before teardown returns. Share behind an `Arc` across connections.
pub struct PointLookupBatcher {
    tx: Option<Sender<PointLookupRequest>>,
    coalescer: Option<JoinHandle<()>>,
}

impl PointLookupBatcher {
    /// Spawn the single coalescer thread bound to `engine`. The coalescer holds an
    /// `Arc<SharedEngine>` and reaches the engine only through its `RwLock` — the
    /// façade owns the "run the batch under one read lock" guarantee (plan §1).
    pub fn new(engine: Arc<SharedEngine>) -> Self {
        Self::with_triggers(engine, DEFAULT_MAX_ITEMS, DEFAULT_MAX_WAIT)
    }

    /// `new` with explicit flush triggers (for tests / Stage-2 tuning).
    pub fn with_triggers(engine: Arc<SharedEngine>, max_items: usize, max_wait: Duration) -> Self {
        let (tx, rx) = mpsc::channel::<PointLookupRequest>();
        let coalescer = thread::Builder::new()
            .name("point-lookup-coalescer".to_string())
            .spawn(move || coalescer_loop(engine, rx, max_items, max_wait))
            .expect("spawn point-lookup coalescer thread");
        Self {
            tx: Some(tx),
            coalescer: Some(coalescer),
        }
    }

    /// Enqueue a batchable equality point-lookup and return the `oneshot` the
    /// caller `await`s. The caller is responsible for having classified `select`
    /// as batchable (an `int4_equality_projection` on a resident, valid-generation
    /// table) — see [`crate::execute_on_shared_engine_batched`]. If the coalescer
    /// has already shut down, the returned receiver resolves immediately to an
    /// error (the sender drops), so callers never hang.
    pub fn enqueue(
        &self,
        select: Select,
        needle: i32,
    ) -> oneshot::Receiver<Result<QueryOutcome, DbError>> {
        let (respond, receiver) = oneshot::channel();
        let request = PointLookupRequest {
            select,
            needle,
            respond,
        };
        match &self.tx {
            // If the send fails the coalescer is gone; dropping `request` drops its
            // `respond`, so `receiver` resolves to a `RecvError` rather than hang.
            Some(tx) => {
                let _ = tx.send(request);
            }
            None => drop(request),
        }
        receiver
    }
}

impl Drop for PointLookupBatcher {
    fn drop(&mut self) {
        // Close the channel so the coalescer's `recv` returns `Err` and it drains
        // the queue + exits; then join so in-flight responses are delivered before
        // the batcher's owner tears down.
        self.tx.take();
        if let Some(handle) = self.coalescer.take() {
            let _ = handle.join();
        }
    }
}

/// The single coalescer thread. Blocks for the first request, then drains a batch
/// (count- or time-triggered via [`DualTriggerBatcher`]), runs the batch under one
/// read lock, and scatters results/errors back to waiters. Exits when the request
/// channel closes (the [`PointLookupBatcher`] was dropped), draining the queue.
fn coalescer_loop(
    engine: Arc<SharedEngine>,
    rx: Receiver<PointLookupRequest>,
    max_items: usize,
    max_wait: Duration,
) {
    let mut batcher = DualTriggerBatcher::<PointLookupRequest>::new(max_items, max_wait);
    loop {
        // Block until there is at least one request (or the channel closed). This
        // is the low-rate park point: an empty batcher never busy-waits.
        let first = match rx.recv() {
            Ok(request) => request,
            // Channel closed and the batcher is empty here (we flush every batch
            // before looping back to this `recv`): clean shutdown, nothing to drain.
            Err(_) => break,
        };
        if let Some(batch) = batcher.enqueue(first, Instant::now()) {
            run_batch(&engine, batch); // count trigger fired on the first item (max_items == 1).
            continue;
        }
        // Partial batch held: keep pulling without blocking past the flush deadline.
        // `recv_timeout` wakes us either on a new request or when the oldest item's
        // `max_wait` elapses, whichever comes first.
        loop {
            let now = Instant::now();
            let wait = batcher
                .time_until_flush_deadline(now)
                .unwrap_or(Duration::ZERO);
            if wait.is_zero() {
                if let Some(batch) = batcher.flush_admin() {
                    run_batch(&engine, batch);
                }
                break;
            }
            match rx.recv_timeout(wait) {
                Ok(request) => {
                    if let Some(batch) = batcher.enqueue(request, Instant::now()) {
                        run_batch(&engine, batch); // count trigger fired.
                        break;
                    }
                    // else: still partial — loop and keep waiting on the deadline.
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if let Some(batch) = batcher.maybe_flush_due_to_time(Instant::now()) {
                        run_batch(&engine, batch);
                    }
                    break;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Owner dropped mid-fill: run the partial batch for real (valid
                    // work that simply had not hit a trigger), then return to the
                    // outer loop whose `recv` now reports the closed channel and we
                    // exit. No request is ever left unanswered.
                    if let Some(batch) = batcher.flush_admin() {
                        run_batch(&engine, batch);
                    }
                    break;
                }
            }
        }
    }
}

/// Run one drained batch under a single read lock: group by `route_id`, then
/// submit+complete each group. The lock is taken once for the whole batch.
fn run_batch(engine: &SharedEngine, batch: Batch<PointLookupRequest>) {
    let requests: Vec<PointLookupRequest> = batch.items.into_iter().map(|it| it.item).collect();
    if requests.is_empty() {
        return;
    }

    // ONE read lock for the entire batch (submit + complete for every group). A
    // poisoned lock means a writer panicked mid-statement: fail every waiter loud
    // rather than serve possibly-torn state (mirrors `execute_on_shared_engine`).
    let guard = match engine.read_engine() {
        Ok(guard) => guard,
        Err(()) => {
            fail_all(requests, crate::poisoned_engine_error());
            return;
        }
    };
    let engine_ref: &Engine = &guard;

    // Prepare a job per request. `prepare_relational_retained_read_job` is `&self`
    // (Stage 0) and computes `route_id`, which encodes shape:schema:table:proj:filter
    // — so equal route_ids share table/columns/filter and differ only by needle.
    // A request whose job fails to prepare (e.g. the snapshot went non-resident
    // between classify and now) is answered immediately with that error and dropped
    // from the batch — it never poisons the rest.
    let mut prepared: Vec<PreparedRequest> = Vec::with_capacity(requests.len());
    for request in requests {
        match engine_ref.prepare_relational_retained_read_job(&request.select) {
            Ok(job) => prepared.push(PreparedRequest {
                respond: request.respond,
                route_id: job.route_id.clone(),
                needle: request.needle,
                job,
            }),
            Err(err) => {
                let _ = request.respond.send(Err(map_execute_error_local(err)));
            }
        }
    }

    // Group prepared requests by route_id, preserving first-seen group order and
    // per-request order within a group. Each group is one engine submit.
    let mut group_order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<PreparedRequest>> = HashMap::new();
    for pr in prepared {
        let route_id = pr.route_id.clone();
        if !groups.contains_key(&route_id) {
            group_order.push(route_id.clone());
        }
        groups.entry(route_id).or_default().push(pr);
    }

    for route_id in group_order {
        let group = groups.remove(&route_id).expect("group present");
        run_group(engine_ref, group);
    }
    // `guard` (the read lock) drops here, after every group completed.
}

/// A request whose retained-read job has been prepared under the read lock.
struct PreparedRequest {
    respond: oneshot::Sender<Result<QueryOutcome, DbError>>,
    route_id: String,
    needle: i32,
    job: gpu_db_engine::RelationalRetainedReadJob,
}

/// Submit+complete one same-`route_id` group (identical table/columns/filter; only
/// needles vary). Identical needles are deduplicated into a single job so the
/// `equal_any` kernel does not scan the same needle twice; each request is then
/// answered from its needle's sliced result. Any submit/complete error is fanned
/// out to every waiter in the group.
fn run_group(engine: &Engine, group: Vec<PreparedRequest>) {
    // Deduplicate needles: one job per distinct needle, remembering which job index
    // each request maps to. Job order == first-seen needle order (stable).
    let mut jobs: Vec<gpu_db_engine::RelationalRetainedReadJob> = Vec::new();
    let mut needle_to_job: HashMap<i32, usize> = HashMap::new();
    let mut request_job_index: Vec<usize> = Vec::with_capacity(group.len());
    for pr in &group {
        let idx = match needle_to_job.get(&pr.needle) {
            Some(&idx) => idx,
            None => {
                let idx = jobs.len();
                needle_to_job.insert(pr.needle, idx);
                jobs.push(pr.job.clone());
                idx
            }
        };
        request_job_index.push(idx);
    }

    // Submit + complete under the caller's already-held read lock. On ANY error,
    // fan it out to every waiter in the group (no hung connection).
    let results = match submit_and_complete(engine, &jobs) {
        Ok(results) => results,
        Err(err) => {
            let mapped = map_execute_error_local(err);
            for pr in group {
                let _ = pr.respond.send(Err(mapped.clone()));
            }
            return;
        }
    };

    // The engine returns one `RelationalSelectResult` per job, in job order. Map
    // each request to its job's result and answer its oneshot. A dropped receiver
    // (client disconnected while parked) makes `send` fail harmlessly.
    debug_assert_eq!(results.len(), jobs.len());
    for (pr, job_idx) in group.into_iter().zip(request_job_index) {
        let outcome = results
            .get(job_idx)
            .map(map_select_result_to_outcome)
            .unwrap_or_else(|| {
                Err(DbError {
                    category: ErrorCategory::Internal,
                    message: "batched point-lookup result slice missing for request".to_string(),
                })
            });
        let _ = pr.respond.send(outcome);
    }
}

/// Submit then complete a job group. Kept as one call so the read lock spans the
/// whole submit→complete window (the engine API itself takes `&self`).
fn submit_and_complete(
    engine: &Engine,
    jobs: &[gpu_db_engine::RelationalRetainedReadJob],
) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
    let submission =
        engine.submit_relational_retained_read_jobs_with_resident_device_memory_probe(jobs)?;
    engine.complete_relational_retained_read_submission(submission)
}

/// Map one engine relational select result into a neutral [`QueryOutcome::Rows`],
/// reusing the façade's `map_column`/`map_value` so the batched path produces
/// byte-identical neutral output to the per-query path.
fn map_select_result_to_outcome(result: &RelationalSelectResult) -> Result<QueryOutcome, DbError> {
    let columns = result.columns.iter().map(map_column).collect();
    let rows = result
        .rows
        .iter()
        .map(|row| row.iter().cloned().map(map_value).collect())
        .collect();
    Ok(QueryOutcome::Rows { columns, rows })
}

fn fail_all(requests: Vec<PointLookupRequest>, err: DbError) {
    for request in requests {
        let _ = request.respond.send(Err(err.clone()));
    }
}

// `map_execute_error` in lib.rs is private to the crate; reuse it.
fn map_execute_error_local(err: ExecuteError) -> DbError {
    crate::map_execute_error(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execute_on_shared_engine;
    use gpu_db_protocol::{parse_command, Command};
    use std::sync::mpsc::TryRecvError;

    fn select(sql: &str) -> Select {
        match parse_command(sql).unwrap() {
            Command::Select(select) => select,
            other => panic!("expected SELECT, got {other:?}"),
        }
    }

    /// Block on a oneshot with a timeout so a buggy coalescer fails the test loudly
    /// (a hung connection) instead of hanging the whole suite.
    fn recv_within(
        mut rx: oneshot::Receiver<Result<QueryOutcome, DbError>>,
        timeout: Duration,
    ) -> Option<Result<QueryOutcome, DbError>> {
        let deadline = Instant::now() + timeout;
        loop {
            match rx.try_recv() {
                Ok(value) => return Some(value),
                Err(oneshot::error::TryRecvError::Empty) => {
                    if Instant::now() >= deadline {
                        return None;
                    }
                    thread::sleep(Duration::from_millis(1));
                }
                Err(oneshot::error::TryRecvError::Closed) => return None,
            }
        }
    }

    /// A CPU-only `SharedEngine` with one table seeded but NOT warmed to GPU
    /// residency: every retained-read job preparation fails (no resident snapshot),
    /// which exercises the error-fanout / drain / dropped-receiver logic without a
    /// GPU. The success-slicing path is GPU-gated (`#[ignore]`) below.
    fn cpu_engine_with_table() -> Arc<SharedEngine> {
        let shared = Arc::new(SharedEngine::new());
        execute_on_shared_engine(&shared, "CREATE TABLE t (id INT)").unwrap();
        execute_on_shared_engine(&shared, "INSERT INTO t (id) VALUES (1)").unwrap();
        shared
    }

    #[test]
    fn prepare_error_is_fanned_out_to_the_waiter_not_hung() {
        // No GPU residency ⇒ job prepare fails; the waiter must get that Err, never hang.
        let engine = cpu_engine_with_table();
        let batcher = PointLookupBatcher::with_triggers(engine, 4, Duration::from_millis(5));
        let rx = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 1);
        let outcome = recv_within(rx, Duration::from_secs(2)).expect("waiter must get a response");
        assert!(
            outcome.is_err(),
            "expected an engine error, got {outcome:?}"
        );
    }

    #[test]
    fn every_request_in_a_batch_gets_a_response() {
        // Several lookups coalesce into one batch; each must be answered (here all are
        // prepare-errors, but completeness is the point — no connection left hung).
        let engine = cpu_engine_with_table();
        let batcher = PointLookupBatcher::with_triggers(engine, 8, Duration::from_millis(5));
        let receivers: Vec<_> = (0..8)
            .map(|needle| batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), needle))
            .collect();
        for rx in receivers {
            let outcome = recv_within(rx, Duration::from_secs(2)).expect("every waiter answered");
            assert!(outcome.is_err());
        }
    }

    #[test]
    fn dropped_receiver_does_not_wedge_the_coalescer() {
        // A client disconnects while parked: drop its receiver. The coalescer's send
        // fails harmlessly and it must keep serving the next request.
        let engine = cpu_engine_with_table();
        let batcher = PointLookupBatcher::with_triggers(engine, 1, Duration::from_millis(5));
        let abandoned = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 7);
        drop(abandoned); // client gone before the coalescer answers.
                         // The next request must still be answered.
        let rx = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 8);
        let outcome =
            recv_within(rx, Duration::from_secs(2)).expect("coalescer still serves after a drop");
        assert!(outcome.is_err());
    }

    #[test]
    fn shutdown_drains_every_queued_request() {
        // Enqueue a burst with a long max_wait so they sit in the queue, then drop the
        // batcher. Every receiver must resolve (a real answer or channel-closed), never
        // hang — the shutdown drain guarantee.
        let engine = cpu_engine_with_table();
        let batcher = PointLookupBatcher::with_triggers(engine, 1024, Duration::from_secs(3600));
        let receivers: Vec<_> = (0..16)
            .map(|needle| batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), needle))
            .collect();
        drop(batcher); // closes the channel; Drop joins the coalescer after it drains.
        for rx in receivers {
            // After join, each receiver is either answered or closed — both are "resolved".
            match recv_within(rx, Duration::from_secs(2)) {
                Some(_resolved) => {}
                None => panic!("a queued request was left unanswered at shutdown"),
            }
        }
    }

    #[test]
    fn select_int4_equality_needle_matches_engine_filter_precedence() {
        // The classifier's needle must equal the bound equality value (dedup key parity).
        let s = select("SELECT id FROM t WHERE id = 42");
        assert_eq!(crate::select_int4_equality_needle(&s), Some(42));
        let non_eq = select("SELECT id FROM t WHERE id > 42");
        assert_eq!(crate::select_int4_equality_needle(&non_eq), None);
    }

    // Sanity that try_recv's disconnected variant is what we rely on for non-hang.
    #[test]
    fn mpsc_try_recv_reports_disconnected_after_sender_drop() {
        let (tx, rx) = mpsc::channel::<u8>();
        drop(tx);
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Disconnected)));
    }

    // --- GPU-gated success/parity tests (require a local NVIDIA driver + GPU) ---

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU (resident route)"]
    fn batched_result_matches_per_query_and_dedups_needles() {
        use gpu_db_engine::Engine;

        // Build + warm a table to GPU residency, then compare the batched path to the
        // per-query path and confirm duplicate needles are deduplicated correctly.
        let mut engine = Engine::new_local();
        engine.execute_text(1, "CREATE TABLE t (id INT)").unwrap();
        engine
            .execute_text(2, "INSERT INTO t (id) VALUES (1), (2), (2), (3)")
            .unwrap();
        engine.populate_relational_residency_snapshot("t").unwrap();
        // If the box has no GPU device memory the route is rejected; skip in that case.
        let probe = parse_command("SELECT id FROM t WHERE id = 2").unwrap();
        let Command::Select(probe_select) = probe else {
            unreachable!()
        };
        if !engine
            .plan_relational_resident_route(&probe_select)
            .accepted
        {
            return;
        }

        let shared = Arc::new(SharedEngine::from_engine(engine));
        let batcher =
            PointLookupBatcher::with_triggers(Arc::clone(&shared), 8, Duration::from_millis(5));

        // Two requests for needle=2 (duplicate) + one for needle=1.
        let r2a = batcher.enqueue(select("SELECT id FROM t WHERE id = 2"), 2);
        let r2b = batcher.enqueue(select("SELECT id FROM t WHERE id = 2"), 2);
        let r1 = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 1);

        let got_2a = recv_within(r2a, Duration::from_secs(5)).unwrap().unwrap();
        let got_2b = recv_within(r2b, Duration::from_secs(5)).unwrap().unwrap();
        let got_1 = recv_within(r1, Duration::from_secs(5)).unwrap().unwrap();

        // Per-query reference via the unchanged path.
        let ref_2 = execute_on_shared_engine(&shared, "SELECT id FROM t WHERE id = 2").unwrap();
        let ref_1 = execute_on_shared_engine(&shared, "SELECT id FROM t WHERE id = 1").unwrap();

        assert_eq!(got_2a, ref_2, "batched needle=2 must match per-query");
        assert_eq!(got_2b, ref_2, "duplicate needle=2 must get the same result");
        assert_eq!(got_1, ref_1, "batched needle=1 must match per-query");
    }
}
