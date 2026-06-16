//! Engine-side point-lookup batcher (Thread-3, Stage 1).
//!
//! Amortizes the per-call host-side CUDA driver-submit floor by coalescing many
//! concurrent equality point-lookups into ONE GPU submission. It owns a single
//! coalescer OS thread (Model 1: synchronous `complete` on that thread). It batches
//! every single-predicate int4-equality projection route class — `int4_equality_projection`
//! (single column), `int4_equality_multi_column_projection` (multiple int4 columns), and
//! `int4_equality_mixed_column_projection` (int4 + text) (Stage 4 widened this from the
//! single-column class only); distinct shapes/tables form distinct `route_id` groups and so
//! distinct engine submits. The async ingress, instead of a
//! `spawn_blocking(execute_on_shared_engine)` per query, hands a batchable
//! `SELECT` to [`PointLookupBatcher::enqueue`] and `await`s the returned
//! `oneshot` while holding no semaphore permit (it is parked, not running).
//!
//! ## Correctness invariants (the audit checks all of these)
//!
//! - **One pinned generation per batch.** The coalescer runs prepare→submit→complete for the whole
//!   drained batch over the shared `&Engine` (no façade lock); the engine's `&self` job APIs pin one
//!   `committed_seq` generation across submit→complete, so the batch reads a single consistent
//!   snapshot even as concurrent writers commit (lock-free read path, write-half MVCC).
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
use gpu_db_sql::Select;
use tokio::sync::oneshot;

use crate::{map_column, map_value, DbError, ErrorCategory, QueryOutcome, SharedEngine};

/// Default flush triggers. `max_wait` is now the *ceiling* of an adaptive wait
/// (see [`AdaptiveWait`]): at connection-count 1 a lone request flushes with
/// ~zero wait (no partner to coalesce with, so paying the timer is pure cost),
/// while at high concurrency the wait opens back up to this ceiling so full
/// batches still form. The count trigger (`max_items`) is unchanged and still
/// short-circuits the wait the instant a batch fills. These are the Stage-1
/// starting point — Stage 2 sweeps them.
const DEFAULT_MAX_ITEMS: usize = 32;
const DEFAULT_MAX_WAIT: Duration = Duration::from_micros(50);

/// EWMA smoothing factor for the recent-batch-size signal that drives the
/// adaptive wait. Higher ⇒ reacts faster to a change in offered concurrency;
/// lower ⇒ steadier. 0.3 reaches ~90% of a step change in ~6 batches, fast
/// enough to open up within a burst yet damped against single-batch noise.
const ADAPTIVE_EWMA_ALPHA: f64 = 0.3;

/// The adaptive-wait controller. It turns a cheap "is coalescing actually
/// happening?" signal into the *effective* time the coalescer is willing to hold
/// a partial batch open, between 0 and the configured `max_wait` ceiling.
///
/// ## Signal
/// An EWMA of recent flushed batch sizes (`ewma_batch_size`). At rest under a
/// lone client every batch is size 1, so the EWMA sits at ~1; under real
/// concurrency batches grow and the EWMA climbs toward `max_items`.
///
/// ## Effective wait
/// `effective = max_wait * frac`, where `frac = (ewma - 1) / (max_items - 1)`
/// clamped to `[0, 1]`. So a lone client (ewma≈1 ⇒ frac≈0) pays ~0 wait, and a
/// saturated client (ewma≈max_items ⇒ frac≈1) waits the full ceiling. This is
/// the *steady-state* term; the inner loop additionally forces the full ceiling
/// the moment a partner is observed to be already queued (see `coalescer_loop`),
/// so a burst coalesces immediately without waiting for the EWMA to ramp.
///
/// ## Bound (no starvation)
/// `effective_wait` is by construction `<= max_wait` (frac is clamped to ≤ 1),
/// and the coalescer always also clamps it against the batcher's real
/// `time_until_flush_deadline`. A request can therefore NEVER be held past the
/// configured `max_wait` regardless of the adaptation — the adaptation only ever
/// shortens the wait, never lengthens it past the ceiling.
#[derive(Debug)]
struct AdaptiveWait {
    max_wait: Duration,
    max_items: usize,
    ewma_batch_size: f64,
}

impl AdaptiveWait {
    fn new(max_items: usize, max_wait: Duration) -> Self {
        Self {
            max_wait,
            max_items,
            // Seed at 1.0 (a lone request) so a cold batcher starts in the
            // ~no-wait regime and only opens up once it observes coalescing.
            ewma_batch_size: 1.0,
        }
    }

    /// The effective wait the coalescer should be willing to hold a partial batch
    /// open for *right now*, derived purely from the steady-state EWMA signal.
    /// Always in `[0, max_wait]`.
    fn effective_wait(&self) -> Duration {
        // `max_items == 1` means the count trigger fires on the first item, so a
        // partial batch is never held and the wait is irrelevant; report 0.
        if self.max_items <= 1 {
            return Duration::ZERO;
        }
        let span = (self.max_items - 1) as f64;
        // `clamp` passes NaN through and `Duration::mul_f64(NaN)` panics; the EWMA is seeded at
        // 1.0 and updated only with finite non-negative samples, so `frac` is finite today — guard
        // the wait math anyway so no future signal source can panic the coalescer thread.
        let frac = ((self.ewma_batch_size - 1.0) / span).clamp(0.0, 1.0);
        let frac = if frac.is_finite() { frac } else { 0.0 };
        self.max_wait.mul_f64(frac)
    }

    /// Fold one flushed batch's size into the EWMA. Called after every flush so
    /// the controller tracks the *actual* recent coalescing rate.
    fn record_batch(&mut self, batch_size: usize) {
        let sample = batch_size as f64;
        self.ewma_batch_size =
            ADAPTIVE_EWMA_ALPHA * sample + (1.0 - ADAPTIVE_EWMA_ALPHA) * self.ewma_batch_size;
    }
}

/// One batchable point-lookup request: a parsed `SELECT`, its int4 needle, and
/// the `oneshot` sender the coalescer answers on. `route_id` is filled in by the
/// coalescer (it needs an `engine.read()` to compute), so requests of different
/// shapes/tables can share the queue and be grouped at flush time. (`route_id` is computed on the
/// shared `&Engine`, no lock.)
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
    /// `Arc<SharedEngine>` and reaches the engine through the shared `&Engine` (no façade lock — the
    /// engine's `&self` job APIs own the "one pinned generation per batch" guarantee).
    pub fn new(engine: Arc<SharedEngine>) -> Self {
        Self::with_triggers(engine, DEFAULT_MAX_ITEMS, DEFAULT_MAX_WAIT)
    }

    /// `new` with explicit flush triggers (for tests / Stage-2 tuning).
    pub fn with_triggers(engine: Arc<SharedEngine>, max_items: usize, max_wait: Duration) -> Self {
        Self::spawn(engine, max_items, max_wait, None)
    }

    /// Shared spawn path. `batch_size_observer` is `None` in production; tests pass
    /// `Some(sender)` to observe each flushed batch's size (the only test seam for
    /// asserting that a burst actually coalesced) — it does not affect behavior.
    fn spawn(
        engine: Arc<SharedEngine>,
        max_items: usize,
        max_wait: Duration,
        batch_size_observer: Option<Sender<usize>>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<PointLookupRequest>();
        let coalescer = thread::Builder::new()
            .name("point-lookup-coalescer".to_string())
            .spawn(move || coalescer_loop(engine, rx, max_items, max_wait, batch_size_observer))
            .expect("spawn point-lookup coalescer thread");
        Self {
            tx: Some(tx),
            coalescer: Some(coalescer),
        }
    }

    /// Test-only: like [`Self::with_triggers`] but every flushed batch's size is
    /// reported on `batch_size_observer`, letting tests assert coalescing behavior
    /// (a lone request ⇒ size-1 batches; a burst ⇒ a larger batch).
    #[cfg(test)]
    fn with_triggers_observed(
        engine: Arc<SharedEngine>,
        max_items: usize,
        max_wait: Duration,
        batch_size_observer: Sender<usize>,
    ) -> Self {
        Self::spawn(engine, max_items, max_wait, Some(batch_size_observer))
    }

    /// Enqueue a batchable equality point-lookup and return the `oneshot` the
    /// caller `await`s. The caller is responsible for having classified `select`
    /// as batchable (a single-predicate int4-equality projection — single-column,
    /// multi-column, or mixed int4/text — on a resident, valid-generation table) —
    /// see [`crate::execute_on_shared_engine_batched`]. If the coalescer
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
    batch_size_observer: Option<Sender<usize>>,
) {
    let mut batcher = DualTriggerBatcher::<PointLookupRequest>::new(max_items, max_wait);
    let mut adaptive = AdaptiveWait::new(max_items, max_wait);
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
            // Count trigger fired on the first item (max_items == 1).
            run_and_record(&engine, batch, &mut adaptive, batch_size_observer.as_ref());
            continue;
        }

        // A partial batch is held. Before deciding how long to wait, drain any
        // requests ALREADY sitting in the channel without blocking: this both
        // coalesces them and is the burst detector — if a partner is already
        // present, concurrency is live *right now*, so we hold the batch open to
        // the full `max_wait` ceiling regardless of the (possibly cold) EWMA.
        let mut partner_present = false;
        let mut flushed_in_drain = false;
        loop {
            match rx.try_recv() {
                Ok(request) => {
                    partner_present = true;
                    if let Some(batch) = batcher.enqueue(request, Instant::now()) {
                        run_and_record(&engine, batch, &mut adaptive, batch_size_observer.as_ref());
                        // count trigger fired.
                        flushed_in_drain = true;
                        break;
                    }
                    // else: still partial — keep draining already-queued partners.
                }
                // Nothing more buffered (channel still open) — stop draining.
                Err(mpsc::TryRecvError::Empty) => break,
                // Owner dropped: run the held partial for real, then let the outer
                // `recv` observe the close and exit. No request left unanswered.
                Err(mpsc::TryRecvError::Disconnected) => {
                    if let Some(batch) = batcher.flush_admin() {
                        run_and_record(&engine, batch, &mut adaptive, batch_size_observer.as_ref());
                    }
                    flushed_in_drain = true;
                    break;
                }
            }
        }
        if flushed_in_drain {
            continue;
        }

        // Effective wait: full ceiling if a partner is already present (live
        // concurrency), else the steady-state EWMA-derived wait (≈0 for a lone
        // client at rest). This is BOUNDED by `max_wait` by construction.
        let effective_wait = if partner_present {
            max_wait
        } else {
            adaptive.effective_wait()
        };
        // Anchor the effective wait to the head item's enqueue time so it shares
        // the same clock as the batcher's real deadline; clamp against that real
        // deadline so a request can NEVER be held past the `max_wait` ceiling.
        let effective_deadline = batcher
            .first_enqueued_at()
            .map(|first_at| first_at + effective_wait);

        // Partial batch held: keep pulling without blocking past the (clamped)
        // effective deadline. `recv_timeout` wakes us on a new request or when the
        // wait elapses, whichever comes first.
        loop {
            let now = Instant::now();
            // The real ceiling countdown (never exceeded), and the adaptive
            // countdown clamped to it. `min` enforces the starvation bound.
            let ceiling = batcher
                .time_until_flush_deadline(now)
                .unwrap_or(Duration::ZERO);
            let adaptive_remaining = effective_deadline
                .map(|d| d.saturating_duration_since(now))
                .unwrap_or(Duration::ZERO);
            let wait = adaptive_remaining.min(ceiling);
            if wait.is_zero() {
                if let Some(batch) = batcher.flush_admin() {
                    run_and_record(&engine, batch, &mut adaptive, batch_size_observer.as_ref());
                }
                break;
            }
            match rx.recv_timeout(wait) {
                Ok(request) => {
                    if let Some(batch) = batcher.enqueue(request, Instant::now()) {
                        run_and_record(&engine, batch, &mut adaptive, batch_size_observer.as_ref());
                        // count trigger fired.
                        break;
                    }
                    // else: still partial — loop and keep waiting on the deadline.
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // The (adaptive or ceiling) deadline elapsed — flush the partial.
                    if let Some(batch) = batcher.flush_admin() {
                        run_and_record(&engine, batch, &mut adaptive, batch_size_observer.as_ref());
                    }
                    break;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Owner dropped mid-fill: run the partial batch for real (valid
                    // work that simply had not hit a trigger), then return to the
                    // outer loop whose `recv` now reports the closed channel and we
                    // exit. No request is ever left unanswered.
                    if let Some(batch) = batcher.flush_admin() {
                        run_and_record(&engine, batch, &mut adaptive, batch_size_observer.as_ref());
                    }
                    break;
                }
            }
        }
    }
}

/// Fold a flushed batch's size into the adaptive-wait signal, then run it. All
/// flush paths (count, adaptive-time, drain, shutdown) go through here so the
/// EWMA tracks the *actual* recent coalescing rate across every flush reason.
/// `observer` is `None` in production; tests use it to assert batch composition.
fn run_and_record(
    engine: &SharedEngine,
    batch: Batch<PointLookupRequest>,
    adaptive: &mut AdaptiveWait,
    observer: Option<&Sender<usize>>,
) {
    let size = batch.items.len();
    adaptive.record_batch(size);
    if let Some(tx) = observer {
        let _ = tx.send(size);
    }
    run_batch(engine, batch);
}

/// Run one drained batch under a single read lock: group by `route_id`, then
/// submit+complete each group. The lock is taken once for the whole batch.
fn run_batch(engine: &SharedEngine, batch: Batch<PointLookupRequest>) {
    let requests: Vec<PointLookupRequest> = batch.items.into_iter().map(|it| it.item).collect();
    if requests.is_empty() {
        return;
    }

    // The whole batch (submit + complete for every group) runs over ONE pinned generation on the
    // shared `&Engine` (no façade lock — the "one read-lock per batch" invariant is now "one pinned
    // generation per batch", enforced by the engine's `&self` job APIs). A committer that panicked
    // mid-commit poisons the engine's commit_mutex: fail every waiter loud rather than serve
    // possibly-torn state (mirrors `execute_on_shared_engine`).
    let engine_ref: &Engine = match engine.read_engine() {
        Ok(engine_ref) => engine_ref,
        Err(()) => {
            fail_all(requests, crate::poisoned_engine_error());
            return;
        }
    };
    if engine_ref.is_commit_path_poisoned() {
        fail_all(requests, crate::poisoned_engine_error());
        return;
    }

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
    // The batch is complete; every group ran over the shared `&Engine` (no lock to release).
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
    use gpu_db_sql::{parse_command, Command};
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

    // --- Adaptive-wait unit tests (no GPU; pure controller + timing/coalescing) ---

    #[test]
    fn adaptive_wait_is_zero_for_a_lone_client_and_full_when_saturated() {
        let max_wait = Duration::from_micros(50);
        let mut a = AdaptiveWait::new(32, max_wait);
        // Cold start is a lone client (EWMA seeded at 1) ⇒ no wait at all.
        assert_eq!(
            a.effective_wait(),
            Duration::ZERO,
            "a lone client must pay ~0 wait"
        );
        // Drive the EWMA to saturation with repeated full batches ⇒ full ceiling.
        for _ in 0..50 {
            a.record_batch(32);
        }
        assert_eq!(
            a.effective_wait(),
            max_wait,
            "a saturated client must wait the full ceiling"
        );
    }

    #[test]
    fn adaptive_wait_is_bounded_by_max_wait_for_any_signal() {
        let max_wait = Duration::from_micros(50);
        let mut a = AdaptiveWait::new(32, max_wait);
        // Even if the EWMA is pushed absurdly high (more than max_items), the
        // fraction clamps to 1.0 so the effective wait never exceeds the ceiling.
        for _ in 0..100 {
            a.record_batch(10_000);
        }
        assert!(
            a.effective_wait() <= max_wait,
            "effective wait {:?} must never exceed the ceiling {:?}",
            a.effective_wait(),
            max_wait
        );
        // And an intermediate signal lands strictly between 0 and the ceiling.
        let mut mid = AdaptiveWait::new(32, max_wait);
        for _ in 0..50 {
            mid.record_batch(16);
        }
        let w = mid.effective_wait();
        assert!(
            w > Duration::ZERO && w < max_wait,
            "a mid-concurrency signal must be between 0 and the ceiling, got {w:?}"
        );
    }

    #[test]
    fn adaptive_wait_max_items_one_never_waits() {
        // max_items == 1 means the count trigger fires on the first item, so the
        // wait is irrelevant and must report zero (no divide-by-zero either).
        let a = AdaptiveWait::new(1, Duration::from_micros(50));
        assert_eq!(a.effective_wait(), Duration::ZERO);
    }

    #[test]
    fn record_batch_moves_the_ewma_toward_the_sample() {
        let mut a = AdaptiveWait::new(32, Duration::from_micros(50));
        let before = a.ewma_batch_size;
        a.record_batch(32);
        assert!(
            a.ewma_batch_size > before,
            "a large batch must raise the EWMA"
        );
        // It is a smoothing average, not a jump to the sample.
        assert!(
            a.ewma_batch_size < 32.0,
            "one sample must not snap the EWMA to the sample value"
        );
    }

    /// A lone (c1-like) request must flush with ~no wait even when the configured
    /// `max_wait` ceiling is large: the old fixed-wait code would block the full
    /// ceiling, the adaptive code returns almost immediately. We use a 1s ceiling
    /// and require the answer well under it (generous margin for CI jitter).
    #[test]
    fn lone_request_flushes_with_near_zero_wait() {
        let engine = cpu_engine_with_table();
        let max_wait = Duration::from_secs(1);
        let batcher = PointLookupBatcher::with_triggers(engine, 32, max_wait);
        let start = Instant::now();
        let rx = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 1);
        // CPU engine ⇒ this resolves to an error, but it must resolve FAST.
        let outcome = recv_within(rx, Duration::from_secs(2)).expect("waiter must get a response");
        let elapsed = start.elapsed();
        assert!(outcome.is_err());
        assert!(
            elapsed < max_wait / 4,
            "a lone request waited {elapsed:?}, near the {max_wait:?} ceiling — adaptive \
             shortening did not kick in"
        );
    }

    /// A concurrent burst must still coalesce: with a count trigger above the burst
    /// size and a non-trivial ceiling, the requests are served in FEWER batches
    /// than there are requests (i.e. at least one batch coalesced > 1 item). The
    /// observer reports each flushed batch's size.
    #[test]
    fn concurrent_burst_still_coalesces() {
        let engine = cpu_engine_with_table();
        let (obs_tx, obs_rx) = mpsc::channel::<usize>();
        // Count trigger (64) above the burst (16) so coalescing is via the wait,
        // not the count trigger; a 200ms ceiling gives the burst time to gather.
        let batcher = PointLookupBatcher::with_triggers_observed(
            engine,
            64,
            Duration::from_millis(200),
            obs_tx,
        );
        let n = 16usize;
        let receivers: Vec<_> = (0..n)
            .map(|needle| batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), needle as i32))
            .collect();
        // Every request is still answered (completeness preserved).
        for rx in receivers {
            let outcome = recv_within(rx, Duration::from_secs(3)).expect("every waiter answered");
            assert!(outcome.is_err());
        }
        // Drop the batcher so the observer channel closes once the coalescer exits,
        // then collect every reported batch size.
        drop(batcher);
        let mut sizes = Vec::new();
        let mut total: usize = 0;
        while let Ok(sz) = obs_rx.recv_timeout(Duration::from_secs(2)) {
            total += sz;
            sizes.push(sz);
        }
        assert_eq!(total, n, "every request must appear in exactly one batch");
        assert!(
            sizes.len() < n,
            "a concurrent burst must coalesce into fewer than {n} batches, got sizes {sizes:?}"
        );
        let max_batch = sizes.iter().copied().max().unwrap_or(0);
        assert!(
            max_batch > 1,
            "at least one batch must have coalesced >1 request, got sizes {sizes:?}"
        );
    }

    /// Even when the adaptive wait is "open" (a partner was present so the coalescer
    /// holds to the full ceiling), a held partial batch must STILL flush within the
    /// configured ceiling — no starvation. We send two requests (so a partner is
    /// present ⇒ full-ceiling hold) with a count trigger above 2, and require both
    /// answered comfortably within a small multiple of the ceiling.
    #[test]
    fn held_partial_batch_never_starves_past_the_ceiling() {
        let engine = cpu_engine_with_table();
        let max_wait = Duration::from_millis(20);
        // max_items=8 so two requests never hit the count trigger and the time
        // bound is the only thing that flushes them.
        let batcher = PointLookupBatcher::with_triggers(engine, 8, max_wait);
        let start = Instant::now();
        let r1 = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 1);
        let r2 = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 2);
        let o1 = recv_within(r1, Duration::from_secs(2)).expect("first waiter answered");
        let o2 = recv_within(r2, Duration::from_secs(2)).expect("second waiter answered");
        let elapsed = start.elapsed();
        assert!(o1.is_err() && o2.is_err());
        // Bound: a held partial flushes by the ceiling. Allow generous slack for
        // scheduling jitter, but it must be a small multiple of max_wait, proving
        // the wait is bounded and not unbounded/forever.
        assert!(
            elapsed < max_wait * 20,
            "a held partial waited {elapsed:?}, far past the {max_wait:?} ceiling — \
             the starvation bound is broken"
        );
    }
}
