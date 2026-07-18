//! Engine-side point-lookup batcher (Thread-3, Stage 1).
//!
//! Amortizes the per-call host-side CUDA driver-submit floor by coalescing many
//! concurrent equality point-lookups into ONE GPU submission. It owns a single
//! coalescer OS thread (Model 1: synchronous `complete` on that thread). It batches the **all-int4**
//! single-predicate int4-equality projection route classes — `int4_equality_projection` (single
//! column) and `int4_equality_multi_column_projection` (multiple int4 columns). The mixed int4+text
//! shape (`int4_equality_mixed_column_projection`) is NOT batched: the resident `equal_any` kernel
//! materializes int4 only, and the text-capable general executor is not CUDA-context-safe on the
//! coalescer thread, so `classify_batchable_point_lookup` routes mixed point lookups to the unchanged
//! per-query path. Distinct shapes/tables form distinct shape-key groups (one prepared resident-read
//! TEMPLATE each, reused across all needles), and so distinct engine submits. The template is the
//! needle-invariant plan, so the expensive plan/bind runs once per shape, not per request (Tier 1,
//! DECISIONS ADR-008). The async ingress, instead of a
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
//!   connection. A drained batch is split into per-shape-key groups; a failing
//!   group (including a template-prepare failure) fails only its own waiters, the
//!   other groups still complete.
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
//! - **Request↔result order + needle dedup.** Within a shape-key group, identical
//!   needles are submitted to the kernel once (the `equal_any` kernel scans every
//!   row against all N needles, so duplicates would be wasted work); each request
//!   is mapped back to its needle's sliced result, preserving per-request order.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use gpu_db_batching::{Batch, DualTriggerBatcher};
use gpu_db_engine::{
    Engine, ExecuteError, RelationalColumn, RelationalPointBatchResult,
    RelationalRetainedBatchResult,
};
use gpu_db_sql::Select;
use tokio::sync::oneshot;

use crate::{map_column, map_value, DbError, DbValue, ErrorCategory, QueryOutcome, SharedEngine};

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

/// Monotonic production-path observations from one point-lookup batcher. A sharded group is counted
/// exactly once: either the engine served it through its batched route, or the batcher had to run its
/// requests individually after that route declined. Combined with the engine's GPU-probe counter,
/// this lets a benchmark distinguish "all sharded groups were fully GPU" from a correct-but-slower
/// per-query fallback without exposing engine internals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointLookupBatcherActivitySnapshot {
    pub sharded_batched_groups: u64,
    pub sharded_per_query_fallback_groups: u64,
}

#[derive(Default)]
struct PointLookupBatcherActivity {
    sharded_batched_groups: AtomicU64,
    sharded_per_query_fallback_groups: AtomicU64,
}

/// Handle to the running batcher. Dropping it closes the request channel, which
/// makes the coalescer drain every still-queued request (each is answered on its
/// `oneshot`) and then exit; the `Drop` impl joins the coalescer so all responses
/// are delivered before teardown returns. Share behind an `Arc` across connections.
pub struct PointLookupBatcher {
    tx: Option<Sender<PointLookupRequest>>,
    coalescer: Option<JoinHandle<()>>,
    activity: Arc<PointLookupBatcherActivity>,
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
        let activity = Arc::new(PointLookupBatcherActivity::default());
        let coalescer_activity = Arc::clone(&activity);
        let coalescer = thread::Builder::new()
            .name("point-lookup-coalescer".to_string())
            .spawn(move || {
                coalescer_loop(
                    engine,
                    rx,
                    max_items,
                    max_wait,
                    batch_size_observer,
                    coalescer_activity,
                )
            })
            .expect("spawn point-lookup coalescer thread");
        Self {
            tx: Some(tx),
            coalescer: Some(coalescer),
            activity,
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

    /// Snapshot this batcher's sharded-route activity. Counters are batcher-local and monotonic.
    pub fn activity_snapshot(&self) -> PointLookupBatcherActivitySnapshot {
        PointLookupBatcherActivitySnapshot {
            sharded_batched_groups: self
                .activity
                .sharded_batched_groups
                .load(AtomicOrdering::Relaxed),
            sharded_per_query_fallback_groups: self
                .activity
                .sharded_per_query_fallback_groups
                .load(AtomicOrdering::Relaxed),
        }
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
    activity: Arc<PointLookupBatcherActivity>,
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
            run_and_record(
                &engine,
                batch,
                &mut adaptive,
                batch_size_observer.as_ref(),
                &activity,
            );
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
                        run_and_record(
                            &engine,
                            batch,
                            &mut adaptive,
                            batch_size_observer.as_ref(),
                            &activity,
                        );
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
                        run_and_record(
                            &engine,
                            batch,
                            &mut adaptive,
                            batch_size_observer.as_ref(),
                            &activity,
                        );
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
                    run_and_record(
                        &engine,
                        batch,
                        &mut adaptive,
                        batch_size_observer.as_ref(),
                        &activity,
                    );
                }
                break;
            }
            match rx.recv_timeout(wait) {
                Ok(request) => {
                    if let Some(batch) = batcher.enqueue(request, Instant::now()) {
                        run_and_record(
                            &engine,
                            batch,
                            &mut adaptive,
                            batch_size_observer.as_ref(),
                            &activity,
                        );
                        // count trigger fired.
                        break;
                    }
                    // else: still partial — loop and keep waiting on the deadline.
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // The (adaptive or ceiling) deadline elapsed — flush the partial.
                    if let Some(batch) = batcher.flush_admin() {
                        run_and_record(
                            &engine,
                            batch,
                            &mut adaptive,
                            batch_size_observer.as_ref(),
                            &activity,
                        );
                    }
                    break;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Owner dropped mid-fill: run the partial batch for real (valid
                    // work that simply had not hit a trigger), then return to the
                    // outer loop whose `recv` now reports the closed channel and we
                    // exit. No request is ever left unanswered.
                    if let Some(batch) = batcher.flush_admin() {
                        run_and_record(
                            &engine,
                            batch,
                            &mut adaptive,
                            batch_size_observer.as_ref(),
                            &activity,
                        );
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
    activity: &PointLookupBatcherActivity,
) {
    let size = batch.items.len();
    adaptive.record_batch(size);
    if let Some(tx) = observer {
        let _ = tx.send(size);
    }
    run_batch(engine, batch, activity);
}

/// Run one drained batch under a single read lock: group by `route_id`, then
/// submit+complete each group. The lock is taken once for the whole batch.
fn run_batch(
    engine: &SharedEngine,
    batch: Batch<PointLookupRequest>,
    activity: &PointLookupBatcherActivity,
) {
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

    // Group requests by a CHEAP shape key (the parsed `Select` with the needle VALUE normalized out —
    // see `point_lookup_shape_key`), preserving first-seen group order and per-request order within a
    // group. The expensive plan/bind (`prepare_relational_retained_read_template`) then runs ONCE per
    // shape in `run_group`, not once per request — removing the ~15µs/item serial host cost that capped
    // the single coalescer (DECISIONS ADR-008 "First measurement"). Each group is one engine submit.
    let mut group_order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<PointLookupRequest>> = HashMap::new();
    for request in requests {
        let key = point_lookup_shape_key(&request.select);
        if !groups.contains_key(&key) {
            group_order.push(key.clone());
        }
        groups.entry(key).or_default().push(request);
    }

    for key in group_order {
        let group = groups.remove(&key).expect("group present");
        run_group(engine_ref, group, activity);
    }
    // The batch is complete; every group ran over the shared `&Engine` (no lock to release).
}

/// A CHEAP shape key for a batchable int4 point lookup. Requests sharing a key share one resident-read
/// template, so the expensive plan/bind runs once per shape. A classify-accepted batchable point lookup
/// (route shapes `int4_equality_projection` / `_multi_column_` / `_mixed_column_`) provably has NO
/// `order_by` / `group_by` / `having` / `distinct` / `limit` / `offset` (those route to other shapes —
/// `resident_route.rs`) and exactly ONE equality predicate, so the template is fully determined by
/// `(table, projection, filter column)` with the needle VALUE excluded. We also fold in the remaining
/// scalar shape flags (cheap, all default here) as belt-and-suspenders against any future shape
/// widening. This is sub-µs — no full-`Select` clone or Debug (which dominated the coalescer when first
/// tried). Mirrors the engine's `filter_groups → filters → filter` predicate precedence.
fn point_lookup_shape_key(select: &Select) -> String {
    let filter_column =
        if let Some(filter) = select.filter_groups.first().and_then(|group| group.first()) {
            filter.column.as_str()
        } else if let Some(filter) = select.filters.first() {
            filter.column.as_str()
        } else if let Some(filter) = select.filter.as_ref() {
            filter.column.as_str()
        } else {
            ""
        };
    format!(
        "{}\u{1}{:?}\u{1}{}\u{1}{}\u{1}{:?}\u{1}{:?}\u{1}{:?}\u{1}{}\u{1}{}",
        select.table,
        select.projection,
        filter_column,
        select.distinct,
        select.group_by,
        select.limit,
        select.offset,
        select.order_by.len(),
        select.having_groups.len(),
    )
}

/// Submit+complete one same-shape group (identical table/columns/filter; only needles vary). The batcher
/// admits ONLY all-int4 projections (`classify_batchable_point_lookup` routes mixed int4+text to the
/// per-query path — see its doc), so this prepares the needle-invariant template ONCE, deduplicates
/// needles into a single `equal_any` submission, and answers each request from its needle's sliced
/// result. Any prepare/submit/complete error is fanned out to every waiter (no hung connection).
fn run_group(
    engine: &Engine,
    group: Vec<PointLookupRequest>,
    activity: &PointLookupBatcherActivity,
) {
    // Deduplicate needles into one submission, then map each request back to its needle's result.
    let (distinct_needles, request_result_index) = dedup_needles(&group);
    // lpb-for-shards: a SHARD-resident int4 point-lookup batch has NO single-buffer snapshot, so serve it via
    // the batched cross-shard gather when `shard_batched_point_read_enabled` is ON. `None` = the flag is OFF,
    // the table is not shard-resident, or the gather declined (e.g. a duplicate int4 key) -> fall through to
    // the single-buffer template (single-buffer tables) / the per-query fallback (shard-resident decline).
    match engine.submit_sharded_point_lookups_batched_compact(&group[0].select, &distinct_needles) {
        Ok(Some(batched)) => {
            activity
                .sharded_batched_groups
                .fetch_add(1, AtomicOrdering::Relaxed);
            debug_assert_eq!(batched.needle_count(), distinct_needles.len());
            distribute_results_batched(group, request_result_index, batched);
            return;
        }
        Ok(None) => {}
        Err(err) => {
            // A CUDA/runtime failure is not an eligibility decline. Fan it out directly; never retry the
            // same request through another relational route and thereby mask the originating device fault.
            fail_group(group, err);
            return;
        }
    }
    // A shard-resident gather decline must take the byte-identical per-query GPU route directly.
    // Do not attempt the single-buffer template: a compatibility snapshot may produce a `Ready`
    // result containing structural NULLs, while the flat retained-batch ABI is intentionally
    // non-null i32-only.
    if engine.resident_shard_count(&group[0].select.table) > 0 {
        activity
            .sharded_per_query_fallback_groups
            .fetch_add(1, AtomicOrdering::Relaxed);
        run_group_per_query(engine, group);
        return;
    }
    // Prepare the shared single-buffer template once for the whole group (group is non-empty by construction).
    let template = match engine.prepare_relational_retained_read_template(&group[0].select) {
        Ok(template) => template,
        Err(err) => {
            // A truly non-resident table (the snapshot went away since classification) fails the
            // group uniformly.
            fail_group(group, err);
            return;
        }
    };
    // Defensive: classify admits only all-int4 projections to the batcher, so a mixed int4+text
    // projection must never reach here — the `equal_any` kernel cannot materialize text (and the
    // text-capable general executor is not CUDA-context-safe on this coalescer thread). Guard with a
    // clean per-waiter error rather than feed a text column to the int4 kernel.
    if !template.is_int4_only_projection() {
        let mapped = DbError {
            category: ErrorCategory::Internal,
            message: "batched point lookup received a non-int4 projection (mixed int4+text shapes \
                      must take the per-query path)"
                .to_string(),
        };
        for request in group {
            let _ = request.respond.send(Err(mapped.clone()));
        }
        return;
    }
    if !template.is_flat_i32_batch_safe() {
        run_group_per_query(engine, group);
        return;
    }
    let batched = match submit_and_complete_template_batched(engine, &template, &distinct_needles) {
        Ok(batched) => batched,
        Err(err) => {
            fail_group(group, err);
            return;
        }
    };
    debug_assert_eq!(batched.needle_count(), distinct_needles.len());
    distribute_results_batched(group, request_result_index, batched);
}

/// lpb-for-shards per-query fallback: serve each request in a shard-resident group individually via the
/// engine's per-query select path (BYTE-IDENTICAL to the unbatched path), used when the batched cross-shard
/// gather declines (e.g. a duplicate int4 key makes the per-shard hash decline) and there is no single-buffer
/// template to fall back to. Slower than the batch, but correct — never fails a valid query.
fn run_group_per_query(engine: &Engine, group: Vec<PointLookupRequest>) {
    for request in group {
        let outcome = match engine.execute_relational_select(&request.select) {
            Ok(result) => {
                let columns = result.columns.iter().map(map_column).collect();
                let rows = result
                    .rows
                    .iter()
                    .map(|row| row.iter().cloned().map(map_value).collect())
                    .collect();
                Ok(QueryOutcome::Rows { columns, rows })
            }
            Err(err) => Err(map_execute_error_local(err)),
        };
        let _ = request.respond.send(outcome);
    }
}

/// Deduplicate a group's needles, returning the distinct needles (first-seen order) and, per request,
/// the index of its needle in that distinct list. Identical needles submit to the kernel once.
fn dedup_needles(group: &[PointLookupRequest]) -> (Vec<i32>, Vec<usize>) {
    let mut distinct_needles: Vec<i32> = Vec::new();
    let mut needle_to_index: HashMap<i32, usize> = HashMap::new();
    let mut request_result_index: Vec<usize> = Vec::with_capacity(group.len());
    for request in group {
        let idx = match needle_to_index.get(&request.needle) {
            Some(&idx) => idx,
            None => {
                let idx = distinct_needles.len();
                needle_to_index.insert(request.needle, idx);
                distinct_needles.push(request.needle);
                idx
            }
        };
        request_result_index.push(idx);
    }
    (distinct_needles, request_result_index)
}

/// Dispatch a BATCHED retained-read result to each request: map the SHARED projected schema to wire ONCE
/// (was per-needle), then slice the one flat `RowBlock` by each needle's range to build that request's
/// `QueryOutcome::Rows` — avoiding the N per-needle `RelationalSelectResult` structs + N column re-maps.
/// Byte-identical neutral output to the per-query path. A dropped receiver makes `send` fail harmlessly.
trait PointBatchResultView {
    fn columns(&self) -> &[RelationalColumn];
    fn ncols(&self) -> usize;
    fn needle_count(&self) -> usize;
    fn needle_values(&self, needle: usize) -> &[i32];
}

impl PointBatchResultView for RelationalRetainedBatchResult {
    fn columns(&self) -> &[RelationalColumn] {
        &self.columns
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn needle_count(&self) -> usize {
        self.needle_count()
    }

    fn needle_values(&self, needle: usize) -> &[i32] {
        self.needle_values(needle)
    }
}

impl PointBatchResultView for RelationalPointBatchResult {
    fn columns(&self) -> &[RelationalColumn] {
        self.columns()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn needle_count(&self) -> usize {
        self.needle_count()
    }

    fn needle_values(&self, needle: usize) -> &[i32] {
        self.needle_values(needle)
    }
}

fn distribute_results_batched<B: PointBatchResultView>(
    group: Vec<PointLookupRequest>,
    request_result_index: Vec<usize>,
    batched: B,
) {
    let columns: Vec<_> = batched.columns().iter().map(map_column).collect();
    let ncols = batched.ncols();
    for (request, result_idx) in group.into_iter().zip(request_result_index) {
        let outcome = if result_idx < batched.needle_count() {
            // This needle's projected i32 values as a flat slice -> chunk into rows, map each i32 straight to
            // `DbValue::Int4` (the int4 route is always Int4 — DECISIONS "Result-path optimization": no
            // SqlValue intermediate).
            let vals = batched.needle_values(result_idx);
            let rows: Vec<Vec<_>> = if ncols == 0 {
                Vec::new()
            } else {
                vals.chunks(ncols)
                    .map(|row| row.iter().map(|&v| DbValue::Int4(v)).collect())
                    .collect()
            };
            Ok(QueryOutcome::Rows {
                columns: columns.clone(),
                rows,
            })
        } else {
            Err(DbError {
                category: ErrorCategory::Internal,
                message: "batched point-lookup result slice missing for request".to_string(),
            })
        };
        let _ = request.respond.send(outcome);
    }
}

/// Fan one error out to every waiter in a group (no hung connection).
fn fail_group(group: Vec<PointLookupRequest>, err: ExecuteError) {
    let mapped = map_execute_error_local(err);
    for request in group {
        let _ = request.respond.send(Err(mapped.clone()));
    }
}

/// Submit then complete (BATCHED) a template+needles batch. One call so the read view spans the whole
/// submit→complete window (the engine API takes `&self`).
fn submit_and_complete_template_batched(
    engine: &Engine,
    template: &gpu_db_engine::RelationalRetainedReadTemplate,
    needles: &[i32],
) -> Result<RelationalRetainedBatchResult, ExecuteError> {
    let submission = engine.submit_relational_retained_template_point_lookups(template, needles)?;
    engine.complete_relational_retained_read_submission_batched(submission)
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
    use crate::{execute_on_shared_engine, execute_on_shared_engine_batched, BatchedDispatch};
    use gpu_db_sql::{parse_command, Command};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::TryRecvError;
    use std::sync::Barrier;

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

    /// A small authoritative engine fixture for scheduler/fanout tests. R3-004 publishes the
    /// inserted relation on the device even when optional auto-admission is disabled.
    fn engine_with_table() -> Arc<SharedEngine> {
        let shared = Arc::new(SharedEngine::new());
        // Keep optional warmup disabled; durable DML still establishes mandatory device authority.
        shared.engine.set_auto_admit_on_commit(false);
        execute_on_shared_engine(&shared, "CREATE TABLE t (id INT)").unwrap();
        execute_on_shared_engine(&shared, "INSERT INTO t (id) VALUES (1)").unwrap();
        shared
    }

    #[test]
    fn prepare_error_is_fanned_out_to_the_waiter_not_hung() {
        // An unknown relation fails during prepare; the waiter must get that Err, never hang.
        let engine = engine_with_table();
        let batcher = PointLookupBatcher::with_triggers(engine, 4, Duration::from_millis(5));
        let rx = batcher.enqueue(select("SELECT id FROM missing WHERE id = 1"), 1);
        let outcome = recv_within(rx, Duration::from_secs(2)).expect("waiter must get a response");
        assert!(
            outcome.is_err(),
            "expected an engine error, got {outcome:?}"
        );
    }

    #[test]
    fn every_request_in_a_batch_gets_a_response() {
        // Several lookups coalesce into one batch; each must be answered successfully.
        let engine = engine_with_table();
        let batcher = PointLookupBatcher::with_triggers(engine, 8, Duration::from_millis(5));
        let receivers: Vec<_> = (0..8)
            .map(|needle| batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), needle))
            .collect();
        for rx in receivers {
            let outcome = recv_within(rx, Duration::from_secs(2)).expect("every waiter answered");
            assert!(outcome.is_ok(), "authoritative lookup failed: {outcome:?}");
        }
    }

    #[test]
    fn dropped_receiver_does_not_wedge_the_coalescer() {
        // A client disconnects while parked: drop its receiver. The coalescer's send
        // fails harmlessly and it must keep serving the next request.
        let engine = engine_with_table();
        let batcher = PointLookupBatcher::with_triggers(engine, 1, Duration::from_millis(5));
        let abandoned = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 7);
        drop(abandoned); // client gone before the coalescer answers.
                         // The next request must still be answered.
        let rx = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 8);
        let outcome =
            recv_within(rx, Duration::from_secs(2)).expect("coalescer still serves after a drop");
        assert!(
            outcome.is_ok(),
            "coalescer failed after receiver drop: {outcome:?}"
        );
    }

    #[test]
    fn shutdown_drains_every_queued_request() {
        // Enqueue a burst with a long max_wait so they sit in the queue, then drop the
        // batcher. Every receiver must resolve (a real answer or channel-closed), never
        // hang — the shutdown drain guarantee.
        let engine = engine_with_table();
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

    /// lpb-for-shards WIRING (Step 1 landing): a SHARD-resident int4 point-lookup batch is classified
    /// batchable (flag ON), served end-to-end through the facade batcher by the batched cross-shard gather,
    /// and returns per-needle results BYTE-IDENTICAL to the per-query path (across present / absent /
    /// multi-shard); the WIRED batched path FIRES (`sharded_point_batch_hits` advances). Also covers the
    /// per-query FALLBACK: a duplicate int4 key declines the gather -> the group is served per-query
    /// (multi-row), NOT failed.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU (resident route)"]
    fn wired_sharded_batch_matches_per_query() {
        use gpu_db_engine::Engine;

        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_auto_admit_on_commit(true);
        engine.set_shard_batched_point_read_enabled(true);
        engine.set_shard_index_probe_enabled(true);
        engine.set_shard_size_target(64); // 200 rows -> several shards
        engine
            .execute_text(1, "CREATE TABLE accounts (id INT, balance INT)")
            .unwrap();
        for i in 0..200i64 {
            engine
                .execute_text(
                    (i as u64) + 2,
                    &format!(
                        "INSERT INTO accounts (id, balance) VALUES ({i}, {})",
                        i * 10
                    ),
                )
                .unwrap();
        }
        // Skip if no GPU (auto-admit produced no shards).
        if engine.resident_shard_count("accounts") == 0 {
            return;
        }
        let shared = Arc::new(SharedEngine::from_engine(engine));
        let batcher =
            PointLookupBatcher::with_triggers(Arc::clone(&shared), 16, Duration::from_millis(5));
        let hb = shared.read_engine().unwrap().sharded_point_batch_hits();
        let activity_before = batcher.activity_snapshot();

        let needles = [0i32, 1, 5, 64, 128, 130, 199, 999, -1];
        let sql = |k: i32| format!("SELECT id, balance FROM accounts WHERE id = {k}");
        let mut rxs = Vec::new();
        for &k in &needles {
            match execute_on_shared_engine_batched(&shared, &batcher, &sql(k)) {
                BatchedDispatch::Batched(rx) => rxs.push(rx),
                BatchedDispatch::Immediate(_) => panic!(
                    "shard-resident int4 point lookup + flag ON must BATCH, not take the immediate path"
                ),
            }
        }
        let got: Vec<QueryOutcome> = rxs
            .into_iter()
            .map(|rx| {
                recv_within(rx, Duration::from_secs(5))
                    .expect("answered")
                    .expect("ok")
            })
            .collect();
        assert!(
            shared.read_engine().unwrap().sharded_point_batch_hits() > hb,
            "the WIRED batched sharded path fired (non-vacuity)"
        );
        assert!(
            batcher.activity_snapshot().sharded_batched_groups
                > activity_before.sharded_batched_groups,
            "positive control: the batcher records a served sharded group"
        );
        for (i, &k) in needles.iter().enumerate() {
            let want = execute_on_shared_engine(&shared, &sql(k)).unwrap();
            assert_eq!(got[i], want, "wired batched == per-query for id={k}");
        }

        // A duplicate int4 key remains batch-correct. The device gather may serve it directly or
        // decline to the per-query GPU executor; either route must return every matching row.
        execute_on_shared_engine(&shared, "CREATE TABLE dup (id INT, balance INT)").unwrap();
        execute_on_shared_engine(
            &shared,
            "INSERT INTO dup (id, balance) VALUES (1,10),(1,20),(2,30)",
        )
        .unwrap();
        if shared.read_engine().unwrap().resident_shard_count("dup") > 0 {
            let want_dup =
                execute_on_shared_engine(&shared, "SELECT id, balance FROM dup WHERE id = 1")
                    .unwrap();
            let got_dup = match execute_on_shared_engine_batched(
                &shared,
                &batcher,
                "SELECT id, balance FROM dup WHERE id = 1",
            ) {
                BatchedDispatch::Batched(rx) => recv_within(rx, Duration::from_secs(5))
                    .expect("answered")
                    .expect("ok"),
                BatchedDispatch::Immediate(_) => {
                    panic!("resident duplicate-key shape must enter the batcher before its gather declines")
                }
            };
            assert_eq!(
                got_dup, want_dup,
                "dup-key batched (per-query fallback) == per-query"
            );
            if let QueryOutcome::Rows { rows, .. } = &want_dup {
                assert_eq!(
                    rows.len(),
                    2,
                    "id=1 has 2 rows (fallback served the multi-row result)"
                );
            }
        }
    }

    /// The production mixed read/write gate in regression size: facade reads stay admitted to the
    /// sharded point-lookup batcher while facade INSERTs commit concurrently. Output correctness is
    /// necessary but not sufficient, so the neutral activity snapshot proves the fully-GPU probe,
    /// resident device append, and host-install elision all fired during the overlap.
    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU (mixed resident workload)"]
    fn mixed_read_write_gpu_route_is_non_vacuous() {
        use gpu_db_engine::Engine;

        let engine = Engine::new_local();
        engine.set_shard_residency_enabled(true);
        engine.set_shard_size_target(32);
        engine.set_shard_index_probe_enabled(true);
        engine.set_shard_batched_point_read_enabled(true);
        engine.set_auto_admit_on_commit(true);
        let shared = Arc::new(SharedEngine::from_engine(engine));
        execute_on_shared_engine(
            &shared,
            "CREATE TABLE mix (id INT PRIMARY KEY, balance INT)",
        )
        .unwrap();
        for id in 0..100usize {
            let balance = if id == 1 {
                "NULL".to_owned()
            } else {
                (id * 7).to_string()
            };
            execute_on_shared_engine(
                &shared,
                &format!("INSERT INTO mix (id, balance) VALUES ({id}, {balance})"),
            )
            .unwrap();
        }
        if shared.gpu_native_activity_snapshot("mix").resident_shards == 0 {
            return;
        }

        let batcher = Arc::new(PointLookupBatcher::with_triggers(
            Arc::clone(&shared),
            32,
            Duration::ZERO,
        ));
        let before = shared.gpu_native_activity_snapshot("mix");
        let active_readers = Arc::new(AtomicUsize::new(1));
        let reader_started = Arc::new(AtomicUsize::new(0));
        let active_writers = Arc::new(AtomicUsize::new(0));
        let writer_done = Arc::new(AtomicBool::new(false));
        let overlapping_writes = Arc::new(AtomicUsize::new(0));
        let overlapping_reads = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(3));

        let reader = {
            let shared = Arc::clone(&shared);
            let batcher = Arc::clone(&batcher);
            let active_readers = Arc::clone(&active_readers);
            let reader_started = Arc::clone(&reader_started);
            let active_writers = Arc::clone(&active_writers);
            let writer_done = Arc::clone(&writer_done);
            let overlapping_reads = Arc::clone(&overlapping_reads);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                reader_started.store(1, Ordering::Release);
                let mut turn = 0usize;
                loop {
                    let id = turn % 100;
                    let sql = format!("SELECT id FROM mix WHERE id = {id}");
                    let outcome = match execute_on_shared_engine_batched(&shared, &batcher, &sql) {
                        BatchedDispatch::Batched(receiver) => receiver
                            .blocking_recv()
                            .expect("batcher stayed alive")
                            .expect("batched read succeeded"),
                        BatchedDispatch::Immediate(_) => {
                            panic!("resident mixed read bypassed the production batcher")
                        }
                    };
                    match outcome {
                        QueryOutcome::Rows { rows, .. } => {
                            assert_eq!(rows, vec![vec![DbValue::Int4(id as i32)]])
                        }
                        other => panic!("unexpected point-read outcome: {other:?}"),
                    }
                    if active_writers.load(Ordering::Acquire) > 0 {
                        overlapping_reads.fetch_add(1, Ordering::Relaxed);
                    }
                    turn += 1;
                    if turn >= 300 && writer_done.load(Ordering::Acquire) {
                        break;
                    }
                }
                active_readers.store(0, Ordering::Release);
            })
        };
        let writer = {
            let shared = Arc::clone(&shared);
            let active_readers = Arc::clone(&active_readers);
            let reader_started = Arc::clone(&reader_started);
            let active_writers = Arc::clone(&active_writers);
            let writer_done = Arc::clone(&writer_done);
            let overlapping_writes = Arc::clone(&overlapping_writes);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                while reader_started.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }
                active_writers.fetch_add(1, Ordering::Release);
                let gpu_before_writes = shared
                    .gpu_native_activity_snapshot("mix")
                    .sharded_gpu_probe_batches;
                for offset in 0..20usize {
                    let id = 10_000 + offset;
                    execute_on_shared_engine(
                        &shared,
                        &format!("INSERT INTO mix (id, balance) VALUES ({id}, {})", id * 3),
                    )
                    .unwrap();
                    if active_readers.load(Ordering::Acquire) > 0 {
                        overlapping_writes.fetch_add(1, Ordering::Relaxed);
                    }
                }
                let gpu_after_writes = shared
                    .gpu_native_activity_snapshot("mix")
                    .sharded_gpu_probe_batches;
                active_writers.fetch_sub(1, Ordering::Release);
                writer_done.store(true, Ordering::Release);
                gpu_after_writes.saturating_sub(gpu_before_writes)
            })
        };

        barrier.wait();
        let gpu_batches_during_writes = writer.join().expect("writer did not panic");
        reader.join().expect("reader did not panic");
        let after = shared.gpu_native_activity_snapshot("mix");
        let gpu_batches = after.sharded_gpu_probe_batches - before.sharded_gpu_probe_batches;
        let sharded_batches = after.sharded_point_batches - before.sharded_point_batches;
        let activity = batcher.activity_snapshot();
        assert!(
            gpu_batches > 0,
            "the fully-GPU multi-shard point probe fired"
        );
        assert!(
            gpu_batches_during_writes > 0,
            "a fully-GPU point batch completed inside the exact writer-active interval"
        );
        assert_eq!(
            gpu_batches, sharded_batches,
            "visibility-sensitive append windows remain on the dense GPU probe"
        );
        assert_eq!(
            sharded_batches, activity.sharded_batched_groups,
            "every batcher sharded group stayed on a batched resident route"
        );
        assert_eq!(
            activity.sharded_per_query_fallback_groups, 0,
            "no sharded group fell back to per-query execution"
        );
        assert!(
            after.open_shard_append_commits > before.open_shard_append_commits,
            "concurrent writes appended to resident device memory"
        );
        assert_eq!(
            after.device_authoritative_commits - before.device_authoritative_commits,
            20,
            "every concurrent write published device-authoritative state"
        );
        assert_eq!(
            overlapping_writes.load(Ordering::Relaxed),
            20,
            "every write committed while the continuously-running reader was live"
        );
        assert!(
            overlapping_reads.load(Ordering::Relaxed) > 0,
            "at least one batched resident read completed while the writer was active"
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU (resident route)"]
    fn mixed_int4_text_point_lookup_routes_to_per_query_not_batcher() {
        use gpu_db_engine::Engine;

        // Regression (adversarial audit, Tier 1): a mixed int4+text projection
        // (`int4_equality_mixed_column_projection`) must NOT be batched — the resident `equal_any` kernel
        // cannot materialize text, and the text-capable general executor errored CUDA 201 (invalid
        // context) on the coalescer thread (a pre-existing latent constraint). `classify` routes it to
        // the per-query path instead. Build + warm a table with a text column AND a NULL row; confirm the
        // dispatch is Immediate (per-query) and byte-identical to the direct per-query path.
        let mut engine = Engine::new_local();
        engine
            .execute_text(1, "CREATE TABLE m (id INT, label TEXT)")
            .unwrap();
        engine
            .execute_text(
                2,
                "INSERT INTO m (id, label) VALUES (1, 'one'), (2, 'two'), (3, NULL)",
            )
            .unwrap();
        engine.populate_relational_residency_snapshot("m").unwrap();
        // Skip if the box has no GPU device memory or the mixed shape is not admitted to the route.
        let probe = parse_command("SELECT id, label FROM m WHERE id = 2").unwrap();
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

        // A mixed int4+text point lookup (including the NULL-label row) must take the Immediate
        // (per-query) dispatch, NOT the batcher, and return rows byte-identical to the per-query path.
        for sql in [
            "SELECT id, label FROM m WHERE id = 2",
            "SELECT id, label FROM m WHERE id = 3",
        ] {
            let outcome = match execute_on_shared_engine_batched(&shared, &batcher, sql) {
                BatchedDispatch::Immediate(result) => result.unwrap(),
                BatchedDispatch::Batched(_) => {
                    panic!("mixed int4+text point lookup must route to per-query, not the batcher: {sql}")
                }
            };
            let reference = execute_on_shared_engine(&shared, sql).unwrap();
            assert_eq!(
                outcome, reference,
                "mixed int4+text must match per-query: {sql}"
            );
        }
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
        let engine = engine_with_table();
        let max_wait = Duration::from_secs(1);
        let batcher = PointLookupBatcher::with_triggers(engine, 32, max_wait);
        let start = Instant::now();
        let rx = batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), 1);
        // The device-authoritative lookup must resolve fast.
        let outcome = recv_within(rx, Duration::from_secs(2)).expect("waiter must get a response");
        let elapsed = start.elapsed();
        assert!(outcome.is_ok(), "authoritative lookup failed: {outcome:?}");
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
        let engine = engine_with_table();
        let (obs_tx, obs_rx) = mpsc::channel::<usize>();
        // Count trigger (64) above the burst (16) so coalescing is via the wait,
        // not the count trigger; a 200ms ceiling gives the burst time to gather.
        let batcher = Arc::new(PointLookupBatcher::with_triggers_observed(
            engine,
            64,
            Duration::from_millis(200),
            obs_tx,
        ));
        let n = 16usize;
        let start = Arc::new(Barrier::new(n + 1));
        let (receiver_tx, receiver_rx) = mpsc::channel();
        let workers: Vec<_> = (0..n)
            .map(|needle| {
                let batcher = Arc::clone(&batcher);
                let start = Arc::clone(&start);
                let receiver_tx = receiver_tx.clone();
                thread::spawn(move || {
                    start.wait();
                    receiver_tx
                        .send(
                            batcher.enqueue(select("SELECT id FROM t WHERE id = 1"), needle as i32),
                        )
                        .unwrap();
                })
            })
            .collect();
        drop(receiver_tx);
        start.wait();
        let receivers: Vec<_> = receiver_rx.iter().collect();
        for worker in workers {
            worker.join().unwrap();
        }
        // Every request is still answered (completeness preserved).
        for rx in receivers {
            let outcome = recv_within(rx, Duration::from_secs(3)).expect("every waiter answered");
            assert!(outcome.is_ok(), "authoritative lookup failed: {outcome:?}");
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
        let engine = engine_with_table();
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
        assert!(
            o1.is_ok() && o2.is_ok(),
            "authoritative lookups failed: {o1:?} {o2:?}"
        );
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
