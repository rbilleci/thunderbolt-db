//! Chronicle-shaped OLTP client lane over the library `FuaWalSegment`.
//!
//! Same client contract as `file-wal-client-coalesced-durable`: logical clients
//! issue one write at a time and each request acks only when its WAL frame is
//! durable AND the volatile store has applied it. The durable path is the
//! production `FuaWalSegment` + `FuaFencePool` (pre-written extents, anonymous
//! staging, FUA write-through fence lanes, contiguous durable cut) — see
//! docs/WRITE_CONVEYOR.md "FUA-Pipelined Durable Lane" for the physics.
//!
//! The appender applies the two load-bearing scheduling laws:
//! - ADAPTIVE FRAME SIZING: `frame ~= pending / fence_lanes`, clamped, so the
//!   backlog spreads across the fence pool (the drive's fast FUA mode needs
//!   the pool full; maximally-packed frames starve it at moderate client
//!   counts).
//! - FENCE-POOL PACING: publish only into a free fence lane, accumulating
//!   while all lanes are busy. Without this the closed loop convoys into
//!   serial fence latency.
//!
//! ```bash
//! cargo run --release -p gpu_db_write_conveyor --example fua_wal_client_bench
//! CONVEYOR_CLIENTS=2048 CONVEYOR_FUA_QD=16 CONVEYOR_EVENTS=4000000 \
//!   cargo run --release -p gpu_db_write_conveyor --example fua_wal_client_bench
//! ```

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::UnsafeCell;
use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use gpu_db_write_conveyor::{
    intent_for, recover_wal_segment_by_scan, stats_for_range, FuaWalSegment, FuaWalSegmentConfig,
    WriteIntent,
};

struct SharedBytes {
    ptr: *mut u8,
    layout: Layout,
}
unsafe impl Send for SharedBytes {}
unsafe impl Sync for SharedBytes {}
impl SharedBytes {
    fn zeroed(len: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(len.max(1), align).expect("shared buffer layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "shared buffer allocation failed");
        Self { ptr, layout }
    }
}
impl Drop for SharedBytes {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

struct IntentSlot(UnsafeCell<WriteIntent>);
unsafe impl Sync for IntentSlot {}

/// Volatile append-store: fixed-width columns, row id == client_seq.
struct VolatileStore {
    keys: SharedBytes,
    value0: SharedBytes,
    seqs: SharedBytes,
}
impl VolatileStore {
    fn new(rows: usize) -> Self {
        Self {
            keys: SharedBytes::zeroed(rows * 8, 64),
            value0: SharedBytes::zeroed(rows * 8, 64),
            seqs: SharedBytes::zeroed(rows * 8, 64),
        }
    }
    /// # Safety
    /// Single apply worker; row written exactly once.
    unsafe fn apply(&self, row: usize, intent: &WriteIntent) {
        unsafe {
            (self.keys.ptr.cast::<u64>()).add(row).write(intent.key);
            (self.value0.ptr.cast::<u64>())
                .add(row)
                .write(intent.value0);
            (self.seqs.ptr.cast::<u64>())
                .add(row)
                .write(intent.client_seq);
        }
    }
    fn validate(&self, rows: u64) -> Result<(), String> {
        // deterministic spot check: every 997th row plus the last row
        let mut row = 0_u64;
        while row < rows {
            let expected = intent_for(row);
            let (key, value0, seq) = unsafe {
                (
                    (self.keys.ptr.cast::<u64>()).add(row as usize).read(),
                    (self.value0.ptr.cast::<u64>()).add(row as usize).read(),
                    (self.seqs.ptr.cast::<u64>()).add(row as usize).read(),
                )
            };
            if key != expected.key || value0 != expected.value0 || seq != expected.client_seq {
                return Err(format!(
                    "store mismatch at row {row}: key={key} value0={value0} seq={seq}"
                ));
            }
            row = if row + 997 < rows || row == rows - 1 {
                row + 997
            } else {
                rows - 1
            };
        }
        Ok(())
    }
}

struct Shared {
    // client ingress ring
    slots: Vec<IntentSlot>,
    ready: Vec<AtomicU64>, // 0 = empty, seq+1 = ready
    ring_mask: u64,
    next_seq: AtomicU64,     // client claim cursor
    consumed_seq: AtomicU64, // appender's contiguous drain frontier (slot reuse gate)
    // durable WAL
    segment: Arc<FuaWalSegment>,
    // volatile store
    store: VolatileStore,
    applied_seq: AtomicU64,
    // Chronicle-style wait strategy: clients PARK on ack instead of
    // yielding. Thousands of yield_now waiters turn ack observation into a
    // scheduler round-trip (measured to dominate tail latency and starve the
    // ring). A/B result: per-seq park/unpark beat bucketed condvar cohorts
    // decisively (condvars pay a thundering herd on every partial-bucket
    // notify plus wake serialization through the bucket mutex). Registry
    // slots are Mutex<Option<Thread>>; the park loop re-checks after
    // registering, so a wake can never be missed; park_timeout bounds any
    // residual race.
    waiters: Vec<Mutex<Option<std::thread::Thread>>>,
    // wake-delivery diagnostics: waker stamps before unpark, client samples
    // stamp->resume (CONVEYOR_STAGE_TIMINGS=1)
    wake_base: Instant,
    unpark_ns: Vec<AtomicU64>,
    events: u64,
}

impl Shared {
    fn acked(&self, seq: u64) -> bool {
        self.segment.durable_record_seq() > seq && self.applied_seq.load(Ordering::Acquire) > seq
    }

    fn ack_frontier(&self) -> u64 {
        self.segment
            .durable_record_seq()
            .min(self.applied_seq.load(Ordering::Acquire))
    }
}

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

fn parse_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => default,
    }
}

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    debug_assert!(!sorted.is_empty());
    sorted[(sorted.len() - 1) * pct / 100]
}

fn format_duration(duration: Duration) -> String {
    let ns = duration.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.2}us", ns as f64 / 1_000.0)
    } else {
        format!("{:.3}ms", ns as f64 / 1_000_000.0)
    }
}

fn print_distribution(label: &str, samples: &mut [Duration]) {
    if samples.is_empty() {
        println!("    {label:<26} no samples");
        return;
    }
    samples.sort_unstable();
    println!(
        "    {label:<26} p50={} p90={} p99={} max={} samples={}",
        format_duration(percentile(samples, 50)),
        format_duration(percentile(samples, 90)),
        format_duration(percentile(samples, 99)),
        format_duration(*samples.last().expect("non-empty samples")),
        samples.len()
    );
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn Error>> {
    let events: u64 = parse_env("CONVEYOR_EVENTS", 5_000_000_u64);
    let clients: usize = parse_env("CONVEYOR_CLIENTS", 512_usize).max(1);
    let fence_qd: usize = parse_env("CONVEYOR_FUA_QD", 16_usize).max(1);
    let block_size: usize = parse_env("CONVEYOR_CLIENT_WAL_BLOCK", 62_usize).max(1);
    // Frame floor: small enough that `clients` closed-loop requests split into
    // >= fence-qd frames (the drive's fast FUA mode needs the pool full), but
    // bounded below so the file capacity worst case stays sane.
    let min_records_default = (clients / (2 * fence_qd)).clamp(4, 16);
    let min_records: usize =
        parse_env("CONVEYOR_CLIENT_MIN_RECORDS_PER_BLOCK", min_records_default)
            .clamp(1, block_size)
            .min(clients);
    let append_group_us: u64 = parse_env("CONVEYOR_CLIENT_APPEND_GROUP_US", 25_u64);
    let park_window =
        Duration::from_micros(parse_env("CONVEYOR_PARK_TIMEOUT_US", 5000_u64).max(10));
    // 0 = one OS thread per client (parked-waiter ack); N>0 = event-loop
    // driver threads each multiplexing clients/N logical clients (no parks,
    // no wakes: acks observed by sweeping two atomics per logical client).
    // The per-request futex wake+park pair (~4-8us kernel time, serialized
    // through the waker pool) is unpayable at millions of acks/s — measured
    // as ~670us of waker scan lag at 2048 threads. Event-loop drivers are
    // the Chronicle/production shape.
    let driver_threads: usize = parse_env("CONVEYOR_CLIENT_DRIVER_THREADS", 64_usize).min(clients);
    let sample_stride: u64 = parse_env(
        "CONVEYOR_CLIENT_LATENCY_SAMPLE_STRIDE",
        (events / 2_000_000).max(1),
    )
    .max(1);
    let validate = parse_bool("CONVEYOR_RAW_VALIDATE", true);
    let keep_file = parse_bool("CONVEYOR_KEEP_FILE", false);
    let (path, generated_path) = match std::env::var("CONVEYOR_FILE") {
        Ok(path) => (PathBuf::from(path), false),
        Err(_) => (
            PathBuf::from(format!("target/fua-wal-client-{}.dat", std::process::id())),
            true,
        ),
    };
    if events == 0 {
        return Err("CONVEYOR_EVENTS must be greater than zero".into());
    }
    if clients as u64 >= 65_536 {
        return Err("CONVEYOR_CLIENTS must be below the waiter-ring coverage (65536)".into());
    }
    // worst case: every block carries only the liveness floor of records
    let capacity_records = events
        .div_ceil(min_records as u64)
        .saturating_add(1024)
        .saturating_mul(block_size as u64);
    if capacity_records > usize::MAX as u64 {
        return Err("FUA WAL capacity exceeds usize".into());
    }

    println!("fua WAL client benchmark (library FuaWalSegment)");
    println!(
        "  events={events} clients={clients} fence-qd={fence_qd} wal-block={block_size} min-block={min_records} append-group={append_group_us}us"
    );

    // CONVEYOR_RECYCLE=1 reuses an existing pre-written segment file under a
    // fresh segment id (steady-state production shape: prep amortized to
    // zero). Requires CONVEYOR_FILE pointing at a file created by a prior
    // CONVEYOR_KEEP_FILE=1 run with identical geometry.
    let recycle = parse_bool("CONVEYOR_RECYCLE", false);
    let recycle_segment_id: u64 = parse_env("CONVEYOR_RECYCLE_SEGMENT_ID", 2_u64);
    if path.exists() && !recycle {
        if !generated_path && !parse_bool("CONVEYOR_OVERWRITE", false) {
            return Err(format!(
                "{} already exists; set CONVEYOR_OVERWRITE=1 to replace it",
                path.display()
            )
            .into());
        }
        std::fs::remove_file(&path)?;
    }
    let setup_start = Instant::now();
    let config = FuaWalSegmentConfig {
        path: path.clone(),
        segment_id: if recycle { recycle_segment_id } else { 1 },
        records: capacity_records as usize,
        block_size,
    };
    let segment = unsafe {
        if recycle {
            FuaWalSegment::recycle(config)?
        } else {
            FuaWalSegment::create(config)?
        }
    };
    let setup = setup_start.elapsed();

    let ring_capacity: u64 = 65_536;
    let shared = Arc::new(Shared {
        slots: (0..ring_capacity)
            .map(|_| IntentSlot(UnsafeCell::new(WriteIntent::default())))
            .collect(),
        ready: (0..ring_capacity).map(|_| AtomicU64::new(0)).collect(),
        ring_mask: ring_capacity - 1,
        next_seq: AtomicU64::new(0),
        consumed_seq: AtomicU64::new(0),
        segment: Arc::clone(&segment),
        store: VolatileStore::new(events as usize),
        applied_seq: AtomicU64::new(0),
        waiters: (0..ring_capacity).map(|_| Mutex::new(None)).collect(),
        wake_base: Instant::now(),
        unpark_ns: (0..ring_capacity).map(|_| AtomicU64::new(0)).collect(),
        events,
    });
    let barrier_clients = if driver_threads > 0 {
        driver_threads
    } else {
        clients
    };
    let start_barrier = Arc::new(Barrier::new(barrier_clients + 1));

    let stage_timings = if parse_bool("CONVEYOR_STAGE_TIMINGS", false) {
        Some(segment.enable_stage_timings())
    } else {
        None
    };

    // --- fence lanes (library pool) ---
    let pool = segment.spawn_fence_pool(fence_qd);

    // --- appender: drain contiguous ready prefix into published WAL frames ---
    let appender_thread = {
        let shared = Arc::clone(&shared);
        let mut appender = segment.appender();
        std::thread::spawn(move || -> io::Result<u64> {
            let mut next_append: u64 = 0;
            let mut pending_since: Option<Instant> = None;
            let group_window = Duration::from_micros(append_group_us);
            let mut frame: Vec<WriteIntent> = Vec::with_capacity(block_size);
            let mut blocks = 0_u64;
            while next_append < shared.events {
                // FENCE-POOL PACING: accumulate while every lane is busy.
                while shared.segment.free_fence_slots(fence_qd) == 0 {
                    if shared.segment.fence_failed() {
                        return Err(io::Error::other("FUA fence lane failed"));
                    }
                    std::hint::spin_loop();
                }
                // ADAPTIVE FRAME SIZING: spread the backlog across the pool.
                // Frame target absorbs the POPULATION into the pipeline:
                // lanes*frame ~= clients keeps the standing queue out of the
                // ring (a pending-based target self-defeats — it sizes frames
                // from the ring leftover, so the leftover never drains).
                // The pending term still grows frames past the population
                // share when the ring backs up at saturation.
                let pending_total = shared
                    .next_seq
                    .load(Ordering::Relaxed)
                    .min(shared.events)
                    .saturating_sub(next_append);
                let population_share = clients.div_ceil(fence_qd);
                let target_frame = ((pending_total as usize).div_ceil(fence_qd))
                    .max(population_share)
                    .clamp(min_records.max(1), block_size);
                // contiguous ready prefix, capped at the target frame
                let mut avail = 0_usize;
                while avail < target_frame {
                    let seq = next_append + avail as u64;
                    if seq >= shared.events {
                        break;
                    }
                    let slot = (seq & shared.ring_mask) as usize;
                    if shared.ready[slot].load(Ordering::Acquire) != seq + 1 {
                        break;
                    }
                    avail += 1;
                }
                if avail == 0 {
                    std::hint::spin_loop();
                    continue;
                }
                let run_tail = next_append + (avail as u64) == shared.events;
                if avail < min_records.min(target_frame) && !run_tail {
                    // liveness floor: wait briefly for a fuller frame, then
                    // ship whatever is contiguous
                    let since = *pending_since.get_or_insert_with(Instant::now);
                    if since.elapsed() < group_window {
                        std::hint::spin_loop();
                        continue;
                    }
                }
                pending_since = None;
                frame.clear();
                for offset in 0..avail as u64 {
                    let slot = ((next_append + offset) & shared.ring_mask) as usize;
                    frame.push(unsafe { shared.slots[slot].0.get().read() });
                }
                appender.publish_intents(&frame)?;
                blocks += 1;
                next_append += avail as u64;
                shared.consumed_seq.store(next_append, Ordering::Release);
            }
            appender.finish();
            Ok(blocks)
        })
    };

    // --- store apply worker: ordered block apply into fixed-width columns ---
    let store_thread = {
        let shared = Arc::clone(&shared);
        std::thread::spawn(move || {
            let mut block: u64 = 0;
            let mut buffer: Vec<WriteIntent> = Vec::new();
            while shared.applied_seq.load(Ordering::Relaxed) < shared.events {
                let Some(first_seq) = shared.segment.read_published_block_into(block, &mut buffer)
                else {
                    if shared.segment.fence_failed() {
                        return;
                    }
                    std::hint::spin_loop();
                    continue;
                };
                for (offset, intent) in buffer.iter().enumerate() {
                    unsafe {
                        shared.store.apply(first_seq as usize + offset, intent);
                    }
                }
                shared
                    .applied_seq
                    .store(first_seq + buffer.len() as u64, Ordering::Release);
                block += 1;
            }
        })
    };

    // --- waker threads: unpark the cohort covered by each frontier advance
    // (thread-per-client mode only; drivers poll and never park — audit
    // finding: idle wakers would spin at 100% and pollute measurements) ---
    let waker_threads: usize = if driver_threads > 0 {
        0
    } else {
        parse_env("CONVEYOR_WAKER_THREADS", 8_usize).max(1)
    };
    let waker_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let waker_handles: Vec<_> = (0..waker_threads)
        .map(|lane| {
            let shared = Arc::clone(&shared);
            let stop = Arc::clone(&waker_stop);
            let wake_lag_sampling = stage_timings.is_some();
            std::thread::spawn(move || {
                // each waker owns seqs where seq % waker_threads == lane
                let mut woken: u64 = lane as u64;
                loop {
                    let frontier = shared.ack_frontier();
                    while woken < frontier {
                        let slot = (woken & shared.ring_mask) as usize;
                        if let Some(thread) = shared.waiters[slot]
                            .lock()
                            .expect("waiter slot poisoned")
                            .take()
                        {
                            if wake_lag_sampling {
                                shared.unpark_ns[slot].store(
                                    shared.wake_base.elapsed().as_nanos() as u64,
                                    Ordering::Relaxed,
                                );
                            }
                            thread.unpark();
                        }
                        woken += waker_threads as u64;
                    }
                    if (frontier >= shared.events && woken >= shared.events)
                        || stop.load(Ordering::Acquire)
                        || shared.segment.fence_failed()
                    {
                        return;
                    }
                    std::hint::spin_loop();
                }
            })
        })
        .collect();

    // --- closed-loop clients: event-loop drivers or thread-per-client ---
    let driver_mode = driver_threads > 0;
    let driver_handles: Vec<_> = if driver_mode {
        let per_driver = clients.div_ceil(driver_threads);
        (0..driver_threads)
            .map(|driver| {
                let shared = Arc::clone(&shared);
                let barrier = Arc::clone(&start_barrier);
                let logical = per_driver.min(clients - (driver * per_driver).min(clients));
                std::thread::spawn(move || {
                    enum Logical {
                        Idle,
                        Waiting { seq: u64, ingress: Instant },
                        Done,
                    }
                    let mut states: Vec<Logical> = (0..logical).map(|_| Logical::Idle).collect();
                    let mut latencies = Vec::new();
                    let wake_lags = Vec::new();
                    let mut done = 0_usize;
                    barrier.wait();
                    while done < logical {
                        let mut progressed = false;
                        for state in states.iter_mut() {
                            // retire a completed wait
                            if let Logical::Waiting { seq, ingress } = state {
                                if shared.acked(*seq) {
                                    if seq.is_multiple_of(sample_stride) {
                                        latencies.push(ingress.elapsed());
                                    }
                                    *state = Logical::Idle;
                                    progressed = true;
                                } else if shared.segment.fence_failed() {
                                    return (latencies, wake_lags);
                                }
                            }
                            // issue the next request
                            if matches!(state, Logical::Idle) {
                                let seq = shared.next_seq.fetch_add(1, Ordering::Relaxed);
                                if seq >= shared.events {
                                    *state = Logical::Done;
                                    done += 1;
                                    continue;
                                }
                                while seq - shared.consumed_seq.load(Ordering::Acquire)
                                    > shared.ring_mask
                                {
                                    std::thread::yield_now();
                                }
                                let ingress = Instant::now();
                                let ring_slot = (seq & shared.ring_mask) as usize;
                                unsafe {
                                    shared.slots[ring_slot].0.get().write(intent_for(seq));
                                }
                                shared.ready[ring_slot].store(seq + 1, Ordering::Release);
                                *state = Logical::Waiting { seq, ingress };
                                progressed = true;
                            }
                        }
                        if !progressed {
                            std::hint::spin_loop();
                        }
                    }
                    (latencies, wake_lags)
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    // --- thread-per-client mode ---
    let client_handles: Vec<_> = if driver_mode {
        Vec::new()
    } else {
        (0..clients)
            .map(|_| {
                let shared = Arc::clone(&shared);
                let barrier = Arc::clone(&start_barrier);
                std::thread::spawn(move || {
                    let mut latencies = Vec::new();
                    let mut wake_lags = Vec::new();
                    barrier.wait();
                    loop {
                        let seq = shared.next_seq.fetch_add(1, Ordering::Relaxed);
                        if seq >= shared.events {
                            return (latencies, wake_lags);
                        }
                        // slot reuse gate
                        while seq - shared.consumed_seq.load(Ordering::Acquire) > shared.ring_mask {
                            std::thread::yield_now();
                        }
                        let ingress = Instant::now();
                        let slot = (seq & shared.ring_mask) as usize;
                        unsafe {
                            shared.slots[slot].0.get().write(intent_for(seq));
                        }
                        shared.ready[slot].store(seq + 1, Ordering::Release);
                        // ack = durable WAL frame + volatile store applied; park
                        // until a waker covers this seq (missed-wake-safe:
                        // register, re-check, then park with a bounded timeout)
                        while !shared.acked(seq) {
                            if shared.segment.fence_failed() {
                                // durability is wedged; abort so the appender's
                                // error can surface through main
                                return (latencies, wake_lags);
                            }
                            let waiter = &shared.waiters[(seq & shared.ring_mask) as usize];
                            *waiter.lock().expect("waiter slot poisoned") =
                                Some(std::thread::current());
                            if shared.acked(seq) || shared.segment.fence_failed() {
                                waiter.lock().expect("waiter slot poisoned").take();
                                continue;
                            }
                            std::thread::park_timeout(park_window);
                            waiter.lock().expect("waiter slot poisoned").take();
                        }
                        if seq.is_multiple_of(sample_stride) {
                            latencies.push(ingress.elapsed());
                            let slot = (seq & shared.ring_mask) as usize;
                            let stamped = shared.unpark_ns[slot].swap(0, Ordering::Relaxed);
                            if stamped != 0 {
                                let now = shared.wake_base.elapsed().as_nanos() as u64;
                                wake_lags.push(Duration::from_nanos(now.saturating_sub(stamped)));
                            }
                        }
                    }
                })
            })
            .collect()
    };

    start_barrier.wait();
    let run_start = Instant::now();
    let mut client_latencies = Vec::new();
    let mut wake_lags = Vec::new();
    for handle in client_handles.into_iter().chain(driver_handles) {
        let (lat, lags) = handle.join().expect("client thread panicked");
        client_latencies.extend(lat);
        wake_lags.extend(lags);
    }
    let elapsed = run_start.elapsed();
    waker_stop.store(true, Ordering::Release);
    for handle in waker_handles {
        handle.join().expect("waker thread panicked");
    }
    let blocks = appender_thread.join().expect("appender panicked")?;
    let fences = pool.join()?;
    store_thread.join().expect("store worker panicked");

    assert_eq!(segment.durable_blocks(), blocks);
    assert_eq!(segment.durable_record_seq(), events);

    // --- validation: scan recovery + store columns ---
    let recover_elapsed = if validate {
        let recovery_start = Instant::now();
        let recovered = recover_wal_segment_by_scan(&path)?;
        let recover_elapsed = recovery_start.elapsed();
        let expected = stats_for_range(0, events);
        if recovered.recovered_blocks != blocks
            || recovered.recovered_records != events
            || recovered.stats != expected
        {
            return Err(format!(
                "FUA WAL recovery mismatch: recovered_blocks={} recovered_records={} expected_blocks={blocks} expected_records={events}",
                recovered.recovered_blocks, recovered.recovered_records
            )
            .into());
        }
        shared.store.validate(events).map_err(io::Error::other)?;
        Some(recover_elapsed)
    } else {
        None
    };
    if !keep_file {
        std::fs::remove_file(&path)?;
    }

    let throughput = events as f64 / elapsed.as_secs_f64();
    let ns_per_write = elapsed.as_nanos() as f64 / events as f64;
    let avg_block = events as f64 / blocks as f64;
    println!(
        "fua-wal-client-durable  {:>10.3} M/s {:>9.2} ns/write  elapsed={:.3}s setup={:.3}s clients={clients} fence-qd={fence_qd} wal-block={block_size} blocks={blocks} avg-block={avg_block:.1} fences={fences} ack=fua-fenced-wal+volatile-store-applied{}",
        throughput / 1_000_000.0,
        ns_per_write,
        elapsed.as_secs_f64(),
        setup.as_secs_f64(),
        match recover_elapsed {
            Some(elapsed) => format!(" recover={:.3}s", elapsed.as_secs_f64()),
            None => " recover=skipped".to_string(),
        }
    );
    print_distribution("client->durable-ack", &mut client_latencies);
    print_distribution("unpark->resume (wake lag)", &mut wake_lags);
    if let Some(timings) = stage_timings {
        let stages = timings.block_stages(blocks);
        let mut queue: Vec<Duration> = stages.iter().map(|s| Duration::from_nanos(s.0)).collect();
        let mut fence: Vec<Duration> = stages.iter().map(|s| Duration::from_nanos(s.1)).collect();
        let mut cut: Vec<Duration> = stages.iter().map(|s| Duration::from_nanos(s.2)).collect();
        print_distribution("publish->fence-start", &mut queue);
        print_distribution("fence write (FUA)", &mut fence);
        print_distribution("fence-done->durable-cut", &mut cut);
    }
    Ok(())
}
