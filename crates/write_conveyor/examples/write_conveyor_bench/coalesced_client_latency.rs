use super::super::{
    assert_expected_stats, coalesced_client_label, durable_sync_mode_label, duration_ns,
    intent_for, lane_block_count, lane_blocks_between, manager_lane_dir, prepare_journal_path,
    prepare_manager_dir, recover_wal_manager_by_scan, recover_wal_segment_by_scan, report_extra,
    stats_for_range, striped_global_prefix_from_lane_counts, wait_for_completion_or_failure, Arc,
    AtomicBool, AtomicU64, Barrier, BlockStageTimeline, ChronicleCutSnapshot, ClientAckMode,
    ClientAppendRingSlot, ClientLatencySamples, ClientRunOptions, CoalescedStageCounters,
    CoalescedWalBackend, ContiguousCompletionBarrier, DrainStats, DurableFlushReason,
    DurableSyncCounters, Duration, Error, FailureGuard, Instant, ManagerPublishedBlockSlot,
    MappedWalSegment, OpenShardAppendStore, Ordering, WalDataSyncMode, WalPrewriteCounters,
    WalPublishedBlock, WalSegmentManager, WalSegmentManagerConfig, WriteIntent,
};

pub(in super::super) fn run_file_wal_client_coalesced_latency(
    events: u64,
    block_size: usize,
    clients: usize,
    workers: usize,
    opts: ClientRunOptions<'_>,
    wal_backend: CoalescedWalBackend,
) -> Result<(), Box<dyn Error>> {
    let file_opts = opts.file;
    let ack_mode = opts.ack_mode;
    let durable_enabled = ack_mode.needs_durable() || opts.background_durable;
    let requested_durable_lanes = opts.durable_lanes.max(1);
    let durable_lane_count = if wal_backend == CoalescedWalBackend::Manager && durable_enabled {
        requested_durable_lanes
    } else {
        1
    };
    if requested_durable_lanes > 1
        && !(wal_backend == CoalescedWalBackend::Manager && durable_enabled)
    {
        return Err(
            "CONVEYOR_DURABLE_LANES>1 is currently supported only for manager-backed coalesced WAL with durable or background-durable enabled"
                .into(),
        );
    }
    if durable_lane_count > 1 && opts.durable_sync_mode == WalDataSyncMode::PrewriteAndFileData {
        return Err("CONVEYOR_DURABLE_LANES>1 does not yet support prewrite-and-file-data".into());
    }
    let _file_cleanup;
    let _manager_cleanup;
    let manager_configs;
    match wal_backend {
        CoalescedWalBackend::Segment => {
            _file_cleanup = Some(prepare_journal_path(
                file_opts.path,
                file_opts.keep_file,
                file_opts.overwrite_file,
            )?);
            _manager_cleanup = None;
            manager_configs = None;
        }
        CoalescedWalBackend::Manager => {
            _file_cleanup = None;
            let cleanup = prepare_manager_dir(
                file_opts.path,
                file_opts.keep_file,
                file_opts.overwrite_file,
            )?;
            let configs = if durable_lane_count == 1 {
                vec![WalSegmentManagerConfig::new(
                    cleanup.path.clone(),
                    "events",
                    opts.manager_records_per_segment,
                    block_size,
                )]
            } else {
                (0..durable_lane_count)
                    .map(|lane_id| {
                        WalSegmentManagerConfig::new(
                            manager_lane_dir(&cleanup.path, lane_id),
                            "events",
                            opts.manager_records_per_segment,
                            block_size,
                        )
                    })
                    .collect()
            };
            manager_configs = Some(configs);
            _manager_cleanup = Some(cleanup);
        }
    }
    let total_events: usize = events.try_into()?;
    let min_records_per_block = opts.min_records_per_block.max(1).min(block_size);
    let effective_min_records_per_block = min_records_per_block.min(clients).max(1);
    let append_ring_capacity = opts
        .append_ring_capacity
        .max(clients.next_power_of_two())
        .max(2)
        .next_power_of_two();
    let append_ring_mask = append_ring_capacity as u64 - 1;
    let minimum_block_budget = total_events
        .div_ceil(effective_min_records_per_block)
        .max(1);
    let sparse_closed_loop_budget =
        opts.append_group_us == 0 || clients <= effective_min_records_per_block.saturating_mul(2);
    let max_blocks = if sparse_closed_loop_budget {
        total_events.max(1)
    } else {
        minimum_block_budget
            .saturating_add(append_ring_capacity)
            .min(total_events)
            .max(1)
    };
    let partial_block_slack = max_blocks.saturating_sub(minimum_block_budget);
    let mapped_records = max_blocks
        .checked_mul(block_size)
        .ok_or("coalesced mapped WAL segment size overflow")?;
    let requested_samples = opts.latency_samples.min(total_events).max(1);
    let sample_stride = (total_events as u64)
        .div_ceil(requested_samples as u64)
        .max(1);
    let base = events / clients as u64;
    let rem = events % clients as u64;
    let active_clients = clients.min(total_events);
    let worker_count = if ack_mode.needs_store() {
        workers.max(1)
    } else {
        0
    };
    let durable_worker_count = if durable_enabled {
        durable_lane_count
    } else {
        0
    };
    let prewrite_worker_count = usize::from(
        durable_enabled && opts.durable_sync_mode == WalDataSyncMode::PrewriteAndFileData,
    );
    let client_driver_threads = opts.client_driver_threads.max(1).min(clients);
    let multiplexed_clients = client_driver_threads < clients;

    let setup_elapsed;
    let elapsed;
    let visible_elapsed;
    let total_elapsed;
    let durable_catchup_elapsed;
    let mut recovery_elapsed = Duration::ZERO;
    let store_validation_elapsed;
    let total;
    let report;
    let actual_blocks;
    let visible_cut_snapshot;
    let final_cut_snapshot;
    let durable_sync_snapshot;
    let durable_sync_timing_report;
    let prewrite_snapshot;
    let stage_counters = opts
        .stage_timings
        .then(|| Arc::new(CoalescedStageCounters::default()));
    let block_timeline = opts
        .stage_timings
        .then(|| Arc::new(BlockStageTimeline::with_capacity(max_blocks)));
    {
        let setup_started = Instant::now();
        let segment = if wal_backend == CoalescedWalBackend::Segment {
            Some(Arc::new(unsafe {
                MappedWalSegment::create(file_opts.path, 1, mapped_records, block_size)?
            }))
        } else {
            None
        };
        let managers = if wal_backend == CoalescedWalBackend::Manager {
            Some(
                manager_configs
                    .as_ref()
                    .expect("manager configs for manager-backed coalesced WAL")
                    .iter()
                    .cloned()
                    .map(|config| unsafe { WalSegmentManager::create(config) })
                    .collect::<std::io::Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let manager_blocks: Option<Arc<[ManagerPublishedBlockSlot]>> =
            if wal_backend == CoalescedWalBackend::Manager {
                Some(
                    (0..max_blocks)
                        .map(|sequence| ManagerPublishedBlockSlot::new(sequence as u64))
                        .collect::<Vec<_>>()
                        .into(),
                )
            } else {
                None
            };
        let queue: Arc<[ClientAppendRingSlot]> = (0..append_ring_capacity)
            .map(|sequence| ClientAppendRingSlot::new(sequence as u64))
            .collect::<Vec<_>>()
            .into();
        let completion = ack_mode
            .needs_store()
            .then(|| Arc::new(ContiguousCompletionBarrier::with_capacity(max_blocks)));
        let store = ack_mode
            .needs_store()
            .then(|| Arc::new(OpenShardAppendStore::with_capacity(max_blocks, block_size)));
        setup_elapsed = setup_started.elapsed();

        let barrier = Arc::new(Barrier::new(
            client_driver_threads + worker_count + durable_worker_count + prewrite_worker_count + 2,
        ));
        let go = Arc::new(AtomicBool::new(false));
        let run_failed = Arc::new(AtomicBool::new(false));
        let enqueue_tail = Arc::new(AtomicU64::new(0));
        let published_blocks = Arc::new(AtomicU64::new(0));
        let appender_done = Arc::new(AtomicBool::new(false));
        let next_apply_block = Arc::new(AtomicU64::new(0));
        let requested_durable_prefix = Arc::new(AtomicU64::new(0));
        let durable_prefix = Arc::new(AtomicU64::new(0));
        let durable_syncs =
            durable_enabled.then(|| Arc::new(DurableSyncCounters::new(opts.stage_timings)));
        let prewrite_counters =
            (prewrite_worker_count != 0).then(|| Arc::new(WalPrewriteCounters::default()));

        let appender_handle = {
            let segment = segment.as_ref().map(Arc::clone);
            let manager_blocks = manager_blocks.as_ref().map(Arc::clone);
            let mut managers = managers;
            let stage_counters = stage_counters.as_ref().map(Arc::clone);
            let block_timeline = block_timeline.as_ref().map(Arc::clone);
            let queue = Arc::clone(&queue);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let run_failed = Arc::clone(&run_failed);
            let published_blocks = Arc::clone(&published_blocks);
            let appender_done = Arc::clone(&appender_done);
            let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
            let enqueue_tail = Arc::clone(&enqueue_tail);
            let append_group_window = Duration::from_micros(opts.append_group_us);
            std::thread::spawn(move || {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                let mut next_sequence = 0_u64;
                let mut next_block_id = 0_u64;
                let mut active_clients_remaining = active_clients;
                let mut batch = Vec::with_capacity(block_size);
                let mut positions = Vec::with_capacity(block_size);
                let mut total = DrainStats::default();
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                while next_sequence < total_events as u64 {
                    if run_failed.load(Ordering::Acquire) {
                        return Err("WAL appender aborted after peer failure".to_string());
                    }
                    batch.clear();
                    positions.clear();
                    let mut active_clients_after_batch = active_clients_remaining;
                    let mut first_ready_at = None::<Instant>;
                    while batch.len() < block_size && next_sequence < total_events as u64 {
                        if run_failed.load(Ordering::Acquire) {
                            return Err("WAL appender aborted after peer failure".to_string());
                        }
                        let slot = &queue[(next_sequence & append_ring_mask) as usize];
                        if let Some(entry) = slot.try_read_published(next_sequence) {
                            first_ready_at.get_or_insert_with(Instant::now);
                            total.observe(entry.intent);
                            batch.push(entry.intent);
                            positions.push(next_sequence);
                            if entry.final_for_client {
                                active_clients_after_batch =
                                    active_clients_after_batch.saturating_sub(1);
                            }
                            next_sequence += 1;
                            continue;
                        }
                        if batch.is_empty() {
                            std::thread::yield_now();
                            continue;
                        }
                        let no_claimed_gap = enqueue_tail.load(Ordering::Acquire) <= next_sequence;
                        let final_tail = next_sequence >= total_events as u64;
                        let aged = append_group_window.is_zero()
                            || first_ready_at
                                .is_some_and(|started| started.elapsed() >= append_group_window);
                        let target_records =
                            effective_min_records_per_block.min(active_clients_after_batch.max(1));
                        let has_minimum_block = batch.len() >= target_records;
                        if final_tail
                            || has_minimum_block
                            || (sparse_closed_loop_budget && aged && no_claimed_gap)
                        {
                            break;
                        }
                        std::thread::yield_now();
                    }
                    if batch.is_empty() {
                        continue;
                    }
                    if let Some(first_ready_at) = first_ready_at {
                        if let Some(stage_counters) = &stage_counters {
                            stage_counters
                                .append_batch_wait
                                .record(first_ready_at.elapsed());
                        }
                    }
                    let publish_started = stage_counters.as_ref().map(|_| Instant::now());
                    let publish_started_ns =
                        block_timeline.as_ref().map(|timeline| timeline.now_ns());
                    let block_prefix = match wal_backend {
                        CoalescedWalBackend::Segment => {
                            let block_id = segment
                                .as_ref()
                                .expect("segment-backed coalesced WAL")
                                .try_publish_intents_position(&batch)
                                .map_err(|error| error.to_string())?
                                .ok_or("coalesced WAL appender published an empty batch")?;
                            block_id + 1
                        }
                        CoalescedWalBackend::Manager => {
                            let lane_id = (next_block_id as usize) % durable_lane_count;
                            let block = managers
                                .as_mut()
                                .expect("manager-backed coalesced WAL")
                                .get_mut(lane_id)
                                .expect("durable WAL lane exists")
                                .publish_intents_handle(&batch)
                                .map_err(|error| error.to_string())?;
                            let global_block_id = next_block_id;
                            if global_block_id >= max_blocks as u64 {
                                return Err(format!(
                                    "manager coalesced WAL block {global_block_id} exceeds benchmark block budget {max_blocks}"
                                ));
                            }
                            if let Some(timeline) = &block_timeline {
                                timeline.record_logged_ns(global_block_id, timeline.now_ns());
                            }
                            manager_blocks.as_ref().expect("manager block directory")
                                [global_block_id as usize]
                                .publish(global_block_id, block);
                            global_block_id + 1
                        }
                    };
                    if let (Some(timeline), Some(logged_ns)) =
                        (block_timeline.as_ref(), publish_started_ns)
                    {
                        if wal_backend == CoalescedWalBackend::Segment {
                            timeline.record_logged_ns(block_prefix - 1, logged_ns);
                        }
                    }
                    if let (Some(stage_counters), Some(publish_started)) =
                        (&stage_counters, publish_started)
                    {
                        stage_counters
                            .append_wal_publish
                            .record(publish_started.elapsed());
                    }
                    next_block_id = block_prefix;
                    let notify_started = stage_counters.as_ref().map(|_| Instant::now());
                    for sequence in positions.iter().copied() {
                        queue[(sequence & append_ring_mask) as usize]
                            .publish_logged_block(block_prefix);
                    }
                    if let (Some(stage_counters), Some(notify_started)) =
                        (&stage_counters, notify_started)
                    {
                        stage_counters
                            .append_client_notify
                            .record(notify_started.elapsed());
                    }
                    active_clients_remaining = active_clients_after_batch;
                    published_blocks.store(block_prefix, Ordering::Release);
                    if durable_enabled {
                        requested_durable_prefix.fetch_max(block_prefix, Ordering::Release);
                    }
                }
                appender_done.store(true, Ordering::Release);
                failure_guard.disarm();
                Ok::<DrainStats, String>(total)
            })
        };

        let prewrite_handle = if prewrite_worker_count != 0 {
            let segment = segment.as_ref().map(Arc::clone);
            let manager_blocks = manager_blocks.as_ref().map(Arc::clone);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let run_failed = Arc::clone(&run_failed);
            let published_blocks = Arc::clone(&published_blocks);
            let appender_done = Arc::clone(&appender_done);
            let prewrite_counters =
                Arc::clone(prewrite_counters.as_ref().expect("prewrite counters"));
            Some(std::thread::spawn(move || -> std::io::Result<()> {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                let mut current = 0_u64;
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                loop {
                    if run_failed.load(Ordering::Acquire) {
                        return Err(std::io::Error::other(
                            "WAL prewrite worker aborted after peer failure",
                        ));
                    }
                    let final_blocks = published_blocks.load(Ordering::Acquire);
                    if appender_done.load(Ordering::Acquire) && current >= final_blocks {
                        break;
                    }
                    if final_blocks <= current {
                        std::thread::yield_now();
                        continue;
                    }
                    let published = match wal_backend {
                        CoalescedWalBackend::Segment => segment
                            .as_ref()
                            .expect("segment-backed prewrite coalesced WAL")
                            .contiguous_published_prefix(current, final_blocks)?,
                        CoalescedWalBackend::Manager => final_blocks,
                    };
                    if published <= current {
                        std::thread::yield_now();
                        continue;
                    }

                    let write_started = Instant::now();
                    let mut written_blocks = 0_u64;
                    match wal_backend {
                        CoalescedWalBackend::Segment => {
                            written_blocks = segment
                                .as_ref()
                                .expect("segment-backed prewrite coalesced WAL")
                                .write_published_data_frontier(published)?;
                        }
                        CoalescedWalBackend::Manager => {
                            let manager_blocks =
                                manager_blocks.as_ref().expect("manager block directory");
                            let mut pending_segment = None::<u64>;
                            let mut pending_write = None::<WalPublishedBlock>;
                            for block_id in current..published {
                                let block = manager_blocks[block_id as usize]
                                    .wait_published(block_id, &run_failed)
                                    .map_err(std::io::Error::other)?;
                                let position = block.position();
                                if pending_segment
                                    .is_some_and(|segment_id| segment_id != position.segment_id)
                                {
                                    if let Some(to_write) = pending_write.take() {
                                        written_blocks +=
                                            to_write.write_published_data_frontier()?;
                                    }
                                }
                                pending_segment = Some(position.segment_id);
                                pending_write = Some(block);
                            }
                            if let Some(to_write) = pending_write {
                                written_blocks += to_write.write_published_data_frontier()?;
                            }
                        }
                    }
                    prewrite_counters.record_write(written_blocks, write_started.elapsed());
                    current = published;
                }
                failure_guard.disarm();
                Ok(())
            }))
        } else {
            None
        };

        let durable_handles = if durable_enabled {
            if durable_lane_count == 1 {
                let segment = segment.as_ref().map(Arc::clone);
                let manager_blocks = manager_blocks.as_ref().map(Arc::clone);
                let barrier = Arc::clone(&barrier);
                let go = Arc::clone(&go);
                let run_failed = Arc::clone(&run_failed);
                let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
                let durable_prefix = Arc::clone(&durable_prefix);
                let published_blocks = Arc::clone(&published_blocks);
                let appender_done = Arc::clone(&appender_done);
                let durable_syncs =
                    Arc::clone(durable_syncs.as_ref().expect("durable sync counters"));
                let block_timeline = block_timeline.as_ref().map(Arc::clone);
                let durable_group_window = Duration::from_micros(opts.durable_group_us);
                let configured_durable_min_blocks = opts.durable_min_blocks as u64;
                let auto_durable_min_blocks = configured_durable_min_blocks == 0;
                let durable_sync_mode = opts.durable_sync_mode;
                vec![std::thread::spawn(move || -> std::io::Result<()> {
                    let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                    let mut durable_min_blocks = if auto_durable_min_blocks {
                        1
                    } else {
                        configured_durable_min_blocks.max(1)
                    };
                    barrier.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    loop {
                        if run_failed.load(Ordering::Acquire) {
                            return Err(std::io::Error::other(
                                "durable WAL worker aborted after peer failure",
                            ));
                        }
                        let current = durable_prefix.load(Ordering::Acquire);
                        let final_blocks = published_blocks.load(Ordering::Acquire);
                        if appender_done.load(Ordering::Acquire) && current >= final_blocks {
                            break;
                        }
                        let mut target = requested_durable_prefix.load(Ordering::Acquire);
                        if target > current {
                            let wait_started = Instant::now();
                            let mut reason = DurableFlushReason::Pressure;
                            if target.saturating_sub(current) < durable_min_blocks {
                                let deadline = Instant::now() + durable_group_window;
                                reason = DurableFlushReason::Deadline;
                                loop {
                                    let observed = requested_durable_prefix.load(Ordering::Acquire);
                                    if observed > target {
                                        target = observed;
                                    }
                                    let final_blocks = published_blocks.load(Ordering::Acquire);
                                    if appender_done.load(Ordering::Acquire)
                                        && observed >= final_blocks
                                    {
                                        reason = DurableFlushReason::Final;
                                        break;
                                    }
                                    if target.saturating_sub(current) >= durable_min_blocks {
                                        reason = DurableFlushReason::Pressure;
                                        break;
                                    }
                                    if durable_group_window.is_zero() || Instant::now() >= deadline
                                    {
                                        break;
                                    }
                                    std::thread::yield_now();
                                }
                            }
                            let published = match wal_backend {
                                CoalescedWalBackend::Segment => segment
                                    .as_ref()
                                    .expect("segment-backed durable coalesced WAL")
                                    .contiguous_published_prefix(current, target)?,
                                CoalescedWalBackend::Manager => target,
                            };
                            if published > current {
                                let wait_elapsed = wait_started.elapsed();
                                let sync_started = Instant::now();
                                let mut sync_frontier_calls = 0_u64;
                                match wal_backend {
                                    CoalescedWalBackend::Segment => {
                                        if let Err(error) = segment
                                            .as_ref()
                                            .expect("segment-backed durable coalesced WAL")
                                            .sync_published_data_frontier_with_mode(
                                                published,
                                                durable_sync_mode,
                                            )
                                        {
                                            run_failed.store(true, Ordering::Release);
                                            return Err(error);
                                        }
                                        sync_frontier_calls = 1;
                                    }
                                    CoalescedWalBackend::Manager => {
                                        let manager_blocks = manager_blocks
                                            .as_ref()
                                            .expect("manager block directory");
                                        let mut pending_segment = None::<u64>;
                                        let mut pending_sync = None::<WalPublishedBlock>;
                                        for block_id in current..published {
                                            let block = manager_blocks[block_id as usize]
                                                .wait_published(block_id, &run_failed)
                                                .map_err(std::io::Error::other)?;
                                            let position = block.position();
                                            if pending_segment.is_some_and(|segment_id| {
                                                segment_id != position.segment_id
                                            }) {
                                                if let Some(to_sync) = pending_sync.take() {
                                                    sync_frontier_calls += 1;
                                                    if let Err(error) = to_sync
                                                        .sync_published_data_frontier_with_mode(
                                                            durable_sync_mode,
                                                        )
                                                    {
                                                        run_failed.store(true, Ordering::Release);
                                                        return Err(error);
                                                    }
                                                }
                                            }
                                            pending_segment = Some(position.segment_id);
                                            pending_sync = Some(block);
                                        }
                                        if let Some(to_sync) = pending_sync {
                                            sync_frontier_calls += 1;
                                            if let Err(error) = to_sync
                                                .sync_published_data_frontier_with_mode(
                                                    durable_sync_mode,
                                                )
                                            {
                                                run_failed.store(true, Ordering::Release);
                                                return Err(error);
                                            }
                                        }
                                    }
                                }
                                let sync_elapsed = sync_started.elapsed();
                                durable_syncs.record_wait(wait_elapsed);
                                durable_syncs.record_sync(
                                    published - current,
                                    sync_frontier_calls,
                                    sync_elapsed,
                                    reason,
                                );
                                if auto_durable_min_blocks {
                                    let sync_ns = duration_ns(sync_elapsed);
                                    durable_min_blocks = if sync_ns >= 100_000 {
                                        4
                                    } else if sync_ns >= 25_000 {
                                        2
                                    } else {
                                        1
                                    };
                                }
                                for block_id in current..published {
                                    if let Some(block_timeline) = &block_timeline {
                                        block_timeline.record_durable(block_id);
                                    }
                                }
                                durable_prefix.store(published, Ordering::Release);
                            } else {
                                std::thread::yield_now();
                            }
                        } else {
                            std::thread::yield_now();
                        }
                    }
                    failure_guard.disarm();
                    Ok(())
                })]
            } else {
                let durable_completion =
                    Arc::new(ContiguousCompletionBarrier::with_capacity(max_blocks));
                (0..durable_lane_count)
                    .map(|lane_id| {
                        let manager_blocks =
                            Arc::clone(manager_blocks.as_ref().expect("manager block directory"));
                        let durable_completion = Arc::clone(&durable_completion);
                        let barrier = Arc::clone(&barrier);
                        let go = Arc::clone(&go);
                        let run_failed = Arc::clone(&run_failed);
                        let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
                        let durable_prefix = Arc::clone(&durable_prefix);
                        let published_blocks = Arc::clone(&published_blocks);
                        let appender_done = Arc::clone(&appender_done);
                        let durable_syncs =
                            Arc::clone(durable_syncs.as_ref().expect("durable sync counters"));
                        let block_timeline = block_timeline.as_ref().map(Arc::clone);
                        let durable_group_window = Duration::from_micros(opts.durable_group_us);
                        let configured_durable_min_blocks = opts.durable_min_blocks as u64;
                        let auto_durable_min_blocks = configured_durable_min_blocks == 0;
                        let durable_sync_mode = opts.durable_sync_mode;
                        std::thread::spawn(move || -> std::io::Result<()> {
                            let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                            let mut next_global_block = lane_id as u64;
                            let lane_stride = durable_lane_count as u64;
                            let mut lane_batch_blocks = if auto_durable_min_blocks {
                                1
                            } else {
                                configured_durable_min_blocks.max(1)
                            };
                            let mut durable_batch = Vec::new();
                            let mut sync_batch = Vec::new();
                            barrier.wait();
                            while !go.load(Ordering::Acquire) {
                                std::hint::spin_loop();
                            }
                            loop {
                                if run_failed.load(Ordering::Acquire) {
                                    return Err(std::io::Error::other(
                                        "striped durable WAL worker aborted after peer failure",
                                    ));
                                }
                                let final_blocks = published_blocks.load(Ordering::Acquire);
                                if appender_done.load(Ordering::Acquire)
                                    && next_global_block >= final_blocks
                                {
                                    break;
                                }
                                let mut target = requested_durable_prefix.load(Ordering::Acquire);
                                if target <= next_global_block {
                                    std::thread::yield_now();
                                    continue;
                                }

                                let wait_started = Instant::now();
                                let mut reason = DurableFlushReason::Pressure;
                                if lane_blocks_between(next_global_block, target, lane_stride)
                                    < lane_batch_blocks
                                {
                                    let deadline = Instant::now() + durable_group_window;
                                    reason = DurableFlushReason::Deadline;
                                    loop {
                                        let observed =
                                            requested_durable_prefix.load(Ordering::Acquire);
                                        if observed > target {
                                            target = observed;
                                        }
                                        let final_blocks = published_blocks.load(Ordering::Acquire);
                                        if appender_done.load(Ordering::Acquire)
                                            && observed >= final_blocks
                                        {
                                            reason = DurableFlushReason::Final;
                                            break;
                                        }
                                        if lane_blocks_between(
                                            next_global_block,
                                            target,
                                            lane_stride,
                                        ) >= lane_batch_blocks
                                        {
                                            reason = DurableFlushReason::Pressure;
                                            break;
                                        }
                                        if durable_group_window.is_zero()
                                            || Instant::now() >= deadline
                                        {
                                            break;
                                        }
                                        std::thread::yield_now();
                                    }
                                }
                                durable_batch.clear();
                                sync_batch.clear();
                                let mut block_id = next_global_block;
                                let mut pending_segment = None::<u64>;
                                let mut pending_sync = None::<WalPublishedBlock>;
                                while block_id < target
                                    && durable_batch.len() < lane_batch_blocks as usize
                                {
                                    let block = manager_blocks[block_id as usize]
                                        .wait_published(block_id, &run_failed)
                                        .map_err(std::io::Error::other)?;
                                    let position = block.position();
                                    if pending_segment
                                        .is_some_and(|segment_id| segment_id != position.segment_id)
                                    {
                                        if let Some(to_sync) = pending_sync.take() {
                                            sync_batch.push(to_sync);
                                        }
                                    }
                                    pending_segment = Some(position.segment_id);
                                    pending_sync = Some(block);
                                    durable_batch.push(block_id);
                                    block_id += lane_stride;
                                }
                                if let Some(to_sync) = pending_sync {
                                    sync_batch.push(to_sync);
                                }
                                if sync_batch.is_empty() {
                                    std::thread::yield_now();
                                    continue;
                                }
                                let wait_elapsed = wait_started.elapsed();
                                let sync_started = Instant::now();
                                for sync_block in &sync_batch {
                                    if let Err(error) = sync_block
                                        .sync_published_data_frontier_with_mode(durable_sync_mode)
                                    {
                                        run_failed.store(true, Ordering::Release);
                                        return Err(error);
                                    }
                                }
                                let sync_elapsed = sync_started.elapsed();
                                durable_syncs.record_wait(wait_elapsed);
                                durable_syncs.record_sync(
                                    durable_batch.len() as u64,
                                    sync_batch.len() as u64,
                                    sync_elapsed,
                                    reason,
                                );
                                if auto_durable_min_blocks {
                                    let sync_ns = duration_ns(sync_elapsed);
                                    lane_batch_blocks = if sync_ns >= 100_000 {
                                        4
                                    } else if sync_ns >= 25_000 {
                                        2
                                    } else {
                                        1
                                    };
                                }
                                for block_id in durable_batch.iter().copied() {
                                    if let Some(block_timeline) = &block_timeline {
                                        block_timeline.record_durable(block_id);
                                    }
                                    let prefix =
                                        durable_completion.complete(block_id).map_err(|error| {
                                            std::io::Error::other(error.to_string())
                                        })?;
                                    durable_prefix.fetch_max(prefix, Ordering::AcqRel);
                                }
                                next_global_block = block_id;
                            }
                            failure_guard.disarm();
                            Ok(())
                        })
                    })
                    .collect()
            }
        } else {
            Vec::new()
        };

        let mut worker_handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let segment = segment.as_ref().map(Arc::clone);
            let manager_blocks = manager_blocks.as_ref().map(Arc::clone);
            let completion = Arc::clone(completion.as_ref().expect("coalesced completion barrier"));
            let store = Arc::clone(store.as_ref().expect("coalesced client store"));
            let stage_counters = stage_counters.as_ref().map(Arc::clone);
            let block_timeline = block_timeline.as_ref().map(Arc::clone);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let run_failed = Arc::clone(&run_failed);
            let next_apply_block = Arc::clone(&next_apply_block);
            let published_blocks = Arc::clone(&published_blocks);
            let appender_done = Arc::clone(&appender_done);
            worker_handles.push(std::thread::spawn(move || {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                let mut total = DrainStats::default();
                let mut payload = Vec::with_capacity(block_size);
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                loop {
                    if run_failed.load(Ordering::Acquire) {
                        return Err("coalesced apply worker aborted after peer failure".to_string());
                    }
                    let block_id = next_apply_block.fetch_add(1, Ordering::Relaxed);
                    if block_id >= max_blocks as u64 {
                        break;
                    }
                    let wait_started = stage_counters.as_ref().map(|_| Instant::now());
                    loop {
                        if run_failed.load(Ordering::Acquire) {
                            return Err(
                                "coalesced apply worker aborted after peer failure".to_string()
                            );
                        }
                        if appender_done.load(Ordering::Acquire)
                            && block_id >= published_blocks.load(Ordering::Acquire)
                        {
                            failure_guard.disarm();
                            return Ok::<DrainStats, String>(total);
                        }
                        let read_started = stage_counters.as_ref().map(|_| Instant::now());
                        let read = match wal_backend {
                            CoalescedWalBackend::Segment => segment
                                .as_ref()
                                .expect("segment-backed coalesced WAL")
                                .read_published_block_into(block_id, &mut payload),
                            CoalescedWalBackend::Manager => {
                                if let Some(block) =
                                    manager_blocks.as_ref().expect("manager block directory")
                                        [block_id as usize]
                                        .try_published(block_id)
                                {
                                    block.read_published_block_into(&mut payload)
                                } else {
                                    None
                                }
                            }
                        };
                        if read.is_some() {
                            if let (Some(stage_counters), Some(wait_started)) =
                                (&stage_counters, wait_started)
                            {
                                stage_counters
                                    .store_wait_for_block
                                    .record(wait_started.elapsed());
                            }
                            if let (Some(stage_counters), Some(read_started)) =
                                (&stage_counters, read_started)
                            {
                                stage_counters
                                    .store_read_block
                                    .record(read_started.elapsed());
                            }
                            let apply_started = stage_counters.as_ref().map(|_| Instant::now());
                            let applied = store
                                .apply_block(block_id, &payload)
                                .map_err(|error| error.to_string())?;
                            if let (Some(stage_counters), Some(apply_started)) =
                                (&stage_counters, apply_started)
                            {
                                stage_counters
                                    .store_apply_block
                                    .record(apply_started.elapsed());
                            }
                            total.add(applied);
                            let complete_started = stage_counters.as_ref().map(|_| Instant::now());
                            completion
                                .complete(block_id)
                                .map_err(|error| error.to_string())?;
                            if let (Some(stage_counters), Some(complete_started)) =
                                (&stage_counters, complete_started)
                            {
                                stage_counters
                                    .store_complete_block
                                    .record(complete_started.elapsed());
                            }
                            if let Some(block_timeline) = &block_timeline {
                                block_timeline.record_store_applied(block_id);
                            }
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
                failure_guard.disarm();
                Ok::<DrainStats, String>(total)
            }));
        }

        let mut client_handles = Vec::with_capacity(client_driver_threads);
        if multiplexed_clients {
            #[derive(Clone, Copy)]
            struct AsyncOutstanding {
                append_sequence: u64,
                intent: WriteIntent,
                started: Option<Instant>,
                logged_at: Option<Instant>,
                block_prefix: u64,
                store_applied_at: Option<Instant>,
            }

            struct AsyncLogicalClient {
                first: u64,
                count: u64,
                next_offset: u64,
                ready_started: Option<Instant>,
                outstanding: Option<AsyncOutstanding>,
            }

            for driver_id in 0..client_driver_threads {
                let queue = Arc::clone(&queue);
                let completion = completion.as_ref().map(Arc::clone);
                let durable_prefix = Arc::clone(&durable_prefix);
                let enqueue_tail = Arc::clone(&enqueue_tail);
                let barrier = Arc::clone(&barrier);
                let go = Arc::clone(&go);
                let run_failed = Arc::clone(&run_failed);
                let driver_clients = clients / client_driver_threads
                    + usize::from(driver_id < clients % client_driver_threads);
                let first_client = (clients / client_driver_threads) * driver_id
                    + (clients % client_driver_threads).min(driver_id);
                let issue_budget = opts.client_driver_issue_budget;
                client_handles.push(std::thread::spawn(move || {
                    let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                    let mut logical_clients = Vec::with_capacity(driver_clients);
                    let mut total_assigned = 0_u64;
                    for client_id in first_client..first_client + driver_clients {
                        let count = base + u64::from((client_id as u64) < rem);
                        let first = base * client_id as u64 + rem.min(client_id as u64);
                        total_assigned += count;
                        logical_clients.push(AsyncLogicalClient {
                            first,
                            count,
                            next_offset: 0,
                            ready_started: None,
                            outstanding: None,
                        });
                    }
                    let sample_capacity = total_assigned
                        .div_ceil(sample_stride)
                        .try_into()
                        .unwrap_or(usize::MAX)
                        .saturating_add(driver_clients);
                    let mut latencies = ClientLatencySamples::with_capacity(sample_capacity);
                    let mut total = DrainStats::default();
                    let mut completed = 0_u64;
                    let mut next_issue_index = 0_usize;
                    barrier.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    let ready_at = Instant::now();
                    for client in logical_clients.iter_mut() {
                        if client.next_offset < client.count {
                            let client_seq = client.first + client.next_offset;
                            client.ready_started =
                                client_seq.is_multiple_of(sample_stride).then_some(ready_at);
                        }
                    }
                    while completed < total_assigned {
                        if run_failed.load(Ordering::Acquire) {
                            return Err("coalesced async client driver aborted after peer failure"
                                .to_string());
                        }
                        let mut progressed = false;

                        let mut issued = 0_usize;
                        for _ in 0..logical_clients.len() {
                            if issued >= issue_budget {
                                break;
                            }
                            let client_index = next_issue_index;
                            next_issue_index = (next_issue_index + 1) % logical_clients.len();
                            let client = &mut logical_clients[client_index];
                            if client.outstanding.is_some() || client.next_offset >= client.count {
                                continue;
                            }
                            let client_seq = client.first + client.next_offset;
                            let intent = intent_for(client_seq);
                            let started = client.ready_started.take();
                            let append_sequence = enqueue_tail.fetch_add(1, Ordering::Relaxed);
                            if append_sequence >= total_events as u64 {
                                return Err("coalesced async client enqueue exceeded event count"
                                    .to_string());
                            }
                            let slot = &queue[(append_sequence & append_ring_mask) as usize];
                            slot.wait_free_and_publish(
                                append_sequence,
                                intent,
                                client.next_offset + 1 == client.count,
                                &run_failed,
                            )?;
                            client.next_offset += 1;
                            client.outstanding = Some(AsyncOutstanding {
                                append_sequence,
                                intent,
                                started,
                                logged_at: None,
                                block_prefix: 0,
                                store_applied_at: None,
                            });
                            issued += 1;
                            progressed = true;
                        }

                        for client in logical_clients.iter_mut() {
                            let Some(outstanding) = client.outstanding.as_mut() else {
                                continue;
                            };
                            let slot =
                                &queue[(outstanding.append_sequence & append_ring_mask) as usize];
                            if outstanding.block_prefix == 0 {
                                let Some(block_prefix) = slot.try_logged_block() else {
                                    continue;
                                };
                                outstanding.block_prefix = block_prefix;
                                outstanding.logged_at = outstanding.started.map(|_| Instant::now());
                                total.observe(outstanding.intent);
                                progressed = true;
                            }
                            if let Some(completion) = completion.as_ref() {
                                if outstanding.store_applied_at.is_none() {
                                    if completion.completed_prefix() < outstanding.block_prefix {
                                        continue;
                                    }
                                    outstanding.store_applied_at =
                                        outstanding.started.map(|_| Instant::now());
                                    progressed = true;
                                }
                            }
                            if ack_mode.needs_durable()
                                && durable_prefix.load(Ordering::Acquire) < outstanding.block_prefix
                            {
                                continue;
                            }

                            let acked_at = Instant::now();
                            if let (Some(started), Some(logged_at)) =
                                (outstanding.started, outstanding.logged_at)
                            {
                                latencies
                                    .client_to_logged
                                    .push(duration_ns(logged_at.duration_since(started)));
                                if completion.is_some() {
                                    latencies
                                        .logged_to_ack
                                        .push(duration_ns(acked_at.duration_since(logged_at)));
                                }
                                if let Some(store_applied_at) = outstanding.store_applied_at {
                                    if ack_mode.needs_durable() {
                                        latencies.store_to_ack.push(duration_ns(
                                            acked_at.duration_since(store_applied_at),
                                        ));
                                    }
                                }
                                latencies
                                    .client_to_ack
                                    .push(duration_ns(acked_at.duration_since(started)));
                            }
                            slot.release(outstanding.append_sequence, append_ring_capacity as u64);
                            client.outstanding = None;
                            completed += 1;
                            if client.next_offset < client.count {
                                let client_seq = client.first + client.next_offset;
                                client.ready_started =
                                    client_seq.is_multiple_of(sample_stride).then_some(acked_at);
                            }
                            progressed = true;
                        }

                        if !progressed {
                            std::thread::yield_now();
                        }
                    }
                    failure_guard.disarm();
                    Ok::<(DrainStats, ClientLatencySamples), String>((total, latencies))
                }));
            }
        } else {
            for client_id in 0..clients {
                let queue = Arc::clone(&queue);
                let completion = completion.as_ref().map(Arc::clone);
                let durable_prefix = Arc::clone(&durable_prefix);
                let enqueue_tail = Arc::clone(&enqueue_tail);
                let barrier = Arc::clone(&barrier);
                let go = Arc::clone(&go);
                let run_failed = Arc::clone(&run_failed);
                let count = base + u64::from((client_id as u64) < rem);
                let first = base * client_id as u64 + rem.min(client_id as u64);
                client_handles.push(std::thread::spawn(move || {
                    let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                    let sample_capacity =
                        (count.div_ceil(sample_stride) as usize).saturating_add(1);
                    let mut latencies = ClientLatencySamples::with_capacity(sample_capacity);
                    let mut total = DrainStats::default();
                    barrier.wait();
                    while !go.load(Ordering::Acquire) {
                        std::hint::spin_loop();
                    }
                    for offset in 0..count {
                        if run_failed.load(Ordering::Acquire) {
                            return Err("coalesced client aborted after peer failure".to_string());
                        }
                        let client_seq = first + offset;
                        let sampled = client_seq.is_multiple_of(sample_stride);
                        let intent = intent_for(client_seq);
                        let started = sampled.then(Instant::now);
                        let append_sequence = enqueue_tail.fetch_add(1, Ordering::Relaxed);
                        if append_sequence >= total_events as u64 {
                            return Err("coalesced client enqueue exceeded event count".to_string());
                        }
                        let slot = &queue[(append_sequence & append_ring_mask) as usize];
                        slot.wait_free_and_publish(
                            append_sequence,
                            intent,
                            offset + 1 == count,
                            &run_failed,
                        )?;
                        let block_prefix = slot.wait_logged_block(&run_failed)?;
                        let logged_at = sampled.then(Instant::now);
                        total.observe(intent);
                        let mut store_applied_at = None;
                        if let Some(completion) = completion.as_ref() {
                            wait_for_completion_or_failure(
                                completion,
                                block_prefix,
                                opts.wait_spins,
                                &run_failed,
                            )?;
                            store_applied_at = sampled.then(Instant::now);
                        }
                        if ack_mode.needs_durable() {
                            while durable_prefix.load(Ordering::Acquire) < block_prefix {
                                if run_failed.load(Ordering::Acquire) {
                                    return Err(
                                        "coalesced durable WAL worker failed before acknowledging request"
                                            .to_string(),
                                    );
                                }
                                std::thread::yield_now();
                            }
                        }
                        slot.release(append_sequence, append_ring_capacity as u64);
                        if let (Some(started), Some(logged_at)) = (started, logged_at) {
                            let acked_at = Instant::now();
                            latencies
                                .client_to_logged
                                .push(duration_ns(logged_at.duration_since(started)));
                            if completion.is_some() {
                                latencies
                                    .logged_to_ack
                                    .push(duration_ns(acked_at.duration_since(logged_at)));
                            }
                            if let Some(store_applied_at) = store_applied_at {
                                if ack_mode.needs_durable() {
                                    latencies.store_to_ack.push(duration_ns(
                                        acked_at.duration_since(store_applied_at),
                                    ));
                                }
                            }
                            latencies
                                .client_to_ack
                                .push(duration_ns(acked_at.duration_since(started)));
                        }
                    }
                    failure_guard.disarm();
                    Ok::<(DrainStats, ClientLatencySamples), String>((total, latencies))
                }));
            }
        }

        barrier.wait();
        let started = Instant::now();
        if let Some(block_timeline) = &block_timeline {
            block_timeline.set_epoch(started);
        }
        go.store(true, Ordering::Release);
        let mut first_error = None::<String>;
        let mut client_total = DrainStats::default();
        let mut samples = ClientLatencySamples::default();
        for handle in client_handles {
            match handle.join() {
                Ok(Ok((thread_total, thread_samples))) => {
                    client_total.add(thread_total);
                    samples.append(thread_samples);
                }
                Ok(Err(message)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(message);
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some("coalesced client thread panicked".to_string());
                    }
                }
            }
        }

        let mut worker_total = DrainStats::default();
        for handle in worker_handles {
            match handle.join() {
                Ok(Ok(thread_total)) => worker_total.add(thread_total),
                Ok(Err(message)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(message);
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some("coalesced apply worker thread panicked".to_string());
                    }
                }
            }
        }

        let appender_total = match appender_handle.join() {
            Ok(Ok(thread_total)) => thread_total,
            Ok(Err(message)) => {
                run_failed.store(true, Ordering::Release);
                if first_error.is_none() {
                    first_error = Some(message);
                }
                DrainStats::default()
            }
            Err(_) => {
                run_failed.store(true, Ordering::Release);
                if first_error.is_none() {
                    first_error = Some("coalesced WAL appender thread panicked".to_string());
                }
                DrainStats::default()
            }
        };
        visible_elapsed = started.elapsed();
        visible_cut_snapshot = ChronicleCutSnapshot {
            published: published_blocks.load(Ordering::Acquire),
            store_applied: completion
                .as_ref()
                .map(|completion| completion.completed_prefix()),
            requested_durable: durable_enabled
                .then(|| requested_durable_prefix.load(Ordering::Acquire)),
            durable: durable_enabled.then(|| durable_prefix.load(Ordering::Acquire)),
        };

        if let Some(handle) = prewrite_handle {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(error.to_string());
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error =
                            Some("coalesced WAL prewrite worker thread panicked".to_string());
                    }
                }
            }
        }

        for handle in durable_handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error = Some(error.to_string());
                    }
                }
                Err(_) => {
                    run_failed.store(true, Ordering::Release);
                    if first_error.is_none() {
                        first_error =
                            Some("coalesced durable WAL worker thread panicked".to_string());
                    }
                }
            }
        }
        if let Some(message) = first_error {
            return Err(std::io::Error::other(message).into());
        }

        total_elapsed = started.elapsed();
        durable_catchup_elapsed = total_elapsed.saturating_sub(visible_elapsed);
        elapsed = total_elapsed;
        actual_blocks = published_blocks.load(Ordering::Acquire) as usize;
        final_cut_snapshot = ChronicleCutSnapshot {
            published: actual_blocks as u64,
            store_applied: completion
                .as_ref()
                .map(|completion| completion.completed_prefix()),
            requested_durable: durable_enabled
                .then(|| requested_durable_prefix.load(Ordering::Acquire)),
            durable: durable_enabled.then(|| durable_prefix.load(Ordering::Acquire)),
        };
        durable_sync_snapshot = durable_syncs
            .as_ref()
            .map(|counters| counters.snapshot())
            .unwrap_or_default();
        durable_sync_timing_report = durable_syncs
            .as_ref()
            .map(|counters| counters.timing_report());
        prewrite_snapshot = prewrite_counters
            .as_ref()
            .map(|counters| counters.snapshot())
            .unwrap_or_default();
        total = if ack_mode.needs_store() {
            worker_total
        } else {
            appender_total
        };
        if client_total != appender_total {
            return Err(format!(
                "coalesced client/appender stats mismatch: client={client_total:?}, appender={appender_total:?}"
            )
            .into());
        }
        report = samples
            .report(requested_samples, sample_stride)
            .ok_or("coalesced client latency sampling produced no samples")?;
        let validate_started = Instant::now();
        if let Some(store) = store.as_ref() {
            let observed = store
                .validate_applied_blocks(actual_blocks as u64)
                .ok_or("coalesced client store validation could not read applied prefix")?;
            let expected = stats_for_range(0, events);
            if observed != expected {
                return Err(format!(
                    "coalesced client store validation mismatch: got {observed:?}, expected {expected:?}"
                )
                .into());
            }
        }
        store_validation_elapsed = validate_started.elapsed();
    }

    assert_expected_stats(coalesced_client_label(ack_mode, wal_backend), total, events)?;
    if durable_enabled {
        let recovery_started = Instant::now();
        let expected = stats_for_range(0, events);
        match wal_backend {
            CoalescedWalBackend::Segment => {
                let recovered = recover_wal_segment_by_scan(file_opts.path)?;
                recovery_elapsed = recovery_started.elapsed();
                if recovered.recovered_blocks != actual_blocks as u64
                    || recovered.recovered_records != events
                    || recovered.stats != expected
                {
                    return Err(format!(
                        "coalesced client durable recovery mismatch: recovered {recovered:?}, expected_blocks={actual_blocks}, expected_stats={expected:?}"
                    )
                    .into());
                }
            }
            CoalescedWalBackend::Manager => {
                let configs = manager_configs
                    .as_ref()
                    .expect("manager configs for manager durable recovery");
                if durable_lane_count == 1 {
                    let recovered = recover_wal_manager_by_scan(&configs[0])?;
                    recovery_elapsed = recovery_started.elapsed();
                    let blocks_per_segment =
                        opts.manager_records_per_segment.div_ceil(block_size) as u64;
                    let recovered_blocks = OpenShardAppendStore::global_block_id(
                        recovered.durable_segment_id,
                        blocks_per_segment,
                        recovered.durable_blocks,
                    );
                    if recovered_blocks != actual_blocks as u64
                        || recovered.recovered_records != events
                        || recovered.stats != expected
                    {
                        return Err(format!(
                            "manager coalesced durable recovery mismatch: recovered {recovered:?}, recovered_blocks={recovered_blocks}, expected_blocks={actual_blocks}, expected_stats={expected:?}"
                        )
                        .into());
                    }
                } else {
                    let mut recovered_records = 0_u64;
                    let mut recovered_stats = DrainStats::default();
                    let mut recovered_lane_block_counts = Vec::with_capacity(configs.len());
                    for (lane_id, config) in configs.iter().enumerate() {
                        let recovered = recover_wal_manager_by_scan(config)?;
                        let blocks_per_segment =
                            opts.manager_records_per_segment.div_ceil(block_size) as u64;
                        let recovered_lane_block_count = OpenShardAppendStore::global_block_id(
                            recovered.durable_segment_id,
                            blocks_per_segment,
                            recovered.durable_blocks,
                        );
                        let expected_lane_blocks =
                            lane_block_count(lane_id, actual_blocks as u64, durable_lane_count);
                        if recovered_lane_block_count != expected_lane_blocks {
                            return Err(format!(
                                "striped manager durable recovery lane {lane_id} mismatch: recovered {recovered:?}, recovered_lane_blocks={recovered_lane_block_count}, expected_lane_blocks={expected_lane_blocks}"
                            )
                            .into());
                        }
                        recovered_lane_block_counts.push(recovered_lane_block_count);
                        recovered_records += recovered.recovered_records;
                        recovered_stats.add(recovered.stats);
                    }
                    let recovered_global_prefix =
                        striped_global_prefix_from_lane_counts(&recovered_lane_block_counts);
                    recovery_elapsed = recovery_started.elapsed();
                    if recovered_global_prefix != actual_blocks as u64
                        || recovered_records != events
                        || recovered_stats != expected
                    {
                        return Err(format!(
                            "striped manager durable recovery mismatch: recovered_global_prefix={recovered_global_prefix}, expected_blocks={actual_blocks}, recovered_records={recovered_records}, expected_records={events}, recovered_stats={recovered_stats:?}, expected_stats={expected:?}"
                        )
                        .into());
                    }
                }
            }
        }
    }

    let avg_block_records = events as f64 / actual_blocks.max(1) as f64;
    let store_validation = if ack_mode.needs_store() {
        format!(
            " store-validate={:.3}s",
            store_validation_elapsed.as_secs_f64()
        )
    } else {
        String::new()
    };
    let durable_min_blocks_label = if opts.durable_min_blocks == 0 {
        "auto".to_string()
    } else {
        opts.durable_min_blocks.to_string()
    };
    let prewrite_extra = if prewrite_worker_count != 0 {
        prewrite_snapshot.format()
    } else {
        String::new()
    };
    let block_budget_label = if sparse_closed_loop_budget {
        "sparse-safe"
    } else {
        "compact"
    };
    let backend_label = match wal_backend {
        CoalescedWalBackend::Segment => "segment",
        CoalescedWalBackend::Manager => "manager",
    };
    let manager_extra = if wal_backend == CoalescedWalBackend::Manager {
        format!(
            " manager-records/segment={}",
            opts.manager_records_per_segment
        )
    } else {
        String::new()
    };
    let durable_lane_extra = if wal_backend == CoalescedWalBackend::Manager && durable_enabled {
        format!(" durable-lanes={durable_lane_count}")
    } else {
        String::new()
    };
    let client_driver_label = if multiplexed_clients {
        format!(
            "async:{client_driver_threads}/issue={}",
            opts.client_driver_issue_budget
        )
    } else {
        "thread-per-client".to_string()
    };
    let cut_extra = if durable_enabled && !ack_mode.needs_durable() {
        format!(
            "{}{}",
            visible_cut_snapshot.format_with_label("chronicle-visible-cuts"),
            final_cut_snapshot.format_with_label("chronicle-final-cuts"),
        )
    } else {
        final_cut_snapshot.format_with_label("chronicle-cuts")
    };
    let background_durable_extra = if opts.background_durable && !ack_mode.needs_durable() {
        let visible_secs = visible_elapsed.as_secs_f64();
        let visible_mps = events as f64 / visible_secs / 1e6;
        let visible_ns_per_write = visible_secs * 1e9 / events as f64;
        format!(
            " background-durable=1 visible-elapsed={:.3}s visible-throughput={:.3}M/s visible-ns/write={:.2} total-with-durable={:.3}s durable-catchup={:.3}s",
            visible_secs,
            visible_mps,
            visible_ns_per_write,
            total_elapsed.as_secs_f64(),
            durable_catchup_elapsed.as_secs_f64(),
        )
    } else if opts.background_durable {
        " background-durable=redundant-with-durable-ack".to_string()
    } else {
        String::new()
    };
    let extra = format!(
        " setup={:.3}s clients={clients} client-drivers={client_driver_label} workers={worker_count} wal-backend={backend_label} wal-block={block_size}{manager_extra}{durable_lane_extra} max-blocks={max_blocks} block-budget={block_budget_label} min-budget={minimum_block_budget} partial-slack={partial_block_slack} blocks={actual_blocks} avg-block={avg_block_records:.1} append-ring={append_ring_capacity} append-group={}us min-block={effective_min_records_per_block} requested-min-block={min_records_per_block} wait-spins={wait_spins} ack={ack}{cuts}{background_durable_extra}{store_validation}{recovery}",
        setup_elapsed.as_secs_f64(),
        opts.append_group_us,
        cuts = cut_extra,
        wait_spins = opts.wait_spins,
        ack = match ack_mode {
            ClientAckMode::Logged => "coalesced-non-durable-logged",
            ClientAckMode::StoreApplied => "coalesced-non-durable-store-applied",
            ClientAckMode::DurableStoreApplied =>
                "coalesced-data-fenced-wal+volatile-store-applied",
        },
        recovery = if durable_enabled {
            format!(
                " durable-group={}us durable-min-blocks={} durable-sync-mode={}{}{} recover-final-durable={:.3}s",
                opts.durable_group_us,
                durable_min_blocks_label,
                durable_sync_mode_label(opts.durable_sync_mode),
                prewrite_extra,
                durable_sync_snapshot.format(),
                recovery_elapsed.as_secs_f64()
            )
        } else {
            String::new()
        }
    );
    report_extra(
        coalesced_client_label(ack_mode, wal_backend),
        events,
        elapsed,
        total,
        &extra,
    );
    report.print(ack_mode);
    if let Some(block_timeline) = &block_timeline {
        if let Some(report) = block_timeline.report(actual_blocks as u64) {
            report.print();
        }
    }
    if let Some(stage_counters) = &stage_counters {
        stage_counters.print();
    }
    if let Some(report) = &durable_sync_timing_report {
        report.print();
    }
    Ok(())
}
