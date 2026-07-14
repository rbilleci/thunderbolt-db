use super::super::{
    assert_expected_stats, durable_sync_mode_label, duration_ns, intent_for, prepare_journal_path,
    recover_wal_segment_by_scan, report_extra, stats_for_range, wait_for_completion_or_failure,
    Arc, AtomicBool, AtomicU64, Barrier, ClientAckMode, ClientLatencySamples, ClientRunOptions,
    ContiguousCompletionBarrier, DrainStats, Duration, Error, FailureGuard, Instant,
    MappedWalSegment, OpenShardAppendStore, Ordering,
};

pub(in super::super) fn run_file_wal_client_latency(
    events: u64,
    block_size: usize,
    clients: usize,
    workers: usize,
    opts: ClientRunOptions<'_>,
) -> Result<(), Box<dyn Error>> {
    let file_opts = opts.file;
    let ack_mode = opts.ack_mode;
    let _cleanup = prepare_journal_path(
        file_opts.path,
        file_opts.keep_file,
        file_opts.overwrite_file,
    )?;
    let total_blocks: usize = events.try_into()?;
    let mapped_records = total_blocks
        .checked_mul(block_size)
        .ok_or("mapped WAL segment size overflow")?;
    let requested_samples = opts.latency_samples.min(total_blocks).max(1);
    let sample_stride = (total_blocks as u64)
        .div_ceil(requested_samples as u64)
        .max(1);
    let base = events / clients as u64;
    let rem = events % clients as u64;
    let worker_count = if ack_mode.needs_store() {
        workers.max(1)
    } else {
        0
    };
    let durable_worker_count = usize::from(ack_mode.needs_durable());

    let setup_elapsed;
    let elapsed;
    let mut recovery_elapsed = Duration::ZERO;
    let store_validation_elapsed;
    let total;
    let report;
    {
        let setup_started = Instant::now();
        let segment = Arc::new(unsafe {
            MappedWalSegment::create(file_opts.path, 1, mapped_records, block_size)?
        });
        let completion = ack_mode
            .needs_store()
            .then(|| Arc::new(ContiguousCompletionBarrier::with_capacity(total_blocks)));
        let store = ack_mode.needs_store().then(|| {
            Arc::new(OpenShardAppendStore::with_capacity(
                total_blocks,
                block_size,
            ))
        });
        setup_elapsed = setup_started.elapsed();

        let barrier = Arc::new(Barrier::new(
            clients + worker_count + durable_worker_count + 1,
        ));
        let go = Arc::new(AtomicBool::new(false));
        let next_apply_block = Arc::new(AtomicU64::new(0));
        let requested_durable_prefix = Arc::new(AtomicU64::new(0));
        let durable_prefix = Arc::new(AtomicU64::new(0));
        let run_failed = Arc::new(AtomicBool::new(false));
        let durable_handle = if ack_mode.needs_durable() {
            let segment = Arc::clone(&segment);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
            let durable_prefix = Arc::clone(&durable_prefix);
            let run_failed = Arc::clone(&run_failed);
            let durable_group_window = Duration::from_micros(opts.durable_group_us);
            let durable_sync_mode = opts.durable_sync_mode;
            Some(std::thread::spawn(move || -> std::io::Result<()> {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                while durable_prefix.load(Ordering::Acquire) < total_blocks as u64 {
                    if run_failed.load(Ordering::Acquire) {
                        return Err(std::io::Error::other(
                            "durable WAL worker aborted after peer failure",
                        ));
                    }
                    let current = durable_prefix.load(Ordering::Acquire);
                    let mut target = requested_durable_prefix.load(Ordering::Acquire);
                    if target > current {
                        if !durable_group_window.is_zero() {
                            let deadline = Instant::now() + durable_group_window;
                            while Instant::now() < deadline {
                                let observed = requested_durable_prefix.load(Ordering::Acquire);
                                if observed > target {
                                    target = observed;
                                }
                                if target >= total_blocks as u64 {
                                    break;
                                }
                                std::thread::yield_now();
                            }
                        }
                        let published = segment.contiguous_published_prefix(current, target)?;
                        if published > current {
                            if let Err(error) = segment.sync_published_data_frontier_with_mode(
                                published,
                                durable_sync_mode,
                            ) {
                                run_failed.store(true, Ordering::Release);
                                return Err(error);
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
            }))
        } else {
            None
        };
        let mut worker_handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let segment = Arc::clone(&segment);
            let completion = Arc::clone(completion.as_ref().expect("client completion barrier"));
            let store = Arc::clone(store.as_ref().expect("client store"));
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let next_apply_block = Arc::clone(&next_apply_block);
            let run_failed = Arc::clone(&run_failed);
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
                        return Err("apply worker aborted after peer failure".to_string());
                    }
                    let block_id = next_apply_block.fetch_add(1, Ordering::Relaxed);
                    if block_id >= total_blocks as u64 {
                        break;
                    }
                    loop {
                        if run_failed.load(Ordering::Acquire) {
                            return Err("apply worker aborted after peer failure".to_string());
                        }
                        if segment
                            .read_published_block_into(block_id, &mut payload)
                            .is_some()
                        {
                            total.add(
                                store
                                    .apply_block(block_id, &payload)
                                    .map_err(|error| error.to_string())?,
                            );
                            completion
                                .complete(block_id)
                                .map_err(|error| error.to_string())?;
                            break;
                        }
                        std::thread::yield_now();
                    }
                }
                failure_guard.disarm();
                Ok::<DrainStats, String>(total)
            }));
        }

        let mut client_handles = Vec::with_capacity(clients);
        for client_id in 0..clients {
            let segment = Arc::clone(&segment);
            let completion = completion.as_ref().map(Arc::clone);
            let requested_durable_prefix = Arc::clone(&requested_durable_prefix);
            let durable_prefix = Arc::clone(&durable_prefix);
            let run_failed = Arc::clone(&run_failed);
            let barrier = Arc::clone(&barrier);
            let go = Arc::clone(&go);
            let count = base + u64::from((client_id as u64) < rem);
            let first = base * client_id as u64 + rem.min(client_id as u64);
            client_handles.push(std::thread::spawn(move || {
                let mut failure_guard = FailureGuard::new(Arc::clone(&run_failed));
                let sample_capacity = (count.div_ceil(sample_stride) as usize).saturating_add(1);
                let mut latencies = ClientLatencySamples::with_capacity(sample_capacity);
                let mut total = DrainStats::default();
                barrier.wait();
                while !go.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                for offset in 0..count {
                    if run_failed.load(Ordering::Acquire) {
                        return Err("client aborted after peer failure".to_string());
                    }
                    let sequence = first + offset;
                    let sampled = sequence.is_multiple_of(sample_stride);
                    let intent = intent_for(sequence);
                    let started = sampled.then(Instant::now);
                    let block_id = segment
                        .try_publish_intents_position(std::slice::from_ref(&intent))
                        .map_err(|error| error.to_string())?
                        .expect("single-intent publish returns a block id");
                    let logged_at = sampled.then(Instant::now);
                    if ack_mode.needs_durable() {
                        requested_durable_prefix.fetch_max(block_id + 1, Ordering::Release);
                    }
                    total.observe(intent);
                    let mut store_applied_at = None;
                    if let Some(completion) = completion.as_ref() {
                        wait_for_completion_or_failure(
                            completion,
                            block_id + 1,
                            opts.wait_spins,
                            &run_failed,
                        )?;
                        store_applied_at = sampled.then(Instant::now);
                    }
                    if ack_mode.needs_durable() {
                        while durable_prefix.load(Ordering::Acquire) < block_id + 1 {
                            if run_failed.load(Ordering::Acquire) {
                                return Err(
                                    "durable WAL worker failed before acknowledging request"
                                        .to_string(),
                                );
                            }
                            std::thread::yield_now();
                        }
                    }
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
                                latencies
                                    .store_to_ack
                                    .push(duration_ns(acked_at.duration_since(store_applied_at)));
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

        barrier.wait();
        let started = Instant::now();
        go.store(true, Ordering::Release);
        let mut client_total = DrainStats::default();
        let mut samples = ClientLatencySamples::default();
        let mut first_error = None::<String>;
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
                        first_error = Some("client thread panicked".to_string());
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
                        first_error = Some("apply worker thread panicked".to_string());
                    }
                }
            }
        }
        if let Some(handle) = durable_handle {
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
                        first_error = Some("durable WAL worker thread panicked".to_string());
                    }
                }
            }
        }
        if let Some(message) = first_error {
            return Err(std::io::Error::other(message).into());
        }
        elapsed = started.elapsed();
        total = if ack_mode.needs_store() {
            worker_total
        } else {
            client_total
        };
        report = samples
            .report(requested_samples, sample_stride)
            .ok_or("client latency sampling produced no samples")?;
        let validate_started = Instant::now();
        if let Some(store) = store.as_ref() {
            let observed = store
                .validate_applied_blocks(total_blocks as u64)
                .ok_or("client store validation could not read applied prefix")?;
            let expected = stats_for_range(0, events);
            if observed != expected {
                return Err(format!(
                    "client store validation mismatch: got {observed:?}, expected {expected:?}"
                )
                .into());
            }
        }
        store_validation_elapsed = validate_started.elapsed();
    }

    assert_expected_stats(ack_mode.label(), total, events)?;
    if ack_mode.needs_durable() {
        let recovery_started = Instant::now();
        let recovered = recover_wal_segment_by_scan(file_opts.path)?;
        recovery_elapsed = recovery_started.elapsed();
        let expected = stats_for_range(0, events);
        if recovered.recovered_blocks != total_blocks as u64
            || recovered.recovered_records != events
            || recovered.stats != expected
        {
            return Err(format!(
                "client durable recovery mismatch: recovered {recovered:?}, expected_blocks={total_blocks}, expected_stats={expected:?}"
            )
            .into());
        }
    }
    let store_validation = if ack_mode.needs_store() {
        format!(
            " store-validate={:.3}s",
            store_validation_elapsed.as_secs_f64()
        )
    } else {
        String::new()
    };
    let extra = format!(
        " setup={:.3}s clients={clients} workers={worker_count} wal-block={block_size} wait-spins={wait_spins} ack={ack}{store_validation}{recovery}",
        setup_elapsed.as_secs_f64(),
        wait_spins = opts.wait_spins,
        ack = match ack_mode {
            ClientAckMode::Logged => "non-durable-logged",
            ClientAckMode::StoreApplied => "non-durable-store-applied",
            ClientAckMode::DurableStoreApplied => "data-fenced-wal+volatile-store-applied",
        },
        recovery = if ack_mode.needs_durable() {
            format!(
                " durable-group={}us durable-sync-mode={} recover={:.3}s",
                opts.durable_group_us,
                durable_sync_mode_label(opts.durable_sync_mode),
                recovery_elapsed.as_secs_f64()
            )
        } else {
            String::new()
        }
    );
    report_extra(ack_mode.label(), events, elapsed, total, &extra);
    report.print(ack_mode);
    Ok(())
}
