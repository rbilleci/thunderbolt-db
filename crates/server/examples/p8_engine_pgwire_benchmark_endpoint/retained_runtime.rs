use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Instant;

use gpu_db_engine::{ResidentDeviceNullBitmapLayout, ResidentDeviceTextColumnLayout};
use gpu_db_execution::CudaResidentDeviceMemoryReadView;
use gpu_db_protocol::backend::{BackendColumn, BackendWriter};
use gpu_db_protocol::{Select, SqlValue};

use super::result_rows::write_select_result_rows;
use super::retained_batch::{
    retained_select_literal_batch_candidate, retained_select_literal_needle,
    RetainedSelectBatchCandidate, RetainedSelectLiteralBatchKey,
};

#[derive(Clone)]
pub(in super::super) struct RetainedReadRuntimeRoute {
    pub(in super::super) table: String,
    pub(in super::super) generation: u64,
    pub(in super::super) row_count: u64,
    pub(in super::super) int4_columns: Vec<String>,
    pub(in super::super) text_columns: Vec<ResidentDeviceTextColumnLayout>,
    pub(in super::super) null_columns: Vec<ResidentDeviceNullBitmapLayout>,
    pub(in super::super) read_view: CudaResidentDeviceMemoryReadView,
}

#[derive(Default)]
pub(in super::super) struct RetainedReadRuntimeStats {
    pub(in super::super) published: u64,
    pub(in super::super) invalidations: u64,
    pub(in super::super) attempts: u64,
    pub(in super::super) hits: u64,
    pub(in super::super) misses: u64,
    pub(in super::super) unsupported: u64,
    pub(in super::super) failures: u64,
    pub(in super::super) batches: u64,
    pub(in super::super) batched_requests: u64,
    pub(in super::super) max_batch: u64,
    pub(in super::super) request_queue_wait_micros_total: u64,
    pub(in super::super) request_queue_wait_micros_max: u64,
    pub(in super::super) batch_execute_wall_micros_total: u64,
    pub(in super::super) batch_execute_wall_micros_max: u64,
    pub(in super::super) route_stats:
        HashMap<RetainedSelectLiteralBatchKey, RetainedReadRuntimeRouteStats>,
}

#[derive(Clone, Default)]
pub(in super::super) struct RetainedReadRuntimeRouteStats {
    pub(in super::super) batches: u64,
    pub(in super::super) requests: u64,
    pub(in super::super) max_batch: u64,
    pub(in super::super) request_queue_wait_micros_total: u64,
    pub(in super::super) request_queue_wait_micros_max: u64,
    pub(in super::super) batch_execute_wall_micros_total: u64,
    pub(in super::super) batch_execute_wall_micros_max: u64,
}

struct RetainedReadRuntimeInner {
    enabled: bool,
    generation: u64,
    in_flight: u64,
    route: Option<RetainedReadRuntimeRoute>,
    stats: RetainedReadRuntimeStats,
}

pub(in super::super) struct RetainedReadRuntime {
    inner: Mutex<RetainedReadRuntimeInner>,
    idle: Condvar,
    work_txs: Vec<mpsc::Sender<RetainedReadRuntimeWork>>,
    batch_max: usize,
}

pub(in super::super) struct RetainedReadRuntimeWork {
    route: RetainedReadRuntimeRoute,
    batch_key: RetainedSelectLiteralBatchKey,
    needle: i32,
    enqueued_at: Instant,
    response_tx: mpsc::Sender<Result<Vec<u8>, String>>,
}

impl RetainedReadRuntime {
    pub(in super::super) fn new(
        enabled: bool,
        batch_max: usize,
        worker_count: usize,
    ) -> (Self, Vec<mpsc::Receiver<RetainedReadRuntimeWork>>) {
        let (work_txs, work_rxs) = if enabled {
            let mut work_txs = Vec::new();
            let mut work_rxs = Vec::new();
            for _ in 0..worker_count.max(1) {
                let (tx, rx) = mpsc::channel();
                work_txs.push(tx);
                work_rxs.push(rx);
            }
            (work_txs, work_rxs)
        } else {
            (Vec::new(), Vec::new())
        };
        (
            Self {
                inner: Mutex::new(RetainedReadRuntimeInner {
                    enabled,
                    generation: 0,
                    in_flight: 0,
                    route: None,
                    stats: RetainedReadRuntimeStats::default(),
                }),
                idle: Condvar::new(),
                work_txs,
                batch_max: batch_max.max(1),
            },
            work_rxs,
        )
    }

    pub(in super::super) fn publish(&self, route: RetainedReadRuntimeRoute) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "retained read runtime lock poisoned".to_string())?;
        if !inner.enabled {
            return Ok(());
        }
        while inner.in_flight > 0 {
            inner = self
                .idle
                .wait(inner)
                .map_err(|_| "retained read runtime lock poisoned".to_string())?;
        }
        inner.generation = inner.generation.saturating_add(1);
        inner.route = Some(route);
        inner.stats.published = inner.stats.published.saturating_add(1);
        Ok(())
    }

    pub(in super::super) fn invalidate(&self) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "retained read runtime lock poisoned".to_string())?;
        if !inner.enabled {
            return Ok(());
        }
        inner.route = None;
        inner.generation = inner.generation.saturating_add(1);
        inner.stats.invalidations = inner.stats.invalidations.saturating_add(1);
        while inner.in_flight > 0 {
            inner = self
                .idle
                .wait(inner)
                .map_err(|_| "retained read runtime lock poisoned".to_string())?;
        }
        Ok(())
    }

    pub(in super::super) fn snapshot_stats(&self) -> Result<RetainedReadRuntimeStats, String> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| "retained read runtime lock poisoned".to_string())?;
        Ok(RetainedReadRuntimeStats {
            published: inner.stats.published,
            invalidations: inner.stats.invalidations,
            attempts: inner.stats.attempts,
            hits: inner.stats.hits,
            misses: inner.stats.misses,
            unsupported: inner.stats.unsupported,
            failures: inner.stats.failures,
            batches: inner.stats.batches,
            batched_requests: inner.stats.batched_requests,
            max_batch: inner.stats.max_batch,
            request_queue_wait_micros_total: inner.stats.request_queue_wait_micros_total,
            request_queue_wait_micros_max: inner.stats.request_queue_wait_micros_max,
            batch_execute_wall_micros_total: inner.stats.batch_execute_wall_micros_total,
            batch_execute_wall_micros_max: inner.stats.batch_execute_wall_micros_max,
            route_stats: inner.stats.route_stats.clone(),
        })
    }

    pub(in super::super) fn try_execute(
        &self,
        sql: &str,
        select: &Select,
    ) -> Result<Option<Vec<u8>>, String> {
        let candidate = retained_select_literal_batch_candidate(sql, select.clone());
        let Some(RetainedSelectBatchCandidate::Literal { batch_key, .. }) = candidate else {
            return Ok(None);
        };
        let needle = match retained_select_literal_needle(select) {
            Some(needle) => needle,
            None => return Ok(None),
        };
        let (route, work_tx) = {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| "retained read runtime lock poisoned".to_string())?;
            if !inner.enabled {
                return Ok(None);
            }
            inner.stats.attempts = inner.stats.attempts.saturating_add(1);
            let Some(route) = inner.route.clone() else {
                inner.stats.misses = inner.stats.misses.saturating_add(1);
                return Ok(None);
            };
            if route.table != batch_key.table || route.generation == 0 {
                inner.stats.misses = inner.stats.misses.saturating_add(1);
                return Ok(None);
            }
            let filter_is_int4 = route
                .int4_columns
                .iter()
                .any(|column| column == &batch_key.filter_column);
            let supported_projection = batch_key.projection_columns.iter().all(|column| {
                route
                    .int4_columns
                    .iter()
                    .any(|candidate| candidate == column)
                    || route
                        .text_columns
                        .iter()
                        .any(|candidate| candidate.name == *column)
            });
            let text_projection_count = batch_key
                .projection_columns
                .iter()
                .filter(|column| {
                    route
                        .text_columns
                        .iter()
                        .any(|candidate| candidate.name == **column)
                })
                .count();
            if !filter_is_int4 || !supported_projection || text_projection_count > 1 {
                inner.stats.unsupported = inner.stats.unsupported.saturating_add(1);
                return Ok(None);
            }
            inner.in_flight = inner.in_flight.saturating_add(1);
            let Some(work_tx) = self.worker_tx_for(&batch_key) else {
                inner.in_flight = inner.in_flight.saturating_sub(1);
                self.idle.notify_all();
                inner.stats.misses = inner.stats.misses.saturating_add(1);
                return Ok(None);
            };
            (route, work_tx)
        };

        let (response_tx, response_rx) = mpsc::channel();
        let work = RetainedReadRuntimeWork {
            route,
            batch_key,
            needle,
            enqueued_at: Instant::now(),
            response_tx,
        };
        if work_tx.send(work).is_err() {
            self.finish_work(1, 0, 0, 1, None, 0, 0, 0)?;
            return Ok(None);
        }
        match response_rx.recv() {
            Ok(Ok(bytes)) => Ok(Some(bytes)),
            Ok(Err(err)) => {
                eprintln!("retained read runtime view failed, falling back to owner: {err}");
                Ok(None)
            }
            Err(_) => Ok(None),
        }
    }

    fn worker_tx_for(
        &self,
        batch_key: &RetainedSelectLiteralBatchKey,
    ) -> Option<mpsc::Sender<RetainedReadRuntimeWork>> {
        if self.work_txs.is_empty() {
            return None;
        }
        let mut hasher = DefaultHasher::new();
        batch_key.hash(&mut hasher);
        let worker_idx = (hasher.finish() as usize) % self.work_txs.len();
        self.work_txs.get(worker_idx).cloned()
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_work(
        &self,
        completed: u64,
        hits: u64,
        batch_size: u64,
        failures: u64,
        batch_key: Option<&RetainedSelectLiteralBatchKey>,
        request_queue_wait_micros_total: u64,
        request_queue_wait_micros_max: u64,
        batch_execute_wall_micros: u64,
    ) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "retained read runtime lock poisoned".to_string())?;
        inner.in_flight = inner.in_flight.saturating_sub(completed);
        inner.stats.hits = inner.stats.hits.saturating_add(hits);
        inner.stats.failures = inner.stats.failures.saturating_add(failures);
        if batch_size > 0 {
            inner.stats.batches = inner.stats.batches.saturating_add(1);
            inner.stats.batched_requests = inner.stats.batched_requests.saturating_add(batch_size);
            inner.stats.max_batch = inner.stats.max_batch.max(batch_size);
            inner.stats.request_queue_wait_micros_total = inner
                .stats
                .request_queue_wait_micros_total
                .saturating_add(request_queue_wait_micros_total);
            inner.stats.request_queue_wait_micros_max = inner
                .stats
                .request_queue_wait_micros_max
                .max(request_queue_wait_micros_max);
            inner.stats.batch_execute_wall_micros_total = inner
                .stats
                .batch_execute_wall_micros_total
                .saturating_add(batch_execute_wall_micros);
            inner.stats.batch_execute_wall_micros_max = inner
                .stats
                .batch_execute_wall_micros_max
                .max(batch_execute_wall_micros);
            if let Some(batch_key) = batch_key {
                let route_stats = inner
                    .stats
                    .route_stats
                    .entry(batch_key.clone())
                    .or_default();
                route_stats.batches = route_stats.batches.saturating_add(1);
                route_stats.requests = route_stats.requests.saturating_add(batch_size);
                route_stats.max_batch = route_stats.max_batch.max(batch_size);
                route_stats.request_queue_wait_micros_total = route_stats
                    .request_queue_wait_micros_total
                    .saturating_add(request_queue_wait_micros_total);
                route_stats.request_queue_wait_micros_max = route_stats
                    .request_queue_wait_micros_max
                    .max(request_queue_wait_micros_max);
                route_stats.batch_execute_wall_micros_total = route_stats
                    .batch_execute_wall_micros_total
                    .saturating_add(batch_execute_wall_micros);
                route_stats.batch_execute_wall_micros_max = route_stats
                    .batch_execute_wall_micros_max
                    .max(batch_execute_wall_micros);
            }
        }
        self.idle.notify_all();
        Ok(())
    }

    pub(in super::super) fn run_worker(
        self: Arc<Self>,
        work_rx: mpsc::Receiver<RetainedReadRuntimeWork>,
    ) {
        let mut backlog = VecDeque::new();
        loop {
            let first = if let Some(work) = backlog.pop_front() {
                work
            } else {
                match work_rx.recv() {
                    Ok(work) => work,
                    Err(_) => break,
                }
            };
            let mut batch = vec![first];
            while batch.len() < self.batch_max {
                let next = if let Some(position) = backlog
                    .iter()
                    .position(|work| work.batch_key == batch[0].batch_key)
                {
                    backlog.remove(position)
                } else {
                    match work_rx.try_recv() {
                        Ok(work) => Some(work),
                        Err(mpsc::TryRecvError::Empty) => None,
                        Err(mpsc::TryRecvError::Disconnected) => None,
                    }
                };
                let Some(work) = next else {
                    break;
                };
                if work.batch_key == batch[0].batch_key {
                    batch.push(work);
                } else {
                    backlog.push_back(work);
                }
            }
            let batch_size = u64::try_from(batch.len()).unwrap_or(u64::MAX);
            let batch_key = batch[0].batch_key.clone();
            let batch_queued_at = Instant::now();
            let mut queue_wait_micros_total = 0_u64;
            let mut queue_wait_micros_max = 0_u64;
            for work in &batch {
                let queue_wait_micros =
                    u64::try_from(batch_queued_at.duration_since(work.enqueued_at).as_micros())
                        .unwrap_or(u64::MAX);
                queue_wait_micros_total = queue_wait_micros_total.saturating_add(queue_wait_micros);
                queue_wait_micros_max = queue_wait_micros_max.max(queue_wait_micros);
            }
            let batch_started = Instant::now();
            match execute_retained_read_runtime_batch(&batch) {
                Ok(outputs) => {
                    let batch_wall_micros =
                        u64::try_from(batch_started.elapsed().as_micros()).unwrap_or(u64::MAX);
                    for (work, output) in batch.into_iter().zip(outputs) {
                        let _ = work.response_tx.send(Ok(output));
                    }
                    let _ = self.finish_work(
                        batch_size,
                        batch_size,
                        batch_size,
                        0,
                        Some(&batch_key),
                        queue_wait_micros_total,
                        queue_wait_micros_max,
                        batch_wall_micros,
                    );
                }
                Err(err) => {
                    let batch_wall_micros =
                        u64::try_from(batch_started.elapsed().as_micros()).unwrap_or(u64::MAX);
                    for work in batch {
                        let _ = work.response_tx.send(Err(err.clone()));
                    }
                    let _ = self.finish_work(
                        batch_size,
                        0,
                        batch_size,
                        batch_size,
                        Some(&batch_key),
                        queue_wait_micros_total,
                        queue_wait_micros_max,
                        batch_wall_micros,
                    );
                }
            }
        }
        for work in backlog {
            let _ = work
                .response_tx
                .send(Err("retained read runtime worker stopped".to_string()));
            let _ = self.finish_work(1, 0, 0, 1, None, 0, 0, 0);
        }
    }
}

fn execute_retained_read_runtime_batch(
    works: &[RetainedReadRuntimeWork],
) -> Result<Vec<Vec<u8>>, String> {
    let first = works
        .first()
        .ok_or_else(|| "retained read runtime batch requires work".to_string())?;
    let route = &first.route;
    let batch_key = &first.batch_key;
    let int4_width = std::mem::size_of::<i32>() as u64;
    let column_bytes = route
        .row_count
        .checked_mul(int4_width)
        .ok_or_else(|| "retained read runtime row-count overflow".to_string())?;
    let column_offset = |column: &str| -> Result<u64, String> {
        let ordinal = route
            .int4_columns
            .iter()
            .position(|candidate| candidate == column)
            .ok_or_else(|| format!("retained read runtime missing int4 column {column}"))?;
        (std::mem::size_of::<u64>() as u64)
            .checked_add(
                (ordinal as u64)
                    .checked_mul(column_bytes)
                    .ok_or_else(|| "retained read runtime column offset overflow".to_string())?,
            )
            .ok_or_else(|| "retained read runtime column offset overflow".to_string())
    };
    let filter_offset = column_offset(&batch_key.filter_column)?;
    let text_projection = batch_key.projection_columns.iter().find_map(|column| {
        route
            .text_columns
            .iter()
            .find(|candidate| candidate.name == *column)
    });
    if let Some(text_layout) = text_projection {
        let int4_projection_columns = batch_key
            .projection_columns
            .iter()
            .filter(|column| **column != text_layout.name)
            .cloned()
            .collect::<Vec<_>>();
        let int4_projection_offsets = int4_projection_columns
            .iter()
            .map(|column| column_offset(column))
            .collect::<Result<Vec<_>, _>>()?;
        return execute_retained_read_runtime_text_batch(
            works,
            route,
            batch_key,
            RetainedReadRuntimeTextBatchPlan {
                filter_offset,
                filter_validity_bitmap_offset: route
                    .null_columns
                    .iter()
                    .find(|layout| layout.name == batch_key.filter_column)
                    .map(|layout| layout.bitmap_byte_offset),
                int4_projection_columns: &int4_projection_columns,
                int4_projection_offsets: &int4_projection_offsets,
                text_layout,
            },
        );
    }
    let projection_offsets = batch_key
        .projection_columns
        .iter()
        .map(|column| column_offset(column))
        .collect::<Result<Vec<_>, _>>()?;
    let mut unique_needles = Vec::new();
    let mut unique_by_needle = HashMap::new();
    let mut work_to_unique = Vec::with_capacity(works.len());
    for work in works {
        let unique_idx = if let Some(idx) = unique_by_needle.get(&work.needle) {
            *idx
        } else {
            let idx = unique_needles.len();
            unique_by_needle.insert(work.needle, idx);
            unique_needles.push(work.needle);
            idx
        };
        work_to_unique.push(unique_idx);
    }
    let submitted = route
        .read_view
        .submit_match_project_i32_equal_any_from_payload(
            filter_offset,
            &unique_needles,
            &projection_offsets,
            route.row_count,
        )
        .map_err(|err| err.to_string())?;
    let (projected_rows, _kernel_event_elapsed_us) = submitted
        .complete_detached()
        .map_err(|err| err.to_string())?;
    let mut rows_by_unique = vec![Vec::new(); unique_needles.len()];
    for row in projected_rows {
        if let Some(rows) = rows_by_unique.get_mut(row.needle_index) {
            rows.push(
                row.values
                    .into_iter()
                    .map(SqlValue::Int4)
                    .collect::<Vec<_>>(),
            );
        }
    }
    let columns = batch_key
        .projection_columns
        .iter()
        .map(|column| BackendColumn::new(column, 23, 4))
        .collect::<Vec<_>>();
    let mut outputs = Vec::with_capacity(works.len());
    for unique_idx in work_to_unique {
        let rows = rows_by_unique
            .get(unique_idx)
            .ok_or_else(|| format!("retained read runtime missing unique result {unique_idx}"))?;
        let mut bytes = Vec::new();
        let mut writer = BackendWriter::new(&mut bytes);
        write_select_result_rows(
            &mut writer,
            &columns,
            &gpu_db_engine::RowBlock::from(rows.clone()),
        )
        .map_err(|err| err.to_string())?;
        writer
            .ready_for_query(false)
            .map_err(|err| err.to_string())?;
        outputs.push(bytes);
    }
    Ok(outputs)
}

struct RetainedReadRuntimeTextBatchPlan<'a> {
    filter_offset: u64,
    filter_validity_bitmap_offset: Option<u64>,
    int4_projection_columns: &'a [String],
    int4_projection_offsets: &'a [u64],
    text_layout: &'a ResidentDeviceTextColumnLayout,
}

fn execute_retained_read_runtime_text_batch(
    works: &[RetainedReadRuntimeWork],
    route: &RetainedReadRuntimeRoute,
    batch_key: &RetainedSelectLiteralBatchKey,
    plan: RetainedReadRuntimeTextBatchPlan<'_>,
) -> Result<Vec<Vec<u8>>, String> {
    let mut unique_needles = Vec::new();
    let mut unique_by_needle = HashMap::new();
    let mut work_to_unique = Vec::with_capacity(works.len());
    for work in works {
        let unique_idx = if let Some(idx) = unique_by_needle.get(&work.needle) {
            *idx
        } else {
            let idx = unique_needles.len();
            unique_by_needle.insert(work.needle, idx);
            unique_needles.push(work.needle);
            idx
        };
        work_to_unique.push(unique_idx);
    }
    let projected_rows = route
        .read_view
        .match_project_i32_equal_any_text_from_payload(
            plan.filter_offset,
            plan.filter_validity_bitmap_offset,
            &unique_needles,
            plan.int4_projection_offsets,
            plan.text_layout.offsets_byte_offset,
            plan.text_layout.bytes_byte_offset,
            plan.text_layout.bytes_len,
            route
                .null_columns
                .iter()
                .find(|layout| layout.name == plan.text_layout.name)
                .map(|layout| layout.bitmap_byte_offset),
            route.row_count,
        )
        .map_err(|err| err.to_string())?;
    let int4_projection_index = plan
        .int4_projection_columns
        .iter()
        .enumerate()
        .map(|(idx, column)| (column.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut rows_by_unique = vec![Vec::new(); unique_needles.len()];
    for row in projected_rows {
        if let Some(rows) = rows_by_unique.get_mut(row.needle_index) {
            let mut values = Vec::with_capacity(batch_key.projection_columns.len());
            for column in &batch_key.projection_columns {
                if column == &plan.text_layout.name {
                    if row.text_is_null {
                        values.push(SqlValue::Null);
                        continue;
                    }
                    values.push(SqlValue::Text(row.text.clone()));
                } else {
                    let value_idx =
                        int4_projection_index.get(column.as_str()).ok_or_else(|| {
                            format!("retained read runtime missing int4 projection {column}")
                        })?;
                    let value = row.values.get(*value_idx).ok_or_else(|| {
                        format!("retained read runtime missing int4 value for {column}")
                    })?;
                    values.push(SqlValue::Int4(*value));
                }
            }
            rows.push(values);
        }
    }
    let columns = batch_key
        .projection_columns
        .iter()
        .map(|column| {
            if column == &plan.text_layout.name {
                BackendColumn::new(column, 25, -1)
            } else {
                BackendColumn::new(column, 23, 4)
            }
        })
        .collect::<Vec<_>>();
    let mut outputs = Vec::with_capacity(works.len());
    for unique_idx in work_to_unique {
        let rows = rows_by_unique
            .get(unique_idx)
            .ok_or_else(|| format!("retained read runtime missing unique result {unique_idx}"))?;
        let mut bytes = Vec::new();
        let mut writer = BackendWriter::new(&mut bytes);
        write_select_result_rows(
            &mut writer,
            &columns,
            &gpu_db_engine::RowBlock::from(rows.clone()),
        )
        .map_err(|err| err.to_string())?;
        writer
            .ready_for_query(false)
            .map_err(|err| err.to_string())?;
        outputs.push(bytes);
    }
    Ok(outputs)
}
