//! Resident-device-memory probe execution (P0 §9.6 decomposition, behavior-
//! preserving): a focused `impl Engine` block holding the GPU resident-route
//! execution surface — every execute_relational_*_with_resident_device_memory_probe
//! method (counts, filtered/partitioned counts, scalar + grouped aggregates,
//! projections incl. equality/multi-column/distinct/ordered), the multi-column
//! projection batch path, plus the select-with-backend dispatch and route-
//! execution observation counter they feed.

use super::*;

/// The GPU-resident filter predicate IR for the plan->kernel compiler (P0 spine 1.3): exactly the
/// shapes the resident kernels evaluate on device. Column fields are indices into the bound table;
/// the executor resolves them to device byte-offsets against the snapshot layout. Grows as more
/// shapes migrate (text prefix, IN, DNF, non-int4 types).
#[derive(Debug, Clone)]
pub(crate) enum ResidentPredicate {
    /// No filter — every row (e.g. COUNT(*) over the whole table).
    All,
    /// `int4_col = needle`.
    Int4Equal { col: usize, needle: i32 },
    /// `int4_col` `<` / `<=` / `>` / `>=` `needle`.
    Int4Compare {
        col: usize,
        needle: i32,
        comparison: CudaI32Comparison,
    },
    /// `int4_col BETWEEN lower AND upper` (both inclusive).
    Int4Between {
        col: usize,
        lower: i32,
        upper: i32,
    },
}

/// A single-row scalar aggregate the resident plan computes over the predicate-selected rows (P0
/// spine 1.3). Column fields are table column indices; the executor resolves device byte-offsets.
/// `Count` carries no column (COUNT(*) counts rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentScalarAggregate {
    Count,
    Sum { col: usize },
    Avg { col: usize },
    Min { col: usize },
    Max { col: usize },
}

/// The relational operator a resident plan evaluates on device over the predicate-selected rows.
/// Grows one variant per migrated shape family (grouped aggregate, projection, ...).
#[derive(Debug, Clone)]
pub(crate) enum ResidentOp {
    /// One single-row scalar aggregate (COUNT/SUM/AVG/MIN/MAX) over the predicate-selected rows.
    ScalarAggregate(ResidentScalarAggregate),
}

/// A compiled GPU-resident query plan (P0 spine 1.3 plan->kernel compiler): a device [predicate]
/// plus the [operator] to evaluate over the rows it selects. [`Engine::execute_resident_plan`] runs
/// the shared residency skeleton once and dispatches the matching device kernel by (predicate, op)
/// shape — replacing the per-shape resident probe methods.
///
/// [predicate]: ResidentPredicate
/// [operator]: ResidentOp
#[derive(Debug, Clone)]
pub(crate) struct ResidentPlan {
    predicate: ResidentPredicate,
    op: ResidentOp,
}

/// Compile a bound COUNT(*) WHERE clause into a [`ResidentPredicate`], or reject shapes the resident
/// count kernels do not (yet) cover. Currently: count-all (no filter) or one int4 equality/range
/// predicate. (Text-prefix, IN, BETWEEN, DNF migrate in follow-up slices.)
fn compile_resident_count_predicate(
    table: &RelationalTable,
    bound: &BoundRelationalSelect,
) -> Result<ResidentPredicate, ExecuteError> {
    if bound.filter_groups.is_empty() {
        return Ok(ResidentPredicate::All);
    }
    if bound.filter_groups.len() == 1 && bound.filter_groups[0].len() == 1 {
        let (col, op, value) = bound.filter_groups[0][0].clone();
        if table.columns[col].ty == SqlType::Int4 {
            if let SqlValue::Int4(needle) = value {
                if op == SelectFilterOp::Eq {
                    return Ok(ResidentPredicate::Int4Equal { col, needle });
                }
                if let Some(comparison) = resident_device_i32_comparison(op) {
                    return Ok(ResidentPredicate::Int4Compare {
                        col,
                        needle,
                        comparison,
                    });
                }
            }
        }
    }
    Err(ExecuteError::Engine(EngineError::ApplyFailed(
        "resident COUNT(*) supports count-all or one int4 equality/range predicate".to_string(),
    )))
}

/// Snapshot row count as the `u64` the resident kernels take.
fn resident_snapshot_row_count(
    snapshot: &RelationalResidencySnapshot,
) -> Result<u64, ExecuteError> {
    u64::try_from(snapshot.row_count).map_err(|_| {
        ExecuteError::Engine(EngineError::ApplyFailed(
            "resident snapshot row count exceeds retained device-memory proof range".to_string(),
        ))
    })
}

/// A device count (`u64`) as the `i64` a COUNT(*) result column carries.
fn resident_count_to_i64(count: u64) -> Result<i64, ExecuteError> {
    i64::try_from(count).map_err(|_| {
        ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "resident device-memory count {count} exceeds supported COUNT(*) result range"
        )))
    })
}

/// Compile the WHERE clause of a resident scalar aggregate (SUM/AVG/MIN/MAX) into a
/// [`ResidentPredicate`]. The retained stats kernels evaluate the predicate on the SAME column they
/// aggregate, so a filter must target `agg_col`; supported shapes are unfiltered, one int4
/// non-equality comparison, or one inclusive int4 `BETWEEN`. (The check order — cross-column before
/// literal type before bound type — matches the pre-unification probe diagnostics.)
fn compile_resident_scalar_aggregate_predicate(
    bound: &BoundRelationalSelect,
    agg_col: usize,
) -> Result<ResidentPredicate, ExecuteError> {
    if bound.filter_groups.is_empty() {
        return Ok(ResidentPredicate::All);
    }
    if bound.filter_groups.len() == 1 {
        let predicates = &bound.filter_groups[0];
        if predicates.len() == 1 {
            let (filter_idx, op, value) = predicates[0].clone();
            if filter_idx != agg_col {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident scalar aggregate requires the predicate column to match the aggregate column"
                        .to_string(),
                )));
            }
            let Some(comparison) = resident_device_i32_comparison(op) else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident scalar aggregate supports only non-equality int4 comparisons"
                        .to_string(),
                )));
            };
            let SqlValue::Int4(needle) = value else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident scalar aggregate supports only int4 comparison literals".to_string(),
                )));
            };
            return Ok(ResidentPredicate::Int4Compare {
                col: agg_col,
                needle,
                comparison,
            });
        }
        if predicates.len() == 2 {
            let mut lower = None;
            let mut upper = None;
            for (filter_idx, op, value) in predicates {
                if *filter_idx != agg_col {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident scalar aggregate requires the predicate column to match the aggregate column"
                            .to_string(),
                    )));
                }
                let SqlValue::Int4(bound_value) = value else {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident scalar aggregate BETWEEN supports only int4 bounds".to_string(),
                    )));
                };
                match op {
                    SelectFilterOp::Gte => lower = Some(*bound_value),
                    SelectFilterOp::Lte => upper = Some(*bound_value),
                    _ => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "resident scalar aggregate BETWEEN supports only inclusive int4 bounds"
                                .to_string(),
                        )));
                    }
                }
            }
            let lower = lower.ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident scalar aggregate BETWEEN requires a lower inclusive bound".to_string(),
                ))
            })?;
            let upper = upper.ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident scalar aggregate BETWEEN requires an upper inclusive bound".to_string(),
                ))
            })?;
            return Ok(ResidentPredicate::Int4Between {
                col: agg_col,
                lower,
                upper,
            });
        }
    }
    Err(ExecuteError::Engine(EngineError::ApplyFailed(
        "resident scalar aggregate supports only one int4 BETWEEN predicate or one int4 comparison on the aggregate column"
            .to_string(),
    )))
}

/// Compile a bound `SELECT` into a [`ResidentPlan`], or reject shapes the resident kernels do not
/// cover. Slice 2 covers the single-row scalar aggregate family: `COUNT(*)` and `SUM/AVG/MIN/MAX`
/// over an int4 column, each with the predicate envelope its kernels support.
fn compile_resident_plan(
    table: &RelationalTable,
    bound: &BoundRelationalSelect,
    select: &Select,
) -> Result<ResidentPlan, ExecuteError> {
    if select.distinct
        || select.group_by.is_some()
        || !select.having_groups.is_empty()
        || !select.order_by.is_empty()
        || select.limit.is_some()
        || select.offset.is_some()
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident scalar aggregate proof supports only single-row COUNT(*)/SUM/AVG/MIN/MAX without DISTINCT/GROUP BY/HAVING/ORDER BY/LIMIT/OFFSET"
                .to_string(),
        )));
    }
    match &select.projection {
        SelectProjection::CountAll => Ok(ResidentPlan {
            predicate: compile_resident_count_predicate(table, bound)?,
            op: ResidentOp::ScalarAggregate(ResidentScalarAggregate::Count),
        }),
        SelectProjection::Sum { column }
        | SelectProjection::Avg { column }
        | SelectProjection::Min { column }
        | SelectProjection::Max { column } => {
            let col = relational_column_index(table, column)?;
            // Compile the predicate first: it validates literal/bound types and the predicate-column
            // match before the aggregate column's int4 check, matching the pre-unification order.
            let predicate = compile_resident_scalar_aggregate_predicate(bound, col)?;
            if table.columns[col].ty != SqlType::Int4 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident scalar aggregate proof currently supports only int4 aggregate columns"
                        .to_string(),
                )));
            }
            let aggregate = match &select.projection {
                SelectProjection::Sum { .. } => ResidentScalarAggregate::Sum { col },
                SelectProjection::Avg { .. } => ResidentScalarAggregate::Avg { col },
                SelectProjection::Min { .. } => ResidentScalarAggregate::Min { col },
                SelectProjection::Max { .. } => ResidentScalarAggregate::Max { col },
                _ => unreachable!("projection matched a scalar aggregate above"),
            };
            Ok(ResidentPlan {
                predicate,
                op: ResidentOp::ScalarAggregate(aggregate),
            })
        }
        _ => Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "resident scalar aggregate proof currently supports only COUNT(*)/SUM/AVG/MIN/MAX(int4_column)"
                .to_string(),
        ))),
    }
}

/// Materialize a SUM/AVG/MIN/MAX result from the retained `CudaI32Stats` a filter/BETWEEN stats
/// kernel returns (shared by the filtered and BETWEEN scalar-aggregate paths). An empty MIN/MAX
/// domain yields the empty-text sentinel the CPU engine emits for an empty aggregate.
fn materialize_resident_scalar_stats(
    aggregate: ResidentScalarAggregate,
    stats: &CudaI32Stats,
) -> Result<SqlValue, ExecuteError> {
    Ok(match aggregate {
        ResidentScalarAggregate::Sum { .. } => SqlValue::Int8(stats.sum),
        ResidentScalarAggregate::Avg { .. } => average_sql_value(
            i128::from(stats.sum),
            usize::try_from(stats.count).map_err(|_| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident scalar aggregate count {} exceeds AVG result range",
                    stats.count
                )))
            })?,
        ),
        ResidentScalarAggregate::Min { .. } => stats
            .min
            .map(SqlValue::Int4)
            .unwrap_or_else(|| SqlValue::Text(String::new())),
        ResidentScalarAggregate::Max { .. } => stats
            .max
            .map(SqlValue::Int4)
            .unwrap_or_else(|| SqlValue::Text(String::new())),
        ResidentScalarAggregate::Count => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident scalar stats materialization received COUNT".to_string(),
            )));
        }
    })
}

/// Reduce the NULL-aware self-grouped stats (M3 — doc 21) of a scalar SUM/AVG/MIN/MAX into one value.
/// The groups cover only the surviving non-NULL (and, when filtered, matching) rows, so a zero-survivor
/// result is SQL NULL for every aggregate (PG: an aggregate of no rows is NULL — never 0 or the
/// empty-text sentinel). Shared by the unfiltered-nullable and filtered-compare-nullable paths. COUNT is
/// not on this path (it routes to `run_resident_count`).
fn reduce_nullable_grouped_stats(
    aggregate: ResidentScalarAggregate,
    grouped_stats: &[CudaI32GroupedStats],
) -> Result<SqlValue, ExecuteError> {
    let total_count = grouped_stats.iter().map(|group| group.count).sum::<u64>();
    let total_sum = grouped_stats.iter().map(|group| group.sum).sum::<i64>();
    Ok(match aggregate {
        ResidentScalarAggregate::Sum { .. } => {
            if total_count == 0 {
                SqlValue::Null
            } else {
                SqlValue::Int8(total_sum)
            }
        }
        ResidentScalarAggregate::Avg { .. } => {
            if total_count == 0 {
                SqlValue::Null
            } else {
                average_sql_value(
                    i128::from(total_sum),
                    usize::try_from(total_count).map_err(|_| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "resident device-memory scalar aggregate count {total_count} exceeds AVG result range"
                        )))
                    })?,
                )
            }
        }
        ResidentScalarAggregate::Min { .. } => grouped_stats
            .iter()
            .map(|group| group.min)
            .min()
            .map(SqlValue::Int4)
            .unwrap_or(SqlValue::Null),
        ResidentScalarAggregate::Max { .. } => grouped_stats
            .iter()
            .map(|group| group.max)
            .max()
            .map(SqlValue::Int4)
            .unwrap_or(SqlValue::Null),
        ResidentScalarAggregate::Count => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident COUNT routed to scalar path".to_string(),
            )));
        }
    })
}

/// D2H byte estimate for a `grouped_stats` reduction: the group columns (i32 group + u64 count + i64 sum
/// + 2×i32 min/max) per group plus the u64 group-count header. Shared by the nullable scalar paths.
fn nullable_grouped_stats_d2h_bytes(copied_group_count: usize) -> u64 {
    copied_group_count
        .checked_mul(
            std::mem::size_of::<i32>()
                + std::mem::size_of::<u64>()
                + std::mem::size_of::<i64>()
                + (2 * std::mem::size_of::<i32>()),
        )
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(u64::MAX)
}

impl Engine {
    /// Plan->kernel driver for the resident scalar-aggregate family (P0 spine 1.3): ONE path for
    /// every resident `COUNT(*)`/`SUM`/`AVG`/`MIN`/`MAX` shape. Binds the select, compiles it into a
    /// [`ResidentPlan`], runs the shared residency skeleton (MVCC pin -> snapshot -> identity ->
    /// validity) once, then dispatches the matching device kernel by (predicate, op) shape —
    /// replacing the per-shape resident count and scalar-aggregate probe methods.
    pub fn execute_resident_plan(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let plan = compile_resident_plan(&table, &bound, select)?;

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot_ref(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
        if snapshot.schema != table.schema || snapshot.table != table.name {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot no longer matches catalog table identity".to_string(),
            )));
        }
        if !snapshot.is_valid() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" resident snapshot is invalid",
                table.name
            ))));
        }

        match plan.op {
            ResidentOp::ScalarAggregate(ResidentScalarAggregate::Count) => {
                self.run_resident_count(&table, bound, &snapshot, &plan.predicate, access_path)
            }
            ResidentOp::ScalarAggregate(aggregate) => self.run_resident_scalar_aggregate(
                &table,
                bound,
                &snapshot,
                &plan.predicate,
                aggregate,
                access_path,
            ),
        }
    }

    /// Physical path for the resident `COUNT(*)` family: dispatch the matching count kernel by
    /// predicate and return the single-row count. Unfiltered COUNT proves residency with a device
    /// header-count (recorded as a route device lookup); the int4 predicate forms scan the payload.
    fn run_resident_count(
        &self,
        table: &RelationalTable,
        bound: BoundRelationalSelect,
        snapshot: &RelationalResidencySnapshot,
        predicate: &ResidentPredicate,
        access_path: RelationalAccessPath,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let snapshot_gpu_id = snapshot.gpu_id;
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;

        let count: i64 = match *predicate {
            ResidentPredicate::All => {
                // Unfiltered COUNT(*): a device header-count kernel proves residency (recorded as a
                // route device lookup), validated against the snapshot's expected row count.
                let lookup_started = Instant::now();
                let row_count = device_memory.count_rows_from_header().map_err(|err| {
                    ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                })?;
                let lookup_micros = lookup_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX);
                if row_count != snapshot.row_count as u64 {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident device-memory row-count proof returned {row_count}, expected {}",
                        snapshot.row_count
                    ))));
                }
                self.read_state
                    .route_telemetry
                    .record_route_device_lookup_micros(&table.name, lookup_micros, 1);
                resident_count_to_i64(row_count)?
            }
            ResidentPredicate::Int4Equal { col, needle } => {
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                let null_bitmap_offset = resident_device_null_column_offset(snapshot, table, col)?;
                let row_count = resident_snapshot_row_count(snapshot)?;
                let matched = device_memory
                    .count_i32_equal_from_payload(byte_offset, row_count, needle, null_bitmap_offset)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                resident_count_to_i64(matched)?
            }
            ResidentPredicate::Int4Compare {
                col,
                needle,
                comparison,
            } => {
                let byte_offset = resident_device_int4_column_offset(snapshot, table, col)?;
                let row_count = resident_snapshot_row_count(snapshot)?;
                let matched = device_memory
                    .count_i32_compare_from_payload(byte_offset, row_count, needle, comparison)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                resident_count_to_i64(matched)?
            }
            ResidentPredicate::Int4Between { .. } => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident COUNT(*) compiler does not emit a BETWEEN predicate".to_string(),
                )));
            }
        };

        Ok(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns),
            rows: (vec![vec![SqlValue::Int8(count)]]).into(),
            planned_target: DeviceTarget::Gpu(snapshot_gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot_gpu_id),
            fallback_reason: None,
            access_path: Arc::new(access_path),
        })
    }

    /// Physical path for the resident scalar `SUM`/`AVG`/`MIN`/`MAX` family: resolve the aggregate
    /// column's payload offset, dispatch the matching stats kernel by predicate, and materialize the
    /// single-row result. Unfiltered SUM uses the header sum kernel; unfiltered AVG/MIN/MAX reduce
    /// the self-grouped stats kernel; filtered/BETWEEN forms read a `CudaI32Stats` from the payload.
    fn run_resident_scalar_aggregate(
        &self,
        table: &RelationalTable,
        bound: BoundRelationalSelect,
        snapshot: &RelationalResidencySnapshot,
        predicate: &ResidentPredicate,
        aggregate: ResidentScalarAggregate,
        access_path: RelationalAccessPath,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let gpu_id = snapshot.gpu_id;
        let agg_col = match aggregate {
            ResidentScalarAggregate::Sum { col }
            | ResidentScalarAggregate::Avg { col }
            | ResidentScalarAggregate::Min { col }
            | ResidentScalarAggregate::Max { col } => col,
            ResidentScalarAggregate::Count => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident COUNT routed to the scalar SUM/AVG/MIN/MAX path".to_string(),
                )));
            }
        };
        let row_count = resident_snapshot_row_count(snapshot)?;

        // M3 (doc 21) — NULL 3VL: SUM/AVG/MIN/MAX skip NULL values. `Some` = the aggregate column has a
        // GPU-resident validity bitmap (it holds ≥1 NULL); `None` = no NULL ⇒ the byte-identical fast
        // path below. A NULL value's payload bytes are a 0 placeholder, so an unaware kernel would fold
        // a phantom 0 into MIN/AVG/SUM — the bitmap is read ON-DEVICE to exclude those rows.
        let agg_null_offset = resident_device_null_column_offset(snapshot, table, agg_col)?;

        // MAX over a provably-empty int4 compare domain: the retained column stats already prove the
        // result is empty, so answer from them with no kernel launch (matches the pre-unification
        // filtered probe's fast path; only the D2H of the i64 result counter is charged). Non-nullable
        // only — a nullable column routes through the NULL-aware filtered kernel below (which finalizes
        // an empty result to SQL NULL, not the empty-text sentinel).
        if let (
            ResidentScalarAggregate::Max { col },
            ResidentPredicate::Int4Compare {
                needle, comparison, ..
            },
        ) = (aggregate, predicate)
        {
            if agg_null_offset.is_none()
                && resident_device_int4_column_stats(snapshot, table, col)
                    .is_some_and(|stats| resident_i32_comparison_domain_is_empty(stats, *needle, *comparison))
            {
                self.metrics
                    .observe_d2h_bytes(std::mem::size_of::<i64>() as u64);
                return Ok(self.resident_scalar_result(
                    bound,
                    SqlValue::Text(String::new()),
                    gpu_id,
                    access_path,
                ));
            }
        }

        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        // The stats kernels scan one column for both the filter and the aggregate; the compiler
        // guarantees a filter targets the aggregate column, so the scanned column is the predicate's
        // (when filtered) or the aggregate's (unfiltered) — identical by construction.
        let scan_col = match predicate {
            ResidentPredicate::Int4Compare { col, .. }
            | ResidentPredicate::Int4Between { col, .. } => *col,
            ResidentPredicate::All | ResidentPredicate::Int4Equal { .. } => agg_col,
        };
        let byte_offset = resident_device_int4_column_offset(snapshot, table, scan_col)?;

        match predicate {
            // Non-nullable fast path: byte-identical to before M3 (no validity bitmap to read).
            ResidentPredicate::All if agg_null_offset.is_none() => match aggregate {
                ResidentScalarAggregate::Sum { .. } => {
                    let started = Instant::now();
                    let sum = device_memory
                        .sum_i32_from_payload(byte_offset, row_count)
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    let elapsed = started.elapsed();
                    self.metrics
                        .observe_d2h_bytes(std::mem::size_of::<i64>() as u64);
                    self.metrics.observe_kernel_exec_ms(
                        elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1),
                    );
                    Ok(self.resident_scalar_result(bound, SqlValue::Int8(sum), gpu_id, access_path))
                }
                ResidentScalarAggregate::Avg { .. }
                | ResidentScalarAggregate::Min { .. }
                | ResidentScalarAggregate::Max { .. } => {
                    // Unfiltered AVG/MIN/MAX reduce the self-grouped stats kernel (group == value
                    // column), so the D2H scales with the distinct-value count it copies back.
                    //
                    // Aggregate-selection mask: this scalar reduction reads only the field(s) it needs,
                    // so request only those on-device (the kernel runs strictly fewer per-row update
                    // atomics; the masked-out fields stay at their init sentinels and are never read).
                    // AVG reads total_count + total_sum (COUNT|SUM); MIN reads group.min (MIN); MAX reads
                    // group.max (MAX). Byte-identical to ALL for the field(s) actually consumed below.
                    let agg_mask = match aggregate {
                        ResidentScalarAggregate::Avg { .. } => {
                            grouped_agg_mask::COUNT | grouped_agg_mask::SUM
                        }
                        ResidentScalarAggregate::Min { .. } => grouped_agg_mask::MIN,
                        ResidentScalarAggregate::Max { .. } => grouped_agg_mask::MAX,
                        _ => unreachable!("matched AVG/MIN/MAX above"),
                    };
                    let started = Instant::now();
                    let grouped_stats = device_memory
                        .grouped_stats_i32_from_payload(
                            byte_offset,
                            byte_offset,
                            row_count,
                            agg_mask,
                        )
                        .map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    let elapsed = started.elapsed();
                    let copied_group_count = grouped_stats.len();
                    let total_count = grouped_stats.iter().map(|group| group.count).sum::<u64>();
                    let total_sum = grouped_stats.iter().map(|group| group.sum).sum::<i64>();
                    let result_value = match aggregate {
                        ResidentScalarAggregate::Avg { .. } => average_sql_value(
                            i128::from(total_sum),
                            usize::try_from(total_count).map_err(|_| {
                                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                    "resident device-memory scalar aggregate count {total_count} exceeds AVG result range"
                                )))
                            })?,
                        ),
                        ResidentScalarAggregate::Min { .. } => grouped_stats
                            .iter()
                            .map(|group| group.min)
                            .min()
                            .map(SqlValue::Int4)
                            .unwrap_or_else(|| SqlValue::Text(String::new())),
                        ResidentScalarAggregate::Max { .. } => grouped_stats
                            .iter()
                            .map(|group| group.max)
                            .max()
                            .map(SqlValue::Int4)
                            .unwrap_or_else(|| SqlValue::Text(String::new())),
                        _ => unreachable!("matched AVG/MIN/MAX above"),
                    };
                    let result_d2h_bytes = copied_group_count
                        .checked_mul(
                            std::mem::size_of::<i32>()
                                + std::mem::size_of::<u64>()
                                + std::mem::size_of::<i64>()
                                + (2 * std::mem::size_of::<i32>()),
                        )
                        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>()))
                        .and_then(|bytes| u64::try_from(bytes).ok())
                        .unwrap_or(u64::MAX);
                    self.metrics.observe_d2h_bytes(result_d2h_bytes);
                    self.metrics.observe_kernel_exec_ms(
                        elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1),
                    );
                    Ok(self.resident_scalar_result(bound, result_value, gpu_id, access_path))
                }
                ResidentScalarAggregate::Count => Err(ExecuteError::Engine(
                    EngineError::ApplyFailed("resident COUNT routed to scalar path".to_string()),
                )),
            },
            // M3 NULL 3VL path (the aggregate column holds ≥1 NULL): route ALL of SUM/AVG/MIN/MAX through
            // the validity-bitmap-aware self-grouped stats kernel, which skips NULL rows ON-DEVICE. The
            // reduction is over only the non-NULL rows; when none survive (all-NULL column) the result is
            // SQL NULL for every aggregate (PG: SUM/AVG/MIN/MAX of no rows is NULL; COUNT is not on this
            // path). `agg_null_offset` is `Some` here by the match guard above.
            ResidentPredicate::All => {
                let started = Instant::now();
                let grouped_stats = device_memory
                    .grouped_stats_i32_nullable_from_payload(
                        byte_offset,
                        byte_offset,
                        row_count,
                        agg_null_offset,
                    )
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
                let elapsed = started.elapsed();
                let result_value = reduce_nullable_grouped_stats(aggregate, &grouped_stats)?;
                self.metrics
                    .observe_d2h_bytes(nullable_grouped_stats_d2h_bytes(grouped_stats.len()));
                self.metrics.observe_kernel_exec_ms(
                    elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1),
                );
                Ok(self.resident_scalar_result(bound, result_value, gpu_id, access_path))
            }
            // M3 filtered-nullable (compare): the filter `<value> <cmp> needle` runs on-device AND NULL
            // values are skipped, so the self-grouped stats cover only the surviving non-NULL matches;
            // zero survivors ⇒ SQL NULL. Reuses the already-NULL-aware grouped kernel (no new kernel).
            ResidentPredicate::Int4Compare {
                needle, comparison, ..
            } if agg_null_offset.is_some() => {
                let started = Instant::now();
                let grouped_stats = device_memory
                    .filtered_grouped_stats_i32_nullable_from_payload(
                        byte_offset,
                        row_count,
                        *needle,
                        *comparison,
                        agg_null_offset,
                    )
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
                let elapsed = started.elapsed();
                let result_value = reduce_nullable_grouped_stats(aggregate, &grouped_stats)?;
                self.metrics
                    .observe_d2h_bytes(nullable_grouped_stats_d2h_bytes(grouped_stats.len()));
                self.metrics.observe_kernel_exec_ms(
                    elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1),
                );
                Ok(self.resident_scalar_result(bound, result_value, gpu_id, access_path))
            }
            ResidentPredicate::Int4Compare {
                needle, comparison, ..
            } => {
                let started = Instant::now();
                let stats = device_memory
                    .filtered_stats_i32_compare_from_payload(
                        byte_offset,
                        row_count,
                        *needle,
                        *comparison,
                    )
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
                let elapsed = started.elapsed();
                let result_value = materialize_resident_scalar_stats(aggregate, &stats)?;
                let result_d2h_bytes = (std::mem::size_of::<u64>()
                    + std::mem::size_of::<i64>()
                    + (2 * std::mem::size_of::<i32>())
                    + std::mem::size_of::<u64>()) as u64;
                self.metrics.observe_d2h_bytes(result_d2h_bytes);
                self.metrics
                    .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));
                Ok(self.resident_scalar_result(bound, result_value, gpu_id, access_path))
            }
            // M3 filtered-nullable (BETWEEN): the NULL-aware between-stats kernel excludes NULL values
            // (a NULL never satisfies the range); a zero-count result ⇒ SQL NULL for every aggregate.
            ResidentPredicate::Int4Between { lower, upper, .. } if agg_null_offset.is_some() => {
                let (lower, upper) = (*lower, *upper);
                let started = Instant::now();
                let stats = device_memory
                    .stats_i32_between_nullable_from_payload(
                        byte_offset,
                        row_count,
                        lower,
                        upper,
                        agg_null_offset,
                    )
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
                let elapsed = started.elapsed();
                let result_value = if stats.count == 0 {
                    SqlValue::Null
                } else {
                    materialize_resident_scalar_stats(aggregate, &stats)?
                };
                let result_d2h_bytes = if lower > upper {
                    0
                } else {
                    (std::mem::size_of::<u64>()
                        + std::mem::size_of::<i64>()
                        + (2 * std::mem::size_of::<i32>())
                        + std::mem::size_of::<u64>()) as u64
                };
                self.metrics.observe_d2h_bytes(result_d2h_bytes);
                if lower <= upper {
                    self.metrics.observe_kernel_exec_ms(
                        elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1),
                    );
                }
                Ok(self.resident_scalar_result(bound, result_value, gpu_id, access_path))
            }
            ResidentPredicate::Int4Between { lower, upper, .. } => {
                let (lower, upper) = (*lower, *upper);
                let started = Instant::now();
                let stats = device_memory
                    .stats_i32_between_from_payload(byte_offset, row_count, lower, upper)
                    .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
                let elapsed = started.elapsed();
                let result_value = materialize_resident_scalar_stats(aggregate, &stats)?;
                let result_d2h_bytes = if lower > upper {
                    0
                } else {
                    (std::mem::size_of::<u64>()
                        + std::mem::size_of::<i64>()
                        + (2 * std::mem::size_of::<i32>())
                        + std::mem::size_of::<u64>()) as u64
                };
                self.metrics.observe_d2h_bytes(result_d2h_bytes);
                if lower <= upper {
                    self.metrics.observe_kernel_exec_ms(
                        elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1),
                    );
                }
                Ok(self.resident_scalar_result(bound, result_value, gpu_id, access_path))
            }
            ResidentPredicate::Int4Equal { .. } => Err(ExecuteError::Engine(
                EngineError::ApplyFailed(
                    "resident scalar aggregate compiler does not emit an equality predicate"
                        .to_string(),
                ),
            )),
        }
    }

    /// Build the single-row result a resident scalar aggregate returns (one column, one value, on
    /// the snapshot's GPU).
    fn resident_scalar_result(
        &self,
        bound: BoundRelationalSelect,
        value: SqlValue,
        gpu_id: u16,
        access_path: RelationalAccessPath,
    ) -> RelationalSelectResult {
        RelationalSelectResult {
            columns: Arc::new(bound.selected_columns),
            rows: (vec![vec![value]]).into(),
            planned_target: DeviceTarget::Gpu(gpu_id),
            executed_target: DeviceTarget::Gpu(gpu_id),
            fallback_reason: None,
            access_path: Arc::new(access_path),
        }
    }

    pub fn execute_relational_membership_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || !select.order_by.is_empty()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.len() < 2
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory membership count proof currently supports only SELECT COUNT(*) with one int4 IN membership predicate"
                    .to_string(),
            )));
        }

        let mut filter_idx = None;
        let mut needles = BTreeSet::new();
        for group in &bound.filter_groups {
            if group.len() != 1 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory membership count proof currently supports only one int4 IN membership predicate"
                        .to_string(),
                )));
            }
            let (idx, op, value) = group[0].clone();
            if op != SelectFilterOp::Eq {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory membership count proof currently supports only int4 equality membership predicates"
                        .to_string(),
                )));
            }
            if filter_idx
                .replace(idx)
                .is_some_and(|existing| existing != idx)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory membership count proof requires all membership values to target the same column"
                        .to_string(),
                )));
            }
            let SqlValue::Int4(needle) = value else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory membership count proof currently supports only int4 membership literals"
                        .to_string(),
                )));
            };
            needles.insert(needle);
        }
        let filter_idx = filter_idx.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory membership count proof requires at least one membership literal"
                    .to_string(),
            ))
        })?;
        if table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory membership count proof currently supports only int4 membership predicates"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
        if snapshot.schema != table.schema || snapshot.table != table.name {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot no longer matches catalog table identity".to_string(),
            )));
        }
        if !snapshot.is_valid() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" resident snapshot is invalid",
                table.name
            ))));
        }
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let null_bitmap_offset = resident_device_null_column_offset(&snapshot, &table, filter_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let needles = needles.into_iter().collect::<Vec<_>>();
        let membership_count = device_memory
            .count_i32_in_from_payload(byte_offset, row_count, &needles, null_bitmap_offset)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let count = i64::try_from(membership_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory membership count {membership_count} exceeds supported COUNT(*) result range"
            )))
        })?;

        Ok(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns),
            rows: (vec![vec![SqlValue::Int8(count)]]).into(),
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path: Arc::new(access_path),
        })
    }

    pub fn execute_relational_between_count_with_resident_device_memory_probe(
        &self,
        select: &Select,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        if select.distinct
            || !matches!(select.projection, SelectProjection::CountAll)
            || select.group_by.is_some()
            || !select.having_groups.is_empty()
            || !select.order_by.is_empty()
            || select.limit.is_some()
            || select.offset.is_some()
            || bound.filter_groups.len() != 1
            || bound.filter_groups[0].len() != 2
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof currently supports only SELECT COUNT(*) with one int4 BETWEEN predicate"
                    .to_string(),
            )));
        }

        let mut filter_idx = None;
        let mut lower = None;
        let mut upper = None;
        for (idx, op, value) in bound.filter_groups[0].iter().cloned() {
            if filter_idx
                .replace(idx)
                .is_some_and(|existing| existing != idx)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory BETWEEN count proof requires both range bounds to target the same column"
                        .to_string(),
                )));
            }
            let SqlValue::Int4(value) = value else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory BETWEEN count proof currently supports only int4 bounds"
                        .to_string(),
                )));
            };
            match op {
                SelectFilterOp::Gte => lower = Some(value),
                SelectFilterOp::Lte => upper = Some(value),
                _ => {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident device-memory BETWEEN count proof currently supports only inclusive int4 bounds"
                            .to_string(),
                    )));
                }
            }
        }
        let filter_idx = filter_idx.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof requires an int4 predicate column"
                    .to_string(),
            ))
        })?;
        let lower = lower.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof requires a lower inclusive bound"
                    .to_string(),
            ))
        })?;
        let upper = upper.ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof requires an upper inclusive bound"
                    .to_string(),
            ))
        })?;
        if table.columns[filter_idx].ty != SqlType::Int4 {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident device-memory BETWEEN count proof currently supports only int4 predicates"
                    .to_string(),
            )));
        }

        let (_query, access_path) =
            self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
        let snapshot = self
            .relational_residency_snapshot(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
        if snapshot.schema != table.schema || snapshot.table != table.name {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot no longer matches catalog table identity".to_string(),
            )));
        }
        if !snapshot.is_valid() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" resident snapshot is invalid",
                table.name
            ))));
        }
        let device_memory = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no retained resident device memory",
                    table.name
                )))
            })?;
        let byte_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let started = Instant::now();
        let between_count = device_memory
            .count_i32_between_from_payload(byte_offset, row_count, lower, upper)
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))?;
        let elapsed = started.elapsed();
        self.metrics
            .observe_d2h_bytes(2 * std::mem::size_of::<u64>() as u64);
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));
        self.metrics
            .observe_kernel_exec_ms(elapsed.as_millis().try_into().unwrap_or(u64::MAX).max(1));
        let count = i64::try_from(between_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "resident device-memory BETWEEN count {between_count} exceeds supported COUNT(*) result range"
            )))
        })?;

        Ok(RelationalSelectResult {
            columns: Arc::new(bound.selected_columns),
            rows: (vec![vec![SqlValue::Int8(count)]]).into(),
            planned_target: DeviceTarget::Gpu(snapshot.gpu_id),
            executed_target: DeviceTarget::Gpu(snapshot.gpu_id),
            fallback_reason: None,
            access_path: Arc::new(access_path),
        })
    }

    pub fn execute_relational_equality_multi_column_projection_batch_with_resident_device_memory_probe(
        &self,
        selects: &[Select],
    ) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        self.execute_relational_equality_multi_column_projection_batch_inner(selects, None, true)
    }

    pub(crate) fn execute_relational_equality_multi_column_projection_batch_inner(
        &self,
        selects: &[Select],
        planned_jobs: Option<&[RelationalRetainedReadJob]>,
        record_route_observation: bool,
    ) -> Result<Vec<RelationalSelectResult>, ExecuteError> {
        if selects.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(jobs) = planned_jobs {
            if jobs.len() != selects.len() {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "retained read job count {} does not match SELECT count {}",
                    jobs.len(),
                    selects.len()
                ))));
            }
        }
        let mut members = Vec::with_capacity(selects.len());
        let mut batch_table: Option<RelationalTable> = None;
        let mut batch_filter_idx: Option<usize> = None;
        let mut batch_selected_indexes: Option<Vec<usize>> = None;
        for (select_idx, select) in selects.iter().enumerate() {
            let query_shape = if let Some(jobs) = planned_jobs {
                jobs[select_idx]
                    .route_id
                    .split(':')
                    .next()
                    .unwrap_or("unknown")
                    .to_string()
            } else {
                let decision = self.plan_relational_resident_route(select);
                if !decision.accepted {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "resident device-memory equality batch currently supports only accepted int4 equality projection, got {}: {}",
                        decision.query_shape, decision.reason
                    ))));
                }
                decision.query_shape
            };
            if !matches!(
                query_shape.as_str(),
                "int4_equality_projection"
                    | "int4_equality_multi_column_projection"
                    | "int4_equality_mixed_column_projection"
            ) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "resident device-memory equality batch currently supports only accepted int4 equality projection, got {}: {}",
                    query_shape, "preplanned retained read job"
                ))));
            }
            let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
            let filter_groups = if !bound.filter_groups.is_empty() {
                bound.filter_groups.clone()
            } else if !bound.filters.is_empty() {
                vec![bound.filters.clone()]
            } else if let Some(filter) = bound.filter.clone() {
                vec![vec![filter]]
            } else {
                Vec::new()
            };
            if select.distinct
                || select.group_by.is_some()
                || !select.having_groups.is_empty()
                || !select.order_by.is_empty()
                || select.limit.is_some()
                || select.offset.is_some()
                || bound.selected_indexes.is_empty()
                || !bound
                    .selected_indexes
                    .iter()
                    .all(|idx| matches!(table.columns[*idx].ty, SqlType::Int4 | SqlType::Text))
                || filter_groups.len() != 1
                || filter_groups[0].len() != 1
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch currently supports SELECT one_or_more_int4_or_text_columns with one int4 equality predicate"
                        .to_string(),
                )));
            }
            let (filter_idx, op, value) = filter_groups[0][0].clone();
            let SqlValue::Int4(needle) = value else {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch currently supports only int4 equality predicates"
                        .to_string(),
                )));
            };
            if op != SelectFilterOp::Eq || table.columns[filter_idx].ty != SqlType::Int4 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch currently supports only int4 equality predicates"
                        .to_string(),
                )));
            }
            if let Some(existing) = &batch_table {
                if existing.name != table.name
                    || existing.schema != table.schema
                    || existing.columns != table.columns
                {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "resident device-memory equality batch cannot mix tables".to_string(),
                    )));
                }
            } else {
                batch_table = Some(table.clone());
            }
            if batch_filter_idx.is_some_and(|existing| existing != filter_idx) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch cannot mix predicate columns"
                        .to_string(),
                )));
            }
            batch_filter_idx = Some(filter_idx);
            if batch_selected_indexes
                .as_ref()
                .is_some_and(|existing| existing != &bound.selected_indexes)
            {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch cannot mix projection columns"
                        .to_string(),
                )));
            }
            batch_selected_indexes = Some(bound.selected_indexes.clone());
            let (_query, access_path) =
                self.relational_select_mvcc_query_pinned(select, &table, &bound, copin_s)?;
            members.push((bound, access_path, needle));
        }

        let table = batch_table.expect("non-empty batch has table");
        let filter_idx = batch_filter_idx.expect("non-empty batch has filter");
        let selected_indexes = batch_selected_indexes.expect("non-empty batch has projections");
        let snapshot = self
            .relational_residency_snapshot_ref(&table.name)
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no resident snapshot",
                    table.name
                )))
            })?;
        if snapshot.schema != table.schema || snapshot.table != table.name {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot no longer matches catalog table identity".to_string(),
            )));
        }
        if !snapshot.is_valid() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" resident snapshot is invalid",
                table.name
            ))));
        }
        let snapshot_gpu_id = snapshot.gpu_id;
        let row_count = u64::try_from(snapshot.row_count).map_err(|_| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "resident snapshot row count exceeds retained device-memory proof range"
                    .to_string(),
            ))
        })?;
        let filter_offset = resident_device_int4_column_offset(&snapshot, &table, filter_idx)?;
        let all_int4_projection = selected_indexes
            .iter()
            .all(|idx| table.columns[*idx].ty == SqlType::Int4);
        let projection_offsets = if all_int4_projection {
            selected_indexes
                .iter()
                .map(|idx| resident_device_int4_column_offset(&snapshot, &table, *idx))
                .collect::<Result<Vec<_>, ExecuteError>>()?
        } else {
            vec![filter_offset]
        };
        let needles = members
            .iter()
            .map(|(_bound, _access_path, needle)| *needle)
            .collect::<Vec<_>>();

        if let Some(device_memory) = self.read_state.residency.device_memory.get(&table.name) {
            device_memory.clear_last_kernel_event_elapsed_us();
        }
        let text_projection_indexes = selected_indexes
            .iter()
            .copied()
            .filter(|idx| table.columns[*idx].ty == SqlType::Text)
            .collect::<Vec<_>>();
        let compact_text_projection_idx = (!all_int4_projection
            && text_projection_indexes.len() == 1)
            .then(|| text_projection_indexes[0]);
        let int4_projection_indexes = selected_indexes
            .iter()
            .copied()
            .filter(|idx| table.columns[*idx].ty == SqlType::Int4)
            .collect::<Vec<_>>();
        let int4_projection_offsets = int4_projection_indexes
            .iter()
            .map(|idx| resident_device_int4_column_offset(&snapshot, &table, *idx))
            .collect::<Result<Vec<_>, ExecuteError>>()?;
        let before_metrics = self.metrics.snapshot();
        let batch_started = Instant::now();
        let compact_text_rows = if let Some(text_idx) = compact_text_projection_idx {
            let device_memory = self
                .read_state
                .residency
                .device_memory
                .get(&table.name)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" has no retained resident device memory",
                        table.name
                    )))
                })?;
            let layout = resident_device_text_column_layout(&snapshot, &table, text_idx)?;
            Some(
                device_memory
                    .match_project_i32_equal_any_text_from_payload(
                        filter_offset,
                        &needles,
                        &int4_projection_offsets,
                        layout.offsets_byte_offset,
                        layout.bytes_byte_offset,
                        layout.bytes_len,
                        row_count,
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?,
            )
        } else {
            None
        };
        let projected_rows = if compact_text_rows.is_none() {
            let device_memory = self
                .read_state
                .residency
                .device_memory
                .get(&table.name)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" has no retained resident device memory",
                        table.name
                    )))
                })?;
            Some(
                device_memory
                    .match_project_i32_equal_any_from_payload(
                        filter_offset,
                        &needles,
                        &projection_offsets,
                        row_count,
                    )
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?,
            )
        } else {
            None
        };
        let batch_micros = batch_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let materialize_started = Instant::now();
        // Stable-order fix (Thread-3 Stage 4): every branch below scatters matched rows into
        // per-needle slices in the kernel's `atom.global.add` SCHEDULE order, which is
        // non-deterministic for >32 matches (multi-warp). Tag each row with its kernel `row_index`
        // and sort each needle's slice ASCENDING by it (after the branch), so the output is
        // deterministic and byte-identical to the per-query ascending order (the `row_indices`
        // order class established by `4b750a94`). The single-element delegation from the per-query
        // mixed/multi-column path flows through here too, so the per-query and batched paths share
        // this one sorted assembly and stay byte-identical by construction.
        let rows_by_select: Vec<Vec<(u64, Vec<SqlValue>)>> = if all_int4_projection {
            let mut rows_by_select = vec![Vec::new(); members.len()];
            let projected_rows = projected_rows.as_ref().ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch missing int4 projection rows"
                        .to_string(),
                ))
            })?;
            for projected in projected_rows {
                rows_by_select[projected.needle_index].push((
                    projected.row_index,
                    projected
                        .values
                        .iter()
                        .copied()
                        .map(SqlValue::Int4)
                        .collect::<Vec<_>>(),
                ));
            }
            rows_by_select
        } else if let (Some(text_idx), Some(compact_rows)) =
            (compact_text_projection_idx, compact_text_rows.as_ref())
        {
            let int4_positions = int4_projection_indexes
                .iter()
                .copied()
                .enumerate()
                .map(|(position, idx)| (idx, position))
                .collect::<BTreeMap<_, _>>();
            let mut rows_by_select = vec![Vec::new(); members.len()];
            for projected in compact_rows {
                let row = selected_indexes
                    .iter()
                    .map(|idx| {
                        if *idx == text_idx {
                            return Ok(SqlValue::Text(projected.text.clone()));
                        }
                        if let Some(position) = int4_positions.get(idx) {
                            return Ok(SqlValue::Int4(projected.values[*position]));
                        }
                        Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "resident device-memory equality batch compact text projection missing projected column"
                                .to_string(),
                        )))
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
                rows_by_select[projected.needle_index].push((projected.row_index, row));
            }
            rows_by_select
        } else {
            let projected_rows = projected_rows.as_ref().ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "resident device-memory equality batch missing mixed projection rows"
                        .to_string(),
                ))
            })?;
            let matched_row_indices = projected_rows
                .iter()
                .map(|projected| projected.row_index)
                .collect::<Vec<_>>();
            let device_memory = self
                .read_state
                .residency
                .device_memory
                .get(&table.name)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" has no retained resident device memory",
                        table.name
                    )))
                })?;
            let mut int4_values = BTreeMap::new();
            let mut text_values = BTreeMap::new();
            for idx in &selected_indexes {
                match table.columns[*idx].ty {
                    SqlType::Int4 => {
                        let byte_offset =
                            resident_device_int4_column_offset(&snapshot, &table, *idx)?;
                        let values = device_memory
                            .project_i32_rows_from_payload(byte_offset, &matched_row_indices)
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?;
                        if values.len() != matched_row_indices.len() {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "resident device-memory equality batch mixed projection int4 column returned {} rows, expected {}",
                                values.len(),
                                matched_row_indices.len()
                            ))));
                        }
                        int4_values.insert(*idx, values);
                    }
                    SqlType::Text => {
                        let layout = resident_device_text_column_layout(&snapshot, &table, *idx)?;
                        let values = device_memory
                            .project_text_rows_from_payload(
                                layout.offsets_byte_offset,
                                layout.bytes_byte_offset,
                                layout.bytes_len,
                                &matched_row_indices,
                            )
                            .map_err(|err| {
                                ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                            })?;
                        if values.len() != matched_row_indices.len() {
                            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "resident device-memory equality batch mixed projection text column returned {} rows, expected {}",
                                values.len(),
                                matched_row_indices.len()
                            ))));
                        }
                        text_values.insert(*idx, values);
                    }
                    // See the multi-column route above: typed columns take the CPU path; this
                    // GPU projection only handles int4/text.
                    SqlType::Int2
                | SqlType::Int8
                | SqlType::Numeric { .. }
                | SqlType::Bool
                | SqlType::Date
                | SqlType::Timestamp
                | SqlType::Uuid => {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "resident device-memory projection supports only int4/text columns"
                                .to_string(),
                        )));
                    }
                }
            }
            let mut rows_by_select = vec![Vec::new(); members.len()];
            for (projected_idx, projected) in projected_rows.iter().enumerate() {
                let row = selected_indexes
                    .iter()
                    .map(|idx| {
                        if let Some(values) = int4_values.get(idx) {
                            return Ok(SqlValue::Int4(values[projected_idx]));
                        }
                        if let Some(values) = text_values.get(idx) {
                            return Ok(SqlValue::Text(values[projected_idx].clone()));
                        }
                        Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "resident device-memory equality batch mixed projection missing projected column"
                                .to_string(),
                        )))
                    })
                    .collect::<Result<Vec<_>, ExecuteError>>()?;
                rows_by_select[projected.needle_index].push((projected.row_index, row));
            }
            rows_by_select
        };
        // Apply the ascending-by-`row_index` order to every needle's slice (see the stable-order
        // note above), then strip the index tag back to the materialized rows.
        let rows_by_select: Vec<Vec<Vec<SqlValue>>> = rows_by_select
            .into_iter()
            .map(|mut slice| {
                slice.sort_by_key(|(row_index, _)| *row_index);
                slice.into_iter().map(|(_, row)| row).collect()
            })
            .collect();
        let materialization_micros = materialize_started
            .elapsed()
            .as_micros()
            .try_into()
            .unwrap_or(u64::MAX);
        let total_rows = rows_by_select.iter().map(Vec::len).sum::<usize>();
        let int4_result_columns = selected_indexes
            .iter()
            .filter(|idx| table.columns[**idx].ty == SqlType::Int4)
            .count();
        let text_result_bytes = if all_int4_projection {
            0
        } else {
            rows_by_select
                .iter()
                .flatten()
                .flat_map(|row| row.iter())
                .filter_map(|value| match value {
                    SqlValue::Text(value) => Some(u64::try_from(value.len()).unwrap_or(u64::MAX)),
                    _ => None,
                })
                .fold(0_u64, u64::saturating_add)
                .saturating_add(
                    u64::try_from(total_rows)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(2 * std::mem::size_of::<u64>() as u64),
                )
        };
        let row_metadata_d2h_bytes = u64::try_from(total_rows)
            .unwrap_or(u64::MAX)
            .saturating_mul((std::mem::size_of::<u32>() + std::mem::size_of::<u64>()) as u64)
            .saturating_add(std::mem::size_of::<u32>() as u64);
        let result_d2h_bytes = u64::try_from(total_rows)
            .unwrap_or(u64::MAX)
            .saturating_mul(
                u64::try_from(int4_result_columns)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(std::mem::size_of::<i32>() as u64),
            )
            .saturating_add(text_result_bytes)
            .saturating_add(row_metadata_d2h_bytes);
        self.metrics.observe_d2h_bytes(result_d2h_bytes);
        let kernel_samples = if all_int4_projection {
            1
        } else {
            selected_indexes.len().saturating_add(1)
        };
        for _ in 0..kernel_samples {
            self.metrics
                .observe_kernel_exec_ms(batch_micros.div_ceil(1000).max(1));
        }
        let kernel_event_elapsed_us = self
            .read_state
            .residency
            .device_memory
            .get(&table.name)
            .and_then(|device_memory| device_memory.last_kernel_event_elapsed_us());
        if let Some(elapsed_us) = kernel_event_elapsed_us {
            self.metrics.observe_kernel_event_elapsed_us(elapsed_us);
        }
        self.read_state
            .route_telemetry
            .record_route_selected_projection_micros(
                &table.name,
                batch_micros,
                batch_micros,
                materialization_micros,
                total_rows,
            );
        // The route-execution observation is recorded once per route execution. When the
        // single-predicate mixed int4+text dispatcher path delegates here (a 1-element slice),
        // that caller's `execute_relational_select_with_resident_route` already records the
        // observation for the whole route, so the delegated call suppresses its own to avoid a
        // double-count (telemetry-only; results are unaffected). The standalone batch/submit
        // callers are not wrapped by the dispatcher and own the record themselves.
        if record_route_observation {
            let after_metrics = self.metrics.snapshot();
            self.read_state
                .route_telemetry
                .record_route_execution_observation(
                    &table.name,
                    RelationalResidentRouteExecutionObservation {
                        h2d_bytes: after_metrics
                            .h2d_bytes_total
                            .saturating_sub(before_metrics.h2d_bytes_total),
                        d2h_bytes: after_metrics
                            .d2h_bytes_total
                            .saturating_sub(before_metrics.d2h_bytes_total),
                        kernel_samples: after_metrics
                            .kernel_exec_samples
                            .saturating_sub(before_metrics.kernel_exec_samples),
                        kernel_ms: after_metrics
                            .kernel_exec_total_ms
                            .saturating_sub(before_metrics.kernel_exec_total_ms),
                        kernel_event_elapsed_us,
                        rows: total_rows,
                        wall_micros: batch_started
                            .elapsed()
                            .as_micros()
                            .try_into()
                            .unwrap_or(u64::MAX),
                    },
                );
        }

        Ok(members
            .into_iter()
            .zip(rows_by_select)
            .map(
                |((bound, access_path, _needle), rows)| RelationalSelectResult {
                    columns: Arc::new(bound.selected_columns),
                    rows: rows.into(),
                    planned_target: DeviceTarget::Gpu(snapshot_gpu_id),
                    executed_target: DeviceTarget::Gpu(snapshot_gpu_id),
                    fallback_reason: None,
                    access_path: Arc::new(access_path),
                },
            )
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn execute_relational_select_with_backend<B: MvccExecutionBackend>(
        &mut self,
        select: &Select,
        backend: &B,
    ) -> Result<RelationalSelectResult, ExecuteError> {
        let (table, bound, copin_s) = self.bind_relational_select_for_execution(select)?;
        let pin = self.pin_relational_read_at(&select.table, copin_s);
        let (query, access_path) =
            self.relational_select_mvcc_query(select, &table, &bound, &pin)?;
        let result = self.execute_mvcc_query_with_fallback_reason(
            pin.store(),
            &query,
            backend,
            None,
            false,
        )?;
        self.finalize_relational_select(select, table, bound, access_path, result)
    }

    /// Test-only: total number of route-execution telemetry observations recorded so far.
    /// Used to assert that a route records its observation exactly once per execution.
    #[cfg(test)]
    pub(crate) fn route_execution_observation_count(&self) -> u64 {
        self.read_state
            .route_telemetry
            .route_execution_observation_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
