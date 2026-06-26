# Handover — S10c (partitioned read path) COMPLETE for int4; S10d is next

**Date:** 2026-06-26. **Branch:** `phase0-m1-engine-facade` (merged to `main` via fast-forward;
`origin/main == ea93fc10`). **Suite:** 729/0 serial (`cargo test -p gpu_db_engine --lib -- --include-ignored
--test-threads=1`). Campaign tracker: `docs/architecture/22-full-gpu-native-read-path.md` §4.

## What landed (this session) — all in `main`, each slice independently AUDITED SHIP
Doc-22 campaign "host out of the data path." 8 commits (`098a870f`..`ea93fc10`):
- **`32a3995b`/`098a870f` — S10a `text_prefix_like_count`** AUDITED SHIP (closeout of a prior stopped session;
  on-device LIKE via `TextLikeMask`). Last non-partitioned S10a probe retired.
- **`96acf025` — S10c slice 0:** inject `ResidentExecSource` / `Option<&ResidentExecSource>` into
  `execute_resident_expr_select_with_binding` (engine_expr.rs). `None` = single store (byte-identical); `Some`
  = inject a buffer. Behavior-preserving foundation.
- **`8da01cb6`/`1592d4a0` — S10c slice 1:** route the 8 `partitioned_*` shapes per-partition through the
  bridge + **retire all 8 `*_partitioned_*_with_resident_device_memory_probe` methods** (+ dead
  `resident_partition_int4_column_offset`). **S10d UNBLOCKED.**
- **`f786a9f1`/`ebeffeb7` — S10c slice 2a:** on-device recompaction. New `cuMemcpyDtoD` primitive +
  `retain_device_memory_recompacted` (crates/execution/src/lib.rs — **no PTX kernel**, pure DtoD) build ONE
  unified int4 buffer; `execute_resident_partitioned_via_general` runs the executor ONCE over it. **Host fully
  out** (AVG is a single on-device quotient). HAZARD 3×serial+3×concurrent clean.
- **`99ef51f7`/`ea93fc10` — S10c slice 2b:** admit partitioned **DISTINCT / GROUP BY / ORDER BY / top-N** over
  the unified buffer (correct across partitions). Threaded `Option<&ResidentExecSource>` through
  `execute_resident_grouped_via_general` / `_distinct_via_general`. Oracle fixtures (`s10c_2b_*` in
  tests/resident_route.rs) + independent cross-partition row-pins.

**Net:** the single-GPU **int4** partitioned read path is complete — probes retired, host fully out, every
shape (scalar agg + projection + DISTINCT/GROUP BY/ORDER BY) on-device.

## Next work, prioritized

### 1. S10d — delete the host finalization + CPU fallback (the campaign's TERMINAL slice; now unblocked)
This is the highest-value next step: S10c retired the last partitioned probes, so per doc-22 §4 S10d can run.
Delete (see doc-22 §4 "S10d" for the exact scoping, written by an earlier session):
- `engine_select_bind.rs` `finalize_relational_select` (host sort/agg/DISTINCT/HAVING/LIMIT),
- `mvcc_read_exec.rs` `cpu_fallback`, the host `sort_by`/`mvcc_row_cmp`,
- the `FirstCudaSliceParityBackend` harness + its ~18 parity tests, the `sql_catalog` cuda-probe cache test,
  and `execute_relational_select_cpu_pinned` + its `concurrency.rs` seam test.
**First step:** grep for every remaining caller of `finalize_relational_select` / `cpu_fallback` and confirm
each read shape now has an on-device route (the resident-route dispatch in engine_select_exec.rs). Any shape
still falling back is a sub-slice to route first. Deletion is LAST (gated on full coverage). Independent audit.

### 2. Text-column recompaction (the one remaining S10c sub-item)
Today the int4-only recompaction means partitioned shapes over a TEXT column reject cleanly (the int4-only
route classifier guards them — no silent wrong data). To support them:
- Extend the recompaction to text: unlike int4 (pure `cuMemcpyDtoD` of fixed-width slices), text needs per-row
  OFFSET REBASING (each partition's `(n+1)` LE-i64 offsets are relative to its own bytes blob; the merged
  offsets must add the running byte total) + bytes-blob concat + 8-alignment recompute. This is NOT pure DtoD
  → either a host DtoH→add→HtoD round-trip on the (small) offsets section OR a tiny rebase kernel (→ HAZARD if
  a kernel). Files: `crates/execution/src/lib.rs` (`retain_device_memory_recompacted` / a sibling),
  `engine_residency.rs` `resident_snapshot_for_unified` (populate `resident_device_text_columns`),
  `engine_expr.rs` `execute_resident_partitioned_via_general` (text segments + the offset rebase), and remove
  the int4-only guard so text shapes route. Differential vs a single-store oracle (mirror the `s10c_2b_*`
  fixtures with a text column). Independent audit.

### 3. True multi-GPU cross-shard combine (deferred; larger)
The current recompaction assumes all partitions share one `gpu_id` (intra-context DtoD). For partitions on
DIFFERENT GPUs the recompaction needs cross-GPU transport (peer copy / NCCL — see
`docs/roadmap/no-nvidia-bootstrap-plan.md:97`). Gated on the multi-GPU transport decision. Note: multi-partition
residency is TEST-ONLY today (only `install_benchmark_relational_residency_owned_partitions`, from
`tests/resident_route.rs`); the production admission/spill producer that splits an over-VRAM table into
partitions does not exist yet (roadmap v2 — `prototype-to-production-plan.md:784,910`). Build the producer
alongside this.

## Discipline (charter — non-negotiable; see doc-22 §3)
- A slice is **GPU-native or it does not land.** Read ground truth before editing.
- Each slice: behavior-preserving; differential test **WITH NULL data** (the S10a/S10b lesson — the deleted
  probes were NULL-blind); kernel/device-touching slices run **HAZARD** (3× + concurrent, zero CUDA 700/716/717);
  each gets a **separate INDEPENDENT adversarial audit** — never self-audit; ADOPT findings, don't defer.
- **GOTCHA: the engine crate is fmt-DIRTY at HEAD (632 `cargo fmt --check` diffs).** NEVER run crate-wide
  `cargo fmt` inside a slice — it reflows ~all files and balloons a focused diff to 17 files / +4000 lines
  (this cost a false start). Hand-format additions; keep slice diffs to the intended files. (Cleaning the fmt
  debt is a deliberate standalone commit, not part of a feature slice — confirm with the user before doing it,
  since CI behavior on the dirt is unknown.)
- Test on the GPU (box has an RTX PRO 6000): `cargo test -p gpu_db_engine --lib -- --include-ignored
  --test-threads=1`. GPU tests are `#[ignore]` or guarded, so `--include-ignored` runs both.

## Key code pointers
- Partitioned bridge: `engine_expr.rs` `execute_resident_partitioned_via_general` (~1992).
- Injectable executor: `engine_expr.rs` `execute_resident_expr_select_with_binding` (`src: Option<&ResidentExecSource>`).
- Recompaction primitive: `crates/execution/src/lib.rs` `retain_device_memory_recompacted` /
  `launch_cuda_resident_device_memory_recompacted` + `RecompactSegment`.
- Unified descriptor: `engine_residency.rs` `resident_snapshot_for_unified` (~1085) /
  `resident_snapshot_for_partition` (~1043).
- Route classifier + partitioned planner: `resident_route.rs` (`resident_route_query_shape` and the
  `_distinct_/_grouped_/_ordered_` shape fns) + `engine_residency.rs` `plan_relational_partitioned_resident_route`
  (~1765, the shape mapping ~1805 + accept-list ~1900 + `required_int4_columns` ~2009).
- Dispatch: `engine_select_exec.rs` `execute_relational_select_with_resident_route` (~398).
- Memory: `~/.claude/.../memory/s10c-partitioned-bridge-plan.md` (full decision + slice log).
