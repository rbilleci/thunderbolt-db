# STRATA — GPU-Resident Shard Data Plane (Wire → GPU, Host as Control Plane)

**Status:** design (proposed). **Priority:** high. **Codename:** **STRATA** (the resident *strata* — horizontal
shards of a table's data laid down on the GPU). **Relationship:** STRATA is the end-to-end architecture that makes
the doc-22 campaign ("host out of the data path") *reachable in production*. doc-22 made the **resident** read path
GPU-native shape-by-shape; STRATA owns the question doc-22 never did — **how a committed table actually becomes a
GPU-resident, sharded, wire-served thing**, and how data flows wire → GPU and back with the host as control plane only.

**Why this exists (the gap STRATA closes).** Verified 2026-06-26:
1. **No production producer of GPU residency exists.** `residency.partitions` is written in exactly one place —
   `install_benchmark_relational_residency_owned_partitions` — whose only callers are in `tests/resident_route.rs`.
   The operator warm path (`warm_relational_residency_with_policy` → `populate_relational_residency_snapshot_on_gpu`)
   produces only a *unified single buffer*, and is operator-triggered (impl-log:1529 — "bounded operator-triggered
   workflow", autonomous scheduling out of scope). So in production a committed table is **non-resident by default**
   and reads run on the **host** (`execute_relational_select_cpu_pinned` → `finalize_relational_select`).
2. **The current sharded read path does not relieve memory pressure.** `execute_resident_partitioned_via_general`
   recompacts all shards with `cuMemcpyDtoD` *inside one CUDA context* (`gpu_id = partitions[0].gpu_id`), so every
   shard must already be on the **same** GPU — but the only reason to shard is to exceed one GPU's budget. Today's
   shards are a same-GPU construct; true spill (cross-GPU) has no execution model (no peer copy, no NCCL).

STRATA resolves both: a **commit-triggered admission producer** that lays a table down as 1..N shards, and a
**push-down + cross-shard-combine** read model that is correct and efficient for both same-GPU and cross-GPU shards.

---

## 1. Terminology — three levels, three names (the disambiguation)

The codebase calls the GPU residency unit `RelationalResidentPartition`, which collides with the SQL meaning of
"partition." STRATA keeps three **distinct, level-specific** names. Never reuse one level's word for another.

| Level | Name (STRATA) | What it is | Visibility | Today |
|---|---|---|---|---|
| **L1 Logical** | **SQL partition** | A user-declared horizontal partition of a table (`PARTITION BY RANGE/LIST/HASH`). Itself a relation with its own rows + catalog identity. | User-visible (SQL/DDL/catalog) | **Not implemented** — reserved term. No `PARTITION BY` in the parser. |
| **L2 Physical residency** | **shard** | A contiguous **row-range** of *one* relation's data, materialized into *one* GPU device buffer (columnar SoA + validity). 1..N shards per relation; **each shard lives on exactly one GPU.** | Invisible to SQL — a data-plane/residency concept. | `RelationalResidentPartition` (to be renamed `ResidentShard`). |
| **L3 Intra-shard layout** | **column section** | Within one shard: the SoA byte sections (`int4` section, `int8` section, `b128`/uuid section, `text` offsets+blob) + per-column validity bitmaps. | Invisible — a memory-layout concept. | `resident_device_*_columns` + `build_relational_device_payload`. |

**The mapping.** `SQL table ─(or, later, each SQL partition)─▶ 1..N shards ─▶ column sections`. When SQL partitions
land (L1, future), each SQL partition independently maps to its own 1..N shards — the two levels compose, they do not
merge. STRATA is entirely **L2/L3**; it does **not** add SQL partitioning. A non-partitioned SQL table is the common
case and still becomes 1..N **shards**.

**Renames this design implies (code):** `RelationalResidentPartition` → `ResidentShard`; `residency.partitions` →
`residency.shards`; `partition_device_memory` → `shard_device_memory`; `partition_id` → `shard_id`;
`plan_relational_partitioned_resident_route` → `plan_relational_sharded_resident_route`;
`execute_resident_partitioned_via_general` → `execute_resident_sharded_via_general`. (Mechanical, behavior-preserving;
sequenced as the first STRATA slice so all later slices speak the disambiguated vocabulary.)

---

## 2. Control plane vs data plane — host is control plane ONLY

STRATA's invariant restates doc-22 §1 in data-plane terms. The **host moves no row data except the two sanctioned
boundaries**: the *staging upload* of a new generation (wire → device) and the *final result readback* (device → wire).

| | **Host (control plane) — MAY** | **Device (data plane) — MUST** |
|---|---|---|
| Write | parse/plan; txn/WAL/replication; **admission decisions** (shard count, GPU placement, budget/evict); the **staging upload** that builds a shard from incoming rows | hold the entire payload (all columns incl. text + validity); be the source of truth for every value |
| Read | parse/plan; kernel orchestration/launch; **shard push-down + combine orchestration**; the single **final result readback** to serialize to the wire | scans, filters, joins, aggregates, sorts, grouping, DISTINCT, HAVING, LIMIT/OFFSET, expression eval, NULL/3VL, **and result materialization** |

Anything else touching row data on the host is a STRATA violation, exactly as in doc-22. The host *decides and
orchestrates*; the GPU *computes*. The two arrows (staging upload, result readback) are the only data-plane host touches.

---

## 3. End-to-end trace (target)

```
WRITE  client ─pgwire 'Q'─▶ run_shared_query_loop ─▶ execute_on_shared_engine ─▶ engine.execute_text
       └─ commit_mutation_at:  WAL.append → repl.propose → wal.flush_all → wait_committed        ── DURABLE
          → apply (publish data generation) → invalidate_relational_residency_for_commit
            (tombstone the mutated relation's shards) → publish_catalog → publish_committed_seq   ── COMMITTED
       └─ ★ ADMISSION PRODUCER (post-commit, best-effort, &self+catalog-guard):
            scan the new generation → choose shard plan (N, placement) → build/append shard buffers
            (staging upload) → admit (budget/evict) → install shards into residency               ── RESIDENT

READ   client ─pgwire 'Q'─▶ execute_on_shared_engine ─▶ engine.execute_relational_select
       └─ resident route:
            1 shard            → direct GPU exec over the shard buffer
            N shards, 1 GPU    → recompact (DtoD) OR partial+combine → exec → result buffer
            N shards, M GPUs   → per-shard partial on each GPU → cross-shard COMBINE (peer/NCCL of partials)
            not resident       → host fallback (cpu_pinned)   ← shrinks to "admission skipped / over-budget" only
       └─ device result buffer ─readback─▶ thin host transform ─▶ QueryOutcome ─▶ encode_outcome ─▶ wire T/D/C/Z
```

The host fallback never disappears by deletion; it disappears by **disuse** — once admission makes the common case
resident, the only reachers of the host path are (a) admission-skipped tables (over-budget before spill exists,
memory pressure, GPU-absent) and (b) catalog/view synthesis (§6-legitimate host). doc-22's S10d deletion is gated on
that reacher set being empty, which STRATA is the path to.

---

## 4. The resident shard model (L2)

A **shard** (today `RelationalResidentPartition`, `resident_storage.rs:616`) is the unit of GPU residency:

- **Identity & range:** `shard_id: u32`, `row_start: usize`, `row_count: usize` — a contiguous row-range of one
  relation. Shards of a relation are stored as a `Vec` sorted by `(row_start, shard_id)`
  (`residency.shards: ArcSwap<BTreeMap<String, Vec<ResidentShard>>>`, `engine_state.rs:510`).
- **Placement:** `gpu_id: u16` — **each shard lives on exactly one GPU** (STRATA invariant; the code carries the
  field but does not yet enforce uniformity — STRATA enforces it explicitly per read).
- **Layout:** `resident_device_int4_columns` + `resident_device_text_columns` (L3 sections), `resident_bytes`,
  `allocated_bytes`, `count_header_byte_offset`, `device_memory_proof`.
- **Lifecycle flags:** `invalidated_by_txn_id`, `invalidated_at_index`, `invalidated_by_memory_pressure`,
  `memory_pressure_active`; `is_valid()` gates reads. Device buffers live in
  `shard_device_memory: ArcSwap<BTreeMap<(table, shard_id), cell>>` (`resident_storage.rs:116`), tombstone-on-invalidate,
  COW-on-install — all `&self`.

**Shard lifecycle:** `Absent → Admitted(valid) → Invalidated(tombstoned, by commit/eviction/pressure) → re-Admitted`.
A mutating commit already tombstones (`invalidate_relational_residency_for_commit` → both unified + shard maps,
`engine_commit.rs:304-311`); the producer re-admits the new generation.

---

## 5. The admission producer (keystone)

A single new component on the **commit path**, the thing that does not exist today.

- **Trigger:** commit-triggered, **after `publish_committed_seq`** (so it scans the *new* generation — before it,
  `committed_seq()` is stale and it would admit old data). Commit-triggered (not read-triggered) because the read path
  is `&self`/lock-free by design and must stay so; admission mutates residency state.
- **Cannot fail a commit:** it runs *after* the txn is durable+committed. Over-budget / memory pressure / GPU-absent →
  the table is simply not admitted (reads use the host path); the commit already returned `Ok`. "Best-effort" is
  inherent to the post-durability placement, not a safeguard bolted on.
- **The `&self`+catalog-guard seam (verified feasible).** Every residency publish primitive is *already* `&self`/COW
  (`device_memory.insert`, `shards`/`snapshots` ArcSwap, `install_snapshot`/`install_shards` on the cache,
  `cuda_driver_probe_runtime`). The only `&mut self` in `populate_*`/`admit_*` is the `ddl_catalog_mut()` shortcut.
  STRATA threads the **already-held** `cat: &mut DdlCatalogState` guard into the producer (exactly as the DDL
  `apply_*(cat, …)` methods do, `engine_commit.rs:427`), so the producer runs on the `&self` commit path.
- **Shard plan (the L2 policy):** from the new generation's visible rows + a **per-shard byte budget** `B`:
  - total ≤ one GPU budget → **N=1** shard (or the existing unified `device_memory` form — the N=1 fast path).
  - total > `B` but ≤ one GPU → **N>1 same-GPU shards** (enables incremental append; recompaction-combinable).
  - total > one GPU budget → **shards across GPUs** (true spill; requires the §6 combine model).
- **Admit/evict:** reuse `admit_relational_residency_snapshot` unchanged — no-budget→admit; >budget→reject (becomes
  "not admitted"); fits→admit; needs room→deterministic eviction by oldest `valid_through_index`. Budget accounting
  already spans unified + shard bytes (`relational_resident_bytes_for_gpu`).
- **Gating:** behind an engine config flag (e.g. `auto_admit_on_commit`, **default off**). Default-off keeps the
  existing suite and the non-resident contracts intact (behavior-preserving); golden wire tests turn it on. **Flipping
  the default is a separate, deliberate step** once perf (incremental) + text + spill are ready — not a deferral, a
  sequencing of a semantic change with real blast radius.

---

## 6. The read path: shard push-down + cross-shard combine

The efficient, correct general model for N shards is **push the query fragment down to each shard, then combine
partials** — the MPP model adapted to GPUs. Recompaction-into-one-buffer is a *special case*, not the primary path.

| Shape | Per-shard partial | Combine | Cross-shard transfer |
|---|---|---|---|
| `COUNT(*)`, `SUM`, `MIN`, `MAX` | scalar | associative reduce | one scalar/shard (tiny) |
| `AVG` | (sum, count) | sum sums / sum counts → quotient | two scalars/shard |
| `GROUP BY` | per-shard group table | merge group tables on combine GPU | groups, not rows |
| `DISTINCT` | per-shard distinct set | union | distinct keys |
| `ORDER BY` / top-N | per-shard sorted run / local top-K | k-way merge / global top-K | K rows/shard (or full for unbounded ORDER BY) |
| projection / filter (no agg) | per-shard result rows | concatenate | the result rows |

- **Same-GPU shards:** combine via `cuMemcpyDtoD` — for projection this *is* today's recompaction
  (`retain_device_memory_recompacted`); for aggregates, scalar/group combine **avoids materializing the unified buffer
  at all** (more efficient than recompaction — STRATA prefers partial-combine for aggregate shapes).
- **Cross-GPU shards:** the *partials* cross GPUs (peer copy `cuMemcpyPeer` or NCCL), never the full data. This is the
  missing execution model that "assume partitions" forces; it is what actually relieves memory pressure.
- **Result readback GPU:** combine lands partials on a designated GPU (e.g. the default), which produces the final
  result buffer for readback. Only that final buffer crosses to the host.

The read route therefore generalizes uniformly: **1 shard** = degenerate combine (identity); **N same-GPU** = DtoD
combine; **N cross-GPU** = peer/NCCL combine. One model, three transports.

---

## 7. Efficient wire ↔ GPU data plane

**Ingest (wire → GPU).** `COPY`/`INSERT` → parse → **WAL (durability)** → stage the incoming rows once → the producer
builds/append-uploads shard buffer(s) → device. The staging upload (§6-sanctioned) is the *only* host data touch on
write; it decodes incoming bytes once into the columnar shard layout and uploads — no second host copy, no host-side
relational work. Append (a new shard) beats rebuild (whole-table re-upload); the shard model is what makes incremental
ingest possible.

**Egress (GPU → wire).** Device produces the **final result buffer**; the host reads it back once and serializes to
the wire. Target efficiency: the device emits results in a layout cheap to serialize so egress is a *thin* transform.
Today egress goes device → `Vec<Vec<SqlValue>>` (host structs) → `DbValue` → text `String` → wire bytes
(`engine_select_exec.rs` → facade `map_value` → `pg_adapter::db_value_text` → `BackendWriter::data_row`). STRATA's
data-plane target is to **collapse the per-value host re-materialization** — read back a compact device result buffer
and render to the wire's text/binary format with minimal host allocation. (Optimization axis; correctness-neutral;
the current path stays the baseline until measured.)

**The boundary fact:** by the time data reaches the facade it is already host-materialized — so STRATA's efficiency
work is squarely *before* the facade, in the result-readback shape, keeping the facade/wire layers protocol-neutral
and data-light.

---

## 8. Constraints & deferred axes (explicit, not hidden)

- **int4-only recompaction.** Sharded reads over a **text** column reject today (per-row offset rebasing unbuilt).
  N=1 shards support text already; N>1 text shards need the text-recompaction sub-item (doc-22 S10c remainder).
- **Single-GPU first.** Cross-shard combine (§6) is the gating piece for true spill; STRATA's producer emits same-GPU
  shards until combine lands. The design *names* combine as first-class so it is not re-discovered late.
- **Full re-admit per commit.** Until incremental shard append/replace lands, a mutating commit re-admits the whole
  relation. Acceptable for correctness and small tables; the shard model is the substrate for the incremental fix.
- **SQL partitions (L1) absent.** No `PARTITION BY` DDL. STRATA neither needs nor adds it; the term is reserved so L1
  can compose over L2 later without renaming.

---

## 9. Golden wire tests (the acceptance spec)

Golden = drive **only SQL over the real pgwire socket** and assert exact output — validating the *entire* trace
(admission → shard read → combine → readback → pg_adapter text rendering). Harness: the existing
`crates/server/tests/pgwire_roundtrip.rs` pattern (bind `127.0.0.1:0` → `thread::spawn(serve)` → `tokio_postgres`
`simple_query` → assert rows). Golden values are hand-computed closed-form rows (S9 oracle style; wire values are
text, NULL → absent column). GPU-guarded (`#[ignore]`, RTX PRO 6000) because admission uploads.

1. **Shapes × NULLs:** `CREATE → INSERT (enough rows to force the target shard count, WITH NULL data) → SELECT`
   covering `COUNT(*)`, filtered projection, `SUM/AVG/MIN/MAX`, `GROUP BY`, `DISTINCT`, `ORDER BY`, top-N. Assert exact
   golden rows.
2. **Route proof (non-vacuous):** assert the query was served by the GPU sharded route (surface `executed_target` /
   shard count to the test), so a silent host-fallback cannot pass the test vacuously.
3. **Differential:** identical golden data laid down as 1 shard vs N same-GPU shards vs (later) cross-GPU shards vs the
   host path → byte-identical wire rows (lift the `s10c_2b_*` cross-shard pattern up to the wire).
4. **Lifecycle:** `SELECT` → `INSERT` (invalidate → re-admit) → `SELECT` returns the new golden rows on the GPU route.

The suite is written **against the producer** — it is the failing/ignored spec that defines "done" for each STRATA
slice (test-driven: design fixed first, then green by construction).

---

## 10. Implementation sequencing

Each slice: behavior-preserving where it can be, differential **with NULL data**, HAZARD for device-touching slices,
**separate independent adversarial audit**, default-off flag until the deliberate flip.

- **S-A — vocabulary rename** (L2/L3 disambiguation, §1): `RelationalResidentPartition`→`ResidentShard` etc. Mechanical,
  behavior-preserving, no functional change. Lands the shared vocabulary first.
- **S-B — admission producer v1 (N=1 unified) behind `auto_admit_on_commit` (default off).** Commit-triggered,
  post-durability, `&self`+guard seam. Makes the GPU path reachable end-to-end via the wire for fits-one-GPU tables
  (int4 + text). Golden tests #1/#2/#4 go green.
- **S-C — same-GPU N>1 shards + partial-combine (int4).** Producer emits multiple same-GPU shards; read path prefers
  partial-combine for aggregates, recompaction for projection. Golden test #3 (same-GPU differential). Incremental
  append.
- **S-D — text shard recompaction/combine** (offset rebasing) so N>1 text shards route. Removes the int4-only guard.
- **S-E — cross-shard combine (peer/NCCL) → true spill / multi-GPU.** The over-budget execution model; cross-GPU
  golden tests. Unblocks the over-budget admission branch.
- **S-F — the deliberate default flip** (`auto_admit_on_commit` on) + migrate the non-resident test contracts; this is
  doc-22 S10d's real precondition (host path becomes disused → deletable).
- **Egress efficiency** (compact result readback) — independent optimization axis, sequence as measured.

---

## 11. Open decisions (for review)

1. **Per-shard byte budget `B`** — fixed bytes? rows? a fraction of GPU budget? Drives N and the same-GPU-vs-spill line.
2. **N=1 representation** — keep the existing unified `device_memory` as the N=1 fast path, or model *everything* as
   shards (N=1 is just one shard)? Uniform-shards is conceptually cleaner; the unified fast path avoids a needless
   single-shard combine. STRATA proposes: **unified `device_memory` is the N=1 shard representation** (no recompaction
   when N=1), shards-vector for N>1.
3. **Combine GPU selection** for cross-GPU reads (default GPU vs least-loaded vs the readback GPU).
4. **Admission synchrony** — admit inside the commit critical section (simplest; serializes commits behind the upload)
   vs just after releasing it (more concurrency; relies on the existing invalidate/best-effort race handling). STRATA
   leans **just-after-release**, best-effort, since the race is already handled by `is_residency_invalidated`.
5. **Egress format** — how far to push the compact device→wire result path in this campaign vs leave as a follow-up.
