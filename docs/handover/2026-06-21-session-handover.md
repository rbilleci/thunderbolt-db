# Session handover — GPU sort operator + GROUP BY breadth (2026-06-21)

**Branch:** `phase0-m1-engine-facade` · **main:** pull latest `origin/main` (last code commit `b889fdbf`; handover docs committed on top) · **tree:** clean.
**Hardware:** RTX PRO 6000 Blackwell (cc 12.0). Build ptxas tops at **sm_90**; runtime JITs to sm_120.
**`--gpu-reset` is DENIED (shared infra).** ALWAYS run GPU tests under `timeout` (a hang can re-zombie the GPU).

This session delivered two complete, end-to-end bodies of work (every slice independently audited + merged to
main), then was redirected to write this handover. **Pick-up priority order (user-set 2026-06-21):**
**(1) GROUP BY follow-ups → (2) joins (M5) → (3) modernization (#21) → (4) remaining follow-ups.**

---

## 1. What this session delivered (all on main, all audited)

### GPU sort operator — COMPLETE (`7b819cd2`..`469599d6`)
Every `ORDER BY` is now a GPU sort; **no CPU relational sort**. Charter rule 1 (sorts are a GPU operation).
- **Bitonic primitive** (`7b819cd2`): `gpu_db_bitonic_sort_i64_step` + launcher. Swap iff `a_after_b==seq_asc`;
  **padding by POSITION (perm value >= n), never a key sentinel** (so DESC is safe).
- **Non-grouped single int key** (`4195a241`): routing gate `select_is_gpu_sortable_projection` routes a
  non-grouped plain-projection ORDER BY on an i64-sortable base col **of a RESIDENT table** to the general
  Expr path. (Fixed 2 audit P1s: the access_path was computed from the full select -> ran+discarded a CPU
  ordered sort -> now an order/limit-cleared clone; non-resident -> falls through to the existing path.)
- **Multi-key int** (`deec2e0e`): `order_by` -> `Vec<SelectOrder>`; `gpu_db_bitonic_sort_multikey_step`
  (row-major `keys[row*K+k]`, desc_mask bit k). **No-CPU-multi-key rule:** choke-point rejections of `len>1`
  above every CPU sort site. (Audit caught a P0: a 3rd path -- the text-entry `Err(_)` arm -> the general
  GROUPED branch sorted by `.first()` only; fixed by rejecting multi-key grouped at parse + a guard.)
- **Text keys** (`8fbc17d8` single, `6abfa991` mixed int+text): `gpu_db_bitonic_sort_text_step`
  (lexicographic, offsets read 2x ld.u32 = 716-safe, prefix=smaller) + `gpu_db_bitonic_sort_hetero_step`
  (per-key `key_plan` dispatch: int from an i64 matrix, text via the resident column; desc_mask keyed to
  ORDER BY position, NOT the slot). + text PROJECTION (varlen gathered host-side from the co-published
  host_rows at the SORTED indices).
- **Sort expressions** (`7be2ff11`): `ORDER BY a+b` evaluated on-device (`arith_value_column_at_indices`,
  checked overflow -> PG error) into an i64 key column. **Also fixed the slice-2 redundant-CPU-sort for
  real** (it keys off `bound.order`, not the `ap_select` clone -- now `bound.order=None` before the
  access-path call). A sort term is an EXPRESSION iff it does NOT resolve via `result_column_name` (columns
  AND aggregates resolve; only arithmetic doesn't).
- **numeric/uuid keys** (`9cba1188`): the hetero `key_plan` widened to a 2-bit kind (0 int,1 text,2 numeric,
  3 uuid); a 16-byte leg copying the GROUP BY i128 (signed-hi/unsigned-lo) + uuid (big-endian) compares.
- **Grouped migration** (`09c17366` 6a + `97bf89bd` 6b): the grouped result's user ORDER BY sorts on the
  GPU (any key type, multi-key). 6a extracted `build_relational_device_payload` (the residency payload
  builder, byte-identical); 6b synthesizes a resident-like payload from the small host grouped result +
  `bitonic_sort_hetero_on_payload` (`resident_base_payload: Option`). NO host-side user sort.
- **Radix fast-path** (`469599d6`): `order_by_sort_i64` dispatches `bitonic_sort_i64` (<10k) /
  `launch_cuda_order_by_sort_i64_radix` (>=10k, reuses the proven `launch_cuda_resident_i64_argsort_radix`).
  2.6x@10k -> 18.9x@1M. The upload lease MUST outlive the radix call (the audit's fault-injection proved it).

### GROUP BY breadth — type matrix CLOSED + all operator gaps (`578e079f`..`b889fdbf`)
- **GROUP BY `<expression>`** (`578e079f`): a `key_base_override` on the single-level GROUP BY kernel
  (override!=0 -> the key base is a derived buffer; override=0 byte-identical). The expr is materialized
  on-device -> the derived key buffer -> override. Projection matches `SELECT a+b` to `GROUP BY a+b` via
  `node_struct_eq` (location-ignoring structural eq on the libpg_query Node). Carried engine-side as
  `group_key_expr` (no sql/lib.rs AST change).
- **bool key + MIN/MAX(bool)** (`ebfb85d5`) -- the type matrix is now CLOSED. Materialize bool->int4 (reuse
  `gpu_db_resident_bool_to_mask`, it already writes int4 0/1) -> `key_base_override` (key) + a new symmetric
  `value_base_override` (value, for MIN/MAX). **NO bool GROUP BY kernel** -> sidesteps the prior concurrency
  hazard (verified gone: 7 parallel runs, zero 700/717). See [[gpu-bool-groupby-blocked]].
- **Composite 2-column keys** (`0d6ea620`): `gpu_db_pack_two_int4_cols` (a no-atomics MAP, `(c0<<32)|c1`)
  -> `key_base_override` -> unpack the result. `group_key_columns: Vec<String>` engine-side. Result-schema
  patch: insert b's RelationalColumn into `bound.selected_columns` + renumber attnums (the binding holds
  only one group_column).
- **COUNT(DISTINCT v)** (`b889fdbf`): sort `(g,v)` via `bitonic_sort_multikey` -> a no-atomics mark kernel
  `gpu_db_mark_new_distinct` (full `(g,v)` tuple compare) -> `group_by SUM(new_distinct)` per group.
  `GroupedAggKind::CountDistinct` (a sql/lib.rs AST addition, ~18 match sites). Combined `COUNT(*),
  COUNT(DISTINCT)` folds via the multi-aggregate merge (aligned by MATERIALIZED group key, not slot order).

### Reusable mechanisms born this session (joins + follow-ups build on these)
- **`key_base_override` / `value_base_override`** on `gpu_db_group_by_i32_count_sum` (single-level): the
  GROUP BY kernel reads the key/value from an absolute device ptr. override=0 = byte-identical column path.
  Used by expr / bool / composite / count-distinct derived keys/values. This is the "derived column" lever.
- **The hetero sort + `bitonic_sort_hetero_on_payload`**: sort any-type tuples; the payload variant sorts a
  synthesized resident-like buffer (the grouped result) -- reusable for joins.
- **`build_relational_device_payload(column_names, column_types, rows)`** (engine_residency.rs): build a
  resident-like columnar device buffer (8-aligned text offsets) from arbitrary host rows.

---

## 2. Open points by priority (the audit)

### PRIORITY 1 — GROUP BY follow-ups
- **Composite keys** (#27 done for exactly TWO int4-section members int2/int4/date -> i64):
  - int8/wider members (combined width > 64 bits) -> **i128/b128 packing** (key_is_i128 path already exists).
  - **text members** -> the rep-row approach: a b128 slot `(tuple_hash, rep_row_idx)` + verify the full tuple
    on a hash collision (REUSE the text-key b128 (hash, rep_idx)+verify from `gpu-group-by-breadth`).
  - **>2 columns** (currently exactly 2). numeric/uuid members.
- **COUNT(DISTINCT v)** (#28 done for int2/4/8/date/timestamp value + plain-column int group key):
  - **text/numeric/uuid value** columns -> needs the HETERO sort over the value (the multi-key sort only does
    i64; route the `(g,v)` sort through `bitonic_sort_hetero` for a non-int v).
  - **scalar `COUNT(DISTINCT v)` with no GROUP BY** (one group).
  - **non-int group key** (expr / bool / composite / text / numeric / uuid group key) -- currently the group
    key must be a plain-column int (the (g,v) matrix is i64x2). Compose with #26/#27.
- **GROUP BY `<expression>`** (#26 done): `ORDER BY <the group expression>` on a grouped result is rejected
  (the grouped sort reads result columns; the group expr IS a result column PG would order by). Minor.
- **Weak-oracle test** (6a audit note): `gpu_grouped_reuse_types_int2_date_timestamp` only checks MIN<MAX for
  timestamps (a consistent byte-swap preserves order); byte-exactness is covered elsewhere. Tighten if bored.

### PRIORITY 2 — Joins (M5) — the big milestone
- The GPU sort + the GROUP BY hash (open-addressing, **b128 i128 keys** = the build side) are exactly the
  **merge/hash-join build+probe primitives**. The CPU nested-loop join engine is DROPPED (charter rule 1);
  joins are GPU operators (partitioned/hash join). See [[gpu-native-charter]], [[gpu-phase3]].
- **Why now:** psql `\d` / pg_dump / ORMs issue MULTI-RELATION CATALOG joins; the engine has 0 joins; this is
  the real unlock for deleting the §9.1 canned-matcher. Do **joins-for-catalog first**.
- **Parser:** the general libpg_query path (`build_select_from_select_stmt`) currently rejects >1 FROM
  relation (`engine_sql_pg.rs:132` "supports exactly one FROM relation (no joins yet)"). libpg_query DOES
  parse `JoinExpr` (it's the full PG parser, already used for the general Expr path) -- so parse the
  multi-relation/`JoinExpr` from_clause there. The hand-rolled parser (legacy pgwire server) does NOT parse
  JOIN and should stay untouched (dual-entry) -- a JOIN query falls to the `Err(_)` arm -> the general path.
- **Executor:** a GPU hash join -- build a hash table on the smaller (resident) side (reuse the GROUP BY
  open-addressing hash / the b128 slot), probe with the other side, materialize matched rows. Then extend to
  the predicate/projection over the joined tuple via the general Expr interpreter.
- **Scope ladder:** inner equi-join (catalog `\d` shape) first -> then the predicate/projection -> then
  outer/multi-way. The residency already holds the relations; the join reads two resident snapshots.

### PRIORITY 3 — Modernization pass (#21, charter rule 3 pre-joins sweep)
- **`.ptx` arch targets** (raise toward the sm_120 floor; build ptxas max is sm_90 -- target sm_90, runtime
  JITs to sm_120): `pack_i32.ptx` sm_60, `gather.ptx` sm_60, `pack.ptx` sm_60, `widen.ptx` sm_60,
  `having.ptx` sm_70, `expr_proto.ptx` sm_90 (already current).
- **Toolchain:** verify whether a newer CUDA toolkit (ptxas that accepts `--gpu-name sm_120`) is installable
  on the box. Current ptxas does NOT list sm_120 (`ptxas --help` -> max < sm_120); the runtime JIT handles
  cc 12.0. If a newer ptxas is available, target sm_120 natively; else document sm_90+runtime-JIT as the
  floor and move on.
- **numeric/uuid MIN/MAX two-pass -> b128 CAS loop:** `gpu_db_group_by_numeric_minmax_lo` (expr_proto.ptx
  ~:5470/:5619) is the pre-b128 lock-free TWO-PASS (hi via atom.min/max.s64 + row_slots scratch, then lo).
  Migrate to a single **`atom.cas.b128` CAS loop** (like the uuid MIN/MAX already does). LESSON: GPU
  spin-locks DEADLOCK -- never reintroduce one; the CAS loop advances on failure (lock-free).
- Audit all kernels to latest standards (atomics, ld/st widths, the 716 alignment guards).

### PRIORITY 4 — Remaining follow-ups (incl. the open CHARTER DEBT)
- **#30 (charter debt):** `engine_expr.rs:1441` -- the multi-aggregate merge's `pass.groups.sort_by(group
  key)` is a HOST sort, and it is **LOAD-BEARING** (it aligns each separate group-by launch's independent
  slot order so the index-merge pairs the right group; a real intermittent 6th-launch divergence exists for
  an i64::MIN int8 key). It needs an on-device equivalent, not removal. A real CPU sort in the relational
  path -> charter debt. (Also: `engine_expr.rs:1524` is a removable no-op guard; `resident_expr.rs:2977` has
  a stale "host-side" comment now that grouped ORDER BY is a GPU sort.)
- **#31 (charter debt):** the `execute_relational_select_text` entry routes only MULTI-aggregate grouped
  multi-key ORDER BY to the GPU sort; a SINGLE-aggregate grouped multi-key ORDER BY via that entry hits the
  pre-existing clean reject at `engine_select_exec.rs:~135` (it WORKS via the direct
  `execute_resident_expr_select_sql` entry). Wire the text-entry single-agg grouped path.
- **Over-all-rows overflow divergence:** `ORDER BY a+b` / `GROUP BY a+b` evaluate the expr over ALL rows then
  gather survivors, so `WHERE (drops the overflowing row) ORDER BY a+b` ERRORS where PG (evaluating only
  survivors) would not. Shared with the WHERE-arith VM. Fix = gather-then-evaluate. P1, not blocking.
- **Case-sensitivity:** the routing gate matches ORDER BY/GROUP BY col names case-INsensitively while exec
  matches case-sensitively -> a case-mismatch yields a clean "column does not exist", never wrong rows.
  Align eventually.
- **#11:** extract execution-crate mod tests into feature files (old, pending).
- COUNT(DISTINCT) `Pass.value_is_int8` is set false while the group-by call passes true -- harmless (the
  result reads `groups[i].sum` low-64; distinct counts fit i64). Note only.

---

## 3. Technical context the new session needs

**The general GPU executor** (charter rule 2): a general `Expr`/operator interpreter (`engine_expr.rs`
`execute_resident_expr_select*`), NOT a catalog of query shapes. Routing: `execute_relational_select_text`
tries the hand-rolled parser; what it rejects falls to the `Err(_)` arm -> `execute_resident_expr_select_sql`
(the libpg_query general path). Keep the hand-rolled parser + legacy pgwire server UNTOUCHED (dual-entry).

**Gotchas (each cost a P0 this session or earlier):**
- **716 misaligned load:** read a 64-bit device value as **2x `ld.global.u32`** when a section can be 4-mod-8
  (e.g. varlen text offsets after a data-dependent blob). Varlen offsets sections must be 8-ALIGNED.
- **Merge by MATERIALIZED key, never slot order:** the `atom.cas` slot-claim order differs between separate
  group-by launches; multi-pass merges (multi-aggregate, count-distinct) MUST re-sort each pass by the
  materialized group key before index-merging.
- **GPU spin-locks DEADLOCK** (zombie context survives SIGKILL): use lock-free `atom.cas` (advance-on-failure
  = open-addressing probe) or `atom.min/max`. Never a spin lock.
- **The bool GROUP BY hazard:** reading the bool bitmap INSIDE the GROUP BY kernel corrupts SHARED state under
  the parallel suite (700/717). The fix that shipped: materialize bool->int4 in a SEPARATE no-atomics MAP +
  the audited int4 path. Any new GROUP BY-kernel-internal type read risks reviving it -- prefer a separate
  materialize + `key_base_override`/`value_base_override`.
- **Lease lifetime:** a derived device buffer's `DeviceArithBuffer`/`PooledBufferLease` MUST be bound to a
  variable that outlives ALL aggregation passes / the kernel call (an early free + pool reuse = UAF). The
  radix + count-distinct audits both fault-injected this.
- **Checked overflow -> PG "integer out of range", never wrap, never CPU.** Inherited from the arith VM.

**Charter (NON-NEGOTIABLE, user-reinforced 2026-06-20 "the charter is the charter for a reason"):** the GPU
executes the WHOLE relational data path; CPU = host/control-plane ONLY; CPU relational execution is tracked
DEBT, never product direction. Do NOT rationalize CPU shortcuts -- "it's small / it's finalization" is NOT a
carve-out (this was an explicit correction re: the grouped sort). See [[gpu-native-charter]].

---

## 4. Process (followed all session; keep it)

- **Per slice:** implement -> **INDEPENDENT adversarial-audit fork** (a SEPARATE agent -- NEVER self-audit,
  a green suite != an audit; see [[independent-audit-required]]) -> gate (ptxas sm_90 + build + host + the
  GPU `--ignored` suites + clippy `--workspace --all-targets -D warnings`) -> commit -> merge. Audits
  fault-inject to prove the tests are non-vacuous; several caught real P0s (the multi-key-grouped bypass, the
  int8-expr garbage, the redundant CPU sort).
- **Merge workflow:** `git commit` -> `git push origin phase0-m1-engine-facade` -> `git checkout main` ->
  `git merge --ff-only phase0-m1-engine-facade` -> `git push origin main` -> `git checkout
  phase0-m1-engine-facade`. (Always on the branch; ff-only to main.)
- **The sensitive GROUP BY kernel:** forks tend to STOP at multi-file changes to the most-audited path
  (correctly cautious). When the plan is fully mapped + de-risked, dispatch with a firm "land it -- the audit
  is the parent's job after, the regression matrix is the safety net". That landed #26/#27/#28.
- **Server flakiness this session:** intermittent 529 / rate-limit killed several forks MID-RUN. The
  implementation usually landed IN-TREE before the report was cut off -- verify `git status` + the build, run
  the gate yourself, then audit + commit. Re-dispatch transient failures.
- **Hazard re-runs:** for anything touching the GROUP BY kernel, run the FULL engine GPU `--ignored` suite
  (the parallel trigger) several times (incl. 2 concurrent engine||execution processes) -> ZERO 700/716/717.
- Commit-message footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` +
  `Claude-Session: https://claude.ai/code/session_0176nyZKJ9BkEwpXXd9sU2Do`.

---

## 5. Key references
- **Commits:** sort `7b819cd2`..`469599d6`; group-by `578e079f`/`ebfb85d5`/`0d6ea620`/`b889fdbf`.
- **Files:** `crates/execution/src/expr_proto.ptx` (kernels), `crates/execution/src/lib.rs` (launchers +
  `bitonic_sort_*` / `order_by_sort_i64` / `*_device` materializers), `crates/engine/src/engine_expr.rs`
  (the general executor + grouped branch), `crates/engine/src/engine_sql_pg.rs` (libpg_query -> Select +
  `parse_group_by` + `build_grouped_projection`), `crates/engine/src/rel_exec_helpers.rs` (binding),
  `crates/engine/src/engine_residency.rs` (`build_relational_device_payload`), `crates/sql/src/lib.rs`
  (`Select` / `SelectProjection` / `GroupedAggKind`).
- **Memories (loaded each session via MEMORY.md):** [[gpu-order-by-sort]], [[gpu-group-by-breadth]],
  [[gpu-bool-groupby-blocked]], [[gpu-native-charter]], [[independent-audit-required]], [[gpu-phase3]],
  [[gpu-sql-to-expr-handoff]].
- **Tasks open:** #11, #21, #30, #31 (+ the GROUP BY/joins follow-ups above are not all ticketed yet).
