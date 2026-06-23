# Full GPU-Native Read Path — Host Out of the Data Path (Campaign)

**Status:** in progress. **Priority:** high. **Mandate (user, 2026-06-23):** go full
GPU-native, **zero charter violations, zero deferrals.** The host is control plane ONLY.
Stop the cross-session pattern of deferring the hard GPU kernel and shipping a host-side
stub. See memory `full-gpu-native-no-deferrals`, `gpu-native-charter`.

## 1. The invariant (the checkable bar)

Every relational decision and every result value is computed on, and read back from, the
**device**. `host_rows` reverts to ingest-staging only.

- **Host MAY (control plane):** wire I/O; SQL parse + plan; kernel orchestration/launch;
  txn coordination; WAL/durability I/O; the COPY/write/DDL **staging upload** (build the next
  device generation from incoming data + upload it); read back the **final device-produced
  result buffer** to serialize to the wire.
- **Host MUST NOT:** scans, filters, joins, aggregates, sorts, grouping, DISTINCT, HAVING,
  LIMIT/OFFSET-applied-to-data, expression eval, NULL/3VL semantics — **and must not
  materialize result values from `host_rows`** (a host data copy) instead of from the device.

There is no excuse to defer: the device already holds the **entire** payload (every column
incl. text, plus validity bitmaps). Every host data touch is a choice, not a necessity.

## 2. Decisions (user, 2026-06-23)

- **Scope = everything, incl. oracle.** Retire the CPU parity oracle AND the GPU-absent
  bootstrap fallback entirely — zero host relational code anywhere, even tests/CI. Parity tests
  use **GPU-native oracles** (on-device serial-vs-parallel / construction / closed-form, per
  `gpu-test-oracles`), never a CPU re-implementation. The engine **requires** a GPU (charter
  sm_120 floor).
- **Start = keystone first** — deterministic on-device GROUP BY output ordering + on-device
  TEXT materialization (both reuse existing kernels: the GPU sort + `project_text_rows_from_payload`).

## 3. Execution discipline

- A slice is **GPU-native or it does not land.** No host stub committed as "done"; no
  "deferred"/"follow-up"/"clean-error-for-now" escape hatch past a hard kernel.
- Each slice: behavior-preserving (verified by the GPU golden/parity suite), kernel-touching
  slices run the **HAZARD** protocol (3× + concurrent, zero 700/716/717), and each gets a
  **separate independent adversarial audit** (`independent-audit-required`) — never self-audit.
- Interim host code that a later slice will replace is **work in progress, not a deferral** —
  the campaign is not done until §4 is fully struck through.

## 4. Complete violation inventory + slice plan (deferral-free)

Every host-side relational touch on the read path found by the 2026-06-23 sweep. Nothing here
is optional; each line is struck through only when it runs on the device.

### Keystone — TEXT materialization + deterministic GROUP BY ordering
- [x] **S1 — resident SELECT projection TEXT on-device.** `engine_expr.rs:4837` read
  `host_rows` for text; now uses `project_text_rows_from_payload`. **DONE `0fcd9e09`** (GPU
  suite 234/0; independent audit in progress).
- [ ] **S2 — GROUP BY result on-device (text + ordering + the pass-alignment merge).** Move:
  the host pass-alignment re-sort ("charter debt #30", `engine_expr.rs:3961`); the default-order
  host sorts (`4116`/`4130`); and all GROUP BY text materialization from `host_rows` via rep-row
  index (`3924` single text key, `3987` wide-key members, `3997/3998` composite text, `4080/4085`
  MIN/MAX text). Approach: materialize group key/value columns into a device payload (text via
  rep-index gather through `project_text_rows_from_payload`); make group output order
  deterministic on-device (single canonical ordering across passes so alignment needs no host
  sort — e.g. sort each pass by key on-device, or a single multi-aggregate pass) and reuse
  `gpu_sort_result_rows` for default + ORDER BY order.

### Result-stage operators on the general executor
- [ ] **S3 — HAVING on-device.** `engine_expr.rs:4168` `rows.retain(...)` → device mask over the
  group payload.
- [ ] **S4 — LIMIT/OFFSET on-device.** `engine_expr.rs:4201` and `4916` host `drain/truncate` →
  device-side window (truncate the index vector before final readback).

### Join (the original charter-audit finding)
- [ ] **S5 — join NULL-key skip in the kernels (V1).** `key_present` reads `host_rows`
  (`engine_expr.rs:2291-2313`) → validity-bitmap param on the build/probe/emit kernels
  (`expr_proto.ptx`), NULL key skipped on-device (sentinel `u64::MAX`=no-bitmap=byte-identical).
  Covers int / text+b128 / N:N variants.
- [ ] **S6 — join pad-WHERE 3VL on-device (V2).** `predicate_truth_on_null_pad` host Kleene
  (`engine_expr.rs:108-153`) → evaluate the all-NULL pad via the device WHERE-3VL mask VM.
- [ ] **S7 — join result materialization on-device (V3 + values).** Final gather reads
  `host_rows` for all columns incl. NULL emission (`engine_expr.rs:2453-2470`) → gather columns
  (incl. text) from each relation's device payload by the carried index vectors; NULL from a
  device validity bit; `JOIN_NULL_ROW` pad → validity 0.

### Resident-probe fallback branches
- [ ] **S8 — `!gpu_ordered` host finalization in resident-probe.** Host sort/HAVING/LIMIT
  branches at `engine_resident_probe.rs:3145-3187` and `3383-3425` → on-device (or route to the
  GPU-ordered path so the branch is dead, then delete it).

### Retire the host relational path entirely (the "incl. oracle" decision)
- [ ] **S9 — replace CPU-oracle parity tests with GPU-native oracles** (serial-vs-parallel /
  construction / closed-form) wherever tests currently assert vs a CPU re-implementation.
- [ ] **S10 — delete the host SQL finalization path** (`engine_select_bind.rs` host
  sort/agg/DISTINCT/HAVING/LIMIT) and the **`mvcc_read_exec.rs` `cpu_fallback`** once S2–S8 make
  the GPU path total for supported types. No CPU relational execution remains.

## 5. Sequencing

S1 ✅ → **S2 (keystone, the big one)** → S3 → S4 (result-stage operators; small, mechanical) →
S5 → S6 → S7 (join) → S8 (probe fallback) → S9 (GPU-native oracles) → S10 (delete host path).
S9 underpins S10 and is done alongside each slice's tests. Order within S3–S8 is flexible; S2
is the keystone and unblocks the most queries.

## 6. Out of scope (legitimately host, NOT violations)

Parse/plan, kernel orchestration, txn/WAL/wire I/O, the COPY/write/DDL staging upload, and the
single final device→wire result readback. Moving SQL parsing or ingest staging onto the GPU is
**not** a goal.
