# Independent design review (round 1, reassessed): `gpu-native-writes.md`

> Independent adversarial review (charter: never self-audit). **Round 1 originally 2026-06-30; reassessed
> 2026-06-30 against Slice 1b-ii-c (`3c1ec401` — open-shard append wired into the PRODUCTION commit path).**
> Reviews the design + the landed Slice-1 implementation from the **code only**, independent of any other
> review. No code changed by this doc.

## Verdict (reassessed)
Slice 1 is progressing cleanly: incremental INSERT via open-shard append is now wired into the production
commit path, and the index-correctness footgun ("Finding A") was caught and non-vacuously tested. The storage
+ durability design remains excellent. **The two structural gaps from round 1 still stand:** (a) this is still
a version-storage / durability effort, not the **concurrency-control / write-throughput** half R3 was
chartered for (ADR-009 deterministic CC); and (b) **index maintenance is still invalidate → O(table) rebuild**
on the next read. Plus a new scope note: the O(rows) fast-path currently covers only **single-log-entry
INSERT-only** commits.

## Reassessment vs the Slice 1b implementation

| # | Concern (round 1) | Status now | Code evidence |
|---|---|---|---|
| 4 | Index rebuild O(table) in Slice 1b | **Partial — correctness closed, cost remains** | `try_append` drops the stale cached index ("Finding A": an in-place append keeps the device ptr, so a generation-blind cached index would miss the appended key) + a non-vacuous index-probe==scan test over **appended** keys. But it is invalidate → **lazy rebuild**, so the next read still pays an O(table) index build (incremental index = Slice 5). |
| 1 | CC / commit-throughput half of R3 | **Improved, still open** | the serial-commit critical section is now O(rows) not O(table) (much shorter lock hold → higher *serial* ceiling) — a real gain. But commits remain **serial under the commit lock**; no pipelined / deterministic-CC model; ADR-009's spine / MV-dependency-graph is unengaged. |
| 2 | Concurrent-commit throughput benchmark | **Open (unchanged)** | still only `r3_dual_store_tax` + `r3_insert_profile`, both **serial single-row**. No concurrent-commit throughput bench. |
| 3 | In-place UPDATE patch torn-row hazard | **Not yet triggered (INSERT-only) — and the append evidence STRENGTHENS the recommendation** | the append writes slots **past `row_count`** (disjoint from the `[0, row_count)` read region) and publishes via `committed_seq` ordering → safe vs concurrent lock-free readers. An in-place UPDATE patch (Slice 4) overwrites *read* slots → **not** safe by the same argument. So "prefer append-new-version-to-open-shard over in-place patch for UPDATE" is now backed by Slice 1b's own design. |
| 5 | Single-row commit latency round-trip-bound | **Open (unchanged)** | still unstated; writes are a batched-throughput play (group-commit + the wave ring), not a single-row-latency win. |

### New observation — Slice 1b fast-path scope (narrow, by design)
The append fast-path is gated to a commit of **exactly one applied log entry** that is INSERT-only
(`to_apply.len() == 1`, int4-resident, headroom available). Multi-statement transactions (multiple log
entries), UPDATE / DELETE, text, no-headroom, and non-int4-resident tables all fall back to the O(table)
invalidate + re-admit. So the flat-tax win is currently **single-statement (possibly multi-row) int4 INSERT
only**. That is a reasonable first slice, but: (a) state the boundary explicitly in the proposal's Slice plan,
and (b) keep the non-vacuity guard — the dual-store-tax benchmark going *flat* IS the signal that the append
actually fired and didn't silently fall back (the wave faked-throughput lesson).

### Credit
"Finding A" — an in-place append keeps the device ptr, so a generation-blind cached index would silently miss
the appended key — is exactly the subtle correctness trap incremental in-place work creates, and it was caught
by the adversarial audit and closed with a non-vacuous test. Good discipline; the slice cadence
(feat -> independent audit -> adopt findings) is working.

## What remains before this is *the* accepted R3 plan (open items, still actionable)

1. **Close the concurrency-control / throughput gap (top priority).** Slice 1b made the *serial* commit
   cheaper; it did not make commits *concurrent*. R3's charter is commit/durability **and deterministic CC**
   at the SLO. Either re-scope the proposal title honestly as "write storage + durability (half of R3)" and
   open a sibling CC proposal, or add a commit-concurrency section: does the commit lock serialize, or hold
   only for seq-assign + WAL-append while the GPU applies in `commit_seq` order via the wave ring? what is the
   target concurrent-commit ceiling, and how does ADR-009's deterministic spine batch commits?

2. **Add a concurrent-commit throughput benchmark** (the write-side analog of `r2_wave_engine_ab`). The
   current gates prove correctness + flat *serial* per-insert cost — never concurrent throughput, which is the
   SLO question. Without it, you can finish Slice 7 and still not know whether the write path hits 100k TPS.

3. **Decide UPDATE = append-new-version vs in-place patch BEFORE Slice 4.** Slice 1b empirically shows append
   (disjoint from the read region) is safe vs the lock-free reader and in-place overwrite is not. Carry that
   forward: for UPDATE, append the new version to the open shard + tombstone the old (old value -> undo
   before-image), rather than patching the hot slot in place. Same for the "stamp `deleted_by` in place on a
   sealed shard" path — reconcile with the immutability claim.

4. **Index maintenance is the next real cost wall (Slice 5), and it gates the end-to-end O(rows) claim.** The
   append commit is O(rows), but the post-commit read rebuilds the index O(table). For an INSERT-then-read
   workload (the realistic case) the dual-store tax is *moved*, not removed, until incremental index
   maintenance lands. Measure the index-rebuild cost in an INSERT-then-point-lookup loop and show whether it
   reintroduces the tax; if it does, pull Slice 5 forward.

5. **State the type/commit scope** (single-statement int4 INSERT today) and the single-row-latency caveat in
   the proposal so expectations match the implementation.

## Net
The implementation is tracking the design well and the adversarial-audit cadence is catching real bugs. The
storage + durability half is sound and shipping. The unfinished half is the same one round 1 flagged —
**concurrency control + concurrent-commit throughput** — and it is now the highest-value next step, since
Slice 1b has shown the *serial* path can be made O(rows). Decide the UPDATE-version-storage approach before
Slice 4 (append, not in-place patch), and treat incremental index maintenance (Slice 5) as the gate on the
end-to-end O(rows) claim, not a follow-on.
