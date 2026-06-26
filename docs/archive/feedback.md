# Feedback: GPU-OLTP Architecture Proposal (5-point plan)

Grounded against the current tree. Two corrections to the proposal's premises before the assessment:

- **WAL durability is *not* a no-op today.** The code has real fsync — `flush_all`, file +
  **parent-directory** fsync, CRC-on-recovery, and an explicit **group-commit fsync** on the concurrent
  DML path (`engine_dml_concurrent.rs:240`: "WAL append + propose + group-commit fsync"; `:242` "the fsync
  completes before we publish"). So point 5's "currently a no-op, v1 blocker" is wrong about the current state.
- **GPUDirect/cuFile: confirmed absent.** That part is genuinely greenfield.
- The batching primitive (microbatch read runtime) exists, consistent with the roadmap's M3/M4 — point 1's
  "embryo" is real.

None of these are *stupid* — it's an internally coherent design (essentially **"Calvin-on-GPU with a
persistent kernel over coherent memory"**), and it's worth noting that **1, 2, and 5 extend direction the
codebase already leans into** (the batcher, ADR-002 deterministic ordering, existing group-commit fsync),
while **3 and 4 are the genuinely new bets.** But several have a misguided sub-claim, and the scorecard
isn't uniform.

| # | Verdict | One-line |
|---|---|---|
| 1 Persistent kernel | **Good, oversold** | Right core insight; "launch latency → zero" ignores occupancy cost + warp divergence |
| 2 Deterministic + OCC | **Best idea; OCC part muddled** | Deterministic spine is the most GPU-native CC choice; OCC is redundant under a total order, and the model silently narrows to one-shot txns |
| 3 Coherent memory | **Right bet, misguided as a *requirement*** | Strategically correct to target; wrong to require it or let it delete explicit placement; untestable on your hardware |
| 4 Hybrid layout + indexes | **Indexes good; row layout questionable** | GPU coalescing on a *batched wave* can favor columnar — inverts the CPU intuition, and it fights point 1 |
| 5 Group commit + GDS | **Group commit good (partly built); GDS misguided** | fsync isn't a no-op; GDS is a bandwidth tool aimed at a latency/correctness floor |

---

## 1. Persistent kernel + lock-free queue — **good direction, oversold claim**

The core insight is correct and it's the right thing to build: stop paying per-query launch. But "collapses
launch latency to zero" hides two costs the plan needs to own:

- **A persistent kernel reserves the GPU.** A never-exiting kernel draining a ring occupies SMs and
  busy-waits continuously. On your box that's a hard problem — it's a *shared* box with `--gpu-reset DENIED`,
  and you've already hit zombie-context-survives-SIGKILL. A spinning resident kernel that wedges is exactly
  that failure mode, now permanent instead of per-launch. You're trading launch latency for a standing
  reservation and a harder kill story.
- **Warp divergence across heterogeneous transactions.** A wave of 10k *identical-shape* point lookups
  coalesces beautifully. A wave of mixed transactions (some insert, some update, some multi-statement,
  different branches) diverges — 32 threads in a warp taking different paths serialize. The persistent-kernel
  model is a **throughput** mechanism for *homogeneous* waves, not a single-transaction-latency mechanism for
  arbitrary ones. Single-txn p50 is still bounded by the wave-loop period + memory round-trip, not zero.

So: build it, but frame it honestly as the throughput engine for homogeneous routes (which is what your
batched read-runtime already is), not as a universal "sub-µs for any transaction" claim.

## 2. Deterministic ordering + OCC — **the best idea here, with two caveats**

Deterministic execution is the single most GPU-native concurrency-control choice, for exactly the reasons
given: it turns CC into a *batch* problem, kills locks/latches/deadlock, and the order *is* the replication
log. And it's **already latent in your architecture** — ADR-002 is deterministic batch ordering, and the
replication path proposes order. This is the strongest strategic call in the list. Two caveats, though:

- **"Deterministic + OCC, both" is conceptually redundant.** Under a total pre-order, within-wave conflicts
  are resolved *by the order* — you don't validate-and-abort, that's what determinism exists to eliminate.
  Bolting OCC (CAS-validate, abort losers, retry) back on top *reintroduces the abort storms* you just
  removed. What you actually want under the order is **dependency-graph parallel execution** (BOHM/PWV-style
  multi-version): build the conflict graph from the wave's read/write sets, run non-conflicting txns in
  parallel, no aborts. So keep the deterministic spine; replace "OCC underneath" with "MV dependency-graph
  execution underneath."
- **Determinism narrows the transaction model.** Calvin/BOHM need the read/write set *before* execution.
  Transactions with dependent reads (read a value, then decide what to write) need a reconnaissance pre-pass
  (Calvin's OLLP) — awkward on a GPU. In practice this pushes you toward **one-shot / stored-procedure
  transactions with pre-declarable access sets**, and *away* from interactive `BEGIN; SELECT; …; UPDATE;
  COMMIT;` over the wire. That's a real semantic narrowing, and it's the same narrowing the whole bundle
  implies (see the closing note). It's a fine target — but choose it consciously, because it changes the
  wire surface.

## 3. Coherent CPU–GPU memory — **right bet, misguided as a hard requirement**

Strategically this is the most important item and you're right to put it in the charter: your OLTP-on-GPU
thesis is *most defensible precisely where the hardware is going* (NVLink-C2C ~900 GB/s on GH200/GB200
deletes the PCIe round-trip). Betting on the trajectory that AI demand keeps improving is sound. But there
are two places it tips from good to risky:

- **Don't make it a hot-path *requirement*.** Your own dev box is an **RTX PRO 6000 — a PCIe part with no
  C2C.** If the hot path *assumes* coherence, you're designing the core data plane on hardware you can't
  test, for a deployment slice (GH200/GB200) that's expensive and rare. Design the transport as an
  *abstraction* so coherent memory is a drop-in fast path, and treat it as the premium deployment — not the
  floor.
- **Don't let coherence delete explicit placement.** "STRATA's spill becomes hardware-managed coherence" is
  the part I'd push back on hardest. Hardware demand-paging over C2C is fast on average but **page-fault-driven
  access wrecks tail latency** — and OLTP lives and dies on p99/p99.9. Keep STRATA's explicit
  residency/admission as the *policy* layer (you control placement, you control the tail); use coherence as
  the cheaper *transport* underneath it, not as a replacement for the policy. Surrendering placement to the
  hardware is trading away the exact thing OLTP can't afford to lose.

So: yes to coherent memory as the strategic target and a first-class *fast path*; no to "required" and no to
"it makes spill free."

## 4. Hybrid row/columnar + GPU-native indexes — **split: indexes good, row layout probably misguided**

- **Indexes: good and necessary**, and already on your roadmap (M5/M6 list "GPU-resident lookup/range/filter
  structure" as deliverables). Hash for equality (you have value-index), ordered/learned for range, always
  *batched coalesced probes* — all correct. One nit: "never B-trees" is slightly too absolute (Harmonia-style
  GPU B-trees exist and work via batched probes), but the spirit is right.
- **Hybrid row-major layout: this is the weakest technical claim, and it may be backwards.** The CPU
  intuition — "a point lookup wants one contiguous row, columnar scatters across N buffers" — *does not
  transfer cleanly to a GPU doing a batched wave of lookups*, which is exactly point 1's model. For a warp of
  32 threads each fetching `row[tid]`, **row-major addresses are strided by row-width → uncoalesced gather**,
  while **columnar reads one dense element per thread per needed column → better coalescing, and only the
  projected columns**. So the very batching that makes the persistent kernel work can make *columnar* the
  better point-access layout, not worse. Adding a second physical layout is large, permanent complexity (dual
  code paths, conversion, maintenance). I'd **measure the coalesced batched-gather on columnar first**; if a
  hybrid is genuinely needed, prefer **PAX** (keeps coalescing) over pure row-major. As stated, this point
  optimizes for the single-thread-single-row case that your own execution model doesn't use.

## 5. Group commit + GPUDirect WAL — **split: group commit good (and partly built); GDS misguided**

- **Group commit: correct, necessary, and *already partially real*** — not a no-op. The fsync floor framing
  is right ("GPU doesn't change physics"), and the wave model amortizing one fsync across thousands of txns
  is the standard answer. But the code already has WAL fsync + parent-dir fsync + CRC-recovery + a
  group-commit fsync on the concurrent path, and PLAN.md §5 already lists fuller group-commit as planned work.
  So the real gap is *GPU-side WAL generation* and tightening group-commit batching, not durability-from-scratch.
- **GPUDirect *Storage* for the WAL: misguided — right problem, wrong tool.** GDS (cuFile) is a *bandwidth*
  technology for large streaming transfers (training data, checkpoints). The WAL hot path is *small,
  sequential, latency-and-correctness-bound*. GDS doesn't move the fsync floor (the durability barrier is the
  cost, not who issues the DMA), and it puts your **crown-jewel correctness** — ordered, crash-consistent,
  ack'd durable writes — onto a novel, hard-to-verify GPU-direct path. The simpler and safer design falls out
  of points 2+3: the GPU generates WAL records → coherent memory → **the host (which already owns the
  deterministic order) writes them durably with group-commit fsync.** WAL delta volume is small; transport
  doesn't matter; correctness does. Reserve GDS for **snapshot/checkpoint materialization** — large,
  bandwidth-bound, genuinely its sweet spot.

---

## The thread tying all five together (the decision underneath)

Notice every point pushes the same direction: **homogeneous, one-shot, statically-analyzable transaction
*waves*.** The persistent kernel wants homogeneous waves (1); deterministic CC wants pre-declared read/write
sets (2); coalesced layout/indexes want batched identical probes (4); group commit wants a wave to amortize
over (5). That's the GPU's genuine happy place — and it's a **real, valuable workload**: high-throughput
stored-procedure OLTP (ledgers, banking postings, order processing with predeclared transaction types).
Calvin was built for exactly this and it's a legitimate, differentiated target.

But it is *not* "drop-in interactive Postgres OLTP." The whole bundle collectively trades away ad-hoc,
interactive, dependent-read, multi-statement transactions — the thing a generic Postgres client does. That's
the same narrowing flagged earlier, now made concrete by the architecture: **this is a coherent design for
deterministic high-throughput batched OLTP, and a poor fit for interactive single-transaction OLTP.**

**Recommendation:** adopt **2 (deterministic spine, MV execution not OCC)** and **1 (persistent kernel,
framed as the throughput engine)** as the core — they reinforce each other and extend what you already have.
Put **3 (coherent memory)** in the charter as the strategic *target* and *fast path*, but not a requirement
and not a replacement for STRATA's placement. Take **the index half of 4**; gate the row-layout half behind
measurement. Keep **group commit from 5**; drop **GDS-for-WAL** until you have a checkpoint workload that
actually wants it. And write down, explicitly, that the target is *deterministic batched OLTP*, not
interactive Postgres OLTP — because points 1–5 only cohere under that choice, and benchmarking against the
wrong target is how this design gets unfairly called a failure.
