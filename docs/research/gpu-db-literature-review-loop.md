# GPU DB Literature Review Loop

## Purpose

Continuously review published research while GPU benchmarking is paused or
limited by hardware availability. The loop should identify, read, analyze, and
synthesize papers that may improve:

- write throughput
- read throughput
- concurrent session scale, with a long-range target of 1M logical sessions
- query latency
- MVCC, snapshot isolation, or comparable visibility models
- GPU execution, batching, micro-batching, and memory layout

The output is not a single predetermined architecture. It is a growing research
journal of candidate techniques, rejected ideas, and benchmarkable hypotheses.

## Source Files

- Candidate queue: `docs/research/gpu-db-paper-candidates.md`
- Journal: `docs/research/gpu-db-literature-journal.md`
- Runtime target context:
  `docs/architecture/11-high-throughput-query-runtime.md`
- P8 storage context:
  `docs/architecture/10-p8-gpu-optimized-storage-engine.md`
- Execution context:
  `docs/architecture/04-execution-model-cpu-gpu.md`
- Session/admission context:
  `docs/architecture/09-session-management-and-admission.md`

## Paper Selection Rules

Each run should process one paper unless the paper is unavailable, irrelevant,
or superseded by a better source found during triage.

Only papers from 2015 onward are eligible for new review. Prefer newer
literature first, especially 2023-present work, unless a slightly older paper is
clearly foundational for a current 2015-present mechanism. Pre-2015 papers may
remain in the journal as historical context if already reviewed, but the loop
must not select them for future runs.

The loop must preserve topic balance. The target is transaction-processing
database design first, with analytics/GPU papers included because they are
important to the engine, not because OLAP is the only goal. Do not process more
than two analytics/GPU-OLAP papers consecutively. When the recent journal skews
analytical, the next paper should come from OLTP/concurrency control,
MVCC/snapshot/storage, high-concurrency runtime/admission, HFT-style low-latency
systems, or query optimization/planning.

Prefer papers from:

- VLDB, SIGMOD, ICDE, CIDR, DaMoN
- OSDI, SOSP, ATC, EuroSys, NSDI, FAST
- arXiv preprints when the idea is directly relevant and no published version
  is obvious
- vendor or systems papers only when they expose enough technical detail to
  evaluate the design

Prefer topics that map directly to current GPU DB design questions:

- **transaction processing and write path**: single-writer or partition-owned
  mutation paths, commit protocols, log-structured ingest, append-only storage,
  checkpointing, replay, and write admission
- **MVCC and snapshots**: MVCC, snapshot isolation, virtual snapshots, HTAP
  snapshot routing, garbage collection, and serializable read designs
- **concurrency and sessions**: lock-free or low-contention concurrency
  control, admission control, high-concurrency network runtimes, and
  multiplexed session architectures
- **HFT/runtime mechanics**: cache-friendly queues, rings, preallocation,
  pinned buffers, low-allocation hot paths, and mechanical sympathy
- **GPU execution and batching**: GPU database scheduling, concurrent GPU query
  processing, memory residency, kernel-launch amortization, grouped lookups,
  grouped aggregates, and result scattering
- **query optimization**: CPU/GPU route choice, learned or adaptive cost models,
  predicate pushdown/transfer, join planning, and workload-aware execution
- **hybrid HTAP**: approaches that balance transactional writes and analytical
  reads without starving either side

## Balance Targets

The candidate queue and journal should keep a healthy mix:

- transaction processing / write path: at least 20%
- MVCC / snapshot / visibility: at least 20%
- runtime, HFT-style mechanics, concurrency, and session scale: at least 20%
- GPU execution / analytics / over-resident execution: at most 30% unless
  explicitly requested for a GPU-specific phase
- query optimization and hybrid HTAP: fill the remaining mix and break ties
  toward underrepresented topics

These are directional targets, not rigid quotas. They exist to prevent the
research loop from optimizing only for analytical scans when the product target
also includes transaction processing.

## Processing Protocol

For each paper:

1. Record title, authors, venue/year, URL/DOI/arXiv id, and retrieval date.
2. Classify its category and primary relevance:
   - transaction processing / write path
   - MVCC / snapshot / visibility
   - runtime / HFT / session scale
   - GPU execution / analytics
   - query optimization / planning
   - hybrid HTAP
3. Summarize the key idea in a few paragraphs.
4. Extract concrete mechanisms, not just high-level claims.
5. Map each mechanism to the current GPU DB architecture:
   - owner domains
   - read snapshots
   - command/response rings
   - GPU execution workers
   - partition/residency model
   - WAL/MVCC/write path
   - session admission/backpressure
6. List risks and mismatches.
7. Produce benchmark candidates:
   - what to implement
   - expected improvement
   - required measurement
   - minimum proof gate
   - failure condition
8. Update the candidate queue:
   - mark the processed paper as reviewed
   - add follow-up papers discovered from citations or related work
9. Update the journal with a dated entry.

## Synthesis Rules

The loop should accumulate patterns across papers. Every few reviewed papers,
add a synthesis entry that compares approaches and names the most promising
benchmark tracks.

Promising tracks should be expressed as implementation hypotheses, for example:

- retained same-shape lookups can be micro-batched by snapshot generation and
  key vector
- write admission should append into preallocated chunk buffers and publish
  visibility at deterministic generation boundaries
- read-heavy workloads should use immutable retained snapshots instead of
  routing through the mutation owner
- GPU execution owners should own CUDA streams and reusable pinned buffers
- network IO should use bounded multiplexed workers and response rings rather
  than thread-per-client processing

## Safety And Quality Bar

- Do not treat a paper as actionable until its mechanism is understood.
- Do not conflate benchmark-only shortcuts with production-safe architecture.
- Do not weaken WAL-before-visibility, SQL result correctness, or invalidation
  semantics to gain throughput.
- Prefer primary sources. Blog posts can inform terminology but should not be
  the main source for a journal entry unless no paper exists.
- If web search is unavailable, use direct URLs, arXiv, ACM/DBLP pages, or
  previously seeded candidates.

## Reporting

Each successful run should leave the repo with:

- one new or updated journal entry
- candidate queue status updated
- any newly discovered follow-up candidates
- a commit on `main`

The visible summary should be short: paper processed, strongest transferable
idea, and next candidate.
