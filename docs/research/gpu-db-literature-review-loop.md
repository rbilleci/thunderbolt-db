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

Prefer papers from:

- VLDB, SIGMOD, ICDE, CIDR, DaMoN
- OSDI, SOSP, ATC, EuroSys, NSDI, FAST
- arXiv preprints when the idea is directly relevant and no published version
  is obvious
- vendor or systems papers only when they expose enough technical detail to
  evaluate the design

Prefer topics that map directly to current GPU DB design questions:

- single-writer or partition-owned mutation paths
- MVCC, snapshot isolation, virtual snapshots, HTAP snapshot routing, or
  low-overhead garbage collection
- lock-free or low-contention concurrency control
- high-concurrency network runtimes and admission control
- GPU database scheduling, concurrent query processing, memory residency, and
  kernel-launch amortization
- query batching, micro-batching, grouped lookups, grouped aggregates, and
  result scattering
- log-structured ingest, append-only storage, checkpointing, and replay
- cache-friendly queues, rings, preallocation, pinned buffers, and mechanical
  sympathy

## Processing Protocol

For each paper:

1. Record title, authors, venue/year, URL/DOI/arXiv id, and retrieval date.
2. Classify its primary relevance:
   - write throughput
   - read throughput
   - concurrency/session scale
   - latency
   - MVCC/snapshot design
   - GPU execution/batching
   - data layout/storage
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
