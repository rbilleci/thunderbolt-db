# GPU DB Literature Journal

This journal synthesizes research papers for GPU database engine design. It is
maintained by the scheduled literature-review loop while GPU benchmarking is
paused or limited.

## Current Research Goals

- Maximize write throughput without weakening WAL-before-visibility or
  SQL-visible MVCC semantics.
- Maximize retained and over-resident read throughput.
- Prepare a runtime architecture that can plausibly scale toward 1M logical
  concurrent sessions through multiplexed IO, bounded queues, and admission
  control.
- Minimize query latency while still using GPU batching where it pays.
- Evaluate MVCC, snapshot isolation, virtual snapshots, RCU-like retained read
  snapshots, or comparable visibility models for this engine.
- Produce benchmarkable implementation candidates rather than only prose.

## Standing Synthesis

The current architecture target, as of `docs/architecture/11-high-throughput-query-runtime.md`,
is:

- async/network IO workers instead of one OS thread per client
- bounded command and response rings
- explicit owner domains for mutation, catalog, residency, GPU execution, and
  optional partitions
- immutable retained read snapshots for concurrent read execution
- GPU execution workers that own CUDA streams, events, pinned buffers, and
  scratch buffers
- micro-batching under latency ceilings for same-shape lookups, aggregates,
  COPY admission, refresh, and response metadata
- explicit backpressure at each queue or memory boundary

Each paper entry should either strengthen, revise, or reject part of that
target.

## Reviewed Papers

### 2026-06-02 - Scalable and Robust Snapshot Isolation for High-Performance Storage Engines

**Citation:** Adnan Alhomssi and Viktor Leis. "Scalable and Robust
Snapshot Isolation for High-Performance Storage Engines." PVLDB 16(6),
2023, pp. 1426-1438. doi:10.14778/3583140.3583157. Retrieved
2026-06-02 from `https://www.vldb.org/pvldb/vol16/p1426-alhomssi.pdf`.

**Category:** MVCC / snapshot / visibility.

**Relevance tags:** snapshot isolation; MVCC visibility; long-reader
robustness; garbage collection; tombstones; buffer-managed storage; HTAP;
write path; WAL and recovery.

**Core idea:** The paper shows that ordinary MVCC snapshot isolation can
still let a single long-running read collapse OLTP throughput because old
versions and tombstones remain physically on the hot access path. The authors
argue that robust HTAP needs more than "readers do not block writers": the
commit protocol, version storage, tombstone tracking, and garbage collection
must make old snapshots cheap for current transactions to ignore.

Their LeanStore design combines three mechanisms. Ordered Snapshot Instant
Commit (OSIC) gives buffer-managed engines instant commit without revisiting
the write set while retaining cheap visibility checks. The Graveyard Index
moves tombstones that only long-running OLAP snapshots can still see out of
the main OLTP index. Adaptive version storage keeps ordinary old versions in
per-worker delta indexes, but converts frequently updated tuples to an inline
FatTuple format so long-running scans do not traverse unbounded chains. In the
paper's main robustness claim, LeanStore sustains about 2 million TPC-C
transactions per second on a 64-core server while a long-running OLAP scan is
active, with logging enabled.

**Concrete mechanisms:**

- OSIC assigns every transaction a start timestamp and commit timestamp from a
  global logical clock. Each worker processes transactions sequentially and
  appends commit timestamps to a fixed-size per-worker Commit Log.
- Visibility uses the worker-local transitive commit invariant. For a version
  written by worker `w` with start timestamp `vts`, a reader with start
  timestamp `ts` treats it as visible iff `LCB(w, ts) > vts`, where `LCB` is
  the last commit timestamp by that worker before `ts`.
- Commit log entries are protected by per-worker mutexes while drawing and
  publishing commit timestamps, avoiding races where a reader misses a just
  committed transaction.
- Readers compute `LCB` lazily only for workers whose versions they actually
  encounter, and cache the answer per snapshot. This avoids constructing a
  full vector of in-progress transactions for every snapshot.
- The commit log is bounded by the number of workers; when full, redundant
  entries are removed while preserving entries that are the `LCB` for active
  snapshots.
- The system tracks OLTP and OLAP transaction watermarks separately. The
  oldest OLTP timestamp can advance even while a long OLAP snapshot remains
  open, enabling tombstone movement and precise pruning that a single global
  oldest-snapshot watermark would block.
- Every user index has a matching Graveyard Index. Tombstones no longer needed
  by OLTP transactions are moved from the main index to the graveyard, so OLTP
  range or queue-like lookups remain logarithmic in currently visible tuples.
- Long OLAP scans merge the main index with the Graveyard Index for the leaf
  range they scan, paying extra work only for old snapshots that may still see
  deleted tuples.
- Tombstone Indexes act as per-worker append-optimized todo lists keyed by the
  deleting transaction timestamp, so GC can range-scan tombstones that are
  ready to move or physically remove.
- The default version layout stores the latest version in the main index and
  older versions in per-worker Delta Indexes keyed by transaction metadata.
  Delta Indexes double as append-friendly version stores and GC todo lists.
- Frequently updated tuples are detected with a small per-tuple chain-length
  heuristic and converted to FatTuples, which inline versions near the latest
  value so precise GC and long-reader reconstruction avoid random off-row I/O.
- Before evicting pages containing FatTuples, LeanStore decomposes them back
  into the chained Delta Index format to avoid leaking old versions on cold
  pages.
- WAL remains the recovery source of truth. The auxiliary Commit Log,
  Graveyard Index, Tombstone Index, and Delta Index are rebuilt or truncated
  during recovery rather than treated as durable authority; FatTuple changes in
  main indexes are logged normally.
- The implementation uses first-writer-wins snapshot isolation. The paper
  notes serializability can be layered with known techniques, but does not
  implement serializable isolation.

**GPU DB mapping:** This is a direct warning for the current GPU DB snapshot
plan: immutable retained snapshots are not enough if old snapshot state remains
on the mutation owner's hot path. GPU-resident readers may be cheap, but their
CPU-side MVCC and index remnants can still degrade writes, deletes, queue-like
tables, prefix scans, and refresh work unless the engine separates "visible to
long readers" from "must be considered by fresh OLTP work."

OSIC maps cleanly to the owner-domain model in
`11-high-throughput-query-runtime.md`. A mutation owner or partition owner can
publish a monotonic generation/timestamp for committed batches, while each
owner keeps a compact commit-log summary for visibility checks. Retained read
snapshots should not require copying a large active-transaction vector per
session; the logical snapshot handle should be a timestamp plus cached
per-owner `LCB` answers discovered on demand.

The separate OLTP/OLAP watermarks are especially relevant to P8. The engine
should distinguish short transactional snapshots, retained read snapshots, and
long analytical or over-resident snapshots. A long GPU scan should not pin all
tombstones and obsolete row versions in hot CPU or GPU structures used by
fresh lookups. The equivalent of the Graveyard Index could be a cold old-
snapshot side structure for tombstones, deleted keys, and old resident segment
metadata that only long snapshots consult.

FatTuple suggests a concrete version-layout rule for mixed workloads:
ordinary rows can use append-friendly off-row deltas or WAL-backed version
records, but frequently updated rows that long snapshots repeatedly scan may
need an inline compact history window or a GPU-friendly delta bundle. This
could matter for hot counters, queue heads, account balances, and catalog or
metadata rows whose version chains would otherwise hurt snapshot reads.

The WAL/recovery separation reinforces the current architecture. GPU resident
snapshots, graveyard-like side structures, per-owner visibility summaries, and
delta indexes should be rebuildable acceleration state unless explicitly made
durable. Visibility may be published only after WAL safety, but the runtime can
still use auxiliary summaries to avoid expensive per-read or per-commit work.

**Risks and mismatches:** OSIC assumes worker threads process transactions one
after another and that each committed version records the writer worker and
start timestamp. A GPU DB with partition owners, async IO workers, and GPU
execution workers must define which owner identity participates in visibility;
using transient network workers would be wrong. The paper targets snapshot
isolation, not full serializability. Graveyard Indexes are described for
B-Tree indexes and row-store LeanStore, so a columnar/GPU-resident layout needs
different physical side structures. FatTuple may bloat hot pages, and the
paper observes extra page misses in out-of-memory experiments. Finally, the
reported throughput comes from LeanStore on CPU/NVMe, not GPU execution, so
the transferable claim is the visibility and GC shape, not the absolute
numbers.

**Benchmark candidates:**

- Add a visibility-summary experiment to the MVCC tuple store: compare current
  `Visibility { read_txn_id }` checks with a per-owner start/commit generation
  cache for retained reads. Minimum gate: identical visible tuple sets under
  insert/update/delete and WAL replay tests.
- Build a queue-like table benchmark modeled after TPC-C `neworder`: insert at
  one end, delete from the other, hold one long retained snapshot open, and
  measure fresh lookup/delete latency as tombstones accumulate. Failure
  condition: p95 lookup latency grows linearly with old tombstone count.
- Prototype an old-snapshot side structure for deleted keys in the CPU
  relational index layer. Fresh snapshots skip it; long retained snapshots
  merge it only when needed. Measure write overhead, fresh lookup latency, and
  long-snapshot scan correctness.
- Track separate snapshot classes in telemetry: short OLTP, retained GPU read,
  long analytical scan, refresh, and recovery. Expose oldest timestamp per
  class and use the gap between short and long readers to drive GC eligibility.
- Add a hot-row version-chain stress test with one long snapshot and repeated
  updates to a small key set. Compare off-row delta chains with an inline
  compact-history representation before considering GPU-resident encoding.
- For P8 resident snapshots, measure whether long GPU scans can pin CPU
  tombstones, deleted-key metadata, or resident segment generations long enough
  to harm write throughput. The proof gate is stable write/read latency while
  one long scan remains open.

### 2026-06-02 - Datacenter RPCs can be General and Fast

**Citation:** Anuj Kalia, Michael Kaminsky, and David G. Andersen.
"Datacenter RPCs can be General and Fast." NSDI 2019, pp. 1-16.
Retrieved 2026-06-02 from the USENIX publication page and PDF,
`https://www.usenix.org/conference/nsdi19/presentation/kalia`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** high-concurrency networking; session admission; bounded
buffers; zero-copy I/O; congestion control; polling runtimes; low-latency
replication; request scheduling.

**Core idea:** eRPC argues that a general-purpose RPC layer can reach
near-specialized datacenter networking performance without requiring RDMA
semantics, lossless fabrics, FPGAs, or programmable switches. The system wins
by optimizing the hot common case: small messages, short dispatch-mode
handlers, uncongested networks, and userspace packet I/O. More expensive paths
for large messages, retransmission, node failure, congestion, and long-running
handlers exist, but they are kept off the short request path whenever possible.

The most relevant result for the GPU DB runtime is that high session count and
low latency are framed as resource-budget problems rather than one-thread-per-
client problems. On the evaluated 100-node lossy Ethernet cluster, eRPC reports
about 10 million small RPCs per second processed by one core in its symmetric
benchmark, maintains peak performance with about 20,000 connections per node,
and keeps 99.99th percentile latency below 700 microseconds in the large
session experiment. It also ports existing Raft and Masstree code, showing that
the network fast path can be general enough to reuse higher-level database and
replication logic.

**Concrete mechanisms:**

- Each user thread owns an `Rpc` endpoint with RX/TX queues, an event loop, and
  multiple sessions. The event loop performs packet I/O, congestion control,
  management work, request-handler invocation, and completion callbacks.
- Sessions are one-to-one connections between two `Rpc` endpoints, not between
  OS processes. Each session supports a bounded number of outstanding requests
  with slot metadata, and additional requests are queued by the library.
- Short handlers run directly on dispatch threads to avoid inter-thread
  communication; long handlers can be marked for worker-thread execution so
  they do not block packet processing or congestion feedback.
- eRPC uses DMA-capable message buffers with a layout optimized for small
  single-packet messages: the first packet header and data are contiguous so the
  NIC can fetch them with one DMA read, while the application still sees a
  contiguous data region.
- Zero-copy transmission is protected by explicit ownership rules. The library
  avoids signaled sends on the common path, but flushes the TX DMA queue during
  retransmission or failure handling before returning a request buffer to the
  application.
- For common-case single-packet requests handled in dispatch mode, the server
  can run the handler over the received packet buffer before returning that
  buffer to the NIC receive queue, avoiding a dynamic message-buffer copy.
- Sessions use packet credits. A client consumes credits when sending packets
  and regains them from responses or explicit credit-return packets. The credit
  count limits receive-queue pressure and implements end-to-end flow control.
- eRPC deliberately uses packet I/O instead of RDMA writes because packet
  receive completion queues scale better than polling many per-client memory
  locations, and CPU-managed connection state avoids NIC SRAM connection-cache
  limits.
- Congestion control is optimized for the uncongested case. eRPC uses Timely-
  style RTT measurement and Carousel-style software rate limiting, but bypasses
  rate updates and the rate limiter when a session remains below the low RTT
  threshold. It batches timestamp reads to reduce per-packet overhead.
- Packet loss is handled at the client with go-back-N rollback of wire-protocol
  state and retransmission. The server is designed to avoid running a request
  handler twice, preserving at-most-once RPC semantics.
- Node failure handling flushes TX queues, drains or rejects rate-limited
  packets, invokes pending client continuations with errors, and frees server
  resources after outstanding handlers finish or no longer need them.
- Evaluation attributes much of the throughput to common-case details:
  disabling congestion-control optimizations, preallocated responses, and
  zero-copy request processing reduces small-RPC throughput materially.

**GPU DB mapping:** eRPC reinforces the target in
`11-high-throughput-query-runtime.md`: logical session scale should be
multiplexed through a small number of IO workers and bounded owner queues, not
through one OS thread per pgwire client. The direct mapping is an internal
request/response datapath where network workers own socket readiness and frame
parsing, while mutation owners, read snapshot workers, and GPU execution owners
receive typed work through fixed-capacity rings.

The session credit design maps cleanly to GPU DB admission. A logical pgwire
session should have bounded credits for active frontend messages, decoded COPY
chunks, retained read requests, response buffers, and possibly GPU staging
slots. Idle logical sessions can remain cheap, but they should not imply a
right to consume pinned buffers, owner queue entries, or GPU work slots. That is
the path toward 1M logical sessions without pretending that 1M requests can be
simultaneously active.

The dispatch-versus-worker split is also important. Short retained reads that
only enqueue a snapshot-compatible GPU or cached response request should stay
on a low-latency path. Long COPY admission, refresh, over-resident scans, CPU
fallback, or transactional validation should move to worker or owner domains
without blocking IO progress or response writes. The GPU DB equivalent of
eRPC's handler annotation is a route descriptor with expected duration, bytes
moved, queue budget, snapshot generation, and whether it can safely run in a
fast dispatch-like path.

For buffer management, eRPC's message-buffer ownership rules are directly
transferable. Pgwire input buffers, decoded COPY chunks, WAL/MVCC batch
buffers, CUDA pinned staging buffers, and encoded response buffers need explicit
states and completion ownership. A response or COPY buffer should not be reused
just because SQL execution has returned; it must also be free of NIC writes,
WAL/MVCC ownership, GPU execution ownership, and response-ring references.

The congestion-control result suggests that GPU DB overload management should
measure queue delay and saturation at every boundary. For intra-process rings,
the analog of Timely's RTT is queue wait plus service time. For pgwire network
IO, RTT-like feedback may be less directly available, but response-ring depth,
socket writability, and per-session credit exhaustion can still drive
admission and shedding decisions.

**Risks and mismatches:** eRPC is an RPC library, not a SQL database runtime.
It does not solve WAL-before-visibility, MVCC validation, snapshot publication,
catalog invalidation, GPU residency, SQL planning, or PostgreSQL protocol
details. Its strongest numbers rely on userspace NIC access and polling; the
current benchmark endpoint is ordinary TCP/pgwire and may not be able to
replicate those latencies without a larger transport change. eRPC's session
model is endpoint-to-endpoint between user threads, while pgwire sessions have
authentication, transactions, prepared statements, portals, COPY state, and
error recovery. The evaluated 20,000 sessions per node is useful evidence but
still far below a 1M logical-session target. Worker-thread dispatch also needs
care: moving long work off IO threads prevents head-of-line blocking, but too
many worker queues can recreate the same scheduling and memory pressure the
design is supposed to avoid.

**Benchmark candidates:**

- Add per-session active-credit accounting to the pgwire benchmark endpoint:
  parsed frontend messages, in-flight engine requests, decoded COPY chunks,
  response buffers, and retained/GPU work slots. Minimum gate: identical SQL
  behavior with explicit overload reasons when a credit class is exhausted.
- Replace the exact-response retained cache path with a bounded
  request/completion-handle prototype for one retained read route. Measure
  owner queue wait, response-buffer reuse, copies, and p50/p99 latency at
  concurrency `1,2,4,8,16,32,64`.
- Build a no-benchmark session-scale memory probe that allocates compact
  logical pgwire session state for large counts while admitting only a bounded
  active subset. Required metrics: bytes per idle session, active credit memory,
  queue depth, and rejection/fallback counts.
- Add dispatch-versus-worker route classification telemetry: short retained
  read, mutation owner, COPY chunk, refresh, over-resident scan, CPU fallback,
  or error path. Failure condition: long work can still block network IO or
  response writes.
- Prototype explicit buffer states for decoded COPY chunks and encoded
  responses: free, filling, submitted, owner-in-flight, GPU/network-in-flight,
  completed, reusable, retired. Proof gate: no buffer reuse before all owning
  domains have completed, with saturation telemetry under concurrency.
- Compare fixed per-session request limits against BDP/queue-depth-like credits
  for retained reads and COPY admission. Expected improvement: bounded memory
  and lower queue tail latency under high logical concurrency. Failure
  condition: single-session throughput regresses when there is no saturation.

### 2026-06-02 - Concurrent Analytical Query Processing with GPUs

**Citation:** Kaibo Wang, Kai Zhang, Yuan Yuan, Siyuan Ma, Rubao Lee, Xiaoning
Ding, and Xiaodong Zhang. "Concurrent Analytical Query Processing with GPUs."
PVLDB 7(11), 2014. Retrieved 2026-06-02 from
`https://www.vldb.org/pvldb/vol7/p1011-wang.pdf`.

**Relevance tags:** GPU execution/batching; read throughput; concurrency/session
scale; latency; data layout/storage; admission/backpressure.

**Core idea:** GPU analytical queries often reserve far more device memory than
they actively use and leave copy engines and compute units idle during CPU
phases, dependencies, and operator transitions. Instead of treating a GPU as a
dedicated accelerator for one query, MultiQx-GPU lets compatible analytical
queries make concurrent progress while a database-level scheduler and
device-memory manager prevent resource conflicts from collapsing throughput.

The reported benefit is throughput rather than single-query latency isolation:
for 69 SSB query pairs whose peak device memory exceeded device capacity,
MultiQx-GPU improved throughput by 39% on average and up to 55% over dedicated
YDB execution at scale factor 14. At scale factor 28, the paper reports 33%
average and 54% maximum improvement. The authors also show why naive
co-running is unsafe: without scheduling, four concurrent queries dropped
throughput to one quarter of the optimum and 65% below one-at-a-time execution
in their experiment.

**Concrete mechanisms:**

- Query scheduling is modeled as admission control over effective GPU resource
  demand, not peak reservation. The paper defines weighted device-memory demand
  as a time-weighted average over the query's operator sequence, where each
  operator contributes its execution time and the maximum device memory
  referenced by any kernel in that operator.
- The scheduler dynamically updates a running query's weighted demand as phases
  change. A postponed query is reconsidered whenever a running query changes
  demand or finishes.
- The device-memory manager intercepts GPU API calls in application space. It
  returns host-side virtual regions first and allocates device memory only when
  a kernel that references the region is about to run.
- Lazy transfer keeps copied input in a host swapping buffer until the data is
  actually needed by a kernel. A copy-on-write optimization can avoid copying
  from the swapping buffer when the original source buffer has not changed.
- Large GPU regions are tracked as fixed-size logical pages for coherence, so
  partial updates and canceled evictions do not force whole-region movement.
- Kernel-level data reference and access advice tells the memory manager which
  regions the next kernel will touch and whether each region is input, output,
  or both. That avoids loading or preserving data that the kernel cannot read
  later.
- Cost-driven replacement (CDR) chooses eviction victims using a cost formula
  that combines bytes that must be copied back, region size, LRU position, and
  a latency factor. In the paper's selected co-running workloads, CDR improves
  throughput 44% on average and 56% at maximum versus LRU.
- The prototype uses `LD_PRELOAD` to intercept CUDA calls, a shared-memory area
  for cross-query state, POSIX message queues for replacement requests, and an
  8 MB logical page size that the paper says preserved over 99% PCIe efficiency.

**GPU DB mapping:** This strongly supports the current plan in
`11-high-throughput-query-runtime.md` to treat GPU execution workers and
residency state as explicit owner domains with bounded admission. The most
transferable idea is not full transparent swapping; it is phase-aware admission
based on measured effective resource demand. For this engine, retained read
workers should expose per-shape demand vectors such as resident bytes touched,
scratch bytes, pinned staging bytes, expected H2D/D2H bytes, kernel count, and
historical execution time. The scheduler can then co-run same-generation reads
only while the sum of effective demand stays inside per-GPU memory, copy-engine,
and scratch-buffer budgets.

The access-advice mechanism maps naturally onto planner route metadata. A
retained route already knows selected columns, predicate columns, aggregate
scratch, output shape, and whether it needs H2D key vectors or only device-side
resident columns. Making that metadata explicit before enqueueing work would
let the GPU execution owner decide whether a query is eligible for immediate
execution, micro-batching, CPU fallback, or overload rejection.

For P8 storage, the paper argues against over-relying on peak resident memory as
the scheduler's sole gate. Published immutable resident snapshots can reserve a
large table or partition while individual query shapes touch only a subset of
columns and scratch. Admission should count the snapshot's fixed residency
budget separately from each request's incremental execution footprint.

**Risks and mismatches:** MultiQx-GPU targets long-running analytical SSB
queries in a process-per-query YDB-style runtime, not SQL-visible pgwire
sessions, MVCC reads, or write-heavy workloads. Its transparent device-memory
swapping could harm the current engine's low-latency retained paths if it
reintroduced unpredictable H2D/D2H movement after residency has already been
published. The paper does not solve WAL-before-visibility, snapshot
compatibility, DDL invalidation, or response-ring backpressure. It also reports
throughput benefits under two-way analytical co-running, not million-session
network concurrency.

**Benchmark candidates:**

- Implement a retained-read demand estimator for existing partitioned retained
  routes: resident bytes referenced, scratch bytes, H2D key/predicate bytes,
  D2H result bytes, kernel count, and last observed CUDA time. Gate pass/fail on
  truthful telemetry with no route correctness changes.
- Compare fixed concurrency caps with weighted-demand admission for same-shape
  retained reads at concurrency `1,2,4,8,16,32,64`. Expected improvement:
  higher throughput once mixed query shapes arrive without increasing p95 queue
  wait. Failure condition: p50 latency regresses for single-request execution or
  admission hides overload.
- Add planner/executor access advice for retained routes and measure whether
  scratch/pinned-buffer reuse and route grouping reduce H2D/D2H bytes and launch
  overhead. Minimum proof gate: output per request includes the predicted versus
  observed demand vector.
- Stress incompatible retained query mixes to find the concurrency point where
  effective-demand admission must reject, fallback to CPU, or drain smaller
  micro-batches. This should be measured without adding transparent GPU memory
  swapping first.

### 2026-06-02 - Data Path Fusion in GPU for Analytical Query Processing

**Citation:** Tsuyoshi Ozawa and Kazuo Goda. "Data Path Fusion in GPU
for Analytical Query Processing." arXiv:2605.10511, submitted 2026-05-11.
Retrieved 2026-06-02 from `https://arxiv.org/pdf/2605.10511`. The
preprint contains PVLDB placeholder metadata; final venue details are unknown.

**Relevance tags:** GPU execution/batching; read throughput; query latency;
data layout/storage; GPU IO; compression; retained/over-resident execution.

**Core idea:** Data Path Fusion argues that GPU analytical engines lose a large
part of their advantage when IO, decompression, and relational operators are
split into separate host-orchestrated GPU kernels. DPF instead makes a fused
CUDA kernel the data-path unit: the kernel issues GPU-initiated storage reads,
decompresses pages, and runs filters, hash probes/builds, or aggregation before
returning control to the host.

The most important shift for this engine is treating "resident read execution"
as a fully described pipeline rather than a sequence of local optimizations.
DPF's gains come from removing host-device synchronization boundaries,
eliminating intermediate materialization between stages, and making compression
formats match GPU thread-level decompression. In the reported representative
configuration, DPF improves end-to-end response over a GOLAP-like baseline by
2.66x to 6.22x on selected TPC-H queries and 3.84x to 16.81x on selected SSB
queries.

**Concrete mechanisms:**

- Each fused kernel combines BaM-based GPU-initiated IO, page decompression, and
  one or more relational operations. The host still generates the query plan and
  launches kernels, but does not orchestrate each IO/decompression/operator
  stage.
- A GPU-side pruning stage uses dictionaries and zone maps to produce pruned
  page lists before the fused query kernels run.
- Thread blocks advance through IO, decompression, and operation stages with
  synchronization barriers inside the kernel. The IO stage can dedicate a subset
  of warps to BaM request submission and completion polling, while all threads
  participate in decompression and operator work.
- Kernel launch configuration is tuned per query. The paper describes using
  one block per SM with 128 threads for typical cases, and larger blocks such as
  1,024 threads when more IO warps or compute parallelism are useful.
- Column grouping is a scheduling choice. Related columns from the same table
  can be read and decompressed together so decoded values co-reside in shared
  memory where possible, avoiding global-memory intermediates before operator
  execution.
- Fixed-length integer columns use GPU-FOR-style mini-block compression, with
  metadata for base value, bit width, and byte offset. Short fixed strings are
  reinterpreted as integers; longer strings use the variable-length path.
- Variable-length strings use FSST per page plus embedded row ids and an
  auxiliary RID prefix-sum index so kernels can locate the page for a row id and
  align pages across columns.
- The loader sorts rows, builds zone maps, writes compressed column pages in two
  passes, and records page offset, page size, and row-count prefix arrays. The
  paper reports additional loading cost within 1.5% versus the baseline layouts
  it evaluates.
- Evaluation isolates components: BaM alone improves TPC-H query response by
  1.12x to 1.51x versus CPU-initiated GDS in their setup, kernel fusion adds
  mixed effects on TPC-H but 1.17x to 3.97x on SSB, and type-specific
  compression provides the largest additional speedup in many cases by reducing
  IO volume and making decompression fine-grained.

**GPU DB mapping:** DPF is most directly applicable to the P8 retained and
over-resident read path, not the current SQL-visible write path. Existing
partitioned retained routes already avoid repeated H2D recopy for resident
columns, but they are still shaped as separate planning, staging, kernel, and
materialization steps. The transferable direction is a route descriptor that
can generate a single fused retained kernel for a stable query family: load key
or predicate vectors if needed, touch resident column groups, evaluate
visibility/predicate logic, reduce or scatter results, and write compact
per-request output buffers.

For over-resident tables, DPF supports a stronger version of the current P8
partition model: partition-local compressed column pages plus zone maps can be
fed by GPU-initiated IO only for pages whose min/max metadata survives pruning.
That would give the engine an explicit third read mode between fully resident
device memory and CPU fallback: GPU-in-data-path cold or warm page execution.
It should remain subordinate to SQL visibility, so the page metadata would need
relation identity, schema generation, source WAL boundary, and visibility
boundary just like resident snapshots.

The compression mechanism also maps to P8's `int4`/`text` first slice. Dense
`int4` resident buffers are good for today's kernels, but over-resident
execution should benchmark GPU-FOR-like compressed page groups for scan-heavy
aggregates. The string/RID design is relevant to future `text LIKE 'prefix%'`
routes because it makes variable-length pages independently decodable by GPU
thread blocks.

**Risks and mismatches:** DPF is an analytical engine prototype, not a
PostgreSQL-compatible MVCC serving runtime. It does not address WAL-before-
visibility, DDL invalidation, snapshot retirement, pgwire response rings, or
session concurrency. The fused kernels can become query-family-specific and may
increase implementation complexity compared with the current narrow retained
route kernels. BaM also assumes raw block-device access and GPU-initiated IO;
that may not be available or desirable in the first production deployment.
Build-side joins are limited by GPU memory, the prototype supports only selected
compression schemes, and general decimal/floating-point support is incomplete.

**Benchmark candidates:**

- Build a "route-fusion" telemetry proof for one retained aggregate family:
  count host/device synchronization points, kernel launches, intermediate
  buffers, H2D/D2H bytes, and CUDA elapsed time before and after combining
  predicate, visibility, and reduction into fewer kernels. Minimum gate:
  identical SQL-visible result and lower launch/materialization count.
- Add compressed resident-page experiments for `int4` aggregate scans using a
  GPU-FOR-like page group beside the current dense resident layout. Expected
  improvement: lower memory traffic or over-resident IO volume. Failure
  condition: decompression overhead loses to dense scans at current row counts.
- Prototype partition-local zone-map pruning for over-resident read planning:
  route only candidate pages/partitions to GPU execution and record pruned
  bytes, touched bytes, and result correctness. Minimum gate: stable pruning
  metadata tied to source WAL and schema generation.
- For future `text` retained routes, test an FSST/RID-index page layout against
  the current offsets-plus-bytes resident representation for prefix predicates.
  Measure decode bandwidth, output scatter cost, and memory footprint.
- Treat GPU-initiated IO as a later-stage benchmark, not an immediate
  dependency. First compare fused retained kernels over already-resident data;
  then evaluate GPUDirect/BaM-style over-resident execution only if storage IO
  becomes the measured bottleneck.

### 2026-06-02 - Scaling GPU-Accelerated Databases beyond GPU Memory Size

**Citation:** Yinan Li, Bailu Ding, Ziyun Wei, Lukas M. Maas, Momin
Al-Ghosien, Spyros Blanas, Nicolas Bruno, Carlo Curino, Matteo Interlandi,
Craig Peeper, Kaushik Rajan, Surajit Chaudhuri, and Johannes Gehrke. "Scaling
GPU-Accelerated Databases beyond GPU Memory Size." PVLDB 18(11), 2025,
pp. 4518-4531. DOI: `10.14778/3749646.3749710`. Retrieved 2026-06-02 from
`https://www.vldb.org/pvldb/vol18/p4518-li.pdf`.

**Relevance tags:** over-resident execution; GPU execution/batching; read
throughput; query latency; data layout/storage; planner cost hooks; CPU/GPU
placement.

**Core idea:** The paper argues that a single GPU can still accelerate
databases far larger than GPU memory if the system stops treating the GPU as
the default scan engine. For over-resident analytical workloads, PCIe bandwidth
is often slower than CPU compressed-column scans, while GPU joins and other
compute-heavy operators can still beat CPU execution even after transfer cost.
The proposed hybrid design therefore filters aggressively on the CPU, preserves
compressed representation after filtering, transfers only reduced compressed
columns over PCIe, and executes compute-heavy subplans on the GPU.

The evaluation integrates these ideas into a custom Microsoft SQL Server plus
TQP GPU engine prototype. On a 24-core A100 VM, the hybrid system runs all 22
TPC-H queries at 1 TB, where the GPU-only TQP baseline runs only 4 without OOM
and HeavyDB runs 9. The paper reports 3.5x overall speedup over SQL Server at
1 TB, with per-query speedups from 0.8x to 9.1x. At 100 GB cold runs, hybrid
execution reduces the GPU hot/cold gap and reports 3.5x speedup over SQL
Server, versus 2.4x for GPU-only cold TQP.

**Concrete mechanisms:**

- Query plans are split at a coprocessor operator. CPU scan operators produce
  filtered compressed inputs for GPU subplans; the GPU engine decompresses,
  executes joins or aggregates, and returns usually small results to the host.
- The scan operator evaluates predicates on the CPU but compacts projected
  columns directly in their compressed format, avoiding CPU decompress plus
  recompress overhead before PCIe transfer.
- For bit-packed values, compaction uses x86 BMI `PEXT`/`PDEP` style bit
  gather/scatter over all values that fit in a 64-bit word. For RLE, it counts
  selected values per run with population count. For dictionary encoding, it
  compacts dictionary indexes and can remove unused dictionary payload entries
  while preserving index positions.
- Predicate filters are not applied blindly. Expensive or weakly selective
  filters may be skipped on CPU and left for GPU execution when the CPU cost
  is not expected to repay transfer savings.
- Bitvector filters propagate selective predicates across equi-joins from
  smaller filtered inputs to larger base-table scans. The scan treats bitvector
  probes as additional predicates, with SIMD optimization.
- Candidate bitvector filters are derived from join-column lineage, then
  selected greedily by estimated benefit. The estimate balances CPU cost to
  build/probe filters and expected PCIe transfer reduction, while stopping once
  a table's estimated selectivity is below a threshold.
- Bitvector representation prioritizes probe throughput over perfect filtering:
  simple bitmaps are preferred for small domains, while cache-sized hash
  bitvectors can trade false positives for faster probing on large domains.
- Streaming processes large filtered inputs in chunks when an operator's state
  fits GPU memory. Partitioning repeatedly scans with partition predicates so
  each join partition pair fits on the GPU; it solves memory capacity but does
  not by itself reduce PCIe traffic.

**GPU DB mapping:** This is directly relevant to the current P8 over-resident
gap. The strongest transferable idea is an explicit three-way route choice:
resident GPU for hot valid snapshots, CPU scan/filter for cold or
over-resident reduction, and GPU execution only for the reduced compute-heavy
tail. For our current retained aggregate and lookup route family, over-resident
planning should not default to "stream full partitions to GPU." It should first
ask whether CPU-visible MVCC/columnar state can produce a compact candidate
vector, partition row-id list, or compressed value batch that is smaller than
the original resident page set.

The compressed-output scan maps to a future CPU canonical or derived-column
layout beside the current row/MVCC source. P8 currently generates dense GPU
resident column buffers from CPU truth. For over-resident tables, a compressed
CPU segment format with direct compaction could let the planner transfer only
selected `int4` value vectors, row ordinals, or join-key batches to GPU workers.
That complements DPF: DPF says fuse IO/decompression/operator stages when the
GPU owns the data path; this paper says let CPU memory bandwidth and SIMD prune
first when PCIe is the dominant boundary.

The filter-selection rule also maps cleanly onto route telemetry in
`11-high-throughput-query-runtime.md`. A route descriptor should expose CPU
filter cost, estimated selectivity, compressed bytes before and after filtering,
H2D bytes avoided, GPU work introduced, queue wait, and fallback reason. That
would let the runtime choose between immediate CPU fallback, CPU-prefilter plus
GPU tail, retained-GPU execution, or overload rejection without hiding the
reason.

For MVCC, the paper does not give a visibility design, but the mechanism can be
made compatible if CPU-side filtering reads from a stable visibility boundary
and carries source WAL/catalog generation into the compressed batch handed to
the GPU execution owner. Bitvector and predicate-transfer filters must be tied
to the same snapshot generation as the scanned partitions; otherwise they can
incorrectly discard rows that became visible after the filter was built.

**Risks and mismatches:** The prototype is analytical and SQL Server-based; it
does not address PostgreSQL-compatible pgwire serving, WAL-before-visibility,
snapshot retirement, DDL invalidation, write throughput, or 1M logical
sessions. The results are TPC-H warm-main-memory runs on A100/H100 cloud VMs,
not low-latency point lookups or mixed OLTP/HTAP writes. CPU-side filtering can
increase CPU contention exactly where this engine also needs network IO,
mutation admission, and MVCC maintenance. The greedy filter model ignores
correlation and cascading effects, and compressed CPU segments would require a
new storage/layout path beyond today's row/MVCC source plus dense retained
buffers.

**Benchmark candidates:**

- Add an over-resident planner experiment for partitioned `order_line`: CPU
  prefilter candidate row ordinals for one selective `int4` predicate, transfer
  only selected values/ordinals to a GPU aggregate kernel, and compare against
  full-partition GPU streaming and CPU-only execution. Minimum gate: identical
  SQL-visible result at one MVCC boundary with measured CPU filter time and
  H2D bytes avoided.
- Prototype a compressed `int4` segment sidecar for one generated benchmark
  column and implement direct selected-value compaction into a GPU staging
  buffer. Expected improvement: lower H2D bytes for over-resident scans.
  Failure condition: CPU compaction time exceeds saved transfer time.
- Add filter-decision telemetry to retained/over-resident route planning:
  estimated selectivity, CPU filter cost, compressed input bytes, filtered
  bytes, selected row count, GPU-tail cost, and chosen route. Proof gate:
  estimates and observations are recorded for accepted and rejected routes.
- Evaluate bitvector prefiltering for a future join-shaped benchmark before
  implementing full GPU joins: build a CPU bitvector from a filtered dimension
  key set, probe the large fact partition during scan, and measure reduced H2D
  bytes, false positives, and CPU overhead.
- Keep partitioned over-resident execution as a capacity mechanism, but require
  each partition route to report whether partitioning reduced peak GPU memory
  only or also reduced PCIe bytes through predicate/bitvector filtering.

### 2026-06-02 - Transaction Repair for Multi-Version Concurrency Control

**Citation:** Mohammad Dashti, Sachin Basil John, Amir Shaikhha, and Christoph
Koch. "Transaction Repair for Multi-Version Concurrency Control." SIGMOD 2017,
pp. 235-250. DOI: `10.1145/3035918.3035919`. Retrieved 2026-06-02 from the
ACM DOI page and the authors' CoRR preprint, "Repairing Conflicts among MVCC
Transactions," `https://arxiv.org/abs/1603.00542`.

**Category:** transaction processing / write path and MVCC / visibility.

**Relevance tags:** MVCC validation; optimistic concurrency control; conflict
repair; write contention; serializability; long transactions; transaction
program dependencies.

**Core idea:** MV3C targets the cost of optimistic MVCC abort-and-restart under
high contention or long-running transactions. Instead of discarding all work
when validation fails, transaction programs are represented as dependency
graphs of predicates and closures. Validation identifies which predicates read
stale committed versions; repair prunes only those invalid predicates and their
descendants, removes versions they created from the transaction undo buffer,
assigns a new start timestamp, and re-executes only the affected closures.

The conceptual fit for this engine is narrow but important: not every conflict
should force redoing parse, admission, CPU-side transaction work, generated
write batches, or GPU-facing refresh preparation. If a future write path can
name the exact read predicates and derived write fragments that depend on
them, then a failed validation can become a bounded repair of the affected
fragment rather than a whole transaction restart.

**Concrete mechanisms:**

- MV3C builds on optimistic timestamp-order MVCC. A transaction reads at a
  start timestamp, creates private versions, and validates before commit
  against versions committed during its lifetime.
- Transaction programs are annotated as predicates with bound closures.
  Predicates perform reads and expose a `match` operation used during
  validation; closures contain deterministic program logic and may instantiate
  child predicates.
- The predicate graph captures dependency direction. If a parent predicate is
  invalid, its descendants are invalid because their context variables or
  result sets may have changed.
- Validation topologically walks the predicate graph, matching predicates
  against committed versions since the transaction's start timestamp. It keeps
  both valid nodes and invalid nodes instead of stopping at the first conflict.
- Repair chooses a new start timestamp, selects invalid roots that have no
  invalid parent, removes versions created by those predicates and descendants
  from the undo buffer, prunes descendants, and re-executes the invalid roots'
  closures at the new timestamp.
- Write-write conflicts can optionally be allowed to continue instead of
  causing premature abort. Validation then decides whether the write was
  effectively blind or depended on a stale read.
- Attribute-level validation narrows false conflicts by intersecting columns
  monitored by a predicate with columns modified in a committed version before
  running predicate-specific matching.
- For expensive predicates such as non-indexed scans, MV3C can keep predicate
  result sets and repair them by incorporating concurrently committed matching
  versions rather than rescanning from scratch.
- The paper proves commit-order serializability for MV3C schedules, assuming
  deterministic closures and correct predicate dependency boundaries.
- Evaluation is single-threaded with interleaved transaction windows, not a
  production multicore engine. It shows low conflict-free overhead under 1% in
  reported cases, stronger gains as conflict rate/window size rises, and little
  benefit on one TPC-C configuration where conflicts mostly cause premature
  aborts before validation.

**GPU DB mapping:** The immediate mapping is not to port MV3C wholesale. The
engine's first production invariant is still WAL-before-visibility, and current
COPY/INSERT admission is more append/batch oriented than stored-procedure
oriented. The transferable design is dependency-named validation and repair:
write fragments, derived indexes, resident invalidation decisions, and refresh
work should be associated with the read predicates or table generations they
depend on.

For write throughput, this suggests a future transaction descriptor that
records read predicate families, modified column sets, generated row-key
ranges, touched value-index columns, and resident generations invalidated. If a
commit-time validation conflict is isolated to one predicate family, the
mutation owner could re-run only the affected fragment and preserve already
prepared independent fragments.

For MVCC/snapshot design, attribute-level predicate validation maps well to
P8's route metadata. A read snapshot or mutation fragment should know which
columns and generations it observed. That lets the engine distinguish "same
row changed in an irrelevant column" from "predicate or output column changed,"
reducing unnecessary abort, fallback, or resident invalidation.

For GPU execution, repairable result-set predicates are analogous to
over-resident or retained scan predicates whose candidate vectors can be
patched with post-start committed versions. This is not safe for arbitrary SQL
yet, but it is a useful benchmark idea for long-running read-modify-write
procedures: keep candidate row ordinals and version boundaries so a validation
failure can append or remove only the changed candidates before rerunning a
small GPU or CPU fragment.

**Risks and mismatches:** MV3C assumes annotated or analyzable transaction
programs with deterministic closures. The current engine accepts SQL over
pgwire, where ad hoc statements often lack stored procedure boundaries and
where exposing dependency annotations to users would be a major product
choice. The evaluated prototype is single-threaded, uses in-memory redo logs,
and does not prove multicore contention, WAL flush, network IO, GPU residency,
or 1M-session behavior. Long version chains under allowed write-write conflicts
can hurt, and the paper itself reports deterioration in one banking experiment
as concurrent uncommitted versions accumulate. The approach also does not
address DDL, snapshot retirement, recovery replay, or GPU cache invalidation.

**Benchmark candidates:**

- Add conflict telemetry to the mutation owner before implementing repair:
  validation failures by table, predicate/column family, write-write versus
  read-write cause, wasted rows/bytes prepared, and whether independent write
  fragments existed. Minimum gate: no correctness change and actionable
  conflict attribution.
- Prototype attribute-level validation for one SQL-visible read-modify-write
  microbenchmark. Expected improvement: fewer unnecessary aborts or owner
  fallbacks when unrelated columns change. Failure condition: validation cost
  exceeds saved retries at low conflict.
- Build a stored-procedure-only repair experiment with two independent update
  fragments and one shared hot counter/fee row. Preserve WAL-before-visibility,
  repair only the hot fragment after validation failure, and compare against
  full abort/retry under concurrency `1,2,4,8,16,32,64`.
- For long scan-plus-update transactions, test result-set repair over a stable
  MVCC boundary: retain candidate row ids from the initial scan, match
  committed versions since start, patch the candidate vector, then rerun only
  the dependent write fragment. Minimum proof gate: commit-order equivalent
  result and explicit version-chain length telemetry.
- Add a guardrail benchmark for allowed write-write conflicts: measure version
  chain traversal and memory growth under a single hot row, and require a cap
  or fallback policy before enabling this behavior outside experiments.

### 2026-06-02 - Demikernel Datapath OS Architecture for Microsecond-scale Datacenter Systems

**Citation:** Irene Zhang, Amanda Raybuck, Pratyush Patel, Kirk Olynyk, Jacob
Nelson, Omar S. Navarro Leija, Ashlie Martinez, Jing Liu, Anna Kornfeld
Simpson, Sujay Jayakar, Pedro Henrique Penna, Max Demoulin, Piali Choudhury,
and Anirudh Badam. "The Demikernel Datapath OS Architecture for
Microsecond-scale Datacenter Systems." SOSP 2021, pp. 195-211. DOI:
`10.1145/3477132.3483569`. Retrieved 2026-06-02 from the Microsoft Research
publication page and author PDF, `https://irenezhang.net/papers/demikernel-sosp21.pdf`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** high-concurrency networking; kernel bypass; zero-copy
I/O; bounded buffers; asynchronous queues; session admission; storage/network
datapaths; low-latency runtime.

**Core idea:** Demikernel argues that microsecond-scale systems need a
datapath OS rather than ad hoc direct use of each kernel-bypass device. The
system keeps a conventional kernel on the control path while moving the
latency-critical I/O datapath into interchangeable user-space library OSes
with a common API. The important abstraction is not "use DPDK/RDMA/SPDK
directly"; it is to make zero-copy buffers, queue ownership, asynchronous I/O,
and CPU scheduling explicit enough that applications can target heterogeneous
fast devices without rewriting their execution model.

The paper builds Demikernel library OS prototypes for Linux and Windows and
ports an echo server, UDP relay, Redis, and TxnStore. Reported echo results
show nanosecond-scale Demikernel overhead per I/O and competitive
microsecond-scale latency versus eRPC, Shenango, and Caladan, while preserving
portability across DPDK, RDMA, Windows, Linux, and Azure settings. For Redis
with persistence, the Demikernel storage/network path reports throughput within
10% of unmodified non-persistent Redis in the evaluated setup. For TxnStore,
the Demikernel ports are competitive with or better than the existing custom
RDMA messaging stack, largely because zero-copy coordination is made explicit.

**Concrete mechanisms:**

- Demikernel defines PDPIX, a portable datapath API whose operations target
  queues instead of POSIX file descriptors. Network and storage devices expose
  common queue creation, asynchronous push/pop, wait, close, and buffer
  operations.
- Library OSes are device-specific but API-compatible. Catnip implements a
  TCP/UDP stack on DPDK, Catmint maps queue operations to RDMA, Catnap uses a
  polling POSIX datapath, Catpaw targets Windows, and Cattree exposes a simple
  kernel-bypass storage stack.
- I/O memory is managed through a DMA-capable heap so buffers remain pinned,
  registered, or huge-page backed as the active device requires. Applications
  can allocate buffers without encoding device-specific registration policy in
  their own hot paths.
- Zero-copy safety is enforced through buffer ownership and use-after-free
  protection. A buffer handed to an outgoing queue cannot be freed or modified
  unsafely while a stack might still need it for retransmission or completion.
- Completion is asynchronous and coroutine-oriented. Applications issue work
  through queue operations, then wait for completions rather than blocking an
  OS thread per operation.
- CPU multiplexing is treated as a datapath requirement: library OS work,
  device polling, and application work must be scheduled at microsecond
  granularity, avoiding coarse kernel thread scheduling where possible.
- Demikernel deliberately hides one-sided RDMA and other highly specialized
  hardware features behind a portable queue API. The authors call out this
  portability/performance tradeoff as a limitation for systems that need direct
  hardware-specific offloads.
- Evaluation separates portability from raw peak performance. The paper
  reports roughly 50 ns processing latency per I/O in prototype paths and
  17-26% peak throughput overhead versus direct kernel-bypass APIs, while
  showing easier ports across multiple devices and environments.

**GPU DB mapping:** This paper is a strong match for the target runtime in
`11-high-throughput-query-runtime.md`. The current benchmark endpoint's
thread-per-client topology should be treated as a correctness harness, not a
session-scale design. Demikernel supports moving toward a portable internal
datapath API with explicit queues, buffer lifetimes, and completions: network
IO workers parse pgwire frames into bounded command rings, mutation/read/GPU
owners consume from typed queues, and response rings return stable buffers to
the IO workers without borrowing mutable engine state.

The most transferable idea is zero-copy coordination as an ownership contract.
For the GPU DB, pinned response buffers, COPY chunks, CUDA staging buffers, and
encoded pgwire output should have explicit states: free, filling, submitted,
in-flight, completed, reusable, or retired. That state machine can span CPU
network IO, mutation admission, WAL/MVCC apply, GPU execution workers, and
response writes without relying on ad hoc lifetimes or copies at every
boundary.

PDPIX's queue-centric interface maps to the engine's owner domains. Instead of
letting a session call engine operations directly, a session should enqueue a
typed request with a buffer handle, snapshot/generation requirement, deadline,
and response route. The completion model then gives a natural place for
backpressure: if command rings, response rings, pinned buffers, CUDA staging
buffers, or mutation batches are exhausted, the IO worker can delay, reject, or
route to fallback before admitting more work.

For 1M logical sessions, the paper reinforces that logical concurrency must be
decoupled from OS-thread concurrency and from scarce datapath resources. A
million sessions can be mostly parked connection/protocol state, while only a
bounded number of queue entries, pinned buffers, active COPY chunks, and GPU
requests are admitted at once. The engine should benchmark session scale by
resource budgets, not by increasing owner threads.

**Risks and mismatches:** Demikernel is an operating-system architecture paper,
not a database serving engine. It does not solve SQL planning, MVCC visibility,
WAL-before-visibility, transaction validation, GPU residency invalidation, or
PostgreSQL protocol semantics. Its portability layer can hide specialized
hardware features that may matter later, such as one-sided RDMA or future
GPU-initiated IO. Some evaluated designs consume a full CPU core for polling
to reduce latency; that may conflict with mutation owners, GPU workers, and
network IO workers on a constrained host. The paper also reports application
ports and echo/Redis/TxnStore experiments, not million-session pgwire behavior.

**Benchmark candidates:**

- Build an internal buffer-lifetime telemetry proof for pgwire responses and
  COPY chunks: count allocations, copies, state transitions, reuse hits,
  blocked submissions, and in-flight bytes at concurrency `1,2,4,8,16,32,64`.
  Minimum gate: identical SQL-visible results and no new unbounded queues.
- Replace one hot endpoint path with explicit request/completion handles across
  IO worker, engine owner, and response writer. Expected improvement: lower
  owner queue wait and fewer copies for repeated retained reads. Failure
  condition: p50 latency regresses for single-session requests or error paths
  leak buffers.
- Simulate 1M logical sessions without running GPU benchmarks: allocate only
  compact per-session protocol state, admit a bounded active subset, and
  measure memory per idle session, active ring depth, response-buffer pressure,
  and overload decisions.
- Add a COPY admission buffer pool with explicit states for decoded row chunks,
  WAL submission, MVCC apply, residency invalidation, and release. Proof gate:
  no chunk can be reused before WAL/MVCC ownership has completed, and queue
  saturation produces an observable backpressure reason.
- Compare polling versus readiness-driven IO worker loops under retained-read
  concurrency. Expected result: polling may reduce p50/p99 latency at the cost
  of CPU burn; the engine needs a configurable policy tied to admission and
  deployment CPU budget.

## Cross-Paper Synthesis

### 2026-06-02 - First Modern Batch Synthesis

The modern reviewed set now covers three complementary design tracks: DPF
pushes fused GPU data paths for resident and over-resident reads, the 2025
hybrid CPU/GPU paper pushes CPU prefiltering plus compressed transfer before a
GPU compute tail, and MV3C pushes dependency-aware validation so conflicts
repair only the affected transaction fragments. Together, they argue for route
descriptors that are richer than "CPU versus GPU": each hot path should name
its snapshot/generation boundary, columns touched, bytes moved, scratch demand,
predicate dependencies, and fallback or repair reason.

Converging design tracks:

- **Descriptor-first execution:** retained reads, over-resident scans, and
  write fragments should all carry explicit demand/dependency metadata before
  admission.
- **Boundary-preserving batching:** micro-batches and fused GPU kernels are
  attractive only when every request in the batch shares a compatible snapshot,
  route shape, and visibility boundary.
- **Reduce before transfer:** over-resident execution should prefer CPU or
  metadata pruning when it reduces PCIe bytes, then hand compact batches to GPU
  workers.
- **Repair before retry:** write-path conflicts should be measured by wasted
  independent work and repaired only where dependency boundaries make that
  equivalent to restart.

Current category gap: the queue still needs more runtime/session-scale and
network-admission papers before the design over-invests in storage and GPU
execution. The next high-value paper should likely be Shenango, Caladan, or a
newer high-concurrency runtime paper, unless MVCC/snapshot robustness becomes
the immediate blocker.

Benchmark priorities:

- Add route descriptor telemetry for retained and over-resident reads:
  snapshot generation, columns, resident bytes, H2D/D2H bytes, scratch bytes,
  kernel count, and queue wait.
- Add mutation conflict attribution before attempting transaction repair:
  conflict cause, columns, prepared bytes/rows wasted, and independent
  fragments available.
- Compare three over-resident routes on the same query: CPU-only,
  CPU-prefilter-plus-GPU-tail, and full partition GPU streaming.
- Keep latency gates explicit for every fusion or batching experiment, because
  the reviewed GPU papers optimize throughput and response time for analytical
  queries, not pgwire OLTP tail latency.

## Reviewed Papers

### 2026-06-02 - Virtual-Memory Assisted Buffer Management

**Citation:** Viktor Leis, Adnan Alhomssi, Tobias Ziegler, Yannick Loeck, and
Christian Dietrich. "Virtual-Memory Assisted Buffer Management." Proceedings of
the ACM on Management of Data 1(1), article 7, SIGMOD/PACMMOD 2023. DOI:
`10.1145/3588687`. Retrieved 2026-06-02 from the TU Braunschweig/TU Hamburg
author preprint, `https://www.ibr.cs.tu-bs.de/vss/Publications/2023/leis_23_sigmod.pdf`.

**Category:** multi-tier cache / data placement.

**Relevance tags:** buffer management; virtual memory; NVMe tiering; explicit
eviction; page fault control; variable-sized pages; OS/DBMS co-design; tier
telemetry.

**Core idea:** The paper proposes `vmcache`, a buffer manager that uses
hardware-supported virtual-memory translation for cached page lookup while
keeping the DBMS, not the operating system, in charge of page faulting,
eviction, dirty-page writeback, and replacement policy. It is a middle path
between ordinary DBMS hash-table buffer pools, which pay software translation
cost on hits, and file-backed `mmap`, which gives fast TLB-backed hits but
hands eviction and fault timing to the OS.

The second contribution, `exmap`, is a Linux kernel-module interface for
scalable page-table manipulation. The authors show that plain Linux virtual
memory operations can become the bottleneck with modern NVMe devices because
page-at-a-time `madvise`/fault behavior triggers TLB shootdowns and centralized
page-allocation costs. `exmap` changes the interface semantics: batch page
freeing, avoid allocation-time shootdowns, keep a private preallocated page
pool, expose per-thread control interfaces, and integrate page allocation with
read I/O through a proxy file descriptor.

The evaluation is storage-engine focused, not SQL-server focused. The authors
compare `vmcache`, `vmcache+exmap`, LeanStore, WiredTiger, and LMDB using a
standalone C++ B+tree random lookup workload and TPC-C-like workload on one
64-core/128-thread EPYC server with a 128 GB cache and a fast Samsung PM1733
NVMe SSD. Reported in-memory results show `vmcache` scaling to roughly 90M
random lookups/s and around 3M TPC-C transactions/s in their setup. For
out-of-memory random lookup, `exmap` improves basic `vmcache` by about 60% and
lets the design become I/O-bound. The paper also reports that full optimistic
page reads add less than 8% overhead versus a simple random DRAM read in their
microbenchmark, while an unsynchronized hash-table translation path is much
slower for cold DRAM accesses.

**Concrete mechanisms:**

- `vmcache` maps the storage address space into anonymous virtual memory rather
  than file-backed `mmap`; storage reads and writes remain explicit through
  calls such as `pread`, async I/O, and `pwrite`.
- Page identifiers map directly to virtual addresses. Cache hits avoid a
  DBMS-level PID-to-pointer hash lookup and rely on hardware page translation
  cached in the TLB.
- Eviction is DBMS-controlled. Dirty pages are written explicitly before the
  page is removed from the page table with `MADV_DONTNEED`.
- A per-page state array is the synchronization source of truth: evicted,
  locked, unlocked, marked, shared-lock counts, and a version counter are
  packed into a 64-bit state word.
- Optimistic reads read the page after sampling state, then validate that the
  page was not modified or evicted by checking the version. Eviction increments
  the version, so a concurrent optimistic read may observe zero-page data but
  fails validation instead of requiring hazard pointers or epoch reclamation.
- The replacement policy can be DBMS-defined. The implementation uses clock,
  marking cached pages and clearing marks on access; the cached-page set is
  tracked in a DRAM-sized hash table used only for misses and eviction, not for
  hits.
- Batch eviction writes dirty pages and removes page-table entries in groups
  of 64 in the implementation, reducing exclusive-lock and page-table churn.
- Variable-sized DBMS pages become easier: a large logical page can occupy a
  contiguous virtual range backed by non-contiguous physical pages, avoiding
  user-space fragmentation and simplifying large strings or compressed column
  chunks.
- `exmap` adds vectorized/scattered allocation/free operations over a virtual
  memory surface, per-thread interfaces with local free lists, page stealing,
  batched TLB shootdowns, lock-free page-table hot paths, and a proxy file
  descriptor that can allocate pages and read backing storage in one operation.
- `exmap` deliberately drops general VM features such as swapping and
  copy-on-write fork for its controlled surface, making page residency and
  memory consumption predictable for the DBMS.

**GPU DB mapping:** The most important transfer is the split between
hardware-assisted address translation and DBMS-owned placement policy. For the
GPU DB, the host-memory and NVMe tiers should not become invisible OS page
cache behavior. We want explicit admission, eviction, fallback, and telemetry.
But the paper argues that explicit DBMS control does not require every hot read
to pay a hash lookup or pointer-swizzling complexity; virtual-address structure
can encode placement when the page or segment identifier is stable.

For P8, this maps cleanly to a future host-side tier beneath GPU residency:
cold partition segments live on NVMe, warm decoded or compressed segments live
in host DRAM, and hot retained column groups live in GPU memory. A vmcache-like
host tier could make partition/segment IDs map to stable virtual ranges while
the cache manager still owns read I/O, dirty writeback, eviction, and
promotion. GPU resident snapshots would remain immutable performance caches,
but their CPU-side source segments could be managed with page-state telemetry
instead of opaque OS cache state.

The variable-sized-page idea is especially useful for GPU DB column groups.
Compressed chunks, text byte buffers, offsets buffers, and partition-local
metadata rarely want one universal 4 KB logical shape. A virtual-memory-backed
host tier can present contiguous logical chunks to decompression, prefix
filtering, or CUDA staging code while avoiding physical fragmentation in DRAM.
That fits the P8 direction of generated GPU column-group segments without
forcing all host-side chunks to be copied into temporary contiguous buffers
before H2D transfer.

For concurrency and latency, the page-state/version pattern is a useful model
for immutable snapshot handles and retained route validation. A read worker can
optimistically access a host segment or resident snapshot only if its generation
is stable; eviction or invalidation increments the generation and forces retry,
CPU fallback, or route rejection. This resembles P8's existing validity flags
but gives a more concrete hot-path contract: state word, version, lock mode,
mark/evict state, and saturation counters should be cheap enough to check on
every route.

`exmap` itself is not an immediate implementation dependency. The transferable
point is that fast storage can make kernel page-table and allocation costs
visible. If the GPU DB later streams over-resident partitions from NVMe through
host DRAM to GPU, the benchmark must measure page-table/fault/allocation costs
separately from device bandwidth, decompression, H2D transfer, and kernel time.
Otherwise, an apparent "NVMe or GPU bottleneck" may actually be host virtual
memory churn.

**Risks and mismatches:** The paper targets CPU storage engines with B+trees,
not a PostgreSQL-compatible SQL server with WAL/MVCC, pgwire, and GPU-resident
execution. Its experiments disable WAL and use the lowest isolation levels in
competitor systems, so the reported TPC-C-like throughput is not a direct
transactional durability result. `exmap` requires a kernel module and new VM
semantics, which may be unacceptable for portability or deployment. The design
also consumes page-table and page-state memory proportional to storage size;
the paper estimates about 16 bytes per 4 KB of storage for full 5-level page
table plus state, which is reasonable for some NVMe tiers but still a real
budget when the engine targets very large cold data.

For GPU DB, the biggest mismatch is that GPU memory is not CPU virtual memory.
TLB-backed CPU page translation does not automatically solve GPU HBM placement,
CUDA allocation lifetime, GPUDirect Storage behavior, or device-side page
faulting. A vmcache-style host tier must remain a source or staging tier, not a
substitute for explicit GPU resident snapshot publication. Variable virtual
pages also do not remove the need for aligned, pinned, and stream-owned H2D
buffers.

**Benchmark candidates:**

- Add host-tier residency telemetry before implementing a new cache: segment
  id, state, generation, page/chunk size, resident host bytes, evict reason,
  last promotion reason, fault/read count, and retry/fallback count. Minimum
  gate: no SQL behavior change and route logs can distinguish CPU canonical,
  host warm, and GPU retained sources.
- Prototype a virtual-address-shaped host segment table for read-only cold or
  warm partitions, without kernel modules: stable segment IDs, explicit async
  reads, explicit eviction, versioned state words, and no hidden OS page-cache
  dependency in correctness. Failure condition: p50 retained-read latency
  regresses or queue wait hides the benefit.
- Measure fixed 4 KB chunks versus variable-sized host column chunks for
  compressed/text-heavy resident refresh. Expected improvement: fewer temporary
  copies and better H2D staging for large string/offset/compressed buffers.
  Required measurement: CPU copy bytes, allocations, H2D bytes, refresh wall
  time, and route-invalidations caused by chunk pressure.
- Add an over-resident partition streaming microbenchmark that separates
  storage read, host allocation/page-table/fault cost, decompression or decode,
  H2D transfer, CUDA execution, and D2H result time. Minimum proof gate:
  reported bottleneck is phase-specific, not a single wall-clock bucket.
- For snapshot validation, test a cheap state-word/generation check on every
  retained read route. Expected improvement: safer optimistic fast paths and
  clearer invalidation retries. Failure condition: the state check or cache-line
  traffic dominates single-session p50 latency.

### 2026-06-02 - Robust Plan Evaluation based on Approximate Probabilistic Machine Learning

**Citation:** Amin Kamali, Verena Kantere, Calisto Zuzarte, and Vincent
Corvinelli. "Robust Plan Evaluation based on Approximate Probabilistic Machine
Learning." PVLDB 18(8), 2025, pp. 2626-2638. doi:10.14778/3742728.3742753.
Retrieved 2026-06-02 from the PVLDB PDF,
`https://www.vldb.org/pvldb/vol18/p2626-kamali.pdf`; arXiv version
`https://arxiv.org/abs/2401.15210`.

**Category:** query optimization / planning.

**Relevance tags:** robust query optimization; learned cost models; plan risk;
estimation risk; route selection; CPU/GPU fallback; workload drift; tail
latency; uncertainty-aware admission.

**Core idea:** Roq treats optimizer estimates as probability distributions
rather than point values. Classical optimizers and many learned optimizers pick
the plan with the best expected cost even when that estimate is highly
uncertain. Roq argues that robust plan selection should consider both the
expected runtime and the risk that an apparently cheap plan becomes bad at
runtime because of cardinality errors, model uncertainty, plan structure, or
workload drift.

The paper formalizes three risk notions. Plan risk is the uncertainty inherent
to a plan's structure and sensitivity to input-cardinality errors. Estimation
risk is uncertainty from limitations of the learned cost model itself.
Suboptimality risk is the probability that a selected plan is slower than
alternatives at runtime. Roq then uses approximate probabilistic machine
learning to predict both cost and uncertainty, and uses that uncertainty during
plan selection instead of only after a bad plan is observed.

In evaluation with IBM Db2-generated candidate plans on CEB, JOB, and DSB, Roq
reports better predictive accuracy than the tested learned baselines and better
tail robustness of selected plans. The paper's strongest planning result is
that uncertainty-aware selection reduces 99th-percentile plan suboptimality
relative to Roq's base learned model, while still preserving practical
inference cost. The authors report that about 10 MC-dropout inference
iterations were enough for stable plan choice, with prototype Python overheads
around tens of milliseconds for the risk-aware strategies.

**Concrete mechanisms:**

- Roq decomposes total cost-estimate uncertainty into data uncertainty and
  model uncertainty. The data uncertainty corresponds to sensitivity to input
  estimates and plan shape; model uncertainty corresponds to lack of knowledge
  in the learned model parameters.
- The learned cost model predicts a mean execution time and variance through
  two output branches. The variance branch is trained with a Gaussian
  negative-log-likelihood-style loss, so the model learns uncertainty along
  with the latency prediction.
- Model uncertainty is estimated with Monte Carlo dropout: dropout remains
  active during inference, multiple predictions are sampled, and the variance
  of predicted means estimates epistemic/model uncertainty.
- Total uncertainty combines the predicted data uncertainty and the sampled
  model uncertainty; Roq can use model-only, data-only, or total uncertainty
  in plan selection.
- Suboptimality-risk selection compares each candidate plan against every
  alternative. Assuming independent normal cost distributions, it estimates the
  probability that plan `i` is slower than plan `j`, averages those pairwise
  risks, and picks the plan with the lowest average risk.
- The pairwise suboptimality calculation is vectorizable with matrices of mean
  differences and combined variances, keeping the extra optimizer-time
  computation small compared with repeated learned-model inference.
- Conservative selection chooses the plan minimizing `mean + factor * sigma`.
  This gives a simpler risk penalty when the non-parametric pairwise
  suboptimality calculation is too expensive or unnecessary.
- A pruning strategy can remove plans whose plan-risk or estimation-risk values
  exceed tuned thresholds before ordinary selection, though the paper reports
  this did not materially improve over the two main risk-aware strategies.
- Query encoding uses join graphs with table, predicate, join, skew,
  selectivity, and graph-level attributes. A TransformerConv GNN produces query
  and table embeddings.
- Plan encoding uses a vectorized plan tree processed by tree convolutional
  neural networks, augmented with table embeddings from the query encoder.
- Training and validation use optimizer-generated candidate plan sets from
  multiple hint configurations, then measure actual execution times as labels.
- The experiments explicitly test workload drift: a minor DSB template shift
  and a larger CEB-to-JOB shift. Roq's GNN query representation and risk-aware
  selection are presented as the source of better robustness under these
  shifts.

**GPU DB mapping:** The GPU DB planner has exactly the kind of fragile route
choice Roq targets. A resident GPU route may have the best expected latency
when the snapshot is valid, queue depth is low, the predicate is selective, and
the result is small. The same route can become a tail-latency trap if
cardinality is wrong, GPU queues are saturated, a refresh invalidates the
snapshot, H2D/D2H bytes are underestimated, or CPU fallback would have avoided
waiting behind a long scan. Roq suggests treating each route estimate as
`expected cost + uncertainty`, not a single deterministic score.

For the current architecture, the first transferable design is not a full
learned optimizer. It is a risk envelope around deterministic CPU/GPU route
rules. A route descriptor can carry expected latency plus uncertainty fields:
cardinality confidence, resident-validity risk, queue-delay variance,
transfer-byte variance, kernel-time variance, refresh/invalidation risk, and
fallback penalty. The planner can then reject a GPU route whose mean is low but
whose tail risk is unacceptable for an OLTP session, while still selecting it
for analytical work where throughput matters more than p99 latency.

The plan-risk versus estimation-risk split maps well to GPU DB telemetry.
Plan risk is route-shape risk: joins, range scans, prefix filters, result-size
scatter, partition fanout, or over-resident streaming tend to become worse
when estimates are wrong. Estimation risk is model or statistics risk: stale
table stats, missing queue-delay samples, new hardware, cold GPU cache state,
or workload drift. Those should be recorded separately so the runtime can tell
whether a bad decision came from a fragile route or an uninformed estimator.

The conservative `mean + factor * sigma` strategy is immediately useful for
production guardrails. For low-latency pgwire sessions, the route chooser could
prefer the lowest risk-adjusted p99 proxy rather than the lowest mean. For
batch/analytical sessions, the factor can be lower, allowing throughput-heavy
GPU routes with higher variance. That gives session policy a measurable shape
instead of hard-coding "GPU if valid" or "CPU fallback if queue full."

The MC-dropout/GNN model is a later-stage idea. The first GPU DB version should
use measured distributions from existing route telemetry: queue wait,
kernel/event time, transfer bytes, result rows, refresh age, invalidation
count, and CPU fallback time. A learned model becomes attractive only after
there is enough route history to train and validate against workload drift.

**Risks and mismatches:** Roq is evaluated for query optimization over
candidate plans generated by Db2 hint configurations, not for a GPU-aware
transactional engine with WAL, MVCC, residency, and session admission. The
paper focuses on plan optimization robustness, not runtime adaptive execution,
backpressure, or correctness after invalidation. Its risk calculations assume
normal transformed target distributions and simplify pairwise plan covariance
with an independence assumption; those assumptions may overestimate or
misestimate GPU route risk if queue delay and transfer contention are strongly
correlated. The reported inference overheads are acceptable for analytical
optimization but may be too high for single-row OLTP requests unless the model
is cached, compiled, or reserved for complex routes. Finally, robust plan
selection can deliberately choose a slower mean plan to reduce tail risk; that
must be tied to session class and SLA rather than applied globally.

**Benchmark candidates:**

- Add risk-adjusted route telemetry for retained reads: mean and variance of
  queue wait, CUDA event time, H2D/D2H bytes, result rows, and response encode
  time by query shape and snapshot generation. Minimum gate: no planner
  behavior change while telemetry can compute `mean + sigma` per route.
- Implement a deterministic conservative route scorer for one CPU-versus-GPU
  retained aggregate: choose by `estimated_mean + k * estimated_stddev`, with
  different `k` for OLTP and analytical session classes. Failure condition:
  p50 improves while p99 or overload rejections regress under concurrency.
- Build a route-risk replay harness from existing benchmark CSVs: replay route
  choices using mean-only, conservative, and pairwise suboptimality-risk
  scorers, then compare chosen-route p50/p95/p99 and fallback counts without
  running GPU benchmarks.
- Track plan risk and estimator risk separately. Plan risk should rise for
  route shapes with high fanout, result scatter, over-resident streaming, or
  invalidation sensitivity; estimator risk should rise when stats are stale or
  telemetry sample count is low.
- Add a workload-drift test for planner route choice: train or calibrate on
  retained equality lookups, then evaluate on range aggregates, invalidated
  snapshots, and higher concurrency. Minimum proof gate: the risk-aware scorer
  falls back or rejects explicitly instead of selecting fragile GPU routes with
  bad tail latency.
- For future learned planners, require the model to output uncertainty and
  compare against deterministic planner baselines. A learned route may be used
  only if it improves tail latency or throughput without increasing
  correctness fallback, invalidation retry, or overload ambiguity.

### 2026-06-02 - Read-safe snapshots for abort/wait-free serializable reads

**Citation:** Takamitsu Shioi, Takashi Kambayashi, Suguru Arakawa,
Ryoji Kurosawa, Satoshi Hikida, and Haruo Yokota. "Read-safe
snapshots: An abort/wait-free serializable read method for read-only
transactions on mixed OLTP/OLAP workloads." Information Systems 124,
2024, article 102385. doi:10.1016/j.is.2024.102385. Retrieved
2026-06-02 from the ScienceDirect open-access article,
`https://www.sciencedirect.com/science/article/pii/S0306437924000437`.
Also read with the closely related arXiv preprint "Serializable HTAP
with Abort-/Wait-free Snapshot Read,"
`https://arxiv.org/abs/2201.07993`.

**Category:** MVCC / snapshot / visibility.

**Relevance tags:** serializable HTAP; MVCC; read-only snapshots;
replica visibility; dependency tracking; SSI; long analytical reads;
abort-free reads; wait-free reads; snapshot publication.

**Core idea:** Read-Safe Snapshots (RSS) try to give read-only OLAP
transactions a serializable MVCC view without forcing either the OLTP
writer side or the analytical reader side to abort or wait because of
the reader's participation. The key observation is that a read-only
transaction does not always need the newest committed version of every
item. It needs a prepared set of committed versions whose dependency
region cannot be reached from transactions outside that set in a way
that would create a serialization cycle.

The paper formalizes such a set as RSS: a set of committed transactions
`P` where no transaction outside `P` can reach a transaction inside `P`
through the multiversion dependency graph. A protected read-only
transaction then reads the most recent versions created by transactions
inside `P`. The authors show that adding those protected read-only
transactions preserves version-ordered conflict serializability, because
their dependency edges cannot close a cycle across the RSS boundary.

For practical construction, the paper specializes the idea to systems
whose OLTP side already uses SSI. Instead of tracking every possible
dependency, it uses start/end transaction state plus concurrent
rw-antidependencies. At a history prefix, transactions are classified
into `Done`, `Clear`, `Obscure`, and active/undone regions. The
algorithm starts with all `Clear` transactions and adds certain
transactions outside `Clear` when they have outgoing dependencies to
`Clear`. SSI's dangerous-structure rule is then used to argue that the
resulting region is still unreachable from outside and can be served as
RSS.

**Concrete mechanisms:**

- RSS defines a serializable snapshot as a dependency-graph region, not
  as "latest committed at timestamp T." This makes it acceptable for a
  read-only query to choose shortly previous versions when the latest
  versions could participate in an anomaly.
- Protected read-only transactions are outside RSS but read only the
  newest versions written by transactions inside RSS. They do not
  perform additional read-set validation at execution time.
- The SSI-based construction uses `Done(p)` for transactions ended by a
  prefix and `Clear(p)` for transactions that ended before all currently
  undone transactions began. Transactions outside `Clear` but with
  rw-dependencies into `Clear` may be added to RSS.
- The implementation records outgoing rw-dependencies, transaction
  start/commit/abort information, and dependency graph state. In the
  multinode PostgreSQL prototype, this metadata is shipped through WAL
  logical messages to a read-only replica.
- The replica runs an RSS construction invoker, maintains active/done
  state and a dependency graph in shared memory, and exports RSS snapshot
  data for read-only transactions.
- Snapshot-preserving transactions keep the needed versions alive until
  the next RSS is constructed. The PostgreSQL prototype also uses
  hot-standby feedback and replication slots to preserve old versions on
  replicas.
- In the single-node prototype, known analytical read-only queries are
  marked read-only and routed to RSS, while normal OLTP transactions
  continue to use SSI.
- The evaluation uses CH-BenCHmark on PostgreSQL 12 prototypes. The
  ScienceDirect abstract reports about 15% overhead versus baseline SI
  throughput in a multinode setting, about 45% better OLTP throughput
  than SafeSnapshots in mixed workload, and no OLAP-throughput
  degradation. The arXiv version reports up to 20% OLTP improvement
  versus SSI+SafeSnapshots in the single-node prototype and roughly 10%
  OLTP overhead versus nonserializable SSI+SI in the multinode replica
  setup.

**GPU DB mapping:** RSS is a strong match for the current GPU DB
question of how to publish immutable retained read snapshots without
making writes wait on long analytical or retained GPU reads. The
transferable idea is to separate "visibility boundary for correctness"
from "latest committed row in CPU truth." A retained GPU snapshot can be
serializable if its source transaction/partition region has a
well-defined unreachable boundary, even when it is not the absolute
latest CPU state.

For P8, this suggests a future snapshot publication contract with two
boundaries. The mutation owner keeps the durable WAL-before-visibility
boundary for CPU truth. The residency owner publishes a retained
read-safe boundary for GPU snapshots: table/partition identity, source
WAL transaction boundary, dependency epoch, invalidation generation, and
the set or interval of transactions represented in the snapshot. Read
workers then execute protected read-only retained queries against that
boundary without joining the mutation owner's conflict tracking on every
request.

RSS also gives a concrete way to think about replica-like GPU memory.
GPU resident state is not the transactional authority; it is closer to a
read-only replica with explicit version preservation. If retained GPU
snapshots are constructed before readers arrive and are tied to
dependency metadata, read-only analytical or aggregate work can avoid
writer aborts, reader waits, and per-query SSI validation. That fits the
target runtime's immutable snapshot publication model better than the
current benchmark encoded response cache.

The dependency metadata should remain narrow at first. A full RSS graph
is probably too much for the current engine, but the paper's
`Clear`/`Done` split maps to engine generations: fully closed mutation
epochs, active mutation epochs, and ambiguous epochs with possible
rw-antidependencies. A first implementation hypothesis is to publish
read-safe retained snapshots only at deterministic generation boundaries
where no active mutation can reach the retained source region. Later,
for higher freshness, dependency-edge logging could allow more recent
obscure transactions to be admitted into the retained read-safe region.

For 1M logical sessions, the most important implication is that
read-only sessions should not all enqueue through the mutation owner just
to prove snapshot safety. RSS-like publication lets many readers share a
prevalidated retained snapshot handle. Session admission can reject,
fallback, or wait for a newer read-safe generation only when the handle
is stale, invalidated, or too old for the session's freshness class.

**Risks and mismatches:** RSS assumes the OLTP side already provides
serializability, and the practical algorithm is tied to SSI properties.
The current GPU DB uses a simpler MVCC tuple store and does not yet
track SSI-style rw-antidependencies, dangerous structures, or
transaction dependency graphs. Adopting RSS literally would add metadata
cost to the write path, which is already throughput-sensitive.

The prototype is PostgreSQL-based and oriented around HTAP read-only
queries and replicas, not GPU execution, CUDA memory, partitioned
resident layouts, or pgwire session scale. Version preservation can also
be expensive; the paper notes PostgreSQL HOT/vacuum/version-retention
effects. In GPU DB, the analogous cost is retained CPU/GPU memory
pressure and old-snapshot retirement. Finally, RSS can deliberately read
previous versions, so session policy must distinguish "serializable
read-safe" from "must include the newest committed mutation."

**Benchmark candidates:**

- Add a retained snapshot freshness/visibility telemetry field that
  distinguishes CPU latest boundary, retained source boundary, and
  invalidation generation. Minimum gate: retained reads report whether
  they used latest, read-safe older, CPU fallback, or rejected state.
- Prototype a conservative read-safe generation boundary without full
  SSI: publish retained snapshots only after all mutations in a closed
  generation have committed and no active mutation started before that
  boundary remains. Expected improvement: fewer owner-thread reads while
  preserving a clear serializable-ish contract for read-only routes.
- Add a CH-BenCHmark-inspired mixed workload proof with one writer lane
  and long retained analytical reads. Measure writer throughput,
  retained-read p50/p99, snapshot age, invalidation count, and fallback
  count. Failure condition: long reads force mutation-owner queue wait or
  stale reads are not explicitly labeled.
- Measure version-preservation cost for retained snapshots: CPU MVCC
  bytes pinned, GPU bytes pinned, snapshot handle count, oldest retained
  generation age, and retirement delay under continuous writes.
- Implement a route policy knob for freshness class: `latest_required`
  routes fall back or wait for CPU truth; `read_safe_allowed` routes may
  use an older retained read-safe generation. Minimum proof gate: the
  SQL-visible result path names the chosen boundary in telemetry.
- Explore dependency-edge logging only after the conservative generation
  benchmark. Required measurement: write-path metadata bytes, commit
  overhead, graph maintenance time, and whether the fresher retained
  snapshot reduces read latency enough to justify the write cost.

### 2026-06-02 - Second Modern Batch Synthesis

**Papers covered:** Virtual-Memory Assisted Buffer Management (SIGMOD/PACMMOD
2023), Robust Plan Evaluation based on Approximate Probabilistic Machine
Learning (PVLDB 2025), and Read-safe snapshots for mixed OLTP/OLAP workloads
(Information Systems 2024).

**Converging design tracks:**

- **Explicit state over opaque delegation.** vmcache argues for DBMS-owned
  placement policy even when virtual-memory hardware accelerates address
  translation. RSS argues for DBMS-owned read-safe snapshot construction even
  when MVCC can expose many committed versions. Roq argues for planner-visible
  uncertainty rather than trusting a point estimate. For GPU DB, the common
  track is explicit route state: tier location, snapshot boundary, generation,
  invalidation risk, queue depth, and confidence should be observable before
  choosing a route.
- **Immutable read handles as the concurrency escape hatch.** RSS and the
  runtime architecture both point toward prevalidated immutable snapshots that
  many read-only sessions can share without joining the mutation owner. vmcache
  adds the host-tier version/state-word angle. The next runtime track should
  publish cheap snapshot handles with state, generation, source boundary, and
  retirement telemetry.
- **Risk-aware fallback beats binary acceleration.** Roq's uncertainty-aware
  selection, vmcache's explicit fault/page-state telemetry, and RSS's
  freshness-versus-serializability distinction all warn against a planner rule
  like "GPU if resident." Route choice should choose among CPU latest, CPU
  host-warm, GPU retained read-safe, GPU latest after refresh, and explicit
  overload based on mean latency, tail risk, and freshness class.
- **Tiering and MVCC are coupled.** Host and GPU cache state cannot be designed
  separately from snapshot retention. A retained GPU snapshot pins CPU MVCC
  versions and GPU buffers; a host-tier segment can be evicted only if no
  active snapshot or refresh depends on it. Tier policy therefore needs
  snapshot-age and oldest-generation pressure metrics, not just bytes and hit
  rate.

**Category gaps:** Recent reviews now have useful coverage in GPU execution,
runtime/session scale, multi-tier placement, MVCC snapshots, transaction repair,
and planning. The queue still needs more modern transaction/write-path papers
that are not purely read-snapshot oriented, especially bulk ingest, commit
grouping, partition-owned mutation, and long/short transaction coexistence.
High-concurrency networking has one strong eRPC entry but still needs
Shenango/Caladan/Shinjuku-style scheduler coverage before designing the
production pgwire IO worker pool.

**Benchmark priorities:**

- Add no-behavior-change route state telemetry first: source tier, snapshot
  boundary, invalidation generation, queue wait distribution, route mean/stddev,
  fallback reason, and freshness class.
- Build a read-safe retained generation proof before full dependency-edge
  logging: one writer lane, one long retained read lane, explicit snapshot age,
  and no mutation-owner queueing for read-only routes.
- Create a route-risk replay harness from existing benchmark CSVs to compare
  mean-only, conservative `mean + k*sigma`, and freshness-aware route choices
  without running GPU benchmarks.
- Measure retention pressure as a first-class tiering metric: CPU MVCC bytes
  pinned, GPU bytes pinned, host warm bytes pinned, oldest retained generation,
  and retirement lag.
- For the next paper, prefer a transaction/concurrency/runtime candidate such
  as Oze, Chiller, Shenango, Caladan, or Shinjuku over another GPU analytics
  paper unless a specific GPU hardware question becomes urgent.

### 2026-06-02 - Oze decentralized graph-based concurrency control

**Citation:** Jun Nemoto, Takashi Kambayashi, Takashi Hoshino, and
Hideyuki Kawashima. "Oze: Decentralized Graph-based Concurrency Control
for Long-running Update Transactions." PVLDB 18(8):2321-2333, 2025.
doi:10.14778/3742728.3742730. Retrieved 2026-06-02 from the PVLDB
PDF, `https://vldb.org/pvldb/vol18/p2321-nemoto.pdf`.

**Category:** transaction processing / write path and concurrency control.

**Relevance tags:** MVSG; serializable concurrency control; long update
transactions; short transaction throughput; dependency graph; dynamic
version ordering; protocol switching; phantom avoidance; epoch GC; OLTP
benchmarking.

**Core idea:** Oze targets mixed workloads where one long-running update
transaction must commit while many short conflicting transactions keep high
throughput. The paper's central claim is that conventional OCC and MVCC
protocols often abort the long transaction as a false positive because they
constrain the serialization order too much, while 2PL can commit the long
transaction only by making short transactions wait behind locks. Oze instead
uses a multi-version serialization graph (MVSG) to search a wider
serializable scheduling space, but decentralizes the graph into record-local
and transaction-local pieces so it can run on many cores.

The motivating workload is BoMB, a bill-of-materials benchmark with one
long product-costing update transaction and five short transactions that can
change raw-material costs, products, quantities, and journal-voucher state.
The long transaction reads and updates a large dependency tree, so it creates
the kind of long-term anti-dependency chains that ordinary OLTP benchmarks
do not stress.

**Concrete mechanisms:**

- Oze stores a record-local graph with the versions of a record and a
  transaction-local graph with the records and followers relevant to a
  transaction about to commit. Serializability is checked by merging the
  target record-local graphs needed for that transaction rather than by
  locking one centralized global graph.
- Reads choose the latest committed version that does not create a cycle in
  the record-local graph. If a newer version is skipped, Oze records
  rw-dependency edges from the reader to the writers of skipped versions.
- Validation first chooses a version order for each written record, installs
  pending versions, then repeatedly merges record-local graphs for read-set
  records and follower records until the transaction-local graph is acyclic
  or an abort is required.
- Dynamic version ordering first tries ordinary postposing, then tries
  order forwarding: placing a new version before selected existing writers
  when that keeps the graph acyclic. Forwarding is bounded within an epoch to
  preserve linearizability and simplify cleanup.
- Oze switches between an MVSG mode and an OCC mode. MVSG mode starts when a
  long transaction aborts; workers return to OCC after no long transaction is
  observed for a configured period. Transition mode maintains graph state
  while validating with OCC, which the paper argues is safe because OCC's
  scheduling space is narrower.
- Long validation can be parallelized by assigning target-record graph merges
  to validator threads, then merging validator-local graphs back into the
  transaction-local graph.
- Phantom avoidance uses a precision-locking-style scan history. Inserts
  check scan predicates during validation and add graph edges from matching
  scanners to the inserter instead of relying on ordinary optimistic index
  node validation.
- Epochs drive graph and version garbage collection. Oze prevents new
  incoming edges to old nodes by avoiding reads of old nonlatest versions and
  disallowing order forwarding across old epochs.
- The evaluation uses CCBench on a dual-socket, 40-core Xeon server and
  compares Oze with Silo, TicToc, MOCC, ERMIA/SSN, Cicada, 2PL variants, and
  D2PL. On BoMB, the paper reports that Oze commits the long transaction
  while achieving four orders of magnitude higher short-transaction
  throughput than optimistic and MVCC protocols and up to five times higher
  throughput than pessimistic protocols. On TPC-C, Oze is comparable but
  below the peak protocol; the paper reports about 27% lower peak throughput
  than Silo. Protocol switching and graph GC have visible costs.

**GPU DB mapping:** Oze is most useful for the future write/snapshot side of
the GPU DB, not for the current retained read benchmark path. The
transferable idea is to treat "long GPU-visible work" as a first-class
transactional participant instead of forcing either the long work or the
short write path into a crude timestamp or lock order. A retained refresh,
partition rebuild, or long analytical update could carry a dependency
region, while short writes publish precise edges against affected records or
partitions. That would let the system distinguish true serialization cycles
from false-positive invalidations.

For P8, this argues for keeping mutation ownership and resident snapshot
publication observable enough to add dependency tracking later. The current
safe design can still invalidate resident generations conservatively, but
the eventual design should leave space for partition-local dependency graphs:
record or key-range edge summaries, scan predicate histories, generation
epochs, and follower sets. Those summaries could decide whether a retained
GPU snapshot or refresh must abort, wait, rebuild, or can continue on an
older serializable order.

Oze also sharpens the runtime plan. The production runtime should not route
all long retained refresh or analytical update work through a single owner
queue merely because it might conflict. A bounded MVSG-like validation lane
could run only for long or ambiguous transactions, while the ordinary short
write path keeps a cheaper OCC/MVCC protocol. That matches the paper's
protocol-switching lesson: precise graph maintenance is powerful, but it is
too expensive to pay for every simple OLTP operation.

For 1M logical sessions, the biggest implication is admission classification.
Sessions that issue ordinary short point writes, read-safe retained reads,
and long refresh/update work should enter different concurrency classes with
different metadata budgets. The system can reject or defer long graph-heavy
work under pressure without forcing every short request to carry graph cost.

**Risks and mismatches:** Oze is CPU OLTP concurrency-control work, not a GPU
execution or storage-tiering paper. The paper does not address WAL durability,
GPU memory, CUDA stream ownership, tier placement, or pgwire session scale.
Its graph maintenance can consume substantial memory; in the protocol
switching experiment, graph size grows while the long transaction validates,
and GC temporarily reduces throughput. That is a serious warning for a GPU DB
whose write path is already sensitive to metadata overhead.

Oze's best results depend on BoMB's specific long-update conflict pattern.
The benefit may be smaller for append-heavy ingest, mostly read-only retained
queries, or workloads where conservative invalidation is cheaper than graph
tracking. The implementation can still produce false-positive aborts because
decentralized concurrent graph choices are more restrictive than an ideal
central MVSG. The paper's phantom handling currently describes range-based
predicates; arbitrary SQL predicates, joins, and GPU-resident column scans
would need their own conservative summaries.

**Benchmark candidates:**

- Build a no-GPU concurrency simulator for one long retained refresh/update
  lane plus many short write/read lanes. Compare conservative generation
  invalidation, 2PL-style waiting, and a coarse partition-local dependency
  graph. Primary metrics: long-work commit rate, short-write throughput,
  queue wait, false invalidation count, graph bytes, and validation time.
- Add metadata-only telemetry to current mutation/residency paths: mutation
  generation, partition id, invalidated resident generation, snapshot age,
  long-operation flag, and whether a conflict was exact, partition-wide, or
  table-wide. This can be done before implementing graph validation.
- Prototype protocol classification before graph logic: cheap path for short
  writes and read-only retained reads, expensive path only for long refresh or
  update transactions. Gate: no measurable overhead on the existing short
  COPY/admission smoke when the expensive class is idle.
- Create a BoMB-inspired mixed workload for GPU DB using one long product-cost
  or aggregate refresh over a dependency tree, plus short row updates and
  point reads. Measure whether long work forces owner queue waits or can be
  explicitly deferred while short requests continue.
- If dependency tracking is attempted, start at partition granularity with
  epoch-bounded GC. Required evidence: graph memory remains bounded,
  validation p99 is visible, and old retained generations retire when the
  oldest active epoch advances.

### 2026-06-02 - Caladan: Mitigating Interference at Microsecond Timescales

**Citation:** Joshua Fried, Zhenyuan Ruan, Amy Ousterhout, and Adam Belay.
"Caladan: Mitigating Interference at Microsecond Timescales." OSDI 2020,
pp. 281-297. Retrieved 2026-06-02 from the USENIX publication page and
PDF, `https://www.usenix.org/conference/osdi20/presentation/fried` and
`https://www.usenix.org/system/files/osdi20-fried.pdf`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** microsecond scheduling; tail latency; admission control;
queueing delay; CPU interference; worker ownership; green threads;
memory-bandwidth pressure; hyperthreading; request concurrency.

**Core idea:** Caladan argues that latency-sensitive services cannot rely on
static resource partitioning, slow tail-latency feedback, or seconds-scale
convergence when interference changes over microseconds. Instead, it dedicates
a scheduler core to continuous control-signal polling and uses fast core
reallocation to preserve both high CPU utilization and microsecond-level tail
latency.

The paper is not a database concurrency-control design, but it is directly
relevant to the GPU DB runtime target. Its strongest transferable lesson is
that admission should react to the resource boundary that is actually
saturating, on the timescale where queueing damage begins. For this engine,
the equivalent boundaries are network IO workers, mutation owners, read
snapshot rings, GPU execution workers, pinned-buffer pools, host-memory
placement, and response rings.

**Concrete mechanisms:**

- Caladan separates latency-critical tasks from best-effort tasks and lets
  latency-critical work hold guaranteed cores while borrowing burstable cores
  when queueing delay or interference requires it.
- A dedicated scheduler core runs controllers every 10 microseconds. It polls
  queueing delay, request processing time, global memory bandwidth, per-core
  LLC miss rates, and voluntary-yield notices.
- The top-level allocator grants extra cores when a task's queueing delay
  exceeds a per-task threshold, subject to constraints from the memory-bandwidth
  and hyperthread controllers.
- The memory-bandwidth controller detects global DRAM saturation, attributes
  bandwidth use through sampled per-core LLC misses, and revokes one core at a
  time from the highest-offending best-effort task until saturation clears.
- The hyperthread controller watches request processing time for
  latency-critical work. When a request exceeds its threshold, Caladan can ban
  the sibling hyperthread and park it with `mwait` until the long request
  completes or capacity constraints require unbanning.
- Caladan uses a KSCHED kernel module to make scheduling operations fast:
  per-core shared-memory command regions, multicast IPIs, asynchronous command
  issue, remote-core execution of expensive scheduling work, remote performance
  counter reads, and shallow idle states that wake on cache-line writes.
- Applications run in a Shenango-derived runtime with green threads,
  kernel-bypass networking, SPDK storage support, shared queue telemetry, and
  work stealing across currently allocated cores.
- The runtime must expose internal request concurrency. If a latency-critical
  task cannot run more independent request work when granted cores, fast
  reallocation cannot help it recover lost capacity.
- In the evaluation, Caladan reports convergence to a new resource
  configuration in about 20 microseconds versus 10-20 seconds for Parties.
  When memcached is colocated with a garbage-collecting best-effort workload,
  the paper reports an 11,000x reduction in 99.9th percentile latency during
  GC cycles, from 580 ms under Parties* to 52 microseconds under Caladan.
- The paper also reports up to 560,000 core reallocations per second in an
  11-latency-critical-task experiment, while a Linux-mechanism variant
  bottlenecks near 285,000 allocations per second.

**GPU DB mapping:** Caladan reinforces the runtime document's bounded-owner
model, but pushes it toward control loops that are explicit and fast enough to
matter. A 1M logical-session target does not imply 1M active workers; it means
many idle or waiting sessions with a bounded active subset whose pressure is
measured at each owner domain. Queueing-delay telemetry should exist for
network ingress rings, mutation rings, read snapshot rings, GPU execution
rings, residency maintenance, and response rings, with rejection or fallback at
the narrowest saturated boundary.

The paper's LC/BE distinction maps to GPU DB request classes. Short retained
reads, short point writes, COPY admission, resident refresh, over-resident
scans, CPU fallback, and background maintenance should not compete as one
undifferentiated queue. Short read/write work needs latency-critical budgets;
refresh, warmup, eviction, long scans, and speculative GPU-tail work should be
throttleable when they create CPU memory-bandwidth, pinned-buffer, or GPU queue
interference.

Caladan's requirement that tasks expose internal concurrency is especially
important. The GPU DB cannot benefit from extra IO or CPU worker capacity if
the mutation owner, parser, retained route executor, or response encoder hides
all work behind one serialized path. The production design should expose
request-level or chunk-level concurrency where correctness allows it, while
keeping mutation visibility and WAL publication in owner domains.

The memory-bandwidth controller suggests a host-tier metric the current P8
plan does not yet emphasize enough: CPU-side filtering, MVCC traversal,
residency refresh, response encoding, and COPY parsing can all saturate memory
bandwidth even when GPU kernels are fast. Route admission should include CPU
memory-bandwidth pressure and host-cache miss telemetry, not only GPU queue
depth and H2D/D2H bytes.

KSCHED is not directly portable, but its shape is useful. Hot-path scheduling
should prefer preallocated shared state, asynchronous control operations,
batched wakeups, and remote/local ownership clarity. For this engine, that
means reusable command/response buffers, explicit queue state, low-allocation
work handoff, and measurable wakeup/dispatch latency before considering more
intrusive kernel or runtime changes.

**Risks and mismatches:** Caladan is an OS/runtime scheduler, not a SQL
database, MVCC engine, or GPU execution system. It does not solve
WAL-before-visibility, catalog invalidation, transaction isolation, snapshot
retirement, or CUDA stream scheduling. Its strongest results rely on a custom
runtime, kernel-bypass networking/storage, a kernel module, modified Linux IPI
support, disabled power-saving features, and a single-socket evaluation setup.

The paper also requires applications to expose concurrency in green threads.
That is a better fit for internal workers than for arbitrary pgwire clients,
transactions, prepared statements, COPY protocol state, and error recovery.
NUMA is explicitly left for future work, and transient-execution risks across
hyperthread siblings are not solved. Finally, a dedicated scheduler core may
be too heavy for an early GPU DB runtime slice; the transferable idea is the
control-loop design, not the immediate adoption of Caladan's full runtime.

**Benchmark candidates:**

- Add no-behavior-change queueing telemetry for each planned owner boundary:
  network ingress, mutation, read snapshot, residency, GPU execution, response,
  pinned-buffer acquisition, and CPU fallback. Minimum gate: p50/p95/p99 queue
  wait and service time are visible per request class.
- Build a request-class admission proof with short retained reads, COPY chunks,
  refresh work, and over-resident scans. Expected improvement: long refresh or
  scan work cannot inflate p99 latency for admitted short reads/writes without
  an explicit saturation reason.
- Add host-memory-pressure telemetry to over-resident route planning: CPU scan
  time, estimated memory bandwidth, cache-miss proxy if available, bytes
  compacted, and GPU-tail bytes. Failure condition: CPU prefiltering can
  silently starve IO or mutation work.
- Prototype a bounded active-session scheduler: keep many logical pgwire
  sessions idle, but allow only a configured number of active requests per
  class and per owner. Measure bytes per idle session, active request memory,
  queue wait, rejection/fallback reasons, and single-session throughput.
- Create a microsecond-scale maintenance throttle experiment where resident
  refresh and eviction run beside retained lookup batches. The proof gate is
  stable lookup p99 with refresh progress visible; the failure condition is
  refresh work monopolizing host memory bandwidth or response buffers.
- Measure whether exposing more internal concurrency helps or hurts: compare
  one serialized retained route executor with request/chunk-level workers that
  still publish through owner domains. Required metrics: throughput, p99
  latency, owner queue wait, correctness status, and allocation count.

### 2026-06-02 - Virtual-Memory Assisted Buffer Management In Tiered Memory

**Citation:** Yeasir Rayhan and Walid G. Aref. "Virtual-Memory Assisted
Buffer Management In Tiered Memory." arXiv:2603.03271, submitted
2026-03-03. Retrieved 2026-06-02 from the arXiv abstract and PDF,
`https://arxiv.org/abs/2603.03271` and
`https://arxiv.org/pdf/2603.03271`.

**Category:** multi-tier cache / data placement.

**Relevance tags:** tiered memory; virtual-memory-assisted buffer management;
remote memory; CXL-like memory; NUMA; page migration; stable virtual
addresses; TLB shootdown; NVMe; buffer replacement; placement economics.

**Core idea:** The paper extends vmcache-style virtual-memory-assisted buffer
management from a two-tier DRAM/disk setting to an `n`-tier
DRAM/remote-memory/disk setting. The central invariant is that a database page
keeps one stable virtual address for its lifetime while the physical frame
backing that address can move among memory-resident tiers. PID translation is
therefore delegated to the OS page table rather than a DBMS hash table, while
the DBMS still controls promotion, demotion, and eviction policy.

The added memory tiers make page migration a first-class bottleneck. The
authors build `vmcache^n` with separate resident sets per memory tier and use
Linux page-migration mechanisms to move pages without changing virtual
addresses. They also propose a custom `move_pages2` syscall that exposes
migration mode and maximum batch size to the buffer manager, reducing migration
overhead by allowing larger batched migration rounds and partial progress after
some page failures. On a 3-tier NUMA/NVMe setup, the paper reports up to about
3.82x higher TPC-C throughput than two-tier vmcache when remote memory is 4x
local DRAM capacity, and summarizes the improvement as up to 4x. The result is
not simply "add slower memory"; the economics improve only past a remote-memory
capacity break-even point because migration overhead can dominate small tiers.

**Concrete mechanisms:**

- `vmcache^n` reserves virtual address space for the database page universe at
  startup. A PID maps to a fixed virtual address; if no physical frame backs it,
  access faults or explicit fixing loads it from disk into a chosen memory tier.
- Memory-resident tiers are modeled as local DRAM, remote memory, and disk in
  the evaluated 3-tier design, but the design generalizes to additional remote
  memory tiers between DRAM and disk.
- Pages are not duplicated across memory tiers in this design. At any point a
  page has one memory-resident physical frame or is evicted to disk, preserving
  the single-address invariant.
- The design only works for memory tiers exposed in System-DRAM mode. DAX or
  App Direct style device mappings are incompatible because those mappings are
  permanently backed by device frames and cannot be remapped across tiers while
  keeping the same virtual address.
- `mbind` is used to place a page loaded from disk into a target memory tier.
  `move_pages` is used to migrate batches of already memory-resident pages
  between tiers while preserving virtual addresses and updating page tables.
- Page state extends vmcache's state word with tier-location bits. The paper
  discusses unlocked, shared-locked, locked, marked, and evicted states plus
  tier-specific variants such as unlocked-in-DRAM or unlocked-in-remote-memory.
- Each memory tier has its own cache and resident set. When a tier reaches a
  threshold, a clock replacement pass marks unlocked pages and migrates a
  selected batch downward, for example from DRAM to remote memory, or evicts to
  disk.
- Promotion from remote memory to DRAM can be triggered on access. The manager
  locks the target page, scans the source resident set for additional unlocked
  pages, and promotes a batch to amortize migration cost.
- Four probabilistic migration flags govern whether pages loaded from disk,
  written back, read from remote memory, or written in remote memory move up or
  down the hierarchy. The paper evaluates migration-ratio effects but does not
  claim one universal policy.
- Native `move_pages` batches pages but uses a fixed kernel batching policy,
  defaults to synchronous migration, and can abort early on some errors. The
  paper identifies TLB shootdowns and abort-on-failure behavior as important
  costs.
- `move_pages2` adds `migration_mode` and `nr_max_batched_migration` knobs.
  Modes include asynchronous migration, synchronous migration, and a lighter
  synchronous mode that avoids blocking on writeback. The batch-size knob lets
  the DBMS tune how many pages are migrated before TLB invalidation.
- `move_pages2` uses optimistic failure handling: it records errors for failed
  pages but migrates as many eligible pages as possible in the invocation
  instead of aborting the rest of the list.
- The implementation is described as roughly 150 Linux 6.8.0 kernel-code-line
  changes across the page-migration path.
- Evaluation uses a CloudLab dual-socket Intel Xeon Silver 4314 system: one
  socket's DRAM is treated as local memory, the other socket's DRAM as remote
  memory, and a PCIe4 NVMe SSD as disk. Workloads are TPC-C and random point
  lookup, with working sets around 190 GB and 130 GB.
- TPC-C benefits most when remote memory is large enough to reduce disk IO. The
  random-read workload sees smaller overall gains because page transfers
  between memory tiers dominate cost even when disk IO is reduced.
- The breakdown shows disk IO dominating TPC-C and memory-tier migration
  dominating random reads. Even `move_pages2` leaves migration as a major cost,
  so the paper warns that better migration mechanisms matter more than removing
  kernel/userspace crossing overhead.

**GPU DB mapping:** The strongest transferable idea is the stable logical
address plus explicit physical-tier movement split. For the GPU DB, a table
segment, resident partition, old snapshot side structure, or cold page should
have a stable logical identity used by planners and snapshots, while placement
metadata says whether the physical bytes are in GPU memory, pinned host DRAM,
ordinary host DRAM, remote/CXL-like memory, compressed host storage, or NVMe.
Readers should not chase a mutable hash-table-like placement structure on
every row if the route can resolve the segment identity once and then operate
through a stable handle.

The paper also warns against treating future CXL or remote memory as a free
capacity extension for GPU DB resident snapshots. Placement only helps when
the remote tier is large enough and migration frequency is low enough to beat
the movement cost. For P8, that means admission should avoid ping-ponging hot
segments between GPU, host DRAM, and slower host tiers. A segment whose access
pattern is random point lookup may be better left in a CPU/host index path than
repeatedly promoted and demoted around GPU execution.

`move_pages2` maps conceptually to batchable tier transitions. The GPU DB will
not literally depend on this custom syscall for GPU memory, but it should expose
the same policy knobs for its own cache manager: migration mode, maximum pages
or bytes per migration batch, partial-progress semantics, per-page status, and
queueable retry. Refresh, warmup, eviction, host-to-device transfer, device-to-
host demotion, and NVMe prefetch should all report how many pages or segments
actually moved rather than treating migration as an all-or-nothing operation.

The single-copy invariant is a useful contrast to GPU caches. The current P8
design deliberately allows durable CPU truth plus rebuildable GPU acceleration
state, so it is not the same as `vmcache^n`. Still, for each published resident
generation, the engine should avoid ambiguous multiple mutable copies. If a
segment has CPU canonical state, GPU resident state, and a remote-memory shadow,
the metadata must state which copy is authoritative for correctness, which
copies are rebuildable, and which generation each route may read.

The tier-mode limitation is also relevant. Future host tiers may be exposed as
System-DRAM, DAX/device memory, RDMA, CXL pooled memory, or storage. A single
placement abstraction cannot assume all tiers support remapping, page faults,
byte-addressability, DMA, pinning, or CUDA access. P8 should classify each tier
by the movement and access primitives it actually supports before planner cost
hooks choose a route.

**Risks and mismatches:** This is an arXiv paper rather than a peer-reviewed
conference version, and its evaluated remote tier is NUMA-remote DRAM rather
than real CXL, GPU memory, RDMA memory, or disaggregated memory. The paper is
about database pages and OS page migration, not MVCC visibility, WAL durability,
CUDA streams, GPU kernel scheduling, or SQL planning. Its single-copy invariant
conflicts with the GPU DB's cache-as-acceleration model unless applied only to
logical placement handles, not to durable correctness ownership.

The custom syscall is not a near-term dependency for this engine. Adopting a
patched Linux kernel would be a major operational burden, and GPU memory
movement uses different APIs. The practical lesson is to measure and batch tier
movement, not to assume `move_pages2` is portable. The probabilistic migration
policy is also underspecified for production GPU DB workloads; copy/admission,
retained reads, over-resident scans, and update-heavy OLTP will need telemetry-
driven policies rather than fixed probabilities. Finally, the paper shows that
random access can be migration-bound even with more memory, which is a direct
risk for point lookup workloads if the GPU DB over-promotes cold keys.

**Benchmark candidates:**

- Add a tier-transition accounting proof for P8 segments: bytes/pages promoted
  from CPU DRAM to GPU, demoted or evicted, prefetched from NVMe, and skipped
  due to pressure. Minimum gate: every retained-route decision reports current
  tier, target tier, generation, movement bytes, and fallback reason.
- Build a synthetic hot/warm/cold segment benchmark with a fixed GPU memory
  budget and a larger host-memory/NVMe dataset. Compare no promotion, greedy
  promotion, batched promotion, and partial-progress promotion. Failure
  condition: p95 query latency or write refresh latency becomes dominated by
  segment migration without visible admission backpressure.
- For random point lookup workloads, test whether resident GPU promotion helps
  after accounting for movement cost. Expected result may be negative; a CPU
  index path should win if keys are too random or the migration batch is too
  small.
- Add a "migration batch size" knob to residency warmup/refresh experiments:
  rows or pages per H2D transfer, segments per refresh batch, and maximum
  pinned-buffer bytes per batch. Measure throughput, p50/p99 latency, pinned
  memory pressure, and partial progress under cancellation or overload.
- Prototype per-segment migration status instead of all-or-nothing refresh:
  admitted, moving, valid, partial, failed, retryable, evicted. Proof gate:
  readers only use fully valid generations, while maintenance can continue
  moving other segments without blocking unrelated valid partitions.
- Create a future-tier capability matrix for GPU DB route planning: GPU HBM,
  pinned host DRAM, ordinary DRAM, System-DRAM CXL, DAX/device memory, NVMe,
  and remote memory. Include whether each tier supports direct GPU access,
  stable virtual addressing, page migration, async DMA, durable recovery, and
  cheap random access.

### 2026-06-02 - PARQO penalty-aware robust plan selection

**Citation:** Haibo Xiu, Pankaj K. Agarwal, and Jun Yang. "PARQO:
Penalty-Aware Robust Plan Selection in Query Optimization." Proceedings of the
VLDB Endowment 17(13):4627-4640, 2024.
DOI: `https://doi.org/10.14778/3704965.3704971`. Retrieved 2026-06-02 from
the PVLDB PDF and arXiv full version,
`https://www.vldb.org/pvldb/vol17/p4627-xiu.pdf` and
`https://arxiv.org/abs/2406.01526`.

**Category:** query optimization / planning.

**Relevance tags:** robust query optimization; route choice; penalty-aware
planning; selectivity uncertainty; parametric query optimization; template
cache; sensitivity analysis; fallback risk; GPU route admission.

**Core idea:** PARQO reframes robust query optimization as minimizing expected
penalty under uncertainty in selectivity estimates. Instead of asking only
whether a candidate plan is cheap at the optimizer's current estimate, it lets
the user define a penalty function relative to the true optimal plan under
possible true selectivities, models likely selectivity errors from the
workload, and selects a plan with lower expected penalty.

The paper's practical contribution is the combination of three mechanisms:
workload-informed error profiles over querylets, sensitivity analysis to find a
small set of human-interpretable selectivity dimensions that most affect
penalty, and candidate robust-plan selection over samples from the error
distribution. PARQO is implemented on PostgreSQL 16.2 by exposing optimizer
`Opt` and `Cost` calls and injecting plans/selectivities through hints, without
changing the PostgreSQL executor.

The evaluation shows why a route that looks locally cheap can be the wrong
route when estimates are fragile. On JOB, PARQO-Sobol outperforms PostgreSQL
on 19 of 33 query templates in the current-instance experiment, underperforms
on 5, and gives a reported 3.23x overall workload speedup. The paper also
reports 2.01x on DSB and 1.36x on STATS. In the parametric-query setting, it
caches robust-query-optimization work per template and uses a KL-divergence
test plus importance sampling to decide whether a previous robust-plan cache is
safe to reuse for a new parameter binding.

**Concrete mechanisms:**

- PARQO takes a query template, candidate plan space, estimated selectivities,
  and a distribution of likely true selectivities conditioned on those
  estimates.
- The default experimental penalty charges extra cost beyond the true optimal
  only after a tolerance threshold. The framework also supports other penalties
  such as probability of exceeding a tolerance, variance of extra cost, or a
  highest-density-region worst case.
- Error profiling uses querylets: single-table local-selection patterns,
  two-table join patterns, and selected three-table patterns that capture some
  dependency between local selections and joins without profiling every
  possible subquery.
- For each querylet, the system tracks estimated and actual cardinalities from
  a workload and stores sampled pairs as an error profile.
- The implementation builds low-estimate and high-estimate error models per
  relevant selectivity dimension using kernel density estimation over
  log-relative errors.
- The final selectivity-error distribution is factorized across dimensions,
  while two- and three-table querylets partly encode dependencies inside each
  dimension's error model.
- Sensitivity analysis is done on the penalty function, not merely on the
  candidate plan's cost function. This asks which dimensions affect plan
  optimality risk, not only which dimensions change the chosen plan's local
  cost.
- PARQO adapts Sobol's global variance-based sensitivity analysis to estimate
  how much each selectivity dimension contributes to variance in expected
  penalty. It also evaluates Morris as a cheaper alternative, but Sobol is the
  stronger method in the experiments.
- Sensitive dimensions are interpretable as selection/join condition
  combinations, so they can become tuning hints: reanalyze stats, sample a
  specific predicate/join, or inspect a dependency that likely causes bad plan
  choice.
- Robust-plan search samples true selectivity vectors from the learned error
  distribution. For each sample, it asks the optimizer for the optimal plan and
  caches the sample, plan, and true-optimal cost.
- Unique sampled optimal plans form the candidate pool. PARQO estimates each
  candidate's expected penalty over the cached samples and chooses the minimum.
- For parametric query optimization, a query template can reuse previous
  robust-optimization work. PARQO checks KL-divergence between the previous
  and new conditional selectivity distributions before reusing sensitive
  dimensions, candidate plans, and cached samples.
- When cached samples are reused for a different parameter binding, importance
  sampling reweights the expected-penalty estimate instead of rerunning all
  optimizer calls.
- Reported model footprints are small in the evaluated benchmarks: about
  13.8 KB for JOB, 13.66 KB for DSB, and 5.84 KB for STATS. The up-front
  robust optimization cost is substantial, for example about 2.13 hours for
  all 33 JOB templates, so the technique is most attractive for repeated
  templates.
- The paper explicitly notes unresolved issues: no theoretical guarantee that
  the chosen candidate is globally optimal for the robustness objective, a
  possibility that the best robust plan is not optimal at any sampled point,
  imperfect error profiles, and open problems around workload drift.

**GPU DB mapping:** The GPU DB planner will face route choices with exactly the
kind of asymmetric risk PARQO models. A resident GPU route may be fastest when
cardinality, residency, queue delay, and transfer estimates are right, but it
can be a bad choice when a predicate is less selective than expected, a
partition is not resident, the GPU queue is saturated, a refresh is pending, or
the CPU fallback path would avoid transfer and launch overhead. A penalty-aware
route selector is a better fit than a single expected-latency score for these
decisions.

The most direct adaptation is to define GPU-route penalties in terms Richard
cares about: p99 query latency above an SLO, write-path interference, refresh
starvation, pinned-buffer budget exhaustion, and correctness-preserving
fallback/rejection. For example, the planner could treat "route exceeded CPU
fallback by more than 20%" or "route caused snapshot queue wait above a
microsecond budget" as penalty events and track which estimate dimensions cause
that risk.

PARQO's sensitive dimensions map well to GPU DB route explainability. Instead
of only logging "GPU route rejected" or "CPU fallback chosen," the planner
should expose whether the fragile dimension is predicate selectivity, resident
partition byte count, output cardinality, H2D/D2H bytes, GPU execution queue
wait, refresh age, pinned-buffer availability, or mutation-invalidation
probability. Those dimensions are actionable: refresh stats, split partitions,
disable a fragile GPU route for a template, raise a cache budget, or set a
tighter admission threshold.

The parametric-cache mechanism is also important. The current retained routes
are repeated templates with changing literals, such as point lookups, range
filters, and grouped aggregates. PARQO suggests storing a plan/route profile per
template, then reusing it only when a distribution-distance gate says the new
literals are close enough to the previous route-risk distribution. This is
cleaner than blindly caching exact encoded responses or blindly reusing a GPU
route for all parameters of the same SQL shape.

For high-concurrency sessions, the penalty model should include runtime state,
not just static cardinality. A route profile may be safe at low load and unsafe
when the GPU execution ring, response ring, or residency owner is saturated.
The paper does not solve this dynamic case, but its framework can be extended:
the "selectivity vector" becomes a broader route-risk vector containing
estimated rows, resident bytes, queue depths, transfer bytes, and invalidation
age.

**Risks and mismatches:** PARQO is a query-optimizer paper, not a GPU database,
transaction system, MVCC design, or runtime scheduler. It assumes the optimizer
can provide exact-ish `Opt` and `Cost` interfaces under injected selectivities
and candidate plans; the current GPU DB planner is far simpler and may not have
enough alternative plans to justify the full machinery yet. The evaluation
depends on PostgreSQL cost estimates and hints, not real GPU execution costs.

The approach also has real overhead. Its up-front robust optimization cost is
only amortized when query templates repeat many times or when queries are
expensive enough that route mistakes dominate. For short OLTP point lookups,
full Sobol/PARQO analysis at request time would be unacceptable; any GPU DB
adaptation must be offline, background, or amortized by template.

The error model is only as good as the profiling workload. PARQO's factorized
model can miss long-range predicate/join dependencies, and GPU DB route risk
will add dimensions the paper does not cover: memory residency, GPU queue
pressure, host memory bandwidth, refresh state, and WAL/MVCC invalidation.
Finally, choosing a robust plan may sacrifice best-case latency; that is
desirable only when the penalty function matches product goals.

**Benchmark candidates:**

- Add a route-risk logging proof for retained routes. For each routed query,
  record estimated rows, actual rows, resident bytes, output rows, GPU queue
  wait, H2D/D2H bytes, execution time, response encoding time, fallback reason,
  and whether CPU fallback would likely have beaten the GPU route.
- Build a penalty-aware CPU/GPU route benchmark for one repeated template:
  compare expected-latency routing, always-GPU, always-CPU, and
  penalty-aware routing under skewed predicate literals. Minimum gate: the
  penalty-aware route reduces p99 or bad-route count without losing more than a
  defined amount of p50.
- Add a route-template cache with a reuse gate. Start with simple distance
  over literal selectivity bucket, resident-partition set, and queue-pressure
  bucket; reject reuse when the route-risk vector changes too much.
- Turn fallback explanations into sensitive-dimension counters: predicate
  selectivity error, partition residency, transfer bytes, GPU queue pressure,
  refresh/invalidation age, pinned-buffer pressure, and output cardinality.
  Proof gate: the top risk dimension is visible for every rejected or
  regretted GPU route.
- For micro-batched retained lookups, test whether route reuse by template and
  key-distribution bucket avoids bad batching decisions. Failure condition:
  grouped GPU execution worsens p99 compared with CPU/index fallback for
  sparse or highly skewed keys.
- Create an offline "route replay" harness from query telemetry. Re-evaluate
  historical requests under alternative penalty functions: p99-first,
  write-interference-first, GPU-throughput-first, and balanced HTAP. The
  useful output is not one universal route policy, but a measurable policy
  frontier.

### 2026-06-02 - Third Modern Batch Synthesis

**Scope:** Oze, Caladan, virtual-memory-assisted tiered buffer management
(`vmcache^n`), and PARQO.

**Converging design tracks:** The last four modern papers push the same engine
shape from different sides: do not let mutable shared state or hidden resource
contention decide behavior implicitly. Oze makes dependency tracking explicit
so long updates do not create unnecessary false conflicts. Caladan makes
runtime pressure explicit at microsecond-scale resource boundaries. `vmcache^n`
makes tier placement and migration explicit while preserving stable logical
identity. PARQO makes route-choice risk explicit through penalty functions,
error profiles, and reuse gates.

For the GPU DB, the shared direction is a measured owner-domain system where
each request carries enough metadata to explain its path: visibility boundary,
dependency or invalidation relation, resident generation, tier location,
runtime queue pressure, and route-risk dimensions. The product target is not
"always push to GPU"; it is predictable admission and routing where the engine
knows when the GPU route is valid, when it is fast, and when it is too fragile.

**Category gaps:** The journal now has good coverage in tiering, runtime
scheduling, robust route choice, GPU execution, and MVCC/read snapshots. The
next few entries should keep pulling OLTP write-path/concurrency and MVCC
storage forward: TicToc, Cicada, ERMIA, Fast Serializable MVCC, Shirakami,
Aria, Chiller, or Memory-Optimized MVCC for Disk-Based Systems are better
balance choices than another GPU OLAP paper unless a specific GPU mechanism is
needed.

**Benchmark priorities:**

- Add a route decision record for every retained query: snapshot generation,
  resident partition set, tier location, queue wait, transfer bytes, output
  rows, risk/fallback reason, and actual elapsed time.
- Build a small policy comparison harness for CPU fallback versus retained GPU
  execution under skewed parameters and queue pressure.
- Tie tier movement to admission: refresh, promotion, demotion, and eviction
  should expose partial progress, batch size, and saturation reasons before
  readers observe a new generation.
- Measure whether bounded active-session scheduling protects short retained
  reads and writes while long refresh/scan work continues to make progress.
- Preserve WAL-before-visibility and immutable snapshot publication as the
  hard correctness boundary; every reviewed mechanism should fit around that
  boundary rather than weakening it.

### 2026-06-03 - TicToc data-driven timestamp OCC

**Citation:** Xiangyao Yu, Andrew Pavlo, Daniel Sanchez, and Srinivas
Devadas. "TicToc: Time Traveling Optimistic Concurrency Control." SIGMOD
2016, pages 1629-1642. DOI:
`https://doi.org/10.1145/2882903.2882935`. Retrieved 2026-06-03 from the
author-hosted PDF, `https://db.cs.cmu.edu/papers/2016/yu-sigmod2016.pdf`.

**Category:** transaction processing / write path and concurrency control.

**Relevance tags:** optimistic concurrency control; serializability; timestamp
allocation; per-tuple visibility metadata; write-set validation; read-set
validation; contention telemetry; snapshot isolation variant; WAL batching;
high-throughput OLTP.

**Core idea:** TicToc removes the global timestamp allocator from
timestamp-order concurrency control. Instead of assigning a transaction a
timestamp before or during execution, each tuple version carries a write
timestamp (`wts`) and read timestamp (`rts`) that define the logical interval
where that version is valid. A transaction records the tuple values and
timestamps it reads or writes, then lazily computes a commit timestamp during
validation from the data it actually touched.

The important shift is that logical serialization order is data-driven rather
than physical-time driven. Two transactions that overlap physically can still
commit if the accessed tuple timestamp intervals admit a serial order, even
when conventional OCC would abort because a read tuple has changed since it was
first observed. TicToc proves serializability by ordering transactions by
commit timestamp and physical commit time when timestamps tie.

The evaluation implements TicToc in DBx1000 and compares against Silo,
Hekaton-style MVCC, two-phase locking with deadlock detection, and no-wait
2PL on a 40-core, 80-hardware-thread, four-socket machine. The paper reports
up to 92% higher throughput than prior algorithms and up to 3.3x lower abort
rate under evaluated workload conditions. In the high-contention four-warehouse
TPC-C variant, TicToc achieves 1.8x better throughput than Silo and 27% lower
abort rate; in medium-contention YCSB, TicToc and Silo have similar
throughput, but TicToc has about 3.3x lower abort rate. Under very high
write contention, TicToc's abort-rate advantage shrinks and the no-wait plus
preemptive-abort optimizations carry more of the performance gain.

**Concrete mechanisms:**

- Every tuple version stores `wts` and `rts`. A version is valid for reads when
  `wts <= commit_ts <= rts`; a write is valid when the new transaction's
  `commit_ts` is greater than the previous version's `rts`.
- Read phase is non-blocking. The transaction stores read-set and write-set
  entries containing tuple pointer, copied data, `wts`, and `rts`.
- The tuple value and timestamp word must be read atomically so the value
  matches the metadata. TicToc implements this with a 64-bit timestamp word
  containing a lock bit, a 15-bit `rts - wts` delta, and a 48-bit `wts`.
- Validation first locks write-set tuples in primary-key order, then computes
  the candidate commit timestamp as the maximum of each read entry's `wts` and
  each write entry's current `rts + 1`.
- Read validation checks whether each read version is valid at `commit_ts`.
  If the copied `rts` is too small, the system tries to extend the tuple's
  current `rts` with compare-and-swap, as long as the tuple's `wts` still
  matches and the tuple is not locked by another transaction outside the
  current write set.
- Write phase installs each write-set value and sets the tuple's `wts` and
  `rts` to `commit_ts`, then unlocks.
- The no-wait optimization aborts and retries validation immediately if a
  write-set lock cannot be acquired, avoiding lock convoying in the commit
  phase.
- The preemptive-abort optimization uses an approximate commit timestamp and
  latest read-tuple `wts` checks to identify transactions that will fail
  read-set validation before they lock the write set.
- The timestamp-history optimization keeps a bounded history of recent `wts`
  values per tuple to avoid some unnecessary aborts, but the paper reports no
  measurable performance gain for its evaluated workloads.
- TicToc sketches snapshot isolation by splitting one serializable timestamp
  into `commit_rts` for reads and `commit_wts` for writes, while checking that
  updated tuples were not modified after the read timestamp.
- For durability, the paper says TicToc can use conventional logging and
  sketches parallel logging batches by forcing transactions in a later batch to
  choose commit timestamps greater than previous-batch timestamps. Scalable
  logging itself is left out of scope.

**GPU DB mapping:** TicToc is most useful as a warning against a single global
transaction or snapshot counter on the GPU DB write path. A future write-heavy
or high-session engine should avoid turning timestamp allocation into the
central bottleneck that every admitted session, mutation owner, read-snapshot
worker, and residency refresh has to touch. Per-record or per-segment
visibility metadata can let independent partitions commit without global
coordination when their conflict sets are disjoint.

For the current P8 design, the nearest transfer is a per-resident-segment
visibility interval. The resident snapshot metadata already needs source WAL
boundary, read timestamp, invalidation generation, and partition identity.
TicToc suggests making those boundaries composable at the data item, segment,
or partition level: a retained read can prove that its snapshot generation is
valid over the transaction's logical read interval instead of simply asking
whether it is the latest physical generation.

TicToc's `rts` extension maps to a possible "read lease extension" for CPU
truth or host-resident versions, but it should not mutate an already-published
GPU snapshot in place. For GPU DB, extension should be owned by the mutation or
visibility owner and should publish a new metadata generation or update only
CPU-side visibility metadata before a snapshot is shared with readers. The hard
boundary remains WAL-before-visibility and immutable retained snapshot
publication.

The paper's logical-time growth measurement is a strong benchmark idea. In GPU
DB, the rate at which per-partition visibility clocks advance relative to
committed transactions can expose actual contention. If logical time grows
slowly while commit count grows quickly, the workload has enough disjointness
for partition-owned commit, read-snapshot sharing, and retained GPU batching.
If one tuple or partition forces every commit to advance the same clock, the
engine should surface that as hot-key or hot-partition admission pressure.

The no-wait and preemptive-abort mechanisms map well to high-concurrency
session admission. A transaction that cannot acquire a narrow write-set or
partition-owner slot should quickly release any partial resources and requeue
or reject with an explicit conflict reason instead of sitting on scarce GPU
staging buffers, response-ring space, or owner-domain locks.

**Risks and mismatches:** TicToc is an in-memory, shared-everything OLTP
concurrency-control paper, not a GPU execution or multi-tier cache paper. Its
tuple-level timestamp word assumes cheap CPU atomics and cache-coherent memory;
that mechanism should not be copied directly into GPU kernels or durable
resident snapshots.

The paper does not solve scalable logging, durable recovery ordering, phantom
prevention for serializable index scans, or multi-version storage retention.
It notes that serializable scanning needs extra index locking or validation,
and leaves applying data-driven timestamps to order-preserving indexes as
future work. That matters for GPU DB range scans and retained column snapshots:
tuple visibility alone is insufficient if predicates can miss inserted rows.

TicToc can also pick logical commit orders that differ from physical commit
order. That is acceptable only if every downstream system agrees on logical
visibility boundaries: WAL records, CPU indexes, resident GPU generations,
response publication, and replay must not accidentally assume physical commit
order is the serialization order.

Finally, the weaker-isolation support is only sketched. The GPU DB should not
adopt a split read/write timestamp model without a precise SQL isolation
contract and tests that cover writes racing retained reads, refresh, eviction,
and replay.

**Benchmark candidates:**

- Add a synthetic timestamp-allocation microbenchmark for the current write
  path: global atomic transaction id, per-partition logical clock, and
  per-segment visibility interval. Minimum gate: report committed rows/sec,
  abort/retry count, p50/p99 commit latency, and cache-line contention under
  disjoint keys and hot-key workloads.
- Track logical visibility-clock growth per relation or partition during COPY,
  INSERT, and future UPDATE workloads. Compare committed transaction count to
  maximum visibility-clock advance. Use the ratio as a contention signal for
  partition ownership and micro-batch eligibility.
- Prototype a CPU-only per-segment visibility interval proof before any GPU
  integration. Readers should validate that a retained snapshot is compatible
  with their read boundary; writers should invalidate or publish new generation
  metadata without mutating published GPU buffers.
- Add a no-wait validation/admission experiment for write-set ownership:
  compare waiting for owner locks versus immediate release/retry under hot
  partitions. Failure condition: retries improve throughput but blow up p99 or
  starve long transactions.
- Extend route telemetry with a conflict reason vocabulary: timestamp clock
  contention, write-set lock conflict, read interval not extensible,
  invalidated resident generation, index/range phantom risk, and WAL batch
  boundary. Proof gate: every aborted or retried write reports exactly one
  primary reason.
- For future snapshot isolation work, test a split read/write timestamp model
  against retained GPU reads: read timestamp chosen before execution, write
  timestamp chosen at commit, and resident generation validated against both.
  Failure condition: any stale retained read can pass after a WAL-visible
  mutation invalidates its segment.

### 2026-06-03 - Shirakami hybrid long-transaction MVCC and short-transaction OCC

**Citation:** Takayuki Tanabe, Shinichi Umegane, Suguru Arakawa,
Ryoji Kurosawa, Takashi Hoshino, Hideyuki Kawashima, Masahiro
Tanaka, and Takashi Kambayashi. "Shirakami: A Hybrid Concurrency
Control Protocol for Tsurugi Relational Database System." arXiv
2303.18142v2, 2026. Retrieved 2026-06-03 from
`https://arxiv.org/abs/2303.18142` and
`https://arxiv.org/pdf/2303.18142`.

**Category:** transaction processing / write path and MVCC / visibility.

**Relevance tags:** hybrid concurrency control; long read-write transactions;
short OLTP transactions; MVCC; OCC; epoch scheduling; write preservation;
phantom avoidance; serializable HTAP; WAL batching; snapshot publication.

**Core idea:** Shirakami targets a workload shape that ordinary OLTP
benchmarks underrepresent: a production database that must run many short
transactions while also allowing long read-write business transactions such as
billing, cost calculation, and batch updates to commit during online activity.
The system combines two protocols instead of dynamically switching one
protocol. Shirakami-LTX handles long read-write transactions with a wider
multiversion view-serializable scheduling space, while Shirakami-OCC keeps a
Silo-like fast path for short transactions.

The main transfer for GPU DB is the explicit separation of transaction classes.
Long work is not hidden inside the same optimistic short-transaction path and
then left to repeatedly abort. It declares enough future write intent to let
short transactions see conflicts early, starts on epoch boundaries, and uses
priority plus order forwarding so some apparent conflicts can still serialize
validly. Short transactions keep the cheap OCC path, but they validate against
long-transaction write preservation and register enough read metadata for long
transactions to avoid breaking already-committed short reads.

The paper implements Shirakami in Tsurugi, a production-grade relational
database system. It reports Tsurugi completing the phone billing benchmark
where PostgreSQL times out at high online concurrency, with 19.7x lower
latency than PostgreSQL at 16 online threads. It also reports 5.6x better
elapsed time on a bill-of-materials benchmark at serializable behavior, while
PostgreSQL READ COMMITTED is faster but anomalous. In direct Shirakami
key-value experiments, S-LTX can outperform S-OCC by up to 680x for rare, long
mixed transactions, but LTX overhead is visible when "long" transactions are
short or frequent.

**Concrete mechanisms:**

- Transactions are classified as S-LTX for long read-write work or S-OCC for
  short work. S-LTX has higher priority than S-OCC, and earlier S-LTX
  transactions have priority over later S-LTX transactions.
- Shirakami maps transaction serialization to epochs. S-OCC serializes at its
  closing epoch; S-LTX serializes at its opening epoch. Within one epoch,
  S-LTX transactions are placed before S-OCC transactions.
- S-LTX transactions are staged for the next epoch, share an epoch snapshot,
  and register write preservation before they begin. Epoch advancement is
  briefly stopped while write preservation is registered so short OCC reads do
  not miss the declared future write area.
- Write preservation is table-level. The paper chooses coarse granularity
  because long transactions with SQL subqueries may not know exact record keys
  before execution, and record-level declaration would be expensive for large
  transactions.
- S-LTX reads check write preservation and record full-scan, range-scan, or
  point-search read information. Writes are buffered locally until validation.
- Order forwarding lets a lower-priority S-LTX transaction that read a version
  later overwritten by a higher-priority S-LTX move before that writer in the
  serialization order, if the epoch lower-bound checks still allow it. This
  admits schedules outside conflict serializability and ordinary MVTO.
- S-LTX commit waits for relevant higher-priority transactions, applies order
  forwarding where possible, aborts when the computed epoch would violate a
  lower bound, and validates writes against registered reader epochs.
- S-OCC follows a Silo-like read and commit path, but its commit validation
  also checks conflicting write preservation. It registers read epochs so S-LTX
  write validation can detect when forwarding would invalidate short
  transactions.
- Phantom avoidance is split by protocol. S-LTX stores predicate-read metadata
  for full scans, range scans, and point searches, and writer-side validation
  aborts writes that would create phantoms. S-OCC either observes live write
  preservation, validates nodes after committed S-LTX writes, or is protected
  by read clues already recorded for committed S-OCC predicate reads.
- The system distinguishes unsafe and safe in-memory snapshots. Unsafe epoch
  snapshots may become inconsistent if later order forwarding changes the
  epoch's serialization contents, so readers can still abort. Safe snapshots
  exist when an epoch closes without order forwarding; read-only transactions
  can use them as a read-only optimization.
- Epoch logging passes precommitted records to Limestone asynchronously. The
  implementation uses WAL, pre-write, non-visible write omission, and
  separately maintained snapshot storage for read-friendly persisted images.
- Write-preservation objects use optimistic locking: a 64-bit word contains a
  lock bit and version counter, and fixed-length arrays sized by possible
  concurrent workers avoid dynamic-size races.
- Tsurugi's surrounding architecture includes a SQL engine with DAG/dataflow
  execution, a transaction pool, catalog cache, a Masstree-derived concurrent
  index called Yakushima, and a log datastore that separates WAL-like log
  storage from asynchronous snapshot storage.

**GPU DB mapping:** Shirakami strengthens the case for classifying requests
before admission. GPU DB should not push short retained reads, COPY chunks,
long refreshes, over-resident scans, and future read-write analytical
transactions through one undifferentiated owner queue. Each request should
carry a class: short retained read, short mutation, COPY batch, long refresh,
long read-only scan, or long read-write transaction. The class should decide
priority, queue, snapshot requirements, write intent, and whether retry,
fallback, or delay is acceptable.

Write preservation maps to a GPU DB "future invalidation intent" record. A
long refresh or long read-write transaction could declare table, partition, or
predicate-family write intent before it starts so short reads and writes know
whether they are racing a higher-priority epoch. Table-level declarations are
too coarse for all GPU DB workloads, but they are a useful first proof for
tables whose retained generations will be rebuilt wholesale. Later, partition
or segment-level write preservation should reduce false positives.

The epoch model maps directly to immutable retained snapshot publication. A
mutation or residency owner can publish generation `N`, stage long work for
generation `N+1`, and decide whether a snapshot is safe for wait-free retained
reads. If order forwarding or long write work can still rewrite an epoch's
logical contents, GPU DB should not treat the generation as a reusable
read-only retained snapshot for new requests.

Shirakami also suggests a clearer admission rule for long work. Long
transactions should not be allowed to occupy scarce pinned buffers, GPU queue
slots, response buffers, or mutation-owner locks while repeatedly losing to
short traffic. Either reserve coarse write intent and give the long work
priority at a known epoch boundary, or route it to a lower-priority background
class whose progress is explicitly best effort.

The safe/unsafe snapshot split is valuable for P8. Current resident snapshots
should be treated as safe only when their source boundary cannot be reordered
by pending mutation, refresh, or long transaction work. Unsafe snapshots may
still be useful for speculative CPU work or internal refresh construction, but
they should not become SQL-visible GPU route handles unless validation and
abort paths are complete.

**Risks and mismatches:** Shirakami is a CPU in-memory transaction engine, not
a GPU execution system. Its write preservation is table-level, which may
create too many false conflicts for high-throughput retained reads unless GPU
DB narrows the declaration to partition, segment, predicate family, or route
family. The protocol assumes the system can identify long transactions before
execution, which is hard for ad hoc SQL unless the planner or application
declares the route class.

The reported Tsurugi/PostgreSQL comparisons use different client APIs and
benchmark implementations, and the paper explicitly says the absolute numbers
should be read as workload-level comparisons rather than identical-program
database bakeoffs. The Shirakami-only experiments are more directly about the
protocol, but they bypass SQL. The paper also shows LTX overhead when long
transactions are short or frequent, so blindly routing medium work to the long
path could hurt latency.

Order forwarding is subtle. GPU DB cannot let logical serialization changes
race WAL durability, resident generation publication, response emission, or
GPU snapshot reuse. Any adaptation must prove that WAL-before-visibility,
phantom avoidance, resident invalidation, and replay all agree on the same
logical order.

**Benchmark candidates:**

- Add request-class admission telemetry: short retained read, short mutation,
  COPY batch, long refresh, long read-only scan, and long read-write
  transaction. Proof gate: every request has exactly one class, one owner
  queue, one priority policy, and one overload/fallback vocabulary.
- Prototype table-level write preservation for one long refresh or bulk update
  path. Short reads should either see the declared future invalidation and
  choose a compatible snapshot/fallback, or report a precise conflict reason.
  Failure condition: a short retained read can route to a generation that a
  higher-priority long write has already declared unsafe.
- Compare table-level versus partition-level write preservation in a synthetic
  mixed workload: many short point reads/writes plus one long partition refresh
  or read-write scan. Required metrics: false conflict count, long-work commit
  latency, short-work p99 latency, abort/retry count, and queue occupancy.
- Add a safe/unsafe resident generation state to a CPU-only prototype. Safe
  generations can serve retained reads; unsafe generations require validation
  or stay internal to refresh. Minimum gate: generation state transitions are
  deterministic under mutation, refresh, abort, and WAL replay tests.
- Build a long-transaction starvation test: keep short writes arriving while a
  long refresh or read-write transaction tries to commit. Compare ordinary OCC
  retry, epoch-priority admission, and write-preservation staging. Failure
  condition: the long path either starves or protects itself by letting short
  p99 latency explode without explicit admission telemetry.
- Extend route telemetry with phantom-risk dimensions for range/prefix routes:
  predicate family, read range, declared write area, source generation, and
  whether a future write could have invalidated the predicate result. Proof
  gate: every range or prefix retained route can explain why concurrent
  writes cannot create a phantom visible to its SQL result.

### 2026-06-03 - Memory-optimized MVCC for disk-backed storage

**Citation:** Michael Freitag, Alfons Kemper, and Thomas Neumann.
"Memory-Optimized Multi-Version Concurrency Control for Disk-Based Database
Systems." PVLDB 15(11), 2022, pages 2797-2810. DOI:
`https://doi.org/10.14778/3551793.3551832`. Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol15/p2797-freitag.pdf`.

**Category:** MVCC / snapshot / visibility and multi-tier storage.

**Relevance tags:** disk-backed MVCC; ephemeral version chains; buffer
management; page-local mapping tables; bulk-write isolation; WAL recovery;
garbage collection; long-reader robustness; tiered storage; snapshot
publication.

**Core idea:** The paper argues that the old split between pure in-memory
OLTP engines and traditional disk-based engines is no longer the right design
boundary. A modern disk-backed DBMS can keep hot working sets in large DRAM
buffers and use SSD/NVMe for scale, but its MVCC layer must avoid persisting
every old version into the database files. The common case is small OLTP write
transactions whose version data fits easily in memory, so version chains can
be treated as ephemeral concurrency-control state while WAL remains the
durable recovery authority.

Umbra implements this by storing only the latest object value on buffer-managed
database pages and keeping before-images in transaction-local in-memory
version buffers. A small in-memory mapping table is attached to each page that
currently has versioned objects, linking stable tuple/object ids to version
chains. Pages can still be evicted, but only the page data is written to disk;
orphaned mapping tables remain in memory and are reattached when the page is
loaded again. Very large write transactions use a separate virtual-version
fallback so they do not allocate unbounded version memory.

The evaluation claims up to an order-of-magnitude transaction-throughput
advantage over PostgreSQL and a commercial disk-based system in TATP/TPC-C
under the tested configuration. More transferably, the detailed experiments
show that enabling snapshot isolation adds about 1.2x overhead versus
non-transactional Umbra, while forcing append-only physical version storage
causes more than a 5x throughput drop. Under constrained buffer memory, Umbra
keeps MVCC memory roughly bounded while the database grows far beyond the
buffer pool.

**Concrete mechanisms:**

- Persistent database pages contain only the newest version of each data
  object. Older before-images are stored in transaction version buffers and
  linked into per-object chains.
- A buffer frame may hold a pointer to a page-local mapping table. The table
  maps stable logical object ids on that page to the head of the in-memory
  version chain.
- The page latch protects both page contents and the mapping table pointer, so
  normal page access does not require a separate global version-map lookup.
- If a versioned page is evicted, the mapping table is retained by the buffer
  manager in an orphan table keyed by page id and reattached when the page is
  cached again.
- Pages without mapping tables are known to have no active version chains, so
  scans can use a cheaper path that reads visible non-deleted objects without
  per-object version lookups.
- Commit processing retimestamps versions in transaction-local buffers and
  does not latch database pages or mapping tables. Empty mappings are pruned
  later during ordinary page maintenance.
- Garbage collection uses active and recently committed transaction lists.
  Version buffers become reclaimable once their commit timestamp is older than
  the minimum start timestamp of active transactions.
- Mapping tables are pruned opportunistically when pages are accessed, during
  buffer-manager work on cold/orphaned mappings, and after empty-chain ratios
  pass a threshold.
- Recovery discards all in-memory MVCC structures. WAL replay rebuilds the
  durable latest page state, and the MVCC subsystem restarts with empty
  version chains and fresh timestamp state.
- Rollback is coordinated with ARIES-style logging by scanning log records,
  writing compensation log records, restoring before-images on pages, and
  unlinking irrelevant versions.
- Bulk operations take exclusive write access while read transactions can
  continue. They create virtual creation/deletion versions by storing one page
  reference epoch plus per-object flags instead of allocating physical
  before-images.
- A persistent bulk-operation epoch makes virtual versions visible after the
  bulk transaction commits. A later bulk operation must wait until prior
  virtual versions are globally visible because one page reference epoch cannot
  represent multiple bulk visibilities.
- The paper supports snapshot isolation; serializability is discussed as a
  possible extension using precision locking, with bulk writes requiring read
  repetition rather than version-buffer scans.

**GPU DB mapping:** The strongest transfer is the separation between durable
truth and rebuildable concurrency acceleration. GPU DB already treats WAL,
checkpoint, archive, and CPU MVCC state as correctness authority while GPU
resident snapshots are acceleration state. This paper suggests the same rule
for CPU-side MVCC auxiliaries: per-page or per-segment version maps, retained
snapshot handles, old-snapshot side structures, resident key vectors, and GPU
visibility summaries should be rebuildable whenever possible rather than
written into the durable table format.

The page-local mapping table maps naturally to a segment-local visibility map
for P8. A resident or host-cached segment could carry a compact optional map
from row ordinal or stable row id to a version/delta chain only when that
segment has active MVCC history. Segments without a map become fast-path
inputs for retained GPU scans and lookups because the kernel or CPU prepass
can know that the latest resident values are globally visible for the relevant
snapshot class.

The orphan mapping-table idea is useful for tiering. If a warm host segment or
cold NVMe page is evicted from the CPU buffer pool while old snapshots still
need its version chain, the engine can retain a small in-memory metadata
object rather than forcing old versions into the durable page. For GPU DB,
that points to evicting heavy resident buffers while preserving compact
snapshot metadata until retained readers release it.

The bulk virtual-version path is a strong model for COPY, refresh, and large
partition rebuilds. Instead of allocating one physical version record per
inserted row during a huge load or refresh, a large operation can publish an
epoch/generation and mark segment-level creation/deletion state. Short reads
continue against older safe generations, while new reads see the bulk epoch
only after WAL safety and publication. The mismatch is that GPU DB needs
partition- or segment-level granularity; a single database-wide exclusive
bulk-writer latch would be too coarse for high session concurrency.

The performance lesson also strengthens the P8 storage direction. Appending
all versions into durable table/storage files is likely to hurt write
throughput, scan locality, and resident snapshot refresh cost. A hybrid row
log plus generated column-group snapshot can keep current values and durable
WAL compact while treating historical versions as bounded, explicitly
collected metadata tied to active snapshots.

**Risks and mismatches:** Umbra is a CPU DBMS with a buffer-managed page
layout, not a GPU-resident execution engine. Raw pointer version chains,
page latches, and buffer-frame pointers are not directly portable to device
memory. The paper assumes small OLTP updates dominate and that large write
transactions can be serialized behind an exclusive write gate; GPU DB may need
concurrent partition-local bulk loads and refreshes.

The in-memory version data is ephemeral, so correctness depends on WAL replay
being able to recover a globally consistent latest state without reconstructing
active transaction state. GPU DB must preserve that property before using any
similar host/GPU auxiliary map. The paper also targets snapshot isolation and
does not implement serializable validation; range/prefix retained routes still
need phantom protection before broader SQL claims.

Finally, the paper's experiments use asynchronous commit and a CPU storage
engine on Optane/NVMe-era hardware. The absolute throughput numbers are not
GPU DB predictions. The transferable claims are the physical versioning shape,
the common-case/fallback split, and the evidence that persisting every version
can dominate transaction cost.

**Benchmark candidates:**

- Prototype a CPU-only segment-local version map for one relation: latest
  values in the segment, optional row-id to version-chain mapping only for
  rows with history, and no map for globally visible segments. Minimum gate:
  identical snapshot results before and after eviction/reload of the segment
  metadata.
- Add telemetry for "unversioned fast-path segment" versus "versioned segment"
  retained reads. Measure p50/p99 lookup and scan latency, version-map lookup
  count, and branch/copy overhead under short snapshots and one long retained
  snapshot.
- Build a bulk-COPY virtual-epoch proof: load a large partition with
  generation-level creation markers instead of per-row version allocation,
  publish visibility only after WAL safety, and let older retained reads finish
  on the previous safe generation.
- Compare physical append-only old-version storage against ephemeral
  before-image buffers for a write-heavy microbenchmark. Required metrics:
  committed rows/sec, WAL bytes, durable table bytes, resident refresh bytes,
  GC work, and scan locality after many updates.
- Add orphan snapshot-metadata accounting: evict resident GPU/host buffers but
  retain compact metadata needed by active readers. Failure condition:
  metadata retained for one long snapshot grows with table size instead of
  number of versioned segments.
- For future serializable routes, test whether segment-local version maps can
  expose enough write/read conflict information for range or prefix predicate
  validation. Failure condition: point updates under a retained range scan can
  create a phantom without a visible route-risk reason.

### 2026-06-03 - Fourth Modern Batch Synthesis

**Scope:** TicToc, Shirakami, and memory-optimized disk-backed MVCC.

**Converging design tracks:** These three papers point toward a GPU DB write
and visibility model that is neither one global timestamp nor one monolithic
owner queue. TicToc argues for data-driven logical time so independent
conflict sets do not serialize on a global allocator. Shirakami argues for
explicit transaction classes and epoch boundaries so long work can coexist
with short work without endless aborts. Umbra's disk-backed MVCC argues for
ephemeral version metadata and durable latest-state/WAL authority, with a
separate fallback for large writes.

For GPU DB, the practical design track is partition- or segment-owned
visibility publication: short mutations use cheap per-partition metadata,
large COPY/refresh work declares its class and future invalidation boundary,
and retained reads execute only from safe immutable generations. Historical
version state should stay compact, local, and rebuildable where possible.
Durable storage should not become bloated just to accelerate active snapshots.

**Category gaps:** The journal has recent momentum in OLTP/MVCC again after a
previous GPU/tiering/optimizer run. The next high-value balance choices are
runtime/session scheduling (`Shenango`, `Shinjuku`, `ZygOS`, `Arachne`),
multi-tier placement (`LeanStore`, `Umbra`, `Nomad`, `Towards Buffer
Management with Tiered Main Memory`), or query route planning
(`PAR2QO`, `Kepler`, `Lero`) before returning to another GPU OLAP engine.

**Benchmark priorities:**

- Define the first CPU-only visibility-publication prototype around
  relation/partition/segment generations, safe versus unsafe generation state,
  and WAL-before-visibility publication.
- Measure whether per-partition logical clocks, data-driven intervals, or one
  global transaction id best predict real contention under COPY, hot-key
  updates, and retained read concurrency.
- Add request-class and generation metadata to route telemetry so each read,
  write, COPY batch, refresh, and long scan can explain its owner queue,
  snapshot generation, conflict reason, and fallback path.
- Keep version history out of durable table files unless a benchmark proves it
  is necessary. The first proof should compare durable append-only versions
  against ephemeral/local version metadata under write-heavy and long-snapshot
  mixes.

### 2026-06-03 - Low-latency transaction scheduling via userspace interrupts

**Citation:** Kaisong Huang, Jiatang Zhou, Zhuoyue Zhao, Dong Xie, and
Tianzheng Wang. "Low-Latency Transaction Scheduling via Userspace Interrupts:
Why Wait or Yield When You Can Preempt?" Proceedings of the ACM on Management
of Data 3(3), SIGMOD 2025, Article 182, pages 1-25. DOI:
`https://doi.org/10.1145/3725319`. Retrieved 2026-06-03 from the ACM DOI
metadata and author PDF at `https://www2.cs.sfu.ca/~tzwang/preemptdb.pdf`.

**Category:** runtime / HFT / session scale and transaction scheduling.

**Relevance tags:** preemptive transaction scheduling; userspace interrupts;
mixed OLTP/analytical workloads; priority admission; low tail latency;
transaction context switching; non-preemptible regions; starvation control;
worker ownership; long refresh isolation.

**Core idea:** PreemptDB revisits an old DBMS warning against preemption. The
paper argues that the warning made sense for pessimistic lock-heavy engines,
where interrupting a long transaction could strand locks and force short
transactions to abort. In modern optimistic and multi-version engines,
long-running reads usually do not hold read locks, and recent x86 userspace
interrupts can deliver a preemption signal without a kernel round trip. That
combination makes it practical to pause a low-priority long transaction,
execute urgent short transactions on the same worker thread, then resume the
paused work instead of aborting it.

The implementation, PreemptDB, is built on ERMIA and uses one scheduling
thread plus pinned worker threads. Each worker has separate high- and
low-priority queues and two transaction contexts. A high-priority arrival lets
the scheduler enqueue a batch of urgent transactions and send one userspace
interrupt to the target worker. The worker's interrupt handler saves the
current transaction context, swaps to the second context, runs one or more
urgent transactions, then switches back to the paused long transaction.

The evaluation uses mixed TPC-C/TPC-H-style workloads, with TPC-H Q2 as the
long low-priority transaction and TPC-C New-Order/Payment as short
high-priority transactions. On the paper's single-socket 32-worker evaluation
setting, enabling the user-interrupt machinery reduced pure TPC-C throughput
by about 1.7%. In the mixed workload, PreemptDB reduced high-priority
transaction latency by 88-96% at the measured percentiles versus a
non-preemptive wait policy, while keeping Q2 latency similar. Under overload,
its starvation threshold trades urgent transaction latency against long-work
progress explicitly.

**Concrete mechanisms:**

- A scheduling thread dispatches transactions from admission into per-worker
  high-priority and low-priority queues. The implementation uses a single
  scheduling thread in the evaluation; the paper reports it was not a
  bottleneck for the tested 32-core scope.
- Each worker owns two transaction contexts and normally executes low-priority
  work in the regular context. A userspace interrupt switches it to the
  preemptive context when high-priority work arrives.
- Passive context switch uses the userspace interrupt frame. The handler saves
  register state, extended register state, stack pointer, instruction pointer,
  flags, and transaction-local state into a transaction control block, then
  switches the stack pointer to the other context.
- Active context switch, used when returning to the paused transaction, uses a
  `swap_context` routine. It temporarily disables user interrupts and checks
  whether the interrupted instruction pointer falls inside the active switch
  region so a nested interrupt cannot corrupt partial stack/register state.
- The design adds transparent context-local storage. Each transaction context
  gets a TLS-shaped storage area, and context switches swap which area is
  exposed as TLS. This protects DBMS and library code that assumes one
  thread-local state block per execution stream.
- Non-preemptible regions are explicit and nestable. The worker keeps a
  context-local lock counter; if an interrupt arrives while the counter is
  nonzero, the handler returns without switching. The paper lists index APIs,
  allocator calls, validation, commit, and abort logic as examples.
- Batched on-demand preemption fills a worker's high-priority queue up to a
  bounded size and sends one interrupt for the batch, avoiding one interrupt
  per urgent transaction.
- Starvation prevention tracks the fraction of cycles spent on high-priority
  work since the paused low-priority transaction began. The scheduler stops
  adding urgent work to a worker, or a worker switches back early, when the
  starvation level exceeds a tunable threshold.
- The current design does not recursively preempt an already running
  high-priority transaction, but the paper notes that more contexts could
  support more priority levels.

**GPU DB mapping:** The strongest transferable idea is not that GPU DB should
immediately depend on Intel `uintr`; it is that long work and urgent work need
a real preemption/admission story once a worker can hold CPU for milliseconds.
In the current runtime plan, long COPY admission, resident refresh, CPU
fallback scans, over-resident reads, and analytical retained scans can
monopolize owner or execution workers just as TPC-H Q2 monopolizes PreemptDB
workers. Short retained lookups, commit-critical mutation steps, cancellation,
and response-drain work should have a way to get CPU service without waiting
for arbitrary long loops to finish.

PreemptDB maps naturally to the owner-domain model in
`11-high-throughput-query-runtime.md` as a tiered scheduling policy. The first
implementation probably should be cooperative safe points and queue budgets
inside known long loops, but the benchmark target should measure the same
thing PreemptDB measures: how quickly urgent short work starts when all
workers are already busy with long work. If cooperative safe points are hard
to place or workload-dependent, hardware-assisted or signal-assisted
preemption becomes a later design option.

The context-local storage lesson matters for GPU DB because owners will carry
state that looks thread-local: WAL batch buffers, parser/session scratch,
CUDA pinned buffer handles, stream-local scratch, allocator state, telemetry
accumulators, and error contexts. If a future runtime allows one OS thread to
pause one logical execution context and run another, those resources cannot
silently alias. A simpler near-term rule is that preemptible long work must
hold only explicitly declared context state, and any domain-local buffers must
have ownership metadata before a dispatch switch.

Non-preemptible regions map to database invariants. GPU DB must not preempt
inside WAL-before-visibility publication, resident generation invalidation,
commit timestamp publication, catalog generation swaps, CUDA buffer
handoff/reuse, or critical latch/allocator sections unless the context switch
can prove the same safety PreemptDB proves. This argues for narrow
non-preemptible spans plus telemetry for "preemption requested but deferred,"
not for broad uninterruptible owner loops.

The starvation threshold maps directly to admission control. A GPU DB policy
that always lets short retained reads preempt refresh or long scans may make
refresh never finish. Conversely, letting refresh monopolize workers breaks
interactive latency. The transferable control variable is the fraction of
worker/GPU/queue service time allocated to each request class, exposed as a
tunable or adaptive policy rather than hidden queue behavior.

**Risks and mismatches:** PreemptDB is a CPU in-memory transaction engine, not
a SQL-over-GPU runtime. Its experiments bypass SQL parsing, networking,
planner overhead, storage IO, GPU kernels, and PostgreSQL protocol state. The
implementation requires userspace-interrupt support and a patched kernel in
the evaluated setup; current deployment targets may not have that facility.

The paper assumes optimistic or MVCC reads make preemption practical. GPU DB
must verify that preempted work is not holding locks, pinned buffers, CUDA
stream state, WAL publication rights, catalog latches, or residency ownership
that would block the urgent path. Hardware preemption of GPU kernels is also
not the same as CPU userspace interrupt preemption; the practical mapping may
be CPU-side worker scheduling and GPU queue admission rather than interrupting
an active kernel.

Transparent context-local storage is powerful but complex. Recreating it in a
Rust/CUDA/pgwire runtime could add more risk than benefit unless benchmarks
show cooperative scheduling cannot meet tail-latency targets. The safer first
step is explicit long-operation slicing and class-based admission. The paper
also leaves automatic starvation-threshold tuning as future work, so GPU DB
should treat service-share thresholds as a benchmark variable, not a solved
policy.

**Benchmark candidates:**

- Build a CPU-only mixed runtime benchmark with all workers occupied by long
  refresh or scan loops, then inject urgent short retained lookups and commit
  tasks. Compare FIFO, cooperative safe points, class-priority queue drain,
  and bounded preemption flags. Minimum gate: urgent work start latency and
  p99 response latency improve without violating generation/WAL ordering.
- Add per-request-class service-share telemetry: short retained read, commit
  critical section, COPY chunk, refresh, CPU fallback scan, GPU resident scan,
  response drain, and cancellation. Failure condition: one class can starve
  another without an explicit policy counter showing why.
- Instrument non-preemptible spans in the prototype runtime: WAL append/flush,
  visibility publication, resident invalidation, catalog generation swap,
  pinned-buffer handoff, response-buffer reuse, and allocator/latch regions.
  Proof gate: every deferred urgent request names the span that blocked it and
  the span duration distribution is bounded.
- Prototype long-operation slicing for refresh and CPU fallback scans. Each
  slice must release or checkpoint enough state for urgent retained reads and
  responses to run. Measure throughput loss against urgent p99 latency gain.
- Test a starvation-threshold policy for refresh/scans versus short lookups:
  cap urgent service time at several percentages and measure refresh
  completion time, short lookup p99, queue depth, and explicit overload count.
- Add a buffer-ownership preemption test: pause a long path while it owns a
  decoded COPY buffer, WAL batch buffer, pinned staging buffer, or response
  buffer; ensure the urgent path cannot reuse or observe that buffer until the
  owning context publishes a safe state.
- Keep hardware-assisted preemption as a later experiment: compare ordinary
  cooperative flags against OS signal or userspace-interrupt-like notification
  only after cooperative slicing fails a measured tail-latency gate.

### 2026-06-03 - Resource-adaptive query execution with paged memory management

**Citation:** Riki Otaki, Charles Benello, Jun Hyuk Chang, Goetz Graefe, and
Aaron J. Elmore. "Resource-Adaptive Query Execution with Paged Memory
Management." CIDR 2025. Retrieved 2026-06-03 from
`https://www.vldb.org/cidrdb/papers/2025/p2-otaki.pdf`.

**Category:** multi-tier cache / data placement and query admission.

**Relevance tags:** adaptive memory allocation; buffer pool execution memory;
query context switching; paged intermediate state; file-cache versus operator
memory; SLA-aware admission; LIPAH; buffer-pool contention; memory pressure;
spill control.

**Core idea:** The paper argues that cloud DBMS resource management is hurt by
two opposite defaults: demand-driven allocation can let workloads thrash shared
resources, while static memory limits leave resources idle when demand shifts.
Its proposed direction is to make query execution memory page-based and managed
by the same buffer-pool machinery that manages persistent data pages. If
operators store intermediate state in buffer-pool pages, the system can resize
working memory, suspend and resume queries, and exchange memory between file
caches and operators by pinning or unpinning pages rather than serializing large
heap objects.

The paper also proposes cost-aware allocation. Each memory consumer exposes a
memory-to-cost relationship, where cost can be an SLA penalty derived from
latency, I/O, or another performance target. The allocator can then move memory
from consumers with low marginal value to consumers with high marginal value,
or use a price/broker-style mechanism. This is intentionally exploratory; the
paper identifies communication, pricing, and guarantee protocols as open
research questions rather than solved production policy.

The concrete implementation idea that is easiest to transfer is LIPAH, Logical
ID with Physical Address Hinting. References to pages carry both a logical page
id and a physical frame-id hint. Access first checks the hinted frame and falls
back to the central page-to-frame mapping only if the frame no longer contains
the requested page. This avoids the expensive unswizzling requirements of
traditional pointer swizzling, works for graph-like structures with cycles, and
reduces contention on the shared mapping table.

**Concrete mechanisms:**

- Query plans are broken into pipelines ending at stateful operators. Stateful
  operators such as sort, aggregation, and hash-table build allocate working
  memory as buffer-pool frames.
- A query can pin working-memory pages while using them, unpin pages when it is
  suspended or when memory should be returned, and later request the pages
  again to resume execution.
- The buffer pool can lazily spill unpinned operator pages, avoiding abrupt
  serialization/deserialization spikes that occur when heap-resident operator
  state must be checkpointed or spilled all at once.
- The paper distinguishes memory-aware operators, which explicitly react to
  page availability, from memory-oblivious operators, which are designed to
  tolerate eviction of working-memory pages.
- Cost-aware allocation uses performance-memory curves to derive
  cost-memory curves. The example is sort memory: enough memory may avoid extra
  merge passes, but that same memory can also reduce file-cache capacity and
  increase I/O elsewhere.
- Exchange-based policies reallocate memory when it benefits both consumers or
  lowers global penalty. Pricing-based policies charge consumers for memory
  based on demand and scarcity, with auctions as one possible broker design.
- LIPAH stores an 8-byte fat pointer: a 4-byte logical page id plus a 4-byte
  frame-id hint. An invalid maximum frame id forces slow-path lookup until the
  hint is refreshed.
- After a slow-path lookup finds or loads a page, the access method may
  opportunistically acquire a write latch on the parent page and update the
  physical hint.
- Unlike pointer swizzling, LIPAH does not require all references to a page to
  be found and unswizzled before eviction. The logical id remains authoritative
  even when the physical hint is stale.
- The preliminary evaluation uses a Rust row-store prototype, 256 KB pages, and
  TPC-H SF1. Seven of 22 TPC-H queries were more than 1.5x slower with paged
  execution than non-paged Rust containers, indicating real overhead that still
  needs layout and zero-copy work.
- In a paged hash-index experiment with 10 million key-value pairs, LIPAH
  reduces insertion and lookup latency versus a normal hash index as thread
  count rises, because linked-page traversal avoids repeated central
  page-to-frame mapping latches.

**GPU DB mapping:** This paper strengthens the P8 direction that memory should
be treated as explicit, observable tiers with admission and backpressure,
rather than as invisible heap growth. GPU DB has at least five memory consumers
that can conflict: CPU canonical/MVCC state, CPU derived indexes and stats,
GPU resident snapshots, pinned host staging buffers, and query/operator
intermediate state. A static budget per component will be too rigid once
retained reads, COPY batches, refreshes, CPU fallbacks, and over-resident scans
run concurrently.

Paged operator state maps to a host-side execution-memory tier. Long CPU
fallback scans, grouped aggregate fallbacks, refresh builders, and over-resident
prefetch/decompression stages should allocate bounded page/chunk objects with
known owners instead of unbounded heap structures. Under pressure, the runtime
could release or demote pages from long low-priority work while protecting
short retained reads, commit-critical mutation work, response buffers, and
pinned GPU staging budgets.

The cost-memory model is a useful admission vocabulary for GPU routes. A
retained lookup batch may have high latency value for a small amount of pinned
or resident memory; a long scan may benefit from much more memory but have a
weaker SLA. The scheduler should be able to explain why memory is granted to
one route and denied to another in terms of marginal latency, I/O, transfer
bytes, GPU queue delay, refresh age, and overload policy.

LIPAH suggests a concrete pattern for tiered metadata. Resident or host-cached
segments can use logical segment/page ids as correctness references plus
physical hints to GPU buffers, CPU frames, pinned staging pages, or NVMe page
locations. If a hint is stale, the lookup falls back to the residency manager's
authoritative map. This avoids letting raw physical addresses or CUDA buffer
handles become durable authority, while still reducing central-map contention
on hot paths.

For P8, the warning is equally important: paged execution is not free. If GPU
DB turns every intermediate into slotted pages, short lookups and small
aggregates may pay more pointer chasing, latch traffic, and serialization cost
than they save. The first implementation should use page/chunk ownership for
large or pressure-sensitive work, while preserving compact fast-path buffers
for latency-critical retained requests.

**Risks and mismatches:** This is a design/exploration paper with preliminary
evaluation, not a mature production engine. The TPC-H prototype is row-based,
single-thread query execution in the reported paged-versus-non-paged
comparison, and lacks a cost-based optimizer. It does not evaluate GPU memory,
pinned memory, NVMe tiering, PostgreSQL protocol sessions, WAL/MVCC
visibility, or million-session admission.

The cost-aware allocation discussion leaves major policy pieces open,
including how often consumers communicate their value curves, how prices or
budgets are set, and how guarantees remain stable under sudden demand changes.
GPU DB should treat it as a benchmark framework, not as a ready allocator.
LIPAH also adds pointer width and still requires latch/correctness discipline;
for GPU-resident buffers, frame-id-like hints must include generation checks so
stale hints cannot route a query to an invalid resident snapshot.

**Benchmark candidates:**

- Add an execution-memory budget model for CPU fallback scans, refresh builds,
  over-resident prefetch, response encoding, and pinned staging buffers.
  Minimum gate: every allocation has an owner, class, byte count, lifetime, and
  overload/fallback reason.
- Prototype paged/chunked intermediate state for one large CPU fallback
  aggregate or sort, while keeping the short retained lookup path on compact
  fast buffers. Measure p50/p99 latency, allocations, spill/demotion events,
  and throughput under memory pressure.
- Build a marginal-value admission experiment: give retained lookups,
  refreshes, COPY chunks, and long scans different SLA penalties, then compare
  static quotas, FIFO allocation, and marginal-cost memory transfer. Failure
  condition: the policy cannot explain why a high-priority short route waited
  behind lower-value memory use.
- Implement logical-id plus physical-hint handles for resident segment lookup
  in a CPU-only prototype: logical segment id, generation, and optional cached
  frame/residency slot. Proof gate: stale hints always fall back to the
  authoritative map and never serve invalidated generations.
- Measure central-map contention separately from data movement by comparing
  normal segment-map lookups with hint-validated lookups at high thread counts.
  Required metrics: map latch/cacheline contention, hit rate, stale-hint rate,
  and p99 lookup latency.
- Add a suspend/resume proof for long refresh or CPU fallback work: release
  unpinned intermediate chunks under pressure, let urgent retained reads run,
  then resume without re-executing from the beginning. Failure condition:
  context switching requires serializing the full operator state or breaks
  WAL/resident generation ordering.

### 2026-06-03 - Polaris priority-aware optimistic concurrency control

**Citation:** Chenhao Ye, Wuh-Chwen Hwang, Keren Chen, and Xiangyao Yu.
"Polaris: Enabling Transaction Priority in Optimistic Concurrency Control."
PACMMOD/SIGMOD 2023, Article 44. doi:10.1145/3588724. Retrieved 2026-06-03
from the author PDF,
`https://chenhao-ye.github.io/publication/polaris/polaris.pdf`.

**Category:** transaction processing / write path and concurrency control.

**Relevance tags:** optimistic concurrency control; transaction priority;
tail latency; starvation avoidance; high-contention OLTP; reservation metadata;
abort-aware priority; liveness; owner-queue admission.

**Core idea:** Polaris extends Silo-style OCC with a small amount of
pessimism so higher-priority transactions are protected from repeated aborts
without turning the whole protocol into locking. A high-priority transaction
can reserve records it reads or writes. Lower-priority transactions may still
read reserved records, but they cannot write them; transactions at the same
priority remain optimistic, and an even higher-priority transaction can
preempt an older lower-priority reservation.

The practical motivation is tail latency under contention. Plain OCC detects
conflicts late, so a long or unlucky transaction can repeatedly execute and
abort while short writers keep changing its read set. Polaris uses priority
to make that conflict visible earlier only where it matters. In the paper's
YCSB evaluation, static priority makes high-priority p999 latency 13x lower
than low-priority p999 latency. With an abort-aware priority policy, Polaris
reports 2x lower YCSB-A p999 latency than Silo at Zipfian theta 0.99 with
1.8% throughput loss, and 1.9x higher throughput plus 17x lower p999 latency
than Silo at theta 1.5. In TPC-C with one warehouse, Polaris bounds p999
latency within about 1 ms while Silo reaches 4.9 ms, with Polaris still
outperforming evaluated 2PL variants on throughput.

**Concrete mechanisms:**

- Polaris keeps Silo's per-record transaction id and adds priority,
  priority-version, and reference-count fields that fit in one atomically
  updated 64-bit word in the implementation.
- A reservation is identified by the record's priority and priority version.
  Multiple transactions at the same priority can reserve the same record, but
  cross-priority reservations do not coexist.
- On record access, a transaction reserves the record if its priority is
  higher than zero. If the record is already reserved at the same priority,
  it increments the reference count. If the record is reserved at a lower
  priority, it preempts that reservation by installing its own priority and
  resetting the reference count. If a higher-priority reservation is present,
  reads can continue without reservation, but writes abort.
- Commit still follows Silo's shape: acquire latches for the write set, then
  validate the read set by checking data versions. Polaris adds a priority
  check before write-set latch acquisition; a transaction cannot latch a
  record whose reservation priority is higher than its own.
- Data version remains the serializability guard. Priority fields guide
  conflict handling but do not determine whether the data read was current.
- Reservation cleanup decrements the reference count for read-only records,
  clears priority when the last reservee leaves, and increments the priority
  version. For written records, cleanup removes reservations because the data
  version has changed.
- The lowest-priority fast path avoids reservation overhead when both the
  record and transaction priority are zero, so the common all-low-priority
  case behaves close to Silo.
- The reported field split is 10 bits for reference count, 4 bits for
  priority, 4 bits for priority version, 1 latch bit, and 45 bits for data
  version. The paper treats priority-version wraparound as a possible priority
  inversion, not a serializability failure.
- The paper's default DB-assigned priority policy starts each transaction at
  an initial priority, leaves it there until an abort threshold is reached,
  and then increments priority every fixed number of additional aborts.
  User-specified priority can be layered above DB-assigned priority.
- The formal proof argues that committed transactions serialize in the order
  in which they acquire all write-set latches, that priorities return to zero
  when no transactions are active, and that a transaction will not be aborted
  if it is the only active highest-priority transaction.
- Durability is considered mostly orthogonal; the paper points back to Silo's
  epoch and logging constructs rather than evaluating WAL/checkpoint behavior.

**GPU DB mapping:** Polaris is useful for the GPU DB write path because it
separates "priority" from "thread scheduling." Recent runtime papers suggest
preempting or slicing long work, but Polaris shows that high-priority progress
also needs conflict semantics. A short commit-critical mutation, a repeatedly
aborted user transaction, or a high-priority control transaction should not
only jump an owner queue; it may also need metadata that prevents lower-priority
writers from invalidating its work after it has already paid execution cost.

The reservation idea maps to per-row, per-key, or per-partition conflict
metadata in a future OCC/MVCC owner. GPU DB does not need to adopt Silo's exact
TID layout, but the shape is attractive: keep version/generation as the
correctness authority, and keep priority/reservation fields as advisory
conflict-control metadata that can be rebuilt or ignored during recovery if
needed. For CPU canonical indexes, a compact reservation sidecar keyed by row
id or hot key could protect high-priority update transactions without blocking
read-only retained snapshots.

Abort-aware priority also maps to admission. A request that has retried due to
conflicts should accumulate priority within a bounded class rather than being
treated like a fresh low-value request forever. The owner-domain runtime can
combine this with queue class and service-share telemetry: once a transaction
crosses an abort threshold, it receives higher conflict priority and possibly
higher owner-queue priority, but not unlimited access to GPU, pinned-buffer, or
WAL budgets.

For retained GPU reads, Polaris is mostly a write-path lesson. Read-only
snapshots should not reserve hot records just because they are long; that
would recreate long-reader write damage. But refresh builders, CPU fallback
transactions, and read-write transactions that will publish new visibility may
benefit from lightweight reservations at the point where they can otherwise be
starved by fresh low-priority writes.

The paper also offers a benchmarkable middle ground between deterministic
batching and full preemption. GPU DB can keep optimistic execution inside a
priority class while adding reservation only when the route has crossed a
retry or service-latency threshold. That fits a transaction engine that wants
high throughput in the common uncontended path but predictable p99/p999 for
urgent or repeatedly aborted work.

**Risks and mismatches:** Polaris is built on single-version Silo-style OCC,
not the current GPU DB MVCC tuple store. Its serializability proof depends on
write-set latch acquisition and read-set validation over per-record TIDs; a
multi-version design with retained snapshots, partition owners, and WAL
publication needs a different proof. The paper does not evaluate durable WAL
flush cost, checkpointing, recovery replay, GPU execution, PostgreSQL protocol
state, distributed clocks, or million-session admission.

Reservations can hurt throughput when too many transactions become high
priority, because lower-priority writers abort earlier and the lowest-priority
fast path stops applying. The bit-budget discussion also assumes fewer than
about a thousand concurrent worker transactions for the 10-bit reference
count; GPU DB's 1M logical-session goal must distinguish logical sessions from
bounded active transactions. Finally, a reservation sidecar can become a hot
cache line or map bottleneck unless it is partitioned by owner, key range, or
resident segment.

**Benchmark candidates:**

- Add a CPU-only OCC/MVCC conflict-priority prototype for one hot-key update
  route. Compare FIFO retry, queue priority only, and Polaris-style
  reservation priority. Minimum gate: identical committed histories and lower
  p99/p999 latency for repeatedly aborted transactions under skew.
- Track abort count and retry age as explicit transaction metadata. Promote
  priority after configurable thresholds, then measure throughput, p50, p99,
  p999, abort count distribution, and starvation under YCSB-like hot keys and
  TPC-C-like new-order/payment mixes.
- Prototype a reservation sidecar for CPU canonical row ids or hot index keys:
  version/generation remains the correctness guard, while priority and
  reservation generation decide whether a lower-priority writer may proceed.
  Failure condition: stale reservation metadata can make an invalid version
  visible or survive WAL recovery as durable authority.
- Test priority-class admission across owner queues and conflict metadata
  together. Expected improvement: urgent commit-critical or repeatedly
  aborted transactions start sooner and abort less often than with queue
  priority alone.
- Add a "too many high-priority transactions" stress case. Required telemetry:
  fraction of active transactions above base priority, reservation preempts,
  lower-priority aborts, fast-path misses, and throughput regression.
- Keep long retained read snapshots out of the reservation path. Benchmark a
  long read-only GPU snapshot plus hot writes and verify that read priority
  does not block fresh writes unless the route is explicitly read-write.

### 2026-06-03 - Fifth Modern Batch Synthesis

**Scope:** Low-latency transaction scheduling via userspace interrupts,
resource-adaptive query execution with paged memory management, and Polaris.

**Converging design tracks:** These three papers converge on class-aware work
rather than a single global queue. PreemptDB attacks CPU service latency for
urgent short work, resource-adaptive execution turns memory into a priced and
revocable resource, and Polaris gives conflict metadata a priority dimension
instead of relying on retry luck. For GPU DB, that suggests each active request
needs a route class, resource budget, conflict priority, and preemption or
suspendability contract before it enters an owner domain.

The second convergence is that priority must remain bounded and explainable.
Preemption can starve long refreshes, adaptive memory can overfit to noisy SLA
curves, and Polaris can degrade throughput when many transactions become high
priority. The design response is not "always prioritize short work"; it is
telemetry-backed service shares, abort-aware promotion, and explicit overload
or demotion reasons at each queue, memory, and conflict boundary.

**Category gaps:** Recent coverage is now strong in runtime scheduling,
tiering/admission, and transaction priority. The journal still needs more
direct work on durable logging/checkpointing for high-throughput engines,
explicit MVCC garbage collection under long retained snapshots, and modern
GPU execution papers after the next non-analytics slot is filled.

**Benchmark priorities:**

- Mixed long/short runtime benchmark: long refresh or scan work saturates
  workers while urgent retained reads and commit-critical transactions arrive.
  Measure start latency, p99/p999, service share, and deferred
  non-preemptible spans.
- Value-aware resource admission benchmark: retained lookups, COPY chunks,
  refresh builders, and long scans compete for pinned memory, response buffers,
  and execution-memory chunks under explicit SLA penalties.
- Priority-aware conflict benchmark: hot-key updates with retry-age promotion,
  reservation sidecars, and queue priority compared against plain FIFO retry.
- Cross-boundary telemetry gate: every request should report route class,
  active resource budget, queue wait, conflict priority, preemption/suspend
  status, and the reason for any rejection, fallback, or demotion.

### 2026-06-03 - Rethinking Logging, Checkpoints, and Recovery

**Citation:** Michael Haubenschild, Caetano Sauer, Thomas Neumann, and Viktor
Leis. "Rethinking Logging, Checkpoints, and Recovery for High-Performance
Storage Engines." SIGMOD 2020, pp. 877-892. doi:10.1145/3318464.3389716.
Retrieved 2026-06-03 from the author PDF,
`https://db.in.tum.de/~leis/papers/rethinkingLogging.pdf`.

**Category:** durable logging, checkpointing, and recovery for high-throughput
storage engines.

**Relevance tags:** per-thread WAL; remote flush avoidance; continuous
checkpointing; bounded recovery; page provisioning; persistent memory log
tail; SSD/NVMe storage; out-of-memory OLTP; recovery-time budgeting;
WAL-before-visibility.

**Core idea:** The paper targets the gap between ARIES-style disk recovery and
lightweight in-memory logging. ARIES has the right feature set for
larger-than-memory storage, fuzzy checkpoints, and index recovery, but a
centralized log and traditional checkpoint bursts are too expensive for modern
multi-core engines. Pure in-memory designs scale better but assume the data
set fits in memory and often give up incremental checkpoints or transparent
index recovery.

The proposed LeanStore design keeps page-oriented recovery features while
using distributed per-worker logs, a persistent-memory first-stage log tail,
remote-flush avoidance, continuous checkpointing, and a dedicated page
provider. In the reported TPC-C experiments with 40 workers, the fully enabled
system reaches about 850k transactions/sec with roughly 19k recovery-component
instructions per transaction. With a 100 GB WAL recovery limit, recovery takes
38 seconds on 40 threads, corresponding to about 2.6 GB/sec of recovered WAL.

**Concrete mechanisms:**

- Each worker thread owns a log partition. A transaction is pinned to one
  worker, so its log records go to one partition, while records for the same
  page may still appear in different partitions.
- The log has stages: a small circular first stage on persistent memory or
  battery-backed DRAM, background staging to SSD, and archive storage for
  media recovery. With persistent memory, commit needs cache-line persistence
  of the local log tail rather than waiting for SSD staging.
- Log records include type, page id, transaction id, GSN, and before/after
  image data. Recovery gathers records for a page from all partitions and
  applies them in GSN order.
- Remote Flush Avoidance tracks, for each page, the log partition of the most
  recent modification. A transaction records the maximum globally flushed GSN
  at start and maintains `needsRemoteFlush`. If a page's prior GSN is already
  globally flushed, or the latest unflushed modification is in the same log,
  the transaction can avoid flushing all remote logs at commit.
- RFA and group commit are separable. With persistent memory, the paper argues
  for RFA without group commit for low-latency commits; without persistent
  memory, RFA reduces the set of transactions that need global group-commit
  waiting.
- Continuous checkpointing couples checkpoint increments to generated WAL
  volume rather than wall-clock time. The buffer pool is split into shards; for
  every `1/S` of the configured WAL limit staged, the checkpointer writes dirty
  pages in the next shard and records the minimum current GSN for that shard.
  The minimum shard GSN, constrained by oldest active transaction GSN, defines
  how far the log can be pruned.
- Page provisioning treats hot, cool, and free pages as a closed system. A
  page-provider thread unswizzles hot pages into a cool FIFO, evicts clean
  pages into a free list, and writes dirty pages at the latest useful moment
  before eviction. Worker threads allocate from the free list without touching
  global eviction structures on the hot path.
- The design uses steal. Before-images are stored in WAL, and transaction
  aborts execute logical inverse operations through normal access paths, then
  write an end-of-transaction record.
- Recovery has analysis, redo, and undo phases. Analysis scans all log chunks,
  separates winner and loser transactions, partitions winner log records by
  page id into thread-local redo tables, and collects undo work. Redo assigns
  page-id ranges to workers, merges and sorts records by `(pageId, GSN)`, and
  replays page by page. Undo logically reverts loser transactions.
- Persistent-memory implementation details include DAX-mapped log chunks,
  non-temporal stores via PMDK, and per-record checksum validation to find the
  last complete log record without repeatedly flushing an end-offset location.

**GPU DB mapping:** This paper is directly relevant to GPU DB's durable
authority boundary. P8 already treats GPU residency as rebuildable cache and
WAL/checkpoint/archive replay as the source of truth. The LeanStore recovery
design gives that boundary a concrete high-throughput shape: per-owner or
per-partition WAL streams, local commit persistence, page or segment generation
ordering, and a recovery path that can reconstruct CPU truth before any GPU
resident snapshot is trusted.

Remote Flush Avoidance maps cleanly to future partition owners. GPU DB should
not require every committing mutation to synchronize every WAL stream just
because streams share a global timestamp. A page-, segment-, or partition-local
last-writer log id plus a global flushed boundary could let independent
mutations commit locally while still detecting the cases where a shared
physical segment requires remote durability before visibility publication.
The correctness invariant remains WAL-before-visibility; RFA is only a way to
prove a remote flush is unnecessary.

Continuous checkpointing is a useful model for resident refresh and cache
rebuild debt. Instead of time-triggered "big refresh" or "big checkpoint"
events, GPU DB can tie background checkpoint, archive, CPU index rebuild,
resident segment refresh, and cold-tier writeback to measured WAL bytes,
dirty-segment bytes, or invalidation debt. Each shard or segment should carry
the source WAL boundary it has persisted or refreshed through, so admission can
explain whether a route is valid, stale, rebuilding, or blocked by a long
active transaction.

The page-provider idea also transfers to multi-tier placement. Workers should
not consult expensive eviction or tier-placement structures in the request hot
path. A residency/page provider can keep a small free list of host pages,
pinned buffers, GPU staging buffers, and resident slots, while doing
unswizzle/demotion/writeback/eviction work outside latency-critical retained
reads and commits.

Finally, the recovery benchmark should shape GPU DB's durability gates. It is
not enough to measure COPY throughput or retained query latency while running;
each write-path improvement should also report bounded recovery work: max WAL
bytes to replay, partitioned replay throughput, index rebuild time, resident
cache invalidated-on-start behavior, and time until GPU routes can safely be
admitted after CPU truth is restored.

**Risks and mismatches:** The paper is CPU storage-engine work, not a GPU
database design. It assumes page-based buffer management, pointer swizzling,
and LeanStore's optimistic synchronization, while GPU DB's current first P8
slice uses MVCC tuple chains and generated GPU column-group snapshots. RFA
depends on page-local GSN ordering and last-modifier metadata; translating it
to MVCC row versions, column segments, and resident snapshot generations needs
a fresh correctness proof.

The evaluation uses persistent memory for the first-stage log tail and fast
SSDs for staging. Future GPU DB deployments may lack persistent memory or may
use CXL, NVRAM, NVMe, or plain DRAM-plus-fsync differently. The paper also
does not evaluate PostgreSQL protocol sessions, GPU execution queues,
snapshot-retirement pressure, or million-session admission. Its steal/undo
choice is not automatically right for every GPU DB mutation path.

**Benchmark candidates:**

- Prototype per-owner WAL partitions for COPY/INSERT in a CPU-only path.
  Compare centralized WAL, partition-local WAL with conservative global flush,
  and RFA-style dependency checks. Required metrics: rows/sec, commit p50/p99,
  remote flush rate, WAL bytes, and replay correctness after crash simulation.
- Add durable-boundary telemetry to resident segment metadata: source WAL id,
  checkpointed/refreshed-through boundary, oldest active reader/transaction,
  invalidation generation, and recovery rebuild requirement. Failure
  condition: a GPU route can be admitted without naming the durable CPU
  boundary it depends on.
- Build a continuous-checkpointing simulator over table/partition shards.
  Trigger increments by WAL bytes or dirty-segment bytes, then measure max
  replay bytes, write amplification, foreground latency disturbance, and
  checkpoint lag under steady COPY plus retained reads.
- Add a page-provider-style resource proof for host pages, pinned staging
  buffers, and GPU resident slots. Workers allocate from bounded free lists;
  a provider performs demotion, writeback, eviction, and refresh cleanup.
  Required telemetry: free-list depth, provider lag, synchronous allocation
  misses, and route fallback caused by provider debt.
- Add recovery gates to write-throughput benchmarks: after a forced restart,
  measure WAL analysis/replay time, CPU index/statistics rebuild time, GPU
  residency invalidated/rebuilt state, and time to first admitted retained GPU
  route.
- Stress long active transactions or retained snapshots against continuous
  checkpoint and log pruning. Verify the oldest active boundary can delay
  pruning without silently serving stale resident data or letting WAL/archive
  usage grow without an explicit overload reason.

### 2026-06-03 - Scalable garbage collection for in-memory MVCC

**Citation:** Jan Boettcher, Viktor Leis, Thomas Neumann, and Alfons
Kemper. "Scalable Garbage Collection for In-Memory MVCC Systems."
Proceedings of the VLDB Endowment 13(2):128-141, 2019.
doi:10.14778/3364324.3364328. Retrieved 2026-06-03 from the VLDB
PDF, `https://www.vldb.org/pvldb/vol13/p128-bottcher.pdf`.

**Category:** MVCC / snapshot / visibility and transaction-processing
write path.

**Relevance tags:** MVCC garbage collection; long retained snapshots;
HTAP version chains; eager pruning; thread-local transaction state;
foreground GC; active timestamp sets; skewed hot tuples; version-chain
telemetry; snapshot-retirement pressure.

**Core idea:** The paper argues that MVCC garbage collection is not a
background housekeeping detail; in mixed HTAP workloads it can become
the bottleneck that slows both readers and writers. Long-running
queries keep old snapshots alive, while short update transactions keep
adding versions. A coarse high-watermark collector then cannot remove
many intermediate versions, so chains grow, reads take longer, GC work
gets more expensive, and writes eventually lose time to chain latching
and foreground cleanup.

The proposed Steam collector makes GC part of normal transaction
processing. Instead of depending on a detached background thread, each
worker tracks local transaction state, exposes a thread-local minimum
start timestamp, and prunes obsolete versions when it already touches a
version chain. Its Eager Pruning of Obsolete Versions (EPO) uses the
set of active transaction start timestamps to remove intermediate
versions that no active transaction can ever observe, rather than only
removing versions older than the global oldest active transaction.

**Concrete mechanisms:**

- Each thread owns a disjoint subset of transactions and updates only
  its own thread-local minimum start timestamp. Other threads compute a
  global minimum by scanning these atomic per-thread minima, avoiding a
  shared transaction-list mutex.
- Steam keeps active and committed transaction metadata in thread-local
  structures. If a thread goes idle, the scheduler can trigger GC so
  cleanup is not indefinitely delayed by lack of foreground work.
- EPO periodically builds a sorted list of active transaction start
  timestamps. For each touched version chain, it walks active timestamps
  and visible versions in order, removing obsolete in-between versions
  that no active transaction needs.
- Version records store only changed attributes for updates. When EPO
  removes an intermediate partial-version record, it may merge missing
  attribute before-images into the surviving visible version so rollback
  and snapshot reads remain correct.
- Steam prunes when a chain is extended by a new version, so a hot chain
  is shortened before it becomes expensive. The paper contrasts this
  with interval GC, where a background worker periodically revisits
  chains after they may already have grown large.
- Active timestamp lists are reused for short transactions and refreshed
  on a small time scale, reducing overhead in pure OLTP cases while
  retaining precision for long-running readers.
- The evaluation compares GC strategies inside HyPer while holding the
  storage and query engine constant. In CH-style mixed workloads, EPO
  keeps maximum chain length at two in one reported setup versus over
  30k with a standard watermark approach; traversed versions during GC
  fall from about 1.2 billion to 4.2 million, and transaction throughput
  rises from about 6.6k/s to 30.6k/s for that experiment.
- In pure TPC-C and skewed key-value updates, the paper reports that
  thread-local foreground approaches scale better than background or
  globally synchronized schemes, and that GC trigger frequency can
  dominate performance when it is fixed rather than tied to generated
  garbage.

**GPU DB mapping:** GPU DB's retained read snapshots can reproduce the
same failure mode as HTAP long queries. A retained GPU snapshot or CPU
read boundary may keep old MVCC versions alive while COPY/INSERT/UPDATE
continues producing new versions. If GC uses only one global oldest
snapshot boundary, hot tuple chains and invalidated resident segments
can accumulate behind one long reader, increasing CPU fallback latency,
refresh-build cost, recovery rebuild work, and memory pressure.

The strongest transferable idea is to make version retirement
owner-local and chain-local rather than a detached global sweep.
Mutation owners or partition owners should track active read boundaries
for their domain, publish local minimums, and do bounded pruning while
they already hold or own the affected tuple/segment state. A background
maintenance owner can still handle idle domains, but it should not be
the only path that prevents hot chains from growing.

EPO also maps to resident snapshot generations. For every table,
partition, or resident segment, GPU DB should distinguish versions
needed by active readers from intermediate versions that no active
snapshot can see. When a mutation extends a hot chain or invalidates a
resident generation, the same owner can prune unreachable intermediate
versions, merge partial before-image state where needed, and update
telemetry for retained bytes, max chain length, oldest active boundary,
and blocked-pruning reason.

The active timestamp set suggests a practical batching lever for GPU
reads. If many read-only retained requests can share a read boundary,
the number of distinct active timestamps falls, EPO has fewer visible
versions to preserve, and compatible GPU micro-batches become easier to
form. This does not justify delaying all reads, but it gives a benchmark
hypothesis for grouping same-shape retained reads by snapshot generation
under a microsecond ceiling.

**Risks and mismatches:** Steam is an in-memory CPU MVCC design built
inside HyPer, not a GPU-resident storage engine. Its version-chain
layout, partial before-image merging, and timestamp semantics may not
match GPU DB's tuple chains, WAL replay model, resident column groups,
or future partitioned owners. The paper does not evaluate CUDA
execution, PostgreSQL protocol sessions, durable WAL/checkpoint
interaction, crash recovery, GPUDirect storage, or million logical
sessions.

Foreground GC can steal time from transactions if pruning work is not
bounded per operation. GPU DB needs per-owner cleanup budgets and
explicit overload behavior so a hot write path does not pause behind an
unbounded chain walk. EPO correctness also depends on accurately knowing
all active read boundaries; a missed retained snapshot would let the
system reclaim a version that a GPU or CPU reader still needs.

**Benchmark candidates:**

- Add MVCC GC telemetry before changing behavior: active read boundary
  count, oldest active boundary, max/avg chain length per table or
  partition, obsolete-but-retained version bytes, pruning time, and the
  reader or route class blocking pruning.
- Prototype owner-local foreground pruning for a hot update route.
  Compare global high-watermark-only GC, background sweep, and
  Steam-style eager chain pruning under one long retained snapshot plus
  skewed updates. Minimum gate: identical visible results and lower
  max chain length without increasing write p99 beyond a fixed budget.
- Test retained GPU snapshot pressure: run long read-only retained
  snapshots while COPY/UPDATE mutates hot keys, then measure CPU
  fallback latency, resident refresh time, stale-generation count,
  obsolete version bytes, and route rejection/fallback reasons.
- Add a bounded pruning budget per mutation owner. Failure condition:
  any single write performs an unbounded chain walk or silently skips
  pruning without increasing a visible cleanup-debt counter.
- Evaluate snapshot-boundary grouping for same-shape retained reads.
  Expected improvement: fewer active read boundaries, shorter preserved
  version sets, larger compatible GPU micro-batches, and no p50 latency
  regression beyond the configured microsecond wait ceiling.
- Add a crash/recovery proof for GC metadata. After pruning, WAL replay
  and checkpoint recovery must reconstruct exactly the versions needed
  by committed visibility boundaries and must not depend on GPU resident
  cache state as durable authority.

### 2026-06-03 - PAR2QO parametric penalty-aware robust query optimization

**Citation:** Haibo Xiu, Yang Li, Qianyu Yang, Pankaj K. Agarwal,
and Jun Yang. "PAR2QO: Parametric Penalty-Aware Robust Query
Optimization." Proceedings of the VLDB Endowment 18(11):4532-4545,
2025. doi:10.14778/3749646.3749711. Retrieved 2026-06-03 from the
VLDB PDF, `https://www.vldb.org/pvldb/vol18/p4532-xiu.pdf`.

**Category:** query optimization / planning.

**Relevance tags:** robust query optimization; parametric query
optimization; plan cache; route templates; selectivity uncertainty;
expected penalty; optimizer overhead; workload generation; GPU/CPU
route choice; admission-time planning.

**Core idea:** PAR2QO combines parametric query optimization with
penalty-aware robust query optimization. Instead of caching one
point-optimal plan for a parameterized SQL template, it builds a
per-template cache of plan-penalty profiles over sampled selectivity
locations. At runtime, it estimates which cached plan has the lowest
expected penalty under the query's selectivity uncertainty, rather
than blindly choosing the cheapest plan at the optimizer's current
estimates.

The paper's main practical point is that robust planning work can be
amortized across many future instances of the same query template.
PAR2QO samples probe locations around workload queries using an
error model, records how candidate plans behave across those
locations, reduces the candidate set, and then performs a relatively
cheap runtime pass over cached profiles. On JOB, the paper reports
up to 1.96x speedup over PostgreSQL and 1.83x over Kepler, while
avoiding some severe regressions that Kepler experiences. It also
reports average preparation time of about 21.5 minutes per JOB
template, versus more than 6.5 hours for Kepler, and runtime
optimization overhead around 26 ms per query versus PostgreSQL's
67 ms in their setup.

**Concrete mechanisms:**

- The system assumes non-intrusive access to an existing DBMS
  optimizer: estimated selectivities for a query, an optimizer call
  under injected selectivities, and a cost call for a given plan under
  injected selectivities.
- Robustness is represented through a user-defined penalty function.
  The default penalty is zero when a plan is within a tolerance of the
  optimal cost at the true selectivities, and otherwise proportional
  to the excess cost over that optimal plan.
- Error profiling learns a conditional distribution of true
  selectivities given estimated selectivities. Following PARQO, the
  paper models errors with querylet profiles for small selection-join
  subqueries rather than attempting to represent a full high-dimensional
  joint distribution directly.
- Offline preparation groups workload queries into selectivity-error
  clusters. If a new training query is close enough to an existing
  cluster by KL divergence, the cluster hit count is incremented
  instead of over-sampling that region.
- For a new cluster, PAR2QO samples `n` probe locations from the error
  distribution around that query's estimated selectivities. The final
  probe set is the union of all cluster samples, and cluster hit counts
  approximate workload frequency for later bias correction.
- For each probe location, PAR2QO invokes the optimizer to obtain a
  plan. It then costs each distinct candidate plan at all probe
  locations and stores a plan-cost matrix.
- The plan-penalty profile matrix records each candidate plan's penalty
  at each probe location, relative to the best candidate cost observed
  at that location.
- Candidate plans can be reduced by a tau-approximate cover heuristic:
  greedily keep plans that are near-optimal at many probe locations.
  The paper also evaluates a conservative reduction based on
  Jensen-Shannon distance between cost profiles.
- Runtime plan selection derives the uncertainty distribution for the
  incoming query, reweights cached probe-location penalties using
  importance sampling, and picks the plan with the lowest estimated
  expected penalty. It needs one selectivity-estimation call but no
  optimizer or plan-cost calls in the hot path.
- CARVER generates training or testing workloads by covering subquery
  cardinality ranges, not only final join result rows. It supports
  equality, inequality, range-style, and token-based LIKE parameter
  generation for selected template shapes.
- The evaluation finds that PAR2QO is strongest when workloads are
  harder or shift across distributions. Kepler can find faster plans
  for some templates because it trains from actual executions, but the
  paper reports larger regressions for Kepler when the selected plan is
  not robust.

**GPU DB mapping:** The direct mapping is not "learn a whole GPU
optimizer." It is a route-template cache for repeated SQL shapes where
the engine already knows a small set of legal routes: CPU tuple/index
path, CPU segment path, GPU resident scan, GPU resident key-vector
lookup, GPU cold-transfer path, and explicit fallback or rejection.
For each template, GPU DB can cache a profile of route penalties over
estimated rows, transfer bytes, resident validity probability, queue
delay, refresh debt, output bytes, and memory pressure.

The penalty-aware objective is more appropriate than a single fastest
route for retained GPU execution. A GPU route with excellent average
latency can be fragile when cardinality, D2H result size, refresh
state, or queue depth is wrong. PAR2QO suggests making that fragility
visible: route selection should minimize expected regret or overload
penalty under uncertainty, not only predicted latency at one estimate.
For example, a resident GPU aggregate might be selected aggressively,
while a resident GPU lookup returning many rows may need a safer CPU
or hybrid route if the result-cardinality uncertainty is high.

Plan-penalty profiles also fit the high-throughput runtime's
admission-time constraints. IO workers and read snapshot workers
should not call an expensive optimizer or rebuild a route search for
every repeated parameterized query. A compact per-template profile can
turn route choice into one bounded pass over cached candidates, with
the selected route explaining which uncertainty dimensions made the
GPU path safe, risky, or rejected.

CARVER maps to benchmark generation for P8. Instead of testing only
popular keys or uniformly sampled parameters, GPU DB should generate
parameterized lookup, prefix LIKE, range, aggregate, and future join
queries that cover subquery cardinality buckets and resident/nonresident
states. That would stress the exact cases where the planner must decide
between GPU resident execution, CPU fallback, refresh, or overload.

**Risks and mismatches:** PAR2QO is evaluated on CPU PostgreSQL plan
selection, not GPU execution, MVCC snapshots, queueing, or tiered
residency. Its cost estimates assume that recosting plans under
injected selectivities is meaningful; GPU DB will need route costs that
include live telemetry such as queue delay, pinned-buffer pressure,
resident generation age, transfer size, and refresh debt. The paper's
runtime overhead of about 26 ms is acceptable for complex analytical
queries but far too high for short retained point lookups unless the
profile scan is made much smaller or cached per parameter bucket.

The method also trusts the cost model once selectivities are corrected,
while GPU routes may be sensitive to CUDA launch overhead, PCIe/NVLink
contention, response encoding, and concurrent kernels. Training from
real executions, as in Kepler, can catch some of those effects, but at
a much higher preparation cost. Finally, workload-derived error models
can go stale when data distribution, resident placement, or session
load changes; GPU DB would need recalibration triggers and fallback
guardrails.

**Benchmark candidates:**

- Build a small route-profile simulator for repeated parameterized
  retained queries. Candidate routes: CPU index lookup, CPU scan, GPU
  resident scan, GPU resident key-vector lookup, and GPU cold transfer.
  Choose routes by predicted latency versus penalty-aware expected
  regret under cardinality and queue-delay uncertainty.
- Add planner telemetry that records why a GPU route was rejected:
  estimated rows, result bytes, resident validity, refresh debt, queue
  depth, pinned-buffer pressure, unsupported predicate, or high penalty
  uncertainty.
- Generate CARVER-style benchmark parameters for the first P8 query
  shapes: `int4 = ?`, `text LIKE 'prefix%'`, bounded aggregates, and
  mixed predicates. Minimum gate: generated cases cover low, medium,
  high, and extreme cardinality buckets rather than only popular keys.
- Compare cheapest-estimate route choice against penalty-aware route
  choice under synthetic selectivity errors and live queue saturation.
  Failure condition: the robust chooser reduces worst-case latency only
  by making ordinary low-risk point lookups slower beyond the configured
  p50 budget.
- Cache route profiles per SQL template and snapshot route family.
  Required measurements: profile size, route-selection time, cache hit
  rate, recalibration frequency, and correctness when schema,
  residency, or visibility generations change.
- Add a drift experiment where resident placement, table distribution,
  and queue load change after profile preparation. Proof gate: stale
  profiles trigger fallback, recalibration, or explicit overload rather
  than silently selecting a fragile GPU route.

### 2026-06-03 - Sixth Modern Batch Synthesis

The latest batch spans durable write-path authority, MVCC cleanup, and
robust route planning: LeanStore logging/recovery, Steam MVCC garbage
collection, and PAR2QO. The converging design track is that hot-path
decisions should be local, bounded, and explainable, while the metadata
that justifies them should be explicit enough for recovery, pruning, and
route fallback.

For write throughput, decentralized WAL streams and remote-flush
avoidance point toward partition or mutation owners that can commit
locally when no remote durable dependency exists. For snapshot health,
Steam reinforces that retained readers must be visible to owner-local
GC, and cleanup debt must be budgeted rather than deferred to an
unbounded background sweep. For read latency, PAR2QO adds a planning
track: repeated retained SQL shapes should choose CPU/GPU/tier routes
by expected penalty under uncertainty, not just by a single optimistic
cost estimate.

The category gap after this batch is still high-concurrency runtime and
session admission. Recent work has covered MVCC, logging, optimizer
robustness, tiering, and transaction scheduling well, but the next few
papers should include at least one IO-worker, request-scheduling, or
network/session-scale system before returning to GPU OLAP.

Benchmark priorities:

- Tie every resident route to durable-boundary metadata and measure
  recovery-to-first-GPU-route time after write-heavy runs.
- Measure owner-local GC under one long retained snapshot and skewed
  updates, including cleanup debt and write p99.
- Prototype penalty-aware route choice for repeated retained queries
  using cardinality, queue delay, result size, and resident validity
  uncertainty.
- Add CARVER-style parameter generation so lookup, prefix, aggregate,
  and mixed predicate benchmarks cover the cardinality cases that make
  CPU/GPU route choice fragile.

### 2026-06-03 - Shenango high-efficiency latency-sensitive runtime

**Citation:** Amy Ousterhout, Joshua Fried, Jonathan Behrens, Adam
Belay, and Hari Balakrishnan. "Shenango: Achieving High CPU Efficiency
for Latency-sensitive Datacenter Workloads." 16th USENIX Symposium on
Networked Systems Design and Implementation (NSDI 2019), pp. 361-378,
2019. Retrieved 2026-06-03 from the USENIX PDF,
`https://www.usenix.org/system/files/nsdi19-ousterhout.pdf`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** kernel-bypass networking; microsecond scheduling;
IO workers; bounded rings; packet queues; runnable-thread queues;
core allocation; burst admission; work stealing; user-level threads;
tail latency; CPU efficiency; session multiplexing; response rings.

**Core idea:** Shenango targets the tension between microsecond tail
latency and CPU efficiency. Kernel-bypass systems can keep latency low
by dedicating cores to polling, but this provisions for peak load and
wastes cycles when traffic is below peak. Shenango instead reallocates
cores across applications at very fine granularity, using a dedicated
IOKernel core plus per-application user-level runtimes. The IOKernel
steers packets, observes queue buildup, and grants or revokes cores
quickly enough to absorb bursts without keeping every latency-sensitive
application fully provisioned.

The most relevant idea for GPU DB is not adopting Shenango wholesale.
It is the control signal: queueing duration at the boundary between IO,
runnable work, and execution resources is a better admission signal than
coarse utilization. For a database trying to support many logical
sessions, the hot path should expose whether requests, response writes,
GPU work, or mutation work have remained queued across microsecond
sampling intervals, then allocate scarce execution resources or reject
work at that boundary.

**Concrete mechanisms:**

- A privileged IOKernel runs on a dedicated busy-spinning core. It
  polls NIC receive and transmit queues, forwards packets through
  shared-memory queues, and orchestrates core allocation for
  per-application runtimes.
- Each runtime has guaranteed cores and burstable cores. It may use
  fewer than its guarantee when idle, may temporarily exceed its
  guarantee when spare cores exist, and can have burstable cores
  preempted by the IOKernel.
- Congestion detection runs every 5 microseconds. For each active
  runtime kthread, the IOKernel compares runqueue and ingress-packet
  queue state with the previous interval. If work is present in the
  same queue across two checks, Shenango treats that as at least one
  interval of queueing delay and grants another core when possible.
- The paper emphasizes queueing duration rather than queue length
  because length thresholds are workload-dependent, while "still queued
  after the next sampling interval" directly indicates delayed work.
- Queue metadata is exposed in one shared cache line per kthread. The
  queues are ring buffers, so the IOKernel can detect persistent work by
  comparing head/tail state across intervals.
- Core selection favors locality: use a hyper-thread sibling of an
  already active core for that application when possible, then a core
  the application recently used, then any idle core, and only then a
  burstable core reclaimed from another application.
- Runtimes use lightweight user-level threads, per-kthread runqueues,
  work stealing, cooperative run-to-completion in the common case, and
  parking when no work is found after brief steal attempts.
- Packet handling can be stolen across runtime cores, including TCP
  protocol handling. This relaxes strict flow-consistent hashing and may
  cause short-timescale packet reordering, which Shenango handles in the
  transport layer when ordering is required.
- Shenango uses shared-memory descriptor rings for ingress packets,
  egress packets, and separate egress command queues to avoid
  head-of-line blocking.
- The implementation relies on DPDK for NIC access, `sched_setaffinity`
  for binding kthreads, `eventfd` for parking/unparking, and targeted
  signals for preempting runtime kthreads.
- In the memcached evaluation, Shenango reports over 5 million requests
  per second with 37 microsecond median and 93 microsecond 99.9th
  percentile response time, while preserving spare cycles for batch
  work. In a sudden-load experiment from 100k to 5 million requests per
  second, it reports almost no additional tail latency, whereas Arachne
  takes more than 500 ms to adapt.
- The paper reports an IOKernel packet-rate ceiling of about 6.5 million
  incoming plus outgoing packets per second in its setup, and notes that
  packet forwarding, not core allocation, is most of the IOKernel cost.

**GPU DB mapping:** GPU DB's production runtime already points toward
network IO workers, bounded command rings, read snapshot workers, GPU
execution workers, mutation owners, and response rings. Shenango gives a
specific admission metric for that topology: measure whether work
remains queued across a microsecond-scale sampling interval, not only
average queue depth or CPU utilization. A request waiting in a network
ingress ring, read snapshot ring, GPU execution ring, mutation ring, or
response ring for two consecutive samples should increment a congestion
signal tagged with the exact boundary.

For 1M logical sessions, the direct transfer is a multiplexed IO-worker
model with explicit per-boundary queue telemetry. GPU DB should avoid
thread-per-session processing and should not dedicate one busy polling
resource per connection. Instead, IO workers can own socket readiness
and protocol parsing, while execution resources are granted to classes
of work: short retained reads, mutation admission, resident refresh, GPU
kernel batches, and response encoding. Shenango's guaranteed/burstable
split maps to reserving minimum capacity for correctness-critical
mutation/WAL work and allowing read-heavy retained routes to burst only
while the relevant rings are healthy.

The queue-duration signal also fits GPU micro-batching. Same-shape
retained reads should be collected under a latency ceiling, but if a GPU
execution ring shows persistent queued work across two sampling
intervals, the scheduler should either drain a larger compatible batch,
allocate another stream/worker if available, fall back to CPU when safe,
or return a precise overload reason. The important part is that the
decision is tied to observed queue wait at the saturated boundary.

Shenango's locality rules translate to owner and worker placement. GPU
execution workers should prefer stable CUDA streams, pinned buffers, and
partition-local metadata rather than moving work randomly. CPU IO and
response workers should prefer recent cores and per-worker buffer caches
to preserve cache locality for row descriptions, protocol state, and
response-ring writes.

**Risks and mismatches:** Shenango is a network/runtime system, not a
database engine. It does not address WAL-before-visibility, MVCC
snapshot correctness, SQL protocol semantics, durable recovery, GPU
resident invalidation, CUDA scheduling, or query planning. Its
centralized IOKernel is also a potential bottleneck; the paper observes
packet forwarding as the dominant IOKernel cost and does not evaluate
multi-socket NUMA scaling.

GPU DB should not infer that a single central scheduler can own all
database resources. Mutation ordering, read snapshot publication,
residency refresh, and GPU execution each have correctness boundaries
that may require separate owners. The useful lesson is the queue-delay
feedback loop, not a mandate to collapse the system into one privileged
runtime core.

There is also a protocol mismatch. Shenango's packet stealing and
resequencing can tolerate transport-level reordering, but PostgreSQL
wire sessions require ordered frontend/backend semantics per session.
GPU DB may steal parsed requests between workers only after preserving
per-session ordering, transaction state, cancellation behavior, and
response sequencing.

**Benchmark candidates:**

- Replace thread-per-client measurement for one bounded path with a
  small IO-worker pool and response rings. Minimum gate: identical SQL
  results and lower per-connection memory/thread cost at concurrency
  `1,2,4,8,16,32,64`, with no hidden owner-thread ordering changes.
- Add per-ring queue-duration telemetry: ingress, mutation, read
  snapshot, residency, GPU execution, and response rings. Record whether
  work survived two consecutive microsecond-scale samples, plus oldest
  queued age and boundary-specific overload reason.
- Compare admission policies for retained reads: queue-length threshold,
  utilization threshold, and Shenango-style persistent-queue-duration
  threshold. Expected result: queue-duration admission should reduce
  p99/p99.9 spikes during bursts without over-provisioning workers.
- Prototype guaranteed and burstable execution budgets. Reserve
  mutation/WAL capacity, then let retained reads, refresh work, and
  response encoding borrow burst capacity only while their queues are
  below overload thresholds.
- Add a burst workload: hold a low baseline of persistent pgwire
  sessions, then jump same-shape retained reads from low rate to the
  highest bounded offered rate. Measure p50/p99/p99.9 latency, queue
  wait by boundary, response-ring lag, GPU batch size, CPU fallback, and
  overload counts.
- Test worker locality: stable assignment of IO/response workers and GPU
  execution workers versus random work stealing. Required measurements:
  cache-miss proxy if available, row-description reuse, pinned-buffer
  reuse, CUDA stream reuse, and tail latency under imbalanced sessions.
- Failure condition: any admission policy improves throughput by
  allowing stale resident snapshots, violating per-session response
  order, starving WAL/mutation progress, or hiding overload without a
  precise rejection/fallback reason.

### 2026-06-03 - SMF schedule-first transaction ordering

**Citation:** Audrey Cheng, Aaron Kabcenell, Jason Chan, Xiao Shi,
Peter Bailis, Natacha Crooks, and Ion Stoica. "Towards Optimal
Transaction Scheduling." Proceedings of the VLDB Endowment 17(11),
pp. 2694-2707, 2024. Retrieved 2026-06-03 from the PVLDB PDF,
`https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf`.

**Category:** transaction processing / write path and concurrency
control.

**Relevance tags:** transaction scheduling; hot-key contention;
MVTSO; operation ordering; owner queues; timestamp assignment;
application hints; conflict-cost reduction; abort reduction;
batch admission; skewed OLTP; low-tail transactions.

**Core idea:** The paper argues that many transaction systems leave
throughput on the table because they execute close to arrival order
and resolve conflicts only after operations arrive or abort. It frames
transaction scheduling as a makespan-minimization problem: for a
finite batch, lower makespan means higher throughput, and different
serializable orders can have very different conflict cost.

The proposed policy, Shortest Makespan First (SMF), greedily builds a
schedule by appending the sampled in-flight transaction that adds the
least incremental makespan. The practical insight is that a small set
of hot keys usually dominates schedule quality, so the scheduler can
use transaction type and hot-key hints rather than full read/write-set
knowledge. R-SMF combines this search policy with MVSchedO, a
schedule-first variant of multi-version timestamp ordering that
assigns timestamps from the selected schedule and delays conflicting
hot-key operations so later scheduled operations do not race ahead.

**Concrete mechanisms:**

- SMF starts with a transaction and repeatedly samples a small number
  of unscheduled in-flight transactions. It appends the candidate that
  produces the smallest estimated makespan increase.
- The default online policy samples five transactions per scheduling
  step and computes makespan only over predicted hot-key operations,
  giving linear complexity in the number of in-flight transactions and
  bounded work per hot key.
- R-SMF uses application hints, primarily transaction type plus known
  hot keys at transaction start, to predict hot-key read/write
  patterns. A simple KNN-style classifier maps metadata vectors to
  canonical hot-key operation sets learned from traces.
- The classifier can be retrained periodically. If hints become
  inaccurate, the paper suggests disabling scheduling or falling back
  after post-execution schedule-quality checks.
- MVSchedO adapts MVTSO by assigning transaction timestamps from SMF
  rather than FIFO arrival order.
- For predicted hot keys, MVSchedO maintains per-key scheduling
  queues. A read or write waits until all conflicting operations with
  lower scheduled timestamps for that key have executed.
- Non-hot-key operations execute immediately under the underlying
  MVTSO rules, keeping overhead low for low-contention traffic.
- If a predicted hot-key access never happens, dependent operations
  are released when the transaction commits or aborts.
- Correctness relies on the fact that MVSchedO permits executions that
  MVTSO could produce under a different timestamp assignment; the
  additional waiting constrains order without weakening
  serializability.
- To avoid starvation, the implementation inserts barriers into the
  scheduling queue so older requests eventually execute before newer
  ones behind the barrier.
- The paper also evaluates a bolt-on SMF layer above existing RocksDB
  OCC and locking protocols. This only delays transaction start, so it
  has smaller gains than operation-level MVSchedO, but it demonstrates
  that scheduling can be layered onto existing engines.
- Evaluation uses RocksDB 8.5 with Epinions, SmallBank, TAOBench,
  TPC-C, and YCSB, plus a TAO prototype. Reported improvements are up
  to 3.9x throughput and 3.2x tail-latency reduction in RocksDB, and
  up to 252% higher throughput with 208% lower p99 latency in the TAO
  prototype. Low-contention overhead is reported within about 5% of
  the baselines.
- Classifier accuracy is critical. On TPC-C, 10% wrong hints still
  improved throughput, but 50% wrong hints hurt performance; with no
  useful hints, scheduling mostly adds overhead.
- SMF's schedule makespan is within 10% of the best tested job-shop
  scheduling heuristics while avoiding their high offline search
  overhead, but it is still a heuristic and can be adversarially bad.

**GPU DB mapping:** GPU DB should treat this as a design for
mutation-owner and partition-owner admission, not as a replacement for
MVCC correctness. The most transferable mechanism is hot-key aware
request ordering before work enters the mutation path. For TPC-C-like
write traffic, the system can tag requests with transaction shape and
early hot keys such as warehouse, district, customer, order, item, or
table partition id, then choose an owner-queue order that separates
high-conflict writes and lets independent requests run in parallel.

The timestamp-assignment lesson is directly relevant to GPU DB's
visibility boundary. Instead of assigning mutation visibility or
snapshot generations strictly by arrival, a partition owner could
assign admission timestamps at deterministic micro-batch boundaries
chosen by a low-conflict scheduler. WAL-before-visibility still holds:
the chosen schedule only decides the serial order and hot-key wait
points; durable append, flush, invalidation, CPU-visible state update,
and resident snapshot publication remain mandatory.

For retained read routes, the paper suggests a narrower scheduling
surface than global request priority. Same-shape retained reads can
continue through read snapshot workers, but read/write or write/write
traffic on known hot keys should expose a conflict class to admission.
When a hot mutation would invalidate a resident partition, the system
can schedule mutation, refresh, and compatible reads around that
partition id rather than letting FIFO order create avoidable aborts,
refresh churn, or owner-queue stalls.

The classifier maps naturally to GPU DB route metadata. SQL template,
relation id, predicate family, key values, transaction mode, and
partition id can become the metadata vector. The first implementation
does not need ML: exact hints from parsed SQL and bind parameters are
enough for many benchmark shapes. A learned classifier only becomes
interesting when stored procedures or multi-statement sessions hide
their eventual hot keys.

**Risks and mismatches:** R-SMF assumes hot-key hints are available
early and reasonably accurate. Ad hoc SQL, multi-statement
transactions, foreign-key cascades, triggers, or queries whose hot keys
are discovered only after an index lookup may not provide enough
metadata at admission time. Wrong predictions can delay independent
work and lower throughput, so GPU DB needs an explicit fallback gate
based on observed aborts, queue wait, and prediction accuracy.

The paper focuses on logical transaction conflicts, not GPU execution,
network IO, disk flushing, or memory-tier pressure. GPU DB's bottleneck
may be CUDA launch overhead, pinned-buffer shortage, response-ring
backlog, WAL fsync, or residency refresh rather than hot-key
serialization. A schedule that minimizes logical conflict cost could
still be poor if it destroys GPU batch shape locality or starves WAL
flush groups.

MVSchedO is serializable over RocksDB-style transactional operations,
but GPU DB currently has explicit WAL, MVCC, resident invalidation, and
snapshot publication invariants. Any schedule-first protocol must
prove that delayed hot-key operations cannot expose stale resident
data, reorder pgwire responses within a session, or publish a GPU
snapshot before its durable visibility boundary.

The reported gains depend heavily on skewed workloads. Low-contention
traffic sees little benefit and still pays classifier and scheduling
cost. For 1M logical sessions, the scheduling work must also be bounded
per owner; a central scheduler over all sessions would become its own
contention point.

**Benchmark candidates:**

- Add an offline simulator for mutation-owner admission over TPC-C-like
  requests. Compare FIFO, key-partition FIFO, random deferral, and
  SMF-style sampled lowest-incremental-conflict ordering. Minimum gate:
  same serializable order semantics and lower modeled conflict wait for
  skewed warehouse/district keys.
- Prototype owner-local hot-key queues for one write-heavy SQL shape.
  Use parsed SQL and bind parameters as exact hints; do not add ML.
  Required measurements: owner queue wait, abort/retry count if any,
  WAL batch size, commit latency p50/p99, and effect on independent
  reads.
- Compare visibility timestamp assignment by arrival order versus
  schedule-selected micro-batch order. Proof gate: WAL-before-visibility
  and resident invalidation tests pass unchanged.
- Add a mixed retained-read plus mutation workload where hot writes
  invalidate one partition while other partitions stay valid. Expected
  result: hot-key/partition-aware scheduling should reduce refresh churn
  and owner queue stalls without delaying independent retained reads.
- Measure classifier/hint failure modes by intentionally corrupting
  key hints at 0%, 10%, and 50%. Failure condition: wrong hints lower
  throughput or p99 latency without triggering FIFO fallback.
- Test GPU batch-shape tension: compare conflict-optimal scheduling
  against scheduling that also preserves compatible retained query
  batches. Required measurements: batch size, CUDA launch count, queue
  wait by boundary, and transaction p99.
- Add a guardrail benchmark for low-contention traffic. Scheduling must
  stay within a small overhead budget, or auto-disable for that template
  and partition.

### 2026-06-03 - Pasha partitioned/shared CXL-pod architecture

**Citation:** Yibo Huang, Newton Ni, Vijay Chidambaram, Emmett
Witchel, and Dixin Tang. "Pasha: An Efficient, Scalable Database
Architecture for CXL Pods." CIDR 2025. Retrieved 2026-06-03
from the CIDR proceedings page and author PDF,
`https://www.vldb.org/cidrdb/2025/pasha-an-efficient-scalable-database-architecture-for-cxl-pods.html`
and `https://www.cs.utexas.edu/~witchel/pubs/huang25cidr-pasha.pdf`.

**Category:** multi-tier cache / data placement and transaction
processing / write path.

**Relevance tags:** CXL memory; future memory tiers; partition
ownership; shared region; OLTP scaling; MVCC version placement;
data movement; local DRAM versus shared memory; partial failure;
elasticity.

**Core idea:** Pasha targets a CXL pod: a small set of independent
hosts connected to shared CXL memory. The paper argues that this
hardware shape can combine the strengths of shared-nothing and
shared-memory databases. Most data remains in host-owned local
DRAM partitions, where access is cheap and synchronization is
local. Data that would otherwise force multi-host transactions is
moved into a shared CXL region, where all hosts can use ordinary
load/store access and shared synchronization metadata instead of
message-heavy two-phase commit.

The strongest result is architectural rather than a finished product:
Pasha tries to turn many multi-host transactions into single-host
transactions that access one local partition plus a shared region. In
the preliminary Sundial/TPC-C experiment, the paper reports up to
5.9x higher throughput than a partitioned shared-nothing baseline
and 1.4x higher throughput than a fully shared-memory baseline in
selected configurations. The evaluation is explicitly preliminary:
the prototype emulates an 8-host CXL pod on one machine, uses only
two worker threads per VM, and does not implement dynamic movement.

**Concrete mechanisms:**

- Data is divided into disjoint host-owned partitions and one shared
  region. Partition data lives in the owning host's local DRAM;
  shared data lives in CXL memory.
- A host that needs a tuple outside its partition asks the owning
  host to move the tuple and metadata such as locks into the shared
  region. After that, the requesting host can complete the
  transaction through local partition access plus direct shared-region
  access.
- While a tuple is resident in the shared region, even the original
  owner accesses it through the shared-region concurrency-control
  protocol.
- The paper assumes only a limited hardware-cache-coherent CXL
  region, for example hundreds of MB, while larger CXL capacity may
  need database-specific software coherence.
- One proposed split is to keep synchronization-heavy metadata in
  the hardware-coherent region and larger tuple payloads in a
  software-coherent region tracked at coarser-than-cache-line
  granularity.
- The measured CXL 1.1 device in the paper has about 2.3x local
  DRAM latency and 58% of local DRAM single-channel bandwidth, while
  still being much lower latency than RDMA-style disaggregated
  memory.
- For MVCC, the paper identifies the cost of moving all tuple
  versions as a central problem. It suggests moving only requested
  versions into the shared region and selectively moving useful
  versions back to partitions, allowing different versions of one
  tuple to reside in different places.
- The MVCC challenge is validation: a transaction may read an old
  local-DRAM version while another host creates a newer CXL-resident
  version, so the protocol must decide when that old-version read can
  still serialize safely.
- Dynamic data movement and partitioning are treated as open design
  problems. The partitioner should minimize shared-region operations,
  not merely minimize multi-host transactions.
- High core counts in a future pod motivate scheduling transactions
  before execution, rather than resolving all conflicts reactively at
  runtime.
- Durability and atomicity still require logging and checkpoints; the
  authors call out parallel logging and partial-failure recovery as
  open challenges for a pod where one host or process may fail while
  others continue.

**GPU DB mapping:** Pasha is not about GPUs, but it is highly relevant
to the future tiering and ownership model. It reinforces that a
high-throughput engine should not make one "shared memory" tier the
default home for all data. Local ownership remains valuable. For GPU
DB, the analogous rule is: keep hot partition-local mutation and CPU
canonical state close to the owner that mutates it, and use shared or
slower tiers only for data whose access pattern justifies the
coordination cost.

The shared-region idea maps to a future host-tier design beneath GPU
residency. Some data may need to be visible to multiple owners,
devices, or nodes: old snapshot side structures, cross-partition hot
tuples, shared catalog metadata, resident-generation descriptors,
route-risk summaries, and future CXL/remote-memory segments. Pasha
suggests making that shared region explicit and small, not treating
CXL or remote memory as a transparent extension of local DRAM.

For MVCC, the paper's "versions may live in different places" warning
is directly useful. GPU DB may eventually hold one tuple's latest
version in CPU canonical memory, old snapshot-visible versions in a
cold side structure, compressed host segments in a warm tier, and
read-optimized column copies in GPU memory. The visibility design must
validate the version chain and placement generation together. A read
snapshot cannot only know `txn_id`; it also needs a stable placement
handle and source generation for every version or segment it may read.

Pasha also strengthens the case for partition owners. If a tuple or
segment moves between owner-local state and a shared tier, movement is
a transactionally visible event that needs ordering, logging,
invalidation, and reader safety. That is close to P8's resident
refresh problem: moving a partition into GPU memory or shared host
memory should publish a new immutable generation only after the CPU
truth and WAL boundary are stable.

Finally, the partial-failure discussion matters for future scale-out.
GPU DB's current single-process path can treat CUDA buffers and
resident snapshots as rebuildable acceleration state. If later CXL or
multi-host tiers appear, the engine should preserve that discipline:
durable WAL/checkpoint authority first, rebuildable shared placement
metadata second, and explicit recovery paths for a failed owner or
movement operation.

**Risks and mismatches:** Pasha is a CIDR architecture paper with
preliminary experiments, not a complete evaluated database. The CXL
pod hardware assumed by the design, especially fine-grained
cross-host cache coherence, was not commercially available for the
full prototype. The experiment uses VMs on one host and a CXL 1.1
device, so inter-host coherence is faster than a real pod would be.

The implementation does not support dynamic data movement; shared
TPC-C tables are pre-moved before tests. MVCC support, software
coherence, partitioning policy, parallel logging, recovery, partial
failure handling, and auto-scaling are research challenges rather
than solved mechanisms. For GPU DB specifically, CXL memory does not
replace GPU HBM, CUDA stream ownership, pinned buffers, WAL ordering,
or pgwire response backpressure. A CXL shared region could easily
become a new contention point if treated as a general heap.

**Benchmark candidates:**

- Add a placement-state simulator for P8 segments: owner-local CPU
  partition, shared host tier, GPU resident generation, and evicted
  cold state. Minimum gate: every movement has an ordered source
  generation, target tier, bytes moved, and reader-visible state.
- Build a version-placement MVCC test where different versions of one
  logical row are held in different simulated tiers. Proof gate:
  snapshot visibility remains correct across update, movement,
  invalidation, replay, and GC boundaries.
- Compare partition-local versus shared-tier metadata for hot
  resident-generation descriptors. Required measurements: lookup
  latency, cache-line contention proxy if available, invalidation
  cost, and effect on independent partitions.
- Add a movement-policy benchmark for a skewed workload: keep hot
  partition-local data local, move cross-partition hot rows or
  metadata to a shared tier, and measure write latency, read latency,
  movement bytes, and invalidation churn.
- Add a future-tier capability matrix to the research backlog:
  local DRAM, pinned DRAM, GPU HBM, CXL system memory,
  hardware-coherent CXL shared region, software-coherent CXL region,
  NVMe, and remote memory. Include direct GPU access, coherence,
  durability role, movement primitive, expected latency/bandwidth, and
  whether the tier is safe for mutable state.
- Treat CXL/shared memory as a future benchmark track only after the
  current owner/ring/resident-snapshot design can report per-tier
  movement and invalidation telemetry. Failure condition: CXL-like
  placement hides stale snapshot reads or turns movement into
  unbounded background work.

## Cross-Paper Synthesis

### 2026-06-03 - Owner-local first, shared only when measured

The recent reviewed papers now converge on a sharper architecture
track: keep mutable hot paths owner-local, expose queue and movement
pressure explicitly, and admit shared or accelerated tiers only when a
route descriptor can prove the generation, bytes, and conflict class.
Pasha adds future CXL/shared-region pressure to this picture; SMF adds
schedule-first hot-key ordering; Shenango adds persistent queue-delay
feedback; vmcache and vmcache^n add explicit tier movement; PARQO and
PAR2QO add risk-aware route reuse.

Converging design tracks:

- **Owner-local mutation first:** partition owners should keep hot
  writes, WAL ordering, and local visibility state close to the owner;
  shared tiers should hold only data or metadata whose cross-owner use
  repays the synchronization cost.
- **Descriptor-gated movement:** moving data to GPU, host warm memory,
  CXL-like shared memory, or NVMe should be an explicit transition with
  generation, bytes, deadline, invalidation risk, and fallback reason.
- **Conflict-aware admission:** hot-key scheduling, queue-duration
  feedback, and route-risk penalties should all feed the same admission
  surface instead of becoming separate ad hoc knobs.
- **Version placement is visibility state:** if tuple versions,
  tombstones, resident segments, or old snapshot side structures live
  in different tiers, a snapshot handle must validate both transaction
  visibility and placement generation.

Current category gaps: GPU execution has enough recent coverage for
now; the next queued paper should favor MVCC/GC, transaction write
path, multi-tier placement, or high-concurrency runtime before another
GPU OLAP paper. The weakest unsolved area is still production-safe
movement: how to move data among owners and tiers without weakening
WAL-before-visibility or causing unbounded snapshot retention.

Benchmark priorities:

- Implement tier/movement telemetry before adding another cache layer:
  source tier, target tier, generation, bytes, queue wait, invalidation
  boundary, and fallback reason.
- Add an MVCC version-placement stress test with versions and
  tombstones split across simulated tiers.
- Compare FIFO, hot-key-aware, and queue-delay-aware admission on a
  mixed retained-read plus mutation workload.
- Require every GPU or future CXL route to report whether it improved
  latency/throughput by reducing movement, reducing conflicts, or only
  shifting work to a less visible queue.

### 2026-06-03 - P-Tree multi-versioned indexes for HTAP snapshots

**Citation:** Yihan Sun, Guy E. Blelloch, Wan Shen Lim, and Andrew
Pavlo. "On Supporting Efficient Snapshot Isolation for Hybrid
Workloads with Multi-Versioned Indexes." PVLDB 13(2), 2019.
Retrieved 2026-06-03 from the PVLDB PDF,
`https://www.vldb.org/pvldb/vol13/p211-sun.pdf`.

**Category:** MVCC / snapshot / visibility and hybrid HTAP.

**Relevance tags:** snapshot isolation; immutable indexes; path
copying; functional data structures; batched writes; nested indexes;
precise GC; read-only HTAP; version retention; parallel bulk updates.

**Core idea:** P-Trees replace tuple-local version chains with
immutable, path-copied index roots. A read transaction acquires a
stable root pointer, and writers create a new tree version by copying
only changed paths. Committing a transaction or update batch publishes
a new top-level "world" root that points at all current indexes, so
readers continue on old roots while new readers see the latest
published root.

The paper's strongest transfer is that visibility can be represented
as an immutable access structure rather than as repeated per-tuple
version-chain traversal. This is attractive for GPU DB retained reads:
the runtime already wants immutable resident snapshots, and P-Trees
show a CPU-side index/snapshot discipline where acquisition is cheap,
readers are wait-free, and old versions are reclaimed precisely once
no root references them.

**Concrete mechanisms:**

- Each tree node stores key/value data, child pointers, subtree size,
  and a reference count. Updates copy affected paths and share
  unchanged subtrees across versions.
- A top-level world object stores pointers to all current indexes.
  Acquiring a snapshot increments the root/world reference; committing
  swaps the current world pointer to a new version.
- Analytical operations such as range, filter, map-reduce, and
  foreach-index are pure and can produce more P-Trees or values
  without modifying input snapshots.
- Bulk insert/delete use sorted update arrays plus divide-and-conquer
  tree operations, allowing an update batch to be committed in
  parallel while readers keep using the prior root.
- Nested and paired indexes embed one tree inside another, providing a
  virtual pre-join or hierarchy without materializing a copied table.
  Updates path-copy through both outer and inner trees.
- For serializable updates, the paper favors batching: collect writes
  for a short interval, detect logical conflicts according to a linear
  order, remove conflicted operations, and commit the conflict-free
  remainder as one parallel batch.
- Memory management uses per-thread node pools plus shared free-node
  blocks. Reference-counted GC recursively frees nodes when a released
  snapshot drops their count to zero.
- Evaluation reports P-Trees outperforming or matching several
  concurrent in-memory tree indexes on YCSB, 4-9x faster analytical
  queries than HyPer/MemSQL on their TPC-H setup, average 62x
  parallel speedup on 72 cores/144 hardware threads, and update
  throughput close to MemSQL on the hybrid TPC-HC workload. The
  update path is still weaker for some write-heavy transactions.

**GPU DB mapping:** The direct GPU DB mapping is a two-level
visibility handle: a CPU canonical immutable snapshot root plus a GPU
resident layout generation derived from that root. Current P8
resident snapshots already carry source WAL/transaction boundaries;
P-Trees suggest that the source boundary should also be a durable
access-structure identity, not only a scalar transaction id. A read
route would prove compatibility by checking the world/root generation,
table/schema identity, resident generation, and invalidation boundary.

The batched commit model maps to mutation-owner admission. Rather
than updating WAL, CPU indexes, MVCC metadata, and resident
invalidation independently per request, a hot partition owner could
drain a bounded write batch, sort/group by key or segment, apply
parallel CPU index changes to a new immutable root, then publish one
visibility boundary and one resident invalidation generation. This
keeps WAL-before-visibility intact while giving reads a stable root.

Nested indexes are useful as a design warning and opportunity. For GPU
lookups, a resident key vector plus column buffers is a narrow nested
structure: it can pre-filter or group rows before scan/aggregation.
But full P-Tree-style nested pre-joins would consume memory and create
refresh work. The product version should treat nested resident
structures as explicit route families with byte cost, update cost, and
invalidation telemetry.

The GC design reinforces the need for retained snapshot retirement.
GPU DB should account for old CPU roots, old resident buffers, and
old placement descriptors together. A snapshot release should make it
clear which CPU index nodes, pinned buffers, and GPU generations are
eligible for reclamation, instead of leaving invalidated resident state
to unbounded background cleanup.

**Risks and mismatches:** P-Trees are CPU in-memory structures, not
GPU kernels or disk-backed production storage. Their pointer-rich
trees are not a natural GPU scan layout, and path copying can be more
expensive than tuple-version append for small, high-rate OLTP writes.
The paper serializes or batches updates in the tested DBMS; that is
acceptable for read-dominant HTAP, but not enough by itself for a
general high-write transaction engine.

The reported OLAP gains depend heavily on nested indexes and a
custom benchmark implementation. HyPer and MemSQL are full systems
with different optimizers, compilation paths, compression, durability,
and production features, so the speedups should be treated as design
signals rather than direct product targets. Reference counting also
adds cache-line contention; the paper reports frequent GC can be
costly, although batching reclamation reduces the overhead.

**Benchmark candidates:**

- Add a simulated immutable table-root generation to the P8 route
  model. Proof gate: retained reads validate table root, schema
  generation, WAL boundary, resident generation, and invalidation
  generation before executing.
- Prototype bounded mutation-owner batch publication for one table:
  drain writes for count/time threshold, apply CPU index/MVCC changes
  to a new generation, invalidate resident state, then publish
  visibility. Required measurements: batch size, WAL latency, root
  publish latency, invalidation count, read fallback count, and p99.
- Compare version-chain visibility lookup against root-generation
  lookup for point reads under long retained snapshots. Minimum gate:
  same correctness across update/delete/replay and lower per-read
  visibility cost when many old versions exist.
- Add a snapshot-retirement accounting test that releases CPU roots,
  resident GPU buffers, and placement descriptors together. Failure
  condition: old generations remain live after the last reader or are
  freed while still referenced.
- Evaluate a narrow nested-resident route for one parent/child or
  key-to-row-group pattern. Required telemetry: resident bytes,
  refresh bytes, invalidation churn, GPU kernel count, and whether
  pre-filtering beats a flat resident scan.
- Stress the batching latency tradeoff with 1 ms, 5 ms, and 50 ms
  mutation-owner windows. The batch path should auto-disable or shrink
  when p50/p99 latency regresses more than the throughput gain
  justifies.

### 2026-06-03 - Runtime-conflict transaction scheduling

**Citation:** Yang Cao, Wenfei Fan, Weijie Ou, Rui Xie, and Wenyue
Zhao. "Transaction Scheduling: From Conflicts to Runtime
Conflicts." Proceedings of the ACM on Management of Data 1(1),
article 26, SIGMOD 2023. Retrieved 2026-06-03 from the University
of Edinburgh accepted manuscript,
`https://www.pure.ed.ac.uk/ws/portalfiles/portal/360117816/Transaction_Scheduling_CAO_DOA16082022_AFV.pdf`.
DOI: `https://doi.org/10.1145/3603164`.

**Category:** transaction processing / write path and runtime /
session-scale scheduling.

**Relevance tags:** OLTP scheduling; runtime conflicts; contention
management; transaction partitioning; proactive deferment; lock-free
progress tracking; abort reduction; owner queues; admission control.

**Core idea:** The paper argues that conventional conflict graphs are
too conservative for multicore OLTP scheduling. Two transactions may
touch conflicting records, but if their scheduled execution intervals
do not overlap, they do not conflict at runtime. The proposed TSkd
tool therefore adds ordering to transaction partitions and treats
runtime overlap as a first-class scheduling dimension, not just key-set
intersection.

The strongest transfer for GPU DB is the distinction between "may
conflict" and "will conflict on this schedule." Current owner/ring
planning tends to bucket work by table, partition, route shape, and
visibility generation. This paper suggests adding a small predicted
runtime window and conflict class so the mutation owner can avoid
needlessly serializing operations whose hot regions or owner phases
do not overlap in time.

**Concrete mechanisms:**

- TSkd has two modules. TsPar turns an existing transaction partition
  into ordered per-thread queues plus a residual set, aiming to
  minimize both makespan and unscheduled residual work. TsDefer works
  online for residual or unbundled transactions by deferring a
  transaction that is likely to collide with currently active work.
- Runtime conflict is defined by both logical conflict and scheduled
  interval overlap. A schedule assigns transactions to queues and
  orders each queue; two logically conflicting transactions are
  runtime-conflict-free if their scheduled runtime intervals do not
  overlap.
- The exact scheduling problem is NP-complete, so the paper uses
  TSgen, a heuristic that reuses the partitioner's conflict graph,
  examines residual transactions, chooses the least-loaded queue, and
  appends a residual transaction only if the resulting queues remain
  runtime-conflict-free.
- TsPar uses rough runtime estimates from execution history,
  partitioner dry-runs, nearby template instances, or fallback access
  set sizes. It is sensitive mainly to relative transaction lengths,
  not exact wall time.
- TsDefer keeps a lock-free progress structure for thread-local
  buffers: each thread owns its queue array plus head and tail
  pointers, while other threads read progress. Before execution, a
  thread performs a bounded number of random lookups into active
  remote transactions' predicted write sets and may move the current
  transaction to the queue tail with configurable probability.
- The deferment knobs are `#lookups` and `deferp%`, trading detection
  overhead against abort/retry reduction. The paper reports best
  short-transaction throughput around two lookups in its tested
  TPC-C/YCSB setup, with larger lookup counts reducing retries but
  adding overhead.
- Evaluation integrates TSkd into DBx1000 with Strife, Schism,
  Horticulture, and CC-only baselines. Reported averages are 131%
  higher throughput for partitioner-based systems and 109% higher
  throughput for CC-only systems, with retry reductions around 45%.
  Benefits are stronger under contention, more cores, longer or more
  variable transaction runtimes, and simulated I/O latency.

**GPU DB mapping:** GPU DB should not copy TSkd as a transaction
partitioner, but the runtime-conflict model maps cleanly to owner
queue admission. A mutation or refresh request can carry a conflict
class, key or partition estimate, predicted CPU/GPU phase duration,
and visibility boundary. The owner can then choose between immediate
execution, short deferment, batching, or fallback based on predicted
overlap, not just "same table" conflict.

For write admission, this complements the batch-publication ideas in
P-Trees and the owner-local tracks from prior synthesis. A partition
owner could drain a bounded window, sort writes by key/segment and
estimated duration, and publish one WAL/visibility boundary while
keeping conflicting long operations from causing avoidable retries or
queue stalls. The key invariant is that reordering happens inside an
explicit admission boundary; external visibility still follows
WAL-before-visibility.

For read paths, TsDefer suggests a cheap pre-execution filter before
routing a retained read to a busy GPU execution queue. If an active
mutation, refresh, or eviction is likely to invalidate the same
partition generation during the read's window, the runtime can defer
briefly, use a still-valid older snapshot, or choose CPU fallback
instead of entering a route that will fail late. This is especially
relevant for 1M logical sessions where many requests may target the
same small set of hot keys.

For observability, the paper reinforces that scheduling should expose
conflict penalties as telemetry. GPU DB should count predicted
runtime conflicts, actual aborts/retries or fallbacks, defer count,
defer wait, active owner phase, active GPU phase, and avoided stale
snapshot attempts. Without those counters, a deferral policy could
look like lower latency while only shifting time into an invisible
queue.

**Risks and mismatches:** TSkd assumes useful read/write-set or access
set estimates for many workloads. That fits stored-procedure-like
OLTP better than arbitrary SQL text, ad hoc predicates, range scans,
or query plans with data-dependent access. GPU DB would need planner
route descriptors and index/statistics support before it can predict
conflicts well enough for broad SQL.

The paper is evaluated in DBx1000, not in a durable production DBMS
with WAL, recovery, DDL, network backpressure, GPU kernels, or
resident cache invalidation. Its RC-free queues are still executed
with CC in the prototype to preserve correctness under inaccurate
estimates, so the reported speedups are signals about conflict
reduction rather than proof that CC can be removed safely.

Deferral can also hurt fairness and tail latency. A hot transaction
may be repeatedly moved to the back of a queue, and a high lookup
count may cost more than the conflict it avoids for short requests.
GPU DB should treat deferment as bounded admission control with
per-class limits, not as an unbounded retry-avoidance trick.

**Benchmark candidates:**

- Add a runtime-conflict admission simulator for one partition owner:
  inputs are key or segment ids, predicted CPU phase, predicted GPU
  phase, and operation class. Compare FIFO, key-grouped batching, and
  runtime-conflict-aware ordering. Minimum gate: same serializable
  result and WAL publication order, with lower retry/fallback count or
  lower p99 under skew.
- Prototype bounded deferment in the benchmark endpoint's retained
  request scheduler without changing SQL semantics. Measurements:
  defer count, defer wait, queue wait, actual fallback count, p50/p99,
  and starvation count. Failure condition: any request exceeds a
  configured defer budget without explicit overload/fallback reason.
- Add an active-phase conflict probe for retained reads: before
  entering a GPU route, check whether the relevant partition has an
  active mutation, refresh, eviction, or invalidation phase likely to
  cross the read window. Proof gate: fewer late invalidation fallbacks
  without stale reads.
- Stress mixed short and long work: short point reads, short writes,
  long refreshes, and long analytical retained scans against hot and
  cold partitions. Compare FIFO owner queues with runtime-window-aware
  admission. Required output: throughput, p99, max wait, conflict
  class, and whether long work starves short retained reads.
- Test `#lookups`-style sampling for conflict prediction using only
  bounded metadata: sample active owner/GPU phases and key ranges
  rather than reading full access sets. Minimum gate: constant-time
  probe with measurable avoided retries/fallbacks.
- Keep the first production rule simple: only allow reordering inside
  an explicit owner batch whose WAL and visibility boundary is known.
  Failure condition: a schedule can make an externally visible result
  differ from FIFO under the same committed transaction order.

### 2026-06-03 - Semantic OCC batching and operation reordering

**Citation:** Bailu Ding, Lucja Kot, and Johannes Gehrke.
"Improving Optimistic Concurrency Control Through Transaction
Batching and Operation Reordering." PVLDB 12(2), 2018.
Retrieved 2026-06-03 from the PVLDB PDF,
`https://www.vldb.org/pvldb/vol12/p169-ding.pdf`. DOI:
`https://doi.org/10.14778/3282495.3282502`.

**Category:** transaction processing / write path and runtime /
session-scale scheduling.

**Relevance tags:** optimistic concurrency control; semantic
batching; validator ordering; storage ordering; feedback vertex set;
tail latency; abort reduction; thread-aware scheduling; write
admission; owner queues.

**Core idea:** The paper treats batching as a semantic transaction
mechanism rather than only a message-packing or group-commit trick.
In a decoupled OCC architecture, transactions read from storage, send
read/write sets to a validator, then install writes if validation
succeeds. Because OCC chooses the final serialization order at commit
time, a batch can be reordered to reduce avoidable stale-read aborts.

The strongest transfer for GPU DB is the separation between physical
batching and semantic batch validity. A mutation owner should not
batch only because it amortizes WAL, index, or GPU-refresh costs. It
should batch because the requests in that boundary can be ordered,
validated, and published with fewer conflicts while preserving a clear
external visibility order.

**Concrete mechanisms:**

- Storage batching buffers reads and writes together. For each object,
  it applies the highest-version pending write first and discards
  older writes for that object, then serves reads. In the paper's OCC
  model, a read that sees an older value while a committed pending
  write to the same object exists is likely to abort later.
- Validator batching buffers validation requests and chooses a
  serialization order for the batch. If two transactions conflict only
  because the writer arrived at the validator before the reader, the
  validator can serialize the reader before the writer and commit both.
- Intra-batch validator reordering is represented as a directed
  dependency graph over transactions, with edges for read-write
  dependencies. If the graph is acyclic, a topological order commits
  all viable transactions. If cycles exist, the validator must abort a
  feedback vertex set to make the graph acyclic.
- Finding the minimum feedback vertex set is NP-hard, so the paper
  proposes greedy algorithms: SCC-based removal, faster sort-based
  removal, and a hybrid that uses precise search for small SCCs. The
  default fast path uses sort-based greedy ordering with degree-based
  policies.
- Policies can optimize different goals. A commit-maximizing policy
  removes high-degree nodes from cycles; a tail-latency policy protects
  transactions that have already restarted; a value policy can prefer
  application-priority transactions.
- For decentralized OCC systems without a central validator, the paper
  uses a thread-aware preprocessing policy: batch transactions, assign
  conflicting transactions to the same execution thread, and exploit
  both lower inter-thread conflict and better cache locality.
- The parallel validator splits batch preparation, transaction
  reordering, and final validation into pipeline components. It also
  pre-validates against committed state before expensive reordering,
  then validates again against the latest state before commit.
- Evaluation reports up to 2.7x prototype throughput improvement and
  up to 82% tail-latency reduction from storage plus validator
  reordering under skew; in Cicada/YCSB, thread-aware reordering gives
  up to 2.2x throughput and 71% 99th-percentile latency reduction; in
  a commercial DBMS-X setup, batching plus reordering improves peak
  throughput by 1.25x, throughput by up to 3.1x, average latency by up
  to 66%, and abort rate by up to 62%.

**GPU DB mapping:** The paper gives a concrete rule for mutation-owner
batching: order inside the batch, publish outside the batch. GPU DB can
collect a bounded owner batch of writes, refreshes, and retained-route
invalidations, build a dependency graph from keys, partitions,
resident generations, and read/write intent, then publish one
WAL-backed visibility boundary after validation. This is stronger than
blind FIFO draining and safer than letting GPU refresh work reorder
around transactional writes.

Storage batching maps to the P8 resident invalidation path. If a
pending committed write will invalidate a resident generation, reads
queued behind it should not be routed to the soon-stale generation just
because the write has not yet been physically applied. The owner can
prefer "apply invalidation/write intent, then admit reads" inside the
batch, while still letting older snapshot holders finish on an already
acquired immutable generation.

Validator reordering maps to a small owner-local conflict graph rather
than a global transaction scheduler. Nodes can represent mutations,
refresh publications, or read-only retained requests that require a
fresh boundary. Edges should be limited to bounded metadata such as
key/range conflicts, partition ids, write-intent ids, and resident
generation invalidations. The first implementation should avoid
unbounded arbitrary SQL read/write set extraction.

The policy framework also fits 1M logical sessions. Long-restarted,
latency-sensitive, or user-visible short requests can receive higher
priority inside a batch without removing correctness checks. The
runtime should expose the policy choice as telemetry: batch size,
cycles found, transactions deferred or aborted, protected retries,
queue wait, validation wait, and whether the batch improved p99 or only
hid work behind the owner.

Thread-aware assignment reinforces current owner-domain thinking. If
conflicting requests are placed on the same owner/partition thread,
they serialize cheaply and share cache or resident metadata. If they
are spread across owners, the system needs more validation,
invalidation, or fallback coordination. This suggests benchmarking
owner assignment and partition splitting as concurrency-control
decisions, not just load-balancing decisions.

**Risks and mismatches:** The paper assumes OCC-style read/write sets
and versioned storage that can ignore older writes when a higher
version exists. GPU DB's SQL plans, MVCC tuple chains, WAL replay, DDL,
resident snapshots, and GPU execution routes need stronger metadata
than the paper's key-value examples. Reordering cannot cross external
transaction boundaries unless the committed order and visibility
boundary remain unambiguous.

Validator reordering adds latency. The paper shows sort-based greedy
ordering is usually the best tradeoff, and that validator reordering
can hurt throughput under extreme contention because dependency graphs
become dense. GPU DB should therefore make semantic batching adaptive:
disable or shrink it for low-conflict templates, cap batch wait, and
fall back to simpler storage/invalidation ordering when graph work
costs more than avoided retries.

The evaluation is mostly in research prototypes, an integration with
Cicada, and an anonymized commercial DBMS middle-tier experiment. It
does not prove durability, crash recovery, GPU route correctness,
network backpressure, or multi-tier snapshot reclamation. Treat the
reported speedups as a strong scheduling signal, not as a directly
portable throughput target.

**Benchmark candidates:**

- Build an owner-batch simulator with key/range conflicts, resident
  invalidation edges, and refresh-publication edges. Compare FIFO,
  storage-write-first ordering, and sort-based feedback-vertex-set
  ordering. Proof gate: identical committed results under the selected
  serial order and lower abort/fallback count under skew.
- Add a bounded mutation-owner batch mode for one table: drain up to N
  requests or T microseconds, validate order, append WAL, invalidate
  resident generations, apply CPU state, and publish one visibility
  boundary. Required telemetry: batch wait, graph edges, cycles,
  aborted/deferred requests, WAL latency, invalidation count, p50/p99.
- Test storage-level read/write ordering for retained routes: when a
  write intent and compatible read are in the same owner batch, mark
  the resident generation invalid before admitting new reads that need
  fresh visibility. Failure condition: a post-mutation read executes
  on a generation that should have been invalidated.
- Compare policy modes: maximize commits, protect restarted requests,
  and protect latency-sensitive reads. Minimum gate: policy telemetry
  shows which transactions were protected and whether p99 improves
  without starving writes.
- Add a thread/owner assignment microbenchmark where conflicting hot
  keys are intentionally co-located or split across owners. Required
  measurements: throughput, p99, cache/metadata hit proxy, cross-owner
  invalidations, and fairness.
- Add an adaptive-batch guardrail: semantic reordering must auto-shrink
  or disable when graph construction/reordering overhead exceeds the
  measured conflict or fallback reduction.

### 2026-06-03 - Cross-paper synthesis: batch boundaries as correctness surfaces

The last three modern reviews sharpen one design track from different
angles. P-Trees make immutable roots and batch publication attractive
for HTAP snapshots. Runtime-conflict scheduling says conflicts are a
function of both keys and execution windows. OCC batching shows that a
batch can be a semantic reorder/validation boundary, not merely a
queue-drain artifact.

Converging design tracks:

- **Batch publication over per-request churn:** hot write paths should
  publish WAL, CPU visibility, resident invalidation, and route
  generations at explicit owner boundaries when latency budgets permit.
- **Local conflict graphs before global scheduling:** start with
  owner-local metadata graphs over keys, partitions, resident
  generations, and active phase windows before attempting broad SQL
  transaction prediction.
- **Immutable snapshot roots plus placement handles:** a retained read
  should validate root generation, WAL/visibility boundary, resident
  generation, placement state, and invalidation boundary as one handle.
- **Adaptive policy is mandatory:** batching, reordering, and deferment
  must shrink or disable when dependency graphs are dense, contention
  is low, or tail latency worsens.

Category gaps: recent work is now rich in transaction scheduling,
MVCC roots, and owner batching. The next review should shift toward
multi-tier placement/GC or high-concurrency runtime/networking before
another optimizer or GPU-OLAP paper. Good next candidates are Nomad,
Hybrid GC for SAP HANA, Taurus logging, or Take Out the TraChe.

Benchmark priorities:

- Implement one owner-batch experiment that includes both transaction
  ordering and resident invalidation ordering.
- Add a snapshot-handle proof that couples immutable CPU root identity
  with GPU resident placement generation.
- Track conflict-graph overhead as a first-class metric; a scheduler
  that reduces aborts while increasing p99 should fail the gate.
- Add an adaptive mode that reports why it chose FIFO, write-first,
  conflict-aware reorder, deferment, or CPU fallback for each batch.

### 2026-06-03 - NOMAD non-exclusive memory tiering

**Citation:** Lingfeng Xiang, Zhen Lin, Weishu Deng, Hui Lu, Jia
Rao, Yifan Yuan, and Ren Wang. "Nomad: Non-Exclusive Memory
Tiering via Transactional Page Migration." OSDI 2024. Retrieved
2026-06-03 from the USENIX page and PDF,
`https://www.usenix.org/conference/osdi24/presentation/xiang` and
`https://www.usenix.org/system/files/osdi24-xiang.pdf`.

**Category:** multi-tier cache / data placement and runtime memory
management.

**Relevance tags:** tiered memory; CXL; non-exclusive placement;
transactional page migration; shadow copies; asynchronous promotion;
thrashing; access tracking; demotion cost; placement telemetry.

**Core idea:** NOMAD challenges exclusive memory tiering, where a
page exists in either fast memory or slow memory but not both. In
newer byte-addressable tiers such as CXL memory, persistent memory,
and storage-class memory, the slow tier can often be read directly.
If the fast tier is under pressure, repeatedly migrating pages can
cost more than leaving warm pages where they are.

The paper proposes non-exclusive tiering: recently promoted pages can
keep shadow copies in the capacity tier. That makes clean demotion
cheap because the system can remap back to the existing shadow instead
of copying data again. The strongest GPU DB transfer is not to let
"move hot data upward" become a reflex. Every GPU/DRAM/NVMe/CXL
promotion should prove that movement cost is lower than direct access,
fallback, or keeping a shadow copy in a lower tier.

**Concrete mechanisms:**

- NOMAD uses transactional page migration for promotion. It copies a
  page from the capacity tier to the fast tier while the original page
  remains mapped and accessible. After the copy, it checks whether the
  page was dirtied during migration. If dirtied, the migration aborts
  and the copied page is discarded; otherwise, the page table mapping
  switches to the fast copy.
- Successful promotion leaves the old capacity-tier page as a shadow
  copy. A clean fast-tier page with a valid shadow can be demoted by
  remapping rather than copying back to the slow tier.
- Shadowing is not allowed to cause unbounded memory use. NOMAD
  safeguards allocation and, when the capacity tier is pressured,
  reclaims shadow pages before evicting ordinary pages.
- The system separates mechanism from policy. The mechanism makes
  migration asynchronous and shadow-backed; the paper notes that
  promotion throttling and better global working-set estimation remain
  policy work.
- NOMAD relies on page-fault-based access tracking, while comparing
  against approaches such as TPP and Memtis that differ in recency,
  frequency, and hardware-counter sampling behavior. The paper argues
  that fault-based tracking is responsive but can be expensive if it
  lands on the critical path.
- Evaluation spans multiple platforms, including CXL and persistent
  memory setups. The paper reports up to 6x improvement over Linux TPP
  under memory pressure, up to 130% over Memtis in some fast-tier-fit
  cases, and also shows workloads such as moderate PageRank where page
  migration is unnecessary or gives negligible benefit.
- The authors explicitly identify weaknesses: when the working set
  exceeds fast-tier capacity, disabling migration can be best; knowing
  when to resume migration is hard because the working set spans tiers;
  and access tracking has a recency/frequency tradeoff.

**GPU DB mapping:** P8 already treats GPU residency as a performance
cache over CPU/WAL truth. NOMAD suggests extending each placement
descriptor with a shadow/copy relationship: GPU HBM may hold a hot
column group, host DRAM may hold the canonical encoded segment, and a
lower tier may hold a warm compressed or page-aligned shadow. Clean
demotion should prefer metadata remap or handle downgrade over
physical copy when a valid lower-tier shadow exists.

Transactional migration maps to resident refresh and promotion. A
GPU DB promotion from host memory or NVMe into GPU memory should copy
or decode into a candidate generation while existing readers continue
on the old placement. At publish time, the residency owner checks the
WAL/visibility boundary, invalidation generation, dirty/mutation
state, and memory budget. If any changed during migration, discard the
candidate generation rather than publishing a stale resident route.

The non-exclusive idea also affects planner route choice. If a warm
partition can be queried directly from host memory or staged with a
small transfer, migrating it fully into GPU memory may increase tail
latency and eviction churn. Route costing should include movement
cost, expected reuse, shadow validity, fast-tier pressure, and whether
the query is latency-sensitive enough to prefer direct lower-tier
access.

For session concurrency, asynchronous migration belongs behind bounded
residency queues. IO workers and read snapshot workers should not
block on page/segment movement unless the query explicitly requires a
fresh resident generation. Telemetry should distinguish queue wait,
copy/decode time, publish validation, aborted promotion, shadow hit,
demotion remap, and real eviction.

**Risks and mismatches:** NOMAD is an OS page-management mechanism,
not a DBMS storage engine. It does not handle SQL visibility, WAL,
catalog generation, row/column encodings, GPU kernel scheduling, or
query-planner semantics. GPU DB cannot let Linux page remapping be the
only correctness boundary for resident snapshots.

The page size and access pattern also differ. GPU DB may move column
groups, compressed chunks, key vectors, pinned buffers, or partition
generations rather than 4KB pages. Shadow copies consume memory and
can become stale quickly under writes, so a DBMS version needs
explicit invalidation, dirty tracking, and reclamation budgets.

Finally, the evaluation shows that migration is not always useful.
For some memory-intensive but not latency-sensitive workloads, direct
remote/CXL access or no migration performed similarly. GPU DB should
therefore benchmark "do not promote" as a first-class policy, not
only compare different promotion strategies.

**Benchmark candidates:**

- Add a tier-placement simulator for one retained partition: exclusive
  placement, inclusive cache, and non-exclusive shadow placement.
  Measurements: movement bytes, remap count, aborted promotion,
  shadow-hit demotion, p50/p99, resident bytes, and fallback count.
  Minimum gate: non-exclusive placement reduces movement or p99 under
  memory pressure without stale reads.
- Prototype transactional resident promotion: build a candidate GPU
  generation while the old generation remains available, then publish
  only if WAL boundary, invalidation generation, schema generation,
  and memory budget still match. Failure condition: any stale or
  post-mutation route is admitted.
- Benchmark "query lower tier directly" against "promote then query"
  for warm host-memory and simulated NVMe partitions. Required output:
  transfer/copy/decode time, GPU kernel time, route latency, and
  expected reuse threshold where promotion wins.
- Add shadow-aware demotion telemetry to the cache manager design:
  evicted bytes, remapped bytes, copied-back bytes, dirty-shadow
  misses, and shadow reclamation reason.
- Stress fast-tier pressure with retained reads plus writes that
  invalidate resident generations. Compare eager promotion, throttled
  promotion, and no-promotion policies. Proof gate: policy reports why
  it moved, skipped, or aborted movement.
- Test access-tracking signals separately: recency-only, frequency
  sampling, route-template reuse count, and mutation invalidation rate.
  Failure condition: the tracking overhead or policy churn exceeds the
  benefit of avoiding cold-route fallback.

### 2026-06-03 - DeToX transactional cache hit rate

**Citation:** Audrey Cheng, David Chu, Terrance Li, Jason Chan,
Natacha Crooks, Joseph M. Hellerstein, Ion Stoica, and Xiangyao
Yu. "Take Out the TraChe: Maximizing (Tra)nsactional Ca(che)
Hit Rate." OSDI 2023, pp. 419-439. Retrieved 2026-06-03 from
the USENIX page and PDF,
`https://www.usenix.org/conference/osdi23/presentation/cheng`
and `https://www.usenix.org/system/files/osdi23-cheng.pdf`.

**Category:** runtime / HFT / session scale, with transaction-aware
cache and data-placement policy.

**Relevance tags:** transactional hit rate; correlated key groups;
critical path latency; cache admission; eviction; prefetching; hot-key
contamination; session-heavy reads; TAOBench; Redis/Postgres/TiKV;
serializable cache coherence.

**Core idea:** DeToX argues that object hit rate is the wrong
metric for transactional workloads. A transaction often benefits from
cache only when a whole dependency group is cached. If one object in a
parallel group still goes to the slower backing store, end-to-end
latency is dominated by that miss, so caching the other objects may
consume capacity without improving latency.

The paper introduces transactional hit rate: the reduction in a
transaction's critical length after cached vertices are removed from
its execution DAG. DeToX uses this metric to score groups of keys
rather than independent objects. The strongest GPU DB transfer is to
treat retained GPU residency and host/NVMe cache placement as
transaction-template placement, not only per-segment popularity. A hot
key, hot partition, or hot resident column group is not automatically
valuable if it is normally requested with uncached companions that keep
the request on the slow path.

**Concrete mechanisms:**

- The paper models a transaction as a DAG whose vertices are reads or
  writes and whose edges are logical dependencies. A cache state
  shortens latency by removing cached vertices from the longest
  non-cached path, called the critical length.
- A complete group is a minimal set of keys whose presence in cache
  reduces critical length. DeToX identifies table-level complete groups
  from transaction execution graphs at compile time, then maps runtime
  key accesses into those groups.
- Group score combines the minimum key frequency in the group, the
  critical-length reduction from caching the group, and group size.
  The minimum frequency matters because a cold companion key can
  contaminate a hot key: the group only helps if all required keys fit.
- Key scores are derived from group scores. DeToX greedily scores the
  highest-value complete group first, then scores remaining keys using
  the best larger complete group that includes already-scored keys.
  Across transactions, it averages instance scores and adds a global
  recency aging factor similar to Greedy-Dual-Size-Frequency.
- Complete groups can be exponential in transaction size, so the paper
  defines interchangeable groups: sets of keys that can be substituted
  for each other in any complete group without changing critical-length
  reduction. This compresses runtime scoring work.
- When transaction code is unavailable, DeToX approximates groups with
  levels: keys sent to the backing store in parallel once dependencies
  are satisfied. This is cheaper and worked for the evaluated
  symmetric workloads, but the paper shows it can miss opportunities
  on unbalanced dependency graphs.
- Prefetching tracks dependency sets after a request. For a key, DeToX
  stores frequent subsequent key sets and prefetches the most popular
  set, with caps and frequency thresholds to bound metadata.
- The implementation is a Java shim over Redis plus Postgres or TiKV.
  Reads check Redis first; misses go to the backing store and populate
  cache. Writes go to the backing store and update cache before write
  locks are released. The shim uses two-phase locking and timeout-based
  deadlock detection to maintain serializability.
- Redis integration is intentionally small: group-aware scoring is
  attached to multi-get style operations and key scores update after a
  transaction completes. Eviction samples ten candidates and evicts the
  lowest-scored candidate.
- Evaluation uses TAOBench, Epinions, SmallBank, and TPC-C. The paper
  reports up to 1.3x higher transactional hit rate, up to 3.4x better
  cache efficiency, 31% higher throughput and 30% lower latency on a
  Redis/Postgres TAOBench setup, and less than 1-2% extra cache-space
  metadata in the highlighted workloads.
- The paper also shows limits. TPC-C mostly does not benefit because
  transactions mix a small set of already-cached hot keys with many
  cold keys, so the cold keys contaminate the larger transaction. A
  TAOBench product group dominated by point reads and tiny read
  transactions also sees little gain over single-object policies.

**GPU DB mapping:** Transactional hit rate should become a candidate
metric for GPU DB residency and route decisions. Current P8 thinking
tracks resident validity, transfer bytes, predicate support, and memory
pressure. DeToX adds a missing question: does this placement reduce the
critical path of the whole logical request or only make one object
inside the request faster?

For retained lookup batches, group scoring can guide which key vectors,
column groups, and partition generations deserve GPU memory together.
For example, if a session template usually reads customer, latest order,
and several order-line rows, caching only the customer key or only the
order partition may leave the request serialized on CPU/NVMe misses.
The residency owner should be able to score a template-level placement
group and explain that a hot object was rejected because its companion
group is too cold, too large, or invalidated too frequently.

For 1M logical sessions, DeToX's levels approximation maps naturally to
the planned IO-worker and read-snapshot rings. Even without static SQL
dependency extraction, the runtime can observe which keys, predicates,
or route families are issued in parallel under one session transaction
or request envelope. That produces a cheap group signal for cache
admission, prefetch, and micro-batch formation.

The prefetching idea maps to warm resident generations. After a request
hits a high-confidence root key or predicate, the residency owner can
schedule bounded warmup for likely companion partitions or column
families. This must remain behind snapshot and WAL validation: prefetch
can prepare candidate generations, but publication still checks schema,
visibility boundary, invalidation generation, and memory budget.

The paper's hot-key contamination result is especially important for
benchmark design. GPU DB should report object/partition hit rate and
transactional route hit rate separately. A route that shows high GPU
resident object hits but still falls back for one companion key should
not be counted as a low-latency retained success.

**Risks and mismatches:** DeToX targets a key-value cache in front of a
backing store, not a SQL engine with MVCC tuple chains, WAL replay,
catalog generations, GPU kernels, and multi-tier resident placement. Its
shim uses two-phase locking and clears the cache after failures; GPU DB
cannot replace MVCC/WAL correctness with external locks or whole-cache
flushes.

The transaction DAG may be hard to infer for arbitrary SQL, joins,
stored procedures, ad hoc queries, or interleaved pgwire sessions. The
levels approximation is promising, but the paper shows it can lose on
unbalanced dependency graphs. GPU DB should therefore treat runtime
level inference as a first slice, while leaving room for explicit route
templates or planner-derived dependency metadata.

The metric optimizes latency, not backing-store load. DeToX sometimes
has lower object hit rate than object-oriented policies. GPU DB must be
able to choose between lowering request latency and lowering CPU/NVMe
load, especially under memory pressure, write-heavy periods, or
checkpoint/recovery work.

Finally, DeToX's evaluated wins depend on workload shape. The TPC-C
result is a warning: write-heavy or very large cold-tail transactions
may not benefit from transactional cache placement, and prefetching can
waste residency bandwidth if invalidation or cold companion keys dominate.

**Benchmark candidates:**

- Add a `transactional_route_hit_rate` metric beside object/partition
  resident hit rate. For each logical request template, compute whether
  all required retained components were available and shortened the
  slowest path. Failure condition: a benchmark claims retained success
  while one required companion route still falls back to CPU/NVMe.
- Build a TAOBench-inspired retained-read benchmark with correlated key
  groups, contaminated hot keys, writes, and skew. Compare per-object
  LRU/frequency admission against transaction-group admission for GPU
  resident key vectors and column groups. Minimum gate: group admission
  improves p95/p99 or transactional route hit rate without stale reads.
- Prototype levels-based grouping in the runtime: group keys/predicates
  issued in parallel within one request envelope, update placement scores
  after completion, and expose group size, minimum companion frequency,
  critical-path miss reason, and eviction reason.
- Add a companion-prefetch experiment: when a root lookup is admitted,
  schedule warmup for likely companion partitions behind a bounded
  residency queue. Required measurements: prefetch hit rate, wasted
  warmups, invalidated warmups, queue wait, bytes moved, and p99 impact.
- Compare placement objectives under memory pressure: maximize resident
  object hits, maximize transactional route hits, minimize movement
  bytes, and hybrid weighted policy. Failure condition: a policy improves
  hit count while increasing end-to-end p99 or write invalidation stalls.
- Add a TPC-C-like negative-control benchmark where hot warehouse/district
  keys are paired with cold order-line/item keys. Proof gate: the policy
  detects contamination and avoids wasting GPU memory on partial groups
  that cannot shorten the request path.
- Test SQL/planner-derived group metadata against runtime level inference
  for a small set of templates. Required output: scoring overhead,
  grouping accuracy, route-hit improvement, and fallback explanations.

### 2026-06-03 - Themis GPU relational pipeline load balancing

**Citation:** Kijae Hong, Kyoungmin Kim, Young-Koo Lee, Yang-Sae
Moon, Sourav S. Bhowmick, and Wook-Shin Han. "Themis: A
GPU-accelerated Relational Query Execution Engine." PVLDB 18(2),
2024, pp. 426-438. doi:10.14778/3705829.3705856. Retrieved
2026-06-03 from `https://www.vldb.org/pvldb/vol18/p426-han.pdf`.

**Category:** GPU execution / analytics, with runtime load balancing.

**Relevance tags:** fused GPU pipelines; skewed joins; warp load
balance; intra-warp idle ratio; inter-warp load imbalance; lazy
materialization; adaptive work sharing; GMEM contention; retained
route batching; skew benchmarks.

**Core idea:** Themis targets a specific failure mode of GPU
relational execution: fused tuple-at-a-time pipelines can still
underuse the GPU badly when joins, filters, or aggregations produce
non-uniform work per input tuple. Even if the input scan is evenly
partitioned, downstream operators create intra-warp idle lanes and
inter-warp stragglers. Prior systems rebalance with fixed thresholds
or shared global-memory buffers, but those either choose the wrong
granularity for some workloads or add heavy synchronization and
memory traffic.

The paper reframes one GPU query pipeline as traversal of an
evaluation tree. Each tree node is an input tuple to one operator,
and child nodes are that operator's outputs. This gives Themis a
fine-grained way to decide which nodes a warp should visit next and
which subtrees should be transferred from a busy warp to an idle
warp. On skewed JCC-H queries, the paper reports that Themis
substantially reduces both intra-warp and inter-warp imbalance and
outperforms the strongest baseline by up to 379x.

**Concrete mechanisms:**

- Themis represents pipeline execution as an evaluation tree whose
  levels correspond to pipeline operators. Visiting a node means
  evaluating the operator at that level for one tuple or offset range.
- Instead of materializing full intermediate tuples, operators produce
  compact offset ranges over table arrays. Attribute values are loaded
  only when a later predicate, aggregation key, or materialization step
  needs them.
- No-imbalance-first-search (NIFS) chooses the highest operator level
  that has at least one full warp of work. If no level has a full warp,
  it chooses the lowest non-empty level. The paper argues this creates
  only the unavoidable final partial warp per level, like breadth-first
  traversal, while using fixed per-warp memory.
- NIFS stores per-level queues of offset ranges in registers. The memory
  bound is constant in query output size: at most `2 * (k - 1) * wSize`
  offset ranges for a pipeline of `k` operators.
- Adaptive work sharing (AWS) handles inter-warp imbalance. A warp
  periodically estimates whether it is the busiest warp, and if so,
  transfers about half of its highest remaining subtrees to an idle warp.
- Workload size is approximated by the highest subtree height and the
  number of subtrees at that height, with the count bucketed on a log
  scale to reduce global tracking churn.
- The redistribution check interval adapts to the number of idle warps:
  when many warps are idle, busy warps check more frequently; when few
  are idle, they check less often to avoid unnecessary overhead.
- To avoid a single contended global queue, Themis uses a hierarchical
  bitmap to find idle warps and a dedicated global-memory buffer per
  warp. Busy warps atomically claim an idle warp, split offset ranges,
  and write the donated work into that warp's buffer.
- The paper uses two imbalance metrics that are directly portable:
  intra-warp idle ratio (average idle lanes per warp iteration divided
  by warp size) and inter-warp load imbalance factor (maximum warp
  execution cycles divided by average warp execution cycles).
- Evaluation uses JCC-H, a skewed TPC-H variant, at scale factor 30 on
  an RTX 3090 with CUDA 11.4. Themis compares against DogQC++ and a
  Pyper-style implementation, with ablations for NIFS and AWS.
- For JCC-H queries expected to suffer inter-warp imbalance, Themis
  reports up to 173x shorter execution than baselines or the no-AWS
  variant. Average and maximum ILIF for Themis are reported as 1.8 and
  4.4, while baselines in those categories show average ILIFs in the
  hundreds.
- For intra-warp imbalance, the no-AWS Themis variant has about 1%
  average idle ratio across 22 JCC-H queries, while the strongest
  baseline is reported at 14x higher and other baselines can approach
  near-total lane idleness on some cases.
- Lazy materialization reduces Themis query time by up to 38% in the
  reported JCC-H experiments, especially when selections filter many
  scanned tuples before later operators need attributes.

**GPU DB mapping:** Themis is most relevant to retained GPU route
execution once the engine moves beyond simple single-operator counts,
filtered scalar aggregates, and point lookups. The current P8 design
has partition-resident column groups and a future runtime with GPU
execution owners. If retained route families add fused filters, joins,
grouped lookups, or grouped aggregates, the scheduler needs to know
whether a batch is actually keeping lanes and warps busy, not merely
whether it launched one large kernel.

The evaluation-tree abstraction maps to route templates. A retained
GPU template could expose levels such as key gather, predicate filter,
join/probe, projection, aggregation, and response scatter. Within a
micro-batch, the GPU worker can track per-level work queues and choose
whether to run immediate per-request kernels, a NIFS-like fused
traversal, or a simpler vectorized path. This keeps the current
correctness boundary intact: SQL visibility, WAL, and snapshot choice
remain outside the kernel, while intra-kernel work scheduling improves
only the execution of an already valid snapshot generation.

Themis also supplies concrete telemetry missing from many GPU DB
benchmarks. P8 measurements should report not just rows/sec and kernel
time, but idle-lane ratio, straggler warp factor, redistribution count,
GMEM buffer traffic, and whether skew is causing retained route latency
tails. These counters can help distinguish "the GPU route is slow
because data moved too much" from "the route is resident but divergent
and imbalanced."

Lazy materialization maps directly to resident column-group design.
For filtered retained reads, the GPU route should carry row ordinals,
offset ranges, or key-vector positions as long as possible, and only
load projected columns or response bytes after predicates eliminate
rows. This is especially important for text columns and wide rows,
where eager per-row materialization would waste HBM bandwidth and
response staging buffers.

AWS is also a warning for micro-batching. A fixed micro-batch size,
fixed join-output threshold, or fixed morsel size can be wrong under
skew. GPU DB should use adaptive split points driven by observed
output fanout, active warp count, queue wait, and idle-worker pressure.
The equivalent at the runtime level is that GPU workers should not
blindly group by request count; they should group by expected execution
work and split skew-heavy subgroups when they create stragglers.

**Risks and mismatches:** Themis is an analytical query execution
paper, not an OLTP or MVCC system. It does not address WAL-before-
visibility, transaction validation, snapshot publication, catalog
invalidation, network response scheduling, or 1M logical session
admission. Its strongest result comes from skewed TPC-H-style joins,
not point transactions or write-heavy paths.

The implementation assumes generated GPU kernels for relational
pipelines and uses internal offset-range representations. GPU DB's
current retained execution is narrower and may not need a full
evaluation-tree scheduler until fused multi-operator routes exist.
The AWS mechanisms also use global-memory metadata, atomics, and
per-warp buffers; under small batches or latency-sensitive point
lookups, this overhead may be larger than the imbalance it removes.

The paper's public PDF is the PVLDB paper, while the references mention
a fuller artifact through a Google Drive URL. This entry is based on
the PVLDB paper and does not assume details from the separate full
artifact beyond what the paper reports.

**Benchmark candidates:**

- Add GPU execution telemetry counters for retained routes:
  intra-warp idle ratio, inter-warp load imbalance factor, per-level
  active work, redistribution count, GMEM redistribution bytes, and
  straggler kernel reason. Minimum gate: counters are zero or marked
  unsupported for simple kernels rather than silently absent.
- Build a skewed retained aggregate/join microbenchmark inspired by
  JCC-H: one hot key generates many matches while most keys generate
  few or none. Compare fixed input partitioning, fixed morsels, and an
  adaptive work-sharing variant. Failure condition: p99 or ILIF grows
  with skew while total rows/sec hides the straggler.
- Prototype lazy projection for filtered resident reads: carry row
  ordinals through predicate evaluation, then load projected int/text
  columns only for survivors. Required measurements: HBM bytes read,
  D2H bytes, kernel time, response encoding time, and correctness under
  null/empty result cases.
- For same-shape lookup micro-batches, test whether fixed batch size is
  enough or whether batches need work-aware splitting by expected
  fanout. Minimum proof: a hot-key batch does not block unrelated cold
  lookup responses behind one straggler subgroup.
- Add a negative-control latency benchmark where the batch is small and
  uniform. Proof gate: any NIFS/AWS-like machinery must be bypassed or
  show overhead below a configured microsecond budget.
- Extend planner route metadata with a skew-risk flag derived from
  statistics or previous telemetry. If skew risk is high, route choice
  should prefer a load-balanced GPU kernel, CPU fallback, or smaller
  GPU sub-batches depending on latency and residency state.
- Compare eager materialization, lazy materialization, and partial
  materialization for text-heavy resident routes. Failure condition:
  eager materialization consumes HBM/D2H bandwidth for rows filtered
  before projection.

### 2026-06-03 - Cross-paper synthesis: placement and scheduling need request-shaped metrics

Recent reviews since the batch-boundary synthesis add three missing
axes to the runtime/storage target. NOMAD says movement between tiers
should be transactional, non-exclusive when useful, and validated at
publish time. DeToX says cache value should be measured by whether a
whole request or transaction critical path is shortened, not whether a
single object hit in cache. Themis says GPU execution quality should be
measured by lane/warp work balance and skew, not just by resident bytes
or launched kernels.

The converging design track is request-shaped admission. A future GPU
DB route should carry a compact descriptor that names the snapshot
generation, required resident components, expected companion keys or
partitions, estimated tier movement, skew/fanout risk, response shape,
and latency budget. The residency owner can use that descriptor to
decide whether to promote, query a lower tier directly, prefetch a
companion generation, or reject. The GPU execution owner can use the
same descriptor to decide whether fixed micro-batching is enough or
whether the batch needs work-aware splitting.

This also changes benchmark priorities. Object hit rate, resident byte
hit rate, and kernel runtime are insufficient. The next benchmark
bundle should include transactional route hit rate, aborted promotion
rate, shadow/direct-lower-tier hit rate, intra-warp idle ratio,
inter-warp imbalance factor, and p95/p99 request latency under skew
and memory pressure. A policy that improves only one local metric while
increasing end-to-end request latency should be treated as a failure.

The remaining category gap is still optimizer integration for these
runtime signals. The journal has PARQO/PAR2QO-style robust planning
coverage, but it has not yet reviewed a paper that turns live execution
feedback, skew, cache placement, and operator choice into a single
route decision for heterogeneous CPU/GPU/tiered systems. That should
shape one of the next queue selections.

### 2026-06-03 - Mordred semantic CPU/GPU placement

**Citation:** Bobbi W. Yogatama, Weiwei Gong, and Xiangyao Yu.
"Orchestrating Data Placement and Query Execution in Heterogeneous
CPU-GPU DBMS." PVLDB 15(11), 2022, pp. 2491-2503.
doi:10.14778/3551793.3551809. Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol15/p2491-yogatama.pdf`.

**Category:** query optimization / planning, multi-tier cache / data
placement, and GPU execution / analytics.

**Relevance tags:** CPU/GPU route choice; semantic cache admission;
fine-grained placement; segment-level plans; correlated segment
scoring; late materialization; PCIe avoidance; segment skipping;
preallocated scratch regions; over-resident execution.

**Core idea:** Mordred treats limited GPU memory as a coupled
placement and execution problem rather than only a cache-size problem.
The complete database remains in CPU memory, while GPU memory holds a
subset of raw data segments. Instead of caching whole columns by LRU or
LFU, Mordred scores sub-column segments by the estimated runtime
benefit of placing that segment and its correlated companions on the
GPU for the current workload.

The second half of the design is that the executor can use partial
placement. A query plan is converted into segment-level subplans, so
one part of a column or operator may run on the GPU while another part
runs on CPU. This avoids the all-or-nothing behavior of GPU-primary or
coprocessor designs that either require all needed input in HBM or
stream uncached data across PCIe on demand.

**Concrete mechanisms:**

- The cache unit is a fixed-size sub-column segment. Mordred makes the
  segment size user-defined and uses 1,048,576 records, or 2^20
  records, as the default.
- Mordred extends LFU with weighted frequency counters. For a segment
  access, it estimates query runtime without the segment cached and
  with the segment plus correlated segments cached; the counter
  increment is the estimated runtime saved.
- Correlated segments are operator-specific. Selection correlation
  covers segments involved in an inseparable predicate over the same
  rows. Hash join correlation requires the build column to be completely
  cached and distributes benefit from probe segments to the build
  segments. Group-by correlation connects aggregation and grouping
  segments, including cross-table correlation after joins.
- The cost model is lightweight and bandwidth-oriented. It estimates
  filter, join probe, PCIe transfer, materialization, and merge costs
  from memory traffic and selectivity, borrowing Crystal's assumption
  that simple CPU/GPU operators saturate memory bandwidth.
- Operator placement is data-driven at segment granularity. An operator
  runs on GPU for the segment groups whose required inputs are resident;
  other segment groups run on CPU. Filter, join probe, and group-by can
  be split across devices.
- Segment grouping merges adjacent or compatible segments with the same
  physical plan so execution does not launch one tiny kernel per
  segment. In the SSB experiments, segment grouping speeds query
  execution by up to 3x. Grouping and final merge are not reported as
  dominant bottlenecks: for one small-cache case the paper reports
  0.3% grouping, 99.2% execution, and 0.5% merging; for a large-cache
  case it reports 4% grouping, 93.6% execution, and 2.4% merging.
- Late materialization transfers row-id pairs or ordinals across PCIe
  instead of full intermediate tuples. The receiving side reconstructs
  needed columns locally, allowing a GPU join to run with only join keys
  resident while projected columns are materialized later on CPU.
- Operator pipelining fuses consecutive operators on the same device so
  intermediate results are not repeatedly written to and reread from
  memory. On GPU this builds on Crystal's tile-based execution.
- Segment skipping stores min/max metadata per segment and skips entire
  segments when predicates cannot match. Mordred extends this to joins
  by using min/max values from the build-side hash table to prune probe
  segments.
- The cache manager divides GPU memory into a raw-data caching region
  and a data-processing region for hash tables and intermediate results.
  The processing region is preallocated; per-query allocation advances a
  pointer and resets it after query completion. CPU scratch allocation
  uses the same pattern.
- CPU metadata tracks segment min/max values, weighted counters,
  location bitmaps, GPU offsets, and free segment slots. The query
  optimizer consumes this placement metadata to form segment-level
  plans and reorder operators to avoid CPU/GPU ping-pong.
- Evaluation uses SSB scale factor 40 for the fit-in-GPU case and scale
  factor 160 for the larger-than-GPU-memory case on an NVIDIA V100 over
  PCIe3. The paper reports semantic-aware caching outperforming the best
  traditional cache policy by 3x, and reports larger end-to-end speedups
  against several prior CPU/GPU DBMS baselines in its SSB comparisons.

**GPU DB mapping:** The strongest transferable idea is that route
admission should score the whole executable shape, not just resident
bytes or object frequency. For GPU DB, a retained route descriptor can
name the snapshot generation, required column groups, companion keys or
partitions, operator family, expected output width, and movement cost.
Placement counters should increase by estimated request-latency saved
only when enough correlated components are present to shorten the
critical path.

Mordred's correlated-segment scoring maps to P8 resident column groups.
A retained lookup or aggregate often needs a key vector, predicate
column, projected columns, visibility metadata, and response shape. GPU
DB should avoid admitting only the hot key vector if missing projected
columns force owner-thread or CPU fallback. Conversely, when row-id or
ordinal late materialization is safe, the GPU route may need only keys
and predicates resident, with projected values assembled later from CPU
or host-tier state.

Segment-level plans map to partition-aware retained routing. The engine
already has partition-valid retained paths for simple aggregate and
lookup shapes. The next planner contract can choose per partition:
execute resident GPU, execute CPU, transfer cold segment, or skip. A
single SQL request can then merge per-partition results deterministically
without pretending that the entire table has one residency state.

The paper also reinforces an explicit tier-boundary optimizer. PCIe
traffic, HBM traffic, CPU memory traffic, merge cost, and queue wait
should be planner-visible. As future tiers appear, the same score can
generalize from CPU/GPU to GPU HBM, pinned host DRAM, CPU DRAM, CXL-like
remote memory, NVMe, and GPUDirect paths. The planner should change
behavior when interconnect economics change rather than baking in a
single "GPU if resident" rule.

Preallocated scratch regions fit the high-throughput runtime target.
GPU execution owners should own scratch arenas, pinned staging buffers,
and per-stream allocation cursors, then reset them at route or batch
boundaries. That preserves low allocation overhead while keeping
correctness metadata outside scratch state.

**Risks and mismatches:** Mordred is an analytical engine, not an OLTP
or MVCC storage engine. It does not address WAL-before-visibility,
transaction validation, snapshot publication, long-reader retention,
write invalidation, recovery, or 1M logical sessions. Its cache
replacement policy assumes the full database is already in CPU memory
and that GPU memory is an acceleration cache for analytical scans and
joins.

The cost model is intentionally lightweight and bandwidth-oriented. It
may not predict latency-sensitive point lookups, queueing delay,
contention, snapshot-reference retention, or response encoding. GPU DB
should use it as a starting feature set for route scoring, not as a
complete optimizer.

Fine-grained segment plans introduce merge and coordination costs. The
paper reports these as small in its SSB setup, but GPU DB's small
request batches and session-heavy workload may make plan splitting
overhead visible at p50/p99. Any implementation must include a negative
control where uniform resident point lookups bypass split-plan
machinery.

**Benchmark candidates:**

- Add a `route_value_score` experiment for retained planning. Compare
  object-frequency admission, byte-frequency admission, and
  semantic/request-shape admission where the score is estimated latency
  saved if all required companions are resident. Minimum gate: higher
  score improves p95/p99 request latency, not just resident byte hit
  rate.
- Build a mixed-residency partition query benchmark: some partitions are
  valid GPU resident, some CPU-only, and some skippable by min/max.
  Compare all-CPU, transfer-to-GPU, and segment-level CPU/GPU execution
  with deterministic merge. Failure condition: partial placement
  produces stale reads or hides fallback reason.
- Prototype ordinal late materialization for retained filters. GPU
  kernels return row ordinals and aggregate/group intermediates; CPU or
  host-tier code materializes wide/text projections only for survivors.
  Required measurements: HBM bytes, PCIe/D2H bytes, CPU materialization
  time, response bytes, and null/empty correctness.
- Add correlated-component telemetry to residency: required components,
  missing companions, partial-route hit, full-route hit, and reason why a
  resident component could not shorten the request. This should be
  reported per route template and partition.
- Test segment skipping as a first-class route decision for retained
  partition metadata. Proof gate: min/max pruning reduces CPU/GPU bytes
  and p99 without changing SQL results across inclusive range,
  no-match, and invalidated-partition cases.
- Add preallocated scratch arenas to a GPU execution-owner prototype and
  compare against per-request allocation. Minimum gate: allocation
  timing disappears from hot retained paths while scratch exhaustion
  yields explicit overload rather than undefined reuse.
- Create a sensitivity benchmark for interconnect economics: replay the
  same route decisions with measured PCIe bandwidth and simulated
  NVLink/CXL/GPUDirect bandwidth. The planner should explain when the
  best route changes from CPU, to partial CPU/GPU, to resident GPU.
- Negative control: uniform resident point lookups with all required
  components present. Any semantic-placement or split-plan machinery
  must either be bypassed or stay under a small microsecond latency
  overhead threshold.

### 2026-06-03 - Morty transaction re-execution

**Citation:** Matthew Burke, Florian Suri-Payer, Jeffrey Helt,
Lorenzo Alvisi, and Natacha Crooks. "Morty: Scaling Concurrency
Control with Re-Execution." EuroSys 2023, pp. 687-702.
doi:10.1145/3552326.3567500. Retrieved 2026-06-03 from the
author PDF, `https://www.cs.cornell.edu/~matthelb/papers/morty-eurosys23.pdf`.

**Category:** transaction processing / write path and MVCC /
visibility.

**Relevance tags:** serializable transactions; contention
management; transaction re-execution; MVTSO; speculative ordering;
interactive transactions; commit validation; write hot spots; partial
retry; long conflict windows.

**Core idea:** Morty argues that high-contention serializable systems
lose throughput because conflicting transactions create serialization
windows that overlap, then conventional OCC or 2PL turns that overlap
into aborts, backoff, lock waiting, or idle CPU time. Its response is
not to guess a better retry delay. Morty gives transactions a
speculative timestamp order, exposes writes early enough for replicas
to detect missed writes, and partially re-executes the affected
transaction continuation so the read shifts forward to the newer
write. The goal is to align contending windows back-to-back without
discarding the whole transaction.

The evaluation is a replicated key-value transaction system, not a
single-node database engine. Still, the contention lesson is directly
useful: if the GPU DB write path eventually supports read-modify-write
transactions, it should measure how much work is wasted by whole
transaction retry before assuming abort/retry is acceptable. Morty
reports that on TPC-C with 100 warehouses it reaches 11.8k committed
transactions/sec in the regional setup, compared with 6.8k for its
replicated MVTSO baseline, 2.7k for TAPIR, and 1.6k for Spanner. On a
highly contended Retwis workload it reports much larger relative gains
and a near-perfect commit rate under increasing skew.

**Concrete mechanisms:**

- Morty defines a serialization window for a transaction's access to
  object `x`: it starts at the write version of `x` that the transaction
  read and ends when the transaction's own write to `x` becomes visible.
  Serializability requires these windows not to overlap for committed
  conflicting writers.
- It also defines validity windows for read validity, then frames
  throughput under contention as a function of how long those windows
  are and how much idle time exists between them.
- Each transaction receives a version from a loosely synchronized
  timestamp plus coordinator id. That version is the speculative total
  order used by MVTSO-style reads, writes, and validation.
- Reads return the newest write version smaller than the transaction's
  version. Replicas remember uncommitted reads and the last write version
  returned for each read.
- Writes are broadcast asynchronously. When a replica receives a write,
  it checks whether an earlier read with a larger transaction version
  missed that write. If so, the replica sends a new read reply to the
  transaction coordinator.
- Re-execution uses a continuation-passing API. The client library keeps
  transaction contexts and continuations for the current execution, then
  replays only the affected continuation with the newer read value when a
  missed write reply arrives.
- Coordinators track read execution history so stale replies do not
  re-execute a branch that the transaction has already moved past or
  abandoned.
- Commit operates at execution granularity. A transaction may have
  multiple executions; one execution can commit, while older executions
  are abandoned rather than treated as committed transaction decisions.
- Prepare validation checks missed reads, other transactions' missed
  reads of this transaction's writes, dirty reads, and whether needed
  committed metadata was already truncated.
- Replica votes distinguish `Commit`, `Abandon-Tentative`, and
  `Abandon-Final`. Depending on quorum agreement, the coordinator can
  skip or run a finalize phase to make the execution decision durable.
- Decide logs committed read/write metadata for future validation and
  removes prepared metadata for abandoned executions. If the whole
  transaction aborts, replicas send new replies to reads that had
  observed its writes.
- Coordinator recovery uses a Paxos-like view-change path for stalled
  execution decisions, preventing a failed coordinator from blocking
  conflicting transactions indefinitely.
- Garbage collection deletes uncommitted read/write metadata after
  transaction decision, and truncation chooses a safe version below which
  execution and committed conflict metadata can be removed.
- The paper's strongest reported high-contention result is that Morty
  can use extra cores for re-execution and reply generation while OCC
  and 2PL baselines leave CPUs mostly idle due to abort/backoff or lock
  waiting.

**GPU DB mapping:** The immediate transfer is a contention metric:
measure conflict-window length and wasted prepared work, not only abort
count. A GPU DB mutation owner should know when a hot row, account,
warehouse, queue head, or catalog item is causing read-write windows to
overlap, how much WAL/index/resident-refresh preparation was discarded,
and whether the conflict was local to one predicate or touched the
whole transaction.

Morty's re-execution shape complements the earlier MV3C transaction
repair entry. MV3C focuses on dependency graphs over predicates; Morty
shows a runtime path for shifting reads forward using retained
continuations and speculative order. For GPU DB, a practical first
version would not expose a broad CPS API to arbitrary SQL clients. It
could instead target internal or stored-procedure-like transaction
fragments where the engine can name dependencies: read hot row, compute
derived writes, append WAL batch, update value indexes, and invalidate
or refresh resident generations.

The speculative timestamp order maps to owner-domain sequencing.
Partition owners can assign monotonic transaction or batch generations
before execution, then use them to identify when a later-arriving write
should force a dependent read fragment to re-run rather than forcing the
entire client transaction through parse, admission, WAL preparation, and
route planning again. This should remain subordinate to
WAL-before-visibility: early values can be used for speculative
re-execution only if commit validation and durable publication still
prevent uncommitted or abandoned writes from becoming externally
visible.

For retained GPU snapshots, the interesting path is conflict-local
refresh repair. If a long read-modify-write or maintenance transaction
builds a candidate row vector or resident invalidation plan, a conflict
on one hot key should not automatically discard unrelated candidate
work. A future route descriptor could record which read values,
partitions, and column families derived each write or refresh fragment,
then re-run only the affected fragment when the owner detects a missed
write.

**Risks and mismatches:** Morty is a replicated key-value store for
interactive transactions, not a PostgreSQL-compatible SQL engine. Its
API assumes continuation-passing transaction code and stored contexts;
ordinary pgwire SQL statements do not naturally expose that structure.
It uses early uncommitted write visibility internally, which is
dangerous unless validation, dirty-read checks, abandoned-execution
cleanup, WAL ordering, and recovery semantics are all precise. The
paper does not address GPU residency, SQL planning, DDL invalidation,
response rings, or million-session admission. Its evaluation is
distributed and contention-heavy, so its throughput numbers should not
be projected onto local GPU execution. Finally, repeated partial
re-execution can burn CPU under extreme skew if the engine does not cap
attempts or switch to an ordered hot-key path.

**Benchmark candidates:**

- Add mutation-owner conflict-window telemetry for one read-modify-write
  microbenchmark: first read time, conflicting write publish time,
  validation time, abort/retry time, and bytes or fragments of prepared
  WAL/index/resident work discarded. Minimum gate: no behavior change and
  explicit attribution by table, key, column family, and owner.
- Build a stored-procedure-only partial retry proof with two independent
  fragments and one hot account or warehouse row. Re-run only the
  fragment whose read missed a newer write, preserve WAL-before-visibility,
  and compare against full abort/retry at concurrency `1,2,4,8,16,32,64`.
- Add a hot-key policy switch benchmark: optimistic retry, partial
  re-execution, and deterministic owner-queue ordering. Failure condition:
  partial re-execution burns more CPU or tail latency than ordered
  execution once conflict-window overlap is continuous.
- Test dependency-tagged resident invalidation planning. Prepare refresh
  or invalidation fragments for several partitions, inject a conflict in
  one partition, and verify that unaffected fragments are reused only when
  their source WAL/catalog/visibility generation remains valid.
- Add safety tests for speculative/internal early values: no abandoned or
  uncommitted execution may become externally visible, survive recovery as
  committed state, or make a retained GPU snapshot valid.
- Track re-execution attempt count, context bytes retained, continuation
  memory, and age of speculative executions. Proof gate: bounded memory
  and an explicit fallback to abort/retry or ordered execution under
  pathological skew.

### 2026-06-03 - LOGER restricted learned query optimization

**Citation:** Tianyi Chen, Jun Gao, Hedui Chen, and Yaofeng Tu.
"LOGER: A Learned Optimizer towards Generating Efficient and Robust
Query Execution Plans." PVLDB 16(7), 2023, pp. 1777-1789.
doi:10.14778/3587136.3587150. Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol16/p1777-gao.pdf`.

**Category:** query optimization / planning.

**Relevance tags:** learned optimizer; robust route choice; bounded
planner hints; join order; physical operator restrictions; beam search;
cost-model uncertainty; CPU/GPU route selection; planner guardrails.

**Core idea:** LOGER argues that learned query optimizers are most
practical when they exploit the existing DBMS optimizer rather than
replace all of its operator knowledge. It uses deep reinforcement
learning to search join order plus per-join operator restrictions, but
lets the DBMS optimizer choose the actual physical operator inside
those restrictions. That gives the learned model a larger search space
than global hint selection, while avoiding the brittle behavior of
directly choosing every physical operator.

The mechanism matters for GPU DB because CPU/GPU/tier route choice has
the same shape. A learned component should not directly overrule
visibility, residency, WAL, overload, or fallback rules. It can instead
learn bounded route restrictions or preferences: avoid GPU for this
template under this queue/tier state, avoid CPU fallback for this
resident aggregate shape, require prefilter before GPU transfer, or
disable an over-resident route when selectivity and transfer risk are
high.

**Concrete mechanisms:**

- Queries are represented as join graphs. Tables are nodes, join
  predicates are edges, and node attributes include learned table
  embeddings, column statistics, indexes, predicate selectivity, and
  inverse-predicate selectivity.
- A Graph Transformer exchanges table and predicate information across
  the join graph. LOGER then uses Tree-LSTM state representations for
  partial join trees during plan search.
- Restricted Operator Search Space (ROSS) changes the action space.
  Instead of selecting a physical join operator directly, an action
  selects a join plus one of four restrictions: no restriction, no
  nested-loop join, no merge join, or no hash join. The DBMS optimizer
  chooses the final physical operator under that restriction.
- Candidate joins are enumerated in a System R-like way that avoids
  Cartesian products when conditional joins are available.
- `epsilon`-beam search keeps multiple search paths. Some paths exploit
  the value model's top candidates, while selected exploration paths
  sample promising alternatives; exploration probability decays but is
  increased for poorly optimized or long-running queries.
- The experience dataset records the best observed reachable relative
  latency for state-action pairs, not just the most recent result.
  Initial expert plans from the DBMS optimizer seed the dataset to
  reduce cold-start failures.
- Reward weighting combines operator-relevant latency with
  operator-irrelevant latency for the same join-order state. This
  reduces the chance that a bad previous operator teaches the model to
  reject a good later join action.
- A log transform compresses disastrous-plan rewards so model training
  pays more attention to distinguishing good plans instead of fitting
  huge outliers.
- The evaluation is on SPJ workloads in PostgreSQL 13.5, an anonymous
  commercial DBMS, JOB, TPC-DS subset, and Stack Overflow. LOGER reports
  2.076x total speedup over PostgreSQL on the JOB test workload after
  full training, with average inference times around tens of
  milliseconds in the reported workloads. The paper also reports that
  Balsa and RTOS failed to finish Stack training within the configured
  time in its comparison.

**GPU DB mapping:** LOGER is a strong argument for a bounded learned
route advisor. GPU DB should keep deterministic planner and runtime
guards as authority: visibility compatibility, resident snapshot
generation, WAL-before-visibility, invalidation state, memory budgets,
queue saturation, and CPU fallback safety. A learned model can then
choose among safe restrictions, such as "no GPU cold transfer," "no
over-resident route without CPU prefilter," "no wide projection on GPU
unless ordinal late materialization is available," or "no batching past
this latency budget."

ROSS maps to CPU/GPU operator families. Instead of requiring a model to
select exact kernels, streams, tiers, prefetch choices, and CPU fallback
paths, the model can learn which route families to disable under a
request descriptor. The existing planner still makes the final route
choice among valid survivors using measured costs and hard invariants.

The reward-weighting idea maps to route telemetry. A bad early choice,
such as missing a required resident companion column or choosing a
fragile over-resident transfer path, should not poison learning about
later route decisions. Route training data should separate
template-level shape quality from incidental queue wait, one saturated
tier, or one missing placement component.

The exploration mechanism also fits the hardware-transition problem.
During early GPU DB development, exact benchmark evidence is expensive
and hardware economics may change with the newer GPU. A route learner
should explore only inside explicitly bounded safe alternatives, record
observed route outcomes, and decay exploration once the route template
has enough stable evidence.

**Risks and mismatches:** LOGER targets select-project-join analytical
queries, not OLTP write paths, MVCC validation, PostgreSQL protocol
serving, or GPU execution. It learns from executed plan latency, so
training cost can be high and unsafe if poor plans are allowed to run
unbounded in production. Its inference times are acceptable for its
workloads but too high for many point-lookups unless route decisions
are cached by template, snapshot class, and tier state. The paper does
not solve cardinality estimation, live queue-delay prediction, memory
pressure, or stale resident snapshots. It also assumes planner hints can
express useful restrictions; GPU DB would need a local route-restriction
language before a LOGER-like method can be tested.

**Benchmark candidates:**

- Add a deterministic route-restriction layer before any learned model:
  flags such as `no_gpu_cold_transfer`, `no_cpu_fallback`,
  `require_resident_snapshot`, `require_cpu_prefilter`, and
  `disable_microbatch`. Minimum gate: each restriction is enforced by
  planner tests and reports an explicit fallback or rejection reason.
- Build an offline route-replay dataset from existing retained and
  over-resident telemetry: query template, snapshot generation,
  resident components, missing companions, queue wait, H2D/D2H bytes,
  CPU filter time, kernel time, response bytes, and chosen route.
  Failure condition: the dataset cannot distinguish planner error from
  transient saturation.
- Compare hard-rule planning, global route hints, and
  ROSS-style per-route-family restrictions for repeated retained
  aggregates and lookups. Expected improvement: fewer fragile GPU route
  choices under uncertain selectivity without weakening correctness.
- Add a negative-control p50 latency test where route-cache hits bypass
  model inference entirely. Learned route advice must not add
  millisecond-scale inference to hot point lookups.
- Use reward weighting in route learning experiments: separate
  template/shape latency from incidental queue or tier-saturation
  latency, and verify that one overloaded GPU interval does not cause a
  permanently bad route preference.
- Test exploration only in shadow mode first. The planner chooses the
  production route, while the advisor logs the alternative safe
  restriction it would have chosen and estimates regret from observed
  telemetry. Promote to active only if regret and safety gates pass.

### 2026-06-03 - Cross-paper synthesis: learned advice needs hard route boundaries

Mordred, Morty, and LOGER converge on the same operational lesson:
optimization should happen over named, bounded units of work. Mordred's
unit is a segment-level executable route with correlated placement.
Morty's unit is a transaction continuation or fragment that can be
re-executed without throwing away unrelated work. LOGER's unit is a
planner action that restricts an unsafe or weak operator family while
leaving the DBMS optimizer in charge of final physical selection.

For GPU DB, the emerging design track is a route descriptor plus a
restriction language. The descriptor names snapshot generation, owner
domain, partition, resident components, required companion columns,
expected transfer, queue budget, skew risk, write/read dependency
fragments, and response shape. The restriction language names what the
planner is not allowed to do for that request or template. Learning can
operate over those restrictions, but only after the deterministic
planner has declared which routes are semantically valid.

This keeps learned systems away from correctness authority. A model may
learn that a CPU prefilter should be required before GPU execution, that
a cold transfer is too risky under current PCIe pressure, or that a hot
transaction fragment should be ordered rather than retried. It may not
declare a stale resident snapshot valid, skip WAL-before-visibility, or
reuse a buffer before the owning domain releases it.

The next benchmark priority is therefore not "train a model." It is to
record enough structured route outcomes that a model could be audited:
route-template id, restriction set, chosen route, rejected safe
alternatives, queue/tier state, correctness generation, latency,
throughput, and fallback reason. Category gaps remain around learned
optimizer diagnostics and learned concurrency-control policy, but those
should be reviewed with the same question: what bounded action can be
learned without surrendering invariants?

### 2026-06-03 - Shinjuku microsecond-scale preemptive scheduling

**Citation:** Kostis Kaffes, Timothy Chong, Jack Tigar Humphries,
Adam Belay, David Mazieres, and Christos Kozyrakis. "Shinjuku:
Preemptive Scheduling for microsecond-scale Tail Latency." NSDI 2019,
pp. 345-360. Retrieved 2026-06-03 from the USENIX publication page and
PDF, `https://www.usenix.org/conference/nsdi19/presentation/kaffes`
and `https://www.usenix.org/system/files/nsdi19-kaffes.pdf`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** microsecond scheduling; tail latency; preemption;
centralized dispatch; request-class queues; head-of-line blocking;
network/runtime split; low-overhead context switch; mixed point/range
queries; session admission.

**Core idea:** Shinjuku argues that low-latency datacenter runtimes
should not assume run-to-completion workers are enough. IX-style
distributed first-come-first-served queues and ZygOS-style work
stealing do well when service times are uniform, but they let short
requests sit behind long requests when the workload is bimodal,
heavy-tailed, or mixed. Shinjuku separates network processing from
request scheduling, funnels request work through centralized
dispatcher state, and uses virtualization-assisted interrupts to
preempt running request contexts every 5 to 15 microseconds when
needed.

The database-relevant result is the RocksDB experiment. With a 99.5%
GET and 0.5% SCAN(1000) mix, Shinjuku reports up to 6.6x higher
throughput and 88% lower tail latency than ZygOS at the same target.
The transfer is not the exact OS design; it is the scheduling lesson:
short retained reads must not be trapped behind long scans, refreshes,
COPY work, mutation batches, or over-resident transfers just because
they arrived on the same worker, connection, owner queue, or GPU stream.

**Concrete mechanisms:**

- Network protocol work identifies request boundaries, then passes
  requests to one or more dispatcher threads. Workers execute
  application request contexts; network replies may be handled by the
  networking subsystem or workers.
- The simplest policy uses one centralized queue. Requests that exceed
  a configured quantum are preempted if queued work exists, then placed
  back at the head or tail depending on whether the workload should
  approximate centralized FCFS or processor sharing.
- The multi-queue policy keeps one queue per request type, each with a
  target tail-latency SLO. Queue choice uses the ratio of head request
  waiting time to that queue's SLO, so short-SLO work is favored early
  while long-SLO work eventually ages into service.
- Preemption is implemented with Dune/x86 virtualization support and
  optimized inter-processor interrupt delivery. The paper reports 298
  cycles sender overhead and 1,212 cycles receiver overhead for its
  optimized interrupt path.
- Worker context switches avoid expensive signal-mask and unnecessary
  floating-point save/restore work, reducing context-switch cost to
  tens of cycles in the measured cases.
- Dispatcher/worker communication uses shared cache-line pairs rather
  than general queues. The paper reports about 211 cycles round-trip
  message latency and estimates a high request-rate ceiling for the
  dispatcher's minimal pointer-passing work.
- A single dispatcher is reported to schedule about 5M requests/sec
  across one socket in the synthetic stress test; two dispatchers reach
  about 9.5M requests/sec across two sockets.
- Shinjuku reduces dependence on high connection counts. RSS only needs
  to distribute traffic across dispatchers, not across every worker, so
  a single-dispatcher configuration can operate efficiently with very
  few client flows.
- Preemption is deliberately disabled around non-thread-safe code and
  allocation paths in the prototype; the paper notes that long time
  spent in such sections can still harm tail latency.

**GPU DB mapping:** This is a direct follow-up to the current
`11-high-throughput-query-runtime.md` target. The runtime should
classify requests by shape and tail-latency budget before they enter a
shared execution bottleneck. Retained point lookups, small aggregates,
COPY chunks, mutation work, refreshes, over-resident scans, CPU
fallbacks, and response encoding should not all share one FIFO path
unless the system can prove their service-time distribution is
compatible.

The multi-queue policy maps naturally to route descriptors. A retained
`COUNT(*)` or key lookup can carry a short SLO and a small queue budget;
refresh, scan, and COPY work can carry larger budgets. Queue selection
can then age long work without letting it monopolize short reads. This
does not require arbitrary SQL preemption at first. The first
benchmarkable version can preempt at cooperative boundaries already
visible to GPU DB: before launching a kernel, between over-resident
chunks, between COPY chunks, before response encoding, or between
resident refresh partitions.

For GPU execution, Shinjuku warns against one non-preemptive stream or
owner queue for mixed service times. GPU DB probably cannot interrupt a
running CUDA kernel cheaply, so the production analog is smaller
bounded chunks, separate queues by route class, latency ceilings for
micro-batches, and explicit yield points before long transfers or
multi-partition scans. The dispatcher lesson also supports keeping
network IO workers separate from request scheduling and engine owners:
network readiness should not be the scheduling policy for database
work.

The connection-count result matters for the 1M logical-session target.
The engine should not depend on RSS-style even distribution across many
active flows. Idle or low-rate sessions should be multiplexed through
compact IO state, then admitted into a much smaller set of active
runtime queues with credits and route-aware scheduling.

**Risks and mismatches:** Shinjuku is a single-address-space OS
prototype, not a PostgreSQL-compatible database runtime. It relies on
Dune, x86 virtualization support, optimized IPIs, and kernel-bypass
style networking assumptions that the current GPU DB benchmark endpoint
does not use. Its preemption applies to CPU request contexts, not
in-flight CUDA kernels, WAL fsyncs, blocking disk IO, or arbitrary SQL
operators holding database locks. Disabling interrupts around unsafe
code and allocation is a warning: any GPU DB critical section that
cannot yield may dominate p99 even if the scheduler is otherwise good.
The RocksDB workload is a key-value GET/SCAN mix in memory, so the
absolute numbers should not be projected onto SQL, MVCC, or GPU
execution. The safe transfer is queue discipline and cooperative
preemption boundaries, not the full OS mechanism.

**Benchmark candidates:**

- Add route-class queue telemetry to the pgwire endpoint: retained
  point read, retained aggregate, COPY chunk, mutation, refresh,
  over-resident scan, CPU fallback, and response encode. Minimum gate:
  no behavior change, with p50/p95/p99 queue wait broken out by class.
- Build a two-class retained benchmark: 99.5% short point reads mixed
  with 0.5% long resident or over-resident scans. Compare single FIFO,
  class queues without preemption, and class queues with cooperative
  yield between scan chunks. Failure condition: short-read p99 remains
  dominated by long scan service time.
- Add latency-budget fields to route descriptors and enforce a
  micro-batch ceiling per class. Expected improvement: same-shape
  batches still form under load, but short retained reads do not wait
  behind refresh or scan batches beyond their budget.
- Test cooperative GPU yield points by splitting one long scan or
  refresh into bounded chunks and rechecking queue pressure between
  chunks. Proof gate: identical SQL-visible result and explicit
  accounting for extra launch/transfer overhead.
- Run a session-distribution memory probe where many logical sessions
  map to a small number of dispatch/admission queues. Required metrics:
  bytes per idle session, active queue depth, route-class credits, and
  overload reason when active work exceeds budget.
- Negative control: homogeneous short retained lookups. Class queues and
  scheduling metadata must not add measurable p50 regression compared
  with the simple hot path when service times are uniform.

### 2026-06-03 - Path to GPU-Initiated I/O for Data-Intensive Systems

**Citation:** Karl B. Torp, Simon A. F. Lund, and Pinar Tozun.
"Path to GPU-Initiated I/O for Data-Intensive Systems." DaMoN 2025,
article 3, pp. 1-9. DOI: `10.1145/3736227.3736232`. Retrieved
2026-06-03 from the IT University of Copenhagen publication record,
`https://pure.itu.dk/en/publications/path-to-gpu-initiated-io-for-data-intensive-systems/`,
and the authors' DaMoN slide deck,
`https://itu-dasyalab.github.io/RAD/talk/files/2025_DaMoN.pdf`. The
ACM PDF endpoint was Cloudflare-blocked during this run, so details
below rely on the open metadata, abstract, and slide-deck evaluation
summary rather than the full proceedings PDF.

**Category:** multi-tier cache / data placement and GPU execution /
analytics.

**Relevance tags:** GPU-initiated IO; NVMe SSDs; GPUDirect Storage;
BaM; SPDK; over-resident execution; CPU/GPU resource management;
storage path design; future GPU memory tiers.

**Core idea:** The paper surveys the current GPU-centric storage access
landscape and then compares BaM, a GPU-initiated storage path, against
SPDK, a CPU-centric kernel-bypass storage interface, with GDS as an
additional reference point. The main result is not "always let the GPU
drive storage." It is more conditional: BaM can match SPDK bandwidth
without putting the CPU on the IO path, but it does so by consuming a
large amount of GPU resource. The paper frames GPU-initiated IO as a
resource-placement decision, not just a faster transfer primitive.

That distinction matters for GPU DB because the engine is trying to
combine low-latency retained reads, over-resident scans, mutations,
refresh, and high logical session count. If storage control work burns
SM cycles or cache capacity that would otherwise run relational kernels,
GPU-initiated IO can make query latency worse even while removing CPU
copy or orchestration overhead. The transferable idea is to treat
storage initiation as a route attribute with measured CPU cost, GPU
cost, PCIe traffic, reuse, and queue impact.

**Concrete mechanisms:**

- The paper's technology recap separates GPUfs and ActivePointers as
  POSIX-like GPU-facing APIs, GDS as CPU-initiated direct storage-to-GPU
  transfer, BaM as GPU-initiated direct access with CPU mostly used for
  setup, and GMT as GPU-initiated access through a GPU/CPU/storage
  cache hierarchy.
- GDS removes the CPU memory copy from the data plane but still leaves
  the CPU initiating storage operations. The slide-deck summary says
  this can remain CPU-bound.
- BaM bypasses the CPU on the storage access path and lets GPU code
  initiate reads, but the evaluation summary emphasizes the cost:
  storage access can saturate the GPU.
- GMT is presented as a three-tier cache across GPU, CPU, and storage.
  The summary says it is attractive when reuse is high, but it spends
  both CPU and GPU resources.
- The evaluation compares random reads on a Gigabyte G292-Z20 with an
  AMD EPYC 7402P, 256 GB DDR4, two NVIDIA Tesla V100 16 GB PCIe Gen 3
  GPUs, and four Samsung 980 PRO 1 TB SSDs on Ubuntu 20.04, NVIDIA
  driver 550, CUDA 12.6, BaM from GitHub master, matching GDS, and
  SPDK v24.09.
- The BaM workload uses `nvm-block-bench`, GDS uses `gdsio`, and SPDK
  uses `bdevperf`. The slide deck reports five repetitions with mean
  and standard deviation and uses NVIDIA `dcgmi` for GPU PCIe traffic.
- In the summarized bandwidth results, BaM is comparable to SPDK and
  better than GDS in the tested setup, but is capped by GPU PCIe Gen 3.
  With four drives, BaM scaling falls off, while one to three drives
  scale linearly in the slide summary.
- For 4 KiB IO, the deck attributes a resource-consumption difference:
  BaM fully saturates the GPU, while SPDK needs a single physical CPU
  core.
- The paper's discussion calls out two adoption needs for real systems:
  better CPU/GPU resource management and the right abstraction. It
  contrasts file abstractions for GPUfs, ActivePointers, and GDS with
  block or array abstractions for BaM and GMT.

**GPU DB mapping:** This paper sharpens the over-resident P8 design
choice. A GPU-initiated read path should not be a default replacement
for CPU prefiltering, explicit residency, or CPU-managed NVMe staging.
It should be one candidate route for cold or warm partitions whose
operator pipeline can overlap IO and compute without starving retained
queries on the same GPU.

The current runtime document already separates GPU execution owners,
residency owners, and bounded queues. Torp et al. add another budget
that should be explicit in those owners: storage-control occupancy on
the GPU. A route descriptor for over-resident work should report
whether storage operations are CPU-initiated GDS/SPDK-style, GPU-
initiated BaM-style, or tier-cache mediated, and it should charge that
choice against GPU queue time, SM occupancy, PCIe bytes, CPU core time,
pinned buffers, and expected reuse.

For P8 data placement, the GMT summary reinforces a tiering rule:
promotion to CPU or GPU cache is worthwhile only when reuse repays the
resource burn. Hot retained snapshots should stay as explicit GPU
resident column groups. Warm over-resident partitions may use CPU DRAM
or compressed host segments plus GDS/SPDK-style staging. BaM-like direct
NVMe reads are more promising for large streaming or pointer-chasing
data paths where CPU initiation overhead dominates and the relational
kernel has enough slack to hide IO control work.

The file-versus-array abstraction split is also useful. PostgreSQL-like
tables should not expose files to query kernels as the first design.
The engine should expose relation/partition/page arrays with stable
schema generation, WAL boundary, visibility boundary, compression
format, and row-id mapping. File-like APIs may be useful for external
formats, but MVCC-safe retained and over-resident routes need typed
database blocks.

**Risks and mismatches:** The accessible material is a publication
record and slide deck, not the full paper text. The exact experimental
graphs, variance, implementation details, and full limitations could
not be inspected from the ACM PDF in this run. The evaluation uses V100
PCIe Gen 3 GPUs and Samsung consumer NVMe SSDs, so bandwidth ceilings
and GPU saturation behavior may differ on newer hardware. Random-read
microbenchmarks do not prove SQL query performance, MVCC visibility,
WAL safety, DDL invalidation, response-ring behavior, or session-scale
latency. BaM-style GPU initiation can also conflict with GPU DB's most
valuable retained-read kernels if both need SMs at the same time.

**Benchmark candidates:**

- Add an over-resident IO route descriptor with explicit initiation
  mode: CPU+SPDK-like, CPU+GDS-like, GPU-initiated, or tier-cache
  mediated. Minimum gate: route telemetry records CPU time, GPU elapsed
  time, H2D/storage bytes, pinned-buffer use, queue wait, and fallback
  reason without changing SQL results.
- Build a synthetic over-resident partition scan that compares CPU
  prefilter plus GPU tail against GPU-direct-style staging when data is
  larger than GPU memory. Expected improvement: direct GPU IO only wins
  when CPU initiation or copy overhead is measured as the bottleneck.
  Failure condition: retained-read p99 regresses because storage-control
  work occupies the GPU.
- Add a resource-isolation benchmark mixing 99% retained point reads
  with 1% over-resident cold scans. Compare no cold scans, CPU-managed
  staging, and GPU-initiated staging. Proof gate: cold scans expose
  their resource budget and cannot silently starve retained lookups.
- Prototype a reuse-sensitive warm-tier policy for one partitioned
  table: keep hot columns in GPU memory, warm compressed segments in
  host memory, and cold blocks on NVMe. Measure promotion hit rate,
  demotion churn, bytes moved, and route latency by tier.
- Treat GPU-initiated IO as a later-stage P8 benchmark after retained
  snapshots and CPU-prefiltered over-resident routes have telemetry.
  The first go/no-go question should be whether GPU storage-control
  occupancy is lower than the query-kernel time it helps hide.

### 2026-06-03 - Aria deterministic OLTP batches

**Citation:** Yi Lu, Xiangyao Yu, Lei Cao, and Samuel Madden.
"Aria: A Fast and Practical Deterministic OLTP Database." PVLDB
13(11), 2020, pp. 2047-2060. DOI: `10.14778/3407790.3407808`.
Retrieved 2026-06-03 from the VLDB PDF,
`https://www.vldb.org/pvldb/vol13/p2047-lu.pdf`.

**Category:** transaction processing / write path and concurrency
control.

**Relevance tags:** deterministic OLTP; batch execution; snapshot
execution; read/write reservation; conflict detection; deterministic
reordering; fallback scheduling; partition owners; replication by input;
mutation batching.

**Core idea:** Aria revisits deterministic transaction processing without
requiring every transaction's read and write set before execution. It
processes an ordered batch in two phases. First, all transactions execute
against the same database snapshot in parallel and record local read and
write sets. Then a deterministic commit phase uses reservation metadata
to decide which transactions can commit without violating serializability.
Transactions that conflict are retried in the next batch or, under high
abort rates, rerun through a deterministic fallback path.

The useful transfer for GPU DB is not full deterministic replication. It
is the separation between speculative same-snapshot work and a small
deterministic publish decision. A mutation or refresh owner could let
many candidate updates, index fragments, or resident invalidation plans
run in parallel, then publish only those whose read/write reservations
are compatible with the chosen WAL and visibility boundary.

**Concrete mechanisms:**

- A sequencing layer gives each transaction a batch id and transaction
  id. Replicas can execute the same input independently because the
  commit decision depends on deterministic metadata, not thread timing.
- The execution phase runs transactions on one snapshot. Writes stay in
  a transaction-local write set and are not installed in the database
  until commit.
- Each transaction makes write reservations after execution. A
  reservation records the smallest transaction id that wrote a key. A
  later transaction that tries to reserve an already-reserved key must
  abort, but it still completes the remaining reservations so the
  reservation table is deterministic.
- The commit phase checks write-after-write and read-after-write
  dependencies against the reservation table. If neither exists with an
  earlier transaction, writes are installed; otherwise the transaction is
  scheduled at the beginning of the next batch with relative order
  preserved.
- Deterministic reordering adds read reservations. A transaction may
  commit despite a read-after-write dependency if it does not also have
  a write-after-read dependency on earlier transactions. This transforms
  some RAW conflicts into WAR conflicts and commits more transactions
  under an equivalent serial order.
- The reordering check remains parallel: each transaction probes read
  and write reservation metadata rather than building one global serial
  dependency schedule.
- Under high write-write conflict rates, Aria can enter a fallback
  phase. Aborted transactions rerun under a Calvin-like deterministic
  lock path using their now-known read/write sets; a moving-average abort
  threshold controls when this fallback is enabled.
- The implementation stores per-record reservation metadata including
  batch id, lock bit, read transaction id, and write transaction id.
  Tables are primary hash tables with secondary hash tables; range
  queries are not supported in the implementation.
- Distributed execution sends remote reads and reservation requests to
  owning nodes and uses barriers between execution and commit phases.
  Remote requests and writes are batched to reduce network overhead.
- The evaluation reports Aria outperforming BOHM, PWV variants, Calvin,
  and synchronous primary-backup on YCSB, with up to 10.3x higher
  throughput than primary-backup in the single-node YCSB setup. On
  TPC-C with 180 partitions, Aria reports 1.9x over BOHM, 1.2x over
  Calvin, and 7.4x over primary-backup, while AriaFB helps on more
  contended TPC-C writes. The paper also reports near-linear scaling to
  16 nodes for YCSB under the tested multi-partition mixes.

**GPU DB mapping:** Aria is a good model for a future write-batch and
resident-refresh publication protocol. GPU DB already treats WAL,
visibility, invalidation, and resident snapshots as owner-controlled
boundaries. Aria suggests a way to do useful work before the final
publication boundary without letting thread timing decide correctness:
build batch-local read/write/residency reservations, then publish only
compatible fragments in deterministic order.

For COPY and mutation batches, a transaction-local or chunk-local write
set could reserve table/key/partition ranges before GPU-resident
invalidation, CPU index append, or WAL-visible publication. Conflicting
chunks would not install partial state; they would retry in a later owner
batch or switch to a deterministic hot-key path. For retained snapshots,
refresh plans could reserve source partitions and companion columns,
then publish a new immutable generation only if the source WAL/catalog
boundary still matches the reservation facts.

Aria's reordering is also relevant to GPU DB's route scheduler. Some
read-after-write conflicts in one batch may be serializable if the
equivalent order places read-only retained work before later writes.
That argues for classifying conflicts as WAW, RAW, and WAR instead of
collapsing all overlap into "abort" or "route through the mutation
owner." The engine should still keep WAL-before-visibility as authority,
but it can avoid needless retry when a deterministic equivalent order is
available.

The fallback mechanism is a practical warning. Optimistic batch
execution is attractive when conflicts are sparse or reorderable, but
hot contended writes need an explicit mode switch. GPU DB should measure
abort/retry pressure by table, partition, and key range, then move hot
keys to partition-owner ordering or stored-procedure-like deterministic
execution before retry storms destroy tail latency.

**Risks and mismatches:** Aria is a stored-procedure OLTP system, not a
PostgreSQL-compatible SQL engine. It targets one-shot, short-lived
transactions and does not support interactive multi-round transactions
or range queries in the implementation. The batch barrier can hurt
latency and throughput when one transaction in a batch is a straggler;
the paper reports an 81% slowdown in an extreme single-straggler
experiment. Deterministic replication by input also differs from GPU
DB's current WAL/checkpoint/archive recovery model. The system uses
single-version execution plus local write sets, whereas GPU DB's storage
roadmap includes MVCC visibility, long retained snapshots, DDL
invalidation, and over-resident tiers. The safe transfer is deterministic
reservation and publish discipline, not wholesale replacement of MVCC.

**Benchmark candidates:**

- Add no-op reservation telemetry to the mutation/COPY owner: keys or
  partition ranges touched, read/write overlap class, earliest
  conflicting transaction or chunk id, and retry/fallback decision.
  Minimum gate: no behavior change and no weakening of WAL-before-
  visibility.
- Build a small deterministic mutation-batch proof with two phases:
  prepare local write sets against one visibility boundary, then publish
  only chunks with no WAW or unsafe RAW conflict. Failure condition: any
  partial write, index append, resident invalidation, or response becomes
  visible before the commit decision.
- Test deterministic reordering on a synthetic workload where writes
  feed later reads but an equivalent serial order can place reads first.
  Compare full abort/retry, Aria-style WAW/RAW checks, and reordering
  with read reservations.
- Add a hot-key mode-switch benchmark. When conflict abort rate crosses a
  threshold, route the hot partition/key range to deterministic owner
  ordering and compare p50/p99, abort count, queue wait, and throughput
  against optimistic retry.
- Prototype resident-refresh reservation facts: source WAL boundary,
  partition ids, selected column families, companion columns, and
  invalidation generation. Proof gate: a refresh fragment is reused only
  when all reservation facts still match.
- Negative control: low-conflict retained reads and writes. Reservation
  metadata must not add measurable overhead to the existing uncontended
  COPY/read path before it earns its keep under contention.

### 2026-06-03 - AutoSteer learned optimizer knob steering

**Citation:** Christoph Anneser, Nesime Tatbul, David Cohen,
Zhenggang Xu, Prithviraj Pandian, Nikolay Laptev, and Ryan Marcus.
"AutoSteer: Learned Query Optimization for Any SQL Database." PVLDB
16(12), 2023, pp. 3515-3527. DOI: `10.14778/3611540.3611544`.
Retrieved 2026-06-03 from the VLDB PDF,
`https://www.vldb.org/pvldb/vol16/p3515-anneser.pdf`.

**Category:** query optimization / planning.

**Relevance tags:** learned query optimization; optimizer knobs;
route choice; bounded hints; query spans; latency-tail reduction;
planner diagnostics; CPU/GPU fallback; workload adaptivity.

**Core idea:** AutoSteer extends Bao-style learned steering so it can
work across SQL engines that expose optimizer knobs. Instead of
requiring experts to hand-pick a static hint-set collection for one
DBMS, it approximates the set of rewrite rules that matter for a
query, greedily explores useful knob combinations, and can then train
a predictor to select a hint-set for new queries.

The transferable result is not that GPU DB should let a model own
planning. It is that bounded, explainable route knobs can be explored
and learned around an existing optimizer while keeping the native
planner in control of legal plans. For GPU DB, this maps to learning
when to choose resident GPU, over-resident GPU, CPU fallback,
prefilter, compression, or admission behavior, provided the learned
choice is constrained by deterministic correctness gates.

**Concrete mechanisms:**

- AutoSteer takes a list of exposed optimizer knobs and interacts with
  the DBMS through connector functions for setting knobs, running
  `EXPLAIN`, and executing queries. The generic connector can be
  implemented with ordinary SQL/session knobs; the custom connector
  integrates with the optimizer to observe rewrite-rule application
  more efficiently.
- A query span approximates the rewrite rules that contribute to a
  query plan. In the custom PrestoDB integration, rewrite rules append
  their identifiers to the span when their conditions match and they
  transform part of the plan.
- AutoSteer starts with the default plan, then tests singleton
  hint-sets drawn from the query span. Hint-sets that beat the default
  are queued for bottom-up greedy expansion; unhelpful singleton
  choices are pruned from later combinations.
- The greedy search also considers alternative rules when one rule can
  replace or disable another rule in the plan. The goal is to discover
  beneficial small combinations without enumerating the full power set
  of optimizer knobs.
- In inference mode, AutoSteer uses Bao's tree convolutional neural
  network approach over query plans to predict execution time and
  select a hint-set without rerunning the full exploration for every
  incoming query.
- The generic integration trades lower engineering effort for more
  overhead because it may need multiple `EXPLAIN` calls to approximate
  query spans. The custom integration requires optimizer changes but
  can collect query-span data during one optimization pass.
- The evaluated systems include PrestoDB, PostgreSQL, SparkSQL, MySQL,
  and DuckDB. Public workloads include JOB, StackOverflow, and TPC-DS;
  the production workload is a Meta PrestoDB dashboarding workload
  over petabyte-scale data and more than 3,000 daily queries.
- In the reported PrestoDB JOB/Stack experiments, AutoSteer-C's best
  known hint-sets improve average runtime by about 30% on JOB and 42%
  on Stack, while inference mode improves them by about 28% and 32%.
- On the Meta dashboard workload, applying the discovered top
  PrestoDB hint-set reduces 99th-percentile dashboard query latency by
  about 20%, while the paper notes a few regressions are acceptable
  when tail and absolute latency improve.
- The main PrestoDB diagnostic finding is that disabling
  `HashGenOptimizer` helps many JOB queries but hurts some cases. This
  led to a size-based heuristic proposal rather than a blanket rule
  removal.
- The paper explicitly calls out cache state, memory footprint, CPU
  time, and multi-query interactions as important production metrics
  that single-query latency optimization does not fully capture.

**GPU DB mapping:** AutoSteer fits the planned GPU DB route layer as
a bounded advisor, not an authority. The deterministic planner should
still prove schema generation, visibility boundary, resident validity,
supported predicate family, memory budget, and fallback legality. A
learned steering layer can then choose among legal knobs such as
resident GPU route, CPU index route, CPU prefilter plus GPU tail,
over-resident chunk size, compression decode path, micro-batch ceiling,
and rejection versus fallback under saturation.

The query-span idea maps to route-span telemetry. For every query
shape, GPU DB can record which route rules actually fired: table
resident and valid, selected columns resident, predicate supported by
GPU kernel, partition count, estimated bytes moved, compression
format, queue class, GPU budget, CPU fallback eligibility, and
snapshot generation. That gives an optimizer-debugging surface before
any model is trained.

AutoSteer's generic/custom connector split also suggests a safe
implementation sequence. First, expose a generic diagnostic mode that
tries planner knobs offline against benchmark queries and records SQL
results, route telemetry, latency, bytes, and queue wait. Only after
useful knobs are identified should GPU DB add a lower-overhead custom
route-span path in the production planner.

The production lessons are especially relevant to a GPU system.
Choosing a route solely by median latency is dangerous when a plan
increases GPU memory pressure, pinned-buffer use, PCIe bytes, CPU
time, or queue interference. The reward signal for GPU DB route
steering should include p95/p99 latency, absolute latency change,
resident-hit preservation, bytes moved by tier, GPU queue wait,
retained-read starvation, and explicit memory-budget violations.

**Risks and mismatches:** AutoSteer targets query optimization for
analytics-heavy SQL engines, not WAL/MVCC correctness, transaction
scheduling, or GPU storage management. Its evaluated workloads are
mostly analytical and dashboard-style; OLTP point reads and COPY/write
admission need different reward signals. Training-mode exploration
executes alternative plans, which can be too expensive or unsafe on
live transactional workloads unless it runs offline or in shadow mode.
The inference model can still select regressing hint-sets, so GPU DB
must keep deterministic guardrails, negative controls, and fallback
caps. The paper also does not solve concurrency-aware planning across
many simultaneous sessions; single-query improvements may be bad if
they consume scarce GPU, host-memory, or IO resources.

**Benchmark candidates:**

- Add route-span telemetry to the planner for supported retained
  routes: legal route knobs, rules fired, rejected route reasons,
  resident generation, partition count, estimated bytes moved, queue
  class, and fallback eligibility. Minimum gate: no SQL behavior
  change and deterministic route explanations for each benchmark query.
- Build an offline route-steering harness that enumerates a bounded
  set of legal GPU DB route knobs for existing retained queries. Measure
  p50/p95/p99 latency, throughput, queue wait, H2D/D2H bytes, GPU
  elapsed time, CPU time, and memory-budget impact.
- Test a constrained learned or table-driven advisor that chooses
  between CPU route, resident GPU route, CPU prefilter plus GPU tail,
  and rejection/fallback under saturation. Failure condition: any model
  choice bypasses visibility, invalidation, or WAL-before-visibility
  checks.
- Add a negative-control workload where the fastest single-query GPU
  route starves retained point reads under concurrency. The advisor
  should learn or be constrained to prefer the route with better p99
  and resource isolation, not just lower isolated runtime.
- Add optimizer-regression reporting in absolute terms as well as
  relative terms: milliseconds saved/lost, bytes saved/lost, GPU queue
  time added, and pinned memory consumed. Proof gate: route changes can
  be accepted or rejected by workload-level tail and resource budgets.
- Keep the first production version rule-based. Use AutoSteer-style
  exploration to discover route heuristics, then encode the winning
  guardrails explicitly before considering online inference.

### 2026-06-03 - Cross-paper synthesis: route learning must be resource bounded

The last three modern entries cover over-resident storage initiation
(`Path to GPU-Initiated I/O`), deterministic OLTP batch publication
(`Aria`), and learned optimizer steering (`AutoSteer`). They converge
on one design track: GPU DB should expose rich route descriptors, but
publication and admission must remain deterministic. A route can be
learned, hinted, or benchmark-discovered only after WAL boundaries,
visibility compatibility, resident validity, owner ownership, and
resource budgets have already made it legal.

The strongest benchmark track is a resource-bounded route-span layer.
Each request should carry the facts that explain its route: snapshot
generation, partition or tier, resident bytes, transfer bytes,
storage-initiation mode, CPU time, GPU queue class, write/reservation
conflict class, and fallback legality. Aria says publish decisions
should be small and deterministic; AutoSteer says planner knobs should
be explored within a bounded span; Torp et al. say GPU-initiated IO is
only a win if its GPU resource burn is charged honestly.

The main category gap after this batch is still practical MVCC storage
and garbage collection under long retained snapshots, especially how
old versions, deleted keys, and hot-row histories move across CPU,
GPU, and future tiers. The next high-value queue choices are the
PVLDB 2017 empirical MVCC evaluation, LeanStore/Umbra tiered buffer
management, or Cicada/ERMIA for memory-optimized transaction engines.

Benchmark priorities:

- route-span telemetry before route learning
- deterministic mutation or refresh reservation facts before parallel
  write publication
- mixed retained-read plus over-resident cold-scan isolation before
  GPU-initiated IO
- workload-level p99 and resource-budget gates before accepting any
  learned planner knob

### 2026-06-03 - Empirical in-memory MVCC design tradeoffs

**Citation:** Yingjun Wu, Joy Arulraj, Jiexi Lin, Ran Xian, and
Andrew Pavlo. "An Empirical Evaluation of In-Memory Multi-Version
Concurrency Control." PVLDB 10(7), 2017, pp. 781-792.
Retrieved 2026-06-03 from the VLDB PDF,
`https://www.vldb.org/pvldb/vol10/p781-Wu.pdf`.

**Category:** MVCC / snapshot / visibility.

**Relevance tags:** MVCC; version storage; visibility checks;
version-chain ordering; garbage collection; index indirection;
serializable reads; long snapshot pressure; write-path memory
allocation; CPU/GPU resident snapshot design.

**Core idea:** The paper evaluates MVCC as a design space rather
than one algorithm. On a 40-core in-memory DBMS implementation, it
compares concurrency control protocols, version storage layouts,
garbage collection strategies, and index pointer schemes under YCSB
and TPC-C. The key lesson for GPU DB is that visibility and storage
layout choices can dominate protocol tweaks: the way versions are
allocated, chained, collected, and indexed decides whether reads stay
cache-friendly and whether updates avoid synchronization hotspots.

For GPU DB, this argues against treating MVCC metadata as a small
correctness detail bolted onto resident columns. The visibility
layout, old-version retention policy, index indirection, and memory
allocation policy need to be benchmarked as part of the retained
snapshot design, especially because GPU scans and micro-batches are
sensitive to pointer chasing and irregular metadata access.

**Concrete mechanisms:**

- The evaluated tuple version header carries transaction id,
  begin timestamp, end timestamp, and a neighboring-version pointer.
  Some protocols add read timestamp or read-count metadata.
- The paper compares MVTO, MVOCC, MV2PL, and SI plus SSN-style
  certification. MVOCC avoids read locks but pays read-set validation
  and can starve long read-only work; certifier schemes reduce some
  false aborts but add dependency tracking; MVTO performs well across
  several tested workloads.
- Append-only storage stores full versions in the table. Oldest-to-
  newest ordering avoids index head updates but forces latest-version
  traversal. Newest-to-oldest ordering makes current reads faster but
  needs index or indirection maintenance when a new version becomes
  head.
- Time-travel storage keeps a master tuple in the main table and old
  versions in a separate table, so indexes continue to point at the
  master version. It helps current-version access but still pays old-
  version maintenance costs.
- Delta storage keeps a master tuple plus delta records containing
  modified old attribute values. It reduces copying for narrow updates
  but makes reads and scans reconstruct values by walking version
  chains and fetching per-attribute deltas.
- For non-inline attributes in append-only storage, sharing unchanged
  values with reference counters avoids copying large values into each
  new version.
- Centralized allocation becomes a scalability point. The paper's
  mitigation is separate memory spaces expanded in fixed-size chunks,
  with worker threads allocating from a single space to reduce
  contention.
- MVCC garbage collection has three steps: detect expired versions,
  unlink them from chains and indexes, and reclaim storage. Epoch-
  based tracking avoids a fully centralized active-transaction check.
- Tuple-level background vacuuming is broadly compatible but can scan
  too much. Cooperative cleaning lets workers record expired versions
  during chain traversal, but it only fits certain append-only chain
  orderings and can miss "dusty corners" that no transaction visits.
- Transaction-level GC reclaims versions by finished transaction or
  epoch, using transaction write sets. In the paper's experiments it
  reduces synchronization overhead and improves update-intensive
  throughput versus tuple-level GC.
- Logical index pointers map stable tuple identifiers to version-chain
  heads through an indirection layer. They reduce secondary-index
  churn under updates but require chain traversal during reads.
- Physical index pointers point directly to exact versions and can
  speed read-heavy secondary-index access, but every new version must
  be inserted into every secondary index.
- The paper notes that MVCC index-only scans are not possible unless
  visibility metadata is embedded in the index; otherwise the executor
  must fetch tuple/version headers to determine visibility.
- Evaluation findings most relevant to GPU DB: N2O append-only chain
  ordering outperforms O2N in the tested YCSB cases; append-only and
  time-travel storage have better table-scan latency than delta
  storage; delta performs well for narrow updates but scan latency can
  grow badly; transaction-level GC improves throughput and memory
  behavior; logical pointers win under update-heavy secondary-index
  workloads.

**GPU DB mapping:** GPU DB's current P8 plan already treats GPU
resident state as immutable acceleration state built from CPU/WAL
truth. This paper says the CPU-side MVCC source and the published
resident layout must be designed together. A resident snapshot should
not require GPU kernels to chase CPU-style version chains. It should
carry dense visibility vectors or compact begin/end arrays generated
at a known WAL/transaction boundary, with enough metadata to prove
which old versions remain retained for active readers.

Append-only N2O plus stable logical tuple ids looks like the safest
near-term CPU truth shape for GPU DB: current-version reads and
refresh builders find the head quickly, while secondary indexes and
resident handles avoid wholesale churn through an indirection layer.
For GPU-resident scans, the engine should materialize only the
versions visible at the snapshot boundary into dense column groups,
not expose long version chains to kernels.

Delta storage is a warning. It is attractive for narrow updates and
write throughput, but it turns multi-attribute reads and scans into
chain reconstruction. GPU DB should keep any delta/update log as a
CPU-side or refresh-side structure until a benchmark proves that a
GPU kernel can apply deltas without destroying latency, coalescing,
and branch behavior.

The GC findings map directly to retained snapshot retirement. GPU DB
needs transaction or generation-level retirement for resident CPU and
GPU buffers: versions, column groups, resident indexes, and pinned
staging buffers should become reclaimable when no active read snapshot
or refresh generation can see them. A tuple-level vacuum is still
useful as a correctness fallback, but the hot path should retire whole
generations or write-batch fragments where possible.

The index-only-scan point is important for GPU indexes. If a resident
key vector or CPU secondary index does not carry visibility metadata,
the executor still needs a visibility probe before returning rows.
For GPU DB, either resident indexes must be generation-specific and
therefore visibility-filtered by construction, or they must store
compact begin/end metadata adjacent to keys so lookup kernels do not
fall back to scattered CPU header checks.

**Risks and mismatches:** The paper evaluates an in-memory CPU DBMS,
not a GPU execution engine, and it excludes logging and recovery from
the study. It also focuses on serializable transaction execution and
does not evaluate PostgreSQL-compatible interactive transactions,
DDL invalidation, over-resident GPU tiers, or NVMe placement. The
reported percentages come from Peloton experiments on 40 CPU cores in
2017, so they should guide benchmark shape rather than be copied as
expected GPU DB gains. The paper's range-query discussion is limited;
phantom prevention and predicate locks remain separate design work.

**Benchmark candidates:**

- Add MVCC storage-shape telemetry for the CPU truth path: average
  version-chain length, head-order policy, old-version bytes, tuple
  header bytes touched per read, secondary-index update count, and
  per-thread allocation contention. Minimum gate: no behavior change.
- Compare resident refresh build time from N2O append-only heads,
  O2N chains, and a delta-log simulation. Measure CPU build time,
  cache misses if available, generated GPU bytes, and retained query
  p50/p99 after publication.
- Prototype generation-level resident retirement. A read snapshot
  pins one resident generation; mutation/refresh publishes a newer
  generation; GC reclaims old CPU/GPU buffers only after all readers
  release. Failure condition: any stale generation is selected for a
  new reader after invalidation.
- Test logical versus physical secondary-index maintenance under
  update-heavy COPY/UPDATE workloads. Measure rows/sec, index bytes,
  chain traversal cost, and retained refresh invalidation pressure.
- Build a visibility-adjacent resident index proof: key vector plus
  compact begin/end generation metadata, compared with a resident key
  vector that requires scattered tuple-header probes. Proof gate:
  identical SQL results and lower p99 for batched point lookups.
- Run a negative delta-storage experiment for GPU scans: keep narrow
  update deltas and apply them during a scan. Failure condition:
  delta application causes enough branch/scatter overhead that a full
  dense snapshot rebuild wins for the target retained workload.

### 2026-06-03 - Chiller contention-centric transaction partitioning

**Citation:** Erfan Zamanian, Julian Shun, Carsten Binnig, and
Tim Kraska. "Chiller: Contention-centric Transaction Execution
and Data Partitioning for Modern Networks." SIGMOD 2020.
Retrieved 2026-06-03 from the arXiv preprint,
`https://arxiv.org/abs/1811.12204`; DOI:
`https://doi.org/10.1145/3318464.3389724`.

**Category:** transaction processing / write path and runtime /
session admission.

**Relevance tags:** contention-aware partitioning; partition
owners; hot-record routing; two-phase locking; distributed
transactions; RDMA-era OLTP; operation reordering; write
admission; owner-local commit; high-contention benchmarks.

**Core idea:** Chiller argues that fast RDMA-era networks change
the dominant objective for distributed OLTP. If remote messaging
and bandwidth are no longer the main bottleneck, minimizing the
number of cross-partition transactions can be the wrong target.
The paper instead optimizes for data contention: put records that
are hot and commonly accessed together where their lock duration
can be minimized, even if that creates more distributed
transactions.

The transferable lesson for GPU DB is not "use Chiller's protocol
as-is." It is that partition ownership and route choice should be
driven by measured conflict cost, not by a static preference for
locality or single-owner execution. A GPU DB write path targeting
many logical sessions needs to know which keys, partitions, and
resident generations create serialization pressure, then route or
batch them so the contended part is as short, owner-local, and
observable as possible.

**Concrete mechanisms:**

- Chiller splits transaction operations into a cold outer region
  and hot inner region. The outer region locks and reads less
  contended records first; the inner region handles the highly
  contended records late and commits them quickly.
- Candidate inner-region operations must access records marked
  contended and must not have primary-key dependencies on
  operations hosted by other partitions. Value dependencies matter
  for correctness checks, but do not necessarily constrain lock
  acquisition order in the same way.
- If all inner-region candidates are hosted by one partition, that
  partition becomes the inner host. If candidates span hosts, the
  prototype chooses the host with the most candidate operations.
- Once outer-region locks are acquired, the coordinator delegates
  the inner region with enough inputs and read-set values for the
  inner host to evaluate transaction constraints. If the inner host
  succeeds, the transaction is considered committed and the outer
  region must finish.
- The paper's partitioner samples transaction read/write sets,
  estimates per-record conflict likelihood, builds a star graph
  with transaction vertices connected to record vertices, and uses
  graph partitioning to minimize weighted cut edges. Cut hot edges
  represent records that would remain in an outer region and keep a
  longer contention span.
- Conflict likelihood models write-write and read-write conflicts
  using sampled read/write rates over the lock window. Pure
  read-only sharing does not create contention in this model.
- The lookup table can focus on hot records above a contention
  threshold; colder records can use ordinary hash or range
  partitioning. In the paper's YCSB local experiment, partial
  lookup coverage gives Chiller useful throughput much earlier
  than distributed-transaction-minimizing baselines.
- Fault tolerance requires special handling because the inner
  region commits before the outer participants finish. Chiller uses
  synchronous log shipping for the inner region before its commit
  point, and recovery rules decide commit/abort from surviving
  inner-host replicas and pending outer-region participants.
- The implementation uses partition-local execution threads,
  hash-bucket lock granularity, replicated bucket-to-partition
  lookup tables, RDMA operations or RPC messages, coroutine workers
  while transactions wait on network operations, and NVM-style logs
  for crash recovery.
- Evaluation claims most relevant to GPU DB: on high-contention
  TPC-C, Chiller scales better with more worker threads than
  NO_WAIT, WAIT_DIE, and OCC; on YCSB distributed with 7 machines,
  the paper reports much lower abort rates and roughly 2x over the
  second-best baseline; on the Instacart-derived workload, combining
  contention-centric partitioning with two-region execution beats
  partitioning or reordering alone.

**GPU DB mapping:** The current GPU DB architecture already names
owner domains and bounded queues. Chiller suggests those owners
should eventually be shaped by measured contention, not just table
or partition identity. For hot warehouses, districts, accounts, or
order-line keys, the runtime should expose a conflict heat signal:
write/read arrival rate, abort or retry rate, owner queue wait, lock
or reservation duration, WAL reservation wait, resident invalidation
rate, and refresh interference. That signal can drive a partition
owner split, hot-key routing table, or micro-batch policy.

The two-region idea maps most safely to GPU DB as a benchmarked
"hot reservation last" write path. Cold validation, non-hot reads,
and WAL record preparation can happen before touching the hottest
reservation or owner queue. The hot owner then performs a tiny,
deterministic commit-critical section: validate the current hot
generation, reserve/apply the hot mutation, append or publish the
necessary WAL fact, invalidate affected resident snapshots, and
release. GPU DB must keep WAL-before-visibility stronger than the
paper's no-failure explanation; any early commit analogue needs a
durable decision record before visible state changes.

Chiller's partitioner also maps to resident data placement. A
GPU-resident partitioning policy should not admit or co-locate data
only by scan locality. It should ask whether hot keys are causing
owner serialization, snapshot invalidation, or refresh rebuild
storms. A small hot-key lookup table, backed by ordinary placement
for cold rows, may be a better first implementation than a full
record-level placement map.

For 1M logical sessions, the coroutine/RDMA details are secondary
but useful: stalled distributed work should yield to other admitted
requests, and transport or owner resources should be bounded and
observable. The runtime should avoid letting a request hold scarce
hot-owner, pinned-buffer, or GPU stream resources while waiting on
unrelated remote or cold work.

**Risks and mismatches:** Chiller assumes stored procedures or
one-shot transaction descriptions so the system can reorder
operations. PostgreSQL-compatible interactive transactions are a
poor fit unless GPU DB restricts the optimization to known stored
procedures, COPY chunks, or internally generated mutation batches.
The protocol is 2PL-centered and does not directly solve MVCC
snapshot visibility, predicate/range phantoms, or GPU-resident
snapshot correctness. Its implementation locks hash buckets and
does not prevent phantoms. The evaluation is distributed CPU/RDMA
OLTP, not GPU execution, and it relies on NVM/RDMA assumptions that
may not match the current single-node GPU DB. Finally, early inner
commit complicates recovery; GPU DB should borrow the shorter hot
critical section and contention-aware placement first, not the exact
commit protocol.

**Benchmark candidates:**

- Add contention heat telemetry to the mutation path: per key or
  bucket write/read arrivals, queue wait, reservation duration,
  retry/abort count, resident invalidation count, and refresh delay.
  Minimum gate: no behavior change and bounded cardinality for hot
  telemetry.
- Build a TPC-C-style hot-record benchmark comparing ordinary
  partition-owner mutation order against "hot reservation last."
  Measure committed rows/sec, p50/p99 latency, owner queue wait,
  WAL wait, invalidation delay, and retry/abort rate.
- Prototype a small hot-key routing table for the most contended
  keys or buckets, leaving cold keys on hash/range placement. Proof
  gate: the table improves high-contention throughput without
  increasing cold-key p99 or creating unbounded lookup metadata.
- Test a mutation micro-batch shape where cold validation and WAL
  payload assembly happen before the hot owner critical section.
  Failure condition: any request observes visibility before its WAL
  decision and invalidation facts are durable/published.
- Add a negative-control workload where minimizing distributed work
  produces worse p99 than contention-aware placement. The planner or
  admission policy should prefer the layout with lower hot-owner
  queue wait even if it performs more cross-owner routing.
- For resident snapshots, measure whether co-locating hot invalidated
  keys by refresh partition reduces rebuild storms versus pure range
  or hash partitioning. Required measurement: refresh bytes,
  invalidated resident generations, retained-read fallback rate, and
  write throughput during refresh.

### 2026-06-03 - MEMTIS access-distribution memory tiering

**Citation:** Taehyung Lee, Sumit Kumar Monga, Changwoo Min, and
Young Ik Eom. "MEMTIS: Efficient Memory Tiering with Dynamic
Page Classification and Page Size Determination." SOSP 2023,
pp. 17-34. doi:10.1145/3600006.3613167. Retrieved 2026-06-03
from the author-hosted ACM paper PDF,
`https://multics69.github.io/pages/pubs/memtis-lee-sosp23.pdf`.

**Category:** multi-tier cache / data placement.

**Relevance tags:** tiered memory; hot/cold placement; CXL; NVM;
huge pages; subpage skew; access sampling; background migration;
host-memory tier policy; resident segment granularity.

**Core idea:** MEMTIS argues that tiered-memory systems make bad
placement decisions when they rely on fixed hotness thresholds,
recency-only approximations, or page-fault critical-path migration.
Instead, the system samples memory accesses with Intel PEBS, builds
a compact access-frequency histogram over allocated pages, and
chooses hot, warm, and cold thresholds from the whole distribution
so the hot set approximates the fast-tier capacity. It also treats
page size as a tiering decision: huge pages help TLB reach, but they
waste scarce fast-tier memory when only a few 4KB subpages are hot.

The strongest transferable idea for GPU DB is "placement by observed
distribution, not static thresholds." P8 already has explicit tiers:
GPU HBM, CPU canonical state, CPU derived indexes/statistics, durable
WAL/checkpoints, and future NVMe/CXL-like tiers. MEMTIS suggests that
promotion, demotion, refresh, and resident-segment granularity should
be driven by live access histograms and skew estimates rather than a
single configured hot-table rule.

**Concrete mechanisms:**

- MEMTIS samples retired LLC load misses and retired store
  instructions using PEBS. A kernel background thread processes
  sampled addresses and updates page and subpage access metadata.
- The sampling interval is adjusted dynamically to keep sampling CPU
  usage under a target, 3% of one core by default. The paper reports
  average sampling-thread CPU usage of 2.016%, maximum 3.0%, and
  average application performance overhead of 0.922%.
- Page hotness is maintained as an exponentially decayed access count.
  Cooling periodically halves access counts; because histogram bins
  are exponential, cooling mostly shifts histogram counts left.
- The page access histogram has 16 exponential bins by default, so the
  metadata for the distribution is tiny. MEMTIS uses this distribution
  to place the hottest pages into the fast tier rather than relying on
  a fixed access-count threshold.
- MEMTIS derives hot, warm, and cold thresholds. Hot pages should move
  to the fast tier; cold pages should move to the capacity tier; warm
  pages are left in place unless free space is needed, reducing
  unnecessary migration of pages near the decision boundary.
- Promotion and demotion run in background migration threads, not in
  the page-fault handler. The fast tier keeps a small free-space
  reserve, 2% in the evaluated configuration, for future allocations
  and promotions.
- A separate emulated base-page histogram tracks 4KB access
  distribution even when the OS currently uses huge pages. MEMTIS
  compares an estimated base-page hit ratio against the actual
  fast-tier hit ratio to decide whether splitting huge pages is worth
  doing.
- Huge-page split is triggered only when the estimated potential hit
  ratio improvement is sufficiently large, 5% or higher in the paper.
  The number of huge pages to split scales with estimated benefit,
  latency gap between tiers, and sampled huge-page activity.
- Split candidates are selected by subpage skew: a huge page with a
  small number of very hot subpages ranks higher than a uniformly hot
  huge page. Splitting, subpage migration, and all-zero subpage
  release happen in the background.
- Base pages are coalesced back into huge pages conservatively, only
  when all constituent base pages are hot.
- The implementation is a Linux 5.15.19 kernel change of about 5,166
  lines. It stores huge-page/subpage metadata inside unused compound
  page metadata where possible and bounds worst-case base-page metadata
  overhead at 0.195% of memory footprint.
- Evaluation uses eight memory-intensive workloads, including Silo
  with YCSB-C and an in-memory Btree lookup benchmark, with DRAM as
  fast tier and Optane NVM or emulated CXL memory as capacity tier.
  MEMTIS reports best performance in 23 of 24 NVM configurations and
  33.6% geomean improvement over the second-best system. Huge-page
  splitting improves Silo and Btree by about 10% overall in the shown
  1:8 setting, and can reduce Btree RSS substantially.

**GPU DB mapping:** GPU DB should not treat the OS page cache or a
single table-level cache bit as a sufficient placement policy. The
P8 cache manager needs its own distribution-aware telemetry for
resident data: per table, partition, column group, resident key vector,
index fragment, pinned buffer class, and cold segment. The hot set
should be chosen against the actual scarce tier: HBM bytes, pinned
host memory, DRAM cache budget, CXL/remote memory budget, or NVMe
prefetch budget.

The hot/warm/cold split maps well to route decisions. Hot resident
segments can stay in GPU memory, warm segments can remain in host
DRAM or compressed host memory without immediate churn, and cold
segments can demote to NVMe or rebuild-on-demand state. Warm segments
are important because GPU DB can otherwise thrash: a retained
snapshot, refresh builder, or batched lookup path might repeatedly
promote/demote nearly-hot partitions and destroy p99 latency.

The huge-page lesson maps to resident segment size. A large resident
partition or column group is only good if its subregions are
uniformly useful. For skewed point-read or session-heavy workloads,
GPU DB may need sub-segment placement: keep hot key ranges, hot
columns, or compact lookup structures in HBM while leaving colder
subranges in host memory. For scan-heavy uniformly hot regions,
larger segments remain attractive because they reduce metadata,
launch, TLB, and transfer overhead.

MEMTIS also strengthens the runtime document's owner model. Tier
promotion and demotion should run off the critical query path through
bounded residency queues. Queries should see explicit states and
fallback reasons, not block unpredictably on page faults or implicit
OS migration. The GPU DB analogue of PEBS does not have to be CPU
hardware sampling only; it can combine route counters, resident
segment touch counts, GPU kernel bytes, H2D/D2H bytes, queue waits,
refresh invalidations, and periodic CPU sampling where available.

For future CXL or remote memory tiers, MEMTIS suggests an evaluation
shape: as the latency gap narrows, placement mistakes are less
catastrophic but still matter. GPU DB should measure whether a
segment belongs in GPU HBM, local DRAM, CXL-like memory, or NVMe by
bytes touched, access skew, transfer cost, refresh cost, and queue
latency, not only by whether the table is "hot."

**Risks and mismatches:** MEMTIS is an OS memory-tiering system, not a
DBMS buffer manager or GPU-resident storage engine. It does not know
about WAL-before-visibility, MVCC snapshots, relation generations,
DDL invalidation, SQL route choice, CUDA streams, or pinned host
buffer ownership. PEBS samples CPU memory references; it does not
directly observe GPU HBM touches, device-side cache behavior, or
NVMe/GPUDirect access. Background migration can still interfere with
query latency if GPU DB makes refresh or pinned-buffer movement too
aggressive. The paper's Silo and Btree experiments are useful
database-adjacent evidence, but they are not PostgreSQL-compatible
workloads and do not include GPU execution.

**Benchmark candidates:**

- Add residency access histograms: per table/partition/column group
  touch count, bytes touched, selected route, queue wait, H2D/D2H
  bytes, and refresh invalidation count. Minimum gate: no behavior
  change and bounded telemetry cardinality.
- Compare static residency admission against distribution-aware
  admission for hot partitions. The hot set should fill a configured
  HBM budget with the highest-value resident segments; failure
  condition: static admission beats the histogram policy on p95/p99
  latency or refresh bytes under skew.
- Build a warm-segment policy where near-hot partitions are not
  immediately demoted if the GPU tier has temporary pressure. Measure
  promotion/demotion count, retained-read fallback rate, and p99
  latency under shifting skew.
- Prototype sub-segment placement for a skewed lookup workload:
  resident full partition versus resident hot key-range/vector plus
  host fallback. Proof gate: identical SQL results, lower HBM bytes,
  and no worse p99 for the cold range.
- Add a negative-control scan workload with uniform access. The policy
  should keep larger contiguous segments and avoid needless
  sub-segmentation; failure condition: splitting increases metadata or
  launch overhead enough to hurt throughput.
- Add a tier-latency sensitivity model for future CXL-like memory:
  emulate local DRAM, slower DRAM, and NVMe-backed cold segments in
  route costing. Required measurement: selected tier, bytes moved,
  queue wait, refresh cost, and p50/p99 query latency.

### 2026-06-03 - Cross-paper synthesis: visibility, contention, and placement need distribution summaries

The last three reviewed papers converge on a useful pattern for GPU
DB: avoid making hot-path decisions from global constants. The
empirical MVCC study says version storage, index pointers, and GC
must be chosen from workload shape; Chiller says partitioning and
write scheduling should follow measured contention rather than
locality alone; MEMTIS says memory placement and page size should
follow access distributions rather than static thresholds.

The shared design track is a family of compact summaries owned by
the right domain. Mutation owners need version-chain, conflict, and
hot-key summaries. Residency owners need access, skew, invalidation,
and refresh-cost summaries. GPU execution owners need route demand,
bytes moved, queue wait, batch size, and scratch/pinned-buffer
summaries. These summaries should drive bounded choices: which
versions retire, which keys get special routing, which resident
segments stay in HBM, and which requests are admitted or deferred.

The category gap remains high-concurrency session admission at very
large logical session counts. Runtime papers have covered eRPC,
Demikernel, Caladan, Shenango, and Shinjuku, but the journal still
needs more work on practical TCP/event-loop service models and
connection-state budgeting for pgwire-like protocols.

Benchmark priorities after this batch:

- generation-level MVCC/resident retirement with chain-length and
  index-maintenance telemetry
- hot-key or hot-bucket routing driven by measured conflict heat
- distribution-aware residency admission with warm-state anti-thrash
  behavior
- sub-segment versus full-segment placement under skewed lookups and
  uniform scans
- a logical-session memory probe that measures idle state, active
  credits, response buffers, and queue saturation separately

### 2026-06-03 - TAS: TCP Acceleration as an OS Service

**Citation:** Antoine Kaufmann, Tim Stamler, Simon Peter, Naveen Kr. Sharma,
Arvind Krishnamurthy, and Thomas Anderson. "TAS: TCP Acceleration as an OS
Service." EuroSys 2019. doi:10.1145/3302424.3303985. Retrieved 2026-06-03
from the author-hosted ACM paper PDF,
`https://homes.cs.washington.edu/~arvind/papers/flextcp.pdf`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** TCP fast path; high-connection-count services; POSIX
sockets; IO-worker isolation; bounded per-flow state; workload-proportional
runtime; congestion policy; packet queues; tail latency.

**Core idea:** TAS argues that datacenter TCP overhead is not only a kernel
crossing problem. General TCP stacks carry too much uncommon-case code,
scattered per-connection state, cache pollution, and shared-state coordination
into the hot path. TAS splits common-case RPC packet processing into a trusted
fast-path OS service on dedicated or dynamically assigned cores, while a slow
path handles connection setup/teardown, congestion policy, timeouts, and other
stateful or uncommon work. Applications keep a POSIX sockets interface through
a user-level library, so the fast path is centralized and policy-compliant
rather than a fully application-owned kernel-bypass stack.

For GPU DB, the useful point is that high logical session scale can be framed
as an explicitly budgeted service with tiny hot-path state, not as one thread
or one heavyweight protocol object per client. TAS reports 102 bytes of
fast-path per-flow state, more than 20,000 active flows per core fitting in
commodity cache by their estimate, and only up to 7% throughput degradation
from peak as its RPC echo benchmark scales to 64K connections. It also reports
up to 90% higher throughput and 57% lower tail latency than IX on evaluated
cloud workloads, while retaining sockets compatibility.

**Concrete mechanisms:**

- TAS has three components: a fast path, a slow path, and an untrusted
  per-application user-space stack connected by shared-memory queues.
- The fast path is a small TCP datapath over DPDK. It parses common-case
  headers, deposits in-order receive payload directly into per-flow circular
  receive buffers, generates acknowledgements, enforces configured send
  rates/windows, segments outgoing payload, and updates local sequence/window
  state.
- Connection setup, teardown, congestion policy, timeout handling, stack
  registry, and exceptional packets are delegated to the slow path because
  they are less common or have non-constant per-packet cost.
- The POSIX sockets surface is provided by a user-level library that can be
  dynamically linked to unmodified applications. TAS also exposes a lower-level
  API when applications can opt into it.
- Each flow has fixed send and receive payload buffers plus compact fast-path
  metadata, including opaque application id, context queue, rate bucket, buffer
  offsets/sizes/head/tail, sequence/ack/window state, peer tuple, limited
  out-of-order state, congestion counters, retransmit counters, and RTT
  estimate.
- Per-flow receive buffers make flow-control window calculation constant time
  and avoid iterating over connections sharing a buffer. Per-flow send buffers
  reduce head-of-line blocking from congestion or receiver flow control.
- Context queues notify applications of receive and transmit progress. If a
  context queue is full, later arrivals can retry notification; if a payload
  receive buffer is full, the packet is dropped and ordinary TCP flow control
  and retransmission take over.
- Congestion control policy runs in the slow path but is enforced by the fast
  path. The paper implements rate-based DCTCP and TIMELY-style mechanisms;
  the fast path periodically exposes ACKed bytes, ECN-marked bytes, fast
  retransmit counts, and RTT estimates.
- Workload proportionality is controlled by monitoring fast-path CPU
  utilization. The slow path removes a core when aggregate idle capacity is
  above a threshold and adds one when idle capacity falls below another
  threshold.
- Scale-up and scale-down avoid draining all queues. TAS steers NIC RSS and
  application routing asynchronously and protects rare wrong-core packet cases
  with per-connection locking.
- The prototype uses 10,127 lines of C across fast path, slow path, and socket
  library. It does not resize connection buffers, does not fully implement
  TCP slow start, and does not support fragmented IP packets.
- Evaluation uses RPC echo, a skewed key-value store, and FlexStorm
  real-time analytics. The key-value workload uses 32K connections and a
  90% GET / 10% SET mix; TAS with sockets improves throughput versus Linux and
  IX and has lower median-to-tail latency than IX in the reported setup.

**GPU DB mapping:** TAS strengthens the current
`11-high-throughput-query-runtime.md` direction: network IO should be an owned
service with small hot-path state, explicit queues, and clear handoff to
mutation, read-snapshot, residency, and GPU execution owners. GPU DB should
not let SQL protocol parsing, response encoding, WAL admission, GPU route
selection, and socket IO share one unbounded client thread. The runtime should
separate common-case pgwire request/response movement from slow or stateful
paths such as authentication, DDL, errors, COPY setup, prepared-statement
changes, large responses, and disconnect cleanup.

The fixed per-flow buffer lesson maps directly to 1M logical sessions. Idle
session state must be small, and active resources must be separate credits:
frontend frame bytes, parsed command slots, response buffers, decoded COPY
chunks, mutation-owner slots, retained read slots, pinned staging buffers, and
GPU execution budget. A million idle sessions should not imply a million
resident response buffers or GPU work handles.

TAS also suggests an internal "fast path / slow path" classification for SQL
work. Common retained reads over a compatible immutable snapshot can use a
short IO-worker path that enqueues a typed request and receives completion on a
response ring. Slow-path work should include DDL, catalog generation changes,
long COPY chunks, refresh, invalidation repair, CPU fallback scans, large
multi-packet responses, and any route that needs extensive planning. The goal
is not to hide slow work, but to keep it from polluting the cache and queue
behavior of the short path.

The workload-proportional core model maps to runtime workers and GPU streams.
GPU DB can scale IO workers, read workers, or GPU execution owners according
to queue depth, queue wait, socket writability, response backlog, and
per-worker CPU usage, but should preserve ownership boundaries. Like TAS,
resource changes should be asynchronous and observable, with a fallback path
for requests that arrive at a temporarily wrong owner or saturated queue.

For P8, TAS is a reminder that network and response overhead can erase
resident GPU wins. Even if retained kernels are zero-H2D and microsecond-scale,
the pgwire path needs compact per-session metadata, reusable buffers, and
bounded response queues so protocol work does not dominate p99 latency.

**Risks and mismatches:** TAS is a TCP stack, not a database runtime. It does
not solve WAL-before-visibility, MVCC validation, snapshot compatibility,
catalog invalidation, SQL planning, GPU residency, or transaction recovery.
Its strongest implementation assumes DPDK, dedicated fast-path cores, and
datacenter network common cases; the current GPU DB benchmark endpoint uses
ordinary TCP/pgwire and may not be able to adopt a TAS-like stack without
deployment tradeoffs. Fixed connection buffer sizes are a poor fit for
arbitrary SQL responses unless GPU DB separates idle logical state from active
resource credits. The evaluated 64K connection scale is useful but still far
below the 1M logical-session target, and sockets compatibility still requires
application relinking in the prototype.

**Benchmark candidates:**

- Build a logical-session memory probe for the pgwire endpoint: allocate
  compact idle session state at `1k`, `10k`, `100k`, and projected `1M`
  counts, while only a bounded active subset owns request/response buffers.
  Minimum gate: report bytes per idle session, bytes per active credit class,
  and named saturation reasons without changing SQL correctness.
- Split pgwire work into fast-path and slow-path route classes in telemetry:
  short retained read, mutation, COPY, DDL/catalog, refresh/invalidation,
  CPU fallback, large response, and error/disconnect. Failure condition:
  long-path work can still block socket progress for unrelated short reads.
- Prototype fixed-capacity response rings per IO worker with reusable encoded
  buffers for one retained read shape. Measure owner queue wait, socket write
  backlog, buffer reuse latency, p50/p99 response time, and overload reasons.
- Compare thread-per-client against a small IO-worker pool for repeated
  retained reads at high logical session counts. Expected improvement: lower
  memory footprint and lower p99 queueing once connection count dominates.
- Add active-resource credits per session: parsed frontend messages, in-flight
  engine requests, COPY chunk bytes, response bytes, and retained/GPU slots.
  Proof gate: a single session can still reach baseline throughput when
  unsaturated, while overload is rejected or delayed at the named boundary.
- Add an IO-worker scale policy experiment driven by queue wait and CPU usage,
  with asynchronous reassignment. The policy should report scale events and
  wrong-owner fallbacks rather than silently increasing latency.

### 2026-06-03 - Hint-QPT: Hints for Robust Query Performance Tuning

**Citation:** Haibo Xiu, Yang Li, Qianyu Yang, Weihang Guo, Yuxi Liu,
Pankaj K. Agarwal, Sudeepa Roy, and Jun Yang. "Hint-QPT: Hints for
Robust Query Performance Tuning." PVLDB 18(12), 2025, pp. 5327-5330.
doi:10.14778/3750601.3750663. Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol18/p5327-xiu.pdf`.

**Category:** query optimization / planning.

**Relevance tags:** robust query optimization; selectivity uncertainty;
plan hints; sensitivity analysis; targeted runtime statistics; route
explainability; GPU fallback decisions; operator diagnostics.

**Core idea:** Hint-QPT is a demonstration system built on PARQO. It starts
from the premise that selectivity estimates will often be wrong, so tuning
should not simply chase the cheapest plan at one estimated point. Instead,
it recommends plans with low expected penalty under a learned distribution of
selectivity errors, identifies the selectivity dimensions whose errors most
affect plan optimality, and lets users refine those dimensions by executing
counting subqueries or manually overriding estimates.

The paper's most transferable idea is not the GUI itself. It is the split
between robust default planning and targeted evidence gathering. A plan can
be allowed to cost more under the optimizer's current estimate if it is less
fragile across plausible true selectivities. When more certainty is needed,
the system should ask for a small number of high-value measurements rather
than probing every subquery or blindly trusting local error magnitude.

**Concrete mechanisms:**

- Hint-QPT profiles selectivity-estimation errors for querylets: small
  subquery templates involving up to three joined tables.
- Component error models for selections and joins are combined into a
  query-specific distribution `f(s | s_hat)` over possible true selectivities
  given optimizer estimates.
- Robust plan selection uses PARQO's penalty objective. A plan incurs zero
  penalty when its cost is within a tolerance factor of the true optimal
  cost; otherwise the excess cost is penalized. The robust plan minimizes
  expected penalty over the selectivity-error distribution.
- The demonstration uses a tolerance factor of `0.2`, so plans within 20% of
  the true optimal cost are treated as acceptable for the penalty function.
- Sensitive dimensions are identified with Sobol-style global sensitivity
  analysis over plan penalty variance, not by simply choosing dimensions with
  the largest estimated error or largest local cost derivative.
- Users can refine a sensitive dimension by executing a counting subquery to
  obtain the actual selectivity, then injecting that selectivity into the
  optimizer and reoptimizing.
- The system visualizes a query as a join graph, shows error distributions for
  selected nodes and edges, highlights sensitive dimensions, compares default,
  robust, and adjusted plans, and annotates operator trees with estimated and
  actual costs/cardinalities.
- It also visualizes plan cost surfaces over two sensitive selectivity
  dimensions, with probability heat over the plausible true-selectivity
  region.
- The implementation works on PostgreSQL 16.2 with minor modifications for
  hint injection and interaction. The authors state the framework is designed
  to work with optimizers that support selectivity and plan hints.
- The paper is a PVLDB demo paper, so detailed performance-evaluation claims
  are limited. It describes the Join Order Benchmark/IMDb demonstration and
  points to PARQO for the deeper validation.

**GPU DB mapping:** GPU DB route choice has the same fragility problem, with
more dimensions. A retained GPU route can be fast when cardinality, selected
bytes, resident validity, GPU queue delay, response size, and refresh risk are
close to estimates, but a wrong estimate can turn the same route into a slow
fallback or an expensive over-resident transfer. The planner should therefore
track not only the cheapest route estimate, but also how brittle the route is.

The immediate mapping is a robust route explanation layer over
`Engine::plan_relational_resident_route(...)`. For each accepted or rejected
resident path, the planner should expose the dimensions that could flip the
decision: selectivity, output rows, selected columns, resident bytes, refresh
age/cost, GPU queue depth, expected kernel launch amortization, CPU fallback
cost, and response bytes. A fragile accepted route should be visible as
fragile, not silently treated as a deterministic win.

Hint-QPT's sensitive-dimension idea maps cleanly to admission-time probes. For
some query templates, GPU DB can cheaply collect one targeted count, range
histogram, resident segment touch count, or queue-delay sample before choosing
between CPU, resident GPU, cold GPU transfer, over-resident execution, or
rejection. The probe should be tied to a route-flip risk, not run merely
because a statistic has high uncertainty.

For retained read micro-batching, robust planning can become a template cache.
Repeated same-shape queries over the same relation can remember which
dimensions usually decide the route: cardinality, batch size, GPU queue wait,
or response size. The planner can then refresh only those facts when a cached
route is near a decision boundary.

The paper also supports an operator-facing diagnostics surface. GPU DB should
be able to explain that a query used CPU fallback because a GPU route was
fragile under selectivity uncertainty, refresh risk, or queue saturation. That
is more useful than a single "unsupported" or "cost too high" reason when
operators are trying to tune resident data placement and planner rules.

**Risks and mismatches:** Hint-QPT is an interactive tuning demonstration, not
a production autonomous optimizer. It focuses on selectivity errors in join
planning, while the current GPU DB first slice is mostly single-table retained
lookups and aggregates. Its use of PostgreSQL hint injection does not directly
define a native optimizer API. Counting subqueries can be too expensive under
high concurrency, especially if they compete with the actual workload for
CPU, GPU, or IO resources. The demo paper provides limited quantitative
evaluation, so PARQO and follow-up robust-optimizer papers remain the better
source for algorithmic validation. Finally, a robust plan that is acceptable
for elapsed query time may still be unacceptable for GPU DB if it consumes
scarce pinned buffers, resident memory, or GPU stream slots needed by many
sessions.

**Benchmark candidates:**

- Add a route-fragility record to resident planner decisions. For each
  accepted/rejected route, report the top route-flip dimensions among
  estimated rows, selected bytes, resident validity, refresh age, GPU queue
  depth, CPU fallback estimate, D2H response bytes, and batch depth. Minimum
  gate: no route behavior change and deterministic explanation output.
- Build a retained route sensitivity benchmark: vary predicate selectivity
  and GPU queue depth around the CPU/GPU decision boundary, then measure how
  often the chosen route is more than 20% slower than the best measured route.
- Prototype targeted pre-route probes for one query shape: a cheap CPU count
  or resident histogram lookup only when the route decision is near the
  boundary. Proof gate: fewer bad route choices without increasing p50
  latency for easy decisions.
- Extend route telemetry to distinguish "unsupported", "not resident",
  "over budget", "queue saturated", "fragile selectivity", and "fragile
  response size" fallback reasons. Failure condition: the planner cannot name
  which fact would have made the GPU route viable.
- Compare three policies on repeated retained templates: cheapest current
  estimate, conservative robust route, and targeted-probe route. Measure
  p50/p95/p99 latency, throughput, GPU queue wait, route flips, and fallback
  counts.
- Add a negative-control case where selectivity uncertainty is high but route
  choice is insensitive. The planner should not run a probe or abandon the
  resident route simply because the statistic itself is uncertain.

### 2026-06-03 - Taurus lightweight parallel logging

**Citation:** Yu Xia, Xiangyao Yu, Andrew Pavlo, and Srinivas Devadas.
"Taurus: Lightweight Parallel Logging for In-Memory Database Management
Systems." PVLDB 14(2), 2020, pp. 189-201. doi:10.14778/3425879.3425889.
Retrieved 2026-06-03 from `https://www.vldb.org/pvldb/vol14/p189-xia.pdf`.

**Category:** transaction processing / write path.

**Relevance tags:** WAL scalability; parallel logging; command logging;
data logging; dependency vectors; recovery ordering; MVCC logging; early lock
release; NVMe; write admission; replay.

**Core idea:** Taurus attacks the single-log bottleneck in in-memory OLTP
engines by writing transactions to multiple log streams while preserving only
the dependency order needed for correct commit and recovery. Instead of forcing
a single global LSN order, each transaction carries an LSN Vector (LV) whose
elements summarize which positions in each log stream the transaction depends
on. A transaction can commit when its own log record and the dependent log
positions are persistent; recovery can replay transactions in any order that
respects those LV dependencies.

The design is especially useful because it supports both data logging and
command logging. Data logging can replay physical changes but writes larger
records. Command logging writes smaller procedure/input records but must replay
transactions in a dependency-respecting order. Taurus keeps that option open
by explicitly logging dependency metadata. On its DBx1000 evaluation, Taurus
reports up to 9.9x runtime speedup over single-stream data logging, 2.9x over
single-stream command logging, and recovery speedups up to 22.9x and 75.6x for
data and command logging respectively. Against parallel baselines, it reports
up to 2.8x better performance on NVMe SSDs and 9.2x on HDDs.

**Concrete mechanisms:**

- Taurus uses one log manager per log file and assigns workers to log
  managers. Each transaction writes one log record to one stream.
- An LV has one element per log stream. `T.LV[i] = x` means transaction `T`
  may depend on transactions in log `i` up to LSN `x`, but not after `x`.
- Tuple-level `readLV` and `writeLV` propagate dependencies between
  transactions. Reads merge prior writer LVs; writes merge prior reader and
  writer LVs so RAW, WAW, and WAR dependencies are captured under 2PL.
- Log records contain the redo/command payload plus a copy of the transaction
  LV before the transaction's own log-stream element is advanced to its
  allocated LSN.
- A global persistent-LV vector (`PLV`) records how far each log stream has
  flushed. A transaction can be marked committed when `PLV >= T.LV` and older
  transactions in the same log stream have committed.
- Early lock release moves log persistence off the lock-holding critical path:
  after publishing tuple LVs and releasing locks, the transaction waits
  asynchronously for dependent log positions to become persistent.
- Log managers use per-worker `allocatedLSN` and `filledLSN` indicators to
  flush only stable regions of a log buffer.
- Recovery uses an end-LV (`ELV`) from log file sizes to decide whether a
  transaction committed before crash, per-log pools for decoded transactions,
  and a global recovered-LV (`RLV`) that advances as replay finishes.
- A transaction is eligible for recovery when `T.LV <= RLV`; this is a
  parallel topological replay over the implicit dependency graph.
- Tuple LV compression stores dependency metadata only for active lock-table
  entries; evicted tuple dependency state is approximated from the current
  persistent LV with a tunable delta that trades metadata footprint for
  artificial dependencies.
- Log-record LV compression periodically writes PLV anchors into log buffers
  and stores only LV dimensions that exceed the latest anchor, reducing log
  bytes at the cost of some recovery parallelism.
- SIMD vector operations reduce LV maintenance cost; the paper reports up to
  89.5% lower LV overhead when the number of log files grows.
- Taurus has OCC and MVCC extensions. For MVCC, versions carry an LV; reads
  merge the visible version LV, updates merge the old version LV, committed
  write versions receive the transaction LV, and recovery uses multi-version
  replay so physically late but logically early transactions can be handled.
- Under high contention, many inter-log dependencies can reduce recovery
  parallelism. Taurus explicitly falls back to serial recovery for highly
  skewed cases in its sensitivity study.

**GPU DB mapping:** Taurus is a strong fit for the GPU DB write path because
it separates durable ordering from a single total log bottleneck. A future
partition-owned mutation path can keep per-owner WAL streams and publish a
compact dependency vector or generation vector for transactions that cross
owners. That would let independent partitions flush and recover in parallel
without pretending all writes share one hot global LSN counter.

The LV idea maps naturally to the current owner-domain architecture. Mutation
owners can publish `PLV`-like flush frontiers; read-snapshot publication can
name the owner flush frontier it depends on; residency refresh can record the
WAL/visibility vector that made a GPU snapshot valid. A retained snapshot would
then carry a generation vector precise enough for correctness and compact
enough for route checks.

For COPY admission, Taurus suggests a benchmarkable alternative to one global
WAL bottleneck: route chunks to partition-local WAL streams, encode cross-
partition dependencies only when a transaction actually crosses owners, and
publish visibility only after the dependent flush frontier is durable. This
preserves WAL-before-visibility while giving independent chunks room to flush
and replay in parallel.

Command logging is relevant but risky. GPU DB could log high-level COPY chunk
metadata or deterministic mutation commands for some bulk paths to reduce log
bytes, while keeping data logging for nondeterministic or externally visible
effects. Taurus shows that command logging needs explicit dependency metadata
and deterministic replay; it is not just "write fewer bytes."

The recovery algorithm also maps to rebuildable GPU residency. During crash
recovery, CPU truth should replay first from durable WAL streams in dependency
order. GPU resident snapshots, indexes, and layout metadata remain rebuildable
acceleration state tied to recovered owner frontiers rather than durable
authority.

**Risks and mismatches:** Taurus is evaluated in DBx1000, not in a PostgreSQL-
compatible engine with arbitrary SQL, WAL segments, MVCC visibility, and GPU
resident snapshots. Its cleanest command-logging path assumes deterministic
stored-procedure style replay, which is narrower than ad hoc SQL. LVs grow
with the number of log streams, so a GPU DB with many partitions needs
compression, sparse encoding, or hierarchy before adopting the idea broadly.
Tuple-level read/write LV tracking can be expensive for wide scans, secondary
indexes, and high-cardinality hot data unless stored in lock-table or version
metadata carefully. High contention can collapse recovery parallelism and may
need serial fallback. The paper's largest gains depend on storage bandwidth,
CPU cache-coherence bottlenecks, and benchmark transaction shape, so the
transferable claim is the dependency-preserving log-stream design, not the
absolute throughput numbers.

**Benchmark candidates:**

- Prototype a two-stream WAL admission model for partitioned COPY chunks:
  each partition owner appends locally, cross-partition chunks carry a small
  dependency vector, and visibility publishes only when the dependent flush
  frontier is durable. Minimum gate: crash/replay tests recover the same rows
  and visibility boundaries as the single-stream path.
- Add WAL-frontier telemetry per mutation owner: allocated LSN, stable-buffer
  LSN, durable LSN, dependent durable frontier, commit-wait time, and reason
  for any visibility delay.
- Compare global-LSN COPY admission with partition-local WAL streams on a
  synthetic workload containing independent chunks and a controlled percentage
  of cross-partition transactions. Measure rows/sec, commit wait, fsync bytes,
  recovery time, and dependency-vector bytes per transaction.
- Build a recovery topological-replay proof over small synthetic log streams.
  Include RAW/WAW/WAR or MVCC-version dependencies, crash truncation, and high-
  contention serial fallback. Failure condition: replay can expose a row or
  resident snapshot before all dependency frontiers are recovered.
- Evaluate sparse or anchored dependency vectors for many partition owners.
  Proof gate: vector metadata remains small for independent single-partition
  writes, while cross-partition writes still block visibility on the correct
  durable frontier.
- Test command-log eligibility for deterministic COPY chunks only. Expected
  improvement: lower WAL bytes and faster recovery for deterministic replay;
  rejection condition: a chunk depends on nondeterministic SQL, external
  state, volatile functions, or catalog state not captured in the command.

### 2026-06-03 - Bounded-delay multiversion concurrency and precise GC

**Citation:** Naama Ben-David, Guy E. Blelloch, Yihan Sun, and Yuanhao
Wei. "Multiversion Concurrency with Bounded Delay and Precise Garbage
Collection." SPAA 2019, pp. 161-172. doi:10.1145/3323165.3323185.
Retrieved 2026-06-03 from
`https://www.cs.cmu.edu/~yihans/papers/concurrency.pdf`.

**Category:** MVCC / snapshots / version reclamation.

**Relevance tags:** bounded-delay readers; precise garbage collection;
functional data structures; path copying; version maintenance; wait-free
snapshot acquire; read-mostly workloads; batched writes; long snapshots;
retained snapshot retirement; memory pressure.

**Core idea:** The paper shows that multiversioning can provide constant
extra delay for read-only transactions while reclaiming old versions as soon
as the last holder releases them, if the database state is represented by
persistent functional data structures and version roots are managed by a
precise version-maintenance object. The key shift is away from per-object
version chains. A reader acquires one current root pointer, runs ordinary
read-only code over immutable data, then releases the root. Reads do not scan
version lists and do not block writers. A single non-conflicting writer has
delay proportional to the number of processes, while concurrent writers are
lock-free but may abort each other.

The paper's PSWF version-maintenance algorithm supports `acquire`, `set`, and
`release`. `acquire` returns the current version in O(1) delay. `set` attempts
to publish a new root and can fail if another writer published after the
writer's acquire. `release` returns exactly the version that became dead, if
this process was the last holder. The authors define this as precise GC: old
reachable state is kept only while it is current or held by an active
transaction. In experiments on a functional balanced-tree map with 140 query
threads and one update thread, PSWF used much less version memory than epoch
or hazard-pointer baselines while keeping comparable query throughput and
better update throughput than the other non-blocking reclamation choices. The
paper reports 60%-90% lower average version memory than epoch/hazard-pointer
implementations, and batched functional-tree updates outperforming tested
concurrent tree baselines by more than 20% on mixed YCSB workloads, with the
important caveat that batching raises update latency.

**Concrete mechanisms:**

- Database state is modeled as immutable memory graphs rooted by a version
  pointer. Updates create a new version by path copying from the old root.
- A transaction acquires exactly one root version. Read-only transactions run
  user code over that immutable root and are considered responsive before
  release-time cleanup completes.
- The Version Maintenance object exposes `acquire(k)`, `set(k, data*)`, and
  `release(k)` for process id `k`, with at most one acquired version per
  process.
- A version is live if it is the current version or if some process acquired
  it and has not released it.
- Precise release returns a dead version exactly when the releasing process is
  the last holder; no version is returned twice.
- PSWF is wait-free for version-maintenance operations, with O(1) acquire and
  O(P) set/release delay for P processes.
- A successful writer publishes a new root with `set`; if another writer has
  already published since this writer's acquire, the `set` can fail and the
  new root must be collected, retried, or aborted.
- Garbage collection after release traces from released roots and can reclaim
  memory in work linear in the amount of garbage collected.
- Batched updates are implemented by accumulating update requests and applying
  them to a functional tree with a parallel multi-insert, giving single-writer
  publication but parallel update construction.
- The evaluation uses a large ordered-map workload and YCSB-style read/update
  mixes. It compares PSWF against epoch, hazard-pointer, RCU, and related VM
  variants, but it is not a SQL engine evaluation.

**GPU DB mapping:** The strongest mapping is to retained read snapshot
publication and retirement. GPU DB's first P8 slice already treats resident
GPU state as immutable acceleration tied to a WAL/visibility boundary. This
paper suggests making snapshot acquisition a deliberately tiny operation:
read workers should grab a generation/root handle, execute against immutable
metadata and buffers, and release the handle without ever walking per-row
version chains on the hot read path.

For 1M logical sessions, the design lesson is to separate logical sessions
from physical snapshot holders. A session should not pin a resident generation
for its whole connection lifetime. It should acquire a snapshot for a single
statement, portal batch, or bounded cursor quantum, then release it quickly so
precise retirement can work. Long portals need an explicit budget and telemetry
because they are the real memory retention event.

The functional-data-structure requirement does not directly mean GPU DB should
rewrite all CPU storage as persistent trees. It does mean resident metadata,
planner route tables, visibility summaries, and segment manifests should be
published by root replacement rather than mutated in place. A residency owner
can build a new immutable manifest for table generation `G+1`, publish the
root, and let readers on `G` drain. The retired root then drives precise
cleanup of device buffers, pinned host buffers, and statistics blocks.

For writes, the single-writer/batched-writer model fits partition owners and
COPY admission better than arbitrary SQL updates. A partition owner can batch
mutations, build a new CPU/GPU-friendly segment or delta root, and publish it
after WAL-before-visibility is satisfied. Cross-partition or highly contended
writes still need the transaction-scheduling and dependency-vector machinery
from the recent SMF, Chiller, and Taurus reviews.

The paper also gives a concrete warning against per-object version lists for
read-heavy GPU routing. If a retained scan or lookup has to chase row-level
version chains to prove visibility, the GPU route loses predictable latency.
For hot resident routes, visibility should be encoded as coarse generation
boundaries, compact begin/end arrays, or prefiltered snapshot manifests, with
chain traversal kept on CPU fallback paths.

**Risks and mismatches:** This is a theory-heavy SPAA paper with data-structure
experiments, not a production DBMS paper. Its strongest bounds assume purely
functional data structures, one acquired version per process, and a read-mostly
or batched-write shape. SQL engines have secondary indexes, catalog state,
variable-length rows, DDL, deletes, vacuum, write amplification, and crash
recovery concerns that are outside the model. O(P) set/release is acceptable
only if P is the physical worker count, not 1M logical sessions. Path copying
can be expensive for wide row updates or large mutable indexes. Precise
release also requires disciplined statement/cursor lifetimes; a single slow
reader can still hold old buffers, even if it does not block writers.

**Benchmark candidates:**

- Implement a retained snapshot acquire/release microbenchmark over immutable
  resident manifests. Gate: acquire cost is constant with respect to row count,
  resident segment count, and historical generation count.
- Add snapshot-holder telemetry: current generation, holder count, oldest held
  generation age, bytes pinned by old generations, and release latency by
  statement/cursor class.
- Compare exact-generation retirement against epoch-style retirement for
  resident buffers under a mixed workload with short statements and a small
  number of long cursors. Measure retained bytes, eviction pressure, and p99
  route latency.
- Build a partition-owner publication proof: apply batched mutations to a new
  immutable manifest, publish only after WAL-before-visibility, and retire the
  prior manifest when holders release. Failure condition: a read can observe a
  stale or partially updated manifest.
- Stress logical session scale separately from physical snapshot holders:
  simulate 1M idle sessions plus a bounded active-statement set and verify that
  memory retention follows active holders, not connection count.
- Add a negative-control resident route that must walk per-row version chains.
  Compare it with a generation-manifest route to quantify the latency and GPU
  divergence cost the design is trying to avoid.

### 2026-06-03 - Cross-paper synthesis: roots, frontiers, and active holders

The last batch spans networking/runtime admission (TAS), robust planner
diagnostics (Hint-QPT), parallel WAL dependency frontiers (Taurus), and
bounded-delay snapshot/version retirement (Ben-David et al.). The convergence
is that the hot path should move small, explicit tokens rather than broad
mutable state: queue tokens for network work, route-fragility facts for the
planner, WAL dependency vectors for durability, and immutable root handles for
snapshot visibility.

The design track that now looks strongest is root-and-frontier publication.
Mutation owners advance durable frontiers. Residency owners publish immutable
roots tied to those frontiers. Read workers acquire short-lived root handles.
Planner decisions name the route dimensions that would invalidate the choice.
GPU execution then consumes only work whose root, frontier, and route facts are
compatible.

The main category gap is still practical HTAP snapshot policy: how to support
fresh OLTP reads, retained analytical scans, and long cursors without letting
old generations dominate GPU memory. The next useful papers should lean into
production MVCC garbage collection, dual-snapshot HTAP, and contention-aware
partitioning rather than another GPU-OLAP pipeline paper.

Benchmark priorities from this batch:

- Measure active snapshot holders, not sessions, as the memory-retention unit.
- Add per-owner WAL/frontier telemetry before attempting multi-stream write
  admission.
- Make route fragility visible at the CPU/GPU/fallback boundary.
- Prove that immutable resident roots retire precisely under long-reader
  pressure before adding more resident route families.

### 2026-06-03 - AnKerDB fine-granular virtual snapshotting

**Citation:** Ankur Sharma, Felix Martin Schuhknecht, and Jens Dittrich.
"Accelerating Analytical Processing in MVCC using Fine-Granular
High-Frequency Virtual Snapshotting." arXiv:1709.04284, 2017. Retrieved
2026-06-03 from `https://arxiv.org/pdf/1709.04284`.

**Category:** MVCC / snapshot / visibility and hybrid HTAP.

**Relevance tags:** heterogeneous OLTP/OLAP execution; virtual snapshots;
MVCC version-chain avoidance; column-granular snapshots; long analytical
reads; snapshot freshness; garbage collection; kernel-assisted COW; route
classification.

**Core idea:** AnKerDB argues that mixed OLTP/OLAP workloads should not force
short updates and long scans through one homogeneous MVCC representation.
OLTP transactions run on the newest versioned columns, while read-only OLAP
transactions run on separate read-only virtual snapshots. Both sides still use
MVCC, but frequent snapshots keep each representation's version chains short,
and analytical scans can often run in tight loops over snapshot columns instead
of chasing long newest-to-oldest chains.

The paper's enabling mechanism is a custom Linux system call, `vm_snapshot`,
which snapshots arbitrary virtual memory ranges inside one process. Unlike
`fork`, it does not duplicate the whole process address space; unlike the
authors' earlier user-space rewiring approach, it avoids many repeated `mmap`
calls when VMAs fragment. In the system evaluation, snapshots are triggered
after 10,000 commits, lazily materialized per accessed column, and kept
consistent by recording a snapshot timestamp before materialization. The paper
reports OLAP latency roughly 2x to 4x lower than homogeneous MVCC baselines
on its mixed TPC-H-inspired workload, and mixed workload throughput almost 2x
higher, while pure OLTP throughput remains comparable to homogeneous full
serializability.

**Concrete mechanisms:**

- The OLTP component owns the current up-to-date column representation.
  Updates first live in transaction-local memory; at commit, old column values
  move into newest-to-oldest version chains and new values overwrite the
  current column.
- Snapshot isolation is extended to full serializability with read-set
  validation based on predicate ranges and recently committed write ranges,
  following the HyPer precision-locking style.
- Read-only analytical transactions are classified into an OLAP component and
  run on read-only virtual snapshots. Fresh OLTP writes proceed on the new
  current representation while scans continue on older snapshot roots.
- On snapshot creation, the virtual duplicate becomes the new OLTP current
  column, while the former current column plus its existing version chains
  become the OLAP snapshot. Later updates build version chains only on the new
  OLTP current column.
- Snapshot creation is timestamp-triggered, but column materialization is
  lazy. A snapshot timestamp is logged after a commit threshold; a transaction
  touching a set of columns materializes only missing snapshots for those
  columns.
- Multi-column snapshot consistency is handled by the shared snapshot
  timestamp. During actual column materialization, writers take shared column
  locks and snapshot materialization takes an exclusive column lock.
- `vm_snapshot(src_addr, length)` duplicates VMAs and, for private mappings,
  copies relevant page-table entries so source and destination virtual ranges
  share physical pages until copy-on-write.
- An extended `vm_snapshot(dst_addr, src_addr, length)` form can place the
  snapshot into an already reserved virtual range, allowing old snapshot
  address space to be reused.
- The authors compare physical copying, `fork`, user-space rewiring, and
  `vm_snapshot`. Their 200 MB column microbenchmark finds `vm_snapshot`
  stable under VMA fragmentation, 68x faster than rewiring after all 51,200
  pages have been touched, and up to 6x faster on writes to snapshotted pages
  because ordinary kernel COW handles the write path.
- Old version garbage collection is simplified for analytical history: when no
  transaction can access an old snapshot and a newer snapshot exists, dropping
  the snapshot also drops the old version chains attached to it.
- Evaluation uses a column-oriented in-memory prototype, TPC-H Q1/Q4/Q6/Q17
  plus full scans for OLAP, and handcrafted update transactions for OLTP. It
  explicitly notes sublinear 8-thread scaling because serializable commit
  validation still has partially sequential protected state.

**GPU DB mapping:** The paper is a strong argument for treating snapshot route
classification as a first-class runtime decision. Short transactional reads,
mutations, long analytical scans, and retained GPU reads should not all pay
the same row-level MVCC traversal cost. The current P8 plan already uses
immutable resident snapshots; AnKerDB sharpens that into a policy: analytical
or GPU-resident scans should run on snapshot roots that make visibility cheap,
while current OLTP writes build fresh deltas elsewhere.

Column-granular lazy materialization maps directly to GPU residency. A GPU DB
does not need to refresh every column or table at each visibility boundary.
It can record a durable visibility/frontier boundary, then materialize only
the column families needed by admitted retained routes. This fits P8's first
slice of `int4` and `text` column groups and gives a concrete benchmark for
refresh cost versus route freshness.

The role swap between old and new columns also suggests a useful mental model
for resident generations. A residency owner can publish generation `G+1` as
the current GPU route while generation `G` remains read-only for active long
queries. Mutations should invalidate or delta-build the current generation
after WAL-before-visibility, not mutate buffers currently held by readers.

The custom kernel call is less directly portable than the policy. GPU DB
should not depend early on a patched Linux kernel, but the mechanism identifies
what must be measured: page-table/COW snapshot cost, column materialization
cost, COW write amplification, and whether a virtual-memory snapshot tier can
be an intermediate CPU host snapshot before GPU refresh. The stronger
transferable idea is column-granular, high-frequency, lazily materialized
snapshot publication, not the exact syscall.

The simplified GC story also complements the recent precise-retirement
synthesis. Old analytical state can retire by generation once active holders
drop it, instead of forcing a hot-path scan through every row version. For GPU
resident buffers, the equivalent is exact holder-counted generation retirement
for device buffers, pinned host buffers, and column manifests.

**Risks and mismatches:** AnKerDB is a prototype from 2017, evaluated on an
8-thread CPU system with a custom Linux 4.8.17 kernel, not a production
PostgreSQL-compatible engine and not a GPU database. Its OLTP transactions are
handcrafted updates rather than TPC-C or arbitrary SQL. The design requires a
correct transaction classifier; a misclassified write or long cursor could
break assumptions or pin memory. Snapshot materialization takes exclusive
column locks, so high-frequency snapshots may still disturb write latency if
columns are hot or wide. Kernel-level virtual snapshotting may not compose
cleanly with pinned CUDA buffers, GPUDirect paths, NUMA placement, huge pages,
or future CXL tiers. The paper reports strong mixed-workload benefits, but
commit validation remains a scaling bottleneck and absolute throughput claims
should not be transferred to the GPU engine.

**Benchmark candidates:**

- Add a CPU-side column-generation snapshot prototype for one admitted table:
  record a visibility boundary, lazily materialize only requested column
  families, and publish an immutable generation root. Minimum gate: identical
  results to the current MVCC tuple path under insert/update/delete and WAL
  replay.
- Compare row-version-chain scans with generation snapshot scans on a mixed
  workload: repeated updates to a controlled fraction of rows plus retained
  `COUNT`, `SUM`, prefix filter, and key lookup routes. Failure condition:
  retained route latency still grows with historical version-chain length.
- Measure snapshot refresh granularity: whole table versus selected column
  family versus selected segment. Required metrics: refresh latency, bytes
  copied or shared, write stall time, holder count, and resulting GPU route
  freshness.
- Prototype holder-counted generation GC for CPU column snapshots and GPU
  resident buffers. Proof gate: old generations retire as soon as the last
  statement/cursor holder releases, without relying on global epoch lag from
  idle sessions.
- Build a negative-control virtual-memory snapshot experiment using ordinary
  OS mechanisms available without a patched kernel, such as fork or mmap/COW
  where feasible. Use it to decide whether VM-assisted host snapshots are
  worth pursuing before GPU refresh.
- Add route classification telemetry for OLTP current path, retained snapshot
  path, long analytical path, and CPU fallback. Expected improvement: policy
  decisions can be tied to observed version-chain length, snapshot age,
  refresh cost, and write-stall budget.

### 2026-06-03 - Lero learning-to-rank query optimization

**Citation:** Rong Zhu, Wei Chen, Bolin Ding, Xingguang Chen, Andreas
Pfadler, Ziniu Wu, and Jingren Zhou. "Lero: A Learning-to-Rank Query
Optimizer." PVLDB 16(6):1466-1479, 2023. DOI
`10.14778/3583140.3583160`. Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol16/p1466-zhu.pdf`; arXiv version:
`https://arxiv.org/abs/2302.06873`.

**Category:** Query optimization / planning.

**Relevance tags:** learned route selection; candidate plan ranking;
pairwise plan comparison; native optimizer augmentation; dynamic workload
adaptation; cardinality perturbation; background exploration; GPU route
fallback; resource-budget-aware planning.

**Core idea:** Lero argues that query optimization does not need a learned
model to predict exact plan latency. It needs a reliable way to rank a bounded
set of candidate plans. Instead of replacing the native optimizer, Lero runs
on top of it, generates a small but diverse candidate set, and uses a
pairwise learning-to-rank comparator to choose among candidates. This keeps
the native optimizer as the cold-start baseline while allowing execution
feedback to correct systematic plan-choice mistakes.

The paper's strongest lesson for GPU DB is that learned route selection should
be shaped as bounded ranking over admissible candidates, not unbounded
latency prediction. A GPU DB planner can enumerate a small candidate set such
as CPU tuple/index path, CPU segment path, GPU resident route, GPU cold
transfer route, and reject/fallback path, then rank only candidates that have
already passed correctness, visibility, residency, and resource-budget gates.

**Concrete mechanisms:**

- Lero uses a native optimizer to generate candidate plans rather than
  learning a full optimizer from scratch.
- The comparator `CmpPlan(P1, P2)` is trained as a binary classifier over two
  plans, with labels derived from observed plan latency order rather than
  absolute latency values.
- Plan embeddings are shared between the two comparator inputs. The practical
  implementation uses a one-dimensional embedding so the model induces a
  simple total order over candidates.
- Offline pre-training uses synthetic plans and native estimated costs, so the
  model starts close to the native optimizer without executing a large cold
  training workload.
- Online training runs alternative candidate plans on idle workers, stores
  execution statistics, and periodically updates the pairwise comparator.
- Candidate exploration perturbs cardinality estimates by scaling factors and
  sub-query size groups, then asks the native optimizer to produce alternative
  plans. This is intended to uncover join-order and physical-operator choices
  hidden by cardinality error.
- Candidate generation is prioritized near the native optimizer's choice and
  has bounded growth, reported as at most `O(q * log_alpha Delta)` candidates
  for `q` tables under the paper's heuristic.
- The authors evaluate on PostgreSQL 13.1 with JOB/IMDB, STATS, TPC-H, and
  TPC-DS-style generated workloads. They report stable-model execution-time
  reductions versus PostgreSQL of 70%, 44%, 21%, and 13% on those benchmark
  families, respectively, with lower regression frequency than Bao/Bao+ on
  STATS.
- The evaluation includes dynamic data insertion on STATS and finds the
  relative ordering labels easier to adapt than exact latency labels.
- The paper explicitly treats varying runtime resource budgets as future work:
  resource budget features could be added to plan embeddings, but that would
  require training data under varied budgets.

**GPU DB mapping:** GPU DB's planner has a harder route-choice problem than
ordinary PostgreSQL plan selection because correctness gates and resource
budgets are first-class. A route may be fast only if a resident generation is
valid, a GPU queue has capacity, a snapshot holder can be acquired, and HBM or
pinned-buffer budgets are not saturated. Lero suggests splitting this into two
layers: deterministic admissibility first, learned ranking second.

For the first planner slice, the native rules should still generate and gate
candidate routes: CPU owner path, immutable resident GPU path, cold transfer
GPU path, partitioned resident path, and overload/fallback. A Lero-like
ranking layer can then compare only candidates whose invariants are already
proved. This prevents the learned model from optimizing through
WAL-before-visibility, MVCC compatibility, invalidation, or memory pressure.

The pairwise-ordering idea maps well to GPU route telemetry. Exact latency
prediction will be noisy across queue depth, batch size, CUDA stream state,
resident bytes, snapshot age, and CPU contention. Pairwise labels such as
"resident partitioned route beat CPU segment path for this query shape under
this queue-depth bucket" are likely cheaper to learn and easier to invalidate
when data placement changes.

Lero's cardinality-perturbation explorer also gives a concrete GPU DB
counterpart: perturb route-relevant estimates rather than arbitrary SQL hints.
Examples include selectivity, expected result rows, resident bytes touched,
transfer bytes, snapshot freshness, queue wait bucket, and refresh cost.
Exploration should remain bounded and should run on idle capacity or shadow
workloads, never on a path that jeopardizes production latency.

The pre-training story maps to starting from deterministic architecture rules:
prefer valid resident routes for supported same-shape hot reads, prefer CPU
fallback when visibility or residency cannot be proven, reject when all queues
are saturated, and avoid cold GPU transfer below a measured row/byte threshold.
Observed measurements can later adjust the ranking without erasing those
rules.

**Risks and mismatches:** Lero is evaluated mainly on analytical join
benchmarks, not OLTP transaction scheduling, write admission, MVCC visibility,
or GPU execution. Its candidate plans are generated through PostgreSQL-style
cardinality perturbation; GPU DB route alternatives include resource and
residency states that may not appear in a normal optimizer search space. The
paper assumes candidate exploration can use idle resources, which is dangerous
under strict latency SLOs unless capped and isolated. The reported gains are
for PostgreSQL workloads and should not be transferred to point lookups,
micro-batched retained aggregates, or high-concurrency pgwire sessions without
measurement. The resource-budget extension is only discussed, not evaluated,
yet GPU DB route quality depends heavily on queue, memory, and CUDA resource
budgets.

**Benchmark candidates:**

- Build a deterministic candidate-route enumerator for one retained query
  family. Candidate set: CPU tuple/index path, CPU segment path, GPU resident
  path, GPU cold transfer path, and explicit overload/reject. Gate: every
  candidate carries a reasoned admissible/not-admissible status before ranking.
- Collect pairwise route labels from bounded smoke runs: for identical query
  shapes, record which admissible route wins under row-count, selectivity,
  resident-byte, queue-depth, snapshot-age, and batch-size buckets. Failure
  condition: ranking decisions cannot be explained by recorded route features.
- Add a shadow-ranking mode that logs the route a Lero-like comparator would
  have chosen while executing the deterministic production route. Minimum
  proof: no correctness behavior changes and no visible latency impact.
- Compare exact-latency regression with pairwise ranking for CPU-vs-GPU route
  choice on retained `COUNT`, `SUM`, point lookup, prefix filter, and cold
  transfer queries. Required metric: misroute rate, p95/p99 latency regression,
  and adaptation speed after resident invalidation or refresh.
- Treat resource budgets as explicit features: GPU queue depth, HBM pressure,
  pinned-buffer availability, active snapshot holders, and mutation-owner queue
  depth. Gate: model recommendations change when a route becomes saturated,
  but deterministic overload rules still dominate.
- Run a negative-control exploration experiment that lets the model rank
  ungated routes. It should fail by demonstrating stale, saturated, or
  non-admissible choices, justifying the deterministic gate-before-rank design.

### 2026-06-03 - Plor predictable low-tail transactions

**Citation:** Youmin Chen, Xiangyao Yu, Paraschos Koutris, Andrea
C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau, and Jiwu Shu. "Plor:
General Transactions with Predictable, Low Tail Latency." SIGMOD 2022,
pages 19-33. DOI `10.1145/3514221.3517879`. Retrieved 2026-06-03 from
the author PDF at
`https://storage.cs.tsinghua.edu.cn/papers/sigmod22plor.pdf/`; DOI page:
`https://doi.org/10.1145/3514221.3517879`.

**Category:** Transaction processing / write path and concurrency control.

**Relevance tags:** predictable tail latency; hybrid OCC/2PL;
high-contention OLTP; commit-priority ordering; latch-free lock metadata;
interactive transactions; delayed write locks; admission under conflict;
bounded retry; mutation-owner queues.

**Core idea:** Plor starts from an observation that matters directly to the
GPU DB write path: throughput and tail latency fail for different reasons.
OCC-style systems often keep throughput high, but high-contention
transactions can abort repeatedly and create extreme 99.9th percentile
latency. WOUND_WAIT-style 2PL reduces starvation by giving older transactions
priority, but loses throughput through blocking and lock overhead. Plor
combines the two by requiring transactions to register read/write locks before
accessing records, while allowing reads to ignore conflicts during the read
phase and delaying conflict detection until commit.

The transferable design is not "use Plor everywhere." It is to treat conflict
metadata as a tail-latency control plane. A transaction can expose enough
state early for conflict arbitration and priority without forcing every
conflict to block immediately. For GPU DB, that suggests a write-admission
path where requests register read/write intentions, visibility boundaries, and
age/priority metadata before doing expensive owner, index, refresh, or GPU
work, then resolve conflicts at explicit publish or commit boundaries.

**Concrete mechanisms:**

- Plor uses pessimistic locking and optimistic reading. Transactions acquire
  read or write locks before accessing records, but readers can ignore current
  writers during the read phase because writers buffer updates privately until
  commit.
- Each worker context stores a worker id, transaction timestamp, and status in
  a compact word. Conflicting older transactions can mark younger transactions
  aborted by changing the target worker status.
- Locks retain transaction state rather than only a binary locked/unlocked
  bit. The lock manager keeps a current writer, an ordered writer wait list,
  and a reader list. This lets commit-time conflict resolution compare
  timestamps and enforce older-first priority.
- Commit has three major steps: upgrade write-set locks to exclusive mode and
  detect read-write conflicts; release read locks; apply buffered updates and
  release write locks. Younger conflicting readers can be killed, while the
  committing transaction waits for older readers.
- Delayed write-lock acquisition is optional. Blind writes can defer write
  locks until commit; read-modify-write records can take read locks first and
  upgrade later. This reduces read-phase blocking, but the paper finds it can
  be too optimistic for stored procedures and more useful in interactive mode.
- Read-only transactions use a dynamic strategy: start with validation-style
  invisible reads, then switch to read locks after repeated aborts. This avoids
  paying read-lock overhead when reads are cheap while still bounding starvation
  for unlucky high-contention readers.
- The latch-free locker splits lock acquisition from conflict detection. Reader
  lists can be represented with lock-free structures, and the implementation
  can compress reader membership plus an exclusive marker into an atomic word
  for platforms with limited worker counts.
- The proof argues conflict serializability by relying on three properties:
  at most one write lock holder for a record, all locks are acquired before
  unlocks within a transaction, and a transaction writes only after upgrading
  the lock to exclusive mode.
- Evaluation uses DBx1000 with YCSB and TPC-C, comparing NO_WAIT, WAIT_DIE,
  WOUND_WAIT, Silo, MOCC, TicToc, and Plor. The machine has two 18-core Intel
  Xeon Gold 6240M CPUs, 192 GiB DRAM, and Optane DCPMM modules; interactive
  mode uses two machines and eRPC over 100 Gbps InfiniBand.
- On high-contention stored-procedure YCSB-A, Plor reports throughput close to
  Silo/TicToc while reducing 99.9th percentile latency by 8.8x to 14.5x at
  relevant operating points. In interactive mode with delayed write-lock
  acquisition, it reports up to 2x higher throughput on YCSB-A and lower tail
  latency on TPC-C saturation points.

**GPU DB mapping:** Plor is most useful for the mutation owner and admission
design. The current architecture already separates immutable read snapshots
from mutable owner state; Plor adds a policy for high-contention mutation:
register conflict metadata early, delay expensive conflict action until a
bounded commit/publish boundary, and give older or repeatedly retried work a
clear path to completion.

For the 1M logical-session target, this argues against a pure best-effort
retry model at the protocol edge. If many sessions issue conflicting writes or
refresh-invalidating commands, naive OCC-style abort/retry can consume worker
cycles and amplify queue delay. A Plor-like write path could carry an arrival
generation, retry count, and read/write-intent summary through the command
ring so the mutation owner can prevent repeated starvation without pinning an
OS thread per session.

The private-buffer rule maps cleanly to WAL-before-visibility. Transactional
writes should stage row, index, and resident-invalidation effects privately
until commit. At commit, the owner can resolve conflicts, append/flush WAL,
publish CPU-visible MVCC state, and invalidate or publish resident snapshot
generations. The Plor ordering is not a replacement for WAL; it is a way to
decide who is allowed to reach the WAL/visibility boundary next.

Delayed write-lock acquisition is a candidate for interactive GPU DB sessions
where network round trips, SQL evaluation, or GPU refresh work make early
exclusive ownership expensive. It is riskier for short stored-procedure-like
mutations because many requests can reach commit together and trigger abort
storms. That distinction matches the engine's need for route-specific policy:
single-row writes, bulk COPY chunks, refresh transactions, and long DDL-like
operations should not share one lock timing rule.

The latch-free lock lesson maps to compact conflict metadata in owner-local
tables or partition-local rings. A first implementation should not copy Plor's
63-worker atomic bitset blindly, but it should measure whether owner-local
fixed arrays, generation counters, and compact reader/writer summaries can
replace heap-allocated wait queues on hot records or hot partitions.

Plor's dynamic read-only behavior also complements retained GPU snapshots.
Most read-only retained routes should remain invisible and lock-free by
holding immutable generations. If a read repeatedly fails because the target
generation is invalidated or a hot write keeps moving the visibility frontier,
it can escalate into a visible read-intent path or choose an older compatible
snapshot rather than spinning through owner retries.

**Risks and mismatches:** Plor is a single-node in-memory transaction protocol,
not an MVCC GPU database and not a PostgreSQL-compatible implementation. It is
evaluated in DBx1000 with row-oriented storage and does not address GPU
residency, HBM pressure, snapshot generations, or SQL planner route choice.
The protocol optimizes tail latency partly by increasing some non-tail
latencies, which may or may not fit workloads with strict p50/p95 goals.
Delayed write-lock acquisition can collapse under stored-procedure workloads
when too many transactions reach commit optimistically. The lock metadata is
worker-count-oriented, so it does not directly scale to 1M logical sessions;
session-level metadata must be multiplexed through bounded workers. The paper
uses read-committed behavior for TPC-C Stock-Level, so weaker isolation cases
must be kept separate from the engine's serializable or snapshot guarantees.

**Benchmark candidates:**

- Add a mutation-owner conflict-priority simulator: fixed worker count,
  1M logical session ids, hot-key Zipfian writes, staged updates, and
  older-first retry priority. Gate: bounded retries and lower p99.9 latency
  without violating WAL-before-visibility ordering.
- Compare three write-admission policies for hot rows or partitions:
  immediate exclusive ownership, OCC-style validate-at-commit, and Plor-like
  intent registration with commit-time priority. Required metrics: throughput,
  abort count, p50/p95/p99.9, owner queue time, and wasted CPU/GPU work.
- Prototype staged mutation buffers that collect row changes, index updates,
  and residency invalidations before commit. Proof gate: no staged state is
  visible before durable WAL publication; failure condition: any stale resident
  snapshot can observe a staged write.
- Measure delayed write-lock acquisition only for interactive or long
  transactions: pgwire multi-statement transactions, bulk COPY chunks, and
  refresh-invalidating writes. Negative gate: if abort storms erase throughput
  or widen tail latency, use early ownership for that route family.
- Add dynamic read-only escalation telemetry. Retained reads start invisible on
  immutable snapshots; after repeated invalidation or conflict retries, record
  when they escalate to visible read intent, older-snapshot routing, CPU
  fallback, or explicit overload.
- Benchmark compact conflict metadata: owner-local arrays or bitsets versus
  heap wait queues for hot keys. Measure cache misses, allocation count,
  commit scan cost, and maximum practical worker count.

### 2026-06-03 - Cross-paper synthesis: admission needs explicit winners

The last three modern reviews, AnKerDB, Lero, and Plor, converge on a useful
design rule: the fast path needs explicit boundaries where the system chooses
what is allowed to win. AnKerDB's virtual snapshots make visibility roots and
active holders explicit. Lero makes route selection a bounded ranking problem
after deterministic gates. Plor makes conflict priority explicit at commit
instead of letting repeated aborts decide tail latency accidentally.

The design track that follows is "gate, rank, then publish." A GPU DB request
should first pass correctness gates: WAL boundary, snapshot compatibility,
resident generation validity, memory budget, and queue capacity. Then an
optimizer or scheduler can rank admissible routes or prioritize conflicting
transactions. Finally, a commit, snapshot, or route publication boundary makes
the decision visible. Learned ranking, high-frequency snapshots, and hybrid
locking are all dangerous if they can bypass those boundaries.

The category gap after this batch is still multi-tier placement with concrete
cost models. Recent work has covered visibility roots, learned route ranking,
and contention priority. The next high-value paper should probably return to
tiered buffer/data placement or explicit CPU/GPU/NVMe memory economics unless
the queue surfaces a very recent MVCC/write-path paper.

Benchmark priorities:

- Implement admission telemetry that records why a route or transaction was
  gated out before any ranking or priority decision.
- Measure tail latency under hot-key writes with repeated retry priority
  versus ordinary OCC retry.
- Pair every learned or heuristic GPU route choice with the snapshot
  generation, queue-depth bucket, and memory-budget state that made it
  admissible.

### 2026-06-03 - mmap is not a buffer-pool substitute

**Citation:** Andrew Crotty, Viktor Leis, and Andrew Pavlo. "Are You Sure You
Want to Use MMAP in Your Database Management System?" CIDR 2022. Retrieved
2026-06-03 from the CIDR PDF at
`https://www.cidrdb.org/cidr2022/papers/p13-crotty.pdf`.

**Category:** Multi-tier cache / buffer management / data placement.

**Relevance tags:** explicit buffer management; OS page cache; mmap; NVMe;
page faults; TLB shootdowns; WAL safety; async IO; tier admission; larger-than-
memory execution; GPU/DRAM/NVMe placement.

**Core idea:** The paper argues that `mmap` is a poor replacement for a DBMS
buffer pool. Its apparent simplicity hides two classes of problems that map
directly to GPU DB tiering: the database gives up transactional and error
control to the operating system, and fast NVMe devices expose OS paging
bottlenecks that were less visible on older storage.

For GPU DB, the transferable idea is mostly negative but important: do not
delegate tier placement to transparent paging when correctness, tail latency,
and route planning need explicit answers. A GPU/DRAM/NVMe design needs to know
which pages, segments, snapshots, and pinned buffers are resident, dirty,
admissible, queued for IO, or safe to evict. The OS page cache may still be
useful as a cold-path helper, but it should not be the planner's source of truth
for residency or latency.

**Concrete mechanisms and findings:**

- The paper explains the mmap path as lazy virtual-to-physical mapping: the DBMS
  receives a pointer, page faults pull file contents into the OS page cache, and
  eviction requires page-table and TLB maintenance. Remote-core TLB invalidation
  creates expensive shootdowns.
- POSIX hints are not control. `madvise` can express broad patterns such as
  random or sequential access, but the OS may ignore hints and the wrong hint can
  be harmful. `mlock` pins pages but does not prevent dirty pages from being
  written back. `msync` is required to force mapped dirty ranges to storage.
- Transactional updates are awkward because the OS can flush dirty mapped pages
  independently of transaction commit. The paper categorizes workarounds as OS
  copy-on-write, user-space copy-on-write, and shadow paging. Each adds
  bookkeeping, extra copies, blocking, single-writer restrictions, or WAL replay
  complexity.
- mmap makes IO stalls implicit. A read-only query can block on an unexpected
  page fault, while a traditional buffer pool can issue explicit asynchronous
  reads through interfaces such as `libaio` or `io_uring`.
- Error handling becomes diffuse. Any code path that touches mapped memory can
  surface storage errors as `SIGBUS`, and transparent eviction means page
  checksums would need validation on every access if the DBMS wants the same
  confidence it gets from explicit reads.
- The experimental setup uses an AMD EPYC 7713 system with 512 GiB RAM, Linux
  5.11, and ten Samsung PM1733 NVMe SSDs. The authors reserve 100 GiB for page
  cache and compare mmap variants against `fio` using direct IO.
- In a larger-than-memory random-read workload over a 2 TiB SSD range, `fio`
  reaches roughly 900K reads/second, while mmap drops sharply after page cache
  eviction starts and later recovers to about half the direct-IO baseline under
  the best matching hint.
- For sequential scans, mmap performs acceptably only during initial loading on
  one SSD. With ten SSDs, the paper reports a roughly 20x gap between direct IO
  and mmap, with mmap failing to exploit the extra device bandwidth.
- The authors attribute the scaling collapse to page-table contention,
  single-threaded page eviction, and TLB shootdowns. They argue that the first
  two are potentially fixable with OS changes, but shootdowns are harder to
  avoid without deeper OS or hardware redesign.
- The conclusion is blunt: avoid mmap when the DBMS needs transactionally safe
  updates, nonblocking page-fault control, robust error handling, or high
  throughput on fast persistent storage. The paper allows narrow mmap use only
  for read-only working sets that fit in memory.

**GPU DB mapping:** This paper strengthens the case for an explicit tier manager
rather than a virtual-memory-driven cold tier. The storage architecture already
treats GPU resident state as versioned acceleration state, not correctness
truth. The same explicitness should extend to host DRAM and NVMe: pages or
segments should carry residency state, source WAL boundary, visibility boundary,
dirty status, in-flight IO state, checksum/validation status, and route
admissibility.

For over-resident GPU execution, relying on page faults to fetch cold host or
NVMe data would hide the latency source from the planner and scheduler. A query
route should know whether it will execute from HBM, pinned host memory,
ordinary host buffers, OS cache, or direct NVMe reads. That enables admission to
reject, prefetch, micro-batch, or fall back before a network worker or GPU
execution owner blocks unpredictably.

The transactional-safety discussion maps directly to WAL-before-visibility. A
future disk/NVMe tier should not expose mutable table bytes through mapped dirty
pages whose writeback the engine cannot order. Writes should stage in owner-
controlled buffers, append and flush WAL, update CPU-visible MVCC/index state,
and only then publish segment generations or schedule durable page writes. mmap
can be used for read-only immutable artifacts only if they are already safe to
lose or rebuild.

The performance results also matter for 1M logical sessions. Transparent page
faults convert cold reads into blocking events at arbitrary program counters.
That fights the runtime goal of bounded command rings, network IO workers, and
explicit queue wait telemetry. Explicit async IO lets the runtime attach cold
reads to admission budgets and completion rings instead of blocking request
handlers or owner domains.

For GPU memory placement, the paper suggests that "resident" must mean more
than "addressable." A pointer into a mapped file is not an admissible fast-path
input unless the engine can prove its physical residency and fault behavior. The
planner should use resident bytes, pending prefetch bytes, queue depth, and
expected transfer paths as first-class cost features.

**Risks and mismatches:** The paper is a position and evaluation paper, not a
complete replacement buffer-pool design. Its experiments are read-only and
storage-focused; it does not evaluate GPU execution, GPUDirect Storage,
PostgreSQL-compatible MVCC, or mixed OLTP/OLAP query plans. Some systems use
mmap successfully in narrower roles, so the right takeaway is not "never map
files" but "never let mmap become the hidden buffer manager for mutable or
larger-than-memory hot paths." The Linux and NVMe details may have evolved since
the 2022 publication, so modern `io_uring`, direct IO, and GPUDirect paths still
need fresh measurement.

**Benchmark candidates:**

- Add a tier-admission simulator that compares transparent OS-cache reads,
  explicit buffered reads, direct IO, and pinned prefetch buffers for cold
  segments. Required metrics: p50/p95/p99 latency, blocked worker time, queue
  depth, read amplification, and route misprediction.
- Implement a read-only immutable segment experiment where mmap is allowed only
  for fully built, checksum-validated cold artifacts. Gate: any page fault or
  `SIGBUS` risk must be visible as route telemetry, not hidden inside query
  execution.
- Compare explicit `io_uring`/direct-IO prefetch against OS readahead for
  over-resident partition scans. Minimum proof: GPU execution owners never
  block on page faults, and cold-read completion is tied to bounded response or
  execution rings.
- Add a negative-control benchmark that deliberately routes cold larger-than-
  memory reads through mmap. Expected failure: worse tail latency or lower NVMe
  bandwidth once eviction begins, validating the need for explicit tier control.
- Extend planner route features with `tier_source`, `resident_bytes`,
  `prefetch_bytes`, `faultable`, `dirty_or_mutable`, and `async_io_handle`
  fields. Gate: routes that may fault are not eligible for retained GPU
  low-latency execution.
- Test WAL ordering with mapped immutable files only: publish a segment after
  WAL and checksum completion, then prove that subsequent mutation invalidates
  the segment generation before any stale mapped bytes can be routed.

### 2026-06-03 - TPP transparent CXL page placement

**Citation:** Hasan Al Maruf, Hao Wang, Abhishek Dhanotia, Johannes Weiner,
Niket Agarwal, Pallab Bhattacharya, Chris Petersen, Mosharaf Chowdhury,
Shobhit Kanaujia, and Prakash Chauhan. "TPP: Transparent Page Placement for
CXL-Enabled Tiered-Memory." ASPLOS 2023. doi:10.1145/3582016.3582063.
Retrieved 2026-06-03 from the authors' PDF at
`https://symbioticlab.org/publications/files/tpp%3Aasplos23/tpp-asplos23.pdf`.

**Category:** Multi-tier cache / buffer management / data placement.

**Relevance tags:** CXL memory; tiered memory; page placement; fast-tier
headroom; demotion; promotion; hysteresis; page-type-aware allocation;
telemetry; memory pressure; future DRAM/CXL/HBM placement.

**Core idea:** TPP is an OS-level, application-transparent page placement
mechanism for CXL-enabled tiered memory. It assumes CXL memory behaves like a
CPU-less NUMA node with higher latency than local DRAM, and tries to keep hot
pages in local memory while moving cold pages to the CXL tier. Its strongest
transferable design pattern is fast-tier headroom management: proactively
demote colder pages before local DRAM is exhausted so new request-related
allocations and promotions of trapped hot pages have room to land.

The paper is also a useful contrast to the prior mmap review. TPP shows that
transparent OS placement can be good enough for broad datacenter memory
capacity expansion when the workload has stable warm/cold regions and the
application does not need DBMS-visible correctness boundaries for every page.
For GPU DB, that is a baseline to measure against, not a replacement for
explicit residency metadata. CXL-like host tiers may be acceptable for ordinary
host allocation and background cold data, while GPU retained routes still need
route-visible placement, validity, and queue/IO state.

**Concrete mechanisms and findings:**

- Chameleon, the paper's characterization tool, samples LLC load misses with
  PEBS and optionally store-side TLB misses. It reports page heat by virtual and
  physical page, page type, and interval history without kernel modification.
- Chameleon duty-cycles sampling across core groups and processes samples in a
  worker thread. The paper reports 3-5% of one core overhead in production
  profiling, with a synthetic all-core bandwidth workload losing about 7%.
- Production profiling finds substantial cold memory: several workloads access
  only a fraction of allocated memory in two-minute windows, and anon pages are
  often hotter than file-backed pages.
- TPP treats CXL memory as a slow NUMA-like tier. Local DRAM remains the fast
  tier; CXL memory is used for colder pages while still being byte-addressable
  and coherent.
- Demotion is integrated into Linux reclamation. Instead of swapping cold
  local pages out, TPP places reclamation candidates on a demotion list and
  migrates them asynchronously to the CXL node.
- Demotion failure is tolerated. If migration fails because the CXL node is low
  on memory, TPP skips that page because later allocation on CXL is still less
  harmful than blocking fast-tier reclamation.
- TPP decouples allocation and reclamation watermarks. Reclamation continues
  until a higher `demotion_watermark` is reached, while new allocations can
  resume at the lower allocation watermark. This keeps local DRAM headroom for
  request bursts and for promotions from CXL.
- The demotion aggressiveness is configurable through a scale factor; the paper
  gives a default in which reclamation begins when only a small percentage of
  local-node capacity is free.
- Promotion is built on NUMA balancing but limited to CXL-node pages. TPP does
  not waste hint-fault sampling on local pages that are already in the fast
  tier.
- To reduce ping-pong migration, a hinted page in CXL is promoted only after a
  hysteresis check. If the page is in the inactive LRU, TPP marks it accessed
  and moves it to active; only a later hot signal makes it a promotion
  candidate.
- Page-type-aware allocation can initially place file cache pages on the CXL
  tier while preserving ordinary allocation for anonymous pages. Hot file cache
  pages can still be promoted later.
- Observability is part of the mechanism. TPP adds counters for demoted anon
  and file pages, sampled pages, promotion attempts, successful promotions,
  promotion failures, and pages that were demoted and later became promotion
  candidates. A `PG_demoted` flag helps expose ping-pong behavior.
- The evaluation uses production workloads on pre-production CXL hardware and
  dual-socket systems configured to mimic target CXL latency. No experiment
  swaps to disk; the question is placement across memory tiers.
- In the paper's summary table, TPP is near the all-local baseline across the
  evaluated workloads, improves default Linux by up to 18%, and outperforms
  NUMA Balancing and AutoTiering by 5-17%.
- Under a constrained 1:4 local-to-CXL setup for one cache workload, TPP keeps
  throughput within about 0.5% of the all-local baseline by serving most hot
  traffic from local memory even though local memory is only a small fraction
  of the working set.
- Component analysis attributes much of the result to decoupled
  allocation/reclamation and active-LRU hysteresis. Without headroom, promotion
  can stall; with hysteresis, promotion traffic drops by 11x and demote-then-
  promote ping-pong falls materially.
- TPP and TMO are described as orthogonal: TPP can turn TMO-style swap
  offloading into a demote-then-swap process, giving pages a second chance in
  CXL before expensive swap behavior.
- The paper's future-work section notes QoS-aware tiering, bandwidth-expansion
  placement, hardware-assisted migration, and combined CXL plus network memory
  tiers as open directions.

**GPU DB mapping:** The immediate GPU DB lesson is to separate "transparent
host allocation tiering" from "DBMS route residency." TPP-style CXL placement
could eventually help ordinary host memory pressure for WAL buffers, CPU
indexes, cold immutable segments, or background snapshot state. It should not
make a retained GPU route admissible unless the DBMS can still prove source WAL
boundary, visibility boundary, resident generation, tier source, expected
latency, and fault/migration risk.

The headroom rule maps directly to the runtime's bounded queues and buffer
budgets. GPU DB should keep explicit headroom in fast tiers: HBM for retained
columns and scratch, pinned host memory for H2D/D2H staging, local DRAM for
owner hot structures and response buffers, and future CXL memory for colder
host-resident artifacts. Admission should start demotion or reject before the
fast tier reaches zero usable space, because new network requests, COPY chunks,
and retained read batches are often short-lived and latency sensitive.

TPP's promotion hysteresis is also useful for cache policy. A cold partition
should not be promoted to GPU memory or pinned DRAM on one accidental access.
The first signal can mark it warm or schedule prefetch metadata; a second signal
within a bounded interval can promote it. The same pattern can reduce churn
between GPU HBM, host DRAM, CXL memory, and NVMe segments when skew changes.

Page-type-aware allocation maps to DBMS object-type-aware placement. Instead of
anon versus file pages, GPU DB can classify WAL buffers, MVCC/version chains,
catalog state, CPU indexes, immutable column segments, compressed cold blocks,
GPU staging buffers, and encoded responses. Some classes should never start in
slow memory; others can live cold and promote only when route telemetry proves
they are hot.

The observability counters should become DBMS placement telemetry. For each
tier, the engine should report demotions, promotions, failed promotions,
demoted-then-promoted churn, bytes migrated, queue wait caused by migration,
and route decisions blocked by missing headroom. That is the difference between
using OS tiering as a hidden allocator and using it as a measured substrate.

**Risks and mismatches:** TPP is an operating-system mechanism for general
datacenter applications, not a database storage manager. It does not know SQL
visibility, WAL ordering, index validity, GPU snapshot generations, or planner
route shapes. Its best results depend on workloads with stable hot/cold regions
over minutes and on CXL latencies close to remote NUMA; GPU DB retained-route
latency targets may need microsecond-scale certainty. TPP's sampling and LRU
signals are page-granular, while GPU DB placement may need segment-, column-,
partition-, query-shape-, or snapshot-generation granularity. Finally, the
paper's evaluation is on CPU datacenter workloads, not GPU execution or
GPUDirect/NVMe paths, so its throughput percentages should be treated as
evidence for the headroom/hysteresis mechanism rather than as GPU DB forecasts.

**Benchmark candidates:**

- Add a tier-headroom simulator for HBM, pinned host memory, local DRAM, and a
  future CXL-like tier. Compare reactive eviction at zero free bytes against
  proactive demotion at a high watermark. Required metrics: p50/p95/p99 route
  latency, rejected requests, migration bytes, and fast-tier allocation stalls.
- Implement a two-signal promotion policy for resident partitions: first access
  marks warm or schedules metadata, second access inside a time/window threshold
  promotes to GPU or pinned DRAM. Failure condition: promotion churn grows under
  alternating hot/cold access.
- Add object-type placement tags to the P8 cache manager: WAL/COPY buffers,
  MVCC versions, catalog, CPU index, immutable segment, compressed cold block,
  pinned staging, encoded response. Gate: placement policy can reject slow-tier
  residency for latency-critical or correctness-sensitive classes.
- Expose placement churn telemetry modeled after TPP counters:
  demoted/promoted bytes by object type, failed promotions, demoted-then-
  promoted bytes, headroom wait time, and tier-admission rejection reasons.
- Compare OS-managed CXL/NUMA placement against DBMS-managed segment placement
  for read-only cold immutable segments. Minimum proof: identical SQL results
  and route telemetry that shows whether a query used HBM, local DRAM, CXL-like
  memory, or explicit IO.
- Build a cache-pollution benchmark where one scan-heavy route and one
  point-lookup route compete for fast-tier memory. Expected improvement:
  headroom and hysteresis protect lookup p99 without starving the scan; failure
  condition: one scan touch evicts hot lookup state.

### 2026-06-03 - ZygOS work-conserving microsecond scheduler

**Citation:** George Prekas, Marios Kogias, and Edouard Bugnion. "ZygOS:
Achieving Low Tail Latency for Microsecond-scale Networked Tasks." SOSP 2017.
doi:10.1145/3132747.3132780. Retrieved 2026-06-03 from the authors' PDF at
`https://marioskogias.github.io/docs/zygos.pdf`.

**Category:** Runtime / HFT-style mechanics / session scale.

**Relevance tags:** work-conserving scheduling; high fan-in connections;
microsecond tasks; network dataplane; head-of-line blocking; task stealing;
socket ownership; inter-processor interrupts; Silo/TPC-C; pgwire IO workers;
response rings; session multiplexing.

**Core idea:** ZygOS argues that strict shared-nothing dataplanes are excellent
for throughput but can waste latency budget when requests are short, arrivals
burst, and connections are pinned to per-core NIC queues. Its transferable
idea is to keep the dataplane virtues that matter, such as polling, per-core
network locality, and low kernel overhead, while adding a work-conserving
shuffle layer that lets idle cores steal ready work without breaking
connection-level ordering.

For GPU DB, the important lesson is not to blindly centralize all work. The
paper's design preserves a home core for each flow's TCP/IP state and sends
remote network work back to that owner. That maps well to the planned owner
domains: network IO workers may steal or drain read work, but mutation state,
socket response ordering, GPU stream ownership, and residency publication need
clear home owners and explicit handoff points.

**Concrete mechanisms and findings:**

- ZygOS separates its runtime into three layers: a lower per-core network layer,
  an intermediate shuffle layer, and an upper application execution layer. The
  lower network layer remains flow-local and mostly coherency-free.
- Each home core owns a shuffle queue containing ready connections. A ready
  connection can be consumed by the home core or stolen atomically by an idle
  remote core.
- Socket events are grouped by socket rather than by packet. A socket is in
  exactly one of `idle`, `ready`, or `busy`; when a core executes an event for
  that socket, it has exclusive access until event processing and response
  generation complete.
- The socket-state rule gives applications simple ordering semantics even when
  work is stolen. It avoids concurrent reads from the same socket producing
  broken parsing, out-of-order responses, or interleaved writes.
- Remote execution sends network-related batched system calls back to the home
  core, so TCP/IP output and timers still run where the connection state lives.
- Idle cores poll for work in several places: their own hardware descriptor
  ring, other cores' shuffle queues, other cores' software packet queues, and
  other cores' hardware descriptor rings.
- Inter-processor interrupts are used as hints to force a home core to process
  pending packets or remote network system calls while it is running user code.
  Missed interrupts hurt latency but not correctness.
- The implementation is derived from IX, Dune, DPDK, and lwIP, with about 2000
  lines of IX kernel changes and about 200 Dune changes. It is a specialized
  OS/dataplane environment, not a drop-in Linux library.
- In synthetic benchmarks with a 99th-percentile SLO of ten times mean service
  time, ZygOS reaches 75% of the ideal zero-overhead centralized-FCFS load for
  10 microsecond exponential tasks and 88% for 25 microsecond tasks.
- The paper reports that ZygOS outperforms IX and Linux for task sizes above a
  few microseconds under tight tail-latency SLOs, but IX can win on very tiny
  memcached-style tasks when adaptive bounded batching dominates the cost.
- For a networked Silo TPC-C setup with a 1000 microsecond 99th-percentile SLO,
  ZygOS sustains 344 KTPS, versus 211 KTPS for Linux and 267 KTPS for IX. The
  authors attribute the IX gap to eliminating head-of-line blocking through
  work-conserving scheduling.
- The Silo experiment disables Silo garbage collection to reduce experimental
  variability, so the result should not be read as a complete transaction-engine
  tail-latency solution.

**GPU DB mapping:** The production runtime already targets network IO workers,
bounded command rings, owner domains, read snapshot workers, GPU execution
workers, and response rings. ZygOS strengthens the case that those rings should
be work-conserving under bursty fan-in, not statically pinned in a way that lets
one IO worker queue while another sits idle.

The socket ownership state machine maps to pgwire session handling. A session
should have a single response-order owner at any instant, even if parsed
requests or read-only work are stolen into read snapshot workers. That keeps
frontend protocol ordering simple while still allowing compatible retained
reads to leave the session's home worker.

The shuffle queue maps to a "ready session" or "ready request" layer between
network parsing and execution. For 1M logical sessions, the runtime should not
create one OS thread per connection; it should keep compact session state,
place ready sessions on bounded queues, and let idle IO/execution workers steal
eligible work under explicit ordering rules.

ZygOS also warns that batching and work conservation trade off. IX's advantage
on tiny memcached tasks came from adaptive bounded batching. GPU DB should
combine both ideas: steal work to avoid idle cores and head-of-line blocking,
but drain compatible batches for retained lookups, aggregates, COPY chunks, and
encoded responses when queue depth is present and the latency ceiling allows
it.

For GPU execution owners, the home-core rule suggests keeping CUDA streams,
pinned buffers, and resident partition handles owned by a specific execution
worker. Other workers can enqueue or steal request descriptors, but device
state and completion publication should remain owned to avoid hidden
cross-thread synchronization.

**Risks and mismatches:** ZygOS is a specialized OS built on IX/Dune, DPDK, and
lwIP; GPU DB currently runs as a normal Rust/database process and cannot assume
that environment. The paper's strongest result is for microsecond in-memory RPC
tasks on one server with 10GbE-era hardware, not PostgreSQL protocol parsing,
TLS, GPU kernels, WAL, MVCC validation, or NVMe tiering. Its Silo experiment
does not include full SQL marshalling and disables garbage collection, so the
numbers are best used as scheduling evidence rather than direct throughput
targets. Inter-processor interrupts may also be the wrong primitive in a normal
process; eventfd, io_uring, futex wakeups, or busy-poll rings may be more
practical.

**Benchmark candidates:**

- Build a pgwire runtime simulator with many logical sessions, per-session
  ready state, and a stealable ready-session queue. Compare static IO-worker
  ownership against work stealing. Required metrics: p50/p95/p99 queue wait,
  idle-worker time, response reordering violations, and request throughput at
  fixed tail-latency SLOs.
- Add a retained-read micro-batch benchmark that combines ZygOS-style stealing
  with IX-style bounded batching. Gate: batching improves throughput without
  worsening p99 beyond a configured microsecond ceiling.
- Prototype a per-session state machine: `idle`, `ready`, `executing`,
  `responding`, and `backpressured`. Minimum proof: pipelined requests on one
  pgwire session never produce out-of-order or interleaved responses while
  read-only requests can still execute away from the session home worker.
- Add telemetry for each runtime ring: eligible steals, successful steals,
  failed steal attempts, home-worker wakeups, remote response handoffs, and
  work executed on non-home workers.
- Compare three admission modes for 1M logical sessions: one-thread-per-session
  baseline, fixed per-IO-worker session partitioning, and stealable ready
  sessions over compact session state. Failure condition: memory growth,
  scheduler overhead, or tail latency scales with connection count rather than
  active request count.
- For GPU execution workers, test home-owned CUDA stream queues with remote
  enqueue only. Gate: no CUDA resource is touched by non-owner workers, and
  queue wait plus batch-drain telemetry explains p99 latency under bursty
  retained lookup traffic.

### 2026-06-03 - GaccO GPU-accelerated OLTP co-execution

**Citation:** Nils Boeschen and Carsten Binnig. "GaccO - A GPU-accelerated
OLTP DBMS." SIGMOD 2022, pages 1003-1016. doi:10.1145/3514221.3517876.
Retrieved 2026-06-03 from the DFKI publication page at
`https://www.dfki.de/en/web/research/projects-and-publications/publication/14413`;
ACM DOI metadata is at `https://doi.org/10.1145/3514221.3517876`. The ACM PDF
was not directly retrievable from this worker, so mechanism details below are
based on the DFKI abstract/metadata plus indexed paper text from the public
document record.

**Category:** Transaction processing / write path, with GPU execution.

**Relevance tags:** GPU OLTP; batched stored procedures; CPU/GPU
co-execution; deterministic concurrency control; single-version GPU copy;
primary CPU copy; update propagation; same-type transaction queues; TPC-C;
NewOrder; Payment; larger-than-HBM working sets.

**Core idea:** GaccO makes the GPU useful for OLTP by refusing to send
individual tiny transactions to the device one at a time. It routes frequent,
stored-procedure transaction types to per-type GPU queues, batches only
transactions of the same type, and executes those batches with a GPU-centric
deterministic concurrency scheme. Less frequent or unsuitable transaction
types continue to run on the CPU.

The design is deliberately a co-execution system rather than a GPU-only
database. CPU memory keeps the primary copy of tables, while GPU memory holds a
secondary copy, potentially only for the subset of data needed by the
GPU-routed transaction types. That lets GaccO avoid moving whole tables during
normal execution and still handle databases larger than GPU memory by relying
on CPU memory for the full working set.

**Concrete mechanisms and findings:**

- Incoming transactions are classified by transaction type. Dominating
  transaction types, such as TPC-C `NewOrder` or `Payment`, can be routed to
  the GPU; non-dominating types run individually on the CPU.
- GPU work is grouped by a separate queue per transaction type. The same-type
  rule reduces branch divergence and makes memory access patterns less random
  than an arbitrary mixed transaction batch.
- GaccO currently assumes stored procedures and uses a static type-to-device
  routing policy. Dynamic load adaptation and ad-hoc transaction support are
  described as future extensions, not solved mechanisms.
- GPU-side execution has pre-processing, transaction-kernel execution, and
  post-processing phases. The pre-processing step orders conflicting accesses
  inside the batch so the batch can execute deterministically.
- The paper describes the GPU concurrency-control scheme as deterministic and
  abort-free for transactions inside a batch. Based on indexed paper text and
  later Epic discussion, this is a single-version deterministic locking style,
  closer to Calvin-like ordered access than to general MVCC.
- CPU tables are multi-versioned, while GPU tables are single-versioned in the
  architecture diagram. CPU/GPU co-execution therefore requires update
  propagation and isolation coordination between the CPU primary copy and GPU
  secondary copy.
- The secondary GPU copy may cover only part of the database, allowing
  larger-than-device-memory workloads when the GPU-routed transaction types can
  operate on resident subsets and updates are propagated.
- The evaluation uses TPC-C variants and reports up to 6x throughput speedup
  over a CPU-only OLTP engine. The paper also claims batched GPU latency is in
  the millisecond range, which it argues is tolerable for some OLTP
  applications.
- Unknown from the accessible text in this run: exact lock-table layout,
  precise conflict ordering algorithm, exact update-propagation barrier,
  durability protocol, hardware configuration details, and full latency
  distribution tables.

**GPU DB mapping:** GaccO is the closest reviewed source so far to "GPU DB as
an OLTP accelerator" rather than "GPU DB as an analytical scan engine." Its
strongest transferable idea is the same-shape transaction lane: choose a small
number of high-frequency request or transaction templates, give each a bounded
queue, and batch only templates whose access pattern, output shape, and
visibility boundary match. That maps directly to retained point lookups,
filtered aggregates, COPY chunk admission, and maybe a future stored-procedure
write path.

The CPU-primary/GPU-secondary storage split is already aligned with P8. WAL,
checkpoint, MVCC tuple state, and CPU indexes remain the durable authority;
GPU-resident layouts are secondary acceleration state. GaccO strengthens the
case that GPU-resident transaction acceleration should not require the whole
database in HBM. The first production-safe variant for GPU DB should be
resident partitions or resident key ranges with explicit update propagation and
route rejection when the needed partition is absent or stale.

GaccO's static stored-procedure routing is useful as a first benchmark shape
even if it is too narrow for final SQL service. GPU DB can start with declared
route templates: retained lookup by key, retained filtered aggregate, and
possibly a single deterministic update template. Each template should publish
its exact read/write set requirements, resident layout, visibility boundary,
and batch compatibility key.

The deterministic batch idea is attractive for writes, but it should be kept
behind WAL-before-visibility. A GPU write batch should not make external
visibility decisions on the device first. A safer adaptation is: CPU mutation
owner admits and WAL-logs a deterministic chunk, GPU computes conflict-free or
ordered effects for eligible resident partitions, CPU owner validates the
device result against the admitted generation, then publishes the visibility
boundary and invalidates or refreshes resident snapshots.

For reads, GaccO's same-type queues reinforce the runtime design from ZygOS and
Shinjuku: work conservation matters, but compatibility boundaries matter more.
Stealing a ready session is safe only if the worker preserves session response
ordering and routes GPU work into the correct same-shape, same-generation batch
lane.

**Risks and mismatches:** GaccO is optimized for stored-procedure OLTP, not
arbitrary SQL planning. The GPU table copy is single-versioned, while the
current engine's correctness model is MVCC with WAL-before-visibility and
invalidated immutable retained snapshots. Millisecond-scale GPU batch latency
may be too high for short point reads or latency-sensitive sessions even when
aggregate throughput is excellent. The static routing policy can misroute work
when contention, residency, or queue depth changes. Finally, the paper's
accessible text does not expose enough detail to treat its lock-table and
update-propagation algorithm as implementation-ready; use it as a benchmark
shape and architecture contrast, not as a drop-in concurrency protocol.

**Benchmark candidates:**

- Build a same-template GPU batch-lane simulator for retained reads and one
  synthetic stored-procedure write template. Compare mixed request batching
  against per-template queues. Metrics: throughput, branch-divergence proxy,
  queue wait, p95/p99 latency, and rejected incompatible requests.
- Add a route-admission experiment with CPU-primary/GPU-secondary partition
  residency. Gate: GPU route is allowed only when table id, partition id,
  source WAL boundary, visibility boundary, and resident layout generation all
  match.
- Prototype a deterministic write-batch proof that logs on the CPU before
  device execution and publishes visibility only after CPU validation of the
  GPU result. Failure condition: any path exposes a GPU-computed write before
  WAL durability and invalidation are complete.
- Compare batch-size triggers for same-type retained lookups: count threshold,
  microsecond threshold, and dual threshold. Required result: throughput gain
  without pushing p99 past the configured latency ceiling.
- Add a larger-than-HBM OLTP residency test where only hot partitions for one
  transaction template are resident on the GPU. Measure route hit rate, update
  propagation bytes, CPU fallback rate, and latency when the working set moves.
- Evaluate static template routing against telemetry-aware routing that can
  fall back to CPU when GPU queue depth, resident freshness, or partition
  absence makes the batch route fragile.

### 2026-06-03 - Cross-paper synthesis: batch lanes need visibility fences

The recent reviews of mmap, TPP, ZygOS, and GaccO converge on one practical
track: high-throughput GPU DB needs narrow, explicit lanes rather than broad
implicit magic. mmap warns against hiding database residency behind OS fault
behavior. TPP shows that tiering works best with headroom, hysteresis, and
promotion/demotion telemetry. ZygOS shows that session work should be
work-conserving without losing per-socket ownership. GaccO shows that GPU OLTP
only becomes plausible when work is batched by compatible transaction shape and
coordinated with a CPU primary copy.

The resulting design track is "same-shape, same-generation lanes." A retained
route should enter a batch only when its template, partition, snapshot
generation, output shape, tier source, and latency budget match. For writes,
the equivalent batch lane must add a WAL and visibility fence: CPU owner
admits, orders, and logs; GPU may accelerate deterministic effects; CPU owner
publishes visibility and resident invalidation after validation.

Category gaps remain around modern GPU-aware transaction protocols and
end-to-end SQL protocol admission under very high logical session counts. The
next useful papers should include GPU OLTP/MVCC follow-ups, deterministic
contention management, and high-concurrency networking/admission sources, not
only analytical GPU execution.

Benchmark priorities:

- Same-template lane benchmark for retained lookups, aggregates, and one
  deterministic write template, with queue wait and batch drain telemetry.
- Visibility-fenced GPU write-batch proof that refuses to publish before WAL,
  invalidation, and CPU validation complete.
- Tier-aware route admission test that rejects when HBM/host headroom or
  resident generation is insufficient.
- Work-conserving session scheduler benchmark that can steal ready work while
  preserving per-session response ordering and batch-lane compatibility.

### 2026-06-03 - BaM GPU-initiated storage access

**Citation:** Zaid Qureshi, Vikram Sharma Mailthody, Isaac Gelado, Seungwon
Min, Amna Masood, Jeongmin Park, Jinjun Xiong, C. J. Newburn, Dmitri
Vainbrand, I-Hsin Chung, Michael Garland, William Dally, and Wen-mei Hwu.
"GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System
Architecture." ASPLOS 2023, pages 325-339. doi:10.1145/3575693.3575748.
Retrieved 2026-06-03 from arXiv at `https://arxiv.org/abs/2203.04910`; author
page available at `https://mgarland.org/papers/2022/bam/`.

**Category:** Multi-tier cache / data placement, with GPU execution and
over-resident storage access.

**Relevance tags:** GPU-initiated IO; GPUDirect storage; NVMe queues; GPU
software cache; cache-line coalescing; warp coalescing; clock replacement;
queue-depth sizing; over-resident execution; CPU orchestration avoidance;
write-back caveats; GPU/CPU consistency boundary.

**Core idea:** BaM moves the storage control path closer to the GPU. Instead
of having CPU code tile a dataset, service GPU page faults, or launch repeated
copy/compute phases, GPU threads can issue fine-grained on-demand requests to
NVMe-backed data through GPU-resident submission/completion queues. A GPU
software cache coalesces redundant requests and gives kernels a memory-like
array API.

The key observation is that GPUs have enough thread-level parallelism to hide
storage latency if the IO path can keep enough requests in flight. BaM applies
Little's Law directly to the storage path: target bandwidth times latency
determines required queue depth, and that queue depth can be distributed
across many device queues and GPU threads.

**Concrete mechanisms and findings:**

- BaM maps NVMe submission queues, completion queues, IO buffers, and doorbell
  registers into GPU-visible memory/address space using a custom Linux driver,
  GPUDirect RDMA, and GPUDirect Async-style doorbell mapping.
- GPU threads access data through `bam::array<T>`. The abstraction computes a
  cache-line offset, lets warp threads coalesce accesses to the same cache
  line, probes GPU cache metadata, and on miss submits storage requests.
- The queue algorithm avoids one giant critical section for GPU thread
  submission. Each queue has local head/tail copies, an atomic ticket counter,
  per-entry turn counters, a mark bit-vector, and a lock used only to advance
  contiguous submitted entries and ring the doorbell. This batches expensive
  PCIe doorbell writes while allowing many threads to prepare entries in
  parallel.
- Completion queues are polled by GPU threads without a lock for the lookup
  phase. Mark bits and a similar head-advance routine publish cleanup progress
  and release SQ entries for reuse.
- The software cache preallocates virtual and physical backing memory at
  startup. A cache miss locks the line, finds an eviction victim, fetches the
  backing line, then marks it valid and increments a reference count. Other
  threads requesting the same line wait rather than issuing redundant IO.
- Eviction uses a clock-style global counter so concurrent evictors are
  assigned different candidate slots. Pinned cache lines with nonzero reference
  counts are skipped.
- Warp coalescing uses CUDA warp primitives so only one leader per same-line
  group manipulates cache metadata; the leader broadcasts the resulting GPU
  address to its group.
- BaM can reach peak IOPs per SSD and scale linearly over the tested Optane
  SSDs. The paper reports 45.8M random read IOPs and 10.6M random write IOPs
  with ten Optane SSDs for 512-byte accesses, about 22.9 GB/s random-read
  bandwidth and 90% of measured PCIe Gen4 x16 bandwidth.
- Compared with NVIDIA GDS in the paper's sequential benchmark, BaM saturates
  the GPU PCIe link at 4KB granularity, while GDS needs much larger IO
  granularity to hide CPU/Linux stack overhead.
- For graph analytics on BFS and connected components, BaM with four Optane
  SSDs is reported as on par with or faster than an optimistic host-memory
  target once end-to-end file loading is included: 1.0x for BFS and 1.49x for
  connected components.
- For data analytics over NYC taxi queries, BaM avoids transferring entire
  columns when later query stages use data-dependent subsets. The paper reports
  up to 5.3x speedup over a RAPIDS baseline with the dataset pinned in the CPU
  page cache.
- Cache capacity is not always the dominant knob. On one graph dataset, 1GB
  cache performs similarly to 8GB because locality is still captured; queue-pair
  count only begins to hurt around 40 or fewer queue pairs in the tested setup.
- Writes exist in the stack as write requests, dirty cache lines, and flush
  APIs, but BaM's programming model leaves crash consistency, checkpointing,
  and CPU/GPU shared-data synchronization to the application. The vector-add
  write-heavy workload is slower than a tiled baseline because the prototype
  does not yet overlap read-miss handling with write-back.

**GPU DB mapping:** BaM is a strong source for the P8 over-resident path, but
not as a transactional storage engine. The transferable mechanism is an
explicit GPU-side cold-tier request lane: retained kernels should be able to
request missing cold blocks, cache them in GPU memory, and expose queue-depth,
cache-hit, coalescing, and doorbell/submit telemetry instead of hiding IO
behind CPU page faults.

For GPU DB, the safe unit is not an arbitrary byte range. It should be a
database-owned resident block: table id, partition id, column group, source WAL
boundary, visibility boundary, encoding generation, and checksum. BaM's
cache-line concept maps to this block only if each block is immutable for the
reader's snapshot. Otherwise, BaM's application-managed consistency is too weak
for MVCC.

BaM's SQ/CQ design is useful for future GPU execution owners. A GPU owner could
own one or more storage queues, pinned staging buffers, cache metadata, and
device handles. Other runtime workers would enqueue logical cold-block
requests, while the GPU owner batches and coalesces actual device queue
operations. This preserves the owner-domain rule and makes doorbell writes,
queue depth, and cache pressure measurable.

The cache-miss coalescing maps directly to over-resident partition access.
Multiple retained queries or warp lanes that need the same cold column block
should wait on one in-flight fill rather than submitting duplicate reads. That
suggests an explicit state machine for GPU DB block residency: `absent`,
`fetching`, `valid`, `pinned_by_readers`, `dirty_or_mutated`, `invalidated`,
and `evicted`.

BaM also reinforces that GPU DB should benchmark IO granularity. The P8 design
currently thinks in resident columns and partitions; BaM shows that the
practical granularity tradeoff is more subtle. Too-small blocks increase
metadata, atomics, and queue pressure; too-large blocks create IO
amplification and waste HBM. GPU DB should measure 512B, 4KB, 16KB, 64KB, and
column-run blocks for point lookups, filtered aggregates, and sparse
over-resident scans.

**Risks and mismatches:** BaM's prototype requires a custom driver, direct NVMe
queue mapping, GPUDirect features, root/system integration, and security
assumptions that may not fit a normal database process. The paper's best
results are for graph and analytical access patterns, not OLTP writes, WAL
flush, MVCC validation, SQL result marshalling, or PostgreSQL protocol latency.
Its write path is application-consistent rather than database-consistent, and
the authors explicitly leave crash guarantees and CPU/GPU sharing
synchronization to applications. GPU DB cannot adopt that model for committed
data.

BaM's cache API overhead can become significant once storage ceases to dominate,
especially from metadata contention, atomics, and polling warp scheduling. The
paper also shows that workloads with insufficient frontier/request parallelism
cannot hide storage latency well. That matters for selective SQL point reads:
GPU-initiated IO only helps when a batch lane has enough independent cold-block
misses or useful compute to overlap.

**Benchmark candidates:**

- Build an over-resident retained-read simulator with database block ids rather
  than raw byte offsets. Gate: duplicate misses for the same block coalesce into
  one in-flight fetch, and every returned block matches table/partition/WAL
  boundary/snapshot generation metadata before it can be read.
- Compare CPU-orchestrated cold-block fetch, OS/UVM-style fault simulation, and
  explicit GPU-owner queue submission for sparse retained scans. Metrics:
  queue depth, coalesced miss count, IO amplification bytes, H2D/DMA bytes,
  p95/p99 latency, and CPU owner time.
- Add a block-size sweep for over-resident column groups: 512B, 4KB, 16KB,
  64KB, and compressed column-run blocks. Failure condition: larger blocks
  improve throughput only by hiding excessive IO amplification or stale
  visibility assumptions.
- Prototype a `fetching` residency state with reader pin counts and
  invalidation generation. Gate: mutation invalidation cannot mark a block
  route-valid for new readers while a stale fetch completion is racing.
- Measure GPU cache metadata overhead separately from storage time by running
  hot-cache retained point lookups through the same cache API. Gate: hot-cache
  metadata overhead stays below a fixed percentage of kernel time, otherwise
  GPU DB should prefer pre-published resident snapshots for hot OLTP reads.
- Evaluate whether a storage-backed route should be admitted only when a
  same-shape batch can provide enough queue depth. Failure condition:
  per-request cold misses create higher p99 than CPU fallback for point
  lookups.
- For future write experiments, require WAL-before-visibility and CPU
  validation around any GPU write-back cache. Gate: GPU dirty blocks cannot be
  externally visible or durable-authoritative without WAL, replay metadata, and
  crash recovery proof.

### 2026-06-03 - ParamTree learned cost-model calibration

**Citation:** Jiani Yang, Sai Wu, Dongxiang Zhang, Jian Dai, Feifei Li, and
Gang Chen. "Rethinking Learned Cost Models: Why Start from Scratch?" Proc. ACM
Manag. Data 1(4), Article 255, SIGMOD 2023. doi:10.1145/3626769. Retrieved
2026-06-03 from `https://15799.courses.cs.cmu.edu/spring2025/papers/15-learned/yang-sigmod2023.pdf`;
metadata cross-checked through DBLP and DOI.

**Category:** Query optimization / planning.

**Relevance tags:** learned cost model; formula-based optimizer calibration;
hardware-aware route costing; dynamic workload refinement; explainable
planning; online tuning; transferability; CPU/GPU route choice; fallback cost;
benchmark Q-error caveat.

**Core idea:** The paper argues against replacing a conventional optimizer
cost model with a fully learned black box when the existing formulas already
encode useful system knowledge. ParamTree keeps the DBMS formula templates and
learns the hidden hyperparameters inside those formulas for each hardware,
software, data, and workload context.

The practical shift is from "learn query plan to latency from scratch" to
"learn which formula parameters should change in this environment." That makes
the model lighter, easier to transfer, and more explainable because the final
cost is still produced by named optimizer terms such as tuple CPU cost, operator
CPU cost, index tuple cost, sequential page cost, and random page cost.

**Concrete mechanisms and findings:**

- The authors separate parameters into R-params, the tunable weights inside a
  formula-based cost model, and C-params, the context variables that influence
  those weights.
- Static C-params include hardware, OS, storage, DBMS, and restart-required
  configuration choices. Dynamic C-params include physical operator type, query
  structure, data type, index correlation, column position, work memory, temp
  buffers, and other runtime-sensitive configuration values.
- ParamTree builds one decision tree per physical operator. Each leaf owns a
  subspace of C-params and a regression-derived set of R-params for that
  operator's cost formula.
- Because explicit labels for the correct R-params are unavailable, the tree
  uses observed query runtime and vectorized cost-formula terms. Leaf R-params
  are fitted by least squares against observed execution cost.
- Node splitting uses parameter-instability tests to find C-params whose
  changes make the fitted R-params unstable. Numeric parameters use a supLM
  style test; categorical parameters use a chi-square style test.
- Offline training builds initial trees from diverse hardware/software
  configurations. Online refinement maintains a buffer of poorly estimated
  queries and expands the relevant leaf when enough queries exceed an error
  threshold.
- Online expansion ranks candidate dynamic C-params using a few-shot response
  surface model and Sobol-style sensitivity analysis, then generates targeted
  samples from query templates rather than sampling blindly from the whole query
  space.
- For costing a physical plan, ParamTree recursively obtains operator-specific
  R-params from the relevant tree and fills the native formula with plan
  statistics, preserving the optimizer's existing plan-cost structure.
- The evaluation uses PostgreSQL 13.3 by default, 20 varied cloud instances for
  generalization, IMDB/JOB, TPC-H, and TPC-DS workloads, with parallel
  execution disabled for stability.
- With exact cardinalities, ParamTree reports median Q-error below 1.11 on
  IMDB job-light and 1.15 on IMDB scale; with DeepDB cardinality estimates, it
  still outperforms the compared learned cost predictors in the reported setup.
- Operator-level results show large median absolute error reductions for
  operators such as Sort, Aggregate, HashJoin, IndexScan, and IndexOnlyScan
  compared with the tuned PostgreSQL cost model.
- ParamTree's transfer experiments report better Q-error than scaled/tuned
  PostgreSQL and a zero-shot learned model across four held-out cloud machines.
  It also transfers across multiple databases with reported Q-error below 1.92.
- Online refinement reaches strong accuracy with relatively few samples in the
  dynamic query experiment: the paper reports mean Q-error 1.26 after 350
  generated samples for the exact-cardinality case.
- Training overhead is much smaller than the tested neural plan encoders in the
  paper's setup: ParamTree training is reported at about 272 seconds versus
  1249 seconds for TCNN and much higher for E2E/QueryFormer.
- Inference overhead is intended to be small because each tree has controlled
  height, reported below 10 nodes, and the output still feeds simple formulas.

**GPU DB mapping:** ParamTree maps well to the planner-cost contract for GPU DB
because the engine should not hide route choice behind an opaque learned
planner. The first GPU route model can stay formula-based and expose terms for
CPU tuple/index cost, GPU launch cost, H2D/D2H bytes, resident snapshot
validity, cold-block fetch cost, decompression cost, result scattering, queue
wait, fallback penalty, and invalidation risk. A ParamTree-style layer can then
calibrate the weights for those terms by hardware and workload.

The useful adaptation is per-route, per-operator calibration. A retained point
lookup, resident aggregate, cold over-resident scan, CPU fallback, and
GPU-assisted join should each have its own parameter tree or bounded
calibration table. Their C-params should include GPU model, HBM capacity,
PCIe/NVLink bandwidth, CUDA stream policy, pinned-buffer budget, resident
generation age, queue depth, batch size, result row count, transfer bytes,
encoding family, table/partition hotness, and snapshot invalidation frequency.

ParamTree also fits the owner-domain model. The planner can ask a route-cost
service for calibrated R-params, but the route service should publish immutable
calibration snapshots rather than mutate planning state in the middle of a
query. New observations from completed queries can flow into an online
refinement buffer owned by a planning/calibration worker. When the worker
expands or updates a calibration tree, it publishes a new generation with
telemetry and rollback ability.

For session concurrency, the paper's online-buffer trigger suggests a
low-noise way to improve route choice without per-request learning overhead.
Only outlier route predictions should enter a bounded refinement buffer.
Admission should continue to use the current calibration generation until a new
one is published. This prevents 1M logical sessions from paying model-training
or lock contention in the hot path.

The cost model should optimize route decisions, not just latency prediction.
GPU DB should measure whether calibrated costs pick better CPU/GPU/fallback
routes under load, stale residency, and tier pressure. A low Q-error model that
still admits the wrong GPU route when a snapshot is invalid or a queue is
saturated is not useful.

**Risks and mismatches:** The paper is about CPU DBMS cost estimation, not GPU
execution, MVCC, WAL ordering, or high-concurrency protocol serving. It assumes
physical plans and operator formulas already exist; GPU DB still needs the
first formula terms for route validity, residency, queueing, and transfer cost.
It also evaluates cost-estimation accuracy, mostly through Q-error, rather than
end-to-end optimizer regret or tail latency under admission pressure.

ParamTree's online refinement requires executing generated sample queries. That
can be expensive or disruptive in a production GPU DB, especially when samples
touch cold tiers or consume scarce HBM. Sampling must be isolated, rate-limited,
or restricted to benchmark/control-plane windows. Exact-cardinality results
are not production-realistic unless the engine invests in statistics and
cardinality estimation; bad cardinality can still dominate route mistakes.

The tree can explain which C-params matter, but it does not automatically
enforce database invariants. Route calibration must be downstream of hard
validity checks: WAL boundary, snapshot generation, resident layout identity,
memory budget, and queue capacity. A learned or calibrated low cost must never
override a failed validity proof.

**Benchmark candidates:**

- Build a formula-based CPU/GPU route model with explicit terms for launch
  cost, transfer bytes, resident validity, queue wait, result scattering,
  cold-block fetch, and CPU fallback. Gate: every term is logged per route
  decision and can be replayed against observed latency.
- Add a ParamTree-style offline calibration experiment using synthetic retained
  lookup, aggregate, and cold-scan templates across batch sizes and residency
  states. Compare default weights, manually tuned weights, and learned
  per-route weights.
- Measure route-choice regret, not only Q-error: for each query template, run
  CPU, GPU resident, GPU cold-transfer, and fallback routes when legal, then
  score whether the calibrated planner chose the lowest-latency legal route.
- Add an online refinement buffer for bad route predictions in the benchmark
  harness. Gate: refinement is bounded, off the hot path, generationed, and
  never changes the route choice for already admitted queries.
- Include invalidation and queue-pressure C-params. Failure condition: the
  calibrated model admits GPU work to a stale resident generation or saturated
  queue because predicted compute time is low.
- Compare template-based sampling against random route sampling. Required
  result: fewer samples to calibrate the retained lookup and aggregate lanes
  without overfitting one data distribution.
- Track whether better prediction improves p95/p99 latency and rejection
  correctness under mixed session load. A model that lowers Q-error but
  increases overload, fallback churn, or p99 route wait should be rejected.

### 2026-06-03 - Arachne core-aware thread management

**Citation:** Henry Qin, Qian Li, Jacqueline Speiser, Peter Kraft, and John
Ousterhout. "Arachne: Core-Aware Thread Management." OSDI 2018, pp. 145-160.
Retrieved 2026-06-03 from
`https://www.usenix.org/system/files/osdi18-qin.pdf`; USENIX page:
`https://www.usenix.org/conference/osdi18/presentation/qin`.

**Category:** Runtime / HFT / session scale.

**Relevance tags:** user-level threads; core-aware scheduling; core arbiter;
exclusive cores; request-granular workers; cache-miss budgeting; thread
creation at request granularity; load factor; hysteresis; performance
isolation; 1M logical sessions; pgwire IO workers; owner-domain scheduling.

**Core idea:** Arachne argues that low-latency services should negotiate over
physical cores, not invisible OS threads. The application should know exactly
which cores it owns, decide how its own short-lived user threads are placed on
those cores, and report changing core demand to a user-space core arbiter.

This is a useful middle ground for GPU DB's runtime target. It does not require
making every logical session an OS thread, and it does not require turning the
whole database into a single event loop. Instead, it treats thread creation as
cheap enough for microsecond-scale request work while keeping physical core
budgets explicit and observable.

**Concrete mechanisms and findings:**

- Arachne uses one kernel thread per allocated core and multiplexes
  application-visible user threads on top of those core-bound kernel threads.
- A separate user-level core arbiter allocates specific cores to applications
  through Linux cpusets. Managed cores are dedicated to Arachne applications;
  unmanaged cores remain available to ordinary Linux-managed threads.
- Applications request cores at priority levels. The arbiter allocates cores
  from high to low priority and asks applications to release cores
  cooperatively; if needed, it can forcibly reclaim after a timeout.
- Communication from applications to the arbiter uses sockets because core
  allocation changes are infrequent. Arbiter-to-application release requests
  use shared memory because the runtime checks them frequently in the
  dispatcher loop.
- The runtime is cooperative, not preemptive. User threads are expected to
  block or finish quickly; blocking kernel calls and page faults are not deeply
  handled.
- Thread context is bound to a single core and reused, so hot stacks and
  runtime metadata often stay in that core's cache.
- Thread creation combines load balancing and context allocation in one shared
  64-bit `maskAndCount` value per active core: 56 bits track occupied thread
  contexts and 8 bits track the count.
- New user threads are placed with the "power of two choices": sample two cores,
  choose the one with fewer active contexts, then reserve a context with CAS.
- The runnable signal, entry address, and small argument list are packed into a
  single cache line. The paper frames cross-core thread creation as a four-cache
  miss operation.
- Arachne avoids ready queues. The dispatcher scans active contexts on its core
  and tests a `wakeupTime` word; a thread is runnable when `wakeupTime` is less
  than or equal to the cycle counter. Wakeup sets `wakeupTime` to zero.
- The default core policy has `exclusive` and `normal` classes. Exclusive
  threads get dedicated cores for long-running pollers; normal threads share a
  disjoint worker-core pool.
- Core estimation uses load factor to scale up and utilization with hysteresis
  to scale down. The paper's default parameters are a 1.5 load-factor threshold,
  50 ms averaging interval, and 9% scale-down hysteresis.
- The reported median primitive costs include cross-core Arachne thread
  creation around 320 ns with hyperthreads active, condition notify around
  272 ns, signal around 254 ns, and thread exit turnaround around 449 ns.
- Memcached-A replaces static worker assignment with request-granular Arachne
  threads. The paper reports 37.5% higher SLO-compliant throughput at median
  latency below 100 us, and 99th-percentile latency 3-40x lower than
  memcached in the tested setup.
- Under dynamic load and colocation with x264, memcached-A uses fewer cores at
  low load, ramps up as load rises, and is almost unaffected by the background
  application because the arbiter gives it dedicated cores.
- The no-arbiter variant performs much worse because Linux can deschedule a
  kernel thread while Arachne continues assigning user threads to it. Dedicated
  cores are therefore not an incidental optimization; they are part of the
  correctness of the latency model.
- RAMCloud-A yields during nested RPC polling and schedules other requests
  during microsecond-scale wait gaps. The paper reports 2.5x higher single-
  server write throughput for 100-byte-object writes, but a 15% lower read-only
  YCSB-C throughput because Arachne's thread invocation/exit overhead exceeds
  the benefit when there is little waiting to hide.
- The paper lists unexplored areas around NUMA policies, reusable core policies,
  and whether the chosen core-estimation parameters generalize.

**GPU DB mapping:** The strongest transfer is not "use Arachne as a library."
It is the resource contract: GPU DB's runtime should expose physical resources
and queue classes to the scheduler instead of hiding them behind one OS thread
per client or one generic work queue. Logical sessions can be virtual, but IO
workers, mutation owners, read-snapshot workers, residency workers, and GPU
execution owners need explicit CPU-core and GPU-stream budgets.

For the current high-throughput runtime design, this supports a split between
exclusive poller/owner lanes and normal request lanes. Network IO workers,
mutation owners, residency owners, and GPU execution owners are closer to
Arachne's exclusive class because they poll sockets, command rings, CUDA
events, or residency queues. Short retained reads, response encoding, and CPU
fallback fragments can be normal request classes placed over a bounded worker
pool.

Arachne's `maskAndCount` and queueless dispatcher suggest a GPU DB experiment:
for very short same-shape read tasks, ready-queue machinery can cost more than
the work. A small fixed pool of per-core task contexts, a compact runnable word,
and power-of-two placement may be enough for CPU-side retained-read dispatch,
response encoding, and GPU completion callbacks. The important part is to
measure cache-line movement, not only request counts.

Core estimation maps to admission. GPU DB should not only count open sessions;
it should estimate active runnable work per class. A 1M-session target should
admit many parked protocol sessions while scaling physical execution lanes from
recent runnable load, queue delay, GPU queue saturation, and response-ring
pressure. Session count is a capacity metric; runnable load is the scheduling
metric.

The RAMCloud result is especially relevant to writes and multi-stage queries.
If a mutation waits briefly on WAL flush, replication, fsync, CUDA event,
or cold-block fetch completion, the owning core should not spin uselessly when
other safe work exists. But the YCSB-C read penalty is a warning: for hot
read-only retained lookups, extra task creation can be a regression unless it
amortizes a real wait, batch, or route-selection cost.

**Risks and mismatches:** Arachne is a runtime paper, not a database
concurrency-control or storage paper. It has no MVCC visibility model, WAL
ordering, snapshot publication protocol, crash recovery, SQL correctness, or
GPU residency semantics. Its cooperative user threads assume short,
well-behaved work; long scans, blocking syscalls, page faults, CPU fallback
joins, and cold-tier reads can break the latency promise unless isolated in
separate classes or made preemptible by other mechanisms.

The core arbiter needs root privileges and cpuset control. That is acceptable
for a controlled benchmark host but not automatically acceptable for production
deployment. A database can still borrow the internal idea by pinning worker
pools and publishing core budgets without adopting a whole-machine arbiter.

Arachne also has a hard scale shape: 56 occupied thread contexts per core in
the described encoding. GPU DB's 1M logical sessions cannot map to per-session
thread contexts. The mapping must be many parked logical sessions to a small
number of active per-core request contexts.

NUMA and hyperthread policy are underexplored in the paper. GPU DB will care
deeply about NUMA locality for NIC queues, pinned host buffers, WAL memory,
NVMe queues, and GPU PCIe/NVLink affinity. A core-aware runtime without
topology-aware placement could move bottlenecks rather than remove them.

**Benchmark candidates:**

- Replace the thread-per-client pgwire benchmark path with a small IO-worker
  pool and fixed request contexts. Gate: equivalent SQL correctness and lower
  p95/p99 latency at the same logical client count before increasing scope.
- Add per-class runnable-load telemetry: IO parse, mutation, retained read,
  residency refresh, GPU launch/completion, CPU fallback, and response encode.
  Gate: scheduler/admission decisions can be replayed from queue depth, load
  factor, utilization, and wait-time logs.
- Prototype per-core retained-read task contexts with power-of-two placement
  and no heap allocation. Compare against generic channels and a ready-queue
  worker pool for same-shape lookup batches.
- Separate exclusive owners from normal request work: one or more pinned lanes
  for network polling, mutation/WAL, residency, and GPU execution; bounded
  normal pools for short read/response tasks. Failure condition: tail latency
  improves only by starving write visibility, refresh, or response delivery.
- Add a parked-session benchmark: many idle or slow logical sessions, small
  active runnable set, and bursty retained reads. Gate: memory per session,
  active contexts per core, queue wait, and p99 response latency remain bounded.
- Measure whether CPU work can be done during short waits on WAL flush, CUDA
  events, or cold-block fetches without violating owner-domain ordering. Gate:
  no work runs while holding state that would make visibility or invalidation
  ambiguous.
- Run a NUMA/topology placement sweep for NIC, WAL buffers, GPU pinned buffers,
  and worker cores. Failure condition: a "core-aware" layout wins average
  throughput but worsens p99 because it ignores PCIe or memory locality.

### 2026-06-03 - Cross-paper synthesis: scheduler decisions must be classed

**Papers compared:** BaM GPU-initiated storage access; ParamTree learned
cost-model calibration; Arachne core-aware thread management.

**Converging design tracks:** These papers converge on the same shape from
different layers: do not hide resource decisions in opaque subsystems. BaM
wants cold-block IO to expose queue depth, coalescing, cache-hit state, and
block granularity. ParamTree wants route costs to expose named formula terms
that can be calibrated by context. Arachne wants applications to see physical
core allocations and schedule their own short-lived work accordingly.

For GPU DB, that points to classed scheduling. A retained point read, resident
aggregate, cold over-resident scan, mutation batch, WAL wait, refresh, response
encode, and CPU fallback should not all be anonymous requests in one queue.
Each class needs its own hard validity checks, resource terms, queue-depth
telemetry, and fallback policy.

**Category gaps:** The recent set has good coverage for over-resident IO,
route calibration, and CPU runtime scheduling. The next underrepresented high-
value papers should still lean MVCC/HTAP visibility or transaction write-path
design rather than another GPU analytics or learned-optimizer paper.

**Benchmark priorities:**

- Build a replayable route/admission log that records class, legality checks,
  cost terms, queue wait, resource generation, and observed latency for each
  request.
- Add a scheduler experiment with separate classes for hot retained reads, cold
  over-resident reads, mutations, refresh, and response encoding. Gate: no class
  improves by starving visibility publication or response completion.
- Combine BaM-style block-miss coalescing with Arachne-style active-context
  limits: a cold route should be admitted only when the batch provides enough
  independent miss parallelism and enough execution contexts to hide latency.
- Treat ParamTree-style learned weights as advisory inside each class, never as
  a replacement for hard snapshot, WAL, residency, and queue-capacity gates.

### 2026-06-03 - Rebirth-Retire adaptive contention control

**Citation:** Qian Zhang, Yiwen Xiang, Jianhao Wei, Yang Yang, Yifan
Li, Xueqing Gong, and Wanggen Liu. "Rebirth-Retire: A Concurrency
Control Protocol Adaptable to Different Levels of Contention." PVLDB
18(9), 2025, pp. 3162-3174. doi:10.14778/3746405.3746435.
Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol18/p3162-zhang.pdf`.

**Category:** Transaction processing / write path and concurrency
control.

**Relevance tags:** contention management; lock retirement; dynamic
timestamps; deadlock avoidance; dependency tracking; hot keys;
long read-only transactions; abort reduction; owner queues; write
admission.

**Core idea:** Rebirth-Retire improves Bamboo/Wound-Retire by making
early lock release demand-driven and by replacing unconditional
"older kills younger" conflict handling with dynamic timestamp
rebirth. A transaction that holds a lock does not proactively retire
it after every access. Instead, a conflicting requester initiates
retirement only when the lock is actually blocking useful work.

The second idea is that an older transaction need not abort a younger
owner unless the dependency graph says rebirth would create a cycle.
The older transaction and its descendants can be assigned larger
timestamps so they logically move after the conflicting younger
transactions. In the paper's DBx1000 experiments, this combination
reduces unnecessary aborts and improves throughput under skewed
YCSB and TPC-C contention while avoiding the low-contention overhead
that hurt active Wound-Retire.

**Concrete mechanisms:**

- Each tuple lock entry keeps `waiters`, `owners`, and `retired` lists
  ordered by transaction timestamp. A retired owner has released the
  lock early but still creates dependencies for transactions that
  observe or conflict with its tentative work.
- Passive Retire is initiated by a waiting conflicting transaction,
  not by the current lock owner. If no younger transaction is blocked,
  no retire operation is paid for.
- Exclusive-lock retirement waits until the owner has finished the
  current write to the tuple. The paper adds a per-version `ready`
  flag so a waiter does not expose an incomplete write.
- Transactions track actual dependency edges with `parents` and
  `children` lists instead of only a dependency counter. This costs
  more metadata but makes deadlock checks possible.
- A transaction can commit only after all parent transactions have
  terminated and it has not been aborted. If an aborted transaction
  releases an exclusive lock, dependent children cascade-abort.
- When an older transaction conflicts with younger lock holders,
  `TxnRebirth` topologically sorts the requester and all descendants
  through child edges. If a younger conflicting holder appears in
  that sorted set, rebirth would create a dependency cycle and that
  younger transaction is aborted. Otherwise, the sorted transactions
  receive larger timestamps.
- The simple timestamp strategy assigns fresh globally largest
  timestamps. The optimized "Larger" strategy assigns timestamps just
  beyond the largest conflicting holder, with worker id bits included
  to avoid duplicates and reduce global timestamp pressure.
- Waiter promotion scans waiters in timestamp order. If a waiter
  conflicts with an exclusive owner, it waits for `version.ready`,
  moves current owners to `retired`, promotes the waiter to `owners`,
  and records dependency edges against the last conflicting retired
  transaction.
- Optimizations include latch-free dependency tracking with an 8-byte
  word for common child-list cases, optimistic reads of descendants,
  minimized rebirth operations by reading older visible versions, and
  software prefetch jump pointers for long version-chain traversal.
- The implementation is in DBx1000, a row-oriented in-memory DBMS
  prototype. Evaluation uses YCSB and TPC-C, with contention varied by
  tuple count, transaction count, operations per transaction, write
  ratio, Zipf skew, and warehouse count.
- Reported findings include passive Retire outperforming active Retire
  across evaluated YCSB contention levels; Rebirth-Active Retire
  improving medium-contention throughput by 49% and reducing abort rate
  by 84% versus Wound-Active Retire; full Rebirth-Retire reaching about
  2x Wound-Retire throughput with about 3x lower abort rate at 40 YCSB
  worker threads in one skewed workload; and high-contention TPC-C
  throughput around 600K transactions/s while other evaluated protocols
  stayed below 300K transactions/s. These are paper-reported results,
  not GPU DB measurements.

**GPU DB mapping:** The transferable design rule is "pay complex
contention machinery only when a real conflict appears." GPU DB's
mutation owners should not maintain expensive dependency, retire, or
rebirth metadata on every write if hot-key contention is absent. A
cheap owner-local path should hold the write batch until commit; only
when waiters, skew, queue age, or abort telemetry crosses a threshold
should it switch to a retire/dependency mode.

Passive Retire maps to publication boundaries. For a hot row, key, or
partition, a mutation batch could expose a completed write subresult to
later same-owner work before the full transaction reaches commit, but
only behind explicit dependency edges and only after WAL and visibility
rules for external readers remain intact. The immediate use is not
external dirty reads; it is reducing owner-queue blocking among
transactions whose dependency order can be represented and later
resolved.

Rebirth maps to adaptive ordering inside a bounded owner queue. If an
older admitted transaction discovers that a younger transaction already
owns a hot key, the owner does not always need to kill the younger work
or stall the queue. It can demote the older transaction's local priority
when the dependency graph remains acyclic, preserving useful work and
reducing abort churn. This is especially relevant for skewed writes
where deterministic order, OCC validation, and pure first-writer-wins
all risk wasting work or extending queue waits.

The `ready` flag is a useful bridge between write execution and
visibility. GPU DB already treats WAL-before-visibility as non-
negotiable, but internal owner scheduling also needs a smaller
"write fragment complete" marker before dependent work can proceed.
For GPU-assisted updates or refreshes, that marker may include CPU value
materialization, WAL record construction, index delta availability, and
resident invalidation status.

The long read-only experiment also matters for retained snapshots.
Rebirth-Retire benefits when long reads do not block short read-write
transactions and can traverse to visible versions efficiently. GPU DB
should keep retained analytical reads from pinning hot write locks or
owner queues, while still recording enough dependency and version-chain
telemetry to know when long snapshots are harming hot-key progress.

**Risks and mismatches:** Rebirth-Retire is a lock-based in-memory OLTP
protocol, not a GPU execution or MVCC snapshot protocol. It permits
dirty reads and writes internally with dependency tracking; GPU DB must
not expose dirty data to SQL clients or weaken WAL-before-visibility.
Any mapping must be inside an owner-domain scheduler or transaction
batch, not a shortcut for external snapshot semantics.

The paper's data structures are tuple-lock lists and transaction
dependency lists in a CPU row store. GPU DB may use partition owners,
columnar resident snapshots, append-only WAL batches, and GPU refresh
state rather than per-tuple locks. Topological sorting over descendant
transactions is acceptable only when the dependency subgraph is small
and measured; a hot GPU DB partition with thousands of blocked requests
could turn the rebirth check into the bottleneck.

The evaluation uses DBx1000 stored procedures, 40 hardware threads,
YCSB, and TPC-C. It does not evaluate PostgreSQL protocol sessions,
GPU kernels, NVMe tiers, crash recovery, DDL, or multi-GPU residency.
The reported throughput numbers are therefore useful for contention
shape, not absolute capacity targets.

**Benchmark candidates:**

- Add a hot-key owner-queue benchmark with skewed updates and reads.
  Compare first-writer-wins, abort/retry, deterministic batch order, and
  a bounded rebirth-style priority-demotion prototype. Gate: identical
  committed state and WAL replay across all policies.
- Add conflict-triggered metadata accounting to the mutation path:
  count when dependency tracking is absent, armed, used, and retired.
  Expected result: low-contention writes do not pay dependency-graph
  overhead.
- Prototype a passive-retire-like internal stage for one stored
  procedure class: after a write fragment is complete and WAL intent is
  formed, dependent same-owner work can proceed behind an explicit edge.
  Failure condition: any SQL-visible snapshot can observe data before
  the configured visibility boundary.
- Measure topological-sort or dependency-walk cost under controlled
  hot-key fanout. Gate: rebirth checks remain bounded by a small
  descendant cap; above the cap, the owner falls back to deterministic
  ordering or explicit overload rather than unbounded graph work.
- Track abort causes separately: true dependency cycle, validation
  failure, stale resident generation, WAL failure, timeout, overload,
  and explicit policy demotion. Expected improvement: fewer false aborts
  under skew without hiding failed correctness checks.
- Add a long retained snapshot plus hot write benchmark. Measure whether
  long reads increase owner lock wait, dependency wait, version-chain
  traversal, or GPU snapshot retirement latency. Gate: fresh write p95
  does not grow linearly with retained-snapshot age.
- Test version-chain prefetch or jump-pointer metadata for hot updated
  keys before GPU encoding. Failure condition: extra metadata slows the
  common short-chain path more than it helps long-chain retained reads.

### 2026-06-03 - BOHM serializable multiversion ordering

**Citation:** Jose M. Faleiro and Daniel J. Abadi. "Rethinking
Serializable Multiversion Concurrency Control." PVLDB 8(11), 2015,
pp. 1190-1201. Retrieved 2026-06-03 from
`https://www.cs.umd.edu/~abadi/papers/rethink-mvcc.pdf`.

**Category:** MVCC / snapshot / visibility.

**Relevance tags:** serializable MVCC; deterministic transaction
ordering; write-set planning; version placeholders; read/write
decoupling; global timestamp avoidance; batch barriers; RCU garbage
collection; owner partitioning.

**Core idea:** BOHM starts from the observation that serializable MVCC
systems often lose the concurrency advantage of multiple versions
because they track reads in shared metadata, validate reads late, or
assign timestamps through contended global counters. Its response is to
separate serialization/version management from transaction execution.
Transactions are first placed in a total order, version placeholders are
created for all declared writes, and only then do execution threads run
transaction logic and fill the prepared versions.

The important distinction from snapshot isolation or optimistic MVCC is
that BOHM is pessimistic without making reads block writes. Reads do not
write bookkeeping into shared record metadata and do not validate at
commit. A read either follows the version chain to the version whose
interval contains the transaction timestamp, or, when read sets are known
early, uses a precomputed pointer to that version. Writes can delay reads
when the chosen version's data has not yet been produced, but reads do
not delay writes.

**Concrete mechanisms:**

- A single input thread appends complete transactions to an in-memory log.
  A transaction's log position is its timestamp, avoiding per-transaction
  atomic fetch-and-increment on a shared timestamp counter.
- Concurrency-control threads own logical partitions of records. For each
  transaction batch, each thread scans write sets and creates placeholder
  versions only for records in its partition.
- A version contains begin timestamp, end timestamp, transaction pointer,
  data, and previous-version pointer. Placeholder insertion sets begin to
  the writer timestamp, end to infinity, the transaction pointer to the
  writer, data uninitialized, and the previous version's end to the new
  timestamp.
- Each record is always handled by the same concurrency-control thread, so
  hot-record version chains are updated by one owner rather than by many
  contending writers.
- If read sets are available, the concurrency-control phase can annotate a
  transaction with direct pointers to the correct versions to read. This
  does not track reads in database records; it writes into preallocated
  transaction-local space.
- Coordination is amortized at batch granularity. Concurrency-control
  threads process an ordered batch independently and synchronize at one
  barrier before execution threads consume that batch.
- Execution threads receive an ordered batch and partition responsibility
  by transaction index. A transaction moves through unprocessed,
  executing, and complete states. If a read needs an unfilled version, the
  worker may recursively execute the producer transaction or retry later.
- Write-write conflicts are resolved by the precreated version order. A
  later write can fill its own version before an earlier write unless it
  has a read dependency on the earlier version.
- Optional garbage collection uses a batch low-watermark. Once every
  execution thread has completed the batch that invalidated an older
  version, that version cannot be visible to future batches and can be
  reclaimed or archived with an RCU-like scheme.
- The evaluation reports nearly 2 million 10-RMW YCSB transactions per
  second, about 20 million record accesses per second, in a concurrency
  control stress test; BOHM also avoids the low-contention global
  timestamp bottleneck seen in the paper's Hekaton and SI baselines and
  performs well when long read-only transactions coexist with updates.

**GPU DB mapping:** BOHM is a strong fit for GPU DB's owner-domain
architecture when a transaction or mutation batch has known write sets.
COPY chunks, stored-procedure-like write templates, partition-local
updates, refresh publication, and deterministic resident invalidation can
all be admitted as complete units, assigned an owner-local order, and
have visibility/version placeholders prepared before expensive CPU or GPU
execution starts.

The largest transferable idea is placeholder-first publication inside
the mutation owner. A mutation batch can allocate row/version slots,
index-delta slots, invalidation records, and response handles in a
deterministic order before it performs value materialization or GPU-
assisted work. Later stages fill those slots, but the serialization
boundary is already known. That reduces the temptation to protect every
read or write with fine-grained shared metadata.

BOHM also reinforces the need to avoid a single global timestamp path.
GPU DB should prefer per-owner or per-batch generation ranges that can be
merged into a global visibility boundary only at publish points. A
network IO worker or GPU execution worker should never be the authority
that hands out visibility timestamps through a contended atomic counter.

The read-set pointer optimization maps to retained read snapshots. For a
known query template, the planner or snapshot publisher can pre-resolve
stable handles: relation generation, partition generation, selected
column buffers, resident key vectors, and visibility boundary. Reads then
consume immutable handles rather than writing read-tracking metadata into
hot mutation structures.

BOHM's batch low-watermark is also useful for snapshot retirement.
Instead of one global oldest-reader value that every component updates,
GPU DB can track per-execution-class progress: mutation batches, retained
GPU reads, refresh batches, and long analytical snapshots. Versions,
tombstones, invalidated resident generations, and response buffers become
reclaimable only after the relevant class watermarks pass the batch that
made them obsolete.

**Risks and mismatches:** BOHM requires whole transactions and deducible
write sets before execution. General SQL sessions, cursor-style
transactions, ad hoc multi-statement workflows, and data-dependent writes
do not naturally satisfy that requirement. The paper mentions
speculative write-set prediction from prior work, but the BOHM mechanism
itself depends on correct planning.

The design is CPU in-memory OLTP, not GPU execution, pgwire session
multiplexing, WAL recovery, or tiered storage. Its placeholder versions
are not a license to expose uncommitted data externally. GPU DB must still
preserve WAL-before-visibility, replay correctness, catalog invalidation,
resident snapshot validity, and SQL error semantics.

The single input log and batch barrier simplify ordering but may become a
latency or admission bottleneck if used too broadly. GPU DB should treat
BOHM-like ordering as a class-specific fast path for compatible mutation
templates, not as the only path for all SQL.

**Benchmark candidates:**

- Prototype a placeholder-first mutation batch for one narrow write
  template. Allocate version/index/invalidation slots in owner order, fill
  them later, and publish visibility only after WAL safety. Gate:
  identical committed state and WAL replay versus the existing path.
- Compare global transaction id allocation with per-owner batch generation
  ranges for COPY admission. Required metrics: timestamp/allocation wait,
  WAL queue wait, visibility publish latency, and replay determinism.
- Add a known-read-set retained route proof that pre-resolves snapshot
  handles and avoids mutation-owner read tracking. Failure condition: any
  read result changes under insert/update/delete or resident invalidation.
- Build a long-read plus write-batch GC test using per-class watermarks.
  Gate: obsolete versions and invalidated resident generations are retired
  without making fresh write latency grow linearly with long snapshot age.
- Measure batch-barrier sensitivity for mutation admission: batch size,
  microsecond ceiling, owner partition count, and hot-key skew. Failure
  condition: throughput improves only by unacceptable p50/p99 latency.
- Add a "write set not known" negative-control benchmark. Expected
  behavior: BOHM-style path rejects or falls back cleanly instead of
  guessing and weakening serializable or WAL semantics.
- For GPU-assisted deterministic updates, test whether CPU owner ordering
  plus GPU value computation can fill precreated version slots faster than
  CPU-only execution while preserving exactly the same publish boundary.

### 2026-06-03 - CAM asynchronous GPU-initiated CPU-managed SSD access

**Citation:** Ziyu Song, Jie Zhang, Jie Sun, Mo Sun, Zihan Yang,
Zheng Zhang, Xuzheng Chen, Fei Wu, Huajin Tang, and Zeke Wang.
"CAM: Asynchronous GPU-Initiated, CPU-Managed SSD Management for
Batching Storage Access." ICDE 2025, pp. 2309-2322,
doi:10.1109/ICDE65448.2025.00175. Retrieved 2026-06-03 from the
author PDF, `https://wangzeke.github.io/doc/cam-ICDE25.pdf`, with
metadata cross-checks from DBLP and the IEEE DOI page.

**Category:** multi-tier cache / data placement.

**Relevance tags:** GPU-SSD data path; NVMe tiering; GPUDirect;
SPDK; over-resident execution; asynchronous prefetch; CPU/GPU
control-plane split; pinned GPU buffers; SSD batching; compute/IO
overlap.

**Core idea:** CAM argues that both common GPU out-of-core designs
leave performance on the table. CPU-managed paths such as POSIX I/O,
libaio, or SPDK can keep GPU compute kernels simple, but often route
SSD data through CPU memory and pay kernel or copy overheads. Fully
GPU-managed paths such as BaM avoid the CPU staging copy and let GPU
thread blocks submit NVMe work directly, but their synchronous API can
consume many GPU SMs waiting on SSD latency and can serialize storage
access with useful GPU computation.

CAM's compromise is GPU-initiated but CPU-managed storage access. The
GPU computes logical block addresses and signals an asynchronous batch,
while persistent CPU-side polling threads use SPDK/user-space NVMe
control to submit requests. The data plane still transfers directly
between SSDs and pinned GPU memory, so CPU memory does not become the
intermediate tier. The control plane is moved off GPU SMs, allowing GPU
compute kernels to use the device while the CPU manages SSD queues.
The paper evaluates this design on an A100 80GB server with 12 Intel
P5510 NVMe SSDs and reports that CAM reaches roughly the same raw
multi-SSD throughput as SPDK/BaM while avoiding GPU SM burn and CPU
memory bandwidth pressure; its end-to-end workloads include GNN
training, sort, and GEMM.

**Concrete mechanisms:**

- The GPU writes an array of logical block addresses for the next
  prefetch/write-back batch into CPU-visible memory, then a leading GPU
  thread publishes a signal that the batch is ready.
- A persistent CPU polling thread observes the signal, submits NVMe
  work through SPDK, waits for completions, and writes a completion
  signal for the GPU.
- GPU threads can perform computation over data fetched by the previous
  stage while the CPU and SSDs process the next stage. The API exposes
  `prefetch`, `prefetch_synchronize`, `write_back`, and
  `write_back_synchronize` so applications can write code that looks
  mostly synchronous while the implementation pipelines IO and compute.
- CAM uses four preallocated synchronization regions: a GPU-written LBA
  array, a GPU-written argument region, a GPU-to-CPU ready signal, and a
  CPU-to-GPU completion signal. The first three are unified-memory
  regions; the completion region is GPU memory with a CPU-visible copy.
- GPU memory allocation goes through CAM rather than plain
  `cudaMalloc`; CAM pins and maps buffers with GDRCopy and
  `nvidia_p2p_get_pages`, then uses physical addresses in NVMe SQEs so
  SSD DMA targets GPU memory directly.
- CPU-side SSD management uses SPDK to bypass the kernel block layer,
  filesystem, page cache, and mode switches. Each CPU thread can manage
  one or more SSDs with dedicated NVMe queue pairs and lock-free driver
  paths.
- CAM dynamically adjusts CPU core allocation for SSD control based on
  the prior batch's compute time versus IO time. If compute dominates,
  fewer CPU threads can be used without extending total time; if IO
  dominates, the control path needs more threads.
- The evaluation states that one CPU thread can control two SSDs without
  throughput loss on their platform, while one thread controlling four
  SSDs drops to about 75% of the one-thread-per-SSD throughput.
- With 12 SSDs and 4KB granularity, the paper reports about 20GB/s CAM
  throughput, close to the measured platform PCIe peak of about 21GB/s
  and below the theoretical 32GB/s due to PCIe overhead and contention.
- The paper contrasts CAM with SPDK plus overlap: SPDK still stages
  through CPU memory, consuming roughly twice the SSD bandwidth in CPU
  memory bandwidth for GPU reads/writes, and its scatter/non-contiguous
  destination path performs poorly at small granularity.
- Reported end-to-end improvements include up to 1.84x for GNN model
  training versus the BaM-based GIDS baseline, up to 1.5x for sort, and
  up to 1.84x for GEMM versus evaluated GPU storage baselines. These
  are paper-reported application results, not GPU DB measurements.
- The authors list three limitations: CAM requires raw SSD access
  without a filesystem, concurrent access by multiple processes risks
  consistency issues, the prototype targets a single GPU, and fully
  exploiting more SSDs still needs CPU cores that scale with device
  count.

**GPU DB mapping:** CAM is most useful for the over-resident P8 path,
where cold or warm partitions live on NVMe but the execution target is
still GPU memory. It suggests that GPU DB should not treat "GPU
initiated" as synonymous with "GPU managed." A retained or over-resident
kernel can compute the next partition/block IDs, publish a compact IO
batch descriptor, and return SMs to useful work while CPU-owned storage
workers drive NVMe queues and completions.

The design maps naturally to the owner-domain runtime. GPU execution
owners should own CUDA streams and staging buffers; storage/IO owners
should own SPDK queue pairs, NVMe submission, raw-device safety, and
completion accounting; residency owners should decide whether a block is
admitted, reused, invalidated, or evicted. CAM's four-region handshake
is a hardware-level analog of the bounded command/response rings already
called out in `11-high-throughput-query-runtime.md`.

For P8, the direct SSD-to-GPU data path is a candidate future tier, not
a replacement for durable WAL/checkpoint authority. NVMe-resident
partitions can be read directly into pinned GPU buffers for scans,
external aggregation, sort/merge, or refresh, but every route still needs
catalog generation, source WAL boundary, checksum or block identity,
visibility boundary, and invalidation checks before results become
SQL-visible.

CAM also sharpens the benchmark question around CPU memory bandwidth.
If an over-resident route stages `NVMe -> CPU DRAM -> GPU HBM`, it can
consume CPU memory bandwidth at roughly twice the SSD bandwidth and can
compete with WAL, MVCC, pgwire, and host cache work. A direct
`NVMe -> GPU HBM` experiment should therefore measure not only GPU
query time but also CPU memory bandwidth, IO worker cores, pinned-memory
budget, and interference with mutation admission.

The dynamic CPU-core policy transfers to tier placement. GPU DB should
not statically reserve one core per NVMe device for every workload. It
should adjust storage-worker budget based on whether the current route
is IO-bound, GPU-bound, response-bound, or mutation-bound, with hard
admission ceilings when CPU cores, queue pairs, pinned buffers, or SSD
bandwidth become saturated.

**Risks and mismatches:** CAM is not a database system. It does not
handle filesystems, SQL transactions, WAL recovery, MVCC visibility,
checksums, page ownership, concurrent tenants, DDL, or multi-process
consistency. Its raw-device requirement is a major mismatch for a
general DBMS unless GPU DB controls its own cold-partition device layout
or uses a carefully isolated block arena.

The evaluated workloads are GNN training, sort, and GEMM, not
transactional queries or HTAP with fresh writes. CAM's best case assumes
predictable next-batch addresses and enough independent compute to cover
IO latency. Point lookups, high-selectivity reads, write-heavy COPY
admission, or data-dependent joins may have pipeline bubbles that CAM
cannot remove.

The prototype is single-GPU and requires pinned GPU memory. GPU DB will
need explicit budgets and fallback behavior for pinned HBM/host mappings,
NVMe queue depth, SSD namespaces, and concurrent sessions. Finally, the
paper reports throughput under a specific A100/12-SSD platform; the
transferable claim is the control/data-plane split and overlap pattern,
not the absolute GB/s target.

**Benchmark candidates:**

- Add an over-resident IO microbenchmark with three paths:
  `NVMe -> CPU DRAM -> GPU`, SPDK-overlapped staging, and a future direct
  `NVMe -> GPU pinned buffer` path. Required metrics: GB/s, p50/p99
  request latency, CPU memory bandwidth, CPU cores consumed, GPU SM idle
  time, and correctness checksum.
- Prototype a bounded GPU-computed block-request descriptor for one
  resident refresh or scan route. The GPU or planner emits block IDs;
  the storage owner drains them through a CPU-managed queue. Gate:
  identical rows and visibility boundaries versus CPU-staged refresh.
- Add a pipeline-bubble benchmark for over-resident scans: predictable
  sequential partitions, random block batches, and data-dependent next
  block selection. Failure condition: direct IO complexity helps only the
  sequential case while harming point lookups or p99 latency.
- Measure CPU memory bandwidth interference during staged over-resident
  execution while COPY/WAL admission is active. Expected result: direct
  SSD-to-GPU transfer should reduce host-memory pressure; failure
  condition: storage-worker polling steals enough CPU to regress write
  throughput.
- Add storage-worker budget telemetry: SSD queue depth, storage ring
  wait, completions per poll, CPU cores assigned, pinned GPU bytes,
  direct-transfer bytes, staged-transfer bytes, and fallback reason.
- Test dynamic storage-worker allocation by route class. Minimum proof:
  reducing storage cores when GPU compute dominates does not change wall
  time, while increasing cores for IO-bound scans improves throughput
  until SSD or PCIe saturation.
- Keep a negative-control path for raw-device safety: if a cold partition
  cannot prove exclusive block ownership, checksum identity, and source
  generation, the direct path must reject or fall back rather than read
  unmanaged blocks.

### 2026-06-03 - Cross-paper synthesis: control planes should stay explicit

The recent BOHM, Rebirth-Retire, and CAM reviews converge on the same
shape from different layers: the system should separate the cheap common
execution path from the control path that establishes ordering, ownership,
and resource safety. BOHM prepares version placeholders before executing
known writes, Rebirth-Retire arms dependency machinery only under real
contention, and CAM moves SSD command management off GPU SMs while keeping
GPU-side initiation and direct data movement.

For GPU DB, that points to three active design tracks. First, mutation
batches need explicit order/allocation/publication slots before expensive
value work starts, but visibility still waits for WAL and invalidation
safety. Second, conflict handling should be adaptive and owner-local:
cheap deterministic order in the common case, bounded dependency or
priority metadata only when hot-key contention appears, and measured
fallback when the graph grows too large. Third, over-resident reads should
separate GPU compute from storage control: GPU/planner code can describe
needed blocks, while CPU-owned storage workers drive NVMe queues,
checksums, generations, and completion rings.

The category gap after this set is still multi-tier transactional
storage. CAM covers direct GPU/SSD movement, but not DBMS page ownership,
filesystem integration, recovery, or shared cold-partition allocation.
A next high-value tiering paper should be LeanStore, Umbra, FOEDUS,
BTrim, or another transactional storage source that explains how hot CPU
working sets coexist with durable/cold data.

Benchmark priorities should now be ordered around control-plane costs:
per-owner generation allocation versus global timestamp allocation,
conflict-triggered dependency metadata overhead, staged versus direct
NVMe-to-GPU transfer under concurrent COPY/WAL admission, and a negative
control that proves unknown write sets or unmanaged raw blocks fall back
cleanly rather than guessing.

### 2026-06-03 - LeanStore low-overhead transactional buffer management

**Citation:** Viktor Leis, Michael Haubenschild, Alfons Kemper, and
Thomas Neumann. "LeanStore: In-Memory Data Management Beyond Main
Memory." ICDE 2018, pp. 185-196, doi:10.1109/ICDE.2018.00026.
Retrieved 2026-06-03 from the author PDF,
`https://db.in.tum.de/~leis/papers/leanstore.pdf`, with metadata
cross-checks from DBLP, TUM, and the IEEE DOI page.

**Category:** multi-tier cache / data placement.

**Relevance tags:** transactional storage; larger-than-memory OLTP;
explicit buffer management; pointer swizzling; SSD/NVMe tiering;
cooling-stage eviction; optimistic page synchronization; epoch
reclamation; NUMA-aware allocation; scan prefetch; hot-index placement.

**Core idea:** LeanStore reopens the buffer-manager question for
main-memory-style transactional systems. Traditional buffer managers are
transparent and can manage tables and indexes uniformly, but their hot
path pays page-id translation, pin/unpin, latch, and replacement-tracking
costs even when almost everything is resident. Pure in-memory engines
avoid that overhead by using virtual-memory pointers, but then handling
data sets beyond DRAM becomes an afterthought or a separate cold-storage
subsystem.

LeanStore's answer is a storage manager that keeps buffer-manager
transparency while making hot-page access close to an in-memory pointer
chase. Resident page references are "swizzled" direct pointers; cold or
cooling references are logical page identifiers. A single tagged word
distinguishes the two states. The normal hot access therefore pays only a
well-predicted tag check instead of a global hash-table translation and
pinning protocol.

The replacement policy is also inverted. Instead of tracking every
access to identify hot pages, LeanStore speculatively unswizzles random
pages and gives them a FIFO "cooling" grace period. If a cooling page is
touched again, it is quickly reswizzled from memory without disk IO; if
it reaches the end of the cooling queue and the epoch rules allow reuse,
it can be flushed and evicted. The paper's evaluation uses TPC-C and
microbenchmarks and reports near in-memory B-tree performance when the
working set fits in DRAM, smoother degradation beyond DRAM on fast SSDs,
and much better scalability than traditional buffer-manager baselines.

**Concrete mechanisms:**

- Page references are stored as swips: either direct virtual-memory
  pointers for resident pages or logical page identifiers for
  non-resident/cooling pages, distinguished by pointer tagging.
- Every page has one owning swip. That avoids multiple decentralized
  references racing to update the same page state during swizzling or
  unswizzling, at the cost of making buffer-managed structures tree-like
  or forest-like.
- Inner pages are not unswizzled while they still contain swizzled
  children. Buffer-managed data structures provide callbacks that iterate
  child swips, letting the buffer manager choose a swizzled child instead
  of writing a parent page containing process-local pointers.
- The cooling stage keeps a bounded percentage of pages, commonly around
  10% in the paper, as unswizzled-but-still-resident pages in a FIFO
  queue plus a page-id hash table. Touching a cooling page removes it
  from the queue and reswizzles it; needing a free frame evicts from the
  queue tail after dirty flushing and epoch checks.
- Cooling-stage and in-flight-IO metadata use a global latch, but only
  on the cold path. The latch is released before blocking IO, so multiple
  SSD reads can run concurrently.
- Concurrent loads of the same cold page are serialized through an
  in-flight IO table. The first thread installs an IO frame and reads;
  later threads wait for the same frame instead of loading duplicate
  copies.
- Page lifetime safety uses epochs rather than per-access pin counters.
  Threads enter a local epoch while traversing data structures and exit
  frequently. A cooling page can be reused only when all active local
  epochs are newer than the page's unswizzle epoch.
- Long operations such as scans are broken into smaller epoch scopes.
  If a page fault occurs, the operation releases locks, exits its epoch,
  performs IO, and restarts traversal. This avoids letting one long scan
  block eviction and reclamation.
- Reads use optimistic version validation rather than latching on every
  lookup. Writes usually latch only the leaf page, restarting with inner
  latches only for structure-modifying operations.
- Buffer frames are physically interleaved with page contents to improve
  locality and avoid cache-associativity pathologies for page headers.
- The buffer pool is one large allocation and can be pre-faulted.
  NUMA-aware mode partitions free lists and tries to allocate pages on
  the requesting thread's NUMA node while retaining a global replacement
  policy.
- A background writer flushes dirty cooling pages to hide write latency.
  Scan prefetch can issue multiple page requests through the in-flight IO
  component; scan-loaded pages may be hinted as cooling so large scans do
  not evict the hot working set.
- The evaluation disables transactions/logging in baselines to isolate
  storage-manager overhead. Reported LeanStore TPC-C throughput is close
  to its in-memory B-tree and far above BerkeleyDB/WiredTiger in the
  tested configurations; with a 20GB buffer pool and growing TPC-C data,
  LeanStore stays close to in-memory behavior while Linux swapping is
  unstable and traditional engines are slower. Absolute numbers are
  platform-specific and should not be treated as GPU DB targets.

**GPU DB mapping:** LeanStore is a strong argument that GPU DB's
CPU/NVMe tiers should remain explicitly managed, not delegated to mmap or
OS swap, but the hot path must avoid classic buffer-pool tax. For P8, the
analog is not literally swizzling row-store B-tree pages into GPU memory.
It is publishing direct resident handles for hot CPU and GPU segments so
the common read route checks a small generation/state tag, then executes
from an immutable pointer-rich snapshot without a catalog hash lookup,
global pin, or per-request replacement update.

The cooling-stage idea maps cleanly to resident GPU partitions. A
partition can move from `Valid` to "cooling resident" by withdrawing it
from new fast-route selection while keeping HBM buffers intact for a
short grace window. If the planner or workload touches it again, the
residency owner can cheaply republish it without a full rebuild. If it
ages out, eviction can release HBM after reader epochs/snapshot
references prove safety. This gives the cache manager a low-overhead
probation state between hot and evicted, which the current P8 state
machine does not yet distinguish.

LeanStore's epoch guidance reinforces the runtime snapshot design. Long
retained scans, refreshes, or over-resident reads must not hold one
global epoch or hazard forever. They should chunk work by partition,
column group, or response batch and release snapshot/epoch state between
chunks when correctness allows. Otherwise one large scan could prevent
resident HBM reclamation, MVCC version cleanup, or cold-tier demotion.

The single-owning-swip rule is also useful as a mental model for GPU DB
resident handles: each mutable residency object should have one owner
that is allowed to change its state, while published readers see stable
immutable handles. Multiple indexes, route caches, and prepared plans
should not independently mutate resident-state flags. They should point
through owner-published generation records.

The scan hinting and prefetch mechanisms are direct benchmark material.
GPU DB can mark large sequential cold scans as "cooling on admission" so
they do not displace hot retained lookup partitions. Conversely,
point-lookup and short aggregate partitions that are reswizzled within
the grace window should remain resident. Over-resident NVMe paths can
combine CAM-style CPU-managed IO with LeanStore-style in-flight request
coalescing so concurrent requests for the same cold partition share one
load or refresh.

**Risks and mismatches:** LeanStore is a CPU storage manager, not a GPU
execution system. Its pointers are process-local CPU addresses; GPU DB
needs CUDA device pointers, host pinned buffers, catalog generations,
visibility boundaries, stream ownership, and invalidation metadata. A
simple pointer-tag check is not enough to prove SQL visibility or resident
route validity.

The paper isolates storage-manager overhead by disabling transactions,
logging, compaction, and compression in comparison systems. GPU DB cannot
drop WAL-before-visibility, MVCC replay, checksums, or DDL invalidation
to get comparable numbers. The useful lesson is control-path shape and
hot-path overhead, not a promise that a buffer manager alone solves write
admission or GPU route correctness.

LeanStore's replacement policy is page-oriented and assumes a single
global replacement policy across buffer-managed structures. GPU DB may
need multiple budgets and policies across HBM, pinned host memory, CPU
DRAM, NVMe, and future CXL/remote tiers. Random speculative cooling may
also be too blunt for expensive GPU resident segments unless combined
with route telemetry, refresh cost, and admission value.

Finally, the paper evaluates fast SSD-backed CPU OLTP and scans, not
pgwire session scale, GPU kernel launch amortization, or mixed HTAP with
fresh writes and retained analytical snapshots. The mapping should remain
an explicit benchmark hypothesis.

**Benchmark candidates:**

- Add a resident-partition cooling state to a small cache-manager proof:
  `Valid -> CoolingResident -> Valid` on reuse, or
  `CoolingResident -> Evicted` after a bounded grace window and snapshot
  release. Gate: no stale route after mutation/DDL and no HBM release
  while a retained reader still holds the generation.
- Measure exact-route hot lookup overhead with and without a direct
  generation handle cache. Required metrics: route lookup nanoseconds,
  owner queue wait, resident validity checks, cache misses, and fallback
  reason. Failure condition: the direct handle hides invalidation or
  schema-generation changes.
- Build an in-flight resident refresh coalescing test. Concurrent misses
  for the same partition should install one refresh frame; later requests
  wait, share, or reject based on admission policy rather than triggering
  duplicate H2D/NVMe work.
- Add a long-scan reclamation test where retained scans release snapshot
  epochs between partitions. Gate: eviction/MVCC cleanup progresses while
  scan correctness remains identical to a single long snapshot when that
  semantic is requested.
- Compare eviction policies for HBM partitions: random cooling, LRU-like
  per-access tracking, workload-value tracking from planner telemetry, and
  scan-hinted cooling. Required metrics: hot-route hit rate, per-request
  overhead, HBM churn, refresh bytes, p99 latency, and COPY/WAL
  interference.
- Add a sequential over-resident scan negative control: scan-loaded
  partitions are admitted as cooling unless repeated reuse is observed.
  Expected result: large scans do not evict point-lookup/aggregate hot
  partitions; failure condition: scan throughput improves by destroying
  retained OLTP latency.
- Track NUMA and pinned-memory placement for host staging buffers.
  Minimum proof: route-local host buffers reduce remote memory accesses
  and do not fragment or exhaust pinned budgets under many sessions.

### 2026-06-03 - Umbra variable-size pages for SSD-backed hot working sets

**Citation:** Thomas Neumann and Michael Freitag. "Umbra: A Disk-Based
System with In-Memory Performance." CIDR 2020. Retrieved 2026-06-03
from the TUM author PDF,
`https://db.in.tum.de/~freitag/papers/p29-neumann-cidr20.pdf`, after
the originally queued VLDB/CIDR URL returned HTTP 403 from the cron
environment. CIDR proceedings metadata was cross-checked at
`https://www.vldb.org/cidrdb/2020/umbra-a-disk-based-system-with-in-memory-performance.html`.

**Category:** multi-tier cache / data placement.

**Relevance tags:** SSD-backed DBMS; variable-size pages; explicit buffer
management; virtual-memory reservation; pointer swizzling; optimistic
latching; PAX page layout; string lifetime; online statistics; adaptive
execution; IO-aware scheduling; hot/cold working-set placement.

**Core idea:** Umbra evolves the in-memory HyPer design into an SSD-backed
DBMS without giving up hot-working-set performance. The paper argues that
pure in-memory systems become uneconomical as DRAM capacity growth slows,
while fast SSDs make a large explicit buffer plus persistent storage a better
default. The key is to keep the cached common path close to in-memory pointer
access while still letting uncached data load predictably.

Umbra's distinctive addition beyond LeanStore is a low-overhead buffer manager
with variable-size pages. Fixed-size pages simplify buffer managers but make
large strings, dictionaries, and compression lookup tables awkward because
objects must be split, copied, or accessed through complex page-spanning
logic. Umbra instead organizes pages into exponentially growing size classes,
reserves one virtual-address region per class, and maps physical memory only
for active frames. That allows large objects to remain contiguous in virtual
memory without fragmenting the physical buffer pool.

The paper also shows that making a memory-optimized engine disk-backed leaks
into other layers. Strings need explicit storage-duration classes because a
database page may be evicted while a pipeline still holds an out-of-line
string reference. Statistics must be maintained online rather than sampled
from cold base tables on demand. Execution is represented as modular steps
inside pipelines so work can be suspended between steps when IO load or route
state requires it. In the evaluation, Umbra reports comparable raw execution
time to HyPer on JOB/TPCH hot-cache runs, buffer-manager overhead below about
6% on average when bypassed, and cold-scan throughput close to SSD random-read
bandwidth on the tested platform.

**Concrete mechanisms:**

- Page size classes start at 64KiB in the prototype and double for larger
  pages. Each class gets a reserved virtual-memory region as large as the
  configured buffer pool, but physical memory is consumed only by active
  frames.
- Active frames are populated with `pread`; evicted frames are flushed with
  `pwrite` if dirty and released with `madvise(MADV_DONTNEED)` so physical
  memory can be reused while the virtual address remains stable.
- The buffer manager tracks the total bytes of active pages across all size
  classes and enforces one global buffer-pool budget rather than separate
  budgets per page size.
- Replacement follows the LeanStore cooling idea: pages are speculatively
  unswizzled and kept resident in a FIFO grace period before actual eviction.
- A swip is a 64-bit tagged reference. Swizzled references hold a resident
  virtual pointer; unswizzled references hold a page id plus a 6-bit size
  class, so loading a cold page does not require external size metadata.
- Every page has exactly one owning swip. Buffer-managed structures are
  therefore organized as trees or forests, and B+Tree leaf pages omit sibling
  links that would create multiple owners for one page.
- Page synchronization uses versioned latches with exclusive, shared, and
  optimistic modes. Optimistic reads remember the version counter and validate
  at release; shared latches are used when operators cannot tolerate page
  eviction during a unit of work.
- Relations are stored as B+Trees keyed by synthetic monotonically increasing
  tuple ids. Inner pages use the smallest page size for high fanout, while
  leaf pages use the smallest class that can hold the inserted tuple.
- Leaf pages use a PAX-style layout: fixed-size attributes in columnar form at
  the front of the page and variable-size data packed at the end.
- Because different-size pages complicate ARIES recovery, Umbra only reuses
  freed disk space for pages of the same size; otherwise recovery might
  interpret stale bytes from a larger old page as a smaller new page's LSN.
- Strings have 16-byte headers. Short strings up to 12 bytes are inline; long
  strings carry length, prefix bytes, and either offset or pointer metadata
  tagged by storage class: persistent, transient, or temporary.
- Query statistics use online reservoir sampling and updateable HyperLogLog
  sketches so the optimizer does not need expensive random reads from
  disk-backed base relations.
- Physical execution plans are decomposed into pipeline steps. Each step is a
  callable state-machine function, and multi-threaded steps use morsels.
  Execution can suspend after a step, avoid thread dispatch for single-morsel
  work, and combine interpretation with adaptive compilation.

**GPU DB mapping:** Umbra reinforces that P8 should avoid two traps at once:
delegating hot/cold movement entirely to the OS, and building an explicit tier
manager whose normal hit path pays classic buffer-pool overhead. GPU DB's
resident route should look more like a direct generation handle with a small
state check than a global lookup, pin, replacement update, and route rebuild
for every query.

Variable-size pages are a useful model for GPU DB column groups and text
payloads. Fixed-size resident partitions are convenient for scheduling, but
text columns, compressed dictionaries, offsets, and GPU lookup tables often
want contiguous regions. The GPU DB analog is a classed segment arena: small
fixed-width columns in regular chunks, larger dictionaries or text payloads in
size-classed contiguous host/GPU regions, and one owner-published descriptor
that carries class, generation, checksum, visibility boundary, and byte budget.

The reserved-virtual-memory trick does not translate directly to CUDA HBM, but
it is highly relevant to CPU DRAM and pinned host staging. Host-side tier
arenas can reserve stable address ranges for segment classes while mapping or
pinning only admitted working sets. For GPU memory, the design lesson is the
same budget surface: active bytes across all segment classes matter more than
object count, and variable-size objects should not require multiple independent
buffer pools that strand capacity.

Umbra's single-owning-swip rule matches the owner-domain architecture. A
resident partition, column dictionary, or cold NVMe block should have one
mutable owner record. Prepared plans, route caches, and read workers should
hold immutable handles to owner-published generations rather than mutating
validity flags independently.

The modular step execution maps to GPU DB route scheduling. Short retained
reads can run immediately; longer scans, refreshes, and over-resident routes
should be decomposed into suspendable steps such as admit, load block, launch
kernel, reduce, encode response, and release generation. That gives the
runtime places to honor IO pressure, GPU queue saturation, and latency ceilings
without blocking network IO or holding a global snapshot epoch.

Umbra's string lifetime split is a concrete warning for pgwire and GPU result
encoding. Values referenced from resident pages or transient staging buffers
must either be copied into response-owned memory or kept alive by a generation
handle until network write completion. A result encoder cannot assume a page,
GPU buffer, or pinned staging slice remains valid just because SQL execution
has produced rows.

**Risks and mismatches:** Umbra is a CPU DBMS. It does not solve CUDA stream
ownership, GPU memory allocation, kernel launch amortization, direct
NVMe-to-GPU IO, pgwire session multiplexing, or GPU-resident MVCC visibility.
The variable-size page design relies on ordinary CPU virtual memory and
`madvise`; device memory and pinned host memory have different fragmentation,
mapping, and registration costs.

The paper's recovery mechanism assumes ARIES over pages. GPU DB currently uses
WAL/checkpoint/archive replay as durable truth and treats GPU residency as
rebuildable acceleration state. The transferable recovery lesson is careful
space-reuse metadata and page/segment identity, not a requirement to adopt
ARIES pages.

The evaluation uses one 8-core CPU system, one Samsung SSD, hot-cache fastest
of five query runs for JOB/TPCH, and a CPU query compiler/runtime. Its absolute
speedups do not predict GPU DB throughput. Also, the paper mostly addresses
database storage and analytical/query execution mechanics; it is not a full
answer for OLTP commit ordering, WAL flush throughput, or 1M logical sessions.

**Benchmark candidates:**

- Prototype classed resident segment arenas for host/GPU metadata: fixed-width
  columns, text offset/byte payloads, dictionaries, and scratch buffers use
  separate size classes but share one active-byte budget. Gate: no stranded
  capacity from per-class pools under mixed `int4`/`text` resident workloads.
- Add a cold/warm partition route benchmark with variable-size payloads:
  compare fixed chunks that split dictionaries versus classed contiguous
  dictionary/text regions. Required metrics: refresh bytes, H2D/D2H bytes,
  kernel indirections, p99 lookup latency, and HBM/host fragmentation.
- Implement an owner-published resident handle shape inspired by swips:
  generation id, segment class, logical page/partition id, resident pointer or
  cold id, validity state, and visibility boundary. Failure condition: any
  prepared plan or route cache can mutate resident validity directly.
- Add a suspendable route-step trace for one retained scan or refresh:
  admission, snapshot acquire, IO/load, GPU launch, reduce, encode, release.
  Gate: IO pressure or GPU saturation can pause between steps without holding
  network IO, mutation owner, or stale resident generation resources.
- Build a result-lifetime correctness test for transient strings and GPU
  buffers. Force eviction/invalidation after execution but before response
  write completion. Gate: responses remain correct because data was copied or
  the producing generation stayed pinned until network ownership ended.
- Add a variable-size segment recovery negative control. Reuse a cold segment
  id with a different size class, crash/replay, and prove the loader rejects
  stale checksum/size metadata rather than interpreting old bytes as a new
  segment.
- Compare direct handle hot-route overhead against a catalog/hash lookup path.
  Required metrics: route lookup time, cache miss rate, generation validation
  cost, invalidation latency, and false-fast-route count.

### 2026-06-03 - Cicada dependably fast multi-core in-memory transactions

**Citation:** Hyeontaek Lim, Michael Kaminsky, and David G. Andersen.
"Cicada: Dependably Fast Multi-Core In-Memory Transactions." SIGMOD 2017,
pp. 21-35. doi:10.1145/3035918.3064015. Retrieved 2026-06-03 from the
author PDF, `https://hyeontaek.com/papers/cicada-sigmod2017.pdf`; ACM DOI
page is `https://dl.acm.org/doi/10.1145/3035918.3064015`.

**Category:** transaction processing / write path and concurrency control.

**Relevance tags:** serializable MVCC; optimistic concurrency control;
multi-core OLTP; distributed clocks; timestamp allocation; deferred index
updates; rapid garbage collection; contention regulation; read-only snapshots;
write-path validation.

**Core idea:** Cicada combines optimistic execution, multi-version records,
and loosely synchronized per-thread clocks to keep serializable in-memory
transactions fast across both low- and high-contention workloads. The paper's
central claim is that MVCC does not have to be slower than single-version OCC
if version search, timestamp allocation, index updates, garbage collection, and
abort backoff are all designed for multicore cache behavior.

The design avoids several common OLTP traps at once. Transactions read shared
committed versions without in-place overwrite copies, prepare writes in local
versions, validate at a chosen timestamp, and only then install or commit
versions and index changes. Per-thread clocks remove the global timestamp
counter bottleneck. Best-effort inlining keeps read-mostly small versions near
record metadata. Rapid quiescent-state garbage collection keeps version chains
short. A global hill-climbing backoff controller regulates abort pressure when
contention is high.

The evaluation compares Cicada with Silo, TicToc, FOEDUS, MOCC, 2PL,
Hekaton, and ERMIA on one 28-core dual-socket server. With persistent logging
and remote clients disabled, Cicada reports up to 2.07M TPC-C transactions/s,
56.5M YCSB transactions/s, and 356M scanned records/s. The paper also reports
up to 3x higher throughput than the next fastest design on contended TPC-C and
shows that replacing Cicada's multi-clock timestamp allocation with a shared
atomic counter drops one high-speed YCSB case from 56.5M to 6.22M tps.

**Concrete mechanisms:**

- Each worker owns a 64-bit local clock. A transaction timestamp combines an
  adjusted local clock with a thread-id suffix, giving unique monotonically
  increasing per-thread timestamps without incrementing one shared counter.
- One-sided clock synchronization periodically reads another worker's clock and
  catches up if the remote clock is ahead. Temporary clock boosting after an
  abort helps a thread escape conflicts caused by a too-early timestamp.
- Read-write transactions use the worker's write timestamp. Read-only
  transactions use a global safe read timestamp, do not track or validate a
  read set, and see a consistent slightly stale snapshot.
- Records are version chains sorted from latest to earliest by write
  timestamp. Each version carries write timestamp, read timestamp, status, data,
  allocation metadata, and immutable fields except status/read timestamp.
- Version search skips later timestamps, waits briefly for pending versions,
  ignores aborted versions, and selects the first committed visible version.
  Writes can early-abort when the visible version's read timestamp or a later
  committed/pending version makes validation likely to fail.
- Validation installs pending versions, updates read timestamps for versions in
  the read set, and rechecks that read and write sets remain serializable at
  the transaction timestamp.
- Before validation, the write set is partially sorted by approximate
  contention using latest-version timestamps so likely conflicts are checked
  first. An early version-consistency check avoids installing pending versions
  that would immediately become garbage.
- Best-effort inlining stores small versions directly in the record head when
  possible and promotes old read-mostly non-inlined versions after they are
  safe, reducing pointer chasing without turning hot update records into an
  inlining contention point.
- Multi-version indexes are ordinary Cicada tables storing record ids. Range
  and absent-key reads add leaf/index nodes to the read set; inserts/removes add
  modified nodes to the write set, deferring index changes until validation and
  avoiding global index mutation by transactions that later abort.
- Redo logging is sketched as per-NUMA logger threads receiving validated write
  sets and appending per-thread redo logs before versions are marked committed.
  Checkpoint threads asynchronously write latest committed versions and advance
  quiescent timestamps.
- Garbage collection uses fine-grained timestamps plus QSBR-style quiescence.
  Threads enqueue committed versions that make older versions reclaimable;
  once all workers have quiesced and the global minimum read timestamp has
  advanced, old versions are detached and returned to local memory pools.
- Contention regulation uses a leader thread to hill-climb the global maximum
  randomized backoff time based on observed committed throughput, rather than
  relying on per-thread local abort heuristics.

**GPU DB mapping:** Cicada's strongest lesson for GPU DB is that MVCC
performance is a whole-system property. A cheap visibility predicate alone is
not enough; timestamp allocation, version layout, index mutation, garbage
collection, and abort/admission policy can each become the bottleneck. For a
future high-throughput write path, the engine should avoid one global
transaction-id counter or one global mutation queue becoming the serializing
point for all sessions.

The multi-clock idea maps naturally to owner domains. Mutation owners or
partition owners can allocate local generation/timestamp ranges, publish
visibility boundaries, and periodically synchronize safe read boundaries
without every transaction contending on a single atomic counter. For
SQL-visible external consistency, Cicada's delayed-notification option is a
warning: if commit acknowledgement must reflect a global visibility order,
that latency must be measured explicitly rather than hidden inside throughput
numbers.

Cicada's read-only snapshot path also informs retained GPU snapshots. A
read-only retained route should be able to use a precomputed safe generation
without tracking a per-query read set or enqueuing through the mutation owner.
That is close to the current read snapshot publication target in
`11-high-throughput-query-runtime.md`, but Cicada adds an implementation hint:
publish a safe read timestamp/generation from per-owner minima and make its
staleness visible in telemetry.

Deferred index updates are directly relevant to COPY/INSERT admission and
resident invalidation. If aborted or rejected transactions modify global CPU
indexes, value indexes, or residency metadata before validation, they can create
exactly the index contention Cicada avoids. GPU DB should prepare value-index
updates, resident invalidations, and refresh notices locally or in owner-private
buffers, then publish them only after WAL and validation make the mutation
eligible for visibility.

Rapid GC maps to snapshot retirement and tier pressure. Long retained GPU
reads, old CPU MVCC versions, stale resident generations, and old deleted-key
side structures must have fine-grained retirement telemetry. A coarse
millisecond-scale epoch can keep too much cold state in hot CPU or GPU memory
when the engine is creating many versions or invalidations per second.

Contention regulation should influence admission. Under high write contention,
blind immediate retries from many pgwire sessions can waste CPU, owner queue
capacity, cache bandwidth, and possibly GPU refresh work. A global or
partition-local backoff/admission controller that optimizes committed
throughput and tail latency is a better first benchmark than unlimited retries.

**Risks and mismatches:** Cicada is a single-node in-memory OLTP engine, not a
PostgreSQL-compatible GPU database. Its evaluation disables persistent logging
and remote clients, so the reported throughput does not include WAL flush
latency, pgwire framing, response writes, network backpressure, GPU residency,
CUDA work, or 1M logical sessions. The durability section is a design sketch,
not the measured configuration.

The multi-clock design does not provide external consistency by default across
threads. Delaying commit acknowledgement until a safe global minimum advances
can add latency, which may matter more for a SQL service than for benchmark
throughput. Cicada also spin-waits on pending versions, which is acceptable
only if pending windows remain very short; GPU DB must not let long refreshes,
WAL stalls, or CUDA work hold equivalent pending states on hot write paths.

Best-effort inlining is CPU-cache-oriented and does not directly solve GPU
columnar layout or HBM residency. Deferred multi-version indexes assume
Cicada's record-id table model; GPU DB will need separate handling for SQL
catalog identities, row ordinals, resident partitions, and rebuildable
acceleration indexes. Finally, the global backoff hill climb optimizes
throughput, while GPU DB also needs p50/p99 latency and fairness among session
classes.

**Benchmark candidates:**

- Add a transaction/generation allocation microbenchmark comparing one global
  atomic counter with per-owner local generation allocation plus periodic safe
  read-boundary publication. Minimum gate: identical visibility ordering in
  WAL replay tests and measured allocation contention under concurrency.
- Instrument current write/COPY admission for early global mutation: value
  index writes, resident invalidation, catalog state changes, and response
  publication before validation/WAL eligibility. Failure condition: an aborted
  or rejected mutation can still touch global hot structures.
- Prototype deferred value-index and resident-invalidation publication for one
  write path: prepare owner-private changes, flush WAL, then publish CPU index
  and residency invalidation in deterministic order. Measure rows/s, owner
  queue wait, index contention, and invalidation latency.
- Add safe read-generation telemetry for retained routes: CPU latest
  generation, retained safe generation, generation staleness, oldest active
  reader, and version bytes pinned. Gate: retained read correctness remains
  identical while route output names the generation used.
- Build a rapid-retirement stress test with repeated updates/deletes to hot
  keys while long retained reads are active. Compare coarse epoch cleanup with
  fine-grained per-owner safe read minima. Failure condition: p95 fresh lookup
  or write latency grows with unreclaimed old versions/tombstones.
- Add contention-aware write retry/admission for one synthetic hot-row or
  hot-partition workload. Compare no backoff, fixed backoff, local backoff, and
  global/partition hill-climbing backoff. Required metrics: committed rows/s,
  abort/retry count, owner queue wait, p99 latency, and fairness between hot
  and cold sessions.
- For read-only retained snapshots, test a no-read-set route using a published
  safe generation versus a mutation-owner-validated read. Expected improvement:
  lower owner queue pressure and p50 latency; failure condition: freshness or
  visibility boundary is ambiguous in telemetry.

### 2026-06-03 - Cross-paper synthesis: hot paths need fast handles and slow-path regulators

**Papers covered:** LeanStore low-overhead transactional buffer management,
Umbra variable-size pages for SSD-backed hot working sets, and Cicada
dependably fast multi-core in-memory transactions.

**Converging design tracks:**

- **Fast handle, explicit owner.** LeanStore's swips, Umbra's owner swips, and
  Cicada's record/version heads all point to the same rule: the hot path should
  resolve a stable handle cheaply, while one owner remains responsible for
  mutable state transitions. GPU DB should route through owner-published
  generation handles rather than letting prepared plans, response caches, and
  residency metadata independently mutate validity.
- **Do not make cold work pollute hot state.** LeanStore cooling, Umbra
  variable-size active-byte budgeting, and Cicada rapid GC all separate hot
  access from old/cold/transient state. GPU DB should let long scans, stale
  retained generations, old MVCC versions, and cold partition refreshes move
  out of hot lookup/write paths quickly and observably.
- **Local first, coordinated only when needed.** Cicada's per-thread clocks and
  local version preparation match the broader owner-domain direction:
  partition or mutation owners should allocate, prepare, and validate locally,
  then publish through explicit global safe boundaries instead of contending on
  one shared counter or one global index state.
- **Slow-path regulators are correctness infrastructure.** Cooling windows,
  variable-size eviction, GC quiescence, and backoff regulation are not just
  performance features. They prevent hot state from being overwhelmed by
  scans, retries, stale versions, and tier migration.

**Category gaps:** The journal has strong recent coverage across transaction
control, runtime scheduling, tier placement, and robust planning. The remaining
near-term gap is whole-stack OLTP communication cost under real client/server
protocols and isolation boundaries; the next high-value runtime candidate is
the CIDR 2025 Looking Glass paper before returning to another storage/tiering
paper.

**Benchmark priorities:**

- Define an owner-published resident/MVCC handle format with generation,
  visibility boundary, tier location, size class, and state-word version; prove
  hot-route validation is cheaper than catalog/hash lookup without hiding
  invalidation.
- Measure cold-state pressure as one metric family: old MVCC bytes, stale
  resident generations, cooling HBM bytes, host warm bytes, pinned bytes, and
  retry/backoff state.
- Add a partition-local timestamp/generation allocator proof before global
  route scaling. Required evidence: no shared-counter bottleneck, deterministic
  safe read-boundary publication, and WAL replay equivalence.
- Treat backoff, cooling, eviction, and GC as admission controllers with p99
  latency and fairness metrics, not only average throughput metrics.

### 2026-06-03 - OLTP Through the Looking Glass 16 Years Later

**Citation:** Xinjing Zhou, Viktor Leis, Xiangyao Yu, and Michael
Stonebraker. "OLTP Through the Looking Glass 16 Years Later:
Communication is the New Bottleneck." CIDR 2025. Retrieved 2026-06-03
from the CIDR proceedings page and PDF,
`https://vldb.org/cidrdb/2025/oltp-through-the-looking-glass-16-years-later-communication-is-the-new-bottleneck.html`
and `https://vldb.org/cidrdb/papers/2025/p17-zhou.pdf`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** OLTP communication; client/server round trips; pgwire
path length; stored procedures; user-code isolation; kernel bypass; IPC;
network IO workers; session multiplexing; DB/OS co-design.

**Core idea:** This paper revisits the 2008 "OLTP through the Looking
Glass" question on modern hardware and argues that the bottleneck has
shifted from traditional engine internals toward communication. The
authors benchmark whole-stack OLTP performance using VoltDB as the main
modern single-partition OLTP engine, with PostgreSQL as a reference point,
and compare client-side transaction logic with stored procedures under
different user-code isolation mechanisms.

For simple OLTP transactions, messaging dominates. In the paper's
server-side CPU breakdown, VoltDB spends less than a quarter of cycles on
transaction processing for YCSB-C, while the DBMS networking layer and
Linux kernel dominate the rest. PostgreSQL spends more cycles in
transaction processing but still shows large kernel networking cost when
data fits in memory. The main lesson is uncomfortable for a GPU database:
making kernels or MVCC faster may improve only a minority of the
end-to-end request path if pgwire, socket handling, request queues,
response queues, and client/server round trips remain expensive.

The paper also shows why stored procedures remain attractive but hard to
productize safely. They reduce round trips and help complex transactions:
in the paper's TPC-C experiment, stored procedures reach up to 2.1x the
maximum throughput of an interactive transaction model and avoid much of
the latency added by repeated SQL parsing/planning/serialization. But
stronger isolation for user-defined code adds large communication costs.
Process isolation with shared memory and polling is the best of the
tested isolated mechanisms, yet still adds substantial overhead versus no
isolation; TCP/domain-socket IPC and VM isolation are much worse.

**Concrete mechanisms:**

- VoltDB's architecture splits network threads from partition-local OLTP
  workers. Network threads interact with Linux sockets, parse requests,
  dispatch stored procedure invocations through message passing, and write
  responses when workers finish.
- Single-partition VoltDB transactions run to completion on one worker
  without locks/latches. Multi-partition transactions serialize through a
  coordinator, but the paper focuses on single-partition work to isolate
  communication overhead in a high-performance path.
- The paper models interactive transactions by breaking a stored
  procedure into one stored procedure per SQL statement or per independent
  batch of statements. This lets it compare multiple client/DB round trips
  against a single stored-procedure invocation using the same transaction
  logic.
- The server-side path is decomposed into eight phases: network receive,
  socket read, request queuing, procedure execution, isolation overhead,
  query execution, response queuing, and network send.
- For YCSB-C at 48 connections, the reported CPU-cycle distribution for
  VoltDB is about 22.6% transaction processing, 38.45% DBMS network layer,
  and 38.95% Linux kernel. PostgreSQL reports about 55.4% transaction
  processing, 4.3% DBMS network layer, and 40.3% Linux kernel.
- In the no-isolation experiments, stored procedures reduce round trips.
  Voter sees up to 72% lower latency and 23% higher achievable throughput
  than the interactive model. TPC-C sees about 3.5x higher latency for
  interactive transactions at similar throughput and up to 2.1x higher
  maximum throughput for stored procedures.
- Kernel bypass is tested by integrating DPDK/F-Stack into VoltDB. The
  integration removes system-call, copy, interrupt, and stack overhead,
  but keeps internal request/response queue overhead. It also forces a
  major networking rewrite, exclusive NIC-style deployment assumptions,
  busy polling, and poorer operational tooling.
- Isolation levels are classified as client/server isolation, no
  isolation, language isolation, OS/process isolation, containerization,
  and virtualization.
- Process/container isolation is tested with TCP/IP, shared memory with
  busy polling, and shared memory with Unix-domain-socket notification
  between the stored-procedure process and the OLTP worker. Shared memory
  with polling performs best among isolated mechanisms, while TCP/domain
  sockets spend much more time in communication.
- VM isolation is much more expensive in the tested cloud setup. The paper
  reports guest-to-host TCP RTT of 176us versus 21us for loopback TCP,
  attributing the gap to QEMU/KVM, nested virtualization, two TCP/IP
  stacks, and VM-boundary crossings.
- The paper's research directions include DBMS network IO architecture,
  better kernel-bypass mechanisms, client-side network-stack bypass,
  eBPF/user-bypass mechanisms, DB/OS co-design, WebAssembly sandboxing,
  and stored-procedure synthesis from client-side code.

**GPU DB mapping:** This is directly relevant to the current runtime
target in `11-high-throughput-query-runtime.md`. The benchmark endpoint
already exposed the same shape of problem: engine and retained CUDA work
can be microsecond-scale while client-visible latency is dominated by one
process/thread/request path, pgwire, response writes, and queue wait. The
paper strengthens the case that GPU DB should treat network/protocol
runtime as a first-class performance component, not a wrapper around the
engine.

The eight-phase breakdown maps cleanly to GPU DB telemetry. Current
reports should keep separating socket receive/parse, ingress queue wait,
owner dispatch, retained snapshot acquisition, GPU queue wait, kernel
time, result materialization, response queue wait, and socket write. A
single "query latency" number is not enough to decide whether to optimize
CUDA kernels, MVCC visibility, command rings, row encoding, or pgwire
writes.

Stored-procedure results suggest a future transaction-shape benchmark:
compare many interactive SQL statements over pgwire with a server-side
transaction bundle that stays inside one owner/partition boundary. GPU DB
does not need unsafe in-process user code to learn from this. It can first
support explicitly declared stored transaction templates or prepared
multi-step command batches whose visibility, WAL, and partition ownership
are checked once and whose result set is encoded once.

The isolation section is a warning against pretending "just add
sandboxing" is free. If GPU DB eventually supports user-defined logic,
tenant-provided filters, or procedural transaction templates, the runtime
must budget for the isolation boundary. The near-term safer transfer is
to keep arbitrary user code out of the engine and instead benchmark a
small fixed DSL or stored-template path that reduces round trips without
opening no-isolation safety risks.

Kernel bypass is not an immediate implementation recommendation, but it
is a benchmark direction. The first architecture proof should use
multiplexed network IO workers, bounded response rings, reusable buffers,
and batched response writes on ordinary Linux. Only after those are
measured should DPDK/F-Stack/io_uring/eBPF-style bypass be considered,
because the paper shows bypass can move the bottleneck but also carries
large engineering and operations costs.

For the 1M logical-session target, the paper reinforces that the unit of
scale is not an OS thread or a transaction worker per client. GPU DB needs
logical sessions multiplexed over a small number of IO workers, with
bounded outstanding work, request credits, and response buffers. Session
admission should expose whether latency comes from client/server RTT,
kernel socket cost, ingress queue saturation, owner serialization, GPU
queue saturation, or response egress.

**Risks and mismatches:** The main evaluation uses VoltDB on 16-vCPU
Google Cloud instances and focuses on single-partition transactions. GPU
DB has PostgreSQL wire compatibility goals, CUDA execution, MVCC/WAL
requirements, resident GPU snapshots, and future over-resident tiers, so
the absolute throughput numbers do not transfer directly. The paper does
not solve distributed transactions, replication, GPU scheduling, or
multi-tenant stored-procedure security.

The DPDK/F-Stack experiment required a major rewrite and one network
thread because of F-Stack constraints. That makes it useful as evidence
about stack overhead, not as a ready-made implementation plan. The
stored-procedure comparison also assumes transaction logic can be moved
server-side; many real applications keep logic client-side for debugging,
deployment, language, and external-service reasons. Finally, the paper's
isolation work mostly measures Java stored procedures and IPC mechanisms,
not Rust-native extension APIs or WebAssembly, so those remain unknown
until separately measured.

**Benchmark candidates:**

- Add an eight-phase pgwire runtime profile matching the paper's shape:
  socket receive/read, parse, ingress queue, owner/worker execution,
  snapshot/GPU queue, materialization, response queue, and socket write.
  Minimum gate: phase totals explain p50/p95 latency for persistent-client
  retained reads without hiding queue wait.
- Build a stored transaction-template benchmark for a single-partition
  TPC-C-like order or payment path: compare interactive pgwire statements
  with one server-side declared template that preserves WAL-before-
  visibility. Required metrics: round trips, bytes, owner queue entries,
  parse/plan cost, response writes, throughput, p99 latency, and abort
  correctness.
- Add a "simple query communication ceiling" benchmark using a retained
  point lookup whose engine/GPU time is intentionally tiny. Gate: no
  storage/kernel optimization work is claimed unless protocol/runtime
  overhead is separately reported.
- Compare response egress strategies for same-shape retained reads:
  per-request writes, IO-worker response batching, reusable row-description
  buffers, and scatter from GPU result batches into per-session response
  slots. Failure condition: row/result correctness changes or socket
  backpressure can pin mutation-owner resources.
- Prototype logical-session credits for persistent clients: cap
  outstanding requests per session and per IO worker, then measure fairness
  and p99 latency under thousands of mostly idle sessions plus a hot subset.
  Failure condition: an idle or slow client consumes response buffers needed
  by active sessions.
- Add a kernel-boundary experiment before considering bypass: compare
  ordinary blocking sockets, epoll-based multiplexing, and io_uring if the
  codebase is ready. Required metrics: CPU cycles/request, syscalls/request,
  context switches, p50/p99 latency, and implementation complexity.
- For future user-defined or procedural logic, benchmark a safe fixed DSL
  or stored-template path against external client logic before evaluating
  WebAssembly/process isolation. Gate: measurable round-trip reduction
  without weakening memory safety, catalog isolation, or WAL/MVCC ordering.

### 2026-06-03 - AGILE: Lightweight and Efficient Asynchronous GPU-SSD Integration

**Citation:** Zhuoping Yang, Jinming Zhuang, Xingzhen Chen, Alex K.
Jones, and Peipei Zhou. "AGILE: Lightweight and Efficient
Asynchronous GPU-SSD Integration." SC 2025. arXiv:2504.19365v3,
2025. DOI: `10.1145/3712285.3759778`. Retrieved 2026-06-03 from
the arXiv abstract and PDF, `https://arxiv.org/abs/2504.19365` and
`https://arxiv.org/pdf/2504.19365`.

**Category:** multi-tier cache / data placement and GPU execution / storage.

**Relevance tags:** GPU-centric I/O; asynchronous NVMe; GPUDirect-style
storage; HBM software cache; SSD queue pairs; completion polling; request
coalescing; over-resident execution; compute/I/O overlap.

**Core idea:** AGILE extends the BaM-style GPU-centric storage path from a
synchronous model to an asynchronous one. GPU threads can issue NVMe requests
and continue useful work while a lightweight GPU service handles completion
queue polling, resource release, and request progress. The paper's strongest
transferable idea is not "let every database kernel touch SSD directly." It is
that over-resident GPU execution needs explicit asynchronous request ownership:
the request issuer, completion poller, cache-line owner, and eviction policy
must be separated enough to avoid deadlock while still being cheap enough for
GPU-scale thread counts.

AGILE also makes the software cache policy pluggable instead of hard-wiring
one replacement rule. That matters for GPU DB because a database tiering policy
cannot be purely page-reuse based. It must include snapshot generation,
visibility boundary, partition identity, predicate shape, write invalidation
risk, and response latency class. The paper is therefore useful as a mechanism
template for a future GPU execution owner that overlaps cold partition fetches
with retained query work, but it does not replace the database's MVCC, WAL, or
planner obligations.

**Concrete mechanisms:**

- The host CPU performs setup: it manages NVMe admin queues, establishes
  GPU-SSD PCIe peer-to-peer communication, exposes SSD doorbell registers to
  the GPU, allocates physically contiguous HBM for NVMe submission/completion
  queues and cache buffers, and registers device-visible physical addresses.
- User GPU kernels interact through three API shapes: `prefetch(src)` into an
  HBM software cache, `async_issue(src, dst)` for direct asynchronous movement
  between SSD addresses and GPU buffers, and an array-like synchronous wrapper
  that hides cache checks for simpler use cases.
- A lightweight AGILE service kernel runs on the GPU. It polls completion
  queues in a non-blocking fashion and releases SQ entries and user barriers
  after completions arrive, so application threads do not hold queue resources
  while waiting for SSD latency.
- Completion processing is warp-centric: a warp checks a 32-entry CQ window,
  tracks a mask of completed entries, advances the CQ doorbell when the window
  is consumed, and rotates across CQs. This keeps CQ polling parallel without
  dedicating all GPU threads to service work.
- SQ entries use explicit states such as empty, updated, and issued. A thread
  writes a command into an available SQ entry, marks it visible, and the
  serialized doorbell updater advances the SQ tail only over visible entries
  to preserve ordering and memory consistency.
- Identical requests are coalesced first at warp level using CUDA warp
  primitives and then through the software cache path, reducing redundant SSD
  reads when many GPU threads request the same page-sized data.
- The HBM software cache uses cache-line states including invalid, busy,
  ready, and modified. Busy lines prevent duplicate requests while an I/O is
  in flight; modified lines are written back before eviction; policy code can
  decide whether to wait or find another line under pressure.
- A "Share Table" extends coherency to user-provided buffers for
  `async_issue`, letting a requested object be found in a thread-owned buffer
  before falling back to the global software cache or SSD. AGILE includes a
  debug mode that tracks lock dependency chains to expose circular waits in
  custom policies.
- Evaluation uses an RTX 5000 Ada GPU and up to three PCIe Gen4 NVMe SSDs.
  The paper reports up to 1.88x speedup over a synchronous I/O model when
  computation and communication can overlap, 4KB random read saturation around
  3.7/7.4/11.1 GB/s for one/two/three SSDs, and 4KB random write saturation
  around 2.2/4.4/6.7 GB/s.
- Against BaM on DLRM inference, AGILE reports 1.3x-1.63x synchronous-mode
  gains across model configurations and up to 1.75x with asynchronous
  prefetching. The asynchronous path underperforms when the software cache is
  too small because prefetches evict data needed by the next epoch, so cache
  capacity must be sized with the application's access window.
- On BFS and SpMV graph experiments, the paper reports lower cache and I/O API
  overhead than BaM, with maximum reductions of about 3.12x for software cache
  overhead and 2.85x for NVMe I/O overhead. It also reports lower per-thread
  register use because CQ polling is moved out of application kernels.

**GPU DB mapping:** AGILE maps most cleanly to the future over-resident path in
P8. A GPU DB execution owner should be able to schedule a partition scan or
lookup batch whose first wave runs on resident HBM pages while later waves
prefetch cold partition chunks from NVMe into HBM or pinned host staging
buffers. The queue and cache state machine from AGILE gives a useful checklist:
separate request issue, completion polling, cache-line state, user buffer
coherency, and eviction; never let application work hold a scarce queue/cache
resource across a wait that only a blocked service can clear.

The paper also reinforces a runtime point from
`11-high-throughput-query-runtime.md`: service work needs an owner. For GPU DB,
that owner should probably be a GPU execution/storage service associated with
specific CUDA streams, NVMe queue pairs, HBM cache budgets, and partition
generations. Query kernels should not independently invent polling, doorbell,
cache, and eviction logic. They should submit requests to a bounded service and
receive explicit completion or fallback reasons.

For MVCC, AGILE's cache-line states need database metadata layered above them.
A ready page is not necessarily visible for a query. GPU DB cache entries must
carry table OID, partition id, column family, source WAL/transaction boundary,
visibility generation, checksum or encoding identity, and invalidation state.
Writes and DDL must invalidate or retire those entries before a newer
visibility boundary can route through them. Modified cache lines are especially
dangerous for this engine: durable database writes still need WAL-before-
visibility, so the first transfer should be read-only cold-data staging, not
GPU-originated durable mutation.

AGILE's result about cache size is a direct benchmark warning. An asynchronous
prefetch path can become slower than synchronous execution if it churns HBM and
generates extra NVMe requests. GPU DB should therefore measure working-set
window, cache-line reuse, queue-pair pressure, HBM bytes, and evictions before
claiming over-resident speedups.

**Risks and mismatches:** AGILE is a GPU/storage systems paper, not a database
system. It does not address SQL semantics, snapshot isolation, WAL durability,
DDL invalidation, catalog generations, PostgreSQL wire serving, admission
control for many client sessions, or mixed transactional writes. Its tested
applications are DLRM, BFS, SpMV, and microbenchmarks, not OLTP or HTAP query
plans. The prototype requires modified kernel/driver plumbing, exposed SSD
doorbells, GPU-visible queues, and device-specific setup that may be
operationally heavy for a database product.

The paper targets a single GPU with multiple SSDs. It discusses CPU DRAM and
multi-GPU extensions, but those are future work, so direct guidance for a
GPU/CPU/NVMe/CXL hierarchy is incomplete. The asynchronous API also requires
manual overlap planning by programmers; the paper suggests compiler support as
future work. For GPU DB, that means the planner/runtime must own overlap
decisions instead of expecting query-kernel authors to place prefetches by hand.

**Benchmark candidates:**

- Build an over-resident partition-fetch simulator before touching production
  storage: issue async reads for cold `order_line` column chunks while a GPU
  worker processes already-resident chunks. Required metrics: HBM cache bytes,
  NVMe queue depth, request coalescing, kernel idle time, D2H/H2D bytes, p50/p99
  latency, and SQL-result equivalence at one snapshot boundary.
- Add a cache-window sensitivity benchmark: vary HBM staging cache from too
  small to comfortably sized for one partition wave and prove when async
  prefetch beats synchronous fetch. Failure condition: eviction churn creates
  extra reads or worsens p99 latency.
- Define a database cache-line descriptor for cold GPU staging:
  table/partition/column, source WAL boundary, visibility generation, byte
  range, encoding id, state, last access epoch, and invalidation reason. Gate:
  a read cannot route through a line unless generation and visibility match.
- Prototype a service-owned completion poller abstraction for GPU execution
  owners, even if the first implementation uses ordinary host-mediated I/O.
  Measure whether offloading completion/progress work from query kernels lowers
  register pressure or improves occupancy.
- Test request coalescing for many same-shape point lookups into a cold
  partition: group identical page/chunk requests before storage access and
  scatter results back to per-session response slots. Gate: no duplicate
  storage reads for the same chunk within one micro-batch.
- Keep GPU-originated writes out of the first design. If writeback is explored,
  require a WAL-before-visibility proof where GPU-produced bytes are not made
  SQL-visible until the mutation owner has durable log evidence and has
  invalidated older resident generations.

### 2026-06-03 - A Wake-Up Call for Kernel-Bypass on Modern Hardware

**Citation:** Matthias Jasny, Muhammad El-Hindi, Tobias Ziegler, and
Carsten Binnig. "A Wake-Up Call for Kernel-Bypass on Modern Hardware."
DaMoN 2025. DOI: `10.1145/3736227.3736235`. Retrieved 2026-06-03 from
the official author-hosted PDF,
`https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/damon25_wake_up_call.pdf`.

**Category:** runtime / HFT / session scale and multi-tier storage I/O.

**Relevance tags:** kernel bypass; DPDK; RDMA; SPDK; io_uring; TCP overhead;
400G networking; PCIe Gen5 NVMe; CPU cycles per request; pgwire ceiling;
over-resident storage; WAL I/O; application-specific transport.

**Core idea:** The paper argues that kernel bypass has moved from an optional
optimization to a required architectural technique for I/O-heavy database
systems on current hardware. The key claim is budgetary: modern NICs and NVMe
arrays can deliver enough packets or I/Os that a traditional kernel stack
spends more CPU cycles per operation than the machine can afford, even before
the DBMS does useful work. That makes the paper a useful counterweight to
"optimize the engine first" thinking. If the GPU DB engine returns a retained
lookup in microseconds, pgwire, TCP, kernel sockets, WAL flush I/O, and cold
partition reads can still dominate throughput and latency.

The strongest transferable idea is to treat every external I/O boundary as a
cycle-budgeted subsystem. GPU DB should not jump directly to DPDK, RDMA, or
SPDK in the first production slice, but it should design queue ownership,
buffer ownership, response rings, WAL admission, and cold-tier reads so that a
kernel-bypass path can replace the ordinary Linux path without rewriting
correctness-critical MVCC or residency logic.

**Concrete mechanisms:**

- The paper computes a packet-processing CPU budget for a 400 Gbit/s NIC:
  a ConnectX-7 can handle about 280 million 64-byte messages per second, which
  leaves roughly 686 CPU cycles per message on a 64-core 3 GHz server.
- A simple 64-byte UDP transfer through the kernel stack costs about 4,032
  cycles in the paper's perf breakdown. UDP and socket processing are the
  largest pieces, but the important point is that overhead is spread across
  driver, IP, UDP, sockets, copies, allocation, and other work, so one local
  kernel tweak is unlikely to close the whole gap.
- DPDK and RDMA-style user-space networking avoid system calls, context
  switches, kernel/user copies, interrupt-driven processing, and generic
  kernel memory-management paths. The paper reports roughly 40 cycles per
  message for kernel-bypass messaging in its motivation comparison.
- On the evaluated 400 Gbit/s setup, DPDK reaches the 64-byte message-rate
  limit with about four cores, while kernel UDP does not saturate the link
  even with 64 cores. For 8 KiB messages, the kernel can eventually saturate
  bandwidth, but needs about sixteen times as many cores as DPDK.
- Latency is dominated by software, not wire time. The paper measures about
  1.2 us wire latency, about 13.7 us end-to-end kernel UDP transfer, and about
  3.5 us with DPDK. RDMA write latency is the best baseline across message
  sizes in the reported comparison.
- AF_XDP is discussed as useful when work can stay inside eBPF, but the paper
  says it was comparable to or worse than standard UDP for database-like cases
  that still require user-space processing.
- TCP remains expensive even over a DPDK-based stack. In the paper's F-Stack
  comparison, TCP over DPDK behaves similarly to kernel TCP for aggregate
  bandwidth, while raw DPDK has much lower overhead. The authors therefore ask
  whether databases need all TCP/IP guarantees, or whether a database-specific
  reliable protocol could preserve the guarantees that matter with less cost.
- For storage, the paper uses eight PCIe Gen5 NVMe SSDs and computes a
  theoretical budget of about 8.8K cycles per 4 KiB read I/O to saturate the
  array using 64 cores. Kernel paths including `pread`, `libaio`, and
  `io_uring` exceed that budget in their measurements.
- `io_uring` with registered buffers and fixed file descriptors improves the
  kernel path but is still much more expensive than user-space storage. Stock
  SPDK and a minimal custom SPDK variant complete 4 KiB reads in about 294 and
  183 cycles respectively in the paper.
- In random-read throughput over the eight SSDs, SPDK-style paths reach the
  measured peak of about 20.65M IOPS with one to two cores, while kernel paths
  require many more cores and some do not reach the device limit.

**GPU DB mapping:** For the current runtime target, this paper backs a clear
sequence. First, keep the near-term Linux implementation honest with phase
telemetry: socket receive, parse, ingress queue, owner/snapshot execution,
GPU queue, response materialization, response queue, socket write, WAL flush,
and storage prefetch should all have visible cycle or time budgets. Second,
shape the runtime around bounded IO workers, command rings, response rings,
reusable buffers, and ownership boundaries that can later sit on ordinary
sockets, io_uring, DPDK, RDMA, or a custom transport.

For the 1M logical-session target, the paper is a warning that sessions cannot
scale as kernel-thread or TCP-heavy units of work. GPU DB needs logical
sessions multiplexed onto a small number of IO workers with explicit
outstanding-request credits and response-buffer budgets. A future bypass path
should see stable database messages and buffers, not raw PostgreSQL frontend
state scattered across per-client threads.

For P8 storage, the storage half maps to cold partition and WAL questions.
Over-resident execution should be able to compare ordinary buffered/direct
Linux I/O, tuned `io_uring`, and eventually SPDK-style queue ownership using
the same database cache-line descriptors and visibility generations. WAL
admission needs the same discipline: before optimizing commit protocol or COPY
parsing, measure whether kernel flush path, group commit, queue depth, or
durable-write scheduling consumes the CPU/latency budget.

The paper also connects to the AGILE review. AGILE moves storage issue and
completion closer to GPU execution; Wake-Up Call argues that the host kernel
path cannot cheaply saturate modern storage either. The combined design track
is service-owned I/O: a GPU DB storage/execution owner should have bounded
queue pairs, registered buffers, generation-tagged cache entries, and explicit
completion progress, regardless of whether the first implementation is
host-mediated or device/GPU initiated.

**Risks and mismatches:** The paper is a five-page call-to-action with
microbenchmarks, not a full DBMS design. It does not solve PostgreSQL wire
compatibility, TLS, authentication, transaction ordering, WAL replay, MVCC
visibility, stored-procedure safety, RDMA failure handling, or operational
deployment. Raw DPDK is not reliable TCP, and the paper itself shows TCP
semantics can erase much of the bypass advantage when implemented naively.

The evaluated network path uses UDP-style microbenchmarks and specialized
hardware. GPU DB's near-term users may run on ordinary Linux kernels, cloud
NICs, and PostgreSQL clients where DPDK/RDMA deployment is unavailable or
undesirable. The safe takeaway is therefore not "replace pgwire now." It is to
make the production runtime narrow enough, measured enough, and buffer-owned
enough that bypass can be tested when hardware and deployment justify it.

**Benchmark candidates:**

- Add a cycle-budgeted pgwire ceiling benchmark: retained point lookup with
  tiny engine time, persistent clients, and phase metrics for socket read,
  parse, ingress queue, owner/GPU work, materialization, response queue, and
  socket write. Gate: p50/p99 latency and CPU/request must name the dominant
  non-engine phase.
- Build an IO-worker multiplexing proof before bypass: replace thread-per-
  client serving with bounded IO workers, reusable response buffers, and
  response rings. Required metrics: logical sessions, active sessions,
  syscalls/request, context switches, bytes copied, queue wait, p99 latency,
  and correctness under slow-client backpressure.
- Compare ordinary sockets, tuned socket options, and `io_uring` for the same
  retained-read workload before DPDK/RDMA. Failure condition: protocol
  correctness, backpressure, or error handling becomes less observable.
- Define a future transport abstraction around database messages, not pgwire
  parser internals: request id, session id, statement/template id, snapshot
  requirement, payload buffer, response buffer, and completion/error state.
  Gate: ordinary pgwire and any bypass experiment can share the same admission
  and response-ring accounting.
- Add a WAL/storage I/O budget benchmark for COPY admission: measure cycles and
  latency per durable chunk across current fsync path, direct I/O if available,
  and `io_uring` registered-buffer experiments. Minimum proof: WAL-before-
  visibility remains unchanged and results separate parser/index cost from
  durable-write cost.
- For over-resident P8, benchmark cold 4 KiB/64 KiB/segment reads through
  ordinary Linux, tuned `io_uring`, and an SPDK simulator or prototype. Required
  metrics: IOPS/GB/s, CPU cycles/I/O, queue depth, HBM/host staging bytes,
  visibility-generation match, and p99 query latency.
- Before any DPDK or RDMA route, write down which TCP guarantees GPU DB truly
  needs for each path: client SQL sessions, inter-owner commands, WAL shipping,
  and cold-tier storage service. Gate: no application-specific protocol can
  weaken ordering, authentication, replay safety, or error reporting.

### 2026-06-03 - Cross-paper synthesis: fast devices require explicit service ownership

**Papers covered:** OLTP Through the Looking Glass 16 Years Later, AGILE:
Lightweight and Efficient Asynchronous GPU-SSD Integration, and A Wake-Up Call
for Kernel-Bypass on Modern Hardware.

**Converging design tracks:**

- **Communication and I/O are now engine work.** Looking Glass shows
  client/server communication can dominate OLTP CPU time, AGILE shows
  over-resident GPU execution needs explicit asynchronous storage progress, and
  Wake-Up Call shows kernel networking and storage stacks can exceed the CPU
  budget for current devices. GPU DB should budget protocol, WAL, and tier I/O
  as core runtime components, not wrappers around CUDA kernels.
- **Service ownership beats scattered polling.** All three papers point toward
  owned service loops: network IO workers for sessions, GPU execution/storage
  owners for cold-page requests and completions, and mutation/WAL owners for
  durable visibility. Query kernels, pgwire handlers, and partition logic
  should submit bounded work to these owners instead of each inventing its own
  polling, buffer lifetime, or backpressure behavior.
- **The first bypass is architectural, not deployment.** GPU DB can prepare for
  DPDK/RDMA/SPDK/GPUDirect-style paths by using stable database messages,
  registered or reusable buffers, generation-tagged cache descriptors, and
  explicit completion states. The first implementation can still use ordinary
  Linux while preserving a clean replacement boundary.
- **Correctness metadata must travel with performance handles.** Fast I/O
  paths are unsafe unless every buffer and cache line carries enough database
  state: session/request id, table/partition/column identity, snapshot or WAL
  boundary, visibility generation, invalidation state, and response ownership.

**Category gaps:** Recent reviews now strongly cover runtime communication,
kernel/storage I/O, and GPU over-resident execution. The next balancing move
should return to transaction processing, MVCC/snapshot visibility, or write
admission rather than another GPU-OLAP or general I/O paper. Good queued
choices are Rapid Data Ingestion through DB-OS Co-design, Fast Serializable
Multi-Version Concurrency Control, ERMIA, or Moving on From Group Commit.

**Benchmark priorities:**

- Create one end-to-end request budget table that includes pgwire CPU,
  queue wait, owner/GPU execution, WAL/storage I/O, response write, and cold
  tier movement. Treat any unmeasured segment as unknown, not free.
- Build service-owned buffer pools for request, response, pinned host staging,
  and cold-tier cache lines with explicit saturation counters.
- Prove IO-worker multiplexing and response-ring backpressure under thousands
  of logical sessions before attempting DPDK/RDMA.
- For P8 over-resident work, test async prefetch and storage queue ownership
  behind a visibility-generation cache descriptor before experimenting with
  GPU-initiated or SPDK-backed production paths.

### 2026-06-03 - Autonomous commit for low-latency NVMe durability

**Citation:** Lam-Duy Nguyen, Adnan Alhomssi, Tobias Ziegler, and Viktor
Leis. "Moving on From Group Commit: Autonomous Commit Enables High
Throughput and Low Latency on NVMe SSDs." Proc. ACM Manag. Data 3(3),
SIGMOD 2025, Article 191. DOI: `10.1145/3725328`. Retrieved 2026-06-03
from the official TUM-hosted PDF,
`https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/latency.pdf`.

**Category:** transaction processing / write path.

**Relevance tags:** WAL; commit processing; decentralized logging; group
commit; NVMe SSDs; write admission; dependency checking; low-tail latency;
lock-free queues; force commit; durable visibility.

**Core idea:** The paper argues that group commit is no longer the obvious
answer for durable transaction processing on modern enterprise NVMe SSDs.
Traditional group commit amortizes slow disk writes by batching many
transactions behind one committer, but that same single-threaded batching
creates large I/O spikes, serial commit acknowledgment, and queuing delays.
The authors show that in LeanStore's decentralized logging path, transaction
execution itself is negligible in a high-throughput YCSB commit-latency
breakdown; queuing and dependency-check acknowledgment dominate.

Autonomous commit replaces one large commit round with worker-local small
flushes plus parallel commit acknowledgment. The storage premise is concrete:
their Kioxia enterprise PCIe 5 NVMe SSD delivers low-latency small random
writes, with 4 KiB writes around 11 us in the paper's microbenchmark and
latency still low under parallel writers. The commit protocol therefore leans
into device parallelism rather than hiding it behind a single batching thread.
The authors report microsecond-range 90th-percentile commit latencies across
YCSB, TATP, and TPC-C variants, and a YCSB throughput improvement of 26.1%
over their best queued group-commit competitor in the main comparison. In
their scalability test, the 16 KiB autonomous variant reaches about 11 million
transactions per second with 192 hardware threads.

**Concrete mechanisms:**

- Each worker owns a local log buffer and flushes it once dirty log entries
  reach a small configurable log flush unit. The evaluated variants use 4 KiB
  for lowest latency and 16 KiB as a throughput/latency compromise.
- Autonomous log flush separates durability progress from a single group
  committer. Workers submit independent small writes instead of waiting for
  a central thread to collect hundreds of megabytes of log records.
- Commit acknowledgment is also decentralized. A worker checks commit
  eligibility for its own pre-committed transactions, or for a small
  acknowledgment group, instead of waiting for one global thread to inspect
  every worker queue.
- The paper keeps the two correctness conditions explicit for decentralized
  logging: a transaction must first become `HARDENED` when its log records are
  durable, then `COMMITTED` only after its dependencies are committed.
- Acknowledgment groups trade synchronization cost against queuing delay. The
  paper finds group sizes of two or four are robust across YCSB and TPC-C,
  while larger groups can help TPC-C but hurt very high-rate YCSB.
- Log stealing reduces latency when small transactions do not fill a worker's
  local flush unit quickly. A worker clones dirty log bytes from peers in the
  same topology group, claims them with CAS on a clean cursor, flushes the
  combined buffer, and publishes durability for the stolen range.
- Stealing is constrained by CPU topology, for example within an L3-sharing
  group, to avoid excessive inter-die traffic on large server CPUs.
- Out-of-order stealing completion is handled by merging notification tasks,
  so a later physical write does not publish a worker's durable prefix ahead
  of earlier stolen log records.
- Under low load, force commit predicts idle periods and probabilistically
  triggers flush plus acknowledgment so sparse transactions do not wait
  forever for a threshold-sized log batch.
- For Global Sequence Number dependency tracking, barrier transactions advance
  idle workers' GSNs without modifying data or writing log records, avoiding
  the straggler problem where one idle worker pins the global minimum durable
  GSN.
- The transaction queue is a single-producer/single-consumer circular
  lock-free queue with serialized transaction metadata stored contiguously and
  cache-line aligned, reducing allocation and latch contention in the commit
  path.

**GPU DB mapping:** This paper is directly relevant to COPY/INSERT commit
latency and to the "WAL-before-visibility" boundary in the current runtime
document. GPU DB should not treat group commit as the only durable publication
shape. A mutation owner or partition owner can still preserve ordered
visibility while allowing owner-local WAL fragments to harden in small,
parallel, generation-tagged writes. Visibility publication remains separate:
rows, resident invalidations, and GPU snapshot generations become SQL-visible
only after the relevant durable ranges and dependency frontiers are satisfied.

The strongest transferable mechanism is a three-frontier write path:
`executed`, `hardened`, and `visible`. Today those may collapse inside one
owner loop, but the benchmark design should expose them separately. COPY
admission can batch row encoding and index preparation, WAL owners can harden
small aligned buffers, and a visibility publisher can acknowledge only the
safe prefix whose dependencies and invalidations are complete.

Autonomous acknowledgment also maps to partition ownership. If future GPU DB
partitions own disjoint write sets, a central commit acknowledger would become
the same bottleneck the paper identifies. Per-owner committed-state summaries,
small acknowledgment groups, and explicit dependency frontiers fit the
existing owner-domain model better than one global commit queue.

Log stealing is useful, but only as an owner-aware mechanism. For GPU DB, a
worker should not steal arbitrary WAL bytes if that blurs ownership of table
invalidation, relation generation, or partition visibility. A safe variant
would steal only sealed WAL fragments carrying table/partition identity,
source transaction range, dependency summary, and a callback for durable-prefix
publication.

The low-load force-commit and barrier ideas matter for 1M logical sessions.
Interactive sessions and bursty clients can produce sparse writes that never
fill a large group-commit batch. The runtime should have a latency ceiling for
durable publication, and idle partition owners should advance harmless
frontier markers so retained readers and GPU snapshot retirement are not
pinned by inactive owners.

**Risks and mismatches:** The design assumes enterprise NVMe behavior, direct
I/O/block-device-style logging, and enough independent write parallelism. It
may underperform on mechanical disks, weak consumer SSDs, cloud volumes with
opaque flush semantics, or file-system paths where `fsync` remains expensive.
The evaluated system is LeanStore on CPU, not a GPU database, and the paper
does not address PostgreSQL protocol serving, CUDA streams, resident cache
invalidation, DDL, replication, or SQL planner routing.

Autonomous commit does not remove the need for correct dependency tracking.
GSN/RFA is low-overhead but uses a weaker commit condition; barrier
transactions mitigate stragglers but do not magically provide precise
causality. GPU DB should treat the paper as a durable publication design, not
as a full MVCC or serializability proof. Log stealing also adds subtle
publication-order hazards; any prototype needs replay and crash tests before
throughput numbers matter.

**Benchmark candidates:**

- Add a WAL commit microbenchmark with explicit `executed -> hardened ->
  visible` phase timers. Compare current group/batch flush, owner-local 4 KiB
  and 16 KiB aligned flush units, and a latency-ceiling force-commit mode.
  Gate: WAL replay produces identical CPU table state and resident
  invalidation generations.
- Build a COPY admission benchmark with bursty clients: many small commits,
  idle gaps, and one long retained read snapshot. Measure commit p50/p90/p99,
  WAL bytes/write, IOPS, write amplification, visible-generation lag, and GPU
  snapshot invalidation delay.
- Prototype per-partition durable frontiers. Each partition owner reports a
  local `hardened` prefix and a local `visible` prefix; a read snapshot chooses
  only a complete frontier vector. Failure condition: a retained read observes
  a row whose WAL range is not replay-safe.
- Test acknowledgment group sizes for GPU DB owner domains: one owner, one
  NUMA/L3 group, and all owners. Required metrics: synchronization cost,
  visible-prefix lag, queue wait, abort/retry behavior, and throughput under
  skewed hot partitions.
- Simulate safe log stealing with sealed WAL fragments only. Gate: stolen
  fragments cannot publish visibility until their original owner's earlier
  fragments are durable and their invalidation callbacks have run.
- Add low-load force-commit and barrier-frontier tests. Sparse write sessions
  should commit under a configured latency ceiling, and idle owners should not
  pin global snapshot retirement or old resident-generation cleanup.

### 2026-06-03 - Modern NVMe storage-engine exploitation

**Citation:** Gabriel Haas and Viktor Leis. "What Modern NVMe Storage
Can Do, And How To Exploit It: High-Performance I/O for
High-Performance Storage Engines." PVLDB 16(9), 2023, pp. 2090-2102.
DOI: `10.14778/3598581.3598584`. Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol16/p2090-haas.pdf`.

**Category:** multi-tier cache / data placement and storage-engine runtime.

**Relevance tags:** NVMe arrays; explicit buffer management; cold partitions;
out-of-memory OLTP; cooperative scheduling; 4 KiB pages; `io_uring`; SPDK;
queue depth; page eviction; CPU budget; direct I/O.

**Core idea:** The paper argues that modern NVMe arrays are no longer a
slow side tier that can be hidden behind old page-fault assumptions. With
eight PCIe 4.0 enterprise SSDs, the authors measure 12.5 million random
4 KiB reads per second, yet existing storage engines use only a fraction
of that capability. Their LeanStore redesign closes much of the gap by
making out-of-memory I/O a hot, scheduler-owned path instead of a blocking
background service.

The headline result is deliberately database-shaped, not only a storage
microbenchmark. With a 16 GB buffer pool and 160 GB TPC-C database,
their optimized LeanStore reaches 1.07 million TPC-C transactions per
second, while RocksDB and WiredTiger are far lower in the same experiment.
With a 400 GB buffer pool and a 4 TB TPC-C database, LeanStore still
reaches about 1.1 million transactions per second, saturating the mixed
read/write bandwidth of the eight SSDs. In the read-only lookup workload,
LeanStore reaches 13.2 million lookups per second because about 10% of
lookups are served from memory while the rest saturate the SSD array.

**Concrete mechanisms:**

- The system uses 4 KiB pages as the best tradeoff among random IOPS,
  bandwidth, latency, and I/O amplification. Larger pages improve byte
  bandwidth but waste too much I/O on small OLTP records.
- SSD parallelism is treated as an explicit scheduling requirement. The
  paper reports that roughly 1000 outstanding I/Os, more than 100 per
  device, are needed to saturate the eight-drive array.
- Blocking `pread` with one OS thread per request is rejected because it
  needs hundreds or thousands of worker threads to maintain queue depth,
  causing context-switch and kernel overhead.
- LeanStore switches to DBMS-managed cooperative tasks. A fixed number of
  worker threads multiplex many lightweight user tasks; page faults yield
  to the scheduler instead of blocking a kernel thread.
- Workers run a symmetric loop that executes user tasks, submits I/O,
  performs eviction, and polls completions. Page eviction and dirty-page
  writing become scheduler work instead of separately tuned background
  threads.
- The I/O backend abstracts `libaio`, `io_uring`, and SPDK. Kernel bypass
  is useful for CPU efficiency, but the paper finds `io_uring` with I/O
  polling can also reach full TPC-C throughput in this setup, albeit with
  more CPU threads.
- Each worker can access all SSDs through per-thread I/O channels. The
  authors reject dedicated I/O threads and single-SSD assignment for the
  main design because both add message passing or special roles.
- Out-of-memory code paths are partitioned and made lock-light. Page
  replacement and I/O-manager data structures are partitioned by page id
  to prevent a formerly cold global lock from becoming the bottleneck.
- A custom RAID 0 layer avoids Linux RAID limits for very high random-read
  throughput.
- The paper treats CPU cycles per I/O as a first-class budget. At 12
  million IOPS on their 64-core AMD server, the rough budget is only about
  13k cycles per I/O before query processing, indexes, MVCC, logging,
  eviction, and scheduling are counted.

**GPU DB mapping:** This is a direct design source for GPU DB's future
cold-partition and over-resident storage tier. A cold table segment should
not enter the runtime as a blocking file read hidden below a query worker.
It should enter as a bounded storage request with page or segment identity,
visibility generation, destination buffer ownership, queue-depth telemetry,
and a completion event that can wake the waiting query or prefetch task.

The strongest transferable idea is that tier I/O must live in the same
scheduler budget as query work. GPU DB already wants network IO workers,
owner domains, response rings, and GPU execution workers. This paper says
the storage tier needs the same treatment: a small number of service-owned
workers or owner loops should keep enough I/O outstanding, run eviction and
promotion work as tasks, and make queue depth, completions, and CPU cycles
visible. A thread-per-cold-read model would repeat the same mistake as the
current thread-per-client benchmark endpoint.

The 4 KiB lesson maps to cache descriptors rather than a fixed final page
size. GPU DB may still execute from larger column groups in HBM, but the
NVMe-facing unit should be benchmarked at 4 KiB, 16 KiB, and segment-sized
granularities with explicit I/O amplification. For point reads and sparse
lookups, 4 KiB cold fetches may be the right storage unit; for GPU scans,
larger compressed column blocks may win. The planner should know which
unit it is buying.

The all-to-all I/O-channel model also matters for partition ownership. A
future implementation can start with ordinary `io_uring` and one storage
service API, while preserving a later SPDK path behind the same request
descriptor. The safe abstraction is not "read file"; it is "fetch page or
segment for table/partition/generation into this owned buffer and publish
completion only if the generation is still valid."

**Risks and mismatches:** The evaluation disables logging and uses a low
isolation level to keep concurrency control from dominating, so the TPC-C
numbers are storage-engine I/O evidence, not a full durable serializable
DBMS result. GPU DB cannot copy those numbers into a WAL/MVCC benchmark
without measuring durable writes, visibility publication, and replay.

The setup uses eight enterprise SSDs, direct I/O, a custom RAID layer,
disabled IOMMU, erased drives before experiments, and carefully tuned
hardware. Cloud block devices, consumer SSDs, filesystems, encryption,
IOMMU, containers, or full drives may behave differently. SPDK also implies
exclusive device access and operational complexity. The near-term benchmark
should therefore compare ordinary Linux direct I/O and `io_uring` first,
with SPDK as a later ceiling experiment.

The paper is CPU storage-engine work. It does not solve GPU buffer
registration, GPUDirect Storage, CUDA stream ordering, GPU page-cache
replacement, PostgreSQL protocol latency, transaction dependency tracking,
or resident snapshot invalidation. Its value is the service shape and
measurement discipline.

**Benchmark candidates:**

- Add an over-resident cold-fetch microbenchmark that varies 4 KiB, 16 KiB,
  64 KiB, and resident-segment fetch units across ordinary direct I/O,
  `io_uring`, and a future SPDK ceiling. Required metrics: IOPS, GB/s,
  CPU cycles per I/O, outstanding requests, p50/p99 latency, I/O
  amplification, and generation-valid completion rate.
- Build a storage-service request descriptor:
  table id, partition id, column group, page/segment id, visibility
  generation, destination buffer id, request owner, and completion state.
  Gate: stale-generation completions cannot publish into a retained read.
- Measure queue-depth requirements on Richard's actual newer GPU host:
  the first proof should find the depth needed to saturate one NVMe device
  and then N devices, before combining storage with CUDA work.
- Extend P8 placement benchmarks so hot HBM, warm host memory, and cold NVMe
  all report promotion, demotion, eviction, and free-buffer pressure in the
  same timeline as query latency.
- Compare storage-worker shapes: dedicated storage threads, owner-loop
  integrated polling, and per-partition storage channels. Failure condition:
  a shape reaches high bandwidth only by hiding queue wait, starving short
  retained reads, or weakening invalidation ordering.
- Add a WAL-plus-cold-read interference benchmark. Mix COPY/WAL writes,
  cold partition reads, and retained hot reads on the same NVMe device.
  Required output: whether writes inflate read p99, how admission reacts,
  and which traffic class gets priority.
- Treat every cold-tier route as a planner decision with a measured CPU/I/O
  budget. A GPU route that saves CUDA time but burns excessive storage
  submission CPU should lose to a CPU or host-memory route under pressure.

### 2026-06-03 - Fast serializable main-memory MVCC

**Citation:** Thomas Neumann, Tobias Muehlbauer, and Alfons Kemper.
"Fast Serializable Multi-Version Concurrency Control for Main-Memory
Database Systems." SIGMOD 2015, pp. 677-689. DOI:
`10.1145/2723372.2749436`. Retrieved 2026-06-03 from the ACM DOI page
and the TUM author PDF,
`https://www-db.cs.tum.edu/~muehlbau/papers/mvcc.pdf`.

**Category:** MVCC / snapshot / visibility and transaction processing.

**Relevance tags:** serializable MVCC; snapshot isolation; predicate
validation; undo buffers; before-image deltas; scan-friendly versioning;
garbage collection; retained snapshots; write admission.

**Core idea:** The paper presents HyPer's MVCC design for main-memory
HTAP: keep the newest tuple version in-place for scan speed, store older
versions as before-image deltas in transaction undo buffers, and add a
serializability check that validates recently committed writes against a
committing transaction's logged read predicates. The goal is to avoid the
usual tradeoff where snapshot isolation is fast but serializability is too
expensive for read-heavy or analytical transactions.

The transferable claim is not that every workload should use this exact
HyPer layout. It is that serializable validation can be made proportional
to recently committed writes during the transaction lifetime rather than
to the full read set. For a GPU database, that is the right asymmetry:
retained scans or batched GPU reads may touch millions of rows, while the
dangerous validation surface for a short update transaction is often the
small set of writes committed since its snapshot boundary.

**Concrete mechanisms:**

- Each transaction receives a start timestamp; update transactions draw a
  commit timestamp at commit, and commit timestamp order is the
  serialization order.
- Updates modify the latest tuple version in-place and append
  before-image deltas to the updating transaction's undo buffer. The tuple
  stores hidden version metadata and a pointer into the version chain.
- Uncommitted versions use temporary high transaction identifiers so only
  the writer can read its own writes. Other writers that encounter an
  uncommitted version abort and restart.
- A reader reconstructs its visible tuple by starting from the latest
  in-place value and applying before-image deltas until it reaches the
  version valid for its start timestamp.
- Update transactions validate serializability at commit by drawing their
  commit timestamp, then scanning undo buffers of transactions that
  committed after the validator's start timestamp.
- Instead of logging every read row, the transaction logs predicates per
  relation and per access path. Index point/range reads become predicates;
  nested-loop index reads can be coarsened into ranges.
- Validation checks each recently committed insert, delete, and update
  against the validator's predicate space. Inserts detect phantoms;
  deletes test whether the removed row belonged to the read set; updates
  test both before-image and after-image.
- Predicate trees compact repeated predicates and evaluate candidate rows
  with cheap per-attribute comparison summaries. The implementation also
  supports attribute-level validation to reduce false aborts when read and
  written attributes do not overlap.
- Garbage collection advances at commit time. Undo buffers older than the
  oldest visible transaction are removed from the recently-committed list,
  version-chain references are tombstoned atomically, and memory is reused
  only after no active transaction can still hold a chain traversal
  reference.
- Indexed-attribute updates are represented as delete plus insert so
  indexes retain entries for all versions visible to active transactions;
  index cleanup follows MVCC garbage collection.
- VersionedPositions synopses record the first and last versioned record
  within fixed record ranges, allowing generated scan code to skip branchy
  version checks across long unversioned spans.
- Evaluation reports that predicate logging overhead is small in their
  TPC-C/TATP tests, that VersionedPositions improve scan performance by
  more than 5.5x over their no-synopsis MVCC scan variant, and that
  validation cost mostly follows the committed write set during a
  transaction rather than the read-set size.

**GPU DB mapping:** This paper sharpens the current retained-snapshot
plan. GPU DB should separate the read snapshot handle from validation
metadata: a retained GPU scan can hold an immutable generation and a
logical predicate summary, while mutation owners validate only against
writes that crossed the generation boundary. That avoids copying or
pinning huge read sets for 1M logical sessions.

The before-image undo-buffer layout is a useful CPU-side contrast to the
current tuple-version chains. For P8, the latest CPU-visible row or
column-group entry can remain optimized for fresh reads and refresh
builds, while old versions live in owner-local undo or delta buffers used
only by retained snapshots and validation. GPU resident snapshots should
remain immutable acceleration state, but their invalidation descriptors
could carry the same ingredients: relation, columns touched, predicate or
key range, before/after value summaries, and commit generation.

Predicate-space validation maps naturally to route admission. Same-shape
retained lookups and scans already know their relation, selected columns,
predicates, and snapshot generation. Recording those summaries in a compact
per-session or per-request structure gives the mutation owner a way to
decide whether a recent write invalidates a route, forces CPU fallback, or
aborts/retries a serializable transaction. Attribute-level validation is
especially relevant for columnar GPU layouts: an update to a non-read
column should not invalidate a read route that never observes that column.

VersionedPositions suggests a benchmarkable resident metadata layer.
Instead of checking every row for version state on the GPU path, P8 can
track per-segment dirty/versioned intervals or bitmaps. Fresh resident
segments run branch-light kernels; only marked ranges consult CPU/GPU delta
metadata or fall back. The lesson is to make "mostly unversioned" a fast
path with stable dimensions, not a branch inside every tuple operation.

**Risks and mismatches:** The paper is single-node, main-memory HyPer
work from 2015. Its implementation uses short critical sections, ordinary
latching for some data structures, in-memory redo-log experiments, and
CPU generated scan code rather than CUDA execution. It does not address
WAL flush ordering on modern NVMe, GPU memory residency, GPUDirect
storage, PostgreSQL wire serving, distributed snapshots, DDL invalidation,
or 1M idle sessions.

Predicate validation favors workloads where the read set is larger than
the recently committed write set during the transaction lifetime. That is
often true for analytical reads and short OLTP updates, but it can fail
under heavy write bursts, long update transactions, broad write predicates,
or highly contended hot partitions. Predicate summaries also introduce
false-positive abort risk, especially with hashed strings, coarsened index
ranges, or complex SQL expressions. GPU DB must treat the mechanism as a
serializable-validation candidate, not as a blanket replacement for all
MVCC conflict handling.

**Benchmark candidates:**

- Add a serializable retained-read validation prototype: log relation,
  selected columns, predicate/key range, and snapshot generation for a
  retained GPU-eligible read; validate against committed write descriptors
  since that generation. Gate: identical abort/fallback decisions versus a
  brute-force read-set checker on randomized insert/update/delete tests.
- Build per-segment versioned-range metadata for P8 resident snapshots.
  Compare always-check visibility, interval/bitmap-gated visibility, and
  CPU fallback for dirty ranges. Required metrics: kernel time, branch
  efficiency, stale-read prevention, and refresh invalidation overhead.
- Compare validation cost as `|R|` grows and `|W_since_start|` varies:
  point lookup, range scan, aggregate scan, and mixed retained GPU
  micro-batches. Failure condition: validation grows with total rows read
  rather than with committed write descriptors.
- Test attribute-level invalidation for columnar resident data. Updating
  an unobserved column should not invalidate or abort a retained route that
  only reads disjoint columns, while updates to predicate columns must
  invalidate or validate precisely.
- Add a long-snapshot GC stress test with before-image or delta buffers:
  hold retained readers open, update hot rows, and verify that fresh reads,
  writes, and resident refresh do not traverse unbounded chains.
- Model commit-generation validation as an owner-domain protocol: each
  mutation owner publishes sealed write descriptors after WAL hardening;
  read workers validate only against descriptors with commit generation
  greater than their snapshot generation.

### 2026-06-03 - Cross-paper synthesis: generations need durable and logical fronts

**Papers covered:** Autonomous commit for low-latency NVMe durability,
Modern NVMe storage-engine exploitation, and Fast Serializable
Multi-Version Concurrency Control.

**Converging design tracks:**

- **Visibility should be a published frontier, not a side effect.**
  Autonomous commit separates executed, hardened, and visible states; the
  NVMe paper makes cold-tier completion an owned scheduler event; fast
  serializable MVCC draws a commit timestamp before validation and makes
  commit order the serialization order. GPU DB should publish visibility as
  an explicit generation only after WAL, invalidation, and validation
  descriptors are complete.
- **Descriptors are the common currency.** WAL fragments, cold page fetches,
  and MVCC validation can all use sealed descriptors carrying owner id,
  table/partition, generation, touched columns, key/predicate range, buffer
  ownership, and completion state. That gives the runtime one way to reason
  about backpressure, stale completions, and route validity.
- **Recent writes matter more than historical reads.** Retained GPU reads
  may scan large snapshots, but serializable validation and invalidation
  should usually test compact write descriptors since the read generation,
  not copy every read row. This matches the need for many logical sessions
  with cheap snapshot handles.
- **Storage queues and MVCC queues must coordinate.** Cold-tier fetches and
  WAL writes can share NVMe devices. Admission has to know whether the next
  generation is waiting on durability, cold data, validation, or response
  buffers, otherwise high throughput will only hide p99 latency elsewhere.

**Category gaps:** The next few runs should keep alternating between modern
transaction/MVCC papers and runtime or optimizer work. The queue still has
good candidates for write admission and DDL/snapshot interaction, including
Rapid Data Ingestion through DB-OS Co-design, Online Schema Evolution is
(Almost) Free for Snapshot Databases, and ERMIA.

**Benchmark priorities:**

- Build a generation-frontier timeline with `executed`, `hardened`,
  `invalidated`, `validated`, `visible`, `resident`, and `responded`
  timestamps for COPY, retained reads, and cold-tier reads.
- Prototype sealed write descriptors once and reuse them for WAL publish,
  retained-route invalidation, serializable validation, and GPU resident
  refresh decisions.
- Measure NVMe interference between WAL flushes and cold partition fetches
  before tuning GPU kernels for over-resident routes.
- Add a retained-read correctness gate where stale cold-tier completions,
  stale GPU resident buffers, and post-snapshot writes all fail closed with
  explicit fallback or retry reasons.

### 2026-06-03 - MosaicDB multi-source latency hiding

**Citation:** Kaisong Huang, Tianzheng Wang, Qingqing Zhou, and Qingzhong
Meng. "The Art of Latency Hiding in Modern Database Engines." PVLDB 17(3),
2023, pp. 577-590. DOI: `10.14778/3632093.3632117`. Retrieved 2026-06-03
from the VLDB PDF, `https://www.vldb.org/pvldb/vol17/p577-huang.pdf`.

**Category:** transaction processing / write path, runtime / HFT / session
scale, and multi-tier cache / data placement.

**Relevance tags:** coroutine-to-transaction execution; latency hiding;
larger-than-memory OLTP; asynchronous I/O; hot/cold data placement;
pipelined scheduling; log flush integration; oversubscription avoidance;
contention regulation.

**Core idea:** MosaicDB argues that modern OLTP engines should hide several
latency sources together rather than optimize one at a time. Existing
coroutine OLTP engines hide pointer-chasing cache misses, but storage I/O,
log flushes, background-thread scheduling, and synchronization can still
erase those gains. MosaicDB keeps the coroutine-to-transaction model and
extends it so a worker can overlap memory stalls, cold-record I/O, and log
flush completion while staying within one software thread per hardware
thread.

The strongest transferable idea is the dual queue: keep a short hot queue
for memory-resident work and a separate cold queue sized by storage
capacity. Once storage is saturated, the scheduler preferentially admits
hot transactions instead of letting slow cold transactions occupy all
request slots. That maps directly to GPU DB's need to keep retained hot
reads and short writes moving while over-resident NVMe or host-tier fetches
are in flight.

**Concrete mechanisms:**

- Transactions are modeled as C++20 stackless coroutines scheduled by one
  worker thread per core or hyperthread. A transaction can suspend on
  predicted cache misses, cold-record I/O, or commit-related asynchronous
  I/O, then resume when its prerequisite is likely ready.
- The design preserves the fast in-memory path by using selective coroutine
  nesting. Hot memory/index/version-chain access keeps the flattened
  two-level coroutine shape from CoroBase, while storage-specific functions
  become nested coroutines because storage latency is large enough to
  amortize extra coroutine-switching overhead.
- Cold records are reached through indexes and per-table indirection arrays.
  If an indirection entry points to storage, the worker issues asynchronous
  I/O and suspends the transaction. On completion, the fetched data is
  converted into an in-memory version-chain node for subsequent access.
- Each worker owns a thread-local `io_uring` module. Transactions sharing a
  worker share its submission and completion queues; SQEs are tagged with
  transaction ids because completions may arrive out of order.
- Storage-aware batch scheduling separates I/O status tracking from
  transaction context. Before resuming a transaction suspended on I/O, the
  scheduler checks thread-local I/O status directly; if the I/O is not
  complete, it skips that transaction without paying a full coroutine
  resume/suspend cycle.
- Vanilla pipelining admits a new request whenever a slot frees up instead
  of waiting for a whole batch to finish, but can let storage-bound
  transactions dominate the queue.
- Dual-queue pipelining assigns separate hot and cold queues plus a staging
  area. A transaction that moves from hot to cold access is transferred to
  the cold queue if capacity exists; the worker mostly services the hot
  queue, periodically checks cold work, and sizes cold admission by IOPS or
  bandwidth so storage is used without starving memory-resident work.
- Durability uses redo-only logging and pipelined or group commit inherited
  from CoroBase. MosaicDB removes dedicated background log-flush/release
  threads: workers issue asynchronous log flushes when buffers fill or time
  out, then lazily check completion and release transactions whose log
  records are durable.
- The one-worker-per-core discipline avoids CPU oversubscription from
  background threads and keeps the OS scheduler largely off the OLTP hot
  path.
- Coroutine interleaving also regulates latch contention. Only one
  coroutine is active per worker at a time, so the number of simultaneous
  contenders for shared structures is bounded by hardware workers rather
  than by all in-flight transactions.
- Evaluation uses a 48-core server with direct I/O, `io_uring`, a Samsung
  980 Pro SSD, Optane, and SATA SSD variants, and workloads including
  read-only/read-write hot/cold microbenchmarks, TPC-C, and a contended
  insert microbenchmark. Reported results include up to 33x higher
  throughput for larger-than-memory workloads, 1.7x TPC-C improvement over
  an oversubscribed coroutine baseline under a fixed CPU budget, and up to
  18% lower latch-contention cycles with 2.38x throughput under the
  high-contention insert workload.

**GPU DB mapping:** MosaicDB is a good runtime shape for the gap between
the current thread-per-client benchmark endpoint and the target owner/ring
runtime. GPU DB should not have one undifferentiated retained-read queue
where cold partition fetches, WAL flush waits, GPU launches, and hot
snapshot reads all consume the same slots. It needs at least hot retained
work, cold-tier fetch work, mutation/commit work, and GPU execution work as
separate bounded queues with admission tied to the resource each queue
burns.

The dual-queue policy maps cleanly to P8 tiering. Hot resident GPU or host
snapshots should have a short queue sized for latency and launch
amortization. Cold NVMe or future CXL-tier work should have a queue sized
by measured storage bandwidth, IOPS, pinned-buffer budget, and stale
generation risk. Once cold resources saturate, new hot retained reads
should still pass if they have resident-compatible snapshots, while new
cold requests should wait, fall back, or reject with an explicit overload
reason.

Selective coroutine nesting is also a useful warning. The GPU DB CPU fast
path should not wrap every index lookup, MVCC check, response encoding, and
route decision in a heavy async abstraction. Lightweight cooperative
suspension may pay for pointer-chasing or cold I/O waits, but ordinary hot
snapshot checks should remain flat and predictable until measurements show
otherwise.

For commit and COPY admission, the background-thread removal suggests a
more integrated WAL completion path. GPU DB can keep WAL-before-visibility
while letting mutation owners issue asynchronous durable writes, continue
with other admitted work, and publish visibility only when durable
completion is observed. The important part is that the completion belongs
to the owner/generation protocol, not to a detached background flusher that
silently creates scheduler pressure or unclear publication ordering.

For 1M logical sessions, the lesson is not "make every session a coroutine."
It is to keep only an admitted active subset in hot queues, classify waiting
causes precisely, and bound the physical resources behind each class. Idle
or blocked logical sessions should not reserve cold I/O slots, CUDA pinned
buffers, or commit queue entries.

**Risks and mismatches:** MosaicDB is a CPU OLTP engine layered on CoroBase,
not a GPU database and not a pgwire-serving system. Its benchmarks bypass
SQL and networking through C++ APIs, so the results do not include protocol
parsing, response encoding, client/server round trips, or PostgreSQL
compatibility costs. The hot/cold layout has no cold-record cache in the
main microbenchmark, by design, which simplifies interpretation but is not
a complete production tiering policy.

The paper's cold store is a log/indirection-array design, not GPU resident
column groups, GPUDirect Storage, or CUDA stream scheduling. It also does
not solve serializable validation, DDL invalidation, query optimization,
or resident snapshot correctness. The dual-queue idea must be adapted so
it does not starve cold work that is required for forward progress, such as
WAL flush, refresh, or a transaction holding locks. Finally, latency hiding
can improve throughput by adding in-flight work; GPU DB must cap that
extra work so p50/p99 query latency and memory pressure remain visible.

**Benchmark candidates:**

- Add a route-class queue prototype for one retained read shape:
  hot-resident, cold-fetch, mutation/commit, and response-completion queues
  with separate capacities. Gate: hot retained reads keep stable p50/p99
  latency while cold fetches are saturated.
- Build an admission experiment that sizes the cold queue by NVMe IOPS,
  bandwidth, pinned host buffer count, and stale-generation completion
  rate. Failure condition: adding cold work increases hot retained p99
  without an explicit saturation metric.
- Compare one integrated mutation-owner WAL completion loop against a
  detached flusher for COPY admission. Required metrics: rows/sec, commit
  wait, visibility publish delay, owner queue wait, OS context switches,
  and p99 retained-read interference.
- Prototype selective cooperative suspension only around CPU index or MVCC
  pointer-chasing paths. Compare flat synchronous lookup, prefetch plus
  cooperative hot queue, and full async wrapping. Failure condition: the
  coroutine path regresses hot resident lookups when no cold or cache-miss
  latency is present.
- Add a mixed hot/cold tier benchmark: resident GPU lookup or aggregate,
  host-memory fallback, NVMe cold fetch, and WAL writes on the same run.
  Required output: which queue is saturated, which class is admitted, and
  whether cold work can make progress without starving hot snapshots.
- Track logical-session active-resource ownership: queued hot request,
  cold I/O slot, WAL flush wait, GPU stream slot, response buffer, or idle.
  Minimum proof: large idle session counts do not increase active queue
  memory or pinned-buffer reservation.

### 2026-06-03 - Tesseract online schema evolution

**Citation:** Tianxun Hu, Tianzheng Wang, and Qingqing Zhou. "Online
Schema Evolution is (Almost) Free for Snapshot Databases." PVLDB 16(2),
2022, pp. 140-153. doi:10.14778/3565816.3565818. Retrieved
2026-06-03 from `https://www.vldb.org/pvldb/vol16/p140-hu.pdf`.

**Category:** MVCC / snapshot / visibility, hybrid HTAP, and
transaction processing / write path.

**Relevance tags:** transactional DDL; schema MVCC; catalog generations;
retained snapshots; DDL invalidation; online migration; change data
capture; commit pipelining; long transactions.

**Core idea:** Tesseract treats schema evolution as ordinary MVCC data
modification. A table's schema is a versioned catalog record, and a DDL
transaction updates that schema record plus any affected table data inside
the database's snapshot isolation protocol. DML transactions read the
schema version visible to their own begin timestamp and interpret data
versions that match that schema. This turns online transactional DDL from
an ad hoc locking or trigger problem into a visibility and commit-ordering
problem.

Basic data-definition-as-modification is correct but too conservative:
a long DDL transaction may touch the whole table, collide with concurrent
DML, and accumulate a huge write set. Tesseract's relaxed DDaM improves
this by migrating data out of place into a new indirection array, letting
concurrent DML pre-commit on the old array, using change data capture to
reconcile concurrent updates, and publishing the new schema in a pending
state before final completion. On a 40-core ERMIA prototype, the paper
reports online transactional schema evolution with no service downtime and
often only up to about a 10% DML throughput drop for heavyweight
copy-oriented DDL such as eager add-column migration.

**Concrete mechanisms:**

- Each table has a schema record in a system catalog table. The schema
  record is itself multi-versioned and uses the same commit timestamp
  visibility rules as ordinary records.
- Reads obtain the visible schema version before reading data. A reader
  then interprets the latest visible data version that conforms to that
  schema version.
- Writes must see both the latest record version and the latest schema
  version. The transaction stores the schema versions it used in a
  `schema_set`.
- Commit draws a commit timestamp, then verifies that each schema in the
  `schema_set` is still the latest schema for its table before stamping
  written data versions. This prevents a transaction that wrote under an
  old schema from committing after a newer schema is installed.
- DDL operations are categorized by whether they need copy and/or verify
  work: examples include modify-column, add-constraint, create-index,
  create-as-select, add/drop-column, and create/drop-table.
- Basic DDaM installs a new schema record and then migrates/verifies data
  with ordinary reads and writes in the same transaction. It is atomic and
  rollback-capable but can cause many conflicts and very large write-set
  tracking overhead.
- Relaxed DDaM creates a new indirection array for the new schema. DDL
  scan threads transform records from the old array and install versions
  into the new array, which remains invisible until the DDL completes.
- At DDL start, the migration records the old indirection-array size and
  scans only that prefix, bounding scan-phase work even if concurrent DML
  appends records.
- Concurrent DML continues against the old indirection array and may
  pre-commit through pipelined commit, but finalization waits until the
  DDL's conflict-resolution phase completes.
- The CDC phase scans log records from a recorded starting LSN through
  the DDL pre-commit point to discover and transform concurrent updates.
  Tesseract can overlap CDC with the scan phase, and uses multiple CDC
  threads.
- After the scan phase, the DDL transaction obtains a pre-commit timestamp,
  makes the new schema visible in a pending state, and directs newly
  started transactions toward the new schema so no new CDC work is added.
- Relaxed snapshots let the DDL scan migrate the latest committed version
  rather than the version visible at the DDL begin timestamp. Migrated
  versions inherit original commit timestamps because they are isolated in
  the new indirection array until publication.
- Some post-pending DML may proceed without waiting if the DDL is copy-only
  and the target record has already been migrated; verification-oriented
  DDL keeps transactions waiting because constraints may still fail.
- Old array replacement waits for existing accesses to finish via
  epoch-based memory management or reference counting.

**GPU DB mapping:** This is directly relevant to the current architecture's
catalog owner, residency owner, and retained snapshot model. GPU DB should
treat schema and route metadata as versioned data with a visible generation,
not as mutable global state that read workers sample opportunistically. A
retained GPU snapshot should carry both a data visibility boundary and a
schema/catalog generation; a read route is valid only when its resident
layout, selected columns, predicates, and response shape match the schema
version visible to the request.

Tesseract also gives a shape for online resident-layout changes. Adding a
column, changing a type representation, building a resident index, or
splitting a table/segment can be modeled as out-of-place construction of a
new resident/canonical indirection or segment generation. Existing readers
continue on the old generation, writers pre-commit through the mutation
owner, and a CDC-like phase applies concurrent writes before the new
generation becomes fully visible.

The pending-schema idea maps to GPU DB route admission. After a DDL or
resident-layout migration reaches a publication frontier, new reads should
route to the new generation only if it is complete enough for their shape;
otherwise they wait, fall back, or receive an explicit pending-generation
reason. Old retained snapshots retire by reference count or epoch, matching
the target immutable snapshot design.

For write throughput, the key warning is that DDL/refresh work should not
be just another huge transaction with a massive write set. The engine needs
sealed migration descriptors and write descriptors: start generation,
source boundary, affected tables/columns, old/new layout ids, starting WAL
or LSN, scan prefix, CDC range, pending state, and final visible generation.
Those descriptors can also drive GPU resident invalidation and response
metadata compatibility.

**Risks and mismatches:** Tesseract targets CPU in-memory ERMIA with
snapshot isolation. It does not implement PostgreSQL protocol DDL, GPU
resident column groups, WAL durability on NVMe, DDL SQL parsing, indexes
as durable performance structures, or serializable isolation. Its logging
experiments use DRAM-backed tmpfs, so the reported DML impact does not
include real durable WAL flush pressure. The paper's separate indirection
arrays are row-version structures, while GPU DB resident layouts are
columnar acceleration state that must be rebuildable from WAL/CPU truth.

CDC and pending-schema handling can also increase tail latency if the CDC
phase falls behind concurrent writes. Aborting the DDL on incompatible
concurrent DML is simple, but a production system may need policy choices:
abort DDL, abort stale DML, block a route class, or fall back to CPU. GPU
DB must keep those choices explicit and avoid publishing a resident layout
whose schema generation and data generation disagree.

**Benchmark candidates:**

- Add catalog-generation handles to retained snapshot metadata: table OID,
  schema generation, resident layout id, source WAL boundary, visibility
  boundary, and response shape id. Gate: a retained read must reject or
  fall back when any generation mismatches.
- Prototype out-of-place resident layout migration for one table: build a
  new column-group snapshot while old retained readers continue, then
  publish it after a sealed mutation/CDC descriptor is complete. Failure
  condition: new readers can observe mixed old schema/new data or new
  schema/old data.
- Add a DDL-versus-retained-read stress test: hold long retained snapshots,
  perform add-column or resident-index-build migration, and measure write
  throughput, read fallback rate, snapshot retirement delay, and p99 route
  latency.
- Implement commit-time schema validation for writes in the CPU relational
  layer: a write records the schema generation it used, and commit fails or
  retries if a newer schema was published before visibility. Gate:
  randomized DDL/DML interleavings never interpret a row under the wrong
  schema.
- Track migration descriptors with `scan_started`, `scan_done`,
  `pending_visible`, `cdc_started`, `cdc_done`, `published`, and
  `old_generation_retired` timestamps. Use them to decide whether a read
  waits, routes old, routes new, or falls back.
- Compare DDL policies under concurrent writes: abort DDL on incompatible
  post-scan writes, block stale writers during pending publication, or let
  copy-only writers proceed when their target record has already migrated.
  Required metrics: DML throughput, DDL completion time, abort counts,
  retained-read p99, and stale-generation rejection reasons.

### 2026-06-03 - Bonspiel low-tail geo-distributed transactions

**Citation:** Fan Cui, Eric Lo, Srijan Srivastava, and Ziliang Lai.
"Bonspiel: Low Tail Latency Transactions in Geo-Distributed Databases."
PVLDB 18(11), 2025, pp. 3840-3853. doi:10.14778/3749646.3749658.
Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol18/p3840-cui.pdf`.

**Category:** transaction processing / write path, runtime / session
scale, and MVCC / snapshot / visibility.

**Relevance tags:** tail latency; transaction priority; optimistic
concurrency control; early write visibility; abort penalty; access-method
selection; contention footprint; admission policy; hot/cold route choice.

**Core idea:** Bonspiel argues that after modern geo-distributed commit
protocols reduce atomic commit to roughly one WAN round trip, p999
transaction latency is often dominated by abort time rather than final
commit time. The paper targets the slow class of multi-region transactions
without sacrificing the common single-region path. Its two mechanisms are
geo-distributed concurrency control (GDCC), which prevents multi-region
transactions from being aborted by conflicting single-region transactions,
and geo-aware access method selection (GAMS), which chooses per record
between reading from the leader and reading from the nearest replica.

The GPU DB setting is not geo-distributed, but the transferable idea is
strong: tail latency should be optimized by reducing the product of abort
or retry rate and failed-round penalty for expensive route classes, not
only by shaving the successful fast path. A retained GPU read, cold-tier
fetch, resident refresh, or long write transaction can have a much larger
retry or wait penalty than a short CPU lookup. Those classes need explicit
priority and route-selection rules that cap p99/p999 damage without
wounding the short path.

**Concrete mechanisms:**

- Bonspiel classifies transactions into single-region (SR) and
  multi-region (MR). MR transactions are rarer in TPC-C-like workloads but
  dominate high-percentile latency because conflicts and retries carry WAN
  cost.
- GDCC is OCC-based. MR transactions reserve records in their read and
  write sets during execution; SR transactions do not reserve records.
- Reserve-lock conflicts use conditional waiting. If an MR transaction
  tries to reserve a record locked by an SR transaction, it proceeds
  without waiting. If the lock holder is another MR transaction, it waits.
- Multiple MR reservations on the same record can succeed in the basic
  scheme. Actual serializability conflicts are resolved later by standard
  OCC validation.
- During validation, an SR transaction aborts if it tries to lock a record
  reserved by an MR transaction. This gives MR transactions abort freedom
  with respect to SR transactions.
- Once an SR transaction has successfully validated and acquired its locks,
  it is wound-free: later transactions do not abort it. This protects the
  common path from unconditional high-priority wounding.
- SR transactions make their writes early visible after successful
  validation but before WAN logging completes, while retaining locks until
  logging completes. This lets MR transactions read fresh values without
  waiting for the SR log round, while preserving serializability and
  recoverability under the paper's assumptions.
- MR transactions do not use early visible writes, limiting cascading abort
  exposure. Bonspiel states that cascading abort chains are bounded to one
  in server-failure cases because only successfully validated SR
  transactions expose early writes and keep locks held.
- A multi-priority optimization raises an MR transaction's priority after a
  configurable number of aborts. Lower-priority MR reservations may then
  fail or wait behind higher-priority MR work, reducing repeated starvation
  among MR transactions.
- GAMS chooses per record between read-leader with reservation and
  read-nearest without reservation. It tracks page-level temperatures based
  on update frequency and adapts a threshold according to observed MR abort
  rate.
- The evaluation uses DBx1000 in C++ with simulated WAN latencies across
  five data centers, TPC-C NEW-ORDER and PAYMENT plus YCSB-A, and compares
  against Spanner, TAPIR, GPAC, and R4-style baselines. The paper reports
  up to 2.2x p999 tail-latency improvement and caps TPC-C p999 around
  1.7-1.8 seconds in its setup, while maintaining competitive average
  latency and throughput.

**GPU DB mapping:** The immediate mapping is to route-class-aware
concurrency control inside the owner runtime. GPU DB should identify
expensive classes whose failed attempts are unusually costly: long
mutation batches, resident snapshot refresh, cold NVMe over-resident
queries, multi-partition retained aggregates, and DDL or resident-layout
migrations. Those classes should not be repeatedly invalidated by short
single-partition reads or cheap writes after they have crossed a meaningful
validation or reservation boundary.

Reservations map to lightweight intent records, not locks that block the
whole system. A long retained refresh could reserve table/partition/schema
generations and route families before it starts expensive GPU or cold-tier
work. Short writers would still be admitted, but at commit they would see
the reservation and either publish after the reserved generation, redirect
to a delta/CDC lane, or wait at a narrow boundary. The goal is the
Bonspiel shape: protect the expensive route from aborts without aborting
already-validated short work.

Early visible writes are more dangerous for GPU DB because WAL-before-
visibility is a hard invariant. The safe analogue is not exposing
unflushed writes to SQL clients; it is exposing post-validation,
pre-publication state only to internal dependent work under a held owner
fence. For example, a resident refresh or batched read could consume a
sealed mutation batch once validation has passed, but user-visible snapshot
publication must still wait for durable WAL and generation publication.
This suggests a three-front model: validated intent, durable boundary, and
SQL-visible generation.

GAMS maps cleanly to CPU/GPU/tier route choice. Hot or frequently mutated
records should prefer leader/owner/current-generation paths even if they
cost more per operation, because stale retained or nearest-like reads will
retry. Cold stable records can use cheaper retained GPU, host snapshot, or
cold cached routes without reservation. The planner should therefore
estimate not only successful execution latency, but also stale-generation
probability and failed-round penalty.

For 1M logical sessions, Bonspiel reinforces that priority is a class
policy, not a thread policy. A rare expensive request class may deserve a
reservation or priority boost after retries, while most logical sessions
remain idle or cheap and should not inherit that priority. Admission should
track retry count, route class, stale-generation cause, and estimated
penalty before boosting.

**Risks and mismatches:** Bonspiel is a geo-distributed database prototype
implemented in DBx1000 with simulated WAN latencies. It does not evaluate
GPU execution, PostgreSQL protocol serving, NVMe tiering, MVCC version
storage inside a production SQL engine, or durable WAL on the local storage
path GPU DB currently cares about. Its early-visible-write mechanism is
safe only under the protocol's validation, lock-holding, and replication
assumptions; GPU DB must not expose unflushed writes as visible SQL state.

The paper optimizes MR tail latency in workloads where MR transactions are
relatively rare. If GPU DB applies similar priority to a class that becomes
dominant, short-path latency or throughput could regress. GAMS also relies
on useful update-frequency statistics and adaptive thresholds; without
good telemetry, route choice could oscillate between stale retained reads
and over-conservative owner reads. Starvation is argued empirically rather
than proven for all workloads.

**Benchmark candidates:**

- Add retry-penalty accounting to retained read and mutation telemetry:
  stale-generation rejects, fallback retries, queue waits, and failed-round
  time by route class. Gate: p99/p999 reports name whether tail is caused
  by final execution, wait, or retry/abort penalty.
- Prototype lightweight reservations for one expensive route class, such as
  partitioned resident refresh. Short writes that encounter a reservation
  must publish through an explicit delta/CDC or post-reservation boundary
  rather than silently invalidating completed refresh work. Failure
  condition: the reservation wounds already-validated short writes or
  exposes stale retained reads.
- Compare three policies for long retained refresh under concurrent writes:
  no reservation, abort-and-retry on mutation, and reservation plus delta
  catch-up. Required metrics: refresh completion time, write throughput,
  retained-read p99/p999, retry count, and stale-generation reject reasons.
- Add route-temperature statistics at table/partition/page granularity:
  update frequency, invalidation frequency, retained hit rate, and fallback
  success rate. Use them to choose owner-current, retained GPU, host
  snapshot, or cold-tier route. Gate: hot mutable records avoid stale
  retained retries, while cold stable records retain low latency.
- Test priority boost after repeated expensive-route aborts. A request
  class gains priority only after measured failed-round cost exceeds a
  threshold. Failure condition: priority boosts improve the long route by
  increasing short-route p99 beyond a configured budget.
- Model validated, durable, and visible fronts separately in mutation
  batches. Minimum proof: internal dependent work may observe validated
  sealed state only under an owner fence, and SQL-visible snapshots never
  advance before WAL durability.

### 2026-06-03 - Cross-paper synthesis: expensive attempts need protected fronts

Bonspiel, MosaicDB, and Tesseract converge on a common design track for GPU
DB: classify work by the resource and correctness front it consumes, then
protect expensive attempts once they pass a meaningful boundary. MosaicDB's
dual hot/cold queues separate storage-latency hiding from hot OLTP work.
Tesseract separates old schema, pending schema, CDC, and published schema
fronts during online migration. Bonspiel separates cheap/common transactions
from rare expensive transactions and uses reservations plus early internal
visibility to reduce retry penalty without wounding the common path.

The shared implementation hypothesis is a multi-front owner protocol:
`validated`, `durable`, `resident-built`, `pending-visible`, and
`SQL-visible` should be explicit states, not comments in the code. Different
route classes may use different fronts, but only under named invariants. GPU
kernels, resident refreshes, and cold-tier fetches can consume validated or
resident-built work internally when an owner fence proves it cannot leak
stale SQL results. Client-visible reads and writes still require the durable
and SQL-visible fronts.

Category gaps remain around network/session admission and query optimizer
integration. The journal has strong recent coverage for MVCC, storage
tiering, GPU-initiated IO, and transaction scheduling, but fewer entries on
database/network co-design and production-grade transport resource sharing.
The next high-value candidates should therefore include modern network/DB
runtime work such as Tigger, DB/network co-design surveys, ScaleRPC, or
Skyloft unless the queue needs a query-optimizer correction.

Benchmark priority should move from one-dimensional throughput curves to
classed tail breakdowns. For each mixed run, report hot retained reads,
cold-tier reads, mutation/COPY admission, refresh or DDL migration, and
response completion separately. The key pass/fail question is whether an
expensive attempt can complete without repeated invalidation while the
common short path keeps its p99 budget and WAL-before-visibility remains
untouched.

### 2026-06-03 - Tigger: a database proxy with user-bypass

**Citation:** Matthew Butrovich, Karthik Ramanathan, John Rollinson,
Wan Shen Lim, William Zhang, Justine Sherry, and Andrew Pavlo. "Tigger:
A Database Proxy That Bounces With User-Bypass." PVLDB 16(11), 2023,
pp. 3335-3348. doi:10.14778/3611479.3611530. Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol16/p3335-butrovich.pdf`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** PostgreSQL protocol proxying; connection pooling;
transaction pooling; eBPF; sockmap; kernel-space fast path; user-bypass;
session multiplexing; workload mirroring; CPU efficiency; cloud-native
connection churn.

**Core idea:** Tigger attacks a specific modern OLTP bottleneck: DBMS
proxies are useful because they multiplex many client sessions over fewer
backend connections, but conventional proxies still copy protocol buffers
between kernel-space and user-space for every request and response. The
paper's "user-bypass" design keeps the Linux TCP/IP stack and socket
semantics, but pushes a small amount of DBMS protocol logic into safe
kernel-resident eBPF handlers. The result is a PostgreSQL-compatible proxy
that preserves ordinary client behavior while turning common forwarding,
pooling, and mirroring operations into kernel-space fast paths.

For GPU DB, the strongest transferable idea is not "put SQL execution in
the kernel." It is the split between a tiny verified fast path and an
ordinary user-space slow path. If 1M logical sessions eventually pass
through pgwire-compatible IO workers, most packets should perform only
bounded framing, state lookup, route selection, and buffer handoff. Work
that requires authentication, SQL parsing, arbitrary allocation, complex
transaction semantics, or GPU scheduling should stay in owner/runtime
domains. Tigger gives a concrete shape for where that boundary can sit.

**Concrete mechanisms:**

- Tigger is built from PgBouncer but replaces hot proxy actions with eBPF
  handlers. The user-space component still performs connection
  establishment, client authentication, user/settings management, and
  exceptional protocol operations.
- Two primary sockmap-attached eBPF handlers process frontend client sockets
  and backend PostgreSQL sockets. They inspect PostgreSQL message headers,
  lengths, and selected bodies to decide whether a buffer is ordinary query
  traffic, session-control traffic, or transaction-completion traffic.
- Kernel-space state is stored in eBPF maps. Tigger uses server-socket maps,
  client-socket maps, an idle-socket stack, and per-socket state metadata to
  link a client socket to a pooled backend socket and later unlink it.
- PostgreSQL messages can span socket buffers, so Tigger records partial
  header state and the next-buffer offset in `SocketStatesMap` rather than
  rescanning blindly.
- Transaction pooling links a client to a backend only when query traffic
  arrives, then releases the backend after transaction completion. Session
  pooling holds the backend for the session and requires less transaction
  status tracking.
- If no user-bypass backend socket is available, or if an operation is not
  supported by the fast path, Tigger falls back to a user-space pool.
- Workload mirroring uses additional eBPF programs. Because socket-layer eBPF
  cannot clone buffers, Tigger clones at the traffic-control layer, rewrites
  metadata to move the clone back to a sockmap handler, and redirects it to
  replica sockets while the primary remains authoritative.
- The design deliberately attaches most DBMS protocol logic at the socket
  layer, not XDP or lower layers, so Linux still owns TCP ordering,
  retransmission, and kTLS-compatible decrypted socket buffers.
- eBPF verifier limits shape the implementation. The paper reports a client
  handler with 267 eBPF instructions, while verifier branch and loop analysis
  expands the checked instruction count substantially. Full authentication,
  full SQL parsing, and richer proxy features are kept out of the kernel path.
- Evaluation uses PostgreSQL 14.5, BenchBase OLTP workloads, 10,000-client
  connection-pooling tests, serverless-style short-lived connection tests,
  workload mirroring, and proxy CPU-efficiency tests on AWS EC2 c6i
  instances. The paper reports up to 29% transaction-latency reduction and
  up to 42% CPU-utilization reduction versus other PostgreSQL proxies in one
  scenario, and 92% lower latency plus 88% less CPU for workload mirroring
  versus Pgpool-II. In the many-client YCSB run, Tigger shows 0.40 ms mean
  and 0.76 ms p99 versus 0.62 ms mean and 1.72 ms p99 with no proxy.

**GPU DB mapping:** The most direct mapping is to the front half of the
target runtime topology: client sockets -> network IO workers -> bounded
command rings. Tigger suggests a staged design in which pgwire IO first
moves from one-thread-per-client to a small set of multiplexed workers, then
optionally promotes only the most stable framing and routing operations into
kernel-assisted fast paths. The "kernel" part is optional; the real design
rule is that the hot ingress path must be small, bounded, and mechanically
simple.

For 1M logical sessions, GPU DB should treat session state as compact route
metadata rather than as an execution thread. A frontend connection can be
linked to a backend/owner/ring only while it has active work, then released
or parked. The Tigger analogue of `IdleSocketsMap` is an explicit pool of
available mutation, read-snapshot, and GPU execution admission slots. A
logical session that has no admitted work should consume socket readiness
state and protocol metadata, not a dedicated engine worker.

Tigger's per-socket map state maps to GPU DB's request metadata: protocol
phase, transaction state, active snapshot generation, target table/partition,
route family, response shape, and linked owner queue. The fast path should
be able to decide "read-only retained route," "mutation owner route,"
"authentication/session control," "unsupported SQL," or "overload/fallback"
without touching large engine state.

Workload mirroring maps to a useful GPU DB benchmark mode: duplicate a
subset of production-like read traffic to an experimental GPU route while
returning the CPU/primary result to the client. The mirror must not become a
correctness authority. It can warm resident snapshots, compare latency and
result hashes, and collect fallback reasons before a GPU route is admitted
for real traffic.

Tigger also clarifies what should not be pushed into the fastest layer.
Authentication, arbitrary SQL parsing, DDL, multi-statement transaction
semantics, WAL-before-visibility sequencing, MVCC visibility, and GPU memory
management exceed a verifier-style mental model. Those belong in owner
domains with explicit publication fronts. A kernel/eBPF path, if ever used,
should do bounded message framing, route-token lookup, and zero-copy or
low-copy forwarding only.

**Risks and mismatches:** Tigger is a PostgreSQL proxy, not a DBMS execution
engine. It does not implement SQL planning, MVCC, WAL, result correctness,
GPU execution, or durable storage. Its benefits appear when proxy overhead,
connection churn, or connection count are important; long OLAP queries or
GPU kernels would hide much of the proxy overhead.

The implementation depends on Linux eBPF, sockmap behavior, kernel verifier
limits, and privileged deployment choices. That is a meaningful operational
risk for a portable database engine. eBPF maps are also not a natural home
for large SQL caches or complex invalidation state. If GPU DB eventually
uses eBPF, the first target should be measurement or narrow routing, not
semantic caching.

Tigger's transaction pooling also interacts with PostgreSQL features such as
prepared statements and session state. The paper disables automatically
prepared JDBC statements in BenchBase to avoid name contamination across
shared backend connections. GPU DB's pgwire compatibility must account for
prepared statements, portals, transactions, temporary state, and session
settings before multiplexing sessions aggressively.

**Benchmark candidates:**

- Replace the benchmark endpoint's thread-per-client model with a
  multiplexed pgwire IO-worker prototype. Required metrics: memory per idle
  session, active session throughput, p50/p99 request latency, context
  switches, queue depth, and response-ring wait time at 10k, 100k, and
  synthetic 1M logical sessions.
- Add a transaction-pooling admission model for short autocommit requests:
  a session holds a mutation/read slot only while a request is active.
  Failure condition: session-local state, prepared statements, or transaction
  boundaries leak across clients.
- Build a protocol fast-path classifier that reads only pgwire framing and a
  cached route token, then sends read-only retained requests directly to
  read-snapshot rings while unsupported messages go to the owner. Gate:
  randomized protocol tests produce identical results and error handling to
  the conservative owner route.
- Add shadow mirroring for retained GPU reads: execute the authoritative CPU
  path for the client, mirror eligible read-only requests to a GPU resident
  route, and compare result hashes, latency, fallback reason, and residency
  generation. Failure condition: mirrored work delays the authoritative path
  beyond a small p99 budget.
- Compare ordinary epoll, `io_uring`, and a simulated eBPF/user-bypass
  classifier for pgwire message forwarding. The proof gate is CPU cycles and
  p99 improvement after preserving authentication, TLS, prepared statement,
  and transaction semantics.
- Add telemetry that separates protocol overhead from execution overhead:
  socket read/write time, framing/classification time, owner queue wait,
  execution time, response encoding, and kernel/user copies if measurable.
  The benchmark should identify whether Tigger-like bypass would actually
  matter before adding Linux-specific machinery.

### 2026-06-03 - CARPO listwise context-aware query plan ranking

**Citation:** Wenrui Zhou, Qiyu Liu, Jingshu Peng, Aoqian Zhang, and
Lei Chen. "CARPO: Leveraging Listwise Learning-to-Rank for
Context-Aware Query Plan Optimization." arXiv:2509.03102v2, revised
2025-10-21. Retrieved 2026-06-03 from
`https://arxiv.org/pdf/2509.03102`.

**Category:** query optimization / planning.

**Relevance tags:** learned query optimization; listwise ranking; plan
candidate sets; top-k fallback; out-of-distribution detection; route choice;
CPU/GPU/tiered planning; robust learned advice.

**Core idea:** CARPO argues that pairwise learned query optimizers can make
locally plausible but globally inconsistent plan choices because they compare
two plans at a time. It instead treats all candidate plans for one query as a
set, embeds each plan, runs a Transformer over the full list to capture
inter-plan context, and trains with a listwise ranking loss. At inference, it
does not blindly trust the learned top-1 plan: a hybrid decision block checks
the top-k ranked plans with an out-of-distribution detector and falls back to
the native PostgreSQL cost-based optimizer when the learned candidates look
unreliable.

The transferable point for GPU DB is that route choice should often rank a
set of feasible routes rather than decide with independent thresholds. A
query may have CPU owner execution, retained GPU execution, CPU prefilter plus
GPU tail, cold-tier streaming, or overload rejection routes whose relative
quality depends on the whole candidate set: snapshot freshness, queue depth,
resident bytes, transfer bytes, and fallback penalty.

**Concrete mechanisms:**

- Candidate plans are generated using a Lero-like exploration strategy that
  perturbs internal PostgreSQL statistics to obtain alternatives beyond the
  native optimizer's default plan.
- Training data is built by physically executing generated candidate plans
  multiple times and sorting them by measured average latency to produce one
  ground-truth ranked list per query.
- The plan embedder is modular. CARPO discusses TreeCNN and TreeLSTM-style
  embedders over plan-tree structure, operator types, estimated costs,
  estimated cardinalities, tables, and predicates.
- The ranking predictor takes the sequence of plan embeddings for one query
  and applies Transformer self-attention so each plan representation is
  contextualized by the other candidate plans in the same list.
- The model predicts scores for assigning each plan to rank positions and is
  trained with a position-aware cross-entropy loss over the whole candidate
  list, rather than a pairwise comparison loss.
- A separate out-of-distribution detector is trained as a binary classifier
  over top-ranked plan features. At inference, CARPO checks ranked candidates
  from best to `k` and selects the first one classified as in-distribution.
- If none of the top-k learned candidates passes the confidence threshold,
  CARPO executes the native PostgreSQL CBO plan.
- The top-k strategy is motivated by the observation that several top plans
  often have very similar physical execution times; picking a nearby
  high-quality plan can be safer than forcing a brittle top-1 decision.
- Evaluation uses PostgreSQL 13.1, TPC-H, and STATS candidate plans. The paper
  reports TPC-H top-1 accuracy of 74.54% for CARPO versus 3.63% for Lero, and
  cumulative TPC-H execution-time values of 3719.16 for CARPO versus
  22577.87 for PostgreSQL and 17732.50 for Lero. On STATS it reports 1628.68
  for CARPO, 1923.09 for PostgreSQL, and 1819.99 for Lero.
- The paper's embedder comparison reports that TreeLSTM improves STATS top-1
  accuracy from 46.43% to 78.57% and lowers cumulative STATS execution time
  from 1628.68 to 1377.60 under the tested setup.

**GPU DB mapping:** CARPO fits the planner side of the route-descriptor
track already emerging in this journal. GPU DB should keep deterministic
eligibility rules first: SQL semantics, snapshot compatibility, WAL/catalog
generation, resident layout validity, memory budgets, and operator support.
Inside that safe candidate set, a learned or calibrated ranker could compare
CPU, retained GPU, CPU-prefilter-plus-GPU-tail, cold-tier GPU, and rejection
or wait policies as a list.

The listwise framing is especially useful when the "best" route depends on
relative tradeoffs. A retained GPU route with a long queue may be worse than
a CPU path; a CPU prefilter route may beat full GPU streaming only when
selectivity and transfer bytes move together; a cold-tier plan may be viable
only when it avoids displacing a hotter resident snapshot. These are not
independent yes/no checks. A CARPO-like ranker would let the planner compare
all admitted route candidates in one context vector.

The top-k fallback rule maps directly to production guardrails. GPU DB should
not execute a learned GPU route simply because a model ranks it first. It
should test the top few candidates against hard confidence gates: known query
shape, supported route family, in-distribution selectivity, observed queue
range, resident generation stability, and bounded stale-route penalty. If no
candidate passes, the engine should use the conservative CPU/owner plan and
emit an explicit fallback reason.

CARPO also strengthens the case for shadow training before planner authority.
GPU DB can mirror eligible read-only traffic, execute CPU-authoritative
results for clients, and collect ranked route evidence across CPU/GPU/tiered
alternatives. Only after enough executed-route evidence exists should a
learned ranker influence the real route, and even then only inside a hard
deterministic envelope.

**Risks and mismatches:** CARPO is an arXiv preprint and the reviewed source
does not claim a peer-reviewed venue. Its evaluation is offline learned plan
selection for PostgreSQL analytical benchmarks, not a production optimizer
inside a write-heavy MVCC engine with GPU residency, WAL durability, and
network admission. Candidate generation requires executing multiple plans,
which can be expensive or unsafe for fresh workloads. The paper's OOD
detector is described at a high level; its calibration, false-negative rate,
and behavior under workload drift would need independent validation.

The reported unit labels are inconsistent in the paper: the abstract and
tables use milliseconds, while the experiment prose describes seconds for the
same cumulative numbers. The relative comparisons are still useful, but GPU
DB should not treat those absolute magnitudes as transferable. CARPO also
optimizes successful query execution time, not retry cost, resident
invalidation, queue tail latency, or memory-tier disruption.

**Benchmark candidates:**

- Build a route-candidate logger for one retained query family. For every
  accepted SQL shape, emit the ranked set of feasible routes with features:
  snapshot generation, route family, resident bytes, H2D/D2H bytes, queue
  depth, estimated selectivity, stale-generation risk, and fallback reason.
  Gate: no route behavior changes and complete feature rows for CPU and GPU
  candidates.
- Add shadow route ranking for read-only queries: execute the authoritative
  conservative route, mirror feasible GPU/tiered alternatives, compare result
  hashes and latency, and record the best observed route per query template.
  Failure condition: shadow work increases authoritative p99 beyond a small
  budget.
- Prototype a hard-envelope top-k route selector after enough shadow data:
  learned or calibrated ranking may choose only among routes that already pass
  deterministic snapshot, residency, operator, memory, and queue gates. If no
  top-k candidate passes confidence checks, fall back to CPU/owner execution.
- Compare independent threshold routing against listwise route ranking for a
  mixed workload with retained reads, CPU fallbacks, and over-resident
  prefilter candidates. Required metrics: p50/p99 latency, wrong-route
  penalty, queue wait, H2D bytes, fallback rate, and result parity.
- Add OOD-style guardrails for GPU route estimates: flag unseen query shapes,
  selectivity ranges, resident-byte ranges, queue-depth ranges, and
  table-generation churn. Minimum proof: guardrails choose conservative routes
  during distribution shifts rather than amplifying tail latency.
- Use top-k similarity telemetry instead of only top-1 accuracy: report how
  close the top three observed route latencies are, and allow low-risk
  selection among them only when their measured penalty spread is small.

### 2026-06-03 - WeBridge synthesized stored procedures for hot paths

**Citation:** Gansen Hu, Zhaoguo Wang, Chuzhe Tang, Jiahuan Shen,
Zhiyuan Dong, Sheng Yao, and Haibo Chen. "WeBridge: Synthesizing
Stored Procedures for Large-Scale Real-World Web Applications."
Proceedings of the ACM on Management of Data 2(1), SIGMOD 2024,
Article 64, pp. 64:1-64:29. doi:10.1145/3639319. Retrieved
2026-06-03 from
`https://chuzhe.me/assets/pdf/2024%20-%20WeBridge-%20Synthesizing%20Stored%20Procedures%20for%20Large-Scale%20Real-World%20Web%20Applications.pdf`.

**Category:** runtime / HFT / session scale, with transaction processing
and query planning relevance.

**Relevance tags:** stored procedures; client/server round trips; hot-path
synthesis; concolic execution; ORM transparency; dependent SQL chains;
transaction lock-hold time; route templates; cold-path fallback; speculative
execution; session admission.

**Core idea:** WeBridge targets a communication bottleneck that matches the
recent Looking Glass and Tigger thread: web applications often issue many
interactive SQL statements through ORM or database-access libraries, and
later statements frequently depend on earlier query results. Ordinary
prefetching or batching cannot collapse those dependent round trips because
the application normally computes the dependency on the web-server side.
WeBridge records real request paths, uses concolic execution to recover SQL
data/control dependencies for hot paths, compiles those paths into stored
procedures, and transparently invokes them through an extended database
driver. The application still executes its original code, but the driver
answers its expected SQL calls from buffered stored-procedure results.

The strongest transferable idea for GPU DB is to treat repeated transaction
or read templates as synthesizable route programs, not just as individual SQL
requests. A high-concurrency pgwire path can continue to support ordinary
interactive SQL, while hot, dependency-shaped request classes are promoted
into bounded server-side programs that run near the owner, retained snapshot,
or GPU execution worker. That reduces network round trips, owner re-entry,
and lock-hold time without requiring every client to hand-author stored
procedures.

**Concrete mechanisms:**

- WeBridge splits into an offline compiler and a runtime library. The runtime
  records request inputs, SQL result sets, and external method return values;
  the compiler identifies hot paths after a replay-count threshold and
  synthesizes stored procedures for those paths.
- Hot-path identification replays recorded request states on a separate
  application instance and compares the sequence of branch decisions. Paths
  that cross the hot threshold become synthesis targets.
- Dependency extraction uses concolic execution. SQL invocations become graph
  vertices containing query template, symbolic parameters, path conditions,
  symbolic query results, and successor links. Edges capture issue order, and
  parameters/path conditions capture data and control dependencies.
- Multiple hot-path graphs are merged by matching equal query templates and
  parameter expressions, then disjoining compatible path conditions. Divergent
  suffixes remain as alternative graph branches.
- Stored procedure generation is rule based. The implementation described in
  the paper supports common SQL statement forms and uses 71 transformation
  rules for arithmetic, comparisons, type mapping, and string operations.
- If computations cannot be expressed in the target stored-procedure language,
  such as unsupported external method calls or MySQL array operations, the
  dependency graph is split into multiple subgraphs and procedures rather than
  pretending the dependency vanished.
- The generated procedures preserve transaction boundaries by carrying BEGIN
  and COMMIT statements from the original sequence instead of implicitly
  opening or closing a transaction for the procedure.
- Runtime integration happens at the database-access driver. On the first SQL
  call for an optimized API, the driver invokes the stored procedure, buffers
  result sets and write-status outputs, resumes the application code, and
  satisfies matching later SQL calls from the buffer.
- Cold-path fallback uses per-query marker variables returned by the stored
  procedure. If only a prefix of the expected statements executed, the driver
  serves that prefix from buffered results and resumes ordinary interactive
  SQL for the remaining cold path.
- Exception handling is delayed to the application statement that would have
  observed the error. Procedure-start errors fall back to the original
  application path because no procedure statement has run yet.
- Speculative execution prunes path conditions that do not distinguish
  neighboring hot branches. For procedures with writes, WeBridge excludes the
  final commit, adds savepoints, validates whether executed statements match
  the application's requested statements, and rolls back incorrect speculative
  writes before continuing.
- Evaluation uses six open-source Java applications from e-commerce,
  blogging, forum, and configuration-management domains, with separate
  client, web, and MySQL 5.7 database machines. The paper reports up to
  79.8% median latency reduction, geometric-mean median latency reduction of
  58.1%, up to 2x peak throughput, and geometric-mean peak throughput
  improvement of 1.34x. It attributes throughput gains partly to reduced SQL
  parse/optimize work and partly to shorter transactions holding contended
  locks.
- The paper also reports that speculative execution reduces Shopizer API
  latency by 10.6%-32.2%, while very low web/database RTT can expose runtime
  overhead; one Sagan API is slower at 0.1 ms RTT.

**GPU DB mapping:** WeBridge argues for a route-template layer between raw
pgwire messages and execution owners. Today a client can send many small
statements that repeatedly enter the owner, refresh route metadata, acquire a
snapshot, and return tiny results. A GPU DB stored-route analogue would
classify a known request path, bind parameters once, and execute a compact
server-side program over owner state, retained snapshots, or GPU workers.
The program could include multiple reads, conditional branches based on
previous results, and writes that publish only at explicit durable/visible
frontiers.

For 1M logical sessions, this is an admission strategy as much as a planning
strategy. Sessions that repeatedly execute known templates should hold a
route handle rather than repeatedly paying full parse/plan/owner admission
cost. The handle still needs deterministic invalidation by catalog generation,
schema, supported operators, snapshot generation, residency state, and
prepared/session state. Unknown or unsupported SQL remains on the conservative
interactive path.

The concolic dependency graph maps to GPU DB's future route descriptor:
statement sequence, parameter dependencies, branch predicates, result shapes,
transaction boundaries, visibility requirements, and required owner domains.
The same graph can decide whether a template is safe for mutation-owner
execution, immutable retained reads, CPU prefilter plus GPU tail, or ordinary
CPU fallback. Importantly, WeBridge keeps cold fallback explicit. GPU DB
should do the same: a synthesized route may serve a correct prefix, but once
it sees an unmatched branch, stale generation, unsupported operator, or
memory-pressure rejection, it must resume through the normal planner/owner
path with visible telemetry.

The speculative-write mechanism is also useful as a warning. GPU DB should
not speculate visible writes for throughput unless rollback and publication
fronts are named. Savepoints, validation markers, and commit exclusion map to
the journal's recurring multi-front design: `validated`, `durable`,
`pending-visible`, `resident-built`, and `SQL-visible` need separate state.
Speculative retained reads are easier, but speculative writes require owner
fences and WAL-before-visibility discipline.

**Risks and mismatches:** WeBridge optimizes web-application request latency,
not DBMS-internal execution. Its correctness relies on REST-style request
handlers without application-side shared mutable state, deterministic SQL,
and good modeling of external method results. GPU DB cannot assume arbitrary
clients are REST-like or that a driver sees full application paths. A native
GPU DB version would need to synthesize from observed SQL transaction traces,
prepared-statement usage, or explicit stored-route definitions.

The implementation targets Java applications, JDBC interception, ORM usage,
and MySQL stored procedures. It does not solve pgwire compatibility,
PostgreSQL session state, portals, cursors, temporary objects, or arbitrary
SQL procedure language differences. The paper's evaluation is strong for
round-trip-heavy web APIs, but when database CPU is already saturated by
compute-intensive queries, throughput converges with the original
application. For GPU DB, synthesized routes help most when communication,
planning, lock-hold duration, or owner re-entry dominate; they will not fix
an overloaded GPU kernel or a saturated storage tier by themselves.

**Benchmark candidates:**

- Add trace-only route-template mining for pgwire sessions: group statement
  sequences by normalized SQL, parameter dependency, transaction boundary,
  branch/fallback outcome, result shape, catalog generation, and route
  family. Gate: no behavior change and a report naming the top hot templates
  by round trips, owner entries, lock-hold time, and retained-read hits.
- Prototype one server-side stored-route for a repeated read-only path:
  bind parameters once, execute two or more dependent reads against a single
  compatible snapshot, and return buffered per-statement results to the
  client-visible protocol layer. Failure condition: result ordering,
  error timing, or transaction semantics differ from the interactive path.
- Compare interactive pgwire versus stored-route execution for a synthetic
  dependent lookup chain under 0.1 ms, 1 ms, and 5 ms client/server RTT.
  Required metrics: p50/p99 latency, owner entries per request, queue wait,
  parse/plan time, response bytes, and fallback count.
- For a write-containing template, implement only a dry-run validator first:
  identify transaction boundaries, required WAL front, branch predicates, and
  rollback points without executing speculative writes. Gate: the validator
  rejects every template whose visible effects cannot be fenced.
- Add cold-path fallback markers for synthesized retained routes: each
  substep reports executed/not-executed, route generation, fallback reason,
  and result hash. Minimum proof: a route can serve a valid prefix and resume
  through the conservative planner without duplicated writes or missing reads.
- Measure whether stored-route execution shortens contended owner critical
  sections. Use a hot-key update/read mix and compare lock/owner hold time,
  abort or retry count, p99 latency, and throughput against ordinary
  statement-by-statement execution.
- Use the mined route templates as training data for the CARPO/PARQO-style
  route-ranker track: rank full request paths, not just single SQL
  statements, while hard eligibility rules preserve snapshot, catalog,
  residency, and WAL invariants.

### 2026-06-03 - gCCTB GPU OLTP concurrency-control study

**Citation:** Zihan Sun, Yuyu Luo, Yong Zhang, Chao Li, and Chunxiao Xing.
"GPU-Accelerated OLTP: An In-Depth Analysis of Concurrency Control
Schemes." arXiv:2406.10158v2, 2026 version, first submitted 2024-06-14.
Retrieved 2026-06-03 from `https://arxiv.org/pdf/2406.10158`.

**Category:** transaction processing / write path, with GPU execution and
concurrency-control benchmarking relevance.

**Relevance tags:** GPU OLTP; concurrency control; OCC; TicToc; Silo; MVCC;
2PL; conflict-graph ordering; GaccO; GPUTx; batch execution; warp density;
block size; latch-free atomics; conflict-resolution overhead; YCSB; TPC-C.

**Core idea:** This paper builds gCCTB, a GPU concurrency-control testbed, and
uses it to compare eight schemes: two 2PL variants, timestamp ordering, MVCC,
Silo, TicToc, GPUTx, and GaccO. The central result is not that "GPU-native"
transaction protocols always win. CPU-oriented optimistic schemes can beat
GPU-specific conflict-graph schemes in read-heavy or medium-contention cases,
because graph preprocessing costs can dominate when there are not enough
conflicts to amortize it. Under high write intensity and high contention,
GaccO's deterministic GPU-oriented conflict handling becomes much stronger.

The strongest transferable lesson is that GPU DB should select write-batch
protocols by measured conflict shape, not by a single favorite isolation
algorithm. Low-contention batches should keep the optimistic validation path
short and avoid expensive preprocessing. High-contention write-heavy batches
may justify deterministic conflict ordering, lock-table preprocessing, or
commutative update grouping, but only when the conflict density is high enough
to pay for the setup.

**Concrete mechanisms:**

- gCCTB uses a batch execution model in which the CPU constructs transaction
  batches, initializes runtime metadata, launches GPU kernels, and lets one GPU
  worker thread execute one transaction. Aborted transactions restart on the
  same thread until they commit.
- Transaction templates call a common CC interface: start, read/write access,
  finalization, and end hooks. Device code is generated and compiled at runtime
  with NVRTC so table format, benchmark, index choice, and CC scheme can be
  changed by configuration.
- The testbed keeps table and index data resident in GPU memory before the
  experiment and leaves updates/results on device. This isolates GPU-side CC
  behavior from PCIe transfer cost.
- The GPU table format in the paper is a row-store array with fixed table size
  during execution. The implemented GPU index is primarily a sorted array with
  binary search; a B+ tree variant is tested as an index-cost comparison.
- Correctness checking uses a lightweight GPU event log for reads, writes, and
  commits. The CPU verifier scans the event order, builds a conflict graph, and
  reports cycles.
- The CPU-oriented schemes pack common control metadata into 64-bit words where
  possible and update them with CUDA atomic compare-and-swap loops. The paper
  explicitly calls out memory fences and volatile reads as necessary for
  ordering and fresh data on the GPU.
- MVCC in gCCTB is intentionally simple: latest-version pointers plus
  preallocated history-version arrays partitioned by worker thread. Writes
  stage old versions locally and delay version-pointer updates until commit.
  Sophisticated OCC+MVCC hybrids such as Hekaton-like designs are left to
  future work.
- Silo and TicToc implementations lock write sets in primary-key order during
  validation and use no-wait behavior when a write-phase lock is unavailable.
  TicToc keeps read/write timestamp structure and consistently performs best
  among CPU-oriented schemes with writes in the paper's tests.
- GPUTx and GaccO preprocess transaction access tables on the GPU using sorted
  `(transaction id, primary key)` pairs and prefix sums. GPUTx assigns ranks
  from the conflict graph; GaccO builds a lock table and makes transactions
  wait for predecessor owners.
- The evaluation varies write ratio, Zipf contention, warp density, and block
  size across YCSB, plus warehouse count for TPC-C Payment and NewOrder.
  Reported findings include CPU-oriented OCC winning in lower-conflict cases,
  GaccO winning in high-write/high-conflict cases, and warp/block parameters
  changing throughput by large factors.
- Warp density is the number of active worker threads per warp. The paper
  reports that lower warp density helps high-contention workloads by reducing
  intra-warp conflicts and aborts, while low-contention read-only workloads can
  peak at a higher but not maximum density.
- Execution-time breakdowns show conflict-resolving time, defined as waiting
  plus abort/retry cost, largely explains CPU-oriented scheme performance under
  contention. In read-only and medium-contention cases, index lookup is also a
  large share of work.
- Latch-free CAS-loop implementations materially help, especially for OCC under
  high contention, but the paper also notes that under extreme contention the
  extra memory traffic in latch-free loops can narrow or reverse the advantage
  for some non-OCC schemes.
- The experimental setup uses CUDA 12.4, an RTX 4090 with 24 GB memory, YCSB
  batches of `2^20` transactions, fixed-size resident tables, no inserts or
  deletes, and predetermined read/write sets.

**GPU DB mapping:** The paper supports a route-classed mutation scheduler for
GPU DB. Current P8 work is read-heavy, but any future GPU-side write execution
should start with a conflict classifier: read-mostly/low-contention optimistic
batches, high-contention hot-key batches, commutative update batches, and
unsupported dynamic batches. Each class can choose a protocol and launch shape
rather than forcing all writes through one GPU CC implementation.

For low-contention and read-heavy paths, the paper argues against expensive
conflict-graph preprocessing. GPU DB should keep retained reads and small
write batches on short optimistic paths with clear validation and fallback.
TicToc-like timestamp metadata and Silo-like compact ownership can remain CPU
or owner-side until measurement shows a GPU batch is large and homogeneous
enough to justify device execution.

For high-contention hot rows, the GaccO result suggests a benchmarkable
alternative: build a per-batch access table for known write templates and
execute a deterministic conflict order on GPU, or at least use the access table
to route hot keys to partition owners with bounded admission. This maps to
future `order_line`, stock, warehouse, account-balance, or counter-like update
paths where the transaction template and keys are known before execution.

The launch-parameter findings are important for latency. A GPU DB scheduler
should not treat thread count, warp density, and block size as static constants
for all transaction routes. A retained read batch, a low-conflict write batch,
and a high-contention write batch can require different launch shapes. The
route descriptor should eventually carry observed abort/conflict rate,
conflict-resolution time, index time, and chosen CUDA launch parameters.

The paper's verification structure also maps well to correctness gates. Before
GPU DB accepts any GPU-side mutation protocol, it should emit a compact event
trace for a tiny deterministic proof, verify serializability or the chosen
isolation contract on CPU, and separately prove WAL-before-visibility. gCCTB
does not handle durable logging, so GPU DB needs an additional durable frontier:
the batch may be serializable on device but still must not become SQL-visible
before WAL is safe and resident invalidations are published.

**Risks and mismatches:** The paper's testbed is not a production DBMS. It
keeps all tables in GPU memory, does not measure PCIe transfer, does not
perform inserts or deletes, uses fixed-size tables, assumes predetermined
read/write sets, and leaves dynamic memory and index maintenance out of scope.
That is a large mismatch with pgwire, WAL replay, MVCC visibility, DDL,
over-resident tiers, and ordinary SQL planning.

The MVCC implementation is deliberately simple and even hits timestamp overflow
under some high-contention launch configurations. Its poor result should not be
read as a general rejection of MVCC for GPU DB; it is evidence that naive
version-chain traversal and timestamp formats are hostile to GPU execution.
Likewise, GaccO's high-contention advantage comes from known batched access
sets and conflict preprocessing, so it does not apply to arbitrary interactive
transactions whose read/write sets are discovered mid-flight.

The evaluation hardware is a single RTX 4090 and the paper does not report
network, WAL, checkpoint, recovery, snapshot publication, or host-tier costs.
The transferable claim is protocol/launch-shape behavior inside a GPU-resident
transaction batch, not end-to-end SQL throughput.

**Benchmark candidates:**

- Add a trace-only write-batch classifier for existing pgwire/COPY and future
  transaction templates: write ratio, hot-key skew, repeated key count,
  known-versus-dynamic read/write set, index family, and route family. Gate: no
  behavior change and a report naming which batches are optimistic-only,
  deterministic-preprocess candidates, or CPU-owner-only.
- Build a tiny GPU-side conflict simulator, not a production mutation path:
  feed synthetic YCSB-style read/write sets and compare optimistic validation,
  per-key deterministic ordering, and wait/abort behavior. Required output:
  abort count, conflict-resolution time, kernel time, and chosen launch shape.
- For one hot-key update template, prototype an access-table construction
  benchmark on device. Measure preprocessing cost versus saved abort/retry
  work as contention and write ratio rise. Failure condition: preprocessing
  dominates below the expected production conflict density.
- Add launch-shape telemetry to retained GPU routes: block size, active
  logical workers per warp, batch size, CUDA time, queue wait, and result
  scatter time. This can start on read routes before any GPU mutation protocol
  exists.
- Extend the MVCC stress tests with a GPU-unfriendly version-chain case: one
  hot key, many updates, one old retained snapshot, and a GPU read batch that
  must resolve visibility. Gate: identify whether the bottleneck is timestamp
  format, chain traversal, memory divergence, or CPU fallback.
- Require any future GPU mutation proof to emit an event log with read/write/
  commit records and run a CPU verifier for cycles or isolation violations.
  Separate proof gate: WAL and resident invalidation must publish before the
  commit becomes SQL-visible.
- Compare sorted-array and resident key-vector lookup costs for retained
  point-query batches under low contention. The paper's index-time breakdown
  suggests that index cost can dominate once conflict cost is low.

### 2026-06-03 - Cross-paper synthesis: GPU writes need classed conflict lanes

The recent Tigger, CARPO, WeBridge, and gCCTB entries converge on the same
shape from different angles: the engine should classify repeated work before it
hits the expensive execution boundary. Tigger reduces protocol/proxy crossing,
CARPO ranks candidate plans as a context-aware set, WeBridge collapses repeated
dependent SQL paths into stored-route programs, and gCCTB shows that GPU write
protocols should be selected by conflict shape and launch economics.

The design track that emerges is a route descriptor with two separate halves.
The first half is semantic eligibility: catalog generation, snapshot/visibility
boundary, WAL requirements, transaction boundary, read/write-set knowledge,
result shape, and fallback legality. The second half is resource/conflict
economics: owner entries saved, round trips saved, GPU resident bytes touched,
index work, predicted conflict density, abort cost, preprocessing cost, and
queue pressure. Hard eligibility rules decide what is legal; a CARPO/PARQO-like
ranker or heuristic only chooses among legal routes.

For GPU DB writes, the immediate benchmark priority is not a full GPU OLTP
engine. It is a classed write-admission lab: trace batches, identify hot-key
and known-template shapes, measure optimistic versus deterministic
preprocessing costs, and prove isolation/WAL publication with event traces.
That keeps the current P8 retained-read work intact while preparing a narrow
future lane for GPU-side mutation only where conflict density and batch shape
make it rational.

Category gaps remain around practical PostgreSQL-protocol session state,
prepared statement/portal invalidation, and multi-tier query spill policies.
The next reviews should lean toward session scheduling or resource-adaptive
execution rather than another pure GPU OLAP paper.

### 2026-06-03 - Databases on Modern Networks

**Citation:** Alberto Lerner, Carsten Binnig, Philippe
Cudre-Mauroux, Rana Hussein, Matthias Jasny, Theo Jepsen, Dan R. K.
Ports, Lasse Thostrup, and Tobias Ziegler. "Databases on Modern
Networks: A Decade of Research That Now Comes into Practice." PVLDB
16(12), 2023, pp. 3894-3897. doi:10.14778/3611540.3611579.
Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol16/p3894-lerner.pdf`.

**Category:** runtime / HFT / session scale, with DB/network co-design
and future distributed owner-domain relevance.

**Relevance tags:** modern networks; RDMA; OS bypass; zero-copy;
programmable switches; semantic routing; in-network transaction
triaging; resource-adaptive session admission; protocol offload;
cloud database architecture; disaggregated storage.

**Core idea:** This short PVLDB tutorial argues that database engines
can no longer treat the network as an opaque socket pipe. Modern cloud
networks provide OS-bypass access, zero-copy transfer, RDMA-like
primitives, and programmable NIC/switch behavior, and those changes
open database designs that were previously impractical. The paper is
not an evaluation paper with one new DBMS mechanism; it distills a
decade of RDMA and programmable-network database work into design
lessons and warnings.

The most transferable idea for GPU DB is that transport choice should
become an explicit architecture boundary. Pgwire-over-TCP remains the
compatibility front door, but high-throughput internal owner messages,
replication/WAL publication, cold-partition movement, and future
multi-node GPU/device traffic should be designed around named
communication primitives: copied socket messages, zero-copy buffers,
one-sided remote reads/writes, two-sided RPC, switch/NIC routing, and
in-network aggregation or triage where the semantics are small enough.

**Concrete mechanisms:**

- Modern server stacks avoid per-message system-call cost by letting an
  application communicate directly with the NIC through OS bypass.
  Data movement can also be zero-copy: the card reads from userspace
  buffers rather than forcing extra send-side copies.
- RDMA and cloud derivatives blur local/remote memory access, but they
  differ materially. The paper notes that cloud providers expose
  different RDMA-like stacks, such as RoCE/InfiniBand, EFA, and 1RMA,
  with differences in one-sided versus two-sided verbs and ordering
  guarantees.
- Early RDMA database work split into two tracks: using RDMA verbs to
  accelerate existing components, and redesigning architectures around
  scalable remote memory or disaggregated layouts.
- The authors' RDMA lessons are conservative: DBMS architectures must
  evolve to exploit RDMA well; correct concurrent RDMA write protocols
  are difficult because they interact with local DMA, PCIe, ordering,
  and synchronization; and database-centric abstractions are needed to
  hide low-level RDMA complexity without losing performance.
- Programmable switches/NICs can perform small semantic actions at line
  rate, but only under restrictive compute and memory models. The paper
  emphasizes that offloaded algorithms and data structures usually need
  redesign rather than recompilation.
- One example is semantic routing for replicated databases: a switch can
  track whether secondary replicas are up to date and redirect safe read
  transactions away from the primary.
- Another example is in-network transaction batching and reordering.
  The switch can group or reorder transactions with high affinity to
  reduce network overhead and improve server cache hit ratios.
- A more aggressive OLTP example is hot-region execution in the
  network: instead of forwarding a transaction against a contended hot
  area, a switch can pull that hot area and process a small operation
  locally.
- Analytical examples include doing joins, aggregations, graph-pattern
  mining, or ML parameter aggregation in the network when the operation
  is simple enough and would otherwise be dominated by reshuffle or
  all-to-all communication.
- The paper frames the open problem as new data-intensive systems that
  fully embrace modern networks, including higher-level RDMA
  communication primitives, database-motivated behavior customization,
  programmable-switch state, and high-level data services.

**GPU DB mapping:** For the current single-node engine, the immediate
lesson is not "replace pgwire with RDMA." It is to make communication
semantics first-class before the system reaches the point where
thread-per-client TCP, copied buffers, and generic channels are baked
into every owner boundary. `11-high-throughput-query-runtime.md`
already names network IO workers, command rings, response rings, GPU
execution owners, and bounded buffers. This paper suggests those
boundaries should carry a transport/resource contract too: copy count,
buffer ownership, ordering guarantee, remote/local memory access,
completion signal, and whether the path is legal for mutation,
retained read, WAL, refresh, or cold-tier movement.

For 1M logical sessions, programmable-network examples map best to
admission and routing, not arbitrary SQL execution. Tiny validated
actions might be eligible for future kernel/NIC/switch help: route a
read to a caught-up replica or owner, reject overload before entering
the engine, classify a request into a hot retained route, or batch
same-affinity requests. Anything involving full SQL semantics, MVCC
visibility, WAL-before-visibility, catalog invalidation, or complex
expression evaluation should remain inside DB-owned code unless the
offloaded state and invalidation protocol are explicitly proved.

RDMA/disaggregated-storage lessons matter for GPU memory and future
tiers. GPU DB's planned tiers are not just HBM, DRAM, and NVMe; future
CXL/remote memory or remote GPU nodes will turn "resident snapshot" and
"cold partition" movement into networked memory movement. A retained
snapshot handle should therefore avoid assuming local addressability.
It should name table/partition identity, visibility boundary, residency
generation, location, transport path, and completion/ordering contract.

The warning about concurrent RDMA writes is directly applicable to WAL
and MVCC publication. One-sided writes can look attractive for pushing
WAL records, visibility summaries, or resident metadata to another
owner/device/node, but SQL visibility still needs a DB-level commit
frontier. GPU DB should not let a remote DMA completion become the same
thing as durable, validated, visible, or invalidated. Those fronts must
remain separate telemetry and correctness states.

**Risks and mismatches:** This is a tutorial and position paper, not a
full experimental system. It does not provide new benchmark numbers for
a specific SQL workload, and many examples are summarized from prior
work. It also targets distributed/cloud database systems, while the
current GPU DB engine is still proving a single-node pgwire and retained
GPU path. RDMA and programmable switches are future transport options,
not prerequisites for the current P8 storage slice.

The offload examples are easy to overapply. Network devices have tiny
state and restricted computation compared with CPU/GPU execution, and
the paper explicitly says database logic must be redesigned for those
models. For GPU DB, the safe takeaway is semantic routing, admission,
simple aggregation, and validated request triage; it is not permission
to move MVCC, arbitrary predicates, or transaction commit into a switch.

**Benchmark candidates:**

- Add a transport-contract inventory for the runtime design: for each
  boundary, record copy count, buffer owner, ordering guarantee,
  completion event, queue depth, and legal route families. Gate: no code
  change, but every hot path has a named contract and fallback reason.
- Instrument the pgwire retained path for bytes copied per request from
  socket read through response write. Required metrics: input copies,
  output copies, buffer allocation count, queue wait, owner entries, and
  p50/p99 latency at concurrency `1,2,4,8,16,32,64`.
- Prototype a compatibility-preserving admission pre-classifier before
  owner enqueue: reject overload, route exact retained reads, or forward
  to the normal planner with an explicit reason. Failure condition:
  SQL-visible behavior or error timing changes for non-retained paths.
- Design a future RDMA/zero-copy COPY-admission experiment as a lab-only
  transport slice: preallocated input buffers, explicit ownership states,
  and WAL-before-visibility barriers. Gate: the proof must show that DMA
  completion, WAL durable completion, resident invalidation, and SQL
  visibility are distinct events.
- Add a semantic-routing simulator for replicated or partitioned read
  snapshots: route reads only to owners/replicas whose visibility
  boundary is high enough. Measure avoided primary/owner queue entries,
  stale-read rejections, and route-table update cost.
- For over-resident execution, model remote/cold partition movement as a
  transport contract rather than only a storage cost. Metrics: bytes
  moved, copy count, queue depth, IO/network completion latency,
  visibility boundary, and fallback reason.
- Add follow-up reviews for database-specific RDMA abstractions and
  programmable-network transaction triage before committing to any
  kernel-bypass or RDMA implementation path.

### 2026-06-03 - Cloud-Native Database Systems and Unikernels

**Citation:** Viktor Leis and Christian Dietrich. "Cloud-Native
Database Systems and Unikernels: Reimagining OS Abstractions for
Modern Hardware." PVLDB 17(8), 2024, pp. 2115-2122.
doi:10.14778/3659437.3659462. Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol17/p2115-leis.pdf`.

**Category:** runtime / HFT / session scale, with multi-tier cache,
virtual-memory snapshot, and DB/OS co-design relevance.

**Relevance tags:** cloud data plane; unikernel; OS bypass; kernel
integration; asynchronous NVMe; asynchronous networking; DBMS-aware
scheduling; virtual-memory snapshots; copy-on-write; TLB control;
polling versus interrupts; hypervisor resource control; POSIX
replacement.

**Core idea:** The paper argues that cloud-hosted database services
make DBMS-specific kernels practical again. Earlier custom database
operating systems failed because users did not want to install a
special OS and vendors would have had to support many hardware
drivers. In a database-as-a-service cloud data plane, the vendor owns
the runtime image and the hardware set is narrower, so a unikernel can
co-locate DBMS and kernel code in one address space while tenant
isolation remains a hypervisor responsibility.

The immediate benefit is less privilege-transition and process
isolation overhead. The deeper claim is more important for GPU DB:
modern databases already schedule CPU work, async IO, memory, and
networking themselves because old POSIX abstractions no longer match
fast NVMe, high-speed NICs, intra-query parallelism, or cloud elasticity.
A DBMS-specific unikernel can expose hardware primitives directly:
page tables, TLB invalidation, hardware timers/preemption, NVMe
submission/completion queues, network queues, interrupt routing,
memory ballooning, and CPU hot-plugging.

**Concrete mechanisms:**

- The paper contrasts three models: traditional one-client-thread
  blocking sockets/storage; modern DBMSes using worker threads plus
  io_uring/SPDK/DPDK-style bypass; and the authors' DBMS-optimized
  unikernel vision with co-designed interfaces over virtualized CPU,
  memory, NVMe, and networking.
- It argues that modern storage and networking make synchronous
  blocking calls inefficient. A single modern SSD can have roughly
  100 concurrent IOs and more than one million requests per second,
  while Linux IO path overhead is high enough that fully exploiting
  multiple NVMes or a 100 Gbit NIC can consume about half the CPU
  cores in cited studies.
- Instead of user-space task systems that are invisible to the OS, a
  unikernel can make DBMS logical jobs kernel-visible, cheap to switch,
  and preemptible. Threads can suspend, manipulate run queues, block
  preemption, or run coroutine-like while the scheduler still sees all
  runnable work.
- The scheduler direction is DBMS-aware: rather than asking the DBMS
  to hard-pick a fixed number of threads for each query, the scheduler
  could ask jobs to parallelize when CPU load, IO utilization, and
  priorities justify it.
- For virtual memory, the paper proposes using page tables for
  database functionality rather than process isolation: buffer
  management, snapshots, dynamic data structures, variable page sizes,
  and high-rate intermediate allocations.
- The authors report that checking whether a random 4 KiB page is
  present inside a 4 GiB region takes 1.8-4.8 us through Linux's
  pagemap interface on their 16-core setup, versus 40-44 ns in OSv.
  They also report faster page-fault paths in OSv for installing
  preallocated frames.
- Their evaluation implements an OSv copy-on-write snapshot primitive
  for an OLTP/OLAP microbenchmark over a 4 GiB mapping. OLTP threads
  perform random atomic updates while periodic OLAP jobs create and
  scan read-only snapshots.
- Two co-designed snapshot optimizations are central. Parallel
  snapshotting lets OLTP threads that hit copy-on-write faults help
  copy page tables for the snapshot. Reader-side TLB invalidation
  removes global TLB shootdowns from the page-fault handler for
  snapshotted regions; readers proactively invalidate before accessing
  snapshotted pages.
- In their benchmark, OSv outperforms Linux for snapshot creation,
  OLAP scanning, OLTP progress during OLAP, and snapshot destruction.
  They also show a `NoFree` variant where freed frames are collected
  rather than returned immediately, making snapshot destruction 96x
  faster than Linux when all OLTP threads assist.
- For NVMe, the paper emphasizes that the device is already
  queue-based: the host writes submission queues and observes
  completion queues, with DMA placing data in memory. A unikernel can
  expose those queues directly instead of layering another OS queue
  on top.
- For networking, the authors note that NIC standardization is weaker
  than NVMe. They suggest AWS ENA as an initial practical target and
  EFA/SRD as a possible alternative to TCP: reliable packet delivery
  without in-order guarantees, requiring DBMS communication design
  around unordered reliable packets.
- Because polling is efficient at high request rates but wasteful at
  low rates, unikernel integration can dynamically switch or route
  interrupts based on workload, unlike many pure kernel-bypass paths.
- Hypervisor integration can expose memory ballooning and CPU
  hot-plugging to DBMS policy so a cloud database data plane can trade
  resource demand against cloud pricing and elasticity.

**GPU DB mapping:** The current GPU DB runtime document already points
away from thread-per-client execution toward network IO workers,
bounded command rings, response rings, owner domains, and GPU execution
owners. This paper strengthens that direction: the long-term runtime
should treat POSIX sockets, blocking IO, and OS thread scheduling as
compatibility surfaces, not as the internal architecture. Pgwire can
remain the client protocol while the hot internal path becomes explicit
queues, owned buffers, IO completion events, and scheduler-visible
logical jobs.

The virtual-memory snapshot section maps directly to retained read
snapshots and HTAP visibility. GPU DB should not copy the paper's
`fork`-style mechanism wholesale, but the principle is valuable:
snapshot publication can be a first-class runtime primitive rather
than an accidental side effect of process VM. A future host-tier
snapshot manager could use page-table or VM-assisted indirection for
CPU canonical/columnar segments, while GPU resident snapshots keep
their own visibility boundary, layout identity, and invalidation
generation. Reader-side TLB invalidation is especially relevant as a
conceptual analogue for moving invalidation cost to explicit
buffer/snapshot fix points rather than global stop-the-world barriers.

For 1M logical sessions, the scheduler argument is stronger than the
unikernel deployment claim. GPU DB needs logical jobs that are visible
to admission and scheduling: pgwire parse/read work, mutation
admission, WAL flush work, residency refresh, retained read batches,
GPU kernel launches, D2H scatter, and response encoding. A scheduler
that only sees OS threads cannot choose well among those. Even on
Linux, the architecture should build DBMS-visible logical jobs and
queue telemetry now; a future unikernel or DB/OS co-design path would
then have the right control plane to consume.

The NVMe queue discussion maps to over-resident execution and cold
partition movement. GPU DB's tier manager should model NVMe as
parallel queue resources with submission depth, completion latency,
DMA target ownership, and CPU overhead, not as opaque file reads.
This supports benchmark designs for cold partition prefetch, retained
snapshot rebuild, checkpoint scan, and compressed segment movement
without assuming the Linux file system remains the final interface.

The networking section also fits the transport-contract idea from the
previous journal entry. If a future cloud deployment can use ENA,
EFA/SRD, RDMA-like primitives, or GPU-adjacent IO, route legality must
be expressed in DB terms: ordering requirement, request id, visibility
boundary, WAL frontier, completion event, retry semantics, and response
ordering. An unordered reliable packet protocol may be fine for
independent retained reads or partition movement, but it is dangerous
for transaction commit and response ordering unless the DBMS supplies
sequence and frontier logic.

**Risks and mismatches:** This is a vision and microbenchmark paper,
not a complete production DBMS. The evaluation focuses on virtual
memory snapshotting in OSv versus Linux, not pgwire, SQL execution,
GPU kernels, WAL durability, crash recovery, or multi-tenant security
audits. The paper's cloud argument also assumes the DBMS vendor
controls the data-plane VM image and can accept unikernel operational
constraints.

Unikernel integration is a future deployment direction, not a near-term
requirement for the current engine. GPU DB should first prove the same
architecture-neutral pieces on Linux: logical job scheduling, bounded
rings, explicit buffer ownership, snapshot publication, NVMe queue-depth
telemetry, and transport contracts. Replacing the OS too early would
risk hiding database design gaps behind a platform port.

Security and observability need caution. A single-address-space kernel
removes isolation boundaries and may lack hardening features unless
added deliberately. That may be acceptable for a single-tenant cloud
data plane, but GPU DB still needs clean fault containment, metrics,
debuggability, and crash/recovery evidence before any specialized OS
path can be treated as production-safe.

The copy-on-write snapshot results are promising but not a direct MVCC
replacement. GPU DB still needs SQL visibility, update/delete version
rules, long-snapshot garbage collection, WAL-before-visibility, and
resident invalidation. VM snapshotting can help host-tier data movement
or HTAP read isolation, but it cannot by itself decide transaction
serializability or durable commit order.

**Benchmark candidates:**

- Build a logical-job inventory for the current pgwire endpoint: classify
  accept/read, parse, planner, owner execution, WAL, residency, GPU
  launch, result materialization, response encode, and write-back as
  separately measurable jobs. Gate: no behavior change and p50/p99 plus
  queue-wait attribution for concurrency `1,2,4,8,16,32,64`.
- Replace the thread-per-client benchmark path with a small IO-worker
  pool only when the inventory shows the owner boundary is not the only
  bottleneck. Proof gate: identical SQL results, no WAL/visibility
  regression, and lower scheduler/context-switch overhead under retained
  read concurrency.
- Add a Linux-hosted "unikernel-shaped" scheduler lab: logical jobs on
  fixed-capacity rings, cooperative yields at known DB boundaries, and
  forced preemption/fairness telemetry. Failure condition: p99 latency
  worsens under mixed long scan plus point lookup workload.
- Prototype host-tier VM snapshot telemetry, not production semantics:
  measure copy-on-write or page-table snapshot costs for a CPU columnar
  segment while a retained GPU snapshot is being refreshed. Required
  metrics: create/destroy time, page faults, TLB shootdown signal if
  available, OLTP update slowdown, and snapshot staleness boundary.
- Add an NVMe queue-depth model for over-resident partition reads:
  issue parallel cold-segment reads with explicit submission depth and
  completion telemetry, then compare against ordinary file-read paths.
  Gate: no GPU benchmark required; report CPU overhead, latency
  distribution, bytes read, queue depth, and fallback reason.
- Create a transport-ordering simulator for future ENA/EFA/SRD-style
  internal messages: independent retained reads may complete out of
  order, while transactions and pgwire responses require sequence and
  commit-frontier enforcement. Failure condition: any simulated response
  can observe a visibility boundary newer than its WAL frontier.
- Evaluate polling-versus-sleep behavior in the IO-worker prototype:
  dynamic switch threshold, CPU burn at low request rate, tail latency
  at burst start, and fairness with GPU execution owners.
- Add a follow-up review of DBOS and the 2024 "Why Files If You Have a
  DBMS?" paper to separate cloud control-plane ideas from storage
  interface ideas before committing to any DB-owned OS path.

### 2026-06-03 - Skyloft user-space preemptive scheduling

**Citation:** Yuekai Jia, Kaifu Tian, Yuyang You, Yu Chen, and Kang
Chen. "Skyloft: A General High-Efficient Scheduling Framework in User
Space." SOSP 2024, pp. 83-99. doi:10.1145/3694715.3695973.
Retrieved 2026-06-03 from
`https://madsys.cs.tsinghua.edu.cn/publication/skyloft-a-general-high-efficient-scheduling-framework-in-user-space/SOSP24-Jia.pdf`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** user-space scheduler; microsecond preemption;
user-mode interrupts; user-space timer interrupts; DPDK; work stealing;
latency-critical and best-effort co-location; heavy-tailed requests;
core reallocation; isolated cores; kernel-bypass networking.

**Core idea:** Skyloft is a user-space scheduling framework that uses
Intel user interrupts and delegated timer interrupts to make
microsecond-scale preemption available without routing every scheduling
decision through the Linux scheduler. It keeps Linux compatibility, but
on isolated cores it runs one active kernel thread per core and schedules
user-level threads itself. That lets it implement policies ranging from
centralized Shinjuku-style scheduling to per-CPU CFS/RR/EEVDF-like
schedulers and work stealing.

The paper's strongest transferable claim is not that GPU DB should adopt
Skyloft wholesale. It is that heavy-tailed service mixes need a runtime
escape hatch from cooperative or run-to-completion scheduling. A database
runtime with microsecond retained reads, longer scans, WAL flushes,
residency refreshes, and cold-tier movement should not allow one long
request class to monopolize an IO worker, owner worker, or execution
lane simply because the scheduler only sees OS threads.

**Concrete mechanisms:**

- Skyloft uses Intel UINTR so one user-space thread or a timer can
  deliver an interrupt directly to a user-space handler. The paper
  reports roughly 0.6 us from sending a user interrupt on one core to
  handling it on another, and roughly 0.3 us to handle a user timer
  interrupt.
- The system supports both dispatcher-driven preemption using user IPIs
  and per-CPU preemption using local APIC timer interrupts delegated to
  user space. The timer path avoids a dedicated timer/dispatcher core for
  per-CPU policies.
- A global interrupt handler updates policy state on each timer tick and
  can enqueue the current task, then enter the scheduler loop if
  preemption is enabled and the policy requests rescheduling.
- For multiple applications, Skyloft creates one kernel thread per
  isolated core per application, but enforces a single-binding rule: at
  most one active kernel thread is bound to an isolated core at a time.
  Within one application it switches user threads directly; between
  applications it uses a small kernel module to atomically suspend one
  kernel thread and wake another.
- Scheduler policies are expressed through small operations such as
  `task_enqueue`, `task_dequeue`, `task_block`, `task_wakeup`,
  `sched_timer_tick`, `sched_balance`, and `sched_poll`, rather than
  tying preemption to one fixed policy.
- Skyloft integrates a DPDK-based network path. Packets are polled on a
  dedicated core, distributed to isolated cores through a shared ring by
  RSS hash, parsed by a lightweight TCP/UDP stack, and blocking request
  threads can be suspended while other user threads run.
- In schbench, Skyloft's per-CPU schedulers achieved much lower wakeup
  latency than Linux CFS/RR/EEVDF under the tested configuration, mainly
  because Linux timer frequency constrained wakeup latency.
- For a synthetic workload with 99.5% 4 us short requests and 0.5% 10 ms
  long requests, Skyloft with a Shinjuku-like policy found a 30 us
  preemption quantum to be the best tradeoff in its setup; shorter
  quanta reduced tail latency but raised interrupt overhead.
- Compared with ghOSt for the synthetic latency-critical workload,
  Skyloft reported higher maximum throughput and lower tail latency,
  attributing the difference to avoiding kernel-thread context switches
  and user-agent/kernel communication on preemption.
- On a RocksDB server with 50% GET and 50% SCAN requests, Skyloft's
  preemptive work-stealing policy sustained 1.9x more load than
  Shenango at a 5 us quantum for a target 99.9% slowdown of 50x. The
  paper also shows Memcached performance within about 2% of Shenango for
  a light-tailed workload.
- The implementation is not pure userspace: it uses a Linux kernel module
  for atomic kernel-thread state transitions and privileged timer setup,
  and it modifies the UINTR kernel patch to support user-space timer
  interrupts.

**GPU DB mapping:** GPU DB's current runtime target already separates
network IO workers, owner domains, read snapshot workers, GPU execution
workers, and response rings. Skyloft sharpens the scheduling question
inside those domains. The first production IO-worker pool should not be
only "fewer threads than sessions"; it should classify work into short
retained reads, protocol parse/encode work, mutation admission, WAL wait,
residency refresh, long scans, and cold-tier movement, then record whether
each class is run-to-completion, cooperatively yielding, preemptible, or
isolated on a separate lane.

For the 1M logical-session target, the single-binding and shared-runqueue
ideas map to bounded worker ownership rather than per-session threads.
Logical sessions can be represented as request state machines and
response continuations, while a small number of active workers run the
currently admitted jobs. A Skyloft-like runtime is future work, but the
near-term benchmark harness can still measure the same issue: how much
tail latency is caused by long jobs occupying scarce workers versus owner
queue wait, GPU launch wait, or socket write-back.

Preemption is especially relevant for mixed retained reads and scans.
The retained path may eventually have requests that take microseconds
and over-resident or aggregate routes that take hundreds of microseconds
or milliseconds. Skyloft suggests treating those as different scheduling
classes with explicit preemption or lane isolation. It does not justify
interrupting arbitrary CUDA kernels or violating snapshot/WAL ordering;
it does justify making CPU-side parse, plan, owner, refresh, and response
work yieldable at known DB boundaries.

The DPDK integration and shared ring path reinforce prior transport
entries: GPU DB should keep pgwire compatibility at the edge while
measuring internal message movement as explicit rings with request ids,
queue wait, backpressure, and buffer ownership. Future kernel-bypass
networking may help, but the benchmarkable insight now is to separate
network polling/parse, admitted DB work, and response writes so one slow
class cannot mask as "client latency" forever.

**Risks and mismatches:** Skyloft depends on Intel Sapphire Rapids user
interrupt support, a modified Linux UINTR patch for timer interrupts, a
kernel module, isolated cores, and a DPDK-style networking stack. Those
are not assumptions the current GPU DB engine should take into its
minimum product path.

The paper evaluates scheduler and key-value workloads, not SQL engines,
MVCC, WAL durability, PostgreSQL protocol semantics, GPU kernel
execution, or CUDA stream scheduling. GPU kernels are not preempted by
Skyloft's CPU user interrupts, and database correctness boundaries still
need explicit yield points rather than arbitrary interruption.

There are safety and isolation concerns for multi-application scheduling:
shared runqueue metadata can be tampered with unless protected, and the
paper discusses MPK or related mechanisms as possible mitigations. For
GPU DB, this means the near-term use is single-process internal request
class scheduling, not cross-tenant userspace scheduling.

The ideal preemption quantum is workload-dependent. Skyloft's 5 us and
30 us cases are evidence that microsecond preemption can matter, not a
constant to copy into GPU DB. The engine needs measurements for pgwire
parse/encode, owner queue wait, retained lookup execution, cold refresh,
and mixed scan behavior before choosing thresholds.

**Benchmark candidates:**

- Add a request-class scheduler trace to the pgwire retained benchmark:
  classify parse, read route, mutation route, WAL wait, refresh, GPU
  launch, D2H/result scatter, response encode, and socket write. Gate:
  no behavior change; report queue wait and service time by class at
  concurrency `1,2,4,8,16,32,64`.
- Build a CPU-only scheduling simulator for mixed retained reads and long
  scans: compare run-to-completion, cooperative yield every N rows,
  separate short/long lanes, work stealing, and processor-sharing-like
  time slicing. Failure condition: p99 retained lookup latency worsens
  while throughput gain is below 10%.
- Add a "slow route injection" benchmark to the current endpoint:
  deliberately mix exact retained reads with an artificial long CPU-side
  refresh or scan job, then measure whether owner/IO worker occupancy or
  response write-back is the tail-latency source.
- Evaluate IO-worker pool sizing under persistent clients: fixed worker
  pool, bounded ingress rings, and response rings versus current
  thread-per-client behavior. Required metrics: RSS/socket distribution,
  queue wait, context switches if available, allocations, and p50/p99.
- Prototype cooperative yield points before any hardware preemption:
  parse loop, batch-drain loop, CPU fallback scan, refresh build, and
  response materialization. Proof gate: SQL-visible result order and
  WAL-before-visibility remain unchanged.
- Add a scheduling-class admission rule: short retained reads can bypass
  or use a separate lane from long refresh/scan work only when snapshot
  compatibility is proven. Failure condition: any read executes against
  an invalid or newer-than-allowed visibility boundary.
- Track a future-only hardware-preemption note: UINTR/Skyloft-style
  scheduling should be revisited only after the Linux IO-worker and
  cooperative scheduling evidence shows CPU worker occupancy is the
  bottleneck rather than owner serialization or GPU execution.

### 2026-06-03 - Cross-paper synthesis: runtime lanes need measurable preemption points

The Databases on Modern Networks, Cloud-Native Database Systems and
Unikernels, and Skyloft entries converge on one runtime principle:
fast hardware does not remove scheduling decisions; it makes hidden
scheduling decisions more expensive. RDMA, user-bypass proxies,
unikernels, NVMe queues, and user-mode interrupts are all ways to avoid
generic kernel paths, but each only helps if the database first names the
work class, ownership boundary, ordering requirement, and completion
frontier.

For GPU DB, the converging design track is a lane-based runtime contract.
Each lane should describe legal request classes, owner state touched,
snapshot or WAL frontier required, buffer ownership, queue capacity,
yield/preemption points, fallback policy, and telemetry. A short retained
read lane can be aggressively protected from long scans or refresh work;
a mutation lane must preserve WAL-before-visibility; a cold-tier lane can
optimize queue depth and transfer overlap; and a response lane must
preserve pgwire-visible ordering even if internal work completes out of
order.

This suggests a near-term benchmark priority before adopting exotic
kernel-bypass or hardware-preemption mechanisms: build Linux-hosted
evidence for classed worker occupancy. The next runtime proof should
show whether retained-read p99 is dominated by owner queue wait, IO
worker occupancy, response encoding, long CPU-side work, or GPU execution
queueing. If long jobs occupy scarce workers, add cooperative yield
points or separate lanes first. If owner serialization dominates, lane
preemption will not fix the bottleneck.

Category gaps remain in practical storage-interface design and
multi-tier spill/admission under memory pressure. The next high-value
queue choices are `Why Files If You Have a DBMS?`, `Towards Buffer
Management with Tiered Main Memory`, or a transaction/write-path paper
if the journal starts leaning too heavily toward runtime papers.

### 2026-06-03 - DBMS-owned large objects instead of files

**Citation:** Lam-Duy Nguyen and Viktor Leis. "Why Files If You Have a
DBMS?" ICDE 2024, pp. 3878-3892. DOI:
`10.1109/ICDE60146.2024.00297`. Retrieved 2026-06-03 from the TUM
author PDF,
`https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/blob.pdf`.

**Category:** multi-tier cache / data placement and storage-interface
design.

**Relevance tags:** BLOB storage; extent sequence; asynchronous BLOB
logging; WAL indirection; virtual-memory aliasing; FUSE; object/file
interoperability; DB-owned storage; NVMe write amplification; metadata
indexing; large-result protocol pressure.

**Core idea:** The paper argues that large binary objects are often kept
outside databases mostly because current DBMS BLOB paths are inefficient
and external programs expect file APIs. Its answer is a DBMS-native BLOB
design that stores each object as a compact extent sequence described by
a single Blob State, logs the Blob State rather than duplicating the full
object in WAL, flushes object extents once at commit, and exposes
read-only DBMS-owned objects through FUSE for interoperability.

The transferable idea for GPU DB is broader than BLOBs. File-system
interfaces are convenient compatibility surfaces, but they are poor
internal contracts for a tiered database runtime. Cold partitions,
compressed column groups, checkpoints, large text values, and future
GPU-adjacent object payloads should have DB-owned descriptors with
visibility, checksum, extent, destination-buffer, and admission metadata,
instead of disappearing behind path strings and opaque file reads.

**Concrete mechanisms:**

- The design stores each object as an extent sequence: a small list of
  contiguous physical page ranges. Extent sizes follow a static tier
  table, so Blob State can record only head page ids, object size,
  extent count, optional tail extent, checksums, and prefix metadata.
- A tail extent can eliminate internal fragmentation for mostly-static
  objects, while tiered normal extents make append/growth cheaper.
- Blob State is stored with the tuple for the BLOB column. Reads first
  retrieve the tuple/Blob State and then issue one asynchronous I/O call
  for missing extents, instead of walking file-system extent trees or
  scanning many TOAST/overflow pages.
- Durability avoids writing the full object twice. The WAL contains Blob
  State, not the entire BLOB content. On commit, the system persists the
  WAL buffer containing Blob State before writing the object extents; on
  recovery, SHA-256 validates whether the extent content is present and
  intact, otherwise the committing transaction is treated as failed.
- A `prevent_evict` flag keeps not-yet-flushed extents from being
  evicted while asynchronous commit-time object writes are still in
  flight.
- Deleted extents are returned to free lists by extent tier at
  transaction commit, making reuse cheap and avoiding complex file-system
  free-space search for the paper's whole-object create/delete workloads.
- Updates can either delta-log and update in place, or clone the affected
  extent and update Blob State. The paper treats the cost choice as
  workload-dependent and notes that whole-object replacement is common.
- Blob State supports indexing without copying the full object into every
  index entry. Equality can use SHA-256; range comparison can use prefix
  bytes and then incrementally dereference extents when required.
- With vmcache/exmap, disjoint extents can be aliased into a contiguous
  virtual address range, avoiding malloc plus memcpy for large reads.
  Worker-local aliasing areas handle common object sizes; a shared
  aliasing area with range locking handles larger objects.
- FUSE integration maps relations to directories and tuples to read-only
  files. `open` starts a transaction, `flush` commits it, and `read`
  resolves the path to Blob State before copying the requested slice.
- The evaluation compares against PostgreSQL, MySQL/InnoDB, SQLite, and
  Ext4/XFS/BtrFS/F2FS. The authors disable `fsync()` for competitors and
  run on a single Samsung 980 Pro SSD, so the results isolate object
  storage and metadata overhead rather than full durable application
  behavior.
- Reported results include higher throughput than file systems and DBMSs
  for large YCSB payloads, 15.6x higher metadata-operation throughput
  than file systems for a 10-object metadata scan, at least 2.9x higher
  cold-cache Wikipedia-object throughput at benchmark start, and up to
  2.1x over a hash-table buffer pool for 10 MB in-memory reads with 16
  workers due to avoiding extra copies.
- The simulated git-clone trace shows the design faster than tested file
  systems, largely because DBMS B-tree metadata lookup replaces repeated
  `open`, `fstat`, and `close` costs.

**GPU DB mapping:** GPU DB should treat this as a storage-interface
design paper, not as a call to add user-facing BLOB features immediately.
The strongest mapping is a DB-owned extent descriptor for cold or warm
column groups. A future over-resident partition can carry table id,
partition id, column group id, visibility generation, checksum, extent
heads or segment ids, compression metadata, destination buffer id, and
publish/abort state. That descriptor is the unit the planner and tier
manager reason about; a POSIX file path is only one possible backing
implementation.

The asynchronous BLOB logging idea maps to WAL-before-visibility for
large placement artifacts. GPU DB cannot publish a retained snapshot
until the durable CPU truth and visibility frontier are safe. But it can
avoid duplicating large cold-tier or checkpoint payloads in WAL by
logging compact descriptors, checksums, and generation metadata, then
validating payload presence during recovery before making any resident or
cold descriptor route-eligible.

The extent-sequence shape also fits GPU resident and over-resident data.
HBM execution may prefer large columnar chunks, while NVMe and host-cache
movement may prefer smaller extents or compressed blocks. Blob State's
lesson is to keep the indirection shallow and DB-visible: the planner
should know how many extents a route touches, expected bytes, I/O
amplification, checksum cost, and whether the object is contiguous enough
for fast DMA or aliasing.

Virtual-memory aliasing is relevant to host-tier staging. If a cold
column group consists of disjoint host or NVMe-backed extents, GPU DB may
want a contiguous CPU-visible view for CPU fallback, compression,
checksum, or transfer preparation without copying everything into a fresh
buffer. The paper also warns that aliasing has TLB shootdown and setup
costs, so the benchmark should compare aliasing against ordinary
preallocated staging buffers by object size and concurrency.

FUSE is useful only as a compatibility lesson. If GPU DB ever exposes
DB-owned large objects or cold partitions to external tools, a file-like
read-only interface can bridge compatibility. The hot internal path
should still remain DB-owned descriptors, rings, and completion events;
FUSE should not become the mechanism by which query execution fetches
cold segments.

The paper's metadata results reinforce the runtime journal's direction:
path-based storage hides work in system calls. For a 1M logical-session
engine, metadata lookups for snapshots, partitions, object descriptors,
and route eligibility should be B-tree/hash/catalog operations inside the
DBMS, with queue wait and owner time measured, not repeated kernel path
walks.

**Risks and mismatches:** The paper studies large objects and strings,
not relational tuple MVCC, joins, GPU kernels, GPUDirect Storage,
PostgreSQL protocol serving, or full over-resident analytical execution.
Its object workloads are mostly create/read/delete/whole-object replace;
partial updates and high-conflict BLOB concurrency are explicitly
secondary.

The evaluation disables `fsync()` for competitor DBMSs and file systems,
while the proposed design uses group commit. That makes the performance
comparison useful for storage-path shape, metadata, copies, and write
amplification, but not a direct durable throughput number for GPU DB's
WAL path.

Blob State logs descriptors before extent content and uses recovery
checksums to decide whether the committing transaction failed. GPU DB
must be careful before applying that pattern to user-visible SQL
transactions: failure classification, client commit acknowledgement,
replay order, and retention of holes/free extents must be proven against
the existing WAL-before-visibility contract.

Virtual-memory aliasing depends on vmcache/exmap-style page-table
control and has TLB invalidation costs. It may be excellent for large
host objects and poor for small retained lookups. The GPU path also needs
registered/pinned memory and CUDA stream ordering, which the paper does
not evaluate.

FUSE solves interoperability but can add its own context-switch and
copying costs. It should remain a boundary API for external tools, not a
core execution path.

**Benchmark candidates:**

- Define a GPU DB cold-segment descriptor modeled after Blob State:
  table/partition/generation, column group, byte size, checksum, extent
  list, compression id, resident-validity frontier, destination buffer,
  and completion state. Proof gate: stale or checksum-failed descriptors
  cannot become route-eligible.
- Add a descriptor-logging thought experiment before code: compare full
  payload WAL logging, descriptor-plus-checksum logging, and
  checkpoint-manifest logging for cold column groups. Required result:
  exact crash/recovery state table and no weakening of
  WAL-before-visibility.
- Benchmark cold-column fetch units using shallow DB descriptors rather
  than file paths: 4 KiB, 16 KiB, 64 KiB, and compressed segment extents.
  Measure I/O amplification, descriptor lookup time, queue wait, bytes
  copied, checksum time, and route rejection reason.
- Compare host staging strategies for large cold groups: malloc+copy,
  reusable pinned staging buffer, and VM aliasing where available. Gate:
  p50/p99 and memory bandwidth by object size at concurrency
  `1,2,4,8,16,32,64`.
- Add a storage-aging microbenchmark for DB-owned extents: mixed
  allocate/delete/grow workloads at 70%, 80%, 90%, and 95% capacity,
  reporting allocation latency, free-list hit rate, fragmentation,
  write amplification, and cold-read p99.
- Treat external file compatibility as a separate benchmark track:
  compare direct DB descriptor reads, FUSE read-only exposure, and normal
  file-system reads for large objects. Failure condition: compatibility
  path pollutes hot query-worker or owner-lane telemetry.
- Add protocol pressure measurements for large results or BLOB-like
  payloads: pgwire row streaming, binary copy/output, and a future
  descriptor-based escape hatch. Required metrics: serialization CPU,
  copies, socket write time, backpressure, and result ordering safety.

### 2026-06-03 - Read-priority flash storage for OLTP stalls

**Citation:** Mijin An, Soojun Im, Dawoon Jung, and Sang-Won Lee.
"Your Read is Our Priority in Flash Storage." PVLDB 15(9):
1911-1923, 2022. DOI: `10.14778/3538598.3538612`. Retrieved
2026-06-03 from the VLDB PDF,
`https://www.vldb.org/pvldb/vol15/p1911-lee.pdf`.

**Category:** multi-tier cache / data placement and storage-engine
runtime, with transaction processing / write path.

**Relevance tags:** read/write interference; dirty-victim stalls;
flash-storage asymmetry; buffer replacement; fused read-write I/O;
storage command interface; in-device read buffer; WAL-safe page
recovery; multi-tenant I/O isolation; NVMe queue utilization; cold-tier
read priority.

**Core idea:** The paper identifies a storage-specific bottleneck in
OLTP systems: on a buffer-pool page miss, the conventional
read-after-write protocol first flushes a dirty victim page and only
then reads the requested page into the freed frame. On SSDs, where
reads are much faster than writes and internal parallelism is high,
that strict ordering turns unrelated slow writes into foreground read
latency. The same pattern can also happen inside the SSD data buffer
when read and write requests share buffer frames.

Its solution has two layers. RW is a fused read/write command that lets
the DBMS submit the dirty-victim write and missing-page read together,
using the SSD buffer to copy the dirty page away before returning the
new page to the host frame. R-Buf separates the SSD's internal read
buffer from its write buffer, so host reads can obtain clean read-buffer
frames without waiting for dirty write-buffer victims. Together, they
turn a serialized host/storage read-after-write path into a read-priority
parallel path.

**Concrete mechanisms:**

- The authors define a read stall as the foreground read wait caused
  only by resource conflict with a dirty victim frame. They report that
  MySQL on SSDs sees more than one fourth of page-missed reads stall
  with a 10% buffer size; Oracle and PostgreSQL also show stalls in
  the same experiment family.
- RW is implemented as an NVMe vendor-specific command with logical
  block addresses for the read page and dirty write page, a shared
  length, and a host buffer pointer that is both the source of the dirty
  page and destination for the requested page.
- The SSD handles RW by allocating a storage buffer for the dirty page,
  DMA-copying the host frame into that buffer, allocating a read buffer
  for the requested page, reading it from NAND, DMA-copying it back to
  the host frame, and only later completing the dirty page write to
  NAND asynchronously.
- Consistency relies on the SSD command queue ordering for successive
  commands to the same page. Durability is treated like normal
  no-force DBMS page writes: the dirty-page write may fail, but redo
  logging recovers the page.
- The MySQL/InnoDB prototype modifies the buffer manager and file I/O
  modules to calculate read/write LBAs and issue RW through `ioctl`.
  The authors report the change as small, with the RW path also
  removing free-list and extra I/O-call work from the dirty-victim
  miss path.
- R-Buf splits the SSD DRAM buffer into read and write buffers with
  hash lookup over both. Reads search the write buffer first for the
  newest page, then the read buffer; a miss allocates only from the
  clean read buffer. Writes use the write buffer.
- If a write hits a page currently in the read buffer, the firmware
  temporarily flags that read-buffer frame as a write frame, prioritizes
  flushing it, then returns the clean frame to the read buffer. If a
  page exists in both buffers, the write-buffer copy wins.
- R-Buf makes controller-level read priority useful earlier: reads can
  reach per-channel queues without first waiting on dirty storage-buffer
  eviction, allowing reads to preempt queued writes or garbage
  collection where the controller supports that.
- The prototype uses a Cosmos+ OpenSSD board with 32 GB MLC NAND and
  32 MB SSD DRAM. The empirical split chosen for TPC-C is 2 MB read
  buffer and 30 MB write buffer because random OLTP reads rarely hit
  in the tiny SSD read buffer; the gain is mostly from avoiding dirty
  read-buffer victims.
- Evaluation uses Linux 5.4, ext4 with `O_DIRECT`, MySQL TPC-C,
  YCSB, SysBench, LinkBench, RocksDB `db_bench`, and PostgreSQL
  TPC-H multi-tenancy. Networking is avoided by running clients on
  the same host.
- Reported results include RW alone improving TPC-C throughput over
  RAW by up to 3.2x, RW plus R-Buf improving it by up to 3.9x, 41%
  fewer interrupts, 51% fewer context switches, and 31% fewer CPU
  instructions per transaction versus RAW plus a shared SSD buffer in
  one TPC-C configuration.
- R-Buf alone shows 5x higher random-read IOPS and 93% lower tail read
  latency than the shared-buffer OpenSSD under concurrent read/write
  FIO. In TPC-C, R-Buf improves throughput over S-Buf by 21-97%
  depending on DBMS buffer size, while write latency can roughly double.
- In a 57-hour TPC-C run, RW plus R-Buf keeps transaction latency and
  throughput stable as write amplification rises, while the RAW/S-Buf
  baseline's 99th latency grows and throughput falls. The authors
  report R-Buf's in-storage read latency staying around native read
  latency while S-Buf rises sharply as dirty victims and WAF increase.
- Multi-tenant experiments show R-Buf helping a read-only TPC-H tenant
  co-running with write-heavy TPC-C, and improving RocksDB
  `readwhilewriting` throughput and read latency when compaction writes
  compete with foreground reads.

**GPU DB mapping:** GPU DB should take this paper as a warning against
sharing one opaque cold-tier path for all durable and over-resident I/O.
If WAL flushes, checkpoint writes, cold-partition reads, refresh reads,
and large result spills all compete for the same buffer pool, queue, or
NVMe lane, a foreground retained read or cold-page route can inherit the
latency of an unrelated write. The runtime and tier manager need
read-priority storage classes before they need a special device command.

The strongest transferable idea is fused resource release, not the
exact NVMe vendor command. When a GPU DB cold-segment miss must evict or
demote a dirty host/NVMe buffer, the system should avoid serializing
"finish demotion, then start read" if the demotion payload can be copied
into an owned write buffer and the read can proceed against a separate
read buffer. In software, that means reusable staging buffers,
descriptor-level ownership transfer, and separate read/write queues can
approximate RW/R-Buf even before firmware support exists.

For WAL and MVCC, the paper's no-force assumption maps cleanly only to
data-page or cold-segment writes that are recoverable from WAL. It must
not be copied onto commit records. GPU DB can let dirty cold segments
write back asynchronously after their descriptor is no longer needed by a
foreground read, but commit acknowledgement and visibility publication
still require the WAL frontier to be durable before a newer generation
is exposed.

For over-resident GPU execution, this suggests separate queue and buffer
budgets for latency-critical reads and bulk writes. A GPU worker waiting
for a cold compressed column group should not queue behind checkpoint
or demotion writes if the data is needed to complete a user query. A
writeback or compaction worker can tolerate longer service time; a
retained lookup, cold-miss fetch, or response materialization lane often
cannot.

The multi-tenant result maps to 1M logical sessions as class isolation.
Even if every session is individually light, many write-heavy or
refresh-heavy sessions can create storage-buffer pressure that harms
read-heavy sessions unless admission reports and reserves read buffers,
write buffers, queue depth, and DMA/staging slots separately.

The paper also reinforces the explicit-service-ownership synthesis from
NVMe and kernel-bypass papers. Hidden shared queues make read latency
depend on invisible write amplification. GPU DB should expose queue wait,
dirty-victim wait, staging-buffer wait, I/O service time, checksum time,
DMA/transfer time, and response writeback time by lane before deciding
whether NVMe firmware, io_uring, SPDK, GPUDirect Storage, or ordinary
preallocated buffers are the next mechanism.

**Risks and mismatches:** The prototype requires firmware changes on
Cosmos+ OpenSSD and an NVMe vendor-specific command, so the exact RW
mechanism is not portable to commodity NVMe devices. GPU DB should first
benchmark software equivalents with separate queues and staging buffers.

R-Buf intentionally sacrifices some write performance to protect reads.
That trade may be right for OLTP reads and cold query misses, but it can
hurt bulk ingest, COPY, checkpoint, or compaction-heavy phases if read
priority is applied globally instead of by request class.

The workload uses page-oriented relational engines, 4 KB pages, a small
OpenSSD board, and mostly block-device behavior. It does not evaluate GPU
kernels, GPUDirect Storage, compressed column groups, WAL group commit,
MVCC visibility publication, or PostgreSQL protocol response pressure.

The durability claim is valid for no-force data pages recoverable by
redo, not for commit records or generation metadata. GPU DB must keep
WAL-before-visibility and descriptor-publish ordering separate from
asynchronous cold-segment writeback.

The experiments avoid networking by co-locating clients and DBMS, so the
reported interrupt/context-switch gains are storage-path gains only.
They are still useful, but pgwire response encoding and socket writeback
may dominate in GPU DB at high session counts.

**Benchmark candidates:**

- Add a CPU-only cold-tier simulator with separate read and write staging
  pools. Compare one shared pool versus read/write-separated pools under
  mixed WAL/checkpoint writes, cold-partition reads, and refresh reads.
  Gate: read p99 improves without violating WAL-before-visibility.
- Track dirty-victim stalls explicitly in the storage path: time waiting
  for free read staging, free write staging, demotion completion, NVMe
  queue admission, and checksum. Failure condition: route summaries only
  report total I/O time without naming the blocker.
- Prototype a software RW equivalent for cold-segment replacement:
  copy the dirty victim into a reusable writeback buffer, immediately
  reuse the read buffer for the requested segment, and finish writeback
  asynchronously when the descriptor remains recoverable from WAL or a
  checkpoint manifest.
- Add a read-priority admission rule for cold query misses: latency-class
  reads get reserved read buffers and queue depth; writeback/compaction
  lanes back off first. Required metrics: throughput, read p50/p99,
  writeback backlog, checkpoint lag, WAL lag, and route rejection reason.
- Benchmark shared NVMe pressure with three classes: WAL flush,
  checkpoint/cold-segment writeback, and cold analytical read. Compare
  FIFO, read-priority, and bounded read-reservation policies at queue
  depths `1,4,8,16,32,64`.
- Test whether read priority harms ingest: run COPY or bulk insert with a
  concurrent retained read workload and report write throughput, commit
  latency, read p99, and pending dirty bytes. Failure condition: read
  protection causes unbounded dirty backlog or missed checkpoint targets.
- For future GPUDirect Storage work, compare a single staging ring against
  separate read and write DMA rings. Proof gate: CUDA stream ordering and
  visibility generation checks remain explicit, and no stale cold segment
  can become route-eligible after failed writeback.

### 2026-06-03 - ScaleRPC reliable-connection resource sharing

**Citation:** Youmin Chen, Youyou Lu, and Jiwu Shu. "Scalable
RDMA RPC on Reliable Connection with Efficient Resource Sharing."
EuroSys 2019. DOI: `10.1145/3302424.3303968`. Retrieved
2026-06-03 from the author PDF,
`https://chenyoumin1993.github.io/papers/eurosys19-scalerpc.pdf`.

**Category:** runtime / HFT / session scale, with transaction
processing and distributed write-path relevance.

**Relevance tags:** high fan-in sessions; RDMA reliable connection;
connection grouping; bounded transport resources; message pools;
CPU cache locality; NIC cache pressure; request warmup; priority
scheduling; SmallBank; one-sided verbs; future network tiers.

**Core idea:** ScaleRPC starts from a concrete failure mode in
RDMA-backed systems: reliable-connection RDMA can be fast at low
connection counts, but throughput collapses as one server talks to
many clients because connection and work-queue state thrashes NIC
caches, and inbound message pools thrash CPU last-level cache. The
paper reports raw outbound RC write throughput dropping from roughly
20 Mops/s to 2 Mops/s as clients grow from 10 to more than 200, and
shows similar fan-in degradation in a distributed file-system metadata
server.

The proposed design preserves reliable-connection semantics and
one-sided verbs by making server-side transport resources shared and
bounded. Connection grouping limits how many clients are actively
served during one time slice, reducing NIC cache thrash. Virtualized
mapping lets many logical client groups share one physical message
pool, keeping the hot inbound write footprint small enough for CPU
cache. Request warmup and priority scheduling reduce the cost of
group switches and avoid wasting time on idle clients.

**Concrete mechanisms:**

- ScaleRPC uses one-sided RDMA writes over reliable connections for
  request and response transfer, so it keeps RC support for large
  payloads and one-sided read/write/atomic verbs instead of switching
  wholesale to unreliable datagrams.
- The server allocates registered huge-page message memory and formats
  it as zones and fixed message blocks. Clients write request payload,
  length, and a valid marker; the server polls the valid marker before
  invoking the RPC handler.
- Connection grouping divides clients into groups and serves one group
  at a time. Only the active group can post requests directly into the
  processing pool, bounding the active QP/WQE pressure on the NIC.
- A priority scheduler tracks each client's observed throughput and
  average request size, using a priority roughly proportional to
  request rate per byte. Higher-priority clients are placed in smaller
  groups with longer time slices, and group split/merge is performed
  lazily when group size leaves a configured legal range.
- Virtualized mapping maps multiple logical group pools onto one
  physical message pool. The message pool is stateless after a request
  is processed, so the next group can overwrite the same addresses
  without clearing per-client memory.
- Context switches save and restore per-group metadata such as client
  ids, offsets, and counters. The server drains suspended requests and
  piggybacks context-switch events in responses; inactive clients can
  be notified with extra writes.
- A warmup pool lets the next group publish local request addresses and
  batch sizes before it becomes active. The server RDMA-reads those
  prepared requests into the warmup pool, then swaps warmup and
  processing pools at the context switch.
- Clients move through warmup, process, and idle states. In process
  state they can write directly to the processing pool; on a
  context-switch event they become idle and begin warmup again.
- Evaluation compares RawWrite, HERD, FaSST, and ScaleRPC on 56 Gbps
  InfiniBand with ConnectX-3 HCAs. With 120 clients, ScaleRPC has much
  lower median latency than RawWrite/FaSST/HERD for batch size 1, but
  a bimodal tail because grouped clients wait for their time slice.
- Hardware-counter analysis attributes the outbound improvement to
  reduced PCIe reads from NIC cache misses, and the inbound improvement
  to lower CPU cache write-allocate pressure from the smaller physical
  message-pool footprint.
- The paper reports ScaleRPC improving Octopus read-oriented metadata
  operations by about 50-90% on average, while write-oriented metadata
  gains are smaller because file-system software work dominates.
- The ScaleTX prototype combines ScaleRPC for execution/logging RPCs
  with one-sided RDMA reads for validation and one-sided writes for
  commit updates. It uses optimistic concurrency control, two-phase
  commit, and an NTP-like synchronization protocol so multiple
  participants switch client groups at the same pace.
- In the SmallBank experiment, ScaleTX outperforms RawWrite, HERD,
  FaSST, and a ScaleRPC-only variant, with the paper reporting up to
  160% improvement over RawWrite at 160 clients.

**GPU DB mapping:** The most useful lesson is not "use RDMA now"; it
is that 1M logical sessions require explicit sharing and scheduling of
scarce transport resources. GPU DB already targets network IO workers,
bounded command rings, response rings, pinned staging buffers, CUDA
streams, and owner lanes. ScaleRPC shows the same pattern one layer
lower: large logical connection counts should not imply proportional
hot NIC state, message buffers, cache footprint, pinned memory, or
active queue slots.

Connection grouping maps to admission classes for pgwire sessions.
Idle or low-rate logical sessions can remain connected, but only a
bounded active set should hold decoded frontend buffers, response
slots, mutation-owner queue entries, retained-read queue slots, or GPU
staging capacity. A GPU DB equivalent of ScaleRPC's time slice would be
a microsecond-limited drain window per session class or route class,
not a global fairness guarantee that lets cold clients thrash hot
state.

Virtualized mapping maps directly to reusable protocol and execution
buffers. The runtime should allocate a bounded number of physical
request/response/staging slots per IO worker or route class, then map
many logical sessions onto those slots only while work is active. This
is especially relevant for COPY chunks, retained lookup requests,
result scattering, and pinned host buffers, where allocating by
session count would defeat the 1M-session goal.

The warmup-pool idea is a useful analogy for GPU micro-batching. The
next compatible batch can publish descriptors while the current batch
runs, but descriptors must not become visible to a GPU execution owner
until their snapshot generation, route shape, staging buffers, and
response slots are all admitted. That could hide batch handoff latency
without letting arbitrary sessions write into execution-owned memory.

The ScaleTX portion is relevant to write-path design because it mixes
message RPCs and one-sided operations by phase. GPU DB should make the
same kind of phase distinction: SQL protocol parsing and transaction
admission remain message-driven; validation, visibility-summary reads,
resident descriptor fetches, and future distributed owner handoffs may
become direct reads or writes only where ownership, durability, and
replay semantics are explicit.

**Risks and mismatches:** ScaleRPC assumes cooperative RDMA clients,
registered memory, and reliable-connection NIC behavior. The current
GPU DB endpoint is TCP/pgwire, not RDMA, and PostgreSQL clients cannot
be trusted to write directly into server memory. The transferable
mechanism is bounded active resource mapping, not the wire protocol.

Connection grouping deliberately trades throughput stability for a
bimodal latency distribution. That may be unacceptable for short
interactive queries unless grouping is applied only to overload,
background, COPY, refresh, or low-priority classes. A retained lookup
or commit acknowledgement may need a tighter service guarantee than
the paper's default 100 microsecond time slice.

The paper's transaction system is distributed key-value OLTP, not SQL
with MVCC visibility, WAL-before-visibility, catalog invalidation,
resident GPU snapshots, or PostgreSQL protocol state. Its one-sided
commit writes should not be copied into GPU DB unless recovery order,
lock release, commit acknowledgement, and stale-resident invalidation
are proven.

The hardware is older 56 Gbps InfiniBand and ConnectX-3. Modern NICs,
TCP stacks, io_uring, kernel bypass, and cloud fabrics change absolute
numbers, but the qualitative warning remains: hidden per-connection
state can become the bottleneck before application logic does.

**Benchmark candidates:**

- Add a session-resource budget simulator for the target pgwire runtime:
  compare per-session buffer allocation against virtualized active-slot
  pools at logical session counts `1k,10k,100k,1M`. Gate: idle sessions
  do not scale pinned memory, decoded message buffers, or response slots
  linearly.
- Prototype an active-session grouping policy in a CPU-only harness:
  bounded active windows for retained reads, COPY chunks, and long scans.
  Measure p50/p99 latency, throughput, fairness, queue wait, and
  starvation under mixed hot/idle clients.
- Add telemetry that separates logical sessions, active admitted
  sessions, active request slots, response slots, pinned staging slots,
  and owner-queue occupancy. Failure condition: overload reports only a
  generic queue-full error without naming the exhausted resource.
- Build a reusable message-slot proof for pgwire request parsing:
  many logical connections map onto a fixed pool of decoded command
  buffers, with ownership states for network IO, owner queue, execution,
  response encoding, and free. Proof gate: no buffer reuse before all
  owners release it.
- Test warmup-style batch descriptor preparation for same-shape retained
  lookups: prepare descriptors while the current GPU batch is executing,
  then publish only descriptors matching snapshot generation and route
  shape. Required metrics: handoff latency, batch size, p99 latency, and
  rejected descriptors by reason.
- Compare strict FIFO admission with priority-by-work policies similar
  to ScaleRPC's request-rate-per-byte priority. Gate: small retained
  reads and commit responses improve tail latency without starving COPY,
  refresh, or scan work.
- For future RDMA or kernel-bypass experiments, benchmark RC-style
  per-connection state, UD-style datagrams, and RPC-over-TCP/io_uring
  with the same logical-session and active-slot telemetry before
  changing the production transport.

### 2026-06-03 - Cross-paper synthesis: logical scale needs active resource budgets

DBMS-owned large objects, read-priority flash storage, and ScaleRPC
all converge on the same design track: logical namespace size and
active hot-resource ownership must be separated. A database can expose
many objects, many cold segments, and many client sessions, but the
hot path must admit only the descriptors, buffers, queue slots, and
I/O lanes that can be served without polluting latency-critical work.

The storage papers argue for DB-owned descriptors and read/write lane
separation. ScaleRPC argues for virtualizing many clients over a small
physical message pool and bounded active connection groups. Together
they suggest a GPU DB runtime where every tier boundary has two counts:
logical population and active admitted population. Logical sessions,
cold segments, resident snapshots, and pending writebacks may be large;
active pinned buffers, GPU batch descriptors, NVMe read slots, response
slots, and mutation-owner entries must be explicitly budgeted.

The main category gap remains a production SQL admission policy that
combines transaction priority, snapshot class, storage lane, and GPU
route fragility. Recent reviews have good ingredients from storage
resource ownership, scheduling, learned robust routing, and MVCC, but
the loop should still look for modern papers that evaluate these
signals together under SQL or HTAP workloads.

Benchmark priorities:

- Track logical versus active counts for sessions, snapshots,
  descriptors, buffers, and queue entries in every runtime report.
- Add mixed read/write/cold-miss workloads where read-priority lanes and
  virtualized request slots are both stressed, proving that p99 read
  latency improves without unbounded dirty backlog.
- Treat buffer ownership as a correctness surface: every reusable slot
  should have a visible owner state and a release frontier tied to WAL,
  snapshot generation, GPU completion, or socket write completion.
- Compare FIFO, classed priority, and active-window grouping policies
  before implementing transport-specific kernel bypass or RDMA support.

### 2026-06-03 - Natto distributed transaction prioritization

**Citation:** Linguan Yang, Xinan Yan, and Bernard Wong. "Natto:
Providing Distributed Transaction Prioritization for High-Contention
Workloads." SIGMOD 2022, pp. 715-729. DOI:
`10.1145/3514221.3526161`. Retrieved 2026-06-03 from the author PDF,
`https://cs.uwaterloo.ca/~bernard/natto.pdf`.

**Category:** transaction processing / write path and runtime admission.

**Relevance tags:** transaction priority; high-contention OLTP;
distributed ordering; abort avoidance; conditional prepare; early
committed-state forwarding; classed admission; hot-key tail latency;
predeclared read/write sets.

**Core idea:** Natto targets a specific but useful transaction shape:
two-round fixed-set interactive transactions, where read keys and write
keys are known up front, the first round reads, and the second round
commits writes derived from those reads. In geo-distributed systems,
different partitions may receive the same transactions in different
orders, so a high-priority transaction can be blocked or aborted by
low-priority work even when every individual partition uses local
priority queues.

Natto uses network-delay measurements to assign each transaction an
execution timestamp equal to the estimated arrival time at its furthest
participant. Earlier-arriving participants buffer the transaction until
that timestamp, giving all participant leaders a common timestamp order
without a centralized sequencer. This common order creates safe windows
for priority abort, conditional prepare, and early committed-state
forwarding. The paper's main evaluation shows lower high-priority tail
latency than Carousel, TAPIR, and 2PL+2PC variants on YCSB+T, Retwis,
and SmallBank deployments across a local WAN-emulation cluster and five
Microsoft Azure regions.

**Concrete mechanisms:**

- Clients assign transaction ids and runtime priority, then use a local
  proxy's recent client-to-leader delay measurements to estimate arrival
  time at every participant leader.
- The transaction timestamp is the future time when the read-and-prepare
  request should have arrived at all participant leaders. Servers queue
  received transactions by timestamp and transaction id, and process a
  transaction only after local time passes the timestamp and it reaches
  the queue head.
- Low-priority transactions use Carousel's OCC read-and-prepare path.
  High-priority transactions use a locking-based prepare path over the
  known read and write keys, avoiding repeated abort/retry when waiting
  is cheaper than another WAN transaction attempt.
- If a high-priority transaction arrives late and conflicts with an
  already queued or prepared smaller-timestamp transaction, Natto aborts
  the late high-priority transaction to preserve deadlock freedom.
- Priority abort lets a participant abort a queued low-priority
  transaction before prepare if it would block a conflicting
  high-priority transaction. The paper notes starvation risk and suggests
  retry-based priority promotion as a mitigation.
- Priority abort is guarded by estimated low-priority completion time:
  if the low-priority transaction is expected to finish before the
  high-priority timestamp, the server can avoid aborting it.
- Conditional prepare handles the case where one participant already
  prepared the low-priority transaction while another participant is
  expected to priority-abort it. The server can conditionally prepare
  the high-priority transaction, replicate the conditional result, and
  let the coordinator commit it only if the low-priority abort condition
  is confirmed.
- Transaction requests carry estimated arrival times and read/write keys
  for all participants so a server can predict whether priority abort is
  likely elsewhere before it receives the remote abort acknowledgement.
- Local early committed-state forwarding lets a transaction read from a
  committed conflicting transaction before that prior transaction's
  updates have completed normal replication, reducing lock hold time by
  roughly one WAN round trip in the intended setting.
- Remote early committed-state forwarding can forward a high-priority
  read to the prior transaction's coordinator when the local participant
  cannot yet serve the committed value.
- The prototype extends Carousel in Go with gRPC and Raft, uses one proxy
  per datacenter for periodic delay probing, and has clients refresh
  delay information from the local proxy periodically.
- Evaluation uses 5 partitions, 3 replicas per partition, 15 data
  servers across five datacenters, 10% high-priority transactions by
  default, 1 million 64-byte key/value pairs unless varied, 60-second
  runs, and latency that includes retries.
- In one YCSB+T high-contention result at 350 txn/s, the paper reports
  over 5000 ms 95th-percentile high-priority latency for Carousel and
  TAPIR versus 656 ms for Natto-TS, with further gains from forwarding
  and priority mechanisms.
- Natto is sensitive to deployment assumptions: it depends on relatively
  stable private-WAN delays and loosely synchronized clocks. Under low or
  moderate network-delay variance, the evaluated Natto variants keep
  lower high-priority latency than the baselines; high variance increases
  late-arrival aborts.

**GPU DB mapping:** The GPU DB does not need geo-distributed timestamps
to serve local pgwire clients, but Natto gives a useful structure for
classed transaction admission under contention. The strongest
transferable idea is a transaction or route descriptor with predeclared
resource and conflict sets: read keys, write keys, touched columns,
resident generations, expected route duration, priority class, and
expiration/deadline. Once that descriptor exists, owner queues can
distinguish "this low-priority COPY chunk can finish before a retained
lookup's deadline" from "this queued low-priority refresh will block a
short high-priority read and should yield, abort, or be delayed."

For the mutation owner, Natto supports priority-aware conflict handling
without giving up a deterministic order. GPU DB could use per-owner
generation timestamps or batch sequence numbers, not wall-clock WAN
timestamps, to order admitted work. Within that order, high-priority
short transactions can wait for earlier conflicting work when bounded,
but low-priority queued work can be retired, delayed, or promoted when
it would repeatedly block hot committed paths.

Conditional prepare maps to speculative preparation with an explicit
condition frontier. For example, a retained read batch, resident refresh,
or derived index update could prepare descriptors, allocate buffers, or
compile a route while waiting for a conflicting low-priority mutation,
but publication must remain conditional on the mutation's abort,
completion, invalidation, or WAL-visible generation. This fits the
standing invariant: work may be prepared early, but visibility and
resident-route eligibility are published only after the condition is
known.

Early committed-state forwarding is riskier but instructive. In GPU DB,
the analogous idea is not to expose unflushed writes. Instead, once WAL
and CPU visibility are safe but residency refresh or downstream
replication is lagging, the owner might forward reads to a CPU-visible
fresh path or a coordinator-owned committed result rather than forcing a
short transaction to wait for a full resident refresh. That creates a
classed fallback rule: high-priority reads may bypass stale GPU
residency and use the freshest safe CPU path; low-priority analytical
work can wait for a refreshed retained snapshot.

The paper also fits the 1M logical-session target. Priority should be
attached to admitted active work, not to every idle session. A small
number of high-priority requests should be able to preempt or bypass
queued low-priority work without letting all high-priority sessions
reserve pinned buffers, GPU descriptors, or mutation-owner slots.

**Risks and mismatches:** Natto is a geo-distributed 2FI transaction
system, not a local SQL engine with arbitrary pgwire statements. It
requires read and write sets to be known at transaction start, so it
maps best to prepared procedures, COPY chunks, point updates, retained
lookups, and known-shape refresh work rather than arbitrary SQL. Its
early committed-state forwarding depends on Carousel's replication and
commit protocol; GPU DB must not copy that mechanism in a way that
weakens WAL-before-visibility or resident invalidation.

Priority abort can starve low-priority work unless retry-age promotion,
budget caps, or fairness windows are explicit. Natto has only two
priority levels in the evaluated prototype, while GPU DB likely needs
classes such as commit acknowledgement, short retained read, COPY
admission, refresh, cold scan, analytical read, and maintenance. The
network-delay timestamp trick is less relevant inside one machine; the
transferable concept is a common owner-visible order plus predicted
service time, not WAN timing. Finally, Natto's absolute latency numbers
come from WAN deployments and Go/gRPC prototypes, so they should inform
conflict policy rather than local microsecond targets.

**Benchmark candidates:**

- Add route/transaction descriptors for one mutation-heavy benchmark:
  priority class, read/write key family, touched columns, resident
  generation, estimated service time, and deadline. Gate: no behavior
  change, but every queue wait and conflict report names the descriptor.
- Implement a CPU-only priority-abort simulation in the mutation owner:
  low-priority queued writes can be delayed or aborted when a short
  high-priority conflicting transaction arrives and the low-priority
  completion estimate exceeds the high-priority budget. Measure abort
  rate, starvation, p50/p99 latency, and write throughput.
- Add retry-age promotion for low-priority transactions or refresh work.
  Failure condition: a low-priority class can be aborted indefinitely
  while high-priority load remains below a configured cap.
- Prototype conditional preparation for retained read batches: prepare
  descriptors and buffers while a conflicting low-priority mutation or
  refresh is pending, but publish only after the condition frontier is
  confirmed. Required metrics: prepared-but-cancelled count, handoff
  latency, stale-generation rejection count, and correctness under WAL
  replay.
- Build an early-safe-fallback benchmark: after WAL-visible commit but
  before GPU resident refresh, route high-priority point reads to the
  CPU-visible fresh path while low-priority analytical reads wait for
  residency. Gate: identical SQL results and explicit reason
  `fresh_cpu_bypass_stale_residency`.
- Compare FIFO, static priority, and bounded priority-abort admission on
  a hot-account SmallBank-like workload at concurrency `1,2,4,8,16,32,64`.
  Expected improvement: lower p99 for short high-priority writes without
  unbounded low-priority backlog.
- For future prepared procedures, test whether predeclared read/write
  sets let COPY or update batches overlap validation, WAL staging, and
  resident invalidation the way Carousel/Natto overlap read/prepare and
  commit phases.

### 2026-06-03 - HybridGC production MVCC garbage collection in SAP HANA

**Citation:** Juchang Lee, Hyungyu Shin, Chang Gyoo Park, Seongyun
Ko, Jaeyun Noh, Yongjae Chuh, Wolfgang Stephan, and Wook-Shin Han.
"Hybrid Garbage Collection for Multi-Version Concurrency Control in
SAP HANA." SIGMOD 2016, pp. 1307-1318. DOI:
`10.1145/2882903.2903734`. Retrieved 2026-06-03 from the ACM DOI
page and a CMU-hosted PDF copy,
`https://15721.courses.cs.cmu.edu/spring2019/papers/05-mvcc3/p1307-lee.pdf`.

**Category:** MVCC / snapshot / visibility.

**Relevance tags:** MVCC garbage collection; long snapshots; HTAP;
statement snapshot isolation; transaction snapshot isolation; version
chains; table-scoped visibility; group commit metadata; memory pressure;
long cursor robustness.

**Core idea:** HybridGC addresses a production HTAP failure mode: a
long-lived analytical cursor or transaction can pin the global minimum
snapshot timestamp, causing obsolete versions to accumulate even though
many are invisible to every active snapshot. In SAP HANA's row store,
this does not only waste memory; it also increases RID-hash collisions
and version-chain traversal, hurting short OLTP work and incremental
fetch latency.

The paper combines three collectors. Global group GC reclaims whole
commit groups cheaply when their commit ids are older than the relevant
snapshot frontier. Table GC uses query-plan or declared table scopes to
move long snapshots out of the global tracker and into per-table snapshot
trackers, so a long scan over one table does not block version reclamation
for unrelated tables. Interval GC reasons over the visible interval of
each version, reclaiming intermediate versions whose interval contains no
active snapshot even when the oldest snapshot is still open.

In the reported HANA experiments, HybridGC keeps version-space size
nearly flat under a long-duration cursor where group-only collectors keep
growing. In one 1000-second TPC-C run, table GC and interval GC reclaimed
379 million and 118 million versions respectively while global group GC
was blocked by the long cursor. With all collector periods at 1 second,
the paper reports about 0.8% throughput overhead for HybridGC versus
global group GC when no long snapshot is present.

**Concrete mechanisms:**

- SAP HANA tracks active snapshot timestamps in an ordered, reference
  counted global snapshot timestamp tracker. A snapshot holds a direct
  pointer to its tracker entry, so releasing a snapshot can decrement the
  entry without scanning all active snapshots.
- Record versions created by the same transaction point to a
  `TransContext`. Transactions committed in the same group commit point
  to a shared `GroupCommitContext`, which receives the commit id once.
  This indirect commit-id publication avoids copying the id to every
  version on the commit fast path.
- Group commit contexts are maintained in commit-id order. Global group
  GC scans this list and can identify entire groups older than the
  minimum relevant snapshot timestamp without traversing each version
  chain first.
- Interval GC models version collection as consecutive interval
  intersection. For ordered active snapshot ids `S` and ordered version
  commit ids `T`, a version `t` is garbage when the next version id falls
  before or at the least active snapshot id greater than or equal to `t`;
  the paper gives a merge algorithm with `O(|S| + |T|)` cost.
- HANA's implemented interval collector first gathers active snapshot
  timestamps, then scans group commit contexts whose commit ids fall
  between the minimum and maximum active snapshot ids, then walks
  reachable version chains from highest commit id downward.
- Table GC detects long-lived snapshots, checks whether their table scope
  is known, copies their snapshot timestamp into per-table trackers for
  the relevant tables, and removes them from the global tracker.
- The per-table optimization applies naturally to statement-level
  snapshot isolation because a compiled query plan exposes the accessed
  tables. It also applies to some transaction-level cases, such as
  precompiled stored procedures or internal transactions that explicitly
  declare touched tables; HANA can reject access to undeclared tables.
- When table GC and interval GC coexist, global GC must consider both
  global and per-table snapshot trackers. The paper notes that HANA
  pre-materializes the union of available trackers to avoid repeatedly
  scanning too many per-table lists.
- The collectors run independently with different periods in the
  experiments: global group GC at 1 second, table GC at 3 seconds, and
  interval GC at 10 seconds.
- The paper states that group and table GC were already available in SAP
  HANA product versions at publication time; interval GC was
  pre-production in the described state.

**GPU DB mapping:** HybridGC sharpens the retained-snapshot design rule:
long GPU reads should not pin a single global visibility frontier that
blocks cleanup for all tables, partitions, or resident generations. The
engine should track snapshot scope at least by table and preferably by
partition/resident segment once route planning can prove that scope. A
long retained scan over one resident segment should not prevent deletion,
version pruning, or resident metadata retirement for unrelated segments.

Group GC maps to the existing WAL/MVCC batch and owner-domain model. COPY
chunks, mutation batches, and group-commit-like WAL flush batches can
publish a shared commit generation object. Cleanup can then retire whole
generation groups when no relevant retained read, transaction, refresh,
or recovery cursor can still see them. This is a better hot-path shape
than stamping or checking every row version independently.

Table GC maps to planner-visible snapshot leases. A retained GPU route
already knows table id, schema generation, predicate family, selected
columns, and eventually partition identity. That descriptor can become a
snapshot lease scope: table, partition, column group, and resident
generation. GC and resident eviction should use those scoped leases
instead of one coarse oldest-read timestamp.

Interval GC is the fallback when scope is unknown or too coarse. Even if a
long transaction pins an old snapshot, intermediate versions whose
visibility intervals contain no active snapshot can be reclaimed. For GPU
DB, the equivalent is a sparse active-snapshot set per owner and a merge
against version/generation chains. That could bound version-chain
traversal for CPU fallback and bound retained segment metadata even when
one old reader is still active.

The paper also argues for measuring GC as a latency feature, not only a
memory feature. HANA's growing version space increased hash collisions and
incremental cursor fetch latency. GPU DB should expect similar harm in CPU
indexes, visibility summaries, resident metadata maps, and stale
generation lists if old versions remain on hot lookup paths.

**Risks and mismatches:** The implementation is specific to SAP HANA's row
store, RID hash table, group commit context objects, and statement-level
snapshot default. GPU DB's current MVCC tuple store, WAL batches, and
resident GPU snapshots have different physical structures. The paper does
not describe serializable isolation, GPU execution, or over-resident
device-memory reclamation.

Table GC depends on knowing snapshot scope. Arbitrary SQL transactions,
dynamic SQL, prepared statements with late binding, and multi-table joins
may not expose a complete safe scope until planning is complete. Scope
declarations must be enforced: if a transaction declares one table and then
touches another, GPU DB must reject, replan, or widen the lease before any
cleanup relies on the narrow scope. Interval GC is more general but can be
more expensive because it touches version chains that group/table GC avoid.

The evaluation embeds TPC-C logic inside HANA to avoid network effects and
uses a 4-socket CPU machine, so its absolute throughput does not transfer
to the pgwire/GPU path. The transferable claim is the collector shape and
the observed failure mode under long snapshots, not the numeric
throughput.

**Benchmark candidates:**

- Add snapshot lease telemetry with scope classes: global transaction,
  table, partition, resident segment, refresh, and recovery cursor. Gate:
  every retained read and long CPU fallback reports its lease scope and
  release timestamp.
- Build a long-retained-scan GC stress: hold a retained scan open on one
  table or partition while applying updates/deletes to unrelated tables and
  hot partitions. Expected improvement: unrelated cleanup and fresh lookup
  latency stay bounded.
- Prototype group-generation cleanup for COPY/WAL batches: versions from
  the same flushed batch share a generation object, and GC first retires
  whole groups before row-by-row pruning. Required metrics: commit-path
  overhead, GC scan work, reclaimed versions per pass, and WAL replay
  correctness.
- Add per-table or per-partition snapshot trackers to the MVCC test
  harness. Minimum proof: a long scoped snapshot pins only matching table
  or partition versions; attempts to access undeclared scope fail or widen
  safely before cleanup.
- Implement an interval-GC simulator over version chains and active
  snapshot sets. Compare global-minimum GC, scoped GC, and interval GC
  under long statement snapshots, long transaction snapshots, and mixed
  short readers.
- Measure hot lookup degradation from stale version metadata: version-chain
  length, hash/index collision or probe depth, visibility-check count, and
  p95 lookup latency while long readers hold old snapshots.
- For resident GPU metadata, test whether stale generation lists or
  invalidated resident segments remain on the route-choice hot path under
  long scans. Failure condition: route planning or fresh lookup latency
  grows with old retained generations unrelated to the query scope.

### 2026-06-03 - DANA directly attached NVMe arrays

**Citation:** Gabriel Haas, Michael Haubenschild, and Viktor Leis.
"Exploiting Directly-Attached NVMe Arrays in DBMS." CIDR 2020.
Retrieved 2026-06-03 from the CIDR PDF,
`https://www.cidrdb.org/cidr2020/papers/p16-haas-cidr20.pdf`.

**Category:** multi-tier cache / data placement.

**Relevance tags:** NVMe arrays; cold partitions; over-resident execution;
asynchronous I/O; O_DIRECT; file-system overhead; SPDK; WAL flush latency;
HTAP I/O interference; page size; RAID; storage admission.

**Core idea:** The paper argues that an array of directly attached PCIe
NVMe SSDs is not just a faster disk tier. With enough drives, aggregate
bandwidth can approach DRAM-like scan bandwidth at much lower capacity
cost, but only if the database and OS I/O path stop wasting CPU cycles and
stop mixing latency-critical WAL or point reads with bulk scans through
coarse, blocking interfaces.

The authors study a DANA setup with four NVMe SSDs and an HTAP-shaped I/O
workload: high-depth random scan reads, one-at-a-time point reads,
rate-limited random background writes, and synchronous WAL writes. The
main conclusion is a design warning: traditional buffered, blocking,
file-system-mediated I/O leaves both throughput and tail latency on the
floor. The useful path is explicit user-space buffering, asynchronous
direct I/O, careful file-system or block-device selection, and a separate
solution for WAL durability that keeps frequent `fdatasync` calls off the
critical path.

**Concrete mechanisms:**

- The benchmark separates four database I/O classes: scan reads optimized
  for throughput, point reads optimized for latency, WAL writes followed
  by sync, and steady background writes. Running them together models an
  HTAP storage tier rather than a pure scan benchmark.
- The baseline mimics a traditional PostgreSQL-like stack: pread/pwrite,
  OS buffering, ext4, software RAID 0, and frequent `fdatasync` for WAL.
  In that setup, the WAL stream reaches only 6 MB/s with about 2.6 ms mean
  flush latency, while scan and point-read performance also suffer.
- Removing `fdatasync` from the simulated critical path raises the WAL
  stream to the target 250 MB/s and improves other I/O classes. The paper
  proposes a small persistent-memory or NVDIMM log tail as the durable
  commit target, with asynchronous flash writeback after commit.
- Switching from blocking calls to asynchronous I/O alone is not enough.
  With OS buffering still enabled, io_uring does not materially reduce CPU
  cost. The large CPU reduction comes from combining asynchronous I/O with
  `O_DIRECT`, dropping CPU cost from multi-cycle-per-byte buffered paths to
  roughly 1 cycle/byte in the reported table.
- Removing the file system and accessing block devices directly improves
  throughput and CPU use over ext4 in the main workload. Among tested file
  systems, XFS is close to direct block-device performance, while Btrfs is
  substantially worse for this workload.
- Random-scan page size is a throughput/latency dial. Larger pages reduce
  per-system-call CPU overhead and improve random scan throughput, with
  64 KB pages reaching more than 11 GB/s in the read-only experiment, but
  they increase latency compared with smaller pages.
- RAID policy changes interference. RAID 0 has little overhead, RAID 10
  doubles physical writes, RAID 5 adds read/write amplification for random
  updates and increases CPU work, and the tested "hardware" RAID performs
  worse than Linux software RAID for this setup.
- SPDK is not a drop-in win. In the mixed fio workload, SPDK performs
  similarly to the kernel direct path but burns more CPU because of polling.
  In a read-only experiment, reducing polling frequency and using large
  batches can cut CPU cost dramatically, but at the price of high per-request
  latency.
- Consumer SSD behavior is unstable under long mixed workloads. SLC cache
  exhaustion and flash garbage collection can cause sudden scan-throughput
  drops and large point-read tail-latency increases. Enterprise SSDs in the
  appendix have more stable behavior, largely due to better flush behavior
  and over-provisioning.
- In a TPC-C comparison, PostgreSQL remains CPU-bound and cannot exploit
  DANA bandwidth, while LeanStore, using O_DIRECT, asynchronous I/O, and no
  critical-path fsync, is much faster in both memory-fit and out-of-memory
  configurations.

**GPU DB mapping:** DANA is a strong fit for the P8 over-resident question:
once data exceeds GPU memory, the cold tier cannot be treated as a passive
file store. GPU DB should model NVMe arrays as an active tier with its own
queue depth, page-size, CPU-cycle, thermal, GC, and flush-latency budgets.
The route planner should decide not only "resident GPU versus CPU" but also
"GPU resident, GPU cold-transfer, CPU direct-I/O scan, CPU point lookup, or
reject/defer because the storage queue is saturated."

The four I/O classes map directly to owner-domain queues. WAL flushes,
resident refresh reads, cold partition scans, point lookups, and background
checkpoint/archive writes should not share one opaque file-system path with
no admission telemetry. A future storage owner should expose per-class queue
depth, issued bytes, completion latency, CPU cycles per byte where available,
and interference caused by WAL syncs or background writes.

The WAL result is particularly relevant to write throughput. GPU DB must not
weaken WAL-before-visibility, but it should avoid one synchronous flash flush
per logical commit if that becomes the copy-admission limiter. The safe
transferable idea is a durable low-latency log-front, such as PMem/NVDIMM or
a future replicated durable log, followed by asynchronous NVMe consolidation.
Until such hardware exists, benchmarks should explicitly measure
`fdatasync`/flush latency and group commit effects instead of hiding them.

For over-resident GPU execution, the page-size tradeoff becomes a segment
granularity tradeoff. Large cold-partition chunks improve NVMe throughput and
reduce CPU overhead, but increase latency and overfetch for point reads.
GPU DB should probably use different physical granularities for cold scans,
resident refresh, point lookup fallback, and WAL/checkpoint streams instead
of one universal page size.

The SPDK result is also a caution for GPU direct-storage ambitions. Kernel
bypass and polling can reduce CPU overhead only when the runtime can batch
large numbers of I/O completions without harming latency-sensitive work. That
matches the existing runtime target: storage/GPU workers need classed queues,
batch-drain limits, and latency ceilings, not a single always-polling
fast path.

**Risks and mismatches:** The paper is an I/O systems study, not a full
database storage design. Its workload uses fio and synthetic I/O classes,
so it does not model SQL planning, MVCC visibility checks, WAL replay,
GPU transfers, compression, or result materialization. The main hardware is
2019-era PCIe 3 consumer SSDs; modern PCIe 4/5 drives, enterprise devices,
CXL memory, and GPUDirect Storage change the exact numbers. The CIDR paper
focuses more on scan-oriented OLAP and HTAP than pure high-contention OLTP,
so GPU DB should transfer the queue and tiering mechanics rather than treat
NVMe as a substitute for resident OLTP memory.

Direct block-device access and SPDK also increase operational complexity:
space management, checksums, crash recovery, allocator metadata, and device
failure handling move into the database. For the current GPU DB, XFS plus
`O_DIRECT`/async I/O may be a better first experimental boundary than a full
block-device or SPDK rewrite. Persistent-memory WAL buffering is only a
future-tier hypothesis unless the hardware is actually present.

**Benchmark candidates:**

- Add a cold-tier I/O classification benchmark with four lanes: resident
  refresh scan, cold point lookup, checkpoint/background write, and WAL flush.
  Gate: report per-lane throughput, p50/p99 latency, queue depth, and
  interference under mixed load.
- Test page or segment granularity for cold partitions: 4 KB, 16 KB, 64 KB,
  and 256 KB reads for random scans and point fallback. Expected result:
  larger chunks improve scan throughput but hurt p99 point latency and
  overfetch.
- Add an explicit `fdatasync`/WAL flush phase profile to COPY admission and
  group-commit experiments. Failure condition: rows/sec claims omit the
  durable flush boundary needed for WAL-before-visibility.
- Prototype an async direct-I/O cold scan harness before any GPU direct
  storage work. Compare buffered file I/O, XFS `O_DIRECT`, and direct block
  access only if available. Minimum proof: no correctness change, stable
  cleanup, and truthful fallback if direct I/O is unsupported.
- For over-resident P8, add planner telemetry that distinguishes
  `gpu_resident`, `gpu_cold_transfer`, `cpu_direct_io`, `cpu_buffered`, and
  `storage_overload_reject` route reasons.
- Run a long-duration cold-tier stability smoke on the target SSD class:
  mixed scan/write/flush load for enough time to expose SLC cache exhaustion,
  device GC, or thermal throttling. Gate: route policy records degraded-tier
  state instead of silently treating old peak bandwidth as current capacity.
- Compare a future storage-worker polling path against sleep/batch policies:
  always-poll, timed sleep, queue-depth-triggered wakeup, and latency-ceiling
  wakeup. Measure CPU cycles, batch size, p99 point read, and scan bandwidth.

### 2026-06-03 - Cross-paper synthesis: scoped fronts must include storage

HybridGC, Natto, and DANA converge on the same architectural shape from
three different angles. Long MVCC readers need scoped snapshot fronts so they
do not pin unrelated cleanup. Prioritized transactions need conditional
fronts so expensive work can prepare without publishing unsafe visibility.
NVMe arrays need storage fronts so WAL, cold scans, point reads, and
background writes do not interfere invisibly behind one file-system queue.

For GPU DB, this suggests a single design track: every expensive path should
name the frontier it is waiting on and the resource class it is consuming.
Examples are WAL durable generation, visibility generation, resident segment
generation, snapshot lease scope, storage queue class, GPU stream class, and
response-buffer ownership. The useful planner/admission question becomes:
"which front is limiting this request, and can a narrower or safer front let
another class proceed?"

The category gap after this batch is still cold-tier and optimizer coupling.
The next tiering papers should focus on how to choose local DRAM, remote/CXL
memory, NVMe, and resident GPU memory per workload. The next optimizer papers
should tie robust route choice to those resource fronts rather than only to
cardinality error.

**Benchmark priorities:**

- scoped snapshot leases plus group-generation GC under long retained scans
- priority-aware mutation/refresh descriptors with starvation protection
- mixed cold-tier I/O lane profiling before over-resident GPU direct-storage
  claims
- planner route reasons that name the limiting front, not just the chosen
  device

### 2026-06-03 - Bf-Tree variable-length mini-pages for larger-than-memory indexes

**Citation:** Xiangpeng Hao and Badrish Chandramouli. "Bf-Tree: A Modern
Read-Write-Optimized Concurrent Larger-Than-Memory Range Index." PVLDB
17(11), 2024, pp. 3442-3455. doi:10.14778/3681954.3682012. Retrieved
2026-06-03 from the PVLDB PDF,
`https://www.vldb.org/pvldb/vol17/p3442-hao.pdf`.

**Category:** multi-tier cache / data placement.

**Relevance tags:** larger-than-memory indexes; buffer management; NVMe;
hot-record caching; write buffering; range scans; cold partitions; cache
granularity; concurrent access methods.

**Core idea:** Bf-Tree argues that traditional larger-than-memory B-Trees
couple the cache granularity to the disk-page granularity. That makes one hot
record pull an entire cold-heavy page into memory, and makes one small update
dirty and rewrite the full page. LSM-style and delta-chain alternatives reduce
some write amplification but can add read, scan, or compaction cost.

The paper's central mechanism is the mini-page: a variable-length in-memory
representation associated with a disk leaf page, but not required to mirror the
whole disk page. A mini-page can hold selected hot records, buffered updates,
range gaps, or grow to a full page when range scans make that worthwhile. The
result is a B-Tree-like range index whose memory component acts more like a
workload-shaped hot/cold tier than a simple page cache.

In the reported YCSB-like evaluation, Bf-Tree is claimed to be 2.5x faster
than RocksDB for scans, 6x faster than a conventional B-Tree for writes, and
2x faster than both B-Trees and LSM-Trees for point lookups. The useful lesson
for GPU DB is not the exact throughput number; it is the storage-engine shape:
the unit stored on NVMe, the unit cached in host memory, and the unit promoted
to GPU memory do not have to be identical.

**Concrete mechanisms:**

- Disk leaves remain page-oriented, while in-memory mini-pages are
  variable-length cache/update objects tied to those leaves.
- Mini-pages can cache individual hot records, range gaps between keys,
  recent updates, or full-page mirrors when sequential range access dominates.
- A fixed-size circular buffer stores mini-pages. Allocation advances a tail
  pointer; deallocation returns regions to a free list.
- Mini-page grow/shrink uses a read-copy-update style replacement: allocate a
  new mini-page, copy the old content plus the change, then publish the new
  pointer.
- When the circular buffer fills, older mini-pages near the head are evicted,
  preserving a bounded memory budget rather than letting hot-record caching
  become a side cache with its own uncontrolled capacity.
- Bf-Tree uses a copy-on-access region as an approximate LRU mechanism. The
  paper's default is 10% of the circular buffer, balancing runtime overhead
  against cache quality.
- Promotion from a disk page into a mini-page is probabilistic. The reported
  default promotion rate is 20%, trading off quick response to workload shifts
  against pollution from one-time cold accesses.
- Buffered writes can be absorbed into mini-pages and later flushed to disk
  pages, reducing page-write amplification for small record updates.
- Range scans can still work efficiently because mini-pages may represent
  gaps or grow toward full-page contents rather than being only a point-record
  cache.
- The implementation is concurrent and larger-than-memory; the paper notes
  careful interaction between mini-pages and disk pages for consistency, and
  WAL replay reapplies operations to the corresponding page during recovery.
- The evaluation highlights cache sensitivity: Bf-Tree's advantage is larger
  when much of the data is on disk, while systems converge as data becomes
  memory resident.

**GPU DB mapping:** Bf-Tree maps directly to P8's open question about resident
segment and cold-partition granularity. GPU DB should not assume that the CPU
MVCC tuple, host-memory cache object, NVMe block, and GPU resident segment all
share one physical size. The transferable design is to give each tier its own
unit: durable WAL records and cold pages for recovery, compact host mini-pages
or mini-segments for hot records and updates, and GPU column/key vectors for
batchable retained reads.

For cold-partition indexes, a mini-page-like host tier could sit between the
CPU canonical tuple store and full GPU residency. Hot equality keys, recent
updates, deleted-key/tombstone markers, or key ranges that repeatedly miss GPU
residency could be cached in bounded host structures without admitting a whole
disk page or full GPU segment. That complements DANA's lane model: storage
lanes decide how bytes move; mini-pages decide which bytes deserve to move.

The write-buffering side is also relevant to WAL/MVCC. Bf-Tree does not remove
the need for WAL-before-visibility, but it suggests that post-WAL index/cache
maintenance can accumulate in compact per-page or per-partition host buffers
before a cold page or GPU segment is rewritten. The GPU DB equivalent would be
delta mini-segments that fresh CPU lookups can merge, while GPU retained
snapshots either use a previous immutable generation or refresh at a
deterministic batch boundary.

Promotion-rate and copy-on-access tuning should become explicit telemetry in
P8. A GPU-resident cache that promotes every observed key will pollute device
memory under scans and one-off lookups. A cache that promotes too slowly will
miss shifting hot sets. The planner/admission layer should expose promotion
reason, sampled hotness, mini-segment bytes, and whether the chosen route
served point lookup, range scan, or write buffering.

**Risks and mismatches:** Bf-Tree is a key-value/range-index design, not a
SQL MVCC storage engine. The paper does not solve tuple visibility,
serializable reads, DDL invalidation, GPU layout, PostgreSQL protocol work, or
columnar analytical execution. The mini-page abstraction is row/key oriented;
GPU DB may need column-family mini-segments, key-order vectors, or tombstone
bundles rather than literal B-Tree mini-pages.

Variable-length buffers add fragmentation, copy cost, and concurrency
complexity. A GPU DB implementation would also need crash recovery rules:
mini-pages should remain rebuildable acceleration state unless deliberately
made durable. Promotion-rate heuristics can be workload sensitive, and the
paper's defaults should be treated as starting points rather than universal
constants. Finally, the evaluation compares storage engines on CPU/NVMe
workloads; it does not measure GPU transfer, kernel launch, resident snapshot
retirement, or MVCC chain traversal.

**Benchmark candidates:**

- Add a host mini-segment simulator for cold `int4` key lookups: compare full
  4 KB page caching, record-level mini-segments, and no host cache under
  Zipfian and shifting-hot workloads. Gate: identical lookup results and
  measured bytes promoted per hit.
- Extend the P8 residency benchmark plan with separate units for NVMe page,
  host mini-segment, and GPU resident column group. Failure condition: one
  hard-coded page/segment size is used for all tiers without telemetry.
- Prototype a post-WAL update buffer for cold index entries that can be merged
  with CPU lookups before rewriting cold pages or refreshing GPU segments.
  Gate: WAL replay rebuilds the same visible index/cache state.
- Measure promotion policies for retained lookup caches: always promote,
  sampled promotion at 1%, 10%, 20%, and promote-after-N-hits. Required
  metrics: hit ratio, resident bytes, p95 lookup latency, GPU fallback count,
  and pollution after a cold scan.
- Add a range-scan stress where point-hot keys are scattered across cold pages.
  Compare full-page caching versus mini-segment/gap caching for both point
  reads and bounded range scans.
- Track route reasons that distinguish `host_mini_segment_hit`,
  `host_mini_segment_merge`, `gpu_resident_hit`, `cold_page_read`, and
  `promotion_rejected_budget`. This makes cache granularity visible to the
  optimizer instead of hidden inside the storage layer.
- Test variable-size cache object accounting under concurrency: grow, shrink,
  evict, and retire mini-segments while retained snapshots hold old
  generations. Proof gate: no reuse before all readers release the generation.

### 2026-06-03 - ERMIA snapshot-friendly mixed-workload OLTP

**Citation:** Kangnyeon Kim, Tianzheng Wang, Ryan Johnson, and Ippokratis
Pandis. "ERMIA: Fast Memory-Optimized Database System for Heterogeneous
Workloads." SIGMOD 2016, pp. 1675-1687. doi:10.1145/2882903.2882905.
Retrieved 2026-06-03 from the author PDF,
`https://www2.cs.sfu.ca/~tzwang/ermia.pdf`; publisher page:
`https://dl.acm.org/doi/10.1145/2882903.2882905`.

**Category:** transaction processing / write path and MVCC / snapshot /
visibility.

**Relevance tags:** heterogeneous OLTP; long read-mostly transactions;
snapshot isolation; serializability; indirection arrays; append-only storage;
centralized logging; epoch reclamation; mixed OLTP/HTAP fairness.

**Core idea:** ERMIA argues that lightweight OCC, while excellent for short
low-contention OLTP, is a poor default for heterogeneous workloads that mix
short write transactions with longer read-mostly transactions. Commit-time
read validation can let long readers consume CPU and then abort, and its
writer-favoring conflict resolution can starve read-mostly work.

ERMIA instead starts from snapshot isolation, then optionally overlays Serial
Safety Net (SSN) to provide serializability. Its physical design makes that
practical: append-only version creation behind latch-free indirection arrays,
a log manager that gives each committing transaction a globally ordered LSN
with one common-case atomic reservation, and fine-grained epoch managers for
log buffers, transaction IDs, and garbage collection.

For GPU DB, the important lesson is that retained read throughput should not
come from "validate late and retry" under mixed write/read pressure. Long
retained scans, refreshes, or GPU micro-batches need immutable snapshots and
early conflict boundaries so they do useful work once admitted. Write
throughput still needs a serialized durability/visibility front, but it should
publish versions and snapshot generations cheaply enough that readers do not
block writers and writers do not invalidate in-flight readers in place.

**Concrete mechanisms:**

- Each logical record has an object ID whose indirection-array slot points to
  the head of an in-memory version chain.
- Inserts allocate a new OID and fill the corresponding indirection slot;
  updates create a new version out of place and install it with CAS against
  the slot head.
- An uncommitted head version acts as the write-write conflict marker. ERMIA
  uses first-updater-wins, so doomed updaters can abort early instead of doing
  a full transaction and discovering the conflict at commit.
- Index leaves store OIDs rather than physical tuple addresses. Updating a
  record usually updates the indirection array and version chain, not every
  index reference.
- Snapshot reads traverse version chains and compare the reader's begin LSN
  with version creation timestamps. If a version is still TID-stamped, the
  reader consults the owner transaction context.
- Transactions keep log descriptors privately during execution, then reserve
  globally ordered log space at pre-commit with a single atomic
  fetch-and-add in the common case.
- The log sequence-number space can contain holes; ERMIA translates logical
  LSNs through segment metadata rather than requiring every allocation to be
  contiguous in physical log files.
- Commit has a pre-commit phase that fixes order, runs the concurrency-control
  protocol, and copies private log records into the reserved log space; then a
  post-commit phase replaces TID stamps on versions with the commit LSN.
- Multiple epoch managers track resources at different time scales: log
  buffers, transaction IDs, and garbage-collectable versions.
- Version garbage collection scans indirection arrays and removes versions no
  longer needed by any active transaction.
- For serializability, ERMIA layers SSN over SI. SSN tracks dependency stamps
  and uses an exclusion-window test at commit, instead of SSI-style dangerous
  structure tracking that can bias aborts toward writers.
- Phantom protection reuses tree node version validation from Silo: range
  reads remember index leaf versions and validate them before commit.
- Recovery treats the log as the durable database and rebuilds volatile OID
  arrays from fuzzy checkpoints plus sequential log scanning.
- The evaluation reports that ERMIA-SI and ERMIA-SSN maintain near-linear
  scalability over the tested 24 hardware threads and preserve read-mostly
  transaction throughput where the Silo-style OCC comparison collapses under
  TPC-C/TPC-E hybrid workloads. Exact numbers are workload dependent.

**GPU DB mapping:** The indirection-array idea maps cleanly to GPU DB's
separation between CPU truth, WAL visibility, and GPU acceleration state. CPU
MVCC records can keep stable logical row IDs while indexes, host mini-segments,
and GPU resident column groups point through versioned publication metadata
instead of physical tuple addresses that change on every update.

The stronger transferable idea is the transaction lifecycle. GPU DB should
make every mutation produce a private descriptor first, reserve durable/logical
order at a narrow commit boundary, then publish visibility and invalidate or
refresh resident generations. That gives retained readers a clear snapshot
boundary and gives COPY/admission benchmarks a concrete phase split:
descriptor build, WAL reservation, WAL flush, visibility publish, residency
invalidation, and optional refresh.

ERMIA also reinforces that read-mostly retained work should be admitted against
immutable generations. A long GPU aggregate or lookup batch should not sit on
mutable owner state and then discover at response time that a writer won. It
should either run against an already-published generation, take a narrower
snapshot lease, or be rejected/fallback before consuming GPU queue time.

The epoch-management design is directly relevant to retained CUDA buffers,
pinned response buffers, and host mini-segments. Published GPU snapshots need
read-copy-update style retirement: writers publish a new generation, readers
finish on the old one, and reclamation occurs only after all active readers and
GPU events have crossed the epoch.

The log design is not a direct replacement for GPU DB WAL, because ERMIA's
experiments write log records asynchronously to tmpfs. Still, the single
reservation point is useful: GPU DB should avoid per-row global log contention
inside COPY and instead reserve or publish ordered chunks when correctness
allows. Durability must still be measured at the real flush boundary.

**Risks and mismatches:** ERMIA is a main-memory CPU OLTP engine, not a GPU
storage engine. Its evaluated hardware is small by current standards, and the
paper does not measure GPU execution, PostgreSQL protocol overhead, CUDA
stream ownership, NVMe cold-tier behavior, or 1M logical sessions.

The logging evidence is especially limited for GPU DB's durability goals:
log records are written asynchronously to tmpfs, so the evaluation does not
prove sustained WAL-before-visibility throughput on real storage. ERMIA's
"log is the database" recovery shape may also conflict with GPU DB's current
WAL/checkpoint/archive model unless adopted only as an internal versioning
pattern.

Indirection arrays add cache misses and metadata pressure. For GPU DB, an
extra logical-to-physical hop may be fine on CPU owner paths but harmful inside
GPU kernels unless resident snapshots flatten visibility into GPU-friendly
vectors. SSN and phantom validation also need careful accounting: dependency
metadata that is cheap at 24 threads may become expensive under very high
logical session counts or long retained readers.

**Benchmark candidates:**

- Split COPY admission telemetry into descriptor build, ordered WAL
  reservation, durable flush, visibility publish, residency invalidation, and
  optional refresh. Gate: no rows/sec claim hides the WAL-before-visibility
  boundary.
- Prototype chunk-level ordered WAL reservation for COPY instead of per-row
  global synchronization. Failure condition: crash replay cannot reconstruct
  the same visible row set and invalidation generation.
- Add a retained-read starvation benchmark: one long retained aggregate or
  lookup batch mixed with short updates. Compare late-validation retry,
  immutable snapshot generation, and owner-serialized execution. Gate:
  read-mostly work either commits consistently or is rejected before consuming
  expensive GPU time.
- Test RCU-style generation retirement for resident CUDA buffers and host
  mini-segments. Proof gate: old buffers are never reused until CPU readers and
  GPU completion events release the generation.
- Add route telemetry for `snapshot_generation`, `visibility_front`,
  `invalidation_generation`, and `retirement_epoch` so planner/admission
  decisions name the logical front being consumed.
- Measure stable logical row IDs plus generated GPU column snapshots against
  physical-row-address indexes under updates. Expected result: indirection
  helps update/index maintenance while resident GPU snapshots should flatten
  the extra hop before kernel execution.
- Evaluate a bounded SSN-like dependency tracker only as a serializability
  experiment for CPU transactions first. Minimum proof: dependency metadata,
  abort reasons, and cleanup cost remain bounded under long retained reads.

### 2026-06-03 - zicIO DB-OS prefetch for rapid ingestion

**Citation:** Kyungmin Lim, Minseok Yoon, Kihwan Kim, Alan David Fekete, and
Hyungsoo Jung. "Rapid Data Ingestion through DB-OS Co-design." Proceedings of
the ACM on Management of Data 3(1), Article 68, SIGMOD 2025, pp. 1-28.
doi:10.1145/3709718. Retrieved 2026-06-03 from the ACM SIGMOD table of
contents and Seoul National University research page:
`https://doi.org/10.1145/3709718`,
`https://sigmodconf.hosting.acm.org/2025/toc-3-1.html`, and
`https://gsds.snu.ac.kr/research-post/rapid-data-ingestion-through-db-os-co-design/`.
Full PDF access was not available through the accessible sources in this run,
so paper-section details not exposed by those primary/author pages are marked
unknown.

**Category:** multi-tier cache / data placement and runtime / DB-OS I/O
coordination.

**Relevance tags:** sequential ingestion; DB/OS co-design; full-device-speed
prefetch; shared memory control plane; OS-issued storage requests; cache-bypass
sharing; page-table sharing; COPY admission; cold-partition prefetch;
over-resident execution.

**Core idea:** zicIO targets the gap between two unsatisfying ingestion
routes. Conventional DBMS/OS stacks preserve compatibility and caching, but
spend substantial CPU time in data-access control. Direct or zero-copy bypass
paths reduce those layers, but concurrent scans over the same table can fetch
the same data repeatedly because they bypass the caching mechanisms that would
normally share it.

The design moves sequential access control into a DB-oriented OS component.
The DBMS supplies precise timing information, and the OS-side component issues
storage requests just before the DBMS needs the bytes. A sharing-enabled path
then restores concurrent sharing at the OS level, so bypassed ingestion does
not degenerate into redundant device traffic when multiple queries read the
same data.

For GPU DB, the transfer is not "put the OS in charge of database
correctness." It is narrower: keep WAL, visibility, and resident generation
ownership in the DBMS, but expose enough route timing to a storage/runtime
service that cold partitions can be prefetched before GPU or CPU workers stall.
The same split may apply to COPY admission: the mutation owner decides durable
order and visibility, while a storage lane can stage sequential bytes and
report backpressure without every request performing its own data-access
bookkeeping.

**Concrete mechanisms:**

- zicIO is presented as a zero-interaction and copy I/O design for sequential
  data ingestion.
- The paper decomposes the design into UzicIO, a user-space library that
  gathers precise DBMS timing information and predicts data needs; KzicIO, an
  OS module that automates access control and directly issues storage-device
  requests; and memSB, a small shared-memory area mapped into both the DBMS and
  OS for coordination.
- The OS-side module prepares data immediately before DBMS consumption, aiming
  to hide or remove known I/O latency sources rather than making the DBMS
  maintain a custom I/O stack.
- The sharing-enabled variant, SKzicIO, addresses concurrent bypass scans by
  sharing data at the OS level through dynamic page-table manipulation.
- The reported implementation was integrated with four databases according to
  the ACM abstract, while the SNU page specifically says three database engines
  were evaluated with and without zicIO under standard data warehouse workloads
  plus microbenchmarks. The source discrepancy likely reflects wording or
  scope differences between integration and evaluation; exact per-engine
  details are unknown from accessible sources.
- Reported evaluation claims include up to 9.95x improvement under TPC-H loads
  and up to 16.31x improvement in sequential-ingestion microbenchmarks.
- The accessible source does not expose exact device models, OS/kernel
  changes, workload parameters, or tail-latency distributions.

**GPU DB mapping:** zicIO fits P8's cold/warm/hot tier problem: when a resident
GPU route misses, the system should not discover the need for cold bytes only
after a GPU worker is already idle. A GPU DB storage lane could accept
route-timing hints from the planner or runtime, prefetch cold partition ranges
into host memory or pinned buffers, and publish readiness/fallback telemetry to
GPU execution owners.

For COPY, the design suggests separating control-plane timing from durability
semantics. The COPY path can keep WAL-before-visibility and chunk-level
commit-order reservation inside the mutation owner, while a lower storage lane
stages sequential input, compressed blocks, or cold checkpoint pages. The
contract should be explicit: storage may prefetch, share, and throttle bytes;
only the DBMS publishes visibility and invalidates resident generations.

SKzicIO's page-table sharing is especially relevant to over-resident reads.
If two logical sessions ask for overlapping cold partitions, bypassing the
buffer pool should not force duplicate NVMe reads or duplicate host-pinned
copies. GPU DB needs a shared cold-partition inflight table keyed by relation,
partition, source generation, byte range, and consumer class. Completion should
fan out to waiting CPU/GPU route work, with admission telemetry showing whether
the request joined an inflight prefetch or issued a new device read.

The UzicIO/KzicIO/memSB split also maps to the runtime owner model. Instead of
letting every query worker call into storage directly, network/read/GPU owners
could publish small timing descriptors into bounded shared rings. A storage
owner consumes those descriptors, schedules prefetch, and writes completion or
pressure signals back. That keeps the hot path measurable and avoids turning
custom I/O into another unbounded side channel.

**Risks and mismatches:** The paper targets sequential data warehouse
ingestion, not OLTP mutation correctness, MVCC visibility, PostgreSQL protocol
latency, or GPU kernel scheduling. Its best evidence is for TPC-H-style and
sequential-ingestion workloads, so the design may not help short point
lookups, write-heavy transactions, or highly random cold-partition access.

Moving access control into an OS module increases deployment and debugging
surface area. GPU DB should not adopt kernel changes before proving the same
contract with a user-space storage owner, `io_uring`, direct I/O, or SPDK-like
lane. Page-table manipulation may also conflict with pinned buffers, GPU DMA,
NUMA placement, container isolation, or future CXL memory tiers.

The accessible sources do not reveal failure-handling details: cancellation,
prefetch misprediction, partial I/O, security boundaries, fsync/durability
interaction, or crash recovery. Those must be treated as unknown, not assumed
safe. For GPU DB, any prefetch layer must remain acceleration state; WAL,
checkpoint, archive, and MVCC replay remain the recovery truth.

**Benchmark candidates:**

- Build a user-space cold-partition prefetch lane before any kernel work:
  planner/runtime submits `(relation, partition, generation, byte_range,
  deadline)` descriptors; GPU/CPU readers either join inflight work, consume a
  ready buffer, or record a precise fallback reason. Gate: no duplicate read
  for concurrent identical cold ranges unless generation differs.
- Add COPY ingestion phase telemetry for input staging separately from WAL
  reservation, WAL flush, visibility publish, and residency invalidation.
  Failure condition: a rows/sec claim improves only by hiding durability or
  visibility work inside "I/O".
- Compare three cold-read paths under overlapping analytical scans: normal
  buffered read, direct I/O with no sharing, and shared inflight prefetch.
  Required metrics: device bytes, host bytes copied, pinned-buffer residency,
  p50/p95 route latency, and duplicate-read count.
- Test prefetch timing hints for over-resident GPU scans: issue prefetch at
  planning time, queue-drain time, and kernel-ready time. Expected result:
  earlier hints reduce GPU idle time only if misprediction and eviction costs
  stay bounded.
- Add a page-sharing compatibility probe before considering page-table tricks:
  measure whether pinned host buffers, CUDA registration, NUMA binding, and
  memory-pressure eviction preserve correctness and do not explode tail
  latency.
- Expose storage-lane backpressure as route reasons:
  `prefetch_joined_inflight`, `prefetch_ready`, `prefetch_miss`,
  `prefetch_cancelled`, `prefetch_budget_rejected`, and
  `prefetch_generation_mismatch`.

### 2026-06-03 - Cross-paper synthesis: I/O lanes need ownership contracts

Recent storage and mixed-workload papers converge on a sharper tiering rule:
fast I/O is not a property of the device alone. DANA and the modern NVMe work
argue for high-parallelism storage lanes; Bf-Tree argues that cache objects
should have tier-specific granularity; ERMIA argues that visibility and
version lifetime must be published through durable/logical fronts; zicIO adds
that the DBMS should expose timing to lower I/O services without surrendering
semantic ownership.

The promising design track is an explicit storage-lane owner contract. GPU DB
should keep mutation order, WAL-before-visibility, MVCC generation, and
resident invalidation in database-owned domains. Separate storage lanes may
prefetch, share inflight cold reads, stage sequential bytes, and report device
pressure, but they must key every action by relation, partition, generation,
and consumer class. That makes the lane fast without letting it invent
visibility.

The main category gap is still end-to-end admission at very high logical
session counts. Runtime papers show how to multiplex and schedule; storage
papers show how to keep devices busy; MVCC papers show how to preserve
snapshots. The missing bridge is a benchmark where 1K-1M logical sessions
compete for a few owner domains, shared cold-prefetch lanes, and GPU execution
queues while every fallback is named.

Priority benchmark track: implement shared cold-partition inflight accounting
before kernel-bypass storage. Measure duplicate device reads, GPU idle time,
queue wait, snapshot-generation mismatch, and prefetch pollution. A useful
first proof is not raw maximum throughput; it is showing that concurrent
sessions with overlapping cold ranges produce one storage action, many
well-routed consumers, and no visibility or resident-generation ambiguity.

### 2026-06-03 - Predicate Transfer for multi-join pre-filtering

**Citation:** Yifei Yang, Hangdong Zhao, Xiangyao Yu, and Paraschos Koutris.
"Predicate Transfer: Efficient Pre-Filtering on Multi-Join Queries." CIDR 2024.
Retrieved 2026-06-03 from the CIDR PDF,
`https://www.cidrdb.org/cidr2024/papers/p22-yang.pdf`.

**Category:** query optimization / planning.

**Relevance tags:** predicate transfer; Bloom filters; multi-join planning;
over-resident transfer reduction; GPU join admission; cardinality feedback;
route robustness; pre-filter scheduling.

**Core idea:** Predicate Transfer generalizes Bloom join from a one-hop
pre-filter into a multi-hop join-graph phase. A local predicate on one table is
encoded as a compact filter, transferred across equi-join edges, transformed at
intermediate tables when join keys change, and used to reduce the base inputs
before the normal join phase runs. The paper borrows the shape of the
Yannakakis semi-join phase, but replaces expensive exact semi-joins with cheaper
Bloom-filter construction and probing.

The design goal is deliberately practical rather than theoretically optimal.
Yannakakis can remove all non-contributing tuples for acyclic joins, but its
semi-join phase pays hash-table build/probe and memory costs. Predicate
Transfer accepts bounded Bloom-filter false positives in exchange for a lighter
pre-filter phase that can also operate on cyclic join graphs and selected
non-inner-join or non-join operators. In the paper's preliminary TPC-H
evaluation on FPDB/Apache Arrow, Predicate Transfer outperforms Bloom join by
3.3x on average, with much larger wins on join-heavy queries such as TPC-H Q5.

**Concrete mechanisms:**

- The optimizer builds a join graph where vertices are tables and equi-join
  edges are join predicates.
- A directed predicate-transfer graph is selected from that join graph. The
  prototype orients every edge from the smaller table to the larger table,
  keeps all join edges, and relies on that heuristic to produce a DAG.
- Execution uses two phases: a predicate-transfer phase that builds and applies
  filters, followed by the ordinary join phase over the reduced table inputs.
- Filter transformation handles key changes across multi-hop transfers. When a
  table receives a filter on one join attribute and must send a filter on
  another, it scans the relevant join-key columns once, applies incoming and
  local filters, and inserts the outgoing join keys for surviving rows into a
  new filter.
- A forward pass starts from leaf/source nodes in topological order. A table
  waits for all incoming filters, scans once regardless of the number of
  incoming or outgoing edges, and emits transformed filters downstream.
- A backward pass reverses edge directions and repeats the same process so
  predicates can flow back toward earlier tables.
- Bloom filters are the prototype representation, but the abstraction permits
  precise filters or future filter types. The paper's tradeoff is no false
  negatives, acceptable false positives, and low construction/probe cost.
- Left and right outer joins can participate only in the direction that
  preserves outer-join semantics; full outer joins block transfer.
- Operators such as projection, sorting, top-K, and filters do not block
  transfer; grouped aggregation is allowed when the join key is a subset of the
  group key. Scalar UDFs may block reverse transfer if not invertible.
- The authors identify transfer-path pruning and better scheduling as future
  work. The prototype always performs full forward and backward passes.
- The filtered tables can be fed into an otherwise ordinary executor. The paper
  also notes that the transfer phase produces updated cardinalities, so a
  replan between transfer and join may improve the final join plan.
- Evaluation uses a single AWS r5.4xlarge, TPC-H SF1 and SF10, one CPU core,
  FPDB, Parquet inputs, and Apache Arrow join/Bloom-filter implementations.
  Results may vary with DBMS-specific Bloom-filter and join costs.

**GPU DB mapping:** Predicate Transfer is a clean planner-side candidate for
reducing GPU and storage work before multi-join routes. For P8, the useful
artifact is not a generic "do joins faster" rule; it is a pre-execution filter
lane keyed by snapshot generation, relation/partition identity, join key, and
query shape. If a selective dimension predicate can flow to a large fact
partition, GPU DB can avoid reading cold NVMe ranges, avoid staging host-pinned
buffers, and avoid launching kernels over rows that cannot survive the join.

The filter-transformation scan maps well to resident or warm column groups. A
GPU/CPU route can scan only the join-key columns for a relation, apply incoming
filters, and produce outgoing key filters before materializing full rows. That
fits the current P8 layout idea of dense column buffers and optional key-order
vectors: filters should operate over narrow key vectors first, then admit full
payload columns only for surviving partitions or row sets.

The two-phase shape also gives the runtime an explicit admission boundary. A
query can spend a bounded amount of CPU/GPU work constructing filters, then
either proceed with smaller inputs, replan, or fall back if filter cost,
false-positive rate, memory budget, or snapshot mismatch makes the route
unattractive. This is safer than letting a GPU join discover bad selectivity
after over-resident reads and kernels have already consumed scarce queues.

For 1M logical-session planning, Predicate Transfer suggests a shared filter
cache or inflight filter table. Same-shape requests against the same snapshot
and join graph should be able to join an existing filter-build phase rather than
each session independently scanning key columns and building equivalent Bloom
filters. The cache key must include visibility and invalidation generation, or
the filter becomes a stale-route bug.

**Risks and mismatches:** The paper is OLAP-oriented and evaluated on TPC-H,
not OLTP transactions, pgwire latency, MVCC writes, or GPU execution. Its
prototype is CPU/Arrow-based, so absolute speedups do not transfer directly.
The full-pass schedule can waste cycles when transferred filters are weak; GPU
DB should add path pruning and budget checks before adopting the technique on
latency-sensitive routes.

Bloom filters have false positives. They are safe as pre-filters but cannot
replace final join predicates, visibility checks, or SQL correctness. The
current paper also treats the transfer graph as fixed during runtime, while GPU
DB's tier state can change between planning and execution because of mutation,
residency invalidation, pressure, or cold-prefetch cancellation. Filter
construction must therefore validate the snapshot generation at both build and
consume time.

The technique helps most when local predicates are selective and can move
through multi-hop join structure. It may not help point lookups, single-table
aggregates, short OLTP transactions, or queries where filter construction costs
more than the avoided work. Outer joins and aggregations need explicit semantic
guards.

**Benchmark candidates:**

- Add a planner experiment for one TPC-H-style multi-join route: build
  snapshot-keyed Bloom filters over dimension predicates, transfer them to a
  large fact-like table, and measure skipped rows, filter bytes, build/probe
  time, and final join correctness.
- Compare three fact-table admission modes for over-resident joins: no
  pre-filter, one-hop Bloom join, and multi-hop predicate transfer. Required
  metrics: NVMe bytes read, H2D bytes, pinned-buffer occupancy, kernel rows
  scanned, p50/p95 latency, and false-positive rate.
- Prototype filter transformation over resident key vectors only. Proof gate:
  outgoing filters are generation-compatible and final SQL results are
  identical to unfiltered joins under insert/update/delete invalidation.
- Add transfer-path pruning telemetry: `filter_selectivity`, `filter_build_us`,
  `filter_probe_us`, `avoided_bytes`, and `route_aborted_due_to_filter_budget`.
  Failure condition: the pre-filter phase increases p95 latency on low-selective
  or small-input queries.
- Test shared same-shape filter inflight accounting for concurrent sessions.
  Gate: concurrent identical requests join one filter-build action and fan out
  to many consumers without duplicate key-vector scans or stale-generation use.
- Replan after filter construction in a CPU-only prototype first. Expected
  improvement: better join order and smaller intermediate materialization after
  transfer-updated cardinalities; failure condition: replanning overhead exceeds
  avoided join work.

### 2026-06-03 - Query Fresh synchronous log shipping with fresh replicas

**Citation:** Tianzheng Wang, Ryan Johnson, and Ippokratis Pandis. "Query
Fresh: Log Shipping on Steroids." PVLDB 11(4), 2017, pp. 406-419.
DOI: `10.1145/3164135.3164137`. Retrieved 2026-06-03 from
`https://www.vldb.org/pvldb/vol11/p406-wang.pdf`.

**Category:** transaction processing / write path and runtime / storage.

**Relevance tags:** WAL shipping; synchronous replication; append-only
storage; replay freshness; RDMA; NVRAM; read replicas; snapshot isolation;
indirection arrays; commit latency; recovery.

**Core idea:** Query Fresh attacks the usual hot-standby tradeoff between
safety, freshness, and primary throughput. Traditional synchronous physical
log shipping keeps committed work safe but makes the primary wait for network
and storage, while backups often expose stale reads because they must replay
logs into a second "real" database copy before queries can see recent changes.

The paper's answer is to stop treating the log as a transient replay input.
In an append-only ERMIA-based storage design, the log is the database: redo-only
committed records are shipped to backup NVRAM log buffers, replay updates
in-memory indirection arrays, and indexes map keys to logical RIDs rather than
physical record locations. This makes replay mostly a scan-and-publish step,
so backups can stay fresh and still reserve most cores for read-only work.

**Concrete mechanisms:**

- The primary ships batches at group-commit or log-flush boundaries. It posts
  one RDMA Write with Immediate per backup, overlaps network transfer with
  local log persistence, and polls completion only to keep RDMA state correct.
- Backup log buffers live in byte-addressable persistent memory in the design.
  The paper is careful that RDMA completion alone does not prove persistence:
  DDIO and CPU caches can make bytes visible before they are durable.
- For a general-purpose server method, the backup must flush or write back the
  received cache lines, issue a fence, and acknowledge persistence before the
  primary can treat the remote copy as durable.
- Log records are redo-only physical records generated only by committed
  transactions. The recovery/replay path therefore avoids undo and does not
  need deterministic re-execution of transaction logic.
- The durable append-only log stores the actual record versions. In-memory
  indirection arrays map each logical RID to the current physical version
  location; indexes point to RIDs, so updates can publish a new version by
  changing indirection rather than updating every index.
- Query Fresh keeps separate data and replay arrays during replay pipelining.
  Replay can make new versions available through the replay array before the
  data array is fully reconciled, reducing freshness lag.
- Replay is parallel and lightweight. For updates, replay mostly sets
  indirection; only inserts need index work in the described ERMIA/Masstree
  implementation.
- Multi-buffering reduces waits for reusable log-buffer space while shipped
  buffers are still being acknowledged or replayed.
- The evaluation uses full TPC-C read/write traffic on the primary and TPC-C
  read-only Stock-Level and Order-Status transactions on backups. With 56Gbps
  InfiniBand, the paper reports about 4-6% primary overhead versus a
  standalone 620k TPS server before network saturation, and support for 4-5
  synchronous backups at roughly 1.4GB/s of log records per backup.
- The paper reports backup replay of 16MB log batches in about 12ms using
  roughly one quarter of the machine's compute resources, and shows pipelined
  replay keeping commit latency within about 1.16x of standalone before the
  network saturates.

**GPU DB mapping:** Query Fresh is relevant less as "use RDMA now" and more as
a storage contract for the GPU DB write path. WAL-before-visibility should
remain the hard boundary, but once a batch is safely durable, derived read
structures should be publishable by cheap indirection/generation updates
instead of re-materializing every secondary structure synchronously.

For P8, the append-only log plus RID indirection suggests a clean split between
durable authority and acceleration state. CPU canonical MVCC versions, GPU
resident column buffers, cold compressed segments, and future remote replicas
can all be rebuilt or advanced from append-only version records, while indexes
and resident snapshots point through stable logical row ids and generation
metadata. That matches the current rule that GPU memory is a cache, not
durable truth.

The replay-pipelining idea maps to resident refresh. A mutation owner can
publish a WAL-safe version boundary, then let residency owners advance
partition-local indirection, compacted column groups, or old-snapshot side
structures asynchronously. Read routes should name whether they are using the
fully reconciled data generation, a replay/pending generation, or must fall
back because the gap is too large.

The RDMA/NVRAM caveat is the most important safety lesson. Any future remote
durability, CXL pool, GPUDirect storage, or GPU-initiated write path must
distinguish transfer completion, visibility to a peer, and persistence. GPU DB
must not publish SQL visibility or invalidate old resident generations merely
because DMA completed; it needs explicit durability acknowledgement tied to
the WAL boundary.

For 1M logical sessions, Query Fresh reinforces batching at commit and replay
boundaries. A large number of sessions should feed bounded WAL reservation,
group-commit, replay, and refresh queues rather than force per-session durable
actions. The useful metrics are batch bytes, commit wait, durable ack wait,
replay lag, freshness lag, and read fallback count.

**Risks and mismatches:** Query Fresh is a replicated main-memory OLTP design,
not a GPU execution engine or PostgreSQL-compatible serving layer. The reported
numbers depend on 56Gbps InfiniBand, NVRAM emulation/assumptions, tmpfs
resident data, ERMIA's redo-only logging, and TPC-C; they should not be copied
as GPU DB targets. The paper explicitly notes that RDMA-over-NVRAM persistence
is subtle and needs extra flush/fence/ack work without protocol extensions.

The design avoids deterministic logical replay by using physical redo-only
records; that can increase log bandwidth. It also still performs index work
for inserts, uses separate indirection arrays that add memory overhead, and
does not solve DDL invalidation, pgwire protocol queues, GPU cache retirement,
or multi-tenant session admission. For GPU DB, replay freshness must remain
subordinate to MVCC snapshot compatibility and WAL recovery.

**Benchmark candidates:**

- Add WAL batch telemetry that separates local durable flush time, remote/future
  tier durable ack time, mutation visibility publish time, residency
  invalidation time, and resident-refresh lag. Failure condition: a throughput
  gain hides any of these phases in one opaque commit timer.
- Prototype an append-only version-log plus logical row-id indirection sidecar
  for one generated table, with CPU indexes and GPU resident snapshots storing
  RIDs/generations rather than physical tuple offsets. Proof gate: identical
  SQL-visible results after insert/update/delete and WAL replay.
- Build a replay-lag benchmark for retained reads: hold a stream of committed
  updates, advance an indirection/replay generation asynchronously, and measure
  when reads use current retained GPU state, pending CPU state, or explicit
  fallback.
- Test group-commit sizing for COPY admission and short transactions with
  metrics for rows/sec, commit p50/p95, batch bytes, durable ack wait, and
  refresh-invalidated bytes. Failure condition: meeting rows/sec requires
  publishing visibility before WAL safety.
- Add a "DMA completion is not durability" simulator for any future remote or
  GPU-initiated storage lane: transfer-complete, peer-visible, persisted, and
  acknowledged states must be distinct in tests.
- Compare update-heavy resident refresh using physical offset references versus
  RID/generation indirection. Expected improvement: fewer index/resident
  metadata rewrites per committed update. Failure condition: extra indirection
  hurts p50 retained lookup latency more than it saves refresh work.

### 2026-06-03 - Concord approximate optimal scheduling for microsecond tails

**Citation:** Rishabh Iyer, Musa Unal, Marios Kogias, and George Candea.
"Achieving Microsecond-Scale Tail Latency Efficiently with Approximate Optimal
Scheduling." SOSP 2023. DOI: `10.1145/3600006.3613136`. Retrieved
2026-06-03 from `https://rishabh246.github.io/files/concord.pdf`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** tail latency; cooperative preemption; bounded local
queues; JBSQ; dispatcher work stealing; request classes; microsecond
scheduling; LevelDB; service-time dispersion; queueing overhead.

**Core idea:** Concord argues that microsecond runtimes can keep most of the
tail-latency benefit of theoretically optimal single-queue, preemptive
scheduling without paying the full throughput cost of implementing those
policies exactly. Instead of strict interrupt-driven preemption and a purely
synchronous single queue, it approximates the same behavior with cooperative
preemption, tiny per-worker queues, and a dispatcher that can run application
work when all workers are already busy.

The useful lesson for GPU DB is not "copy Concord as a serving runtime." It is
that strict global scheduling mechanisms can be too expensive when request
service times are measured in microseconds. A database runtime can preserve
tail goals by giving short retained reads, long scans, mutation batches, and
refresh jobs explicit scheduling points and bounded queues, while avoiding
cache-coherence and interrupt costs on every hot request.

**Concrete mechanisms:**

- Concord uses an asymmetric model with one dispatcher and pinned worker
  threads. The dispatcher has global scheduling visibility; workers own request
  execution and local queue consumption.
- Preemption notifications use a per-core dedicated cache line instead of
  inter-processor interrupts. The dispatcher writes the line when a request
  reaches its quantum; compiler-instrumented worker code polls the line,
  yields, and lets the dispatcher requeue the preempted request.
- The compiler inserts probes at function entries, around calls to
  uninstrumented code, and at loop back-edges. The paper reports about 1%
  average instrumentation overhead across benchmark suites and preemption
  timing within a small window around a 5 microsecond quantum.
- Preemption is safety-first: instrumented code avoids yielding in external
  calls, and applications can expose lock-held state so the runtime does not
  preempt inside critical sections.
- Concord replaces a purely pull-based single queue with
  Join-Bounded-Shortest-Queue. A central queue remains, but the dispatcher can
  push into tiny per-worker queues; the evaluated default is JBSQ(2), intended
  to hide dispatcher-worker communication delay without materially harming load
  balance.
- The dispatcher is work-conserving. When all per-worker queues are full, it
  runs not-yet-started requests for one quantum using a more expensive
  self-preempting instrumentation path, then resumes dispatching.
- The API is event-shaped: `setup`, `setup_worker`, and `handle_request`.
  A request is active on only one thread at a time, though preemption can move
  it between workers over its lifetime.
- Evaluation compares Concord with Shinjuku and Persephone on synthetic
  service-time distributions and LevelDB. The paper reports up to 52% higher
  microbenchmark throughput and up to 83% higher LevelDB throughput while
  meeting the same tail-latency SLOs. For the LevelDB workload, GETs were about
  600ns, PUT/DELETE about 2.3 microseconds, and SCANs about 500 microseconds.
- The main limitations are source-code/LLVM requirements and the current
  single-dispatcher design, which can bottleneck at higher core counts or very
  short service times.

**GPU DB mapping:** Concord maps directly to the planned runtime split between
network IO workers, bounded command rings, read snapshot workers, GPU
execution owners, and response rings. The first transferable idea is bounded
local queueing: a retained-read worker or GPU execution owner should be able to
hold a tiny queue of compatible work so it does not stall on every dispatcher
handoff, but queue depth must be part of the tail-latency contract rather than
an unbounded throughput knob.

The second idea is cooperative scheduling at semantic safe points. GPU DB
should not preempt arbitrary mutation, WAL, CUDA, or MVCC visibility code.
Instead, long CPU scans, refresh jobs, COPY batches, and result encoding loops
can expose explicit budget checks after safe chunks: after a WAL batch boundary,
between resident partitions, after a key-vector batch, after a kernel launch
or event wait, or after a response-buffer drain. That gives short retained
reads a way to cut in without violating WAL-before-visibility or snapshot
compatibility.

The third idea is to treat service-time dispersion as an admission signal.
Concord wins most when a workload mixes very short and very long requests.
GPU DB has exactly that shape: point lookups from retained snapshots may be
microseconds, while cold scans, refreshes, joins, and writes can occupy queues
orders of magnitude longer. Runtime telemetry should therefore track request
class, estimated service distribution, queue wait, budget-yield count, and
preemption-disabled time.

For 1M logical sessions, the dispatcher lesson is double-edged. A central
dispatcher gives useful global visibility and can apply admission policy, but
it can also become the bottleneck. GPU DB should likely use replicated
dispatcher/owner domains keyed by relation, partition, device, or session
shard, with each domain exposing a small set of comparable queue metrics to a
higher-level admission controller.

**Risks and mismatches:** Concord is an OS/runtime paper, not a database paper.
Its correctness model does not include SQL transactions, WAL durability, MVCC
visibility, DDL invalidation, CUDA stream ownership, or disk/NVMe tiering.
Compiler instrumentation also assumes source availability and compiled code;
that does not apply to arbitrary SQL expressions unless they are compiled into
engine-owned loops or templates.

GPU kernels are not cheaply preemptible in the same way as CPU code. The GPU DB
mapping should use admission, chunk sizing, kernel boundaries, stream priority,
and queue classes before assuming mid-kernel preemption. The single-dispatcher
prototype is also not a 1M-session architecture by itself; it mainly informs
per-domain scheduling mechanics and telemetry.

**Benchmark candidates:**

- Add a runtime microbenchmark with mixed retained point reads and long scans:
  compare one global read queue, tiny per-worker queues, and classed queues.
  Measure p50/p95/p99, throughput, worker idle time, queue depth, and
  head-of-line blocking.
- Instrument safe budget-yield points in CPU scan/result-encoding prototypes.
  Proof gate: short retained reads keep their p99 target while long work makes
  forward progress and SQL results remain identical.
- Test JBSQ-like depth choices for retained read workers: depth 1, 2, 4, and
  unbounded. Failure condition: deeper queues improve throughput only by
  hiding unacceptable p99 or freshness lag.
- Add service-time dispersion telemetry per route family: point lookup,
  aggregate, cold scan, refresh, COPY batch, write transaction, and result
  encode. Expected result: admission choices improve when they use classed
  dispersion rather than one global queue length.
- Simulate a work-conserving dispatcher for CPU-only query handling. Gate:
  dispatcher work improves throughput under high load without delaying
  preemption/admission decisions beyond the configured microsecond budget.
- For GPU routes, benchmark cooperative chunking at kernel boundaries rather
  than mid-kernel interruption: split a long scan into partition chunks and
  measure launch overhead, fairness, queue wait, and total latency.

### 2026-06-03 - Cross-paper synthesis: runtime queues need safe approximation boundaries

Concord, Predicate Transfer, and Query Fresh all point to the same design
track: expose explicit boundaries where expensive global work can be batched,
approximated, or shared, but make those boundaries visible enough that database
invariants still win.

The converging tracks are:

- **Safe-frontier scheduling.** Concord's cooperative yield points, Query
  Fresh's WAL/replay boundaries, and Predicate Transfer's filter-build phase
  all create places where work can pause, fan out, or replan without corrupting
  correctness. GPU DB should name these frontiers in telemetry: WAL durable,
  visibility published, resident generation valid, filter generation built,
  kernel chunk complete, response batch drained.
- **Tiny bounded queues over perfect global queues.** Concord shows that exact
  single-queue/preemption semantics can cost too much at microsecond scale.
  That reinforces bounded command/response rings and small worker-local queues,
  provided each queue reports depth, wait time, compatibility key, and tail
  impact.
- **Shared preparatory work.** Predicate Transfer builds filters once before
  the final join; Query Fresh replays shipped logs into shared fresh replicas;
  Concord amortizes dispatcher decisions across local queues. The retained GPU
  read path should look for same-shape shared work keyed by snapshot,
  partition, route family, and generation.
- **Durability and logical freshness are separate clocks.** Query Fresh makes
  this explicit, and the other two papers imply the same rule: work that is
  useful for speed must still be tagged with the visibility or route boundary
  that made it safe.

Category gaps after this batch: multi-tier placement has good coverage, but
OLTP-specific larger-than-memory eviction and explicit hot/cold transaction
movement still need more attention. Runtime scheduling is now well represented;
the next high-value pick should lean toward storage-tier policy, transaction
execution under contention, or MVCC/HTAP snapshot maintenance unless a very
recent GPU storage paper is unusually relevant.

Benchmark priorities:

- Build one mixed-service runtime benchmark with retained point reads, long
  scans, refresh work, and writes, then compare global queues, classed queues,
  and tiny per-worker queues.
- Add WAL/visibility/residency generation telemetry before optimizing commit
  batching or resident refresh; otherwise throughput wins can hide unsafe
  publication.
- Prototype shared same-shape prework for filters or retained lookup batches,
  with strict generation keys and invalidation checks.
- Measure cold-tier and refresh work as schedulable classes, not background
  noise, because they compete with the same queues needed by short sessions.

### 2026-06-03 - FastMap scalable mmap for fast storage

**Citation:** Anastasios Papagiannis, Giorgos Xanthakis, Giorgos Saloustros,
Manolis Marazakis, and Angelos Bilas. "Optimizing Memory-mapped I/O for Fast
Storage Devices." USENIX ATC 2020. Retrieved 2026-06-03 from
`https://www.usenix.org/system/files/atc20-papagiannis.pdf`.

**Category:** multi-tier cache / data placement and runtime / storage.

**Relevance tags:** mmap; page cache; fast NVMe; Optane; page faults; TLB
shootdowns; per-core metadata; dirty-page writeback; queue depth; DRAM cache;
out-of-memory execution; Silo/TPC-C; YCSB; MonetDB/TPC-H.

**Core idea:** FastMap asks whether mmap can be made viable for fast storage
once the Linux mmap path's multicore bottlenecks are removed. The paper's
answer is qualified but useful: ordinary Linux mmap does not scale well for
random page faults on modern multicore machines and fast storage, but a
specialized path for data-intensive file-backed mappings can remove central
contention, raise device queue depth, and make mapped storage competitive for
larger-than-memory workloads.

For GPU DB, the key takeaway is not "delegate cold-tier placement to mmap."
It is that if a tier uses virtual memory or mapped files, the page-fault,
reverse-mapping, eviction, writeback, and TLB-invalidation paths become hot
database runtime paths. They need the same owner-domain, per-core, bounded
queue, and telemetry discipline as query execution.

**Concrete mechanisms:**

- FastMap replaces Linux's shared `address_space` hot path with per-file data
  and per-VMA structures. A per-file structure tracks cached device blocks and
  dirty-page metadata; a per-VMA structure provides fuller reverse mappings.
- Clean and dirty metadata are separated. The all-page lookup path uses
  per-core radix trees, while dirty pages live in per-core red-black trees.
  Marking a page dirty therefore avoids updating a single tagged shared radix
  tree.
- Pages are assigned to per-core structures by page offset. This reduces
  insertion, deletion, and dirty-mark contention while preserving lock-free
  lookups where possible.
- FastMap uses fuller reverse mappings so eviction and writeback can find
  affected virtual mappings directly, rather than scanning broad VMA sets under
  coarse read locks.
- It implements a dedicated DRAM cache instead of depending on the Linux page
  cache and swapper. The cache has separate clean and dirty queues, per-core
  clean queues, per-core free lists, and a static memory buffer that does not
  create additional Linux page-cache pressure.
- Eviction prefers clean pages and evicts batches, currently 512 pages in the
  prototype, to amortize page-table manipulation and TLB invalidation.
- Writeback uses multiple threads, per-thread dirty queues, sorted dirty trees,
  and merged consecutive IO requests. The prototype begins writeback when dirty
  pages exceed 75% of cache pages.
- TLB invalidations are batched over ranges. The paper accepts some false TLB
  invalidations in exchange for fewer expensive cross-core shootdowns.
- FastMap can sit above VFS and below file systems through a stackable file
  system wrapper, or expose a virtual block-device path. Page fetch/eviction
  uses direct IO to avoid double-caching through the normal Linux page cache.
- The evaluation reports that Linux mmap scales only to about 8 threads in the
  random page-fault microbenchmark, while FastMap scales to 80 cores and
  reaches up to 11.8x more random IOPS on `null_blk`.
- On an Optane SSD, FastMap reports up to 5.27x higher throughput in the
  memory-extension graph benchmark, about 2.48x average improvement across
  out-of-memory YCSB workloads on Kreon, and very large Silo/TPC-C gains when
  Silo's heap is backed by mapped fast storage.
- The paper also shows smaller MonetDB/TPC-H gains, averaging about 6.06%,
  because the workload has more sequential access and less system-time pressure
  than the random page-fault-heavy cases.

**GPU DB mapping:** FastMap is a direct caution for the GPU DB cold and warm
tier. If cold partitions, host-compressed segments, future CXL/remote memory,
or DB-owned files are exposed through mmap-like access, Linux's default page
cache path can become the bottleneck even when NVMe or persistent memory is
fast enough. That matches the existing P8 principle that tier movement should
be explicit and observable rather than silently delegated.

The transferable design is to treat tier metadata as partitioned owner state.
GPU DB's cache manager should have separate metadata for valid resident
segments, dirty or pending refresh state, evictable clean host pages, and
durable cold extents. A single global page tree or one lock around all
resident/cold metadata would recreate the bottleneck FastMap removes. Per-core
or per-partition tier queues should report queue depth, evictions, writeback
bytes, page faults, and TLB or mapping invalidation costs.

FastMap's clean/dirty split maps well to WAL-safe GPU residency. Durable CPU
truth, clean cold pages, dirty mutation batches, invalidated GPU generations,
and refreshing resident segments should not share one undifferentiated
"buffer" state. Separate state machines make it possible to evict clean cold
segments aggressively, delay dirty writeback safely, and reject resident reads
when invalidation or refresh pressure is too high.

The device queue-depth result is important for over-resident GPU execution.
Small random storage access is no longer automatically disqualifying on fast
NVMe, but it only works when the software path can keep enough concurrent IO
in flight. GPU DB should measure cold-tier queue depth, request size, merge
rate, and CPU time per fault or read, not just storage latency.

FastMap also informs the virtual-memory-assisted buffer-manager papers already
reviewed. VM tricks can reduce explicit copy and cache lookup overhead on hits,
but misses, eviction, reverse mapping, and shootdown costs must be first-class
benchmark dimensions. For GPU DB, any VM-assisted snapshot or cold-partition
scheme should have an escape hatch to an explicit DB-owned async IO path when
fault handling steals CPU from query admission or response rings.

**Risks and mismatches:** FastMap is a Linux-kernel prototype, not a DBMS
storage engine and not a GPU data path. Its correctness model is page-cache
coherence and mapped-file persistence, not SQL transactions, WAL visibility,
MVCC snapshots, DDL invalidation, or CUDA stream ownership. The paper targets
data-intensive applications with little file sharing and infrequent `fork`;
general-purpose mmap semantics would need more memory and edge-case handling.

The strongest reported gains come from page-fault-heavy, larger-than-memory,
random-access workloads. Sequential analytics see smaller wins, and GPU DB
should not assume mmap optimization alone solves compressed scans, joins,
or resident refresh. The design also spends more metadata memory on reverse
mappings and per-core structures, batches TLB invalidations with possible
false invalidations, and depends on kernel changes that are not available in a
portable userspace engine.

**Benchmark candidates:**

- Add a cold-tier access benchmark with three paths: explicit async pread or
  io_uring, ordinary mmap, and a simulated DB-owned page cache. Measure
  p50/p95/p99, CPU system time, page-fault count, storage queue depth, request
  size, and throughput.
- Build a tier-metadata contention benchmark for P8: one global cache-manager
  lock versus per-partition/per-core queues for clean, dirty, invalidated, and
  refreshing states. Proof gate: higher concurrency without hiding invalidation
  or WAL-safe publication delays.
- Add telemetry for over-resident routes that separates page/fault wait,
  storage wait, CPU metadata time, GPU kernel time, and response encoding.
  Failure condition: a faster route cannot explain which tier phase moved.
- Test clean/dirty/invalidated separation for resident refresh: evict clean
  cold host pages, preserve dirty WAL-pending batches, and reject reads on
  invalidated generations. Gate: no stale SQL-visible retained read under
  concurrent mutation and eviction.
- Measure batched mapping invalidation if any VM-assisted host snapshot path is
  introduced. Expected benefit: fewer shootdown-like events; failure condition:
  false invalidations harm short retained-read p99 more than the batching helps.
- Compare cold random lookups at 4KB, 16KB, 64KB, and merged request sizes.
  Expected result: a tier route needs enough queue depth and merge rate before
  GPU execution can hide storage latency.

### 2026-06-03 - Tiered-Indexing hot-record migration for skewed access methods

**Citation:** Xinjing Zhou, Xiangpeng Hao, Xiangyao Yu, and Michael
Stonebraker. "Tiered-Indexing: Optimizing Access Methods for Skew." The VLDB
Journal 34, article 45, 2025. Retrieved 2026-06-03 from
`https://doi.org/10.1007/s00778-025-00928-6`.

**Category:** multi-tier cache / data placement and transaction access methods.

**Relevance tags:** skew; hot/cold records; buffer-managed indexes; record
migration; B+tree; hash table; heap file; LSM-tree; range scans; update
amplification; optimistic lock coupling; LeanStore; RocksDB; YCSB.

**Core idea:** Tiered-Indexing argues that ordinary page-granular buffer
management wastes memory under skew because a single hot record can pin a page
full of cold records. Instead of adding a separate record cache, the paper
keeps one buffer-pool budget and decomposes an access method into hot and cold
tiers. Records migrate between tiers according to hotness, while both tiers
remain page-based structures that can use conventional logging and recovery.

The key transferable point is that hot/cold placement should be part of the
access method, not an unrelated cache beside it. A point-lookup-heavy GPU DB
table can waste GPU HBM, host DRAM, and NVMe bandwidth if it caches whole
partitions or pages merely because a few records are hot. The paper's design
suggests a middle ground between "cache the full resident partition" and
"build a read-only exact record cache": promote hot records or key ranges into
an access-method-owned hot tier that still supports writes, scans, and recovery
through normal storage-engine boundaries.

**Concrete mechanisms:**

- A Tiered-Indexing structure maintains a hierarchy of index structures with
  different hotness levels. Each tier supports the same operations as the
  original one-tier structure, and records move between tiers.
- For 2-tier designs, the buffer pool is logically split into hot and cold
  regions. The paper's implementation uses LeanStore and configures 90% of
  frames for the hot region and 10% for the cold region, while noting that this
  split is not universally optimal.
- Hot-tier records carry small migration metadata: reference, dirty, and
  deletion bits. Accesses set reference state; pressure on the hot tier
  triggers downward migration of colder records.
- 2-Hash uses hot and cold hash tables. The paper discusses shared versus
  independent hash functions; shared hashing keeps the corresponding hot and
  cold records in related key space, while independent hashing can change
  migration and collision behavior.
- 2-Heap uses two heaps, each with its own indexes. Downward migration can
  scan the heap directly or use an indexed heap scan. The indexed strategy
  orders evictions by key so maintenance of the cold heap's index is more
  sequential.
- 2B+tree keeps hot and cold B+trees. Eviction walks the hot tree in key order
  with an approximate clock-style policy, selecting records whose reference bit
  is clear and inserting them into nearby cold-tree leaf ranges.
- BiLSM-tree extends RocksDB with upward migration on point reads. It monitors
  block-cache miss rate and average point-read depth, then adapts migration
  sampling rates to move hot records upward without turning every miss into
  eager migration.
- The implementation uses optimistic locking with a 64K-entry array of 64-bit
  version numbers keyed by record hash. Writers use atomic compare-and-swap;
  the design accepts possible false conflicts when distinct records map to the
  same version slot.
- The paper emphasizes that migration changes physical representation, not
  logical user content, so page-based physiological WAL and ARIES-style
  recovery can still apply to the proposed page-based tiers.
- Experiments use YCSB over 100 million records, 8-byte keys, 120-byte payloads,
  16KB pages, direct IO, no Linux page cache, and mostly Zipfian skew. At small
  memory budgets, 2B+tree reports up to 9.8x over a one-tier B+tree and 12.3x
  over TreeLine during loading, with much lower insert IO amplification.
- The paper reports wider gains on update-heavy YCSB-F than read-only YCSB-C
  because page-granular designs pay for both poor memory utilization and
  read/evict traffic when updating disk pages.
- For LSM range scans, row-cache-style record caching can be much worse because
  the row cache cannot serve broad range queries; BiLSM-tree keeps block-cache
  usefulness while moving hot records in the tree hierarchy.

**GPU DB mapping:** GPU DB currently treats GPU resident table/partition state
as an explicit versioned performance cache backed by WAL and CPU truth.
Tiered-Indexing suggests adding a finer hot-record or hot-key tier underneath
that same correctness model. For skewed point lookups, a full resident
partition may be overkill; a hot-key resident tier keyed by table, partition,
snapshot generation, and predicate shape could keep only the records that
drive most traffic in HBM, while colder records remain in host or NVMe-backed
structures.

The design also maps to CPU host indexes. The P8 storage track should avoid a
separate read-only record cache that bypasses WAL, MVCC, or range-query
semantics. A better first experiment is an access-method-owned hot tier whose
entries carry MVCC visibility boundaries and invalidation generation. Hot
records can be promoted after observed reads and demoted under pressure, but
the mutation owner still controls WAL-before-visibility and publishes new
readable generations.

For GPU execution, hot-tier placement creates a natural micro-batch key:
snapshot generation plus hot-key tier id plus query shape. Same-shape point
lookups that hit the hot tier can be gathered into a compact key vector, while
misses fall back to the cold partition path. The benchmark should measure
whether this improves p99 without starving cold scans or update refresh.

The paper's migration lesson is also useful for host/NVMe tiering. Downward
migration should produce storage-friendly access patterns, not a stream of
random record writes caused by a generic LRU list. GPU DB demotion from HBM to
host, or host to NVMe, should prefer partition/key-order batches and report IO
amplification, merge rate, and invalidation cost.

**Risks and mismatches:** Tiered-Indexing is an access-method paper, not a GPU
database design. It does not handle CUDA memory ownership, kernel launch
amortization, GPU-resident columnar layouts, pgwire response rings, SQL joins,
or full MVCC visibility on GPU. The evaluation disables WAL and the Linux page
cache to isolate buffer-pool behavior, so durable commit and recovery costs are
not part of the main throughput numbers.

The hot/cold split is not free. Migration metadata, version-slot conflicts,
background migration workers, and hot-tier invalidation can add latency or
contention. A GPU DB hot-record tier also risks duplicating data across HBM,
host DRAM, and cold storage unless residency budgets are explicit. Finally,
skew changes over time; aggressive upward migration may create thrash, while
lazy migration may miss short-lived hot sets.

**Benchmark candidates:**

- Add a skewed retained-lookup benchmark with Zipf factors 0.7, 0.9, and 0.96:
  compare full resident partition, hot-key resident tier, host-index fallback,
  and cold partition path. Measure p50/p95/p99, HBM bytes, hit rate, queue
  wait, and invalidation count.
- Prototype hot-key promotion keyed by table OID, partition id, snapshot
  generation, key column, and query shape. Proof gate: a promoted entry never
  survives a mutation or DDL invalidation beyond its visibility boundary.
- Measure demotion policy: random LRU demotion versus key-order or
  partition-order demotion. Expected result: ordered demotion lowers NVMe or
  host write amplification and refresh churn under skew.
- Test lazy versus eager promotion sampling for hot lookups. Failure condition:
  eager promotion improves average throughput but worsens p99 or blocks
  mutation/refresh owners under distribution shifts.
- Add an update-heavy skew workload that alternates hot point reads and hot-key
  writes. Gate: hot-tier correctness preserves WAL-before-visibility and no
  retained read observes stale values after mutation publication.
- Compare range scans with and without a hot-record tier. The tier must not
  steal so much memory from column/partition pages that scans regress more than
  point lookups improve.

### 2026-06-03 - Carousel time-indexed shaping for bounded session admission

**Citation:** Ahmed Saeed, Nandita Dukkipati, Vytautas Valancius, Vinh The
Lam, Carlo Contavalli, and Amin Vahdat. "Carousel: Scalable Traffic Shaping at
End Hosts." SIGCOMM 2017. Retrieved 2026-06-03 from
`https://saeed.github.io/files/carousel-sigcomm17.pdf`; DOI
`https://doi.org/10.1145/3098822.3098852`.

**Category:** runtime / HFT / session scale.

**Relevance tags:** traffic shaping; rate limiting; pacing; timing wheel;
deferred completions; backpressure; per-core ownership; lock-free coordination;
response rings; session admission; incast; high connection count.

**Core idea:** Carousel replaces per-flow or per-class token-bucket queues with
a single time-indexed queue per CPU core. Each packet receives an earliest
release timestamp from pacing and rate-limit policies, then the shaper releases
due packets from a timing wheel. The key system lesson is that scalable
admission is not just a rate formula; it also needs bounded queued work,
backpressure to the producer, and ownership that avoids shared hot locks.

For GPU DB, the strongest transferable idea is to shape work by release time
and resource budget at the narrow boundary where saturation occurs: network
egress, response encoding, mutation admission, read-snapshot execution, or GPU
kernel dispatch. A million logical sessions cannot each own an unbounded queue
or thread. They need small per-owner schedulers that can pace accepted work,
delay completions, and expose overload before CPU memory, socket buffers, or
GPU staging buffers fill.

**Concrete mechanisms:**

- Carousel computes packet timestamps from one or more policies. Each policy
  advances a latest timestamp by packet length divided by the policy rate; the
  final release time is the maximum timestamp so the packet does not violate
  any active policy.
- Packets are inserted into a timing wheel: a circular array of time slots,
  where each slot holds a FIFO list of packet references due in that time
  range. Insert and extract are O(1) for the intended shaping workload.
- The wheel is configured by slot granularity and horizon. The paper gives an
  example with 8 microsecond granularity and a 4 second horizon, yielding
  500K slots for 1.5 Mbps minimum-rate support with 1500 byte packets.
- Packets with release times beyond the horizon can either be placed at the
  last slot, allowing temporary overshoot, or dropped when a hard limit is
  required.
- Carousel avoids per-packet allocation by using a preallocated global pool of
  nodes. Each node can hold references to multiple packets, amortizing node
  movement and avoiding `std::list` allocation cost.
- Deferred Completions hold the producer completion signal until the shaped
  packet actually leaves. This bounds the number of packets in the shaper and
  pushes back through existing transport mechanisms instead of buffering or
  dropping large backlogs.
- The implementation supports out-of-order completions because shaped release
  order can differ from arrival order. A driver-side map tracks outstanding
  packets so completion can follow actual departure rather than original order.
- Carousel stores packet references, not packet payloads. The paper reports
  roughly 8 MB of shaper memory for one million outstanding packet references.
- Scaling across cores uses one independent timing wheel per core. Connections
  hash to a core-local shaper for lock-free enqueue/dequeue on the data path.
- Shared aggregate rates across cores are handled by a NIC-level bandwidth
  allocator that periodically redistributes rates using water filling. Updates
  are lazy, around 100 ms in the implementation, to avoid locking each packet.
- Receiver-side ingress shaping is possible by pacing acknowledgements rather
  than buffering incoming data packets. The receiver emits ACK progress at the
  configured rate to control sender behavior during incast.
- In microbenchmarks, Carousel's timing-wheel overhead with the global pool is
  reported around 11-12 ns per packet, versus 21-22 ns with `std::list`
  slots, and is insensitive to the number of packets held.
- Production video-serving experiments across 25 servers report about 6.4%
  median and 8.2% 90th-percentile improvement in Gbps/CPU versus Linux
  FQ/pacing, with similar retransmission rates. The paper attributes this to
  lower networking CPU, larger batching, and lower shaping overhead.

**GPU DB mapping:** The direct mapping is a time-wheel-like admission lane for
high-concurrency request and response scheduling. The runtime already calls for
bounded ingress, read snapshot, mutation, GPU execution, and response rings.
Carousel adds a concrete policy: assign admitted work a release time or earliest
service time based on per-class budgets, then drain due work from an O(1)
time-indexed queue owned by one IO or execution worker.

The response path is the cleanest first target. When many sessions produce
same-shape retained results, response writes can burst and inflate socket
buffers even if GPU execution is cheap. A per-IO-worker response timing wheel
could pace large responses, COPY acknowledgements, or client classes while
holding request credits until bytes are actually accepted by the socket. That
mirrors Deferred Completions: do not free a session's request credit merely
because the engine produced a row buffer; free it when the response lane has
made observable progress.

The mutation path can use the same idea at chunk boundaries. COPY or INSERT
admission should publish credits only after WAL-safe chunks have been appended,
invalidated, and made eligible for visibility. If the WAL, MVCC index, or
resident invalidation lane is behind, producers should see bounded backpressure
instead of building unlimited pending chunks.

For GPU execution, Carousel argues against one queue per session or one lock
around all retained reads. GPU DB can hash compatible retained work to
partition/device/shape owners, pace work by queue depth and latency budget, and
rebalance only through periodic aggregate budget updates. Shared fairness does
not need per-request global locking.

The ACK-shaping idea maps to logical-session admission. A protocol worker can
delay readiness or request-credit advancement for sessions that are over budget
instead of accepting more SQL messages and buffering them in memory. That is
especially relevant for 1M logical sessions, where per-session memory must be
nearly constant and small.

**Risks and mismatches:** Carousel is a packet shaper, not a database runtime.
It has no SQL transaction semantics, WAL, MVCC visibility, query cancellation,
or GPU stream ownership. Its timestamp consolidation works for pacing and rate
limits, but the paper explicitly says Carousel is not a generic scheduler; strict
priority or preemptive scheduling needs different machinery.

The production evaluation is video egress, not request/response OLTP. GPU DB
must test whether time-slot granularity and horizon choices harm p50 query
latency or create unfairness between tiny point lookups and large result sets.
Deferred completions also require careful protocol integration: freeing a credit
on engine completion instead of network write completion would lose the
backpressure benefit, while freeing it too late could underutilize the engine.

The per-core aggregate-rate rebalancer uses lazy updates around 100 ms, which is
probably too slow for some microsecond-scale query lanes. GPU DB should treat
that as a WAN/video-serving choice, not a fixed constant. Finally, a timing
wheel can bunch work at slot boundaries; if many retained reads become due in
one slot, the runtime still needs batch-size and latency ceilings.

**Benchmark candidates:**

- Build a response-ring admission benchmark with three modes: immediate credit
  release on engine result, socket-write completion credit release, and
  time-wheel-paced socket-write completion. Gate: lower p99 memory and queue
  depth without reducing correct result throughput under many slow clients.
- Add a synthetic 1M logical-session harness with a small active subset and
  many idle sessions. Measure per-session bytes, IO-worker queue depth,
  response backlog, and p50/p99 for retained point reads.
- Prototype a per-IO-worker timing wheel for response chunks with 4, 8, 16, and
  32 microsecond slots. Failure condition: slot batching increases p50 or p99
  for tiny retained reads more than it reduces CPU or memory pressure.
- Test deferred request credits: a session may send another request only after
  the prior response is accepted by the response lane or an explicit pipeline
  credit is returned. Gate: no unbounded pending SQL messages under slow-client
  or overload conditions.
- Compare one global admission queue, per-session queues, and per-owner
  time-indexed queues for retained lookup bursts. Expected result: per-owner
  queues preserve throughput while avoiding global lock and per-session memory
  growth.
- Add overload telemetry for each shaped lane: released items, delayed items,
  horizon drops/rejections, slot occupancy, credit wait, and bytes held. A
  route is not acceptable unless it can name the saturated lane.

### 2026-06-03 - Cross-paper synthesis: hot placement still needs paced fronts

FastMap, Tiered-Indexing, and Carousel point at the same production rule from
different layers: fast hot paths fail when slow-path resources are allowed to
accumulate invisibly. FastMap partitions page-cache metadata and writeback so
fast storage does not collapse under global kernel locks. Tiered-Indexing moves
hot records into access-method-owned tiers so skew does not waste page and
buffer budgets. Carousel shapes packet release and producer credits so many
flows do not turn efficient batching into memory blowup.

For GPU DB, the converging design track is **budgeted owner fronts**. Every
hot placement choice should have a matching admission front: HBM hot-key tiers
need promotion/demotion budgets, cold NVMe partitions need IO queue budgets,
and response rings need socket/protocol credit budgets. A resident route is not
just "valid or invalid"; it should be valid, admitted, and paced by the owner
that can see the scarce resource.

The current category gap is still the write/MVCC boundary between these fronts.
The queue has many strong tiering and runtime papers, but the next few reviews
should keep pulling from logging, recovery, transaction-cache, and snapshot-GC
work so paced hot reads do not outrun WAL-before-visibility or version cleanup.

Benchmark priorities:

- Pair every hot-tier experiment with an overload test: slow clients, cold
  misses, invalidation storms, and memory pressure should produce named
  rejection or delay reasons, not hidden queue growth.
- Add one end-to-end "paced retained read" benchmark that reports GPU queue,
  response queue, socket credit, HBM residency, and cold-tier wait separately.
- Add one "skew plus mutation" benchmark where hot-key promotion competes with
  WAL-safe invalidation and response pacing. The proof gate is no stale read,
  bounded memory, and an explainable p99.
- Track category balance by selecting the next paper from transaction logging,
  MVCC GC, or runtime scheduling rather than another pure GPU-OLAP paper unless
  the queue demands it.

### 2026-06-03 - PACMAN parallel command-log recovery

**Citation:** Yingjun Wu, Wentian Guo, Chee-Yong Chan, and Kian-Lee
Tan. "Fast Failure Recovery for Main-Memory DBMSs on Multicores."
SIGMOD 2017, pp. 267-281. doi:10.1145/3035918.3064011. Retrieved
2026-06-03 from `https://yingjunwu.github.io/papers/sigmod2017.pdf`.

**Category:** transaction processing / write path.

**Relevance tags:** command logging; WAL replay; failure recovery;
checkpointing; stored procedures; dependency analysis; recovery scheduling;
parallel replay; owner domains; post-crash warmup.

**Core idea:** PACMAN attacks the usual command-log tradeoff in main-memory
OLTP systems. Tuple-level logging can replay in parallel but creates large
runtime log volume; transaction-level or command logging records only a stored
procedure id and parameter values, keeping the steady-state write path light
but making crash replay look serial. PACMAN keeps the cheap command log and
recovers parallelism by analyzing stored procedures ahead of time and using
logged parameter values during recovery.

The paper models recovery as ordered data-flow over a pre-crash commit order.
Compile-time analysis decomposes stored procedures into dependency-respecting
slices and integrates all procedures into a global dependency graph. Recovery
then instantiates that graph for each log batch, uses runtime parameter values
to discover which pieces really touch disjoint keys, and pipelines multiple
log batches so later independent pieces can start before an earlier batch is
fully replayed. In the TPC-C evaluation on a 40-core Peloton setup, serial
command-log replay takes over 4,200 seconds after a 5-minute run, while the
parallel command-log recovery variant is reported as 18x faster; with dynamic
intra- and inter-batch parallelism, PACMAN's recovery time drops below
300 seconds with 40 threads.

**Concrete mechanisms:**

- Transaction-level log records store the invoked stored-procedure identifier
  and input parameter values. Log entries are grouped into ordered log batches;
  batches are reloaded and processed in durable commit order.
- PACMAN's static intra-procedure analysis extracts flow dependencies
  including define-use and control dependencies, plus data dependencies where
  operations touch the same table and at least one operation writes.
- Stored procedures are split into slices. Mutually data-dependent operations
  remain in the same slice, and flow-dependent spans keep the intervening
  operations needed to preserve local ordering.
- Each procedure receives a local dependency graph whose nodes are slices and
  whose directed edges represent must-happen-before ordering.
- Inter-procedure analysis merges slices from different procedures into a
  global dependency graph when they may conflict. The resulting graph captures
  ordering across procedure families, not just within one transaction template.
- During recovery, each log-batch entry becomes transaction pieces
  instantiated from the global graph. Pieces belonging to the same graph block
  form a piece-set ordered by the transaction order in the batch.
- To avoid excessive fine-grained synchronization, PACMAN initially coordinates
  execution at the piece-set level instead of waking children after every piece.
- Dynamic intra-batch analysis uses parameter values from log entries and from
  already replayed pieces to identify concrete key spaces. Operations or pieces
  inside a piece-set can run in parallel when they touch disjoint tuples and
  have no flow dependency.
- The paper calls out common read-modify-write and foreign-key access patterns
  as cases where parameter-aware dynamic analysis can recover parallelism that
  conservative static table-level dependency analysis hides.
- Inter-batch pipelining lets a piece-set in a later log batch begin once its
  dependent piece-sets in the same batch and the same block in the preceding
  batch have completed, avoiding a full barrier between batches.
- Recovery cores are assigned to graph blocks by estimating workload
  distribution while reloading logs. Work inside each block can then be
  dispatched across those assigned cores using dynamic conflict checks.
- Ad-hoc or nondeterministic transactions fall back to tuple-level logical
  logging. PACMAN treats their replay as write-only transaction pieces with
  known write sets, preserving generality but losing some command-log benefit.
- The evaluation reports that tuple-level recovery scales only up to a point
  because recovery threads need latches on modified tuples; PACMAN schedules
  replay order ahead of time and avoids those recovery latches.
- In PACMAN's time breakdown at 40 threads, thread scheduling becomes the
  dominant residual cost, around 30% of recovery time, while log data loading
  and dynamic analysis are reported as lightweight.

**GPU DB mapping:** This is directly relevant to GPU DB's WAL-before-visibility
contract. The current engine treats WAL/checkpoint/archive replay as the
durable truth and GPU resident state as rebuildable acceleration. PACMAN
suggests that the write path can keep a compact logical or command-oriented
record for selected deterministic mutation classes, while replay can still be
parallel if the engine records enough route metadata to reconstruct dependency
graphs.

The natural GPU DB unit is not an arbitrary SQL string. It is a bounded command
shape owned by a mutation or partition owner: COPY chunk admission, point
insert, deterministic update family, resident invalidation, index append, or
refresh publication. Each shape can declare read/write key spaces, partition
id, table generation, visibility boundary, and side effects. Recovery can then
instantiate a dependency graph over those shapes and replay independent pieces
across CPU partition owners while GPU-resident caches remain invalid until a
verified rebuild/warmup phase.

PACMAN also argues for separating runtime logging cost from recovery scheduling
cost. GPU DB should not bloat the hot write path with tuple-level copies merely
to make recovery easy. A better benchmark is a hybrid log: minimal WAL records
for deterministic owner commands, plus fallback logical or physical payloads
for ad-hoc SQL, nondeterministic operations, and complex updates. Recovery can
name which parts replay through command graphs and which parts replay through
row-level records.

The dynamic parameter analysis maps to retained and partitioned layouts. A
batch of logged updates may all target one table, but concrete keys or
partition ids can still be disjoint. Recovery should exploit that disjointness
for CPU truth rebuild and for regenerating derived indexes, statistics, and
resident invalidation generations. Only after CPU truth and visibility
boundaries are reconstructed should GPU execution owners rebuild or publish
resident snapshots.

Finally, PACMAN's scheduling bottleneck is a warning for the 1M-session runtime:
recovery is another high-concurrency scheduler. If post-crash replay uses one
central ready queue, the scheduling layer can become the bottleneck even when
log loading and conflict analysis are cheap. Recovery should probably reuse
the production owner-ring structure with block/partition-local work queues and
explicit progress telemetry.

**Risks and mismatches:** PACMAN depends on stored procedures or similarly
deterministic templates. GPU DB currently accepts general SQL, so command-log
recovery would need a narrow admitted set and a conservative fallback path.
The static analysis assumes read and write sets are easy to compute; complex
predicates, secondary-index scans, text predicates, user functions, and
nondeterministic SQL may not fit.

The paper evaluates CPU main-memory recovery in Peloton, not GPU-resident
rebuild, NVMe-tiered storage, or pgwire session recovery. It also focuses on
replaying committed effects after a checkpoint, not on distributed consensus,
replication, or WAL flush latency. Absolute recovery times from a 2017 40-core
machine should not be treated as current performance targets. The transferable
claim is the dependency/scheduling shape, not the hardware numbers.

**Benchmark candidates:**

- Add a deterministic command-log replay prototype for one narrow mutation
  shape, such as COPY chunk append into a single table/partition. Gate: replay
  reconstructs identical CPU truth, indexes, visibility generations, and
  resident invalidation state as tuple-level WAL replay.
- Compare three recovery logs for the same workload: tuple-level physical/logical
  records, compact command records, and hybrid command-plus-fallback records.
  Measure steady-state write throughput, WAL bytes, recovery time, and
  correctness after crash injection.
- Build a recovery dependency graph keyed by table id, partition id, command
  shape, and concrete key range. Gate: disjoint partitions replay in parallel;
  conflicting updates preserve commit-order visibility.
- Add a post-crash warmup benchmark: replay CPU truth first, then rebuild GPU
  resident snapshots by partition. Failure condition: any query can observe a
  resident route before its source WAL boundary and invalidation generation are
  proven.
- Measure scheduler overhead during replay with one global ready queue versus
  per-owner recovery rings. Expected result: owner-local queues reduce
  scheduling contention at high replay parallelism.
- Test ad-hoc fallback ratio by mixing deterministic COPY/INSERT commands with
  complex UPDATE/DELETE statements. Gate: increasing fallback share degrades
  recovery predictably without invalidating command-log correctness.
- Expose recovery telemetry: log bytes loaded, command records replayed,
  fallback records replayed, dependency blocks, ready queue depth, conflict
  stalls, CPU truth rebuild time, resident rebuild time, and first-safe-query
  timestamp.

### 2026-06-03 - Bounded multiversion garbage collection

**Citation:** Yuanhao Wei, Guy E. Blelloch, Panagiota Fatourou, and
Eric Ruppert. "Practically and Theoretically Efficient Garbage Collection for
Multiversioning." arXiv:2212.13557v2, 2023. Retrieved 2026-06-03 from
`https://arxiv.org/abs/2212.13557`.

**Category:** MVCC / snapshot / visibility.

**Relevance tags:** multiversion garbage collection; long read-only
transactions; version chains; range tracking; epoch reclamation; lock-free
lists; retained snapshots; bounded memory; update/read tradeoff.

**Core idea:** The paper compares multiversion garbage-collection schemes in a
single experimental setting and then introduces two practical collectors,
DL-RT and SL-RT, that combine range tracking with simpler concurrent version
list structures. The problem is the familiar MVCC failure mode: a long
read-only transaction can keep an epoch open, while frequent updates create
many obsolete intermediate versions that are not needed by any active reader
but still remain reachable.

Epoch-based reclamation is simple and fast, but it only reclaims old tail
versions and can leave obsolete versions in the middle of a chain. Periodic or
update-triggered compaction can remove middle versions, but may scan lists that
contain little garbage and lacks strong worst-case space bounds. The paper's
range-tracking variants identify which timestamp intervals are still protected
by active read-only transactions, then remove versions whose intervals are no
longer needed.

The result is not a universal winner. EBR and an optimized Steam variant often
have the best update throughput on friendly workloads, but EBR can use up to
10x more memory under long read transactions or oversubscription, and Steam can
show high space use in hierarchical multiversion structures. SL-RT and DL-RT
are closer to EBR/Steam throughput than the earlier theoretically bounded
BBF+ implementation while preserving predictable space behavior.

**Concrete mechanisms:**

- Each object keeps a timestamp-sorted version list. A read-only transaction
  announces its timestamp and reads the newest version whose timestamp is not
  greater than that read timestamp.
- A version with timestamp `t1` followed by a newer version at `t2` is needed
  only if an active read-only transaction timestamp falls in `[t1, t2)`. The
  latest version is always needed.
- EBR advances global epochs and safely reclaims versions overwritten before
  old epochs, but it does not remove obsolete versions that are trapped between
  still-needed versions in the middle of a list.
- Compaction-based schemes read announced timestamps, sort them, and traverse
  version lists to remove versions whose valid intervals contain no active
  announcement.
- Range tracking, inherited from BBF+, records non-current version intervals
  and identifies obsolete versions more directly than periodically scanning all
  lists or compacting every updated list.
- DL-RT uses a practical doubly linked list, PDL, so identified middle versions
  can be removed from their local neighborhood. It relaxes the constant
  amortized-time machinery of BBF+'s TreeDL because long consecutive removal
  chains were rare in the experiments.
- SL-RT uses a simple singly linked list, SSL, and compacts by traversing a
  list. This can be faster and smaller when version chains are short, because
  it avoids back pointers and extra pointer updates.
- The paper also implements a lock-free Steam variant using the same SSL list
  structure, reducing the cost of Steam's older list-level locking.
- The theoretical bound for DL-RT and SL-RT keeps reachable versions within a
  constant factor of the maximum number of needed versions, plus terms tied to
  process count and list count rather than unbounded obsolete chains.
- The evaluation applies all collectors to the same multiversion balanced tree
  and multiversion hash table, isolating collector effects from broader DBMS
  concurrency-control differences.
- Workloads vary data-structure size, update/read mix, read-only transaction
  size, thread count, oversubscription, and Zipf skew.
- Long read-only transactions and oversubscription are the adverse cases:
  epochs are delayed, update-heavy paths create many versions, and EBR's
  retained old versions inflate space.
- The authors report that in most experiments SL-RT has the best space
  behavior; EBR reaches more than an order of magnitude more memory in one
  adverse hash-table case.
- Throughput effects are workload dependent. Maintaining range tracking costs
  extra update work, but shorter version lists can improve read-only
  transaction traversal, so mixed throughput does not have a single winner.
- The paper's experiments rely on Java GC to reclaim unlinked nodes; the MVGC
  schemes decide when versions become unreachable from version lists, not how
  allocator-level reclamation is implemented.

**GPU DB mapping:** GPU DB needs version and retained-snapshot cleanup to be
bounded by active readers, not by hope that old epochs eventually clear. A
future retained-read runtime may have many short point reads, some long
analytical or refresh reads, and possibly a large number of idle logical
sessions. EBR alone is too blunt if one long read or stalled session keeps
obsolete row versions, resident generations, or visibility summaries alive
across a heavy write burst.

The transferable design is to expose the same interval rule in GPU DB's MVCC
metadata: each CPU tuple version, resident partition generation, and derived
index generation should have a begin/end visibility interval, and the runtime
should know which read timestamps or snapshot generations are actively held.
Garbage collection can then reclaim versions that no active reader can still
observe, including middle versions, instead of waiting for all older snapshots
to drain.

Range tracking also maps to retained GPU snapshots. A resident partition
generation should not be kept merely because it is older than the newest
generation; it should be kept only if an admitted read has a compatible
snapshot handle. That suggests a common "active snapshot interval" service
shared by CPU MVCC chains, resident metadata, and refresh/invalidation
retirement. Publication remains WAL-before-visibility: GC can remove
unneeded old state only after newer visible state is safely published and no
active reader can name the old interval.

SL-RT is probably the first implementation shape to benchmark. GPU DB's first
MVCC chains and resident generation lists should usually be short if mutation
batching and partition invalidation work. A singly linked list plus explicit
compaction at generation or update boundaries may be faster and smaller than a
fully general doubly linked version chain. DL-RT becomes interesting if hot
rows or hot partitions accumulate long chains and middle-version removal
without full traversal becomes necessary.

The paper also argues for adversarial GC benchmarks, not only normal-case
throughput. GPU DB should test long retained reads, slow clients holding
snapshot handles, oversubscribed workers, skewed hot keys, and write bursts.
The pass condition is bounded retained bytes and predictable collector work,
not only high query throughput when all readers are short.

**Risks and mismatches:** This is a concurrent-data-structure paper rather
than a full DBMS MVCC system. It does not cover SQL isolation levels, WAL,
crash recovery, durable undo, DDL, vacuum-visible indexes, distributed
transactions, GPU memory, or disk/NVMe tiers. Its read-only transactions are
range queries over multiversion trees or hash tables, not arbitrary SQL plans.

The evaluation uses Java and relies on automatic memory management after
versions are unlinked. GPU DB will need explicit allocator, pinned-buffer,
host-memory, and device-memory retirement. A version being unreachable from a
CPU list is not enough if a CUDA stream, response encoder, or protocol worker
still holds a buffer reference.

Range tracking adds update-path metadata work. If every small write must update
a global tracking object, the collector could become another owner-thread
bottleneck. GPU DB should shard tracking by table/partition/owner and treat a
single global active-timestamp array only as a baseline. Also, exact interval
tracking for every row may be too expensive; coarse partition-generation
tracking may be the right first proof even if row-level chains use a simpler
epoch fallback.

**Benchmark candidates:**

- Implement a synthetic MVCC-chain benchmark with three collectors: EBR-only
  tail reclamation, update-triggered full-chain compaction, and range-tracked
  middle-version reclamation. Gate: under one long retained read plus hot-key
  updates, range tracking keeps retained versions bounded while preserving
  correct snapshot reads.
- Add retained-generation GC for GPU resident partitions: publish generations,
  hold/release snapshot handles, invalidate on writes, and reclaim old
  generations only when no active handle can observe them. Failure condition:
  stale resident reads, premature buffer free, or unbounded generation growth.
- Compare SL-style list compaction versus DL-style local removal for hot
  row-version chains and resident partition generation lists. Measure update
  cost, read traversal length, retained bytes, and collector CPU time.
- Add an oversubscription test where one worker holding a snapshot is delayed
  while many updates run. Gate: bounded memory and explicit telemetry naming
  the protected interval and blocked reclamation reason.
- Track MVCC GC telemetry per owner: active snapshot count, oldest active
  timestamp, protected interval count, versions removed from tails, versions
  removed from middles, retained bytes, collector queue depth, and failed
  reclaim reasons.
- Test coarse partition-generation range tracking before row-level exact
  tracking. Expected result: a coarse first version catches most retained
  snapshot memory blowups with lower write-path metadata cost.
