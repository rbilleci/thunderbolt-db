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
