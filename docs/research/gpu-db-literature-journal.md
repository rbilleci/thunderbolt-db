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
