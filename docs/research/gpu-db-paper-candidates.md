# GPU DB Paper Candidates

This queue is maintained by the literature-review loop. Status values:

- `queued`: candidate identified but not processed
- `reviewed`: journal entry exists
- `skipped`: unavailable, weak relevance, or superseded

Selection policy: future reviews must use papers from 2015 onward, preferably
newer literature from 2023-present. Pre-2015 entries may remain as historical
context, but should not be selected by the loop.

Balance policy: transaction-processing use cases are first-class. The loop
should not process more than two analytics/GPU-OLAP papers consecutively. If
recent reviews skew analytical, the next candidate should come from
transaction processing, MVCC/snapshots, runtime/session scale, HFT-style
mechanical sympathy, multi-tier cache/data placement, or query optimization.
Database file-system design for storage/indexing and WAL throughput are
explicit priority lanes. When adding or selecting storage papers, prefer sources
that expose concrete file layout, index metadata, WAL/checkpoint ordering,
read/write throughput, recovery, compaction, or tail-latency mechanisms.

## Research Search Terms

### GC, reclamation, and in-memory DB state

Use these search terms to expand the GC lane beyond ordinary MVCC cleanup and
evaluate where modern GC ideas matter inside the DB engine:

- `"moving garbage collector" database engine stable handles`
- `"compacting garbage collection" in-memory database`
- `"concurrent compacting garbage collector" "read barrier" "write barrier"`
- `"generational garbage collection" "region" "database"`
- `"region-based memory management" "query execution" database`
- `"arena allocation" "query execution" database engine`
- `"epoch-based reclamation" "in-memory database" MVCC`
- `"hazard pointers" "database index" memory reclamation`
- `"RCU" "database system" "metadata" reclamation`
- `"multiversion garbage collection" "database" "bounded memory"`
- `"MVCC garbage collection" "long-running transactions" HTAP`
- `"old version reclamation" "snapshot isolation" "in-memory"`
- `"moving collector" "stable handles" "object relocation"`
- `"object relocation" "handle table" "database"`
- `"copying collector" "persistent data structures" database`
- `"priority garbage collection" software caches database`
- `"cache-aware garbage collection" "database buffer"`
- `"memory pressure" "query cache" garbage collection`
- `"DBMS buffer management" "garbage collection" "compaction"`
- `"log-structured storage" "garbage collection" tail latency`
- `"value log garbage collection" "LSM" "latency"`
- `"blob garbage collection" database storage engine`
- `"checkpoint garbage collection" "write-ahead log" database`
- `"catalog version garbage collection" database`
- `"plan cache" "garbage collection" database`
- `"GPU memory management" "database" "garbage collection"`
- `"pinned memory" "garbage collection" "GPU" database`
- `"Bower" "garbage collection" database`
- `"Bower" "moving collector" runtime`
- `"BOHM" MVCC garbage collection database`
- `"Boehm" "conservative garbage collection" database engine`

Where this matters most for GPU DB, in priority order:

1. MVCC version chains and retained snapshot cleanup, because long readers and
   GPU-resident snapshots can turn correctness into unbounded memory growth.
2. Route/catalog/plan publication metadata, because stale descriptors can
   accumulate under high session counts and must be reclaimed without blocking
   readers.
3. Query temporary arenas and result buffers, because they are high-churn and
   should be reclaimed or recycled with near-zero coordination overhead.
4. Resident CPU/GPU snapshot state, pinned host buffers, and command rings,
   because movement/compaction pressure here directly affects latency and
   memory residency.
5. Warm/cold cache and storage segments, because eviction, demotion,
   compaction, and value-log cleanup are the storage-engine analogues of
   prioritized or moving GC.
6. WAL/checkpoint/replay side state, because safe reclamation depends on
   durability horizons, replica acknowledgement, and recovery guarantees.

## Seed Queue

### Transaction processing, write path, and concurrency control

- `reviewed` — **TicToc: Time Traveling Optimistic Concurrency Control**,
  Yu et al., SIGMOD 2016.
  URL: `https://dl.acm.org/doi/10.1145/2882903.2882935`
  PDF: `https://db.cs.cmu.edu/papers/2016/yu-sigmod2016.pdf`
  Why: timestamp-based optimistic concurrency control for high-throughput
  transaction processing; useful for comparing MVCC/snapshot timestamp choices.
- `reviewed` — **Cicada: Dependably Fast Multi-Core In-Memory Transactions**,
  Lim et al., SIGMOD 2017.
  URL: `https://dl.acm.org/doi/10.1145/3035918.3064015`
  Why: high-throughput multicore transaction processing with concurrency
  control, versioning, and contention management tradeoffs.
- `reviewed` — **ERMIA: Fast Memory-Optimized Database System for Heterogeneous
  Workloads**, Kim et al., SIGMOD 2016.
  URL: `https://dl.acm.org/doi/10.1145/2882903.2882905`
  Why: memory-optimized transactional engine for mixed workloads; relevant to
  balancing read snapshots and write throughput.
- `reviewed` — **Transaction Repair for Multi-Version Concurrency Control**,
  Dashti et al., SIGMOD 2017.
  URL: `https://dl.acm.org/doi/10.1145/3035918.3035919`
  Preprint: `https://arxiv.org/abs/1603.00542`
  Why: MVCC transaction repair approach that may inform conflict handling
  without throwing away all work. The previously queued arXiv 2024 URL was
  unrelated and has been corrected to the SIGMOD 2017 paper.
- `reviewed` — **GPU-Accelerated OLTP: An In-Depth Analysis of Concurrency
  Control Schemes**, Sun et al., arXiv 2024 / ICDE 2026.
  URL: `https://arxiv.org/abs/2406.10158`
  PDF: `https://arxiv.org/pdf/2406.10158`
  Why: modern GPU OLTP concurrency-control testbed comparing OCC, MVCC,
  timestamp ordering, locking, and GPU conflict-ordering schemes under YCSB
  and TPC-C; selected after recent reviews skewed storage/indexing and the
  queue lacked a strong modern transaction/GPU-OLTP candidate. Journal entry
  added 2026-06-06.
- `reviewed` — **GaccO: A GPU-Accelerated OLTP DBMS**, Boeschen and Binnig,
  SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526123`
  Why: GPU-Accelerated OLTP identifies GaccO as the strongest high-contention
  GPU conflict-ordering path; useful for studying batch preprocessing,
  all-access conflict treatment, and GPU transaction route admission.
  Journal entry already exists; this stale seed-queue duplicate was corrected
  from `queued` to `reviewed` on 2026-06-06.
- `reviewed` — **LTPG: Large-Batch Transaction Processing on GPUs with
  Deterministic Concurrency Control**, Wei et al., ICDE 2024.
  URL: `https://doi.org/10.1109/ICDE60146.2024.00296`
  PDF:
  `https://vbn.aau.dk/ws/portalfiles/portal/821323666/New_LTPG.pdf`
  Why: modern deterministic large-batch GPU transaction processing cited by
  the GPU OLTP survey; useful for comparing conflict-ordered GPU batches with
  CPU-owned WAL/MVCC publication. Journal entry already exists; this stale
  seed-queue duplicate was corrected from `queued` to `reviewed` and its DOI
  was fixed on 2026-06-06.
- `reviewed` — **PLOR: General Transactions with Predictable, Low Tail
  Latency**, Chen et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517879`
  PDF: `https://storage.cs.tsinghua.edu.cn/papers/sigmod22plor.pdf`
  Why: GPU-Accelerated OLTP cites PLOR as a low-tail transaction design;
  useful for mapping predictable transaction execution and latency control to
  GPU DB's hot-key admission and session SLOs. Journal entry added
  2026-06-06; the previously queued DOI was corrected to the SIGMOD 2022
  paper metadata.
- `reviewed` — **Mostly-Optimistic Concurrency Control for Highly Contended
  Dynamic Workloads on a Thousand Cores**, Wang and Kimura, PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol10/p49-wang.pdf`
  Why: cited by GPU-Accelerated OLTP as a contention-oriented CPU-side
  concurrency-control baseline; useful for comparing lightweight optimistic
  fallback against GPU conflict ordering under hot keys. Journal entry already
  exists; this stale seed-queue duplicate was corrected from `queued` to
  `reviewed` on 2026-06-06.
- `reviewed` — **Improving Optimistic Concurrency Control through Transaction
  Batching and Operation Reordering**, Ding, Kot, and Gehrke, PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol12/p169-ding.pdf`
  DOI: `https://doi.org/10.14778/3282495.3282502`
  Why: PLOR contrasts batching/reordering as a throughput and tail-latency
  direction for OCC; useful for comparing route-level reordering against
  GPU DB's owner rings, priority retry budgets, and hot-key write admission.
  Journal entry added 2026-06-06.
- `reviewed` — **Polyjuice: High-Performance Transactions via Learned
  Concurrency Control**, Wang et al., OSDI 2021.
  URL: `https://www.usenix.org/conference/osdi21/presentation/wang-jiachen`
  PDF: `https://www.usenix.org/system/files/osdi21-wang-jiachen.pdf`
  Why: PLOR contrasts modular/learned concurrency-control choices with
  protocol-internal priority; useful for deciding whether GPU DB should learn
  route policies while keeping hot-path correctness and tail-priority rules
  explicit. Journal entry added 2026-06-06.
- `reviewed` — **Deferred Runtime Pipelining for Contentious Multicore Software
  Transactions**, Mu, Angel, and Shasha, EuroSys 2019.
  URL: `https://doi.org/10.1145/3302424.3303966`
  PDF: `https://www.cis.upenn.edu/~sga001/papers/drp-eurosys19.pdf`
  Why: PLOR contrasts runtime pipelining and transaction chopping with
  priority-based low-tail conflict handling; useful for evaluating whether GPU
  DB hot operations should pipeline sub-steps through owner queues without
  requiring static read/write sets. Journal entry already exists; this stale
  duplicate was corrected from `queued` to `reviewed` on 2026-06-06.

### MVCC, snapshots, and visibility

- `reviewed` — **Scalable and Robust Snapshot Isolation for High-Performance
  Storage Engines**, PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p1426-alhomssi.pdf`
  Why: scalable snapshot isolation, long-reader robustness, and GC ideas.
- `reviewed` — **Read-Safe Snapshots: An abort/wait-free serializable read
  method for read-only transactions on mixed OLTP/OLAP workloads**, Information
  Systems 2024.
  URL: `https://www.sciencedirect.com/science/article/pii/S0306437924000437`
  Why: recent MVCC read-only transaction design for serializable snapshots
  under mixed OLTP/OLAP workloads.
- `reviewed` — **On Supporting Efficient Snapshot Isolation for Hybrid
  Workloads with Multi-Versioned Indexes**, PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol13/p211-sun.pdf`
  Why: P-Tree index for efficient snapshot isolation and MVCC in multicore
  in-memory HTAP storage.
- `reviewed` — **An Empirical Evaluation of In-Memory Multi-Version Concurrency
  Control**, PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p781-Wu.pdf`
  Why: MVCC design tradeoffs, version storage, validation, and GC behavior.
- `reviewed` — **Accelerating Analytical Processing in MVCC using Fine-Granular
  High-Frequency Virtual Snapshotting**, arXiv 2017.
  URL: `https://arxiv.org/abs/1709.04284`
  Why: HTAP-style analytical snapshots without blocking write progress.
- `reviewed` — **MD-MVCC: Multi-version Concurrency Control for Schema Changes
  in Azure SQL Database**, Antonopoulos et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4791-antonopoulos.pdf`
  DOI: `https://doi.org/10.14778/3750601.3750605`
  Why: modern production metadata-MVCC design selected after recent reviews
  called for more MVCC/snapshot work; relevant to versioned route metadata,
  DDL/read concurrency, snapshot-safe plan reuse, and route-metadata GC.
- `reviewed` — **Scalable Garbage Collection for In-Memory MVCC Systems**,
  Boettcher, Leis, Neumann, and Kemper, PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol13/p128-bottcher.pdf`
  DOI: `https://doi.org/10.14778/3364324.3364328`
  Why: selected after the recent journal called for more MVCC garbage
  collection work and the queue lacked a strong queued candidate in that lane;
  useful for bounding hot version chains under long retained CPU/GPU read
  snapshots with exact active-generation pruning. Journal entry added
  2026-06-07.
- `reviewed` — **Practically and Theoretically Efficient Garbage Collection for
  Multiversioning**, Blelloch et al., arXiv 2022.
  URL: `https://arxiv.org/abs/2212.13557`
  Why: discovered while searching for modern MVCC garbage-collection
  follow-ups; useful for checking whether newer multiversion GC algorithms can
  provide bounded retired-memory guarantees for retained snapshots beyond
  Steam's in-memory DBMS implementation. Stale duplicate corrected on
  2026-06-07; journal entry already exists under the PPoPP 2023 paper
  metadata.

### Runtime scale, HFT-style mechanics, and admission

- `reviewed` — **OLTP Through the Looking Glass 16 Years Later:
  Communication is the New Bottleneck**, Zhou et al., CIDR 2025.
  URL: `https://vldb.org/cidrdb/2025/oltp-through-the-looking-glass-16-years-later-communication-is-the-new-bottleneck.html`
  PDF: `https://vldb.org/cidrdb/papers/2025/p17-zhou.pdf`
  Why: whole-stack OLTP breakdown showing communication, isolation, and
  client/server round trips as modern bottlenecks; directly relevant to pgwire,
  session multiplexing, and stored-procedure versus interactive transaction
  route choices.
- `reviewed` — **Shenango: Achieving High CPU Efficiency for Latency-sensitive
  Datacenter Workloads**, NSDI 2019.
  URL: `https://www.usenix.org/conference/nsdi19/presentation/ousterhout`
  Why: user-level scheduling and CPU allocation for latency-sensitive services;
  relevant to multiplexed IO and query workers under 1M logical sessions.
- `reviewed` — **Caladan: Mitigating Interference at Microsecond Timescales**,
  OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/fried`
  Why: runtime scheduling and resource allocation for microsecond-scale tail
  latency, useful for admission and worker ownership design.
- `reviewed` — **Demikernel: An Operating System Architecture for
  Microsecond-scale Datacenter Systems**, SOSP 2021.
  URL: `https://dl.acm.org/doi/10.1145/3477132.3483569`
  PDF: `https://irenezhang.net/papers/demikernel-sosp21.pdf`
  Why: low-latency OS/network stack architecture relevant to session and
  response-ring design.
- `skipped` — **Design Choices in Low-Latency C++ Systems: Empirical Insights
  With Applications to High-Frequency Trading**, SSRN 2026.
  URL: `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6513601`
  Why: modern HFT-oriented low-latency systems survey; useful for queue,
  allocation, cache, and thread-pinning patterns. Skipped in this cron worker
  because both the SSRN landing page and Delivery PDF were blocked by a
  Cloudflare challenge on 2026-06-03.

### Multi-tier cache, buffer management, and data placement

- `reviewed` — **LeanStore: In-Memory Data Management beyond Main Memory**,
  Leis et al., ICDE 2018.
  URL: `https://doi.org/10.1109/ICDE.2018.00026`
  Metadata:
  `https://portal.fis.tum.de/en/publications/leanstore-in-memory-data-management-beyond-main-memory`
  Why: low-overhead storage manager that keeps in-memory performance for hot
  data while transparently handling SSD-resident data; directly relevant to
  GPU/DRAM/NVMe tiering and transactional working sets.
- `reviewed` — **LeanStore: A High-Performance Storage Engine for NVMe SSDs**,
  Leis, PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p4536-leis.pdf`
  DOI: `https://doi.org/10.14778/3685800.3685915`
  Why: modern synthesis of LeanStore's NVMe-optimized OLTP storage engine,
  including virtual-memory-assisted caching, write-aware replacement, IO
  scheduling, MVCC, logging, checkpointing, and recovery; directly relevant to
  GPU DB's CPU/NVMe cold-tier and recovery design.
- `reviewed` — **Umbra: A Disk-Based System with In-Memory Performance**,
  Neumann and Freitag, CIDR 2020.
  URL: `https://www.vldb.org/cidrdb/papers/2020/p29-neumann-cidr20.pdf`
  Why: variable-size pages and low-overhead buffer management for cached hot
  working sets with graceful uncached access; useful for resident snapshot and
  host/NVMe tier design.
- `reviewed` — **Are You Sure You Want to Use MMAP in Your Database Management
  System?**, Crotty et al., CIDR 2022.
  URL: `https://www.cidrdb.org/cidr2022/papers/p13-crotty.pdf`
  Why: evaluates OS page-cache and mmap tradeoffs versus explicit DBMS buffer
  management; important for deciding whether tier movement should be explicit
  or delegated to the OS.
- `reviewed` — **Virtual-Memory Assisted Buffer Management**, Leis et al.,
  SIGMOD/PACMMOD 2023.
  URL: `https://tore.tuhh.de/entities/publication/f82ebf12-6f97-4161-8ad6-d1e94645e33a`
  Why: combines DBMS buffer management with virtual-memory mechanisms for fast
  storage and multicore CPUs; relevant to host-memory tier policy and fault
  telemetry.
- `reviewed` — **Multi-Tier Buffer Management and Storage System Design for
  Non-Volatile Memory**, arXiv 2019.
  URL: `https://arxiv.org/abs/1901.10938`
  Why: explicit multi-tier DBMS buffer design across DRAM and non-volatile
  storage; useful for promotion/demotion policy and tier-aware page layout.
- `reviewed` — **Efficient Compactions Between Storage Tiers with PrismDB**,
  arXiv 2020.
  URL: `https://arxiv.org/abs/2008.02352`
  Why: multi-tier storage compaction across fast and slow devices; relevant to
  cold/warm segment organization and write amplification when data moves
  between tiers.

### Query optimizers, planning, and route choice

- `reviewed` — **Lero: A Learning-to-Rank Query Optimizer**, arXiv 2023.
  URL: `https://arxiv.org/abs/2302.06873`
  Why: learned ranking layered on native optimizers; relevant to route choice
  without replacing deterministic planner rules.
- `reviewed` — **AutoSteer: Learned Query Optimization for Any SQL Database**,
  PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p3515-anneser.pdf`
  Why: learned tuning of optimizer knobs for existing SQL systems; relevant to
  GPU route knobs and fallback decisions.
- `reviewed` — **Rethinking Learned Cost Models: Why Start from Scratch?**,
  SIGMOD 2023.
  URL: `https://15799.courses.cs.cmu.edu/spring2025/papers/15-learned/yang-sigmod2023.pdf`
  DOI: `https://doi.org/10.1145/3626769`
  Why: learned cost-model calibration rather than full replacement; useful for
  CPU/GPU route estimation.
- `reviewed` — **Robust Plan Evaluation based on Approximate Probabilistic
  Machine Learning**, Kamali et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p2626-kamali.pdf`
  arXiv: `https://arxiv.org/abs/2401.15210`
  Why: risk-aware optimization may map to choosing CPU/GPU/overload/fallback
  routes under uncertain latency.

### GPU query execution and analytics

- `reviewed` — **Concurrent Analytical Query Processing with GPUs**,
  Wang et al., PVLDB 2014.
  URL: `https://www.vldb.org/pvldb/vol7/p1011-wang.pdf`
  Why: directly relevant to concurrent GPU query scheduling and resource
  sharing. Reviewed before the 2015-present policy was added; keep as
  historical context.
- `reviewed` — **Concurrent query processing in a GPU-based database system**,
  PLOS ONE 2019.
  URL: `https://pmc.ncbi.nlm.nih.gov/articles/PMC6467383/`
  Why: batch-level optimization model for concurrent GPU database workloads.
- `reviewed` — **Data Path Fusion in GPU for Analytical Query Processing**,
  arXiv 2026.
  URL: `https://arxiv.org/abs/2605.10511`
  Why: modern GPU-driven data path fusion that combines IO, decompression, and
  query work into GPU execution.
- `reviewed` — **RTCUDB: Building Databases with RT Processors**, arXiv 2024.
  URL: `https://arxiv.org/abs/2412.09337`
  Why: explores ray-tracing cores for database query processing and may suggest
  alternate hardware mapping for lookup/search-heavy paths.
- `reviewed` — **GOLAP: A GPU-in-Data-Path Architecture for High-Speed OLAP**,
  2024.
  URL: `https://doi.org/10.1145/3698812`
  PDF: `https://www.dfki.de/fileadmin/user_upload/import/16459_3698812.pdf`
  Why: GPU-in-data-path design for compressed block streaming, decompression,
  and scan processing.
- `reviewed` — **Revisiting Query Performance in GPU Database Systems**,
  arXiv 2023.
  URL: `https://arxiv.org/abs/2302.00734`
  Why: cross-stack GPU DBMS performance, resource utilization, and concurrent
  query recommendations.
- `reviewed` — **Efficiently Processing Joins and Grouped Aggregations on GPUs**,
  Wu, Koutsoukos, and Alonso, SIGMOD/PACMMOD 2025.
  URL: `https://arxiv.org/abs/2312.00720`
  DOI: `https://doi.org/10.1145/3709689`
  Why: modern evaluation of GPU joins, grouped aggregation, and workload-aware
  implementation selection.
- `skipped` — **Red Fox: An Execution Environment for Relational Query
  Processing on GPUs**, 2013.
  URL:
  `https://casl.gatech.edu/publications/red-fox-an-execution-environment-for-relational-query-processing-on-gpus/`
  Why: pre-2015; keep only as historical context.
- `skipped` — **GPU Join Processing Revisited**, DaMoN 2012.
  URL: `https://research.ibm.com/publications/gpu-join-processing-revisited`
  Why: pre-2015; keep only as historical context.

### Historical context, skipped by date policy

- `skipped` — **High-Performance Concurrency Control Mechanisms for Main-Memory
  Databases**, VLDB 2012.
  URL: `https://www.vldb.org/pvldb/vol5/p298_per-akelarson_vldb2012.pdf`
  Why: pre-2015; keep only as historical context.
- `skipped` — **Serializable Snapshot Isolation in PostgreSQL**, VLDB 2012.
  URL: `https://www.vldb.org/pvldb/vol5/p1850_danports_vldb2012.pdf`
  Why: pre-2015; keep only as historical context.
- `skipped` — **Disruptor: High performance alternative to bounded queues for
  exchanging data between concurrent threads**, LMAX technical paper.
  URL: `https://lmax-exchange.github.io/disruptor/files/Disruptor-1.0.pdf`
  Why: pre-2015/non-paper background; keep only as historical context.
- `skipped` — **The C10K problem**, Dan Kegel.
  URL: `http://www.kegel.com/c10k.html`
  Why: pre-2015/non-paper background; keep only as historical context.

## Newly Discovered Queue

Append new candidates here as each paper is processed.

- `reviewed` — **Laser: Buffer-Aware Learned Query Scheduling in
  Master-Standby Databases**, Huang and Li, PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol18/p743-li.pdf`
  DOI: `https://doi.org/10.14778/3712221.3712239`
  Code: `https://github.com/hyw498169842/LASER`
  Why: modern buffer-aware learned query scheduling discovered because the
  current queued locality/scheduling candidates skewed older; useful for
  routing retained reads by physical residency footprint, buffer/GPU locality,
  and load balance. Journal entry added 2026-06-06.
- `reviewed` — **LSched: A Workload-Aware Learned Query Scheduler for Analytical
  Database Systems**, Sabek, Ukyab, and Kraska, SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526158`
  PDF: `https://people.csail.mit.edu/ibrahimsabek/pdf/22_paper_lsched.pdf`
  Why: Laser cites LSched as learned query scheduling related work; useful for
  comparing reinforcement-learning or workload-aware scheduling with explicit
  route-footprint scheduling for GPU DB. Journal entry added 2026-06-06.
- `reviewed` — **Self-Tuning Query Scheduling for Analytical Workloads**,
  Wagner, Kohn, and Neumann, SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457260`
  Why: Laser cites it as single-server analytical scheduling work; useful for
  deciding whether GPU DB's read runtime should tune queue order by observed
  latency, cache locality, and short-query priority. Journal entry added
  2026-06-06; the DOI was corrected to `10.1145/3448016.3457260`.
- `reviewed` — **Buffer Pool Aware Query Scheduling via Deep Reinforcement
  Learning**, Zhang, Marcus, Kleiman, and Papaemmanouil, arXiv 2020.
  URL: `https://arxiv.org/abs/2007.10568`
  Why: Laser cites it as buffer-pool-aware scheduling; useful for contrasting
  learned buffer reuse against deterministic residency metadata and route
  certificates. Journal entry added 2026-06-06 from arXiv v3.
- `reviewed` — **SkinnerDB: Regret-Bounded Query Evaluation via Reinforcement
  Learning**, Trummer et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p2074-trummer.pdf`
  DOI: `https://doi.org/10.14778/3229863.3236263`
  Why: SmartQueue cites SkinnerDB as reinforcement-learning query execution
  work; useful for contrasting queue-level cache-aware scheduling with
  intra-query adaptive join-order switching and regret-bound-driven route
  exploration. Journal entry added 2026-06-06 from the arXiv/SIGMOD 2019
  full paper after the VLDB PDF endpoint timed out from the cron worker.
- `reviewed` — **Quickstep: A Data Platform Based on the Scaling-up Approach**,
  Patel et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p663-patel.pdf`
  Why: LSched is implemented on Quickstep's block/work-order execution model;
  useful for extracting morsel/work-order, scheduler, and resource-estimation
  mechanisms that can inform GPU DB operator fragments and route-feature
  telemetry. Journal entry added 2026-06-06.
- `reviewed` — **Learning Scheduling Algorithms for Data Processing Clusters**,
  Mao et al., SIGCOMM 2019.
  URL: `https://doi.org/10.1145/3341302.3342080`
  Code: `https://github.com/hongzimao/decima-sim`
  Why: LSched contrasts Decima's black-box DAG scheduling with DB-specific
  physical-plan features; useful as a control point for what should remain
  outside GPU DB's hot scheduler when learned policies are evaluated. Journal
  entry added 2026-06-07 from the author PDF.
- `queued` — **Firmament: Fast, Centralized Cluster Scheduling at Scale**,
  Gog, Schwarzkopf, Gleave, Watson, and Hand, OSDI 2016.
  URL: `https://www.usenix.org/conference/osdi16/technical-sessions/presentation/gog`
  PDF: `https://pdos.csail.mit.edu/papers/firmament:osdi16.pdf`
  Why: Decima contrasts centralized and distributed cluster schedulers;
  Firmament is a modern scalable centralized scheduler useful for comparing
  flow-network placement, scheduler latency, and global admission decisions
  against GPU DB's local owner rings and route-advisor policy snapshots.
- `queued` — **Graphene: Packing and Dependency-Aware Scheduling for
  Data-Parallel Clusters**, Grandl et al., OSDI 2016.
  URL: `https://www.usenix.org/conference/osdi16/technical-sessions/presentation/grandl`
  PDF:
  `https://www.usenix.org/system/files/conference/osdi16/osdi16-grandl-graphene.pdf`
  Why: Decima uses Graphene-style DAG-aware scheduling as a baseline; useful
  for comparing deterministic troublesome-node and packing heuristics with
  learned route-DAG scheduling before GPU DB adds any learned admission
  advisor.
- `queued` — **TetriSched: Global Rescheduling with Adaptive Plan-ahead in
  Dynamic Heterogeneous Clusters**, Tumanov et al., EuroSys 2016.
  URL: `https://doi.org/10.1145/2901318.2901355`
  PDF: `https://www.cs.cmu.edu/~harchol/Papers/EUROSYS16.pdf`
  Why: Decima and Graphene both point at plan-ahead scheduling; TetriSched is
  useful for comparing reservation-aware choices between preferred GPU-like
  resources and fallback resources under deadlines and mis-estimated runtimes.
- `reviewed` — **Simple Adaptive Query Processing vs. Learned Query
  Optimizers: Observations and Analysis**, Zhang et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p2962-zhang.pdf`
  DOI: `https://doi.org/10.14778/3611479.3611500`
  Why: discovered while reviewing SkinnerDB; modern comparison of simple
  adaptive query processing and learned optimizers, useful for deciding when
  GPU DB route choice should rely on deterministic adaptive probes instead of
  a learned policy. Journal entry added 2026-06-06 from the open-access VLDB
  Journal 2025 extended paper after the VLDB PDF endpoint timed out from the
  cron worker.
- `reviewed` — **Lemo: A Cache-Enhanced Learned Optimizer for Concurrent
  Queries**, Mo et al., SIGMOD/PACMMOD 2024.
  URL: `https://dl.acm.org/doi/10.1145/3626734`
  Author page: `https://mlxdb.github.io/publication/sigmod24-lemo/`
  Why: the adaptive-query-processing paper identifies Lemo as a newer learned
  optimizer for concurrent queries; useful for comparing learned cache sharing
  and multi-query route selection with deterministic batch-owner filter reuse
  and GPU residency telemetry. Journal entry added 2026-06-06 from the
  author/project page and indexed ACM/PACMMOD metadata after direct ACM PDF
  fetches returned Cloudflare 403 pages; the queued DOI was corrected from
  `10.1145/3654972` to `10.1145/3626734`.
- `reviewed` — **LIMAO: A Framework for Lifelong Modular Learned Query
  Optimization**, Zhang et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4546-zhang.pdf`
  Why: discovered while checking newer learned-optimizer follow-ups; useful
  for contrasting reusable learned sub-plan knowledge with GPU DB's explicit
  route descriptors, adaptive thresholds, and no-training fallback path.
  Journal entry added 2026-06-06 from the arXiv PDF after the VLDB PDF
  endpoint stalled from the cron worker.
- `queued` — **SkinnerDB: Regret-bounded Query Evaluation via Reinforcement
  Learning**, Trummer et al., ACM TODS 2021.
  URL: `https://doi.org/10.1145/3464389`
  Open PDF: `https://par.nsf.gov/servlets/purl/10377793`
  Why: extended journal version discovered while reviewing the PVLDB/SIGMOD
  SkinnerDB line; useful if the loop needs deeper formal and implementation
  details for intra-query learning, progress tracking, and specialized
  execution-engine support.
- `queued` — **HybridQO: Hybrid Learned Query Optimizer**, Zhu et al.,
  CIDR 2022.
  URL: `https://www.cidrdb.org/cidr2022/papers/p16-hilprecht.pdf`
  Why: LIMAO contrasts prior dynamic-environment learned optimizers; useful
  for comparing learned query optimization under workload, data, and schema
  shifts with GPU DB's deterministic route eligibility and adaptive route
  scoring.
- `reviewed` — **LEON: A New Framework for ML-Aided Query Optimization**,
  Chen et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p2261-chen.pdf`
  Why: LIMAO cites LEON among learned optimizer systems that adapt to changing
  data; useful for comparing learned plan search or cost feedback against
  modular lifelong route-cost learning. Journal entry added 2026-06-06.
- `reviewed` — **Eraser: Eliminating Performance Regression on Learned Query
  Optimizer**, Weng et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p926-zhu.pdf`
  DOI: `https://doi.org/10.14778/3641204.3641205`
  Why: LEON emphasizes stability and bounded regression for ML-aided
  optimizers; Eraser is a modern follow-up for checking guardrails,
  regression detection, and fallback strategies before GPU DB trusts learned
  route scoring in production. Journal entry added 2026-06-06; the queued URL
  and DOI were corrected to the PVLDB PDF metadata.
- `reviewed` — **RankPQO: Learning-to-Rank for Parametric Query Optimization**,
  Mo et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p863-mo.pdf`
  Why: LEON's pairwise ranking objective is relevant to repeated same-shape
  routes; RankPQO may inform parameter-sensitive plan or route caching for
  pgwire prepared statements, retained lookups, and GPU/CPU fallback choices.
  Journal entry already exists; this stale duplicate was corrected from
  `queued` to `reviewed` on 2026-06-06.
- `reviewed` — **Learned Query Optimizer: What is New and What is Next**,
  Zhu, Weng, Ding, and Zhou, SIGMOD Companion 2024.
  URL: `https://doi.org/10.1145/3626246.3654692`
  Author PDF: `https://bolinding.github.io/papers/sigmod24learnedqo.pdf`
  Why: Eraser's authors cite this tutorial as broader learned-optimizer
  deployment context; useful for checking which LQO pieces are mature enough
  for GPU DB route scoring and which should remain guarded by deterministic
  eligibility and fallback rules. Journal entry added 2026-06-07 from the
  author PDF and DOI metadata.
- `reviewed` — **PilotScope: Steering Databases with Machine Learning Drivers**,
  Zhu et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p980-zhu.pdf`
  DOI: `https://doi.org/10.14778/3641204.3641209`
  Code: `https://github.com/alibaba/pilotscope`
  Why: Learned Query Optimizer highlights PilotScope as a deployment bridge for
  ML drivers that push/pull plans, hints, cardinalities, and runtime data
  through database-specific interactors; useful for designing GPU DB route
  advisor hooks without putting Python or model lifecycle work on the hot path.
  Journal entry added 2026-06-07 from the PVLDB PDF and author mirror after
  the first direct PVLDB curl stalled.
- `reviewed` — **The Holon Approach for Simultaneously Tuning Multiple
  Components in a Self-Driving Database Management System with Machine
  Learning via Synthesized Proto-Actions**, Zhang, Lim, Butrovich, and
  Pavlo, PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p3373-zhang.pdf`
  DOI: `https://doi.org/10.14778/3681954.3682007`
  Code: `https://github.com/17zhangw/protox`
  Why: PilotScope shows that multiple AI4DB drivers can interact badly; Holon
  is a modern follow-up for coordinating knobs, hints, indexes, and other route
  actions as joint policy units before GPU DB combines route scoring, cache
  admission, and scheduler advisors. Journal entry added 2026-06-07 from the
  primary VLDB PDF; direct `curl` to VLDB timed out, but the browser fetch path
  retrieved the indexed open-access PDF.
- `queued` — **A Unified and Efficient Coordinating Framework for Autonomous
  DBMS Tuning**, Zeng et al., arXiv 2023.
  URL: `https://arxiv.org/abs/2303.05710`
  Why: Holon contrasts multi-tuner coordinator approaches that tune individual
  DBMS components through separate agents; useful as a follow-up for deciding
  whether GPU DB route scoring, cache admission, and scheduler policy should
  coordinate through a shared action model or remain separate advisors.
- `queued` — **LlamaTune: Sample-Efficient DBMS Configuration Tuning**,
  Kanellis et al., VLDB 2022.
  URL:
  `https://www.microsoft.com/en-us/research/publication/llamatune-sample-efficient-dbms-configuration-tuning/`
  Why: Holon uses sample-efficiency and limited tuning budgets as deployment
  constraints; useful for comparing low-sample knob/advisor tuning against GPU
  DB's need to learn route thresholds without spending many production-like
  benchmark hours.
- `queued` — **Cardinality Estimation in DBMS: A Comprehensive Benchmark
  Evaluation**, Han et al., PVLDB 2021.
  URL: `https://kai-zeng.github.io/papers/benchmark_vldb_2021.pdf`
  DOI: `https://doi.org/10.14778/3503585.3503586`
  Why: PilotScope uses learned cardinality drivers and STATS-CEB in its
  evaluation; this benchmark paper is useful for deciding which cardinality
  telemetry and end-to-end route-regret measurements GPU DB should collect
  before trusting learned CPU/GPU route advice.
- `queued` — **QueryFormer: A Tree Transformer Model for Query Plan
  Representation**, Zhao, Cong, Shi, and Miao, PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p1658-zhao.pdf`
  Why: Learned Query Optimizer cites plan-embedding models as reusable inputs
  for cost estimation and other optimizer tasks; useful for comparing compact
  plan-shape embeddings against deterministic route-family keys for CPU/GPU
  cost and latency prediction.
- `queued` — **Adaptive Concurrent Query Execution Framework for an
  Analytical In-Memory Database System**, Deshmukh, Memisoglu, and Patel,
  IEEE BigData Congress 2017.
  URL: `https://doi.org/10.1109/BIGDATACONGRESS.2017.38`
  Why: Quickstep cites this as the scheduling framework behind its elastic
  concurrent query execution; useful for a deeper look at policy-enforced
  work-order scheduling, query suspension, and priority/resource allocation
  before GPU DB implements retained-route fragment scheduling.
- `queued` — **ByteSlice: Pushing the Envelope of Main Memory Data
  Processing with a New Storage Layout**, Feng, Lo, Kao, and Xu, SIGMOD
  2015.
  URL: `https://doi.org/10.1145/2723372.2747642`
  Why: Quickstep cites ByteSlice among modern vectorized/block-oriented
  execution directions; useful for comparing bit-sliced CPU layouts with
  GPU DB's encoded resident `int4` columns and early predicate pruning.
- `reviewed` — **Sundial: Harmonizing Concurrency Control and Caching in a
  Distributed OLTP Database Management System**, Yu et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p1289-yu.pdf`
  DOI: `https://doi.org/10.14778/3231751.3231763`
  Why: Polyjuice contrasts fixed hybrid CC choices with learned policies, and
  Sundial combines logical leases, distributed transaction concurrency control,
  and cache coherence; useful for GPU DB's snapshot leases, retained-route
  cache coherence, and multi-tier read/write admission. Journal entry added
  2026-06-06.
- `reviewed` — **No Compromises: Distributed Transactions with Consistency,
  Availability, and Performance**, Dragojevic et al., SOSP 2015.
  URL:
  `https://www.usenix.org/conference/sosp15/technical-sessions/presentation/dragojevic`
  Why: Sundial cites this as a hardware/network-assisted distributed
  transaction baseline; useful for contrasting low-latency remote access,
  RDMA-shaped transaction execution, and consistency guarantees with GPU DB's
  future warm/cold-tier and replicated-owner paths. Journal entry added
  2026-06-06 from the SIGOPS SOSP 2015 PDF after the stale USENIX URL
  redirected to a 404 page.
- `reviewed` — **Fast In-memory Transaction Processing using RDMA and HTM**,
  Wei, Shi, Chen, Chen, and Chen, SOSP 2015.
  URL: `https://doi.org/10.1145/2815400.2815419`
  PDF:
  `https://sigops.org/sosp/sosp15/current/2015-Monterey/printable/158-wei.pdf`
  Why: FaRM's SOSP cohort paper explores a different hardware-assisted OLTP
  path by combining RDMA with hardware transactional memory; useful for
  comparing owner-mediated validation and WAL publication against HTM-assisted
  local concurrency control and remote access. Journal entry added
  2026-06-06.
- `reviewed` — **DHTM: Durable Hardware Transactional Memory**, Joshi,
  Nagarajan, Cintra, and Viglas, ISCA 2018.
  URL: `https://doi.org/10.1109/ISCA.2018.00045`
  PDF: `https://www.pure.ed.ac.uk/ws/portalfiles/portal/59203973/DHTM.pdf`
  Why: DrTM relies on HTM plus separate NVRAM logging for durability; DHTM is a
  hardware-oriented follow-up for comparing whether durable transactional
  memory ideas can simplify or bound future CPU warm-tier route metadata,
  undo/redo records, and crash-consistent descriptor updates. Journal entry
  added 2026-06-06; the queued DOI was corrected from `.00047` to `.00045`.
- `queued` — **RHKV: An RDMA and HTM friendly key-value store for
  data-intensive computing**, Shi et al., Future Generation Computer Systems
  2019.
  DOI: `https://doi.org/10.1016/j.future.2018.10.001`
  Why: DrTM's location-cache and HTM/RDMA hash-table mechanisms motivate a
  narrower key-value follow-up; useful for comparing address-only caches,
  incarnation validation, and remote write support before GPU DB caches
  resident or cold-tier route locations.
- `reviewed` — **Everything is a Transaction: Unifying Logical Concurrency
  Control and Physical Data Structure Maintenance in Database Management
  Systems**, Pavlo et al., CIDR 2021.
  URL: `https://www.cidrdb.org/cidr2021/papers/cidr2021_paper06.pdf`
  Author PDF: `https://www.pdl.cmu.edu/PDL-FTP/Database/zhang-CIDR21.pdf`
  Why: discovered while following Polyjuice's related-work line around
  logical concurrency and physical maintenance; useful for folding index
  refresh, resident-cache invalidation, and cold-tier maintenance into
  transactional visibility instead of treating them as detached background
  jobs. Journal entry already exists; this stale duplicate was corrected from
  `queued` to `reviewed` on 2026-06-06.
- `reviewed` — **DUMBO: Making durable read-only transactions fly on hardware
  transactional memory**, Barreto et al., arXiv 2024.
  URL: `https://arxiv.org/abs/2410.16110`
  Why: DHTM focuses on durable update transactions; DUMBO is a modern durable
  HTM follow-up for read-only transactions, useful for comparing persistent
  read barriers and retained-snapshot fast paths before GPU DB adds durable
  route metadata around read-only execution. Journal entry added 2026-06-06
  from arXiv v1; the queued author metadata was corrected.
- `reviewed` — **Persistent HyTM via Fast Path Fine-Grained Locking**,
  Coccimiglio, Brown, and Ravi, arXiv 2025.
  URL: `https://arxiv.org/abs/2501.14783`
  Why: discovered while checking DHTM follow-ups; useful for contrasting
  hardware-fast-path persistence with software fallback paths, progress
  guarantees, and fine-grained lock interaction under future persistent-memory
  transaction metadata. Journal entry added 2026-06-06 from arXiv v2; the
  queued author metadata was corrected.
- `queued` — **SPHT: Scalable Persistent Hardware Transactions**, Vila et al.,
  2021.
  URL: `https://doi.org/10.1145/3437801.3441581`
  Why: DHTM is one early durable-HTM design; SPHT appears in the follow-up
  durable transaction line and is useful for checking whether hardware
  persistence support scales beyond a single simulated cache/log-buffer design.
- `queued` — **Crafty: Efficient, HTM-Compatible Persistent Transactions**,
  Genc, Bond, and Xu, PLDI 2020.
  URL: `https://doi.org/10.1145/3385412.3385981`
  Why: NV-HALT cites Crafty as an existing HTM-compatible persistent
  transaction design; useful for comparing fast-path persistence,
  instrumentation, and fallback requirements against route-publication tokens.
- `reviewed` — **TL4x: Buffered Durable Transactions on Disk as Fast as in
  Memory**, Assa, Correia, Ramalhete, Schiavoni, and Felber, PPoPP 2023.
  URL: `https://doi.org/10.1145/3572848.3577495`
  Why: NV-HALT compares against the Trinity/TL2 persistent-transaction line;
  useful for checking whether buffered durable transactions suggest simpler
  software-only durability/fallback baselines before GPU DB reaches for
  hardware-assisted persistence. Journal entry added 2026-06-06 from the
  PPoPP page and Zenodo author PDF; the paper DOI is
  `https://doi.org/10.1145/3572848.3577495`.
- `reviewed` — **ArchTM: Architecture-Aware, High Performance Transaction for
  Persistent Memory**, Wu, Ren, Peng, and Li, FAST 2021.
  URL: `https://www.usenix.org/conference/fast21/presentation/wu-kai`
  PDF: `https://www.usenix.org/system/files/fast21-wu-kai.pdf`
  Why: DUMBO's durability optimizations are HTM-specific; ArchTM is a primary
  persistent-memory transaction follow-up that emphasizes avoiding small
  random writes and improving sequential/coalesced persistence, useful for a
  software/storage-oriented baseline for future GPU DB CXL/NVM route
  metadata. Journal entry added 2026-06-06.
- `queued` — **Failure-Atomic Persistent Memory Updates via JUSTDO Logging**,
  Izraelevitz, Kelly, and Kolli, ASPLOS 2016.
  URL: `https://doi.org/10.1145/2872362.2872410`
  Why: ArchTM cites JUSTDO as a persistent-memory logging baseline; useful for
  comparing minimal logging, persist ordering, and recovery annotation against
  CoW-style route-publication records.
- `queued` — **Durable Transactional Memory Can Scale with TimeStone**,
  Krishnan et al., ASPLOS 2020.
  URL: `https://doi.org/10.1145/3373376.3378493`
  Why: ArchTM references TimeStone in the durable transaction line; useful for
  checking scalable durable transaction metadata and persistence barriers
  before GPU DB designs future NVM/CXL warm-tier route descriptors.
- `reviewed` — **SpecPMT: Speculative Logging for Resolving Crash Consistency
  Overhead of Persistent Memory**, Ye et al., ASPLOS 2023.
  URL: `https://doi.org/10.1145/3575693.3575696`
  PDF: `https://research.csc.ncsu.edu/picture/publications/papers/asplos23b_specLog.pdf`
  Author PDF: `https://yuanchaoxu6.github.io/files/ASPLOS2023_SpecPMT.pdf`
  Why: DUMBO reduces read and marker waits in durable HTM; SpecPMT is a modern
  speculative-logging follow-up for persistent memory, useful for comparing
  bounded speculative logs, hot/cold data handling, and crash-consistency
  overhead before GPU DB considers persistent warm-tier metadata. Journal entry
  added 2026-06-06 from the author PDF after direct curl of the seeded NCSU
  PDF URL returned an HTML page.
- `reviewed` — **Clobber-NVM: Log Less, Re-execute More**, Xu,
  Izraelevitz, and Swanson, ASPLOS 2021.
  URL: `https://doi.org/10.1145/3445814.3446730`
  PDF: `https://y4xu.github.io/clobber-nvm.pdf`
  Why: SpecPMT compares with re-execution and log-reduction approaches;
  useful for checking whether deterministic re-execution can reduce durable
  metadata for GPU DB route maintenance, checkpoint replay, or warm-tier
  updates without weakening SQL-visible side effects. Journal entry added
  2026-06-06; the queued DOI was corrected from `.3446748` to `.3446730`.
- `reviewed` — **ASAP: A Speculative Approach to Persistence**, Yadalam,
  Shah, Yu, and Swift, HPCA 2022.
  URL: `https://doi.org/10.1109/HPCA53966.2022.00070`
  Why: SpecPMT discusses speculative persistence approaches that relax
  ordering around persistence; useful for comparing when GPU DB can safely
  speculate on durable descriptor publication and when it must return
  explicit overload or wait for WAL-before-visibility. Journal entry added
  2026-06-07 from the author PDF.
- `queued` — **MOD: Minimally Ordered Durable Data Structures**, Haria,
  Hill, and Swift, ASPLOS 2020.
  URL: `https://doi.org/10.1145/3373376.3378482`
  Why: SpecPMT cites MOD as evidence that ordering can be minimized for
  durable data structures; useful for deriving minimal persist dependencies
  for route descriptors, resident metadata, and future CXL/NVM warm-tier
  structures.
- `queued` — **iDO: Compiler-Directed Failure Atomicity for Nonvolatile
  Memory**, Lee et al., MICRO 2018.
  URL: `https://doi.org/10.1109/MICRO.2018.00051`
  Author PDF: `https://sekwonlee.github.io/files/micro18_ido.pdf`
  Why: Clobber-NVM compares against iDO's idempotent-region
  recovery-via-resumption design; useful for isolating when compiler-marked
  deterministic replay regions beat conventional undo/redo logging for future
  warm-tier metadata or route-publication records.
- `queued` — **Asynchronous Persistence with ASAP**, Yadalam, Shah,
  Yu, and Swift, arXiv 2023.
  URL: `https://arxiv.org/abs/2302.13394`
  Why: follow-up from the ASAP line that appears to move speculation toward
  asynchronous atomic-region commit; useful for checking whether bounded
  recovery witnesses can support delayed commit acknowledgement without
  weakening WAL-before-visibility.

### Database file-system design, storage, and indexing

- `reviewed` — **Native Cloud Object Storage in Db2 Warehouse: Implementing a
  Fast and Cost-Efficient Cloud Storage Architecture**, Kalmuk et al.,
  SIGMOD/PODS Companion 2024.
  URL: `https://research.ibm.com/publications/native-cloud-object-storage-in-db2-warehouse-implementing-a-fast-and-cost-efficient-cloud-storage-architecture`
  DOI: `https://doi.org/10.1145/3626246.3653393`
  Why: modern production DBMS storage architecture over durable object storage;
  relevant to separating database-owned storage metadata, page/object layout,
  cache hierarchy, and read throughput from conventional local file-system
  assumptions. Journal entry added 2026-06-06.
- `reviewed` — **Vortex: A Stream-oriented Storage Engine For Big Data
  Analytics**, Edara, Forbes, and Li, SIGMOD/PODS Companion 2024.
  URL: `https://research.google/pubs/vortex-a-stream-oriented-storage-engine-for-big-data-analytics/`
  PDF: `https://www.cs.cmu.edu/~15721-f24/papers/Google_Vortex.pdf`
  Why: recent high-throughput storage-engine design for streaming and batch
  analytics; useful for GPU DB ingestion, scan throughput, file layout,
  indexing metadata, and bounded freshness tradeoffs. Journal entry added
  2026-06-06.
- `reviewed` — **DEX: Scalable Range Indexing on Disaggregated Memory**, VLDB
  2024.
  URL: `https://www.microsoft.com/en-us/research/publication/dex-scalable-range-indexing-on-disaggregated-memory/`
  Why: modern scalable B+-tree/range-index design for a remote/disaggregated
  memory tier; useful for comparing GPU DB cold/warm range indexes, remote
  placement metadata, and read-path latency under tiered storage. Journal entry
  added 2026-06-06.
- `reviewed` — **Sherman: A Write-Optimized Distributed B+Tree Index on
  Disaggregated Memory**, Wang, Lu, and Shu, SIGMOD 2022.
  URL: `https://arxiv.org/abs/2112.07320`
  DOI: `https://doi.org/10.1145/3514221.3526054`
  Why: DEX compares against Sherman as a one-sided RDMA B+-tree baseline;
  useful for write-optimized remote index layouts, RDMA command coalescing,
  hierarchical locks, and entry/node versioning before GPU DB adopts
  disaggregated range indexes. Journal entry added 2026-06-06.
- `reviewed` — **FORD: Fast One-sided RDMA-based Distributed Transactions for
  Disaggregated Persistent Memory**, Zhang et al., FAST 2022.
  URL: `https://www.usenix.org/conference/fast22/presentation/zhang-ming`
  PDF: `https://www.usenix.org/system/files/fast22-zhang-ming.pdf`
  Why: Sherman references FORD as a disaggregated persistent-memory
  transaction direction; useful for comparing one-sided RDMA transaction
  ordering, persistence, and remote metadata updates against GPU DB's future
  warm/cold-tier transaction and index routes. Journal entry added
  2026-06-06; the previously queued USENIX URL was corrected.
- `reviewed` — **Fast Distributed Transactions for RDMA-based Disaggregated
  Memory**, Lu et al., USENIX ATC 2025.
  URL: `https://www.usenix.org/conference/atc25/presentation/lu`
  Why: modern follow-up that compares against FORD and targets faster
  distributed transactions over RDMA-based disaggregated memory; useful for
  checking whether FORD's one-sided, rollback-oriented commit path has been
  superseded by newer localized-validation or hybrid-RDMA designs. Journal
  entry added 2026-06-06.
- `reviewed` — **DecLock: A Case of Decoupled Locking for Disaggregated
  Memory**, Zhang, Cheng, Chen, Wei, and Chen, arXiv 2025.
  URL: `https://arxiv.org/abs/2505.17641`
  Why: discovered while reviewing HDTX; useful for comparing HDTX's
  decentralized priority lock queues against a newer design that reduces
  memory-node NIC contention by decoupling lock ownership transfer from
  centralized lock-state maintenance. Journal entry added 2026-06-06.
- `reviewed` — **Big Metadata: When Metadata is Big Data**, Edara and
  Pasumansky, PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p3083-edara.pdf`
  DOI: `https://doi.org/10.14778/3476311.3476385`
  Why: Vortex relies on Big Metadata for large-scale column properties and
  partition pruning; useful for GPU DB's route metadata, resident-fragment
  pruning, snapshot-safe metadata publication, and metadata compaction.
  Journal entry added 2026-06-06 from the PVLDB PDF via browser fetch after
  direct `curl` to the same URL timed out.
- `reviewed` — **Pravega: A Tiered Storage System for Data Streams**,
  Gracia-Tinedo et al., Middleware 2023.
  URL: `https://doi.org/10.1145/3590140.3629113`
  Why: Vortex compares against Pravega's stream/tier abstraction; useful for
  studying stream-oriented tiering, truncation, transactions, and data
  placement across hot and cold storage. Journal entry added 2026-06-06 from
  DBLP/DOI metadata and the accessible Middleware 2023 PDF mirror after the
  ACM DOI page was blocked by a Cloudflare challenge.
- `queued` — **Virtual Log-Structured Storage for High-Performance
  Streaming**, Marcu, Costan, Nicolae, and Antoniu, IEEE CLUSTER 2021.
  URL: `https://doi.org/10.1109/Cluster48925.2021.00025`
  Why: Pravega cites this as evidence that too many parallel storage writes can
  saturate underlying drives; useful for comparing segment multiplexing,
  virtualized log structure, and cold-tier write coalescing for GPU DB ingest.
- `queued` — **Data Ingestion for the Connected World**, Meehan, Aslantas,
  Zdonik, Tatbul, and Du, CIDR 2017.
  URL:
  `https://www.cidrdb.org/cidr2017/papers/p47-meehan-cidr17.pdf`
  Why: Pravega identifies this as one of the few storage-focused treatments of
  tail and historical stream ingestion; useful for comparing ingestion
  semantics, stream/table boundaries, and storage contracts for retained
  GPU-readable histories.
- `queued` — **Delta Lake: High-Performance ACID Table Storage over Cloud
  Object Stores**, Armbrust et al., PVLDB 2020.
  URL: `https://doi.org/10.14778/3415478.3415560`
  Why: Big Metadata compares with Delta Lake's transaction-log-to-columnar
  metadata compaction model; useful for GPU DB cold-tier manifests, ACID object
  storage metadata, checkpoint compaction, and metadata-as-data contrasts.
- `reviewed` — **ALock: Asymmetric Lock Primitive for RDMA Systems**, Baran,
  Nelson-Slivon, Tseng, and Palmieri, SPAA 2024.
  URL: `https://doi.org/10.1145/3626183.3659977`
  arXiv: `https://arxiv.org/abs/2404.17980`
  Why: HDTX cites ALock as modern RDMA lock related work; useful for comparing
  priority scheduling with local/remote cohort locking when GPU DB future tiers
  mix local CPU accesses and remote/disaggregated-memory accesses. Journal
  entry added 2026-06-06.
- `reviewed` — **ShiftLock: Mitigate One-sided RDMA Lock Contention via
  Handover**, Gao, Wang, and Shu, USENIX FAST 2025.
  URL: `https://www.usenix.org/conference/fast25/presentation/gao`
  Why: DecLock compares directly against ShiftLock's MCS-style handover for
  RDMA reader-writer locks; useful for isolating whether GPU DB future-tier
  locks need centralized waiter queues, predecessor handoff, or phase-fair
  reader batching under hot remote indexes. Journal entry added 2026-06-06
  from the USENIX page and PDF.
- `reviewed` — **Citron: Distributed Range Lock Management with One-sided
  RDMA**, Gao, Lu, Xie, Wang, and Shu, USENIX FAST 2023.
  URL: `https://www.usenix.org/conference/fast23/presentation/gao`
  Why: ShiftLock cites Citron as one-sided RDMA range-lock work; useful for
  comparing point-lock handoff against range-lock metadata, interval conflicts,
  and future remote-tier range/index ownership. Journal entry added
  2026-06-06 from the USENIX page and PDF.
- `queued` — **Distributed Lock Management with RDMA: Decentralization without
  Starvation**, Yoon, Chowdhury, and Mozafari, SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3196890`
  Why: ShiftLock contrasts DSLR as a decentralized RDMA reader-writer lock
  baseline; useful for comparing starvation-free reader-writer semantics,
  backoff, release counters, and lock-table traffic before adopting handoff
  queues for GPU DB remote-tier metadata.
- `reviewed` — **Fast and Scalable In-Network Lock Management using Lock
  Fission**, Zhang, Cheng, Chen, and Chen, OSDI 2024.
  URL: `https://www.usenix.org/conference/osdi24/presentation/zhang-hanze`
  PDF: `https://www.usenix.org/system/files/osdi24-zhang-hanze.pdf`
  Why: DecLock contrasts software lock handoff with in-network lock
  management; useful as a foil before GPU DB assumes switch/NIC assistance for
  gateway, remote-tier, or disaggregated-memory lock coordination. Journal
  entry added 2026-06-06.
- `queued` — **NetLock: Fast, Centralized Lock Management Using
  Programmable Switches**, Chen et al., SIGCOMM 2020.
  URL: `https://doi.org/10.1145/3387514.3405857`
  Code: `https://github.com/netx-repo/NetLock/`
  Why: FissLock's main in-network lock-management baseline; useful if GPU DB
  needs a direct comparison between full on-switch participant state and
  fissioned compact grant metadata before considering NIC/switch-assisted
  route admission.
- `queued` — **SeqDLM: A Sequencer-Based Distributed Lock Manager for
  Efficient Shared File Access in a Parallel File System**, Chen et al.,
  SC 2022.
  URL:
  `https://sc22.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap149.html`
  DOI: `https://doi.org/10.1109/SC41404.2022.00060`
  PDF:
  `https://madsys.cs.tsinghua.edu.cn/publication/seqdlm-a-sequencer-based-distributed-lock-manager-for-efficient-shared-file-access-in-a-parallel-file-system/SC2022-chen.pdf`
  Why: Citron cites SeqDLM as a CPU/sequencer-based distributed lock manager
  for parallel shared-file access; useful for comparing one-sided static
  remote range locks against sequenced early-grant/early-revocation semantics
  when GPU DB studies cold-tier file/object concurrency.
- `queued` — **The Case for Distributed Shared-Memory Databases with
  RDMA-Enabled Memory Disaggregation**, Zhou et al., arXiv 2022.
  URL: `https://arxiv.org/abs/2207.03027`
  Why: FORD assumes disaggregated persistent memory as a transaction substrate;
  this database-focused position paper is useful for deciding which DB
  components should treat RDMA memory as shared state versus an explicit
  remote tier with owner-mediated publication.
- `reviewed` — **SMART: A High-Performance Adaptive Radix Tree for
  Disaggregated Memory**, Luo et al., OSDI 2023.
  URL: `https://www.usenix.org/conference/osdi23/presentation/luo`
  PDF: `https://www.usenix.org/system/files/osdi23-luo.pdf`
  Why: DEX compares against SMART as a trie/radix-tree disaggregated-memory
  index baseline; useful for contrasting B+-tree range routing with adaptive
  radix indexing and limited compute-side cache coherence. Journal entry added
  2026-06-06; the previously queued PVLDB/DOI metadata was corrected to the
  OSDI 2023 paper.
- `reviewed` — **Cabin: a Compressed Adaptive Binned Scan Index**,
  Chen and Chen, SIGMOD/PACMMOD 2024.
  URL: `https://doi.org/10.1145/3639312`
  PDF: `https://www.shimin-chen.com/papers/cabin-sigmod24.pdf`
  Why: recent scan-index design for compact auxiliary predicate metadata;
  relevant to deciding when GPU DB should maintain budgeted scan descriptors
  over resident, warm, or cold segments instead of relying only on full scans
  or B-tree-like access paths. Journal entry added 2026-06-06; the previously
  queued title/venue wording was corrected to the PACMMOD paper metadata.
- `reviewed` — **PULSE: Accelerating Distributed Pointer-Traversals on
  Disaggregated Memory**, Tang et al., ASPLOS 2025.
  URL: `https://arxiv.org/abs/2305.02388`
  DOI: `https://doi.org/10.1145/3669940.3707253`
  Why: SMART shows that remote pointer traversal saturates memory-side IOPS;
  PULSE explores pushing pointer traversal work closer to disaggregated memory,
  a useful future-tier contrast before GPU DB puts more range-index logic in
  remote memory or storage-side execution. Journal entry added 2026-06-06; the
  previously queued author/year metadata was corrected to the ASPLOS 2025
  paper.
- `reviewed` — **AIFM: High-Performance, Application-Integrated Far Memory**,
  Ruan et al., OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/ruan`
  PDF: `https://www.usenix.org/system/files/osdi20-ruan.pdf`
  Why: PULSE compares against AIFM as a data-structure-aware far-memory cache;
  useful for deciding whether GPU DB future-tier placement should expose
  application/route semantics to the memory manager before adopting
  accelerator-side pointer traversal. Journal entry added 2026-06-07.
- `queued` — **Can Far Memory Improve Job Throughput?**, Amaro et al.,
  EuroSys 2020.
  URL: `https://doi.org/10.1145/3342195.3387512`
  Why: AIFM compares against Fastswap from this line of page-granular far
  memory work; useful for measuring when transparent swap-like remote memory is
  enough and when GPU DB needs route-aware object/segment placement.
- `queued` — **Software-defined Far Memory in Warehouse-Scale Computers**,
  Lagar-Cavilla et al., ASPLOS 2019.
  URL: `https://doi.org/10.1145/3297858.3304053`
  Why: AIFM cites warehouse-scale cold-memory measurements; useful for deciding
  whether GPU DB's DRAM/NVMe/future-tier policy should use access-age,
  memory-pressure, and application hints rather than only cache-hit counters.
- `queued` — **Remote Regions: A Simple Abstraction for Remote Memory**,
  Aguilera et al., USENIX ATC 2018.
  URL: `https://www.usenix.org/conference/atc18/presentation/aguilera`
  Why: AIFM positions remote regions as a lower-level remote-memory
  abstraction; useful for comparing explicit region lifetimes with GPU DB's
  resident snapshot handles, pinned buffers, and cold-tier segment leases.
- `queued` — **StRoM: Smart Remote Memory**, Sidler, Wang, Chiosa,
  Kulkarni, and Alonso, EuroSys 2020.
  URL: `https://doi.org/10.1145/3342195.3387529`
  Why: AIFM's active remote components resemble smart remote-memory offload;
  useful for comparing remote-side filtering/aggregation with GPU DB's future
  storage-side or fabric-side pruning before pulling cold fragments to HBM.
- `queued` — **Rethinking the Encoding of Integers for Scans on Skewed Data**,
  Prammer and Patel, SIGMOD/PACMMOD 2023.
  URL: `https://doi.org/10.1145/3626751`
  PDF: `https://www.pdl.cmu.edu/ftp/Database/rethinking-encoding.pdf`
  Why: modern bit-parallel scan encoding work discovered while reviewing
  Cabin; useful for comparing compact scan indexes against encoded resident
  column layouts that move pruning-relevant bits earlier for skewed data.
- `queued` — **DINOMO: An Elastic, Scalable, High-Performance Key-Value Store
  for Disaggregated Persistent Memory**, Wei et al., arXiv 2022.
  URL: `https://arxiv.org/abs/2209.08743`
  Why: SMART's related ecosystem includes disaggregated persistent-memory
  key-value stores; useful for comparing ownership partitioning, adaptive
  caching, selective replication, and log-free indexing against GPU DB's
  warm/cold tier metadata and owner domains.
- `reviewed` — **StaR: Breaking the Scalability Limit for RDMA**, Wang et al.,
  ICNP 2021.
  URL: `https://doi.org/10.1109/ICNP52444.2021.9651935`
  PDF: `https://icnp21.cs.ucr.edu/papers/icnp21camera-paper30.pdf`
  Why: ALock relies on QP-thrashing limits in commodity RNICs; useful for
  evaluating whether future GPU DB remote-tier/session paths should reduce
  QP state, multiplex connections, or expose RNIC-cache pressure as admission
  telemetry. Journal entry added 2026-06-06.
- `reviewed` — **SRNIC: A Scalable Architecture for RDMA NICs**, Wang et al.,
  NSDI 2023.
  URL: `https://www.usenix.org/conference/nsdi23/presentation/wang-zilong`
  PDF: `https://www.usenix.org/system/files/nsdi23-wang-zilong.pdf`
  Why: StaR solves fan-in RNIC state by moving state to the low-concurrency
  endpoint; SRNIC is a newer open USENIX follow-up that redesigns on-chip
  RDMA data structures with cache-free QP scheduling and memory-free
  selective repeat, useful for comparing descriptor placement against
  hardware-scalable reliable transport before GPU DB adopts remote-tier
  session or storage paths. Journal entry added 2026-06-06.
- `queued` — **Revisiting Network Support for RDMA**, Mittal et al.,
  SIGCOMM 2018.
  URL: `https://doi.org/10.1145/3230543.3230557`
  PDF: `https://cs.nyu.edu/~apanda/assets/papers/sigcomm18-irn.pdf`
  arXiv: `https://arxiv.org/abs/1806.08159`
  Why: SRNIC builds on IRN's PFC-free lossy-RDMA direction with selective
  repeat; useful for comparing whether GPU DB future remote tiers need
  lossless fabrics, selective retransmission, or database-owned fallback when
  remote-memory transport becomes congested or lossy.
- `queued` — **Fast RDMA-based Ordered Key-Value Store using Remote Learned
  Cache**, Wei, Chen, and Chen, OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/wei`
  PDF: `https://www.usenix.org/system/files/osdi20-wei.pdf`
  Why: ALock contrasts HTM/RDMA synchronization from this line of work; useful
  for comparing remote learned-cache index routing, local/remote access
  asymmetry, and fallback behavior for future warm-tier key lookups.
- `reviewed` — **SkyStore: Cost-Optimized Object Storage Across Regions and
  Clouds**, Liu et al., PVLDB 2025.
  URL: `https://research.ibm.com/publications/skystore-cost-optimized-object-storage-across-regions-and-clouds`
  PDF: `https://www.vldb.org/pvldb/vol18/p2084-liu.pdf`
  DOI: `https://doi.org/10.14778/3734839.3734846`
  Why: follow-up object-storage placement work discovered during the Db2 native
  COS review; relevant to future cold-tier placement, replication, cost-aware
  promotion, and region/cloud-aware object movement policy. Journal entry
  added 2026-06-06.
- `reviewed` — **ByteHouse: A Cloud-Native OLAP Engine with Incremental
  Computation and Multi-Modal Retrieval**, arXiv 2026.
  URL: `https://arxiv.org/abs/2602.08226`
  Why: modern cloud-native warehouse architecture with SSD-backed cache and a
  virtual file-system layer; useful as a contrast point for DB-owned NVMe
  caches, remote object layout, and local-access abstraction choices. Journal
  entry added 2026-06-06 from arXiv v2.
- `reviewed` — **CloudCast: High-Throughput, Cost-Aware Overlay Multicast in the
  Cloud**, Wooders et al., NSDI 2024.
  URL: `https://www.usenix.org/conference/nsdi24/presentation/wooders`
  Why: SkyStore builds on the Skyplane/SkyPilot cloud-placement ecosystem and
  cites CloudCast for cost-aware cloud overlays; useful for comparing
  multi-destination cold-tier replication, route fanout, transfer throughput,
  and cost-aware object movement before GPU DB adopts remote object tiers.
  Journal entry added 2026-06-06 from the USENIX page and PDF.
- `reviewed` — **Skyplane: Optimizing Transfer Cost and Throughput Using
  Cloud-Aware Overlays**, Jain et al., NSDI 2023.
  URL: `https://www.usenix.org/conference/nsdi23/presentation/jain`
  arXiv: `https://arxiv.org/abs/2210.07259`
  Why: SkyStore cites Skyplane as related intercloud transfer work; useful for
  evaluating whether cold-tier promotion should use direct object reads,
  staged transfer, or overlay routing when future GPU DB deployments span
  regions, object stores, or disaggregated storage pools. Journal entry added
  2026-06-06 from the USENIX page and PDF.
- `queued` — **BDS: A Centralized Near-Optimal Overlay Network for
  Inter-Datacenter Data Replication**, Zhang et al., EuroSys 2018.
  URL: `https://doi.org/10.1145/3190508.3190532`
  Why: Cloudcast cites BDS as a bandwidth-oriented inter-datacenter overlay
  replication baseline; useful for comparing throughput-first overlay routing
  with GPU DB's future cost/freshness-aware cold-tier and replica movement.
- `queued` — **CodedBulk: Inter-Datacenter Bulk Transfers Using Network
  Coding**, Tseng et al., NSDI 2021.
  URL: `https://www.usenix.org/conference/nsdi21/presentation/tseng`
  Why: Skyplane cites CodedBulk as a bulk-transfer multicast direction;
  useful for comparing coded replication/fanout with segment-stripe cold-tier
  movement, checkpoint distribution, and replica warmup under partial-link
  bottlenecks.
- `queued` — **Cost-Effective Cloud Edge Traffic Engineering with CASCARA**,
  Singh et al., NSDI 2021.
  URL: `https://www.usenix.org/conference/nsdi21/presentation/singh`
  Why: Skyplane contrasts provider-side traffic engineering with
  customer-visible cost/throughput planning; useful for deciding which tier
  placement decisions should remain database-owned versus delegated to
  provider or fabric-level traffic engineering.

### WAL, logging, and read/write throughput

- `reviewed` — **PALF: Replicated Write-Ahead Logging for Distributed
  Databases**, Xu et al., PVLDB 2024.
  PDF: `https://www.vldb.org/pvldb/vol17/p3745-xu.pdf`
  DOI: `https://doi.org/10.14778/3685800.3685803`
  Why: production distributed WAL design from OceanBase with append-only log
  files, replication, recovery, and read/write performance implications;
  directly relevant to GPU DB WAL-before-visibility and future replica paths.
  Journal entry added 2026-06-06.
- `reviewed` — **DecLog: Decentralized Logging in Non-Volatile Memory for Time
  Series Database Systems**, Zheng et al., PVLDB 2023.
  PDF: `https://www.vldb.org/pvldb/vol17/p1-zheng.pdf`
  DOI: `https://doi.org/10.14778/3617838.3617839`
  Why: decentralized WAL/logging path for high-ingest workloads; useful for
  comparing owner-local log buffers, NVM/SSD flush behavior, and write
  throughput under massive append pressure.
  Journal entry added 2026-06-06.
- `reviewed` — **NVWAL: Exploiting NVRAM in Write-Ahead Logging**,
  Kim et al., ASPLOS 2016.
  URL: `https://doi.org/10.1145/2872362.2872392`
  Metadata: `https://dblp.org/rec/conf/asplos/KimKBNW16`
  Why: DecLog contrasts NVM WAL designs that use persistent-memory ordering
  and consolidated flushing; useful for comparing hardware-shaped WAL records,
  NVRAM log placement, and SQLite-style transactional durability with GPU DB's
  future NVM/CXL-tier WAL options. Journal entry added 2026-06-06 from the
  KAIST OS Lab PDF mirror.
- `reviewed` — **Improving database performance by leveraging network-assisted
  logging**, Future Generation Computer Systems 2025.
  URL: `https://www.sciencedirect.com/science/article/pii/S0167739X25000809`
  DOI: `https://doi.org/10.1016/j.future.2025.107785`
  Why: recent WAL-overhead reduction paper; useful as a foil for local durable
  WAL, remote durable logging, NIC-assisted persistence, and the throughput
  cost of synchronous commit.
  Journal entry added 2026-06-06 from the accessible ScienceDirect preview;
  full body/details were not openly available.
- `reviewed` — **BVLSM: Write-Efficient LSM-Tree Storage via WAL-Time Key-Value
  Separation**, arXiv 2025.
  URL: `https://arxiv.org/abs/2506.04678`
  Why: WAL-time key-value separation links write-ahead logging directly to LSM
  write amplification, memory pressure, and read/write jitter; useful for GPU
  DB cold-tier ingest and compaction policy.
  Journal entry added 2026-06-06.
- `reviewed` — **MatrixKV: Reducing Write Stalls and Write Amplification in
  LSM-tree Based KV Stores with Matrix Container**, Yao et al., USENIX ATC
  2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/yao`
  Why: BVLSM contrasts NVM-oriented key-value separation and write-stall
  reduction approaches; useful for comparing explicit DRAM/NVM/NVMe tiered
  value placement and write-stall smoothing. Journal entry added 2026-06-06
  from the USENIX page and PDF.
- `reviewed` — **Differentiated Key-Value Storage Management for Balanced I/O
  Performance**, Li et al., USENIX ATC 2021.
  URL: `https://www.usenix.org/conference/atc21/presentation/li-yongkun`
  Why: BVLSM cites differentiated KV storage management as related work;
  useful for deciding whether GPU DB cold-tier value payloads should be routed
  by size, update frequency, and read/write interference rather than a single
  separation threshold. Journal entry added 2026-06-06 from the USENIX page
  and PDF.
- `reviewed` — **SILK: Preventing Latency Spikes in Log-Structured Merge
  Key-Value Stores**, Balmau et al., USENIX ATC 2019.
  URL: `https://www.usenix.org/conference/atc19/presentation/balmau`
  Why: DiffKV's balanced-write/read/scan design still leaves foreground
  compaction and merge interference as a tail-latency question; SILK is a
  focused follow-up for scheduler-level compaction smoothing and latency-spike
  control in LSM-style cold/warm tiers. Journal entry added 2026-06-06 from
  the USENIX page and PDF.
- `reviewed` — **SplinterDB: Closing the Bandwidth Gap for NVMe Key-Value
  Stores**, Conway et al., USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/conway`
  Why: DiffKV evaluates commodity SSD LSM tradeoffs; SplinterDB is a modern
  NVMe-oriented KV-store design useful for comparing write-optimized indexing,
  space amplification, and scan/read behavior before GPU DB commits to an
  LSM-like cold-tier structure. Journal entry added 2026-06-06.
- `queued` — **Redesigning LSMs for Nonvolatile Memory with NoveLSM**,
  Kannan, Bhat, Gavrilovska, Arpaci-Dusseau, and Arpaci-Dusseau, USENIX
  ATC 2018.
  URL: `https://www.usenix.org/conference/atc18/presentation/kannan`
  Why: MatrixKV's main NVM-LSM baseline; useful for comparing large
  persistent MemTables against bounded matrix-container compaction before GPU
  DB adopts a warm NVM/NVMe write-staging tier.
- `queued` — **SLM-DB: Single-Level Key-Value Store with Persistent Memory**,
  Kaiyrakhmet et al., USENIX FAST 2019.
  URL: `https://www.usenix.org/conference/fast19/presentation/kaiyrakhmet`
  Why: MatrixKV cites SLM-DB as an NVM/SSD LSM alternative that collapses
  levels; useful for comparing single-level persistent-memory indexing against
  MatrixKV-style bounded first-tier compaction and P8 cold-tier segment
  refresh.
- `queued` — **Monkey: Optimal Navigable Key-Value Store**, Dayan,
  Athanassoulis, and Idreos, SIGMOD 2017.
  URL: `https://doi.org/10.1145/3035918.3064054`
  Author PDF: `https://stratos.seas.harvard.edu/files/stratos/files/monkeykeyvaluestore.pdf`
  Why: SILK contrasts scheduler-level smoothing with LSM parameter tuning;
  Monkey is a primary modern source for analytically assigning Bloom-filter
  memory and LSM shape, useful for separating cold-tier layout tuning from
  maintenance admission control.
- `queued` — **Dostoevsky: Better Space-Time Trade-Offs for LSM-Tree Based
  Key-Value Stores via Adaptive Removal of Superfluous Merging**, Dayan and
  Idreos, SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3196927`
  Author PDF: `https://stratos.seas.harvard.edu/files/stratos/files/dostoevskykvstore.pdf`
  Why: SILK cites LSM tuning and reduced internal work as complementary but
  insufficient for tails; Dostoevsky is useful for comparing lazy-leveling and
  fluid LSM layouts against explicit compaction scheduling under GPU DB
  write/read SLOs.
- `queued` — **WiscKey: Separating Keys from Values in SSD-conscious
  Storage**, Lu et al., USENIX FAST 2016.
  URL: `https://www.usenix.org/conference/fast16/technical-sessions/presentation/lu`
  PDF: `https://www.usenix.org/system/files/conference/fast16/fast16-papers-lu.pdf`
  Why: SILK's related work contrasts key-value separation as a throughput and
  write-amplification technique; WiscKey is a foundational post-2015 primary
  source for deciding when GPU DB should separate keys, medium payloads, and
  large values across NVMe-friendly tiers.
- `queued` — **Tucana: Design and Implementation of a Fast and Efficient
  Scale-up Key-value Store**, Papagiannis et al., USENIX ATC 2016.
  URL: `https://www.usenix.org/conference/atc16/technical-sessions/presentation/papagiannis`
  Why: SplinterDB identifies Tucana as the closest B-epsilon-tree style SSD
  key-value-store predecessor; useful for comparing CPU cost, concurrency, and
  write amplification before adopting branchy cold-tier indexes.
- `queued` — **PebblesDB: Building Key-Value Stores using Fragmented
  Log-Structured Merge Trees**, Raju et al., SOSP 2017.
  URL: `https://doi.org/10.1145/3132747.3132765`
  Why: SplinterDB adapts and extends fragmentation/size-tiering ideas from
  PebblesDB; useful for testing fragmented LSM layouts against active-branch
  cold-tier directories and short-range scan penalties.
- `reviewed` — **Taurus: Lightweight Parallel Logging for In-Memory Database
  Management Systems**, Xia, Yu, Pavlo, and Devadas, SIGMOD/PACMMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3389713`
  PDF: `https://db.cs.cmu.edu/papers/2020/p677-xia.pdf`
  Why: modern parallel logging design that tracks transaction dependencies
  across multiple log streams; useful follow-up to NVWAL's single-writer
  persistent-memory protocol when GPU DB evaluates partition-owned mutation
  logs, group commit, and parallel recovery. Journal entry already exists
  under the PVLDB 2020 metadata; this stale duplicate was corrected from
  `queued` to `reviewed` on 2026-06-06.
- `reviewed` — **High Throughput Replication with Integrated Membership
  Management**, Fouto, Preguica, and Leitao, USENIX ATC 2022.
  URL: `https://www.usenix.org/conference/atc22/presentation/fouto`
  Why: PALF contrasts separate metadata/reconfiguration choices with integrated
  membership replication; useful for comparing future GPU DB replicated WAL
  membership, failover, and availability tradeoffs. Journal entry added
  2026-06-06.
- `queued` — **DistributedLog: A High Performance Replicated Log Service**,
  Guo, Dhamankar, and Stewart, ICDE 2017.
  URL: `https://doi.org/10.1109/ICDE.2017.172`
  Why: PALF compares CSN-style database ordering with replicated log services;
  useful for deciding whether GPU DB should keep WAL bundled with mutation
  owners or expose an independent replicated log service.
- `reviewed` — **Moving on From Group Commit: Autonomous Commit Enables High
  Throughput and Low Latency on NVMe SSDs**, Nguyen, Alhomssi, Ziegler, and
  Leis, PACMMOD/SIGMOD 2025.
  URL: `https://doi.org/10.1145/3725328`
  PDF: `https://lamduynguyen.github.io/assets/pdf/latency.pdf`
  Why: discovered while reviewing Chardonnay's fast 2PC and pipelined WAL
  assumptions; useful for comparing per-worker small log writes and parallel
  commit acknowledgement against group commit when GPU DB tries to keep
  WAL-before-visibility latency low on modern NVMe. Journal entry already
  exists; this stale duplicate was corrected from `queued` to `reviewed` on
  2026-06-07.
- `reviewed` — **Rethinking Logging, Checkpoints, and Recovery for
  High-Performance Storage Engines**, Haubenschild, Sauer, Neumann, and Leis,
  SIGMOD/PACMMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3389716`
  PDF: `https://db.in.tum.de/~leis/papers/rethinkingLogging.pdf`
  Why: cited by ITLogging as a modern high-performance logging and recovery
  baseline; useful for comparing edge-staged request logs with canonical
  command/physiological logging, checkpoint boundaries, and recovery latency.
  Journal entry added 2026-06-06; the previously queued DOI was corrected to
  the SIGMOD 2020 paper metadata.
- `reviewed` — **FineLine: Log-structured Transactional Storage and Recovery**,
  Sauer, Graefe, and Haerder, PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p2249-sauer.pdf`
  DOI: `https://doi.org/10.14778/3275366.3275373`
  Why: the LeanStore recovery paper contrasts FineLine's single-storage,
  log-structured recovery design with page-based WAL/checkpointing; useful for
  testing whether GPU DB cold-tier segments should preserve a separate
  WAL/database split or collapse some persistent data into indexed log
  structures. Journal entry added 2026-06-06.
- `queued` — **Instant recovery with write-ahead logging**,
  Graefe, Guy, Sauer, and Haerder, Datenbank-Spektrum 2015.
  DOI: `https://doi.org/10.1007/s13222-015-0204-3`
  Why: FineLine builds on instant-recovery ideas such as on-demand page repair,
  restart, restore, and log-history access; useful for comparing indexed-log
  storage with a more conservative WAL design that opens quickly and performs
  redo/undo work lazily.
- `reviewed` — **Can Applications Recover from fsync Failures?**, Rebello,
  Patel, Alagappan, A. Arpaci-Dusseau, and R. Arpaci-Dusseau, USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/rebello`
  Why: TL4x explicitly calls out `msync()` failure handling; useful for
  checking how GPU DB should surface, retry, or quarantine failed flushes in
  WAL, checkpoint, resident-image, and cold-tier manifest publication paths.
  Journal entry added 2026-06-07 from the USENIX page and PDF.
- `reviewed` — **Finding Crash-Consistency Bugs with Bounded Black-Box Crash
  Testing**, Mohan et al., OSDI 2018.
  URL: `https://www.usenix.org/conference/osdi18/presentation/mohan`
  Why: the fsync-failure paper contrasts transient block-write failures with
  crash-consistency testing; useful for designing a bounded crash/fault matrix
  over GPU DB WAL, checkpoint, manifest, and route-publication states. Journal
  entry added 2026-06-07 from the USENIX page and PDF.
- `reviewed` — **Application Crash Consistency and Performance with CCFS**,
  Pillai et al., FAST 2017.
  URL: `https://www.usenix.org/conference/fast17/technical-sessions/presentation/pillai`
  Why: cited by the fsync-failure paper as application-level crash-consistency
  support; useful for comparing file-system assistance, consistency contracts,
  and performance overhead before GPU DB builds DB-owned durable publication
  checks. Journal entry added 2026-06-07 from the USENIX page and PDF.
- `queued` — **Isotope: Transactional Isolation for Block Storage**,
  Shin, Balakrishnan, Marian, and Weatherspoon, FAST 2016.
  URL: `https://www.usenix.org/conference/fast16/technical-sessions/presentation/shin`
  Why: CCFS notes that block-level atomicity and isolation can simplify
  stream-separated crash consistency; useful for comparing lower-level storage
  transaction support against GPU DB's owner-stream WAL/checkpoint/manifest
  publication contracts.
- `queued` — **Lightweight Application-Level Crash Consistency on
  Transactional Flash Storage**, Min, Kang, Kim, Lee, and Eom, USENIX ATC
  2015.
  URL: `https://www.usenix.org/conference/atc15/technical-session/presentation/min`
  Why: CCFS contrasts CFS-style application-level atomicity with stream
  ordering; useful for checking whether a narrower transactional storage
  primitive can protect GPU DB cold-tier metadata without replacing database
  WAL/MVCC semantics.
- `skipped` — **On the Complexity of Crafting Crash-Consistent Applications**,
  Pillai et al., OSDI 2014.
  URL: `https://www.usenix.org/conference/osdi14/technical-sessions/presentation/pillai`
  Why: pre-2015 historical source cited by the fsync paper; skipped by the
  2015-present selection policy.
- `reviewed` — **TIPS: Making Volatile Index Structures Persistent with
  DRAM-NVMM Tiering**, Ramanathan et al., USENIX ATC 2021.
  URL: `https://www.usenix.org/conference/atc21/presentation/krishnan`
  Why: TL4x's replica-copy approach contrasts with tiering volatile index
  structures onto persistent memory; useful for comparing route/index
  reconstruction, persistent update granularity, and DRAM/NVMM placement
  before GPU DB persists warm-tier indexes or resident-fragment directories.
  Journal entry added 2026-06-07 from the USENIX page and PDF.
- `queued` — **Pronto: Easy and Fast Persistence for Volatile Data
  Structures**, Liu et al., ASPLOS 2020.
  URL: `https://doi.org/10.1145/3373376.3378456`
  Why: TIPS compares against PRONTO's operation-log/snapshot conversion of
  volatile indexes; useful for deciding when route metadata can be persisted
  by semantic logs versus when the hot structure itself should live in a
  future persistent tier.
- `queued` — **RECIPE: Converting Concurrent DRAM Indexes to Persistent-Memory
  Indexes**, Lee et al., SOSP 2019.
  URL: `https://doi.org/10.1145/3341301.3359635`
  Why: TIPS contrasts RECIPE's index-specific conversion guidelines and
  weaker durability assumptions; useful for checking which persistent-index
  conversions are too fragile for GPU DB route/index publication.
- `queued` — **NVTraverse: In NVRAM Data Structures, the Destination is More
  Important than the Journey**, Friedman et al., PLDI 2020.
  URL: `https://doi.org/10.1145/3385412.3386031`
  Why: TIPS compares against NVTraverse for durable-linearizable lock-free
  indexes; useful for deciding whether GPU DB future-tier indexes should pay
  per-traversal persistence costs or keep persistence off read paths.
- `queued` — **Chipmunk: Investigating Crash-Consistency in Persistent-Memory
  File Systems**, LeBlanc et al., EuroSys 2023.
  URL: `https://doi.org/10.1145/3552326.3567498`
  PDF: `https://jamesbornholt.com/papers/chipmunk-eurosys23.pdf`
  Why: modern follow-up to CrashMonkey/ACE for persistent-memory file systems;
  useful for checking whether GPU DB's future PM/CXL warm-tier metadata needs
  PM-specific crash-state generation beyond block-device persistence points.
- `queued` — **Synthesis-Aided Crash Consistency for Storage Systems**, Van
  Geffen, Wang, Torlak, and Bornholt, ECOOP 2023.
  URL: `https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECOOP.2023.35`
  DOI: `https://doi.org/10.4230/LIPIcs.ECOOP.2023.35`
  Why: discovered while reviewing B3; useful for comparing black-box witness
  generation with storage code synthesis and crash-consistency-by-construction
  for future WAL/checkpoint/manifest state machines.
- `reviewed` — **Scalable and Accurate Application-Level Crash-Consistency
  Testing via Representative Testing**, Gu et al., PACMPL/OOPSLA 2025.
  URL: `https://arxiv.org/abs/2503.01390`
  DOI: `https://doi.org/10.1145/3720431`
  Why: modern application-level crash-testing follow-up discovered from the
  B3 line; useful for deriving representative crash states for GPU DB's
  database-level WAL, checkpoint, object-manifest, and route-publication
  operations without enumerating every low-level storage state. Journal entry
  added 2026-06-07 from the arXiv PDF and ACM metadata.
- `reviewed` — **DURINN: Adversarial Memory and Thread Interleaving for
  Detecting Durable Linearizability Bugs**, Fu, Lee, and Min, OSDI 2022.
  URL: `https://www.usenix.org/conference/osdi22/presentation/fu`
  Why: Pathfinder explicitly does not systematically explore thread
  synchronization interleavings; DURINN is a primary follow-up for combining
  durable persistence ordering with adversarial concurrency schedules before
  GPU DB trusts crash witnesses for multi-owner WAL, catalog, and resident
  metadata updates. Journal entry added 2026-06-07 from the USENIX page and
  PDF.
- `queued` — **PMTest: A Fast and Flexible Testing Framework for Persistent
  Memory Programs**, Liu, Wei, Zhao, Kolli, and Khan, ASPLOS 2019.
  URL: `https://doi.org/10.1145/3297858.3304015`
  PDF: `https://cseweb.ucsd.edu/~jzhao/files/pmtest-asplos2019.pdf`
  Why: DURINN contrasts application-level and annotation-based persistent
  memory testing tools; useful for checking whether GPU DB can express
  route-publication ordering and durability guarantees as reusable assertions
  before building heavier adversarial interleaving tests.
- `queued` — **Jaaru: Efficiently Model Checking Persistent Memory
  Programs**, Gorjiara, Xu, and Demsky, ASPLOS 2021.
  URL: `https://doi.org/10.1145/3445814.3446735`
  PDF: `https://web.cs.ucla.edu/~harryxu/papers/jaaru-asplos21.pdf`
  Why: DURINN compares against exhaustive/model-checking approaches for
  persistent memory; useful for deciding when symbolic persistence-state
  exploration is worthwhile for GPU DB WAL/checkpoint/manifest state machines
  versus targeted representative crash tests.
- `queued` — **Agamotto: How Persistent is your Persistent Memory
  Application?**, Neal et al., OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/neal`
  Why: Pathfinder contrasts pattern-based and symbolic/fuzzing approaches
  with representative testing; useful for checking whether persistent-memory
  route descriptors, warm-tier indexes, or future CXL metadata need
  symbolic path generation in addition to representative crash-state pruning.
- `queued` — **Witcher: Systematic Crash Consistency Testing for Non-Volatile
  Memory Key-Value Stores**, Fu et al., SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483556`
  Why: Pathfinder compares against Witcher's guardian-pattern pruning for
  MMIO key-value structures; useful for deciding whether GPU DB should encode
  expected durability protocols as explicit bug patterns, representative
  update behaviors, or both.

- `queued` — **Atlas: Scalable and Available State Machine Replication**,
  Enes et al., EuroSys 2020.
  URL: `https://doi.org/10.1145/3342195.3387543`
  Why: ChainPaxos compares against distributed-load SMR protocols and cites
  Atlas as a planet-scale replication baseline; useful for contrasting
  dependency-aware replica execution, geo latency, and conflict handling with
  pipeline-shaped replicated WAL or future multi-owner commit routes.
- `queued` — **Toward a Generic Fault Tolerance Technique for Partial Network
  Partitioning**, Alfatafta et al., OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/alfatafta`
  Why: ChainPaxos motivates integrated membership partly through partial
  partition hazards in externally coordinated systems; useful for testing
  whether GPU DB replicated owners, route membership, and cold-tier placement
  remain safe under asymmetric partitions.

- `reviewed` — **Towards Optimal Transaction Scheduling**, Cheng et al.,
  PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf`
  DOI: `https://doi.org/10.14778/3681954.3681956`
  Code: `https://github.com/audreyccheng/transaction-scheduling`
  Why: modern schedule-first OLTP work selected because the queue lacked a
  strong 2023-present full paper in transaction scheduling; directly relevant
  to hot-key admission, abort/fallback reduction, and route ordering under
  contention.
- `reviewed` — **Intelligent Transaction Scheduling to Enhance Concurrency in
  High-Contention Workloads**, Chen, Shen, and Wu, Applied Sciences 2025.
  URL: `https://www.mdpi.com/2076-3417/15/11/6341`
  DOI: `https://doi.org/10.3390/app15116341`
  Why: discovered while reviewing OCC batching/reordering; useful as a modern
  dependency-aware scheduling follow-up that combines hot-data partitioning,
  fine-grained operation scheduling, and learned scheduling under high
  contention. Lower priority than SIGMOD/VLDB/OSDI transaction papers, but
  relevant if the queue needs more recent contention-scheduling contrasts.
  Journal entry added 2026-06-06 from accessible HTML/DOAJ metadata after the
  MDPI PDF endpoint returned HTTP 403.
- `reviewed` — **ForeSight: A Predictive-Scheduling Deterministic Database**,
  Huang et al., arXiv 2025.
  URL: `https://arxiv.org/abs/2508.17375`
  DOI: `https://doi.org/10.48550/arXiv.2508.17375`
  Why: discovered while reviewing DCoS; useful as a newer transaction
  scheduling follow-up that predicts conflicts without pre-obtained read/write
  sets, integrates MVCC-style fallback, and generates conflict-aware
  deterministic schedules under skew. Journal entry added 2026-06-06 from
  arXiv v2.
- `queued` — **DoppelGanger++: Towards Fast Dependency Graph Generation for
  Database Replay**, Lee et al., PACMMOD 2024.
  DOI: `https://doi.org/10.1145/3639305`
  Why: ForeSight cites SSFS/DoppelGanger++ as a fast dependency-graph
  generation baseline; useful for separating replay-oriented dependency graph
  construction from online route scheduling and conflict prediction.
- `queued` — **Practical Deterministic Transaction Processing with Low-cost
  Re-execution**, Li, Wang, and Huang, ICPADS 2024.
  DOI: `https://doi.org/10.1109/ICPADS63350.2024.00065`
  Why: ForeSight contrasts DMUCCA as a deterministic re-execution and
  access-pattern-weighted reordering design; useful for comparing
  MVCC-style fallback, retry admission, and residual conflict handling under
  hot transaction batches.
- `reviewed` — **Transaction Scheduling: From Conflicts to Runtime Conflicts**,
  Cao et al., PACMMOD/SIGMOD 2023.
  URL: `https://doi.org/10.1145/3603164`
  PDF:
  `https://www.pure.ed.ac.uk/ws/portalfiles/portal/360117816/Transaction_Scheduling_CAO_DOA16082022_AFV.pdf`
  Why: schedule-first OLTP work contrasts static/runtime conflict scheduling
  and proactive deferral; useful for comparing bounded SMF-style admission
  with lighter runtime probes for hot writes.
- `reviewed` — **Polaris: Enabling Transaction Priority in Optimistic
  Concurrency Control**, Ye et al., PACMMOD/SIGMOD 2023.
  URL: `https://doi.org/10.1145/3588724`
  PDF: `https://chenhao-ye.github.io/publication/polaris/polaris.pdf`
  Why: priority-aware optimistic concurrency control with lightweight
  reservation; useful follow-up for protecting high-priority retained reads,
  writes, refreshes, or latency-sensitive sessions under hot-key contention.
- `skipped` — **The Yin and Yang of Processing Data Warehousing Queries on GPU
  Devices**, Yuan et al., PVLDB 2013.
  URL: `https://www.vldb.org/pvldb/vol6/p817-yuan.pdf`
  Why: pre-2015; keep only as historical context.
- `skipped` — **High-Throughput Transaction Executions on Graphics Processors**,
  He and Yu, PVLDB 2011.
  URL: `https://www.vldb.org/pvldb/vol4/p314-he.pdf`
  Why: pre-2015; keep only as historical context.
- `reviewed` — **Themis: A GPU-accelerated Relational Query Execution Engine**,
  Hong et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol18/p426-han.pdf`
  DOI: `https://doi.org/10.14778/3705829.3705856`
  Why: modern GPU relational engine with execution and load-balancing details
  relevant to fused retained route design.
- `reviewed` — **GPU Acceleration of SQL Analytics on Compressed Data**,
  Huang et al., PVLDB 2025.
  URL: `https://arxiv.org/abs/2506.10092`
  DOI: `https://doi.org/10.14778/3778092.3778095`
  Why: evaluates compressed-data SQL execution on GPUs and may inform
  dense-versus-compressed resident page benchmarks.
- `reviewed` — **Path to GPU-Initiated I/O for Data-Intensive Systems**,
  Torp et al., DaMoN 2025.
  URL: `https://doi.org/10.1145/3736227.3736232`
  Why: practical evaluation of GPU-initiated IO paths for data-intensive
  systems, directly relevant to DPF-style over-resident execution.
- `reviewed` — **Scaling GPU-Accelerated Databases beyond GPU Memory Size**,
  Li et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4518-li.pdf`
  Why: modern out-of-GPU-memory database execution work relevant to P8
  over-resident partitioning and CPU/GPU fallback.
- `reviewed` — **Vortex: Overcoming Memory Capacity Limitations in
  GPU-Accelerated Large-Scale Data Analytics**, Yuan et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol18/p1250-yuan.pdf`
  Why: multi-GPU/interconnect approach to capacity limits, useful for future
  over-resident and multi-device planning.
- `reviewed` — **Tigger: A Database Proxy That Bounces with User-Bypass**,
  Butrovich et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p3335-butrovich.pdf`
  Why: user-bypass/eBPF database proxy work cited by Looking Glass 2.0;
  relevant to reducing protocol proxying, client/server communication, and
  kernel boundary overhead.
- `reviewed` — **Cloud-Native Database Systems and Unikernels: Reimagining OS
  Abstractions for Modern Hardware**, Leis and Dietrich, PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p2115-leis.pdf`
  Why: DB/OS co-design direction for shrinking communication and isolation
  overhead while preserving hardware-backed boundaries.
- `reviewed` — **WeBridge: Synthesizing Stored Procedures for Large-Scale
  Real-World Web Applications**, Hu et al., PACMMOD 2024.
  URL: `https://dl.acm.org/doi/10.1145/3639319`
  PDF:
  `https://chuzhe.me/assets/pdf/2024%20-%20WeBridge-%20Synthesizing%20Stored%20Procedures%20for%20Large-Scale%20Real-World%20Web%20Applications.pdf`
  Why: stored-procedure synthesis for reducing client/server transaction
  round trips while keeping application logic maintainable.
- `reviewed` — **Practical DB-OS Co-Design with Privileged Kernel Bypass**,
  Zhou et al., SIGMOD 2025.
  URL: `https://dl.acm.org/doi/10.1145/3709714`
  PDF: `https://zxjcarrot.github.io/files/libdbos_SIGMOD25.pdf`
  Why: follow-up DB/OS kernel-bypass design from the Looking Glass 2.0 authors;
  directly relevant to low-overhead networking, IPC, and isolation boundaries.
- `reviewed` — **CockroachDB: The Resilient Geo-Distributed SQL Database**,
  Taft et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3386134`
  Why: Polaris cites CockroachDB as a practical system exposing transaction
  priorities; useful for contrasting record-level OCC priority with
  distributed SQL priority, contention handling, and admission behavior.
- `reviewed` — **GeoGauss: Strongly Consistent and Light-Coordinated OLTP for
  Geo-Replicated SQL Database**, Zhou et al., PACMMOD/SIGMOD 2023.
  URL: `https://arxiv.org/abs/2304.09692`
  DOI: `https://doi.org/10.1145/3588916`
  Why: modern geo-replicated SQL transaction design that compares against
  CockroachDB on TPC-C; useful for contrasting closed timestamps, leaseholder
  ownership, and distributed commit coordination with lighter geo-OLTP
  protocols.
- `reviewed` — **Chablis: Fast and General Transactions in Geo-Distributed
  Systems**, Eldeeb, Bernstein, Cidon, and Yang, CIDR 2024.
  URL: `https://www.vldb.org/cidrdb/papers/2024/p4-eldeeb.pdf`
  Why: discovered after the queue's strongest modern transaction follow-ups
  were either already reviewed or closed-access; useful for separating local
  write visibility from global retained-snapshot publication with
  epoch-based MVCC, regional publishers, leader-lease validation, and
  lock-free global snapshot reads. Journal entry added 2026-06-07.
- `reviewed` — **Chardonnay: Fast and General Datacenter Transactions for
  On-Disk Databases**, Eldeeb et al., OSDI 2023.
  URL: `https://www.usenix.org/conference/osdi23/presentation/eldeeb`
  PDF: `https://www.usenix.org/system/files/osdi23-eldeeb.pdf`
  Why: Chablis extends Chardonnay's single-datacenter epoch-based transaction
  design; useful for deeper mechanisms around local epoch services,
  lock-free strictly serializable snapshots, eRPC batching, and on-disk WAL
  integration before GPU DB adopts local visibility publishers. Journal entry
  added 2026-06-07.
- `reviewed` — **Q-Store: Distributed, Multi-partition Transactions via
  Queue-oriented Execution and Communication**, Qadah, Gupta, and Sadoghi,
  EDBT 2020.
  URL: `https://doi.org/10.5441/002/edbt.2020.08`
  PDF: `https://openproceedings.org/2020/conf/edbt/paper_39.pdf`
  Why: GeoGauss compares against Q-Store's deterministic queue-oriented
  distributed transaction processing; useful for evaluating whether GPU DB
  owner rings should become explicit operation queues for multi-partition
  transactions without forcing every route through a single global schedule.
- `reviewed` — **QueCC: A Queue-oriented, Control-free Concurrency
  Architecture**, Qadah and Sadoghi, Middleware 2018.
  URL: `https://doi.org/10.1145/3274808.3274810`
  PDF: `https://expolab.org/papers/quecc.pdf`
  Why: Q-Store's direct predecessor for queue-oriented concurrency inside a
  node; useful for comparing owner-domain operation queues, contention
  avoidance, and control-free execution before distributed queue routing.
- `reviewed` — **T-Part: Partitioning of Transactions for Forward-Pushing in
  Deterministic Database Systems**, Wu et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915227`
  PDF: `https://www.slmt.tw/papers/tpart.pdf`
  Why: Q-Store contrasts T-Part's dependency-graph scheduling with
  queue-oriented planning; useful for evaluating whether GPU DB should split
  hot write plans by dependency graph, fixed owner queue, or runtime conflict
  class.
- `reviewed` — **Predicate Transfer: Efficient Pre-Filtering on Multi-Join
  Queries**, Yang et al., CIDR 2024.
  URL: `https://www.cidrdb.org/cidr2024/papers/p22-yang.pdf`
  Why: modern predicate-transfer/pre-filtering work cited by the 2025 hybrid
  CPU-GPU paper; relevant to reducing over-resident transfer before GPU joins.
- `reviewed` — **Pushing Data-Induced Predicates Through Joins in Big-Data
  Clusters**, Kandula, Orr, and Chaudhuri, PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol13/p252-orr.pdf`
  DOI: `https://doi.org/10.14778/3368289.3368292`
  Why: predicate transfer cites it as a related predicate-through-join
  approach; useful for comparing runtime Bloom-filter transfer with
  statistics-driven plan-time data skipping.
- `reviewed` — **Free Join: Unifying Worst-Case Optimal and Traditional Joins**,
  Wang, Willsey, and Suciu, PACMMOD 2023.
  URL: `https://arxiv.org/abs/2301.10841`
  Author page: `https://www.mwillsey.com/papers/freejoin`
  Why: predicate transfer references modern worst-case-optimal join work; useful
  for deciding when GPU DB should keep binary joins plus filters versus expose
  a different multiway join route.
- `reviewed` — **Orchestrating data placement and query execution in
  heterogeneous CPU-GPU DBMS**, Yogatama et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p2491-yogatama.pdf`
  Why: cost-based CPU/GPU placement and execution orchestration for
  heterogeneous CPU-GPU database systems.
- `reviewed` — **Dynamic Resource Management for Efficient Utilization of
  Multitasking GPUs**, Park, Park, and Mahlke, ASPLOS 2017.
  URL: `https://doi.org/10.1145/3037697.3037707`
  PDF: `https://cccp.eecs.umich.edu/papers/jasonjk-asplos17.pdf`
  Why: GPU Maestro-style dynamic resource partitioning for multitasking GPUs;
  useful follow-up to kernel/batch concurrency scheduling for GPU DB streams.
- `reviewed` — **Classification-Driven Search for Effective SM Partitioning in
  Multitasking GPUs**, Zhao, Wang, and Eeckhout, ICS 2018.
  URL: `https://doi.org/10.1145/3205289.3205311`
  PDF: `https://users.elis.ugent.be/~leeckhou/papers/ics18.pdf`
  Why: low-overhead SM partitioning search for multitasking GPUs; relevant to
  GPU execution-owner admission, resident query co-scheduling, and fairness.
- `reviewed` — **Fast Equi-Join Algorithms on GPUs: Design and Implementation**,
  Rui and Tu, SSDBM 2017.
  URL: `https://doi.org/10.1145/3085504.3085521`
  PMC: `https://pmc.ncbi.nlm.nih.gov/articles/PMC10829000/`
  Why: modern-enough GPU join implementation paper cited by the concurrency
  study; useful for comparing route-specific kernels before concurrent
  co-scheduling.
- `reviewed` — **Distributed GPU Joins on Fast RDMA-capable Networks**,
  Thostrup et al., PACMMOD 2023.
  URL: `https://doi.org/10.1145/3588709`
  Why: follow-up GPU join work on scaling join state and data movement across
  RDMA-connected GPUs; relevant to future multi-device and network-aware
  route planning.
- `reviewed` — **Heterogeneous Intra-Pipeline Device-Parallel Aggregations**,
  Kroviakov et al., DaMoN 2024.
  URL: `https://doi.org/10.1145/3662010.3663441`
  PDF: `https://www-db.cs.tum.edu/~anneser/heterogeneous_aggregations.pdf`
  Why: recent aggregation work across CPU/GPU devices; useful follow-up for
  deciding when grouped aggregation stays on GPU, splits by fragment, or falls
  back to CPU under mixed route pressure.
- `reviewed` — **Everything is a Transaction: Unifying Logical
  Concurrency Control and Physical Data Structure Maintenance in
  Database Management Systems**, Zhang et al., CIDR 2021.
  URL:
  `https://www.vldb.org/cidrdb/2021/everything-is-a-transaction-unifying-logical-concurrency-control-and-physical-data-structure-maintenance-in-database-management.html`
  PDF: `https://db.cs.cmu.edu/papers/2021/cidr2021_paper06.pdf`
  Why: deferred-action framework that integrates physical maintenance
  with MVCC timestamps; useful for resident snapshot retirement, GPU
  buffer cleanup, index cleaning, and non-blocking layout changes.
- `reviewed` — **Scalable Garbage Collection for In-Memory MVCC Systems**,
  Bottcher et al., PVLDB 2019.
  URL: `https://dl.acm.org/doi/10.14778/3364324.3364328`
  Why: DAF cites it as a modern MVCC garbage-collection design; useful
  for long-reader robustness, version-chain cleanup, and cooperative
  cleanup benchmarks.
- `reviewed` — **Mainlining Databases: Supporting Fast Transactional
  Workloads on Universal Columnar Data File Formats**, Li et al.,
  PVLDB 2021.
  URL: `https://db.cs.cmu.edu/papers/2020/p534-li.pdf`
  Why: DAF's NoisePage context uses PAX/Arrow-like storage; this paper
  may inform GPU DB's CPU canonical layout, Arrow-compatible column
  groups, and transactional/analytical format choices.
- `reviewed` — **FASTER: A Concurrent Key-Value Store with In-Place
  Updates**, Chandramouli et al., SIGMOD 2018.
  URL: `https://www.microsoft.com/en-us/research/publication/faster-a-concurrent-key-value-store-with-in-place-updates/`
  Why: DAF compares against FASTER's epoch protection; useful for
  high-throughput hybrid log, session-visible epoch advancement, and
  read-cache/write-path tradeoffs.
- `reviewed` — **Accelerating GPU Data Processing using FastLanes
  Compression**, Afroozeh et al., DaMoN 2024.
  URL: `https://doi.org/10.1145/3662010.3663450`
  Why: modern GPU compressed-data execution follow-up for resident and
  over-resident compressed page experiments.
- `reviewed` — **Fast Serializable Multi-Version Concurrency Control for
  Main-Memory Database Systems**, Neumann et al., SIGMOD 2015.
  URL: `https://dl.acm.org/doi/10.1145/2723372.2749436`
  PDF: `https://www-db.cs.tum.edu/~muehlbau/papers/mvcc.pdf`
  Why: direct OMVCC baseline for transaction repair, with timestamp,
  validation, and version-chain design relevant to serializable MVCC in a
  memory-resident engine.
- `reviewed` — **Constant Time Recovery in Azure SQL Database**,
  Antonopoulos et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p2143-antonopoulos.pdf`
  Why: MD-MVCC depends on SQL Server's versioned recovery infrastructure for
  data-modifying schema changes; useful for GPU DB recovery frontiers,
  WAL-before-visibility, and bounded availability under long transactions.
- `reviewed` — **Exploiting Directly-Attached NVMe Arrays in DBMS**, Haas,
  Haubenschild, and Leis, CIDR 2020.
  URL: `https://www.cidrdb.org/cidr2020/papers/p16-haas-cidr20.pdf`
  Why: direct follow-up for explicit NVMe tier economics and high-parallelism
  IO paths that should inform GPU DB cold-partition and over-resident
  placement benchmarks.
- `reviewed` — **A Progress Report on DBOS: A Database-oriented Operating
  System**, Li et al., CIDR 2022.
  URL: `https://www.vldb.org/cidrdb/papers/2022/p26-li.pdf`
  Why: cloud database/OS co-design follow-up for treating scheduling,
  workflows, and system state as database-managed services.
- `reviewed` — **Why Files If You Have a DBMS?**, Nguyen and Leis, ICDE 2024.
  URL: `https://doi.org/10.1109/ICDE60146.2024.00297`
  PDF: `https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/blob.pdf`
  Why: storage-interface follow-up from the unikernel paper's related work;
  useful for evaluating file-system avoidance and DB-owned NVMe/cold-tier
  layouts.
- `reviewed` — **Optimizing Memory-mapped I/O for Fast Storage Devices**,
  Papagiannis et al., USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/papagiannis`
  PDF: `https://www.usenix.org/system/files/atc20-papagiannis.pdf`
  Why: OS-level mmap scalability work cited by the CIDR 2022 mmap paper; useful
  as a contrasting source on whether modified mmap paths can ever be safe or
  fast enough for GPU DB cold-tier experiments.
- `reviewed` — **Leveraging Lock Contention to Improve OLTP Application
  Performance**, Yan and Cheung, PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p444-yan.pdf`
  Why: program-analysis and contention-aware execution ideas that complement
  MV3C's dependency-annotated transaction repair path.
- `reviewed` — **The FastLanes Compression Layout: Decoding >100 Billion
  Integers per Second with Scalar Code**, Afroozeh and Boncz, PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf`
  DOI: `https://doi.org/10.14778/3598581.3598587`
  Why: source design for FastLanes' dependency-free column encodings; useful
  for deciding whether GPU DB resident segments should adopt interleaved
  bit-packing and cascaded encodings before GPU-specific kernels.
- `reviewed` — **Tile-Based Lightweight Integer Compression in GPU**,
  Shanbhag, Yogatama, Yu, and Madden, SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526132`
  PDF: `https://anilshanbhag.com/static/papers/gpufor_sigmod22.pdf`
  Why: direct GPU compression baseline compared by the FastLanes-GPU paper;
  useful for benchmarking tile granularity, global-memory traffic, and
  compression-versus-occupancy tradeoffs.
- `reviewed` — **The FastLanes File Format**, Afroozeh et al., PVLDB 2025.
  URL: `https://vldb.org/pvldb/vol18/p4629-afroozeh.pdf`
  Why: modern file-format follow-up that may connect GPU-friendly compressed
  vectors to disk/NVMe cold-tier layout and CPU/GPU shared data placement.
- `reviewed` — **No Cap, This Memory Slaps: Breaking Through the Memory
  Wall of Transactional Database Systems with Processing-in-Memory**,
  Kim et al., PVLDB 2025.
  URL: `https://www.pdl.cmu.edu/PDL-FTP/associated/p4241-kim.pdf`
  DOI: `https://doi.org/10.14778/3749646.3749690`
  Code: `https://github.com/hyoungjook/OLTPim`
  Why: modern OLTP near-data system that separates tuple payloads from
  pointer-chasing index and MVCC metadata; relevant to accelerator-side
  visibility summaries, rebuildable metadata, batching, and tier placement.
- `reviewed` — **PIM-Tree: A Skew-Resistant Index for
  Processing-in-Memory**, Kang et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol16/p946-kang.pdf`
  Why: OLTPim cites PIM-Tree as the skew-resistant alternative to its simpler
  hash/range partitioned PIM indexes; useful for GPU DB skew-aware resident
  key-vector and metadata placement.
- `reviewed` — **GaccO: A GPU-Accelerated OLTP DBMS**, Boeschen and Binnig,
  SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517876`
  Why: OLTPim contrasts prior GPU OLTP systems; useful follow-up for
  comparing GPU transaction batching and conflict handling with OLTPim-style
  near-data metadata placement.
- `reviewed` — **LTPG: Large-Batch Transaction Processing on GPUs with
  Deterministic Concurrency Control**, Wei et al., ICDE 2024.
  URL: `https://doi.org/10.1109/ICDE60146.2024.00296`
  PDF:
  `https://vbn.aau.dk/ws/portalfiles/portal/821323666/New_LTPG.pdf`
  Metadata:
  `https://vbn.aau.dk/en/publications/ltpg-large-batch-transaction-processing-on-gpus-with-deterministi`
  Why: modern GPU transaction-processing paper discovered while searching for
  underrepresented transaction/GPU concurrency work; relevant to deterministic
  GPU batches without predefined read/write sets. Journal entry exists from
  2026-06-04; the accessible author-manuscript URL above replaces the
  previously blocked `/files/` link.
- `reviewed` — **GPU-Accelerated OLTP: An In-Depth Analysis of Concurrency
  Control Schemes**, Sun et al., arXiv 2024/2026.
  URL: `https://arxiv.org/abs/2406.10158`
  Why: accessible modern GPU OLTP concurrency-control evaluation selected as a
  fallback after LTPG full-text retrieval was blocked; useful for warp/block
  launch tuning, GPU OCC/MVCC tradeoffs, latch-free metadata design, and
  conflict-resolution benchmark design.
- `reviewed` — **PLOR: General Transactions with Predictable, Low Tail Latency**,
  Chen et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517879`
  PDF: `https://storage.cs.tsinghua.edu.cn/papers/sigmod22plor.pdf`
  Why: cited by the GPU OLTP CC study as a hybrid pessimistic/optimistic
  concurrency-control direction; useful for tail-latency-aware retained reads
  and hot-write fallback lanes.
- `reviewed` — **A Study of the Fundamental Performance Characteristics of GPUs
  and CPUs for Database Analytics**, Shanbhag, Yu, and Madden, SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3380595`
  PDF: `https://anilshanbhag.com/static/papers/crystal_sigmod20.pdf`
  Why: Crystal's tile-based execution model is the execution substrate used by
  the SIGMOD 2022 GPU compression paper; useful for separating compression
  effects from baseline GPU query operator and memory-traffic behavior.
- `reviewed` — **Native Store Extension for SAP HANA**, Sherkat et al.,
  PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p2047-sherkat.pdf`
  DOI: `https://doi.org/10.14778/3352063.3352123`
  Why: selected after the last synthesis called for HTAP freshness and
  warm-tier DBMS designs; relevant to byte-compatible hot/warm column formats,
  load-unit placement, buffer-cache prefetch, page-level eviction, and
  advisor-driven tiering.
- `reviewed` — **Real-time Analytical Processing with SQL Server**, Larson et al.,
  PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol8/p1740-Larson.pdf`
  DOI: `https://doi.org/10.14778/2824032.2824071`
  Why: HANA NSE contrasts SQL Server's columnstore-on-OLTP approach; useful for
  comparing dual-store maintenance, operational analytics freshness, and write
  overhead against GPU DB resident snapshots.
- `reviewed` — **Data Blocks: Hybrid OLTP and OLAP on Compressed Storage using
  both Vectorization and Compilation**, Lang et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882925`
  Why: HANA NSE cites Data Blocks as a hybrid compressed-storage approach;
  useful for CPU/GPU shared compressed segments and fused vectorized execution.
  Journal entry exists from 2026-06-04; this stale duplicate was marked
  reviewed on 2026-06-05.
- `reviewed` — **Page As You Go: Piecewise Columnar Access in SAP HANA**,
  Sherkat et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2903729`
  Why: direct predecessor to HANA NSE's pageable column design; useful if the
  GPU DB needs more detail on piecewise dictionary/vector access and prefetch.
- `reviewed` — **Pipelined Query Processing in Coprocessor Environments**,
  Funke et al., SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3183734`
  PDF:
  `https://dbis.cs.tu-dortmund.de/storages/dbis-cs/r/papers/2018/pipelined-query-processing/pipelined-query-processing.pdf`
  Why: Crystal evaluates GPU-as-coprocessor query compilation against efficient
  CPU baselines; useful follow-up for deciding whether GPU DB should ever use
  pipelined transfer routes when data is not resident. Previously queued under
  the informal HorseQC name with an incorrect VLDB Journal DOI; corrected and
  reviewed on 2026-06-05.
- `reviewed` — **Relaxed Operator Fusion for In-Memory Databases: Making
  Compilation, Vectorization, and Prefetching Work Together at Last**,
  Menon et al., PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol11/p1-menon.pdf`
  Why: Crystal highlights CPU fusion/vectorization limits on chained
  operators; useful CPU-side baseline before claiming GPU route wins for fused
  retained query shapes.
- `reviewed` — **Hardware-Sensitive Scan Operator Variants for Compiled
  Selection Pipelines**, Broneske, Meister, and Saake, BTW 2017.
  URL: `https://dl.gi.de/items/f0e4190e-8c26-4d8d-a46d-63e8c2a04569`
  Handle: `https://dl.gi.de/handle/20.500.12116/642`
  PDF: `https://dl.gi.de/bitstreams/6553ed86-57b6-4dad-a5ab-f8d25c646f2e/download`
  Why: ROF cites this as related work on compiled scan variants; useful for
  tuning CPU fallback and warm-tier scan routes before routing work to GPU.
- `reviewed` — **One Loop Does Not Fit All**, Pantela and Idreos, SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2764944`
  PDF: `https://scholar.harvard.edu/files/stratos/files/oneloopdoesnotfitall.pdf`
  Why: ROF contrasts one-size-fits-all compiled loops with staged route shapes;
  useful for route-specific code generation and planner decisions. The previous
  queued DOI was corrected during the 2026-06-06 review.
- `reviewed` — **Hardware-Conscious Hash-Joins on GPUs**, Sioulas et al.,
  ICDE 2019.
  URL: `https://doi.org/10.1109/ICDE.2019.00068`
  Metadata: `https://www.eurecom.fr/en/publication/5780`
  PDF: `https://www.eurecom.fr/publication/5780/download/data-publi-5780.pdf`
  Why: direct follow-up to Fast Equi-Join that systematically evaluates
  partitioning, data location, and skew for GPU hash joins; useful for
  deciding when resident joins should use partitioned hash routes versus
  simpler non-partitioned kernels.
- `queued` — **Join Algorithms on GPUs: A Revisit After Seven Years**, Rui,
  Li, and Tu, IEEE Big Data Workshop 2015.
  PDF: `https://cse.usf.edu/~tuy/pub/BigData15-Join.pdf`
  Why: immediate predecessor to Fast Equi-Join that measures how older GPU
  join algorithms age across hardware generations; useful for separating
  hardware-refresh effects from algorithmic redesign in GPU DB benchmarks.
- `reviewed` — **Push vs. Pull-Based Loop Fusion in Query Engines**, Shaikhha,
  Dashti, and Koch, arXiv 2016 / Journal of Functional Programming 2018.
  arXiv: `https://arxiv.org/abs/1610.09166`
  DOI: `https://doi.org/10.1017/S0956796818000102`
  Why: direct follow-up for the One Loop and ROF route-shape thread; compares
  push and pull pipelining under fair query-compilation conditions and may help
  decide whether GPU DB's CPU fallback and retained routes should use push,
  pull, or stream-fusion-like generated pipelines.
- `queued` — **How to Architect a Query Compiler**, Shaikhha et al.,
  SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915244`
  Why: implementation context for DBLAB-style query compiler architecture used
  by the push/pull paper; useful for lowering route certificates into
  specialized CPU fallback and generated retained-route code without leaking
  abstractions into hot loops.
- `queued` — **Fast Queries over Heterogeneous Data Through Engine
  Customization**, Karpathiotakis, Alagiannis, and Ailamaki, PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p972-karpathiotakis.pdf`
  Why: cited as modern engine customization work; relevant to CPU/GPU route
  specialization when data lives in heterogeneous formats and tiers.
- `reviewed` — **Rethinking SIMD Vectorization for In-Memory Databases**,
  Polychroniou and Ross, SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2747645`
  Why: hardware-sensitive scan variants cite modern SIMD selection work; useful
  for calibrating CPU fallback, predicate-vector width, mask extraction, and
  Bloom-filter-style prefilter routes against GPU resident scans. Journal entry
  added 2026-06-05 using the accessible course PDF:
  `https://pages.cs.wisc.edu/~shivaram/cs744-readings/rethink-simd.pdf`.
- `reviewed` — **Everything You Always Wanted to Know About Compiled and
  Vectorized Queries But Were Afraid to Ask**, Kersten et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p2209-kersten.pdf`
  Why: follow-up on compiled versus vectorized CPU query execution; useful for
  deciding when GPU DB should rely on generated CPU fallback, vectorized warm
  scans, or fused retained GPU routes. Journal entry added 2026-06-05.
- `reviewed` — **High Performance Transactions via Early Write Visibility**,
  Faleiro, Abadi, and Hellerstein, PVLDB 2017.
  URL: `https://doi.org/10.14778/3055540.3055553`
  Why: QueCC discusses early write visibility as a way to reduce cascading
  abort and undo-buffer overhead; useful for evaluating when GPU DB can publish
  intra-batch writes before full transaction completion without weakening
  serializability or WAL-before-visibility.
- `reviewed` — **Design Principles for Scaling Multi-core OLTP Under High
  Contention**, Ren, Faleiro, and Abadi, SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882958`
  Why: ORTHRUS separates concurrency-control work from transaction execution;
  useful contrast to QueCC's no-control execution and GPU DB's owner-domain
  command rings under hot-key contention.
- `reviewed` — **Exploiting Single-Threaded Model in Multi-Core In-Memory
  Systems**, Yao et al., IEEE TKDE 2016.
  URL: `https://doi.org/10.1109/TKDE.2016.2578319`
  Why: QueCC contrasts LADS dependency-graph execution with priority queues;
  useful for deciding whether GPU DB should use dependency graphs, owner
  queues, or lighter runtime conflict classes for prepared multi-step writes.
- `reviewed` — **Improving High Contention OLTP Performance via Transaction
  Scheduling**, Prasaad, Cheung, and Suciu, arXiv 2018.
  URL: `https://arxiv.org/abs/1810.01997`
  Why: ORTHRUS sharpens the case for planned access and specialized
  concurrency-control work; this follow-up clusters conflict-free transactions
  before executing residual contended work and may inform GPU DB admission
  lanes for hot-key batches.
- `reviewed` — **Chiller: Contention-centric Transaction Execution and Data
  Partitioning for Modern Networks**, Zamanian et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3389724`
  Preprint: `https://arxiv.org/abs/1811.12204`
  Why: contention-aware partitioning and transaction execution for fast
  networks, relevant to partition owners, admission, and high-contention
  write-path routing.
- `reviewed` — **Datacenter RPCs can be General and Fast**, Kalia et al.,
  NSDI 2019.
  URL: `https://www.usenix.org/conference/nsdi19/presentation/kalia`
  Why: eRPC's session, message-buffer, congestion-control, and polling design
  is a direct follow-up for high-concurrency pgwire/network admission.
- `reviewed` — **Scalable RDMA RPC on Reliable Connection with Efficient
  Resource Sharing**, Chen et al., EuroSys 2019.
  URL: `https://doi.org/10.1145/3302424.3303968`
  Preprint: `https://chenyoumin1993.github.io/papers/eurosys19-scalerpc.pdf`
  Why: ScaleRPC-style resource sharing over RDMA connection state may inform
  future session multiplexing and bounded transport resource budgets.
- `reviewed` — **Carousel: Scalable Traffic Shaping at End Hosts**,
  Saeed et al., SIGCOMM 2017.
  URL: `https://doi.org/10.1145/3098822.3098852`
  Why: rate-limiter design used by eRPC; relevant to per-session admission,
  congestion shaping, and bounded response scheduling at high connection
  counts.
- `reviewed` — **TIMELY: RTT-based Congestion Control for the Datacenter**,
  Mittal et al., SIGCOMM 2015.
  URL: `https://doi.org/10.1145/2785956.2787510`
  Why: eRPC's congestion-control path builds on Timely; useful for deciding
  whether GPU DB network admission should use RTT/queue-delay telemetry.
- `reviewed` — **Programmable Packet Scheduling at Line Rate**, Sivaraman et al.,
  SIGCOMM 2016.
  URL: `https://doi.org/10.1145/2934872.2934899`
  PDF: `https://people.csail.mit.edu/alizadeh/papers/pifo-sigcomm16.pdf`
  Why: Carousel discusses PIFO as a programmable scheduling primitive; useful
  for comparing time-wheel admission with rank-based response/request
  scheduling when GPU DB needs more than simple pacing.
- `reviewed` — **Programmable Packet Scheduling with a Single Queue**,
  Yu et al., SIGCOMM 2021.
  URL: `https://doi.org/10.1145/3452296.3472887`
  PDF: `https://conferences.sigcomm.org/sigcomm/2021/files/papers/3452296.3472887.pdf`
  Why: modern follow-up on approximating programmable scheduling with a single
  FIFO-style queue; relevant to bounded high-concurrency response shaping
  without expensive per-session queues.
- `reviewed` — **Low-Latency Transaction Scheduling via Userspace
  Interrupts: Why Wait or Yield When You Can Preempt?**, Huang et al.,
  PACMMOD/SIGMOD 2025.
  URL: `https://doi.org/10.1145/3725319`
  PDF: `https://www2.cs.sfu.ca/~tzwang/preemptdb.pdf`
  Code: `https://github.com/sfu-dis/preemptdb`
  Why: modern best-paper transaction scheduling work using userspace
  interrupts and optimistic concurrency to preempt long low-priority
  transactions for short high-priority work; selected because recent synthesis
  called for more transaction/runtime papers after HTAP/GPU scheduling work.
- `reviewed` — **Concurrency Control as a Service**, Zhou et al.,
  PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p2761-zhou.pdf`
  DOI: `https://doi.org/10.14778/3746405.3746406`
  Code: `https://github.com/iDC-NEU/CCaaS`
  Why: modern disaggregated concurrency-control service with sharded
  multi-write OCC, epoch validation, deterministic conflict resolution, and
  asynchronous log pushdown; selected after the last synthesis called for more
  transaction/MVCC write-publication work.
- `reviewed` — **RCBench: an RDMA-enabled transaction framework for analyzing
  concurrency control algorithms**, Zhao et al., VLDB Journal 2023.
  URL: `https://doi.org/10.1007/s00778-023-00821-0`
  PDF: `https://link.springer.com/content/pdf/10.1007/s00778-023-00821-0.pdf`
  Code/PDF: `https://github.com/dbiir/RCBench`
  Why: CCaaS motivates independently scaled conflict-resolution resources;
  RCBench may provide a modern distributed/RDMA concurrency-control benchmark
  framework for comparing protocol scalability under data-node fan-out.
  Journal entry exists from 2026-06-05; the GitHub technical-report PDF was
  used after the Springer PDF endpoint returned an HTML access page.
- `reviewed` — **Epoxy: ACID Transactions Across Diverse Data Stores**,
  Kraft et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p2742-kraft.pdf`
  DOI: `https://doi.org/10.14778/3611479.3611484`
  Why: CCaaS contrasts Epoxy's MVCC control-panel approach; useful for
  comparing cross-store MVCC metadata, global snapshots, and atomic commit
  without forcing all stores to implement a 2PC participant protocol.
  Journal entry exists from 2026-06-05.
- `reviewed` — **Scalable Distributed Transactions across Heterogeneous
  Stores**, Dey, Fekete, and Rohm, ICDE 2015.
  URL: `https://doi.org/10.1109/ICDE.2015.7113278`
  Why: Epoxy compares against Cherry Garcia's key-value-oriented
  heterogeneous-store transaction protocol; useful for contrasting
  client-coordinated commit and minimal-store assumptions with Epoxy-style
  coordinator-owned global snapshots.
  Journal entry added 2026-06-05.
- `reviewed` — **ScalarDB: Universal Transaction Manager for Polystores**,
  Yamada et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p3768-yamada.pdf`
  DOI: `https://doi.org/10.14778/3611540.3611563`
  Why: modern production-oriented follow-up for Cherry Garcia and Epoxy-style
  transactions across heterogeneous stores; useful for comparing coordinator
  metadata, snapshot/serializable guarantees, and adapter requirements when GPU
  DB spans CPU truth, GPU resident state, and cold-tier services.
  Journal entry added 2026-06-05.
- `reviewed` — **Apache ShardingSphere: A Holistic and Pluggable Platform for
  Data Sharding**, Li et al., ICDE 2022.
  URL: `https://doi.org/10.1109/ICDE53745.2022.00231`
  Why: ScalarDB compares against XA-style middleware and references
  ShardingSphere's transaction-management ecosystem; useful for contrasting
  pluggable sharding/routing middleware with GPU DB's owner-domain route
  metadata and transaction context routing. Journal entry added 2026-06-06
  using the accessible SphereEx PDF:
  `https://download.sphere-ex.com/paper/a-holistic-and-pluggable-platform-for-data-sharding.pdf`.
- `queued` — **The BigDAWG Polystore System**, Duggan et al., SIGMOD Record
  2015.
  URL: `https://doi.org/10.1145/2814710.2814713`
  Why: ScalarDB frames modern polystores as a successor to earlier federated
  systems; BigDAWG is useful background for comparing islands, shims, and
  cross-engine routing when GPU DB grows CPU/GPU/cold-tier execution surfaces.
- `reviewed` — **Shinjuku: Preemptive Scheduling for Microsecond-scale Tail
  Latency**, Kaffes et al., NSDI 2019.
  URL: `https://www.usenix.org/conference/nsdi19/presentation/kaffes`
  PDF: `https://www.usenix.org/system/files/nsdi19-kaffes.pdf`
  Why: PreemptDB's related work contrasts Shinjuku's hardware-virtualization
  preemption for microsecond services; useful for comparing DB-internal
  preemption with runtime-level request preemption for mixed point/range query
  latency. Reviewed later in this queue.
- `reviewed` — **CoroBase: Coroutine-Oriented Main-Memory Database Engine**,
  He, Lu, and Wang, PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p431-he.pdf`
  arXiv: `https://arxiv.org/abs/2010.15981`
  Why: PreemptDB contrasts cooperative coroutine scheduling with preemptive
  scheduling; useful for evaluating whether GPU DB should use cooperative
  latency hiding for memory/NVMe stalls, preemption for urgent routes, or both.
- `reviewed` — **Asynchronous Memory Access Chaining**, Kocberber, Falsafi, and
  Grot, PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol9/p252-kocberber.pdf`
  DOI: `https://doi.org/10.14778/2850578.2850581`
  Why: CoroBase uses AMAC-style hand-crafted state machines as the upper-bound
  comparison for pointer-stall hiding; useful for deciding whether selected
  GPU DB metadata paths deserve explicit state-machine optimization instead
  of general coroutine machinery.
- `reviewed` — **Asynchronized Concurrency: The Secret to Scaling Concurrent
  Search Data Structures**, David, Guerraoui, and Trigonakis, ASPLOS 2015.
  URL: `https://doi.org/10.1145/2694344.2694359`
  Project: `https://dcl.epfl.ch/site/ascylib`
  Why: AMAC uses ASCYLIB's concurrent skip list workload; useful for comparing
  latch avoidance, search/update path simplification, and portable scalability
  in CPU-side indexes and route metadata structures.
- `queued` — **Improving Main Memory Hash Joins on Intel Xeon Phi Processors:
  An Experimental Approach**, Jha et al., PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol8/p642-Jha.pdf`
  Why: AMAC cites many-core hash-join tuning as related hardware-conscious
  database work; useful for separating CPU many-core memory-level parallelism
  from GPU/accelerator route choices when joins or grouped lookups spill to CPU.
- `queued` — **Robust Query Processing in Co-Processor-accelerated Databases**,
  Bress, Funke, and Teubner, SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882936`
  Why: cited by the GPU hash-join paper as a GPU co-processor DBMS baseline;
  useful for route robustness, placement decisions, and CPU/GPU fallback
  behavior under operator-at-a-time execution.
- `queued` — **An Experimental Comparison of Thirteen Relational Equi-Joins in
  Main Memory**, Schuh, Chen, and Dittrich, SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882917`
  Why: cited as a CPU join baseline source; useful for keeping GPU resident
  join claims honest against CPU warm-tier and fallback join implementations.
- `reviewed` — **Optimistic Concurrency with OPTIK**, Guerraoui and
  Trigonakis, PPoPP 2016.
  URL: `https://doi.org/10.1145/2851141.2851146`
  PDF: `https://infoscience.epfl.ch/record/217219/files/PPoPP16_OPTIK.pdf`
  Why: ASCY follow-up from the same group; useful for optimistic read/validate
  patterns in CPU-side route metadata, hot catalog maps, and lightweight
  retained-snapshot indexes.
- `reviewed` — **In the Search for Optimal Concurrency**, Gramoli et al.,
  SIROCCO 2016.
  URL:
  `http://sirocco2016.hiit.fi/preproceedings/In_the_Search_for_Optimal_Concurrency.pdf`
  Why: ASCY-related concurrency-optimality work; useful as a correctness
  counterweight when deciding whether simplified route metadata structures
  sacrifice valid concurrent schedules for speed.
- `queued` — **A Concurrency-Optimal List-Based Set**, Gramoli, Kuznetsov,
  Ravi, and Shang, DISC 2015 brief announcement / arXiv 2015.
  URL: `https://arxiv.org/abs/1502.01633`
  Why: implementation follow-up cited by the SIROCCO 2016 concurrency
  optimality paper; useful for seeing how semantic-aware validation plus
  restartability becomes an actual concurrent index/data-structure design.
- `reviewed` — **PathCAS: An Efficient Middle Ground for Concurrent Search Data
  Structures**, Brown, Sigouin, and Alistarh, PPoPP 2022.
  URL: `https://doi.org/10.1145/3503221.3508410`
  PDF: `https://research-explorer.ista.ac.at/download/11181/11731`
  Why: modern follow-up that combines multi-word CAS and transactional-memory
  ideas for concurrent search structures; useful for route metadata updates
  that need atomic multi-location publication without full STM overhead.
- `reviewed` — **Reuse, Don't Recycle: Transforming Lock-Free Algorithms That
  Throw Away Descriptors**, Arbel-Raviv and Brown, DISC 2017.
  URL: `https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.DISC.2017.4`
  PDF:
  `https://drops.dagstuhl.de/storage/00lipics/lipics-vol091-disc2017/LIPIcs.DISC.2017.4/LIPIcs.DISC.2017.4.pdf`
  Why: PathCAS relies on descriptor reuse to avoid descriptor allocation and
  reclamation overhead; useful for bounded per-worker command descriptors,
  route-publication descriptors, and lock-free helper metadata under high
  session counts.
- `reviewed` — **A Template for Implementing Fast Lock-free Trees Using HTM**,
  Brown, PODC 2017.
  URL: `https://arxiv.org/abs/1708.04838`
  PDF: `https://mc.uwaterloo.ca/pubs/3path/paper.podc17.pdf`
  Why: Reuse Don't Recycle notes that HTM can reduce descriptor allocation on
  fast paths but still needs an efficient lock-free fallback; useful for
  evaluating whether route metadata updates should use HTM as an optional fast
  path while keeping descriptor-reuse fallback progress.
- `reviewed` — **To Lock, Swap, or Elide: On the Interplay of Hardware
  Transactional Memory and Lock-Free Indexing**, Makreshanski, Levandoski, and
  Stutsman, PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol8/p1298-makreshanski.pdf`
  Why: Brown cites this database-index HTM/k-CAS work; useful for comparing
  HTM elision, lock-free Bw-tree-style indexing, and multi-word CAS as
  production route-metadata and CPU-index update options.
- `reviewed` — **SP-PIFO: Approximating Push-In First-Out Behaviors using
  Strict-Priority Queues**, Alcoz, Dietmuller, and Vanbever, NSDI 2020.
  URL: `https://www.usenix.org/conference/nsdi20/presentation/alcoz`
  Why: direct predecessor to AIFO that uses a small set of strict-priority
  queues to approximate PIFO; useful for comparing one-queue admission with
  multi-lane response scheduling.
- `reviewed` — **Swift: Delay is Simple and Effective for Congestion Control in
  the Datacenter**, Kumar et al., SIGCOMM 2020.
  URL: `https://doi.org/10.1145/3387514.3406591`
  Why: AIFO depends on fast-converging end-host congestion control; useful for
  mapping delay-based feedback to GPU DB ingress, response-ring, and tier
  admission signals.
- `reviewed` — **RTScan: Efficient Scan with Ray Tracing Cores**, Lv et al.,
  PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p1460-lv.pdf`
  DOI: `https://doi.org/10.14778/3648160.3648183`
  Why: RTCUDB cites RTScan as the main RT-core scan baseline; useful for
  isolating predicate-only ray tracing from RTCUDB's fused scan/group/aggregate
  mapping.
- `reviewed` — **RTIndex: Exploiting Hardware-Accelerated GPU Raytracing for
  Database Indexing**, Henneberg and Schuhknecht, arXiv 2023.
  URL: `https://arxiv.org/abs/2303.01139`
  Why: RTCUDB cites RTIndex as related RT-core indexing work; useful for
  evaluating whether resident equality/range indexes can map to BVH traversal
  without forcing the whole query into a ray-tracing job.
- `reviewed` — **DBOS: A DBMS-oriented Operating System**,
  Skiadopoulos et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p21-skiadopoulos.pdf`
  Why: full DBOS system paper with scheduler, file-system, and IPC experiments;
  useful follow-up for database-owned OS services, stored-procedure runtime
  boundaries, and transaction-backed system state.
- `reviewed` — **HopsFS: Scaling Hierarchical File System Metadata Using NewSQL
  Databases**, Niazi et al., FAST 2017.
  URL: `https://www.usenix.org/conference/fast17/technical-sessions/presentation/niazi`
  Why: DBOS cites DB-backed file-system metadata; relevant to GPU DB catalog,
  cold-tier namespace, and metadata-service scaling without bespoke
  filesystem-like state.
- `reviewed` — **KVell: The Design and Implementation of a Fast Persistent
  Key-Value Store**, Lepers et al., SOSP 2019.
  URL: `https://doi.org/10.1145/3341301.3359628`
  Why: persistent key-value design for high-throughput direct storage and
  multi-core request paths; useful as a contrast to FASTER's HybridLog for
  GPU DB cold-tier point lookups, log replay, and explicit IO ownership.
- `reviewed` — **FASTER: An Embedded Concurrent Key-Value Store for State
  Management**, Chandramouli et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p1930-chandramouli.pdf`
  Why: system/demo follow-up for FASTER as an embedded state store; useful for
  API, checkpoint, and workload-shaping context if the engine adopts
  HybridLog-like point-state structures.
- `reviewed` — **FileScale: Fast and Elastic Metadata Management for
  Distributed File Systems**, Liao and Abadi, SoCC 2023.
  URL: `https://doi.org/10.1145/3620678.3624784`
  PDF: `https://www.cs.umd.edu/~abadi/papers/filescale.pdf`
  Why: modern follow-up to database-backed file-system metadata that adds
  caching/routing layers to avoid synchronous database round trips while
  preserving distributed transaction support; useful for GPU DB catalog,
  route-cache, and cold-tier namespace scaling.
- `reviewed` — **BinDex: A Two-Layered Index for Fast and Robust Scans**,
  Li et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3380563`
  PDF: `https://kay21s.github.io/Bindex2020.pdf`
  Why: RTScan's main CPU/CUDA baseline; useful for comparing two-layer bitmap
  filtering, refinement cost, memory footprint, and update limitations against
  GPU-resident predicate-index experiments.
- `reviewed` — **A GPU Multiversion B-Tree**, Awad, Porumbescu, and Owens,
  PACT 2022.
  URL: `https://doi.org/10.1145/3559009.3569681`
  Why: RTIndeX compares against the Owens group GPU B+-tree line; a
  multiversion GPU tree is directly relevant to resident index snapshots,
  batched lookups, and update/version support that RTIndeX lacks.
- `reviewed` — **Learned Index on GPU**, Zhong et al., ICDE Workshops 2022.
  URL: `https://doi.org/10.1109/ICDEW55742.2022.00024`
  Why: RTIndeX identifies learned GPU indexes as a related accelerator-friendly
  route; useful for comparing BVH/RT-core lookup against model-guided
  resident lookup and route-cost calibration.
- `reviewed` — **Facebook's Tectonic Filesystem: Efficiency from Exascale**,
  Pan et al., FAST 2021.
  URL: `https://www.usenix.org/conference/fast21/presentation/pan`
  Why: FileScale contrasts Tectonic's sharded metadata/key-value approach with
  cross-partition transaction support; useful for comparing exascale storage
  metadata efficiency, placement, and bounded consistency tradeoffs.
- `reviewed` — **CalvinFS: Consistent WAN Replication and Scalable Metadata
  Management for Distributed File Systems**, Thomson and Abadi, FAST 2015.
  URL: `https://www.usenix.org/conference/fast15/technical-sessions/presentation/thomson`
  Why: FileScale cites CalvinFS as the deterministic-transaction metadata
  predecessor; useful for comparing route-cache WAL buffering with ordered
  transaction scheduling for namespace/catalog updates.
- `reviewed` — **ShardFS vs. IndexFS: Replication vs. Caching Strategies for
  Distributed Metadata Management in Cloud Storage Systems**, Xiao et al.,
  SoCC 2015.
  URL: `https://doi.org/10.1145/2806777.2806844`
  Why: FileScale contrasts namespace partitioning and caching approaches with
  distributed-transaction metadata; useful for evaluating route-cache
  placement, replication, and cache-invalidation options.
- `reviewed` — **SwitchFS: Asynchronous Metadata Updates for Distributed
  Filesystems with In-Network Coordination**, Xu et al., EuroSys 2026;
  arXiv 2024/2025.
  URL: `https://arxiv.org/abs/2410.08618`
  DOI: `https://doi.org/10.1145/3767295.3769349`
  Why: modern follow-up for metadata update admission and coordination;
  reviewed for evaluating whether cold-tier namespace, catalog, or route-cache
  mutations can be made asynchronous without weakening visible ordering.
- `reviewed` — **A Morsel-Driven Query Execution Engine for Heterogeneous
  Multi-Cores**, Dursun et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p2218-dursun.pdf`
  DOI: `https://doi.org/10.14778/3352063.3352137`
  Why: SiliconDB-style adaptive push scheduling and queueing model for
  heterogeneous processing units; useful for comparing fixed GPU/CPU fragment
  proportions with adaptive queue sizing under accelerator pressure.
- `reviewed` — **Shinjuku: Preemptive Scheduling for microsecond-scale Tail
  Latency**, Kaffes et al., NSDI 2019.
  URL: `https://www.usenix.org/conference/nsdi19/presentation/kaffes`
  PDF: `https://www.usenix.org/system/files/nsdi19-kaffes.pdf`
  Why: microsecond-scale request scheduling and preemption; relevant to
  separating short retained reads from long mutation, scan, or refresh work.
- `reviewed` — **CARPO: Leveraging Listwise Learning-to-Rank for
  Context-Aware Query Plan Optimization**, arXiv 2025.
  URL: `https://arxiv.org/abs/2509.03102`
  Why: modern listwise follow-up to Lero-style plan ranking; relevant to
  whether GPU DB route selection should rank candidate CPU/GPU/tiered plans as
  a set instead of pairwise comparisons only.
- `reviewed` — **ROME: Robust Query Optimization via Parallel Multi-Plan
  Execution**, Wei and Trummer, PACMMOD/SIGMOD 2024.
  URL: `https://doi.org/10.1145/3654973`
  PDF:
  `https://15799.courses.cs.cmu.edu/spring2025/papers/23-mongodb/wei-sigmod2024.pdf`
  Why: modern robust query-optimization work discovered because the remaining
  queued optimizer papers were older 2015-2016 sources; useful for bounded
  speculative CPU/GPU route insurance, route diversity, and loser-cancellation
  benchmarks under uncertain selectivity and transfer costs. Journal entry
  added 2026-06-06.
- `reviewed` — **OLTP Through the Looking Glass 16 Years Later:
  Communication is the New Bottleneck**, Zhou et al., CIDR 2025.
  URL:
  `https://www.vldb.org/cidrdb/2025/oltp-through-the-looking-glass-16-years-later-communication-is-the-new-bottleneck.html`
  PDF: `https://www.vldb.org/cidrdb/papers/2025/p17-zhou.pdf`
  Why: modern whole-stack OLTP breakdown showing communication and isolation
  costs as dominant bottlenecks; directly relevant to pgwire/session
  admission, stored-procedure boundaries, and owner/runtime queue design.
- `reviewed` — **Fast Failure Recovery for Main-Memory DBMSs on Multicores**,
  Wu et al., SIGMOD 2017.
  URL: `https://doi.org/10.1145/3035918.3064011`
  PDF: `https://yingjunwu.github.io/papers/sigmod2017.pdf`
  Why: command-log recovery with static and dynamic dependency analysis;
  useful for GPU DB WAL replay, checkpoint rebuild, and post-crash CPU/GPU
  cache warmup design.
- `reviewed` — **What Modern NVMe Storage Can Do, And How To Exploit It:
  High-Performance I/O for High-Performance Storage Engines**, Haas and Leis,
  PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p2090-haas.pdf`
  Why: LeanStore 2024 cites it as the direct NVMe IO-path study; useful for
  quantifying cold-tier queue depth, page size, SPDK/io_uring tradeoffs, and
  CPU-cycle budgets before GPU DB adopts explicit NVMe placement.
- `reviewed` — **Towards Buffer Management with Tiered Main Memory**, Hao et al.,
  PACMMOD 2024.
  URL: `https://doi.org/10.1145/3639286`
  DBLP: `https://dblp.org/rec/journals/pacmmod/HaoZYS24`
  Why: modern tiered-memory buffer management follow-up from LeanStore's
  related work; useful for extending GPU DB placement beyond DRAM/NVMe toward
  CXL, remote memory, or future intermediate tiers.
- `reviewed` — **RUMA has it: Rewired User-space Memory Access is Possible!**,
  Schuhknecht et al., PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p768-schuhknecht.pdf`
  Why: virtual-memory rewiring precursor to AnKerDB; useful for judging
  whether VM-assisted host snapshots or page remapping can support
  column-granular CPU/GPU snapshot publication without a patched kernel.
- `reviewed` — **Rethinking Serializable Multiversion Concurrency Control**,
  Faleiro and Abadi, PVLDB 2015.
  URL: `https://www.cs.umd.edu/~abadi/papers/rethink-mvcc.pdf`
  Why: BOHM decouples serialization/version management from transaction
  execution, a direct follow-up to MVCC version-chain and timestamp bottlenecks.
- `reviewed` — **Memory-Optimized Multi-Version Concurrency Control for
  Disk-Based Database Systems**, Freitag et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p2797-freitag.pdf`
  Why: memory-optimized MVCC in a disk-backed engine; relevant to version
  storage, undo chains, and cache-aware snapshot visibility.
- `reviewed` — **FOEDUS: OLTP Engine for a Thousand Cores and NVRAM**,
  Kimura, SIGMOD 2015.
  URL: `https://dl.acm.org/doi/10.1145/2723372.2746480`
  Tech report: `https://www.labs.hpe.com/techreports/2015/HPL-2015-37.pdf`
  Why: many-core OLTP and NVRAM-oriented storage architecture cited by TicToc;
  relevant to partition ownership, logging, NUMA locality, and future tiers.
- `reviewed` — **G-Learned Index: Enabling Efficient Learned Index on GPU**,
  Liu et al., IEEE TPDS 2024.
  URL: `https://doi.org/10.1109/TPDS.2024.3381214`
  Why: modern follow-up to early GPU learned-index work; useful for comparing
  PGM-on-GPU with a more engineered GPU learned-index design before choosing a
  resident point-lookup index family.
- `reviewed` — **The PGM-index: a fully-dynamic compressed learned index with
  provable worst-case bounds**, Ferragina and Vinciguerra, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p1162-ferragina.pdf`
  DOI: `https://doi.org/10.14778/3389133.3389135`
  Why: source design for the GPU-PGM paper; useful for understanding update,
  error-bound, and space guarantees before adapting a learned index to MVCC
  resident snapshots.
- `reviewed` — **Freely Moving Between the OLTP and OLAP Worlds**, Gubner et al.,
  PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p5113-gubner.pdf`
  Why: modern SQL Server/Azure SQL HTAP follow-up cited by the search path for
  SQL Server real-time analytics; useful for comparing newer hot/cold
  movement, analytical freshness, and operational workload isolation against
  the 2015 columnstore-on-OLTP design.
- `reviewed` — **PolarDB-IMCI: A Cloud-Native HTAP Database System at Alibaba**,
  Wang et al., SIGMOD 2023.
  URL: `https://arxiv.org/abs/2305.08468`
  PDF: `https://haozesong.github.io/data/sigmod23-polar.pdf`
  Why: Hermes compares against PolarDB-IMCI's cloud-native analytical
  accelerator; useful for contrasting row-id mapping, log replication, vector
  execution, and multi-node HTAP placement against a single-node accelerator.
- `reviewed` — **ByteHTAP: ByteDance's HTAP System with High Data Freshness and
  Strong Data Consistency**, Chen et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p3411-chen.pdf`
  DOI: `https://doi.org/10.14778/3554821.3554832`
  Why: Hermes directly compares its delta/main-store merge strategy with
  ByteHTAP; useful for evaluating shared-storage HTAP, freshness thresholds,
  delete bitmaps, and storage-layer pushdown against GPU DB retained snapshots.
- `reviewed` — **Towards Optimal Transaction Scheduling**, Cheng et al.,
  PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf`
  DOI: `https://doi.org/10.14778/3681954.3681956`
  Artifact: `https://github.com/audreyccheng/transaction-scheduling`
  Why: modern schedule-first transaction processing with hot-key prediction and
  MVTSO-derived operation-order enforcement; relevant to mutation-owner
  admission, hot-key write ordering, and contention-aware batching.
- `reviewed` — **Diva: Making MVCC Systems HTAP-Friendly**, Kim et al.,
  SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526135`
  Why: vDriver successor cited by the LeanStore paper; relevant to precise
  MVCC garbage collection and reducing long-reader damage in HTAP workloads.
- `reviewed` — **Virtual-Memory Assisted Buffer Management In Tiered Memory**,
  Rayhan and Aref, arXiv 2026.
  URL: `https://arxiv.org/abs/2603.03271`
  Why: extends vmcache-style virtual-memory-assisted buffer management to
  multiple memory tiers such as DRAM, remote memory/CXL-like tiers, and disk;
  directly relevant to future GPU DB host-tier and cold-partition placement.
- `queued` — **Citus: Distributed PostgreSQL for Data-Intensive Applications**,
  Cubukcu et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457551`
  Why: ShardingSphere compares against Citus as a PostgreSQL sharding
  middleware; useful follow-up for contrasting extension-level distributed
  planning, colocated joins, shard routing, and coordinator/worker execution
  with GPU DB route metadata and CPU/GPU/cold-tier ownership.
- `reviewed` — **PARQO: Penalty-Aware Robust Plan Selection in Query
  Optimization**, Xiu et al., PVLDB 2024.
  URL: `https://doi.org/10.14778/3704965.3704971`
  arXiv: `https://arxiv.org/abs/2406.01526`
  Why: robust plan selection with user-defined penalty functions and
  workload-informed selectivity-error models; useful for deciding when a
  fast GPU route is too fragile under uncertain cardinality, transfer, or
  queue-delay estimates.
- `reviewed` — **LOGER: A Learned Optimizer towards Generating Efficient and
  Robust Query Execution Plans**, Chen et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p1777-gao.pdf`
  Why: combines learned optimizer search with DBMS operator restrictions and
  beam search; relevant to adding learned GPU route suggestions without
  surrendering deterministic planner guardrails.
- `reviewed` — **Robust Query Processing in Co-Processor-Accelerated
  Databases**, Bress et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882936`
  Why: SiliconDB contrasts adaptive runtime scheduling with static
  co-processor decisions; useful for route robustness when GPU queue,
  transfer, or selectivity estimates are wrong.
- `reviewed` — **Pipelined Query Processing in Coprocessor Environments**,
  Funke et al., SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3183734`
  PDF:
  `https://dbis.cs.tu-dortmund.de/storages/dbis-cs/r/papers/2018/pipelined-query-processing/pipelined-query-processing.pdf`
  Why: SiliconDB cites it as coprocessor query processing background; relevant
  to deciding how much CPU/GPU pipeline state should be fused versus
  materialized across transfer and queue boundaries.
- `reviewed` — **Adaptive NUMA-Aware Data Placement and Task Scheduling for
  Analytical Workloads in Main-Memory Column-Stores**, Psaroudakis et al.,
  PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol10/p37-psaroudakis.pdf`
  Why: related NUMA placement and task scheduling work cited by SiliconDB;
  useful for CPU-side partition owners, route-local memory placement, and
  fallback scheduling when GPU work is saturated.
- `queued` — **Azure Data Lake Store: A Hyperscale Distributed File Service
  for Big Data Analytics**, Ramakrishnan et al., SIGMOD 2017.
  URL: `https://doi.org/10.1145/3035918.3056100`
  PDF: `https://www.cs.ucf.edu/~kienhua/classes/COP5711/Papers/MSazure2017.pdf`
  Why: Tectonic compares against ADLS's layered metadata approach; useful for
  contrasting range-partitioned metadata, relational/distributed-system
  integration, and data-lake scale metadata routing with Tectonic's
  hash-partitioned design.
- `reviewed` — **FITing-Tree: A Data-aware Index Structure**, Galakatos et al.,
  SIGMOD 2019.
  URL: `https://doi.org/10.1145/3299869.3319860`
  arXiv: `https://arxiv.org/abs/1801.10207`
  Why: bounded-error piecewise-linear learned-index predecessor to PGM; useful
  for comparing B-tree-indexed segments with fully learned recursive routing
  and for understanding update/retraining costs. Journal entry added
  2026-06-05.
- `reviewed` — **Design Tradeoffs of Data Access Methods**, Athanassoulis and
  Idreos, SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2912569`
  Why: FITing-Tree frames its error knob around the broader access-method
  tuning problem; useful for turning GPU DB resident index choices into
  explicit memory/latency/update tradeoff policies. Journal entry added
  2026-06-05.
- `reviewed` — **UpBit: Scalable In-Memory Updatable Bitmap Indexing**,
  Athanassoulis et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915964`
  Why: FITing-Tree discusses bitmap-index compression as related work; useful
  for comparing delete/update-friendly bitmap summaries against learned
  key-position models for GPU resident predicate filters. Journal entry added
  2026-06-06 using the accessible author PDF:
  `https://cs-people.bu.edu/mathan/publications/sigmod16-athanassoulis.pdf`.
- `reviewed` — **CUBIT: Concurrent Updatable Bitmap Indexing**, Wang and
  Athanassoulis, PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p399-athanassoulis.pdf`
  arXiv: `https://arxiv.org/abs/2410.16929`
  Artifact: `https://github.com/junchangwang/CUBIT`
  Why: modern concurrent successor to UpBit that adds real-time updates,
  lightweight snapshotting, wait-free queries, and latch-free consolidation;
  useful follow-up for GPU DB resident predicate indexes under concurrent
  mutation and retained snapshot reads. Journal entry added 2026-06-06.
- `queued` — **In-Place Updates in Tree-Encoded Bitmaps**, Weissenberger,
  Vaitl, and Markl, SSDBM 2022.
  URL: `https://doi.org/10.1145/3538712.3538745`
  Why: CUBIT cites tree-encoded bitmap updates as recent bitmap-index update
  work; useful for comparing structure-modifying compressed bitmap updates
  against CUBIT-style horizontal deltas and GPU DB generationed predicate
  masks.
- `reviewed` — **Harnessing Epoch-Based Reclamation for Efficient Range
  Queries**, Arbel-Raviv and Brown, PPoPP 2018.
  URL: `https://doi.org/10.1145/3178487.3178489`
  Why: CUBIT uses epoch/RCU-style reclamation ideas for bitmap snapshot
  versions; useful for retained-snapshot retirement, route metadata
  reclamation, and wait-free range/read paths under concurrent updates.
  Journal entry added 2026-06-06 using the author PDF:
  `https://www.cs.toronto.edu/~tabrown/ebrrq/paper.ppopp18.pdf`.
- `queued` — **Designing Access Methods: The RUM Conjecture**,
  Athanassoulis et al., EDBT 2016.
  URL: `https://doi.org/10.5441/002/edbt.2016.42`
  Author page:
  `https://stratos.seas.harvard.edu/publications/designing-access-methods-rum-conjecture`
  Why: direct formal source for the read-update-memory design-space model used
  by the SIGMOD access-method tutorial; useful if route certificates need a
  more precise RUM cost vocabulary.
- `queued` — **Pangea: Monolithic Distributed Storage for Data Analytics**,
  Ghosh et al., PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p681-ghosh.pdf`
  Why: Tectonic's related work contrasts monolithic analytics storage with
  layered filesystem designs; useful for evaluating whether GPU DB cold-tier
  placement should centralize data placement, caching, and failure recovery or
  keep them as explicit route-owned tiers.
- `reviewed` — **HSM: A Hybrid Slowdown Model for Multitasking GPUs**,
  Zhao, Jahre, and Eeckhout, ASPLOS 2020.
  URL: `https://users.elis.ugent.be/~leeckhou/papers/asplos2020.pdf`
  Why: modern follow-up from the same GPU multitasking line that models
  cross-kernel slowdown; relevant to route resource-class calibration and
  conservative GPU co-scheduling.
- `reviewed` — **EEMARQ: Efficient Lock-Free Range Queries with Memory
  Reclamation**, Sheffi, Ramalhete, and Petrank, arXiv 2022.
  URL: `https://arxiv.org/abs/2210.17086`
  Why: modern follow-up to epoch-based range queries that explicitly combines
  lock-free range queries and memory reclamation; useful for checking whether
  retained route metadata can avoid blocking reclamation while preserving
  linearizable/range-snapshot semantics. Journal entry added 2026-06-06.
- `reviewed` — **Constant-time Snapshots with Applications to Concurrent Data
  Structures**, Wei et al., PPoPP 2021.
  URL: `https://doi.org/10.1145/3437801.3441602`
  PDF: `https://www.cs.cmu.edu/~guyb/papers/3437801.3441602.pdf`
  Why: EEMARQ compares against vCAS and discusses the cost/benefit of
  per-field versioned CAS snapshots; useful for comparing descriptor-level
  versioning, indirection overhead, and constant-time snapshot mechanics before
  GPU DB chooses retained route metadata publication structures. Journal entry
  added 2026-06-06; the previously queued DOI suffix was corrected from
  `3441612` to `3441602`.
- `reviewed` — **VERLIB: Concurrent Versioned Pointers**, Blelloch and Wei,
  PPoPP 2024.
  URL: `https://doi.org/10.1145/3627535.3638501`
  PDF: `https://par.nsf.gov/servlets/purl/10539480`
  Artifact: `https://zenodo.org/records/10447617`
  Why: direct successor to constant-time snapshots that removes much of the
  recorded-once indirection constraint and packages versioned pointers as a C++
  library; useful before choosing GPU DB's retained route metadata and
  resident-index publication primitives. Journal entry added 2026-06-06.
- `reviewed` — **Multiverse: Transactional Memory with Dynamic
  Multiversioning**, Coccimiglio, Brown, and Ravi, arXiv/PPoPP 2026.
  URL: `https://arxiv.org/abs/2601.09735`
  Why: discovered while reviewing VERLIB; useful for comparing dynamic
  versioned/unversioned transaction modes with GPU DB's need to keep common
  short writes cheap while still supporting long retained reads and
  snapshot-heavy metadata scans. Journal entry added 2026-06-06 from arXiv
  v4.
- `queued` — **TLF: Transactional Lock Fusion**, Blelloch, Kent, and Wei,
  SPAA 2025.
  DOI: `https://doi.org/10.1145/3694906.3743341`
  Why: Multiverse cites TLF as the multiversion STM line built on Verlib;
  useful for comparing fused transactional locking and versioned-pointer
  metadata before GPU DB chooses publication primitives for retained route
  metadata.
- `queued` — **Scaling Up Transactions with Slower Clocks**, Ramalhete and
  Correia, PPoPP 2024.
  DOI: `https://doi.org/10.1145/3627535.3638477`
  Why: Multiverse uses DCTL as its fast unversioned STM baseline; useful for
  contrasting global-clock reduction, encounter-time locking, and starvation
  fallback with GPU DB's compact unversioned route metadata cells.
- `reviewed` — **VBR: Version Based Reclamation**, Sheffi et al., arXiv 2021.
  URL: `https://arxiv.org/abs/2107.13843`
  Why: optimistic memory reclamation scheme related to EBR/hazard-pointer
  tradeoffs; useful for deciding whether route metadata and resident-index
  descriptors can reclaim aggressively without global epoch stalls. Reviewed
  on 2026-06-06 using the arXiv PDF.
- `queued` — **Predicting and reining in application-level slowdown on
  spatial multitasking GPUs**, Wei et al., JPDC 2020.
  URL: `https://doi.org/10.1016/j.jpdc.2020.03.009`
  PDF: `https://www.cs.sjtu.edu.cn/~leng-jw/resources/Files/wzhao_ipdps19.pdf`
  Why: HSM compares against Themis/KSM-style neural slowdown prediction;
  useful follow-up for contrasting low-counter hybrid models with learned
  slowdown predictors and SM-allocation engines for GPU execution owners.
- `queued` — **The Processing-in-Memory Model**, Kang et al., SPAA 2021.
  URL: `https://doi.org/10.1145/3409964.3461806`
  arXiv: `https://arxiv.org/abs/2105.04305`
  Why: source model used by PIM-Tree for host/PIM work, depth, IO rounds, and
  communication analysis; useful for defining future-tier cost metrics for
  accelerator-side metadata placement.
- `queued` — **Concurrent Data Structures with Near-Data-Processing: an
  Architecture-Aware Implementation**, Choe et al., SPAA 2019.
  URL: `https://doi.org/10.1145/3323165.3323200`
  Why: PIM-Tree contrasts prior range-partitioned near-data ordered indexes;
  useful for understanding when simple per-tier range partitioning fails under
  skew and how much explicit rebalancing or route fallback GPU DB needs.
- `reviewed` — **NUBA: Non-Uniform Bandwidth GPUs**, Zhao et al., ASPLOS 2023.
  URL: `https://doi.org/10.1145/3575693.3575745`
  PDF: `https://users.elis.ugent.be/~leeckhou/papers/ASPLOS_2023.pdf`
  Why: newer off-chip/on-chip bandwidth-aware GPU architecture work from the
  CD-search authors; relevant to treating GPU bandwidth locality and
  partitioning as route-certificate inputs for future hardware.
- `reviewed` — **Oze: Decentralized Graph-Based Concurrency Control for
  Long-Running Update Transactions**, Nemoto et al., PVLDB 2025.
  URL: `https://vldb.org/pvldb/vol18/p2321-nemoto.pdf`
  Why: modern multi-version serialization-graph concurrency control for
  heterogeneous long/short transaction workloads; relevant to dependency
  tracking, false-positive conflict reduction, and long retained refresh or
  write transactions.
- `reviewed` — **Shirakami: A Hybrid Concurrency Control Protocol for Tsurugi
  Relational Database System**, Tanabe et al., arXiv 2026.
  URL: `https://arxiv.org/abs/2303.18142`
  Why: production-oriented hybrid protocol combining long-transaction MVCC and
  short-transaction OCC, cited in the same Tsurugi/RSS research ecosystem and
  relevant to GPU DB mixed OLTP/analytical ownership boundaries.
- `reviewed` — **Aria: A Fast and Practical Deterministic OLTP Database**, Lu,
  Yu, Cao, and Madden, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p2047-lu.pdf`
  Why: deterministic OLTP execution without a global serial schedule bottleneck;
  useful as a contrast point for Oze, mutation batching, and whether
  predeclared GPU DB write/read sets can improve concurrency without forcing
  long retained refresh work to stall short transactions.
- `reviewed` — **Arachne: Core-Aware Thread Management**, Qin et al.,
  OSDI 2018.
  URL: `https://www.usenix.org/conference/osdi18/presentation/qin`
  Why: user-level core-aware thread management cited by Caladan; useful for
  exposing internal request concurrency to a scheduler without adopting a full
  Caladan-style interference-control stack.
- `queued` — **Efficient Dynamic Resource Management for Spatial Multitasking
  GPUs**, Zhu et al., IEEE Transactions on Cloud Computing 2024.
  URL: `https://doi.org/10.1109/TCC.2024.3511548`
  Why: modern spatial-multitasking resource allocation follow-up to GPU
  Maestro-style GPU sharing; useful for comparing software-visible GPU
  partition policy with current hardware and cloud scheduling assumptions.
- `reviewed` — **Towards Efficient and Practical GPU Multitasking in the Era of
  LLM**, arXiv 2025.
  URL: `https://arxiv.org/abs/2508.08448`
  Why: recent GPU multitasking position paper that surveys sharing, isolation,
  and resource-management requirements; useful for checking whether database
  GPU owners should expose a GPU-OS-like control plane before adopting
  hardware-specific scheduling assumptions.
- `reviewed` — **ZygOS: Achieving Low Tail Latency for Microsecond-scale
  Networked Tasks**, Prekas, Kogias, and Bugnion, SOSP 2017.
  URL: `https://dl.acm.org/doi/10.1145/3132747.3132780`
  PDF: `https://marioskogias.github.io/docs/zygos.pdf`
  Why: work-conserving dataplane scheduler for high-connection-count
  microsecond services, including Silo/TPC-C evaluation; relevant to pgwire
  IO-worker and request-stealing choices.
- `reviewed` — **Pasha: An Efficient, Scalable Database Architecture for CXL
  Pods**, Huang et al., CIDR 2025.
  URL: `https://www.vldb.org/cidrdb/papers/2025/p8-huang.pdf`
  Why: CXL-pod database architecture cited by vmcache^n; relevant to
  disaggregated/tiered memory placement and future host-memory expansion.
- `reviewed` — **Resource-Adaptive Query Execution with Paged Memory
  Management**, Otaki, Benello, Elmore, and Graefe, CIDR 2025.
  URL: `https://www.vldb.org/cidrdb/papers/2025/p2-otaki.pdf`
  Why: paged-memory and resource-adaptive execution work cited by vmcache^n;
  relevant to query admission and execution under memory-tier pressure.
- `reviewed` — **PowerTCP: Pushing the Performance Limits of Datacenter
  Networks**, Addanki, Michel, and Schmid, NSDI 2022.
  URL: `https://www.usenix.org/conference/nsdi22/presentation/addanki`
  arXiv: `https://arxiv.org/abs/2112.14309`
  Why: modern congestion-control follow-up that combines queue length and
  queue-change signals; useful for comparing TIMELY-style delay gradients with
  request/response ring delay and queue-depth admission.
- `reviewed` — **FNCC: Fast Notification Congestion Control in Data Center
  Networks**, arXiv 2024.
  URL: `https://arxiv.org/abs/2405.07608`
  Why: sub-RTT congestion notification via ACK-carried telemetry; relevant to
  future high-concurrency response shaping if GPU DB exposes fast path
  queue-delay hints from IO workers or gateways.
- `reviewed` — **Nomad: Non-Exclusive Memory Tiering via Transactional Page
  Migration**, Xiang et al., OSDI 2024.
  URL: `https://www.usenix.org/conference/osdi24/presentation/xiang`
  Why: transactional page migration for tiered memory cited by vmcache^n;
  useful for comparing OS-assisted migration against explicit DBMS ownership.
- `reviewed` — **Improving Optimistic Concurrency Control Through Transaction
  Batching and Operation Reordering**, Ding, Kot, and Gehrke, PVLDB 2018.
  URL: `https://doi.org/10.14778/3282495.3282502`
  PDF: `https://dl.acm.org/doi/pdf/10.14778/3282495.3282502`
  Why: modern follow-up for QURO-style reordering at storage and validation
  stages under OCC; useful for comparing application-level query ordering
  with engine-owned micro-batching and dependency-aware validation.
- `reviewed` — **Counting Is All You Need for Instant Tuple Discovery:
  Enabling Real-Time HTAP in Standalone DBMSs**, Lim et al., PACMMOD 2025.
  URL: `https://doi.org/10.1145/3769775`
  Why: modern follow-up from the same HTAP/MVCC research ecosystem; tuple
  discovery and incremental transformation may inform retained snapshot
  refresh, generation directories, and row-to-column GPU resident builds.
- `reviewed` — **Towards Buffer Management with Tiered Main Memory**, Hao et al.,
  PACMMOD/SIGMOD 2024.
  URL: `https://doi.org/10.1145/3639286`
  Why: modern tiered-main-memory buffer management cited by vmcache^n;
  relevant to DRAM/remote-memory/NVMe policy design and placement economics.
- `reviewed` — **TiQuE: Improving the Transactional Performance of Analytical
  Systems for True Hybrid Workloads**, PACMMOD 2023.
  URL: `https://doi.org/10.14778/3598581.3598598`
  Why: cited by Counting Is All You Need; relevant to comparing analytical
  system retrofits against a standalone transactional engine with retained
  transformation and snapshot-refresh metadata.
- `reviewed` — **Two is Better Than One: The Case for 2-Tree for Skewed Data
  Sets**, Zhou, Yu, Graefe, and Stonebraker, CIDR 2023.
  URL: `https://www.cidrdb.org/cidr2023/papers/p57-zhou.pdf`
  Code: `https://github.com/zxjcarrot/2-Tree`
  Why: direct predecessor to the Three-Tree/tiered-buffer-pool design; useful
  for record-level hot/cold migration, low-cost access statistics, and
  preserving range-scan behavior while separating hot and cold index records.
- `reviewed` — **Hermes: Off-the-Shelf Real-Time Transactional Analytics**,
  Milkai et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p2334-milkai.pdf`
  DOI: `https://doi.org/10.14778/3742728.3742731`
  Why: same HTAP/transactional-analytics ecosystem; useful for comparing
  instant tuple discovery with off-the-shelf real-time analytical routing and
  freshness/overhead tradeoffs.
- `reviewed` — **PAR2QO: Parametric Penalty-Aware Robust Query Optimization**,
  Xiu et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4532-xiu.pdf`
  DOI: `https://doi.org/10.14778/3749646.3749711`
  Why: follow-up to PARQO that focuses on parametric robust query
  optimization and plan-penalty profile caching; relevant to repeated retained
  GPU route templates and admission-time route reuse.
- `queued` — **Griffin: Hardware-Software Support for Efficient Page Migration
  in Multi-GPU Systems**, Baruah et al., HPCA 2020.
  URL: `https://doi.org/10.1109/HPCA47549.2020.00055`
  Why: NUBA contrasts page migration with LAB/MDR placement; useful for
  evaluating when GPU DB should migrate, replicate, or rebuild resident
  segment pages across future multi-GPU and partitioned-memory hardware.
- `reviewed` — **Locality-Centric Data and Threadblock Management for Massive
  GPUs**, Khairy et al., MICRO 2020.
  URL: `https://doi.org/10.1109/MICRO50266.2020.00086`
  PDF:
  `https://d1qx31qr3h6wln.cloudfront.net/publications/MICRO_2020_Threadblock_Management.pdf`
  Why: NUBA cites locality-centric data/threadblock management for massive GPU
  locality; useful for mapping query fragments, CTAs, and resident segment
  placement to hardware-local partitions instead of relying only on generic
  GPU scheduling. Journal entry added 2026-06-05.
- `reviewed` — **Concurrency Control as a Service**, Zhou et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p2761-zhou.pdf`
  DOI: `https://doi.org/10.14778/3746405.3746406`
  Why: modern execution-CC-storage disaggregation with sharded multi-write OCC,
  asynchronous log push-down, and independently scalable conflict-resolution
  resources; directly relevant to mutation-owner decomposition and write-path
  admission.
- `reviewed` — **Modeling Concurrency Control as a Learnable Function**,
  Pan et al., arXiv 2026.
  URL: `https://arxiv.org/abs/2503.10036`
  Why: modern learned concurrency-control design selected after the queue had
  no stronger queued 2023-present OLTP/concurrency paper that improved recent
  category balance; useful for operation-level conflict-action tables,
  workload drift handling, stored-procedure versus interactive transaction
  policy splits, and hot-write admission benchmarks.
- `reviewed` — **A Hybrid Approach to Integrating Deterministic and
  Non-deterministic Concurrency Control in Database Systems**, Hong et al.,
  PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p1376-lu.pdf`
  DOI: `https://doi.org/10.14778/3718057.3718066`
  Why: modern hybrid deterministic/OCC design with global validation and
  logging integration; useful for deciding whether GPU DB write batches should
  route through deterministic or optimistic lanes by workload.
- `reviewed` — **Towards Optimal Transaction Scheduling**, Cheng et al.,
  PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf`
  DOI: `https://doi.org/10.14778/3681954.3681956`
  Why: schedule-first concurrency-control approach with fine-grained operation
  ordering; useful for comparing engine-owned transaction scheduling against
  route-level micro-batching and conflict-aware admission.
- `reviewed` — **Fluid Co-processing: GPU Bloom-filters for CPU Joins**,
  Gubner et al., DaMoN 2019.
  URL: `https://doi.org/10.1145/3329785.3329934`
  PDF: `https://t1mm3.github.io/assets/papers/damon19.pdf`
  Why: modern follow-up to robust co-processor placement that uses GPU work as
  selective early pruning for CPU joins; relevant to split CPU/GPU fragments,
  predicate-transfer placement, and transfer-aware route budgets.
- `queued` — **Architecting a Pluggable Query Executor for Emerging
  Co-Processors**, Gurumurthy, PhD thesis 2024.
  URL: `https://opendata.uni-halle.de//handle/1981185920/117483`
  DOI: `https://doi.org/10.25673/115529`
  Why: recent co-processor query executor design that decomposes operators into
  reusable task/device layers; useful for deciding whether GPU DB route
  descriptors should expose device-specific kernels, portable fragments, or a
  layered execution abstraction.
- `reviewed` — **F1 Lightning: HTAP as a Service**, Yang et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p3313-yang.pdf`
  DOI: `https://doi.org/10.14778/3415478.3415553`
  Why: loosely coupled production HTAP system with transparent query federation
  over existing transactional stores; useful follow-up to PolarDB-IMCI for
  contrasting redo-replay replicas with service-layer freshness and route
  integration. Journal entry added 2026-06-05.
- `reviewed` — **TiDB: A Raft-based HTAP Database**, Huang et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p3072-huang.pdf`
  DOI: `https://doi.org/10.14778/3415478.3415535`
  Why: Raft learner based row-to-column HTAP replication; useful follow-up to
  PolarDB-IMCI for comparing physical REDO reuse with consensus-log columnar
  replicas, freshness, consistency, and workload isolation. Journal entry
  added 2026-06-05.
- `reviewed` — **Hints for Robust Query Performance Tuning**, Xiu et al.,
  PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p5327-xiu.pdf`
  Why: PARQO-adjacent robust tuning work that may turn sensitive cardinality
  dimensions into actionable hints; useful for exposing why a GPU route,
  fallback route, or refresh decision is fragile.
- `reviewed` — **Kepler: Robust Learning for Faster Parametric Query
  Optimization**, Doshi et al., PACMMOD/SIGMOD 2023.
  URL: `https://arxiv.org/abs/2306.06798`
  Why: robust parametric query optimization using executed-query evidence;
  useful as a contrast to PARQO's cost-model-based route cache for repeated
  SQL templates.
- `reviewed` — **RankPQO: Learning-to-Rank for Parametric Query Optimization**,
  Mo et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p863-mo.pdf`
  Why: modern parametric query optimization follow-up cited around Kepler-style
  route selection; useful for comparing fastest-plan classification against
  ranking candidate CPU/GPU/tier routes.
- `reviewed` — **Plor: General Transactions with Predictable, Low Tail
  Latency**, Chen et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517879`
  PDF: `https://storage.cs.tsinghua.edu.cn/papers/sigmod22plor.pdf/`
  Why: Shirakami cites Plor as modern transaction scheduling work; relevant to
  predictable low-tail mutation and admission paths under mixed transaction
  sizes.
- `reviewed` — **Bf-Tree: A Modern Read-Write-Optimized Concurrent
  Larger-Than-Memory Range Index**, Hao and Chandramouli, PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p3442-hao.pdf`
  Why: variable-length mini-pages decouple cached hot records and update
  buffers from disk pages; relevant to GPU DB cold-partition index pages and
  host/NVMe cache granularity.
- `reviewed` — **Query Fresh: Log Shipping on Steroids**, Wang, Johnson, and
  Pandis, PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol11/p406-wang.pdf`
  Why: DANA cites RDMA log shipping as an alternative low-latency WAL path;
  useful for comparing local persistent-log buffers, remote durable logging,
  and WAL-before-visibility tradeoffs.
- `reviewed` — **Low-Latency Transaction Scheduling via Userspace Interrupts:
  Why Wait or Yield When You Can Preempt?**, Huang et al.,
  PACMMOD/SIGMOD 2025.
  URL: `https://doi.org/10.1145/3725319`
  Why: modern userspace-interrupt transaction scheduling cited by Shirakami;
  relevant to preempting or isolating long mutation/refresh work from short
  retained reads.
- `reviewed` — **Massively Parallel Multi-Versioned Transaction Processing**,
  Qian and Goel, OSDI 2024.
  URL: `https://www.usenix.org/conference/osdi24/presentation/qian`
  Why: Shirakami cites this modern MVCC transaction-processing work; relevant
  to high-core-count versioned execution and future owner/partition scaling.
- `reviewed` — **GaccO - A GPU-accelerated OLTP DBMS**, Boeschen and Binnig,
  SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517876`
  Why: GPU single-version deterministic locking baseline compared by Epic;
  relevant to deciding when commutative GPU updates beat general MVCC.
- `reviewed` — **Caracal: Contention Management with Deterministic Concurrency
  Control**, Qin, Brown, and Goel, SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483591`
  PDF: `https://www.eecg.utoronto.ca/~ashvin/publications/caracal.pdf`
  Why: deterministic MVCC contention-management baseline for Epic; useful for
  CPU-side owner/partition batching and skewed write-set planning.
- `reviewed` — **High Performance Transactions via Early Write Visibility**,
  Faleiro, Abadi, and Hellerstein, PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p613-faleiro.pdf`
  Why: deterministic transaction protocol cited by Epic; relevant to exposing
  writes early inside an ordered batch without violating external visibility.
- `reviewed` — **Rethinking Logging, Checkpoints, and Recovery for
  High-Performance Storage Engines**, Haubenschild et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3389716`
  Why: decentralized logging and checkpointing foundation used by Umbra;
  relevant to WAL-before-visibility, batch commit, replay, and separating
  durable authority from rebuildable GPU residency state.
- `reviewed` — **Understanding Manycore Scalability of File Systems**,
  Min et al., USENIX ATC 2016.
  URL:
  `https://www.usenix.org/conference/atc16/technical-sessions/presentation/min`
  Why: DBOS cites file-system global locks and manycore scalability as a
  storage-service bottleneck; useful for comparing DB-owned cold-tier metadata,
  partitioned resident-placement state, and kernel/filesystem contention risks.
- `reviewed` — **Scalable Garbage Collection for In-Memory MVCC Systems**,
  Boettcher et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol13/p128-bottcher.pdf`
  Why: Steam-style MVCC version garbage collection cited by the Umbra MVCC
  paper; relevant to bounded version retention, long retained snapshots, and
  per-owner GC without global contention.
- `queued` — **Beyond the Socket: NUMA-Aware GPUs**, Milic et al.,
  MICRO 2017.
  URL: `https://doi.org/10.1145/3123939.3124534`
  PDF:
  `https://research.nvidia.com/sites/default/files/pubs/2017-10_Beyond-the-socket%3A/milic_micro17.pdf`
  Why: direct predecessor to LADM on multi-socket NUMA GPU interconnect,
  cache, and phase-aware policy; useful for comparing hardware-visible
  locality controls with route-level resident segment placement.
- `queued` — **MCM-GPU: Multi-Chip-Module GPUs for Continued Performance
  Scalability**, Arunkumar et al., ISCA 2017.
  URL: `https://doi.org/10.1145/3079856.3080231`
  PDF:
  `https://research.nvidia.com/sites/default/files/publications/ISCA_2017_MCMGPU.pdf`
  Why: source architecture for chiplet-style GPU NUMA locality and inter-GPM
  traffic reduction; useful for future multi-GPU/chiplet residency placement
  and deciding when GPU DB should migrate, replicate, or schedule near data.
- `reviewed` — **BTrim - Hybrid In-Memory Database Architecture for Extreme
  Transaction Processing in VLDBs**, Gurajada et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p1889-gurajada.pdf`
  Why: hybrid disk/in-memory transactional architecture cited by the Umbra
  MVCC paper; useful as a contrast point for hot working-set placement and
  contention reduction across CPU memory and durable storage.
- `queued` — **SpanFS: A Scalable File System on Fast Storage Devices**,
  Kwon et al., USENIX ATC 2015.
  URL: `https://www.usenix.org/conference/atc15/technical-session/presentation/kwon`
  Why: manycore file-system scalability follow-up using partitioning to reduce
  lock contention; useful for comparing partitioned cold-tier metadata and
  resident-placement ownership against locality loss.
- `reviewed` — **Eiffel: Efficient and Flexible Software Packet Scheduling**,
  Saeed et al., NSDI 2019.
  URL: `https://www.usenix.org/conference/nsdi19/presentation/saeed`
  Why: SP-PIFO cites Eiffel as an alternative programmable scheduling design;
  useful for comparing strict-priority queue approximation with software
  integer-priority queues for request/response scheduling.
- `queued` — **Cost/Performance in Modern Data Stores: How Data Caching
  Systems Succeed**, Lomet, DaMoN 2018.
  URL: `https://doi.org/10.1145/3211922.3211931`
  Why: economic and architectural argument cited by Umbra against pure
  in-memory-only systems; useful for setting GPU DB tier-placement budgets and
  cost/performance targets across HBM, DRAM, NVMe, and future memory tiers.
- `queued` — **Locality-aware Partitioning in Parallel Database Systems**,
  Zamanian et al., SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2749437`
  Why: cited by the adaptive NUMA placement paper for copartitioned joins and
  workload-aware partitioning; useful for comparing static locality-aware
  partition plans with GPU DB's runtime route and placement telemetry.
- `queued` — **Scaling Up Mixed Workloads: A Battle of Data Freshness,
  Flexibility, and Scheduling**, Psaroudakis et al., TPCTC 2015.
  URL: `https://doi.org/10.1007/978-3-319-28186-9_7`
  Why: related SAP HANA scheduling work from the same research line; useful
  for mixed OLTP/OLAP freshness, scheduler, and admission tradeoffs before
  GPU DB combines retained snapshots with writes.
- `reviewed` — **Adaptive Execution of Compiled Queries**, Kohn, Leis, and
  Neumann, ICDE 2018.
  URL: `https://doi.org/10.1109/ICDE.2018.00027`
  PDF: `https://zenodo.org/records/2157816/files/adaptiveexecution.pdf`
  Why: Umbra's adaptive bytecode/JIT execution foundation; relevant to deciding
  when GPU DB should interpret, compile, batch, or route short SQL plans
  without paying excessive setup latency.
- `queued` — **Procella: Unifying Serving and Analytical Data at YouTube**,
  Chattopadhyay et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p2022-chattopadhyay.pdf`
  DOI: `https://doi.org/10.14778/3352063.3352121`
  Why: F1 Lightning reuses Procella-like encoded column/vectorized execution
  ideas; useful for CPU/GPU shared columnar wire formats, high-QPS serving
  caches, and avoiding data conversion at query/storage boundaries.
- `queued` — **Parallel Replication across Formats in SAP HANA for Scaling Out
  Mixed OLTP/OLAP Workloads**, Lee et al., PVLDB 2017.
  URL: `https://doi.org/10.14778/3137765.3137767`
  Why: F1 Lightning contrasts its loosely coupled CDC service with SAP HANA's
  tighter log-replay replica architecture; useful for comparing freshness,
  source-engine modification cost, and row-to-column replication paths.
- `queued` — **BatchDB: Efficient Isolated Execution of Hybrid OLTP+OLAP
  Workloads for Interactive Applications**, Makreshanski et al., SIGMOD 2017.
  URL: `https://doi.org/10.1145/3035918.3035959`
  PDF: `https://www.doc.ic.ac.uk/~jgiceva/papers/SIGMOD_batchdb.pdf`
  Why: TiDB contrasts BatchDB's primary-secondary HTAP replication without
  high-availability consensus; useful for comparing replica isolation,
  freshness, and performance predictability with GPU DB retained snapshot
  refresh.
- `reviewed` — **L-Store: A Real-time OLTP and OLAP System**, Sadoghi et al.,
  EDBT 2018.
  URL: `https://research.ibm.com/publications/l-store-a-real-time-oltp-and-olap-system`
  arXiv: `https://arxiv.org/abs/1601.04084`
  Why: TiDB contrasts L-Store's lineage-based single-engine HTAP design;
  useful for comparing contention-free staging, base/tail lineage, and
  historical query support with P8 stable-plus-delta resident generations.
  Journal entry added 2026-06-05.
- `queued` — **Updatable Learned Index with Precise Positions**, Wu et al.,
  PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p1276-wu.pdf`
  Why: G-Learned Index compares against dynamic learned-index behavior; LIPP's
  precise-position and dynamic-adjustment design is useful for deciding whether
  resident learned indexes can support refresh and update deltas without wide
  last-mile searches.
- `queued` — **ALEX: An Updatable Adaptive Learned Index**, Ding et al.,
  SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3389711`
  Why: G-Learned Index uses ALEX as a dynamic learned-index baseline; useful for
  comparing model-node expansion, update handling, and CPU fallback against GPU
  resident learned-index routes.
- `reviewed` — **Retrofitting High Availability Mechanism to Tame Hybrid
  Transaction/Analytical Processing**, Shen et al., OSDI 2021.
  URL: `https://www.usenix.org/conference/osdi21/presentation/shen`
  PDF: `https://www.usenix.org/system/files/osdi21-shen.pdf`
  Why: ByteHTAP contrasts VEGITO's backup-based fresh HTAP design; useful for
  comparing log-apply replicas, block-based multiversion column layout, and
  high-availability/read-freshness tradeoffs against GPU DB resident snapshots.
  Journal entry added 2026-06-05.
- `queued` — **Real-Time LSM-Trees for HTAP Workloads**, Saxena et al.,
  arXiv 2021.
  URL: `https://arxiv.org/abs/2101.06801`
  Why: ByteHTAP cites real-time LSM-tree work in the HTAP freshness ecosystem;
  useful for evaluating whether GPU DB cold/warm tiers should expose
  LSM-style freshness windows, merge pressure, and snapshot-aware compaction.
- `queued` — **Adaptive HTAP through Elastic Resource Scheduling**, Raza,
  Chrysogelos, Anadiotis, and Ailamaki, SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3389723`
  Why: ByteHTAP compares against elastic HTAP scheduling; useful for deciding
  when GPU DB should shift resources between mutation, refresh, and retained
  read owners instead of fixing static OLTP/OLAP resource splits.
- `reviewed` — **An Empirical Evaluation of Columnar Storage Formats**,
  Zeng et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol17/p148-zeng.pdf`
  arXiv: `https://arxiv.org/abs/2304.05028`
  Why: comparative study of Parquet/ORC/Arrow internals, including format
  inefficiencies for machine-learning workloads and GPU decoding; useful
  baseline for deciding whether P8 should adopt an existing file format,
  a FastLanes-style resident segment, or an internal-only GPU layout.
- `reviewed` — **Mainlining Databases: Supporting Fast Transactional Workloads on
  Universal Columnar Data File Formats**, Li et al., arXiv 2020.
  URL: `https://arxiv.org/abs/2004.14471`
  Why: transactional workload support over universal columnar file formats;
  useful counterpoint to FastLanes for whether GPU DB cold/warm segments can
  also serve point lookups and mutation-adjacent routes without separate
  row-store copies.
- `reviewed` — **LithOS: An Operating System for Efficient Machine Learning on
  GPUs**, Coppock et al., arXiv 2025.
  URL: `https://arxiv.org/abs/2504.15465`
  Why: GPU multitasking paper cites LithOS as a recent GPU OS direction;
  useful for comparing OS-like GPU scheduling and isolation with database-owned
  GPU execution owners.
- `reviewed` — **KRISP: Enabling Kernel-wise Right-sizing for Spatial
  Partitioned GPU Inference Servers**, Chow, Jahanshahi, and Wong, HPCA 2023.
  URL: `https://doi.org/10.1109/HPCA56546.2023.10071121`
  PDF: `https://www.cs.ucr.edu/~ajaha004/files/KRISP.pdf`
  Why: LithOS contrasts KRISP's kernel-wise resource sizing with transparent
  TPC scheduling; useful for GPU DB route admission when kernels have uneven
  SM/TPC scaling and a fixed per-query reservation wastes accelerator capacity.
- `queued` — **CoFRIS: Coordinated Frequency and Resource Scaling for GPU
  Inference Servers**, Chow and Wong, IGSC 2023.
  URL: `https://doi.org/10.1145/3634769.3634808`
  Why: LithOS cites CoFRIS as a GPU frequency/resource scaling predecessor;
  useful for separating route latency budgets, energy-aware DVFS, and
  capacity-right-sizing policies for always-on retained read services.
- `reviewed` — **SGDRC: Software-Defined Dynamic Resource Control for
  Concurrent DNN Inference on NVIDIA GPUs**, Zhang et al., PPoPP 2025.
  URL: `https://doi.org/10.1145/3710848.3710863`
  PDF: `https://people.cs.vt.edu/~huaicheng/p/ppopp25-sgdrc.pdf`
  Why: recent GPU resource-control follow-up cited by LithOS; useful for
  comparing software-visible GPU partition control against LithOS-style
  transparent atomization before GPU DB relies on hardware-specific scheduling
  hooks. Journal entry added 2026-06-05.
- `reviewed` — **Orion: Interference-aware, Fine-grained GPU Sharing for ML
  Applications**, Strati, Ma, and Klimovic, EuroSys 2024.
  URL: `https://doi.org/10.1145/3627703.3629578`
  PDF: `https://fotstrt.github.io/files/2024-orion.pdf`
  Why: SGDRC compares against Orion's interference-aware colocation policy;
  useful for deciding when GPU DB should use profiling-based compatible
  co-runners instead of explicit partitioning for retained reads, scans, and
  refresh work. Journal entry added 2026-06-05.
- `queued` — **Transparent GPU Sharing in Container Clouds for Deep Learning
  Workloads**, Wu et al., NSDI 2023.
  URL: `https://www.usenix.org/conference/nsdi23/presentation/wu`
  Why: SGDRC contrasts TGS temporal multiplexing and CUDA-container switching
  overhead; useful for evaluating whether GPU DB request classes should ever
  use exclusive time slices instead of spatial sharing or chunked preemption.
- `reviewed` — **Paella: Low-Latency Model Serving with Software-Defined GPU
  Scheduling**, Ng, Demoulin, and Liu, SOSP 2023.
  URL: `https://doi.org/10.1145/3600006.3613163`
  PDF: `https://kelvin-ng.github.io/assets/sosp2023-final224.pdf`
  Why: Orion contrasts Paella as a low-latency GPU serving scheduler that is
  not compute/memory-profile aware; useful for comparing model-serving
  scheduling policies against GPU DB route-class and chunking policies.
  Journal entry added 2026-06-06 from the author PDF after the ACM PDF
  endpoint returned HTTP 403.
- `queued` — **FaaSwap: SLO-Aware, GPU-Efficient Serverless Inference via
  Model Swapping**, Gunasekaran et al., arXiv 2023.
  URL: `https://arxiv.org/abs/2306.03622`
  Why: discovered while retrieving Paella; useful for comparing SLO-aware GPU
  request scheduling, model/state swapping, and GPU memory locality against
  GPU DB response deadlines, resident-fragment pressure, and fallback policy.
- `queued` — **Fast Distributed Inference Serving for Large Language Models**,
  Wu et al., arXiv 2023.
  URL: `https://arxiv.org/abs/2305.05920`
  Why: discovered while retrieving Paella; useful for comparing preemptive GPU
  scheduling and host/GPU state movement under head-of-line blocking with GPU
  DB's long-scan versus short-retained-read admission problem.
- `queued` — **Zico: Efficient GPU Memory Sharing for Concurrent DNN
  Training**, Lim et al., USENIX ATC 2021.
  URL: `https://www.usenix.org/conference/atc21/presentation/lim`
  Why: Orion contrasts Zico's training-oriented GPU memory sharing and
  forward/backward scheduling with operator-profile-aware colocation; useful
  for GPU DB refresh and scan jobs that have large resident memory footprints.
- `queued` — **StreamBox: A Lightweight GPU Sandbox for Serverless Inference
  Workflow**, Wu et al., USENIX ATC 2024.
  URL: `https://www.usenix.org/conference/atc24/presentation/wu-hao`
  Why: SGDRC names StreamBox as a transparent colocation/sandbox direction;
  useful for comparing GPU runtime fault isolation and queue boundaries before
  colocating database kernels with externally generated GPU tasks.
- `reviewed` — **Microsecond-scale Preemption for Concurrent GPU-accelerated DNN
  Inferences**, Han et al., OSDI 2022.
  URL: `https://www.usenix.org/conference/osdi22/presentation/han`
  Why: REEF-style GPU preemption is cited by the multitasking paper; useful
  for evaluating whether retained-read latency can be protected by
  kernel-boundary or finer-grained preemption.
- `queued` — **Hardware Compute Partitioning on NVIDIA GPUs**, Bakita and
  Anderson, RTAS 2023.
  URL: `https://doi.org/10.1109/RTAS58335.2023.00012`
  Why: cited by the multitasking paper for SM/control mechanisms; useful for
  understanding what resource partitioning can be exposed to GPU DB route
  admission without depending on unsupported driver behavior.
- `reviewed` — **Polaris: Enabling Transaction Priority in Optimistic
  Concurrency Control**, Ye et al., PACMMOD/SIGMOD 2023.
  URL: `https://doi.org/10.1145/3588724`
  PDF: `https://chenhao-ye.github.io/publication/polaris/polaris.pdf`
  Why: priority-aware OCC cited by PreemptDB; relevant to combining request
  priority with conflict handling instead of only changing worker scheduling.
- `queued` — **Universal Packet Scheduling**, Mittal, Agarwal, Ratnasamy, and
  Shenker, NSDI 2016.
  URL: `https://www.usenix.org/conference/nsdi16/technical-sessions/presentation/mittal`
  Why: Eiffel cites universal packet scheduling as a flexible scheduling
  objective; useful for comparing request-ranking policies that emulate
  shortest-job, deadline, and slack-aware queueing in GPU DB admission.
- `queued` — **Fast Compilation and Execution of SQL Queries with WebAssembly**,
  Grulich, Dorok, Bress, and Schallehn, arXiv 2021.
  URL: `https://arxiv.org/abs/2104.15098`
  Why: modern follow-up to adaptive query execution that uses WebAssembly/V8
  tiered execution; useful for comparing bytecode/JIT route startup cost with
  portable fragment execution before GPU DB commits to a custom IR.
- `queued` — **How to Architect a Query Compiler**, Shaikhha et al.,
  SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915244`
  Why: adaptive execution cites it as a query-compiler architecture direction;
  useful for deciding how much of GPU DB's route compiler should be reusable
  staged code, hand-written kernels, or a compact fragment IR.
- `queued` — **NUMFabric: Fast and Flexible Bandwidth Allocation in
  Datacenters**, Nagaraj et al., SIGCOMM 2016.
  URL: `https://doi.org/10.1145/2934872.2934890`
  PDF: `https://web.stanford.edu/~skatti/pubs/sigcomm16-num.pdf`
  Why: PIFO cites NUMFabric as a flexible bandwidth-allocation use case; useful
  for comparing utility-driven admission and weighted fair queueing when GPU DB
  request classes compete for network, owner-ring, and accelerator capacity.
- `reviewed` — **HetExchange: Encapsulating Heterogeneous CPU-GPU Parallelism in
  JIT Compiled Engines**, Chrysogelos et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p544-chrysogelos.pdf`
  DOI: `https://doi.org/10.14778/3303753.3303760`
  Why: Fluid Co-processing contrasts fragment-level GPU assistance with
  exchange-style whole-pipeline routing; useful for comparing planner-time
  CPU/GPU placement with runtime split-route fallback.
- `reviewed` — **Performance-Optimal Filtering: Bloom Overtakes Cuckoo at High
  Throughput**, Lang et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p502-lang.pdf`
  Why: Fluid Co-processing uses performance-optimal Bloom filter modeling as
  its CPU/GPU filter sizing foundation; relevant to resident join summaries,
  false-positive budgets, and route-specific filter sizing.
- `queued` — **Andromeda: Performance, Isolation, and Velocity at Scale in
  Cloud Network Virtualization**, Dalton et al., NSDI 2018.
  URL: `https://www.usenix.org/conference/nsdi18/presentation/dalton`
  Why: Eiffel motivates software scheduling at end hosts and virtualized
  networks; Andromeda is a primary large-scale system source for isolation,
  fast path design, and software/hardware network split tradeoffs.
- `queued` — **Clockwork: Predictable Low Latency for Deep Learning Inference**,
  Gujarati et al., OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/gujarati`
  Why: REEF contrasts predictable DNN serving systems; useful for comparing
  admission-time latency prediction, batching, and deadline scheduling against
  GPU DB retained-read and best-effort scan co-scheduling.
- `queued` — **Baymax: Qos Awareness and Increased Utilization for Non-Preemptive
  Accelerators in Warehouse Scale Computers**, Chai et al., ASPLOS 2019.
  URL: `https://doi.org/10.1145/3297858.3304018`
  Why: REEF cites QoS-aware non-preemptive accelerator sharing; useful as a
  counterpoint when GPU DB cannot kill or restart long-running database kernels
  and must rely on padding, admission, or spatial isolation instead.
- `reviewed` — **MRVs: Enforcing Numeric Invariants in Parallel Updates to
  Hotspots with Randomized Splitting**, Faria and Pereira, PACMMOD/SIGMOD 2023.
  URL: `https://doi.org/10.1145/3588723`
  Why: TiQuE cites MRVs as a way to reduce wasted work around optimistic
  hotspot updates; relevant to GPU DB write admission when many logical
  sessions update counters or bounded inventory-like values.
- `reviewed` — **Towards Generic Fine-Grained Transaction Isolation in
  Polystores**, Faria, Pereira, Alonso, and Vilaca, Poly@VLDB 2021.
  URL: `https://link.springer.com/chapter/10.1007/978-3-030-93663-1_3`
  PDF: `https://rmpvilaca.github.io/assets/pdf/FPAV21.pdf`
  Why: TiQuE cites this as an earlier layered-isolation direction for
  polystores; useful for future multi-engine GPU DB routes where transactional
  metadata may span CPU, GPU-resident, and cold-tier execution engines.
- `reviewed` — **Totally-Ordered Prefix Parallel Snapshot Isolation**, Faria and
  Pereira, PaPoC@EuroSys 2021.
  DOI: `https://doi.org/10.1145/3447865.3457966`
  PDF:
  `https://repositorio.inesctec.pt/server/api/core/bitstreams/67827bf5-65e9-490a-b1d5-a4b976a732f4/content`
  Why: follow-up from the same authors on a restricted Parallel Snapshot
  Isolation model that orders a prefix of history; useful for comparing
  low-wait distributed snapshot freshness against GPU DB's snapshot generation
  and route-validity rules.
- `reviewed` — **On Reading Fresher Snapshots in Parallel Snapshot Isolation**,
  Javidi Kishi and Palmieri, ICDCS 2020.
  URL: `https://doi.org/10.1109/ICDCS47774.2020.00127`
  PDF: `https://www.cse.lehigh.edu/~palmieri/files/pubs/CR-ICDCS-2020.pdf`
  Why: cited by TOPSI as a PSI freshness direction; useful for comparing
  snapshot freshness and abort-rate tradeoffs against scalar-prefix visibility
  certificates.
- `reviewed` — **FW-KV: Improving Read Guarantees in PSI**, Javidi Kishi and
  Palmieri, Middleware 2021.
  URL: `https://doi.org/10.1145/3464298.3476131`
  PDF: `https://www.cse.lehigh.edu/~palmieri/files/pubs/CR-MIDDLEWARE-2021.pdf`
  Why: full FPSI successor that evaluates fresher PSI snapshots on YCSB and
  TPC-C; useful for checking whether version-access metadata remains practical
  under OLTP contention and read-mostly workloads.
- `reviewed` — **SSS: Scalable Key-Value Store with External Consistent and
  Abort-free Read-only Transactions**, Javidi Kishi, Peluso, Korth, and
  Palmieri, ICDCS 2019.
  URL: `https://doi.org/10.1109/ICDCS.2019.00065`
  PDF: `https://www.cse.lehigh.edu/~palmieri/files/pubs/CR-icdcs2019.pdf`
  arXiv: `https://arxiv.org/abs/1901.03772`
  Why: related vector-clock and snapshot-queuing design for abort-free,
  externally consistent read-only transactions without centralized
  synchronization; useful for comparing PSI freshness against stronger
  client-visible ordering.
- `reviewed` — **Cure: Strong Semantics Meets High Availability and Low
  Latency**, Akkoorath et al., ICDCS 2016.
  URL: `https://doi.org/10.1109/ICDCS.2016.98`
  PDF: `https://webperso.info.ucl.ac.be/~pvr/icdcs2016-cure.pdf`
  Why: SSS contrasts stronger external consistency with causally consistent
  transactional replication; useful for deciding which weaker snapshot or
  replica-freshness guarantees are acceptable, if any, for remote retained
  GPU snapshots or future replicated owners.
- `queued` — **GMU: Genuine Multiversion Update-Serializable Partial Data
  Replication**, Peluso et al., IEEE TPDS 2016.
  URL: `https://doi.org/10.1109/TPDS.2015.2465906`
  Why: FW-KV names GMU as a related multiversion partial-replication design
  that advances snapshots while preserving update-serializable guarantees;
  useful for comparing version-access metadata with stronger replicated MVCC
  visibility rules.
- `reviewed` — **Amazon Aurora: Design Considerations for High Throughput
  Cloud-Native Relational Databases**, Verbitski et al., SIGMOD 2017.
  URL: `https://doi.org/10.1145/3035918.3056101`
  PDF:
  `https://cdn.amazon.science/dc/2b/4ef2b89649f9a393d37d3e042f4e/amazon-aurora-design-considerations-for-high-throughput-cloud-native-relational-databases.pdf`
  Why: TOPSI names Aurora as a disaggregated-storage target; useful for
  comparing GPU DB's WAL/storage-publication boundaries with a log-structured
  cloud relational storage service.
- `reviewed` — **Socrates: The New SQL Server in the Cloud**, Antonopoulos et
  al., SIGMOD 2019.
  URL: `https://doi.org/10.1145/3299869.3314047`
  Why: Aurora-related cloud database storage/compute separation; useful for
  comparing page-server/log-service roles, checkpointing, and cold-page
  retrieval against GPU DB's CPU/NVMe/GPU tier publication model.
- `reviewed` — **Towards Optimal Transaction Scheduling**, Cheng et al.,
  PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf`
  Why: modern transaction scheduling work cited by PreemptDB; useful for
  contrasting non-preemptive priority ordering with interrupt-driven
  preemption and owner-queue admission.
- `reviewed` — **Data Blocks: Hybrid OLTP and OLAP on Compressed Storage Using
  Both Vectorization and Compilation**, Lang et al., SIGMOD 2016.
  URL: `https://www-db.cs.tum.edu/downloads/publications/datablocks.pdf`
  Why: Hermes and Mainlining Databases keep returning to stable columnar data
  plus mutation-adjacent transactional access; useful for older-but-eligible
  compressed HTAP storage mechanics. Mainlining Databases also cites
  HyPer/Data Blocks as a compressed cold-data HTAP baseline; useful for
  comparing hot/cold columnar block compression, positional pruning metadata,
  and OLTP-safe tuple access against P8 resident segment designs.
- `reviewed` — **How Good is My HTAP System?**, Milkai et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526148`
  PDF: `https://pages.cs.wisc.edu/~chronis/files/howgoodismyhtap.pdf`
  Code: `https://github.com/UWHustle/HATtrick`
  Why: Hermes evaluates real-time analytics against HATtrick; useful for
  shaping GPU DB mixed transactional/analytical workload gates beyond separate
  OLTP and OLAP clients.
- `reviewed` — **FlexPushdownDB: Hybrid Pushdown and Caching in a Cloud DBMS**,
  Yang et al., PVLDB 2021.
  URL: `https://vldb.org/pvldb/vol14/p2101-yang.pdf`
  DOI: `https://doi.org/10.14778/3476249.3476265`
  Why: Hermes uses FlexPushdownDB as an AP engine; relevant to deciding which
  filtering, aggregation, and cache work should happen near storage, host
  memory, or GPU execution workers.
- `queued` — **PushdownDB: Accelerating a DBMS using S3 Computation**,
  Yu et al., ICDE 2020.
  URL: `https://doi.org/10.1109/ICDE48307.2020.00166`
  Why: direct predecessor to FlexPushdownDB's pushdown-only baseline; useful
  for isolating S3 Select-style filtering/aggregation pushdown economics
  before GPU DB adopts hybrid CPU/GPU/NVMe segment routes.
- `queued` — **AQUOMAN: An Analytic-Query Offloading Machine**,
  Xu et al., MICRO 2020.
  URL: `https://doi.org/10.1109/MICRO50266.2020.00042`
  Why: FlexPushdownDB cites analytic query offloading as a near-storage
  pushdown direction; useful for comparing hardware-assisted pushdown and
  multiway SQL offload against GPU DB's explicit owner and tier boundaries.
- `reviewed` — **L-Store: A Real-time OLTP and OLAP System**, Sadoghi et al.,
  EDBT 2018.
  URL: `https://arxiv.org/abs/1601.04084`
  Why: Mainlining Databases contrasts L-Store's lineage/tail-page architecture
  with relaxed Arrow blocks; useful for evaluating lineage-based staging,
  historic visibility, and lazy columnar consolidation for retained snapshots.
  Stale duplicate marked reviewed on 2026-06-05.
- `queued` — **Real-Time LSM-Trees for HTAP Workloads**, Saxena et al.,
  arXiv 2021.
  URL: `https://arxiv.org/abs/2101.06801`
  Why: lifecycle-aware LSM layout design is a follow-up to universal columnar
  and hot/cold block conversion; useful for comparing row-to-column movement
  by storage level rather than by in-memory block age.
- `reviewed` — **F1 Lightning: HTAP as a Service**, Yang et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p3313-yang.pdf`
  DOI: `https://doi.org/10.14778/3415478.3415553`
  Why: HATtrick classifies isolated/hybrid HTAP designs and evaluates TiDB-style
  split engines; F1 Lightning gives a production loose-coupling design for
  fresh analytical copies, CDC, compaction, and federated query integration.
- `queued` — **Procella: Unifying serving and analytical data at YouTube**,
  Chattopadhyay et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p2022-chattopadhyay.pdf`
  DOI: `https://doi.org/10.14778/3352063.3352121`
  Why: F1 Lightning reuses Procella-like columnar/vectorized ideas; useful for
  comparing unified serving/analytics layouts, encoded-column execution, and
  mixed workload routing against GPU-resident column groups.
- `reviewed` — **OLxPBench: Real-time, Semantically Consistent, and
  Domain-specific are Essential in Benchmarking, Designing, and Implementing
  HTAP Systems**, arXiv 2022.
  URL: `https://arxiv.org/abs/2203.16095`
  Why: HATtrick exposes the need to measure freshness and mixed throughput;
  OLxPBench is a modern benchmark follow-up focused on semantic consistency and
  domain-specific real-time analytics.
- `queued` — **Pixels: An Efficient Column Store for Cloud Data Lakes**,
  Bian and Ailamaki, ICDE 2022.
  URL: `https://doi.org/10.1109/ICDE53745.2022.00286`
  Why: cited by the columnar-format evaluation as a modern file-format
  optimization; useful for comparing cloud/cold-tier column groups,
  metadata layout, and coalesced reads against P8 resident and
  over-resident segment directories.
- `queued` — **GSLICE: Controlled Spatial Sharing of GPUs for a Scalable
  Inference Platform**, Dhakal, Kulkarni, and Ramakrishnan, SoCC 2020.
  URL: `https://doi.org/10.1145/3419111.3421284`
  Why: KRISP compares against GSLICE's model-wise GPU spatial sharing and
  shadow-instance resizing; useful for GPU DB admission when route classes need
  spatial isolation but kernel-wise control is unavailable.
- `queued` — **Multi-model Machine Learning Inference Serving with GPU Spatial
  Partitioning**, Choi et al., arXiv 2021.
  URL: `https://arxiv.org/abs/2109.01611`
  Why: KRISP compares against Gpulet-style model-wise right-sizing; useful for
  contrasting request-epoch GPU partition changes with per-route and
  per-kernel GPU DB resource reservations.
- `queued` — **PARIS and ELSA: An Elastic Scheduling Algorithm for
  Reconfigurable Multi-GPU Inference Servers**, Kim, Choi, and Rhu, arXiv
  2022.
  URL: `https://arxiv.org/abs/2202.13481`
  Why: KRISP contrasts PARIS/ELSA's model- and batch-size-aware MIG scheduling;
  useful for future multi-GPU resident-route placement and shadow-capacity
  planning.
- `reviewed` — **BtrBlocks: Efficient Columnar Compression for Data Lakes**,
  Kuschewski et al., SIGMOD 2023.
  URL: `https://doi.org/10.1145/3589263`
  Why: cited by the columnar-format evaluation as a sampling-based
  encoding-selection design; useful for deciding whether P8 should choose
  compression per segment from measured decode speed instead of fixed
  global codecs.
- `queued` — **FSST: Fast Random Access String Compression**,
  Boncz, Neumann, and Leis, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf`
  Why: cited as a modern lightweight string-compression technique;
  directly relevant to P8 text column layout, prefix predicates, and
  CPU/GPU tradeoffs for retained string routes.
- `queued` — **Antidote: A Highly-Available Geo-Replicated Database with
  Stronger Guarantees**, Akkoorath et al., arXiv 2018.
  URL: `https://arxiv.org/abs/1802.06459`
  Why: Cure was implemented over the Antidote platform; the later system paper
  may add production-oriented detail on causal transactions, CRDT execution,
  and globally stable snapshot management.
- `queued` — **Causal Consistency and Latency Optimality: Friend or Foe?**,
  Mehdi et al., PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol11/p161-mehdi.pdf`
  Why: direct modern follow-up for causal consistency metadata and latency
  tradeoffs; useful for checking whether vector-clock or dependency-tracking
  snapshot frontiers can be latency-optimal under replicated retained reads.
- `queued` — **DL-Store: A Distributed Hybrid OLTP and OLAP Data Processing
  Engine**, Zhang, Sadoghi, and Jacobsen, ICDCS 2016.
  URL: `https://doi.org/10.1109/ICDCS.2016.47`
  Why: L-Store cites DL-Store as a distributed hybrid OLTP/OLAP follow-up;
  useful for comparing lineage-style single-node base/tail publication with
  distributed partitioning, freshness, and analytical routing.

- `reviewed` — **Harnessing GPU Power for Enhanced OLTP: A Study in Concurrency
  Control Schemes**, arXiv 2024.
  URL: `https://arxiv.org/abs/2406.10158`
  Why: modern GPU OLTP concurrency-control evaluation that compares GPU-adapted
  2PL, timestamp ordering, MVCC, OCC, GPUTx, and GaccO-style conflict-graph or
  deterministic locking schemes; useful follow-up for deciding which GPU write
  batch protocol is benchmark-worthy.
- `reviewed` — **Accelerating in-memory transaction processing using general
  purpose graphics processing units**, Gao et al., Future Generation Computer
  Systems 2019.
  URL: `https://doi.org/10.1016/j.future.2019.03.034`
  Why: GPU-TPS is a 2015-present GPU OLTP system baseline referenced by the
  gCCTB study; useful for comparing GPU transaction execution models, GPU hash
  table/B+ tree indexing, and SmallBank/TPC-C write-path claims against
  conflict-control-only testbeds.
- `queued` — **Thread-Level Locking for SIMT Architectures**, Gao et al.,
  IEEE TPDS 2020.
  URL: `https://doi.org/10.1109/TPDS.2019.2955705`
  Why: GPU-TPS builds on GPU-side fine-grained locking constraints; useful for
  isolating whether thread-level locks can safely support GPU DB write-batch
  admission, conflict handling, and hot-row synchronization.
- `queued` — **Towards a General and Efficient Linked-List Hash Table on
  GPUs**, Gao et al., HPCC/SmartCity/DSS 2019.
  URL: `https://doi.org/10.1109/HPCC/SmartCity/DSS.2019.00064`
  Why: GPU-TPS depends on GPU-friendly unordered indexes; useful for comparing
  resident hash-table update costs against sorted vectors, B-trees, and
  rebuild-at-generation-boundary index routes.
- `reviewed` — **LibPreemptible: Enabling Fast, Adaptive, and
  Hardware-Assisted User-Space Scheduling**, Li et al., HPCA 2024.
  URL: `https://doi.org/10.1109/HPCA57654.2024.00075`
  PDF: `https://people.csail.mit.edu/delimitrou/papers/2024.hpca.libpreemptible.pdf`
  Why: general hardware-assisted userspace preemption framework cited by
  PreemptDB; useful if GPU DB wants preemption mechanics outside a full
  transaction-engine rewrite.
- `reviewed` — **Skyloft: A General High-Efficient Scheduling Framework in
  User Space**, Jia et al., SOSP 2024.
  URL:
  `https://madsys.cs.tsinghua.edu.cn/publication/skyloft-a-general-high-efficient-scheduling-framework-in-user-space/SOSP24-Jia.pdf`
  DOI: `https://doi.org/10.1145/3694715.3695973`
  Why: modern user-space scheduling framework with user-mode-interrupt
  preemption and DPDK integration; useful as a follow-up to Arachne,
  Shenango, and Shinjuku for deciding whether GPU DB needs preemptive
  user-space request classes around long scans and short retained reads.
- `reviewed` — **Achieving Microsecond-Scale Tail Latency Efficiently with
  Approximate Optimal Scheduling**, Iyer et al., SOSP 2023.
  URL: `https://doi.org/10.1145/3600006.3613136`
  PDF: `https://rishabh246.github.io/files/concord.pdf`
  Why: Concord is Skyloft's approximate-optimal scheduling comparison point;
  useful for deciding whether GPU DB should approximate processor-sharing
  policies for heavy-tailed retained reads, scans, and refresh jobs without a
  fully general preemptive runtime.
- `reviewed` — **Fast Core Scheduling with Userspace Process Abstraction**,
  Lin et al., SOSP 2024.
  URL: `https://doi.org/10.1145/3694715.3695976`
  PDF: `https://chenyoumin1993.github.io/papers/sosp24-vessel.pdf`
  Why: Vessel-style userspace process abstraction is a direct contrast to
  Skyloft's shared scheduler model; relevant to tenant/session isolation,
  core sharing, and minimizing inter-application switching cost.
- `reviewed` — **Efficient Scheduling Policies for Microsecond-Scale Tasks**,
  McClure et al., NSDI 2022.
  URL: `https://www.usenix.org/conference/nsdi22/presentation/mcclure`
  PDF: `https://www.usenix.org/system/files/nsdi22-paper-mcclure_2.pdf`
  Why: policy-focused evaluation of work stealing, static allocation, and
  core reallocation for microsecond tasks; useful before choosing GPU DB
  IO-worker, retained-read, and background-refresh scheduling policies.
- `queued` — **PolarDB Serverless: A Cloud Native Database for
  Disaggregated Data Centers**, Cao et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457550`
  Why: Socrates and Aurora leave open how elastic compute interacts with a
  disaggregated storage layer; PolarDB Serverless is a modern follow-up for
  elastic buffer ownership, cold-page access, and storage/compute separation.
- `reviewed` — **FoundationDB: A Distributed Unbundled Transactional Key Value
  Store**, Zhou et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457559`
  Why: Socrates uses separate services for log, page serving, and durable
  storage; FoundationDB is a useful contrast for unbundled transactional
  storage, log/transaction service separation, and deterministic recovery
  boundaries.
- `reviewed` — **Syrup: User-defined Scheduling across the Stack**,
  Kaffes et al., SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483548`
  Why: cited by the NSDI 2022 scheduling-policy paper as a cross-stack
  scheduling design; useful for deciding whether GPU DB should expose query
  class, priority, and owner-boundary scheduling hints through the runtime
  instead of hard-coding one queue policy.
- `queued` — **Lemo: A Cache-Enhanced Learned Optimizer for Concurrent
  Queries**, Mo et al., PACMMOD 2023.
  URL: `https://doi.org/10.1145/3626713`
  Why: RankPQO cites this concurrent-query learned optimizer work; useful for
  testing whether GPU DB route choice should cache decisions under concurrent
  queue pressure instead of treating each retained query in isolation.
- `reviewed` — **Cost-based or Learning-based? A Hybrid Query Optimizer for
  Query Plan Selection**, Yu et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p3924-li.pdf`
  DOI: `https://doi.org/10.14778/3565838.3565846`
  Why: RankPQO cites hybrid plan selection work; relevant to keeping GPU DB
  deterministic cost rules as guardrails while adding measured route-ranking
  hints for CPU/GPU/tier choices.
- `reviewed` — **Everything You Always Wanted to Know About Compiled and
  Vectorized Queries But Were Afraid to Ask**, Kersten et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p2209-kersten.pdf`
  DOI: `https://doi.org/10.14778/3275366.3275370`
  Why: HetExchange motivates JIT integration against vectorized execution;
  this paper is a focused CPU execution-model baseline for deciding when GPU
  DB route fragments should be compiled, vectorized, interpreted, or staged.
  Stale duplicate marked reviewed on 2026-06-05.
- `queued` — **Voodoo - A Vector Algebra for Portable Database Performance on
  Modern Hardware**, Pirk et al., PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p1707-pirk.pdf`
  DOI: `https://doi.org/10.14778/3007328.3007336`
  Why: HetExchange cites Voodoo as a portable hardware-conscious algebra;
  useful for comparing route descriptors and device providers with a
  declarative intermediate representation for CPU/GPU portability.
- `reviewed` — **How to Architect a Query Compiler, Revisited**, Tahboub,
  Essertel, and Rompf, SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3196893`
  PDF: `https://www.cs.purdue.edu/homes/rompf/papers/tahboub-sigmod18.pdf`
  Why: Kersten et al. cite query-compiler architecture as a major source of
  maintainability complexity; useful for deciding whether GPU DB route
  fragments should use staged compilation, a compact IR, or handwritten
  kernels with deterministic planner guardrails. Journal entry added
  2026-06-06.
- `queued` — **Query Compilation Without Regrets**, Grulich et al.,
  PACMMOD/SIGMOD 2024.
  URL: `https://doi.org/10.1145/3654970`
  PDF: `https://nebula.stream/paper/grulich_sigmod2024.pdf`
  Why: modern Nautilus follow-up that bridges interpretation and compilation;
  useful for deciding whether GPU DB should use cached generated fragments,
  adaptive tiered execution, or interpreter-first execution for latency-sensitive
  route shapes.
- `queued` — **A Common Runtime for High Performance Data Analysis**, Palkar
  et al., CIDR 2017.
  URL: `https://www.cidrdb.org/cidr2017/papers/p51-palkar-cidr17.pdf`
  Why: Kersten et al. discuss hybrid execution and language integration;
  useful for comparing a shared fragment runtime against separate CPU, GPU,
  and fallback execution stacks.
- `reviewed` — **Query Performance Prediction for Concurrent Queries using
  Graph Embedding**, Zhou et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p1416-zhou.pdf`
  DOI: `https://doi.org/10.14778/3397230.3397238`
  Why: Lemo and concurrent-query optimizer work point back to graph-based
  interference prediction; useful for route admission that estimates shared
  table/index/tier/GPU-stream contention instead of optimizing each retained
  query in isolation.
- `queued` — **AdaptDB: Adaptive Partitioning for Distributed Joins**,
  Lu, Shanbhag, Jindal, and Madden, PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p589-lu.pdf`
  Why: data-induced predicates cite adaptive partitioning as a physical-layout
  alternative; useful for comparing static range-set pruning with workload-
  adaptive partition refinement for GPU resident and cold-tier segments.
- `queued` — **Skipping-oriented Partitioning for Columnar Layouts**,
  Sun, Franklin, Wang, and Wu, PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p421-sun.pdf`
  Why: data-induced predicates contrast against workload-aware skipping
  layouts; useful for deciding whether GPU DB should reshape resident/cold
  partitions around observed predicates or rely on lightweight per-generation
  statistics.
- `reviewed` — **Mind the Gap: A Case for Informed Request Scheduling at the
  NIC**, Humphries, Kaffes, Mazieres, and Kozyrakis, HotNets 2019.
  URL: `https://doi.org/10.1145/3365609.3365856`
  PDF: `https://cs.stanford.edu/~jhumphri/documents/mind-the-gap.pdf`
  Why: Syrup cites this NIC-side informed request scheduling work; relevant to
  deciding whether GPU DB should push route-class or key-home steering closer
  to the network edge before requests enter owner queues.
- `queued` — **MittOS: Supporting Millisecond Tail Tolerance with Fast
  Rejecting SLO-Aware OS Interface**, Hao et al., SOSP 2017.
  URL: `https://doi.org/10.1145/3132747.3132774`
  Why: Syrup cites fast rejection with deadline/SLO hints; relevant to GPU DB
  overload responses when retained-read, response, mutation, or GPU queues
  cannot meet a request's latency budget.
- `queued` — **TAS: TCP Acceleration as an OS Service**, Shenango/TAS related
  work, SOSP 2019.
  URL: `https://doi.org/10.1145/3341301.3359657`
  Why: Syrup contrasts request scheduling with end-host transport scheduling;
  useful for future pgwire/transport choices if ordinary kernel TCP remains
  the high-concurrency bottleneck.
- `reviewed` — **Flexible Resource Allocation for Relational
  Database-as-a-Service**, Arora et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p4202-narasayya.pdf`
  Why: modern DBaaS resource-allocation paper cited by Resource-Adaptive Query
  Execution; relevant to pricing or value-of-memory admission policies for
  multi-tenant/session-heavy GPU DB workloads.
- `reviewed` — **Centiman: Elastic, High Performance Optimistic Concurrency
  Control by Watermarking**, Ding et al., SoCC 2015.
  URL: `https://doi.org/10.1145/2806777.2806837`
  PDF:
  `https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/centiman_socc_2015.pdf`
  Why: PWV contrasts against watermark-based OCC; relevant to timestamp
  frontiers, elastic admission, and deciding whether GPU DB write batches
  should expose commit/read watermarks instead of only per-transaction
  validation.
- `reviewed` — **Bonspiel: Low Tail Latency Transactions in
  Geo-Distributed Databases**, Cui et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p3840-cui.pdf`
  Why: modern low-tail transaction follow-up that cites Plor; relevant to
  latency-aware transaction routing, predictable commit paths, and predictable
  transaction admission and retry behavior under high contention.
- `reviewed` — **Rebirth-Retire: A Concurrency Control Protocol Adaptable to
  Different Levels of Contention**, Zhang et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p3162-zhang.pdf`
  Why: modern MVCC/concurrency-control work that discusses Plor and adapts to
  changing workload conditions; useful for deciding when GPU DB should switch
  conflict policy by route, contention, or transaction size.
- `queued` — **FoundationDB Record Layer: A Multi-Tenant Structured Datastore**,
  Kornacker et al., arXiv 2019.
  URL: `https://arxiv.org/abs/1901.04452`
  Why: FoundationDB points to the Record Layer as a production structured
  layer over a transactional key-value core; useful for comparing SQL/catalog,
  secondary-index, and aggregate-index layering against GPU DB route metadata
  and lower-half storage boundaries.
- `reviewed` — **Robust External Hash Aggregation in the Solid State Age**,
  Kuiper, Boncz, and Muhleisen, ICDE 2024.
  URL: `https://doi.org/10.1109/ICDE60146.2024.00288`
  PDF: `https://hannes.muehleisen.org/publications/icde2024-out-of-core-kuiper-boncz-muehleisen.pdf`
  Why: DuckDB external aggregation work cited by Resource-Adaptive Query
  Execution; useful for paged intermediate state, spill-resistant aggregates,
  and over-resident query execution under bounded memory.
- `reviewed` — **Saving Private Hash Join**, Kuiper, Gross, Boncz, and
  Muhleisen, PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p2748-kuiper.pdf`
  DOI: `https://doi.org/10.14778/3742728.3742762`
  Why: direct follow-up to robust external hash aggregation for larger-than-
  memory joins, runtime compression of materialized intermediates, and dynamic
  memory allocation across concurrent blocking operators.
- `reviewed` — **Cooperative Memory Management for Table and Temporary Data**,
  Lasch et al., SiMoD/SIGMOD 2023.
  URL: `https://doi.org/10.1145/3596225.3596230`
  PDF:
  `https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00064480/979-8-4007-0783-4_2023_2.pdf`
  Why: related cooperative-memory baseline for sharing memory between table
  caching and temporary query data; useful for comparing DuckDB's unified
  paged temporary allocations with explicit GPU DB tier budgets.
- `reviewed` — **On the Impact of Memory Allocation on High-Performance Query
  Processing**, Durner, Leis, and Neumann, DaMoN 2019.
  URL: `https://doi.org/10.1145/3329785.3329918`
  Why: cited by cooperative memory management for allocation overhead in
  high-performance query execution; useful for separating allocator latency,
  temporary-buffer reuse, and query-route setup costs in GPU DB admission.
- `reviewed` — **Transaction Scheduling: From Conflicts to Runtime Conflicts**,
  Cao et al., SIGMOD 2023.
  URL: `https://doi.org/10.1145/3603164`
  Preprint:
  `https://www.research.ed.ac.uk/files/360117816/Transaction_Scheduling_CAO_DOA16082022_AFV.pdf`
  Why: modern transaction scheduling paper from SIGMOD 2023; relevant to
  contrasting conflict-graph scheduling with runtime resource conflicts and
  owner-queue admission for mixed GPU DB transaction classes.
- `reviewed` — **Improving Optimistic Concurrency Control through Transaction
  Batching and Operation Reordering**, Ding, Kot, and Gehrke, PVLDB 2018.
  URL: `https://doi.org/10.14778/3282495.3282502`
  PDF: `https://www.vldb.org/pvldb/vol12/p169-ding.pdf`
  Why: OCC batching and operation-reordering work; useful for deciding when
  grouped write admission can safely reorder operation execution within a
  deterministic commit boundary, when GPU DB write admission should batch
  full transaction stages rather than only WAL, index, or GPU refresh
  substeps, and when batch-level write-set ordering should be preferred over
  per-request optimistic validation in hot partitions.
- `reviewed` — **Swift: Delay is Simple and Effective for Congestion Control in
  the Datacenter**, Kumar et al., SIGCOMM 2020.
  URL: `https://doi.org/10.1145/3387514.3406591`
  Why: PowerTCP contrasts against delay-based congestion control; useful for
  deciding when absolute delay, rather than only delay gradient or queue depth,
  should drive GPU DB IO-worker, response-ring, and route-class pacing.
- `reviewed` — **HPCC: High Precision Congestion Control**, Li et al.,
  SIGCOMM 2019.
  URL: `https://doi.org/10.1145/3341302.3342085`
  Why: PowerTCP builds on HPCC-style in-band network telemetry; relevant to
  whether GPU DB should export precise per-boundary service telemetry to
  schedulers instead of relying on coarse queue depths.
- `queued` — **Revisiting Network Support for RDMA**, Mittal et al.,
  SIGCOMM 2018.
  URL: `https://doi.org/10.1145/3230543.3230557`
  Why: HPCC evaluates IRN-style loss recovery and fixed-window inflight
  limiting as an orthogonal flow-control path; useful for comparing precise
  congestion feedback against simpler bounded-inflight request admission for
  future high-concurrency GPU DB transports.
- `queued` — **ghOSt: Fast & Flexible User-Space Delegation of Linux
  Scheduling**, Narayanan et al., SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483542`
  Why: Vessel contrasts against kernel-mediated user-space scheduler
  delegation; useful for deciding whether GPU DB should expose request,
  session, and core-placement decisions through a scheduler agent while
  keeping Linux as the enforcement boundary.
- `reviewed` — **RingLeader: Efficiently Offloading Intra-Server
  Orchestration to NICs**, Lin et al., NSDI 2023.
  URL: `https://www.usenix.org/conference/nsdi23/presentation/lin`
  PDF: `https://www.usenix.org/system/files/nsdi23-lin.pdf`
  Why: Vessel cites SmartNIC/offloaded scheduling as an orthogonal direction;
  useful for comparing host-side preemption with NIC-assisted request
  placement for microsecond services, and relevant to future NIC/DPU-aware
  admission and response steering for million-session GPU DB deployments.
  The previously queued Shinjuku-Offload title/URL pairing pointed to a
  different paper; the NSDI 2023 primary source is RingLeader.
- `reviewed` — **R2P2: Making RPCs First-Class Datacenter Citizens**,
  Kogias et al., USENIX ATC 2019.
  URL: `https://www.usenix.org/conference/atc19/presentation/kogias-r2p2`
  Why: RingLeader builds on R2P2's request-level dispatch and JBSQ lineage;
  useful for deciding whether GPU DB should expose request descriptors,
  route classes, and response steering below the SQL execution layer.
- `reviewed` — **RackSched: A Microsecond-Scale Scheduler for
  Rack-Scale Computers**, Zhu et al., OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/zhu`
  PDF: `https://www.usenix.org/system/files/osdi20-zhu.pdf`
  Why: RingLeader contrasts against rack/server-level scheduling and
  centralized orchestration; useful for comparing host-only scheduling,
  rack-aware admission, and NIC-assisted dispatch for session-heavy GPU DB.
- `queued` — **The nanoPU: Redesigning the CPU-Network Interface to
  Minimize RPC Tail Latency**, Ibanez et al., arXiv 2020.
  URL: `https://arxiv.org/abs/2010.12114`
  Why: RingLeader contrasts against nanoPU-style CPU/network-interface
  redesign and per-service JBSQ; useful as a more radical endpoint for
  request dispatch, packet steering, and CPU/NIC co-design.
- `reviewed` — **SKQ: Event Scheduling for Optimizing Tail Latency in a
  Traditional OS Kernel**, Zhao, Gu, and Mashtizadeh, USENIX ATC 2021.
  URL: `https://www.usenix.org/conference/atc21/presentation/zhao-siyao`
  PDF: `https://www.usenix.org/system/files/atc21-zhao.pdf`
  Why: LibPreemptible contrasts against traditional-kernel event scheduling;
  useful for deciding how much GPU DB can improve pgwire/event-loop tail
  latency through event prioritization and delivery control before adopting
  hardware-assisted preemption or kernel bypass.
- `reviewed` — **Homa: A Receiver-Driven Low-Latency Transport Protocol Using
  Network Priorities**, Montazeri, Li, Alizadeh, and Ousterhout, SIGCOMM 2018.
  URL: `https://doi.org/10.1145/3230543.3230564`
  arXiv: `https://arxiv.org/abs/1803.09615`
  Why: R2P2 names Homa as a compatible congestion/transport direction; useful
  for comparing receiver-driven credits, message-size-aware priority, and
  bounded in-flight grants against GPU DB request/response-ring admission.
- `reviewed` — **CAM: Asynchronous GPU-Initiated, CPU-Managed SSD Management for
  Batching Storage Access**, Song et al., ICDE 2025.
  URL: `https://doi.org/10.1109/ICDE65448.2025.00175`
  Why: hybrid GPU-initiated but CPU-managed SSD control path; useful follow-up
  to Torp et al. for reducing GPU busy-wait/control-plane burn while still
  overlapping storage access with GPU compute.
- `reviewed` — **GPU-Initiated On-Demand High-Throughput Storage Access in the
  BaM System Architecture**, Qureshi et al., ASPLOS 2023.
  URL: `https://arxiv.org/abs/2203.04910`
  Why: foundational modern BaM design evaluated by Torp et al.; relevant to
  understanding GPU-side request queues, page caches, and when direct NVMe
  access helps over-resident data paths.
- `reviewed` — **GMT: GPU Orchestrated Memory Tiering for the Big Data Era**,
  Chang et al., ASPLOS 2024.
  URL: `https://doi.org/10.1145/3620666.3651353`
  Why: three-tier GPU/CPU/storage cache approach summarized by Torp et al.;
  relevant to future explicit tier promotion and demotion policies when reuse
  can justify CPU and GPU resource use.
- `reviewed` — **DRAGON: Breaking GPU Memory Capacity Limits with Direct NVM
  Access**, Markthub et al., SC 2018.
  URL: `https://doi.org/10.1109/SC.2018.00035`
  PDF: `https://www.osti.gov/servlets/purl/1489577`
  Why: BaM contrasts against DRAGON's UVM/page-fault path; useful for comparing
  transparent GPU page-fault extension against explicit GPU-initiated queues
  and cache-line admission.
- `reviewed` — **ActivePointers: A Case for Software Address Translation on
  GPUs**, Shahar, Bergman, and Silberstein, ISCA 2016.
  URL: `https://doi.org/10.1109/ISCA.2016.21`
  Why: DRAGON contrasts against software GPU address translation; useful as a
  cautionary baseline for per-access translation overhead and kernel
  modification costs in larger-than-GPU-memory execution.
- `queued` — **Towards High Performance Paged Memory for GPUs**, Zheng et al.,
  HPCA 2016.
  URL: `https://doi.org/10.1109/HPCA.2016.7446077`
  Why: DRAGON cites this hardware-paged-memory line as context for GPU page
  faulting; relevant to deciding when page-granular migration can help or hurt
  database resident snapshots and tiered execution.
- `queued` — **HippogriffDB: Balancing I/O and GPU Bandwidth in Big Data
  Analytics**, Li et al., PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p1647-li.pdf`
  Why: BaM cites HippogriffDB as a GPUDirect OLAP data-movement baseline;
  useful for database-specific comparison of CPU-orchestrated GPU storage
  transfers, I/O amplification, and GPU bandwidth saturation.
- `queued` — **GPU Database Systems Characterization and Optimization**,
  Cao et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol17/p441-cao.pdf`
  Why: Crystal-Opt follow-up discussed by Themis; useful for comparing
  GPU primitive optimization against warp-level pipeline load balancing.
- `reviewed` — **Tile-based Lightweight Integer Compression in GPU**,
  Shanbhag, Yogatama, Yu, and Madden, SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526132`
  PDF: `https://anilshanbhag.com/static/papers/gpufor_sigmod22.pdf`
  Why: GPU compression baseline relevant to temporary and resident integer
  compression when route-local encoding must trade HBM footprint against
  kernel occupancy and memory traffic.
- `queued` — **Design Trade-Offs for a Robust Dynamic Hybrid Hash Join**,
  Jahangiri, Carey, and Freytag, PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p2257-jahangiri.pdf`
  DOI: `https://doi.org/10.14778/3547305.3547327`
  Why: Saving Private Hash Join cites this recent dynamic partitioning work;
  useful for comparing spill granularity, adaptive partition policy, skew
  behavior, and robust hybrid hash join choices before mapping them to HBM,
  host-memory, and NVMe overflow routes.
- `queued` — **To Partition, or Not to Partition, That is the Join Question in
  a Real System**, Bandle, Giceva, and Neumann, SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3452831`
  Why: Saving Private Hash Join cites this hardware-conscious join evaluation;
  useful for deciding when GPU DB should avoid materializing/partitioning
  because memory traffic dominates, especially for selective joins and
  retained resident inputs.
- `reviewed` — **BtrBlocks: Efficient Columnar Compression for Data Lakes**,
  Kuschewski, Sauerwein, Alhomssi, and Leis, SIGMOD 2023.
  URL: `https://doi.org/10.1145/3589263`
  Why: modern columnar compression framework referenced by the compressed GPU
  analytics paper; useful for choosing host/cold-tier column encodings before
  deciding which forms are worth promoting into GPU-resident snapshots.
  Note: duplicate queue seed of the earlier BtrBlocks entry; reviewed once in
  the literature journal on 2026-06-04.
- `queued` — **Improving Execution Efficiency of Just-in-Time Compilation
  Based Query Processing on GPUs**, Paul et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol14/p202-paul.pdf`
  Why: Pyper baseline for Themis, with intra-warp shuffle and redistribution
  mechanics relevant to retained GPU pipeline fusion.
- `queued` — **Chimp: Efficient Lossless Floating Point Compression for Time
  Series Databases**, Liakos, Papakonstantinopoulou, and Kotidis, PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p3058-liakos.pdf`
  DOI: `https://doi.org/10.14778/3551793.3551848`
  Why: BtrBlocks compares Pseudodecimal Encoding against Chimp for double
  compression; useful for deciding whether float-heavy resident/cold segments
  need decimal-aware compression, time-series-oriented delta compression, or
  simple dictionary/FOR fallbacks.
- `queued` — **Towards Cost-Optimal Query Processing in the Cloud**, Leis and
  Kuschewski, PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p1606-leis.pdf`
  DOI: `https://doi.org/10.14778/3461535.3461549`
  Why: BtrBlocks uses cost-per-scan economics rather than only throughput;
  relevant to GPU DB tier placement because HBM, DRAM, NVMe, and future memory
  tiers should be evaluated by route cost, freshness, and bandwidth bottleneck,
  not raw decompression speed alone.
- `queued` — **Accelerating Multi-way Joins on the GPU**, Lai et al.,
  VLDB Journal 2022.
  URL: `https://doi.org/10.1007/s00778-021-00702-4`
  Why: fixed-granularity work-sharing baseline for Themis; relevant to
  multi-join skew, GMEM redistribution cost, and adaptive batching thresholds.
- `reviewed` — **Multiversion Concurrency with Bounded Delay and Precise
  Garbage Collection**, Ben-David et al., SPAA 2019.
  URL: `https://doi.org/10.1145/3323165.3323185`
  PDF: `https://www.cs.cmu.edu/~yihans/papers/concurrency.pdf`
  arXiv: `https://arxiv.org/abs/1803.08617`
  Why: follow-up theoretical and experimental foundation for P-Tree-style
  functional data structures, bounded reader/writer delay, and precise
  version reclamation; relevant to long retained snapshots and GC budgets.
- `reviewed` — **Morty: Scaling Concurrency Control with Re-Execution**,
  Burke et al., EuroSys 2023.
  URL: `https://doi.org/10.1145/3552326.3567500`
  Why: re-execution-based concurrency control cited by R-SMF; relevant to
  retrying or repairing conflicted transactions without throwing away all
  scheduling and snapshot work.
- `queued` — **Occam's Razor for Distributed Protocols**, Lai et al.,
  SoCC 2024.
  URL: `https://doi.org/10.1145/3698038.3698514`
  Why: R4 is Bonspiel's atomic-commit baseline; useful for separating
  unavoidable commit latency from abort/retry and contention-footprint costs
  in distributed or multi-owner transaction protocols.
- `reviewed` — **Caerus: Low-Latency Distributed Transactions for
  Geo-Replicated Systems**, Hildred et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol17/p469-hildred.pdf`
  Why: modern geo-replicated transaction protocol cited by Bonspiel; useful as
  a contrast for placement-aware commit and low-latency multi-partition
  transaction routing.
- `reviewed` — **Natto: Providing Distributed Transaction Prioritization for
  High-Contention Workloads**, Yang, Yan, and Wong, SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526161`
  PDF: `https://cs.uwaterloo.ca/~bernard/natto.pdf`
  Why: priority-based distributed transaction handling cited by Bonspiel;
  relevant to deciding whether GPU DB should prioritize long/remote or
  expensive route classes without wounding short local work.
- `reviewed` — **Carousel: Low-Latency Transaction Processing for
  Globally-Distributed Data**, Yan et al., SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3196912`
  PDF: `https://www.cs.cornell.edu/~hongbo/files/carousel-sigmod-2018.pdf`
  Why: Natto's base protocol; useful for evaluating fixed-set interactive
  transactions that overlap read/prepare, commit, and replication phases,
  which maps to GPU DB route descriptors with predeclared read/write sets.
  Journal entry added 2026-06-06.
- `queued` — **Consus: Taming the Paxi**, Escriva and van Renesse,
  arXiv 2016.
  URL: `https://arxiv.org/abs/1612.03457`
  Why: Carousel contrasts Consus as a geographically replicated transaction
  protocol that reaches consensus on commit outcome across datacenters; useful
  for comparing fixed-footprint route overlap with commit-decision consensus
  when GPU DB eventually has replicated owners or remote accelerator pools.

- `queued` — **Data Partitioning for In-Memory Systems: Myths, Challenges,
  and Opportunities**, Zhang, Deshmukh, and Patel, CIDR 2019.
  URL: `https://www.cidrdb.org/cidr2019/papers/p128-zhang-cidr19.pdf`
  Why: cited by the DaMoN 2019 allocator study around allocation-heavy
  partitioned hash joins; relevant to partition sizing, NUMA locality, and
  whether GPU DB should materialize, partition, or stream intermediate state.
- `queued` — **Analyzing the Impact of System Architecture on the Scalability
  of OLTP Engines for High-Contention Workloads**, Appuswamy et al.,
  PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol11/p121-appuswamy.pdf`
  Why: cited by the allocator study as an OLTP scalability baseline; useful
  for separating allocator, NUMA, latch, and architecture bottlenecks under
  high-contention transactional workloads.
- `reviewed` — **Take Out the TraChe: Maximizing (Tra)nsactional Ca(che) Hit
  Rate**, Cheng et al., OSDI 2023.
  URL: `https://www.usenix.org/conference/osdi23/presentation/cheng`
  Why: transaction-cache hit-rate work from the R-SMF/TAO line; relevant to
  session-heavy read/write routing, hot object placement, and keeping retained
  read snapshots useful under transactional cache pressure.
- `reviewed` — **Taurus: Lightweight Parallel Logging for In-Memory Database
  Management Systems**, Xia, Yu, Pavlo, and Devadas, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol14/p189-xia.pdf`
  arXiv: `https://arxiv.org/abs/2010.06760`
  Why: modern parallel logging with dependency vectors; useful follow-up for
  comparing explicit dependency encoding against RFA-style remote-flush
  avoidance in per-owner GPU DB WAL streams.
- `reviewed` — **Adaptive logging: Optimizing logging and recovery costs in
  distributed in-memory databases**, Yao et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915208`
  PDF: `https://www.cs.albany.edu/~jhh/courses/readings/yao.sigmod16.pdf`
  Why: distributed in-memory command/data logging tradeoff cited by Taurus;
  useful for deciding whether GPU DB should vary log payloads and recovery
  strategy by transaction class or partition.
- `queued` — **Let's Talk About Storage & Recovery Methods for Non-Volatile
  Memory Database Systems**, Arulraj, Pavlo, and Dulloor, SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2749441`
  Why: recovery-design follow-up cited by Adaptive Logging; relevant to
  separating volatile GPU/DRAM acceleration state from durable WAL/checkpoint
  truth as future memory tiers arrive.
- `reviewed` — **Guaranteeing Recoverability via Partially Constrained
  Transaction Logs**, Zhou et al., arXiv 2019.
  URL: `https://arxiv.org/abs/1901.06491`
  Why: Poplar-style partial log ordering tracks RAW/WAW dependencies instead of
  forcing one serial LSN stream; useful follow-up for per-owner GPU DB WAL
  streams and parallel crash recovery. Journal entry added 2026-06-06.
- `queued` — **Border-Collie: A Wait-free, Read-optimal Algorithm for
  Database Logging on Multicore Hardware**, Kim et al., SIGMOD 2019.
  URL: `https://doi.org/10.1145/3299869.3319869`
  Why: multicore logging algorithm cited by Taurus; relevant to minimizing
  reader-side coordination and cache coherence in the WAL publication path.
- `queued` — **Write-Behind Logging**, Arulraj, Perron, and Pavlo,
  PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol10/p337-arulraj.pdf`
  DOI: `https://doi.org/10.14778/3025111.3025116`
  Why: Poplar and the NVM recovery lane raise the boundary between
  write-ahead durability, dependency ordering, and near-instant recovery;
  WBL is a primary 2015-present contrast that logs changed regions after
  flushing updates on byte-addressable NVM.
- `queued` — **ExpressPass: End-to-End Credit-Based Congestion Control for
  Datacenters**, Cho et al., SIGCOMM 2017.
  URL: `https://doi.org/10.1145/3098822.3098843`
  PDF: `https://conferences.sigcomm.org/sigcomm/2017/files/program-ccr-final/91-Cho.pdf`
  Why: FNCC contrasts against credit-based congestion avoidance; useful for
  evaluating whether GPU DB response rings should use receiver-issued credits
  rather than only reactive queue-delay backpressure.
- `reviewed` — **Taurus Database: How to be Fast, Available, and Frugal in the
  Cloud**, Depoutovitch et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3386129`
  arXiv: `https://arxiv.org/abs/2412.02792`
  Why: cloud database storage architecture with append-only storage,
  replication, recovery, and constant-time snapshots; relevant to future
  cloud/disaggregated durability and snapshot tiers.
- `reviewed` — **Near Data Processing in Taurus Database**, Lin et al.,
  arXiv 2025.
  URL: `https://arxiv.org/abs/2506.20010`
  Why: direct Taurus follow-up that pushes selection, projection, and
  aggregation into the storage layer; useful for comparing GPU DB cold-tier
  pushdown with GPU-resident and CPU fallback routes. Journal entry added
  2026-06-05.
- `queued` — **Taurus MM: bringing multi-master to the cloud**, Depoutovitch
  et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p3488-depoutovitch.pdf`
  Why: direct Taurus follow-up on multi-master cloud database design; useful
  for comparing cross-owner write ordering, conflict handling, and snapshot
  publication in a disaggregated architecture.
- `queued` — **Near-Data Processing in Database Systems on Native
  Computational Storage under HTAP Workloads**, Vincon et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p1991-petrov.pdf`
  Why: Taurus NDP related work points to update-aware NDP; useful for
  comparing shared-state snapshot propagation and transactional guarantees
  when pushing cold-tier scans or aggregates into storage devices.
- `reviewed` — **Hybrid Garbage Collection for Multi-Version Concurrency Control
  in SAP HANA**, Lee et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2903734`
  Why: production HTAP MVCC garbage-collection design contrasted with Steam;
  useful for evaluating interval GC, long transaction handling, and practical
  memory-pressure policies.
- `queued` — **Read-log-update: A Lightweight Synchronization Mechanism for
  Concurrent Programming**, Matveev et al., SOSP 2015.
  URL: `https://doi.org/10.1145/2815400.2815406`
  Why: contrasted by bounded-delay MVCC as a two-version design where readers
  avoid blocking but writers still wait for quiescence; useful as a practical
  baseline for deciding when GPU DB can tolerate RCU/RLU-style grace periods
  versus precise snapshot release.
- `queued` — **Accelerating Hybrid Transactional/Analytical Processing Using
  Consistent Dual-Snapshot**, Li et al., DASFAA 2019.
  URL: `https://doi.org/10.1007/978-3-030-18576-3_41`
  Why: dual-snapshot HTAP design cited by Steam; relevant to separating
  retained analytical snapshots from fresh transactional visibility.
- `reviewed` — **Bao: Making Learned Query Optimization Practical**,
  Marcus et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3452838`
  PDF: `https://people.csail.mit.edu/tatbul/publications/bao_sigmod21.pdf`
  Why: learned hint-based optimizer baseline compared by PAR2QO; useful for
  deciding whether GPU route tuning should learn bounded hints around a
  deterministic planner rather than replace route rules.
- `reviewed` — **Online Schema Evolution is (Almost) Free for Snapshot
  Databases**, Hu et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol16/p140-hu.pdf`
  Why: modern snapshot-database follow-up discovered while reviewing
  serializable MVCC; relevant to DDL/catalog generation changes, retained
  snapshots, and whether schema evolution can avoid blocking GPU-resident
  readers.
- `queued` — **BullFrog: Online Schema Evolution via Lazy Evaluation**,
  Bhattacherjee et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3452842`
  PDF: `https://www.cs.umd.edu/~mwh/papers/bullfrog-sigmod.pdf`
  Why: lazy physical migration for online schema changes in PostgreSQL;
  useful contrast to Tesseract's MVCC-native out-of-place migration and
  CDC design for retained snapshots and catalog generations.
- `queued` — **Learned Query Superoptimization**, Trummer, arXiv 2023.
  URL: `https://arxiv.org/abs/2303.15308`
  Why: Bao follow-up direction that learns improvements beyond bounded native
  hint sets; useful contrast once GPU DB has a safe route-certificate action
  space and needs to decide whether to widen learned route search.
- `queued` — **Learned Query Optimizer in Alibaba MaxCompute: Challenges,
  Analysis, and Solutions**, Zhou et al., arXiv 2026.
  URL: `https://arxiv.org/abs/2602.07336`
  Why: modern deployability-focused learned-optimizer paper from a production
  cloud analytics setting; useful for stress-testing Bao-style route learning
  against dynamic execution environments and missing statistics.

- `reviewed` — **AGILE: Lightweight and Efficient Asynchronous GPU-SSD
  Integration**, Yang et al., SC 2025.
  URL: `https://arxiv.org/abs/2504.19365`
  Why: modern asynchronous GPU-centric SSD access library; useful contrast to
  CAM's CPU-managed control plane and BaM's synchronous GPU polling path.
- `reviewed` — **Hyperion: Co-Optimizing SSD Access and GPU Computation for
  Cost-Efficient GNN Training**, ICDE 2025.
  URL: `https://doi.org/10.1109/ICDE65448.2025.00031`
  Why: modern GPU-initiated asynchronous SSD access and cache co-optimization;
  useful follow-up for deciding when GPU DB over-resident execution should
  jointly plan GPU work, SSD reads, and CPU/GPU memory cache placement.
- `queued` — **Asynchrony and GPUs: Bridging this Dichotomy for I/O with
  AGIO**, Han et al., ASPLOS 2026.
  URL: `https://doi.org/10.1145/3779212.3790130`
  Why: apparent follow-up from the AGILE/GMT/BaM research line; likely useful
  for deciding whether asynchronous GPU-originated I/O should be exposed as a
  library API, compiler/runtime primitive, or GPU execution-owner service.
- `queued` — **Breaking the Storage-Compute Bottleneck in Billion-Scale ANNS:
  A GPU-Driven Asynchronous I/O Framework**, arXiv 2025.
  URL: `https://arxiv.org/abs/2507.10070`
  Why: GPU-driven async I/O plus compute/I/O balancing in an over-resident
  search workload; useful for route policies that choose graph degree, batch
  size, and storage prefetch depth together.
- `reviewed` — **TAS: TCP Acceleration as an OS Service**, Kaufmann et al.,
  EuroSys 2019.
  URL: `https://os.mpi-sws.org/projects/tas.html`
  PDF: `https://homes.cs.washington.edu/~arvind/papers/flextcp.pdf`
  Why: Shenango's related kernel-bypass/runtime context points to TAS as a
  multi-tenant TCP acceleration service; useful for comparing a central
  IO/runtime service against GPU DB's planned IO-worker and response-ring
  topology.
- `queued` — **NetKernel: Making Network Stack Part of the Virtualized
  Infrastructure**, Gamage et al., arXiv 2019.
  URL: `https://arxiv.org/abs/1903.07119`
  Why: TAS contrasts itself with network-stack-as-a-service designs; useful
  for deciding when GPU DB session/network state should remain an in-process
  IO-worker service versus an externalized infrastructure service.
- `reviewed` — **A CXL-Powered Database System: Opportunities and Challenges**,
  Guo and Li, ICDE 2024.
  URL: `https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/CXL_ICDE.pdf`
  Why: Pasha contrasts itself with this CXL database position paper; useful
  for a broader capability matrix of CXL memory, pooling, coherence, and
  database architecture constraints.
- `queued` — **So Far and yet so Near - Accelerating Distributed Joins with
  CXL**, Baumstark et al., DaMoN 2024.
  URL: `https://doi.org/10.1145/3662010.3663449`
  Why: Pasha cites this CXL data-management work; relevant to deciding when
  CXL/shared-memory tiers help cross-partition joins versus GPU or CPU
  partition-local execution.
- `reviewed` — **Handling Highly Contended OLTP Workloads Using Fast Dynamic
  Partitioning**, Prasaad, Cheung, and Suciu, SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3389764`
  Why: Strife is a key partitioner baseline used by the runtime-conflict
  scheduler; useful for hot-key partitioning, residual transaction handling,
  and contention-aware owner assignment.
- `queued` — **Consolidating Concurrency Control and Consensus for Commits
  under Conflicts**, Mu et al., OSDI 2016.
  URL: `https://www.usenix.org/conference/osdi16/technical-sessions/presentation/mu`
  PDF: `https://www.usenix.org/system/files/conference/osdi16/osdi16-mu.pdf`
  Why: NCC compares against Janus-style transaction reordering and notes the
  cost of exchanging ordering information; useful for evaluating whether GPU
  DB should ever merge cross-owner commit ordering with replication or keep
  owner-local WAL publication simpler.
- `reviewed` — **Scheduling OLTP Transactions via Learned Abort Prediction**,
  Sheng, Tomasic, Zhang, and Pavlo, aiDM 2019.
  URL: `https://doi.org/10.1145/3329859.3329871`
  PDF: `https://db.cs.cmu.edu/papers/2019/a1-sheng.pdf`
  Why: lightweight learned transaction-to-thread assignment cited by TSkd;
  relevant to admission-time prediction before choosing an owner, CPU route,
  or deferred execution path.
- `reviewed` — **Intelligent Transaction Scheduling via Conflict Prediction in
  OLTP DBMS**, Zhang, Tomasic, and Pavlo, arXiv 2024.
  URL: `https://arxiv.org/abs/2409.01675`
  Why: longer modern follow-up to abort-prediction scheduling that studies
  lightweight history/state policies, canonical references, continuous
  adaptation, and workload-distribution shifts; useful for deciding whether
  GPU DB should start with interpretable conflict-history admission before
  heavier learned schedulers.
- `reviewed` — **Design Principles for Scaling Multi-core OLTP Under High
  Contention**, Ren, Faleiro, and Abadi, SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882955`
  PDF: `http://www.cs.umd.edu/~abadi/papers/orthrus-sigmod16.pdf`
  Why: TSkd contrasts with Orthrus-style separation of transaction logic and
  concurrency-control cores; useful for deciding whether GPU DB mutation
  owners, read-snapshot workers, and conflict/admission workers should be
  separated under hot-key contention. Journal entry added 2026-06-06; DOI
  corrected in the journal citation to `10.1145/2882903.2882958`.
- `queued` — **Self-Driving Database Management Systems**, Pavlo et al.,
  CIDR 2017.
  URL: `https://www.cidrdb.org/cidr2017/papers/p42-pavlo-cidr17.pdf`
  Why: TSkd's evaluation uses Peloton, whose self-driving DBMS direction is
  relevant to collecting workload evidence, adapting scheduling state, and
  deciding how much conflict-history telemetry should feed GPU DB route
  admission before introducing heavier learned control loops.
- `reviewed` — **Polyjuice: High-Performance Transactions via Learned
  Concurrency Control**, Wang et al., OSDI 2021.
  URL: `https://www.usenix.org/conference/osdi21/presentation/wang-jiachen`
  Why: learned concurrency-control policy selection cited by TSkd; useful as
  a contrast to deterministic owner-queue rules and runtime-conflict telemetry.
- `reviewed` — **Centiman: Elastic, High Performance Optimistic Concurrency
  Control by Watermarking**, Ding et al., SoCC 2015.
  URL: `https://doi.org/10.1145/2806777.2806837`
  PDF:
  `https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/centiman_socc_2015.pdf`
  Why: OCC validator/storage architecture cited by the batching paper; useful
  for comparing watermark-based validation, decoupled compute/storage, and
  versioned write installation against GPU DB owner boundaries.
- `reviewed` — **QueCC: A Queue-Oriented, Control-Free Concurrency
  Architecture**, Qadah and Sadoghi, Middleware 2018.
  URL: `https://doi.org/10.1145/3274808.3274810`
  Why: queue-oriented transaction execution cited by the batching paper;
  relevant to deterministic owner queues, queue-local ordering, and whether
  control-free execution can coexist with WAL-before-visibility.
- `reviewed` — **Mostly-Optimistic Concurrency Control for Highly Contended
  Dynamic Workloads on a Thousand Cores**, Wang and Kimura, PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol10/p49-wang.pdf`
  Why: hybrid OCC/pessimistic contention handling cited by the batching paper;
  useful for deciding when GPU DB owner queues should switch from optimistic
  validation to contention-aware ordered execution.
- `reviewed` — **Using Read Promotion and Mixed Isolation Levels for Performant
  Yet Serializable Execution of Transaction Programs**, Vandevoort et al.,
  PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p2846-vandevoort.pdf`
  DOI: `https://doi.org/10.14778/3746405.3746412`
  Why: modern mixed-isolation and read-promotion follow-up discovered while
  reviewing Centiman's read-only bypass path; relevant to route-level choices
  that keep serializability while letting safe read-heavy templates avoid the
  strongest validation path.
- `reviewed` — **Detecting Robustness against MVRC for Transaction Programs
  with Predicate Reads**, Vandevoort et al., EDBT 2023.
  URL: `https://doi.org/10.48786/edbt.2023.48`
  Why: read-promotion paper cites this as a direction for transaction programs
  with predicate reads; relevant to GPU DB route templates that include ranges,
  prefix predicates, and phantom-sensitive retained scans.
- `queued` — **Serializable use of Read Committed isolation level**, Alomari
  and Fekete, AICCSA 2015.
  URL: `https://doi.org/10.1109/AICCSA.2015.7507195`
  Why: EDBT 2023 refines this type-I counterflow-cycle robustness test; useful
  as the simpler baseline for deciding whether route-template certification
  needs predicate-aware type-II graph checks.
- `queued` — **Checking Robustness Against Snapshot Isolation**, Beillahi,
  Bouajjani, and Enea, CAV 2019.
  URL: `https://doi.org/10.1007/978-3-030-25540-4_16`
  Why: EDBT 2023 cites it as static robustness work for snapshot isolation;
  useful for comparing MVRC route templates with future snapshot-isolation
  certificates for retained GPU reads.
- `reviewed` — **Allocating Isolation Levels to Transactions in a Multiversion
  Setting**, Vandevoort, Ketsman, and Neven, PODS 2023.
  URL: `https://doi.org/10.1145/3584372.3588672`
  PDF:
  `https://documentserver.uhasselt.be/bitstream/1942/42231/2/main%20%281%29.pdf`
  Why: direct predecessor to the view/conflict robustness result; useful for
  automated RC/SI/SSI route-template allocation before deciding whether a
  retained read or write template can bypass the strongest isolation path.
- `reviewed` — **When View- and Conflict-Robustness Coincide for Multiversion
  Concurrency Control**, Vandevoort et al., PACMMOD 2024.
  URL: `https://doi.org/10.1145/3651592`
  arXiv: `https://arxiv.org/abs/2403.17665`
  Why: modern mixed-isolation robustness follow-up; useful for deciding whether
  GPU DB can rely on conflict-robustness checks for route templates or needs a
  broader view-robustness model for MVCC-visible retained reads.
- `reviewed` — **Robustness against Read Committed for Transaction Templates**,
  Vandevoort et al., PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p2141-vandevoort.pdf`
  DOI: `https://doi.org/10.14778/3476249.3476268`
  Why: transaction-template robustness and selective read-promotion baseline;
  useful for benchmarking whether known OLTP command shapes can safely use a
  cheaper RC-style route while preserving serializable outcomes. Journal entry
  added 2026-06-06.
- `queued` — **Robustness Against Read Committed for Transaction Templates with
  Functional Constraints**, Vandevoort et al., ICDT 2022.
  URL: `https://arxiv.org/abs/2201.05021`
  Why: extends robustness analysis with functional constraints; useful for GPU
  DB templates where primary keys, unique constraints, and derived keys may
  prove that cheaper route-level isolation is still safe.
- `reviewed` — **Detock: High Performance Multi-region Transactions at Scale**,
  Li et al., SIGMOD 2023.
  URL:
  `https://api.drum.lib.umd.edu/server/api/core/bitstreams/5619e587-270c-4859-8a3e-8947e2bc9928/content`
  Why: Caerus compares against Detock as a concurrent single-WAN-round
  graph-based geo-transaction design; useful for contrasting integrated
  concurrency-control/execution with Caerus-style scheduler-independent
  partial-order sequencing.
- `reviewed` — **Tiga: Accelerating Geo-Distributed Transactions with
  Synchronized Clocks**, Geng, Mu, Sivaraman, and Prabhakar, SOSP 2025.
  URL: `https://anirudhsk.github.io/papers/tiga_sosp.pdf`
  DOI: `https://doi.org/10.1145/3731569.3764854`
  Technical report: `https://arxiv.org/abs/2509.05759`
  Why: modern synchronized-clock geo-transaction design related to Caerus and
  Detock; useful for deciding whether owner-boundary timestamps can reduce
  coordination without relying on full deterministic execution.
- `reviewed` — **NCC: Natural Concurrency Control for Strictly Serializable
  Datastores by Avoiding the Timestamp-Inversion Pitfall**, Lu, Mu, Sen,
  and Lloyd, OSDI 2023.
  URL: `https://www.usenix.org/conference/osdi23/presentation/lu`
  PDF: `https://www.usenix.org/system/files/osdi23-lu.pdf`
  Why: Tiga repeatedly contrasts against NCC's timestamp-inversion handling;
  directly relevant to cross-owner strict-serializability tests and whether
  GPU DB needs generation agreement for indirect dependency chains.
- `queued` — **Nezha: Deployable and High-Performance Consensus Using
  Synchronized Clocks**, Geng, Sivaraman, Prabhakar, and Rosenblum, PVLDB
  2023.
  URL: `https://doi.org/10.14778/3574245.3574250`
  Why: Tiga builds on Nezha-style synchronized-clock consensus; useful as a
  narrower source on depending on clock synchronization for performance while
  preserving correctness with explicit recovery paths.
- `reviewed` — **Epoch-based Optimistic Concurrency Control in Geo-replicated
  Databases**, arXiv 2026.
  URL: `https://arxiv.org/abs/2602.21566`
  Why: modern geo-replicated concurrency-control work with epoch-based
  asynchronous replication and deterministic re-execution; useful as a
  follow-up for owner-local epochs, partial WAL ordering, and conflict-graph
  retry policies.
- `reviewed` — **MEMTIS: Efficient Memory Tiering with Dynamic Page
  Classification and Page Size Determination**, Lee et al., SOSP 2023.
  URL: `https://doi.org/10.1145/3600006.3613167`
  Why: hardware-counter-guided page classification and dynamic page-size
  decisions compared against NOMAD; relevant to tier-placement telemetry,
  access-frequency sampling, and huge-page/subpage placement tradeoffs.
- `reviewed` — **Larger-than-Memory Data Management on Modern Storage Hardware
  for In-Memory OLTP Database Systems**, Ma et al., DaMoN 2016.
  URL: `https://doi.org/10.1145/2933349.2933358`
  PDF: `https://db.cs.cmu.edu/papers/2016/ma-damon2016.pdf`
  Why: LeanStore contrasts with anti-caching-style cold tuple movement; this
  2016 evaluation gives OLTP-specific eviction/retrieval policy evidence
  across modern storage devices.
- `queued` — **"Anti-Caching"-based Elastic Memory Management for Big Data**,
  Zhang et al., ICDE 2015.
  URL: `https://doi.org/10.1109/ICDE.2015.7113330`
  Why: modern enough follow-up in the anti-caching line; useful contrast for
  tuple-granular cold movement versus LeanStore-style page/index-transparent
  placement.
- `reviewed` — **Page As You Go: Piecewise Columnar Access In SAP HANA**,
  Sherkat et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2903729`
  Why: production columnar cold-block access design cited by LeanStore; useful
  for deciding whether GPU DB should page full resident segments, column
  groups, or smaller compressed blocks. Journal entry exists from 2026-06-05;
  this stale duplicate was marked reviewed on 2026-06-05.
- `reviewed` — **TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory**,
  Al Maruf et al., ASPLOS 2023.
  URL: `https://doi.org/10.1145/3582016.3582063`
  PDF: `https://symbioticlab.org/publications/files/tpp%3Aasplos23/tpp-asplos23.pdf`
  Why: Linux CXL transparent page placement baseline compared by NOMAD;
  useful for deciding where OS-managed promotion/demotion is enough and where
  GPU DB needs explicit DBMS placement handles.
- `reviewed` — **TAOBench: An End-to-End Benchmark for Social Network
  Workloads**, Cheng et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p1965-cheng.pdf`
  Why: DeToX's most important real-world workload source; useful for a
  session-heavy transactional cache/residency benchmark with correlated
  point reads, read transactions, writes, skew, and contaminated hot keys.
- `queued` — **FlightTracker: Consistency across Read-Optimized Online Stores
  at Facebook**, Shi et al., OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/shi`
  PDF: `https://www.usenix.org/system/files/osdi20-shi.pdf`
  Why: TAOBench's TAO context depends on read-optimized caches and consistency
  tokens; useful for comparing retained GPU snapshot freshness, read-your-writes
  guarantees, and route-local consistency tickets under high fan-out reads.
- `queued` — **RAMP-TAO: Layering Atomic Transactions on Facebook's Online TAO
  Data Store**, Cheng et al., PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p3014-cheng.pdf`
  Why: TAOBench discusses TAO's read-only and write-only transactional needs;
  RAMP-TAO is the natural follow-up for cache-friendly read transactions,
  fractured-read avoidance, and opt-in transactional metadata for read-dominant
  social graph workloads.
- `queued` — **ChronoCache: Predictive and Adaptive Mid-Tier Query Result
  Caching**, Glasbergen et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3380593`
  Why: DeToX's predictive caching baseline; useful for comparing dependency
  prefetching and adaptive query-result caching against transaction-hit-rate
  placement for retained GPU snapshots.
- `queued` — **HetExchange: Encapsulating Heterogeneous CPU-GPU Parallelism in
  JIT Compiled Engines**, Chrysogelos et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p544-chrysogelos.pdf`
  Why: Mordred contrasts its segment-level placement with HetExchange's
  heterogeneous exchange operator; useful for deciding whether GPU DB should
  express CPU/GPU split execution as planner operators, runtime route groups,
  or both.
- `queued` — **Pump Up the Volume: Processing Large Data on GPUs with Fast
  Interconnects**, Lutz et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3389705`
  Why: Mordred notes that interconnect bandwidth changes CPU/GPU placement
  economics; this multi-GPU/NVLink-oriented follow-up is relevant to future
  NVLink/CXL/GPUDirect tiers and over-resident execution.
- `queued` — **Meerkat: Multicore-Scalable Replicated Transactions Following
  the Zero-Coordination Principle**, Szekeres et al., EuroSys 2020.
  URL: `https://www.microsoft.com/en-us/research/uploads/prod/2020/05/meerkat-eurosys20.pdf`
  Why: Morty contrasts itself with Meerkat-style integrated commit and
  replicated transaction processing; useful for comparing decentralized
  agreement, per-core validation, and contention behavior against
  re-execution or owner-queue designs.
- `queued` — **Enabling CXL Memory Expansion for In-Memory Database
  Management Systems**, Ahn et al., DaMoN 2022.
  URL: `https://doi.org/10.1145/3533737.3535090`
  Why: CXL DB position paper cites this as direct IMDBMS evidence; useful
  for quantifying OLTP/OLAP impact from placing hot and cold database state in
  CXL-attached memory.
- `queued` — **ZNS: Avoiding the Block Interface Tax for Flash-based SSDs**,
  Bjorling et al., USENIX ATC 2021.
  URL: `https://www.usenix.org/conference/atc21/presentation/bjorling`
  Why: Hyperion's GPU-initiated SSD path still pays block-interface and
  request-granularity costs; ZNS is useful for evaluating whether future GPU
  DB cold-tier segments should expose zone-aware allocation, append, and
  placement contracts instead of relying on conventional block IO.
- `reviewed` — **SplinterDB: Closing the Bandwidth Gap for NVMe Key-Value
  Stores**, Conway et al., USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/conway`
  Why: Hyperion's argument depends on extracting cheap NVMe bandwidth;
  SplinterDB is a storage-engine follow-up for comparing write-optimized
  indexing, compaction, and bandwidth utilization against GPU DB cold-tier
  point lookup and segment-directory designs. Duplicate queue entry corrected
  to reviewed on 2026-06-06.
- `queued` — **Elastic Use of Far Memory for In-Memory Database Management
  Systems**, Lee et al., DaMoN 2023.
  URL: `https://doi.org/10.1145/3592980.3595311`
  Why: CXL DB position paper cites this as a CXL pooling study; useful for
  measuring bandwidth limits and when explicit DBMS placement beats transparent
  far-memory expansion.
- `reviewed` — **Database Kernels: Seamless Integration of Database Systems and
  Fast Storage via CXL**, Lee et al., CIDR 2024.
  URL: `https://www.cidrdb.org/cidr2024/papers/p43-lee.pdf`
  Why: CXL DB position paper cites CXL-enabled SSD/storage integration; useful
  for deciding whether GPU DB cold-tier operators should run through a
  database-owned CXL storage service.
- `queued` — **dLSM: An LSM-based Index for Memory Disaggregation**, Wang et
  al., ICDE 2023.
  URL: `https://doi.org/10.1109/ICDE55515.2023.00217`
  Why: CXL DB position paper cites disaggregated-memory index design; useful
  for comparing B-tree node placement with LSM-style far-memory indexes for
  cold partitions and write-heavy tables.
- `reviewed` — **Mako: Speculative Distributed Transactions with
  Geo-Replication**, Shen et al., OSDI 2025.
  URL: `https://www.usenix.org/conference/osdi25/presentation/shen-weihai`
  PDF: `https://www.usenix.org/system/files/osdi25-shen-weihai.pdf`
  Why: Minerva contrasts against Mako's speculative geo-replicated
  transaction path; useful for comparing execution/replication decoupling,
  deterministic replay, and geo-replication costs against owner-local GPU DB
  WAL and snapshot publication.
- `reviewed` — **Fast Commitment for Geo-Distributed Transactions via
  Decentralized Co-coordinators**, Zhang et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p2555-hu.pdf`
  DOI: `https://doi.org/10.14778/3675034.3675046`
  Why: Mako compares against D2PC as a modern geo-distributed transaction
  baseline; useful for evaluating whether decentralized commit coordination
  can reduce owner or shard-leader bottlenecks without speculative rollback
  machinery.
- `reviewed` — **DINT: Fast In-Kernel Distributed Transactions with eBPF**,
  Zhou et al., NSDI 2024.
  URL: `https://www.usenix.org/conference/nsdi24/presentation/zhou-yang`
  PDF: `https://www.usenix.org/system/files/nsdi24-zhou-yang.pdf`
  Why: Mako cites DINT among recent fast distributed transaction systems;
  relevant to comparing kernel/eBPF-assisted transaction routing with GPU DB's
  user-space rings, admission queues, and protocol-edge ownership.
- `reviewed` — **Electrode: Accelerating Distributed Protocols with eBPF**,
  Zhou, Wang, Dharanipragada, and Yu, NSDI 2023.
  URL: `https://www.usenix.org/conference/nsdi23/presentation/zhou`
  PDF: `https://www.usenix.org/system/files/nsdi23-zhou.pdf`
  Why: DINT builds on the broader idea of moving distributed-protocol
  frequent paths into eBPF; useful for deciding whether GPU DB should use
  kernel-side protocol classification or keep all route-state transitions in
  user-space owner rings.
- `queued` — **SPRIGHT: Extracting the Server from Serverless Computing!
  High-Performance eBPF-Based Event-Driven, Shared-Memory Processing**,
  Qi et al., SIGCOMM 2022.
  URL: `https://doi.org/10.1145/3544216.3544225`
  Why: Electrode cites SPRIGHT as another eBPF/shared-memory fast path; useful
  for comparing packet-edge acceleration with shared-memory response pipelines
  and sidecar/proxy bypass for high-session SQL routing.
- `queued` — **XRP: In-Kernel Storage Functions with eBPF**, Zhong et al.,
  OSDI 2022.
  URL: `https://www.usenix.org/conference/osdi22/presentation/zhong`
  Why: Electrode discusses XRP as a related eBPF offload for storage
  functions; useful for deciding whether tiny validated index or metadata
  probes can live near the kernel/storage boundary without violating SQL
  visibility or WAL recovery contracts.
- `reviewed` — **Xenic: SmartNIC-Accelerated Distributed Transactions**,
  Schuh et al., SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483555`
  PDF: `https://homes.cs.washington.edu/~arvind/papers/xenic.pdf`
  Why: DINT contrasts against SmartNIC transaction offload; useful for
  comparing eBPF/kernel-side admission with future NIC/DPU-side transaction
  routing and request steering. Journal entry added 2026-06-06; the stale
  DOI was corrected during review.
- `queued` — **AlNiCo: SmartNIC-accelerated Contention-aware Request
  Scheduling for Transaction Processing**, Li et al., USENIX ATC 2022.
  URL: `https://www.usenix.org/conference/atc22/presentation/li-junru`
  PDF: `https://www.usenix.org/system/files/atc22-li-junru.pdf`
  Why: discovered while reviewing Xenic; useful follow-up on using
  SmartNIC-side compact feature vectors and feedback to steer incoming
  transactions to CPU workers while reducing contention, which maps to GPU
  DB gateway admission and owner-ring selection.
- `reviewed` — **Ocean Vista: Gossip-based Visibility Control for Speedy
  Geo-Distributed Transactions**, Fan and Golab, PVLDB 2019.
  URL: `https://doi.org/10.14778/3342263.3342627`
  PDF: `https://www.vldb.org/pvldb/vol12/p1471-fan.pdf`
  Why: Mako contrasts against integrated replication and concurrency-control
  protocols; useful for comparing visibility-control metadata against GPU DB
  snapshot publication and route certificates.
- `reviewed` — **A Hybrid Approach to Integrating Deterministic and
  Non-Deterministic Concurrency Control in Database Systems**, Hong et al.,
  PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p1376-lu.pdf`
  DOI: `https://doi.org/10.14778/3718057.3718066`
  Code: `https://github.com/dbiir/HDCC`
  Why: Minerva relates this HDCC line to Aria-style OCC plus deterministic
  rescheduling; useful for deciding when GPU DB should switch from optimistic
  validation to deterministic owner execution under high contention.
- `reviewed` — **TDSQL: Tencent Distributed Database System**, Chen et al.,
  PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p3869-chen.pdf`
  DOI: `https://doi.org/10.14778/3685800.3685812`
  Why: HDCC cites TDSQL as a production distributed DBMS context for hybrid
  concurrency-control mechanisms; useful for comparing research-grade Calvin/OCC
  integration with deployed MVCC, logging, failover, and transaction routing.
  Journal entry added 2026-06-06; the stale DOI was corrected during review.
- `queued` — **Scalable Replay-Based Replication For Fast Databases**, Qin,
  Goel, and Brown, PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p2025-qin.pdf`
  Why: TDSQL's production replication path raises backup catch-up and log
  transfer as throughput risks; replay-based replication is a primary follow-up
  for comparing transaction-input shipping, parallel backup replay, and
  replication bandwidth against GPU DB's future WAL/replica publication path.
- `reviewed` — **Epoch-Based Commit and Replication in Distributed OLTP
  Databases**, Lu et al., PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p743-lu.pdf`
  Why: COCO is Minerva's epoch-commit baseline; useful for comparing
  epoch-sized commit, replication, and validation units with GPU DB
  WAL-before-visibility batches and retained snapshot publication. Journal
  entry already exists; this stale duplicate was corrected from `queued` to
  `reviewed` on 2026-06-06.
- `queued` — **Basil: Breaking up BFT with ACID (transactions)**,
  Suri-Payer et al., SOSP 2021.
  URL: `https://www.cs.cornell.edu/~matthelb/papers/basil-sosp21.pdf`
  Why: Morty compares its MVTSO lineage with Basil's transactional BFT path;
  useful as a future contrast for timestamp-order validation, delayed write
  visibility, and commit coordination when durability/replication becomes a
  GPU DB architecture question.
- `queued` — **Balsa: Learning a Query Optimizer Without Expert
  Demonstrations**, Yang et al., SIGMOD 2022.
  URL: `https://arxiv.org/abs/2201.01441`
  PDF: `https://zongheng.me/pubs/balsa-sigmod2022.pdf`
  Why: LOGER compares against Balsa's simulator-bootstrapped DRL optimizer;
  useful for deciding whether GPU DB route learning can bootstrap from
  simulation when benchmark execution is expensive or hardware is changing.
- `queued` — **Neo: A Learned Query Optimizer**, Marcus et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf`
  Why: CARPO cites Neo as an end-to-end learned optimizer baseline; useful for
  contrasting full learned plan search against bounded GPU DB route ranking
  and deterministic fallback rules.
- `queued` — **Optimizing Distributed Protocols with Query Rewrites**,
  Chu et al., SIGMOD/PACMMOD 2024.
  URL: `https://dl.acm.org/doi/10.1145/3654906`
  Author page:
  `https://www.microsoft.com/en-us/research/publication/optimizing-distributed-protocols-with-query-rewrites/`
  Why: follow-up discovered while reviewing WeBridge; uses rule-driven
  rewrites, dependency analysis, and spatiotemporal correctness reasoning to
  scale coordination protocols, which may inform stored-procedure route
  synthesis, owner split rules, and safe batching of GPU DB transaction
  frontiers.
- `queued` — **Is Your Learned Query Optimizer Behaving As You Expect? A
  Machine Learning Perspective**, Lehmann et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p1565-lehmann.pdf`
  Why: modern learned-optimizer diagnostic work; useful after LOGER/PARQO
  coverage to keep learned GPU route suggestions explainable, bounded, and
  testable instead of treating model output as an opaque planner authority.
- `queued` — **Steering Query Optimizers: A Practical Take on Big Data
  Workloads**, Negi et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457297`
  Why: AutoSteer builds on query-span and steering ideas from this work; useful
  for comparing random, greedy, and expert-guided exploration of bounded
  optimizer knobs before applying route learning to GPU DB planner decisions.
- `reviewed` — **RackSched: A Microsecond-Scale Scheduler for Rack-Scale
  Computers**, Zhu et al., OSDI 2020.
  URL: `https://arxiv.org/abs/2010.05969`
  PDF: `https://www.usenix.org/system/files/osdi20-zhu.pdf`
  Why: duplicate RackSched queue entry; reviewed from the OSDI 2020 primary
  source in the journal. Shinjuku follow-up direction for rack-scale request scheduling;
  relevant to comparing centralized, partitioned, and rack-aware admission
  when GPU DB eventually spans multiple owners, devices, or nodes.
- `reviewed` — **SLOG: Serializable, Low-latency, Geo-replicated
  Transactions**, Ren et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p1747-ren.pdf`
  Why: deterministic-transaction follow-up direction related to Aria's
  replication motivation; useful for comparing input replication,
  deterministic ordering, and low-latency commit paths when GPU DB eventually
  separates local owner domains from replicated durability.
- `queued` — **ALOHA-KV: High Performance Read-only and Write-only
  Distributed Transactions**, Fan, Golab, and Morrey, SoCC 2017.
  URL: `https://doi.org/10.1145/3127479.3127487`
  PDF:
  `https://acmsocc.org/2017/assets/socc17-finalpapers/socc17-final249-acmpaginated.pdf`
  Why: Ocean Vista builds on the epoch/watermark line from ALOHA-KV; useful
  for comparing route classes where GPU DB can prove requests are read-only or
  write-only and batch visibility without full read/write transaction
  coordination.
- `queued` — **Scalable Transaction Processing Using Functors**, Fan and
  Golab, ICDCS 2018.
  URL: `https://doi.org/10.1109/ICDCS.2018.00101`
  Why: Ocean Vista uses functor placeholders for read-write transactions;
  useful for deciding whether GPU DB can store deterministic transaction
  continuations at visibility boundaries and execute them after snapshot
  watermarks advance.
- `queued` — **An Evaluation of Distributed Concurrency Control**,
  Harding et al., PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p553-harding.pdf`
  Why: Aria's evaluation highlights the difficulty of comparing deterministic
  and nondeterministic concurrency-control systems fairly; this Deneva study is
  useful for shaping GPU DB transaction/admission benchmarks across locking,
  OCC, timestamp ordering, deterministic execution, MVCC, and partition-owner
  designs.
- `queued` — **FaSST: Fast, Scalable and Simple Distributed Transactions with
  Two-Sided RDMA Datagram RPCs**, Kalia et al., OSDI 2016.
  URL: `https://www.usenix.org/conference/osdi16/technical-sessions/presentation/kalia`
  Why: Chiller contrasts modern fast-network transaction execution with older
  distributed-transaction assumptions; FaSST is a primary RDMA OLTP baseline
  for bounded RPC buffers, fast commit messaging, and partition-local execution.
- `queued` — **Rethinking Database High Availability with RDMA Networks**,
  Zamanian et al., VLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p1637-zamanian.pdf`
  Why: Chiller's inner-region commit depends on careful replication and
  recovery; this follow-up is relevant to GPU DB durability and availability
  once owner domains, WAL publication, and resident refresh become distributed.
- `reviewed` — **Towards an Adaptable Systems Architecture for Memory Tiering at
  Warehouse-Scale**, Duraisamy et al., ASPLOS 2023.
  URL: `https://doi.org/10.1145/3582016.3582031`
  Why: TMTS is the warehouse-scale memory-tiering design contrasted by MEMTIS;
  useful for SLO-oriented demotion, cold-page histograms, and how OS/runtime
  tiering policies interact with application-specific placement.
- `queued` — **Mosaic Pages: Big TLB Reach with Small Pages**, Gosakan et al.,
  ASPLOS 2023.
  URL: `https://doi.org/10.1145/3582016.3582021`
  Why: MEMTIS cites Mosaic Pages for address-translation pressure in large
  memory systems; relevant to choosing GPU DB host-page and resident-segment
  granularity without blindly relying on huge pages.
- `queued` — **Beyond malloc efficiency to fleet efficiency: a hugepage-aware
  memory allocator**, Hunter et al., OSDI 2021.
  URL: `https://www.usenix.org/conference/osdi21/presentation/hunter`
  PDF: `https://www.usenix.org/system/files/osdi21-hunter.pdf`
  Why: TMTS uses allocation hints to separate hot and cold objects before
  page-tiering policy runs; Temeraire/TCMalloc is the allocator-side source for
  huge-page-aware packing and subrelease decisions that may inform GPU DB
  object-family arenas.
- `queued` — **Pond: CXL-Based Memory Pooling Systems for Cloud Platforms**,
  Li et al., ASPLOS 2023.
  URL: `https://doi.org/10.1145/3575693.3578835`
  Why: MEMTIS uses CXL latency assumptions from Pond; useful for future CXL
  memory-pool tiers, remote-memory latency budgets, and explicit placement
  boundaries between local DRAM, pooled memory, and GPU-resident state.
- `queued` — **vTMM: Tiered Memory Management for Virtual Machines**,
  Sha, Li, Luo, Wang, and Wang, EuroSys 2023.
  URL: `https://doi.org/10.1145/3552326.3587449`
  Why: Memstrata contrasts vTMM as a dynamic software tiering manager for VMs;
  useful for comparing page-modification-log access tracking and VM-aware
  migration with hardware-managed CXL tiering and DB-owned placement guards.
- `queued` — **Jovis: A Visualization Tool for PostgreSQL Query Optimizer**,
  Choi et al., arXiv 2024.
  URL: `https://arxiv.org/abs/2411.14788`
  Why: Hint-QPT cites Jovis as a PostgreSQL optimizer visualization system;
  useful for exposing route-choice internals, join-order sensitivity, and
  planner fallback reasons to operators without hiding deterministic rules.
- `queued` — **Extensible Query Optimizers in Practice**, Ding, Narasayya,
  and Chaudhuri, Foundations and Trends in Databases 2024.
  URL: `https://doi.org/10.1561/1900000077`
  Why: Hint-QPT cites this modern optimizer survey; useful background for
  adding GPU route hints and robust-cost extensions while preserving a native,
  inspectable optimizer contract.
- `queued` — **How Good Are Query Optimizers, Really?**, Leis et al.,
  PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol9/p204-leis.pdf`
  Why: JOB is the workload foundation used by Hint-QPT/PARQO to expose
  selectivity-estimation failures; useful as a 2015-present baseline for
  testing CPU/GPU route-choice fragility under join and selectivity errors.
- `reviewed` — **Zero-Shot Cost Models for Out-of-the-box Learned Cost
  Prediction**, Hilprecht and Binnig, PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p2361-hilprecht.pdf`
  Why: ParamTree compares against zero-shot transfer; useful for deciding
  whether GPU DB route-cost calibration should pretrain across databases or
  stay close to formula-based, per-route parameter tuning.
- `queued` — **Cost Models for Big Data Query Processing: Learning,
  Retrofitting, and Our Findings**, Siddiqui et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3380584`
  Why: ParamTree cites retrofitting learned cost models into existing
  optimizers; relevant to calibrating GPU scan, transfer, and fallback costs
  without replacing the native planner contract.
- `reviewed` — **How Good are Learned Cost Models, Really? Insights from Query
  Optimization Tasks**, Heinrich et al., SIGMOD/PACMMOD 2025.
  URL: `https://doi.org/10.1145/3725309`
  arXiv: `https://arxiv.org/abs/2502.01229`
  Why: modern learned-cost-model evaluation directly tests whether better
  prediction improves optimizer outcomes; useful for GPU DB route-choice
  benchmarks that must optimize latency, not only Q-error.
- `queued` — **Transparent Memory Offloading in Datacenters**, Weiner et al.,
  ASPLOS 2022.
  URL: `https://doi.org/10.1145/3503222.3507761`
  Why: TPP treats TMO as an orthogonal pressure-stall-driven offloading layer;
  useful for comparing DBMS-owned tier admission against system-wide memory
  saving, throttling, and demote-then-swap behavior.
- `queued` — **Direct Access, High-Performance Memory Disaggregation with
  DirectCXL**, Gouk et al., USENIX ATC 2022.
  URL: `https://www.usenix.org/conference/atc22/presentation/gouk`
  Why: TPP discusses network and CXL memory tiers as complementary; DirectCXL
  is a primary follow-up for direct CXL memory disaggregation, remote-memory
  latency budgets, and future placement boundaries beyond local DRAM/HBM/NVMe.
- `reviewed` — **R2P2: Making RPCs First-Class Datacenter Citizens**,
  Kogias et al., USENIX ATC 2019.
  URL: `https://www.usenix.org/conference/atc19/presentation/kogias-r2p2`
  PDF: `https://www.usenix.org/system/files/atc19-kogias-r2p2_0.pdf`
  Why: ZygOS follow-up by overlapping authors that exposes RPC request/response
  pairs to endpoints and network scheduling; relevant to pgwire-style request
  admission, response routing, and bounded outstanding request counts.
- `skipped` — **Homa: A Receiver-Driven Low-Latency Transport Protocol Using
  Network Priorities**, Montazeri et al., SIGCOMM 2018.
  URL: `https://doi.org/10.1145/3230543.3230564`
  arXiv: `https://arxiv.org/abs/1803.09615`
  Why: duplicate queue entry; reviewed once under the earlier Homa entry.
- `queued` — **Frequent Background Polling on a Shared Thread, Using
  Lightweight Compiler Interrupts**, Basu, Montanari, and Eriksson, PLDI 2021.
  URL: `https://doi.org/10.1145/3453483.3454049`
  Why: Concord compares against this compiler-instrumented polling approach;
  useful for deciding whether GPU DB should use explicit cooperative yield
  probes, request-budget probes, or cheaper route-local cancellation checks
  around long scans, refresh jobs, and mutation batches.
- `reviewed` — **RPCValet: NI-Driven Tail-Aware Balancing of microsecond-scale
  RPCs**, Daglis, Sutherland, and Falsafi, ASPLOS 2019.
  URL: `https://doi.org/10.1145/3297858.3304070`
  PDF: `https://faculty.cc.gatech.edu/~adaglis/files/papers/RPCValet_asplos19.pdf`
  Why: Concord cites RPCValet as a JBSQ-style dispatcher placement point;
  useful for comparing CPU-owned IO-worker scheduling with NIC-assisted
  request steering, bounded per-worker queues, and response-path priorities.
- `queued` — **NetClone: Fast, Scalable, and Dynamic Request Cloning for
  Microsecond-Scale RPCs**, Sutherland et al., arXiv 2023.
  URL: `https://arxiv.org/abs/2307.13285`
  Why: RPCValet and related microsecond RPC schedulers focus on dispatch;
  NetClone is a modern follow-up on request cloning, useful for testing
  whether GPU DB should ever duplicate short retained reads across CPU/GPU
  lanes under tail pressure, and what cancellation/cleanup costs that creates.
- `queued` — **Dagger: Accelerating RPCs in Cloud Microservices Through
  Tightly-Coupled Reconfigurable NICs**, Kumar et al., ISCA 2021.
  URL: `https://arxiv.org/abs/2106.01482`
  Why: RPCValet points toward tighter CPU/NI co-design; Dagger is a later
  tightly-coupled NIC/RPC-stack design that may inform future pgwire offload,
  request parsing, and response steering boundaries.
- `queued` — **Rain: RDMA-assisted In-Network Scheduling for
  Microsecond-scale Workloads**, arXiv 2026.
  URL: `https://arxiv.org/abs/2606.03352`
  Why: modern in-network/RDMA scheduling follow-up; useful for comparing
  switch-assisted scheduling against GPU DB's in-process route admission and
  for testing whether slice-aware scheduling maps to route lanes.
- `reviewed` — **Releasing Locks As Early As You Can: Reducing Contention of
  Hotspots by Violating Two-Phase Locking**, Guo, Wu, Yan, and Yu,
  SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457294`
  PDF: `https://pages.cs.wisc.edu/~yxy/pubs/bamboo.pdf`
  Code: `https://github.com/ScarletGuo/Bamboo-Public`
  Why: Bamboo/Wound-Retire is the direct baseline improved by
  Rebirth-Retire; useful for comparing active lock retirement, dirty
  dependency tracking, and hotspot write admission before adopting a
  passive-retire variant.
- `reviewed` — **Deferred Runtime Pipelining for Contentious Multicore
  Software Transactions**, Mu, Angel, and Shasha, EuroSys 2019.
  URL: `https://doi.org/10.1145/3302424.3303966`
  PDF: `https://www.cis.upenn.edu/~sga001/papers/drp-eurosys19.pdf`
  Why: Bamboo contrasts DRP's deferred execution and runtime
  pipelining against active dirty-read lock retirement; useful for
  deciding whether GPU DB hot-write templates should expose full
  deferred operation graphs, retire locks opportunistically, or mix the
  two per route class.
- `queued` — **Dynamic Timestamp Allocation for Reducing Transaction
  Aborts**, Arora et al., IEEE CLOUD 2018.
  URL: `https://doi.org/10.1109/CLOUD.2018.00041`
  Why: dynamic timestamp baseline discussed by Rebirth-Retire; useful for
  deciding whether GPU DB should allocate commit/order ranges per owner or
  transaction class rather than relying on a single global timestamp path.
- `reviewed` — **QueCC: A Queue-oriented, Control-free Concurrency
  Architecture**, Qadah and Sadoghi, Middleware 2018.
  URL: `https://doi.org/10.1145/3274808.3274810`
  PDF: `https://expolab.org/papers/quecc.pdf`
  Why: BOHM-adjacent deterministic two-phase planning/execution design for
  many-core transaction processing; useful for comparing queue-oriented
  planning against owner-local placeholder-first MVCC batches.
- `reviewed` — **Serval: A Wait-free Multi-version Deterministic Concurrency
  Control Scheme**, Li, Onishi, and Kawashima, CANDAR 2024.
  URL: `https://doi.org/10.1109/CANDAR64496.2024.00028`
  Metadata: `https://keio.elsevierpure.com/ja/publications/serval-a-wait-free-multi-version-deterministic-concurrency-contro/`
  Poster: `https://apsys2024.github.io/posters/apsys24posters-paper58.pdf`
  Why: Caracal follow-up that replaces global version-array latch pressure for
  contended rows with bitmaps and dynamic local version arrays; useful for
  deciding whether GPU DB write batches should keep per-owner local version
  arrays before publishing a merged visibility front.
- `reviewed` — **Dodo: A scalable optimistic deterministic concurrency control
  protocol**, Li et al., Future Generation Computer Systems 2024.
  URL: `https://doi.org/10.1016/j.future.2024.05.004`
  Why: modern deterministic concurrency control design that removes some
  state-of-the-art scalability bottlenecks; useful as a follow-up after
  Serval/Caracal for comparing deterministic batch ordering when full
  read/write sets are not always known.
- `queued` — **Optimistic Transaction Processing in Deterministic Database**,
  Dong, Tang, Wang, and Zang, Journal of Computer Science and Technology 2020.
  URL: `https://jcst.ict.ac.cn/cn/article/id/2622`
  Why: Dodo compares against DOCC as the predecessor that commits in
  predetermined order but blocks under multicore pressure; useful for
  isolating lazy determinism versus Dodo-style staged re-execution.
- `reviewed` — **Gria: an efficient deterministic concurrency control protocol**,
  Wang et al., Frontiers of Computer Science 2024.
  URL: `https://doi.org/10.1007/s11704-023-2605-z`
  Metadata: `https://academic.hep.com.cn/fcs/CN/Y2024/V18/I4/184204`
  Why: Dodo's author line includes Gria as an Aria follow-up with auto-scaling
  batches, multi-version write-after-write avoidance, reordering, and
  rechecking; useful for GPU DB batch-size and deterministic rerun policy.
- `queued` — **Cheetah: An Efficient Deterministic Concurrency Control Scheme
  with Non-Visible Write Elimination and Re-Designed Garbage Collection**, Li,
  Onishi, and Kawashima, IEEE CLUSTER Workshops 2024.
  URL: `https://doi.org/10.1109/CLUSTERWorkshops61563.2024.00053`
  Why: Caracal follow-up on eliminating non-visible writes and improving GC
  locality; relevant to pruning deterministic write-batch placeholders and
  old versions before they pollute retained GPU snapshot refresh.
- `reviewed` — **A Wake-Up Call for Kernel-Bypass on Modern Hardware**,
  Jasny et al., DaMoN 2025.
  URL: `https://doi.org/10.1145/3736227.3736235`
  PDF:
  `https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/damon25_wake_up_call.pdf`
  Why: concise modern evidence that kernel networking and storage stacks cannot
  saturate 400G/800G NICs or PCIe Gen5 SSD arrays within realistic CPU budgets;
  useful for deciding when GPU DB should move from epoll/io_uring proofs toward
  kernel-bypass network or storage experiments.
- `reviewed` — **Rapid Data Ingestion through DB-OS Co-design**, Lim et al.,
  PACMMOD/SIGMOD 2025.
  URL: `https://doi.org/10.1145/3709718`
  Why: DB/OS prefetch and shared-memory coordination design for high-rate
  ingestion; relevant to COPY admission, cold-partition prefetch, and deciding
  how much data-movement timing information the DBMS should expose to lower
  I/O layers.
- `reviewed` — **Moving on From Group Commit: Autonomous Commit Enables High
  Throughput and Low Latency on NVMe SSDs**, Nguyen et al., SIGMOD 2025.
  URL: `https://doi.org/10.1145/3725328`
  PDF:
  `https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/latency.pdf`
  Why: DaMoN 2025 kernel-bypass paper cites it as a modern WAL/storage
  follow-up; directly relevant to COPY admission, WAL flush scheduling, and
  whether GPU DB should decouple commit progress from group-commit bottlenecks.
- `reviewed` — **The Art of Latency Hiding in Modern Database Engines**,
  Huang et al., PVLDB 2023.
  URL: `https://doi.org/10.14778/3632093.3632117`
  PDF: `https://www.vldb.org/pvldb/vol17/p577-huang.pdf`
  Why: autonomous commit builds on its flush-pipelining and latency-hiding
  context; useful for deciding which background commit, scheduling, and I/O
  overlap techniques still matter before adopting fully autonomous WAL
  publication.
- `queued` — **Snap: a Microkernel Approach to Host Networking**, Marty et
  al., SOSP 2019.
  URL: `https://doi.org/10.1145/3341301.3359657`
  Why: zicIO's DB/OS co-design references user-space OS services that preserve
  isolation while moving fast-path work out of monolithic kernels; useful for
  future pgwire/network-service split, runtime ownership, and upgradeable
  datapath boundaries.
- `reviewed` — **DACE: A Database-Agnostic Cost Estimator**, Liang et al.,
  ICDE 2024.
  URL: `https://doi.org/10.1109/ICDE60146.2024.00374`
  Why: the 2025 learned-cost-model study found DACE competitive on physical
  operator selection largely because it preserves PostgreSQL cost estimates as
  inputs; useful for a hybrid GPU route-cost model that learns residuals while
  keeping deterministic planner expertise visible.
- `reviewed` — **Stage: Query Execution Time Prediction in Amazon Redshift**,
  Wu et al., SIGMOD 2024.
  URL: `https://doi.org/10.1145/3626246.3653391`
  PDF: `https://assets.amazon.science/e6/a8/0f59e3b14ffdbe68f419b3682edb/stage-query-execution-time-prediction-in-amazon-redshift.pdf`
  Why: hierarchical production query-time prediction with cache, local model,
  global model, and uncertainty; useful for routing GPU/CPU work, admission,
  and resource control without relying on one monolithic learned estimator.
- `reviewed` — **PRICE: A Pretrained Model for Cross-Database Cardinality
  Estimation**, Zeng et al., arXiv 2024.
  URL: `https://arxiv.org/abs/2406.01027`
  Why: cross-database cardinality estimation is the counterpart to DACE's
  residual-cost path; useful for deciding whether GPU route choice should keep
  cardinality and residual-latency learning as separate planner signals.
- `queued` — **Learned Cardinality Estimation: A Design Space Exploration
  and A Comparative Evaluation**, Sun et al., PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol15/p752-sun.pdf`
  DOI: `https://doi.org/10.14778/3503585.3503586`
  Why: PRICE compares against the learned-cardinality-estimation design space;
  useful for stress-testing whether a GPU DB planner should learn
  cardinality, residual latency, or route ranking, and for choosing benchmark
  metrics beyond raw q-error.
- `reviewed` — **Your Read is Our Priority in Flash Storage**, An et al.,
  PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p1911-lee.pdf`
  Why: autonomous commit cites it for flash-storage behavior; relevant to
  balancing WAL writes, cold-partition reads, and read-latency priority when
  GPU DB shares NVMe devices between durability and over-resident execution.
- `reviewed` — **What Modern NVMe Storage Can Do, And How To Exploit It:
  High-Performance I/O for High-Performance Storage Engines**, Haas and Leis,
  PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p2090-haas.pdf`
  Why: primary storage-engine source behind the DaMoN 2025 SSD argument; useful
  for sizing NVMe queue depth, IO granularity, direct IO, and CPU budgets before
  GPU DB attempts over-resident cold-partition execution.
- `reviewed` — **Databases on Modern Networks: A Decade of Research that now
  comes into Practice**, Lerner et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p3894-lerner.pdf`
  Why: modern survey and call-to-action for database/network co-design; useful
  for organizing future pgwire, RDMA, DPDK, and application-specific transport
  benchmark tracks.
- `reviewed` — **D-RDMA: Bringing Zero-Copy RDMA to Database Systems**,
  Ryser, Lerner, Forencich, and Cudre-Mauroux, CIDR 2022.
  URL: `https://vldb.org/cidrdb/papers/2022/p77-ryser.pdf`
  Why: database-specific RDMA abstraction cited by the modern-networks paper;
  relevant to future zero-copy COPY admission, remote partition movement, and
  preserving DB-level ordering above DMA completion.
- `queued` — **DFI: The Data Flow Interface for High-Speed Networks**,
  Thostrup et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457310`
  Why: higher-level network programming abstraction cited by the
  modern-networks paper; useful for comparing typed DB dataflow contracts
  against raw RDMA verbs at owner and storage-tier boundaries.
- `queued` — **Design Guidelines for Correct, Efficient, and Scalable
  Synchronization Using One-Sided RDMA**, Ziegler et al., SIGMOD 2023.
  URL: `https://doi.org/10.1145/3589295`
  Why: correctness-focused one-sided RDMA guidance cited by the
  modern-networks paper; relevant before any GPU DB WAL, visibility-summary,
  or remote-residency metadata path uses one-sided writes.
- `queued` — **P4DB - The Case for In-Network OLTP**, Jasny et al.,
  SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517867`
  Why: programmable-network hot-region OLTP execution cited by the
  modern-networks paper; useful for defining the narrow boundary between safe
  semantic admission/triage and unsafe offloading of full MVCC semantics.
- `queued` — **Design and Evaluation of an RDMA-aware Data Shuffling Operator
  for Parallel Database Systems**, Liu, Yin, and Blanas, EuroSys 2017.
  URL: `https://doi.org/10.1145/3064176.3064217`
  Why: D-RDMA uses RDMA data shuffle as a motivating fragmented-transfer
  workload; this paper is a useful source for database-level shuffle operator
  design before deciding whether GPU DB needs remote partition exchange or
  zero-copy distributed result routing.
- `reviewed` — **BMC: Accelerating Memcached using Safe In-kernel Caching and
  Pre-stack Processing**, Ghigoff et al., NSDI 2021.
  URL: `https://www.usenix.org/conference/nsdi21/presentation/ghigoff`
  PDF: `https://www.usenix.org/system/files/nsdi21-ghigoff.pdf`
  Why: Tigger's related work points to BMC as another eBPF user-bypass design;
  useful for evaluating whether tiny validated cache/protocol operations belong
  in kernel-space fast paths, and where correctness/invalidation makes that too
  risky for SQL.
- `skipped` — **KVell: the Design and Implementation of a Fast Persistent
  Key-Value Store**, Lepers et al., SOSP 2019.
  URL: `https://doi.org/10.1145/3341301.3359628`
  Why: Haas and Leis identify KVell as one of the closest systems to full
  NVMe-array exploitation; useful as a contrasting partitioned KV design for
  queue depth, SPDK usage, and limitations around range queries and small
  database payloads. Skipped as a duplicate of the reviewed KVell queue entry.
- `reviewed` — **KVell+: Snapshot Isolation without Snapshots**, Lepers et al.,
  OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/lepers`
  PDF: `https://www.usenix.org/system/files/osdi20-lepers.pdf`
  Why: direct KVell follow-up that avoids conventional snapshot version
  retention for OLAP-style scans; relevant to long retained GPU reads, MVCC
  space amplification, and cleanup latency.
- `queued` — **SILK: Preventing Latency Spikes in Log-Structured Merge
  Key-Value Stores**, Balmau et al., USENIX ATC 2019.
  URL: `https://www.usenix.org/conference/atc19/presentation/balmau`
  Why: KVell+ cites SILK as a fast-storage KV related work item; relevant to
  bounding cleanup, compaction, and cold-tier latency spikes while retained GPU
  reads and write-heavy traffic share storage devices.
- `queued` — **BatchDB: Efficient Isolated Execution of Hybrid OLTP+OLAP
  Workloads for Interactive Applications**, Makreshanski et al., SIGMOD 2017.
  URL: `https://doi.org/10.1145/3035918.3064039`
  Why: KVell+ contrasts batch/replica-style HTAP isolation with online
  commutative scans; useful for deciding when GPU DB should isolate analytics
  through retained snapshots, replicas, or bounded execution batches.
- `queued` — **X-Engine: An Optimized Storage Engine for Large-scale
  E-commerce Transaction Processing**, Huang et al., SIGMOD 2019.
  URL: `https://doi.org/10.1145/3299869.3314041`
  Why: KVell+ cites X-Engine's production emphasis on storage-space pressure
  and garbage collection; relevant to WAL/checkpoint/LSM-tier pressure and
  write-heavy transactional storage under retained read snapshots.
- `queued` — **Reaping the Performance of Fast NVM Storage with uDepot**,
  Kourtis et al., FAST 2019.
  URL: `https://www.usenix.org/conference/fast19/presentation/kourtis`
  Why: KVell compares against NVM-oriented persistent KV designs; useful for
  contrasting explicit user-space storage, page-cache avoidance, and small
  synchronization points in cold-tier point lookups.
- `queued` — **WiscKey: Separating Keys from Values in SSD-conscious
  Storage**, Lu et al., FAST 2016.
  URL: `https://www.usenix.org/conference/fast16/technical-sessions/presentation/lu`
  Why: KVell contrasts its unordered final-location writes against key/value
  separation plus LSM compaction; useful for deciding whether GPU DB cold
  storage should separate key/index metadata from row or segment payloads.
- `queued` — **Exploiting Coroutines to Attack the "Killer Nanoseconds"**,
  Jonathan et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p1702-jonathan.pdf`
  Why: MosaicDB's coroutine-to-transaction foundation; useful for deciding
  whether retained lookup and MVCC/index traversals should use cooperative
  software-prefetch lanes before adding heavier storage or GPU scheduling.
- `queued` — **Interleaving with Coroutines: A Systematic and Practical
  Approach to Hide Memory Latency in Index Joins**, Psaropoulos et al.,
  VLDB Journal 2019.
  URL: `https://doi.org/10.1007/s00778-018-0533-6`
  Why: MosaicDB cites it as a practical coroutine/prefetching baseline;
  relevant to CPU-side index, version-chain, and host-resident join paths
  that feed retained GPU or fallback routes.
- `queued` — **To FUSE or Not to FUSE: Performance of User-Space File
  Systems**, Vangoor, Tarasov, and Zadok, FAST 2017.
  URL: `https://www.usenix.org/conference/fast17/technical-sessions/presentation/vangoor`
  PDF: `https://www.usenix.org/system/files/conference/fast17/fast17-vangoor.pdf`
  Why: the ICDE 2024 BLOB paper uses FUSE to expose DBMS-owned objects as
  files; this primary FUSE performance study is useful for quantifying the
  interoperability tax before any GPU DB DB-backed-file or cold-tier API path.
- `reviewed` — **Don't Hold My Data Hostage: A Case For Client Protocol
  Redesign**, Raasveldt and Muehleisen, PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p1022-muehleisen.pdf`
  Why: the ICDE 2024 BLOB paper identifies client/server networking and
  serialization overheads as major BLOB bottlenecks; this paper is relevant to
  large result/BLOB protocol design and pgwire-compatible escape hatches.
- `queued` — **Vectorized UDFs in Column-Stores**, Raasveldt and
  Muehleisen, SSDBM 2016.
  URL: `https://doi.org/10.1145/2949689.2949705`
  Why: Don't Hold My Data Hostage contrasts protocol redesign with pushing
  computation into the database; this source is a useful companion for deciding
  when GPU DB should export columnar result chunks versus run user logic near
  resident column data.
- `queued` — **File Systems Fated for Senescence? Nonsense, Says Science!**,
  Conway et al., FAST 2017.
  URL: `https://www.usenix.org/conference/fast17/technical-sessions/presentation/conway`
  PDF: `https://www.usenix.org/system/files/conference/fast17/fast17-conway.pdf`
  Why: the ICDE 2024 BLOB paper discusses file-system aging and fragmentation;
  this gives a primary storage-systems baseline for comparing DBMS extent
  recycling against file-system aging under mixed object allocation/deletion.
- `queued` — **MittOS: Supporting Millisecond Tail Tolerance with Fast Rejecting
  SLO-Aware OS Interface**, Hao et al., SOSP 2017.
  URL: `https://doi.org/10.1145/3132747.3132756`
  Why: R2P2 cites it for fast rejection under SLO pressure; relevant to GPU DB
  overload behavior when a request cannot meet a latency budget after queue,
  snapshot, transfer, or GPU-worker admission checks.
- `queued` — **pHost: Distributed Near-Optimal Datacenter Transport over
  Commodity Network Fabric**, Gao et al., CoNEXT 2015.
  URL: `https://doi.org/10.1145/2716281.2836096`
  Why: R2P2 contrasts against low-latency datacenter transports; useful for
  comparing endpoint/network scheduling, packet spraying, and whether transport
  policy or DB-level request admission should own short RPC tail behavior.
- `queued` — **2R: Efficiently Isolating Cold Pages in Flash Storages**,
  Kang et al., PVLDB 2020.
  URL: `https://doi.org/10.14778/3407790.3407805`
  Why: read-priority flash storage follow-up cited by the RW/R-Buf paper;
  useful for separating cold-page write amplification from hot read latency
  when GPU DB shares NVMe between WAL/checkpoint traffic and cold partitions.
- `queued` — **SaS: SSD as SQL Database System**, Park et al., PVLDB 2021.
  URL: `https://doi.org/10.14778/3461535.3461539`
  Why: RW/R-Buf notes that in-storage SQL designs still face read-stall issues;
  useful as a boundary case for deciding which query or storage actions belong
  in an SSD/device tier versus the DB-owned GPU/runtime tiers.
- `queued` — **Design Guidelines for High Performance RDMA Systems**,
  Kalia, Kaminsky, and Andersen, USENIX ATC 2016.
  URL: `https://www.usenix.org/conference/atc16/technical-sessions/presentation/kalia`
  Why: ScaleRPC attributes RDMA scalability problems to NIC/CPU/memory
  resource effects; this primary guideline paper is useful before any GPU DB
  transport or remote-owner path uses one-sided verbs, registered memory, or
  polling loops.
- `queued` — **Deconstructing RDMA-enabled Distributed Transactions: Hybrid is
  Better!**, Wei et al., OSDI 2018.
  URL: `https://www.usenix.org/conference/osdi18/presentation/wei`
  Why: ScaleRPC's ScaleTX and the modern-networks thread both argue for
  phase-specific combinations of RPC and one-sided verbs; DrTM-H is a primary
  transaction-processing baseline for deciding which validation, commit, and
  replication steps can safely bypass server CPU work.
- `queued` — **RDMA-Enabled Concurrency Control Protocols for Transactions in
  the Cloud Era**, Wang and Qian, IEEE Transactions on Cloud Computing 2021.
  URL: `https://doi.org/10.1109/TCC.2021.3110946`
  arXiv: `https://arxiv.org/abs/2002.12664`
  Why: RCBench contrasts itself with RCC as an earlier unified RDMA
  concurrency-control framework; useful for comparing phase-wise hybrid
  RPC/one-sided designs against RCBench's one-sided-only primitive contract.
- `queued` — **The End of a Myth: Distributed Transactions Can Scale**,
  Zamanian et al., PVLDB 2017.
  URL: `https://arxiv.org/abs/1607.00655`
  Why: Query Fresh and modern-network discussions point to RDMA-enabled
  distributed transaction designs; NAM-DB is useful for comparing remote memory
  access, snapshot isolation, and partition ownership against GPU DB's local
  owner-domain plus future remote-tier ambitions.
- `queued` — **LITE Kernel RDMA Support for Datacenter Applications**,
  Tsai and Zhang, SOSP 2017.
  URL: `https://doi.org/10.1145/3132747.3132762`
  Why: ScaleRPC contrasts software resource sharing with kernel-level RDMA
  abstractions; LITE is a useful boundary paper for comparing application-owned
  connection/message pools against OS-mediated safety and registration control.
- `reviewed` — **Practically and Theoretically Efficient Garbage Collection for
  Multiversioning**, Wei et al., arXiv 2022/2023.
  URL: `https://arxiv.org/abs/2212.13557`
  Why: modern MVGC follow-up with experimental and theoretical collector
  variants; useful for comparing HybridGC-style production heuristics against
  bounded collector costs for retained snapshots.
- `reviewed` — **Space and Time Bounded Multiversion Garbage Collection**,
  Ben-David et al., arXiv 2021.
  URL: `https://arxiv.org/abs/2108.02775`
  Why: range-tracking approach for old-version reclamation; relevant to
  bounding retained snapshot metadata and version-chain cleanup under long
  GPU reads.
- `reviewed` — **Tiered-Indexing: Optimizing Access Methods for Skew**,
  Zhou, Hao, Yu, and Stonebraker, VLDB Journal 2025.
  URL: `https://doi.org/10.1007/s00778-025-00928-6`
  Why: modern follow-up that generalizes hot-record migration across
  buffer-managed access methods; useful for comparing Bf-Tree mini-pages with
  explicit tiered hot/cold structures under skewed GPU DB lookup workloads.
- `reviewed` — **Efficiently Making (Almost) Any Concurrency Control Mechanism
  Serializable**, Wang, Johnson, Fekete, and Pandis, VLDB Journal 2017.
  URL: `https://doi.org/10.1007/s00778-017-0463-8`
  arXiv: `https://arxiv.org/abs/1605.04292`
  Why: ERMIA uses Serial Safety Net as its serializability certifier; useful
  for deciding whether GPU DB can layer bounded dependency validation over
  snapshot-friendly read execution without falling back to pessimistic locks.
- `reviewed` — **One-shot Garbage Collection for In-memory OLTP through
  Temporality-aware Version Storage**, Raza et al., SIGMOD 2023.
  URL: `https://doi.org/10.1145/3588699`
  PDF: `https://infoscience.epfl.ch/record/305174/files/3588699.pdf`
  Why: SSN and recent MVCC scan/storage reviews point to version-chain and
  reader-retention costs as a write-path bottleneck; useful for comparing
  temporal clustering and one-shot reclamation with GPU DB retained-snapshot
  retirement and old-delta compaction. Journal entry added 2026-06-06.
- `reviewed` — **TB-Collect: Efficient Garbage Collection for Non-Volatile
  Memory Online Transaction Processing Engines**, Wei et al., 2025.
  URL: `https://www.mdpi.com/2079-9292/14/10/2080`
  DOI: `https://doi.org/10.3390/electronics14102080`
  Why: discovered while reviewing OneShotGC; applies block-level MVCC garbage
  collection ideas to NVM OLTP engines, useful for comparing future NVM/CXL
  version storage against GPU DB's DRAM/NVMe retained-delta cleanup. Journal
  entry added 2026-06-06.
- `reviewed` — **A Version-aware Data Layout for Heterogeneous Workloads in
  In-Memory Database Systems**, Zhang et al., Research Square preprint 2024.
  URL:
  `https://assets-eu.researchsquare.com/files/rs-4105094/v1_covered_cd54b494-1a7d-4e7d-86f8-cde2a6c17356.pdf`
  Why: discovered through vWeaver/OneShotGC related work; useful for comparing
  version-centric layouts, index-only version searches, and epoch/range
  partitioning against GPU DB visible-row maps and retained snapshot arrays.
  Journal entry added 2026-06-07 from the Research Square PDF.
- `queued` — **MV-PBT: Multi-Version Index for Large Datasets and HTAP
  Workloads**, Riegger et al., arXiv 2019.
  URL: `https://arxiv.org/abs/1910.08023`
  Why: cited by the version-aware layout paper as a multi-version index line;
  useful for comparing partitioned B-tree version search structures with
  GPU DB's visible-row maps, retained snapshot directories, and long-snapshot
  range reads.
- `queued` — **Polynesia: Enabling Effective Hybrid Transactional/Analytical
  Databases with Specialized Hardware/Software Co-Design**, Boroumand et al.,
  arXiv 2021.
  URL: `https://arxiv.org/abs/2103.00798`
  Why: cited by the version-aware layout paper around HTAP storage layouts;
  useful for comparing hardware/software co-design ideas against GPU DB's
  CPU/GPU resident layout, MVCC visibility metadata, and hybrid read/write
  route selection.
- `reviewed` — **Falcon: Fast OLTP Engine for Persistent Cache and
  Non-Volatile Memory**, Ji et al., SOSP 2023.
  URL:
  `https://madsys.cs.tsinghua.edu.cn/publication/falcon-fast-oltp-engine-for-persistent-cache-and-non-volatile-memory/`
  PDF:
  `https://madsys.cs.tsinghua.edu.cn/publication/falcon-fast-oltp-engine-for-persistent-cache-and-non-volatile-memory/SOSP23-ji.pdf`
  DOI: `https://doi.org/10.1145/3600006.3613141`
  Why: TB-Collect cites Falcon as a modern NVM OLTP engine that processes
  millions of transactions per second while preserving crash consistency;
  useful for comparing persistent-cache/eADR assumptions, small log windows,
  and selective data flushes with future GPU DB CXL/NVM metadata paths.
  Journal entry added 2026-06-07.
- `queued` — **Zen: a High-Throughput Log-Free OLTP Engine for
  Non-Volatile Main Memory**, Liu, Chen, and Chen, PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p835-liu.pdf`
  DOI: `https://doi.org/10.14778/3446095.3446105`
  Why: TB-Collect uses Zen as a background-scanning NVM GC and log-free OLTP
  baseline; useful for comparing metadata-enhanced tuple caches, log-free
  persistent transactions, and NVM space management with WAL-before-visibility
  and GPU DB warm-tier durability constraints.
- `queued` — **Silo: Speculative Hardware Logging for Atomic Durability in
  Persistent Memory**, Zhang and Hua, IEEE Transactions on Computers 2024.
  URL: `https://doi.org/10.1109/TC.2023.3332118`
  Why: Falcon cites Silo as a hardware logging design that keeps transactional
  logs on chip and writes them back only on crash; useful as a contrast to
  Falcon's software small-log-window approach before GPU DB assumes future
  hardware support for durable route-publication windows.
- `queued` — **BBB: Simplifying Persistent Programming using Battery-Backed
  Buffers**, Alshboul, Ramrakhyani, Wang, and Solihin, HPCA 2021.
  URL: `https://doi.org/10.1109/HPCA51647.2021.00078`
  Why: Falcon names BBB as a persistent-cache alternative to eADR; useful for
  checking whether battery-backed or protected host buffers can provide a
  durable domain for GPU DB commit windows without forcing every WAL or route
  descriptor byte to underlying NVM media during normal execution.
- `queued` — **RABIT: Efficient Range Queries with Bitmap Indexing**,
  Wang, Xiao, and Athanassoulis, PACMMOD 2025.
  URL: `https://cs-people.bu.edu/mathan/publications/pacmmod25-wang.pdf`
  Why: vWeaver-related range-query work with update-friendly bitmap indexing;
  useful for comparing native index-only scans and lightweight multi-version
  index layers against GPU DB resident key vectors and visible-row bitmaps.
- `queued` — **Don't Shoot Down TLB Shootdowns!**, Amit, Tai, and Wei,
  EuroSys 2020.
  URL: `https://doi.org/10.1145/3342195.3387525`
  Why: FastMap shows batched TLB invalidation can trade extra TLB misses for
  better mmap scalability; this follow-up studies scalable TLB shootdown
  mechanisms directly and is useful for any GPU DB tier that depends on
  mapped cold pages, virtual-memory-assisted buffers, or remapped snapshots.
- `queued` — **Scalable Range Locks for Scalable Address Spaces and Beyond**,
  Kogan, Dice, and Issa, EuroSys 2020.
  URL: `https://doi.org/10.1145/3342195.3387513`
  Why: FastMap's remaining bottleneck includes shared address-space metadata;
  scalable range-locking is a modern comparison point for page-fault,
  mmap/munmap, and non-overlapping virtual-address operations in
  VM-assisted buffer managers.
- `reviewed` — **From FASTER to F2: Evolving Concurrent Key-Value Store
  Designs for Large Skewed Workloads**, Kanellis, Chandramouli, Hart, and
  Venkataraman, PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4910-kanellis.pdf`
  arXiv: `https://arxiv.org/abs/2305.01516`
  DOI: `https://doi.org/10.14778/3750601.3750615`
  Why: Tiered-Indexing cites F2 as skewed log-structured storage work; useful
  for comparing record hotness, write buffering, and skew adaptation against
  GPU DB hot/cold resident lookup placement.
- `reviewed` — **Spooky: Granulating LSM-Tree Compactions Correctly**,
  Dayan et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p3071-dayan.pdf`
  DOI: `https://doi.org/10.14778/3551793.3551853`
  Why: F2 contrasts against LSM compaction policies; Spooky is a modern
  compaction-granularity baseline for deciding whether GPU DB cold-tier
  segment refresh should move whole runs, partitions, pages, or smaller
  key-range fragments.
- `queued` — **DLC: A New Compaction Scheme for LSM-tree with High Stability
  and Low Latency**, Jin et al., EDBT 2021.
  URL: `https://edbt2021proceedings.github.io/docs/p137.pdf`
  DOI: `https://doi.org/10.5441/002/edbt.2021.65`
  Why: Spooky cites DLC as related LSM stability work; useful for comparing
  compaction scheduling that reduces latency spikes against Spooky's
  granularity-oriented write/space amplification control for cold-tier segment
  refresh.
- `queued` — **RocksDB: Evolution of Development Priorities in a Key-value
  Store Serving Large-scale Applications**, Dong et al., ACM TOS 2021.
  URL: `https://doi.org/10.1145/3483840`
  Why: F2 uses RocksDB and MixGraph-style production skew as a workload
  baseline; this paper can ground GPU DB skew, memory-budget, and write
  amplification benchmarks in production KV workload evolution.
- `queued` — **EvenDB: Optimizing Key-Value Storage for Spatial Locality**,
  Gilad et al., EuroSys 2020.
  URL: `https://doi.org/10.1145/3342195.3387523`
  Why: Tiered-Indexing cites EvenDB as record-placement work; useful for
  studying whether physical clustering by access locality can reduce cold-tier
  reads and resident refresh churn under skew.
- `reviewed` — **Jiffy: A Lock-Free Skip List with Batch Updates and
  Snapshots**, Kobus, Kokocinski, and Wojciechowski, PPoPP 2022.
  URL: `https://doi.org/10.1145/3503221.3508437`
  arXiv: `https://arxiv.org/abs/2102.01044`
  Why: cited by the MVGC paper as a modern multiversion/snapshot data structure;
  useful for comparing batched update publication and wait-free range snapshot
  support against GPU DB retained-read generations. Journal entry added
  2026-06-07 from arXiv and the author PDF.
- `queued` — **MV-RLU: Scaling Read-Log-Update with Multi-Versioning**,
  Kim et al., ASPLOS 2019.
  URL: `https://doi.org/10.1145/3297858.3304040`
  Why: cited by the MVGC paper as a practical multiversioning system; useful
  for contrasting reader-side logging, version lifetime, and reclamation costs
  with GPU DB MVCC chains and long retained snapshots.
- `queued` — **Bundling Linked Data Structures for Linearizable Range
  Queries**, Nelson-Slivon, Hassan, and Palmieri, PPoPP 2022.
  URL: `https://doi.org/10.1145/3503221.3508412`
  arXiv: `https://arxiv.org/abs/2201.00874`
  Why: discovered while reviewing Jiffy; uses bundled references and
  TSC-shaped range-query snapshots over linked data structures, useful for
  comparing per-link version bundles against GPU DB route-fragment revisions
  and snapshot-safe range traversal.
- `reviewed` — **Constant-Time Snapshots with Applications to Concurrent Data
  Structures**, Wei et al., PPoPP 2021.
  URL: `https://arxiv.org/abs/2007.02372`
  Why: the bounded MVGC paper applies its collector to this versioned-CAS
  snapshot framework; useful for deciding whether GPU DB should expose
  retained snapshot handles over lock-free CPU data structures before or
  alongside SQL-facing MVCC chains.
- `queued` — **KiWi: A Key-Value Map for Scalable Real-Time Analytics**,
  Basin et al., PPoPP 2017.
  URL: `https://doi.org/10.1145/3018743.3018761`
  PDF: `https://people.csail.mit.edu/idish/ftp/kiwi.pdf`
  Why: compared by constant-time snapshots as a state-of-the-art range-query
  key-value map; useful for evaluating per-key publication, range-query
  atomicity, and update/query tradeoffs for retained metadata indexes.
- `queued` — **Lock-free Contention Adapting Search Trees**, Winblad,
  Sagonas, and Jonsson, SPAA 2018.
  URL: `https://doi.org/10.1145/3210377.3210413`
  PDF: `https://user.it.uu.se/~bengt/Papers/Full/spaa18.pdf`
  Why: constant-time snapshots compares against LFCA's adaptive synchronization
  granularity; useful for route metadata and resident index structures whose
  best synchronization granularity changes with range-query size and
  contention.
- `queued` — **Triton Join: Efficiently Scaling to a Large Join State on GPUs
  with Fast Interconnects**, Lutz et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517911`
  Why: Vortex compares against Triton Join's fast-interconnect join state
  strategy; useful for deciding when over-resident GPU DB joins should depend
  on high-bandwidth CPU/GPU links versus explicit multi-GPU IO forwarding.
- `queued` — **MG-Join: A Scalable Join for Massively Parallel Multi-GPU
  Architectures**, Paul et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457254`
  Why: Vortex contrasts multi-GPU memory-capacity scaling with IO forwarding;
  MG-Join is a direct follow-up for future multi-device join partitioning and
  interconnect-aware placement.
- `queued` — **Managing Non-Volatile Memory in Database Systems**, van Renen
  et al., SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3196897`
  Why: NVM-aware multi-tier buffer-manager baseline compared by the 2019
  adaptive migration paper; useful for page admission queues and recent-access
  filters before designing GPU DB host/NVMe tier promotion.
- `queued` — **Strata: A Cross Media File System**, Kwon et al., SOSP 2017.
  URL: `https://doi.org/10.1145/3132747.3132770`
  Why: cross-media NVM/SSD file-system design cited by the 2019 multi-tier
  buffer paper; useful as a DB-external contrast for per-application logs,
  performance isolation, and when DB-owned tiering should bypass the file
  system.
- `reviewed` — **Design Principles for Scaling Multi-core OLTP Under High
  Contention**, Ren et al., SIGMOD 2016 / arXiv 2015.
  URL: `https://arxiv.org/abs/1512.06168`
  Why: ORTHRUS-style separation of transaction execution stages and advanced
  transaction planning is a direct follow-up for FOEDUS's many-core OCC
  scaling limits under high contention, and may inform GPU DB mutation-owner
  admission and partitioned write lanes. Journal entry added 2026-06-06 from
  the SIGMOD 2016 author PDF.
- `reviewed` — **SplinterDB: Closing the Bandwidth Gap for NVMe Key-Value
  Stores**, Conway et al., USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/conway`
  PDF: `https://www.usenix.org/system/files/atc20-conway.pdf`
  Why: PrismDB cites SplinterDB as an NVMe-specialized KV-store comparison;
  its STB-epsilon-tree, concurrent cache, and reduced write amplification are
  relevant to CPU/NVMe tier limits before GPU resident refresh. Duplicate queue
  entry corrected to reviewed on 2026-06-06.
- `queued` — **SpanDB: A Fast, Cost-Effective LSM-tree Based KV Store on
  Hybrid Storage**, Chen et al., FAST 2021.
  URL: `https://www.usenix.org/conference/fast21/presentation/chen-hao`
  Why: PrismDB compares against SpanDB as a hybrid-storage LSM baseline; useful
  for deciding which WAL, SPDK, and upper-level placement ideas survive against
  GPU DB's explicit warm/cold tier model.
- `queued` — **Mutant: Balancing Storage Cost and Latency in LSM-Tree Data
  Stores**, Yoon et al., SoCC 2018.
  URL: `https://www.cs.cmu.edu/~juncheny/publications/socc18-Mutant.pdf`
  Why: PrismDB contrasts object-level MSC with SSTable-level access-frequency
  tiering; useful as a simpler placement baseline for cold resident segments.
- `queued` — **The eXpress Data Path: Fast Programmable Packet Processing in
  the Operating System Kernel**, Hohlfeld et al., CoNEXT 2018.
  URL: `https://doi.org/10.1145/3281411.3281443`
  Why: BMC relies on XDP as its earliest packet-processing hook; this primary
  XDP paper is useful before considering any pgwire classification, overload
  rejection, or exact-response cache near the kernel/network boundary.
- `queued` — **NetCache: Balancing Key-Value Stores with Fast In-Network
  Caching**, Jin et al., SOSP 2017.
  URL: `https://doi.org/10.1145/3132747.3132764`
  Why: BMC compares against switch-based key-value caching; NetCache is useful
  for drawing the line between safe cached read responses and unsafe offload of
  SQL visibility, invalidation, and write semantics.
- `queued` — **Adopting Worst-Case Optimal Joins in Relational Database
  Systems**, Freitag et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p1891-freitag.pdf`
  DOI: `https://doi.org/10.14778/3407790.3407797`
  Why: Free Join builds on this lazy trie and binary/WCOJ hybrid baseline;
  useful for comparing an optimizer switch between binary and WCOJ plans with
  GPU DB's possible spectrum of binary, multiway, factorized, and GPU-resident
  join routes.
- `reviewed` — **Modeling Concurrency Control as a Learnable Function
  (CCaaLF/NeurCC)**, Pan et al., arXiv 2025, revised 2026.
  URL: `https://arxiv.org/abs/2503.10036`
  Why: modern learned-concurrency-control follow-up to Polyjuice-style policy
  search; useful for deciding whether GPU DB should limit itself to offline
  policy tables or consider broader learned functions for owner admission,
  wait placement, and retry/backoff decisions.
- `reviewed` — **Native Store Extension for SAP HANA**, Sherkat et al.,
  PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p2047-sherkat.pdf`
  Why: BTrim-related SAP tiering work for keeping warm/cold data outside the
  hot in-memory store; useful for comparing row-level IMRS packing with
  columnar warm-store placement and explicit cold-tier access.
- `queued` — **Rethink the Scan in MVCC Databases**, Kim et al.,
  SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3452783`
  Why: modern MVCC scan/access-method work from the vDriver/DIVA ecosystem;
  useful for retained snapshot scans where version traversal can erase index
  benefits and GPU routes need a compact visible-version access structure.
- `reviewed` — **Adaptive Optimistic Concurrency Control for Heterogeneous
  Workloads**, Guo et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p584-guo.pdf`
  DOI: `https://doi.org/10.14778/3303753.3303763`
  Why: NeurCC uses adaptive hot/warm/cold conflict detection as a baseline;
  useful for comparing simple contention-aware OCC policy switching against
  learned action tables before adding GPU DB owner-queue policy learning.
- `queued` — **Toward Coordination-free and Reconfigurable Mixed Concurrency
  Control**, Tang et al., USENIX ATC 2018.
  URL: `https://www.usenix.org/conference/atc18/presentation/tang`
  PDF: `https://www.usenix.org/system/files/conference/atc18/atc18-tang.pdf`
  Why: CormCC is one of NeurCC's adaptive CC baselines; useful for comparing
  partition-level mixed protocols and online reconfiguration with per-route
  learned conflict actions.
- `reviewed` — **GeminiFS: A Companion File System for GPUs**, Qiu et al.,
  FAST 2025.
  URL: `https://www.usenix.org/conference/fast25/presentation/qiu`
  PDF: `https://www.usenix.org/system/files/fast25-qiu.pdf`
  Why: modern GPU-facing storage interface that cites GMT; useful for comparing
  file-system-level GPU IO services with DB-owned GPU/host/NVMe tier managers.
  Journal entry added 2026-06-06.
- `queued` — **Cheetah: Metadata Aggregation for Fast Object Storage without
  Distributed Ordering**, Zhang et al., EuroSys 2025.
  URL: `https://doi.org/10.1145/3689031.3696080`
  PDF: `https://home.cse.ust.hk/~kaichen/papers/cheetah-eurosys25.pdf`
  Why: GeminiFS embeds per-file block maps to avoid GPU-side metadata
  traversal; Cheetah is a modern follow-up on aggregating storage metadata to
  remove distributed write ordering, relevant to GPU DB cold-tier object
  metadata, checkpoint manifests, and DB-owned file/object layout.
- `queued` — **Characterizing Emerging Page Replacement Policies**, Wu et al.,
  IISWC 2024.
  URL: `https://www.cs.yale.edu/homes/abhishek/mwu-iiswc24.pdf`
  Why: modern page replacement study citing GMT; useful for testing whether
  reuse-prediction and learned/scan-resistant cache policies survive GPU DB
  mixed lookup, scan, refresh, and over-resident workloads.
- `queued` — **Elastic RSS: Co-Scheduling Packets and Cores Using Programmable
  NICs**, Rucker, Shahbaz, Swamy, and Olukotun, APNet 2019.
  URL: `https://doi.org/10.1145/3343180.3343184`
  Why: Mind the Gap contrasts fixed RSS with NIC-side policies that incorporate
  fine-grained load feedback; useful for GPU DB route-class steering when IO
  workers, owner queues, and GPU execution lanes need elastic core assignment
  without a global dispatcher bottleneck.
- `queued` — **Just In Time Delivery: Leveraging Operating Systems Knowledge
  for Better Datacenter Congestion Control**, Ousterhout, Belay, and Zhang,
  HotCloud 2019.
  URL: `https://www.usenix.org/conference/hotcloud19/presentation/ousterhout`
  Why: Mind the Gap cites OS/network co-design where packets should arrive
  just in time for processing; relevant to GPU DB response-ring and ingress
  pacing when queue saturation should slow admission before p99 latency spikes.
- `queued` — **Supporting data-driven I/O on GPUs using GPUfs**, Shahar and
  Silberstein, SYSTOR 2016.
  URL: `https://doi.org/10.1145/2928275.2928282`
  Why: ActivePointers integrates with a revised GPUfs page cache; this
  companion source should expose the lower-level GPU file-system batching and
  page-cache mechanisms behind data-driven GPU I/O.
- `queued` — **GPUpIO: The Case for I/O-Driven Preemption on GPUs**, Zeno,
  Mendelson, and Silberstein, GPGPU 2016.
  URL: `https://doi.org/10.1145/2884045.2884051`
  Why: ActivePointers identifies long major page faults as a GPU preemption
  problem; this follow-up is relevant to preventing storage faults or
  over-resident misses from wasting SM resources and hurting short GPU queries.
- `queued` — **Page Placement Strategies for GPUs within Heterogeneous Memory
  Systems**, Agarwal et al., ASPLOS 2015.
  URL: `https://doi.org/10.1145/2694344.2694381`
  Why: ActivePointers cites GPU page-placement work as nearby hardware memory
  management; useful for comparing explicit DB-owned placement with
  hardware/runtime page placement across GPU, host, and future memory tiers.
- `reviewed` — **Bf-Tree: A Modern Read-Write-Optimized Concurrent
  Larger-Than-Memory Range Index**, Hao and Chandramouli, PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p3442-hao.pdf`
  DOI: `https://doi.org/10.14778/3681954.3682012`
  Why: discovered while chasing tiered-memory buffer-management follow-ups;
  variable-length mini-pages decouple cache granularity from disk pages and
  may inform GPU DB hot-record, warm-page, and cold-NVMe placement.
- `reviewed` — **LiquidCache: Efficient Pushdown Caching for Cloud-Native Data
  Analytics**, Hao et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p5662-hao.pdf`
  Why: modern cache-placement follow-up from the same tiering research area;
  useful for comparing DB-owned cache admission and pushdown placement with
  GPU DB resident, host, and cold-tier policies. Reviewed on 2026-06-06.
- `queued` — **Making congestion control robust to per-packet load balancing in
  datacenters**, arXiv 2025.
  URL: `https://arxiv.org/abs/2509.07907`
  Why: modern Swift follow-up that studies robustness under per-packet load
  balancing; useful if GPU DB later maps delay-based admission across
  multi-path gateways, distributed IO workers, or multiple response lanes.
- `reviewed` — **SMaRTT: Sender-based Marked Rapidly-adapting Trimmed &
  Timed Transport**, Bonato et al., arXiv 2024.
  URL: `https://arxiv.org/abs/2404.01630`
  Why: recent congestion-control comparison against Swift, PowerTCP, and other
  datacenter schemes; useful for deciding whether GPU DB admission should
  combine delay, queue-depth, and optional explicit notification signals. The
  queued FASTFLOW title now resolves to the SMaRTT arXiv paper.
- `reviewed` — **Ultra Ethernet's Design Principles and Architectural
  Innovations**, Hoefler et al., arXiv 2025.
  URL: `https://arxiv.org/abs/2508.08906`
  Why: SMaRTT positions itself as the basis for UEC NSCC; the broader UEC
  design may inform future GPU DB transport assumptions, multipath routing,
  out-of-order placement, and packet-trimming availability.
  Journal entry exists from 2026-06-05.
- `reviewed` — **Bolt: Sub-RTT Congestion Control for Ultra-Low Latency**,
  Arslan et al., NSDI 2023.
  URL: `https://www.usenix.org/conference/nsdi23/presentation/arslan`
  Why: SMaRTT cites Bolt as a recent sub-RTT congestion-control baseline;
  useful for comparing fast congestion notification with GPU DB response-ring
  and gateway admission telemetry.
- `queued` — **Credit-Scheduled Delay-Bounded Congestion Control for
  Datacenters**, Cho, Jang, and Han, SIGCOMM 2017.
  URL: `https://doi.org/10.1145/3098822.3098840`
  PDF: `https://keonjang.github.io/papers/sigcomm17ep.pdf`
  Why: Bolt contrasts against ExpressPass-style credit scheduling; useful for
  comparing explicit admission credits with delay/queue feedback for GPU DB
  ingress, response rings, and bounded micro-batch launch.
- `reviewed` — **Transaction Healing: Scaling Optimistic Concurrency Control on
  Multicores**, Wu, Chan, and Tan, SIGMOD 2016.
  URL: `https://dl.acm.org/doi/10.1145/2882903.2915202`
  PDF: `https://yingjunwu.github.io/papers/sigmod2016.pdf`
  Why: AOCC cites transaction healing as a semantic repair path for OCC; useful
  for deciding when GPU DB should retry, repair, or reissue only dependent
  pieces of a transaction instead of aborting the full command envelope.
- `reviewed` — **BCC: Reducing False Aborts in Optimistic Concurrency Control
  with Low Cost for In-Memory Databases**, Yuan et al., PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p504-yuan.pdf`
  DOI: `https://doi.org/10.14778/2904121.2904126`
  Why: AOCC cites BCC as a low-overhead false-abort reduction baseline; useful
  for GPU DB contention handling where serializable write lanes should avoid
  unnecessary aborts without weakening visibility guarantees. Journal entry
  added 2026-06-05.
- `reviewed` — **O|R|P|E - A Data Semantics Driven Concurrency Control**,
  Lessner, Laux, and Connolly, International Journal On Advances in Software
  2016; arXiv 2023.
  URL: `https://arxiv.org/abs/2308.09121`
  Why: follow-up discovered while reviewing Transaction Healing; explores
  choosing optimistic, reconciliation, pessimistic, or escrow-style
  concurrency classes from data semantics, useful for comparing semantic
  conflict repair with route-level isolation classes for hot GPU DB write
  paths.
- `reviewed` — **CRDV: Conflict-free Replicated Data Views**, Faria and Pereira,
  PACMMOD 2025.
  URL: `https://doi.org/10.1145/3709675`
  Author page: `https://nuno-faria.github.io/crdv/`
  PDF:
  `https://repositorio.inesctec.pt/bitstreams/2fe68b69-5453-4943-bc3c-a42b9a78c8e3/download`
  Why: MRVs leaves open whether randomized splitting generalizes beyond
  numeric bounded counters; CRDV appears to continue the same line of
  application-visible replicated/derived data structures for reducing
  coordination, relevant to write-hot derived views and contention-safe
  denormalized state.
- `reviewed` — **The CacheLib Caching Engine: Design and Experiences at Scale**,
  Berg et al., OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/berg`
  Why: 2-Tree cites CacheLib as evidence that working sets shift over time;
  its production cache admission, eviction, slab, and workload-class mechanics
  are useful follow-up material for GPU DB host-memory and resident-cache
  policy.
- `queued` — **Kangaroo: Caching Billions of Tiny Objects on Flash**,
  McAllister et al., SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483568`
  Why: CacheLib's SOC design highlights flash-cache DRAM-index pressure for
  tiny objects; Kangaroo is a direct follow-up for billions-of-objects flash
  indexing, admission, and write-amplification control in future cold or warm
  metadata tiers.
- `queued` — **BzTree: A High-Performance Latch-Free Range Index for
  Non-Volatile Memory**, Arulraj et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p553-arulraj.pdf`
  DOI: `https://doi.org/10.14778/3173075.3173077`
  Why: 2-Tree names concurrent tree coordination as future work and cites
  latch-free range-index designs; BzTree is a relevant follow-up for
  hot-tier range indexes, NVM/future-tier persistence, and migration-safe
  updates.
- `reviewed` — **No Cap, This Memory Slaps: Breaking Through the Memory Wall
  of Transactional Database Systems with Processing-in-Memory**, Kim,
  Zhao, Pavlo, and Gibbons, PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4241-kim.pdf`
  DOI: `https://doi.org/10.14778/3749646.3749690`
  Artifact: `https://github.com/hyoungjook/OLTPim`
  Why: modern OLTP near-data-processing design that splits tuple data,
  indexes, MVCC metadata, batching, and logging across CPU DRAM and PIM;
  useful for comparing GPU DB's future GPU/HBM, host-memory, and
  near-memory owner placement decisions.
- `queued` — **AsyncDIMM: Achieving Asynchronous Execution in DIMM-Based
  Near-Memory Processing**, Chen et al., HPCA 2025.
  URL: `https://doi.org/10.1109/HPCA61900.2025.00048`
  Why: OLTPim identifies mux-switch/control latency as a core blocker and
  cites asynchronous DIMM execution as future hardware support; relevant to
  deciding how GPU DB should price future near-memory tiers and asynchronous
  offload queues.
- `queued` — **PIM-Tree: A Skew-Resistant Index for
  Processing-in-Memory**, Kang et al., PVLDB 2022.
  URL: `https://doi.org/10.14778/3574245.3574254`
  Why: OLTPim contrasts its one-round hash/range partitioned indexes with
  PIM-Tree's skew-resistant placement; useful for balancing hot-key skew
  against extra near-tier request rounds in future cache/index placement.
- `queued` — **HybriDS: Cache-conscious Concurrent Data Structures for
  Near-Memory Processing Architectures**, Choe et al., SPAA 2022.
  URL: `https://doi.org/10.1145/3490148.3538572`
  Why: OLTPim cites HybriDS for keeping frequently used upper tree levels in
  CPU cache while lower pointer-chasing work runs near memory; relevant to
  GPU DB's split CPU/GPU/future-tier index residency decisions.
- `reviewed` — **HetCache: Synergising NVMe Storage and GPU Acceleration for
  Memory-Efficient Analytics**, Nicholson, Raza, Chrysogelos, and Ailamaki,
  CIDR 2023.
  URL: `https://www.cidrdb.org/cidr2023/papers/p84-nicholson.pdf`
  Why: GOLAP identifies HetCache as nearby CPU/GPU data-placement work for
  disk-backed analytics; useful for comparing cache-placement decisions against
  GOLAP-style compressed SSD-to-GPU streaming and GPU DB's explicit residency
  manager.
- `reviewed` — **Workload Placement on Heterogeneous CPU-GPU Systems**,
  Carvalho, Simitsis, Queralt, and Romero, PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p4241-carvalho.pdf`
  DOI: `https://doi.org/10.14778/3685800.3685845`
  Why: modern tutorial and taxonomy for CPU/GPU placement strategies, cost
  prediction, placement granularity, and code-management choices; useful for
  turning HetCache-style access-path hints into a broader route-placement
  contract. Journal entry exists from 2026-06-05.
- `reviewed` — **Adaptive Compression for Databases**, Windheuser et al.,
  EDBT 2024.
  URL: `https://doi.org/10.48786/EDBT.2024.13`
  PDF: `https://openproceedings.org/2024/conf/edbt/paper-43.pdf`
  Why: GOLAP cites adaptive compression of cold column sections; useful for
  deciding whether GPU DB cold/warm segments should choose compression
  parameters from access statistics instead of a single fixed resident/cold
  format. Journal entry exists from 2026-06-05.
- `queued` — **High-Throughput BitPacking Compression**, Lisa, Nguyen,
  Habich, Kumar, and Lehner, DSD 2019.
  URL: `https://doi.org/10.1109/DSD.2019.00101`
  Why: AdaCom uses bit packing as its compact segment format; useful for
  comparing CPU/GPU-friendly integer packing throughput, alignment choices,
  and decompression overhead for resident and warm column groups.
- `queued` — **Evaluating Lightweight Integer Compression Algorithms in
  Column-Oriented In-Memory DBMS**, Heinzl et al., ADMS@VLDB 2021.
  URL:
  `https://hpi.de/fileadmin/user_upload/fachgebiete/rabl/publications/2021/ADMS_2021_Integer_Compression.pdf`
  Why: AdaCom cites this integer-compression evaluation; useful for choosing
  first P8 integer segment encodings before testing GPU resident, CPU warm,
  and NVMe cold formats.
- `queued` — **Waiting to Decompress: Lazy Loading of Compressed Data in
  Main-Memory Database Systems**, Kipf et al., CIDR 2026.
  URL: `https://vldb.org/cidrdb/papers/2026/p34-kipf.pdf`
  Why: modern compression/lazy-loading follow-up that cites AdaCom; useful for
  deciding whether GPU DB warm and cold column groups should defer
  decompression until route admission proves the bytes are needed.
- `queued` — **The Art of Balance: A RateupDB Experience of Building a
  CPU/GPU Hybrid Database Product**, Lee et al., PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p2999-lee.pdf`
  DOI: `https://doi.org/10.14778/3476311.3476372`
  Why: cited by the CPU/GPU placement taxonomy as a product-oriented hybrid
  database placement source; useful for comparing automatic route placement
  against operator-tuned balance rules and production code-management costs.
- `queued` — **Parla: A Python Orchestration System for Heterogeneous
  Architectures**, Lee et al., SC 2022.
  URL: `https://doi.org/10.1109/SC22.2022.00032`
  Why: cited by the CPU/GPU placement taxonomy as a task-level heterogeneous
  orchestration system; useful for comparing GPU DB owner rings with
  dependency-aware runtime task placement across CPU and GPU resources.
- `queued` — **ByteSlice: Pushing the Envelope of Main Memory Data
  Processing with a New Storage Layout**, Feng, Lo, Kao, and Xu, SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2747642`
  Why: Data Blocks contrasts against sub-byte and byte-sliced encodings; useful
  for deciding whether GPU DB resident or host-side cold chunks should use
  byte-addressable codes, bit-sliced predicates, or format-specific routes.
- `queued` — **Efficient Lightweight Compression Alongside Fast Scans**,
  Polychroniou and Ross, DaMoN 2015.
  URL: `https://doi.org/10.1145/2771937.2771943`
  Why: Data Blocks compares against SIMD bit-packing/unpacking work; useful
  for benchmarking compressed predicate execution against positional
  decompression and sparse-result extraction in P8 text/int4 segments.
- `queued` — **A Padded Encoding Scheme to Accelerate Scans by Leveraging
  Skew**, Li, Chasseur, and Patel, SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2737799`
  Why: Data Blocks notes padded encoding as a possible secondary-index-like
  structure; useful for testing whether skew-aware compressed scan layouts
  belong in GPU DB as route-specific acceleration rather than canonical storage.
- `reviewed` — **Pipelined Query Processing in Coprocessor Environments**,
  Funke, Bress, Noll, Markl, and Teubner, SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3183734`
  PDF:
  `https://dbis.cs.tu-dortmund.de/storages/dbis-cs/r/papers/2018/pipelined-query-processing/pipelined-query-processing.pdf`
  Why: Revisiting GPU DB performance identifies kernel fusion and avoiding
  intermediate materialization as central GPU DB efficiency levers; this
  follow-up should expose pipeline/fusion design choices for coprocessor
  query engines.
- `queued` — **Data-Parallel Query Processing on Non-Uniform Data**,
  Funke and Teubner, PVLDB 2020.
  URL: `https://doi.org/10.14778/3389133.3389139`
  Why: Revisiting GPU DB performance highlights memory stalls, skew, and cache
  behavior; this follow-up is relevant to GPU DB route choices under skewed
  retained lookup, join, and aggregation workloads.
- `queued` — **GPU-Accelerated Database Systems: Survey and Open Challenges**,
  Bress et al., VLDB Journal 2021.
  URL: `https://doi.org/10.1007/s00778-020-00624-x`
  Why: broader GPU DB survey from the HorseQC/CoGaDB ecosystem; useful for
  checking whether retained routes, transfer-aware planning, and GPU memory
  hierarchy benchmarks cover known open challenges.
- `queued` — **HippogriffDB: Balancing I/O and GPU Bandwidth in Big Data
  Analytics**, Li et al., PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p1647-li.pdf`
  Why: directly related fused data-path work cited by HorseQC; useful for
  comparing pre-fabricated fused kernels with query-generated compound kernels
  and P8 resident/cold transfer balance.
- `queued` — **GPL: A GPU-Based Pipelined Query Processing Engine**, Paul,
  He, and He, SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915214`
  Why: related GPU pipeline engine using GPU pipes/local execution contexts;
  useful for comparing explicit compound kernels with hardware/runtime
  pipeline handoff costs.
- `reviewed` — **GPU-Accelerated OLTP: An In-Depth Analysis of
  Concurrency Control Schemes**, Sun et al., arXiv 2024; v2 2026.
  URL: `https://arxiv.org/abs/2406.10158`
  PDF: `https://arxiv.org/pdf/2406.10158`
  Why: selected because the remaining ready queue skewed toward GPU analytics
  and this modern GPU OLTP testbed directly compares OCC, MVCC, 2PL,
  conflict-graph ordering, launch parameters, and conflict-resolution overhead
  for batched transaction execution.
- `reviewed` — **GaccO - A GPU-accelerated OLTP DBMS**, Boeschen and
  Binnig, SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517876`
  Metadata: `https://www.dfki.de/web/forschung/projekte-publikationen/publikation/14413`
  Why: the GPU OLTP concurrency-control study identifies GaccO as the
  strongest high-conflict/write-heavy GPU-oriented baseline; useful for
  transaction batching, CPU/GPU co-execution, and conflict staging.
- `reviewed` — **An Analysis of Concurrency Control Protocols for
  In-Memory Databases with CCBench**, Tanabe et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p3531-tanabe.pdf`
  DOI: `https://doi.org/10.14778/3424573.3424575`
  Why: the GPU OLTP concurrency-control study contrasts against CPU CC
  benchmarking gaps; CCBench is useful for separating CPU-owner concurrency
  limits from GPU-specific SIMT and launch-parameter effects.
- `reviewed` — **Engineering a High-Performance GPU B-Tree**, Awad et al.,
  PPoPP 2019.
  URL: `https://doi.org/10.1145/3293883.3295706`
  PDF: `https://par.nsf.gov/servlets/purl/10101116`
  Why: the GPU OLTP study finds index lookup can dominate low-contention
  runs; a GPU B-tree gives a concrete follow-up for retained equality/range
  indexes and batched lookup/update paths.
- `reviewed` — **A GPU Multiversion B-Tree**, Awad, Porumbescu, and Owens,
  PACT 2022.
  URL: `https://openreview.net/forum?id=RJ95nyPhcp`
  DOI: `https://doi.org/10.1145/3559009.3569681`
  Author page/slides: `https://maawad.github.io/`
  Code: `https://github.com/owensgroup/MVGpuBTree`
  Why: the reviewed GPU OLTP paper warns that naive MVCC version-chain
  traversal can erase read benefits; a GPU multiversion tree is a direct
  follow-up for compact visibility-aware index structures.
- `reviewed` — **Scalable OLTP in the Cloud: What's the BIG DEAL?**,
  Helland, CIDR 2024.
  URL: `https://www.cidrdb.org/cidr2024/papers/p63-helland.pdf`
  Why: selected because the ready queue skewed toward GPU analytics and this
  modern RCSI/MVCC scaling thought experiment maps directly to commit-time
  organization, time/key owner partitions, delayed visibility, and scalable
  queue semantics for GPU DB snapshot routing.
- `reviewed` — **Is Scalable OLTP in the Cloud a Solved Problem?**,
  Ziegler, Bernstein, Leis, and Binnig, CIDR 2023.
  URL: `https://www.cidrdb.org/cidr2023/papers/p50-ziegler.pdf`
  Why: Helland's CIDR 2024 paper responds to this cloud OLTP design analysis;
  useful for comparing single-writer shared storage, multiple-writer coherent
  caching, shared-nothing partitioning, and hot-tuple cache-coherence tradeoffs.
- `reviewed` — **Eigen: End-to-End Resource Optimization for Large-Scale
  Databases on the Cloud**, Li et al., PVLDB 2023.
  URL: `https://doi.org/10.14778/3611540.3611565`
  PDF: `https://www.vldb.org/pvldb/vol16/p3795-zhou.pdf`
  Why: cited by the Azure SQL DBaaS resource-allocation paper as a related
  cloud database resource optimizer; useful for comparing workload-level
  resource recommendation and control-loop design with GPU DB admission,
  cache budgets, and tenant/session placement.
- `queued` — **Tenant Placement in Over-subscribed Database-as-a-Service
  Clusters**, Konig et al., PVLDB 2022.
  URL: `https://doi.org/10.14778/3489496.3489498`
  Why: the flexible resource-allocation paper builds on this tenant-placement
  model; useful for separating per-node resource brokering from cluster-level
  placement, failover cost, and resource-violation prediction.
- `queued` — **Toto - Benchmarking the Efficiency of a Cloud Service**,
  Moeller, Ye, Lin, and Lang, SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457555`
  Why: cited as Azure SQL infrastructure for measuring database resource
  contention; useful for shaping no-GPU scalability probes that measure active
  session budgets, cache pressure, and resource contention per logical route.
- `reviewed` — **ScaleStore: A Fast and Cost-Efficient Storage Engine using
  DRAM, NVMe, and RDMA**, Ziegler, Binnig, and Leis, SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526187`
  PDF:
  `https://www.informatik.tu-darmstadt.de/media/systems/pdf_publications/ScaleStore_preprint.pdf`
  Why: the CIDR 2023 cloud OLTP paper uses ScaleStore as the shared-cache
  blueprint; useful for resident-object directory design, RDMA/NVMe cache
  placement, and page-level coherence tradeoffs.
- `reviewed` — **Sundial: Harmonizing Concurrency Control and Caching in a
  Distributed OLTP Database Management System**, Yu et al., PVLDB 2018.
  URL: `https://doi.org/10.14778/3231751.3231763`
  PDF: `https://www.vldb.org/pvldb/vol11/p1289-yu.pdf`
  Why: cited as distributed OLTP concurrency-control work that links caching
  and timestamp visibility; useful for GPU DB's coherent retained snapshots and
  owner-visible generation routing.
- `reviewed` — **STAR: Scaling Transactions through Asymmetric Replication**,
  Lu, Yu, and Madden, PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p1316-lu.pdf`
  arXiv: `https://arxiv.org/abs/1811.02059`
  Why: discovered while reviewing Sundial and adjacent PVLDB distributed OLTP
  work; useful for comparing lease-based cached remote reads with asymmetric
  replication, locality, and serializable transaction routing.
- `queued` — **Falcon: A Timestamp-based Protocol to Maximize the Cache
  Efficiency in the Distributed Shared Memory**, Zhang et al., IPDPS 2022.
  URL: `https://doi.org/10.1109/IPDPS53621.2022.00037`
  Why: the CIDR 2023 paper names Falcon as a modern alternative to directory
  invalidation; useful for comparing timestamped coherence with explicit
  invalidation for GPU/host resident-object directories.
- `queued` — **Doppel: A Framework for Transactional Data Structures in
  Modern Database Systems**, Neumann, Freitag, and Kemper, PVLDB 2018.
  URL: `https://doi.org/10.14778/3275366.3275371`
  Why: STAR cites Doppel as a related commutative-update direction; useful for
  deciding when GPU DB write routes can safely treat increments, appends, or
  aggregate maintenance as operation deltas instead of full value rewrites.
- `queued` — **TAPIR: Building Consistent Transactions with Inconsistent
  Replication**, Zhang et al., SOSP 2015.
  URL: `https://doi.org/10.1145/2815400.2815404`
  Why: STAR contrasts against transactional replication approaches that reduce
  coordination; useful for comparing commit/replication boundaries when GPU DB
  eventually adds replicated owners or remote accelerator pools.
- `reviewed` — **Massively Parallel Multi-Versioned Transaction Processing**,
  Qian and Goel, OSDI 2024.
  URL: `https://www.usenix.org/conference/osdi24/presentation/qian`
  PDF: `https://www.usenix.org/system/files/osdi24-qian.pdf`
  Why: selected after the remaining ready queue skewed toward GPU analytics;
  Epic is a modern deterministic MVCC OLTP design that uses GPU-parallel
  indexing/initialization to precompute direct version locations, avoid
  version-chain search, and reclaim epoch scratchpad versions wholesale.
- `reviewed` — **Aria: A Fast and Practical Deterministic OLTP Database**,
  Lu et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p2047-lu.pdf`
  DOI: `https://doi.org/10.14778/3407790.3407808`
  Why: Epic compares against Aria's deterministic abort/fallback strategy;
  useful for deciding when GPU DB should prefer deterministic rerun,
  lock-based fallback, or owner-serialized execution for mispredicted
  read/write sets. Journal entry added 2026-06-06.
- `queued` — **Fast Abort-Freedom for Deterministic Transactions**,
  Chen, Wu, Zhong, and Eriksson, IPDPS 2024.
  URL: `https://par.nsf.gov/servlets/purl/10548863`
  Why: discovered while reviewing Aria; modern deterministic-transaction
  follow-up focused on reducing aborts, useful for testing whether GPU DB can
  keep batch/snapshot execution while avoiding retry amplification under
  hot-key or dependency-heavy workloads.
- `reviewed` — **Caracal: Contention Management with Deterministic
  Concurrency Control**, Qin, Demke Brown, and Goel, SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483591`
  PDF: `https://www.eecg.utoronto.ca/~ashvin/publications/caracal.pdf`
  Why: Epic contrasts its shared-memory initialization and direct version
  lookup with Caracal's sorted-array version allocation; relevant to
  high-contention MVCC batch planning and direct version slot assignment.
- `queued` — **The Tale of 1000 Cores: An Evaluation of Concurrency Control
  on Real(ly) Large Multi-Socket Hardware**, Bang, May, Petrov, and Binnig,
  DaMoN 2020.
  URL: `https://doi.org/10.1145/3399666.3399916`
  Why: CCBench cites it as a modern thousand-core concurrency-control
  evaluation; useful for deciding whether GPU DB owner-thread and
  partition-owner experiments should emulate many-core effects or require
  real multi-socket validation.
- `reviewed` — **NWR: Rethinking Thomas Write Rule for Omittable Write
  Operations**, Nakazono et al., arXiv 2020.
  URL: `https://arxiv.org/abs/1904.08119`
  Why: CCBench points to non-visible writes as a version-lifetime direction;
  NWR is relevant to blind-write and stale-version elision without weakening
  WAL-before-visibility or SQL-visible conflict behavior.
- `reviewed` — **No False Negatives: Accepting All Useful Schedules in a Fast
  Serializable Many-Core System**, Durner and Neumann, ICDE 2019.
  URL: `https://doi.org/10.1109/ICDE.2019.00071`
  PDF: `https://db.cs.tum.edu/~durner/papers/no-false-negatives-icde19.pdf`
  Why: NWR contrasts against graph-based serializable scheduling; useful for
  deciding whether GPU DB should accept more useful mixed read/write schedules
  with explicit conflict graphs instead of relying only on abort-heavy OCC or
  narrow non-visible-write elision.
- `queued` — **Velox: Meta's Unified Execution Engine**,
  Pedreira et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p3372-pedreira.pdf`
  Why: FastLanes explicitly targets in-flight compressed vectors used by
  engines such as Velox; useful for comparing vector representation,
  operator reuse, and CPU/GPU route compatibility for compressed execution.
- `reviewed` — **Data Blocks: Hybrid OLTP and OLAP on Compressed Storage Using
  Both Vectorization and Compilation**, Lang et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882925`
  Why: FastLanes cites it as compressed execution context; useful for deciding
  whether GPU DB should keep one compressed storage representation that serves
  OLTP lookups, retained scans, and compiled/vectorized operators. Journal
  entry exists from 2026-06-04; this stale duplicate was marked reviewed on
  2026-06-05.
- `queued` — **ByteSlice: Pushing the Envelope of Main Memory Data Processing
  with a New Storage Layout**, Feng et al., SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2747642`
  Why: FastLanes contrasts decompression-first layouts with scan-first
  bit/byte-sliced layouts; useful for resident predicate scans and point
  lookups when full materialization is unnecessary.
- `queued` — **Keep CALM and CRDT On**, Laddad et al., PVLDB 2022.
  URL: `https://doi.org/10.14778/3574245.3574268`
  Why: CRDV cites it as prior work on analyzing arbitrary queries over
  CRDTs; useful for deciding which retained/replicated derived views are
  monotonic enough to run without owner coordination.
- `queued` — **Conflict-free Replicated Relations for Multi-Synchronous
  Database Management at Edge**, Yu and Ignat, IEEE SMDS 2020.
  URL: `https://doi.org/10.1109/SMDS49396.2020.00020`
  Why: CRDV contrasts against relational CRDT approaches with simpler
  conflict semantics; useful as a baseline for SQL-native replicated route
  tables, derived counters, and edge/session-local state.
- `queued` — **Exploiting Single-Threaded Model in Multi-Core In-Memory
  Systems**, Yao et al., IEEE TKDE 2016.
  URL: `https://doi.org/10.1109/TKDE.2016.2578319`
  Why: QueCC contrasts against LADS-style dependency-graph-driven execution;
  useful for comparing deterministic queue planning with graph partitioning
  when GPU DB splits stored-procedure fragments across owner lanes.
- `queued` — **Scaling Multicore Databases via Constrained Parallel
  Execution**, Wang et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882934`
  Why: QueCC cites transaction chopping and constrained parallel execution as
  related fragment models; useful for deciding when GPU DB should expose
  template-level dependency constraints instead of only route-local queues.
- `reviewed` — **Analyzing and Implementing GPU Hash Tables**, Awad et al.,
  APOCS 2023.
  URL: `https://doi.org/10.1137/1.9781611977578.ch3`
  Author/code page: `https://github.com/owensgroup/BGHT`
  Why: the GPU multiversion B-tree project identifies BGHT as a companion
  GPU data-structure design with device-side APIs; useful for comparing
  retained B-tree versus hash-index lookup routes, probe bounds, and
  snapshot-friendly index rebuild options.
- `queued` — **Mega-KV: A Case for GPUs to Maximize the Throughput of
  In-Memory Key-Value Stores**, Zhang et al., PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol8/p1226-zhang.pdf`
  Why: compared by the BGHT paper as a GPU key-value/hash-table baseline;
  useful for deciding whether resident point-lookup indexes should be a DBMS
  route only or a GPU-side key-value service boundary.
- `queued` — **Data-Parallel Hashing Techniques for GPU Architectures**,
  Lessley and Childs, IEEE TPDS 2020.
  URL: `https://doi.org/10.1109/TPDS.2019.2926406`
  Why: BGHT compares against broader GPU hashing approaches; useful for
  separating probing/layout effects from application-specific database lookup
  behavior.
- `reviewed` — **More Bang For Your Buck(et): Fast and Space-efficient
  Hardware-accelerated Coarse-granular Indexing on GPUs**, Henneberg et al.,
  arXiv 2024.
  URL: `https://arxiv.org/abs/2406.03965`
  Why: discovered while following GPU index literature around MVGpuBTree; uses
  hardware-accelerated coarse-granular indexing ideas that may compete with or
  complement resident GPU B-tree/hash indexes for selective predicates.
- `reviewed` — **RTIndeX: Exploiting Hardware-Accelerated GPU Raytracing for
  Database Indexing**, Henneberg and Schuhknecht, PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p4268-schuhknecht.pdf`
  DOI: `https://doi.org/10.14778/3625054.3625061`
  Why: cgRX generalizes RTIndeX/RX and measures against it; useful as the
  fine-grained RT-core index baseline before adopting coarse resident buckets.
- `queued` — **Autopilot: Workload Autoscaling at Google**, Rzadca et al.,
  EuroSys 2020.
  URL: `https://doi.org/10.1145/3342195.3387524`
  PDF: `https://dl.acm.org/doi/pdf/10.1145/3342195.3387524`
  Why: Eigen uses Autopilot as a baseline for resource prediction; useful for
  comparing smoothed resource limits, safety margins, and control-loop
  stability for GPU DB active-session, pinned-buffer, and cache budgets.
- `reviewed` — **Moneyball: Proactive Auto-Scaling in Microsoft Azure SQL
  Database Serverless**, Poppe et al., PVLDB 2022.
  URL: `https://doi.org/10.14778/3514061.3514073`
  PDF: `https://www.vldb.org/pvldb/vol15/p1279-poppe.pdf`
  Why: Eigen contrasts serverless database pause/resume and proactive
  provisioning work; useful for session-admission and cold/warm route
  provisioning policies when dormant tenants or idle logical sessions become
  active quickly.
- `reviewed` — **Seagull: An Infrastructure for Load Prediction and Optimized
  Resource Allocation**, Poppe et al., PVLDB 2021.
  URL: `https://doi.org/10.14778/3425879.3425886`
  PDF: `https://www.vldb.org/pvldb/vol14/p154-poppe.pdf`
  Why: Moneyball transfers lessons from Azure SQL provisioned-database load
  prediction; useful for comparing per-tenant/session historical route demand,
  lightweight predictors, and maintenance overhead for GPU DB admission and
  warm-cache budgeting.
- `queued` — **Resource Central: Understanding and Predicting Workloads for
  Improved Resource Management in Large Cloud Platforms**, Cortez et al.,
  SOSP 2017.
  URL: `https://doi.org/10.1145/3132747.3132772`
  Why: Seagull cites Resource Central as model-serving and workload-prediction
  infrastructure; useful for comparing fleet-scale resource prediction,
  model/version management, and prediction-service overhead before GPU DB
  builds ML-driven admission or warm-cache controllers.
- `queued` — **Predictive Provisioning: Efficiently Anticipating Usage in
  Azure SQL Database**, Viswanathan et al., ICDE 2017.
  URL: `https://doi.org/10.1109/ICDE.2017.164`
  Why: Seagull cites Azure SQL predictive provisioning as a database-specific
  resource anticipation baseline; useful for comparing idle detection,
  overbooking, and pre-provisioning policies against GPU DB warm-route and
  logical-session admission loops.
- `reviewed` — **LTPG: Large-Batch Transaction Processing on GPUs with
  Deterministic Concurrency Control**, Wei, Gu, Li, and Yu, ICDE 2024.
  URL: `https://doi.org/10.1109/ICDE60146.2024.00296`
  Metadata: `https://vbn.aau.dk/en/publications/ltpg-large-batch-transaction-processing-on-gpus-with-deterministi`
  Why: GaccO motivates same-type GPU transaction batching; LTPG is a newer
  deterministic GPU transaction-processing follow-up for testing whether
  larger batches and conflict planning can preserve latency while improving
  throughput.
- `queued` — **GalOP: Towards a GPU-accelerated OLTP DBMS**, Boeschen and
  Binnig, DaMoN 2021.
  URL: `https://doi.org/10.1145/3465998.3466007`
  Metadata: `https://www.dfki.de/en/web/research/projects-and-publications/publication/14419`
  Why: precursor to GaccO with a deterministic GPU concurrency scheme;
  useful for isolating which design choices came from the smaller prototype
  versus the later CPU/GPU co-execution storage design.
- `reviewed` — **Mostly-Optimistic Concurrency Control for Highly Contended
  Dynamic Workloads on a Thousand Cores**, Wang and Kimura, PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol10/p49-wang.pdf`
  DOI: `https://doi.org/10.14778/3015274.3015276`
  Why: No False Negatives contrasts graph-based scheduling with hybrid
  pessimistic/optimistic hot-tuple handling; useful for deciding when GPU DB
  should switch hot keys or route families from optimistic batch admission to
  owner-serialized or lock-like handling.
- `reviewed` — **Opportunities for Optimism in Contended Main-Memory Multicore
  Transactions**, Huang et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p629-huang.pdf`
  Why: MOCC cites hybrid optimistic/pessimistic contention handling as a
  related direction; useful for comparing transaction-level and record-level
  promotion when hot keys appear dynamically.
- `queued` — **The Impact of Timestamp Granularity in Optimistic Concurrency
  Control**, Yu et al., arXiv 2018.
  URL: `https://arxiv.org/abs/1811.04967`
  Why: MOCC and TicToc both depend on decentralized OCC commit metadata;
  useful for measuring whether tuple, page, partition, or route-level
  timestamps best fit GPU DB validation and visibility summaries.
- `reviewed` — **Zero-sided RDMA: Network-driven Data Shuffling for
  Disaggregated Heterogeneous Cloud DBMSs**, Jasny, Thostrup, Tamimi,
  Koch, Istvan, and Binnig, PACMMOD 2024.
  URL: `https://doi.org/10.1145/3639291`
  PDF:
  `https://www.informatik.tu-darmstadt.de/media/systems/pdf_publications/zerosided_rdma_sigmod.pdf`
  Why: follow-up from the same network/accelerator line that offloads RDMA
  data movement to programmable switches; useful for future GPU DB
  accelerator-pool shuffling, global-order replication, and CPU-free
  producer/consumer rings.
- `skipped` — **Zero-sided RDMA: Network-driven Data Shuffling**, Jasny,
  Thostrup, and Binnig, DaMoN 2023.
  URL: `https://doi.org/10.1145/3592980.3595302`
  Why: compact workshop version of switch-driven RDMA shuffling; useful if the
  loop needs the smaller source before the full PACMMOD 2024 version. Skipped
  because the full PACMMOD 2024 version has now been reviewed.
- `queued` — **Efficiently Joining Large Relations on Multi-GPU Systems**,
  Maltenberger, Tolovski, and Rabl, PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4653-maltenberger.pdf`
  DOI: `https://doi.org/10.14778/3749646.3749720`
  Why: recent multi-GPU join follow-up that cites distributed GPU RDMA joins;
  useful for comparing network-scale partitioned joins with intra-node
  multi-GPU sort-merge, P2P interconnect use, out-of-core joins, and transfer
  scheduling.
- `reviewed` — **Adaptive Concurrency Control: Despite the Looking Glass, One
  Concurrency Control Does Not Fit All**, Tang, Jiang, and Elmore, CIDR 2017.
  URL: `http://cidrdb.org/cidr2017/papers/p63-tang-cidr17.pdf`
  Why: Strife contrasts with adaptive per-cluster concurrency-control
  selection; useful for deciding whether GPU DB hot partitions should switch
  among owner-serialized, optimistic, lock-like, or GPU-batched routes based on
  measured conflict shape instead of one global write-path policy.
- `queued` — **Clay: Fine-Grained Adaptive Partitioning for General Database
  Schemas**, Serafini et al., PVLDB 2016.
  URL: `https://doi.org/10.14778/3007328.3007331`
  Why: ACC points to fine-grained adaptive partitioning as related dynamic
  placement work; useful for deciding how GPU DB should migrate hot keys or
  route boundaries without forcing a global owner-layout rebuild.
- `queued` — **Leopard: Lightweight Edge-Oriented Partitioning and Replication
  for Dynamic Graphs**, Huang and Abadi, PVLDB 2016.
  URL: `https://doi.org/10.14778/2904483.2904495`
  Why: ACC cites Leopard as online partitioning for dynamic datasets; useful
  for comparing lightweight scoring and incremental boundary changes against
  route-descriptor based hot-key clustering.
- `reviewed` — **Hybrid Deterministic and Nondeterministic Execution of
  Transactions in Actor Systems**, Liu et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526172`
  PDF: `https://hjemmesider.diku.dk/~vmarcos/pubs/LSS_22-hybridtxnsactors.pdf`
  Why: HDCC contrasts against Snapper's batch-level Calvin/2PL validation;
  useful for comparing coarse batch validation with finer per-transaction
  dependency tracking when GPU DB mixes deterministic batches and optimistic
  owner lanes.
- `reviewed` — **Knock Out 2PC with Practicality Intact: A High-performance and
  General Distributed Transaction Protocol**, Lai et al., ICDE 2023.
  URL: `https://doi.org/10.1109/ICDE55515.2023.00179`
  arXiv: `https://arxiv.org/abs/2302.12517`
  Why: HDCC contrasts deterministic execution against 2PC-like deterministic
  optimistic protocols; useful for measuring whether GPU DB multi-owner write
  batches should avoid 2PC entirely or make the commit path cheaper.
- `reviewed` — **Lotus: Scalable Multi-Partition Transactions on Single-Threaded
  Partitioned Databases**, Zhou, Yu, Graefe, and Stonebraker, PVLDB 2022.
  URL: `https://doi.org/10.14778/3551793.3551843`
  PDF: `https://www.vldb.org/pvldb/vol15/p2939-zhou.pdf`
  Why: Primo contrasts Lotus' epoch/lock-holding tradeoff with asynchronous
  watermarks; useful for comparing single-threaded partition owners,
  multi-partition write routing, and whether GPU DB should hold or release
  owner-lane locks across publication boundaries.
- `queued` — **Plan-Structured Deep Neural Network Models for Query
  Performance Prediction**, Marcus and Papaemmanouil, PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p1733-marcus.pdf`
  DOI: `https://doi.org/10.14778/3342263.3342646`
  Why: direct baseline for GPredictor's concurrent-query model; useful for
  comparing single-plan latency prediction against GPU DB route descriptors
  that must incorporate shared cache, lock, and accelerator-resource edges.
- `queued` — **DyCuckoo: Dynamic Hash Tables on GPUs**, Li et al.,
  ICDE 2021.
  URL: `https://doi.org/10.1109/ICDE51399.2021.00070`
  Why: RTIndeX compares against static GPU index baselines and highlights
  update/rebuild limits; DyCuckoo is a modern dynamic GPU hash-table follow-up
  for point-lookup routes that need device-side updates.
- `queued` — **GPU LSM: A Dynamic Dictionary Data Structure for the GPU**,
  Ashkiani et al., IPDPS 2018.
  URL: `https://doi.org/10.1109/IPDPS.2018.00053`
  Why: RTIndeX's update weakness raises the question of log-structured GPU
  indexes; useful for comparing rebuild-heavy resident structures with
  mutable GPU dictionary layers under MVCC generation boundaries.
- `queued` — **GPUrdma: GPU-side library for high performance networking from
  GPU kernels**, Daoud, Wated, and Silberstein, ROSS 2016.
  URL: `https://doi.org/10.1145/2931088.2931091`
  Why: Zero-sided RDMA contrasts against accelerator-driven RDMA stacks;
  useful for measuring when GPU-side networking is worth its SIMT/control-flow
  cost versus CPU-, switch-, or NIC-driven data movement.
- `queued` — **Rack-Scale In-Memory Join Processing using RDMA**, Barthels,
  Loesing, Alonso, and Kossmann, SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2750547`
  Why: Zero-sided RDMA uses RDMA shuffle/join literature as its DBMS baseline;
  useful for comparing classic rack-scale RDMA repartitioning with future GPU
  DB accelerator-pool shuffle and resident-partition movement.
- `queued` — **Zeus: Locality-aware Distributed Transactions**,
  Katsarakis et al., EuroSys 2021.
  URL: `https://doi.org/10.1145/3447786.3456245`
  Why: ScaleStore contrasts Zeus' ownership/movement approach with its
  coherent DRAM/NVMe page protocol; useful for deciding when GPU DB should
  move hot objects to owner lanes versus run distributed commit or remote
  accelerator access.
- `reviewed` — **X-SSD: A Storage System with Native Support for Database Logging
  and Replication**, Lee et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526188`
  Why: Database Kernels cites X-SSD as a dedicated storage-device path for
  database logging; useful for comparing append-only CXL logging kernels with
  WAL-before-visibility and recovery accelerator designs.
- `reviewed` — **Delilah: eBPF-Offload on Computational Storage**, Hedam et al.,
  DaMoN 2023.
  URL: `https://doi.org/10.1145/3592980.3595319`
  PDF: `https://hed.am/papers/2023-DaMoN.pdf`
  Why: Database Kernels cites Delilah as a programmable computational-storage
  example; useful for deciding whether storage-side database functions should
  be fixed kernels, eBPF-like user functions, or DBMS-owned operators.
- `queued` — **BPF-oF: Storage Function Pushdown Over the Network**,
  Delamare et al., arXiv 2023.
  URL: `https://arxiv.org/abs/2312.06808`
  Why: modern remote-storage pushdown protocol over NVMe-oF using eBPF-like
  storage functions; useful follow-up for comparing local computational
  storage offload with disaggregated cold-tier predicate/data-massaging
  routes.
- `queued` — **An Evaluation of WebAssembly and eBPF as Offloading Mechanisms
  in the Context of Computational Storage**, Delamare et al., arXiv 2021.
  URL: `https://arxiv.org/abs/2111.01947`
  Why: compares programmable offload mechanisms for computational storage;
  useful for deciding whether GPU DB cold-tier functions should use eBPF,
  WebAssembly, fixed kernels, or DBMS-owned generated operators.
- `queued` — **Hello Bytes, Bye Blocks: PCIe Storage Meets Compute Express Link
  for Memory Expansion (CXL-SSD)**, Jung, HotStorage 2022.
  URL: `https://doi.org/10.1145/3538643.3539745`
  Why: Database Kernels contrasts naive CXL-Flash memory expansion with richer
  DBK semantics; useful for benchmarking simple load/store CXL storage against
  explicit database-owned placement and offload contracts.

- `queued` — **A Dynamic Hash Table for the GPU**, Ashkiani, Farach-Colton,
  and Owens, IPDPS 2018.
  URL: `https://doi.org/10.1109/IPDPS.2018.00052`
  Why: the GPU B-Tree paper uses its warp cooperative work-sharing strategy;
  useful for comparing resident GPU hash indexes against B-tree and LSM-style
  mutable dictionary routes for point lookups and batched updates.
- `queued` — **PLayer: Expanding Coherence Protocol Stack with a Persistence
  Layer**, Braun, Ramdas, Friedman, and Alonso, DIMES 2023.
  URL: `https://doi.org/10.1145/3609308.3625270`
  Why: Database Kernels points to coherent virtually materialized views and
  persistence-layer coherence mechanisms; useful for future row/column view
  invalidation across CPU, CXL, and GPU-resident representations.
- `reviewed` — **Correct, Fast Remote Persistence**, Kashyap et al., arXiv 2019.
  URL: `https://arxiv.org/abs/1909.02092`
  Why: X-SSD cites remote-PM persistence ambiguity as a motivation; useful for
  defining exact durability acknowledgements when WAL, replicas, NICs, GPUs,
  and future persistent tiers overlap.
- `queued` — **Rethinking Database High Availability with RDMA Networks**,
  Zamanian, Yu, Stonebraker, and Kraska, PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p1637-zamanian.pdf`
  DOI: `https://doi.org/10.14778/3342263.3342639`
  Why: X-SSD contrasts device-managed log propagation with Active Memory's
  RDMA-based fresh replica path; useful for comparing storage-owned durability
  with replica-owned visibility and freshness.
- `reviewed` — **Write-behind Logging**, Arulraj, Perron, and Pavlo,
  PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol10/p337-arulraj.pdf`
  DOI: `https://doi.org/10.14778/3025111.3025116`
  Why: X-SSD's fast-side API can host PM-oriented logging schemes; useful for
  comparing delayed durability placement with GPU DB's WAL-before-visibility
  invariant and batch publication boundary.
- `queued` — **Octopus: an RDMA-enabled Distributed Persistent Memory File
  System**, Lu et al., USENIX ATC 2017.
  URL: `https://www.usenix.org/conference/atc17/technical-sessions/presentation/lu`
  Why: Correct, Fast Remote Persistence cites Octopus as early RDMA plus PM
  work; useful for comparing remote persistent-memory file semantics with GPU
  DB WAL, cold-tier namespace, and replica durability counters.
- `queued` — **Mojim: A Reliable and Highly-Available Non-Volatile Memory
  System**, Zhang et al., ASPLOS 2015.
  URL: `https://doi.org/10.1145/2694344.2694370`
  Why: Correct, Fast Remote Persistence cites Mojim as an early HA NVM system;
  useful for comparing primary/backup memory replication and failure handling
  with explicit WAL-before-visibility and retained snapshot clocks.
- `queued` — **Failure-Atomic Persistent Memory Updates via JUSTDO Logging**,
  Izraelevitz, Kelly, and Kolli, ASPLOS 2016.
  URL: `https://doi.org/10.1145/2872362.2872410`
  Why: Correct, Fast Remote Persistence uses local persistence-domain and
  flush-ordering assumptions that build on PM logging work; useful for
  comparing local PM commit records with remote durability acknowledgement
  recipes.
- `reviewed` — **Robust Query Driven Cardinality Estimation under Changing
  Workloads**, Negi et al., PVLDB 2023.
  URL: `https://doi.org/10.14778/3583140.3583164`
  PDF: `https://www.vldb.org/pvldb/vol16/p1520-negi.pdf`
  Why: Stage calls out workload and data drift as a predictor weakness; this
  paper is a modern follow-up for robust route-cardinality signals when GPU DB
  query mixes or resident-cache contents shift.
- `queued` — **Warper: Efficiently Adapting Learned Cardinality Estimators to
  Data and Workload Drifts**, Li, Lu, and Kandula, SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517871`
  Why: Robust query-driven cardinality estimation contrasts against retraining
  and drift-adaptation systems; useful for deciding when GPU DB route models
  should retrain, adapt online, or fall back to anchored DBMS statistics.
- `reviewed` — **ALECE: An Attention-based Learned Cardinality Estimator for SPJ
  Queries on Dynamic Workloads**, Li et al., PVLDB 2023.
  URL: `https://arxiv.org/abs/2310.05349`
  Why: modern learned cardinality estimator for dynamic workloads; useful for
  comparing data-update-aware route estimates against simpler DBMS-statistics
  correction and explicit route telemetry. Journal entry added 2026-06-06.
- `reviewed` — **CardOOD: Robust Query-driven Cardinality Estimation under
  Out-of-Distribution Workloads**, Li, Zhao, Yu, and Wang, arXiv 2024 /
  VLDB Journal 2026.
  URL: `https://arxiv.org/abs/2412.05864`
  DOI: `https://doi.org/10.1007/s00778-026-00979-3`
  Why: direct follow-up on out-of-distribution robustness for query-driven
  cardinality estimation; useful for GPU DB when tenant workloads, resident
  cache contents, or mixed CPU/GPU route families drift from training logs.
- `reviewed` — **Data-Agnostic Cardinality Learning from Imperfect
  Workloads**, Wu et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p2519-wu.pdf`
  Why: modern query-driven cardinality estimator for incomplete and
  imbalanced join templates; useful follow-up for GPU DB route models when
  training logs do not cover all tenant query shapes.
- `reviewed` — **Convolution and Cross-Correlation of Count Sketches Enables
  Fast Cardinality Estimation of Multi-Join Queries**, Heddes et al.,
  PACMMOD 2024.
  URL: `https://doi.org/10.1145/3654932`
  arXiv: `https://arxiv.org/abs/2402.15953`
  Why: GRASP's learned count sketch mechanism points to count-sketch
  composition as a compact way to approximate join correlations; useful for
  comparing learned sketches with more direct sketch algebra for GPU DB route
  estimates.
- `queued` — **JoinSketch: A Sketch Algorithm for Accurate and Unbiased
  Inner-Product Estimation**, Wang et al., PACMMOD 2023.
  URL: `https://doi.org/10.1145/3589318`
  Why: cited as a skew-aware sketching complement; useful for comparing
  heavy-key separation with count-sketch multi-join estimates for GPU route
  cardinality and resident predicate-filter sizing.
- `queued` — **COMPASS: Online Sketch-Based Query Optimization for In-Memory
  Databases**, Izenov et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457281`
  Why: direct sketch-based query-optimization baseline for online filtered
  cardinality estimation; useful for deciding which predicates should be
  handled at sketch-ingest time versus route-evaluation time.
- `queued` — **Query Performance Prediction for Concurrent Queries using Graph
  Embedding**, Zhou, Sun, Li, and Feng, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p1416-zhou.pdf`
  DOI: `https://doi.org/10.14778/3397230.3397238`
  Why: Zero-shot cost models cite this concurrent-query prediction work;
  useful for extending single-route CPU/GPU latency prediction with queue
  depth, co-running kernels, response-ring pressure, and tier-stall features.
- `queued` — **Pessimistic Cardinality Estimation: Tighter Upper Bounds for
  Intermediate Join Cardinalities**, Cai, Balazinska, and Suciu, SIGMOD 2019.
  URL: `https://doi.org/10.1145/3299869.3319894`
  Why: cited as an upper-bound alternative to sketch and learned estimators;
  useful for guardrailing GPU route choices when an underestimated join could
  overflow HBM, pinned buffers, or response-ring budgets.
- `queued` — **ASM: Harmonizing Autoregressive Model, Sampling, and
  Multi-dimensional Statistics Merging for Cardinality Estimation**,
  Kim et al., PACMMOD 2024.
  URL: `https://doi.org/10.1145/3639300`
  Why: GRASP contrasts data-agnostic query-driven estimates with
  statistics/data-assisted estimators; ASM is a modern reference point for
  deciding when GPU DB should use resident samples, base statistics, or
  query-only telemetry in route costing.
- `queued` — **Estimating Filtered Group-By Queries is Hard: Deep Learning
  to the Rescue**, Kipf et al., AIDB 2019.
  PDF: `https://db.in.tum.de/~kipf/papers/learnedgroupby.pdf`
  Why: GRASP leaves Group By and Distinct outside scope, while GPU DB's
  retained route family already includes aggregates and distinct projections;
  this is a targeted follow-up for group-cardinality estimates in route
  selection.
- `queued` — **Flow-Loss: Learning Cardinality Estimates That Matter**,
  Negi et al., PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p2019-negi.pdf`
  DOI: `https://doi.org/10.14778/3476249.3476259`
  Why: ALECE optimizes q-error and end-to-end query plans, while Flow-Loss
  trains estimators against plan-impact-sensitive loss; useful for deciding
  whether GPU DB route estimators should optimize route regret and overload
  risk instead of raw cardinality error.
- `queued` — **A Unified Transferable Model for ML-Enhanced DBMS**, Wu et al.,
  CIDR 2022.
  URL: `https://www.cidrdb.org/cidr2022/papers/p20-wu.pdf`
  Why: ALECE is schema/workload trained, while transferable DBMS models ask
  how learned components move across databases and tasks; useful before GPU DB
  depends on per-tenant route models that may need cold-start or migration
  behavior.
- `reviewed` — **Updateable Data-Driven Cardinality Estimator with Bounded
  Q-error**, Li et al., arXiv 2024.
  URL: `https://arxiv.org/abs/2408.17209`
  Why: updateable cardinality estimator with bounded-error claims; useful
  counterpoint to offline query-driven retraining when GPU DB table updates
  and resident snapshots shift faster than route logs can be relabeled.
- `reviewed` — **LMSFC: A Novel Multidimensional Index Based on Learned
  Monotonic Space Filling Curves**, Gao et al., PVLDB 2023.
  URL: `https://doi.org/10.14778/3603581.3603598`
  PDF: `https://www.vldb.org/pvldb/vol16/p2605-gao.pdf`
  Why: ICE uses multidimensional index filtering efficiency as the main
  control on estimator variance and cites LMSFC as a learned
  space-filling-curve index; useful for comparing route-cardinality sketches
  against resident multidimensional index layouts.
- `queued` — **Tsunami: A Learned Multi-dimensional Index for Correlated
  Data and Skewed Workloads**, Ding, Nathan, Alizadeh, and Kraska,
  PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p74-ding.pdf`
  DOI: `https://doi.org/10.14778/3425879.3425880`
  Why: LMSFC compares against Tsunami as a workload/data-aware learned
  multidimensional index; useful for testing whether GPU DB resident
  predicate indexes should adapt by correlated regions before trying a
  learned global space-filling curve.
- `queued` — **Learning Multi-Dimensional Indexes**, Nathan et al.,
  SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3380579`
  arXiv: `https://arxiv.org/abs/1912.01668`
  Why: LMSFC and Tsunami both build on Flood's learned multidimensional
  layout ideas; useful as the baseline for whether route-specific index
  construction should jointly optimize data layout, grid partitioning, and
  query workload rather than only adding a secondary resident index.
- `reviewed` — **PACE: Poisoning Attacks on Learned Cardinality Estimation**,
  Zhang, Zhang, Li, and Chai, PACMMOD 2024.
  URL: `https://doi.org/10.1145/3639292`
  Why: ICE treats estimator freshness and updateability as planner inputs, but
  route models also need robustness against bad or adversarial training/query
  feedback; useful for designing guarded learned route telemetry.
- `queued` — **Detect, Distill and Update: Learned DB Systems Facing Out of
  Distribution Data**, Kurmanji and Triantafillou, PACMMOD 2023.
  URL: `https://doi.org/10.1145/3588929`
  Why: PACE's VAE normality pressure shows simple distribution checks can be
  modeled by an attacker; this related OOD-detection paper is useful for
  comparing defensive workload gating, retraining, and route fallback policies.
- `queued` — **AutoCE: An Accurate and Efficient Model Advisor for Learned
  Cardinality Estimation**, Zhang et al., ICDE 2023.
  URL: `https://doi.org/10.1109/ICDE55515.2023.00200`
  Why: PACE suggests CE-model vulnerability varies by model family and
  hyperparameters; AutoCE is a follow-up for choosing route-estimator families
  by workload, robustness, and training cost instead of adopting one learned
  model globally.
- `reviewed` — **Buffer Pool Aware Query Scheduling via Deep Reinforcement
  Learning**, Zhang et al., AIDB@VLDB 2020.
  URL:
  `https://drive.google.com/file/d/1trNYAcQ3S71SHu5dbtkBR2hjcK-dIHt21c-/view`
  Why: Stage identifies buffer-pool and cache state as hard-to-featurize
  environment factors; useful for comparing learned scheduling with explicit
  GPU/host/NVMe residency telemetry and cache-aware admission. Duplicate
  queue entry marked reviewed on 2026-06-06; journal entry uses the arXiv v3
  source at `https://arxiv.org/abs/2007.10568`.
- `reviewed` — **Auto-WLM: Machine Learning Enhanced Workload Management in
  Amazon Redshift**, Saxena et al., SIGMOD Companion 2023.
  URL: `https://doi.org/10.1145/3555041.3589677`
  Why: Stage compares against Redshift's prior workload-manager predictor;
  useful for understanding the production queue, priority, concurrency-scaling,
  and resource-control hooks that a GPU DB route predictor would influence.
- `reviewed` — **Adaptive HTAP through Elastic Resource Scheduling**, Raza,
  Chrysogelos, Anadiotis, and Ailamaki, SIGMOD 2020.
  URL: `https://arxiv.org/abs/2004.05437`
  DOI: `https://doi.org/10.1145/3318464.3389783`
  Why: OLxPBench cites elastic HTAP scheduling as a related design point;
  useful for comparing static resident route allocation against runtime
  resource exchange among OLTP, OLAP, and refresh/freshness work.
- `queued` — **HTAPBench: Hybrid Transactional and Analytical Processing
  Benchmark**, Coelho et al., ICPE 2017.
  URL: `https://doi.org/10.1145/3030207.3030228`
  PDF: `https://rmpvilaca.github.io/assets/pdf/CPVPO17.pdf`
  Why: OLxPBench explicitly contrasts against HTAPBench's benchmark model;
  useful historical-but-eligible baseline for deciding which GPU DB benchmark
  gates must include hybrid transactions, freshness, and semantic schema
  overlap instead of only concurrent OLTP plus OLAP streams.
- `queued` — **Heracles: Improving Resource Efficiency at Scale**, Lo et al.,
  ISCA 2015.
  URL: `https://doi.org/10.1145/2749469.2749475`
  Why: Adaptive HTAP points to hardware/software resource controls for limiting
  interference; useful for GPU DB admission rules that cap memory-bandwidth,
  CPU, and accelerator-resource theft while protecting OLTP p95 latency.
- `queued` — **PerfIso: Performance Isolation for Commercial
  Latency-Sensitive Services**, Iorgulescu et al., USENIX ATC 2018.
  URL: `https://www.usenix.org/conference/atc18/presentation/iorgulescu`
  Why: Adaptive HTAP suggests live performance monitoring to bound elastic
  resource sharing; PerfIso is a primary systems follow-up for interference
  control and isolation when query, refresh, and mutation work compete.
- `queued` — **Columnstore and B+ Tree - Are Hybrid Physical Designs
  Important?**, Dziedzic et al., SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3190660`
  Why: Adaptive HTAP contrasts runtime resource scheduling with hybrid access
  methods; useful for deciding when GPU DB should adapt physical route shape
  versus moving resources or freshness boundaries.
- `reviewed` — **Deferred Runtime Pipelining for Contentious Multicore Software
  Transactions**, Mu, Angel, and Shasha, EuroSys 2019.
  URL: `https://doi.org/10.1145/3302424.3303966`
  PDF: `https://www.cis.upenn.edu/~sga001/papers/drp-eurosys19.pdf`
  Why: Opportunities for Optimism contrasts manual commit-time updates with
  DRP's lazy/deferred execution; useful for comparing automatic transaction
  chopping against explicit GPU DB route-shape annotations for hot writes.
- `reviewed` — **OCToPus: Semantic-aware Concurrency Control for Blockchain
  Transactions**, Miller, Korth, and Palmieri, PPoPP 2024 poster.
  URL:
  `https://ppopp24.sigplan.org/details/PPoPP-2024-papers/42/POSTER-OCToPus-Semantic-aware-Concurrency-Control-for-Blockchain-Transactions`
  PDF: `https://par.nsf.gov/servlets/purl/10495038`
  Why: recent semantic-aware concurrency-control work with a GPU-accelerated
  graph fallback path; useful for comparing O|R|P|E-style semantic classes with
  deterministic DAG fallback for constrained transaction domains.
- `reviewed` — **Block-STM: Scaling Blockchain Execution by Turning Ordering
  Curse to a Performance Blessing**, Gelashvili et al., PPoPP 2023.
  URL: `https://doi.org/10.1145/3572848.3577524`
  arXiv: `https://arxiv.org/abs/2203.06871`
  Why: OCToPus cites Block-STM as a deterministic ordered blockchain execution
  baseline; useful for comparing optimistic parallel execution, dependency
  tracking, and re-execution costs against semantic fast paths.
- `reviewed` — **Forerunner: Constraint-based Speculative Transaction Execution
  for Ethereum**, Chen et al., SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483564`
  Why: Block-STM contrasts against constraint/pre-execution hints for smart
  contract transactions; useful for comparing off-critical-path route hints
  with active-window dependency learning. Journal entry added 2026-06-06.
- `queued` — **An Empirical Study of Speculative Concurrency in Ethereum Smart
  Contracts**, Bartoletti and Pompianu, arXiv 2019.
  URL: `https://arxiv.org/abs/1901.01376`
  Why: Forerunner's motivation and later blockchain execution work make this a
  useful measurement baseline for how much parallel/speculative work exists in
  smart-contract traces before adopting richer constraint-checked route hints.
- `reviewed` — **NEMO: Faster Parallel Execution for Highly Contended Blockchain
  Workloads**, Ezard, Ileri, and Decouchant, arXiv 2025.
  URL: `https://arxiv.org/abs/2510.15122`
  Why: modern high-contention blockchain execution follow-up discovered during
  the Forerunner review; useful for contrasting object-model OCC and
  contention-aware execution with GPU DB's hot-key write-window admission.
  Journal entry added 2026-06-07 from arXiv v1.
- `reviewed` — **Deferred Objects to Enhance Smart Contract Programming with
  Optimistic Parallel Execution**, Mitenkov et al., arXiv 2024.
  URL: `https://arxiv.org/abs/2405.06117`
  Why: NEMO contrasts RapidLane's deferred-object path as a way to turn
  conflict-heavy smart-contract operations into parallelizable work; useful for
  comparing semantic deferral, predicted results, and commit-time validation
  against GPU DB's hot-key write-window and route-hint admission designs.
  Journal entry added 2026-06-07 from arXiv v1.
- `queued` — **Utilizing Parallelism in Smart Contracts on Decentralized
  Blockchains by Taming Application-Inherent Conflicts**, Garamvolgyi,
  Liu, Zhou, Long, and Wu, ICSE 2022.
  URL: `https://doi.org/10.1145/3510003.3510086`
  Why: RapidLane contrasts deferred objects with prior attempts to tame
  application-inherent smart-contract conflicts; useful for comparing
  programmer-visible conflict structure, semantic hints, and high-contention
  execution before GPU DB exposes any hot-write route annotations.
- `queued` — **Practical Smart Contract Sharding with Ownership and
  Commutativity Analysis**, Pirlea, Kumar, and Sergey, PLDI 2021.
  URL: `https://doi.org/10.1145/3453483.3454112`
  Why: RapidLane discusses commutativity and sharding as related ways to avoid
  shared-state bottlenecks; useful for comparing ownership and commutativity
  declarations with GPU DB's owner domains, partition-local writes, and
  semantic deferred-delta lanes.
- `reviewed` — **Processing Transactions in a Predefined Order**, Saad et al.,
  PPoPP 2019.
  URL: `https://doi.org/10.1145/3293883.3295730`
  PDF: `https://www.cse.lehigh.edu/~palmieri/files/pubs/CR-ppopp2019.pdf`
  Why: Block-STM compares against predefined-order STM approaches; useful for
  deciding whether GPU DB should use commit-order forwarding, flat combining,
  or Block-STM-style collaborative validation for admitted write windows.
  Journal entry added 2026-06-06; DOI corrected from the stale queued value.
- `queued` — **Lerna: Parallelizing Dependent Loops Using Speculation**,
  Lou et al., SYSTOR 2018.
  URL: `https://doi.org/10.1145/3211890.3211894`
  PDF: `https://www.cse.lehigh.edu/~palmieri/files/pubs/CR-systor2018.pdf`
  Why: Processing Transactions in a Predefined Order names Lerna as a runtime
  that can integrate ordered STM for speculative execution; useful for
  comparing automatic loop/window speculation with explicit GPU DB write-window
  admission and rollback boundaries.
- `queued` — **SPEEDEX: A Scalable, Parallelizable, and Economically Efficient
  Decentralized EXchange**, Ramseyer, Goel, and Mazieres, arXiv 2021.
  URL: `https://arxiv.org/abs/2111.02719`
  Why: OCToPus cites SPEEDEX as a constrained financial-transaction execution
  model; useful for comparing semantic batching and deterministic ordering when
  transaction shape is narrower than general SQL.
- `queued` — **CFS: Scaling Metadata Service for Distributed File System via
  Pruned Scope of Critical Sections**, Wang et al., EuroSys 2023.
  URL: `https://doi.org/10.1145/3552326.3587443`
  Why: SwitchFS compares against CFS's fine-grained parent/child-separated
  metadata partitioning; useful for deciding how much route-cache and
  cold-tier namespace contention can be removed by shrinking the synchronous
  critical section before adding asynchronous dirty-state coordination.
- `queued` — **MetaWBC: POSIX-compliant metadata write-back caching for
  distributed file systems**, Qian et al., SC 2022.
  URL: `https://doi.org/10.1109/SC41404.2022.00060`
  Why: SwitchFS contrasts with metadata write-back caching; useful for
  comparing client-side delayed metadata visibility against database-owned
  route-cache, catalog, and cold-tier metadata publication rules.
- `reviewed` — **Self-Tuning Query Scheduling for Analytical Workloads**,
  Wagner, Kohn, and Neumann, SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457260`
  Why: Auto-WLM contrasts production admission and elasticity with
  self-tuned scheduling policies; useful for comparing low-overhead heuristic
  tuning against route-specific GPU/CPU/NVMe scheduling knobs. Duplicate
  candidate marked reviewed after the 2026-06-06 journal entry.
- `reviewed` — **LSched: A Workload-Aware Learned Query Scheduler for
  Analytical Database Systems**, Sabek, Ukyab, and Kraska, SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526158`
  PDF: `https://people.csail.mit.edu/ibrahimsabek/pdf/22_paper_lsched.pdf`
  Why: Auto-WLM cites learned analytical scheduling as related work; useful
  for deciding whether GPU DB route scheduling should learn from plan shape
  and system state or stay with guardrailed heuristics. Duplicate candidate
  marked reviewed; journal entry added 2026-06-06.
- `queued` — **Database-Agnostic Workload Management**, Jain, Yan, Cruanes,
  and Howe, CIDR 2019.
  URL: `https://www.cidrdb.org/cidr2019/papers/p82-jain-cidr19.pdf`
  Why: Auto-WLM cites database-agnostic workload management as related
  production-oriented scheduling work; useful for comparing external workload
  control with an engine-integrated GPU route/admission controller.
- `reviewed` — **IsoDiff: Debugging Anomalies Caused by Weak Isolation**, Gan,
  Ren, Ripberger, Blanas, and Wang, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p2773-gan.pdf`
  DOI: `https://doi.org/10.14778/3407790.3407860`
  Why: the mixed-isolation allocation paper contrasts static robustness with
  trace-based anomaly debugging; useful for a GPU DB isolation-template
  validation harness that compares static route certification with observed
  weak-isolation anomaly traces.
- `reviewed` — **Elle: Inferring Isolation Anomalies from Experimental
  Observations**, Kingsbury and Alvaro, PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p268-alvaro.pdf`
  arXiv: `https://arxiv.org/abs/2003.10554`
  Why: IsoDiff focuses on static trace-derived application anomaly debugging;
  Elle is the complementary experiment-driven isolation checker for database
  implementations, useful for validating GPU DB isolation claims with generated
  workloads and concise anomaly witnesses.
- `reviewed` — **Cobra: Making Transactional Key-Value Stores Verifiably
  Serializable**, Tan, Zhao, Mu, and Walfish, OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/tan`
  PDF: `https://www.usenix.org/system/files/osdi20-tan.pdf`
  Why: Elle checks generated histories with traceable datatypes; Cobra is a
  continuous black-box serializability verifier for transactional key-value
  workloads, useful for comparing route-history checking against solver-backed
  verification and transaction segmentation. Journal entry added 2026-06-06.
- `reviewed` — **Vbox: Efficient Black-Box Serializability Verification**, Sun
  and Zou, arXiv 2025.
  URL: `https://arxiv.org/abs/2503.05163`
  Why: discovered while reviewing Cobra; claims broader black-box
  serializability checking with predicate database operations and more use of
  transaction time information, directly relevant to moving GPU DB audit traces
  beyond point-key histories. Journal entry added 2026-06-06.
- `reviewed` — **Efficient Black-box Checking of Snapshot Isolation in
  Databases**, PolySI, arXiv 2023.
  URL: `https://arxiv.org/abs/2301.07313`
  PDF: `https://arxiv.org/pdf/2301.07313`
  DOI: `https://doi.org/10.14778/3583140.3583145`
  Why: discovered while reviewing Cobra's isolation-checking follow-up line;
  useful for validating MVCC/snapshot routes when the engine intentionally
  offers snapshot isolation or retained read snapshots rather than full
  serializability. Journal entry added 2026-06-07 from arXiv v2 / PVLDB 2023
  metadata.
- `queued` — **CoFI: Consistency-Guided Fault Injection for Cloud Systems**,
  Chen, Dou, Wang, and Qin, ASE 2020.
  URL: `https://doi.org/10.1145/3324884.3416548`
  Author PDF: `https://wsdou.github.io/papers/2020-ase-cofi.pdf`
  Project: `https://hanseychen.github.io/CoFI/`
  Why: PolySI's discussion points to fault injection as a natural way to
  trigger isolation bugs; useful for deciding how GPU DB should schedule
  network partitions, crash/replay interruptions, and tier/route invalidation
  faults around inconsistent states before running black-box SI witnesses.
- `reviewed` — **Viper: A Fast Snapshot Isolation Checker**, Zhang, Ji, Mu,
  and Tan, EuroSys 2023.
  URL: `https://doi.org/10.1145/3552326.3567492`
  PDF: `https://mpaxos.com/pub/viper-eurosys23.pdf`
  Why: discovered while reviewing Cobra; a fast SI checker is relevant to
  building low-overhead benchmark witnesses for retained snapshots, range
  reads, and MVCC route validation. Journal entry added 2026-06-06; the
  previously queued author metadata was corrected.
- `queued` — **On the Complexity of Checking Transactional Consistency**,
  Biswas and Enea, OOPSLA 2019.
  URL: `https://doi.org/10.1145/3360591`
  arXiv: `https://arxiv.org/abs/1908.04509`
  Why: Viper builds on the result that black-box checking SI is NP-complete;
  useful for separating hot-path route certification from offline or reduced
  history checking in GPU DB validation.
- `queued` — **Seeing is Believing: A Client-Centric Specification of
  Database Isolation**, Crooks, Pu, Alvisi, and Clement, PODC 2017.
  URL: `https://doi.org/10.1145/3087801.3087802`
  PDF: `https://www.cs.cornell.edu/~youerpu/papers/2017-podc-seeing.pdf`
  Why: Viper uses the Crooks et al. hierarchy of SI variants; useful for
  defining GPU DB retained-snapshot contracts in terms clients can observe
  across sessions, fallbacks, and refresh boundaries.
- `queued` — **Verifying Transactional Consistency of MongoDB**, Ouyang,
  Wei, Huang, Li, and Pan, arXiv 2021.
  URL: `https://arxiv.org/abs/2111.14946`
  Why: Viper contrasts black-box checking with MongoDB white-box verification;
  useful for deciding which GPU DB invariants should be proved from internal
  route/WAL/residency protocol facts rather than only tested from histories.
- `queued` — **Leopard: A Black-Box Approach for Efficiently Verifying Various
  Isolation Levels**, Li et al., ICDE 2023.
  URL: `https://doi.org/10.1109/ICDE55515.2023.00061`
  Why: Vbox contrasts Leopard as a protocol-aware isolation verifier; useful
  for deciding when GPU DB validation should use lightweight online checks for
  declared protocols versus protocol-agnostic serializability/SI witnesses.
- `queued` — **Detecting Isolation Bugs via Transaction Oracle Construction**,
  Dou et al., ICSE 2023.
  URL: `https://doi.org/10.1109/ICSE48619.2023.00101`
  Why: Vbox cites transaction-oracle work as evidence that claimed isolation
  can fail in practice; useful for generating adversarial SQL histories that
  exercise GPU DB retained snapshots, predicate routes, and fallback ordering.
- `reviewed` — **Understanding Transaction Bugs in Database Systems**, Cui et al.,
  ICSE 2024.
  URL: `https://doi.org/10.1145/3597503.3639207`
  PDF: `https://criszy.github.io/papers/2024-icse-txbug.pdf`
  Why: Vbox cites modern transaction-bug evidence; useful for turning real
  anomaly classes into GPU DB regression workloads for MVCC, WAL visibility,
  predicate reads, and route invalidation. Journal entry added 2026-06-06.
- `queued` — **Anomaly Pattern-guided Transaction Bug Testing in Relational
  Databases**, Xu et al., arXiv 2025.
  URL: `https://arxiv.org/abs/2511.17377`
  Why: discovered while reviewing the ICSE 2024 TXBug study; useful follow-up
  for turning empirical transaction-bug patterns into generated adversarial
  histories for retained snapshots, write publication, and route invalidation.
- `queued` — **DynaMast: Adaptive Dynamic Mastering for Replicated Systems**,
  Abebe, Glasbergen, and Daudjee, ICDE 2020.
  URL: `https://doi.org/10.1109/ICDE48307.2020.00123`
  Why: Detock contrasts home movement with dynamic mastering; useful for
  deciding whether GPU DB owner, route-cache, or resident-partition authority
  should migrate under changing locality or stay fixed with explicit fallback.
- `queued` — **MorphoSys: Automatic Physical Design Metamorphosis for
  Distributed Database Systems**, Abebe, Glasbergen, and Daudjee, PVLDB 2020.
  URL: `https://doi.org/10.14778/3424573.3424578`
  Why: Detock cites adaptive physical design for distributed locality changes;
  relevant to tier/partition placement when hot tables move between CPU,
  GPU-resident, NVMe, and future memory tiers.
- `queued` — **Smooth Scan: Robust Access Path Selection Without
  Cardinality Estimation**, Borovica-Gajic et al., VLDB Journal 2018.
  URL: `https://doi.org/10.1007/s00778-017-0477-4`
  Why: BinDex contrasts against adaptive morphing between index and scan
  behavior; useful for route designs that degrade smoothly when GPU resident
  predicate-index selectivity or queue pressure estimates are wrong.
- `queued` — **Column Sketches: A Scan Accelerator for Rapid and Robust
  Predicate Evaluation**, Hentschel, Kester, and Idreos, SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3196894`
  Why: BinDex uses Column Sketches as a main robust-scan baseline; useful for
  comparing lossy compressed host/GPU predicate filters with binned bitmap
  refinement under tight memory budgets.
- `queued` — **Morton Filters: Faster, Space-Efficient Cuckoo Filters via
  Biasing, Compression, and Decoupled Logical Sparsity**, Breslow and Jayasena,
  PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p1041-breslow.pdf`
  Why: Performance-Optimal Filtering compares against Morton filters as a
  SIMD-friendly Cuckoo-filter variant; useful for resident set summaries where
  delete support or lower false positives may justify more CPU/GPU lookup
  work.
- `queued` — **Monkey: Optimal Navigable Key-Value Store**, Dayan,
  Athanassoulis, and Idreos, SIGMOD 2017.
  URL: `https://doi.org/10.1145/3035918.3064054`
  PDF:
  `https://stratos.seas.harvard.edu/files/stratos/files/monkeykeyvaluestore.pdf`
  Why: Performance-Optimal Filtering cites Monkey for level-specific Bloom
  filter tuning in LSM-style storage; useful if GPU DB adopts log-structured
  cold or warm segments with tier-specific false-positive budgets.
- `queued` — **Access Path Selection in Main-Memory Optimized Data Systems:
  Should I Scan or Should I Probe?**, Kester, Athanassoulis, and Idreos,
  SIGMOD 2017.
  URL: `https://doi.org/10.1145/3035918.3064049`
  Why: BinDex cites this as the selectivity/concurrency access-path motivation;
  useful for turning resident scan, predicate-index, and CPU fallback choices
  into a measured planner route boundary instead of a fixed threshold.
- `queued` — **PA-Tree: Polled-Mode Asynchronous B+ Tree for NVMe**, Wang,
  Zhang, He, and Zhang, ICDE 2020.
  URL: `https://doi.org/10.1109/ICDE48307.2020.00055`
  Why: Haas and Leis contrast PA-Tree's SPDK/polled-mode B+-tree design with
  a full storage-engine path; useful for isolating whether GPU DB cold-tier
  point lookup benefits come from polling alone or from worker-integrated
  eviction, task scheduling, and partitioned I/O metadata.
- `queued` — **Append is Near: Log-based Data Management on ZNS SSDs**,
  Purandare, Wilcox, Litz, and Finkelstein, CIDR 2022.
  URL: `https://www.cidrdb.org/cidr2022/papers/p28-purandare.pdf`
  Why: Haas and Leis identify specialized SSD interfaces as a related path;
  useful for comparing conventional NVMe page ownership with zone-append
  layouts for WAL, cold partitions, and write-amplification control.
- `queued` — **Better database cost/performance via batched I/O on
  programmable SSD**, Do, Picoli, Lomet, and Bonnet, VLDB Journal 2021.
  URL: `https://doi.org/10.1007/s00778-020-00643-2`
  Why: Haas and Leis cite database/storage-device co-design; useful as a
  counterpoint to host-owned SPDK/io_uring paths when evaluating whether GPU DB
  should keep cold-tier batching in the engine or eventually use near-storage
  offload for log, scan, or filter work.
- `queued` — **Let's Talk About Storage & Recovery Methods for
  Non-Volatile Memory Database Systems**, Arulraj, Pavlo, and Dulloor,
  SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2749441`
  Why: WBL builds on this NVM storage/recovery survey; useful for choosing
  which GPU DB tiers should use byte-addressable persistence, WAL, checkpoints,
  shadowing, or explicit recovery rebuilds.
- `queued` — **FOEDUS: OLTP Engine for a Thousand Cores and NVRAM**, Kimura,
  SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2746480`
  Why: WBL contrasts FOEDUS's dual-page hybrid DRAM/NVM design; useful for
  comparing volatile mutable pages plus durable snapshots with GPU DB's CPU
  truth, GPU-resident snapshots, and future persistent tiers.
- `queued` — **Persistent B+-Trees in Non-Volatile Main Memory**, Chen and Jin,
  PVLDB 2015.
  URL: `https://doi.org/10.14778/2735479.2735489`
  Why: WBL's Peloton implementation stores indexes as persistent B+trees;
  useful for deciding whether future host/persistent indexes should be durable
  performance state or rebuilt from WAL and route metadata.
- `reviewed` — **Bf-Tree: A Modern Read-Write-Optimized Concurrent
  Larger-Than-Memory Range Index**, Hao and Chandramouli, PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p3442-hao.pdf`
  DOI: `https://doi.org/10.14778/3681954.3682012`
  Why: follow-up by the Three-Tree first author on larger-than-memory range
  indexing. Marked reviewed because duplicate journal entries already exist;
  useful for comparing page-granular tiering with an index design that
  explicitly balances reads, writes, concurrency, and cold storage.
- `queued` — **TreeLine: An Update-in-Place Key-Value Store for Modern
  Storage**, Yu et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol16/p99-yu.pdf`
  DOI: `https://doi.org/10.14778/3561261.3561267`
  Why: CTR and Bf-Tree both point toward recoverable hot/cold storage state;
  TreeLine is a modern storage follow-up for insert forecasting, record
  caching, and update-in-place behavior on fast storage.
- `reviewed` — **WALTZ: Leveraging Zone Append to Tighten the Tail Latency of
  LSM Tree on ZNS SSD**, Lee, Kim, and Lee, PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p2884-lee.pdf`
  DOI: `https://doi.org/10.14778/3611479.3611495`
  Why: modern log-structured storage follow-up for separating write
  amplification, tail latency, and cold-tier device semantics from
  page-oriented WAL/checkpoint assumptions.
- `queued` — **Compaction-aware zone allocation for LSM based key-value store
  on ZNS SSDs**, Lee et al., HotStorage 2022.
  URL: `https://doi.org/10.1145/3538643.3539745`
  Why: WALTZ cites CAZA as a WAF-oriented ZNS allocation design based on
  adjacent-level key-range overlap; useful for separating tail-latency
  reservation from cold-tier write-amplification control.
- `queued` — **Lifetime-leveling LSM-tree compaction for ZNS SSD**, Jung and
  Shin, HotStorage 2022.
  URL: `https://doi.org/10.1145/3538643.3539746`
  Why: WALTZ cites LL-compaction as an SST-splitting approach for reducing
  ZNS write amplification; useful for GPU DB cold-tier segment compaction and
  refresh rewrite policy.
- `queued` — **CruiseDB: An LSM-Tree Key-Value Store with Both Better Tail
  Throughput and Tail Latency**, Liang and Chai, ICDE 2021.
  URL: `https://doi.org/10.1109/ICDE51399.2021.00095`
  Why: WALTZ contrasts CruiseDB's write-admission and tail-latency controls
  with zone-append WAL design; useful for deciding whether GPU DB should shape
  writes at admission, WAL append, compaction, or all three.
- `reviewed` — **Tiered-Indexing: Optimizing Access Methods for Skew**, Zhou,
  Hao, Yu, and Stonebraker, VLDB Journal 2025.
  URL: `https://doi.org/10.1007/s00778-025-00928-6`
  Code: `https://github.com/zxjcarrot/2-Tree`
  Why: follow-up to Two-Tree/Three-Tree on access-method tiering under skew;
  relevant to GPU DB hot/cold resident index placement and selective
  promotion of upper, lower, or leaf-heavy structures.
- `reviewed` — **An Examination of CXL Memory Use Cases for In-Memory Database
  Management Systems using SAP HANA**, Ahn et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p3827-ahn.pdf`
  DOI: `https://doi.org/10.14778/3685800.3685809`
  Why: modern empirical CXL-memory placement study for a commercial in-memory
  DBMS; selected because the recent synthesis called for more tier and
  placement work after several transaction-publication reviews.
- `reviewed` — **CXL Memory Performance for In-Memory Data Processing**,
  Weisgut, Ritter, Tozun, Benson, and Rabl, PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p3119-weisgut.pdf`
  Why: newer CXL memory benchmark paper for in-memory data processing; useful
  follow-up for separating raw CXL latency/bandwidth effects from
  DBMS-object-placement effects.
- `reviewed` — **Database Kernels: Seamless Integration of Database Systems and
  Fast Storage via CXL**, Lee, Lerner, Bonnet, and Cudre-Mauroux, CIDR 2024.
  URL:
  `https://www.vldb.org/cidrdb/2024/database-kernels-seamless-integration-of-database-systems-and-fast-storage-via-cxl.html`
  Why: CXL storage/database co-design paper cited by the SAP HANA CXL work;
  useful for comparing simple CXL memory expansion with DBMS-owned storage
  functions, logging, and cold-tier pushdown. Marked reviewed because the
  same paper already has a journal entry from the earlier queued copy.
- `reviewed` — **NeoMem: Hardware/Software Co-Design for CXL-Native Memory
  Tiering**, Zhong et al., arXiv 2024.
  URL: `https://arxiv.org/abs/2403.18702`
  Why: CXL-native tiering design; useful for contrasting OS/hardware-managed
  placement with GPU DB's explicit object-family placement and route telemetry.
- `reviewed` — **MEMTIS: Efficient Memory Tiering with Dynamic Page
  Classification and Page Size Determination**, Lee et al., SOSP 2023.
  URL:
  `https://cosmoss-jigu.github.io/pages/pubs/memtis-lee-sosp23.pdf`
  Why: NeoMem compares against Memtis as a distribution-aware software tiering
  baseline; useful for separating hardware-side access telemetry from
  application-visible page-size and hot-set classification policy.
- `reviewed` — **HybridTier: an Adaptive and Lightweight CXL-Memory Tiering
  System**, Liu et al., ASPLOS 2025 / arXiv 2023.
  URL: `https://arxiv.org/abs/2312.04789`
  DOI: `https://doi.org/10.1145/3676642.3736119`
  Code: `https://github.com/kevins981/hybridtier-asplos25-artifact`
  Why: NeoMem frames frequency-sensitive page promotion as the central CXL
  tiering challenge; useful for comparing software frequency estimation with
  device-side hot-page telemetry before relying on OS-transparent placement.
- `reviewed` — **MTM: Rethinking Memory Profiling and Migration for Multi-Tiered
  Large Memory**, Ren et al., EuroSys 2024.
  URL: `https://doi.org/10.1145/3627703.3650075`
  PDF: `https://pasalabs.org/papers/2024/Eurosys24_M3_Camera_Ready.pdf`
  Why: HybridTier omits end-to-end comparison because source was unavailable at
  submission; MTM is a modern multi-tier profiling/migration baseline for
  comparing DB-owned placement against application-transparent page movement.
  Journal entry added 2026-06-06; DOI corrected during review.
- `queued` — **Cori: Dancing to the Right Beat of Periodic Data Movements over
  Hybrid Memory Systems**, Doudali, Zahka, and Gavrilovska, IPDPS 2021.
  URL: `https://doi.org/10.1109/IPDPS49936.2021.00043`
  Why: MTM cites Cori as related periodic data-movement work; useful for
  comparing phase-aware movement cadence and migration timing against GPU DB's
  tier-control loops and refresh/demotion intervals.
- `queued` — **Kleio: A Hybrid Memory Page Scheduler with Machine
  Intelligence**, Doudali, Blagodurov, Vishnu, Gurumurthi, and Gavrilovska,
  HPDC 2019.
  URL: `https://doi.org/10.1145/3307681.3325408`
  Why: MTM cites ML-assisted hybrid-memory scheduling; useful contrast before
  GPU DB uses learned placement or route-regret signals for HBM/DRAM/future-tier
  object migration.
- `reviewed` — **FlexMem: Adaptive Page Profiling and Migration for Tiered
  Memory**, Xu et al., USENIX ATC 2024.
  URL: `https://www.usenix.org/conference/atc24/presentation/xu-dong`
  Why: HybridTier cites FlexMem as a contemporary frequency-based tiering
  system; useful for comparing adaptive profiling overhead with GPU DB's
  route-object heat telemetry.
- `reviewed` — **UniMem: Redesigning Disaggregated Memory within A Unified
  Local-Remote Memory Hierarchy**, Zhong et al., USENIX ATC 2024.
  URL: `https://www.usenix.org/conference/atc24/presentation/zhong`
  PDF: `https://www.usenix.org/system/files/atc24-zhong.pdf`
  Why: FlexMem contrasts demand-triggered page demotion with systems that
  demote when a promotion is about to fail; UniMem is a modern disaggregated
  memory design for comparing hotness, fragmentation, and critical-path
  migration against GPU DB's explicit route-owned tier placement.
- `queued` — **Using Local Cache Coherence for Disaggregated Memory Systems**,
  Puddu et al., Operating Systems Review 2023.
  URL: `https://ivanpuddu.com/files/papers/kona_osr2023.pdf`
  Why: UniMem uses Kona as the cache-coherent disaggregated-memory baseline;
  reviewing Kona would expose the baseline fake-physical-memory indirection,
  accelerator-local cache design, and coherence assumptions that UniMem tries
  to replace.
- `reviewed` — **Tiered Memory Management: Access Latency is the Key!**,
  Vuppalapati and Agarwal, SOSP 2024.
  URL: `https://doi.org/10.1145/3694715.3695968`
  PDF: `https://www.cs.cornell.edu/~ragarwal/pubs/colloid.pdf`
  Why: HybridTier notes Colloid as complementary latency-balanced tiering;
  useful for comparing hotness-only placement with latency-balancing policy
  when CXL/far-memory paths have heterogeneous access costs. Journal entry
  added 2026-06-06; DOI corrected during review.
- `reviewed` — **Managing Memory Tiers with CXL in Virtualized Environments**,
  Zhong et al., OSDI 2024.
  URL: `https://www.usenix.org/conference/osdi24/presentation/zhong-yuhong`
  Why: HybridTier references Memstrata as a CXL tier manager for virtualized
  environments; useful for multi-tenant placement, isolation, and whether GPU
  DB can trust host/VM tiering for route-critical memory. Journal entry added
  2026-06-06 from the USENIX PDF.
- `reviewed` — **Understanding the Host Network**, Vuppalapati, Agarwal,
  Schuh, Kasikci, Krishnamurthy, and Agarwal, SIGCOMM 2024.
  URL: `https://doi.org/10.1145/3651890.3672271`
  PDF:
  `https://www.cs.cornell.edu/~saksham/assets/pdf/UnderstandingHostNetwork.pdf`
  Why: Colloid depends on host-network contention and CHA-level latency
  measurement; useful for understanding CPU/memory/peripheral interconnect
  contention before GPU DB treats HBM, PCIe/NVLink, CXL, NIC, storage, and
  host-memory movement as independent route resources. Journal entry added
  2026-06-07 from the ACM/author PDF.
- `queued` — **Host Congestion Control**, Agarwal, Krishnamurthy, and
  Agarwal, SIGCOMM 2023.
  URL: `https://doi.org/10.1145/3603269.3604878`
  PDF: `https://homes.cs.washington.edu/~arvind/papers/hcc.pdf`
  Why: Understanding the Host Network names hostCC as a direction for
  host-network resource allocation; useful for deciding whether GPU DB should
  expose host-interconnect pressure as local credits before NIC, storage, or
  GPU data movers saturate shared memory-controller domains.
- `queued` — **Hostping: Diagnosing Intra-host Network Bottlenecks in RDMA
  Servers**, Liu et al., NSDI 2023.
  URL: `https://www.usenix.org/conference/nsdi23/presentation/liu-kefei`
  PDF: `https://www.usenix.org/system/files/nsdi23-liu-kefei.pdf`
  Why: Understanding the Host Network compares against intra-host bottleneck
  diagnosis work; useful for a GPU DB observability lane that can distinguish
  NIC/RDMA, PCIe, memory-controller, and CPU-copy bottlenecks under 1M-session
  gateway load.
- `queued` — **IDIO: Network-Driven, Inbound Network Data Orchestration on
  Server Processors**, Alian et al., MICRO 2022.
  URL: `https://doi.org/10.1109/MICRO56248.2022.00042`
  Metadata:
  `https://par.nsf.gov/biblio/10395078-idio-network-driven-inbound-network-data-orchestration-server-processors`
  Why: Understanding the Host Network cites dynamic direct-cache-access
  mechanisms as a future datapath; useful for comparing NIC/storage inbound
  data placement with GPU DB pinned buffers, CPU response encoding, and
  GPU-bound staging.
- `reviewed` — **Fetch Me If You Can: Evaluating CPU Cache Prefetching and Its
  Reliability on High Latency Memory**, Mahling, Weisgut, and Rabl, DaMoN 2025.
  URL: `https://doi.org/10.1145/3736227.3736231`
  PDF:
  `https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/rabl/publications/2025/Mahling-DaMoN25-Prefetching.pdf`
  Code: `https://github.com/hpides/prefetching`
  Why: CXL Memory Performance points to software prefetching for random
  high-latency memory accesses; useful for deciding whether cold host indexes,
  far-memory B+trees, and route metadata can hide CXL/future-tier latency.
- `reviewed` — **How to Be Fast and Not Furious: Looking Under the Hood of CPU
  Cache Prefetching**, Kuhn, Muhlig, and Teubner, DaMoN 2024.
  URL: `https://doi.org/10.1145/3662010.3663451`
  PDF:
  `https://dbis.cs.tu-dortmund.de/storages/dbis-cs/r/papers/2024/sw-prefetching-survey/sw-prefetching.pdf`
  Why: Fetch Me If You Can builds on this CPU-prefetch characterization work;
  useful for turning coroutine, AMAC, and state-machine prefetching into
  hardware-calibrated route policies instead of hard-coded prefetch distances.
- `queued` — **APT-GET: Profile-guided Timely Software Prefetching**, Jamilan
  et al., EuroSys 2022.
  URL: `https://doi.org/10.1145/3492321.3519583`
  Why: How to Be Fast and Not Furious points to profile-guided prefetch timing
  as a way to adapt prefetch distance to workload and hardware; useful for
  calibrating CPU metadata, host-index, and future-tier lookup lanes.
- `queued` — **FetchBench: Systematic Identification and Characterization of
  Proprietary Prefetchers**, Schluter et al., CCS 2023.
  URL: `https://doi.org/10.1145/3576915.3623124`
  Why: How to Be Fast and Not Furious depends on undocumented hardware
  prefetcher behavior; useful for deciding which route-prefetch assumptions
  require local hardware characterization before production use.
- `reviewed` — **Databases in the Era of Memory-Centric Computing**, Chronis
  et al., CIDR 2025.
  URL:
  `https://www.vldb.org/cidrdb/2025/databases-in-the-era-of-memory-centric-computing.html`
  PDF: `https://www.vldb.org/cidrdb/papers/2025/p6-chronis.pdf`
  Why: CXL Memory Performance cites memory-centric database designs as a
  broader architectural direction; useful for comparing GPU DB's explicit
  owner/tier model with memory-centric pooled designs and database operators.
- `reviewed` — **A Case Against CXL Memory Pooling**, Levis, Lin, and Tai,
  HotNets 2023.
  URL: `https://doi.org/10.1145/3626111.3628195`
  Why: memory-centric database designs cite it as the main cautionary source
  on CXL pooling; useful for stress-testing GPU DB's future memory-pool
  assumptions against networking, failure, and deployment costs.
- `reviewed` — **Demystifying CXL Memory with Genuine CXL-Ready Systems and
  Devices**, Sun et al., MICRO 2023.
  URL: `https://doi.org/10.1145/3613424.3614256`
  arXiv: `https://arxiv.org/abs/2303.15375`
  Why: the HotNets CXL-pooling critique relies on this real-hardware CXL
  latency and bandwidth evidence; useful for replacing abstract CXL tier
  assumptions with measured load/store, copy, random-access, and NUMA-like
  behavior.
- `reviewed` — **TMO: Transparent Memory Offloading in Datacenters**, Weiner et
  al., ASPLOS 2022.
  URL: `https://doi.org/10.1145/3503222.3507731`
  PDF: `https://www.cs.cmu.edu/~dskarlat/publications/tmo_asplos22.pdf`
  Why: Demystifying CXL compares against transparent page placement and
  offloading policy; TMO is a production datacenter memory-offload baseline
  for deciding which GPU DB objects can be transparently demoted and which
  require explicit route-owned placement.
- `reviewed` — **HeMem: Scalable Tiered Memory Management for Big Data
  Applications and Real NVM**, Raybuck et al., SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483550`
  PDF: `https://www.cs.utexas.edu/~witchel/380L/papers/raybuck21sosp.pdf`
  Why: Demystifying CXL cites HeMem as a tiered-memory management baseline;
  useful for comparing hardware-event sampling and user-level tier policy
  against DB-owned placement of hot metadata, cold segments, and old
  snapshots.
- `queued` — **MaxMem: Colocation and Performance for Big Data Applications
  on Tiered Main Memory Servers**, Raybuck et al., arXiv 2023.
  URL: `https://arxiv.org/abs/2312.00647`
  Why: HeMem follow-up that extends user-space tiered-memory management to
  multi-application colocation and QoS; useful for comparing per-process heat
  gradients and fast-tier miss ratios with GPU DB admission when multiple
  tenants, retained snapshots, and warm segments contend for future tiers.
- `queued` — **Nimble Page Management for Tiered Memory Systems**, Yan et al.,
  ASPLOS 2019.
  URL: `https://doi.org/10.1145/3297858.3304024`
  Author page:
  `https://normal.zone/publications/2019-04-15-ASPLOS-2019/`
  Why: Demystifying CXL shows page migration can harm latency-sensitive
  workloads; Nimble is a primary OS page-migration mechanism for measuring
  migration throughput, migration interference, and whether GPU DB should avoid
  opaque page movement on short read paths.
- `queued` — **Thermostat: Application-Transparent Page Management for
  Two-Tiered Main Memory**, Agarwal and Wenisch, ASPLOS 2017.
  URL: `https://doi.org/10.1145/3037697.3037706`
  Why: TMO cites Thermostat as an application-transparent hot/cold page
  management baseline; useful for comparing DB-owned tier policy with
  transparent page migration when GPU DB adds CXL, NVM, or far-memory tiers.
- `queued` — **Effectively Prefetching Remote Memory with Leap**, Al Maruf and
  Chowdhury, USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/maruf`
  Why: TMO cites Leap as a remote-memory prefetching design; useful for
  evaluating whether future GPU DB cold-tier or far-memory lookups can hide
  migration latency without polluting hot DRAM/HBM placement.
- `queued` — **Dremel: A Decade of Interactive SQL Analysis at Web Scale**,
  Melnik et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p3461-melnik.pdf`
  DOI: `https://doi.org/10.14778/3415478.3415568`
  Why: the memory-centric CIDR paper uses BigQuery's shuffle service as an
  existence proof for disaggregated intermediate memory; useful for comparing
  GPU DB response rings, intermediate result placement, and checkpointed
  distributed query stages.
- `reviewed` — **Don't Look Back, Look into the Future: Prescient Data
  Partitioning and Migration for Deterministic Database Systems**, Lin et al.,
  SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3452827`
  PDF: `https://www.cs.nthu.edu.tw/~shwu/pubs/shwu-sigmod-21.pdf`
  Why: T-Part's author/project line evolves dependency-aware transaction
  placement into future-window routing and migration; useful for comparing
  active-window route certificates with data movement and owner-boundary
  changes under shifting hot spots.
- `reviewed` — **MgCrab: Transaction Crabbing for Live Migration in
  Deterministic Database Systems**, Lin et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p597-lin.pdf`
  Why: direct follow-up from the T-Part/ElaSQL line on live migration during
  deterministic execution; useful for deciding how GPU DB can move hot
  partitions, resident segments, or owner assignments without stopping
  admitted transaction batches.
- `queued` — **Squall: Fine-Grained Live Reconfiguration for Partitioned
  Main Memory Databases**, Elmore et al., SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2749446`
  Why: cited by LADS as related elastic partitioning work; useful for comparing
  dependency-graph batch routing with online movement of hot partitions,
  owner domains, and resident segment assignments.
- `queued` — **Foedus: OLTP Engine for a Thousand Cores and NVRAM**, Kimura,
  SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2746480`
  Why: LADS compares against FOEDUS-style multicore/NVRAM transaction
  engines; useful for storage/runtime design across many cores, logging, and
  future non-volatile or far-memory tiers.
- `queued` — **PIAS: Practical Information-Agnostic Flow Scheduling for
  Commodity Data Center Networks**, Bai et al., NSDI 2015.
  URL:
  `https://www.usenix.org/conference/nsdi15/technical-sessions/presentation/bai`
  Why: Homa contrasts sender-side multilevel feedback priority assignment
  against receiver-driven SRPT approximation; useful for GPU DB request and
  response scheduling when exact route size is unknown at admission time.
- `reviewed` — **NDP: Re-architecting Datacenter Networks and Stacks for Low
  Latency**, Handley et al., SIGCOMM 2017.
  URL: `https://doi.org/10.1145/3098822.3098825`
  Why: Homa compares against NDP's receiver-side pulling and bounded queues;
  useful for evaluating how much GPU DB should trade bandwidth utilization for
  low queueing delay at network, response-ring, and owner-ingress boundaries.
- `queued` — **Presto: Edge-based Load Balancing for Fast Datacenter
  Networks**, He et al., SIGCOMM 2015.
  URL: `https://doi.org/10.1145/2785956.2787507`
  Why: NDP contrasts packet spraying and load-balancing schemes such as Presto;
  useful for deciding whether GPU DB response/request traffic should rely on
  route-level spreading, endpoint pacing, or DB-owned admission when short
  requests collide on shared queues.
- `reviewed` — **HyBench: A New Benchmark for HTAP Databases**, Zhang et al.,
  PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p939-zhang.pdf`
  DOI: `https://doi.org/10.14778/3641204.3641213`
  Why: selected after the recent synthesis called for HTAP freshness and route
  evaluation; useful for measuring mixed write/read pressure with explicit
  freshness rather than treating stale analytical speed as sufficient.
- `reviewed` — **F1 Lightning: HTAP as a Service**, Yang et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p3313-yang.pdf`
  DOI: `https://doi.org/10.14778/3415478.3415553`
  Why: HyBench compares HTAP systems and motivates real-time analytical
  freshness; F1 Lightning is a production HTAP service paper useful for
  contrasting serving-time freshness, ingestion, and resource isolation.
- `reviewed` — **TiDB: A Raft-based HTAP Database**, Huang et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p3072-huang.pdf`
  DOI: `https://doi.org/10.14778/3415478.3415535`
  Why: HyBench evaluates HTAP tradeoffs across systems; TiDB/TiFlash provides
  a primary-source HTAP design with MVCC, Raft replication, and analytical
  replicas to compare against retained GPU snapshots.
- `queued` — **Oracle Database In-Memory: A dual format in-memory database**,
  Lahiri et al., ICDE 2015.
  URL: `https://doi.org/10.1109/ICDE.2015.7113300`
  Why: TiDB contrasts dual-format in-memory replicas with Raft learners;
  useful for comparing transaction-owned row truth plus queryable column
  acceleration when the analytical copy is updated inside the primary DBMS.
- `reviewed` — **Real-Time Analytical Processing with SQL Server**, Larson et
  al., PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol8/p1740-Larson.pdf`
  DOI: `https://doi.org/10.14778/2824032.2824071`
  Why: TiDB cites SQL Server's integrated Hekaton/Apollo HTAP path; useful for
  comparing migration from hot transactional rows into compressed columnar
  structures against GPU DB's resident refresh and safe-generation policy.
- `reviewed` — **SLOG: Serializable, Low-latency, Geo-replicated
  Transactions**, Ren, Li, and Abadi, PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p1747-ren.pdf`
  DOI: `https://doi.org/10.14778/3342263.3342647`
  Why: Hermes builds on deterministic execution and partition/locality
  assumptions; SLOG is a follow-up line for locality-aware routing under
  strict serializability and may inform owner-placement and route-freshness
  policy when GPU DB partitions or replicas become geographically or
  tier-wise distributed.
- `queued` — **Rocksteady: Fast Migration for Low-latency In-memory
  Storage**, Kulkarni et al., SOSP 2017.
  URL: `https://doi.org/10.1145/3132747.3132784`
  PDF: `https://chinkulkarni.github.io/public/rocksteady.pdf`
  Why: MgCrab contrasts transactional deterministic migration with
  low-latency in-memory key-value migration; useful for comparing early
  ownership transfer, workload-skew-aware movement, and adaptive background
  migration against GPU DB resident segment and owner-domain movement.
- `reviewed` — **When Database Meets New Storage Devices: Understanding and
  Exposing Performance Mismatches via Configurations**, He et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p1712-he.pdf`
  DOI: `https://doi.org/10.14778/3587136.3587145`
  Artifact: `https://github.com/TimHe95/S3M`
  Why: selected after recent synthesis called for storage-device scheduling
  work; exposes concrete IO-size, IO-parallelism, and sequentiality mismatches
  when DBMS storage routes assume faster media automatically helps.
- `reviewed` — **Rearchitecting Linux Storage Stack for microsecond Latency and
  High Throughput**, Hwang, Vuppalapati, Peter, and Agarwal, OSDI 2021.
  URL: `https://www.usenix.org/conference/osdi21/presentation/hwang`
  Why: cited by the PVLDB 2023 storage-device mismatch paper as a
  device-sensitive layer direction; useful for comparing DB-owned cold-tier IO
  owners with OS-level request switching and NVMe queue dispatch.
- `queued` — **TCP ≈ RDMA: CPU-efficient Remote Storage Access with i10**,
  Hwang, Cai, Tang, and Agarwal, NSDI 2020.
  URL: `https://www.usenix.org/conference/nsdi20/presentation/hwang`
  Why: blk-switch uses i10 as the Linux remote-storage baseline; useful for
  separating CPU-efficient NVMe-over-network access from the later
  latency/throughput isolation mechanisms in blk-switch.
- `queued` — **sRoute: Treating the Storage Stack Like a Network**, Thereska
  et al., FAST 2016.
  URL: `https://www.usenix.org/conference/fast16/technical-sessions/presentation/thereska`
  Why: blk-switch's related work names sRoute as a policy-based storage-stack
  design; useful for comparing network-like storage routing policy with
  GPU DB's explicit cold-tier IO owners and response-ring scheduling.
- `reviewed` — **Write Dependency Disentanglement with Horae**, Liao, Lu, Xu,
  and Shu, OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/liao`
  Why: cited by the PVLDB 2023 storage-device mismatch paper as ordered async
  IO work; useful for deciding how WAL-before-visibility can coexist with more
  parallel durable writes.
- `queued` — **Crash Consistent Non-Volatile Memory Express**, Liao, Lu, Yang,
  and Shu, SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483555`
  Why: storage-device mismatch work points to crash-consistent NVMe paths;
  useful for separating durable ordering constraints from unnecessary
  synchronous queue-depth-1 IO in future WAL/checkpoint routes.
- `reviewed` — **Barrier-Enabled IO Stack for Flash Storage**, Won et al.,
  FAST 2018.
  URL: `https://www.usenix.org/conference/fast18/presentation/won`
  PDF: `https://www.usenix.org/system/files/conference/fast18/fast18-won.pdf`
  Why: Horae contrasts its multi-queue/multi-device control/data split with
  BarrierFS; useful for comparing fbarrier-style ordering without durability
  against GPU DB WAL, checkpoint, and cold-tier publication boundaries.
- `queued` — **Application Crash Consistency and Performance with CCFS**,
  Pillai et al., FAST 2017.
  URL: `https://www.usenix.org/conference/fast17/technical-sessions/presentation/pillai`
  Why: BarrierFS contrasts its block/device-level ordering with CCFS-style
  application crash-consistency mechanisms; useful for deciding whether GPU DB
  should expose route-level ordering groups above WAL/checkpoint IO.
- `queued` — **SpanFS: A Scalable File System on Fast Storage Devices**,
  Kang et al., USENIX ATC 2015.
  URL: `https://www.usenix.org/conference/atc15/technical-session/presentation/kang`
  Why: BarrierFS names SpanFS as a multi-transaction journaling approach on
  fast storage; useful for comparing partitioned commit lanes with GPU DB
  owner-domain WAL and checkpoint queues.
- `reviewed` — **Generic Version Control: Configurable Versioning for
  Application-Specific Requirements**, Yilmaz and Dittrich, CIDR 2025.
  URL:
  `https://mail.vldb.org/cidrdb/2025/generic-version-control-configurable-versioning-for-application-specific-requirements.html`
  PDF: `https://mail.vldb.org/cidrdb/papers/2025/p24-yilmaz.pdf`
  Why: selected after the recent synthesis called for more MVCC/visibility
  and transaction-route proof work; proposes explicit validation-time
  conflict detection and reconciliation functions that may reduce false
  aborts without moving semantic repair back to application round trips.
- `queued` — **TARDiS: A Branch-and-Merge Approach To Weak Consistency**,
  Crooks et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915213`
  Why: GenericVC cites TARDiS as branch/merge related work; useful for
  contrasting database-layer reconciliation with application-visible
  branch-and-merge semantics under weaker consistency.
- `reviewed` — **OrpheusDB: Bolt-on Versioning for Relational Databases**,
  Huang et al., PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p1130-huang.pdf`
  DOI: `https://doi.org/10.14778/3115404.3115417`
  Why: GenericVC contrasts bolt-on relational versioning with a unified
  transaction/version layer; useful for deciding whether GPU DB old-version
  lineage should be a first-class MVCC structure or an external branch table.
- `reviewed` — **MindPalace: Version Reconciliation for Collaborative
  Databases**, Ranjan, Shang, Krishnan, and Elmore, SoCC 2021.
  URL: `https://doi.org/10.1145/3472883.3486980`
  arXiv: `https://arxiv.org/abs/2110.01778`
  Why: GenericVC cites MindPalace's auto-mergeability as related
  reconciliation work; useful for comparing semantic merge rules against
  strict SQL validation and retained-snapshot correctness.
- `reviewed` — **Skeena: Efficient and Consistent Cross-Engine Transactions**,
  Zhang et al., arXiv 2021.
  URL: `https://arxiv.org/abs/2108.00632`
  Why: MindPalace's branch reconciliation raises the broader question of how
  version and snapshot metadata crosses engines; Skeena is a modern
  cross-engine transaction source for comparing lightweight snapshot tracking
  and atomic commit across CPU, GPU, and cold-tier execution engines.
- `reviewed` — **Industrial-Strength OLTP Using Main Memory and Many Cores**,
  Avni et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p3099-avni.pdf`
  DOI: `https://doi.org/10.14778/3415478.3415537`
  Why: Skeena cites this main-memory engine line as a production fast-engine
  target; useful for comparing cross-engine snapshot coordination with
  many-core OLTP ownership, logging, and memory-resident transaction paths.
- `queued` — **Harmony: A Heterogeneous Database System Built for Hybrid
  Transactional and Analytical Processing**, Psaroudakis et al., PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p1161-psaroudakis.pdf`
  DOI: `https://doi.org/10.14778/2994509.2994530`
  Why: Skeena cites SAP HANA's heterogeneous-engine direction; useful for
  contrasting cross-engine OLTP correctness with HTAP table placement,
  analytical freshness, and transaction-aware engine routing.
- `reviewed` — **Low-Overhead Asynchronous Checkpointing in Main-Memory Database
  Systems**, Ren, Diamond, Abadi, and Thomson, SIGMOD 2016.
  URL: `https://www.cs.yale.edu/homes/dna/papers/fast-checkpoint-sigmod16.pdf`
  DOI: `https://doi.org/10.1145/2882903.2915966`
  Why: MOT reuses this checkpointing line; useful for designing asynchronous
  CPU truth checkpoints that do not stop GPU resident snapshot refresh,
  invalidation, or WAL replay.
- `reviewed` — **Index Checkpoints for Instant Recovery in In-Memory Database
  Systems**, Lee, Xie, Ma, and Chen, PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p1671-lee.pdf`
  DOI: `https://doi.org/10.14778/3529337.3529350`
  Why: follow-up recovery paper that makes rebuildable in-memory indexes part
  of the checkpoint/recovery frontier; useful for deciding when GPU DB CPU
  indexes, route metadata, or resident-index acceleration state should be
  checkpointed instead of rebuilt after WAL replay.
- `queued` — **A Scalable Linearizable Multi-Index Table**, Sheffi,
  Golan-Gueta, and Petrank, ICDCS 2018.
  URL: `https://doi.org/10.1109/ICDCS.2018.00029`
  Why: MOT contrasts its industrial optimistic multi-index insert protocol with
  this multi-index table work; useful for deciding which index-update
  atomicity guarantees must be database-transactional versus data-structure
  local.
- `reviewed` — **Native Store Extension for SAP HANA**, Sherkat et al.,
  PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p2047-sherkat.pdf`
  Why: MOT names memory capacity and tiering as future work; HANA NSE is a
  primary-source warm-tier design for comparing in-memory OLTP acceleration
  with disk-backed warm data placement.
- `queued` — **GalOP: Towards a GPU-Accelerated OLTP DBMS**, Boeschen and
  Binnig, DaMoN/SIGMOD 2021.
  URL: `https://doi.org/10.1145/3465998.3466007`
  Metadata: `https://www.dfki.de/en/web/research/projects-and-publications/publication/14419`
  Why: precursor to GaccO; useful for separating the early GPU OLTP execution
  argument from the later CPU/GPU co-execution and update-propagation design.
- `reviewed` — **Harnessing GPU Power for Enhanced OLTP: A Study in Concurrency
  Control Schemes**, arXiv 2024.
  URL: `https://arxiv.org/abs/2406.10158`
  Why: modern GPU OLTP concurrency-control comparison discovered while
  reviewing GaccO; useful for comparing GPU-friendly locking/OCC/MVCC choices
  before adopting large homogeneous transaction batches.
- `reviewed` — **Transactional Data Structure Libraries**, Spiegelman et al.,
  PLDI 2016.
  URL: `https://doi.org/10.1145/2908080.2908112`
  Author PDF:
  `https://people.csail.mit.edu/idish/ftp/TransactionalLibrariesPLDI16.pdf`
  Why: DRP builds on STO-style transactional objects; useful for deciding
  whether GPU DB route metadata, resident indexes, and queued intentions should
  expose data-structure-specific transaction hooks instead of generic tuple
  read/write validation only.
- `queued` — **DB2 with BLU Acceleration: So Much More Than Just a Column
  Store**, Raman et al., ICDE 2015.
  URL: `https://doi.org/10.1109/ICDE.2015.7113303`
  Why: Page As You Go contrasts HANA page-loadable columns with DB2 BLU's
  page-backed compressed column groups and scan prefetch; useful for comparing
  warm-tier page/cache policy and columnar metadata placement before picking a
  GPU DB warm-column format.
- `reviewed` — **Improving Optimistic Concurrency Control through Transaction
  Batching and Operation Reordering**, Ding, Kot, and Gehrke, PVLDB 2018.
  URL: `https://doi.org/10.14778/3282495.3282502`
  PDF: `https://www.vldb.org/pvldb/vol12/p169-ding.pdf`
  Why: PLOR contrasts batching/reordering as a tail-latency-aware OCC
  direction; useful for deciding whether GPU DB hot-write admission should
  reorder compatible operations inside bounded latency ceilings instead of
  relying only on timestamp priority.
- `reviewed` — **High-Performance ACID via Modular Concurrency Control**, Xie
  et al., SOSP 2015.
  URL: `https://doi.org/10.1145/2815400.2815430`
  PDF:
  `https://sigops.org/s/conferences/sosp/2015/current/2015-Monterey/263-xie-online.pdf`
  Why: PLOR cites Callas-style modular concurrency control as a mixed-protocol
  alternative; useful for comparing per-route concurrency-control modules with
  GPU DB owner domains, retained reads, and hot-write fallback lanes.
- `reviewed` — **In-Network Support for Transaction Triaging**, Lerner et al.,
  PVLDB 2021.
  URL: `https://vldb.org/pvldb/vol14/p1626-lerner.pdf`
  Why: modern follow-up for transaction admission before full execution;
  useful for comparing engine-owned hot-key batching with earlier network or
  gateway triage of likely-conflicting requests.
- `queued` — **Infinite Resources for Optimistic Concurrency Control**,
  Jepsen et al., NetCompute 2018.
  URL: `https://doi.org/10.1145/3229591.3229597`
  Why: cited by Transaction Triaging as in-network transaction execution work;
  useful for comparing switch/NIC-level conflict prefilters with GPU DB's
  owner-domain validation and hot-key admission queues.
- `queued` — **Eris: Coordination-Free Consistent Transactions Using
  In-Network Concurrency Control**, Li, Michael, and Ports, SOSP 2017.
  URL: `https://doi.org/10.1145/3132747.3132751`
  Why: Transaction Triaging contrasts portable stream shaping with
  concurrency-control-specific in-network ordering; useful for deciding whether
  any GPU DB gateway or NIC prefilter should ever participate in serial order.
- `queued` — **Harmonia: Near-Linear Scalability for Replicated Storage with
  in-Network Conflict Detection**, Zhu et al., PVLDB 2019.
  URL: `https://doi.org/10.14778/3368289.3368301`
  Why: Transaction Triaging cites Harmonia as in-network conflict detection for
  replicated storage; useful for separating cheap conflict hints from
  authoritative WAL/MVCC visibility decisions.
- `reviewed` — **Indexed Log File: Towards Main Memory Database Instant
  Recovery**, Magalhaes, Brayner, Monteiro, and Moraes, EDBT 2021.
  URL: `https://openproceedings.org/2021/conf/edbt/p110.pdf`
  DOI: `https://doi.org/10.5441/002/edbt.2021.34`
  Why: Index Checkpoints builds on indexed-log recovery; useful for comparing
  log-offset indexes, on-demand tuple restore, and whether GPU DB should make
  cold CPU tuple reconstruction lazy while keeping route metadata eager.
- `reviewed` — **FineLine: Log-structured Transactional Storage and
  Recovery**, Sauer, Graefe, and Harder, PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p2249-sauer.pdf`
  DOI: `https://doi.org/10.14778/3275366.3275373`
  Why: Indexed Log File contrasts with FineLine's indexed single-storage log;
  useful for deciding whether GPU DB cold CPU truth should remain WAL plus
  materialized state or move selected partitions toward log-structured,
  tuple-addressable recovery storage. Duplicate queue entry marked reviewed
  after the journal entry was added on 2026-06-06.
- `queued` — **Fast Failure Recovery for Main-Memory DBMSs on Multicores**,
  Wu, Guo, Chan, and Tan, SIGMOD 2017.
  URL: `https://doi.org/10.1145/3035918.3064011`
  Why: Index Checkpoints uses PACMAN-style parallel recovery as a baseline;
  useful for separating parallel log replay, index rebuild, and GPU resident
  acceleration rebuild in recovery benchmarks.
- `reviewed` — **A Comparative Study of Consistent Snapshot Algorithms for
  Main-Memory Database Systems**, Li et al., IEEE TKDE 2021.
  URL: `https://doi.org/10.1109/TKDE.2019.2930987`
  arXiv: `https://arxiv.org/abs/1810.04915`
  Why: Index Checkpoints relies on tuple snapshot consistency while accepting
  non-transaction-consistent index checkpoints; useful for choosing CPU truth
  checkpoint algorithms before deciding which derived indexes are persisted.
- `reviewed` — **Low-Overhead Asynchronous Checkpointing in Main-Memory
  Database Systems**, Ren, Diamond, Abadi, and Thomson, SIGMOD 2016.
  URL: `https://www.cs.yale.edu/homes/dna/papers/fast-checkpoint-sigmod16.pdf`
  DOI: `https://doi.org/10.1145/2882903.2915966`
  Why: Li et al. use CALC as the virtual-snapshot baseline; useful for
  comparing deferred consistent snapshots that avoid blocking active
  transactions with GPU DB retained snapshot publication and checkpoint
  boundaries. Duplicate queue entry corrected after the 2026-06-05 journal
  review already covered CALC.
- `queued` — **Data Blocks: Hybrid OLTP and OLAP on Compressed Storage using
  both Vectorization and Compilation**, Lang et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882925`
  Why: Li et al. cite it as an HTAP snapshot consumer; useful for comparing
  compressed read-optimized blocks with GPU DB resident column groups and
  CPU/GPU route selection.
- `reviewed` — **Chablis: Fast and General Transactions in Geo-Distributed
  Systems**, Eldeeb et al., CIDR 2024.
  URL:
  `https://vldb.org/cidrdb/2024/chablis-fast-and-general-transactions-in-geo-distributed-systems.html`
  PDF: `https://vldb.org/cidrdb/papers/2024/p4-eldeeb.pdf`
  Why: discovered while reviewing Callas and modern transaction-routing
  follow-ups; useful for comparing multi-versioned transactional routing,
  fast local read-write transactions, and lock-free snapshot reads against GPU
  DB route certificates and owner-local hot paths.
- `reviewed` — **Chardonnay: Fast and General Datacenter Transactions for
  On-Disk Databases**, Eldeeb et al., OSDI 2023.
  URL: `https://www.usenix.org/conference/osdi23/presentation/eldeeb`
  Why: Chablis builds on Chardonnay's local epoch service and lock-free
  snapshot-read protocol; useful for a deeper single-datacenter version of
  epoch publication, fast 2PC, and on-disk MVCC visibility without geo
  publisher latency.
- `reviewed` — **Cornus: Atomic Commit for a Cloud DBMS with Storage
  Disaggregation**, Guo et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol16/p379-guo.pdf`
  DOI: `https://doi.org/10.14778/3565816.3565837`
  Why: Chardonnay's transaction-state-store path cites Cornus-style atomic
  commit work; useful for comparing fast commit-state durability and
  coordinator failure handling when GPU DB separates WAL, owner ordering, and
  resident publication state.
- `reviewed` — **EasyCommit: A Non-blocking Two-phase Commit Protocol**,
  Gupta and Sadoghi, EDBT 2018.
  URL: `https://expolab.org/papers/easy-commit.pdf`
  DOI: `https://doi.org/10.5441/002/edbt.2018.15`
  Why: Cornus contrasts with non-blocking 2PC variants that add message
  redundancy rather than storage-layer CAS; useful for comparing failure
  progress, participant autonomy, and extra message cost in owner-domain
  commit protocols.
- `reviewed` — **Multi-version Range Concurrency Control in Deuteronomy**,
  Levandoski et al., PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol8/p2146-levandoski.pdf`
  DOI: `https://doi.org/10.14778/2831360.2831367`
  Why: Chardonnay relies on range leaders, range locking, and MVCC versions;
  this is a 2015-present primary source for range-level MVCC concurrency that
  may inform prefix scans, resident key-range certificates, and lock-free
  retained reads.
- `reviewed` — **High Performance Transactions in Deuteronomy**, Levandoski,
  Lomet, Sengupta, Stutsman, and Wang, CIDR 2015.
  URL: `https://www.cidrdb.org/cidr2015/Papers/CIDR15_Paper15.pdf`
  Project page:
  `https://www.microsoft.com/en-us/research/publication/high-performance-transactions-in-deuteronomy/`
  Why: direct source for Deuteronomy's high-throughput TC/DC split, latch-free
  MVCC table, redo-log version cache, epoch management, and fast commit path;
  useful if GPU DB adopts logical transaction ownership over separate storage
  and resident-index components.
- `reviewed` — **Deuteronomy 2.0: Record Caching and Latch Freedom**,
  Lomet, arXiv 2025.
  URL: `https://arxiv.org/abs/2504.14435`
  Why: modern follow-up from a Deuteronomy author that revisits
  record-granular caching, delta updating, and latch-free state publication;
  useful for refining GPU DB's CPU log/read cache, resident-delta policy, and
  route-metadata update path.
- `queued` — **Bwe-tree: An Evolution of Bw-tree on Fast Storage**, Wang
  et al., ICDE 2024.
  URL: `https://doi.org/10.1109/ICDE60146.2024.00405`
  Why: Deuteronomy 2.0 cites this modern Bw-tree evolution on fast storage;
  useful for comparing notice/delta publication with newer fast-storage index
  behavior before adopting latch-free resident-index rebuilds.
- `queued` — **VLL: A Lock Manager Redesign for Main Memory Database
  Systems**, Ren, Thomson, and Abadi, VLDB Journal 2015.
  URL: `https://www.cs.yale.edu/homes/dna/papers/vldbj-vll.pdf`
  DOI: `https://doi.org/10.1007/s00778-014-0377-7`
  Why: Deuteronomy contrasts VLL's logical range locking with MV timestamp
  ranges; useful for comparing lightweight pessimistic range protection
  against MVCC range certificates for retained prefix scans.
- `skipped` — **Releasing Locks as Early as You Can: Reducing Contention of
  Hotspots by Violating Two-Phase Locking**, Guo, Wu, Yan, and Yu,
  SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457294`
  PDF: `https://pages.cs.wisc.edu/~yxy/pubs/bamboo.pdf`
  Why: duplicate queue entry; Bamboo was already reviewed under the earlier
  Rebirth-Retire follow-up block.
- `queued` — **Adaptive Work Placement for Query Processing on Heterogeneous
  Computing Resources**, Karnagel, Habich, and Lehner, PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p733-karnagel.pdf`
  DOI: `https://doi.org/10.14778/3067421.3067427`
  Why: HorseQC cites adaptive heterogeneous work placement; useful for deciding
  when GPU DB should route a pipeline to CPU, GPU, or split execution based on
  transfer cost and operator support.
- `queued` — **Generating Custom Code for Efficient Query Execution on
  Heterogeneous Processors**, Bress et al., arXiv 2017.
  URL: `https://arxiv.org/abs/1709.00700`
  Why: HorseQC's CoGaDB integration reuses Hawk-style code generation; useful
  for route-specific CPU/GPU codegen without tying planner correctness to one
  hardware backend.
- `reviewed` — **Reactors: A Case for Predictable, Virtualized Actor Database
  Systems**, Shah and Vaz Salles, SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3183752`
  Why: Snapper uses actor-database work as its programming-model baseline;
  useful for comparing actor-style owner domains, predictable virtualized
  state, and transaction placement against GPU DB partition owners. Journal
  entry added 2026-06-06 from the ACM/author PDF; the queued DOI was
  corrected to `10.1145/3183713.3183752`.
- `queued` — **An Evaluation of Intra-Transaction Parallelism in
  Actor-Relational Database Systems**, Shah and Vaz Salles, arXiv 2022.
  URL: `https://arxiv.org/abs/2204.10743`
  Why: modern follow-up from the Reactors authors; useful for stress-testing
  whether actor-style intra-transaction parallelism really pays once
  transaction logic, communication cost, and contention are varied.
- `queued` — **Actor Database Systems: A Manifesto**, Shah and Vaz Salles,
  arXiv 2017.
  URL: `https://arxiv.org/abs/1707.06507`
  Why: Reactors references the broader actor-relational design space; useful
  background if GPU DB considers exposing owner-domain programming or route
  decomposition as a product-level abstraction rather than only an internal
  runtime implementation detail.
- `reviewed` — **Epoch-based Commit and Replication in Distributed OLTP
  Databases**, Lu, Yu, Cao, and Madden, PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p743-lu.pdf`
  DOI: `https://doi.org/10.14778/3446095.3446098`
  Why: Snapper cites epoch-style batching as a related deterministic commit
  mechanism; useful for comparing batch/epoch visibility publication with
  GPU DB mutation-owner generations and retained read frontiers.
- `queued` — **Minimizing Commit Latency of Transactions in Geo-Replicated
  Data Stores**, Nawab et al., SIGMOD 2015.
  URL: `https://doi.org/10.1145/2723372.2723729`
  Why: COCO cites Helios as a commit/consensus optimization for
  geo-replicated transactions; useful for comparing epoch-sized durability
  barriers with lower-latency replicated commit paths if GPU DB later spreads
  owner domains across nodes or regions.
- `reviewed` — **REPS: Recycled Entropy Packet Spraying for Adaptive Load
  Balancing and Failure Mitigation**, Bonato et al., arXiv 2024/EuroSys 2026.
  URL: `https://arxiv.org/abs/2407.21625`
  DOI: `https://doi.org/10.1145/3767295.3769320`
  Why: Ultra Ethernet names REPS as a path-aware entropy recycling strategy;
  useful for comparing self-clocking packet/path selection with GPU DB
  response-ring and multi-gateway load balancing. Reviewed on 2026-06-05.
- `reviewed` — **An Edge-Queued Datagram Service for All Datacenter Traffic**,
  Olteanu et al., NSDI 2022.
  URL: `https://www.usenix.org/conference/nsdi22/presentation/olteanu`
  PDF: `https://www.usenix.org/system/files/nsdi22-paper-olteanu.pdf`
  Why: REPS uses EQDS as a congestion-control/transport baseline; useful for
  comparing receiver-driven credits, edge queuing, and packet trimming with GPU
  DB's owner-ring admission, response backpressure, and future internal
  transport paths. Reviewed on 2026-06-06.
- `reviewed` — **Flowcut Switching: High-Performance Adaptive Routing with
  In-Order Delivery Guarantees**, Bonato et al., arXiv 2025.
  URL: `https://arxiv.org/abs/2506.21406`
  Why: Ultra Ethernet cites Flowcut as a newer routing direction; useful for
  deciding whether future GPU DB transport paths need packet spraying,
  flowlet switching, or in-order route classes for SQL responses.
- `reviewed` — **Network Load Balancing with In-network Reordering Support
  for RDMA**, Song et al., SIGCOMM 2023.
  URL: `https://www.comp.nus.edu.sg/~lijl/papers/conweave-sigcomm23.pdf`
  Why: Flowcut contrasts with ConWeave's switch-side in-network reordering;
  useful for comparing endpoint/NIC pause-and-reroute against fabric-buffered
  reordering when future GPU DB gateways need in-order high-throughput flows.
- `reviewed` — **When Cloud Storage Meets RDMA**, Gao et al., NSDI 2021.
  URL: `https://www.usenix.org/conference/nsdi21/presentation/gao`
  Why: ConWeave cites it as production-scale cloud-storage RDMA context;
  useful for understanding real RDMA service mixes, CPU offload, and
  storage/network tail-latency interactions before GPU DB considers
  RDMA-connected storage or gateway paths. Journal entry added 2026-06-06.
- `reviewed` — **Backpressure Flow Control**, Goyal et al., NSDI 2022.
  URL: `https://www.usenix.org/conference/nsdi22/presentation/goyal`
  Why: ConWeave discusses switch resource exhaustion and cites backpressure
  as related switch-flow-control work; useful for comparing explicit
  transport backpressure with GPU DB response-ring, reorder-buffer, and
  active-session admission limits. Reviewed on 2026-06-05.
- `reviewed` — **Data Center Ethernet and Remote Direct Memory Access:
  Issues at Hyperscale**, Hoefler et al., IEEE Computer 2023.
  URL: `https://doi.org/10.1109/MC.2023.3261184`
  Why: Ultra Ethernet motivates its design as an answer to RoCE/RDMA
  deployment pain; useful background for avoiding fragile lossless-network
  assumptions in GPU DB's future gateway and accelerator-fabric design.
  Reviewed on 2026-06-06.
- `queued` — **SRDMA: Efficient NIC-Based Authentication and Encryption for
  Remote Direct Memory Access**, Taranov et al., USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/taranov`
  Why: the RDMA hyperscale paper flags multi-tenant authentication and
  encryption as first-class issues; useful for deciding whether any future GPU
  DB RDMA/gateway path can preserve tenant isolation without excessive
  per-connection state or CPU-side crypto overhead.
- `queued` — **ReDMArk: Bypassing RDMA Security Mechanisms**, Rothenberger
  et al., USENIX Security 2021.
  URL:
  `https://www.usenix.org/conference/usenixsecurity21/presentation/rothenberger`
  Why: the RDMA hyperscale paper cites RDMA security weaknesses; useful
  negative case before GPU DB exposes remote memory, accelerator buffers, or
  tenant-visible gateway fast paths.
- `reviewed` — **Decibel: The Relational Dataset Branching System**,
  Maddox et al., PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p624-maddox.pdf`
  DOI: `https://doi.org/10.14778/2947618.2947619`
  Why: OrpheusDB contrasts its bolt-on relational approach with Decibel's
  native versioned storage engine; useful for comparing branch-aware storage
  primitives against GPU DB's MVCC lineage, retained snapshots, and
  old-version reconstruction cost. Journal entry added 2026-06-06.
- `reviewed` — **Principles of Dataset Versioning: Exploring the
  Recreation/Storage Tradeoff**, Bhattacherjee et al., PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol8/p1346-bhattacherjee.pdf`
  arXiv: `https://arxiv.org/abs/1505.05211`
  Why: OrpheusDB builds on the recreation/storage tradeoff for versioned
  datasets; useful for turning GPU DB snapshot-retention, checkpoint, and
  cold-version reconstruction policy into an explicit cost frontier.
  Journal entry added 2026-06-06.
- `queued` — **DataHub: Collaborative Data Science & Dataset Version
  Management at Scale**, Bhardwaj et al., CIDR 2015.
  URL: `https://www.cidrdb.org/cidr2015/Papers/CIDR15_Paper18.pdf`
  Why: Decibel is a DataHub component; useful for understanding the broader
  collaborative-data workload, access-control, provenance, and version-query
  surface that motivated branch-aware relational storage.
- `queued` — **Towards a Unified Query Language for Provenance and
  Versioning**, Chavan et al., TaPP 2015.
  URL: `https://www.usenix.org/conference/tapp15/workshop-program/presentation/chavan`
  Why: Decibel's VQuel support builds on this work; useful for deciding how
  much version/provenance query surface GPU DB should expose for retained
  generations, diffs, replay frontiers, and debugging.
- `reviewed` — **Type-Aware Transactions for Faster Concurrent Code**,
  Herman, Inala, Huang, Tsai, Kohler, Liskov, and Shrira, EuroSys 2016.
  URL: `https://doi.org/10.1145/2901318.2901348`
  Author PDF: `https://read.seas.harvard.edu/~kohler/pubs/herman16type-aware.pdf`
  Why: TDSL cites STO as independently developed semantic transactional
  object work; useful for comparing route-specific conflict predicates,
  datatype-owned commit hooks, and reduced read/write-set bookkeeping
  against generic tuple-level validation.
- `queued` — **Automatic Scalable Atomicity via Semantic Locking**,
  Golan-Gueta, Ramalingam, Sagiv, and Yahav, PPoPP 2015.
  URL:
  `https://www.microsoft.com/en-us/research/publication/automatic-scalable-atomicity-via-semantic-locking/`
  DOI: `https://doi.org/10.1145/2688500.2688511`
  Why: Type-Aware Transactions contrasts STO with automatic semantic locking;
  useful for comparing rollback-free pessimistic semantic locks against
  datatype-owned optimistic predicates and route-specific conflict contracts.
- `reviewed` — **SAP HANA Adoption of Non-Volatile Memory**, Andrei et al.,
  PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p1754-andrei.pdf`
  DOI: `https://doi.org/10.14778/3137765.3137780`
  Why: HANA NSE cites NVM as the nearer-memory tier between DRAM and disk;
  useful for comparing byte-compatible hot/warm structures against persistent
  memory placement, restart, and tier-specific durability behavior before GPU
  DB adds future CXL/NVM tiers.
- `reviewed` — **Characterizing, Modeling, and Benchmarking RocksDB Key-Value
  Workloads at Facebook**, Cao et al., FAST 2020.
  URL: `https://www.usenix.org/conference/fast20/presentation/cao-zhichao`
  Why: SKQ's RocksDB evaluation uses the ZippyDB workload model from this
  paper; useful for constructing realistic GET/SEEK/PUT service-time mixes,
  skewed tail-latency probes, and storage-adjacent runtime benchmarks before
  testing GPU DB pgwire/event-loop admission against only synthetic clients.
- `queued` — **Optimizing Space Amplification in RocksDB**, Dong et al.,
  CIDR 2017.
  URL: `https://www.cidrdb.org/cidr2017/papers/p82-dong-cidr17.pdf`
  Why: the FAST 2020 RocksDB workload paper depends on RocksDB's LSM and
  compaction behavior; this primary RocksDB design paper is useful for
  comparing space amplification, compaction, and cold-tier write pressure
  before GPU DB designs realistic LSM-like benchmark traces.
- `reviewed` — **FPTree: A Hybrid SCM-DRAM Persistent and Concurrent B-Tree for
  Storage Class Memory**, Oukid et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915251`
  Why: SAP HANA NVM cites FPTree as a persistent-memory index direction;
  useful for comparing DRAM-resident volatile inner nodes plus persistent
  leaves against GPU DB's CPU warm indexes, resident key vectors, and rebuild
  policy after restart. Reviewed on 2026-06-06.
- `queued` — **On Testing Persistent-Memory-Based Software**, Oukid et al.,
  DaMoN 2016.
  URL: `https://doi.org/10.1145/2933349.2933355`
  Why: SAP HANA NVM flags persistent-memory testing as a separate challenge;
  useful for designing crash/restart fault-injection gates before any future
  GPU DB CXL/NVM tier stores durable or semi-durable route metadata.
- `reviewed` — **Deadlocks in Datacenter Networks: Why Do They Form, and How to
  Avoid Them**, Hu et al., HotNets 2016.
  URL: `https://doi.org/10.1145/3005745.3005760`
  Why: Backpressure Flow Control relies on avoiding cyclic buffer dependencies;
  useful for translating network backpressure deadlock rules into GPU DB
  owner-ring, response-ring, and gateway admission graphs. Reviewed on
  2026-06-05.
- `queued` — **Tagger: Practical PFC Deadlock Prevention in Data Center
  Networks**, Hu et al., CoNEXT 2017.
  URL: `https://doi.org/10.1145/3143361.3143368`
  Why: BFC cites Tagger-style deadlock prevention for pause/resume paths;
  useful follow-up for ensuring selective backpressure and bounded queues
  cannot create distributed wait cycles across GPU DB runtime domains.
- `queued` — **Congestion Control for Large-Scale RDMA Deployments**,
  Zhu et al., SIGCOMM 2015.
  URL: `https://doi.org/10.1145/2785956.2787484`
  Why: the HotNets deadlock paper identifies DCQCN-style end-to-end congestion
  control as useful for reducing PFC generation but too delayed to eliminate it;
  useful for comparing end-to-end delay/rate feedback with GPU DB's local
  ring-pressure and credit signals.
- `reviewed` — **Selection Pushdown in Column Stores using Bit Manipulation
  Instructions**, Li, Lu, and Chandramouli, SIGMOD/PACMMOD 2023.
  URL:
  `https://www.microsoft.com/en-us/research/publication/selection-pushdown-in-column-stores-using-bit-manipulation-instructions/`
  PDF: `https://badrish.net/papers/bmi-sigmod2023.pdf`
  DOI: `https://doi.org/10.1145/3589323`
  Why: modern compressed-column scan work using bit-manipulation and
  SIMD-style mechanisms; useful follow-up after Rethinking SIMD for deciding
  whether CPU warm-tier compressed scans can beat GPU transfer or resident
  refresh on selective predicates. Author spelling, DOI, and PDF were
  corrected during review on 2026-06-05.
- `reviewed` — **Crystal: A Unified Cache Storage System for Analytical
  Databases**, Durner, Chandramouli, and Li, PVLDB 2021.
  URL: `https://vldb.org/pvldb/vol14/p2432-durner.pdf`
  DOI: `https://doi.org/10.14778/3476249.3476292`
  Why: Selection Pushdown and Microsoft data-lake work point to a
  query-aware cache layer with push-down predicates and region caching; useful
  for GPU DB's warm/cold tier placement and cache-admission contract.
  Reviewed on 2026-06-06.
- `queued` — **ReCache: Reactive Caching for Fast Analytics over
  Heterogeneous Data**, Azim, Karpathiotakis, and Ailamaki, PVLDB 2017.
  URL: `https://infoscience.epfl.ch/record/232607/files/p375-azim.pdf`
  DOI: `https://doi.org/10.14778/3157794.3157801`
  Why: LiquidCache notes that simple LRU is weak for analytical cache
  workloads; ReCache is a primary follow-up on workload-aware cache
  replacement and layout adaptation that may inform GPU DB resident and
  warm-tier admission policy.
- `queued` — **SOC: A Succinct Adaptive Semantic OLAP Caching**, You et al.,
  Data Science and Engineering 2025.
  URL: `https://link.springer.com/article/10.1007/s41019-025-00290-1`
  DOI: `https://doi.org/10.1007/s41019-025-00290-1`
  Why: discovered while reviewing Crystal's semantic-region caching; useful as
  a modern follow-up on compact semantic cache summaries, aggregate-result
  inference, and adaptive cache bounds for repeated OLAP-style routes.
- `reviewed` — **Lance: Efficient Random Access in Columnar Storage through
  Adaptive Structural Encodings**, Pace et al., arXiv 2025.
  URL: `https://arxiv.org/abs/2504.15247`
  Why: ByteHouse stores persistent multimodal data in formats including Lance;
  useful for comparing random-access columnar layout, vector/metadata access,
  and GPU DB's cold-tier point lookup path before adopting a self-describing
  file format for mixed scalar/text/vector columns. Journal entry added
  2026-06-06 from the arXiv paper.
- `queued` — **An Empirical Evaluation of Columnar Storage Formats**,
  Zeng et al., arXiv 2023.
  URL: `https://arxiv.org/abs/2304.05028`
  Why: ByteHouse's Sniffer format raises the question of tier-specific columnar
  file layout; this modern evaluation is useful for comparing Parquet, ORC,
  Arrow, and GPU-decoding implications before GPU DB fixes its own
  HBM/DRAM/NVMe segment format.
- `queued` — **Towards Functional Decomposition of Storage Formats**,
  Prammer, Zeng, Meng, McKinney, Zhang, Pavlo, and Patel, CIDR 2025.
  URL: `https://db.cs.cmu.edu/papers/2025/p19-prammer.pdf`
  Why: Lance argues structural encoding should be configurable rather than
  baked into one file format; this primary follow-up is useful for decomposing
  GPU DB's cold/warm segment format into independently benchmarked layout,
  metadata, compression, and access-method components.
- `reviewed` — **The Five-Minute Rule for the Cloud: Caching in Analytics
  Systems**, Duwe, Anadiotis, Lamb, Lersch, Leskes, Ritter, and Tozun,
  CIDR 2025.
  URL:
  `https://vldb.org/cidrdb/2025/the-five-minute-rule-for-the-cloud-caching-in-analytics-systems.html`
  PDF: `https://vldb.org/cidrdb/papers/2025/p4-duwe.pdf`
  Why: Lance frames NVMe as a cache layer for cloud/object storage; this
  follow-up is useful for deciding when GPU DB should keep hot/warm columnar
  fragments in HBM, DRAM, NVMe, or object storage based on access frequency,
  object-store latency, and cache cost. Journal entry added 2026-06-07 from
  the CIDR/VLDB PDF.
- `queued` — **Exploiting Cloud Object Storage for High-Performance
  Analytics**, Durner, Leis, and Neumann, PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p2769-durner.pdf`
  DOI: `https://doi.org/10.14778/3611540.3611549`
  Why: the cloud five-minute-rule paper uses this work for object-store
  latency and performance data; useful for comparing direct object-store
  access, request parallelism, caching, and cold-tier object sizing before GPU
  DB models a remote/object storage tier.
- `queued` — **Predicate Caching: Query-Driven Secondary Indexing for Cloud
  Data Warehouses**, Schmidt, Kipf, Horn, Saxena, and Kraska, SIGMOD/PACMMOD
  2024.
  URL: `https://doi.org/10.1145/3654903`
  Why: the cloud five-minute-rule paper points to query-driven secondary
  indexing for cloud warehouses; useful for deciding when GPU DB should
  materialize predicate-specific warm fragments or on-the-fly indexes instead
  of caching whole cold-tier objects.
- `queued` — **Bullion: A Column Store for Machine Learning**, Liao, Liu,
  Chen, and Abadi, CIDR 2025.
  URL:
  `https://vldb.org/cidrdb/2025/bullion-a-column-store-for-machine-learning.html`
  PDF: `https://vldb.org/cidrdb/papers/2025/p26-liao.pdf`
  Why: Lance highlights ML-style nested and wide-column workloads; Bullion is
  a modern primary follow-up for comparing wide/nested column-store layout
  against GPU DB's future vector/text/scalar segment design.
- `queued` — **SyPer: Connecting the Pieces for Hybrid Transactional and
  Analytical Processing**, Wang et al., PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p673-wang.pdf`
  DOI: `https://doi.org/10.14778/3055540.3055545`
  Why: Vegito contrasts SyPer as a snapshot/replica-style HTAP design; useful
  for comparing virtual snapshot freshness, analytical isolation, and OLTP
  degradation against backup-based and GPU-resident snapshot publication.
- `reviewed` — **1RMA: Re-Envisioning Remote Memory Access for Multi-Tenant
  Datacenters**, Singhvi et al., SIGCOMM 2020.
  URL: `https://doi.org/10.1145/3387514.3405897`
  PDF: `https://pages.cs.wisc.edu/~asinghvi/papers/1rma.pdf`
  Why: EQDS builds on 1RMA-style receiver-driven credits for RDMA-compatible
  traffic; useful for comparing tenant isolation, pull-based admission, and
  remote-memory access semantics against GPU DB's future storage or gateway
  fabric. Journal entry added 2026-06-06; the stale DOI was corrected during
  review.
- `reviewed` — **RDMA Performance Isolation with Justitia**, Zhang, Tan,
  Stephens, and Chowdhury, arXiv 2019.
  URL: `https://arxiv.org/abs/1905.04437`
  Why: 1RMA contrasts connection-oriented sender shaping and performance
  isolation with its connection-free finite-resource model; useful for
  comparing software-only pacing, fairness, and tenant isolation before GPU DB
  designs route credits for shared gateway or remote-tier access. Journal
  entry added 2026-06-06 from the arXiv PDF.
- `reviewed` — **Aeolus: A Building Block for Proactive Transport in
  Datacenters**, Hu et al., SIGCOMM 2020.
  URL: `https://doi.org/10.1145/3387514.3405878`
  PDF: `https://www.cse.ust.hk/~kaichen/papers/aeolus-sigcomm20.pdf`
  Why: EQDS names Aeolus as a Homa-like proactive transport option; useful for
  comparing receiver-driven low-latency request/response transport with GPU
  DB's command-ring credits and micro-batch admission. Journal entry added
  2026-06-06.
- `queued` — **TFC: Token Flow Control in Data Center Networks**, Zhang, Ren,
  Shu, and Cheng, EuroSys 2016.
  URL:
  `https://www.microsoft.com/en-us/research/publication/tfc-token-flow-control-in-data-center-networks/`
  DOI: `https://doi.org/10.1145/2901318.2901336`
  Why: Aeolus cites TFC as a proactive transport that explicitly allocates
  link bandwidth using tokens at switches; useful for comparing switch-side
  token allocation, zero-queueing goals, and highly concurrent flow control
  with GPU DB's local ring credits and scheduled/speculative admission lanes.
- `queued` — **Rogue: RDMA over Generic Unconverged Ethernet**, Le, Stephens,
  Singhvi, Akella, and Swift, SoCC 2018.
  URL: `https://doi.org/10.1145/3267809.3267828`
  Why: Pangu flags PFC-free and lossy RDMA as future production directions;
  useful before GPU DB assumes lossless fabrics, PFC, or switch tuning for
  accelerator/storage network paths.
- `queued` — **Silo: Predictable Message Latency in the Cloud**, Jang,
  Sherry, Ballani, and Moncaster, SIGCOMM 2015.
  URL: `https://doi.org/10.1145/2785956.2787479`
  Why: Justitia cites Silo as a datacenter latency/bandwidth guarantee design;
  useful for comparing burst allowances, latency reservations, and network
  admission against GPU DB's local gateway/ring credit model.
- `reviewed` — **REWIND: Recovery Write-Ahead System for In-Memory
  Non-Volatile Data-Structures**, Chatzistergiou, Cintra, and Viglas,
  PVLDB 2015.
  URL: `https://www.vldb.org/pvldb/vol8/p497-chatzistergiou.pdf`
  DOI: `https://doi.org/10.14778/2735479.2735483`
  Why: FPTree's split/delete micro-logs and persistent allocator make crash
  repair a first-class index concern; REWIND is a primary follow-up for
  comparing write-ahead recovery schemes for persistent data structures before
  GPU DB stores durable route metadata or future-tier indexes. Journal entry
  added 2026-06-06.
- `reviewed` — **NBR: Neutralization Based Reclamation**, Singh et al.,
  PPoPP 2021.
  URL: `https://doi.org/10.1145/3437801.3441625`
  arXiv: `https://arxiv.org/abs/2012.14542`
  Why: VBR compares against signaling-based robust reclamation; useful for
  deciding whether route metadata and CPU-side resident indexes should use
  cooperative optimistic retries, neutralization of stalled workers, or a
  simpler epoch contract under 1M-session pressure. Journal entry added
  2026-06-06.
- `reviewed` — **Snapshot-Free, Transparent, and Robust Memory Reclamation for
  Lock-Free Data Structures**, Nikolaev and Ravindran, PLDI 2021.
  URL: `https://doi.org/10.1145/3453483.3454090`
  arXiv: `https://arxiv.org/abs/1905.07903`
  Why: VBR contrasts reclamation designs by robustness, transparency, and
  fence overhead; Hyaline-style snapshot-free reclamation is a useful
  follow-up for route-publication descriptors and lock-free CPU indexes that
  should avoid pinning retired state behind long readers. Journal entry added
  2026-06-06.
- `reviewed` — **Publish on Ping: A Better Way to Publish Reservations in Memory
  Reclamation for Concurrent Data Structures**, arXiv 2025.
  URL: `https://arxiv.org/abs/2501.04250`
  DOI: `https://doi.org/10.1145/3710848.3710890`
  Why: discovered while reviewing NBR; combines signal-style prompting with
  delayed publication of reservations, making it a modern follow-up for
  lowering hazard-pointer-style read overhead without allowing unbounded
  retired route metadata. Journal entry added 2026-06-06.
- `reviewed` — **Crystalline: Fast and Memory Efficient Wait-Free
  Reclamation**, Nikolaev and Ravindran, arXiv 2021.
  URL: `https://arxiv.org/abs/2108.02763`
  Why: Publish on Ping references Crystalline as related robust reclamation
  work; useful for comparing bounded-garbage, wait-free, and memory-efficient
  retirement policies for route descriptors and lock-free CPU-side indexes.
  Journal entry added 2026-06-06.
- `reviewed` — **Universal Wait-Free Memory Reclamation**, Nikolaev and
  Ravindran, PPoPP 2020.
  URL: `https://doi.org/10.1145/3332466.3374540`
  Why: Crystalline compares against WFE as the prior general wait-free
  reclamation baseline; useful for deciding whether the full wait-free
  machinery is justified versus a bounded lock-free route-descriptor scheme.
  Journal entry added 2026-06-07 from the author PDF and arXiv metadata.
- `queued` — **Hazard Eras: Non-Blocking Memory Reclamation**,
  Ramalhete and Correia, SPAA 2017.
  URL: `https://doi.org/10.1145/3087556.3087588`
  Author PDF: `https://github.com/pramalhe/ConcurrencyFreaks/raw/master/papers/hazarderas-2017.pdf`
  Why: WFE extends Hazard Eras and inherits its bounded-retired-object safety
  model; useful for deciding whether GPU DB route descriptors need the full
  wait-free helper path or can use the simpler lock-free era baseline.
- `queued` — **Fast and Robust Memory Reclamation for Concurrent Data
  Structures**, Balmau, Guerraoui, Herlihy, and Zablotchi, SPAA 2016.
  URL: `https://doi.org/10.1145/2935764.2935790`
  Why: WFE contrasts QSense-style OS-scheduler/signal-assisted reclamation
  with non-blocking manual schemes; useful for judging whether session-owner
  cleanup should ever depend on runtime interruption of slow readers.
- `queued` — **A Marriage of Pointer- and Epoch-Based Reclamation**, Kang
  and Jung, PLDI 2020.
  URL: `https://doi.org/10.1145/3385412.3386008`
  Why: Crystalline discusses PEBR as a hybrid pointer/epoch design; useful
  contrast for GPU DB if bounded route metadata needs simpler restart-based
  protection rather than wait-free helper handoff.
- `queued` — **Stamp-it: a More Thread-efficient, Concurrent Memory
  Reclamation Scheme in the C++ Memory Model**, Poeter and Traff, SPAA 2018.
  URL: `https://doi.org/10.1145/3210377.3210660`
  Why: Crystalline cites Stamp-it as a bounded-reclamation-cost epoch
  direction; useful for comparing monotonic stamp-based retirement with
  route generation tokens and owner-local cleanup queues.
- `reviewed` — **Concurrent Deferred Reference Counting with Constant-Time
  Overhead**, Anderson, Blelloch, and Wei, PLDI 2021.
  URL: `https://doi.org/10.1145/3453483.3454060`
  PDF: `https://www.cs.cmu.edu/~guyb/papers/3453483.3454060.pdf`
  Why: Publish on Ping's related work includes automatic/reference-counting
  style reclamation; useful as a contrast to hazard/epoch designs before GPU
  DB picks a route metadata lifetime scheme. Journal entry added 2026-06-07.
- `queued` — **Concurrent Fixed-Size Allocation and Free in Constant Time**,
  Blelloch and Wei, DISC 2020 brief announcement / arXiv 2020.
  URL: `https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.DISC.2020.51`
  arXiv: `https://arxiv.org/abs/2008.04296`
  Why: Concurrent Deferred Reference Counting depends on bounded auxiliary
  memory and fixed-size reclamation costs; useful for route-descriptor pools,
  response-buffer slabs, and owner-local allocation where allocation/free
  should remain constant-time under high session counts.
- `reviewed` — **OrcGC: Automatic Lock-Free Memory Reclamation**, Correia,
  Ramalhete, and Felber, PPoPP 2021.
  URL: `https://doi.org/10.1145/3437801.3441596`
  PDF: `https://zenodo.org/records/7886712/files/OrcGC-zenodo.pdf`
  Why: Hyaline contrasts automatic and reference-counting style reclamation
  designs; OrcGC is a useful follow-up for deciding whether GPU DB route
  metadata should remain manually retired by owner domains or hide
  protection/deallocation in a more automatic descriptor API. Journal entry
  added 2026-06-06 from the author/Zenodo PDF.
- `reviewed` — **Practically and Theoretically Efficient Garbage Collection for
  Multiversioning**, Wei, Blelloch, Fatourou, and Ruppert, PPoPP 2023.
  URL: `https://doi.org/10.1145/3572848.3577508`
  arXiv: `https://arxiv.org/abs/2212.13557`
  PDF: `https://www.cs.cmu.edu/~guyb/papers/3572848.3577508.pdf`
  Why: OrcGC is a general lock-free reclamation design, but GPU DB's hardest
  reclamation pressure is multiversion state; this follow-up directly studies
  MVGC on versioned trees and hash tables with space bounds. Journal entry
  already exists; this stale duplicate was corrected from `queued` to
  `reviewed` on 2026-06-06.
- `reviewed` — **Efficient Hardware Primitives for Immediate Memory Reclamation
  in Optimistic Data Structures**, Singh, Brown, and Spear, arXiv 2023.
  URL: `https://arxiv.org/abs/2302.12958`
  Why: OrcGC still delays reclamation through hazard-style protection and
  handoff; Conditional Access is a hardware-primitive contrast for whether
  coherence-assisted immediate reclamation could ever matter for future CPU
  warm indexes or route descriptors. Journal entry added 2026-06-06 from
  arXiv v1.
- `queued` — **Memory Tagging: Minimalist Synchronization for Scalable
  Concurrent Data Structures**, Alistarh, Brown, and Singhal, SPAA 2020.
  URL: `https://doi.org/10.1145/3350755.3400243`
  PDF: `https://mc.uwaterloo.ca/pubs/spaa20_memtags/paper.pdf`
  Why: Conditional Access is inspired by this cache-line tagging primitive;
  useful for comparing validation-before-access synchronization with
  immediate reclamation and route-descriptor protection.
- `queued` — **Hand-Over-Hand Transactions with Precise Memory Reclamation**,
  Zhou, Luchangco, and Spear, SPAA 2017.
  URL: `https://doi.org/10.1145/3087556.3087583`
  Why: Conditional Access contrasts itself with short hardware transactions
  for precise reclamation; useful as a boundary case for whether GPU DB should
  rely on HTM-like critical sections for CPU-side indexes or metadata.
- `reviewed` — **To Store or Not to Store: a graph theoretical approach for
  Dataset Versioning**, Guo, Li, Sukprasert, Khuller, Deshpande, and
  Mukherjee, arXiv 2024.
  URL: `https://arxiv.org/abs/2402.11741`
  Why: discovered while reviewing the PVLDB 2015 dataset-versioning
  storage/recreation frontier; useful as a modern follow-up on graph-based
  storage and reconstruction optimization before GPU DB turns snapshot
  retention into an online multi-tier policy. Journal entry added
  2026-06-06; the stale author field was corrected during review.
- `reviewed` — **CHEX: Multiversion Replay with Ordered Checkpoints**,
  Manne et al., PVLDB 2022.
  URL: `https://doi.org/10.14778/3514061.3514075`
  Why: To Store or Not to Store cites CHEX as a graph snapshot/versioning
  system; useful for comparing checkpoint placement, replay depth, and
  version retrieval latency against GPU DB retained snapshots and cold
  version reconstruction. Journal entry added 2026-06-06.
- `queued` — **Materialization and Reuse Optimizations for Production Data
  Science Pipelines**, Derakhshan et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526186`
  Why: To Store or Not to Store cites production-pipeline materialization as
  related version reuse work; useful for deciding when GPU DB should
  materialize intermediate retained fragments, deltas, or checkpointed
  generations instead of recomputing them.
- `queued` — **Mosaic: A Budget-Conscious Storage Engine for Relational
  Database Systems**, Vogel et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p2662-vogel.pdf`
  DOI: `https://doi.org/10.14778/3407790.3407858`
  Why: To Store or Not to Store cites Mosaic as device-aware placement work;
  useful for comparing cost-aware DRAM/NVM/SSD placement with GPU DB's
  HBM/DRAM/NVMe resident snapshot and warm-tier policy.
- `queued` — **Your Notebook is not Crumby Enough, REPLace it**,
  Brachmann et al., CIDR 2020.
  URL: `https://www.cidrdb.org/cidr2020/papers/p5-brachmann-cidr20.pdf`
  Why: CHEX depends on REPL-style cell lineage and cites this notebook
  provenance/replay line; useful for deciding how fine-grained retained
  route lineage should be before GPU DB can safely reuse intermediate
  snapshot fragments or explain replay decisions.
- `queued` — **Multiversion Hindsight Logging for Continuous Training**,
  arXiv 2023.
  URL: `https://arxiv.org/abs/2310.07898`
  Why: discovered while reviewing CHEX; useful as a modern follow-up on
  replaying log statements across many model/program versions, with
  checkpoint-based parallelism that may map to GPU DB version-tree replay
  and cold snapshot reconstruction benchmarks.
- `reviewed` — **Efficient Logging in Non-Volatile Memory by Exploiting
  Coherency Protocols**, Cohen, Friedman, and Larus, OOPSLA/PACMPL 2017.
  URL: `https://arxiv.org/abs/1709.02610`
  DOI: `https://doi.org/10.1145/3133891`
  Why: discovered while reviewing REWIND; useful follow-up on persist-order
  costs, coherence-induced reordering, and single-round-trip NVM logging before
  GPU DB adopts any byte-addressable warm-tier log or durable metadata path.
  Journal entry added 2026-06-06 from the arXiv/PACMPL paper.
- `queued` — **Fine-Grain Checkpointing with In-Cache-Line Logging**,
  Cohen, Aksun, Avni, and Larus, ASPLOS 2019.
  URL: `https://arxiv.org/abs/1902.00660`
  DOI: `https://doi.org/10.1145/3297858.3304046`
  Why: discovered while reviewing REWIND; useful follow-up on low-overhead
  persistent Masstree-style structures, in-cache-line undo records, and
  checkpoint granularity for future CPU warm indexes or route metadata.
- `reviewed` — **DudeTM: Building Durable Transactions with Decoupling for
  Persistent Memory**, Liu et al., ASPLOS 2017.
  PDF:
  `https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/dudetm_asplos17.pdf`
  DOI: `https://doi.org/10.1145/3037697.3037714`
  Why: PCSO logging contrasts DudeTM's background persistence and durability
  latency tradeoff; useful for comparing foreground one-flush durability with
  decoupled logging when GPU DB evaluates NVM/CXL write-path staging.
  Journal entry added 2026-06-07 from the Microsoft Research author PDF.
- `queued` — **DUMBO: Making durable read-only transactions fly on hardware
  transactional memory**, Dias, Felber, Fetzer, and Ramalhete, arXiv 2024.
  URL: `https://arxiv.org/abs/2410.16110`
  Why: discovered while reviewing DudeTM; useful modern follow-up on durable
  read-only transaction costs, persistent HTM design, and whether GPU DB can
  separate durable read validation from write-path persistence under retained
  snapshot workloads.
- `queued` — **Log-Structured Non-Volatile Main Memory**, Hu, Ren, Badam, and
  Moscibroda, USENIX ATC 2017.
  URL: `https://www.usenix.org/conference/atc17/technical-sessions/presentation/hu`
  Why: PCSO logging contrasts log-structured NVM management that turns writes
  into append operations indexed by volatile metadata; useful for comparing
  persistent warm-tier logs, allocator recovery, and route-metadata rebuild
  strategies.
