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

## Seed Queue

### Transaction processing, write path, and concurrency control

- `reviewed` — **TicToc: Time Traveling Optimistic Concurrency Control**,
  Yu et al., SIGMOD 2016.
  URL: `https://dl.acm.org/doi/10.1145/2882903.2882935`
  PDF: `https://db.cs.cmu.edu/papers/2016/yu-sigmod2016.pdf`
  Why: timestamp-based optimistic concurrency control for high-throughput
  transaction processing; useful for comparing MVCC/snapshot timestamp choices.
- `queued` — **Cicada: Dependably Fast Multi-Core In-Memory Transactions**,
  Lim et al., SIGMOD 2017.
  URL: `https://dl.acm.org/doi/10.1145/3035918.3064015`
  Why: high-throughput multicore transaction processing with concurrency
  control, versioning, and contention management tradeoffs.
- `queued` — **ERMIA: Fast Memory-Optimized Database System for Heterogeneous
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
- `queued` — **On Supporting Efficient Snapshot Isolation for In-Memory
  Database Storage**, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p211-sun.pdf`
  Why: P-Tree index for efficient snapshot isolation and MVCC in multicore
  in-memory HTAP storage.
- `queued` — **An Empirical Evaluation of In-Memory Multi-Version Concurrency
  Control**, PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p781-Wu.pdf`
  Why: MVCC design tradeoffs, version storage, validation, and GC behavior.
- `queued` — **Accelerating Analytical Processing in MVCC using Fine-Granular
  High-Frequency Virtual Snapshotting**, arXiv 2017.
  URL: `https://arxiv.org/abs/1709.04284`
  Why: HTAP-style analytical snapshots without blocking write progress.

### Runtime scale, HFT-style mechanics, and admission

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
- `queued` — **Design Choices in Low-Latency C++ Systems: Empirical Insights
  With Applications to High-Frequency Trading**, SSRN 2026.
  URL: `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6513601`
  Why: modern HFT-oriented low-latency systems survey; useful for queue,
  allocation, cache, and thread-pinning patterns.

### Multi-tier cache, buffer management, and data placement

- `queued` — **LeanStore: In-Memory Data Management beyond Main Memory**,
  Leis et al., ICDE 2018.
  URL: `https://doi.org/10.1109/ICDE.2018.00026`
  Metadata:
  `https://portal.fis.tum.de/en/publications/leanstore-in-memory-data-management-beyond-main-memory`
  Why: low-overhead storage manager that keeps in-memory performance for hot
  data while transparently handling SSD-resident data; directly relevant to
  GPU/DRAM/NVMe tiering and transactional working sets.
- `queued` — **Umbra: A Disk-Based System with In-Memory Performance**,
  Neumann and Freitag, CIDR 2020.
  URL: `https://www.vldb.org/cidrdb/papers/2020/p29-neumann-cidr20.pdf`
  Why: variable-size pages and low-overhead buffer management for cached hot
  working sets with graceful uncached access; useful for resident snapshot and
  host/NVMe tier design.
- `queued` — **Are You Sure You Want to Use MMAP in Your Database Management
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
- `queued` — **Multi-Tier Buffer Management and Storage System Design for
  Non-Volatile Memory**, arXiv 2019.
  URL: `https://arxiv.org/abs/1901.10938`
  Why: explicit multi-tier DBMS buffer design across DRAM and non-volatile
  storage; useful for promotion/demotion policy and tier-aware page layout.
- `queued` — **Efficient Compactions Between Storage Tiers with PrismDB**,
  arXiv 2020.
  URL: `https://arxiv.org/abs/2008.02352`
  Why: multi-tier storage compaction across fast and slow devices; relevant to
  cold/warm segment organization and write amplification when data moves
  between tiers.

### Query optimizers, planning, and route choice

- `queued` — **Lero: A Learning-to-Rank Query Optimizer**, arXiv 2023.
  URL: `https://arxiv.org/abs/2302.06873`
  Why: learned ranking layered on native optimizers; relevant to route choice
  without replacing deterministic planner rules.
- `queued` — **AutoSteer: Learned Query Optimization for Any SQL Database**,
  PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p3515-anneser.pdf`
  Why: learned tuning of optimizer knobs for existing SQL systems; relevant to
  GPU route knobs and fallback decisions.
- `queued` — **Rethinking Learned Cost Models: Why Start from Scratch?**,
  SIGMOD 2023.
  URL: `https://15799.courses.cs.cmu.edu/spring2025/papers/15-learned/yang-sigmod2023.pdf`
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
- `queued` — **Concurrent query processing in a GPU-based database system**,
  PLOS ONE 2019.
  URL: `https://pmc.ncbi.nlm.nih.gov/articles/PMC6467383/`
  Why: batch-level optimization model for concurrent GPU database workloads.
- `reviewed` — **Data Path Fusion in GPU for Analytical Query Processing**,
  arXiv 2026.
  URL: `https://arxiv.org/abs/2605.10511`
  Why: modern GPU-driven data path fusion that combines IO, decompression, and
  query work into GPU execution.
- `queued` — **RTCUDB: Building Databases with RT Processors**, arXiv 2024.
  URL: `https://arxiv.org/abs/2412.09337`
  Why: explores ray-tracing cores for database query processing and may suggest
  alternate hardware mapping for lookup/search-heavy paths.
- `queued` — **GOLAP: A GPU-in-Data-Path Architecture for High-Speed OLAP**,
  2024.
  URL: `https://dl.acm.org/doi/10.1145/3654925`
  Why: GPU-in-data-path design for compressed block streaming, decompression,
  and scan processing.
- `queued` — **Revisiting Query Performance in GPU Database Systems**,
  arXiv 2023.
  URL: `https://arxiv.org/abs/2302.00734`
  Why: cross-stack GPU DBMS performance, resource utilization, and concurrent
  query recommendations.
- `queued` — **Efficiently Processing Joins and Grouped Aggregations on GPUs**,
  arXiv 2023.
  URL: `https://arxiv.org/abs/2312.00720`
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

- `skipped` — **The Yin and Yang of Processing Data Warehousing Queries on GPU
  Devices**, Yuan et al., PVLDB 2013.
  URL: `https://www.vldb.org/pvldb/vol6/p817-yuan.pdf`
  Why: pre-2015; keep only as historical context.
- `skipped` — **High-Throughput Transaction Executions on Graphics Processors**,
  He and Yu, PVLDB 2011.
  URL: `https://www.vldb.org/pvldb/vol4/p314-he.pdf`
  Why: pre-2015; keep only as historical context.
- `queued` — **Themis: A GPU-accelerated Relational Query Execution Engine**,
  Hong et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol18/p426-hong.pdf`
  Why: modern GPU relational engine with execution and load-balancing details
  relevant to fused retained route design.
- `queued` — **GPU Acceleration of SQL Analytics on Compressed Data**,
  Huang et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol19/p320-huang.pdf`
  Why: evaluates compressed-data SQL execution on GPUs and may inform
  dense-versus-compressed resident page benchmarks.
- `queued` — **Path to GPU-Initiated I/O for Data-Intensive Systems**,
  Torp et al., DaMoN 2025.
  URL: `https://doi.org/10.1145/3736227.3736233`
  Why: practical evaluation of GPU-initiated IO paths for data-intensive
  systems, directly relevant to DPF-style over-resident execution.
- `reviewed` — **Scaling GPU-Accelerated Databases beyond GPU Memory Size**,
  Li et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4518-li.pdf`
  Why: modern out-of-GPU-memory database execution work relevant to P8
  over-resident partitioning and CPU/GPU fallback.
- `queued` — **Vortex: Overcoming Memory Capacity Limitations in
  GPU-Accelerated Large-Scale Data Analytics**, Yuan et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol18/p1250-yuan.pdf`
  Why: multi-GPU/interconnect approach to capacity limits, useful for future
  over-resident and multi-device planning.
- `queued` — **Predicate Transfer: Efficient Pre-Filtering on Multi-Join
  Queries**, Yang et al., CIDR 2024.
  URL: `https://www.cidrdb.org/cidr2024/papers/p22-yang.pdf`
  Why: modern predicate-transfer/pre-filtering work cited by the 2025 hybrid
  CPU-GPU paper; relevant to reducing over-resident transfer before GPU joins.
- `queued` — **Orchestrating data placement and query execution in
  heterogeneous CPU-GPU DBMS**, Yogatama et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p2491-yogatama.pdf`
  Why: cost-based CPU/GPU placement and execution orchestration for
  heterogeneous DBMS designs.
- `queued` — **Accelerating GPU Data Processing using FastLanes
  Compression**, Afroozeh et al., DaMoN 2024.
  URL: `https://doi.org/10.1145/3662010.3663450`
  Why: modern GPU compressed-data execution follow-up for resident and
  over-resident compressed page experiments.
- `queued` — **Fast Serializable Multi-Version Concurrency Control for
  Main-Memory Database Systems**, Neumann et al., SIGMOD 2015.
  URL: `https://dl.acm.org/doi/10.1145/2723372.2749436`
  Why: direct OMVCC baseline for transaction repair, with timestamp,
  validation, and version-chain design relevant to serializable MVCC in a
  memory-resident engine.
- `queued` — **Leveraging Lock Contention to Improve OLTP Application
  Performance**, Yan and Cheung, PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p444-yan.pdf`
  Why: program-analysis and contention-aware execution ideas that complement
  MV3C's dependency-annotated transaction repair path.
- `queued` — **Chiller: Contention-centric Transaction Execution and Data
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
- `queued` — **Scalable RDMA RPC on Reliable Connection with Efficient
  Resource Sharing**, Chen et al., EuroSys 2019.
  URL: `https://doi.org/10.1145/3302424.3303983`
  Preprint: `https://chenyoumin1993.github.io/papers/eurosys19-scalerpc.pdf`
  Why: ScaleRPC-style resource sharing over RDMA connection state may inform
  future session multiplexing and bounded transport resource budgets.
- `queued` — **Carousel: Scalable Traffic Shaping at End Hosts**,
  Saeed et al., SIGCOMM 2017.
  URL: `https://doi.org/10.1145/3098822.3098852`
  Why: rate-limiter design used by eRPC; relevant to per-session admission,
  congestion shaping, and bounded response scheduling at high connection
  counts.
- `queued` — **TIMELY: RTT-based Congestion Control for the Datacenter**,
  Mittal et al., SIGCOMM 2015.
  URL: `https://doi.org/10.1145/2785956.2787510`
  Why: eRPC's congestion-control path builds on Timely; useful for deciding
  whether GPU DB network admission should use RTT/queue-delay telemetry.
- `queued` — **Shinjuku: Preemptive Scheduling for microsecond-scale Tail
  Latency**, Kaur et al., NSDI 2019.
  URL: `https://www.usenix.org/conference/nsdi19/presentation/kagami`
  Why: microsecond-scale request scheduling and preemption; relevant to
  separating short retained reads from long mutation, scan, or refresh work.
- `reviewed` — **Memory-Optimized Multi-Version Concurrency Control for
  Disk-Based Database Systems**, Freitag et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p2797-freitag.pdf`
  Why: memory-optimized MVCC in a disk-backed engine; relevant to version
  storage, undo chains, and cache-aware snapshot visibility.
- `queued` — **FOEDUS: OLTP Engine for a Thousand Cores and NVRAM**,
  Kimura, SIGMOD 2015.
  URL: `https://dl.acm.org/doi/10.1145/2723372.2746480`
  Tech report: `https://www.labs.hpe.com/techreports/2015/HPL-2015-37.pdf`
  Why: many-core OLTP and NVRAM-oriented storage architecture cited by TicToc;
  relevant to partition ownership, logging, NUMA locality, and future tiers.
- `queued` — **Diva: Making MVCC Systems HTAP-Friendly**, Kim et al.,
  SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526125`
  Why: vDriver successor cited by the LeanStore paper; relevant to precise
  MVCC garbage collection and reducing long-reader damage in HTAP workloads.
- `reviewed` — **Virtual-Memory Assisted Buffer Management In Tiered Memory**,
  Rayhan and Aref, arXiv 2026.
  URL: `https://arxiv.org/abs/2603.03271`
  Why: extends vmcache-style virtual-memory-assisted buffer management to
  multiple memory tiers such as DRAM, remote memory/CXL-like tiers, and disk;
  directly relevant to future GPU DB host-tier and cold-partition placement.
- `reviewed` — **PARQO: Penalty-Aware Robust Plan Selection in Query
  Optimization**, Xiu et al., PVLDB 2024.
  URL: `https://doi.org/10.14778/3704965.3704971`
  arXiv: `https://arxiv.org/abs/2406.01526`
  Why: robust plan selection with user-defined penalty functions and
  workload-informed selectivity-error models; useful for deciding when a
  fast GPU route is too fragile under uncertain cardinality, transfer, or
  queue-delay estimates.
- `queued` — **LOGER: A Learned Optimizer towards Generating Efficient and
  Robust Query Execution Plans**, Chen et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p1777-gao.pdf`
  Why: combines learned optimizer search with DBMS operator restrictions and
  beam search; relevant to adding learned GPU route suggestions without
  surrendering deterministic planner guardrails.
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
- `queued` — **Aria: A Fast and Practical Deterministic OLTP Database**, Lu,
  Yu, Cao, and Madden, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p2047-lu.pdf`
  Why: deterministic OLTP execution without a global serial schedule bottleneck;
  useful as a contrast point for Oze, mutation batching, and whether
  predeclared GPU DB write/read sets can improve concurrency without forcing
  long retained refresh work to stall short transactions.
- `queued` — **Arachne: Core-Aware Thread Management**, Qin et al.,
  OSDI 2018.
  URL: `https://www.usenix.org/conference/osdi18/presentation/qin`
  Why: user-level core-aware thread management cited by Caladan; useful for
  exposing internal request concurrency to a scheduler without adopting a full
  Caladan-style interference-control stack.
- `queued` — **ZygOS: Achieving Low Tail Latency for Microsecond-scale
  Networked Tasks**, Prekas, Kogias, and Bugnion, SOSP 2017.
  URL: `https://dl.acm.org/doi/10.1145/3132747.3132780`
  PDF: `https://marioskogias.github.io/docs/zygos.pdf`
  Why: work-conserving dataplane scheduler for high-connection-count
  microsecond services, including Silo/TPC-C evaluation; relevant to pgwire
  IO-worker and request-stealing choices.
- `queued` — **Pasha: An Efficient, Scalable Database Architecture for CXL
  Pods**, Huang et al., CIDR 2025.
  URL: `https://www.vldb.org/cidrdb/papers/2025/p8-huang.pdf`
  Why: CXL-pod database architecture cited by vmcache^n; relevant to
  disaggregated/tiered memory placement and future host-memory expansion.
- `reviewed` — **Resource-Adaptive Query Execution with Paged Memory
  Management**, Otaki, Benello, Elmore, and Graefe, CIDR 2025.
  URL: `https://www.vldb.org/cidrdb/papers/2025/p2-otaki.pdf`
  Why: paged-memory and resource-adaptive execution work cited by vmcache^n;
  relevant to query admission and execution under memory-tier pressure.
- `queued` — **Nomad: Non-Exclusive Memory Tiering via Transactional Page
  Migration**, Xiang et al., OSDI 2024.
  URL: `https://www.usenix.org/conference/osdi24/presentation/xiang`
  Why: transactional page migration for tiered memory cited by vmcache^n;
  useful for comparing OS-assisted migration against explicit DBMS ownership.
- `queued` — **Towards Buffer Management with Tiered Main Memory**, Hao et al.,
  PACMMOD/SIGMOD 2024.
  URL: `https://doi.org/10.1145/3639303`
  Why: modern tiered-main-memory buffer management cited by vmcache^n;
  relevant to DRAM/remote-memory/NVMe policy design and placement economics.
- `reviewed` — **PAR2QO: Parametric Penalty-Aware Robust Query Optimization**,
  Xiu et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4532-xiu.pdf`
  DOI: `https://doi.org/10.14778/3749646.3749711`
  Why: follow-up to PARQO that focuses on parametric robust query
  optimization and plan-penalty profile caching; relevant to repeated retained
  GPU route templates and admission-time route reuse.
- `queued` — **Hints for Robust Query Performance Tuning**, Xiu et al.,
  PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p5327-xiu.pdf`
  Why: PARQO-adjacent robust tuning work that may turn sensitive cardinality
  dimensions into actionable hints; useful for exposing why a GPU route,
  fallback route, or refresh decision is fragile.
- `queued` — **Kepler: Robust Learning for Faster Parametric Query
  Optimization**, Doshi et al., PACMMOD/SIGMOD 2023.
  URL: `https://arxiv.org/abs/2306.06798`
  Why: robust parametric query optimization using executed-query evidence;
  useful as a contrast to PARQO's cost-model-based route cache for repeated
  SQL templates.
- `queued` — **Plor: General Transactions with Predictable, Low Tail
  Latency**, Chen et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517839`
  Why: Shirakami cites Plor as modern transaction scheduling work; relevant to
  predictable low-tail mutation and admission paths under mixed transaction
  sizes.
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
- `queued` — **GaccO - A GPU-accelerated OLTP DBMS**, Boeschen and Binnig,
  SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517846`
  Why: GPU single-version deterministic locking baseline compared by Epic;
  relevant to deciding when commutative GPU updates beat general MVCC.
- `queued` — **Caracal: Contention Management with Deterministic Concurrency
  Control**, Qin, Brown, and Goel, SOSP 2021.
  URL: `https://doi.org/10.1145/3477132.3483572`
  Why: deterministic MVCC contention-management baseline for Epic; useful for
  CPU-side owner/partition batching and skewed write-set planning.
- `queued` — **High Performance Transactions via Early Write Visibility**,
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
- `reviewed` — **Scalable Garbage Collection for In-Memory MVCC Systems**,
  Boettcher et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol13/p128-bottcher.pdf`
  Why: Steam-style MVCC version garbage collection cited by the Umbra MVCC
  paper; relevant to bounded version retention, long retained snapshots, and
  per-owner GC without global contention.
- `queued` — **BTrim - Hybrid In-Memory Database Architecture for Extreme
  Transaction Processing in VLDBs**, Gurajada et al., PVLDB 2018.
  URL: `https://www.vldb.org/pvldb/vol11/p1889-guradaja.pdf`
  Why: hybrid disk/in-memory transactional architecture cited by the Umbra
  MVCC paper; useful as a contrast point for hot working-set placement and
  contention reduction across CPU memory and durable storage.
- `reviewed` — **Polaris: Enabling Transaction Priority in Optimistic
  Concurrency Control**, Ye et al., PACMMOD/SIGMOD 2023.
  URL: `https://doi.org/10.1145/3588724`
  PDF: `https://chenhao-ye.github.io/publication/polaris/polaris.pdf`
  Why: priority-aware OCC cited by PreemptDB; relevant to combining request
  priority with conflict handling instead of only changing worker scheduling.
- `queued` — **Towards Optimal Transaction Scheduling**, Cheng et al.,
  PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf`
  Why: modern transaction scheduling work cited by PreemptDB; useful for
  contrasting non-preemptive priority ordering with interrupt-driven
  preemption and owner-queue admission.
- `queued` — **LibPreemptible: Enabling Fast, Adaptive, and
  Hardware-Assisted User-Space Scheduling**, Li et al., HPCA 2024.
  URL: `https://doi.org/10.1109/HPCA57654.2024.00075`
  Why: general hardware-assisted userspace preemption framework cited by
  PreemptDB; useful if GPU DB wants preemption mechanics outside a full
  transaction-engine rewrite.
- `queued` — **Flexible Resource Allocation for Relational
  Database-as-a-Service**, Arora et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p4202-narasayya.pdf`
  Why: modern DBaaS resource-allocation paper cited by Resource-Adaptive Query
  Execution; relevant to pricing or value-of-memory admission policies for
  multi-tenant/session-heavy GPU DB workloads.
- `queued` — **Robust External Hash Aggregation in the Solid State Age**,
  Kuiper, Boncz, and Muhleisen, ICDE 2024.
  URL: `https://doi.org/10.1109/ICDE60146.2024.00211`
  Why: DuckDB external aggregation work cited by Resource-Adaptive Query
  Execution; useful for paged intermediate state, spill-resistant aggregates,
  and over-resident query execution under bounded memory.
- `queued` — **Transaction Scheduling: From Conflicts to Runtime Conflicts**,
  Cao et al., SIGMOD 2023.
  URL: `https://doi.org/10.1145/3603164`
  Preprint:
  `https://www.research.ed.ac.uk/files/360117816/Transaction_Scheduling_CAO_DOA16082022_AFV.pdf`
  Why: modern transaction scheduling paper from SIGMOD 2023; relevant to
  contrasting conflict-graph scheduling with runtime resource conflicts and
  owner-queue admission for mixed GPU DB transaction classes.
- `queued` — **Improving Optimistic Concurrency Control through Transaction
  Batching and Operation Reordering**, Ding, Kot, and Gehrke, PVLDB 2018.
  URL: `https://doi.org/10.14778/3282495.3282502`
  PDF: `https://www.vldb.org/pvldb/vol12/p169-ding.pdf`
  Why: OCC batching and operation-reordering work; useful for deciding when
  GPU DB write admission should batch full transaction stages rather than only
  WAL, index, or GPU refresh substeps.
- `queued` — **Taurus: Lightweight Parallel Logging for In-Memory Database
  Management Systems**, Xia, Yu, Pavlo, and Devadas, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol14/p189-xia.pdf`
  arXiv: `https://arxiv.org/abs/2010.06760`
  Why: modern parallel logging with dependency vectors; useful follow-up for
  comparing explicit dependency encoding against RFA-style remote-flush
  avoidance in per-owner GPU DB WAL streams.
- `queued` — **Taurus Database: How to be Fast, Available, and Frugal in the
  Cloud**, Depoutovitch et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3386129`
  arXiv: `https://arxiv.org/abs/2412.02792`
  Why: cloud database storage architecture with append-only storage,
  replication, recovery, and constant-time snapshots; relevant to future
  cloud/disaggregated durability and snapshot tiers.
- `queued` — **Hybrid Garbage Collection for Multi-Version Concurrency Control
  in SAP HANA**, Lee et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915238`
  Why: production HTAP MVCC garbage-collection design contrasted with Steam;
  useful for evaluating interval GC, long transaction handling, and practical
  memory-pressure policies.
- `queued` — **Accelerating Hybrid Transactional/Analytical Processing Using
  Consistent Dual-Snapshot**, Li et al., DASFAA 2019.
  URL: `https://doi.org/10.1007/978-3-030-18576-3_41`
  Why: dual-snapshot HTAP design cited by Steam; relevant to separating
  retained analytical snapshots from fresh transactional visibility.
- `queued` — **Bao: Making Learned Query Optimization Practical**,
  Marcus et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3452838`
  Why: learned hint-based optimizer baseline compared by PAR2QO; useful for
  deciding whether GPU route tuning should learn bounded hints around a
  deterministic planner rather than replace route rules.
- `queued` — **TAS: TCP Acceleration as an OS Service**, Kaufmann et al.,
  EuroSys 2019.
  URL: `https://os.mpi-sws.org/projects/tas.html`
  PDF: `https://homes.cs.washington.edu/~arvind/papers/flextcp.pdf`
  Why: Shenango's related kernel-bypass/runtime context points to TAS as a
  multi-tenant TCP acceleration service; useful for comparing a central
  IO/runtime service against GPU DB's planned IO-worker and response-ring
  topology.
