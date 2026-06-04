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
- `queued` — **Concurrent query processing in a GPU-based database system**,
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
  heterogeneous DBMS designs.
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
- `queued` — **A Study of the Fundamental Performance Characteristics of GPUs
  and CPUs for Database Analytics**, Shanbhag, Yu, and Madden, SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3380595`
  Why: Crystal's tile-based execution model is the execution substrate used by
  the SIGMOD 2022 GPU compression paper; useful for separating compression
  effects from baseline GPU query operator and memory-traffic behavior.
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
- `queued` — **BinDex: A Two-Layered Index for Fast and Robust Scans**,
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
- `queued` — **AsyncFS: Metadata Updates Made Asynchronous for Distributed
  Filesystems with In-Network Coordination**, Zhou et al., arXiv 2024.
  URL: `https://arxiv.org/abs/2410.08618`
  Why: modern follow-up for metadata update admission and coordination;
  useful for evaluating whether cold-tier namespace, catalog, or route-cache
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
- `queued` — **G-Learned Index: Enabling Efficient Learned Index on GPU**,
  Liu et al., IEEE TPDS 2024.
  URL: `https://doi.org/10.1109/TPDS.2024.3384966`
  Why: modern follow-up to early GPU learned-index work; useful for comparing
  PGM-on-GPU with a more engineered GPU learned-index design before choosing a
  resident point-lookup index family.
- `queued` — **The PGM-index: a fully-dynamic compressed learned index with
  provable worst-case bounds**, Ferragina and Vinciguerra, PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p1162-ferragina.pdf`
  DOI: `https://doi.org/10.14778/3389133.3389135`
  Why: source design for the GPU-PGM paper; useful for understanding update,
  error-bound, and space guarantees before adapting a learned index to MVCC
  resident snapshots.
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
- `queued` — **Pangea: Monolithic Distributed Storage for Data Analytics**,
  Ghosh et al., PVLDB 2017.
  URL: `https://www.vldb.org/pvldb/vol10/p681-ghosh.pdf`
  Why: Tectonic's related work contrasts monolithic analytics storage with
  layered filesystem designs; useful for evaluating whether GPU DB cold-tier
  placement should centralize data placement, caching, and failure recovery or
  keep them as explicit route-owned tiers.
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
- `reviewed` — **Concurrency Control as a Service**, Zhou et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p2761-zhou.pdf`
  DOI: `https://doi.org/10.14778/3746405.3746406`
  Why: modern execution-CC-storage disaggregation with sharded multi-write OCC,
  asynchronous log push-down, and independently scalable conflict-resolution
  resources; directly relevant to mutation-owner decomposition and write-path
  admission.
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
- `queued` — **Adaptive Execution of Compiled Queries**, Kohn, Leis, and
  Neumann, ICDE 2018.
  URL: `https://doi.org/10.1109/ICDE.2018.00027`
  Why: Umbra's adaptive bytecode/JIT execution foundation; relevant to deciding
  when GPU DB should interpret, compile, batch, or route short SQL plans
  without paying excessive setup latency.
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
- `queued` — **NUMFabric: Fast and Flexible Bandwidth Allocation in
  Datacenters**, Nagaraj et al., SIGCOMM 2016.
  URL: `https://doi.org/10.1145/2934872.2934890`
  PDF: `https://web.stanford.edu/~skatti/pubs/sigcomm16-num.pdf`
  Why: PIFO cites NUMFabric as a flexible bandwidth-allocation use case; useful
  for comparing utility-driven admission and weighted fair queueing when GPU DB
  request classes compete for network, owner-ring, and accelerator capacity.
- `queued` — **HetExchange: Encapsulating Heterogeneous CPU-GPU Parallelism in
  JIT Compiled Engines**, Bress et al., CIDR 2019.
  URL: `https://www.cidrdb.org/cidr2019/papers/p59-bress-cidr19.pdf`
  Why: Fluid Co-processing contrasts fragment-level GPU assistance with
  exchange-style whole-pipeline routing; useful for comparing planner-time
  CPU/GPU placement with runtime split-route fallback.
- `queued` — **Performance-Optimal Filtering: Bloom Overtakes Cuckoo at High
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
- `reviewed` — **MRVs: Enforcing Numeric Invariants in Parallel Updates to
  Hotspots with Randomized Splitting**, Faria and Pereira, PACMMOD/SIGMOD 2023.
  URL: `https://doi.org/10.1145/3588723`
  Why: TiQuE cites MRVs as a way to reduce wasted work around optimistic
  hotspot updates; relevant to GPU DB write admission when many logical
  sessions update counters or bounded inventory-like values.
- `queued` — **Towards Generic Fine-Grained Transaction Isolation in
  Polystores**, Faria, Pereira, Alonso, and Vilaca, HDMS 2022.
  URL: `https://link.springer.com/chapter/10.1007/978-3-031-13216-2_6`
  Why: TiQuE cites this as an earlier layered-isolation direction for
  polystores; useful for future multi-engine GPU DB routes where transactional
  metadata may span CPU, GPU-resident, and cold-tier execution engines.
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
- `queued` — **FlexPushdownDB: Hybrid Pushdown and Caching in a Cloud DBMS**,
  Yang et al., PVLDB 2021.
  URL: `https://vldb.org/pvldb/vol14/p2101-yang.pdf`
  DOI: `https://doi.org/10.14778/3476249.3476265`
  Why: Hermes uses FlexPushdownDB as an AP engine; relevant to deciding which
  filtering, aggregation, and cache work should happen near storage, host
  memory, or GPU execution workers.
- `queued` — **L-Store: A Real-time OLTP and OLAP System**, Sadoghi et al.,
  EDBT 2018.
  URL: `https://arxiv.org/abs/1601.04084`
  Why: Mainlining Databases contrasts L-Store's lineage/tail-page architecture
  with relaxed Arrow blocks; useful for evaluating lineage-based staging,
  historic visibility, and lazy columnar consolidation for retained snapshots.
- `queued` — **Real-Time LSM-Trees for HTAP Workloads**, Saxena et al.,
  arXiv 2021.
  URL: `https://arxiv.org/abs/2101.06801`
  Why: lifecycle-aware LSM layout design is a follow-up to universal columnar
  and hot/cold block conversion; useful for comparing row-to-column movement
  by storage level rather than by in-memory block age.
- `queued` — **F1 Lightning: HTAP as a Service**, Yang et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p3313-yang.pdf`
  DOI: `https://doi.org/10.14778/3415478.3415553`
  Why: HATtrick classifies isolated/hybrid HTAP designs and evaluates TiDB-style
  split engines; F1 Lightning gives a production loose-coupling design for
  fresh analytical copies, CDC, compaction, and federated query integration.
- `queued` — **OLxPBench: Real-time, Semantically Consistent, and
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
- `queued` — **BtrBlocks: Efficient Columnar Compression for Data Lakes**,
  Kuschewski et al., SIGMOD 2023.
  URL: `https://doi.org/10.1145/3589265`
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
- `queued` — **HPCC: High Precision Congestion Control**, Li et al.,
  SIGCOMM 2019.
  URL: `https://doi.org/10.1145/3341302.3342085`
  Why: PowerTCP builds on HPCC-style in-band network telemetry; relevant to
  whether GPU DB should export precise per-boundary service telemetry to
  schedulers instead of relying on coarse queue depths.
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
- `queued` — **RackSched: A Microsecond-Scale Scheduler for
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
- `queued` — **SKQ: Event Scheduling for Optimizing Tail Latency in a
  Traditional OS Kernel**, Zhao, Gu, and Mashtizadeh, USENIX ATC 2021.
  URL: `https://www.usenix.org/conference/atc21/presentation/zhao-siyao`
  PDF: `https://www.usenix.org/system/files/atc21-zhao.pdf`
  Why: LibPreemptible contrasts against traditional-kernel event scheduling;
  useful for deciding how much GPU DB can improve pgwire/event-loop tail
  latency through event prioritization and delivery control before adopting
  hardware-assisted preemption or kernel bypass.
- `queued` — **Homa: A Receiver-Driven Low-Latency Transport Protocol Using
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
- `queued` — **BtrBlocks: Efficient Columnar Compression for Data Lakes**,
  Kuschewski, Sauerwein, Alhomssi, and Leis, SIGMOD 2023.
  URL: `https://doi.org/10.1145/3589263`
  Why: modern columnar compression framework referenced by the compressed GPU
  analytics paper; useful for choosing host/cold-tier column encodings before
  deciding which forms are worth promoting into GPU-resident snapshots.
- `queued` — **Improving Execution Efficiency of Just-in-Time Compilation
  Based Query Processing on GPUs**, Paul et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol14/p202-paul.pdf`
  Why: Pyper baseline for Themis, with intra-warp shuffle and redistribution
  mechanics relevant to retained GPU pipeline fusion.
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
- `queued` — **Carousel: Low-Latency Transaction Processing for
  Globally-Distributed Data**, Yan et al., SIGMOD 2018.
  URL: `https://doi.org/10.1145/3183713.3196912`
  PDF: `https://www.cs.cornell.edu/~hongbo/files/carousel-sigmod-2018.pdf`
  Why: Natto's base protocol; useful for evaluating fixed-set interactive
  transactions that overlap read/prepare, commit, and replication phases,
  which maps to GPU DB route descriptors with predeclared read/write sets.

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
- `queued` — **Guaranteeing Recoverability via Partially Constrained
  Transaction Logs**, Guo et al., arXiv 2019.
  URL: `https://arxiv.org/abs/1901.06491`
  Why: Poplar-style partial log ordering tracks RAW/WAW dependencies instead of
  forcing one serial LSN stream; useful follow-up for per-owner GPU DB WAL
  streams and parallel crash recovery.
- `queued` — **Border-Collie: A Wait-free, Read-optimal Algorithm for
  Database Logging on Multicore Hardware**, Kim et al., SIGMOD 2019.
  URL: `https://doi.org/10.1145/3299869.3319869`
  Why: multicore logging algorithm cited by Taurus; relevant to minimizing
  reader-side coordination and cache coherence in the WAL publication path.
- `queued` — **ExpressPass: End-to-End Credit-Based Congestion Control for
  Datacenters**, Cho et al., SIGCOMM 2017.
  URL: `https://doi.org/10.1145/3098822.3098843`
  PDF: `https://conferences.sigcomm.org/sigcomm/2017/files/program-ccr-final/91-Cho.pdf`
  Why: FNCC contrasts against credit-based congestion avoidance; useful for
  evaluating whether GPU DB response rings should use receiver-issued credits
  rather than only reactive queue-delay backpressure.
- `queued` — **Taurus Database: How to be Fast, Available, and Frugal in the
  Cloud**, Depoutovitch et al., SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3386129`
  arXiv: `https://arxiv.org/abs/2412.02792`
  Why: cloud database storage architecture with append-only storage,
  replication, recovery, and constant-time snapshots; relevant to future
  cloud/disaggregated durability and snapshot tiers.
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
- `queued` — **Bao: Making Learned Query Optimization Practical**,
  Marcus et al., SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3452838`
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
- `queued` — **Scheduling OLTP Transactions via Learned Abort Prediction**,
  Sheng, Tomasic, Zhang, and Pavlo, aiDM 2019.
  URL: `https://doi.org/10.1145/3329859.3329871`
  Why: lightweight learned transaction-to-thread assignment cited by TSkd;
  relevant to admission-time prediction before choosing an owner, CPU route,
  or deferred execution path.
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
- `queued` — **Detecting Robustness against MVRC for Transaction Programs
  with Predicate Reads**, Vandevoort et al., EDBT 2023.
  URL: `https://doi.org/10.48786/edbt.2023.47`
  Why: read-promotion paper cites this as a direction for transaction programs
  with predicate reads; relevant to GPU DB route templates that include ranges,
  prefix predicates, and phantom-sensitive retained scans.
- `queued` — **Allocating Isolation Levels to Transactions in a Multiversion
  Setting**, Vandevoort, Ketsman, and Neven, PODS 2023.
  URL: `https://doi.org/10.1145/3584372.3588672`
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
- `queued` — **Robustness against Read Committed for Transaction Templates**,
  Vandevoort et al., PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p2141-vandevoort.pdf`
  DOI: `https://doi.org/10.14778/3476249.3476268`
  Why: transaction-template robustness and selective read-promotion baseline;
  useful for benchmarking whether known OLTP command shapes can safely use a
  cheaper RC-style route while preserving serializable outcomes.
- `queued` — **Robustness Against Read Committed for Transaction Templates with
  Functional Constraints**, Vandevoort et al., ICDT 2022.
  URL: `https://arxiv.org/abs/2201.05021`
  Why: extends robustness analysis with functional constraints; useful for GPU
  DB templates where primary keys, unique constraints, and derived keys may
  prove that cheaper route-level isolation is still safe.
- `queued` — **Detock: High Performance Multi-region Transactions at Scale**,
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
- `queued` — **Larger-than-Memory Data Management on Modern Storage Hardware
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
- `queued` — **Page As You Go: Piecewise Columnar Access In SAP HANA**,
  Sherkat et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2903734`
  Why: production columnar cold-block access design cited by LeanStore; useful
  for deciding whether GPU DB should page full resident segments, column
  groups, or smaller compressed blocks.
- `reviewed` — **TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory**,
  Al Maruf et al., ASPLOS 2023.
  URL: `https://doi.org/10.1145/3582016.3582063`
  PDF: `https://symbioticlab.org/publications/files/tpp%3Aasplos23/tpp-asplos23.pdf`
  Why: Linux CXL transparent page placement baseline compared by NOMAD;
  useful for deciding where OS-managed promotion/demotion is enough and where
  GPU DB needs explicit DBMS placement handles.
- `queued` — **TAOBench: An End-to-End Benchmark for Social Network
  Workloads**, Cheng et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p1965-cheng.pdf`
  Why: DeToX's most important real-world workload source; useful for a
  session-heavy transactional cache/residency benchmark with correlated
  point reads, read transactions, writes, skew, and contaminated hot keys.
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
- `queued` — **SplinterDB: Closing the Bandwidth Gap for NVMe Key-Value
  Stores**, Conway et al., USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/conway`
  Why: Hyperion's argument depends on extracting cheap NVMe bandwidth;
  SplinterDB is a storage-engine follow-up for comparing write-optimized
  indexing, compaction, and bandwidth utilization against GPU DB cold-tier
  point lookup and segment-directory designs.
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
- `queued` — **Mako: Speculative Distributed Transactions with
  Geo-Replication**, Shen et al., OSDI 2025.
  URL: `https://www.usenix.org/conference/osdi25/presentation/shen-weihai`
  PDF: `https://www.usenix.org/system/files/osdi25-shen-weihai.pdf`
  Why: Minerva contrasts against Mako's speculative geo-replicated
  transaction path; useful for comparing execution/replication decoupling,
  deterministic replay, and geo-replication costs against owner-local GPU DB
  WAL and snapshot publication.
- `queued` — **A Hybrid Approach to Integrating Deterministic and
  Non-Deterministic Concurrency Control in Database Systems**, Hong et al.,
  PVLDB 2025.
  URL: `https://dblp.org/rec/journals/pvldb/HongZLDCPZ25`
  Why: Minerva relates this HDCC line to Aria-style OCC plus deterministic
  rescheduling; useful for deciding when GPU DB should switch from optimistic
  validation to deterministic owner execution under high contention.
- `queued` — **Epoch-Based Commit and Replication in Distributed OLTP
  Databases**, Lu et al., PVLDB 2021.
  URL: `https://www.vldb.org/pvldb/vol14/p743-lu.pdf`
  Why: COCO is Minerva's epoch-commit baseline; useful for comparing
  epoch-sized commit, replication, and validation units with GPU DB
  WAL-before-visibility batches and retained snapshot publication.
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
- `queued` — **RackSched: A Microsecond-Scale Scheduler for Rack-Scale
  Computers**, Sreekanti et al., arXiv 2020.
  URL: `https://arxiv.org/abs/2010.05969`
  Why: Shinjuku follow-up direction for rack-scale request scheduling;
  relevant to comparing centralized, partitioned, and rack-aware admission
  when GPU DB eventually spans multiple owners, devices, or nodes.
- `queued` — **SLOG: Serializable, Low-latency, Geo-replicated
  Transactions**, Ren et al., PVLDB 2019.
  URL: `https://www.vldb.org/pvldb/vol12/p1747-ren.pdf`
  Why: deterministic-transaction follow-up direction related to Aria's
  replication motivation; useful for comparing input replication,
  deterministic ordering, and low-latency commit paths when GPU DB eventually
  separates local owner domains from replicated durability.
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
- `queued` — **Towards an Adaptable Systems Architecture for Memory Tiering at
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
- `queued` — **Pond: CXL-Based Memory Pooling Systems for Cloud Platforms**,
  Li et al., ASPLOS 2023.
  URL: `https://doi.org/10.1145/3575693.3578835`
  Why: MEMTIS uses CXL latency assumptions from Pond; useful for future CXL
  memory-pool tiers, remote-memory latency budgets, and explicit placement
  boundaries between local DRAM, pooled memory, and GPU-resident state.
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
- `queued` — **Zero-Shot Cost Models for Out-of-the-box Learned Cost
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
- `queued` — **Homa: A Receiver-Driven Low-Latency Transport Protocol Using
  Network Priorities**, Montazeri et al., SIGCOMM 2018.
  URL: `https://doi.org/10.1145/3230543.3230564`
  arXiv: `https://arxiv.org/abs/1803.09615`
  Why: receiver-driven short-message transport and priority scheduling for
  datacenter RPCs; relevant to future GPU DB network admission and tail-latency
  budgeting once pgwire sessions are multiplexed over fewer IO workers.
- `queued` — **Frequent Background Polling on a Shared Thread, Using
  Lightweight Compiler Interrupts**, Basu, Montanari, and Eriksson, PLDI 2021.
  URL: `https://doi.org/10.1145/3453483.3454049`
  Why: Concord compares against this compiler-instrumented polling approach;
  useful for deciding whether GPU DB should use explicit cooperative yield
  probes, request-budget probes, or cheaper route-local cancellation checks
  around long scans, refresh jobs, and mutation batches.
- `queued` — **RPCValet: NI-Driven Tail-Aware Balancing of microsecond-scale
  RPCs**, Sutherland et al., ASPLOS 2019.
  URL: `https://doi.org/10.1145/3297858.3304050`
  Why: Concord cites RPCValet as a JBSQ-style dispatcher placement point;
  useful for comparing CPU-owned IO-worker scheduling with NIC-assisted
  request steering, bounded per-worker queues, and response-path priorities.
- `queued` — **Releasing Locks As Early As You Can: Reducing Contention of
  Hotspots by Violating Two-Phase Locking**, Guo, Wu, Yan, and Yu,
  SIGMOD 2021.
  URL: `https://doi.org/10.1145/3448016.3457294`
  Why: Bamboo/Wound-Retire is the direct baseline improved by
  Rebirth-Retire; useful for comparing active lock retirement, dirty
  dependency tracking, and hotspot write admission before adopting a
  passive-retire variant.
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
- `queued` — **Serval: A Wait-free Multi-version Deterministic Concurrency
  Control Scheme**, Li, Onishi, and Kawashima, CANDAR 2024.
  URL: `https://doi.org/10.1109/CANDAR64496.2024.00028`
  Why: Caracal follow-up that replaces global version-array latch pressure for
  contended rows with bitmaps and dynamic local version arrays; useful for
  deciding whether GPU DB write batches should keep per-owner local version
  arrays before publishing a merged visibility front.
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
- `queued` — **PRICE: A Pretrained Model for Cross-Database Cardinality
  Estimation**, Zeng et al., arXiv 2024.
  URL: `https://arxiv.org/abs/2406.01027`
  Why: cross-database cardinality estimation is the counterpart to DACE's
  residual-cost path; useful for deciding whether GPU route choice should keep
  cardinality and residual-latency learning as separate planner signals.
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
- `queued` — **D-RDMA: Bringing Zero-Copy RDMA to Database Systems**,
  Ryser, Lerner, Forencich, and Cudre-Mauroux, CIDR 2022.
  URL: `https://www.cidrdb.org/cidr2022/papers/p6-ryser.pdf`
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
- `queued` — **Efficiently Making (Almost) Any Concurrency Control Mechanism
  Serializable**, Wang, Johnson, Fekete, and Pandis, VLDB Journal 2017.
  URL: `https://doi.org/10.1007/s00778-017-0463-8`
  arXiv: `https://arxiv.org/abs/1605.04292`
  Why: ERMIA uses Serial Safety Net as its serializability certifier; useful
  for deciding whether GPU DB can layer bounded dependency validation over
  snapshot-friendly read execution without falling back to pessimistic locks.
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
- `queued` — **Jiffy: A Lock-Free Skip List with Batch Updates and
  Snapshots**, Kobus, Kokocinski, and Wojciechowski, PPoPP 2022.
  URL: `https://doi.org/10.1145/3503221.3508437`
  Why: cited by the MVGC paper as a modern multiversion/snapshot data structure;
  useful for comparing batched update publication and wait-free range snapshot
  support against GPU DB retained-read generations.
- `queued` — **MV-RLU: Scaling Read-Log-Update with Multi-Versioning**,
  Kim et al., ASPLOS 2019.
  URL: `https://doi.org/10.1145/3297858.3304040`
  Why: cited by the MVGC paper as a practical multiversioning system; useful
  for contrasting reader-side logging, version lifetime, and reclamation costs
  with GPU DB MVCC chains and long retained snapshots.
- `queued` — **Constant-Time Snapshots with Applications to Concurrent Data
  Structures**, Wei et al., PPoPP 2021.
  URL: `https://arxiv.org/abs/2007.02372`
  Why: the bounded MVGC paper applies its collector to this versioned-CAS
  snapshot framework; useful for deciding whether GPU DB should expose
  retained snapshot handles over lock-free CPU data structures before or
  alongside SQL-facing MVCC chains.
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
- `queued` — **Design Principles for Scaling Multi-core OLTP Under High
  Contention**, Ren et al., SIGMOD 2016 / arXiv 2015.
  URL: `https://arxiv.org/abs/1512.06168`
  Why: ORTHRUS-style separation of transaction execution stages and advanced
  transaction planning is a direct follow-up for FOEDUS's many-core OCC
  scaling limits under high contention, and may inform GPU DB mutation-owner
  admission and partitioned write lanes.
- `queued` — **SplinterDB: Closing the Bandwidth Gap for NVMe Key-Value
  Stores**, Conway et al., USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/conway`
  PDF: `https://www.usenix.org/system/files/atc20-conway.pdf`
  Why: PrismDB cites SplinterDB as an NVMe-specialized KV-store comparison;
  its STB-epsilon-tree, concurrent cache, and reduced write amplification are
  relevant to CPU/NVMe tier limits before GPU resident refresh.
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
- `queued` — **Native Store Extension for SAP HANA**, Sherkat et al.,
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
- `queued` — **GeminiFS: A Companion File System for GPUs**, Qiu et al.,
  FAST 2025.
  URL: `https://www.usenix.org/conference/fast25/presentation/qiu`
  PDF: `https://www.usenix.org/system/files/fast25-qiu.pdf`
  Why: modern GPU-facing storage interface that cites GMT; useful for comparing
  file-system-level GPU IO services with DB-owned GPU/host/NVMe tier managers.
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
- `queued` — **LiquidCache: Efficient Pushdown Caching for Cloud-Native Data
  Analytics**, Hao et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p5662-hao.pdf`
  Why: modern cache-placement follow-up from the same tiering research area;
  useful for comparing DB-owned cache admission and pushdown placement with
  GPU DB resident, host, and cold-tier policies.
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
- `queued` — **Ultra Ethernet's Design Principles and Architectural
  Innovations**, Hoefler et al., arXiv 2025.
  URL: `https://arxiv.org/abs/2508.08906`
  Why: SMaRTT positions itself as the basis for UEC NSCC; the broader UEC
  design may inform future GPU DB transport assumptions, multipath routing,
  out-of-order placement, and packet-trimming availability.
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
- `queued` — **BCC: Reducing False Aborts in Optimistic Concurrency Control
  with Low Cost for In-Memory Databases**, Yuan et al., PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p504-yuan.pdf`
  DOI: `https://doi.org/10.14778/2904121.2904126`
  Why: AOCC cites BCC as a low-overhead false-abort reduction baseline; useful
  for GPU DB contention handling where serializable write lanes should avoid
  unnecessary aborts without weakening visibility guarantees.
- `queued` — **O|R|P|E - A Data Semantics Driven Concurrency Control**,
  Hemm et al., arXiv 2023.
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
- `queued` — **Workload Placement on Heterogeneous CPU-GPU Systems**,
  Carvalho, Simitsis, Queralt, and Romero, PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p4241-carvalho.pdf`
  DOI: `https://doi.org/10.14778/3685800.3685845`
  Why: modern tutorial and taxonomy for CPU/GPU placement strategies, cost
  prediction, placement granularity, and code-management choices; useful for
  turning HetCache-style access-path hints into a broader route-placement
  contract.
- `queued` — **Adaptive Compression for Databases**, Windheuser et al.,
  EDBT 2024.
  URL: `https://doi.org/10.48786/EDBT.2024.13`
  Why: GOLAP cites adaptive compression of cold column sections; useful for
  deciding whether GPU DB cold/warm segments should choose compression
  parameters from access statistics instead of a single fixed resident/cold
  format.
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
- `queued` — **Aria: A Fast and Practical Deterministic OLTP Database**,
  Lu et al., PVLDB 2020.
  URL: `https://www.vldb.org/pvldb/vol13/p2047-lu.pdf`
  DOI: `https://doi.org/10.14778/3407790.3407808`
  Why: Epic compares against Aria's deterministic abort/fallback strategy;
  useful for deciding when GPU DB should prefer deterministic rerun,
  lock-based fallback, or owner-serialized execution for mispredicted
  read/write sets.
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
- `queued` — **Data Blocks: Hybrid OLTP and OLAP on Compressed Storage Using
  Both Vectorization and Compilation**, Lang et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2882925`
  Why: FastLanes cites it as compressed execution context; useful for deciding
  whether GPU DB should keep one compressed storage representation that serves
  OLTP lookups, retained scans, and compiled/vectorized operators.
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
- `queued` — **Hybrid Deterministic and Nondeterministic Execution of
  Transactions in Actor Systems**, Liu et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3526172`
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
- `queued` — **X-SSD: A Storage System with Native Support for Database Logging
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
- `queued` — **ALECE: An Attention-based Learned Cardinality Estimator for SPJ
  Queries on Dynamic Workloads**, Li et al., PVLDB 2023.
  URL: `https://arxiv.org/abs/2310.05349`
  Why: modern learned cardinality estimator for dynamic workloads; useful for
  comparing data-update-aware route estimates against simpler DBMS-statistics
  correction and explicit route telemetry.
- `queued` — **CardOOD: Robust Query-driven Cardinality Estimation under
  Out-of-Distribution Workloads**, 2024.
  URL: `https://arxiv.org/abs/2412.05864`
  Why: direct follow-up on out-of-distribution robustness for query-driven
  cardinality estimation; useful for GPU DB when tenant workloads, resident
  cache contents, or mixed CPU/GPU route families drift from training logs.
- `queued` — **Buffer Pool Aware Query Scheduling via Deep Reinforcement
  Learning**, Zhang et al., AIDB@VLDB 2020.
  URL:
  `https://drive.google.com/file/d/1trNYAcQ3S71SHu5dbtkBR2hjcK-dIHt21c-/view`
  Why: Stage identifies buffer-pool and cache state as hard-to-featurize
  environment factors; useful for comparing learned scheduling with explicit
  GPU/host/NVMe residency telemetry and cache-aware admission.
- `queued` — **Auto-WLM: Machine Learning Enhanced Workload Management in
  Amazon Redshift**, Saxena et al., SIGMOD Companion 2023.
  URL: `https://doi.org/10.1145/3555041.3589677`
  Why: Stage compares against Redshift's prior workload-manager predictor;
  useful for understanding the production queue, priority, concurrency-scaling,
  and resource-control hooks that a GPU DB route predictor would influence.
- `queued` — **Deferred Runtime Pipelining for Contentious Multicore Software
  Transactions**, Mu, Angel, and Shasha, EuroSys 2019.
  URL: `https://doi.org/10.1145/3302424.3303966`
  PDF: `https://www.cis.upenn.edu/~sga001/papers/drp-eurosys19.pdf`
  Why: Opportunities for Optimism contrasts manual commit-time updates with
  DRP's lazy/deferred execution; useful for comparing automatic transaction
  chopping against explicit GPU DB route-shape annotations for hot writes.
