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
- `queued` — **Design Choices in Low-Latency C++ Systems: Empirical Insights
  With Applications to High-Frequency Trading**, SSRN 2026.
  URL: `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6513601`
  Why: modern HFT-oriented low-latency systems survey; useful for queue,
  allocation, cache, and thread-pinning patterns.

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
- `reviewed` — **Themis: A GPU-accelerated Relational Query Execution Engine**,
  Hong et al., PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol18/p426-han.pdf`
  DOI: `https://doi.org/10.14778/3705829.3705856`
  Why: modern GPU relational engine with execution and load-balancing details
  relevant to fused retained route design.
- `queued` — **GPU Acceleration of SQL Analytics on Compressed Data**,
  Huang et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol19/p320-huang.pdf`
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
- `queued` — **Vortex: Overcoming Memory Capacity Limitations in
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
- `queued` — **Cloud-Native Database Systems and Unikernels: Reimagining OS
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
- `queued` — **Predicate Transfer: Efficient Pre-Filtering on Multi-Join
  Queries**, Yang et al., CIDR 2024.
  URL: `https://www.cidrdb.org/cidr2024/papers/p22-yang.pdf`
  Why: modern predicate-transfer/pre-filtering work cited by the 2025 hybrid
  CPU-GPU paper; relevant to reducing over-resident transfer before GPU joins.
- `reviewed` — **Orchestrating data placement and query execution in
  heterogeneous CPU-GPU DBMS**, Yogatama et al., PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p2491-yogatama.pdf`
  Why: cost-based CPU/GPU placement and execution orchestration for
  heterogeneous DBMS designs.
- `queued` — **Accelerating GPU Data Processing using FastLanes
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
- `queued` — **Exploiting Directly-Attached NVMe Arrays in DBMS**, Haas,
  Haubenschild, and Leis, CIDR 2020.
  URL: `https://www.cidrdb.org/cidr2020/papers/p16-haas-cidr20.pdf`
  Why: direct follow-up for explicit NVMe tier economics and high-parallelism
  IO paths that should inform GPU DB cold-partition and over-resident
  placement benchmarks.
- `queued` — **Optimizing Memory-mapped I/O for Fast Storage Devices**,
  Papagiannis et al., USENIX ATC 2020.
  URL: `https://www.usenix.org/conference/atc20/presentation/papagiannis`
  PDF: `https://www.usenix.org/system/files/atc20-papagiannis.pdf`
  Why: OS-level mmap scalability work cited by the CIDR 2022 mmap paper; useful
  as a contrasting source on whether modified mmap paths can ever be safe or
  fast enough for GPU DB cold-tier experiments.
- `queued` — **Leveraging Lock Contention to Improve OLTP Application
  Performance**, Yan and Cheung, PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol9/p444-yan.pdf`
  Why: program-analysis and contention-aware execution ideas that complement
  MV3C's dependency-annotated transaction repair path.
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
- `queued` — **OLTP Through the Looking Glass 16 Years Later:
  Communication is the New Bottleneck**, Zhou et al., CIDR 2025.
  URL:
  `https://www.vldb.org/cidrdb/2025/oltp-through-the-looking-glass-16-years-later-communication-is-the-new-bottleneck.html`
  PDF: `https://www.vldb.org/cidrdb/papers/2025/p17-zhou.pdf`
  Why: modern whole-stack OLTP breakdown showing communication and isolation
  costs as dominant bottlenecks; directly relevant to pgwire/session
  admission, stored-procedure boundaries, and owner/runtime queue design.
- `queued` — **Fast Failure Recovery for Main-Memory DBMSs on Multicores**,
  Zheng et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915239`
  Why: Cicada cites parallel durability and recovery as the path for scalable
  logging/checkpointing; useful for GPU DB WAL replay, checkpoint rebuild, and
  post-crash CPU/GPU cache warmup design.
- `queued` — **RUMA has it: Rewired User-space Memory Access is Possible!**,
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
- `queued` — **FOEDUS: OLTP Engine for a Thousand Cores and NVRAM**,
  Kimura, SIGMOD 2015.
  URL: `https://dl.acm.org/doi/10.1145/2723372.2746480`
  Tech report: `https://www.labs.hpe.com/techreports/2015/HPL-2015-37.pdf`
  Why: many-core OLTP and NVRAM-oriented storage architecture cited by TicToc;
  relevant to partition ownership, logging, NUMA locality, and future tiers.
- `queued` — **Diva: Making MVCC Systems HTAP-Friendly**, Kim et al.,
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
- `reviewed` — **Nomad: Non-Exclusive Memory Tiering via Transactional Page
  Migration**, Xiang et al., OSDI 2024.
  URL: `https://www.usenix.org/conference/osdi24/presentation/xiang`
  Why: transactional page migration for tiered memory cited by vmcache^n;
  useful for comparing OS-assisted migration against explicit DBMS ownership.
- `queued` — **Towards Buffer Management with Tiered Main Memory**, Hao et al.,
  PACMMOD/SIGMOD 2024.
  URL: `https://doi.org/10.1145/3639286`
  Why: modern tiered-main-memory buffer management cited by vmcache^n;
  relevant to DRAM/remote-memory/NVMe policy design and placement economics.
- `reviewed` — **PAR2QO: Parametric Penalty-Aware Robust Query Optimization**,
  Xiu et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p4532-xiu.pdf`
  DOI: `https://doi.org/10.14778/3749646.3749711`
  Why: follow-up to PARQO that focuses on parametric robust query
  optimization and plan-penalty profile caching; relevant to repeated retained
  GPU route templates and admission-time route reuse.
- `reviewed` — **Hints for Robust Query Performance Tuning**, Xiu et al.,
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
- `reviewed` — **Plor: General Transactions with Predictable, Low Tail
  Latency**, Chen et al., SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517879`
  PDF: `https://storage.cs.tsinghua.edu.cn/papers/sigmod22plor.pdf/`
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
- `reviewed` — **GaccO - A GPU-accelerated OLTP DBMS**, Boeschen and Binnig,
  SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517876`
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
- `queued` — **Cost/Performance in Modern Data Stores: How Data Caching
  Systems Succeed**, Lomet, DaMoN 2018.
  URL: `https://doi.org/10.1145/3211922.3211931`
  Why: economic and architectural argument cited by Umbra against pure
  in-memory-only systems; useful for setting GPU DB tier-placement budgets and
  cost/performance targets across HBM, DRAM, NVMe, and future memory tiers.
- `queued` — **Adaptive Execution of Compiled Queries**, Kohn, Leis, and
  Neumann, ICDE 2018.
  URL: `https://doi.org/10.1109/ICDE.2018.00027`
  Why: Umbra's adaptive bytecode/JIT execution foundation; relevant to deciding
  when GPU DB should interpret, compile, batch, or route short SQL plans
  without paying excessive setup latency.
- `reviewed` — **Polaris: Enabling Transaction Priority in Optimistic
  Concurrency Control**, Ye et al., PACMMOD/SIGMOD 2023.
  URL: `https://doi.org/10.1145/3588724`
  PDF: `https://chenhao-ye.github.io/publication/polaris/polaris.pdf`
  Why: priority-aware OCC cited by PreemptDB; relevant to combining request
  priority with conflict handling instead of only changing worker scheduling.
- `reviewed` — **Towards Optimal Transaction Scheduling**, Cheng et al.,
  PVLDB 2024.
  URL: `https://www.vldb.org/pvldb/vol17/p2694-cheng.pdf`
  Why: modern transaction scheduling work cited by PreemptDB; useful for
  contrasting non-preemptive priority ordering with interrupt-driven
  preemption and owner-queue admission.

- `queued` — **Harnessing GPU Power for Enhanced OLTP: A Study in Concurrency
  Control Schemes**, arXiv 2024.
  URL: `https://arxiv.org/abs/2406.10158`
  Why: modern GPU OLTP concurrency-control evaluation that compares GPU-adapted
  2PL, timestamp ordering, MVCC, OCC, GPUTx, and GaccO-style conflict-graph or
  deterministic locking schemes; useful follow-up for deciding which GPU write
  batch protocol is benchmark-worthy.
- `queued` — **LibPreemptible: Enabling Fast, Adaptive, and
  Hardware-Assisted User-Space Scheduling**, Li et al., HPCA 2024.
  URL: `https://doi.org/10.1109/HPCA57654.2024.00075`
  Why: general hardware-assisted userspace preemption framework cited by
  PreemptDB; useful if GPU DB wants preemption mechanics outside a full
  transaction-engine rewrite.
- `queued` — **Skyloft: A General High-Efficient Scheduling Framework in
  User Space**, Jia et al., SOSP 2024.
  URL:
  `https://madsys.cs.tsinghua.edu.cn/publication/skyloft-a-general-high-efficient-scheduling-framework-in-user-space/SOSP24-Jia.pdf`
  DOI: `https://doi.org/10.1145/3694715.3695973`
  Why: modern user-space scheduling framework with user-mode-interrupt
  preemption and DPDK integration; useful as a follow-up to Arachne,
  Shenango, and Shinjuku for deciding whether GPU DB needs preemptive
  user-space request classes around long scans and short retained reads.
- `queued` — **Flexible Resource Allocation for Relational
  Database-as-a-Service**, Arora et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p4202-narasayya.pdf`
  Why: modern DBaaS resource-allocation paper cited by Resource-Adaptive Query
  Execution; relevant to pricing or value-of-memory admission policies for
  multi-tenant/session-heavy GPU DB workloads.
- `reviewed` — **Bonspiel: Low Tail Latency Transactions in
  Geo-Distributed Databases**, Cui et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p3840-cui.pdf`
  Why: modern low-tail transaction follow-up that cites Plor; relevant to
  predictable transaction admission and retry behavior under high contention.
- `reviewed` — **Rebirth-Retire: A Concurrency Control Protocol Adaptable to
  Different Levels of Contention**, Zhang et al., PVLDB 2025.
  URL: `https://www.vldb.org/pvldb/vol18/p3162-zhang.pdf`
  Why: modern MVCC/concurrency-control work that discusses Plor and adapts to
  changing workload conditions; useful for deciding when GPU DB should switch
  conflict policy by route, contention, or transaction size.
- `queued` — **Robust External Hash Aggregation in the Solid State Age**,
  Kuiper, Boncz, and Muhleisen, ICDE 2024.
  URL: `https://doi.org/10.1109/ICDE60146.2024.00211`
  Why: DuckDB external aggregation work cited by Resource-Adaptive Query
  Execution; useful for paged intermediate state, spill-resistant aggregates,
  and over-resident query execution under bounded memory.
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
- `queued` — **GPU Orchestrated Memory Tiering**, Bae et al., 2024.
  URL: `https://doi.org/10.1145/3620666.3651341`
  Why: three-tier GPU/CPU/storage cache approach summarized by Torp et al.;
  relevant to future explicit tier promotion and demotion policies when reuse
  can justify CPU and GPU resource use.
- `queued` — **DRAGON: Breaking GPU Memory Capacity Limits with Direct NVM
  Access**, Markthub et al., SC 2018.
  URL: `https://doi.org/10.1109/SC.2018.00035`
  PDF: `https://www.osti.gov/servlets/purl/1489577`
  Why: BaM contrasts against DRAGON's UVM/page-fault path; useful for comparing
  transparent GPU page-fault extension against explicit GPU-initiated queues
  and cache-line admission.
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
- `queued` — **Caerus: Low-Latency Distributed Transactions for
  Geo-Replicated Systems**, Hildred et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol17/p469-hildred.pdf`
  Why: modern geo-replicated transaction protocol cited by Bonspiel; useful as
  a contrast for placement-aware commit and low-latency multi-partition
  transaction routing.
- `queued` — **Natto: Providing Distributed Transaction Prioritization for
  High-Contention Workloads**, Yang, Yan, and Wong, SIGMOD 2022.
  URL: `https://doi.org/10.1145/3514221.3517869`
  Why: priority-based distributed transaction handling cited by Bonspiel;
  relevant to deciding whether GPU DB should prioritize long/remote or
  expensive route classes without wounding short local work.
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
- `queued` — **Adaptive logging: Optimizing logging and recovery costs in
  distributed in-memory databases**, Yao et al., SIGMOD 2016.
  URL: `https://doi.org/10.1145/2882903.2915221`
  Why: distributed in-memory command/data logging tradeoff cited by Taurus;
  useful for deciding whether GPU DB should vary log payloads and recovery
  strategy by transaction class or partition.
- `queued` — **Border-Collie: A Wait-free, Read-optimal Algorithm for
  Database Logging on Multicore Hardware**, Kim et al., SIGMOD 2019.
  URL: `https://doi.org/10.1145/3299869.3319869`
  Why: multicore logging algorithm cited by Taurus; relevant to minimizing
  reader-side coordination and cache coherence in the WAL publication path.
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
- `queued` — **Hyperion: Co-Optimizing SSD Access and GPU Computation for
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
- `queued` — **A CXL-Powered Database System: Opportunities and Challenges**,
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
- `queued` — **Handling Highly Contended OLTP Workloads Using Fast Dynamic
  Partitioning**, Prasaad, Cheung, and Suciu, SIGMOD 2020.
  URL: `https://doi.org/10.1145/3318464.3389708`
  Why: Strife is a key partitioner baseline used by the runtime-conflict
  scheduler; useful for hot-key partitioning, residual transaction handling,
  and contention-aware owner assignment.
- `queued` — **Scheduling OLTP Transactions via Learned Abort Prediction**,
  Sheng, Tomasic, Zhang, and Pavlo, aiDM 2019.
  URL: `https://doi.org/10.1145/3329859.3329871`
  Why: lightweight learned transaction-to-thread assignment cited by TSkd;
  relevant to admission-time prediction before choosing an owner, CPU route,
  or deferred execution path.
- `queued` — **Polyjuice: High-Performance Transactions via Learned
  Concurrency Control**, Wang et al., OSDI 2021.
  URL: `https://www.usenix.org/conference/osdi21/presentation/wang-jiachen`
  Why: learned concurrency-control policy selection cited by TSkd; useful as
  a contrast to deterministic owner-queue rules and runtime-conflict telemetry.
- `queued` — **Centiman: Elastic, High Performance Optimistic Concurrency
  Control by Watermarking**, Ding et al., SoCC 2015.
  URL: `https://doi.org/10.1145/2806777.2806842`
  Why: OCC validator/storage architecture cited by the batching paper; useful
  for comparing watermark-based validation, decoupled compute/storage, and
  versioned write installation against GPU DB owner boundaries.
- `queued` — **QueCC: A Queue-Oriented, Control-Free Concurrency
  Architecture**, Qadah and Sadoghi, Middleware 2018.
  URL: `https://doi.org/10.1145/3274808.3274820`
  Why: queue-oriented transaction execution cited by the batching paper;
  relevant to deterministic owner queues, queue-local ordering, and whether
  control-free execution can coexist with WAL-before-visibility.
- `queued` — **Mostly-Optimistic Concurrency Control for Highly Contended
  Dynamic Workloads on a Thousand Cores**, Wang and Kimura, PVLDB 2016.
  URL: `https://www.vldb.org/pvldb/vol10/p49-wang.pdf`
  Why: hybrid OCC/pessimistic contention handling cited by the batching paper;
  useful for deciding when GPU DB owner queues should switch from optimistic
  validation to contention-aware ordered execution.
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
- `queued` — **How Good are Learned Cost Models, Really? Insights from Query
  Optimization Tasks**, Woltmann et al., SIGMOD 2025.
  URL: `https://doi.org/10.1145/3725309`
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
- `queued` — **R2P2: Making RPCs First-Class Datacenter Citizens**,
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
- `queued` — **QueCC: A Queue-oriented, Control-free Concurrency
  Architecture**, Qadah and Sadoghi, Middleware 2018.
  URL: `https://doi.org/10.1145/3274808.3274810`
  PDF: `https://expolab.org/papers/quecc.pdf`
  Why: BOHM-adjacent deterministic two-phase planning/execution design for
  many-core transaction processing; useful for comparing queue-oriented
  planning against owner-local placeholder-first MVCC batches.
- `reviewed` — **A Wake-Up Call for Kernel-Bypass on Modern Hardware**,
  Jasny et al., DaMoN 2025.
  URL: `https://doi.org/10.1145/3736227.3736235`
  PDF:
  `https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/damon25_wake_up_call.pdf`
  Why: concise modern evidence that kernel networking and storage stacks cannot
  saturate 400G/800G NICs or PCIe Gen5 SSD arrays within realistic CPU budgets;
  useful for deciding when GPU DB should move from epoll/io_uring proofs toward
  kernel-bypass network or storage experiments.
- `queued` — **Rapid Data Ingestion through DB-OS Co-design**, Lim et al.,
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
- `queued` — **Your Read is Our Priority in Flash Storage**, An et al.,
  PVLDB 2022.
  URL: `https://www.vldb.org/pvldb/vol15/p1911-an.pdf`
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
- `queued` — **Databases on Modern Networks: A Decade of Research that now
  comes into Practice**, Lerner et al., PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p3894-lerner.pdf`
  Why: modern survey and call-to-action for database/network co-design; useful
  for organizing future pgwire, RDMA, DPDK, and application-specific transport
  benchmark tracks.
- `queued` — **BMC: Accelerating Memcached using Safe In-kernel Caching and
  Pre-stack Processing**, Ghigoff et al., NSDI 2021.
  URL: `https://www.usenix.org/conference/nsdi21/presentation/ghigoff`
  PDF: `https://www.usenix.org/system/files/nsdi21-ghigoff.pdf`
  Why: Tigger's related work points to BMC as another eBPF user-bypass design;
  useful for evaluating whether tiny validated cache/protocol operations belong
  in kernel-space fast paths, and where correctness/invalidation makes that too
  risky for SQL.
- `queued` — **KVell: the Design and Implementation of a Fast Persistent
  Key-Value Store**, Bartholomew et al., SOSP 2019.
  URL: `https://doi.org/10.1145/3341301.3359628`
  Why: Haas and Leis identify KVell as one of the closest systems to full
  NVMe-array exploitation; useful as a contrasting partitioned KV design for
  queue depth, SPDK usage, and limitations around range queries and small
  database payloads.
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
