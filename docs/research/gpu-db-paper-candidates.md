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
mechanical sympathy, or query optimization.

## Seed Queue

### Transaction processing, write path, and concurrency control

- `queued` — **TicToc: Time Traveling Optimistic Concurrency Control**,
  Yu et al., SIGMOD 2016.
  URL: `https://dl.acm.org/doi/10.1145/2882903.2882935`
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
- `queued` — **Transaction Repair for Multi-Version Concurrency Control**,
  arXiv 2024.
  URL: `https://arxiv.org/abs/2405.14761`
  Why: recent MVCC transaction repair approach that may inform conflict
  handling without throwing away all work.

### MVCC, snapshots, and visibility

- `queued` — **Scalable and Robust Snapshot Isolation for High-Performance
  Storage Engines**, PVLDB 2023.
  URL: `https://www.vldb.org/pvldb/vol16/p1426-alhomssi.pdf`
  Why: scalable snapshot isolation, long-reader robustness, and GC ideas.
- `queued` — **Read-Safe Snapshots: An abort/wait-free serializable read
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

- `queued` — **Shenango: Achieving High CPU Efficiency for Latency-sensitive
  Datacenter Workloads**, NSDI 2019.
  URL: `https://www.usenix.org/conference/nsdi19/presentation/ousterhout`
  Why: user-level scheduling and CPU allocation for latency-sensitive services;
  relevant to multiplexed IO and query workers under 1M logical sessions.
- `queued` — **Caladan: Mitigating Interference at Microsecond Timescales**,
  OSDI 2020.
  URL: `https://www.usenix.org/conference/osdi20/presentation/shenango`
  Why: runtime scheduling and resource allocation for microsecond-scale tail
  latency, useful for admission and worker ownership design.
- `queued` — **Demikernel: An Operating System Architecture for
  Microsecond-scale Datacenter Systems**, SOSP 2021.
  URL: `https://dl.acm.org/doi/10.1145/3477132.3483554`
  Why: low-latency OS/network stack architecture relevant to session and
  response-ring design.
- `queued` — **Design Choices in Low-Latency C++ Systems: Empirical Insights
  With Applications to High-Frequency Trading**, SSRN 2026.
  URL: `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6513601`
  Why: modern HFT-oriented low-latency systems survey; useful for queue,
  allocation, cache, and thread-pinning patterns.

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
- `queued` — **Roq: Robust Query Optimization Based on a Risk-aware Learned
  Cost Model**, arXiv 2024.
  URL: `https://arxiv.org/abs/2401.15210`
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
