# Design for a PostgreSQL‑Compatible GPU‑Accelerated Database Engine for Core Banking Systems

## 1 Introduction and Requirements

Modern core‑banking workloads require strict transactional guarantees, auditability and regulatory compliance.  Relational database management systems (RDBMS) have been the default choice for financial applications because they are **ACID‑compliant** (atomicity, consistency, isolation, durability) and provide strong data integrity and audit trails.  A 2026 industry overview notes that the financial industry chooses relational DBMS because they deliver **ACID compliance and data integrity**, with PostgreSQL and Oracle often recommended for banking due to their reliability and advanced features【976295677179754†L60-L82】.  The goal of this design is to build an engine that:

* Speaks the **PostgreSQL frontend/back‑end protocol** (targeting **PostgreSQL 16** wire‑protocol semantics) so that existing PostgreSQL drivers and tools can connect without modification.  The wire protocol is message‑based; it has a startup phase with authentication and parameter negotiation, followed by normal operation where clients send simple or extended queries and the server responds with structured messages.  The extended query protocol supports **pipelining** – sending multiple queries without waiting for previous ones – which reduces network round trips【583181887545674†L581-L604】.
* Implements the **core SQL standard** (data types, DDL, DML, transactions, triggers, stored procedures) and PostgreSQL 16 semantics, including multi‑version concurrency control (MVCC).  PostgreSQL uses MVCC so that each statement sees a snapshot of the database as it existed at some point in time, preventing readers from blocking writers and vice‑versa【997194816604558†L33-L47】.  This isolation model must be preserved.
* Provides **ACID properties** and high availability suitable for banking.  Durability should be ensured by a disk‑resident write‑ahead log (WAL) and replication across nodes.  The system must support **strong consistency** and auditing.
* Supports **hybrid CPU–GPU execution**.  GPU memory is limited (high‑end devices provide up to ~192 GB)【64261708151474†L63-L66】, so the engine cannot assume that all data fits on the GPU.  Hot partitions or tables will be cached or replicated into GPU memory, while the remainder stays in CPU memory or on disk.  The system must operate at full correctness in **CPU‑only mode** when GPUs are unavailable.
* Scales across **multiple GPUs**.  Recent research highlights that multi‑GPU DBMS architectures can aggregate GPU memory capacity and computational power but require careful data placement and replication strategies【64261708151474†L64-L75】.  The design should support clustering multiple GPUs as a unified resource and coordinate caching and replication across them【64261708151474†L91-L110】.

### 1.1 Performance targets

The engine targets the following service‑level objectives for a mid‑size core banking deployment:

| Metric | Target |
|--------|--------|
| Sustained OLTP throughput | >= 50 000 TPS |
| Peak burst throughput | >= 200 000 TPS |
| P50 latency (simple OLTP) | < 2 ms |
| P99 latency (simple OLTP) | < 10 ms |
| P99.9 latency (simple OLTP) | < 50 ms |
| Concurrent connections | >= 10 000 |
| RPO (Recovery Point Objective) | 0 (no committed transaction loss) |
| RTO (Recovery Time Objective) | < 30 s automated failover; < 5 min full GPU recovery |

### 1.2 Technology choices

* **Process model:**  **Single‑process, multi‑threaded** (unlike PostgreSQL's fork‑per‑connection model).  A single OS process owns all CUDA contexts, the shared buffer pool, the WAL writer, and the batch scheduler.  Tokio async runtime and dedicated thread pools (Section 2.2) handle concurrency.  This avoids the overhead of IPC for GPU context sharing and enables lock‑free shared data structures.  Child processes are used only for external utilities (`pg_dump`, `pg_basebackup`) and WAL archiver commands.
* **Implementation language:**  **Rust** for the host‑side engine (protocol, planner, storage, WAL, connection management).  Rust is chosen for memory safety without garbage collection — the borrow checker and ownership system eliminate use‑after‑free, double‑free, and data‑race bugs at compile time, which is critical for a banking‑grade system.  CUDA interop is achieved via the **`cudarc`** crate (safe Rust bindings to the CUDA driver API) for host‑side GPU management, while GPU kernels are authored as `.cu` files compiled by NVCC and linked via `build.rs`.  This hybrid approach gives idiomatic Rust on the host with native CUDA performance on the device.  Managed‑language runtimes (JVM, Python, Go) are excluded from the data path due to GC pauses and GIL constraints.  **`unsafe` discipline:**  `unsafe` blocks are confined to: CUDA FFI calls (wrapped by `cudarc`), pinned‑memory management, lock‑free atomics, and the GPU buffer pool's page‑table manipulation.  All `unsafe` blocks are annotated with `// SAFETY:` comments and audited via `cargo-geiger`.
* **SQL parser:**  **`libpg_query`** (the C library that extracts PostgreSQL's `gram.y` parser into a standalone library, producing Protobuf‑serialized parse trees) via the **`pg_query`** Rust crate (safe FFI bindings).  This gives exact PostgreSQL grammar compatibility with zero Python dependency.  The Python wrapper `pglast` may be used as a reference/test oracle but is not used in the production hot path.  The Protobuf parse tree from `libpg_query` is deserialized (via `prost`) into the engine's internal Rust AST representation via a generated visitor that maps PostgreSQL node types to engine‑native plan‑input nodes.  This translation layer isolates the planner from `libpg_query`'s Protobuf schema, allowing internal AST evolution without parser changes.
* **Build system:**  **Cargo** as the primary build system.  A `build.rs` script invokes NVCC to compile `.cu` kernel files into fatbins (targets `sm_80`, `sm_90`, `sm_100`, `sm_120`) and links them into the final binary.  Cargo profiles for dev/test/release.  `cargo-deb` and `cargo-rpm` for packaging (DEB/RPM/tarball).
* **Dependency management:**  Cargo with `Cargo.lock` for pinned dependency versions.  Key crates: `cudarc` (CUDA driver API bindings), `pg_query` (libpg_query FFI), `tokio` (async runtime), `prost` (Protobuf), `tracing` (structured logging), `tikv-jemallocator` (jemalloc allocator).  Critical dependencies (`libpg_query` C library) vendored via `build.rs`.  A Software Bill of Materials (SBOM) generated via `cargo-sbom` for each release as required by banking regulators.

The remainder of this document outlines a detailed architecture that satisfies these requirements.  It draws on existing PostgreSQL documentation, GPU‑accelerated OLTP/OLAP research and GPUDirect Storage documentation to inform the design.

### 1.3 Component architecture and layering

The system is organized into the following layers, with each layer depending only on layers below it:

1. **Platform layer:**  OS, CUDA runtime, NVML, NCCL, filesystem.
2. **Storage I/O and WAL:**  Disk I/O, WAL writer, WAL archiver, GPUDirect Storage interface.
3. **Buffer and memory management:**  CPU shared buffer pool, GPU buffer pool, pinned‑memory pool, slab allocators.
4. **Catalog and metadata:**  Schema definitions, system catalogs (`pg_catalog`), shard mappings, GPU placement metadata, statistics.
5. **Transaction manager:**  Transaction ID assignment, snapshot creation, visibility determination, lock management, commit/abort, MVCC garbage collection.
6. **Execution engine:**  CPU operators, GPU kernels, batch scheduler, operator fusion, result merging.
7. **Query planner/optimizer:**  AST transformation, cost model, CPU/GPU routing, plan caching, statistics integration.
8. **Session manager:**  Session state, prepared statements, portals, GUC parameters, resource limits, GPU resource reservations.
9. **Protocol layer:**  Wire protocol, authentication, pipelining, COPY, replication protocol.
10. **Management and observability:**  Metrics, health probes, admin API, audit logging.

Cross‑cutting concerns (structured logging, configuration, error taxonomy) are injected via interfaces, not hard‑wired.

Each layer exposes a defined interface.  Key interface contracts include:

* **Storage API:**  `tuple_fetch(tid)`, `seq_scan_open/next/close()`, `index_scan_open/next/close()`, `tuple_insert/update/delete()` with MVCC semantics.  Thread‑safety guarantees and buffer‑ownership semantics documented per method.
* **GPU memory manager API:**  `allocate(size, pool)`, `free(ptr, pool)`, `pin(page_id)`, `unpin(page_id)`, `evict(policy)`, `prefetch(shard_id, stream)`.
* **Planner‑to‑executor handoff:**  A plan‑node tree with a common `Operator` interface (`open/next/close`); each node tagged with execution device (CPU or GPU device ID).
* **WAL writer contract:**  `wal_insert(record)`, `wal_flush(lsn)`, `wal_callback_on_flush(lsn, callback)`.  Flush semantics: `fdatasync()` by default; configurable via `wal_sync_method`.
* **Transaction manager API:**  `txn_begin(isolation)`, `txn_commit()`, `txn_abort()`, `snapshot_create()`, `visibility_check(tuple_header, snapshot)`, `lock_acquire/release()`.

## 2 Protocol and SQL Compatibility Layer

### 2.1 Wire protocol implementation

* **Message‑based architecture:**  PostgreSQL uses a message‑based protocol with separate startup and normal‑operation phases.  A client begins a session by sending a startup message (user, database and protocol version).  The server then chooses an authentication method, returning an authentication request such as `AuthenticationMD5Password` or `AuthenticationSASL`【359884074571291†L34-L101】.  After authentication, the connection enters normal operation and clients send Query, Bind/Execute, FunctionCall or Copy messages.  Each message begins with a type byte and a length field【359884074571291†L32-L40】.  The engine must implement these message formats so that libpq and other drivers can connect transparently.

* **Protocol version targeting:**  Target PostgreSQL protocol version 3.0 with additive extensions from PostgreSQL 16 (e.g., protocol‑level query cancellation).  Implement version negotiation in the startup phase.  Define a compatibility matrix of supported client library versions (libpq, JDBC, psycopg2, npgsql).  Any proprietary extensions (e.g., GPU execution hints) use PostgreSQL's extensibility mechanisms (parameter status messages, notice responses) rather than new message types.

* **Authentication:**  Support the password methods used by PostgreSQL (clear‑text, MD5, SCRAM‑SHA‑256/SASL, GSSAPI).  The protocol defines specific message types for each method【359884074571291†L84-L101】; the engine should map these to its internal authentication modules (e.g., TLS + SASL).  Integrate with enterprise identity systems.  MD5 is deprecated; SCRAM‑SHA‑256 is the preferred default.

* **Simple vs. extended query:**  Simple Query sends a single SQL statement; Extended Query splits preparation, binding and execution into separate messages.  The extended protocol allows pipelining; a client may send a series of queries without waiting for previous ones to finish, reducing round trips【583181887545674†L581-L604】.  The server must track query handles, portal states and prepared statements across these messages.

* **Pipelining and Sync:**  In pipelined mode the client omits Sync messages to send dependent statements in a single transaction; the server must skip subsequent commands on error until a Sync is received【583181887545674†L593-L600】.  Implementing this correctly ensures network efficiency while preserving transactional semantics.

* **COPY, function call and replication:**  Support PostgreSQL's `COPY FROM STDIN` and `COPY TO STDOUT` sub‑protocols for bulk data import/export【583181887545674†L659-L699】.  Implement function call and replication sub‑protocols (e.g., `START_REPLICATION`) as in PostgreSQL so that replication tools operate unmodified.

### 2.2 Session management

The engine maintains per‑session state in a **Session Manager** subsystem:

* **Session state structure:**  Authenticated user, current database, transaction state (idle / in‑transaction / aborted), prepared statements, named portals, GUC parameter overrides, GPU resource reservations, temporary tables.
* **Concurrency model:**  Async I/O via **Tokio** runtime using `io_uring` (Linux, via `tokio-uring`) or `kqueue` (macOS) with a small pool of I/O threads handling protocol encode/decode.  Parsed requests are fed into the batching/execution pipeline.  This avoids thread‑per‑connection overhead while supporting >= 10 000 concurrent connections.  Rust `async`/`await` manages the many concurrent protocol state machines, with each connection as a lightweight Tokio task.
* **Thread pool architecture:**  Separate thread pools for: (a) I/O threads for protocol handling, (b) planner/optimizer threads, (c) a batcher thread that assembles micro‑batches and dispatches to GPU streams, (d) WAL writer threads, (e) background workers (vacuum, checkpoint, statistics).  **Sizing guidance:**  I/O threads = number of CPU cores / 4 (minimum 2); planner threads = number of CPU cores / 4; batcher threads = 1 per GPU; WAL writer = 1 dedicated thread; background workers = configurable (default 3).  All pool sizes exposed as GUC parameters.  Total thread count should not exceed 2× CPU core count to avoid context‑switch overhead.
* **Connection pooling:**  Built‑in transaction‑mode connection pooling (similar to PgBouncer).  Connections are returned to the pool at transaction boundaries.  Support `DISCARD ALL` semantics for external pooler compatibility.  Under overload, connections queue with a configurable timeout rather than being rejected.
* **Resource limits:**  Per‑session limits on GPU memory, concurrent queries, and temporary storage.  `max_connections` parameter controls the system‑wide limit.
* **Session cleanup:**  On disconnect: release GPU memory, roll back in‑flight transactions, close portals, free prepared statements.

### 2.3 SQL dialect and features

The engine should parse PostgreSQL 16's SQL dialect using `libpg_query` (linked as a C library):

* **Core SQL operations:**  `SELECT`, `INSERT`, `UPDATE`, `DELETE`, joins, subqueries, aggregates, views, triggers and stored procedures.  Ensure compliance with the SQL standard.
* **Data types and functions:**  Numeric, string, boolean, timestamp, JSON, arrays, geometric types and domain types; built‑in functions and operators.  **All monetary calculations use fixed‑point decimal arithmetic (not IEEE 754 floating‑point) on both CPU and GPU** to ensure exact results required by banking standards.  The `NUMERIC`/`DECIMAL` type is implemented as a **128‑bit fixed‑point representation** (sign bit + 127‑bit magnitude) supporting up to 38 significant digits — matching PostgreSQL's maximum precision.  On the host, Rust's native `i128`/`u128` types provide 128‑bit arithmetic with no external dependency.  On GPU, a custom `uint128` struct (two `uint64_t` components) in CUDA device code implements carry‑propagating addition, subtraction, and multiplication kernels.  Division uses a restoring‑division algorithm.  Rounding mode: `ROUND_HALF_EVEN` (banker's rounding) by default, configurable per session.  All intermediate arithmetic results use the full 128‑bit representation to prevent precision loss during multi‑step calculations (e.g., compound interest).
* **Transaction control:**  Support `BEGIN`, `COMMIT`, `ROLLBACK`, savepoints, isolation levels (Read Committed, Repeatable Read, Serializable).  PostgreSQL implements MVCC and Serializable Snapshot Isolation so that reading never blocks writing【997194816604558†L33-L47】.  Two‑phase commit (`PREPARE TRANSACTION` / `COMMIT PREPARED`) supported for distributed banking middleware.
* **Extensions:**  Provide an extension framework covering: custom type registration (OID allocation, I/O functions, binary send/receive), custom operator registration (operator classes, selectivity estimators), custom index access methods (AM interface), custom aggregate functions, planner/executor hook points.  Extensions may register CPU‑only operators; GPU kernel registration by extensions is a Phase 4 feature.  Foreign Data Wrapper (FDW) interface supported (CPU‑only execution path).
* **NOTIFY/LISTEN:**  Asynchronous notification channels, matching PostgreSQL semantics.  `NOTIFY channel, 'payload'` delivers messages to all sessions listening on that channel.  Notifications are delivered at transaction commit.  Implemented via an in‑memory channel registry on the CPU side; GPU batch commits trigger notification delivery as part of the post‑commit callback.  Essential for banking event‑driven architectures (e.g., real‑time balance alerts, fraud detection triggers).
* **Logical decoding / Change Data Capture (CDC):**  The WAL contains logical records sufficient to reconstruct row‑level changes.  A **logical decoding** framework reads the WAL and emits a stream of change events (INSERT, UPDATE, DELETE with old/new row values) in a pluggable output format (JSON, Protobuf, or PostgreSQL‑compatible `pgoutput` for use with standard CDC tools like Debezium).  Logical replication slots track consumer progress by LSN, preventing WAL recycling until consumed.  This is critical for banking integration: feeding changes to downstream systems (fraud engines, data warehouses, regulatory reporting) without polling.

### 2.4 System catalog compatibility

Implement the core PostgreSQL system catalog views (`pg_catalog` schema) with compatible column names and types: `pg_class`, `pg_attribute`, `pg_type`, `pg_proc`, `pg_namespace`, `pg_index`, `pg_constraint`, `pg_stat_activity`, `pg_stat_user_tables`, `pg_stat_user_indexes`, `pg_stat_statements`, and `information_schema`.  This is essential for compatibility with ORMs (Django, Rails, Hibernate), migration tools (Flyway, Liquibase), and monitoring tools (pgAdmin, DBeaver).

Add a `gpu_catalog` schema for engine‑specific metadata: device status, shard placement, GPU memory usage, batch queue depth, kernel execution statistics.

### 2.5 Error handling and error reporting

* **Error taxonomy:**  Four categories: (a) client errors (syntax, constraint violation, authorization — SQLSTATE class 22/23/42), (b) transient internal errors (GPU OOM, timeout, serialization failure — SQLSTATE class 40/53), (c) fatal internal errors (WAL corruption, GPU hardware fault — SQLSTATE class 58), (d) administrative notifications (parameter changes, GPU migration events).
* **GPU error propagation:**  CUDA kernels cannot throw exceptions.  Each transaction in a GPU batch writes a status code into a device‑side status array.  After kernel completion, the status array is copied to the host via `cudaMemcpyAsync`.  The batcher maps each error to the originating client connection and sends the appropriate PostgreSQL `ErrorResponse` message with correct SQLSTATE code.
* **Partial batch failure:**  When a batch of N transactions executes and K fail (e.g., constraint violations), the N‑K successful transactions commit normally.  The K failed transactions receive individual error responses.  The deterministic ordering model ensures this is safe.
* **Error response format:**  Full PostgreSQL‑compatible `ErrorResponse` messages with fields: severity, SQLSTATE code, message, detail, hint, position, internal position, internal query, where, schema, table, column, constraint, file, line, routine.

## 3 Data Model and Storage

### 3.1 Table storage layout

Banking workloads combine OLTP and analytical processing.  Row‑oriented storage is better for single‑row reads/writes, while columnar layout helps scanning and analytics.  We adopt a **hybrid storage engine**:

* **Row‑oriented base tables** for transactional records (accounts, transactions, ledger entries).  Each row stores metadata (transaction ID, creation timestamp) used for MVCC.  Updates append a new version rather than overwriting.
* **Columnar projections or materialized segments** for analytical queries on large tables (e.g., reporting, compliance).  Analytical segments can be loaded onto GPUs for acceleration.
* **TOAST (The Oversized‑Attribute Storage Technique):**  Values exceeding the page threshold (default ~2 KB) are compressed and/or stored out‑of‑line in a companion TOAST table, matching PostgreSQL semantics.  TOAST values on CPU use the standard chunk storage.  On GPU, oversized values are not transferred to GPU memory — queries referencing TOASTed columns fall back to CPU unless the column is detoasted at scan time and fits in the GPU buffer pool.  JSON and TEXT columns commonly trigger TOAST; the planner accounts for detoast cost in the CPU/GPU routing decision.
* **Variable‑length data on GPU:**  For GPU‑resident shards, variable‑length columns (VARCHAR, TEXT, BYTEA) that fit within the TOAST threshold are stored in a **variable‑length pool** per shard: a contiguous byte array with a parallel offset/length index.  The offset index is aligned for coalesced access.  Fixed‑length columns remain in SoA arrays.  Variable‑length columns that are not accessed by a GPU kernel are not transferred (column pruning).

### 3.2 CPU‑side buffer pool

The CPU‑side storage layer manages a **shared buffer pool** with the following characteristics:

* **Page size:**  8 KB (matching PostgreSQL) or configurable up to 64 KB for analytical scan amortization.
* **Buffer lookup:**  Hash table mapping `(relation_id, block_number)` to buffer slot.
* **Pin/unpin protocol:**  Reference counting with `pin(buffer)` / `unpin(buffer)`.  Pinned pages cannot be evicted.
* **Replacement policy:**  Two‑queue / ARC (Adaptive Replacement Cache) to handle the hot/cold split in banking workloads without cache pollution from full‑table analytical scans.
* **Dirty‑page management:**  Per‑page dirty flag.  A **background writer** continuously flushes dirty pages to smooth checkpoint I/O.
* **NUMA awareness:**  On multi‑socket servers, buffer pool memory is allocated on the NUMA node closest to the GPU's PCIe root complex using `set_mempolicy`.
* **Huge pages:**  Use 2 MB or 1 GB huge pages to reduce TLB pressure.

### 3.3 CPU‑side memory allocator

* **Per‑query arena allocator:**  Short‑lived allocations (parse trees, plan nodes, result buffers) use a slab/arena allocator.  Bulk deallocation at query/batch completion eliminates fragmentation.
* **Pinned (page‑locked) memory pool:**  Pre‑allocated pool of `cudaHostAlloc` memory (4–16 GB) for all CPU‑GPU transfers.  Required for `cudaMemcpyAsync` and GPUDirect Storage.  Managed with a free‑list.  If the pool is exhausted, backpressure is applied (queue new batches) rather than falling back to unpinned transfers.
* **General‑purpose allocator:**  jemalloc via `tikv-jemallocator` (set as the global allocator) to reduce contention in multi‑threaded code.

### 3.4 Online DDL and schema changes

Schema changes (DDL) require coordination between CPU and GPU:

* **Schema‑change lock:**  `ALTER TABLE` acquires an `AccessExclusiveLock` (matching PostgreSQL semantics), blocking all concurrent reads and writes on the affected table.  GPU batches touching the table are drained before the lock is granted.
* **GPU cache invalidation:**  After a DDL change, all GPU‑resident shards for the affected table are invalidated (SoA column arrays, cuckoo hash indexes, visibility bitmaps).  Shards are rebuilt from CPU/disk state with the new schema before GPU execution resumes.
* **Plan cache invalidation:**  All cached plans referencing the altered relation are invalidated at both AST and physical‑plan tiers.
* **Online index creation:**  Support `CREATE INDEX CONCURRENTLY` with a two‑pass approach (initial build on a snapshot, then catch up with concurrent modifications) to minimize lock duration.  GPU‑resident indexes are rebuilt asynchronously.
* **Column additions and type changes:**  Column additions with defaults use a "virtual column" approach (default stored in catalog, not materialized immediately) to avoid rewriting all GPU‑resident data.  Type changes that alter storage format require full shard rebuild.

### 3.5 Partitioning and placement

Because GPU memory is limited, we partition tables into **shards**.  Each shard is assigned to CPU memory, a GPU device or both.  The **Unified Multi‑GPU Abstraction** proposed in research treats multiple GPUs as one large logical GPU, simplifying placement decisions【64261708151474†L91-L96】.  The unified abstraction maintains a **device‑to‑shard mapping** queryable by the failure handler and the planner.  Each shard has a health status: `ACTIVE`, `STALE`, `RECOVERING`, `OFFLINE`.

Data placement policies include:

* **Hot partition caching:**  Frequently accessed shards are replicated in GPU memory to accelerate reads and writes.  Cold shards remain in CPU memory or on disk.
* **Cache‑aware replication:**  Lancelot's evaluation shows that coordinating caching and replication across GPUs – selectively replicating shuffle‑intensive data – yields up to 2.5× performance improvement【64261708151474†L20-L31】.  The engine adopts a similar cost‑based policy.  For resilience, critical shards (account balances, ledger partitions) are replicated across at least two GPUs when multiple GPUs are available.
* **Dynamic migration:**  Monitor access patterns and migrate shards between CPU and GPU memory based on hotness.  Migrations are pre‑staged during low‑traffic periods.  Never trigger synchronous shard migration on the critical path of a query.  Use GPUDirect Storage to load partitions directly from disk into GPU memory.

### 3.6 Indexes and metadata

Each table shard maintains separate indexes:

* **GPU‑resident indexes:**  **Cuckoo hash tables** for primary‑key lookups (O(1) worst‑case, coalescing‑friendly two‑probe structure).  Sized to 50–60% load factor.  Each slot: 4‑byte key hash + 8‑byte version‑head pointer + 4‑byte metadata = 16 bytes, aligning 8 slots to a 128‑byte cache line for warp‑coalesced access.  The hash function uses XOR‑based bank interleaving (`index = key ^ (key >> 5)`) to avoid shared‑memory bank conflicts when probing in shared memory.  **Sorted arrays** for secondary indexes and range queries (GPU binary search achieves ~200M lookups/s on A100).  Concurrent inserts use lock‑free `atomicCAS`.  Index rebuild as a background operation at epoch boundaries.
* **CPU‑resident indexes:**  B‑trees, GiST or GIN indexes as in PostgreSQL for CPU shards.  Cross‑device queries merge results from GPU and CPU indexes.

Metadata tables (catalogs) maintain schema definitions, shard mappings, and GPU placement information.

### 3.7 Write‑ahead log (WAL) and checkpoints

#### 3.7.1 WAL format

The WAL uses a **hybrid record format** with a fixed‑size header followed by a payload:

| Field | Size | Description |
|-------|------|-------------|
| LSN | 8 bytes | Monotonically increasing Log Sequence Number |
| prev_LSN | 8 bytes | LSN of the previous record (for backward traversal) |
| xid | 8 bytes | Transaction ID |
| record_type | 2 bytes | Physical page‑image, logical operation, or DDL |
| resource_manager_id | 2 bytes | Which subsystem owns this record type |
| payload_length | 4 bytes | Length of the variable payload |
| flags | 2 bytes | Compression flag, FPI flag, etc. |
| CRC‑32C | 4 bytes | Checksum over header + payload |

**Payload types:**  Physical records (full‑page image + delta) for heap/index pages; logical records (relation OID, tuple data) for high‑level operations.  Physical records are schema‑independent and allow byte‑exact page reconstruction.

Each dirty page (CPU and GPU) is stamped with the LSN of the last WAL record that modified it.  Page LSNs drive checkpoint dirty‑page identification.

All multi‑byte fields use **little‑endian** byte order (matching x86/ARM64 host and CUDA device memory layout) to avoid byte‑swapping overhead.  WAL segment files are fixed‑size (64 MB, configurable), named by starting LSN.  A versioned "magic number" and format version in the segment header enable tooling compatibility detection.

#### 3.7.2 Control file

A **control file** is the root metadata structure for the database instance.  It stores:

* Current WAL LSN and checkpoint location (redo point).
* Database system identifier (unique 64‑bit ID assigned at `initdb`).
* WAL format version, page size, and data checksum enablement flag.
* Timeline ID (incremented on each PITR recovery to distinguish WAL histories).
* Oldest active transaction ID and oldest frozen transaction ID.
* Catalog version.

The control file is updated at each checkpoint (step 7 in 3.7.8) and `fsync`‑ed.  It is small enough to fit in a single disk sector, ensuring atomicity of writes on modern storage.  A backup copy is maintained for redundancy.

#### 3.7.3 Full‑page writes (torn‑page protection)

After each checkpoint, the first WAL record modifying a given page includes a **full‑page image (FPI)**.  During recovery, the FPI is restored before subsequent deltas are applied.  This guarantees torn‑page protection regardless of storage behavior during crashes.

#### 3.7.4 WAL flush and group commit

* **Flush mechanism:**  `fdatasync()` by default (avoids metadata flush overhead).  Configurable via `wal_sync_method` parameter (`fsync`, `fdatasync`, `open_datasync`, `open_sync`).  `O_DIRECT` for WAL writes to avoid double‑buffering.
* **WAL buffer:**  Fixed‑size shared‑memory ring buffer.  Writers append via lock‑free CAS on the write pointer.  A WAL flusher thread advances the flush pointer.
* **Group commit:**  GPU batch execution naturally enables group commit: all transactions in an epoch share a single WAL flush, amortizing fsync cost.  For CPU‑side transactions that bypass batching, a separate group‑commit mechanism collects pending WAL records and flushes them together (`commit_delay` / `commit_siblings` tuning knobs).
* **WAL compression:**  Optional per‑record LZ4 compression (fast) or zstd (high ratio).  Indicated by a flag in the record header.

#### 3.7.5 Filesystem and storage requirements

* **Recommended filesystems:**  XFS (preferred for WAL — optimized for concurrent sequential writes, stable `fsync` semantics) or ext4 with `data=ordered` journaling mode.  ZFS is supported but `O_DIRECT` must be disabled (ZFS uses its own ARC); WAL flush uses `fsync` instead.  Never use `data=writeback` on ext4 — it can lose data on crash.
* **WAL storage:**  Dedicated NVMe SSD for WAL (separate from data files) to avoid I/O contention between sequential WAL writes and random data reads.  Sustained sequential write bandwidth >= 1 GB/s recommended for 200K TPS target.  Enterprise‑grade NVMe with power‑loss protection (PLP) required — consumer SSDs may lose buffered writes on power failure, violating fsync durability.
* **Data storage:**  NVMe SSDs for data files.  RAID is not required (replication provides redundancy), but RAID‑10 is acceptable.  For large datasets, tiered storage: NVMe for hot data, SATA SSD or HDD for archived WAL and backups.
* **Battery‑backed write cache (BBWC):**  If using hardware RAID, BBWC is required to ensure that write‑back caching honors `fsync` semantics.  Without BBWC, disable write‑back caching on the RAID controller.
* **GPUDirect Storage:**  Requires a GDS‑compatible filesystem (ext4, XFS) and NVMe devices.  GDS is not supported on network filesystems (NFS, CIFS).  Fall back to pinned‑memory transfers when GDS is unavailable.

#### 3.7.6 WAL segment lifecycle and archiving

Segment lifecycle: active -> filled -> archived -> recyclable.  A segment is never recycled until: (a) archived to durable object storage (S3/GCS/Azure Blob), (b) replicated to all synchronous replicas, (c) its LSN precedes the latest checkpoint's redo point.  `wal_keep_size` parameter retains extra segments for lagging replicas.  WAL generation rate and archive lag are monitored with alerts.

#### 3.7.7 Durability guarantees

* **Disk‑resident WAL:**  A commit is acknowledged only after the WAL record has been durably flushed.  The WAL is the single source of truth.
* **WAL‑before‑visibility invariant (critical):**  No transaction's results may be visible to other transactions on the GPU until the corresponding WAL records are durably flushed.  The epoch‑advance (making new versions visible) is gated on WAL flush confirmation from the CPU.  This prevents dirty reads of data that could be lost on crash.
* **GPU as execution cache:**  GPU memory is volatile.  The design treats GPU‑resident data as a cache of the authoritative state defined by the WAL and checkpointed snapshots.  On failure, GPU memory is repopulated by replaying the WAL into GPU structures.
* **Streaming WAL into GPU:**  Use **GPUDirect Storage** to transfer WAL segments directly from NVMe or distributed storage into GPU memory.  GPUDirect Storage provides a direct DMA path between storage and GPU memory, bypassing CPU bounce buffers【938574903164919†L39-L43】.  The cuFile APIs are suited for **coarse‑grained streaming transfers** (64 KB+ for efficiency)【938574903164919†L153-L160】; WAL streaming is batched accordingly.  When GDS is unavailable (desktop GPUs, unsupported filesystems), fall back to pinned CPU memory with `cudaMemcpyAsync`.
* **No sole‑GPU‑copy invariant:**  A committed page's only durable copy must never exist solely in GPU memory.  The WAL + most recent checkpoint must always be sufficient to reconstruct full state.

#### 3.7.8 Checkpoint algorithm

1. Record the current WAL LSN as the checkpoint's **redo point**.
2. Flush all GPU‑resident dirty pages to CPU/disk (device‑to‑host transfer via `cudaMemcpyAsync` on a dedicated checkpoint stream; checkpoint cannot complete until transfer is confirmed).
3. Write all dirty CPU/disk pages (pages whose LSN >= previous checkpoint's redo point) to their on‑disk locations.  **Checkpoint throttling** spreads writes to avoid saturating I/O bandwidth.
4. `fsync()` all data files.
5. Write a checkpoint WAL record containing the redo point, list of active transactions, and oldest active XID.
6. `fsync()` the WAL.
7. Update the control file with the new checkpoint location; `fsync()` the control file.

Checkpoint frequency is tuned to bound WAL replay time to the RTO target.

#### 3.7.9 Data checksums

* **Page‑level CRC‑32C:**  Computed for every data page.  Stored in page header.  Verified on every read from disk.  Verification failure raises alert and triggers recovery from replica or backup.
* **WAL record CRC‑32C:**  Included in every record header (see 3.7.1).  Verified during recovery replay and streaming replication.
* **GPU ECC:**  ECC‑enabled GPU memory is **required for production banking deployments**.  Non‑ECC GPUs (e.g., RTX 5090) may be used for development with documented risk.  ECC error counters checked after every batch via `cudaDeviceGetAttribute`.

### 3.8 Dirty‑page write‑back from GPU

Dirty GPU pages are written back to the CPU/disk authoritative store periodically (every N seconds or at checkpoint, whichever is sooner).  A per‑GPU dirty‑page bitmap tracks modified pages.  After write‑back and fsync, dirty flags are cleared.  This bounds the amount of WAL that must be replayed on GPU failure.

## 4 GPU Integration

### 4.1 Execution model and concurrency control

GPUs offer massive parallelism and high memory bandwidth.  Research has shown that GPUs can process thousands of transactions concurrently【206946629484378†L75-L83】.  However, OLTP workloads present challenges because control flow is irregular and each transaction may follow a different path【206946629484378†L128-L135】.  Concurrency control schemes designed for CPUs behave differently on GPUs and must be redesigned【206946629484378†L146-L152】.  Two key insights inform our design:

* **Batched deterministic execution:**  The Epic database introduces a multi‑versioned GPU‑based deterministic OLTP engine.  It batches transactions into epochs and establishes a serial order before execution.  During an initialization phase, Epic allocates versions based on write sets and calculates version locations.  Transactions then access versions directly during the execution phase without searching【150335027845516†L90-L107】.  Batched execution eliminates version search overhead and leverages GPU parallelism.
* **MVCC and epoch‑based garbage collection:**  Epic stores intermediate versions separately and reclaims them at epoch boundaries【150335027845516†L108-L118】.  Multi‑version concurrency control allows reads and writes to proceed concurrently and avoids read/write conflicts【150335027845516†L27-L40】.  The engine will adopt an epoch‑based MVCC design inspired by Epic, with deterministic ordering for GPU‑resident transactions.

#### 4.1.1 Transaction classification and routing

Transactions are classified before execution:

* **Auto‑commit single‑statement transactions** touching GPU‑resident data: routed to GPU batch pipeline.
* **Interactive multi‑statement transactions** (`BEGIN`...`COMMIT`): execute on CPU with conventional MVCC.  Individual statements within the transaction that touch GPU‑resident data may issue GPU reads, but the transaction does not enter the GPU batch pipeline.  This avoids the unsolved problem of multi‑batch interactive transactions.
* **Two‑phase commit transactions** (`PREPARE TRANSACTION`): execute on CPU.
* **Read‑only queries touching only CPU‑resident data:** execute on CPU, bypassing batching entirely (no latency penalty).
* **Analytical queries:** may be offloaded to GPU kernels outside the OLTP batch pipeline.

#### 4.1.2 Adaptive batch formation

Batching uses a **dual‑trigger** mechanism rather than a fixed batch size:

* **Count threshold:**  Configurable (default 256 transactions).  Larger values improve GPU throughput but increase latency.
* **Time deadline:**  Configurable (default 1 ms since first transaction in the batch).  Caps worst‑case queuing delay.
* Whichever trigger fires first closes the batch.  Both parameters are tunable at runtime.

**Transaction‑type binning:**  At batch formation, transactions are grouped by type (balance inquiry, debit, credit, transfer) so that threads within the same warp execute the same code path, minimizing warp divergence.

**Batch size floor:**  All batches are padded to a multiple of 32 (warp size) with no‑op transactions to avoid partial‑warp waste.

**Latency analysis:**  At 50 000 TPS, a 1 ms deadline accumulates ~50 transactions per batch.  This is below optimal GPU utilization but acceptable for latency SLAs.  Multiple threads per transaction (32 threads for index lookups, version allocation, constraint checks) inflate parallelism to 50 × 32 = 1 600 threads.  At peak 200 000 TPS, a 1 ms window yields ~200 transactions = 6 400 threads — better GPU utilization.

#### 4.1.3 GPU execution pipeline

1. **Transaction batching (CPU):**  Group incoming transactions via dual‑trigger batching.  Pre‑compute read/write sets and assign a commit order.  Build redo records and allocate version slots.  Estimate GPU memory requirements; if estimated memory exceeds available GPU memory, split the batch or route to CPU.
2. **Initialize concurrency control (GPU):**  Run a kernel to allocate new versions, set predecessor pointers and update per‑row metadata.  Fuse this with the execution phase into a single kernel when possible (each thread block initializes its slots in shared memory, then executes) to eliminate one kernel launch.
3. **Execution phase (GPU):**  Launch kernels to perform reads and writes using the predetermined order.  Because the order is fixed, there is no need for locks.  Per‑transaction error codes are written into a device‑side status array.
4. **Apply to memory (GPU):**  New versions become visible **only after WAL flush confirmation** (WAL‑before‑visibility invariant).  A separate kernel updates derived indexes.
5. **WAL generation and flush (CPU):**  Concurrent with GPU execution, the CPU serializes redo records.  After the WAL is durably flushed, the epoch‑advance signal is sent to the GPU, and commit acknowledgements are sent to clients.

**CUDA stream pipeline:**  Use at least three CUDA streams per GPU: (a) H2D data transfer stream, (b) compute/kernel execution stream, (c) D2H result stream.  This allows batch N's compute to overlap with batch N+1's data transfer and batch N−1's result writeback.  Use `cudaEventRecord` / `cudaStreamWaitEvent` for inter‑stream synchronization rather than `cudaDeviceSynchronize`.

**CUDA Graphs:**  Capture the multi‑kernel pipeline (initialize + execute + apply + index‑update) as a `cudaGraph_t` and replay it per batch.  This reduces per‑launch overhead from ~5 μs/kernel to ~1 μs for the entire pipeline.

**Kernel launch overhead analysis:**  5 kernels × 7 μs = 35 μs overhead per batch.  At 2 ms target latency, this is < 2% — acceptable.  CUDA Graphs reduce this further.

**Shared memory budget per block:**  On sm_80 through sm_120, configurable shared memory ranges from 48 KB (default) to 164 KB+ (sm_90+).  With 256 threads per block and 48 KB shared memory, each transaction gets ~192 bytes — sufficient for a few row pointers and scratch variables.  Use `cudaFuncSetAttribute()` to request maximum shared memory carveout for memory‑intensive kernels, documenting the tradeoff with L1 cache capacity.  For transactions exceeding their shared‑memory allocation, a **spill‑to‑global‑memory** path uses a pre‑allocated per‑block scratch area (10–20× slower but preserves correctness).  Bulk operations (e.g., a transfer debiting one account and crediting 10 000 accounts) are decomposed into sub‑batches at the CPU level.

**Cooperative groups:**  For kernels requiring grid‑wide synchronization (e.g., epoch barrier between initialize and execute phases), use `cudaLaunchCooperativeKernel` with CUDA cooperative groups rather than splitting into multiple launches with CPU‑side synchronization.

### 4.2 Query processing on GPU

* **Operator fusion:**  Analytical queries (scans, filters, aggregations, joins) are fused into GPU kernels.  For OLTP queries, only point reads/writes are offloaded to the GPU if the data is present in GPU memory.  Complex expressions or user‑defined functions fall back to the CPU.
* **GPU hash join:**  Build phase allocates a hash table in GPU global memory (open addressing, linear probing, 70% load factor).  Each build thread inserts one tuple using `atomicCAS`.  Probe phase: each thread hashes its probe key and walks the chain.  For joins where the build side fits in shared memory (< 48 KB), a **shared‑memory hash join** is used — the build side is loaded cooperatively by the thread block, then each probe thread accesses shared memory (avoiding global memory latency).  For larger builds, a **partitioned hash join** splits both sides into GPU‑cache‑friendly partitions (radix partitioning), then executes shared‑memory joins per partition.  Anti‑joins and semi‑joins supported via bloom filter pre‑filtering: a GPU‑resident bloom filter (64 KB–1 MB, 8 hash functions) is built from the inner side and checked before probing the hash table, eliminating non‑matching tuples early.
* **GPU memory management:**  See Section 4.3.
* **Replication and synchronization across GPUs:**  When using multiple GPUs, hot shards may be replicated on multiple devices.  Cross‑GPU communication uses NVLink or PCIe.  The engine detects conflicting updates across replicated shards and propagates changes via the WAL stream.  Strong consistency for cross‑GPU replicated shards: a write must be applied to all replicas before the transaction commits.

### 4.3 GPU buffer pool and memory management

#### 4.3.1 GPU buffer pool

* **GPU page size:**  16 KB (matching NVMe/GPUDirect transfer granularity).
* **GPU page table:**  Maps logical page IDs to GPU physical addresses.  Resides in GPU global memory for kernel access; CPU‑side shadow copy for management.
* **Replacement policy:**  CLOCK (second‑chance) with per‑page reference bit.  Reference bits set by execution kernels, cleared by a background eviction kernel.  CLOCK is chosen over LRU because it avoids expensive per‑access atomic timestamp updates.
* **Pin counts:**  `atomicAdd` on a per‑page counter in GPU global memory.  Pinned pages cannot be evicted.
* **Dirty tracking:**  Per‑page dirty bit in the page table, set atomically by execution kernels.  Dirty pages written back to CPU/disk at checkpoint.
* **Memory budget (RTX 5090, 32 GB):**  ~2 GB for CUDA runtime/kernels/stack, ~24 GB for buffer pool, ~6 GB for intermediate results/version pools/indexes.  Scale proportionally for H100 (80 GB), B200 (192 GB).

#### 4.3.2 GPU slab allocator

Fixed‑size slab classes for GPU memory: 64B, 128B, 256B, 512B, 1KB, 4KB, 8KB.  Banking rows have predictable sizes.  Eliminates external fragmentation.  Epoch‑based GC returns freed slots to the correct slab.  Avoid `cudaMalloc`/`cudaFree` at fine granularity — pre‑allocate large pools at startup and sub‑allocate.  Use `cudaMallocAsync` with CUDA memory pools (`cudaMemPool_t`) on sm_80+ for stream‑ordered allocation.

Track fragmentation metric (largest‑free‑block / total‑free‑memory).  Trigger compaction during quiet epoch boundaries when fragmentation exceeds threshold.

#### 4.3.3 GPU memory pressure management

Define GPU memory pressure levels:

| Level | Threshold | Action |
|-------|-----------|--------|
| LOW | < 60% utilized | Normal operation |
| MEDIUM | 60–80% utilized | Increase eviction aggressiveness |
| HIGH | 80–90% utilized | Evict cold shards proactively, reduce batch sizes |
| CRITICAL | > 90% utilized | Stop accepting GPU‑bound work; route all to CPU |

Before submitting a batch, estimate memory requirements (input + intermediate versions + output).  If estimated > available, split the batch or route to CPU.

GPU memory reservation: reserve 10–15% of GPU memory for intermediate results and version slots.  Shard caching never consumes this reservation.

#### 4.3.4 Memory coalescing for MVCC version chains

Avoid traditional pointer‑based version chains on GPU.  Use contiguous, pre‑allocated **version arrays** indexed by `(row_id, epoch)`:

* **Structure‑of‑Arrays (SoA) format:**  Separate arrays per column (amount, balance, timestamp).  When 32 warp threads read the "balance" column of 32 consecutive rows, SoA produces a single 128‑byte coalesced read.
* **Current‑version table:**  Flat array indexed by row_id storing data inline for the common "read current version" case.  Only historical‑version reads (Repeatable Read, Serializable) traverse the version array.
* **Visibility bitmap:**  Precompute visibility in a separate kernel pass that reads all xmin values (coalesced), computes visibility bits, stores them in a bitmap.  The execution kernel uses the bitmap rather than re‑reading metadata.
* **Alignment:**  Version array elements aligned to 4/8‑byte boundaries.  Short fields padded to 4 bytes within SoA layout.
* **NULL handling in SoA:**  A per‑column **null bitmap** stored as a separate array (1 bit per row, packed into 32‑bit words for warp‑coalesced reads).  GPU kernels check the null bitmap before reading column values, avoiding undefined memory access.  Null‑aware arithmetic operators short‑circuit on null inputs.
* **GPU expression evaluation:**  Simple expressions (arithmetic, comparison, CASE/WHEN, COALESCE, IS NULL) are compiled into GPU‑executable expression trees at plan time.  Each expression node is represented as an opcode in a flat bytecode array interpreted by a GPU expression evaluator kernel.  For frequently executed expressions, NVRTC (runtime compilation) generates optimized PTX from CUDA C expression templates, avoiding interpretation overhead.  The host‑side Rust code invokes NVRTC via `cudarc`'s safe wrappers.  Expressions involving string operations (LIKE, regex) or complex casts fall back to CPU.

### 4.4 CUDA error handling and GPU abstraction

#### 4.4.1 CUDA error checking

Every CUDA API call goes through `cudarc`'s `Result`‑returning wrappers, leveraging Rust's type system:

* All `cudarc` methods return `Result<T, DriverError>`.  The `?` operator propagates errors through the call chain.  A custom error handler maps `DriverError` variants to engine‑level actions (log, increment counter, trigger circuit breaker).
* **Kernel launch errors:**  In debug/test builds, a `cuCtxSynchronize()` is called after each kernel launch to catch asynchronous errors immediately.  In release builds, errors are detected asynchronously via stream event callbacks (`cuLaunchHostFunc`) to avoid serializing the pipeline.
* On error: the handler logs the CUDA error string, the source location (via `#[track_caller]`), the CUDA stream, and the batch ID; increments the per‑GPU error counter; triggers the circuit breaker if the error is unrecoverable (ECC uncorrectable, context destroyed).
* Full synchronous validation is used in CI/test builds but **never** on the production hot path.

#### 4.4.2 CUDA driver API vs. runtime API

The engine uses the **CUDA driver API** exclusively, accessed via the **`cudarc`** crate which provides safe Rust wrappers around driver API functions.  Using the driver API (rather than the runtime API) is the natural choice for Rust because `cudarc` wraps the driver API and it provides finer control:

* **Virtual memory management** (`cuMemCreate`, `cuMemMap`, `cuMemSetAccess`): required for fine‑grained page‑table control in the GPU buffer pool (mapping/unmapping individual 16 KB pages without reallocation).
* **Context management** (`cuCtxCreate`, `cuCtxDestroy`): explicit context lifecycle for multi‑GPU with per‑GPU error isolation.
* **Module loading** (`cuModuleLoad`): loading pre‑compiled fatbins and NVRTC‑compiled PTX for JIT‑compiled expressions.
* **Kernel launching** (`cuLaunchKernel`): launching `.cu`‑compiled kernels from Rust with type‑safe parameter passing via `cudarc::LaunchConfig`.

All CUDA calls go through `cudarc`'s `Result`‑returning API, which converts CUDA error codes into Rust `Result<T, DriverError>`.  The `?` operator propagates errors naturally through the call chain without macros.

#### 4.4.3 GPU abstraction layer for testing

A **GPU backend trait** abstracts all GPU operations:

* `trait GpuBackend` with methods: `allocate()`, `free()`, `launch_kernel()`, `memcpy_async()`, `synchronize()`, `get_device_properties()`.
* **Production implementation (`CudaBackend`):** Delegates to CUDA driver API via `cudarc`.
* **Mock implementation (`CpuBackend`):** Executes kernel logic on CPU using reference Rust implementations.  Used in unit tests to validate transaction logic, MVCC visibility, and batch semantics without requiring GPU hardware.
* **Record/replay implementation (`RecordingBackend`):** Records all GPU API calls with arguments and results; replays for deterministic regression testing.

This abstraction is a test‑time concern.  It uses Rust generics with monomorphization (`impl GpuBackend` or `<B: GpuBackend>`) to eliminate virtual dispatch overhead in production builds — the compiler generates specialized code for `CudaBackend` with zero indirection.

### 4.5 Triggers and stored procedures on GPU

Triggers and stored procedures are classified into tiers:

* **Tier 1 — Simple predicates and field mutations** (e.g., `NEW.updated_at = now()`, `CHECK` constraints):  Compiled to GPU‑executable expressions and inlined into the batch execution kernel.
* **Tier 2 — Row‑level triggers with SQL side effects** (e.g., `INSERT INTO audit_log ...`):  The batch executor detects these during planning and appends secondary write operations to the batch.  For banking audit triggers, an optimized fast path appends audit rows in bulk as part of the GPU batch.
* **Tier 3 — Procedural logic** (PL/pgSQL with loops, conditionals, exception handling, dynamic SQL):  Executed on CPU.  Transactions containing Tier 3 triggers/procedures are routed to the CPU execution path.

A "GPU‑eligible" subset of trigger/procedure semantics is documented.  Users are advised how to stay within it for maximum performance.

## 5 Hybrid CPU–GPU and Multi‑GPU Scaling

The dataset in a core banking system often exceeds a single GPU's memory.  A hybrid design leverages CPU memory and multiple GPUs:

* **Hybrid CPU‑GPU execution:**  CPU memory for cold data, GPU memory for hot partitions.  CPU‑GPU DBMS designs exploit both CPU and GPU parallelism while minimizing data transfer overhead【64261708151474†L64-L71】.  Queries may execute partially on CPU and GPU, with results merged.
* **Cross‑device join strategy:**  When a join spans CPU‑resident and GPU‑resident data, the planner chooses between: (a) **ship‑inner‑to‑GPU** — transfer the smaller (inner) relation to GPU memory and execute the join on GPU; (b) **ship‑results‑to‑CPU** — scan the GPU‑resident side, stream qualifying rows to CPU, and join on CPU; (c) **partition‑wise join** — if both sides are partitioned on the join key, execute partition‑local joins on whichever device holds each partition pair, then union results.  The cost model (Section 5.1) selects the cheapest strategy based on data sizes and transfer costs.
* **GPU sort operations:**  GPU‑accelerated merge sort for `ORDER BY`, `GROUP BY` pre‑sorting, and sort‑merge joins.  Uses a bitonic sort network for small arrays (< 32K elements within a thread block) and a multi‑pass radix sort for larger datasets across global memory.  Sort stability is guaranteed for deterministic query results.
* **Multi‑GPU scaling:**  Multi‑GPU systems aggregate memory and compute power【64261708151474†L64-L75】.  Lancelot's **cache‑aware replication policy** selectively replicates data to balance caching and replication costs【64261708151474†L100-L110】.  The replication policy includes a resilience dimension: critical shards replicated on >= 2 GPUs.
* **Unified multi‑GPU abstraction:**  Treat multiple GPUs as a single logical device to reduce complexity【64261708151474†L91-L96】.  The abstraction maintains per‑device shard mappings, health state, and circuit breakers.  Query planners operate over the unified GPU memory space; the runtime routes operations to specific devices.

### 5.1 Cost model for CPU/GPU routing

The query planner uses an explicit cost model to decide execution device:

| Factor | Description |
|--------|-------------|
| **Data locality** | Is the partition GPU‑resident?  PCIe transfer cost (~25 GB/s PCIe 4.0, ~64 GB/s PCIe 5.0) added if not. |
| **Operator suitability** | Full scans, aggregations, hash joins favor GPU.  Point lookups, index‑nested‑loop joins with small outer tables favor CPU. |
| **Transfer cost** | Estimated bytes to move between CPU and GPU, including result set size for D2H. |
| **Batch size** | GPU execution requires sufficient parallelism to amortize kernel launch overhead.  Single isolated transactions below a threshold always execute on CPU. |
| **GPU queue depth** | Current utilization of GPU compute and memory.  If GPUs are saturated, route to CPU. |
| **Result size** | Large result sets increase D2H transfer cost. |

The planner produces cost estimates for pure‑CPU, pure‑GPU, and hybrid plans, then selects the cheapest.

**Adaptive re‑routing:**  If GPU resources are unavailable at execution time (not just plan time), the executor transparently re‑routes operators to CPU equivalents.  Every GPU operator has a CPU counterpart with identical semantics.

**Cost model calibration:**  Cost unit weights (cpu_tuple_cost, gpu_tuple_cost, gpu_transfer_cost_per_byte, gpu_kernel_launch_cost, seq_page_cost, random_page_cost) are exposed as GUC parameters with sensible defaults derived from hardware benchmarking.  A **self‑tuning mode** logs actual vs. estimated costs and periodically adjusts weights using linear regression on the logged data.  Initial defaults are calibrated on the development GPU (RTX 5090) and documented; production deployments are expected to re‑calibrate via `ANALYZE` + a calibration workload.

**Index‑only scans:**  When a query can be satisfied entirely from an index (covering index), the executor skips the heap fetch.  On GPU, index‑only scans on cuckoo hash tables return the version‑head pointer directly; if the visibility bitmap confirms all‑visible, no version array access is needed.  On CPU, the visibility map is consulted (matching PostgreSQL behavior).  Index‑only scans are a significant optimization for banking balance inquiries where the balance column is included in the index.

**Parallel sequential scans:**  For large analytical scans on CPU‑resident data, multiple worker threads scan disjoint page ranges in parallel, each feeding results into a shared result queue.  The planner decides the degree of parallelism based on table size, available CPU cores, and `max_parallel_workers_per_gather`.  GPU‑accelerated scans are inherently parallel (thousands of threads) and do not use this mechanism.

**Statistics collection:**  A statistics subsystem maintains per‑column histograms, most‑common‑values lists, n‑distinct estimates, null fractions, and correlation data (equivalent to `pg_statistic`).  Statistics are split by storage location (GPU‑resident vs. CPU‑resident).  An `ANALYZE` command (and auto‑analyze daemon) samples GPU‑resident data via lightweight GPU sampling kernels.  Data modification counters per shard trigger automatic re‑analysis.

### 5.2 Plan caching

* **Two‑tier cache:**  (a) parsed AST cache (keyed on normalized query text), (b) full physical plan cache including GPU kernel selection and device routing.
* **Invalidation:**  When a shard migrates between CPU and GPU, all cached plans referencing that shard are invalidated at the physical‑plan tier (AST tier retained).
* **Prepared statements:**  Per‑session cache stores parsed AST and optionally a generic plan.  After 5 executions, the engine decides whether to use the generic plan or re‑plan with specific parameter values (matching PostgreSQL behavior).  Compiled GPU kernel references cached alongside the plan.

### 5.3 GPU‑to‑GPU communication

* **Shard replication/migration:**  Use NCCL Broadcast or AllGather for replicating shards to multiple GPUs simultaneously (good fit for bulk collectives).
* **Point‑to‑point row fetches:**  Use `cudaMemcpyPeerAsync` for direct GPU‑to‑GPU transfers over NVLink/PCIe (lower latency for small, targeted transfers than NCCL).
* **Distributed joins/aggregations:**  Custom shuffle operator with double‑buffering via `cudaMemcpyPeerAsync`.  Profile NCCL AlltoAll as an alternative.
* **Topology detection:**  Use `cudaDeviceGetP2PAttribute` to detect NVLink vs. PCIe.  On PCIe‑only systems, route cross‑GPU data through CPU‑pinned staging buffers.
* **Multi‑node:**  GPUDirect RDMA for direct GPU‑to‑GPU transfers across machines (relevant for HA/replication).

### 5.4 Portability across NVIDIA GPU tiers

Although the initial prototype may be tuned on a **GeForce RTX 5090**, the implementation should be written to support higher‑end NVIDIA server GPUs as well.  NVIDIA's current compute‑capability table lists the **RTX 5090** at **12.0**, **GB200/B200** at **10.0**, **GH200/H200/H100** at **9.0**, and **A100** at **8.0**【turn527829view0†L16-L27】【turn527829view0†L43-L52】【turn527829view0†L102-L105】.

To remain portable, the `build.rs` script invokes NVCC to produce **multi‑architecture CUDA binaries** (fatbins) for targets `sm_80`, `sm_90`, `sm_100`, and `sm_120`.  Architecture‑targeted binaries, no PTX‑only dependency【turn527829view1†L61-L78】.  At startup, the engine queries device properties via `cudarc` and selects the matching SASS from the fatbin.  If no matching architecture is found, the engine logs a warning and falls back to CPU‑only mode rather than using PTX JIT (which has unpredictable compilation times in a latency‑sensitive banking context).

The runtime abstracts **topology differences** via profile‑driven configuration:

* kernel launch geometry and occupancy targets,
* memory‑pool sizing and eviction thresholds,
* inter‑GPU transport selection (PCIe vs. NVLink/NVSwitch),
* WAL apply batch sizes,
* GPUDirect Storage enablement,
* NCCL collective and peer‑to‑peer communication settings,
* CUDA Unified Memory aggressiveness (near‑zero overhead on NVLink‑C2C systems like GH200/GB200; disabled on PCIe‑only).

**Occupancy strategy:**  Target 50–75% occupancy for OLTP kernels (each thread needs 40–64 registers for MVCC + arithmetic + index access).  Use `__launch_bounds__` on all performance‑critical kernels.  Analytical scan kernels target maximum occupancy.  Profile with `ncu` (Nsight Compute) in CI.  **Register spilling:**  If a kernel exceeds the register budget (causing spills to local memory), the compiler flag `--maxrregcount` is used to cap register usage per kernel.  Spill rate is monitored in CI via `ncu` metrics (`l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_read`); kernels with spill rate > 5% are flagged for optimization.

**L2 cache partitioning (sm_80+):**  On H100/A100, use `cudaAccessPolicyWindow` to reserve a portion of the L2 cache for OLTP hot data (cuckoo hash index, current‑version table).  Prevents analytical scan traffic from evicting OLTP working set.  Default: 50% of L2 reserved for OLTP when analytical and OLTP workloads run concurrently.  Disabled on GPUs without L2 partitioning support.

**Tensor cores:**  Tensor cores (available on A100/H100/B200) are **not used** for OLTP operations — they are optimized for dense matrix multiply (FP16/BF16/INT8) which does not map to banking transaction processing or 128‑bit fixed‑point arithmetic.  Tensor cores may be evaluated in a future phase for GPU‑accelerated analytical workloads (matrix‑based joins or ML‑in‑database inference) but are out of scope for the core OLTP engine.

**GPU memory encryption:**  H100/H200 support Confidential Computing with hardware memory encryption (GPU TEE).  A100 and RTX 5090 do not.  For production banking deployments handling sensitive data, H100+ GPUs are recommended.

**Multi‑Instance GPU (MIG):**  For multi‑tenant deployments, H100/A100 MIG provides hardware‑level GPU isolation between tenants.

The practical rule is: **a solution designed on the 5090 can support higher‑end NVIDIA server GPUs, but only if the codebase is intentionally multi‑architecture and capability‑driven rather than 5090‑specific.**

### 5.5 Async memory operations and pinned memory

* **Pinned memory pool:**  Pre‑allocated at startup (4–16 GB).  All CPU‑GPU transfers use pinned memory via `cudaHostAlloc` with `cudaHostAllocPortable` for multi‑GPU visibility.
* **Double‑buffering:**  While the GPU processes batch N from buffer A, the CPU fills buffer B with batch N+1.  Swap each epoch.
* **All transfers use `cudaMemcpyAsync` on non‑default streams.**  Never use synchronous `cudaMemcpy` in the transaction path.
* **Result retrieval:**  D2H via `cudaMemcpyAsync` with `cudaLaunchHostFunc` callback to notify the connection handler, avoiding CPU thread blocking.
* **Unified Memory:**  Evaluated for the cold‑data access path (on‑demand page migration for cold partitions).  Not used for the hot OLTP path (page‑fault overhead unacceptable).  Enabled aggressively on NVLink‑C2C systems (GH200/GB200) where overhead is near‑zero.  CUDA Virtual Memory Management API (`cuMemCreate`, `cuMemMap`) for fine‑grained page‑table control.

## 6 Transaction Management and Consistency

### 6.1 Transaction manager

A dedicated **Transaction Manager** subsystem owns transaction lifecycle.  Both CPU and GPU execution paths call through a common interface:

* `txn_begin(isolation_level)` → allocates transaction ID.
* `snapshot_create()` → captures the set of committed transactions at a point in time.
* `visibility_check(tuple_header, snapshot)` → determines if a tuple version is visible.  GPU path uses a bulk variant: `batch_visibility_check(tuple_headers[], snapshot) → bitmap`.
* `lock_acquire(resource, mode)` / `lock_release(resource)` → lightweight lock manager using a lock‑free hash table.  Supports exclusive/shared row‑level locks, table‑level locks (for DDL), advisory locks.
* `txn_commit()` / `txn_abort()` → finalize transaction, update commit log.
* **Deadlock detection:**  Periodic deadlock detector (configurable interval, default 1 second) with wait‑for graph analysis.

### 6.2 MVCC and isolation

The engine adopts **multi‑version concurrency control** like PostgreSQL.  Each row carries transaction metadata (xmin/xmax).  When a row is updated, a new version is appended; readers continue to see the old version until they are serialized.  MVCC ensures that reading does not block writing and vice‑versa【997194816604558†L33-L47】.

### 6.3 Isolation levels

Support PostgreSQL's isolation levels: Read Committed, Repeatable Read and Serializable.  Serializable Snapshot Isolation (SSI) implemented by tracking conflict graphs and aborting transactions that would create cycles.  For deterministic batches executed on GPUs, serializable behavior arises naturally from fixed ordering.

### 6.4 Locking, hot rows and advisory locks

Though MVCC minimizes locking, some operations (schema changes, hot rows, advisory locks) still require locks.  The lock manager supports PostgreSQL‑compatible lock modes and exposes the advisory lock API.

**Hot‑row handling:**  In core banking, certain rows (e.g., internal settlement accounts) may be touched by a large fraction of transactions.  The deterministic batch model serializes all updates to a hot row within an epoch.  Between epochs, the batch planner detects hot‑row conflicts and serializes affected batches to maintain correctness.  For extreme hot rows (> 5 000 updates/s), a dedicated CPU fast‑path may bypass GPU batching.

### 6.5 MVCC garbage collection

* **GPU‑side (epoch‑based):**  Dead versions reclaimed at epoch boundaries.  The GPU GC respects the global "oldest active snapshot" horizon shared with the CPU.
* **CPU‑side (vacuum):**  A vacuum process identifies dead tuples (invisible to all active snapshots), reclaims space, updates the visibility map, and freezes old transaction IDs to prevent wraparound.  An **autovacuum daemon** triggers based on dead‑tuple thresholds and modification rates.
* **Coordination:**  GPU GC must not reclaim versions still visible to CPU‑side snapshots.  A global oldest‑active‑snapshot horizon ensures consistency.

## 7 Fault Tolerance, Replication and Recovery

### 7.1 Durable WAL and replication

The WAL is the authoritative log.  Synchronous replication ensures zero data loss (RPO = 0); asynchronous replication offers higher throughput at the cost of potential lag.  WAL replication health monitored per replica; if a replica falls behind a configurable threshold, it is removed from the synchronous set to avoid blocking commits.

**WAL sender/receiver protocol:**  The primary runs a **WAL sender** process per replica, streaming WAL records in real time via the replication sub‑protocol (`START_REPLICATION`).  Replicas run a **WAL receiver** process that connects to the primary, receives WAL, writes it to local segment files, and signals the local WAL applier.  Feedback messages from the receiver report `write_lsn`, `flush_lsn`, and `apply_lsn`, allowing the primary to track replica progress for synchronous‑commit acknowledgment and WAL retention decisions.  The protocol is wire‑compatible with PostgreSQL's streaming replication so that standard tools (`pg_basebackup`, `pg_receivewal`) work unmodified.

### 7.2 GPU failure modes and handling

#### 7.2.1 GPU health state machine

Each GPU maintains a health state: `HEALTHY` → `DEGRADED` → `FAULTED` → `RECOVERING`.

| Failure mode | Detection | Response |
|-------------|-----------|----------|
| ECC correctable error | `nvmlDeviceGetMemoryErrorCounter` polling (1–5 s) | Log, increment counter, page retirement |
| ECC uncorrectable error | Same | Quarantine GPU → FAULTED, failover shards to CPU/replica GPU |
| Kernel timeout | Watchdog thread checking stream events | Abort batch, re‑execute on CPU, GPU → DEGRADED |
| Driver crash / GPU reset | CUDA context validation | GPU → FAULTED, invalidate all cached shards, rebuild from WAL |
| Thermal throttling | NVML temperature/clock monitoring | Reduce batch size, shed load to CPU, GPU → DEGRADED |
| GPU memory exhaustion | `cudaMalloc` failure | Abort batch, evict cold shards, retry; if retry fails, open circuit breaker |

#### 7.2.2 Graceful CPU fallback

The system operates at full correctness in **CPU‑only mode**.  A runtime capability flag (`gpu_available`) is checked by the planner.  When false, all plans use CPU‑only operators.  Transition between modes is transparent to connected clients.

If a GPU fails mid‑batch: abort the batch, return all transactions to the CPU‑side queue, re‑execute on CPU.  Since the WAL was not flushed for uncommitted batches, no durability invariant is violated.

#### 7.2.3 Partial multi‑GPU failure

* **Device‑to‑shard mapping** queryable by the failure handler.  On GPU loss, immediately identify affected shards and replica availability.
* Shards whose sole copy was on the failed GPU transition to `STALE` and are served from CPU/disk until rebuilt on a surviving GPU.
* **Minimum GPU quorum:**  If > 50% of GPUs fail, degrade to CPU‑only mode rather than overloading survivors (prevents thermal cascade).
* **Bulkheading:**  Each GPU has independent circuit breakers, health state, and queue limits.  A failure in GPU‑0 does not affect GPU‑1.

### 7.3 Circuit breakers and backpressure

* **GPU circuit breaker:**  Three states: CLOSED (normal), OPEN (GPU failing — all work routed to CPU), HALF‑OPEN (probe with a small batch to test GPU health).  Transition from CLOSED → OPEN after N consecutive failures or error rate exceeding threshold.
* **Maximum queue depth** for pending GPU batches.  When full, route new transactions to CPU or return a retriable error.  Never allow unbounded queue growth.
* **Per‑batch execution timeout.**  If a GPU kernel does not complete within the timeout, trigger GPU reset and open the circuit breaker.
* **GPU latency monitoring:**  Track p50/p95/p99.  If p99 exceeds threshold, proactively shed load to CPU.

### 7.4 Checkpointing and crash recovery

* **Checkpointing:**  See Section 3.7.8.
* **Crash recovery:**  On startup, load the latest checkpoint and replay the WAL to reconstruct CPU state.  Begin accepting transactions in **CPU‑only mode** immediately ("progressive recovery").  Repopulate GPU memory in the background by streaming WAL via GPUDirect.  Transition to hybrid mode once GPU state is current.  EPIC's epoch boundaries limit version data to reconstruct【150335027845516†L108-L118】.
* **GPU recovery time:**  For 24 GB of GPU data, replay at 10 GB/s (PCIe 4.0) ≈ 2.4 s plus kernel processing.  Target < 5 minutes for full GPU recovery including WAL replay.

### 7.5 High availability and Raft consensus

* **Strict majority quorum** for all write commits (3‑node: 2 ACK; 5‑node: 3 ACK).  System cannot tolerate simultaneous failure of a majority.
* **Leader lease** with bounded duration.  Primary stops accepting reads and writes if lease expires without renewal (critical for linearizability during network partitions).
* **Fencing tokens:**  Every WAL entry carries a monotonically increasing epoch number tied to the Raft term.  Storage and replication layers reject writes with stale epoch numbers.  Prevents zombie‑primary data corruption.
* **Leader activation delay:**  New primary waits until all committed log entries are applied before accepting writes.
* **Connection draining on failover:**  On planned failover: stop new connections, allow in‑flight transactions to complete (30 s timeout), terminate remaining with `57P01 admin_shutdown`.  On crash: new primary returns `57P03 cannot_connect_now` until recovery completes.
* **Minimum production topology:**  3 nodes minimum, 5 recommended, spread across failure domains.
* **Clock skew tolerance:**  Transaction ordering relies on LSNs (not wall‑clock timestamps), making it immune to clock drift.  However, leader leases depend on bounded clock skew between nodes.  The lease duration must exceed the maximum expected clock skew.  Require NTP synchronization with a maximum drift of 500 ms; the default lease duration of 10 seconds provides ample margin.  `now()` timestamps in SQL queries use the local wall clock; for cross‑node consistency of application‑visible timestamps, NTP is required but not enforced by the engine (documented as an operational requirement).  Hybrid Logical Clocks (HLC) are considered as a future enhancement for causally consistent timestamps across nodes.

### 7.6 Network partition handling

* **Minority partition:**  Node partitioned from Raft majority stops accepting writes within leader lease timeout.  Clients receive specific error code indicating read‑only/unavailable.
* **Intra‑node GPU communication failure:**  NVLink/PCIe error detected via NCCL or peer‑to‑peer transfer failure.  Isolate affected GPU pair, fall back to CPU‑mediated transfer or mark unreachable GPU as faulted.
* **Asymmetric partitions:**  Raft handles correctly; tested explicitly.

### 7.7 Cascading failure prevention

* **Per‑GPU thermal/utilization ceilings:**  If redistributing load from a failed GPU would push survivors above 80% utilization, route excess to CPU.
* **WAL retention hard cap:**  If synchronous replicas cannot keep up, switch lagging replicas to asynchronous with operator notification.  Never allow WAL backlog to exhaust disk space.
* **Connection‑layer load shedding:**  When at capacity, reject new connections/transactions with retriable error.
* **Global system health score:**  Aggregates GPU health, CPU load, memory pressure, WAL lag, replication status.  Below threshold: activate degraded mode (disable analytical queries, prioritize OLTP).

### 7.8 Graceful shutdown and signal handling

* **SIGTERM (smart shutdown):**  Stop accepting new connections.  Wait for all active sessions to disconnect voluntarily.  Checkpoint.  Shut down GPU contexts.  Exit.
* **SIGINT (fast shutdown):**  Stop accepting new connections.  Abort all in‑flight transactions.  Drain GPU batch queues (wait up to 5 s).  Checkpoint.  Shut down GPU contexts.  Exit.
* **SIGQUIT (immediate shutdown):**  Emergency exit without checkpoint.  Recovery from WAL on next startup.
* **Shutdown sequence:**  (1) Set accepting_connections = false, (2) drain connections per mode, (3) flush WAL, (4) execute checkpoint, (5) `cudaDeviceSynchronize` on all GPUs, (6) destroy CUDA contexts, (7) close files, (8) exit.
* **Long‑running query timeout:**  `statement_timeout` GUC parameter (configurable per session).  GPU‑side enforcement via the kernel watchdog — if a batch exceeds the timeout, it is aborted and clients receive SQLSTATE `57014 query_canceled`.
* **Async‑signal safety:**  Signal handlers only set atomic flags (`AtomicBool` with `Ordering::SeqCst`).  All shutdown logic runs in the Tokio event loop after the flag is observed — never in the signal handler itself.  This avoids calling non‑async‑signal‑safe functions (allocator, mutex, CUDA API calls) from signal context.  Tokio's `signal::unix::signal()` API is preferred to avoid raw signal handlers entirely, integrating signals into the async event loop as `Stream` items.

### 7.9 Point‑in‑time recovery (PITR)

Combine base backups with continuous WAL archiving.  Recovery accepts a target LSN, transaction ID, or timestamp and replays WAL up to that point.  Parameters: `recovery_target_time`, `recovery_target_lsn`, `recovery_target_xid`, `recovery_target_action` (pause, promote, shutdown).  GPU state is irrelevant during PITR — recovery targets CPU/disk state; GPU caches rebuilt afterward.

**Timeline management:**  Each PITR recovery increments a **timeline ID** (stored in the control file).  WAL segments include the timeline ID in their filename, preventing confusion between pre‑ and post‑recovery WAL histories.  This allows creating a "tree" of recovery histories for auditing and regulatory forensics.

### 7.10 Backup strategies

* **Physical base backup (online):**  Force checkpoint, record start LSN, copy data files while system operates, record end LSN.  Consistent when combined with WAL between start and end LSNs.
* **Incremental backup:**  Track per‑page "last modified LSN"; copy only pages modified since previous backup's LSN.  Manifest with block checksums for verification.
* **Logical backup:**  `pg_dump`‑compatible utility for cross‑engine migration and partial restores.
* **Backup verification:**  `pg_verifybackup`‑equivalent checking file checksums against manifest and WAL segment availability.

### 7.11 Multi‑region disaster recovery

Banking regulators require geographic redundancy.  The engine supports multi‑region deployments:

* **Active‑passive DR:**  A primary Raft cluster in Region A with asynchronous WAL streaming to a standby cluster in Region B.  RPO > 0 for cross‑region (bounded by replication lag; target < 1 s under normal conditions).  RPO = 0 within the primary region (synchronous replication).
* **Failover trigger:**  Manual operator decision (regulatory requirement — automated cross‑region failover can cause split‑brain with WAN partitions).  Automated monitoring alerts when cross‑region lag exceeds threshold.
* **Standby promotion:**  Region B standby promoted to primary via Raft reconfiguration.  DNS/load‑balancer update directs clients to new primary.  GPU caches rebuilt from local WAL on the newly promoted nodes.
* **WAL archiving for DR:**  Continuous WAL archiving to cross‑region object storage (S3 Cross‑Region Replication, GCS multi‑region buckets).  Guarantees PITR capability even if both Region A and Region B suffer simultaneous failure.
* **Network bandwidth:**  Cross‑region WAL streaming requires sustained bandwidth proportional to write rate.  At 200K TPS with ~200 bytes per WAL record average, WAL generation is ~40 MB/s; cross‑region link must sustain this with headroom.
* **Operational runbook:**  DR failover, failback, and split‑brain resolution procedures are documented as part of Phase 5 compliance certification.

### 7.12 Cross‑device consistency verification

* Periodically compute checksum (xxHash) over each shard on each device (GPU and CPU) and compare.  Divergence triggers alert and re‑synchronization from WAL.
* After applying WAL batches to GPU structures, hash affected pages and compare against CPU‑side expected hash.  Catches GPU kernel bugs and silent memory corruption.
* For replicated shards in critical banking paths, consider quorum reads (read from two copies and compare) for operations like balance queries.

### 7.13 Startup validation and warm‑up

The engine performs a deterministic startup sequence before accepting connections:

1. **Configuration validation:**  Parse and validate all GUC parameters.  Detect conflicting settings (e.g., `synchronous_commit = off` with `wal_level = logical`), missing required parameters, and values outside valid ranges.  Abort startup with a clear diagnostic on invalid configuration.
2. **Control file and WAL integrity check:**  Read the control file, verify its CRC, and check WAL format version compatibility.  If the control file is corrupt, attempt recovery from the backup copy.  If both copies are unreadable, abort with instructions for manual recovery.
3. **Crash recovery:**  If the control file indicates an unclean shutdown, replay WAL from the last checkpoint's redo point.  Corrupted WAL records (CRC mismatch) during recovery are handled as follows: if the corruption is in the last partial WAL segment (tail corruption), recovery stops at the last valid record — this is expected for crash recovery.  If corruption occurs before the last known‑good flush point, recovery aborts with a diagnostic; operator intervention is required (restore from replica or backup).
4. **GPU discovery and validation:**  Enumerate CUDA devices via `cudaGetDeviceProperties`.  For each device: verify compute capability is in the supported set (sm_80+), check ECC status (warn if disabled), verify driver version compatibility, allocate a test buffer and run a diagnostic kernel (arithmetic + memory access).  Devices failing validation are marked `FAULTED` and excluded.  If no valid GPU is found, start in CPU‑only mode with a warning.
5. **Memory pool initialization:**  Allocate CPU shared buffer pool, pinned memory pool (`cudaHostAlloc`), and GPU buffer pools.  If allocation fails (insufficient memory), reduce pool sizes to minimums or abort with guidance.
6. **Shared buffer preloading (warm‑up):**  Optionally preload configured tables/indexes into the CPU buffer pool and GPU memory at startup (`gpu_db.preload_tables` GUC parameter).  For banking systems, preloading the accounts table and primary‑key index eliminates cold‑start latency spikes.  Preloading is asynchronous — the engine accepts connections as soon as recovery is complete, but marks preloading tables as "warming" in `pg_stat_gpu`.
7. **Readiness gate:**  Set accepting_connections = true only after steps 1–5 complete.  Readiness probe returns healthy.  Liveness probe returns healthy as soon as the process starts and the event loop is running.

### 7.14 Health checking and probes

* **Liveness probe:**  Process alive, event loop running.  Does not depend on GPU or disk.  Failure triggers restart.
* **Readiness probe:**  Engine ready to accept connections.  Returns not‑ready during startup, recovery, or primary step‑down.
* **Health detail endpoint:**  Comprehensive status: GPU device states, memory per GPU, WAL lag, replication status, Raft role, circuit breaker states, recent error rates.
* **Internal GPU health polling:**  Every 1–5 seconds: CUDA context validity, ECC counters, temperature, memory utilization, kernel completion.
* **Dead‑man's switch:**  If no GPU batch completes within configurable timeout, assume GPU hang and initiate recovery.

### 7.15 Client retry semantics

Clients receiving transient errors must know how to retry safely.  The engine provides explicit guidance via error responses:

* **Serialization failures (SQLSTATE 40001):**  Client should retry the entire transaction immediately.  No backoff needed — the deterministic batch model resolves conflicts within the next epoch.
* **GPU resource exhaustion (SQLSTATE 53200):**  Client should retry with exponential backoff (initial 10 ms, max 1 s).  The error indicates the transaction was routed to CPU but CPU capacity was also exhausted.  `Retry‑After` hint included in the error detail field.
* **Connection limit exceeded (SQLSTATE 53300):**  Client should retry with backoff or use a connection pool.  The engine queues connection attempts for `connection_queue_timeout` (default 5 s) before rejecting.
* **Admin shutdown (SQLSTATE 57P01):**  Client should reconnect to the cluster (via load balancer or DNS).  This is a planned failover; the new primary should be available within the RTO.
* **Crash recovery in progress (SQLSTATE 57P03):**  Client should retry with backoff until the readiness probe reports healthy.
* **Idempotency support:**  For banking transactions, clients should include an application‑level idempotency key.  The engine provides a `gpu_db.last_committed_idempotency_key(key)` function that checks whether a transaction with the given key has already committed, preventing duplicate execution after retries.

### 7.16 Rate limiting and tenant resource isolation

* **Per‑client rate limiting:**  Configurable TPS limit per authenticated role (`ALTER ROLE ... SET gpu_db.max_tps = 5000`).  Excess transactions receive SQLSTATE `53300` with a `Retry‑After` hint.  Rate limiting uses a token‑bucket algorithm per session, refilled at the configured rate.
* **Per‑database resource quotas:**  GPU memory quota, CPU connection limit, and WAL generation rate limit per database.  Prevents a runaway database from starving others.
* **Multi‑tenant isolation (MIG):**  On H100/A100 with MIG enabled, each tenant is assigned a MIG instance with hardware‑isolated GPU memory and compute.  Shard placement respects MIG boundaries.
* **Query timeout enforcement:**  `statement_timeout` (per‑session) and `gpu_db.max_batch_timeout` (system‑wide) enforce wall‑clock limits on both CPU and GPU execution paths.

## 8 Security and Compliance

Core banking systems demand stringent security.  **Security is integrated from Phase 1, not bolted on later.**  The storage API includes a security context parameter from the start, and the tuple access path has a hook point for RLS policy evaluation.

* **Authentication and authorization:**  Integrate with enterprise identity providers (Kerberos, LDAP, OAuth).  SCRAM‑SHA‑256 as default authentication.  Support row‑level security (RLS) policies — enforced on both CPU and GPU execution paths (GPU kernels apply RLS predicates before returning results).
* **Encryption:**  Data at rest (disk, backups) and in transit (TLS 1.2+ minimum, no weak ciphers).  GPU memory encryption via Confidential Computing on H100+ GPUs.  On non‑CC GPUs, document the risk of plaintext GPU memory.
* **Audit logging:**  Tamper‑evident logs using **cryptographic hash chains** (each audit record includes the hash of the previous record).  Separate from operational logs.  Stored on append‑only/write‑once storage where available.  Records: client identity, SQL statements, affected rows, timestamp, outcome.  Audit log entries cannot be deleted or modified by database administrators.
* **Compliance:**  PCI DSS (Requirements 3, 7, 8, 10 mapped to engine features), GDPR, Basel III, SOC 2 Type II.  Data masking functions enforced on GPU paths.  GDPR data deletion verified end‑to‑end: heap pages, MVCC old versions, WAL segments after archival, GPU caches on all devices, replicated copies, backup snapshots.  Retention policies and multi‑region disaster recovery.

## 9 Observability

### 9.1 Structured logging

Structured logging via the **`tracing`** crate with a JSON‑formatted subscriber (`tracing-subscriber` with `fmt::json`).  Every log span/event includes: timestamp, severity, component (protocol, planner, GPU executor, WAL, replication), connection ID, transaction ID, batch ID.  Separate compliance audit log (tamper‑evident, append‑only) from operational log.  Runtime log‑level changes without restart (`SET gpu_db.log_level = 'debug'`) via `tracing`'s dynamic filter reload.

### 9.2 Metrics

Prometheus‑compatible metrics endpoint:

* Transactions per second (by batch, by GPU, CPU‑only)
* Batch size histogram and batching wait time
* GPU memory utilization per device
* GPU kernel execution time per batch
* GPU temperature, power draw, SM utilization, ECC error counts (via NVML/DCGM)
* WAL write/flush/replay rates and latency
* Connection count, query latency percentiles (P50, P95, P99)
* Replication lag per replica
* Error rates by category
* Plan cache hit rate, statistics freshness

### 9.3 EXPLAIN and query tracing

`EXPLAIN` and `EXPLAIN ANALYZE` show the hybrid execution plan:

* Which operators execute on CPU vs. GPU (with device ID)
* Estimated vs. actual row counts per operator
* GPU kernel launch parameters
* Data transfer volumes CPU ↔ GPU
* Batch wait time
* CPU planning time vs. GPU execution time

OpenTelemetry distributed tracing: a single transaction traced from client connection through batching, GPU execution, WAL write, and response.

### 9.4 System catalog views

PostgreSQL‑compatible `pg_stat_activity`, `pg_stat_user_tables`, `pg_stat_user_indexes`, `pg_stat_bgwriter`, `pg_stat_statements`.  Extended with GPU columns (device ID, batch wait time, GPU kernel time, data transfer time).

Custom `pg_stat_gpu` view: per‑device memory utilization, kernel occupancy, batch queue depth, shard cache hit rate, eviction rate, ECC error counts.

### 9.5 Administrative operations

Admin API (gRPC or REST) and SQL commands for:

* GPU device status inspection (`SELECT * FROM pg_stat_gpu`)
* Manual partition migration (`ALTER TABLE ... SET GPU = ...`)
* GPU drain mode (stop scheduling, wait for in‑flight, evacuate data)
* GPU reinitialization after failure
* Configuration changes (GUC parameters: `SET` / `SHOW` / `RESET`)
* Per‑session, per‑database, and system‑wide parameter scopes

### 9.6 I/O scheduling

The I/O scheduler prioritizes:

1. WAL writes (highest — on commit critical path)
2. Foreground data reads for active queries
3. Prefetch reads (sequential readahead for analytical scans, GPUDirect prefetch to GPU)
4. Background checkpoint writes
5. Analytical scan reads (lowest, throttleable)

Use `O_DIRECT` for WAL and checkpoint writes.  GPUDirect Storage for transfers > 64 KB; pinned‑memory `cudaMemcpyAsync` for smaller transfers.

## 10 Testing Strategy

### 10.1 Compatibility testing

* **PostgreSQL regression suite:**  Adapt upstream `src/test/regress` (200+ test files).  Track compatibility score per release.  Phase gates: Phase 1 >= 60%, Phase 4 >= 95%.
* **pgTAP tests:**  For every SQL feature claimed in Section 2.3.
* **Driver test suites:**  pgjdbc, psycopg2, npgsql test suites run against the engine.
* **ORM validation:**  Django, Rails (ActiveRecord), Hibernate against representative banking schemas.
* **Compatibility matrix:**  Every PostgreSQL feature mapped to: fully supported, partially supported (deviations documented), or not supported.

### 10.2 GPU/CPU correctness verification

* **Dual‑execution harness:**  Every query run on both CPU‑only and GPU paths; results compared row‑by‑row with deterministic ordering.  CI‑mandatory.
* **Fixed‑point arithmetic tests:**  Verify exact decimal results for all arithmetic operations and rounding modes (ROUND_HALF_EVEN).
* **MVCC visibility equivalence:**  Transactions reading GPU‑cached data see exactly the same versions as CPU readers given the same snapshot.
* **Partial batch failure:**  Verify rollback correctly restores GPU‑resident versions.

### 10.3 Deterministic concurrency testing

* **Linearizability checker:**  Integrate Elle or Jepsen's knossos.  Validate GPU transaction histories against serial specification.  >= 10 000 randomized sequences per CI build.
* **Replay harness:**  Record batch ordering and read/write sets; replay on single‑threaded CPU executor.  Results must match.
* **Epoch boundary testing:**  Transactions spanning epoch boundaries, GC at boundaries, concurrent reads during transitions.

### 10.4 Performance regression testing

* **Continuous benchmarking pipeline:**  Fixed hardware, run on every merge to main.
* **Workloads:**  TPC‑B/pgbench (baseline OLTP), custom banking workload (transfers, balance inquiries, statement generation, end‑of‑day batch), TPC‑H subset (analytical).
* **Statistical regression detection:**  Change‑point detection or Mann‑Whitney U tests.  Not fixed thresholds.
* **Tracked metrics:**  P50/P95/P99 latency, TPS, GPU memory/compute utilization, CPU‑GPU transfer volume, WAL throughput, batch formation latency.

### 10.5 Chaos and fault injection

* **GPU fault injection:**  GPU device reset mid‑transaction (`cudaDeviceReset`), GPU memory allocation failure, NVLink degradation, GPUDirect I/O error, PCIe timeout.
* **Crash recovery:**  Kill process at every WAL write stage.  Verify consistent recovery with no phantom/lost transactions.
* **Checkpoint corruption:**  Truncate/corrupt checkpoint file; verify fallback to earlier checkpoint + WAL replay.
* **Raft partition testing:**  Leader failure, follower failure, network partition, split‑brain scenarios.  Jepsen or Toxiproxy.
* **GPU mid‑batch failure:**  Verify batch failover to CPU.
* **Replication lag:**  Synchronous replica unreachable; verify primary behavior.

### 10.6 Data integrity verification

* **Background consistency checker:**  Periodically verifies GPU↔disk data match, MVCC version chain integrity, index‑heap consistency, cross‑GPU shard equality.
* **Banking‑specific:**  "Double‑spend" test (concurrent debits on same account), end‑of‑day ledger reconciliation (sum debits = sum credits).

### 10.7 Security testing

* **Wire protocol penetration testing:**  Buffer overflows, authentication bypass, state machine violations.
* **RLS on GPU paths:**  Verify GPU kernels cannot bypass RLS filters.
* **GPU memory isolation:**  Verify one session cannot read another's GPU memory.
* **TLS enforcement, certificate validation, cipher negotiation.**
* **Audit log completeness and immutability.**

### 10.8 Load and stress testing

* **GPU memory pressure:**  Gradually increase working set to exhaustion; measure latency degradation curve.  Test with working set 10× GPU memory.
* **72‑hour sustained load:**  Detect memory leaks, GPU fragmentation, WAL growth, GC failures.
* **Thundering herd:**  10 000 connections querying the same hot account simultaneously.
* **Cross‑GPU communication stress:**  Maximize cross‑GPU transfer and measure throughput degradation.
* **Backpressure verification:**  When GPU falls behind, verify graceful degradation (queue/slow) not catastrophic failure (OOM/deadlock).
* **Batch formation at low load:**  At 10 TPS, verify time‑deadline trigger fires and latency remains acceptable.

### 10.9 Upgrade and migration testing

* **Version upgrade testing:**  Verify that WAL format changes between engine versions are detected by the control file's format version field.  Upgrades requiring WAL format migration use a `pg_upgrade`‑compatible utility or logical dump/restore.
* **Rolling upgrade:**  Test Raft cluster upgrade with one node at a time (stop, upgrade, restart).  Verify that mixed‑version clusters maintain replication compatibility during the upgrade window.
* **Data migration from PostgreSQL:**  Test `pg_dump` import from PostgreSQL 14/15/16 into the engine.  Verify schema, data, and constraint fidelity.

### 10.10 Test infrastructure

* **CI/CD with GPU runners:**  Define minimum test environment (GPU model, driver version, CUDA toolkit version).  GPU‑equipped CI from Phase 1.
* **Sanitizers in CI:**  Debug builds with Rust's built‑in bounds checking and overflow detection.  `cargo-miri` for detecting undefined behavior in `unsafe` code.  CUDA‑memcheck and compute‑sanitizer for GPU memory errors.  `cargo-geiger` to audit `unsafe` usage.  For the CUDA C kernel code: AddressSanitizer and compute‑sanitizer.
* **Protocol fuzzing:**  `cargo-fuzz` (libFuzzer‑based) targeting the wire‑protocol parser and SQL plan translator.  Grammar‑based fuzzing for SQL parser via `libpg_query` test corpus.  `arbitrary` crate for structured fuzz input generation.
* **Property‑based testing:**  `proptest` crate for MVCC visibility rules, transaction ordering, and serialization equivalence.
* **Test coverage:**  Target >= 80% line coverage for core subsystems (storage, WAL, transaction manager, protocol).  GPU kernel coverage tracked via Nsight Compute profiling (SM instruction coverage).
* **Banking‑specific synthetic data generator:**  Realistic distributions (Benford's law for amounts, temporal patterns, realistic account hierarchies).  Non‑PII.

## 11 Implementation Plan

The following phases outline an implementation roadmap:

### Phase 1 – Protocol shell, CPU engine, and security foundation

1. **Protocol shim:**  Implement startup/authentication (SCRAM‑SHA‑256), simple/extended query handling, pipelining, COPY.  Session manager with async I/O event loop.
2. **SQL parser and planner:**  Integrate `libpg_query` (C library).  Implement cost‑based planner with table scans, index scans, joins, aggregations.  Plan cache (AST tier).  Statistics collector with `ANALYZE` command.
3. **Storage layer:**  Row‑oriented storage with MVCC metadata, CPU shared buffer pool (ARC replacement), disk‑based WAL with hybrid record format (little‑endian, CRC‑32C), control file, full‑page writes, page‑level checksums.  B‑tree indexes.  Crash recovery.  Background writer.  Vacuum and autovacuum.  Online DDL support (schema‑change locks, `CREATE INDEX CONCURRENTLY`).
4. **Security foundation:**  Security context parameter in storage API.  RLS hook point in tuple access path (initially permissive).  Audit log with hash chains.  TLS for wire protocol.
5. **Observability foundation:**  Structured logging, Prometheus metrics endpoint, basic `pg_stat_*` views, `EXPLAIN`.
6. **System catalogs:**  `pg_catalog` views for tool compatibility.
7. **Configuration:**  GUC parameter system with file/runtime/per‑session scopes.  Startup validation sequence with configuration conflict detection.
8. **Rate limiting and resource isolation:**  Per‑role TPS limits, per‑database resource quotas, connection queuing.
9. **Testing:**  PostgreSQL regression suite (target >= 60% pass rate), pgTAP, dual‑execution harness (CPU‑only at this phase), CI pipeline with `cargo-miri`, `cargo-geiger`, `cargo-fuzz`, protocol fuzzing.

### Phase 2 – GPU engine integration

1. **GPU buffer pool:**  Slab allocator, page table, CLOCK replacement, dirty tracking, memory pressure levels.  Pinned memory pool.
2. **WAL streaming:**  GPUDirect Storage integration with pinned‑memory fallback.  GPU WAL applier kernel.
3. **Batched concurrency control:**  Dual‑trigger adaptive batching, transaction‑type binning, epoch‑based MVCC.  CUDA stream pipeline (3 streams/GPU).  CUDA Graphs for kernel pipeline.
4. **GPU operators:**  Kernels for scans, filters, point reads/updates, simple aggregations, sorts (bitonic/radix).  SoA version arrays with null bitmaps, current‑version table, visibility bitmaps.  Cuckoo hash indexes.  GPU expression evaluator (bytecode + NVRTC JIT for hot expressions).
5. **Error handling:**  GPU error propagation via status arrays, partial batch failure, circuit breaker, GPU health state machine.  CUDA error checking macros wrapping all API calls.  GPU backend abstraction with mock implementation for CPU‑only testing.
6. **Cost model:**  CPU/GPU routing in planner.  Adaptive re‑routing at execution time.
7. **Trigger classification:**  Tier 1 (GPU‑inlined), Tier 2 (bulk audit append), Tier 3 (CPU fallback).
8. **Testing:**  Dual‑execution harness (CPU vs. GPU), linearizability checker, GPU fault injection, compute‑sanitizer, performance benchmarks.

### Phase 3 – Multi‑GPU and hybrid execution

1. **Unified GPU abstraction:**  Device‑to‑shard mapping with health state and circuit breakers per device.  Cache‑aware replication with resilience dimension.
2. **Distributed query planner:**  Hybrid plans splitting work between CPU and GPUs.  Operator pipelining, async execution, result merging.
3. **Cross‑device coordination:**  Transaction coordination across GPUs.  Cross‑device consistency verification (checksum comparison).  Quorum reads for critical paths.
4. **GPU‑to‑GPU communication:**  NCCL for broadcast/replicate, `cudaMemcpyPeerAsync` for point‑to‑point, topology detection.
5. **Testing:**  Multi‑GPU chaos testing, cross‑GPU consistency verification, partition testing, load/stress testing.

### Phase 4 – Advanced SQL, banking features, and extensions

1. **Full SQL coverage:**  Window functions, recursive queries, stored procedures, triggers, extensions.  FDW interface.  Two‑phase commit.
2. **Secondary indexes and analytics:**  GPU‑accelerated columnar operators, multi‑column indexes.  Analytical functions for reporting and compliance.
3. **Security hardening:**  Full RLS enforcement on GPU paths, data masking, GDPR end‑to‑end deletion verification, GPU memory encryption on H100+.
4. **Extension framework:**  Type/operator/AM registration, planner hooks.  GPU kernel registration by extensions.
5. **PITR and backup:**  Point‑in‑time recovery with timeline management, physical/incremental/logical backup, backup verification.  WAL sender/receiver for streaming replication.  Rolling upgrade support.
6. **Logical decoding and CDC:**  Logical decoding framework with pgoutput‑compatible output plugin.  Replication slots with LSN tracking.  Debezium compatibility testing.
7. **Testing:**  Compatibility target >= 95%.  ORM validation.  Security penetration testing.  GDPR/PCI DSS control mapping tests.

### Phase 5 – Hardening and certification

1. **Performance tuning:**  Benchmark on banking workloads.  Tune batch sizes, replication policies, GPU kernels.  Latency‑throughput curve documentation.  72‑hour stress tests.
2. **HA hardening:**  Raft with fencing tokens, leader leases, connection draining.  Jepsen testing.  Network partition tests.  RTO/RPO verification.
3. **Compliance certification:**  Engage auditors for PCI DSS, SOC 2.  SBOM.  Operational procedures documentation.
4. **Operational tooling:**  GPU drain mode, online shard migration, admin API, monitoring dashboards, graceful shutdown (SIGTERM/SIGINT/SIGQUIT handling), GPU firmware update procedures.

## 12 Conclusion

This design combines the reliability and features of PostgreSQL with the computational power of GPUs.  By maintaining a disk‑based WAL with the WAL‑before‑visibility invariant and using GPU memory as an execution cache, the engine ensures durability while accelerating workloads.  A multi‑GPU, hybrid CPU–GPU architecture with cache‑aware replication allows the system to handle datasets larger than a single GPU's capacity【64261708151474†L64-L75】.  Adaptive batching with dual‑trigger formation balances GPU throughput against banking latency SLAs.  Full CPU‑only fallback guarantees correctness when GPUs are unavailable.

The design addresses critical concerns identified through multi‑perspective review: explicit interface contracts between all subsystems, a cost model for CPU/GPU routing, comprehensive error taxonomy and GPU error propagation, page‑level checksums and full‑page writes for data integrity, circuit breakers and backpressure for resilience, security integrated from Phase 1, and a testing strategy spanning compatibility, correctness, chaos, and performance.

Adhering to PostgreSQL 16's protocol and SQL semantics (with `libpg_query` for grammar parity and `pg_catalog` for tool compatibility) ensures that existing banking middleware, ORMs, and monitoring tools operate without modification.  The phased implementation plan outlines a path from a minimal protocol shim to a fully featured, highly available, GPU‑accelerated RDBMS suitable for core banking systems.

## Appendix: Conceptual Architecture Diagram

The following diagram illustrates the high‑level architecture of the proposed engine.  It shows the CPU host with disk‑resident storage and WAL, multiple GPUs with their own memory partitions, and clients connecting via the PostgreSQL wire protocol.  Arrows depict the flow of WAL data from disk to GPU memory and the data flow between CPU and GPU.

![Hybrid CPU–GPU database architecture]({{file:file-1xyzngUSjQuKhw8ZA5NpL7}})
