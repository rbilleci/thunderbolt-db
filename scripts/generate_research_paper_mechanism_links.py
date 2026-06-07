#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


HEADING_RE = re.compile(r"^### (?P<date>\d{4}-\d{2}-\d{2}) - (?P<title>.+)$", re.MULTILINE)
CATEGORY_RE = re.compile(r"^\*\*Category:\*\*\s*(?P<value>.+)$|^Category:\s*(?P<plain>.+)$", re.MULTILINE)
TAGS_RE = re.compile(r"^\*\*Relevance tags:\*\*\s*(?P<value>.+)$", re.MULTILINE)
CITATION_RE = re.compile(r"^\*\*Citation:\*\*\s*(?P<value>.+)$", re.MULTILINE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")

METADATA_SNIPPET_PREFIXES = (
    "**citation:**",
    "**category:**",
    "category:",
    "**relevance tags:**",
)

STRONG_SNIPPET_PREFIXES = (
    "**core idea:**",
    "**concrete mechanisms:**",
    "**gpu db mapping:**",
    "**risks and mismatches:**",
)

VISIBLE_EVIDENCE_ALIASES: dict[str, list[str]] = {
    "bounded_descriptor_reclamation": [
        "bounded conflict paths",
        "cold-tier movement",
        "completion states",
        "control path",
        "descriptor",
        "ephemeral version metadata",
        "fallback legality",
        "generation counters",
        "generation boundaries",
        "concurrent sharing",
        "data-intensive applications",
        "fast-tier headroom",
        "gpu direct storage",
        "gpu execution",
        "gpu global memory",
        "gpu operators",
        "heap budgets",
        "admission key",
        "hidden host resources",
        "hot-cache granularity",
        "hot-key resident tier",
        "kernel-space fast paths",
        "learned query optimizers",
        "lifetime",
        "lookup metadata",
        "memory budget",
        "metadata movement",
        "modern nics",
        "many short, balanced rays",
        "nic hardware timestamp",
        "optimizer settings",
        "named, bounded units of work",
        "old descriptors",
        "operation orders",
        "order-preserving dictionaries",
        "ordering, ownership, and resource safety",
        "owner-published generation handles",
        "parallel multi-plan",
        "per-request churn",
        "profile",
        "publish/install phase",
        "resource budget",
        "resource conflict",
        "resident invalidation",
        "recovery path",
        "record/range level",
        "relation/partition identity",
        "reusable buffer",
        "registered or reusable buffers",
        "reuse",
        "route-history",
        "route envelope",
        "route generations",
        "route hints",
        "safe-frontier scheduling",
        "safe cleanup",
        "safe prefetch envelope",
        "safe for sql visibility",
        "semantic routing",
        "shape-compatible batches",
        "snapshot or epoch generation",
        "small key-value, and log operations",
        "specialized storage hardware",
        "source wal",
        "shared-region operations",
        "stable database messages",
        "stable ownership",
        "stored procedures",
        "training-domain features",
        "transport choice",
        "transport shape",
        "bounded mechanism",
        "ordering contract required by the receiver",
        "transaction ordering",
        "cyclic buffer dependency",
        "dictionary encoding is not generally order-preserving",
        "correctness proof is cheap",
        "compact domain",
        "compact route-state vectors",
        "transactional progress",
        "cxl memory",
        "cache-coherence behavior",
        "conflict temperature",
        "estimator provenance",
        "physical placement",
        "recovery debt",
        "route-validation layer",
        "typed state",
        "value distributions",
        "accelerated state",
        "accelerator affinity",
        "control metadata bypassing large payload queues",
        "dirty metadata state",
        "metadata has accelerator affinity",
        "mutation-owner time",
        "predicted resource class",
        "receiver-owned credit benchmarks",
        "route-certificate invariant tests",
        "snapshot-handle stress tests",
        "visible generation",
        "write-batch protocols",
        "generation-tagged cache descriptors",
        "explicit completion states",
        "encoding-selection",
        "immutable segment generations",
        "can it be restarted",
        "can it join this batch",
        "cleanup work can lag behind visibility",
        "false-positive precision",
        "simd lookup paths",
        "visibility proof",
        "resident layout identity",
        "bandwidth-class metadata",
        "gpu memory management",
    ],
    "cost_based_route_optimizer": [
        "access shapes",
        "alternative plans",
        "cardinality",
        "cost model",
        "cost estimate",
        "dirty-page metadata",
        "durable latest-state",
        "compact recovery progress",
        "durable internal metadata",
        "co-schedule compatible retained reads and joins",
        "fast route",
        "gpu cache metadata",
        "large-result analytical verification",
        "large-result analytical",
        "layout",
        "layout choices",
        "locally optimal plans",
        "metadata updates",
        "metadata operations",
        "optimizer flag settings",
        "optimizer's scope",
        "parallel multi-plan",
        "predicate/range",
        "predicate/range checking",
        "proof",
        "rank admissible routes",
        "compact profile",
        "control payloads",
        "dirty-state oracle",
        "metadata movement and validation",
        "operator costs",
        "route candidates",
        "route choice",
        "route prototype",
        "route-certificate",
        "route families",
        "runtime design pressure",
        "normal visibility and predicate checks",
        "predicate/value proof",
        "typed service capabilities",
        "refine/materialization cost",
        "read-route memory budget",
        "read side complement",
        "estimated rows and bytes",
        "selected bytes",
        "tier miss classes",
        "volatile access paths",
        "query shape",
        "cardinality estimates",
        "cuda stream ownership",
        "cpu fallback index leaf",
        "dbms-controlled remapping",
        "fine-grained promotion metadata",
        "mvcc visibility stamps",
        "publication layer",
        "resident segment-map traversal",
        "route certificates",
        "local and remote shard caches",
        "training-domain features",
        "compact recovery progress",
        "sql-native predicate/range checking",
        "large-result analytical verification",
        "typed service capabilities",
        "generation and update contracts",
        "predicate/value proof",
        "metadata has accelerator affinity",
        "skew makes the accelerator path",
        "common metadata route",
        "middle and fallback lanes",
        "read/write intent",
        "cheap request metadata",
        "selected subset of columns",
        "distinct counts",
        "histograms",
        "cpu-side publication mechanics",
        "tiered storage authority",
        "tenant-specific predicates",
        "stored-procedure behavior",
        "route-generation descriptor",
        "resident snapshot pointers",
        "resident refresh",
        "short read routes",
        "simple retained reads",
        "gpu, and residency work",
    ],
    "cpu_fallback_policy": [
        "failed operator on cpu",
        "fallback reason",
        "measured thresholds",
        "gpu transfer",
        "escape hatch",
        "materializing full rows",
        "outgoing key filters",
        "cpu fragment set",
        "drain",
        "gpu fragment set",
        "overload",
        "reject",
        "stale cached data",
        "temporary hbm bytes",
        "compensates the partial effects",
        "re-executes the aborted transaction",
        "middle and fallback lanes",
        "visibility certificate",
        "resource certificate",
        "fallback paths",
        "stale operations",
        "descriptor incarnation",
        "reusable descriptors",
    ],
    "db_owned_cold_objects": [
        "critical path",
        "disk leaf page",
        "hot/cold movement",
        "promotion and demotion",
        "query-covering region",
        "request critical path",
        "whole request or transaction critical path",
        "whole request",
        "storage layout",
        "write amplification",
        "object families need measured tier classes",
        "critical path is shortened",
        "uncontrolled compaction work",
        "promotion and demotion do not turn",
        "pages close to the partition",
        "lightweight access tracking",
        "aggressively to compress",
    ],
    "deficit_fairness": [
        "forced rollbacks",
        "level-plus-gradient",
        "multi-tenant deployment",
        "p99.9 latency",
        "response lane",
        "route class",
        "scarce warm resources",
        "unlucky transaction",
        "active conflicts",
        "best-effort long work",
        "default scheduling",
        "high-priority short work",
        "not every record or session",
        "synthetic retained reads",
        "slow responses",
        "gradient-only policies",
        "level-plus-gradient policies",
        "exact accounting",
        "current window and base rtt",
        "reducing burstiness",
        "efficiency from fairness",
    ],
    "effective_session_counting": [
        "caching/storage pages",
        "global log",
        "hot index heads",
        "many sessions",
        "page copying",
        "page-table updates",
        "per-session remote atomics",
        "queue control words",
        "social-network workloads",
    ],
    "deterministic_hot_write_templates": [
        "commit-time",
        "deferred",
        "global queue",
        "handoff queues",
        "hot key or partition",
        "hot key vector",
        "hot working sets",
        "middle trees",
        "mini-page",
        "owner-local metadata graphs",
        "pre-validate",
        "read-write conflicts",
        "tier-aware",
        "top, middle, and lower trees",
        "accelerator memory allocation failure",
        "global hash table",
        "local segmented reductions",
        "pre-aggregation",
        "stored-procedure",
        "commit-validation functions",
        "variable order",
        "variable-length in-memory",
        "hot shared objects",
        "hot shared counters",
        "deferred commit-time deltas",
        "commit coordination",
        "fast route needs a certificate",
        "paused links form a directed cycle",
        "structured buffer pools",
        "write-set as invisible multi-version placeholders",
        "globally ordered versions",
        "postprocessing",
        "control-flow divergence",
        "code-shape choice",
        "semantic pressure",
        "resource pressure",
        "expected gpu work",
        "isolation proof state",
        "cross-route conflict telemetry",
        "conditional-and short-circuiting",
        "bitwise-and evaluation choices",
    ],
    "dependency_witnesses": [
        "allocator state",
        "cold bottom tree",
        "commit-time proof",
        "connection-level ordering",
        "explicit admission credits",
        "hot top tree",
        "persistence ordering",
        "precondition",
        "phase fences",
        "stronger global owner",
        "proof inputs",
        "proof-carrying metadata",
        "resident `int4`/`int8` key column",
        "route certificates",
        "schema access patterns",
        "schedule, cancel, recover",
        "access-method proof",
        "cascade-abort shape",
        "conflict/owner proof",
        "conflicted suffix work",
        "isolation proof",
        "residency proof",
        "speculative write window",
        "write-set and dependency evidence",
        "work-conserving shuffle",
        "stronger global owner",
        "visibility proof",
        "conflict proof",
        "primitive-budget proof",
        "proof state",
        "operation-level dependency evidence",
        "anomaly witnesses",
        "frontier proof",
        "placement proof",
        "flow proof",
        "proof surface",
        "local, bounded pressure signals",
        "escape paths",
        "accelerate only the part",
        "route decision",
        "different control flow",
        "ordered in a chain",
    ],
    "gpu_oltp_conflict_ordering": [
        "cold data blocks",
        "conflict detection",
        "conflict graph",
        "conflict metadata",
        "conflict-robustness",
        "gpu write protocols",
        "gpu-resident execution",
        "hot-key",
        "lowest isolation levels",
        "neworder",
        "priority dimension",
        "transactional durability",
        "transaction processing",
        "adaptation behavior",
        "compatible, high-volume shapes",
        "deterministic resolution",
        "edges follow the original block order",
        "fallback path builds a dag",
        "gpu-side deterministic concurrency-control",
        "operation that always succeeds",
        "write-set",
        "age-aware policy",
        "hot transaction",
        "conflict certificate",
        "affected table/key/range",
        "write-heavy pages",
        "extra copy threads",
        "zero-shot cost models",
        "explicit contracts",
        "conflict-history transaction scheduling",
        "rcbench",
        "conflict shape",
        "ordering shape",
    ],
    "htap_freshness_router": [
        "generation-indexed metadata",
        "refresh, pruning",
        "scan-oriented olap",
        "selective reads",
        "updates accumulate",
        "accelerator budgets",
        "long scans",
        "grouped lookup batch",
        "decompression pass",
        "refresh job",
        "compute-bound",
        "vram-bandwidth-bound",
        "refresh chunk",
        "index probe",
        "scan fragment",
        "read-committed transaction shapes",
        "cold-tier decompression",
        "modern transaction routing",
        "runtime admission",
        "versions they need",
        "write-path structure",
    ],
    "immutable_route_roots": [
        "catalog generation",
        "compact route-cell",
        "conditional fronts",
        "durable/logical fronts",
        "generation-carrying wakeups",
        "pre-execution filter lane",
        "pointer publication",
        "publishing intent",
        "query family",
        "restriction language",
        "route advisor",
        "route certificate",
        "route decision record",
        "schema generation",
        "slot order",
        "typed route descriptor",
        "visibility boundary",
        "budget fields",
        "compact route descriptor",
        "generation tokens",
        "lease/owner state",
        "predicate constants",
        "storage-function ids",
        "data semantics",
        "dirty metadata state",
        "observed freshness lag",
        "overload behavior",
        "predicted resource class",
        "source wal or transaction boundary",
        "cardinality estimates",
        "own generation and update path",
        "route descriptor with budget fields",
        "conflict budget",
        "placement budget",
    ],
    "learned_optimizer_advisor": [
        "co-execution",
        "cpu prefilter",
        "gpu analytical engines",
        "kernel launch parameters",
        "same multiprocessor resource pool",
        "optimizer flag settings",
        "emerging hot pages",
        "false positives are demoted",
        "adaptive demotion",
        "execution model",
        "event delivery",
        "piecewise-linear index",
        "refinement cost",
    ],
    "log_structured_warm_tier": [
        "nvme lane",
        "nvme prefetch budget",
        "pm-backed byte-addressable fast side",
        "publish visibility",
        "pinned host memory",
        "remote memory budget",
        "durable append results",
        "reserved zones",
        "visibility publication",
        "zns append path",
        "byte-compatible placement rule",
        "route-level logical format",
        "base resident snapshot",
        "cpu mvcc truth",
        "gpu resident snapshots",
    ],
    "multi_tier_placement": [
        "accelerator data cache",
        "bitmap indexes",
        "candidate waypoint tiers",
        "cxl memory",
        "cxl-like coherent shared memory",
        "dictionary rebuild",
        "eviction",
        "future-tier snapshots",
        "gpu memory",
        "gpus",
        "hot/cold movement",
        "local dram",
        "memory bandwidth workloads",
        "networking/runtime admission",
        "pinned memory",
        "promotion and demotion",
        "resident",
        "resident column groups",
        "scarce tier",
        "streaming access",
        "target tiers",
        "transfer",
        "update state",
        "dirty bits",
        "fixed-size value slots",
        "in-kernel set-associative cache",
        "replica owns recovery",
        "codec and decode facts",
        "classed admission plane",
        "explicit frontiers",
        "pinned-buffer pools",
        "retained reads",
        "durable frontier",
        "resource budgets",
        "protocol-edge decisions",
        "gpu analytical engines",
        "single lane",
        "bounded reorder buffer",
        "explicit memory admission",
        "conflict policy",
    ],
    "mvcc_gc_frontiers": [
        "active snapshots",
        "bounded-delay snapshot",
        "cleanup horizons",
        "demotion is integrated",
        "explicit frontiers",
        "lifetime/reclamation",
        "no helper can still reach it",
        "physical segment shape",
        "needed old versions",
        "safe cleanup",
        "route preflight",
        "owner movement",
        "resident refresh",
        "gc of old versions",
        "local frontier reads",
        "global frontier reads",
        "datatype callbacks",
        "rollback",
        "resource credits consumed",
        "cleanup obligation",
    ],
    "owner_ring_bundling": [
        "active credits",
        "allocator state",
        "bounded common cases",
        "cache-maintenance",
        "class-aware work",
        "classed admission plane",
        "expected spill page pressure",
        "gpu queue delay",
        "lane",
        "learned optimizer steering",
        "local wait",
        "owner boundaries",
        "owner remains responsible",
        "owner logic",
        "owner set",
        "owner-thread",
        "partition-owned",
        "pinned host spill buffers",
        "p99 retained latency",
        "scarce concurrency-control resources",
        "queue depth",
        "queue-depth-limited",
        "queue wait",
        "refresh starvation",
        "request descriptor",
        "result contract",
        "correctness boundary held",
        "resident cache budget",
        "rich read/write behavior",
        "runtime scoring work",
        "scarce fast-tier memory",
        "opaque queues and byte ranges",
        "ray-tracing scene",
        "scoped owner",
        "scheduling rank",
        "service ownership",
        "storage-latency hiding",
        "temporary gpu scratch budget",
        "tier budget",
        "response release",
        "response size",
        "refresh risk",
        "stable portals",
        "worker",
        "exclusive ownership",
        "host writes",
        "response-ring heads",
        "session-credit counters",
        "credit issuer",
        "hot index heads",
        "queue control words",
        "queue-pressure bucket",
        "tenant/workload bucket",
        "wal reservation delay",
        "stale-generation rejection",
        "crash/cancel safety",
        "queue or tier budget that owns admission",
        "which class of work won",
        "which class paid",
        "correctness boundary",
        "gpu runtime",
        "compute units",
        "range object",
        "ix update postings",
        "key-space partitioning",
        "snapshot_ts",
        "safe window",
        "selected tier",
        "locality routing",
        "filter-friendly encodings",
        "bit-packed dictionary keys",
        "fsst-compressed dictionary values",
        "touched owners",
        "resident segments",
        "point lookup",
        "resident aggregate",
        "background refresh",
    ],
    "resource_dag_scheduling": [
        "communication cost",
        "clear resource consequences",
        "device-local partial results",
        "fallback behavior",
        "logical pause",
        "lifetime",
        "physical segment shape",
        "ready work",
        "release mechanics",
        "request scheduling",
        "resource consequences",
        "route contract",
        "active windows",
        "active windows can be separated",
        "classed route contract",
        "deterministic dag fallback",
        "scheduling intent",
        "work they admit has stable ownership",
        "scarce resources",
        "scheduling evidence",
        "scarce warm resources",
        "current load",
        "complementary resource profile",
        "memory-bound work",
        "freshness boundary",
        "request-to-route mapping",
        "scheduling lane",
        "touched relation/key set",
        "resource class, conflict class, and visibility class",
        "decision-specific history",
        "request should declare",
        "access mode",
        "two requests that name the same object",
        "shared resources",
        "page-loadable resources",
        "resource certificate",
        "gpu stage class",
        "single-task gpu ownership",
        "performance guarantees",
        "fault isolation",
        "thousands of outstanding requests",
    ],
    "isolation_trace_oracle": [
        "single-task gpu ownership",
        "performance guarantees",
        "fault isolation",
        "operation-level dependency evidence",
        "anomaly witnesses",
        "cardinalities",
        "true cardinalities",
    ],
    "semantic_crash_oracle": [
        "storage availability",
        "publish a decision",
        "conservative freezing mode",
        "failed paths",
        "ecmp routing converges",
    ],
    "retained_gpu_snapshots": [
        "allocator epoch",
        "batch order",
        "explicit objects",
        "freshness boundary",
        "gpu resident invalidation",
        "gpu queue delay",
        "physical-worker reachability",
        "pgwire row messages",
        "read snapshot",
        "shared/exclusive state",
        "residency",
        "resident invalidated",
        "resident refreshed",
        "resident generation",
        "resident handles",
        "resident layout",
        "resident validity",
        "retained read",
        "retained lookup",
        "retained resources",
        "resident index",
        "reusable response metadata",
        "snapshot scope",
        "snapshot generation",
        "valid resident segments",
        "movement chunks",
        "refresh installs",
        "version-chain distance",
        "code-shape choice",
        "compressed warm-tier vector scan",
        "resident gpu scan",
        "gpu lookup batch",
        "retained analytical route",
        "replay frontier",
        "base resident snapshot",
        "write admission",
        "merge a base resident snapshot with deltas",
    ],
    "same_shape_microbatching": [
        "aligned reads",
        "bounded chunks",
        "filter-build phase",
        "aggregate or join continuation",
        "kernel launch amortization",
        "large batches",
        "large but draining",
        "medium and large batches",
        "operator work",
        "post-execution conflict detection",
        "prepared statements",
        "sweep batch size",
        "pull/credit-driven",
        "aggregate or join continuation",
        "rt route",
        "micro-batch pinning",
        "gpu resident micro-batch",
    ],
    "snapshot_frontier_vectors": [
        "active snapshots",
        "bounded multiversion",
        "bounded ingress rings",
        "central admission and route scheduler",
        "coherent gpu",
        "explicit frontiers",
        "semantic class",
        "logical versus active counts",
        "mvcc snapshots",
        "owner domains",
        "resident generation",
        "retained snapshot",
        "retained analytical route",
        "replay frontier",
        "snapshot or wal boundary",
        "snapshot compatibility",
        "snapshot retirement",
        "visibility metadata",
        "snapshot/publication boundary",
        "code-shape choice",
        "fallback reason",
        "tier-confidence state",
        "visibility heads",
        "visibility boundary",
    ],
    "stable_handle_indirection": [
        "mutable state transitions",
        "record/version heads",
        "remote memory",
        "resident index",
        "resolve a stable handle",
        "safe kernel-resident",
        "tree nodes",
        "tuple versions",
        "userspace buffers",
        "gpu pointer publication",
        "promotion into local dram",
        "random loads",
        "streaming prefetch",
        "zero-copy",
        "route generation",
        "response-ring identity",
        "session's full state",
    ],
    "wal_before_visibility": [
        "operator family",
        "physical encoding",
        "queue or tier budget",
        "selected durability certificate",
        "cuda pinned memory",
        "durability counters",
        "durable append results",
        "gpu transfers",
        "postgresql protocol paths",
        "read-route memory budgets",
        "retained snapshot column",
        "wal safety",
        "write-heavy mutation metadata",
        "per-origin visibility",
        "session-monotonic fronts",
        "range visibility",
        "logical certificate",
        "physical execution contexts",
        "invalid for new readers",
    ],
    "vector_credit_admission": [
        "admission lane",
        "bounded admission slot",
        "budget",
        "can spill",
        "capacity",
        "correctness metadata",
        "footprint-aware",
        "ingress",
        "edge admission and routing layer",
        "mutation admission",
        "numeric, constrained, commutative data",
        "mvcc validation",
        "pinned-buffer staging",
        "policy-table choices",
        "route-admission p999",
        "flow proof fields",
        "pressure",
        "queueing delay",
        "response identity",
        "resource proof",
        "resource budget",
        "retry backoff",
        "saturated boundary",
        "target core",
        "admission record is compact",
        "frontier that makes the work safe",
        "wait points",
    ],
}


MECHANISM_TERMS: dict[str, list[str]] = {
    "wal_before_visibility": [
        "wal",
        "write-ahead log",
        "logging",
        "durability",
        "durable commit",
        "commit protocol",
        "replay",
        "checkpoint",
        "fsync",
        "visibility",
        "persistent memory",
        "nvm",
        "recovery",
    ],
    "immutable_route_roots": [
        "root",
        "route root",
        "generation",
        "manifest",
        "route descriptor",
        "catalog descriptor",
        "immutable",
        "publish",
        "publication",
        "checksum",
        "old-or-new",
    ],
    "dependency_witnesses": [
        "dependency",
        "witness",
        "fence",
        "ordered",
        "ordering",
        "depends",
        "proof",
        "asynchronous persistence",
        "publication proof",
        "recoverable speculation",
    ],
    "semantic_crash_oracle": [
        "crash",
        "failure",
        "oracle",
        "recovery test",
        "crash consistency",
        "failure state",
        "adversarial",
        "durinn",
        "chipmunk",
        "pathfinder",
    ],
    "isolation_trace_oracle": [
        "isolation",
        "serializability",
        "snapshot isolation",
        "trace",
        "black-box",
        "verifier",
        "polygraph",
        "jepsen",
        "leopard",
        "polysi",
        "cobra",
        "viper",
        "elle",
    ],
    "retained_gpu_snapshots": [
        "retained",
        "resident",
        "gpu-resident",
        "gpu snapshot",
        "snapshot route",
        "read snapshot",
        "htap",
        "vweaver",
        "vdriver",
        "diva",
        "gpu memory",
        "hbm",
    ],
    "snapshot_frontier_vectors": [
        "frontier",
        "epoch",
        "snapshot",
        "multi-version",
        "multiversion",
        "multi-master",
        "generation vector",
        "visibility generation",
        "long reader",
        "begin timestamp",
        "commit timestamp",
    ],
    "mvcc_gc_frontiers": [
        "mvcc garbage",
        "garbage collection",
        "old version",
        "version chain",
        "reclamation",
        "retention",
        "prune",
        "cleanup",
        "bounded memory",
        "long readers",
    ],
    "bounded_descriptor_reclamation": [
        "hazard",
        "epoch reclamation",
        "era",
        "retire",
        "retired",
        "lock-free",
        "wait-free",
        "route descriptor",
        "catalog descriptor",
        "plan descriptor",
        "publish on ping",
        "hyaline",
        "crystalline",
        "nbr",
        "oracgc",
        "orcgc",
    ],
    "stable_handle_indirection": [
        "handle",
        "indirection",
        "pointer",
        "forwarding",
        "relocation",
        "moving",
        "page table",
        "address translation",
        "remote memory",
        "far memory",
        "aifm",
        "ruma",
        "verlib",
    ],
    "vector_credit_admission": [
        "credit",
        "token",
        "admission",
        "flow control",
        "backpressure",
        "queueing",
        "congestion",
        "zero queue",
        "pifo",
        "sp-pifo",
        "justitia",
        "hostcc",
        "tfc",
        "1rma",
    ],
    "effective_session_counting": [
        "session",
        "connection",
        "logical session",
        "effective flow",
        "fan-in",
        "million",
        "multiplex",
        "gateway",
        "network edge",
        "rdma connection",
        "staR",
        "srnic",
    ],
    "owner_ring_bundling": [
        "owner",
        "ring",
        "bundle",
        "queue",
        "reactor",
        "handoff",
        "dispatch",
        "scheduler state",
        "mechanical sympathy",
        "bounded bundle",
        "work order",
    ],
    "resource_dag_scheduling": [
        "dag",
        "scheduler",
        "scheduling",
        "resource",
        "heterogeneous",
        "plan-ahead",
        "cluster scheduler",
        "troublesome",
        "graphene",
        "tetrisched",
        "firmament",
        "decima",
    ],
    "deficit_fairness": [
        "fairness",
        "deficit",
        "priority",
        "tenant",
        "slo",
        "latency guarantee",
        "bounded unfairness",
        "starve",
        "jitter",
    ],
    "same_shape_microbatching": [
        "micro-batch",
        "microbatch",
        "batching",
        "same-shape",
        "prepared",
        "kernel launch",
        "coalesced",
        "grouped lookup",
        "batched read",
        "large batch",
    ],
    "gpu_oltp_conflict_ordering": [
        "gpu oltp",
        "conflict",
        "conflict ordering",
        "transaction batch",
        "ltpg",
        "gacco",
        "large-batch transaction",
        "access set",
        "ycsb",
        "tpc-c",
    ],
    "deterministic_hot_write_templates": [
        "deterministic",
        "hot key",
        "hot write",
        "contention",
        "contended",
        "template",
        "queue position",
        "abort",
        "retry",
        "occ",
        "plor",
        "aria",
        "decent",
    ],
    "cpu_fallback_policy": [
        "fallback",
        "cpu fallback",
        "unsupported",
        "stale",
        "over budget",
        "escape hatch",
        "cpu route",
        "split route",
    ],
    "htap_freshness_router": [
        "freshness",
        "htap",
        "staleness",
        "fresh",
        "router",
        "wait budget",
        "read-committed",
        "analytical read",
        "f1 lightning",
        "vedb",
    ],
    "cost_based_route_optimizer": [
        "optimizer",
        "planning",
        "cost model",
        "cardinality",
        "join",
        "route choice",
        "access path",
        "predicate",
        "pruning",
        "metadata",
        "holon",
        "skinnerdb",
        "quickstep",
    ],
    "learned_optimizer_advisor": [
        "learned",
        "machine learning",
        "advisor",
        "adaptive",
        "model",
        "training",
        "confidence",
        "expert optimizer",
        "bao",
        "leon",
        "lemo",
        "eraser",
        "alece",
    ],
    "multi_tier_placement": [
        "tier",
        "tiering",
        "placement",
        "cache",
        "buffer",
        "hbm",
        "dram",
        "nvme",
        "cxl",
        "far memory",
        "hot page",
        "cold",
        "promotion",
        "demotion",
        "mosaic",
        "colloid",
        "memstrata",
    ],
    "log_structured_warm_tier": [
        "log-structured",
        "append",
        "warm tier",
        "persistent index",
        "nvm",
        "nova",
        "rewind",
        "lsnvmm",
        "falcon",
        "segment log",
        "mapping rebuild",
    ],
    "db_owned_cold_objects": [
        "object",
        "blob",
        "cold tier",
        "filesystem",
        "file system",
        "storage layout",
        "compaction",
        "rocksdb",
        "vortex",
        "skyplane",
        "cloudcast",
        "bytehouse",
        "lance",
        "pravega",
        "geminifs",
    ],
}


CATEGORY_FALLBACKS: list[tuple[str, list[str]]] = [
    ("wal|logging|durability|recovery|persistent|nvm|checkpoint", ["wal_before_visibility", "semantic_crash_oracle"]),
    ("mvcc|snapshot|visibility|isolation", ["snapshot_frontier_vectors", "isolation_trace_oracle"]),
    ("garbage|reclamation|gc", ["mvcc_gc_frontiers", "bounded_descriptor_reclamation"]),
    ("runtime|session|network|admission|hft|concurrency", ["vector_credit_admission", "owner_ring_bundling"]),
    ("gpu|execution|analytics", ["retained_gpu_snapshots", "same_shape_microbatching"]),
    ("optimizer|planning|query", ["cost_based_route_optimizer", "htap_freshness_router"]),
    ("tier|cache|placement|storage|buffer|file", ["multi_tier_placement", "db_owned_cold_objects"]),
    ("transaction|write|oltp|contention", ["deterministic_hot_write_templates", "wal_before_visibility"]),
]


REVIEW_STATUS_BY_CONFIDENCE = {
    "high": "auto_accepted",
    "medium": "auto_accepted",
    "low": "pending_low_confidence_review",
    "needs_review": "manual_review_required",
}

REVIEW_PRIORITY_BY_CONFIDENCE = {
    "high": "none",
    "medium": "none",
    "low": "normal",
    "needs_review": "high",
}


REVIEW_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    (
        "2026-06-07-cross-paper-synthesis-route-admission-now-needs-three-witnesses",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The synthesis explicitly requires valid snapshot/catalog/resident generations and safe descriptor publication.",
    },
    (
        "2026-06-07-cross-paper-synthesis-route-admission-now-needs-three-witnesses",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Plan-ahead admission and route-class resource proof touch route optimization, but the entry is mainly about admission witnesses.",
    },
    (
        "2026-06-07-cross-paper-synthesis-learned-route-control-needs-deterministic-envelopes",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry names validation restore, native fallback, overload rejection, and fallback causes as required route controls.",
    },
    (
        "2026-06-07-cross-paper-synthesis-learned-route-control-needs-deterministic-envelopes",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Learned outputs are constrained to immutable policy generations published for cheap worker evaluation.",
    },
    (
        "2026-06-07-cross-paper-synthesis-tier-placement-needs-price-proof-and-cleanup-horizons",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The retained-route contract includes visibility frontiers, publication/recovery witnesses, and crash-safe derived metadata publication.",
    },
    (
        "2026-06-07-cross-paper-synthesis-tier-placement-needs-price-proof-and-cleanup-horizons",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Publication/recovery witness tables and proof fields are central to the route contract described by the entry.",
    },
    (
        "2026-06-07-cross-paper-synthesis-tier-placement-needs-price-proof-and-cleanup-horizons",
        "mvcc_gc_frontiers",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Active-holder cleanup horizons and bounded retained-generation cleanup are explicit benchmark priorities.",
    },
    (
        "2026-06-07-cross-paper-synthesis-tier-placement-needs-price-proof-and-cleanup-horizons",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry is specifically about retained routes and resident derived state with measured demotion/admission rules.",
    },
    (
        "2026-06-07-cross-paper-synthesis-persistent-metadata-needs-overlay-replay-and-witnesses",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The synthesis proposes a tiered metadata owner spanning DRAM overlays, warm/cold indexes, resident manifests, and future CXL/NVM directories.",
    },
    (
        "2026-06-07-cross-paper-synthesis-persistent-metadata-needs-overlay-replay-and-witnesses",
        "mvcc_gc_frontiers",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Descriptor lifetime is bounded by generation, reference count, or snapshot epoch, with cleanup backlog made visible.",
    },
    (
        "2026-06-07-cross-paper-synthesis-persistent-metadata-needs-overlay-replay-and-witnesses",
        "semantic_crash_oracle",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Crash-state matrices and failure-survival witnesses are named as proof gates for derived metadata.",
    },
    (
        "2026-06-06-cross-paper-synthesis-durable-metadata-needs-recoverable-shape",
        "cost_based_route_optimizer",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The only optimizer signal is a future category-gap suggestion, not evidence for a mechanism link.",
    },
    (
        "2026-06-06-cross-paper-synthesis-durable-publication-needs-small-proofs-with-bounded-fallback",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Retired metadata bytes and descriptor generation are explicit fields in the proposed publication-token harness.",
    },
    (
        "2026-06-06-cross-paper-synthesis-durable-publication-needs-small-proofs-with-bounded-fallback",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fallback state, reader fallback, and rejected half-published states are part of the publication record and benchmark.",
    },
    (
        "2026-06-06-cross-paper-synthesis-durable-publication-needs-small-proofs-with-bounded-fallback",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry frames durability and visibility as small explicit proofs consumed by readers, recovery, and fault injection.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-proof-before-execution-not-cleanup-after-failure",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Quickstep-style query plans, work orders, route fragments, and shared admission vocabulary support route planning.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-proof-before-execution-not-cleanup-after-failure",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Work-order scheduling and route-fragment ownership are relevant, though the entry does not prescribe owner rings directly.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-proof-before-execution-not-cleanup-after-failure",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fallback policy and stale-generation rejection are required before enqueue, GPU launch, or response publication.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-proof-before-execution-not-cleanup-after-failure",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Freshness is not the mechanism under discussion; the entry focuses on route proof and cancellation safety.",
    },
    (
        "2026-06-06-cross-paper-synthesis-cold-tier-movement-needs-route-certificates-not-background-mystery-copies",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The hazard term is incidental; the entry is about route certificates and tier movement, not reclamation.",
    },
    (
        "2026-06-06-cross-paper-synthesis-cold-tier-movement-needs-route-certificates-not-background-mystery-copies",
        "db_owned_cold_objects",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Cold/warm data placement, stripes, waypoints, storage movement, and owner-validated tier paths are the main subject.",
    },
    (
        "2026-06-06-cross-paper-synthesis-cold-tier-movement-needs-route-certificates-not-background-mystery-copies",
        "htap_freshness_router",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Freshness budgets and stale-hint rejection are explicit proof gates for route certificates.",
    },
    (
        "2026-06-06-cross-paper-synthesis-cold-tier-movement-needs-route-certificates-not-background-mystery-copies",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The route certificate includes fallback reasons and rejects routes that cannot prove WAL-before-visibility.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-correctness-needs-external-witnesses-too",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Predicate descriptors and operator-boundary route proof give a weak but useful route-planning signal.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-correctness-needs-external-witnesses-too",
        "multi_tier_placement",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Tier-aware indexes appear only as a category-gap direction, not evidence for this mechanism link.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-semantic-certificates-reusable-descriptors-and-generation",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Retained reads, resident buffer generation, and stale-reader tests directly support retained GPU snapshots.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-semantic-certificates-reusable-descriptors-and-generation",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The freshness term describes descriptor validity, not HTAP freshness routing.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-and-scheduling-need-request-shaped-metrics",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry uses descriptors and generations as route metadata, but does not discuss descriptor lifetime or reclamation.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-and-scheduling-need-request-shaped-metrics",
        "db_owned_cold_objects",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Object hit rate and lower-tier access are benchmark signals, though the entry mainly argues for request-shaped routing metrics.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-and-scheduling-need-request-shaped-metrics",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Residency and GPU execution owners consume the same compact route descriptor to admit, prefetch, split, or reject work.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-and-scheduling-need-request-shaped-metrics",
        "same_shape_microbatching",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry explicitly tests whether fixed micro-batching is sufficient or work-aware batch splitting is required.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-and-scheduling-need-request-shaped-metrics",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot and companion generations are named route-descriptor fields used to make admission and publication decisions.",
    },
    (
        "2026-06-03-cross-paper-synthesis-learned-advice-needs-hard-route-boundaries",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Learned advice is constrained by deterministic route validity and hot transaction fragments may be ordered rather than retried.",
    },
    (
        "2026-06-03-cross-paper-synthesis-learned-advice-needs-hard-route-boundaries",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The route descriptor carries resident components, companion columns, expected transfer, and cold-transfer risk under PCIe pressure.",
    },
    (
        "2026-06-03-cross-paper-synthesis-learned-advice-needs-hard-route-boundaries",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Owner domains and queue budgets are part of the descriptor, but the entry is mostly about learned restriction boundaries.",
    },
    (
        "2026-06-03-cross-paper-synthesis-visibility-contention-and-placement-need-distribution-summaries",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The reclamation signal comes from MVCC version retirement, not descriptor reclamation.",
    },
    (
        "2026-06-03-cross-paper-synthesis-visibility-contention-and-placement-need-distribution-summaries",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Residency owners use access, skew, invalidation, refresh-cost, and HBM-residency summaries to decide retained segments.",
    },
    (
        "2026-06-03-cross-paper-synthesis-visibility-contention-and-placement-need-distribution-summaries",
        "vector_credit_admission",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "GPU execution summaries include queue wait, bytes moved, batch size, scratch buffers, and admitted/deferred choices.",
    },
    (
        "2026-06-03-cross-paper-synthesis-admission-needs-explicit-winners",
        "deficit_fairness",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry makes hot-key conflict priority and tail-latency winners explicit after correctness gates.",
    },
    (
        "2026-06-03-cross-paper-synthesis-control-planes-should-stay-explicit",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Retirement and dependency metadata appear through Rebirth-Retire, but the synthesis mainly addresses explicit control paths.",
    },
    (
        "2026-06-03-cross-paper-synthesis-control-planes-should-stay-explicit",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Generation allocation, publication slots, checksums, and completion rings are explicit control-plane responsibilities.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-devices-require-explicit-service-ownership",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Reusable buffers and generation-tagged cache descriptors need lifetime boundaries, but reclamation is not the entry's main point.",
    },
    (
        "2026-06-03-cross-paper-synthesis-expensive-attempts-need-protected-fronts",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Resident-built work and retained reads are named route classes protected by explicit owner fronts.",
    },
    (
        "2026-06-03-cross-paper-synthesis-expensive-attempts-need-protected-fronts",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The protected-front discussion does not provide evidence for descriptor reclamation.",
    },
    (
        "2026-06-03-cross-paper-synthesis-expensive-attempts-need-protected-fronts",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "The entry classifies work by resource fronts and owner fences, but does not require a full DAG scheduler.",
    },
    (
        "2026-06-03-cross-paper-synthesis-expensive-attempts-need-protected-fronts",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Durable and SQL-visible fronts remain required before client-visible reads and writes.",
    },
    (
        "2026-06-03-cross-paper-synthesis-gpu-writes-need-classed-conflict-lanes",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Route descriptors are central, but the entry does not discuss descriptor reclamation or lifetime.",
    },
    (
        "2026-06-03-cross-paper-synthesis-gpu-writes-need-classed-conflict-lanes",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Known-template writes, deterministic preprocessing, conflict density, and abort cost are the entry's immediate benchmark target.",
    },
    (
        "2026-06-03-cross-paper-synthesis-gpu-writes-need-classed-conflict-lanes",
        "isolation_trace_oracle",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The proposed write-admission lab must prove isolation and WAL publication with event traces.",
    },
    (
        "2026-06-03-cross-paper-synthesis-gpu-writes-need-classed-conflict-lanes",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Owner entries saved, queue pressure, and route-conflict economics are part of the resource half of the route descriptor.",
    },
    (
        "2026-06-03-cross-paper-synthesis-runtime-lanes-need-measurable-preemption-points",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Mutation lanes explicitly preserve WAL-before-visibility and pgwire-visible ordering across runtime lanes.",
    },
    (
        "2026-06-03-cross-paper-synthesis-runtime-lanes-need-measurable-preemption-points",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Each lane names ordering requirements, completion frontiers, owner state, and proof boundaries.",
    },
    (
        "2026-06-03-cross-paper-synthesis-runtime-lanes-need-measurable-preemption-points",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The OCC term is incidental; this synthesis is about runtime lanes and preemption, not deterministic hot-write templates.",
    },
    (
        "2026-06-03-cross-paper-synthesis-runtime-lanes-need-measurable-preemption-points",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Lane contracts carry snapshot frontiers and completion frontier telemetry for retained reads and responses.",
    },
    (
        "2026-06-03-cross-paper-synthesis-logical-scale-needs-active-resource-budgets",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "The entry budgets active resources and I/O lanes, though it does not require DAG-shaped scheduling.",
    },
    (
        "2026-06-03-cross-paper-synthesis-i-o-lanes-need-ownership-contracts",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Descriptors appear as storage-lane identifiers, not as evidence for reclamation mechanics.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-placement-still-needs-paced-fronts",
        "mvcc_gc_frontiers",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry explicitly calls for version cleanup and WAL-safe invalidation under paced hot placement.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-placement-still-needs-paced-fronts",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "HBM hot-key tiers, resident route validity, and paced retained reads are central benchmark targets.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-placement-still-needs-paced-fronts",
        "vector_credit_admission",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Socket/protocol credits, response queue budgets, and named rejection or delay reasons are explicit admission surfaces.",
    },
    (
        "2026-06-03-cross-paper-synthesis-admission-needs-tier-aware-memory-fronts",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "The route envelope names response destination, selected worker, and bounded slots, but not owner rings directly.",
    },
    (
        "2026-06-03-cross-paper-synthesis-admission-needs-tier-aware-memory-fronts",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The benchmark pass condition explicitly preserves WAL/MVCC correctness while tier and route admission change.",
    },
    (
        "2026-06-03-cross-paper-synthesis-admission-needs-tier-aware-memory-fronts",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The proposed route envelope carries fallback permission and reports why requests stayed on CPU, waited, or were rejected.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-needs-costed-generations",
        "db_owned_cold_objects",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Cold-segment overlap and tier movement are tracked as generation costs, though the entry is mainly about generation accounting.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-paths-need-declared-boundaries",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Declared write boundaries must compare global mutation order with owner-local sequence fronts while preserving WAL-before-visibility replay equivalence.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-paths-need-declared-boundaries",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fallback authority, confidence fallback, stale-generation prevention, and overload rejection are first-class route descriptor results.",
    },
    (
        "2026-06-03-cross-paper-synthesis-declared-boundaries-need-schedulable-budgets",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Route choice ranks already-valid CPU/GPU/tier candidates under runtime state and records correctness predicates and outcomes.",
    },
    (
        "2026-06-03-cross-paper-synthesis-declared-boundaries-need-schedulable-budgets",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "HTAP appears only as a future placement category gap, not evidence for freshness routing in this entry.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-data-needs-interval-ownership",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry centers read-only retained generations and resident copies with explicit tier intent and visibility intervals.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-data-needs-interval-ownership",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Active snapshot intervals decide admission, eviction, GC, and refresh safety for old versions and retained generations.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-data-needs-interval-ownership",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The freshness signal is stale-route prevention for interval ownership, not HTAP read routing.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-data-needs-interval-ownership",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Visibility intervals and publication boundaries make retained generations immutable route inputs.",
    },
    (
        "2026-06-03-cross-paper-synthesis-schedulers-need-level-and-slope",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The scheduler chooses among execution, batching, fallback, pacing, or rejection using route pressure fields.",
    },
    (
        "2026-06-03-cross-paper-synthesis-schedulers-need-level-and-slope",
        "wal_before_visibility",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "WAL, checkpoint, and recovery are named as future category gaps rather than evidence for this mechanism.",
    },
    (
        "2026-06-03-cross-paper-synthesis-schedulers-need-level-and-slope",
        "deficit_fairness",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Pacing and rejection under pressure imply admission tradeoffs, but the entry does not specify a fairness algorithm.",
    },
    (
        "2026-06-03-cross-paper-synthesis-schedulers-need-level-and-slope",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "HTAP is mentioned as a future literature gap, not as support for freshness routing.",
    },
    (
        "2026-06-03-cross-paper-synthesis-schedulers-need-level-and-slope",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Retained routes, mutation batches, refresh jobs, and response writes carry explicit runtime pressure fields for scheduling.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-tiers-need-semantic-units",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The matched term is incidental; the entry does not discuss descriptor lifetime or reclamation.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-tiers-need-semantic-units",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Semantic hot-tier admission can choose GPU, CPU, host, or cold fallback without weakening WAL-before-visibility.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-tiers-need-semantic-units",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Contention and retry policy tables touch hot-write templates, though the entry mainly addresses placement units.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-tiers-need-semantic-units",
        "vector_credit_admission",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Runtime pressure fields guide admission, but the entry does not require vectorized credit accounting.",
    },
    (
        "2026-06-03-cross-paper-synthesis-specialized-data-paths-need-declared-shape-contracts",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The route descriptor proof and visibility-preserving segment proof make dependencies explicit before specialized execution.",
    },
    (
        "2026-06-03-cross-paper-synthesis-specialized-data-paths-need-declared-shape-contracts",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Queue or tier budget ownership appears in the descriptor, but the entry is broader than owner-ring bundling.",
    },
    (
        "2026-06-03-cross-paper-synthesis-simple-queues-need-stable-memory-contracts",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Scheduling rank and memory contracts expose resource consequences, but the proposed harness is not DAG-specific.",
    },
    (
        "2026-06-03-cross-paper-synthesis-adaptive-scheduling-must-choose-a-shared-clock",
        "cost_based_route_optimizer",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The optimizer signal comes from future category guidance, not from this entry's adaptive scheduling design track.",
    },
    (
        "2026-06-03-cross-paper-synthesis-adaptive-scheduling-must-choose-a-shared-clock",
        "htap_freshness_router",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Hermes-style freshness windows and snapshot generation are part of the route envelope for HTAP-like replicas.",
    },
    (
        "2026-06-03-cross-paper-synthesis-visibility-metadata-wants-planned-routes",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Route descriptors are discussed as metadata carriers, not as evidence for descriptor reclamation.",
    },
    (
        "2026-06-03-cross-paper-synthesis-frontier-metrics-make-tradeoffs-visible",
        "effective_session_counting",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry explicitly separates logical sessions from runnable requests in the admission plane.",
    },
    (
        "2026-06-03-cross-paper-synthesis-frontier-metrics-make-tradeoffs-visible",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Short retained reads and visible freshness or visibility generation are central to the mixed-frontier benchmark.",
    },
    (
        "2026-06-03-cross-paper-synthesis-frontier-metrics-make-tradeoffs-visible",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Bounded lanes and queue attribution are relevant, though the entry frames them as classed admission rather than owner rings.",
    },
    (
        "2026-06-04-cross-paper-synthesis-warm-state-should-be-bounded-semantic-and-visible",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "WAL/visibility publication is one of the named saturated boundaries the mixed workload must identify.",
    },
    (
        "2026-06-04-cross-paper-synthesis-warm-state-should-be-bounded-semantic-and-visible",
        "vector_credit_admission",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Response credits, fixed-byte pools, queue-depth limits, and backpressure telemetry are explicit warm-state controls.",
    },
    (
        "2026-06-04-cross-paper-synthesis-warm-state-should-be-bounded-semantic-and-visible",
        "deficit_fairness",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Slow clients and competing owner pools imply fairness pressure, but no deficit policy is specified.",
    },
    (
        "2026-06-04-cross-paper-synthesis-warm-state-should-be-bounded-semantic-and-visible",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Warm mini-segments and response chunks must carry visibility-generation tags and publication boundaries.",
    },
    (
        "2026-06-02-first-modern-batch-synthesis",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry compares CPU-only, CPU-prefilter-plus-GPU-tail, and full GPU routes under explicit route descriptors.",
    },
    (
        "2026-06-02-first-modern-batch-synthesis",
        "gpu_oltp_conflict_ordering",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Mutation conflict attribution and dependency-aware repair are explicit benchmark priorities for write fragments.",
    },
    (
        "2026-06-02-third-modern-batch-synthesis",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry discusses stable identity and tier movement, but does not provide descriptor lifetime or reclamation evidence.",
    },
    (
        "2026-06-03-sixth-modern-batch-synthesis",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Repeated route choice across CPU, GPU, and tier placement under resident validity uncertainty is central to the entry.",
    },
    (
        "2026-06-03-owner-local-first-shared-only-when-measured",
        "gpu_oltp_conflict_ordering",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Hot-key scheduling, conflict-aware admission, and route-risk penalties are explicitly tied to write admission.",
    },
    (
        "2026-06-04-cross-paper-synthesis-gpu-routes-need-separate-resource-conflict-and-visibility-classes",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The resource class prices HBM, DRAM, L2, transfer, setup, and placement behavior for accelerated routes.",
    },
    (
        "2026-06-04-cross-paper-synthesis-gpu-routes-need-separate-resource-conflict-and-visibility-classes",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Visibility class explicitly distinguishes immutable snapshots and compact latest-visible summaries from CPU-owned truth.",
    },
    (
        "2026-06-04-cross-paper-synthesis-gpu-routes-need-separate-resource-conflict-and-visibility-classes",
        "mvcc_gc_frontiers",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "MVCC-version retirement appears only as a remaining proof-gate gap, not as evidence for GC-frontier mechanics.",
    },
    (
        "2026-06-04-cross-paper-synthesis-gpu-routes-need-separate-resource-conflict-and-visibility-classes",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Launch parameters and overload semantics imply scheduling concerns, but the entry does not require DAG-shaped resources.",
    },
    (
        "2026-06-04-cross-paper-synthesis-write-batches-snapshots-and-compressed-routes-all-need-explicit-physical-i",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The route descriptor carries visibility boundaries, row/version locations, and published-generation state for retained reads.",
    },
    (
        "2026-06-04-cross-paper-synthesis-write-batches-snapshots-and-compressed-routes-all-need-explicit-physical-i",
        "same_shape_microbatching",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Write batches and vector-order descriptors touch batch shape, though the entry mainly focuses on physical intent.",
    },
    (
        "2026-06-04-cross-paper-synthesis-robust-routes-need-budgeted-temporary-state-not-just-resident-data",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot generation is a first-class field in the budgeted route descriptor before admission.",
    },
    (
        "2026-06-04-cross-paper-synthesis-robust-routes-need-budgeted-temporary-state-not-just-resident-data",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Operator choice and temporary-state budgets are routed before admission rather than left to opaque execution.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-should-be-a-typed-contract-not-an-owner-thread-habit",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Template class, write footprint, owner frontiers, and conflict intent are named route-contract fields.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-should-be-a-typed-contract-not-an-owner-thread-habit",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot, route-hint, and residency generations must be explainable before accepted requests proceed.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-budgets-need-fast-typed-feedback",
        "vector_credit_admission",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Admission consumes lane-specific credits and rejects deterministically when resident, temporary, or response budgets are exhausted.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-budgets-need-fast-typed-feedback",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fallback legality and fallback bypass prevention are explicit fields and pass conditions for the route-budget contract.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-budgets-need-fast-typed-feedback",
        "owner_ring_bundling",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry focuses on typed budgets and feedback timing, not owner-ring structure or ownership bundling.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-budgets-need-fast-typed-feedback",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Resident bytes are budgeted, but retained snapshot semantics are secondary to temporary and response credits.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-budgets-need-fast-typed-feedback",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The pass condition explicitly forbids adaptive route hints from bypassing WAL or visibility fallback.",
    },
    (
        "2026-06-04-cross-paper-synthesis-split-routes-need-semantic-gates-before-learned-correction",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fallback permission is part of semantic route eligibility before adaptive ranking may choose a split route.",
    },
    (
        "2026-06-04-cross-paper-synthesis-split-routes-need-semantic-gates-before-learned-correction",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The descriptor is used for eligibility, but the entry does not discuss descriptor lifetime or reclamation.",
    },
    (
        "2026-06-04-cross-paper-synthesis-split-routes-need-semantic-gates-before-learned-correction",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Deterministic route legality is separated from residual cost correction and fragment-proportion ranking.",
    },
    (
        "2026-06-04-cross-paper-synthesis-split-routes-need-semantic-gates-before-learned-correction",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Write omission and coalescing are allowed only for explicitly non-visible blind-write classes.",
    },
    (
        "2026-06-04-cross-paper-synthesis-split-routes-need-semantic-gates-before-learned-correction",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot visibility and resident layout compatibility are deterministic gates before split routes are accepted.",
    },
    (
        "2026-06-04-cross-paper-synthesis-leases-turn-placement-into-a-validity-interval",
        "owner_ring_bundling",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The route descriptor carries placement leases and queue class, but does not discuss owner-ring bundling.",
    },
    (
        "2026-06-04-cross-paper-synthesis-robust-routes-need-separate-truth-residency-and-scratch-contracts",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Truth, execution placement, resident inputs, and scratch capacity are explicitly separated as route contracts.",
    },
    (
        "2026-06-04-cross-paper-synthesis-robust-routes-need-separate-truth-residency-and-scratch-contracts",
        "same_shape_microbatching",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry cites same-shape GPU batching as one of the contracts that must be admitted explicitly.",
    },
    (
        "2026-06-04-cross-paper-synthesis-robust-routes-need-separate-truth-residency-and-scratch-contracts",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Durable truth and visibility boundaries are explicit prerequisites for resident and temporary execution contracts.",
    },
    (
        "2026-06-04-cross-paper-synthesis-conflict-shape-should-drive-routing-not-just-protocol-choice",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Owner-local serialization, conflict-free lanes, and stable boundary ids are named routing choices.",
    },
    (
        "2026-06-04-cross-paper-synthesis-conflict-shape-should-drive-routing-not-just-protocol-choice",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Typed route planning chooses among retained reads, owner serialization, validation, GPU batches, and CPU fallback.",
    },
    (
        "2026-06-04-rtscan-maps-conjunctive-filters-onto-ray-tracing-cores",
        "same_shape_microbatching",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "RTScan benefits from grouped conjunctive predicate shapes, though the entry is primarily about resident indexing.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-metadata-should-be-cached-ordered-and-explainable",
        "effective_session_counting",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "A metadata-cache microbenchmark must prove route lookup does not scale with logical session count.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-metadata-should-be-cached-ordered-and-explainable",
        "htap_freshness_router",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Freshness mode and cheaper snapshot reads are required fields in the ordered route metadata contract.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-separate-write-point-state-and-resident-index-contracts",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Fence waits and generation boundaries provide dependency signals, but the entry is broader than witness construction.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-separate-write-point-state-and-resident-index-contracts",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Resident-index routes are tied to explicit MVCC generations and truth boundaries before they may serve point lookups.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-separate-write-point-state-and-resident-index-contracts",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Hot point state, colder log-shaped records, and resident GPU indexes are deliberately separated by placement contract.",
    },
    (
        "2026-06-04-synthesis-staged-writes-and-bounded-movement-routes-converge",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Write routes expose transaction template, conflict mode, staged execution, and deterministic fallback priorities.",
    },
    (
        "2026-06-04-cross-paper-synthesis-placement-and-write-safety-both-need-budgeted-route-state",
        "gpu_oltp_conflict_ordering",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Conflict temperature and hot-key write promotion are explicit route-state fields for write safety.",
    },
    (
        "2026-06-04-cross-paper-synthesis-placement-and-write-safety-both-need-budgeted-route-state",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Lease and owner state are sampled by owners, but the entry does not prescribe owner-ring bundling directly.",
    },
    (
        "2026-06-04-cross-paper-synthesis-placement-and-write-safety-both-need-budgeted-route-state",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Budgeted route descriptors are central, but immutable publication roots are not the entry's main mechanism.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-typed-service-and-conflict-contracts",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry discusses route metadata and cold tiers, but not descriptor lifetime or reclamation.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-typed-service-and-conflict-contracts",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot generation, ordering, and publication safety are required fields in the typed route metadata.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-typed-service-and-conflict-contracts",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "The synthesis calls for query planning that combines latency risk with resource budgets, but only as a category gap.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-typed-service-and-conflict-contracts",
        "db_owned_cold_objects",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Cold-tier service capabilities and cold-tier pushdown costs are explicit parts of the route contract.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-separate-execution-and-publication-frontiers",
        "effective_session_counting",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Many logical sessions appear only as a category gap, not as evidence for session-counting mechanics.",
    },
    (
        "2026-06-04-cross-paper-synthesis-resident-indexes-need-route-envelopes-and-rebuild-economics",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The resident route envelope names source snapshot boundaries, invalidation causes, and queries served per generation.",
    },
    (
        "2026-06-04-cross-paper-synthesis-resident-indexes-need-route-envelopes-and-rebuild-economics",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry talks about route-envelope facts and warm-state economics, not descriptor reclamation.",
    },
    (
        "2026-06-04-cross-paper-synthesis-resident-indexes-need-route-envelopes-and-rebuild-economics",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fallback allowance and fallback reason are required telemetry fields for accelerated resident routes.",
    },
    (
        "2026-06-04-cross-paper-synthesis-resident-indexes-need-route-envelopes-and-rebuild-economics",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Hybrid CPU/GPU scheduling appears as a next-paper direction, while the entry mainly covers resident envelopes.",
    },
    (
        "2026-06-04-cross-paper-synthesis-resident-indexes-need-route-envelopes-and-rebuild-economics",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Source snapshot boundaries and immutable resident snapshots are central to the index route envelope.",
    },
    (
        "2026-06-04-cross-paper-synthesis-admission-must-price-contention-memory-and-allocation-before-work-enters-h",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The matched arena/allocation language is about temporary memory, not descriptor retirement.",
    },
    (
        "2026-06-04-cross-paper-synthesis-admission-must-price-contention-memory-and-allocation-before-work-enters-h",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Pre-admission pricing classifies route resources and chooses wait, batch ordering, spill, fallback, or overload.",
    },
    (
        "2026-06-04-cross-paper-synthesis-storage-routes-need-hidden-contention-budgets",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The mixed-route stress harness explicitly combines short retained lookups with long cold scans and HBM budgets.",
    },
    (
        "2026-06-04-cross-paper-synthesis-storage-routes-need-hidden-contention-budgets",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The route-envelope discussion names shared state, but not descriptor lifetime or reclamation.",
    },
    (
        "2026-06-04-cross-paper-synthesis-storage-routes-need-hidden-contention-budgets",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot generation and mutation/WAL publication are explicit correctness dimensions in the route envelope.",
    },
    (
        "2026-06-04-cross-paper-synthesis-storage-routes-need-hidden-contention-budgets",
        "log_structured_warm_tier",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "WAL appends and NVM route budgets are mentioned, but not a log-structured warm-tier mechanism.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-placement-needs-stable-pressure-before-movement",
        "htap_freshness_router",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Mixed OLTP/OLAP freshness and fallback decisions are explicit route-placement concerns.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-placement-needs-stable-pressure-before-movement",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The pressure ledger uses descriptors, but does not discuss descriptor retirement or reclamation.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-placement-needs-stable-pressure-before-movement",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Robust cardinality, deterministic statistics, pruning fragments, and fallback ranking support route optimization.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-semantic-boundaries-before-caches-scale",
        "gpu_oltp_conflict_ordering",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Semantic write conflict boundaries and conflict-splitting benchmarks are core to the hot-route contract.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-semantic-boundaries-before-caches-scale",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Owner lookup and route-cache queue accounting matter, but the synthesis does not directly require owner rings.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-semantic-boundaries-before-caches-scale",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Resident indexes, retained route validity, and visibility boundaries are central to the descriptor.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-needs-workload-shaped-proof-repair-and-latency-gates",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The descriptor attaches isolation proof, dependency scope, and conflict/repair contracts before route admission.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-needs-workload-shaped-proof-repair-and-latency-gates",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Certified route templates and unsafe-template fallback are explicit benchmark priorities.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-live-control-loops",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Route certificates name kernel DAGs, resource shapes, measured co-run compatibility, and scheduling gates.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-live-control-loops",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot generation, delta overlays, tombstones, and requested read boundaries are explicit certificate fields.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-live-control-loops",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Static cost estimates are checked against live route-class telemetry before route choice is trusted.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-live-control-loops",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Certified snapshot generations are required before measured fast routes can be selected.",
    },
    (
        "2026-06-04-cross-paper-synthesis-publication-certificates-need-local-staging-and-explicit-durability-clocks",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Owner-local pending arrays and local staging are named write-path publication surfaces.",
    },
    (
        "2026-06-04-cross-paper-synthesis-publication-certificates-need-local-staging-and-explicit-durability-clocks",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The write path separates local staging, WAL durability, MVCC visibility, and GPU resident-snapshot refresh frontiers.",
    },
    (
        "2026-06-04-cross-paper-synthesis-publication-certificates-need-local-staging-and-explicit-durability-clocks",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "GPU freshness is a publication frontier here, not evidence for HTAP freshness routing.",
    },
    (
        "2026-06-04-cross-paper-synthesis-publication-certificates-need-local-staging-and-explicit-durability-clocks",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Named certificate and generation counters define which publication boundary is safe.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-should-combine-freshness-estimates-and-measured-resourc",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot generation, dirty frontier coverage, freshness requirement, and stale-snapshot rejection are core certificate fields.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-should-combine-freshness-estimates-and-measured-resourc",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fallback and stale-snapshot rejection must be explainable from recorded route certificate fields.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-is-now-a-route-certificate-dimension",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry explicitly compares CPU fallback, GPU stable-only execution, and GPU-plus-delta merge under refresh lag.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-is-now-a-route-certificate-dimension",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Published resident generations must belong to clean durable prefixes or be rejected before serving reads.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-is-now-a-route-certificate-dimension",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Retained GPU reads and resident publication boundaries are the main freshness-certificate consumers.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-scheduling-intent",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Bounded active windows carry route templates, key classes, predicted conflicts, and legal scheduling order.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-scheduling-intent",
        "htap_freshness_router",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Freshness clocks are certificate inputs, though the entry mainly targets scheduling intent rather than HTAP routing.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-scheduling-intent",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Isolation certificates, resident/local boundaries, and optional scheduling order are explicit proof fields.",
    },
    (
        "2026-06-04-cross-paper-synthesis-admission-needs-active-window-state",
        "multi_tier_placement",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Tier placement appears only as a future category gap, not as evidence for the active-window admission model.",
    },
    (
        "2026-06-04-cross-paper-synthesis-admission-needs-active-window-state",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Hot-work active-window state includes key or segment hash, priority class, queue position, and defer/fallback decisions.",
    },
    (
        "2026-06-04-cross-paper-synthesis-admission-needs-active-window-state",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "The slice is about bounded admission and scheduling state, not a full DAG scheduler.",
    },
    (
        "2026-06-04-cross-paper-synthesis-admission-needs-active-window-state",
        "vector_credit_admission",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Owner-local admission records queue position, service estimate, priority, fallback counters, and bounded deferment.",
    },
    (
        "2026-06-04-cross-paper-synthesis-publish-certified-generations-not-mutable-shortcuts",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Certified generations include visibility, resident, invalidation, and freshness boundaries for retained reads.",
    },
    (
        "2026-06-04-cross-paper-synthesis-publish-certified-generations-not-mutable-shortcuts",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Out-of-distribution route models and unsuitable generations must fall back conservatively.",
    },
    (
        "2026-06-04-cross-paper-synthesis-publish-certified-generations-not-mutable-shortcuts",
        "htap_freshness_router",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Older retained snapshots may bypass owners only when their certified freshness and isolation contract is sufficient.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-tier-schedule-and-estimate-provenance",
        "vector_credit_admission",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Queue wait, accepted/rejected decisions, and admission facts are tracked, but vector credits are not the main subject.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-tier-schedule-and-estimate-provenance",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The certificate records estimator confidence, OOD reason, accepted/rejected state, and fallback choice.",
    },
    (
        "2026-06-04-cross-paper-synthesis-active-window-certificates-should-choose-the-write-lane",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fast lanes must emit snapshot generation, WAL/invalidation boundary, and visibility-bound certificate telemetry.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-tier-schedule-and-codec-facts",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "GPU resident segments and certified decoded generations are first-class route certificate targets.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-tier-schedule-and-codec-facts",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fallback reason is an explicit field when codec, freshness, decode, or ownership facts do not support the route.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-tier-schedule-and-codec-facts",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Schedule, dependency, pushed-value handles, and ownership facts are part of the certificate proof.",
    },
    (
        "2026-06-04-cross-paper-synthesis-future-tiers-need-local-hot-remote-cold-contracts",
        "effective_session_counting",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Active memory leases require idle logical sessions to reserve no tier payload memory.",
    },
    (
        "2026-06-04-cross-paper-synthesis-future-tiers-need-local-hot-remote-cold-contracts",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Hot control state and scheduling lanes are local-owner concerns, though owner rings are not specified directly.",
    },
    (
        "2026-06-04-cross-paper-synthesis-future-tiers-need-local-hot-remote-cold-contracts",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Route certificates must declare fallback behavior when freshness, access mode, or tier placement is unsuitable.",
    },
    (
        "2026-06-04-cross-paper-synthesis-future-tiers-need-local-hot-remote-cold-contracts",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Batch write scheduler simulation is a relevant follow-up, but the entry's primary focus is tier-locality contracts.",
    },
    (
        "2026-06-05-cross-paper-synthesis-gpu-htap-routes-need-freshness-frontiers-plus-resource-contracts",
        "effective_session_counting",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Request assignment and completion capacity are relevant, but the synthesis focuses on route tokens rather than explicit logical-session counting.",
    },
    (
        "2026-06-05-cross-paper-synthesis-future-routes-need-calibrated-movement-windows",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Movement certificates include visibility generation, owner/placement targets, publication boundaries, and fallback behavior.",
    },
    (
        "2026-06-05-cross-paper-synthesis-active-windows-need-dependency-evidence",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Active-window certificates carry route shape, resource estimates, cardinality bounds, and admitted route estimates.",
    },
    (
        "2026-06-05-cross-paper-synthesis-active-windows-need-dependency-evidence",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The proof gate requires replay-equivalent WAL, deterministic visibility, and publication-order evidence.",
    },
    (
        "2026-06-05-cross-paper-synthesis-active-windows-need-dependency-evidence",
        "deficit_fairness",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The p99 and bounded-window discussion is about correctness and admission, not tenant fairness or deficit scheduling.",
    },
    (
        "2026-06-05-cross-paper-synthesis-active-windows-need-dependency-evidence",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot and publication boundaries are explicit certificate fields for admitted active windows.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-metadata-needs-read-mostly-validation-cells",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Owner-domain ordering is named as future semantic work, while the main evidence is read-mostly route metadata validation.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-metadata-needs-read-mostly-validation-cells",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Versioned publication cells and generation validation directly support snapshot/frontier route checks.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-estimate-freshness-and-tier-freshness",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry tracks freshness frontiers and telemetry generations, not descriptor lifetime or reclamation.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-estimate-freshness-and-tier-freshness",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Stale visibility, placement, or estimator frontiers cause fallback, refresh, exact probe, or explicit overload reasons.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-isolation-evidence-not-just-performance-evidence",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Hot write-conflict routes require explicit batch order, staged retry policy, and prefix publication.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-isolation-evidence-not-just-performance-evidence",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Route certificates include workload shape, data layout, route shape, touched column families, and planning evidence.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-isolation-evidence-not-just-performance-evidence",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Schema generation, transaction frontier, snapshot boundary, and compact published ids are certificate fields.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-isolation-evidence-not-just-performance-evidence",
        "learned_optimizer_advisor",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Learned physical order is treated as a route certificate that can be correct but stale as a performance route.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-influence-control",
        "isolation_trace_oracle",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Elle, IsoDiff, generated histories, and isolation evidence are explicit proof gates for route safety.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-influence-control",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The certificate records the owner lane and cross-lane ordering state that made a route acceptable.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-influence-control",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Mixed deterministic/OCC write lanes require explicit ordering and replay evidence before fast-route publication.",
    },
    (
        "2026-06-05-hpcc-uses-precise-in-flight-telemetry-instead-of-queue-depth-guessing",
        "deficit_fairness",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "HPCC targets low latency and stable headroom, but it is not primarily a fairness or deficit-scheduling mechanism.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-needs-pressure-shaped-contracts-across-rings-io-and-hot-keys",
        "effective_session_counting",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The benchmark priority explicitly holds many logical sessions idle while active work consumes bounded route resources.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-semantic-conflict-shape-not-only-resource-shape",
        "gpu_oltp_conflict_ordering",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Conflict policy, validation work, reconciliation, and identical committed results are central proof gates.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-semantic-conflict-shape-not-only-resource-shape",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Durable frontier, deterministic replay, and no exposure of non-public batch versions are required.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-semantic-conflict-shape-not-only-resource-shape",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The active-window harness includes fixed-capacity hot conflict policy and deterministic repair/fallback behavior.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-semantic-conflict-shape-not-only-resource-shape",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Storage-tier sources are noted as route-certificate inputs, but semantic conflict shape is the entry's main focus.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-semantic-conflict-shape-not-only-resource-shape",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Fixed-capacity completion cells relate to owner-local work control, but owner rings are not specified directly.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-now-need-tier-merge-and-snapshot-contracts",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Semantic mergeability depends on predicate/value proof and explicit cross-tier contract fields.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-now-need-tier-merge-and-snapshot-contracts",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Cross-tier snapshot mapping and generation registries are explicit certificate fields.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-now-need-tier-merge-and-snapshot-contracts",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry discusses mapped generations and tier contracts, not descriptor retirement or reclamation.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-now-need-tier-merge-and-snapshot-contracts",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Placement policy and route-certificate switches influence route choice, though the entry is not primarily about optimization.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-now-need-tier-merge-and-snapshot-contracts",
        "gpu_oltp_conflict_ordering",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Semantic conflict proof and identical SQL-visible results are explicit simulator switches and proof gates.",
    },
    (
        "2026-06-05-cross-paper-synthesis-accelerator-metadata-needs-affinity-publication-and-skew-gates",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Accelerator metadata requires publication safety for WAL, snapshot, resident generation, and route eligibility.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-fallback-lanes-and-gpu-route-contracts",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "HTAP appears only as a category gap; the entry focuses on generic frontiers, fallback lanes, and GPU execution shape.",
    },
    (
        "2026-06-05-cross-paper-synthesis-tail-contracts-need-age-fan-out-and-accelerator-budget",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry discusses route pressure, chunking, and conflict age, not descriptor lifetime or reclamation.",
    },
    (
        "2026-06-05-cross-paper-synthesis-tail-contracts-need-age-fan-out-and-accelerator-budget",
        "deficit_fairness",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Retry age, priority contracts, and bounded entry for short retained reads directly support fairness-aware admission.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-metadata-must-prove-both-correctness-and-pressure-shape",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The pressure proof explicitly includes fallback lane and timeout conditions as route-admission fields.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-schedulers-need-class-proof-and-completion-locality",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Integrated HTAP freshness is listed as a category gap, not as evidence for the scheduler contract in this entry.",
    },
    (
        "2026-06-05-cross-paper-synthesis-htap-freshness-and-modular-transaction-lanes-are-converging",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Stable load-unit metadata supports placement, but the entry does not discuss descriptor retirement or reclamation.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-must-preflight-both-ownership-and-tiers",
        "vector_credit_admission",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Tier resource readiness, pinned resources, and admission before owner lock hold time directly support credit-style preflight.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-must-preflight-both-ownership-and-tiers",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The route certificate has explicit ownership/order proof and compact cross-owner ordering records.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-must-preflight-both-ownership-and-tiers",
        "mvcc_gc_frontiers",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Old-version GC appears as a stale-certificate benchmark, but the entry mainly centers ownership and tier preflight.",
    },
    (
        "2026-06-05-cross-paper-synthesis-staged-route-boundaries-should-be-budgeted-first-class-state",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Freshness-bounded main plus delta generations and staged GPU inputs are central to the retained-route contract.",
    },
    (
        "2026-06-05-cross-paper-synthesis-staged-route-boundaries-should-be-budgeted-first-class-state",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "The entry budgets execution-stage state, but does not specify owner-ring handoff as the primary mechanism.",
    },
    (
        "2026-06-05-cross-paper-synthesis-staged-route-boundaries-should-be-budgeted-first-class-state",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Page handles and pin/unpin contracts imply bounded staged-state lifetimes, though reclamation is not the main topic.",
    },
    (
        "2026-06-05-cross-paper-synthesis-staged-route-boundaries-should-be-budgeted-first-class-state",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Best-CPU-staged baselines and visible temporary route budgets are explicit inputs to route choice.",
    },
    (
        "2026-06-05-cross-paper-synthesis-commit-decisions-need-a-recoverable-visibility-contract",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Durable decisions, resident generations, and cleanup-lag boundaries form a recoverable route-publication artifact.",
    },
    (
        "2026-06-05-cross-paper-synthesis-freshness-windows-need-compact-proof-indexes",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The resident proof index provides segment statistics, refinement width, and fallback inputs for CPU/GPU route choice.",
    },
    (
        "2026-06-05-cross-paper-synthesis-freshness-windows-need-compact-proof-indexes",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Freshness boundaries, delete/delta summaries, and expected refinement width act as compact proof fields for route safety.",
    },
    (
        "2026-06-05-cross-paper-synthesis-freshness-windows-need-compact-proof-indexes",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The proof-index layer is attached to immutable resident generations with explicit freshness boundaries.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-boundaries-need-primitive-budgets",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Primitive-budget accounting and publication cells are discussed, but descriptor lifetime and reclamation are not.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-boundaries-need-primitive-budgets",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Conflict shape, hot queues, and conflict lanes are explicit proof fields before hot-write admission.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-boundaries-need-primitive-budgets",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "CAS-sized route-publication cells carry generation and validity state for route boundaries.",
    },
    (
        "2026-06-05-cross-paper-synthesis-fallback-tiering-and-recovery-all-need-route-frontiers",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Retained reads, serviceable frontiers, and rebuildable derived tiers are explicit route-certificate fields.",
    },
    (
        "2026-06-05-cross-paper-synthesis-fallback-tiering-and-recovery-all-need-route-frontiers",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Code-shape choice, route selector rejection, CPU calibration, and fallback reasons directly support costed route choice.",
    },
    (
        "2026-06-05-cross-paper-synthesis-serviceable-snapshots-beat-invisible-acceleration",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Resident generation, safe window, recovery frontier, and route-certificate trace fields define immutable publication inputs.",
    },
    (
        "2026-06-05-cross-paper-synthesis-serviceable-snapshots-beat-invisible-acceleration",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The route selector chooses among GPU resident, warm CPU columnar, CPU tuple/index, restore-on-demand, or rejection tiers.",
    },
    (
        "2026-06-05-cross-paper-synthesis-serviceable-snapshots-also-need-locality-proof",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The locality proof names where data lives, execution locality, resident segments, access ranges, and fallback tiers.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-need-semantic-proof-surfaces",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The benchmark priorities explicitly require retire/debt telemetry for lazy shortcuts, membership maps, and resident generations.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-need-semantic-proof-surfaces",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "HTAP appears only as a future category gap; the entry's evidence is semantic proof surfaces rather than freshness routing.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "same_shape_microbatching",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry discusses route execution shape and semantic conflict proof, not batching or same-shape microbatch admission.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Resident bytes and policy state are route-certificate fields here, but descriptor lifetime and reclamation are not discussed.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "learned_optimizer_advisor",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "NeurCC and policy-table versions provide a weak learned-policy signal, although the entry is broader route-proof synthesis.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The semantic fast-path proof must preserve SQL-visible histories under WAL, snapshot retention, cancellation, and replay.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Route certificates include resource and execution shape, giving a weak planning signal rather than a detailed optimizer mechanism.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot retention and route certificate frontier fields are part of the required semantic fast-path proof.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Resident bytes are route-certificate evidence, but placement is not the main mechanism of this synthesis entry.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-must-budget-fan-out-tiers-and-accelerator-interference",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry budgets fan-out, tiers, and accelerator interference, but does not address descriptor reclamation.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-must-budget-fan-out-tiers-and-accelerator-interference",
        "htap_freshness_router",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Retained-read freshness is contextual here; the mechanism is admission budgeting, not HTAP freshness routing.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-must-budget-fan-out-tiers-and-accelerator-interference",
        "vector_credit_admission",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The synthesis explicitly requires admission to budget fan-out, tier bytes, and GPU compute/memory-bandwidth shape before work enters.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-must-budget-fan-out-tiers-and-accelerator-interference",
        "log_structured_warm_tier",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "NVM is cited for tier placement discipline, not for a log-structured warm-tier mechanism.",
    },
    (
        "2026-06-05-cross-paper-synthesis-publication-proof-needs-placement-proof",
        "retained_gpu_snapshots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The GPU retained route must name source WAL boundary, snapshot mode, generation lineage, owner lease, and resident byte families.",
    },
    (
        "2026-06-05-cross-paper-synthesis-publication-proof-needs-placement-proof",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "The entry couples publication and placement proof, but it does not discuss descriptor lifetime or reclamation.",
    },
    (
        "2026-06-05-cross-paper-synthesis-publication-proof-needs-metadata-authority",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Typed metadata records carry cleanup responsibility for long retained reads and route refresh pressure, but reclamation is secondary.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-choice-needs-staged-proof",
        "dependency_witnesses",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Staged proof and completion feedback act as witness surfaces, but the entry is not primarily about dependency ordering.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-choice-needs-staged-proof",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "The route stages include visibility checks and correctness metadata, though WAL publication is not the entry's main focus.",
    },
    (
        "2026-06-05-cross-paper-synthesis-staged-acceleration-needs-precise-fallback",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Each route stage carries generation, stale-generation, and fallback reason codes before publication or retry.",
    },
    (
        "2026-06-05-cross-paper-synthesis-staged-acceleration-needs-precise-fallback",
        "db_owned_cold_objects",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The synthesis explicitly includes NVMe/cold-tier stages, storage pushdown, ambiguous-row fallback, and cold-tier placement.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-private-formats-receiver-credits-and-execution-shape-proo",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Receiver credits, owner-granted buffer/stream capacity, response rings, and explicit queue edges directly support owner-ring routing.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-private-formats-receiver-credits-and-execution-shape-proo",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "removed_low_confidence_noise",
        "review_priority": "none",
        "review_note": "Base/delta private-format proof is relevant to route state, but descriptor reclamation is not discussed.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-private-formats-receiver-credits-and-execution-shape-proo",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry ties optimizer estimates, execution-shape choice, selectivity feedback, and runtime pressure into route decisions.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-private-formats-receiver-credits-and-execution-shape-proo",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Private formats, data movement, resident snapshots, and placement-sensitive execution shape are part of the route certificate.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-proof-now-spans-publication-reclamation-and-storage-placement",
        "effective_session_counting",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The synthesis contrasts stalled logical sessions with physical route lifetime proof and restartable/reacquirable route descriptors.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-publication-needs-explicit-fences",
        "multi_tier_placement",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Resident buffer generations and placement gaps touch tier placement, but the synthesis mainly concerns publication fences.",
    },
    (
        "2026-06-06-cross-paper-synthesis-remote-routes-need-tiny-authorities-and-external-witnesses",
        "isolation_trace_oracle",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The Viper-inspired benchmark requires begin, commit, read-from, route generation, and invalidation facts for isolation checking.",
    },
    (
        "2026-06-06-cross-paper-synthesis-hot-paths-need-compact-authorities-and-schedulable-residuals",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The entry explicitly calls for local heavy-state reclamation and descriptor safety checks for delayed grants and reused buffers.",
    },
    (
        "2026-06-06-cross-paper-synthesis-hot-paths-need-compact-authorities-and-schedulable-residuals",
        "gpu_oltp_conflict_ordering",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Contention-aware transaction scheduling, hot/cold decomposition, and TPC-C-style skew are central benchmark targets.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-publication-needs-prediction-plus-fallback",
        "deterministic_hot_write_templates",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Template robustness, predicted batch execution, and deterministic conflict prediction directly support certified hot route templates.",
    },
    (
        "2026-06-06-cross-paper-synthesis-resource-credits-should-travel-with-route-work",
        "deficit_fairness",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Session and tenant budgets, protected scheduled work, disposable speculation, and SLO-oriented overload signals support fairness policy.",
    },
    (
        "2026-06-06-cross-paper-synthesis-resource-credits-should-travel-with-route-work",
        "resource_dag_scheduling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Route work carries explicit resource proofs so the runtime can schedule, batch, reject, redirect, and cancel speculative DAG-like work.",
    },
    (
        "2026-06-06-cross-paper-synthesis-resource-credits-should-travel-with-route-work",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Snapshot-generation eligibility, retry generations, revocation generations, and stale-completion rejection support immutable route roots.",
    },
    (
        "2026-06-06-cross-paper-synthesis-maintenance-needs-credits-generations-and-preemption",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Route work descriptors and schedulable units touch owner-domain execution, but owner-ring bundling is not the primary focus.",
    },
    (
        "2026-06-06-cross-paper-synthesis-maintenance-needs-credits-generations-and-preemption",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Maintenance chunks carry source WAL boundaries and need crash-safe generation publication before visible effects.",
    },
    (
        "2026-06-06-cross-paper-synthesis-reusable-work-needs-visible-lifetime-contracts",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The route asset registry is explicitly optimizer-visible and exposes reusable assets for plan and route selection.",
    },
    (
        "2026-06-06-cross-paper-synthesis-reusable-work-needs-visible-lifetime-contracts",
        "learned_optimizer_advisor",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Lemo-style reusable intermediate state and learned plan selection are core inputs to the visible lifetime contract.",
    },
    (
        "2026-06-07-cross-paper-synthesis-route-hints-need-measured-trust",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Hints must carry fallback paths, stale-route rejection, retry budgets, and degradation telemetry before trust is spent.",
    },
    (
        "2026-06-07-cross-paper-synthesis-tier-movement-needs-semantic-guards",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Retired descriptor memory, bounded worker-tied protection, protected-object counts, and unreclaimed bytes are central guards.",
    },
    (
        "2026-06-07-cross-paper-synthesis-tier-movement-needs-semantic-guards",
        "effective_session_counting",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The synthesis explicitly prefers bounded protection tied to active workers over logical session count.",
    },
    (
        "2026-06-07-cross-paper-synthesis-tier-movement-needs-semantic-guards",
        "snapshot_frontier_vectors",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Tier movement is eligible only when visibility boundaries and retained-read snapshot correctness agree with placement state.",
    },
    (
        "2026-06-07-cross-paper-synthesis-tier-movement-needs-semantic-guards",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Durability and visibility boundaries are separate records that must agree before resident or warm objects are visible.",
    },
    (
        "2026-06-07-cross-paper-synthesis-correctness-needs-external-witnesses-plus-failure-states",
        "cost_based_route_optimizer",
    ): {
        "review_status": "reviewed_weak_signal",
        "review_priority": "none",
        "review_note": "Learned and adaptive route ranking require deterministic eligibility gates, but detailed cost optimization is secondary.",
    },
    (
        "2026-06-07-cross-paper-synthesis-correctness-needs-external-witnesses-plus-failure-states",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Route evidence includes resident/storage generations, deterministic eligibility, manifest boundaries, and quarantine states.",
    },
    (
        "2026-06-07-cross-paper-synthesis-recovery-needs-compact-witnesses",
        "wal_before_visibility",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The fault matrix explicitly tests WAL and manifest crash recovery before promoted routes can be considered safe.",
    },
    (
        "2026-06-07-cross-paper-synthesis-recovery-needs-compact-witnesses",
        "immutable_route_roots",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Compact generated witnesses and post-condition checks protect owner-boundary route facts and recovered publication state.",
    },
    (
        "2026-06-07-cross-paper-synthesis-recovery-needs-compact-witnesses",
        "owner_ring_bundling",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "The design track names compact witness generation for each owner boundary, including resource-credit and recovery witnesses.",
    },
    (
        "2026-06-07-cross-paper-synthesis-freshness-needs-explicit-search-metadata",
        "bounded_descriptor_reclamation",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Independent witness lifetimes include objects that can retire, refresh, or demote separately from wide payload columns.",
    },
    (
        "2026-06-07-cross-paper-synthesis-freshness-needs-explicit-search-metadata",
        "cpu_fallback_policy",
    ): {
        "review_status": "reviewed_supported",
        "review_priority": "none",
        "review_note": "Fallback reasons are first-class route witnesses when pressure, freshness lag, or visibility-map absence rejects a route.",
    },
}


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:96] or "entry"


def split_entries(journal: str) -> list[dict]:
    matches = list(HEADING_RE.finditer(journal))
    entries: list[dict] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(journal)
        body = journal[start:end].strip()
        title = match.group("title").strip()
        date = match.group("date")
        category = extract_match(CATEGORY_RE, body)
        tags = split_tags(extract_match(TAGS_RE, body))
        citation = extract_match(CITATION_RE, body)
        entries.append(
            {
                "id": f"{date}-{slugify(title)}",
                "date": date,
                "title": title,
                "entry_type": "synthesis" if "cross-paper synthesis" in title.lower() else "paper",
                "category": category,
                "relevance_tags": tags,
                "citation": citation,
                "body": body,
            }
        )
    return entries


def extract_match(pattern: re.Pattern, body: str) -> str:
    match = pattern.search(body)
    if not match:
        return ""
    value = match.groupdict().get("value") or match.groupdict().get("plain") or ""
    return value.strip()


def split_tags(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def score_terms(text: str, terms: list[str]) -> tuple[int, list[str]]:
    score = 0
    matched: list[str] = []
    for term in terms:
        term_l = term.lower()
        count = text.count(term_l)
        if not count:
            continue
        weight = 3 if " " in term_l or "-" in term_l else 1
        score += count * weight
        matched.append(term)
    return score, matched[:8]


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def non_metadata_sentences(paragraphs: list[str]) -> list[str]:
    sentences: list[str] = []
    for paragraph in paragraphs:
        if paragraph.lower().startswith(METADATA_SNIPPET_PREFIXES):
            continue
        for sentence in SENTENCE_RE.split(paragraph):
            sentence = compact_whitespace(sentence)
            if sentence:
                sentences.append(sentence)
    return sentences


def expand_short_evidence_sentence(paragraph: str, sentence_index: int, minimum: int = 80) -> str:
    sentences = [compact_whitespace(sentence) for sentence in SENTENCE_RE.split(paragraph)]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences or sentence_index >= len(sentences):
        return ""

    snippet = sentences[sentence_index]
    if len(snippet) >= minimum:
        return snippet

    if sentence_index + 1 < len(sentences):
        expanded = f"{snippet} {sentences[sentence_index + 1]}"
        if len(expanded) >= minimum:
            return expanded

    if sentence_index > 0:
        expanded = f"{sentences[sentence_index - 1]} {snippet}"
        if len(expanded) >= minimum:
            return expanded

    if sentence_index + 1 < len(sentences):
        return f"{snippet} {sentences[sentence_index + 1]}"
    if sentence_index > 0:
        return f"{sentences[sentence_index - 1]} {snippet}"
    return snippet


def truncate_evidence_snippet(snippet: str, terms: list[str], limit: int = 360) -> str:
    if len(snippet) <= limit:
        return snippet

    term_lowers = [term.lower() for term in terms if term != "category fallback"]
    snippet_lower = snippet.lower()
    term_positions = [
        snippet_lower.find(term)
        for term in term_lowers
        if snippet_lower.find(term) >= 0
    ]
    if not term_positions or min(term_positions) < limit - 3:
        return snippet[: limit - 3].rstrip() + "..."

    first_term = min(term_positions)
    context_start = max(0, first_term - 120)
    if context_start > 0:
        boundary = snippet.rfind(" ", 0, context_start)
        if boundary >= 0:
            context_start = boundary + 1
    truncated = snippet[context_start : context_start + limit - 3].rstrip()
    if context_start > 0:
        truncated = "..." + truncated
    if context_start + len(truncated.lstrip(".")) < len(snippet):
        truncated = truncated[: limit - 3].rstrip() + "..."
    return truncated


def evidence_snippet(body: str, terms: list[str]) -> str:
    paragraphs = [compact_whitespace(part) for part in body.split("\n\n")]
    paragraphs = [part for part in paragraphs if part]
    term_lowers = [term.lower() for term in terms if term != "category fallback"]

    best = ""
    best_score = -1
    best_paragraph = ""
    best_sentence_index = -1
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        metadata_penalty = 6 if paragraph_lower.startswith(METADATA_SNIPPET_PREFIXES) else 0
        section_bonus = 2 if paragraph_lower.startswith(STRONG_SNIPPET_PREFIXES) else 0
        for sentence_index, sentence in enumerate(SENTENCE_RE.split(paragraph)):
            sentence = compact_whitespace(sentence)
            if not sentence:
                continue
            haystack = sentence.lower()
            term_hits = sum(1 for term in term_lowers if term in haystack)
            if term_lowers and term_hits == 0:
                continue
            length_bonus = 1 if len(sentence) >= 80 else 0
            score = (term_hits * 4) + section_bonus + length_bonus - metadata_penalty
            if score > best_score:
                best = sentence
                best_score = score
                best_paragraph = paragraph
                best_sentence_index = sentence_index

    if (not best or best.lower().startswith(METADATA_SNIPPET_PREFIXES)) and paragraphs:
        fallback_sentences = non_metadata_sentences(paragraphs)
        strong_fallbacks = [
            sentence
            for sentence in fallback_sentences
            if sentence.lower().startswith(STRONG_SNIPPET_PREFIXES) or len(sentence) >= 80
        ]
        if strong_fallbacks:
            best = strong_fallbacks[0]
        elif fallback_sentences:
            best = fallback_sentences[0]
        elif not best:
            best = paragraphs[0]
    elif best and len(best) < 80:
        best = expand_short_evidence_sentence(best_paragraph, best_sentence_index) or best
    return truncate_evidence_snippet(best, terms)


def evidence_quality_details(link: dict) -> tuple[str, str]:
    span = link["evidence_span"]
    snippet = span.get("snippet", "")
    snippet_lower = snippet.lower()
    if link["link_basis"] == "fallback":
        return "fallback_review", "fallback link requires manual evidence review"
    if snippet_lower.startswith(METADATA_SNIPPET_PREFIXES):
        return "metadata_only", "snippet came from journal metadata"
    matched_terms = [term for term in span.get("matched_terms", []) if term != "category fallback"]
    snippet_term_hits = sum(1 for term in matched_terms if term.lower() in snippet_lower)
    visible_alias_hits = sum(
        1
        for term in VISIBLE_EVIDENCE_ALIASES.get(link["mechanism_id"], [])
        if term in snippet_lower
    )
    if snippet_term_hits == 0:
        if visible_alias_hits:
            return "weak_direct", "substantive snippet contains visible mechanism aliases"
    if snippet_lower.startswith(STRONG_SNIPPET_PREFIXES) and snippet_term_hits >= 1:
        return "direct", "strong journal section contains matched mechanism terms"
    if len(snippet) < 80:
        if link["confidence"] == "high" and snippet_term_hits >= 1:
            return "direct", "concise high-confidence snippet contains matched mechanism terms"
        return "short_snippet", "snippet is too short for strong generated evidence"
    if snippet_term_hits >= 2 or link["confidence"] == "high":
        if snippet_term_hits >= 2:
            return "direct", "snippet contains multiple matched mechanism terms"
        return "direct", "high-confidence link has a substantive journal snippet"
    if snippet_term_hits == 1 and visible_alias_hits >= 1 and link["confidence"] != "low":
        return "direct", "snippet contains matched mechanism term plus visible aliases"
    if snippet_term_hits == 1 and link.get("review_status") == "reviewed_supported":
        return "direct", "reviewed-supported link has a substantive matched snippet"
    if snippet_term_hits == 1:
        return "weak_direct", "substantive snippet contains only one matched mechanism term"
    return "weak_direct", "substantive snippet has no matched mechanism terms"


def evidence_quality(link: dict) -> str:
    return evidence_quality_details(link)[0]


def evidence_support_reason(link: dict, mechanism_name: str) -> str:
    terms = [term for term in link["evidence_terms"] if term != "category fallback"]
    if terms:
        term_text = ", ".join(terms[:4])
        return f"Matched journal terms ({term_text}) to {mechanism_name}."
    return f"Category fallback mapped this journal entry to {mechanism_name}; review before relying on the link."


def attach_evidence_span(entry: dict, link: dict, mechanism_name: str) -> None:
    link["evidence_span"] = {
        "journal_entry_id": entry["id"],
        "journal_anchor": f"### {entry['date']} - {entry['title']}",
        "matched_terms": link["evidence_terms"],
        "snippet": evidence_snippet(entry.get("body", ""), link["evidence_terms"]),
        "support_reason": evidence_support_reason(link, mechanism_name),
    }
    quality, reason = evidence_quality_details(link)
    link["evidence_span"]["quality"] = quality
    link["evidence_span"]["quality_reason"] = reason


def fallback_mechanisms(category: str, text: str) -> list[str]:
    haystack = f"{category}\n{text[:3000]}".lower()
    result: list[str] = []
    for pattern, mechanism_ids in CATEGORY_FALLBACKS:
        if re.search(pattern, haystack):
            result.extend(mechanism_ids)
    return list(dict.fromkeys(result))


def link_entry(entry: dict, mechanism_ids: set[str]) -> list[dict]:
    haystack = "\n".join(
        [
            entry["title"],
            entry.get("category", ""),
            " ".join(entry.get("relevance_tags", [])),
            entry.get("body", ""),
        ]
    ).lower()
    links: list[dict] = []
    for mechanism_id, terms in MECHANISM_TERMS.items():
        if mechanism_id not in mechanism_ids:
            continue
        score, matched = score_terms(haystack, terms)
        if score < 2:
            continue
        confidence = "high" if score >= 12 else "medium" if score >= 5 else "low"
        links.append(
            {
                "mechanism_id": mechanism_id,
                "confidence": confidence,
                "score": score,
                "evidence_terms": matched,
                "link_basis": "keyword",
            }
        )

    links.sort(key=lambda item: (-item["score"], item["mechanism_id"]))
    links = links[:6]
    if links:
        return links

    fallback = fallback_mechanisms(entry.get("category", ""), haystack)
    return [
        {
            "mechanism_id": mechanism_id,
            "confidence": "needs_review",
            "score": 0,
            "evidence_terms": ["category fallback"],
            "link_basis": "fallback",
        }
        for mechanism_id in fallback
        if mechanism_id in mechanism_ids
    ]


def apply_review_overrides(entry: dict, links: list[dict]) -> list[dict]:
    reviewed: list[dict] = []
    for link in links:
        override = REVIEW_OVERRIDES.get((entry["id"], link["mechanism_id"]))
        if override:
            link["review_status"] = override["review_status"]
            link["review_priority"] = override["review_priority"]
            link["review_note"] = override["review_note"]
            if override["review_status"] == "removed_low_confidence_noise":
                continue
        reviewed.append(link)
    return reviewed


def build_index(entries: list[dict], mechanisms: dict) -> dict:
    mechanism_ids = {item["id"] for item in mechanisms["mechanisms"]}
    mechanism_names = {item["id"]: item["name"] for item in mechanisms["mechanisms"]}
    records: list[dict] = []
    mechanism_counts: Counter = Counter()
    confidence_counts: Counter = Counter()
    evidence_span_counts: Counter = Counter()
    evidence_quality_counts: Counter = Counter()
    evidence_quality_reason_counts: Counter = Counter()
    review_status_counts: Counter = Counter()
    review_priority_counts: Counter = Counter()
    type_counts: Counter = Counter()
    unlinked: list[str] = []

    for entry in entries:
        links = link_entry(entry, mechanism_ids)
        for link in links:
            confidence = link["confidence"]
            link["review_status"] = REVIEW_STATUS_BY_CONFIDENCE[confidence]
            link["review_priority"] = REVIEW_PRIORITY_BY_CONFIDENCE[confidence]
        links = apply_review_overrides(entry, links)
        if not links:
            unlinked.append(entry["id"])
        for link in links:
            attach_evidence_span(entry, link, mechanism_names.get(link["mechanism_id"], link["mechanism_id"]))
            mechanism_counts[link["mechanism_id"]] += 1
            confidence_counts[link["confidence"]] += 1
            evidence_span_counts["links_with_evidence_span"] += 1
            if link["evidence_span"]["snippet"]:
                evidence_span_counts["links_with_evidence_snippet"] += 1
            if link["evidence_span"]["support_reason"]:
                evidence_span_counts["links_with_support_reason"] += 1
            evidence_quality_counts[link["evidence_span"]["quality"]] += 1
            evidence_quality_reason_counts[link["evidence_span"]["quality_reason"]] += 1
            review_status_counts[link["review_status"]] += 1
            review_priority_counts[link["review_priority"]] += 1
        type_counts[entry["entry_type"]] += 1
        records.append(
            {
                "id": entry["id"],
                "date": entry["date"],
                "entry_type": entry["entry_type"],
                "title": entry["title"],
                "category": entry.get("category", ""),
                "relevance_tags": entry.get("relevance_tags", []),
                "citation": entry.get("citation", ""),
                "mechanism_links": links,
            }
        )

    mechanisms_without_links = sorted(mechanism_ids - set(mechanism_counts))
    return {
        "schema": "gpu-db-research-paper-mechanism-links-v3",
        "description": "Generated traceability from literature journal entries to architecture mechanisms, including generated evidence spans and evidence quality reasons. Review low-confidence and fallback links before making architectural commitments.",
        "source_journal": "docs/research/gpu-db-literature-journal.md",
        "source_mechanisms": "docs/research/architecture-compatibility/mechanisms.json",
        "summary": {
            "entries": len(records),
            "paper_entries": type_counts["paper"],
            "synthesis_entries": type_counts["synthesis"],
            "linked_entries": len(records) - len(unlinked),
            "unlinked_entries": len(unlinked),
            "mechanisms_with_links": len(mechanism_counts),
            "mechanisms_without_links": len(mechanisms_without_links),
            "confidence_counts": dict(sorted(confidence_counts.items())),
            "evidence_span_counts": dict(sorted(evidence_span_counts.items())),
            "evidence_quality_counts": dict(sorted(evidence_quality_counts.items())),
            "evidence_quality_reason_counts": dict(sorted(evidence_quality_reason_counts.items())),
            "review_status_counts": dict(sorted(review_status_counts.items())),
            "review_priority_counts": dict(sorted(review_priority_counts.items())),
            "links_requiring_review": review_status_counts["pending_low_confidence_review"]
            + review_status_counts["manual_review_required"],
            "low_confidence_links": confidence_counts["low"],
            "pending_low_confidence_links": review_status_counts["pending_low_confidence_review"],
            "reviewed_low_confidence_links": review_status_counts["reviewed_supported"]
            + review_status_counts["reviewed_weak_signal"],
            "removed_low_confidence_links": sum(
                1
                for override in REVIEW_OVERRIDES.values()
                if override["review_status"] == "removed_low_confidence_noise"
            ),
        },
        "mechanism_counts": dict(sorted(mechanism_counts.items())),
        "mechanisms_without_links": mechanisms_without_links,
        "unlinked_entry_ids": unlinked,
        "records": records,
    }


def write_markdown(index: dict, mechanisms: dict, output: Path) -> None:
    names = {item["id"]: item["name"] for item in mechanisms["mechanisms"]}
    mechanism_counts = Counter(index["mechanism_counts"])
    confidence_counts = index["summary"]["confidence_counts"]
    evidence_span_counts = index["summary"]["evidence_span_counts"]
    evidence_quality_counts = index["summary"]["evidence_quality_counts"]
    evidence_quality_reason_counts = index["summary"]["evidence_quality_reason_counts"]
    review_status_counts = index["summary"]["review_status_counts"]
    review_priority_counts = index["summary"]["review_priority_counts"]
    records = index["records"]
    fallback_records = [
        record
        for record in records
        if any(link["confidence"] == "needs_review" for link in record["mechanism_links"])
    ]
    pending_low_confidence_records = [
        record
        for record in records
        if any(link["review_status"] == "pending_low_confidence_review" for link in record["mechanism_links"])
    ]
    reviewed_low_confidence_records = [
        record
        for record in records
        if any(link["review_status"].startswith("reviewed_") for link in record["mechanism_links"])
    ]
    evidence_quality_audit = [
        (record, link)
        for record in records
        for link in record["mechanism_links"]
        if link["evidence_span"]["quality"] != "direct"
    ]

    lines = [
        "# GPU DB Research Paper-Mechanism Coverage",
        "",
        "This report is generated from the literature journal and mechanism",
        "catalog. It answers whether journal entries are represented in the",
        "architecture compatibility layers.",
        "",
        "Regenerate with:",
        "",
        "```sh",
        "python3 scripts/generate_research_paper_mechanism_links.py",
        "```",
        "",
        "## Summary",
        "",
        f"- journal entries: {index['summary']['entries']}",
        f"- paper entries: {index['summary']['paper_entries']}",
        f"- synthesis entries: {index['summary']['synthesis_entries']}",
        f"- linked entries: {index['summary']['linked_entries']}",
        f"- unlinked entries: {index['summary']['unlinked_entries']}",
        f"- mechanisms with links: {index['summary']['mechanisms_with_links']}",
        f"- mechanisms without links: {index['summary']['mechanisms_without_links']}",
        "",
        "## Link Confidence",
        "",
    ]
    for confidence, count in sorted(confidence_counts.items()):
        lines.append(f"- {confidence}: {count}")

    lines.extend(["", "## Evidence Spans", ""])
    total_links = sum(confidence_counts.values())
    lines.append(f"- links with evidence span: {evidence_span_counts.get('links_with_evidence_span', 0)} / {total_links}")
    lines.append(f"- links with evidence snippet: {evidence_span_counts.get('links_with_evidence_snippet', 0)} / {total_links}")
    lines.append(f"- links with support reason: {evidence_span_counts.get('links_with_support_reason', 0)} / {total_links}")
    lines.append("")
    lines.append("Evidence quality counts:")
    for quality, count in sorted(evidence_quality_counts.items()):
        lines.append(f"- {quality}: {count}")
    lines.append("")
    lines.append("Evidence quality reason counts:")
    for reason, count in sorted(evidence_quality_reason_counts.items()):
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## Evidence Span Quality Audit", ""])
    if evidence_quality_audit:
        for record, link in evidence_quality_audit[:80]:
            span = link["evidence_span"]
            snippet = span["snippet"].replace("|", "\\|")
            lines.append(
                f"- `{record['id']}` -> {link['mechanism_id']}:{span['quality']} ({span['quality_reason']}): {snippet}"
            )
        if len(evidence_quality_audit) > 80:
            lines.append(f"- ... {len(evidence_quality_audit) - 80} more")
    else:
        lines.append("- none")

    lines.extend(["", "## Review Triage", ""])
    lines.append(f"- links requiring review: {index['summary']['links_requiring_review']}")
    lines.append(f"- low-confidence links: {index['summary']['low_confidence_links']}")
    lines.append(f"- pending low-confidence links: {index['summary']['pending_low_confidence_links']}")
    lines.append(f"- reviewed low-confidence links: {index['summary']['reviewed_low_confidence_links']}")
    lines.append(f"- removed low-confidence links: {index['summary']['removed_low_confidence_links']}")
    lines.append("")
    lines.append("Review status counts:")
    for status, count in sorted(review_status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.append("")
    lines.append("Review priority counts:")
    for priority, count in sorted(review_priority_counts.items()):
        lines.append(f"- {priority}: {count}")

    lines.extend(["", "## Mechanism Coverage", ""])
    for mechanism_id, count in mechanism_counts.most_common():
        lines.append(f"- `{mechanism_id}` ({names.get(mechanism_id, mechanism_id)}): {count}")
    if index["mechanisms_without_links"]:
        lines.extend(["", "## Mechanisms Without Links", ""])
        for mechanism_id in index["mechanisms_without_links"]:
            lines.append(f"- `{mechanism_id}` ({names.get(mechanism_id, mechanism_id)})")

    lines.extend(["", "## Fallback Entries Needing Manual Review", ""])
    if fallback_records:
        for record in fallback_records[:200]:
            links = ", ".join(link["mechanism_id"] for link in record["mechanism_links"])
            lines.append(f"- `{record['id']}` -> {links}")
        if len(fallback_records) > 200:
            lines.append(f"- ... {len(fallback_records) - 200} more")
    else:
        lines.append("- none")

    lines.extend(["", "## Entries With Pending Low-Confidence Links", ""])
    if pending_low_confidence_records:
        for record in pending_low_confidence_records[:200]:
            links = ", ".join(
                link["mechanism_id"]
                for link in record["mechanism_links"]
                if link["review_status"] == "pending_low_confidence_review"
            )
            lines.append(f"- `{record['id']}` -> {links}")
        if len(pending_low_confidence_records) > 200:
            lines.append(f"- ... {len(pending_low_confidence_records) - 200} more")
    else:
        lines.append("- none")

    lines.extend(["", "## Reviewed Low-Confidence Links", ""])
    if reviewed_low_confidence_records:
        for record in reviewed_low_confidence_records[:200]:
            links = ", ".join(
                f"{link['mechanism_id']}:{link['review_status']}"
                for link in record["mechanism_links"]
                if link["review_status"].startswith("reviewed_")
            )
            lines.append(f"- `{record['id']}` -> {links}")
        if len(reviewed_low_confidence_records) > 200:
            lines.append(f"- ... {len(reviewed_low_confidence_records) - 200} more")
    else:
        lines.append("- none")

    if index["unlinked_entry_ids"]:
        lines.extend(["", "## Unlinked Entries", ""])
        for entry_id in index["unlinked_entry_ids"]:
            lines.append(f"- `{entry_id}`")

    lines.extend(["", "## Paper Entries", ""])
    for record in records:
        link_text = ", ".join(
            f"{link['mechanism_id']}:{link['confidence']}:{link['review_status']}"
            for link in record["mechanism_links"]
        )
        lines.append(f"- `{record['id']}` ({record['entry_type']}): {link_text}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate journal-to-mechanism traceability for GPU DB research")
    parser.add_argument("--journal", type=Path, default=Path("docs/research/gpu-db-literature-journal.md"))
    parser.add_argument(
        "--mechanisms",
        type=Path,
        default=Path("docs/research/architecture-compatibility/mechanisms.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("docs/research/architecture-compatibility/paper-mechanism-links.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/research/architecture-compatibility/paper-mechanism-coverage.md"),
    )
    args = parser.parse_args()

    journal = args.journal.read_text(encoding="utf-8")
    mechanisms = json.loads(args.mechanisms.read_text(encoding="utf-8"))
    index = build_index(split_entries(journal), mechanisms)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(index, mechanisms, args.markdown_output)


if __name__ == "__main__":
    main()
