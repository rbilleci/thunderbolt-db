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
        "state-machine operation",
        "highest-numbered accepted values",
        "fast routes should be allowed to optimize",
        "learned route choices need a reliability gate",
        "bounded pressure signals and escape paths",
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
        "compact roots",
        "explicit resource proofs",
        "generation markers",
        "relation generation",
        "what recovery may trust",
        "system is deadlocked",
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
        "outstanding memory loads",
        "cache-miss latency",
        "semantic regions prove coverage",
        "table generation, columns, predicates, and overlap rules",
        "validated storage metadata",
        "correctness and failure state",
        "visible state transition",
        "metadata for recovery, validation, and backup",
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
        "packet pacing",
        "desired load",
        "preserving line-rate starts",
        "gpu execution fairness",
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
        "transaction execution",
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
        "concurrent readers/writers",
        "ddl invalidation",
        "smart-contract object model",
        "compact witness generation",
        "semantic freshness witnesses",
        "searchable visibility witnesses",
        "correctness and failure state",
        "leave behind enough proof",
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
        "compressed bitvectors",
        "sparse update state",
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
        "semantic regions prove coverage",
        "data product",
        "version tree",
        "resident generation",
        "gpu/cpu execution credit",
        "cross-reactor work",
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
        "route-shape choice",
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
        "resident access methods",
        "read/update/memory costs",
        "compressed bitvectors",
        "selective reads",
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
        "stale-certificate tests",
        "tier-pin failures",
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
        "battery-backed host buffers",
        "small durable commit window",
        "dictionary encoding",
        "nested structures",
        "route metadata should be published",
        "batch/snapshot conflict decision",
        "bounded key-range column merge",
        "volatile whenever recovery can rebuild",
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
        "known conflicts",
        "priority queue",
        "direct dependents",
        "active windows can be separated cheaply",
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
        "base resident snapshot with deltas",
        "write admission",
        "merge a base resident snapshot with deltas",
        "retained point lookup",
        "refresh stream",
        "gpu copy bandwidth",
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
        "right rows are reachable",
        "partial wal frontiers",
        "costed retention frontiers",
        "append-ordered tiered stream storage",
        "base resident snapshot with deltas",
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
        "hidden host-network queues",
        "mapping publication",
        "deterministic eligibility gates",
        "physical warm-tier movement",
        "per-fragment metadata scan",
        "semantic freshness witnesses",
        "searchable visibility witnesses",
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

RELATION_TYPES = {
    "supports",
    "warns_against",
    "contradicts",
    "alternative_to",
    "only_valid_if",
    "benchmark_required",
}

RELATION_TYPE_DESCRIPTIONS = {
    "supports": "paper evidence supports or motivates this mechanism",
    "warns_against": "paper evidence warns against adopting this mechanism without constraints",
    "contradicts": "paper evidence conflicts with this mechanism",
    "alternative_to": "paper evidence describes an alternative to this mechanism",
    "only_valid_if": "paper evidence supports this mechanism only under named conditions",
    "benchmark_required": "paper evidence is inconclusive without a benchmark or proof gate",
}

RELATION_CANDIDATE_CUES = {
    "contradicts": [
        "contradicts",
        "conflicts with",
        "incompatible",
        "violates",
    ],
    "warns_against": [
        "risk",
        "risks and mismatches",
        "warning",
        "fragile",
        "does not automatically",
        "does not fit",
        "not directly",
        "not enough",
        "not generally",
        "not just",
        "too strict",
    ],
    "only_valid_if": [
        "only if",
        "only when",
        "unless",
        "precondition",
        "valid but",
        "validity constraints",
        "requires proof",
        "requires all",
        "must prove",
        "must reject",
        "must restore",
        "condition",
        "assumes",
        "restrict",
    ],
    "benchmark_required": [
        "benchmark",
        "microbenchmark",
        "measure",
        "proof gate",
        "stress",
        "evaluate",
        "prototype",
        "test ",
    ],
    "alternative_to": [
        "alternative",
        "alternatives",
        "instead of",
        "rather than",
    ],
}

RELATION_CANDIDATE_PRIORITY = [
    "contradicts",
    "warns_against",
    "only_valid_if",
    "benchmark_required",
    "alternative_to",
]


RELATION_REVIEW_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    (
        "2026-06-07-taurus-mm-makes-multi-master-snapshots-cheap-enough-for-shared-storage",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Fragility describes the baseline multi-master design, while the compact scalar/vector frontier state supports this mechanism.",
    },
    (
        "2026-06-07-leopard-turns-isolation-semantics-into-an-online-verifier",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The verifier is an external reconstruction path rather than in-kernel descriptor instrumentation.",
    },
    (
        "2026-06-07-leopard-turns-isolation-semantics-into-an-online-verifier",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The entry explicitly routes this visibility tracing idea through backlog, anomaly, latency, and throughput measurement.",
    },
    (
        "2026-06-07-decentsched-makes-deterministic-hot-writes-self-schedule",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than clause motivates deterministic templates instead of abort/retry ordering.",
    },
    (
        "2026-06-07-decentsched-makes-deterministic-hot-writes-self-schedule",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Owner-local queue ordering is presented as an alternative to routing all ordering through one global owner.",
    },
    (
        "2026-06-07-decentsched-makes-deterministic-hot-writes-self-schedule",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The witness idea is tied to measuring false-positive waits, cache misses, search time, and metadata footprint.",
    },
    (
        "2026-06-07-justdo-turns-logging-into-resumable-progress-state",
        "dependency_witnesses",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The retained evidence is explicitly in risks and mismatches and depends on cheap persist ordering.",
    },
    (
        "2026-06-07-cross-paper-synthesis-route-admission-now-needs-three-witnesses",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Admission is supported only when semantic, resource, and recovery witnesses are all present.",
    },
    (
        "2026-06-07-cross-paper-synthesis-route-admission-now-needs-three-witnesses",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Route-root publication is valid only with proven snapshot/catalog/resident generations and safe descriptor state.",
    },
    (
        "2026-06-07-cross-paper-synthesis-route-admission-now-needs-three-witnesses",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The durable compact-progress contrast is not an optimizer alternative; it remains a weak supporting signal.",
    },
    (
        "2026-06-07-hostping-makes-host-interconnect-health-a-route-precondition",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fallback is valid only when path-degraded state is distinguished from stale generation and unsupported predicates.",
    },
    (
        "2026-06-07-hostping-makes-host-interconnect-health-a-route-precondition",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained GPU reads are valid only while the resident snapshot is valid and path degradation is handled.",
    },
    (
        "2026-06-07-tetrisched-plans-scarce-accelerators-in-space-and-time",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The entry requires p50/p99, GPU utilization, fallback pressure, missed-SLO, and stale-route measurements.",
    },
    (
        "2026-06-07-tetrisched-plans-scarce-accelerators-in-space-and-time",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained snapshots are part of heterogeneous sets with explicit runtime and validity constraints.",
    },
    (
        "2026-06-07-tetrisched-plans-scarce-accelerators-in-space-and-time",
        "same_shape_microbatching",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Same-shape micro-batching is one supported scheduling option, not an alternative to the mechanism itself.",
    },
    (
        "2026-06-07-tetrisched-plans-scarce-accelerators-in-space-and-time",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The route-advisor idea is framed as simulator work over route alternatives before adoption.",
    },
    (
        "2026-06-07-cross-paper-synthesis-format-metadata-is-now-route-metadata",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evidence lists route-certificate and mixed-freshness benchmarks as the decision gate.",
    },
    (
        "2026-06-07-adaptive-htap-treats-freshness-as-a-resource-scheduling-input",
        "htap_freshness_router",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Freshness routing is valid only if OLAP gains do not violate WAL, visibility, or mutation latency budgets.",
    },
    (
        "2026-06-07-adaptive-htap-treats-freshness-as-a-resource-scheduling-input",
        "resource_dag_scheduling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The paper frames runtime scheduling as an alternative to fixed unified or decoupled HTAP modes.",
    },
    (
        "2026-06-07-adaptive-htap-treats-freshness-as-a-resource-scheduling-input",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "GPU-specific placement costs are explicitly not measured and need a proof gate.",
    },
    (
        "2026-06-07-doppelganger-makes-dependency-ordering-a-streaming-state-problem",
        "effective_session_counting",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Per-request dependency state at 1M sessions is valid only if bounded, sampled, or restricted.",
    },
    (
        "2026-06-07-doppelganger-makes-dependency-ordering-a-streaming-state-problem",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The word test names the replay system context; the snippet still supports deterministic conflict ordering.",
    },
    (
        "2026-06-07-durinn-turns-visibility-vs-durability-gaps-into-adversarial-tests",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evaluated systems are smaller than a database engine, so retained GPU snapshots require DB-scale evaluation.",
    },
    (
        "2026-06-07-version-aware-layout-makes-mvcc-visibility-a-search-key",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Grouping old deltas by generation supports frontier-style snapshot skipping despite the rather-than cue.",
    },
    (
        "2026-06-07-version-aware-layout-makes-mvcc-visibility-a-search-key",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The paper does not evaluate WAL durability or crash recovery, so WAL-before-visibility needs validation.",
    },
    (
        "2026-06-07-version-aware-layout-makes-mvcc-visibility-a-search-key",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-local old-version buffers are explicitly proposed as prototype-and-measure work.",
    },
    (
        "2026-06-07-version-aware-layout-makes-mvcc-visibility-a-search-key",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The word tests describes the visibility-check operation; the evidence supports MVCC GC frontier mechanics.",
    },
    (
        "2026-06-07-cross-paper-synthesis-learned-route-control-needs-deterministic-envelopes",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis explicitly requires fixed, learned-ranking, and learned-batch benchmark variants.",
    },
    (
        "2026-06-07-cross-paper-synthesis-learned-route-control-needs-deterministic-envelopes",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Deterministic envelopes are tied to route-DAG telemetry and continuous mixed-arrival benchmarks.",
    },
    (
        "2026-06-07-cross-paper-synthesis-learned-route-control-needs-deterministic-envelopes",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Learned policy output is valid only when published as immutable generations that hot workers evaluate cheaply.",
    },
    (
        "2026-06-07-pilotscope-turns-learned-planning-into-bounded-push-pull-drivers",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Learned policy publication is useful only if retired metadata remains bounded and readers are not stalled.",
    },
    (
        "2026-06-07-asap-treats-persist-ordering-as-recoverable-speculation",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The entry explicitly requires owner queue, descriptor publication, stale-route, and two-device speculation measurements.",
    },
    (
        "2026-06-07-steam-prunes-mvcc-garbage-on-the-write-path-before-chains-grow",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Steam reports active-list snapshot costs, but GPU DB still needs a retained-snapshot frontier benchmark.",
    },
    (
        "2026-06-07-steam-prunes-mvcc-garbage-on-the-write-path-before-chains-grow",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The proposed transfer is gated on write/read latency, cleanup debt, retired bytes, and snapshot-age measurements.",
    },
    (
        "2026-06-07-rapidlane-turns-hot-shared-counters-into-deferred-commit-time-deltas",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Deferred object operations avoid immediate read-modify-write descriptor churn for narrow hot-counter cases.",
    },
    (
        "2026-06-07-rapidlane-turns-hot-shared-counters-into-deferred-commit-time-deltas",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "The hot-key fast path is valid only with typed deferred deltas, explicit preconditions, and commit-time proof.",
    },
    (
        "2026-06-07-tips-keeps-persistent-indexes-out-of-the-request-s-critical-path",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Rebuildable resident metadata is acceptable only if normal SQL commits still obey WAL-before-visibility.",
    },
    (
        "2026-06-07-tips-keeps-persistent-indexes-out-of-the-request-s-critical-path",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Overlay-based retained reads need explicit fallback when pending overlay depth would exceed the route SLO.",
    },
    (
        "2026-06-07-cross-paper-synthesis-persistent-metadata-needs-overlay-replay-and-witnesses",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis names overlay-depth, crash-state, replay-lag, and snapshot-correct range/pruning benchmarks.",
    },
    (
        "2026-06-06-cross-paper-synthesis-durable-metadata-needs-recoverable-shape",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner handoff is supported only when the communication and commit boundary are worth making explicit.",
    },
    (
        "2026-06-06-cross-paper-synthesis-durable-metadata-needs-recoverable-shape",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The only-if phrase is corpus-planning guidance; the mechanism evidence supports immutable retained snapshots.",
    },
    (
        "2026-06-06-splinterdb-turns-nvme-storage-into-a-cpu-efficiency-problem",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Branch-sized sequential rebuilds are presented as an alternative to scattered tuple-chain reads.",
    },
    (
        "2026-06-06-memstrata-makes-cxl-tiering-an-isolation-and-outlier-control-problem",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Hardware-managed tiering is useful only if route-critical object placement is not treated as stable by assumption.",
    },
    (
        "2026-06-06-colloid-balances-loaded-tier-latency-instead-of-hoarding-hot-pages",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The loaded-latency evidence motivates measuring contention-sensitive hot-write placement choices.",
    },
    (
        "2026-06-06-dumbo-makes-durable-read-only-transactions-wait-only-for-older-non-durable-writes",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The marker-array design is explicitly framed as a GPU DB WAL metadata benchmark.",
    },
    (
        "2026-06-06-dumbo-makes-durable-read-only-transactions-wait-only-for-older-non-durable-writes",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Compact per-owner publication arrays are proposed instead of every read touching a heavyweight transaction table.",
    },
    (
        "2026-06-06-dumbo-makes-durable-read-only-transactions-wait-only-for-older-non-durable-writes",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Per-owner compact state arrays require read, writer-publication, stale-generation, and retired-token measurements.",
    },
    (
        "2026-06-06-leon-keeps-learned-route-choice-behind-an-expert-optimizer",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "LEON frames learned planning as assistance to, not replacement for, mature optimizer enumeration and costing.",
    },
    (
        "2026-06-06-leon-keeps-learned-route-choice-behind-an-expert-optimizer",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "ML-guided exploration is valid only if it does not consume GPU or pinned-buffer credits needed by admitted work.",
    },
    (
        "2026-06-06-cross-paper-synthesis-durable-publication-needs-small-proofs-with-bounded-fallback",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained lookup, mutation, refresh, retired-metadata, and p99 route behavior are named measurement gates.",
    },
    (
        "2026-06-06-dhtm-treats-durability-as-part-of-the-transaction-fast-path",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The durability fast path needs write latency, retained-read fallback, staleness, and recovery replay measurements.",
    },
    (
        "2026-06-06-dhtm-treats-durability-as-part-of-the-transaction-fast-path",
        "cpu_fallback_policy",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Silent fast-path expansion risks unbounded tail latency without typed overflow and fallback outcomes.",
    },
    (
        "2026-06-06-dhtm-treats-durability-as-part-of-the-transaction-fast-path",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshots need durability-bandwidth stress tests for WAL, invalidation, descriptors, and GPU staging.",
    },
    (
        "2026-06-06-drtm-turns-hardware-transactions-into-a-local-fast-path-with-remote-locks-as-proof",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Local fast execution is valid only after remote, cold, resident, WAL, and invalidation dependencies become bounded proofs.",
    },
    (
        "2026-06-06-drtm-turns-hardware-transactions-into-a-local-fast-path-with-remote-locks-as-proof",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained-route execution is valid only when every remote and resident dependency has a bounded proof.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-proof-before-execution-not-cleanup-after-failure",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis requires queue wait, retained latency, refresh starvation, WAL reservation, stale-generation, and safety measurements.",
    },
    (
        "2026-06-06-nvwal-makes-durable-logging-a-byte-granular-persistent-memory-protocol",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "NVWAL targets SQLite-style mobile logging and warns against direct adoption for multi-session GPU-resident snapshots.",
    },
    (
        "2026-06-06-nvwal-makes-durable-logging-a-byte-granular-persistent-memory-protocol",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Byte-addressable logging argues for transaction-shaped records rather than page-write shaped owner coordination.",
    },
    (
        "2026-06-06-nvwal-makes-durable-logging-a-byte-granular-persistent-memory-protocol",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The evidence favors semantic log and allocator proofs rather than page-write dependency tracking.",
    },
    (
        "2026-06-06-lance-makes-random-columnar-access-a-structural-encoding-problem",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Lance is not a transactional WAL/MVCC/recovery design, so it cautions against inferring WAL-before-visibility support.",
    },
    (
        "2026-06-06-farm-makes-distributed-commit-a-reservation-backed-rdma-log-protocol",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Reservation-backed owner coordination is useful only after hot-key latency, abort, queue-wait, and unused-capacity measurements.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-metadata-needs-proof-fields-bounded-lifetime-and-sampled-movement",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Route decisions support WAL visibility only when durability, visibility, freshness, placement, and queue authority are provable.",
    },
    (
        "2026-06-06-citron-makes-remote-range-locks-a-static-metadata-protocol",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Ancestor/descendant range counters are an alternative coordination shape to arbitrary interval-set or central-queue ownership.",
    },
    (
        "2026-06-06-citron-makes-remote-range-locks-a-static-metadata-protocol",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue contrasts compact bounded range metadata with heaps; it supports bounded descriptor shaping.",
    },
    (
        "2026-06-06-paella-turns-gpu-scheduling-into-a-software-owned-dispatch-contract",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Queued GPU routes are valid only when dispatch proves schema, snapshot, resident, and output-order generations.",
    },
    (
        "2026-06-06-skyplane-makes-cold-tier-movement-a-constrained-overlay-plan",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cold-tier placement depends on measured throughput across HBM, host, DRAM, NVMe, object, and remote tiers.",
    },
    (
        "2026-06-06-skyplane-makes-cold-tier-movement-a-constrained-overlay-plan",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The cloud-transfer planning evidence needs a GPU DB session-scale benchmark before supporting session counting.",
    },
    (
        "2026-06-06-skyplane-makes-cold-tier-movement-a-constrained-overlay-plan",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Future cold-tier movement descriptors require measured throughput, transfer cost, queue capacity, and freshness-budget gates.",
    },
    (
        "2026-06-06-cross-paper-synthesis-cold-tier-movement-needs-route-certificates-not-background-mystery-copies",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Measured cold and warm route inputs are part of the intended multi-tier placement mechanism, not separate audit debt.",
    },
    (
        "2026-06-06-cross-paper-synthesis-cold-tier-movement-needs-route-certificates-not-background-mystery-copies",
        "wal_before_visibility",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless clause is corpus-planning guidance; the retained evidence still keeps WAL and visibility as preferred next constraints.",
    },
    (
        "2026-06-06-cloudcast-turns-cold-tier-replication-into-an-explicit-cost-time-and-stripe-routing-optimization",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CloudCast-style stripe routes need prototype validation before combining WAL boundary, visibility, encoding, and checksum lineage.",
    },
    (
        "2026-06-06-xenic-puts-transaction-protocol-state-on-the-network-edge",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Gateway route-edge caching is explicitly framed as a 10K, 100K, and simulated 1M logical-session benchmark.",
    },
    (
        "2026-06-06-tdsql-makes-scale-out-oltp-a-proxy-shard-and-jitter-control-problem",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The stress-run stability evidence warns against conflict-ordering designs that optimize throughput without rollback and jitter controls.",
    },
    (
        "2026-06-06-tdsql-makes-scale-out-oltp-a-proxy-shard-and-jitter-control-problem",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The gateway, planner, mutation-owner, residency-owner, and runtime queue path must be measured as one transaction path.",
    },
    (
        "2026-06-06-tdsql-makes-scale-out-oltp-a-proxy-shard-and-jitter-control-problem",
        "deficit_fairness",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The stability result warns that fairness policy cannot treat throughput as sufficient without jitter and rollback accounting.",
    },
    (
        "2026-06-06-shiftlock-turns-hot-remote-locks-into-handoff-queues",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Handoff-eligible contention states support deterministic hot-write templates instead of blind retry loops.",
    },
    (
        "2026-06-06-shiftlock-turns-hot-remote-locks-into-handoff-queues",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "ShiftLock does not measure 1M logical sessions or GPU DB queues, so session-count transfer needs benchmarking.",
    },
    (
        "2026-06-06-shiftlock-turns-hot-remote-locks-into-handoff-queues",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The paper omits mixed local and remote placement, CUDA buffers, and cold-tier traffic measurements.",
    },
    (
        "2026-06-06-shiftlock-turns-hot-remote-locks-into-handoff-queues",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "ShiftLock is distributed locking rather than SQL WAL, MVCC, GPU execution, or recovery, so direct WAL inference is unsafe.",
    },
    (
        "2026-06-06-cross-paper-synthesis-adaptive-routes-need-local-caches-reusable-learning-and-handoff-under-cont",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The handoff-queue contrast supports owner bundling by moving hot authority pressure into explicit owner-local queues.",
    },
    (
        "2026-06-06-cross-paper-synthesis-adaptive-routes-need-local-caches-reusable-learning-and-handoff-under-cont",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue rejects hammering shared objects and supports deterministic handoff states for hot writes.",
    },
    (
        "2026-06-06-bytehouse-makes-disaggregated-storage-local-through-ssd-chunks-and-route-modes",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The warm-tier cache descriptor is explicitly prototype work with local NVMe, host DRAM, HBM, and object offsets.",
    },
    (
        "2026-06-06-crystalline-bounds-reclamation-without-session-shaped-snapshots",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The transfer requires stressing 1M logical sessions over fixed workers before relying on physical-worker protection.",
    },
    (
        "2026-06-06-crystalline-bounds-reclamation-without-session-shaped-snapshots",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cleanup-owner queue telemetry and pinned-buffer release behavior need stress validation before owner bundling adoption.",
    },
    (
        "2026-06-06-cross-paper-synthesis-adaptive-routes-also-need-bounded-metadata-lifetimes",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Descriptor reclamation transfer is gated on generation checks, retired-byte bounds, route retries, and p99 latency.",
    },
    (
        "2026-06-06-cross-paper-synthesis-adaptive-routes-also-need-bounded-metadata-lifetimes",
        "dependency_witnesses",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue describes named fallback proof boundaries, which directly support dependency witnesses.",
    },
    (
        "2026-06-06-adaptive-filters-beat-brittle-route-confidence-without-training",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Adaptive filters are presented as a no-training alternative that can match or beat learned query optimizers.",
    },
    (
        "2026-06-06-adaptive-filters-beat-brittle-route-confidence-without-training",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evidence is benchmark-centered and needs GPU DB validation before informing descriptor lifetime policy.",
    },
    (
        "2026-06-06-multiverse-versions-only-when-long-readers-prove-they-need-it",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "MVCC frontiers are valid only if long readers do not force unbounded metadata growth or reclamation stalls.",
    },
    (
        "2026-06-06-poplar-relaxes-wal-order-to-the-dependencies-recovery-actually-needs",
        "wal_before_visibility",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Poplar suggests per-owner or per-partition durability frontiers instead of forcing every read behind unrelated WAL traffic.",
    },
    (
        "2026-06-06-poplar-relaxes-wal-order-to-the-dependencies-recovery-actually-needs",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Poplar proposes durable owner-stream frontiers as an alternative to a single global LSN for retained snapshot publication.",
    },
    (
        "2026-06-06-poplar-relaxes-wal-order-to-the-dependencies-recovery-actually-needs",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Dependency-ordered recovery is explicitly gated on replay benchmarks and same-visible-snapshot proof checks.",
    },
    (
        "2026-06-06-fisslock-splits-fast-grant-facts-from-heavy-waiter-state",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Moving fast grant facts out of heavy waiter state is valid only if database-owned WAL, MVCC, and recovery proofs remain intact.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-authorities-should-publish-small-facts-and-keep-heavy-state-local",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis names WAL/recovery and transaction scheduling as the next evaluation gap for publication boundaries.",
    },
    (
        "2026-06-06-smartqueue-treats-cache-residency-as-scheduler-state",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts greedy execution with cache-aware scheduling, which supports placement-aware routing.",
    },
    (
        "2026-06-06-ford-makes-remote-durable-transactions-a-round-trip-budget",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Mutable cold-tier descriptor acquisition is framed as something the engine should test before adopting the route shape.",
    },
    (
        "2026-06-06-pulse-moves-pointer-traversal-to-the-future-memory-tier",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Near-memory pointer traversal is useful only when the continuation program stays restricted and iterator-shaped.",
    },
    (
        "2026-06-06-pulse-moves-pointer-traversal-to-the-future-memory-tier",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "PULSE is not a SQL WAL, MVCC, isolation, or recovery protocol, so it cautions against direct WAL inference.",
    },
    (
        "2026-06-06-pulse-moves-pointer-traversal-to-the-future-memory-tier",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evidence requires remote pointer-hop evaluation, so optimizer transfer needs route-cost benchmarking beyond byte counts.",
    },
    (
        "2026-06-06-cross-paper-synthesis-retained-routes-need-bounded-reconstruction-placement-and-retirement",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier demotion is gated on SLO reconstruction latency and descriptor-retirement stress tests.",
    },
    (
        "2026-06-06-cross-paper-synthesis-retained-routes-need-bounded-reconstruction-placement-and-retirement",
        "effective_session_counting",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Session-counting transfer is valid only if reclamation state avoids scaling with logical sessions and heavy cleanup on IO workers.",
    },
    (
        "2026-06-06-cross-paper-synthesis-retained-routes-need-bounded-reconstruction-placement-and-retirement",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Immutable route-root adoption is gated on reconstruction-latency budgets and descriptor-retirement stress.",
    },
    (
        "2026-06-06-cross-paper-synthesis-retained-routes-need-bounded-reconstruction-placement-and-retirement",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Hyaline-style physical-worker reachability is presented as an alternative retirement basis for old descriptors and resident handles.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-correctness-needs-external-witnesses-too",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-root correctness is routed through route-history traces and deterministic write-window benchmarks before solver verification.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-correctness-needs-external-witnesses-too",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL boundary correctness is explicitly tied to trace-format and serial-replay benchmark gates.",
    },
    (
        "2026-06-06-cobra-turns-serializability-into-an-off-path-route-history-check",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evidence calls for selected stress-test route-history records including WAL boundary and publication generation.",
    },
    (
        "2026-06-06-cobra-turns-serializability-into-an-off-path-route-history-check",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Descriptor-lifetime transfer needs stress-test route-history records before it can support retained execution.",
    },
    (
        "2026-06-06-cobra-turns-serializability-into-an-off-path-route-history-check",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Immutable route roots need route-history stress traces that capture route shape, snapshot boundary, and publication generation.",
    },
    (
        "2026-06-06-sundial-unifies-cache-validity-and-transaction-order-with-logical-leases",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Sundial omits CUDA, GPU placement, pinned buffers, session multiplexing, NVMe recovery, and MVCC storage costs.",
    },
    (
        "2026-06-06-sundial-unifies-cache-validity-and-transaction-order-with-logical-leases",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Persisted route metadata needs recovery tests and measurements before leases can inform WAL visibility boundaries.",
    },
    (
        "2026-06-06-sundial-unifies-cache-validity-and-transaction-order-with-logical-leases",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Validity intervals and generation ranges are presented as an alternative to simple fresh-or-stale retained snapshot state.",
    },
    (
        "2026-06-06-cross-paper-synthesis-budgeted-metadata-must-carry-route-proof-not-only-speed",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Bounded retirement is explicitly part of the route-certificate benchmark and proof-gate backlog.",
    },
    (
        "2026-06-06-cross-paper-synthesis-budgeted-metadata-must-carry-route-proof-not-only-speed",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The not-just cue rejects faster optimistic retry and supports explicit hot-write conflict ownership templates.",
    },
    (
        "2026-06-06-cross-paper-synthesis-budgeted-metadata-must-carry-route-proof-not-only-speed",
        "deficit_fairness",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The evidence supports fairness policy by requiring explicit priority and fairness rules beyond faster retry.",
    },
    (
        "2026-06-06-cross-paper-synthesis-budgeted-metadata-must-carry-route-proof-not-only-speed",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Placement is explicitly named as a benchmark requiring tiered metadata and fallback or promotion telemetry.",
    },
    (
        "2026-06-06-cabin-makes-scan-indexes-budgetable-resident-metadata",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cabin's resident scan-index idea is a size and memory-budget problem that needs GPU DB testing before descriptor adoption.",
    },
    (
        "2026-06-06-polyjuice-treats-concurrency-control-as-a-learned-route-policy",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The learned policy is evaluated through benchmark baselines and needs deterministic hot-write comparison before adoption.",
    },
    (
        "2026-06-06-orthrus-separates-conflict-ownership-from-transaction-execution",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts cache-coherence contention with queueing at an owner, directly supporting owner bundling.",
    },
    (
        "2026-06-06-cross-paper-synthesis-tail-control-needs-bounded-retry-not-only-faster-queues",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "GPU conflict ordering is gated on p99.9, max retry count, and route-stampede benchmarks.",
    },
    (
        "2026-06-06-sherman-makes-remote-indexes-write-friendly-by-moving-proof-to-tiny-ordered-updates",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Descriptor publication is safe only if generation state cannot point at uncommitted or partially updated entries.",
    },
    (
        "2026-06-06-sherman-makes-remote-indexes-write-friendly-by-moving-proof-to-tiny-ordered-updates",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Ordered future-tier update and publish coalescing is explicitly framed as something to test before adoption.",
    },
    (
        "2026-06-06-sherman-makes-remote-indexes-write-friendly-by-moving-proof-to-tiny-ordered-updates",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue contrasts wait points; the ordered generation-marker publish still supports compact route roots.",
    },
    (
        "2026-06-06-leanstore-recovery-makes-wal-a-sharded-tiered-and-checkpoint-bounded-pipeline",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Edge logging can publish visibility only when canonical recovery or deterministic replay covers the write.",
    },
    (
        "2026-06-06-leanstore-recovery-makes-wal-a-sharded-tiered-and-checkpoint-bounded-pipeline",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evidence names DRAM, NVMe, battery-backed, CXL/NVDIMM-like, and edge-staged benchmark variants.",
    },
    (
        "2026-06-06-leanstore-recovery-makes-wal-a-sharded-tiered-and-checkpoint-bounded-pipeline",
        "log_structured_warm_tier",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "LeanStore-specific buffer-manager and PMem/NVMe assumptions warn against direct warm-tier transfer.",
    },
    (
        "2026-06-06-itlogging-turns-wal-overhead-into-an-admission-boundary-problem",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner handoff is gated on throughput, latency, owner CPU, copied bytes, recovery work, and crash-state tests.",
    },
    (
        "2026-06-06-itlogging-turns-wal-overhead-into-an-admission-boundary-problem",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The TPC-C and LinkBench evidence needs GPU DB conflict-ordering validation before architectural adoption.",
    },
    (
        "2026-06-06-itlogging-turns-wal-overhead-into-an-admission-boundary-problem",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Staged request references require payload checksum and generation-stamp validation before route-root adoption.",
    },
    (
        "2026-06-06-eemarq-makes-retained-range-snapshots-compatible-with-aggressive-reclamation",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained range snapshots are valid only when stale index entries revalidate snapshot and route generations.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-proofs-need-reclamation-proofs-too",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Descriptor lifetime, stale-route, long-reader, scheduling, and recovery witnesses are named stress gates.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-proofs-need-reclamation-proofs-too",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained reads need stress coverage for descriptor reuse, stale routes, scheduling refusal, and recovery proof.",
    },
    (
        "2026-06-06-smart-makes-remote-index-traversal-a-cache-validation-and-iops-shaping-problem",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner-owned leaf queues are valid only if read-after-write and invalidation-after-read ordering are proven.",
    },
    (
        "2026-06-06-smart-makes-remote-index-traversal-a-cache-validation-and-iops-shaping-problem",
        "stable_handle_indirection",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Radix-style indirection helps only with remote locking, concurrent access handling, and cached-node validation.",
    },
    (
        "2026-06-06-lsched-makes-query-scheduling-a-physical-plan-and-pressure-problem",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Adaptive scheduling is explicitly evaluated outside the trusted serving path before being adopted.",
    },
    (
        "2026-06-06-carousel-overlaps-read-prepare-commit-and-replication-when-the-route-shape-is-known",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fixed-footprint same-shape micro-batching is called out as a measurement gate.",
    },
    (
        "2026-06-06-carousel-overlaps-read-prepare-commit-and-replication-when-the-route-shape-is-known",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Deterministic hot-write templates require abort-after-GPU-work, latency, and queue-wait measurements.",
    },
    (
        "2026-06-06-carousel-overlaps-read-prepare-commit-and-replication-when-the-route-shape-is-known",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Admission-time preflight is prototype work gated by identical serializable outcomes and WAL-before-visibility behavior.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-overlap-needs-proof-shaped-admission",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fallback policy depends on measured p99, conflict-rate crossover, and explicit fallback-reason output.",
    },
    (
        "2026-06-06-laser-buffer-aware-learned-scheduling-should-route-by-residency-footprint-not-only-load",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Laser's replicated PostgreSQL query-routing scope cautions against direct retained GPU snapshot transfer.",
    },
    (
        "2026-06-06-laser-buffer-aware-learned-scheduling-should-route-by-residency-footprint-not-only-load",
        "resource_dag_scheduling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue supports physical-footprint grouping rather than treating GPU queue depth as the only signal.",
    },
    (
        "2026-06-06-query-compiler-architecture-should-preserve-planner-facts-until-code-generation",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Specializing static plan structure at generation time supports preserving route facts through publication.",
    },
    (
        "2026-06-06-query-compiler-architecture-should-preserve-planner-facts-until-code-generation",
        "cpu_fallback_policy",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue contrasts generic loops with certificate-specialized fallback and staging code.",
    },
    (
        "2026-06-06-query-compiler-architecture-should-preserve-planner-facts-until-code-generation",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The alternative access paths are the costed placement choices this mechanism is meant to expose.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-semantic-certificates-reusable-descriptors-and-generation",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue supports version-validated recycled metadata rather than unbounded pinned retired state.",
    },
    (
        "2026-06-06-datacenter-ethernet-and-rdma-issues-at-hyperscale",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The networking evidence omits database transactions, GPU kernels, WAL, MVCC, and cache-placement measurements.",
    },
    (
        "2026-06-06-rome-robust-query-optimization-via-parallel-multi-plan-execution",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue is an optimizer comparison; the retained link remains only a weak descriptor-lifetime support signal.",
    },
    (
        "2026-06-06-rome-robust-query-optimization-via-parallel-multi-plan-execution",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Parallel optimizer-generated alternatives are presented as an alternative to relying on learned optimizer advice.",
    },
    (
        "2026-06-06-decibel-the-relational-dataset-branching-system",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Decibel warns that timestamps alone do not make retained historical snapshots cheap or physically safe.",
    },
    (
        "2026-06-06-decibel-the-relational-dataset-branching-system",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Recovery publication is valid only with demotion, rebuild, compressed-delta, WAL replay, and checksum proof.",
    },
    (
        "2026-06-06-decibel-the-relational-dataset-branching-system",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The SimpleDB prototype omits modern contention, GPU execution, CUDA ownership, NVMe tiering, and pgwire scale.",
    },
    (
        "2026-06-02-scalable-and-robust-snapshot-isolation-for-high-performance-storage-engines",
        "owner_ring_bundling",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The entry directly warns that retained snapshots are not enough if old snapshot state stays on the mutation owner's hot path.",
    },
    (
        "2026-06-02-datacenter-rpcs-can-be-general-and-fast",
        "vector_credit_admission",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Packet receive queues and CPU-managed connection state are presented as an alternative flow-control shape to RDMA-write polling.",
    },
    (
        "2026-06-02-concurrent-analytical-query-processing-with-gpus",
        "resource_dag_scheduling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "MultiQx-GPU is framed as concurrent compatible query scheduling instead of dedicating one GPU to one query.",
    },
    (
        "2026-06-02-data-path-fusion-in-gpu-for-analytical-query-processing",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The evaluate cue is part of visibility evaluation inside a fused route descriptor, so the link remains supporting evidence.",
    },
    (
        "2026-06-02-data-path-fusion-in-gpu-for-analytical-query-processing",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The future text route explicitly requires testing an FSST/RID-index page layout against the current resident representation.",
    },
    (
        "2026-06-02-data-path-fusion-in-gpu-for-analytical-query-processing",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The evaluate cue is part of generated kernel work; the stable route family and descriptor still support route-root publication.",
    },
    (
        "2026-06-02-scaling-gpu-accelerated-databases-beyond-gpu-memory-size",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The evaluate cue names CPU predicate evaluation, while the evidence supports choosing CPU/GPU split routes by cost.",
    },
    (
        "2026-06-02-first-modern-batch-synthesis",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained reads are attractive only when admitted work shares a compatible snapshot, route shape, and visibility boundary.",
    },
    (
        "2026-06-02-first-modern-batch-synthesis",
        "same_shape_microbatching",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Micro-batching is supported only when every request in the batch shares compatible snapshot, route, and visibility boundaries.",
    },
    (
        "2026-06-02-first-modern-batch-synthesis",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Route optimization is useful only when demand and dependency metadata are explicit before admission.",
    },
    (
        "2026-06-02-virtual-memory-assisted-buffer-management",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The paper targets CPU B+tree storage engines and cautions against direct transfer to GPU-resident WAL/MVCC execution.",
    },
    (
        "2026-06-02-virtual-memory-assisted-buffer-management",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Versioned eviction validation is presented as an alternative to hazard-pointer or epoch-style reclamation.",
    },
    (
        "2026-06-02-robust-plan-evaluation-based-on-approximate-probabilistic-machine-learning",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The evidence identifies estimation risk from learned cost-model limitations, cautioning against unbounded learned advice.",
    },
    (
        "2026-06-02-robust-plan-evaluation-based-on-approximate-probabilistic-machine-learning",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Risk pruning and estimation uncertainty warn that ordinary route selection must account for model confidence.",
    },
    (
        "2026-06-02-robust-plan-evaluation-based-on-approximate-probabilistic-machine-learning",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The independence and normality assumptions may misestimate GPU queue and transfer contention for hot-write routes.",
    },
    (
        "2026-06-02-robust-plan-evaluation-based-on-approximate-probabilistic-machine-learning",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Roq is evaluated for query optimization rather than a GPU-aware transactional engine with WAL, MVCC, residency, and admission.",
    },
    (
        "2026-06-02-read-safe-snapshots-for-abort-wait-free-serializable-reads",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained snapshot reclamation is valid only if long reads do not force owner-queue waits or unlabeled stale reads.",
    },
    (
        "2026-06-02-second-modern-batch-synthesis",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns against binary GPU-if-resident placement without route risk, freshness, and fallback telemetry.",
    },
    (
        "2026-06-02-second-modern-batch-synthesis",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that snapshot and route state need explicit telemetry before retained metadata is trusted.",
    },
    (
        "2026-06-02-second-modern-batch-synthesis",
        "htap_freshness_router",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Freshness routing must avoid binary acceleration and account for uncertainty, fault state, and serializability class.",
    },
    (
        "2026-06-02-second-modern-batch-synthesis",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The evidence supports observable immutable read handles with explicit tier, snapshot, generation, and invalidation state.",
    },
    (
        "2026-06-02-second-modern-batch-synthesis",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns retained GPU reads must be chosen by freshness, tail risk, and overload state rather than residency alone.",
    },
    (
        "2026-06-02-virtual-memory-assisted-buffer-management-in-tiered-memory",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier planning needs proof that readers see only valid generations while maintenance moves other segments.",
    },
    (
        "2026-06-02-virtual-memory-assisted-buffer-management-in-tiered-memory",
        "stable_handle_indirection",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Stable handles need a proof gate for valid-generation reads during concurrent tier maintenance.",
    },
    (
        "2026-06-02-virtual-memory-assisted-buffer-management-in-tiered-memory",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "contradicts",
        "relation_review_note": "The single-copy invariant conflicts with cache-as-acceleration correctness ownership unless limited to logical placement handles.",
    },
    (
        "2026-06-02-parqo-penalty-aware-robust-plan-selection",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Penalty-aware route choices require benchmarks for skewed predicates, residency misses, saturated GPU queues, and fallback.",
    },
    (
        "2026-06-02-parqo-penalty-aware-robust-plan-selection",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback adoption is gated on tests where transfer and launch overhead beat a nominally resident GPU route.",
    },
    (
        "2026-06-02-parqo-penalty-aware-robust-plan-selection",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The KL-divergence test is a route-reuse mechanism, so this false-positive test cue remains supporting placement evidence.",
    },
    (
        "2026-06-02-parqo-penalty-aware-robust-plan-selection",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained GPU route use is gated on measuring when queue pressure, refresh, or bad estimates make fallback better.",
    },
    (
        "2026-06-02-third-modern-batch-synthesis",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis requires a route-decision record and CPU-vs-retained-GPU policy comparison before adoption.",
    },
    (
        "2026-06-02-third-modern-batch-synthesis",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fallback policy is explicitly routed through route-decision records and a CPU-vs-retained-GPU comparison harness.",
    },
    (
        "2026-06-02-third-modern-batch-synthesis",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-root adoption is gated on recorded snapshot, residency, tier, queue, transfer, and fallback facts.",
    },
    (
        "2026-06-02-third-modern-batch-synthesis",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The risk wording names route-risk metadata, while the retained evidence supports measured owner-domain request paths.",
    },
    (
        "2026-06-03-tictoc-data-driven-timestamp-occ",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Timestamp-history and validation checks need GPU DB visibility-frontier measurements before transfer.",
    },
    (
        "2026-06-03-tictoc-data-driven-timestamp-occ",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Data-driven timestamps support visibility only if WAL, CPU indexes, GPU generations, publication, and replay agree.",
    },
    (
        "2026-06-03-tictoc-data-driven-timestamp-occ",
        "isolation_trace_oracle",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evidence says evaluated workloads showed no measurable gain, so trace-oracle transfer needs explicit evaluation.",
    },
    (
        "2026-06-03-shirakami-hybrid-long-transaction-mvcc-and-short-transaction-occ",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The entry directly calls for a CPU-only prototype and conflict, latency, retry, and queue metrics.",
    },
    (
        "2026-06-03-memory-optimized-mvcc-for-disk-backed-storage",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Page-reference epochs and read repetition are presented as a different visibility-tracking shape than frontier vectors.",
    },
    (
        "2026-06-03-memory-optimized-mvcc-for-disk-backed-storage",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "ARIES-style rollback and rebuildable MVCC state require WAL replay and bulk-operation validation before GPU DB transfer.",
    },
    (
        "2026-06-03-memory-optimized-mvcc-for-disk-backed-storage",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Rebuildable CPU-side MVCC auxiliaries are a structural alternative to durable retained GPU snapshot state.",
    },
    (
        "2026-06-03-fourth-modern-batch-synthesis",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis explicitly requires measuring logical-clock choices for WAL-before-visibility publication.",
    },
    (
        "2026-06-03-low-latency-transaction-scheduling-via-userspace-interrupts",
        "deficit_fairness",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Pausing and resuming urgent work is presented as an alternative fairness mechanism to aborting long transactions.",
    },
    (
        "2026-06-03-resource-adaptive-query-execution-with-paged-memory-management",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The paper omits GPU memory, pinned memory, NVMe tiering, WAL/MVCC, and million-session admission evaluation.",
    },
    (
        "2026-06-03-resource-adaptive-query-execution-with-paged-memory-management",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Resizable buffer-pool pages are an alternative state-management shape to serialized heap descriptor objects.",
    },
    (
        "2026-06-03-resource-adaptive-query-execution-with-paged-memory-management",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The preliminary design/exploration status cautions against treating it as mature hot-write template evidence.",
    },
    (
        "2026-06-03-resource-adaptive-query-execution-with-paged-memory-management",
        "vector_credit_admission",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue contrasts invisible heap growth while supporting explicit observable memory admission and backpressure.",
    },
    (
        "2026-06-03-polaris-priority-aware-optimistic-concurrency-control",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Priority-aware conflict ordering is gated on throughput, tail latency, abort distribution, and starvation measurements.",
    },
    (
        "2026-06-03-polaris-priority-aware-optimistic-concurrency-control",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The paper omits durable WAL flush, checkpoint, recovery, GPU execution, protocol-state, and session-scale evaluation.",
    },
    (
        "2026-06-03-polaris-priority-aware-optimistic-concurrency-control",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Session-counting transfer needs benchmarks because the evaluated system omits PostgreSQL protocol state and million-session admission.",
    },
    (
        "2026-06-03-fifth-modern-batch-synthesis",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis names tier and resource-admission measurements as the gate for placement choices.",
    },
    (
        "2026-06-03-fifth-modern-batch-synthesis",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The class-aware work contrast supports owner-local queues over a single global queue.",
    },
    (
        "2026-06-03-fifth-modern-batch-synthesis",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Priority conflict metadata is contrasted with retry luck and supports explicit conflict-ordering policy.",
    },
    (
        "2026-06-03-scalable-garbage-collection-for-in-memory-mvcc",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue describes GC pruning policy, not an optimizer alternative, and remains route-cost support.",
    },
    (
        "2026-06-03-par2qo-parametric-penalty-aware-robust-query-optimization",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Penalty-profile plan caching is an optimizer-side alternative to specializing a single parameterized template path.",
    },
    (
        "2026-06-03-par2qo-parametric-penalty-aware-robust-query-optimization",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The retained evidence explicitly stresses GPU resident execution versus CPU fallback decisions.",
    },
    (
        "2026-06-03-sixth-modern-batch-synthesis",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis requires measuring recovery-bound route metadata and penalty-aware route choice before adoption.",
    },
    (
        "2026-06-03-sixth-modern-batch-synthesis",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The LeanStore, Steam, and PAR2QO synthesis names MVCC cleanup as a next test target.",
    },
    (
        "2026-06-03-sixth-modern-batch-synthesis",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshots are gated on recovery-to-first-GPU-route and long-snapshot cleanup measurements.",
    },
    (
        "2026-06-03-sixth-modern-batch-synthesis",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis routes durable write-path authority through WAL/recovery tests before architectural adoption.",
    },
    (
        "2026-06-03-sixth-modern-batch-synthesis",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-local GC and route planning require queue, cleanup-debt, retained-read, and write-tail measurements.",
    },
    (
        "2026-06-03-pasha-partitioned-shared-cxl-pod-architecture",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The entry names movement-policy measurements before shared-tier placement can be trusted.",
    },
    (
        "2026-06-03-pasha-partitioned-shared-cxl-pod-architecture",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Parallel logging and checkpoint support is called out as future work that must be measured before WAL visibility transfer.",
    },
    (
        "2026-06-03-pasha-partitioned-shared-cxl-pod-architecture",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The partitioner minimizes shared-region operations rather than solving descriptor lifetime directly.",
    },
    (
        "2026-06-03-owner-local-first-shared-only-when-measured",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Shared or accelerated tiers are admitted only when descriptor generation, bytes, and conflict class are proven.",
    },
    (
        "2026-06-03-owner-local-first-shared-only-when-measured",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Route-root publication is supported only when the route descriptor proves generation, bytes, and conflict class.",
    },
    (
        "2026-06-03-owner-local-first-shared-only-when-measured",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "GPU OLTP conflict routing is valid only when owner-local state can prove the conflict class before admission.",
    },
    (
        "2026-06-03-p-tree-multi-versioned-indexes-for-htap-snapshots",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue contrasts independent updates while the evidence supports publishing one immutable root boundary.",
    },
    (
        "2026-06-03-p-tree-multi-versioned-indexes-for-htap-snapshots",
        "wal_before_visibility",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The bounded write-batch evidence keeps WAL and visibility publication together, so this remains supporting evidence.",
    },
    (
        "2026-06-03-p-tree-multi-versioned-indexes-for-htap-snapshots",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue rejects unbounded cleanup and directly supports explicit snapshot-release reclamation frontiers.",
    },
    (
        "2026-06-03-runtime-conflict-transaction-scheduling",
        "cpu_fallback_policy",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Choosing CPU fallback instead of a likely failing GPU route is the intended fallback mechanism.",
    },
    (
        "2026-06-03-semantic-occ-batching-and-operation-reordering",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "A small owner-local conflict graph supports owner bundling rather than a global transaction scheduler.",
    },
    (
        "2026-06-03-nomad-non-exclusive-memory-tiering",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained reads avoid blocking only if the query does not require a fresh resident generation.",
    },
    (
        "2026-06-03-detox-transactional-cache-hit-rate",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The risks section says the cache setting lacks SQL, MVCC, WAL replay, GPU kernels, and multi-tier resident placement.",
    },
    (
        "2026-06-03-themis-gpu-relational-pipeline-load-balancing",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Same-shape micro-batches require HBM, transfer, kernel, response, and null-result measurements before adoption.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-and-scheduling-need-request-shaped-metrics",
        "db_owned_cold_objects",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis requires request-shaped cache-value measurements before cold-object ownership choices are trusted.",
    },
    (
        "2026-06-03-mordred-semantic-cpu-gpu-placement",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Residency transfer is gated on HBM, PCIe, CPU materialization, response-byte, and correctness measurements.",
    },
    (
        "2026-06-03-morty-transaction-re-execution",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Early uncommitted visibility is safe only with precise validation, dirty-read checks, cleanup, WAL ordering, and recovery.",
    },
    (
        "2026-06-03-morty-transaction-re-execution",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Partial re-execution versus deterministic owner ordering is explicitly framed as a concurrency benchmark.",
    },
    (
        "2026-06-03-morty-transaction-re-execution",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Morty's replicated key-value scope cautions against direct descriptor-reclamation transfer to PostgreSQL-compatible SQL.",
    },
    (
        "2026-06-03-loger-restricted-learned-query-optimization",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Learned join-order advice is valid only while the DBMS optimizer still owns physical operator restrictions.",
    },
    (
        "2026-06-03-loger-restricted-learned-query-optimization",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue is about optimizer scope; the retained evidence still supports keeping bounded operator knowledge explicit.",
    },
    (
        "2026-06-03-loger-restricted-learned-query-optimization",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Route learning is acceptable only inside explicitly bounded safe alternatives with stable route-template evidence.",
    },
    (
        "2026-06-03-loger-restricted-learned-query-optimization",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "The learned route transfer is useful only when decisions are cached by template, snapshot class, and tier state.",
    },
    (
        "2026-06-03-cross-paper-synthesis-learned-advice-needs-hard-route-boundaries",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that learned advice can choose risky CPU, cold-transfer, or hot-fragment routes without hard boundaries.",
    },
    (
        "2026-06-03-cross-paper-synthesis-learned-advice-needs-hard-route-boundaries",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Route descriptors are useful only with a restriction language that names allowed and disallowed route behavior.",
    },
    (
        "2026-06-03-cross-paper-synthesis-learned-advice-needs-hard-route-boundaries",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Hot-write template routing is valid only when restrictions define what the planner may not do for the request.",
    },
    (
        "2026-06-03-cross-paper-synthesis-learned-advice-needs-hard-route-boundaries",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that learned advice may choose cold transfers that are too risky under current pressure.",
    },
    (
        "2026-06-03-cross-paper-synthesis-learned-advice-needs-hard-route-boundaries",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The descriptor explicitly names owner domain, queue budget, skew risk, and dependency fragments, supporting owner bundling.",
    },
    (
        "2026-06-03-shinjuku-microsecond-scale-preemptive-scheduling",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained routes are safe only when their service-time distribution is compatible with the queue class.",
    },
    (
        "2026-06-03-path-to-gpu-initiated-i-o-for-data-intensive-systems",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Promotion to CPU or GPU cache is worthwhile only when reuse repays the tier-resource cost.",
    },
    (
        "2026-06-03-path-to-gpu-initiated-i-o-for-data-intensive-systems",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Random-read microbenchmarks do not prove SQL, MVCC, WAL, response-ring, or session-scale safety.",
    },
    (
        "2026-06-03-path-to-gpu-initiated-i-o-for-data-intensive-systems",
        "immutable_route_roots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The accessible evidence is only a publication record and slides, so route-root transfer needs caution.",
    },
    (
        "2026-06-03-aria-deterministic-oltp-batches",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Aria's conflict-class reordering is an alternative to treating every hot overlap as abort or owner routing.",
    },
    (
        "2026-06-03-aria-deterministic-oltp-batches",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner routing is explicitly gated on conflict-rate crossover, tail latency, queue wait, and abort measurements.",
    },
    (
        "2026-06-03-aria-deterministic-oltp-batches",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Aria probes read/write reservation metadata rather than building one global serial dependency witness graph.",
    },
    (
        "2026-06-03-empirical-in-memory-mvcc-design-tradeoffs",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Generation-root publication is proposed as prototype work with CPU build, cache, GPU-byte, and latency metrics.",
    },
    (
        "2026-06-03-chiller-contention-centric-transaction-partitioning",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The entry requires a hot-record benchmark comparing owner-order routing against optimistic retry.",
    },
    (
        "2026-06-03-chiller-contention-centric-transaction-partitioning",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Chiller is 2PL-centered and does not directly solve MVCC snapshot or GPU-resident visibility correctness.",
    },
    (
        "2026-06-03-memtis-access-distribution-memory-tiering",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "MEMTIS is OS memory tiering, not DBMS-owned WAL, MVCC, or GPU-resident snapshot management.",
    },
    (
        "2026-06-03-cross-paper-synthesis-visibility-contention-and-placement-need-distribution-summaries",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis names distribution-aware placement, anti-thrash behavior, and sub-segment benchmarks as required gates.",
    },
    (
        "2026-06-03-cross-paper-synthesis-visibility-contention-and-placement-need-distribution-summaries",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Admission policy is routed through measured chain length, conflict heat, placement, and session-memory probes.",
    },
    (
        "2026-06-03-tas-tcp-acceleration-as-an-os-service",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evaluated 64K-connection scale is below the 1M logical-session target and needs session-scale validation.",
    },
    (
        "2026-06-03-hint-qpt-hints-for-robust-query-performance-tuning",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fallback reasons and route-fragility handling are framed as a proof gate for bad-choice reduction.",
    },
    (
        "2026-06-03-hint-qpt-hints-for-robust-query-performance-tuning",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Bad cardinality, byte, queue, response, or refresh estimates can turn retained GPU routes into slow fallbacks.",
    },
    (
        "2026-06-03-hint-qpt-hints-for-robust-query-performance-tuning",
        "owner_ring_bundling",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Route-risk evidence warns that owner routing needs estimate and queue-risk constraints before adoption.",
    },
    (
        "2026-06-03-hint-qpt-hints-for-robust-query-performance-tuning",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Hint-QPT is an interactive tuning demonstration, not production descriptor-lifetime evidence.",
    },
    (
        "2026-06-03-taurus-lightweight-parallel-logging",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Dependency-vector transfer is gated on throughput, commit wait, fsync bytes, recovery time, and replay proof.",
    },
    (
        "2026-06-03-taurus-lightweight-parallel-logging",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Command logging is cleanest only under deterministic stored-procedure replay, narrower than ad hoc SQL.",
    },
    (
        "2026-06-03-bounded-delay-multiversion-concurrency-and-precise-gc",
        "effective_session_counting",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Session counting is valid only if retention follows active holders, not idle connection count.",
    },
    (
        "2026-06-03-cross-paper-synthesis-roots-frontiers-and-active-holders",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The synthesis redirects next work toward production MVCC GC and dual-snapshot HTAP rather than GPU-OLAP pipelines.",
    },
    (
        "2026-06-03-cross-paper-synthesis-roots-frontiers-and-active-holders",
        "wal_before_visibility",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The evidence favors small route, queue, and dependency tokens over broad mutable WAL-related state movement.",
    },
    (
        "2026-06-03-ankerdb-fine-granular-virtual-snapshotting",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot frontiers need proof that old generations retire when the last holder releases.",
    },
    (
        "2026-06-03-ankerdb-fine-granular-virtual-snapshotting",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "VM snapshot microbenchmarks motivate but do not replace GPU DB old-version GC measurements.",
    },
    (
        "2026-06-03-ankerdb-fine-granular-virtual-snapshotting",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Column-generation route roots are explicitly proposed as a CPU-side prototype before adoption.",
    },
    (
        "2026-06-03-ankerdb-fine-granular-virtual-snapshotting",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Generation retirement after holder release is an alternative to scanning every row-version descriptor.",
    },
    (
        "2026-06-03-ankerdb-fine-granular-virtual-snapshotting",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Refresh-cost versus route-freshness behavior is named as the deciding benchmark.",
    },
    (
        "2026-06-03-lero-learning-to-rank-query-optimization",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Learning-to-rank transfer requires route-choice evaluation under cardinality error and bounded candidate growth.",
    },
    (
        "2026-06-03-lero-learning-to-rank-query-optimization",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Reported optimizer gains still require GPU DB benchmark validation before learned advice is trusted.",
    },
    (
        "2026-06-03-lero-learning-to-rank-query-optimization",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "A retained GPU route is fast only if generation validity, queue capacity, holders, and memory budgets hold.",
    },
    (
        "2026-06-03-lero-learning-to-rank-query-optimization",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Tier placement advice is useful only if resident validity, GPU capacity, snapshot holders, and budgets hold.",
    },
    (
        "2026-06-03-plor-predictable-low-tail-transactions",
        "effective_session_counting",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Worker-count-oriented lock metadata cautions that logical sessions must be multiplexed through bounded workers.",
    },
    (
        "2026-06-03-cross-paper-synthesis-admission-needs-explicit-winners",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Explicit commit-priority selection is presented as an alternative to letting abort/retry behavior decide tail latency.",
    },
    (
        "2026-06-03-cross-paper-synthesis-admission-needs-explicit-winners",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "The synthesis keeps tiered placement as future work unless new MVCC or write-path evidence changes the priority.",
    },
    (
        "2026-06-03-cross-paper-synthesis-admission-needs-explicit-winners",
        "deficit_fairness",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Conflict-priority admission is framed as an explicit winner policy rather than fairness emerging from repeated aborts.",
    },
    (
        "2026-06-03-mmap-is-not-a-buffer-pool-substitute",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Mapped immutable-file routing needs WAL ordering, checksum, mutation invalidation, and stale-byte tests before adoption.",
    },
    (
        "2026-06-03-mmap-is-not-a-buffer-pool-substitute",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Resident GPU inputs are valid only when physical residency and fault behavior are proven, not merely addressable.",
    },
    (
        "2026-06-03-tpp-transparent-cxl-page-placement",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Transparent placement must not admit retained routes unless WAL, visibility, generation, tier, latency, and migration risk are provable.",
    },
    (
        "2026-06-03-zygos-work-conserving-microsecond-scheduler",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The ZygOS transfer is explicitly gated on retained-read micro-batch latency, queue-wait, idle-worker, and ordering measurements.",
    },
    (
        "2026-06-03-zygos-work-conserving-microsecond-scheduler",
        "resource_dag_scheduling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than clause limits throughput extrapolation; the evidence still supports scheduling as a resource contract.",
    },
    (
        "2026-06-03-gacco-gpu-accelerated-oltp-co-execution",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Same-type GPU OLTP lanes are valid only under stored-procedure assumptions and static type-to-device routing.",
    },
    (
        "2026-06-03-gacco-gpu-accelerated-oltp-co-execution",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The OLTP-rather-than-analytics contrast is scope context; bounded per-template queues still support descriptor shaping.",
    },
    (
        "2026-06-03-cross-paper-synthesis-batch-lanes-need-visibility-fences",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring batching is gated on same-template lane, visibility-fenced write-batch, and tier-aware route admission tests.",
    },
    (
        "2026-06-03-bam-gpu-initiated-storage-access",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "GPU-side cold-tier request lanes are presented as an alternative to hiding IO behind CPU page faults.",
    },
    (
        "2026-06-03-bam-gpu-initiated-storage-access",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "GPU-initiated NVMe queues are an alternative to CPU tiling, page-fault service, and repeated copy/compute phases.",
    },
    (
        "2026-06-03-paramtree-learned-cost-model-calibration",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Parameterized cost transfer is useful only after GPU DB has explicit formula terms for validity, residency, queueing, and transfer cost.",
    },
    (
        "2026-06-03-paramtree-learned-cost-model-calibration",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner queue wait, fallback penalty, invalidation risk, and transfer costs must be measured before owner routing decisions rely on the model.",
    },
    (
        "2026-06-03-arachne-core-aware-thread-management",
        "effective_session_counting",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Session multiplexing is useful only if tail-latency gains do not starve write visibility, refresh, or response delivery.",
    },
    (
        "2026-06-03-arachne-core-aware-thread-management",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Tier work is valid in cooperative scheduling only when long scans, page faults, fallback joins, and cold reads are isolated or preemptible.",
    },
    (
        "2026-06-03-arachne-core-aware-thread-management",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "CPU fallback is compatible with cooperative workers only when blocking or long fallback work is isolated or made preemptible.",
    },
    (
        "2026-06-03-arachne-core-aware-thread-management",
        "resource_dag_scheduling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Arachne-style resource exposure is presented as an alternative to one OS thread per client or one generic work queue.",
    },
    (
        "2026-06-03-rebirth-retire-adaptive-contention-control",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hot-write ordering needs a benchmark comparing first-writer-wins, abort/retry, deterministic batch order, and rebirth-style demotion.",
    },
    (
        "2026-06-03-rebirth-retire-adaptive-contention-control",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The conflicts-with cue names transaction conflicts; retired-owner metadata and latch-free dependency tracking still support bounded descriptor state.",
    },
    (
        "2026-06-03-rebirth-retire-adaptive-contention-control",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Bounded rebirth checks with fallback are presented as an alternative to unbounded dependency graph work.",
    },
    (
        "2026-06-03-cross-paper-synthesis-control-planes-should-stay-explicit",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Dependency witnesses are gated on control-plane measurements for owner generations, conflict metadata, storage transfer, and fallback controls.",
    },
    (
        "2026-06-03-umbra-variable-size-pages-for-ssd-backed-hot-working-sets",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner handoff must be tested under IO pressure, GPU saturation, pause behavior, and transient result-lifetime correctness.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-paths-need-fast-handles-and-slow-path-regulators",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier placement depends on measurements for cold-state pressure, stale generations, HBM cooling, host spill, and retained-route validation cost.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-paths-need-fast-handles-and-slow-path-regulators",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Owner-published generation handles directly support immutable route roots despite the rather-than cue.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-paths-need-fast-handles-and-slow-path-regulators",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The evidence supports bounded descriptor lifetimes by separating hot handles from cold cleanup and transient state.",
    },
    (
        "2026-06-03-oltp-through-the-looking-glass-16-years-later",
        "isolation_trace_oracle",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Whole-stack OLTP isolation evidence needs GPU DB validation against stored procedures, client logic, and PostgreSQL-style baselines.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-devices-require-explicit-service-ownership",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fast-device placement requires measured service-owned buffers, saturation counters, session backpressure, and async cold-tier behavior.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-devices-require-explicit-service-ownership",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring service ownership is gated on measured IO-worker multiplexing, response-ring backpressure, and saturation counters.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-devices-require-explicit-service-ownership",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fast-path handles are safe only when buffers carry snapshot, WAL, visibility, and generation metadata.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-devices-require-explicit-service-ownership",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Session counting requires measured multiplexing and backpressure under thousands of logical sessions before fast-device adoption.",
    },
    (
        "2026-06-03-autonomous-commit-for-low-latency-nvme-durability",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshot transfer is explicitly gated on WAL replay equivalence and bursty COPY admission measurements.",
    },
    (
        "2026-06-03-modern-nvme-storage-engine-exploitation",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "NVMe tier placement needs measured queue-depth saturation before combining storage with CUDA and resident-generation safety.",
    },
    (
        "2026-06-03-modern-nvme-storage-engine-exploitation",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The storage-engine evidence cannot support WAL visibility without durable-write, publication, and replay measurements.",
    },
    (
        "2026-06-03-modern-nvme-storage-engine-exploitation",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The evaluation disables logging and weakens isolation, warning against direct durable GPU OLTP conflict-ordering transfer.",
    },
    (
        "2026-06-03-modern-nvme-storage-engine-exploitation",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Page or segment fetch routes are valid only when owned buffers publish completion for still-valid generations.",
    },
    (
        "2026-06-03-cross-paper-synthesis-generations-need-durable-and-logical-fronts",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier placement is routed through a prototype generation-frontier timeline and sealed-descriptor measurements.",
    },
    (
        "2026-06-03-cross-paper-synthesis-generations-need-durable-and-logical-fronts",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshots need prototype generation-frontier traces spanning execution, durability, invalidation, residency, and response.",
    },
    (
        "2026-06-03-mosaicdb-multi-source-latency-hiding",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Cold and future CXL tier queues must be sized by measured bandwidth, IOPS, pinned-buffer budget, and stale-generation risk.",
    },
    (
        "2026-06-03-mosaicdb-multi-source-latency-hiding",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "MosaicDB motivates owner-ring runtime shape, but the transfer still needs benchmarking against the current endpoint.",
    },
    (
        "2026-06-03-tesseract-online-schema-evolution",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Schema-frontier snapshot transfer needs tests for overlapped CDC, relaxed snapshots, and pending-schema routing.",
    },
    (
        "2026-06-03-tesseract-online-schema-evolution",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "New route generations are valid only when complete enough for the requested shape, otherwise reads wait or fall back.",
    },
    (
        "2026-06-03-tesseract-online-schema-evolution",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained GPU snapshots are valid only when visibility, schema generation, layout, predicates, and response shape agree.",
    },
    (
        "2026-06-03-bonspiel-low-tail-geo-distributed-transactions",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Bonspiel omits GPU execution, PostgreSQL serving, NVMe tiering, MVCC storage, and local durable WAL measurements.",
    },
    (
        "2026-06-03-carpo-listwise-context-aware-query-plan-ranking",
        "cpu_fallback_policy",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Fallback routing must account for stale-generation risk and fallback reasons rather than trusting ranked routes alone.",
    },
    (
        "2026-06-03-webridge-synthesized-stored-procedures-for-hot-paths",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Synthesized hot paths require hot-key update/read benchmarks against ordinary statement execution before adoption.",
    },
    (
        "2026-06-03-webridge-synthesized-stored-procedures-for-hot-paths",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Stored-procedure route roots are valid only when conditional branches and writes publish at explicit durable and visible frontiers.",
    },
    (
        "2026-06-03-gcctb-gpu-oltp-concurrency-control-study",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Per-batch access tables and deterministic GPU conflict order are presented as benchmarkable alternatives for hot rows.",
    },
    (
        "2026-06-03-gcctb-gpu-oltp-concurrency-control-study",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Resident GPU state in the testbed still needs DB-scale retained-snapshot and generated-code configuration measurements.",
    },
    (
        "2026-06-03-gcctb-gpu-oltp-concurrency-control-study",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Write-batch protocol choice should be selected by measured conflict shape rather than assumed isolation preference.",
    },
    (
        "2026-06-03-cross-paper-synthesis-gpu-writes-need-classed-conflict-lanes",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Classed write templates are explicitly framed as a lab for measuring optimistic versus deterministic preprocessing costs.",
    },
    (
        "2026-06-03-cross-paper-synthesis-gpu-writes-need-classed-conflict-lanes",
        "isolation_trace_oracle",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Isolation trace transfer is gated on event traces that prove isolation and WAL publication for measured write batches.",
    },
    (
        "2026-06-03-databases-on-modern-networks",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Future CXL, remote memory, and remote GPU tiers warn against treating local HBM/DRAM/NVMe placement as sufficient.",
    },
    (
        "2026-06-03-databases-on-modern-networks",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Offloaded paths are valid only when WAL, MVCC, catalog invalidation, and recovery state remain database-owned or fully proved.",
    },
    (
        "2026-06-03-databases-on-modern-networks",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Networked memory movement warns against assuming retained snapshots and cold partitions remain local-placement problems.",
    },
    (
        "2026-06-03-databases-on-modern-networks",
        "stable_handle_indirection",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Zero-copy NIC reads from userspace buffers are presented as an alternative state-movement shape to local stable handles.",
    },
    (
        "2026-06-03-skyloft-user-space-preemptive-scheduling",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Scheduler and key-value results do not cover SQL, MVCC, WAL durability, PostgreSQL protocol, GPU kernels, or CUDA scheduling.",
    },
    (
        "2026-06-03-dbms-owned-large-objects-instead-of-files",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Large-object tiering transfer depends on measured object throughput and metadata scans rather than filesystem-style assumptions.",
    },
    (
        "2026-06-03-dbms-owned-large-objects-instead-of-files",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-time and queue-wait benefits need DBMS metadata-path measurements before owner bundling can rely on large-object catalog routing.",
    },
    (
        "2026-06-03-scalerpc-reliable-connection-resource-sharing",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Reliable-connection sharing must be measured with logical, admitted, request-slot, response-slot, and staging-slot counters.",
    },
    (
        "2026-06-03-scalerpc-reliable-connection-resource-sharing",
        "deficit_fairness",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fairness transfer requires p50/p99, queue-wait, starvation, and mixed hot/idle client measurements.",
    },
    (
        "2026-06-03-cross-paper-synthesis-logical-scale-needs-active-resource-budgets",
        "deficit_fairness",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Read-priority lanes and virtualized slots need mixed read/write/cold-miss stress measurements before fairness policy adoption.",
    },
    (
        "2026-06-03-cross-paper-synthesis-logical-scale-needs-active-resource-budgets",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner bundling is gated on runtime reports that separate logical sessions, active queues, buffers, snapshots, and dirty backlog.",
    },
    (
        "2026-06-03-cross-paper-synthesis-logical-scale-needs-active-resource-budgets",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot frontier vectors require stress reports proving bounded snapshot and descriptor counts under active resource budgets.",
    },
    (
        "2026-06-03-cross-paper-synthesis-logical-scale-needs-active-resource-budgets",
        "resource_dag_scheduling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis explicitly calls for evaluation of scheduling, storage ownership, routing, and MVCC signals together under SQL or HTAP workloads.",
    },
    (
        "2026-06-03-natto-distributed-transaction-prioritization",
        "deficit_fairness",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Priority abort supports fairness only when retry-age promotion, budget caps, or fairness windows prevent low-priority starvation.",
    },
    (
        "2026-06-03-natto-distributed-transaction-prioritization",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "contradicts",
        "relation_review_note": "Late high-priority transactions can abort already queued smaller-timestamp work, conflicting with deterministic hot-write ordering.",
    },
    (
        "2026-06-03-natto-distributed-transaction-prioritization",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Natto's two priority levels are too narrow for GPU DB freshness classes without class-count and latency evaluation.",
    },
    (
        "2026-06-03-natto-distributed-transaction-prioritization",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained read preparation is valid only when publication remains conditional on mutation abort, completion, invalidation, or WAL-visible generation.",
    },
    (
        "2026-06-03-hybridgc-production-mvcc-garbage-collection-in-sap-hana",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "HybridGC motivates WAL replay correctness, but commit-path overhead and GC scan work must be measured in the MVCC harness.",
    },
    (
        "2026-06-03-dana-directly-attached-nvme-arrays",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than clause rejects treating NVMe as resident OLTP memory; the retained evidence still supports explicit tiering mechanics.",
    },
    (
        "2026-06-03-dana-directly-attached-nvme-arrays",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "DANA's transferable queue mechanics support owner-ring bundling despite the resident-memory contrast.",
    },
    (
        "2026-06-03-dana-directly-attached-nvme-arrays",
        "log_structured_warm_tier",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The not-just cue describes richer NVMe-tier behavior, not a warning against log-structured warm-tier design.",
    },
    (
        "2026-06-03-dana-directly-attached-nvme-arrays",
        "htap_freshness_router",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The paper's scan-oriented OLAP/HTAP focus is an alternative workload shape to pure retained OLTP freshness routing.",
    },
    (
        "2026-06-03-cross-paper-synthesis-scoped-fronts-must-include-storage",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue broadens robust route choice beyond cardinality error and supports resource-aware cost routing.",
    },
    (
        "2026-06-03-cross-paper-synthesis-scoped-fronts-must-include-storage",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Conditional fronts are valid only when prepared work cannot publish unsafe visibility.",
    },
    (
        "2026-06-03-bf-tree-variable-length-mini-pages-for-larger-than-memory-indexes",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Variable-size cache objects need concurrency tests for growth, shrink, eviction, and retirement under retained snapshots.",
    },
    (
        "2026-06-03-bf-tree-variable-length-mini-pages-for-larger-than-memory-indexes",
        "db_owned_cold_objects",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "LSM-style and delta-chain cold-object layouts are alternatives with different read, scan, and compaction costs.",
    },
    (
        "2026-06-03-ermia-snapshot-friendly-mixed-workload-oltp",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Resident GPU snapshots should be evaluated against ERMIA-style indirection to prove the extra hop is flattened before kernel execution.",
    },
    (
        "2026-06-03-zicio-db-os-prefetch-for-rapid-ingestion",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue directly supports owner-published timing descriptors in bounded shared rings.",
    },
    (
        "2026-06-03-predicate-transfer-for-multi-join-pre-filtering",
        "same_shape_microbatching",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue motivates joining an existing same-shape filter-build phase instead of duplicating per-session work.",
    },
    (
        "2026-06-03-query-fresh-synchronous-log-shipping-with-fresh-replicas",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Cheap indirection and generation updates after durable batches directly support immutable route-root publication.",
    },
    (
        "2026-06-03-query-fresh-synchronous-log-shipping-with-fresh-replicas",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue supports bounded generation updates rather than repeated secondary-structure rematerialization.",
    },
    (
        "2026-06-03-concord-approximate-optimal-scheduling-for-microsecond-tails",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Bounded local owner queues are valid only when queue depth remains part of the tail-latency contract.",
    },
    (
        "2026-06-03-carousel-time-indexed-shaping-for-bounded-session-admission",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The not-just cue supports ownership and bounded queued work as required complements to scalable admission.",
    },
    (
        "2026-06-03-carousel-time-indexed-shaping-for-bounded-session-admission",
        "vector_credit_admission",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Credit admission is valid only with bounded queued work, producer backpressure, and ownership that avoids shared hot locks.",
    },
    (
        "2026-06-03-carousel-time-indexed-shaping-for-bounded-session-admission",
        "deficit_fairness",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Time-slot granularity and horizon choices need p50 and fairness tests across tiny lookups and large result sets.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-placement-still-needs-paced-fronts",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Paced retained reads require overload benchmarks reporting GPU queue, response queue, socket credit, HBM residency, and cold-tier wait separately.",
    },
    (
        "2026-06-03-pacman-parallel-command-log-recovery",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The risk cue names PACMAN's dependency on deterministic templates; that dependency supports the mechanism rather than warning against it.",
    },
    (
        "2026-06-03-bounded-multiversion-garbage-collection",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained GPU generations are supported only when active handles bound reclamation and preserve correct snapshot reads.",
    },
    (
        "2026-06-03-vortex-over-resident-multi-gpu-io-forwarding",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Vortex-style over-resident execution is valid only under an explicit streaming model where GPU memory is not assumed resident.",
    },
    (
        "2026-06-03-vortex-over-resident-multi-gpu-io-forwarding",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue contrasts ad hoc copies with owned stream and transfer scheduling, which supports owner-ring bundling.",
    },
    (
        "2026-06-03-cross-paper-synthesis-snapshot-bounded-io-bounded-execution",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis explicitly routes retained snapshot pressure through long-reader, recovery, residency, and pinned-buffer measurements.",
    },
    (
        "2026-06-03-cross-paper-synthesis-snapshot-bounded-io-bounded-execution",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless cue is corpus-planning guidance, not a caution about owner-ring bundling.",
    },
    (
        "2026-06-03-learned-cost-models-need-optimizer-task-proof",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Learned cost-model transfer is gated on task-specific optimizer evaluation before it can shape GPU DB routing.",
    },
    (
        "2026-06-03-learned-cost-models-need-optimizer-task-proof",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Learned advising is supported only as a hybrid that preserves traditional optimizer estimates and task-specific metrics.",
    },
    (
        "2026-06-03-learned-cost-models-need-optimizer-task-proof",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fallback costing must be validated with measured endpoint telemetry rather than assumed transfer, launch, and queue costs.",
    },
    (
        "2026-06-03-learned-cost-models-need-optimizer-task-proof",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The fragile-route cue applies to fallback selection; the reviewed evidence does not make owner-ring bundling a caution relation.",
    },
    (
        "2026-06-03-learned-cost-models-need-optimizer-task-proof",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Optimizer-task evidence is not enough to set descriptor lifetime policy without GPU DB descriptor measurements.",
    },
    (
        "2026-06-03-learned-cost-models-need-optimizer-task-proof",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The retained route matrix is explicitly framed as a benchmark over CPU, resident GPU, streamed GPU, and rejection paths.",
    },
    (
        "2026-06-03-adaptive-multi-tier-buffer-management-for-nvm",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The NVM hierarchy motivates multi-tier placement only after GPU DB measures HBM, host, CXL-like memory, and NVMe behavior.",
    },
    (
        "2026-06-03-adaptive-multi-tier-buffer-management-for-nvm",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The paper studies CPU-visible NVM and SSD without GPU snapshot publication, MVCC visibility, WAL, or session pressure.",
    },
    (
        "2026-06-03-cross-paper-synthesis-admission-needs-tier-aware-memory-fronts",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained point reads under a small HBM budget are explicitly listed as benchmark work.",
    },
    (
        "2026-06-03-cross-paper-synthesis-admission-needs-tier-aware-memory-fronts",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fallback policy is gated on latency, drop, queue-depth, slowdown, eviction, demotion, and stale-generation measurements.",
    },
    (
        "2026-06-03-foedus-thousand-core-oltp-with-dual-pages",
        "wal_before_visibility",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts refresh paths; publication is still explicitly gated by WAL-before-visibility.",
    },
    (
        "2026-06-03-caracal-deterministic-contention-management",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Short read-snapshot batches and mutation epochs need measured ceilings before deterministic placeholders are adopted.",
    },
    (
        "2026-06-03-prismdb-multi-tier-compaction",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "PrismDB's compaction condition is not WAL-based and the entry explicitly notes that crash recovery is outside its design.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-needs-costed-generations",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Immutable generations are useful only when build, publication, and retirement costs are explicit.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-needs-costed-generations",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Descriptor reclamation is supported only when generation retirement costs remain visible and bounded.",
    },
    (
        "2026-06-03-cross-paper-synthesis-placement-needs-costed-generations",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Mutable work is fast only when it has a scoped owner and explicit costs for publication and retirement.",
    },
    (
        "2026-06-03-caerus-partial-order-transaction-sequencing",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Partial-order sequencing argues against unnecessary total ordering and offers owner-local ordering as an alternative.",
    },
    (
        "2026-06-03-caerus-partial-order-transaction-sequencing",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-local partial sequences need graph-size, SCC, queue-delay, abort, throttle, and p99 latency measurements.",
    },
    (
        "2026-06-03-caerus-partial-order-transaction-sequencing",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Owner-boundary vectors are presented as an alternative to one monolithic LSN for published snapshots.",
    },
    (
        "2026-06-03-caerus-partial-order-transaction-sequencing",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL-before-visibility must be tested against replay-equivalent visibility boundaries and owner-sequence snapshot handles.",
    },
    (
        "2026-06-03-kepler-robust-parametric-query-optimization",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Kepler-style advisor transfer requires isolated candidate execution and template-specific validation before GPU DB adoption.",
    },
    (
        "2026-06-03-bmc-safe-in-kernel-pre-stack-caching",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The XDP cache is an alternative fast-path placement shape rather than a full database-owned tiering mechanism.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-paths-need-declared-boundaries",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Typed route descriptors and undeclared-side-effect rejection need validation against injected fast-path checks.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-paths-need-declared-boundaries",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Immutable route publication needs benchmarks comparing exact-response and retained-snapshot fast paths with validation cost.",
    },
    (
        "2026-06-03-pwv-early-write-visibility",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Piece-level early visibility is explicitly framed as a narrower benchmark track before template adoption.",
    },
    (
        "2026-06-03-pwv-early-write-visibility",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Piece-level owner coordination is valid only if no read can observe pre-WAL or rollbackable state.",
    },
    (
        "2026-06-03-vessel-fast-userspace-core-scheduling",
        "resource_dag_scheduling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Vessel's global CPU-resource scheduler is an alternative to application-local DAG scheduling first.",
    },
    (
        "2026-06-03-vessel-fast-userspace-core-scheduling",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Moving CPU time among sessions, retained reads, write owners, refresh jobs, and fallback needs direct runtime measurement.",
    },
    (
        "2026-06-03-cross-paper-synthesis-declared-boundaries-need-schedulable-budgets",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless clause is corpus-planning guidance; the retained evidence still supports schedulable tier boundaries.",
    },
    (
        "2026-06-03-cross-paper-synthesis-declared-boundaries-need-schedulable-budgets",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Worker-budget movement between typed runtime rings is explicitly named as benchmark work.",
    },
    (
        "2026-06-03-cross-paper-synthesis-declared-boundaries-need-schedulable-budgets",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The synthesis favors runtime-state route ranking over black-box route production.",
    },
    (
        "2026-06-03-space-and-time-bounded-multiversion-garbage-collection",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "A cross-tier GC policy is presented as an alternative to separate ad hoc watermarks.",
    },
    (
        "2026-06-03-space-and-time-bounded-multiversion-garbage-collection",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Bounded reclamation is valid only when retained versions do not grow beyond active sparse snapshot needs.",
    },
    (
        "2026-06-03-space-and-time-bounded-multiversion-garbage-collection",
        "db_owned_cold_objects",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Range-tracking objects and restricted version lists support database-owned cold-object metadata.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-data-needs-interval-ownership",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier-intent metadata and separated resident/cold budgets are explicitly routed through benchmark comparison.",
    },
    (
        "2026-06-03-cross-paper-synthesis-hot-data-needs-interval-ownership",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Long-snapshot and skewed-update behavior must be measured before relying on retained generation placement.",
    },
    (
        "2026-06-03-leveraging-lock-contention-to-improve-oltp-application-performance",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Contention-aware reordering is valid only under dependency constraints that preserve legal unit positions.",
    },
    (
        "2026-06-03-efficient-scheduling-policies-for-microsecond-scale-tasks",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The workload is datacenter task scheduling, not SQL, WAL, MVCC, GPU kernels, or NVMe tiering.",
    },
    (
        "2026-06-03-free-join-unified-binary-and-worst-case-optimal-joins",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The paper motivates a join-route spectrum rather than a binary GPU-hash-join versus CPU-fallback decision.",
    },
    (
        "2026-06-03-free-join-unified-binary-and-worst-case-optimal-joins",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Free Join's variable-order plan form is an alternative query execution shape, not a hot-write template.",
    },
    (
        "2026-06-03-free-join-unified-binary-and-worst-case-optimal-joins",
        "cpu_fallback_policy",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The evidence argues for a route-choice spectrum instead of a binary GPU/CPU fallback split.",
    },
    (
        "2026-06-03-timely-rtt-based-congestion-control-for-the-datacenter",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "RTT and queueing signals need measurement before they can become vector admission credits.",
    },
    (
        "2026-06-03-timely-rtt-based-congestion-control-for-the-datacenter",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "TIMELY-style signals are usable only if required NIC timestamp and ACK assumptions hold for the database path.",
    },
    (
        "2026-06-03-powertcp-power-based-congestion-control",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Synthetic session-incast and boundary-dominance reporting are named as the proof gate for session counting.",
    },
    (
        "2026-06-03-cross-paper-synthesis-schedulers-need-level-and-slope",
        "resource_dag_scheduling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The synthesis presents declared-boundary scheduling choices instead of global concurrency guesses.",
    },
    (
        "2026-06-03-accelerating-gpu-data-processing-with-fastlanes-compression",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Compressed cold-partition GPU decoding must be measured against dense transfers and CPU fallback.",
    },
    (
        "2026-06-03-polyjuice-learned-concurrency-control-policies",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Polyjuice's limited workload coverage requires GPU DB conflict-shape benchmarking before transfer.",
    },
    (
        "2026-06-03-polyjuice-learned-concurrency-control-policies",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Polyjuice is in-memory multicore OLTP and does not provide GPU, WAL/recovery, or multi-tier storage guarantees.",
    },
    (
        "2026-06-03-btrim-hybrid-in-memory-row-store-for-extreme-oltp",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The hot-row lesson is explicitly that contention should be measured at the resource where it occurs.",
    },
    (
        "2026-06-03-libpreemptible-hardware-assisted-user-space-scheduling",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-specific deadline admission and chunked refresh behavior are explicit test requirements.",
    },
    (
        "2026-06-03-syrup-user-defined-scheduling-across-the-stack",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Syrup's compact route descriptor across boundaries is an alternative to isolated local scheduling decisions.",
    },
    (
        "2026-06-03-syrup-user-defined-scheduling-across-the-stack",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Passing compact descriptors across layers is an alternative to relying only on immutable route-root publication.",
    },
    (
        "2026-06-03-gmt-gpu-orchestrated-memory-tiering-for-the-big-data-era",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "GPU-side tier orchestration is SQL-visible only if CPU owners establish visibility and generation boundaries first.",
    },
    (
        "2026-06-03-gmt-gpu-orchestrated-memory-tiering-for-the-big-data-era",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Direct GPU promotion requests versus residency-owner rings need H2D, NVMe, host-memory, latency, and interference measurements.",
    },
    (
        "2026-06-03-dbos-database-oriented-operating-system-stack",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Replacing a worker table with partition-owner and route-class counters directly supports bounded owner-ring state.",
    },
    (
        "2026-06-03-dbos-database-oriented-operating-system-stack",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The VoltDB stored-procedure prototype does not directly solve GPU DB WAL, MVCC visibility, protocol, or CUDA ownership.",
    },
    (
        "2026-06-03-the-fastlanes-file-format",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Shared vector and expression descriptors across HBM and host/NVMe tiers support explicit tier placement.",
    },
    (
        "2026-06-03-the-fastlanes-file-format",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Format-level lightweight compression is an alternative storage-shape concern rather than descriptor-lifetime management.",
    },
    (
        "2026-06-03-cross-paper-synthesis-tier-aware-execution-needs-metadata-before-movement",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier miss classes and bytes-avoided accounting are explicitly framed as prototype measurements.",
    },
    (
        "2026-06-03-cross-paper-synthesis-tier-aware-execution-needs-metadata-before-movement",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback is part of the measured tier-miss matrix rather than an already-set policy.",
    },
    (
        "2026-06-03-cross-paper-synthesis-tier-aware-execution-needs-metadata-before-movement",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained route behavior needs descriptor-hit, HBM-hit, host-promotion, NVMe-fetch, and stale-generation measurements.",
    },
    (
        "2026-06-03-cross-paper-synthesis-tier-aware-execution-needs-metadata-before-movement",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route costing depends on measured bytes avoided, queue slots avoided, and tier miss classes.",
    },
    (
        "2026-06-03-mainlining-databases-supporting-fast-transactional-workloads-on-universal-columnar-data-file-for",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Hot/cold block state and fixed-size metadata updates support explicit multi-tier placement metadata.",
    },
    (
        "2026-06-03-mainlining-databases-supporting-fast-transactional-workloads-on-universal-columnar-data-file-for",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Sign-bit timestamps and before-image reconstruction need GPU DB snapshot-frontier tests before adoption.",
    },
    (
        "2026-06-03-mainlining-databases-supporting-fast-transactional-workloads-on-universal-columnar-data-file-for",
        "htap_freshness_router",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Cooling and frozen segment refresh directly supports freshness routing instead of whole-table rebuilds.",
    },
    (
        "2026-06-03-mainlining-databases-supporting-fast-transactional-workloads-on-universal-columnar-data-file-for",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Version-free frozen routes are valid only when tuple identity, visibility, refresh, eviction, and WAL replay agree.",
    },
    (
        "2026-06-03-rtcudb-ray-tracing-core-query-execution",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Specialized retained routes are valid only when side-structure build amortization beats simple CUDA retained scans.",
    },
    (
        "2026-06-03-rtcudb-ray-tracing-core-query-execution",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Ray-tracing routes need same-answer, bytes-read, and atomic-contention benchmarks before optimizer use.",
    },
    (
        "2026-06-03-rtcudb-ray-tracing-core-query-execution",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Fused BVH query execution is an alternative route shape rather than a descriptor reclamation mechanism.",
    },
    (
        "2026-06-03-rtcudb-ray-tracing-core-query-execution",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Immutable RT route publication needs measured visibility-side-structure cost against plain CUDA retained routes.",
    },
    (
        "2026-06-03-rtcudb-ray-tracing-core-query-execution",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fallback decisions require build-cost, resident-byte, latency, bandwidth, and invalidation measurements.",
    },
    (
        "2026-06-03-cross-paper-synthesis-specialized-data-paths-need-declared-shape-contracts",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Specialized route roots are valid only when descriptors prove encoding, visibility generation, side structures, wait, and fallback.",
    },
    (
        "2026-06-03-cross-paper-synthesis-specialized-data-paths-need-declared-shape-contracts",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Dependency witnesses are useful only when response feedback observes queue and tier saturation before admission.",
    },
    (
        "2026-06-03-mind-the-gap-informed-request-scheduling-at-the-nic",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "CXL-like shared memory is presented as an alternative tier interface to PCIe packet messaging alone.",
    },
    (
        "2026-06-03-mind-the-gap-informed-request-scheduling-at-the-nic",
        "resource_dag_scheduling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "NIC-informed request scheduling is an alternative scheduling layer to application-local resource DAG scheduling.",
    },
    (
        "2026-06-03-mind-the-gap-informed-request-scheduling-at-the-nic",
        "vector_credit_admission",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "NIC just-in-time packet delivery is an alternative admission signal to database-owned vector credits.",
    },
    (
        "2026-06-03-programmable-packet-scheduling-with-a-single-queue",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "AIFO resource tradeoffs and workload simulations need GPU DB admission measurements before transfer.",
    },
    (
        "2026-06-03-programmable-packet-scheduling-with-a-single-queue",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "A small FIFO ring with rank-aware admission directly supports owner-ring bundling over per-session queues.",
    },
    (
        "2026-06-03-activepointers-software-address-translation-on-gpus",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "GPU-side page-fault handling and I/O page tables are an alternative memory-management path to CPU-routed snapshots.",
    },
    (
        "2026-06-03-activepointers-software-address-translation-on-gpus",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "GPU page-cache hash tables and transfer batches need descriptor lifetime and lock-free read measurements.",
    },
    (
        "2026-06-03-activepointers-software-address-translation-on-gpus",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Publishing new generations instead of mutating active mappings directly supports immutable route roots.",
    },
    (
        "2026-06-03-cross-paper-synthesis-simple-queues-need-stable-memory-contracts",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Mixed retained and over-resident admission must report route rank, snapshot generation, memory lease, misses, and fallback.",
    },
    (
        "2026-06-03-modeling-concurrency-control-as-a-learnable-function",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Learned per-operation concurrency choices are an alternative protocol-selection concern to bounded descriptor reclamation.",
    },
    (
        "2026-06-03-sp-pifo-strict-priority-approximation-of-programmable-scheduling",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Mapping rich ranks onto a few bounded request or response rings supports owner-ring bundling.",
    },
    (
        "2026-06-03-sp-pifo-strict-priority-approximation-of-programmable-scheduling",
        "resource_dag_scheduling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "SP-PIFO's strict-priority approximation is an alternative scheduling abstraction to explicit resource-DAG execution.",
    },
    (
        "2026-06-03-cross-paper-synthesis-adaptive-scheduling-must-share-commit-generation-truth",
        "vector_credit_admission",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue contrasts blind aborts with deterministic epoch repair; the retained evidence supports bounded admission signals.",
    },
    (
        "2026-06-03-ruma-rewired-user-space-memory-access",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "RUMA-style rewired snapshots are useful only when snapshot intervals, reader attachment, and mapping lifetime stay explicit.",
    },
    (
        "2026-06-03-an-empirical-evaluation-of-columnar-storage-formats",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Analytical file-format evidence does not establish WAL, MVCC, recovery, or in-place mutation safety.",
    },
    (
        "2026-06-03-counting-is-all-you-need-for-instant-tuple-discovery",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Deterministic future-placement vectors support retained snapshot refresh rather than making this an alternative to snapshots.",
    },
    (
        "2026-06-03-counting-is-all-you-need-for-instant-tuple-discovery",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Deterministic scatter placement directly supports multi-tier placement metadata despite the rather-than cue.",
    },
    (
        "2026-06-03-cross-paper-synthesis-refresh-metadata-is-becoming-the-storage-design",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Trace-assisted refresh metadata supports placement descriptors instead of being a separate alternative mechanism.",
    },
    (
        "2026-06-03-cross-paper-synthesis-refresh-metadata-is-becoming-the-storage-design",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The not-just cue expands route costing to generation-indexed metadata rather than warning against cost-based routing.",
    },
    (
        "2026-06-03-cross-paper-synthesis-refresh-metadata-is-becoming-the-storage-design",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Generation-indexed metadata for retained reads supports retained snapshots; the not-just cue rejects byte-only storage thinking.",
    },
    (
        "2026-06-03-cross-paper-synthesis-refresh-metadata-is-becoming-the-storage-design",
        "htap_freshness_router",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Refresh and pruning metadata supports freshness routing; the not-just cue is a modeling constraint, not a caution relation.",
    },
    (
        "2026-06-03-towards-buffer-management-with-tiered-main-memory",
        "stable_handle_indirection",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Remote-memory indexing choices need GPU DB handle-lifetime and placement benchmarks before stable-handle adoption.",
    },
    (
        "2026-06-03-tiga-synchronized-clock-transaction-ordering",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Measured ring, WAL, and mutation-owner delays are needed before publication generations can drive route roots.",
    },
    (
        "2026-06-03-tiga-synchronized-clock-transaction-ordering",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Speculative execution remains valid only when replay or WAL state proves order before client visibility.",
    },
    (
        "2026-06-03-tiga-synchronized-clock-transaction-ordering",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Cross-owner dependency handling is useful only when generation tracking drives admission instead of hiding saturation.",
    },
    (
        "2026-06-03-aocc-adaptive-validation-for-heterogeneous-occ",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained aggregate reads over resident partitions need HTAP transaction-shape tests before relying on this transfer.",
    },
    (
        "2026-06-03-a-cxl-powered-database-system-opportunities-and-challenges",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Distinct CXL, DRAM, HBM, NVMe, and authority zones directly support explicit multi-tier placement.",
    },
    (
        "2026-06-03-a-cxl-powered-database-system-opportunities-and-challenges",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "CXL tiering warns against treating retained snapshot placement as simply system memory versus disk.",
    },
    (
        "2026-06-03-a-cxl-powered-database-system-opportunities-and-challenges",
        "stable_handle_indirection",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Per-partition placement descriptors support stable handle indirection across future host and device tiers.",
    },
    (
        "2026-06-03-a-cxl-powered-database-system-opportunities-and-challenges",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "GPU/CXL interaction is future work, so descriptor reclamation needs direct measurement before adoption.",
    },
    (
        "2026-06-03-tique-transactions-in-the-query-engine",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Priority retry and reservation modes need comparison against OCC aborts and owner execution templates.",
    },
    (
        "2026-06-03-ringleader-offloads-intra-server-orchestration-to-nics",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Logical session count needs explicit lookup-latency and starvation benchmarks before relying on NIC orchestration.",
    },
    (
        "2026-06-03-ringleader-offloads-intra-server-orchestration-to-nics",
        "resource_dag_scheduling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "FPGA scheduling latency must be measured in the DB request path before it can inform resource-DAG scheduling.",
    },
    (
        "2026-06-03-mrvs-split-bounded-numeric-hotspots-across-records",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Split numeric hotspots are useful only if invariant checks and snapshot reads preserve ordering within cost bounds.",
    },
    (
        "2026-06-03-mrvs-split-bounded-numeric-hotspots-across-records",
        "wal_before_visibility",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Split bounded counters remain ordinary MVCC rows under WAL-before-visibility, so the instead-of cue does not reclassify the link.",
    },
    (
        "2026-06-03-hattrick-throughput-frontier-for-htap-evaluation",
        "htap_freshness_router",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Freshness routing is valid only when telemetry attributes throughput loss to bounded queue and freshness budgets.",
    },
    (
        "2026-06-03-hattrick-throughput-frontier-for-htap-evaluation",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "HATtrick's simplified benchmark needs GPU DB-specific transfer, launch, buffer, and invalidation measurements.",
    },
    (
        "2026-06-03-hattrick-throughput-frontier-for-htap-evaluation",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Analytical recency tracking needs descriptor-lifetime and retained-reader measurements in GPU DB workloads.",
    },
    (
        "2026-06-03-cross-paper-synthesis-frontier-metrics-make-tradeoffs-visible",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Freshness routing needs frontier metrics that split retained reads, long scans, refreshes, and write batches.",
    },
    (
        "2026-06-03-cross-paper-synthesis-frontier-metrics-make-tradeoffs-visible",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Session-counting transfer requires benchmarks separating logical sessions from runnable requests and route classes.",
    },
    (
        "2026-06-03-cross-paper-synthesis-frontier-metrics-make-tradeoffs-visible",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshots need visibility-generation metrics under mixed reads, scans, refreshes, and write batches.",
    },
    (
        "2026-06-03-cross-paper-synthesis-frontier-metrics-make-tradeoffs-visible",
        "owner_ring_bundling",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that owner bundling must expose winners, losers, and correctness boundaries, not only throughput.",
    },
    (
        "2026-06-03-2-tree-record-level-hot-cold-migration-for-skewed-indexes",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Hot/cold migration is valid only if write-back and recovery complexity still preserve WAL-before-visibility and latency targets.",
    },
    (
        "2026-06-03-2-tree-record-level-hot-cold-migration-for-skewed-indexes",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The hot/cold tree structure is presented as an alternative to a separate cached-row dependency shape.",
    },
    (
        "2026-06-03-ncc-response-timed-strict-serializability-for-naturally-ordered-transactions",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "NCC targets distributed participant/coordinator stores, cautioning against direct transfer to single-node GPU-resident snapshots.",
    },
    (
        "2026-06-03-ncc-response-timed-strict-serializability-for-naturally-ordered-transactions",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "NCC verifies naturally ordered execution after the fact instead of paying dependency fences before every transaction.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-routes-need-measurable-boundaries",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier placement is explicitly routed through mixed-frontier benchmarks with bytes-by-tier and fallback measurements.",
    },
    (
        "2026-06-03-cross-paper-synthesis-fast-routes-need-measurable-boundaries",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot frontier transfer needs mixed read/write benchmarks and correctness traces before architecture adoption.",
    },
    (
        "2026-06-04-eiffel-software-packet-scheduling-for-request-admission",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts priority-queue implementations; the bounded integer queue evidence still supports compact descriptors.",
    },
    (
        "2026-06-04-deferred-actions-as-mvcc-safe-maintenance-scheduling",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue rejects ad hoc cleanup while supporting gated hot/cold conversion and tier maintenance lanes.",
    },
    (
        "2026-06-04-deferred-actions-as-mvcc-safe-maintenance-scheduling",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Deferred cleanup is valid only when old CPU, catalog, and GPU-resident metadata cannot be freed too early.",
    },
    (
        "2026-06-04-deferred-actions-as-mvcc-safe-maintenance-scheduling",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshot cleanup needs mutation throughput, read latency, queue-depth, oldest-reader, and version-chain measurements.",
    },
    (
        "2026-06-04-deferred-actions-as-mvcc-safe-maintenance-scheduling",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Queue and timestamp discipline supports owner-managed maintenance ordering rather than undermining owner bundling.",
    },
    (
        "2026-06-04-hybridlog-for-hot-in-place-point-updates-over-cold-storage",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "HybridLog-style mutable tails are useful only for narrow hot key/value or MVCC metadata tables with cold spill boundaries.",
    },
    (
        "2026-06-04-hybridlog-for-hot-in-place-point-updates-over-cold-storage",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "FASTER's WAL-elimination sketch is not directly transferable to SQL durability without a stronger recovery proof.",
    },
    (
        "2026-06-04-data-blocks-for-byte-addressable-compressed-htap-cold-chunks",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Cold-chunk route optimization is valid only when workload knowledge and frozen metadata can improve future predicates.",
    },
    (
        "2026-06-04-kvell-the-design-and-implementation-of-a-fast-persistent-key-value-store",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "KVell scans can inform retained snapshots only after consistent MVCC visibility boundaries are added.",
    },
    (
        "2026-06-04-bf-tree-variable-length-mini-pages-for-larger-than-memory-indexes",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner serialization, lock-free metadata, and partition-local locks must be measured against WAL-before-visibility constraints.",
    },
    (
        "2026-06-04-bf-tree-variable-length-mini-pages-for-larger-than-memory-indexes",
        "cpu_fallback_policy",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Explicit mini-segment misses are proposed as an alternative to rediscovering repeated misses through CPU fallback.",
    },
    (
        "2026-06-04-cross-paper-synthesis-warm-state-should-be-bounded-semantic-and-visible",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Partition-owned, batched, queue-limited cold-tier IO supports explicit multi-tier placement rather than hidden mmap behavior.",
    },
    (
        "2026-06-04-cross-paper-synthesis-warm-state-should-be-bounded-semantic-and-visible",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The partition-owned cold-tier IO evidence supports owner bundling despite the rather-than cue.",
    },
    (
        "2026-06-04-cross-paper-synthesis-warm-state-should-be-bounded-semantic-and-visible",
        "deficit_fairness",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cross-route accounting is explicitly framed as a mixed-workload benchmark for fairness and owner-pool contention.",
    },
    (
        "2026-06-04-oltpim-near-memory-placement-for-oltp-indexes-and-mvcc-metadata",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "OLTPim's placement transfer is explicitly a near-memory/HBM/host/NVMe placement test for GPU DB.",
    },
    (
        "2026-06-04-oltpim-near-memory-placement-for-oltp-indexes-and-mvcc-metadata",
        "wal_before_visibility",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue concerns rebuildable metadata; the evidence still supports WAL/checkpoint recovery as durable truth.",
    },
    (
        "2026-06-04-oltpim-near-memory-placement-for-oltp-indexes-and-mvcc-metadata",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "GPU-resident metadata is useful only for small payloads and compact visibility-qualified outputs.",
    },
    (
        "2026-06-04-revisiting-gpu-db-query-performance-and-resource-allocation",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Placement classification is valid only with queueing, cache interference, pinned-buffer, and invalidation-cost metrics.",
    },
    (
        "2026-06-04-revisiting-gpu-db-query-performance-and-resource-allocation",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Required-column residency must be tested against full resident segments, refresh cost, and retained-query latency.",
    },
    (
        "2026-06-04-revisiting-gpu-db-query-performance-and-resource-allocation",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evaluated systems are mostly analytical, so learned route advice needs retained-snapshot model evaluation.",
    },
    (
        "2026-06-04-gpu-oltp-concurrency-control-needs-conflict-aware-launch-policy",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The GPU OLTP evaluation omits WAL flush, recovery, DDL, transfers, and dynamic maintenance, limiting WAL inference.",
    },
    (
        "2026-06-04-cross-paper-synthesis-gpu-routes-need-separate-resource-conflict-and-visibility-classes",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL-safe retained mutation needs publication-lag, conflict-cost, and visibility-summary benchmark proof.",
    },
    (
        "2026-06-04-rcsi-scale-comes-from-treating-time-and-versions-as-first-class-routing-keys",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Approximate owner queues are valid only if internal ordering is not confused with SQL-visible correctness semantics.",
    },
    (
        "2026-06-04-flexible-resource-allocation-needs-database-visible-value-metrics",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue rejects ad hoc LRU lists and supports a unified value metric across placement tiers.",
    },
    (
        "2026-06-04-flexible-resource-allocation-needs-database-visible-value-metrics",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner bundling is useful only when cold or idle sessions cannot evict high-value active route state.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-descriptors-should-carry-value-visibility-and-scheduling-intent",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Descriptor lifetime is tied to explicit FIFO, JSQ, priority, and deadline benchmark comparisons.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-descriptors-should-carry-value-visibility-and-scheduling-intent",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-root invalidation is named as a range/generation test against table-wide invalidation.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-descriptors-should-carry-value-visibility-and-scheduling-intent",
        "resource_dag_scheduling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Scheduler resource-value claims require explicit queue, priority, and deadline policy benchmarks.",
    },
    (
        "2026-06-04-epic-deterministic-mvcc-removes-version-search-from-gpu-oltp-batches",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evaluated deterministic GPU OLTP path needs GPU DB workload measurements before adoption.",
    },
    (
        "2026-06-04-epic-deterministic-mvcc-removes-version-search-from-gpu-oltp-batches",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Scratchpad resident deltas and publication generations are explicitly framed as prototype work.",
    },
    (
        "2026-06-04-epic-deterministic-mvcc-removes-version-search-from-gpu-oltp-batches",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Epic's one-shot stored-procedure assumptions warn against descriptor policy that ignores interactive SQL state.",
    },
    (
        "2026-06-04-ccbench-exposes-cache-delay-and-version-lifetime-factors-in-concurrency-control",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot frontier transfer is gated on separate mutation-owner, retained-read, GPU-write, and cleanup measurements.",
    },
    (
        "2026-06-04-fastlanes-makes-compressed-column-layout-a-route-level-choice",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Compressed versus uncompressed GPU, host, and transfer placement needs direct evaluation.",
    },
    (
        "2026-06-04-fastlanes-makes-compressed-column-layout-a-route-level-choice",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Compressed in-flight vectors are an alternative to eager full-width materialization in route descriptors.",
    },
    (
        "2026-06-04-fastlanes-makes-compressed-column-layout-a-route-level-choice",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "FastLanes is analytical compression work, so it cautions against inferring WAL or MVCC safety.",
    },
    (
        "2026-06-04-cross-paper-synthesis-write-batches-snapshots-and-compressed-routes-all-need-explicit-physical-i",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis routes deterministic write templates through an explicit contention and layout benchmark matrix.",
    },
    (
        "2026-06-04-cross-paper-synthesis-write-batches-snapshots-and-compressed-routes-all-need-explicit-physical-i",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-serialized writes must be measured with layout and retained-read interactions before adoption.",
    },
    (
        "2026-06-04-cross-paper-synthesis-write-batches-snapshots-and-compressed-routes-all-need-explicit-physical-i",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Dense and compressed retained reads are explicitly part of the required benchmark matrix.",
    },
    (
        "2026-06-04-cross-paper-synthesis-write-batches-snapshots-and-compressed-routes-all-need-explicit-physical-i",
        "same_shape_microbatching",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Micro-batching remains useful only when p50 and p99 latency budgets are explicit, not throughput-only.",
    },
    (
        "2026-06-04-dbos-makes-runtime-state-queryable-without-making-every-fast-path-a-table-lookup",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue supports owner bundling by making capacity, placement, messages, and service state visible together.",
    },
    (
        "2026-06-04-dbos-makes-runtime-state-queryable-without-making-every-fast-path-a-table-lookup",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Placement metadata is valid only if buffer reuse waits for visible completion transitions.",
    },
    (
        "2026-06-04-cross-paper-synthesis-robust-routes-need-budgeted-temporary-state-not-just-resident-data",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Budgeted descriptor state needs a benchmark proving temporary bytes do not hide overload.",
    },
    (
        "2026-06-04-cross-paper-synthesis-robust-routes-need-budgeted-temporary-state-not-just-resident-data",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier placement for temporary and resident bytes is explicitly routed through no-GPU memory-credit measurements.",
    },
    (
        "2026-06-04-cross-paper-synthesis-robust-routes-need-budgeted-temporary-state-not-just-resident-data",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot generation and materialization shape metadata require route-telemetry prototype validation.",
    },
    (
        "2026-06-04-cross-paper-synthesis-robust-routes-need-budgeted-temporary-state-not-just-resident-data",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis requires proving large temporary operators do not evict hot retained snapshots.",
    },
    (
        "2026-06-04-centiman-watermarks-turn-occ-validation-into-an-asynchronous-frontier",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Tier retirement is valid only when all active read frontiers have advanced past the placement generation.",
    },
    (
        "2026-06-04-centiman-watermarks-turn-occ-validation-into-an-asynchronous-frontier",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The not-just cue requires explicit frontier proof, which directly supports snapshot frontier vectors.",
    },
    (
        "2026-06-04-hopsfs-moves-metadata-scale-into-transactional-shards",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained snapshots are valid only if resident metadata is not orphaned and stale admissions are rejected.",
    },
    (
        "2026-06-04-hopsfs-moves-metadata-scale-into-transactional-shards",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Subtree-style maintenance and invalid resident generation repair are explicitly prototype gates.",
    },
    (
        "2026-06-04-repmila-allocates-isolation-per-transaction-template-then-promotes-only-the-reads-that-pay-for-t",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Read-only template promotion is valid only when measured abort savings exceed added conflict traffic.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-should-be-a-typed-contract-not-an-owner-thread-habit",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Descriptor safety is supported only when authority is partitioned and route hints are generation-checked.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-should-be-a-typed-contract-not-an-owner-thread-habit",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Route optimization is useful only when metadata authority and generation checks are explicit.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-should-be-a-typed-contract-not-an-owner-thread-habit",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis names proof of visibility, authority, residency, and conflict intent as the first gate.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-should-be-a-typed-contract-not-an-owner-thread-habit",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Tiering follow-ups support placement only when they feed the same route-safety contract.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-should-be-a-typed-contract-not-an-owner-thread-habit",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Mixed visibility choices are safe only when attached to transaction templates.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-should-be-a-typed-contract-not-an-owner-thread-habit",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Metadata route hints support route roots only when authority is partitioned and generation-checked.",
    },
    (
        "2026-06-04-filescale-keeps-metadata-transactions-authoritative-while-caching-the-common-route",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-subtree maintenance and dirty metadata flush behavior are explicitly named as benchmark work.",
    },
    (
        "2026-06-04-filescale-keeps-metadata-transactions-authoritative-while-caching-the-common-route",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-resolution cost, stale-route repair, and owner queue depth are explicit measurement gates.",
    },
    (
        "2026-06-04-saving-private-hash-join-makes-temporary-memory-a-shared-route-budget",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Observed operator sizes support cost-based route planning rather than static optimizer guesses.",
    },
    (
        "2026-06-04-bolt-makes-admission-feedback-arrive-before-the-queue-is-already-stale",
        "vector_credit_admission",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Typed lane-credit admission is valid only if p99 latency and per-session fairness do not regress.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-budgets-need-fast-typed-feedback",
        "vector_credit_admission",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Lane-specific credits and direct bottleneck feedback directly support vector credit admission.",
    },
    (
        "2026-06-04-nwr-omits-blind-writes-only-when-another-visible-version-makes-them-unreachable",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Blind writes risk wasting owner slots, WAL bandwidth, invalidations, and retained snapshot retirement work.",
    },
    (
        "2026-06-04-nwr-omits-blind-writes-only-when-another-visible-version-makes-them-unreachable",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Blind-write coalescing needs latency, WAL, invalidation, and retained-snapshot retirement measurements.",
    },
    (
        "2026-06-04-nwr-omits-blind-writes-only-when-another-visible-version-makes-them-unreachable",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Resident-generation churn with blind-write coalescing must be measured before retained-snapshot adoption.",
    },
    (
        "2026-06-04-nwr-omits-blind-writes-only-when-another-visible-version-makes-them-unreachable",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The prototype is not a production SQL recovery system, so direct WAL-before-visibility inference is unsafe.",
    },
    (
        "2026-06-04-nwr-omits-blind-writes-only-when-another-visible-version-makes-them-unreachable",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Template-level validation and fallback support deterministic hot-write templates despite the cautionary wording.",
    },
    (
        "2026-06-04-nwr-omits-blind-writes-only-when-another-visible-version-makes-them-unreachable",
        "owner_ring_bundling",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Blind writes can consume owner queue capacity even when only the latest value matters.",
    },
    (
        "2026-06-04-dace-learns-planner-residuals-instead-of-replacing-planner-expertise",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The evidence warns that learned residuals must not replace deterministic placement and fallback facts.",
    },
    (
        "2026-06-04-dace-learns-planner-residuals-instead-of-replacing-planner-expertise",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The planner must keep deterministic resident-generation and fallback facts around learned residuals.",
    },
    (
        "2026-06-04-heterogeneous-aggregations-split-a-pipeline-by-calibrated-fragments",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Fragmented CPU/GPU aggregation is an alternative to choosing only a fully GPU-resident aggregate.",
    },
    (
        "2026-06-04-heterogeneous-aggregations-split-a-pipeline-by-calibrated-fragments",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "CPU/GPU split aggregation offers a calibrated placement alternative to a single resident execution choice.",
    },
    (
        "2026-06-04-heterogeneous-aggregations-split-a-pipeline-by-calibrated-fragments",
        "cpu_fallback_policy",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Explicit CPU/GPU fragment sets and fallback legality support a structured CPU fallback policy.",
    },
    (
        "2026-06-04-cachelib-makes-cache-policy-a-typed-storage-contract",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Warm-tier placement is valid only if metadata footprint and false-positive cold reads stay within budget.",
    },
    (
        "2026-06-04-cachelib-makes-cache-policy-a-typed-storage-contract",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Resident-resource handles are explicitly proposed as prototype work before retained-read adoption.",
    },
    (
        "2026-06-04-cachelib-makes-cache-policy-a-typed-storage-contract",
        "stable_handle_indirection",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Typed object lifecycles with handles and pools directly support stable handle indirection.",
    },
    (
        "2026-06-04-cachelib-makes-cache-policy-a-typed-storage-contract",
        "db_owned_cold_objects",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Database-visible lifecycle, admission, tier, and restart contracts support DB-owned cold objects.",
    },
    (
        "2026-06-04-pifos-make-scheduling-policy-explicit-at-enqueue-time",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "PIFO-style ranking is an alternative to relying only on FIFO arrival order inside owner rings.",
    },
    (
        "2026-06-04-pifos-make-scheduling-policy-explicit-at-enqueue-time",
        "effective_session_counting",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Session shaping is valid only if burst recovery does not push short-response p99 beyond target.",
    },
    (
        "2026-06-04-kvell-propagates-old-scan-versions-instead-of-retaining-snapshots",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "MVCC retention versus propagation needs version-byte, queue-wait, mutation-p99, and cleanup benchmarks.",
    },
    (
        "2026-06-04-quecc-makes-write-contention-a-planning-problem-not-an-execution-surprise",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Write-template route descriptors require batch-size, latency, WAL grouping, and abort/reject measurements.",
    },
    (
        "2026-06-04-quecc-makes-write-contention-a-planning-problem-not-an-execution-surprise",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "QueCC's in-memory update model warns against adopting its persistence shape for WAL-before-visibility.",
    },
    (
        "2026-06-04-tectonic-turns-cold-tier-efficiency-into-explicit-traffic-classes-and-sealed-metadata",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier time across NVMe, host, PCIe, HBM, kernels, and pinned buffers is an explicit measurement gate.",
    },
    (
        "2026-06-04-hetcache-makes-cache-placement-execution-centric-across-cpu-gpu-and-nvme",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "HetCache targets analytical scans, so it cautions against WAL, MVCC, and session-admission inference.",
    },
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
    if link.get("review_status") == "reviewed_weak_signal":
        return "reviewed_weak_signal", "reviewed weak-signal link has substantive retained evidence"
    if snippet_term_hits == 0:
        if visible_alias_hits:
            if link["confidence"] != "low" or link.get("review_status") == "reviewed_supported":
                return "direct", "snippet contains visible mechanism aliases"
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


def attach_relation_type(link: dict) -> None:
    relation_type = link.get("relation_type", "supports")
    if relation_type not in RELATION_TYPES:
        raise ValueError(f"unknown paper-mechanism relation type: {relation_type}")
    link["relation_type"] = relation_type
    link["relation_reason"] = RELATION_TYPE_DESCRIPTIONS[relation_type]
    link.setdefault("relation_review_status", "unreviewed")


def relation_candidate_details(link: dict) -> tuple[str, str]:
    if link["relation_type"] != "supports":
        return link["relation_type"], "already classified as a non-support relation"

    snippet = link["evidence_span"].get("snippet", "").lower()
    for relation_type in RELATION_CANDIDATE_PRIORITY:
        for cue in RELATION_CANDIDATE_CUES[relation_type]:
            if cue in snippet:
                return relation_type, f"evidence snippet contains relation cue: {cue}"
    return "supports", "no generated non-support relation cue found"


def attach_evidence_span(entry: dict, link: dict, mechanism_name: str) -> None:
    attach_relation_type(link)
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
    candidate_type, candidate_reason = relation_candidate_details(link)
    link["relation_candidate"] = {
        "candidate_relation_type": candidate_type,
        "candidate_reason": candidate_reason,
    }
    if link["relation_review_status"] == "unreviewed":
        if candidate_type == "supports":
            link["relation_review_status"] = "not_required"
        else:
            link["relation_review_status"] = "candidate_pending_review"


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
        relation_override = RELATION_REVIEW_OVERRIDES.get((entry["id"], link["mechanism_id"]))
        if relation_override:
            relation_type = relation_override["relation_type"]
            link["relation_type"] = relation_type
            link["relation_review_status"] = (
                "reviewed_support_candidate"
                if relation_type == "supports"
                else "reviewed_reclassified"
            )
            link["relation_review_note"] = relation_override["relation_review_note"]
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
    relation_type_counts: Counter = Counter()
    relation_candidate_counts: Counter = Counter()
    relation_review_status_counts: Counter = Counter()
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
            relation_type_counts[link["relation_type"]] += 1
            relation_candidate_counts[link["relation_candidate"]["candidate_relation_type"]] += 1
            relation_review_status_counts[link["relation_review_status"]] += 1
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
        "schema": "gpu-db-research-paper-mechanism-links-v4",
        "description": "Generated traceability from literature journal entries to architecture mechanisms, including generated evidence spans, evidence quality reasons, and typed paper-mechanism relations. Review low-confidence and fallback links before making architectural commitments.",
        "source_journal": "docs/research/gpu-db-literature-journal.md",
        "source_mechanisms": "docs/research/architecture-compatibility/mechanisms.json",
        "relation_types": {
            relation_type: RELATION_TYPE_DESCRIPTIONS[relation_type]
            for relation_type in sorted(RELATION_TYPES)
        },
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
            "relation_type_counts": dict(sorted(relation_type_counts.items())),
            "relation_candidate_counts": dict(sorted(relation_candidate_counts.items())),
            "relation_review_status_counts": dict(sorted(relation_review_status_counts.items())),
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
    relation_type_counts = index["summary"]["relation_type_counts"]
    relation_candidate_counts = index["summary"]["relation_candidate_counts"]
    relation_review_status_counts = index["summary"]["relation_review_status_counts"]
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
        if link["evidence_span"]["quality"] not in ("direct", "reviewed_weak_signal")
    ]
    reviewed_weak_signal_records = [
        (record, link)
        for record in records
        for link in record["mechanism_links"]
        if link["evidence_span"]["quality"] == "reviewed_weak_signal"
    ]
    non_support_relation_records = [
        (record, link)
        for record in records
        for link in record["mechanism_links"]
        if link["relation_type"] != "supports"
    ]
    non_support_relation_candidate_records = [
        (record, link)
        for record in records
        for link in record["mechanism_links"]
        if link["relation_candidate"]["candidate_relation_type"] != "supports"
    ]
    pending_non_support_relation_candidate_records = [
        (record, link)
        for record, link in non_support_relation_candidate_records
        if link["relation_review_status"] == "candidate_pending_review"
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

    lines.extend(["", "## Link Relation Types", ""])
    for relation_type, count in sorted(relation_type_counts.items()):
        description = RELATION_TYPE_DESCRIPTIONS.get(relation_type, "")
        lines.append(f"- {relation_type}: {count} ({description})")

    lines.extend(["", "## Generated Non-Support Relation Candidates", ""])
    lines.append(
        "These are generated audit candidates. Reviewed slices either reclassify them or keep the original support relation with an explicit note."
    )
    lines.append("")
    lines.append("Candidate relation counts:")
    for relation_type, count in sorted(relation_candidate_counts.items()):
        description = RELATION_TYPE_DESCRIPTIONS.get(relation_type, "")
        lines.append(f"- {relation_type}: {count} ({description})")
    lines.append("")
    lines.append("Candidate review status counts:")
    for status, count in sorted(relation_review_status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.append("")
    lines.append("Pending candidate audit queue:")
    if pending_non_support_relation_candidate_records:
        for record, link in pending_non_support_relation_candidate_records[:80]:
            span = link["evidence_span"]
            candidate = link["relation_candidate"]
            snippet = span["snippet"].replace("|", "\\|")
            lines.append(
                f"- `{record['id']}` -> {link['mechanism_id']}:{candidate['candidate_relation_type']} ({candidate['candidate_reason']}): {snippet}"
            )
        if len(pending_non_support_relation_candidate_records) > 80:
            lines.append(f"- ... {len(pending_non_support_relation_candidate_records) - 80} more")
    else:
        lines.append("- none")

    lines.extend(["", "## Non-Support Relation Audit", ""])
    if non_support_relation_records:
        for record, link in non_support_relation_records[:80]:
            span = link["evidence_span"]
            snippet = span["snippet"].replace("|", "\\|")
            lines.append(
                f"- `{record['id']}` -> {link['mechanism_id']}:{link['relation_type']} ({link['relation_review_note']}): {snippet}"
            )
        if len(non_support_relation_records) > 80:
            lines.append(f"- ... {len(non_support_relation_records) - 80} more")
    else:
        lines.append("- none")

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

    lines.extend(["", "## Reviewed Weak-Signal Evidence", ""])
    if reviewed_weak_signal_records:
        for record, link in reviewed_weak_signal_records[:80]:
            span = link["evidence_span"]
            snippet = span["snippet"].replace("|", "\\|")
            lines.append(
                f"- `{record['id']}` -> {link['mechanism_id']}:{span['quality']} ({span['quality_reason']}): {snippet}"
            )
        if len(reviewed_weak_signal_records) > 80:
            lines.append(f"- ... {len(reviewed_weak_signal_records) - 80} more")
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
            f"{link['mechanism_id']}:{link['relation_type']}:{link['confidence']}:{link['review_status']}"
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
