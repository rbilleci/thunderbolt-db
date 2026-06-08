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
DOI_RE = re.compile(r"(?:doi:\s*|doi\.org/)(?P<doi>10\.\d{4,9}/[^\s`]+)", re.IGNORECASE)
ARXIV_RE = re.compile(r"(?:arxiv[:\s]+|arxiv\.org/(?:abs|pdf)/)(?P<arxiv>\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s`)>]+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
QUOTED_TITLE_RE = re.compile(r'"(?P<title>[^"]+)"')

IDENTIFIER_REVIEW_OVERRIDES: dict[str, dict[str, str]] = {
    "2026-06-02-caladan-mitigating-interference-at-microsecond-timescales": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/osdi20/presentation/fried",
    },
    "2026-06-02-concurrent-analytical-query-processing-with-gpus": {
        "doi": "10.14778/2732967.2732976",
        "doi_review": "repaired_from_crossref_pvldb_record",
        "review_source": "https://doi.org/10.14778/2732967.2732976",
    },
    "2026-06-02-datacenter-rpcs-can-be-general-and-fast": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/nsdi19/presentation/kalia",
    },
    "2026-06-03-2-tree-record-level-hot-cold-migration-for-skewed-indexes": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://www.cidrdb.org/cidr2023/papers/p57-zhou.pdf",
    },
    "2026-06-03-a-cxl-powered-database-system-opportunities-and-challenges": {
        "doi": "10.1109/icde60146.2024.00447",
        "doi_review": "repaired_from_crossref_ieee_record",
        "review_source": "https://doi.org/10.1109/icde60146.2024.00447",
    },
    "2026-06-03-arachne-core-aware-thread-management": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/osdi18/presentation/qin",
    },
    "2026-06-03-bmc-safe-in-kernel-pre-stack-caching": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/nsdi21/presentation/ghigoff",
    },
    "2026-06-03-bohm-serializable-multiversion-ordering": {
        "doi": "10.14778/2809974.2809981",
        "doi_review": "repaired_from_crossref_pvldb_record",
        "review_source": "https://doi.org/10.14778/2809974.2809981",
    },
    "2026-06-03-btrim-hybrid-in-memory-row-store-for-extreme-oltp": {
        "doi": "10.14778/3229863.3229875",
        "doi_review": "repaired_from_crossref_pvldb_record",
        "review_source": "https://doi.org/10.14778/3229863.3229875",
    },
    "2026-06-03-dana-directly-attached-nvme-arrays": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://www.cidrdb.org/cidr2020/papers/p16-haas-cidr20.pdf",
    },
    "2026-06-03-dbos-database-oriented-operating-system-stack": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://www.vldb.org/cidrdb/papers/2022/p26-li.pdf",
    },
    "2026-06-03-detox-transactional-cache-hit-rate": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/osdi23/presentation/cheng",
    },
    "2026-06-03-efficient-scheduling-policies-for-microsecond-scale-tasks": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_dblp_and_official_record_have_no_arxiv",
        "review_source": "https://www.usenix.org/conference/nsdi22/presentation/mcclure",
    },
    "2026-06-03-empirical-in-memory-mvcc-design-tradeoffs": {
        "doi": "10.14778/3067421.3067427",
        "doi_review": "repaired_from_pvldb_record",
        "review_source": "https://doi.org/10.14778/3067421.3067427",
    },
    "2026-06-03-fastmap-scalable-mmap-for-fast-storage": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/atc20/presentation/papagiannis",
    },
    "2026-06-03-leveraging-lock-contention-to-improve-oltp-application-performance": {
        "doi": "10.14778/2876473.2876479",
        "doi_review": "repaired_from_pvldb_record",
        "review_source": "https://doi.org/10.14778/2876473.2876479",
    },
    "2026-06-03-mmap-is-not-a-buffer-pool-substitute": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://db.cs.cmu.edu/mmap-cidr2022/",
    },
    "2026-06-03-ncc-response-timed-strict-serializability-for-naturally-ordered-transactions": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv": "2305.14270",
        "arxiv_review": "repaired_from_arxiv_record",
        "review_source": "https://arxiv.org/abs/2305.14270",
    },
    "2026-06-03-nomad-non-exclusive-memory-tiering": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv": "2401.13154",
        "arxiv_review": "repaired_from_arxiv_record",
        "review_source": "https://arxiv.org/abs/2401.13154",
    },
    "2026-06-03-oltp-through-the-looking-glass-16-years-later": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://vldb.org/cidrdb/2025/oltp-through-the-looking-glass-16-years-later-communication-is-the-new-bottleneck.html",
    },
    "2026-06-03-p-tree-multi-versioned-indexes-for-htap-snapshots": {
        "doi": "10.14778/3364324.3364334",
        "doi_review": "repaired_from_pvldb_record",
        "review_source": "https://doi.org/10.14778/3364324.3364334",
    },
    "2026-06-03-pasha-partitioned-shared-cxl-pod-architecture": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://vldb.org/cidrdb/2025/pasha-an-efficient-scalable-database-architecture-for-cxl-pods.html",
    },
    "2026-06-03-polyjuice-learned-concurrency-control-policies": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv": "2105.10329",
        "arxiv_review": "repaired_from_arxiv_record",
        "review_source": "https://arxiv.org/abs/2105.10329",
    },
    "2026-06-03-predicate-transfer-for-multi-join-pre-filtering": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv": "2307.15255",
        "arxiv_review": "repaired_from_arxiv_record",
        "review_source": "https://arxiv.org/abs/2307.15255",
    },
    "2026-06-03-pwv-early-write-visibility": {
        "doi": "10.14778/3055540.3055553",
        "doi_review": "repaired_from_pvldb_record",
        "review_source": "https://doi.org/10.14778/3055540.3055553",
    },
    "2026-06-03-r2p2-request-response-pairs-for-rpc-admission": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/atc19/presentation/kogias-r2p2",
    },
    "2026-06-03-resource-adaptive-query-execution-with-paged-memory-management": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://www.vldb.org/cidrdb/papers/2025/p2-otaki.pdf",
    },
    "2026-06-03-ringleader-offloads-intra-server-orchestration-to-nics": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/nsdi23/presentation/lin",
    },
    "2026-06-03-shenango-high-efficiency-latency-sensitive-runtime": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/nsdi19/presentation/ousterhout",
    },
    "2026-06-03-shinjuku-microsecond-scale-preemptive-scheduling": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/nsdi19/presentation/kaffes",
    },
    "2026-06-03-smf-schedule-first-transaction-ordering": {
        "doi": "10.14778/3681954.3681956",
        "doi_review": "repaired_from_pvldb_record",
        "review_source": "https://doi.org/10.14778/3681954.3681956",
    },
    "2026-06-03-sp-pifo-strict-priority-approximation-of-programmable-scheduling": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/nsdi20/presentation/alcoz",
    },
    "2026-06-03-umbra-variable-size-pages-for-ssd-backed-hot-working-sets": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://www.vldb.org/cidrdb/papers/2020/p29-neumann-cidr20.pdf",
    },
    "2026-06-03-vessel-fast-userspace-core-scheduling": {
        "doi": "10.1145/3694715.3695976",
        "doi_review": "repaired_from_acm_record",
        "review_source": "https://doi.org/10.1145/3694715.3695976",
    },
    "2026-06-04-acc-chooses-concurrency-control-per-cluster-instead-of-globally": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://www.cidrdb.org/cidr2017/papers/p63-tang-cidr17.pdf",
    },
    "2026-06-04-bolt-makes-admission-feedback-arrive-before-the-queue-is-already-stale": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/nsdi23/presentation/arslan",
    },
    "2026-06-04-cachelib-makes-cache-policy-a-typed-storage-contract": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/osdi20/presentation/berg",
    },
    "2026-06-04-calvinfs-makes-namespace-metadata-a-deterministic-transaction-workload": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/fast15/technical-sessions/presentation/thomson",
    },
    "2026-06-04-concurrent-query-prediction-needs-explicit-interference-edges": {
        "doi": "10.14778/3397230.3397238",
        "doi_review": "repaired_from_pvldb_record",
        "review_source": "https://doi.org/10.14778/3397230.3397238",
    },
    "2026-06-04-cooperative-memory-management-turns-cache-pressure-into-an-admission-choice": {
        "doi": "10.1145/3596225.3596230",
        "doi_review": "repaired_from_acm_record",
        "review_source": "https://doi.org/10.1145/3596225.3596230",
    },
    "2026-06-04-d-rdma-makes-fragmented-database-transfer-a-nic-scheduling-problem": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://vldb.org/cidrdb/2022/d-rdma-bringing-zero-copy-rdma-to-database-systems.html",
    },
    "2026-06-04-data-blocks-for-byte-addressable-compressed-htap-cold-chunks": {
        "doi": "10.1145/2882903.2882925",
        "doi_review": "repaired_from_acm_record",
        "review_source": "https://doi.org/10.1145/2882903.2882925",
    },
    "2026-06-04-database-kernels-turn-cxl-storage-into-typed-database-services": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://vldb.org/cidrdb/2024/database-kernels-seamless-integration-of-database-systems-and-fast-storage-via-cxl.html",
    },
    "2026-06-04-deferred-actions-as-mvcc-safe-maintenance-scheduling": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_cidr_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_cidr_record_has_no_arxiv",
        "review_source": "https://www.vldb.org/cidrdb/2021/everything-is-a-transaction-unifying-logical-concurrency-control-and-physical-data-structure-maintenance-in-database-management.html",
    },
    "2026-06-04-detock-resolves-ordering-cycles-instead-of-aborting-them": {
        "doi": "10.1145/3589293",
        "doi_review": "repaired_from_acm_record",
        "review_source": "https://doi.org/10.1145/3589293",
    },
    "2026-06-04-dint-keeps-frequent-transaction-steps-inside-the-kernel-datapath": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/nsdi24/presentation/zhou-yang",
    },
    "2026-06-04-eiffel-software-packet-scheduling-for-request-admission": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv": "1810.03060",
        "arxiv_review": "repaired_from_arxiv_record",
        "review_source": "https://arxiv.org/abs/1810.03060",
    },
    "2026-06-04-epic-deterministic-mvcc-removes-version-search-from-gpu-oltp-batches": {
        "doi_status": "reviewed_absent",
        "doi_review": "reviewed_official_usenix_record_has_no_doi",
        "arxiv_status": "reviewed_absent",
        "arxiv_review": "reviewed_official_usenix_record_has_no_arxiv",
        "review_source": "https://www.usenix.org/conference/osdi24/presentation/qian",
    },
}

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
        "2026-06-05-cross-paper-synthesis-route-metadata-must-prove-both-correctness-and-pressure-shape",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The three-lane admission benchmark is the named proof gate for hot-write template adoption.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-metadata-must-prove-both-correctness-and-pressure-shape",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fallback is valid only when the pressure proof names owner, queue budget, setup cost, fallback lane, and timeout condition.",
    },
    (
        "2026-06-05-gpu-locality-is-a-bandwidth-contract-not-just-a-cache-hint",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Placement policy must model and measure local/remote GPU bandwidth symptoms before assuming direct hardware control.",
    },
    (
        "2026-06-05-gpu-locality-is-a-bandwidth-contract-not-just-a-cache-hint",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner assignment for resident chunks, key vectors, and CUDA queues is explicitly framed as the benchmarkable analogue.",
    },
    (
        "2026-06-05-learned-route-hints-should-be-bounded-inspectable-and-opt-in",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Bao-style route hints require latency and regret evaluation before informing cost-based route optimization.",
    },
    (
        "2026-06-05-learned-route-hints-should-be-bounded-inspectable-and-opt-in",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Learned route planning must be measured for competition with owner queues before owner bundling can rely on it.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-schedulers-need-class-proof-and-completion-locality",
        "resource_dag_scheduling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Resource scheduling fields and completion-locality measurements are explicit proof gates for route scheduling.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-schedulers-need-class-proof-and-completion-locality",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring completion locality must be measured across cold-tier, GPU, and CPU fallback completions.",
    },
    (
        "2026-06-05-sap-hana-nse-makes-warm-data-a-first-class-column-store-load-unit",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "NSE targets CPU column-store warm storage and cautions against direct GPU-resident MVCC snapshot transfer.",
    },
    (
        "2026-06-05-sap-hana-nse-makes-warm-data-a-first-class-column-store-load-unit",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Warm-buffer policy is safe only if recovery and WAL replay have a distinct emergency path from user-query buffers.",
    },
    (
        "2026-06-05-sap-hana-nse-makes-warm-data-a-first-class-column-store-load-unit",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Mutable-delta plus immutable-main freshness routing needs write-throughput, freshness-lag, and replay progress measurements.",
    },
    (
        "2026-06-05-sql-server-real-time-analytics-overlays-columnar-reads-onto-oltp-storage",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The mutable-delta plus immutable-resident-main read path is explicitly prototype work for retained snapshots.",
    },
    (
        "2026-06-05-sql-server-real-time-analytics-overlays-columnar-reads-onto-oltp-storage",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "CPU delta stores and compressed row groups are an alternative tiering shape to immediate resident-row position scans.",
    },
    (
        "2026-06-05-sql-server-real-time-analytics-overlays-columnar-reads-onto-oltp-storage",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The SQL Server CPU columnstore scope cautions against direct descriptor-reclamation transfer to GPU execution.",
    },
    (
        "2026-06-05-cross-paper-synthesis-htap-freshness-and-modular-transaction-lanes-are-converging",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Adaptive fallback from optimistic hot-write paths is named as a measured route-certificate gate.",
    },
    (
        "2026-06-05-cross-paper-synthesis-htap-freshness-and-modular-transaction-lanes-are-converging",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hot-write template adoption depends on measured conflict telemetry across retained reads, refresh, and fallback.",
    },
    (
        "2026-06-05-chablis-decouples-global-snapshot-epochs-from-local-transaction-latency",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Snapshot frontiers are needed only when retained snapshots or cross-owner routes require broader visibility coordination.",
    },
    (
        "2026-06-05-chablis-decouples-global-snapshot-epochs-from-local-transaction-latency",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Two-level publication needs benchmarking before replacing per-transaction owner or residency-domain queries.",
    },
    (
        "2026-06-05-chablis-decouples-global-snapshot-epochs-from-local-transaction-latency",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Lock-free retained reads need stale-certificate and overlapping-writer benchmarks before descriptor adoption.",
    },
    (
        "2026-06-05-chablis-decouples-global-snapshot-epochs-from-local-transaction-latency",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained snapshots should join broader frontier coordination only when cross-owner or retained-read routes need it.",
    },
    (
        "2026-06-05-slog-keeps-local-transactions-fast-with-lock-only-cross-owner-ordering",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring bundling needs p50/p99, queue-wait, and cross-owner write measurements under the same session load.",
    },
    (
        "2026-06-05-chardonnay-turns-cold-data-reads-into-pre-lock-admission-work",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cold-page admission must measure lock hold time and p99 before tier placement can rely on pre-lock routing.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-must-preflight-both-ownership-and-tiers",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "MVCC frontiers require route-preflight, stale-certificate, and old-version GC measurements before adoption.",
    },
    (
        "2026-06-05-hermes-keeps-htap-freshness-cheap-with-row-id-deltas-and-mergeable-columnar-generations",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Descriptor reclamation is valid only if changed-row overlays prove filtered main-segment rows after compaction or snapshot retirement.",
    },
    (
        "2026-06-05-hermes-keeps-htap-freshness-cheap-with-row-id-deltas-and-mergeable-columnar-generations",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hermes omits CUDA, GPU pressure, pinned buffers, transfers, NVMe tiers, and session-scale admission evaluation.",
    },
    (
        "2026-06-05-page-as-you-go-makes-columnar-residency-page-granular-without-abandoning-vectorized-execution",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Same-shape batching needs warm-column point lookup and micro-batch pinning benchmarks across resident and paged dictionaries.",
    },
    (
        "2026-06-05-relaxed-operator-fusion-makes-materialization-a-route-shape-decision",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Micro-batch sizing must pass a proof gate showing planner staging is avoided when overhead dominates.",
    },
    (
        "2026-06-05-easycommit-makes-non-blocking-commit-a-message-redundancy-tradeoff",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "MVCC cleanup transfer is gated on durable-write, owner-cleanup, coordinator-failure, and resource-retention measurements.",
    },
    (
        "2026-06-05-cross-paper-synthesis-commit-decisions-need-a-recoverable-visibility-contract",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The only-if phrase is corpus-planning guidance; the retained evidence still supports owner-ring visibility contracts.",
    },
    (
        "2026-06-05-cross-paper-synthesis-commit-decisions-need-a-recoverable-visibility-contract",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The only-if phrase names future reading priorities, while the retained synthesis supports recoverable visibility frontiers.",
    },
    (
        "2026-06-05-cross-paper-synthesis-tail-contracts-need-age-fan-out-and-accelerator-budget",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-visible priority lanes are tied to measured bounds for long GPU, refresh, and decompression work.",
    },
    (
        "2026-06-05-ocean-vista-turns-visibility-into-batched-watermark-gossip",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Batched watermark gossip is presented as an alternative to synchronous per-transaction visibility coordination.",
    },
    (
        "2026-06-05-ocean-vista-turns-visibility-into-batched-watermark-gossip",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback below durable boundaries is explicitly framed as a comparison/prototype gate.",
    },
    (
        "2026-06-05-ocean-vista-turns-visibility-into-batched-watermark-gossip",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Invisible multi-version placeholders and gossiped watermarks are an alternative to immediate conflict resolution.",
    },
    (
        "2026-06-05-tmo-makes-tiering-a-pressure-controlled-feedback-loop",
        "effective_session_counting",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Session-memory classification is valid only if it avoids write-cap violations and unrelated cold-tier p99 inflation.",
    },
    (
        "2026-06-05-tmo-makes-tiering-a-pressure-controlled-feedback-loop",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evidence requires proof gates and refault benchmarks before tiering WAL or visibility-adjacent state.",
    },
    (
        "2026-06-05-tmo-makes-tiering-a-pressure-controlled-feedback-loop",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fallback choices under pressure annotations are explicitly driven by measured stall budgets.",
    },
    (
        "2026-06-05-htm-is-a-primitive-for-tiny-atomic-publications-not-a-general-index-concurrency-plan",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "HTM-protected traversal is valid only if retained snapshot tracking survives realistic key and payload shapes.",
    },
    (
        "2026-06-05-cross-paper-synthesis-publish-small-retire-explicitly-place-by-pressure",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Descriptor publication and retirement are routed through route-certificate and retained-snapshot benchmarks.",
    },
    (
        "2026-06-05-cross-paper-synthesis-publish-small-retire-explicitly-place-by-pressure",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshots need publication benchmarks across CAS, owner-message, and optional HTM mechanisms.",
    },
    (
        "2026-06-05-flexpushdowndb-hybrid-pushdown-and-caching-in-a-cloud-dbms",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hybrid placement is gated on merge cost, transfer bytes, queue wait, and latency measurements.",
    },
    (
        "2026-06-05-flexpushdowndb-hybrid-pushdown-and-caching-in-a-cloud-dbms",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cache-miss admission choices between promotion, cold routes, CPU fallback, and rejection must be tested.",
    },
    (
        "2026-06-05-hsm-a-hybrid-slowdown-model-for-multitasking-gpus",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The evidence is GPU-compute benchmarking, so SQL placement transfer needs database-specific validation.",
    },
    (
        "2026-06-05-occ-batching-makes-commit-order-a-bounded-optimization-problem",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Batch-size sweet spots for hot writes, point lookups, refresh, and fallback are explicit measurement gates.",
    },
    (
        "2026-06-05-crystal-resident-gpu-execution-wins-when-transfer-is-not-the-bottleneck",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route choice must match measured winners across selectivity and resident-state changes.",
    },
    (
        "2026-06-05-cross-paper-synthesis-resident-routes-need-fairness-resource-class-and-transfer-proof",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained query routing is gated on route-certificate benchmarks that log residency, transfer, and pressure.",
    },
    (
        "2026-06-05-cross-paper-synthesis-resident-routes-need-fairness-resource-class-and-transfer-proof",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Micro-batch adoption is tied to four-route comparator and mixed-workload benchmarks.",
    },
    (
        "2026-06-05-cross-paper-synthesis-resident-routes-need-fairness-resource-class-and-transfer-proof",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Useful reordering and fusion are valid only inside bounded validation windows and compatible visibility boundaries.",
    },
    (
        "2026-06-05-cross-paper-synthesis-resident-routes-need-fairness-resource-class-and-transfer-proof",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that fast routes need current certificates, not isolated hot-write assumptions.",
    },
    (
        "2026-06-05-cross-paper-synthesis-resident-routes-need-fairness-resource-class-and-transfer-proof",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner routing is valid only when route certificates prove resource class, queue wait, and co-run pressure.",
    },
    (
        "2026-06-05-optimal-concurrency-is-accepted-correct-schedules-not-just-fewer-locks",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Optimizer metadata should not be judged by one lock-count or throughput benchmark.",
    },
    (
        "2026-06-05-optimal-concurrency-is-accepted-correct-schedules-not-just-fewer-locks",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback metadata needs accepted-schedule, retry, latency, allocation, and stale-route measurements.",
    },
    (
        "2026-06-05-optimal-concurrency-is-accepted-correct-schedules-not-just-fewer-locks",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained lookup fallback is explicitly routed through interleaving, retry, latency, and stale-route tests.",
    },
    (
        "2026-06-05-index-checkpoints-move-recovery-risk-from-rebuild-time-to-derived-state-correctness",
        "stable_handle_indirection",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Logical node indirection provides an alternative to direct child pointers and full root-to-leaf copying.",
    },
    (
        "2026-06-05-adaptive-execution-makes-compilation-a-runtime-route-not-a-startup-tax",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Short retained reads and first-run statements need route tests before choosing compilation or batching.",
    },
    (
        "2026-06-05-adaptive-execution-makes-compilation-a-runtime-route-not-a-startup-tax",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Compilation and specialization are valid only when they do not starve network IO, owners, or GPU queues.",
    },
    (
        "2026-06-05-transaction-triaging-turns-admission-metadata-into-execution-locality",
        "owner_ring_bundling",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The evidence warns that generic load-balanced ingress can create wrong-owner hops and cold metadata paths.",
    },
    (
        "2026-06-05-transaction-triaging-turns-admission-metadata-into-execution-locality",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Silo evaluation does not cover pgwire, MVCC, WAL, recovery, or GPU-resident snapshots.",
    },
    (
        "2026-06-05-transaction-triaging-turns-admission-metadata-into-execution-locality",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The wrong-owner risk cautions against placement policies that add cross-owner hops and cold indirection.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-metadata-must-prove-both-correctness-and-pressure-shape",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner routing is valid only when the pressure proof names owner, budgets, setup cost, fallback, and timeout.",
    },
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
    (
        "2026-06-04-silicondb-adapts-morsel-scheduling-to-heterogeneous-accelerators",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Accelerator rewrites are useful only when route descriptors and telemetry prove an end-to-end latency win.",
    },
    (
        "2026-06-04-silicondb-adapts-morsel-scheduling-to-heterogeneous-accelerators",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Dedicated versus piggybacked accelerator queue ownership must be evaluated before adopting the owner shape.",
    },
    (
        "2026-06-04-silicondb-adapts-morsel-scheduling-to-heterogeneous-accelerators",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Queue buildup and straggler risk make the route-cost model a measurement gate rather than immediate support.",
    },
    (
        "2026-06-04-sundial-logical-leases-unify-serializable-ordering-and-cache-coherence",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Logical leases can inform retained snapshots only after lease-renewal and fallback behavior is benchmarked.",
    },
    (
        "2026-06-04-eigen-manages-database-capacity-as-a-three-layer-resource-flow",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The resource-vector evidence supports retained snapshot routes declaring explicit resource demand.",
    },
    (
        "2026-06-04-gacco-batches-same-shape-oltp-on-gpu-while-cpu-owns-the-full-database",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The accessible evidence leaves recovery and logging unknown, so WAL-before-visibility needs validation.",
    },
    (
        "2026-06-04-no-false-negatives-makes-serializable-conflict-acceptance-explicit",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The single-version graph scheduler evidence directly supports explicit GPU OLTP conflict ordering.",
    },
    (
        "2026-06-04-no-false-negatives-makes-serializable-conflict-acceptance-explicit",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Epoch-marked prior row snapshots support frontier-style visibility for mixed OLTP and OLAP reads.",
    },
    (
        "2026-06-04-no-false-negatives-makes-serializable-conflict-acceptance-explicit",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner telemetry is useful only when routes cannot enter execution without enough conflict information.",
    },
    (
        "2026-06-04-robust-co-processor-routes-need-data-residency-and-heap-budgets",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Operator-level CPU restart on accelerator allocation failure is an alternative to deterministic hot-write templating.",
    },
    (
        "2026-06-04-distributed-gpu-joins-hide-network-shuffle-under-gpu-work",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Overlapping network shuffle with GPU work is presented as an alternative to retained snapshot locality assumptions.",
    },
    (
        "2026-06-04-distributed-gpu-joins-hide-network-shuffle-under-gpu-work",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Skew-aware placement and GPUDirect topology choices are explicitly route-cost benchmark gates.",
    },
    (
        "2026-06-04-distributed-gpu-joins-hide-network-shuffle-under-gpu-work",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The movement-overlap evidence supports placement decisions that account for unavoidable tier transfers.",
    },
    (
        "2026-06-04-spooky-granulates-lsm-compaction-by-largest-level-boundaries",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Refresh boundaries carrying visibility generation and WAL frontier support snapshot frontier vectors.",
    },
    (
        "2026-06-04-spooky-granulates-lsm-compaction-by-largest-level-boundaries",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The entry explicitly maps refresh partition groups and visibility generations to retained GPU snapshots.",
    },
    (
        "2026-06-04-ccaas-separates-conflict-resolution-from-execution-and-storage",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Subdomain owner saturation must be measured with a write-heavy benchmark before adopting the owner split.",
    },
    (
        "2026-06-04-strife-turns-high-contention-oltp-into-batch-local-owner-lanes",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "TPC-C, YCSB, and hot-record measurements are required before transferring the conflict-ordering shape.",
    },
    (
        "2026-06-04-strife-turns-high-contention-oltp-into-batch-local-owner-lanes",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Conflict-free queues and separated cluster discovery support explicit owner-lane bundling.",
    },
    (
        "2026-06-04-strife-turns-high-contention-oltp-into-batch-local-owner-lanes",
        "effective_session_counting",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Admission by active conflict shape rather than raw connection count supports effective session counting.",
    },
    (
        "2026-06-04-strife-turns-high-contention-oltp-into-batch-local-owner-lanes",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained reads coexist with hot write clusters only when descriptor density amortizes setup and transfer costs.",
    },
    (
        "2026-06-04-rtscan-maps-conjunctive-filters-onto-ray-tracing-cores",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "RT route selection depends on downstream device locality, result size, rebuild policy, and telemetry benchmarks.",
    },
    (
        "2026-06-04-rtscan-maps-conjunctive-filters-onto-ray-tracing-cores",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Attaching RT resident indexes to immutable retained generations supports route-root publication.",
    },
    (
        "2026-06-04-acc-chooses-concurrency-control-per-cluster-instead-of-globally",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cluster-specific concurrency-control selection needs GPU DB workload benchmarking before adoption.",
    },
    (
        "2026-06-04-acc-chooses-concurrency-control-per-cluster-instead-of-globally",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Mixed protocol routing is safe only if WAL replay and cache rebuild share deterministic visibility order.",
    },
    (
        "2026-06-04-acc-chooses-concurrency-control-per-cluster-instead-of-globally",
        "effective_session_counting",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Using active conflict shape rather than raw session count supports the effective-session admission model.",
    },
    (
        "2026-06-04-hdcc-interleaves-deterministic-batches-with-optimistic-lanes",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Hot-write templates are valid only when retry and deferred-commit conditions prove consistent reads.",
    },
    (
        "2026-06-04-hdcc-interleaves-deterministic-batches-with-optimistic-lanes",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Deterministic batch logs are useful only with explicit replay order, checkpoint, and invalidation frontiers.",
    },
    (
        "2026-06-04-hdcc-interleaves-deterministic-batches-with-optimistic-lanes",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Dependency frontiers carrying WAL epoch, owner generation, and batch id support snapshot frontier vectors.",
    },
    (
        "2026-06-04-hdcc-interleaves-deterministic-batches-with-optimistic-lanes",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Transaction-shaped route descriptors support explicit owner routing beyond a single global rule.",
    },
    (
        "2026-06-04-hdcc-interleaves-deterministic-batches-with-optimistic-lanes",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Per-item metadata placement is valid only if cache-contention hot spots are partitioned or sampled carefully.",
    },
    (
        "2026-06-04-hdcc-interleaves-deterministic-batches-with-optimistic-lanes",
        "dependency_witnesses",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The test cue describes retained batch dependency state; the WAL, owner, catalog, and batch frontiers support dependency witnesses.",
    },
    (
        "2026-06-04-cross-paper-synthesis-mixed-routes-need-replayable-frontiers",
        "wal_before_visibility",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless clause is corpus-planning guidance; the evidence keeps logging and replay frontiers as required design inputs.",
    },
    (
        "2026-06-04-cross-paper-synthesis-mixed-routes-need-replayable-frontiers",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The synthesis names scheduling and multi-owner replay as continuing route constraints, supporting explicit owner coordination.",
    },
    (
        "2026-06-04-cross-paper-synthesis-mixed-routes-need-replayable-frontiers",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless clause is corpus-priority guidance; the retained evidence still supports tier-aware routing within the mixed-route design.",
    },
    (
        "2026-06-04-cgrx-trades-exact-gpu-index-entries-for-bucketed-rt-core-lookups",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Bucketed RT-core lookups are usable only when old versions, deleted keys, and node retirement follow snapshot-generation reclamation.",
    },
    (
        "2026-06-04-calvinfs-makes-namespace-metadata-a-deterministic-transaction-workload",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts metadata-service architecture, not route optimization; deterministic metadata routing remains supporting evidence.",
    },
    (
        "2026-06-04-calvinfs-makes-namespace-metadata-a-deterministic-transaction-workload",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Immutable descriptor publication is valid only if source generations remain unchanged during compaction or segment rewrite.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-metadata-should-be-cached-ordered-and-explainable",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless clause is review-priority guidance; route metadata still supports explicit tier-placement decisions.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-metadata-should-be-cached-ordered-and-explainable",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis explicitly requires a metadata-cache microbenchmark proving lookup cost does not scale with logical sessions.",
    },
    (
        "2026-06-04-faster-embedded-state-stores-keep-hot-updates-in-place-while-cold-state-spills",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hot-set drift across DRAM and NVMe-like cold state is explicitly listed as a placement measurement gate.",
    },
    (
        "2026-06-04-faster-embedded-state-stores-keep-hot-updates-in-place-while-cold-state-spills",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts global checks with epoch refresh, which directly supports deferred descriptor reclamation.",
    },
    (
        "2026-06-04-faster-embedded-state-stores-keep-hot-updates-in-place-while-cold-state-spills",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Checkpoint, drift, latency, and recovery metrics are named measurement gates before WAL visibility transfer.",
    },
    (
        "2026-06-04-faster-embedded-state-stores-keep-hot-updates-in-place-while-cold-state-spills",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained resident indexes are valid only as rebuildable acceleration state with visible write-amplification and latency limits.",
    },
    (
        "2026-06-04-star-phase-switches-ownership-instead-of-paying-distributed-commit-on-every-transaction",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Phase-fenced metadata requires throughput, latency, abort, fence-overhead, and recovery proof measurements.",
    },
    (
        "2026-06-04-star-phase-switches-ownership-instead-of-paying-distributed-commit-on-every-transaction",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Dependency frontiers for phase-fenced metadata need throughput, latency, abort, and replay measurements before adoption.",
    },
    (
        "2026-06-04-rtindex-turns-rt-cores-into-a-read-mostly-gpu-secondary-index",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "RT-core indexes belong in HBM only when telemetry shows saved lookup work is worth the resident bytes.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-separate-write-point-state-and-resident-index-contracts",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner queue wait, fence wait, GPU queue wait, resident-index rebuild wait, and response-ring wait are explicit benchmark gates.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-separate-write-point-state-and-resident-index-contracts",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Resident index metadata is useful only when read-mostly, batched, and rebuilt at generation boundaries.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-separate-write-point-state-and-resident-index-contracts",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Route-root publication is valid only when resident indexes are read-mostly, batched, and generation-boundary rebuilt.",
    },
    (
        "2026-06-04-gpu-tps-maps-oltp-writes-onto-simt-with-grouping-locks-and-gpu-indexes",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "SmallBank and TPC-C evidence still needs GPU DB conflict-ordering evaluation before architectural adoption.",
    },
    (
        "2026-06-04-gpu-tps-maps-oltp-writes-onto-simt-with-grouping-locks-and-gpu-indexes",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts GPU OLTP with scan-only GPU use, which supports retained resident execution rather than opposing it.",
    },
    (
        "2026-06-04-gpu-tps-maps-oltp-writes-onto-simt-with-grouping-locks-and-gpu-indexes",
        "same_shape_microbatching",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Conflict risk is an admission-key input for same-shape write batches, not a warning against micro-batching itself.",
    },
    (
        "2026-06-04-gpu-tps-maps-oltp-writes-onto-simt-with-grouping-locks-and-gpu-indexes",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Conflict-risk metadata is part of shaping write batches and retained descriptors, not evidence against descriptor reclamation.",
    },
    (
        "2026-06-04-ltpg-removes-predefined-read-write-sets-from-gpu-batch-transactions",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The proof gate requires staged telemetry, WAL publication checks, and predefined-versus-dynamic batch comparison.",
    },
    (
        "2026-06-04-mocc-selectively-locks-only-hot-read-conflict-records",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "MOCC's CPU main-memory OLTP scope cautions against direct GPU, SQL planning, and tier-placement transfer.",
    },
    (
        "2026-06-04-adaptive-logging-makes-recovery-cost-a-write-path-budget",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The recovery-cost metric is explicitly a proof gate for dependency-footprint experiments on batched writes.",
    },
    (
        "2026-06-04-adaptive-logging-makes-recovery-cost-a-write-path-budget",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue contrasts fixed log detail with tunable resource policy, which remains supporting optimizer-advice evidence.",
    },
    (
        "2026-06-04-scalestore-treats-dram-remote-memory-and-nvme-as-one-coherent-page-tier",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Per-object fairness is presented as an alternative coordination shape to one global owner queue.",
    },
    (
        "2026-06-04-scalestore-treats-dram-remote-memory-and-nvme-as-one-coherent-page-tier",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained copies are safe only when incompatible resident copies are superseded before publishing visibility and old readers hold leases.",
    },
    (
        "2026-06-04-scalestore-treats-dram-remote-memory-and-nvme-as-one-coherent-page-tier",
        "wal_before_visibility",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts memory residency choices; the evidence remains supporting context for WAL-governed cold-tier routing.",
    },
    (
        "2026-06-04-database-kernels-turn-cxl-storage-into-typed-database-services",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The CXL-storage path is promising, but the entry explicitly treats device-side preparation as an early warning rather than immediate placement support.",
    },
    (
        "2026-06-04-database-kernels-turn-cxl-storage-into-typed-database-services",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "The descriptor transfer assumes cooperative database software and specialized storage hardware beyond the first CPU/GPU/NVMe target.",
    },
    (
        "2026-06-04-gpu-b-trees-need-warp-shaped-nodes-and-restart-on-contention-updates",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Warp-shaped resident indexes are viable only behind database visibility rules and generation-scoped invalidation.",
    },
    (
        "2026-06-04-gpu-b-trees-need-warp-shaped-nodes-and-restart-on-contention-updates",
        "same_shape_microbatching",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts lane-independent lookup with warp-cooperative lookup, which directly supports same-shape micro-batching.",
    },
    (
        "2026-06-04-smf-schedules-hot-conflicts-before-concurrency-control-sees-them",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The sampled-hint scheduling evidence needs workload evaluation before becoming a deterministic hot-write template.",
    },
    (
        "2026-06-04-smf-schedules-hot-conflicts-before-concurrency-control-sees-them",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts serial timestamp racing with predicted hot-key queues, which supports explicit conflict ordering.",
    },
    (
        "2026-06-04-tile-based-gpu-integer-compression-keeps-decode-inside-the-route",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Compressed generations are useful only when refresh cost, retired bytes, invalidation frequency, and p95 impact stay bounded.",
    },
    (
        "2026-06-04-tile-based-gpu-integer-compression-keeps-decode-inside-the-route",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Compressed GPU segment refresh must be measured at batch boundaries before relying on WAL/MVCC invalidation safety.",
    },
    (
        "2026-06-04-delilah-exposes-the-real-cost-of-programmable-storage-offload",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Programmable-storage placement needs setup, throughput, verification, recovery, and byte-savings measurements.",
    },
    (
        "2026-06-04-delilah-exposes-the-real-cost-of-programmable-storage-offload",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The paper omits SQL operators, MVCC, recovery, GPU handoff, and multi-tenant admission, so descriptor transfer needs validation.",
    },
    (
        "2026-06-04-delilah-exposes-the-real-cost-of-programmable-storage-offload",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Storage-side filtering supports owner routing only when the route has explicit byte-range, cache-maintenance, and result contracts.",
    },
    (
        "2026-06-04-primo-removes-2pc-by-making-commit-conflict-free-before-it-starts",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Asynchronous group commit needs crash-point validation before it can satisfy WAL-before-visibility.",
    },
    (
        "2026-06-04-primo-removes-2pc-by-making-commit-conflict-free-before-it-starts",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The separate execution and publication frontier is explicitly presented as the benchmarkable route-root hook.",
    },
    (
        "2026-06-04-primo-removes-2pc-by-making-commit-conflict-free-before-it-starts",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Distributed commit throughput results need GPU DB hot-write and latency evaluation before conflict-ordering transfer.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-separate-execution-and-publication-frontiers",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis names telemetry and crash/recovery tests as the gate for execution-frontier versus publication-frontier route roots.",
    },
    (
        "2026-06-04-cross-paper-synthesis-hot-routes-need-separate-execution-and-publication-frontiers",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fallback routes are valid only when stale, rolled-back, invalidated, or over-budget publication frontiers are rejected.",
    },
    (
        "2026-06-04-rtindex-maps-resident-indexes-onto-rtx-bvh-traversal",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "RTIndex assumes resident data and large batches, which cautions against direct p50-sensitive OLTP snapshot adoption.",
    },
    (
        "2026-06-04-rtindex-maps-resident-indexes-onto-rtx-bvh-traversal",
        "same_shape_microbatching",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Large-batch resident traversal evidence warns against assuming the same shape fits short OLTP lookup batches automatically.",
    },
    (
        "2026-06-04-bght-makes-gpu-hash-indexes-a-probe-budgeted-route-not-just-a-lookup-primitive",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Visible-delta correctness and stale-hit filtering are proof gates before resident hash descriptors can be trusted.",
    },
    (
        "2026-06-04-bght-makes-gpu-hash-indexes-a-probe-budgeted-route-not-just-a-lookup-primitive",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hash-route fallback needs miss-rate, batch-depth, memory-pressure, and rebuild-economics measurements.",
    },
    (
        "2026-06-04-cross-paper-synthesis-resident-indexes-need-route-envelopes-and-rebuild-economics",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Resident indexes must expose probe, load-factor, hit/miss, and construction-risk facts before the planner can trust snapshots.",
    },
    (
        "2026-06-04-cross-paper-synthesis-resident-indexes-need-route-envelopes-and-rebuild-economics",
        "resource_dag_scheduling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Scarce warm resources should remain scheduled only when decision-specific history proves that they pay for themselves.",
    },
    (
        "2026-06-04-cooperative-memory-management-turns-cache-pressure-into-an-admission-choice",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained lookups are safe only when unrelated scan, refresh, allocation, or memory-return latency cannot leak into the route.",
    },
    (
        "2026-06-04-cooperative-memory-management-turns-cache-pressure-into-an-admission-choice",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Temporary-memory admission needs fixed-partition, cooperative-eviction, cooperative-spill, and reject-under-SLO benchmarks.",
    },
    (
        "2026-06-04-cooperative-memory-management-turns-cache-pressure-into-an-admission-choice",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The read-mostly workload leaves dirty-page writeback, WAL/checkpoint interaction, and update latency unevaluated.",
    },
    (
        "2026-06-04-schedule-first-concurrency-turns-hot-key-contention-into-an-admission-problem",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Schedule-first queues are an alternative to abort/retry templates for hot conflicting operations.",
    },
    (
        "2026-06-04-schedule-first-concurrency-turns-hot-key-contention-into-an-admission-problem",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "SMF conflict-cost scheduling needs GPU DB workload tests before becoming the conflict-ordering policy.",
    },
    (
        "2026-06-04-schedule-first-concurrency-turns-hot-key-contention-into-an-admission-problem",
        "resource_dag_scheduling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Makespan-driven scheduling requires resource-DAG measurements under GPU DB arrival and conflict patterns.",
    },
    (
        "2026-06-04-allocator-behavior-is-part-of-the-query-route-contract",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The CPU analytical allocator scope cautions against inferring retained GPU snapshot behavior directly.",
    },
    (
        "2026-06-04-stage-makes-route-prediction-a-latency-budgeted-hierarchy-not-one-model",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Hierarchical learned prediction is useful only when uncertainty and expected route duration justify the added inference cost.",
    },
    (
        "2026-06-04-stage-makes-route-prediction-a-latency-budgeted-hierarchy-not-one-model",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Stage's Redshift analytical scope cautions against direct retained GPU snapshot transfer to WAL-governed OLTP/HTAP routes.",
    },
    (
        "2026-06-04-stage-makes-route-prediction-a-latency-budgeted-hierarchy-not-one-model",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner routing is useful only when queue and reservation telemetry predicts the execution window, not just admission-time load.",
    },
    (
        "2026-06-04-stage-makes-route-prediction-a-latency-budgeted-hierarchy-not-one-model",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Expensive global policy metadata is justified only when uncertainty and expected route duration outweigh reclamation and hot-path costs.",
    },
    (
        "2026-06-04-hyperion-treats-gpu-storage-access-as-a-schedulable-pipeline",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cold-chunk overlap must prove retained lookup p99 stays within budget under PCIe/NVMe and staging-buffer pressure.",
    },
    (
        "2026-06-04-manycore-file-systems-expose-hidden-cold-tier-contention",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL and checkpoint segment choices need filesystem contention, fsync, and recovery-scan measurements before adoption.",
    },
    (
        "2026-06-04-compound-gpu-pipelines-trade-materialization-for-explicit-reduction-pressure",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Compound fused kernels are a promising same-shape batch design but still require lookup, projection, and scatter benchmarks.",
    },
    (
        "2026-06-04-compound-gpu-pipelines-trade-materialization-for-explicit-reduction-pressure",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route telemetry must be benchmarked to prove it separates atomic contention from transfer bottlenecks.",
    },
    (
        "2026-06-04-lotus-keeps-partition-owners-single-threaded-but-multiplexes-multi-partition-waits",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Lotus-like logging must be measured for retained invalidation recovery without scanning unrelated GPU-resident artifacts.",
    },
    (
        "2026-06-04-lotus-keeps-partition-owners-single-threaded-but-multiplexes-multi-partition-waits",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Sequencer-local publication batches need crash and visibility tests before route roots can rely on them.",
    },
    (
        "2026-06-04-fluid-co-processing-should-offload-narrow-pruning-not-whole-queries-by-default",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The stress cue describes GPU-friendly Bloom pruning, which supports narrow retained snapshot routes rather than cautioning against them.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-placement-needs-stable-pressure-before-movement",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Placement changes require route-pressure, stable-window, cross-owner stealing, and CPU-fallback measurements.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-placement-needs-stable-pressure-before-movement",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner movement and stealing need separate memory-heavy and CPU-heavy pressure measurements before policy adoption.",
    },
    (
        "2026-06-04-namespace-metadata-is-a-route-cache-design-problem",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-cache coherence policies need central-owner, per-worker generation-cache, and replicated-descriptor benchmarks.",
    },
    (
        "2026-06-04-namespace-metadata-is-a-route-cache-design-problem",
        "effective_session_counting",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Prefix invalidation can create a renewal storm at 1M logical sessions, cautioning against naive session-scale caching.",
    },
    (
        "2026-06-04-semantic-repair-beats-full-occ-restart-when-conflict-scope-is-small",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner routing needs memory-footprint, cache-reuse, queue-time, p99, and false-invalidation benchmarks.",
    },
    (
        "2026-06-04-view-serializability-does-not-buy-extra-safe-mvcc-route-templates-for-rc-si-ssi",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Mixed-isolation templates require owner-queue, validation-work, abort/retry, and p99 measurements.",
    },
    (
        "2026-06-04-view-serializability-does-not-buy-extra-safe-mvcc-route-templates-for-rc-si-ssi",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The static-analysis scope lacks production SQL and throughput evaluation, so descriptor transfer needs benchmarking.",
    },
    (
        "2026-06-04-view-serializability-does-not-buy-extra-safe-mvcc-route-templates-for-rc-si-ssi",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue contrasts single-version conflict order with explicit snapshot-visible dependencies, supporting frontier vectors.",
    },
    (
        "2026-06-04-hybrid-benchmarks-must-put-fresh-analytical-reads-inside-the-transaction-path",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "OLxPBench defines the fresh-analytical-in-transaction workload shape that should gate HTAP freshness routing.",
    },
    (
        "2026-06-04-hybrid-benchmarks-must-put-fresh-analytical-reads-inside-the-transaction-path",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The distributed JDBC benchmark scope cautions against directly inferring PostgreSQL-wire retained GPU snapshot behavior.",
    },
    (
        "2026-06-04-hybrid-benchmarks-must-put-fresh-analytical-reads-inside-the-transaction-path",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The GPU DB mapping directly connects the benchmark shape to retained snapshots and route descriptors.",
    },
    (
        "2026-06-04-hybrid-benchmarks-must-put-fresh-analytical-reads-inside-the-transaction-path",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The distributed JDBC benchmark leaves PostgreSQL wire multiplexing and logical-session scale as required measurements.",
    },
    (
        "2026-06-04-hybrid-benchmarks-must-put-fresh-analytical-reads-inside-the-transaction-path",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The MemSQL/TiDB contrast motivates a tier-placement matrix across row, CPU column, GPU, cold-transfer, and fallback paths.",
    },
    (
        "2026-06-04-hybrid-benchmarks-must-put-fresh-analytical-reads-inside-the-transaction-path",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Queue handoff delay and hidden invalidation cost need benchmarks inside the fresh transaction path.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-needs-workload-shaped-proof-repair-and-latency-gates",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "HTAP freshness should advance through workload-shaped proof and latency gates rather than another standalone scan benchmark.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-safety-needs-workload-shaped-proof-repair-and-latency-gates",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cache placement needs workload-shaped proof, repair, and latency gates before movement policy adoption.",
    },
    (
        "2026-06-04-semantic-data-classes-can-make-concurrency-admission-route-aware",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hot-write templates require proof gates comparing optimistic retry, semantic repair, ownership queues, and escrow reservation.",
    },
    (
        "2026-06-04-semantic-data-classes-can-make-concurrency-admission-route-aware",
        "vector_credit_admission",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Escrow-like admission is valid only for numeric constrained commutative updates with explicit preconditions.",
    },
    (
        "2026-06-04-production-workload-management-needs-cheap-predictions-plus-hard-guardrails",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner bundling is valid only when descriptors carry predicted queue, mutation-owner, refresh, and response-buffer pressure.",
    },
    (
        "2026-06-04-production-workload-management-needs-cheap-predictions-plus-hard-guardrails",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fallback policy is valid only when descriptors expose predicted queue, GPU, pinned-buffer, resident-age, and fallback costs.",
    },
    (
        "2026-06-04-mixed-isolation-can-be-a-route-contract-not-just-a-session-default",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Mixed-isolation route templates require owner-queue, retained-read throughput, abort/retry, and p95/p99 validation.",
    },
    (
        "2026-06-04-mixed-isolation-can-be-a-route-contract-not-just-a-session-default",
        "effective_session_counting",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The abstract workload model cautions against inferring arbitrary SQL behavior at million-session scale.",
    },
    (
        "2026-06-04-gpu-learned-indexes-need-batch-shaped-residency-contracts",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Learned-index batching needs throughput, latency, launch-count, CPU last-mile, result-byte, and queue-wait measurements.",
    },
    (
        "2026-06-04-detock-resolves-ordering-cycles-instead-of-aborting-them",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Opportunistic multi-owner ordering requires SCC stability, delayed-command, queue-wait, retry, and tail-latency measurements.",
    },
    (
        "2026-06-04-detock-resolves-ordering-cycles-instead-of-aborting-them",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Detock rewrites stable ordering cycles instead of relying on globally ordered hot-write templates.",
    },
    (
        "2026-06-04-detock-resolves-ordering-cycles-instead-of-aborting-them",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Detock's replicated conflict-graph repair is an alternative to per-route dependency witness admission.",
    },
    (
        "2026-06-04-pgm-gives-learned-indexes-a-bounded-route-certificate",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Learned index advice is valid only when exposed through bounded route certificates rather than opaque model output.",
    },
    (
        "2026-06-04-cross-paper-synthesis-fast-routes-now-need-certificates-not-hints",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-root publication needs retained lookup, refresh, stale-visibility, conflict, and DDL invalidation benchmarks.",
    },
    (
        "2026-06-04-cross-paper-synthesis-fast-routes-now-need-certificates-not-hints",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Multi-owner conflict, invalidation, and refresh handoff require explicit certificate benchmarks before adoption.",
    },
    (
        "2026-06-04-cross-paper-synthesis-fast-routes-now-need-certificates-not-hints",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Conservative owner fallback must be benchmarked against CPU and GPU PGM retained point-lookup routes.",
    },
    (
        "2026-06-04-gpu-query-concurrency-as-a-resource-fitting-problem",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner bundling is valid only when stream packing preserves ordering, visibility, result-scatter ownership, and measured certificate fields.",
    },
    (
        "2026-06-04-gpu-query-concurrency-as-a-resource-fitting-problem",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Explicit retained-route co-scheduling is presented as an alternative to hoping concurrent CUDA streams overlap usefully.",
    },
    (
        "2026-06-04-snapshot-reconstruction-as-an-optimizable-route",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot reconstruction placement needs native/cold-tier benchmarks with retained-generation, old-byte, rebuild, and fallback telemetry.",
    },
    (
        "2026-06-04-snapshot-reconstruction-as-an-optimizable-route",
        "isolation_trace_oracle",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Snapshot reconstruction evidence transfers only under full SI/MVCC assumptions focused on read-path reconstruction.",
    },
    (
        "2026-06-04-gpu-sharing-should-be-measured-not-guessed",
        "same_shape_microbatching",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The evidence says same-shape batching alone is insufficient without measured route-pair compatibility and interference constraints.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-live-control-loops",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless cue is a research-queue priority note; the entry still identifies metadata and tier placement as live control-loop gaps.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-live-control-loops",
        "same_shape_microbatching",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Live route-class compatibility telemetry is proposed as an alternative to relying only on static same-shape batching.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-live-control-loops",
        "resource_dag_scheduling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "GPU scheduling policy needs route-certificate tests, co-run compatibility measurements, and latency-budget gates.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-live-control-loops",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fast measured routes are valid only when snapshot certificates prove the requested read boundary.",
    },
    (
        "2026-06-04-x-ssd-moves-wal-propagation-into-the-storage-device",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL credit admission needs COPY/INSERT microbenchmarks and durability-counter proof before visibility publication.",
    },
    (
        "2026-06-04-correct-remote-durability-depends-on-the-whole-path",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Replayable ingress rings are valid only if recovery can interpret them without volatile context and recycling is bounded.",
    },
    (
        "2026-06-04-correct-remote-durability-depends-on-the-whole-path",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Dependency publication is valid only when the route certificate proves the selected remote durability policy is complete.",
    },
    (
        "2026-06-04-correct-remote-durability-depends-on-the-whole-path",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Whole-path durability requirements caution against treating descriptor cleanup as an API-local reclamation problem.",
    },
    (
        "2026-06-04-bindex-turns-predicate-scans-into-a-memory-budgeted-route",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "BinDex-style route choice is valid only when memory is sufficient and selection dominates transfer, queue, visibility, and join costs.",
    },
    (
        "2026-06-04-bindex-turns-predicate-scans-into-a-memory-budgeted-route",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained predicate indexes are useful only where selectivity and update behavior avoid fragile random-access tree costs.",
    },
    (
        "2026-06-04-bindex-turns-predicate-scans-into-a-memory-budgeted-route",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Segment-local tier placement is valid only if bitmap count does not make route planning or result merging dominate.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-memory-budgeted-predicate-routes",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Durability counters and read-route memory budgets need benchmark gates before accelerator shortcuts publish visibility.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-memory-budgeted-predicate-routes",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback must be measured in the predicate-route matrix across selectivity, update rate, HBM pressure, and queue depth.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-memory-budgeted-predicate-routes",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained GPU predicate routes require resident scan, sketch, bitmap/refine, GPU tree, and fallback benchmarks.",
    },
    (
        "2026-06-04-cross-paper-synthesis-publication-certificates-need-local-staging-and-explicit-durability-clocks",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless cue is a source-priority note; the synthesis still preserves multi-tier placement as a relevant publication-certificate gap.",
    },
    (
        "2026-06-04-price-separates-portable-cardinality-priors-from-database-specific-tuning",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "PRICE-style priors are useful only when join-condition features and route-specific costs stay visible to the optimizer.",
    },
    (
        "2026-06-04-price-separates-portable-cardinality-priors-from-database-specific-tuning",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Transferable feature summaries are an alternative to training a new opaque model for every database deployment.",
    },
    (
        "2026-06-04-price-separates-portable-cardinality-priors-from-database-specific-tuning",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of cue concerns model training; bounded feature descriptors still support explicit route metadata.",
    },
    (
        "2026-06-04-price-separates-portable-cardinality-priors-from-database-specific-tuning",
        "effective_session_counting",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The entry explicitly warns against per-session learned state at the 1M logical-session target.",
    },
    (
        "2026-06-04-adaptive-htap-makes-freshness-a-resource-scheduling-input",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hardware accelerators are future work, so GPU tier placement needs direct HBM, transfer, and launch measurements.",
    },
    (
        "2026-06-04-cd-search-makes-gpu-co-scheduling-a-classified-resource-partition-problem",
        "htap_freshness_router",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Freshness co-scheduling is valid only when measured resource pairs preserve p95 latency and write-path freshness.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-should-combine-freshness-estimates-and-measured-resourc",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Freshness estimates and dirty-frontier coverage need route-certificate measurements before policy adoption.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-should-combine-freshness-estimates-and-measured-resourc",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless clause is source-priority guidance; the retained certificate evidence still supports tier-placement routing.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-should-combine-freshness-estimates-and-measured-resourc",
        "resource_dag_scheduling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Measured GPU resource classes must be validated before they can drive resource-DAG co-scheduling.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-should-combine-freshness-estimates-and-measured-resourc",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot-generation and dirty-frontier fields need measured route-certificate validation before transfer.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-should-combine-freshness-estimates-and-measured-resourc",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fallback is valid only when the recorded certificate explains stale-snapshot rejection and route choice.",
    },
    (
        "2026-06-04-mako-decouples-fast-speculative-certification-from-slow-durable-replication",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Speculative writes may feed retained snapshots only when the durable visibility boundary proves client-visible safety.",
    },
    (
        "2026-06-04-mako-decouples-fast-speculative-certification-from-slow-durable-replication",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Partition-local append lanes and visibility watermarks require retained-snapshot coverage benchmarks.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-durability-shape-and-tier-state",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The three-state publication path requires a benchmark before GPU-resident retained snapshots can rely on it.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-durability-shape-and-tier-state",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner queue and visibility trace fields are explicitly benchmark inputs rather than established owner policy.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-durability-shape-and-tier-state",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-root publication needs certificate benchmarks covering tier source, snapshot generation, and durable visibility.",
    },
    (
        "2026-06-04-modern-nvme-makes-cold-tier-i-o-a-hot-path-scheduling-problem",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Cold-fetched data can feed retained GPU snapshots only when it proves the requested visibility boundary.",
    },
    (
        "2026-06-04-modern-nvme-makes-cold-tier-i-o-a-hot-path-scheduling-problem",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The million-TPC-C result weakens logging and isolation, so it warns against direct conflict-ordering transfer.",
    },
    (
        "2026-06-04-modern-nvme-makes-cold-tier-i-o-a-hot-path-scheduling-problem",
        "effective_session_counting",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Session scaling is valid only if cold-tier queues and memory growth remain bounded and visible.",
    },
    (
        "2026-06-04-write-behind-logging-makes-durability-a-visibility-gap-contract",
        "wal_before_visibility",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Write-behind logging proposes compact visibility certificates as an alternative to replaying a large physical log.",
    },
    (
        "2026-06-04-write-behind-logging-makes-durability-a-visibility-gap-contract",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "GPU-resident snapshots may include gap-era rows only when the route certificate marks and enforces invisibility.",
    },
    (
        "2026-06-04-write-behind-logging-makes-durability-a-visibility-gap-contract",
        "log_structured_warm_tier",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Changed-state-first persistence is an alternative warm-tier logging shape to tuple after-image logging.",
    },
    (
        "2026-06-04-write-behind-logging-makes-durability-a-visibility-gap-contract",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Clean and uncertain timestamp ranges are an alternative frontier representation to ordinary WAL-derived vectors.",
    },
    (
        "2026-06-04-write-behind-logging-makes-durability-a-visibility-gap-contract",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Compact durability certificates are presented as an alternative publication root for future persistent tiers.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-is-now-a-route-certificate-dimension",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Tier placement is valid only when freshness and visibility gaps are explicit certificate fields.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-is-now-a-route-certificate-dimension",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fast local freshness is safe only when the route proves convergence back to a durable global prefix.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-is-now-a-route-certificate-dimension",
        "htap_freshness_router",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Freshness routing is valid only when transient local order and stable publication boundaries are both proved.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-is-now-a-route-certificate-dimension",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback needs controlled refresh-lag benchmarks before certificate decisions are trusted.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-is-now-a-route-certificate-dimension",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained GPU reads require refresh-lag and crash/restart benchmarks across stable and delta-merge routes.",
    },
    (
        "2026-06-04-mvrc-robustness-turns-route-isolation-into-a-static-template-property",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Static route certification must measure template retirement, fallback, abort/retry, and p99 behavior before adoption.",
    },
    (
        "2026-06-04-mvrc-robustness-turns-route-isolation-into-a-static-template-property",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The robustness claim is backed by proof-of-concept benchmarks and needs GPU DB conflict-order validation before adoption.",
    },
    (
        "2026-06-04-three-tree-makes-intermediate-memory-a-first-class-buffer-tier",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Page movement can transfer only if each movement is tied to immutable publication and WAL-before-visibility ordering.",
    },
    (
        "2026-06-04-three-tree-makes-intermediate-memory-a-first-class-buffer-tier",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Intermediate-buffer movement is valid only when route roots publish immutable snapshot and resident-state boundaries.",
    },
    (
        "2026-06-04-schedule-first-oltp-turns-hot-key-conflict-order-into-an-admission-primitive",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hot-key scheduling has published evaluations, but GPU DB deterministic templates still need workload-specific latency and abort benchmarks.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-now-need-scheduling-intent",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The bounded active-request window is proposed as an alternative to global reordering for dependency control.",
    },
    (
        "2026-06-04-dint-keeps-frequent-transaction-steps-inside-the-kernel-datapath",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Kernel datapath admission needs p50/p99 latency and false-delay measurements under hot-key skew before it can guide session counting.",
    },
    (
        "2026-06-04-dint-keeps-frequent-transaction-steps-inside-the-kernel-datapath",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "DINT keeps fast operations in eBPF maps instead of using ordinary user-space descriptor traffic.",
    },
    (
        "2026-06-04-runtime-conflicts-make-transaction-order-a-measurable-resource",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Runtime-conflict ordering remains coupled to the underlying CC protocol and needs GPU DB prototype measurements.",
    },
    (
        "2026-06-04-runtime-conflicts-make-transaction-order-a-measurable-resource",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Runtime-conflict templates are useful only when probe cost stays below saved abort, fallback, and refresh cost.",
    },
    (
        "2026-06-04-cross-paper-synthesis-admission-needs-active-window-state",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The admission design is explicitly routed through a simulator benchmark across edge classification, priority, and deferment policies.",
    },
    (
        "2026-06-04-cardood-treats-route-estimator-drift-as-a-first-class-optimizer-risk",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The test-query phrase describes estimator drift evaluation, while the evidence still supports learned optimizer risk tracking.",
    },
    (
        "2026-06-04-cardood-treats-route-estimator-drift-as-a-first-class-optimizer-risk",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback decisions need explicit cross-group evaluation across resident state, freshness, lookup shape, queue pressure, and tenants.",
    },
    (
        "2026-06-04-cardood-treats-route-estimator-drift-as-a-first-class-optimizer-risk",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Distribution-alignment methods are optimizer-side alternatives to replacing execution with deterministic hot-write templates.",
    },
    (
        "2026-06-04-cockroachdb-makes-transaction-routing-an-ownership-problem",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "CockroachDB's distributed CPU/RocksDB/Raft setting cautions against directly inferring retained GPU snapshot behavior.",
    },
    (
        "2026-06-04-cockroachdb-makes-transaction-routing-an-ownership-problem",
        "owner_ring_bundling",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Adaptive ownership movement can improve average latency while worsening tail latency and fallback unpredictability.",
    },
    (
        "2026-06-04-cockroachdb-makes-transaction-routing-an-ownership-problem",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Freshness routing needs benchmark evidence for snapshot age, owner contention, invalidation races, and freshness-sensitive fallbacks.",
    },
    (
        "2026-06-04-geogauss-batches-replica-consistency-without-per-transaction-coordination",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Commit-generation publication is promising but needs mutation, refresh, invalidation, and read-snapshot release measurements.",
    },
    (
        "2026-06-04-geogauss-batches-replica-consistency-without-per-transaction-coordination",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Same-generation write merge policies require throughput, latency, abort/retry, and WAL flush grouping benchmarks.",
    },
    (
        "2026-06-04-cross-paper-synthesis-publish-certified-generations-not-mutable-shortcuts",
        "cpu_fallback_policy",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The unless clause is source-priority guidance; the synthesis still preserves GPU/CPU route execution as relevant evidence.",
    },
    (
        "2026-06-04-cxl-memory-should-be-placed-by-object-behavior-not-by-capacity-alone",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Object-behavior placement needs direct resident-scan, lookup, visibility-check, temp-table, and response-buffer tests.",
    },
    (
        "2026-06-04-cxl-memory-should-be-placed-by-object-behavior-not-by-capacity-alone",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "SAP HANA on commercial CXL memory is a CPU in-memory setting, cautioning against direct retained-GPU snapshot transfer.",
    },
    (
        "2026-06-04-cxl-memory-should-be-placed-by-object-behavior-not-by-capacity-alone",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CXL expansion and shared-memory failover evidence needs GPU DB session-scale evaluation before shaping session counting.",
    },
    (
        "2026-06-04-q-store-turns-transaction-execution-into-ordered-operation-queues",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Ordered operation queues need throughput, latency, abort/retry, owner-depth, and visibility-lag benchmarks for hot-write templates.",
    },
    (
        "2026-06-04-q-store-turns-transaction-execution-into-ordered-operation-queues",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshots require proof-gate validation that mutation fragments are observed before newer read snapshots publish.",
    },
    (
        "2026-06-04-grasp-makes-imperfect-route-logs-useful-for-cardinality-estimates",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Composable per-table primitive models are presented as an alternative to one global or per-template learned estimator.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-tier-schedule-and-estimate-provenance",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Admission telemetry remains benchmarkable rather than proven, especially for queueing, eviction, and storage-tier scheduling.",
    },
    (
        "2026-06-04-quecc-turns-hot-transactions-into-planned-priority-queues",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner queue certification needs WAL-before-visibility, invalidation-order, and micro-batch trigger benchmarks.",
    },
    (
        "2026-06-04-cxl-memory-needs-workload-shaped-placement-not-capacity-only-tiering",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Workload-shaped placement requires prototype route-certificate measurements across access patterns and tier classes.",
    },
    (
        "2026-06-04-cxl-memory-needs-workload-shaped-placement-not-capacity-only-tiering",
        "effective_session_counting",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The CXL placement evidence warns against unbounded per-session and per-route state at the 1M logical-session target.",
    },
    (
        "2026-06-04-orthrus-separates-contention-control-from-transaction-execution",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-message queue layouts need proof-gate validation that unplanned access restarts or falls back before visibility changes.",
    },
    (
        "2026-06-04-orthrus-separates-contention-control-from-transaction-execution",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "ORTHRUS is a contention-control prototype rather than a full DBMS, so hot-write template transfer needs database-path benchmarks.",
    },
    (
        "2026-06-04-strife-turns-contention-into-batch-time-conflict-free-lanes",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue contrasts logical sessions with execution threads, while the evidence supports compact owner-drained request queues.",
    },
    (
        "2026-06-04-cross-paper-synthesis-active-window-certificates-should-choose-the-write-lane",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner queue and priority-lane interactions need active-window benchmarks before choosing the write lane.",
    },
    (
        "2026-06-04-cross-paper-synthesis-active-window-certificates-should-choose-the-write-lane",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CXL placement and object-family tier classes are named measurement inputs for active-window certificate routing.",
    },
    (
        "2026-06-04-cross-paper-synthesis-active-window-certificates-should-choose-the-write-lane",
        "db_owned_cold_objects",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cold-object participation in active-window lane choice requires measured tier classes and certificate validation.",
    },
    (
        "2026-06-04-cross-paper-synthesis-active-window-certificates-should-choose-the-write-lane",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Conflict-graph write-lane selection is explicitly a measurement gate before runtime ordering policy adoption.",
    },
    (
        "2026-06-04-t-part-partitions-transactions-then-pushes-writes-forward",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "T-Part partitions pending transactions from a dependency graph instead of using per-route dependency witness admission.",
    },
    (
        "2026-06-04-btrblocks-chooses-compression-per-block-by-measured-decode-value",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The compression evidence transfers only through measured tier metrics for bytes, decode cost, and route time.",
    },
    (
        "2026-06-04-btrblocks-chooses-compression-per-block-by-measured-decode-value",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner handoff for decoded, compressed, or pinned batches depends on measured route cost rather than a proven queue policy.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-tier-schedule-and-codec-facts",
        "htap_freshness_router",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Aurora-style redo and consistency boundaries are presented as an alternative to implicit freshness from page state.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-tier-schedule-and-codec-facts",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained GPU snapshot route certificates need commit-to-readable and codec-refresh benchmarks before adoption.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-tier-schedule-and-codec-facts",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Ordered redo and consistency boundaries are framed as an alternative proof source to implicit page-state dependency tracking.",
    },
    (
        "2026-06-04-fpsi-makes-freshness-a-first-contact-snapshot-policy",
        "owner_ring_bundling",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "FPSI warns against treating snapshot generation as one global scalar across multiple owner and worker domains.",
    },
    (
        "2026-06-04-fpsi-makes-freshness-a-first-contact-snapshot-policy",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "FPSI cautions that retained snapshot freshness must be chosen at first contact rather than guessed from a single generation.",
    },
    (
        "2026-06-04-memory-centric-databases-make-pooled-memory-a-query-route-resource",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Pooled-memory routes require proof gates for canonical WAL/MVCC recovery source and deterministic invalidation paths.",
    },
    (
        "2026-06-04-cxl-pooling-is-a-costed-route-not-transparent-memory",
        "stable_handle_indirection",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Stable handles may cross into CXL or remote memory only when measurements prove the path is not correctness-critical hot state.",
    },
    (
        "2026-06-04-cross-paper-synthesis-future-tiers-need-local-hot-remote-cold-contracts",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Future-tier placement needs route-certificate prototypes and far-memory sensitivity tests before policy adoption.",
    },
    (
        "2026-06-04-cross-paper-synthesis-future-tiers-need-local-hot-remote-cold-contracts",
        "htap_freshness_router",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "First-contact freshness certificates are presented as an alternative to guessing route freshness later.",
    },
    (
        "2026-06-04-cross-paper-synthesis-future-tiers-need-local-hot-remote-cold-contracts",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Active memory leases must prove idle logical sessions reserve no tier payload before shaping session-count policy.",
    },
    (
        "2026-06-04-demystifying-cxl-memory-with-genuine-cxl-ready-systems-and-devices",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "CXL placement is valid only when route decisions account for access mode, cache behavior, promotion bytes, and local control state.",
    },
    (
        "2026-06-04-demystifying-cxl-memory-with-genuine-cxl-ready-systems-and-devices",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The CXL evaluation omits WAL, MVCC, recovery, and GPU transfer paths, so WAL-before-visibility needs direct validation.",
    },
    (
        "2026-06-04-homa-makes-receiver-admission-a-latency-control-surface",
        "deficit_fairness",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Homa's priority thresholds were precomputed, so deficit fairness needs online message-size and latency benchmarks.",
    },
    (
        "2026-06-04-homa-makes-receiver-admission-a-latency-control-surface",
        "effective_session_counting",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Session counting is valid only if memory and queue slots scale with outstanding fragments rather than fan-out width.",
    },
    (
        "2026-06-04-homa-makes-receiver-admission-a-latency-control-surface",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-priority lanes need p50/p99, buffered-byte, overload, and fallback measurements before placement policy transfer.",
    },
    (
        "2026-06-04-octopus-uses-semantic-fast-paths-with-gpu-dag-fallback",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "contradicts",
        "relation_review_note": "Octopus fallback DAG order and always-success compensation conflict with ordinary SQL conflict-ordering assumptions.",
    },
    (
        "2026-06-04-cross-paper-synthesis-route-certificates-need-placement-credits-and-semantic-proof",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained snapshots may move across future tiers only when route latency and receiver-owned credit proofs allow it.",
    },
    (
        "2026-06-04-hybench-frames-htap-as-freshness-bound-mixed-pressure-not-olap-plus-oltp-in-isolation",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "HyBench frames freshness as a benchmark dimension, so the freshness router needs workload-specific measurement gates.",
    },
    (
        "2026-06-04-f1-lightning-turns-htap-into-safe-time-routing-over-a-replicated-analytical-lsm",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Resident read descriptors are safe only when the requested timestamp or freshness SLO stays inside the safe window.",
    },
    (
        "2026-06-04-d2pc-decentralizes-commit-coordination-to-shorten-conflict-windows",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "D2PC-style templates assume an existing OCC or 2PL store with replicated transaction logs and votes.",
    },
    (
        "2026-06-04-d2pc-decentralizes-commit-coordination-to-shorten-conflict-windows",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Dependency witnesses transfer only if prepared ordering and WAL reservation reduce hidden owner hold time.",
    },
    (
        "2026-06-04-d2pc-decentralizes-commit-coordination-to-shorten-conflict-windows",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "D2PC's geo-distributed design warns against inferring single-node GPU descriptor reclamation directly.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-safe-routes-and-short-commit-windows",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Freshness-safe routing is explicitly framed as a HyBench-style workload dimension that needs measurement.",
    },
    (
        "2026-06-04-cross-paper-synthesis-freshness-safe-routes-and-short-commit-windows",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Shorter commit windows must be measured against owner hold time before changing owner-ring boundaries.",
    },
    (
        "2026-06-04-md-mvcc-makes-schema-metadata-snapshot-visible-instead-of-globally-blocking",
        "db_owned_cold_objects",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Versioned schema metadata is an alternative control-object shape to a single mutable database-owned object.",
    },
    (
        "2026-06-04-md-mvcc-makes-schema-metadata-snapshot-visible-instead-of-globally-blocking",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Snapshot-visible metadata versions are presented instead of in-place cache mutation for placement authority.",
    },
    (
        "2026-06-04-md-mvcc-makes-schema-metadata-snapshot-visible-instead-of-globally-blocking",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The entry warns that global metadata overwrite can break long readers unless MVCC frontiers protect schema state.",
    },
    (
        "2026-06-04-md-mvcc-makes-schema-metadata-snapshot-visible-instead-of-globally-blocking",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Old route-version and payload-reference retirement is explicitly a long-snapshot proof gate.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-needs-compact-frontiers-and-sampled-queues",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Session counting needs the proposed 1M-idle-session admission harness before policy adoption.",
    },
    (
        "2026-06-05-gpu-multitasking-needs-explicit-compute-memory-and-fault-isolation-contracts",
        "isolation_trace_oracle",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The GPU multitasking paper maps requirements but lacks database speedup and isolation measurements.",
    },
    (
        "2026-06-05-tidb-makes-htap-freshness-a-consensus-derived-route-property",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The TiDB transfer explicitly requires freshness-lag telemetry and CH-benCHmark-derived HTAP gates.",
    },
    (
        "2026-06-05-tidb-makes-htap-freshness-a-consensus-derived-route-property",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "CPU fallback is needed only when freshness catch-up is too slow for tail-sensitive GPU lookup routes.",
    },
    (
        "2026-06-05-cross-paper-synthesis-gpu-htap-routes-need-freshness-frontiers-plus-resource-contracts",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Fresh analytical routes depend on source-log frontiers rather than an unqualified replica or cache hit.",
    },
    (
        "2026-06-05-fw-kv-improves-psi-freshness-with-version-access-metadata",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Hot-write templates are safe only when route-token partitions reject incompatible generations or fall back.",
    },
    (
        "2026-06-05-corobase-hides-pointer-stalls-by-batching-transactions-as-coroutines",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained point-read windows require p50/p99, stall, queue-wait, and throughput measurement.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-scheduling-needs-both-urgency-and-stall-hiding",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cleanup lag and route-local memory for retired snapshots and buffers are explicit measurement gates.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-scheduling-needs-both-urgency-and-stall-hiding",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The selection note is a reviewed support cue for direct cache and tier-placement coverage, not a caution relation.",
    },
    (
        "2026-06-05-waltz-moves-wal-write-serialization-into-the-zns-device",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Reserve-before-visibility WAL admission is valid only if background allocation and rewrite stalls stay bounded.",
    },
    (
        "2026-06-05-waltz-moves-wal-write-serialization-into-the-zns-device",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner serialization changes need write-vs-append tail measurements before replacing host-side coordination.",
    },
    (
        "2026-06-05-hermes-routes-near-future-transactions-to-avoid-partition-ping-pong",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Hermes uses near-future queued work to guide movement rather than only the current owner queue head.",
    },
    (
        "2026-06-05-hermes-routes-near-future-transactions-to-avoid-partition-ping-pong",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Hermes-style routing assumes read/write sets are available before deterministic execution.",
    },
    (
        "2026-06-05-cpu-prefetching-only-hides-future-tier-latency-when-fill-buffer-pressure-is-bounded",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The paper omits retained GPU snapshot correctness and protocol concurrency, so the transfer needs evaluation.",
    },
    (
        "2026-06-05-cpu-prefetching-only-hides-future-tier-latency-when-fill-buffer-pressure-is-bounded",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Descriptor generation, segment-map, MVCC-header, and host-index probes are proposed as a microbenchmark.",
    },
    (
        "2026-06-05-cpu-prefetching-only-hides-future-tier-latency-when-fill-buffer-pressure-is-bounded",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Immutable route-root metadata prefetching needs generation-table and segment-map lookup benchmarks.",
    },
    (
        "2026-06-05-count-sketch-multi-join-estimates-should-be-route-budget-inputs-not-oracle-costs",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Sketch-driven tier placement is valid only when underestimated routes have explicit HBM and fallback guards.",
    },
    (
        "2026-06-05-amac-makes-pointer-stall-hiding-a-bounded-state-machine-lane",
        "cpu_fallback_policy",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "AMAC's scaling result warns that optimized CPU fallback routes can raise tail latency for shared resources.",
    },
    (
        "2026-06-05-cross-paper-synthesis-active-windows-need-dependency-evidence",
        "dependency_witnesses",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The instead-of clause supports dependency evidence as the replacement for blind speculative retry.",
    },
    (
        "2026-06-05-cross-paper-synthesis-active-windows-need-dependency-evidence",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The active-window synthesis favors dependency and budget certificates over cost estimation alone.",
    },
    (
        "2026-06-05-cross-paper-synthesis-active-windows-need-dependency-evidence",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The WAL relation is explicitly a proof gate for replay equivalence, deterministic visibility, and bounded p99.",
    },
    (
        "2026-06-05-asynchronized-concurrency-the-secret-to-scaling-concurrent-search-data-structures",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "ASCYLIB transfer to descriptor metadata requires read/write workload evaluation before reclamation policy use.",
    },
    (
        "2026-06-05-asynchronized-concurrency-the-secret-to-scaling-concurrent-search-data-structures",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-serialized route metadata needs the proposed concurrent-map versus owner-map microbenchmark.",
    },
    (
        "2026-06-05-asynchronized-concurrency-the-secret-to-scaling-concurrent-search-data-structures",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "ASCYLIB-style retained reads are useful only if failed-search coherence traffic stays bounded.",
    },
    (
        "2026-06-05-optimistic-concurrency-with-optik",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Optik-style stale-route retry needs p95/p99 validation under invalidation and republication pressure.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-metadata-needs-read-mostly-validation-cells",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis makes bounded retired-cell memory and p99 route lookup latency explicit proof gates.",
    },
    (
        "2026-06-05-sss-scalable-key-value-store-with-external-consistent-and-abort-free-read-only-transactions",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Per-key reader queuing needs evaluation against tuple-level owner grouping and 2PC validation costs.",
    },
    (
        "2026-06-05-sss-scalable-key-value-store-with-external-consistent-and-abort-free-read-only-transactions",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Vector-clock and per-key queue metadata are valid only if compressed or grouped to bound tuple-level cost.",
    },
    (
        "2026-06-05-neomem-hardware-software-co-design-for-cxl-native-memory-tiering",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "NeoMem transfer depends on CXL-style microbenchmarks for retained snapshot buffers and route metadata.",
    },
    (
        "2026-06-05-ice-makes-dynamic-cardinality-estimates-an-updateable-index-problem",
        "cpu_fallback_policy",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "ICE motivates safer CPU routes and exact probes as alternatives to unbounded planner-estimation work.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-estimate-freshness-and-tier-freshness",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained GPU routing is valid only when visibility, schema, resident, and movement frontiers agree.",
    },
    (
        "2026-06-05-mgcrab-transaction-crabbing-for-live-migration-in-deterministic-database-systems",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner migration is plausible only with spare capacity and unsafe under saturated GPU or mutation queues.",
    },
    (
        "2026-06-05-ndp-re-architecting-datacenter-networks-and-stacks-for-low-latency",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "NDP-style response rings require incast benchmarks before setting retained micro-batch admission policy.",
    },
    (
        "2026-06-05-constant-time-snapshots-make-snapshot-handles-cheap-but-old-object-reads-pay-the-update-distance",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Old retained handles need proof that generation lifetime and HBM release remain bounded under readers.",
    },
    (
        "2026-06-05-cross-paper-synthesis-cheap-handles-still-need-bounded-payloads",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Cheap snapshot handles remain valid only when route certificates reject duplicate or stale execution.",
    },
    (
        "2026-06-05-cross-paper-synthesis-cheap-handles-still-need-bounded-payloads",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Descriptor retention is valid only with bounded payloads and generation-matched route certificates.",
    },
    (
        "2026-06-05-cross-paper-synthesis-cheap-handles-still-need-bounded-payloads",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Session counting requires snapshot-handle stress tests at 100K to 1M logical sessions.",
    },
    (
        "2026-06-05-cross-paper-synthesis-cheap-handles-still-need-bounded-payloads",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner queues are valid only if receiver-owned credits bound response, refresh, and movement payloads.",
    },
    (
        "2026-06-05-pathcas-validates-search-paths-without-full-transactional-memory",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "PathCAS-style route updates are valid only if searched descriptor nodes retain the same generation.",
    },
    (
        "2026-06-05-descriptor-reuse-turns-helping-metadata-into-a-bounded-per-worker-resource",
        "cpu_fallback_policy",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Reusable descriptors provide a bounded metadata fast path rather than expanding CPU fallback state.",
    },
    (
        "2026-06-05-socrates-separates-log-truth-page-availability-and-cheap-durable-storage",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Socrates-style residency ownership needs global-versus-partitioned owner latency and recovery benchmarks.",
    },
    (
        "2026-06-05-foundationdb-unbundles-transaction-processing-logging-and-storage-reads",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Deterministic hot-write lanes are framed as alternatives to FDB-style restart-only conflict handling.",
    },
    (
        "2026-06-05-lmsfc-learns-the-resident-multidimensional-order-not-just-the-lookup-model",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Learned resident segments need delta and invalidation tests before retained snapshot adoption.",
    },
    (
        "2026-06-05-lmsfc-learns-the-resident-multidimensional-order-not-just-the-lookup-model",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Learned multidimensional ordering needs proof-gate tests for stale layouts and warm-tier prefetch.",
    },
    (
        "2026-06-05-dodo-makes-deterministic-batch-order-scale-by-staging-retries",
        "wal_before_visibility",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Dodo-style durable safe-prefix publication is an alternative to independently admitting each conflict.",
    },
    (
        "2026-06-05-dodo-makes-deterministic-batch-order-scale-by-staging-retries",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Prepared deterministic batches provide an alternative to routing every conflict through owner queues.",
    },
    (
        "2026-06-05-dodo-makes-deterministic-batch-order-scale-by-staging-retries",
        "same_shape_microbatching",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Dodo favors ordered deterministic batches over blind retries for repeated hot-key conflicts.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-isolation-evidence-not-just-performance-evidence",
        "isolation_trace_oracle",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that isolation labels are insufficient without operation-level anomaly evidence.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-isolation-evidence-not-just-performance-evidence",
        "dependency_witnesses",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Weak-isolation routes need dependency evidence and anomaly witnesses, not just nominal isolation levels.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-isolation-evidence-not-just-performance-evidence",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route cost must include certificate creation, trace capture, anomaly analysis, and replay witnesses.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-isolation-evidence-not-just-performance-evidence",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Compact route certificates need benchmarking across schema, WAL, snapshot, layout, and isolation fields.",
    },
    (
        "2026-06-05-elle-turns-isolation-claims-into-generated-history-witnesses",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Elle-style history witnesses are an alternative proof path to complete serial-order reconstruction.",
    },
    (
        "2026-06-05-elle-turns-isolation-claims-into-generated-history-witnesses",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Retained snapshot failures should produce direct witnesses rather than only mismatch counters.",
    },
    (
        "2026-06-05-elle-turns-isolation-claims-into-generated-history-witnesses",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Elle-style history witnesses require benchmark transactions that record snapshot and WAL frontier evidence.",
    },
    (
        "2026-06-05-tmts-makes-far-memory-tiering-an-slo-controlled-admission-problem",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "TMTS tiering transfer depends on evaluated latency-pressure behavior before GPU DB placement adoption.",
    },
    (
        "2026-06-05-hdcc-mixes-deterministic-batches-and-optimistic-transactions-with-explicit-proof-points",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "HDCC conflict ordering is tied to Deneva evaluations, so GPU DB adoption needs comparable workload gates.",
    },
    (
        "2026-06-05-hdcc-mixes-deterministic-batches-and-optimistic-transactions-with-explicit-proof-points",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot-frontier use is explicitly a replay proof gate for live order and visibility reconstruction.",
    },
    (
        "2026-06-05-pace-treats-learned-route-models-as-poisonable-state",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "PACE's progressive generator-surrogate loop is an adversarial alternative to trusted learned advisors.",
    },
    (
        "2026-06-05-pace-treats-learned-route-models-as-poisonable-state",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "PACE does not study placement or GPU queues, warning against inferring tier-placement support from it.",
    },
    (
        "2026-06-05-pace-treats-learned-route-models-as-poisonable-state",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "PACE transfers as defensive mutable optimizer state rather than descriptor reclamation machinery.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-influence-control",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "The synthesis calls for placement work only after stronger networking, admission, or cold-tier evidence.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-influence-control",
        "deficit_fairness",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Deficit fairness needs route-safety benchmarks under pressure, slow-tier neighbors, and poisoning workloads.",
    },
    (
        "2026-06-05-new-storage-devices-need-route-visible-io-shape-contracts",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "HTAP freshness routing needs explicit WAL/COPY chunk and queue-depth tests before adoption.",
    },
    (
        "2026-06-05-bamboo-retires-hotspot-locks-before-transaction-commit",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Bamboo's early-retirement dependency path is an alternative to occupying descriptor-owner wait slots.",
    },
    (
        "2026-06-05-bamboo-retires-hotspot-locks-before-transaction-commit",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Bamboo replaces serialized waiting with commit-dependency proofs for retired hot-key updates.",
    },
    (
        "2026-06-05-bamboo-retires-hotspot-locks-before-transaction-commit",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Bamboo warns that retained GPU snapshots must not observe retired but uncommitted versions.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-needs-pressure-shaped-contracts-across-rings-io-and-hot-keys",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that hot-write routes need dependency and cascade-abort shape, not just write labels.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-needs-pressure-shaped-contracts-across-rings-io-and-hot-keys",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring bundling needs active-window benchmarks with response, WAL, NVMe, and dependency pressure.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-needs-pressure-shaped-contracts-across-rings-io-and-hot-keys",
        "dependency_witnesses",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that dependency witnesses must expose dependency and cascade-abort shape explicitly.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-needs-pressure-shaped-contracts-across-rings-io-and-hot-keys",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Cold-tier placement must expose IO shape and flush cadence rather than hide behind generic admission.",
    },
    (
        "2026-06-05-cross-paper-synthesis-admission-needs-pressure-shaped-contracts-across-rings-io-and-hot-keys",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Effective session counting needs active-window benchmarks with idle sessions and backpressure.",
    },
    (
        "2026-06-05-gria-makes-deterministic-batches-adaptive-and-multi-versioned",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Gria's deterministic epoch ordering is an alternative to per-transaction dependency witness coordination.",
    },
    (
        "2026-06-05-horae-separates-durable-order-control-from-parallel-data-writes",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Horae frames device persistence choices as alternatives to normal cost-based route optimization.",
    },
    (
        "2026-06-05-electrode-keeps-protocol-fast-paths-in-the-kernel-not-full-logic",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Electrode motivates protocol-edge wakeup benchmarks at 10K to 1M logical sessions.",
    },
    (
        "2026-06-05-genericvc-turns-mvcc-conflicts-into-configurable-validation-work",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "GenericVC offers configurable commit validation as an alternative to fixed deterministic abort rules.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-semantic-conflict-shape-not-only-resource-shape",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot frontier vectors need active-window benchmarks covering conflict policy and retained visibility.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-need-semantic-conflict-shape-not-only-resource-shape",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL-before-visibility needs active-window benchmarks across control frontiers and completion cells.",
    },
    (
        "2026-06-05-mindpalace-makes-auto-mergeability-an-instance-specific-validation-target",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "MindPalace-style conflict predicates are useful only when route conditions can be derived and evaluated.",
    },
    (
        "2026-06-05-mindpalace-makes-auto-mergeability-an-instance-specific-validation-target",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "MindPalace omits high-concurrency WAL, crash replay, and serializable predicate evaluations.",
    },
    (
        "2026-06-05-mindpalace-makes-auto-mergeability-an-instance-specific-validation-target",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshot transfer needs evaluation against CPU row state and GPU-resident summaries.",
    },
    (
        "2026-06-05-hybridtier-tracks-both-long-term-heat-and-short-term-momentum-for-cxl-tiering",
        "db_owned_cold_objects",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "HybridTier is transparent page tiering, warning against treating it as DB-owned cold-object placement.",
    },
    (
        "2026-06-05-skeena-coordinates-snapshots-and-commits-across-autonomous-engines",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Skeena's cross-engine snapshot registry is an alternative to a single GPU DB frontier vector design.",
    },
    (
        "2026-06-05-skeena-coordinates-snapshots-and-commits-across-autonomous-engines",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained GPU snapshots need table and segment placement experiments modeled on Skeena-style mixes.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-now-need-tier-merge-and-snapshot-contracts",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The synthesis replaces a single GPU-route eligibility flag with explicit semantic, placement, and freshness fields.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-certificates-now-need-tier-merge-and-snapshot-contracts",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The route-certificate framing shifts conflict handling from a generic GPU-eligible decision to explicit proof and fallback fields.",
    },
    (
        "2026-06-05-oltpim-splits-pointer-chasing-metadata-from-tuple-payloads-for-near-memory-oltp",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "OLTPim latency and throughput results do not directly transfer to durable WAL, fsync, archive, checkpoint, or cold-tier paths.",
    },
    (
        "2026-06-05-oltpim-splits-pointer-chasing-metadata-from-tuple-payloads-for-near-memory-oltp",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained lookup transfer is explicitly gated on replaying CPU truth, rebuilding metadata, and skew benchmarks.",
    },
    (
        "2026-06-05-mot-productionizes-many-core-occ-inside-a-full-sql-engine",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "MOT is CPU main-memory OLTP and omits GPU execution, retained MVCC snapshots, over-resident analytics, and heterogeneous tiers.",
    },
    (
        "2026-06-05-mot-productionizes-many-core-occ-inside-a-full-sql-engine",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "MOT does not evaluate NVMe, CXL, GPU memory, or over-resident tiering, so its results caution against direct placement transfer.",
    },
    (
        "2026-06-05-pim-tree-makes-near-data-ordered-indexes-skew-resistant",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "GPU-resident keys or summaries are useful only when hot ranges and hot keys are measured and routed through cheap fallback.",
    },
    (
        "2026-06-05-reef-protects-urgent-gpu-work-by-resetting-idempotent-best-effort-kernels",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Restartable GPU work can support WAL boundaries only if killed refresh/statistics kernels rebuild from CPU truth safely.",
    },
    (
        "2026-06-05-flexmem-adapts-tier-migration-to-emerging-hot-pages-and-promotion-failures",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "FlexMem transfer depends on proving route telemetry can attribute latency to migration or tier misses.",
    },
    (
        "2026-06-05-flexmem-adapts-tier-migration-to-emerging-hot-pages-and-promotion-failures",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Warm-segment protection for retained snapshots is gated on p95/p99 latency, hit ratio, transfer bytes, refresh bytes, and churn.",
    },
    (
        "2026-06-05-flexmem-adapts-tier-migration-to-emerging-hot-pages-and-promotion-failures",
        "stable_handle_indirection",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "FlexMem warns against treating CXL, far memory, host DRAM, or HBM residency as simple LRU or hotness-threshold state.",
    },
    (
        "2026-06-05-flexmem-adapts-tier-migration-to-emerging-hot-pages-and-promotion-failures",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The memory-benchmark evidence needs mixed OLTP/HTAP workload measurement before informing freshness-sensitive routing.",
    },
    (
        "2026-06-05-cross-paper-synthesis-accelerator-routes-need-adaptive-tier-confidence",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained GPU placement is supported only when recent batches and failed admissions justify the memory they displace.",
    },
    (
        "2026-06-05-gpu-joins-need-hardware-shaped-partition-and-output-contracts",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Resident join routing is valid only when reuse and output-buffer capacity can hide transfer behind useful GPU work.",
    },
    (
        "2026-06-05-gpu-joins-need-hardware-shaped-partition-and-output-contracts",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Tiered join placement works only when planner-visible reuse and output capacity justify HBM residency or cold streaming.",
    },
    (
        "2026-06-05-gpu-joins-need-hardware-shaped-partition-and-output-contracts",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Block or thread output allocation supports deterministic templates only under fill and skew conditions that preserve useful work.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-fallback-lanes-and-gpu-route-contracts",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The synthesis names end-to-end mixed write, resident join, long-snapshot, and high-session benchmarks as the remaining gate.",
    },
    (
        "2026-06-05-deferred-runtime-pipelining-turns-hot-writes-into-ordered-intentions",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Deferred runtime intentions avoid immediate execution and descriptor churn through explicit dependency records.",
    },
    (
        "2026-06-05-larger-than-memory-oltp-needs-device-specific-cold-paths",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Compact cold summaries are valid only if false-positive cold reads do not dominate p99 across measured device paths.",
    },
    (
        "2026-06-05-gpu-oltp-concurrency-is-launch-shape-and-conflict-resolution-bound",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The gCCTB evidence is benchmark-centered and requires comparable GPU DB concurrency evaluation before policy adoption.",
    },
    (
        "2026-06-05-gpu-oltp-concurrency-is-launch-shape-and-conflict-resolution-bound",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring transfer depends on measuring GPU DB launch shape and conflict-resolution behavior against the evaluated schemes.",
    },
    (
        "2026-06-05-zero-shot-cost-models-separate-route-shape-from-database-state",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Zero-shot cost predictions are useful only when observed Q-error and bad-route rate remain under deterministic thresholds.",
    },
    (
        "2026-06-05-taobench-turns-session-scale-into-correlated-request-pressure",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained read benchmarks must model correlated fan-out and partial invalidation rather than repeated identical point lookups.",
    },
    (
        "2026-06-05-taobench-turns-session-scale-into-correlated-request-pressure",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "TAOBench provides a better 1M logical-session target, but the GPU DB session policy still needs benchmark validation.",
    },
    (
        "2026-06-05-taobench-turns-session-scale-into-correlated-request-pressure",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "TAOBench's high fan-out transactions warn that hot-write templates must account for tail latency and contention risk.",
    },
    (
        "2026-06-05-taobench-turns-session-scale-into-correlated-request-pressure",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring transfer needs measurements for logical sessions, active requests, fan-out, response bytes, pressure, and contamination.",
    },
    (
        "2026-06-05-taobench-turns-session-scale-into-correlated-request-pressure",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained snapshots are valid only if benchmarks include correlated bursts, tenant sharing, and partial generation invalidation.",
    },
    (
        "2026-06-05-lithos-treats-gpu-sharing-as-an-os-scheduling-problem",
        "resource_dag_scheduling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The TPC scheduler maps to owner-held resource budgets rather than a single FIFO stream scheduling model.",
    },
    (
        "2026-06-05-plor-gives-aborted-hot-transactions-timestamp-priority",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "PLOR transfer needs mixed hot-write, fan-out, and retained-read measurements before changing owner-ring priorities.",
    },
    (
        "2026-06-05-plor-gives-aborted-hot-transactions-timestamp-priority",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The WAL interaction must be measured to see whether append or flush work extends hot-key owner hold time.",
    },
    (
        "2026-06-05-cross-paper-synthesis-commit-decisions-need-a-recoverable-visibility-contract",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The only-if phrase is corpus-planning guidance; the retained synthesis still supports tiering as a visibility-contract topic.",
    },
    (
        "2026-06-05-deuteronomy-makes-range-mvcc-a-logical-route-certificate",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than clause describes partition initialization; the retained evidence still supports range-owned routing metadata.",
    },
    (
        "2026-06-05-deuteronomy-turns-the-recovery-log-into-a-version-cache-and-delivery-queue",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained snapshots are valid only if recovery cannot remove versions that have already been exposed to clients.",
    },
    (
        "2026-06-05-krisp-makes-gpu-partitions-a-per-kernel-admission-decision",
        "same_shape_microbatching",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "KRISP's per-kernel partition admission is an alternative to fixed process-wide or batch-wide GPU allocation.",
    },
    (
        "2026-06-05-krisp-makes-gpu-partitions-a-per-kernel-admission-decision",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Descriptor policy can use KRISP-style partition cues only when per-kernel restriction state remains bounded and explicit.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-and-schedulers-must-become-stage-level-contracts",
        "resource_dag_scheduling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Stage-level resource scheduling is explicitly routed through p50/p99 retained-lookup and co-scheduling stressors.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-and-schedulers-must-become-stage-level-contracts",
        "wal_before_visibility",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Logical range-visibility certificates are framed as an alternative to treating visibility as storage-side aftermath.",
    },
    (
        "2026-06-05-hetexchange-turns-cpu-gpu-routing-into-optimizer-visible-operators",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback adoption needs the named proof gate that every route decision explains resident, transfer, fallback, or rejection.",
    },
    (
        "2026-06-05-hetexchange-turns-cpu-gpu-routing-into-optimizer-visible-operators",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "HetExchange uses heuristic insertion, so optimizer-driven heterogeneous route search remains prototype work.",
    },
    (
        "2026-06-05-performance-optimal-filters-need-route-specific-false-positive-budgets",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route-local filter summaries require measured lookup cost and false-positive budgets before optimizer adoption.",
    },
    (
        "2026-06-05-performance-optimal-filters-need-route-specific-false-positive-budgets",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Filter-descriptor transfer depends on evaluating false-positive precision, SIMD lookup cost, and practical filter sizes.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-choice-now-needs-cost-resource-and-conflict-certificates",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained lookup certificates need measured construction overhead and mixed-workload outcomes before adoption.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-choice-now-needs-cost-resource-and-conflict-certificates",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier placement depends on measuring continuation cost across GPU memory, CPU DRAM, pinned host buffers, NVMe, and future tiers.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-choice-now-needs-cost-resource-and-conflict-certificates",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fallback policy is explicitly included in the route-certificate and mixed-workload benchmark gate.",
    },
    (
        "2026-06-05-gpu-joins-need-partitioning-placement-and-skew-as-explicit-route-traits",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Join fallback needs proof-gate comparisons across resident, streamed, co-processed, CPU fallback, and rejected routes.",
    },
    (
        "2026-06-05-polardb-imci-makes-freshness-a-replay-pipeline-not-a-side-channel",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "WAL visibility routing is valid only if skipped segments and false-positive continuation costs remain explainable.",
    },
    (
        "2026-06-05-polardb-imci-makes-freshness-a-replay-pipeline-not-a-side-channel",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Insertion-ordered column-index row groups are an alternative physical route shape to primary-key-ordered planning assumptions.",
    },
    (
        "2026-06-05-fiting-tree-makes-resident-index-memory-a-tunable-error-budget",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Resident index placement needs HBM-benefit and stale-generation retirement measurements before snapshot adoption.",
    },
    (
        "2026-06-05-cross-paper-synthesis-freshness-windows-need-compact-proof-indexes",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshots need freshness, index-proof, delete-density, and fallback measurements before adding GPU index families.",
    },
    (
        "2026-06-05-cross-paper-synthesis-freshness-windows-need-compact-proof-indexes",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback is part of the resident lookup proof route and must be measured under skew before adoption.",
    },
    (
        "2026-06-05-ccaas-separates-conflict-metadata-from-execution-and-storage",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Conflict-metadata ownership is explicitly a prototype gate tied to p95 saturated-owner latency attribution.",
    },
    (
        "2026-06-05-deuteronomy-2-0-turns-cache-granularity-into-a-hot-path-contract",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Descriptor reclamation needs stale-reader retirement tests over rebuild waste, stalls, latency, and retained memory.",
    },
    (
        "2026-06-05-deuteronomy-2-0-turns-cache-granularity-into-a-hot-path-contract",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot frontier transfer depends on stale-reader tests proving old generations retire only after compatible readers drain.",
    },
    (
        "2026-06-05-barrierfs-separates-storage-order-from-durability-waits",
        "wal_before_visibility",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "BarrierFS-style ordering boundaries are an alternative to treating every visibility-adjacent route boundary as flush-and-wait.",
    },
    (
        "2026-06-05-barrierfs-separates-storage-order-from-durability-waits",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cold-tier IO ownership needs queue-depth benchmarks that preserve dependency epochs and crash-safe route freshness.",
    },
    (
        "2026-06-05-barrierfs-separates-storage-order-from-durability-waits",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Ordering boundaries are an alternative witness shape to immediate durability waits at every route boundary.",
    },
    (
        "2026-06-05-barrierfs-separates-storage-order-from-durability-waits",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Cold-tier placement needs IO-owner queue-depth benchmarks while preserving dependency and freshness epochs.",
    },
    (
        "2026-06-05-barrierfs-separates-storage-order-from-durability-waits",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Ordered storage boundaries are an alternative to requiring immutable route roots to wait on every durability edge.",
    },
    (
        "2026-06-05-epoxy-makes-snapshot-metadata-a-cross-engine-contract",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Epoxy frames CPU truth, GPU snapshots, route metadata, and cold tiers as separate engines with explicit snapshot boundaries.",
    },
    (
        "2026-06-05-epoxy-makes-snapshot-metadata-a-cross-engine-contract",
        "isolation_trace_oracle",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Epoxy provides snapshot isolation, so it cautions against treating its metadata contract as serializable trace evidence.",
    },
    (
        "2026-06-05-conflict-history-can-route-hot-transactions-before-validation",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshots need mixed read/write invalidation tests before conflict-history routing can be adopted.",
    },
    (
        "2026-06-05-rcbench-makes-remote-concurrency-control-cost-a-primitive-budget",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "RCBench omits WAL durability and crash recovery, warning against direct WAL-before-visibility transfer.",
    },
    (
        "2026-06-05-cpu-fallback-scans-need-route-specific-code-shapes",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "GPU routing is valid only when calibrated CPU fallback is compared under the same visibility boundary.",
    },
    (
        "2026-06-05-cpu-fallback-scans-need-route-specific-code-shapes",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The condition cue describes scan predicate code shape, while the retained link remains only weak support for deterministic templates.",
    },
    (
        "2026-06-05-cpu-fallback-scans-need-route-specific-code-shapes",
        "same_shape_microbatching",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The evidence warns that same-shape batches are insufficient without CPU/GPU fallback code-shape awareness.",
    },
    (
        "2026-06-05-hemem-makes-tier-policy-asynchronous-and-application-visible",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The test cue is from fastest-memory wording; the snippet directly supports tier placement policy.",
    },
    (
        "2026-06-05-taurus-separates-durable-log-truth-from-eventually-current-page-service",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Taurus treats pages and snapshots as repairable service tiers rather than correctness-owning placement state.",
    },
    (
        "2026-06-05-horseqc-makes-gpu-transfer-routes-prove-pipeline-density",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "GPU-favorable routes must beat CPU fallback after transfer and launch costs are included.",
    },
    (
        "2026-06-05-horseqc-makes-gpu-transfer-routes-prove-pipeline-density",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner bundling is safe only if hot GPU groups do not starve short retained lookups sharing the worker.",
    },
    (
        "2026-06-05-horseqc-makes-gpu-transfer-routes-prove-pipeline-density",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Older GPU and PCIe-era measurements require recalibration for NVLink, large HBM, and future CXL tiers.",
    },
    (
        "2026-06-05-f1-lightning-turns-htap-into-a-freshness-windowed-service",
        "htap_freshness_router",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "F1 Lightning frames HTAP as a freshness-windowed side service rather than a replacement transactional engine.",
    },
    (
        "2026-06-05-f1-lightning-turns-htap-into-a-freshness-windowed-service",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained routes need timestamp-window, refresh-byte, stale-window, and fallback benchmarks before adoption.",
    },
    (
        "2026-06-05-f1-lightning-turns-htap-into-a-freshness-windowed-service",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Merged CPU/GPU reads are valid only when the snapshot frontier proves timestamp and correctness boundaries.",
    },
    (
        "2026-06-05-f1-lightning-turns-htap-into-a-freshness-windowed-service",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Immutable route roots need retained-route timestamp-window and column-generation refresh benchmarks.",
    },
    (
        "2026-06-05-snapper-mixes-deterministic-batches-with-dynamic-transactions",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Actor-library latency results do not cover GPU kernels, WAL fsync, NVMe, or pgwire behavior.",
    },
    (
        "2026-06-05-snapper-mixes-deterministic-batches-with-dynamic-transactions",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Local partition frontier chains are presented as an alternative to forcing every global generation into the hot path.",
    },
    (
        "2026-06-05-snapper-mixes-deterministic-batches-with-dynamic-transactions",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Single-server actor results leave distributed placement, coordinator locality, and hierarchical ordering as future evaluation work.",
    },
    (
        "2026-06-05-tidb-turns-consensus-replication-into-an-htap-freshness-path",
        "htap_freshness_router",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "TiDB derives HTAP freshness from replication rather than an external ETL pipeline or shared execution path.",
    },
    (
        "2026-06-05-tidb-turns-consensus-replication-into-an-htap-freshness-path",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained GPU reads need timestamp and schema frontier prototype validation.",
    },
    (
        "2026-06-05-tidb-turns-consensus-replication-into-an-htap-freshness-path",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The rather-than cue describes HTAP freshness architecture; the retained link remains weak support for descriptor lifecycle thinking.",
    },
    (
        "2026-06-05-ladm-makes-gpu-locality-a-schedulable-data-certificate",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Placement transfer requires latency, queue, HBM, remote-transfer, and fallback-rate measurements.",
    },
    (
        "2026-06-05-cross-paper-synthesis-serviceable-snapshots-also-need-locality-proof",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained analytical routes are serviceable only when timestamp, schema, and replay frontiers are proven.",
    },
    (
        "2026-06-05-cross-paper-synthesis-serviceable-snapshots-also-need-locality-proof",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Placement is useful only when locality proof is paired with timestamp, schema, and replay frontier proof.",
    },
    (
        "2026-06-05-coco-batches-commit-and-replication-into-epoch-barriers",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Epoch frontier publication requires throughput, latency, WAL-byte, and freshness-lag benchmarks.",
    },
    (
        "2026-06-05-coco-batches-commit-and-replication-into-epoch-barriers",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Immutable roots need epoch-generation publication benchmarks before replacing per-transaction publication.",
    },
    (
        "2026-06-05-coco-batches-commit-and-replication-into-epoch-barriers",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hot-write templates require epoch rollback and exact-equivalence prototype validation.",
    },
    (
        "2026-06-05-snapshot-algorithms-must-be-measured-for-spikes-not-only-throughput",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Full snapshot-dump evaluation does not settle incremental resident refresh or WAL replay behavior.",
    },
    (
        "2026-06-05-cross-paper-synthesis-routes-need-placement-flow-and-frontier-proofs",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Admission needs route-certificate benchmarks covering frontier, placement, and flow proof fields.",
    },
    (
        "2026-06-05-cross-paper-synthesis-routes-need-placement-flow-and-frontier-proofs",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback belongs in the same placement, flow, and frontier benchmark matrix as accepted GPU routes.",
    },
    (
        "2026-06-05-adaptive-compression-should-be-a-tier-policy-not-a-column-default",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Compression placement thresholds must be remeasured for HBM, DRAM, NVMe, and future tiers.",
    },
    (
        "2026-06-05-adaptive-compression-should-be-a-tier-policy-not-a-column-default",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The integer-column prototype leaves string, index, join, SQL expression, CUDA, and memory-management coverage as benchmark debt.",
    },
    (
        "2026-06-05-orpheusdb-makes-old-version-lookup-a-partitioning-problem",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Old-version placement needs separate GPU memory, NVMe, and write-heavy transactional measurements before transfer.",
    },
    (
        "2026-06-05-orpheusdb-makes-old-version-lookup-a-partitioning-problem",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Partition refresh supports WAL recovery only when measured reconstruction cost crosses a tolerated threshold.",
    },
    (
        "2026-06-05-cross-paper-synthesis-frontiers-need-semantic-proof-surfaces",
        "dependency_witnesses",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The not-just-timestamp cue strengthens the need for compact dependency proof surfaces rather than weakening the mechanism.",
    },
    (
        "2026-06-05-hana-nse-makes-warm-placement-byte-compatible-not-separate-engine",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Warm placement is safe only when crash recovery and log replay keep emergency buffer paths separate from normal cache pressure.",
    },
    (
        "2026-06-05-skq-makes-event-delivery-a-schedulable-resource",
        "deficit_fairness",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fair event delivery is valid only if high-priority routes cannot starve regular traffic or delay cancellation indefinitely.",
    },
    (
        "2026-06-05-execution-routes-should-choose-fusion-by-data-behavior",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Execution-model isolation is an alternative planning signal to treating learned advice or whole-engine identity as the route decision.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The synthesis frames route-shape proof as an alternative to one-time engine identity or opaque learned route advice.",
    },
    (
        "2026-06-05-flowcut-keeps-adaptive-network-routing-in-order-by-draining-active-flows",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Session-count transfer depends on measured response order, reorder bytes, cancellation latency, and in-flight bytes.",
    },
    (
        "2026-06-05-neurcc-makes-concurrency-control-a-learned-action-table",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Learned hot-write choices are valid only if they never expose dirty state or partial retries after pgwire-visible results.",
    },
    (
        "2026-06-05-neurcc-makes-concurrency-control-a-learned-action-table",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The lookup-table cue supports bounded learned advice by avoiding model inference in the hot path.",
    },
    (
        "2026-06-05-type-aware-transactions-make-conflict-semantics-a-data-structure-contract",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Datatype-owned conflict predicates directly support explicit GPU OLTP conflict ordering rather than a universal word-level rule.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The synthesis supports costed route choice by requiring route state, access costs, and datatype conflict predicates.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Drain frontiers and resident route state support explicit snapshot frontier tracking for ordered route policies.",
    },
    (
        "2026-06-05-cross-paper-synthesis-route-policies-need-execution-shape-proof",
        "multi_tier_placement",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Resident access costs and route-shape proof support tier placement as an explicit route-policy input.",
    },
    (
        "2026-06-05-sap-hana-nvm-keeps-hot-mutability-out-of-the-persistent-tier",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Persistent-tier placement is valid only when mutable deltas, visibility metadata, route state, and buffers remain in DRAM or HBM unless benchmarks prove otherwise.",
    },
    (
        "2026-06-05-sap-hana-nvm-keeps-hot-mutability-out-of-the-persistent-tier",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Physical block removal is safe only when retained snapshots, rollback, and recovery replay no longer name the block.",
    },
    (
        "2026-06-05-sap-hana-nvm-keeps-hot-mutability-out-of-the-persistent-tier",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retained snapshots may use persistent backing only if hot mutability and visibility metadata stay in fast memory or are benchmark-proven.",
    },
    (
        "2026-06-05-sap-hana-nvm-keeps-hot-mutability-out-of-the-persistent-tier",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Session-admission transfer needs benchmarks because the evidence omits GPU kernels, CUDA transfers, GPUDirect, CXL, and pgwire sessions.",
    },
    (
        "2026-06-05-sgdrc-splits-gpu-service-quality-into-sm-and-vram-channel-budgets",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Sliding-window GPU reservation is an alternative owner-capacity policy to fixed foreground fractions.",
    },
    (
        "2026-06-05-sgdrc-splits-gpu-service-quality-into-sm-and-vram-channel-budgets",
        "resource_dag_scheduling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Dynamic SM reservation and eviction are alternatives to static resource DAG partitioning for latency-sensitive kernels.",
    },
    (
        "2026-06-05-sgdrc-splits-gpu-service-quality-into-sm-and-vram-channel-budgets",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The DNN-inference and TVM-like compilation assumptions caution against direct descriptor-reclamation transfer.",
    },
    (
        "2026-06-05-conweave-masks-rdma-rerouting-disorder-inside-the-network",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Rerouting supports dependency witnesses only if pgwire-visible responses preserve order or explicitly declare independent completion.",
    },
    (
        "2026-06-05-conweave-masks-rdma-rerouting-disorder-inside-the-network",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Frequent route changes are safe only with a bounded mechanism that restores the receiver's ordering contract.",
    },
    (
        "2026-06-05-orion-co-schedules-gpu-kernels-by-resource-complementarity",
        "deficit_fairness",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Best-effort co-scheduling is fair only when the added kernel is small and resource-complementary to the high-priority job.",
    },
    (
        "2026-06-05-orion-co-schedules-gpu-kernels-by-resource-complementarity",
        "resource_dag_scheduling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Complementary-kernel scheduling is valid only when resource profiles prove the best-effort work will not harm the priority route.",
    },
    (
        "2026-06-05-l-store-stages-write-optimized-deltas-into-read-optimized-pages-by-lineage",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Independent column and index refresh cautions against coarse snapshot validity without generation and delta frontiers.",
    },
    (
        "2026-06-05-cross-paper-synthesis-routes-need-ordered-profiled-lineage-certified-publication",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Per-column and per-index lineage frontiers are presented as an alternative to a single coarse immutable route validity bit.",
    },
    (
        "2026-06-05-cross-paper-synthesis-routes-need-ordered-profiled-lineage-certified-publication",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that queue delay, freshness, and tier placement are plan risks rather than after-the-fact runtime events.",
    },
    (
        "2026-06-05-cross-paper-synthesis-routes-need-ordered-profiled-lineage-certified-publication",
        "htap_freshness_router",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "The synthesis warns that freshness must be a first-class route risk rather than an after-the-fact runtime event.",
    },
    (
        "2026-06-05-cherry-garcia-commits-heterogeneous-store-writes-through-recoverable-metadata",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Heterogeneous-store routing is valid only when the route can prove common read, conditional-write, durability, and metadata capabilities.",
    },
    (
        "2026-06-05-unimem-makes-far-memory-useful-by-separating-addressability-filtering-and-promotion",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Promotion is useful only when placement proof shows the promoted unit has enough useful bytes for the fast tier.",
    },
    (
        "2026-06-05-cross-paper-synthesis-publication-proof-needs-placement-proof",
        "vector_credit_admission",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Admission work should proceed only if no nearer runtime, session, network, or scheduling evidence fills the same gap.",
    },
    (
        "2026-06-05-backpressure-flow-control-makes-admission-local-selective-and-bounded",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Active-session lane and buffer-credit admission is explicitly framed as a benchmark before adoption.",
    },
    (
        "2026-06-05-backpressure-flow-control-makes-admission-local-selective-and-bounded",
        "deficit_fairness",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fairness is required only when hot sessions can reacquire scarce active lanes ahead of quiet sessions.",
    },
    (
        "2026-06-05-deadlock-safety-needs-packet-level-pressure-not-just-cycle-detection",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Template routing is safe only with restrictions or structured resources that prevent cyclic buffer dependencies.",
    },
    (
        "2026-06-05-scalardb-makes-transaction-authority-an-adapter-visible-metadata-layer",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Adapter-visible route choice is valid only for stores that expose linearizable reads, conditional mutation, durability, and metadata room.",
    },
    (
        "2026-06-05-reps-turns-path-choice-into-tiny-recycled-endpoint-state",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Recycled endpoint hints must be benchmarked inside owner rings and response lanes before relying on the transfer.",
    },
    (
        "2026-06-05-rethinking-simd-vectorization-for-in-memory-databases",
        "cpu_fallback_policy",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Operator-specific layout work is needed before CPU fallback and warm-tier routes assume generic acceleration wins.",
    },
    (
        "2026-06-05-rethinking-simd-vectorization-for-in-memory-databases",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Explicit vector primitives are presented as an alternative to hoping scalar operators expose useful hardware behavior.",
    },
    (
        "2026-06-05-selection-pushdown-in-column-stores-using-bit-manipulation-instructions",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Compressed chunk pushdown supports retained snapshots only if freshness under mutations is handled separately.",
    },
    (
        "2026-06-05-selection-pushdown-in-column-stores-using-bit-manipulation-instructions",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Dictionary-order and predicate-shape caveats warn against treating compressed descriptors as generally reusable.",
    },
    (
        "2026-06-05-selection-pushdown-in-column-stores-using-bit-manipulation-instructions",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU compressed scan, CPU prefilter, GPU transfer, resident GPU scan, and fallback thresholds require measurement.",
    },
    (
        "2026-06-05-selection-pushdown-in-column-stores-using-bit-manipulation-instructions",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The non-order-preserving dictionary warning is a route-shape caveat; owner-ring evidence remains a support candidate.",
    },
    (
        "2026-06-05-vegito-turns-ha-backups-into-fresh-columnar-htap-replicas",
        "htap_freshness_router",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "A fresh HA backup retrofitted as a columnar replica is an alternative to separate ETL or dual-layout HTAP routing.",
    },
    (
        "2026-06-05-vegito-turns-ha-backups-into-fresh-columnar-htap-replicas",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Backup-like generation semantics are valid only if they do not add hidden commit-time replication cost.",
    },
    (
        "2026-06-05-taurus-ndp-makes-cold-tier-pushdown-best-effort-and-mvcc-safe",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Disaggregated analytical pushdown cautions against direct inference for GPU-resident OLTP snapshots.",
    },
    (
        "2026-06-05-taurus-ndp-makes-cold-tier-pushdown-best-effort-and-mvcc-safe",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Cold-tier pushdown is valid only if it does not silently worsen later retained or CPU fallback routes.",
    },
    (
        "2026-06-05-bcc-reduces-false-occ-aborts-with-bounded-dependency-checks",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Read-only snapshot convenience can hide metadata costs, so publication and long-reader frontiers must be explicit.",
    },
    (
        "2026-06-06-liquidcache-makes-pushdown-a-cache-format-problem",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Warm-tier encoded pushdown must be measured against CPU tuple scan, GPU resident scan, and cold segment scan baselines.",
    },
    (
        "2026-06-06-liquidcache-makes-pushdown-a-cache-format-problem",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Selective decode batch sizing needs microbenchmarks before the optimizer can price the route.",
    },
    (
        "2026-06-06-liquidcache-makes-pushdown-a-cache-format-problem",
        "wal_before_visibility",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "File-catalog consistency is an alternative to SQL MVCC, WAL-before-visibility, DDL generation, and row-update chains.",
    },
    (
        "2026-06-06-liquidcache-makes-pushdown-a-cache-format-problem",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained filters need warm encoded, resident GPU, CPU, and cold segment benchmark comparisons.",
    },
    (
        "2026-06-06-liquidcache-makes-pushdown-a-cache-format-problem",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Filter-friendly cache encodings are an alternative way to shape work instead of owner-oriented durable format changes.",
    },
    (
        "2026-06-06-shardingsphere-makes-route-metadata-a-first-class-execution-boundary",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Proxy compatibility and million-session multiplexing need benchmarks before embedded route fast paths are trusted.",
    },
    (
        "2026-06-06-upbit-keeps-bitmap-filters-mutable-by-separating-sparse-update-state",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Segment-local bitmap and delta summaries are prototype work before they inform route costing.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-private-formats-plus-publication-proof",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Private pushdown formats and mutable predicate evidence are alternatives to rewriting base compressed representation for route choice.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-private-formats-plus-publication-proof",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback must be measured alongside raw CPU, encoded warm, sparse bitmap, and GPU resident routes.",
    },
    (
        "2026-06-06-cubit-makes-updatable-bitmap-indexes-concurrent-with-logged-horizontal-deltas",
        "htap_freshness_router",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Append-heavy updates and retained analytical reads require an HTAP freshness benchmark before adopting bitmap maintenance.",
    },
    (
        "2026-06-06-cubit-makes-updatable-bitmap-indexes-concurrent-with-logged-horizontal-deltas",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Selective bitmap-powered scan, aggregation, and join routes need evaluation before general route-cost adoption.",
    },
    (
        "2026-06-06-one-loop-does-not-fit-all-makes-execution-shape-selectivity-dependent",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Execution-shape transfer to descriptor reclamation needs evaluation beyond the narrow column-store predicate pipeline.",
    },
    (
        "2026-06-06-one-loop-does-not-fit-all-makes-execution-shape-selectivity-dependent",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The entry explicitly routes tier-placement transfer through HBM, DRAM, CPU-cache, latency, and correctness measurements.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-private-formats-receiver-credits-and-execution-shape-proo",
        "dependency_witnesses",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Execution shape varying by selectivity and movement warns against fixed dependency witnesses that ignore route shape.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-private-formats-receiver-credits-and-execution-shape-proo",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Receiver-credit admission is explicitly tied to measuring staged masks and bitmap deltas under output-buffer pressure.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-routes-need-private-formats-receiver-credits-and-execution-shape-proo",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Receiver-clocked explicit-edge queueing is presented as an alternative to hiding overload inside shared owner cores.",
    },
    (
        "2026-06-06-crystal-turns-cache-entries-into-semantic-regions-not-blocks",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Crystal maps to semantic placement regions, but analytical cloud-storage results need GPU and OLTP placement validation.",
    },
    (
        "2026-06-06-crystal-turns-cache-entries-into-semantic-regions-not-blocks",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Semantic route regions are an alternative framing to anonymous page-style retained GPU snapshot residency.",
    },
    (
        "2026-06-06-push-and-pull-are-route-shapes-not-engine-religions",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback must be benchmarked against the same predicate and result shapes as each GPU route.",
    },
    (
        "2026-06-06-push-and-pull-are-route-shapes-not-engine-religions",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The paper omits GPU, MVCC, write-concurrency, sessions, and tiering, so retained snapshots need separate validation.",
    },
    (
        "2026-06-06-push-and-pull-are-route-shapes-not-engine-religions",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Push/pull execution evidence does not cover GPU memory or tiered storage, so placement transfer is a benchmark gate.",
    },
    (
        "2026-06-06-epoch-reclamation-can-double-as-a-range-query-snapshot-source",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Reusable per-owner descriptors are presented as an alternative to fresh descriptor allocation for each publication.",
    },
    (
        "2026-06-06-fptree-persistent-leaves-volatile-routing-and-crash-bounded-index-repair",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Restart rebuilds and publication ordering must be benchmarked before volatile route metadata informs cost choices.",
    },
    (
        "2026-06-06-fptree-persistent-leaves-volatile-routing-and-crash-bounded-index-repair",
        "stable_handle_indirection",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Persistent-pointer-style route identities require crash-interruption correctness measurements before handle adoption.",
    },
    (
        "2026-06-06-fptree-persistent-leaves-volatile-routing-and-crash-bounded-index-repair",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Data-first and bitmap-or-generation-last publication needs a route-root microbenchmark before adoption.",
    },
    (
        "2026-06-06-vbr-reclaims-route-metadata-by-validating-versions-instead-of-waiting-on-readers",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "VBR-style reclamation is valid only under its CAS, invalidation, retirement, and no-relink assumptions.",
    },
    (
        "2026-06-06-db2-native-cos-keeps-database-pages-by-moving-the-storage-contract-underneath-them",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Moving storage contracts underneath pages is an alternative to treating GPU-memory descriptor state as durable truth.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-proof-now-spans-publication-reclamation-and-storage-placement",
        "effective_session_counting",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "VBR-style memory-lifetime validation is framed as an alternative to letting stalled sessions pin metadata forever.",
    },
    (
        "2026-06-06-pangu-makes-rdma-a-fast-path-with-tcp-as-the-safety-valve",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fast zero-copy/offload contracts are useful only with monitored failover paths that degrade instead of freezing storage.",
    },
    (
        "2026-06-06-bvlsm-moves-value-separation-into-wal-admission",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Compact durable value records are an alternative to forcing payload, visibility, route metadata, and compaction through one root.",
    },
    (
        "2026-06-06-bvlsm-moves-value-separation-into-wal-admission",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Payload-write dispatch must be measured against a single mutation owner before changing owner-bundling policy.",
    },
    (
        "2026-06-06-bvlsm-moves-value-separation-into-wal-admission",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Compact durable records are an alternative to reclaiming bulky payload and route metadata through one owner queue.",
    },
    (
        "2026-06-06-bvlsm-moves-value-separation-into-wal-admission",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Value-separation admission is an alternative data-shaping contract to routing all cost-relevant state through one owner queue.",
    },
    (
        "2026-06-06-dex-keeps-remote-range-indexes-scalable-with-logical-ownership",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Bucketed logical ownership and coalesced refreshes are alternatives to centralized metadata-cache owner queues.",
    },
    (
        "2026-06-06-gpu-accelerated-oltp-shows-concurrency-control-is-a-route-shape",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "GPU batch ordering should enter owner routing only when admission proves same-domain conflict density pays for the path.",
    },
    (
        "2026-06-06-gpu-accelerated-oltp-shows-concurrency-control-is-a-route-shape",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "The retained-route evidence assumes preloaded fixed-size tables, no inserts or deletes, and known read/write sets.",
    },
    (
        "2026-06-06-gpu-accelerated-oltp-shows-concurrency-control-is-a-route-shape",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "GPU OLTP assumptions do not cover interactive SQL, DDL, MVCC chains, WAL replay, recovery, or arbitrary predicates.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-proof-now-includes-admission-shape",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hybrid CPU/OCC and GPU conflict ordering requires measurements plus explicit accepted and rejected route facts.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-proof-now-includes-admission-shape",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The bounded retained-route runtime directly supports retained snapshots; the instead-of cue names stampede avoidance.",
    },
    (
        "2026-06-06-oneshotgc-makes-mvcc-cleanup-a-partition-publication-problem",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "OneShotGC's in-memory CPU prototype cautions against inferring GPU WAL, recovery, and device-residency behavior.",
    },
    (
        "2026-06-06-publish-on-ping-makes-reclamation-demand-driven-instead-of-read-path-pessimistic",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "POP-style reclamation needs database descriptor benchmarks beyond the public safe-memory-reclamation suites.",
    },
    (
        "2026-06-06-publish-on-ping-makes-reclamation-demand-driven-instead-of-read-path-pessimistic",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained-snapshot publication needs a long-reader stress benchmark before snapshot-frontier transfer is trusted.",
    },
    (
        "2026-06-06-publish-on-ping-makes-reclamation-demand-driven-instead-of-read-path-pessimistic",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "POP's CPU safe-memory-reclamation scope cautions against direct MVCC GC frontier inference.",
    },
    (
        "2026-06-06-chainpaxos-makes-replication-throughput-a-pipeline-and-membership-problem",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Chain-ordered replica acknowledgements are an alternative witness shape to leader fan-in dependency tracking.",
    },
    (
        "2026-06-06-cross-paper-synthesis-retirement-freshness-and-replication-all-need-explicit-fences",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Dependency witnesses are supported only when each owner publishes the named fence for its route boundary.",
    },
    (
        "2026-06-06-fineline-turns-durable-storage-into-an-indexed-recovery-log",
        "log_structured_warm_tier",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "FineLine's single indexed log is an alternative to maintaining separate synchronized persistent tier representations.",
    },
    (
        "2026-06-06-fineline-turns-durable-storage-into-an-indexed-recovery-log",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Indexed-log recovery must be benchmarked for commit p99 and resident snapshot reconstruction before frontier adoption.",
    },
    (
        "2026-06-06-occ-batching-turns-contention-into-a-reorderable-route-batch",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Batch reordering uses a validator dependency graph as an alternative to committing in arrival order.",
    },
    (
        "2026-06-06-occ-batching-turns-contention-into-a-reorderable-route-batch",
        "same_shape_microbatching",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Semantic OCC batching is an alternative to treating same-shape microbatching as only amortized execution.",
    },
    (
        "2026-06-06-occ-batching-turns-contention-into-a-reorderable-route-batch",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "GPU OLTP conflict ordering needs prototype measurements for reorder cost, validation bottlenecks, and abort behavior.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-publication-needs-explicit-fences",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fast publication paths support dependency witnesses only when the safe fence is explicitly named.",
    },
    (
        "2026-06-06-self-tuning-scheduling-makes-route-priority-a-measured-control-loop",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Database-owned scheduling is an alternative control loop rather than evidence for descriptor reclamation itself.",
    },
    (
        "2026-06-06-hdtx-coalesces-remote-transaction-fences-without-giving-up-priority",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Aggressive route admission is valid only if WAL, visibility, residency, and retirement fences stay observable.",
    },
    (
        "2026-06-06-verlib-makes-snapshot-handles-a-pointer-primitive",
        "stable_handle_indirection",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "VERLIB-style indirection supports stable handles only when metadata reuse is proven safe for old snapshots.",
    },
    (
        "2026-06-06-verlib-makes-snapshot-handles-a-pointer-primitive",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Route costing can use VERLIB-like indirection only when sharing and shortcut conditions are explicit.",
    },
    (
        "2026-06-06-cross-paper-synthesis-publication-primitives-need-retry-safe-handles",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "MVCC frontiers need stale-read, double-publication, rollback, and long-reader stress gates.",
    },
    (
        "2026-06-06-viper-turns-snapshot-isolation-into-begin-commit-graph-acyclicity",
        "wal_before_visibility",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Viper-style begin/commit graph logging is an alternative proof path to treating retained reads as fast-path success.",
    },
    (
        "2026-06-06-alock-splits-local-and-remote-lock-cohorts-instead-of-forcing-loopback",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Bounded descriptor reclamation needs stale remote handoff and hot local read contention tests.",
    },
    (
        "2026-06-06-cross-paper-synthesis-remote-routes-need-tiny-authorities-and-external-witnesses",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Compact route authorities are valid only when obsolete route generations cannot wake work.",
    },
    (
        "2026-06-06-star-moves-rdma-connection-state-off-the-fan-in-bottleneck",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Route descriptors need state-cache, queue-depth, pinned-buffer, and response-backlog measurements before adoption.",
    },
    (
        "2026-06-06-star-moves-rdma-connection-state-off-the-fan-in-bottleneck",
        "stable_handle_indirection",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Compact request descriptors are an alternative to retaining every session's full state in owners or GPU workers.",
    },
    (
        "2026-06-06-srnic-minimizes-nic-resident-per-connection-state",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Descriptor admission is valid only with snapshot, resident, output-buffer, and execution-credit proofs.",
    },
    (
        "2026-06-06-srnic-minimizes-nic-resident-per-connection-state",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Immutable route roots are valid only when each read route carries generation and credit proofs.",
    },
    (
        "2026-06-06-dcos-schedules-hot-transaction-pieces-without-making-every-transaction-fine-grained",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "DCoS supports conflict ordering only with an underlying serializable CC and runtime-pipelining mechanism.",
    },
    (
        "2026-06-06-dcos-schedules-hot-transaction-pieces-without-making-every-transaction-fine-grained",
        "effective_session_counting",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Hot-transaction cohort scheduling is an alternative to per-session cleverness at million-session scale.",
    },
    (
        "2026-06-06-cross-paper-synthesis-hot-paths-need-compact-authorities-and-schedulable-residuals",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "The synthesis names write-path storage-tiering gaps rather than directly supporting current multi-tier placement.",
    },
    (
        "2026-06-06-cross-paper-synthesis-hot-paths-need-compact-authorities-and-schedulable-residuals",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring bundling needs fast-authority and residual-scheduling benchmarks before adoption.",
    },
    (
        "2026-06-06-cross-paper-synthesis-hot-paths-need-compact-authorities-and-schedulable-residuals",
        "gpu_oltp_conflict_ordering",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "GPU OLTP conflict cohorts must be benchmarked before adopting global minima, queues, or learned policies.",
    },
    (
        "2026-06-06-template-robustness-certifies-cheap-read-committed-routes-before-runtime",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Deterministic template adoption needs certificate and measurement gates for rare or write-heavy routes.",
    },
    (
        "2026-06-06-vortex-makes-streaming-ingest-the-storage-authority-then-continuously-reshapes-it-for-reads",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Fragment-generation unions are an alternative to whole-table invalidation for retained resident reads.",
    },
    (
        "2026-06-06-vortex-makes-streaming-ingest-the-storage-authority-then-continuously-reshapes-it-for-reads",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Vortex's BigQuery-scale analytics scope cautions against direct OLTP GPU placement transfer.",
    },
    (
        "2026-06-06-foresight-schedules-hot-transactions-before-spending-execution-work",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Prediction-driven snapshot use is valid only while schema, join, and changed-condition fallbacks are explicit.",
    },
    (
        "2026-06-06-foresight-schedules-hot-transactions-before-spending-execution-work",
        "cpu_fallback_policy",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Fallback is required when prediction admits work that later reads stale resident state.",
    },
    (
        "2026-06-06-foresight-schedules-hot-transactions-before-spending-execution-work",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Conflict estimation supports route costing; the conflict cue is not evidence against the optimizer mechanism.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-publication-needs-prediction-plus-fallback",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot-generation publication proof is named as a route-certificate benchmark gate.",
    },
    (
        "2026-06-06-cross-paper-synthesis-fast-publication-needs-prediction-plus-fallback",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Dependency witnesses need route-certificate benchmarks that expose conflict prediction and fallback reasons.",
    },
    (
        "2026-06-06-vbox-makes-predicate-serializability-checking-compact-enough-for-route-audits",
        "isolation_trace_oracle",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Predicate-aware verification directly supports the isolation trace oracle despite the not-just cue.",
    },
    (
        "2026-06-06-vbox-makes-predicate-serializability-checking-compact-enough-for-route-audits",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot frontier route facts are explicitly part of the proposed benchmark transaction records.",
    },
    (
        "2026-06-06-vbox-makes-predicate-serializability-checking-compact-enough-for-route-audits",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Witness quality and overlap-window measurements are required before adopting the verifier-derived dependency shape.",
    },
    (
        "2026-06-06-vbox-makes-predicate-serializability-checking-compact-enough-for-route-audits",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "CPU fallback paths need mixed predicate read/write serial-order tests before they can be trusted.",
    },
    (
        "2026-06-06-dataset-version-retention-should-be-a-graph-frontier-not-an-age-rule",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Graph-frontier retention needs long-reader and eviction stress measurements before adoption.",
    },
    (
        "2026-06-06-dataset-version-retention-should-be-a-graph-frontier-not-an-age-rule",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL replay edges are part of a proposed snapshot-retention graph benchmark, not settled support.",
    },
    (
        "2026-06-06-cross-paper-synthesis-tiered-histories-need-one-logical-address-space",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshots need version-graph benchmarks across HBM, DRAM, NVMe, and WAL replay.",
    },
    (
        "2026-06-06-cross-paper-synthesis-tiered-histories-need-one-logical-address-space",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "MVCC frontier transfer depends on version-graph retention measurements across tiered histories.",
    },
    (
        "2026-06-06-skinnerdb-turns-bad-join-orders-into-bounded-exploration-cost",
        "cpu_fallback_policy",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Bounded exploration compares CPU fallback against retained GPU and cold-transfer route alternatives.",
    },
    (
        "2026-06-06-chex-turns-multiversion-replay-into-bounded-checkpoint-placement",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Tiered checkpoint placement is valid only if pinned generations do not force unbounded HBM or DRAM growth.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-decisions-need-explainable-metadata-bounded-exploration-and-version-",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Bounded route exploration is presented as an alternative to relying only on deterministic route costing.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-decisions-need-explainable-metadata-bounded-exploration-and-version-",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier selection requires route-certificate and version-tree placement benchmarks before adoption.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-decisions-need-explainable-metadata-bounded-exploration-and-version-",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Visibility-boundary and WAL-related route facts are part of the proposed benchmark gate.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-decisions-need-explainable-metadata-bounded-exploration-and-version-",
        "deterministic_hot_write_templates",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hot-write template adoption is tied to route-certificate and bounded-exploration benchmarks.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-decisions-need-explainable-metadata-bounded-exploration-and-version-",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Optimized version trees are an alternative to naive generation retention and descriptor retirement.",
    },
    (
        "2026-06-06-cross-paper-synthesis-route-decisions-need-explainable-metadata-bounded-exploration-and-version-",
        "immutable_route_roots",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Route metadata as a first-class columnar product is an alternative publication shape to opaque route roots.",
    },
    (
        "2026-06-06-rewind-makes-byte-addressable-durability-a-log-structure-problem",
        "log_structured_warm_tier",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "REWIND's byte-addressable NVM scope cautions against direct SQL/GPU warm-tier transfer.",
    },
    (
        "2026-06-06-rewind-makes-byte-addressable-durability-a-log-structure-problem",
        "multi_tier_placement",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "REWIND omits GPU database tiering and warns against inferring HBM/DRAM/NVMe placement behavior.",
    },
    (
        "2026-06-06-alece-makes-dynamic-cardinality-a-query-data-attention-problem",
        "multi_tier_placement",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Live route-state vectors provide an alternative to static table statistics for tier placement decisions.",
    },
    (
        "2026-06-06-alece-makes-dynamic-cardinality-a-query-data-attention-problem",
        "wal_before_visibility",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Visibility-generation route vectors complement WAL safety but are not direct WAL-before-visibility evidence.",
    },
    (
        "2026-06-06-alece-makes-dynamic-cardinality-a-query-data-attention-problem",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Compact route-state vectors are an alternative to descriptor-heavy planning metadata.",
    },
    (
        "2026-06-06-cross-paper-synthesis-planning-needs-live-state-but-correctness-still-needs-hard-gates",
        "vector_credit_admission",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The retained evidence directly supports hard queue-capacity and memory-budget admission gates.",
    },
    (
        "2026-06-06-cross-paper-synthesis-planning-needs-live-state-but-correctness-still-needs-hard-gates",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL and checkpoint safety are hard proof gates that need explicit route validation.",
    },
    (
        "2026-06-06-cross-paper-synthesis-planning-needs-live-state-but-correctness-still-needs-hard-gates",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained GPU route selection needs regret benchmarks under changing distributions, generations, and pressure.",
    },
    (
        "2026-06-06-aeolus-protects-scheduled-work-by-making-speculation-disposable",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Speculative same-shape batching needs measured first-request latency, rejection rate, and scheduled-lane tail latency.",
    },
    (
        "2026-06-06-aeolus-protects-scheduled-work-by-making-speculation-disposable",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Aeolus omits SQL isolation, MVCC visibility, WAL durability, GPU launch overhead, and 1M-session evaluation.",
    },
    (
        "2026-06-06-mtm-makes-tier-placement-a-sampled-control-loop-not-a-static-hot-page-rule",
        "multi_tier_placement",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "MTM supports tier placement only when placement is driven by profiling quality instead of a fixed hottest-page rule.",
    },
    (
        "2026-06-06-1rma-makes-remote-memory-access-connection-free-and-credit-shaped",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Remote/cold-tier placement needs p99 retained-lookup and typed-overload prototype measurements before adoption.",
    },
    (
        "2026-06-06-1rma-makes-remote-memory-access-connection-free-and-credit-shaped",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring bundling needs measured remote queue congestion, solicitation-window congestion, and completion delay.",
    },
    (
        "2026-06-06-cross-paper-synthesis-resource-credits-should-travel-with-route-work",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier-placement transfer is explicitly routed through route-credit simulation and cold-tier chunking benchmarks.",
    },
    (
        "2026-06-06-cross-paper-synthesis-resource-credits-should-travel-with-route-work",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Vector-credit admission needs route-credit simulation with typed overload and revocation completions.",
    },
    (
        "2026-06-06-cross-paper-synthesis-resource-credits-should-travel-with-route-work",
        "deficit_fairness",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Protected scheduled credits and typed pressure signals support fairness rather than a competing mechanism.",
    },
    (
        "2026-06-06-aria-makes-deterministic-oltp-a-batch-snapshot-conflict-filter",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Compact stable operation inputs are an alternative to shipping full materialized results across owners.",
    },
    (
        "2026-06-06-cross-paper-synthesis-batches-need-bounded-credits-and-escape-hatches",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL and visibility boundaries must be measured alongside p99 maintenance latency and write retry amplification.",
    },
    (
        "2026-06-06-tb-collect-makes-nvm-mvcc-cleanup-a-block-level-write-amplification-problem",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Block or cohort retirement by visibility epoch is an alternative to per-version snapshot frontier cleanup.",
    },
    (
        "2026-06-06-diffkv-makes-value-placement-a-scan-write-ordering-dial",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner bundling around value placement needs rewrite, stale-byte, snapshot-lag, queue-wait, and crash-boundary measurements.",
    },
    (
        "2026-06-06-silk-makes-compaction-a-foreground-slo-scheduling-problem",
        "resource_dag_scheduling",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "SILK shows scheduling still degrades under long write peaks when resource headroom is insufficient.",
    },
    (
        "2026-06-06-cross-paper-synthesis-maintenance-needs-credits-generations-and-preemption",
        "vector_credit_admission",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Shared credits across reads, writes, refresh, cleanup, and cold-tier merge require mixed-plane benchmarks.",
    },
    (
        "2026-06-06-reactors-make-owner-domains-a-programmable-latency-boundary",
        "owner_ring_bundling",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Owner-domain distribution is useful only if communication hops do not improve average throughput while regressing p99.",
    },
    (
        "2026-06-06-orcgc-makes-reclamation-bounds-part-of-the-hot-path-contract",
        "stable_handle_indirection",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Stable handle adoption needs CPU-path measurements for pointer protection, reference counts, epochs, and owner-local handles.",
    },
    (
        "2026-06-06-orcgc-makes-reclamation-bounds-part-of-the-hot-path-contract",
        "mvcc_gc_frontiers",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "MVCC GC transfer needs prototype proof that retired bytes stay bounded without blocking WAL visibility publication.",
    },
    (
        "2026-06-06-justitia-makes-shared-fabric-admission-a-multi-resource-credit-problem",
        "vector_credit_admission",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Justitia's low-latency fabric evidence applies only when shared-resource interference is explicitly controlled.",
    },
    (
        "2026-06-06-justitia-makes-shared-fabric-admission-a-multi-resource-credit-problem",
        "effective_session_counting",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Session-scale transfer is valid only when RDMA-like latency assumptions survive shared application load.",
    },
    (
        "2026-06-06-cross-paper-synthesis-owner-routes-need-credits-and-bounded-cleanup",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The synthesis directly supports bounded retired-metadata handoff rather than an unbounded background cleanup detail.",
    },
    (
        "2026-06-06-cross-paper-synthesis-owner-routes-need-credits-and-bounded-cleanup",
        "vector_credit_admission",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Multi-resource token control is retained as direct support for vector-credit admission.",
    },
    (
        "2026-06-06-cross-paper-synthesis-owner-routes-need-credits-and-bounded-cleanup",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Effective session counting needs route-hop, credit, queue-wait, cleanup-generation, and tiny-read stress measurements.",
    },
    (
        "2026-06-06-conditional-access-makes-reclamation-a-cache-coherence-contract",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Conditional Access supports reclamation only if per-read validation overhead beats lower-overhead reclamation schemes.",
    },
    (
        "2026-06-06-conditional-access-makes-reclamation-a-cache-coherence-contract",
        "immutable_route_roots",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Retired route metadata is useful only with bounded lifetime and without depending on unavailable CPU primitives.",
    },
    (
        "2026-06-06-clobber-nvm-makes-durable-metadata-replay-a-deterministic-input-problem",
        "immutable_route_roots",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Durable route descriptors, resident directories, checkpoint indexes, and manifests support immutable route-root metadata.",
    },
    (
        "2026-06-06-lemo-makes-concurrent-query-optimization-cache-aware",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Lemo-style cached intermediates need reuse, invalidation, HBM-pressure, and multi-tier artifact placement measurements.",
    },
    (
        "2026-06-06-lemo-makes-concurrent-query-optimization-cache-aware",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "Missing Lemo model, replacement, evaluation, and failure details caution against direct learned-advisor transfer.",
    },
    (
        "2026-06-06-lemo-makes-concurrent-query-optimization-cache-aware",
        "same_shape_microbatching",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Shared key-vector and response-metadata reuse needs latency, launch, HBM, reuse-hit, and fallback benchmarks.",
    },
    (
        "2026-06-06-lemo-makes-concurrent-query-optimization-cache-aware",
        "effective_session_counting",
    ): {
        "relation_type": "supports",
        "relation_review_note": "Binding reuse to worker and route cohorts directly supports effective session counting over session-local caches.",
    },
    (
        "2026-06-06-cross-paper-synthesis-reusable-work-needs-visible-lifetime-contracts",
        "wal_before_visibility",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Reusable work supports WAL visibility only when deterministic replay inputs and validation contracts remain explicit.",
    },
    (
        "2026-06-06-cross-paper-synthesis-reusable-work-needs-visible-lifetime-contracts",
        "learned_optimizer_advisor",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Learned reuse is valid only when hot-path state lifetime, reuse, and retirement contracts are explicit.",
    },
    (
        "2026-06-06-geminifs-makes-gpu-storage-metadata-explicit-enough-for-device-side-io",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Device-side IO publication needs crash tests around WAL/checkpoint fences before route visibility can rely on it.",
    },
    (
        "2026-06-06-geminifs-makes-gpu-storage-metadata-explicit-enough-for-device-side-io",
        "db_owned_cold_objects",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "DB-owned segment files are benchmarkable but require stale-descriptor tests across moves, resizes, and compaction.",
    },
    (
        "2026-06-06-geminifs-makes-gpu-storage-metadata-explicit-enough-for-device-side-io",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Segment-map publication needs latency and crash-boundary measurements before becoming an immutable route-root contract.",
    },
    (
        "2026-06-07-nemo-treats-partial-write-set-knowledge-as-a-contention-throttle",
        "resource_dag_scheduling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The conflict cue describes what NEMO schedules around; retained evidence supports dependency-aware resource ordering.",
    },
    (
        "2026-06-07-nemo-treats-partial-write-set-knowledge-as-a-contention-throttle",
        "dependency_witnesses",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "NEMO-style dependency witnesses apply only under deterministic serializability, lazy block commit, and smart-contract objects.",
    },
    (
        "2026-06-07-cross-paper-synthesis-route-hints-need-measured-trust",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Hint-driven fallback needs measured precision, retry, stale-route rejection, fallback count, and p99 degradation evidence.",
    },
    (
        "2026-06-07-falcon-makes-persistent-cache-durability-a-write-amplification-problem",
        "immutable_route_roots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Durable route descriptors need a measured commit window before forcing every route or manifest update to media.",
    },
    (
        "2026-06-07-falcon-makes-persistent-cache-durability-a-write-amplification-problem",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-local durable windows require write-amplification and latency benchmarks before owner bundling adopts them.",
    },
    (
        "2026-06-07-wfe-bounds-descriptor-retirement-with-helper-assisted-eras",
        "deficit_fairness",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Helper-assisted fairness applies only if allocation and retirement paths help slow readers before era advancement.",
    },
    (
        "2026-06-07-cross-paper-synthesis-tier-movement-needs-semantic-guards",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Tier movement is valid only when snapshot correctness, reclamation protection, and intended-tier admission all agree.",
    },
    (
        "2026-06-07-polysi-makes-snapshot-claims-black-box-testable",
        "retained_gpu_snapshots",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Retained snapshot claims require external witness histories and stress runs before adoption.",
    },
    (
        "2026-06-07-polysi-makes-snapshot-claims-black-box-testable",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "WAL visibility boundaries need route-audit stress histories before the mechanism can be trusted.",
    },
    (
        "2026-06-07-host-interconnects-need-route-credits-not-just-bandwidth-counters",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Owner-ring routing needs queue wait, topology contention, response backlog, and p99 benchmark telemetry.",
    },
    (
        "2026-06-07-host-interconnects-need-route-credits-not-just-bandwidth-counters",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Logical-session scaling requires stress tests that expose topology-induced latency inflation.",
    },
    (
        "2026-06-07-deferred-reference-counting-makes-descriptor-lifetime-automatic-but-bounded",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Descriptor reclamation requires route-lookup, atomic-write, retired-backlog, and invalidation-latency benchmarks.",
    },
    (
        "2026-06-07-deferred-reference-counting-makes-descriptor-lifetime-automatic-but-bounded",
        "effective_session_counting",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Effective session counting needs 1M logical-session benchmarks over bounded workers before adopting deferred counts.",
    },
    (
        "2026-06-07-deferred-reference-counting-makes-descriptor-lifetime-automatic-but-bounded",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Snapshot protection benefits are benchmark evidence and need GPU DB validation before shaping frontier policy.",
    },
    (
        "2026-06-07-deferred-reference-counting-makes-descriptor-lifetime-automatic-but-bounded",
        "owner_ring_bundling",
    ): {
        "relation_type": "supports",
        "relation_review_note": "The retained evidence supports bounded owner-accounted cleanup instead of hiding cleanup in latency-sensitive reads.",
    },
    (
        "2026-06-07-b3-turns-crash-consistency-into-bounded-witness-generation",
        "semantic_crash_oracle",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Crash consistency depends on generated operation sequences, recovery, and oracle comparison tests.",
    },
    (
        "2026-06-07-b3-turns-crash-consistency-into-bounded-witness-generation",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Catalog generations and route descriptors need crash-test coverage before informing reclamation policy.",
    },
    (
        "2026-06-07-cross-paper-synthesis-recovery-needs-compact-witnesses",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "The recovery synthesis names WAL and manifest crash recovery as explicit benchmark gates.",
    },
    (
        "2026-06-07-ccfs-makes-durability-ordering-a-per-stream-contract",
        "semantic_crash_oracle",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Per-stream durability evidence is valid only if durable free-and-retire boundaries prevent reused-id observations.",
    },
    (
        "2026-06-07-ccfs-makes-durability-ordering-a-per-stream-contract",
        "dependency_witnesses",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "False-dependency behavior needs benchmarks before dependency-stream separation can inform route witnesses.",
    },
    (
        "2026-06-07-chardonnay-turns-epoch-snapshots-into-pre-lock-admission",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Tier admission needs retained-read latency, write throughput, version retention, publisher overhead, and stale-read risk measurements.",
    },
    (
        "2026-06-07-cross-paper-synthesis-recovery-needs-semantic-state-spaces",
        "snapshot_frontier_vectors",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Semantic crash-state grouping is an alternative evidence shape to raw snapshot frontier state enumeration.",
    },
    (
        "2026-06-07-cross-paper-synthesis-recovery-needs-semantic-state-spaces",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Semantic stream separation needs hot-commit p99 measurement under checkpoint and cold-tier compaction pressure.",
    },
    (
        "2026-06-07-holon-turns-independent-tuning-knobs-into-joint-route-actions",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Holon does not evaluate GPU memory, pinned buffers, NVMe tiers, or logical-session admission, so transfer requires benchmarks.",
    },
    (
        "2026-06-07-lsnvmm-makes-the-log-the-home-location",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "only_valid_if",
        "relation_review_note": "Log-structured reclamation is valid only if no snapshot, DMA, replay cursor, or route descriptor can reach recycled chunks.",
    },
    (
        "2026-06-07-cross-paper-synthesis-warm-tiers-need-logical-witnesses-and-movable-homes",
        "wal_before_visibility",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Warm-tier movement requires crash and recovery benchmarks proving physical movement never breaks logical records.",
    },
    (
        "2026-06-07-cross-paper-synthesis-warm-tiers-need-logical-witnesses-and-movable-homes",
        "log_structured_warm_tier",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Appending a new physical copy and publishing a mapping is an alternative to fixed-home warm-tier updates.",
    },
    (
        "2026-06-07-cross-paper-synthesis-warm-tiers-need-logical-witnesses-and-movable-homes",
        "multi_tier_placement",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Movable warm-tier homes need stable-id, publication, cleaner, retained-snapshot, and route-holon benchmarks before adoption.",
    },
    (
        "2026-06-07-hostcc-makes-host-congestion-a-local-control-loop",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Host-local owner-ring pressure control needs measured admission changes before NIC, PCIe, pinned-buffer, or memory queues saturate.",
    },
    (
        "2026-06-07-vweaver-turns-mvcc-scan-visibility-into-an-access-path",
        "immutable_route_roots",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "vWeaver cautions that immutable resident snapshots also need a compact visibility directory, not only route-root buffers.",
    },
    (
        "2026-06-07-vweaver-turns-mvcc-scan-visibility-into-an-access-path",
        "wal_before_visibility",
    ): {
        "relation_type": "warns_against",
        "relation_review_note": "vWeaver warns that WAL/catalog/resident generations are insufficient unless MVCC visibility search metadata is explicit.",
    },
    (
        "2026-06-07-cross-paper-synthesis-freshness-needs-explicit-search-metadata",
        "cpu_fallback_policy",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Fallback routing needs route-witness, visibility-directory, and mixed-pressure benchmarks before freshness policy adoption.",
    },
    (
        "2026-06-07-zen-minimizes-persistent-write-amplification-by-moving-cc-metadata-out-of-nvm",
        "cost_based_route_optimizer",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "ZEN's split durable-payload and volatile-CC design requires route-cost tests before optimizer policy can transfer.",
    },
    (
        "2026-06-07-mod-makes-durability-fast-by-minimizing-ordered-persist-barriers",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "MOD-style structural sharing needs cross-owner catalog and residency metadata measurements before owner bundling adoption.",
    },
    (
        "2026-06-07-mod-makes-durability-fast-by-minimizing-ordered-persist-barriers",
        "dependency_witnesses",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Out-of-place durable shadow publication is an alternative to dependency witnesses built around undo/redo overwrite recovery.",
    },
    (
        "2026-06-07-graphene-schedules-scarce-resources-by-troublesome-work-first",
        "owner_ring_bundling",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Bounded troublesome-work selection is an alternative to rescoring an entire owner-ring backlog on every wakeup.",
    },
    (
        "2026-06-07-graphene-schedules-scarce-resources-by-troublesome-work-first",
        "htap_freshness_router",
    ): {
        "relation_type": "alternative_to",
        "relation_review_note": "Graphene frames parsed route work as a resource DAG rather than independent freshness-router queue entries.",
    },
    (
        "2026-06-07-asap-moves-persistence-waits-behind-dependency-witnesses",
        "bounded_descriptor_reclamation",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Descriptor compaction and chain collapse need measurements that preserve recovery order while bounding retired metadata.",
    },
    (
        "2026-06-07-tfc-makes-credits-the-queueing-boundary",
        "owner_ring_bundling",
    ): {
        "relation_type": "benchmark_required",
        "relation_review_note": "Per-boundary token budgets for owner, retained-read, GPU, refresh, and response rings require prototype validation.",
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


def clean_citation_value(value: str) -> str:
    value = value.replace("\n", " ")
    value = value.replace("`", "")
    return compact_whitespace(value)


def citation_title(citation: str, fallback_title: str) -> str:
    match = QUOTED_TITLE_RE.search(citation)
    if match:
        return compact_whitespace(match.group("title"))
    return fallback_title


def citation_authors(citation: str) -> list[str]:
    if not citation:
        return []
    before_title = citation.split('"', 1)[0]
    before_title = re.sub(r"\bet al\.\s*$", "et al.", before_title).strip(" .")
    if not before_title:
        return []
    before_title = before_title.replace(" and ", ", ")
    return [part.strip() for part in before_title.split(",") if part.strip()]


def citation_venue(citation: str) -> str:
    if not citation:
        return ""
    after_title = citation.split('"', 2)[2] if citation.count('"') >= 2 else citation
    after_title = re.sub(r"\bDOI:\s*.*$", "", after_title, flags=re.IGNORECASE)
    after_title = re.sub(r"\bRetrieved\s+\d{4}-\d{2}-\d{2}.*$", "", after_title, flags=re.IGNORECASE)
    after_title = URL_RE.sub("", after_title)
    legacy_venue = compact_whitespace(re.sub(r"\b(19|20)\d{2}\b.*$", "", after_title).strip(" .,:;"))
    if legacy_venue:
        return legacy_venue
    after_title = re.sub(r"^\s*[.,;:]?\s*\b(19|20)\d{2}\b\s*", "", after_title)
    after_title = re.sub(r"\bpages?\s+.*$", "", after_title, flags=re.IGNORECASE)
    after_title = re.sub(r"\bpp\.\s+.*$", "", after_title, flags=re.IGNORECASE)
    return compact_whitespace(after_title.strip(" .,:;"))


def normalize_identifier(value: str) -> str:
    return value.strip().rstrip(".,;").lower()


def normalize_arxiv(value: str) -> str:
    return re.sub(r"v\d+$", "", normalize_identifier(value))


def citation_url(urls: list[str], doi: str, arxiv: str) -> str:
    if urls:
        return urls[0]
    if arxiv:
        return f"https://arxiv.org/abs/{arxiv}"
    if doi:
        return f"https://doi.org/{doi}"
    return ""


def citation_year(citation: str, entry_date: str) -> str:
    citation_without_urls = re.sub(r"\bRetrieved\s+\d{4}-\d{2}-\d{2}.*$", "", citation, flags=re.IGNORECASE)
    citation_without_urls = URL_RE.sub("", citation_without_urls)
    citation_without_urls = re.sub(r"\barxiv:?\s*\d{4}\.\d{4,5}(?:v\d+)?", "", citation_without_urls, flags=re.IGNORECASE)
    found = [
        match.group(0)
        for match in YEAR_RE.finditer(citation_without_urls)
        if 1990 <= int(match.group(0)) <= int(entry_date[:4])
    ]
    if found:
        return max(found)
    return entry_date[:4]


def paper_identity_missing_fields(identity: dict) -> list[str]:
    required_fields = ("doi", "arxiv", "url", "venue")
    return [field for field in required_fields if not identity[field]]


def paper_identifier_audit(identity: dict, citation: str) -> dict:
    doi = identity["doi"]
    arxiv = identity["arxiv"]
    venue = identity["venue"].lower()
    citation = citation.lower()
    title = identity["title"]

    if doi:
        doi_status = "present"
        doi_review = "citation_supplies_doi"
    elif arxiv and (
        "submitted" in citation
        or "preprint" in citation
        or "arxiv" in venue
        or "final venue details are unknown" in citation
    ):
        doi_status = "not_expected_yet"
        doi_review = "arxiv_preprint_without_final_doi"
    elif arxiv:
        doi_status = "secondary_missing"
        doi_review = "arxiv_record_without_doi_in_journal_citation"
    else:
        doi_status = "needs_identifier_review"
        doi_review = "no_doi_or_arxiv_identifier_in_journal_citation"

    if arxiv:
        arxiv_status = "present"
        arxiv_review = "citation_supplies_arxiv"
    elif doi:
        arxiv_status = "secondary_missing"
        arxiv_review = "publisher_record_has_doi_but_no_arxiv_in_journal_citation"
    else:
        arxiv_status = "needs_identifier_review"
        arxiv_review = "no_doi_or_arxiv_identifier_in_journal_citation"

    return {
        "doi_status": doi_status,
        "doi_review": doi_review,
        "arxiv_status": arxiv_status,
        "arxiv_review": arxiv_review,
        "needs_identifier_review": doi_status == "needs_identifier_review"
        or arxiv_status == "needs_identifier_review",
        "review_status": "generated_identifier_audit",
        "title": title,
    }


def apply_identifier_review_override(identity: dict, entry_id: str) -> None:
    override = IDENTIFIER_REVIEW_OVERRIDES.get(entry_id)
    if not override:
        return
    if override.get("doi"):
        identity["doi"] = normalize_identifier(override["doi"])
        if not identity["url"]:
            identity["url"] = citation_url([], identity["doi"], identity["arxiv"])
    if override.get("arxiv"):
        identity["arxiv"] = normalize_arxiv(override["arxiv"])
        if not identity["url"]:
            identity["url"] = citation_url([], identity["doi"], identity["arxiv"])

    audit = identity["identifier_audit"]
    if identity["doi"]:
        audit["doi_status"] = "present"
        audit["doi_review"] = override.get("doi_review", "repaired_identifier_review")
    elif override.get("doi_status"):
        audit["doi_status"] = override["doi_status"]
        audit["doi_review"] = override.get("doi_review", audit["doi_review"])

    if identity["arxiv"]:
        audit["arxiv_status"] = "present"
        audit["arxiv_review"] = override.get("arxiv_review", "repaired_identifier_review")
    elif override.get("arxiv_status"):
        audit["arxiv_status"] = override["arxiv_status"]
        audit["arxiv_review"] = override.get("arxiv_review", audit["arxiv_review"])
    elif identity["doi"] and audit["arxiv_status"] == "needs_identifier_review":
        audit["arxiv_status"] = "secondary_missing"
        audit["arxiv_review"] = "publisher_record_has_doi_but_no_arxiv_in_journal_citation"

    audit["needs_identifier_review"] = (
        audit["doi_status"] == "needs_identifier_review"
        or audit["arxiv_status"] == "needs_identifier_review"
    )
    audit["review_status"] = "reviewed_identifier_repair"
    audit["review_source"] = override.get("review_source", "")


def build_paper_identity(entry: dict) -> dict | None:
    if entry["entry_type"] != "paper":
        return None
    citation = clean_citation_value(entry.get("citation", ""))
    title = citation_title(citation, entry["title"])
    doi_match = DOI_RE.search(citation)
    arxiv_match = ARXIV_RE.search(citation)
    urls = [url.rstrip(".,;") for url in URL_RE.findall(citation)]
    doi = normalize_identifier(doi_match.group("doi")) if doi_match else ""
    arxiv = normalize_arxiv(arxiv_match.group("arxiv")) if arxiv_match else ""
    year = citation_year(citation, entry["date"])
    identity_key = f"title:{slugify(title)}"
    identity = {
        "paper_id": f"paper-{year}-{slugify(title)}",
        "identity_key": identity_key,
        "title": title,
        "authors": citation_authors(citation),
        "venue": citation_venue(citation),
        "year": year,
        "doi": doi,
        "arxiv": arxiv,
        "url": citation_url(urls, doi, arxiv),
        "journal_entry_id": entry["id"],
        "source": "generated_from_journal_citation",
        "review_status": "generated_identity",
    }
    identity["identifier_audit"] = paper_identifier_audit(identity, citation)
    apply_identifier_review_override(identity, entry["id"])
    identity["missing_fields"] = paper_identity_missing_fields(identity)
    return identity


def classify_entry_type(title: str, citation: str) -> str:
    title_lower = title.lower()
    if not citation or "synthesis" in title_lower or "cross-paper synthesis" in title_lower:
        return "synthesis"
    return "paper"


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
        citation = extract_citation(body)
        entry_type = classify_entry_type(title, citation)
        entries.append(
            {
                "id": f"{date}-{slugify(title)}",
                "date": date,
                "title": title,
                "entry_type": entry_type,
                "category": category,
                "relevance_tags": tags,
                "citation": citation,
                "body": body,
            }
        )
    return entries


def extract_citation(body: str) -> str:
    lines = body.splitlines()
    citation_lines: list[str] = []
    in_citation = False
    for line in lines:
        if line.startswith("**Citation:**"):
            in_citation = True
            citation_lines.append(line.removeprefix("**Citation:**").strip())
            continue
        if not in_citation:
            continue
        if not line.strip():
            break
        if line.startswith("**Category:**") or line.startswith("Category:"):
            break
        citation_lines.append(line.strip())
    return compact_whitespace(" ".join(part for part in citation_lines if part))


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


def normalize_paper_records(paper_identities: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for identity in paper_identities:
        groups[identity["identity_key"]].append(identity)

    paper_records: list[dict] = []
    duplicate_groups: list[dict] = []
    for _, identities in sorted(groups.items(), key=lambda item: item[1][0]["paper_id"]):
        identities = sorted(identities, key=lambda item: item["journal_entry_id"])
        canonical = identities[0]
        with_doi = next((identity for identity in identities if identity["doi"]), canonical)
        with_arxiv = next((identity for identity in identities if identity["arxiv"]), canonical)
        with_url = next((identity for identity in identities if identity["url"]), canonical)
        with_venue = next((identity for identity in identities if identity["venue"]), canonical)
        with_authors = next((identity for identity in identities if identity["authors"]), canonical)
        doi_audit = next(
            (
                identity["identifier_audit"]
                for identity in identities
                if identity["identifier_audit"]["doi_status"] == "present"
            ),
            with_doi["identifier_audit"],
        )
        arxiv_audit = next(
            (
                identity["identifier_audit"]
                for identity in identities
                if identity["identifier_audit"]["arxiv_status"] == "present"
            ),
            with_arxiv["identifier_audit"],
        )
        journal_entry_ids = [identity["journal_entry_id"] for identity in identities]
        record = {
            "paper_id": canonical["paper_id"],
            "title": canonical["title"],
            "authors": with_authors["authors"],
            "venue": with_venue["venue"],
            "year": canonical["year"],
            "doi": with_doi["doi"],
            "arxiv": with_arxiv["arxiv"],
            "url": with_url["url"],
            "journal_entry_ids": journal_entry_ids,
            "identity_key": canonical["identity_key"],
            "duplicate_of": "",
            "source": canonical["source"],
            "review_status": canonical["review_status"],
            "identifier_audit": {
                "doi_status": doi_audit["doi_status"],
                "doi_review": doi_audit["doi_review"],
                "arxiv_status": arxiv_audit["arxiv_status"],
                "arxiv_review": arxiv_audit["arxiv_review"],
                "needs_identifier_review": doi_audit["doi_status"] == "needs_identifier_review"
                or arxiv_audit["arxiv_status"] == "needs_identifier_review",
                "review_status": "generated_identifier_audit",
            },
            "missing_fields": paper_identity_missing_fields(
                {
                    "doi": with_doi["doi"],
                    "arxiv": with_arxiv["arxiv"],
                    "url": with_url["url"],
                    "venue": with_venue["venue"],
                }
            ),
        }
        paper_records.append(record)
        if len(identities) > 1:
            duplicate_groups.append(
                {
                    "paper_id": canonical["paper_id"],
                    "journal_entry_ids": journal_entry_ids,
                    "duplicate_count": len(identities),
                    "identity_key": canonical["identity_key"],
                }
            )
    return paper_records, duplicate_groups


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
    paper_identity_counts: Counter = Counter()
    paper_identity_missing_counts: Counter = Counter()
    paper_identity_missing_field_sets: Counter = Counter()
    paper_identifier_audit_counts: Counter = Counter()
    paper_identity_missing_audit: list[dict] = []
    paper_identifier_review_audit: list[dict] = []
    paper_identities: list[dict] = []
    unlinked: list[str] = []

    for entry in entries:
        paper_identity = build_paper_identity(entry)
        if paper_identity:
            paper_identities.append(paper_identity)
            paper_identity_counts["paper_entries_with_identity"] += 1
            if paper_identity["doi"]:
                paper_identity_counts["paper_entries_with_doi"] += 1
            else:
                paper_identity_missing_counts["missing_doi"] += 1
            if paper_identity["arxiv"]:
                paper_identity_counts["paper_entries_with_arxiv"] += 1
            else:
                paper_identity_missing_counts["missing_arxiv"] += 1
            if paper_identity["url"]:
                paper_identity_counts["paper_entries_with_url"] += 1
            else:
                paper_identity_missing_counts["missing_url"] += 1
            if paper_identity["venue"]:
                paper_identity_counts["paper_entries_with_venue"] += 1
            else:
                paper_identity_missing_counts["missing_venue"] += 1
            missing_fields = paper_identity["missing_fields"]
            missing_key = ",".join(missing_fields) if missing_fields else "none"
            paper_identity_missing_field_sets[missing_key] += 1
            identifier_audit = paper_identity["identifier_audit"]
            paper_identifier_audit_counts[f"doi_{identifier_audit['doi_status']}"] += 1
            paper_identifier_audit_counts[f"arxiv_{identifier_audit['arxiv_status']}"] += 1
            if identifier_audit["needs_identifier_review"]:
                paper_identifier_review_audit.append(
                    {
                        "journal_entry_id": entry["id"],
                        "paper_id": paper_identity["paper_id"],
                        "title": paper_identity["title"],
                        "doi_status": identifier_audit["doi_status"],
                        "arxiv_status": identifier_audit["arxiv_status"],
                        "doi_review": identifier_audit["doi_review"],
                        "arxiv_review": identifier_audit["arxiv_review"],
                    }
                )
            if missing_fields:
                paper_identity_missing_audit.append(
                    {
                        "journal_entry_id": entry["id"],
                        "paper_id": paper_identity["paper_id"],
                        "title": paper_identity["title"],
                        "missing_fields": missing_fields,
                        "identifier_audit": identifier_audit,
                        "citation": paper_identity["citation"] if "citation" in paper_identity else entry.get("citation", ""),
                    }
                )
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
                "paper_identity": paper_identity,
                "mechanism_links": links,
            }
        )

    mechanisms_without_links = sorted(mechanism_ids - set(mechanism_counts))
    paper_records, duplicate_groups = normalize_paper_records(paper_identities)
    return {
        "schema": "gpu-db-research-paper-mechanism-links-v9",
        "description": "Generated traceability from literature journal entries to architecture mechanisms, including generated evidence spans, evidence quality reasons, typed paper-mechanism relations, and normalized paper identity records. Review low-confidence, fallback, and incomplete identity records before making architectural commitments.",
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
            "paper_identity_counts": dict(sorted(paper_identity_counts.items())),
            "paper_identifier_audit_counts": dict(sorted(paper_identifier_audit_counts.items())),
            "paper_identity_missing_counts": dict(sorted(paper_identity_missing_counts.items())),
            "paper_identity_missing_field_sets": dict(sorted(paper_identity_missing_field_sets.items())),
            "paper_records": len(paper_records),
            "paper_duplicate_groups": len(duplicate_groups),
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
        "paper_records": paper_records,
        "paper_duplicate_groups": duplicate_groups,
        "paper_identity_missing_audit": sorted(
            paper_identity_missing_audit,
            key=lambda item: (
                "url" not in item["missing_fields"],
                "venue" not in item["missing_fields"],
                len(item["missing_fields"]),
                item["journal_entry_id"],
            ),
        ),
        "paper_identifier_review_audit": sorted(
            paper_identifier_review_audit,
            key=lambda item: item["journal_entry_id"],
        ),
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
    paper_identity_counts = index["summary"]["paper_identity_counts"]
    paper_identifier_audit_counts = index["summary"]["paper_identifier_audit_counts"]
    paper_identity_missing_counts = index["summary"]["paper_identity_missing_counts"]
    paper_identity_missing_field_sets = index["summary"]["paper_identity_missing_field_sets"]
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

    lines.extend(["", "## Paper Identity Records", ""])
    lines.append(f"- normalized paper records: {index['summary']['paper_records']}")
    lines.append(f"- paper entries with generated identity: {paper_identity_counts.get('paper_entries_with_identity', 0)}")
    lines.append(f"- duplicate identity groups: {index['summary']['paper_duplicate_groups']}")
    lines.append("")
    lines.append("Generated identity field counts:")
    for field, count in sorted(paper_identity_counts.items()):
        lines.append(f"- {field}: {count}")
    lines.append("")
    lines.append("Generated identity missing-field counts:")
    for field, count in sorted(paper_identity_missing_counts.items()):
        lines.append(f"- {field}: {count}")
    lines.append("")
    lines.append("Generated identity missing-field sets:")
    for fields, count in sorted(
        paper_identity_missing_field_sets.items(),
        key=lambda item: (item[0] == "none", item[0]),
    ):
        lines.append(f"- {fields}: {count}")
    lines.append("")
    lines.append("Generated DOI/arXiv identifier audit:")
    for status, count in sorted(paper_identifier_audit_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.append(
        f"- actionable_identifier_review: {len(index['paper_identifier_review_audit'])}"
    )
    lines.append("")
    lines.append("Actionable DOI/arXiv identifier audit:")
    if index["paper_identifier_review_audit"]:
        for item in index["paper_identifier_review_audit"][:40]:
            lines.append(
                f"- `{item['journal_entry_id']}` -> `{item['paper_id']}`: "
                f"doi={item['doi_status']}, arxiv={item['arxiv_status']}: {item['title']}"
            )
        if len(index["paper_identifier_review_audit"]) > 40:
            lines.append(f"- ... {len(index['paper_identifier_review_audit']) - 40} more")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Missing identity metadata audit:")
    if index["paper_identity_missing_audit"]:
        for item in index["paper_identity_missing_audit"][:40]:
            fields = ", ".join(item["missing_fields"])
            lines.append(
                f"- `{item['journal_entry_id']}` -> `{item['paper_id']}` missing {fields}: {item['title']}"
            )
        if len(index["paper_identity_missing_audit"]) > 40:
            lines.append(f"- ... {len(index['paper_identity_missing_audit']) - 40} more")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Duplicate identity audit:")
    if index["paper_duplicate_groups"]:
        for group in index["paper_duplicate_groups"][:80]:
            entries = ", ".join(group["journal_entry_ids"])
            lines.append(f"- `{group['paper_id']}` ({group['duplicate_count']} entries): {entries}")
        if len(index["paper_duplicate_groups"]) > 80:
            lines.append(f"- ... {len(index['paper_duplicate_groups']) - 80} more")
    else:
        lines.append("- none")

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
