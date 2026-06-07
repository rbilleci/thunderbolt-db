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


def build_index(entries: list[dict], mechanisms: dict) -> dict:
    mechanism_ids = {item["id"] for item in mechanisms["mechanisms"]}
    records: list[dict] = []
    mechanism_counts: Counter = Counter()
    confidence_counts: Counter = Counter()
    review_status_counts: Counter = Counter()
    review_priority_counts: Counter = Counter()
    type_counts: Counter = Counter()
    unlinked: list[str] = []

    for entry in entries:
        links = link_entry(entry, mechanism_ids)
        if not links:
            unlinked.append(entry["id"])
        for link in links:
            confidence = link["confidence"]
            link["review_status"] = REVIEW_STATUS_BY_CONFIDENCE[confidence]
            link["review_priority"] = REVIEW_PRIORITY_BY_CONFIDENCE[confidence]
            mechanism_counts[link["mechanism_id"]] += 1
            confidence_counts[confidence] += 1
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
        "schema": "gpu-db-research-paper-mechanism-links-v1",
        "description": "Generated traceability from literature journal entries to architecture mechanisms. Review low-confidence and fallback links before making architectural commitments.",
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
            "review_status_counts": dict(sorted(review_status_counts.items())),
            "review_priority_counts": dict(sorted(review_priority_counts.items())),
            "links_requiring_review": review_status_counts["pending_low_confidence_review"]
            + review_status_counts["manual_review_required"],
            "low_confidence_links": confidence_counts["low"],
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
    review_status_counts = index["summary"]["review_status_counts"]
    review_priority_counts = index["summary"]["review_priority_counts"]
    records = index["records"]
    fallback_records = [
        record
        for record in records
        if any(link["confidence"] == "needs_review" for link in record["mechanism_links"])
    ]
    low_confidence_records = [
        record
        for record in records
        if any(link["confidence"] == "low" for link in record["mechanism_links"])
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

    lines.extend(["", "## Review Triage", ""])
    lines.append(f"- links requiring review: {index['summary']['links_requiring_review']}")
    lines.append(f"- low-confidence links: {index['summary']['low_confidence_links']}")
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

    lines.extend(["", "## Entries With Low-Confidence Links", ""])
    if low_confidence_records:
        for record in low_confidence_records[:200]:
            links = ", ".join(
                link["mechanism_id"]
                for link in record["mechanism_links"]
                if link["confidence"] == "low"
            )
            lines.append(f"- `{record['id']}` -> {links}")
        if len(low_confidence_records) > 200:
            lines.append(f"- ... {len(low_confidence_records) - 200} more")
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
