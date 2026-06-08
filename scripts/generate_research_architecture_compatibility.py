#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import shorten


MECHANISM_SCHEMA = "gpu-db-research-mechanisms-v2"

DECISION_STATUSES = {
    "adopt_now",
    "prototype",
    "benchmark_only",
    "defer",
    "reject",
    "unknown",
}

DECISION_STATUS_DESCRIPTIONS = {
    "adopt_now": "make this a baseline architecture invariant",
    "prototype": "build the first implementation behind an explicit proof gate",
    "benchmark_only": "keep as an experiment until measurement decides adoption",
    "defer": "postpone until prerequisite mechanisms or product pressure exist",
    "reject": "do not include in the architecture",
    "unknown": "not enough reviewed evidence to decide yet",
}

DECISION_OVERRIDES = {
    "wal_before_visibility": {
        "decision_status": "adopt_now",
        "decision_rationale": "Durability before SQL-visible publication is a non-negotiable correctness invariant, and many downstream route/root/fallback mechanisms require it.",
    },
    "immutable_route_roots": {
        "decision_status": "adopt_now",
        "decision_rationale": "Compact immutable generations give the architecture a shared publication contract for routes, catalogs, residency, and visibility.",
    },
    "dependency_witnesses": {
        "decision_status": "prototype",
        "decision_rationale": "Async publication needs explicit dependency proof, but the witness representation should be prototyped before fixing hot-path shape.",
    },
    "semantic_crash_oracle": {
        "decision_status": "adopt_now",
        "decision_rationale": "Crash-state validation is required to prove WAL, route-root, catalog, and residency publication boundaries before optimized paths are trusted.",
    },
    "isolation_trace_oracle": {
        "decision_status": "adopt_now",
        "decision_rationale": "Fallback and retained-snapshot routes need continuous external validation that returned versions match the declared isolation contract.",
    },
    "retained_gpu_snapshots": {
        "decision_status": "prototype",
        "decision_rationale": "Resident read generations are central to the GPU value proposition, but freshness, HBM pressure, and version retention need prototype evidence.",
    },
    "snapshot_frontier_vectors": {
        "decision_status": "prototype",
        "decision_rationale": "Cross-owner visibility proof is required for retained snapshots, while the scalar/vector split needs implementation evidence.",
    },
    "mvcc_gc_frontiers": {
        "decision_status": "prototype",
        "decision_rationale": "Exact active-reader frontiers are needed to bound old versions, but retention behavior under GPU snapshots must be measured.",
    },
    "bounded_descriptor_reclamation": {
        "decision_status": "prototype",
        "decision_rationale": "Immutable publication requires safe descriptor lifetime, while the specific hazard/era/epoch strategy should follow churn measurements.",
    },
    "stable_handle_indirection": {
        "decision_status": "prototype",
        "decision_rationale": "Tier movement and compaction need stable identity, but lookup overhead must be measured before the handle shape is fixed.",
    },
    "vector_credit_admission": {
        "decision_status": "adopt_now",
        "decision_rationale": "Route choice must expose bounded resource budgets up front so GPU, CPU, WAL, buffer, and response queues cannot hide overload.",
    },
    "effective_session_counting": {
        "decision_status": "prototype",
        "decision_rationale": "Large logical-session counts are a target workload, but active-flow accounting needs a simulator before adoption as a fixed runtime rule.",
    },
    "owner_ring_bundling": {
        "decision_status": "prototype",
        "decision_rationale": "Owner-local bounded drains are promising for low-allocation scheduling, with selection policy and skip telemetry still needing proof.",
    },
    "resource_dag_scheduling": {
        "decision_status": "benchmark_only",
        "decision_rationale": "DAG scheduling could improve scarce-resource packing, but estimation errors can hurt p99 and must be benchmarked against simpler queues.",
    },
    "deficit_fairness": {
        "decision_status": "prototype",
        "decision_rationale": "Fairness counters are needed to bound batching and DAG scheduling bias, but policy constants should come from mixed-class measurements.",
    },
    "same_shape_microbatching": {
        "decision_status": "prototype",
        "decision_rationale": "Repeated route shapes are a core GPU execution opportunity, while batch limits need latency and occupancy curves before adoption.",
    },
    "gpu_oltp_conflict_ordering": {
        "decision_status": "benchmark_only",
        "decision_rationale": "GPU conflict preprocessing is useful only for specific hot-key batch regimes and must compete with CPU OCC and owner serialization.",
    },
    "deterministic_hot_write_templates": {
        "decision_status": "benchmark_only",
        "decision_rationale": "Hot-write templates may beat abort/retry loops for narrow key shapes, but they should remain benchmark-gated until workload fit is proven.",
    },
    "cpu_fallback_policy": {
        "decision_status": "adopt_now",
        "decision_rationale": "Every accelerated route needs an explicit safe fallback reason so correctness and latency budgets survive stale residency or resource pressure.",
    },
    "htap_freshness_router": {
        "decision_status": "prototype",
        "decision_rationale": "Freshness-aware routing is fundamental to retained reads, but exact wait, refresh, and fallback thresholds need prototype feedback.",
    },
    "cost_based_route_optimizer": {
        "decision_status": "prototype",
        "decision_rationale": "The system needs deterministic route choice over CPU, GPU, retained, refresh, and cold paths before learning or advanced placement can matter.",
    },
    "learned_optimizer_advisor": {
        "decision_status": "defer",
        "decision_rationale": "Learned route advice should wait until deterministic route telemetry, guardrails, and baseline costs are stable.",
    },
    "multi_tier_placement": {
        "decision_status": "prototype",
        "decision_rationale": "HBM/DRAM/NVMe placement is necessary for capacity, but movement costs and hit-rate targets need simulator and prototype data.",
    },
    "log_structured_warm_tier": {
        "decision_status": "benchmark_only",
        "decision_rationale": "A rebuildable warm tier may simplify recovery, but append-map, root-publication, and in-place metadata variants need direct comparison.",
    },
    "db_owned_cold_objects": {
        "decision_status": "prototype",
        "decision_rationale": "Cold object manifests should be DB-owned to preserve recovery and routing semantics, with compaction and backup boundaries still to prove.",
    },
}

EDGE_SYMBOLS = {
    "requires": "R",
    "strengthens": "S",
    "compatible": "C",
    "tension": "T",
    "conflicts": "X",
    "alternative_to": "A",
    "unknown_needs_benchmark": "?",
}

EDGE_PRIORITY = {
    "conflicts": 6,
    "requires": 5,
    "tension": 4,
    "alternative_to": 3,
    "strengthens": 2,
    "compatible": 1,
    "unknown_needs_benchmark": 0,
}

LENSES = {
    "end_to_end_spine": [
        "wal_before_visibility",
        "immutable_route_roots",
        "dependency_witnesses",
        "semantic_crash_oracle",
        "snapshot_frontier_vectors",
        "retained_gpu_snapshots",
        "vector_credit_admission",
        "owner_ring_bundling",
        "cost_based_route_optimizer",
        "cpu_fallback_policy",
    ],
    "mvcc_and_freshness": [
        "retained_gpu_snapshots",
        "snapshot_frontier_vectors",
        "mvcc_gc_frontiers",
        "htap_freshness_router",
        "isolation_trace_oracle",
        "cpu_fallback_policy",
        "bounded_descriptor_reclamation",
    ],
    "runtime_admission": [
        "vector_credit_admission",
        "effective_session_counting",
        "owner_ring_bundling",
        "resource_dag_scheduling",
        "deficit_fairness",
        "same_shape_microbatching",
        "cpu_fallback_policy",
    ],
    "storage_and_recovery": [
        "wal_before_visibility",
        "immutable_route_roots",
        "dependency_witnesses",
        "semantic_crash_oracle",
        "stable_handle_indirection",
        "multi_tier_placement",
        "log_structured_warm_tier",
        "db_owned_cold_objects",
    ],
    "optimizer_and_execution": [
        "cost_based_route_optimizer",
        "learned_optimizer_advisor",
        "htap_freshness_router",
        "retained_gpu_snapshots",
        "same_shape_microbatching",
        "gpu_oltp_conflict_ordering",
        "deterministic_hot_write_templates",
        "resource_dag_scheduling",
    ],
}

RECOMMENDED_STACK = [
    "wal_before_visibility",
    "immutable_route_roots",
    "dependency_witnesses",
    "semantic_crash_oracle",
    "snapshot_frontier_vectors",
    "mvcc_gc_frontiers",
    "bounded_descriptor_reclamation",
    "stable_handle_indirection",
    "retained_gpu_snapshots",
    "vector_credit_admission",
    "effective_session_counting",
    "owner_ring_bundling",
    "resource_dag_scheduling",
    "deficit_fairness",
    "same_shape_microbatching",
    "htap_freshness_router",
    "cost_based_route_optimizer",
    "cpu_fallback_policy",
    "multi_tier_placement",
    "isolation_trace_oracle",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(mechanisms: dict[str, dict], edges: list[dict]) -> None:
    seen = set(mechanisms)
    errors: list[str] = []

    for edge in edges:
        if edge.get("from") not in seen:
            errors.append(f"unknown edge source: {edge.get('from')}")
        if edge.get("to") not in seen:
            errors.append(f"unknown edge target: {edge.get('to')}")
        if edge.get("type") not in EDGE_SYMBOLS:
            errors.append(f"unknown edge type: {edge.get('type')}")

    for lens_name, ids in LENSES.items():
        for mechanism_id in ids:
            if mechanism_id not in seen:
                errors.append(f"unknown mechanism in lens {lens_name}: {mechanism_id}")

    for mechanism_id in RECOMMENDED_STACK:
        if mechanism_id not in seen:
            errors.append(f"unknown mechanism in recommended stack: {mechanism_id}")

    missing_decisions = seen - set(DECISION_OVERRIDES)
    extra_decisions = set(DECISION_OVERRIDES) - seen
    for mechanism_id in sorted(missing_decisions):
        errors.append(f"missing decision status override: {mechanism_id}")
    for mechanism_id in sorted(extra_decisions):
        errors.append(f"unknown mechanism in decision status overrides: {mechanism_id}")

    for mechanism in mechanisms.values():
        status = mechanism.get("decision_status")
        rationale = mechanism.get("decision_rationale", "")
        if status not in DECISION_STATUSES:
            errors.append(f"unknown decision status for {mechanism['id']}: {status}")
        if not rationale:
            errors.append(f"missing decision rationale for {mechanism['id']}")

    if errors:
        raise SystemExit("\n".join(errors))


def apply_decision_statuses(mechanism_doc: dict) -> dict[str, dict]:
    mechanism_doc["schema"] = MECHANISM_SCHEMA
    for mechanism in mechanism_doc["mechanisms"]:
        decision = DECISION_OVERRIDES[mechanism["id"]]
        mechanism["decision_status"] = decision["decision_status"]
        mechanism["decision_rationale"] = decision["decision_rationale"]
    return {item["id"]: item for item in mechanism_doc["mechanisms"]}


def edge_index(edges: list[dict]) -> dict[tuple[str, str], list[dict]]:
    indexed: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for edge in edges:
        indexed[(edge["from"], edge["to"])].append(edge)
    return indexed


def strongest_edge(edges: list[dict]) -> dict | None:
    if not edges:
        return None
    return max(edges, key=lambda item: EDGE_PRIORITY[item["type"]])


def matrix_cell(indexed: dict[tuple[str, str], list[dict]], left: str, right: str) -> str:
    if left == right:
        return "--"
    forward = strongest_edge(indexed.get((left, right), []))
    reverse = strongest_edge(indexed.get((right, left), []))
    if forward and reverse:
        return f"{EDGE_SYMBOLS[forward['type']]}/{EDGE_SYMBOLS[reverse['type']]}"
    if forward:
        return EDGE_SYMBOLS[forward["type"]]
    if reverse:
        return f"{EDGE_SYMBOLS[reverse['type']]}<"
    return "."


def mechanism_label(mechanism: dict) -> str:
    return f"{mechanism['id']} ({mechanism['name']})"


def write_matrix(lines: list[str], title: str, ids: list[str], mechanisms: dict[str, dict], indexed: dict) -> None:
    lines.extend([f"## {title}", ""])
    headers = ["mechanism"] + ids
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row_id in ids:
        row = [row_id]
        for col_id in ids:
            row.append(matrix_cell(indexed, row_id, col_id))
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "",
        "Legend: `R` requires, `S` strengthens, `C` compatible, `T` tension, `X` conflicts, `A` alternative, `?` unknown. A trailing `<` means the strongest edge points from the column mechanism back to the row mechanism.",
        "",
        "Mechanisms in this lens:",
    ])
    for mechanism_id in ids:
        mechanism = mechanisms[mechanism_id]
        lines.append(f"- `{mechanism_id}`: {mechanism['summary']}")
    lines.append("")


def write_markdown(mechanisms: dict[str, dict], edge_doc: dict, output: Path) -> None:
    edges = edge_doc["edges"]
    indexed = edge_index(edges)
    edge_counts = Counter(edge["type"] for edge in edges)
    decision_counts = Counter(mechanism["decision_status"] for mechanism in mechanisms.values())
    mechanisms_by_layer: dict[str, list[dict]] = defaultdict(list)
    for mechanism in mechanisms.values():
        mechanisms_by_layer[mechanism["layer"]].append(mechanism)

    lines = [
        "# GPU DB Research Architecture Compatibility",
        "",
        "This document is generated from mechanism cards and typed compatibility",
        "edges. It turns the literature journal into a design-composition map:",
        "papers provide evidence, mechanisms provide contracts, and edges show",
        "which mechanisms can be combined in an end-to-end architecture.",
        "",
        "Regenerate with:",
        "",
        "```sh",
        "python3 scripts/generate_research_architecture_compatibility.py",
        "```",
        "",
        "## Source Layers",
        "",
        "- Layer 1: `docs/research/architecture-compatibility/mechanisms.json`",
        "- Layer 2: `docs/research/architecture-compatibility/compatibility-edges.json`",
        "- Layer 3: this generated compatibility view",
        "- Paper traceability: `docs/research/architecture-compatibility/paper-mechanism-links.json`",
        "- Paper coverage report: `docs/research/architecture-compatibility/paper-mechanism-coverage.md`",
        "",
        "## Summary",
        "",
        f"- mechanisms: {len(mechanisms)}",
        f"- edges: {len(edges)}",
    ]
    for edge_type in sorted(edge_counts):
        lines.append(f"- {edge_type}: {edge_counts[edge_type]}")
    lines.append("")
    lines.append("Decision status counts:")
    for status in sorted(DECISION_STATUSES):
        lines.append(f"- {status}: {decision_counts.get(status, 0)} ({DECISION_STATUS_DESCRIPTIONS[status]})")

    lines.extend(["", "## Recommended End-to-End Spine", ""])
    for mechanism_id in RECOMMENDED_STACK:
        mechanism = mechanisms[mechanism_id]
        lines.append(f"- `{mechanism_id}` ({mechanism['layer']}): {mechanism['summary']}")

    lines.extend(["", "## Mechanism Cards By Layer", ""])
    for layer in sorted(mechanisms_by_layer):
        lines.extend([f"### {layer.title()}", ""])
        for mechanism in sorted(mechanisms_by_layer[layer], key=lambda item: item["id"]):
            sources = ", ".join(mechanism.get("source_papers", []))
            provides = "; ".join(mechanism.get("provides", []))
            requires = "; ".join(mechanism.get("requires", []))
            lines.extend(
                [
                    f"#### `{mechanism['id']}` - {mechanism['name']}",
                    "",
                    mechanism["summary"],
                    "",
                    f"- evidence: {sources}",
                    f"- provides: {provides}",
                    f"- requires: {requires}",
                    f"- decision: {mechanism['decision_status']} - {mechanism['decision_rationale']}",
                    f"- benchmark: {mechanism['benchmark']}",
                    "",
                ]
            )

    lines.extend(["## Compatibility Matrices", ""])
    for lens_name, ids in LENSES.items():
        write_matrix(lines, lens_name.replace("_", " ").title(), ids, mechanisms, indexed)

    lines.extend(["## Tension And Alternative Zones", ""])
    interesting = [
        edge
        for edge in edges
        if edge["type"] in {"tension", "conflicts", "alternative_to", "unknown_needs_benchmark"}
    ]
    if interesting:
        for edge in interesting:
            source = mechanisms[edge["from"]]
            target = mechanisms[edge["to"]]
            reason = shorten(edge["reason"], width=180, placeholder="...")
            lines.append(
                f"- `{edge['type']}`: `{source['id']}` ({source['name']}) -> `{target['id']}` ({target['name']}): {reason}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Full Edge List", ""])
    for edge in sorted(edges, key=lambda item: (item["type"], item["from"], item["to"])):
        lines.append(f"- `{edge['from']}` --{edge['type']}--> `{edge['to']}`: {edge['reason']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GPU DB research architecture compatibility views")
    parser.add_argument(
        "--mechanisms",
        type=Path,
        default=Path("docs/research/architecture-compatibility/mechanisms.json"),
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("docs/research/architecture-compatibility/compatibility-edges.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/architecture-compatibility.md"),
    )
    args = parser.parse_args()

    mechanism_doc = load_json(args.mechanisms)
    mechanisms = apply_decision_statuses(mechanism_doc)
    edge_doc = load_json(args.edges)
    validate(mechanisms, edge_doc["edges"])
    args.mechanisms.write_text(json.dumps(mechanism_doc, indent=2) + "\n", encoding="utf-8")
    write_markdown(mechanisms, edge_doc, args.output)


if __name__ == "__main__":
    main()
