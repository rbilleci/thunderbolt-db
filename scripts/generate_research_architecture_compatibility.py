#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import shorten


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

    if errors:
        raise SystemExit("\n".join(errors))


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
        "",
        "## Summary",
        "",
        f"- mechanisms: {len(mechanisms)}",
        f"- edges: {len(edges)}",
    ]
    for edge_type in sorted(edge_counts):
        lines.append(f"- {edge_type}: {edge_counts[edge_type]}")

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
    mechanisms = {item["id"]: item for item in mechanism_doc["mechanisms"]}
    edge_doc = load_json(args.edges)
    validate(mechanisms, edge_doc["edges"])
    write_markdown(mechanisms, edge_doc, args.output)


if __name__ == "__main__":
    main()
