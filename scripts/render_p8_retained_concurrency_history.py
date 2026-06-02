#!/usr/bin/env python3
import csv
import json
import math
import os
import sys
from html import escape
from pathlib import Path


QUERIES = [
    ("order_line_count_all", "COUNT all"),
    ("order_line_lookup_ol_o_id_multi_column", "Lookup"),
]


def load_previous(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("series") == "persistent_tokio_postgres":
                item = dict(row)
                item["series"] = "persistent_one_shot"
                rows.append(item)
            else:
                rows.append(dict(row))
    return rows


def load_current(path):
    rows = []
    series = os.environ.get(
        "GPU_DB_P8_RETAINED_HISTORY_SERIES", "steady_state_after_response_path"
    )
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("kind") != "engine_backed_pgwire_concurrency_metric":
                continue
            rows.append(
                {
                    "series": series,
                    "query": item["query"],
                    "concurrency": str(item["concurrency"]),
                    "client_driver": item["client_driver"],
                    "request_count": str(item["request_count"]),
                    "requests_per_client": str(item["requests_per_client"]),
                    "warmup_requests_per_client": str(item.get("warmup_requests_per_client", 0)),
                    "p50_us": str(item["p50_us"]),
                    "p95_us": str(item["p95_us"]),
                    "p99_us": str(item["p99_us"]),
                    "throughput_qps": f'{float(item["throughput_qps"]):.6f}',
                    "error_count": str(item["error_count"]),
                    "phase_samples": str(item["phase_samples"]),
                    "phase_scheduler_queue_wait_avg_us": str(
                        item["phase_scheduler_queue_wait_avg_us"]
                    ),
                    "phase_scheduler_queue_wait_max_us": str(
                        item["phase_scheduler_queue_wait_max_us"]
                    ),
                    "phase_engine_execute_avg_us": str(item["phase_engine_execute_avg_us"]),
                    "phase_client_write_avg_us": str(item["phase_client_write_avg_us"]),
                    "phase_result_materialize_avg_us": str(
                        item["phase_result_materialize_avg_us"]
                    ),
                    "phase_retained_wall_avg_us": str(item["phase_retained_wall_avg_us"]),
                    "phase_cuda_event_avg_us": str(item["phase_cuda_event_avg_us"]),
                    "phase_d2h_avg_bytes": str(item["phase_d2h_avg_bytes"]),
                    "phase_kernel_delta_avg": str(item["phase_kernel_delta_avg"]),
                }
            )
    return rows


def numeric(row, key):
    value = row.get(key, "")
    if value in ("", "null", None):
        return 0.0
    return float(value)


def write_csv(path, rows):
    fields = [
        "series",
        "query",
        "concurrency",
        "client_driver",
        "request_count",
        "requests_per_client",
        "warmup_requests_per_client",
        "p50_us",
        "p95_us",
        "p99_us",
        "throughput_qps",
        "error_count",
        "phase_samples",
        "phase_scheduler_queue_wait_avg_us",
        "phase_scheduler_queue_wait_max_us",
        "phase_engine_execute_avg_us",
        "phase_client_write_avg_us",
        "phase_result_materialize_avg_us",
        "phase_retained_wall_avg_us",
        "phase_cuda_event_avg_us",
        "phase_d2h_avg_bytes",
        "phase_kernel_delta_avg",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def points(rows, query, series, key):
    selected = [
        (numeric(row, "concurrency"), numeric(row, key))
        for row in rows
        if row["query"] == query and row["series"] == series
    ]
    return sorted(selected)


def chart(path, title, rows, key, ylabel, phase_query=None):
    width, height = 920, 420
    left, right, top, bottom = 70, 30, 45, 65
    plot_w = width - left - right
    plot_h = height - top - bottom
    series_specs = [
        ("persistent_one_shot", "#3b82f6", "one-shot persistent"),
        ("steady_state_after_response_path", "#047857", "steady-state after"),
        ("retained_response_cache_fast_path", "#dc2626", "response cache fast path"),
    ]
    if phase_query:
        query_specs = [(phase_query, dict(QUERIES)[phase_query])]
    else:
        query_specs = QUERIES
    all_values = []
    for query, _ in query_specs:
        for series, _, _ in series_specs:
            all_values.extend(value for _, value in points(rows, query, series, key))
    max_y = max(all_values) if all_values else 1.0
    if max_y <= 0:
        max_y = 1.0
    max_y *= 1.12

    def x(value):
        return left + (math.log2(value) / 6.0) * plot_w

    def y(value):
        return top + plot_h - (value / max_y) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="20" fill="#111827">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#d1d5db"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#d1d5db"/>',
    ]
    for c in [1, 2, 4, 8, 16, 32, 64]:
        xx = x(c)
        lines.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{xx:.1f}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151">{c}</text>')
    for tick in [0, max_y / 4, max_y / 2, max_y * 3 / 4, max_y]:
        yy = y(tick)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#374151">{tick:.0f}</text>')
    lines.append(f'<text x="{left + plot_w / 2}" y="{height - 8}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">concurrency</text>')
    lines.append(f'<text x="16" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 16 {top + plot_h / 2})" font-family="sans-serif" font-size="13" fill="#111827">{escape(ylabel)}</text>')

    legend_y = 52
    for idx, (query, query_label) in enumerate(query_specs):
        dash = "" if idx == 0 else ' stroke-dasharray="6 4"'
        for sidx, (series, color, label) in enumerate(series_specs):
            row_points = points(rows, query, series, key)
            if not row_points:
                continue
            coord = " ".join(f"{x(cx):.1f},{y(cy):.1f}" for cx, cy in row_points)
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{dash} points="{coord}"/>')
            for cx, cy in row_points:
                lines.append(f'<circle cx="{x(cx):.1f}" cy="{y(cy):.1f}" r="3.5" fill="{color}"/>')
            legend_x = left + 260 * idx + 160 * sidx
            lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"{dash}/>')
            lines.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#111827">{escape(query_label)} {escape(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def phase_chart(path, title, rows, query):
    phases = [
        ("queue wait", "phase_scheduler_queue_wait_avg_us", "#2563eb"),
        ("engine execute", "phase_engine_execute_avg_us", "#059669"),
        ("retained wall", "phase_retained_wall_avg_us", "#7c3aed"),
        ("materialize", "phase_result_materialize_avg_us", "#d97706"),
        ("pgwire write", "phase_client_write_avg_us", "#dc2626"),
        ("cuda event", "phase_cuda_event_avg_us", "#0891b2"),
    ]
    phase_series = os.environ.get(
        "GPU_DB_P8_RETAINED_PHASE_SERIES",
        os.environ.get("GPU_DB_P8_RETAINED_HISTORY_SERIES", "steady_state_after_response_path"),
    )
    current = [row for row in rows if row["series"] == phase_series and row["query"] == query]
    current.sort(key=lambda row: numeric(row, "concurrency"))
    width, height = 920, 420
    left, right, top, bottom = 85, 30, 45, 65
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_total = max(
        [sum(numeric(row, key) for _, key, _ in phases) for row in current] or [1.0]
    )
    max_total *= 1.12

    def x(index):
        return left + index * (plot_w / max(1, len(current)))

    def y(value):
        return top + plot_h - (value / max_total) * plot_h

    bar_w = min(58, plot_w / max(1, len(current)) * 0.65)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="20" fill="#111827">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#d1d5db"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#d1d5db"/>',
    ]
    for tick in [0, max_total / 4, max_total / 2, max_total * 3 / 4, max_total]:
        yy = y(tick)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#374151">{tick:.0f}</text>')
    for idx, row in enumerate(current):
        base = 0.0
        xx = x(idx) + (plot_w / max(1, len(current)) - bar_w) / 2
        for _, key, color in phases:
            value = numeric(row, key)
            y0 = y(base + value)
            y1 = y(base)
            lines.append(f'<rect x="{xx:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{max(0.5, y1 - y0):.1f}" fill="{color}"/>')
            base += value
        lines.append(f'<text x="{xx + bar_w / 2:.1f}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151">{row["concurrency"]}</text>')
    legend_x = left
    legend_y = 52
    for label, _, color in phases:
        lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 18}" y="{legend_y}" font-family="sans-serif" font-size="12" fill="#111827">{escape(label)}</text>')
        legend_x += 130
    lines.append(f'<text x="{left + plot_w / 2}" y="{height - 8}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">concurrency</text>')
    lines.append(f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})" font-family="sans-serif" font-size="13" fill="#111827">average phase time (us)</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: render_p8_retained_concurrency_history.py PREVIOUS_CSV CURRENT_JSONL OUT_DIR"
        )
    previous_csv = Path(sys.argv[1])
    current_jsonl = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_previous(previous_csv) + load_current(current_jsonl)
    write_csv(out_dir / "steady-state-response-optimization-metrics.csv", rows)
    chart(out_dir / "throughput-history.svg", "Retained throughput history", rows, "throughput_qps", "qps")
    chart(out_dir / "p50-latency-history.svg", "Retained p50 latency history", rows, "p50_us", "p50 latency (us)")
    phase_chart(
        out_dir / "count-phase-breakdown.svg",
        "COUNT phase averages",
        rows,
        "order_line_count_all",
    )
    phase_chart(
        out_dir / "lookup-phase-breakdown.svg",
        "Lookup phase averages",
        rows,
        "order_line_lookup_ol_o_id_multi_column",
    )


if __name__ == "__main__":
    main()
