#!/usr/bin/env python3
"""Aggregate N engine-backed-pgwire concurrency-smoke runs into median + CI + CV.

Minimal Phase-5 noise-control slice (prototype->production plan §5.7): the existing
harness emits ONE sample per (query, concurrency) cell per run, and cannot resolve
sub-10% deltas (±15-40% per-cell variance). This reads N independent runs'
`metrics.jsonl` files and reports, per cell and per metric:

- median, mean, sample stddev, and the coefficient of variation (CV = stddev/mean,
  the run-to-run noise measure);
- a 95% confidence interval for the mean (Student-t) and its half-width as a percent
  of the median — the **precision of this baseline estimate** (NOT a detectable
  effect); and
- the **A/B minimum detectable effect**: the smallest true difference a later
  before/after comparison (two independent N-sample groups) could detect at
  alpha=0.05 two-sided and power=0.80, i.e. (t_.975 + t_.80)*sd*sqrt(2/n) / median.
  This is the threshold a step-4 latency claim must actually clear; it is ~2x the
  CI half-width, so do NOT use the precision figure as the gate.

Records with `error_count>0` or `correctness_status != "pass"` are EXCLUDED from the
statistics (and counted separately) so a contaminated cell cannot pollute a median.

It does NOT change the measured path; it only post-processes run artifacts.

Usage:
  aggregate_concurrency_runs.py OUT_DIR metrics1.jsonl metrics2.jsonl ...
Writes OUT_DIR/{aggregate.json,aggregate.csv,summary.md} and prints a short summary.
"""

import json
import statistics
import sys
from pathlib import Path

KIND = "engine_backed_pgwire_concurrency_metric"

# Metrics to aggregate (jsonl field -> short label for tables).
METRICS = [
    ("p50_us", "p50_us"),
    ("p95_us", "p95_us"),
    ("p99_us", "p99_us"),
    ("throughput_qps", "qps"),
    ("wall_us", "wall_us"),
    ("phase_scheduler_queue_wait_avg_us", "queue_wait_us"),
    ("phase_engine_execute_avg_us", "engine_exec_us"),
    ("phase_cuda_event_avg_us", "cuda_us"),
]

# Student-t two-sided 95% critical values (t_.975) by degrees of freedom.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}

# Student-t one-sided 80th-percentile values (t_.80), for the power=0.80 term.
T80 = {
    1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896,
    8: 0.889, 9: 0.883, 10: 0.879, 11: 0.876, 12: 0.873, 13: 0.870, 14: 0.868,
    15: 0.866, 16: 0.865, 17: 0.863, 18: 0.862, 19: 0.861, 20: 0.860,
    21: 0.859, 22: 0.858, 23: 0.858, 24: 0.857, 25: 0.856, 26: 0.856,
    27: 0.855, 28: 0.855, 29: 0.854, 30: 0.854, 40: 0.851, 60: 0.848, 120: 0.845,
}


def _t_lookup(table, df):
    # Conservative: round DOWN to the nearest tabulated df (smaller df -> larger t),
    # so the returned critical value is never smaller than the true one. (t decreases
    # as df grows, so picking a df above would understate the interval.)
    if df < 1:
        return float("nan")
    keys = [k for k in table if k <= df]
    return table[max(keys)] if keys else table[min(table)]


def t_crit(df):
    return _t_lookup(T95, df)


def t80_crit(df):
    return _t_lookup(T80, df)


def stats_for(values):
    n = len(values)
    out = {
        "n": n, "values": values,
        "median": None, "mean": None, "min": None, "max": None, "stddev": None,
        "cv_pct": None, "ci95_halfwidth": None, "ci95_low": None, "ci95_high": None,
        "ci95_halfwidth_pct_of_median": None, "ab_mde_abs": None, "ab_mde_pct": None,
    }
    if n == 0:
        return out
    out["median"] = statistics.median(values)
    out["mean"] = statistics.mean(values)
    out["min"] = min(values)
    out["max"] = max(values)
    if n < 2:
        # variance/CI/MDE are undefined for a single sample -> leave them null
        # (do NOT report 0%, which reads as "zero noise").
        return out
    sd = statistics.stdev(values)  # sample stddev, ddof=1
    mean, median = out["mean"], out["median"]
    out["stddev"] = sd
    out["cv_pct"] = (100.0 * sd / mean) if mean else None
    half = t_crit(n - 1) * sd / (n ** 0.5)
    out["ci95_halfwidth"] = half
    out["ci95_low"] = mean - half
    out["ci95_high"] = mean + half
    # Precision of THIS estimate (not a detectable effect): CI half-width / median.
    out["ci95_halfwidth_pct_of_median"] = (100.0 * half / median) if median else None
    # A/B minimum detectable effect: smallest true difference a before/after of two
    # independent N=n groups detects at alpha=0.05 (two-sided), power=0.80, equal
    # variance, df = 2n-2. This is the gate a later latency claim must clear.
    if median:
        df2 = 2 * n - 2
        mde = (t_crit(df2) + t80_crit(df2)) * sd * ((2.0 / n) ** 0.5)
        out["ab_mde_abs"] = mde
        out["ab_mde_pct"] = 100.0 * mde / median
    return out


def load_runs(paths):
    cells = {}   # (query, concurrency) -> field -> [values from GOOD records]
    health = {}  # (query, concurrency) -> counters
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line or '"kind"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != KIND:
                continue
            key = (rec["query"], int(rec["concurrency"]))
            h = health.setdefault(
                key, {"errors": 0, "correctness": set(), "records": 0, "excluded_bad": 0}
            )
            h["records"] += 1
            err = int(rec.get("error_count", 0) or 0)
            corr = rec.get("correctness_status", "unknown")
            h["errors"] += err
            h["correctness"].add(corr)
            if err > 0 or corr != "pass":
                # contaminated cell: exclude its values from the statistics
                h["excluded_bad"] += 1
                continue
            mbucket = cells.setdefault(key, {f: [] for f, _ in METRICS})
            for field, _ in METRICS:
                if field in rec and rec[field] is not None:
                    mbucket[field].append(float(rec[field]))
    return cells, health


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def mx(xs):
    return max(xs) if xs else float("nan")


def num(x, nd):
    return "" if x is None else "{:.{nd}f}".format(x, nd=nd)


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: aggregate_concurrency_runs.py OUT_DIR metrics1.jsonl ...")
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_paths = [p for p in sys.argv[2:] if Path(p).is_file()]
    n_files = len(metric_paths)
    if n_files == 0:
        sys.exit("no metrics.jsonl files found")

    cells, health = load_runs(metric_paths)

    agg = {}
    for key, mbucket in cells.items():
        query, conc = key
        agg.setdefault(query, {})[conc] = {
            field: stats_for(vals) for field, vals in mbucket.items()
        }

    # per-cell good-sample count (n on p50); used for the uneven-n integrity check.
    ns = [agg[q][c]["p50_us"]["n"] for q in agg for c in agg[q]]
    n_min, n_max = (min(ns), max(ns)) if ns else (0, 0)
    excluded_bad = sum(h["excluded_bad"] for h in health.values())
    uneven = n_min != n_max

    # ---- aggregate.json ----
    json_out = {
        "kind": "engine_backed_pgwire_concurrency_median_of_n",
        "metric_files": metric_paths,
        "files": n_files,
        "samples_per_cell_min": n_min,
        "samples_per_cell_max": n_max,
        "excluded_bad_records": excluded_bad,
        "metrics_aggregated": [f for f, _ in METRICS],
        "ab_mde_definition": "two-sample alpha=0.05 two-sided, power=0.80, df=2n-2",
        "cells": {
            f"{q}@c{c}": {
                "query": q,
                "concurrency": c,
                "samples": agg[q][c]["p50_us"]["n"],
                "records_total": health[(q, c)]["records"],
                "excluded_bad": health[(q, c)]["excluded_bad"],
                "errors_total": health[(q, c)]["errors"],
                "correctness": sorted(health[(q, c)]["correctness"]),
                "metrics": {
                    label: {k: v for k, v in agg[q][c][field].items() if k != "values"}
                    for field, label in METRICS
                },
                "raw": {label: agg[q][c][field]["values"] for field, label in METRICS},
            }
            for q in agg for c in agg[q]
        },
    }
    (out_dir / "aggregate.json").write_text(json.dumps(json_out, indent=2))

    # ---- aggregate.csv ----
    csv_lines = [
        "query,concurrency,metric,n,median,mean,stddev,cv_pct,min,max,"
        "ci95_low,ci95_high,ci95_halfwidth_pct_of_median,ab_mde_pct,"
        "excluded_bad,errors_total,correctness"
    ]
    for q in sorted(agg):
        for c in sorted(agg[q]):
            h = health[(q, c)]
            corr = "|".join(sorted(h["correctness"]))
            for field, label in METRICS:
                s = agg[q][c][field]
                if s["n"] == 0:
                    continue
                csv_lines.append(
                    ",".join([
                        q, str(c), label, str(s["n"]),
                        num(s["median"], 1), num(s["mean"], 1), num(s["stddev"], 2),
                        num(s["cv_pct"], 2), num(s["min"], 1), num(s["max"], 1),
                        num(s["ci95_low"], 1), num(s["ci95_high"], 1),
                        num(s["ci95_halfwidth_pct_of_median"], 2), num(s["ab_mde_pct"], 2),
                        str(h["excluded_bad"]), str(h["errors"]), corr,
                    ])
                )
    (out_dir / "aggregate.csv").write_text("\n".join(csv_lines) + "\n")

    # ---- across-cell noise/threshold summaries (only cells with n>=2) ----
    def collect(field, stat):
        return [agg[q][c][field][stat] for q in agg for c in agg[q]
                if agg[q][c][field]["n"] >= 2 and agg[q][c][field][stat] is not None]

    p50_cv = collect("p50_us", "cv_pct")
    qps_cv = collect("throughput_qps", "cv_pct")
    p50_prec = collect("p50_us", "ci95_halfwidth_pct_of_median")
    p50_mde = collect("p50_us", "ab_mde_pct")

    # ---- summary.md ----
    queries = sorted(agg)
    md = []
    md.append("# Engine-backed pgwire concurrency — median-of-{} (noise-controlled)".format(n_min if not uneven else "{}–{}".format(n_min, n_max)))
    md.append("")
    md.append("Files aggregated: {}. Per cell: median + 95% CI for the mean (Student-t) "
              "+ coefficient of variation (CV = stddev/mean). Records with errors or "
              "non-pass correctness are excluded from the statistics.".format(n_files))
    if uneven:
        md.append("")
        md.append("> ⚠ Uneven sample counts across cells (n={}–{}): some records were "
                  "excluded (bad) or missing. Treat cross-cell comparisons with care."
                  .format(n_min, n_max))
    if excluded_bad:
        md.append("")
        md.append("> ⚠ {} record(s) excluded for error_count>0 or correctness≠pass."
                  .format(excluded_bad))
    md.append("")
    md.append("## Harness noise and the step-4 gate")
    md.append("")
    md.append("- p50 CV across cells: median **{:.1f}%**, max **{:.1f}%**".format(med(p50_cv), mx(p50_cv)))
    md.append("- qps CV across cells: median **{:.1f}%**, max **{:.1f}%**".format(med(qps_cv), mx(qps_cv)))
    md.append("- p50 estimate precision (95% CI half-width / median): "
              "median **{:.1f}%**, max **{:.1f}%** — precision of *this* baseline, "
              "not a detectable effect.".format(med(p50_prec), mx(p50_prec)))
    md.append("- **p50 A/B minimum detectable effect** (two-sample, α=0.05, power=0.80, "
              "N each side): median **{:.1f}%**, max **{:.1f}%**.".format(med(p50_mde), mx(p50_mde)))
    md.append("")
    md.append("A step-4 before/after latency change must exceed the relevant cell's "
              "**A/B minimum detectable effect** (≈2× the CI half-width) to count as "
              "real rather than noise.")
    md.append("")
    md.append("## c64 cells (median [95% CI])")
    md.append("")
    # None-robust display helpers (a cell with n<2 has null CI/CV/MDE).
    def med_ci(s):
        if s["median"] is None:
            return "n/a"
        if s["ci95_low"] is None:
            return "{:.0f} [n/a]".format(s["median"])  # latency/qps CI low clamped to 0
        return "{:.0f} [{:.0f}–{:.0f}]".format(s["median"], max(0.0, s["ci95_low"]), s["ci95_high"])

    def one(x, nd=0):
        return "—" if x is None else "{:.{nd}f}".format(x, nd=nd)

    md.append("| query | p50 µs median [CI] | qps median [CI] | queue-wait µs med | CUDA µs med | p50 CV% | p50 A/B MDE% | err |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for q in queries:
        c = 64 if 64 in agg[q] else (sorted(agg[q])[-1] if agg[q] else None)
        if c is None:
            continue
        p50, qps = agg[q][c]["p50_us"], agg[q][c]["throughput_qps"]
        qw, cu = agg[q][c]["phase_scheduler_queue_wait_avg_us"], agg[q][c]["phase_cuda_event_avg_us"]
        md.append("| {q} | {p50} | {qps} | {qw} | {cu} | {cv} | {mde} | {err} |".format(
            q=q.replace("order_line_", ""),
            p50=med_ci(p50), qps=med_ci(qps),
            qw=one(qw["median"]), cu=one(cu["median"]),
            cv=one(p50["cv_pct"], 1), mde=one(p50["ab_mde_pct"], 1),
            err=health[(q, c)]["errors"],
        ))
    md.append("")
    md.append("Full per-cell, per-metric statistics: `aggregate.csv` / `aggregate.json`.")
    md.append("")
    (out_dir / "summary.md").write_text("\n".join(md))

    # ---- stdout ----
    print("files={} samples_per_cell={}".format(
        n_files, n_min if not uneven else "{}-{}".format(n_min, n_max)))
    print("cells={}".format(sum(len(agg[q]) for q in agg)))
    print("p50_cv_pct_median={:.1f} max={:.1f}".format(med(p50_cv), mx(p50_cv)))
    print("qps_cv_pct_median={:.1f} max={:.1f}".format(med(qps_cv), mx(qps_cv)))
    print("p50_ci_halfwidth_pct_median={:.1f} max={:.1f}".format(med(p50_prec), mx(p50_prec)))
    print("p50_ab_mde_pct_median={:.1f} max={:.1f}".format(med(p50_mde), mx(p50_mde)))
    total_err = sum(h["errors"] for h in health.values())
    bad = sorted({s for h in health.values() for s in h["correctness"]} - {"pass"})
    print("errors_total={} correctness_non_pass={} excluded_bad={} uneven_n={}".format(
        total_err, bad or "none", excluded_bad, uneven))
    print("artifacts={0}/aggregate.csv,{0}/aggregate.json,{0}/summary.md".format(out_dir))


if __name__ == "__main__":
    main()
