#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RUNNING_RE = re.compile(r"^\s*Running unittests .+?/([^/\s]+)-[0-9a-f]+\)\s*$")
TEST_RE = re.compile(r"^test\s+(.+?)\s+\.\.\.\s+(ok|FAILED|ignored)\s*$")


def classify(test_id: str) -> list[str]:
    buckets: list[str] = []
    if "psql_golden::" in test_id:
        if re.search(r"bootstrap|startup|auth|connect|simple_query|session_reset|prepare", test_id):
            buckets.append("protocol.client_flows")
        if re.search(r"transaction|begin|commit|rollback", test_id):
            buckets.append("sql.transaction_flows")
        if re.search(r"relational|create_table|insert|select", test_id):
            buckets.append("sql.relational_foundation")
    if "gpu_db_protocol::" in test_id:
        if re.search(r"startup|frontend|ssl|cancel|session_lifecycle", test_id):
            buckets.append("protocol.client_flows")
        if re.search(r"parses_|rejects_", test_id):
            buckets.append("sql.parser_features")
        if re.search(r"relational|create_table|insert|select", test_id):
            buckets.append("sql.relational_foundation")
    if re.search(r"relational|create_table|insert|select", test_id):
        buckets.append("sql.relational_foundation")
    if re.search(r"relational_catalog|catalog_schema|type_metadata|column_id|relation_oid", test_id):
        buckets.append("sql.catalog_schema_types")
    if re.search(r"transaction|commit|rollback|begin", test_id):
        buckets.append("sql.transaction_flows")
    if re.search(r"wal|durable|visibility|checkpoint|replay", test_id):
        buckets.append("durability.invariants")
    if re.search(r"replication|raft|snapshot|leader|follower", test_id):
        buckets.append("replication.role_and_log")
    if re.search(r"gpu|fallback|batch", test_id):
        buckets.append("execution.gpu_routing_and_batching")
    if not buckets:
        buckets.append("uncategorized")
    return buckets


def parse_log(log_path: Path) -> tuple[list[dict], Counter]:
    current_crate = "unknown_crate"
    tests: list[dict] = []
    status_counts: Counter = Counter()

    for line in log_path.read_text(encoding="utf-8").splitlines():
        run_match = RUNNING_RE.match(line)
        if run_match:
            current_crate = run_match.group(1)
            continue

        test_match = TEST_RE.match(line)
        if not test_match:
            continue

        test_name, status = test_match.groups()
        status_counts[status] += 1
        test_id = f"{current_crate}::{test_name}"
        tests.append(
            {
                "id": test_id,
                "status": status,
                "buckets": classify(test_id),
            }
        )

    return tests, status_counts


def parse_psql_report(report_path: Path | None) -> tuple[list[dict], Counter]:
    if report_path is None or not report_path.exists():
        return [], Counter()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    tests: list[dict] = []
    status_counts: Counter = Counter()

    for scenario in report.get("scenarios", []):
        raw_status = scenario.get("status", "failed")
        status = "ok" if raw_status == "passed" else "FAILED"
        test_id = scenario.get("id", f"psql_golden::{scenario.get('name', 'unknown')}")
        status_counts[status] += 1
        tests.append(
            {
                "id": test_id,
                "status": status,
                "buckets": classify(test_id),
            }
        )

    return tests, status_counts


def summarize(tests: list[dict], status_counts: Counter) -> dict:
    by_bucket = defaultdict(lambda: Counter({"total": 0, "ok": 0, "FAILED": 0, "ignored": 0}))
    failing_by_bucket: Counter = Counter()

    for test in tests:
        for bucket in test["buckets"]:
            by_bucket[bucket]["total"] += 1
            by_bucket[bucket][test["status"]] += 1
            if test["status"] == "FAILED":
                failing_by_bucket[bucket] += 1

    bucket_summary = {
        bucket: {
            "total": counters["total"],
            "passed": counters["ok"],
            "failed": counters["FAILED"],
            "ignored": counters["ignored"],
        }
        for bucket, counters in sorted(by_bucket.items())
    }

    return {
        "totals": {
            "total": sum(status_counts.values()),
            "passed": status_counts["ok"],
            "failed": status_counts["FAILED"],
            "ignored": status_counts["ignored"],
        },
        "buckets": bucket_summary,
        "top_failing_categories": [
            {"bucket": bucket, "failed": count}
            for bucket, count in failing_by_bucket.most_common(5)
        ],
    }


def load_baseline(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_trend(summary: dict, baseline: dict | None) -> dict:
    if baseline is None:
        return {
            "baseline_available": False,
            "note": "No baseline configured yet. Add docs/compatibility/scorecard.baseline.json to enable diffs.",
        }

    current_failed = summary["totals"]["failed"]
    baseline_failed = baseline.get("totals", {}).get("failed", 0)

    baseline_buckets = baseline.get("buckets", {})
    current_buckets = summary.get("buckets", {})
    bucket_failed_delta = {}
    for bucket in sorted(set(baseline_buckets) | set(current_buckets)):
        current_bucket_failed = current_buckets.get(bucket, {}).get("failed", 0)
        baseline_bucket_failed = baseline_buckets.get(bucket, {}).get("failed", 0)
        bucket_failed_delta[bucket] = current_bucket_failed - baseline_bucket_failed

    return {
        "baseline_available": True,
        "baseline_failed": baseline_failed,
        "current_failed": current_failed,
        "failed_delta": current_failed - baseline_failed,
        "bucket_failed_delta": bucket_failed_delta,
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Compatibility Scorecard",
        "",
        "## Totals",
        f"- total: {report['totals']['total']}",
        f"- passed: {report['totals']['passed']}",
        f"- failed: {report['totals']['failed']}",
        f"- ignored: {report['totals']['ignored']}",
        "",
        "## Bucket Summary",
    ]

    for bucket, values in report["buckets"].items():
        lines.append(
            f"- {bucket}: total={values['total']} passed={values['passed']} failed={values['failed']} ignored={values['ignored']}"
        )

    lines += ["", "## Top failing categories"]
    if report["top_failing_categories"]:
        for item in report["top_failing_categories"]:
            lines.append(f"- {item['bucket']}: {item['failed']}")
    else:
        lines.append("- none")

    lines += ["", "## Trend hook", f"- {json.dumps(report['trend_hook'])}"]
    if report["trend_hook"].get("baseline_available") and report["trend_hook"].get(
        "bucket_failed_delta"
    ):
        lines += ["", "## Bucket failed deltas vs baseline"]
        for bucket, delta in report["trend_hook"]["bucket_failed_delta"].items():
            lines.append(f"- {bucket}: {delta:+d}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compatibility scorecard from cargo test logs")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", required=False, type=Path)
    parser.add_argument("--baseline", required=False, type=Path)
    parser.add_argument("--psql-report", required=False, type=Path)
    args = parser.parse_args()

    tests, status_counts = parse_log(args.input)
    psql_tests, psql_status_counts = parse_psql_report(args.psql_report)
    tests.extend(psql_tests)
    status_counts.update(psql_status_counts)
    summary = summarize(tests, status_counts)
    summary["trend_hook"] = build_trend(summary, load_baseline(args.baseline))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown, summary)


if __name__ == "__main__":
    main()
