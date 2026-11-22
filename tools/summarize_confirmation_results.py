"""Export audited confirmation aggregates without raw contexts or ledger paths.

Reads completed reports only; no credentials, live ledgers or model requests.
Timing labels deliberately differ from the earlier shared-reference experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relabel_timing(value):
    if isinstance(value, list):
        return [relabel_timing(item) for item in value]
    if not isinstance(value, dict):
        return value
    names = {
        "elapsed_seconds": "reference_plus_verification_seconds",
        "elapsed_scope": "timing_scope",
        "post_initial_usage": "reference_plus_verification_usage",
    }
    return {
        names.get(
            key, key
        ): "Reference generation plus verification; shared initial SQL generation excluded."
        if key == "elapsed_scope"
        else relabel_timing(item)
        for key, item in value.items()
    }


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def export(args):
    raw, audit, budget, protocol = (
        read(path) for path in (args.summary, args.audit, args.budget_audit, args.protocol)
    )
    if (
        raw["report_status"] != "complete"
        or raw["completed_cells"] != 80
        or raw["missing_cells"]
        or raw["incomplete_records"]
        or audit.get("audit_passed") is not True
        or audit["completed_cells"] != 80
        or audit["unique_tasks"] != 40
        or not all(
            audit[key] is True
            for key in (
                "shared_sql_verified",
                "full_database_rescore_verified",
                "frozen_runtime_archive_driver_manifest_verified",
            )
        )
        or audit["protocol_digest"] != raw["protocol_digest"]
        or budget.get("audit_passed") is not True
    ):
        raise ValueError("COMPLETE_INDEPENDENTLY_AUDITED_CONFIRMATION_REQUIRED")
    rows = []
    for method in protocol["methods"]:
        selected = [row for row in raw["rows"] if row["method"] == method]
        if len(selected) != 40 or len({row["case_id"] for row in selected}) != 40:
            raise ValueError("CONFIRMATION_REQUIRES_40_DISTINCT_PAIRED_TASKS")
        correct = sum(row["score"]["correct_delivery"] for row in selected)
        wrong = sum(row["score"]["wrong_release"] for row in selected)
        accept = raw["initial_reference_acceptance"][method]
        rows.append(
            {
                "method": method,
                "tasks": 40,
                "correct_delivery": correct,
                "wrong_release": wrong,
                "non_delivery": 40 - correct - wrong,
                "initial_reference_accepted": accept["accepted"],
                "reference_validation_failures": accept["validation_failures"],
                "reference_plus_verification_median_seconds": quantile(
                    [row["elapsed_seconds"] for row in selected], 0.5
                ),
                "reference_plus_verification_p95_seconds": quantile(
                    [row["elapsed_seconds"] for row in selected], 0.95
                ),
            }
        )
    summary = {
        "format": "dataverify-structured-confirmation-public-v1",
        "status": "complete_independently_audited",
        "date": "2026-11-07",
        "protocol_digest": raw["protocol_digest"],
        "source_report_sha256": sha(args.summary),
        "independent_audit_sha256": sha(args.audit),
        "final_budget_audit_sha256": sha(args.budget_audit),
        "manifest_digest": protocol["manifest_digest"],
        "tasks": 40,
        "cells": 80,
        "databases": protocol["databases"],
        "outcomes": rows,
        "analysis": relabel_timing(raw["analysis"]),
        "reference_acceptance_note": raw["reference_acceptance_note"],
        "timing_scope": raw["timing_scope"],
        "final_campaign_cost": {
            key: budget[key]
            for key in (
                "accounting_scope",
                "cumulative_allowance_cny",
                "conservative_reserved_cny",
                "known_usage_estimated_cny",
                "unknown_attempts",
                "unknown_reserved_upper_cny",
                "known_estimate_plus_unknown_reserve_cny",
                "reported_total_tokens",
                "all_http_attempts",
                "confirmation_unique_events_matched_to_ledgers",
                "limitation",
            )
        },
        "interpretation": [
            "No evidence from this comparison supports enabling strict schema by default.",
            "Same new questions in four existing schemas; not an unseen-database evaluation.",
            "Shared SQL, independently generated references per mode; all other prompts and full-policy limits held fixed.",
            "Paired analysis reports json_object minus strict_schema, not the reverse.",
            "Zero-event empirical bootstrap intervals do not establish zero population risk.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--budget-audit", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/experiments/structured-confirmation")
    )
    result = export(parser.parse_args())
    print(
        json.dumps({"status": result["status"], "tasks": result["tasks"], "cells": result["cells"]})
    )


if __name__ == "__main__":
    main()
