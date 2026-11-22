"""Verify private frozen runs and export only replayable, synthetic evidence.

This offline command makes no API requests. It regrades every program, checks
the frozen sources and reconciles HTTP usage against the paid ledger before
creating a public projection. Model messages, endpoint and private paths stay local.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.agent.graph_checkpoint import read_budget
from dataverify.evaluation.frozen_cases import grade_frozen_run
from dataverify.evaluation.frozen_experiment import (
    aggregate,
    baseline_gates,
    experiment_cases,
    source_hashes,
)
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_and_export(directory: Path) -> dict:
    """Fail closed on altered source, omitted trials, scoring drift or lost usage."""
    manifest, report = load(directory / "manifest.json"), load(directory / "summary.json")
    if digest_json(manifest) != report["manifest_digest"]:
        raise ValueError("MANIFEST_DIGEST_MISMATCH")
    if source_hashes(ROOT) != manifest["files"]:
        raise ValueError("FROZEN_SOURCE_CHANGED")
    cases = experiment_cases()
    if [case.case_id for case in cases] != [row["case_id"] for row in report["rows"]]:
        raise ValueError("INCOMPLETE_OR_REORDERED_COHORT")
    projections = []
    for case, row in zip(cases, report["rows"]):
        record = load(directory / (case.case_id + ".json"))
        if grade_frozen_run(case, record) != row["score"]:
            raise ValueError("SCORE_REPLAY_MISMATCH")
        if baseline_gates(case) != row["baseline"]:
            raise ValueError("BASELINE_REPLAY_MISMATCH")
        if record["http_events"] != row["http_events"]:
            raise ValueError("HTTP_TRACE_MISMATCH")
        if record["binding"]["source_digest"] != digest_json(case.source.to_dict()):
            raise ValueError("SOURCE_BINDING_MISMATCH")
        if record["bound_source"] != case.source.to_dict():
            raise ValueError("SOURCE_CONTENT_MISMATCH")
        publication = record.get("publication") or {}
        projections.append(
            {
                **row,
                "programs": {
                    "candidate": (record.get("candidate") or {}).get("sql"),
                    "published_sql": publication.get("sql"),
                    "published_rows": publication.get("rows"),
                    "reference": (record.get("plan") or {}).get("reference"),
                    "audit": (record.get("audit_plan") or {}).get("reference"),
                },
                "trace": [
                    {
                        key: event[key]
                        for key in (
                            "node",
                            "error",
                            "target",
                            "generated_counterexample",
                            "candidate_visible",
                            "reference_visible",
                        )
                        if key in event
                    }
                    for event in record["trace"]
                ],
            }
        )
    events = [event for row in projections for event in row["http_events"]]
    if [event["attempt"] for event in events] != list(range(1, len(events) + 1)):
        raise ValueError("HTTP_ATTEMPTS_NOT_CONTIGUOUS")
    budget = read_budget(directory / report["latest_journal"])
    if not budget["reported_cost_complete"] or budget["http_attempts"] != len(events):
        raise ValueError("INCOMPLETE_USAGE_LEDGER")
    for event, entry in zip(events, budget["reservation_ledger"]):
        if entry["status"] != "usage_complete" or any(
            event["usage"][key] != entry[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise ValueError("EVENT_LEDGER_MISMATCH")
    usage = summarize_usage(events)
    if usage != report["usage"]:
        raise ValueError("USAGE_TOTAL_MISMATCH")
    computed = aggregate(report["rows"])
    if any(report[key] != value for key, value in computed.items()):
        raise ValueError("AGGREGATE_MISMATCH")
    return {
        "format": "dataverify-frozen-evidence-v1",
        "scope": report["scope"],
        "date": "2026-11-07",
        "limitations": [
            "Four public development intents, not an unseen benchmark or production traffic",
            "Controlled fault injections and natural generation are scored separately",
            "31 datasets per trajectory do not create 31 independent tasks",
            "Execution gates are not a no-auditor Agent ablation",
            "All roles share one model; contexts are separate, errors are not statistically independent",
            "One trial per case; failures retained without whole-task reruns",
        ],
        "manifest_digest": report["manifest_digest"],
        "base_git_commit": manifest["base_git_commit"],
        "files": manifest["files"],
        "model": report["model"],
        "environment": manifest["environment"],
        "parameters": manifest["parameters"],
        **computed,
        "rows": projections,
        "incremental_cost": {
            "estimated_cny": budget["estimated_reported_cny"],
            "reserved_cny": budget["reserved_upper_cny"],
            "pricing": budget["pricing"],
            "is_cloud_bill": False,
        },
        "verification": {
            "scoring_replayed": True,
            "baseline_replayed": True,
            "source_hashes_match": True,
            "all_http_usage_reconciled": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify_and_export(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"attempted": result["attempted"], "usage": result["usage"]}))


if __name__ == "__main__":
    main()
