"""Audit saved paired records with handwritten arithmetic and token receipts; offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.evaluation.dispute_recovery import study_cases
from dataverify.evaluation.expanded_cases import expected, scoring_data
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.evaluation.query_ablation import METHODS, full_source
from dataverify.runtime.reporting import digest_json, summarize_usage
from dataverify.verification.discovery_runtime import execute_readonly, rows_equal


def audit(directory):
    def load(relative):
        return json.loads((directory / relative).read_text(encoding="utf-8"))

    protocol = load("protocol.json")
    protocol_id = digest_json(protocol)
    if any(
        file_digest(directory / "source" / p) != value
        for p, value in protocol["source_hashes"].items()
    ):
        raise ValueError("Frozen source hash mismatch")
    summary = load("summary.json")
    if summary["protocol"] != protocol_id:
        raise ValueError("Unmatched summary")
    rows = {(r["case"], r["method"]): r for r in summary["rows"]}
    cases = study_cases()
    if len(summary["rows"]) != len(rows) or set(rows) != {
        (c.case_id, m) for c in cases for m in METHODS
    }:
        raise ValueError("Incomplete or duplicate cases")
    events = []
    delivered = 0
    for case in cases:
        fixtures = scoring_data(case.base).values()
        full = full_source(case.base.source)
        for method in METHODS:
            entry = load(f"records/{case.case_id}--{method}.json")
            binding = entry["binding"]
            if (
                binding
                != {
                    "protocol": protocol_id,
                    "case": case.case_id,
                    "method": method,
                    "model": protocol["model"],
                }
                or not entry["completed"]
            ):
                raise ValueError("Record identity mismatch")
            result = entry["record"]
            row = rows[case.case_id, method]
            if method == "query_off" and result.get("query_observations"):
                raise ValueError("Query-off arm received active observations")
            if row["status"] != result["status"]:
                raise ValueError("Status mismatch")
            valid = False
            if result["status"] == "reviewed":
                delivered += 1
                pub = result["publication"]
                sql = pub["sql"]
                valid = all(
                    rows_equal(execute_readonly(sql, d), expected(case.base, d)) for d in fixtures
                )
                valid = (
                    valid
                    and rows_equal(pub["rows"], expected(case.base, full))
                    and rows_equal(execute_readonly(sql, full), pub["rows"])
                )
            if (
                valid != row["correct"]
                or (result["status"] == "reviewed" and not valid) != row["wrong"]
            ):
                raise ValueError(f"Independent delivery grade mismatch for {case.case_id}/{method}")
            events.extend(entry["http_events"])
    ids = [item["event_id"] for item in events]
    if len(ids) != len(set(ids)):
        raise ValueError("Reused payment receipt ID")
    return {
        "protocol": protocol_id,
        "records": len(rows),
        "publications_checked": delivered,
        "usage": summarize_usage(events),
    }


def audit_correction(parent, directory):
    """Check revised API receipts, source bindings, exposed actions and publication rows."""
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    parent_id = digest_json(json.loads((parent / "protocol.json").read_text(encoding="utf-8")))
    if parent_id != protocol["parent"] or any(
        file_digest(parent / "records" / name) != value
        for name, value in protocol["control_records"].items()
    ):
        raise ValueError("Reused control record changed")
    if any(
        file_digest(directory / "source" / name) != value
        for name, value in protocol["source_hashes"].items()
    ):
        raise ValueError("Correction source changed")
    protocol_id = digest_json(protocol)
    events, paired = [], []
    original_ids = set()
    for name in protocol["control_records"]:
        data = json.loads((parent / "records" / name).read_text(encoding="utf-8"))
        original_ids.update(e["event_id"] for e in data["http_events"])
    for case in study_cases():
        name = f"{case.case_id}--query_off.json"
        new = json.loads((directory / "records" / name).read_text(encoding="utf-8"))
        if not new["completed"] or new["binding"] != {
            "case": case.case_id,
            "method": "query_off",
            "model": protocol["model"],
            "protocol": protocol_id,
        }:
            raise ValueError("Correction binding mismatch")
        if new["record"].get("query_observations") or new["record"].get("query_calls", 0) != 0:
            raise ValueError("Unexpected active query")
        for request in new["requests"]:
            if (
                '"action":"query"' in request[0]["content"]
                or json.loads(request[1]["content"])["remaining"]["queries"] != 0
            ):
                raise ValueError("Disabled action advertised")
        source = full_source(case.base.source)
        valid = False
        if new["record"]["status"] == "reviewed":
            pub = new["record"]["publication"]
            valid = all(
                rows_equal(execute_readonly(pub["sql"], d), expected(case.base, d))
                for d in scoring_data(case.base).values()
            )
            valid = (
                valid
                and rows_equal(pub["rows"], expected(case.base, source))
                and rows_equal(pub["rows"], execute_readonly(pub["sql"], source))
            )
        if valid != new["score"]["correct_delivery"]:
            raise ValueError("Correction grades disagree with executions")
        on = json.loads(
            (parent / "records" / name.replace("query_off", "query_on")).read_text(encoding="utf-8")
        )
        paired.append(
            {
                "case": case.case_id,
                "corrected_off_correct": valid,
                "previous_on_correct": on["score"]["correct_delivery"],
                "off_status": new["record"]["status"],
            }
        )
        events.extend(new["http_events"])
    ids = [e["event_id"] for e in events]
    if len(set(ids)) != len(ids) or original_ids.intersection(ids):
        raise ValueError("Duplicated original or correction receipts")
    return {"protocol": protocol_id, "paired": paired, "new_usage": summarize_usage(events)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--corrected", type=Path)
    args = parser.parse_args()
    report = audit(args.directory)
    if args.corrected:
        report["aligned_query_off"] = audit_correction(args.directory, args.corrected)
    print(json.dumps(report, indent=2))
