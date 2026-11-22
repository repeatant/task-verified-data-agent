"""Independently audit a completed dispute experiment without model/API calls.

Run only after run-status.json declares completion. This tool reads completed
records, never live budget ledgers. It does not import either production scoring
function and writes counts/digests only: no model contexts or raw SQL in reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.evaluation.expanded_cases import expanded_cases, expected, scoring_data
from dataverify.verification.discovery_runtime import execute_readonly, named_tables
from dataverify.verification.probe_runtime import ProbeData
from dataverify.verification.reference_expression import calculate

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
ROLES = ("candidate", "reference", "auditor")
METRICS = (
    "initial_sql_correct",
    "final_sql_correct",
    "repaired_initial_error",
    "correct_sql_damaged",
    "correct_task_rejected",
    "delivered",
    "correct_delivery",
    "wrong_release",
    "unchanged_delivery",
    "all_final_programs_correct",
    "wrong_reference_released",
    "wrong_auditor_released",
    "joint_correct_delivery",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def same_rows(left, right):
    """Independent unordered multiset comparison with the stated numeric tolerance."""
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    remaining = list(right)
    for row in left:
        if not isinstance(row, (list, tuple)):
            return False
        for index, other in enumerate(remaining):
            if not isinstance(other, (list, tuple)) or len(row) != len(other):
                continue
            matches = True
            for first, second in zip(row, other):
                if type(first) in (int, float) and type(second) in (int, float):
                    equal = (
                        first == second
                        if type(first) is type(second) is int
                        else math.isclose(first, second, rel_tol=0, abs_tol=1e-9)
                    )
                else:
                    equal = type(first) is type(second) and first == second
                if not equal:
                    matches = False
                    break
            if matches:
                remaining.pop(index)
                break
        else:
            return False
    return not remaining


def program_grade(case, program, *, sql):
    datasets = scoring_data(case)
    if not isinstance(program, str) or not program.strip():
        return {
            "present": False,
            "correct": None,
            "passed": 0,
            "total": len(datasets),
            "failures": [],
        }
    passed, failures = 0, []
    for name, source in datasets.items():
        try:
            actual = (
                execute_readonly(program, source)
                if sql
                else calculate(program, named_tables(source))
            )
            matched = same_rows(actual, expected(case, source))
            passed += matched
            if not matched:
                failures.append({"dataset": name, "kind": "output_mismatch"})
        except Exception as error:
            # Keep only the class; the report must not echo program/context text.
            failures.append({"dataset": name, "kind": type(error).__name__})
    return {
        "present": True,
        "correct": passed == len(datasets),
        "passed": passed,
        "total": len(datasets),
        "failures": failures,
    }


def usage_counts(events):
    counts = {field + "_known": 0 for field in TOKEN_FIELDS}
    unknown = 0
    for event in events:
        values = event.get("usage") or {}
        valid = {key: type(values.get(key)) is int and values[key] >= 0 for key in TOKEN_FIELDS}
        consistent = (
            all(valid.values())
            and values["prompt_tokens"] + values["completion_tokens"] == values["total_tokens"]
        )
        unknown += not consistent
        for key in TOKEN_FIELDS:
            if valid[key]:
                counts[key + "_known"] += values[key]
    return {
        **counts,
        "event_count": len(events),
        "unknown_usage_events": unknown,
        "usage_complete": unknown == 0,
    }


def field(record, outer, inner):
    return (record.get(outer) or {}).get(inner)


def reconstruct_case(spec, available):
    condition = "correlated_wrong" if spec["condition"] == "all_wrong" else spec["condition"]
    case = available[(spec["parent_id"], condition)]
    reference = case.initial_plan()
    auditor = case.initial_plan(auditor=True)
    if spec["condition"] == "all_wrong":
        auditor = {**auditor, "reference": case.bad_reference}
    return case, reference, auditor


def grade_cell(case, reference, auditor, method, record, visible):
    initial_programs = {
        "candidate": case.initial_sql,
        "reference": reference["reference"],
        "auditor": auditor["reference"],
    }
    final_programs = {
        "candidate": field(record, "candidate", "sql"),
        "reference": field(record, "plan", "reference"),
        "auditor": field(record, "audit_plan", "reference"),
    }
    initial = {
        role: program_grade(case, code, sql=role == "candidate")
        for role, code in initial_programs.items()
    }
    final = {
        role: program_grade(case, code, sql=role == "candidate")
        for role, code in final_programs.items()
    }
    publication = program_grade(case, field(record, "publication", "sql"), sql=True)
    delivered = record.get("status") == "reviewed" and publication["present"]
    source = case.source if method == "legacy_full" else visible
    rows_consistent = False
    if publication["present"]:
        try:
            rows_consistent = same_rows(
                execute_readonly(field(record, "publication", "sql"), source),
                field(record, "publication", "rows"),
            )
        except Exception:
            pass
    correct_delivery = delivered and publication["correct"] is True and rows_consistent
    initial_correct = initial["candidate"]["correct"] is True
    final_correct = final["candidate"]["correct"] is True
    sql = final_programs["candidate"]
    changed = isinstance(sql, str) and sql.strip() != case.initial_sql.strip()
    roles = {}
    for role in ROLES:
        supported = method != "self_review" or role == "candidate"
        roles[role] = {
            "initial_correct": initial[role]["correct"],
            "final_correct": final[role]["correct"] is True if supported else None,
            "repaired": initial[role]["correct"] is False and final[role]["correct"] is True
            if supported
            else None,
            "damaged": initial[role]["correct"] is True and final[role]["correct"] is not True
            if supported
            else None,
            "initial_grade": initial[role],
            "final_grade": final[role] if supported else None,
        }
    metrics = {
        "initial_sql_correct": initial_correct,
        "final_sql_correct": final_correct,
        "repaired_initial_error": initial["candidate"]["correct"] is False
        and final_correct
        and changed,
        "correct_sql_damaged": initial_correct and final["candidate"]["correct"] is False,
        "correct_task_rejected": initial_correct and not delivered,
        "delivered": delivered,
        "correct_delivery": correct_delivery,
        "wrong_release": delivered and not correct_delivery,
        "unchanged_delivery": delivered and not changed,
        "all_final_programs_correct": all(final[r]["correct"] is True for r in ROLES)
        if method != "self_review"
        else None,
        "joint_correct_delivery": correct_delivery
        and final["reference"]["correct"] is True
        and final["auditor"]["correct"] is True
        if method != "self_review"
        else None,
        "wrong_reference_released": delivered and final["reference"]["correct"] is not True
        if method != "self_review"
        else None,
        "wrong_auditor_released": delivered and final["auditor"]["correct"] is not True
        if method != "self_review"
        else None,
        "publication_rows_consistent": rows_consistent if delivered else None,
        "sql_text_changed": changed,
    }
    return {
        "metrics": metrics,
        "roles": roles,
        "initial": initial["candidate"],
        "final": final,
        "publication": publication,
        "publication_source_scope": "original_source"
        if method == "legacy_full"
        else "shared_visible_diagnostic_fixture",
    }


def audit(output):
    status = read_json(output / "run-status.json")
    if status.get("complete") is not True or status.get("stopped"):
        raise ValueError("AUDIT_REQUIRES_COMPLETED_STOPPED_EXECUTION")
    protocol = read_json(output / "protocol.json")
    summary = read_json(output / "summary.json")
    issues = []

    def check(condition, code, location):
        if not condition:
            issues.append({"code": code, "location": location})

    def equal(actual, wanted, code, location):
        check(actual == wanted, code, location)

    # Verify the code used for this audit still matches the frozen implementation.
    # This new standalone audit script is intentionally absent from that manifest.
    with zipfile.ZipFile(output / "frozen-source.zip") as archived:
        for relative, wanted in protocol["source_hashes"].items():
            equal(file_digest(ROOT / relative), wanted, "SOURCE_HASH_MISMATCH", relative)
            equal(
                hashlib.sha256(archived.read(relative)).hexdigest(),
                wanted,
                "ARCHIVE_HASH_MISMATCH",
                relative,
            )
    equal(summary.get("protocol_digest"), digest(protocol), "SUMMARY_PROTOCOL_DIGEST", "summary")
    methods, specs = protocol["methods"], protocol["cases"]
    expected_grid = {(spec["case_id"], method) for spec in specs for method in methods}
    files = sorted((output / "records").glob("*.json"))
    equal(len(files), len(expected_grid), "CELL_COUNT", "records")
    available = {(case.parent_id, case.condition): case for case in expanded_cases()}
    visible_data = protocol["visible_diagnostic_fixture"]
    visible = ProbeData(visible_data["orders"], visible_data["lines"])
    by_id = {spec["case_id"]: spec for spec in specs}
    independent_rows, events, seen_grid = [], [], set()
    for path in files:
        document = read_json(path)
        if document.get("completed") is not True:
            raise ValueError("AUDIT_FOUND_INCOMPLETE_RECORD")
        row, record = document["row"], document["record"]
        case_id, method = row["case_id"], row["method"]
        cell = (case_id, method)
        location = case_id + "--" + method
        check(cell in expected_grid, "UNEXPECTED_CELL", location)
        check(cell not in seen_grid, "DUPLICATE_CELL", location)
        seen_grid.add(cell)
        if cell not in expected_grid:
            continue
        spec = by_id[case_id]
        case, reference, auditor = reconstruct_case(spec, available)
        binding = {
            "task_digest": digest(case.task.public_input()),
            "source_digest": digest(case.source.to_dict()),
            "visible_diagnostic_digest": digest(visible.to_dict()),
            "initial_sql_digest": digest(case.initial_sql),
            "reference_digest": digest(reference),
            "auditor_digest": digest(auditor),
            "scoring_suite_digest": digest(
                {name: source.to_dict() for name, source in scoring_data(case).items()}
            ),
        }
        for key, value in binding.items():
            equal(spec.get(key), value, "PROTOCOL_CASE_BINDING", location + ":" + key)
            equal(document["binding"].get(key), value, "RECORD_CASE_BINDING", location + ":" + key)
        equal(
            document["binding"].get("protocol_digest"),
            digest(protocol),
            "RECORD_PROTOCOL_DIGEST",
            location,
        )
        equal(document["binding"].get("model"), protocol["model"], "RECORD_MODEL", location)
        equal(document["binding"].get("method"), method, "RECORD_METHOD", location)
        equal(row["task_id"], case.task.task_id, "ROW_TASK", location)
        equal(row["condition"], spec["condition"], "ROW_CONDITION", location)
        equal(row["status"], record["status"], "ROW_STATUS", location)
        computed = grade_cell(case, reference, auditor, method, record, visible)
        saved_score = row["score"]
        for key, value in computed["metrics"].items():
            equal(saved_score.get(key), value, "CELL_METRIC", location + ":" + key)
        equal(
            saved_score["publication_source_scope"],
            computed["publication_source_scope"],
            "PUBLICATION_SCOPE",
            location,
        )
        for role in ROLES:
            for key in ("initial_correct", "final_correct", "repaired", "damaged"):
                equal(
                    saved_score["roles"][role][key],
                    computed["roles"][role][key],
                    "ROLE_METRIC",
                    location + ":" + role + ":" + key,
                )
            for key in ("present", "correct", "passed", "total"):
                equal(
                    saved_score[role][key],
                    computed["final"][role][key],
                    "PROGRAM_GRADE",
                    location + ":" + role + ":" + key,
                )
        for key in ("present", "correct", "passed", "total"):
            equal(
                saved_score["initial"][key],
                computed["initial"][key],
                "INITIAL_GRADE",
                location + ":" + key,
            )
            equal(
                saved_score["publication"][key],
                computed["publication"][key],
                "PUBLICATION_GRADE",
                location + ":" + key,
            )
        cell_events = document.get("http_events", [])
        equal(cell_events, row.get("http_events"), "HTTP_EVENT_COPY", location)
        equal(usage_counts(cell_events), row["usage"], "CELL_USAGE", location)
        cap = (
            protocol["limits"]["baseline_model_calls"]
            if method == "self_review"
            else protocol["limits"]["resolver_model_calls"]
        )
        check(len(cell_events) <= cap, "HTTP_CALL_CAP", location)
        for event in cell_events:
            check(
                isinstance(event.get("event_id"), str) and bool(event["event_id"]),
                "HTTP_EVENT_ID",
                location,
            )
            check(
                type(event.get("attempt")) is int and event["attempt"] > 0, "HTTP_ATTEMPT", location
            )
            values = event.get("usage") or {}
            complete = all(type(values.get(k)) is int and values[k] >= 0 for k in TOKEN_FIELDS)
            complete = (
                complete
                and values["prompt_tokens"] + values["completion_tokens"] == values["total_tokens"]
            )
            equal(event.get("usage_complete"), complete, "HTTP_USAGE_FLAG", location)
            if event.get("status") == "completed":
                check(complete and event.get("http_status") == 200, "HTTP_COMPLETION", location)
        events.extend(cell_events)
        independent_rows.append(
            {
                "case_id": case_id,
                "method": method,
                "condition": spec["condition"],
                "status": record["status"],
                "task_id": case.task.task_id,
                **computed,
                "usage": usage_counts(cell_events),
                "record_sha256": file_digest(path),
            }
        )
    equal(seen_grid, expected_grid, "GRID_INCOMPLETE", "records")
    event_ids = [event.get("event_id") for event in events]
    duplicates = [key for key, count in Counter(event_ids).items() if count > 1]
    check(not duplicates, "HTTP_EVENT_DUPLICATE", "records")
    equal(usage_counts(events), summary["usage_all_new_requests"], "TOTAL_USAGE", "summary")
    totals = {}
    for method in methods:
        selected = [row for row in independent_rows if row["method"] == method]
        aggregate = {
            "trajectories": len(selected),
            "distinct_requirements": len({row["task_id"] for row in selected}),
            "statuses": dict(Counter(row["status"] for row in selected)),
        }
        for metric in METRICS:

            def eligible(row):
                if metric in {"correct_sql_damaged", "correct_task_rejected"}:
                    return row["metrics"]["initial_sql_correct"]
                if metric == "repaired_initial_error":
                    return row["initial"]["correct"] is False
                return row["metrics"][metric] is not None

            aggregate[metric] = {
                "numerator": sum(row["metrics"][metric] is True for row in selected),
                "denominator": sum(eligible(row) for row in selected),
            }
        aggregate["roles"] = {}
        for role in ROLES:
            aggregate["roles"][role] = {}
            for key in ("final_correct", "repaired", "damaged"):

                def eligible_role(row):
                    item = row["roles"][role]
                    if item[key] is None:
                        return False
                    if key == "repaired":
                        return item["initial_correct"] is False
                    if key == "damaged":
                        return item["initial_correct"] is True
                    return True

                aggregate["roles"][role][key] = {
                    "numerator": sum(row["roles"][role][key] is True for row in selected),
                    "denominator": sum(eligible_role(row) for row in selected),
                }
        method_events = [
            event
            for document_path in files
            for document in [read_json(document_path)]
            if document["row"]["method"] == method
            for event in document["http_events"]
        ]
        aggregate["usage"] = usage_counts(method_events)
        equal(aggregate, summary["methods"][method], "METHOD_AGGREGATES", method)
        totals[method] = aggregate
    return {
        "audit_kind": "independent_execution_recount",
        "passed": not issues,
        "cells": len(independent_rows),
        "expected_cells": len(expected_grid),
        "protocol_digest": digest(protocol),
        "source_hashes_checked": len(protocol["source_hashes"]),
        "grading_datasets_per_program": 25,
        "issues": issues,
        "http_event_count": len(events),
        "unique_http_event_count": len(set(event_ids)),
        "duplicate_http_event_ids": duplicates,
        "usage": usage_counts(events),
        "methods": totals,
        "cells_recomputed": independent_rows,
        "limitations": [
            "No provider-side billing reconciliation; no budget ledgers read.",
            "Uses the frozen handwritten expected() and independent SQL/DSL execution, not score_result or score_controlled.",
            "Program correctness means agreement on the fixed 25 grading datasets; no arbitrary-input proof.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed experiment directory")
    parser.add_argument(
        "--report", type=Path, help="Default: independent-audit.json inside experiment output"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    report = args.report.resolve() if args.report else output / "independent-audit.json"
    try:
        result = audit(output)
    except Exception as error:
        # Do not print exception content that could contain model text or paths.
        print(json.dumps({"audit_completed": False, "error_type": type(error).__name__}))
        raise SystemExit(2) from None
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "cells": result["cells"],
                "issues": len(result["issues"]),
                "http_events": result["http_event_count"],
                "known_tokens": result["usage"]["total_tokens_known"],
                "report": str(report),
            },
            ensure_ascii=False,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
