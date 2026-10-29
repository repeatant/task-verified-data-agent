"""Independently score saved LangGraph runs; never generate SQL or call a model.

Several process reports can describe one resumed thread. Score its latest
completed publication once, while counting every process's actual HTTP events.
Targets and added boundary data remain in the evaluator, outside the agent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.probe_cases import get_case
from dataverify.probe_grading import grade_probe_record
from dataverify.reporting import summarize_usage, write_json


def evaluate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Count resumed threads once and require a continuous accounting lineage."""
    if not reports or any(report.get("runtime") != "langgraph" for report in reports):
        raise ValueError("Provide saved LangGraph process reports in execution order")
    latest: dict[str, dict] = {}
    events = []
    resume_http_attempts = 0
    accounting_ok = True
    for report in reports:
        thread = report["thread_id"]
        previous = latest.get(thread)
        if previous and (
            previous["binding"] != report["binding"] or previous["case_id"] != report["case_id"]
        ):
            raise ValueError("A thread cannot combine changed sources or configurations")
        if previous:
            resume_http_attempts += len(report.get("http_events", []))
        latest[thread] = report
        events.extend(report.get("http_events", []))
        accounting_ok &= report.get("audit_ledger_complete") is True
    for previous, current in zip(reports, reports[1:]):
        accounting_ok &= previous.get("budget_end") == current.get("budget_start")
    budget_start, budget_end = reports[0].get("budget_start"), reports[-1].get("budget_end")
    if budget_start is not None and budget_end is not None:
        expected = list(range(budget_start["http_attempts"] + 1, budget_end["http_attempts"] + 1))
        accounting_ok &= [event.get("attempt") for event in events] == expected
        accounting_ok &= budget_end.get("reported_cost_complete") is True
    else:
        accounting_ok &= budget_start is None and budget_end is None and not events
    usage = summarize_usage(events)
    accounting_ok &= usage["usage_complete"]
    rows = []
    for thread, record in latest.items():
        case = get_case(record["case_id"])
        grade = grade_probe_record(case, record)
        rows.append(
            {
                "case_id": case.case_id,
                "thread_id": thread,
                "track": record["track"],
                "status": record["status"],
                "model_attempts": record["model_attempts"],
                "grade": grade,
                "publication": record["publication"],
                "actions": record["actions"],
            }
        )
    return {
        "runtime": "langgraph",
        "runtime_fingerprints": sorted(
            {report["binding"].get("runtime", "unspecified") for report in reports}
        ),
        "scope": "Fixed synthetic schema; disclosed development smoke, not an unseen benchmark or a framework comparison.",
        "rows": rows,
        "completed_threads": sum(row["status"] == "completed" for row in rows),
        "correct_threads": sum(row["grade"]["correct_task"] for row in rows),
        "wrong_releases": sum(row["grade"]["wrong_release"] for row in rows),
        "process_reports": len(reports),
        "resume_processes": len(reports) - len(latest),
        "resume_http_attempts": resume_http_attempts,
        "real_model_used": bool(events),
        "http_events": events,
        "usage": usage,
        "this_run_http_attempts": len(events),
        "budget_start": budget_start,
        "budget_end": budget_end,
        "source_integrity_ok": all(report.get("source_integrity_ok") is True for report in reports),
        "audit_ledger_complete": bool(accounting_ok),
        "limitation": "Controlled initial SQL is injected. Paused/resumed process reports are one task. Old DB-GPT compression percentages are not LangGraph results.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    summary = evaluate_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "completed_threads",
                    "correct_threads",
                    "wrong_releases",
                    "this_run_http_attempts",
                    "usage",
                    "audit_ledger_complete",
                )
            },
            indent=2,
        )
    )
    return (
        0
        if summary["audit_ledger_complete"]
        and summary["source_integrity_ok"]
        and summary["correct_threads"] == len(summary["rows"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
