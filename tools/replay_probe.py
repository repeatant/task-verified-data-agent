"""Replay one saved counterexample locally, without model/API or gold SQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.probe_runtime import ProbeData, ProbeTask
from dataverify.probe_testing import ExecutionBudget, ProbePair, evaluate_pair


def replay_report(cell: dict, *, compact: bool = True) -> dict:
    report = cell.get("report", cell)
    task = ProbeTask(**report["confirmed_task"])
    evidence = report["counterexample"]
    if evidence is None:
        raise ValueError("This trace has no counterexample")
    name = "compact" if compact else "full"
    payload = evidence[name]
    pair = ProbePair(payload["property_id"], ProbeData(**payload["base"]),
                     payload["added_orders"], payload["added_lines"])
    budget = ExecutionBudget(4, 5)
    verdict = evaluate_pair(report["candidate_sql"], task, pair, budget)
    expected_type = evidence[name + "_verdict"].get("failure_type")
    verified = verdict["status"] == "violation" and verdict.get("failure_type") == expected_type
    return {"reproduced": verified, "property_id": pair.property_id,
            "record_count": pair.record_count, "queries": budget.queries,
            "verdict": verdict, "real_model_used": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    path = args.trace.resolve()
    allowed_roots=((ROOT/"reports").resolve(),(ROOT/"examples").resolve())
    if not any(path.is_relative_to(base) for base in allowed_roots) or path.stat().st_size > 2_000_000:
        raise ValueError("Expected a bounded JSON trace inside project reports/examples")
    result = replay_report(json.loads(path.read_text(encoding="utf-8")), compact=not args.full)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
