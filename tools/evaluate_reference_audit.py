"""Replay explicitly authored attribution scenarios without model API calls."""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.agent.discovery_graph import run_discovery
from dataverify.evaluation.reference_audit_cases import audit_scenarios
from dataverify.runtime.reporting import write_json
from dataverify.verification.discovery_runtime import rows_equal


async def evaluate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for scenario in audit_scenarios():
        writer, reviewer, auditor = scenario.roles()
        result = await run_discovery(
            scenario.task,
            scenario.source,
            writer,
            reviewer,
            auditor=auditor,
            confirmed_examples=scenario.confirmed_examples,
            checkpoint_path=output / (scenario.name + ".sqlite3"),
            thread_id=scenario.name,
        )
        write_json(output / (scenario.name + ".json"), result)
        row = {
            "name": scenario.name,
            "status": result["status"],
            "expected_status": scenario.expected_status,
            "status_matches": result["status"] == scenario.expected_status,
            "published_source_rows_match_gold": rows_equal(
                result["publication"]["rows"] or [], scenario.expected_rows
            ),
            "sql_changed": result["candidate"]["sql"] != scenario.initial_sql,
            "writer_attempts": result["attempts"],
            "reference_revisions": result["reference_revisions"],
            "known_limitation": scenario.known_limitation,
        }
        cases.append(row)
        print(row)
    summary = {
        "scope": "Authored behavior/fault-injection checks. NOT live model accuracy or independent benchmark.",
        "model_actions": "Scripted, including the attribution target.",
        "real_model_used": False,
        "http_calls": 0,
        "cases": cases,
        "known_limitations_are_not_successes": True,
    }
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / ("reference-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
    )
    asyncio.run(evaluate(parser.parse_args().output.resolve()))
