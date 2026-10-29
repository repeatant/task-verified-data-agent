"""Offline integration check against DB-GPT's actual native agent loop.

Run with .venv-dbgpt/Scripts/python.exe tools/validate_dbgpt.py.
No real model calls, private data or API key loading.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.database import seed_demo_database
from dataverify.dbgpt_bridge import (
    BAD_ORDER_SUM,
    GOOD_LINE_JOIN,
    GOOD_ORDER_SUM,
    UPSTREAM_COMMIT,
    run_native_case,
    sql_candidate,
)
from dataverify.guarded_task import ConfirmedMetricTask


def request_text(record, index=1):
    return json.dumps(record["llm_requests"][index], ensure_ascii=False)


async def validate(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    write_summary(output, {"status": "running", "all_checks_passed": False})
    database = seed_demo_database(output / "synthetic.sqlite3")
    order_task = ConfirmedMetricTask("Calculate total order revenue.", "order_revenue")
    line_task = ConfirmedMetricTask("Calculate total order-line revenue.", "line_revenue")
    bad = sql_candidate(BAD_ORDER_SUM)
    good = sql_candidate(GOOD_ORDER_SUM)
    records = {}
    for mode in ("original", "code", "evidence"):
        records[mode] = await run_native_case(
            database,
            order_task,
            [bad, good],
            mode=mode,
            output_dir=output / "runtime",
        )
    records["valid_detail_join"] = await run_native_case(
        database,
        line_task,
        [sql_candidate(GOOD_LINE_JOIN, "line_revenue")],
        output_dir=output / "runtime",
    )
    records["metric_drift"] = await run_native_case(
        database,
        order_task,
        [sql_candidate(GOOD_LINE_JOIN, "line_revenue"), good],
        output_dir=output / "runtime",
    )
    records["retry_exhausted"] = await run_native_case(
        database,
        order_task,
        [bad, bad],
        output_dir=output / "runtime",
    )
    records["malformed_output"] = await run_native_case(
        database,
        order_task,
        ["not JSON", good],
        output_dir=output / "runtime",
    )
    # Retain raw observations even when an assertion catches an integration bug.
    (output / "native_traces.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    original = records["original"]
    original_action = original["native_reply"]["action_report"]
    original_content = json.loads(original_action["content"])
    assert original["upstream_success"], original["native_reply"]
    assert original["llm_calls"] == 1
    # Pin what the upstream really accepted rather than reconstructing its value.
    assert original_content["data"][0]["revenue"] == 540, original_content
    for mode in ("code", "evidence"):
        record = records[mode]
        assert record["upstream_success"], record["native_reply"]
        assert record["llm_calls"] == 2
        assert record["publication"]["accepted_value"] == 280
        assert record["actions"][0]["verification_status"] == "rejected"
        error_code = record["actions"][0]["error_code"]
        assert error_code in request_text(record), request_text(record)
        act_events = [event for event in record["stream_events"] if event["event_type"] == "act"]
        first_action = act_events[0]["payload"]["action_output"]
        assert first_action["view"] is None
        assert json.loads(first_action["content"])["accepted_value"] is None
        assert record["raw_resource_queries"] == []  # No separate SQL re-execution.
    # The evidence must travel through upstream's real messages, not a local trace.
    witness = records["evidence"]["actions"][0]["evidence"][0]
    assert witness, records["evidence"]["actions"][0]
    for key in witness:
        assert key in request_text(records["evidence"]), (key, witness)
    # Schema legitimately contains source_table/grain_column in every group;
    # check actual counterexample facts rather than those shared metadata keys.
    for diagnostic_key in ("duplicate_grain", "max_multiplicity", "order_id=3"):
        assert diagnostic_key not in request_text(records["code"]), diagnostic_key
    assert records["valid_detail_join"]["publication"]["accepted_value"] == 280
    assert records["valid_detail_join"]["llm_calls"] == 1
    assert records["metric_drift"]["actions"][0]["error_code"] == "METRIC_DRIFT"
    assert records["metric_drift"]["actions"][0]["execution_status"] == "blocked"
    assert records["metric_drift"]["publication"]["accepted_value"] == 280
    assert records["retry_exhausted"]["publication"]["status"] == "unverified"
    assert records["retry_exhausted"]["publication"]["accepted_value"] is None
    assert records["malformed_output"]["publication"]["accepted_value"] == 280

    summary = {
        "upstream_commit": UPSTREAM_COMMIT,
        "validation_kind": "offline_native_loop_integration_not_effectiveness_benchmark",
        "cases": {
            name: {
                "upstream_success": record["upstream_success"],
                "llm_calls": record["llm_calls"],
                "published_value": record["publication"]["accepted_value"]
                if record["publication"]
                else None,
                "first_rejection": record["actions"][0].get("error_code")
                if record["actions"]
                else None,
            }
            for name, record in records.items()
        },
        "original_accepted_wrong_value": original_content["data"][0]["revenue"],
        "evidence_observed_in_native_retry": True,
        "code_group_has_no_counterexample": True,
        "all_checks_passed": True,
    }
    summary["status"] = "passed"
    write_summary(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_summary(output: Path, summary: dict):
    partial = output / "summary.part.json"
    partial.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(output / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "dbgpt_native_20261028")
    args = parser.parse_args()
    try:
        asyncio.run(validate(args.output.resolve()))
    except Exception as error:
        write_summary(
            args.output.resolve(),
            {
                "status": "failed",
                "all_checks_passed": False,
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            },
        )
        raise
