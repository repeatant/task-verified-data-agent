"""Run a two-task smoke and the frozen probe experiment under ONE budget.

Offline unless --live is explicit. A live run reads the configured ordinary
Qwen credentials internally; credentials are never printed or written here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.dbgpt_qwen import QwenConfig
from dataverify.probe_cost_budget import MonetaryRequestBudget, validate_pricing
from dataverify.probe_live_benchmark import run_probe_benchmark


def write(path: Path, document: dict) -> None:
    partial = path.with_suffix(".part.json")
    partial.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    partial.replace(path)


def restore_completed_budget(snapshot: dict) -> MonetaryRequestBudget:
    """Replay an earlier completed ledger; never reset spending after a failure.

    Unknown/in-flight usage cannot be safely resumed. This is a continuation
    helper for our own audited reports, not a general provider billing importer.
    """
    if (
        snapshot.get("reported_cost_complete") is not True
        or snapshot.get("stopped_reason")
        or snapshot.get("unknown_count") != 0
        or snapshot.get("in_flight_count") != 0
        or snapshot.get("unmatched_usage_events") != 0
    ):
        raise ValueError("Only complete, unstopped ledgers can be continued")
    expected_prices = {
        "model": "qwen3.8-flash",
        "region": "cn-beijing",
        "currency": "CNY",
        "input_cny_per_million_tokens": 0.8,
        "output_cny_per_million_tokens": 2.7,
    }
    if any(snapshot.get("pricing", {}).get(key) != value for key, value in expected_prices.items()):
        raise ValueError("Prior ledger pricing does not match this experiment")
    budget = MonetaryRequestBudget(
        limit=80,
        max_input_chars=1_000_000,
        max_reported_tokens=250_000,
        money_limit_cny=5,
        max_output_tokens=900,
    )
    for entry in snapshot.get("reservation_ledger", []):
        if entry.get("status") != "usage_complete":
            raise ValueError("Prior reservation has incomplete usage")
        attempt = budget.reserve(entry["prompt_chars"])
        if attempt != entry["attempt"]:
            raise ValueError("Prior ledger attempts are not sequential")
        budget.record_tokens(
            {key: entry[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        )
    restored = budget.snapshot()
    for key in (
        "http_attempts",
        "input_chars",
        "reported_total_tokens",
        "complete_count",
        "reported_prompt_tokens",
        "reported_completion_tokens",
        "reserved_upper_cny",
        "estimated_reported_cny",
        "max_output_tokens",
        "money_limit_cny",
        "reservation_ledger",
    ):
        if restored[key] != snapshot.get(key):
            raise ValueError("Prior ledger could not be reproduced exactly")
    return budget


async def workflow(
    output: Path, *, live: bool = False, previous_workflow: Path | None = None
) -> dict:
    output = output.resolve()
    if not output.is_relative_to((ROOT / "reports").resolve()):
        raise ValueError("Workflow output must remain inside project reports")
    config = QwenConfig.from_local_environment() if live else None
    pricing = validate_pricing(config) if live else None
    budget = MonetaryRequestBudget(
        limit=80,
        max_input_chars=1_000_000,
        max_reported_tokens=250_000,
        money_limit_cny=5,
        max_output_tokens=900,
    )
    if previous_workflow is not None:
        previous_workflow = previous_workflow.resolve()
        if not live or not previous_workflow.is_relative_to((ROOT / "reports").resolve()):
            raise ValueError("Continuation requires an audited live workflow inside reports")
        previous = json.loads(previous_workflow.read_text(encoding="utf-8"))
        if previous.get("real_model_used") is not True:
            raise ValueError("Only a live ledger can be continued")
        budget = restore_completed_budget(previous["budget"])
    output.mkdir(parents=True, exist_ok=False)
    document = {
        "status": "running",
        "real_model_used": live,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pricing": pricing,
        "budget": budget.snapshot(),
        "policy": "One shared budget for smoke and full, CNY 5 maximum preflight reservation; no automatic HTTP retries.",
    }
    if previous_workflow is not None:
        document["continued_from"] = str(previous_workflow.relative_to(ROOT))
    write(output / "workflow.json", document)
    try:
        smoke = await run_probe_benchmark(
            output / "smoke",
            live=live,
            config=config,
            seeds=(17,),
            include_controlled=False,
            case_ids=("d01", "d03"),
            modes=("compact",),
            budget=budget,
        )
        smoke_ok = (
            smoke["all_cells_completed"]
            and smoke["audit_ledger_complete"]
            and all(
                not row.get("grade", {}).get("operational_failure", True) for row in smoke["rows"]
            )
            and budget.snapshot()["reported_cost_complete"]
            and not budget.snapshot()["stopped_reason"]
        )
        document.update(
            smoke_passed=smoke_ok,
            smoke_report="smoke/summary.json",
            budget=budget.snapshot(),
            smoke_all_tasks_correct=all(
                row.get("grade", {}).get("correct_task") for row in smoke["rows"]
            ),
            smoke_gate="Transport, audit and runtime operational health; a safely rejected model answer remains a scored failure.",
        )
        write(output / "workflow.json", document)
        if not smoke_ok:
            document.update(status="smoke_failed", full_run_started=False)
        else:
            full = await run_probe_benchmark(
                output / "full", live=live, config=config, budget=budget
            )
            document.update(
                status=full["status"],
                full_run_started=True,
                full_report="full/summary.json",
                summary=full["summary"],
                usage_full=full["usage"],
                audit_ledger_complete=full["audit_ledger_complete"],
            )
    except BaseException as exc:
        # Keep a partial, safe ledger even if interrupted or a provider fails.
        budget.stop(
            "WORKFLOW_INTERRUPTED"
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            else "WORKFLOW_ERROR"
        )
        document.update(
            status="interrupted"
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            else "error",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        document.update(
            budget=budget.snapshot(), finished_at_utc=datetime.now(timezone.utc).isoformat()
        )
        write(output / "workflow.json", document)
    return document


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--previous-workflow", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / ("probe_workflow_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
    )
    args = parser.parse_args()
    result = asyncio.run(
        workflow(args.output, live=args.live, previous_workflow=args.previous_workflow)
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "budget": result["budget"],
                "report": str(args.output.resolve() / "workflow.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
