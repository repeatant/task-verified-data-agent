"""Run a checkpointed task-verification graph; offline unless --live is explicit."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from dataverify.experiment_budget import load_prior_budget
from dataverify.graph_checkpoint import (
    GraphResumeError,
    JournaledBudget,
    exclusive_run,
    read_budget,
    resume_budget,
)
from dataverify.graph_model import InjectedFirstGraphModel, QwenGraphModel, ScriptedGraphModel
from dataverify.probe_cases import Case, get_case, get_cases
from dataverify.probe_cost_budget import MonetaryRequestBudget, validate_pricing
from dataverify.probe_langgraph import run_graph_case
from dataverify.qwen_config import QwenClientError, QwenConfig
from dataverify.reporting import summarize_usage, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.case_id for case in get_cases()], default="d06")
    parser.add_argument(
        "--track",
        choices=("controlled", "natural"),
        help="controlled injects the disclosed candidate; natural uses model generation only",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "artifacts/langgraph/checkpoints.sqlite3"
    )
    parser.add_argument(
        "--thread", default="demo", help="Use a new thread for each new task/configuration"
    )
    parser.add_argument("--mode", choices=("full", "compact", "adaptive"), default="adaptive")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after", choices=("verify", "feedback"))
    parser.add_argument(
        "--output",
        type=Path,
        help="Private JSON run report; defaults to a timestamped artifacts file",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--budget-file",
        type=Path,
        help="Durable cumulative budget journal, required for live calls",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--prior-budget",
        type=Path,
        help="Audited project report used only to initialize a new live journal",
    )
    initialization.add_argument(
        "--new-budget-cny",
        type=float,
        help="First-use allowance in CNY, >0 and <=5; never use to reset an existing allowance",
    )
    parser.add_argument("--max-extra-requests", type=int, choices=range(1, 13), default=6)
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.live and not args.budget_file:
        parser.error("--live requires --budget-file; cumulative spending must not reset")
    if args.live and args.track is None:
        parser.error("--live requires an explicit --track natural or --track controlled")
    if not args.live and (args.budget_file or args.prior_budget or args.new_budget_cny is not None):
        parser.error("budget options are only valid with --live")
    if args.new_budget_cny is not None and (
        not math.isfinite(args.new_budget_cny) or not 0 < args.new_budget_cny <= 5
    ):
        parser.error("--new-budget-cny must be finite, greater than zero and at most CNY 5")
    has_seed = args.prior_budget is not None or args.new_budget_cny is not None
    if args.live and not args.budget_file.exists() and not has_seed:
        parser.error("a new --budget-file requires --prior-budget or --new-budget-cny")
    if args.live and args.budget_file.exists() and has_seed:
        parser.error("an existing budget journal cannot be reseeded")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.thread):
        parser.error(
            "--thread must contain 1..80 alphanumeric, underscore, dot or hyphen characters"
        )
    paths = [args.checkpoint, args.output, args.budget_file]
    resolved = [path.resolve() for path in paths if path is not None]
    if len(set(resolved)) != len(resolved):
        parser.error("checkpoint, output and budget must use different paths")
    args.track = args.track or "controlled"


async def _completed_attempts(checkpoint: Path, thread_id: str) -> int:
    """Inspect only the committed attempt count; the runner validates all state."""
    if not checkpoint.is_file():
        raise GraphResumeError("THREAD_NOT_FOUND")
    with exclusive_run(checkpoint):
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint.resolve())) as saver:
            saved = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
    if saved is None:
        raise GraphResumeError("THREAD_NOT_FOUND")
    attempts = saved.checkpoint.get("channel_values", {}).get("attempts")
    if type(attempts) is not int or not 0 <= attempts <= 3:
        raise GraphResumeError("INVALID_SAVED_ATTEMPT_COUNT")
    return attempts


def _action(case: Case, sql: str) -> dict[str, str]:
    return {
        "sql": sql,
        "task_id": case.task.task_id,
        "thought": "disclosed offline development replay",
    }


def _offline_model(case: Case, track: str, completed: int) -> ScriptedGraphModel:
    # The CLI owns the full disclosed replay plan. The runtime model API accepts
    # only remaining responses; it never silently skips arbitrary caller input.
    plan = [_action(case, case.reference_sql)]
    if track == "controlled":
        plan.insert(0, _action(case, case.candidate_sql))
    remaining = plan[completed:] or [_action(case, case.reference_sql)]
    return ScriptedGraphModel(remaining)


def _load_live_budget(args: argparse.Namespace) -> JournaledBudget:
    if args.budget_file.exists():
        budget = resume_budget(read_budget(args.budget_file))
    elif args.prior_budget:
        budget = load_prior_budget(
            args.prior_budget,
            project_root=ROOT,
            max_extra_requests=args.max_extra_requests,
        )
    else:
        budget = MonetaryRequestBudget(
            limit=args.max_extra_requests, money_limit_cny=args.new_budget_cny
        )
    return JournaledBudget(budget, args.budget_file)


async def _run_with_model(args: argparse.Namespace, case: Case, model: Any) -> dict[str, Any]:
    budget = getattr(model, "budget", None)
    budget_start = budget.snapshot() if budget is not None else None
    try:
        record = await run_graph_case(
            case.task,
            case.source,
            model,
            checkpoint_path=args.checkpoint,
            thread_id=args.thread,
            mode=args.mode,
            seed=args.seed,
            max_attempts=args.max_attempts,
            resume=args.resume,
            pause_after=args.pause_after,
        )
    finally:
        await model.aclose()
    events = deepcopy(model.http_events)
    budget_end = budget.snapshot() if budget is not None else None
    return {
        **record,
        "case_id": case.case_id,
        "track": args.track,
        "client_kind": model.client_kind,
        "real_model_used": bool(events),
        "model_calls_this_process": len(model.requests),
        "http_events": events,
        "usage": summarize_usage(events),
        "turn_origins": deepcopy(getattr(model, "turn_origins", [])),
        "budget_start": budget_start,
        "budget_end": budget_end,
        "this_run_http_attempts": len(events),
        "source_integrity_ok": record["source_unchanged"],
        "audit_ledger_complete": _accounting_complete(events, budget_start, budget_end),
        "evaluation_scope": (
            "disclosed development fixture; offline responses are authored, not model performance"
            if not args.live
            else "one disclosed development fixture; controlled first faults are not natural model errors"
        ),
    }


def _accounting_complete(events: list[dict], start: dict | None, end: dict | None) -> bool:
    if start is None or end is None:
        return not events
    expected = list(range(start["http_attempts"] + 1, end["http_attempts"] + 1))
    return bool(
        end.get("reported_cost_complete") is True
        and [event.get("attempt") for event in events] == expected
        and summarize_usage(events)["usage_complete"]
    )


async def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    case = get_case(args.case)
    completed = await _completed_attempts(args.checkpoint, args.thread) if args.resume else 0
    if not args.live:
        return await _run_with_model(args, case, _offline_model(case, args.track, completed))
    # Serialize journal load, requests, final usage and client cleanup. The graph
    # separately leases its checkpoint file, so different budget threads cannot
    # overwrite each other's reservations or silently recreate the allowance.
    with exclusive_run(args.budget_file):
        budget = _load_live_budget(args)
        config = QwenConfig.from_local_environment()
        validate_pricing(config)
        model = QwenGraphModel(config, budget)
        if args.track == "controlled":
            model = InjectedFirstGraphModel(
                model,
                _action(case, case.candidate_sql),
                already_injected=completed >= 1,
            )
        return await _run_with_model(args, case, model)


def _error_code(error: Exception) -> str:
    text = str(error)
    return text if re.fullmatch(r"[A-Z][A-Z0-9_]{0,119}", text) else "GRAPH_RUN_FAILED"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    try:
        record = asyncio.run(run_from_args(args))
    except (GraphResumeError, QwenClientError, ValueError, OSError) as error:
        print(json.dumps({"status": "error", "error_code": _error_code(error)}, ensure_ascii=False))
        return 2
    output = args.output or ROOT / "artifacts/langgraph" / (
        args.thread + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "runtime": record["runtime"],
                "thread_id": record["thread_id"],
                "publication": record["publication"],
                "model_attempts": record["model_attempts"],
                "this_run_http_attempts": record["this_run_http_attempts"],
                "report": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if record["status"] == "paused":
        return 0
    return (
        0
        if record["publication"]["status"] == "verified" and record["audit_ledger_complete"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
