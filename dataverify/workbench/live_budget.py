"""One-time, explicit spending limit for a standalone workbench checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from dataverify.agent.graph_checkpoint import (
    JournaledBudget,
    exclusive_run,
    read_budget,
    resume_budget,
)
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget

ROOT = Path(__file__).resolve().parents[2]


def historical_campaign() -> Path | None:
    configured = os.environ.get("DATAVERIFY_BUDGET_CAMPAIGN")
    if configured is not None:
        if not configured.strip():
            raise ValueError("WORKBENCH_EMPTY_CAMPAIGN_PATH")
        return Path(configured).resolve()
    for root in (ROOT, ROOT.parent / "github-publish-v1-20261107"):
        path = root / "reports" / "expanded-v1-20261107"
        if path.exists():
            return path.resolve()
    return None


def account_dir():
    return ROOT / "artifacts" / "workbench" / "accounting"


def initialize_local(limit):
    if historical_campaign() is not None:
        raise ValueError("WORKBENCH_HISTORICAL_CAMPAIGN_TAKES_PRECEDENCE")
    budget = MonetaryRequestBudget(limit=5000, money_limit_cny=limit, max_output_tokens=900)
    directory = account_dir()
    with exclusive_run(directory / "account"):
        receipt, ledger = directory / "authorization.json", directory / "budget.json"
        if receipt.exists() or ledger.exists():
            raise ValueError("WORKBENCH_ACCOUNT_ALREADY_EXISTS")
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump({"id": uuid4().hex, "limit_cny": limit}, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        JournaledBudget(budget, ledger)


def local_account():
    directory = account_dir()
    receipt = json.loads((directory / "authorization.json").read_text("utf-8"))
    snapshot = read_budget(directory / "budget.json")
    if snapshot["money_limit_cny"] != receipt["limit_cny"] or not receipt["id"]:
        raise ValueError("WORKBENCH_ACCOUNT_BINDING_CHANGED")
    if not snapshot["reported_cost_complete"] or snapshot["stopped_reason"]:
        raise ValueError("WORKBENCH_ACCOUNT_REQUIRES_REVIEW")
    return receipt, snapshot


def load_local_lane():
    """Caller owns the fixed account lease for the entire invocation."""
    receipt, snapshot = local_account()
    return receipt, JournaledBudget(resume_budget(snapshot), account_dir() / "budget.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize a bounded local Qwen account once.")
    parser.add_argument(
        "--limit-cny", type=float, required=True, help="Positive maximum, up to 5 CNY."
    )
    args = parser.parse_args()
    try:
        initialize_local(args.limit_cny)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Initialized bounded local Qwen account ({args.limit_cny:g} CNY).")
