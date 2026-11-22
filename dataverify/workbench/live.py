"""Connect the workbench to the existing, cumulative Qwen spending journal."""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

from dataverify.agent.discovery_model import QwenReviewModel
from dataverify.agent.dispute_model import QwenDisputeModel
from dataverify.agent.dispute_no_query import QueryDisabledDisputeModel
from dataverify.agent.graph_checkpoint import exclusive_run, read_budget
from dataverify.agent.graph_model import QwenGraphModel
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.campaign_budget import CellBudget
from dataverify.runtime.probe_cost_budget import validate_pricing
from dataverify.runtime.reporting import digest_json

from . import live_budget

MAX_CALLS = 12


def _receipt_path(campaign):
    path = campaign / "budget120" / "authorization.json"
    if not path.is_file():
        raise ValueError("WORKBENCH_EXISTING_BUDGET_REQUIRED")
    return path


def _current_lane(campaign, index):
    directory = _receipt_path(campaign).parent
    prefix = f"budget-lane-{index}"
    pointer = directory / f"{prefix}.current.json"
    name = (
        json.loads(pointer.read_text("utf-8"))["current"] if pointer.exists() else f"{prefix}.json"
    )
    if Path(name).name != name or not name.startswith(prefix + "."):
        raise ValueError("WORKBENCH_INVALID_BUDGET_POINTER")
    return read_budget(directory / name)


def _available(snapshot):
    if snapshot.get("stopped_reason") or snapshot.get("in_flight_count"):
        return 0.0
    return snapshot["money_limit_cny"] - snapshot["reserved_upper_cny"]


def qwen_ready() -> tuple[bool, str]:
    """Read-only status for the UI; acquire leases and validate history on use."""
    try:
        validate_pricing(QwenConfig.from_local_environment())
        campaign = live_budget.historical_campaign()
        if campaign is None:
            _, snapshot = live_budget.local_account()
            amounts = [_available(snapshot)]
        else:
            amounts = [_available(_current_lane(campaign, i)) for i in range(4)]
        if max(amounts) <= 0:
            return False, "累计账本已达到调用上限。"
        return True, "普通千问 API 已配置，按既有累计账本预留后调用。"
    except (ValueError, RuntimeError, OSError, KeyError, TypeError):
        return False, "需要配置普通千问 API 和授权账本；独立安装请按 README 初始化限额。"


class TaskBudget(CellBudget):
    """Count the whole task, including calls before a process restart."""

    def __init__(self, lane, prior_indices, journal_path):
        super().__init__(lane, MAX_CALLS)
        self.indices = list(prior_indices)
        self.journal_path = journal_path

    def reserve(self, chars):
        from dataverify.integrations.qwen_config import QwenClientError

        if len(self.indices) >= self.limit or chars > 32_000:
            raise QwenClientError("WORKBENCH_TASK_REQUEST_LIMIT")
        attempt = self.lane.reserve(chars)
        self.indices.append(attempt)
        _save(self.journal_path, self.indices)
        return attempt

    def snapshot(self):
        # Discovery's continuation check needs a stable *task-scoped* prefix.
        # Other tasks may legitimately have used the same cumulative lane.
        snapshot = self.lane.snapshot()
        indices = set(self.indices)
        records = [row for row in snapshot["reservation_ledger"] if row["attempt"] in indices]
        return {
            "http_attempts": len(records),
            "http_attempt_limit": self.limit,
            "money_limit_cny": snapshot["money_limit_cny"],
            "max_output_tokens": snapshot["max_output_tokens"],
            "pricing": snapshot["pricing"],
            "stopped_reason": snapshot["stopped_reason"],
            "reservation_ledger": records,
            "reported_cost_complete": all(r["status"] == "usage_complete" for r in records),
            "reported_total_tokens": sum(r.get("total_tokens") or 0 for r in records),
        }


def _save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value), encoding="utf-8")
    temp.replace(path)


@contextmanager
def _historical_lane(campaign, directory):
    from tools import continue_expanded as driver

    _receipt_path(campaign)
    with ExitStack() as leases:
        # Same lease order as the historical paid campaign. An occupied lane
        # fails closed instead of racing its append or creating a new allowance.
        leases.enter_context(exclusive_run(campaign / "allocation"))
        for i in range(4):
            leases.enter_context(exclusive_run(campaign / f"lane-{i}-execution"))
        location, receipt = driver.allocate_extension(
            campaign, campaign / "runs/main-v1", approved_limit=120
        )
        lane_file = directory / "budget-binding.json"
        if lane_file.exists():
            binding = json.loads(lane_file.read_text("utf-8"))
            index = binding["lane"]
            if (
                type(index) is not int
                or index not in range(4)
                or binding["receipt"] != digest_json(receipt)
            ):
                raise ValueError("WORKBENCH_BUDGET_BINDING_CHANGED")
        else:
            amounts = [_available(_current_lane(campaign, i)) for i in range(4)]
            if max(amounts) <= 0:
                raise ValueError("WORKBENCH_BUDGET_EXHAUSTED")
            index = amounts.index(max(amounts))
            _save(lane_file, {"receipt": digest_json(receipt), "lane": index})
        yield driver.load_lane(location, index, receipt)


@contextmanager
def _local_lane(directory):
    with exclusive_run(live_budget.account_dir() / "account"):
        receipt, lane = live_budget.load_local_lane()
        binding = {"local_account": digest_json(receipt)}
        path = directory / "budget-binding.json"
        if path.exists() and json.loads(path.read_text("utf-8")) != binding:
            raise ValueError("WORKBENCH_BUDGET_BINDING_CHANGED")
        if not path.exists():
            _save(path, binding)
        yield lane


@contextmanager
def live_models(directory: Path, *, allow_queries: bool = False):
    config = QwenConfig.from_local_environment()
    validate_pricing(config)
    campaign = live_budget.historical_campaign()
    account = (
        _historical_lane(campaign, directory) if campaign is not None else _local_lane(directory)
    )
    with account as lane:
        journal = directory / "call-indices.json"
        indices = json.loads(journal.read_text("utf-8")) if journal.exists() else []
        budget = TaskBudget(lane, indices, journal)
        investigator = QwenDisputeModel if allow_queries else QueryDisabledDisputeModel
        yield tuple(
            cls(config, budget, max_tokens=min(1600, lane.snapshot()["max_output_tokens"]))
            for cls in (QwenGraphModel, QwenReviewModel, QwenReviewModel, investigator)
        )
