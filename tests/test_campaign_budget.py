"""Cost lineage is durable across cells, lanes and known-truncation continuation."""

import json
from copy import deepcopy

import pytest

from dataverify.agent.graph_checkpoint import JournaledBudget, read_budget
from dataverify.integrations.qwen_config import QwenClientError
from dataverify.runtime.campaign_budget import (
    CellBudget,
    authorize_campaign,
    continue_accounted_truncation,
    lane_budget,
)
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget
from dataverify.runtime.reporting import write_json


def setup_campaign(tmp_path, workers=4):
    parents = [tmp_path / "old5.json", tmp_path / "old3.json"]
    for path, allowance in zip(parents, (5, 3)):
        JournaledBudget(MonetaryRequestBudget(money_limit_cny=allowance), path)
    output = tmp_path / "campaign"
    receipt = authorize_campaign(output, parents, workers=workers)
    return output, parents, receipt


def account(budget, output=5):
    attempt = budget.reserve(100)
    usage = {"prompt_tokens": 10, "completion_tokens": output, "total_tokens": 10 + output}
    budget.record_tokens(usage)
    return attempt, usage


def truncated_report(budget):
    start = budget.snapshot()
    attempt, usage = account(budget, 1800)
    budget.stop("PROVIDER_ERROR")
    return {
        "budget_start": start,
        "budget_end": budget.snapshot(),
        "http_events": [
            {
                "attempt": attempt,
                "http_status": 200,
                "usage_complete": True,
                "usage": usage,
                "status": "failed",
                "error_code": "GRAPH_QWEN_RESPONSE_TRUNCATED",
            }
        ],
    }


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_allocations_preserve_old_eight_and_never_exceed_sixty(tmp_path, workers):
    output, parents, receipt = setup_campaign(tmp_path, workers)
    lanes = [lane_budget(output, index, receipt) for index in range(workers)]
    assert sum(item.snapshot()["money_limit_cny"] for item in lanes) == 52
    assert sum(read_budget(path)["money_limit_cny"] for path in parents) == 8
    assert authorize_campaign(output, parents, workers=workers) == receipt


@pytest.mark.parametrize("workers", [0, 3, 5, True])
def test_unsupported_lane_counts_fail_before_allocation(tmp_path, workers):
    with pytest.raises(ValueError, match="ALLOWANCE_OR_WORKERS"):
        setup_campaign(tmp_path, workers)


def test_parent_allocation_cannot_be_reused_elsewhere(tmp_path):
    _, parents, _ = setup_campaign(tmp_path)
    with pytest.raises(ValueError, match="ALREADY_ALLOCATED"):
        authorize_campaign(tmp_path / "other", parents)


def test_changed_authorization_or_parent_rejected_on_resume(tmp_path):
    output, parents, receipt = setup_campaign(tmp_path)
    mutated = deepcopy(receipt)
    mutated["new_allowance_cny"] = 100
    write_json(output / "authorization.json", mutated)
    with pytest.raises(ValueError, match="AUTHORIZATION_CHANGED"):
        lane_budget(output, 0, receipt)
    write_json(output / "authorization.json", receipt)
    old = json.loads(parents[0].read_text(encoding="utf-8"))
    old["budget"]["money_limit_cny"] = 6
    write_json(parents[0], old)
    with pytest.raises(ValueError, match="ALLOWANCE_OR_WORKERS"):
        lane_budget(output, 0, receipt)


def test_resume_replays_every_reservation_without_refunding(tmp_path):
    output, _, receipt = setup_campaign(tmp_path)
    lane = lane_budget(output, 0, receipt)
    account(lane)
    assert lane_budget(output, 0, receipt).snapshot() == lane.snapshot()
    lane2 = lane_budget(output, 1, receipt)
    assert lane2.snapshot()["http_attempts"] == 0


def test_cell_limit_is_local_but_accounting_is_cumulative(tmp_path):
    output, _, receipt = setup_campaign(tmp_path)
    lane = lane_budget(output, 0, receipt)
    cell = CellBudget(lane, limit=1)
    account(cell)
    with pytest.raises(QwenClientError, match="CELL_HTTP"):
        account(cell)
    assert lane.snapshot()["stopped_reason"] is None
    account(CellBudget(lane, limit=1))
    assert lane.snapshot()["http_attempts"] == 2


def test_accounted_truncation_preserves_old_file_and_new_history(tmp_path):
    output, _, receipt = setup_campaign(tmp_path)
    lane = lane_budget(output, 0, receipt)
    report = truncated_report(lane)
    old_path, old_bytes = lane.path, lane.path.read_bytes()
    renewed = continue_accounted_truncation(output, 0, receipt, lane, report)
    assert old_path.read_bytes() == old_bytes
    assert renewed.path != old_path
    assert renewed.snapshot()["reservation_ledger"] == lane.snapshot()["reservation_ledger"]
    assert renewed.snapshot()["reserved_upper_cny"] == lane.snapshot()["reserved_upper_cny"]
    assert renewed.snapshot()["stopped_reason"] is None
    account(renewed)
    assert lane_budget(output, 0, receipt).snapshot() == renewed.snapshot()
    second_report = truncated_report(renewed)
    second = continue_accounted_truncation(output, 0, receipt, renewed, second_report)
    assert second.snapshot()["http_attempts"] == 3
    assert lane_budget(output, 0, receipt).snapshot() == second.snapshot()


@pytest.mark.parametrize("mutation", ["usage", "error", "http", "parent"])
def test_other_failures_cannot_clear_stop(tmp_path, mutation):
    output, _, receipt = setup_campaign(tmp_path)
    lane = lane_budget(output, 0, receipt)
    report = truncated_report(lane)
    if mutation == "usage":
        report["http_events"][0]["usage"]["completion_tokens"] -= 1
    elif mutation == "error":
        report["http_events"][0]["error_code"] = "GRAPH_QWEN_TRANSPORT_TIMEOUT"
    elif mutation == "http":
        report["http_events"][0]["http_status"] = 500
    else:
        report["budget_end"]["http_attempts"] += 1
    with pytest.raises(ValueError):
        continue_accounted_truncation(output, 0, receipt, lane, report)
    assert lane.snapshot()["stopped_reason"] == "PROVIDER_ERROR"
    assert not (output / "budget-lane-0.current.json").exists()


def test_changed_continuation_parent_is_detected(tmp_path):
    output, _, receipt = setup_campaign(tmp_path)
    lane = lane_budget(output, 0, receipt)
    report = truncated_report(lane)
    continue_accounted_truncation(output, 0, receipt, lane, report)
    old = json.loads(lane.path.read_text(encoding="utf-8"))
    old["budget"]["reserved_upper_cny"] = 0
    write_json(lane.path, old)
    with pytest.raises(ValueError, match="PARENT_CHANGED"):
        lane_budget(output, 0, receipt)
