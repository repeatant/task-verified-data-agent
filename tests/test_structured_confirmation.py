"""Offline confirmation contracts: pairing, context boundaries and no rerolls."""

import asyncio
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataverify.agent.graph_checkpoint import JournaledBudget
from dataverify.evaluation import structured_confirmation as confirm
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.campaign_budget import CampaignBudget
from dataverify.runtime.reporting import digest_json, write_json
from dataverify.verification.discovery_runtime import DiscoveryTask
from dataverify.verification.relational_data import RelationalData

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "confirmation_cli", ROOT / "tools/evaluate_structured_confirmation.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

PLAN = {
    "action": "plan",
    "basis": [{"source": "question", "quote": "Sum amounts."}],
    "assumptions": [],
    "reference": "[[sum(r['amount'] for r in tables['t'])]]",
    "fixtures": [],
}


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    schema = {"t": ("amount",)}
    task = DiscoveryTask("confirm_0", "Sum amounts.", schema=schema, allow_source_only_plan=True)
    source = RelationalData(schema, {"t": [{"amount": 2}]})
    bundle = SimpleNamespace(
        case_id="confirm_0",
        parent_id="p",
        cluster_id="c",
        family="db",
        condition="natural",
        kind="public",
        task=task,
        source=source,
        truth={"db_path": "not_opened", "gold_rows": [[987654321]]},
    )
    config = QwenConfig(
        "sk-test-not-real", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.8-flash"
    )
    lane = JournaledBudget(
        CampaignBudget(money_limit_cny=28, max_output_tokens=1800), tmp_path / "budget-lane-0.json"
    )
    models = []

    class FakeModel:
        client_kind = "test"

        def __init__(self, config, budget, *, max_tokens, structured_output=False):
            assert max_tokens == 1800
            self.budget, self.structured_output = budget, structured_output
            self.requests, self.http_events, self.closed = [], [], False
            models.append(self)

        async def generate(self, messages):
            self.requests.append(deepcopy(messages))
            attempt = self.budget.reserve(sum(len(m["content"]) for m in messages))
            usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            self.budget.record_tokens(usage)
            self.http_events.append(
                {"attempt": attempt, "status": "completed", "usage": usage, "usage_complete": True}
            )
            return deepcopy(self.action)

        async def aclose(self):
            self.closed = True

    class Writer(FakeModel):
        action = {"task_id": "confirm_0", "sql": "SELECT sum(amount) FROM t"}

    class Reviewer(FakeModel):
        action = PLAN

    monkeypatch.setattr(confirm, "QwenGraphModel", Writer)
    monkeypatch.setattr(confirm, "QwenReviewModel", Reviewer)
    monkeypatch.setattr(
        confirm,
        "score_public",
        lambda truth, record: {
            "correct_delivery": record["status"] == "reviewed",
            "initial_sql_correct": True,
            "wrong_release": False,
        },
    )
    output = tmp_path / "run"
    for sub in ("initial", "references", "records", "checkpoints"):
        (output / sub).mkdir(parents=True)
    return SimpleNamespace(
        bundle=bundle, config=config, lane=lane, output=output, models=models, Reviewer=Reviewer
    )


def test_two_formats_share_sql_but_have_separate_references_and_full_policy(scenario, monkeypatch):
    s = scenario
    graph_calls = []

    async def graph(task, source, writer, reviewer, *, auditor, **kwargs):
        assert writer.budget is reviewer.budget is auditor.budget
        assert kwargs["max_attempts"] == kwargs["max_reviews"] == 3
        assert kwargs["max_challenge_examples"] == 40
        assert "policy" not in kwargs  # Preserves the full-policy default.
        candidate = await writer.generate([{"role": "user", "content": "shared SQL"}])
        reference = await reviewer.generate([{"role": "user", "content": "separate reference"}])
        await auditor.generate([{"role": "user", "content": json.dumps(task.public_input())}])
        graph_calls.append((candidate, reference, writer.client_kind, reviewer.client_kind))
        return {"status": "reviewed", "candidate": candidate, "publication": candidate}

    monkeypatch.setattr(confirm, "run_discovery", graph)

    async def run():
        common = await confirm.shared_sql(s.bundle, s.config, s.lane, s.output / "initial/q.json")
        records = []
        for method in confirm.METHODS:
            records.append(
                await confirm.paired_cell(s.bundle, method, common, s.config, s.lane, s.output)
            )
        used = s.lane.snapshot()["http_attempts"]
        for method in confirm.METHODS:
            await confirm.paired_cell(s.bundle, method, common, s.config, s.lane, s.output)
        assert s.lane.snapshot()["http_attempts"] == used == 5
        return records

    records = asyncio.run(run())
    assert graph_calls[0][0] == graph_calls[1][0]
    assert graph_calls[0][1] == graph_calls[1][1] == PLAN
    assert graph_calls[1][2].endswith("_json_schema_v1")
    assert graph_calls[1][3].endswith("_json_schema_v1")
    assert len(list((s.output / "references").glob("*.json"))) == 2
    assert all(m.closed for m in s.models)
    assert "987654321" not in str([m.requests for m in s.models])
    assert "gold_rows" not in str([m.requests for m in s.models])
    for model in s.models:
        if isinstance(model, s.Reviewer) and model.requests:
            assert "SELECT sum(amount)" not in str(model.requests)
    assert all(r["row"]["initial_reference_accepted"] for r in records)
    assert (
        records[0]["row"]["shared_initial_http_events"]
        == records[1]["row"]["shared_initial_http_events"]
    )


def test_invalid_reference_retains_three_failures_and_is_not_rerolled(scenario, monkeypatch):
    s = scenario
    s.Reviewer.action = {"action": "finish", "reason": "Not a reference plan"}

    async def forbidden(*args, **kwargs):
        pytest.fail("Invalid initial reference must not enter verification graph")

    monkeypatch.setattr(confirm, "run_discovery", forbidden)

    async def run():
        common = await confirm.shared_sql(s.bundle, s.config, s.lane, s.output / "initial/q.json")
        result = await confirm.paired_cell(
            s.bundle, "strict_schema", common, s.config, s.lane, s.output
        )
        repeat = await confirm.paired_cell(
            s.bundle, "strict_schema", common, s.config, s.lane, s.output
        )
        assert repeat == result
        assert s.lane.snapshot()["http_attempts"] == 4
        assert result["row"]["status"] == "initial_reference_invalid"
        assert result["row"]["reference_validation_failures"] == 3

    asyncio.run(run())


def test_incomplete_or_changed_cache_requires_review(tmp_path):
    path = tmp_path / "cache.json"
    confirm.begin(path, {"stage": "paid"})
    with pytest.raises(ValueError, match="REQUIRES_REVIEW"):
        confirm.cached(path, {"stage": "paid"})
    write_json(path, {"binding": {"stage": "paid"}, "completed": True, "error": "failed"})
    assert confirm.cached(path, {"stage": "paid"})["error"] == "failed"
    with pytest.raises(ValueError, match="REQUIRES_REVIEW"):
        confirm.cached(path, {"stage": "changed"})


def test_load_driver_is_dynamic_hash_bound_and_preserves_runtime_path(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys\nsys.path.insert(0, 'unexpected')\n"
        "from dataverify.agent.discovery_model import QwenReviewModel\n"
        "def allocate_extension(*args, **kw): pass\n"
        "def load_lane(*args, **kw): pass\n"
        "def lane_quarantines(*args, **kw): return {3}\n"
        "def rollover_truncation(*args, **kw): pass\n"
    )
    old = list(cli.sys.path)
    module = cli.load_driver(driver, cli.file_digest(driver))
    assert cli.sys.path == old
    assert module.lane_quarantines(None) == {3}
    with pytest.raises(ValueError, match="DRIVER_CHANGED"):
        cli.load_driver(driver, "not-the-hash")


def test_foreign_dataverify_module_is_rejected(monkeypatch):
    monkeypatch.setitem(
        cli.sys.modules, "dataverify.foreign", SimpleNamespace(__file__="C:/foreign/a.py")
    )
    with pytest.raises(ValueError, match="FOREIGN_RUNTIME_MODULE"):
        cli.assert_runtime_isolated()


def test_forecast_does_not_promise_authorized_budget_covers_worst_case():
    forecast = cli.reservation_forecast()
    assert forecast["absolute_max_requests"] == 1241
    assert forecast["absolute_max_reserved_cny"] > 120
    assert forecast["planning_reserved_cny"] < forecast["absolute_max_reserved_cny"]


def test_smoke_requires_usage_and_never_repeats_failed_request(scenario, monkeypatch):
    s = scenario
    monkeypatch.setattr(cli, "QwenReviewModel", s.Reviewer)
    protocol = {"test": "only"}
    s.Reviewer.action = {"action": "finish", "reason": "wrong smoke action"}

    async def run():
        first = await cli.smoke(s.config, s.lane, s.output, protocol)
        s.Reviewer.action = {"action": "clarify", "reason": "fixed after result"}
        second = await cli.smoke(s.config, s.lane, s.output, protocol)
        assert first == second
        assert not second["passed"]
        assert s.lane.snapshot()["http_attempts"] == 1
        assert second["binding"]["protocol_digest"] == digest_json(protocol)

    asyncio.run(run())


def test_summary_explicitly_lists_uncompleted_pairs(scenario):
    result = cli.save_summary(scenario.output, {"cases": [{"case_id": "confirm_0"}]})
    assert result["report_status"] == "interim"
    assert result["missing_cells"] == [
        {"case_id": "confirm_0", "method": method} for method in confirm.METHODS
    ]
    assert result["completed_cells"] == 0


def test_protocol_binds_actual_database_counts_and_rejects_main_overlap(
    scenario, monkeypatch, tmp_path
):
    bundles = [
        SimpleNamespace(
            **{
                **vars(scenario.bundle),
                "case_id": f"q{i}",
                "family": "db1" if i < 18 else "db2" if i < 36 else "db3" if i < 38 else "db4",
            }
        )
        for i in range(40)
    ]
    monkeypatch.setattr(cli, "public_bundles", lambda manifest, split: bundles)
    monkeypatch.setattr(cli, "local_hashes", lambda: {"runtime.py": "frozen"})
    driver = tmp_path / "driver.py"
    driver.write_text("# never executed")
    protocol = cli.build_protocol({}, {"cases": [{"case_id": "old"}]}, driver)
    assert protocol["database_task_counts"] == {"db1": 18, "db2": 18, "db3": 2, "db4": 2}
    assert protocol["expected_cells"] == 80
    with pytest.raises(ValueError, match="OVERLAPS_MAIN"):
        cli.build_protocol({}, {"cases": [{"case_id": "q0"}]}, driver)
