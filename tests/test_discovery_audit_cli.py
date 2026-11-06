"""Controlled injections and CLI continuation policy, with no model HTTP calls."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dataverify.evaluation.discovery_roles import InjectedFirstReviewModel
from dataverify.evaluation.reference_audit_cases import audit_scenarios
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget
from dataverify.runtime.reporting import digest_json
from tools import evaluate_discovery as cli


class FakeClient:
    client_kind = "offline_fake_transport"
    model_name = "offline-fake"

    def __init__(self, config=None, budget=None):
        self.budget = budget
        self.requests, self.http_events, self.usage = [], [], []
        self.closed = False

    async def generate(self, messages):
        self.requests.append(deepcopy(messages))
        return {"action": "finish", "reason": "Fake transport only"}

    async def aclose(self):
        self.closed = True


def test_first_reference_injection_has_no_request_or_budget_charge():
    scenario = audit_scenarios()[0]
    budget = MonetaryRequestBudget()
    delegate = FakeClient(budget=budget)
    model = InjectedFirstReviewModel(delegate, scenario.reference_plan)
    before = budget.snapshot()
    messages = [{"role": "user", "content": "original requirement"}]

    async def run():
        injected = await model.generate(messages)
        assert injected == scenario.reference_plan
        assert not delegate.requests and budget.snapshot() == before
        injected["reference"] = "caller mutation"
        assert model._plan["reference"] != "caller mutation"
        result = await model.generate(messages)
        assert result["action"] == "finish" and len(delegate.requests) == 1
        assert model.turn_origins == ["injected_reference_no_api_call", "real_qwen_review_http"]
        assert model.http_events is delegate.http_events
        await model.aclose()
        assert delegate.closed

    asyncio.run(run())


def test_initial_reference_injection_can_be_marked_consumed():
    delegate = FakeClient()
    model = InjectedFirstReviewModel(
        delegate, audit_scenarios()[0].reference_plan, already_injected=True
    )
    result = asyncio.run(model.generate([{"role": "user", "content": "resume"}]))
    assert result["action"] == "finish" and len(delegate.requests) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--audit-scenario", "wrong_reference_rounding"],
        ["--live", "--audit-scenario", "wrong_reference_rounding", "--cases", "order_average"],
        ["--live", "--audit-scenario", "wrong_reference_rounding", "--input", "unused.json"],
        ["--input-char-stop", "1500000"],
        ["--live", "--input-char-stop", "1500000"],
        [
            "--live",
            "--prior-journal",
            "prior.json",
            "--budget-file",
            "never-written.json",
            "--input-char-stop",
            "2000001",
        ],
    ],
)
def test_invalid_cli_modes_fail_before_reading_credentials(argv, monkeypatch):
    def forbid():
        pytest.fail("Credential loading must not occur for rejected CLI arguments")

    monkeypatch.setattr(cli.QwenConfig, "from_local_environment", staticmethod(forbid))
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


def test_input_character_stop_cannot_reseed_existing_journal(tmp_path):
    journal = tmp_path / "exists.json"
    journal.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--live",
                "--prior-journal",
                str(tmp_path / "prior.json"),
                "--budget-file",
                str(journal),
                "--input-char-stop",
                "1500000",
            ]
        )
    assert journal.read_text(encoding="utf-8") == "{}"


def paid_budget():
    budget = MonetaryRequestBudget(limit=10)
    budget.reserve(1000)
    budget.record_tokens({"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300})
    return budget


def test_technical_stops_preserve_every_paid_record():
    budget = paid_budget()
    before = budget.snapshot()
    cli.apply_technical_limits(budget, token_stop=500000, input_char_stop=2000000)
    after = budget.snapshot()
    assert after == {**before, "reported_token_stop_threshold": 500000, "input_char_limit": 2000000}


@pytest.mark.parametrize(
    "limits",
    [
        {"input_char_stop": 999},
        {"input_char_stop": 2000001},
        {"token_stop": 299},
        {"token_stop": 500001},
        {"token_stop": 500000, "input_char_stop": 999},
    ],
)
def test_invalid_technical_stop_does_not_partially_mutate_budget(limits):
    budget = paid_budget()
    before = budget.snapshot()
    with pytest.raises(ValueError):
        cli.apply_technical_limits(budget, **limits)
    assert budget.snapshot() == before


def test_controlled_scenario_keeps_scoring_answers_out_of_model_context(tmp_path, monkeypatch):
    scenario = replace(
        audit_scenarios()[0], final_sql="SCORING_SQL_SENTINEL", expected_rows=[[912345678]]
    )
    monkeypatch.setattr(cli, "audit_scenarios", lambda: [scenario])
    monkeypatch.setattr(cli.QwenConfig, "from_local_environment", staticmethod(lambda: object()))
    monkeypatch.setattr(cli, "validate_pricing", lambda config: None)
    monkeypatch.setattr(cli, "QwenGraphModel", FakeClient)
    monkeypatch.setattr(cli, "QwenReviewModel", FakeClient)

    async def fake_run(task, source, writer, reviewer, *, auditor, **kwargs):
        msg = [{"role": "user", "content": json.dumps(task.public_input())}]
        assert (await reviewer.generate(msg)) == scenario.reference_plan
        candidate = await writer.generate(msg)
        await auditor.generate(msg)
        await reviewer.generate(msg)
        all_context = str(writer.requests + reviewer.requests + auditor.requests)
        assert "SCORING_SQL_SENTINEL" not in all_context and "912345678" not in all_context
        assert writer.budget is reviewer.budget is auditor.budget
        return {
            "status": "reviewed",
            "attempts": 1,
            "fixtures": [],
            "trace": [],
            "publication": {"reason": "Fake graph result", "sql": candidate["sql"], "rows": [[0]]},
            "reference_revisions": 0,
            "binding": {"source_digest": digest_json(source.to_dict())},
        }

    monkeypatch.setattr(cli, "run_discovery", fake_run)
    args = SimpleNamespace(
        output=tmp_path,
        live=True,
        cases=None,
        audit_scenario=[scenario.name],
        input=None,
        track="controlled",
    )
    scores = asyncio.run(cli.evaluate(args))
    assert scores[0]["audit_scenario"] == scenario.name
    assert scores[0]["published_source_rows_match_gold"] is False
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    record = json.loads((tmp_path / "case-0.json").read_text(encoding="utf-8"))
    assert summary["oracle_generation"] == "controlled_initial_reference_then_real_qwen"
    assert summary["real_model_used"] is False
    assert record["controlled_injections"]["initial_reference"] is True
    assert len(record["model_contexts"]["auditor"]) == 1
