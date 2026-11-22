import asyncio
import json
from pathlib import Path

import pytest

from dataverify.evaluation import signoff_controls as controls
from dataverify.verification.discovery_runtime import execute_readonly, named_tables
from dataverify.verification.reference_expression import calculate
from tools.evaluate_split_signoff import at_fork


@pytest.mark.parametrize("case", controls.cases(), ids=lambda c: c.name)
def test_wrong_candidates_are_scored_independently_of_preview_agreement(case):
    data = controls.sources(case)
    sql_rows = execute_readonly(case.sql, case.preview)
    ref_rows = calculate(case.reference, named_tables(case.preview))
    assert (sql_rows == ref_rows) == (case.name == "wrong-null")
    assert any(
        execute_readonly(case.sql, fixture) != controls.expected(case, fixture)
        for fixture in data.values()
    )
    rec = {
        "status": "reviewed",
        "candidate": {"sql": case.sql},
        "plan": case.plan(case.reference),
        "audit_plan": case.plan(case.auditor),
        "publication": {"sql": case.sql, "rows": execute_readonly(case.sql, data["full_source"])},
    }
    score = controls.grade(case, rec)
    assert score["wrong_release"] and not score["correct_delivery"]
    assert score["reference_good"] == (case.name == "wrong-store")


@pytest.mark.parametrize(
    ("name", "sql"),
    [
        (
            "wrong-store",
            "SELECT COUNT(DISTINCT l.order_id) FROM order_lines l JOIN orders o ON l.order_id=o.order_id WHERE o.channel='store' AND l.status='cancelled'",
        ),
        ("wrong-null", "SELECT COUNT(DISTINCT channel) FROM orders"),
    ],
)
def test_correct_counterparts_pass_but_wrong_display_rows_fail(name, sql):
    case = next(c for c in controls.cases() if c.name == name)
    full = controls.sources(case)["full_source"]
    record = {
        "status": "reviewed",
        "candidate": {"sql": sql},
        "plan": case.plan(case.reference),
        "audit_plan": case.plan(case.auditor),
        "publication": {"sql": sql, "rows": execute_readonly(sql, full)},
    }
    assert controls.grade(case, record)["correct_delivery"]
    record["publication"]["rows"] = [[12345]]
    assert controls.grade(case, record)["wrong_release"]


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("case", controls.cases(), ids=lambda c: c.name)
def test_well_formed_finish_exercises_the_gate_even_if_live_replies_were_invalid(case, split):
    async def force_finish(messages):
        context = json.loads(messages[1]["content"])
        replay = context["latest_replay"]["id"]
        action = {"action": "finish", "reason": "Exercise the publication check."}
        if split:
            action.update(signoff_ids=[replay], supporting_ids=["q0001"])
        else:
            action["evidence_ids"] = [replay]
        return action

    rec, _ = asyncio.run(at_fork(Path("."), "negative", case.name, split, force_finish))
    if case.name == "wrong-store":
        assert rec["publication"]["sql"] is None
        assert rec["trace"][-1]["code"] == "DISPUTE_OBSERVED_DISAGREEMENT"
    else:
        # A correlated error can agree on the preview and pass either finite gate.
        assert rec["status"] == "reviewed"
        assert controls.grade(case, rec)["wrong_release"]


def test_accepted_main_trial_signoffs_still_pass_the_legacy_gate():
    """Replay the actual accepted decisions with only their signing evidence."""
    root = Path(__file__).resolve().parents[1] / "reports"
    trial = root / "split-signoff-v1"
    if not (trial / "summary.json").is_file():
        pytest.skip("The local paid trial is unavailable")
    accepted = 0
    for row in json.loads((trial / "summary.json").read_text("utf-8"))["rows"]:
        if row["arm"] != "split" or row["status"] != "reviewed":
            continue
        group, case = row["group"], row["case"]
        item = json.loads((trial / "records" / f"{group}-{case}-split.json").read_text("utf-8"))
        decision = item["record"]["actions"][-1]

        async def use_signing_evidence(messages):
            # The action contract differs. Everything visible about the business
            # question, programs, source and current replay must match.
            assert json.loads(messages[1]["content"]) == json.loads(
                item["final_prompt"][1]["content"]
            )
            return {
                "action": "finish",
                "reason": decision["reason"],
                "evidence_ids": decision["signoff_ids"],
            }

        old, _ = asyncio.run(
            at_fork(root / "terminal-dev-v1", group, case, False, use_signing_evidence)
        )
        assert old["status"] == "reviewed"
        for field in ("sql", "rows", "source_id", "replay_id"):
            assert old["publication"][field] == item["record"]["publication"][field]
        accepted += 1
    assert accepted == 6
