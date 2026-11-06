"""Protection against incorrect reference programs and unsupported attribution."""

import asyncio
import json
from copy import deepcopy

import pytest

from dataverify.agent.discovery_graph import run_discovery
from dataverify.agent.discovery_model import ScriptedReviewModel
from dataverify.agent.graph_checkpoint import GraphResumeError
from dataverify.evaluation.reference_audit_cases import audit_scenarios
from dataverify.runtime.reporting import digest_json
from dataverify.verification.discovery_runtime import rows_equal


def scenario_named(name):
    return next(s for s in audit_scenarios() if s.name == name)


def run(scenario, tmp_path, roles=None, **kwargs):
    writer, reviewer, auditor = roles or scenario.roles()
    result = asyncio.run(
        run_discovery(
            scenario.task,
            scenario.source,
            writer,
            reviewer,
            auditor=auditor,
            confirmed_examples=scenario.confirmed_examples,
            checkpoint_path=tmp_path / "audit.db",
            thread_id="audit",
            **kwargs,
        )
    )
    return result, (writer, reviewer, auditor)


@pytest.mark.parametrize("scenario", audit_scenarios(), ids=lambda s: s.name)
def test_attribution_scenarios(scenario, tmp_path):
    result, (writer, _, auditor) = run(scenario, tmp_path)
    assert result["status"] == scenario.expected_status
    if scenario.known_limitation:
        # Agreement of three correlated implementations is not a semantic proof.
        assert not rows_equal(result["publication"]["rows"], scenario.expected_rows)
        assert "not a semantic proof" in result["publication"]["scope"]
    elif result["status"] == "reviewed":
        assert rows_equal(result["publication"]["rows"], scenario.expected_rows)
    else:
        assert result["publication"]["rows"] is None
    assert len(writer.requests) == (2 if scenario.target == "candidate" else 1)
    assert len(auditor.requests) == 1
    if scenario.target == "reference":
        assert result["reference_revisions"] == 1
        assert result["candidate"]["sql"] == scenario.initial_sql
        assert len(result["plan_versions"]) == 2
        assert result["plan_versions"][0]["plan"] == scenario.reference_plan
        assert result["plan"] == scenario.independent_plan
        retained = {digest_json(x["data"]) for x in result["fixtures"]}
        assert all(digest_json(x["data"]) in retained for x in scenario.reference_plan["fixtures"])
    else:
        assert result["reference_revisions"] == 0


def test_audit_context_is_blind_and_checked_answers_are_not_disclosed(tmp_path):
    scenario = scenario_named("correlated_error_with_anchor")
    result, (_, _, auditor) = run(scenario, tmp_path)
    context = json.loads(auditor.requests[0][-1]["content"])
    assert "task" in context and "data_samples" in context
    assert "candidate_sql" not in context and "expected" not in context
    assert "reference_program" not in context and "reference" not in context
    assert scenario.reference_plan["reference"] not in str(auditor.requests)
    assert scenario.initial_sql not in str(auditor.requests)
    assert result["publication"]["rows"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("checks", ["invented-evidence"]),
        ("basis", [{"source": "question", "quote": "Requirement never provided"}]),
        ("target", "candidate"),
    ],
)
def test_unsupported_attribution_cannot_modify_correct_sql(field, value, tmp_path):
    scenario = scenario_named("wrong_reference_rounding")
    roles = scenario.roles()
    reviewer = roles[1]
    reviewer.actions[1][field] = value
    result, (writer, _, _) = run(scenario, tmp_path, roles)
    assert result["status"] != "reviewed"
    assert len(writer.requests) == 1
    assert result["candidate"]["sql"] == scenario.initial_sql
    assert result["reference_revisions"] == 0


def test_revision_budget_cannot_be_bypassed(tmp_path):
    scenario = scenario_named("wrong_reference_rounding")
    result, (writer, _, _) = run(scenario, tmp_path, max_reference_revisions=0)
    assert result["status"] != "reviewed"
    assert result["reference_revisions"] == 0
    assert len(writer.requests) == 1


def test_bare_repair_request_cannot_skip_attribution(tmp_path):
    scenario = scenario_named("wrong_reference_rounding")
    writer, reviewer, auditor = scenario.roles()
    reviewer.actions[1] = {"action": "repair", "reason": "Just match the expected value"}
    result, _ = run(scenario, tmp_path, (writer, reviewer, auditor))
    assert result["status"] != "reviewed"
    assert len(writer.requests) == 1


def test_auditor_clarification_preserves_candidate(tmp_path):
    scenario = scenario_named("correct_pair")
    writer, reviewer, _ = scenario.roles()
    auditor = ScriptedReviewModel([{"action": "clarify", "reason": "Need a confirmed scope"}])
    result, _ = run(scenario, tmp_path, (writer, reviewer, auditor))
    assert result["status"] == "needs_clarification"
    assert len(writer.requests) == 1


def test_roles_cannot_share_one_client(tmp_path):
    scenario = scenario_named("correct_pair")
    writer, reviewer, _ = scenario.roles()
    with pytest.raises(ValueError, match="SEPARATE"):
        run(scenario, tmp_path, (writer, reviewer, reviewer))


@pytest.mark.parametrize(
    "program,error_code",
    [
        ("[[unclosed", "REFERENCE_SYNTAX"),
        (
            "mapping = {'a': 1}\nfor key in mapping:\n    value = mapping.get(key)\nresult = [[value]]",
            "REFERENCE_FINAL_EXPRESSION_REQUIRED",
        ),
    ],
)
def test_auditor_program_format_retry_keeps_context_blind(tmp_path, program, error_code):
    scenario = scenario_named("correct_pair")
    writer, reviewer, auditor = scenario.roles()
    invalid = deepcopy(scenario.independent_plan)
    invalid["reference"] = program
    auditor.actions.insert(0, invalid)
    result, _ = run(scenario, tmp_path, (writer, reviewer, auditor))
    assert result["status"] == "reviewed"
    assert len(auditor.requests) == 2
    assert scenario.initial_sql not in str(auditor.requests)
    assert scenario.reference_plan["reference"] not in str(auditor.requests[:1])
    feedback = json.loads(auditor.requests[1][-1]["content"])
    assert feedback["validation_error"] == error_code
    assert "expected" not in feedback and "reference_program" not in feedback
    if error_code == "REFERENCE_FINAL_EXPRESSION_REQUIRED":
        assert {d["node"] for d in feedback["dsl_diagnostics"]} == {"Assign", "For", "Attribute"}


def test_rebuilt_reference_survives_completed_resume_without_new_calls(tmp_path):
    scenario = scenario_named("wrong_reference_rounding")
    first, _ = run(scenario, tmp_path)
    resumed, (writer, reviewer, auditor) = run(scenario, tmp_path, resume=True)
    assert resumed["status"] == "reviewed"
    assert resumed["reference_revisions"] == 1
    assert resumed["plan_versions"] == first["plan_versions"]
    assert not writer.requests and not reviewer.requests and not auditor.requests


def test_changed_caller_evidence_invalidates_checkpoint(tmp_path):
    scenario = scenario_named("correlated_error_with_anchor")
    run(scenario, tmp_path)
    writer, reviewer, auditor = scenario.roles()
    with pytest.raises(GraphResumeError, match="BINDING"):
        asyncio.run(
            run_discovery(
                scenario.task,
                scenario.source,
                writer,
                reviewer,
                auditor=auditor,
                confirmed_examples=(),
                checkpoint_path=tmp_path / "audit.db",
                thread_id="audit",
                resume=True,
            )
        )
