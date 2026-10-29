"""Native-loop separation of capability hints, repair advice and evidence modes."""

import asyncio
import importlib.util
import json

import pytest

if importlib.util.find_spec("dbgpt") is None:
    pytest.skip("Optional pinned DB-GPT runtime", allow_module_level=True)

from dataverify.probe_agent_bridge import ProbeResource, probe_sql_candidate, run_probe_native_case
from dataverify.probe_cases import get_case
from dataverify.probe_capabilities import policy_features


@pytest.mark.parametrize("policy", ["legacy_v1", "capability_only", "guidance_only", "capability_v2"])
def test_resource_exposes_only_selected_capability_component(policy):
    case = get_case("d10")
    resource = ProbeResource(case.task, feedback_policy=policy)
    try:
        payload = json.loads(resource.get_schema_link(resource._db_name))
        assert ("execution_contract" in payload) == policy_features(policy)["capability_description"]
        assert payload["confirmed_task"] == case.task.to_dict()
        assert "reference_sql" not in json.dumps(payload)
    finally:
        resource.close()


@pytest.mark.parametrize("policy", ["legacy_v1", "capability_only", "guidance_only", "capability_v2"])
def test_ablation_components_enter_the_correct_native_context(tmp_path, policy):
    case = get_case("d10")
    record = asyncio.run(run_probe_native_case(
        case.task, case.source, output_dir=tmp_path,
        feedback_policy=policy, max_attempts=2,
        responses=[probe_sql_candidate(case.candidate_sql, case.task.task_id),
                   probe_sql_candidate(case.reference_sql, case.task.task_id)],
    ))
    assert record["publication"]["status"] == "verified"
    assert record["feedback_features"] == policy_features(policy)
    first_prompt = json.dumps(record["llm_requests"][0])
    second_prompt = json.dumps(record["llm_requests"][1])
    features = policy_features(policy)
    assert ("execution_contract" in first_prompt) == features["capability_description"]
    assert ("repair_guidance" in second_prompt) == features["classified_guidance"]
    assert case.reference_sql not in second_prompt
    assert "reference_sql" not in second_prompt
    assert "fault_label" not in second_prompt
    assert record["http_events"] == []
    assert record["source_unchanged"]
    assert record["raw_resource_queries"] == 0


def test_adaptive_escalation_is_observed_before_the_final_native_repair(tmp_path):
    case = get_case("d06")
    bad = probe_sql_candidate(case.candidate_sql, case.task.task_id)
    record = asyncio.run(run_probe_native_case(
        case.task, case.source, output_dir=tmp_path, mode="adaptive", max_attempts=3,
        responses=[bad, bad, probe_sql_candidate(case.reference_sql, case.task.task_id)],
    ))
    assert record["publication"]["status"] == "verified"
    assert len(record["actions"]) == 3
    first, second = [action["feedback"]["feedback_strategy"] for action in record["actions"][:2]]
    assert first["selected_mode"] == "compact"
    assert second["selected_mode"] == "full"
    assert second["escalation_reason"]
    assert "feedback_strategy" in json.dumps(record["llm_requests"][2])
    assert record["actions"][2]["regression_checks"]
    assert record["http_events"] == []
