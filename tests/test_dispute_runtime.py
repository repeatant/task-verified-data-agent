"""Three-way changes retain evidence, tolerate faulty roles, and never confer truth."""

from copy import deepcopy

import pytest

from dataverify.evaluation.discovery_cases import data
from dataverify.verification.discovery_runtime import DiscoveryTask
from dataverify.verification.dispute_runtime import DisputeSession
from dataverify.verification.relational_data import RelationalData
from dataverify.verification.sql_tools import SQLTools

GOOD_SQL = "SELECT COALESCE(SUM(amount),0) FROM order_lines"
BAD_SQL = "SELECT COALESCE(SUM(ABS(amount)),0) FROM order_lines"
GOOD_DSL = "[[sum(r['amount'] for r in tables['order_lines'])]]"
BAD_DSL = "[[sum(abs(r['amount']) for r in tables['order_lines'])]]"


def build(
    *, candidate=GOOD_SQL, reference=GOOD_DSL, auditor=GOOD_DSL, retained=(), confirmed=False
):
    task = DiscoveryTask("signed_sum", "Sum signed line amounts; keep negative refund values.")
    source = data([100, -20])

    def plan(program):
        return {
            "action": "plan",
            "basis": [{"source": "question", "quote": task.question}],
            "assumptions": [],
            "reference": program,
            "fixtures": [{"name": "seed", "data": data([7]).to_dict()}],
        }

    examples = (
        [
            {
                "name": "human_checked_refund",
                "data": source.to_dict(),
                "expected": [[80]],
                "source": "Trusted caller confirmed: 100 plus -20 is 80.",
            }
        ]
        if confirmed
        else []
    )
    return DisputeSession(
        task, source, candidate, plan(reference), plan(auditor), retained, examples
    )


def revise(session, target, program, *, evidence=None, quote=None):
    evidence = session.replay()["id"] if evidence is None else evidence
    return session.revise(
        target,
        program,
        "Preserve signed amounts rather than applying absolute value.",
        [evidence],
        quote or "keep negative refund values",
    )


@pytest.mark.parametrize("target", ["candidate", "reference", "auditor"])
def test_each_role_can_be_revised_without_changing_other_programs(target):
    session = build(**{target: BAD_SQL if target == "candidate" else BAD_DSL})
    before = session.snapshot()
    first = session.replay()
    assert first["agreement"] is False
    result = revise(
        session,
        target,
        GOOD_SQL if target == "candidate" else GOOD_DSL,
        evidence=first["checks"][0]["id"],
    )
    assert result["accepted"] is True
    assert result["observation"]["agreement"] is True
    assert result["observation"]["semantic_status"] == "not_proven"
    after = session.snapshot()
    assert after["versions"][target] == 2
    assert len(after["history"][target]) == 2
    assert after["history"][target][0]["program"] == before["programs"][target]
    for other in set(before["programs"]) - {target}:
        assert after["programs"][other] == before["programs"][other]
        assert after["versions"][other] == 1


@pytest.mark.parametrize(
    "target,broken",
    [
        ("candidate", "SELECT missing_column FROM order_lines"),
        ("reference", "[["),
        ("auditor", "[[1/0]]"),
    ],
)
def test_initial_program_errors_are_observations_and_remain_repairable(target, broken):
    session = build(**{target: broken})
    initial = session.replay()
    check_role = "independent" if target == "auditor" else target
    assert all(check[check_role]["status"] == "error" for check in initial["checks"])
    assert not initial["agreement"]
    fixed = revise(session, target, GOOD_SQL if target == "candidate" else GOOD_DSL)
    assert fixed["accepted"] is True
    assert fixed["observation"]["agreement"] is True


@pytest.mark.parametrize(
    "target,broken",
    [
        ("candidate", "DELETE FROM order_lines"),
        ("candidate", "SELECT missing_column FROM order_lines"),
        ("reference", "__import__('os').environ"),
        ("auditor", "[[1/0]]"),
    ],
)
def test_invalid_revision_is_recorded_without_replacing_active_version(target, broken):
    session = build()
    before = session.snapshot()
    rejected = revise(session, target, broken)
    assert rejected["accepted"] is False
    assert rejected["error"]
    after = session.snapshot()
    assert after["programs"] == before["programs"]
    assert after["versions"] == before["versions"]
    assert after["history"] == before["history"]
    assert after["attempts"][-1]["program"] == broken
    assert after["attempts"][-1]["accepted"] is False
    assert rejected["observation"]["agreement"] is True


def test_retained_fixture_execution_failure_prevents_revision():
    empty = {"name": "empty", "data": data([]).to_dict()}
    session = build(retained=[empty])
    result = revise(
        session,
        "auditor",
        "[[sum(r['amount'] for r in tables['order_lines'])/len(tables['order_lines'])]]",
    )
    assert result["accepted"] is False
    assert result["error"]["code"] == "DISPUTE_REVISION_EXECUTION_ERROR"
    trial = session.snapshot()["attempts"][-1]["trial"]
    assert any(check["independent"]["status"] == "error" for check in trial["checks"])


def test_all_retained_inputs_and_program_history_survive_revisions():
    first = {"name": "negative_only", "data": data([-3]).to_dict()}
    second = {"name": "empty", "data": data([]).to_dict()}
    session = build(auditor=BAD_DSL, retained=[first])
    initial_count = len(session.snapshot()["fixtures"])
    session.append_fixtures([second])
    result = revise(session, "auditor", GOOD_DSL)
    assert result["accepted"] is True
    state = session.snapshot()
    assert len(state["fixtures"]) == initial_count + 1
    assert {f["name"] for f in state["fixtures"]} >= {"negative_only", "empty", "seed"}
    assert state["history"]["auditor"][0]["program"] == BAD_DSL
    assert state["history"]["auditor"][1]["program"] == GOOD_DSL
    # Snapshot mutations do not replace retained inputs or live programs.
    state["fixtures"].clear()
    state["programs"]["candidate"] = BAD_SQL
    assert len(session.snapshot()["fixtures"]) == initial_count + 1
    assert session.snapshot()["programs"]["candidate"] == GOOD_SQL


def test_two_agreeing_wrong_programs_do_not_block_local_correction():
    session = build(reference=BAD_DSL, auditor=BAD_DSL)
    first = revise(session, "reference", GOOD_DSL)
    assert first["accepted"] is True
    assert first["observation"]["agreement"] is False
    second = revise(session, "auditor", GOOD_DSL)
    assert second["accepted"] is True
    assert second["observation"]["agreement"] is True


def test_common_mode_error_is_agreement_not_truth():
    session = build(candidate=BAD_SQL, reference=BAD_DSL, auditor=BAD_DSL)
    result = session.replay()
    assert result["agreement"] is True
    assert result["semantic_status"] == "not_proven"
    source = next(
        check
        for check in result["checks"]
        if any(o["kind"] == "bound_source" for o in check["origins"])
    )
    assert source["candidate"]["rows"] == [[120]]  # Actual signed sum is 80.
    assert session.snapshot()["versions"] == dict.fromkeys(["candidate", "reference", "auditor"], 1)
    assert session.snapshot()["attempts"] == []


def test_confirmed_example_blocks_a_new_wrong_program_even_with_a_true_quote():
    session = build(confirmed=True)
    result = revise(session, "candidate", BAD_SQL)
    assert result["accepted"] is False
    assert result["error"]["code"] == "DISPUTE_CONFIRMED_EXAMPLE_REGRESSION"
    assert session.snapshot()["programs"]["candidate"] == GOOD_SQL


def test_true_quote_and_real_observation_are_not_semantic_authorization():
    session = build()
    result = revise(session, "candidate", BAD_SQL)
    # Without caller-confirmed expected values, the host cannot prove that a
    # model's interpretation is right. Keep this limitation visible in the API.
    assert result["accepted"] is True
    assert result["observation"]["agreement"] is False
    assert result["semantic_status"] == "not_proven"


@pytest.mark.parametrize(
    "evidence,code",
    [
        ("invented_observation", "DISPUTE_UNKNOWN_EVIDENCE"),
        ("data_looks_like_a_check", "DISPUTE_UNKNOWN_EVIDENCE"),
    ],
)
def test_model_cannot_fabricate_a_registry_evidence_id(evidence, code):
    result = revise(build(auditor=BAD_DSL), "auditor", GOOD_DSL, evidence=evidence)
    assert not result["accepted"]
    assert result["error"]["code"] == code


def test_old_version_evidence_and_old_fixture_set_cannot_authorize_new_revision():
    session = build(reference=BAD_DSL, auditor=BAD_DSL)
    initial = session.replay()
    assert revise(session, "reference", GOOD_DSL, evidence=initial["id"])["accepted"]
    stale = revise(session, "auditor", GOOD_DSL, evidence=initial["checks"][0]["id"])
    assert stale["error"]["code"] == "DISPUTE_STALE_EVIDENCE"
    latest = session.replay()
    session.append_fixtures([{"name": "new_case", "data": data([-8]).to_dict()}])
    stale = revise(session, "auditor", GOOD_DSL, evidence=latest["id"])
    assert stale["error"]["code"] == "DISPUTE_STALE_EVIDENCE"


def test_quote_must_occur_in_trusted_requirement():
    result = revise(
        build(auditor=BAD_DSL), "auditor", GOOD_DSL, quote="All refunds must be discarded."
    )
    assert not result["accepted"]
    assert result["error"]["code"] == "DISPUTE_REQUIREMENT_QUOTE"


def test_external_query_is_registered_but_does_not_count_as_three_way_agreement():
    session = build(auditor=BAD_DSL)
    before = session.replay()
    tool = SQLTools(data([100, -20]))
    observation = tool.execute(
        "SELECT amount FROM order_lines WHERE amount < 0", observation_id="query_1"
    )
    assert session.record_external_observation(observation) == "query_1"
    after = session.replay()
    assert after["summary"] == before["summary"]
    assert after["agreement"] is False
    fixed = revise(session, "auditor", GOOD_DSL, evidence="query_1")
    assert fixed["accepted"] is True
    with pytest.raises(ValueError, match="ID_REUSED"):
        session.record_external_observation(observation)


def test_changed_tool_record_and_source_are_rejected():
    session = build()
    first = SQLTools(data([100, -20])).execute(GOOD_SQL, observation_id="query_1")
    changed = deepcopy(first)
    changed["rows"] = [[999999]]
    with pytest.raises(ValueError, match="OBSERVATION_CHANGED"):
        session.record_external_observation(changed)
    session.record_external_observation(first)
    different = SQLTools(data([7])).execute(GOOD_SQL, observation_id="query_2")
    with pytest.raises(ValueError, match="QUERY_SOURCE_CHANGED"):
        session.record_external_observation(different)


def test_no_actual_program_change_is_not_a_repair():
    result = revise(build(), "auditor", GOOD_DSL)
    assert not result["accepted"]
    assert result["error"]["code"] == "DISPUTE_NO_PROGRAM_CHANGE"


def test_generic_schema_is_replayed_on_same_bounded_rows():
    source = RelationalData(
        {"measurements": ["value"]}, {"measurements": [[2], [5]]}, {"measurements": ["INTEGER"]}
    )
    task = DiscoveryTask(
        "generic",
        "Sum measurements.value.",
        schema=source.schema,
        column_types=source.column_types,
        allow_source_only_plan=True,
    )
    correct = {
        "action": "plan",
        "basis": [{"source": "question", "quote": task.question}],
        "assumptions": [],
        "reference": "[[sum(r['value'] for r in tables['measurements'])]]",
        "fixtures": [],
    }
    wrong = {**correct, "reference": "[[0]]"}
    session = DisputeSession(task, source, "SELECT SUM(value) FROM measurements", correct, wrong)
    initial = session.replay()
    result = session.revise(
        "auditor",
        correct["reference"],
        "Use all source measurements.",
        [initial["id"]],
        task.question,
    )
    assert result["accepted"] is True
    assert result["observation"]["checks"][0]["independent"]["rows"] == [[7]]


def test_foreign_schema_fixture_is_not_accepted_as_new_evidence():
    session = build()
    foreign = RelationalData({"other": ["value"]}, {"other": [[100]]})
    with pytest.raises(ValueError, match="SCHEMA_MISMATCH"):
        session.append_fixtures([{"name": "foreign", "data": foreign.to_dict()}])


def test_session_requires_actual_bounded_source_not_path():
    with pytest.raises(TypeError, match="BOUNDED_SOURCE"):
        DisputeSession(DiscoveryTask("x", "Read value."), "file.sqlite", GOOD_SQL, {}, {})
