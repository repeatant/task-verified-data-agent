"""Authored role-repair contracts; real model efficacy requires separate experiments."""

import asyncio
import json
import sqlite3
from copy import deepcopy

import pytest

from dataverify.agent.dispute_graph import DisputePolicy, run_dispute_graph
from dataverify.agent.dispute_model import is_final_decision, is_split_finish, parse_dispute_action
from dataverify.agent.graph_model import GraphModelError
from dataverify.evaluation.discovery_cases import get_discovery_cases
from dataverify.verification.discovery_runtime import DiscoveryTask
from dataverify.verification.relational_data import RelationalData, SQLiteSource


class ContextScript:
    """Resolve host-issued identifiers for authored actions, without reasoning."""

    def __init__(self, actions, on_request=None):
        self.actions, self.requests, self.first_id = deepcopy(actions), [], None
        self.first_check_id = None
        self.on_request = on_request

    async def generate(self, messages):
        self.requests.append(deepcopy(messages))
        context = json.loads(messages[1]["content"])
        current = context["latest_replay"]["id"]
        self.first_id = self.first_id or current
        check_id = context["latest_replay"]["checks"][0]["id"]
        self.first_check_id = self.first_check_id or check_id
        placeholders = {
            "$current": current,
            "$first": self.first_id,
            "$check": check_id,
            "$first_check": self.first_check_id,
            "$bare_check": check_id.split(":")[-1],
        }
        action = deepcopy(self.actions[len(self.requests) - 1])
        for field in ("evidence_ids", "signoff_ids", "supporting_ids"):
            if field in action:
                action[field] = [placeholders.get(item, item) for item in action[field]]
        if self.on_request:
            self.on_request(len(self.requests))
        # Pass raw terminal proposals through to exercise host-side enforcement.
        return (
            action
            if is_final_decision(messages) or is_split_finish(messages)
            else parse_dispute_action(action)
        )


def revision(target, program, ids=None):
    return {
        "action": "revise",
        "target": target,
        "program": program,
        "reason": "Authored defect explanation based on actual replay and original requirement.",
        "requirement_quote": "先按订单汇总明细金额",
        "evidence_ids": ids or ["$current"],
    }


def finish(ids=None):
    return {
        "action": "finish",
        "reason": "Bounded agreement, not semantic truth.",
        "evidence_ids": ids if ids is not None else ["$current"],
    }


def query():
    return {
        "action": "query",
        "sql": "SELECT order_id, SUM(amount) FROM order_lines GROUP BY order_id",
        "hypothesis": "Inspect per-order totals rather than a line-weighted mean.",
    }


def run(actions, *, bad=(), initial_sql=None, plan=None, audit=None, policy=None, **kwargs):
    case = get_discovery_cases()[0]
    plan, audit = deepcopy(plan or case.offline_plan()), deepcopy(audit or case.offline_plan())
    for role, item in (("reference", plan), ("auditor", audit)):
        if role in bad:
            item["reference"] = "[[0]]"
    model = ContextScript(actions)
    record = asyncio.run(
        run_dispute_graph(
            case.task,
            case.source,
            initial_sql or (case.bad_sql if "candidate" in bad else case.good_sql),
            plan,
            audit,
            model,
            policy=policy or DisputePolicy(max_actions=len(actions)),
            **kwargs,
        )
    )
    return record, model


@pytest.mark.parametrize("target", ["candidate", "reference", "auditor"])
def test_each_fallible_role_can_be_repaired_without_copying_a_majority(target):
    case = get_discovery_cases()[0]
    replacement = case.good_sql if target == "candidate" else case.offline_reference
    result, _ = run([revision(target, replacement), finish()], bad=(target,))
    assert result["status"] == "reviewed"
    assert result["replays"][0]["agreement"] is False
    assert result["replay"]["agreement"] is True
    assert result["versions"][target] == 2
    assert len(result["history"][target]) == 2
    assert result["publication"]["rows"] == case.expected
    assert result["query_calls"] == 0  # Existing observed difference was sufficient.


def test_two_wrong_programs_can_be_changed_sequentially_without_majority_gate():
    case = get_discovery_cases()[0]
    result, _ = run(
        [
            revision("candidate", case.good_sql),
            revision("reference", case.offline_reference),
            finish(),
        ],
        bad=("candidate", "reference"),
    )
    assert result["status"] == "reviewed"
    assert result["replays"][1]["agreement"] is False
    assert result["replay"]["agreement"] is True
    assert result["versions"] == {"candidate": 2, "reference": 2, "auditor": 1}


def test_all_correct_programs_are_retained():
    result, _ = run([finish()])
    assert result["status"] == "reviewed"
    assert result["versions"] == {"candidate": 1, "reference": 1, "auditor": 1}
    assert result["revision_attempts"] == 0
    assert len(result["publication_observations"]) == 1


def test_reserved_last_turn_can_finish_a_repaired_query_and_retains_evidence():
    case = get_discovery_cases()[0]
    policy = DisputePolicy(max_actions=3, reserve_final_decision=True)
    result, model = run(
        [revision("candidate", case.good_sql), query(), finish()], bad=("candidate",), policy=policy
    )
    assert result["status"] == "reviewed"
    assert result["publication"]["rows"] == case.expected
    assert result["model_calls"] == 3 and result["query_calls"] == 1
    assert not is_final_decision(model.requests[1]) and is_final_decision(model.requests[2])
    context = json.loads(model.requests[-1][1]["content"])
    assert context["remaining"] == dict(
        actions_including_this_one=1, queries=0, revisions=0, tools_enabled=False
    )
    assert context["query_observations"][0]["id"] == "q0001"


@pytest.mark.parametrize("action", [query(), revision("candidate", "SELECT 1")])
def test_reserved_last_turn_cannot_do_nonterminal_work_even_when_replay_agrees(action):
    result, _ = run([action], policy=DisputePolicy(max_actions=1, reserve_final_decision=True))
    assert result["status"] == "inconclusive"
    assert result["reason"] == "DISPUTE_INVALID_FINAL_DECISION"
    assert result["query_calls"] == result["revision_attempts"] == 0
    assert result["publication_observations"] == []


@pytest.mark.parametrize(
    ("action", "status"), [("abstain", "inconclusive"), ("clarify", "needs_clarification")]
)
def test_final_decision_can_stop_without_sending_an_answer(action, status):
    reason = (
        "Cannot resolve current discrepancy."
        if action == "abstain"
        else "Which date determines recognition?"
    )
    result, _ = run(
        [{"action": action, "reason": reason}],
        policy=DisputePolicy(max_actions=1, reserve_final_decision=True),
    )
    assert result["status"] == status and result["reason"] == reason
    assert result["publication"]["rows"] is None
    assert result["trace"][-1]["node"] == action


def test_final_decision_keeps_all_replay_guards():
    result, _ = run(
        [finish()],
        bad=("candidate",),
        policy=DisputePolicy(max_actions=1, reserve_final_decision=True),
    )
    assert result["status"] == "inconclusive" and result["publication_observations"] == []
    assert any(node.get("code") == "DISPUTE_OBSERVED_DISAGREEMENT" for node in result["trace"])


@pytest.mark.parametrize("ids", [["$check"], ["$current", "$check"]])
def test_finish_accepts_current_versioned_check_id_with_or_without_root(ids):
    result, _ = run([finish(ids)])
    assert result["status"] == "reviewed"
    assert result["publication"]["replay_id"] == result["replay"]["id"]
    assert len(result["publication_observations"]) == 1


@pytest.mark.parametrize("ids", [["$current", "unknown"], ["$bare_check"], ["q0001"]])
def test_finish_rejects_unknown_bare_or_query_ids_even_with_valid_current_root(ids):
    result, _ = run([finish(ids)])
    assert result["status"] == "inconclusive"
    assert result["publication_observations"] == []
    assert any(
        event.get("code") == "DISPUTE_FINISH_REQUIRES_CURRENT_REPLAY_OR_CHECK"
        for event in result["trace"]
    )


@pytest.mark.parametrize("old_id", ["$first", "$first_check"])
def test_finish_rejects_mixed_current_and_old_round_ids(old_id):
    case = get_discovery_cases()[0]
    result, _ = run(
        [revision("candidate", case.good_sql), finish(["$current", old_id])],
        bad=("candidate",),
    )
    assert result["replay"]["agreement"] is True
    assert result["status"] == "inconclusive"
    assert result["publication_observations"] == []


def test_citing_matching_child_cannot_bypass_other_replay_disagreement():
    result, _ = run([finish(["$check"])], bad=("candidate",))
    cited = result["replay"]["checks"][0]
    assert cited["candidate"]["rows"] == cited["reference"]["rows"] == cited["independent"]["rows"]
    assert result["replay"]["agreement"] is False
    assert result["status"] == "inconclusive"
    assert result["publication_observations"] == []
    assert any(event.get("code") == "DISPUTE_OBSERVED_DISAGREEMENT" for event in result["trace"])


@pytest.mark.parametrize("split", [False, True])
def test_common_wrong_agreement_is_a_documented_negative_control(split):
    result, _ = run(
        [split_finish(["$current"], []) if split else finish()],
        bad=("reference", "auditor"),
        initial_sql="SELECT COUNT(*)*0 FROM orders",
        policy=DisputePolicy(max_actions=1, split_finish_evidence=split),
    )
    assert result["status"] == "reviewed"
    assert result["publication"]["rows"] == [[0]]  # Wrong for original task, finite agreement only.
    assert result["replay"]["semantic_status"] == "not_proven"
    assert "not_semantic_verification" in result["publication"]["assurance"]


def test_query_evidence_can_support_revision_but_is_not_itself_a_verdict():
    case = get_discovery_cases()[0]
    result, _ = run(
        [query(), revision("candidate", case.good_sql, ["q0001"]), finish()], bad=("candidate",)
    )
    assert result["status"] == "reviewed"
    assert result["query_calls"] == 1
    assert result["query_observations"][0]["rows"] == [[1, 120], [2, 80], [3, -20]]
    assert result["history"]["candidate"][1]["evidence_ids"] == ["q0001"]


def test_query_cannot_authorize_finish_while_programs_disagree():
    result, _ = run([query(), finish()], bad=("auditor",))
    assert result["status"] == "inconclusive"
    assert result["publication"]["rows"] is None
    assert any(event.get("code") == "DISPUTE_OBSERVED_DISAGREEMENT" for event in result["trace"])


def split_finish(signoff, supporting):
    return {
        "action": "finish",
        "reason": "Checked the current bounded executions.",
        "signoff_ids": signoff,
        "supporting_ids": supporting,
    }


@pytest.mark.parametrize("terminal", [False, True])
def test_supplement_current_signoff_without_expanding_its_authority(terminal):
    policy = DisputePolicy(
        max_actions=2, split_finish_evidence=True, reserve_final_decision=terminal
    )
    result, _ = run([query(), split_finish(["$check"], ["q0001"])], policy=policy)
    assert result["status"] == "reviewed"
    assert result["publication"]["supporting_ids"] == ["q0001"]
    assert result["publication"]["signoff_ids"] == [result["replay"]["checks"][0]["id"]]
    legacy, _ = run([query(), finish(["$check", "q0001"])])
    assert legacy["status"] == "inconclusive"


@pytest.mark.parametrize(
    ("signoff", "support"),
    [
        ([], ["q0001"]),
        (["q0001"], []),
        (["$check", "q0001"], []),
        (["$bare_check"], []),
        (["$current"], ["$check"]),
        (["$current"], ["q9999"]),
        (["$check"], ["delivery_0001"]),
        (["$current"], ["q0001", "q0001"]),
    ],
)
def test_query_cannot_substitute_for_signoff_or_spoof_a_supporting_id(signoff, support):
    result, _ = run(
        [query(), split_finish(signoff, support)],
        policy=DisputePolicy(max_actions=2, split_finish_evidence=True),
    )
    assert result["status"] == "inconclusive"
    assert not result["publication_observations"]


def test_both_columns_exclude_stale_evidence_after_a_revision():
    case = get_discovery_cases()[0]
    for decision in [split_finish(["$first"], []), split_finish(["$current"], ["q0001"])]:
        result, _ = run(
            [query(), revision("candidate", case.good_sql), decision],
            policy=DisputePolicy(max_actions=3, split_finish_evidence=True),
            bad=("candidate",),
        )
        assert result["replay"]["agreement"] is True
        assert result["status"] == "inconclusive"
        assert not result["publication_observations"]


def test_split_valid_citations_cannot_clear_an_observed_disagreement_or_assumption():
    case = get_discovery_cases()[0]
    plan = case.offline_plan()
    plan["assumptions"] = ["Which interval does the user intend?"]
    for kwargs, reason in [
        ({"plan": plan}, "DISPUTE_UNRESOLVED_ASSUMPTIONS"),
        ({"bad": ("auditor",)}, "DISPUTE_OBSERVED_DISAGREEMENT"),
    ]:
        result, _ = run(
            [query(), split_finish(["$check"], ["q0001"])],
            policy=DisputePolicy(max_actions=2, split_finish_evidence=True),
            **kwargs,
        )
        assert any(e.get("code") == reason for e in result["trace"])
        assert not result["publication_observations"]


def test_failed_or_truncated_queries_are_unusable_as_support():
    from dataverify.verification.probe_runtime import ProbeData

    base = get_discovery_cases()[0].source
    many = ProbeData(
        base.orders, tuple((i, 1, i, "settled", "2026-09-01T00:00:00") for i in range(1, 205))
    )
    for sql in ("SELECT unknown_field FROM orders", "SELECT line_id FROM order_lines"):
        action = dict(query(), sql=sql)
        result, _ = run(
            [action, split_finish(["$current"], ["q0001"])],
            query_source=many,
            policy=DisputePolicy(max_actions=2, split_finish_evidence=True),
        )
        assert not result["query_observations"][0]["complete"]
        assert result["status"] == "inconclusive"
        assert any(
            e.get("code") == "DISPUTE_SUPPORT_REQUIRES_CURRENT_COMPLETE_QUERY"
            for e in result["trace"]
        )


def test_repeat_query_is_cached_and_stops_no_progress_loop():
    result, _ = run([query(), query(), query(), finish()])
    assert result["status"] == "inconclusive"
    assert result["reason"] == "DISPUTE_NO_PROGRESS"
    assert result["model_calls"] == 3
    assert result["query_calls"] == 1


def test_prompt_marks_query_evidence_stale_after_revision_without_mutating_record():
    case = get_discovery_cases()[0]
    result, model = run(
        [query(), revision("candidate", case.good_sql, ["q0001"]), finish()],
        bad=("candidate",),
    )
    before = json.loads(model.requests[1][1]["content"])["query_observations"][0]
    after = json.loads(model.requests[2][1]["content"])["query_observations"][0]
    assert before["current_version_evidence"] is True
    assert after["current_version_evidence"] is False
    assert "current_version_evidence" not in result["query_observations"][0]


def test_old_version_evidence_is_rejected_but_attempt_and_trial_history_survive():
    case = get_discovery_cases()[0]
    result, _ = run(
        [
            revision("candidate", case.good_sql),
            revision("reference", case.offline_reference, ["$first"]),
            revision("reference", case.offline_reference),
            finish(),
        ],
        bad=("candidate", "reference"),
        policy=DisputePolicy(max_actions=4, max_revisions=3),
    )
    assert result["status"] == "reviewed"
    assert result["attempts"][1]["accepted"] is False
    assert result["attempts"][1]["error"]["code"] == "DISPUTE_STALE_EVIDENCE"
    assert result["attempts"][1]["program"] == case.offline_reference


def test_invalid_revision_preserves_old_program_and_specific_feedback():
    result, model = run(
        [
            revision("auditor", "import os\n[[0]]"),
            {"action": "clarify", "reason": "Need a supported calculation."},
        ],
        bad=("auditor",),
    )
    assert result["status"] == "needs_clarification"
    assert result["audit_plan"]["reference"] == "[[0]]"
    assert result["attempts"][0]["accepted"] is False
    assert "error" in json.loads(model.requests[1][1]["content"])["last_feedback"]


def test_finish_cannot_drop_retained_disagreement():
    case = get_discovery_cases()[0]
    # Initial seed agrees for wrong line-average SQL; retained data exposes it.
    model = ContextScript([finish()])
    result = asyncio.run(
        run_dispute_graph(
            case.task,
            case.seed,
            case.bad_sql,
            case.offline_plan(),
            case.offline_plan(),
            model,
            fixtures=[{"name": "retained_difference", "data": case.distinguishing.to_dict()}],
            policy=DisputePolicy(max_actions=1),
        )
    )
    assert result["status"] == "inconclusive"
    assert any(item["name"] == "retained_difference" for item in result["fixtures"])


def test_unknown_evidence_and_made_up_requirement_are_rejected():
    case = get_discovery_cases()[0]
    invalid = revision("candidate", case.good_sql, ["not-returned"])
    result, _ = run(
        [invalid, {"action": "clarify", "reason": "Cannot establish support."}], bad=("candidate",)
    )
    assert result["candidate"]["sql"] == case.bad_sql
    assert result["attempts"][0]["error"]["code"] == "DISPUTE_UNKNOWN_EVIDENCE"


def test_real_query_scope_and_final_delivery_differ_from_fixture(tmp_path):
    path = tmp_path / "PRIVATE_SOURCE.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, amount INTEGER)")
        connection.executemany("INSERT INTO items VALUES(?,?)", [(1, 20), (2, 30)])
    original = SQLiteSource.from_path(path)
    fixture = RelationalData(
        original.schema, {"items": [[1, 1]]}, original.column_types, original.primary_keys
    )
    task = DiscoveryTask(
        "sum",
        "Return sum of amount.",
        "Integer amounts.",
        schema=fixture.schema,
        column_types=fixture.column_types,
        primary_keys=fixture.primary_keys,
        allow_source_only_plan=True,
    )
    plan = {
        "action": "plan",
        "basis": [{"source": "question", "quote": "sum of amount"}],
        "assumptions": [],
        "reference": "[[sum(r['amount'] for r in tables['items'])]]",
        "fixtures": [],
    }
    model = ContextScript(
        [
            {
                "action": "query",
                "sql": "SELECT COUNT(*) FROM items",
                "hypothesis": "Observe original data size.",
            },
            finish(),
        ]
    )
    result = asyncio.run(
        run_dispute_graph(
            task, fixture, "SELECT SUM(amount) FROM items", plan, plan, model, query_source=original
        )
    )
    assert result["replay"]["checks"][0]["candidate"]["rows"] == [[1]]
    assert result["query_observations"][0]["rows"] == [[2]]
    assert result["publication"]["rows"] == [[50]]
    assert result["publication"]["scope"] == "original_database"
    assert result["publication"]["agreement_scope"] == "same_bounded_fixture"
    assert "PRIVATE_SOURCE" not in str(model.requests)
    assert "PRIVATE_SOURCE" not in str(result)


@pytest.mark.parametrize("split", [False, True])
def test_changed_original_database_blocks_delivery_even_when_fixture_agrees(tmp_path, split):
    path = tmp_path / "changing.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items(x INTEGER)")
        connection.execute("INSERT INTO items VALUES(1)")
    source = SQLiteSource.from_path(path)
    fixture = RelationalData(source.schema, {"items": [[1]]}, source.column_types)
    task = DiscoveryTask(
        "s",
        "Return sum x.",
        schema=fixture.schema,
        column_types=fixture.column_types,
        allow_source_only_plan=True,
    )
    plan = {
        "action": "plan",
        "basis": [{"source": "question", "quote": "sum x"}],
        "assumptions": [],
        "reference": "[[sum(r['x'] for r in tables['items'])]]",
        "fixtures": [],
    }

    def change(_):
        with sqlite3.connect(path) as connection:
            connection.execute("INSERT INTO items VALUES(99)")

    model = ContextScript(
        [split_finish(["$current"], []) if split else finish()], on_request=change
    )
    result = asyncio.run(
        run_dispute_graph(
            task,
            fixture,
            "SELECT SUM(x) FROM items",
            plan,
            plan,
            model,
            query_source=source,
            policy=DisputePolicy(max_actions=1, split_finish_evidence=split),
        )
    )
    assert result["replay"]["agreement"] is True
    assert result["status"] == "inconclusive"
    assert result["publication"]["rows"] is None
    assert result["publication_observations"][0]["status"] == "error"
    assert "SOURCE_CHANGED" in result["publication_observations"][0]["error"]["code"]


@pytest.mark.parametrize(
    "action",
    [
        {**revision("auditor", "[[0]]"), "fixtures": []},
        {**revision("unknown", "[[0]]")},
        {**revision("candidate", "select 1"), "reason": "x" * 1001},
    ],
)
def test_action_contract_rejects_authority_changes_unknown_roles_and_oversize_reason(action):
    with pytest.raises(GraphModelError):
        parse_dispute_action(action)
