"""Opt-in three-party dispute handling; observed agreement is not an oracle.

One diagnosis context may query a bound source and repair any program. Fixtures,
definitions and confirmed outcomes remain host-owned. All previous program versions
and observations survive. This small experiment does not replace the legacy graphs.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from dataverify.runtime.reporting import digest_json
from dataverify.verification.dispute_runtime import DisputeSession
from dataverify.verification.sql_tools import SQLTools

from .discovery_context import model_context
from .dispute_model import (
    DISPUTE_PROMPT,
    FINAL_DECISION_PROMPT,
    SPLIT_DISPUTE_PROMPT,
    SPLIT_FINAL_PROMPT,
    parse_dispute_action,
    parse_final_decision,
)
from .evidence_graph import EvidencePolicy, _message, _prompt_observation
from .graph_model import GraphModelError


@dataclass(frozen=True)
class DisputePolicy(EvidencePolicy):
    max_actions: int = 4
    max_queries: int = 2
    max_no_progress: int = 2
    reserve_final_decision: bool = False
    split_finish_evidence: bool = False

    def __post_init__(self):
        super().__post_init__()
        if type(self.max_no_progress) is not int or not 1 <= self.max_no_progress <= 4:
            raise ValueError("DISPUTE_NO_PROGRESS_POLICY")
        if type(self.reserve_final_decision) is not bool:
            raise ValueError("DISPUTE_FINAL_DECISION_POLICY")
        if type(self.split_finish_evidence) is not bool:
            raise ValueError("DISPUTE_SPLIT_FINISH_POLICY")


class DisputeState(TypedDict):
    replay: dict
    replays: list[dict]
    query_observations: list[dict]
    publication_observations: list[dict]
    query_cache: dict
    action: dict | None
    actions: list[dict]
    feedback: dict | None
    trace: list[dict]
    model_calls: int
    query_calls: int
    revision_attempts: int
    no_progress: int
    status: str
    reason: str
    publication: dict


class _DisputeNodes:
    def __init__(self, task, session, tools, model, policy):
        self.task, self.session, self.tools = task, session, tools
        self.model, self.policy = model, policy

    def event(self, state, node, **details):
        return state["trace"] + [{"node": node, **details}]

    def reject(self, state, code):
        return {
            "action": None,
            "feedback": {"code": code},
            "no_progress": state["no_progress"] + 1,
            "trace": self.event(state, "action_rejected", code=code),
        }

    def final_turn(self, state):
        return (
            self.policy.reserve_final_decision
            and state["model_calls"] + 1 == self.policy.max_actions
        )

    @staticmethod
    def current_query(item, state):
        key = digest_json([item["sql"], state["replay"]["version_binding"]])
        return state["query_cache"].get(key) == item["id"]

    def messages(self, state):
        snapshot, metadata = self.session.snapshot(), self.tools.metadata()
        replay = model_context(
            {
                key: value
                for key, value in state["replay"].items()
                if key
                in {"id", "versions", "checks", "summary", "agreement", "scope", "semantic_status"}
            }
        )
        for check in replay["checks"]:
            for key in ("version_binding", "check_id"):
                check.pop(key, None)
        context = {
            "task": self.task.public_input(),
            "programs": snapshot["programs"],
            "versions": snapshot["versions"],
            "unresolved_assumptions": {
                role: plan.get("assumptions", []) for role, plan in snapshot["plans"].items()
            },
            "latest_replay": replay,
            "query_source": {k: metadata[k] for k in ("scope", "source_id", "limits")},
            "query_observations": [
                {
                    **_prompt_observation(item),
                    "current_version_evidence": self.current_query(item, state),
                }
                for item in state["query_observations"]
            ],
            "publication_observations": [
                _prompt_observation(item) for item in state["publication_observations"]
            ],
            "last_feedback": state["feedback"],
            "remaining": {
                "actions_including_this_one": self.policy.max_actions - state["model_calls"],
                "queries": self.policy.max_queries - state["query_calls"],
                "revisions": self.policy.max_revisions - state["revision_attempts"],
                "tools_enabled": self.policy.tools_enabled,
            },
        }
        final_turn = self.final_turn(state)
        if final_turn:
            context["remaining"].update(queries=0, revisions=0, tools_enabled=False)

        prompt = FINAL_DECISION_PROMPT if final_turn else DISPUTE_PROMPT
        if self.policy.split_finish_evidence:
            prompt = SPLIT_FINAL_PROMPT if final_turn else SPLIT_DISPUTE_PROMPT

        def encoded():
            return [_message("system", prompt), _message("user", context)]

        # Preserve programs, task and execution differences. Only fixture row
        # context may be omitted, explicitly; complete replays stay in the record.
        while sum(len(item["content"]) for item in encoded()) > self.policy.max_context_chars:
            candidates = []
            for check in replay["checks"]:
                data = check.get("data", {})
                tables = data.get("tables", data) if isinstance(data, dict) else {}
                for name, rows in tables.items():
                    if isinstance(rows, list) and rows:
                        candidates.append((rows, check, name))
            if not candidates:
                raise GraphModelError("DISPUTE_CONTEXT_BUDGET")
            rows, check, name = max(candidates, key=lambda item: len(json.dumps(item[0])))
            rows.pop()
            omitted = check.setdefault("context_rows_omitted", {})
            omitted[name] = omitted.get(name, 0) + 1
        return encoded()

    async def act(self, state):
        called = False
        try:
            messages = self.messages(state)
            called = True
            parse = parse_final_decision if self.final_turn(state) else parse_dispute_action
            action = parse(
                await self.model.generate(messages),
                split_finish_evidence=self.policy.split_finish_evidence,
            )
            return {
                "action": action,
                "model_calls": state["model_calls"] + 1,
                "actions": state["actions"] + [action],
                "feedback": None,
                "trace": self.event(state, "model_action", action=action["action"]),
            }
        except GraphModelError as error:
            return {
                **self.reject(state, error.code),
                "model_calls": state["model_calls"] + int(called),
                "status": "running" if error.retryable else "inconclusive",
                "reason": error.code,
            }

    def query(self, state):
        if not self.policy.tools_enabled:
            return self.reject(state, "DISPUTE_TOOLS_DISABLED")
        action = state["action"]
        cache_key = digest_json([action["sql"], state["replay"]["version_binding"]])
        if cache_key in state["query_cache"]:
            return {
                **self.reject(state, "DISPUTE_QUERY_ALREADY_OBSERVED"),
                "feedback": {"cached_observation_id": state["query_cache"][cache_key]},
            }
        if state["query_calls"] >= self.policy.max_queries:
            return self.reject(state, "DISPUTE_QUERY_BUDGET")
        observation = self.tools.execute(
            action["sql"],
            observation_id=f"q{state['query_calls'] + 1:04d}",
            purpose=action["hypothesis"],
        )
        self.session.record_external_observation(observation)
        return {
            "action": None,
            "query_calls": state["query_calls"] + 1,
            "query_observations": state["query_observations"] + [observation],
            "query_cache": {**state["query_cache"], cache_key: observation["id"]},
            "feedback": {"query_observation_id": observation["id"]},
            "no_progress": 0 if observation["status"] == "ok" else state["no_progress"] + 1,
            "trace": self.event(state, "query", observation_id=observation["id"]),
        }

    def revise(self, state):
        if state["revision_attempts"] >= self.policy.max_revisions:
            return self.reject(state, "DISPUTE_REVISION_BUDGET")
        action = state["action"]
        result = self.session.revise(
            action["target"],
            action["program"],
            action["reason"],
            action["evidence_ids"],
            action["requirement_quote"],
        )
        replay = result["observation"]
        feedback = {k: result[k] for k in ("accepted", "error", "versions")}
        if not result["accepted"]:
            trial = self.session.snapshot()["attempts"][-1].get("trial", {})
            role = "independent" if action["target"] == "auditor" else action["target"]
            feedback["rejected_trial_results"] = [
                {
                    "check_id": check["check_id"],
                    "result": check[role],
                    "confirmed_expected": check.get("confirmed_expected"),
                }
                for check in trial.get("checks", [])
                if check[role]["status"] != "ok" or "confirmed_expected" in check
            ][:3]
        return {
            "action": None,
            "replay": replay,
            "replays": state["replays"] + [replay],
            "revision_attempts": state["revision_attempts"] + 1,
            "feedback": feedback,
            "no_progress": 0 if result["accepted"] else state["no_progress"] + 1,
            "trace": self.event(
                state,
                "revision",
                target=action["target"],
                adopted=result["accepted"],
                replay_id=replay["id"],
            ),
        }

    def finish(self, state):
        action = state["action"]
        current_ids = {state["replay"]["id"]} | {check["id"] for check in state["replay"]["checks"]}
        signoff_field = "signoff_ids" if self.policy.split_finish_evidence else "evidence_ids"
        cited_ids = set(action[signoff_field])
        if not cited_ids or not cited_ids <= current_ids:
            return self.reject(state, "DISPUTE_FINISH_REQUIRES_CURRENT_REPLAY_OR_CHECK")
        citations = {}
        if self.policy.split_finish_evidence:
            available = {
                item["id"]
                for item in state["query_observations"]
                if item["status"] == "ok"
                and item["complete"]
                and not item["truncated"]
                and item["source_id"] == self.tools.source_id
                and self.current_query(item, state)
            }
            if not set(action["supporting_ids"]) <= available:
                return self.reject(state, "DISPUTE_SUPPORT_REQUIRES_CURRENT_COMPLETE_QUERY")
            citations = {field: list(action[field]) for field in ("signoff_ids", "supporting_ids")}
        # Citations identify current evidence; they do not select which checks
        # must pass. The whole latest replay remains the acceptance boundary.
        snapshot = self.session.snapshot()
        if any(plan.get("assumptions") for plan in snapshot["plans"].values()):
            return self.reject(state, "DISPUTE_UNRESOLVED_ASSUMPTIONS")
        if not state["replay"]["agreement"]:
            return self.reject(state, "DISPUTE_OBSERVED_DISAGREEMENT")
        observation = self.tools.execute(
            snapshot["programs"]["candidate"],
            observation_id=f"delivery_{len(state['publication_observations']) + 1:04d}",
            purpose="final_candidate_execution",
        )
        self.session.record_external_observation(observation)
        delivery = state["publication_observations"] + [observation]
        if observation["status"] != "ok" or not observation["complete"]:
            return {
                **self.reject(state, "DISPUTE_DELIVERY_EXECUTION_FAILED"),
                "publication_observations": delivery,
                "feedback": {
                    "code": "DISPUTE_DELIVERY_EXECUTION_FAILED",
                    "observation_id": observation["id"],
                    "error": observation["error"],
                },
            }
        return {
            "action": None,
            "status": "reviewed",
            "reason": state["action"]["reason"],
            "publication": {
                "sql": snapshot["programs"]["candidate"],
                "rows": deepcopy(observation["rows"]),
                "assurance": "finite_three_program_agreement_not_semantic_verification",
                "scope": observation["scope"],
                "agreement_scope": "same_bounded_fixture",
                "source_id": observation["source_id"],
                "observation_id": observation["id"],
                "replay_id": state["replay"]["id"],
                **citations,
            },
            "publication_observations": delivery,
            "trace": self.event(state, "finish", replay_id=state["replay"]["id"]),
        }

    def clarify(self, state):
        return {
            "action": None,
            "status": "needs_clarification",
            "reason": state["action"]["reason"],
            "trace": self.event(state, "clarify"),
        }

    def abstain(self, state):
        return {
            "action": None,
            "status": "inconclusive",
            "reason": state["action"]["reason"],
            "trace": self.event(state, "abstain"),
        }

    def next_action(self, state):
        if (
            state["status"] != "running"
            or state["model_calls"] >= self.policy.max_actions
            or state["no_progress"] >= self.policy.max_no_progress
        ):
            return "finalize"
        return "act"

    def route_action(self, state):
        if state["status"] != "running":
            return "finalize"
        return state["action"]["action"] if state["action"] else self.next_action(state)

    def finalize(self, state):
        if state["status"] == "running":
            reason = (
                "DISPUTE_NO_PROGRESS"
                if state["no_progress"] >= self.policy.max_no_progress
                else "DISPUTE_MODEL_BUDGET_EXHAUSTED"
            )
            return {
                "status": "inconclusive",
                "reason": reason,
                "publication": {"sql": None, "rows": None, "reason": reason},
            }
        if state["status"] != "reviewed":
            return {"publication": {"sql": None, "rows": None, "reason": state["reason"]}}
        return {}


async def run_dispute_graph(
    task,
    source_fixture,
    candidate_sql,
    reference_plan,
    auditor_plan,
    model,
    *,
    query_source=None,
    policy=None,
    fixtures=(),
    confirmed_examples=(),
):
    """Diagnose supplied programs; caller owns model/API budget and source binding.

    No program is implicitly trusted. Query scope never changes fixture agreement.
    Checkpoint/resume is deliberately outside this bounded prototype.
    """
    policy = policy or DisputePolicy()
    if not isinstance(policy, DisputePolicy):
        raise ValueError("DISPUTE_POLICY")
    session = DisputeSession(
        task,
        source_fixture,
        candidate_sql,
        reference_plan,
        auditor_plan,
        retained_fixtures=fixtures,
        confirmed_examples=confirmed_examples,
    )
    tools = SQLTools(source_fixture if query_source is None else query_source)
    if tools.metadata()["schema"] != task.public_input()["schema"]:
        raise ValueError("DISPUTE_QUERY_SOURCE_SCHEMA_MISMATCH")
    nodes = _DisputeNodes(task, session, tools, model, policy)
    builder = StateGraph(DisputeState)
    for name in ("act", "query", "revise", "finish", "clarify", "abstain", "finalize"):
        builder.add_node(name, getattr(nodes, name))
    builder.add_edge(START, "act")
    builder.add_conditional_edges("act", nodes.route_action)
    for name in ("query", "revise", "finish"):
        builder.add_conditional_edges(name, nodes.next_action)
    builder.add_edge("clarify", "finalize")
    builder.add_edge("abstain", "finalize")
    builder.add_edge("finalize", END)
    replay = session.replay()
    result = await builder.compile(name="three_party_dispute").ainvoke(
        {
            "replay": replay,
            "replays": [replay],
            "query_observations": [],
            "publication_observations": [],
            "query_cache": {},
            "action": None,
            "actions": [],
            "feedback": None,
            "trace": [],
            "model_calls": 0,
            "query_calls": 0,
            "revision_attempts": 0,
            "no_progress": 0,
            "status": "running",
            "reason": "",
            "publication": {},
        },
        {"recursion_limit": 3 * policy.max_actions + 8},
    )
    snapshot = session.snapshot()
    budget = getattr(model, "budget", None)
    return {
        **dict(result),
        "method": "three_party_dispute_prototype",
        "task_id": task.task_id,
        "candidate": {"sql": snapshot["programs"]["candidate"]},
        "plan": snapshot["plans"]["reference"],
        "audit_plan": snapshot["plans"]["auditor"],
        "versions": snapshot["versions"],
        "history": snapshot["history"],
        "attempts": snapshot["attempts"],
        "fixtures": snapshot["fixtures"],
        "policy": asdict(policy),
        "query_source": tools.metadata(),
        "budget": budget.snapshot() if budget is not None else None,
    }
