"""Opt-in, bounded SQL investigation with actual database observations.

This prototype leaves the three-way discovery graph unchanged. It tests a smaller
question: can an independent review context select useful read-only queries and
make supported SQL repairs? ``reviewed`` means executed and model-reviewed, not
verified against an oracle or proven semantically correct. Evaluation must score
both correct deliveries and damaged initially-correct queries externally.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from dataverify.runtime.reporting import digest_json
from dataverify.verification.discovery_runtime import DiscoveryTask
from dataverify.verification.sql_tools import SQLTools

from .evidence_model import EVIDENCE_PROMPT, parse_evidence_action
from .graph_model import GraphModelError


@dataclass(frozen=True)
class EvidencePolicy:
    """Predeclared resource limits, including a tool-disabled ablation."""

    tools_enabled: bool = True
    max_actions: int = 6
    max_queries: int = 3
    max_revisions: int = 2
    max_context_chars: int = 11_000

    def __post_init__(self):
        if type(self.tools_enabled) is not bool:
            raise ValueError("EVIDENCE_TOOL_POLICY")
        for name, low, high in (
            ("max_actions", 1, 12),
            ("max_queries", 0, 8),
            ("max_revisions", 0, 4),
            ("max_context_chars", 4000, 38_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("EVIDENCE_BUDGET_POLICY")


class EvidenceState(TypedDict):
    candidate: dict
    current_observation_id: str
    observations: list[dict]
    versions: list[dict]
    trace: list[dict]
    actions: list[dict]
    action: dict | None
    feedback: dict | None
    model_calls: int
    query_calls: int
    revision_attempts: int
    status: str
    reason: str
    publication: dict


def _message(role, value):
    return {
        "role": role,
        "content": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
    }


def _prompt_observation(observation):
    """Bound prompt size without changing the durable observation or its scope."""
    result = {
        key: deepcopy(value)
        for key, value in observation.items()
        if key
        in {
            "id",
            "sql",
            "status",
            "columns",
            "rows",
            "truncated",
            "complete",
            "error",
            "purpose",
            "candidate_version",
            "truncation_reason",
        }
    }
    rows = result.get("rows") or []
    shown = rows[:12]
    # Whole rows are retained; any omission is explicit and cannot be mistaken
    # for tool-level completeness. Full observations stay in the returned record.
    while shown and len(json.dumps(shown, ensure_ascii=False)) > 3000:
        shown = shown[:-1]
    result["rows"] = shown
    result["context_rows_omitted"] = len(rows) - len(shown)
    return result


class _EvidenceNodes:
    def __init__(self, task, tools, initial_sql, model, policy):
        self.task, self.tools = task, tools
        self.initial_sql, self.model, self.policy = initial_sql, model, policy

    def observation(self, state, sql, purpose):
        identifier = f"e{len(state['observations']) + 1:04d}"
        result = self.tools.execute(sql, observation_id=identifier, purpose=purpose)
        return {
            **result,
            "candidate_version": state["candidate"]["version"],
            "purpose": purpose,
        }

    def event(self, state, node, **values):
        return state["trace"] + [{"node": node, **values}]

    def start(self, state):
        observation = self.observation(state, self.initial_sql, "initial_candidate")
        version = {
            "version": 1,
            "sql": self.initial_sql,
            "sql_digest": digest_json(self.initial_sql),
            "observation_id": observation["id"],
            "accepted": True,
            "origin": "initial_candidate",
        }
        return {
            "observations": [observation],
            "current_observation_id": observation["id"],
            "versions": [version],
            "trace": self.event(state, "initial_execution", observation_id=observation["id"]),
        }

    def messages(self, state):
        metadata = self.tools.metadata()
        # Task metadata is supplied once. The trusted tool binding adds source
        # scope/limits, never a filesystem path or scoring-only gold answer.
        source = {
            key: value for key, value in metadata.items() if key in {"source_id", "scope", "limits"}
        }
        context = {
            "task": self.task.public_input(),
            "source": source,
            "candidate": state["candidate"],
            "current_execution_id": state["current_observation_id"],
            "observations": [_prompt_observation(item) for item in state["observations"]],
            "actions": [
                {key: value for key, value in action.items() if key != "sql"}
                for action in state["actions"][-2:]
            ],
            "last_feedback": state["feedback"],
            "remaining": {
                "model_actions_including_this_one": self.policy.max_actions - state["model_calls"],
                "queries": self.policy.max_queries - state["query_calls"],
                "revision_attempts": self.policy.max_revisions - state["revision_attempts"],
                "tools_enabled": self.policy.tools_enabled,
            },
        }

        def messages():
            return [_message("system", EVIDENCE_PROMPT), _message("user", context)]

        while sum(len(item["content"]) for item in messages()) > self.policy.max_context_chars:
            with_rows = [item for item in context["observations"] if item["rows"]]
            if with_rows:
                largest = max(with_rows, key=lambda item: len(json.dumps(item["rows"])))
                largest["rows"].pop()
                largest["context_rows_omitted"] += 1
            elif context["actions"]:
                context["actions"].pop(0)
            else:
                # Do not silently truncate the original requirement, candidate,
                # evidence identifiers, errors or SQL under investigation.
                raise GraphModelError("EVIDENCE_CONTEXT_BUDGET")
        return messages()

    async def act(self, state):
        called = False
        try:
            messages = self.messages(state)
            called = True
            action = parse_evidence_action(await self.model.generate(messages))
            return {
                "action": action,
                "model_calls": state["model_calls"] + 1,
                "actions": state["actions"] + [deepcopy(action)],
                "feedback": None,
                "trace": self.event(state, "model_action", action=action["action"]),
            }
        except GraphModelError as error:
            return {
                "action": None,
                "model_calls": state["model_calls"] + int(called),
                "feedback": {"code": error.code, "retryable": error.retryable},
                "status": "running" if error.retryable else "inconclusive",
                "reason": error.code,
                "trace": self.event(state, "model_error", code=error.code),
            }

    def reject(self, state, code):
        return {
            "action": None,
            "feedback": {"code": code},
            "trace": self.event(state, "action_rejected", code=code),
        }

    def query(self, state):
        if not self.policy.tools_enabled:
            return self.reject(state, "EVIDENCE_TOOLS_DISABLED")
        if state["query_calls"] >= self.policy.max_queries:
            return self.reject(state, "EVIDENCE_QUERY_BUDGET")
        action = state["action"]
        observation = self.observation(state, action["sql"], action["hypothesis"])
        return {
            "action": None,
            "query_calls": state["query_calls"] + 1,
            "observations": state["observations"] + [observation],
            "feedback": {"query_observation_id": observation["id"]},
            "trace": self.event(state, "query", observation_id=observation["id"]),
        }

    def citations(self, state, action):
        observations = {item["id"]: item for item in state["observations"]}
        if any(identifier not in observations for identifier in action["evidence_ids"]):
            return "EVIDENCE_UNKNOWN_OBSERVATION"
        # A SQL error is an observation and can support a schema/syntax repair.
        # Merely passing this check never proves the model's interpretation.
        if any(
            observations[identifier]["candidate_version"] != state["candidate"]["version"]
            for identifier in action["evidence_ids"]
        ):
            return "EVIDENCE_STALE_CANDIDATE_VERSION"
        return None

    def revise(self, state):
        action = state["action"]
        if state["revision_attempts"] >= self.policy.max_revisions:
            return self.reject(state, "EVIDENCE_REVISION_BUDGET")
        code = self.citations(state, action)
        if code:
            return self.reject(state, code)
        if not any(
            action["requirement_quote"] in text
            for text in (self.task.question, self.task.definitions)
        ):
            return self.reject(state, "EVIDENCE_QUOTE_NOT_IN_REQUIREMENT")
        if action["sql"].strip() == state["candidate"]["sql"].strip():
            return self.reject(state, "EVIDENCE_UNCHANGED_REVISION")
        next_version = len(state["versions"]) + 1
        observation = self.observation(state, action["sql"], "proposed_revision")
        observation["candidate_version"] = next_version
        accepted = observation["status"] == "ok" and observation["complete"]
        version = {
            "version": next_version,
            "sql": action["sql"],
            "sql_digest": digest_json(action["sql"]),
            "previous_version": state["candidate"]["version"],
            "observation_id": observation["id"],
            "accepted": accepted,
            "origin": "model_revision",
            "reason": action["reason"],
            "requirement_quote": action["requirement_quote"],
            "evidence_ids": list(action["evidence_ids"]),
        }
        result = {
            "action": None,
            "observations": state["observations"] + [observation],
            "versions": state["versions"] + [version],
            "revision_attempts": state["revision_attempts"] + 1,
            "feedback": {
                "revision_execution_id": observation["id"],
                "adopted": accepted,
                "scope": "execution only; business correctness remains unproven",
            },
            "trace": self.event(
                state, "revision_execution", observation_id=observation["id"], adopted=accepted
            ),
        }
        if accepted:
            result.update(
                candidate={"sql": action["sql"], "version": next_version},
                current_observation_id=observation["id"],
            )
        return result

    def finish(self, state):
        action = state["action"]
        code = self.citations(state, action)
        if code:
            return self.reject(state, code)
        if state["current_observation_id"] not in action["evidence_ids"]:
            return self.reject(state, "EVIDENCE_FINISH_MUST_CITE_CURRENT_EXECUTION")
        current = next(
            item for item in state["observations"] if item["id"] == state["current_observation_id"]
        )
        if current["status"] != "ok" or not current["complete"]:
            return self.reject(state, "EVIDENCE_CURRENT_EXECUTION_INCOMPLETE")
        return {
            "action": None,
            "status": "reviewed",
            "reason": action["reason"],
            "publication": {
                "sql": state["candidate"]["sql"],
                "rows": deepcopy(current["rows"]),
                "reason": action["reason"],
                "assurance": "executed_and_model_reviewed_not_semantically_verified",
                "observation_id": current["id"],
                "evidence_ids": list(action["evidence_ids"]),
            },
            "trace": self.event(state, "finish", observation_id=current["id"]),
        }

    def clarify(self, state):
        return {
            "action": None,
            "status": "needs_clarification",
            "reason": state["action"]["reason"],
            "trace": self.event(state, "clarify"),
        }

    def next_action(self, state):
        if state["status"] != "running" or state["model_calls"] >= self.policy.max_actions:
            return "finalize"
        return "act"

    def route_action(self, state):
        if state["status"] != "running":
            return "finalize"
        return state["action"]["action"] if state["action"] else self.next_action(state)

    def finalize(self, state):
        status = "inconclusive" if state["status"] == "running" else state["status"]
        reason = state["reason"]
        if state["status"] == "running":
            reason = "EVIDENCE_MODEL_BUDGET_EXHAUSTED"
        return {
            "status": status,
            "reason": reason,
            "publication": state["publication"]
            if status == "reviewed"
            else {"sql": None, "rows": None, "reason": reason},
        }


async def run_evidence_graph(
    task: DiscoveryTask,
    source,
    initial_sql: str,
    model,
    *,
    policy: EvidencePolicy | None = None,
) -> dict:
    """Investigate one bound candidate without reading a gold answer or writer history.

    No API client is created implicitly. Callers own the supplied model and its
    budget. The returned trace preserves every observation and proposed SQL,
    including failed revisions. This prototype does not replace old graph
    checkpoints or support resuming a partially paid run.
    """
    policy = policy or EvidencePolicy()
    if not isinstance(task, DiscoveryTask) or not isinstance(policy, EvidencePolicy):
        raise ValueError("EVIDENCE_INPUT_TYPE")
    if not isinstance(initial_sql, str) or not 1 <= len(initial_sql) <= 12_000:
        raise ValueError("EVIDENCE_INITIAL_SQL")
    tools = SQLTools(source)
    public = task.public_input()
    if public["schema"] != tools.metadata()["schema"]:
        raise ValueError("EVIDENCE_TASK_SOURCE_SCHEMA_MISMATCH")
    nodes = _EvidenceNodes(task, tools, initial_sql, model, policy)
    builder = StateGraph(EvidenceState)
    for name in ("start", "act", "query", "revise", "finish", "clarify", "finalize"):
        builder.add_node(name, getattr(nodes, name))
    builder.add_edge(START, "start")
    builder.add_edge("start", "act")
    builder.add_conditional_edges("act", nodes.route_action)
    for name in ("query", "revise", "finish"):
        builder.add_conditional_edges(name, nodes.next_action)
    builder.add_edge("clarify", "finalize")
    builder.add_edge("finalize", END)
    graph = builder.compile(name="sql_evidence_investigation")
    state = await graph.ainvoke(
        {
            "candidate": {"sql": initial_sql, "version": 1},
            "current_observation_id": "",
            "observations": [],
            "versions": [],
            "trace": [],
            "actions": [],
            "action": None,
            "feedback": None,
            "model_calls": 0,
            "query_calls": 0,
            "revision_attempts": 0,
            "status": "running",
            "reason": "",
            "publication": {},
        },
        {"recursion_limit": 3 * policy.max_actions + 8},
    )
    budget = getattr(model, "budget", None)
    return {
        **dict(state),
        "method": "evidence_investigation_prototype",
        "task_id": task.task_id,
        "initial_sql": initial_sql,
        "policy": asdict(policy),
        "source": tools.metadata(),
        "budget": budget.snapshot() if budget is not None else None,
    }
