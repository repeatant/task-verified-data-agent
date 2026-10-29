"""A checkpointed LangGraph executor over the shared task-verification core.

One graph owns one bounded repair loop. No DB-GPT loop is nested inside it.
Checkpoints contain JSON-compatible state, never clients, credentials or a live
guard. The guard codec preserves witnesses but deliberately drops acceptance;
publication always re-executes a candidate against the current bound source.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from .graph_checkpoint import GraphResumeError, exclusive_run, validate_budget_continuation
from .graph_guard import dump_guard, restore_guard
from .graph_model import GraphModel, GraphModelError, parse_graph_action
from .probe_capabilities import policy_features, sql_capabilities, validate_policy
from .probe_guard import ProbeGuard, source_digest
from .probe_runtime import ProbeData, ProbeTask
from .reporting import digest_json

_FORMAT = "probe-langgraph-v1"
_RUNTIME_FILES = (
    "probe_langgraph.py",
    "graph_checkpoint.py",
    "graph_guard.py",
    "graph_model.py",
    "probe_guard.py",
    "probe_runtime.py",
    "probe_testing.py",
    "probe_capabilities.py",
    "feedback_strategy.py",
    "qwen_config.py",
    "probe_cost_budget.py",
    "experiment_budget.py",
    "reporting.py",
)


class GraphState(TypedDict):
    """Replace state fields explicitly; append lists only at their owner node."""

    binding: dict[str, Any]
    messages: list[dict[str, str]]
    attempts: int
    call_pending: bool
    candidate: dict[str, str] | None
    action_error: str | None
    operational_error: str | None
    guard_state: dict[str, Any]
    feedback: dict[str, Any]
    actions: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    budget: dict[str, Any] | None
    publication: dict[str, Any]
    publication_verification: dict[str, Any] | None


def runtime_fingerprint() -> str:
    """Refuse an old thread if verifier code or its execution dependencies change."""
    root = Path(__file__).parent
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _RUNTIME_FILES
    }
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "langgraph",
            "langgraph-checkpoint",
            "langgraph-checkpoint-sqlite",
            "langchain-core",
            "aiosqlite",
            "sqlglot",
            "hypothesis",
        )
    }
    return digest_json({"files": files, "versions": versions})


def _model_budget(model: GraphModel) -> dict | None:
    budget = getattr(model, "budget", None)
    return budget.snapshot() if budget is not None else None


def _unverified(task: ProbeTask, code: str) -> dict:
    return {
        "status": "unverified",
        "task_id": task.task_id,
        "rows": None,
        "verified_sql": None,
        "error_code": code,
        "scope": "registered necessary properties only; not a semantic proof",
    }


class _GraphNodes:
    """Keep node behavior beside its routing; collaborators stay outside state."""

    def __init__(self, task: ProbeTask, data: ProbeData, model: GraphModel, options: dict):
        self.task, self.data, self.model, self.options = task, data, model, options

    def guard(self, state: GraphState) -> ProbeGuard:
        return restore_guard(state["guard_state"], self.task, self.data, **self.options_for_guard())

    def options_for_guard(self) -> dict:
        return {key: self.options[key] for key in ("mode", "seed", "feedback_policy")}

    def prepare_call(self, state: GraphState) -> dict:
        # This intent is checkpointed with durability='sync' before generate.
        # If generate dies before its result is committed, resume must stop.
        return {
            "call_pending": True,
            "attempts": state["attempts"] + 1,
            "candidate": None,
            "action_error": None,
            "publication": _unverified(self.task, "MODEL_CALL_PENDING"),
            "trace": state["trace"] + [{"node": "prepare_call", "attempt": state["attempts"] + 1}],
        }

    async def generate(self, state: GraphState) -> dict:
        candidate, action_error, operational_error = None, None, None
        try:
            candidate = parse_graph_action(await self.model.generate(deepcopy(state["messages"])))
        except GraphModelError as exc:
            if exc.retryable:
                action_error = exc.code
            else:
                operational_error = exc.code
        return {
            "candidate": candidate,
            "action_error": action_error,
            "operational_error": operational_error,
            "call_pending": False,
            "budget": _model_budget(self.model),
            "trace": state["trace"] + [{"node": "generate", "attempt": state["attempts"]}],
        }

    def verify(self, state: GraphState) -> dict:
        guard = self.guard(state)
        candidate = state["candidate"]
        if state["operational_error"]:
            feedback = guard.invalidate("MODEL_OPERATION_FAILED", "Model request stopped safely.")
        elif state["action_error"] or candidate is None:
            feedback = guard.invalidate()
        else:
            feedback = guard.execute(candidate["sql"], candidate["task_id"])
        return {
            "feedback": feedback,
            "guard_state": dump_guard(guard),
            "actions": state["actions"] + [guard.last_result],
            "trace": state["trace"]
            + [{"node": "verify", "attempt": state["attempts"], "status": feedback["status"]}],
        }

    def after_verify(self, state: GraphState) -> str:
        if (
            state["operational_error"]
            or state["feedback"].get("status") == "verified"
            or state["attempts"] >= self.options["max_attempts"]
        ):
            return "publish"
        return "feedback"

    def feedback(self, state: GraphState) -> dict:
        action = state["candidate"] or {"error": state["action_error"]}
        return {
            "messages": state["messages"]
            + [
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
                {"role": "user", "content": json.dumps(state["feedback"], ensure_ascii=False)},
            ],
            "trace": state["trace"] + [{"node": "feedback", "attempt": state["attempts"]}],
        }

    def publish(self, state: GraphState) -> dict:
        publication = _unverified(self.task, state["operational_error"] or "NO_VERIFIED_ACTION")
        candidate = state["candidate"]
        verification = None
        if not state["operational_error"] and state["feedback"].get("status") == "verified":
            # A checkpoint is evidence, not publication authority. Re-check the
            # candidate and every retained witness instead of trusting rows.
            guard = self.guard(state)
            guard.execute(candidate["sql"], candidate["task_id"])
            publication, verification = guard.publish(), guard.last_result
        elif state["feedback"].get("error_code") and not state["operational_error"]:
            publication["error_code"] = state["feedback"]["error_code"]
        return {
            "publication": publication,
            "trace": state["trace"] + [{"node": "publish", "status": publication["status"]}],
            "publication_verification": verification,
        }

    def compile(self, checkpointer: AsyncSqliteSaver):
        graph = StateGraph(GraphState)
        graph.add_node("prepare_call", self.prepare_call)
        graph.add_node("generate", self.generate)
        graph.add_node("verify", self.verify)
        graph.add_node("feedback", self.feedback)
        graph.add_node("publish", self.publish)
        graph.add_edge(START, "prepare_call")
        graph.add_edge("prepare_call", "generate")
        graph.add_edge("generate", "verify")
        graph.add_conditional_edges("verify", self.after_verify, ["publish", "feedback"])
        graph.add_edge("feedback", "prepare_call")
        graph.add_edge("publish", END)
        return graph.compile(checkpointer=checkpointer, name="task_verified_sql")


def _initial_state(
    task: ProbeTask, data: ProbeData, model: GraphModel, options: dict
) -> GraphState:
    public = task.public_input()
    public["supported_sql"] = [
        "One SELECT with SUM(order_lines.amount); COALESCE(SUM(...),0) is permitted.",
        "Read order_lines directly or use one INNER/LEFT JOIN on registered order_id.",
        "AND of direct column/literal comparisons and IS NULL; canonical timestamp bounds.",
        "Grouped tasks: order/channel key first, numeric SUM second; one GROUP BY key.",
        "COALESCE(orders.channel, confirmed other_label) is permitted; grouping must agree.",
        "No CTE, subquery, window, UNION, DISTINCT, arithmetic, ORDER BY, LIMIT or arbitrary functions.",
    ]
    public["feedback_features"] = policy_features(options["feedback_policy"])
    if public["feedback_features"]["capability_description"]:
        public["execution_contract"] = sql_capabilities()
    guard = ProbeGuard(task, data, **{k: options[k] for k in ("mode", "seed", "feedback_policy")})
    return {
        "binding": {
            "format": _FORMAT,
            "source": source_digest(task, data),
            "runtime": runtime_fingerprint(),
            "options": options,
            "client_kind": model.client_kind,
            "model": getattr(model, "model_name", None),
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "Produce one JSON action with sql, task_id and optional thought. "
                    "Use SQLite within supported_sql. Keep caller-confirmed requirements. "
                    "No prose, Markdown, tools or invented numeric answers. After rejected "
                    "verification, repair SQL using the supplied feedback."
                ),
            },
            {"role": "user", "content": json.dumps(public, ensure_ascii=False)},
        ],
        "attempts": 0,
        "call_pending": False,
        "candidate": None,
        "action_error": None,
        "operational_error": None,
        "guard_state": dump_guard(guard),
        "feedback": {},
        "actions": [],
        "trace": [],
        "budget": _model_budget(model),
        "publication": _unverified(task, "NO_VERIFIED_ACTION"),
        "publication_verification": None,
    }


async def run_graph_case(
    task: ProbeTask,
    data: ProbeData,
    model: GraphModel,
    *,
    checkpoint_path: Path,
    thread_id: str,
    mode: str = "adaptive",
    seed: int = 17,
    max_attempts: int = 3,
    feedback_policy: str = "capability_v2",
    resume: bool = False,
    pause_after: str | None = None,
) -> dict[str, Any]:
    """Run or resume one local thread; the caller owns model/budget lifetimes.

    ``pause_after`` is a development seam for safe checkpoint demonstrations.
    Only verify/feedback boundaries are exposed. Unknown in-node HTTP outcomes
    are refused, even if the provider may have completed the request.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", thread_id):
        raise ValueError("Use a short alphanumeric thread_id")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be in 1..3")
    if pause_after not in (None, "verify", "feedback"):
        raise ValueError("pause_after must be verify or feedback")
    options = dict(
        mode=mode,
        seed=seed,
        max_attempts=max_attempts,
        feedback_policy=validate_policy(feedback_policy),
    )
    initial = _initial_state(task, data, model, options)
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    nodes = _GraphNodes(task, data, model, options)

    with exclusive_run(checkpoint_path):
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            graph = nodes.compile(saver)
            snapshot = await graph.aget_state(config)
            exists = bool(snapshot.values)
            if exists != resume:
                raise GraphResumeError("THREAD_NOT_FOUND" if resume else "THREAD_ALREADY_EXISTS")
            if resume:
                if snapshot.values.get("binding") != initial["binding"]:
                    raise GraphResumeError("TASK_SOURCE_RUNTIME_OR_CONFIG_CHANGED")
                if snapshot.values.get("call_pending"):
                    raise GraphResumeError("MODEL_CALL_OUTCOME_UNKNOWN")
                validate_budget_continuation(snapshot.values.get("budget"), initial["budget"])
            if resume and not snapshot.next:
                state = {**snapshot.values, **nodes.publish(snapshot.values)}
            else:
                state = await graph.ainvoke(
                    None if resume else initial,
                    config,
                    durability="sync",
                    interrupt_after=[pause_after] if pause_after else None,
                )
            after = await graph.aget_state(config)
    paused = bool(after.next)
    return {
        "runtime": "langgraph",
        "thread_id": thread_id,
        "status": "paused" if paused else "completed",
        "next_nodes": list(after.next),
        "model_attempts": state["attempts"],
        "actions": state["actions"],
        "publication": _unverified(task, "RUN_PAUSED") if paused else state["publication"],
        "publication_verification": state.get("publication_verification"),
        "trace": state["trace"],
        "operational_error": state["operational_error"],
        "budget": _model_budget(model),
        "source_unchanged": source_digest(task, data) == initial["binding"]["source"],
        "binding": initial["binding"],
    }
