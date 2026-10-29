"""LangGraph orchestration: plan -> tools -> verification feedback -> bounded repair."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .checker import CheckedQueryExecutor
from .contracts import describe_contracts
from .database import database_sha256
from .providers import (
    SYSTEM_PROMPT,
    Action,
    OfflineScriptedPlanner,
    Planner,
    ProviderError,
    QwenPlanner,
)


class RunState(TypedDict):
    messages: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    action: Action | None
    steps: int
    failed_queries: int
    accepted: dict[str, Any] | None
    stopped: bool
    stop_reason: str


class AgentRunner:
    def __init__(
        self, db_path: Path, trace_dir: Path | None = None, *, planner: Planner | None = None
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.trace_dir = Path(trace_dir).resolve() if trace_dir is not None else None
        self.injected_planner = planner

    def run(
        self,
        question: str = "统计订单总收入",
        case: str = "join_fanout",
        max_repairs: int = 1,
        provider: str = "offline",
    ) -> dict[str, Any]:
        if not 0 <= max_repairs <= 2:
            raise ValueError("max_repairs must be between 0 and 2")
        if not question.strip() or len(question) > 2000:
            raise ValueError("question must contain 1 to 2000 characters")
        if provider not in {"offline", "qwen"}:
            raise ValueError("provider must be offline or qwen")
        planner = self.injected_planner or (
            OfflineScriptedPlanner(case) if provider == "offline" else QwenPlanner()
        )
        usage_start = len(getattr(planner, "usage", []))
        executor = CheckedQueryExecutor(self.db_path)
        run_id = uuid4().hex
        source_hash = database_sha256(self.db_path)
        max_steps = 8

        def plan(state: RunState) -> dict[str, Any]:
            if state["steps"] >= max_steps:
                return {"stopped": True, "stop_reason": "tool_step_limit"}
            try:
                action = planner.choose(state["messages"])
            except ProviderError as exc:
                return {
                    "stopped": True,
                    "stop_reason": str(exc),
                    "trace": state["trace"] + [{"event": "provider_error", "detail": str(exc)}],
                }
            if action.kind == "final":
                return {
                    "stopped": True,
                    "stop_reason": "verified_result"
                    if state["accepted"]
                    else "no_verified_evidence",
                }
            if action.kind != "tool":
                return {"stopped": True, "stop_reason": "invalid_planner_action"}
            return {"action": action, "messages": state["messages"] + [action.assistant_message()]}

        def execute(state: RunState) -> dict[str, Any]:
            action = state["action"]
            assert action is not None
            failed = state["failed_queries"]
            accepted = state["accepted"]
            stopped = False
            stop_reason = ""
            if action.tool_name == "inspect_schema" and not action.arguments:
                result = {
                    "tables": {
                        "orders": ["order_id", "total"],
                        "order_lines": ["line_id", "order_id", "amount"],
                    },
                    "contracts": describe_contracts(),
                    "supported_sql": "one unfiltered SUM over a registered measure; at most one registered INNER JOIN",
                }
            elif action.tool_name == "query_metric":
                # Any new query invalidates prior accepted evidence until checked again.
                accepted = None
                sql = action.arguments.get("sql")
                metric_id = action.arguments.get("metric_id")
                if (
                    set(action.arguments) != {"sql", "metric_id"}
                    or not isinstance(sql, str)
                    or len(sql) > 4096
                    or not isinstance(metric_id, str)
                ):
                    result = {
                        "verification_status": "unsupported",
                        "accepted_value": None,
                        "error_code": "INVALID_ARGUMENTS",
                        "feedback": "Provide exactly sql and metric_id within the tool schema.",
                    }
                else:
                    result = executor.execute(sql, metric_id)
                if (
                    result.get("verification_status") == "verified"
                    and result.get("accepted_value") is not None
                ):
                    accepted = {**result, "evidence_id": f"{run_id}:step-{state['steps'] + 1}"}
                    result["evidence_id"] = accepted["evidence_id"]
                    stopped = True
                    stop_reason = "verified_result"
                else:
                    failed += 1
                    if failed > max_repairs:
                        stopped = True
                        stop_reason = "repair_limit"
            else:
                result = {
                    "error_code": "UNKNOWN_TOOL_OR_ARGUMENTS",
                    "feedback": "Only registered tools with valid arguments are allowed.",
                }
            event = {
                "step": state["steps"] + 1,
                "event": "tool_result",
                "tool": action.tool_name,
                "arguments": action.arguments,
                "result": result,
            }
            message = {
                "role": "tool",
                "name": action.tool_name,
                "tool_call_id": action.call_id,
                "content": json.dumps(result, ensure_ascii=False),
            }
            return {
                "messages": state["messages"] + [message],
                "trace": state["trace"] + [event],
                "steps": state["steps"] + 1,
                "failed_queries": failed,
                "accepted": accepted,
                "stopped": stopped,
                "stop_reason": stop_reason,
            }

        def route(state: RunState) -> str:
            return "finish" if state["stopped"] else "execute"

        def after_tool(state: RunState) -> str:
            return "finish" if state["stopped"] else "plan"

        builder = StateGraph(RunState)
        builder.add_node("plan", plan)
        builder.add_node("execute", execute)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges("plan", route, {"finish": END, "execute": "execute"})
        builder.add_conditional_edges("execute", after_tool, {"finish": END, "plan": "plan"})
        graph = builder.compile()
        state = graph.invoke(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "trace": [],
                "action": None,
                "steps": 0,
                "failed_queries": 0,
                "accepted": None,
                "stopped": False,
                "stop_reason": "",
            },
            {"recursion_limit": 30},
        )
        if source_hash != database_sha256(self.db_path):
            state["accepted"] = None
            state["stop_reason"] = "source_changed_during_run"
        accepted = state["accepted"]
        verified = accepted is not None
        value = accepted["accepted_value"] if verified else None
        request_usage = getattr(planner, "usage", [])[usage_start:]
        token_usage = {
            field: sum(
                entry[field]
                for entry in request_usage
                if isinstance(entry, dict) and type(entry.get(field)) is int
            )
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        responses_with_usage = sum(
            isinstance(entry, dict) and any(type(entry.get(field)) is int for field in token_usage)
            for entry in request_usage
        )
        result = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": planner.name,
            "real_model_used": planner.real_model,
            "model": getattr(planner, "model", None),
            "model_usage": {"responses_with_usage": responses_with_usage, **token_usage},
            "question": question,
            "case": case if provider == "offline" else None,
            "status": "verified" if verified else "unverified",
            "value": value,
            "answer": (
                f"已核验指标 {accepted['metric_id']} = {value:g}；证据 {accepted['evidence_id']}。"
                if verified
                else "当前没有通过核验的统计结果，不能据此给出确定结论。"
            ),
            "stop_reason": state["stop_reason"],
            "failed_queries": state["failed_queries"],
            "steps": state["steps"],
            "source_sha256": source_hash,
            "source_fingerprint_algorithm": "sha256(main-file + visible-wal)",
            "verification_scope": "registered metric; one unfiltered SUM; zero or one registered INNER JOIN; not arbitrary SQL or business intent validation",
            "trace": state["trace"],
            "trace_path": None,
        }
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            target = self.trace_dir / f"{run_id}.json"
            result["trace_path"] = str(target)
            with target.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
        return result
