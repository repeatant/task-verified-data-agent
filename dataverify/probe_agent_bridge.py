"""Connect probe diagnostics to the pinned, real DB-GPT native Agent loop.

No model call or credential lookup occurs on import. This adapter changes only
the Action, correctness hook and final publication boundary; thinking, memory,
generate_reply and retry scheduling remain the upstream implementation.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal
from uuid import uuid4

from dbgpt._private.pydantic import Field, PrivateAttr
from dbgpt.agent import AgentContext, AgentMessage, LLMConfig, UserProxyAgent
from dbgpt.agent.core.action.base import ActionOutput
from dbgpt.agent.expand.actions.chart_action import ChartAction, SqlInput
from dbgpt.agent.expand.data_scientist_agent import DataScientistAgent
from dbgpt.agent.resource.database import DBResource
from dbgpt.core import LLMClient

from .dbgpt_bridge import ScriptedLLMClient, UPSTREAM_COMMIT, verify_upstream_files
from .probe_guard import ProbeGuard
from .probe_capabilities import policy_features, sql_capabilities, validate_policy
from .probe_runtime import ProbeData, ProbeTask, TABLE_SCHEMA


def probe_data_sha256(data: ProbeData) -> str:
    payload = json.dumps(asdict(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ProbeResource(DBResource):
    """Schema-only native resource. All execution must go through ProbeGuard.

    Exposing a DBResource keeps the native DataScientistAgent protocol intact,
    but its raw query route is deliberately unavailable, including to native
    ChartAction should it ever be called accidentally.
    """

    def __init__(self, task: ProbeTask, *, feedback_policy: str = "capability_v2"):
        self.task = task
        self.feedback_policy = validate_policy(feedback_policy)
        self.raw_query_attempts = 0
        name = "probe-memory-" + uuid4().hex
        super().__init__(name=name, db_name=name, db_type="sqlite", dialect="sqlite")

    def get_schema_link(self, db: str, question: str | None = None):
        if db != self._db_name:
            raise ValueError("Only the bound probe task schema is available")
        payload = {
            "confirmed_task": self.task.to_dict(),
            "tables": {name: list(columns) for name, columns in TABLE_SCHEMA.items()},
            "types": {"money": "bounded integer minor units; refunds may be negative",
                      "settled_at": "canonical timezone-free ISO seconds YYYY-MM-DDTHH:MM:SS",
                      "status": ["settled", "pending", "cancelled"]},
            "relation": "orders.order_id = order_lines.order_id; child FK is valid",
            "supported_sql": [
                "Exactly one SELECT, one SUM(order_lines.amount); COALESCE(SUM(...),0) is permitted.",
                "Read order_lines directly or use one registered INNER/LEFT JOIN with orders.",
                "AND of direct column/literal comparisons, IS NULL, and canonical timestamp bounds.",
                "For grouped tasks output the order ID or channel key first, and the SUM second; one GROUP BY key.",
                "COALESCE(orders.channel, confirmed other_label) is permitted; SELECT and GROUP BY must use the same expression.",
                "No CTE, subquery, window, UNION, DISTINCT amount, arithmetic, LIMIT, ORDER BY or arbitrary function.",
            ],
            "execution": "All actions are verified by the bound task guard; raw resource queries are disabled.",
        }
        if policy_features(self.feedback_policy)["capability_description"]:
            payload["execution_contract"] = sql_capabilities()
        return json.dumps(payload, ensure_ascii=False)

    def _sync_query(self, db: str, sql: str):
        self.raw_query_attempts += 1
        raise RuntimeError("RAW_PROBE_QUERY_DISABLED_USE_GUARDED_ACTION")

    def close(self):
        self._executor.shutdown(wait=True)


class ProbeSqlInput(SqlInput):
    task_id: str = Field(
        ..., min_length=1, max_length=80,
        description="Repeat the caller-confirmed task_id exactly; never change it during repair.",
    )


def probe_publication(guard: ProbeGuard, upstream_success: bool) -> dict:
    """Publish only guard-owned data when BOTH native loop and guard succeeded."""
    raw = guard.publish()
    complete = (
        isinstance(raw, dict) and raw.get("status") == "verified"
        and isinstance(raw.get("rows"), list)
        and type(raw.get("verified_sql")) is str and bool(raw["verified_sql"].strip())
    )
    accepted = bool(upstream_success and complete)
    code = raw.get("error_code") if isinstance(raw, dict) else "INVALID_PUBLICATION_ENVELOPE"
    if not accepted and not code:
        code = "UPSTREAM_DID_NOT_FINISH" if not upstream_success else "INVALID_PUBLICATION_ENVELOPE"
    return {
        "task_id": guard.task.task_id,
        "status": "verified" if accepted else "unverified",
        "rows": deepcopy(raw["rows"]) if accepted else None,
        "verified_sql": raw["verified_sql"] if accepted else None,
        "error_code": None if accepted else code,
        "scope": raw.get("scope", "registered task properties only; not a complete correctness proof")
        if isinstance(raw, dict) else "registered task properties only",
    }


class ProbeChartAction(ChartAction):
    """Native Action extension that verifies before creating any output view."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.guard: ProbeGuard | None = None
        self.audit: list[dict] = []

    @property
    def out_model_type(self):
        return ProbeSqlInput

    async def run(self, ai_message: str, **kwargs):
        if self.guard is None:
            raise RuntimeError("A fresh ProbeGuard must be bound")
        try:
            candidate = self._input_convert(ai_message, ProbeSqlInput)
        except Exception:
            feedback = self.guard.invalidate(
                "INVALID_ACTION_FORMAT", "Return one JSON object with display_type, sql, task_id and thought.",
            )
        else:
            feedback = await asyncio.to_thread(self.guard.execute, candidate.sql, candidate.task_id)
        self.audit.append(deepcopy(self.guard.last_result))
        publication = probe_publication(self.guard, True)
        accepted = publication["status"] == "verified"
        content = json.dumps(feedback, ensure_ascii=False)
        return ActionOutput(
            is_exe_success=accepted,
            content=content, observations=content,
            view=json.dumps(publication, ensure_ascii=False) if accepted else None,
            resource_type="database", resource_value="caller-confirmed-immutable-probe-data",
            have_retry=bool(feedback.get("retryable", True)),
        )


class ProbeDataScientistAgent(DataScientistAgent):
    """Inherited generation/retry loop, task-bound execution/correctness hooks."""

    _probe_guard: ProbeGuard = PrivateAttr()

    def __init__(self, *, guard: ProbeGuard, **kwargs):
        super().__init__(**kwargs)
        self._probe_guard = guard
        self._init_actions([ProbeChartAction])
        self.actions[0].guard = guard

    async def correctness_check(self, message: AgentMessage):
        if self._probe_guard.publish().get("status") == "verified":
            return True, None
        # Only mode-filtered public feedback enters the native retry context.
        return False, message.action_report.content if message.action_report else "NO_ACTION"

    async def adjust_final_message(self, is_success: bool, reply_message: AgentMessage):
        publication = probe_publication(self._probe_guard, is_success)
        reply_message.success = publication["status"] == "verified"
        reply_message.content = json.dumps(publication, ensure_ascii=False)
        if reply_message.action_report:
            reply_message.action_report.content = reply_message.content
            reply_message.action_report.view = reply_message.content if reply_message.success else None
            reply_message.action_report.is_exe_success = reply_message.success


def probe_sql_candidate(sql: str, task_id: str) -> dict[str, str]:
    return {
        "display_type": "response_table", "sql": sql, "task_id": task_id,
        "thought": "Controlled integration candidate; not a natural model prediction.",
    }


def _safe_error(error: BaseException) -> dict[str, str]:
    # Upstream can embed errors in messages. Never persist a raw exception body,
    # path, response payload, request header or arbitrary provider error string.
    text = str(error)
    code = text if re.fullmatch(r"[A-Z][A-Z0-9_]{0,159}", text) else type(error).__name__
    return {"type": type(error).__name__, "code": code}


async def run_probe_native_case(
    task: ProbeTask, data: ProbeData, *,
    mode: Literal["full", "compact", "code", "adaptive"] = "compact", seed: int = 17,
    max_attempts: int = 2, output_dir: Path,
    llm_client: LLMClient | None = None, responses: list[dict | str] | None = None,
    feedback_policy: str = "capability_v2",
) -> dict[str, Any]:
    """Run the real pinned loop with one immutable task and fresh guard.

    Scripted clients are explicitly offline. Controlled first candidates may be
    injected by the caller's labeled client; only its subsequent real HTTP calls
    may be reported as model repairs. Only ``publication`` is application output.
    """
    if not isinstance(task, ProbeTask) or not isinstance(data, ProbeData):
        raise ValueError("Expected immutable confirmed task and probe data")
    if mode not in {"full", "compact", "code", "adaptive"} or type(seed) is not int:
        raise ValueError("Invalid feedback mode or seed")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be in 1..3")
    if (llm_client is None) == (responses is None):
        raise ValueError("Choose exactly one explicit client or offline response script")
    validate_policy(feedback_policy)
    upstream_check = verify_upstream_files()
    output_dir = Path(output_dir).resolve()
    run_output = output_dir / ("probe-native-" + uuid4().hex)
    run_output.mkdir(parents=True, exist_ok=False)
    source_before = probe_data_sha256(data)
    client = llm_client if llm_client is not None else ScriptedLLMClient(responses)
    resource = ProbeResource(task, feedback_policy=feedback_policy)
    guard = ProbeGuard(task, data, mode=mode, seed=seed, feedback_policy=feedback_policy)
    context = AgentContext(
        conv_id="probe-native-" + uuid4().hex, language="en", output_dir=str(run_output),
        temperature=0, verbose=False,
    )
    agent = ProbeDataScientistAgent(guard=guard, max_retry_count=max_attempts, max_timeout=120, stream_out=False)
    sender = UserProxyAgent()
    events: list[dict] = []
    reply = None
    operational_error = None
    cleanup_errors = []

    async def capture_event(event_type, payload):
        events.append({"event_type": event_type, "payload": deepcopy(payload)})

    try:
        agent = await agent.bind(context).bind(LLMConfig(llm_client=client)).bind(resource).build()
        sender = await sender.bind(context).build()
        question = (
            task.question + "\nCaller-confirmed task_id=" + task.task_id
            + ". Preserve all confirmed task requirements during repairs. "
            "Return one action JSON; the guard executes SQL and owns the result."
        )
        reply = await agent.generate_reply(
            AgentMessage(content=question, current_goal=question), sender=sender,
            stream_callback=capture_event,
        )
    except Exception as error:
        operational_error = _safe_error(error)
    finally:
        for owner in (resource, agent, sender):
            try:
                if isinstance(owner, ProbeResource):
                    owner.close()
                else:
                    owner.executor.shutdown(wait=True)
            except Exception as error:
                cleanup_errors.append(_safe_error(error))
        if callable(getattr(client, "aclose", None)):
            try:
                await client.aclose()
            except Exception as error:
                cleanup_errors.append(_safe_error(error))
    source_after = probe_data_sha256(data)
    upstream_success = bool(reply is not None and reply.success)
    publication = probe_publication(guard, upstream_success and operational_error is None)
    if source_before != source_after:
        publication.update(status="unverified", rows=None, verified_sql=None, error_code="SOURCE_DATA_CHANGED")
    return {
        "mode": mode, "seed": seed, "feedback_policy": feedback_policy,
        "feedback_features": policy_features(feedback_policy),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file_check": upstream_check,
        "native_loop": "DataScientistAgent.generate_reply (inherited)",
        "client_kind": getattr(client, "client_kind", "deterministic_offline_script_not_real_model"),
        "upstream_success": upstream_success, "publication": publication,
        "native_reply": reply.to_dict() if reply is not None else None,
        "llm_requests": deepcopy(getattr(client, "requests", [])),
        "llm_calls": len(getattr(client, "requests", [])),
        "http_events": deepcopy(getattr(client, "http_events", [])),
        "model_usage": deepcopy(getattr(client, "usage", [])),
        "turn_origins": deepcopy(getattr(client, "turn_origins", [])),
        "actions": deepcopy(agent.actions[0].audit),
        "guard_audit": deepcopy(guard.audit),
        "raw_resource_queries": resource.raw_query_attempts,
        "stream_events": events,
        "source_sha256_before": source_before, "source_sha256_after": source_after,
        "source_unchanged": source_before == source_after,
        "operational_error": operational_error, "cleanup_errors": cleanup_errors,
    }
