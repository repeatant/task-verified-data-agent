"""Optional adapter to the *real*, pinned DB-GPT DataScientistAgent loop.

Import only in the dedicated DB-GPT environment. We replace its Action and
correctness hook, never generate_reply(), thinking(), verify() or memory/retry.
The scripted client below tests integration plumbing, not LLM reasoning quality.
Each run owns a fresh task guard, agent, LLM client and conversation.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Literal
from uuid import uuid4

from dbgpt._private.pydantic import Field, PrivateAttr
from dbgpt.agent import AgentContext, AgentMessage, LLMConfig, UserProxyAgent
from dbgpt.agent.core.action.base import ActionOutput
from dbgpt.agent.expand.actions.chart_action import ChartAction, SqlInput
from dbgpt.agent.expand.data_scientist_agent import DataScientistAgent
from dbgpt.agent.resource.database import DBResource
from dbgpt.core import LLMClient, ModelMetadata, ModelOutput

from .contracts import TABLE_COLUMNS, describe_contracts
from .guarded_task import ConfirmedMetricTask, GuardedTaskExecutor


UPSTREAM_COMMIT = "06a46eea2d248a0f2b3e9096035424df4f2dc2eb"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAD_ORDER_SUM = (
    "SELECT SUM(o.total) AS revenue FROM orders o "
    "INNER JOIN order_lines l ON o.order_id = l.order_id"
)
GOOD_ORDER_SUM = "SELECT SUM(total) AS revenue FROM orders"
GOOD_LINE_JOIN = (
    "SELECT SUM(l.amount) AS revenue FROM orders o "
    "INNER JOIN order_lines l ON o.order_id = l.order_id"
)


def verify_upstream_files(manifest_path: Path | None = None) -> dict:
    """Detect a different installed module or changed pinned extension points.

    The bootstrap checks the Git commit. This runtime check fingerprints the
    selected loop/action/interface files, not every transitive upstream module.
    """
    manifest_path = manifest_path or PROJECT_ROOT / "docs/research/dbgpt_pin.json"
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    upstream = PROJECT_ROOT / ".upstream/DB-GPT"
    expected_module = upstream / "packages/dbgpt-core/src/dbgpt/agent/expand/data_scientist_agent.py"
    if Path(inspect.getfile(DataScientistAgent)).resolve() != expected_module.resolve():
        raise RuntimeError("DB-GPT was not imported from the pinned local checkout")
    if manifest["commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("Unexpected upstream commit in runtime manifest")
    for relative, expected_hash in manifest["sha256"].items():
        path = (upstream / relative).resolve()
        if not path.is_relative_to(upstream.resolve()):
            raise RuntimeError("Invalid upstream manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Pinned upstream source changed: {relative}")
    return {
        "commit": UPSTREAM_COMMIT, "checked_files": len(manifest["sha256"]),
        "module": str(expected_module.relative_to(PROJECT_ROOT)),
    }


class ReadOnlyDemoResource(DBResource):
    """Minimal actual DB-GPT resource; only synthetic SQLite files belong here.

    This is the original baseline's SQL reader, not our semantic checker.
    DB-GPT's original ChartAction and correctness_check call this resource.
    SQLite read-only mode and a VM deadline bound that diagnostic path.
    """

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.queries: list[str] = []
        super().__init__(
            name=f"synthetic-{uuid4().hex}", db_name=str(self.path),
            db_type="sqlite", dialect="sqlite",
        )

    def get_schema_link(self, db: str, question: str | None = None):
        return json.dumps(describe_contracts(), ensure_ascii=False)

    def _sync_query(self, db: str, sql: str):
        if str(self.path) != db:
            raise ValueError("Only the bound synthetic database is available")
        self.queries.append(sql)
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        deadline = time.monotonic() + 2
        try:
            connection.execute("PRAGMA query_only=ON")
            # The original comparator needs execution safety without semantic
            # grain checks. SQLite read-only mode alone still allows ATTACH and
            # some PRAGMA operations, so keep its resource on registered reads.
            def authorize(action, table, column, database, trigger):
                if action == sqlite3.SQLITE_SELECT:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ:
                    allowed = database == "main" and table in TABLE_COLUMNS and (column in TABLE_COLUMNS[table] or column == "")
                    return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_FUNCTION and (column or "").lower() == "sum":
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            connection.set_authorizer(authorize)
            connection.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
            cursor = connection.execute(sql)
            try:
                return [column[0] for column in cursor.description], cursor.fetchall()
            finally:
                cursor.close()
        finally:
            connection.close()

    def close(self):
        self._executor.shutdown(wait=True)


class ScriptedLLMClient(LLMClient):
    """Deterministic *test double*, recording real upstream LLM requests.

    Responses are supplied by the caller; no inference/network/token billing.
    Fail on extra calls rather than silently recycling a successful response.
    Token counts are only a rough length proxy needed by the upstream interface.
    """

    def __init__(self, responses: list[dict | str]):
        self.responses = deepcopy(responses)
        self.requests: list[list[dict]] = []
        self.model_name = "offline-scripted-dataverify"
        self._cache: dict = {}

    @property
    def cache(self):
        return self._cache

    async def models(self):
        return [ModelMetadata(model=self.model_name, context_length=32768)]

    async def count_token(self, model: str, prompt: str):
        return max(1, len(prompt) // 3)

    def _respond(self, request):
        self.requests.append([message.model_dump(mode="json") for message in request.get_messages()])
        index = len(self.requests) - 1
        if index >= len(self.responses):
            raise AssertionError("Unexpected extra upstream LLM request")
        response = self.responses[index]
        text = response if isinstance(response, str) else json.dumps(response)
        return ModelOutput(error_code=0, text=text)

    async def generate(self, request, message_converter=None):
        return self._respond(request)

    async def generate_stream(self, request, message_converter=None):
        yield self._respond(request)


class GuardedSqlInput(SqlInput):
    metric_id: str = Field(
        ..., description="Repeat the caller-confirmed metric_id; do not change it."
    )


class GuardedChartAction(ChartAction):
    """Check before emitting a view; never execute the raw native ChartAction.

    Feedback content is intentionally visible as an observation to the agent;
    rejected candidate values are retained only in the private audit trace.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.guard: GuardedTaskExecutor | None = None
        self.audit: list[dict] = []

    @property
    def out_model_type(self):
        return GuardedSqlInput

    async def run(self, ai_message: str, **kwargs):
        if self.guard is None:
            raise RuntimeError("A fresh task guard must be bound before execution")
        try:
            candidate = self._input_convert(ai_message, GuardedSqlInput)
        except Exception:
            feedback = {
                "verification_status": "rejected",
                "error_code": "INVALID_ACTION_FORMAT",
                "feedback": "Return one JSON object matching the required action schema.",
                "accepted_value": None,
            }
            self.audit.append(deepcopy(feedback))
            return ActionOutput(
                is_exe_success=False, content=json.dumps(feedback), view=None,
            )

        # Existing checker owns the read transaction for query and probes.
        feedback = await asyncio.to_thread(
            self.guard.execute, candidate.sql, candidate.metric_id,
        )
        audit = self.guard.last_result
        publication = self.guard.publish()
        if publication["error_code"] in {"SOURCE_CHANGED_DURING_TASK", "SOURCE_UNAVAILABLE"}:
            # Source can change between execute() and this rendering boundary.
            # Keep the action content/view/retry decision consistent, too.
            terminal = {
                "verification_status": "rejected", "accepted_value": None,
                "error_code": publication["error_code"],
                "feedback": "Source snapshot changed or is unavailable; start a new confirmed task.",
            }
            feedback.update(terminal)
            audit.update(terminal)
        self.audit.append(audit)
        accepted = feedback["verification_status"] == "verified"
        if not accepted:
            # A candidate scalar is not a publishable result. Explicit evidence
            # can still describe why it was rejected; code mode has none.
            feedback["value"] = None
        content = json.dumps(feedback, ensure_ascii=False)
        return ActionOutput(
            # Parsed/executed SQL still needs the agent's correctness_check.
            # Blocked SQL goes through the native failure-content retry branch.
            is_exe_success=feedback["execution_status"] == "executed",
            content=content,
            observations=content,
            view=json.dumps(publication, ensure_ascii=False) if accepted else None,
            resource_type="database",
            resource_value="caller-confirmed-sqlite-snapshot",
            have_retry=feedback["error_code"] not in {
                "SOURCE_CHANGED_DURING_TASK", "SOURCE_UNAVAILABLE",
            },
        )


class GuardedDataScientistAgent(DataScientistAgent):
    """Extension points only; inherited native loop/memory perform the retry."""

    _guard: GuardedTaskExecutor = PrivateAttr()

    def __init__(self, *, guard: GuardedTaskExecutor, **kwargs):
        super().__init__(**kwargs)
        self._guard = guard
        self._init_actions([GuardedChartAction])
        self.actions[0].guard = guard

    async def correctness_check(self, message: AgentMessage):
        publication = self._guard.publish()
        if publication["status"] == "verified":
            return True, None
        # Only the action's mode-filtered feedback enters upstream memory.
        # last_result contains full evidence and must never leak into code mode.
        return False, message.action_report.content if message.action_report else "NO_ACTION"

    async def adjust_final_message(self, is_success: bool, reply_message: AgentMessage):
        publication = gated_publication(self._guard, is_success)
        reply_message.success = publication["status"] == "verified"
        # Upstream ignores this hook's return value: mutate the actual object.
        reply_message.content = json.dumps(publication, ensure_ascii=False)
        if reply_message.action_report:
            reply_message.action_report.content = reply_message.content
            reply_message.action_report.view = reply_message.content if reply_message.success else None
            reply_message.action_report.is_exe_success = reply_message.success


def gated_publication(guard: GuardedTaskExecutor, upstream_success: bool) -> dict:
    """Also used outside the native loop for exceptions/timeouts skipping hooks."""
    publication = guard.publish()
    if not upstream_success:
        publication.update(
            status="unverified", value=None, accepted_value=None, verified_sql=None,
            error_code=publication["error_code"] or "UPSTREAM_DID_NOT_FINISH",
        )
    return publication


def sql_candidate(sql: str, metric_id: str = "order_revenue") -> dict:
    return {
        "display_type": "response_table", "sql": sql, "metric_id": metric_id,
        "thought": "Controlled integration candidate; not a real model prediction.",
    }


async def run_native_case(
    db_path: Path,
    task: ConfirmedMetricTask,
    responses: list[dict | str] | None = None,
    *,
    mode: Literal["original", "code", "evidence"] = "evidence",
    max_attempts: int = 2,
    output_dir: Path,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Run actual upstream generate_reply; return a private diagnostic record.

    In guarded modes ONLY `publication` is an application-facing answer.
    Requests, action audit and native streaming events are offline diagnostics.
    The original mode is deliberately an unmodified upstream comparator.
    """
    if mode not in {"original", "code", "evidence"}:
        raise ValueError("mode must be original, code or evidence")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    upstream_check = verify_upstream_files()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if llm_client is not None and responses is not None:
        raise ValueError("Choose an actual LLM client or scripted responses, not both")
    if llm_client is None and responses is None:
        raise ValueError("A client or explicitly scripted responses are required")
    client = llm_client if llm_client is not None else ScriptedLLMClient(responses)
    resource = ReadOnlyDemoResource(db_path)
    guard = None if mode == "original" else GuardedTaskExecutor(db_path, task, feedback_mode=mode)
    context = AgentContext(
        conv_id=f"dataverify-{uuid4().hex}", language="en", output_dir=str(output_dir),
        temperature=0, verbose=False,
    )
    agent = (
        DataScientistAgent(max_retry_count=max_attempts, max_timeout=120, stream_out=False)
        if guard is None else
        GuardedDataScientistAgent(guard=guard, max_retry_count=max_attempts, max_timeout=120, stream_out=False)
    )
    sender = UserProxyAgent()
    events: list[dict] = []

    async def capture_event(event_type, payload):
        events.append({"event_type": event_type, "payload": deepcopy(payload)})

    try:
        agent = await agent.bind(context).bind(LLMConfig(llm_client=client)).bind(resource).build()
        sender = await sender.bind(context).build()
        # The same confirmed task text is supplied in all comparison groups.
        question = (
            task.question + "\nCaller-confirmed metric_id=" + task.metric_id +
            ". Preserve this metric during repairs."
        )
        reply = await agent.generate_reply(
            AgentMessage(content=question, current_goal=question), sender=sender,
            stream_callback=capture_event,
        )
        publication = gated_publication(guard, bool(reply.success)) if guard else None
        return {
            "mode": mode, "upstream_commit": UPSTREAM_COMMIT,
            "upstream_file_check": upstream_check,
            "client_kind": getattr(client, "client_kind", "deterministic_offline_script_not_real_model"),
            "native_loop": "DataScientistAgent.generate_reply (inherited)",
            "upstream_success": bool(reply.success),
            "publication": publication,
            "native_reply": reply.to_dict(),
            "llm_requests": deepcopy(client.requests),
            "llm_calls": len(client.requests),
            "http_events": deepcopy(getattr(client, "http_events", [])),
            "model_usage": deepcopy(getattr(client, "usage", [])),
            "turn_origins": deepcopy(getattr(client, "turn_origins", [])),
            "actions": deepcopy(agent.actions[0].audit) if guard else [],
            "raw_resource_queries": list(resource.queries),
            "stream_events": events,
        }
    finally:
        resource.close()
        agent.executor.shutdown(wait=True)
        sender.executor.shutdown(wait=True)
        if callable(getattr(client, "aclose", None)):
            await client.aclose()
