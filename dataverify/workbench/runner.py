"""Run one question on a registered database and publish only inspected evidence."""

from __future__ import annotations

import sqlite3
from contextlib import AsyncExitStack, closing, nullcontext
from hashlib import sha256
from pathlib import Path

from dataverify.agent.discovery_audit import evidence_agrees
from dataverify.agent.discovery_graph import run_discovery
from dataverify.agent.dispute_graph import DisputePolicy, run_dispute_graph
from dataverify.runtime.reporting import digest_json
from dataverify.verification.discovery_runtime import DiscoveryTask, rows_equal, validate_rows
from dataverify.verification.reference_expression import calculate
from dataverify.verification.relational_data import SQLiteSource, quote_identifier
from dataverify.verification.sql_tools import SQLTools

from .live import live_models, qwen_ready
from .offline import offline_models
from .recovery import JournaledModel
from .sample_source import bind_source, get_source, source_catalog

__all__ = ["run_task", "source_catalog", "qwen_ready"]

REJECTION_DETAILS = {
    "DISPUTE_FINISH_REQUIRES_CURRENT_REPLAY_OR_CHECK": "交付引用须全部来自当前版本的核验回放或检查。调查查询只能作辅助证据。",
    "DISPUTE_SUPPORT_REQUIRES_CURRENT_COMPLETE_QUERY": "辅助说明须引用当前程序版本在绑定数据源上成功且完整的查询。",
    "DISPUTE_OBSERVED_DISAGREEMENT": "当前版本仍有执行差异。",
    "DISPUTE_UNRESOLVED_ASSUMPTIONS": "仍有业务假设需要确认。",
    "DISPUTE_INVALID_ACTION": "模型回复的动作格式无效。",
    "DISPUTE_INVALID_FINAL_DECISION": "最后一次回复不符合终结动作格式。",
}


def _dispute_binding_version():
    """Refuse replay against changed prompts, execution rules, or tool policies."""
    package = Path(__file__).resolve().parents[1]
    files = (
        "agent/dispute_graph.py",
        "agent/dispute_model.py",
        "agent/dispute_no_query.py",
        "agent/evidence_graph.py",
        "agent/evidence_model.py",
        "agent/discovery_context.py",
        "agent/graph_model.py",
        "verification/dispute_runtime.py",
        "verification/sql_tools.py",
        "verification/discovery_runtime.py",
        "verification/reference_audit.py",
        "verification/reference_expression.py",
        "verification/relational_data.py",
        "workbench/runner.py",
        "workbench/offline.py",
        "workbench/recovery.py",
    )
    return digest_json({name: sha256((package / name).read_bytes()).hexdigest() for name in files})


def _can_investigate(state):
    """Route only observed disputes. Preserve explicit user questions and errors."""
    audit = state.get("audit_evidence") or {}
    return bool(
        state["status"] in ("unresolved", "needs_clarification")
        and state.get("candidate")
        and state.get("plan")
        and state.get("audit_plan")
        and audit.get("checks")
        and not evidence_agrees(audit)
        and (
            state.get("attribution")
            or state["reason"] in ("REPAIR_BUDGET_EXHAUSTED", "REFERENCE_REVISION_LIMIT")
        )
    )


def _events(state, dispute):
    evidence = []
    revisions = []
    previous = None
    for event in state["trace"]:
        if event["node"] == "generate":
            sql = event["sql"]
            evidence.append(
                {
                    "stage": "discovery",
                    "title": "生成候选 SQL",
                    "detail": f"第 {event['attempt']} 版",
                    "sql": sql,
                }
            )
            if previous and previous != sql:
                revisions.append(
                    {
                        "target": "candidate",
                        "before": previous,
                        "after": sql,
                        "reason": "核验后重新生成。",
                    }
                )
            previous = sql
        elif event["node"] == "assess_audit":
            summary = event["evidence"]["summary"]
            evidence.append(
                {
                    "stage": "discovery",
                    "title": "三份程序在同一组输入上执行",
                    "detail": "有限样本上一致"
                    if evidence_agrees(event["evidence"])
                    else "发现执行差异",
                    "checks": len(event["evidence"]["checks"]),
                    "all_executed": summary.get("all_executed", False),
                }
            )
    versions = state.get("plan_versions") or []
    for old, new in zip(versions, versions[1:]):
        revisions.append(
            {
                "target": "reference",
                "before": old["plan"]["reference"],
                "after": new["plan"]["reference"],
                "reason": new["reason"],
            }
        )
    if dispute:
        queries_enabled = dispute["policy"]["tools_enabled"]
        evidence.append(
            {
                "stage": "dispute",
                "title": "分歧调查策略",
                "detail": (
                    f"允许额外只读调查，实际执行 {dispute['query_calls']} 次。"
                    if queries_enabled
                    else "仅使用已有执行结果和版本回放，额外调查查询已关闭。"
                ),
                "allow_queries": queries_enabled,
                "reserve_final_decision": dispute["policy"].get("reserve_final_decision", False),
                "query_count": dispute["query_calls"],
            }
        )
        # A later budget exit can mask why an attempted delivery was rejected.
        # Keep every rejection in order, including failures before a parsed action.
        for event in dispute["trace"]:
            if event["node"] != "action_rejected":
                continue
            code = event["code"]
            detail = REJECTION_DETAILS.get(code, "该动作未通过校验。")
            evidence.append(
                {"stage": "dispute", "title": "未通过的动作", "detail": f"{detail}（{code}）"}
            )
        for observation in dispute["query_observations"]:
            evidence.append(
                {
                    "stage": "dispute",
                    "title": "执行只读调查查询",
                    "detail": observation["purpose"],
                    "sql": observation["sql"],
                    "rows": observation["rows"],
                    "id": observation["id"],
                    "scope": observation["scope"],
                    "status": observation["status"],
                }
            )
        for replay in dispute["replays"]:
            first = next(
                (
                    c
                    for c in replay["checks"]
                    if c["candidate"] != c["reference"] or c["candidate"] != c["independent"]
                ),
                replay["checks"][0] if replay["checks"] else {},
            )
            evidence.append(
                {
                    "stage": "dispute",
                    "title": "重放版本并比较执行结果",
                    "detail": "各程序在保留输入上一致"
                    if replay["agreement"]
                    else "仍有分歧或执行失败",
                    "id": replay["id"],
                    "versions": replay["versions"],
                    "scope": replay["scope"],
                    "sample": {
                        key: first[key]
                        for key in ("id", "candidate", "reference", "independent")
                        if key in first
                    },
                }
            )
        programs = {role: history[0]["program"] for role, history in dispute["history"].items()}
        for attempt in dispute["attempts"]:
            role = attempt["target"]
            revisions.append(
                {
                    "target": role,
                    "before": programs.get(role),
                    "after": attempt["program"],
                    "reason": attempt["reason"],
                    "accepted": attempt["accepted"],
                    "quote": attempt["requirement_quote"],
                    "evidence_ids": attempt["evidence_ids"],
                }
            )
            if attempt["accepted"]:
                programs[role] = attempt["program"]
    return evidence, revisions


# Workbench sources are small registered snapshots. Keep full-source reference
# evaluation local, within the interpreter's normal limits, and out of prompts.
_FULL_REPLAY_MAX_ROWS = 500


def _full_reference_check(
    db: SQLiteSource, selected: dict, expected: list
) -> tuple[int, str | None]:
    tables = {}
    row_count = 0
    with closing(db.connect()) as connection:
        connection.execute("BEGIN")
        for table, columns in db.schema.items():
            remaining = _FULL_REPLAY_MAX_ROWS - row_count
            records = connection.execute(
                f"SELECT * FROM {quote_identifier(table)} LIMIT ?", (remaining + 1,)
            ).fetchall()
            if len(records) > remaining:
                return 0, "FULL_REFERENCE_ROW_LIMIT"
            tables[table] = [dict(zip(columns, record)) for record in records]
            row_count += len(records)
    for key in ("plan", "audit_plan"):
        try:
            output = validate_rows(calculate(selected[key]["reference"], tables))
        except ValueError:
            return row_count, "FULL_REFERENCE_EXECUTION_FAILED"
        if not rows_equal(output, expected):
            return row_count, "FULL_REFERENCE_DISAGREEMENT"
    return row_count, None


def _delivery(db, tools, preview, selected, dispute, evidence):
    sql = selected["candidate"]["sql"]
    publication = selected["publication"]
    if publication.get("sql") != sql:
        return None, "FINAL_SOURCE_CHECK_FAILED"
    if dispute:
        if (
            publication.get("scope") != "original_database"
            or publication.get("source_id") != tools.source_id
        ):
            return None, "FINAL_SOURCE_CHECK_FAILED"
    else:
        # Discovery publishes fixture answers. They must reproduce on that
        # fixture even when the same aggregate differs on the complete database.
        check = SQLTools(preview).execute(sql, observation_id="discovery_scope_replay")
        if (
            check["status"] != "ok"
            or not check["complete"]
            or not rows_equal(check["rows"], publication["rows"])
        ):
            return None, "FINAL_SOURCE_CHECK_FAILED"

    final = tools.execute(sql, observation_id="publication", purpose="核对最终交付来源")
    if final["status"] != "ok" or not final["complete"]:
        return None, "FINAL_SOURCE_CHECK_FAILED"
    if dispute and not rows_equal(final["rows"], publication["rows"]):
        return None, "FINAL_SOURCE_CHECK_FAILED"

    try:
        count, failure = _full_reference_check(db, selected, final["rows"])
        source_unchanged = SQLTools(db).source_id == tools.source_id
    except (sqlite3.Error, OSError):
        return None, "FINAL_SOURCE_CHECK_FAILED"
    if not source_unchanged:
        return None, "FINAL_SOURCE_CHECK_FAILED"
    explanation = {
        "FULL_REFERENCE_ROW_LIMIT": "数据量超过工作台的完整复算上限。",
        "FULL_REFERENCE_EXECUTION_FAILED": "参考程序在完整登记数据库上未能完成。",
        "FULL_REFERENCE_DISAGREEMENT": "完整登记数据库上的参考结果与候选 SQL 有差异。",
    }
    evidence.append(
        {
            "stage": "publication",
            "title": "在完整登记数据库上复算",
            "detail": explanation[failure]
            if failure
            else f"两份参考程序与 SQL 在 {count} 条源记录上的结果一致。",
            "scope": "original_database",
        }
    )
    return (None, failure) if failure else (final, None)


async def run_task(
    *,
    run_id: str,
    question: str,
    definitions: str,
    source_id: str,
    mode: str,
    directory: Path,
    allow_queries: bool = False,
    reserve_final_decision: bool = False,
    split_finish_evidence: bool = False,
    on_stage=None,
) -> dict:
    """Persist checkpoints in the service-owned directory for this task version.

    Graph signoff uses bounded fixtures. Delivery checks both reference programs
    and the candidate on the complete registered demonstration database.
    """
    if mode not in {"offline", "qwen"}:
        raise ValueError("UNKNOWN_MODEL_MODE")
    if type(allow_queries) is not bool:
        raise ValueError("INVALID_QUERY_POLICY")
    if type(reserve_final_decision) is not bool:
        raise ValueError("INVALID_FINAL_DECISION_POLICY")
    if type(split_finish_evidence) is not bool:
        raise ValueError("INVALID_SIGNOFF_POLICY")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    db = bind_source(source_id, directory)
    preview = db.preview(rows_per_table=50, retain_parents=True)
    task = DiscoveryTask(
        run_id,
        question,
        get_source(source_id)["definitions"]
        + ("\n用户补充的业务口径：" + definitions if definitions else ""),
        schema=db.schema,
        column_types=db.column_types,
        primary_keys=db.primary_keys,
        foreign_keys=db.foreign_keys,
        allow_source_only_plan=True,
    )
    checkpoint = directory / "discovery.sqlite3"
    notify = on_stage or (lambda stage: None)
    tools = SQLTools(db)
    notify("discovery")
    dispute = None
    # Offline demonstrates a scripted investigation. Live Qwen defaults to
    # replay-only reasoning; its prompt, parser and executor share this choice.
    queries_enabled = mode == "offline" or allow_queries
    thread_id = run_id if queries_enabled else f"{run_id}:replay-only"
    if reserve_final_decision:
        thread_id += ":final-decision"
    if split_finish_evidence:
        thread_id += ":split-evidence"
    if mode == "qwen":
        provider = live_models(directory, allow_queries=allow_queries)
    else:
        try:
            provider = nullcontext(
                offline_models(
                    task,
                    definitions,
                    source_id=source_id,
                    split_finish_evidence=split_finish_evidence,
                )
            )
        except ValueError as exc:
            reason = {
                "OFFLINE_QUESTION_NOT_REGISTERED": "离线模式仅支持数据源列出的三个示例。自由提问请使用千问实时模式。",
                "OFFLINE_DEFINITION_NOT_REGISTERED": "此口径超出了当前离线示例。请按澄清问题给出的选项作答，或使用千问实时模式。",
            }.get(str(exc))
            if reason is None:
                raise
            return {
                "status": "inconclusive",
                "reason": reason,
                "sql": None,
                "candidate_sql": None,
                "rows": None,
                "columns": [],
                "evidence": [],
                "revisions": [],
                "source_id": source_id,
                "source_version": tools.source_id,
                "mode": mode,
                "summary": reason,
                "scope": "离线演示范围",
                "clarification": None,
                "usage": {"requests": 0, "total_tokens": 0, "complete": True},
            }
    with provider as clients:
        writer, reviewer, auditor, investigator = clients
        async with AsyncExitStack() as stack:
            for client in clients:
                stack.push_async_callback(client.aclose)
            state = await run_discovery(
                task,
                preview,
                writer,
                reviewer,
                auditor=auditor,
                checkpoint_path=checkpoint,
                # A saved discovery cannot silently resume with a different
                # downstream action contract. Pre-policy runs allowed queries.
                thread_id=thread_id,
                resume=checkpoint.is_file(),
            )
            if _can_investigate(state):
                notify("dispute")
                binding = {
                    "format": "workbench-dispute-v1",
                    "runtime": _dispute_binding_version(),
                    "mode": mode,
                    "allow_queries": queries_enabled,
                    "reserve_final_decision": reserve_final_decision,
                    "split_finish_evidence": split_finish_evidence,
                    "task": task.public_input(),
                    "source": tools.source_id,
                    "sql": state["candidate"]["sql"],
                    "reference": state["plan"],
                    "auditor": state["audit_plan"],
                    "fixtures": state["fixtures"],
                    "model": getattr(investigator, "model_name", "offline"),
                }
                diagnosis = JournaledModel(
                    investigator, directory / "dispute-calls.sqlite3", binding=digest_json(binding)
                )
                dispute = await run_dispute_graph(
                    task,
                    preview,
                    state["candidate"]["sql"],
                    state["plan"],
                    state["audit_plan"],
                    diagnosis,
                    query_source=db,
                    policy=DisputePolicy(
                        tools_enabled=queries_enabled,
                        reserve_final_decision=reserve_final_decision,
                        split_finish_evidence=split_finish_evidence,
                    ),
                    fixtures=state["fixtures"],
                )
            budget = getattr(writer, "budget", None)
            accounting = budget.snapshot() if budget is not None else {}

    notify("publication")
    selected = dispute or state
    status = selected["status"]
    reason = selected.get("reason") or selected.get("publication", {}).get("reason") or ""
    sql = (selected.get("candidate") or {}).get("sql")
    published_sql = None
    rows, columns = None, []
    evidence, revisions = _events(state, dispute)
    if status == "reviewed":
        final, failure = _delivery(db, tools, preview, selected, dispute, evidence)
        if final is not None:
            published_sql, rows, columns = sql, final["rows"], final["columns"]
            evidence.append(
                {
                    "stage": "publication",
                    "title": "从绑定数据库读取最终结果",
                    "detail": f"返回 {len(rows)} 行",
                    "id": final["id"],
                    "scope": final["scope"],
                    "sql": sql,
                    "rows": rows,
                }
            )
        else:
            status, reason = "inconclusive", failure
    if status not in {"reviewed", "needs_clarification"}:
        status = "inconclusive"
    citations = None
    if published_sql is not None and dispute and "signoff_ids" in dispute["publication"]:
        # Keep the authority and the observations exactly as validated by the graph.
        # An unsuccessful final execution must never look like an approved delivery.
        pub = dispute["publication"]
        citations = {"signoff_ids": pub["signoff_ids"], "supporting_ids": pub["supporting_ids"]}
    return {
        "status": status,
        "reason": reason,
        "sql": published_sql,
        "candidate_sql": sql,
        "rows": rows,
        "columns": columns,
        "evidence": evidence,
        "revisions": revisions,
        "finish_citations": citations,
        "source_id": source_id,
        "source_version": tools.source_id,
        "mode": mode,
        "scope": "保留样本用于有限核验。交付前须在完整登记库上执行 SQL 并与两份参考程序复核。",
        "summary": {
            "reviewed": "有限核验通过，已执行最终查询。",
            "inconclusive": "未能完成核验。",
            "needs_clarification": "需要补充业务口径。",
        }[status],
        "clarification": reason if status == "needs_clarification" else None,
        "usage": {
            "requests": accounting.get("http_attempts", 0),
            "total_tokens": accounting.get("reported_total_tokens", 0),
            "complete": accounting.get("reported_cost_complete", True),
        },
    }
