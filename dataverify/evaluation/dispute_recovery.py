"""Small, opt-in development check for three-program disagreement recovery.

These are six controlled fault trajectories on previously seen requirements,
not six new public questions. A shared visible fixture creates useful execution
conflicts. Independent scoring fixtures and fault labels never enter the model.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

from dataverify.agent.discovery_graph import run_discovery
from dataverify.agent.discovery_model import QwenReviewModel
from dataverify.agent.graph_model import GraphModelError, InjectedFirstGraphModel, QwenGraphModel
from dataverify.evaluation import evidence_recovery as bounded
from dataverify.evaluation.discovery_roles import InjectedFirstReviewModel
from dataverify.evaluation.expanded_cases import expanded_cases, scoring_data
from dataverify.evaluation.expanded_experiment import SQL_PROMPT, _events, message
from dataverify.evaluation.expanded_scoring import _outcomes, score_controlled
from dataverify.evaluation.structured_confirmation import begin, cached
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json
from dataverify.verification.discovery_runtime import execute_readonly, rows_equal
from dataverify.verification.probe_runtime import ProbeData
from dataverify.verification.sql_tools import SQLTools

METHODS = ("self_review", "legacy_full", "dispute_resolver")
MAX_ACTIONS = 4
MAX_QUERIES = 2
MAX_REVISIONS = 2
CASE_SELECTION = (
    ("upper_boundary_1", "sql_wrong"),
    ("refund_sign_1", "reference_wrong"),
    ("status_scope_1", "auditor_wrong"),
    ("duplicate_amounts_1", "correlated_wrong"),
    ("aggregate_threshold_1", "correct"),
    ("lower_boundary_1", "all_wrong"),
)


@dataclass(frozen=True)
class DisputeCase:
    """Evaluation-only wrapper; never serialize this object into model messages."""

    case_id: str
    base: object
    condition: str

    @property
    def initial_sql(self):
        return self.base.initial_sql

    @property
    def reference_plan(self):
        return self.base.initial_plan()

    @property
    def auditor_plan(self):
        plan = self.base.initial_plan(auditor=True)
        if self.condition == "all_wrong":
            plan = {**plan, "reference": self.base.bad_reference}
        return plan


def study_cases():
    """Freeze by named old mechanisms, before reading any new model results."""
    available = {(c.parent_id, c.condition): c for c in expanded_cases()}
    return [
        DisputeCase(
            f"dispute-{index:02d}",
            available[(parent, "correlated_wrong" if condition == "all_wrong" else condition)],
            condition,
        )
        for index, (parent, condition) in enumerate(CASE_SELECTION, start=1)
    ]


def visible_diagnostic_fixture():
    """One fixed input dataset for every arm and condition; no expected answer.

    Constructed as ordinary diverse records, independently of model outcomes and
    the private scoring set. It is visible experimental support, not an alleged
    automatically discovered counterexample or an unseen-error benchmark.
    """
    middle = "2026-09-15T00:00:00"
    return ProbeData(
        ((1, 501, "store"), (2, 803, "store"), (3, 613, "web"), (4, 109, None), (5, 907, None)),
        (
            (1, 1, 70, "settled", middle),
            (2, 1, 70, "settled", middle),
            (3, 1, -45, "settled", middle),
            (4, 2, 19, "settled", "2026-09-01T00:00:00"),
            (5, 2, 23, "settled", "2026-10-01T00:00:00"),
            (6, 2, 31, "cancelled", middle),
            (7, 3, 13, "pending", middle),
            (8, 3, 41, "settled", middle),
            (9, 5, -7, "settled", middle),
        ),
    )


def case_binding(case):
    base = case.base
    return {
        "task_digest": digest_json(base.task.public_input()),
        "source_digest": digest_json(base.source.to_dict()),
        "visible_diagnostic_digest": digest_json(visible_diagnostic_fixture().to_dict()),
        "initial_sql_digest": digest_json(case.initial_sql),
        "reference_digest": digest_json(case.reference_plan),
        "auditor_digest": digest_json(case.auditor_plan),
        "scoring_suite_digest": digest_json(
            {name: data.to_dict() for name, data in scoring_data(base).items()}
        ),
    }


def _initial_record(case):
    return {
        "status": "inconclusive",
        "candidate": {"sql": case.initial_sql},
        "plan": case.reference_plan,
        "audit_plan": case.auditor_plan,
        "publication": {},
    }


def score_result(case, method, record):
    """Score all final programs; baseline reference/auditor grades are N/A."""
    before = score_controlled(case.base, _initial_record(case))
    final = score_controlled(case.base, record)
    # The prototype publishes a query_source result (the diagnostic fixture),
    # whereas the old scorer assumed publication always came from base.source.
    # Keep all 25 program grades intact; verify returned rows on the actual,
    # host-bound publication source rather than silently mixing the two scopes.
    publication = record.get("publication", {})
    rows_consistent = False
    publication_source = (
        case.base.source if method == "legacy_full" else visible_diagnostic_fixture()
    )
    if publication.get("sql"):
        try:
            rows_consistent = rows_equal(
                execute_readonly(publication["sql"], publication_source),
                publication.get("rows"),
            )
        except (ValueError, TypeError):
            pass
    final.update(
        _outcomes(
            record,
            final["initial"],
            final["candidate"],
            final["publication"],
            publication_rows_consistent=rows_consistent,
        )
    )
    final["publication_source_scope"] = (
        "original_source" if method == "legacy_full" else "shared_visible_diagnostic_fixture"
    )
    final["joint_correct_delivery"] = (
        final["correct_delivery"]
        and final["reference"]["correct"] is True
        and final["auditor"]["correct"] is True
        if method != "self_review"
        else None
    )
    roles = {}
    for role, grade_key in (
        ("candidate", "candidate"),
        ("reference", "reference"),
        ("auditor", "auditor"),
    ):
        supported = method != "self_review" or role == "candidate"
        initial, current = before[grade_key], final[grade_key]
        roles[role] = {
            "initial_correct": initial["correct"],
            "final_correct": current["correct"] is True if supported else None,
            "grade": current if supported else None,
            "repaired": initial["correct"] is False and current["correct"] is True
            if supported
            else None,
            "damaged": initial["correct"] is True and current["correct"] is not True
            if supported
            else None,
            "scope": "25 fixed grading fixtures"
            if supported
            else "N/A: baseline produces no reference/auditor",
        }
    sql = record.get("candidate", {}).get("sql")
    changed = isinstance(sql, str) and sql.strip() != case.initial_sql.strip()
    final.update(
        roles=roles,
        sql_text_changed=changed,
        repaired_initial_error=roles["candidate"]["repaired"] is True and changed,
        unchanged_delivery=final["delivered"] and not changed,
        all_final_programs_correct=(
            all(item["final_correct"] is True for item in roles.values())
            if method != "self_review"
            else None
        ),
        wrong_reference_released=(
            final["delivered"] and roles["reference"]["final_correct"] is not True
            if method != "self_review"
            else None
        ),
        wrong_auditor_released=(
            final["delivered"] and roles["auditor"]["final_correct"] is not True
            if method != "self_review"
            else None
        ),
    )
    if method == "self_review":
        final["joint_correct_delivery"] = None
    return final


async def self_review(case, model):
    """One candidate-only self-review sees the identical two visible datasets."""
    data = {"source": case.base.source, "shared_diagnostic": visible_diagnostic_fixture()}
    observations = [
        SQLTools(source).execute(case.initial_sql, observation_id=f"e{index:04d}", purpose=name)
        for index, (name, source) in enumerate(data.items(), start=1)
    ]
    action = await model.generate(
        [
            message("system", SQL_PROMPT),
            message(
                "user",
                {
                    "task": case.base.task.public_input(),
                    "candidate": {"sql": case.initial_sql},
                    "visible_inputs": {name: source.to_dict() for name, source in data.items()},
                    "executions": observations,
                    "instruction": "Review SQL against the original requirement and the visible executions. Fix it if needed; preserve it otherwise. Successful execution alone does not establish correctness. No answer key is supplied.",
                },
            ),
        ]
    )
    if action["task_id"] != case.base.task.task_id:
        raise ValueError("TASK_ID_MISMATCH")
    final = [
        SQLTools(source).execute(action["sql"], observation_id=f"e{index:04d}", purpose=name)
        for index, (name, source) in enumerate(data.items(), start=3)
    ]
    delivered = all(item["status"] == "ok" and item["complete"] for item in final)
    return {
        "status": "reviewed" if delivered else "inconclusive",
        "candidate": {"sql": action["sql"]},
        "publication": {"sql": action["sql"], "rows": final[1]["rows"]} if delivered else {},
        "observations": observations + final,
        "model_calls": 1,
        "query_calls": 0,
    }


async def run_cell(case, method, config, lane, output, protocol_digest):
    """Persist begin-before-payment and retain every completed outcome unchanged."""
    from dataverify.agent.dispute_graph import DisputePolicy, run_dispute_graph
    from dataverify.agent.dispute_model import QwenDisputeModel

    if method not in METHODS:
        raise ValueError("DISPUTE_EVALUATION_METHOD")
    path = output / "records" / f"{case.case_id}--{method}.json"
    binding = {
        **case_binding(case),
        "method": method,
        "model": config.model,
        "protocol_digest": protocol_digest,
    }
    previous = cached(path, binding)
    if previous is not None:
        return previous
    checkpoint = output / "checkpoints" / (case.case_id + "--legacy_full.sqlite3")
    if method == "legacy_full" and checkpoint.exists():
        raise ValueError("DISPUTE_UNRECORDED_CHECKPOINT_REQUIRES_REVIEW")
    begin(path, binding)
    start, started = lane.snapshot(), time.perf_counter()
    limit = 1 if method == "self_review" else MAX_ACTIONS
    budget = bounded.BoundedCellBudget(lane, limit)
    if method == "legacy_full":
        writer = QwenGraphModel(config, budget, max_tokens=bounded.OUTPUT_TOKEN_CAP)
        reviewer = QwenReviewModel(config, budget, max_tokens=bounded.OUTPUT_TOKEN_CAP)
        auditor = QwenReviewModel(config, budget, max_tokens=bounded.OUTPUT_TOKEN_CAP)
        models = [
            ("legacy_writer", writer),
            ("legacy_reference", reviewer),
            ("legacy_auditor", auditor),
        ]
    else:
        model_type = QwenGraphModel if method == "self_review" else QwenDisputeModel
        model = model_type(config, budget, max_tokens=bounded.OUTPUT_TOKEN_CAP)
        models = [(method, model)]
    try:
        if method == "self_review":
            record = await self_review(case, model)
        elif method == "legacy_full":
            visible = {"name": "shared_diagnostic", "data": visible_diagnostic_fixture().to_dict()}
            reference_plan = {
                **case.reference_plan,
                "fixtures": [*case.reference_plan["fixtures"], visible],
            }
            auditor_plan = {
                **case.auditor_plan,
                "fixtures": [*case.auditor_plan["fixtures"], visible],
            }
            record = await run_discovery(
                case.base.task,
                case.base.source,
                InjectedFirstGraphModel(
                    writer, {"sql": case.initial_sql, "task_id": case.base.task.task_id}
                ),
                InjectedFirstReviewModel(reviewer, reference_plan),
                auditor=InjectedFirstReviewModel(auditor, auditor_plan),
                checkpoint_path=checkpoint,
                thread_id="dispute_recovery",
                max_attempts=3,
                max_reviews=3,
                max_challenge_examples=40,
            )
        else:
            record = await run_dispute_graph(
                case.base.task,
                case.base.source,
                case.initial_sql,
                case.reference_plan,
                case.auditor_plan,
                model,
                query_source=visible_diagnostic_fixture(),
                fixtures=(
                    {"name": "shared_diagnostic", "data": visible_diagnostic_fixture().to_dict()},
                ),
                confirmed_examples=(),
                policy=DisputePolicy(
                    max_actions=MAX_ACTIONS,
                    max_queries=MAX_QUERIES,
                    max_revisions=MAX_REVISIONS,
                    max_context_chars=bounded.INPUT_CHAR_CAP,
                ),
            )
    except (GraphModelError, ValueError) as error:
        record = {
            "status": "inconclusive",
            "candidate": {"sql": case.initial_sql},
            "publication": {},
            "error": str(error)[:160],
        }
        if method != "self_review":
            record.update(plan=case.reference_plan, audit_plan=case.auditor_plan)
    finally:
        for _, model in models:
            await model.aclose()
    events = _events(models, lane)
    row = {
        "case_id": case.case_id,
        "task_id": case.base.task.task_id,
        "condition": case.condition,
        "parent_id": case.base.parent_id,
        "family": case.base.family,
        "method": method,
        "status": record["status"],
        "score": score_result(case, method, record),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "query_calls": record.get("query_calls", 0),
        "usage": summarize_usage(events),
        "http_events": events,
    }
    result = {
        "binding": binding,
        "completed": True,
        "row": row,
        "record": record,
        "contexts": {role: model.requests for role, model in models},
        "http_events": events,
        "budget_start": start,
        "budget_end": lane.snapshot(),
    }
    write_json(path, result)
    return result


def summarize(output, protocol):
    records = [bounded.read_json(path) for path in (output / "records").glob("*.json")]
    rows = [r["row"] for r in records if r.get("completed") and "row" in r]
    observed = {(row["case_id"], row["method"]) for row in rows}
    missing = [
        [case["case_id"], method]
        for case in protocol["cases"]
        for method in METHODS
        if (case["case_id"], method) not in observed
    ]
    metrics = (
        "initial_sql_correct",
        "final_sql_correct",
        "repaired_initial_error",
        "correct_sql_damaged",
        "correct_task_rejected",
        "delivered",
        "correct_delivery",
        "wrong_release",
        "unchanged_delivery",
        "all_final_programs_correct",
        "wrong_reference_released",
        "wrong_auditor_released",
        "joint_correct_delivery",
    )
    methods = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]

        def eligible_metric(row, metric):
            if metric in {"correct_sql_damaged", "correct_task_rejected"}:
                return row["score"]["initial_sql_correct"]
            if metric == "repaired_initial_error":
                return row["score"]["initial"]["correct"] is False
            return row["score"].get(metric) is not None

        def eligible_role(row, role, metric):
            item = row["score"]["roles"][role]
            if item[metric] is None:
                return False
            if metric == "repaired":
                return item["initial_correct"] is False
            if metric == "damaged":
                return item["initial_correct"] is True
            return True

        methods[method] = {
            "trajectories": len(selected),
            "distinct_requirements": len({row["task_id"] for row in selected}),
            "statuses": dict(Counter(row["status"] for row in selected)),
            **{
                metric: {
                    "numerator": sum(row["score"].get(metric) is True for row in selected),
                    "denominator": sum(eligible_metric(row, metric) for row in selected),
                }
                for metric in metrics
            },
            "roles": {
                role: {
                    key: {
                        "numerator": sum(
                            row["score"]["roles"][role][key] is True for row in selected
                        ),
                        "denominator": sum(eligible_role(row, role, key) for row in selected),
                    }
                    for key in ("final_correct", "repaired", "damaged")
                }
                for role in ("candidate", "reference", "auditor")
            },
            "usage": summarize_usage([event for row in selected for event in row["http_events"]]),
        }
    result = {
        "protocol_digest": digest_json(protocol),
        "complete": not missing,
        "completed_cells": len(rows),
        "expected_cells": len(protocol["cases"]) * len(METHODS),
        "controlled_fault_trajectories": len(protocol["cases"]),
        "distinct_original_requirements": protocol["distinct_original_requirements"],
        "missing": missing,
        "methods": methods,
        "rows": rows,
        "usage_all_new_requests": summarize_usage([e for row in rows for e in row["http_events"]]),
        "negative_control": [row for row in rows if row["condition"] == "all_wrong"],
        "limitations": protocol["limitations"],
    }
    write_json(output / "summary.json", result)
    return result
