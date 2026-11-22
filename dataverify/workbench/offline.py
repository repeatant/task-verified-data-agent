"""Scripted model responses for registered examples. SQL and checks execute normally."""

from __future__ import annotations

import json

from dataverify.agent.dispute_model import ScriptedDisputeModel
from dataverify.agent.graph_model import ScriptedGraphModel

from . import penguin_source as penguins
from . import sample_source as orders
from .sample_source import AMBIGUOUS_COUNT, COUNT_LINES, NET_AMOUNT

_SETTLED = " FROM order_lines WHERE status = 'settled'"
_VALUES = "[r['amount'] for r in tables['order_lines'] if r['status'] == 'settled']"
_ORDERS = "[r['order_id'] for r in tables['order_lines'] if r['status'] == 'settled']"
_SQL_COUNT = "SELECT COUNT(*) AS settled_lines" + _SETTLED
_REF_COUNT = f"[[len({_VALUES})]]"
_FINISH = {"action": "finish", "reason": "已按样例约定检查查询范围和结果。"}


class OfflineReview:
    """Choose the authored reply by node input so a saved-node resume also works."""

    client_kind = "offline_workbench_review"

    def __init__(self, plan, decision):
        self.plan, self.decision = plan, decision

    async def generate(self, messages):
        context = json.loads(messages[-1]["content"])
        return self.decision if "candidate_sql" in context else self.plan

    async def aclose(self):
        pass


class CountRepair:
    """A deterministic repair scenario; no model or semantic classifier involved."""

    def __init__(self, *, split_finish_evidence):
        self.split_finish_evidence = split_finish_evidence

    async def generate(self, messages):
        state = json.loads(messages[-1]["content"])
        replay = state["latest_replay"]["id"]
        current_query = next(
            (q for q in state["query_observations"] if q["current_version_evidence"]), None
        )
        query = {
            "action": "query",
            "sql": _SQL_COUNT,
            "hypothesis": "按明细计数是否与按金额去重得到不同结果？",
        }
        if not self.split_finish_evidence and not state["query_observations"]:
            return query
        for target, program in (("candidate", _SQL_COUNT), ("reference", _REF_COUNT)):
            if state["programs"][target] != program:
                return {
                    "action": "revise",
                    "target": target,
                    "program": program,
                    "reason": "计数单位是明细，同额记录须分别计入。",
                    "requirement_quote": "相同金额的不同明细分别计数",
                    "evidence_ids": [replay],
                }
        finish = {
            "action": "finish",
            "reason": "三份当前程序已在保留输入上重放。",
        }
        if self.split_finish_evidence:
            if current_query is None:
                return query
            return {**finish, "signoff_ids": [replay], "supporting_ids": [current_query["id"]]}
        return {**finish, "evidence_ids": [replay]}

    async def aclose(self):
        pass


def _penguin_models(task, definitions):
    question = task.question
    if question not in (penguins.SPECIES_COUNTS, penguins.MASS_COVERAGE, penguins.AMBIGUOUS_COUNT):
        raise ValueError("OFFLINE_QUESTION_NOT_REGISTERED")
    definition = definitions.strip().rstrip("。")
    if definition and not (
        question == penguins.AMBIGUOUS_COUNT and definition in ("全部记录", "有体重记录")
    ):
        raise ValueError("OFFLINE_DEFINITION_NOT_REGISTERED")

    rows = "tables['penguins']"
    if question == penguins.SPECIES_COUNTS:
        sql = "SELECT species, COUNT(*) AS observations FROM penguins GROUP BY species ORDER BY species"
        names = f"names = [row['species'] for row in {rows}]\n"
        reference = names + "[[s, len([n for n in names if n == s])] for s in sorted(set(names))]"
        auditor = (
            f"records = {rows}\n"
            "species = sorted({r['species'] for r in records})\n"
            "[[s, sum(1 for r in records if r['species'] == s)] for s in species]"
        )
    elif question == penguins.MASS_COVERAGE or definition == "有体重记录":
        sql = "SELECT COUNT(body_mass_g) AS observations FROM penguins"
        reference = f"[[sum(1 for r in {rows} if r['body_mass_g'] is not None)]]"
        auditor = f"[[len({rows}) - len([r for r in {rows} if r['body_mass_g'] is None])]]"
    else:
        sql = "SELECT COUNT(*) AS observations FROM penguins"
        reference, auditor = f"[[len({rows})]]", f"[[sum(1 for r in {rows})]]"

    plan = {
        "action": "plan",
        "assumptions": [],
        "basis": [{"source": "question", "quote": question}],
        "fixtures": [],
    }
    if definition:
        plan["basis"].append({"source": "definitions", "quote": definition})
    first = {**plan, "reference": reference}
    if question == penguins.AMBIGUOUS_COUNT and not definition:
        first = {"action": "clarify", "reason": "“可用于分析”指“全部记录”还是“有体重记录”？"}
    return (
        ScriptedGraphModel([{"sql": sql, "task_id": task.task_id}]),
        OfflineReview(first, _FINISH),
        OfflineReview({**plan, "reference": auditor}, _FINISH),
        ScriptedDisputeModel([]),
    )


def offline_models(task, definitions, *, source_id=orders.DATASET, split_finish_evidence=False):
    if source_id == penguins.DATASET:
        return _penguin_models(task, definitions)
    if source_id != orders.DATASET:
        raise ValueError("UNKNOWN_REGISTERED_SOURCE")
    question = task.question
    if question not in (COUNT_LINES, NET_AMOUNT, AMBIGUOUS_COUNT):
        raise ValueError("OFFLINE_QUESTION_NOT_REGISTERED")
    # Never ignore a user-specified definition. The demo has only two authored
    # answers to the clarification case. Free-form definitions need the live mode.
    normalized = definitions.strip().rstrip("。")
    if normalized and not (
        question == AMBIGUOUS_COUNT and normalized in ("按明细条数", "按订单数")
    ):
        raise ValueError("OFFLINE_DEFINITION_NOT_REGISTERED")
    plan = {
        "action": "plan",
        "assumptions": [],
        "basis": [{"source": "question", "quote": question}],
        "fixtures": [],
    }
    if question == COUNT_LINES:
        sql = "SELECT COUNT(DISTINCT amount) AS settled_lines" + _SETTLED
        reference = f"[[len(set({_VALUES}))]]"
        auditor = _REF_COUNT
        decision = {
            "action": "attribute",
            "target": "ambiguous",
            "basis": plan["basis"],
            "checks": [],
            "reason": "当前两个实现都按金额去重，与独立核算有分歧。",
        }
    elif question == NET_AMOUNT:
        sql = "SELECT COALESCE(SUM(amount), 0) AS net_amount" + _SETTLED
        reference = auditor = f"[[sum({_VALUES})]]"
        decision = _FINISH
    elif normalized == "按订单数":
        sql = "SELECT COUNT(DISTINCT order_id) AS settled_orders" + _SETTLED
        reference = auditor = f"[[len(set({_ORDERS}))]]"
        decision = _FINISH
    else:
        sql, reference, auditor, decision = _SQL_COUNT, _REF_COUNT, _REF_COUNT, _FINISH
    first = {**plan, "reference": reference}
    if question == AMBIGUOUS_COUNT and not normalized:
        first = {"action": "clarify", "reason": "请输入“按明细条数”或“按订单数”确认计数单位。"}
    return (
        ScriptedGraphModel([{"sql": sql, "task_id": task.task_id}]),
        OfflineReview(first, decision),
        OfflineReview({**plan, "reference": auditor}, _FINISH),
        CountRepair(split_finish_evidence=split_finish_evidence),
    )
