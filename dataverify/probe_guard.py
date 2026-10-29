"""Mode-isolated execution and regression gate for the probe Agent.

The model receives a selected diagnostic, not the private audit or gold SQL.
Final results come only from a fresh source execution after all registered
checks. This module never imports evaluation cases or model credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .feedback_strategy import FeedbackStrategy, candidate_fingerprint
from .probe_capabilities import policy_features, repair_guidance, validate_policy
from .probe_runtime import ProbeData, ProbeTask, execute_sql
from .probe_testing import (
    ExecutionBudget,
    ProbeBudgetExceeded,
    ProbePair,
    evaluate_pair,
    probe_candidate,
)


def source_digest(task: ProbeTask, data: ProbeData) -> str:
    payload = {"task": task.to_dict(), "data": data.to_dict()}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def source_output_check(rows: Any, task: ProbeTask) -> dict[str, Any]:
    """Check the publication type/shape, without knowing any expected answer.

    A scalar SUM with no matching rows may yield NULL in valid SQLite SQL. That
    is not a usable numeric task result. Empty grouped relations remain legal.
    This gate is shared by legacy/v2 feedback, independently of diagnostic text.
    """

    def number(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value)

    valid = isinstance(rows, list)
    if task.output_kind == "scalar":
        valid = (
            valid
            and len(rows) == 1
            and isinstance(rows[0], list)
            and len(rows[0]) == 1
            and number(rows[0][0])
        )
    elif valid:
        expected_key = int if task.output_kind == "orders" else str
        keys = set()
        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or type(row[0]) is not expected_key
                or not number(row[1])
                or row[0] in keys
            ):
                valid = False
                break
            keys.add(row[0])
    return {
        "valid": bool(valid),
        "code": None if valid else "INVALID_RESULT_CONTRACT",
        "scope": "Numeric result shape and unique typed keys only; not an answer oracle.",
    }


@dataclass(frozen=True)
class _RegressionRun:
    """Keep regression verdicts and their shared execution cost together."""

    verdicts: list[dict[str, Any]]
    queries: int

    @property
    def failed(self) -> bool:
        return any(verdict["status"] != "passed" for verdict in self.verdicts)


class ProbeGuard:
    """Own immutable task/data and select only the permitted public feedback."""

    def __init__(
        self,
        task: ProbeTask,
        data: ProbeData,
        mode: str = "compact",
        seed: int = 17,
        *,
        feedback_policy: str = "capability_v2",
    ):
        if not isinstance(task, ProbeTask) or not isinstance(data, ProbeData):
            raise ValueError("Expected confirmed task and legal immutable data")
        if mode not in {"full", "compact", "code", "adaptive"} or type(seed) is not int:
            raise ValueError("Unknown feedback mode or invalid seed")
        self.task, self.data, self.mode, self.seed = task, data, mode, seed
        self.feedback_policy = validate_policy(feedback_policy)
        self._source_digest = source_digest(task, data)
        self._audit: list[dict] = []
        self._regressions: dict[str, ProbePair] = {}
        self._accepted: dict | None = None
        self._last: dict = {}
        self._terminal_code: str | None = None
        self._feedback_strategy = FeedbackStrategy(mode)

    @property
    def audit(self) -> list[dict]:
        return deepcopy(self._audit)

    @property
    def last_result(self) -> dict:
        return deepcopy(self._last)

    @staticmethod
    def _candidate_fingerprint(sql: str | None) -> str | None:
        """Compatibility entry point for the task-local strategy's identity."""
        return candidate_fingerprint(sql)

    def _select_feedback_mode(self, code: str | None, sql: str | None = None) -> str:
        """Observe failure state without changing any acceptance condition."""
        return self._feedback_strategy.select(code, sql)

    def _feedback(
        self,
        code: str | None,
        message: str,
        *,
        accepted: bool = False,
        rows: list | None = None,
        evidence: str | None = None,
    ) -> dict:
        result = {
            "task_id": self.task.task_id,
            "status": "verified" if accepted else "unverified",
            "verification_status": "verified" if accepted else "rejected",
            "execution_status": "executed" if accepted else "not_accepted",
            "error_code": code,
            "feedback": message,
            "rows": deepcopy(rows) if accepted else None,
            "scope": "registered task properties and bounded tests; not a complete semantic proof",
        }
        if evidence is not None and self.mode != "code":
            result["diagnostic_evidence"] = evidence
        if not accepted and policy_features(self.feedback_policy)["classified_guidance"]:
            guidance = repair_guidance(code, self.task)
            if guidance is not None:
                result["repair_guidance"] = guidance
        if self.mode == "adaptive":
            result["feedback_strategy"] = self._feedback_strategy.metadata(
                evidence_available=evidence is not None
            )
        return result

    def invalidate(
        self, code: str = "INVALID_ACTION_FORMAT", feedback: str = "Return one valid JSON action."
    ) -> dict:
        self._accepted = None
        self._select_feedback_mode(code)
        public = self._feedback(code, feedback)
        record = {
            "sql": None,
            "candidate_sql": None,
            "feedback": public,
            "probe_report": None,
            "source_execution": None,
            "regression_checks": [],
        }
        return self._record_result(record)

    def _remember(self, report: dict) -> None:
        evidence = report.get("counterexample")
        if not evidence:
            return
        payload = evidence["compact"]
        pair = ProbePair(
            payload["property_id"],
            ProbeData(**payload["base"]),
            payload["added_orders"],
            payload["added_lines"],
        )
        key = json.dumps(pair.payload(), sort_keys=True, ensure_ascii=False)
        self._regressions.setdefault(key, pair)

    def execute(self, sql: str, proposed_task_id: str) -> dict:
        """Check one candidate, retain its evidence, and replace its publication.

        Reading this method follows the runtime pipeline: task validation,
        candidate checks, regression, feedback selection, and audit recording.
        """
        # Even a malformed new action must invalidate an earlier result.
        self._accepted = None
        rejected = self._validate_bound_task(proposed_task_id)
        if rejected is not None:
            return rejected

        report = probe_candidate(sql, self.task, mode="directed", seed=self.seed)
        source = execute_sql(sql, self.data, self.task)
        output_check = (
            source_output_check(source["rows"], self.task) if source["status"] == "ok" else None
        )

        # 新发现的反例也加入本轮回归，避免候选修改后绕过原来的失败条件。
        self._remember(report)
        regression = self._run_regressions(sql)
        accepted = bool(
            report.get("acceptance")
            and source["status"] == "ok"
            and output_check["valid"]
            and not regression.failed
        )
        if accepted:
            self._select_feedback_mode(None)
            public = self._feedback(
                None,
                "Registered checks passed on source and stored counterexamples.",
                accepted=True,
                rows=source["rows"],
            )
            self._accepted = {"rows": deepcopy(source["rows"]), "verified_sql": sql}
        else:
            public = self._rejection_feedback(sql, report, source, output_check, regression.failed)

        return self._record_result(
            {
                "sql": sql,
                "candidate_sql": sql,
                "feedback": deepcopy(public),
                "probe_report": report,
                "source_execution": source,
                "source_output_check": output_check,
                "regression_checks": regression.verdicts,
                "regression_query_attempts": regression.queries,
            }
        )

    def _validate_bound_task(self, proposed_task_id: str) -> dict | None:
        """Reject terminal/source/task changes before any query is attempted."""
        if self._terminal_code:
            return self.invalidate(self._terminal_code, "The task source is no longer valid.")
        if source_digest(self.task, self.data) != self._source_digest:
            self._terminal_code = "SOURCE_OR_TASK_CHANGED"
            return self.invalidate(self._terminal_code, "Start a new confirmed task.")
        if type(proposed_task_id) is not str or proposed_task_id != self.task.task_id:
            return self.invalidate(
                "TASK_DRIFT", "Preserve the caller-confirmed task_id and requirements."
            )
        return None

    def _run_regressions(self, sql: str) -> _RegressionRun:
        """Execute all retained witnesses under one bounded regression budget."""
        verdicts: list[dict[str, Any]] = []
        budget = ExecutionBudget(max(2, 2 * len(self._regressions)), 8)
        for pair in self._regressions.values():
            try:
                verdict = evaluate_pair(sql, self.task, pair, budget)
            except ProbeBudgetExceeded:
                verdict = {"status": "budget_exhausted", "property_id": pair.property_id}
            verdicts.append(verdict)
        return _RegressionRun(verdicts, budget.queries)

    def _rejection_feedback(
        self,
        sql: str,
        report: dict,
        source: dict,
        output_check: dict | None,
        regression_failed: bool,
    ) -> dict:
        """Preserve error precedence and disclose only the selected evidence."""
        static = report["static"]
        code = (
            static.get("code")
            or source.get("code")
            or (output_check["code"] if output_check else None)
            or (
                "REGRESSION_CHECK_FAILED"
                if regression_failed
                else "PROPERTY_" + report["status"].upper()
            )
        )
        message = static.get("message") or (
            "Result requires a numeric measure; NULL is not a verified answer. Empty grouped relations are allowed."
            if output_check and not output_check["valid"]
            else "Candidate did not pass every registered check. Preserve the original requirements."
        )
        evidence = report.get("counterexample")
        selected_mode = self._select_feedback_mode(code, sql)
        selected = (
            evidence[selected_mode + "_feedback"]
            if evidence and selected_mode in {"full", "compact"}
            else None
        )
        return self._feedback(code, message, evidence=selected)

    def _record_result(self, record: dict) -> dict:
        """Snapshot private diagnostics before returning independent feedback."""
        self._last = deepcopy(record)
        self._audit.append(deepcopy(record))
        return deepcopy(record["feedback"])

    def publish(self) -> dict:
        """Return only the last accepted result while its source remains valid."""
        if source_digest(self.task, self.data) != self._source_digest:
            self._accepted = None
            self._terminal_code = "SOURCE_OR_TASK_CHANGED"
        accepted = self._accepted is not None and self._terminal_code is None
        code = self._terminal_code or (
            self._last.get("feedback", {}).get("error_code") if not accepted else None
        )
        return {
            "status": "verified" if accepted else "unverified",
            "task_id": self.task.task_id,
            "rows": deepcopy(self._accepted["rows"]) if accepted else None,
            "verified_sql": self._accepted["verified_sql"] if accepted else None,
            "error_code": code or (None if accepted else "NO_VERIFIED_ACTION"),
            "scope": "registered necessary properties only; complete correctness independently evaluated",
        }
