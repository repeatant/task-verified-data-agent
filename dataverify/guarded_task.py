"""A confirmed metric boundary around the existing SQL contract checker.

The caller confirms the metric before model execution. This module neither
infers natural-language intent nor lets a model change that confirmation. Each
executor belongs to one task/run; use separate instances for concurrent tasks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .checker import CheckedQueryExecutor
from .contracts import METRIC_CATALOG
from .database import database_sha256


@dataclass(frozen=True, slots=True)
class ConfirmedMetricTask:
    """Caller-confirmed inputs, not evidence of automatic intent understanding."""

    question: str
    metric_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if len(self.question) > 2000:
            raise ValueError("question must contain at most 2000 characters")
        if not isinstance(self.metric_id, str) or self.metric_id not in METRIC_CATALOG:
            raise ValueError("metric_id must be an explicitly registered metric")


class GuardedTaskExecutor:
    """Keep metric identity fixed through execution, repair and publication.

    ``execute`` returns planner feedback. ``last_result`` retains full audit
    evidence even in code-only feedback mode. ``publish`` accepts no externally
    supplied result: its value comes only from the last verified execution.

    Source fingerprints are conservative change detection (main file + visible
    WAL), not a transactional guarantee after publication. Query/probe snapshot
    consistency remains the responsibility of the reused CheckedQueryExecutor.
    """

    def __init__(
        self,
        db_path: Path,
        task: ConfirmedMetricTask,
        *,
        feedback_mode: Literal["code", "evidence"] = "evidence",
    ) -> None:
        if not isinstance(task, ConfirmedMetricTask):
            raise TypeError("task must be a ConfirmedMetricTask")
        if feedback_mode not in ("code", "evidence"):
            raise ValueError("feedback_mode must be code or evidence")
        self._task = task
        self._db_path = Path(db_path).resolve()
        self._feedback_mode = feedback_mode
        self._executor = CheckedQueryExecutor(self._db_path)
        self._source_sha256 = database_sha256(self._db_path)
        self._source_error: str | None = None
        self._last_result: dict[str, Any] | None = None
        self._accepted: dict[str, Any] | None = None

    @property
    def task(self) -> ConfirmedMetricTask:
        return self._task

    @property
    def last_result(self) -> dict[str, Any] | None:
        """Full diagnostic evidence; modifying the copy cannot authorize output."""
        return deepcopy(self._last_result)

    def _source_is_unchanged(self) -> bool:
        if self._source_error:
            return False
        try:
            unchanged = database_sha256(self._db_path) == self._source_sha256
        except OSError:
            self._source_error = "SOURCE_UNAVAILABLE"
            return False
        if not unchanged:
            self._source_error = "SOURCE_CHANGED_DURING_TASK"
        return unchanged

    def _blocked_result(self, sql: str, code: str, feedback: str) -> dict[str, Any]:
        return {
            "sql": sql,
            "metric_id": self.task.metric_id,
            "confirmed_metric_id": self.task.metric_id,
            "execution_status": "blocked",
            "verification_status": "rejected",
            "value": None,
            "accepted_value": None,
            "checks": [],
            "error_code": code,
            "feedback": feedback,
            "evidence": [],
        }

    def _feedback_view(self, result: dict[str, Any]) -> dict[str, Any]:
        output = deepcopy(result)
        if self._feedback_mode == "code":
            # Do not leave examples in free text/check metadata in this ablation.
            output["feedback"] = output["error_code"] or "VERIFIED"
            output["checks"] = []
            output["evidence"] = []
        return output

    def execute(self, sql: str, proposed_metric_id: str) -> dict[str, Any]:
        """Reject metric drift before any SQL; a repair must keep its target."""
        self._accepted = None
        if proposed_metric_id != self.task.metric_id:
            result = self._blocked_result(
                sql, "METRIC_DRIFT",
                "The proposed metric differs from the caller-confirmed task; "
                "repair the SQL without changing the confirmed metric.",
            )
            result["proposed_metric_id"] = proposed_metric_id
            result["evidence"] = [{
                "kind": "metric_identity_mismatch",
                "confirmed_metric_id": self.task.metric_id,
                "proposed_metric_id": proposed_metric_id,
            }]
        elif not self._source_is_unchanged():
            result = self._blocked_result(
                sql, self._source_error or "SOURCE_UNAVAILABLE",
                "The task's source snapshot changed or is unavailable; "
                "start a newly confirmed task before publishing a result.",
            )
        else:
            result = self._executor.execute(sql, self.task.metric_id)
            result["confirmed_metric_id"] = self.task.metric_id
            # Preserve diagnostic evidence, but never accept it after source drift.
            if not self._source_is_unchanged():
                result.update(
                    verification_status="rejected", accepted_value=None,
                    error_code=self._source_error,
                    feedback="Source changed during execution; result not accepted.",
                )
            elif (
                result.get("execution_status") == "executed"
                and result.get("verification_status") == "verified"
                and result.get("accepted_value") is not None
            ):
                self._accepted = deepcopy(result)
        self._last_result = deepcopy(result)
        return self._feedback_view(result)

    def publish(self) -> dict[str, Any]:
        """Publish internal accepted evidence, never a model's claimed answer."""
        if not self._source_is_unchanged():
            self._accepted = None
        accepted = self._accepted
        value = accepted["accepted_value"] if accepted is not None else None
        error_code = self._source_error
        if accepted is None and error_code is None:
            error_code = (
                self._last_result.get("error_code") if self._last_result else None
            ) or "NO_VERIFIED_EVIDENCE"
        return {
            "question": self.task.question,
            "metric_id": self.task.metric_id,
            "status": "verified" if accepted is not None else "unverified",
            "value": value,
            "accepted_value": value,
            "error_code": error_code,
            "source_sha256": self._source_sha256,
            "source_fingerprint_algorithm": "sha256(main-file + visible-wal)",
            "intent_confirmation": "caller_confirmed_not_inferred",
            "verified_sql": accepted["sql"] if accepted is not None else None,
        }
