"""Select feedback detail from one task's observed failures.

This module only chooses information to return. It cannot accept SQL, increase
budgets, access model credentials, or consult evaluation answers.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import sqlglot


def candidate_fingerprint(sql: str | None) -> str | None:
    """Ignore SQL formatting while preserving literals and extra statements."""
    if type(sql) is not str:
        return None

    normalized, kind = sql, "text:"
    if len(sql) <= 16_000:
        try:
            statements = sqlglot.parse(sql, read="sqlite")
            if len(statements) == 1 and statements[0] is not None:
                normalized = statements[0].sql(dialect="sqlite", normalize=True, comments=False)
                kind = "ast:"
        except (sqlglot.errors.SqlglotError, ValueError, TypeError, RecursionError):
            # Invalid SQL still has an identity; failed parsing must never
            # authorize it or discard the part that made it invalid.
            pass
    return kind + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class FeedbackStrategy:
    """Task-local evidence selection; escalation never changes acceptance.

    A successful candidate clears the failure streak. Once this task has needed
    full evidence, keep that detail level until a new strategy is constructed.
    """

    mode: str
    selected_mode: str = field(default="compact", init=False)
    escalation_reason: str | None = field(default=None, init=False)
    _failed_candidates: set[str] = field(default_factory=set, init=False, repr=False)
    _previous_error_code: str | None = field(default=None, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)

    def select(self, error_code: str | None, sql: str | None = None) -> str:
        """Observe a result and choose the next public evidence format."""
        if self.mode != "adaptive":
            return self.mode
        if error_code is None:
            self._previous_error_code = None
            self._consecutive_failures = 0
            return self.selected_mode

        self._consecutive_failures += 1
        fingerprint = candidate_fingerprint(sql)
        if self.selected_mode == "compact":
            reason = self._escalation_trigger(error_code, fingerprint)
            if reason is not None:
                self.selected_mode = "full"
                self.escalation_reason = reason

        if fingerprint is not None:
            self._failed_candidates.add(fingerprint)
        self._previous_error_code = error_code
        return self.selected_mode

    def _escalation_trigger(self, error_code: str, fingerprint: str | None) -> str | None:
        """Keep specific repeat diagnoses ahead of the generic failure streak."""
        if fingerprint is not None and fingerprint in self._failed_candidates:
            return "repeated_failed_candidate"
        if error_code == self._previous_error_code:
            return "persistent_error_code"
        if self._consecutive_failures >= 2:
            return "consecutive_failures"
        return None

    def metadata(self, *, evidence_available: bool) -> dict[str, Any]:
        """Describe selection without inventing evidence that was not found."""
        return {
            "selected_mode": self.selected_mode,
            "escalation_reason": self.escalation_reason,
            "evidence_available": evidence_available,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Serialize task-local progress without SQL text or model messages."""
        if len(self._failed_candidates) > 32 or self._consecutive_failures > 32:
            raise ValueError("Feedback checkpoint exceeds the bounded task history")
        return {
            "version": 1,
            "mode": self.mode,
            "selected_mode": self.selected_mode,
            "escalation_reason": self.escalation_reason,
            "failed_candidates": sorted(self._failed_candidates),
            "previous_error_code": self._previous_error_code,
            "consecutive_failures": self._consecutive_failures,
        }

    @classmethod
    def from_checkpoint(cls, state: dict[str, Any]) -> FeedbackStrategy:
        """Restore validated diagnostic progress; no acceptance is represented."""
        expected = {
            "version",
            "mode",
            "selected_mode",
            "escalation_reason",
            "failed_candidates",
            "previous_error_code",
            "consecutive_failures",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("Invalid feedback checkpoint fields")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("Unsupported feedback checkpoint version")
        cls._validate_checkpoint_selection(state)
        cls._validate_checkpoint_history(state)
        strategy = cls(state["mode"])
        strategy.selected_mode = state["selected_mode"]
        strategy.escalation_reason = state["escalation_reason"]
        strategy._failed_candidates = set(state["failed_candidates"])
        strategy._previous_error_code = state["previous_error_code"]
        strategy._consecutive_failures = state["consecutive_failures"]
        return strategy

    @staticmethod
    def _validate_checkpoint_selection(state: dict[str, Any]) -> None:
        mode, selected, reason = (
            state[key] for key in ("mode", "selected_mode", "escalation_reason")
        )
        if (
            type(mode) is not str
            or mode not in {"full", "compact", "code", "adaptive"}
            or type(selected) is not str
            or selected not in {"full", "compact"}
        ):
            raise ValueError("Invalid feedback checkpoint mode")
        reasons = {"repeated_failed_candidate", "persistent_error_code", "consecutive_failures"}
        if reason is not None and (type(reason) is not str or reason not in reasons):
            raise ValueError("Unknown feedback escalation reason")
        if (selected == "compact") != (reason is None):
            raise ValueError("Inconsistent feedback escalation state")

    @staticmethod
    def _validate_checkpoint_history(state: dict[str, Any]) -> None:
        failed = state["failed_candidates"]
        if (
            not isinstance(failed, list)
            or len(failed) > 32
            or any(
                type(item) is not str or not re.fullmatch(r"(?:ast|text):[0-9a-f]{64}", item)
                for item in failed
            )
            or len(set(failed)) != len(failed)
        ):
            raise ValueError("Invalid failed-candidate history")
        code, streak = state["previous_error_code"], state["consecutive_failures"]
        if code is not None and (
            type(code) is not str or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", code)
        ):
            raise ValueError("Invalid previous error code")
        if type(streak) is not int or not 0 <= streak <= 32 or ((code is None) != (streak == 0)):
            raise ValueError("Invalid failure streak")
        if streak >= 2 and state["selected_mode"] != "full":
            raise ValueError("Inconsistent feedback escalation state")
        if state["mode"] != "adaptive" and (
            state["selected_mode"] != "compact"
            or state["escalation_reason"] is not None
            or failed
            or code
            or streak
        ):
            raise ValueError("Fixed feedback modes cannot contain adaptive progress")
