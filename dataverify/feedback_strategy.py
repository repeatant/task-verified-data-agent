"""Select feedback detail from one task's observed failures.

This module only chooses information to return. It cannot accept SQL, increase
budgets, access model credentials, or consult evaluation answers.
"""

from __future__ import annotations

import hashlib
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
