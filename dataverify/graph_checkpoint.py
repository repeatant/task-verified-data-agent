"""Local run leases and a durable, never-refunded model-spending journal.

LangGraph checkpoints record completed nodes. They cannot make an HTTP request
transactional with SQLite. A separate budget journal commits a reservation
before POST; an unresolved request prevents automatic continuation. Local file
locks serialize writers, including different threads sharing one allowance.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

from .experiment_budget import restore_budget
from .probe_cost_budget import MonetaryRequestBudget


class GraphResumeError(RuntimeError):
    """A safe machine-readable reason to start a new task or review a run."""


@contextmanager
def exclusive_run(path: Path) -> Iterator[None]:
    """Hold an OS lease; a crashed process releases it without deleting files."""
    lock_path = Path(str(path.resolve()) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise GraphResumeError("CONCURRENT_RUN_REFUSED") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_budget(path: Path) -> dict[str, Any]:
    """Read private accounting only; no credentials or model messages live here."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != "graph-budget-v1":
        raise GraphResumeError("INVALID_BUDGET_JOURNAL")
    snapshot = document.get("budget")
    if not isinstance(snapshot, dict):
        raise GraphResumeError("INVALID_BUDGET_JOURNAL")
    return snapshot


def resume_budget(snapshot: dict[str, Any]) -> MonetaryRequestBudget:
    """Reproduce the ledger without extending the original request allowance."""
    budget = restore_budget(
        snapshot, max_extra_requests=1, money_limit_cny=snapshot.get("money_limit_cny")
    )
    limit = snapshot.get("http_attempt_limit")
    if type(limit) is not int or limit < budget.used:
        raise GraphResumeError("INVALID_BUDGET_LIMIT")
    budget.limit = limit
    return budget


class JournaledBudget:
    """Persist accounting before returning control to the HTTP transport.

    The caller holds ``exclusive_run(path)`` for the whole model session. If a
    write fails, the client must not POST or silently use an older snapshot.
    """

    def __init__(self, budget: MonetaryRequestBudget, path: Path):
        self.delegate, self.path = budget, path.resolve()
        if self.path.exists() and read_budget(self.path) != budget.snapshot():
            raise GraphResumeError("STALE_BUDGET_REFUSED")
        self._commit()

    def _commit(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".part.json")
        payload = {"format": "graph-budget-v1", "budget": self.delegate.snapshot()}
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(self.path)
        if os.name != "nt":
            # Windows does not expose a portable directory fsync through os.
            descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def reserve(self, prompt_chars: int) -> int:
        attempt = self.delegate.reserve(prompt_chars)
        self._commit()
        return attempt

    def record_tokens(self, usage: dict) -> None:
        try:
            self.delegate.record_tokens(usage)
        finally:
            self._commit()

    def stop(self, reason: str) -> None:
        self.delegate.stop(reason)
        self._commit()

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.delegate.snapshot())


def validate_budget_continuation(previous: dict | None, current: dict | None) -> None:
    """Allow newer complete spending, never a reset or a larger run allowance."""
    if previous is None and current is None:
        return
    if previous is None or current is None:
        raise GraphResumeError("MODEL_BUDGET_CHANGED")
    if current.get("reported_cost_complete") is not True or current.get("stopped_reason"):
        raise GraphResumeError("UNRESOLVED_MODEL_SPENDING")
    for key in (
        "money_limit_cny",
        "http_attempt_limit",
        "input_char_limit",
        "reported_token_stop_threshold",
        "max_output_tokens",
        "pricing",
    ):
        if previous.get(key) != current.get(key):
            raise GraphResumeError("MODEL_BUDGET_CHANGED")
    old, new = previous.get("reservation_ledger"), current.get("reservation_ledger")
    if not isinstance(old, list) or not isinstance(new, list) or new[: len(old)] != old:
        raise GraphResumeError("MODEL_BUDGET_RESET_REFUSED")
