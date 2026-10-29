"""JSON-only continuation of witnesses and task-local feedback progress.

Accepted results are deliberately absent. A resumed execution must verify its
last SQL again before publication. This codec loads neither Python objects nor
evaluation answers and checks every retained witness against the bound task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .feedback_strategy import FeedbackStrategy
from .probe_runtime import ProbeData, ProbeTask
from .probe_testing import ProbePair, pair_is_applicable

if TYPE_CHECKING:
    from .probe_guard import ProbeGuard

_STATE_FIELDS = {"version", "binding", "regressions", "feedback_strategy", "terminal_code"}
_PAIR_FIELDS = {"property_id", "base", "added_orders", "added_lines", "record_count"}


@dataclass(frozen=True)
class GuardContinuation:
    """Validated in-memory values, never used as a checkpoint payload itself."""

    witnesses: tuple[ProbePair, ...]
    strategy: FeedbackStrategy
    terminal_code: str | None


def _load_pair(payload: Any, task: ProbeTask) -> ProbePair:
    if not isinstance(payload, dict) or set(payload) != _PAIR_FIELDS:
        raise ValueError("Invalid checkpoint witness fields")
    if type(payload["property_id"]) is not str or payload["property_id"] not in task.properties:
        raise ValueError("Checkpoint witness is outside the confirmed properties")
    base = payload["base"]
    if not isinstance(base, dict) or set(base) != {"orders", "lines"}:
        raise ValueError("Invalid checkpoint witness base")
    for rows in (base["orders"], base["lines"], payload["added_orders"], payload["added_lines"]):
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise ValueError("Checkpoint witness rows must be JSON arrays")
    pair = ProbePair(
        payload["property_id"], ProbeData(**base), payload["added_orders"], payload["added_lines"]
    )
    if type(payload["record_count"]) is not int or payload["record_count"] != pair.record_count:
        raise ValueError("Checkpoint witness record count differs")
    if not pair_is_applicable(task, pair):
        raise ValueError("Checkpoint witness does not meet the property preconditions")
    return pair


def validate_guard_state(state: Any, binding: dict[str, Any], task: ProbeTask) -> GuardContinuation:
    """Bound size, schema, identity and legal paired data before restoration."""
    if not isinstance(state, dict) or set(state) != _STATE_FIELDS:
        raise ValueError("Invalid guard checkpoint fields")
    try:
        encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise ValueError("Guard checkpoint must contain finite JSON values") from None
    if len(encoded.encode("utf-8")) > 1_000_000:
        raise ValueError("Guard checkpoint exceeds the bounded state size")
    if type(state["version"]) is not int or state["version"] != 1:
        raise ValueError("Unsupported guard checkpoint version")
    saved_binding = state["binding"]
    if (
        not isinstance(saved_binding, dict)
        or set(saved_binding) != set(binding)
        or type(saved_binding["seed"]) is not int
        or saved_binding != binding
    ):
        raise ValueError("Guard checkpoint task, data or configuration changed")
    witnesses = state["regressions"]
    if not isinstance(witnesses, list) or len(witnesses) > 32:
        raise ValueError("Guard checkpoint permits at most 32 witnesses")
    pairs = tuple(_load_pair(payload, task) for payload in witnesses)
    identities = [json.dumps(pair.payload(), sort_keys=True, ensure_ascii=False) for pair in pairs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate checkpoint witnesses")
    strategy = FeedbackStrategy.from_checkpoint(state["feedback_strategy"])
    if strategy.mode != binding["mode"]:
        raise ValueError("Checkpoint feedback mode differs")
    terminal = state["terminal_code"]
    if terminal is not None and terminal != "SOURCE_OR_TASK_CHANGED":
        raise ValueError("Unknown guard terminal state")
    return GuardContinuation(pairs, strategy, terminal)


def dump_guard(guard: ProbeGuard) -> dict[str, Any]:
    """Export a JSON copy through the guard's explicit continuation boundary."""
    return guard.checkpoint_state()


def restore_guard(
    state: dict[str, Any],
    task: ProbeTask,
    data: ProbeData,
    mode: str = "compact",
    seed: int = 17,
    *,
    feedback_policy: str = "capability_v2",
) -> ProbeGuard:
    """Build a fresh guard with retained evidence and no accepted publication."""
    from .probe_guard import ProbeGuard

    guard = ProbeGuard(task, data, mode, seed, feedback_policy=feedback_policy)
    guard.restore_checkpoint_state(state)
    return guard
