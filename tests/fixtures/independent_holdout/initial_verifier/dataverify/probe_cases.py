"""Disclosed, controlled development cases for method-transfer experiments.

Nothing in this module is a production task or an unseen evaluation task.  The
candidate defects are deliberately authored; their labels do not imply that a
model generated them, or that their values already differ on ``source``.

The complete reference SQL and Python target evaluator live here, outside the
runtime verifier.  Runtime code must not import this module.  ``public_input``
is the allowlisted model input and never contains a reference, fault label,
expected value, or suggested repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .probe_runtime import ProbeData, ProbeTask


START = "2026-09-01T00:00:00"
END = "2026-10-01T00:00:00"
_PERIOD = (
    "l.status = 'settled' AND l.settled_at >= '2026-09-01T00:00:00' "
    "AND l.settled_at < '2026-10-01T00:00:00'"
)
_WHERE = " WHERE " + _PERIOD
_SCALAR = "SELECT COALESCE(SUM(l.amount), 0) AS revenue FROM order_lines l" + _WHERE
_SCALAR_JOIN = (
    "SELECT COALESCE(SUM(l.amount), 0) AS revenue FROM orders o "
    "INNER JOIN order_lines l ON o.order_id = l.order_id" + _WHERE
)
_ORDER = (
    "SELECT o.order_id, COALESCE(SUM(l.amount), 0) AS revenue FROM orders o "
    "LEFT JOIN order_lines l ON o.order_id = l.order_id AND " + _PERIOD
    + " GROUP BY o.order_id"
)
_CHANNEL = (
    "SELECT COALESCE(o.channel, '其他') AS channel, SUM(l.amount) AS revenue "
    "FROM orders o INNER JOIN order_lines l ON o.order_id = l.order_id"
    + _WHERE + " GROUP BY COALESCE(o.channel, '其他')"
)


@dataclass(frozen=True)
class Case:
    """Private evaluation case; frozen runtime task/data are owned separately."""

    case_id: str
    task: ProbeTask
    source: ProbeData
    candidate_sql: str
    reference_sql: str
    fault_label: str | None
    developer_note: str
    split: str = "controlled_development"

    def __post_init__(self) -> None:
        if self.split != "controlled_development":
            raise ValueError("These disclosed cases are controlled development data")
        for name in ("case_id", "candidate_sql", "reference_sql", "developer_note"):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.fault_label is not None and (
            type(self.fault_label) is not str or not self.fault_label.strip()
        ):
            raise ValueError("fault_label must be None or a nonempty string")

    def public_input(self) -> dict[str, Any]:
        """Model-facing fields only; intentionally exclude IDs and source rows."""
        task = self.task
        return {
            "question": task.question,
            "confirmed_task": {
                "properties": list(task.properties),
                "output_kind": task.output_kind,
                "start": task.start,
                "end": task.end,
                "status": task.status,
                "other_label": task.other_label,
            },
        }


@lru_cache(maxsize=1)
def get_cases() -> tuple[Case, ...]:
    """Construct the frozen suite, importing runtime data types only on demand."""
    from .probe_runtime import ProbeData, ProbeTask

    scalar = ProbeTask(
        "p01", "统计九月已结算明细的净收入；包括合法负金额退款。",
        ("exclude_end", "exclude_status", "include_start", "include_negative"),
        output_kind="scalar",
    )
    orders = ProbeTask(
        "p02", "按订单列出九月已结算明细的净收入；没有有效明细的订单也保留零值。",
        ("preserve_zero_orders",), output_kind="orders",
    )
    channels = ProbeTask(
        "p03", "按渠道统计九月已结算明细净收入；NULL 渠道归入其他并与其他合并。",
        ("merge_other_channels",), output_kind="channels",
    )
    # All timestamps are interior and all rows are settled. Several bad SQLs
    # intentionally look correct here; follow-up inputs must expose the defect.
    normal = ProbeData(
        ((1, 50_000, "其他"), (2, 22_000, "线下")),
        ((11, 1, 10_000, "settled", "2026-09-15T12:00:00"),
         (12, 1, 10_000, "settled", "2026-09-16T12:00:00"),
         (21, 2, 15_000, "settled", "2026-09-17T12:00:00")),
    )
    # This diagnostic fixture tests the independent target evaluator as well as
    # active probes. It is deliberately labelled and is not a natural error.
    mixed = ProbeData(
        ((1, 50_000, "其他"), (2, 22_000, None), (3, 0, "线下")),
        ((11, 1, 10_000, "settled", START),
         (12, 1, -3_700, "settled", "2026-09-15T12:00:00"),
         (21, 2, 15_000, "pending", "2026-09-17T12:00:00"),
         (22, 2, 20_000, "settled", END)),
    )
    data = (
        ("d01", scalar, normal, _SCALAR, _SCALAR, None, "正确标量查询。"),
        ("d02", scalar, normal, _SCALAR_JOIN, _SCALAR, None, "正确登记关联查询。"),
        ("d03", orders, normal, _ORDER, _ORDER, None, "正确保留零值订单的左关联。"),
        ("d04", channels, normal, _CHANNEL, _CHANNEL, None, "正确归一化渠道分组。"),
        ("d05", scalar, normal,
         "SELECT COALESCE(SUM(l.amount), 0) FROM order_lines l "
         "WHERE l.settled_at >= '2026-09-01T00:00:00' "
         "AND l.settled_at < '2026-10-01T00:00:00'",
         _SCALAR, "missing_status", "当前均为已结算；漏状态可能暂时数值相同。"),
        ("d06", scalar, normal, _SCALAR.replace("l.settled_at <", "l.settled_at <="),
         _SCALAR, "inclusive_end", "当前无上界记录；十月一日零点不得计入九月。"),
        ("d07", scalar, normal, _SCALAR.replace("l.settled_at >=", "l.settled_at >"),
         _SCALAR, "exclusive_start", "当前无下界记录；九月一日零点应计入。"),
        ("d08", scalar, normal, _SCALAR.replace("SUM(l.amount)", "SUM(DISTINCT l.amount)"),
         _SCALAR, "amount_distinct", "同额不同明细不可合并；已有静态规则应拦截。"),
        ("d09", scalar, normal, _SCALAR_JOIN.replace("SUM(l.amount)", "SUM(o.total)"),
         _SCALAR, "wrong_measure_source", "订单总额不是本任务的明细度量。"),
        ("d10", orders, normal,
         "SELECT o.order_id, COALESCE(SUM(l.amount), 0) AS revenue FROM orders o "
         "LEFT JOIN order_lines l ON o.order_id = l.order_id" + _WHERE
         + " GROUP BY o.order_id",
         _ORDER, "left_join_filtered_in_where", "当前每单有有效明细；新增零值订单会消失。"),
        ("d11", channels, normal, _CHANNEL.replace("GROUP BY COALESCE(o.channel, '其他')", "GROUP BY o.channel"),
         _CHANNEL, "display_group_collision", "NULL 与字面其他并存时产生相同展示键。"),
        ("d12", scalar, normal, _SCALAR + " AND l.amount > 0",
         _SCALAR, "positive_only", "额外排除退款；是否已被静态检查抓到由实测决定。"),
        ("d13", scalar, normal, _SCALAR + " AND l.amount >= 10000",
         _SCALAR, "undeclared_amount_floor", "额外金额门槛；当前每条恰好不低于门槛。"),
        ("d14", scalar, normal, _SCALAR.replace("l.status = 'settled'", "l.status != 'pending'"),
         _SCALAR, "wrong_status_predicate", "pending 被排除但 cancelled 仍会被计入。"),
        ("d15", scalar, mixed, _SCALAR + " AND l.amount > 0",
         _SCALAR, "positive_only_observed", "已有负金额，独立评分在源库就能发现差异。"),
        ("d16", scalar, mixed, _SCALAR, _SCALAR, None, "正确混合状态／边界／退款查询。"),
    )
    return tuple(Case(*entry) for entry in data)


def get_case(case_id: str) -> Case:
    for case in get_cases():
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def reference_result(case: Case) -> list[list[Any]]:
    """Exact target from Python row records, never executing either SQL.

    Dates are declared canonical ISO-second strings, so lexical half-open
    comparison agrees with the task's timestamp order. All money is integer
    minor units, avoiding rounding-based acceptance of an incorrect cent.
    """
    task, data = case.task, case.source
    eligible = [
        row for row in data.lines
        if row[3] == task.status and task.start <= row[4] < task.end
    ]
    if task.output_kind == "scalar":
        return [[sum(row[2] for row in eligible)]]
    if task.output_kind == "orders":
        totals = {row[0]: 0 for row in data.orders}
        for _, order_id, amount, _, _ in eligible:
            totals[order_id] += amount
        return [[key, value] for key, value in sorted(totals.items())]
    if task.output_kind == "channels":
        channels = {row[0]: row[2] for row in data.orders}
        totals: dict[str, int] = {}
        for _, order_id, amount, _, _ in eligible:
            channel = channels[order_id]
            key = task.other_label if channel is None else channel
            totals[key] = totals.get(key, 0) + amount
        return [[key, value] for key, value in sorted(totals.items())]
    raise ValueError("Unknown output kind; no target can be assigned")


def _number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _copy_rows(value: Any) -> list[list[Any]] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if any(not isinstance(row, (list, tuple)) for row in value):
        return None
    return [list(row) for row in value]


def grade_rows(actual: Any, expected: Any) -> dict[str, Any]:
    """Grade numeric scalar/grouped rows independently of the runtime verdict.

    Scalar shape is one row/one cell. Grouped results can be empty and row order
    is irrelevant; duplicate keys, missing zero rows, NULL/nonfinite values and
    boolean lookalikes are explicitly rejected. Exact numeric equality is used
    because target money uses integer minor units.
    """
    actual_rows, expected_rows = _copy_rows(actual), _copy_rows(expected)
    if expected_rows is None:
        raise ValueError("Invalid reference rows")
    if actual_rows is None:
        return {"correct": False, "code": "INVALID_ROWS"}
    if any(len(row) not in (1, 2) for row in expected_rows):
        raise ValueError("Invalid reference row width")
    scalar = bool(expected_rows and len(expected_rows[0]) == 1)
    if scalar:
        if len(expected_rows) != 1 or not _number(expected_rows[0][0]):
            raise ValueError("Invalid scalar reference")
        if len(actual_rows) != 1 or len(actual_rows[0]) != 1:
            return {"correct": False, "code": "SCALAR_SHAPE_MISMATCH"}
        value = actual_rows[0][0]
        if not _number(value):
            return {"correct": False, "code": "INVALID_MEASURE"}
        correct = value == expected_rows[0][0]
        return {"correct": correct, "code": "MATCH" if correct else "VALUE_MISMATCH"}
    if any(len(row) != 2 for row in expected_rows):
        raise ValueError("Inconsistent reference shape")

    def grouped(rows: list[list[Any]]):
        result: dict[tuple[type, Any], Any] = {}
        for row in rows:
            if len(row) != 2:
                return None, "GROUP_SHAPE_MISMATCH"
            key, amount = row
            if type(key) not in (int, str) or not _number(amount):
                return None, "INVALID_GROUP_VALUE"
            typed_key = (type(key), key)
            if typed_key in result:
                return None, "DUPLICATE_GROUP_KEY"
            result[typed_key] = amount
        return result, None

    target, target_error = grouped(expected_rows)
    if target_error:
        raise ValueError(f"Invalid grouped reference: {target_error}")
    observed, error = grouped(actual_rows)
    if error:
        return {"correct": False, "code": error}
    if observed.keys() != target.keys():
        return {"correct": False, "code": "GROUP_COVERAGE_MISMATCH"}
    correct = observed == target
    return {"correct": correct, "code": "MATCH" if correct else "VALUE_MISMATCH"}
