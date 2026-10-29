"""SQL核验组件的受控对照：Python行记录oracle独立于SQL解析器和核验器。

比较同一候选SQL的“只读裸执行”与“执行并核验”，不调用模型。
固定查询模板与合成故障只能评估核验机制，不能当作LLM准确率或生产收益。
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
import sqlite3
import tempfile
import time
from typing import Any

from .checker import CheckedQueryExecutor


FAMILIES = (
    "one_to_one", "fanout", "missing_order", "orphan_line", "cancelled_error",
    "null_measure", "text_measure", "duplicate_grain", "null_grain", "empty",
)
DEFAULT_SEEDS = (101, 211, 307, 401)


@dataclass(frozen=True)
class Fixture:
    family: str
    seed: int
    orders: tuple[tuple[Any, Any], ...]
    lines: tuple[tuple[Any, Any, Any], ...]


@dataclass(frozen=True)
class Candidate:
    name: str
    sql: str
    metric_id: str
    measure_source: str
    joined: bool = False
    value_distinct: bool = False


CANDIDATES = (
    Candidate("order_direct", "SELECT SUM(total) FROM orders", "order_revenue", "orders"),
    Candidate("order_join", "SELECT SUM(o.total) FROM orders o INNER JOIN order_lines l ON o.order_id=l.order_id", "order_revenue", "orders", joined=True),
    Candidate("order_distinct", "SELECT SUM(DISTINCT total) FROM orders", "order_revenue", "orders", value_distinct=True),
    Candidate("line_direct", "SELECT SUM(amount) FROM order_lines", "line_revenue", "order_lines"),
    Candidate("line_join", "SELECT SUM(l.amount) FROM order_lines l JOIN orders o ON l.order_id=o.order_id", "line_revenue", "order_lines", joined=True),
    Candidate("wrong_metric_source", "SELECT SUM(amount) FROM order_lines", "order_revenue", "order_lines"),
)


def make_fixture(family: str, seed: int) -> Fixture:
    """按种子改变金额、订单数量与关联分布；故障类别单独记录，可复现。"""
    if family not in FAMILIES:
        raise ValueError(f"Unknown fixture family: {family}")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    count = rng.randint(5, 12)
    cents = [rng.randint(100, 25000) for _ in range(count)]
    cents[1] = cents[0]  # 不同订单同额，SUM(DISTINCT 金额)不能替代按订单去重。
    orders: list[tuple[Any, Any]] = [(index + 1, amount / 100) for index, amount in enumerate(cents)]
    lines: list[tuple[Any, Any, Any]] = []
    for index, amount in enumerate(cents):
        pieces = rng.randint(2, 4) if family == "fanout" else 1
        if family == "missing_order" and index == 0:
            pieces = 0
        if family == "cancelled_error":
            # 重复首笔、遗漏同额第二笔：聚合数值碰巧相等，行粒度仍然错误。
            pieces = 2 if index == 0 else 0 if index == 1 else 1
        if pieces:
            quotient, remainder = divmod(amount, pieces)
            for part in range(pieces):
                lines.append((len(lines) + 1, index + 1, (quotient + (part < remainder)) / 100))
    if family == "orphan_line":
        lines.append((len(lines) + 1, count + 1000, rng.randint(100, 9000) / 100))
    elif family == "null_measure":
        orders[0] = (orders[0][0], None)
    elif family == "text_measure":
        orders[0] = (orders[0][0], "damaged-value")
    elif family == "duplicate_grain":
        orders.append(orders[0])
    elif family == "null_grain":
        orders[0] = (None, orders[0][1])
    elif family == "empty":
        orders, lines = [], []
    return Fixture(family, seed, tuple(orders), tuple(lines))


def _numeric(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def expected_result(fixture: Fixture, candidate: Candidate) -> dict[str, Any]:
    """独立oracle：从行记录检查金额、唯一粒度和关联后的完整覆盖。

    候选模板的语义在Candidate中人工声明；这里不解析SQL、不调用核验器，
    也不宣称能解释任意查询。WHERE/GROUP BY等不在本受控实验中。
    """
    target = "orders" if candidate.metric_id == "order_revenue" else "order_lines"
    records = fixture.orders if target == "orders" else fixture.lines
    ids = [row[0] for row in records]
    amounts = [row[1] if target == "orders" else row[2] for row in records]
    source_valid = (
        bool(records) and None not in ids and len(ids) == len(set(ids))
        and all(_numeric(value) for value in amounts)
    )
    value = math.fsum(amounts) if source_valid else None
    valid = source_valid and candidate.measure_source == target and not candidate.value_distinct
    if candidate.joined and valid:
        observed = []
        for order in fixture.orders:
            for line in fixture.lines:
                # 模拟SQL等值JOIN中NULL不与任何值相等的规则。
                if order[0] is not None and line[1] is not None and order[0] == line[1]:
                    observed.append(order[0] if target == "orders" else line[0])
        valid = Counter(observed) == Counter(ids)
    return {"should_accept": bool(valid), "metric_value": value, "source_rows": len(records)}


def _write_fixture(path: Path, fixture: Fixture) -> None:
    with closing(sqlite3.connect(path)) as db:
        # 不加主键/非空约束，刻意让损坏源数据进入测试，交给核验器拒绝。
        db.executescript(
            "CREATE TABLE orders(order_id INTEGER,total REAL);"
            "CREATE TABLE order_lines(line_id INTEGER,order_id INTEGER,amount REAL);"
        )
        db.executemany("INSERT INTO orders VALUES(?,?)", fixture.orders)
        db.executemany("INSERT INTO order_lines VALUES(?,?,?)", fixture.lines)
        db.commit()


def _raw_execute(path: Path, sql: str) -> dict[str, Any]:
    """无口径核验的对照；只执行本文件预定义的只读候选SQL。"""
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            result = db.execute(sql).fetchone()
        value = result[0] if result else None
        accepted = _numeric(value)
        return {"accepted": accepted, "value": value if accepted else None}
    except sqlite3.Error:
        return {"accepted": False, "value": None}


def _matches(value: Any, expected: Any) -> bool:
    return _numeric(value) and _numeric(expected) and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-8)


def run_evaluation(
    *, seeds: tuple[int, ...] = DEFAULT_SEEDS, families: tuple[str, ...] = FAMILIES,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """运行受控对照；夹具临时创建并清理，逐例记录SQL、预期和反例。"""
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain distinct integers")
    if not families or any(family not in FAMILIES for family in families) or len(set(families)) != len(families):
        raise ValueError("families must be distinct registered fixture families")
    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    unchanged = True
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="checker-eval-", dir=work_dir) as temporary:
        for seed in seeds:
            for family in families:
                fixture = make_fixture(family, seed)
                database = Path(temporary) / f"{family}-{seed}.sqlite3"
                _write_fixture(database, fixture)
                digest = hashlib.sha256(database.read_bytes()).hexdigest()
                executor = CheckedQueryExecutor(database)
                for candidate in CANDIDATES:
                    expectation = expected_result(fixture, candidate)
                    baseline = _raw_execute(database, candidate.sql)
                    began_check = time.perf_counter()
                    checked = executor.execute(candidate.sql, candidate.metric_id)
                    check_ms = (time.perf_counter() - began_check) * 1000
                    accepted = checked["verification_status"] == "verified" and checked["accepted_value"] is not None
                    baseline_correct = expectation["should_accept"] and _matches(baseline["value"], expectation["metric_value"])
                    guarded_correct = expectation["should_accept"] and _matches(checked["accepted_value"], expectation["metric_value"])
                    cases.append({
                        "case_id": f"{seed}:{family}:{candidate.name}",
                        "seed": seed, "family": family, "candidate": candidate.name,
                        "sql": candidate.sql, "metric_id": candidate.metric_id,
                        "expected": expectation,
                        "baseline": {**baseline, "wrong_result_released": baseline["accepted"] and not baseline_correct},
                        "guarded": {
                            "accepted": accepted, "value": checked["accepted_value"],
                            "verification_status": checked["verification_status"],
                            "error_code": checked["error_code"], "evidence": checked["evidence"],
                            "wrong_result_released": accepted and not guarded_correct,
                            "valid_query_rejected": expectation["should_accept"] and not guarded_correct,
                            "elapsed_ms": round(check_ms, 4),
                        },
                        "numeric_match_but_invalid": (
                            not expectation["should_accept"] and _matches(baseline["value"], expectation["metric_value"])
                        ),
                    })
                unchanged = unchanged and digest == hashlib.sha256(database.read_bytes()).hexdigest()
    valid_count = sum(case["expected"]["should_accept"] for case in cases)
    baseline_wrong = sum(case["baseline"]["wrong_result_released"] for case in cases)
    guarded_wrong = sum(case["guarded"]["wrong_result_released"] for case in cases)
    false_reject = sum(case["guarded"]["valid_query_rejected"] for case in cases)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "controlled-sql-fault-injection-checker-comparison",
        "real_model_used": False, "llm_accuracy_benchmark": False,
        "scope": "registered all-row SUM metrics; zero or one registered INNER JOIN",
        "limitation": (
            "随机合成数据库与固定候选SQL模板的受控故障对照；预期口径由Python行记录独立计算。"
            "不评估自然语言意图、真实模型泛化、修正成功率或生产收益，不能宣称LLM准确率提升。"
        ),
        "seeds": list(seeds), "families": list(families),
        "dataset_count": len(seeds) * len(families), "candidate_count": len(cases),
        "valid_candidate_count": valid_count, "invalid_candidate_count": len(cases) - valid_count,
        "baseline_wrong_result_releases": baseline_wrong,
        "guarded_wrong_result_releases": guarded_wrong,
        "guarded_valid_query_rejections": false_reject,
        "numeric_match_but_invalid_count": sum(case["numeric_match_but_invalid"] for case in cases),
        "source_databases_unchanged": unchanged,
        "passed": unchanged and valid_count > 0 and guarded_wrong == 0 and false_reject == 0,
        "elapsed_seconds": round(time.perf_counter() - started, 4), "cases": cases,
    }
