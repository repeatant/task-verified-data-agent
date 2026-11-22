"""Catalog of host-registered datasets and their per-task SQLite bindings."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from dataverify.verification.probe_runtime import TABLE_SCHEMA
from dataverify.verification.relational_data import SQLiteSource

from . import penguin_source

DATASET = "demo-orders"
COUNT_LINES = "统计所有已结算明细条数，相同金额的不同明细分别计数。"
NET_AMOUNT = "统计所有已结算明细的净金额，退款按负值计入。"
AMBIGUOUS_COUNT = "统计已结算的数量。"
DEFINITIONS = (
    "金额单位为分，负数表示退款。settled 表示已结算，pending 表示待结算，"
    "cancelled 表示已取消。订单的 total 与明细金额的合计可能不同。"
)


def source_catalog() -> list[dict]:
    return [
        {
            "id": DATASET,
            "name": "订单与结算明细（示例库）",
            "description": "本地合成的 3 张订单、6 条明细，包含退款、等额明细和多种结算状态。",
            "schema": {table: list(cols) for table, cols in TABLE_SCHEMA.items()},
            "definitions": DEFINITIONS,
            "examples": [COUNT_LINES, NET_AMOUNT, AMBIGUOUS_COUNT],
        },
        penguin_source.catalog_entry(),
    ]


def get_source(source_id: str) -> dict:
    for source in source_catalog():
        if source["id"] == source_id:
            return source
    raise ValueError("UNKNOWN_REGISTERED_SOURCE")


def bind_source(source_id: str, directory: Path) -> SQLiteSource:
    if source_id == penguin_source.DATASET:
        return penguin_source.bind(directory)
    if source_id != DATASET:
        raise ValueError("UNKNOWN_REGISTERED_SOURCE")
    path = directory / "orders.sqlite3"
    if not path.exists():
        # All rows are synthetic, including the repeated amount and refund. Build
        # a complete copy before exposing it to a resumed run.
        temporary = directory / "orders.build.sqlite3"
        temporary.unlink(missing_ok=True)
        with closing(sqlite3.connect(temporary)) as db, db:
            db.executescript("""
                PRAGMA foreign_keys=ON;
                CREATE TABLE orders(order_id INTEGER PRIMARY KEY, total INTEGER, channel TEXT);
                CREATE TABLE order_lines(
                    line_id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(order_id),
                    amount INTEGER, status TEXT, settled_at TEXT
                );
                INSERT INTO orders VALUES (1,250,'web'),(2,90,'store'),(3,75,'web');
                INSERT INTO order_lines VALUES
                    (1,1,100,'settled','2026-09-03T08:00:00'),
                    (2,1,100,'settled','2026-09-03T08:15:00'),
                    (3,1,-25,'settled','2026-09-04T10:00:00'),
                    (4,2,90,'settled','2026-09-05T10:00:00'),
                    (5,2,20,'cancelled','2026-09-06T12:00:00'),
                    (6,3,75,'pending','2026-10-01T09:00:00');
            """)
        temporary.replace(path)
    return SQLiteSource.from_path(path)
