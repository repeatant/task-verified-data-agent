"""Small controlled fixture. Existing files are deliberately never overwritten."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path


def seed_demo_database(path: Path) -> Path:
    """Create three orders with different line counts, including equal totals.

    Order revenue = 100 + 100 + 80 = 280. Joining and summing the
    order-level total gives 540; SUM(DISTINCT total) incorrectly gives 180.
    This is experimental data, not a production dataset or an LLM benchmark.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return path
    os.close(fd)
    try:
        with sqlite3.connect(path) as db:
            db.executescript(
                """
                CREATE TABLE orders (
                    order_id INTEGER PRIMARY KEY,
                    total REAL NOT NULL
                );
                CREATE TABLE order_lines (
                    line_id INTEGER PRIMARY KEY,
                    order_id INTEGER NOT NULL REFERENCES orders(order_id),
                    amount REAL NOT NULL
                );
                INSERT INTO orders VALUES (1,100),(2,100),(3,80);
                INSERT INTO order_lines VALUES
                    (1,1,40),(2,1,60),(3,2,100),
                    (4,3,20),(5,3,30),(6,3,30);
                """
            )
    except Exception:
        # Only remove the file created exclusively by this invocation.
        path.unlink(missing_ok=True)
        raise
    return path


def database_sha256(path: Path) -> str:
    """Physical source fingerprint, including committed data in a visible WAL.

    This is a conservative before/after change detector, not a cross-run
    logical version identifier. Per-query consistency is a separate read
    transaction in CheckedQueryExecutor.
    """
    digest = hashlib.sha256()
    path = Path(path)
    for suffix in ("", "-wal"):
        candidate = Path(str(path) + suffix)
        digest.update(suffix.encode("ascii") + b"\0")
        try:
            with candidate.open("rb") as stream:
                digest.update(b"present\0")
                while block := stream.read(65536):
                    digest.update(block)
        except FileNotFoundError:
            if not suffix:
                raise
            digest.update(b"absent\0")
    return digest.hexdigest()
