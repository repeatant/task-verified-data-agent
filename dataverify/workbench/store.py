"""Transactional state for the local, single-worker application."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class Conflict(ValueError):
    """The submitted action does not apply to the current task state."""


class RunStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, source_id TEXT NOT NULL,
                mode TEXT NOT NULL, definitions TEXT NOT NULL, status TEXT NOT NULL,
                stage TEXT, result TEXT, clarification TEXT, error TEXT,
                parent_id TEXT UNIQUE REFERENCES runs(id), request_key TEXT UNIQUE,
                allow_queries INTEGER NOT NULL DEFAULT 0,
                split_finish_evidence INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )""")
            columns = {r[1] for r in db.execute("PRAGMA table_info(runs)")}
            if "allow_queries" not in columns:
                # Tasks created with the first UI allowed investigation queries.
                db.execute("ALTER TABLE runs ADD COLUMN allow_queries INTEGER NOT NULL DEFAULT 1")
            if "split_finish_evidence" not in columns:
                db.execute(
                    "ALTER TABLE runs ADD COLUMN split_finish_evidence INTEGER NOT NULL DEFAULT 0"
                )

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(
        self,
        question,
        source_id,
        mode,
        *,
        allow_queries=False,
        split_finish_evidence=None,
        request_key=None,
        parent_id=None,
        answer=None,
    ):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            definitions = ""
            if parent_id:
                parent = db.execute("SELECT * FROM runs WHERE id=?", (parent_id,)).fetchone()
                if parent is None:
                    raise KeyError(parent_id)
                if parent["status"] != "needs_clarification":
                    raise Conflict("此任务当前不需要补充口径。")
                definitions = "\n".join(filter(None, (parent["definitions"], answer)))
                if len(definitions) > 2000:
                    raise Conflict("累计补充口径超过 2000 字，请用完整口径新建任务。")
                previous = db.execute(
                    "SELECT * FROM runs WHERE parent_id=?", (parent_id,)
                ).fetchone()
                if previous:
                    if previous["definitions"] != definitions:
                        raise Conflict("此版本已提交过另一份口径，请查看后续任务。")
                    return previous["id"]
                question, source_id, mode = parent["question"], parent["source_id"], parent["mode"]
                allow_queries = bool(parent["allow_queries"])
                split_finish_evidence = bool(parent["split_finish_evidence"])
            if mode == "offline":
                allow_queries = True  # The authored demonstration includes its investigation.
            if request_key:
                previous = db.execute(
                    "SELECT * FROM runs WHERE request_key=?", (request_key,)
                ).fetchone()
                if previous:
                    same_request = (
                        previous["question"],
                        previous["source_id"],
                        previous["mode"],
                        previous["allow_queries"],
                    ) == (question, source_id, mode, allow_queries)
                    same_signoff = split_finish_evidence is None or (
                        bool(previous["split_finish_evidence"]) == split_finish_evidence
                    )
                    if not same_request or not same_signoff:
                        raise Conflict("重复请求标识对应另一个任务。")
                    return previous["id"]
            if split_finish_evidence is None:
                split_finish_evidence = mode == "offline"
            run_id = uuid4().hex
            db.execute(
                """INSERT INTO runs(id,question,source_id,mode,definitions,status,parent_id,
                       request_key,allow_queries,split_finish_evidence)
                       VALUES (?,?,?,?,?,'queued',?,?,?,?)""",
                (
                    run_id,
                    question,
                    source_id,
                    mode,
                    definitions,
                    parent_id,
                    request_key,
                    int(allow_queries),
                    int(split_finish_evidence),
                ),
            )
            return run_id

    def get(self, run_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            run = dict(row)
            run.pop("request_key")
            run["allow_queries"] = bool(run["allow_queries"])
            run["split_finish_evidence"] = bool(run["split_finish_evidence"])
            run["result"] = json.loads(run["result"]) if run["result"] else None
            child = db.execute("SELECT id FROM runs WHERE parent_id=?", (run_id,)).fetchone()
            run["child_id"] = child[0] if child else None
            return run

    def recent(self):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute("""SELECT id,question,source_id,mode,status,stage,
                       created_at,parent_id FROM runs ORDER BY rowid DESC LIMIT 50""")
            ]

    def claim(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM runs WHERE status='queued' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if row:
                db.execute("UPDATE runs SET status='running',error=NULL WHERE id=?", (row[0],))
                return row[0]
            return None

    def recover_on_start(self):
        with self.connection() as db:
            db.execute("""UPDATE runs SET status='interrupted',error='服务中断，可检查并恢复已保存的进度。'
                       WHERE status='running'""")

    def resume(self, run_id):
        with self.connection() as db:
            cur = db.execute(
                "UPDATE runs SET status='queued',error=NULL WHERE id=? AND status='interrupted'",
                (run_id,),
            )
            if not cur.rowcount:
                raise Conflict("只有中断的任务可以恢复。")

    def stage(self, run_id, stage):
        with self.connection() as db:
            db.execute("UPDATE runs SET stage=? WHERE id=? AND status='running'", (stage, run_id))

    def finish(self, run_id, status, *, result=None, error=None, clarification=None):
        with self.connection() as db:
            db.execute(
                """UPDATE runs SET status=?,result=?,error=?,clarification=?
                       WHERE id=? AND status='running'""",
                (
                    status,
                    json.dumps(result, ensure_ascii=False, allow_nan=False)
                    if result is not None
                    else None,
                    error,
                    clarification,
                    run_id,
                ),
            )
