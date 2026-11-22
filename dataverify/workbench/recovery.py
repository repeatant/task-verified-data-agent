"""Rebuild a dispute session from durable model replies and live read-only executions.

LangGraph nodes in this graph share a mutable DisputeSession. Replaying its
validated actions reconstructs that object, including rejected revisions.
Every prompt must still match. An interrupted external request is never retried.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from dataverify.agent.graph_checkpoint import GraphResumeError
from dataverify.agent.graph_model import GraphModelError
from dataverify.runtime.reporting import digest_json


class JournaledModel:
    def __init__(self, model, path: Path, *, binding):
        self.model = model
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.position = 0
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY, digest TEXT)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (seq INTEGER PRIMARY KEY, "
                "prompt TEXT NOT NULL, outcome TEXT, payload TEXT)"
            )
            identity = digest_json(binding)
            db.execute("INSERT OR IGNORE INTO identity VALUES (1,?)", (identity,))
            if db.execute("SELECT digest FROM identity WHERE id=1").fetchone()[0] != identity:
                raise GraphResumeError("DISPUTE_RESUME_BINDING_CHANGED")

    @property
    def budget(self):
        return getattr(self.model, "budget", None)

    async def generate(self, messages):
        seq = self.position
        key = digest_json(messages)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT prompt,outcome,payload FROM calls WHERE seq=?", (seq,)
            ).fetchone()
            if row is not None:
                if row[0] != key:
                    raise GraphResumeError("DISPUTE_RESUME_PROMPT_CHANGED")
                if row[1] is None:
                    raise GraphResumeError("DISPUTE_REQUEST_OUTCOME_UNKNOWN")
                self.position += 1
                reply = json.loads(row[2])
                if row[1] == "error":
                    raise GraphModelError(reply["code"], retryable=reply["retryable"])
                return reply
            db.execute("INSERT INTO calls(seq,prompt) VALUES (?,?)", (seq, key))

        # The transaction above commits before a request can leave this process.
        # A crash after dispatch leaves a pending record requiring reconciliation.
        try:
            reply = await self.model.generate(messages)
        except GraphModelError as exc:
            self._record(seq, "error", {"code": exc.code, "retryable": exc.retryable})
            raise
        self._record(seq, "ok", reply)
        return reply

    def _record(self, seq, outcome, payload):
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE calls SET outcome=?,payload=? WHERE seq=?", (outcome, encoded, seq))
        self.position += 1
