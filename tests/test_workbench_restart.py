"""Exercise the recovery boundary with an actual process exit between graph nodes."""

import asyncio
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from dataverify.workbench.runner import run_task
from dataverify.workbench.sample_source import COUNT_LINES, DATASET

CHILD = r"""
import asyncio
import os
import sys
from pathlib import Path
from dataverify.workbench.recovery import JournaledModel
from dataverify.workbench.runner import run_task
from dataverify.workbench.sample_source import COUNT_LINES, DATASET

record = JournaledModel._record
def exit_after_committed_revision(self, seq, outcome, payload):
    record(self, seq, outcome, payload)
    if seq == 1:
        os._exit(73)
JournaledModel._record = exit_after_committed_revision

asyncio.run(run_task(run_id="restart", question=COUNT_LINES, definitions="",
                     source_id=DATASET, mode="offline", directory=Path(sys.argv[1])))
"""


def test_process_exit_after_observation_and_revision_reconstructs_session(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        timeout=45,
    )
    assert result.returncode == 73
    path = tmp_path / "dispute-calls.sqlite3"
    with closing(sqlite3.connect(path)) as db:
        prefix = list(db.execute("SELECT seq,prompt,payload FROM calls ORDER BY seq"))
    assert [json.loads(row[2])["action"] for row in prefix] == ["query", "revise"]

    restored = asyncio.run(
        run_task(
            run_id="restart",
            question=COUNT_LINES,
            definitions="",
            source_id=DATASET,
            mode="offline",
            directory=tmp_path,
        )
    )
    assert restored["status"] == "reviewed"
    assert restored["rows"] == [[4]]
    assert [r["target"] for r in restored["revisions"]] == ["candidate", "reference"]
    with closing(sqlite3.connect(path)) as db:
        history = list(db.execute("SELECT seq,prompt,payload FROM calls ORDER BY seq"))
    assert history[:2] == prefix
    assert [json.loads(row[2])["action"] for row in history[2:]] == ["revise", "finish"]
