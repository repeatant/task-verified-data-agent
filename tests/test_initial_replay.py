"""The historical replay must really import its frozen verifier in a new process."""

import json
import subprocess
import sys
from pathlib import Path


def test_initial_replay_preserves_original_misses_after_current_code_changes():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", str(root / "tools/replay_initial_challenge.py")],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    report = json.loads(result.stdout)
    assert report["mode"] == "original_frozen_verifier_replay"
    assert (report["passed"], report["scheduled"]) == (21, 24)
    assert report["miss_ids"] == ["h09", "h13", "h15"]
    assert report["real_model_used"] is False
