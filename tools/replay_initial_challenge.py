"""Replay the original, frozen verifier on independent challenge inputs.

Offline only. Intentionally loads the pre-fix source snapshot in a new Python
process, so the original misses remain demonstrable after the main code fixes.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
public = ROOT / "examples/independent_holdout"
if public.is_dir():
    dataset = public / "frozen_challenges.json"
    freeze = public / "verifier_manifest.json"
    snapshot = public / "initial_verifier"
    provenance = public / "provenance.json"
else:
    dataset = ROOT / "reports/independent_holdout_20261029/frozen_challenges.json"
    freeze = ROOT / "reports/holdout_verifier_freeze_20261029/manifest.json"
    snapshot = freeze.parent / "source"
    provenance = dataset.parent / "author_manifest.json"

manifest = json.loads(freeze.read_text(encoding="utf-8"))
author = json.loads(provenance.read_text(encoding="utf-8"))
if hashlib.sha256(dataset.read_bytes()).hexdigest() != author["challenge_sha256"]:
    raise RuntimeError("Independent challenge hash changed")
for name, expected in manifest["files"].items():
    if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Frozen initial verifier source changed")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(snapshot))

# Bind the package to the frozen snapshot before the evaluator adds its own
# project root to sys.path. Otherwise it can silently select the current code.
from dataverify.probe_guard import ProbeGuard

if not Path(inspect.getfile(ProbeGuard)).resolve().is_relative_to(snapshot.resolve()):
    raise RuntimeError("Initial replay imported the current verifier instead of frozen source")

from evaluate_independent_holdout import load_challenges, score

challenges = load_challenges(json.loads(dataset.read_text(encoding="utf-8")))
rows = [score(item) for item in challenges]
result = {
    "mode": "original_frozen_verifier_replay",
    "real_model_used": False,
    "scheduled": len(rows),
    "passed": sum(row["passed"] for row in rows),
    "semantic_wrong_releases": sum(row["semantic_wrong_release"] for row in rows),
    "miss_ids": [row["id"] for row in rows if row["semantic_wrong_release"]],
    "limitation": "Historical verifier evidence, intentionally not the current verification policy.",
}
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
