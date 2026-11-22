"""Export the reviewed, synthetic duplicate-amount case without exposing model contexts.

The case is pinned by SHA-256. The public artifact contains only the question,
adopted changes and executed observations. Never copy the full model request or ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RECORD_SHA256 = "30261f2ec03a7bee468e7d617ffbe8725ef9af235b5bfd3aa82c9948e7c1d9a0"
ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "reports/dispute-recovery-v1/records/dispute-04--dispute_resolver.json"
PUBLIC = ROOT / "dataverify/workbench/data/duplicate-amount.json"


def export(path: Path) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECORD_SHA256:
        raise ValueError("Historical trace checksum mismatch")
    raw_case = json.loads(raw)
    rec = raw_case["record"]
    (observation,) = rec["query_observations"]
    publication = rec["publication"]
    final = next(
        o for o in rec["publication_observations"] if o["id"] == publication["observation_id"]
    )
    task = json.loads(raw_case["contexts"]["dispute_resolver"][0][1]["content"])["task"]
    evidence = []
    for i, replay in enumerate(rec["replays"]):
        check = next(
            check
            for check in replay["checks"]
            if any(item["name"] == "shared_diagnostic" for item in check["origins"])
        )
        vals = [check[role]["rows"][0][0] for role in ("candidate", "reference", "independent")]
        evidence.append(
            {
                "title": ("初始分歧", "修改候选 SQL 后", "修改参考计算后")[i],
                "scope": "历史实验的合成诊断样本",
                "detail": f"候选 SQL 返回 {vals[0]}，参考计算返回 {vals[1]}，独立核算返回 {vals[2]}。",
                "id": check["id"],
            }
        )
        if i == 0:
            evidence.append(
                {
                    "title": "读取实际行数",
                    "scope": "历史实验的合成诊断样本",
                    "detail": f"只读查询返回 {observation['rows'][0][0]} 行符合条件的记录。需求明确要求同额的不同记录分别计数。",
                    "sql": observation["sql"],
                    "id": observation["id"],
                }
            )
    revisions = []
    for action in rec["actions"]:
        if action["action"] == "revise":
            role = action["target"]
            old, new = rec["history"][role]
            revisions.append(
                {
                    "target": role,
                    "before": old["program"],
                    "after": new["program"],
                    "reason": new["reason"],
                    "evidence_ids": new["evidence_ids"],
                }
            )
    return {
        "id": "duplicate-amount",
        "status": rec["status"],
        "mode": "replay",
        "source_id": "historical-duplicate-fixture",
        "question": task["question"],
        "provenance": {
            "case": raw_case["row"]["case_id"],
            "method": raw_case["binding"]["method"],
            "model": raw_case["binding"]["model"],
            "raw_sha256": RECORD_SHA256,
            "protocol_sha256": raw_case["binding"]["protocol_digest"],
            "original_definitions": task["definitions"],
        },
        "result": {
            "sql": publication["sql"],
            "columns": final["columns"],
            "rows": publication["rows"],
            "scope": "2026-11-07 真实模型实验的固定合成诊断样本",
            "summary": "需求要求逐条计数。去除候选 SQL 和参考计算中的金额去重后，在该样本上得到 6 条。此处展示历史记录。",
            "evidence": evidence,
            "revisions": revisions,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ORIGINAL)
    parser.add_argument("--output", type=Path, default=PUBLIC)
    args = parser.parse_args()
    content = export(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
