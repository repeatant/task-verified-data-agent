"""Project two pinned synthetic Qwen decisions to a public replay without exposing prompts."""

import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "reports/split-signoff-v1/records"
PUBLIC = ROOT / "dataverify/workbench/data/split-signoff.json"
HASHES = {
    "old": "f69ca61ac8177fecc8a0025826d18b1f5bef9aeae03d8b441c9ebf82991febde",
    "split": "9ab3ee5b7f4ba725c193e35f82d11d7d6c8adb354691fd92352f9d738df8ef7d",
}


def export():
    pair = {}
    for arm, digest in HASHES.items():
        path = RECORDS / f"terminal-dispute-03-{arm}.json"
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"Changed {arm} arm")
        pair[arm] = json.loads(raw)
    old, new = [pair[arm]["record"] for arm in ("old", "split")]
    context = pair["split"]["final_prompt"][1]
    if context != pair["old"]["final_prompt"][1]:
        raise ValueError("Cannot show this pair as sharing the same evidence")
    pub = new["publication"]
    (signoff,) = pub["signoff_ids"]
    (support,) = pub["supporting_ids"]
    check = next(c for c in new["replay"]["checks"] if c["id"] == signoff)
    query = next(o for o in new["query_observations"] if o["id"] == support)
    final = next(o for o in new["publication_observations"] if o["id"] == pub["observation_id"])
    old_ids = old["actions"][-1]["evidence_ids"]
    if old_ids != [signoff, support] or old["publication"]["rows"] is not None:
        raise ValueError("Legacy mixed citation rejection changed")
    code = old["trace"][-1]["code"]
    return {
        "id": "split-signoff",
        "status": new["status"],
        "mode": "replay",
        "source_id": "synthetic-september-settlement",
        "question": "统计九月已结算明细的收入，排除待处理和已取消的明细。",
        "provenance": {
            "case": "dispute-03",
            "model": pair["split"]["event"]["model"],
            "record_sha256": HASHES,
            "protocol_sha256": pair["split"]["binding"]["protocol"],
            "question": json.loads(context["content"])["task"]["question"],
        },
        "result": {
            "sql": pub["sql"],
            "columns": final["columns"],
            "rows": pub["rows"],
            "finish_citations": {"signoff_ids": [signoff], "supporting_ids": [support]},
            "summary": "两次真实千问调用拿到相同的候选 SQL、核算程序和执行证据。旧协议因混合引用而拒绝交付；分栏后，相同 SQL 在完整合成库上执行并交付。",
            "scope": "2026-11-21 冻结实验。预览用来比较三份程序，完整合成库用来交付，二者数据量不同。",
            "evidence": [
                {
                    "title": "旧协议如何拒绝",
                    "detail": "最终引用 "
                    + "、".join(old_ids)
                    + "。其中调查查询不能签核，整项交付被拒绝。",
                    "scope": code,
                },
                {
                    "title": "三份程序在预览上核验",
                    "detail": "同一版本在同一份预览上的执行结果。",
                    "id": signoff,
                    "sample": {
                        role: check[role] for role in ("candidate", "reference", "independent")
                    },
                },
                {
                    "title": "辅助查询核对了什么",
                    "detail": "在完整合成库上运行候选 SQL，查看九月已结算明细的合计金额。",
                    "id": support,
                    "sql": query["sql"],
                    "rows": query["rows"],
                },
                {
                    "title": "最终交付",
                    "detail": "通过签核后在同一完整合成库重新执行。候选 SQL 另经 25 份评分数据检验。",
                    "id": final["id"],
                    "sql": pub["sql"],
                    "rows": pub["rows"],
                },
            ],
        },
    }


if __name__ == "__main__":
    PUBLIC.write_text(
        json.dumps(export(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
