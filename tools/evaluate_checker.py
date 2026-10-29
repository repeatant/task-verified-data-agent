"""离线核验对照入口：不读取API配置，不联网，不覆盖现有演示数据库。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.evaluation import DEFAULT_SEEDS, FAMILIES, run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="裸SQL执行与统计口径核验的受控对照（不是LLM准确率）"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/checker_stress_evaluation.json"
    )
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(
            seeds=tuple(args.seeds),
            families=tuple(args.families),
            work_dir=ROOT / "artifacts/evaluation",
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(output)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
