"""Command-line entry points: demo, offline evaluation, and local API server."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from dataverify.database import seed_demo_database
from dataverify.engine import AgentRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "join_fanout",
    "distinct_fix",
    "clean_order_total",
    "clean_line_total",
    "unsupported",
)


def _runner() -> AgentRunner:
    database = PROJECT_ROOT / "data" / "demo.sqlite3"
    if not database.exists():
        seed_demo_database(database)
    traces = PROJECT_ROOT / "artifacts" / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    return AgentRunner(db_path=database, trace_dir=traces)


def _print_json(value: dict) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _evaluate() -> dict:
    """Check five fixed paths against independent, declared expectations.

    The planner is scripted and knows these cases. The report must therefore not
    be interpreted as an LLM benchmark, generalization test, or baseline gain.
    """

    runner = _runner()
    results = []
    for case in CASES:
        result = runner.run(
            question="统计订单总收入" if case != "clean_line_total" else "统计明细总收入",
            case=case,
            max_repairs=1,
            provider="offline",
        )
        expected_status = "unverified" if case == "unsupported" else "verified"
        expected_value = None if case == "unsupported" else 280
        passed = result["status"] == expected_status and (
            expected_value is None or result.get("value") == expected_value
        )
        results.append(
            {
                "case": case,
                "run_id": result["run_id"],
                "provider": result["provider"],
                "status": result["status"],
                "value": result.get("value"),
                "expected_status": expected_status,
                "expected_value": expected_value,
                "pipeline_check_passed": passed,
                "trace_path": result.get("trace_path"),
            }
        )

    report_path = PROJECT_ROOT / "reports" / "offline_evaluation.json"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "offline-scripted-pipeline-validation",
        "llm_evaluation": False,
        "limitation": (
            "固定用例与脚本化规划器，仅验证执行、核验、修正和拒绝核验路径；"
            "不能推断真实模型能力、泛化效果或相对基线提升。"
        ),
        "case_count": len(results),
        "pipeline_checks_passed": sum(item["pipeline_check_passed"] for item in results),
        "results": results,
        "report_path": str(report_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="数据分析 Agent 最小框架（默认离线验证）")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="运行一个本地演示并保存执行轨迹")
    demo.add_argument("--case", choices=CASES, default="join_fanout")
    demo.add_argument("--question", default="统计订单总收入")
    demo.add_argument("--max-repairs", type=int, choices=(0, 1, 2), default=1)
    demo.add_argument("--provider", choices=("offline", "qwen"), default="offline")

    commands.add_parser("evaluate", help="离线固定用例验证，不是大模型能力评测")

    serve = commands.add_parser("serve", help="仅在本机启动 FastAPI")
    serve.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
    serve.add_argument("--port", type=_port_number, default=8765)

    arguments = parser.parse_args(argv)
    if arguments.command == "serve":
        import uvicorn

        from dataverify.api import create_app

        uvicorn.run(
            create_app(
                db_path=PROJECT_ROOT / "data" / "demo.sqlite3",
                trace_dir=PROJECT_ROOT / "artifacts" / "traces",
            ),
            host=arguments.host,
            port=arguments.port,
        )
        return 0

    try:
        if arguments.command == "evaluate":
            result = _evaluate()
            _print_json(result)
            return 0 if result["pipeline_checks_passed"] == result["case_count"] else 1

        result = _runner().run(
            question=arguments.question,
            case=arguments.case,
            max_repairs=arguments.max_repairs,
            provider=arguments.provider,
        )
        _print_json(result)
        # A completed run can still have no verified result. Distinct exit
        # codes let shell/CI consumers reject unsupported or failed checks
        # instead of treating every successfully printed JSON as a success.
        return 0 if result["status"] == "verified" else 1
    except (ValueError, RuntimeError, OSError):
        _print_json(
            {
                "status": "error",
                "message": (
                    "任务参数、模型配置或本地存储不可用。千问需在本机配置 DASHSCOPE_API_KEY；"
                    "离线验证请使用 --provider offline。"
                ),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
