"""Local server and small command-line client for the workbench task API."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description="DataVerify SQL 核验工作台")
    parser.add_argument("--port", type=int, default=8765)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("sources", help="列出可选数据源及离线示例问题")
    ask = commands.add_parser("ask", help="提交到已启动的本机工作台")
    ask.add_argument("question")
    ask.add_argument("--source", default="demo-orders", help="数据源编号，可通过 sources 查看")
    ask.add_argument("--mode", choices=["offline", "qwen"], default="offline")
    ask.add_argument("--allow-queries", action="store_true", help="分歧时允许额外调查查询")
    ask.add_argument(
        "--split-finish-evidence",
        action="store_true",
        default=None,
        help="分别提交当前版本签核依据和辅助调查引用",
    )
    get = commands.add_parser("get", help="读取同一任务的最新执行结果")
    get.add_argument("run_id")
    args = parser.parse_args()
    if args.command is None:
        import uvicorn

        uvicorn.run("dataverify.workbench.app:app", host="127.0.0.1", port=args.port)
        return 0

    address = f"http://127.0.0.1:{args.port}/api/workbench/runs"
    if args.command == "ask":
        payload = {
            "question": args.question,
            "source_id": args.source,
            "mode": args.mode,
            "allow_queries": args.allow_queries,
            "split_finish_evidence": args.split_finish_evidence,
        }
        req = Request(address, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    elif args.command == "sources":
        req = Request(f"http://127.0.0.1:{args.port}/api/workbench/config")
    else:
        from uuid import UUID

        try:
            run_id = UUID(args.run_id).hex
        except ValueError:
            parser.error("run_id 须为任务返回的 UUID")
        req = Request(f"{address}/{run_id}")
    try:
        # This client only connects to loopback, regardless of system proxy settings.
        with build_opener(ProxyHandler({})).open(req, timeout=15) as response:
            data = json.load(response)
            if args.command == "sources":
                for source in data["sources"]:
                    print(f"{source['id']} · {source['label']}")
                    for example in source["examples"]:
                        print(f"  {example}")
            else:
                print(json.dumps(data, ensure_ascii=False, indent=2))
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except URLError:
        print("无法连接本机工作台，请先启动服务。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
