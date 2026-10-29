"""Exercise a real local HTTP listener, then shut it down; no remote model."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataverify.api import create_app


def main() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(), host="127.0.0.1", port=port, log_level="error", access_log=False
        )
    )
    worker = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 8
        while not server.started:
            if not worker.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Local HTTP server failed to start")
            time.sleep(0.04)
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=8
        ) as client:
            health = client.get("/health")
            run = client.post(
                "/runs", json={"case": "join_fanout", "provider": "offline", "max_repairs": 1}
            )
            docs = client.get("/openapi.json")
        assert health.status_code == run.status_code == docs.status_code == 200
        result = run.json()
        assert result["provider"] == "offline-scripted" and not result["real_model_used"]
        assert result["status"] == "verified" and result["value"] == 280
        queries = [
            event["result"] for event in result["trace"] if event.get("tool") == "query_metric"
        ]
        assert queries[0]["value"] == 540 and queries[0]["accepted_value"] is None
        report = {
            "local_http_verified": True,
            "bind_address": "127.0.0.1",
            "health_status": health.status_code,
            "runs_status": run.status_code,
            "openapi_status": docs.status_code,
            "provider": result["provider"],
            "real_model_used": False,
            "rejected_value": queries[0]["value"],
            "accepted_value": result["value"],
            "run_id": result["run_id"],
            "trace_path": result["trace_path"],
            "server_left_running": False,
        }
    finally:
        server.should_exit = True
        worker.join(timeout=5)
        listener.close()
    if worker.is_alive():
        raise RuntimeError("Local server did not shut down")
    target = ROOT / "reports/http_smoke.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
