"""Serve the original DataVerify graphs through one local task queue."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictBool, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dataverify import __version__
from dataverify.agent.graph_checkpoint import GraphResumeError, exclusive_run

from . import runner
from .sample_source import bind_source
from .store import Conflict, RunStore

ROOT = Path(__file__).resolve().parents[2]
REPLAYS = {
    "duplicate-amount": "等额明细被重复去重",
    "split-signoff": "同一条正确 SQL 为何被拒",
    "penguins-2009": "2009 年有多少条企鹅观测",
}
PREFIX = "/api/workbench"
logger = logging.getLogger(__name__)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


def csv_result(run: dict) -> Response:
    result = run.get("result") or {}
    columns = result.get("columns")
    if run["status"] != "reviewed" or not columns:
        raise HTTPException(409, "此任务没有可导出的已核验结果。")

    def cell(value):
        # Keep numbers numeric but prevent text being evaluated by spreadsheets.
        if isinstance(value, str) and (
            value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
        ):
            return "'" + value
        return value

    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow([cell(column) for column in columns])
    for row in result["rows"]:
        values = [row[column] for column in columns] if isinstance(row, dict) else row
        writer.writerow([cell(value) for value in values])
    return Response(
        stream.getvalue().encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="dataverify-{run["id"]}.csv"'},
    )


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Text
    source_id: Text
    mode: Literal["offline", "qwen"]
    allow_queries: StrictBool = False
    split_finish_evidence: StrictBool | None = None


class ClarificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: Text


async def process_queue(store, directory, execute, wake):
    while True:
        wake.clear()
        run_id = store.claim()
        if run_id is None:
            await wake.wait()
            continue
        task = store.get(run_id)
        try:
            result = await execute(
                run_id=run_id,
                question=task["question"],
                definitions=task["definitions"],
                source_id=task["source_id"],
                mode=task["mode"],
                allow_queries=task["allow_queries"],
                split_finish_evidence=task["split_finish_evidence"],
                directory=directory / run_id,
                on_stage=lambda stage: store.stage(run_id, stage),
            )
            status = result["status"]
            if status not in {"reviewed", "needs_clarification", "inconclusive"}:
                raise ValueError("UNRECOGNIZED_GRAPH_STATUS")
            store.finish(run_id, status, result=result, clarification=result.get("clarification"))
        except GraphResumeError as exc:
            # Only known, fixed messages are suitable for the client.
            store.finish(run_id, "interrupted", error="检查点、账本或模型调用状态需核对，已暂停。")
            logger.warning("Run %s stopped at recovery guard (%s)", run_id, type(exc).__name__)
        except Exception as exc:
            logger.error("Run %s failed (%s)", run_id, type(exc).__name__)
            store.finish(run_id, "failed", error="任务运行失败，已保存任务编号和最近阶段。")


def create_app(state_dir=None, *, execute=None, get_catalog=None, get_qwen_status=None):
    state_dir = Path(state_dir) if state_dir else ROOT / "artifacts" / "workbench"
    execute = execute or runner.run_task
    get_catalog = get_catalog or runner.source_catalog
    get_qwen_status = get_qwen_status or runner.qwen_ready

    @asynccontextmanager
    async def lifespan(app):
        with exclusive_run(state_dir / "worker"):
            store = RunStore(state_dir / "runs.sqlite3")
            store.recover_on_start()
            wake = asyncio.Event()
            worker = asyncio.create_task(process_queue(store, state_dir / "tasks", execute, wake))
            app.state.store, app.state.wake = store, wake
            try:
                yield
            finally:
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker
                store.recover_on_start()

    app = FastAPI(title="DataVerify 工作台", version=__version__, lifespan=lifespan)
    local_hosts = {"localhost", "127.0.0.1", "testserver"}
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(local_hosts))

    @app.middleware("http")
    async def local_browser_actions(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and request.method not in {"GET", "HEAD", "OPTIONS"}:
            try:
                parsed = urlsplit(origin)
                allowed = parsed.scheme in {"http", "https"} and parsed.hostname in local_hosts
            except ValueError:
                allowed = False
            if not allowed:
                return JSONResponse(status_code=403, content={"detail": "请从本机工作台提交操作。"})
        return await call_next(request)

    @app.exception_handler(Conflict)
    async def conflict(_request: Request, exc: Conflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def missing(_request: Request, _exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": "任务不存在。"})

    def source_list():
        return [
            dict(src, label=src.get("label") or src.get("name") or src["id"])
            for src in get_catalog()
        ]

    def validate_mode(mode):
        if mode == "qwen":
            ready, reason = get_qwen_status()
            if not ready:
                raise HTTPException(503, reason)

    @lru_cache(maxsize=8)
    def preview_registered(source_id: str):
        # The catalog contains bundled, immutable sources. Cache only public sample rows.
        with TemporaryDirectory() as directory:
            preview = bind_source(source_id, Path(directory)).preview(rows_per_table=5)
            return {"tables": preview.named_tables()}

    def enqueue_response(run_id):
        app.state.wake.set()
        return app.state.store.get(run_id)

    @app.get(PREFIX + "/config")
    def config():
        ready, reason = get_qwen_status()
        return {
            "sources": source_list(),
            "modes": [
                {"id": "offline", "label": "离线示例", "available": True},
                {"id": "qwen", "label": "千问实时", "available": ready, "reason": reason},
            ],
            "replays": [{"id": key, "label": label} for key, label in REPLAYS.items()],
        }

    @app.get(PREFIX + "/sources/{source_id}/preview")
    def preview_source(source_id: str):
        if source_id not in {src["id"] for src in source_list()}:
            raise HTTPException(404, "数据源不存在。")
        return preview_registered(source_id)

    @app.get(PREFIX + "/replays/{replay_id}")
    def replay(replay_id: str):
        if replay_id not in REPLAYS:
            raise HTTPException(404, "回放不存在。")
        path = Path(__file__).with_name("data") / f"{replay_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get(PREFIX + "/replays/{replay_id}/result.csv")
    def export_replay(replay_id: str):
        return csv_result(replay(replay_id))

    @app.get(PREFIX + "/runs")
    def recent():
        return {"runs": app.state.store.recent()}

    @app.post(PREFIX + "/runs", status_code=202)
    async def submit(body: TaskRequest, idempotency_key: Annotated[UUID | None, Header()] = None):
        if body.source_id not in {src["id"] for src in source_list()}:
            raise HTTPException(422, "请选择已注册的数据源。")
        validate_mode(body.mode)
        run_id = app.state.store.create(
            body.question,
            body.source_id,
            body.mode,
            allow_queries=body.allow_queries,
            split_finish_evidence=body.split_finish_evidence,
            request_key=str(idempotency_key) if idempotency_key else None,
        )
        return enqueue_response(run_id)

    @app.get(PREFIX + "/runs/{run_id}")
    def get_run(run_id: str):
        return app.state.store.get(run_id)

    @app.get(PREFIX + "/runs/{run_id}/result.csv")
    def export_run(run_id: str):
        return csv_result(get_run(run_id))

    @app.post(PREFIX + "/runs/{run_id}/clarifications", status_code=202)
    async def clarify(run_id: str, body: ClarificationRequest):
        original = app.state.store.get(run_id)
        validate_mode(original["mode"])
        child = app.state.store.create(None, None, None, parent_id=run_id, answer=body.answer)
        return enqueue_response(child)

    @app.post(PREFIX + "/runs/{run_id}/resume", status_code=202)
    async def resume(run_id: str):
        original = app.state.store.get(run_id)
        validate_mode(original["mode"])
        app.state.store.resume(run_id)
        return enqueue_response(run_id)

    dist = ROOT / "frontend" / "dist"
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="workbench")
    return app


app = create_app()
