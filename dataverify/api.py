"""Local HTTP interface for the data verification pipeline.

The default provider is a deterministic, scripted planner. Its results validate
the execution/check/repair pipeline; they are not measurements of an LLM.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from dataverify.database import seed_demo_database
from dataverify.engine import AgentRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]

Case = Literal[
    "join_fanout",
    "distinct_fix",
    "clean_order_total",
    "clean_line_total",
    "unsupported",
]
Provider = Literal["offline", "qwen"]


class RunRequest(BaseModel):
    """Only task controls are exposed; callers cannot select a database path."""

    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ] = "统计订单总收入"
    case: Case = "join_fanout"
    max_repairs: int = Field(default=1, ge=0, le=2, strict=True)
    provider: Provider = "offline"


def create_app(
    db_path: Path | None = None,
    trace_dir: Path | None = None,
) -> FastAPI:
    """Build the local app, allowing test-owned paths only through Python.

    Database creation happens at application startup. An existing database is
    reused and is never replaced with demo data by this interface.
    """

    database = Path(db_path) if db_path is not None else PROJECT_ROOT / "data" / "demo.sqlite3"
    traces = Path(trace_dir) if trace_dir is not None else PROJECT_ROOT / "artifacts" / "traces"

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if not database.exists():
            seed_demo_database(database)
        traces.mkdir(parents=True, exist_ok=True)
        application.state.runner = AgentRunner(db_path=database, trace_dir=traces)
        yield

    application = FastAPI(
        title="数据分析执行核验与反馈修正",
        version="0.1.0",
        description=(
            "本地最小框架。默认使用离线脚本化规划器验证工具执行、核验及反馈修正，"
            "不代表已经完成真实大模型能力评测。"
        ),
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError):
        # FastAPI's default response echoes each invalid input value. Keep the
        # useful location/type information without reflecting a submitted key
        # or other private task content back into responses and diagnostics.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"type": error["type"], "loc": list(error["loc"]), "msg": error["msg"]}
                    for error in exc.errors()
                ]
            },
        )

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "default_provider": "offline",
            "execution_mode": "offline-scripted",
            "real_model_validation": False,
            "dataset": "demo-orders-v1",
            "note": "默认仅验证脚本化规划器与执行核验管线；尚非真实模型效果评测。",
        }

    @application.post("/runs")
    def run_task(request: RunRequest) -> dict:
        try:
            return application.state.runner.run(
                question=request.question,
                case=request.case,
                max_repairs=request.max_repairs,
                provider=request.provider,
            )
        except ValueError as exc:
            # Provider configuration errors contain variable names, never key values.
            # Return a fixed message so a future SDK exception cannot expose secrets.
            if request.provider == "qwen":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "千问尚未就绪或请求不受支持；请在本机配置 DASHSCOPE_API_KEY，"
                        "并核对模型配置。不要通过此接口或聊天提交密钥。"
                    ),
                ) from exc
            raise HTTPException(status_code=400, detail="任务参数无效。") from exc
        except (RuntimeError, OSError) as exc:
            raise HTTPException(
                status_code=503,
                detail="运行环境、本地存储或模型服务暂不可用，请检查本机配置。",
            ) from exc

    return application


app = create_app()
