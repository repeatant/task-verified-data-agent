import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from dataverify.agent.graph_checkpoint import GraphResumeError
from dataverify.workbench.recovery import JournaledModel
from dataverify.workbench.store import RunStore


def test_restart_keeps_claimed_job_and_rejects_changed_or_duplicate_clarification(tmp_path):
    from dataverify.workbench.store import Conflict

    path = tmp_path / "jobs.sqlite3"
    first = RunStore(path)
    parent = first.create("数量", "demo-orders", "offline")
    assert first.claim() == parent
    restarted = RunStore(path)
    restarted.recover_on_start()
    assert restarted.get(parent)["status"] == "interrupted"
    assert restarted.claim() is None
    restarted.resume(parent)
    assert restarted.claim() == parent
    restarted.finish(parent, "needs_clarification", clarification="计数单位？")
    child = restarted.create(None, None, None, parent_id=parent, answer="明细")
    assert restarted.create(None, None, None, parent_id=parent, answer="明细") == child
    with pytest.raises(Conflict):
        restarted.create(None, None, None, parent_id=parent, answer="订单")
    assert restarted.get(parent)["child_id"] == child
    assert restarted.get(child)["definitions"] == "明细"


def test_dispute_restart_reuses_completed_reply_and_blocks_unknown_or_drift(tmp_path):
    path = tmp_path / "calls.sqlite3"
    messages = [{"role": "user", "content": "one"}]

    class Model:
        calls = 0

        async def generate(self, prompt):
            self.calls += 1
            if self.calls == 2:
                raise asyncio.CancelledError
            return {"action": "query", "sql": "SELECT 1"}

    model = Model()
    first = JournaledModel(model, path, binding={"version": 1})
    expected = asyncio.run(first.generate(messages))
    second = [{"role": "user", "content": "two"}]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(first.generate(second))
    restored = JournaledModel(model, path, binding={"version": 1})
    assert asyncio.run(restored.generate(messages)) == expected
    with pytest.raises(GraphResumeError, match="OUTCOME_UNKNOWN"):
        asyncio.run(restored.generate(second))
    assert model.calls == 2
    drifted = JournaledModel(model, path, binding={"version": 1})
    with pytest.raises(GraphResumeError, match="PROMPT_CHANGED"):
        asyncio.run(drifted.generate(second))
    with pytest.raises(GraphResumeError, match="BINDING_CHANGED"):
        JournaledModel(model, path, binding={"version": 2})


def test_http_submission_idempotency_history_and_clarification(tmp_path):
    from dataverify.workbench.app import PREFIX, create_app

    async def execute(*, definitions, on_stage, **kwargs):
        on_stage("discovery")
        if not definitions:
            return {"status": "needs_clarification", "clarification": "如何计数？"}
        return {"status": "reviewed", "rows": [[3]], "summary": definitions}

    def wait(client, run_id):
        for _ in range(200):
            response = client.get(f"{PREFIX}/runs/{run_id}").json()
            if response["status"] not in {"queued", "running"}:
                return response
            time.sleep(0.01)
        pytest.fail("worker did not settle")

    app = create_app(
        tmp_path,
        execute=execute,
        get_catalog=lambda: [{"id": "demo-orders"}],
        get_qwen_status=lambda: (False, "没有配置"),
    )
    with TestClient(app) as client:
        body = {"question": "数量", "source_id": "demo-orders", "mode": "offline"}
        headers = {"Idempotency-Key": "05ff8b49-5934-4723-80a9-d4f3ca570360"}
        response = client.post(f"{PREFIX}/runs", json=body, headers=headers)
        assert response.status_code == 202
        run_id = response.json()["id"]
        assert client.post(f"{PREFIX}/runs", json=body, headers=headers).json()["id"] == run_id
        assert (
            client.post(
                f"{PREFIX}/runs", json={**body, "question": "其他"}, headers=headers
            ).status_code
            == 409
        )
        assert wait(client, run_id)["clarification"] == "如何计数？"
        follow = client.post(f"{PREFIX}/runs/{run_id}/clarifications", json={"answer": "明细"})
        assert follow.status_code == 202
        assert wait(client, follow.json()["id"])["result"]["rows"] == [[3]]
        assert len(client.get(f"{PREFIX}/runs").json()["runs"]) == 2
        assert (
            client.post(f"{PREFIX}/runs", json={**body, "source_id": "/etc/file"}).status_code
            == 422
        )
        assert client.post(f"{PREFIX}/runs", json={**body, "mode": "qwen"}).status_code == 503


def test_query_permission_is_bound_to_retry_clarification_and_legacy_rows(tmp_path):
    from dataverify.workbench.store import Conflict

    path = tmp_path / "runs.sqlite3"
    store = RunStore(path)
    prior = store.create("老任务", "demo-orders", "qwen", allow_queries=True)
    old_demo = store.create("旧演示", "demo-orders", "offline", request_key="old-demo-key")
    with store.connection() as db:
        db.execute("ALTER TABLE runs DROP COLUMN allow_queries")
        db.execute("ALTER TABLE runs DROP COLUMN split_finish_evidence")
    store = RunStore(path)
    assert store.create("旧演示", "demo-orders", "offline", request_key="old-demo-key") == old_demo
    assert not store.get(old_demo)["split_finish_evidence"]
    assert store.get(prior)["allow_queries"] is True
    assert store.get(prior)["split_finish_evidence"] is False
    assert store.claim() == prior
    store.finish(prior, "needs_clarification", clarification="按哪个字段计数？")
    child = store.create(None, None, None, parent_id=prior, answer="订单")
    assert store.get(child)["allow_queries"] is True
    assert store.get(child)["split_finish_evidence"] is False
    demo = store.create("新离线演示", "demo-orders", "offline")
    assert store.get(demo)["split_finish_evidence"] is True
    enabled = store.create(
        "试用", "demo-orders", "qwen", split_finish_evidence=True, request_key="trial-key"
    )
    assert store.get(enabled)["split_finish_evidence"] is True
    with pytest.raises(Conflict):
        store.create(
            "试用", "demo-orders", "qwen", split_finish_evidence=False, request_key="trial-key"
        )
    new = store.create("一条新提问", "demo-orders", "qwen", request_key="test-key")
    assert store.get(new)["allow_queries"] is False
    with pytest.raises(Conflict):
        store.create(
            "一条新提问", "demo-orders", "qwen", request_key="test-key", allow_queries=True
        )
