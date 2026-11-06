"""Authored offline decisions for integration checks, never model-quality evidence."""

from __future__ import annotations

import json
from copy import deepcopy

from dataverify.agent.discovery_model import (
    QwenReviewModel,
    ScriptedReviewModel,
    parse_review_action,
)
from dataverify.agent.graph_model import _validated_messages


class InjectedFirstReviewModel:
    """Inject one disclosed reference plan; subsequent decisions use the delegate."""

    client_kind = "injected_first_reference_then_real_qwen_review_http"

    def __init__(self, delegate: QwenReviewModel, plan: dict, *, already_injected: bool = False):
        self.delegate = delegate
        self._plan = parse_review_action(plan)
        if self._plan["action"] != "plan":
            raise ValueError("INJECTED_REFERENCE_MUST_BE_PLAN")
        self._injected = already_injected
        self.requests: list[list[dict]] = []
        self.turn_origins: list[str] = []

    @property
    def budget(self):
        return self.delegate.budget

    @property
    def model_name(self):
        return self.delegate.model_name

    @property
    def http_events(self):
        return self.delegate.http_events

    @property
    def usage(self):
        return self.delegate.usage

    async def generate(self, messages):
        self.requests.append(_validated_messages(messages))
        if not self._injected:
            self._injected = True
            self.turn_origins.append("injected_reference_no_api_call")
            return deepcopy(self._plan)
        self.turn_origins.append("real_qwen_review_http")
        return await self.delegate.generate(messages)

    async def aclose(self):
        await self.delegate.aclose()


class AuthoredReviewModel(ScriptedReviewModel):
    """Replay fixed decisions while binding them to runtime-generated evidence IDs.

    The test author chooses the target beforehand. Only the opaque check IDs are
    taken from tool output; this class performs no independent semantic reasoning.
    """

    client_kind = "offline_authored_attribution"

    async def generate(self, messages):
        index = len(self.requests)
        if index < len(self.actions):
            action = self.actions[index]
            if action.get("action") == "attribute" and action.get("checks") == ["$evidence"]:
                context = json.loads(messages[-1]["content"])
                summary = context["evidence"]["summary"]
                key = action["target"] + "_difference_ids"
                action["checks"] = summary.get(key, [])[:2]
        return await super().generate(messages)


def authored_attribution(task, target: str) -> dict:
    """An explicitly authored decision, with actual evidence IDs filled on replay."""
    return {
        "action": "attribute",
        "target": target,
        "reason": "Authored integration scenario; target selected by the test author.",
        "basis": [{"source": "question", "quote": task.question}],
        "checks": ["$evidence"] if target != "ambiguous" else [],
    }
