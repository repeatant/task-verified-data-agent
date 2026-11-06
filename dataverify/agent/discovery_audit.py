"""Blind recalculation and evidence-gated attribution for the discovery graph.

This mixin owns no model truth. It preserves the original candidate until an
executed disagreement and a supported attribution authorize a specific change.
Reference revisions are append-only versions, never edits to make a test pass.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from dataverify.runtime.reporting import digest_json
from dataverify.verification.discovery_runtime import validate_plan
from dataverify.verification.reference_audit import (
    authorize_decision,
    find_distinguishing_fixture,
    inspect_dispute,
    merge_fixtures,
)

from .discovery_model import (
    ATTRIBUTION_PROMPT,
    AUDIT_PROMPT,
    parse_review_action,
    plan_validation_feedback,
)
from .graph_model import GraphModelError


def message(role: str, value: Any) -> dict:
    return {
        "role": role,
        "content": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
    }


def budget_snapshot(model) -> dict | None:
    budget = getattr(model, "budget", None)
    return budget.snapshot() if budget is not None else None


def evidence_agrees(evidence: dict) -> bool:
    """Agreement on retained executions, not a claim of semantic correctness."""
    summary = evidence.get("summary", {})
    return bool(
        summary.get("all_executed")
        and summary.get("candidate_matches_reference")
        and summary.get("candidate_matches_independent")
        and summary.get("reference_matches_independent")
        and summary.get("candidate_matches_confirmed") is not False
        and summary.get("reference_matches_confirmed") is not False
        and summary.get("independent_matches_confirmed") is not False
    )


class ReferenceAuditNodes:
    """Node methods share the bounded models/state owned by DiscoveryNodes."""

    def prepare_audit(self, state):
        return {"pending": "audit", "audit_attempts": state["audit_attempts"] + 1}

    def blind_audit_messages(self, state) -> list[dict]:
        # Raw rows may be observed; neither program, result nor suggestive fixture
        # labels enter this context. Caller-confirmed expected values stay local.
        samples = [self.source.to_dict()] + [f["data"] for f in state["fixtures"][:2]]
        return [
            message("system", AUDIT_PROMPT),
            message("user", {"task": self.task.public_input(), "data_samples": samples}),
        ]

    async def audit(self, state):
        messages = state["audit_messages"] or self.blind_audit_messages(state)
        action = None
        try:
            action = parse_review_action(await self.auditor.generate(deepcopy(messages)))
            if action["action"] == "clarify":
                return {
                    "pending": None,
                    "status": "needs_clarification",
                    "reason": action["reason"],
                    "budget": budget_snapshot(self.auditor),
                    "audit_messages": messages + [message("assistant", action)],
                }
            plan = validate_plan(action, self.task)
            if plan["assumptions"]:
                return {
                    "pending": None,
                    "status": "needs_clarification",
                    "reason": "; ".join(plan["assumptions"]),
                    "budget": budget_snapshot(self.auditor),
                    "audit_plan": plan,
                    "audit_messages": messages + [message("assistant", action)],
                }
            return {
                "audit_plan": plan,
                "fixtures": merge_fixtures(state["fixtures"], plan["fixtures"]),
                "pending": None,
                "budget": budget_snapshot(self.auditor),
                "audit_messages": messages + [message("assistant", action)],
                "trace": self.event(
                    state,
                    "audit",
                    plan_digest=digest_json(plan),
                    candidate_visible=False,
                    reference_visible=False,
                ),
            }
        except (ValueError, GraphModelError) as error:
            code = error.code if isinstance(error, GraphModelError) else str(error)
            if state["audit_attempts"] < 3 and (
                isinstance(error, ValueError) or getattr(error, "retryable", False)
            ):
                return {
                    "pending": None,
                    "budget": budget_snapshot(self.auditor),
                    "audit_messages": messages
                    + [
                        message("assistant", action or {"error": code}),
                        message("user", plan_validation_feedback(action, code, self.task)),
                    ],
                    "trace": self.event(state, "invalid_audit", error=code),
                }
            return self.failure(state, "audit", error)

    def assess_audit(self, state):
        """Re-execute all versions' retained inputs before any edit is permitted."""
        try:
            fixtures = state["fixtures"]
            challenge_key = digest_json(
                [
                    state["candidate"]["sql"],
                    state["plan"]["reference"],
                    state["audit_plan"]["reference"],
                ]
            )
            challenged = state["challenged_pairs"]
            challenge = None
            if challenge_key not in challenged:
                challenge = find_distinguishing_fixture(
                    state["candidate"]["sql"],
                    state["plan"],
                    state["audit_plan"],
                    max_examples=self.max_challenge_examples,
                )
                if challenge is not None:
                    fixtures = merge_fixtures(fixtures, [challenge])
                challenged = challenged + [challenge_key]
            evidence = inspect_dispute(
                self.task,
                self.source,
                state["candidate"]["sql"],
                state["plan"],
                state["audit_plan"],
                fixtures,
                confirmed_examples=self.confirmed_examples,
            )
            return {
                "fixtures": fixtures,
                "challenged_pairs": challenged,
                "audit_evidence": evidence,
                "attribution": {},
                "attribution_attempts": 0,
                "tool_error": None,
                "trace": self.event(
                    state,
                    "assess_audit",
                    evidence=evidence,
                    generated_counterexample=challenge is not None,
                ),
            }
        except ValueError as error:
            return self.failure(state, "assess_audit", error)

    def prepare_attribute(self, state):
        return {"pending": "attribute", "attribution_attempts": state["attribution_attempts"] + 1}

    async def attribute(self, state):
        context = {
            "task": self.task.public_input(),
            "candidate_sql": state["candidate"]["sql"],
            "reference_program": state["plan"]["reference"],
            "independent_program": state["audit_plan"]["reference"],
            "evidence": state["audit_evidence"],
            "previous_tool_error": state.get("tool_error"),
        }
        messages = [message("system", ATTRIBUTION_PROMPT), message("user", context)]
        try:
            action = parse_review_action(await self.reviewer.generate(deepcopy(messages)))
            if action["action"] == "clarify":
                return {
                    "pending": None,
                    "status": "needs_clarification",
                    "reason": action["reason"],
                    "budget": budget_snapshot(self.reviewer),
                    "trace": self.event(state, "attribute", action=action),
                }
            decision = authorize_decision(action, self.task, state["audit_evidence"])
            attribution = {
                **decision,
                "action": action,
                "evidence_digest": digest_json(state["audit_evidence"]),
            }
            update = {
                "pending": None,
                "budget": budget_snapshot(self.reviewer),
                "attribution": attribution,
                "tool_error": None,
                "trace": self.event(state, "attribute", decision=attribution),
            }
            if decision["target"] == "ambiguous":
                update.update(status="needs_clarification", reason=decision["reason"])
            elif decision["target"] == "candidate" and state["attempts"] >= self.max_attempts:
                update.update(status="unresolved", reason="REPAIR_BUDGET_EXHAUSTED")
            elif (
                decision["target"] == "reference"
                and state["reference_revisions"] >= self.max_reference_revisions
            ):
                update.update(status="needs_clarification", reason="REFERENCE_REVISION_LIMIT")
            return update
        except (ValueError, GraphModelError) as error:
            code = error.code if isinstance(error, GraphModelError) else str(error)
            if state["attribution_attempts"] < 2 and (
                isinstance(error, ValueError) or getattr(error, "retryable", False)
            ):
                return {
                    "pending": None,
                    "budget": budget_snapshot(self.reviewer),
                    "tool_error": {"code": code},
                    "trace": self.event(state, "invalid_attribution", error=code),
                }
            return self.failure(state, "attribute", error)

    def checked_authorization(self, state, target: str) -> dict:
        """Tie an edit to current executions, not a stale model verdict."""
        attribution = state["attribution"]
        if attribution.get("target") != target:
            raise ValueError("ATTRIBUTION_REQUIRED")
        evidence = inspect_dispute(
            self.task,
            self.source,
            state["candidate"]["sql"],
            state["plan"],
            state["audit_plan"],
            state["fixtures"],
            confirmed_examples=self.confirmed_examples,
        )
        if digest_json(evidence) != attribution.get("evidence_digest"):
            raise ValueError("ATTRIBUTION_EVIDENCE_CHANGED")
        decision = authorize_decision(attribution["action"], self.task, evidence)
        if decision["target"] != target:
            raise ValueError("ATTRIBUTION_NOT_SUPPORTED")
        return decision

    def replace_reference(self, state):
        try:
            decision = self.checked_authorization(state, "reference")
        except ValueError as error:
            return self.failure(state, "replace_reference", error)
        versions = deepcopy(state["plan_versions"])
        versions[-1]["status"] = "superseded"
        versions.append(
            {
                "version": len(versions) + 1,
                "plan": deepcopy(state["audit_plan"]),
                "status": "active",
                "reason": decision["reason"],
                "evidence_digest": state["attribution"]["evidence_digest"],
                "origin": "blind_independent_recalculation",
            }
        )
        return {
            "plan": deepcopy(state["audit_plan"]),
            "plan_versions": versions,
            "reference_revisions": state["reference_revisions"] + 1,
            "trace": self.event(
                state,
                "replace_reference",
                version=len(versions),
                retained_fixture_count=len(state["fixtures"]),
                candidate_unchanged=True,
            ),
        }
