"""Versioned three-program replay, without promoting agreement to truth.

All three programs run on exactly the same bounded datasets. Original-database
SQL observations may support an investigation, but never enter this comparison
as if a reference program had computed the full database. The trusted host owns
the session and registers tool results; model output proposes a local revision.
"""

from __future__ import annotations

from copy import deepcopy

from dataverify.runtime.reporting import digest_json
from dataverify.verification.discovery_runtime import (
    DiscoveryTask,
    fixture_data,
    validate_model_fixtures,
    validate_plan,
    validate_sql,
)
from dataverify.verification.probe_runtime import TABLE_SCHEMA, ProbeData
from dataverify.verification.reference_audit import (
    inspect_dispute,
    merge_fixtures,
    validate_confirmed_examples,
)
from dataverify.verification.reference_expression import parse_expression
from dataverify.verification.relational_data import RelationalData

_ROLES = ("candidate", "reference", "auditor")
_CHECK_ROLE = {"candidate": "candidate", "reference": "reference", "auditor": "independent"}


def _execution_agreement(evidence: dict) -> bool:
    """Keep the existing graph's replay criterion without an agent dependency."""
    summary = evidence["summary"]
    return bool(
        summary["all_executed"]
        and all(
            summary[key]
            for key in (
                "candidate_matches_reference",
                "candidate_matches_independent",
                "reference_matches_independent",
                "candidate_matches_confirmed",
                "reference_matches_confirmed",
                "independent_matches_confirmed",
            )
        )
    )


def _program(value, role: str) -> str:
    maximum = 12_000 if role == "candidate" else 4_000
    if type(value) is not str or not 1 <= len(value) <= maximum:
        raise ValueError("DISPUTE_PROGRAM_LENGTH")
    return value


def _initial_plan(plan: dict, task: DiscoveryTask) -> dict:
    """Validate the envelope without denying recovery of a broken initial DSL."""
    if not isinstance(plan, dict) or "reference" not in plan:
        raise ValueError("DISPUTE_PLAN_FIELDS")
    original = _program(plan["reference"], "reference")
    # validate_plan also executes its expression. A harmless expression isolates
    # host/schema/fixture validation from the initial program's execution error.
    validated = validate_plan({**deepcopy(plan), "reference": "[[0]]"}, task)
    validated["reference"] = original
    return validated


class DisputeSession:
    """Append-only programs, retained inputs and version-bound observations.

    Accepting a revision means it is executable and respects supplied confirmed
    examples. It does not assert the model's rationale is valid or the program
    is semantically correct. No program is automatically copied from a majority.
    """

    def __init__(
        self,
        task: DiscoveryTask,
        source_fixture: ProbeData | RelationalData,
        candidate_sql: str,
        reference_plan: dict,
        auditor_plan: dict,
        retained_fixtures=(),
        confirmed_examples=(),
    ):
        if not isinstance(task, DiscoveryTask):
            raise TypeError("DISPUTE_TASK_REQUIRED")
        if not isinstance(source_fixture, (ProbeData, RelationalData)):
            raise TypeError("DISPUTE_BOUNDED_SOURCE_REQUIRED")
        self._task = deepcopy(task)
        options = (task.schema, task.column_types, task.primary_keys, task.foreign_keys)
        if task.schema is None:
            if not isinstance(source_fixture, ProbeData):
                raise ValueError("DISPUTE_TASK_SCHEMA_MISMATCH")
            options = ()
        self._source = fixture_data(deepcopy(source_fixture.to_dict()), *options)
        self._source_id = digest_json(self._source.to_dict())
        self._programs = {"candidate": _program(candidate_sql, "candidate")}
        self._plans = {
            "reference": _initial_plan(reference_plan, self._task),
            "auditor": _initial_plan(auditor_plan, self._task),
        }
        self._programs.update({role: plan["reference"] for role, plan in self._plans.items()})
        self._fixtures = []
        self._validate_and_append_fixtures(list(retained_fixtures))
        for plan in self._plans.values():
            self._validate_and_append_fixtures(plan["fixtures"])
        self._confirmed = validate_confirmed_examples(list(confirmed_examples))
        # Confirmed inputs come from the trusted caller, but still must obey the
        # same task schema. A different-schema confirmed answer is not authority.
        for example in self._confirmed:
            self._validate_fixtures([{"name": example["name"], "data": example["data"]}])
        self._versions = dict.fromkeys(_ROLES, 1)
        self._history = {
            role: [{"version": 1, "program": code, "origin": "initial"}]
            for role, code in self._programs.items()
        }
        self._attempts = []
        self._observations = {}
        self._replay_count = 0
        self._external_source_id = None

    def _validate_fixtures(self, fixtures: list[dict]) -> list[dict]:
        # The existing validator accepts 1-3 per model action; constructor data
        # and accumulated history can contain more, under merge_fixtures' limit.
        validated = []
        for fixture in fixtures:
            validated.extend(validate_model_fixtures([deepcopy(fixture)], self._task, self._source))
        return validated

    def _validate_and_append_fixtures(self, fixtures: list[dict]) -> None:
        self._fixtures = merge_fixtures(self._fixtures, self._validate_fixtures(fixtures))

    def append_fixtures(self, fixtures: list[dict]) -> dict:
        """Keep old inputs and append legal new inputs; expectations are not gold."""
        if not isinstance(fixtures, list):
            raise ValueError("DISPUTE_FIXTURES_REQUIRED")
        self._validate_and_append_fixtures(fixtures)
        return self.replay()

    def _binding(self) -> str:
        return digest_json(
            {
                "task": self._task.public_input(),
                "source": self._source_id,
                "versions": self._versions,
                "programs": self._programs,
                "plans": self._plans,
                "fixtures": self._fixtures,
                "confirmed": self._confirmed,
            }
        )

    def snapshot(self) -> dict:
        """Return a detached state snapshot; callers cannot rewrite the session."""
        return deepcopy(
            {
                "task_id": self._task.task_id,
                "source_id": self._source_id,
                "scope": "same_bounded_fixture",
                "programs": self._programs,
                "plans": self._plans,
                "versions": self._versions,
                "history": self._history,
                "fixtures": self._fixtures,
                "attempts": self._attempts,
                "version_binding": self._binding(),
                "semantic_status": "not_proven",
            }
        )

    def _inspect(self, programs=None, plans=None) -> dict:
        programs = programs or self._programs
        plans = plans or self._plans
        return inspect_dispute(
            self._task,
            self._source,
            programs["candidate"],
            plans["reference"],
            plans["auditor"],
            self._fixtures,
            self._confirmed,
        )

    def replay(self) -> dict:
        """Execute all current programs on identical retained/confirmed data."""
        evidence = self._inspect()
        self._replay_count += 1
        binding = self._binding()
        observation_id = f"replay_{self._replay_count:04d}_{binding[:12]}"
        evidence.update(
            id=observation_id,
            version_binding=binding,
            versions=deepcopy(self._versions),
            source_id=self._source_id,
            scope="same_bounded_fixture",
            agreement=_execution_agreement(evidence),
            semantic_status="not_proven",
        )
        self._observations[observation_id] = {"binding": binding, "kind": "replay"}
        for check in evidence["checks"]:
            check["id"] = observation_id + ":" + check["check_id"]
            check["version_binding"] = binding
            self._observations[check["id"]] = {"binding": binding, "kind": "replay_check"}
        return deepcopy(evidence)

    def record_external_observation(self, observation: dict) -> str:
        """Register a SQLTools result supplied by the host, never model JSON.

        The digest checks integrity of the record, not tool-origin authenticity.
        The graph is the trust boundary and must call this method only with the
        direct return of its bound executor. Original-DB facts do not influence
        replay agreement or become expected answers for a synthetic fixture.
        """
        if not isinstance(observation, dict):
            raise ValueError("DISPUTE_TOOL_OBSERVATION")
        record = deepcopy(observation)
        observation_id = record.get("id")
        if type(observation_id) is not str or not 1 <= len(observation_id) <= 200:
            raise ValueError("DISPUTE_TOOL_OBSERVATION_ID")
        if observation_id in self._observations:
            raise ValueError("DISPUTE_OBSERVATION_ID_REUSED")
        if record.get("scope") not in {"original_database", "provided_fixture"}:
            raise ValueError("DISPUTE_TOOL_SCOPE")
        if record.get("status") not in {"ok", "truncated", "rejected", "error", "resource_limit"}:
            raise ValueError("DISPUTE_TOOL_STATUS")
        if type(record.get("source_id")) is not str or len(record["source_id"]) != 64:
            raise ValueError("DISPUTE_TOOL_SOURCE")
        actual_digest = record.pop("observation_digest", None)
        record.pop("elapsed_ms", None)
        if actual_digest != digest_json(record):
            raise ValueError("DISPUTE_TOOL_OBSERVATION_CHANGED")
        if self._external_source_id not in (None, record["source_id"]):
            raise ValueError("DISPUTE_QUERY_SOURCE_CHANGED")
        self._external_source_id = record["source_id"]
        self._observations[observation_id] = {
            "binding": self._binding(),
            "kind": "external_query",
            "observation": deepcopy(observation),
        }
        return observation_id

    def _revision_evidence(self, evidence_ids, requirement_quote) -> None:
        if (
            not isinstance(evidence_ids, (list, tuple))
            or not 1 <= len(evidence_ids) <= 25
            or any(type(value) is not str for value in evidence_ids)
            or len(set(evidence_ids)) != len(evidence_ids)
        ):
            raise ValueError("DISPUTE_EVIDENCE_IDS")
        for observation_id in evidence_ids:
            evidence = self._observations.get(observation_id)
            if evidence is None:
                raise ValueError("DISPUTE_UNKNOWN_EVIDENCE")
            if evidence["binding"] != self._binding():
                raise ValueError("DISPUTE_STALE_EVIDENCE")
        if (
            type(requirement_quote) is not str
            or not requirement_quote.strip()
            or len(requirement_quote) > 6_000
            or not any(
                requirement_quote in text for text in (self._task.question, self._task.definitions)
            )
        ):
            raise ValueError("DISPUTE_REQUIREMENT_QUOTE")

    def revise(
        self, target: str, program: str, reason: str, evidence_ids, requirement_quote: str
    ) -> dict:
        """Attempt one local revision, preserving both accepted and failed history."""
        attempt = {
            "target": target if isinstance(target, str) else None,
            "program": program[:12_001] if isinstance(program, str) else None,
            "reason": reason[:1_001] if isinstance(reason, str) else None,
            "evidence_ids": list(evidence_ids) if isinstance(evidence_ids, (list, tuple)) else [],
            "requirement_quote": requirement_quote[:6_001]
            if isinstance(requirement_quote, str)
            else None,
            "base_versions": deepcopy(self._versions),
            "base_binding": self._binding(),
        }
        try:
            if target not in _ROLES:
                raise ValueError("DISPUTE_TARGET")
            _program(program, target)
            if type(reason) is not str or not 1 <= len(reason) <= 1_000:
                raise ValueError("DISPUTE_REASON")
            self._revision_evidence(evidence_ids, requirement_quote)
            if program == self._programs[target]:
                raise ValueError("DISPUTE_NO_PROGRAM_CHANGE")
            proposed_programs, proposed_plans = deepcopy(self._programs), deepcopy(self._plans)
            proposed_programs[target] = program
            if target == "candidate":
                validate_sql(
                    program,
                    self._source.schema
                    if isinstance(self._source, RelationalData)
                    else TABLE_SCHEMA,
                )
            else:
                parse_expression(program)
                proposed_plans[target] = validate_plan(
                    {**proposed_plans[target], "reference": program}, self._task
                )
            trial = self._inspect(proposed_programs, proposed_plans)
            attempt["trial"] = deepcopy(trial)
            check_role = _CHECK_ROLE[target]
            if any(check[check_role]["status"] != "ok" for check in trial["checks"]):
                raise ValueError("DISPUTE_REVISION_EXECUTION_ERROR")
            if not trial["summary"][f"{check_role}_matches_confirmed"]:
                raise ValueError("DISPUTE_CONFIRMED_EXAMPLE_REGRESSION")
        except (ValueError, TypeError) as error:
            attempt.update(accepted=False, error={"code": str(error), "message": str(error)})
            self._attempts.append(deepcopy(attempt))
            return {
                "accepted": False,
                "error": attempt["error"],
                "observation": self.replay(),
                "versions": deepcopy(self._versions),
            }

        self._programs, self._plans = proposed_programs, proposed_plans
        self._versions[target] += 1
        record = {
            "version": self._versions[target],
            "program": program,
            "origin": "explicit_model_proposal",
            "reason": reason,
            "evidence_ids": list(evidence_ids),
            "requirement_quote": requirement_quote,
            "base_binding": attempt["base_binding"],
            "semantic_status": "not_proven",
        }
        self._history[target].append(record)
        attempt.update(accepted=True, error=None, versions=deepcopy(self._versions))
        self._attempts.append(deepcopy(attempt))
        return {
            "accepted": True,
            "error": None,
            "observation": self.replay(),
            "versions": deepcopy(self._versions),
            "semantic_status": "not_proven",
        }
