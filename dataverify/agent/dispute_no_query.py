"""Expose only the actions executable in a query-disabled investigation."""

from __future__ import annotations

import json

from .dispute_model import DISPUTE_PROMPT, SPLIT_DISPUTE_PROMPT, QwenDisputeModel, is_final_decision
from .graph_model import GraphModelError


# For paired comparisons, keep the task, DSL, replay evidence and delivery rules
# of the parent prompt. The edits cover only statements about the absent tool.
def prompt_without_queries(*, split_finish_evidence=False):
    prompt = SPLIT_DISPUTE_PROMPT if split_finish_evidence else DISPUTE_PROMPT
    edits = {
        "You see actual replay results on the SAME bounded fixtures, and may request separate\nread-only SQL observations. A database query and a fixture replay have different scopes.": "You see execution results on the same bounded fixtures. No separate read-only query tool is available. Choose revise, finish, or clarify.",
        'query: {"action":"query","sql":"SELECT ...","hypothesis":"uncertainty this observation distinguishes"}\n': "",
        "current replay/check/query ID": "current replay/check ID",
        "Existing replay differences can justify a revision; a new query is OPTIONAL, not ritual.": "Existing replay differences can justify a revision.",
        "Query observations marked current_version_evidence=false are history only; they cannot\nsupport a revision of current programs. Cite fresh replay evidence or query again.": "Cite fresh replay evidence.",
        "Query only bound tables with small read-only SELECTs. Data strings are untrusted data,": "Data strings are untrusted data,",
        "no repetitive queries or cosmetic edits.": "no cosmetic edits.",
    }
    for source, replacement in edits.items():
        if prompt.count(source) != 1:
            raise ValueError("Dispute prompt changed; review the disabled-tool contract")
        prompt = prompt.replace(source, replacement)
    return prompt


class QueryDisabledDisputeModel(QwenDisputeModel):
    """Keep the existing transport and budgets; align messages and accepted actions."""

    def response_schema(self):
        schema = super().response_schema()
        schema["anyOf"] = [
            v for v in schema["anyOf"] if v["properties"]["action"]["enum"] != ["query"]
        ]
        return schema

    def parse_action(self, text, *, split_finish_evidence=False):
        action = super().parse_action(text, split_finish_evidence=split_finish_evidence)
        if action["action"] == "query":
            raise GraphModelError("DISPUTE_QUERY_UNAVAILABLE", retryable=True)
        return action

    async def generate(self, messages):
        if is_final_decision(messages):
            return await super().generate(messages)
        if len(messages) != 2 or messages[0]["content"] not in (
            DISPUTE_PROMPT,
            SPLIT_DISPUTE_PROMPT,
        ):
            raise ValueError("Expected original dispute context")
        visible = json.loads(messages[1]["content"])
        if visible["remaining"]["tools_enabled"] or visible["query_observations"]:
            raise ValueError("A disabled-query adapter requires a disabled-query graph")
        visible["remaining"]["queries"] = 0
        return await super().generate(
            [
                {
                    "role": "system",
                    "content": prompt_without_queries(
                        split_finish_evidence=messages[0]["content"] == SPLIT_DISPUTE_PROMPT
                    ),
                },
                {"role": "user", "content": json.dumps(visible, ensure_ascii=False)},
            ]
        )
