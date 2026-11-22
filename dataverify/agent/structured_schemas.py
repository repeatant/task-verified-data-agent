"""Optional provider-side JSON shape contracts; never semantic truth or SQL safety.

Qwen documents anyOf, const, optional properties and dynamic additionalProperties
for strict structured output. Local action, stage, DSL and fixture validation
remain mandatory. Reference: https://help.aliyun.com/zh/model-studio/qwen-structured-output
"""

from __future__ import annotations


def _object(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def _text(maximum, minimum=1):
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


def writer_schema() -> dict:
    """Only writer action shape; SQL validity is checked after generation."""
    return _object(
        {"sql": _text(20_000), "task_id": _text(80), "thought": _text(2000, 0)},
        required=["sql", "task_id"],
    )


def reviewer_schema() -> dict:
    """One disjoint action union, retaining separate local stage authorization."""
    scalar = {
        "type": ["string", "number", "null"],
        "maxLength": 4000,
        "minimum": -(10**15),
        "maximum": 10**15,
    }
    row = {
        "anyOf": [
            {"type": "object", "additionalProperties": scalar, "maxProperties": 128},
            {"type": "array", "items": scalar, "maxItems": 128},
        ]
    }
    tables = {
        "type": "object",
        "minProperties": 1,
        "maxProperties": 64,
        "additionalProperties": {"type": "array", "items": row, "maxItems": 50},
    }
    patch = _object(
        {
            "base": {"const": "bound_source"},
            "changes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": _object(
                    {
                        "table": _text(200),
                        "row": {"type": "integer", "minimum": 0, "maximum": 49},
                        "values": {
                            "type": "object",
                            "minProperties": 1,
                            "maxProperties": 128,
                            "additionalProperties": scalar,
                        },
                    }
                ),
            },
        }
    )
    fixture = _object({"name": _text(120), "data": {"anyOf": [tables, patch]}})
    basis = {
        "type": "array",
        "maxItems": 8,
        "items": _object(
            {
                "source": {"enum": ["question", "definitions"]},
                "quote": _text(6000),
            }
        ),
    }
    reason = _text(1000)
    plan = _object(
        {
            "action": {"const": "plan"},
            "basis": {**basis, "minItems": 1},
            "assumptions": {"type": "array", "items": _text(500), "maxItems": 8},
            "reference": _text(4000),
            "fixtures": {"type": "array", "items": {"$ref": "#/$defs/fixture"}, "maxItems": 3},
        }
    )
    probe = _object(
        {
            "action": {"const": "probe"},
            "reason": reason,
            "fixtures": {
                "type": "array",
                "items": {"$ref": "#/$defs/fixture"},
                "minItems": 1,
                "maxItems": 3,
            },
        }
    )
    simple = [
        _object({"action": {"const": action}, "reason": reason})
        for action in ("finish", "repair", "clarify")
    ]
    attribute = _object(
        {
            "action": {"const": "attribute"},
            "target": {"enum": ["candidate", "reference", "ambiguous"]},
            "reason": reason,
            "basis": basis,
            "checks": {"type": "array", "items": _text(200), "maxItems": 25},
        }
    )
    return {
        "type": "object",
        "$defs": {"fixture": fixture},
        "anyOf": [plan, probe, *simple, attribute],
    }
