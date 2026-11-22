"""Remove repeated schema metadata from prompt copies, never from saved evidence.

Every discovery message already carries task.schema/types/keys once. Generic
fixture envelopes repeat those fields for local validation and replay; models
need the row tables, not the same schema dozens of times. All observations,
check IDs, results and failures remain present. Legacy fixtures are unchanged.
"""

from __future__ import annotations

from copy import deepcopy

_ENVELOPE_FIELDS = {"schema", "column_types", "tables", "primary_keys", "foreign_keys"}


def model_context(value):
    """Return a recursively copied prompt view without mutating durable records."""
    if isinstance(value, dict):
        if set(value) == _ENVELOPE_FIELDS and isinstance(value["tables"], dict):
            return model_context(value["tables"])
        return {key: model_context(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [model_context(item) for item in value]
    return deepcopy(value)
