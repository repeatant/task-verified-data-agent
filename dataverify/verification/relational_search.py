"""Bounded witness search from typed observed rows, preserving declared keys.

This is a disagreement finder, not a source of business truth. The search only
claims the supplied SQL schema constraints; unknown business invariants remain
unknown. Gold SQL and benchmark labels are never available here.
"""

from __future__ import annotations

import time

from .relational_data import RelationalData


def find_relational_fixture(sql, references, source, *, max_examples=40):
    from hypothesis import HealthCheck, Phase, find, settings
    from hypothesis import strategies as st
    from hypothesis.errors import NoSuchExample, Unsatisfiable

    from .reference_audit import _candidate_result, _reference_result, _same

    if not isinstance(source, RelationalData):
        raise ValueError("RELATIONAL_SEARCH_SOURCE")
    # Subsets and value substitutions retain PKs/FKs; data is bounded before
    # crossing into Hypothesis. Preserve only source-supported cell values.
    protected = {table: set(source.primary_keys.get(table, ())) for table in source.schema}
    for fk in source.foreign_keys:
        protected[fk["table"]].update(fk["columns"])
        protected[fk["references"]].update(fk["reference_columns"])

    @st.composite
    def legal_data(draw):
        tables = {}
        for table, columns in source.schema.items():
            existing = list(source.tables[table])[:10]
            rows = []
            for original in existing:
                if not draw(st.booleans()):
                    continue
                row = list(original)
                for index, column in enumerate(columns):
                    if column not in protected[table] and draw(st.booleans()):
                        values = list(dict.fromkeys(r[index] for r in existing))
                        row[index] = draw(st.sampled_from(values))
                rows.append(row)
            tables[table] = rows
        # Removing a parent removes dangling children, including chained FKs.
        for _ in range(sum(map(len, tables.values())) + 1):
            changed = False
            for fk in source.foreign_keys:
                parent_cols = source.schema[fk["references"]]
                child_cols = source.schema[fk["table"]]
                target = {
                    tuple(r[parent_cols.index(c)] for c in fk["reference_columns"])
                    for r in tables[fk["references"]]
                }
                retained = []
                for row in tables[fk["table"]]:
                    key = tuple(row[child_cols.index(c)] for c in fk["columns"])
                    if None in key or key in target:
                        retained.append(row)
                changed |= len(retained) != len(tables[fk["table"]])
                tables[fk["table"]] = retained
            if not changed:
                break
        return RelationalData(
            source.schema, tables, source.column_types, source.primary_keys, source.foreign_keys
        )

    started, count, witness = time.monotonic(), 0, None

    class SearchLimit(Exception):
        pass

    def differs(data):
        nonlocal count, witness
        if count >= max_examples * 12 or time.monotonic() - started > 3:
            raise SearchLimit
        count += 1
        results = [_candidate_result(sql, data)] + [
            _reference_result(ref, data) for ref in references
        ]
        valid = [r for r in results if r["status"] == "ok"]
        disagreement = bool(valid) and (
            len(valid) != len(results) or any(not _same(valid[0], item) for item in valid[1:])
        )
        if disagreement:
            witness = data
        return disagreement

    try:
        witness = find(
            legal_data(),
            differs,
            settings=settings(
                max_examples=max_examples,
                database=None,
                derandomize=True,
                deadline=None,
                phases=(Phase.generate, Phase.shrink),
                suppress_health_check=(HealthCheck.too_slow,),
            ),
        )
    except (NoSuchExample, Unsatisfiable, SearchLimit):
        pass
    return (
        None
        if witness is None
        else {"name": "searched_relational_disagreement", "data": witness.to_dict()}
    )
