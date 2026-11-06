"""Explicit recovery from a fully accounted truncated discovery response.

No refund, allowance reset, or recovery from unknown transport outcomes. The
original journal/report remain unchanged and the continuation keeps lineage.
"""

from copy import deepcopy


def reviewed_truncation(snapshot: dict, report: dict) -> dict:
    if snapshot != report.get("budget_end"):
        raise ValueError("TRUNCATION_REPORT_JOURNAL_MISMATCH")
    if (
        snapshot.get("stopped_reason") != "PROVIDER_ERROR"
        or snapshot.get("reported_cost_complete") is not True
        or any(
            snapshot.get(key) != 0
            for key in ("unknown_count", "in_flight_count", "unmatched_usage_events")
        )
    ):
        raise ValueError("ONLY_ACCOUNTED_TRUNCATION_RECOVERABLE")
    start = report.get("budget_start", {})
    begin, finish = start.get("http_attempts"), snapshot.get("http_attempts")
    if type(begin) is not int or type(finish) is not int or not 0 <= begin < finish:
        raise ValueError("TRUNCATION_ATTEMPT_RANGE")
    ledger = snapshot.get("reservation_ledger", [])
    events = report.get("http_events", [])
    if (
        ledger[:begin] != start.get("reservation_ledger")
        or len(ledger) != finish
        or [event.get("attempt") for event in events] != list(range(begin + 1, finish + 1))
    ):
        raise ValueError("TRUNCATION_HISTORY_MISMATCH")
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    for index, event in enumerate(events):
        usage = event.get("usage", {})
        if (
            event.get("http_status") != 200
            or event.get("usage_complete") is not True
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in fields)
            or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
            or any(usage[key] != ledger[event["attempt"] - 1].get(key) for key in fields)
        ):
            raise ValueError("TRUNCATION_USAGE_MISMATCH")
        if index == len(events) - 1:
            if (
                event.get("status") != "failed"
                or event.get("error_code") != "GRAPH_QWEN_RESPONSE_TRUNCATED"
                or usage["completion_tokens"] != snapshot["max_output_tokens"]
            ):
                raise ValueError("LAST_EVENT_NOT_ACCOUNTED_TRUNCATION")
        elif event.get("status") != "completed":
            raise ValueError("OTHER_PROVIDER_FAILURE")
    for field, total in zip(
        fields, ("reported_prompt_tokens", "reported_completion_tokens", "reported_total_tokens")
    ):
        if snapshot[total] - start[total] != sum(e["usage"][field] for e in events):
            raise ValueError("TRUNCATION_TOTAL_MISMATCH")
    result = deepcopy(snapshot)
    result["stopped_reason"] = None
    return result
