"""Preflight cost reservation for the bounded Beijing Qwen probe experiment.

No environment lookup or network I/O occurs here.  Reported costs use the
published, uncached list rates, not the user's cloud bill or free-quota balance.
Every HTTP attempt keeps its conservative reservation, including failures.

The inherited client supplies only character count to reserve(), so input is
reserved at four tokens per Unicode character plus 4,096 framing tokens. This
is deliberately conservative for the bounded text-JSON protocol, not a general
tokenizer theorem or a guarantee of the provider's billing implementation.
Calls must be serialized: record_tokens has no attempt identifier, so this
budget refuses a second in-flight reservation rather than misattribute usage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlsplit

from .qwen_config import QwenClientError, QwenConfig, RequestBudget

PRICE_SOURCE = "https://help.aliyun.com/zh/model-studio/qwen3-8-flash"
PRICE_CHECKED_DATE = "2026-10-29"
PRICE_MODEL = "qwen3.8-flash"
_NANO_PER_CNY = 1_000_000_000
_INPUT_NANO_PER_TOKEN = 800  # CNY 0.8 / 1,000,000 tokens
_OUTPUT_NANO_PER_TOKEN = 2_700  # CNY 2.7 / 1,000,000 tokens
_AUTHORIZED_MAX_CNY = Decimal("5")
_FRAMING_TOKEN_RESERVATION = 4_096


def _cny(nano: int) -> float:
    return float(Decimal(nano) / _NANO_PER_CNY)


def validate_pricing(config: QwenConfig) -> dict[str, Any]:
    """Refuse a model/route to which the recorded Beijing prices do not apply.

    The normal QwenConfig performs endpoint and subscription-key validation.
    Revalidate it here as well; never return its key or full endpoint URL.
    """
    if not isinstance(config, QwenConfig):
        raise QwenClientError("PRICE_VALIDATION_REQUIRES_QWEN_CONFIG")
    QwenConfig(config.api_key, config.base_url, config.model)
    if config.model != PRICE_MODEL:
        raise QwenClientError("PRICE_MODEL_NOT_APPROVED")
    host = urlsplit(config.base_url).hostname or ""
    workspace_host = re.fullmatch(
        r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cn-beijing\.maas\.aliyuncs\.com",
        host,
        flags=re.ASCII,
    )
    if host != "dashscope.aliyuncs.com" and workspace_host is None:
        raise QwenClientError("PRICE_REGION_NOT_APPROVED")
    return {
        "model": PRICE_MODEL,
        "region": "cn-beijing",
        "currency": "CNY",
        "input_cny_per_million_tokens": 0.8,
        "output_cny_per_million_tokens": 2.7,
        "discounts_and_free_quota_included": False,
        "price_source": PRICE_SOURCE,
        "price_checked_date": PRICE_CHECKED_DATE,
    }


@dataclass
class MonetaryRequestBudget(RequestBudget):
    """RequestBudget-compatible, serialized, never-refunded cost ledger.

    Use validate_pricing(config) before running a paid client and configure the
    client's max_tokens <= max_output_tokens. The budget cannot inspect payload
    options through the inherited reserve(prompt_chars) interface.
    """

    limit: int = 80
    max_input_chars: int = 1_000_000
    max_reported_tokens: int = 250_000
    money_limit_cny: float = 5.0
    max_output_tokens: int = 900
    _money_limit_nano: int = field(default=0, init=False, repr=False)
    _output_cap: int = field(default=0, init=False, repr=False)
    _reserved_nano: int = field(default=0, init=False, repr=False)
    _reported_nano: int = field(default=0, init=False, repr=False)
    _reported_prompt_tokens: int = field(default=0, init=False, repr=False)
    _reported_completion_tokens: int = field(default=0, init=False, repr=False)
    _complete_count: int = field(default=0, init=False, repr=False)
    _unknown_count: int = field(default=0, init=False, repr=False)
    _unmatched_usage_events: int = field(default=0, init=False, repr=False)
    _pending: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _ledger: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.money_limit_cny) not in (int, float, Decimal):
            raise ValueError("money_limit_cny must be a finite positive amount")
        try:
            money = Decimal(str(self.money_limit_cny))
        except InvalidOperation:
            raise ValueError("money_limit_cny must be a finite positive amount") from None
        if not money.is_finite() or not Decimal(0) < money <= _AUTHORIZED_MAX_CNY:
            raise ValueError("Money allowance must be positive and at most CNY 5")
        nano = money * _NANO_PER_CNY
        if nano != nano.to_integral_value() or nano < 1:
            raise ValueError("Money allowance requires integer nanoyuan precision")
        if type(self.max_output_tokens) is not int or not 1 <= self.max_output_tokens <= 900:
            raise ValueError("max_output_tokens must be between 1 and 900")
        self._money_limit_nano = int(nano)
        self._output_cap = self.max_output_tokens

    def _stop_locked(self, reason: str, *, unknown_pending: bool) -> None:
        safe_reason = (
            reason
            if type(reason) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", reason)
            else "STOP_REASON_REDACTED"
        )
        self.stopped_reason = self.stopped_reason or safe_reason
        if unknown_pending and self._pending is not None:
            self._pending["status"] = "usage_unknown"
            self._pending["stop_reason"] = safe_reason
            self._unknown_count += 1
            self._pending = None

    def reserve(self, prompt_chars: int) -> int:
        if type(prompt_chars) is not int or prompt_chars < 0:
            raise ValueError("prompt_chars must be a nonnegative integer")
        # Do not call super().reserve() while holding the non-reentrant lock.
        with self._lock:
            if self.stopped_reason:
                raise QwenClientError("LIVE_BUDGET_STOPPED_" + self.stopped_reason)
            if self._pending is not None:
                raise QwenClientError("LIVE_IN_FLIGHT_RESERVATION")
            if self.used >= self.limit:
                raise QwenClientError("LIVE_HTTP_REQUEST_LIMIT")
            if self.input_chars + prompt_chars > self.max_input_chars:
                raise QwenClientError("LIVE_INPUT_CHARACTER_LIMIT")
            if self.reported_tokens >= self.max_reported_tokens:
                raise QwenClientError("LIVE_REPORTED_TOKEN_LIMIT")
            input_upper = 4 * prompt_chars + _FRAMING_TOKEN_RESERVATION
            reserved = (
                input_upper * _INPUT_NANO_PER_TOKEN + self._output_cap * _OUTPUT_NANO_PER_TOKEN
            )
            if self._reserved_nano + reserved > self._money_limit_nano:
                self._stop_locked("MONEY_RESERVATION_LIMIT", unknown_pending=False)
                raise QwenClientError("LIVE_MONEY_RESERVATION_LIMIT")
            self.used += 1
            self.input_chars += prompt_chars
            self._reserved_nano += reserved
            self._pending = {
                "attempt": self.used,
                "prompt_chars": prompt_chars,
                "input_token_upper": input_upper,
                "output_token_upper": self._output_cap,
                "reserved_upper_cny": _cny(reserved),
                "status": "in_flight",
            }
            self._ledger.append(self._pending)
            return self.used

    def record_tokens(self, usage: dict) -> None:
        with self._lock:
            if self._pending is None:
                self._unmatched_usage_events += 1
                self._stop_locked("UNRESERVED_USAGE", unknown_pending=False)
                raise QwenClientError("LIVE_UNRESERVED_USAGE")
            raw = usage if isinstance(usage, dict) else {}
            prompt, completion, total = (
                raw.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            if any(type(value) is not int or value < 0 for value in (prompt, completion, total)):
                self._stop_locked("USAGE_UNAVAILABLE", unknown_pending=True)
                return
            if total != prompt + completion:
                self._stop_locked("USAGE_INCONSISTENT", unknown_pending=True)
                return
            pending = self._pending
            # Known usage remains in the estimate even when a provider exceeds
            # the reservation; the experiment must then stop, not hide it.
            nano = prompt * _INPUT_NANO_PER_TOKEN + completion * _OUTPUT_NANO_PER_TOKEN
            self.reported_tokens += total
            self._reported_prompt_tokens += prompt
            self._reported_completion_tokens += completion
            self._reported_nano += nano
            self._complete_count += 1
            pending.update(
                status="usage_complete",
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
                estimated_reported_cny=_cny(nano),
            )
            self._pending = None
            if prompt > pending["input_token_upper"] or completion > pending["output_token_upper"]:
                pending["status"] = "usage_exceeded_reservation"
                self._stop_locked("USAGE_EXCEEDED_RESERVATION", unknown_pending=False)
            if self._reported_nano > self._money_limit_nano:
                self._stop_locked("REPORTED_MONEY_LIMIT_EXCEEDED", unknown_pending=False)

    def stop(self, reason: str) -> None:
        with self._lock:
            self._stop_locked(reason, unknown_pending=True)

    def snapshot(self) -> dict[str, Any]:
        # Do not call super().snapshot() while holding the inherited lock.
        with self._lock:
            complete = (
                self._unknown_count == 0
                and self._unmatched_usage_events == 0
                and self._pending is None
                and self._complete_count == self.used
            )
            return {
                "http_attempts": self.used,
                "http_attempt_limit": self.limit,
                "input_chars": self.input_chars,
                "input_char_limit": self.max_input_chars,
                "reported_total_tokens": self.reported_tokens,
                "reported_token_stop_threshold": self.max_reported_tokens,
                "stopped_reason": self.stopped_reason,
                "money_limit_cny": _cny(self._money_limit_nano),
                "reserved_upper_cny": _cny(self._reserved_nano),
                "estimated_reported_cny": _cny(self._reported_nano),
                "reported_cost_complete": complete,
                "complete_count": self._complete_count,
                "unknown_count": self._unknown_count,
                "in_flight_count": int(self._pending is not None),
                "unmatched_usage_events": self._unmatched_usage_events,
                "reported_prompt_tokens": self._reported_prompt_tokens,
                "reported_completion_tokens": self._reported_completion_tokens,
                "max_output_tokens": self._output_cap,
                "input_token_reservation_rule": "4 * prompt_chars + 4096 framing tokens",
                "reservation_ledger": [dict(entry) for entry in self._ledger],
                "pricing": {
                    "model": PRICE_MODEL,
                    "region": "cn-beijing",
                    "currency": "CNY",
                    "input_cny_per_million_tokens": 0.8,
                    "output_cny_per_million_tokens": 2.7,
                    "price_source": PRICE_SOURCE,
                    "price_checked_date": PRICE_CHECKED_DATE,
                    "discounts_and_free_quota_included": False,
                },
                "limitation": "Conservative preflight reservation and list-rate usage estimate; not a verified cloud bill or a provider-side spending cap.",
            }
