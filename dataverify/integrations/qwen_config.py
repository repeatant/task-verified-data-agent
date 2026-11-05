"""Ordinary Qwen configuration and request budgets without a model runtime.

No credential lookup or network I/O occurs on import. Both native DB-GPT and
LangGraph adapters share these types; environment lookup remains explicit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from threading import Lock
from urllib.parse import urlsplit


class QwenClientError(RuntimeError):
    """Safe diagnostic text only: upstream includes this message in its logs."""


@dataclass(frozen=True)
class QwenConfig:
    api_key: str = field(repr=False)
    base_url: str
    model: str = "qwen3.8-flash"

    def __post_init__(self):
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise QwenClientError("QWEN_API_KEY_NOT_CONFIGURED")
        if self.api_key.startswith("sk-sp-"):
            raise QwenClientError("ORDINARY_API_REQUIRED_SUBSCRIPTION_KEY_REJECTED")
        try:
            parsed = urlsplit(self.base_url)
            port = parsed.port
        except (ValueError, TypeError, AttributeError):
            raise QwenClientError("ORDINARY_QWEN_HTTPS_ENDPOINT_REQUIRED") from None
        host = parsed.hostname or ""
        ordinary_host = host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"} or (
            host.endswith(".maas.aliyuncs.com") and "token-plan" not in host
        )
        if (
            parsed.scheme != "https"
            or not ordinary_host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or port not in (None, 443)
            or parsed.path.rstrip("/") != "/compatible-mode/v1"
        ):
            raise QwenClientError("ORDINARY_QWEN_HTTPS_ENDPOINT_REQUIRED")
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 200:
            raise QwenClientError("INVALID_QWEN_MODEL")

    @classmethod
    def from_local_environment(cls):
        """Read only three named variables; use Windows user scope if absent.

        GUI processes may predate setx. Do not print/copy a key into project files
        just to refresh that process environment. No .env file is auto-loaded.
        """
        values = {
            name: os.environ.get(name, "")
            for name in (
                "DASHSCOPE_API_KEY",
                "DASHSCOPE_BASE_URL",
                "DASHSCOPE_MODEL",
            )
        }
        if os.name == "nt":
            try:
                import winreg

                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
                    for name in values:
                        if not values[name]:
                            try:
                                values[name] = winreg.QueryValueEx(handle, name)[0]
                            except FileNotFoundError:
                                pass
            except OSError:
                pass
        return cls(
            values["DASHSCOPE_API_KEY"],
            values["DASHSCOPE_BASE_URL"] or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            values["DASHSCOPE_MODEL"] or "qwen3.8-flash",
        )


@dataclass
class RequestBudget:
    """Shared across conversations; reserve before every actual HTTP attempt."""

    limit: int = 6
    max_input_chars: int = 150_000
    max_reported_tokens: int = 30_000
    used: int = field(default=0, init=False)
    input_chars: int = field(default=0, init=False)
    reported_tokens: int = field(default=0, init=False)
    stopped_reason: str | None = field(default=None, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        for value in (self.limit, self.max_input_chars, self.max_reported_tokens):
            if type(value) is not int or value < 1:
                raise ValueError("Budget limits must be positive integers")

    def reserve(self, prompt_chars: int) -> int:
        with self._lock:
            if self.stopped_reason:
                raise QwenClientError("LIVE_BUDGET_STOPPED_" + self.stopped_reason)
            if self.used >= self.limit:
                raise QwenClientError("LIVE_HTTP_REQUEST_LIMIT")
            if self.input_chars + prompt_chars > self.max_input_chars:
                raise QwenClientError("LIVE_INPUT_CHARACTER_LIMIT")
            if self.reported_tokens >= self.max_reported_tokens:
                raise QwenClientError("LIVE_REPORTED_TOKEN_LIMIT")
            self.used += 1
            self.input_chars += prompt_chars
            return self.used

    def stop(self, reason: str):
        with self._lock:
            self.stopped_reason = self.stopped_reason or reason

    def record_tokens(self, usage: dict):
        with self._lock:
            tokens = usage.get("total_tokens")
            if type(tokens) is int:
                self.reported_tokens += tokens

    def snapshot(self):
        with self._lock:
            return {
                "http_attempts": self.used,
                "http_attempt_limit": self.limit,
                "input_chars": self.input_chars,
                "input_char_limit": self.max_input_chars,
                "reported_total_tokens": self.reported_tokens,
                "reported_token_stop_threshold": self.max_reported_tokens,
                "stopped_reason": self.stopped_reason,
            }
