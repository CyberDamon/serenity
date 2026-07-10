"""Unified LLM client abstraction.

Why this exists: serenity runs frontier models for forward live predictions and
legacy models for backtest. They share the same prompt/response contract, so
framework code calls `llm.complete(...)` without caring which model is behind it.

Cross-provider design:
  - NewAPIClient handles current WorldRouter/New API models through an
    OpenAI-compatible /chat/completions endpoint
  - AnthropicClient handles both frontier + legacy Claude (same SDK, different model strings)
  - OpenAIClient handles GPT-4-0613 for cross-LLM backtest validation (V1)

All clients:
  - Enforce JSON output via response_schema
  - Respect daily cost cap via shared CostTracker (in-process; later: Redis)
  - Retry transient errors via tenacity (exponential backoff)
  - Surface model.training_cutoff so temporal_guard can verify backtest correctness

ASCII flow:

  framework.run() ──▶ llm.complete(system, user, schema, cost_estimate)
                              │
                              ▼
                       CostTracker.check_budget()
                              │
                              ├── over budget? ──▶ raise CostCapExceeded
                              ▼
                       tenacity retry (3x exp backoff, transient errors only)
                              │
                              ▼
                       New API / Anthropic / OpenAI SDK
                              │
                              ▼
                       parse JSON via response_schema (strict)
                              │
                              ▼
                       CostTracker.record(actual_cost)
                              │
                              ▼
                       return LLMResponse(text, cost_usd, ...)

NOT in this module (deferred to V1):
  - Streaming (we collect full response)
  - Prompt caching (Anthropic supports it but adds complexity)
  - Multi-model fallback chain (one client = one provider; orchestrator picks)
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from serenity.config import settings

log = logging.getLogger(__name__)

# 单次 LLM 调用超时（秒）。防止某次调用挂死拖垮整题/整轮；配合 tenacity 重试。
# opus thinking 较慢但不应超过此值。SDK 自带 max_retries=0（重试交给 tenacity，避免双重）。
_LLM_TIMEOUT_S = 150.0


# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Single completion result."""

    text: str
    parsed_json: dict | None = None
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class CostCapExceeded(RuntimeError):
    """Raised when a request would exceed the daily cost cap."""


class LLMTransientError(RuntimeError):
    """Wrapped transient errors (rate limit, 5xx, network) — eligible for retry."""


class LLMFatalError(RuntimeError):
    """Wrapped fatal errors (auth, invalid request) — not eligible for retry."""


# ─────────────────────────────────────────────────────────────────────────────
# Cost tracker
# ─────────────────────────────────────────────────────────────────────────────


class CostTracker:
    """Thread-safe in-process daily cost tracker.

    Tracks per-day spend keyed by UTC date. Reset on date roll. Multi-process
    coordination is not supported in v0 (single cron worker); V1 may move to Redis.
    """

    def __init__(self, daily_cap_usd: float) -> None:
        self._lock = threading.Lock()
        self._cap = daily_cap_usd
        self._today: date = datetime.now(UTC).date()
        self._spend: float = 0.0

    def _roll_if_needed(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._today:
            log.info("cost_tracker: rolling to new day %s, prev spend=$%.2f", today, self._spend)
            self._today = today
            self._spend = 0.0

    def check_budget(self, estimated_cost_usd: float) -> None:
        """Raise CostCapExceeded if this request would exceed the cap.（不记账，仅探测）"""
        with self._lock:
            self._roll_if_needed()
            if self._spend + estimated_cost_usd > self._cap:
                raise CostCapExceeded(
                    f"daily LLM cost cap hit: spent=${self._spend:.2f} "
                    f"estimated=+${estimated_cost_usd:.4f} cap=${self._cap:.2f}"
                )

    def reserve(self, estimated_cost_usd: float) -> None:
        """原子预留：锁内"检查+记账"合一，防并发请求同时过检查后超 cap。

        超 cap 则 raise CostCapExceeded 且不记账。调用后用 record(actual-estimated) 结算差额。
        """
        with self._lock:
            self._roll_if_needed()
            if self._spend + estimated_cost_usd > self._cap:
                raise CostCapExceeded(
                    f"daily LLM cost cap hit: spent=${self._spend:.2f} "
                    f"estimated=+${estimated_cost_usd:.4f} cap=${self._cap:.2f}"
                )
            self._spend += estimated_cost_usd

    def record(self, delta_usd: float) -> None:
        """调整已记账额（可正可负）：reserve 预留估算后，用实际值结算差额。"""
        with self._lock:
            self._roll_if_needed()
            self._spend += delta_usd

    @property
    def today_spend(self) -> float:
        with self._lock:
            self._roll_if_needed()
            return self._spend


# Singleton — single cron worker means one tracker is sufficient
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker(settings.daily_llm_cost_cap_usd)
    return _cost_tracker


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — knows training cutoffs for backtest temporal_guard
# ─────────────────────────────────────────────────────────────────────────────


# Pricing table (USD per 1M tokens) — source: provider docs as of 2026-05.
# Used for cost estimation BEFORE the call. Actual is recorded after based on
# response.usage. Update when providers reprice.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # Anthropic — current frontier (serenity 默认 ensemble 之一)
    # training_cutoff 为保守占位；纯前向下 temporal_guard 基本不触发（开放项2 待核实）。
    "claude-opus-4-8": {
        "provider": "new_api",
        "training_cutoff": date(2026, 1, 1),
        "input_cost_per_mtok": 15.0,
        "output_cost_per_mtok": 75.0,
    },
    "claude-sonnet-5": {
        "provider": "new_api",
        "training_cutoff": date(2026, 1, 1),
        "input_cost_per_mtok": 3.0,
        "output_cost_per_mtok": 15.0,
    },
    "claude-opus-4-7": {
        "provider": "new_api",
        "training_cutoff": date(2026, 1, 1),
        "input_cost_per_mtok": 15.0,
        "output_cost_per_mtok": 75.0,
    },
    "claude-sonnet-4-6-20251001": {
        "provider": "anthropic",
        "training_cutoff": date(2025, 7, 1),
        "input_cost_per_mtok": 3.0,
        "output_cost_per_mtok": 15.0,
    },
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "training_cutoff": date(2025, 7, 1),
        "input_cost_per_mtok": 1.0,
        "output_cost_per_mtok": 5.0,
    },
    # Legacy Claude — backtest only (training cutoff < target backtest as_of dates)
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "training_cutoff": date(2024, 4, 1),  # Anthropic-published cutoff
        "input_cost_per_mtok": 3.0,
        "output_cost_per_mtok": 15.0,
    },
    # Alias for older reference (deprecated by Anthropic, kept for compat)
    "claude-3-5-sonnet-20240620": {
        "provider": "anthropic",
        "training_cutoff": date(2024, 4, 1),
        "input_cost_per_mtok": 3.0,
        "output_cost_per_mtok": 15.0,
    },
    # OpenAI GPT-5 family (Aug 2025 release) — current frontier
    "gpt-5.5": {
        "provider": "new_api",
        "training_cutoff": date(2024, 10, 1),
        "input_cost_per_mtok": 10.0,
        "output_cost_per_mtok": 45.0,
    },
    "gpt-5": {
        "provider": "openai",
        "training_cutoff": date(2024, 10, 1),
        "input_cost_per_mtok": 1.25,
        "output_cost_per_mtok": 10.0,
    },
    "gpt-5-mini": {
        "provider": "openai",
        "training_cutoff": date(2024, 10, 1),
        "input_cost_per_mtok": 0.25,
        "output_cost_per_mtok": 2.0,
    },
    "gpt-5-nano": {
        "provider": "openai",
        "training_cutoff": date(2024, 10, 1),
        "input_cost_per_mtok": 0.05,
        "output_cost_per_mtok": 0.40,
    },
    # OpenAI GPT-4o — forward predictions (best reasoning, affordable)
    "gpt-4o": {
        "provider": "openai",
        "training_cutoff": date(2024, 5, 1),  # conservative estimate for latest alias
        "input_cost_per_mtok": 2.5,
        "output_cost_per_mtok": 10.0,
    },
    "gpt-4o-2024-08-06": {
        "provider": "openai",
        "training_cutoff": date(2024, 5, 1),
        "input_cost_per_mtok": 2.5,
        "output_cost_per_mtok": 10.0,
    },
    "gpt-4o-2024-11-20": {
        "provider": "openai",
        "training_cutoff": date(2024, 10, 1),
        "input_cost_per_mtok": 2.5,
        "output_cost_per_mtok": 10.0,
    },
    # Legacy OpenAI — backtest cross-validation (very old cutoff = clean)
    "gpt-4-0613": {
        "provider": "openai",
        "training_cutoff": date(2021, 9, 1),
        "input_cost_per_mtok": 30.0,
        "output_cost_per_mtok": 60.0,
    },
}


def get_training_cutoff(model: str) -> date:
    """For temporal_guard: assert as_of_date < cutoff to prevent backtest contamination."""
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model!r}; add to MODEL_REGISTRY before backtest use")
    return MODEL_REGISTRY[model]["training_cutoff"]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    cfg = MODEL_REGISTRY[model]
    return (input_tokens / 1e6) * cfg["input_cost_per_mtok"] + (output_tokens / 1e6) * cfg[
        "output_cost_per_mtok"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Client interface
# ─────────────────────────────────────────────────────────────────────────────


class LLMClient(Protocol):
    """Provider-agnostic completion client.

    Implementations: AnthropicClient, OpenAIClient.
    """

    model: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        response_schema: dict | None = None,
        estimated_input_tokens: int = 2000,
        estimated_output_tokens: int = 500,
    ) -> LLMResponse:
        """Run a single completion. Raises CostCapExceeded if over budget."""
        ...

    @property
    def training_cutoff(self) -> date:
        """For temporal_guard."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# New API implementation (OpenAI-compatible router)
# ─────────────────────────────────────────────────────────────────────────────


def _new_api_base_url() -> str:
    base = settings.new_api_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _usage_tokens(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "input_tokens", 0)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", 0)
    return int(prompt_tokens or 0), int(completion_tokens or 0)


class NewAPIClient:
    """OpenAI-compatible client for the configured NEW_API router."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        if model not in MODEL_REGISTRY or MODEL_REGISTRY[model]["provider"] != "new_api":
            raise ValueError(f"{model!r} not in New API registry")
        self.model = model
        self._api_key = settings.new_api_key.get_secret_value() if api_key is None else api_key
        self._base_url = (base_url.rstrip("/") if base_url else _new_api_base_url())
        self._cost_tracker = get_cost_tracker()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise LLMFatalError("openai SDK not installed; pip install openai") from e
            if not self._api_key:
                raise LLMFatalError("NEW_API_KEY not set")
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=_LLM_TIMEOUT_S, max_retries=0)
        return self._client

    @property
    def training_cutoff(self) -> date:
        return MODEL_REGISTRY[self.model]["training_cutoff"]

    @retry(
        retry=retry_if_exception_type(LLMTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=20),
        reraise=True,
    )
    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        response_schema: dict | None = None,
        estimated_input_tokens: int = 2000,
        estimated_output_tokens: int = 500,
    ) -> LLMResponse:
        estimated = estimate_cost(self.model, estimated_input_tokens, estimated_output_tokens)
        self._cost_tracker.reserve(estimated)

        client = self._ensure_client()
        is_reasoning = self.model.startswith(("gpt-5", "claude-opus-4"))
        effective_max_tokens = max_tokens * 4 if is_reasoning else max_tokens
        user_content = user
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": effective_max_tokens,
        }
        if response_schema is not None:
            schema_hint = json.dumps(response_schema, ensure_ascii=False)
            kwargs["messages"][-1]["content"] += (
                "\n\nCRITICAL: Respond with one valid JSON object matching this JSON Schema. "
                "No markdown fences. No prose outside the JSON object.\n"
                f"JSON Schema: {schema_hint}"
            )
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            cls_name = type(e).__name__
            if cls_name in {"RateLimitError", "APIConnectionError", "APITimeoutError"}:
                raise LLMTransientError(f"{cls_name}: {e}") from e
            if cls_name == "APIStatusError" and getattr(e, "status_code", 0) >= 500:
                raise LLMTransientError(f"{cls_name}: {e}") from e
            raise LLMFatalError(f"{cls_name}: {e}") from e

        choice = resp.choices[0]
        text = choice.message.content or ""
        parsed_json: dict | None = None
        if response_schema is not None and text:
            try:
                parsed_json = json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMFatalError(f"New API returned non-JSON despite schema prompt: {e}") from e

        input_tokens, output_tokens = _usage_tokens(resp.usage)
        actual_cost = estimate_cost(self.model, input_tokens, output_tokens)
        self._cost_tracker.record(actual_cost - estimated)

        return LLMResponse(
            text=text,
            parsed_json=parsed_json,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=actual_cost,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic implementation
# ─────────────────────────────────────────────────────────────────────────────


class AnthropicClient:
    """Wraps anthropic SDK. Handles both frontier + legacy Claude models."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        if model not in MODEL_REGISTRY or MODEL_REGISTRY[model]["provider"] != "anthropic":
            raise ValueError(f"{model!r} not in Anthropic registry")
        self.model = model
        self._api_key = settings.anthropic_api_key.get_secret_value() if api_key is None else api_key
        self._cost_tracker = get_cost_tracker()
        self._client = None  # lazy init so tests can mock without API key

    def _ensure_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic  # lazy import
            except ImportError as e:
                raise LLMFatalError("anthropic SDK not installed; pip install anthropic") from e
            if not self._api_key:
                raise LLMFatalError("ANTHROPIC_API_KEY not set")
            self._client = Anthropic(api_key=self._api_key, timeout=_LLM_TIMEOUT_S, max_retries=0)
        return self._client

    @property
    def training_cutoff(self) -> date:
        return MODEL_REGISTRY[self.model]["training_cutoff"]

    @retry(
        retry=retry_if_exception_type(LLMTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=20),
        reraise=True,
    )
    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        response_schema: dict | None = None,
        estimated_input_tokens: int = 2000,
        estimated_output_tokens: int = 500,
    ) -> LLMResponse:
        # Budget check first — saves API call when capped
        estimated = estimate_cost(self.model, estimated_input_tokens, estimated_output_tokens)
        self._cost_tracker.reserve(estimated)

        client = self._ensure_client()

        # claude-opus-4-x 是 extended thinking 模型，内部推理会消耗大量 token。
        # 若 max_tokens 不足，tool_use 的 input 会被截断（prob 字段丢失）。
        # 与 OpenAIClient 对 GPT-5 的处理保持一致：给 Opus 4.x 乘以 5 倍空间。
        is_thinking = self.model.startswith("claude-opus-4")
        effective_max_tokens = max_tokens * 5 if is_thinking else max_tokens

        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": effective_max_tokens,
        }
        # Enforce JSON via tool_use when schema provided.
        if response_schema is not None:
            kwargs["tools"] = [
                {
                    "name": "submit_prediction",
                    "description": "Submit structured prediction output.",
                    "input_schema": response_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "submit_prediction"}

        try:
            resp = client.messages.create(**kwargs)
        except Exception as e:
            # anthropic.RateLimitError / APIConnectionError / APIStatusError(5xx) → transient
            # anthropic.AuthenticationError / BadRequestError → fatal
            cls_name = type(e).__name__
            if cls_name in {"RateLimitError", "APIConnectionError", "APITimeoutError"}:
                raise LLMTransientError(f"{cls_name}: {e}") from e
            if cls_name == "APIStatusError" and getattr(e, "status_code", 0) >= 500:
                raise LLMTransientError(f"{cls_name}: {e}") from e
            raise LLMFatalError(f"{cls_name}: {e}") from e

        # Parse response: text from content blocks; tool_use input becomes parsed_json
        text_parts: list[str] = []
        parsed_json: dict | None = None
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                parsed_json = block.input  # already a dict per SDK
        text = "\n".join(text_parts)

        actual_cost = estimate_cost(self.model, resp.usage.input_tokens, resp.usage.output_tokens)
        self._cost_tracker.record(actual_cost - estimated)

        return LLMResponse(
            text=text,
            parsed_json=parsed_json,
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=actual_cost,
        )


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI implementation (legacy backtest cross-validation)
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIClient:
    """Wraps openai SDK. Used for GPT-4-0613 backtest cross-validation (V1)."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        if model not in MODEL_REGISTRY or MODEL_REGISTRY[model]["provider"] != "openai":
            raise ValueError(f"{model!r} not in OpenAI registry")
        self.model = model
        self._api_key = settings.openai_api_key.get_secret_value() if api_key is None else api_key
        self._cost_tracker = get_cost_tracker()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise LLMFatalError("openai SDK not installed; pip install openai") from e
            if not self._api_key:
                raise LLMFatalError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=self._api_key, timeout=_LLM_TIMEOUT_S, max_retries=0)
        return self._client

    @property
    def training_cutoff(self) -> date:
        return MODEL_REGISTRY[self.model]["training_cutoff"]

    @retry(
        retry=retry_if_exception_type(LLMTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=20),
        reraise=True,
    )
    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        response_schema: dict | None = None,
        estimated_input_tokens: int = 2000,
        estimated_output_tokens: int = 500,
    ) -> LLMResponse:
        estimated = estimate_cost(self.model, estimated_input_tokens, estimated_output_tokens)
        self._cost_tracker.reserve(estimated)

        client = self._ensure_client()
        # GPT-5 / o-series are reasoning models: an undisclosed chunk of the
        # token budget goes to internal reasoning before any visible output
        # tokens are produced. With max_completion_tokens=1500 the model often
        # spends all 1500 on reasoning and returns an empty completion. We
        # multiply the caller's budget by 4 for these models so visible
        # output has room (input cost is unchanged; only output is metered).
        is_reasoning = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
        token_param = "max_completion_tokens" if is_reasoning else "max_tokens"
        effective_max_tokens = max_tokens * 4 if is_reasoning else max_tokens
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            token_param: effective_max_tokens,
        }
        # GPT-5 supports `reasoning_effort` to cap how much it ruminates;
        # "medium" is the sweet spot for serenity's prompt complexity.
        if self.model.startswith("gpt-5"):
            kwargs["reasoning_effort"] = "medium"
        if response_schema is not None:
            # Model capability tiers for structured output:
            #   gpt-4-0613 and older: no response_format support at all — use prompt injection
            #   gpt-4-turbo-preview and later: json_object mode
            #   gpt-4o and later: json_schema + strict mode
            _no_format = self.model in {"gpt-4-0613", "gpt-4-0314", "gpt-3.5-turbo-0613"}
            _supports_strict = not _no_format and self.model not in {
                "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4-turbo-2024-04-09"
            }
            if _no_format:
                # Inject JSON instruction into user message; parse free-text response
                schema_hint = ", ".join(response_schema.get("required", []))
                kwargs["messages"][-1]["content"] += (
                    f"\n\nCRITICAL: Respond with a SINGLE valid JSON object. "
                    f"Required fields: {schema_hint}. "
                    "No markdown fences. No prose. Start your response with { and end with }."
                )
            elif _supports_strict:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "prediction",
                        "schema": response_schema,
                        "strict": True,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
                schema_hint = ", ".join(response_schema.get("required", []))
                kwargs["messages"][-1]["content"] += (
                    f"\n\nOutput a single JSON object with these required fields: {schema_hint}. "
                    "No markdown fences. Start with {{ and end with }}."
                )

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            cls_name = type(e).__name__
            if cls_name in {"RateLimitError", "APIConnectionError", "APITimeoutError"}:
                raise LLMTransientError(f"{cls_name}: {e}") from e
            if cls_name == "APIStatusError" and getattr(e, "status_code", 0) >= 500:
                raise LLMTransientError(f"{cls_name}: {e}") from e
            raise LLMFatalError(f"{cls_name}: {e}") from e

        choice = resp.choices[0]
        text = choice.message.content or ""
        parsed_json: dict | None = None
        if response_schema is not None and text:
            import json as _json

            try:
                parsed_json = _json.loads(text)
            except _json.JSONDecodeError as e:
                raise LLMFatalError(f"OpenAI returned non-JSON despite strict schema: {e}") from e

        actual_cost = estimate_cost(
            self.model,
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
        )
        self._cost_tracker.record(actual_cost - estimated)

        return LLMResponse(
            text=text,
            parsed_json=parsed_json,
            model=self.model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            cost_usd=actual_cost,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def make_client(model: str) -> LLMClient:
    """Build a client for the named model. Routes by provider in MODEL_REGISTRY."""
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model!r}")
    provider = MODEL_REGISTRY[model]["provider"]
    if provider == "new_api":
        return NewAPIClient(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai":
        return OpenAIClient(model)
    raise ValueError(f"unsupported provider {provider!r}")


def make_frontier_client() -> LLMClient:
    """Forward live default. Honors settings.llm_frontier_model."""
    model = settings.llm_frontier_model
    return make_client(model)


def make_legacy_client() -> LLMClient:
    """Backtest default — Claude 3.5 Sonnet (cutoff 2024-04)."""
    return make_client(settings.llm_legacy_model)
