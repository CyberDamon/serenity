"""Base classes for pluggable IR / methodological reasoning frameworks.

A Framework is a structured prompt template that forces the LLM to consider a
specific evidence dimension (military balance / institutions / norms / power
transition / geography / historical base rates). The LLM returns a JSON object
with probability + reasoning + sources; aggregator combines outputs across
frameworks (D7: skip failed, partial=true flag).

ASCII flow:

  market_question + news_context ──▶ Framework.run(llm)
                                            │
                                            ▼
                                  prompt = system + user
                                            │
                                            ▼
                                  llm.complete(response_schema=PROB_SCHEMA)
                                            │
                                            ▼
                                  FrameworkOutput(prob, reasoning, sources,
                                                 contamination_warning, ...)

Failure handling:
  - LLMTransientError after retries → status='failed', failure_reason set
  - Invalid prob (not in [0,1] or missing) → status='failed'
  - LLM returns non-JSON → status='failed'

Aggregator skips failed outputs and flags prediction.partial_aggregation=True.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from serenity.agent.llm_client import (
    CostCapExceeded,
    LLMClient,
    LLMFatalError,
    LLMTransientError,
)
from serenity.data.news.types import NewsItem

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt response schema — every framework's LLM output must match this
# ─────────────────────────────────────────────────────────────────────────────


PROB_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "yes_side_interpretation": {
            "type": "string",
            "minLength": 20,
            "description": (
                "Disambiguation step (REQUIRED — fill this BEFORE prob). State precisely "
                "what 'YES resolves' means for this market in plain English. Examples: "
                "'YES = Donald Trump is the elected US President as of resolution date', "
                "'YES = Iran officially closes its airspace before May 15 2026 (verified by news)', "
                "'YES = the Democratic candidate wins the 2024 US Presidential Election'. "
                "If the question is ambiguous (e.g. 'Which party wins?' — YES could be either "
                "party depending on Polymarket's outcome labels), pick the interpretation that "
                "matches the literal question text, and flag the ambiguity in `reasoning`."
            ),
        },
        "prob": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "Probability that the market resolves YES under the interpretation "
                "you stated in `yes_side_interpretation`."
            ),
        },
        "reasoning": {
            "type": "string",
            "minLength": 50,
            "description": (
                "200-400 word analysis from this framework's perspective. If the question "
                "is ambiguous, explicitly note your interpretation choice and how the prob "
                "would flip under the alternative reading."
            ),
        },
        "key_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-5 bullet evidence points the prob hinges on",
        },
        "sources_cited": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "key_quote": {"type": "string"},
                },
                "required": ["url", "title", "key_quote"],
                "additionalProperties": False,
            },
        },
        "contamination_warning": {
            "type": "boolean",
            "description": "True ONLY if you have specific factual knowledge that this event has ALREADY occurred or ALREADY resolved (from training data or news context). False if topic is merely familiar.",
        },
        "contamination_confidence": {
            "type": "number",
            "description": "How strong is the contamination warning (0=clean, 1=definitely contaminated)",
        },
    },
    "required": [
        "yes_side_interpretation",
        "prob",
        "reasoning",
        "key_evidence",
        "sources_cited",
        "contamination_warning",
        "contamination_confidence",
    ],
    "additionalProperties": False,
}


@dataclass
class FrameworkOutput:
    """One framework's contribution to a prediction."""

    framework_name: str
    status: str  # 'ok' | 'failed' | 'timeout'
    prob: float | None = None
    yes_side_interpretation: str = ""
    reasoning: str = ""
    key_evidence: list[str] = field(default_factory=list)
    sources_cited: list[dict] = field(default_factory=list)
    contamination_warning: bool = False
    contamination_confidence: float = 0.0
    failure_reason: str = ""
    llm_call_cost_usd: float = 0.0
    llm_model: str = ""
    reference_class_n: int | None = None
    reference_class_confidence: float = 1.0  # 匹配质量置信度（Codex⑤：乘入锚权重）


@dataclass
class Market:
    """Subset of market fields needed by frameworks. Not the DB model."""

    token_id: str
    question: str
    market_price: float  # current YES midpoint
    resolution_date_iso: str | None = None


_NEWS_VERIFICATION_PREAMBLE = """STEP -1 — News verification gate (READ FIRST, BEFORE ANY ANALYSIS).

The news context below may contain hypothetical, future-conditional, op-ed, or
analytical articles — not all of them describe events that have actually happened.
A single article asserting "X did Y in February 2026" is NOT proof X did Y.

Before you treat any article as evidence the market should resolve YES:
  - Check the verb tense and modality. "could", "might", "would", "is expected to",
    "may" → speculation. "did", "has", "was confirmed" → claimed factual report.
  - Check whether multiple independent sources corroborate the same event with
    consistent dates.
  - Check the publisher: an op-ed or analysis piece (e.g. "War on the Rocks",
    "Foreign Affairs" essays) is NOT a factual newswire report.
  - Note that the agent's training data may "leak" outcomes — if you find yourself
    "remembering" the event happened, set contamination_warning=true and
    DOWNWEIGHT, do not upweight.

If the market price (the crowd's bet) disagrees with what your news context
seems to suggest, the more likely explanation is that the news is speculative or
mis-read — not that the market is wildly wrong. Calibrate prob CLOSER to
market_price, not farther, when news evidence is hypothetical, opinion-based, or
uncorroborated.

After completing this gate, proceed to the framework-specific steps below.

"""


def _format_news_block(news: list[NewsItem], max_articles: int = 8) -> str:
    """Render NewsItems as a numbered block for the LLM prompt.

    Tags from pre_filter are shown as bracketed labels on the headline line so
    each framework's LLM call sees structured event-type context without having
    to re-extract it from raw prose.
    """
    if not news:
        return "(no recent news available)"
    lines: list[str] = []
    for i, item in enumerate(news[:max_articles], 1):
        date_str = item.published_at.strftime("%Y-%m-%d")
        src = item.source
        tag_str = ""
        if item.tags:
            tag_str = " " + " ".join(f"[{t.upper()}]" for t in item.tags)
        lines.append(f"[{i}] {date_str} | {src}{tag_str} | {item.title}")
        if item.summary:
            lines.append(f"    {item.excerpt(max_chars=300)}")
        if item.url:
            lines.append(f"    URL: {item.url}")
    return "\n".join(lines)


class Framework(ABC):
    """Abstract base for pluggable analytical frameworks.

    Subclasses define:
      - name: short identifier stored in framework_outputs.framework_name
      - system_prompt: framework-specific instructions (the "lens")
      - user_template: format(question=, news_block=, market_price=, as_of=) → str

    The base `run(market, news, llm, as_of_date)` handles:
      - prompt assembly
      - LLM call with PROB_SCHEMA
      - exception → FrameworkOutput(status='failed', failure_reason=...)
      - prob validation + clipping
    """

    name: ClassVar[str] = ""
    system_prompt: ClassVar[str] = ""
    user_template: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Sanity check — catch missing class attributes at import time.
        if not cls.name:
            raise TypeError(f"{cls.__name__} must set class attribute `name`")
        if not cls.system_prompt:
            raise TypeError(f"{cls.__name__} must set class attribute `system_prompt`")

    def render_user_prompt(
        self, market: Market, news: list[NewsItem], as_of_date: str
    ) -> str:
        resolution_date = market.resolution_date_iso or "unknown (infer from question)"
        body = self.user_template.format(
            question=market.question,
            market_price=market.market_price,
            news_block=_format_news_block(news),
            as_of_date=as_of_date,
            resolution_date=resolution_date,
        )
        return _NEWS_VERIFICATION_PREAMBLE + body

    @abstractmethod
    def post_process(self, output: FrameworkOutput, parsed: dict) -> None:
        """Hook for framework-specific extraction from parsed LLM JSON.

        Override to read framework-specific extra fields. By default does nothing.
        """

    def run(
        self,
        *,
        market: Market,
        news: list[NewsItem],
        llm: LLMClient,
        as_of_date: str,
    ) -> FrameworkOutput:
        """Execute the framework on this market. Always returns a FrameworkOutput
        (never raises) — failures are encoded as status='failed'.

        V9: retry once on missing_prob / missing_parsed_json — common Opus 4.7
        failure mode is returning a partial JSON without the required `prob`
        field. A second call usually succeeds.
        """
        out = self._run_once(market, news, llm, as_of_date)
        if out.status == "failed" and out.failure_reason in {
            "missing_prob",
            "missing_parsed_json",
            "non_json_response",
            "prob_out_of_range",
        }:
            log.info(
                "framework %s retry after %s on token=%s",
                self.name, out.failure_reason, market.token_id[:16],
            )
            retry = self._run_once(market, news, llm, as_of_date)
            if retry.status == "ok":
                out = retry
            # If retry also failed but with a different reason, keep the
            # original out's reason for debugging.
        self._log_reasoning(out, market)
        return out

    def _log_reasoning(self, out: FrameworkOutput, market: Market) -> None:
        """Emit the framework's reasoning so it surfaces in the run log."""
        if out.status != "ok" or not out.reasoning:
            return
        prob_text = f"{out.prob:.3f}" if out.prob is not None else "-"
        details = out.reasoning.strip()
        if out.key_evidence:
            evidence = "\n".join(f"  • {e}" for e in out.key_evidence)
            details = f"{details}\n  关键证据 (key evidence):\n{evidence}"
        log.info(
            "framework %s reasoning (token=%s, prob=%s):\n%s",
            self.name,
            market.token_id[:16],
            prob_text,
            details,
        )

    def _run_once(
        self,
        market: Market,
        news: list[NewsItem],
        llm: LLMClient,
        as_of_date: str,
    ) -> FrameworkOutput:
        """Single LLM round-trip + JSON parse. Used by run() and its retry."""
        out = FrameworkOutput(framework_name=self.name, status="ok", llm_model=llm.model)
        user_p = self.render_user_prompt(market, news, as_of_date)
        try:
            resp = llm.complete(
                system=self.system_prompt,
                user=user_p,
                max_tokens=1500,
                response_schema=PROB_SCHEMA,
                estimated_input_tokens=2500,
                estimated_output_tokens=800,
            )
        except CostCapExceeded:
            # 预算触顶：向上传播，让 runner 干净地"停止拉新题+标 cost_cap"，
            # 而不是被下面的通用 except 吞成"框架失败"（会伪装成预测质量差）。
            raise
        except LLMTransientError as e:
            out.status = "failed"
            out.failure_reason = f"transient_after_retries: {e}"
            return out
        except LLMFatalError as e:
            out.status = "failed"
            out.failure_reason = f"fatal: {e}"
            return out
        except Exception as e:
            out.status = "failed"
            out.failure_reason = f"unexpected: {type(e).__name__}: {e}"
            return out

        out.llm_call_cost_usd = resp.cost_usd

        # Parse: prefer tool_use parsed_json; fall back to text JSON
        parsed: dict | None = resp.parsed_json
        if parsed is None and resp.text:
            try:
                parsed = json.loads(resp.text)
            except json.JSONDecodeError as e:
                out.status = "failed"
                out.failure_reason = f"non_json_response: {e}"
                return out
        if not isinstance(parsed, dict):
            out.status = "failed"
            out.failure_reason = "missing_parsed_json"
            return out

        # Validate prob
        prob = parsed.get("prob")
        if not isinstance(prob, (int, float)):
            out.status = "failed"
            out.failure_reason = "missing_prob"
            return out
        if not (0.0 <= prob <= 1.0):
            out.status = "failed"
            out.failure_reason = f"prob_out_of_range: {prob}"
            return out

        out.prob = float(prob)
        out.yes_side_interpretation = parsed.get("yes_side_interpretation", "")
        out.reasoning = parsed.get("reasoning", "")
        out.key_evidence = list(parsed.get("key_evidence") or [])
        out.sources_cited = list(parsed.get("sources_cited") or [])
        out.contamination_warning = bool(parsed.get("contamination_warning", False))
        out.contamination_confidence = float(parsed.get("contamination_confidence", 0.0))

        # Hard contamination override: if the market's resolution date falls
        # before the LLM's training cutoff, the model almost certainly "knows"
        # the outcome from training data — self-reported warning is unreliable.
        # Override both flags regardless of what the LLM said.
        if market.resolution_date_iso:
            try:
                from datetime import date as _date
                res_date = _date.fromisoformat(market.resolution_date_iso)
                cutoff = llm.training_cutoff
                if res_date <= cutoff:
                    if not out.contamination_warning:
                        log.info(
                            "hard contamination flag: %s resolution=%s <= cutoff=%s",
                            self.name, res_date, cutoff,
                        )
                    out.contamination_warning = True
                    # Escalate confidence to at least 0.8 for hard-override cases
                    out.contamination_confidence = max(out.contamination_confidence, 0.8)
            except ValueError:
                pass  # unparseable date — leave LLM self-report as-is

        try:
            self.post_process(out, parsed)
        except Exception as e:
            log.warning("post_process error in %s: %s", self.name, e)
        return out
