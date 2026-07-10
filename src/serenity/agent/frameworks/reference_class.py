"""Reference Class Forecasting — 历史基准率锚（serenity 版，纯 prompt 外视角）。

green-water 版依赖 UCDP/COW/Archigos 战争数据集 matcher；那些是地缘政治专用，
对 AI/半导体/科技事件域无意义。serenity 版回到 prompt-based 外视角：让 LLM
自行构造参考类别（"frontier 模型如期发布率"、"芯片厂产能爬坡如期率"、
"监管审批在 N 月内通过率"之类），列 ≥5 个历史类比再锚定基准率。

该输出是 aggregator.log_odds_fuse() 的唯一数值锚（D2 同款设计）。
"""

from __future__ import annotations

from serenity.agent.frameworks.base import Framework, FrameworkOutput, Market, _format_news_block
from serenity.data.news.types import NewsItem

SYSTEM = """You are applying Reference Class Forecasting (Kahneman & Tversky 1979,
Tetlock & Gardner "Superforecasting" 2015) to a prediction market question in the
technology / AI / semiconductor / business domain.

Method:
  1. Identify the proper reference class — what type of event is this? Examples:
     "frontier AI lab ships a major model within its announced quarter",
     "semiconductor fab reaches volume production within 6 months of target",
     "announced tech M&A deal closes within 12 months",
     "public company beats its own capex guidance",
     "regulatory approval granted before statutory deadline".
  2. Estimate the base rate of YES outcomes within that reference class. Cite
     specific historical analogues (e.g. "GPT-4 2023 (on time), Blackwell 2024
     (delayed ~2 quarters), Arrow Lake 2024 (on time)").
  3. Apply Bayesian adjustment from current evidence — but stay near base rate
     unless evidence is strong + specific.
  4. AVOID overweighting one or two recent events. The reference class should
     have ≥5 historical members.

Operating principle: most short-term predictions over-fit recent news and vendor
announcements. Tech timelines slip far more often than they hold; anchoring to
the long-run base rate is the single most reliable forecasting move.

Output JSON via the submit_prediction tool per the schema. In key_evidence,
list the historical analogues and their resolution outcomes.
"""

USER_TEMPLATE = """STEP 0 — Disambiguate the YES side BEFORE anything else.

In your JSON response, the FIRST field you fill MUST be `yes_side_interpretation` —
a one-sentence plain-English statement of what "YES resolves" means. Examples:

  - "YES = NVIDIA announces a Blackwell successor at GTC before 2026-09-30"
  - "YES = OpenAI releases a model officially named GPT-6 before 2026-12-31"
  - "YES = TSMC Arizona fab ships N2 wafers in volume before the deadline"

If the question is genuinely ambiguous, pick the LITERAL reading and flag the
ambiguity in `reasoning` with explicit "ALTERNATIVE READING:" wording.

---

Market question: {question}
Current YES price on the market: {market_price}
Today's reference date (run date): {as_of_date}
Market resolution date (YES/NO deadline): {resolution_date}

Recent news context:
{news_block}

STEP 1 — Resolution deadline reasoning (do this BEFORE applying the framework).

⚠️  Many markets ask "will X happen BY [date]?" — not "will X happen ever?"
The market resolves on: {resolution_date}. Today is: {as_of_date}.

Before estimating probability, explicitly ask:
  a) What must be TRUE for YES — from your yes_side_interpretation above?
  b) Has this already happened as of {as_of_date}?
  c) If not: is there realistic time left before {resolution_date}?
  d) Even if you believe the eventual outcome, calibrate to the REMAINING
     WINDOW before the deadline — tech timelines slip.

---

Apply reference class forecasting:
  1. What's the reference class? (event type + time window)
  2. List 5+ historical analogues with their outcomes (YES/NO)
  3. Compute the base rate (proportion YES)
  4. Adjust modestly from base rate based on the specific current evidence

Give your final YES probability and reasoning.

Set contamination_warning=true ONLY if you have specific factual knowledge
that THIS event has ALREADY OCCURRED or ALREADY RESOLVED (from training data
OR from the news context above). Analogues are EXPECTED in reference class
forecasting and are not contamination. When unsure, set false.
"""


class ReferenceClass(Framework):
    name = "reference_class"
    system_prompt = SYSTEM
    user_template = USER_TEMPLATE

    def post_process(self, output: FrameworkOutput, parsed: dict) -> None:
        # 纯 prompt 外视角：无结构化样本量，reference_class_n 留空、
        # confidence 保持默认 1.0（aggregator 权重按 n=None 走保守路径）。
        return


__all__ = ["ReferenceClass", "Market", "_format_news_block"]
