"""元认知自检（Forecasting_Bot 6 点清单）。

对聚合结论跑一轮 LLM 复核：复述防 bait-and-switch / 一致性 / blind-spot /
status-quo 惯性，返回可能修正后的 prob。core 记录 self_check_delta 使效果可度量
（Codex：self_check 是又一个变换，必须能被测量而非黑箱）。

降级：LLM 失败 → 返回原 raw_prob（delta=0）。
"""

from __future__ import annotations

import json
import logging

from serenity.agent.core import AgentPrediction, SelfCheckFn
from serenity.agent.llm_client import LLMClient

log = logging.getLogger(__name__)

_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "revised_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "note": {"type": "string"},
    },
    "required": ["revised_prob", "note"],
    "additionalProperties": False,
}

_PROMPT = """You are auditing a forecast before submission. The market question is:
{question}

The ensemble's current YES probability is {prob:.3f}.
Framework interpretations of what YES means:
{interp}

Run this 6-point checklist, then output a possibly-revised probability:
1. Paraphrase the resolution in <30 words. Does {prob:.3f} answer THAT question, not a
   nearby one? (bait-and-switch guard)
2. Base-rate rooting: is {prob:.3f} anchored to a sensible reference-class base rate, or has it
   drifted far from it without strong justification? Outside view first.
3. Consistency: "{prob:.3f} out of 100 times, this resolves YES" — does that read sensibly?
4. Evidence validity: name the 2-3 pieces of evidence {prob:.3f} hinges on; are they
   corroborated and non-speculative? Downweight if the forecast rests on flimsy/uncorroborated claims.
5. Blind spot: name the one scenario most likely to make {prob:.3f} look silly in hindsight.
6. Status-quo: the world changes slowly; if nothing changes, does the base outcome hold?

Only move the probability if the checklist reveals a real error. Small moves are fine; do NOT
rewrite the forecast. Return one JSON object matching the schema.
As of: {as_of}"""


def make_self_check(llm: LLMClient) -> SelfCheckFn:
    """构造 self_check 钩子（供 core.predict 的 self_check= 参数）。"""

    def _check(pred: AgentPrediction) -> float:
        interp = "\n".join(
            f"- {o.framework_name}: {o.yes_side_interpretation[:120]}"
            for o in pred.ir_outputs
            if o.status == "ok" and o.yes_side_interpretation
        ) or "(none)"
        try:
            resp = llm.complete(
                system="You output only JSON.",
                user=_PROMPT.format(
                    question=pred.market.question,
                    prob=pred.raw_prob,
                    interp=interp,
                    as_of=pred.as_of_date.isoformat(),
                ),
                max_tokens=500,
                response_schema=_SCHEMA,
                estimated_input_tokens=600,
                estimated_output_tokens=200,
            )
            parsed = resp.parsed_json or (json.loads(resp.text) if resp.text else None)
            if not isinstance(parsed, dict):
                return pred.raw_prob
            revised = float(parsed["revised_prob"])
            if not 0.0 <= revised <= 1.0:
                return pred.raw_prob
            return revised
        except Exception as e:
            log.warning("self_check failed, keeping raw prob: %s", e)
            return pred.raw_prob

    return _check
