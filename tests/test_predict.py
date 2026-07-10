"""ensemble + core.predict mock-LLM 单元测（离线，无需实 key）。"""

from __future__ import annotations

from datetime import date

import pytest

from serenity.agent.core import predict
from serenity.agent.ensemble import run_framework_ensemble
from serenity.agent.frameworks import Market
from serenity.agent.frameworks import GenericAnalyst
from serenity.agent.llm_client import LLMResponse

Q = "Will Pedro Sánchez be the next leader out before 2027?"


class FakeLLM:
    """返回固定 prob 的假 client。fail=True 则每次 complete 抛错。"""

    def __init__(self, model: str, prob: float = 0.3, fail: bool = False):
        self.model = model
        self._prob = prob
        self._fail = fail

    @property
    def training_cutoff(self) -> date:
        return date(2020, 1, 1)  # 早于任何题的 resolve，避免硬污染覆盖

    def complete(self, *, system, user, max_tokens=1024, response_schema=None,
                 estimated_input_tokens=2000, estimated_output_tokens=500) -> LLMResponse:
        if self._fail:
            raise RuntimeError("simulated model outage")
        parsed = {
            "yes_side_interpretation": "YES = the named leader leaves office before 2027-01-01.",
            "prob": self._prob,
            "reasoning": "x" * 60,
            "key_evidence": ["e1", "e2", "e3"],
            "sources_cited": [],
            "contamination_warning": False,
            "contamination_confidence": 0.0,
        }
        return LLMResponse(text="", parsed_json=parsed, model=self.model, cost_usd=0.001)


def _market() -> Market:
    return Market(token_id="tok_test_123456", question=Q, market_price=0.25,
                  resolution_date_iso="2026-12-31")


# ── ensemble D7：存活语义 ──


def test_ensemble_geomean_of_survivors():
    fw = GenericAnalyst
    out = run_framework_ensemble(
        fw, market=_market(), news=[],
        llms=[FakeLLM("m1", prob=0.2), FakeLLM("m2", prob=0.8)], as_of_date="2026-07-01",
    )
    assert out.status == "ok"
    # 几何平均 sqrt(0.2*0.8)=0.4
    assert abs(out.prob - 0.4) < 1e-6
    assert "2/2 models" in out.reasoning


def test_ensemble_one_model_down_still_ok():
    fw = GenericAnalyst
    out = run_framework_ensemble(
        fw, market=_market(), news=[],
        llms=[FakeLLM("m1", prob=0.3), FakeLLM("m2", fail=True)], as_of_date="2026-07-01",
    )
    assert out.status == "ok"
    assert abs(out.prob - 0.3) < 1e-6  # 只剩存活的 m1


def test_ensemble_all_down_fails():
    fw = GenericAnalyst
    out = run_framework_ensemble(
        fw, market=_market(), news=[],
        llms=[FakeLLM("m1", fail=True), FakeLLM("m2", fail=True)], as_of_date="2026-07-01",
    )
    assert out.status == "failed"
    assert "ensemble_all_failed" in out.failure_reason


# ── core.predict 全链 ──


def test_predict_end_to_end():
    pred = predict(
        market=_market(), news=[],
        llms=[FakeLLM("claude-opus-4-8", prob=0.3), FakeLLM("gpt-5.5", prob=0.3)],
        as_of_date=date(2026, 7, 1),
    )
    assert 0.0 <= pred.final_prob <= 1.0
    assert pred.final_prob == pred.raw_prob  # Phase 1 无 calibrator，恒等
    assert pred.aggregated.n_ir_valid >= 1
    assert pred.route_label == "generic"
    assert pred.llm_cost_usd > 0


def test_predict_requires_llms():
    with pytest.raises(ValueError, match="至少一个"):
        predict(market=_market(), news=[], llms=[])


def test_cost_cap_propagates_not_swallowed():
    """预算触顶应向上传播（供 runner 标 cost_cap），不被吞成框架失败。"""
    from serenity.agent.llm_client import CostCapExceeded

    class CapLLM(FakeLLM):
        def complete(self, **kw):
            raise CostCapExceeded("daily cap hit")

    with pytest.raises(CostCapExceeded):
        predict(market=_market(), news=[], llms=[CapLLM("m1")], as_of_date=date(2026, 7, 1))


def test_predict_calibrator_hook_applied():
    pred = predict(
        market=_market(), news=[], llms=[FakeLLM("m1", prob=0.3)],
        as_of_date=date(2026, 7, 1),
        calibrator=lambda raw: min(0.99, raw + 0.1),  # Phase 3 钩子模拟
    )
    assert abs(pred.final_prob - min(0.99, pred.raw_prob + 0.1)) < 1e-9
