"""prior 单元测：δ 网格映射 / 先缩放后封顶 / 引用必须 / fail-closed / placebo。"""

from __future__ import annotations

import pytest

from serenity.gate.gate import GateResult
from serenity.prior.prior import (
    PriorResult,
    _generate,
    _map_delta,
    apply_delta,
    generate_placebo_prior,
    generate_prior,
    retrieve_beliefs,
    sample_placebo_beliefs,
    RetrievedBelief,
)


def _gate_in():
    return GateResult(state="in_domain", rationale="规则直判：命中 NVDA")


def _gate_adj():
    return GateResult(state="adjacent", rationale="Fed rates → neocloud_financing one hop")


def _beliefs(n=3):
    return [
        RetrievedBelief(id=i, claim=f"belief {i} about supply chain bottlenecks",
                        domain="optics_cpo", tickers="NVDA", stance="bullish",
                        confidence="high")
        for i in range(1, n + 1)
    ]


class _ScriptedLLM:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.model = "scripted"
        self.n_calls = 0

    def complete(self, **kw):
        from serenity.agent.llm_client import LLMResponse
        self.n_calls += 1
        item = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text="", parsed_json=item, model=self.model, cost_usd=0.001)


def _payload(direction="yes", strength="strong", ids=(1,)):
    return {"direction": direction, "strength": strength,
            "rationale": "capex flows into optics; cited beliefs directly address the mechanism",
            "belief_ids": list(ids)}


# ── δ 网格映射（全枚举 × in/adjacent，评审定稿网格）──


@pytest.mark.parametrize("strength,expected", [("weak", 0.10), ("moderate", 0.20), ("strong", 0.35)])
def test_grid_in_domain(strength, expected):
    assert _map_delta("yes", strength, "in_domain") == pytest.approx(expected)
    assert _map_delta("no", strength, "in_domain") == pytest.approx(-expected)


@pytest.mark.parametrize("strength,expected", [("weak", 0.05), ("moderate", 0.10), ("strong", 0.175)])
def test_grid_adjacent_halved(strength, expected):
    assert _map_delta("yes", strength, "adjacent") == pytest.approx(expected)


def test_direction_none_is_zero():
    assert _map_delta("none", "none", "in_domain") == 0.0


def test_scale_then_cap_order(monkeypatch):
    """先缩放后封顶：0.8 × strong(0.35) = 0.28 ≤ 0.35（评审 8A 断言）。"""
    from serenity.config import settings
    monkeypatch.setattr(settings, "prior_scale", 0.8)
    assert _map_delta("yes", "strong", "in_domain") == pytest.approx(0.28)


def test_scale_out_of_range_rejected(monkeypatch):
    """系数 ∈(0,1] 硬约束：>1 会击穿封顶语义，必须拒绝。"""
    from serenity.config import settings
    monkeypatch.setattr(settings, "prior_scale", 1.5)
    with pytest.raises(ValueError):
        _map_delta("yes", "strong", "in_domain")


# ── LLM 层：引用必须 + fail-closed（评审 4A CRITICAL）──


def test_generate_happy_path():
    llm = _ScriptedLLM(_payload())
    r = _generate(llm, title="Q", deadline=None, beliefs=_beliefs(), gate_state="in_domain")
    assert r.delta == pytest.approx(0.35)
    assert r.direction == "yes" and r.strength == "strong" and r.belief_ids == [1]
    assert r.parse_error is None


def test_zero_retrieval_hits_is_legal_zero_delta():
    llm = _ScriptedLLM(RuntimeError("must not be called"))
    r = _generate(llm, title="Q", deadline=None, beliefs=[], gate_state="in_domain")
    assert r.delta == 0.0 and r.parse_error is None
    assert llm.n_calls == 0  # 空检索不花钱


def test_missing_belief_ids_fail_closed():
    """direction!=none 但没引用 → 重试 → 仍缺 → δ=0 + parse_error（设计红线）。"""
    llm = _ScriptedLLM(_payload(ids=()), _payload(ids=()))
    r = _generate(llm, title="Q", deadline=None, beliefs=_beliefs(), gate_state="in_domain")
    assert llm.n_calls == 2
    assert r.delta == 0.0 and r.parse_error == "missing_belief_ids"


def test_belief_ids_out_of_range_fail_closed():
    llm = _ScriptedLLM(_payload(ids=(999,)))
    r = _generate(llm, title="Q", deadline=None, beliefs=_beliefs(), gate_state="in_domain")
    assert r.delta == 0.0 and r.parse_error == "belief_ids_out_of_range"


def test_bad_enum_retries_then_fail_closed():
    llm = _ScriptedLLM({"direction": "maybe", "strength": "huge", "rationale": "x" * 40, "belief_ids": [1]})
    r = _generate(llm, title="Q", deadline=None, beliefs=_beliefs(), gate_state="in_domain")
    assert llm.n_calls == 2 and r.delta == 0.0 and r.parse_error == "bad_direction"


def test_bad_enum_recovers_on_retry():
    llm = _ScriptedLLM({"direction": "maybe"}, _payload(strength="moderate"))
    r = _generate(llm, title="Q", deadline=None, beliefs=_beliefs(), gate_state="in_domain")
    assert r.delta == pytest.approx(0.20) and r.parse_error is None


def test_llm_exception_fail_closed():
    llm = _ScriptedLLM(RuntimeError("gateway down"))
    r = _generate(llm, title="Q", deadline=None, beliefs=_beliefs(), gate_state="in_domain")
    assert r.delta == 0.0 and r.parse_error.startswith("llm_error")


def test_direction_none_ignores_ids():
    llm = _ScriptedLLM({"direction": "none", "strength": "none",
                        "rationale": "beliefs do not bear on this question at all", "belief_ids": []})
    r = _generate(llm, title="Q", deadline=None, beliefs=_beliefs(), gate_state="in_domain")
    assert r.delta == 0.0 and r.direction == "none" and r.parse_error is None


# ── 检索（v1 字段过滤）──


def _seed_beliefs(tmpdb):
    from serenity.store.dao import session_scope
    from serenity.store.models import BeliefPrimitive

    with session_scope() as s:
        s.add(BeliefPrimitive(claim="NVDA networking attach rates rising", domain="semis_supply_chain",
                              tickers="NVDA", stance="bullish", confidence="high",
                              belief_set_version="v1"))
        s.add(BeliefPrimitive(claim="HBM supply tight through 2026", domain="memory_hbm_nand",
                              tickers="MU,SK", stance="bullish", confidence="medium",
                              belief_set_version="v1"))
        s.add(BeliefPrimitive(claim="Neocloud financing quality deteriorating", domain="neocloud_financing",
                              tickers="", stance="bearish", confidence="high",
                              belief_set_version="v1"))


def test_retrieve_by_ticker(tmpdb):
    _seed_beliefs(tmpdb)
    got = retrieve_beliefs(title="Will NVDA ship GB300 in volume by Q4?",
                           gate=_gate_in(), version="v1")
    assert [b.tickers for b in got] == ["NVDA"]


def test_retrieve_by_domain_keyword(tmpdb):
    _seed_beliefs(tmpdb)
    got = retrieve_beliefs(title="Will HBM prices rise 50% by December?",
                           gate=_gate_in(), version="v1")
    assert any("HBM" in b.claim for b in got)


def test_retrieve_adjacent_uses_gate_rationale_domain(tmpdb):
    _seed_beliefs(tmpdb)
    got = retrieve_beliefs(title="Will the Fed cut rates in September?",
                           gate=_gate_adj(), version="v1")
    assert any(b.domain == "neocloud_financing" for b in got)


def test_retrieve_no_hits_empty(tmpdb):
    _seed_beliefs(tmpdb)
    got = retrieve_beliefs(title="Will the Lakers win the NBA title?",
                           gate=GateResult(state="adjacent", rationale="no domain named here"),
                           version="v1")
    assert got == []


# ── placebo（评审 6A）──


def test_placebo_excludes_real_ids_and_deterministic(tmpdb):
    _seed_beliefs(tmpdb)
    got1 = sample_placebo_beliefs(question_id="q1", version="v1", k=2, exclude_ids={1})
    got2 = sample_placebo_beliefs(question_id="q1", version="v1", k=2, exclude_ids={1})
    assert [b.id for b in got1] == [b.id for b in got2]  # 同题可复现
    assert all(b.id != 1 for b in got1)
    got3 = sample_placebo_beliefs(question_id="q2", version="v1", k=2, exclude_ids={1})
    # 不强制不同（小样本可能撞），但种子按题变化
    assert isinstance(got3, list)


def test_generate_placebo_same_flow(tmpdb):
    _seed_beliefs(tmpdb)
    real = PriorResult(delta=0.35, retrieved_ids=[1])
    llm = _ScriptedLLM(_payload(direction="no", strength="weak", ids=(2,)))
    r = generate_placebo_prior(llm, question_id="q1", title="Q", deadline=None,
                               gate=_gate_in(), version="v1", real_result=real)
    assert r.delta == pytest.approx(-0.10)


def test_generate_prior_end_to_end(tmpdb):
    _seed_beliefs(tmpdb)
    llm = _ScriptedLLM(_payload(ids=(1,)))
    r = generate_prior(llm, title="Will NVDA ship GB300 in volume by Q4?",
                       deadline=None, gate=_gate_in(), version="v1")
    assert r.delta == pytest.approx(0.35) and r.belief_ids == [1]


# ── apply_delta ──


def test_apply_delta_math():
    import math
    p = 0.5
    out = apply_delta(p, 0.35)
    assert out == pytest.approx(1 / (1 + math.exp(-0.35)))
    assert apply_delta(p, 0.0) == pytest.approx(0.5)
    assert apply_delta(p, -0.35) == pytest.approx(1 - out)


def test_apply_delta_clips_extremes():
    assert 0.0 < apply_delta(1e-12, -5.0) < 1.0
    assert 0.0 < apply_delta(1.0, 5.0) < 1.0
