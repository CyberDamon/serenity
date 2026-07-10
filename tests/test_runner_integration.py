"""daily 集成测试 [→E2E]：mock LLM + mock Yarrow，走完 ①闸门→②generic→③先验/placebo
→④三臂落库→⑤提交 全管线，断言每步落库产物（设计文档 Test Requirements）。"""

from __future__ import annotations

import json
from datetime import date

import pytest

from serenity.yarrow.client import YarrowQuestionDTO


# ── mock 件 ──────────────────────────────────────────────────────────────────


def _q(qid, title, category="tech", resolve="2026-12-31T00:00:00Z"):
    return YarrowQuestionDTO(id=qid, type="binary", title=title, status="open",
                             category=category, scheduled_resolve_time=resolve)


class FakeYarrow:
    def __init__(self, questions):
        self.questions = questions
        self.submitted_payloads: list[list[dict]] = []

    def iter_questions(self, **kw):
        yield from self.questions

    def batch_cross_market(self, ids):
        return {}

    def submit_forecasts(self, payload):
        self.submitted_payloads.append(payload)
        return {"ok": True}


class FrameworkFakeLLM:
    """generic 臂假模型：对 PROB_SCHEMA 请求返回固定 prob。"""

    def __init__(self, model, prob=0.4):
        self.model = model
        self._prob = prob

    @property
    def training_cutoff(self):
        return date(2020, 1, 1)

    def complete(self, *, system, user, max_tokens=1024, response_schema=None, **kw):
        from serenity.agent.llm_client import LLMResponse
        req = (response_schema or {}).get("required", [])
        if "prob" in req:  # framework PROB_SCHEMA
            parsed = {
                "yes_side_interpretation": "YES = the stated event occurs before deadline.",
                "prob": self._prob, "reasoning": "x" * 60, "key_evidence": ["e1", "e2"],
                "sources_cited": [], "contamination_warning": False,
                "contamination_confidence": 0.0,
            }
        elif "revised_prob" in req:  # self_check
            parsed = {"revised_prob": self._prob, "note": "ok"}
        else:
            parsed = {k: "x" * 30 for k in req}
        return LLMResponse(text="", parsed_json=parsed, model=self.model, cost_usd=0.001)


class GateFakeLLM:
    """闸门假模型：按题名关键词裁 adjacent/out。"""

    model = "gate-fake"

    def complete(self, *, system, user, max_tokens=1024, response_schema=None, **kw):
        from serenity.agent.llm_client import LLMResponse
        if "fed" in user.lower():
            parsed = {"state": "adjacent",
                      "rationale": "rate costs → neocloud_financing via one causal hop"}
        else:
            parsed = {"state": "out_of_domain", "rationale": "unrelated to covered domains"}
        return LLMResponse(text="", parsed_json=parsed, model=self.model, cost_usd=0.0001)


class PriorFakeLLM:
    model = "prior-fake"

    def __init__(self, direction="yes", strength="moderate"):
        self.direction = direction
        self.strength = strength

    def complete(self, *, system, user, max_tokens=1024, response_schema=None, **kw):
        from serenity.agent.llm_client import LLMResponse
        # 从提示里抓第一个 [id]，保证引用合法
        import re
        m = re.search(r"\[(\d+)\]", user)
        bid = int(m.group(1)) if m else 1
        parsed = {"direction": self.direction, "strength": self.strength,
                  "rationale": "cited beliefs address the supply-chain mechanism directly here",
                  "belief_ids": [bid]}
        return LLMResponse(text="", parsed_json=parsed, model=self.model, cost_usd=0.0005)


def _seed_belief_version(version="vtest"):
    from datetime import datetime

    from serenity.store.dao import session_scope
    from serenity.store.models import BeliefPrimitive, BeliefSetMeta

    with session_scope() as s:
        s.add(BeliefSetMeta(version=version, created_at=datetime(2026, 7, 1),
                            n_beliefs=2, active=True))
        s.add(BeliefPrimitive(claim="NVDA rack-scale systems drive optics attach rates up",
                              domain="semis_supply_chain", tickers="NVDA", stance="bullish",
                              confidence="high", belief_set_version=version))
        s.add(BeliefPrimitive(claim="Neocloud financing quality is deteriorating structurally",
                              domain="neocloud_financing", tickers="", stance="bearish",
                              confidence="high", belief_set_version=version))


def _run(tmpdb, questions, *, submit=False, monkeypatch=None, **kw):
    from serenity.yarrow.runner import run_daily

    _seed_belief_version()
    fake = FakeYarrow(questions)
    llms = [FrameworkFakeLLM("m1", 0.4), FrameworkFakeLLM("m2", 0.4)]
    res = run_daily(
        client=fake, llms=llms, gate_llm=GateFakeLLM(), prior_llm=PriorFakeLLM(),
        submit=submit, max_questions=5, research=False,
        min_evidence=0, min_sources=0,  # 集成测不接检索，关证据门
        self_check=None, calibrator=lambda p: p,
        **kw,
    )
    return res, fake


# ── 全管线 ──


def test_daily_three_arms_persisted(tmpdb):
    """in_domain 题走完全管线：三臂概率 + 闸门判据 + belief_ids + 版本全部落库。"""
    res, _ = _run(tmpdb, [_q("q1", "Will NVDA ship GB300 racks in volume by Q4 2026?")])
    assert res.gated_in == 1

    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction
    with session_scope() as s:
        row = s.query(Prediction).filter_by(question_id="q1").one()
        assert row.gate_state == "in_domain"
        assert row.gate_rationale and "NVDA" in row.gate_rationale
        assert row.generic_prob == pytest.approx(0.4, abs=0.05)
        # moderate/in_domain → δ=+0.20
        assert row.delta_log_odds == pytest.approx(0.20)
        assert row.final_prob > row.generic_prob  # yes 方向抬升
        assert row.placebo_prob is not None
        assert json.loads(row.belief_ids)  # 引用非空
        assert row.belief_set_version == "vtest"
        assert row.submit_status == "dry_run"


def test_daily_adjacent_halved_delta(tmpdb):
    res, _ = _run(tmpdb, [_q("q2", "Will the Fed cut rates in September 2026?")])
    assert res.gated_adjacent == 1
    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction
    with session_scope() as s:
        row = s.query(Prediction).filter_by(question_id="q2").one()
        assert row.gate_state == "adjacent"
        assert abs(row.delta_log_odds) == pytest.approx(0.10)  # moderate 减半


def test_daily_out_of_domain_abstains_with_shadow(tmpdb, monkeypatch):
    """out 题弃权：不提交；抽样 ≤N 条跑 generic shadow，其余只记闸门判定。"""
    from serenity.config import settings
    monkeypatch.setattr(settings, "out_shadow_sample", 1)
    qs = [_q(f"o{i}", f"Will the Lakers win game {i}?") for i in range(3)]
    res, fake = _run(tmpdb, qs)
    assert res.gated_out == 3 and res.submitted == 0

    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction
    with session_scope() as s:
        rows = s.query(Prediction).all()
        assert len(rows) == 3
        shadows = [r for r in rows if r.skip_reason == "out_of_domain_shadow"]
        minimal = [r for r in rows if r.skip_reason == "out_of_domain"]
        assert len(shadows) == 1 and len(minimal) == 2
        assert shadows[0].generic_prob is not None  # shadow 有 generic 臂
        assert shadows[0].final_prob is None        # 但没有先验臂
        assert all(r.gate_rationale for r in rows)  # 判据全落库


def test_daily_submit_payload_has_reasoning(tmpdb):
    """提交载荷带 report.reasoning（正式后端 422 护栏），且引信念转述。"""
    res, fake = _run(tmpdb, [_q("q1", "Will NVDA ship GB300 racks in volume by Q4 2026?")],
                     submit=True)
    assert res.submitted == 1
    payload = fake.submitted_payloads[0][0]
    assert payload["question_id"] == "q1"
    assert 0 < payload["probability_yes"] < 1
    reasoning = payload["report"]["reasoning"]
    assert "generic baseline" in reasoning and "delta" in reasoning
    assert "optics attach" in reasoning  # 信念转述进了报告

    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction
    with session_scope() as s:
        row = s.query(Prediction).filter_by(question_id="q1").one()
        assert row.submit_status == "submitted"
        assert row.first_submit_ts is not None


def test_daily_resubmit_window_dedupe(tmpdb):
    """同题 3 天内不重提（覆盖语义护栏）。"""
    from datetime import UTC, datetime, timedelta

    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction

    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    with session_scope() as s:
        s.add(Prediction(question_id="q1", prediction_date=yesterday,
                         submit_status="submitted"))
    res, fake = _run(tmpdb, [_q("q1", "Will NVDA ship GB300 racks in volume by Q4 2026?")],
                     submit=True)
    assert res.submitted == 0 and fake.submitted_payloads == []


def test_daily_empty_question_list_ok(tmpdb):
    """拉题为空 → 正常退出非崩溃。"""
    res, _ = _run(tmpdb, [])
    assert res.seen == 0 and res.submitted == 0 and res.items == []


def test_daily_non_binary_format_skipped(tmpdb):
    res, _ = _run(tmpdb, [_q("qn", "How many Fed rate cuts will there be in 2026?")])
    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction
    with session_scope() as s:
        row = s.query(Prediction).filter_by(question_id="qn").one()
        assert row.skip_reason == "non_binary_format"
        assert row.final_prob is None


def test_daily_requires_active_belief_version(tmpdb):
    """无激活信念库 → 明确报错（先验臂依赖冻结版本，评审 3A）。"""
    from serenity.yarrow.runner import run_daily

    with pytest.raises(RuntimeError, match="distill"):
        run_daily(client=FakeYarrow([]), llms=[FrameworkFakeLLM("m1")],
                  gate_llm=GateFakeLLM(), prior_llm=PriorFakeLLM(),
                  self_check=None, calibrator=lambda p: p, research=False)


def test_daily_llm_gateway_down_isolates_question(tmpdb):
    """generic 臂整体失败（双模型全挂）→ 该题标 failed，不崩整轮。"""

    class DownLLM(FrameworkFakeLLM):
        def complete(self, **kw):
            raise RuntimeError("gateway down")

    from serenity.yarrow.runner import run_daily

    _seed_belief_version()
    fake = FakeYarrow([_q("q1", "Will NVDA ship GB300 racks in volume by Q4 2026?")])
    res = run_daily(client=fake, llms=[DownLLM("m1"), DownLLM("m2")],
                    gate_llm=GateFakeLLM(), prior_llm=PriorFakeLLM(),
                    submit=True, research=False, min_evidence=0, min_sources=0,
                    self_check=None, calibrator=lambda p: p)
    # 双模型全挂 → aggregate 无有效输出 → 提交门 aggregator_model_failed，shadow 落库
    assert res.submitted == 0
    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction
    with session_scope() as s:
        row = s.query(Prediction).filter_by(question_id="q1").one()
        assert row.submit_status in ("skipped", "failed")
