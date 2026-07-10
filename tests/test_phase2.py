"""Phase 2 单元测：research 查询生成 / self_check 可度量 / logit 分歧门（离线）。"""

from __future__ import annotations

from datetime import date, datetime, timezone


from serenity.agent.core import predict
from serenity.agent.frameworks import Market
from serenity.agent.frameworks.base import FrameworkOutput
from serenity.agent.llm_client import LLMResponse
from serenity.agent.self_check import make_self_check
from serenity.data.news.types import NewsItem
from serenity.agent.aggregator import logit_dispersion
from serenity.yarrow.runner import _submission_gate as _gate

Q = "Will Pedro Sánchez be the next leader out before 2027?"


class SchemaFakeLLM:
    """按 response_schema 的 required 字段返回预置值的假 client。"""

    def __init__(self, model="m1", values=None, fail=False):
        self.model = model
        self._values = values or {}
        self._fail = fail

    @property
    def training_cutoff(self):
        return date(2020, 1, 1)

    def complete(self, *, system, user, max_tokens=1024, response_schema=None, **kw):
        if self._fail:
            raise RuntimeError("outage")
        req = (response_schema or {}).get("required", [])
        out = {}
        for k in req:
            if k in self._values:
                out[k] = self._values[k]
            elif k == "prob":
                out[k] = 0.3
            elif "prob" in k:
                out[k] = 0.5
            elif k in ("historical", "current", "key_evidence", "sources_cited"):
                out[k] = ["a", "b"] if k in ("historical", "current") else []
            elif "confidence" in k:
                out[k] = 0.0
            elif k == "contamination_warning":
                out[k] = False
            else:
                out[k] = "x" * 30
        return LLMResponse(text="", parsed_json=out, model=self.model, cost_usd=0.001)


def _fw(name, prob):
    return FrameworkOutput(framework_name=name, status="ok", prob=prob)


# ── logit 分歧 ──


def test_logit_dispersion_low_when_agree():
    assert logit_dispersion([0.30, 0.33, 0.31]) < 0.3


def test_logit_dispersion_high_when_split():
    assert logit_dispersion([0.05, 0.95]) > 2.0


def test_logit_dispersion_single_is_zero():
    assert logit_dispersion([0.4]) == 0.0


# ── self_check 可度量 delta ──


def test_self_check_revises_and_is_measurable():
    check = make_self_check(SchemaFakeLLM(values={"revised_prob": 0.55, "note": "n"}))
    pred = predict(
        market=Market(token_id="t123456789", question=Q, market_price=0.25,
                      resolution_date_iso="2026-12-31"),
        news=[], llms=[SchemaFakeLLM(values={"prob": 0.3})], as_of_date=date(2026, 7, 1),
        self_check=check,
    )
    # self_check 把 raw 改到 0.55，delta 被记录且可度量
    assert abs(pred.raw_prob - 0.55) < 1e-9
    assert pred.self_check_delta is not None
    assert abs(pred.self_check_delta - (0.55 - 0.3)) < 1e-6


def test_self_check_degrades_on_failure():
    check = make_self_check(SchemaFakeLLM(fail=True))
    pred = predict(
        market=Market(token_id="t123456789", question=Q, market_price=0.25,
                      resolution_date_iso="2026-12-31"),
        news=[], llms=[SchemaFakeLLM(values={"prob": 0.3})], as_of_date=date(2026, 7, 1),
        self_check=check,
    )
    # 失败 → 保持 raw，delta≈0
    assert abs(pred.self_check_delta) < 1e-9


# ── L2 agentic 多轮检索 ──


def test_research_fallback_to_rss_without_searx(monkeypatch):
    """无 SEARX token → 退回 RSS 瀑布（assemble_news_context）并打标。"""
    import serenity.data.research.agentic as ag
    from serenity.config import settings

    monkeypatch.setattr(settings.searx_token, "get_secret_value", lambda: "")
    called = {}

    def fake_assemble(query, *, top_k=8):
        called["q"] = query
        return [NewsItem(source="rss:x", url="u", title="troops deploy to border",
                         published_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                         summary="military deployment reported")]

    monkeypatch.setattr(ag, "assemble_news_context", fake_assemble)
    items, rlog = ag.assemble_research(Q, llm=SchemaFakeLLM(), as_of_date="2026-07-01")
    assert called["q"] == Q
    assert items[0].tags  # pre_filter 打了标
    assert rlog["backend"] == "rss"


def test_research_agentic_multiround(monkeypatch):
    """有 SEARX token → 多轮 agentic：LLM 出词→SearX 搜→续轮，need_more=false 停。"""
    import serenity.data.research.agentic as ag
    from serenity.config import settings

    monkeypatch.setattr(settings.searx_token, "get_secret_value", lambda: "sx-test")

    # 前 2 轮要更多、给词；第 3 轮 need_more=false 停
    scripted = [
        {"analysis": "round1 understanding", "search_queries": ["q1a", "q1b"], "need_more": True},
        {"analysis": "round2 synthesis", "search_queries": ["q2a"], "need_more": True},
        {"analysis": "final synthesis with evidence", "search_queries": [], "need_more": False},
    ]
    calls = {"n": 0}

    class _LLM(SchemaFakeLLM):
        def complete(self, **kw):
            from serenity.agent.llm_client import LLMResponse
            out = scripted[min(calls["n"], len(scripted) - 1)]
            calls["n"] += 1
            return LLMResponse(text="", parsed_json=out, model="m", cost_usd=0.001)

    searched = []

    def fake_searx(query, *, max_results=4, days_back=21):
        searched.append(query)
        return [NewsItem(source=f"x-{query}", url=f"http://x/{query}", title=f"art-{query}",
                         published_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                         summary="troops deploy escalation")]

    monkeypatch.setattr(ag, "_searx_search", fake_searx)

    analysis, sources, queries = ag.agentic_research(Q, llm=_LLM(), as_of_date="2026-07-01", max_rounds=5)
    assert analysis == "final synthesis with evidence"       # 用了末轮分析
    assert searched == ["q1a", "q1b", "q2a"]                  # 两轮共 3 次搜索后停
    assert queries == ["q1a", "q1b", "q2a"]                   # 返回用过的检索词(审计)
    assert len(sources) == 3 and sources[0].tags             # 3 篇去重源 + 打标
    assert calls["n"] == 3                                    # 3 轮 LLM（末轮 need_more=false 停）


def test_searx_item_source_is_domain_not_literal_searx(monkeypatch):
    """QA-001 回归：SearX 结果 source 用域名（否则证据门来源多样性把全部当 1 源→误杀）。"""
    from serenity.config import settings
    from serenity.data.news.live import searx

    monkeypatch.setattr(settings.searx_token, "get_secret_value", lambda: "sx-test")
    monkeypatch.setattr(searx, "_request", lambda *a, **k: {"results": [
        {"title": "t", "url": "https://www.bbc.com/news/x", "content": "c" * 80},
        {"title": "t", "url": "https://ft.com/y", "content": "c" * 80},
    ]})
    a, b = searx.search("q")
    assert a.source == "bbc.com"      # www. 去掉
    assert b.source == "ft.com"
    assert a.source != b.source       # 不同出版方 → 不同来源，多样性门可用


def test_research_assemble_agentic_prepends_brief(monkeypatch):
    """assemble_research 在 agentic 路径下：首条是 synthesis 简报 + 源文章。"""
    import serenity.data.research.agentic as ag
    from serenity.config import settings

    monkeypatch.setattr(settings.searx_token, "get_secret_value", lambda: "sx-test")
    monkeypatch.setattr(ag, "agentic_research",
                        lambda *a, **k: ("BRIEF TEXT", [NewsItem(
                            source="bbc.com", url="u", title="t",
                            published_at=datetime(2026, 6, 1, tzinfo=timezone.utc), summary="s")],
                            ["q1", "q2"]))
    items, rlog = ag.assemble_research(Q, llm=SchemaFakeLLM(), as_of_date="2026-07-01")
    assert items[0].source == "agentic:brief" and "BRIEF TEXT" in items[0].summary
    assert items[1].source == "bbc.com"
    # 检索审计 dict：backend/queries/sources/brief
    assert rlog["backend"] == "searx" and rlog["queries"] == ["q1", "q2"]
    assert rlog["n_sources"] == 1 and "BRIEF TEXT" in rlog["brief"]


# ── 门：证据过薄 / logit 分歧 ──


class _Agg:
    def __init__(self, ir_logit_std=0.0):
        self.contamination_filter_triggered = False
        self.n_ir_valid = 3
        self.partial_aggregation = False
        self.ir_logit_std = ir_logit_std


class _Pred:
    def __init__(self, probs):
        # gate 现在读 aggregated.ir_logit_std（单一度量），由 aggregator 预计算
        self.aggregated = _Agg(ir_logit_std=logit_dispersion(probs))
        self.ir_outputs = [_fw(f"f{i}", p) for i, p in enumerate(probs)]


class _Q:
    scheduled_resolve_time = "2027-01-01T00:00:00Z"


def _gate_kw(**over):
    kw = dict(now=datetime(2026, 7, 1, tzinfo=timezone.utc), min_lead_days=3,
              evidence=(5, 3), min_evidence=2, min_sources=2, logit_dispersion_max=1.1)
    kw.update(over)
    return kw


def test_non_binary_format_guard():
    """v1 仅二元题（评审定稿）：格式怪题兜底正则。领域过滤由三态闸门负责（见 test_gate）。"""
    from serenity.yarrow.runner import _NON_BINARY_RE
    assert _NON_BINARY_RE.search("How many Fed rate cuts in 2026?")
    assert _NON_BINARY_RE.search("Which of the following companies will be first to 5T market cap?")
    assert not _NON_BINARY_RE.search("Will NVIDIA announce a Blackwell successor at GTC 2026?")


def test_gate_evidence_too_thin_by_count():
    assert _gate(_Pred([0.3, 0.31, 0.29]), _Q(), **_gate_kw(evidence=(0, 0))) == "evidence_too_thin"


def test_gate_evidence_too_thin_by_sources():
    # 篇数够(4)但来源只有 1 家 → 单源回声，仍判 too_thin
    assert _gate(_Pred([0.3, 0.31, 0.29]), _Q(), **_gate_kw(evidence=(4, 1))) == "evidence_too_thin"


def test_gate_disagreement_logit():
    assert _gate(_Pred([0.05, 0.95]), _Q(), **_gate_kw()) == "disagreement_logit"


def test_gate_passes_when_healthy():
    assert _gate(_Pred([0.30, 0.33, 0.31]), _Q(), **_gate_kw()) is None


def test_evidence_quality_counts_recent_and_sources():
    from serenity.yarrow.runner import _evidence_quality
    now = datetime(2026, 7, 2, tzinfo=timezone.utc)
    news = [
        NewsItem(source="rss:bbc", url="u1", title="t", published_at=datetime(2026, 7, 1, tzinfo=timezone.utc), summary="x" * 60),
        NewsItem(source="rss:ft", url="u2", title="t", published_at=datetime(2026, 6, 30, tzinfo=timezone.utc), summary="y" * 60),
        NewsItem(source="rss:bbc", url="u3", title="t", published_at=datetime(2026, 1, 1, tzinfo=timezone.utc), summary="z" * 60),  # 太旧
        NewsItem(source="rss:cnn", url="u4", title="t", published_at=datetime(2026, 7, 1, tzinfo=timezone.utc), summary="短"),      # 摘要太短
        NewsItem(source="agentic:brief", url="", title="brief", published_at=now, summary="B" * 100),  # 简报计篇不计源
    ]
    recent, sources = _evidence_quality(news, now, max_age_days=30)
    assert recent == 3           # bbc(7-1)+ft(6-30)+brief；旧的和短摘要被剔
    assert sources == 2          # bbc, ft（brief 不计来源）


def test_self_check_shift_clamped():
    """QA ISSUE-004 回归：无证据复核层的修正幅度夹在 ±0.15（实测曾 0.20→0.55）。"""
    check = make_self_check(SchemaFakeLLM(values={"revised_prob": 0.55, "note": "n"}))
    pred = predict(
        market=Market(token_id="t123456789", question=Q, market_price=0.25,
                      resolution_date_iso="2026-12-31"),
        news=[], llms=[SchemaFakeLLM(values={"prob": 0.2})], as_of_date=date(2026, 7, 1),
        self_check=check,
    )
    assert abs(pred.raw_prob - 0.35) < 1e-9  # 0.2 + 0.15 夹住，而非 0.55
    assert abs(pred.self_check_delta - 0.15) < 1e-9
