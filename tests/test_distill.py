"""distill 单元测：语料载入 / 抽取容错 / 时间戳截断 / 版本 hash / 重建守卫。"""

from __future__ import annotations

import json

import pytest

from serenity.distill.pipeline import (
    compute_version,
    load_corpus,
    rebuild_guard,
    run_distill,
)


def _write_corpus(tmp_path, tweets):
    p = tmp_path / "tweets.json"
    p.write_text(json.dumps(tweets))
    return str(p)


def _tweet(tid, date, text, **kw):
    return {"id": tid, "createdAtISO": f"{date}T12:00:00Z", "text": text, **kw}


# ── 语料载入 ──


def test_load_corpus_normal_and_sorted(tmp_path):
    path = _write_corpus(tmp_path, [
        _tweet("2", "2026-02-01", "HBM supply is the bottleneck for AI accelerators"),
        _tweet("1", "2026-01-01", "InP substrates sold out through H1"),
    ])
    tweets = load_corpus(path)
    assert [t["id"] for t in tweets] == ["1", "2"]  # 按时间排序
    assert tweets[0]["date"] == "2026-01-01"


def test_load_corpus_skips_bad_rows_and_dups(tmp_path):
    path = _write_corpus(tmp_path, [
        _tweet("1", "2026-01-01", "real analytical post about optics"),
        _tweet("1", "2026-01-01", "duplicate id"),                    # 重复 id
        _tweet("2", "2026-01-02", ""),                                 # 空文本
        {"createdAtISO": "2026-01-03T00:00:00Z", "text": "no id"},    # 缺 id
        _tweet("3", "2026-01-04", "pure retweet", isRetweet=True),    # 纯转推
        _tweet("4", "2026-01-05", "quote with comment", isRetweet=True, isQuote=True),
    ])
    tweets = load_corpus(path)
    assert [t["id"] for t in tweets] == ["1", "4"]


def test_load_corpus_empty_file(tmp_path):
    assert load_corpus(_write_corpus(tmp_path, [])) == []


def test_load_corpus_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(str(tmp_path / "nope.json"))


# ── 抽取容错（坏输出跳过计数，不落脏数据）──


def _extract_payload(claim_tweet_ids=("1",), made_at="2026-01-01"):
    return {
        "belief_primitives": [{
            "claim": "Hyperscaler capex flows into optical interconnect bottlenecks",
            "domain": "optics_cpo", "tickers": ["ANET"], "stance": "bullish",
            "confidence": "high", "causal_links": ["capex→optics demand"],
            "tweet_ids": list(claim_tweet_ids),
        }],
        "ticker_theses": [{
            "ticker": "anet", "subsector": "networking",
            "thesis": "Share gains in AI back-end networks", "confidence": "medium",
            "tweet_ids": list(claim_tweet_ids),
        }],
        "historical_claims": [{
            "claim": "InP substrate demand will exceed supply through H1 2026",
            "direction": "yes", "made_at": made_at, "horizon": "H1 2026",
            "tweet_ids": list(claim_tweet_ids),
        }],
    }


def test_run_distill_end_to_end(tmpdb, tmp_path, scripted_llm):
    path = _write_corpus(tmp_path, [
        _tweet("1", "2026-01-01", "InP substrates sold out; optics is the bottleneck"),
    ])
    llm = scripted_llm([_extract_payload()])
    rep = run_distill(corpus_path=path, llm=llm, batch_size=40, concurrency=1)
    assert rep.skipped_reason is None
    assert rep.version and rep.n_beliefs == 1 and rep.n_tickers == 1 and rep.n_claims == 1

    from serenity.distill.pipeline import active_version
    assert active_version() == rep.version

    from serenity.store.dao import session_scope
    from serenity.store.models import BeliefPrimitive, TickerKnowledge
    with session_scope() as s:
        b = s.query(BeliefPrimitive).one()
        assert b.domain == "optics_cpo" and b.belief_set_version == rep.version
        t = s.query(TickerKnowledge).one()
        assert t.ticker == "ANET"  # 大写归一


def test_run_distill_bad_batch_skipped_no_dirty_data(tmpdb, tmp_path, scripted_llm):
    """坏输出（parsed_json=None）→ 批次计失败，不落脏数据。"""
    path = _write_corpus(tmp_path, [
        _tweet("1", "2026-01-01", "post one about HBM"),
        _tweet("2", "2026-01-02", "post two about NAND"),
    ])
    llm = scripted_llm([None])  # 所有批次坏输出
    rep = run_distill(corpus_path=path, llm=llm, batch_size=1, concurrency=1)
    assert rep.n_batches == 2 and rep.n_batches_failed == 2
    assert rep.skipped_reason  # 全失败 → 产出为空
    from serenity.distill.pipeline import active_version
    assert active_version() is None  # 没写任何版本


def test_run_distill_hallucinated_tweet_ids_dropped(tmpdb, tmp_path, scripted_llm):
    """引用不存在的 tweet_id = 幻觉 → 该条目丢弃。"""
    path = _write_corpus(tmp_path, [_tweet("1", "2026-01-01", "real post")])
    llm = scripted_llm([_extract_payload(claim_tweet_ids=("999",))])
    rep = run_distill(corpus_path=path, llm=llm, batch_size=40, concurrency=1)
    assert rep.skipped_reason  # 全部被丢 → 产出为空


# ── 时间戳截断（评审 8A）──


def test_historical_claim_made_at_forced_to_earliest_source(tmpdb, tmp_path, scripted_llm):
    """LLM 给的 made_at 与来源不符时，以最早来源推文日期为准。"""
    path = _write_corpus(tmp_path, [_tweet("1", "2026-01-01", "forward looking call")])
    llm = scripted_llm([_extract_payload(made_at="2026-06-30")])  # LLM 谎报晚日期
    rep = run_distill(corpus_path=path, llm=llm, batch_size=40, concurrency=1)
    from serenity.store.dao import session_scope
    from serenity.store.models import HistoricalClaim
    with session_scope() as s:
        c = s.query(HistoricalClaim).one()
        assert c.made_at == "2026-01-01"  # 截断到最早来源日期
    assert rep.n_claims == 1


# ── 版本 hash ──


def test_compute_version_content_addressed():
    a = [{"claim": "X", "domain": "d", "stance": "bullish"}]
    b = [{"claim": "X", "domain": "d", "stance": "bullish"}]
    c = [{"claim": "Y", "domain": "d", "stance": "bullish"}]
    assert compute_version(a) == compute_version(b)
    assert compute_version(a) != compute_version(c)
    # 顺序无关
    two = [{"claim": "X", "domain": "d", "stance": "bullish"},
           {"claim": "Y", "domain": "d", "stance": "bearish"}]
    assert compute_version(two) == compute_version(list(reversed(two)))


# ── 重建守卫（评审 3A CRITICAL）──


def _seed_version(version="v1abc", with_forecast=False, resolved=False):
    from datetime import datetime

    from serenity.store.dao import session_scope
    from serenity.store.models import BeliefSetMeta, Prediction, Resolution

    with session_scope() as s:
        s.add(BeliefSetMeta(version=version, created_at=datetime(2026, 7, 1),
                            n_beliefs=1, active=True))
        if with_forecast:
            s.add(Prediction(
                question_id="q1", prediction_date="2026-07-02", gate_state="in_domain",
                generic_prob=0.4, final_prob=0.5, placebo_prob=0.4,
                belief_set_version=version, submit_status="submitted",
            ))
            if resolved:
                s.add(Resolution(question_id="q1", resolution_kind="yes", outcome=1.0))


def test_rebuild_guard_first_distill_allowed(tmpdb):
    assert rebuild_guard(force=False) is None


def test_rebuild_guard_no_forecasts_allowed(tmpdb):
    _seed_version(with_forecast=False)
    assert rebuild_guard(force=False) is None


def test_rebuild_guard_blocks_mid_experiment(tmpdb):
    """实验期（有 forecast 引用、配对样本未满）拒绝重建。"""
    _seed_version(with_forecast=True)
    reason = rebuild_guard(force=False)
    assert reason and "实验期冻结" in reason


def test_rebuild_guard_force_opens_new_segment(tmpdb):
    _seed_version(with_forecast=True)
    assert rebuild_guard(force=True) is None


def test_rebuild_guard_allows_after_experiment_full(tmpdb, monkeypatch):
    from serenity.config import settings
    _seed_version(with_forecast=True, resolved=True)
    monkeypatch.setattr(settings, "experiment_min_paired", 1)  # 1 条即满
    assert rebuild_guard(force=False) is None


def test_paired_sample_count_requires_three_arms_and_nonvoid(tmpdb):
    """守卫口径 = 主指标口径（Codex 验收 F3）：缺 placebo 或 void 不计数。"""
    from datetime import datetime

    from serenity.distill.pipeline import paired_sample_count
    from serenity.store.dao import session_scope
    from serenity.store.models import BeliefSetMeta, Prediction, Resolution

    with session_scope() as s:
        s.add(BeliefSetMeta(version="v1", created_at=datetime(2026, 7, 1), active=True))
        # 三臂齐 + 非 void → 计数
        s.add(Prediction(question_id="ok", prediction_date="2026-07-02", gate_state="in_domain",
                         generic_prob=0.4, final_prob=0.5, placebo_prob=0.4,
                         belief_set_version="v1"))
        s.add(Resolution(question_id="ok", resolution_kind="yes", outcome=1.0))
        # 缺 placebo → 不计
        s.add(Prediction(question_id="nop", prediction_date="2026-07-02", gate_state="in_domain",
                         generic_prob=0.4, final_prob=0.5, belief_set_version="v1"))
        s.add(Resolution(question_id="nop", resolution_kind="yes", outcome=1.0))
        # void → 不计
        s.add(Prediction(question_id="vd", prediction_date="2026-07-02", gate_state="in_domain",
                         generic_prob=0.4, final_prob=0.5, placebo_prob=0.4,
                         belief_set_version="v1"))
        s.add(Resolution(question_id="vd", resolution_kind="void", outcome=0.5, is_void=True))
    assert paired_sample_count("v1") == 1
