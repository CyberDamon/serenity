"""配对统计 CRITICAL 测试：合成数据已知效应 → CI 覆盖真值；零效应 → CI 跨零。

统计代码坏了不报错、只给错误结论——必须用已知答案验证（设计文档 Test Requirements）。
"""

from __future__ import annotations

import random

import pytest

from serenity.scoring.paired import (
    ArmDiffCI,
    brier,
    cluster_key,
    clustered_bootstrap_ci,
    collect_paired_rows,
    paired_report,
)


def test_brier():
    assert brier(0.7, 1.0) == pytest.approx(0.09)
    assert brier(0.7, 0.0) == pytest.approx(0.49)


# ── 合成数据验证（CRITICAL）──


def _synthetic_diffs(true_effect: float, n_clusters: int = 30, per_cluster: int = 3,
                     noise: float = 0.05, seed: int = 7):
    """簇内相关的配对差：diff = true_effect + 簇效应 + 噪声。"""
    rng = random.Random(seed)
    diffs, clusters = [], []
    for c in range(n_clusters):
        cluster_shift = rng.gauss(0, noise)
        for _ in range(per_cluster):
            diffs.append(true_effect + cluster_shift + rng.gauss(0, noise / 2))
            clusters.append(f"c{c}")
    return diffs, clusters


def test_ci_covers_known_negative_effect():
    """植入 -0.04 的真实效应 → CI 覆盖真值且全负（明确有效判定）。"""
    diffs, clusters = _synthetic_diffs(true_effect=-0.04)
    ci = clustered_bootstrap_ci(diffs, clusters, n_boot=2000)
    assert ci.ci_low < -0.04 < ci.ci_high
    assert ci.ci_high < 0  # 效应远大于噪声 → 全负
    assert ci.verdict.startswith("明确有效")


def test_ci_spans_zero_on_null_effect():
    """零效应 → CI 跨零（不允许假阳性）。"""
    diffs, clusters = _synthetic_diffs(true_effect=0.0)
    ci = clustered_bootstrap_ci(diffs, clusters, n_boot=2000)
    assert ci.ci_low < 0 < ci.ci_high
    assert "跨零" in ci.verdict


def test_ci_positive_effect_verdict_harmful():
    diffs, clusters = _synthetic_diffs(true_effect=0.05)
    ci = clustered_bootstrap_ci(diffs, clusters, n_boot=2000)
    assert ci.ci_low > 0 and ci.verdict.startswith("明确有害")


def test_clustered_wider_than_naive_under_cluster_correlation():
    """簇内强相关时，聚类 CI 应比"把每题当独立簇"的天真 CI 宽（假窄检验，评审 7A）。"""
    diffs, clusters = _synthetic_diffs(true_effect=-0.02, n_clusters=10, per_cluster=6,
                                       noise=0.08, seed=11)
    clustered = clustered_bootstrap_ci(diffs, clusters, n_boot=2000)
    naive = clustered_bootstrap_ci(diffs, [str(i) for i in range(len(diffs))], n_boot=2000)
    assert (clustered.ci_high - clustered.ci_low) > (naive.ci_high - naive.ci_low)


def test_ci_deterministic_given_seed():
    diffs, clusters = _synthetic_diffs(true_effect=-0.03)
    a = clustered_bootstrap_ci(diffs, clusters, n_boot=500, seed=42)
    b = clustered_bootstrap_ci(diffs, clusters, n_boot=500, seed=42)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_ci_edge_cases():
    assert clustered_bootstrap_ci([], [], n_boot=10).n == 0
    one = clustered_bootstrap_ci([0.1, 0.2], ["c1", "c1"], n_boot=10)
    assert one.n_clusters == 1 and one.mean_diff == pytest.approx(0.15)
    import math
    assert math.isnan(one.ci_low)  # 单簇没法 bootstrap，诚实返回 NaN
    with pytest.raises(ValueError):
        clustered_bootstrap_ci([0.1], ["a", "b"], n_boot=10)


# ── cluster key ──


def test_cluster_key_ticker_and_month():
    from serenity.gate.gate import GateVocab
    vocab = GateVocab(tickers={"NVDA"}, domains={"memory_hbm_nand"})
    assert cluster_key("Will NVDA ship GB300 by Q4?", "2026-07", vocab) == "NVDA:2026-07"
    assert cluster_key("Will HBM prices rise?", "2026-07", vocab).startswith("memory_hbm_nand:")
    assert cluster_key("Will something odd happen?", "2026-07", vocab) == "something:2026-07"


# ── 端到端（tmpdb）──


def _seed_paired(tmpdb, n=6):
    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction, Resolution

    with session_scope() as s:
        for i in range(n):
            qid = f"q{i}"
            s.add(Prediction(
                question_id=qid, prediction_date="2026-07-05",
                title=f"Will NVDA event {i} happen?", gate_state="in_domain",
                generic_prob=0.5, final_prob=0.6, placebo_prob=0.5,
                belief_set_version="v1", submit_status="submitted",
            ))
            s.add(Resolution(question_id=qid, resolution_kind="yes", outcome=1.0))
        # 干扰行：out_of_domain 与未结算不得进入配对
        s.add(Prediction(question_id="qx", prediction_date="2026-07-05", title="t",
                         gate_state="out_of_domain", generic_prob=0.5, final_prob=0.5,
                         belief_set_version="v1", submit_status="skipped"))


def test_collect_paired_rows_filters(tmpdb):
    _seed_paired(tmpdb)
    rows = collect_paired_rows("v1")
    assert len(rows) == 6
    assert all(r.gate_state == "in_domain" for r in rows)


def test_paired_report_end_to_end(tmpdb):
    _seed_paired(tmpdb)
    rep = paired_report("v1", n_boot=200)
    assert rep.n_paired == 6
    # serenity(0.6→outcome1) brier=0.16 < generic(0.5) brier=0.25 → Δ = -0.09
    assert rep.vs_generic.mean_diff == pytest.approx(0.16 - 0.25)
    assert rep.brier_serenity == pytest.approx(0.16)
    assert rep.brier_generic == pytest.approx(0.25)
    assert rep.vs_placebo is not None


def test_paired_report_empty(tmpdb):
    rep = paired_report("v1")
    assert rep.n_paired == 0 and rep.warnings


def test_paired_report_warns_when_placebo_arm_dead(tmpdb):
    """placebo 臂整体坏掉（主总体为空）时告警仍必须出现（Codex 二轮 F4）。"""
    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction, Resolution

    with session_scope() as s:
        s.add(Prediction(question_id="q1", prediction_date="2026-07-05", title="t",
                         gate_state="in_domain", generic_prob=0.5, final_prob=0.6,
                         placebo_prob=None, belief_set_version="v1",
                         submit_status="submitted"))
        s.add(Resolution(question_id="q1", resolution_kind="yes", outcome=1.0))
    rep = paired_report("v1", n_boot=50)
    assert rep.n_paired == 0
    assert any("缺 placebo 臂" in w for w in rep.warnings)
