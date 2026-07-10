"""Phase 3 单元测：在线重校准 hold-out 门控(D3) + reconcile 映射/快照(D4/D8/D13)。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from serenity.agent.calibration import (
    Calibrator,
    _pav,
    fit_calibrator,
    reset_calibrator,
)


# ── 校准器基本 ──


def test_identity_default():
    c = Calibrator()
    for p in (0.01, 0.3, 0.5, 0.99):
        assert abs(c.apply(p) - p) < 1e-9


def test_below_min_samples_stays_identity():
    pairs = [(0.5, 1.0)] * 10
    c = fit_calibrator(pairs, min_samples=60)
    assert c.method == "identity"


def test_scorer_void_outcome_not_truncated():
    """D8 回归：void 结算 outcome=0.5 不能被 scorer 截成 0。"""
    from serenity.scoring import scorer

    # p=0.9 对 void(0.5)：(0.9-0.5)^2=0.16；若被截成 0 则会是 0.81
    assert abs(scorer.summarize([0.9], [0.5]).brier_mean - 0.16) < 1e-9
    # 校准分箱的 mean_outcome 也应保留 0.5
    bins = scorer.calibration_bins([0.9], [0.5], n_bins=10)
    non_empty = [b for b in bins if b[3] > 0]
    assert non_empty and abs(non_empty[0][2] - 0.5) < 1e-9


def test_pav_monotonic():
    xs, ys = _pav([1, 2, 3, 4], [0.0, 1.0, 0.0, 1.0])
    assert ys == sorted(ys)  # 单调非降


# ── hold-out 门控（D3 关键测）──


def test_calibration_improves_on_miscalibrated():
    # raw 恒为 0.5，但真实 80% 为 YES → 恒等 Brier 高，Platt 应上移并改善
    pairs = [(0.5, 1.0 if i % 5 != 0 else 0.0) for i in range(80)]  # 80% ones
    c = fit_calibrator(pairs, min_samples=60)
    assert c.method == "platt"
    assert c.version == 1
    assert c.holdout_brier_model < c.holdout_brier_identity
    # 校准后应把 0.5 上移到接近 0.8
    assert c.apply(0.5) > 0.65


def test_calibration_gated_by_low_effective_n():
    # 原始 80 条本会拟合出改善的 Platt，但聚簇有效 N=10<30 → 门控拒绝，保持恒等
    pairs = [(0.5, 1.0 if i % 5 else 0.0) for i in range(80)]
    c = fit_calibrator(pairs, min_samples=60, effective_n=10, min_effective_n=30)
    assert c.method == "identity"
    # 有效 N 充足则照常上线
    c2 = fit_calibrator(pairs, min_samples=60, effective_n=50, min_effective_n=30)
    assert c2.method == "platt"


def test_calibration_rejected_when_no_gain():
    # raw 恒 0.5，真实 50/50 → 恒等已最优，Platt 无增益 → 门控拒绝，保持恒等
    pairs = [(0.5, float(i % 2)) for i in range(80)]  # 平衡
    c = fit_calibrator(pairs, min_samples=60)
    assert c.method == "identity"  # 未通过 hold-out 门控


def test_kill_switch_reset(tmp_path):
    path = str(tmp_path / "cal.json")
    # 先存一个非恒等
    Calibrator(method="platt", version=3, platt_a=2.0, platt_b=0.5).save(path)
    assert Calibrator.load(path).method == "platt"
    reset_calibrator(path)
    assert Calibrator.load(path).method == "identity"


# ── reconcile 映射 + 快照 + 校准对（临时 DB）──


@pytest.fixture()
def tmpdb(tmp_path, monkeypatch):
    import serenity.store.dao as dao
    from serenity.config import settings

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/t.db")
    dao._engine = None
    dao._SessionLocal = None
    dao.init_db()
    return dao


def test_map_outcome():
    from serenity.scoring.reconcile import _map_outcome

    assert _map_outcome("yes") == (1.0, False)
    assert _map_outcome("no") == (0.0, False)
    assert _map_outcome("5050") == (0.5, True)  # void（D8）
    assert _map_outcome("void") == (0.5, True)
    assert _map_outcome("") is None
    assert _map_outcome(None) is None


class _FakeQ:
    def __init__(self, kind):
        self.resolution_kind = kind
        self.actual_resolve_time = "2026-06-01T00:00:00Z"


class _FakeClient:
    def __init__(self, kinds):
        self._kinds = kinds

    def get_question(self, qid):
        return _FakeQ(self._kinds[qid])


def test_reconcile_maps_and_scores_including_shadow(tmpdb):
    from serenity.scoring.reconcile import build_calibration_pairs, reconcile
    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction, Resolution

    past = datetime(2026, 5, 1, tzinfo=UTC)
    with session_scope() as s:
        # 一条已提交 + 一条 shadow(skipped)，均已过 resolve_time
        s.add(Prediction(question_id="q_sub", prediction_date="2026-05-01", category="politics",
                         raw_prob=0.2, final_prob=0.2, submit_status="submitted",
                         question_resolve_time=past))
        s.add(Prediction(question_id="q_shadow", prediction_date="2026-05-01", category="politics",
                         raw_prob=0.7, final_prob=0.7, submit_status="skipped",
                         question_resolve_time=past))

    client = _FakeClient({"q_sub": "no", "q_shadow": "yes"})
    now = datetime(2026, 6, 15, tzinfo=UTC)
    res = reconcile(client=client, now=now)

    assert res.newly_resolved == 2
    with session_scope() as s:
        assert s.get(Resolution, "q_sub").outcome == 0.0
        assert s.get(Resolution, "q_shadow").outcome == 1.0
    assert res.snapshots_written >= 1

    # build_calibration_pairs: all 含 shadow(2 条)，submitted 仅 1 条（D13）
    assert len(build_calibration_pairs(scope="all")) == 2
    assert len(build_calibration_pairs(scope="submitted")) == 1


def test_cli_inspect(tmpdb, capsys):
    import json

    from serenity.cli import main
    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction

    with session_scope() as s:
        s.add(Prediction(
            question_id="abc12345-de", prediction_date="2026-07-03", title="Test Q",
            raw_prob=0.3, final_prob=0.3, route_label="regime_change", submit_status="submitted",
            research=json.dumps({"backend": "tavily", "queries": ["q1", "q2"], "n_sources": 2,
                                 "sources": [{"url": "https://cnn.com/x", "title": "CNN", "source": "cnn.com"}],
                                 "brief": "some synthesis"}),
        ))
    rc = main(["inspect", "abc123"])  # 前缀匹配
    out = capsys.readouterr().out
    assert rc == 0
    assert "backend=tavily" in out
    assert "q1" in out and "cnn.com" in out


def test_recently_submitted_dedup(tmpdb):
    """同题 3 天不重提（车队惯例，覆盖语义护栏）：窗口内 submitted/pending 去重；
    skipped 与窗口外的可重做。"""
    from datetime import date

    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction
    from serenity.yarrow.runner import _recently_submitted

    with session_scope() as s:
        s.add(Prediction(question_id="q_sub", prediction_date="2026-07-03", submit_status="submitted"))
        s.add(Prediction(question_id="q_pending", prediction_date="2026-07-03", submit_status="pending"))
        s.add(Prediction(question_id="q_skip", prediction_date="2026-07-03", submit_status="skipped"))
        s.add(Prediction(question_id="q_2d_ago", prediction_date="2026-07-01", submit_status="submitted"))
        s.add(Prediction(question_id="q_old", prediction_date="2026-06-25", submit_status="submitted"))
    got = _recently_submitted(date(2026, 7, 3), 3)
    assert got == {"q_sub", "q_pending", "q_2d_ago"}


def test_reconcile_skips_future_and_missing(tmpdb):
    from serenity.scoring.reconcile import reconcile
    from serenity.store.dao import session_scope
    from serenity.store.models import Prediction

    now = datetime(2026, 6, 15, tzinfo=UTC)
    with session_scope() as s:
        # 未到 resolve_time → 不查（D10）
        s.add(Prediction(question_id="q_future", prediction_date="2026-06-01",
                         raw_prob=0.3, final_prob=0.3, submit_status="submitted",
                         question_resolve_time=now + timedelta(days=30)))
        # 已过但 resolution_kind 缺失 → still_pending
        s.add(Prediction(question_id="q_lag", prediction_date="2026-05-01",
                         raw_prob=0.3, final_prob=0.3, submit_status="submitted",
                         question_resolve_time=now - timedelta(days=5)))

    client = _FakeClient({"q_lag": None})
    res = reconcile(client=client, now=now)
    assert res.checked == 1  # q_future 被 D10 过滤
    assert res.newly_resolved == 0
    assert res.still_pending == 1  # q_lag 滞后
