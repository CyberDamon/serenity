"""三臂配对比较：聚类 bootstrap 95% CI（评审 6A/7A 定稿）。

主指标：in_domain/adjacent 题上
  d_generic  = brier(serenity) - brier(generic)    （总效应；负 = serenity 更好）
  d_placebo  = brier(serenity) - brier(placebo)    （信念内容净效应，剥离 prompt 结构）
方法论主张成立须两个对比同向为负。

聚类 bootstrap：AI 供应链题按公司/财报/行情批量出现，题目不独立——普通
bootstrap 会假窄。以 (topic_key, 月份) 为簇整簇重抽（cluster bootstrap，
percentile CI）。topic_key = 题目命中的首个信念库 ticker，否则闸门 domain，
否则题目首词。

以效应量+区间为口径（不设显著性硬门槛，样本功效论证见设计文档）：
  区间全负 = 明确有效；跨零 = 报点估计继续积累；全正 = 明确有害。
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field

from sqlalchemy import select

from serenity.gate.gate import _DOMAIN_KEYWORDS, GateVocab, _rule_match, load_gate_vocab
from serenity.store.dao import init_db, session_scope
from serenity.store.models import Prediction, Resolution

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


@dataclass
class PairedRow:
    question_id: str
    title: str
    month: str  # YYYY-MM（prediction_date）
    gate_state: str
    outcome: float
    generic_prob: float
    serenity_prob: float
    placebo_prob: float | None
    cluster: str = ""


@dataclass
class ArmDiffCI:
    n: int
    n_clusters: int
    mean_diff: float
    ci_low: float
    ci_high: float

    @property
    def verdict(self) -> str:
        if self.ci_high < 0:
            return "明确有效（CI 全负）"
        if self.ci_low > 0:
            return "明确有害（CI 全正）"
        return "跨零——继续积累样本"


@dataclass
class PairedReport:
    version: str | None
    n_paired: int = 0
    brier_generic: float | None = None
    brier_serenity: float | None = None
    brier_placebo: float | None = None
    brier_market: float | None = None
    n_market: int = 0
    vs_generic: ArmDiffCI | None = None
    vs_placebo: ArmDiffCI | None = None
    warnings: list[str] = field(default_factory=list)


def brier(p: float, outcome: float) -> float:
    return (float(p) - float(outcome)) ** 2


def _topic_key(title: str, vocab: GateVocab) -> str:
    hit_tickers, hit_kw = _rule_match(title, vocab)
    if hit_tickers:
        return hit_tickers[0]
    if hit_kw:
        return hit_kw[0].split(":", 1)[0]
    for w in _WORD_RE.findall(title):
        lw = w.lower()
        if lw not in ("will", "the", "before", "does", "did", "for", "and", "with"):
            return lw
    return "misc"


def cluster_key(title: str, month: str, vocab: GateVocab) -> str:
    return f"{_topic_key(title, vocab)}:{month}"


def collect_paired_rows(version: str | None = None) -> list[PairedRow]:
    """拉已结算的三臂配对样本（gate∈{in,adjacent}；placebo 可缺，主对比不受影响）。"""
    init_db()
    with session_scope() as s:
        q = (
            select(Prediction, Resolution.outcome)
            .join(Resolution, Resolution.question_id == Prediction.question_id)
            .where(Prediction.gate_state.in_(("in_domain", "adjacent")))
            .where(Prediction.generic_prob.is_not(None))
            .where(Prediction.final_prob.is_not(None))
            .where(Resolution.outcome.is_not(None))
            .where(Resolution.is_void.is_(False))
        )
        if version:
            q = q.where(Prediction.belief_set_version == version)
        rows = s.execute(q).all()
    out: list[PairedRow] = []
    for pred, outcome in rows:
        out.append(PairedRow(
            question_id=pred.question_id,
            title=pred.title or "",
            month=(pred.prediction_date or "")[:7],
            gate_state=pred.gate_state,
            outcome=float(outcome),
            generic_prob=float(pred.generic_prob),
            serenity_prob=float(pred.final_prob),
            placebo_prob=float(pred.placebo_prob) if pred.placebo_prob is not None else None,
            cluster="",
        ))
    return out


def clustered_bootstrap_ci(
    diffs: list[float],
    clusters: list[str],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 20260710,
) -> ArmDiffCI:
    """整簇重抽的 percentile bootstrap CI。

    合成数据可验证（CRITICAL 测试）：植入已知效应 → CI 覆盖真值；
    零效应 → CI 跨零。
    """
    if len(diffs) != len(clusters):
        raise ValueError("diffs 与 clusters 长度不一致")
    n = len(diffs)
    if n == 0:
        return ArmDiffCI(n=0, n_clusters=0, mean_diff=float("nan"),
                         ci_low=float("nan"), ci_high=float("nan"))
    by_cluster: dict[str, list[float]] = {}
    for d, c in zip(diffs, clusters):
        by_cluster.setdefault(c, []).append(d)
    keys = sorted(by_cluster)
    mean_diff = sum(diffs) / n
    if len(keys) < 2:
        return ArmDiffCI(n=n, n_clusters=len(keys), mean_diff=mean_diff,
                         ci_low=float("nan"), ci_high=float("nan"))

    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(len(keys)):
            sample.extend(by_cluster[keys[rng.randrange(len(keys))]])
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return ArmDiffCI(
        n=n, n_clusters=len(keys), mean_diff=mean_diff,
        ci_low=boots[lo_idx], ci_high=boots[hi_idx],
    )


def paired_report(version: str | None = None, *, n_boot: int = 5000) -> PairedReport:
    rows = collect_paired_rows(version)
    rep = PairedReport(version=version, n_paired=len(rows))
    if not rows:
        rep.warnings.append("暂无已结算配对样本")
        return rep

    vocab = load_gate_vocab(version) if version else GateVocab(
        tickers=set(), domains=set(_DOMAIN_KEYWORDS)
    )
    for r in rows:
        r.cluster = cluster_key(r.title, r.month, vocab)

    rep.brier_generic = sum(brier(r.generic_prob, r.outcome) for r in rows) / len(rows)
    rep.brier_serenity = sum(brier(r.serenity_prob, r.outcome) for r in rows) / len(rows)

    d_gen = [brier(r.serenity_prob, r.outcome) - brier(r.generic_prob, r.outcome) for r in rows]
    rep.vs_generic = clustered_bootstrap_ci(d_gen, [r.cluster for r in rows], n_boot=n_boot)

    with_placebo = [r for r in rows if r.placebo_prob is not None]
    if with_placebo:
        rep.brier_placebo = sum(brier(r.placebo_prob, r.outcome) for r in with_placebo) / len(with_placebo)
        d_pla = [
            brier(r.serenity_prob, r.outcome) - brier(r.placebo_prob, r.outcome)
            for r in with_placebo
        ]
        rep.vs_placebo = clustered_bootstrap_ci(
            d_pla, [r.cluster for r in with_placebo], n_boot=n_boot
        )
    else:
        rep.warnings.append("无 placebo 臂样本（旧数据？）")

    # 次指标：vs 提交时市场价
    with session_scope() as s:
        mrows = s.execute(
            select(Prediction.market_implied_prob, Resolution.outcome)
            .join(Resolution, Resolution.question_id == Prediction.question_id)
            .where(Prediction.gate_state.in_(("in_domain", "adjacent")))
            .where(Prediction.market_implied_prob.is_not(None))
            .where(Resolution.outcome.is_not(None))
            .where(Resolution.is_void.is_(False))
        ).all()
    if mrows:
        rep.n_market = len(mrows)
        rep.brier_market = sum(brier(m, o) for m, o in mrows) / len(mrows)
    return rep
