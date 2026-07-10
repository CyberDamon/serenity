"""结算 → 评分 → 喂重校准（数据流③）。

D4：以**本地 predictions 表驱动**（不扫全局已结算列表），逐题 GET 详情读 resolution_kind。
D10：只查 question_resolve_time 已过、且尚无 Resolution 的预测；有界并发。
D8：resolution_kind yes/no→1/0；void/非 yes-no→0.5（与 Yarrow 50-50 口径一致）。
D13：shadow（被跳过的）预测也参与拉 outcome + 校准训练。
Codex：分域 Brier 报聚簇有效样本 n_effective + 置信区间。

评分口径：
  - Brier 快照评 final_prob（真正会提交/已提交的值）。
  - 重校准训练对用 raw_prob（学 raw→outcome 映射，见 build_calibration_pairs）。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select

from serenity.scoring import scorer
from serenity.store.dao import init_db, session_scope
from serenity.store.models import CalibrationSnapshot, Prediction, Resolution
from serenity.yarrow.client import YarrowClient, parse_yarrow_time

log = logging.getLogger(__name__)

_SCORABLE = ("submitted", "skipped", "dry_run")  # shadow(skipped) 也计（D13）


def _naive_utc(dt: datetime | None) -> datetime | None:
    """归一化为 naive-UTC。SQLite 不存时区，读回是 naive；比较前统一去 tz。"""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


@dataclass
class ReconcileResult:
    checked: int = 0
    newly_resolved: int = 0
    still_pending: int = 0
    snapshots_written: int = 0


def _map_outcome(resolution_kind: str | None) -> tuple[float, bool] | None:
    """resolution_kind → (outcome, is_void)。None=未终结（下轮再拉）。"""
    if resolution_kind is None:
        return None
    kind = resolution_kind.strip().lower()
    if kind == "yes":
        return 1.0, False
    if kind == "no":
        return 0.0, False
    if kind == "":
        return None
    return 0.5, True  # void/5050/其它（D8）


def reconcile(
    *, client: YarrowClient | None = None, concurrency: int = 8, now: datetime | None = None
) -> ReconcileResult:
    init_db()
    client = client or YarrowClient()
    now = now or datetime.now(UTC)
    result = ReconcileResult()

    # D4+D10：本地待结算 + 已过 resolve_time + 尚无 Resolution 的 distinct question_id
    with session_scope() as s:
        resolved_ids = {r.question_id for r in s.execute(select(Resolution)).scalars()}
        rows = s.execute(
            select(Prediction.question_id, Prediction.question_resolve_time)
            .where(Prediction.submit_status.in_(_SCORABLE))
        ).all()
    now_naive = _naive_utc(now)
    pending: list[str] = []
    seen: set[str] = set()
    for qid, resolve_t in rows:
        if qid in resolved_ids or qid in seen:
            continue
        if resolve_t is not None and _naive_utc(resolve_t) > now_naive:
            continue  # D10：还没到结算时间，不查
        seen.add(qid)
        pending.append(qid)
    result.checked = len(pending)

    # 有界并发拉详情
    def _fetch(qid: str):
        try:
            q = client.get_question(qid)
            return qid, q.resolution_kind, parse_yarrow_time(q.actual_resolve_time)
        except Exception as e:
            log.debug("reconcile get_question %s failed: %s", qid, e)
            return qid, None, None

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        fetched = list(ex.map(_fetch, pending))

    for qid, kind, resolved_at in fetched:
        mapped = _map_outcome(kind)
        if mapped is None:
            result.still_pending += 1  # 滞后/未终结，下轮再拉
            continue
        outcome, is_void = mapped
        with session_scope() as s:
            row = s.get(Resolution, qid) or Resolution(question_id=qid)
            row.resolution_kind = kind
            row.outcome = outcome
            row.is_void = is_void
            row.resolved_at = _naive_utc(resolved_at or now)
            if s.get(Resolution, qid) is None:
                s.add(row)
        result.newly_resolved += 1

    result.snapshots_written = _write_snapshots(now)
    log.info("reconcile: checked=%d resolved=%d pending=%d snapshots=%d",
             result.checked, result.newly_resolved, result.still_pending, result.snapshots_written)
    return result


def _effective_n(keys: list[str]) -> float:
    """聚簇有效样本量（Codex）：题围绕同一事件聚簇，用粗粒度 key 估独立事件数。

    近似：不同 cluster key 的数目（key = category|title 前 3 词）。介于 [distinct, n]。
    """
    if not keys:
        return 0.0
    distinct = len(set(keys))
    # 有效样本在 distinct（完全相关）与 n（完全独立）之间，取几何折中做保守估计
    return float((distinct * len(keys)) ** 0.5)


def _cluster_key(category: str | None, title: str | None) -> str:
    words = (title or "").lower().split()[:3]
    return f"{category or '?'}|{' '.join(words)}"


def _write_snapshots(now: datetime) -> int:
    """按域 × scope(all含shadow / submitted) 写 Brier/ECE/CI 快照。"""
    snap_date = now.date().isoformat()
    written = 0
    with session_scope() as s:
        joined = s.execute(
            select(
                Prediction.final_prob, Prediction.category, Prediction.title,
                Prediction.submit_status, Resolution.outcome,
            ).join(Resolution, Resolution.question_id == Prediction.question_id)
        ).all()

    def _snap(rows, domain: str, scope: str) -> CalibrationSnapshot | None:
        probs = [float(r[0]) for r in rows if r[0] is not None]
        outs = [float(r[4]) for r in rows if r[0] is not None]
        if not probs:
            return None
        summ = scorer.summarize(probs, outs)
        keys = [_cluster_key(r[1], r[2]) for r in rows if r[0] is not None]
        n_eff = _effective_n(keys)
        # 正态近似 CI（用有效样本量，Codex）
        var = float(np.var([(p - o) ** 2 for p, o in zip(probs, outs)])) if len(probs) > 1 else 0.0
        se = (var / n_eff) ** 0.5 if n_eff > 0 else 0.0
        return CalibrationSnapshot(
            snapshot_date=snap_date, domain=domain, scope=scope,
            n=len(probs), n_effective=n_eff,
            brier_mean=summ.brier_mean,
            brier_ci_low=max(0.0, summ.brier_mean - 1.96 * se),
            brier_ci_high=summ.brier_mean + 1.96 * se,
            ece=summ.expected_calibration_error,
        )

    with session_scope() as s:
        for scope, pred in (("all", lambda r: True),
                            ("submitted", lambda r: r[3] == "submitted")):
            subset = [r for r in joined if pred(r)]
            # 总体
            snap = _snap(subset, "all", scope)
            if snap:
                s.add(snap)
                written += 1
            # 分域
            domains = {r[1] for r in subset if r[1]}
            for d in domains:
                snap = _snap([r for r in subset if r[1] == d], d, scope)
                if snap:
                    s.add(snap)
                    written += 1
    return written


def build_calibration_data(scope: str = "all") -> tuple[list[tuple[float, float]], float]:
    """返回 ((raw_prob, outcome) 对, 聚簇有效样本量)。scope='all' 含 shadow（D13）。

    有效样本量喂给 fit_calibrator 的门控（Codex：防同事件聚簇高估独立信息量）。
    """
    init_db()
    with session_scope() as s:
        rows = s.execute(
            select(
                Prediction.raw_prob, Resolution.outcome, Prediction.submit_status,
                Prediction.category, Prediction.title,
            ).join(Resolution, Resolution.question_id == Prediction.question_id)
        ).all()
    pairs: list[tuple[float, float]] = []
    keys: list[str] = []
    for raw, outcome, status, category, title in rows:
        if raw is None or outcome is None:
            continue
        if scope == "submitted" and status != "submitted":
            continue
        pairs.append((float(raw), float(outcome)))
        keys.append(_cluster_key(category, title))
    return pairs, _effective_n(keys)


def build_calibration_pairs(scope: str = "all") -> list[tuple[float, float]]:
    """(raw_prob, outcome) 对。保留兼容；有效 N 门控请用 build_calibration_data。"""
    return build_calibration_data(scope)[0]
