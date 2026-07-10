"""Brier scoring + calibration metrics for backtest evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreSummary:
    """Aggregate score over a set of predictions."""

    n: int
    brier_mean: float
    log_loss_mean: float
    accuracy: float  # directional hit rate (prob > 0.5 vs outcome)
    expected_calibration_error: float = 0.0
    mean_confidence: float = 0.0


def brier_score(prob: float, outcome: int) -> float:
    """Single-prediction Brier. (prob - outcome)^2. Lower = better."""
    return (prob - outcome) ** 2


def log_loss(prob: float, outcome: int, *, eps: float = 1e-9) -> float:
    """Single-prediction log loss. Outcome must be 0 or 1."""
    p = max(min(prob, 1 - eps), eps)
    if outcome == 1:
        return -np.log(p)
    return -np.log(1 - p)


def summarize(probs: list[float], outcomes: list[int]) -> ScoreSummary:
    """Aggregate metrics across many predictions."""
    if len(probs) != len(outcomes):
        raise ValueError(f"len(probs)={len(probs)} != len(outcomes)={len(outcomes)}")
    if not probs:
        return ScoreSummary(n=0, brier_mean=0.0, log_loss_mean=0.0, accuracy=0.0)
    p_arr = np.array(probs, dtype=float)
    # float（非 int）：否则 void 结算的 outcome=0.5(D8) 会被截成 0，Brier 失真。
    o_arr = np.array(outcomes, dtype=float)
    brier = float(np.mean((p_arr - o_arr) ** 2))
    eps = 1e-9
    p_clip = np.clip(p_arr, eps, 1 - eps)
    ll = float(np.mean(-(o_arr * np.log(p_clip) + (1 - o_arr) * np.log(1 - p_clip))))
    accuracy = float(np.mean((p_arr > 0.5) == (o_arr >= 0.5)))
    ece = expected_calibration_error(probs, outcomes)
    mean_confidence = float(np.mean(np.maximum(p_arr, 1 - p_arr)))
    return ScoreSummary(
        n=len(probs),
        brier_mean=brier,
        log_loss_mean=ll,
        accuracy=accuracy,
        expected_calibration_error=ece,
        mean_confidence=mean_confidence,
    )


def calibration_bins(
    probs: list[float], outcomes: list[int], *, n_bins: int = 10
) -> list[tuple[float, float, float, int]]:
    """Return (bin_lo, bin_hi, mean_outcome, count) per bin. For dashboard plotting."""
    if not probs:
        return []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    p = np.array(probs, dtype=float)
    o = np.array(outcomes, dtype=float)  # float: 保留 void 结算的 0.5（D8），勿截 0
    out: list[tuple[float, float, float, int]] = []
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            out.append((lo, hi, float("nan"), 0))
        else:
            out.append((lo, hi, float(o[mask].mean()), n_in))
    return out


def expected_calibration_error(
    probs: list[float], outcomes: list[int], *, n_bins: int = 10
) -> float:
    """Weighted average gap between mean predicted prob and observed frequency."""
    if len(probs) != len(outcomes):
        raise ValueError(f"len(probs)={len(probs)} != len(outcomes)={len(outcomes)}")
    if not probs:
        return 0.0

    total = len(probs)
    p = np.array(probs, dtype=float)
    o = np.array(outcomes, dtype=float)  # float: 保留 void 结算的 0.5（D8），勿截 0
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        ece += (n_in / total) * abs(float(p[mask].mean()) - float(o[mask].mean()))
    return float(ece)


@dataclass
class EdgeSummary:
    """Agent-vs-market edge diagnostics for samples with a market probability."""

    n: int
    mean_agent_edge: float
    mean_agent_brier_delta: float
    agent_beats_market_rate: float


def summarize_vs_market(
    agent_probs: list[float], market_probs: list[float], outcomes: list[int]
) -> EdgeSummary:
    """Compare agent probabilities against market consensus probabilities."""
    if not (len(agent_probs) == len(market_probs) == len(outcomes)):
        raise ValueError("agent_probs, market_probs, and outcomes must have equal length")
    if not agent_probs:
        return EdgeSummary(
            n=0,
            mean_agent_edge=0.0,
            mean_agent_brier_delta=0.0,
            agent_beats_market_rate=0.0,
        )

    a = np.array(agent_probs, dtype=float)
    m = np.array(market_probs, dtype=float)
    o = np.array(outcomes, dtype=float)  # float: 保留 void 结算的 0.5（D8），勿截 0
    agent_brier = (a - o) ** 2
    market_brier = (m - o) ** 2
    deltas = market_brier - agent_brier
    return EdgeSummary(
        n=len(agent_probs),
        mean_agent_edge=float(np.mean(a - m)),
        mean_agent_brier_delta=float(np.mean(deltas)),
        agent_beats_market_rate=float(np.mean(deltas > 0)),
    )
