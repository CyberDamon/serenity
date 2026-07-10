"""Framework aggregator — geometric mean + log-odds fusion + std-disagreement filter.

Per design v2.3 Revision 3 Section 2.1 + D7 (eng review):
  1. IR ensemble = geometric mean of valid framework probs (D7: skip failed)
  2. Anchor = reference_class output (when available)
  3. Final = log_odds_fuse(ir_geometric_mean, reference_class_anchor,
                            weight = min(0.7, n / 50))
  4. Disagreement filter: if std-dev across IR probs > 0.15, mark
     `disagreement_filter_triggered=True` → trade_eligible=False
     (the prediction is still STORED — we just won't ticket it)
  5. partial_aggregation flag set when any framework status != 'ok'

Mathematical notes:
  - Geometric mean is robust to outliers in [0,1] probability space
  - Log-odds (Bayesian) fusion is the right way to combine ind. probability estimates;
    weight on base rate scales with sample size confidence (n=50 → max weight 0.7)
  - Clipping protects against log(0) when a framework outputs exact 0 or 1

ASCII flow:

  framework_outputs (list)
        │
        ▼
  filter status='ok' ──▶ valid_outputs
        │                       │
        │                       ▼
        │             std(probs) > 0.15? ──▶ disagreement_filter=True
        │                       │                       │
        │                       ▼                       ▼
        │              geometric_mean(probs)    final prob still computed
        │                       │                       │
        │                       ▼                       │
        ▼              log_odds_fuse(geo_mean,◀────────┘
  reference_class    base_rate, weight)
  output (or None)            │
                              ▼
                        AggregatedResult(prob, partial, trade_eligible, ...)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from serenity.agent.frameworks.base import FrameworkOutput

EPS = 1e-9  # clip prob to [EPS, 1-EPS] before log-odds


@dataclass
class AggregatedResult:
    """Output of aggregator — feeds risk_control / ticket_builder downstream."""

    final_prob: float  # final aggregated YES probability
    ir_geometric_mean: float  # before log-odds fusion with anchor
    reference_class_prob: float | None  # anchor (None if reference_class failed)
    ir_std: float  # std-dev across IR probs (prob 空间，观测用)

    n_ir_valid: int  # how many IR frameworks succeeded (1..5)
    n_ir_total: int  # how many were attempted (always 5 for v0 IR)

    partial_aggregation: bool  # True when any framework status != 'ok'
    disagreement_filter_triggered: bool  # True when ir_logit_std > logit_std_filter（logit 空间）
    trade_eligible: bool  # False if partial / disagreement / contamination / n_ir_valid < 2

    failed_frameworks: list[str]  # names that failed (for skip_reason / debug)

    ir_logit_std: float = 0.0  # logit 空间离散度（提交门单一度量；Codex）

    # V3 additions
    n_contamination_warnings: int = 0  # how many valid frameworks self-flagged contamination
    contamination_filter_triggered: bool = False  # True when ≥50% valid frameworks flagged contamination

    # V4: granular skip reason (set when trade_eligible=False)
    # Codes: 'aggregator_partial', 'aggregator_disagreement', 'aggregator_contamination',
    # 'aggregator_few_frameworks', 'aggregator_all_failed', None when eligible.
    skip_reason: str | None = None

    # V5: structured disagreement analysis (always computed, most useful when
    # disagreement_filter_triggered=True). Names the epistemic camps on each side.
    disagreement_axis: str | None = None       # e.g. "material_power_bullish_vs_normative_bearish"
    disagreement_high_side: list[str] = field(default_factory=list)
    disagreement_low_side: list[str] = field(default_factory=list)


# Epistemic camp groupings — used by disagreement analysis.
# Frameworks not listed here are treated as "unclassified" (neutral).
_CAMP: dict[str, str] = {
    "offensive_realism":       "material_power",
    "power_transition":        "material_power",
    "geopolitics":             "material_power",
    "coercive_bargaining":     "strategic_interaction",
    "misperception":           "strategic_interaction",
    "constructivism":          "normative",
    "neoliberal_institutionalism": "normative",
    "selectorate":             "domestic_politics",
    "reference_class":         "methodological",
}


def _analyze_disagreement(
    valid_ir: list[FrameworkOutput],
) -> tuple[str | None, list[str], list[str]]:
    """Split valid frameworks into high/low sides and name the disagreement axis.

    Returns (axis_label, high_side_names, low_side_names).
    axis_label is None when fewer than 2 frameworks are available.
    """
    if len(valid_ir) < 2:
        return None, [], []

    median_prob = statistics.median(o.prob for o in valid_ir)  # type: ignore[arg-type]
    high = [o for o in valid_ir if o.prob is not None and o.prob >= median_prob]
    low  = [o for o in valid_ir if o.prob is not None and o.prob < median_prob]

    high_names = [o.framework_name for o in high]
    low_names  = [o.framework_name for o in low]

    # Identify which camps dominate each side
    def dominant_camp(names: list[str]) -> str:
        counts: dict[str, int] = {}
        for n in names:
            c = _CAMP.get(n, "unclassified")
            counts[c] = counts.get(c, 0) + 1
        if not counts:
            return "unclassified"
        return max(counts, key=lambda k: counts[k])

    high_camp = dominant_camp(high_names)
    low_camp  = dominant_camp(low_names)

    if high_camp == low_camp:
        axis = f"within_{high_camp}"
    else:
        axis = f"{high_camp}_bullish_vs_{low_camp}_bearish"

    return axis, high_names, low_names


def logit_dispersion(probs: list[float]) -> float:
    """IR 概率在 logit 空间的总体标准差（Codex：优于概率空间 std 的不确定度度量）。

    单一实现，供 aggregator 存 ir_logit_std、runner 提交门共用，避免双轨阈值漂移。
    """
    if len(probs) < 2:
        return 0.0
    logits = [math.log(min(max(p, EPS), 1 - EPS) / (1 - min(max(p, EPS), 1 - EPS))) for p in probs]
    return statistics.pstdev(logits)


def geometric_mean(probs: list[float]) -> float:
    """Geometric mean over [0,1] probs. Clipped to [EPS, 1-EPS] to avoid log(0)."""
    if not probs:
        return 0.5  # neutral fallback (caller should never use if list empty)
    clipped = [min(max(p, EPS), 1 - EPS) for p in probs]
    log_sum = sum(math.log(p) for p in clipped)
    return math.exp(log_sum / len(clipped))


def log_odds_fuse(prob_a: float, prob_b: float, weight_b: float) -> float:
    """Bayesian log-odds fusion of two probability estimates.

    weight_b ∈ [0, 1] is how much we trust prob_b vs prob_a.
    weight_b=0 → return prob_a unchanged. weight_b=1 → return prob_b.

    log-odds(p) = log(p / (1-p)). Fused log-odds is linear in weights, then
    convert back via sigmoid.
    """
    if not (0.0 <= weight_b <= 1.0):
        raise ValueError(f"weight_b must be in [0,1], got {weight_b}")
    pa = min(max(prob_a, EPS), 1 - EPS)
    pb = min(max(prob_b, EPS), 1 - EPS)
    lo_a = math.log(pa / (1 - pa))
    lo_b = math.log(pb / (1 - pb))
    fused_lo = (1 - weight_b) * lo_a + weight_b * lo_b
    # Sigmoid
    return 1.0 / (1.0 + math.exp(-fused_lo))


def aggregate(
    ir_outputs: list[FrameworkOutput],
    reference_class_output: FrameworkOutput | None,
    *,
    std_filter: float = 0.15,  # 已弃用（prob 空间）；保留仅为向后兼容，不再驱动分歧判定
    logit_std_filter: float = 1.1,  # 单一分歧阈值（logit 空间），与 runner 提交门同源
    reference_class_max_weight: float,
    reference_class_n_for_full_weight: int,
    reference_class_n: int | None = None,
    reference_class_confidence: float = 1.0,
) -> AggregatedResult:
    """Combine framework outputs into a single AggregatedResult.

    Args:
      ir_outputs: list of FrameworkOutput from the 5 IR frameworks. Failed
        outputs (status != 'ok') are skipped per D7.
      reference_class_output: optional FrameworkOutput from reference_class.
        When status='ok' provides the Bayesian anchor.
      std_filter: threshold (e.g. 0.15) above which to flag disagreement.
      reference_class_max_weight: cap on anchor weight (e.g. 0.7).
      reference_class_n_for_full_weight: sample-size for full weight (e.g. 50).
      reference_class_n: actual number of historical analogues found by the
        reference_class framework. If None, defaults to 5 (conservative).
    """
    valid_ir = [o for o in ir_outputs if o.status == "ok" and o.prob is not None]
    failed_ir = [o.framework_name for o in ir_outputs if o.status != "ok"]

    n_ir_valid = len(valid_ir)
    n_ir_total = len(ir_outputs)

    if n_ir_valid == 0:
        # All IR frameworks failed — bail with neutral anchor or 0.5
        ref_prob = (
            reference_class_output.prob
            if (
                reference_class_output is not None
                and reference_class_output.status == "ok"
                and reference_class_output.prob is not None
            )
            else None
        )
        return AggregatedResult(
            final_prob=ref_prob if ref_prob is not None else 0.5,
            ir_geometric_mean=0.5,
            reference_class_prob=ref_prob,
            ir_std=0.0,
            n_ir_valid=0,
            n_ir_total=n_ir_total,
            partial_aggregation=True,
            disagreement_filter_triggered=False,
            trade_eligible=False,
            failed_frameworks=failed_ir,
            skip_reason="aggregator_all_failed",
        )

    ir_probs = [o.prob for o in valid_ir]  # type: ignore[misc]  # we filtered None
    ir_geo = geometric_mean(ir_probs)
    ir_std = statistics.pstdev(ir_probs) if len(ir_probs) > 1 else 0.0
    ir_logit_std = logit_dispersion(ir_probs)

    partial = bool(failed_ir) or (
        reference_class_output is not None and reference_class_output.status != "ok"
    )
    # 分歧判定统一走 logit 空间（与 runner 提交门同一度量+同一阈值），消除双轨漂移。
    # ir_std（prob 空间）仅保留作观测，不再驱动 disagreement/trade_eligible。
    disagreement = ir_logit_std > logit_std_filter

    # V3 contamination guard: when ≥50% of valid frameworks self-report
    # contamination_warning=True, the LLM is recalling training data rather
    # than predicting. Reject the ticket — calibration is untrustworthy.
    n_contam = sum(1 for o in valid_ir if o.contamination_warning)
    contam_triggered = n_ir_valid >= 2 and n_contam * 2 >= n_ir_valid

    # Apply reference class anchor if available
    if (
        reference_class_output is not None
        and reference_class_output.status == "ok"
        and reference_class_output.prob is not None
    ):
        ref_prob = float(reference_class_output.prob)
        n = reference_class_n if reference_class_n is not None else 5
        # 锚权重 = 样本量项 × 匹配置信度（Codex⑤：错配/低置信参考类不该拿满权重压过集成）
        weight = min(reference_class_max_weight, n / reference_class_n_for_full_weight)
        weight *= max(0.0, min(1.0, reference_class_confidence))
        final = log_odds_fuse(ir_geo, ref_prob, weight)
    else:
        ref_prob = None
        final = ir_geo

    trade_eligible = (
        not partial
        and not disagreement
        and not contam_triggered
        and n_ir_valid >= 2
    )

    # V4: granular skip reason — pick the most specific failure
    if trade_eligible:
        skip = None
    elif contam_triggered:
        skip = "aggregator_contamination"
    elif disagreement:
        skip = "aggregator_disagreement"
    elif partial:
        skip = "aggregator_partial"
    elif n_ir_valid < 2:
        skip = "aggregator_few_frameworks"
    else:
        skip = "aggregator_ineligible"

    # V5: structured disagreement — always compute, most useful when disagreement=True
    dis_axis, dis_high, dis_low = _analyze_disagreement(valid_ir)

    return AggregatedResult(
        final_prob=final,
        ir_geometric_mean=ir_geo,
        reference_class_prob=ref_prob,
        ir_std=ir_std,
        ir_logit_std=ir_logit_std,
        n_ir_valid=n_ir_valid,
        n_ir_total=n_ir_total,
        partial_aggregation=partial,
        disagreement_filter_triggered=disagreement,
        trade_eligible=trade_eligible,
        failed_frameworks=failed_ir,
        n_contamination_warnings=n_contam,
        contamination_filter_triggered=contam_triggered,
        skip_reason=skip,
        disagreement_axis=dis_axis,
        disagreement_high_side=dis_high,
        disagreement_low_side=dis_low,
    )
