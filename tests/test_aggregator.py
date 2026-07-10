"""aggregator 单元测：reference-class 置信度降权（Codex⑤）。"""

from __future__ import annotations

from serenity.agent.aggregator import aggregate
from serenity.agent.frameworks.base import FrameworkOutput


def _ok(name, p):
    return FrameworkOutput(framework_name=name, status="ok", prob=p)


def _agg(conf):
    ir = [_ok("a", 0.2), _ok("b", 0.2), _ok("c", 0.2)]  # ir_geo ≈ 0.2
    ref = FrameworkOutput(framework_name="reference_class", status="ok", prob=0.8)
    return aggregate(
        ir, ref, std_filter=0.15, reference_class_max_weight=0.7,
        reference_class_n_for_full_weight=50, reference_class_n=100,
        reference_class_confidence=conf,
    )


def test_confidence_downweights_anchor():
    hi = _agg(1.0)   # 满权重锚(0.8) 把 final 往上拉
    lo = _agg(0.3)   # 低置信 → 锚权重被压到 0.21，final 更靠近 ir_geo(0.2)
    assert lo.final_prob < hi.final_prob
    assert abs(lo.final_prob - 0.2) < abs(hi.final_prob - 0.2)


def test_zero_confidence_equals_ir_geo():
    z = _agg(0.0)   # confidence=0 → 锚权重 0 → final == ir_geo
    assert abs(z.final_prob - z.ir_geometric_mean) < 1e-9


def _disagree(probs):
    ir = [_ok(f"f{i}", p) for i, p in enumerate(probs)]
    return aggregate(
        ir, None, logit_std_filter=1.1,
        reference_class_max_weight=0.7, reference_class_n_for_full_weight=50,
    )


def test_disagreement_uses_logit_single_metric():
    # 分歧判定统一走 logit（非 prob-std）：极分裂 → triggered；紧凑 → 不 triggered
    split = _disagree([0.05, 0.95])
    assert split.disagreement_filter_triggered is True
    assert split.ir_logit_std > 1.1
    tight = _disagree([0.30, 0.33, 0.31])
    assert tight.disagreement_filter_triggered is False
    assert tight.ir_logit_std < 1.1
