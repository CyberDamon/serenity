"""在线经验重校准（Brier 关键杠杆，D3 决议 + Codex 精修）。

把模型 raw_prob 经一个从前向结算数据学到的映射校正为 final_prob。

安全闸门（D3）——绝不无脑上线：
  1. **hold-out 门控**：新映射先在留出集算 Brier，只有优于恒等映射才上线。
  2. **Platt 优先**：小样本用 Platt（1 参 logistic，稳）；有效样本大（≥200）才切 isotonic(PAV)。
  3. **版本化 + 一键回退**：状态存 JSON，可 reset 回恒等。

Codex 提醒（写入决议）：Brier 0.1 主要是区分度目标，不是校准目标；重校准只修
reliability、造不出 resolution。故此模块只做"稳健、可回退、经门控"的校准，不承诺造神。

ASCII：
  (raw_prob, outcome) 历史对 ──split──► train / holdout
        train ─► fit Platt/PAV ─► candidate
        holdout ─► Brier(candidate) < Brier(identity)? ──是─► 上线并持久化
                                                        └─否─► 保持恒等
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_EPS = 1e-6
_DEFAULT_STATE_PATH = "calibrator.json"


def _clip(p: float) -> float:
    return min(max(p, _EPS), 1 - _EPS)


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


# ─────────────────────────────────────────────────────────────────────────────
# Isotonic via PAV（pool-adjacent-violators）——避免 sklearn 依赖
# ─────────────────────────────────────────────────────────────────────────────


def _pav(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    """对按 x 排序的 (x,y) 拟合单调非降阶梯。返回 (x_sorted, fitted_y)。"""
    order = sorted(range(len(x)), key=lambda i: x[i])
    xs = [x[i] for i in order]
    ys = [y[i] for i in order]
    # PAV：维护块 (sum, count, value)
    blocks: list[list[float]] = []  # [sum, count]
    for v in ys:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2 = blocks.pop()
            s1, c1 = blocks.pop()
            blocks.append([s1 + s2, c1 + c2])
    fitted: list[float] = []
    for s, c in blocks:
        fitted.extend([s / c] * int(c))
    return xs, fitted


# ─────────────────────────────────────────────────────────────────────────────
# Calibrator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Calibrator:
    """校准映射。method='identity'|'platt'|'isotonic'。"""

    method: str = "identity"
    version: int = 0
    # platt: p = sigmoid(a*logit(raw)+b)
    platt_a: float = 1.0
    platt_b: float = 0.0
    # isotonic: 分段查表（x 升序 raw，y 对应校准值）
    iso_x: list[float] = field(default_factory=list)
    iso_y: list[float] = field(default_factory=list)
    n_train: int = 0
    holdout_brier_identity: float | None = None
    holdout_brier_model: float | None = None

    def apply(self, raw: float) -> float:
        raw = _clip(float(raw))
        if self.method == "platt":
            return _clip(_sigmoid(self.platt_a * _logit(raw) + self.platt_b))
        if self.method == "isotonic" and self.iso_x:
            return _clip(float(np.interp(raw, self.iso_x, self.iso_y)))
        return raw  # identity

    # ── 持久化 ──

    def to_json(self) -> dict:
        return {
            "method": self.method, "version": self.version,
            "platt_a": self.platt_a, "platt_b": self.platt_b,
            "iso_x": self.iso_x, "iso_y": self.iso_y, "n_train": self.n_train,
            "holdout_brier_identity": self.holdout_brier_identity,
            "holdout_brier_model": self.holdout_brier_model,
        }

    @classmethod
    def from_json(cls, d: dict) -> Calibrator:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str = _DEFAULT_STATE_PATH) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str = _DEFAULT_STATE_PATH) -> Calibrator:
        p = Path(path)
        if not p.exists():
            return cls()  # identity
        try:
            return cls.from_json(json.loads(p.read_text()))
        except Exception as e:
            log.warning("calibrator load failed (%s), 用恒等", e)
            return cls()


def _brier(probs: list[float], outcomes: list[float]) -> float:
    return float(np.mean([(p - o) ** 2 for p, o in zip(probs, outcomes)]))


def _fit_platt(raw: list[float], y: list[float]) -> tuple[float, float]:
    """最小化 log-loss 拟合 p=sigmoid(a*logit(raw)+b)。scipy 优化。"""
    from scipy.optimize import minimize

    s = np.array([_logit(r) for r in raw])
    yv = np.array(y)

    def nll(theta):
        a, b = theta
        z = a * s + b
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, _EPS, 1 - _EPS)
        return -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    a, b = res.x
    return float(a), float(b)


def fit_calibrator(
    pairs: list[tuple[float, float]],
    *,
    min_samples: int = 60,
    holdout_frac: float = 0.3,
    isotonic_min_n: int = 200,
    seed_pairs_sorted: bool = False,
    effective_n: float | None = None,
    min_effective_n: int = 30,
) -> Calibrator:
    """从 (raw_prob, outcome) 拟合校准器，hold-out 门控（D3）。

    返回：优于恒等 → 拟合好的 Calibrator；否则 → identity Calibrator（不上线）。
    void 题的 outcome=0.5 也可入 pairs（D8），对拟合是弱信号但不崩。

    Codex 统计严谨性：政治题围绕同一事件聚簇，原始条数会高估独立信息量。
    调用方可传 effective_n（聚簇有效样本量）；不足 min_effective_n 则不上线（保持恒等），
    避免用"看似很多、实则 10-20 个独立事件"的样本拟合出过拟合映射。
    """
    n = len(pairs)
    if n < min_samples:
        log.info("calibrate: 样本 %d < %d，保持恒等", n, min_samples)
        return Calibrator(method="identity", n_train=n)
    if effective_n is not None and effective_n < min_effective_n:
        log.info("calibrate: 聚簇有效样本 %.1f < %d（原始 %d），保持恒等",
                 effective_n, min_effective_n, n)
        return Calibrator(method="identity", n_train=n)

    # 确定性 hold-out 切分（不引入随机源，便于 resume/复现）
    idx = list(range(n))
    if not seed_pairs_sorted:
        idx.sort(key=lambda i: pairs[i][0])  # 按 raw 排序后隔行取，保证两集分布相近
    holdout_idx = set(idx[:: int(1 / holdout_frac)]) if holdout_frac > 0 else set()
    train = [pairs[i] for i in idx if i not in holdout_idx]
    holdo = [pairs[i] for i in idx if i in holdout_idx]
    if len(train) < 10 or len(holdo) < 5:
        return Calibrator(method="identity", n_train=n)

    raw_tr = [p for p, _ in train]
    y_tr = [o for _, o in train]
    raw_ho = [p for p, _ in holdo]
    y_ho = [o for _, o in holdo]

    # 候选：小样本 Platt，大样本 isotonic
    use_iso = len(train) >= isotonic_min_n
    cand = Calibrator(method="isotonic" if use_iso else "platt", n_train=len(train))
    if use_iso:
        xs, fit = _pav(raw_tr, y_tr)
        cand.iso_x, cand.iso_y = xs, fit
    else:
        cand.platt_a, cand.platt_b = _fit_platt(raw_tr, y_tr)

    # hold-out 门控
    b_identity = _brier(raw_ho, y_ho)
    b_model = _brier([cand.apply(r) for r in raw_ho], y_ho)
    cand.holdout_brier_identity = b_identity
    cand.holdout_brier_model = b_model

    if b_model < b_identity:
        cand.version = 1
        log.info("calibrate: 上线 %s（holdout Brier %.4f < 恒等 %.4f）",
                 cand.method, b_model, b_identity)
        return cand
    log.info("calibrate: 拒绝 %s（holdout Brier %.4f ≥ 恒等 %.4f），保持恒等",
             cand.method, b_model, b_identity)
    return Calibrator(method="identity", n_train=n,
                      holdout_brier_identity=b_identity, holdout_brier_model=b_model)


def reset_calibrator(path: str = _DEFAULT_STATE_PATH) -> None:
    """一键回退恒等（kill-switch）。"""
    Calibrator().save(path)
