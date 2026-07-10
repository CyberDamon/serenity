"""Agent 编排（generic 对照臂）：market + news + 双模型 → AggregatedResult。

单一入口 predict()。runner 只调它得到 generic 臂；Serenity 先验（prior/placebo）
在 runner 层作用于本输出的 log-odds 之上，不进入本模块。

ASCII（评审 1A 定稿的对照臂配方）：
  predict(market, news, llms, as_of_date)
        │
        ├─ GenericAnalyst.run(llm=A) ─► out_A ┐   （inside view，每模型独立输出，
        ├─ GenericAnalyst.run(llm=B) ─► out_B ┤    双模型分歧 = ir_logit_std 提交门）
        └─ run_framework_ensemble(ReferenceClass, llms) ─► ref 锚（外视角）
                       │
                       ▼
        aggregate(几何平均(out_A,out_B) ──log-odds──► 与 ref 锚融合)
                       │
        [self_check 钩子]  [calibrator 钩子]
                       ▼
        AgentPrediction{raw_prob, final_prob, ...}
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

from serenity.agent.aggregator import AggregatedResult, aggregate
from serenity.agent.ensemble import run_framework_ensemble
from serenity.agent.frameworks import GenericAnalyst, Market, ReferenceClass
from serenity.agent.frameworks.base import FrameworkOutput
from serenity.agent.llm_client import LLMClient
from serenity.config import settings
from serenity.data.news.types import NewsItem

log = logging.getLogger(__name__)

SelfCheckFn = Callable[["AgentPrediction"], float]
CalibratorFn = Callable[[float], float]


@dataclass
class AgentPrediction:
    market: Market
    as_of_date: date
    aggregated: AggregatedResult
    ir_outputs: list[FrameworkOutput]  # GenericAnalyst 每模型一条
    reference_class_output: FrameworkOutput | None
    raw_prob: float  # 聚合后、重校准前
    final_prob: float  # 重校准后
    route_label: str = "generic"
    llm_models: str = ""
    news_context_size: int = 0
    self_check_delta: float | None = None
    llm_cost_usd: float = 0.0


def predict(
    *,
    market: Market,
    news: list[NewsItem],
    llms: list[LLMClient],
    as_of_date: date | None = None,
    self_check: SelfCheckFn | None = None,
    calibrator: CalibratorFn | None = None,
) -> AgentPrediction:
    """generic 对照臂：GenericAnalyst 每模型独立输出 + ReferenceClass 锚 → 聚合。

    每模型输出保持独立（不先几何平均成一条）：aggregate 内部做几何平均，
    同时跨模型分歧成为 ir_logit_std —— runner 提交门直接复用（模型间
    分歧过大 = 不确定性高 = 不提交），语义与 green-water n_ir_valid≥2 对齐。
    """
    if not llms:
        raise ValueError("predict() 需要至少一个 LLMClient")
    if as_of_date is None:
        as_of_date = date.today()
    as_of_iso = as_of_date.isoformat()

    def _run_generic(llm: LLMClient) -> FrameworkOutput:
        return GenericAnalyst().run(market=market, news=news, llm=llm, as_of_date=as_of_iso)

    def _run_ref() -> FrameworkOutput:
        return run_framework_ensemble(
            ReferenceClass, market=market, news=news, llms=llms, as_of_date=as_of_iso
        )

    # generic × 每模型 + ref 集成，全部并发（墙钟 ≈ 1 次调用）。
    with ThreadPoolExecutor(max_workers=len(llms) + 1) as ex:
        generic_futures = [ex.submit(_run_generic, llm) for llm in llms]
        ref_future = ex.submit(_run_ref)
        ir_outputs = [f.result() for f in generic_futures]
        ref_output: FrameworkOutput | None = ref_future.result()

    aggregated = aggregate(
        ir_outputs,
        ref_output,
        logit_std_filter=settings.framework_logit_std_filter,
        reference_class_max_weight=0.7,
        reference_class_n_for_full_weight=50,
        reference_class_n=(ref_output.reference_class_n if ref_output else None),
        reference_class_confidence=(
            ref_output.reference_class_confidence if ref_output else 1.0
        ),
    )

    raw_prob = float(aggregated.final_prob)
    llm_cost = sum(o.llm_call_cost_usd for o in ir_outputs) + (
        ref_output.llm_call_cost_usd if ref_output else 0.0
    )

    pred = AgentPrediction(
        market=market,
        as_of_date=as_of_date,
        aggregated=aggregated,
        ir_outputs=ir_outputs,
        reference_class_output=ref_output,
        raw_prob=raw_prob,
        final_prob=raw_prob,
        llm_models=",".join(settings.ensemble_model_list),
        news_context_size=len(news),
        llm_cost_usd=llm_cost,
    )

    if self_check is not None:
        checked = float(self_check(pred))
        pred.self_check_delta = checked - raw_prob
        raw_prob = checked
        pred.raw_prob = raw_prob

    pred.final_prob = float(calibrator(raw_prob)) if calibrator is not None else raw_prob
    return pred
