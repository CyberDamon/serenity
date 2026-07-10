"""多模型集成：每个框架跨 ≥2 个异构 frontier 模型各跑一次，框架内几何平均。

D7 决议：框架内对**存活模型**几何平均；仅当该框架**所有**模型都失败才判框架失败。
一家 provider 抖动不杀框架。

⚠️ Codex 精修（写入 DESIGN 决议）：多模型误差并不独立（同证据/同叙事），
几何池化可能放大共同过度自信而非纯降方差——"双重多样性→降方差"是待前向验证的假设，
非既定事实。ir_std / 分歧监控用于观察这一点。

ASCII：
  FrameworkClass ──┬─ run(llm=opus-4-8) ─► out_a
                   └─ run(llm=gpt-5.x)  ─► out_b
                          │
             存活 prob 几何平均 ─► 合并 FrameworkOutput(prob, 各模型明细)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from serenity.agent.aggregator import geometric_mean
from serenity.agent.frameworks.base import Framework, FrameworkOutput, Market
from serenity.agent.llm_client import LLMClient
from serenity.data.news.types import NewsItem

log = logging.getLogger(__name__)


def run_framework_ensemble(
    FrameworkClass: type[Framework],
    *,
    market: Market,
    news: list[NewsItem],
    llms: list[LLMClient],
    as_of_date: str,
) -> FrameworkOutput:
    """在多个模型上跑同一框架，几何平均存活输出。永不抛异常。

    合并规则：
      - prob = geometric_mean(存活模型的 prob)
      - status: 有 ≥1 存活 → 'ok'；全失败 → 'failed'（failure_reason 汇总）
      - contamination_warning: 任一存活模型报 True 即 True（保守）
      - reasoning/key_evidence/sources: 取第一个存活模型的（附模型标注）
      - reference_class_n: 取第一个带该字段的存活模型
      - llm_call_cost_usd: 各模型求和
    """
    # 每个模型用**独立的框架实例**（Framework.run 可能有状态，如 reference_class
    # 在 render_user_prompt 里写 self._last_estimate；线程共享会 race）。模型并发。
    def _one(llm: LLMClient) -> FrameworkOutput:
        return FrameworkClass().run(market=market, news=news, llm=llm, as_of_date=as_of_date)

    if len(llms) == 1:
        per_model = [_one(llms[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(llms)) as ex:
            per_model = list(ex.map(_one, llms))

    ok = [o for o in per_model if o.status == "ok" and o.prob is not None]
    total_cost = sum(o.llm_call_cost_usd for o in per_model)

    if not ok:
        reasons = "; ".join(f"{o.llm_model}:{o.failure_reason}" for o in per_model) or "all_failed"
        merged = FrameworkOutput(
            framework_name=FrameworkClass.name,
            status="failed",
            failure_reason=f"ensemble_all_failed[{reasons}]",
            llm_call_cost_usd=total_cost,
            llm_model="+".join(o.llm_model for o in per_model),
        )
        return merged

    probs = [float(o.prob) for o in ok]  # type: ignore[arg-type]
    combined_prob = geometric_mean(probs)
    lead = ok[0]

    models_note = ", ".join(f"{o.llm_model}={o.prob:.3f}" for o in ok)
    merged = FrameworkOutput(
        framework_name=FrameworkClass.name,
        status="ok",
        prob=combined_prob,
        yes_side_interpretation=lead.yes_side_interpretation,
        reasoning=f"[ensemble {len(ok)}/{len(per_model)} models: {models_note}]\n{lead.reasoning}",
        key_evidence=lead.key_evidence,
        sources_cited=lead.sources_cited,
        contamination_warning=any(o.contamination_warning for o in ok),
        contamination_confidence=max((o.contamination_confidence for o in ok), default=0.0),
        llm_call_cost_usd=total_cost,
        llm_model="+".join(o.llm_model for o in ok),
        reference_class_n=next((o.reference_class_n for o in ok if o.reference_class_n is not None), None),
        reference_class_confidence=next(
            (o.reference_class_confidence for o in ok if o.reference_class_n is not None), 1.0
        ),
    )
    log.info(
        "ensemble %s on %s → %d/%d ok, combined_prob=%.3f",
        FrameworkClass.name, market.token_id[:16], len(ok), len(per_model), combined_prob,
    )
    return merged
