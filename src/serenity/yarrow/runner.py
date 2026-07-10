"""daily runner：三臂实验管线（generic / placebo / serenity）。

流程（设计文档定稿）：
  iter open binary（全类别）
    → ① gate 三态闸门（规则直判 in_domain；LLM 裁 adjacent/out；fail-closed→out）
         out_of_domain：弃权；每轮抽 ≤out_shadow_sample 条跑 generic shadow 复盘闸门
    → ② generic 臂：research 检索 + core.predict（双模型 + 外视角锚）
    → ③ serenity 先验：检索信念 → 方向+强度档位 → δ（先缩放后封顶）
       placebo 先验（评审 6A）：随机信念同流程 → placebo δ（shadow 不提交）
    → ④ 三臂归因落库（generic_prob / placebo_prob / final_prob + belief_ids + 判据）
    → ⑤ 提交（v1 全部规则：二元题 + gate∈{in,adjacent} + generic 自检门通过）
         report.reasoning = Serenity 风格转述（引信念，不引推文原文，评审 9A）

同题 3 天内不重复提交（车队惯例，Yarrow 重提交是覆盖语义）。
"""

from __future__ import annotations

import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from serenity.agent.core import AgentPrediction, CalibratorFn, SelfCheckFn, predict
from serenity.agent.frameworks import Market
from serenity.agent.llm_client import CostCapExceeded, LLMClient, make_client
from serenity.agent.self_check import make_self_check
from serenity.config import settings
from serenity.data.research.agentic import assemble_research
from serenity.distill.pipeline import active_version
from serenity.gate.gate import GateResult, classify_question, load_gate_vocab
from serenity.prior.prior import (
    PriorResult,
    apply_delta,
    generate_placebo_prior,
    generate_prior,
)
from serenity.store.dao import init_db, session_scope
from serenity.store.models import BeliefPrimitive, Prediction
from serenity.yarrow.client import (
    FORECAST_BATCH_MAX,
    YarrowClient,
    YarrowQuestionDTO,
    parse_yarrow_time,
)

log = logging.getLogger(__name__)


@dataclass
class RunItem:
    question_id: str
    title: str
    final_prob: float | None
    submit_status: str
    skip_reason: str | None = None
    gate_state: str | None = None
    generic_prob: float | None = None
    placebo_prob: float | None = None
    delta_log_odds: float | None = None
    market_implied_prob: float | None = None


@dataclass
class RunResult:
    run_date: date
    mode: str
    belief_set_version: str = ""
    seen: int = 0
    gated_in: int = 0
    gated_adjacent: int = 0
    gated_out: int = 0
    submitted: int = 0
    skipped: int = 0
    items: list[RunItem] = field(default_factory=list)


def _build_llms(models: list[str] | None = None) -> list[LLMClient]:
    return [make_client(m) for m in (models or settings.ensemble_model_list)]


# 非二元/梗题兜底（v1 仅二元题，评审定稿）。qtype=binary 过滤为主，这里只兜格式怪题。
_NON_BINARY_RE = re.compile(r"\bhow (many|much)\b|\bwhich of\b", re.IGNORECASE)


def run_daily(
    *,
    client: YarrowClient | None = None,
    llms: list[LLMClient] | None = None,
    gate_llm: LLMClient | None = None,
    prior_llm: LLMClient | None = None,
    submit: bool = False,
    max_questions: int | None = None,
    max_scan: int = 150,
    min_lead_days: int = 1,
    min_evidence: int = 2,
    min_sources: int = 2,
    logit_dispersion_max: float | None = None,
    question_concurrency: int = 3,
    research: bool = True,
    self_check: SelfCheckFn | None = None,
    calibrator: CalibratorFn | None = None,
    resubmit_window_days: int = 3,
) -> RunResult:
    """跑一轮 daily。submit=False 为 dry-run（三臂落库，不提交）。

    max_questions = 本轮要凑够的 in_domain/adjacent 候选数；
    max_scan = 闸门扫描上限（控 gate LLM 成本）。
    """
    init_db()
    version = active_version()
    if not version:
        raise RuntimeError("无激活信念库版本——先跑 `serenity distill`（评审 3A：先验臂依赖冻结版本）")
    # v1 实验冻结守卫（Codex 验收 F2）：缩放系数固定 1.0，env 改动会静默改变处理强度。
    if abs(settings.prior_scale - 1.0) > 1e-9:
        raise RuntimeError(
            f"v1 实验期 prior_scale 固定 1.0（评审 8A），当前={settings.prior_scale}。"
            "启用缩放属 v2 变更 = 新实验段。"
        )

    client = client or YarrowClient()
    if llms is None:
        llms = _build_llms()
        # 对照臂定义冻结（Codex 验收 F6）：generic 臂 = 恰好双模型；
        # 配置漂移会静默改变对照组强度。显式传 llms（测试/实验脚本）不受限。
        if len(llms) != 2:
            raise RuntimeError(
                f"generic 对照臂要求恰好 2 个 ensemble 模型（评审 1A），"
                f"当前 ENSEMBLE_MODELS 给了 {len(llms)} 个。"
            )
    gate_llm = gate_llm or make_client(settings.gate_model)
    prior_llm = prior_llm or make_client(settings.prior_model)
    aux_llm = llms[0]
    if self_check is None:
        self_check = make_self_check(aux_llm)
    if calibrator is None:
        from serenity.agent.calibration import Calibrator
        cal = Calibrator.load()
        calibrator = cal.apply
        log.info("daily: 加载校准器 method=%s version=%s", cal.method, cal.version)
    max_questions = max_questions or settings.yarrow_max_questions_per_run
    if logit_dispersion_max is None:
        logit_dispersion_max = settings.framework_logit_std_filter

    today = datetime.now(UTC).date()
    now = datetime.now(UTC)
    result = RunResult(
        run_date=today, mode="submit" if submit else "dry_run", belief_set_version=version
    )
    vocab = load_gate_vocab(version)

    # ── ① 扫描 + 闸门 ──
    recently_done = _recently_submitted(today, resubmit_window_days)
    category = (settings.yarrow_categories.split(",")[0].strip() or None) if settings.yarrow_categories else None
    candidates: list[tuple[YarrowQuestionDTO, GateResult]] = []
    out_pool: list[tuple[YarrowQuestionDTO, GateResult]] = []
    raw_seen = 0
    for q in client.iter_questions(status="open", qtype="binary", category=category):
        raw_seen += 1
        if raw_seen > max_scan:
            log.info("触达扫描上限 max_scan=%d，停止拉题", max_scan)
            break
        if q.id in recently_done:
            continue
        if _NON_BINARY_RE.search(q.title or ""):
            _persist_minimal(q, today, "skipped", "non_binary_format", version, None, now)
            result.skipped += 1
            continue
        gate = classify_question(
            title=q.title, deadline=q.scheduled_resolve_time, vocab=vocab, llm=gate_llm
        )
        if gate.state == "out_of_domain":
            result.gated_out += 1
            out_pool.append((q, gate))
            continue
        if gate.state == "in_domain":
            result.gated_in += 1
        else:
            result.gated_adjacent += 1
        candidates.append((q, gate))
        if len(candidates) >= max_questions:
            break
    result.seen = raw_seen

    # out_of_domain：抽样 ≤N 条跑 generic shadow（闸门误判复盘）；其余只记闸门判定。
    rng = random.Random(f"outshadow:{today.isoformat()}")
    shadow_picks = rng.sample(out_pool, min(settings.out_shadow_sample, len(out_pool))) if out_pool else []
    shadow_ids = {q.id for q, _ in shadow_picks}
    for q, gate in out_pool:
        if q.id not in shadow_ids:
            _persist_minimal(q, today, "skipped", "out_of_domain", version, gate, now)
            result.skipped += 1
            result.items.append(RunItem(q.id, q.title, None, "skipped", "out_of_domain", gate.state))

    cross = _fetch_cross_markets(client, [q.id for q, _ in candidates + shadow_picks])

    # ── ②③ 并发预测（generic + prior + placebo）──
    def _predict_one(item: tuple[YarrowQuestionDTO, GateResult, bool]):
        q, gate, is_out_shadow = item
        market_prob = cross.get(q.id)
        market = Market(
            token_id=q.id, question=q.title,
            market_price=market_prob if market_prob is not None else 0.5,
            resolution_date_iso=q.scheduled_resolve_time,
        )
        news: list = []
        rlog: dict = {}
        if research:
            try:
                news, rlog = assemble_research(q.title, llm=aux_llm, as_of_date=today.isoformat())
            except Exception as e:
                log.warning("research failed for %s: %s", q.id, e)
        try:
            pred = predict(
                market=market, news=news, llms=llms, as_of_date=today,
                self_check=self_check, calibrator=calibrator,
            )
        except CostCapExceeded:
            log.warning("cost cap hit at %s — 停止有效预测", q.id)
            return q, gate, is_out_shadow, None, None, None, market_prob, (0, 0), rlog, "cost_cap_hit"
        except Exception as e:
            log.warning("predict failed for %s: %s", q.id, e)
            return q, gate, is_out_shadow, None, None, None, market_prob, (0, 0), rlog, f"predict_error:{type(e).__name__}"

        prior = placebo = None
        if not is_out_shadow:
            try:
                prior = generate_prior(
                    prior_llm, title=q.title, deadline=q.scheduled_resolve_time,
                    gate=gate, version=version,
                )
                placebo = generate_placebo_prior(
                    prior_llm, question_id=q.id, title=q.title,
                    deadline=q.scheduled_resolve_time, gate=gate, version=version,
                    real_result=prior,
                )
            except Exception as e:
                # 先验层意外异常（检索/DB 等）：fail-closed δ=0，不吞 generic 臂
                log.warning("prior generation failed for %s: %s", q.id, e)
                prior = prior or PriorResult(delta=0.0, parse_error=f"prior_error:{type(e).__name__}")
                placebo = placebo or PriorResult(delta=0.0, parse_error=f"placebo_error:{type(e).__name__}")
        return q, gate, is_out_shadow, pred, prior, placebo, market_prob, _evidence_quality(news, now), rlog, None

    work = [(q, g, False) for q, g in candidates] + [(q, g, True) for q, g in shadow_picks]
    qconc = max(1, question_concurrency)
    if qconc == 1 or len(work) <= 1:
        computed = [_predict_one(w) for w in work]
    else:
        with ThreadPoolExecutor(max_workers=qconc) as ex:
            computed = list(ex.map(_predict_one, work))

    # ── ④⑤ 串行落库 + 提交 ──
    to_submit: list[tuple[str, float, str]] = []
    for q, gate, is_out_shadow, pred, prior, placebo, market_prob, ev, rlog, perr in computed:
        if perr is not None:
            status = "skipped" if perr == "cost_cap_hit" else "failed"
            reason = "cost_cap" if perr == "cost_cap_hit" else perr
            _persist(q, today, gate, None, None, None, status, reason, market_prob, version, now, rlog)
            result.skipped += 1
            result.items.append(RunItem(q.id, q.title, None, status, reason, gate.state))
            continue

        generic_prob = pred.final_prob
        if is_out_shadow:
            # 闸门误判复盘样本：只有 generic 臂，无先验
            _persist(q, today, gate, pred, None, None, "skipped", "out_of_domain_shadow",
                     market_prob, version, now, rlog)
            result.skipped += 1
            result.items.append(RunItem(
                q.id, q.title, None, "skipped", "out_of_domain_shadow", gate.state,
                generic_prob=generic_prob,
            ))
            continue

        serenity_prob = apply_delta(generic_prob, prior.delta)
        placebo_prob = apply_delta(generic_prob, placebo.delta)

        skip_reason = _submission_gate(
            pred, q, now=now, min_lead_days=min_lead_days,
            evidence=ev, min_evidence=min_evidence, min_sources=min_sources,
            logit_dispersion_max=logit_dispersion_max,
        )
        if skip_reason is None:
            status = "dry_run" if not submit else "pending"
            to_submit.append((q.id, serenity_prob, _compose_reasoning(pred, prior, gate, serenity_prob, market_prob)))
        else:
            status = "skipped"  # shadow：三臂照存
            result.skipped += 1
        _persist(q, today, gate, pred, prior, placebo, status, skip_reason, market_prob, version, now, rlog,
                 serenity_prob=serenity_prob, placebo_prob=placebo_prob)
        result.items.append(RunItem(
            q.id, q.title, serenity_prob, status, skip_reason, gate.state,
            generic_prob=generic_prob, placebo_prob=placebo_prob,
            delta_log_odds=prior.delta, market_implied_prob=market_prob,
        ))

    if submit and to_submit:
        _submit(client, to_submit, result, now)

    log.info(
        "daily done: seen=%d in=%d adj=%d out=%d submitted=%d skipped=%d mode=%s version=%s",
        result.seen, result.gated_in, result.gated_adjacent, result.gated_out,
        result.submitted, result.skipped, result.mode, version,
    )
    return result


# ── 提交门（沿 green-water 自检门控；n_ir_valid 语义 = 双模型存活数）──────────────


def _submission_gate(
    pred: AgentPrediction,
    q: YarrowQuestionDTO,
    *,
    now: datetime,
    min_lead_days: int,
    evidence: tuple[int, int],
    min_evidence: int,
    min_sources: int,
    logit_dispersion_max: float,
) -> str | None:
    """设计文档提交规则③"generic 管线自检通过（沿 green-water 自检门控）"= 本函数整套：
    污染过滤 / 双模型存活 / logit 分歧 / 证据质量 / 最小提前量。这些是质量与反作弊门，
    不是"置信度/edge 过滤"（后者被设计明令禁止——那会给配对样本引入选择偏差；
    本门被跳过的题三臂照常 shadow 落库，配对分析不受影响）。"""
    agg = pred.aggregated
    if agg.contamination_filter_triggered:
        return "aggregator_contamination"
    if agg.n_ir_valid < 2:
        return "aggregator_model_failed"  # 双模型须都存活
    if agg.ir_logit_std > logit_dispersion_max:
        return "disagreement_logit"
    recent_count, distinct_sources = evidence
    if recent_count < min_evidence or distinct_sources < min_sources:
        return "evidence_too_thin"
    resolve_t = parse_yarrow_time(q.scheduled_resolve_time)
    if resolve_t is not None and resolve_t - now < timedelta(days=min_lead_days):
        return "lead_time_too_short"
    return None


def _evidence_quality(news, now: datetime, *, max_age_days: int = 30) -> tuple[int, int]:
    """(近期有效篇数, 不同来源数)。有效 = 摘要>40 字且 max_age_days 内。"""
    cutoff = _naive_utc_dt(now) - timedelta(days=max_age_days)
    recent = 0
    sources: set[str] = set()
    for it in news:
        summ = (it.summary or "").strip()
        if len(summ) < 40:
            continue
        pub = _naive_utc_dt(it.published_at) if it.published_at else None
        if pub is not None and pub < cutoff:
            continue
        recent += 1
        if it.source and not it.source.startswith("agentic:"):
            sources.add(it.source)
    return recent, len(sources)


def _naive_utc_dt(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def _recently_submitted(today: date, window_days: int) -> set[str]:
    """近 window_days 天内已提交/待提交的 question_id（同题不重提，覆盖语义护栏）。"""
    from sqlalchemy import select

    cutoff = (today - timedelta(days=window_days)).isoformat()
    with session_scope() as s:
        rows = s.execute(
            select(Prediction.question_id)
            .where(Prediction.prediction_date >= cutoff)
            .where(Prediction.submit_status.in_(("submitted", "pending")))
        ).all()
    return {r[0] for r in rows}


def _fetch_cross_markets(client: YarrowClient, ids: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for i in range(0, len(ids), 60):
        batch = ids[i : i + 60]
        try:
            dto = client.batch_cross_market(batch)
            for qid in batch:
                cm = dto.get(qid)
                out[qid] = cm.market_implied_prob if (cm and cm.matched) else None
        except Exception as e:
            log.debug("cross-market unavailable: %s", e)
            for qid in batch:
                out[qid] = None
    return out


# ── 落库 ──────────────────────────────────────────────────────────────────────


def _persist_minimal(
    q: YarrowQuestionDTO, today: date, status: str, reason: str,
    version: str, gate: GateResult | None, now: datetime,
) -> None:
    """out_of_domain 未抽中 shadow 的题：只记闸门判定（判据必落库）。"""
    with session_scope() as s:
        existing = s.query(Prediction).filter_by(
            question_id=q.id, prediction_date=today.isoformat()
        ).one_or_none()
        row = existing or Prediction(question_id=q.id, prediction_date=today.isoformat())
        row.title = q.title
        row.category = q.category
        row.gate_state = gate.state if gate else None
        row.gate_rationale = gate.rationale if gate else None
        row.belief_set_version = version
        row.submit_status = status
        row.skip_reason = reason
        row.prediction_ts = now
        row.question_resolve_time = parse_yarrow_time(q.scheduled_resolve_time)
        if gate and gate.parse_error:
            row.parse_errors = json.dumps([f"gate:{gate.parse_error}"])
        if existing is None:
            s.add(row)


def _persist(
    q: YarrowQuestionDTO,
    today: date,
    gate: GateResult,
    pred: AgentPrediction | None,
    prior: PriorResult | None,
    placebo: PriorResult | None,
    status: str,
    skip_reason: str | None,
    market_prob: float | None,
    version: str,
    now: datetime,
    research_log: dict | None = None,
    *,
    serenity_prob: float | None = None,
    placebo_prob: float | None = None,
) -> None:
    parse_errors: list[str] = []
    if gate.parse_error:
        parse_errors.append(f"gate:{gate.parse_error}")
    if prior and prior.parse_error:
        parse_errors.append(f"prior:{prior.parse_error}")
    if placebo and placebo.parse_error:
        parse_errors.append(f"placebo:{placebo.parse_error}")

    with session_scope() as s:
        existing = s.query(Prediction).filter_by(
            question_id=q.id, prediction_date=today.isoformat()
        ).one_or_none()
        row = existing or Prediction(question_id=q.id, prediction_date=today.isoformat())
        row.title = q.title
        row.category = q.category
        if research_log:
            row.research = json.dumps(research_log, ensure_ascii=False)
        row.gate_state = gate.state
        row.gate_rationale = gate.rationale
        row.belief_set_version = version
        row.raw_prob = pred.raw_prob if pred else None
        row.generic_prob = pred.final_prob if pred else None
        row.final_prob = serenity_prob
        row.placebo_prob = placebo_prob
        row.delta_log_odds = prior.delta if prior else None
        row.placebo_delta_log_odds = placebo.delta if placebo else None
        row.prior_direction = prior.direction if prior else None
        row.prior_strength = prior.strength if prior else None
        row.belief_ids = json.dumps(prior.belief_ids) if prior else None
        row.prior_rationale = prior.rationale if prior else None
        row.parse_errors = json.dumps(parse_errors) if parse_errors else None
        row.ir_std = pred.aggregated.ir_std if pred else None
        row.n_ir_valid = pred.aggregated.n_ir_valid if pred else None
        row.route_label = pred.route_label if pred else None
        row.llm_models = pred.llm_models if pred else None
        row.self_check_delta = pred.self_check_delta if pred else None
        row.market_implied_prob = market_prob
        row.submit_status = status
        row.skip_reason = skip_reason
        row.prediction_ts = now
        row.question_resolve_time = parse_yarrow_time(q.scheduled_resolve_time)
        if existing is None:
            s.add(row)


# ── 提交报告（Serenity 风格：引信念转述，不引推文原文，评审 9A）──────────────────


def _compose_reasoning(
    pred: AgentPrediction,
    prior: PriorResult,
    gate: GateResult,
    serenity_prob: float,
    market_prob: float | None,
    max_chars: int = 1800,
) -> str:
    head = (
        f"YES probability = {serenity_prob:.3f} "
        f"(generic baseline {pred.final_prob:.3f}, analyst-prior delta "
        f"{prior.delta:+.2f} log-odds, gate={gate.state})."
    )
    if market_prob is not None:
        head += f" Market-implied YES={market_prob:.3f}."
    parts = [head, ""]
    if prior.direction != "none" and prior.belief_ids:
        claims = _belief_claims(prior.belief_ids)
        parts.append(
            f"Supply-chain view ({prior.direction.upper()} tilt, {prior.strength}): "
            f"{' '.join(prior.rationale.split())[:500]}"
        )
        if claims:
            parts.append("Anchored on distilled theses:")
            parts.extend(f"- {c[:220]}" for c in claims[:4])
    else:
        parts.append(
            "Analyst belief base offers no directional edge on this question; "
            "probability rests on the generic evidence-based estimate."
        )
    if pred.reference_class_output and pred.reference_class_output.status == "ok":
        parts.append(
            f"Outside view anchor (reference class) = {pred.reference_class_output.prob:.3f}."
        )
    text = "\n".join(parts)
    return text[:max_chars]


def _belief_claims(belief_ids: list[int]) -> list[str]:
    if not belief_ids:
        return []
    with session_scope() as s:
        rows = s.query(BeliefPrimitive).filter(BeliefPrimitive.id.in_(belief_ids)).all()
        return [r.claim for r in rows]


def _submit(
    client: YarrowClient,
    to_submit: list[tuple[str, float, str]],
    result: RunResult,
    now: datetime,
) -> None:
    """逐 chunk 提交，单 chunk 失败不拖垮后续（Codex 验收 F5）：
    失败 chunk 的行标 submit_status='failed' + skip_reason='submit_error'，
    绝不留假 'pending'（假 pending 会被 3 天去重挡住重试、又进不了 reconcile）。"""
    for i in range(0, len(to_submit), FORECAST_BATCH_MAX):
        chunk = to_submit[i : i + FORECAST_BATCH_MAX]
        chunk_ids = {qid for qid, _p, _r in chunk}
        payload = [
            {"question_id": qid, "probability_yes": p, "report": {"reasoning": reasoning}}
            for qid, p, reasoning in chunk
        ]
        try:
            client.submit_forecasts(payload)
        except Exception as e:
            log.error("submit chunk failed (%d 条): %s", len(chunk), e)
            _mark_chunk(chunk_ids, "failed", f"submit_error:{type(e).__name__}", result, now)
            continue
        _mark_chunk(chunk_ids, "submitted", None, result, now)
        result.submitted += len(chunk)


def _mark_chunk(
    chunk_ids: set[str], status: str, skip_reason: str | None,
    result: RunResult, now: datetime,
) -> None:
    with session_scope() as s:
        for qid in chunk_ids:
            row = s.query(Prediction).filter_by(question_id=qid).order_by(
                Prediction.id.desc()
            ).first()
            if row:
                row.submit_status = status
                row.skip_reason = skip_reason or row.skip_reason
                if status == "submitted":
                    row.first_submit_ts = row.first_submit_ts or now
    for item in result.items:
        if item.question_id in chunk_ids:
            item.submit_status = status
            if skip_reason:
                item.skip_reason = skip_reason
