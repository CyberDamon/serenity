"""Serenity 信念先验：检索 → LLM 方向+强度档位 → δ 网格 → 先缩放后封顶。

设计红线（评审定稿，实验期冻结，改 = 新实验段）：
  - LLM 不直接吐数值：只输出方向（yes/no/none）+ 强度档位（weak/moderate/strong），
    程序侧映射 δ 网格 ±(0.10/0.20/0.35)，adjacent 减半 ±(0.05/0.10/0.175)。
  - 必须引用 belief_ids（检索结果的子集）；缺引用/引用越界 → fail-closed δ=0。
  - 先缩放（系数 ∈(0,1]，v1 固定 1.0）后封顶。
  - 解析失败：带提醒重试 1 次 → 仍失败 fail-closed δ=0 + parse_error 落库。
  - placebo 负控制（评审 6A）：随机信念喂同一流程，同规则映射 δ，shadow 落库。

ASCII（δ 映射：档位 → 网格 → 缩放 → 封顶）：

  LLM {direction, strength}
        │ direction=none 或无引用 ──► δ = 0
        ▼
  |δ| = grid[strength]           # (0.10, 0.20, 0.35)
        │ gate=adjacent ──► |δ| ×= adjacent_factor (0.5)
        ▼
  δ = sign(direction) × |δ| × prior_scale     # 先缩放
        ▼
  δ = clamp(δ, -cap, +cap)                    # 后封顶；cap = grid 最大值
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field

from sqlalchemy import select

from serenity.agent.llm_client import LLMClient
from serenity.config import settings
from serenity.gate.gate import _DOMAIN_KEYWORDS, GateResult, _rule_match, load_gate_vocab
from serenity.store.dao import session_scope
from serenity.store.models import BeliefPrimitive

log = logging.getLogger(__name__)

_CONF_RANK = {"high": 2, "medium": 1, "low": 0}
EPS = 1e-6


@dataclass
class RetrievedBelief:
    id: int
    claim: str
    domain: str
    tickers: str
    stance: str
    confidence: str
    causal_links: list[str] = field(default_factory=list)


@dataclass
class PriorResult:
    delta: float  # 缩放封顶后的有符号 δ（log-odds）
    direction: str = "none"  # yes | no | none
    strength: str = "none"  # weak | moderate | strong | none
    belief_ids: list[int] = field(default_factory=list)
    rationale: str = ""
    parse_error: str | None = None
    cost_usd: float = 0.0
    retrieved_ids: list[int] = field(default_factory=list)


# ── 检索（v1：结构化字段过滤，评审定稿；embedding 后置 v2）──────────────────────


def _all_beliefs(version: str) -> list[RetrievedBelief]:
    with session_scope() as s:
        rows = s.execute(
            select(BeliefPrimitive).where(BeliefPrimitive.belief_set_version == version)
        ).scalars().all()
        return [
            RetrievedBelief(
                id=r.id, claim=r.claim, domain=r.domain, tickers=r.tickers or "",
                stance=r.stance or "neutral", confidence=r.confidence or "low",
                causal_links=json.loads(r.causal_links or "[]"),
            )
            for r in rows
        ]


def retrieve_beliefs(
    *,
    title: str,
    gate: GateResult,
    version: str,
    top_k: int | None = None,
) -> list[RetrievedBelief]:
    """按 ticker/domain 关键词过滤信念，置信度降序取 top_k。

    - in_domain：题目直接命中的 ticker / domain 关键词 → 对应信念
    - adjacent：闸门判据文本里点名的 domain → 该 domain 信念
    - 都没有 → 空（δ=0 的合法无信号路径）
    """
    top_k = top_k or settings.prior_retrieval_top_k
    beliefs = _all_beliefs(version)
    if not beliefs:
        return []

    vocab = load_gate_vocab(version)
    hit_tickers, hit_kw = _rule_match(title, vocab)
    hit_domains = {h.split(":", 1)[0] for h in hit_kw}
    if gate.state == "adjacent":
        # 闸门判据被要求点名目标 domain；从判据文本抓 domain 枚举名
        low = gate.rationale.lower()
        for d in _DOMAIN_KEYWORDS:
            if d in low:
                hit_domains.add(d)

    ticker_set = {t.upper() for t in hit_tickers}
    matched = [
        b for b in beliefs
        if (ticker_set and ticker_set & {t for t in b.tickers.split(",") if t})
        or (b.domain in hit_domains)
    ]
    matched.sort(key=lambda b: _CONF_RANK.get(b.confidence, 0), reverse=True)
    return matched[:top_k]


def sample_placebo_beliefs(
    *,
    question_id: str,
    version: str,
    k: int,
    exclude_ids: set[int],
) -> list[RetrievedBelief]:
    """placebo 负控制：按 question_id 播种的可复现随机抽样（避开真实检索命中）。"""
    beliefs = [b for b in _all_beliefs(version) if b.id not in exclude_ids]
    if not beliefs:
        return []
    rng = random.Random(f"placebo:{question_id}:{version}")
    k = min(k, len(beliefs))
    return rng.sample(beliefs, k)


# ── LLM 先验 ──────────────────────────────────────────────────────────────────

PRIOR_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["yes", "no", "none"]},
        "strength": {"type": "string", "enum": ["weak", "moderate", "strong", "none"]},
        "rationale": {
            "type": "string", "minLength": 30,
            "description": "which cited beliefs move the probability and via what causal chain",
        },
        "belief_ids": {
            "type": "array", "items": {"type": "integer"},
            "description": "ids of the beliefs (from the provided list) that support this adjustment",
        },
    },
    "required": ["direction", "strength", "rationale", "belief_ids"],
    "additionalProperties": False,
}

PRIOR_SYSTEM = """You channel the distilled belief system of "Serenity", an
AI/semiconductor supply-chain analyst. You are given (a) a market question and
(b) a numbered list of the analyst's distilled beliefs.

Decide whether these SPECIFIC beliefs give an edge over a generic forecaster
on this question, and in which direction:

- direction: "yes" if the beliefs push the probability of YES up, "no" if down,
  "none" if the beliefs don't meaningfully bear on this question.
- strength: how strongly the cited beliefs bear on the resolution criterion —
  "strong" only when a high-confidence belief directly addresses the question's
  mechanism; "weak" for indirect/one-hop relevance; "none" with direction=none.
- belief_ids: the ids of the beliefs you actually rely on. REQUIRED whenever
  direction != none. Only ids from the provided list.
- rationale: name the causal chain from the cited beliefs to this question's
  resolution criterion, in the analyst's supply-chain reasoning style
  (capex flows → bottlenecks → who benefits). Paraphrase; never quote posts.

Discipline: the beliefs are a snapshot; if the question's mechanism is outside
what they cover, say direction=none. Do NOT stretch relevance."""

PRIOR_USER_TEMPLATE = """Market question: {title}
Resolution deadline: {deadline}

Analyst beliefs (id | domain | stance/confidence | claim):
{beliefs_block}

Per the system instructions, output direction / strength / belief_ids / rationale."""


def _beliefs_block(beliefs: list[RetrievedBelief]) -> str:
    lines = []
    for b in beliefs:
        causal = f" | causal: {'; '.join(b.causal_links[:2])}" if b.causal_links else ""
        lines.append(f"[{b.id}] {b.domain} | {b.stance}/{b.confidence} | {b.claim}{causal}")
    return "\n".join(lines)


def _map_delta(direction: str, strength: str, gate_state: str) -> float:
    """档位 → δ 网格 → adjacent 减半 → 先缩放后封顶（评审 7A/8A 定稿顺序）。"""
    if direction == "none" or strength == "none":
        return 0.0
    grid = settings.prior_delta_grid_values  # (weak, moderate, strong)
    mag = {"weak": grid[0], "moderate": grid[1], "strong": grid[2]}[strength]
    if gate_state == "adjacent":
        mag *= settings.prior_adjacent_factor
    sign = 1.0 if direction == "yes" else -1.0
    scale = settings.prior_scale
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"prior_scale 必须 ∈(0,1]: {scale}")
    delta = sign * mag * scale  # 先缩放
    cap = grid[2]  # 后封顶：|δ| ≤ 网格最大值
    return max(-cap, min(cap, delta))


def _generate(
    llm: LLMClient,
    *,
    title: str,
    deadline: str | None,
    beliefs: list[RetrievedBelief],
    gate_state: str,
) -> PriorResult:
    """共用流程：真实先验与 placebo 都走这里（评审 6A：同一流程不同信念）。"""
    retrieved_ids = [b.id for b in beliefs]
    if not beliefs:
        return PriorResult(delta=0.0, rationale="no_matching_beliefs", retrieved_ids=[])

    user = PRIOR_USER_TEMPLATE.format(
        title=title, deadline=deadline or "unknown", beliefs_block=_beliefs_block(beliefs)
    )
    total_cost = 0.0
    last_err = ""
    for attempt in (1, 2):
        try:
            resp = llm.complete(
                system=PRIOR_SYSTEM,
                user=user if attempt == 1 else (
                    user + "\n\nREMINDER: respond ONLY via the tool. direction != none "
                    "REQUIRES non-empty belief_ids drawn from the provided list."
                ),
                max_tokens=800,
                response_schema=PRIOR_SCHEMA,
                estimated_input_tokens=1500,
                estimated_output_tokens=300,
            )
        except Exception as e:
            last_err = f"llm_error:{type(e).__name__}"
            log.warning("prior llm error (attempt %d): %s", attempt, e)
            continue
        total_cost += resp.cost_usd
        parsed = resp.parsed_json
        err = _validate(parsed, set(retrieved_ids))
        if err is None:
            direction = parsed["direction"]
            strength = parsed["strength"] if direction != "none" else "none"
            cited = [int(i) for i in parsed["belief_ids"]] if direction != "none" else []
            return PriorResult(
                delta=_map_delta(direction, strength, gate_state),
                direction=direction,
                strength=strength,
                belief_ids=cited,
                rationale=parsed["rationale"].strip(),
                cost_usd=total_cost,
                retrieved_ids=retrieved_ids,
            )
        last_err = err
        log.info("prior parse failed (attempt %d): %s", attempt, err)

    # fail-closed（评审 4A）：δ=0，parse_error 落库
    return PriorResult(
        delta=0.0, parse_error=last_err, cost_usd=total_cost,
        rationale=f"fail-closed：先验生成失败（{last_err}），δ=0",
        retrieved_ids=retrieved_ids,
    )


def _validate(parsed: object, allowed_ids: set[int]) -> str | None:
    if not isinstance(parsed, dict):
        return "missing_parsed_json"
    if parsed.get("direction") not in ("yes", "no", "none"):
        return "bad_direction"
    if parsed.get("strength") not in ("weak", "moderate", "strong", "none"):
        return "bad_strength"
    if not isinstance(parsed.get("rationale"), str) or len(parsed["rationale"].strip()) < 20:
        return "missing_rationale"
    ids = parsed.get("belief_ids")
    if not isinstance(ids, list):
        return "bad_belief_ids"
    if parsed["direction"] != "none":
        if not ids:
            return "missing_belief_ids"  # 引用必须（设计红线）
        try:
            if not {int(i) for i in ids} <= allowed_ids:
                return "belief_ids_out_of_range"
        except (TypeError, ValueError):
            return "bad_belief_ids"
    if parsed["direction"] != "none" and parsed["strength"] == "none":
        return "direction_without_strength"
    return None


def generate_prior(
    llm: LLMClient,
    *,
    title: str,
    deadline: str | None,
    gate: GateResult,
    version: str,
) -> PriorResult:
    beliefs = retrieve_beliefs(title=title, gate=gate, version=version)
    return _generate(llm, title=title, deadline=deadline, beliefs=beliefs, gate_state=gate.state)


def generate_placebo_prior(
    llm: LLMClient,
    *,
    question_id: str,
    title: str,
    deadline: str | None,
    gate: GateResult,
    version: str,
    real_result: PriorResult,
) -> PriorResult:
    """placebo 臂：随机信念（避开真实检索命中）喂同一流程。k 对齐真实检索规模。"""
    k = max(len(real_result.retrieved_ids), 4)
    beliefs = sample_placebo_beliefs(
        question_id=question_id, version=version, k=k,
        exclude_ids=set(real_result.retrieved_ids),
    )
    return _generate(llm, title=title, deadline=deadline, beliefs=beliefs, gate_state=gate.state)


# ── δ 应用 ────────────────────────────────────────────────────────────────────


def apply_delta(prob: float, delta: float) -> float:
    """sigmoid(logit(p) + δ)，两端裁剪防 0/1。"""
    p = min(1.0 - EPS, max(EPS, float(prob)))
    logit = math.log(p / (1.0 - p)) + float(delta)
    out = 1.0 / (1.0 + math.exp(-logit))
    return min(1.0 - EPS, max(EPS, out))
