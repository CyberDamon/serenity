"""三态领域闸门：in_domain | adjacent | out_of_domain。

判定标准（设计文档定稿 + 评审 4A fail-closed）：

  规则层（确定性，先走，零成本）：
    题目文本直接命中信念库任一 ticker 或 domain 词表 ──► in_domain
  LLM 层（规则未命中时裁决 adjacent / out）：
    仅经因果链一跳或宏观外溢关联 ──► adjacent
    其余 ──► out_of_domain
  解析失败：带格式提醒重试 1 次 → 仍失败 fail-closed 为 out_of_domain
             （parse_error 落库，单题隔离）

ASCII 判定树：

  question.title
      │  ticker/domain 词表直接命中？
      ├── 是 ──► in_domain（rationale = 命中词列表）
      └── 否 ──► LLM 裁决 {adjacent|out} + 判据文本
                    │ 解析失败 → 重试 1 次
                    └── 仍失败 ──► out_of_domain (parse_error)

判据文本（rationale）必须非空落库，供每周人工抽标复盘（滚动 4 周 n 评估）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import select

from serenity.agent.llm_client import LLMClient
from serenity.distill.prompts import DOMAINS
from serenity.store.dao import session_scope
from serenity.store.models import BeliefPrimitive, TickerKnowledge

log = logging.getLogger(__name__)

# domain 词表：domain 标签 → 题目文本里的可匹配关键词（小写）。
# 与 distill.prompts.DOMAINS 对应；只放高精度词，宁漏勿滥（漏的交给 LLM 层）。
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "optics_cpo": ["optical transceiver", "co-packaged optics", "cpo", "photonics",
                   "800g", "1.6t", "optical interconnect"],
    "inp_compound_semis": ["inp", "indium phosphide", "compound semiconductor", "gaas",
                           "silicon carbide", "gan "],
    "memory_hbm_nand": ["hbm", "high bandwidth memory", "nand", "dram", "memory chip"],
    "neocloud_financing": ["neocloud", "coreweave", "gpu cloud", "ai cloud provider"],
    "ai_power_grid": ["data center power", "datacenter power", "grid", "power demand",
                      "nuclear power", "smr", "electricity demand"],
    "robotics_physical_ai": ["humanoid robot", "robotics", "physical ai"],
    "semis_supply_chain": ["semiconductor", "chip fab", "foundry", "wafer", "capex",
                           "tsmc", "lithography", "euv", "chip export", "export control"],
    "ai_models_labs": ["openai", "anthropic", "gpt-", "claude", "gemini", "frontier model",
                       "ai model", "llm", "artificial intelligence", " agi"],
    "macro_market": [],  # 宏观词不做规则直判（太宽，交给 LLM 层判 adjacent）
}

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9.\-]*")


@dataclass
class GateVocab:
    tickers: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)


def load_gate_vocab(version: str) -> GateVocab:
    """从信念库载入该版本的 ticker 与 domain 集合。"""
    vocab = GateVocab()
    with session_scope() as s:
        for (tickers,) in s.execute(
            select(BeliefPrimitive.tickers).where(BeliefPrimitive.belief_set_version == version)
        ):
            for t in (tickers or "").split(","):
                if t.strip():
                    vocab.tickers.add(t.strip().upper())
        for (d,) in s.execute(
            select(BeliefPrimitive.domain).where(BeliefPrimitive.belief_set_version == version).distinct()
        ):
            vocab.domains.add(d)
        for (tk,) in s.execute(
            select(TickerKnowledge.ticker).where(TickerKnowledge.belief_set_version == version)
        ):
            vocab.tickers.add(tk.upper())
    return vocab


@dataclass
class GateResult:
    state: str  # in_domain | adjacent | out_of_domain
    rationale: str  # 判据文本（必须非空）
    matched_terms: list[str] = field(default_factory=list)
    parse_error: str | None = None
    cost_usd: float = 0.0


def _rule_match(title: str, vocab: GateVocab) -> tuple[list[str], list[str]]:
    """返回 (命中的 ticker, 命中的 domain 关键词)。"""
    low = f" {title.lower()} "
    words = set(_WORD_RE.findall(low))
    hit_tickers = sorted(
        t for t in vocab.tickers
        if len(t) >= 2 and t.lower() in words  # 全词匹配防 "A"/"AI" 之类误击
    )
    hit_kw: list[str] = []
    for domain in vocab.domains or set(_DOMAIN_KEYWORDS):
        for kw in _DOMAIN_KEYWORDS.get(domain, []):
            if kw in low:
                hit_kw.append(f"{domain}:{kw.strip()}")
    return hit_tickers, hit_kw


GATE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "state": {"type": "string", "enum": ["adjacent", "out_of_domain"]},
        "rationale": {
            "type": "string", "minLength": 20,
            "description": "判据：若 adjacent，写明经哪条因果链一跳关联到哪个 domain",
        },
    },
    "required": ["state", "rationale"],
    "additionalProperties": False,
}

GATE_SYSTEM = """You are the domain gate for a forecasting agent whose expert
belief base covers ONLY these domains (AI/semiconductor supply chain analyst):
{domains}

The question did NOT directly mention any covered ticker or domain keyword
(rule layer already checked). Decide between:

- "adjacent": the question connects to a covered domain via ONE causal hop or
  a macro spillover the analyst's beliefs plausibly inform. Example: a Fed rate
  decision affects neocloud financing costs → adjacent. You MUST name the
  specific causal chain and target domain in rationale.
- "out_of_domain": everything else (politics, sports, entertainment, weather,
  crypto price action, general elections, ...). When in doubt, choose
  out_of_domain — the agent abstains rather than pretends expertise.

Output JSON via the tool."""

GATE_USER_TEMPLATE = """Question: {title}
Resolution deadline: {deadline}

Classify adjacent vs out_of_domain per the system instructions."""


def classify_question(
    *,
    title: str,
    deadline: str | None,
    vocab: GateVocab,
    llm: LLMClient,
) -> GateResult:
    """三态判定。规则层直判 in_domain；LLM 层裁决 adjacent/out；fail-closed。"""
    hit_tickers, hit_kw = _rule_match(title, vocab)
    if hit_tickers or hit_kw:
        terms = hit_tickers + hit_kw
        return GateResult(
            state="in_domain",
            rationale=f"规则直判：题目直接命中 {', '.join(terms[:8])}",
            matched_terms=terms,
        )

    system = GATE_SYSTEM.format(domains=", ".join(DOMAINS))
    user = GATE_USER_TEMPLATE.format(title=title, deadline=deadline or "unknown")
    total_cost = 0.0
    last_err = ""
    for attempt in (1, 2):
        try:
            resp = llm.complete(
                system=system,
                user=user if attempt == 1 else (
                    user + "\n\nREMINDER: respond ONLY via the tool with fields "
                    "`state` (adjacent|out_of_domain) and `rationale` (>=20 chars)."
                ),
                max_tokens=400,
                response_schema=GATE_SCHEMA,
                estimated_input_tokens=600,
                estimated_output_tokens=150,
            )
        except Exception as e:
            last_err = f"llm_error:{type(e).__name__}"
            log.warning("gate llm error (attempt %d): %s", attempt, e)
            continue
        total_cost += resp.cost_usd
        parsed = resp.parsed_json
        if (
            isinstance(parsed, dict)
            and parsed.get("state") in ("adjacent", "out_of_domain")
            and isinstance(parsed.get("rationale"), str)
            and len(parsed["rationale"].strip()) >= 10
        ):
            return GateResult(
                state=parsed["state"],
                rationale=parsed["rationale"].strip(),
                cost_usd=total_cost,
            )
        last_err = "gate_parse_error"
        log.info("gate parse failed (attempt %d) on %r", attempt, title[:60])

    # fail-closed（评审 4A）：解析/调用两次失败 → out_of_domain 弃权
    return GateResult(
        state="out_of_domain",
        rationale=f"fail-closed：LLM 裁决失败（{last_err}），按 out_of_domain 弃权",
        parse_error=last_err,
        cost_usd=total_cost,
    )
