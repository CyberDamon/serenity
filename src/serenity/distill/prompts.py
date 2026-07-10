"""蒸馏提示词与输出 schema。

两阶段：
  1) extract  —— 按时间序分批（~40 条/批），抽信念原语 / 标的论点 / 历史 call
  2) consolidate —— 按 domain 合并近重复信念（跨批去重）

抽取纪律（评审 8A）：
  - historical_claims 只收"当时向前看"的判断（made_at = 最早来源推文日期）；
    回顾/庆祝/事后修正（"我早说过…"）一律不算 call。
  - 所有 claim 都是转述（paraphrase），不复制推文原文（评审 9A）。
"""

from __future__ import annotations

DOMAINS = [
    "optics_cpo",           # 光模块 / CPO / 光子学
    "inp_compound_semis",   # InP 衬底 / 化合物半导体
    "memory_hbm_nand",      # 存储 / HBM / NAND
    "neocloud_financing",   # neocloud 融资质量
    "ai_power_grid",        # AI 电力 / 电网需求
    "robotics_physical_ai", # 机器人 / physical AI
    "semis_supply_chain",   # 半导体供应链（capex 传导、代工、设备）
    "ai_models_labs",       # AI 模型 / 实验室动态
    "macro_market",         # 宏观 / 市场结构
    "other",
]

EXTRACT_SYSTEM = """You are a research analyst distilling the public posts of
"Serenity" (@aleabitoreddit), an AI/semiconductor supply-chain analyst, into a
structured belief base. You will receive a chronological batch of their posts
(each with id + date + text).

Extract THREE kinds of items:

1. belief_primitives — recurring worldview claims: causal models, structural
   theses, supply/demand mechanics. PARAPHRASE in your own words (one sentence,
   English). Do NOT copy post text verbatim. Example paraphrases:
   "Hyperscaler capex growth flows disproportionately into optical
   interconnect, making 800G/1.6T transceiver suppliers the bottleneck."

2. ticker_theses — per-ticker investment theses actually argued in the posts
   (ticker must appear in the posts; do not invent).

3. historical_claims — FORWARD-LOOKING calls with a direction and an implicit
   or explicit time window, AS STATED AT POST TIME. STRICT RULES:
   - Only include claims that were predictions when written ("X will happen",
     "expect Y by Q3"). EXCLUDE victory laps, retrospectives, "as I said
     before" posts, and any post-hoc framing.
   - made_at MUST be the date of the earliest post making the claim.
   - PARAPHRASE the claim; keep it falsifiable ("InP substrate demand will
     exceed supply through H1 2026").

Quality bar: fewer, sharper items beat many vague ones. Skip pure jokes,
replies with no analytical content, and engagement posts. Confidence reflects
how strongly/repeatedly the author holds the view IN THIS BATCH.

Output JSON via the tool per schema. Use domain values from:
{domains}
"""

EXTRACT_USER_TEMPLATE = """Posts batch ({n} posts, {date_from} .. {date_to}):

{posts_block}

Extract belief_primitives / ticker_theses / historical_claims per the system
instructions. Every item must cite the source post ids in tweet_ids."""

EXTRACT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "belief_primitives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "minLength": 20},
                    "domain": {"type": "string"},
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "stance": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "causal_links": {"type": "array", "items": {"type": "string"}},
                    "tweet_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["claim", "domain", "stance", "confidence", "tweet_ids"],
                "additionalProperties": False,
            },
        },
        "ticker_theses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "subsector": {"type": "string"},
                    "thesis": {"type": "string", "minLength": 20},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "tweet_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["ticker", "thesis", "confidence", "tweet_ids"],
                "additionalProperties": False,
            },
        },
        "historical_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "minLength": 20},
                    "direction": {"type": "string", "enum": ["yes", "no"]},
                    "made_at": {"type": "string", "description": "ISO date of earliest source post"},
                    "horizon": {"type": "string", "description": "outcome window, e.g. 'by Q3 2026'"},
                    "tweet_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["claim", "direction", "made_at", "tweet_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["belief_primitives", "ticker_theses", "historical_claims"],
    "additionalProperties": False,
}

CONSOLIDATE_SYSTEM = """You merge near-duplicate belief claims extracted from
different batches of the same author's posts. Given a list of claims in ONE
domain (each with an index), group claims that express the SAME underlying
thesis and produce one merged claim per group.

Rules:
- Merged claim = the sharpest phrasing (paraphrase, one sentence).
- stance/confidence = the majority of the group (confidence upgraded one level
  if the same thesis recurs in ≥3 distinct claims — repetition = conviction).
- Keep genuinely distinct theses separate. Do not over-merge.
- member_indexes must cover which input claims were merged into each output.

Output JSON via the tool per schema."""

CONSOLIDATE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "merged": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "minLength": 20},
                    "stance": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "causal_links": {"type": "array", "items": {"type": "string"}},
                    "member_indexes": {
                        "type": "array", "items": {"type": "integer"}, "minItems": 1,
                    },
                },
                "required": ["claim", "stance", "confidence", "member_indexes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["merged"],
    "additionalProperties": False,
}
