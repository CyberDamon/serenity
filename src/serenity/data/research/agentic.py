"""主动 agentic 检索（L2：逐点复刻 Forecasting_Bot 的 agentic_search 多轮循环）。

对齐参考实现 Bot/search.py 的 agentic_search：
  round0  INITIAL   ：分析问题(不作答) → 产出搜索词
  roundN  CONTINUATION：给上轮分析 + 新搜索结果 → 写完整分析 + 发现信息缺口→续搜；
                        LLM 判定信息足够(need_more=false)或无新词 → 停；最多 max_rounds 轮
后端用 SearX 搜索 + `/extract` 抓原文（2026-07-07 起替代 Tavily——后者套餐用量
超限、432 fail-closed，见 search-service.md），把参考实现的
Serper 搜 + Bright Data 抓全文 + o3 逐篇摘要合并成一步。

产出：(synthesis 分析文本, 去重后的源文章 NewsItem 列表)。二者都喂给框架：
synthesis 作为首条"研究简报"NewsItem，源文章经 pre_filter 打事件标签。

降级：SearX 无 token/失败 → 退回 redline 被动 RSS 瀑布（assemble_news_context）。
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from serenity.agent.llm_client import LLMClient
from serenity.config import settings
from serenity.data.news.live import assemble_news_context, searx
from serenity.data.news.pre_filter import tag_news
from serenity.data.news.types import NewsItem

log = logging.getLogger(__name__)

# 每轮 LLM 的结构化输出（用 schema 替代参考实现的正则解析，更稳）。
_ROUND_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "对该预测问题的完整、基于已见搜索结果的分析（首轮可为对所需信息的理解）。",
        },
        "search_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "为填补信息缺口而发起的关键词搜索词（≤6 词/条，无引号）；信息已足够则留空。",
        },
        "need_more": {
            "type": "boolean",
            "description": "是否还需要更多检索。false = 分析已足够、停止。",
        },
    },
    "required": ["analysis", "search_queries", "need_more"],
    "additionalProperties": False,
}

_INITIAL_PROMPT = """You are a research assistant for a superforecaster. You do NOT forecast;
you gather and synthesize evidence.

Forecasting question:
{question}
As of: {as_of}

This is ROUND 1. Do NOT answer yet. Analyze what needs to be researched, then produce
targeted keyword search queries covering BOTH:
  - historical base rate / precedent for this kind of event, and
  - the latest developments / structural changes bearing on it.
Queries must be keyword-optimized (≤6 words, no quotes). Set need_more=true.
Return JSON matching the schema (analysis = your understanding of what to research)."""

_CONTINUATION_PROMPT = """You are continuing research for a superforecaster (round {round_n}).

Forecasting question:
{question}
As of: {as_of}

Your analysis so far:
{prev_analysis}

Queries already used: {used}

New search results this round:
{results}

Write a COMPLETE, up-to-date analysis grounded in the search results (cite sources inline
like "per BBC"). Distinguish fact from speculation; note recency. If material gaps remain,
add NEW keyword queries (different from used) and set need_more=true; if the evidence is
sufficient, leave search_queries empty and set need_more=false. Return JSON per schema."""


def _searx_search(query: str, *, max_results: int = 5, days_back: int = 21) -> list[NewsItem]:
    """一次 SearX 搜索 + 批量 `/extract` 抓原文（Tavily advanced+raw_content 的等价物）。

    source 用文章域名（而非统一 "searx"），否则证据门的来源多样性会把所有结果当成
    1 个来源 → 误判 evidence_too_thin（沿用 QA-001 的修法）。
    """
    time_range = "month" if days_back <= 31 else "year"
    items = searx.search(query, max_results=max_results, time_range=time_range)
    if not items:
        return items
    try:
        full_text = searx.extract_many([it.url for it in items if it.url], max_chars=4000)
    except Exception as e:
        log.warning("searx extract_many failed for %r: %s", query, e)
        full_text = {}
    for it in items:
        text = full_text.get(it.url)
        if text:
            it.summary = text[:4000]
    return items


def agentic_research(
    question: str,
    *,
    llm: LLMClient,
    as_of_date: str,
    max_rounds: int = 5,
    queries_per_round: int = 4,
    max_results_per_query: int = 4,
) -> tuple[str, list[NewsItem]]:
    """多轮 agentic 检索。返回 (synthesis 分析, 源文章 NewsItem 列表)。"""
    analysis = ""
    used: list[str] = []
    seen_urls: set[str] = set()
    sources: list[NewsItem] = []
    round_results_block = "(none yet)"

    for rnd in range(max_rounds):
        if rnd == 0:
            prompt = _INITIAL_PROMPT.format(question=question, as_of=as_of_date)
        else:
            prompt = _CONTINUATION_PROMPT.format(
                round_n=rnd + 1, question=question, as_of=as_of_date,
                prev_analysis=analysis or "(none)", used=", ".join(used) or "(none)",
                results=round_results_block,
            )
        try:
            resp = llm.complete(
                system="You output only JSON. You are an evidence-gathering research assistant.",
                user=prompt, max_tokens=1600, response_schema=_ROUND_SCHEMA,
                estimated_input_tokens=1500, estimated_output_tokens=800,
            )
            parsed = resp.parsed_json or (json.loads(resp.text) if resp.text else None)
        except Exception as e:
            log.warning("agentic round %d LLM failed: %s", rnd, e)
            break
        if not isinstance(parsed, dict):
            break

        if parsed.get("analysis"):
            analysis = str(parsed["analysis"])

        fresh = [str(q).strip() for q in (parsed.get("search_queries") or [])
                 if str(q).strip() and str(q).strip() not in used][:queries_per_round]
        if not parsed.get("need_more", True) or not fresh:
            break  # LLM 判定信息足够，或无新查询

        used.extend(fresh)
        log.info("agentic round %d queries: %s", rnd + 1, fresh)

        # 并发跑本轮查询
        def _one(q):
            try:
                return _searx_search(q, max_results=max_results_per_query)
            except Exception as e:
                log.warning("searx search failed for %r: %s", q, e)
                return []
        with ThreadPoolExecutor(max_workers=min(4, len(fresh))) as ex:
            round_results = [r for batch in ex.map(_one, fresh) for r in batch]

        # 去重累积 + 构造本轮结果 block 供续轮 LLM 阅读
        block_lines = []
        for item in round_results:
            if not item.url or item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            sources.append(item)
            block_lines.append(f"[{item.title[:80]}] {item.summary[:500]}")
        round_results_block = "\n\n".join(block_lines) or "(no new results)"

    tag_news(sources)
    return analysis, sources, used


def _research_log(backend: str, queries: list[str], sources: list[NewsItem], brief: str) -> dict:
    """可审计的检索记录（落库/日志用）。"""
    return {
        "backend": backend,
        "queries": queries,
        "n_sources": len(sources),
        "sources": [{"url": s.url, "title": s.title, "source": s.source} for s in sources[:20]],
        "brief": brief[:2000] if brief else "",
    }


def assemble_research(
    question: str,
    *,
    llm: LLMClient | None = None,
    as_of_date: str = "",
    top_k: int = 8,
) -> tuple[list[NewsItem], dict]:
    """runner 入口。SearX 有 token 且有 llm → 多轮 agentic；否则退回 RSS 瀑布。

    返回 (NewsItem 列表, 检索审计 dict)。列表首条是 synthesis「研究简报」，其余为源文章。
    审计 dict 供 runner 落库，事后可追溯"这题搜了什么词、拿到哪些源"。
    """
    if llm is not None and settings.searx_token.get_secret_value():
        analysis, sources, queries = agentic_research(question, llm=llm, as_of_date=as_of_date)
        items: list[NewsItem] = []
        if analysis:
            items.append(NewsItem(
                source="agentic:brief", url="", title=f"研究简报: {question[:60]}",
                published_at=datetime.now(UTC), summary=analysis[:2000],
            ))
        items.extend(sources[:top_k])
        log.info("agentic research: %d 源文章 + %s简报 (queries=%s)",
                 len(sources), "1 " if analysis else "0 ", queries)
        return items, _research_log("searx", queries, sources, analysis)

    # 降级：无 SearX token → redline 被动 RSS 瀑布
    log.info("agentic research 降级 RSS（SEARX_TOKEN 未配）")
    items = assemble_news_context(question, top_k=top_k)
    tag_news(items)
    return items, _research_log("rss", [], items, "")
