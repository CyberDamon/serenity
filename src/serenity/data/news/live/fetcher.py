"""Unified live-news retrieval: SearX → Tavily → Brave → RSS waterfall + BM25 scoring.

Strategy (v0):
  1. SearX search (shared self-hosted metasearch, free/no hard limit) → primary
  2. Tavily search same query → secondary if configured (paid, metered)
  3. Brave search same query → tertiary if Tavily missing/rate-limited
  4. RSS feeds with BM25 over titles → cheap fallback / supplementary breadth

The fetcher returns top-K (default 8) most-relevant items for `market.question`
ranked by recency x keyword-match.

For BACKTEST as_of_date queries see serenity.data.news.archive (different module).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from serenity.data.news.live import brave, rss, searx, tavily
from serenity.data.news.live.brave import BraveNotConfigured, BraveTransientError
from serenity.data.news.live.searx import SearxNotConfigured, SearxTransientError
from serenity.data.news.live.tavily import TavilyNotConfigured, TavilyTransientError
from serenity.data.news.types import NewsItem

log = logging.getLogger(__name__)


_TOKENIZE = re.compile(r"\w+", re.UNICODE)


@dataclass
class NewsContextDiagnostics:
    """Quality diagnostics for one assembled news context."""

    context_size: int
    source_count: int
    newest_age_hours: float | None
    query_token_coverage: float
    is_thin: bool
    is_stale: bool


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKENIZE.findall(text)]


def _score_recency(item: NewsItem, *, now: datetime, half_life_hours: float = 168.0) -> float:
    """Exponential decay: full weight at publish, 0.5 weight at half_life_hours.

    168h = 7-day half-life. Geopolitical events unfold over weeks so we need
    older relevant articles to retain weight vs 48h (which was too aggressive).
    """
    age = (now - item.published_at).total_seconds() / 3600.0
    if age < 0:
        age = 0
    return math.exp(-math.log(2) * age / half_life_hours)


def _score_keyword_match(item: NewsItem, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    body = f"{item.title} {item.summary}"
    item_tokens = set(_tokenize(body))
    overlap = query_tokens & item_tokens
    return len(overlap) / len(query_tokens)


def assemble_news_context(
    query: str,
    *,
    top_k: int = 8,
    now: datetime | None = None,
    use_searx: bool = True,
    use_tavily: bool = True,
    use_brave: bool = True,
    use_rss: bool = True,
    extra_queries: tuple[str, ...] = (),
) -> list[NewsItem]:
    """Pull news from configured sources and return top-K relevant items.

    Args:
      query: anchor query (usually market.question). Always issued.
      top_k: how many items to return (default 8 — matches v2.3 design)
      now:   "current time" reference for recency scoring. Defaults to UTC now.
      use_*: enable/disable individual sources (for tests)
      extra_queries: V6.1 — additional search terms covering YES/NO/neutral
                    angles. Each query goes through Tavily + Brave with reduced
                    per-query budget so total API spend stays bounded. RSS pool
                    is shared (only fetched once). All items merge into one
                    BM25-scored pool against the anchor query.
    """
    if now is None:
        now = datetime.now(UTC)
    pool: list[NewsItem] = []

    # Build the list of queries to issue. Anchor query first; dedupe later by url.
    queries = (query, *extra_queries)
    # Reduce per-query budget when many queries to bound API spend
    per_query_max = max(4, 10 // max(1, len(queries) - 1)) if len(queries) > 1 else 10
    per_query_days = 14

    if use_searx:
        for q in queries:
            try:
                pool.extend(searx.search(q, max_results=per_query_max, time_range="month"))
            except SearxNotConfigured as e:
                log.debug("SearX not configured — skipping: %s", e)
                break
            except SearxTransientError as e:
                log.warning("SearX transient failure on %r: %s", q[:40], e)

    if use_tavily:
        for q in queries:
            try:
                pool.extend(tavily.search(q, max_results=per_query_max, days_back=per_query_days))
            except TavilyNotConfigured as e:
                log.warning("Tavily unavailable — skipping for this context: %s", e)
                break
            except TavilyTransientError as e:
                log.warning("Tavily transient failure on %r: %s", q[:40], e)

    if use_brave:
        for q in queries:
            try:
                pool.extend(brave.search(q, max_results=per_query_max))
            except BraveNotConfigured as e:
                log.debug("Brave not configured — skipping: %s", e)
                break
            except BraveTransientError as e:
                log.warning("Brave transient failure on %r: %s", q[:40], e)

    if use_rss:
        # RSS is a broad pool; we filter aggressively by BM25 below.
        rss_items = rss.fetch_all_feeds(max_items_per_feed=20)
        # Pre-filter to last 14 days to keep ranking cheap
        cutoff = now - timedelta(days=14)
        pool.extend(i for i in rss_items if i.published_at >= cutoff)

    if not pool:
        return []

    # Score against anchor query only — extra_queries widen recall, anchor
    # decides ranking. Items matching multiple queries get higher score via
    # the "all query tokens combined" set below.
    combined_tokens = set(_tokenize(query))
    for q in extra_queries:
        combined_tokens |= set(_tokenize(q))
    query_tokens = combined_tokens
    # Stopword pruning — tiny list, fine for v0
    stopwords = {
        "will",
        "the",
        "a",
        "an",
        "is",
        "are",
        "be",
        "by",
        "of",
        "to",
        "in",
        "on",
        "and",
        "or",
        "before",
        "after",
        "during",
        "this",
        "that",
    }
    query_tokens -= stopwords

    scored: list[tuple[float, NewsItem]] = []
    for item in pool:
        kw = _score_keyword_match(item, query_tokens)
        rec = _score_recency(item, now=now)
        # Equal-weight kw + rec; tune in V1 based on backtest signal
        score = 0.5 * kw + 0.5 * rec
        scored.append((score, item))

    # Dedupe by url, keep highest score
    by_url: dict[str, tuple[float, NewsItem]] = {}
    for s, it in scored:
        if not it.url:
            continue
        if it.url not in by_url or s > by_url[it.url][0]:
            by_url[it.url] = (s, it)

    ranked = sorted(by_url.values(), key=lambda t: t[0], reverse=True)
    return [item for _, item in ranked[:top_k]]


def diagnose_news_context(
    query: str,
    items: list[NewsItem],
    *,
    now: datetime | None = None,
    min_items: int = 3,
    stale_after_hours: float = 72.0,
) -> NewsContextDiagnostics:
    """Compute cheap retrieval-quality diagnostics for dashboard/CLI reporting."""
    if now is None:
        now = datetime.now(UTC)
    if not items:
        return NewsContextDiagnostics(
            context_size=0,
            source_count=0,
            newest_age_hours=None,
            query_token_coverage=0.0,
            is_thin=True,
            is_stale=True,
        )

    newest = max(item.published_at for item in items)
    newest_age_hours = max(0.0, (now - newest).total_seconds() / 3600.0)
    sources = {item.source for item in items if item.source}
    query_tokens = set(_tokenize(query)) - {
        "will", "the", "a", "an", "is", "are", "be", "by", "of", "to", "in", "on"
    }
    context_tokens = set(_tokenize(" ".join(f"{i.title} {i.summary}" for i in items)))
    coverage = (len(query_tokens & context_tokens) / len(query_tokens)) if query_tokens else 0.0
    return NewsContextDiagnostics(
        context_size=len(items),
        source_count=len(sources),
        newest_age_hours=newest_age_hours,
        query_token_coverage=coverage,
        is_thin=len(items) < min_items,
        is_stale=newest_age_hours > stale_after_hours,
    )
