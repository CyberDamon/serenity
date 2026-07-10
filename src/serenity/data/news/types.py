"""Shared news article representation across all sources.

Both live fetchers (Tavily/Brave/RSS) and archive (GDELT/Wayback/LLM web_search)
produce NewsItem so downstream retrieval/aggregation code is source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class NewsItem:
    """One news article with timestamp + provenance."""

    source: str  # e.g. 'tavily', 'gdelt', 'brave', 'rss:reuters', 'wayback'
    url: str
    title: str
    published_at: datetime  # timezone-aware UTC
    summary: str = ""  # 200-400 char excerpt for prompt
    text: str = ""  # full text when available; expensive to fetch — usually empty
    tags: list[str] = field(default_factory=list)  # pre_filter labels, e.g. ['securitization_signal']

    def excerpt(self, max_chars: int = 400) -> str:
        body = self.summary or self.text or self.title
        if len(body) <= max_chars:
            return body
        return body[: max_chars - 3] + "..."
