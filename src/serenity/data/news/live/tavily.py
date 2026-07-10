"""Tavily web search adapter.

Tavily is a paid LLM-friendly search API ($0.01-0.05 per call). We use it for
forward news retrieval (live market predictions). Costs tracked under
DAILY_SEARCH_COST_CAP_USD (separate from LLM cap).

API: POST https://api.tavily.com/search  with {query, search_depth, max_results, ...}
"""

from __future__ import annotations

import http.client
import json
import logging
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from serenity.config import settings
from serenity.data.news.types import NewsItem

log = logging.getLogger(__name__)


class TavilyTransientError(RuntimeError):
    pass


class TavilyNotConfigured(RuntimeError):
    pass


@retry(
    retry=retry_if_exception_type(TavilyTransientError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    reraise=True,
)
def search(query: str, max_results: int = 8, days_back: int = 7) -> list[NewsItem]:
    """Run a Tavily news search. Returns NewsItem list (possibly empty)."""
    api_key = settings.tavily_api_key.get_secret_value()
    if not api_key:
        raise TavilyNotConfigured("TAVILY_API_KEY not set")

    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",  # 'advanced' is 2x cost; v0 stays cheap
        "topic": "news",
        "max_results": max_results,
        "days": days_back,
        "include_answer": False,
        "include_raw_content": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", errors="replace")
        msg = f"Tavily {e.code}: {e.reason} {body}".strip()
        if 500 <= e.code < 600 or e.code == 429:
            raise TavilyTransientError(msg) from e
        raise TavilyNotConfigured(msg) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise TavilyTransientError(str(e)) from e
    except (http.client.IncompleteRead, http.client.HTTPException, ConnectionError) as e:
        # Connection dropped mid-body — protocol error from resp.read(), not a
        # URLError subclass, so it would otherwise escape unretried.
        raise TavilyTransientError(f"Tavily truncated response: {e!r}") from e

    items: list[NewsItem] = []
    for r in data.get("results", []):
        published = r.get("published_date")
        try:
            published_at = (
                datetime.fromisoformat(published.replace("Z", "+00:00"))
                if published
                else datetime.now(UTC)
            )
        except (ValueError, AttributeError):
            published_at = datetime.now(UTC)
        items.append(
            NewsItem(
                source="tavily",
                url=r.get("url", ""),
                title=r.get("title", ""),
                published_at=published_at,
                summary=r.get("content", "")[:600],
            )
        )
    return items
