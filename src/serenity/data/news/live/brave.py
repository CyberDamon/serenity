"""Brave Search API adapter — second-tier alternative to Tavily.

Cheaper than Tavily ($3-5 per 1k queries) but less news-optimized. We use Brave
when Tavily is rate-limited or for cross-source diversity.

API: GET https://api.search.brave.com/res/v1/news/search?q=...&count=10
Header: X-Subscription-Token: $BRAVE_API_KEY
"""

from __future__ import annotations

import http.client
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from serenity.config import settings
from serenity.data.news.types import NewsItem

log = logging.getLogger(__name__)


class BraveTransientError(RuntimeError):
    pass


class BraveNotConfigured(RuntimeError):
    pass


@retry(
    retry=retry_if_exception_type(BraveTransientError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    reraise=True,
)
def search(query: str, max_results: int = 8) -> list[NewsItem]:
    api_key = settings.brave_api_key.get_secret_value()
    if not api_key or api_key.strip().upper() in {"REPLACE_ME", "PLACEHOLDER", "TODO"}:
        raise BraveNotConfigured("BRAVE_API_KEY not set")

    url = "https://api.search.brave.com/res/v1/news/search?" + urllib.parse.urlencode(
        {"q": query, "count": max_results, "freshness": "pw"}
    )
    req = urllib.request.Request(
        url,
        headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", errors="replace")
        msg = f"Brave {e.code}: {e.reason} {body}".strip()
        if 500 <= e.code < 600 or e.code == 429:
            raise BraveTransientError(msg) from e
        raise BraveNotConfigured(msg) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise BraveTransientError(str(e)) from e
    except (http.client.IncompleteRead, http.client.HTTPException, ConnectionError) as e:
        # Connection dropped mid-body — protocol error from resp.read(), not a
        # URLError subclass, so it would otherwise escape unretried.
        raise BraveTransientError(f"Brave truncated response: {e!r}") from e

    items: list[NewsItem] = []
    for r in data.get("results") or []:
        # r.get("age") 形如 '2 hours ago'，太模糊 → 回退 now（Codex 已记为精修点）
        published_at = datetime.now(UTC)
        items.append(
            NewsItem(
                source="brave",
                url=r.get("url", ""),
                title=r.get("title", ""),
                published_at=published_at,
                summary=(r.get("description") or "")[:600],
            )
        )
    return items
