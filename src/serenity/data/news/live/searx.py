"""SearX metasearch adapter — shared self-hosted service on the Yarrow box.

Free, no per-query cost, no hard rate limit (unlike Tavily's metered plan).
Two endpoints behind one bearer token: `/__searx/search` (snippets) and
`/__extract/extract` (full article text via trafilatura, batchable ≤20 urls).
See ../../../../../search-service.md for the full contract.
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


class SearxTransientError(RuntimeError):
    pass


class SearxNotConfigured(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    token = settings.searx_token.get_secret_value()
    if not token:
        raise SearxNotConfigured("SEARX_TOKEN not set")
    return {"Authorization": f"Bearer {token}"}


def _request(method: str, url: str, *, headers: dict[str, str], data: bytes | None, timeout: int):
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", errors="replace")
        msg = f"SearX {e.code}: {e.reason} {body}".strip()
        if e.code == 403:
            raise SearxNotConfigured(msg) from e
        if 500 <= e.code < 600 or e.code == 429:
            raise SearxTransientError(msg) from e
        raise SearxTransientError(msg) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise SearxTransientError(str(e)) from e
    except (http.client.IncompleteRead, http.client.HTTPException, ConnectionError) as e:
        raise SearxTransientError(f"SearX truncated response: {e!r}") from e


@retry(
    retry=retry_if_exception_type(SearxTransientError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=10),
    reraise=True,
)
def search(
    query: str,
    max_results: int = 8,
    *,
    categories: str = "news,general",
    time_range: str | None = None,
) -> list[NewsItem]:
    """Run a SearX search. Returns NewsItem list (possibly empty)."""
    headers = _headers()
    params = {"q": query, "format": "json", "categories": categories, "language": "en"}
    if time_range:
        params["time_range"] = time_range
    url = f"{settings.searx_base_url}/__searx/search?" + urllib.parse.urlencode(params)
    data = _request("GET", url, headers=headers, data=None, timeout=20)

    items: list[NewsItem] = []
    for r in (data.get("results") or [])[:max_results]:
        published_raw = r.get("publishedDate")
        try:
            published_at = (
                datetime.fromisoformat(str(published_raw).replace("Z", "+00:00"))
                if published_raw
                else datetime.now(UTC)
            )
        except ValueError:
            published_at = datetime.now(UTC)
        netloc = urllib.parse.urlparse(r.get("url") or "").netloc.lower().removeprefix("www.")
        items.append(
            NewsItem(
                source=netloc or "searx",
                url=r.get("url", ""),
                title=r.get("title", ""),
                published_at=published_at,
                summary=(r.get("content") or "")[:600],
            )
        )
    return items


@retry(
    retry=retry_if_exception_type(SearxTransientError),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=2, min=1, max=6),
    reraise=True,
)
def extract_many(urls: list[str], *, max_chars: int = 4000) -> dict[str, str]:
    """Batch-extract full article text for up to 20 urls. Returns {url: text} (misses omitted)."""
    if not urls:
        return {}
    headers = {**_headers(), "Content-Type": "application/json"}
    url = f"{settings.searx_base_url}/__extract/extract"
    body = json.dumps({"urls": urls[:20], "max_chars": max_chars}).encode("utf-8")
    data = _request("POST", url, headers=headers, data=body, timeout=90)
    out: dict[str, str] = {}
    for r in data.get("results") or []:
        if r.get("ok") and r.get("text"):
            out[r.get("url", "")] = r["text"]
    return out
