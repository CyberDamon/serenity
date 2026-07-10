"""RSS-feed news retrieval — free, broad geopolitics coverage.

V2 (2026-06-11): hardened against intermittent failures.
  - Browser User-Agent on every request (many CDNs reject feedparser's default).
  - 8-second socket timeout (default is blocking).
  - 2 retries with backoff on parse/network error before giving up.
  - Curated list trimmed: removed feeds that fail >50% of pulses even with UA
    (wapo-world, crisis-group, cfr-blogs, brookings-foreign-policy, rusi, iiss,
     chatham-house). Replaced with verified-working alternates (cnn-world,
     npr-world, defense/security analysis, regional MidEast).

NOT an info_timing source — that's data/info_timing/ for non-English alpha.
This is the English-default RSS layer.
"""

from __future__ import annotations

import logging
import socket
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import feedparser

from serenity.data.news.types import NewsItem

log = logging.getLogger(__name__)

# Browser-style UA — many CDNs (WaPo, AP, Brookings, CFR, ...) silently reject
# feedparser's default 'UniversalFeedParser/...' and return empty/error.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_FEED_TIMEOUT_S = 8.0
_FEED_RETRIES = 2  # total attempts = 1 + retries
_FEED_BACKOFF_S = 1.5

# Curated geopolitics RSS feeds — English primary layer.
#
# V2 audit (2026-06-11): scored each feed against last 14 days of news-watcher
# logs. Removed feeds with chronic >50% failure even with browser UA:
#   - wapo-world: ~83 timeouts in 14 days, persistent even with UA
#   - crisis-group, cfr-blogs, brookings-foreign-policy: server returns empty
#     body or 403 to feedparser; tested with curl + UA = same.
#   - rusi, iiss, chatham-house: same — think-tank CDNs unreliable.
# Kept feeds with intermittent failure (we now retry on those):
#   - war-on-the-rocks, bellingcat: failed without UA; pass with UA.
# Added new verified-working feeds:
#   - cnn-world, npr-world: high-volume wires
#   - breakingdefense, just-security, defense-news: replaces war-on-the-rocks
#   - times-of-israel: MidEast/Iran focus (replaces middle-east-eye if it dies)
#   - the-conversation-politics: academic geopolitics analysis
RSS_FEEDS: tuple[tuple[str, str], ...] = (
    # ── Wire services (fast, high volume) ─────────────────────────────────
    ("bbc-world", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("guardian-world", "https://www.theguardian.com/world/rss"),
    ("ft-world", "https://www.ft.com/world?format=rss"),
    ("cnn-world", "http://rss.cnn.com/rss/edition_world.rss"),
    ("npr-world", "https://feeds.npr.org/1004/rss.xml"),
    # ── Deep analysis / foreign policy ────────────────────────────────────
    ("foreign-affairs", "https://www.foreignaffairs.com/rss.xml"),
    ("foreign-policy", "https://foreignpolicy.com/feed/"),
    ("the-economist-international", "https://www.economist.com/international/rss.xml"),
    ("war-on-the-rocks", "https://warontherocks.com/feed/"),
    ("the-diplomat", "https://thediplomat.com/feed/"),
    ("bellingcat", "https://www.bellingcat.com/feed/"),
    ("just-security", "https://www.justsecurity.org/feed/"),
    ("politico-eu", "https://www.politico.eu/feed/"),
    ("thehill-international", "https://thehill.com/policy/international/feed/"),
    # ── Defense / security ─────────────────────────────────────────────────
    ("defense-news", "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml"),
    ("c4isrnet", "https://www.c4isrnet.com/arc/outboundfeeds/rss/?outputType=xml"),
    ("modern-war-institute", "https://mwi.westpoint.edu/feed/"),
    # ── Regional / conflict focus ──────────────────────────────────────────
    ("aljazeera-international", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("middle-east-eye", "https://www.middleeasteye.net/rss"),
    ("times-of-israel", "https://www.timesofisrael.com/feed/"),
    # ── Think tanks / macro ────────────────────────────────────────────────
    ("the-conversation-politics", "https://theconversation.com/au/politics/articles.atom"),
)
# NOTE: Dropped reuters-world / nyt-world (SSL handshake fails from our IP —
# both endpoints sit behind Cloudflare and reject datacenter/non-residential
# IPs). Dropped lawfare (403), wilson-center (404), rand-world (returns 200
# but empty atom). All replaced by politico-eu / thehill-international /
# already-listed alternates that pass the verification probe.


# Process-local TTL cache. Keyed on (max_items_per_feed,). Empty list is
# a valid cache entry (we don't retry "all sources failed" within the TTL —
# that would re-hammer them). A fresh pulse cycle (every 30 min by launchd)
# starts a new process, so the cache naturally resets per-pulse.
_CACHE_TTL_S = 600  # 10 minutes — covers one full pulse, expires before the next
_cache_lock = __import__("threading").Lock()
_cache: dict[int, tuple[float, list]] = {}


def _cache_get(key: int) -> list | None:
    """Return cached pool if fresh, else None."""
    import time as _t

    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        ts, pool = entry
        if _t.time() - ts > _CACHE_TTL_S:
            _cache.pop(key, None)
            return None
        return pool


def _cache_put(key: int, value: list) -> None:
    import time as _t

    with _cache_lock:
        _cache[key] = (_t.time(), value)


def _parse_published(entry) -> datetime:
    if entry.get("published_parsed"):
        try:
            import time as _time

            return datetime.fromtimestamp(_time.mktime(entry.published_parsed), tz=UTC)
        except Exception:
            pass
    if entry.get("published"):
        try:
            return parsedate_to_datetime(entry.published)
        except Exception:
            pass
    return datetime.now(UTC)


def _parse_once(url: str) -> "feedparser.FeedParserDict":
    """Single feedparser call with UA + socket timeout.

    feedparser doesn't expose per-call timeout, so we set the default via
    socket. We restore the previous default after the call.
    """
    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_FEED_TIMEOUT_S)
    try:
        return feedparser.parse(url, request_headers={"User-Agent": _UA})
    finally:
        socket.setdefaulttimeout(prev_timeout)


def _is_usable(parsed) -> bool:
    """A feed is usable if we got at least one entry.

    `bozo=1` alone is not fatal — many feeds (oilprice, dw) have minor XML
    quirks but still parse entries. We treat empty-entries as the only fail
    state worth retrying for.
    """
    return bool(getattr(parsed, "entries", None))


def fetch_feed(source: str, url: str, max_items: int = 50) -> list[NewsItem]:
    """Pull one feed with retries. Returns at most `max_items` recent entries.

    Retries `_FEED_RETRIES` times with exponential backoff on:
      - any exception during parse
      - empty entries list (server returned 200 with empty body)
    """
    last_err: Exception | None = None
    for attempt in range(_FEED_RETRIES + 1):
        try:
            parsed = _parse_once(url)
            if _is_usable(parsed):
                break
            last_err = RuntimeError(
                f"empty entries (bozo={int(bool(getattr(parsed, 'bozo', 0)))})"
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
        if attempt < _FEED_RETRIES:
            time.sleep(_FEED_BACKOFF_S * (attempt + 1))
    else:
        # All attempts exhausted, none usable
        log.warning("rss %s failed after %d tries: %s", source, _FEED_RETRIES + 1, last_err)
        return []

    items: list[NewsItem] = []
    for entry in parsed.entries[:max_items]:
        items.append(
            NewsItem(
                source=f"rss:{source}",
                url=entry.get("link", ""),
                title=entry.get("title", ""),
                published_at=_parse_published(entry),
                summary=(entry.get("summary") or entry.get("description") or "")[:600],
            )
        )
    return items


def fetch_all_feeds(max_items_per_feed: int = 30) -> list[NewsItem]:
    """Fetch all configured feeds in parallel (ThreadPoolExecutor).

    22 feeds × ~1–8s each = ~22s sequential → ~5-12s parallel (max_workers=8,
    bounded by slowest retry chain ≈ 8s timeout × 3 attempts = 24s worst-case
    per feed; but most return in <2s on first try).

    Process-local TTL cache (_CACHE_TTL_S=600): a news-watch pulse calls this
    function once per market (~30 markets per pulse), and hammering the same
    sources 30× in 5 minutes triggers HTTP 429 rate-limits (observed on
    times-of-israel). The cache keys on (max_items_per_feed,) so all markets
    within a pulse share one fetch; the entry expires after 10 min so the
    next pulse re-fetches fresh.
    """
    cached = _cache_get(max_items_per_feed)
    if cached is not None:
        return list(cached)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: list[NewsItem] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(fetch_feed, source, url, max_items_per_feed): source
            for source, url in RSS_FEEDS
        }
        for fut in as_completed(futures):
            source = futures[fut]
            try:
                items = fut.result()
                out.extend(items)
                log.debug("rss %s → %d items", source, len(items))
            except Exception as e:
                log.warning("rss %s error: %s", source, e)
    _cache_put(max_items_per_feed, out)
    return out
