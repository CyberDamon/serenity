"""蒸馏管线：本地语料 JSON → 三张信念表 + 版本登记。

ASCII：
  corpus.json（本地，不进 repo）
      │ load + 过滤（纯转推/无文本跳过）
      ▼
  按时间序分批（batch_size 条/批）
      │ extract：LLM 批抽取（并发，坏输出跳过+计数，不落脏数据）
      ▼
  belief 原语（跨批重复） ── consolidate：按 domain LLM 合并近重复 ──► 终稿信念
  ticker 论点（按 ticker 归并，保最高置信）
  historical_claims（时间戳截断：made_at ≤ 语料内最早来源推文日期）
      │
      ▼
  belief_set_version = sha256(排序后的 claim|domain|stance)[:16]
      │ 重建守卫（评审 3A）：实验期（配对样本<min 且已有 forecast 引用）拒绝，
      │ --force 才放行 = 新实验段
      ▼
  三张表 + BeliefSetMeta（旧版本 active→False）
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select

from serenity.agent.llm_client import LLMClient, make_client
from serenity.config import settings
from serenity.distill.prompts import (
    CONSOLIDATE_SCHEMA,
    CONSOLIDATE_SYSTEM,
    DOMAINS,
    EXTRACT_SCHEMA,
    EXTRACT_SYSTEM,
    EXTRACT_USER_TEMPLATE,
)
from serenity.store.dao import init_db, session_scope
from serenity.store.models import (
    BeliefPrimitive,
    BeliefSetMeta,
    HistoricalClaim,
    Prediction,
    Resolution,
    TickerKnowledge,
)

log = logging.getLogger(__name__)


@dataclass
class DistillReport:
    version: str
    n_tweets_used: int = 0
    n_batches: int = 0
    n_batches_failed: int = 0
    n_beliefs_raw: int = 0
    n_beliefs: int = 0
    n_tickers: int = 0
    n_claims: int = 0
    n_claims_rejected_timestamp: int = 0
    cost_usd: float = 0.0
    corpus_span: str = ""
    skipped_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


# ── 语料载入 ──────────────────────────────────────────────────────────────────


def load_corpus(path: str | Path | None = None) -> list[dict]:
    """载入本地推文 JSON。只取分析型内容：跳过无评论纯转推与空文本。"""
    p = Path(path or settings.corpus_json_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"语料不存在: {p} —— 先本地 clone yan-labs/serenity-aleabitoreddit（不进 repo）"
        )
    raw = json.loads(p.read_text())
    tweets = raw if isinstance(raw, list) else raw.get("tweets", [])
    out: list[dict] = []
    seen_ids: set[str] = set()
    for t in tweets:
        tid = str(t.get("id") or "")
        text = (t.get("text") or "").strip()
        if not tid or not text or tid in seen_ids:
            continue
        if t.get("isRetweet") and not t.get("isQuote"):
            continue  # 纯转推无本人观点
        created = t.get("createdAtISO") or t.get("createdAt") or ""
        out.append({"id": tid, "date": str(created)[:10], "text": text})
        seen_ids.add(tid)
    out.sort(key=lambda x: x["date"])
    return out


def _batches(tweets: list[dict], batch_size: int) -> list[list[dict]]:
    return [tweets[i : i + batch_size] for i in range(0, len(tweets), batch_size)]


def _posts_block(batch: list[dict]) -> str:
    lines = []
    for t in batch:
        text = " ".join(t["text"].split())
        lines.append(f"[{t['id']}] {t['date']} | {text}")
    return "\n".join(lines)


# ── LLM 抽取 ──────────────────────────────────────────────────────────────────


def _extract_batch(llm: LLMClient, batch: list[dict]) -> tuple[dict | None, float]:
    """单批抽取。坏输出返回 (None, cost)——跳过不落脏数据。"""
    user = EXTRACT_USER_TEMPLATE.format(
        n=len(batch),
        date_from=batch[0]["date"],
        date_to=batch[-1]["date"],
        posts_block=_posts_block(batch),
    )
    system = EXTRACT_SYSTEM.format(domains=", ".join(DOMAINS))
    try:
        resp = llm.complete(
            system=system,
            user=user,
            max_tokens=4000,
            response_schema=EXTRACT_SCHEMA,
            estimated_input_tokens=6000,
            estimated_output_tokens=2000,
        )
    except Exception as e:  # 单批失败不拖垮全量（含 transient 重试后仍失败）
        log.warning("extract batch failed: %s", e)
        return None, 0.0
    parsed = resp.parsed_json
    if not isinstance(parsed, dict) or "belief_primitives" not in parsed:
        return None, resp.cost_usd
    return parsed, resp.cost_usd


def _valid_domain(d: str) -> str:
    return d if d in DOMAINS else "other"


def _consolidate_domain(
    llm: LLMClient, domain: str, items: list[dict]
) -> tuple[list[dict], float]:
    """按 domain 合并近重复。合并失败时 fail-open 返回原 items（不丢信念）。"""
    if len(items) <= 1:
        return items, 0.0
    listing = "\n".join(f"[{i}] ({it['stance']}/{it['confidence']}) {it['claim']}" for i, it in enumerate(items))
    try:
        resp = llm.complete(
            system=CONSOLIDATE_SYSTEM,
            user=f"Domain: {domain}\nClaims:\n{listing}",
            max_tokens=4000,
            response_schema=CONSOLIDATE_SCHEMA,
            estimated_input_tokens=3000,
            estimated_output_tokens=1500,
        )
    except Exception as e:
        log.warning("consolidate %s failed（fail-open 保留原始条目）: %s", domain, e)
        return items, 0.0
    parsed = resp.parsed_json
    if not isinstance(parsed, dict) or not parsed.get("merged"):
        return items, resp.cost_usd

    merged_out: list[dict] = []
    covered: set[int] = set()
    for m in parsed["merged"]:
        idxs = [i for i in m.get("member_indexes", []) if 0 <= i < len(items)]
        if not idxs:
            continue
        members = [items[i] for i in idxs]
        covered.update(idxs)
        tweet_ids = sorted({tid for it in members for tid in it["tweet_ids"]})
        tickers = sorted({tk for it in members for tk in it.get("tickers", [])})
        causal = m.get("causal_links") or [c for it in members for c in it.get("causal_links", [])]
        dates = sorted(d for it in members for d in [it.get("first_seen", ""), it.get("last_seen", "")] if d)
        merged_out.append({
            "claim": m["claim"],
            "domain": domain,
            "tickers": tickers,
            "stance": m["stance"],
            "confidence": m["confidence"],
            "causal_links": causal[:8],
            "tweet_ids": tweet_ids,
            "first_seen": dates[0] if dates else "",
            "last_seen": dates[-1] if dates else "",
        })
    # 未被覆盖的原始条目保留（LLM 漏归组不等于该信念不存在）
    for i, it in enumerate(items):
        if i not in covered:
            merged_out.append(it)
    return merged_out, resp.cost_usd


# ── 版本与守卫 ────────────────────────────────────────────────────────────────


def compute_version(beliefs: list[dict]) -> str:
    """内容 hash：对排序后的 claim|domain|stance 做 sha256，取 16 hex。"""
    keys = sorted(f"{b['claim']}|{b['domain']}|{b['stance']}" for b in beliefs)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def paired_sample_count(version: str) -> int:
    """当前版本下已结算的配对样本数（gate∈{in,adjacent} 且三臂齐 且有真值）。"""
    with session_scope() as s:
        n = s.execute(
            select(func.count())
            .select_from(Prediction)
            .join(Resolution, Resolution.question_id == Prediction.question_id)
            .where(Prediction.belief_set_version == version)
            .where(Prediction.gate_state.in_(("in_domain", "adjacent")))
            .where(Prediction.generic_prob.is_not(None))
            .where(Prediction.final_prob.is_not(None))
            .where(Resolution.outcome.is_not(None))
        ).scalar_one()
    return int(n)


def _forecasts_referencing(version: str) -> int:
    with session_scope() as s:
        n = s.execute(
            select(func.count()).select_from(Prediction)
            .where(Prediction.belief_set_version == version)
        ).scalar_one()
    return int(n)


def active_version() -> str | None:
    with session_scope() as s:
        row = s.execute(
            select(BeliefSetMeta.version)
            .where(BeliefSetMeta.active.is_(True))
            .order_by(BeliefSetMeta.created_at.desc())
        ).first()
    return row[0] if row else None


def rebuild_guard(force: bool) -> str | None:
    """实验期重建守卫。返回拒绝原因（None = 放行）。"""
    cur = active_version()
    if cur is None:
        return None  # 首次蒸馏
    n_ref = _forecasts_referencing(cur)
    if n_ref == 0:
        return None  # 实验未开始，可自由重建
    n_paired = paired_sample_count(cur)
    if n_paired >= settings.experiment_min_paired:
        return None  # 实验段已满，可重建（自然开新段）
    if force:
        log.warning(
            "--force 重建：版本 %s 实验中断（paired=%d/%d, forecasts=%d）→ 开新实验段",
            cur, n_paired, settings.experiment_min_paired, n_ref,
        )
        return None
    return (
        f"实验期冻结（评审 3A）：版本 {cur} 已有 {n_ref} 条 forecast、"
        f"配对样本 {n_paired}/{settings.experiment_min_paired}。"
        "重建会污染处理组。确要重建用 --force（= 开新实验段，样本不合并）。"
    )


# ── 主流程 ────────────────────────────────────────────────────────────────────


def run_distill(
    *,
    corpus_path: str | None = None,
    model: str | None = None,
    batch_size: int = 40,
    concurrency: int = 4,
    force: bool = False,
    llm: LLMClient | None = None,
) -> DistillReport:
    init_db()
    reason = rebuild_guard(force)
    if reason:
        rep = DistillReport(version="", skipped_reason=reason)
        log.error("%s", reason)
        return rep

    llm = llm or make_client(model or settings.distill_model)
    tweets = load_corpus(corpus_path)
    if not tweets:
        return DistillReport(version="", skipped_reason="语料为空")
    corpus_span = f"{tweets[0]['date']}..{tweets[-1]['date']}"
    corpus_max_date = tweets[-1]["date"]
    tweet_dates = {t["id"]: t["date"] for t in tweets}

    batches = _batches(tweets, batch_size)
    rep = DistillReport(version="", n_tweets_used=len(tweets), n_batches=len(batches),
                        corpus_span=corpus_span)

    # ── extract（并发；单批失败跳过计数）──
    def _one(batch: list[dict]):
        return _extract_batch(llm, batch)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        results = list(ex.map(_one, batches))

    raw_beliefs: list[dict] = []
    raw_tickers: list[dict] = []
    raw_claims: list[dict] = []
    for parsed, cost in results:
        rep.cost_usd += cost
        if parsed is None:
            rep.n_batches_failed += 1
            continue
        for b in parsed.get("belief_primitives", []):
            ids = [str(i) for i in b.get("tweet_ids", []) if str(i) in tweet_dates]
            if not ids:
                continue  # 引用不到真实推文 = 幻觉，丢弃
            dates = sorted(tweet_dates[i] for i in ids)
            raw_beliefs.append({
                "claim": b["claim"], "domain": _valid_domain(b.get("domain", "other")),
                "tickers": [t.upper() for t in b.get("tickers", [])],
                "stance": b.get("stance", "neutral"), "confidence": b.get("confidence", "low"),
                "causal_links": b.get("causal_links", []), "tweet_ids": ids,
                "first_seen": dates[0], "last_seen": dates[-1],
            })
        for t in parsed.get("ticker_theses", []):
            ids = [str(i) for i in t.get("tweet_ids", []) if str(i) in tweet_dates]
            if not ids or not t.get("ticker"):
                continue
            raw_tickers.append({
                "ticker": t["ticker"].upper(), "subsector": t.get("subsector"),
                "thesis": t["thesis"], "confidence": t.get("confidence", "low"),
                "tweet_ids": ids,
            })
        for c in parsed.get("historical_claims", []):
            ids = [str(i) for i in c.get("tweet_ids", []) if str(i) in tweet_dates]
            if not ids:
                continue
            earliest = min(tweet_dates[i] for i in ids)
            made_at = c.get("made_at") or earliest
            # 时间戳截断（评审 8A）：made_at 必须 = 最早来源推文日期（±容忍 0），
            # 且不得晚于语料末尾。LLM 给的日期与来源不符时以来源为准。
            if made_at != earliest:
                made_at = earliest
            if made_at > corpus_max_date:
                rep.n_claims_rejected_timestamp += 1
                continue
            raw_claims.append({
                "claim": c["claim"], "direction": c.get("direction"),
                "made_at": made_at, "horizon": c.get("horizon"), "tweet_ids": ids,
            })
    rep.n_beliefs_raw = len(raw_beliefs)

    if not raw_beliefs:
        rep.skipped_reason = "抽取产出为空（全部批次失败？）"
        return rep

    # ── consolidate：按 domain 合并 ──
    by_domain: dict[str, list[dict]] = {}
    for b in raw_beliefs:
        by_domain.setdefault(b["domain"], []).append(b)
    final_beliefs: list[dict] = []
    for domain, items in by_domain.items():
        merged, cost = _consolidate_domain(llm, domain, items)
        rep.cost_usd += cost
        final_beliefs.extend(merged)

    # ticker 论点按 (ticker) 归并：保最高置信、合并 tweet_ids
    conf_rank = {"low": 0, "medium": 1, "high": 2}
    tick_map: dict[str, dict] = {}
    for t in raw_tickers:
        cur = tick_map.get(t["ticker"])
        if cur is None or conf_rank.get(t["confidence"], 0) > conf_rank.get(cur["confidence"], 0):
            base = dict(t)
            if cur:
                base["tweet_ids"] = sorted(set(cur["tweet_ids"]) | set(t["tweet_ids"]))
            tick_map[t["ticker"]] = base
        else:
            cur["tweet_ids"] = sorted(set(cur["tweet_ids"]) | set(t["tweet_ids"]))

    version = compute_version(final_beliefs)
    rep.version = version
    rep.n_beliefs = len(final_beliefs)
    rep.n_tickers = len(tick_map)
    rep.n_claims = len(raw_claims)

    # ── 落库（先失活旧版本，再写新版本）──
    now = datetime.now(UTC).replace(tzinfo=None)
    with session_scope() as s:
        for meta in s.query(BeliefSetMeta).filter_by(active=True).all():
            meta.active = False
        existing = s.get(BeliefSetMeta, version)
        if existing is not None:
            # 同内容重跑：幂等，激活即可
            existing.active = True
            rep.warnings.append(f"版本 {version} 已存在（内容一致），仅重新激活")
        else:
            for b in final_beliefs:
                s.add(BeliefPrimitive(
                    claim=b["claim"], domain=b["domain"],
                    tickers=",".join(b["tickers"]) or None,
                    stance=b["stance"], confidence=b["confidence"],
                    causal_links=json.dumps(b.get("causal_links", []), ensure_ascii=False),
                    source_tweet_ids=json.dumps(b["tweet_ids"]),
                    first_seen=b.get("first_seen") or None,
                    last_seen=b.get("last_seen") or None,
                    belief_set_version=version,
                ))
            for t in tick_map.values():
                s.add(TickerKnowledge(
                    ticker=t["ticker"], subsector=t.get("subsector"),
                    thesis=t["thesis"], confidence=t["confidence"],
                    source_tweet_ids=json.dumps(t["tweet_ids"]),
                    belief_set_version=version,
                ))
            for c in raw_claims:
                s.add(HistoricalClaim(
                    claim=c["claim"], direction=c.get("direction"),
                    made_at=c["made_at"], horizon=c.get("horizon"),
                    source_tweet_ids=json.dumps(c["tweet_ids"]),
                    belief_set_version=version,
                ))
            s.add(BeliefSetMeta(
                version=version, created_at=now,
                n_beliefs=rep.n_beliefs, n_tickers=rep.n_tickers, n_claims=rep.n_claims,
                corpus_span=corpus_span, distill_model=llm.model, active=True,
            ))
    log.info(
        "distill done: version=%s beliefs=%d(raw %d) tickers=%d claims=%d "
        "batches=%d(failed %d) cost=$%.2f",
        version, rep.n_beliefs, rep.n_beliefs_raw, rep.n_tickers, rep.n_claims,
        rep.n_batches, rep.n_batches_failed, rep.cost_usd,
    )
    return rep
