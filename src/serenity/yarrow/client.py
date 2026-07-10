"""Yarrow REST client — 从零实现，严格对齐 serenity/yarrow-agent-integration.md。

覆盖 Phase 0 需要的全部动作：
  - 认证：SIWE nonce → EIP-191 签名 → access token；api-key 创建/轮换
  - 读：列题（分页，服务端每页硬顶 20）、单题详情（含 resolution_kind）、
        cross-market 批量、链上快照、track-record
  - 写：批量提交 forecast、撤回

关键事实（已 curl 生产 API 核实）：
  - 已结算 binary 题带 `resolution_kind ∈ {"yes","no", ...}`。yes→1 / no→0；
    其它值（未终结 / void-5050）→ outcome_int 返回 None，由 reconcile 决定处理。
  - 列题端点 page_size 被服务端硬顶 20 条/页，须用 page 翻页。

ASCII 流（reconcile 命门）：

  本地待结算预测 ─► get_question(id) ─► resolution_kind ─► outcome_int(yes=1/no=0/其它=None)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
from eth_account import Account
from eth_account.messages import encode_defunct

from serenity.config import settings

# 服务端对列题端点的每页上限（已实测）。
YARROW_PAGE_SIZE_CAP = 20
# 批量提交/cross-market 上限（见文档）。
FORECAST_BATCH_MAX = 100
CROSS_MARKET_BATCH_MAX = 60


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class YarrowAPIError(RuntimeError):
    """非 2xx 响应基类。4xx 业务错默认不重试。"""

    def __init__(
        self,
        status_code: int,
        code: str = "unknown",
        detail: str = "",
        raw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"Yarrow {status_code} {code}: {detail}")
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.raw = raw or {}


class YarrowRateLimitError(YarrowAPIError):
    """429；带 Retry-After 秒数，可重试。"""

    def __init__(self, retry_after_s: float, raw: dict[str, Any] | None = None) -> None:
        super().__init__(429, "rate_limited", f"retry after {retry_after_s}s", raw)
        self.retry_after_s = retry_after_s


class YarrowServerError(YarrowAPIError):
    """5xx；可重试。"""


class YarrowTransientError(YarrowAPIError):
    """网络/传输失败；可重试。"""


# ─────────────────────────────────────────────────────────────────────────────
# DTOs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class YarrowAuthResult:
    access: str
    refresh: str
    user_id: str
    expires_at: str
    refresh_exp: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class YarrowQuestionDTO:
    id: str
    type: str
    title: str
    status: str | None = None
    description: str | None = None
    resolution_criteria: str | None = None
    scheduled_close_time: str | None = None
    scheduled_resolve_time: str | None = None
    actual_close_time: str | None = None
    actual_resolve_time: str | None = None
    # ★ 已结算 binary 题的真值字段（"yes" / "no" / void 等）。open 题为 None。
    resolution_kind: str | None = None
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    range_min: float | None = None
    range_max: float | None = None
    open_lower_bound: bool = False
    open_upper_bound: bool = False
    source: str | None = None
    source_id: str | None = None
    source_url: str | None = None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> YarrowQuestionDTO:
        return cls(
            id=str(data["id"]),
            type=str(data.get("type") or ""),
            title=str(data.get("title") or ""),
            status=data.get("status"),
            description=data.get("description"),
            resolution_criteria=data.get("resolution_criteria"),
            scheduled_close_time=data.get("scheduled_close_time"),
            scheduled_resolve_time=data.get("scheduled_resolve_time"),
            actual_close_time=data.get("actual_close_time"),
            actual_resolve_time=data.get("actual_resolve_time"),
            resolution_kind=data.get("resolution_kind"),
            category=data.get("category"),
            tags=list(data.get("tags") or []),
            range_min=data.get("range_min"),
            range_max=data.get("range_max"),
            open_lower_bound=bool(data.get("open_lower_bound", False)),
            open_upper_bound=bool(data.get("open_upper_bound", False)),
            source=data.get("source"),
            source_id=data.get("source_id"),
            source_url=data.get("source_url"),
            raw=data,
        )

    @property
    def outcome_int(self) -> int | None:
        """binary 真值 → 1(yes) / 0(no)；未终结或 void/5050/未知 → None。

        D8 决议：void 的 Brier 处理由 reconcile 决定（当 0.5 计分），此处只暴露
        干净的 yes/no 映射，非 yes/no 一律返回 None，让上层显式决定。
        """
        kind = (self.resolution_kind or "").strip().lower()
        if kind == "yes":
            return 1
        if kind == "no":
            return 0
        return None

    @property
    def is_resolved(self) -> bool:
        return self.outcome_int is not None


@dataclass
class YarrowCrossMarketDTO:
    question_id: str
    matched: bool
    market: dict[str, Any] = field(default_factory=dict)
    twin: dict[str, Any] | None = None
    arb_signals: list[dict[str, Any]] = field(default_factory=list)
    lp_signals: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_json(cls, question_id: str, data: dict[str, Any]) -> YarrowCrossMarketDTO:
        return cls(
            question_id=question_id,
            matched=bool(data.get("matched", False)),
            market=dict(data.get("market") or {}),
            twin=data.get("twin") if isinstance(data.get("twin"), dict) else None,
            arb_signals=list(data.get("arb_signals") or []),
            lp_signals=list(data.get("lp_signals") or []),
            raw=data,
        )

    @property
    def yes_bid(self) -> float | None:
        return _optional_float(self.market.get("yes_bid"))

    @property
    def yes_ask(self) -> float | None:
        return _optional_float(self.market.get("yes_ask"))

    @property
    def market_implied_prob(self) -> float | None:
        bid, ask = self.yes_bid, self.yes_ask
        if bid is None or ask is None:
            return None
        if not 0.0 <= bid <= ask <= 1.0:
            return None
        return (bid + ask) / 2

    @property
    def event_is_stale(self) -> bool:
        return bool(self.market.get("event_is_stale", False))


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────


class YarrowClient:
    """同步 Yarrow client。读端点多为公开无鉴权；写端点需 api_key。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or settings.yarrow_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else (
            settings.yarrow_api_key.get_secret_value() or None
        )
        self._client = http_client or httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> YarrowClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── 认证 ──

    def nonce(self, address: str) -> dict[str, Any]:
        return self._request(
            "POST", "/api/auth/social/siwe/nonce/", json={"address": address}, auth=False
        )

    def siwe_login(self, *, address: str, private_key: str) -> YarrowAuthResult:
        nonce = self.nonce(address)
        message = str(nonce["message"])  # 原样回传，一字节不改
        signature = self.sign_siwe_message(message=message, private_key=private_key)
        data = self._request(
            "POST",
            "/api/auth/social/siwe/",
            json={"address": address, "signature": signature, "message": message},
            auth=False,
        )
        return YarrowAuthResult(
            access=str(data["access"]),
            refresh=str(data.get("refresh") or ""),
            user_id=str(data.get("user_id") or ""),
            expires_at=str(data.get("expires_at") or ""),
            refresh_exp=str(data.get("refresh_exp") or ""),
            raw=data,
        )

    def create_api_key(self, access_token: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/auth/api-key/",
            headers={"Authorization": f"Bearer {access_token}"},
            expected_status={200, 201},
        )

    def rotate_api_key(self, old_key: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/auth/api-key/rotate/",
            json={"old_key": old_key},
            headers={"Authorization": f"Bearer {old_key}"},
        )

    @staticmethod
    def sign_siwe_message(*, message: str, private_key: str) -> str:
        signed = Account.sign_message(encode_defunct(text=message), private_key=private_key)
        sig = signed.signature.hex()
        return sig if sig.startswith("0x") else f"0x{sig}"

    # ── 读 ──

    def list_questions(
        self,
        *,
        status: str = "open",
        qtype: str = "binary",
        category: str | None = None,
        page: int = 1,
        page_size: int = YARROW_PAGE_SIZE_CAP,
    ) -> tuple[list[YarrowQuestionDTO], bool]:
        """单页列题。返回 (questions, has_more)。page_size 会被服务端硬顶 20。"""
        params: dict[str, Any] = {
            "status": status,
            "type": qtype,
            "page": page,
            "page_size": min(page_size, YARROW_PAGE_SIZE_CAP),
        }
        if category:
            params["category"] = category
        data = self._request("GET", "/api/questions/", params=params, auth=False)
        rows = data.get("results", data if isinstance(data, list) else [])
        questions = [YarrowQuestionDTO.from_json(r) for r in rows]
        has_more = bool(data.get("has_more")) if isinstance(data, dict) else False
        return questions, has_more

    def iter_questions(
        self,
        *,
        status: str = "open",
        qtype: str = "binary",
        category: str | None = None,
        max_items: int | None = None,
    ) -> Iterator[YarrowQuestionDTO]:
        """跨页迭代所有题（服务端每页 20，此处自动翻页）。"""
        page = 1
        seen = 0
        while True:
            questions, has_more = self.list_questions(
                status=status, qtype=qtype, category=category, page=page
            )
            if not questions:
                return
            for q in questions:
                yield q
                seen += 1
                if max_items is not None and seen >= max_items:
                    return
            if not has_more:
                return
            page += 1

    def get_question(self, question_id: str) -> YarrowQuestionDTO:
        """单题详情（公开端点）。已结算 binary 题带 resolution_kind —— reconcile 命门。"""
        data = self._request(
            "GET", f"/api/questions/{question_id}/", auth=bool(self.api_key)
        )
        return YarrowQuestionDTO.from_json(data)

    def get_onchain_snapshot(self, question_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/onchain/snapshot/{question_id}/", auth=False)

    def batch_cross_market(
        self, question_ids: list[str]
    ) -> dict[str, YarrowCrossMarketDTO]:
        if len(question_ids) > CROSS_MARKET_BATCH_MAX:
            raise ValueError(f"cross-market batch 最多 {CROSS_MARKET_BATCH_MAX} 个 id")
        if not question_ids:
            return {}
        data = self._request(
            "POST", "/api/cross-market/batch/", json={"ids": question_ids}, auth=False
        )
        return _parse_cross_market_batch(data, question_ids)

    def get_track_record(self, *, signed: bool = False) -> dict[str, Any]:
        path = "/api/users/me/track-record/signed/" if signed else "/api/users/me/track-record/"
        return self._request("GET", path)

    # ── 写 ──

    def submit_forecasts(self, forecasts: list[dict[str, Any]]) -> None:
        """批量提交。单条也走此端点。幂等（同 user+question 后写覆盖前写）。"""
        if not self.api_key:
            raise YarrowAPIError(401, "missing_api_key", "YARROW_API_KEY is required")
        if len(forecasts) > FORECAST_BATCH_MAX:
            raise ValueError(f"forecast batch 最多 {FORECAST_BATCH_MAX} 条")
        for item in forecasts:
            if "probability_yes" in item:
                prob = float(item["probability_yes"])
                if not 0.0 <= prob <= 1.0:
                    raise ValueError(f"probability_yes 必须 ∈ [0,1]，得到 {prob}")
        self._request(
            "POST",
            "/api/questions/forecast/",
            json={"forecasts": forecasts},
            headers={"X-Yarrow-Source": "agent"},
            expected_status={201},
        )

    def withdraw_forecasts(self, question_ids: list[str]) -> int:
        if not self.api_key:
            raise YarrowAPIError(401, "missing_api_key", "YARROW_API_KEY is required")
        data = self._request(
            "POST", "/api/questions/withdraw/", json={"question_ids": question_ids}
        )
        return int((data or {}).get("count", 0))

    # ── HTTP core ──

    def _request(
        self,
        method: str,
        path: str,
        *,
        auth: bool = True,
        expected_status: set[int] | None = None,
        headers: dict[str, str] | None = None,
        max_attempts: int = 3,
        **kwargs: Any,
    ) -> Any:
        expected = expected_status or {200}
        req_headers = dict(headers or {})
        if auth and self.api_key and "Authorization" not in req_headers:
            req_headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: YarrowAPIError | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._client.request(
                    method, f"{self.base_url}{path}", headers=req_headers, **kwargs
                )
            except httpx.RequestError as e:
                last_error = YarrowTransientError(0, "network_error", str(e))
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise last_error from e

            if resp.status_code in expected:
                return resp.json() if resp.content else None

            error = self._error_from_response(resp)
            # 429 与 5xx/传输错可重试；其余 4xx 业务错立即失败。
            if isinstance(error, YarrowRateLimitError):
                last_error = error
                if attempt < max_attempts:
                    time.sleep(error.retry_after_s)
                    continue
            elif isinstance(error, YarrowServerError | YarrowTransientError):
                last_error = error
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
            raise error

        raise last_error or YarrowAPIError(0, "unknown", "request failed")

    @staticmethod
    def _error_from_response(resp: httpx.Response) -> YarrowAPIError:
        raw: dict[str, Any] = {}
        try:
            data = resp.json()
            if isinstance(data, dict):
                raw = data
        except ValueError:
            raw = {}

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            try:
                retry_after_s = float(retry_after) if retry_after is not None else 1.0
            except ValueError:
                retry_after_s = 1.0
            return YarrowRateLimitError(retry_after_s=retry_after_s, raw=raw)

        code = str(raw.get("code") or f"http_{resp.status_code}")
        detail = str(raw.get("detail") or resp.text[:200])
        if resp.status_code >= 500:
            return YarrowServerError(resp.status_code, code, detail, raw)
        return YarrowAPIError(resp.status_code, code, detail, raw)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def parse_yarrow_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_cross_market_batch(
    data: Any, question_ids: list[str]
) -> dict[str, YarrowCrossMarketDTO]:
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return _parse_cross_market_list(results, question_ids)
        if isinstance(results, dict):
            return {
                str(qid): YarrowCrossMarketDTO.from_json(str(qid), item)
                for qid, item in results.items()
                if isinstance(item, dict)
            }
        if data and all(isinstance(v, dict) for v in data.values()):
            return {
                str(qid): YarrowCrossMarketDTO.from_json(str(qid), item)
                for qid, item in data.items()
            }
    if isinstance(data, list):
        return _parse_cross_market_list(data, question_ids)
    return {}


def _parse_cross_market_list(
    rows: list[Any], question_ids: list[str]
) -> dict[str, YarrowCrossMarketDTO]:
    out: dict[str, YarrowCrossMarketDTO] = {}
    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        qid = (
            item.get("question_id")
            or item.get("id")
            or (question_ids[idx] if idx < len(question_ids) else None)
        )
        if qid is None:
            continue
        out[str(qid)] = YarrowCrossMarketDTO.from_json(str(qid), item)
    return out
