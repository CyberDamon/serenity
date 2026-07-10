"""Yarrow client mock 单元测（D9 第一层，全离线，进 CI）。

覆盖覆盖图里的关键分支：resolution_kind 映射(D8) / 分页 / 提交校验 / 错误码+重试。
"""

from __future__ import annotations

import json

import httpx
import pytest

from serenity.yarrow import (
    YarrowAPIError,
    YarrowClient,
    YarrowQuestionDTO,
)
from serenity.yarrow.client import YarrowRateLimitError


def _client(handler) -> YarrowClient:
    transport = httpx.MockTransport(handler)
    return YarrowClient(
        base_url="https://yarrow.test",
        api_key="yk_test",
        http_client=httpx.Client(transport=transport),
    )


# ── resolution_kind → outcome_int（D8 命门）──


@pytest.mark.parametrize(
    "kind,expected",
    [("yes", 1), ("no", 0), ("YES", 1), ("No", 0), ("5050", None), ("void", None), (None, None)],
)
def test_outcome_int_mapping(kind, expected):
    q = YarrowQuestionDTO.from_json(
        {"id": "q1", "type": "binary", "title": "t", "resolution_kind": kind}
    )
    assert q.outcome_int == expected
    assert q.is_resolved == (expected is not None)


def test_question_parses_core_fields():
    q = YarrowQuestionDTO.from_json(
        {
            "id": "abc",
            "type": "binary",
            "title": "Starmer out by June 30, 2026?",
            "resolution_kind": "yes",
            "category": "politics",
            "source": "polymarket",
            "source_id": "123",
            "scheduled_resolve_time": "2026-06-30T00:00:00Z",
        }
    )
    assert q.outcome_int == 1
    assert q.source_id == "123"
    assert q.category == "politics"


# ── 分页 has_more ──


def test_list_questions_has_more_and_cap():
    seen_params = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.update(dict(request.url.params))
        return httpx.Response(
            200,
            json={"count": 40, "has_more": True, "page": 1,
                  "results": [{"id": f"q{i}", "type": "binary", "title": "t"} for i in range(20)]},
        )

    client = _client(handler)
    qs, has_more = client.list_questions(status="open", qtype="binary", page_size=200)
    assert len(qs) == 20
    assert has_more is True
    # page_size 请求侧被封顶到 20
    assert seen_params["page_size"] == "20"


def test_iter_questions_walks_pages():
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(dict(request.url.params).get("page", "1"))
        if page == 1:
            return httpx.Response(200, json={"has_more": True,
                "results": [{"id": f"a{i}", "type": "binary", "title": "t"} for i in range(20)]})
        return httpx.Response(200, json={"has_more": False,
            "results": [{"id": "b0", "type": "binary", "title": "t"}]})

    client = _client(handler)
    ids = [q.id for q in client.iter_questions(status="resolved")]
    assert len(ids) == 21
    assert ids[-1] == "b0"


# ── 提交校验 ──


def test_submit_rejects_out_of_range():
    client = _client(lambda r: httpx.Response(201))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        client.submit_forecasts([{"question_id": "q", "probability_yes": 1.5}])


def test_submit_rejects_too_many():
    client = _client(lambda r: httpx.Response(201))
    batch = [{"question_id": f"q{i}", "probability_yes": 0.5} for i in range(101)]
    with pytest.raises(ValueError, match="最多"):
        client.submit_forecasts(batch)


def test_submit_requires_api_key():
    # api_key="" 强制无 key（None 会回退到 settings/.env 的真实 key，那是生产正确行为）
    client = YarrowClient(base_url="https://yarrow.test", api_key="",
                          http_client=httpx.Client(transport=httpx.MockTransport(
                              lambda r: httpx.Response(201))))
    with pytest.raises(YarrowAPIError) as ei:
        client.submit_forecasts([{"question_id": "q", "probability_yes": 0.5}])
    assert ei.value.code == "missing_api_key"


def test_submit_ok_sends_source_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["src"] = request.headers.get("X-Yarrow-Source")
        captured["body"] = json.loads(request.content)
        return httpx.Response(201)

    client = _client(handler)
    client.submit_forecasts([{"question_id": "q", "probability_yes": 0.72}])
    assert captured["src"] == "agent"
    assert captured["body"]["forecasts"][0]["probability_yes"] == 0.72


# ── 错误码 + 重试 ──


def test_429_then_success_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"detail": "slow down"})
        return httpx.Response(200, json={"results": [], "has_more": False})

    client = _client(handler)
    qs, _ = client.list_questions()
    assert qs == []
    assert calls["n"] == 2  # 重试了一次


def test_422_business_error_no_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"code": "prob_invalid", "detail": "bad"})

    client = _client(handler)
    with pytest.raises(YarrowAPIError) as ei:
        client.submit_forecasts([{"question_id": "q", "probability_yes": 0.5}])
    assert ei.value.status_code == 422
    assert ei.value.code == "prob_invalid"
    assert calls["n"] == 1  # 4xx 立即失败，不重试


def test_rate_limit_error_type():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"}, json={})

    client = _client(handler)
    with pytest.raises(YarrowRateLimitError):
        # max_attempts 用尽后抛最后一个 429
        client._request("GET", "/api/questions/", auth=False, max_attempts=1)


# ── cross-market 解析 ──


def test_cross_market_batch_parses_dict_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": {
            "q1": {"matched": True, "market": {"yes_bid": 0.19, "yes_ask": 0.21}},
            "q2": {"matched": False},
        }})

    client = _client(handler)
    out = client.batch_cross_market(["q1", "q2"])
    assert out["q1"].matched is True
    assert abs(out["q1"].market_implied_prob - 0.20) < 1e-9
    assert out["q2"].matched is False
    assert out["q2"].market_implied_prob is None


def test_cross_market_batch_too_many():
    client = _client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="最多"):
        client.batch_cross_market([f"q{i}" for i in range(61)])
