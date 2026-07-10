"""CostTracker 原子预留（并发防超 cap）单元测。"""

from __future__ import annotations

import pytest

from serenity.agent.llm_client import CostCapExceeded, CostTracker


def test_reserve_records_and_blocks_over_cap():
    ct = CostTracker(daily_cap_usd=1.0)
    ct.reserve(0.6)                       # 预留即记账
    assert abs(ct.today_spend - 0.6) < 1e-9
    ct.record(0.1)                        # 结算实际差额 → 0.7
    assert abs(ct.today_spend - 0.7) < 1e-9
    ct.reserve(0.2)                       # 0.9 ≤ 1.0 ok
    with pytest.raises(CostCapExceeded):
        ct.reserve(0.2)                   # 1.1 > 1.0 → 拦下
    assert abs(ct.today_spend - 0.9) < 1e-9  # 被拦的预留不记账


def test_reserve_is_atomic_check_plus_record():
    # reserve 把"检查+记账"合到一把锁内：连续预留到刚好触顶，下一笔必被拦
    ct = CostTracker(daily_cap_usd=1.0)
    ct.reserve(1.0)
    with pytest.raises(CostCapExceeded):
        ct.reserve(0.0001)
