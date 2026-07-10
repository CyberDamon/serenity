"""共享 fixtures：临时 SQLite 库 + schema 驱动的假 LLM。"""

from __future__ import annotations

from datetime import date

import pytest

from serenity.agent.llm_client import LLMResponse


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    import serenity.store.dao as dao
    from serenity.config import settings

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/t.db")
    dao._engine = None
    dao._SessionLocal = None
    dao.init_db()
    yield dao
    dao._engine = None
    dao._SessionLocal = None


class ScriptedLLM:
    """按调用顺序返回预置 parsed_json 的假 client。

    scripts: list——每次 complete 弹出一个元素：
      dict → parsed_json；Exception 实例 → raise；None → parsed_json=None（坏输出）。
    列表耗尽后重复最后一个。
    """

    def __init__(self, scripts: list, model: str = "fake-model"):
        self.scripts = list(scripts)
        self.model = model
        self.calls: list[dict] = []

    @property
    def training_cutoff(self) -> date:
        return date(2020, 1, 1)

    def complete(self, *, system, user, max_tokens=1024, response_schema=None, **kw):
        self.calls.append({"system": system, "user": user, "schema": response_schema})
        item = self.scripts.pop(0) if len(self.scripts) > 1 else self.scripts[0]
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text="", parsed_json=item, model=self.model, cost_usd=0.001)


@pytest.fixture
def scripted_llm():
    return ScriptedLLM
