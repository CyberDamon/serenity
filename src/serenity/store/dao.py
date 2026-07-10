"""DB 引擎 + session 管理。SQLite 默认，DATABASE_URL 可切 MySQL。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from serenity.config import settings
from serenity.store.models import Base

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _ensure_engine():
    global _engine, _SessionLocal
    if _engine is None:
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        _engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine, _SessionLocal


def init_db() -> None:
    """建表（幂等）+ 轻量列迁移（create_all 不给已存在的表加新列）。"""
    engine, _ = _ensure_engine()
    Base.metadata.create_all(engine)
    _ensure_columns(engine)


# 新增列的轻量迁移（无 alembic）：SQLite/MySQL 均支持 ADD COLUMN。
# 只加列、不改类型/删列——够 serenity 早期演进用。
_EXPECTED_COLUMNS: dict[str, dict[str, str]] = {
    "predictions": {
        "research": "TEXT",
        # 三臂 + 闸门 + 实验完整性（serenity 增列；新库由 create_all 覆盖，
        # 这里兜底老库升级）
        "generic_prob": "FLOAT",
        "gate_state": "VARCHAR(16)",
        "gate_rationale": "TEXT",
        "delta_log_odds": "FLOAT",
        "prior_direction": "VARCHAR(8)",
        "prior_strength": "VARCHAR(12)",
        "belief_ids": "TEXT",
        "prior_rationale": "TEXT",
        "placebo_prob": "FLOAT",
        "placebo_delta_log_odds": "FLOAT",
        "belief_set_version": "VARCHAR(64)",
        "parse_errors": "TEXT",
    },
}


def _ensure_columns(engine) -> None:
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table, cols in _EXPECTED_COLUMNS.items():
            if table not in existing_tables:
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for col, sqltype in cols.items():
                if col not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {sqltype}"))


@contextmanager
def session_scope() -> Iterator[Session]:
    _, SessionLocal = _ensure_engine()
    assert SessionLocal is not None
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
