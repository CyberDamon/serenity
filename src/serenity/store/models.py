"""SQLAlchemy 数据模型（SQLite/MySQL 通用）。

七张表：
  predictions           —— 每题三臂结果（generic/placebo/serenity；含弃权 shadow）
  resolutions           —— 已结算真值（reconcile 写）
  calibration_snapshots —— 滚动 Brier/ECE 观测
  belief_primitives     —— 蒸馏产物：Serenity 信念原语（转述，非推文原文）
  ticker_knowledge      —— 蒸馏产物：按标的的论点知识
  historical_claims     —— 蒸馏产物：可回测历史 call（仅描述性，评审 8A）
  belief_set_meta       —— 信念库版本登记（冻结/实验段，评审 3A）

设计要点：
  - 三臂（评审 6A）：generic_prob（对照）/ placebo_prob（负控制，shuffled 信念）/
    final_prob（serenity 臂 = 提交值）。三值都随行落库，配对比较在 scoring 层做。
  - shadow：submit_status='skipped' 的行也存全部概率，防选择偏差。
  - belief_set_version 随每条 forecast 落库：处理组可追溯（评审 3A）。
  - 信念表只存转述 claim + tweet_id 引用，不存推文原文（评审 9A 合规）。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("question_id", "prediction_date", name="uq_q_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(String(64), index=True)
    prediction_date: Mapped[str] = mapped_column(String(10))  # ISO date
    title: Mapped[str | None] = mapped_column(Text, default=None)
    category: Mapped[str | None] = mapped_column(String(32), default=None)

    # ── generic 臂（对照）──
    raw_prob: Mapped[float | None] = mapped_column(Float, default=None)  # 聚合后未校准
    generic_prob: Mapped[float | None] = mapped_column(Float, default=None)  # 校准后 generic
    ir_std: Mapped[float | None] = mapped_column(Float, default=None)
    n_ir_valid: Mapped[int | None] = mapped_column(Integer, default=None)
    route_label: Mapped[str | None] = mapped_column(String(48), default=None)
    llm_models: Mapped[str | None] = mapped_column(String(128), default=None)
    market_implied_prob: Mapped[float | None] = mapped_column(Float, default=None)
    self_check_delta: Mapped[float | None] = mapped_column(Float, default=None)

    # ── 领域闸门（三态 + 判据，评审 issue 1/4A）──
    gate_state: Mapped[str | None] = mapped_column(String(16), default=None)  # in_domain|adjacent|out_of_domain
    gate_rationale: Mapped[str | None] = mapped_column(Text, default=None)  # 判据文本，必填落库

    # ── serenity 先验臂（提交臂）──
    final_prob: Mapped[float | None] = mapped_column(Float, default=None)  # serenity 臂 = 提交值
    delta_log_odds: Mapped[float | None] = mapped_column(Float, default=None)  # 缩放封顶后的实际 δ
    prior_direction: Mapped[str | None] = mapped_column(String(8), default=None)  # yes|no|none
    prior_strength: Mapped[str | None] = mapped_column(String(12), default=None)  # weak|moderate|strong|none
    belief_ids: Mapped[str | None] = mapped_column(Text, default=None)  # JSON list[int]
    prior_rationale: Mapped[str | None] = mapped_column(Text, default=None)

    # ── placebo 负控制臂（评审 6A；shadow，不提交）──
    placebo_prob: Mapped[float | None] = mapped_column(Float, default=None)
    placebo_delta_log_odds: Mapped[float | None] = mapped_column(Float, default=None)

    # ── 实验完整性（评审 3A/4A）──
    belief_set_version: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    parse_errors: Mapped[str | None] = mapped_column(Text, default=None)  # JSON list[str]，fail-closed 审计

    # 'submitted' | 'skipped'(shadow) | 'dry_run' | 'withdrawn' | 'failed' | 'pending'
    submit_status: Mapped[str] = mapped_column(String(16), default="dry_run", index=True)
    skip_reason: Mapped[str | None] = mapped_column(String(48), default=None)

    prediction_ts: Mapped[datetime | None] = mapped_column(default=None)
    first_submit_ts: Mapped[datetime | None] = mapped_column(default=None)
    question_resolve_time: Mapped[datetime | None] = mapped_column(default=None)
    # 主动检索审计（JSON）：{backend, queries[], sources:[{url,title,source}], brief, n_sources}
    research: Mapped[str | None] = mapped_column(Text, default=None)


class Resolution(Base):
    __tablename__ = "resolutions"

    question_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    resolution_kind: Mapped[str | None] = mapped_column(String(16), default=None)  # yes/no/void
    outcome: Mapped[float | None] = mapped_column(Float, default=None)  # 1/0；void=0.5
    is_void: Mapped[bool] = mapped_column(default=False)
    resolved_at: Mapped[datetime | None] = mapped_column(default=None)


class CalibrationSnapshot(Base):
    __tablename__ = "calibration_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_date: Mapped[str] = mapped_column(String(10), index=True)
    domain: Mapped[str] = mapped_column(String(48), default="all")
    scope: Mapped[str] = mapped_column(String(16), default="all")  # all(含shadow) / submitted
    n: Mapped[int] = mapped_column(Integer, default=0)
    n_effective: Mapped[float | None] = mapped_column(Float, default=None)
    brier_mean: Mapped[float | None] = mapped_column(Float, default=None)
    brier_ci_low: Mapped[float | None] = mapped_column(Float, default=None)
    brier_ci_high: Mapped[float | None] = mapped_column(Float, default=None)
    ece: Mapped[float | None] = mapped_column(Float, default=None)


# ─────────────────────────────────────────────────────────────────────────────
# 蒸馏产物（distill 写，daily 读；只存转述 claim + tweet_id 引用，评审 9A）
# ─────────────────────────────────────────────────────────────────────────────


class BeliefPrimitive(Base):
    """一条信念原语：Serenity 反复表达的世界观主张（转述）。"""

    __tablename__ = "belief_primitives"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim: Mapped[str] = mapped_column(Text)  # 转述的主张，一句话
    domain: Mapped[str] = mapped_column(String(48), index=True)  # optics_cpo|memory_hbm|...
    tickers: Mapped[str | None] = mapped_column(String(256), default=None)  # 逗号分隔大写
    stance: Mapped[str | None] = mapped_column(String(12), default=None)  # bullish|bearish|neutral
    confidence: Mapped[str | None] = mapped_column(String(12), default=None)  # low|medium|high
    causal_links: Mapped[str | None] = mapped_column(Text, default=None)  # JSON list[str] "A→B"
    source_tweet_ids: Mapped[str | None] = mapped_column(Text, default=None)  # JSON list[str]
    first_seen: Mapped[str | None] = mapped_column(String(10), default=None)  # ISO date
    last_seen: Mapped[str | None] = mapped_column(String(10), default=None)
    belief_set_version: Mapped[str] = mapped_column(String(64), index=True)


class TickerKnowledge(Base):
    """按标的聚合的论点（子板块 + 信心等级）。"""

    __tablename__ = "ticker_knowledge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    subsector: Mapped[str | None] = mapped_column(String(48), default=None)
    thesis: Mapped[str] = mapped_column(Text)  # 转述论点
    confidence: Mapped[str | None] = mapped_column(String(12), default=None)
    source_tweet_ids: Mapped[str | None] = mapped_column(Text, default=None)
    belief_set_version: Mapped[str] = mapped_column(String(64), index=True)


class HistoricalClaim(Base):
    """可回测的历史 call。仅描述性报告用，不进生产 δ 公式（评审 8A）。

    抽取纪律：claim 时间戳必须早于结局窗口（made_at = 最早来源推文日期）；
    胜利回顾/事后修正不算 call。
    """

    __tablename__ = "historical_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim: Mapped[str] = mapped_column(Text)  # 转述："X 将在 Y 时间窗内发生"
    direction: Mapped[str | None] = mapped_column(String(8), default=None)  # yes|no
    made_at: Mapped[str] = mapped_column(String(10))  # 最早来源推文 ISO date
    horizon: Mapped[str | None] = mapped_column(Text, default=None)  # 结局窗口描述
    resolved_status: Mapped[str] = mapped_column(String(12), default="unresolved")
    # correct | incorrect | unresolved | ambiguous
    outcome_note: Mapped[str | None] = mapped_column(Text, default=None)
    source_tweet_ids: Mapped[str | None] = mapped_column(Text, default=None)
    belief_set_version: Mapped[str] = mapped_column(String(64), index=True)


class BeliefSetMeta(Base):
    """信念库版本登记：冻结与实验段（评审 3A）。

    实验期（配对样本 < settings.experiment_min_paired）重建被拒；
    --force 重建 = 新 version = 新实验段，样本不与旧段合并。
    """

    __tablename__ = "belief_set_meta"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)  # 内容 hash
    created_at: Mapped[datetime | None] = mapped_column(default=None)
    n_beliefs: Mapped[int] = mapped_column(Integer, default=0)
    n_tickers: Mapped[int] = mapped_column(Integer, default=0)
    n_claims: Mapped[int] = mapped_column(Integer, default=0)
    corpus_span: Mapped[str | None] = mapped_column(String(32), default=None)  # "2025-07-02..2026-07-09"
    distill_model: Mapped[str | None] = mapped_column(String(64), default=None)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
