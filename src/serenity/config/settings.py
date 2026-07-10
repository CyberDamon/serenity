"""运行时配置，从环境变量 + .env 加载（环境变量优先）。

用法：`from serenity.config import settings`

Phase 0 只需要 Yarrow 段 + DATABASE_URL。LLM / 新闻 / 成本段留给 Phase 1+，
此处先声明好字段，避免后续到处改。凭证一律走环境变量，绝不写进可提交的文件。
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Yarrow ──
    # 正式站 yarrowlab.ai 的后端是 api.yarrowlab.ai（与文档默认的
    # yarrow-api.littlepinkpotato.com 是两套独立后端：不同 DB、不同 question ID、
    # 不同 api-key）。提交须打正式后端，否则不显示在 yarrowlab.ai 页面。
    yarrow_base_url: str = Field(default="https://api.yarrowlab.ai")
    yarrow_api_key: SecretStr = Field(default=SecretStr(""))
    yarrow_wallet_address: str = Field(default="")
    # 仅一次性 SIWE bootstrap 用；长跑 daemon 应留空并只用 api key。
    yarrow_private_key: SecretStr = Field(default=SecretStr(""))
    # 定向类别循环扫描（QA ISSUE-002）：Yarrow 全局题流被体育/加密梗题刷屏，
    # 真题聚在这些类别里（finance 含财报电话会题——Serenity 相邻领域）。
    # 空 = 全局扫（兜底）。
    yarrow_categories: str = Field(default="finance,tech,politics")
    yarrow_max_questions_per_run: int = Field(default=10)

    # ── serenity 信念先验（设计文档定稿参数；实验期冻结，改 = 新实验段，评审 7A）──
    # 语料：本地 clone 的推文归档（不进 repo，评审 9A）
    corpus_json_path: str = Field(
        default="~/corpora/serenity-aleabitoreddit/data/aleabitoreddit_tweets.json"
    )
    distill_model: str = Field(default="gpt-5.5")  # 蒸馏批处理（一次性）
    gate_model: str = Field(default="gpt-5.5")  # 闸门裁决（便宜、量大）
    prior_model: str = Field(default="claude-opus-4-7")  # 先验推理（质量优先）
    # δ 网格：weak/moderate/strong → ±grid[0]/[1]/[2]（log-odds）；adjacent 减半
    prior_delta_grid: str = Field(default="0.10,0.20,0.35")
    prior_adjacent_factor: float = Field(default=0.5)
    # v1 固定 1.0（评审 8A：historical_claims 降级描述性）；∈(0,1]，先缩放后封顶
    prior_scale: float = Field(default=1.0)
    prior_retrieval_top_k: int = Field(default=12)
    # 实验完整性：配对样本满该数前 distill 重建被拒（评审 3A）
    experiment_min_paired: int = Field(default=60)
    # out_of_domain 题的 generic shadow 抽样上限/轮（闸门误判复盘用；控成本）
    out_shadow_sample: int = Field(default=2)

    # ── LLM 网关（Phase 1+）──
    new_api_url: str = Field(default="https://new-api.100xsoon.com")
    new_api_key: SecretStr = Field(default=SecretStr(""))
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    # ensemble：每框架跨这些异构 frontier 模型各跑一次（D7 存活几何平均）。
    # 逗号分隔，均须在 MODEL_REGISTRY 中。
    # 注：new-api 网关（地缘政治预测agent产品分组）当前仅服务 opus-4-7 + gpt-5.5；
    # opus-4-8/sonnet-5 已登记进 MODEL_REGISTRY 但网关未上线，勿设为默认（会 503）。
    ensemble_models: str = Field(default="claude-opus-4-7,gpt-5.5")
    llm_frontier_model: str = Field(default="claude-opus-4-7")
    llm_legacy_model: str = Field(default="claude-3-5-sonnet-20240620")

    # ── 新闻源（Phase 2+）──
    # SearX（自建 metasearch，Yarrow 共享盒子）：免费、无按次成本、无硬性限流，
    # 现为主检索后端（2026-07-07 起替代 Tavily——后者套餐用量超限 432 fail-closed）。
    searx_base_url: str = Field(default="https://api.yarrowlab.ai")
    searx_token: SecretStr = Field(default=SecretStr(""))
    # 保留：Tavily/Brave 曾是主/备检索源，现降级为未配置时的可选项（不再是默认路径）。
    tavily_api_key: SecretStr = Field(default=SecretStr(""))
    brave_api_key: SecretStr = Field(default=SecretStr(""))

    # ── 存储 ──
    database_url: str = Field(default="sqlite:///serenity.db")

    # ── 聚合/门控 ──
    # 单一分歧阈值（logit 空间）：aggregator 的 disagreement 判定与 runner 提交门共用，
    # 避免双轨漂移。≈ 概率空间 std 0.15。
    framework_logit_std_filter: float = Field(default=1.1)

    # ── 成本上限 ──
    daily_llm_cost_cap_usd: float = Field(default=40.0)

    @property
    def yarrow_category_set(self) -> set[str]:
        return {c.strip().lower() for c in self.yarrow_categories.split(",") if c.strip()}

    @property
    def ensemble_model_list(self) -> list[str]:
        return [m.strip() for m in self.ensemble_models.split(",") if m.strip()]

    @property
    def prior_delta_grid_values(self) -> tuple[float, float, float]:
        """weak/moderate/strong 对应的 |δ|（log-odds）。"""
        parts = [float(x) for x in self.prior_delta_grid.split(",")]
        if len(parts) != 3:
            raise ValueError(f"prior_delta_grid 需要 3 个值: {self.prior_delta_grid!r}")
        return parts[0], parts[1], parts[2]


settings = Settings()
