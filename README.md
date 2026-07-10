# serenity

上 Yarrow 平台的**信念蒸馏预测 agent**：把 X 账号 [@aleabitoreddit](https://x.com/aleabitoreddit)
（"Serenity"，AI/半导体供应链分析师）的公开推文蒸馏成结构化认知档案（信念原语 +
因果模型 + 历史 call），作为**信源/先验**注入预测管线——不是角色扮演。

三臂对照实验（方法论验证：人格蒸馏对 Brier 有没有边际贡献）：

```
题目 → 三态领域闸门（in_domain | adjacent | out_of_domain，fail-closed→弃权）
     → ① generic 臂（对照）：双模型 GenericAnalyst + ReferenceClass 外视角，log-odds 融合
     → ② placebo 臂（负控制）：随机信念喂同一先验流程，shadow 落库
     → ③ serenity 臂（提交）：检索信念 → 方向+强度档位 → δ 网格（|δ|≤0.35，
        adjacent 减半，先缩放后封顶）→ sigmoid(logit(generic)+δ)
     → 三臂归因落库 → 选择性提交（带 report.reasoning，引信念转述）
```

结论口径：serenity vs generic 与 serenity vs placebo 的**配对 Brier 差聚类
bootstrap 95% CI**（样本 ≥60，纯前向；两个对比同向为负才算方法论成立）。

## 本地用法

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp .env.example .env          # 填 NEW_API_KEY；YARROW 身份用 bootstrap 生成

# 语料（本地 clone，评审 9A：推文原文不进本 repo）
git clone --depth 1 https://github.com/yan-labs/serenity-aleabitoreddit ~/corpora/serenity-aleabitoreddit

.venv/bin/serenity status                # 公开只读健康检查
.venv/bin/serenity distill               # 一次性蒸馏 → 信念库（实验期重建被拒，--force 开新段）
.venv/bin/serenity inventory             # M0 题源盘点：闸门三态分布 + 结算周期
.venv/bin/serenity bootstrap             # SIWE 持久钱包换 API key，写回 .env
.venv/bin/serenity daily --max 5         # 三臂 dry-run（落库不提交）
.venv/bin/serenity daily --max 10 --submit
.venv/bin/serenity reconcile             # 拉结算真值
.venv/bin/serenity stats                 # 三臂配对报告（聚类 bootstrap CI）
.venv/bin/serenity inspect <qid>         # 单题三臂明细 + 先验归因
```

测试：`.venv/bin/pytest`。

## 实验纪律（评审定稿，实验期冻结）

- **信念库版本冻结**：`belief_set_version`（内容 hash）随每条 forecast 落库；
  配对样本满 60 前 `distill` 重建被拒，`--force` = 开新实验段（样本不合并）。
- **参数冻结**：gate 判定 / δ 网格 / 封顶 / 缩放系数与信念库同段冻结，改 = 新段。
- **fail-closed**：gate/prior 的 LLM 输出解析失败 → 重试 1 次 → 弃权/δ=0，
  `parse_errors` 落库；单题隔离。
- **提交规则（就三条）**：二元题 + 闸门 ∈ {in, adjacent} + generic 自检门通过。
  不做置信度/edge 过滤（防配对样本选择偏差）。
- **historical_claims 仅描述性**：v1 缩放系数固定 1.0，不进生产 δ 公式。

## GitHub Actions

`.github/workflows/daily.yml` 每天 23:00 UTC（北京次日 07:00）自动跑
`daily --submit` → `reconcile` → `calibrate`，状态库 `state/serenity.db`
（含预测 + 信念库三表）commit 回 repo。

Secrets：`NEW_API_KEY` / `YARROW_API_KEY`（bootstrap 生成）/ `SEARX_TOKEN`（可选）。
Settings → Actions → Workflow permissions 选 **Read and write**。

> ⚠️ 私钥（`YARROW_PRIVATE_KEY`）不进 Actions secrets——daemon 只用 API key。
> ⚠️ 正式后端 `api.yarrowlab.ai`：forecast 必须带 `report.reasoning`（否则 422）；
> 同题重提交是覆盖语义（daily 已做 3 天窗口去重）。

## 关键文档

设计文档（3 轮评审 + Codex 盲审定稿）：
`~/.gstack/projects/poly-agent/damon-nogit-design-20260710-164207.md`
