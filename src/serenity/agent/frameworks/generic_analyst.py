"""Generic Analyst — generic 对照臂的 inside-view 框架（评审 issue 1A）。

serenity 的实验对照组：一个不带任何领域框架、不带人格先验的通用分析师。
它与 reference_class（外视角锚）一起构成 generic 臂：

  generic 臂 = aggregate( [GenericAnalyst × 每模型],  ReferenceClass 集成锚 )

设计约束（勿改，实验对照组定义）：
  - 不注入任何 Serenity 信念（那是 prior 模块的事，发生在 aggregate 之后）
  - 不用领域理论框架（保持 "generic" 语义）
  - 证据 = 检索到的新闻 + 模型自身知识，仅此而已
"""

from __future__ import annotations

from serenity.agent.frameworks.base import Framework, FrameworkOutput

SYSTEM = """You are a careful, calibrated generalist forecaster analyzing a
prediction market question. You have no special domain framework — reason from
the evidence at hand and general world knowledge.

Method:
  1. State precisely what YES means (disambiguation first).
  2. Weigh the evidence in the news context for and against YES.
  3. Consider the time remaining to the deadline and what must happen for YES.
  4. Give a calibrated probability. Avoid round-number anchoring (0.5/0.9);
     avoid overconfidence — most surprising claims resolve NO.

Output JSON via the submit_prediction tool per the schema.
"""

USER_TEMPLATE = """Market question: {question}
Current YES price on the market: {market_price}
Today's reference date (run date): {as_of_date}
Market resolution date (YES/NO deadline): {resolution_date}

Recent news context:
{news_block}

Steps:
  1. yes_side_interpretation — one sentence, what exactly resolves YES.
  2. Enumerate the strongest evidence FOR and AGAINST yes (from news + knowledge).
  3. Deadline check: is there realistic time left before {resolution_date}?
  4. Final calibrated YES probability with 200-400 word reasoning.

Set contamination_warning=true ONLY if you have specific factual knowledge that
THIS event has ALREADY OCCURRED or ALREADY RESOLVED. When unsure, set false.
"""


class GenericAnalyst(Framework):
    name = "generic_analyst"
    system_prompt = SYSTEM
    user_template = USER_TEMPLATE

    def post_process(self, output: FrameworkOutput, parsed: dict) -> None:
        return
