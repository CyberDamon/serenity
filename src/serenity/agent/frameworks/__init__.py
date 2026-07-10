"""serenity frameworks 包：generic 对照臂的两个组件。

green-water 的 8 个 IR 框架 + router 已剥除（评审 1A：对照臂 = 双模型
GenericAnalyst inside view + ReferenceClass 外视角锚，log-odds 融合）。
"""

from serenity.agent.frameworks.base import Framework, FrameworkOutput, Market
from serenity.agent.frameworks.generic_analyst import GenericAnalyst
from serenity.agent.frameworks.reference_class import ReferenceClass

METHODOLOGICAL_FRAMEWORKS: tuple[type[Framework], ...] = (ReferenceClass,)

__all__ = [
    "Framework",
    "FrameworkOutput",
    "GenericAnalyst",
    "Market",
    "METHODOLOGICAL_FRAMEWORKS",
    "ReferenceClass",
]
