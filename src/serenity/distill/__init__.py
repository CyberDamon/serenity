"""蒸馏管线：本地推文语料 → 三张信念表（一次性批处理 + 版本冻结）。"""

from serenity.distill.pipeline import DistillReport, run_distill

__all__ = ["DistillReport", "run_distill"]
