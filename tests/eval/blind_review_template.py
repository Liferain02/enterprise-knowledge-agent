"""Deep Research V2 独立盲评评分文件协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BlindAlias = Literal["候选甲", "候选乙"]


@dataclass(frozen=True)
class CandidateScore:
    correctness: int
    completeness: int
    evidence: int

    def validate(self) -> None:
        for value in (self.correctness, self.completeness, self.evidence):
            if value not in (0, 1, 2):
                raise ValueError("正确性、完整性、证据依据必须是 0～2 的整数")

    @property
    def total(self) -> int:
        return self.correctness + self.completeness + self.evidence
