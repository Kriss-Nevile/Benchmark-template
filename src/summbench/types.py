from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DatasetExample:
    sample_id: int
    article: str
    reference_summary: str


@dataclass(slots=True)
class JudgeResult:
    explanation: str
    is_hallucinated: bool | None

