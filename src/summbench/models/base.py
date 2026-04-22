from __future__ import annotations

from abc import ABC, abstractmethod


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful and precise summarization assistant. "
    "Summarize the source text factually and concisely. "
    "Do not add details that are not supported by the source."
)


class SummarizationModel(ABC):
    """Base class for any model that can summarize text."""

    def __init__(self, name: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.name = name
        self.system_prompt = system_prompt

    def load(self) -> None:
        """Load heavy resources only when needed."""

    @abstractmethod
    def generate_summary(self, source: str, temperature: float = 0.0) -> str:
        """Generate one summary for a source article."""

    def generate_samples(
        self,
        source: str,
        count: int,
        temperature: float = 0.8,
    ) -> list[str]:
        samples: list[str] = []
        for _ in range(count):
            samples.append(self.generate_summary(source, temperature=temperature))
        return samples

    def close(self) -> None:
        """Release resources after the benchmark if necessary."""

