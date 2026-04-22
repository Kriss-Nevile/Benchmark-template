from __future__ import annotations

import os

from .base import SummarizationModel


class OpenAIChatSummarizer(SummarizationModel):
    """OpenAI chat model adapter for the benchmark runner."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        super().__init__(name=model_name)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    def load(self) -> None:
        if self._client is not None:
            return

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY before running the example."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Run: pip install -e .[openai]"
            ) from exc

        self._client = OpenAI(api_key=self.api_key)

    def generate_summary(self, source: str, temperature: float = 0.0) -> str:
        self.load()
        assert self._client is not None

        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": source},
            ],
        )
        return response.choices[0].message.content.strip()

