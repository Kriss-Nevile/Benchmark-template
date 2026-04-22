from __future__ import annotations

import os
import re

from summbench.types import JudgeResult


class GeminiHallucinationJudge:
    """Optional Gemini-based LLM judge for hallucination detection."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI")
        self.temperature = temperature
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set the GEMINI environment variable first."
            )

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. Run: pip install -e .[judge]"
            ) from exc

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def evaluate(self, context: str, summary: str) -> JudgeResult:
        model = self._load_model()
        response = model.generate_content(
            self._build_prompt(context=context, summary=summary),
            generation_config={"temperature": self.temperature},
        )
        full_response = response.text.strip()
        return JudgeResult(
            explanation=full_response,
            is_hallucinated=self._parse_boolean_label(full_response),
        )

    @staticmethod
    def _build_prompt(context: str, summary: str) -> str:
        return f"""You are an expert fact-checker. Your task is to determine if a SUMMARY contains hallucinations.

DEFINITIONS:
- Hallucination = Information that is NOT present in the source OR contradicts the source
- NOT a hallucination = Information that is directly supported by or reasonably inferred from the source

INSTRUCTIONS:
1. Compare the SUMMARY against the SOURCE CONTEXT carefully
2. Identify any claims in the summary not supported by the source
3. Provide your reasoning in 2-3 short sentences
4. End with EXACTLY one of these two labels on the last line:
   - HALLUCINATED: True
   - HALLUCINATED: False

SOURCE CONTEXT:
\"\"\"{context}\"\"\"

SUMMARY TO EVALUATE:
\"\"\"{summary}\"\"\"
"""

    @staticmethod
    def _parse_boolean_label(full_response: str) -> bool | None:
        match = re.search(r"HALLUCINATED:\s*(True|False)", full_response, re.IGNORECASE)
        if not match:
            return None
        return match.group(1).lower() == "true"

