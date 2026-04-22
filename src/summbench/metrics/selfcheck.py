from __future__ import annotations

import numpy as np
from nltk import sent_tokenize

from summbench.nltk_utils import ensure_nltk_resources


class SelfCheckMetrics:
    """[DEPRECATED] SelfCheck-based metrics taken from the original notebook."""

    def __init__(self, bertscore_lang: str = "en") -> None:
        self.bertscore_lang = bertscore_lang
        self._official_model = None

    def compute_official_score(self, main_answer: str, samples: list[str]) -> float | None:
        ensure_nltk_resources()
        model = self._load_official_model()
        if model is None:
            return None

        main_sentences = sent_tokenize(main_answer, language="english")
        if not main_sentences:
            return None

        sentence_scores = model.predict(
            sentences=main_sentences,
            sampled_passages=samples,
        )
        return float(np.mean(sentence_scores))

    def compute_custom_score(self, main_answer: str, samples: list[str]) -> float | None:
        ensure_nltk_resources()
        main_sentences = sent_tokenize(main_answer)
        if not main_sentences or not samples:
            return None

        try:
            from bert_score import score as bertscore
        except ImportError as exc:
            raise ImportError("bert-score is not installed. Run: pip install -e .") from exc

        all_sentence_scores: list[float] = []

        for answer_sentence in main_sentences:
            sample_max_scores: list[float] = []

            for sample in samples:
                sample_sentences = sent_tokenize(sample)
                if not sample_sentences:
                    continue

                _, _, f1_scores = bertscore(
                    [answer_sentence] * len(sample_sentences),
                    sample_sentences,
                    lang=self.bertscore_lang,
                    rescale_with_baseline=True,
                )
                sample_max_scores.append(float(max(f1_scores)))

            if sample_max_scores:
                hallucination_score = 1 - float(np.mean(sample_max_scores))
                all_sentence_scores.append(hallucination_score)

        if not all_sentence_scores:
            return None
        return float(np.mean(all_sentence_scores))

    def _load_official_model(self):
        if self._official_model is not None:
            return self._official_model

        try:
            from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
        except ImportError:
            return None

        self._official_model = SelfCheckBERTScore(rescale_with_baseline=True)
        return self._official_model

