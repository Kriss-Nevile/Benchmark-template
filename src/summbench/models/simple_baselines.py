from __future__ import annotations

from nltk import sent_tokenize

from summbench.nltk_utils import ensure_nltk_resources

from .base import SummarizationModel


class LeadSentenceBaseline(SummarizationModel):
    """A tiny baseline that returns the first few sentences of the article."""

    def __init__(self, sentence_count: int = 2) -> None:
        super().__init__(name="lead-sentence-baseline")
        self.sentence_count = sentence_count

    def generate_summary(self, source: str, temperature: float = 0.0) -> str:
        del temperature
        ensure_nltk_resources()
        sentences = sent_tokenize(source)
        return " ".join(sentences[: self.sentence_count]).strip()

