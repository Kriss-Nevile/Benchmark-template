from __future__ import annotations

from nltk import sent_tokenize

from summbench.nltk_utils import ensure_nltk_resources


class SentenceSimilarityMetric:
    """Sentence-level similarity metric adapted from the notebook."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. Run: pip install -e ."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score(self, source: str, summary: str) -> tuple[float, float, float]:
        ensure_nltk_resources()
        model = self._load_model()

        source_sentences = sent_tokenize(source)
        summary_sentences = sent_tokenize(summary)
        if not source_sentences or not summary_sentences:
            return 0.0, 0.0, 0.0

        try:
            from sentence_transformers import util
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install -e ."
            ) from exc

        source_embeddings = model.encode(
            source_sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        summary_embeddings = model.encode(
            summary_sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        cosine_similarity = util.cos_sim(summary_embeddings, source_embeddings)

        precision = float(cosine_similarity.max(dim=1).values.mean().item())
        recall = float(cosine_similarity.max(dim=0).values.mean().item())
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = float(2 * (precision * recall) / (precision + recall))
        return precision, recall, f1


class BERTScoreMetric:
    """Standalone BERTScore metric."""

    def __init__(self, lang: str = "en", rescale_with_baseline: bool = True) -> None:
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline

    def score(self, source: str, summary: str) -> tuple[float, float, float]:
        try:
            from bert_score import score as bertscore
        except ImportError as exc:
            raise ImportError("bert-score is not installed. Run: pip install -e .") from exc

        if not source.strip() or not summary.strip():
            return 0.0, 0.0, 0.0

        p, r, f1 = bertscore(
            [summary],
            [source],
            lang=self.lang,
            rescale_with_baseline=self.rescale_with_baseline,
        )
        return float(p[0]), float(r[0]), float(f1[0])

