from __future__ import annotations

from nltk import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from summbench.nltk_utils import ensure_nltk_resources


def compute_meteor_score(reference: str, candidate: str) -> float:
    ensure_nltk_resources()
    reference_tokens = [word_tokenize(reference.lower())]
    candidate_tokens = word_tokenize(candidate.lower())
    return float(meteor_score(reference_tokens, candidate_tokens))


def compute_rouge_scores(reference: str, candidate: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    return {
        "rouge1_f1": float(rouge_scores["rouge1"].fmeasure),
        "rouge2_f1": float(rouge_scores["rouge2"].fmeasure),
        "rougeL_f1": float(rouge_scores["rougeL"].fmeasure),
    }

