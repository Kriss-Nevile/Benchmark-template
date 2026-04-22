from __future__ import annotations


def combine_hallucination_signals(
    source_f1: float | None,
    reference_f1: float | None,
    official_selfcheck_score: float | None,
    source_weight: float = 0.3,
    reference_weight: float = 0.3,
    selfcheck_weight: float = 0.4,
) -> float | None:
    """[DEPRECATED] Combine the main hallucination signals using the notebook formula."""
    if source_f1 is None or reference_f1 is None or official_selfcheck_score is None:
        return None

    return (
        source_weight * (1 - source_f1)
        + reference_weight * (1 - reference_f1)
        + selfcheck_weight * official_selfcheck_score
    )

