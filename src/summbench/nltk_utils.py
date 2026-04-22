from __future__ import annotations

import nltk

_RESOURCE_MAP = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_resources() -> None:
    """Download the minimum NLTK resources if they are missing."""
    for resource_name, resource_path in _RESOURCE_MAP.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download NLTK resource '{resource_name}'. "
                    "Please install it manually before running the benchmark."
                ) from exc

