from __future__ import annotations

import csv
from pathlib import Path

from .types import DatasetExample


def load_dataset_from_csv(
    csv_path: str | Path,
    article_column: str = "article",
    reference_column: str = "highlights",
    start_index: int = 0,
    limit: int | None = None,
    indices: list[int] | None = None,
) -> list[DatasetExample]:
    """Load a CSV dataset into a simple list of examples."""
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return []
    if article_column not in rows[0]:
        raise ValueError(f"Missing article column: {article_column}")
    if reference_column not in rows[0]:
        raise ValueError(f"Missing reference column: {reference_column}")

    indexed_rows = list(enumerate(rows))

    if indices is not None:
        selected = [indexed_rows[index] for index in indices]
    else:
        selected = indexed_rows[start_index:]
        if limit is not None:
            selected = selected[:limit]

    examples: list[DatasetExample] = []
    for row_index, row in selected:
        examples.append(
            DatasetExample(
                sample_id=row_index,
                article=str(row[article_column]),
                reference_summary=str(row[reference_column]),
            )
        )
    return examples
