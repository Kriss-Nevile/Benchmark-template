from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from summbench.metrics.composite import combine_hallucination_signals
from summbench.metrics.lexical import compute_meteor_score, compute_rouge_scores
from summbench.metrics.semantic import SentenceSimilarityMetric
from summbench.metrics.selfcheck import SelfCheckMetrics
from summbench.models.base import SummarizationModel
from summbench.types import DatasetExample


class BenchmarkRunner:
    """Runs the full benchmark loop for any model adapter."""

    def __init__(
        self,
        model: SummarizationModel,
        judge=None,
        sample_count_for_selfcheck: int = 3,
        main_temperature: float = 0.0,
        sample_temperature: float = 0.8,
    ) -> None:
        self.model = model
        self.judge = judge
        self.sample_count_for_selfcheck = sample_count_for_selfcheck
        self.main_temperature = main_temperature
        self.sample_temperature = sample_temperature
        self.selfcheck = SelfCheckMetrics()
        self.sentence_similarity = SentenceSimilarityMetric()

    def run(
        self,
        examples: list[DatasetExample],
        output_dir: str | Path,
        save_every: int = 25,
    ) -> pd.DataFrame:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.load()

        results: list[dict] = []
        started_at = time.time()

        try:
            for offset, example in enumerate(tqdm(examples, desc=f"Benchmarking {self.model.name}"), start=1):
                row = self._evaluate_one(example)
                results.append(row)

                if offset % save_every == 0:
                    self._save_partial(results=results, output_dir=output_path)
        finally:
            self.model.close()

        dataframe = pd.DataFrame(results)
        self._save_final(
            dataframe=dataframe,
            output_dir=output_path,
            total_time=time.time() - started_at,
        )
        return dataframe

    def _evaluate_one(self, example: DatasetExample) -> dict:
        model_summary = self.model.generate_summary(
            example.article,
            temperature=self.main_temperature,
        )
        samples = self.model.generate_samples(
            example.article,
            count=self.sample_count_for_selfcheck,
            temperature=self.sample_temperature,
        )

        official_selfcheck = self.selfcheck.compute_official_score(model_summary, samples)
        custom_selfcheck = self.selfcheck.compute_custom_score(model_summary, samples)

        _, _, source_f1 = self.sentence_similarity.score(example.article, model_summary)
        _, _, reference_f1 = self.sentence_similarity.score(
            example.reference_summary,
            model_summary,
        )

        row = {
            "sample_id": example.sample_id,
            "model_name": self.model.name,
            "source_length": len(example.article.split()),
            "reference_length": len(example.reference_summary.split()),
            "model_summary_length": len(model_summary.split()),
            "article": example.article,
            "reference_summary": example.reference_summary,
            "model_summary": model_summary,
            "official_selfcheck_score": official_selfcheck,
            "custom_selfcheck_score": custom_selfcheck,
            "source_f1": source_f1,
            "reference_f1": reference_f1,
            "hallucination_score": combine_hallucination_signals(
                source_f1=source_f1,
                reference_f1=reference_f1,
                official_selfcheck_score=official_selfcheck,
            ),
        }

        row.update(compute_rouge_scores(example.reference_summary, model_summary))
        row["meteor_score"] = compute_meteor_score(example.reference_summary, model_summary)

        if self.judge is not None:
            judge_result = self.judge.evaluate(example.article, model_summary)
            row["hallucination_judge"] = judge_result.is_hallucinated
            row["hallucination_explanation"] = judge_result.explanation

        return row

    @staticmethod
    def _save_partial(results: list[dict], output_dir: Path) -> None:
        pd.DataFrame(results).to_csv(output_dir / "detailed_results_partial.csv", index=False)

    @staticmethod
    def _save_final(dataframe: pd.DataFrame, output_dir: Path, total_time: float) -> None:
        dataframe.to_csv(output_dir / "detailed_results.csv", index=False)

        numeric_only = dataframe.select_dtypes(include=["number", "bool"])
        summary = {
            "rows_evaluated": int(len(dataframe)),
            "total_time_seconds": total_time,
            "average_time_per_row_seconds": total_time / len(dataframe) if len(dataframe) else 0.0,
            "metric_means": numeric_only.mean(numeric_only=True).to_dict(),
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

