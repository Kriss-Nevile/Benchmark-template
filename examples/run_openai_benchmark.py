from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from summbench.dataset import load_dataset_from_csv
from summbench.metrics import GeminiHallucinationJudge
from summbench.models import OpenAIChatSummarizer
from summbench.pipeline import BenchmarkRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv()
    dataset = load_dataset_from_csv(PROJECT_ROOT / "data" / "sample_dataset.csv")

    model = OpenAIChatSummarizer(model_name="gpt-4o-mini")
    judge = GeminiHallucinationJudge(model_name="gemini-2.5-flash")
    runner = BenchmarkRunner(model=model, judge=judge)
    runner.run(
        examples=dataset,
        output_dir=PROJECT_ROOT / "outputs" / "openai_example",
    )


if __name__ == "__main__":
    main()

