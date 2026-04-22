from __future__ import annotations

from pathlib import Path

from summbench.dataset import load_dataset_from_csv
from summbench.models import HuggingFaceCausalSummarizer
from summbench.pipeline import BenchmarkRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = load_dataset_from_csv(PROJECT_ROOT / "data" / "sample_dataset.csv")

    model = HuggingFaceCausalSummarizer(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype="float16",
        load_in_4bit=True,
    )
    runner = BenchmarkRunner(model=model)
    runner.run(
        examples=dataset,
        output_dir=PROJECT_ROOT / "outputs" / "huggingface_example",
    )


if __name__ == "__main__":
    main()

