import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from summbench.dataset import load_dataset_from_huggingface
from summbench.models import HuggingFaceSeq2SeqSummarizer
from summbench.pipeline import BenchmarkRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class VietnameseBenchmarkRunner(BenchmarkRunner):
    """Custom runner that renames 'model_summary' to 'predictions' during saves."""
    
    @staticmethod
    def _save_partial(results: list[dict], output_dir: Path) -> None:
        df = pd.DataFrame(results)
        if "model_summary" in df.columns:
            df.rename(columns={"model_summary": "predictions"}, inplace=True)
        df.to_csv(output_dir / "detailed_results_partial.csv", index=False)

    @staticmethod
    def _save_final(dataframe: pd.DataFrame, output_dir: Path, total_time: float) -> None:
        df = dataframe.copy()
        if "model_summary" in df.columns:
            df.rename(columns={"model_summary": "predictions"}, inplace=True)
            
        df.to_csv(output_dir / "detailed_results.csv", index=False)

        numeric_only = df.select_dtypes(include=["number", "bool"])
        summary = {
            "rows_evaluated": int(len(df)),
            "total_time_seconds": total_time,
            "average_time_per_row_seconds": total_time / len(df) if len(df) else 0.0,
            "metric_means": numeric_only.mean(numeric_only=True).to_dict(),
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run Vietnamese Summarization Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        choices=["vit5", "bartpho"],
        default="vit5",
        help="Which model to benchmark: 'vit5' or 'bartpho'",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="How often to save intermediate results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to PROJECT_ROOT/outputs/<model_name>",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples for testing. Set to None to run all.",
    )
    args = parser.parse_args()

    # Model specific configuration
    if args.model == "vit5":
        model_name = "VietAI/vit5-large"
    else:
        model_name = "vinai/bartpho-word"

    print(f"Loading dataset: Tuan-NT/vietnamese-summarization")
    dataset = load_dataset_from_huggingface(
        dataset_name="Tuan-NT/vietnamese-summarization",
        split="test",
        article_column="document",
        reference_column="summary",
        limit=args.limit
    )
    print(f"Loaded {len(dataset)} examples from test split.")

    print(f"Initializing model: {model_name}")
    model = HuggingFaceSeq2SeqSummarizer(
        model_name=model_name,
        max_new_tokens=1024,
        torch_dtype="float16", # Usually better to use fp16 for these large models if on GPU
        device_map="auto"
    )

    runner = VietnameseBenchmarkRunner(model=model)

    if args.output_dir is None:
        out_dir = PROJECT_ROOT / "outputs" / f"vietnamese_{args.model}_benchmark"
    else:
        out_dir = Path(args.output_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting benchmark, saving to {out_dir}")
    runner.run(
        examples=dataset,
        output_dir=out_dir,
        save_every=args.save_every
    )
    print("Benchmark finished!")

if __name__ == "__main__":
    main()
