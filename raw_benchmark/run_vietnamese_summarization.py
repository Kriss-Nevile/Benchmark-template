import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from summbench.models import HuggingFaceSeq2SeqSummarizer

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
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required. Run: pip install datasets") from exc

    dataset = load_dataset("Tuan-NT/vietnamese-summarization", split="test")
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    print(f"Loaded {len(dataset)} examples from test split.")

    print(f"Initializing model: {model_name}")
    model = HuggingFaceSeq2SeqSummarizer(
        model_name=model_name,
        max_new_tokens=1024,
        torch_dtype="float16",
        device_map="auto"
    )

    if args.output_dir is None:
        out_dir = PROJECT_ROOT / "outputs" / f"vietnamese_{args.model}_benchmark"
    else:
        out_dir = Path(args.output_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting inference, saving to {out_dir}")
    
    # We will only keep original columns + prediction
    results = []
    
    for i, row in enumerate(tqdm(dataset, desc=f"Benchmarking {args.model}")):
        # row contains: 'id', 'document', 'summary', 'source'
        prediction = model.generate_summary(row["document"], temperature=0.0)
        
        # Create a new dict with original data + prediction
        result_row = dict(row)
        result_row["prediction"] = prediction
        results.append(result_row)
        
        if (i + 1) % args.save_every == 0:
            pd.DataFrame(results).to_csv(out_dir / "predictions_partial.csv", index=False)
            
    pd.DataFrame(results).to_csv(out_dir / "predictions.csv", index=False)
    print("Benchmark finished!")

if __name__ == "__main__":
    main()
