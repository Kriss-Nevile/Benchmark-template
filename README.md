# Summarization Benchmark Template

This repository turns the ideas from `benchmark.ipynb` and `Benchmarkanalysis.ipynb` into a small Python project that is easier to reuse, extend, and hand to someone new to coding.

## What this template gives you

- One place for metric implementations.
- One place for model adapters.
- One benchmark runner that works with many models.
- Beginner-friendly examples.
- Outputs saved as `.csv` and `.json`.

The template keeps the same main evaluation ideas from the notebooks:

- ROUGE
- METEOR
- sentence-level similarity
- SelfCheck-style scoring
- optional Gemini LLM-as-judge
- combined hallucination score

## Folder layout

```text
summarization-benchmark-template/
|-- data/
|   `-- sample_dataset.csv
|-- examples/
|   |-- run_huggingface_benchmark.py
|   |-- run_lead_sentence_baseline.py
|   `-- run_openai_benchmark.py
|-- outputs/
|   `-- .gitkeep
|-- src/
|   `-- summbench/
|       |-- dataset.py
|       |-- nltk_utils.py
|       |-- types.py
|       |-- metrics/
|       |-- models/
|       |-- pipeline/
|       `-- templates/
`-- tests/
```

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 2. Install the package

Core template only:

```bash
pip install -e .
```

If you want OpenAI, Gemini judge, and Hugging Face examples too:

```bash
pip install -e .[all]
```

### 3. Run the simplest example

This one works with a very simple baseline model and does not need an API key:

```bash
python examples/run_lead_sentence_baseline.py
```

### 4. Use your own dataset

Put a CSV file in `data/` with these columns:

- `article`
- `highlights`

Example:

```csv
article,highlights
"Long article text here","Reference summary here"
```

## How to plug in a new model

There are three easy ways:

### Option 1. Use OpenAI

See `examples/run_openai_benchmark.py`.

### Option 2. Use a Hugging Face causal language model

See `examples/run_huggingface_benchmark.py`.

### Option 3. Make your own adapter

Copy the class in `src/summbench/templates/model_template.py` and fill in:

- `load()`
- `generate_summary()`

That is the only part most users need to customize.

## What gets saved

Each benchmark run writes:

- `detailed_results.csv`
- `summary.json`

The detailed file includes per-example metrics such as:

- `official_selfcheck_score`
- `custom_selfcheck_score`
- `source_f1`
- `reference_f1`
- `hallucination_score`
- `rouge1_f1`
- `rouge2_f1`
- `rougeL_f1`
- `meteor_score`

If a judge is enabled, it also includes:

- `hallucination_judge`
- `hallucination_explanation`

## Notes about dependencies

- `sentence-transformers` and `bert-score` can be slow the first time they run.
- The official SelfCheck score tries to use `selfcheckgpt` if it is installed. If it is not installed, the runner keeps going and leaves that column empty.
- The Gemini judge is optional. You only need it if you want LLM-as-judge hallucination checks.

## Beginner roadmap

If you are new to the project, use this order:

1. Run `examples/run_lead_sentence_baseline.py`
2. Read `src/summbench/models/base.py`
3. Read `src/summbench/templates/model_template.py`
4. Replace the template with your own model
5. Run the benchmark again

## Mapping back to the notebooks

This project mainly refactors code from:

- `benchmark.ipynb`
- `Benchmarkanalysis.ipynb`

The logic was separated into modules so the metrics do not need to be rewritten every time a new model is tested.

