# ACTP supplementary reproducibility artifact

This artifact contains the row-aligned data and analysis code used in the revised paper.

## Contents

- `data/gpt4o_blind/`: blind GPT-4o judgments. Each JSONL row preserves the prompt, raw candidate output, parsed reasoning and answer, correctness, parse flag, trajectory score, and judge rationale.
- `data/qwen_reference_aware/`: secondary reference-aware Qwen2.5-32B-Instruct judgments used only for the additional judge-sensitivity analysis.
- `scripts/`: artifact-relative analysis programs plus the credential-safe Batch API judge pipeline used for the blind GPT-4o evaluation.
- `expected_outputs/`: paper-ready reports and key CSV outputs from the verified run.

## Environment

Python 3.10 or newer. Exact versions from the verified rerun are pinned in `requirements.txt`.

## Commands

From the artifact root:

```bash
python scripts/inspect_structure.py
python scripts/analyze_results.py
python scripts/analyze_forced_condition.py
python scripts/cross_judge_analysis.py
```

The scripts write detailed reports, CSV tables, and figures beside the scripts. Reproducing API inference is not required to reproduce the reported statistical analysis because all row-level judge outputs and rationales are included.

## Recreate judge requests

The released `scripts/openai_judge_pipeline.py` freezes the blind rubric, prepares resumable OpenAI Batch API requests, validates responses, retries unresolved requests, and merges judgments in source-row order. Preparing a pilot makes no API calls:

```bash
python scripts/openai_judge_pipeline.py prepare --project-root . --run-dir judge_pilot --model gpt-4o --mode blind --limit-per-dataset 1
```

The `run` command requires an `OPENAI_API_KEY` environment variable and paid API access. The script never accepts or prints a key. Existing row-level judgments are included, so API access is not required to reproduce the paper's statistical analysis.

## Recreate judge requests

The released `scripts/openai_judge_pipeline.py` freezes the blind rubric, prepares resumable OpenAI Batch API requests, validates responses, retries unresolved requests, and merges judgments in source-row order. Preparing a pilot makes no API calls:

```bash
python scripts/openai_judge_pipeline.py prepare --project-root . --run-dir judge_pilot --model gpt-4o --mode blind --limit-per-dataset 1
```

The `run` command requires an `OPENAI_API_KEY` environment variable and paid API access. The script never accepts or prints a key. Existing row-level judgments are included, so API access is not required to reproduce the paper's statistical analysis.

## Security

No API keys, access tokens, or credentials are included. The original exploratory notebook is intentionally excluded.