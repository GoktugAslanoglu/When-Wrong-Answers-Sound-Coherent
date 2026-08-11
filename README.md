# ACTP supplementary reproducibility artifact

This artifact contains the row-aligned data and analysis code used in the revised paper.

## Contents

- `data/gpt4o_blind/`: blind GPT-4o judgments. Each JSONL row preserves the prompt, raw candidate output, parsed reasoning and answer, correctness, parse flag, trajectory score, and judge rationale.
- `data/qwen_reference_aware/`: secondary reference-aware Qwen2.5-32B-Instruct judgments used only for the additional judge-sensitivity analysis.
- `scripts/`: analysis programs with artifact-relative input paths.
- `expected_outputs/`: paper-ready reports and key CSV outputs from the verified run.

## Environment

Python 3.10 or newer with `numpy`, `pandas`, `scipy`, and `matplotlib`.

## Commands

From the artifact root:

```bash
python scripts/inspect_structure.py
python scripts/analyze_results.py
python scripts/analyze_forced_condition.py
python scripts/cross_judge_analysis.py
```

The scripts write detailed reports, CSV tables, and figures beside the scripts. Reproducing API inference is not required to reproduce the reported statistical analysis because all row-level judge outputs and rationales are included.

## Security

No API keys, access tokens, or credentials are included. The original exploratory notebook is intentionally excluded.