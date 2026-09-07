# When Wrong Answers Sound Coherent

Reproducibility materials for **“When Wrong Answers Sound Coherent: Evaluating Answer-Conditioned Explanatory Generation in LLMs”** by Göktuğ Aslanoğlu and Viswanadh Vadlamani.

The paper introduces **Answer-Conditioned Trace Probing (ACTP)**, a paired evaluation framework for separating answer correctness from the apparent coherence of an explanation. Condition A lets a model answer and explain freely. Condition B supplies that model’s Condition-A answer as a fixed anchor and asks it to justify the anchor across five narrative prompt templates.

The study evaluates Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct on 397 abductive-reasoning items and 30 arithmetic controls. GPT-4o is the primary blind trajectory judge; Qwen2.5-32B-Instruct is included as a secondary, reference-aware judge-sensitivity analysis.

## Repository status

The analysis code and saved model/judge outputs are sufficient to reproduce the reported statistics without making API calls. This is an analysis-reproduction artifact, not a complete end-to-end generation package: it does not currently include the code and immutable configuration used to generate the original candidate-model responses.

## Contents

```text
.
├── data/
│   ├── gpt4o_blind/              # Primary blind-judge rows
│   └── qwen_reference_aware/     # Secondary judge-sensitivity rows
├── expected_outputs/             # Reference reports and key tables
├── scripts/
│   ├── inspect_structure.py
│   ├── analyze_results.py
│   ├── analyze_forced_condition.py
│   ├── cross_judge_analysis.py
│   └── openai_judge_pipeline.py
├── requirements.txt              # Statistical analysis dependencies
└── requirements-judge.txt        # Optional OpenAI judge dependency
```

Each judge directory contains four JSONL files: abductive and control results for each candidate model. The primary directory contains 8,540 scored rows in total (3,970 abductive and 300 control rows per candidate model); the secondary directory has the same row structure.

Each row records identifiers, condition and template, candidate prompt and output, parsed answer and reasoning, exact-answer correctness, parse diagnostics, trajectory score, and judge rationale. The two candidate-model files align on source item, topic, template, condition, and gold answer. In the abductive split, 710 Condition-B prompts differ across candidate models because 142 source items produced different model-specific Condition-A anchors, each reused across five templates. Condition-B values therefore should not be interpreted as a clean between-model ability comparison.

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/GoktugAslanoglu/When-Wrong-Answers-Sound-Coherent.git
cd When-Wrong-Answers-Sound-Coherent
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the environment with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on macOS/Linux.

## Reproduce the analysis

Run these commands from the repository root:

```bash
python scripts/inspect_structure.py
python scripts/analyze_results.py
python scripts/analyze_forced_condition.py
python scripts/cross_judge_analysis.py
```

The analysis commands write reports, tables, and figures to `reproduced_outputs/`. Reference outputs from the verified run are stored in `expected_outputs/`.

The analysis uses the source item as the clustering unit because each abductive item appears under five templates and two conditions. Exact-answer accuracy and judged trajectory coherence are reported as separate outcomes. For Condition B, consult `reproduced_outputs/forced_condition_report.md` and the corresponding CSV files before making cross-model comparisons.

## Optional: reconstruct GPT-4o judge requests

Re-running the judge is **not** required to reproduce the statistical analysis. To inspect the frozen rubric and create a four-request pilot without contacting the API:

```bash
python -m pip install -r requirements-judge.txt
python scripts/openai_judge_pipeline.py prepare \
  --project-root . \
  --run-dir judge_pilot \
  --model gpt-4o \
  --mode blind \
  --limit-per-dataset 1
```

Submitting a prepared run requires paid OpenAI API access and an `OPENAI_API_KEY` supplied through the process environment:

```bash
python scripts/openai_judge_pipeline.py run --run-dir judge_pilot
```

The pipeline does not accept an API key as a command-line argument and does not print it. Do not commit `.env` files, credentials, Batch API response directories, or copied notebook outputs.

## Data provenance and licensing

The abductive items originate from [SemEval-2026 Task 12: Abductive Event Reasoning](https://github.com/sooo66/semeval2026-task12-dataset). Please cite the task paper in addition to this work:

```bibtex
@article{cao2026aer,
  title   = {SemEval-2026 Task 12: Abductive Event Reasoning: Towards Real-World Event Causal Inference for Large Language Models},
  author  = {Cao, Pengfei and Yang, Mingxuan and Chen, Yubo and Zhang, Chenlong and Liu, Mingxuan and Liu, Kang and Zhao, Jun},
  year    = {2026},
  eprint  = {2603.21720},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url     = {https://arxiv.org/abs/2603.21720}
}
```

## Citation

Until the arXiv identifier is available, cite the manuscript as:

```bibtex
@misc{aslanoglu2026wronganswers,
  title  = {When Wrong Answers Sound Coherent: Evaluating Answer-Conditioned Explanatory Generation in LLMs},
  author = {Aslanoğlu, Göktuğ and Vadlamani, Viswanadh},
  year   = {2026},
  note   = {Accepted to the non-archival track at INLG 2026}
}
```

Replace this entry with the arXiv citation after the preprint is assigned an identifier.

## Security and privacy

No API keys or access tokens are intentionally included. Before every public release, scan both the working tree and Git history for credentials, local paths, private email addresses, and generated run directories. If a credential has ever been exposed, revoke it; deleting it from the latest commit is not sufficient.

## Authors

- Göktuğ Aslanoğlu — Independent Researcher
- Viswanadh Vadlamani — Safe App
