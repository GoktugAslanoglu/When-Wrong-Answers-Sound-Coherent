# GPT-4o blind trajectory-judge analysis

## Data integrity

All four JSONL files parsed successfully. They contain 3,970 abductive and 300 control rows per candidate model (8,540 judgments total). Every row has a finite score on the preregistered 0.2 grid and a non-empty rationale. The Qwen and Llama files are aligned one-to-one on sample, topic, template, condition, prompt, and gold answer.

The abductive set comprises 397 source items, each repeated across five templates and two conditions. The control set comprises 30 source items under the same 5 × 2 repeated-measures structure. Confidence intervals and hypothesis tests below therefore use the source item as the clustering unit.

## Headline descriptive results

| Split | Model | Condition | n | Mean score [95% CI] | Accuracy [95% CI] | Parse errors | Score ≥ 0.8 |
|---|---|---:|---:|---:|---:|---:|---:|
| abductive | Llama-3.1-8B | A | 1985 | 0.640 [0.625, 0.655] | 0.577 [0.543, 0.611] | 0.094 | 0.519 |
| abductive | Llama-3.1-8B | B | 1985 | 0.732 [0.717, 0.746] | 0.634 [0.587, 0.681] | 0.019 | 0.704 |
| abductive | Qwen2.5-7B | A | 1985 | 0.720 [0.707, 0.734] | 0.682 [0.649, 0.715] | 0.018 | 0.681 |
| abductive | Qwen2.5-7B | B | 1985 | 0.776 [0.765, 0.788] | 0.754 [0.711, 0.796] | 0.000 | 0.811 |
| control | Llama-3.1-8B | A | 150 | 0.805 [0.781, 0.829] | 0.900 [0.862, 0.938] | 0.193 | 0.773 |
| control | Llama-3.1-8B | B | 150 | 0.793 [0.779, 0.808] | 0.933 [0.898, 0.969] | 0.207 | 0.780 |
| control | Qwen2.5-7B | A | 150 | 0.995 [0.990, 1.000] | 1.000 [1.000, 1.000] | 0.000 | 1.000 |
| control | Qwen2.5-7B | B | 150 | 0.992 [0.983, 1.001] | 1.000 [1.000, 1.000] | 0.000 | 0.993 |

## Paired abductive comparisons

Across conditions and templates, Qwen exceeded Llama by 0.062 trajectory-score points (95% CI 0.051 to 0.074; item-clustered paired t-test p <.001) and by 0.112 in answer accuracy (95% CI 0.075 to 0.149; p <.001).

Condition B changed Llama's mean trajectory score by 0.092 (95% CI 0.078 to 0.105; p <.001) and Qwen's by 0.056 (95% CI 0.046 to 0.066; p <.001). The model × condition interaction was -0.035 (95% CI -0.052 to -0.019; p <.001).

For answer accuracy, the B − A change was 0.057 for Llama (p <.001) and 0.072 for Qwen (p <.001).

## Relationship between reasoning quality and answer correctness

| Split | Model | Mean score: correct | Mean score: incorrect | Gap | AUC | Incorrect with score ≥ 0.8 | Correct with score ≤ 0.4 |
|---|---|---:|---:|---:|---:|---:|---:|
| abductive | Llama-3.1-8B | 0.757 | 0.576 | 0.181 | 0.718 | 0.400 | 0.155 |
| abductive | Qwen2.5-7B | 0.786 | 0.654 | 0.132 | 0.679 | 0.560 | 0.094 |
| control | Llama-3.1-8B | 0.860 | 0.128 | 0.732 | 0.919 | 0.040 | 0.149 |
| control | Qwen2.5-7B | 0.993 | NA | NA | NA | NA | 0.003 |

AUC measures how well the trajectory score ranks correct above incorrect answers. The non-zero rate of high scores among incorrect answers is expected for an outcome-independent trajectory rubric: a coherent reasoning path can still end with the wrong option. Accordingly, trajectory quality and task accuracy should be reported as related but distinct outcomes.

## Template sensitivity

Averaged across models and conditions, the mean within-item range across the five prompt templates was 0.206 trajectory-score points (95% CI 0.197 to 0.215). Template-level results are in `template_summary.csv`; this variability should be modeled or clustered rather than treating templates as independent samples.

## Control interpretation

Qwen answered all 300 control prompts correctly; Llama accuracy was 0.917. The judge assigned a perfect score to 0.973 of Qwen controls and 0.740 of Llama controls. This supports basic judge sensitivity, while Llama's 60 control parse errors explain much of its lower control performance and should be disclosed.

## Recommended paper claims

1. Report GPT-4o trajectory scores and exact-answer accuracy as separate dependent variables.
2. Describe the judge as blind to the stored gold label and model/condition identity, and state the six-level rubric explicitly.
3. Use item-clustered paired inference because every source item appears under ten prompt variants per model.
4. Treat the controls as a sanity check, not as part of the abductive-effect estimate.
5. Include the complete prompt, model snapshot/date, structured-output schema, retry procedure, and all four result files in the reproducibility package.

## Output files

- `analysis_summary.json`: machine-readable audit and statistics
- `group_summary.csv`: model × condition summaries
- `template_summary.csv`: template-level summaries
- `paired_cluster_tests.csv`: clustered paired tests and effect sizes
- `correctness_association.csv`: score/correctness relationship
- `score_distribution.csv`: full six-level score distribution
- `diagnostic_summary.csv`: parse-error and marker diagnostics
- `figure_condition_model.png` and `figure_score_distribution.png`: paper-ready figures
