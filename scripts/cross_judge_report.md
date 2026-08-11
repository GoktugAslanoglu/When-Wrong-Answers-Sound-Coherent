# Cross-judge robustness: Qwen2.5-32B vs GPT-4o

All 8,540 rows aligned one-to-one across the original Qwen2.5-32B judge files and the new blind GPT-4o judge files.

| Candidate outputs | Pearson r | Spearman rho | Mean Qwen-judge TS | Mean GPT-4o TS | MAE | Within 0.2 |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.553 | 0.540 | 0.630 | 0.686 | 0.143 | 0.847 |
| Qwen2.5-7B | 0.366 | 0.422 | 0.753 | 0.748 | 0.262 | 0.689 |

## Condition-shift replication

| Candidate outputs | Judge | B - A shift [95% CI] | p |
|---|---|---:|---:|
| Llama-3.1-8B | Qwen2.5-32B | 0.064 [0.051, 0.076] | <.001 |
| Llama-3.1-8B | GPT-4o | 0.092 [0.078, 0.105] | <.001 |
| Qwen2.5-7B | Qwen2.5-32B | 0.061 [0.037, 0.084] | <.001 |
| Qwen2.5-7B | GPT-4o | 0.056 [0.046, 0.066] | <.001 |

The two judges need not agree exactly because they operationalize the rubric differently. The robustness claim should therefore be based on rank/linear agreement and replication of the directional condition effect, not identical absolute scores.
