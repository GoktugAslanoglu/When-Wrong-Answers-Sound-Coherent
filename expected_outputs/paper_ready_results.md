# Paper-ready interpretation of GPT-4o blind-judge results

## Design and validation

The analysis contains 8,540 complete GPT-4o judgments: 3,970 abductive and
300 control trajectories for each candidate model. The abductive data comprise
397 source items crossed with five prompt templates and two experimental
conditions; controls comprise 30 source items under the same 5 × 2 structure.
All scores are finite values on the six-level 0.0–1.0 rubric and all rationales
are non-empty. Qwen and Llama rows align one-to-one. Inference uses source-item
clustered paired comparisons rather than treating the repeated templates as
independent observations.

## Descriptive abductive results

| Model | Condition | Mean trajectory score [95% CI] | Accuracy [95% CI] | Parse-error rate | Score ≥ 0.8 |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | A | 0.640 [0.625, 0.655] | 0.577 [0.543, 0.611] | 0.094 | 0.519 |
| Llama-3.1-8B | B | 0.732 [0.717, 0.746] | 0.634 [0.587, 0.681] | 0.019 | 0.704 |
| Qwen2.5-7B | A | 0.720 [0.707, 0.734] | 0.682 [0.649, 0.715] | 0.018 | 0.681 |
| Qwen2.5-7B | B | 0.776 [0.765, 0.788] | 0.754 [0.711, 0.796] | 0.000 | 0.811 |

## Clean model comparison: Condition A

Condition A is the appropriate unconfounded comparison of candidate-model
ability. Qwen exceeded Llama by 0.080 trajectory-score points (95% CI 0.066 to
0.094, item-clustered paired p < .001) and by 0.104 in exact-answer accuracy
(95% CI 0.074 to 0.134, p < .001). These advantages remained after restricting
the comparison to rows that parsed successfully for both models: +0.059 in
trajectory score (95% CI 0.045 to 0.073, p < .001) and +0.077 in accuracy
(95% CI 0.047 to 0.107, p < .001).

## Condition B is a forced-answer intervention

Condition B explicitly provides a pre-determined option and asks the model to
justify it. The option is fixed across the five templates of an item but differs
between the Qwen and Llama files. The models receive the same forced option on
only 64.2% of source items. The forced options are correct on 64.0% of Llama
items and 75.3% of Qwen items, while compliance is 98.4% and 99.95%,
respectively. Consequently, observed Condition B accuracy (63.4% for Llama and
75.4% for Qwen) almost exactly tracks forced-option accuracy. Raw cross-model
Condition B accuracy is therefore confounded and should not be presented as an
independent model-ability result.

## Evidence of post-hoc rationalization

The most informative analysis restricts Condition B to incorrectly forced
options. When Llama complied with an incorrect option, exact-answer accuracy was
0%, yet the mean trajectory score was 0.642 and 51.9% of explanations received
a score of at least 0.8. On the same items, forcing the incorrect answer raised
the trajectory score relative to Condition A by 0.060 (95% CI 0.036 to 0.084,
p < .001).

For Qwen, compliant incorrect forced answers likewise had 0% exact-answer
accuracy, but a mean trajectory score of 0.694; 65.8% scored at least 0.8.
Relative to Condition A on the same items, the trajectory score increased by
0.038 (95% CI 0.016 to 0.061, p < .001).

Thus, answer conditioning can increase judged coherence even when it reliably
produces a rationalization of an incorrect conclusion. This supports a
post-hoc-rationalization interpretation, not improved task correctness. Because
the judge rubric intentionally separates trajectory quality from answer
correctness, both outcomes must be reported separately.

## Sensitivity and diagnostics

Condition B reduced Llama parse errors from 9.4% to 1.9%. After retaining only
A/B pairs that parsed successfully in both conditions, Llama's accuracy change
was +0.026 (95% CI −0.007 to 0.058, p = .119), indicating that its significant
all-row accuracy increase was largely a parsing/reliability effect. Qwen's
parse-clean accuracy change remained +0.063 (95% CI 0.032 to 0.094, p < .001).

Across models and conditions, the mean within-item range across the five
templates was 0.206 trajectory-score points (95% CI 0.197 to 0.215), confirming
meaningful prompt-template sensitivity. Qwen answered all 300 controls
correctly; Llama control accuracy was 91.7%, with 60 parse-error rows.

Correct answers received higher trajectory scores than incorrect answers, but
the relationship was not deterministic. On abductive items the score/correctness
AUC was 0.718 for Llama and 0.679 for Qwen. Forty percent of Llama's incorrect
answers and 56% of Qwen's incorrect answers still received scores of at least
0.8, further demonstrating that the judge measures coherent trajectory quality
rather than exact-answer correctness.

The judge-rationale audit found no references to gold/golden labels, candidate
model identity, or Condition A/B labels. Rationales were almost entirely unique
and had a median length of 25 words. A blinded human-validation subset remains
recommended before publication.

## Claims that the paper can safely make

1. In the unconditioned setting, Qwen outperforms Llama on both trajectory
   quality and exact-answer accuracy.
2. Forced-answer prompting produces near-deterministic compliance.
3. Models can produce explanations judged coherent even when compelled toward
   an incorrect answer.
4. Forced-answer prompting can raise blind trajectory-quality scores on
   incorrectly forced items, consistent with post-hoc rationalization.
5. Parsing reliability and prompt-template sensitivity materially affect the
   observed results and require clustered analysis and sensitivity checks.

The paper should not claim that Condition B generally improves reasoning or that
its raw Qwen-versus-Llama difference is a clean ability comparison.
