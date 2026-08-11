# Forced-answer and sensitivity analysis

## What Condition B actually tests

Condition B supplies a pre-determined answer in the prompt and requires the candidate model to justify it. The forced option is constant across all five templates for each source item, but it is not the same intervention for both candidate models: each model's file contains its own forced option. Therefore, raw Condition B differences between Qwen and Llama are confounded by the quality of the supplied options and must not be presented as a clean model-ability comparison.

| Candidate model | Forced-option accuracy | Compliance | Observed B accuracy | Mean B score | B score ≥ 0.8 | Parse errors |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.640 | 0.984 | 0.634 | 0.732 | 0.704 | 0.019 |
| Qwen2.5-7B | 0.753 | 0.999 | 0.754 | 0.776 | 0.811 | 0.000 |

The two models receive the same forced option on only 0.642 of source items. Their observed B accuracies closely track forced-option accuracy because compliance is nearly deterministic.

## Post-hoc rationalization under incorrect forced answers

When Llama complied with an incorrect supplied option, exact-answer accuracy was 0.000, but mean GPT-4o trajectory score was 0.642 and 0.519 received a score of at least 0.8. Relative to Condition A on those same items, the mean trajectory score increased by 0.060 (95% CI 0.036 to 0.084; p <.001).

For Qwen, compliant incorrect forced answers had exact-answer accuracy 0.000, mean trajectory score 0.694, and high-quality rate 0.658. The paired trajectory-score increase over Condition A was 0.038 (95% CI 0.016 to 0.061; p <.001).

This is the strongest evidence in the result set: answer conditioning can make an explanation appear more coherent to an outcome-independent judge even when it reliably rationalizes an incorrect conclusion. It supports a post-hoc rationalization claim, not an improvement in causal-task correctness.

## Parse-clean sensitivity

After retaining only A/B pairs that parsed successfully in both conditions, Llama's B − A accuracy change was 0.026 (95% CI -0.007 to 0.058; p .119). Thus, Llama's significant primary accuracy increase is largely explained by fewer parsing failures in B.

Qwen's parse-clean B − A accuracy change remained 0.063 (95% CI 0.032 to 0.094; p <.001).

## Judge audit

The abductive judge rationales were almost all unique, with median length 25 words. No rationale mentioned a gold/golden label, Qwen/Llama identity, or a Condition A/B label. This supports the intended blindness of the stored judgments, although a separate human-validation subset is still advisable before publication.

## Reporting rule

Use Condition A for the clean Qwen-versus-Llama ability comparison. Analyze Condition B primarily as an answer-conditioning/compliance intervention, stratified by whether the forced option is correct. Report primary all-row results together with the parse-clean sensitivity analysis.
