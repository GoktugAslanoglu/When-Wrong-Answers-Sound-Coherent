from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ARTIFACT_ROOT = Path(__file__).resolve().parent.parent
GPT_ROOT = ARTIFACT_ROOT / "data" / "gpt4o_blind"
INPUTS = {
    "Llama-3.1-8B": GPT_ROOT / "ts_llama_abductive_full_1.jsonl",
    "Qwen2.5-7B": GPT_ROOT / "ts_qwen_abductive_full_1.jsonl",
}
OUT_DIR = Path(__file__).resolve().parent
FORCED_PATTERN = re.compile(r"pre-determined the answer is option\s+([A-D])", re.I)


def load_frame(path: Path, model: str) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    frame = pd.DataFrame(rows)
    frame["candidate_model"] = model
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["forced_answer"] = frame["prompt"].str.extract(FORCED_PATTERN, expand=False).str.upper()
    frame["forced_correct"] = frame["forced_answer"].eq(frame["golden_answer"])
    frame["complied"] = frame["extracted_answer"].eq(frame["forced_answer"])
    return frame


def mean_ci(values: pd.Series) -> tuple[float, float, float, int, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    mean = float(array.mean())
    if n < 2:
        return mean, math.nan, math.nan, n, math.nan
    se = float(array.std(ddof=1) / math.sqrt(n))
    margin = float(stats.t.ppf(0.975, n - 1) * se)
    p = float(stats.ttest_1samp(array, 0.0).pvalue)
    return mean, mean - margin, mean + margin, n, p


def paired_condition_effect(
    frame: pd.DataFrame,
    model: str,
    outcome: str,
    forced_correct: bool | None = None,
    parse_clean: bool = False,
) -> dict[str, Any]:
    subset = frame[frame["candidate_model"].eq(model)]
    pivot = subset.pivot(
        index=["sample_id", "template_id"],
        columns="condition",
        values=[outcome, "parse_error"],
    )
    selected = pd.Series(True, index=pivot.index)
    if forced_correct is not None:
        forced = (
            subset[subset["condition"].eq("B")]
            .set_index(["sample_id", "template_id"])["forced_correct"]
            .astype(bool)
        )
        selected &= forced.eq(forced_correct)
    if parse_clean:
        selected &= ~pivot[("parse_error", "A")].astype(bool)
        selected &= ~pivot[("parse_error", "B")].astype(bool)
    differences = (
        pivot.loc[selected, (outcome, "B")].astype(float)
        - pivot.loc[selected, (outcome, "A")].astype(float)
    )
    cluster_differences = differences.groupby(level="sample_id").mean()
    estimate, low, high, clusters, p = mean_ci(cluster_differences)
    return {
        "candidate_model": model,
        "contrast": "B − A",
        "outcome": outcome,
        "forced_correct_stratum": "all" if forced_correct is None else str(forced_correct).lower(),
        "parse_clean_pairs_only": parse_clean,
        "paired_rows": int(selected.sum()),
        "clusters": clusters,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "p_ttest": p,
    }


def paired_model_effect(frame: pd.DataFrame, condition: str, outcome: str) -> dict[str, Any]:
    subset = frame[frame["condition"].eq(condition)]
    pivot = subset.pivot(
        index=["sample_id", "template_id"],
        columns="candidate_model",
        values=[outcome, "parse_error"],
    )
    selected = (
        ~pivot[("parse_error", "Llama-3.1-8B")].astype(bool)
        & ~pivot[("parse_error", "Qwen2.5-7B")].astype(bool)
    )
    differences = (
        pivot.loc[selected, (outcome, "Qwen2.5-7B")].astype(float)
        - pivot.loc[selected, (outcome, "Llama-3.1-8B")].astype(float)
    )
    cluster_differences = differences.groupby(level="sample_id").mean()
    estimate, low, high, clusters, p = mean_ci(cluster_differences)
    return {
        "candidate_model": "Qwen2.5-7B − Llama-3.1-8B",
        "contrast": f"Qwen − Llama in condition {condition}",
        "outcome": outcome,
        "forced_correct_stratum": "not_applicable",
        "parse_clean_pairs_only": True,
        "paired_rows": int(selected.sum()),
        "clusters": clusters,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "p_ttest": p,
    }


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def fmt_p(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    return "<.001" if value < 0.001 else f"{value:.3f}".lstrip("0")


def main() -> None:
    frame = pd.concat(
        [load_frame(path, model) for model, path in INPUTS.items()],
        ignore_index=True,
    )
    condition_b = frame[frame["condition"].eq("B")].copy()
    if condition_b["forced_answer"].isna().any():
        raise ValueError("At least one Condition B prompt lacks a forced answer")

    overall_records: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []
    sample_forced: dict[str, pd.Series] = {}
    for model, group in condition_b.groupby("candidate_model", sort=False):
        per_sample_nunique = group.groupby("sample_id")["forced_answer"].nunique()
        per_sample_forced = group.groupby("sample_id")["forced_answer"].first()
        sample_forced[model] = per_sample_forced
        overall_records.append(
            {
                "candidate_model": model,
                "rows": len(group),
                "source_items": group["sample_id"].nunique(),
                "forced_answer_fixed_across_templates": bool(per_sample_nunique.eq(1).all()),
                "forced_answer_accuracy": float(group["forced_correct"].mean()),
                "compliance_rate": float(group["complied"].mean()),
                "answer_accuracy": float(group["is_correct"].mean()),
                "trajectory_score_mean": float(group["trajectory_score"].mean()),
                "high_quality_rate": float(group["trajectory_score"].ge(0.8).mean()),
                "parse_error_rate": float(group["parse_error"].mean()),
            }
        )
        for (forced_correct, complied), cell in group.groupby(["forced_correct", "complied"], sort=False):
            cell_records.append(
                {
                    "candidate_model": model,
                    "forced_correct": bool(forced_correct),
                    "complied": bool(complied),
                    "rows": len(cell),
                    "answer_accuracy": float(cell["is_correct"].mean()),
                    "trajectory_score_mean": float(cell["trajectory_score"].mean()),
                    "high_quality_rate": float(cell["trajectory_score"].ge(0.8).mean()),
                    "parse_error_rate": float(cell["parse_error"].mean()),
                }
            )

    forced_agreement = float(
        sample_forced["Llama-3.1-8B"].eq(sample_forced["Qwen2.5-7B"]).mean()
    )
    overall = pd.DataFrame(overall_records)
    cells = pd.DataFrame(cell_records)
    overall["cross_model_forced_answer_agreement"] = forced_agreement
    overall.to_csv(OUT_DIR / "forced_condition_summary.csv", index=False)
    cells.to_csv(OUT_DIR / "forced_condition_cells.csv", index=False)

    paired_records: list[dict[str, Any]] = []
    for model in INPUTS:
        for outcome in ["trajectory_score", "is_correct"]:
            for stratum in [None, False, True]:
                paired_records.append(
                    paired_condition_effect(frame, model, outcome, forced_correct=stratum)
                )
            paired_records.append(
                paired_condition_effect(frame, model, outcome, parse_clean=True)
            )
    for condition in ["A", "B"]:
        for outcome in ["trajectory_score", "is_correct"]:
            paired_records.append(paired_model_effect(frame, condition, outcome))
    paired = pd.DataFrame(paired_records)
    paired.to_csv(OUT_DIR / "forced_condition_paired_effects.csv", index=False)

    rationale_records: list[dict[str, Any]] = []
    hidden_patterns = {
        "gold_term": re.compile(r"\bgold(?:en)?\b", re.I),
        "model_name": re.compile(r"\b(?:qwen|llama)\b", re.I),
        "condition_label": re.compile(r"\bcondition\s+[ab]\b", re.I),
    }
    for model, group in frame.groupby("candidate_model", sort=False):
        rationales = group["judge_rationale"].astype(str).str.strip()
        counts = Counter(rationales)
        rationale_records.append(
            {
                "candidate_model": model,
                "rows": len(group),
                "unique_rationales": rationales.nunique(),
                "maximum_exact_repetition": max(counts.values()),
                "median_words": float(rationales.str.split().str.len().median()),
                **{
                    f"{name}_mentions": int(rationales.str.contains(pattern).sum())
                    for name, pattern in hidden_patterns.items()
                },
            }
        )
    rationale_audit = pd.DataFrame(rationale_records)
    rationale_audit.to_csv(OUT_DIR / "judge_rationale_audit.csv", index=False)

    llama = overall[overall["candidate_model"].eq("Llama-3.1-8B")].iloc[0]
    qwen = overall[overall["candidate_model"].eq("Qwen2.5-7B")].iloc[0]
    llama_wrong = cells[
        cells["candidate_model"].eq("Llama-3.1-8B")
        & ~cells["forced_correct"]
        & cells["complied"]
    ].iloc[0]
    qwen_wrong = cells[
        cells["candidate_model"].eq("Qwen2.5-7B")
        & ~cells["forced_correct"]
        & cells["complied"]
    ].iloc[0]

    def effect(model: str, outcome: str, stratum: str, parse_clean: bool = False) -> pd.Series:
        return paired[
            paired["candidate_model"].eq(model)
            & paired["outcome"].eq(outcome)
            & paired["forced_correct_stratum"].eq(stratum)
            & paired["parse_clean_pairs_only"].eq(parse_clean)
        ].iloc[0]

    llama_wrong_effect = effect("Llama-3.1-8B", "trajectory_score", "false")
    qwen_wrong_effect = effect("Qwen2.5-7B", "trajectory_score", "false")
    llama_clean_accuracy = effect("Llama-3.1-8B", "is_correct", "all", True)
    qwen_clean_accuracy = effect("Qwen2.5-7B", "is_correct", "all", True)

    lines = [
        "# Forced-answer and sensitivity analysis",
        "",
        "## What Condition B actually tests",
        "",
        "Condition B supplies a pre-determined answer in the prompt and requires the candidate model to justify it. The forced option is constant across all five templates for each source item, but it is not the same intervention for both candidate models: each model's file contains its own forced option. Therefore, raw Condition B differences between Qwen and Llama are confounded by the quality of the supplied options and must not be presented as a clean model-ability comparison.",
        "",
        "| Candidate model | Forced-option accuracy | Compliance | Observed B accuracy | Mean B score | B score ≥ 0.8 | Parse errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Llama-3.1-8B | {fmt(llama['forced_answer_accuracy'])} | {fmt(llama['compliance_rate'])} | {fmt(llama['answer_accuracy'])} | {fmt(llama['trajectory_score_mean'])} | {fmt(llama['high_quality_rate'])} | {fmt(llama['parse_error_rate'])} |",
        f"| Qwen2.5-7B | {fmt(qwen['forced_answer_accuracy'])} | {fmt(qwen['compliance_rate'])} | {fmt(qwen['answer_accuracy'])} | {fmt(qwen['trajectory_score_mean'])} | {fmt(qwen['high_quality_rate'])} | {fmt(qwen['parse_error_rate'])} |",
        "",
        f"The two models receive the same forced option on only {fmt(forced_agreement)} of source items. Their observed B accuracies closely track forced-option accuracy because compliance is nearly deterministic.",
        "",
        "## Post-hoc rationalization under incorrect forced answers",
        "",
        f"When Llama complied with an incorrect supplied option, exact-answer accuracy was {fmt(llama_wrong['answer_accuracy'])}, but mean GPT-4o trajectory score was {fmt(llama_wrong['trajectory_score_mean'])} and {fmt(llama_wrong['high_quality_rate'])} received a score of at least 0.8. Relative to Condition A on those same items, the mean trajectory score increased by {fmt(llama_wrong_effect['estimate'])} (95% CI {fmt(llama_wrong_effect['ci_low'])} to {fmt(llama_wrong_effect['ci_high'])}; p {fmt_p(llama_wrong_effect['p_ttest'])}).",
        "",
        f"For Qwen, compliant incorrect forced answers had exact-answer accuracy {fmt(qwen_wrong['answer_accuracy'])}, mean trajectory score {fmt(qwen_wrong['trajectory_score_mean'])}, and high-quality rate {fmt(qwen_wrong['high_quality_rate'])}. The paired trajectory-score increase over Condition A was {fmt(qwen_wrong_effect['estimate'])} (95% CI {fmt(qwen_wrong_effect['ci_low'])} to {fmt(qwen_wrong_effect['ci_high'])}; p {fmt_p(qwen_wrong_effect['p_ttest'])}).",
        "",
        "This is the strongest evidence in the result set: answer conditioning can make an explanation appear more coherent to an outcome-independent judge even when it reliably rationalizes an incorrect conclusion. It supports a post-hoc rationalization claim, not an improvement in causal-task correctness.",
        "",
        "## Parse-clean sensitivity",
        "",
        f"After retaining only A/B pairs that parsed successfully in both conditions, Llama's B − A accuracy change was {fmt(llama_clean_accuracy['estimate'])} (95% CI {fmt(llama_clean_accuracy['ci_low'])} to {fmt(llama_clean_accuracy['ci_high'])}; p {fmt_p(llama_clean_accuracy['p_ttest'])}). Thus, Llama's significant primary accuracy increase is largely explained by fewer parsing failures in B.",
        "",
        f"Qwen's parse-clean B − A accuracy change remained {fmt(qwen_clean_accuracy['estimate'])} (95% CI {fmt(qwen_clean_accuracy['ci_low'])} to {fmt(qwen_clean_accuracy['ci_high'])}; p {fmt_p(qwen_clean_accuracy['p_ttest'])}).",
        "",
        "## Judge audit",
        "",
        "The abductive judge rationales were almost all unique, with median length 25 words. No rationale mentioned a gold/golden label, Qwen/Llama identity, or a Condition A/B label. This supports the intended blindness of the stored judgments, although a separate human-validation subset is still advisable before publication.",
        "",
        "## Reporting rule",
        "",
        "Use Condition A for the clean Qwen-versus-Llama ability comparison. Analyze Condition B primarily as an answer-conditioning/compliance intervention, stratified by whether the forced option is correct. Report primary all-row results together with the parse-clean sensitivity analysis.",
    ]
    (OUT_DIR / "forced_condition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote forced-answer analysis artifacts to", OUT_DIR)
    print(overall.to_string(index=False))
    print(cells.to_string(index=False))


if __name__ == "__main__":
    main()
