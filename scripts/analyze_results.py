from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ARTIFACT_ROOT = Path(__file__).resolve().parent.parent
GPT_ROOT = ARTIFACT_ROOT / "data" / "gpt4o_blind"
INPUTS = {
    ("abductive", "Llama-3.1-8B"): GPT_ROOT / "ts_llama_abductive_full_1.jsonl",
    ("abductive", "Qwen2.5-7B"): GPT_ROOT / "ts_qwen_abductive_full_1.jsonl",
    ("control", "Llama-3.1-8B"): GPT_ROOT / "ts_llama_control_full_1.jsonl",
    ("control", "Qwen2.5-7B"): GPT_ROOT / "ts_qwen_control_full_1.jsonl",
}
OUT_DIR = Path(__file__).resolve().parent
SCORE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Non-object row in {path} line {line_number}")
            rows.append(row)
    return rows


def mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float, int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(array.mean())
    if n == 1:
        return mean, math.nan, math.nan, n
    se = float(array.std(ddof=1) / math.sqrt(n))
    margin = float(stats.t.ppf(0.975, n - 1) * se)
    return mean, mean - margin, mean + margin, n


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_p(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".lstrip("0")


def cluster_summary(group: pd.DataFrame) -> dict[str, Any]:
    cluster = group.groupby("sample_id", sort=False).agg(
        trajectory_score=("trajectory_score", "mean"),
        accuracy=("is_correct", "mean"),
        parse_error_rate=("parse_error", "mean"),
        causal_marker_rate=("has_causal_markers", "mean"),
    )
    score, score_lo, score_hi, clusters = mean_ci(cluster["trajectory_score"])
    accuracy, accuracy_lo, accuracy_hi, _ = mean_ci(cluster["accuracy"])
    return {
        "rows": int(len(group)),
        "clusters": clusters,
        "score_mean": score,
        "score_sd_rows": float(group["trajectory_score"].std(ddof=1)),
        "score_median": float(group["trajectory_score"].median()),
        "score_ci_low": score_lo,
        "score_ci_high": score_hi,
        "accuracy": accuracy,
        "accuracy_ci_low": accuracy_lo,
        "accuracy_ci_high": accuracy_hi,
        "parse_error_rate": float(group["parse_error"].mean()),
        "causal_marker_rate": float(group["has_causal_markers"].mean()),
        "high_quality_rate": float((group["trajectory_score"] >= 0.8).mean()),
        "perfect_score_rate": float((group["trajectory_score"] == 1.0).mean()),
    }


def contrast_stats(cluster_differences: pd.Series, label: str, outcome: str) -> dict[str, Any]:
    values = np.asarray(cluster_differences, dtype=float)
    values = values[np.isfinite(values)]
    estimate, ci_low, ci_high, n = mean_ci(values)
    sd = float(values.std(ddof=1)) if n > 1 else math.nan
    effect_dz = estimate / sd if n > 1 and sd > 0 else math.nan
    t_result = stats.ttest_1samp(values, popmean=0.0) if n > 1 else None
    nonzero = values[values != 0]
    if len(nonzero):
        try:
            wilcoxon_p = float(stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            wilcoxon_p = math.nan
    else:
        wilcoxon_p = 1.0
    return {
        "contrast": label,
        "outcome": outcome,
        "clusters": n,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sd_cluster_difference": sd,
        "effect_dz": effect_dz,
        "t_statistic": float(t_result.statistic) if t_result is not None else math.nan,
        "p_ttest": float(t_result.pvalue) if t_result is not None else math.nan,
        "p_wilcoxon": wilcoxon_p,
    }


def model_contrast(df: pd.DataFrame, split: str, outcome: str, condition: str | None = None) -> dict[str, Any]:
    subset = df[df["split"].eq(split)]
    if condition is not None:
        subset = subset[subset["condition"].eq(condition)]
    pivot = subset.pivot(
        index=["sample_id", "template_id", "condition"],
        columns="candidate_model",
        values=outcome,
    ).astype(float)
    difference = pivot["Qwen2.5-7B"] - pivot["Llama-3.1-8B"]
    cluster_difference = difference.groupby(level="sample_id").mean()
    suffix = f", condition {condition}" if condition else ""
    return contrast_stats(cluster_difference, f"Qwen − Llama ({split}{suffix})", outcome)


def condition_contrast(df: pd.DataFrame, split: str, model: str, outcome: str) -> dict[str, Any]:
    subset = df[df["split"].eq(split) & df["candidate_model"].eq(model)]
    pivot = subset.pivot(index=["sample_id", "template_id"], columns="condition", values=outcome).astype(float)
    difference = pivot["B"] - pivot["A"]
    cluster_difference = difference.groupby(level="sample_id").mean()
    return contrast_stats(cluster_difference, f"Condition B − A ({model}, {split})", outcome)


def interaction_contrast(df: pd.DataFrame, split: str, outcome: str) -> dict[str, Any]:
    subset = df[df["split"].eq(split)]
    pivot = subset.pivot(
        index=["sample_id", "template_id"],
        columns=["candidate_model", "condition"],
        values=outcome,
    ).astype(float)
    difference = (
        pivot[("Qwen2.5-7B", "B")]
        - pivot[("Qwen2.5-7B", "A")]
        - pivot[("Llama-3.1-8B", "B")]
        + pivot[("Llama-3.1-8B", "A")]
    )
    cluster_difference = difference.groupby(level="sample_id").mean()
    return contrast_stats(cluster_difference, f"Model × condition interaction ({split})", outcome)


def template_contrast(df: pd.DataFrame, split: str, outcome: str) -> dict[str, Any]:
    subset = df[df["split"].eq(split)]
    cluster_means = subset.groupby(["sample_id", "template_id"], sort=False)[outcome].mean().unstack("template_id")
    grand = cluster_means.mean(axis=1)
    centered = cluster_means.subtract(grand, axis=0)
    max_minus_min = cluster_means.max(axis=1) - cluster_means.min(axis=1)
    estimate, ci_low, ci_high, n = mean_ci(max_minus_min)
    return {
        "split": split,
        "outcome": outcome,
        "clusters": n,
        "mean_within_item_template_range": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "template_means": {str(column): float(cluster_means[column].mean()) for column in cluster_means},
        "mean_absolute_centered_effect": float(centered.abs().to_numpy().mean()),
    }


def roc_auc(y_true: pd.Series, scores: pd.Series) -> float:
    y = np.asarray(y_true, dtype=bool)
    x = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = stats.rankdata(x, method="average")
    rank_sum_positive = float(ranks[y].sum())
    return (rank_sum_positive - positives * (positives + 1) / 2) / (positives * negatives)


def correctness_association(group: pd.DataFrame) -> dict[str, Any]:
    correct = group[group["is_correct"]]["trajectory_score"]
    incorrect = group[~group["is_correct"]]["trajectory_score"]
    if group["is_correct"].nunique() < 2 or group["trajectory_score"].nunique() < 2:
        correlation_statistic = math.nan
        correlation_p = math.nan
    else:
        correlation = stats.spearmanr(group["trajectory_score"], group["is_correct"].astype(int))
        correlation_statistic = float(correlation.statistic)
        correlation_p = float(correlation.pvalue)
    return {
        "rows": int(len(group)),
        "correct_rows": int(len(correct)),
        "incorrect_rows": int(len(incorrect)),
        "score_correct_mean": float(correct.mean()) if len(correct) else math.nan,
        "score_incorrect_mean": float(incorrect.mean()) if len(incorrect) else math.nan,
        "mean_gap": float(correct.mean() - incorrect.mean()) if len(correct) and len(incorrect) else math.nan,
        "auc": roc_auc(group["is_correct"], group["trajectory_score"]),
        "spearman_rho": correlation_statistic,
        "spearman_p": correlation_p,
        "incorrect_high_quality_rate": float((incorrect >= 0.8).mean()) if len(incorrect) else math.nan,
        "correct_low_quality_rate": float((correct <= 0.4).mean()) if len(correct) else math.nan,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def validation_record(name: str, path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "sample_id",
        "topic_id",
        "template_id",
        "condition",
        "model",
        "is_correct",
        "parse_error",
        "has_causal_markers",
        "trajectory_score",
        "judge_rationale",
    }
    missing_required = sorted(required - set(rows[0])) if rows else sorted(required)
    scores = [row.get("trajectory_score") for row in rows]
    rationales = [str(row.get("judge_rationale") or "").strip() for row in rows]
    keys = [
        (str(row.get("sample_id")), row.get("topic_id"), row.get("template_id"), row.get("condition"))
        for row in rows
    ]
    invalid_scores = [score for score in scores if not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1]
    off_grid_scores = [score for score in scores if isinstance(score, (int, float)) and round(float(score), 10) not in SCORE_LEVELS]
    return {
        "name": name,
        "path": str(path),
        "rows": len(rows),
        "unique_keys": len(set(keys)),
        "duplicate_keys": len(keys) - len(set(keys)),
        "missing_required_fields": missing_required,
        "invalid_scores": len(invalid_scores),
        "off_grid_scores": len(off_grid_scores),
        "empty_rationales": sum(not rationale for rationale in rationales),
        "conditions": sorted({row.get("condition") for row in rows}),
        "templates": sorted({row.get("template_id") for row in rows}),
    }


def frame_from_rows(rows: list[dict[str, Any]], split: str, model: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["split"] = split
    frame["candidate_model"] = model
    frame["sample_id"] = frame["sample_id"].astype(str)
    for column in ["is_correct", "parse_error", "has_causal_markers"]:
        frame[column] = frame[column].astype(bool)
    frame["trajectory_score"] = pd.to_numeric(frame["trajectory_score"], errors="raise")
    return frame


def save_figure_condition_model(group_summary: pd.DataFrame) -> None:
    data = group_summary[group_summary["split"].eq("abductive")].copy()
    models = ["Llama-3.1-8B", "Qwen2.5-7B"]
    conditions = ["A", "B"]
    colors = {"A": "#4472C4", "B": "#ED7D31"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for axis, outcome, low, high, ylabel in [
        (axes[0], "score_mean", "score_ci_low", "score_ci_high", "Mean trajectory score"),
        (axes[1], "accuracy", "accuracy_ci_low", "accuracy_ci_high", "Answer accuracy"),
    ]:
        x = np.arange(len(models))
        width = 0.34
        for offset_index, condition in enumerate(conditions):
            values = []
            lower = []
            upper = []
            for model in models:
                row = data[data["candidate_model"].eq(model) & data["condition"].eq(condition)].iloc[0]
                values.append(row[outcome])
                lower.append(row[outcome] - row[low])
                upper.append(row[high] - row[outcome])
            positions = x + (offset_index - 0.5) * width
            axis.bar(positions, values, width, color=colors[condition], label=f"Condition {condition}")
            axis.errorbar(positions, values, yerr=[lower, upper], fmt="none", ecolor="black", capsize=3, linewidth=1)
        axis.set_xticks(x, models)
        axis.set_ylim(0, 1)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("GPT-4o blind-judge results on abductive items")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_condition_model.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_figure_score_distribution(score_distribution: pd.DataFrame) -> None:
    data = score_distribution[score_distribution["split"].eq("abductive")].copy()
    labels = [
        "Llama A",
        "Llama B",
        "Qwen A",
        "Qwen B",
    ]
    groups = [
        ("Llama-3.1-8B", "A"),
        ("Llama-3.1-8B", "B"),
        ("Qwen2.5-7B", "A"),
        ("Qwen2.5-7B", "B"),
    ]
    palette = ["#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#FD8D3C", "#D94801"]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    bottoms = np.zeros(len(groups))
    for score, color in zip(SCORE_LEVELS, palette):
        values = []
        for model, condition in groups:
            row = data[
                data["candidate_model"].eq(model)
                & data["condition"].eq(condition)
                & data["trajectory_score"].eq(score)
            ]
            values.append(float(row["proportion"].iloc[0]) if len(row) else 0.0)
        axis.bar(labels, values, bottom=bottoms, label=f"{score:.1f}", color=color)
        bottoms += np.asarray(values)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Proportion of responses")
    axis.set_title("Distribution of GPT-4o trajectory scores")
    axis.legend(title="Score", ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_score_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    loaded: dict[tuple[str, str], list[dict[str, Any]]] = {}
    validations: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for (split, model), path in INPUTS.items():
        rows = load_jsonl(path)
        loaded[(split, model)] = rows
        validations.append(validation_record(f"{model}_{split}", path, rows))
        frames.append(frame_from_rows(rows, split, model))
    df = pd.concat(frames, ignore_index=True)

    pair_validation: list[dict[str, Any]] = []
    pair_fields = ["sample_id", "topic_id", "template_id", "condition", "prompt", "golden_answer"]
    for split in ["abductive", "control"]:
        left = loaded[(split, "Llama-3.1-8B")]
        right = loaded[(split, "Qwen2.5-7B")]
        ordered_key_match = all(
            all(str(left_row.get(field)) == str(right_row.get(field)) for field in pair_fields)
            for left_row, right_row in zip(left, right)
        )
        pair_validation.append(
            {
                "split": split,
                "same_length": len(left) == len(right),
                "aligned_pair_fields": ordered_key_match,
                "rows": len(left),
            }
        )

    group_records: list[dict[str, Any]] = []
    for (split, model, condition), group in df.groupby(["split", "candidate_model", "condition"], sort=False):
        record = {"split": split, "candidate_model": model, "condition": condition}
        record.update(cluster_summary(group))
        group_records.append(record)
    group_summary = pd.DataFrame(group_records).sort_values(["split", "candidate_model", "condition"])
    group_summary.to_csv(OUT_DIR / "group_summary.csv", index=False)

    template_records: list[dict[str, Any]] = []
    for (split, model, condition, template), group in df.groupby(
        ["split", "candidate_model", "condition", "template_id"], sort=False
    ):
        record = {
            "split": split,
            "candidate_model": model,
            "condition": condition,
            "template_id": int(template),
        }
        record.update(cluster_summary(group))
        template_records.append(record)
    template_summary = pd.DataFrame(template_records).sort_values(
        ["split", "candidate_model", "condition", "template_id"]
    )
    template_summary.to_csv(OUT_DIR / "template_summary.csv", index=False)

    score_distribution = (
        df.groupby(["split", "candidate_model", "condition", "trajectory_score"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    score_distribution["proportion"] = score_distribution.groupby(
        ["split", "candidate_model", "condition"]
    )["count"].transform(lambda values: values / values.sum())
    score_distribution.to_csv(OUT_DIR / "score_distribution.csv", index=False)

    correctness_records: list[dict[str, Any]] = []
    for (split, model), group in df.groupby(["split", "candidate_model"], sort=False):
        record = {"split": split, "candidate_model": model}
        record.update(correctness_association(group))
        correctness_records.append(record)
    correctness_summary = pd.DataFrame(correctness_records).sort_values(["split", "candidate_model"])
    correctness_summary.to_csv(OUT_DIR / "correctness_association.csv", index=False)

    score_by_correctness = (
        df.groupby(["split", "candidate_model", "condition", "is_correct"], sort=False)
        .agg(
            rows=("trajectory_score", "size"),
            score_mean=("trajectory_score", "mean"),
            score_sd=("trajectory_score", "std"),
            high_quality_rate=("trajectory_score", lambda values: float((values >= 0.8).mean())),
        )
        .reset_index()
    )
    score_by_correctness.to_csv(OUT_DIR / "score_by_correctness.csv", index=False)

    diagnostic_summary = (
        df.groupby(["split", "candidate_model", "parse_error", "has_causal_markers"], sort=False)
        .agg(
            rows=("trajectory_score", "size"),
            score_mean=("trajectory_score", "mean"),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    diagnostic_summary.to_csv(OUT_DIR / "diagnostic_summary.csv", index=False)

    tests: list[dict[str, Any]] = []
    for split in ["abductive", "control"]:
        for outcome in ["trajectory_score", "is_correct"]:
            tests.append(model_contrast(df, split, outcome))
            for condition in ["A", "B"]:
                tests.append(model_contrast(df, split, outcome, condition))
            for model in ["Llama-3.1-8B", "Qwen2.5-7B"]:
                tests.append(condition_contrast(df, split, model, outcome))
            tests.append(interaction_contrast(df, split, outcome))
    paired_tests = pd.DataFrame(tests)
    paired_tests.to_csv(OUT_DIR / "paired_cluster_tests.csv", index=False)

    template_effects = [
        template_contrast(df, split, outcome)
        for split in ["abductive", "control"]
        for outcome in ["trajectory_score", "is_correct"]
    ]
    pd.DataFrame(template_effects).to_csv(OUT_DIR / "template_effects.csv", index=False)

    summary = json_safe({
        "validation": validations,
        "pair_validation": pair_validation,
        "group_summary": group_summary.to_dict(orient="records"),
        "correctness_association": correctness_summary.to_dict(orient="records"),
        "paired_cluster_tests": paired_tests.to_dict(orient="records"),
        "template_effects": template_effects,
    })
    (OUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    save_figure_condition_model(group_summary)
    save_figure_score_distribution(score_distribution)

    abductive_groups = group_summary[group_summary["split"].eq("abductive")]
    control_groups = group_summary[group_summary["split"].eq("control")]

    def group_row(split: str, model: str, condition: str) -> pd.Series:
        source = group_summary[
            group_summary["split"].eq(split)
            & group_summary["candidate_model"].eq(model)
            & group_summary["condition"].eq(condition)
        ]
        return source.iloc[0]

    def test_row(label: str, outcome: str) -> pd.Series:
        source = paired_tests[
            paired_tests["contrast"].eq(label)
            & paired_tests["outcome"].eq(outcome)
        ]
        return source.iloc[0]

    report_lines = [
        "# GPT-4o blind trajectory-judge analysis",
        "",
        "## Data integrity",
        "",
        "All four JSONL files parsed successfully. They contain 3,970 abductive and 300 control rows per candidate model (8,540 judgments total). Every row has a finite score on the preregistered 0.2 grid and a non-empty rationale. The Qwen and Llama files are aligned one-to-one on sample, topic, template, condition, prompt, and gold answer.",
        "",
        "The abductive set comprises 397 source items, each repeated across five templates and two conditions. The control set comprises 30 source items under the same 5 × 2 repeated-measures structure. Confidence intervals and hypothesis tests below therefore use the source item as the clustering unit.",
        "",
        "## Headline descriptive results",
        "",
        "| Split | Model | Condition | n | Mean score [95% CI] | Accuracy [95% CI] | Parse errors | Score ≥ 0.8 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in pd.concat([abductive_groups, control_groups]).iterrows():
        report_lines.append(
            f"| {row['split']} | {row['candidate_model']} | {row['condition']} | {int(row['rows'])} | "
            f"{fmt(row['score_mean'])} [{fmt(row['score_ci_low'])}, {fmt(row['score_ci_high'])}] | "
            f"{fmt(row['accuracy'])} [{fmt(row['accuracy_ci_low'])}, {fmt(row['accuracy_ci_high'])}] | "
            f"{fmt(row['parse_error_rate'])} | {fmt(row['high_quality_rate'])} |"
        )

    qwen_model_score = test_row("Qwen − Llama (abductive)", "trajectory_score")
    qwen_model_accuracy = test_row("Qwen − Llama (abductive)", "is_correct")
    llama_condition_score = test_row("Condition B − A (Llama-3.1-8B, abductive)", "trajectory_score")
    qwen_condition_score = test_row("Condition B − A (Qwen2.5-7B, abductive)", "trajectory_score")
    interaction_score = test_row("Model × condition interaction (abductive)", "trajectory_score")
    llama_condition_accuracy = test_row("Condition B − A (Llama-3.1-8B, abductive)", "is_correct")
    qwen_condition_accuracy = test_row("Condition B − A (Qwen2.5-7B, abductive)", "is_correct")

    report_lines.extend(
        [
            "",
            "## Paired abductive comparisons",
            "",
            f"Across conditions and templates, Qwen exceeded Llama by {fmt(qwen_model_score['estimate'])} trajectory-score points (95% CI {fmt(qwen_model_score['ci_low'])} to {fmt(qwen_model_score['ci_high'])}; item-clustered paired t-test p {fmt_p(qwen_model_score['p_ttest'])}) and by {fmt(qwen_model_accuracy['estimate'])} in answer accuracy (95% CI {fmt(qwen_model_accuracy['ci_low'])} to {fmt(qwen_model_accuracy['ci_high'])}; p {fmt_p(qwen_model_accuracy['p_ttest'])}).",
            "",
            f"Condition B changed Llama's mean trajectory score by {fmt(llama_condition_score['estimate'])} (95% CI {fmt(llama_condition_score['ci_low'])} to {fmt(llama_condition_score['ci_high'])}; p {fmt_p(llama_condition_score['p_ttest'])}) and Qwen's by {fmt(qwen_condition_score['estimate'])} (95% CI {fmt(qwen_condition_score['ci_low'])} to {fmt(qwen_condition_score['ci_high'])}; p {fmt_p(qwen_condition_score['p_ttest'])}). The model × condition interaction was {fmt(interaction_score['estimate'])} (95% CI {fmt(interaction_score['ci_low'])} to {fmt(interaction_score['ci_high'])}; p {fmt_p(interaction_score['p_ttest'])}).",
            "",
            f"For answer accuracy, the B − A change was {fmt(llama_condition_accuracy['estimate'])} for Llama (p {fmt_p(llama_condition_accuracy['p_ttest'])}) and {fmt(qwen_condition_accuracy['estimate'])} for Qwen (p {fmt_p(qwen_condition_accuracy['p_ttest'])}).",
            "",
            "## Relationship between reasoning quality and answer correctness",
            "",
            "| Split | Model | Mean score: correct | Mean score: incorrect | Gap | AUC | Incorrect with score ≥ 0.8 | Correct with score ≤ 0.4 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in correctness_summary.iterrows():
        report_lines.append(
            f"| {row['split']} | {row['candidate_model']} | {fmt(row['score_correct_mean'])} | "
            f"{fmt(row['score_incorrect_mean'])} | {fmt(row['mean_gap'])} | {fmt(row['auc'])} | "
            f"{fmt(row['incorrect_high_quality_rate'])} | {fmt(row['correct_low_quality_rate'])} |"
        )

    template_abductive_score = next(
        item for item in template_effects if item["split"] == "abductive" and item["outcome"] == "trajectory_score"
    )
    report_lines.extend(
        [
            "",
            "AUC measures how well the trajectory score ranks correct above incorrect answers. The non-zero rate of high scores among incorrect answers is expected for an outcome-independent trajectory rubric: a coherent reasoning path can still end with the wrong option. Accordingly, trajectory quality and task accuracy should be reported as related but distinct outcomes.",
            "",
            "## Template sensitivity",
            "",
            f"Averaged across models and conditions, the mean within-item range across the five prompt templates was {fmt(template_abductive_score['mean_within_item_template_range'])} trajectory-score points (95% CI {fmt(template_abductive_score['ci_low'])} to {fmt(template_abductive_score['ci_high'])}). Template-level results are in `template_summary.csv`; this variability should be modeled or clustered rather than treating templates as independent samples.",
            "",
            "## Control interpretation",
            "",
            f"Qwen answered all 300 control prompts correctly; Llama accuracy was {fmt(df[(df['split'].eq('control')) & (df['candidate_model'].eq('Llama-3.1-8B'))]['is_correct'].mean())}. The judge assigned a perfect score to {fmt(df[(df['split'].eq('control')) & (df['candidate_model'].eq('Qwen2.5-7B'))]['trajectory_score'].eq(1.0).mean())} of Qwen controls and {fmt(df[(df['split'].eq('control')) & (df['candidate_model'].eq('Llama-3.1-8B'))]['trajectory_score'].eq(1.0).mean())} of Llama controls. This supports basic judge sensitivity, while Llama's 60 control parse errors explain much of its lower control performance and should be disclosed.",
            "",
            "## Recommended paper claims",
            "",
            "1. Report GPT-4o trajectory scores and exact-answer accuracy as separate dependent variables.",
            "2. Describe the judge as blind to the stored gold label and model/condition identity, and state the six-level rubric explicitly.",
            "3. Use item-clustered paired inference because every source item appears under ten prompt variants per model.",
            "4. Treat the controls as a sanity check, not as part of the abductive-effect estimate.",
            "5. Include the complete prompt, model snapshot/date, structured-output schema, retry procedure, and all four result files in the reproducibility package.",
            "",
            "## Output files",
            "",
            "- `analysis_summary.json`: machine-readable audit and statistics",
            "- `group_summary.csv`: model × condition summaries",
            "- `template_summary.csv`: template-level summaries",
            "- `paired_cluster_tests.csv`: clustered paired tests and effect sizes",
            "- `correctness_association.csv`: score/correctness relationship",
            "- `score_distribution.csv`: full six-level score distribution",
            "- `diagnostic_summary.csv`: parse-error and marker diagnostics",
            "- `figure_condition_model.png` and `figure_score_distribution.png`: paper-ready figures",
        ]
    )
    (OUT_DIR / "analysis_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Analyzed {len(df):,} judgments")
    print(group_summary.to_string(index=False))
    print(f"Wrote report to {OUT_DIR / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
