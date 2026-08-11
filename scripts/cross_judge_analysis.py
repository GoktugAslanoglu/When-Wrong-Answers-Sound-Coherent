from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ARTIFACT_ROOT = Path(__file__).resolve().parent.parent
QWEN_ROOT = ARTIFACT_ROOT / "data" / "qwen_reference_aware"
GPT_ROOT = ARTIFACT_ROOT / "data" / "gpt4o_blind"
OUT_DIR = Path(__file__).resolve().parent

FILES = {
    ("abductive", "Llama-3.1-8B"): (QWEN_ROOT / "ts_llama_abductive_full_1.jsonl", GPT_ROOT / "ts_llama_abductive_full_1.jsonl"),
    ("abductive", "Qwen2.5-7B"): (QWEN_ROOT / "ts_qwen_abductive_full_1.jsonl", GPT_ROOT / "ts_qwen_abductive_full_1.jsonl"),
    ("control", "Llama-3.1-8B"): (QWEN_ROOT / "ts_llama_control_full_1.jsonl", GPT_ROOT / "ts_llama_control_full_1.jsonl"),
    ("control", "Qwen2.5-7B"): (QWEN_ROOT / "ts_qwen_control_full_1.jsonl", GPT_ROOT / "ts_qwen_control_full_1.jsonl"),
}

def load(path: Path, score_name: str) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    frame = pd.DataFrame(rows)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame = frame.rename(columns={"trajectory_score": score_name})
    return frame[["sample_id", "topic_id", "template_id", "condition", score_name]]


def correlation_record(group: pd.DataFrame, split: str, model: str, condition: str) -> dict:
    pearson = stats.pearsonr(group["qwen_judge_score"], group["gpt4o_judge_score"])
    spearman = stats.spearmanr(group["qwen_judge_score"], group["gpt4o_judge_score"])
    difference = group["gpt4o_judge_score"] - group["qwen_judge_score"]
    return {
        "split": split,
        "candidate_model": model,
        "condition": condition,
        "rows": len(group),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "mean_qwen_judge": float(group["qwen_judge_score"].mean()),
        "mean_gpt4o_judge": float(group["gpt4o_judge_score"].mean()),
        "mean_difference_gpt_minus_qwen": float(difference.mean()),
        "mae": float(difference.abs().mean()),
        "exact_agreement": float(np.isclose(difference, 0.0).mean()),
        "agreement_within_0_2": float(difference.abs().le(0.2 + 1e-9).mean()),
    }


def cluster_shift(frame: pd.DataFrame, judge_column: str) -> pd.Series:
    pivot = frame.pivot(
        index=["sample_id", "template_id"],
        columns="condition",
        values=judge_column,
    )
    return (pivot["B"] - pivot["A"]).groupby(level="sample_id").mean()


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array)))
    margin = float(stats.t.ppf(0.975, len(array) - 1) * se)
    p = float(stats.ttest_1samp(array, 0.0).pvalue)
    return mean, mean - margin, mean + margin, p


def main() -> None:
    frames: list[pd.DataFrame] = []
    alignment: list[dict] = []
    for (split, model), (qwen_path, gpt_path) in FILES.items():
        qwen = load(qwen_path, "qwen_judge_score")
        gpt = load(gpt_path, "gpt4o_judge_score")
        keys = ["sample_id", "topic_id", "template_id", "condition"]
        merged = qwen.merge(gpt, on=keys, how="outer", validate="one_to_one", indicator=True)
        alignment.append(
            {
                "split": split,
                "candidate_model": model,
                "qwen_rows": len(qwen),
                "gpt4o_rows": len(gpt),
                "matched_rows": int(merged["_merge"].eq("both").sum()),
                "unmatched_rows": int(~merged["_merge"].eq("both").sum()),
            }
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"Cross-judge alignment failure for {split}/{model}")
        merged = merged.drop(columns="_merge")
        merged["split"] = split
        merged["candidate_model"] = model
        frames.append(merged)
    frame = pd.concat(frames, ignore_index=True)

    records: list[dict] = []
    for (split, model), group in frame.groupby(["split", "candidate_model"], sort=False):
        records.append(correlation_record(group, split, model, "all"))
        for condition, condition_group in group.groupby("condition", sort=False):
            records.append(correlation_record(condition_group, split, model, str(condition)))
    correlation = pd.DataFrame(records)
    correlation.to_csv(OUT_DIR / "cross_judge_correlations.csv", index=False)
    pd.DataFrame(alignment).to_csv(OUT_DIR / "cross_judge_alignment.csv", index=False)

    template = (
        frame.groupby(["split", "candidate_model", "condition", "template_id"], sort=False)
        .agg(
            rows=("sample_id", "size"),
            qwen_judge_mean=("qwen_judge_score", "mean"),
            gpt4o_judge_mean=("gpt4o_judge_score", "mean"),
        )
        .reset_index()
    )
    template["gpt_minus_qwen"] = template["gpt4o_judge_mean"] - template["qwen_judge_mean"]
    template.to_csv(OUT_DIR / "cross_judge_template_means.csv", index=False)

    shift_records: list[dict] = []
    for (split, model), group in frame.groupby(["split", "candidate_model"], sort=False):
        for judge_label, column in [
            ("Qwen2.5-32B", "qwen_judge_score"),
            ("GPT-4o", "gpt4o_judge_score"),
        ]:
            differences = cluster_shift(group, column)
            estimate, low, high, p = mean_ci(differences)
            shift_records.append(
                {
                    "split": split,
                    "candidate_model": model,
                    "judge": judge_label,
                    "clusters": len(differences),
                    "condition_shift_B_minus_A": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "p_ttest": p,
                }
            )
    shifts = pd.DataFrame(shift_records)
    shifts.to_csv(OUT_DIR / "cross_judge_condition_shifts.csv", index=False)

    abductive = correlation[
        correlation["split"].eq("abductive") & correlation["condition"].eq("all")
    ]
    lines = [
        "# Cross-judge robustness: Qwen2.5-32B vs GPT-4o",
        "",
        "All 8,540 rows aligned one-to-one across the original Qwen2.5-32B judge files and the new blind GPT-4o judge files.",
        "",
        "| Candidate outputs | Pearson r | Spearman rho | Mean Qwen-judge TS | Mean GPT-4o TS | MAE | Within 0.2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in abductive.iterrows():
        lines.append(
            f"| {row['candidate_model']} | {row['pearson_r']:.3f} | {row['spearman_rho']:.3f} | "
            f"{row['mean_qwen_judge']:.3f} | {row['mean_gpt4o_judge']:.3f} | "
            f"{row['mae']:.3f} | {row['agreement_within_0_2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Condition-shift replication",
            "",
            "| Candidate outputs | Judge | B - A shift [95% CI] | p |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in shifts[shifts["split"].eq("abductive")].iterrows():
        p_text = "<.001" if row["p_ttest"] < 0.001 else f"{row['p_ttest']:.3f}"
        lines.append(
            f"| {row['candidate_model']} | {row['judge']} | "
            f"{row['condition_shift_B_minus_A']:.3f} "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {p_text} |"
        )
    lines.extend(
        [
            "",
            "The two judges need not agree exactly because they operationalize the rubric differently. The robustness claim should therefore be based on rank/linear agreement and replication of the directional condition effect, not identical absolute scores.",
        ]
    )
    (OUT_DIR / "cross_judge_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(correlation.to_string(index=False))
    print(shifts.to_string(index=False))


if __name__ == "__main__":
    main()
