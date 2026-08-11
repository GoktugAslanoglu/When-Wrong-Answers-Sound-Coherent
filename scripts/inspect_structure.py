from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parent.parent
GPT_ROOT = ARTIFACT_ROOT / "data" / "gpt4o_blind"
FILES = {
    "llama_abductive": GPT_ROOT / "ts_llama_abductive_full_1.jsonl",
    "qwen_abductive": GPT_ROOT / "ts_qwen_abductive_full_1.jsonl",
    "llama_control": GPT_ROOT / "ts_llama_control_full_1.jsonl",
    "qwen_control": GPT_ROOT / "ts_qwen_control_full_1.jsonl",
}

def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


for name, path in FILES.items():
    rows = load(path)
    print(f"\n{name}: {len(rows)}")
    print("keys:", list(rows[0]))
    for field in ["model", "condition", "template_id", "topic_id", "is_correct", "parse_error", "has_causal_markers", "trajectory_score"]:
        values = Counter(row.get(field) for row in rows)
        print(field, dict(sorted(values.items(), key=lambda item: str(item[0]))))
    ids = Counter(str(row.get("sample_id")) for row in rows)
    print("sample_ids", len(ids), "multiplicity", dict(sorted(Counter(ids.values()).items())))

for left_name, right_name in [("llama_abductive", "qwen_abductive"), ("llama_control", "qwen_control")]:
    left = load(FILES[left_name])
    right = load(FILES[right_name])
    key = lambda row: (str(row.get("sample_id")), row.get("topic_id"), row.get("template_id"), row.get("condition"))
    left_keys = [key(row) for row in left]
    right_keys = [key(row) for row in right]
    print(f"\npairing {left_name} vs {right_name}")
    print("same ordered keys", left_keys == right_keys)
    print("unique left/right", len(set(left_keys)), len(set(right_keys)))
    print("set differences", len(set(left_keys) - set(right_keys)), len(set(right_keys) - set(left_keys)))
