"""Hypothesis 3: is entropy collapse asymmetric?

Buckets per-token entropy three ways — position within the response
(early/mid/late thirds), token type (number / operator / structure / text,
classified from the BPE token string), and rollout correctness — and fits a
least-squares slope of mean entropy vs rollout step per bucket. "Collapse
asymmetry" then reads directly as slope differences between buckets.
"""

import re
from typing import Any

from src.analysis.loading import RolloutRecord, steps_in
from src.analysis.stats import ols_slope

# Qwen uses GPT-2-style byte-level BPE: "Ġ" marks a leading space, "Ċ" a newline.
_NUMBER_RE = re.compile(r"^\d+([.,]\d+)?$")
_OPERATOR_CHARS = set("+-*/×÷=()")
_STRUCTURE_TOKENS = {"<answer>", "</answer>", "<", ">", "answer", "step", ":", "steps"}

POSITION_BUCKETS = ("early", "mid", "late")
TOKEN_TYPES = ("number", "operator", "structure", "text")


def classify_token(token: str) -> str:
    stripped = token.replace("Ġ", "").replace("Ċ", "").strip()
    if not stripped:
        return "structure"  # pure whitespace/newline tokens
    lowered = stripped.lower()
    if lowered in _STRUCTURE_TOKENS:
        return "structure"
    if _NUMBER_RE.match(stripped):
        return "number"
    if all(ch in _OPERATOR_CHARS for ch in stripped):
        return "operator"
    return "text"


def position_bucket(index: int, length: int) -> str:
    if length <= 0:
        return "early"
    third = index * 3 // length
    return POSITION_BUCKETS[min(third, 2)]


def analyze_entropy(records: list[RolloutRecord]) -> dict[str, Any]:
    steps = steps_in(records)
    per_step: list[dict[str, Any]] = []

    for step in steps:
        step_records = [r for r in records if r.step == step and r.entropies]

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        def add(bucket: str, value: float) -> None:
            sums[bucket] = sums.get(bucket, 0.0) + value
            counts[bucket] = counts.get(bucket, 0) + 1

        for record in step_records:
            length = len(record.entropies)
            correctness = "correct" if record.correct else "incorrect"
            for idx, entropy_value in enumerate(record.entropies):
                add("overall", entropy_value)
                add(f"position/{position_bucket(idx, length)}", entropy_value)
                add(f"correctness/{correctness}", entropy_value)
                if idx < len(record.tokens):
                    add(f"token_type/{classify_token(record.tokens[idx])}", entropy_value)

        entry: dict[str, Any] = {"step": step, "n_rollouts": len(step_records)}
        for bucket, total in sums.items():
            entry[bucket] = {"mean_entropy": total / counts[bucket], "n_tokens": counts[bucket]}
        per_step.append(entry)

    all_buckets = ["overall"]
    all_buckets += [f"position/{b}" for b in POSITION_BUCKETS]
    all_buckets += [f"token_type/{t}" for t in TOKEN_TYPES]
    all_buckets += ["correctness/correct", "correctness/incorrect"]

    slopes: dict[str, Any] = {}
    for bucket in all_buckets:
        xs, ys = [], []
        for entry in per_step:
            if bucket in entry:
                xs.append(float(entry["step"]))
                ys.append(entry[bucket]["mean_entropy"])
        slopes[bucket] = ols_slope(xs, ys)

    return {"per_step": per_step, "slopes": slopes}
