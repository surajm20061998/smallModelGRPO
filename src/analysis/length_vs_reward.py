"""Hypothesis 1: is length growth explained by correctness, or is it reward hacking?

Computes per-step response-length distributions conditioned on rollout reward
(correct vs incorrect), plus the length-vs-step slope within each population.
If length grows with a similar slope in the *incorrect* population, growth is
not explained by improved reasoning.
"""

from typing import Any

from src.analysis.loading import RolloutRecord, steps_in
from src.analysis.stats import ols_slope, summarize


def analyze_length_vs_reward(records: list[RolloutRecord]) -> dict[str, Any]:
    steps = steps_in(records)
    per_step: list[dict[str, Any]] = []

    for step in steps:
        step_records = [r for r in records if r.step == step]
        lengths_all = [float(r.num_tokens) for r in step_records]
        lengths_correct = [float(r.num_tokens) for r in step_records if r.correct]
        lengths_incorrect = [float(r.num_tokens) for r in step_records if not r.correct]
        per_step.append(
            {
                "step": step,
                "accuracy": sum(1 for r in step_records if r.correct) / len(step_records),
                "format_rate": sum(r.format_reward for r in step_records) / len(step_records),
                "length_all": summarize(lengths_all),
                "length_correct": summarize(lengths_correct),
                "length_incorrect": summarize(lengths_incorrect),
            }
        )

    def slope_for(key: str) -> dict[str, Any]:
        xs = [float(entry["step"]) for entry in per_step]
        ys = [entry[key]["mean"] if entry[key]["n"] > 0 else None for entry in per_step]
        return ols_slope(xs, ys)

    by_difficulty: dict[str, dict[str, Any]] = {}
    for bucket in sorted({r.difficulty for r in records}):
        xs, ys = [], []
        for step in steps:
            bucket_lengths = [
                float(r.num_tokens) for r in records if r.step == step and r.difficulty == bucket
            ]
            if bucket_lengths:
                xs.append(float(step))
                ys.append(sum(bucket_lengths) / len(bucket_lengths))
        by_difficulty[bucket] = ols_slope(xs, ys)

    return {
        "per_step": per_step,
        "length_slope_all": slope_for("length_all"),
        "length_slope_correct": slope_for("length_correct"),
        "length_slope_incorrect": slope_for("length_incorrect"),
        "length_slope_by_difficulty": by_difficulty,
    }
