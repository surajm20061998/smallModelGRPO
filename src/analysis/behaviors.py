"""Hypothesis 2: do reasoning behaviors emerge in phases?

Regex detectors for three behavior families, counted per rollout and
aggregated to per-step frequencies. These are surface-pattern detectors; an
LLM-judge precision/recall validation layer is a planned follow-up, so treat
absolute frequencies with care and trust the *trends* more than the levels.
"""

import re
from typing import Any

from src.analysis.loading import RolloutRecord, steps_in

BEHAVIOR_PATTERNS: dict[str, list[str]] = {
    "backtracking": [
        r"\bwait\b",
        r"\bhmm+\b",
        r"\bactually\b",
        r"let me (try again|re-?consider|start over|redo)",
        r"try (again|another|a different)",
        r"(doesn'?t|does not|didn'?t|did not) work",
        r"(doesn'?t|does not) (equal|give|match)",
        r"(that'?s|this is) (not right|wrong|incorrect)",
        r"scratch that",
        r"not the target",
        r"too (big|small|large|high|low)\b",
    ],
    "verification": [
        r"let'?s? (me )?(check|verify|confirm)",
        r"\bverify(ing)?\b",
        r"double-?check",
        r"check(ing)? (the|my|this|that|if|whether)",
        r"to confirm",
        r"which (is|equals|gives) (the )?target",
        r"(equals|matches) the target",
        r"\bindeed\b",
        r"as (required|expected|desired)",
    ],
    "case_analysis": [
        r"\bcase \d",
        r"\boption \d",
        r"\balternativ",
        r"another (approach|way|option|combination)",
        r"(approach|attempt) \d",
        r"first,? (let'?s )?(try|attempt)",
        r"\bor we (can|could)\b",
        r"what if\b",
        r"if we (use|take|start|combine)",
        r"(other|different) (combination|possibilit)",
    ],
}

_COMPILED = {
    family: [re.compile(p, re.IGNORECASE) for p in patterns]
    for family, patterns in BEHAVIOR_PATTERNS.items()
}


def count_behaviors(text: str) -> dict[str, int]:
    return {
        family: sum(len(pattern.findall(text)) for pattern in patterns)
        for family, patterns in _COMPILED.items()
    }


def analyze_behaviors(records: list[RolloutRecord]) -> dict[str, Any]:
    steps = steps_in(records)
    families = list(BEHAVIOR_PATTERNS)
    per_step: list[dict[str, Any]] = []

    for step in steps:
        step_records = [r for r in records if r.step == step]
        counts = [count_behaviors(r.response) for r in step_records]
        entry: dict[str, Any] = {"step": step, "n_rollouts": len(step_records)}
        for family in families:
            family_counts = [c[family] for c in counts]
            entry[family] = {
                # fraction of rollouts showing the behavior at least once
                "rollout_frequency": sum(1 for c in family_counts if c > 0) / len(family_counts),
                "mean_matches_per_rollout": sum(family_counts) / len(family_counts),
            }
        per_step.append(entry)

    first_last: dict[str, Any] = {}
    if per_step:
        for family in families:
            first_last[family] = {
                "first_step_frequency": per_step[0][family]["rollout_frequency"],
                "last_step_frequency": per_step[-1][family]["rollout_frequency"],
            }

    return {
        "families": families,
        "per_step": per_step,
        "first_vs_last": first_last,
    }
