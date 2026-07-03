"""Phase-2 analysis CLI.

Consumes a training run directory's autopsy artifacts and writes per-hypothesis
JSON metrics plus a combined markdown report into ``<run-dir>/analysis/``.

Usage:
    python -m src.analysis.run_analysis --run-dir runs/local/<run-name>
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.analysis.behaviors import analyze_behaviors
from src.analysis.entropy import analyze_entropy
from src.analysis.length_vs_reward import analyze_length_vs_reward
from src.analysis.loading import load_rollout_records, steps_in


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _slope(entry: dict[str, Any]) -> str:
    return _fmt(entry.get("slope"), 4)


def build_report(
    run_dir: Path,
    length_results: dict[str, Any],
    behavior_results: dict[str, Any],
    entropy_results: dict[str, Any],
    num_rollouts: int,
    steps: list[int],
) -> str:
    lines: list[str] = []
    lines.append(f"# Autopsy analysis: `{run_dir.name}`")
    lines.append("")
    lines.append(
        f"Recorded steps: {steps[0]}–{steps[-1]} ({len(steps)} snapshots), "
        f"{num_rollouts} probe rollouts total."
    )
    lines.append("")

    # --- Hypothesis 1 ---
    lines.append("## H1 — Length vs reward (reward hacking)")
    lines.append("")
    lines.append("| step | accuracy | format | mean len (all) | mean len (correct) | mean len (incorrect) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for entry in length_results["per_step"]:
        correct_mean = entry["length_correct"].get("mean") if entry["length_correct"]["n"] else None
        incorrect_mean = entry["length_incorrect"].get("mean") if entry["length_incorrect"]["n"] else None
        lines.append(
            f"| {entry['step']} | {_fmt(entry['accuracy'])} | {_fmt(entry['format_rate'])} "
            f"| {_fmt(entry['length_all'].get('mean'), 1)} | {_fmt(correct_mean, 1)} "
            f"| {_fmt(incorrect_mean, 1)} |"
        )
    lines.append("")
    lines.append(
        f"Length slope (tokens/step): all = {_slope(length_results['length_slope_all'])}, "
        f"correct = {_slope(length_results['length_slope_correct'])}, "
        f"incorrect = {_slope(length_results['length_slope_incorrect'])}."
    )
    lines.append("")
    lines.append(
        "Reading: if the incorrect-population slope is comparable to (or larger than) "
        "the correct-population slope, length growth is not explained by improved "
        "reasoning — consistent with reward hacking."
    )
    lines.append("")

    # --- Hypothesis 2 ---
    lines.append("## H2 — Behavior emergence (regex detectors)")
    lines.append("")
    lines.append("| step | backtracking | verification | case_analysis |")
    lines.append("| --- | --- | --- | --- |")
    for entry in behavior_results["per_step"]:
        lines.append(
            f"| {entry['step']} "
            f"| {_fmt(entry['backtracking']['rollout_frequency'])} "
            f"| {_fmt(entry['verification']['rollout_frequency'])} "
            f"| {_fmt(entry['case_analysis']['rollout_frequency'])} |"
        )
    lines.append("")
    lines.append("(Values are the fraction of probe rollouts containing the behavior at least once.)")
    lines.append("")

    # --- Hypothesis 3 ---
    lines.append("## H3 — Entropy collapse asymmetry")
    lines.append("")
    lines.append("Entropy-vs-step slope per bucket (nats/step; more negative = faster collapse):")
    lines.append("")
    lines.append("| bucket | slope | snapshots |")
    lines.append("| --- | --- | --- |")
    for bucket, slope_entry in entropy_results["slopes"].items():
        lines.append(f"| {bucket} | {_slope(slope_entry)} | {slope_entry['n']} |")
    lines.append("")
    lines.append("Per-step overall mean entropy:")
    lines.append("")
    lines.append("| step | mean entropy (nats) | tokens |")
    lines.append("| --- | --- | --- |")
    for entry in entropy_results["per_step"]:
        overall = entry.get("overall")
        if overall:
            lines.append(f"| {entry['step']} | {_fmt(overall['mean_entropy'])} | {overall['n_tokens']} |")
    lines.append("")
    return "\n".join(lines)


def compute_analyses(records) -> dict[str, dict[str, Any]]:
    return {
        "length_vs_reward": analyze_length_vs_reward(records),
        "behaviors": analyze_behaviors(records),
        "entropy": analyze_entropy(records),
    }


def analyze_run(run_dir: str | Path, max_step: int | None = None) -> Path:
    run_dir = Path(run_dir)
    records = load_rollout_records(run_dir, max_step=max_step)
    steps = steps_in(records)

    results = compute_analyses(records)
    length_results = results["length_vs_reward"]
    behavior_results = results["behaviors"]
    entropy_results = results["entropy"]

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "length_vs_reward.json").write_text(json.dumps(length_results, indent=2))
    (out_dir / "behaviors.json").write_text(json.dumps(behavior_results, indent=2))
    (out_dir / "entropy.json").write_text(json.dumps(entropy_results, indent=2))

    report = build_report(
        run_dir, length_results, behavior_results, entropy_results, len(records), steps
    )
    (out_dir / "report.md").write_text(report)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Training run output directory")
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Ignore autopsy snapshots beyond this rollout step (for trimming A/B windows).",
    )
    args = parser.parse_args()

    out_dir = analyze_run(args.run_dir, max_step=args.max_step)
    print(f"Analysis written to {out_dir}")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
