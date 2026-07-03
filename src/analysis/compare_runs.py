"""Side-by-side comparison of two or more analyzed runs (e.g. an ablation A/B).

Runs the per-run analysis if it hasn't been produced yet, then renders a
markdown table of the headline metrics from each run.

Usage:
    python -m src.analysis.compare_runs \
        --run-dirs runs/local/stdnorm_on_s42 runs/local/stdnorm_off_s42 \
        --output runs/local/stdnorm_comparison.md
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.analysis.loading import load_rollout_records
from src.analysis.run_analysis import analyze_run, compute_analyses


def _load_or_analyze(run_dir: Path, max_step: int | None = None) -> dict[str, dict[str, Any]]:
    if max_step is not None:
        # Trimmed window: compute in memory, don't clobber the persisted full analysis.
        return compute_analyses(load_rollout_records(run_dir, max_step=max_step))
    analysis_dir = run_dir / "analysis"
    if not (analysis_dir / "entropy.json").exists():
        analyze_run(run_dir)
    return {
        name: json.loads((analysis_dir / f"{name}.json").read_text())
        for name in ("length_vs_reward", "behaviors", "entropy")
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def headline_metrics(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    length = results["length_vs_reward"]
    behaviors = results["behaviors"]
    entropy = results["entropy"]

    per_step = length["per_step"]
    first, last = per_step[0], per_step[-1]

    metrics: dict[str, Any] = {
        "probe accuracy (first → last)": f"{_fmt(first['accuracy'], 3)} → {_fmt(last['accuracy'], 3)}",
        "format rate (first → last)": f"{_fmt(first['format_rate'], 3)} → {_fmt(last['format_rate'], 3)}",
        "mean length (first → last)": (
            f"{_fmt(first['length_all'].get('mean'), 1)} → {_fmt(last['length_all'].get('mean'), 1)}"
        ),
        "length slope: all": _fmt(length["length_slope_all"].get("slope")),
        "length slope: correct": _fmt(length["length_slope_correct"].get("slope")),
        "length slope: incorrect": _fmt(length["length_slope_incorrect"].get("slope")),
        "entropy slope: overall": _fmt(entropy["slopes"]["overall"].get("slope")),
        "entropy slope: position/early": _fmt(entropy["slopes"]["position/early"].get("slope")),
        "entropy slope: position/late": _fmt(entropy["slopes"]["position/late"].get("slope")),
        "entropy slope: numbers": _fmt(entropy["slopes"]["token_type/number"].get("slope")),
        "entropy slope: operators": _fmt(entropy["slopes"]["token_type/operator"].get("slope")),
        "entropy slope: text": _fmt(entropy["slopes"]["token_type/text"].get("slope")),
        "entropy slope: correct rollouts": _fmt(entropy["slopes"]["correctness/correct"].get("slope")),
        "entropy slope: incorrect rollouts": _fmt(entropy["slopes"]["correctness/incorrect"].get("slope")),
    }
    for family, entry in behaviors["first_vs_last"].items():
        metrics[f"{family} freq (first → last)"] = (
            f"{_fmt(entry['first_step_frequency'], 3)} → {_fmt(entry['last_step_frequency'], 3)}"
        )
    return metrics


def build_comparison(run_dirs: list[Path], max_step: int | None = None) -> str:
    per_run = {
        run_dir.name: headline_metrics(_load_or_analyze(run_dir, max_step=max_step))
        for run_dir in run_dirs
    }
    names = list(per_run)
    all_metrics = list(per_run[names[0]])

    lines = ["# Run comparison", ""]
    if max_step is not None:
        lines.append(f"(window trimmed to rollout steps ≤ {max_step})")
        lines.append("")
    lines.append("| metric | " + " | ".join(names) + " |")
    lines.append("| --- |" + " --- |" * len(names))
    for metric in all_metrics:
        row = " | ".join(str(per_run[name].get(metric, "—")) for name in names)
        lines.append(f"| {metric} | {row} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", default=None, help="Optional path to write the markdown table")
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Trim all runs to rollout steps <= this value for a fair common window.",
    )
    args = parser.parse_args()

    comparison = build_comparison([Path(d) for d in args.run_dirs], max_step=args.max_step)
    if args.output:
        Path(args.output).write_text(comparison)
        print(f"Comparison written to {args.output}")
    else:
        print(comparison)


if __name__ == "__main__":
    main()
