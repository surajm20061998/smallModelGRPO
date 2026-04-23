#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--metric", default="math_val_accuracy")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sort_run_name(name: str) -> tuple[int, str]:
    if name == "full":
        return (10**9, name)
    try:
        return (int(name), name)
    except ValueError:
        return (10**8, name)


def maybe_plot_curves(curve_data: dict[str, list[tuple[int, float]]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(8, 5))
    for run_name, points in sorted(curve_data.items(), key=lambda item: sort_run_name(item[0])):
        if not points:
            continue
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        plt.plot(xs, ys, marker="o", label=run_name)

    plt.xlabel("Optimizer step")
    plt.ylabel("MATH validation accuracy")
    plt.title("SFT validation curves by dataset size")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    return True


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_data: dict[str, list[tuple[int, float]]] = {}
    table_rows: list[dict] = []

    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: sort_run_name(p.name)):
        eval_history = load_jsonl(run_dir / "eval_history.jsonl")
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        points = []
        for row in eval_history:
            if args.metric in row:
                points.append((row["train_step"], row[args.metric]))
        curve_data[run_dir.name] = points

        table_rows.append(
            {
                "run_name": run_dir.name,
                "train_size": summary.get("train_size"),
                "best_selection_metric": summary.get("best_selection_metric"),
                "best_selection_value": summary.get("best_selection_value"),
                "intellect_test_accuracy": summary.get("intellect_test_accuracy"),
                "math_test_accuracy": summary.get("math_test_accuracy"),
                "optimizer_steps_completed": summary.get("optimizer_steps_completed"),
                "best_checkpoint_path": summary.get("best_checkpoint_path"),
            }
        )

    (output_dir / "sft_summary.json").write_text(json.dumps(table_rows, indent=2))

    with (output_dir / "sft_summary.tsv").open("w") as f:
        headers = [
            "run_name",
            "train_size",
            "best_selection_metric",
            "best_selection_value",
            "intellect_test_accuracy",
            "math_test_accuracy",
            "optimizer_steps_completed",
            "best_checkpoint_path",
        ]
        f.write("\t".join(headers) + "\n")
        for row in table_rows:
            f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")

    plot_created = maybe_plot_curves(curve_data, output_dir / "math_val_curves.png")

    print(f"Wrote summary JSON to {output_dir / 'sft_summary.json'}")
    print(f"Wrote summary TSV to {output_dir / 'sft_summary.tsv'}")
    if plot_created:
        print(f"Wrote validation curve plot to {output_dir / 'math_val_curves.png'}")
    else:
        print("matplotlib not available; skipped plot generation")


if __name__ == "__main__":
    main()
