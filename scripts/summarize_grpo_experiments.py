import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument(
        "--group-by",
        default="learning_rate",
        choices=[
            "learning_rate",
            "loss_type",
            "length_normalization",
            "std_normalization",
            "run_name",
        ],
    )
    parser.add_argument(
        "--metric",
        default="countdown_dev_accuracy",
        choices=[
            "countdown_dev_accuracy",
            "countdown_dev_answer_rate",
            "countdown_dev_format_rate",
        ],
    )
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_run_dirs(base: Path) -> list[Path]:
    run_dirs = []
    for config_path in base.rglob("config.json"):
        run_dirs.append(config_path.parent)
    return sorted(set(run_dirs))


def format_label(config: dict[str, Any], run_dir: Path, group_by: str) -> str:
    if group_by == "learning_rate":
        value = config.get("learning_rate")
        return f"lr={value:g}" if isinstance(value, (int, float)) else run_dir.name
    if group_by == "loss_type":
        return str(config.get("loss_type", run_dir.name))
    if group_by == "length_normalization":
        return str(config.get("length_normalization", run_dir.name))
    if group_by == "std_normalization":
        return f"std_norm={bool(config.get('normalize_by_std', True))}"
    if group_by == "run_name":
        return str(config.get("run_name") or run_dir.name)
    return run_dir.name


def summarize_run(run_dir: Path, group_by: str, metric: str) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    eval_history_path = run_dir / "eval_history.jsonl"

    config = load_json(config_path)
    label = format_label(config, run_dir, group_by)

    summary = load_json(summary_path) if summary_path.exists() else {}
    eval_rows = load_jsonl(eval_history_path) if eval_history_path.exists() else []

    best_metric = None
    final_metric = None
    best_step = None
    final_step = None
    xs: list[int] = []
    ys: list[float] = []

    for row in eval_rows:
        if metric not in row:
            continue
        xs.append(int(row["rollout_step"]))
        ys.append(float(row[metric]))

    if ys:
        best_metric = max(ys)
        final_metric = ys[-1]
        best_step = xs[ys.index(best_metric)]
        final_step = xs[-1]

    return {
        "run_dir": str(run_dir),
        "label": label,
        "status": "completed" if summary_path.exists() else "incomplete",
        "learning_rate": config.get("learning_rate"),
        "loss_type": config.get("loss_type"),
        "length_normalization": config.get("length_normalization"),
        "normalize_by_std": config.get("normalize_by_std"),
        "best_dev_accuracy": summary.get("best_dev_accuracy", best_metric),
        "final_dev_accuracy": summary.get("final_dev_accuracy", final_metric),
        "final_dev_format_rate": summary.get("final_dev_format_rate"),
        "final_dev_answer_rate": summary.get("final_dev_answer_rate"),
        "countdown_test_accuracy": summary.get("countdown_test_accuracy"),
        "countdown_test_format_rate": summary.get("countdown_test_format_rate"),
        "countdown_test_answer_rate": summary.get("countdown_test_answer_rate"),
        "train_steps_completed": summary.get("train_steps_completed"),
        "rollout_steps_completed": summary.get("rollout_steps_completed"),
        "curve_x": xs,
        "curve_y": ys,
        "best_metric_step": best_step,
        "final_metric_step": final_step,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "label",
        "status",
        "learning_rate",
        "loss_type",
        "length_normalization",
        "normalize_by_std",
        "best_dev_accuracy",
        "final_dev_accuracy",
        "final_dev_format_rate",
        "final_dev_answer_rate",
        "countdown_test_accuracy",
        "countdown_test_format_rate",
        "countdown_test_answer_rate",
        "train_steps_completed",
        "rollout_steps_completed",
        "best_metric_step",
        "final_metric_step",
        "run_dir",
    ]
    with path.open("w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            values = [row.get(column, "") for column in columns]
            f.write("\t".join("" if value is None else str(value) for value in values) + "\n")


def maybe_plot(path: Path, rows: list[dict[str, Any]], metric: str, group_by: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(9, 5.5))
    for row in rows:
        xs = row["curve_x"]
        ys = row["curve_y"]
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=row["label"])

    ylabel = metric.replace("_", " ")
    plt.xlabel("Rollout step")
    plt.ylabel(ylabel)
    plt.title(f"GRPO validation curves by {group_by.replace('_', ' ')}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    return True


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_prefix = Path(args.output_prefix) if args.output_prefix else runs_dir / f"grpo_{args.group_by}"

    run_dirs = find_run_dirs(runs_dir)
    rows = [summarize_run(run_dir, args.group_by, args.metric) for run_dir in run_dirs]
    rows.sort(key=lambda row: row["label"])

    json_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    tsv_path = output_prefix.with_name(output_prefix.name + "_summary.tsv")
    plot_path = output_prefix.with_name(output_prefix.name + "_curves.png")

    json_rows = []
    for row in rows:
        clean = dict(row)
        clean.pop("curve_x", None)
        clean.pop("curve_y", None)
        json_rows.append(clean)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(json_rows, f, indent=2)
    write_tsv(tsv_path, rows)

    print(f"Wrote summary JSON to {json_path}")
    print(f"Wrote summary TSV to {tsv_path}")
    if maybe_plot(plot_path, rows, args.metric, args.group_by):
        print(f"Wrote plot to {plot_path}")
    else:
        print("matplotlib not available; skipped plot generation")


if __name__ == "__main__":
    main()
