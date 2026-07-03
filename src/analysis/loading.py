"""Load autopsy artifacts from a run directory into flat per-rollout records.

The recorder writes one JSON per probe per recorded step
(``autopsy/rollouts/step_XXXX/probe_NNN.json``). This module flattens them so
the analysis passes can iterate over rollouts without caring about layout.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STEP_DIR_RE = re.compile(r"^step_(\d+)$")


@dataclass
class RolloutRecord:
    step: int
    probe_id: str
    difficulty: str
    reward: float
    format_reward: float
    answer_reward: float
    response: str
    tokens: list[str] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def correct(self) -> bool:
        return self.reward >= 1.0


def _autopsy_dir(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    autopsy = run_dir / "autopsy"
    if not autopsy.is_dir():
        raise FileNotFoundError(
            f"No autopsy/ directory under {run_dir}. Was the run launched with "
            "--enable-autopsy-recorder?"
        )
    return autopsy


def load_probe_manifest(run_dir: str | Path) -> list[dict[str, Any]]:
    manifest_path = _autopsy_dir(run_dir) / "probe_set.json"
    return json.loads(manifest_path.read_text())


def load_rollout_records(
    run_dir: str | Path,
    max_step: int | None = None,
) -> list[RolloutRecord]:
    rollouts_dir = _autopsy_dir(run_dir) / "rollouts"
    if not rollouts_dir.is_dir():
        raise FileNotFoundError(f"No rollouts under {rollouts_dir}")

    records: list[RolloutRecord] = []
    for step_dir in sorted(rollouts_dir.iterdir()):
        match = STEP_DIR_RE.match(step_dir.name)
        if not step_dir.is_dir() or match is None:
            continue
        step = int(match.group(1))
        if max_step is not None and step > max_step:
            continue
        for probe_path in sorted(step_dir.glob("probe_*.json")):
            payload = json.loads(probe_path.read_text())
            difficulty = payload.get("meta", {}).get("difficulty_bucket", "unknown")
            for rollout in payload.get("rollouts", []):
                reward = rollout.get("reward", {})
                records.append(
                    RolloutRecord(
                        step=step,
                        probe_id=payload["probe_id"],
                        difficulty=difficulty,
                        reward=float(reward.get("reward", 0.0)),
                        format_reward=float(reward.get("format_reward", 0.0)),
                        answer_reward=float(reward.get("answer_reward", 0.0)),
                        response=rollout.get("response", ""),
                        tokens=rollout.get("response_tokens", []),
                        log_probs=rollout.get("response_log_probs", []),
                        entropies=rollout.get("response_entropies", []),
                    )
                )

    if not records:
        raise ValueError(f"No rollout records found under {rollouts_dir}")
    return records


def steps_in(records: list[RolloutRecord]) -> list[int]:
    return sorted({record.step for record in records})
