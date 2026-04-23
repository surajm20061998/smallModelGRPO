# An Empirical Autopsy of Small-Model GRPO

This repository reproduces SFT + GRPO training and adds instrumentation for
longitudinal behavior analysis (rollouts, rewards, token log-probs, entropy).

The Python entrypoints are set up for `uv run python -m ...` usage from a fresh
checkout. If `data-distrib/` is missing but `data.tgz` is present, the required
local dataset files are extracted automatically on first use.

## Core entrypoints

- `src.train.run_sft`: supervised fine-tuning on Intellect + MATH evaluation
- `src.train.run_grpo`: GRPO training on Countdown + evaluation
- `src.infer.infer_batch`: batched inference analysis on MATH prompts
- `src.eval.evaluate_math`: quick MATH/Intellect eval script

## Phase-1 autopsy instrumentation

`run_grpo` supports opt-in fixed-probe logging:

- fixed held-out probe set (`probe_set.json`)
- rollout artifacts per checkpoint step (`autopsy/rollouts/step_XXXX/`)
- per-token tensors (`autopsy/tensors/step_XXXX/rollout_tensors.pt`)
- periodic checkpoint snapshots (`checkpoints/autopsy_step_XXXX/`)
- scalar log history (`autopsy_history.jsonl`)

### Recommended pilot command

```bash
uv run python -m src.train.run_grpo \
  --output-dir runs/autopsy_v1/pilot_seed42 \
  --run-name autopsy_pilot_seed42 \
  --num-rollout-steps 20 \
  --eval-every 5 \
  --enable-autopsy-recorder \
  --autopsy-every 5 \
  --autopsy-num-probe-prompts 50 \
  --autopsy-probe-split dev \
  --autopsy-probe-seed 123 \
  --autopsy-checkpoint-every 10 \
  --wandb-mode offline
```

### Key recorder flags

- `--enable-autopsy-recorder`: turn instrumentation on
- `--autopsy-every`: record every N rollout steps
- `--autopsy-num-probe-prompts`: fixed probe-set size
- `--autopsy-probe-split`: source split (`train`/`dev`/`test`)
- `--autopsy-probe-seed`: deterministic probe-set seed
- `--autopsy-group-size`: rollouts per probe prompt (defaults to `--group-size`)
- `--autopsy-checkpoint-every`: save model snapshots every N rollout steps

## Weights & Biases tracking defaults

Training scripts now default to:

- `entity=sm12377-new-york-university`
- `project=smallModelGrpo`

and upload full run output directories as artifacts by default:

- use `--no-wandb-log-output-artifact` to skip artifact upload
- use `--wandb-mode disabled` to disable W&B entirely

## One-command autopsy suite

Use:

```bash
bash scripts/run_autopsy_suite.sh
```

This launches:

1. pilot runs (default seeds: `42 43 44`, 20 steps)
2. full runs (default seeds: `42 43 44`, 200 steps)
3. one off-policy contrast run (default seed: first seed)

Useful overrides:

- `AUTOPSY_OUT_ROOT` (default `runs/autopsy_v1`)
- `AUTOPSY_SEEDS` (default `"42 43 44"`)
- `WANDB_MODE` (default `online`)
- `CUDA_DEVICES` (default `0,1`)
- `RUN_PILOT`, `RUN_FULL`, `RUN_OFFPOLICY` (set to `0` to skip a stage)
