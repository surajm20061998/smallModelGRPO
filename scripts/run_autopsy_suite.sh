#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_ROOT="${AUTOPSY_OUT_ROOT:-runs/autopsy_v1}"
WANDB_MODE="${WANDB_MODE:-online}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
SEEDS_STR="${AUTOPSY_SEEDS:-42 43 44}"
IFS=' ' read -r -a SEEDS <<< "$SEEDS_STR"

PILOT_STEPS="${AUTOPSY_PILOT_STEPS:-20}"
FULL_STEPS="${AUTOPSY_FULL_STEPS:-200}"
OFFPOLICY_STEPS="${AUTOPSY_OFFPOLICY_STEPS:-100}"

EVAL_EVERY="${AUTOPSY_EVAL_EVERY:-5}"
AUTOPSY_EVERY="${AUTOPSY_EVERY:-10}"
AUTOPSY_NUM_PROBE_PROMPTS="${AUTOPSY_NUM_PROBE_PROMPTS:-50}"
AUTOPSY_PROBE_SPLIT="${AUTOPSY_PROBE_SPLIT:-dev}"
AUTOPSY_PROBE_SEED="${AUTOPSY_PROBE_SEED:-123}"
AUTOPSY_CHECKPOINT_EVERY="${AUTOPSY_CHECKPOINT_EVERY:-50}"
AUTOPSY_LOGPROB_BATCH_SIZE="${AUTOPSY_LOGPROB_BATCH_SIZE:-4}"

RUN_PILOT="${RUN_PILOT:-1}"
RUN_FULL="${RUN_FULL:-1}"
RUN_OFFPOLICY="${RUN_OFFPOLICY:-1}"

run_grpo_common() {
  local output_dir="$1"
  local run_name="$2"
  local seed="$3"
  shift 3

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" uv run python -m src.train.run_grpo \
    --output-dir "$output_dir" \
    --run-name "$run_name" \
    --seed "$seed" \
    --eval-every "$EVAL_EVERY" \
    --enable-autopsy-recorder \
    --autopsy-every "$AUTOPSY_EVERY" \
    --autopsy-num-probe-prompts "$AUTOPSY_NUM_PROBE_PROMPTS" \
    --autopsy-probe-split "$AUTOPSY_PROBE_SPLIT" \
    --autopsy-probe-seed "$AUTOPSY_PROBE_SEED" \
    --autopsy-checkpoint-every "$AUTOPSY_CHECKPOINT_EVERY" \
    --autopsy-logprob-batch-size "$AUTOPSY_LOGPROB_BATCH_SIZE" \
    --wandb-mode "$WANDB_MODE" \
    "$@"
}

echo "========================================"
echo "Running GRPO autopsy suite"
echo "Repo root: $REPO_ROOT"
echo "Output root: $OUT_ROOT"
echo "Seeds: ${SEEDS[*]}"
echo "W&B mode: $WANDB_MODE"
echo "CUDA devices: $CUDA_DEVICES"
echo "========================================"

if [[ "$RUN_PILOT" == "1" ]]; then
  echo ""
  echo "=== Stage 1/3: Pilot runs (${PILOT_STEPS} steps) ==="
  for SEED in "${SEEDS[@]}"; do
    run_grpo_common \
      "$OUT_ROOT/pilot_seed${SEED}" \
      "autopsy_pilot_seed${SEED}" \
      "$SEED" \
      --num-rollout-steps "$PILOT_STEPS" \
      --autopsy-every 5 \
      --autopsy-checkpoint-every 10
  done
fi

if [[ "$RUN_FULL" == "1" ]]; then
  echo ""
  echo "=== Stage 2/3: Full runs (${FULL_STEPS} steps) ==="
  for SEED in "${SEEDS[@]}"; do
    run_grpo_common \
      "$OUT_ROOT/full_seed${SEED}" \
      "autopsy_full_seed${SEED}" \
      "$SEED" \
      --num-rollout-steps "$FULL_STEPS"
  done
fi

if [[ "$RUN_OFFPOLICY" == "1" ]]; then
  echo ""
  echo "=== Stage 3/3: Off-policy contrast (${OFFPOLICY_STEPS} steps) ==="
  # Use first seed for the contrast run by default.
  CONTRAST_SEED="${SEEDS[0]}"
  run_grpo_common \
    "$OUT_ROOT/offpolicy_seed${CONTRAST_SEED}" \
    "autopsy_offpolicy_seed${CONTRAST_SEED}" \
    "$CONTRAST_SEED" \
    --num-rollout-steps "$OFFPOLICY_STEPS" \
    --rollout-batch-size 64 \
    --group-size 4 \
    --epochs-per-rollout-batch 4 \
    --loss-type grpo_clip \
    --learning-rate 1e-6 \
    --max-new-tokens 256 \
    --eval-every 10 \
    --autopsy-logprob-batch-size 2
fi

echo ""
echo "Autopsy suite complete."
