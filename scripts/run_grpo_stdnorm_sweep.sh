#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="${GRPO_STDNORM_SWEEP_OUT:-/scratch/$USER/grpo_experiments/std_norm_sweep}"
mkdir -p "$OUT_ROOT"

STD_OPTIONS_STR="${GRPO_STD_OPTIONS:-true false}"
IFS=' ' read -r -a STD_OPTIONS <<< "$STD_OPTIONS_STR"

LEARNING_RATE="${GRPO_STDNORM_LR:-1e-5}"
LOSS_TYPE="${GRPO_STDNORM_LOSS_TYPE:-reinforce_with_baseline}"
LENGTH_NORMALIZATION="${GRPO_STDNORM_LENGTH_NORMALIZATION:-masked_mean}"
NUM_ROLLOUT_STEPS="${GRPO_NUM_ROLLOUT_STEPS:-200}"
ROLLOUT_BATCH_SIZE="${GRPO_ROLLOUT_BATCH_SIZE:-16}"
GROUP_SIZE="${GRPO_GROUP_SIZE:-8}"
EPOCHS_PER_ROLLOUT_BATCH="${GRPO_EPOCHS_PER_ROLLOUT_BATCH:-1}"
TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-16}"
MICROBATCH_SIZE="${GRPO_MICROBATCH_SIZE:-1}"
ROLLOUT_TEMPERATURE="${GRPO_ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_MIN_TOKENS="${GRPO_ROLLOUT_MIN_TOKENS:-4}"
MAX_NEW_TOKENS="${GRPO_MAX_NEW_TOKENS:-1024}"
EVAL_EVERY="${GRPO_EVAL_EVERY:-5}"
DEV_MAX_EXAMPLES="${GRPO_DEV_MAX_EXAMPLES:-256}"
TEST_MAX_EXAMPLES="${GRPO_TEST_MAX_EXAMPLES:-1024}"
POLICY_DEVICE="${GRPO_POLICY_DEVICE:-cuda:0}"
VLLM_DEVICE="${GRPO_VLLM_DEVICE:-cuda:1}"
WANDB_MODE="${WANDB_MODE:-online}"
MODEL_ID="${GRPO_MODEL_ID:-Qwen/Qwen2.5-Math-1.5B-Instruct}"

SWEEP_LOG="$OUT_ROOT/sweep.log"
STATUS_JSONL="$OUT_ROOT/status.jsonl"

echo "Starting GRPO std-normalization sweep at $(date)" | tee -a "$SWEEP_LOG"
echo "normalize_by_std options: ${STD_OPTIONS[*]}" | tee -a "$SWEEP_LOG"
echo "Learning rate: $LEARNING_RATE | loss type: $LOSS_TYPE | length normalization: $LENGTH_NORMALIZATION" | tee -a "$SWEEP_LOG"

for STD_OPTION in "${STD_OPTIONS[@]}"; do
  RUN_NAME_SUFFIX="$STD_OPTION"
  RUN_DIR="$OUT_ROOT/std_norm_${RUN_NAME_SUFFIX}"
  RUN_LOG="$OUT_ROOT/std_norm_${RUN_NAME_SUFFIX}.log"
  RUN_NAME="grpo_std_norm_${RUN_NAME_SUFFIX}"

  STD_FLAG="--no-normalize-by-std"
  if [[ "$STD_OPTION" == "true" ]]; then
    STD_FLAG="--normalize-by-std"
  fi

  mkdir -p "$RUN_DIR"
  echo "" | tee -a "$SWEEP_LOG"
  echo "=== Running normalize_by_std=$STD_OPTION | out=$RUN_DIR ===" | tee -a "$SWEEP_LOG"

  if CUDA_VISIBLE_DEVICES=0,1 uv run python -m src.train.run_grpo \
    --model-id "$MODEL_ID" \
    --prompt-path "$REPO_ROOT/configs/prompts/countdown.prompt" \
    --countdown-train-path "$REPO_ROOT/data-distrib/countdown/train_10k.parquet" \
    --countdown-dev-path "$REPO_ROOT/data-distrib/countdown/dev.parquet" \
    --countdown-test-path "$REPO_ROOT/data-distrib/countdown/test.parquet" \
    --output-dir "$RUN_DIR" \
    --run-name "$RUN_NAME" \
    --num-rollout-steps "$NUM_ROLLOUT_STEPS" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --group-size "$GROUP_SIZE" \
    --epochs-per-rollout-batch "$EPOCHS_PER_ROLLOUT_BATCH" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --microbatch-size "$MICROBATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --loss-type "$LOSS_TYPE" \
    --length-normalization "$LENGTH_NORMALIZATION" \
    --cliprange 0.2 \
    --rollout-temperature "$ROLLOUT_TEMPERATURE" \
    --rollout-min-tokens "$ROLLOUT_MIN_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --eval-every "$EVAL_EVERY" \
    --countdown-dev-max-examples "$DEV_MAX_EXAMPLES" \
    --countdown-test-max-examples "$TEST_MAX_EXAMPLES" \
    --policy-device "$POLICY_DEVICE" \
    --vllm-device "$VLLM_DEVICE" \
    --gradient-checkpointing \
    $STD_FLAG \
    --eval-before-train \
    --wandb-mode "$WANDB_MODE" \
    > "$RUN_LOG" 2>&1; then
    STATUS="completed"
  else
    STATUS="failed"
  fi

  printf '{"normalize_by_std":"%s","learning_rate":"%s","loss_type":"%s","length_normalization":"%s","run_dir":"%s","status":"%s"}\n' \
    "$STD_OPTION" "$LEARNING_RATE" "$LOSS_TYPE" "$LENGTH_NORMALIZATION" "$RUN_DIR" "$STATUS" >> "$STATUS_JSONL"
  echo "=== Finished normalize_by_std=$STD_OPTION | status=$STATUS ===" | tee -a "$SWEEP_LOG"
done

echo "Std-normalization sweep finished at $(date)" | tee -a "$SWEEP_LOG"
