#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="/scratch/$USER/grpo_runs/offpolicy"

CUDA_VISIBLE_DEVICES=0,1 uv run python -m src.train.run_grpo \
  --prompt-path "$REPO_ROOT/configs/prompts/countdown.prompt" \
  --countdown-train-path "$REPO_ROOT/data-distrib/countdown/train_10k.parquet" \
  --countdown-dev-path "$REPO_ROOT/data-distrib/countdown/dev.parquet" \
  --countdown-test-path "$REPO_ROOT/data-distrib/countdown/test.parquet" \
  --output-dir "$OUT" \
  --run-name "grpo_offpolicy_countdown" \
  --num-rollout-steps 100 \
  --rollout-batch-size 64 \
  --group-size 4 \
  --epochs-per-rollout-batch 4 \
  --train-batch-size 16 \
  --microbatch-size 1 \
  --learning-rate 1e-6 \
  --loss-type grpo_clip \
  --cliprange 0.1 \
  --rollout-temperature 1.0 \
  --rollout-min-tokens 4 \
  --max-new-tokens 256 \
  --eval-every 10 \
  --countdown-dev-max-examples 256 \
  --countdown-test-max-examples 1024 \
  --policy-device cuda:0 \
  --vllm-device cuda:1 \
  --gradient-checkpointing \
  --normalize-by-std \
  --eval-before-train
