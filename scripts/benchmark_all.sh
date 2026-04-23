#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="/scratch/$USER/grpo_runs/onpolicy"

CUDA_VISIBLE_DEVICES=0,1 uv run python -m src.train.run_grpo \
  --prompt-path "$REPO_ROOT/configs/prompts/countdown.prompt" \
  --countdown-train-path "$REPO_ROOT/data-distrib/countdown/train_10k.parquet" \
  --countdown-dev-path "$REPO_ROOT/data-distrib/countdown/dev.parquet" \
  --countdown-test-path "$REPO_ROOT/data-distrib/countdown/test.parquet" \
  --output-dir "$OUT" \
  --run-name "grpo_onpolicy_countdown" \
  --num-rollout-steps 200 \
  --rollout-batch-size 16 \
  --group-size 8 \
  --epochs-per-rollout-batch 1 \
  --train-batch-size 16 \
  --microbatch-size 1 \
  --learning-rate 1e-5 \
  --loss-type reinforce_with_baseline \
  --cliprange 0.2 \
  --rollout-temperature 0.7 \
  --rollout-min-tokens 4 \
  --max-new-tokens 1024 \
  --eval-every 5 \
  --countdown-dev-max-examples 256 \
  --countdown-test-max-examples 1024 \
  --policy-device cuda:0 \
  --vllm-device cuda:1 \
  --gradient-checkpointing \
  --normalize-by-std \
  --eval-before-train
