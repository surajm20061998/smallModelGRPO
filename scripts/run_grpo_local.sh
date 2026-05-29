#!/usr/bin/env bash
# Local GRPO launcher for Apple-silicon (MPS) / CPU — no CUDA, no vLLM, no W&B.
# Uses the HuggingFace generate() rollout backend (src/infer/hf_generate.py).
#
# Quick smoke test (2 rollout steps, tiny probe set):
#   SMOKE=1 scripts/run_grpo_local.sh
#
# Full local run (defaults below), override anything via env vars:
#   NUM_ROLLOUT_STEPS=50 BACKEND=mps scripts/run_grpo_local.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
BACKEND="${BACKEND:-mps}"          # mps | cpu | cuda
DTYPE="${DTYPE:-float32}"          # float32 is safest for MPS/CPU training
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
OLD_LOGPROB_BATCH_SIZE="${OLD_LOGPROB_BATCH_SIZE:-4}"

NUM_ROLLOUT_STEPS="${NUM_ROLLOUT_STEPS:-50}"
EVAL_EVERY="${EVAL_EVERY:-10}"
# Eval sets default to A100-scale (256/1024); cap them hard for local runs since
# greedy eval generation is a major wall-clock cost on MPS.
DEV_MAX_EXAMPLES="${DEV_MAX_EXAMPLES:-32}"
TEST_MAX_EXAMPLES="${TEST_MAX_EXAMPLES:-64}"

AUTOPSY_EVERY="${AUTOPSY_EVERY:-5}"
AUTOPSY_NUM_PROBE_PROMPTS="${AUTOPSY_NUM_PROBE_PROMPTS:-8}"
AUTOPSY_PROBE_SPLIT="${AUTOPSY_PROBE_SPLIT:-dev}"
AUTOPSY_PROBE_SEED="${AUTOPSY_PROBE_SEED:-123}"
AUTOPSY_CHECKPOINT_EVERY="${AUTOPSY_CHECKPOINT_EVERY:-25}"
AUTOPSY_LOGPROB_BATCH_SIZE="${AUTOPSY_LOGPROB_BATCH_SIZE:-2}"

SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-runs/local}"
RUN_NAME="${RUN_NAME:-grpo_local_seed${SEED}}"

# Smoke mode: just enough to confirm the loop produces autopsy artifacts.
if [[ "${SMOKE:-0}" == "1" ]]; then
  NUM_ROLLOUT_STEPS=2
  EVAL_EVERY=2
  AUTOPSY_EVERY=1
  AUTOPSY_NUM_PROBE_PROMPTS=4
  AUTOPSY_CHECKPOINT_EVERY=2
  MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-64}"
  RUN_NAME="grpo_local_smoke"
  OUT_ROOT="${OUT_ROOT:-runs/local}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/$RUN_NAME}"

# Allow CPU fallback for any MPS-unsupported op rather than hard-failing.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "========================================"
echo "Local GRPO run"
echo "  repo root:    $REPO_ROOT"
echo "  model:        $MODEL_ID"
echo "  backend:      $BACKEND ($DTYPE)"
echo "  output dir:   $OUTPUT_DIR"
echo "  rollout steps:$NUM_ROLLOUT_STEPS  (batch $ROLLOUT_BATCH_SIZE x group $GROUP_SIZE)"
echo "  max new tok:  $MAX_NEW_TOKENS"
echo "  smoke mode:   ${SMOKE:-0}"
echo "========================================"

python -m src.train.run_grpo \
  --model-id "$MODEL_ID" \
  --backend "$BACKEND" \
  --dtype "$DTYPE" \
  --gen-batch-size "$GEN_BATCH_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --seed "$SEED" \
  --num-rollout-steps "$NUM_ROLLOUT_STEPS" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --group-size "$GROUP_SIZE" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --microbatch-size "$MICROBATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --old-logprob-batch-size "$OLD_LOGPROB_BATCH_SIZE" \
  --countdown-dev-max-examples "$DEV_MAX_EXAMPLES" \
  --countdown-test-max-examples "$TEST_MAX_EXAMPLES" \
  --eval-every "$EVAL_EVERY" \
  --gradient-checkpointing \
  --enable-autopsy-recorder \
  --autopsy-every "$AUTOPSY_EVERY" \
  --autopsy-num-probe-prompts "$AUTOPSY_NUM_PROBE_PROMPTS" \
  --autopsy-probe-split "$AUTOPSY_PROBE_SPLIT" \
  --autopsy-probe-seed "$AUTOPSY_PROBE_SEED" \
  --autopsy-checkpoint-every "$AUTOPSY_CHECKPOINT_EVERY" \
  --autopsy-logprob-batch-size "$AUTOPSY_LOGPROB_BATCH_SIZE" \
  --wandb-mode disabled \
  "$@"

echo ""
echo "Local GRPO run complete. Artifacts under: $OUTPUT_DIR"
