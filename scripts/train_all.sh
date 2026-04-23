#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="/scratch/$USER/sft_runs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TUNE_DIR="$OUT/tuning"
TUNE_CANDIDATES="${TUNE_CANDIDATES:-1e-5:1:16,2e-5:1:32,5e-5:1:32,2e-5:2:16}"
TUNE_STEPS="${TUNE_STEPS:-50}"
TUNE_TRAIN_SIZE="${TUNE_TRAIN_SIZE:-256}"

echo "========================================"
echo "Running SFT hyperparameter tuning"
echo "Tuning output dir: $TUNE_DIR"
echo "========================================"

CUDA_VISIBLE_DEVICES=0,1 uv run python -m src.train.tune_sft \
  --intellect-train-path "$REPO_ROOT/data-distrib/intellect_math/train" \
  --output-dir "$TUNE_DIR" \
  --train-size "$TUNE_TRAIN_SIZE" \
  --max-optimizer-steps "$TUNE_STEPS" \
  --candidate-configs "$TUNE_CANDIDATES" \
  --gradient-checkpointing \
  --policy-device cuda:0

BEST_LR=$(python3 -c "import json; print(json.load(open('$TUNE_DIR/best_config.json'))['learning_rate'])")
BEST_MICROBATCH=$(python3 -c "import json; print(json.load(open('$TUNE_DIR/best_config.json'))['microbatch_size'])")
BEST_GRAD_ACCUM=$(python3 -c "import json; print(json.load(open('$TUNE_DIR/best_config.json'))['gradient_accumulation_steps'])")

echo "Selected hyperparameters:"
echo "  learning_rate=$BEST_LR"
echo "  microbatch_size=$BEST_MICROBATCH"
echo "  gradient_accumulation_steps=$BEST_GRAD_ACCUM"

SIZES=(128 256 512 1024 full)

for SIZE in "${SIZES[@]}"; do
  if [[ "$SIZE" == "full" ]]; then
    TRAIN_SIZE=-1
  else
    TRAIN_SIZE="$SIZE"
  fi

  echo "========================================"
  echo "Starting run for SIZE=$SIZE"
  echo "Repo root: $REPO_ROOT"
  echo "Output dir: $OUT/$SIZE"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=0,1 uv run python -m src.train.run_sft \
    --prompt-path "$REPO_ROOT/configs/prompts/intellect.prompt" \
    --intellect-train-path "$REPO_ROOT/data-distrib/intellect_math/train" \
    --intellect-dev-path "$REPO_ROOT/data-distrib/intellect_math/dev" \
    --intellect-test-path "$REPO_ROOT/data-distrib/intellect_math/test" \
    --math-val-split "train[-500:]" \
    --math-test-split "test" \
    --intellect-val-max-examples 500 \
    --math-val-max-examples 500 \
    --math-test-max-examples 500 \
    --output-dir "$OUT/$SIZE" \
    --run-name "sft_$SIZE" \
    --train-size "$TRAIN_SIZE" \
    --num-epochs 200 \
    --max-optimizer-steps 300 \
    --microbatch-size "$BEST_MICROBATCH" \
    --gradient-accumulation-steps "$BEST_GRAD_ACCUM" \
    --learning-rate "$BEST_LR" \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --eval-every 25 \
    --save-every 100 \
    --policy-device cuda:0 \
    --vllm-device cuda:1 \
    --gradient-checkpointing \
    --wandb-mode online \
    --eval-before-train

  echo "Finished SIZE=$SIZE"
done

echo "All runs completed."
