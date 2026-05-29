## MacBook
cd ~/workspace/grpoExperiments/mathSFTandGRPO

### 0. (optional) fast sanity check — ~15 min incl. model download
SMOKE=1 bash scripts/run_grpo_local.sh

### 1. Main autopsy run: reinforce_with_baseline, 60 steps  (~1–1.5 h)
NUM_ROLLOUT_STEPS=60 ROLLOUT_BATCH_SIZE=8 GROUP_SIZE=4 TRAIN_BATCH_SIZE=8 \
AUTOPSY_NUM_PROBE_PROMPTS=8 AUTOPSY_EVERY=5 EVAL_EVERY=10 \
SEED=42 RUN_NAME=local_reinforce_s42 \
bash scripts/run_grpo_local.sh --loss-type reinforce_with_baseline

### 2. (optional) loss-type contrast: grpo_clip, same budget  (~1–1.5 h)
NUM_ROLLOUT_STEPS=60 ROLLOUT_BATCH_SIZE=8 GROUP_SIZE=4 TRAIN_BATCH_SIZE=8 \
AUTOPSY_NUM_PROBE_PROMPTS=8 AUTOPSY_EVERY=5 EVAL_EVERY=10 \
SEED=42 RUN_NAME=local_grpoclip_s42 \
bash scripts/run_grpo_local.sh --loss-type grpo_clip --learning-rate 1e-6

### 3. (optional) second seed for the main run
SEED=43 RUN_NAME=local_reinforce_s43 NUM_ROLLOUT_STEPS=60 ROLLOUT_BATCH_SIZE=8 \
GROUP_SIZE=4 TRAIN_BATCH_SIZE=8 AUTOPSY_NUM_PROBE_PROMPTS=8 AUTOPSY_EVERY=5 EVAL_EVERY=10 \
bash scripts/run_grpo_local.sh --loss-type reinforce_with_baseline



## On Cluster

git clone <your-repo> && cd mathSFTandGRPO
uv sync                       # uses pyproject.toml (CUDA + vLLM) — NOT pyproject-mac.toml

### Full original pipeline:
bash scripts/train_all.sh             # SFT sweep on Intellect-Math (Qwen2.5-Math-1.5B)
bash scripts/run_autopsy_suite.sh     # GRPO autopsy: pilot + full + off-policy, seeds 42 43 44
bash scripts/run_grpo_baseline_sweep.sh
bash scripts/run_grpo_lengthnorm_sweep.sh
bash scripts/run_grpo_stdnorm_sweep.sh
bash scripts/run_grpo_lr_sweep.sh

### 2 A100s caveats
CUDA_DEVICES=0 bash scripts/run_autopsy_suite.sh   # then it still defaults vllm-device cuda:1 → override:
### simplest: call run_grpo directly with both on one GPU
python -m src.train.run_grpo --vllm-device cuda:0 --vllm-gpu-memory-utilization 0.6 \
  --enable-autopsy-recorder --num-rollout-steps 200 --output-dir runs/autopsy_v1/full_seed42 ...
