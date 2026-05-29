# An Empirical Autopsy of Small-Model GRPO

Reproducible SFT + GRPO training for small math reasoners (Qwen2.5-Math-1.5B
family) on the Intellect-Math (SFT) and Countdown (GRPO) datasets, plus a
Phase-1 instrumentation layer that records longitudinal training signal
(rollouts, rewards, per-token log-probs and entropy, checkpoint snapshots) so
that GRPO dynamics on small models can be dissected after the fact.

The Python entrypoints are set up for `uv run python -m ...` usage from a
fresh checkout. If `data-distrib/` is missing but `data.tgz` is present, the
required dataset files are extracted on first use by [data_bootstrap.py](src/data_bootstrap.py).

Both training loops are backend-agnostic: the default CUDA path uses vLLM for
fast rollouts, while a `--backend mps`/`--backend cpu` path swaps in a
HuggingFace `generate()` rollout engine so the *same* loop (including the
autopsy recorder) runs on an Apple-silicon MacBook with no CUDA or vLLM. See
[Running locally (Apple Silicon / CPU)](#running-locally-apple-silicon--cpu).

---

## Research plan

The motivating question is *why* GRPO behaves the way it does on small models
during the early phase of training, separate from end-task accuracy. The
hypotheses being probed (see [GRPO_AUTOPSY_HYPOTHESES.md](docs/GRPO_AUTOPSY_HYPOTHESES.md)):

1. **Length growth is partly reward hacking** rather than improved reasoning.
2. **Reasoning behaviors emerge in phases** (backtracking, verification, case
   analysis) instead of growing smoothly.
3. **Entropy collapse is asymmetric** across token positions, token types
   (numbers/operators/structure/text), and correct vs. incorrect rollouts.
4. **Standard-deviation normalization effects are stronger at small scale**
   than papers using larger models report.

The corresponding falsifiable measurements (length conditional on reward,
regex behavior detectors, per-bucket entropy slopes, etc.) are all enabled by
the autopsy instrumentation described below. A **fixed held-out probe set**
(stratified by Countdown difficulty) is generated once per run and re-used at
every recording step so that trajectories are directly comparable across
checkpoints and seeds.

---

## What is implemented

### Training

- [src/train/sft.py](src/train/sft.py) — tokenization with a response mask,
  `get_response_log_probs`, and a single SFT microbatch step.
- [src/train/run_sft.py](src/train/run_sft.py) — full SFT loop on
  Intellect-Math with periodic MATH/Intellect dev evaluation via vLLM,
  best-checkpoint selection, train/eval JSONL history, and W&B logging.
- [src/train/tune_sft.py](src/train/tune_sft.py) — small grid search over
  `(lr, microbatch, grad_accum)` triples; emits `best_config.json` consumed by
  [scripts/train_all.sh](scripts/train_all.sh).
- [src/train/grpo.py](src/train/grpo.py) — group-normalized advantage
  computation and three loss variants:
  - `no_baseline` (REINFORCE on raw rewards)
  - `reinforce_with_baseline` (group-mean baseline, optional std-norm)
  - `grpo_clip` (PPO-style ratio clipping using saved `old_log_probs`)
- [src/train/run_grpo.py](src/train/run_grpo.py) — full GRPO loop on
  Countdown: rollouts → group-normalized advantages → microbatched
  policy update → periodic dev eval and rollout-example dumping; supports
  gradient checkpointing, configurable length normalization (`masked_mean` /
  `masked_normalize`), the autopsy recorder, and pluggable rollout backends
  (`--backend cuda|mps|cpu`).
- [src/train/masking.py](src/train/masking.py) — `masked_mean` and
  `masked_normalize` helpers shared by SFT and GRPO.
- Both `run_sft.py` and `run_grpo.py` expose `--backend {cuda,mps,cpu}`,
  `--dtype {bfloat16,float16,float32}`, and `--gen-batch-size`. On CUDA they
  use vLLM and sync policy weights into it each step; on MPS/CPU they use the
  HF generator, which holds a live reference to the policy so weight "syncing"
  is a no-op.

### Autopsy instrumentation (Phase 1, implemented)

- [src/autopsy/probe_set.py](src/autopsy/probe_set.py) — stratified
  (easy/medium/hard) fixed Countdown probe set with deterministic seed.
- [src/autopsy/rollout_recorder.py](src/autopsy/rollout_recorder.py) — per
  step:
  - generates `group_size` rollouts per fixed probe prompt via vLLM,
  - rescores them with the live policy to capture per-token log-probs and
    entropy,
  - writes per-probe JSON (`autopsy/rollouts/step_XXXX/probe_NNN.json`) and a
    consolidated tensor bundle (`autopsy/tensors/step_XXXX/rollout_tensors.pt`),
  - supports automatic batch-size backoff on OOM,
  - returns scalar `autopsy/*` metrics that are also logged to W&B.
- Run-level autopsy artifacts include `probe_set.json`, `autopsy_history.jsonl`
  (scalar log), and periodic checkpoint snapshots in
  `checkpoints/autopsy_step_XXXX/`.

### Reward functions

- [src/grading/grader_math.py](src/grading/grader_math.py) — boxed-answer
  + LaTeX/SymPy verifier used for SFT MATH evaluation (vendored from
  understand-r1-zero, ~1k lines).
- [src/grading/grader_countdown.py](src/grading/grader_countdown.py) —
  parses `<answer>...</answer>` blocks, supports both single-equation and
  step-by-step formats, uses an AST + `Fraction` evaluator to verify that the
  expression equals the target and only consumes available numbers.

### Inference / evaluation

- [src/eval/evaluate_math.py](src/eval/evaluate_math.py) — quick
  Intellect-test + MATH-test accuracy probe via vLLM.
- [src/infer/infer_batch.py](src/infer/infer_batch.py) — batched MATH
  inference with fast/slow grader comparison and category breakdown
  (`correct_format_and_answer`, `formatted_but_wrong`,
  `unformatted_or_unparseable`, suspected parser issues).
- [src/infer/hf_generate.py](src/infer/hf_generate.py) — local (non-CUDA)
  rollout backend: a HuggingFace `generate()` wrapper that returns
  vLLM-`RequestOutput`-shaped objects, so the training loops and autopsy
  recorder use it as a drop-in replacement for vLLM on MPS/CPU. Handles
  left-padded batching, stop-string truncation, and greedy/sampling modes.

### Orchestration scripts

- [scripts/train_all.sh](scripts/train_all.sh) — runs `tune_sft` then SFT at
  sizes `{128, 256, 512, 1024, full}` using the tuned hyperparameters.
- [scripts/benchmark_all.sh](scripts/benchmark_all.sh) — on-policy GRPO
  reference run on Countdown.
- [scripts/run_grpo_offpolicy.sh](scripts/run_grpo_offpolicy.sh) — `grpo_clip`
  off-policy contrast run.
- GRPO ablation sweeps:
  - [scripts/run_grpo_baseline_sweep.sh](scripts/run_grpo_baseline_sweep.sh)
    — `no_baseline` vs. `reinforce_with_baseline`.
  - [scripts/run_grpo_lengthnorm_sweep.sh](scripts/run_grpo_lengthnorm_sweep.sh)
    — `masked_mean` vs. `masked_normalize`.
  - [scripts/run_grpo_stdnorm_sweep.sh](scripts/run_grpo_stdnorm_sweep.sh) —
    std-normalization on/off.
  - [scripts/run_grpo_lr_sweep.sh](scripts/run_grpo_lr_sweep.sh) — learning
    rate scan.
- [scripts/run_autopsy_suite.sh](scripts/run_autopsy_suite.sh) — one-command
  pilot + full + off-policy autopsy runs across seeds (CUDA).
- [scripts/run_grpo_local.sh](scripts/run_grpo_local.sh) — Apple-silicon/CPU
  launcher for GRPO + autopsy (no CUDA/vLLM/W&B); supports `SMOKE=1` for a fast
  end-to-end sanity run. See [Running locally](#running-locally-apple-silicon--cpu).
- [scripts/summarize_sft_results.py](scripts/summarize_sft_results.py) and
  [scripts/summarize_grpo_experiments.py](scripts/summarize_grpo_experiments.py)
  — aggregate JSONL histories into per-run summaries.
- GPU-utilization helpers ([gpu_keepalive.sh](scripts/gpu_keepalive.sh),
  [gpu_keepalive_adaptive.py](scripts/gpu_keepalive_adaptive.py),
  [utilizer_cuda0.py](scripts/utilizer_cuda0.py),
  [utilizer_cuda1.py](scripts/utilizer_cuda1.py)) to keep cloud instances
  from being reclaimed during long sweeps.

### Data & configuration

- [src/data_bootstrap.py](src/data_bootstrap.py) — lazy extraction of
  `data.tgz → data-distrib/` with a path-traversal-safe extractor.
- [configs/prompts/countdown.prompt](configs/prompts/countdown.prompt) and
  [configs/prompts/intellect.prompt](configs/prompts/intellect.prompt) —
  prompt templates for Countdown and Intellect/MATH respectively.

---

## What is *not* yet implemented

These are placeholders left for follow-on phases of the autopsy work.

- **Config YAMLs** ([configs/base.yaml](configs/base.yaml),
  [configs/sft.yaml](configs/sft.yaml), [configs/grpo.yaml](configs/grpo.yaml),
  [configs/eval.yaml](configs/eval.yaml)) — currently empty (0 lines);
  everything is configured via CLI flags today.
- **Single-prompt inference** ([src/infer/infer_single.py](src/infer/infer_single.py))
  — empty file.
- **Eval benchmarks aggregator** ([src/eval/benchmarks.py](src/eval/benchmarks.py))
  — empty file.
- **`scripts/eval_all.sh`** — empty file (no aggregated post-training eval
  pipeline yet).
- **Phase-2 analysis pipeline** — the recorder produces raw artifacts, but
  the analyses that consume them are not in the repo yet:
  - length-vs-reward / length-vs-step plots
  - regex-based reasoning-behavior detectors (backtracking, verification,
    case analysis) and the LLM-judge precision/recall layer
  - per-bucket entropy-slope analysis (token position, token type,
    correctness)
  - cross-seed and cross-loss-type aggregation reports
- **Off-policy importance-ratio analysis** beyond raw clipfrac logging.
- **CLI/Notebook viewers** for the saved `rollout_tensors.pt` bundles.

---

## Core entrypoints

| Command | Purpose |
| --- | --- |
| `uv run python -m src.train.run_sft` | SFT on Intellect + MATH eval |
| `uv run python -m src.train.tune_sft` | Hyperparameter search for SFT |
| `uv run python -m src.train.run_grpo` | GRPO training on Countdown + autopsy |
| `uv run python -m src.infer.infer_batch` | Batched MATH inference w/ category report |
| `uv run python -m src.eval.evaluate_math` | Quick MATH/Intellect accuracy probe |

## Phase-1 autopsy: recommended pilot command

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

- `--enable-autopsy-recorder` — turn instrumentation on
- `--autopsy-every` — record every N rollout steps
- `--autopsy-num-probe-prompts` — fixed probe-set size
- `--autopsy-probe-split` — source split (`train` / `dev` / `test`)
- `--autopsy-probe-seed` — deterministic probe-set seed
- `--autopsy-group-size` — rollouts per probe prompt (defaults to `--group-size`)
- `--autopsy-checkpoint-every` — save model snapshots every N rollout steps
- `--autopsy-logprob-batch-size` — microbatch size for scoring; auto-backoff on OOM

### Output layout per run

```
<output-dir>/
  config.json
  train_history.jsonl
  rollout_history.jsonl
  rollout_examples.jsonl
  eval_history.jsonl
  summary.json
  best_metric.json
  checkpoints/
    best/ last/
    autopsy_step_XXXX/        # periodic full HF checkpoints
  autopsy/
    probe_set.json            # fixed probe manifest (with difficulty buckets)
    autopsy_history.jsonl     # scalar log
    rollouts/step_XXXX/probe_NNN.json
    tensors/step_XXXX/rollout_tensors.pt
```

## One-command autopsy suite

```bash
bash scripts/run_autopsy_suite.sh
```

Launches:

1. pilot runs (default seeds: `42 43 44`, 20 steps)
2. full runs (default seeds: `42 43 44`, 200 steps)
3. one off-policy contrast run (default seed: first seed)

Useful overrides: `AUTOPSY_OUT_ROOT`, `AUTOPSY_SEEDS`, `WANDB_MODE`,
`CUDA_DEVICES`, `RUN_PILOT`, `RUN_FULL`, `RUN_OFFPOLICY`.

## Running locally (Apple Silicon / CPU)

The CUDA rollout path depends on vLLM, which is CUDA-only. To run the *same*
GRPO + autopsy loop on a MacBook (M-series MPS) or CPU, use the HF generate
backend via [scripts/run_grpo_local.sh](scripts/run_grpo_local.sh):

```bash
# Fast end-to-end sanity check (2 rollout steps, tiny probe set, short outputs)
SMOKE=1 bash scripts/run_grpo_local.sh

# Full local run (override anything via env vars)
NUM_ROLLOUT_STEPS=50 bash scripts/run_grpo_local.sh
```

Defaults are tuned for an M3 Pro: `Qwen/Qwen2.5-0.5B-Instruct`, `--backend mps`,
`--dtype float32`, small rollout/group/batch sizes, `--max-new-tokens 256`,
autopsy enabled, and `--wandb-mode disabled`. The script sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` so any MPS-unsupported op falls back to CPU
rather than crashing. Output lands under `runs/local/<run-name>/` with the same
layout as a CUDA run.

To wire a custom command by hand, the relevant flags are:

- `--backend mps` (or `cpu`) — selects the HF generate rollout engine
- `--dtype float32` — safest for training on MPS/CPU (bf16 is the CUDA default)
- `--gen-batch-size` — micro-batch size for HF generation

> **Note on `Qwen2.5-Math`:** there is no public `Qwen2.5-Math-0.5B`, so the
> small-scale local target is the general-purpose `Qwen/Qwen2.5-0.5B-Instruct`.
> Mac dependencies (no vllm/wandb/accelerate) live in
> [pyproject-mac.toml](pyproject-mac.toml); only `numpy`, `torch`,
> `transformers`, and `datasets` are required for the GRPO/Countdown path.

## Weights & Biases tracking

Training scripts default to:

- `entity=sm12377-new-york-university`
- `project=smallModelGrpo`

and upload the full output directory as a run artifact by default:

- `--no-wandb-log-output-artifact` to skip artifact upload
- `--wandb-mode disabled` to turn W&B off entirely
