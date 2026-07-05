# Project Overview: An Empirical Autopsy of Small-Model GRPO

End-to-end walkthrough of this repository: purpose, research plan, architecture,
algorithms, and the engineering tradeoffs behind each design decision.
For mock-interview Q&A with defenses of every decision below, see
[INTERVIEW_PREP.md](INTERVIEW_PREP.md).

---

## 1. Purpose

A reproducible SFT + RL (GRPO) training stack for small math-reasoning LLMs,
instrumented to answer **why** GRPO behaves the way it does on small models —
not just whether accuracy goes up. The core contribution is an "autopsy" layer
that records longitudinal, per-token training signal (log-probs, entropy,
rollouts, rewards) on a **fixed probe set** at every stage of training, so
training dynamics like entropy collapse and length inflation can be dissected
after the fact.

The project began as an NYU "Building LLM Reasoners" alignment assignment
(SFT + GRPO correctness) and was extended into a research project: the autopsy
instrumentation, ablation sweeps, analysis pipeline, and multi-backend
engineering are the original work.

## 2. Research question and hypotheses

Standard RLVR (RL with verifiable rewards) papers report *outcomes* — accuracy,
response length. This project asks about *mechanisms* during early training on
small (0.5B–1.5B) models. Four falsifiable hypotheses
(see [GRPO_AUTOPSY_HYPOTHESES.md](GRPO_AUTOPSY_HYPOTHESES.md)):

1. **Length growth is partly reward hacking** — responses get longer without
   getting more correct. Test: track `length | reward=1` vs `length | reward=0`
   over steps; if lengths grow in the *incorrect* population too, growth is not
   explained by better reasoning.
2. **Reasoning behaviors emerge in phases** — backtracking, verification,
   case-analysis show step-function emergence, not smooth growth. Test: regex
   detectors per behavior family vs. rollout step (LLM-judge validation layer
   planned for precision/recall).
3. **Entropy collapse is asymmetric** — policy entropy does not fall uniformly;
   it falls differently by token *position* (early/mid/late), token *type*
   (numbers/operators/structure/text), and rollout *correctness*. Test:
   entropy-vs-step slopes per bucket.
4. **Std-normalization effects are amplified at small scale** — GRPO's
   divide-by-group-std is controversial (Dr. GRPO argued it biases toward
   low-variance = easy/hard-saturated prompts); with group_size 4–8 on small
   models the std estimate is noisy, so the effect should be larger than
   large-model papers report. Test: std-norm on/off sweep with everything else
   fixed.

**Methodological key — the fixed probe set.** You cannot compare entropy at
step 5 vs step 50 if the prompts differ. A difficulty-stratified
(~34/33/33 easy/medium/hard, bucketed by numbers-count and target magnitude)
sample of held-out Countdown problems is drawn once per run with a fixed seed,
saved as `probe_set.json`, and re-rolled at every recording step. Same prompts
across checkpoints *and* across seeds → trajectories are directly comparable.

## 3. Tasks and training stages

| Stage | Task | Data | Model | Why |
|---|---|---|---|---|
| SFT | MATH-style problems | Intellect-Math (chat traces w/ ground truth) | Qwen2.5-Math-1.5B | Behavior-cloning baseline; data-scaling study at n ∈ {128, 256, 512, 1024, full} |
| GRPO | **Countdown** | 10k train / 1k dev / 1k test parquet | Qwen2.5-Math-1.5B-Instruct (A100) / Qwen2.5-0.5B-Instruct (local) | RL from a verifiable reward |

**Why Countdown for the RL stage:** given numbers `[3,7,11,2]` reach target
`30` using `+−×÷`, each number once. It is *programmatically verifiable*
(no LLM judge, no reward model → no reward-model-hacking confound),
*difficulty-controllable* (numbers count × target size), and *hard for small
models*, so there is real learning signal. Reward decomposes into
`format_reward` (emitted an `<answer>` block) and `answer_reward` (math is
right); the scalar training reward equals `answer_reward`. Logging both
separately exposes dynamics like "model learned the format long before the
math."

## 4. System architecture

```
                        ┌────────────────────────────────────────────┐
                        │            run_grpo.py (driver)            │
                        └────────────────────────────────────────────┘
   per rollout step:
   1. sample N/G unique prompts ──► 2. GENERATOR: G rollouts each
                                       ├─ cuda: vLLM engine (GPU 1)
                                       └─ mps/cpu: HFGenerator wraps live policy
   3. countdown_reward_fn per rollout
   4. group-normalized advantages  (view(-1, G), subtract group mean, ÷ std optional)
   5. (grpo_clip only) score old_log_probs with policy BEFORE update
   6. train: shuffle → train batches → microbatches → loss.backward() per micro
      → clip grad norm → optimizer.step()
   7. every autopsy_every steps: RolloutRecorder on the FIXED probe set
   8. every eval_every steps: greedy dev eval → best-checkpoint selection
   9. cuda only: load_policy_into_vllm_instance(policy, llm)  (weight re-sync)
```

**The dual-model design.** On CUDA there are *two copies* of the model: a
HuggingFace policy (trainable, `cuda:0`) and a vLLM inference engine (fast
generation, `cuda:1`). After each update the HF `state_dict` is pushed into
vLLM's model runner. Why: HF `generate()` is far too slow for RL rollout
volume; vLLM (paged attention, continuous batching, prefix caching — rollout
groups share a prompt prefix, so prefix caching is a big win) is ~10–20×
faster but cannot train. Cost: 2× weight memory, a sync step, and two GPUs
(or careful memory splitting on one).

**Token-level plumbing** (`tokenize_prompt_and_output` in
[src/train/sft.py](../src/train/sft.py)): concatenate prompt+response ids,
build a boolean `response_mask` that is True only on response tokens, then
shift — `input_ids = full[:, :-1]`, `labels = full[:, 1:]`, mask aligned to
labels. Every loss in the repo (SFT and GRPO) is computed per-token then
reduced under this mask, so prompt tokens and padding never receive gradient.
Log-probs are `log_softmax(logits).gather(-1, labels)`; entropy is exact over
the full vocab (`−Σ p log p`), not a sampled estimate. Prompt and response
are tokenized *separately* then concatenated — this can differ from
tokenizing the joined string at the boundary token, but it is consistent
between training and scoring, which is what correctness requires.

**Stop-sequence handling.** Generation stops on `</answer>` (the
`--stop-sequence` flag). Engines exclude the matched stop string from output
by default, but the grader's regex needs the closing tag present, so
`get_output_text` re-appends it when `finish_reason == "stop"` and the stop
reason matches. Both the vLLM path and `HFGenerator` expose the same
`finish_reason`/`stop_reason` fields so this logic is backend-agnostic.

**Run outputs.** Every run writes a self-describing directory: `config.json`
(the full flag snapshot — the reproducibility record), append-only JSONL
histories (`train_history`, `rollout_history`, `eval_history`,
`rollout_examples` with a few sampled generations per step), `summary.json`
(best/final metrics), `best_metric.json`, and `checkpoints/{best,last}` plus
optional periodic `autopsy_step_XXXX` snapshots. Best-checkpoint selection is
on dev accuracy; the final test evaluation reloads `best`, not `last`.
Training-signal metrics logged per optimizer step include loss, grad-norm,
clipfrac (grpo_clip), advantage mean/std, and — when the entropy bonus is
active — mean token entropy.

## 5. The GRPO math

All in [src/train/grpo.py](../src/train/grpo.py).

**Advantages** — `compute_group_normalized_rewards`: rewards reshaped to
`(num_prompts, group_size)`; `A = r − mean_group`, optionally
`A = (r − mean) / (std + 1e-6)`. The *group mean is a critic-free baseline* —
no value network (PPO needs one); the price is G× more generations per prompt.

**Three loss variants** (the main ablation axis):

1. `no_baseline` — REINFORCE on raw rewards: `L = −r · logπ(t)` per token.
   High-variance control condition.
2. `reinforce_with_baseline` — `L = −A · logπ(t)`. Vanilla GRPO for the
   strictly on-policy case (one gradient epoch per rollout batch).
3. `grpo_clip` — PPO-style: `ratio = exp(logπ − logπ_old)`, objective
   `min(ratio·A, clip(ratio, 1±ε)·A)`, loss is its negative; logs `clipfrac`.
   Needed when taking *multiple* gradient epochs per rollout batch
   (`--epochs-per-rollout-batch > 1`), because after the first step the data
   is off-policy. `old_log_probs` are scored with the policy in inference mode
   *before* updating.

**Length normalization** (second ablation axis): after per-token loss, reduce
over the sequence either by

- `masked_mean` — divide by *that response's* token count → every response
  contributes equally regardless of length, but the per-token gradient on long
  responses is smaller (1/len scaling), or
- `masked_normalize` — divide by a *constant* (max response length in batch) →
  per-token gradient is length-independent, so long responses contribute
  proportionally more total gradient.

This choice interacts directly with hypothesis 1: `masked_mean` can *mute*
the incentive against long wrong answers (the Dr. GRPO argument); here it is
an isolated switch.

**Entropy bonus** (third ablation axis): optional `--entropy-bonus-coef β`
subtracts `β · H(π_t)` from the per-token loss (i.e., maximizes entropy),
using the *differentiable* full-vocab entropy from the same forward pass that
produces the policy log-probs. This is the standard exploration bonus from
A3C/PPO, applied here as a direct intervention on hypothesis 3: if entropy
collapse is a driver of premature convergence, a small β should delay collapse
in exactly the token buckets where it is steepest, measurable via the autopsy
entropy-slope analysis. β=0 (default) is a no-op and preserves prior behavior.

**Gradient accumulation detail**: each microbatch calls `loss.backward()`
itself (loss pre-divided by accumulation steps) inside the step function; the
driver clips grad-norm and calls `optimizer.step()` once per train batch.
Microbatch size 1 + gradient checkpointing was the memory posture on A100.
One correctness edge handled in SFT: the final partial accumulation window
rescales by the *actual* number of remaining microbatches, so the last
optimizer step of an epoch is not systematically down-weighted.

## 5b. The SFT stage

[src/train/run_sft.py](../src/train/run_sft.py) +
[src/train/sft.py](../src/train/sft.py) +
[src/train/tune_sft.py](../src/train/tune_sft.py).

- **Loss**: `sft_microbatch_train_step` — negative mean of per-example
  response log-prob sums, computed under the response mask via
  `masked_normalize` (sum divided by a `normalize_constant`, default 1.0),
  divided by the accumulation count before `backward()`. Standard
  teacher-forced NLL restricted to response tokens.
- **Data**: Intellect-Math chat traces (`messages` with system/user/assistant
  roles) flattened into `prompt` (system + user) and `output` (assistant),
  with `ground_truth` carried for eval. Data-scaling runs subsample train to
  n ∈ {128, 256, 512, 1024, full}.
- **Evaluation**: two eval families each cycle — held-out Intellect dev, and
  MATH (hiyouga/math12k) val — graded by the vendored boxed-answer/SymPy
  verifier, with category accounting (`formatted_but_wrong`, `unformatted`)
  to separate parsing failures from reasoning failures. Best checkpoint is
  selected on `--selection-metric` (default `math_val_accuracy`).
- **Hyperparameter search**: `tune_sft.py` grid over
  (lr, microbatch, grad-accum) triples emits `best_config.json`, consumed by
  [scripts/train_all.sh](../scripts/train_all.sh) for the scaling sweep.
- The same backend flags (`--backend/--dtype/--gen-batch-size`) were threaded
  through SFT as GRPO, so the SFT loop is also runnable locally — though the
  full SFT recipe (seq len up to 2048, 300 optimizer steps × 16 grad-accum,
  500-example evals) is ~12+ h on the M3 Pro and was deliberately skipped
  locally: `Qwen2.5-0.5B-Instruct` is already instruction-tuned and the
  autopsy research questions live in the GRPO stage.

## 6. Autopsy recorder (the core instrumentation)

[src/autopsy/rollout_recorder.py](../src/autopsy/rollout_recorder.py). At
every `autopsy_every` rollout steps:

1. Generate `group_size` rollouts per probe prompt with the *live* policy at
   training temperature (0.7 — the sampling distribution training actually
   sees, not greedy).
2. Grade each rollout.
3. **Re-score every rollout with the policy in a teacher-forced forward pass**
   to extract per-token log-probs *and* per-token full-vocab entropy. vLLM
   gives text fast but not clean per-token entropy under the training
   tokenization — hence the two-pass design: generate fast, score exactly.
4. Write two artifact forms:
   - human-inspectable per-probe JSON — prompt, ground truth, difficulty meta,
     and per rollout: response text, reward dict, tokens, token ids, per-token
     log-probs, per-token entropies;
   - a consolidated `rollout_tensors.pt` per step — `(num_rollouts × seq_len)`
     tensors for log-probs, entropy, mask, ids — for vectorized analysis.
5. **OOM backoff**: the scoring loop halves its batch size on CUDA OOM down to
   1 before failing with an actionable error — autopsy shares memory with
   training and should degrade rather than kill an 8-hour run.
6. Periodic full HF checkpoint snapshots (`checkpoints/autopsy_step_XXXX/`) so
   analyses not anticipated up front can be recomputed later.

**Probe-set construction**
([src/autopsy/probe_set.py](../src/autopsy/probe_set.py)): difficulty rule —
`easy` = ≤3 numbers and target ≤50, `hard` = ≥4 numbers and target ≥250,
`medium` = everything else; targets ~34/33/33 with shortfall backfilled from
the remaining pool; a dedicated probe RNG seed (default 123, independent of
the training seed) makes the same probe set reproducible across runs and
arms. Each `ProbeExample` is a frozen dataclass carrying probe_id, dataset
index, formatted prompt, ground truth, and difficulty metadata — all
serialized into the manifest.

Design principle: **record raw signal, defer analysis.** The recorder saves
everything needed to test all four hypotheses; the consumers live in the
separate Phase-2 analysis module (below) and can always be re-run over saved
artifacts.

## 7. Phase-2 analysis module

[src/analysis/](../src/analysis/) consumes a run directory's autopsy
artifacts and produces per-run JSON metrics + a markdown report. One command:

```bash
python -m src.analysis.run_analysis --run-dir runs/local/<run-name>
```

Components:

- **Loader** ([loading.py](../src/analysis/loading.py)) — reads
  `autopsy/probe_set.json` and every `rollouts/step_XXXX/probe_NNN.json` into
  flat per-rollout records `(step, probe_id, difficulty, reward, response,
  tokens, log_probs, entropies)`.
- **Length vs reward** ([length_vs_reward.py](../src/analysis/length_vs_reward.py))
  — hypothesis 1: per-step response-length distributions (mean/std/quantiles),
  conditioned on `reward=1` vs `reward=0`, plus per-difficulty breakdowns and
  the length-vs-step slope in each reward population.
- **Behavior detectors** ([behaviors.py](../src/analysis/behaviors.py)) —
  hypothesis 2: regex families for *backtracking* ("wait", "doesn't work",
  "let me try again", …), *verification* ("let me check", "verify", …), and
  *case analysis* ("case 1", "alternatively", "another approach", …), counted
  per rollout and aggregated to per-step frequencies.
- **Entropy slopes** ([entropy.py](../src/analysis/entropy.py)) — hypothesis 3:
  mean token entropy per step, bucketed three ways — position within the
  response (early/mid/late thirds), token type (number / operator / structure /
  text, classified from the token string), and rollout correctness — plus a
  least-squares slope of entropy vs step per bucket, so "collapse asymmetry"
  is a direct slope comparison.
- **CLI + report** ([run_analysis.py](../src/analysis/run_analysis.py)) —
  writes `analysis/{length_vs_reward,behaviors,entropy}.json` and a combined
  `analysis/report.md` into the run dir; `compare_runs.py` renders a
  side-by-side table for A/B ablations (e.g., std-norm on vs off).

Implementation notes worth knowing cold:

- **Statistics** ([stats.py](../src/analysis/stats.py)) are dependency-free:
  interpolated quantiles and ordinary least-squares slopes (returning
  `slope=None` for n<2 or zero x-variance rather than crashing on degenerate
  windows).
- **Token-type classification** handles Qwen's GPT-2-style byte-BPE surface
  forms: strip the `Ġ` (space) and `Ċ` (newline) markers, then classify —
  all-digits → number, all chars in `+-*/×÷=()` → operator, answer-scaffold
  tokens (`<answer>`, `step`, `:`, whitespace) → structure, else text. It is
  a heuristic validated by inspection, not a ground-truth labeling — stated
  as such wherever results depend on it.
- **Fair A/B windows**: `--max-step` trims runs to a common step range (used
  when one arm died later than another); in trim mode `compare_runs` computes
  in memory rather than clobbering the persisted full analysis.
- Behavior detectors are surface regexes; the planned LLM-judge layer for
  precision/recall calibration is future work, so behavior *trends* are more
  trustworthy than absolute frequencies.

The module is pure-Python over the JSON artifacts (torch only for optional
tensor loading), so it runs anywhere — including on artifacts regenerated
locally on a laptop.

## 8. The Countdown grader

[src/grading/grader_countdown.py](../src/grading/grader_countdown.py) — a safe
expression verifier, not `eval()`:

- Extracts `<answer>…</answer>`; accepts **two formats** — a single expression
  `(1+2)/3` or multi-line `Step X: a + b = c` chains.
- Parses with Python's `ast` in eval mode, allowing **only** numeric constants,
  unary ±, and `+−*/` — arbitrary code cannot execute (a real concern when the
  text being evaluated comes from a policy that is being *optimized* against
  the reward).
- All arithmetic in `Fraction` — exact rational math (`1/3 * 3 == 1`; floats
  would leak wrong verdicts).
- **Number-consumption checking**: operands are multiset-matched against the
  available numbers and removed as used; in step format each step's result is
  *added back* to the pool (Countdown rules). A wrong intermediate equation
  (`lhs != rhs`) fails the whole answer.
- Robust to `×`, `÷`, `3 x 4`, code fences, and step-prefix noise.

The SFT-side MATH grader ([grader_math.py](../src/grading/grader_math.py)) is
a vendored ~1k-line boxed-answer + SymPy/latex2sympy verifier (from
understand-r1-zero) — reused, not written here; the Countdown grader is
original.

## 9. Backends and the hardware story

**Phase A (A100s, original).** Policy on `cuda:0` bf16, vLLM on `cuda:1`, W&B
online logging with full output-dir artifacts, GPU keep-alive scripts for
cloud instances that reclaim idle GPUs.

**Phase B (M3 Pro MacBook).** A100 access (and the original run artifacts)
were lost. vLLM is CUDA-only, so the rollout engine was abstracted:

- [src/infer/hf_generate.py](../src/infer/hf_generate.py) — `HFGenerator`
  wraps HF `generate()` but returns **vLLM-`RequestOutput`-shaped objects**
  (`out.outputs[0].text/.finish_reason/.stop_reason`), so the training loop,
  eval, and recorder needed *zero* call-site changes. Handles stop-string
  truncation and greedy/sampling modes.
- **It never pads.** Prompts are grouped by exact token length and only
  equal-length prompts are batched together (original order restored on
  return). This exists because padded mixed-length batches produce NaN
  logits under SDPA on MPS — see the post-mortem in §10 — and it doubled as
  a ~2.5× speedup, since pad tokens were pure compute overhead. Rollout
  groups repeat one prompt, so they always batch at full width; mixed-length
  eval batches degrade to per-length chunks.
- `HFGenerator` holds a **live reference to the policy module** — the vLLM
  weight-sync step becomes a no-op locally. One model in memory instead of two.
- `--backend {auto,cuda,mps,cpu}` with `auto` (default) detecting
  cuda > mps > cpu; a fail-fast guard raises a clear error if cuda is selected
  without vLLM installed. dtype defaults: bf16 on cuda, **fp32 on MPS/CPU**
  (bf16 training on MPS is numerically risky / poorly supported).
- **MPS memory discipline**: `free_accelerator_memory` runs `gc.collect()`
  before `torch.mps.empty_cache()` (autograd-graph objects pin Metal buffers
  until Python GC runs) and is called at the generation↔training phase
  boundaries of every rollout step; the local launcher lifts the MPS
  allocator's high-watermark cap (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`)
  because Metal-side kernel/graph caches ("other allocations") grow across a
  long run and the default cap kills backward passes that unified memory
  could absorb.
- Model downscaled to `Qwen/Qwen2.5-0.5B-Instruct` (no public Math-0.5B
  exists), batch sizes shrunk, eval sets capped (32/64 instead of 256/1024 —
  measured: greedy eval was the dominant local wall-clock cost), W&B off.
- [scripts/run_grpo_local.sh](../scripts/run_grpo_local.sh) is the local
  launcher: M3-sane defaults, every knob an env-var override, a `SMOKE=1`
  mode (2 steps, tiny probes) for end-to-end sanity in ~15 min, and
  `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops fall back to CPU
  instead of crashing.
- **Measured performance** (M3 Pro, 0.5B fp32): ~89 tok/s aggregate
  generation (batch 4, 256 new tokens), ~1.5 s per training microbatch
  (bs=1, seq≈320, grad checkpointing). Real rollout steps: ~3.5 min with the
  original padded generator, **~1.3 min after the no-padding fix** — a
  45-step autopsy run completes in ~35 min.
- **Environment**: the working local env is conda Python 3.13 + torch 2.8
  (MPS) + transformers 4.55 + datasets — only four packages matter for the
  GRPO/Countdown path (sympy/math-verify are SFT-grader-only).
  `pyproject-mac.toml` (no vllm/wandb/accelerate) exists for a
  from-scratch uv setup; the CUDA dependency set lives in `pyproject.toml`.
- **Ops lesson learned the hard way**: fp32 checkpoints of even a 0.5B are
  ~2 GB, and per-step `last` saves plus every-25-step snapshots filled the
  disk mid-run. Local ablations run with snapshots disabled
  (`AUTOPSY_CHECKPOINT_EVERY=0`), and `runs/` is gitignored.

## 9b. Data, tracking, and supporting tooling

- **Data bootstrap** ([src/data_bootstrap.py](../src/data_bootstrap.py)):
  if `data-distrib/` is missing but `data.tgz` is present, required dataset
  files are extracted lazily on first use, through a **path-traversal-safe
  extractor** (member paths are resolved and checked against the target root
  before extraction). A fresh clone with the tarball trains with no manual
  setup step.
- **Prompts** live in `configs/prompts/` — `countdown.prompt` instructs
  step-by-step reasoning and the `<answer>…</answer>` output contract (both
  the step-list and single-equation forms are shown by example);
  `intellect.prompt` is the MATH-style template.
- **Experiment tracking**: W&B (`entity sm12377-new-york-university`,
  project `smallModelGrpo`) with `define_metric` so `train/*` and `eval/*`
  chart against their own step counters, and optional upload of the entire
  output dir as a run artifact. Every W&B record is *also* written to local
  JSONL first — the runs never depend on the tracker being up, which is why
  local runs simply set `--wandb-mode disabled`.
- **Standalone inference/eval tools**:
  [src/infer/infer_batch.py](../src/infer/infer_batch.py) (batched MATH
  inference with fast/slow grader comparison and a category breakdown that
  separates parser failures from reasoning failures) and
  [src/eval/evaluate_math.py](../src/eval/evaluate_math.py) (quick
  Intellect/MATH accuracy probe). Both are vLLM-based utility scripts from
  the A100 phase.
- **Orchestration scripts** (`scripts/`): `train_all.sh` (SFT tune + scaling
  sweep), `benchmark_all.sh` (on-policy GRPO reference),
  `run_grpo_{baseline,lengthnorm,stdnorm,lr}_sweep.sh` (one ablation axis
  each), `run_grpo_offpolicy.sh` (grpo_clip contrast),
  `run_autopsy_suite.sh` (pilot + full + off-policy × seeds, one command),
  `run_grpo_local.sh` (Mac), `summarize_{sft_results,grpo_experiments}.py`
  (JSONL → per-run summaries), and GPU keep-alive utilities for cloud
  instances that reclaim idle GPUs. All accept env-var overrides rather than
  requiring edits.

## 10. Experiments

The original A100 results were lost with the instance; the experimental design
lives in the orchestration scripts and the runs are being regenerated locally:

- **SFT data-scaling** (A100): `tune_sft` grid over (lr, microbatch,
  grad-accum) → best config → SFT at n ∈ {128, 256, 512, 1024, full},
  selecting on MATH-val accuracy.
- **GRPO ablation sweeps** (A100 scripts; std-norm axis re-run locally):
  baseline (`no_baseline` vs `reinforce_with_baseline`), length norm
  (`masked_mean` vs `masked_normalize`), std-norm on/off, LR scan, and an
  off-policy contrast (`grpo_clip`, 4 epochs/rollout-batch, lr 1e-6 — lower LR
  because off-policy reuse amplifies instability).
- **Autopsy suite** (A100): pilots (20 steps) + full runs (200 steps) × seeds
  {42, 43, 44} + off-policy contrast — multi-seed because small-scale RL is
  noisy and single-seed dynamics claims are weak.
- **Local ablations** (M3 Pro, current): std-norm on/off at 45 steps with
  identical seeds/probe sets (hypothesis 4) — results below; entropy-bonus
  ablation available via `--entropy-bonus-coef` as the next arm.

### The v1 ablation post-mortem (a case study in why the autopsy exists)

The first local std-norm A/B produced **bit-identical metrics across both
arms** — flagged immediately by `compare_runs.py`. Diagnosis from the logged
signal, no rerun needed:

1. `rollout/reward_mean == 0.0` on **all 113 rollout steps** of both runs, and
   `train/grad_norm == 0.0` throughout → with an all-zero reward group,
   `A = r − mean(r) = 0` *regardless* of std normalization → zero gradient →
   **the policy never took a single effective optimization step**. Both "runs"
   were the same frozen initial model sampling with the same seed.
2. Why zero reward: mean response length was ~250 of the 256-token cap — the
   0.5B model almost never emitted `</answer>` within budget (format rate
   ~6%), and never a correct answer. A probe measured 512 tokens → format
   19%, answers still 0/16. Classic RLVR cold start: no reward variance, no
   signal.
3. A second bug surfaced during the diagnosis: **padded mixed-length batches
   produce NaN logits under SDPA on the MPS backend**. Training rollouts were
   accidentally immune (a rollout group repeats one prompt → no padding), but
   greedy dev/test evals batch different-length prompts — argmax over NaN does
   not crash, it silently corrupts. Fixed by making `HFGenerator` batch only
   equal-length prompts (rollouts unaffected; eval degrades to per-length
   chunks; padding is never constructed at all).

Fixes shipped: the generator length-grouping above, plus
`--format-reward-weight w` — a shaped **training** reward
`r = answer + w · format` that gives partial credit for a well-formed
`<answer>` block, creating group-level reward variance from step 1 (the
recorder and all eval metrics keep the pure reward). The v2 rerun with
`w = 0.2` showed nonzero reward and advantage variance at step 1.

Interview framing: the fixed-probe autopsy turned a "mysteriously flat"
ablation into a three-line root cause (zero reward → zero advantage → frozen
policy), exposed a silent numerical bug in the eval path, and motivated a
principled reward-shaping fix — exactly the class of failure this
instrumentation was built to catch.

Getting the rerun through took two more attempts, which is itself
instructive. Once real gradients flowed, the training backward OOM'd at step
~28 on a 519 MiB allocation (exactly one fp32 logits buffer: ~890 tokens ×
152k vocab × 4 B). Attempt one — `torch.mps.empty_cache()` at the
generation↔training phase boundaries — moved the failure from step 28 to
step 30: the growing term was "other allocations" (Metal-side kernel/graph
caches plus autograd objects whose Metal buffers are pinned until Python GC
runs), which `empty_cache` cannot touch. Attempt two fixed it:
`gc.collect()` *before* `empty_cache()` (so buffers are actually
reclaimable) plus lifting the MPS allocator's high-watermark cap
(`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`) in the local launcher, and
de-risking 60→45 steps (slope estimation needs ~9 snapshots, not 12). The
no-padding generator also turned out to be ~2.5× faster
(~1.3 min/rollout-step vs ~3.5), since padded tokens were pure overhead —
both 45-step arms completed in ~35 minutes each. A disk-full crash along the
way (2 GB fp32 checkpoints × per-step `last` saves + periodic snapshots on a
laptop) is why local ablations run with snapshots disabled.

### Std-norm ablation results (v4, local M3 Pro)

Setup: `Qwen2.5-0.5B-Instruct`, 45 rollout steps, seed 42, group size 4,
8 rollouts/step, shaped reward `answer + 0.2·format`, identical fixed probe
set; the **only** difference between arms is `--normalize-by-std`. Full
tables: `runs/local/stdnorm_v4_comparison.md` and each run's
`analysis/report.md`.

| metric | std-norm ON | std-norm OFF |
| --- | --- | --- |
| probe format rate (first → last) | 0.59 → **0.00** (collapse) | 0.66 → **1.00** (mastery) |
| final dev format rate | 0.0 | 1.0 |
| train grad-norm (mean / max) | 2.92 / 13.1 | 0.18 / 2.8 |
| probe mean length (first → last) | 90 → 79 tokens (rambling) | 68 → 12 tokens (terse answers) |
| entropy slope, overall | +0.0023 (drifts up) | −0.0064 (healthy collapse) |
| entropy slope, text vs operator tokens | +0.002 / +0.008 | **−0.012 / +0.002** |

Readings (n=1 seed, 0.5B model — treat as a demonstration, not a law):

1. **Hypothesis 4 supported dramatically.** With sparse binary-ish rewards
   and group size 4, a group where one rollout formats has tiny reward std;
   dividing by `(std + ε)` amplifies that advantage into enormous updates
   (16× larger mean grad-norm), and the std-norm arm *destroys* the format
   behavior it had at initialization. The unnormalized arm learns format to
   100% smoothly. This is the Dr. GRPO amplification pathology reproduced at
   laptop scale.
2. **A reward-shaping hack surfaced (H1-adjacent).** The successful arm
   drove response length 68 → 12 tokens: it emits minimal `<answer>` blocks
   that earn the format credit without solving anything — optimizing the
   proxy, not the task. Next intervention: decay the format weight over
   steps, or gate it on nonzero answer reward.
3. **Entropy collapse is asymmetric (H3 signal).** In the learning arm,
   text-token entropy collapses fastest (−0.012/step) while operator-token
   entropy *rises* slightly — the model commits to the scaffold first and
   keeps exploring the arithmetic. The collapsing arm shows the opposite
   (entropy drifting up everywhere) — a degeneration signature, not
   exploration.

## 11. What is deliberately NOT here

- KL penalty to a reference model — matches R1-Zero-style minimal GRPO; noted
  as a known design choice (the entropy bonus was chosen as the exploration
  intervention instead, since it needs no second model in memory — which
  matters on a laptop). It is also a scientific choice: a KL leash would
  mask exactly the instabilities (e.g., the std-norm collapse) this project
  measures.
- **Resume-from-checkpoint** — exact RL resume requires restoring policy,
  optimizer state, data-order RNG, *and* generation RNG to be trajectory-
  faithful; a non-bitwise resume injects an invisible discontinuity into
  every longitudinal autopsy series. At 35–90-minute local runs,
  restart-from-scratch was the cheaper correctness guarantee. First thing to
  build for A100-scale sweeps.
- LLM-judge validation of the behavior regexes (planned; regex counts are
  reported as-is).
- Cross-seed aggregation with variance bands (`compare_runs` compares
  individual runs today).
- Plots — the analysis emits JSON + markdown tables; figures are generated
  ad hoc.
- Config YAMLs, `infer_single.py`, `benchmarks.py`, `eval_all.sh` — empty
  placeholders; everything is CLI-flag configured, and each run's
  `config.json` snapshot is the reproducibility record.

## 12. Rapid-fire Q&A

- **GRPO vs PPO?** GRPO replaces the learned value function/critic with the
  group-mean reward as baseline. Cheaper and simpler; the cost is G rollouts
  per prompt and a noisier baseline at small G.
- **Why is dividing by group std controversial?** It up-weights low-variance
  groups. All-correct or all-wrong groups have zero advantage anyway;
  near-saturated groups get amplified gradients → difficulty bias (Dr. GRPO).
  At group_size 4–8 the std estimate itself is very noisy — hypothesis 4.
- **Why old_log_probs from the policy, and when?** Scored before the update;
  only needed when reusing a rollout batch for >1 gradient epoch, where the
  clip ratio corrects for the policy having moved.
- **Why per-token entropy from a second forward pass instead of from
  generation?** Exact full-vocab entropy under the training tokenization,
  decoupled from the (swappable, possibly vLLM) generation engine.
- **Why an entropy bonus rather than a KL term?** Both regularize against
  premature determinism; KL-to-reference needs a frozen second model in memory
  every training forward (prohibitive locally), while the entropy bonus reuses
  logits already computed. It also targets hypothesis 3 directly.
- **Biggest engineering lesson?** Program to an interface: making the HF
  backend mimic vLLM's output shape meant the entire training/eval/recording
  stack was backend-agnostic for free.
- **A debugging war story?** The v1 std-norm ablation came back bit-identical
  across arms. The instrumentation reduced it to: all-zero rewards → zero
  group-relative advantages → `grad_norm = 0` for 113 steps → two frozen
  models on the same seed. Root cause was RLVR cold start (0.5B never
  finishes `</answer>` in 256 tokens); fix was shaped reward
  (`answer + 0.2·format`). The same investigation uncovered an MPS bug where
  padded batches yield NaN logits — crashing sampled generation but *silently
  corrupting* greedy eval — fixed by equal-length batch grouping.
