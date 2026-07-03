# Project Overview: An Empirical Autopsy of Small-Model GRPO

End-to-end walkthrough of this repository: purpose, research plan, architecture,
algorithms, and the engineering tradeoffs behind each design decision.

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
the full vocab (`−Σ p log p`), not a sampled estimate.

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
  eval, and recorder needed *zero* call-site changes. Handles left-padded
  batching (decoder-only requirement), stop-string truncation, and
  greedy/sampling modes.
- `HFGenerator` holds a **live reference to the policy module** — the vLLM
  weight-sync step becomes a no-op locally. One model in memory instead of two.
- `--backend {auto,cuda,mps,cpu}` with `auto` (default) detecting
  cuda > mps > cpu; a fail-fast guard raises a clear error if cuda is selected
  without vLLM installed. dtype defaults: bf16 on cuda, **fp32 on MPS/CPU**
  (bf16 training on MPS is numerically risky / poorly supported).
- Model downscaled to `Qwen/Qwen2.5-0.5B-Instruct` (no public Math-0.5B
  exists), batch sizes shrunk, eval sets capped (32/64 instead of 256/1024 —
  measured: greedy eval was the dominant local wall-clock cost), W&B off.
- Verified end-to-end: a 2-step smoke run on MPS produced the full artifact
  tree. Measured throughput: ~89 tok/s generation (batch 4, 256 new tokens),
  ~1.5 s per train microbatch → ~1–1.5 min per rollout step, so a 60-step
  autopsy run ≈ 1–1.5 h locally.

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
- **Local ablations** (M3 Pro, current): std-norm on/off at 60 steps with
  identical seeds/probe sets (hypothesis 4), analyzed with the Phase-2 module;
  entropy-bonus ablation available via `--entropy-bonus-coef`.

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

Two more MPS engineering fixes were needed to get the rerun through: the
backward pass OOM'd around step 30 because Metal-side memory ("other
allocations": kernel/graph caches plus GC-delayed autograd objects) grows
across steps — fixed with `gc.collect()` before `torch.mps.empty_cache()` at
the generation↔training phase boundaries, and by lifting the MPS allocator's
high-watermark cap (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`) in the local
launcher. The no-padding generator also turned out to be ~2.5× faster
(~1.3 min/rollout-step vs ~3.5), since padded tokens were pure overhead.

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
  matters on a laptop).
- LLM-judge validation of the behavior regexes (planned; regex counts are
  reported as-is).
- Config YAMLs, `infer_single.py`, `benchmarks.py`, `eval_all.sh` — empty
  placeholders; everything is CLI-flag configured.

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
