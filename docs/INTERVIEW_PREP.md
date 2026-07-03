# Interview Prep: Technical Deep-Dive Defense

Mock-interview Q&A for this project, written from the perspective of a
frontier-lab interviewer doing a hostile-but-fair technical deep dive. Every
answer is grounded in this repo's actual code, measurements, failures, and
results. Companion docs: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the
narrative, [GRPO_AUTOPSY_HYPOTHESES.md](GRPO_AUTOPSY_HYPOTHESES.md) for the
research design.

**How to use this:** each section escalates from warm-up to attack questions.
"Trap" callouts mark where an interviewer is fishing for a shallow answer.
Numbers in answers are real — memorize the bolded ones; they are what makes
an answer land as "this person actually did the work."

**The 30-second pitch:**
> I built a full SFT + GRPO training stack for small math reasoners, then
> instrumented it to answer *why* GRPO behaves the way it does at small scale
> — not just whether accuracy moves. The core artifact is an "autopsy" layer:
> a fixed, difficulty-stratified probe set re-rolled at every training stage,
> with per-token log-probs and entropy recorded, so training dynamics like
> entropy collapse and reward hacking are measurable after the fact. It paid
> off concretely: my instrumentation caught a silently frozen policy, a
> numerically corrupt eval path on Apple silicon, and produced a clean
> laptop-scale demonstration that GRPO's std-normalization can destroy
> learning at small group sizes — the policy's format rate went 0.66→1.0
> without it and 0.59→0.0 with it, from the exact same seed.

---

## 1. GRPO fundamentals — the math you must not fumble

**Q: Walk me through GRPO. How is it different from PPO?**

**A:** Both optimize a clipped policy-gradient objective; the difference is
the baseline. PPO trains a separate value network and uses GAE advantages.
GRPO deletes the critic: you sample a *group* of G rollouts per prompt, and
the advantage of rollout *i* is its reward minus the group mean,
`A_i = r_i − mean(r_1..r_G)`, optionally divided by the group std. The group
mean is a Monte-Carlo estimate of the value of the prompt. You pay for the
missing critic with G× more generation per prompt and a noisier baseline; you
gain simplicity, no value-model memory, and no value-model *mis*fit. In my
implementation ([src/train/grpo.py](../src/train/grpo.py)),
`compute_group_normalized_rewards` reshapes rewards to
`(num_prompts, group_size)`, subtracts the row mean, and optionally divides by
`(row_std + 1e-6)`.

**Q: Prove the group-mean baseline doesn't bias the gradient.**

**A:** For any baseline `b` that's independent of the sampled action,
`E_π[b·∇log π(a)] = b·∇E_π[1] = b·∇1 = 0`, so subtracting it leaves the
expected gradient unchanged and only reduces variance. Subtlety worth
volunteering: the group mean *includes* rollout *i* itself, so `b` is not
strictly independent of `a_i` — this introduces an O(1/G) bias. Some
implementations use leave-one-out (LOO / RLOO) means to fix this; at G=8 the
effect is small, at my local G=4 it's larger — one more reason small-group
GRPO is statistically fragile, which is exactly the regime I studied.

**Q: Your repo has three loss variants. Why all three?**

**A:** They form a controlled ladder. `no_baseline` is REINFORCE on raw
rewards — the high-variance control. `reinforce_with_baseline` is
`−A_i·log π_t` — vanilla on-policy GRPO, correct when you take exactly one
gradient epoch per rollout batch. `grpo_clip` adds the PPO ratio machinery:
`ratio = exp(logπ − logπ_old)`, objective `min(ratio·A, clip(ratio,1±ε)·A)`
with ε=0.2, logging `clipfrac`. It's only *needed* when
`--epochs-per-rollout-batch > 1` makes the data off-policy after the first
step. Keeping them as one function with a switch
(`compute_policy_gradient_loss`) means an ablation changes exactly one flag.

*Trap — "why is clipping the `min` of two terms and not just clamping the
ratio?"* The `min` makes the objective *pessimistic*: clipping only binds
when it would make the objective larger (further policy movement in the
already-favored direction). If you clamped the ratio always, you'd also
weaken the gradient that pulls the policy *back* when it has drifted the
wrong way against the advantage sign.

**Q: When do you score `old_log_probs`, and with which model?**

**A:** With the current policy, in inference mode, *before* any update from
that rollout batch — they must represent the sampling distribution that
generated the data. In the loop this is `score_old_log_probs(...)` right
after tokenization, only when `loss_type == "grpo_clip"`. Scoring them lazily
after the first gradient step would silently zero the first epoch's ratios.

**Q: Explain your two length-normalization modes and why anyone cares.**

**A:** After computing per-token loss, you reduce over the response
dimension. `masked_mean` divides by *that response's* token count: every
response contributes equally to the batch loss, but each token of a long
response carries a 1/len-scaled gradient. `masked_normalize` divides by a
*constant* (max response length in the batch): per-token gradient magnitude
is length-independent, so long responses contribute proportionally more total
gradient. This is not pedantry — it's the Dr. GRPO length-bias argument:
per-response mean normalization mutes the penalty on long *wrong* answers
(each wrong token gets a smaller gradient the longer you ramble), which is a
candidate mechanism for RLVR length inflation. In my stack it's one flag
(`--length-normalization`) so it's ablatable in isolation.

**Q: Why is dividing by group std controversial? You clearly had an opinion.**

**A:** Three compounding problems. (1) It reweights *prompts*: a group with
low reward variance — nearly-all-correct or nearly-all-wrong — gets its
advantages amplified by 1/std, so the easiest and hardest prompts dominate
the gradient. (2) At small G the std estimate is extremely noisy: with G=4
binary-ish rewards, one odd rollout out of four gives std≈0.5·range, and a
3-vs-1 group divides by ~0.43, more than doubling the advantage. (3) The
ε in `(std+1e-6)` means near-degenerate groups explode rather than vanish.
My ablation made this concrete: identical seed and data, and the std-norm arm
ran at **16× higher mean grad-norm (2.92 vs 0.18, max 13.1)** and *destroyed*
its initial format behavior (0.59→0.00) while the unnormalized arm learned it
to 1.00.

*Trap — "isn't that just a learning-rate effect? Divide the LR by 16 and
std-norm is fine."* Partially, and you should concede it: std-norm raises the
*average* effective step size, and a global LR cut would reduce the blowups.
But the pathology is the *non-uniformity* — it selectively amplifies
low-variance groups, so no single global LR equalizes the two arms; you'd be
trading collapse for near-zero learning on high-variance groups. The clean
follow-up experiment (which I'd run next) is std-norm ON at lr/16: my
prediction is no collapse but much slower format learning than the
unnormalized arm, isolating scale from selectivity.

---

## 2. Task and reward design

**Q: Why Countdown as the RL task instead of MATH/GSM8K?**

**A:** Three properties. It's *programmatically verifiable* — an exact
symbolic check, no LLM judge and no learned reward model, so an entire class
of reward-hacking confounds is off the table. It's *difficulty-controllable*
— numbers count × target magnitude gives a natural stratification I use for
the probe set. And it's *genuinely hard for small models*, so there's a
learning signal to observe rather than a saturated benchmark. The reward
decomposes into `format_reward` (emitted a well-formed `<answer>` block) and
`answer_reward` (the arithmetic verifies), logged separately — which is how I
could see "learned the format, not the math" as a *dynamics* fact.

**Q: Walk me through the verifier. What attack surface does it have?**

**A:** [grader_countdown.py](../src/grading/grader_countdown.py). Extraction:
regex for `<answer>…</answer>`. Parsing: Python `ast.parse(mode="eval")` and
a recursive evaluator that whitelists numeric constants, unary ±, and
`+−*/` — nothing else evaluates, so a policy being *optimized against this
function* can't smuggle code in. Arithmetic: exact `Fraction`s, so
`(1/3)*3 == 1` — floats would produce wrong verdicts on divisions. Number
usage: operands are multiset-matched against the available numbers and
consumed; in the step format (`Step 1: 1+2 = 3`) each intermediate result is
appended back to the pool, per Countdown rules, and any step whose
`lhs != rhs` fails the whole answer. It also normalizes `×`, `÷`, `3 x 4`,
code fences, and step prefixes. Known remaining softness: the answer-block
regex is permissive (case-insensitive, DOTALL), and the model can (and did)
satisfy `format_reward` with degenerate short blocks — which is a *shaping*
problem, not a verifier bug.

**Q: You added reward shaping mid-project. Isn't that moving the goalposts?**

**A:** It's a documented, principled response to a measured cold start — not
tuning toward a desired result. The 0.5B model produced **zero reward on all
113 rollout steps** of the first ablation: mean response length was ~250 of a
256-token cap (it almost never emitted `</answer>`; format rate ~6%), so
every group was all-zeros, every advantage was exactly zero, and
`grad_norm == 0.0` for the entire run — the policy never moved. I probed
512 tokens: format 19%, answers still 0/16. So no token budget rescues the
*answer* signal on this model; the group needs variance from somewhere.
`--format-reward-weight w` makes the *training* reward `answer + w·format`
(w=0.2), while the recorder and all eval metrics keep the pure reward. It
worked immediately — step 1 of the rerun had reward_mean 0.075 and advantage
std 0.65. Crucially, the shaping is *identical across both ablation arms*, so
the A/B contrast is untouched.

*Trap — "your shaped reward got hacked, didn't it?"* Yes — own it, it's the
best part. The successful arm drove mean response length from 68 to **12
tokens**: minimal `<answer>` blocks that earn the 0.2 format credit with no
real attempt at the target. That is textbook proxy optimization, caught by my
own instrumentation (length-vs-reward analysis), and it's hypothesis 1 in
miniature. The next intervention is explicit in the docs: decay `w` over
steps or gate it on nonzero answer reward within the group.

---

## 3. Systems architecture

**Q: Draw the training loop. Where does the time go?**

**A:** Per rollout step: (1) sample `rollout_batch_size/group_size` unique
prompts, (2) generate G rollouts per prompt — vLLM on CUDA, my HF-generate
backend on MPS/CPU, (3) grade → group-normalized advantages, (4) for
grpo_clip only, score old log-probs, (5) shuffle → train batches →
microbatches, each microbatch runs teacher-forced forward, per-token loss
under the response mask, `loss.backward()`; clip grad-norm and
`optimizer.step()` per train batch, (6) every N steps: autopsy recorder on
the fixed probe set; every M steps: greedy dev eval + best-checkpoint
selection, (7) on CUDA, push updated weights into the vLLM engine.
**Generation dominates** — measured locally ~89 tok/s aggregate for the 0.5B
in fp32 on MPS vs ~1.5 s per training microbatch; on A100s it's why vLLM
exists in the loop at all.

**Q: Why run two copies of the model on CUDA? Defend the memory cost.**

**A:** The trainable HF policy and the vLLM engine are different runtimes
over the same weights. HF `generate()` is far too slow for RL rollout volume;
vLLM's paged-attention KV cache, continuous batching, and prefix caching
(rollout groups share the full prompt prefix — a large win by construction)
give order-of-magnitude faster sampling but can't backprop. So: policy on
`cuda:0`, vLLM on `cuda:1`, and after each update
`load_policy_into_vllm_instance` pushes the `state_dict` into
`llm_engine.model_executor.driver_worker.model_runner.model.load_weights`.
Costs: 2× weights, a per-step sync (a ~3 GB device-to-device copy for 1.5B
bf16 — amortized against minutes of generation), and version skew risk since
I'm reaching into vLLM internals. For a 2-GPU A100 node this is the right
trade; the sync cost only becomes structural at much larger scale, where
you'd move to a disaggregated design.

**Q: Why not just use TRL / verl / OpenRLHF?**

**A:** Two honest reasons. First, provenance: the training loop began as
coursework where the algorithmic core had to be written by hand — and for a
research project *about* GRPO's internals, owning every line of the advantage
and loss computation is a feature: when I ablate std-norm I know precisely
what changes, with no framework defaults (KL terms, reward whitening, length
penalties) silently in the loop. Second, the autopsy layer needs surgical
hooks — rescoring rollouts with the live policy mid-step, recording
per-token entropy under the training tokenization — which fight framework
abstractions. The cost I accept: no battle-tested distributed backend, no
resume-from-checkpoint, fewer eyes on the code. For a production RLHF system
I would start from verl or TRL and contribute hooks instead.

**Q: Explain your tokenization/masking scheme. Where do people get this wrong?**

**A:** `tokenize_prompt_and_output` concatenates prompt+response token ids,
builds a boolean mask that's True only on response tokens, then shifts:
`input_ids = full[:, :-1]`, `labels = full[:, 1:]`, mask aligned to labels.
Everything downstream — SFT loss, GRPO loss, entropy recording — reduces
per-token quantities under that mask, so prompt tokens and padding never
receive gradient. Log-probs are `log_softmax(logits).gather(-1, labels)`.
Classic mistakes this avoids: off-by-one between logits and labels (logits at
position t predict token t+1), leaking loss onto prompt tokens, and padding
contaminating means — my `masked_mean` divides by mask sum, not sequence
length. One deliberate quirk: prompt and response are tokenized *separately*
and concatenated, which can differ from tokenizing the joined string at the
boundary token; it's consistent across training and scoring, which is what
correctness requires here.

**Q: How does gradient accumulation interact with your loss scaling?**

**A:** Each microbatch computes its loss, divides by the number of
microbatches in the train batch, and calls `backward()` itself inside
`grpo_microbatch_train_step`; the driver clips grad-norm and steps the
optimizer once per train batch. The division must happen before `backward()`
so accumulated gradients equal the full-batch gradient. An edge case I handle
in SFT: the final partial accumulation window rescales by the *actual*
number of remaining microbatches, not the nominal one, so the last optimizer
step isn't systematically down-weighted.

**Q: You lost your A100s mid-project. What did the port actually require?**

**A:** vLLM is CUDA-only, so I abstracted the generator behind an interface:
[hf_generate.py](../src/infer/hf_generate.py) wraps HF `generate()` but
returns vLLM-`RequestOutput`-shaped objects (`out.outputs[0].text`,
`.finish_reason`, `.stop_reason`) — the training loop, eval, and recorder
needed *zero* call-site changes. Because `HFGenerator` holds a live reference
to the policy module, the weight-sync step becomes a no-op locally: one model
in memory instead of two. Backend selection is `--backend auto|cuda|mps|cpu`
(auto picks cuda>mps>cpu), dtype defaults bf16-on-CUDA / fp32-on-MPS, and
selecting cuda without vLLM installed fails fast with an actionable error
instead of a raw ImportError. The same flags were threaded through both
`run_grpo.py` and `run_sft.py`, so the CUDA path is preserved verbatim.

*Trap — "why fp32 on MPS? That halves your effective memory."* bf16 training
on MPS is poorly supported and numerically risky (loss-scale issues, op
fallbacks silently upcasting); fp16 needs loss scaling I didn't want to own
for a 0.5B model that fits comfortably in fp32. Correctness of the *research
measurements* beats speed here — entropy slopes computed on a numerically
shaky substrate are worthless.

---

## 4. The autopsy instrumentation — the original contribution

**Q: Why a fixed probe set? Why not analyze the training rollouts you
already have?**

**A:** Training rollouts come from *different prompts each step* — any
longitudinal statement ("entropy fell", "length grew") confounds policy
change with prompt-mix change. The probe set is drawn once per run —
difficulty-stratified ~34/33/33 easy/medium/hard by numbers-count and target
magnitude, deterministic seed, saved to `probe_set.json` — and re-rolled at
every recording step. Same prompts across checkpoints *and across seeds and
ablation arms* means trajectories are directly comparable; my std-norm A/B is
only clean because both arms probe the identical eight problems. Training
rollouts *are* also logged (`rollout_history.jsonl`, examples), but as
diagnostics, not the measurement instrument.

**Q: Why generate-then-rescore? You're paying two passes.**

**A:** Because the two passes answer different questions and no single pass
does both. Generation must come from the *live sampling distribution*
(temperature 0.7 — what training actually sees, not greedy). But generation
engines don't give me clean per-token, full-vocab entropy under my training
tokenization — vLLM's logprobs are top-k under its own processing, and HF
generate's scores are pre-processor logits I'd have to re-normalize
carefully. So I re-run the sampled text through a teacher-forced forward
pass (`get_response_log_probs` with `return_token_entropy=True`) and get
exact `−Σ p log p` over the full vocab plus per-token log-probs, computed
identically on every backend. The rescoring cost is bounded (8 probes × 4
rollouts every 5 steps) and it bought me backend-independence of all
measurements.

**Q: What exactly lands on disk, and why both JSON and tensors?**

**A:** Per recorded step: one human-inspectable JSON per probe (prompt,
ground truth, difficulty meta, and per rollout the text, reward dict, tokens,
token ids, per-token log-probs and entropies) plus a consolidated
`rollout_tensors.pt` (log-probs, entropy, mask, ids as
`(num_rollouts × seq_len)` tensors). JSON is for eyeballing single
trajectories and for the pure-Python analysis module; tensors are for
vectorized analyses I haven't invented yet. Same reason I snapshot full HF
checkpoints periodically: **record raw signal, defer analysis** — you cannot
re-run an A100 sweep after graduation (I know, I lost mine), but you can
always re-analyze saved artifacts.

**Q: Your recorder has OOM backoff. Why is that load-bearing?**

**A:** The recorder shares accelerator memory with a training process at its
peak. Its scoring loop catches CUDA OOM, halves the batch size down to 1,
and only then fails — with a message naming the exact flags to lower. Design
principle: instrumentation must *degrade*, never kill an 8-hour run. The
mirror image of that principle showed up locally: it was the *training*
backward that OOM'd on MPS, and the fix (below) went into the loop, not the
recorder.

---

## 5. Debugging war stories — expect these to fill half the interview

**Q: Tell me about a bug that a normal metrics dashboard would have missed.**

**A:** The frozen-policy ablation. My first std-norm A/B came back with
**bit-identical metrics across both arms** — every slope, every frequency.
A dashboard showing "accuracy 0, loss 0" looks like "model is just weak."
The comparison table made identity undeniable, and the logs reduced it to
three lines: `reward_mean == 0.0` on all 113 rollout steps →
group-relative advantages identically zero (with *or without* std-norm — 0/σ
is still 0) → `grad_norm == 0.0` throughout → the optimizer never made an
effective update, so both "runs" were the same frozen initial model sampling
with the same seed. Root cause was RLVR cold start: the 0.5B almost never
finished `</answer>` inside 256 tokens (mean length 249.8/256, format ~6%).
Fix: shaped reward (above). The meta-lesson I'd state plainly: **a run that
does nothing looks exactly like a run that's learning slowly unless you log
the gradient norm** — it's the cheapest dead-man's switch in RL.

**Q: You found a numerical bug on MPS. Walk me through the forensics.**

**A:** While probing token budgets, sampled generation crashed:
`torch.multinomial: probability tensor contains inf/nan`. The crash was in
*my probe script* but training had run for hours — so what differed? The
probe batched *different-length* prompts (left-padded); training rollout
groups repeat *one* prompt per batch of 4 — no padding, by accident of
construction. Hypothesis: padded mixed-length batches produce NaN logits
under SDPA on the MPS backend. Test: identical-prompt batch sampled fine;
mixed-length batch crashed — confirmed. The nasty part: **greedy decoding
doesn't crash on NaN — argmax over NaNs silently returns garbage**, and my
dev/test evals batch mixed-length prompts greedily, so v1's eval numbers were
untrustworthy (they read 0.0, which was "right," but for the wrong reason).
Fix in `HFGenerator`: group prompts by exact token length and batch only
equal lengths — padding is never constructed at all. Bonus: dropping pad
tokens made rollout steps ~2.5× faster (~1.3 vs ~3.5 min/step).

*Trap — "why not just fix the attention mask / upgrade torch?"* The mask was
already correct — this is a backend numerical bug beneath my API surface. An
upgrade might fix it, but I don't control the user's torch build; the
equal-length grouping is correct on every backend, costs nothing for
rollouts (identical prompts), and degrades evals to per-length chunks. I
chose the fix that removes the failure *class*, not the instance.

**Q: And the OOM saga? You shipped two fixes that didn't work first.**

**A:** Accurate, and worth being precise about why. Once real gradients
flowed (post-shaping), backward OOM'd at step ~28 —
`MPS allocated 8.98 GiB, other allocations 13.24 GiB, max allowed 22.64 GiB`,
failing on a **519 MiB** request, which is exactly one fp32 logits buffer
(~890 tokens × 152k vocab × 4 B). Fix #1: `torch.mps.empty_cache()` at the
generation↔training phase boundaries. Result: died at step 30 — moved the
needle two steps. The tell was in the error itself: "other allocations" —
memory *outside* PyTorch's pool — was the growing term, i.e., Metal
kernel/graph caches and autograd graph objects whose Metal buffers are only
released after Python GC runs. Fix #2: `gc.collect()` *before*
`empty_cache()` (so buffers are actually reclaimable), plus lifting the MPS
allocator's high-watermark cap (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`) —
the cap was killing an allocation that unified memory could absorb. I also
de-risked the run from 60→45 steps since slope estimation needs ~9
snapshots, not 12. Result: both arms completed in ~35 min each. The pattern
I'd generalize: on MPS, `empty_cache` without `gc.collect` is a no-op for
graph-held memory, and the watermark default is tuned for interactive use,
not training.

**Q: Anything embarrassing you'd own?**

**A:** Two. I filled the disk — fp32 checkpoints of even a 0.5B are ~2 GB
and I was snapshotting every 25 steps on a laptop; run 2 died mid-save with
9.7 GiB free. Fixed by disabling snapshots for the reruns and deleting
checkpoints of the *provably frozen* v1 models (their weights equal the base
model — zero grad-norm for the whole run — so nothing was lost). And my
original time estimate for the local runs was off by ~2.5× until I measured
step timestamps — padded generation and thermal effects; estimate from
artifact mtimes, not from first principles.

---

## 6. Results — defend the science

**Q: State your ablation result and every caveat you'd put on it.**

**A:** Result: with everything held fixed (model, seed, data order, probe
set, shaping w=0.2, 45 steps, G=4), `--normalize-by-std` **on** collapsed
format behavior 0.59→0.00 with mean grad-norm 2.92 (max 13.1); **off**
learned format 0.66→**1.00** (dev format rate 1.0) with mean grad-norm 0.18.
Entropy corroborates: the learning arm shows controlled entropy decline
(−0.0064/step overall), the collapsing arm drifts *up* (+0.0023) — a
degeneration signature. Caveats, unprompted: **n=1 seed** (the arms share a
seed, so the contrast is paired, but I can't speak to run-to-run variance);
0.5B non-math model, so magnitudes won't transfer to 7B+; reward is shaped,
so the group-variance structure std-norm amplifies is partly of my own
making; 45 steps — I can't rule out the std-norm arm recovering later,
though grad-norm 13 spikes make "recovery" optimistic; and the probe set is
8 prompts × 4 rollouts, fine for a 0→1 format contrast, underpowered for
subtle effects.

**Q: What would falsify your interpretation?**

**A:** Three experiments, all one-command in this repo: (1) std-norm ON at
lr/16 — if it learns cleanly, the effect is mostly scale, not selectivity,
and my "non-uniform amplification" framing weakens; (2) two more seeds per
arm — if arms overlap across seeds, the single-seed contrast was luck;
(3) larger G (8, 16) with std-norm ON — the pathology should shrink like
1/√G if my variance story is right. I'd also gate the format credit and
re-run, to check the collapse isn't an artifact of shaping specifically.

**Q: Your entropy-asymmetry claim sounds like reading tea leaves from two
points.**

**A:** Fair challenge for the smoke run — but the v4 slopes are over 9
snapshots. In the learning arm, text-token entropy collapses fastest
(−0.012/step) while operator-token entropy slightly *rises* (+0.002): the
model commits to the scaffold and stays uncertain over the arithmetic — which
is what you'd *want* exploration to look like. I'd still call it a "signal,
not a finding" until it survives more seeds and an LLM-judge check on the
token classifier — the classifier is a regex over BPE strings
(number/operator/structure/text), and I've validated it by inspection, not
systematically.

**Q: Nothing ever solved a Countdown problem. Doesn't that undermine the
whole exercise?**

**A:** It bounds the claims, and I keep the claims inside the bound. The
project's dependent variables are *dynamics* — format acquisition, gradient
scale, entropy structure, length behavior — all of which moved measurably
and differentially across arms. "0.5B-Instruct can't solve Countdown in 45
steps at 256 tokens" was expected (it's why the original design used
Math-1.5B on A100s). The infrastructure's value is exactly that the same
commands scale up when compute returns; the local runs are the
methodological demonstration.

---

## 7. Design-defense lightning round

**Q: Why no KL penalty to a reference model? Every production RLHF system
has one.**

**A:** Deliberate, two-layer answer. Scientifically: R1-Zero-style minimal
GRPO omits it, and my project studies GRPO's *intrinsic* dynamics — adding a
KL anchor would mask exactly the instabilities I'm trying to measure (the
std-norm collapse might have been invisible under a KL leash). Practically:
KL-to-reference requires a frozen second model forward on every microbatch —
prohibitive on a laptop. When I wanted an anti-collapse intervention, I chose
an entropy bonus instead: `−β·H(π_t)` added per-token, using the
*differentiable* entropy from logits already computed in the training forward
— zero extra model, zero extra forward, and it targets hypothesis 3
directly. I verified the loss delta analytically (equals `−β·mean(H)`) and
that gradient flows through the entropy term.

**Q: Why CLI flags everywhere instead of config files? Your `configs/*.yaml`
are empty.**

**A:** Guilty as designed. Every run's full config is snapshotted to
`config.json` in its output dir, which is the property that matters for
reproducibility; the YAML placeholders were for a config-driven sweep
runner that never became the bottleneck — bash sweep scripts with env-var
overrides (`run_autopsy_suite.sh`, `run_grpo_local.sh`) got the same job
done with less machinery. In a team setting I'd adopt Hydra/OmegaConf for
composability; solo, flags + snapshot + wrapper scripts is honestly fewer
failure modes.

**Q: No resume-from-checkpoint. Your runs died three times. Defend that.**

**A:** The weakest point in the stack, and I'll say so before you do. Exact
RL resume is genuinely hard — you must restore policy, optimizer state,
data-order RNG, *and* the generation RNG stream to reproduce the trajectory,
and a mid-run restore that isn't bitwise-faithful is arguably worse than a
restart for a *measurement* project because it introduces an invisible
discontinuity into every longitudinal series. At 35–90 minute local runs,
restart-from-scratch was the cheaper correctness guarantee. At A100 sweep
scale it's the first thing I'd build: checkpoint the optimizer + RNG states
alongside the model every eval, and accept a documented discontinuity flag
in the autopsy series at resume points.

**Q: Your local eval caps are 32 dev / 64 test examples. That's tiny.**

**A:** Measured trade: greedy eval generation was the dominant local
wall-clock cost (the defaults, 256/1024, would have added ~40 minutes per
eval cycle — more than the training itself). At a 0→1 format-rate contrast,
n=32 gives a standard error around ±0.09 — adequate for the effects I was
reading, useless for a 2-point accuracy claim, which is why I never make
one locally. The caps are env-var overrides, not hardcoded.

**Q: Why Qwen2.5-0.5B-Instruct specifically?**

**A:** Forced move plus a constraint: no public Qwen2.5-Math-0.5B exists, so
the smallest Math-family model was off the table; 0.5B-Instruct is the
largest model where a full GRPO step loop (fp32 policy + gradients + AdamW
state + generation KV) fits comfortably in unified memory at usable speed —
measured 89 tok/s generation, 1.5 s/microbatch. The instruct variant matters:
a base model would have an even worse format cold-start, and my first
experiment already died of that with an instruct model.

**Q: You reached into vLLM internals
(`llm_engine.model_executor.driver_worker...`). That'll break on upgrade.**

**A:** Yes — it's a version-pinned hack, standard in small RLHF stacks of
this vintage, and it's quarantined in one function
(`load_policy_into_vllm_instance`) so an upgrade breaks one symbol, not the
loop. The durable fix is vLLM's collective-RPC weight-update path or a
disaggregated rollout service; for a pinned research environment, the
one-liner with a pinned `vllm` version was the right cost-benefit.

---

## 8. Scale-up questions (they will ask; have a view)

**Q: How would you take this to a 70B model on a cluster?**

**A:** Split the monolith into a *trainer* and a *rollout fleet*. Trainer:
FSDP or megatron-style TP/PP shards, bf16, microbatched exactly as now — the
loss code is already per-token/mask-based and shape-agnostic. Rollouts: a
pool of vLLM (or SGLang) servers with tensor-parallel shards, fed prompts
over RPC; weight sync via broadcast of sharded state dicts (or vLLM's
update-weights RPC) every K steps rather than every step, accepting bounded
staleness — which is precisely when my `grpo_clip` variant with
`old_log_probs` stops being optional and becomes the correctness mechanism.
The autopsy recorder ports cleanly because it's already two-pass: probes go
to the rollout fleet; rescoring is a sharded forward on the trainer. The
things that need rethinking at scale: JSON-per-probe becomes a metadata
store + Parquet/Arrow tensors; and checkpoint snapshots move to async
distributed checkpointing.

**Q: Generation is your bottleneck. What are the three biggest levers?**

**A:** (1) Prefix caching — rollout groups share the entire prompt, so the
prefill for G−1 of G rollouts is nearly free; vLLM gives this, and my local
backend gets a weaker version by batching identical prompts. (2) Continuous
batching so eval and rollout traffic share the engine at high occupancy.
(3) Overlap: generate rollout batch t+1 while training on batch t — one step
of off-policyness, handled by the clip ratio. Locally the honest lever was
different: removing pad tokens (the equal-length batching fix) was worth
2.5× before any clever scheduling.

**Q: What would you log that you aren't already?**

**A:** Per-group advantage/std histograms (I log batch-level means — the
std-norm pathology would have been visible a step earlier in the tails);
importance-ratio distributions beyond mean clipfrac for the off-policy runs;
KL(π_t‖π_0) on the probe set as a drift odometer even without a KL loss
term; and wall-clock per phase (gen/score/train/eval) — my ETA misses came
from not having phase timings as first-class metrics.

---

## 9. Rapid-fire — one-breath answers

- **Advantage with all-identical rewards in a group?** Zero (and with
  std-norm, 0/(0+ε) is still zero) — such groups contribute no gradient;
  this is why all-zero rewards froze the policy.
- **Why `advantage_eps`?** Avoid division blow-up as std→0; ε=1e-6 is small
  enough that near-degenerate groups still explode — arguably it should be
  1e-2-scale or the group should be dropped; that's a defensible change I
  didn't make.
- **Entropy formula and where it's computed?** `H = −Σ_v p_v log p_v` over
  the full vocab from `log_softmax(logits)`; computed in the teacher-forced
  pass; differentiable when used as a bonus, detached when logged.
- **Why does the recorder append the stop string back?** vLLM excludes the
  matched stop by default; the grader's regex needs `</answer>` present, so
  `get_output_text` re-appends it when `finish_reason=="stop"` matched it.
- **Why left-padding for decoder generation (before the no-pad fix)?**
  Right-padding puts pads between prompt and generation, corrupting the
  first generated positions; left-padding keeps the prompt adjacent to
  generation. (Now moot locally — I never pad.)
- **`masked_mean` vs `masked_normalize` in one line?** Per-example mean vs
  constant-denominator sum: equal-per-response vs equal-per-token gradient
  weighting.
- **What does `clipfrac` tell you?** Fraction of tokens where the clipped
  objective was the binding term — a drift/staleness gauge; persistently
  high clipfrac means your data is too off-policy for your ε.
- **Temperature for probes and why?** 0.7 — the training sampling
  distribution; measuring entropy of greedy rollouts would answer a
  different (less relevant) question.
- **Why is `best_metric` selected on dev and reported on test?** Standard
  selection/report split — `summary.json` carries both; the final test eval
  reloads the best checkpoint, not the last policy.
- **One thing you'd delete from the project?** The per-rollout-step
  `save_pretrained(last_dir)` — a 2 GB serialization every step is the
  disk-full incident waiting to recur; last-checkpoint cadence should be
  time-based.

---

## 10. Questions to ask *them* (signals seniority)

1. When your RL runs plateau or collapse, what's the standard forensic
   tooling — do teams keep per-token trajectories, or is it metric
   dashboards and rerun-with-more-logging?
2. How do you handle the generation/training weight-sync problem at your
   scale — synchronized engines, bounded staleness, or fully asynchronous
   with importance correction?
3. Is reward shaping treated as a config knob or a reviewed design change?
   Who owns "the reward got hacked" incidents?
4. How much of the training stack is owned in-house vs built on open
   frameworks, and where has that boundary bitten you?
