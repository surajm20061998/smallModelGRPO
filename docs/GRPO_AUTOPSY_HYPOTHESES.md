# GRPO Autopsy: Hypotheses and Measurements

Use this as the Day-1 planning artifact before full experiment sweeps.

## Candidate theses

1. Length growth is partly reward hacking rather than improved reasoning.
2. Reasoning behaviors (backtracking, verification, case analysis) emerge in phases.
3. Entropy collapse is asymmetric across token categories.
4. Standard-deviation normalization effects are stronger at small scale.

## Falsifiable measurements

### A) Length and reward hacking

- Length distribution over training steps (mean + variance + quantiles)
- Length conditional on reward:
  - `length | reward=1`
  - `length | reward=0`
- Per-problem trajectory on fixed probes

### B) Reasoning behavior emergence

- Regex detector frequency per behavior family:
  - backtracking
  - verification
  - case analysis
- LLM-judge validated subset for precision/recall checks
- Behavior frequency vs rollout step

### C) Entropy collapse asymmetry

- Token entropy by:
  - token position bucket (early/mid/late)
  - token type (numbers/operators/structure/text)
  - correctness bucket (correct/incorrect rollout)
- Entropy-vs-step slope comparisons across buckets

## Fixed-probe policy

- Use a fixed held-out probe set for longitudinal analysis
- Keep probe set constant across checkpoints and seeds
- Save probe manifest (`probe_set.json`) with:
  - dataset index
  - difficulty bucket
  - prompt text
  - ground truth metadata
