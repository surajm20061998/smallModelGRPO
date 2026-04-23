import argparse
import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

try:
    import wandb
except Exception:
    wandb = None

from src.data_bootstrap import ensure_repo_data_path, resolve_repo_path
from src.grading.grader_countdown import (
    countdown_reward_fn,
    format_countdown_prompt,
    load_countdown_prompt,
    load_countdown_split,
    prepare_countdown_eval,
)
from src.autopsy.probe_set import build_fixed_countdown_probe_set
from src.autopsy.rollout_recorder import RolloutRecorder
from src.train.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from src.train.sft import get_response_log_probs, tokenize_prompt_and_output


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument(
        "--prompt-path",
        default=str(REPO_ROOT / "configs" / "prompts" / "countdown.prompt"),
    )
    parser.add_argument(
        "--countdown-train-path",
        default=str(REPO_ROOT / "data-distrib" / "countdown" / "train_10k.parquet"),
    )
    parser.add_argument(
        "--countdown-dev-path",
        default=str(REPO_ROOT / "data-distrib" / "countdown" / "dev.parquet"),
    )
    parser.add_argument(
        "--countdown-test-path",
        default=str(REPO_ROOT / "data-distrib" / "countdown" / "test.parquet"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument(
        "--loss-type",
        default="reinforce_with_baseline",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    )
    parser.add_argument("--normalize-by-std", dest="normalize_by_std", action="store_true")
    parser.add_argument("--no-normalize-by-std", dest="normalize_by_std", action="store_false")
    parser.set_defaults(normalize_by_std=True)
    parser.add_argument(
        "--length-normalization",
        default="masked_mean",
        choices=["masked_mean", "masked_normalize"],
    )

    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-min-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--stop-sequence", default="</answer>")
    parser.add_argument("--old-logprob-batch-size", type=int, default=8)
    parser.add_argument("--num-rollout-examples-to-log", type=int, default=3)

    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-before-train", action="store_true")
    parser.add_argument("--countdown-dev-max-examples", type=int, default=256)
    parser.add_argument("--countdown-test-max-examples", type=int, default=1024)

    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:1")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wandb-project", default="smallModelGrpo")
    parser.add_argument("--wandb-entity", default="sm12377-new-york-university")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-output-artifact", dest="wandb_log_output_artifact", action="store_true")
    parser.add_argument("--no-wandb-log-output-artifact", dest="wandb_log_output_artifact", action="store_false")
    parser.set_defaults(wandb_log_output_artifact=True)
    parser.add_argument("--enable-autopsy-recorder", action="store_true")
    parser.add_argument("--autopsy-every", type=int, default=10)
    parser.add_argument("--autopsy-num-probe-prompts", type=int, default=50)
    parser.add_argument("--autopsy-probe-seed", type=int, default=123)
    parser.add_argument(
        "--autopsy-probe-split",
        default="dev",
        choices=["train", "dev", "test"],
    )
    parser.add_argument(
        "--autopsy-group-size",
        type=int,
        default=None,
        help="Rollouts per fixed probe prompt. Defaults to group-size.",
    )
    parser.add_argument(
        "--autopsy-checkpoint-every",
        type=int,
        default=50,
        help="Also snapshot a checkpoint every N rollout steps when recorder is enabled.",
    )
    parser.add_argument(
        "--autopsy-logprob-batch-size",
        type=int,
        default=16,
        help="Microbatch size for autopsy policy scoring (log-probs/entropy).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def init_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def init_policy(model_id: str, device: str, gradient_checkpointing: bool):
    from transformers import AutoModelForCausalLM

    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    if gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    return policy


def ensure_pynvml_compat() -> None:
    try:
        import pynvml  # type: ignore
    except Exception:
        return

    if hasattr(pynvml, "nvmlDeviceGetCudaComputeCapability"):
        return

    def _compat_nvml_device_get_cuda_compute_capability(handle):
        try:
            device_index = pynvml.nvmlDeviceGetIndex(handle)
        except Exception:
            device_index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
        return int(major), int(minor)

    pynvml.nvmlDeviceGetCudaComputeCapability = _compat_nvml_device_get_cuda_compute_capability


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    ensure_pynvml_compat()
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def maybe_init_wandb(args, config: dict[str, Any]):
    if args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed but wandb logging was requested")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=config,
        mode=args.wandb_mode,
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("rollout_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    wandb.define_metric("rollout/*", step_metric="rollout_step")
    return run


def maybe_log_wandb(run, metrics: dict[str, Any]) -> None:
    if run is not None:
        run.log(metrics)


def maybe_log_wandb_output_artifact(run, output_dir: Path, args) -> None:
    if run is None or not args.wandb_log_output_artifact:
        return
    if wandb is None:
        return

    artifact_name = args.run_name or output_dir.name or "grpo_run"
    artifact_name = artifact_name.replace("/", "-")
    artifact = wandb.Artifact(
        name=f"{artifact_name}-outputs",
        type="training-run-output",
        description="End-to-end GRPO run outputs (metrics, rollouts, checkpoints, autopsy artifacts).",
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)


def make_sampling_params(
    *,
    temperature: float,
    max_tokens: int,
    stop_sequence: str | None,
    top_p: float = 1.0,
    min_tokens: int = 0,
):
    from vllm import SamplingParams

    kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
    }
    if stop_sequence:
        kwargs["stop"] = [stop_sequence]
        # Countdown grading expects the closing tag to remain in the text.
        kwargs["include_stop_str_in_output"] = True
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("include_stop_str_in_output", None)
        return SamplingParams(**kwargs)


def get_output_text(vllm_output, stop_sequence: str | None = None) -> str:
    if not getattr(vllm_output, "outputs", None):
        return ""
    if len(vllm_output.outputs) == 0:
        return ""
    output = vllm_output.outputs[0]
    text = output.text
    if stop_sequence and stop_sequence not in text:
        finish_reason = getattr(output, "finish_reason", None)
        stop_reason = getattr(output, "stop_reason", None)
        if finish_reason == "stop" and stop_reason == stop_sequence:
            text = text + stop_sequence
    return text


def evaluate_countdown(
    llm,
    prompts: list[str],
    ground_truths: list[dict[str, Any]],
    max_new_tokens: int,
    stop_sequence: str | None,
    max_examples: int | None = None,
    prefix: str = "eval",
) -> dict[str, float]:
    if max_examples is not None and max_examples > 0:
        prompts = prompts[:max_examples]
        ground_truths = ground_truths[:max_examples]

    outputs = llm.generate(
        prompts,
        make_sampling_params(
            temperature=0.0,
            max_tokens=max_new_tokens,
            stop_sequence=stop_sequence,
        ),
    )

    rewards = []
    format_rewards = []
    answer_rewards = []
    for output, ground_truth in zip(outputs, ground_truths):
        reward = countdown_reward_fn(get_output_text(output, stop_sequence), ground_truth)
        rewards.append(reward["reward"])
        format_rewards.append(reward["format_reward"])
        answer_rewards.append(reward["answer_reward"])

    num_examples = len(rewards)
    return {
        f"{prefix}_accuracy": float(sum(rewards) / num_examples) if num_examples else 0.0,
        f"{prefix}_format_rate": float(sum(format_rewards) / num_examples) if num_examples else 0.0,
        f"{prefix}_answer_rate": float(sum(answer_rewards) / num_examples) if num_examples else 0.0,
        f"{prefix}_num_examples": num_examples,
    }


def sample_rollout_examples(
    dataset,
    num_prompts: int,
    prompt_template: str,
    rng: random.Random,
) -> tuple[list[str], list[dict[str, Any]]]:
    if num_prompts <= len(dataset):
        indices = rng.sample(range(len(dataset)), num_prompts)
    else:
        indices = [rng.randrange(len(dataset)) for _ in range(num_prompts)]

    prompts = []
    ground_truths = []
    for idx in indices:
        example = dataset[int(idx)]
        prompts.append(format_countdown_prompt(example, prompt_template))
        ground_truths.append(example["reward_model"]["ground_truth"])
    return prompts, ground_truths


def build_rollout_examples(
    repeated_prompts: list[str],
    repeated_ground_truths: list[dict[str, Any]],
    rollout_responses: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    examples = []
    for idx, (prompt, ground_truth, response) in enumerate(
        zip(repeated_prompts, repeated_ground_truths, rollout_responses)
    ):
        if idx >= limit:
            break
        reward = countdown_reward_fn(response, ground_truth)
        examples.append(
            {
                "index": idx,
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                "reward": reward,
            }
        )
    return examples


def score_old_log_probs(
    policy,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, input_ids.shape[0], batch_size):
            end = start + batch_size
            outputs = get_response_log_probs(
                model=policy,
                input_ids=input_ids[start:end].to(device),
                labels=labels[start:end].to(device),
                return_token_entropy=False,
            )
            chunks.append(outputs["log_probs"].cpu())
    return torch.cat(chunks, dim=0)


def main() -> None:
    from transformers import AutoModelForCausalLM

    args = parse_args()
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if args.train_batch_size <= 0 or args.microbatch_size <= 0:
        raise ValueError("train_batch_size and microbatch_size must be positive")
    if args.train_batch_size % args.microbatch_size != 0:
        raise ValueError("train_batch_size must be divisible by microbatch_size")
    if args.loss_type == "grpo_clip" and args.epochs_per_rollout_batch < 1:
        raise ValueError("epochs_per_rollout_batch must be positive")
    if args.enable_autopsy_recorder and args.autopsy_every <= 0:
        raise ValueError("autopsy_every must be positive when autopsy recorder is enabled")
    if args.enable_autopsy_recorder and args.autopsy_num_probe_prompts <= 0:
        raise ValueError("autopsy_num_probe_prompts must be positive")
    if args.enable_autopsy_recorder and args.autopsy_logprob_batch_size <= 0:
        raise ValueError("autopsy_logprob_batch_size must be positive")

    set_seed(args.seed)
    rng = random.Random(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    best_dir = ckpt_dir / "best"
    last_dir = ckpt_dir / "last"
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = output_dir / "hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    save_json(output_dir / "config.json", config)

    prompt_path = resolve_repo_path(args.prompt_path)
    train_path = ensure_repo_data_path(args.countdown_train_path)
    dev_path = ensure_repo_data_path(args.countdown_dev_path)
    test_path = ensure_repo_data_path(args.countdown_test_path)

    prompt_template = load_countdown_prompt(prompt_path)
    train_ds = load_countdown_split(train_path, cache_dir=hf_cache_dir)
    dev_ds = load_countdown_split(dev_path, cache_dir=hf_cache_dir)
    test_ds = load_countdown_split(test_path, cache_dir=hf_cache_dir)

    dev_prompts, dev_ground_truths = prepare_countdown_eval(
        dev_ds,
        prompt_template,
        max_examples=args.countdown_dev_max_examples,
    )
    test_prompts, test_ground_truths = prepare_countdown_eval(
        test_ds,
        prompt_template,
        max_examples=args.countdown_test_max_examples,
    )

    tokenizer = init_tokenizer(args.model_id)
    policy = init_policy(args.model_id, args.policy_device, args.gradient_checkpointing)
    llm = init_vllm(
        model_id=args.model_id,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)

    run = maybe_init_wandb(args, config)
    best_dev_accuracy = -float("inf")
    best_checkpoint_path = None
    last_dev_metrics: dict[str, float] | None = None
    train_step = 0
    eval_step = 0
    recorder: RolloutRecorder | None = None
    autopsy_group_size = args.autopsy_group_size if args.autopsy_group_size is not None else args.group_size
    if args.enable_autopsy_recorder:
        split_to_dataset = {"train": train_ds, "dev": dev_ds, "test": test_ds}
        probe_source = split_to_dataset[args.autopsy_probe_split]
        probe_set = build_fixed_countdown_probe_set(
            dataset=probe_source,
            prompt_template=prompt_template,
            num_prompts=args.autopsy_num_probe_prompts,
            seed=args.autopsy_probe_seed,
        )
        recorder = RolloutRecorder(
            output_dir=output_dir / "autopsy",
            probe_set=probe_set,
            group_size=autopsy_group_size,
            max_new_tokens=args.max_new_tokens,
            stop_sequence=args.stop_sequence,
            logprob_batch_size=args.autopsy_logprob_batch_size,
        )
        recorder.save_probe_manifest()

    def run_eval(tag: str, rollout_step: int) -> dict[str, float]:
        nonlocal eval_step, best_dev_accuracy, best_checkpoint_path, last_dev_metrics

        policy.eval()
        torch.cuda.empty_cache()
        load_policy_into_vllm_instance(policy, llm)

        metrics = evaluate_countdown(
            llm,
            dev_prompts,
            dev_ground_truths,
            max_new_tokens=args.max_new_tokens,
            stop_sequence=args.stop_sequence,
            max_examples=None,
            prefix="countdown_dev",
        )

        eval_step += 1
        log_record = {
            "tag": tag,
            "rollout_step": rollout_step,
            "train_step": train_step,
            "eval_step": eval_step,
            **metrics,
        }
        append_jsonl(output_dir / "eval_history.jsonl", log_record)

        maybe_log_wandb(
            run,
            {
                "eval_step": eval_step,
                "rollout_step": rollout_step,
                **{f"eval/{k}": v for k, v in metrics.items()},
            },
        )

        dev_accuracy = metrics["countdown_dev_accuracy"]
        last_dev_metrics = metrics
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_checkpoint_path = str(best_dir)
            policy.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_json(
                output_dir / "best_metric.json",
                {
                    "best_dev_accuracy": best_dev_accuracy,
                    "train_step": train_step,
                    "rollout_step": rollout_step,
                    "eval_step": eval_step,
                },
            )

        policy.train()
        return metrics

    if args.eval_before_train:
        run_eval(tag="before_train", rollout_step=0)

    num_prompts_per_rollout = args.rollout_batch_size // args.group_size
    for rollout_step in range(1, args.num_rollout_steps + 1):
        policy.eval()
        load_policy_into_vllm_instance(policy, llm)

        rollout_prompts, rollout_ground_truths = sample_rollout_examples(
            train_ds,
            num_prompts=num_prompts_per_rollout,
            prompt_template=prompt_template,
            rng=rng,
        )

        repeated_prompts = []
        repeated_ground_truths = []
        for prompt, ground_truth in zip(rollout_prompts, rollout_ground_truths):
            repeated_prompts.extend([prompt] * args.group_size)
            repeated_ground_truths.extend([ground_truth] * args.group_size)

        rollout_outputs = llm.generate(
            repeated_prompts,
            make_sampling_params(
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
                min_tokens=args.rollout_min_tokens,
                max_tokens=args.max_new_tokens,
                stop_sequence=args.stop_sequence,
            ),
        )
        rollout_responses = [
            get_output_text(output, args.stop_sequence) for output in rollout_outputs
        ]

        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=countdown_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.normalize_by_std,
        )

        tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        old_log_probs = None
        if args.loss_type == "grpo_clip":
            old_log_probs = score_old_log_probs(
                policy=policy,
                input_ids=tokenized["input_ids"],
                labels=tokenized["labels"],
                batch_size=args.old_logprob_batch_size,
                device=args.policy_device,
            )

        rollout_record = {
            "rollout_step": rollout_step,
            "train_step": train_step,
            "rollout/loss_type": args.loss_type,
            "rollout/length_normalization": args.length_normalization,
            "rollout/normalize_by_std": bool(args.normalize_by_std),
            "rollout/reward_mean": reward_metadata["reward_mean"],
            "rollout/reward_std": reward_metadata["reward_std"],
            "rollout/reward_min": reward_metadata["reward_min"],
            "rollout/reward_max": reward_metadata["reward_max"],
            "rollout/format_reward_mean": reward_metadata["format_reward_mean"],
            "rollout/answer_reward_mean": reward_metadata["answer_reward_mean"],
            "rollout/advantage_mean": float(advantages.mean().item()),
            "rollout/advantage_std": float(advantages.std().item()),
        }
        append_jsonl(output_dir / "rollout_history.jsonl", rollout_record)
        maybe_log_wandb(run, {"rollout_step": rollout_step, **rollout_record})
        append_jsonl(
            output_dir / "rollout_examples.jsonl",
            {
                "rollout_step": rollout_step,
                "examples": build_rollout_examples(
                    repeated_prompts,
                    repeated_ground_truths,
                    rollout_responses,
                    args.num_rollout_examples_to_log,
                ),
            },
        )
        if recorder is not None and rollout_step % args.autopsy_every == 0:
            torch.cuda.empty_cache()
            autopsy_metrics = recorder.record_step(
                step=rollout_step,
                llm=llm,
                policy=policy,
                tokenizer=tokenizer,
                sampling_params=make_sampling_params(
                    temperature=args.rollout_temperature,
                    top_p=args.rollout_top_p,
                    min_tokens=args.rollout_min_tokens,
                    max_tokens=args.max_new_tokens,
                    stop_sequence=args.stop_sequence,
                ),
                policy_device=args.policy_device,
            )
            append_jsonl(
                output_dir / "autopsy_history.jsonl",
                {
                    "rollout_step": rollout_step,
                    "train_step": train_step,
                    **autopsy_metrics,
                },
            )
            maybe_log_wandb(run, {"rollout_step": rollout_step, **autopsy_metrics})
            if args.autopsy_checkpoint_every > 0 and rollout_step % args.autopsy_checkpoint_every == 0:
                snapshot_dir = ckpt_dir / f"autopsy_step_{rollout_step:04d}"
                policy.save_pretrained(snapshot_dir)
                tokenizer.save_pretrained(snapshot_dir)

        policy.train()
        num_examples = tokenized["input_ids"].shape[0]
        for epoch_idx in range(args.epochs_per_rollout_batch):
            permutation = torch.randperm(num_examples)

            for batch_start in range(0, num_examples, args.train_batch_size):
                batch_indices = permutation[batch_start : batch_start + args.train_batch_size]
                microbatches = batch_indices.split(args.microbatch_size)
                optimizer.zero_grad(set_to_none=True)

                micro_loss_total = 0.0
                clipfrac_values = []

                for micro_indices in microbatches:
                    current_input_ids = tokenized["input_ids"][micro_indices].to(args.policy_device)
                    current_labels = tokenized["labels"][micro_indices].to(args.policy_device)
                    current_response_mask = tokenized["response_mask"][micro_indices].to(args.policy_device)

                    current_log_probs = get_response_log_probs(
                        model=policy,
                        input_ids=current_input_ids,
                        labels=current_labels,
                        return_token_entropy=False,
                    )["log_probs"]

                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=current_log_probs,
                        response_mask=current_response_mask,
                        gradient_accumulation_steps=len(microbatches),
                        loss_type=args.loss_type,
                        raw_rewards=raw_rewards[micro_indices].unsqueeze(-1).to(args.policy_device),
                        advantages=advantages[micro_indices].unsqueeze(-1).to(args.policy_device),
                        old_log_probs=(
                            old_log_probs[micro_indices].to(args.policy_device)
                            if old_log_probs is not None
                            else None
                        ),
                        cliprange=args.cliprange if args.loss_type == "grpo_clip" else None,
                        length_normalization=args.length_normalization,
                    )

                    micro_loss_total += float(loss.detach().cpu().item())
                    if "clipfrac" in metadata:
                        clipfrac_values.append(float(metadata["clipfrac"].detach().cpu().item()))

                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_step += 1

                train_record = {
                    "rollout_step": rollout_step,
                    "epoch_within_rollout": epoch_idx + 1,
                    "train_step": train_step,
                    "train/loss": micro_loss_total,
                    "train/loss_type": args.loss_type,
                    "train/length_normalization": args.length_normalization,
                    "train/normalize_by_std": bool(args.normalize_by_std),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                    "train/clipfrac": float(sum(clipfrac_values) / len(clipfrac_values)) if clipfrac_values else 0.0,
                    "train/reward_mean": reward_metadata["reward_mean"],
                    "train/answer_reward_mean": reward_metadata["answer_reward_mean"],
                    "train/advantage_mean": float(advantages.mean().item()),
                    "train/advantage_std": float(advantages.std().item()),
                }
                if "normalize_constant" in metadata:
                    train_record["train/normalize_constant"] = float(
                        metadata["normalize_constant"].detach().cpu().item()
                    )
                append_jsonl(output_dir / "train_history.jsonl", train_record)
                maybe_log_wandb(run, {"train_step": train_step, **train_record})

        policy.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)

        if rollout_step % args.eval_every == 0:
            run_eval(tag=f"rollout_{rollout_step}", rollout_step=rollout_step)

    if best_checkpoint_path is None:
        policy.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        best_checkpoint_path = str(best_dir)
        best_dev_accuracy = float("nan")

    del policy
    torch.cuda.empty_cache()

    best_policy = AutoModelForCausalLM.from_pretrained(
        best_checkpoint_path,
        torch_dtype=torch.bfloat16,
    ).to(args.policy_device)
    best_policy.config.use_cache = False
    best_policy.eval()
    load_policy_into_vllm_instance(best_policy, llm)

    final_metrics = evaluate_countdown(
        llm,
        test_prompts,
        test_ground_truths,
        max_new_tokens=args.max_new_tokens,
        stop_sequence=args.stop_sequence,
        max_examples=None,
        prefix="countdown_test",
    )
    save_json(
        output_dir / "summary.json",
        {
            "best_checkpoint_path": best_checkpoint_path,
            "best_dev_accuracy": best_dev_accuracy,
            "train_steps_completed": train_step,
            "rollout_steps_completed": args.num_rollout_steps,
            "final_dev_accuracy": (
                last_dev_metrics["countdown_dev_accuracy"] if last_dev_metrics is not None else None
            ),
            "final_dev_format_rate": (
                last_dev_metrics["countdown_dev_format_rate"] if last_dev_metrics is not None else None
            ),
            "final_dev_answer_rate": (
                last_dev_metrics["countdown_dev_answer_rate"] if last_dev_metrics is not None else None
            ),
            "final_dev_num_examples": (
                last_dev_metrics["countdown_dev_num_examples"] if last_dev_metrics is not None else None
            ),
            **final_metrics,
        },
    )

    maybe_log_wandb(run, {f"final/{k}": v for k, v in final_metrics.items()})
    if run is not None:
        maybe_log_wandb_output_artifact(run, output_dir, args)
        run.finish()


if __name__ == "__main__":
    main()
