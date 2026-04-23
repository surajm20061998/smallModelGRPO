import argparse
import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset

try:
    import wandb
except Exception:
    wandb = None

from src.data_bootstrap import ensure_repo_data_path, resolve_repo_path

from src.train.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument(
        "--prompt-path",
        default=str(REPO_ROOT / "configs" / "prompts" / "intellect.prompt"),
    )

    parser.add_argument(
        "--intellect-train-path",
        default=str(REPO_ROOT / "data-distrib" / "intellect_math" / "train"),
    )
    parser.add_argument(
        "--intellect-dev-path",
        default=str(REPO_ROOT / "data-distrib" / "intellect_math" / "dev"),
    )
    parser.add_argument(
        "--intellect-test-path",
        default=str(REPO_ROOT / "data-distrib" / "intellect_math" / "test"),
    )

    parser.add_argument("--train-size", type=int, default=-1)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--max-optimizer-steps", type=int, default=300)

    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--normalize-constant", type=float, default=1.0)

    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)

    parser.add_argument("--intellect-val-max-examples", type=int, default=500)
    parser.add_argument("--math-val-split", default="train[-500:]")
    parser.add_argument("--math-test-split", default="test")
    parser.add_argument("--math-val-max-examples", type=int, default=500)
    parser.add_argument("--math-test-max-examples", type=int, default=500)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)

    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:1")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)

    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--selection-metric", default="math_val_accuracy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--wandb-project", default="smallModelGrpo")
    parser.add_argument("--wandb-entity", default="sm12377-new-york-university")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-output-artifact", dest="wandb_log_output_artifact", action="store_true")
    parser.add_argument("--no-wandb-log-output-artifact", dest="wandb_log_output_artifact", action="store_false")
    parser.set_defaults(wandb_log_output_artifact=True)

    parser.add_argument("--eval-before-train", action="store_true")
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


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text()


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


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
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


def format_intellect_example(ex: dict[str, Any]) -> dict[str, str]:
    msgs = ex.get("messages", [])
    sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    prompt = sys_msg + "\n\n" + user_msg if sys_msg else user_msg
    return {
        "prompt": prompt,
        "output": assistant_msg,
        "ground_truth": ex.get("ground_truth", ""),
    }


def load_intellect_split(path: str | Path, size: int | None, seed: int, shuffle: bool) -> Dataset:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if size is not None and size > 0:
        ds = ds.select(range(min(size, len(ds))))
    return ds


def build_sft_examples(ds: Dataset) -> list[dict[str, str]]:
    return [format_intellect_example(ex) for ex in ds]


def make_sft_collate_fn(tokenizer):
    def collate_fn(examples: list[dict[str, str]]) -> dict[str, Any]:
        prompts = [ex["prompt"] for ex in examples]
        outputs = [ex["output"] for ex in examples]
        batch = tokenize_prompt_and_output(prompts, outputs, tokenizer)
        batch["ground_truths"] = [ex["ground_truth"] for ex in examples]
        batch["prompts"] = prompts
        return batch
    return collate_fn


def prepare_intellect_eval(path: str | Path) -> tuple[list[str], list[str]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    prompts, gts = [], []
    for ex in ds:
        formatted = format_intellect_example(ex)
        prompts.append(formatted["prompt"])
        gts.append(formatted["ground_truth"])
    return prompts, gts


def prepare_math_eval(split: str, prompt_template: str) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    ds = load_dataset("hiyouga/math12k", split=split)
    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in ds]
    gts = [ex["answer"] for ex in ds]
    return prompts, gts


def compute_masked_average(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return ((values * mask_f).sum() / denom).detach().cpu().item()


def evaluate_prompts(
    llm,
    prompts: list[str],
    ground_truths: list[str],
    temperature: float,
    max_new_tokens: int,
    max_examples: int | None = None,
    prefix: str = "eval",
) -> dict[str, float]:
    from vllm import SamplingParams
    from src.grading.grader_math import question_only_reward_fn

    if max_examples is not None and max_examples > 0:
        prompts = prompts[:max_examples]
        ground_truths = ground_truths[:max_examples]

    params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)

    num = len(outputs)
    correct = 0.0
    format_success = 0.0
    formatted_but_wrong = 0
    unformatted = 0

    for output, gt in zip(outputs, ground_truths):
        text = output.outputs[0].text if output.outputs else ""
        reward = question_only_reward_fn(text, gt)
        correct += float(reward["reward"])
        format_success += float(reward["format_reward"])
        if reward["format_reward"] == 1.0 and reward["answer_reward"] == 0.0:
            formatted_but_wrong += 1
        if reward["format_reward"] == 0.0:
            unformatted += 1

    accuracy = correct / num if num else 0.0
    format_rate = format_success / num if num else 0.0

    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_format_rate": format_rate,
        f"{prefix}_formatted_but_wrong": formatted_but_wrong,
        f"{prefix}_unformatted": unformatted,
        f"{prefix}_num_examples": num,
    }


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
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    return run


def maybe_log_wandb(run, metrics: dict[str, Any]) -> None:
    if run is not None:
        run.log(metrics)


def maybe_log_wandb_output_artifact(run, output_dir: Path, args) -> None:
    if run is None or not args.wandb_log_output_artifact:
        return
    if wandb is None:
        return

    artifact_name = args.run_name or output_dir.name or "sft_run"
    artifact_name = artifact_name.replace("/", "-")
    artifact = wandb.Artifact(
        name=f"{artifact_name}-outputs",
        type="training-run-output",
        description="End-to-end SFT run outputs (metrics, checkpoints, summaries).",
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)


def main() -> None:
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    args = parse_args()
    set_seed(args.seed)

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

    config = vars(args).copy()
    save_json(output_dir / "config.json", config)

    prompt_path = resolve_repo_path(args.prompt_path)
    intellect_train_path = ensure_repo_data_path(args.intellect_train_path)
    intellect_dev_path = ensure_repo_data_path(args.intellect_dev_path)
    intellect_test_path = ensure_repo_data_path(args.intellect_test_path)

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

    prompt_template = load_prompt(prompt_path)

    train_size = None if args.train_size <= 0 else args.train_size
    train_ds = load_intellect_split(intellect_train_path, train_size, args.seed, shuffle=True)
    train_examples = build_sft_examples(train_ds)

    train_loader = DataLoader(
        train_examples,
        batch_size=args.microbatch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_sft_collate_fn(tokenizer),
    )

    intellect_dev_prompts, intellect_dev_gts = prepare_intellect_eval(intellect_dev_path)
    intellect_test_prompts, intellect_test_gts = prepare_intellect_eval(intellect_test_path)
    math_val_prompts, math_val_gts = prepare_math_eval(args.math_val_split, prompt_template)
    math_test_prompts, math_test_gts = prepare_math_eval(args.math_test_split, prompt_template)

    run = maybe_init_wandb(args, config)

    best_metric = -float("inf")
    best_checkpoint_path = None
    optimizer_step = 0
    micro_step = 0
    eval_step = 0

    def run_eval(tag: str) -> dict[str, float]:
        nonlocal eval_step, best_metric, best_checkpoint_path

        policy.eval()
        torch.cuda.empty_cache()
        load_policy_into_vllm_instance(policy, llm)

        metrics = {}
        metrics.update(
            evaluate_prompts(
                llm,
                intellect_dev_prompts,
                intellect_dev_gts,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                max_examples=args.intellect_val_max_examples,
                prefix="intellect_dev",
            )
        )
        metrics.update(
            evaluate_prompts(
                llm,
                math_val_prompts,
                math_val_gts,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                max_examples=args.math_val_max_examples,
                prefix="math_val",
            )
        )

        eval_step += 1
        log_record = {"tag": tag, "eval_step": eval_step, "train_step": optimizer_step, **metrics}
        append_jsonl(output_dir / "eval_history.jsonl", log_record)

        maybe_log_wandb(
            run,
            {"eval_step": eval_step, **{f"eval/{k}": v for k, v in metrics.items()}},
        )

        selected = metrics[args.selection_metric]
        if selected > best_metric:
            best_metric = selected
            best_checkpoint_path = str(best_dir)
            policy.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_json(
                output_dir / "best_metric.json",
                {
                    "selection_metric": args.selection_metric,
                    "best_metric": best_metric,
                    "train_step": optimizer_step,
                    "eval_step": eval_step,
                },
            )

        policy.train()
        return metrics

    if args.eval_before_train:
        run_eval(tag="before_train")

    stop_training = False
    for epoch in range(args.num_epochs):
        accum_counter = 0
        accum_target = args.gradient_accumulation_steps

        for batch_idx, batch in enumerate(train_loader, start=1):
            if accum_counter == 0:
                remaining_microbatches = len(train_loader) - batch_idx + 1
                accum_target = min(args.gradient_accumulation_steps, remaining_microbatches)

            micro_step += 1
            accum_counter += 1

            input_ids = batch["input_ids"].to(args.policy_device)
            labels = batch["labels"].to(args.policy_device)
            response_mask = batch["response_mask"].to(args.policy_device)

            outputs = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            policy_log_probs = outputs["log_probs"]
            mean_token_entropy = float("nan")

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=accum_target,
                normalize_constant=args.normalize_constant,
            )

            raw_loss = loss.detach().cpu().item() * accum_target
            mean_response_log_prob = compute_masked_average(policy_log_probs, response_mask)
            mean_token_entropy = float("nan")
            num_response_tokens = int(response_mask.sum().detach().cpu().item())

            if accum_counter == accum_target:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                accum_counter = 0

                train_record = {
                    "epoch": epoch,
                    "micro_step": micro_step,
                    "train_step": optimizer_step,
                    "raw_loss": raw_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                    "mean_response_log_prob": mean_response_log_prob,
                    "mean_token_entropy": mean_token_entropy,
                    "num_response_tokens": num_response_tokens,
                }
                append_jsonl(output_dir / "train_history.jsonl", train_record)

                maybe_log_wandb(
                    run,
                    {
                        "train_step": optimizer_step,
                        "train/raw_loss": raw_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": train_record["grad_norm"],
                        "train/mean_response_log_prob": mean_response_log_prob,
                        "train/mean_token_entropy": mean_token_entropy,
                        "train/num_response_tokens": num_response_tokens,
                    },
                )

                if optimizer_step % args.eval_every == 0:
                    run_eval(tag=f"step_{optimizer_step}")

                if optimizer_step % args.save_every == 0:
                    policy.save_pretrained(last_dir)
                    tokenizer.save_pretrained(last_dir)

                if optimizer_step >= args.max_optimizer_steps:
                    stop_training = True
                    break

        if stop_training:
            break

    policy.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)

    if best_checkpoint_path is None:
        policy.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        best_checkpoint_path = str(best_dir)

    del policy
    torch.cuda.empty_cache()

    best_policy = AutoModelForCausalLM.from_pretrained(
        best_checkpoint_path,
        torch_dtype=torch.bfloat16,
    ).to(args.policy_device)
    best_policy.config.use_cache = False
    best_policy.eval()

    load_policy_into_vllm_instance(best_policy, llm)

    final_metrics = {}
    final_metrics.update(
        evaluate_prompts(
            llm,
            intellect_test_prompts,
            intellect_test_gts,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_examples=None,
            prefix="intellect_test",
        )
    )
    final_metrics.update(
        evaluate_prompts(
            llm,
            math_test_prompts,
            math_test_gts,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_examples=args.math_test_max_examples,
            prefix="math_test",
        )
    )

    summary = {
        "train_size": len(train_examples),
        "best_checkpoint_path": best_checkpoint_path,
        "best_selection_metric": args.selection_metric,
        "best_selection_value": best_metric,
        "optimizer_steps_completed": optimizer_step,
        **final_metrics,
    }
    save_json(output_dir / "summary.json", summary)

    maybe_log_wandb(run, {f"final/{k}": v for k, v in final_metrics.items()})

    if run is not None:
        maybe_log_wandb_output_artifact(run, output_dir, args)
        run.finish()


if __name__ == "__main__":
    main()
