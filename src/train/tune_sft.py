import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

from src.data_bootstrap import ensure_repo_data_path
from src.train.sft import get_response_log_probs, sft_microbatch_train_step, tokenize_prompt_and_output


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument(
        "--intellect-train-path",
        default=str(REPO_ROOT / "data-distrib" / "intellect_math" / "train"),
    )
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--max-optimizer-steps", type=int, default=50)
    parser.add_argument("--normalize-constant", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate-configs",
        default="1e-5:1:16,2e-5:1:32,5e-5:1:32,2e-5:2:16",
        help="Comma-separated lr:microbatch:gradaccum triples",
    )
    parser.add_argument("--target-loss-drop", type=float, default=0.40)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def format_intellect_example(ex: dict[str, Any]) -> dict[str, str]:
    msgs = ex.get("messages", [])
    sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    prompt = sys_msg + "\n\n" + user_msg if sys_msg else user_msg
    return {"prompt": prompt, "output": assistant_msg}


def load_intellect_split(path: str | Path, size: int | None, seed: int) -> Dataset:
    ds = load_from_disk(str(path)).shuffle(seed=seed)
    if size is not None and size > 0:
        ds = ds.select(range(min(size, len(ds))))
    return ds


def build_sft_examples(ds: Dataset) -> list[dict[str, str]]:
    return [format_intellect_example(ex) for ex in ds]


def make_sft_collate_fn(tokenizer):
    def collate_fn(examples: list[dict[str, str]]) -> dict[str, Any]:
        prompts = [ex["prompt"] for ex in examples]
        outputs = [ex["output"] for ex in examples]
        return tokenize_prompt_and_output(prompts, outputs, tokenizer)

    return collate_fn


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


def parse_candidate_configs(spec: str) -> list[dict[str, Any]]:
    configs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        lr_str, microbatch_str, grad_accum_str = item.split(":")
        configs.append(
            {
                "learning_rate": float(lr_str),
                "microbatch_size": int(microbatch_str),
                "gradient_accumulation_steps": int(grad_accum_str),
            }
        )
    if not configs:
        raise ValueError("At least one candidate config is required")
    return configs


def run_candidate(
    args: argparse.Namespace,
    tokenizer,
    train_examples: list[dict[str, str]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    candidate_seed = args.seed
    set_seed(candidate_seed)

    policy = init_policy(args.model_id, args.policy_device, args.gradient_checkpointing)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=candidate["learning_rate"],
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)

    train_loader = DataLoader(
        train_examples,
        batch_size=candidate["microbatch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=make_sft_collate_fn(tokenizer),
    )

    raw_losses: list[float] = []
    optimizer_step = 0
    stop_training = False

    try:
        for epoch in range(1000):
            accum_counter = 0
            accum_target = candidate["gradient_accumulation_steps"]

            for batch_idx, batch in enumerate(train_loader, start=1):
                if accum_counter == 0:
                    remaining_microbatches = len(train_loader) - batch_idx + 1
                    accum_target = min(candidate["gradient_accumulation_steps"], remaining_microbatches)

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

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=outputs["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=accum_target,
                    normalize_constant=args.normalize_constant,
                )

                raw_losses.append(float(loss.detach().cpu().item() * accum_target))

                if accum_counter == accum_target:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    accum_counter = 0

                    if optimizer_step >= args.max_optimizer_steps:
                        stop_training = True
                        break

            if stop_training:
                break

        if not raw_losses:
            raise RuntimeError("No training losses were recorded")

        initial_loss = raw_losses[0]
        best_loss = min(raw_losses)
        final_loss = raw_losses[-1]
        best_loss_drop = (initial_loss - best_loss) / initial_loss if initial_loss != 0 else 0.0
        final_loss_drop = (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0.0

        return {
            **candidate,
            "status": "ok",
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "best_loss_drop": best_loss_drop,
            "final_loss_drop": final_loss_drop,
            "optimizer_steps_completed": optimizer_step,
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return {
                **candidate,
                "status": "oom",
                "error": str(exc),
            }
        raise
    finally:
        del optimizer
        del policy
        torch.cuda.empty_cache()


def candidate_sort_key(result: dict[str, Any], target_loss_drop: float) -> tuple[float, float, float]:
    if result["status"] != "ok":
        return (1.0, float("inf"), float("inf"))

    gap_to_target = max(0.0, target_loss_drop - result["best_loss_drop"])
    return (
        gap_to_target,
        -result["best_loss_drop"],
        result["best_loss"],
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = ensure_repo_data_path(args.intellect_train_path)
    tokenizer = init_tokenizer(args.model_id)
    train_ds = load_intellect_split(train_path, args.train_size, args.seed)
    train_examples = build_sft_examples(train_ds)

    results = []
    for candidate in parse_candidate_configs(args.candidate_configs):
        result = run_candidate(args, tokenizer, train_examples, candidate)
        results.append(result)
        save_json(output_dir / "tuning_results.json", results)

    best_result = sorted(
        results,
        key=lambda result: candidate_sort_key(result, args.target_loss_drop),
    )[0]

    save_json(
        output_dir / "best_config.json",
        {
            "learning_rate": best_result["learning_rate"],
            "microbatch_size": best_result["microbatch_size"],
            "gradient_accumulation_steps": best_result["gradient_accumulation_steps"],
            "tuning_status": best_result["status"],
            "best_loss_drop": best_result.get("best_loss_drop"),
            "final_loss_drop": best_result.get("final_loss_drop"),
            "initial_loss": best_result.get("initial_loss"),
            "best_loss": best_result.get("best_loss"),
            "final_loss": best_result.get("final_loss"),
        },
    )


if __name__ == "__main__":
    main()
