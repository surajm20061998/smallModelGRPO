"""Minimal evaluation script for MATH and Intellect test sets."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from src.data_bootstrap import ensure_repo_data_path, resolve_repo_path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text()


def evaluate(llm, prompts, ground_truths):
    """Run evaluation and return accuracy."""
    from vllm import SamplingParams
    from src.grading.grader_math import question_only_reward_fn

    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    correct = 0
    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])
        correct += reward["reward"]

    return correct / len(outputs)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--prompt-path", default="configs/prompts/intellect.prompt")
    parser.add_argument("--intellect-path", default="data-distrib/intellect_math/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    from vllm import LLM

    prompt_template = load_prompt(resolve_repo_path(args.prompt_path))

    # Load model
    ensure_pynvml_compat()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Evaluate on Intellect test
    print(f"\n=== Intellect Test ({args.intellect_path}) ===")
    dataset = load_from_disk(str(ensure_repo_data_path(args.intellect_path)))
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    prompts, gts = [], []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))

    print(f"[Sample] {prompts[0][:200]}...")
    acc = evaluate(llm, prompts, gts)
    print(f"Intellect Accuracy: {acc:.4f}")

    # Evaluate on MATH
    print("\n=== MATH Test ===")
    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]

    print(f"[Sample] {prompts[0][:200]}...")
    acc = evaluate(llm, prompts, gts)
    print(f"MATH Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
