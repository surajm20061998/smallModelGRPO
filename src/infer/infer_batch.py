import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm



def load_prompt(prompt_path: str) -> str:
    return Path(prompt_path).read_text()

def make_math_prompts(dataset, prompt_template: str) -> tuple[list[str], list[str], list[str]]:
    problems = [ex["problem"] for ex in dataset]
    ground_truths = [ex["answer"] for ex in dataset]
    prompts = [prompt_template + "\n\n" + problem for problem in problems]
    return prompts, problems, ground_truths

def get_output_text(vllm_output) -> str:
    if not getattr(vllm_output, "outputs", None):
        return ""
    if len(vllm_output.outputs) == 0:
        return ""
    return vllm_output.outputs[0].text
    
def categorize_reward(reward: dict[str, float]) -> str:
    format_reward = reward["format_reward"]
    answer_reward = reward["answer_reward"]

    if format_reward == 1.0 and answer_reward == 1.0:
        return "correct_format_and_answer"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "formatted_but_wrong"
    if format_reward == 0.0 and answer_reward == 0.0:
        return "unformatted_or_unparseable"

    return "other"


def truncate(text: str, max_chars: int = 500) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>..."


def tail(text: str, max_chars: int = 500) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return "...<truncated>..." + text[-max_chars:]


def print_examples(records: list[dict], category: str, k: int) -> None:
    matches = [r for r in records if r["category"] == category][:k]
    print(f"\n===== {category} : showing {len(matches)} example(s) =====")

    for i, r in enumerate(matches, start=1):
        print(f"\n--- Example {i} ---")
        print(f"index: {r['index']}")
        print(f"ground_truth: {r['ground_truth']}")
        print(f"parsed_answer: {r['parsed_answer']}")
        print(f"contains_boxed: {r['contains_boxed']}")
        print(f"fast_reward: {r['reward_fast']}")
        print(f"slow_reward: {r['reward_slow']}")
        print(f"suspected_parser_issue: {r['suspected_parser_issue']}")
        print(f"problem: {truncate(r['problem'], 300)}")
        print(f"output_tail: {tail(r['output_text'], 700)}")
        
def print_parser_issue_examples(records: list[dict], k: int) -> None:
    matches = [r for r in records if r["suspected_parser_issue"]][:k]
    print(f"\n===== suspected_parser_issue : showing {len(matches)} example(s) =====")

    for i, r in enumerate(matches, start=1):
        print(f"\n--- Example {i} ---")
        print(f"index: {r['index']}")
        print(f"ground_truth: {r['ground_truth']}")
        print(f"parsed_answer: {r['parsed_answer']}")
        print(f"contains_boxed: {r['contains_boxed']}")
        print(f"fast_reward: {r['reward_fast']}")
        print(f"slow_reward: {r['reward_slow']}")
        print(f"problem: {truncate(r['problem'], 300)}")
        print(f"output_tail: {tail(r['output_text'], 700)}")


def save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--prompt-path", default="configs/prompts/intellect.prompt")
    parser.add_argument("--output-dir", default="zero_shot_math_analysis")
    parser.add_argument("--inspect-per-category", type=int, default=10)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    from src.grading.grader_math import extract_answer, question_only_reward_fn

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = load_prompt(args.prompt_path)

    print("Loading MATH dataset...")
    dataset = load_dataset("hiyouga/math12k", split=args.split)
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    prompts, problems, ground_truths = make_math_prompts(dataset, prompt_template)

    print(f"Loaded {len(prompts)} examples from split={args.split}")

    print("Initializing vLLM...")
    ensure_pynvml_compat()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )

    print("Generating outputs...")
    outputs = llm.generate(prompts, params)

    records = []
    for idx, (problem, prompt, gt, output) in enumerate(
        tqdm(zip(problems, prompts, ground_truths, outputs), total=len(prompts), desc="Scoring")
    ):
        text = get_output_text(output)
        parsed_answer = extract_answer(text)
        reward_fast = question_only_reward_fn(text, gt, fast=True)
        reward_slow = question_only_reward_fn(text, gt, fast=False)
        category = categorize_reward(reward_fast)

        record = {
            "index": idx,
            "problem": problem,
            "prompt": prompt,
            "ground_truth": gt,
            "output_text": text,
            "parsed_answer": parsed_answer,
            "contains_boxed": "\\boxed" in text,
            "reward_fast": reward_fast,
            "reward_slow": reward_slow,
            "category": category,
            "suspected_parser_issue": (
                reward_fast["reward"] == 0.0 and reward_slow["reward"] == 1.0
            ),
        }
        records.append(record)

    counts = Counter(r["category"] for r in records)
    n = len(records)

    num_correct = counts["correct_format_and_answer"]
    num_formatted_wrong = counts["formatted_but_wrong"]
    num_unformatted = counts["unformatted_or_unparseable"]
    num_parser_issues = sum(r["suspected_parser_issue"] for r in records)

    summary = {
        "model": args.model,
        "split": args.split,
        "num_examples": n,
        "counts": dict(counts),
        "accuracy": num_correct / n if n else 0.0,
        "format_success_rate": (num_correct + num_formatted_wrong) / n if n else 0.0,
        "unformatted_or_unparseable_rate": num_unformatted / n if n else 0.0,
        "suspected_parser_issue_count": num_parser_issues,
        "suspected_parser_issue_rate": num_parser_issues / n if n else 0.0,
    }

    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2))

    save_jsonl(output_dir / "records.jsonl", records)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print_examples(records, "correct_format_and_answer", args.inspect_per_category)
    print_examples(records, "formatted_but_wrong", args.inspect_per_category)
    print_examples(records, "unformatted_or_unparseable", args.inspect_per_category)
    print_parser_issue_examples(records, args.inspect_per_category)

    print(f"\nSaved records to: {output_dir / 'records.jsonl'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
