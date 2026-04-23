import random
from dataclasses import dataclass
from typing import Any

from src.grading.grader_countdown import format_countdown_prompt


@dataclass(frozen=True)
class ProbeExample:
    probe_id: str
    dataset_index: int
    prompt: str
    ground_truth: dict[str, Any]
    meta: dict[str, Any]


def _difficulty_bucket(example: dict[str, Any]) -> str:
    gt = example["reward_model"]["ground_truth"]
    numbers = gt["numbers"]
    target = abs(int(gt["target"]))

    if len(numbers) <= 3 and target <= 50:
        return "easy"
    if len(numbers) >= 4 and target >= 250:
        return "hard"
    return "medium"


def build_fixed_countdown_probe_set(
    dataset,
    prompt_template: str,
    num_prompts: int,
    seed: int,
) -> list[ProbeExample]:
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    rng = random.Random(seed)
    bucket_to_indices: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}
    for idx in range(len(dataset)):
        bucket = _difficulty_bucket(dataset[int(idx)])
        bucket_to_indices[bucket].append(idx)

    target_by_bucket = {
        "easy": max(1, int(round(num_prompts * 0.34))),
        "medium": max(1, int(round(num_prompts * 0.33))),
        "hard": max(1, int(round(num_prompts * 0.33))),
    }
    # Ensure exact total count.
    while sum(target_by_bucket.values()) < num_prompts:
        target_by_bucket["medium"] += 1
    while sum(target_by_bucket.values()) > num_prompts:
        for name in ("medium", "easy", "hard"):
            if target_by_bucket[name] > 1 and sum(target_by_bucket.values()) > num_prompts:
                target_by_bucket[name] -= 1

    selected_indices: list[int] = []
    remaining_pool = list(range(len(dataset)))

    for bucket_name in ("easy", "medium", "hard"):
        choices = list(bucket_to_indices[bucket_name])
        rng.shuffle(choices)
        take = min(target_by_bucket[bucket_name], len(choices))
        selected_indices.extend(choices[:take])

    if len(selected_indices) < num_prompts:
        selected_set = set(selected_indices)
        candidates = [idx for idx in remaining_pool if idx not in selected_set]
        rng.shuffle(candidates)
        selected_indices.extend(candidates[: num_prompts - len(selected_indices)])

    selected_indices = selected_indices[:num_prompts]
    rng.shuffle(selected_indices)

    probe_set: list[ProbeExample] = []
    for probe_idx, dataset_idx in enumerate(selected_indices):
        example = dataset[int(dataset_idx)]
        gt = example["reward_model"]["ground_truth"]
        prompt = format_countdown_prompt(example, prompt_template)
        probe_set.append(
            ProbeExample(
                probe_id=f"probe_{probe_idx:03d}",
                dataset_index=int(dataset_idx),
                prompt=prompt,
                ground_truth=gt,
                meta={
                    "difficulty_bucket": _difficulty_bucket(example),
                    "numbers_count": len(gt["numbers"]),
                    "target_abs": abs(int(gt["target"])),
                },
            )
        )
    return probe_set
