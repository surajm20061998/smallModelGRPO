import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from .sft import get_response_log_probs, tokenize_prompt_and_output


class PackedSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


def _load_instruction_examples(dataset_path: str | os.PathLike) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        examples = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        raise ValueError("JSON dataset must contain a list of examples")

    raise ValueError(f"Unsupported dataset file format: {path}")


def _format_instruction_example(example: dict[str, Any]) -> str:
    prompt = example.get("prompt")
    response = example.get("response")
    if prompt is None or response is None:
        raise ValueError("Each example must contain 'prompt' and 'response' keys")

    prompt = str(prompt)
    response = str(response)
    separator = "\n\n" if prompt and not prompt.endswith(("\n", " ")) else ""
    return prompt + separator + response


def _tokenize_document(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=True)

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    if bos_token_id is not None and (len(token_ids) == 0 or token_ids[0] != bos_token_id):
        token_ids = [bos_token_id] + token_ids
    if eos_token_id is not None and (len(token_ids) == 0 or token_ids[-1] != eos_token_id):
        token_ids = token_ids + [eos_token_id]

    return token_ids


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    examples = _load_instruction_examples(dataset_path)
    if shuffle:
        import random

        examples = examples.copy()
        random.shuffle(examples)

    token_stream: list[int] = []
    for example in examples:
        document = _format_instruction_example(example)
        token_stream.extend(_tokenize_document(tokenizer, document))

    chunk_length = seq_length + 1
    packed_examples: list[dict[str, torch.Tensor]] = []

    for start in range(0, len(token_stream) - chunk_length + 1, chunk_length):
        chunk = token_stream[start : start + chunk_length]
        packed_examples.append(
            {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }
        )

    return PackedSFTDataset(packed_examples)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


_OPTION_RE = re.compile(r"\b([ABCD])\b")


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    if model_output is None:
        return None

    text = str(model_output).strip()
    if not text:
        return None

    normalized = text.upper()

    patterns = [
        r"ANSWER\s+IS\s*\(?([ABCD])\)?",
        r"CORRECT\s+ANSWER\s*[:\-]?\s*\(?([ABCD])\)?",
        r"OPTION\s*\(?([ABCD])\)?",
        r"CHOICE\s*\(?([ABCD])\)?",
        r"^\s*\(?([ABCD])\)?[\.\:\)]?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        if matches:
            return matches[-1]

    option_texts = mmlu_example.get("options", [])
    matched_letters = []
    for idx, option_text in enumerate(option_texts[:4]):
        if option_text and str(option_text).strip().lower() in text.lower():
            matched_letters.append(chr(ord("A") + idx))
    if len(set(matched_letters)) == 1:
        return matched_letters[0]

    letter_matches = _OPTION_RE.findall(normalized)
    if letter_matches:
        return letter_matches[-1]
    return None


_GSM8K_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    if model_output is None:
        return None

    matches = _GSM8K_NUMBER_RE.findall(str(model_output))
    if not matches:
        return None

    number = matches[-1].replace(",", "")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def _compute_sequence_log_prob(
    lm: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
) -> torch.Tensor:
    tokenized = tokenize_prompt_and_output([prompt], [response], tokenizer)
    device = next(lm.parameters()).device

    outputs = get_response_log_probs(
        model=lm,
        input_ids=tokenized["input_ids"].to(device),
        labels=tokenized["labels"].to(device),
        return_token_entropy=False,
    )
    response_mask = tokenized["response_mask"].to(device)
    return (outputs["log_probs"] * response_mask.to(outputs["log_probs"].dtype)).sum(dim=-1).squeeze(0)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    chosen_logp = _compute_sequence_log_prob(lm, tokenizer, prompt, response_chosen)
    rejected_logp = _compute_sequence_log_prob(lm, tokenizer, prompt, response_rejected)

    with torch.inference_mode():
        ref_chosen_logp = _compute_sequence_log_prob(lm_ref, tokenizer, prompt, response_chosen)
        ref_rejected_logp = _compute_sequence_log_prob(lm_ref, tokenizer, prompt, response_rejected)

    preference_logit = beta * (
        (chosen_logp - rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
    )
    return -F.logsigmoid(preference_logit)
