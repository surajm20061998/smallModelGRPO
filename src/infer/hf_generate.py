"""Local (non-CUDA) rollout backend.

Provides a drop-in replacement for the subset of vLLM used by the training
loops, so the same call sites work on Apple-silicon (MPS) or CPU where vLLM is
unavailable. ``HFGenerator.generate`` returns objects shaped like vLLM's
``RequestOutput`` (``out.outputs[0].text`` / ``.finish_reason`` / ``.stop_reason``)
so downstream code — including the autopsy recorder — needs no changes.

Because the generator holds a reference to the live policy model, weight
updates during training are reflected automatically; there is no separate
inference engine to sync weights into.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class HFSamplingParams:
    """vLLM-SamplingParams-compatible subset used by the training loops."""

    temperature: float = 1.0
    top_p: float = 1.0
    min_tokens: int = 0
    max_tokens: int = 512
    stop: list[str] | None = None
    include_stop_str_in_output: bool = False


@dataclass
class _HFCompletion:
    text: str
    finish_reason: str
    stop_reason: str | None


@dataclass
class _HFRequestOutput:
    outputs: list[_HFCompletion]


class HFGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = max(1, int(batch_size))

    @staticmethod
    def _apply_stop(
        text: str,
        stop: list[str] | None,
        include_stop_str_in_output: bool,
    ) -> tuple[str, str, str | None]:
        if stop:
            earliest_idx = -1
            matched = None
            for candidate in stop:
                idx = text.find(candidate)
                if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                    earliest_idx = idx
                    matched = candidate
            if matched is not None:
                if include_stop_str_in_output:
                    text = text[: earliest_idx + len(matched)]
                else:
                    text = text[:earliest_idx]
                return text, "stop", matched
        return text, "length", None

    @torch.inference_mode()
    def generate(self, prompts: list[str], sampling_params: Any) -> list[_HFRequestOutput]:
        temperature = float(getattr(sampling_params, "temperature", 1.0))
        top_p = float(getattr(sampling_params, "top_p", 1.0))
        max_new_tokens = int(getattr(sampling_params, "max_tokens", 512))
        min_new_tokens = int(getattr(sampling_params, "min_tokens", 0) or 0)
        stop = getattr(sampling_params, "stop", None)
        include_stop = bool(getattr(sampling_params, "include_stop_str_in_output", False))
        do_sample = temperature > 0.0

        tokenizer = self.tokenizer
        prev_use_cache = getattr(self.model.config, "use_cache", None)
        was_training = self.model.training

        self.model.config.use_cache = True
        self.model.eval()

        # Batch only equal-length prompts together so padding is never needed.
        # Padded mixed-length batches produce NaN logits in SDPA on the MPS
        # backend (sampled generation then crashes in torch.multinomial; greedy
        # decoding silently corrupts instead). Rollout groups repeat a single
        # prompt so this costs nothing there; mixed-length eval batches degrade
        # to per-length chunks.
        ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        by_length: dict[int, list[int]] = {}
        for idx, ids in enumerate(ids_list):
            by_length.setdefault(len(ids), []).append(idx)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if min_new_tokens > 0:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        if stop:
            gen_kwargs["stop_strings"] = list(stop)
            gen_kwargs["tokenizer"] = tokenizer

        results: list[_HFRequestOutput | None] = [None] * len(prompts)
        try:
            for indices in by_length.values():
                for start in range(0, len(indices), self.batch_size):
                    chunk = indices[start : start + self.batch_size]
                    input_ids = torch.tensor(
                        [ids_list[i] for i in chunk], dtype=torch.long
                    ).to(self.device)
                    attention_mask = torch.ones_like(input_ids)

                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_kwargs,
                    )
                    new_tokens = generated[:, input_ids.shape[1] :]
                    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                    for prompt_idx, text in zip(chunk, texts):
                        final_text, finish_reason, stop_reason = self._apply_stop(
                            text, stop, include_stop
                        )
                        results[prompt_idx] = _HFRequestOutput(
                            outputs=[
                                _HFCompletion(
                                    text=final_text,
                                    finish_reason=finish_reason,
                                    stop_reason=stop_reason,
                                )
                            ]
                        )
        finally:
            if prev_use_cache is not None:
                self.model.config.use_cache = prev_use_cache
            if was_training:
                self.model.train()

        return results  # type: ignore[return-value]
