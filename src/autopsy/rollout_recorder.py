import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.autopsy.probe_set import ProbeExample
from src.grading.grader_countdown import countdown_reward_fn
from src.train.sft import get_response_log_probs, tokenize_prompt_and_output


class RolloutRecorder:
    def __init__(
        self,
        output_dir: Path,
        probe_set: list[ProbeExample],
        group_size: int,
        max_new_tokens: int,
        stop_sequence: str | None,
        logprob_batch_size: int,
    ) -> None:
        self.output_dir = output_dir
        self.probe_set = probe_set
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.stop_sequence = stop_sequence
        self.logprob_batch_size = max(1, int(logprob_batch_size))

        self.rollouts_dir = output_dir / "rollouts"
        self.tensors_dir = output_dir / "tensors"
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)
        self.tensors_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_output_text(vllm_output, stop_sequence: str | None = None) -> str:
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

    @staticmethod
    def _response_token_ids_for_row(
        input_ids_row: torch.Tensor,
        labels_row: torch.Tensor,
        response_mask_row: torch.Tensor,
    ) -> list[int]:
        token_ids: list[int] = []
        if response_mask_row.shape[0] == 0:
            return token_ids

        first_is_response = bool(response_mask_row[0].item())
        if first_is_response:
            token_ids.append(int(input_ids_row[0].item()))

        for t in range(response_mask_row.shape[0]):
            if bool(response_mask_row[t].item()):
                token_ids.append(int(labels_row[t].item()))
        return token_ids

    def save_probe_manifest(self) -> None:
        manifest_path = self.output_dir / "probe_set.json"
        payload = [asdict(probe) for probe in self.probe_set]
        manifest_path.write_text(json.dumps(payload, indent=2))

    def _score_tokenized_with_backoff(
        self,
        *,
        policy,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        policy_device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch_size = self.logprob_batch_size
        last_error: Exception | None = None

        while batch_size >= 1:
            try:
                log_probs_chunks = []
                entropy_chunks = []
                with torch.inference_mode():
                    for start in range(0, input_ids.shape[0], batch_size):
                        end = min(start + batch_size, input_ids.shape[0])
                        scored = get_response_log_probs(
                            model=policy,
                            input_ids=input_ids[start:end].to(policy_device),
                            labels=labels[start:end].to(policy_device),
                            return_token_entropy=True,
                        )
                        log_probs_chunks.append(scored["log_probs"].detach().cpu())
                        entropy_chunks.append(scored["token_entropy"].detach().cpu())
                return (
                    torch.cat(log_probs_chunks, dim=0),
                    torch.cat(entropy_chunks, dim=0),
                    batch_size,
                )
            except torch.cuda.OutOfMemoryError as exc:
                last_error = exc
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                last_error = exc

            torch.cuda.empty_cache()
            if batch_size == 1:
                break
            batch_size = max(1, batch_size // 2)

        raise RuntimeError(
            "Autopsy recorder ran out of memory while scoring token log-probs/entropy. "
            "Try lowering --autopsy-logprob-batch-size and/or --autopsy-group-size."
        ) from last_error

    def record_step(
        self,
        *,
        step: int,
        llm,
        policy,
        tokenizer,
        sampling_params,
        policy_device: str,
    ) -> dict[str, Any]:
        repeated_prompts: list[str] = []
        repeated_ground_truths: list[dict[str, Any]] = []
        repeated_probe_ids: list[str] = []
        for probe in self.probe_set:
            repeated_prompts.extend([probe.prompt] * self.group_size)
            repeated_ground_truths.extend([probe.ground_truth] * self.group_size)
            repeated_probe_ids.extend([probe.probe_id] * self.group_size)

        outputs = llm.generate(repeated_prompts, sampling_params)
        responses = [
            self._get_output_text(output, stop_sequence=self.stop_sequence) for output in outputs
        ]
        rewards = [
            countdown_reward_fn(response, gt)
            for response, gt in zip(responses, repeated_ground_truths)
        ]

        tokenized = tokenize_prompt_and_output(repeated_prompts, responses, tokenizer)
        response_log_probs, token_entropy, used_batch_size = self._score_tokenized_with_backoff(
            policy=policy,
            input_ids=tokenized["input_ids"],
            labels=tokenized["labels"],
            policy_device=policy_device,
        )
        response_mask = tokenized["response_mask"].detach().cpu()
        input_ids = tokenized["input_ids"].detach().cpu()
        labels = tokenized["labels"].detach().cpu()

        step_dir = self.rollouts_dir / f"step_{step:04d}"
        tensor_step_dir = self.tensors_dir / f"step_{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        tensor_step_dir.mkdir(parents=True, exist_ok=True)

        probe_id_to_probe = {probe.probe_id: probe for probe in self.probe_set}
        records_by_probe: dict[str, list[dict[str, Any]]] = {probe.probe_id: [] for probe in self.probe_set}

        for idx, probe_id in enumerate(repeated_probe_ids):
            token_ids = self._response_token_ids_for_row(
                input_ids_row=input_ids[idx],
                labels_row=labels[idx],
                response_mask_row=response_mask[idx],
            )
            token_text = tokenizer.convert_ids_to_tokens(token_ids)
            token_lp = response_log_probs[idx][response_mask[idx]].tolist()
            token_ent = token_entropy[idx][response_mask[idx]].tolist()
            records_by_probe[probe_id].append(
                {
                    "rollout_index_within_probe": len(records_by_probe[probe_id]),
                    "response": responses[idx],
                    "reward": rewards[idx],
                    "response_tokens": token_text,
                    "response_token_ids": token_ids,
                    "response_log_probs": token_lp,
                    "response_entropies": token_ent,
                }
            )

        for probe in self.probe_set:
            prompt_record = {
                "step": step,
                "probe_id": probe.probe_id,
                "dataset_index": probe.dataset_index,
                "prompt": probe.prompt,
                "ground_truth": probe.ground_truth,
                "meta": probe.meta,
                "rollouts": records_by_probe[probe.probe_id],
            }
            (step_dir / f"{probe.probe_id}.json").write_text(json.dumps(prompt_record, indent=2))

        torch.save(
            {
                "response_log_probs": response_log_probs,
                "token_entropy": token_entropy,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "labels": labels,
                "repeated_probe_ids": repeated_probe_ids,
            },
            tensor_step_dir / "rollout_tensors.pt",
        )

        num_rollouts = len(responses)
        mean_reward = float(sum(item["reward"] for item in rewards) / num_rollouts) if num_rollouts else 0.0
        return {
            "autopsy/step": step,
            "autopsy/num_prompts": len(self.probe_set),
            "autopsy/group_size": self.group_size,
            "autopsy/num_rollouts": num_rollouts,
            "autopsy/mean_reward": mean_reward,
            "autopsy/logprob_batch_size_used": used_batch_size,
        }
