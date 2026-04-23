from typing import Callable, Literal

import torch

from .masking import masked_mean, masked_normalize


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("rollout batch size must be divisible by group_size")

    raw_rewards = []
    format_rewards = []
    answer_rewards = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_output = reward_fn(response, ground_truth)
        raw_rewards.append(float(reward_output["reward"]))
        format_rewards.append(float(reward_output["format_reward"]))
        answer_rewards.append(float(reward_output["answer_reward"]))

    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
    grouped_rewards = raw_rewards_tensor.view(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)

    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        advantages = grouped_rewards - group_means

    metadata = {
        "reward_mean": raw_rewards_tensor.mean().item(),
        "reward_std": raw_rewards_tensor.std().item(),
        "reward_min": raw_rewards_tensor.min().item(),
        "reward_max": raw_rewards_tensor.max().item(),
        "format_reward_mean": float(sum(format_rewards) / len(format_rewards)),
        "answer_reward_mean": float(sum(answer_rewards) / len(answer_rewards)),
    }

    return advantages.reshape(-1), raw_rewards_tensor, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    raw_rewards_or_advantages = raw_rewards_or_advantages.to(
        device=policy_log_probs.device,
        dtype=policy_log_probs.dtype,
    )
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = advantages.to(
        device=policy_log_probs.device,
        dtype=policy_log_probs.dtype,
    )
    old_log_probs = old_log_probs.to(
        device=policy_log_probs.device,
        dtype=policy_log_probs.dtype,
    )

    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages

    objective = torch.minimum(unclipped_objective, clipped_objective)
    loss = -objective

    was_clipped = clipped_objective < unclipped_objective

    metadata = {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "unclipped_objective": unclipped_objective,
        "clipped_objective": clipped_objective,
        "was_clipped": was_clipped,
        "clipfrac": was_clipped.to(policy_log_probs.dtype).mean(),
    }
    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for no_baseline"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    if loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages is required for reinforce_with_baseline"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    if loss_type == "grpo_clip":
        assert advantages is not None, "advantages is required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs is required for grpo_clip"
        assert cliprange is not None, "cliprange is required for grpo_clip"
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    raise ValueError(f"Unknown loss_type: {loss_type}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_normalization: Literal["masked_mean", "masked_normalize"] = "masked_mean",
    normalize_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    if length_normalization == "masked_mean":
        per_example_loss = masked_mean(
            tensor=per_token_loss,
            mask=response_mask,
            dim=-1,
        )
    elif length_normalization == "masked_normalize":
        if normalize_constant is None:
            normalize_constant = float(response_mask.sum(dim=-1).max().item())
        per_example_loss = masked_normalize(
            tensor=per_token_loss,
            mask=response_mask,
            dim=-1,
            normalize_constant=normalize_constant,
        )
    else:
        raise ValueError(f"Unknown length_normalization: {length_normalization}")

    loss = per_example_loss.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = dict(metadata)
    metadata["per_example_loss"] = per_example_loss.detach()
    metadata["mean_loss"] = loss.detach()
    if normalize_constant is not None:
        metadata["normalize_constant"] = torch.tensor(
            normalize_constant,
            device=policy_log_probs.device,
            dtype=policy_log_probs.dtype,
        ).detach()
    return loss, metadata
