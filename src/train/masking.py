import torch


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    masked_tensor = tensor * mask

    if dim is None:
        return masked_tensor.sum() / mask.sum()

    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    masked_tensor = tensor * mask

    if dim is None:
        return masked_tensor.sum() / normalize_constant

    return masked_tensor.sum(dim=dim) / normalize_constant
