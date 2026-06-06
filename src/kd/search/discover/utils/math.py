
import torch


_EXP_CLAMP_MAX = 50.0


_DEFAULT_EPS = 1e-10


def safe_div(
    a: torch.Tensor, b: torch.Tensor, eps: float = _DEFAULT_EPS
) -> torch.Tensor:
    sign = torch.where(b >= 0, 1.0, -1.0)
    return a / (b + eps * sign)


def safe_exp(x: torch.Tensor, max_val: float = _EXP_CLAMP_MAX) -> torch.Tensor:
    return torch.exp(torch.clamp(x, max=max_val))
