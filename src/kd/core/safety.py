
import torch
from torch import Tensor

from kd.core import safety_counters as _counters


def safe_div(a: Tensor, b: Tensor, eps: float = 1e-10) -> Tensor:
    _counters.record_div(b, eps)
    sign_b = torch.sign(b)
    sign_b = torch.where(sign_b == 0, torch.ones_like(sign_b), sign_b)
    return a / (b + eps * sign_b)


def safe_exp(x: Tensor, min_val: float = -50.0, max_val: float = 50.0) -> Tensor:
    _counters.record_exp(x, min_val, max_val)
    return torch.exp(torch.clamp(x, min=min_val, max=max_val))


def safe_log(x: Tensor, eps: float = 1e-10) -> Tensor:
    _counters.record_log(x, eps)
    return torch.log(torch.clamp(x.abs(), min=eps))
