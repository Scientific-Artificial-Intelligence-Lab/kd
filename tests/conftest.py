"""Shared test fixtures for kd2."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Default test device - auto-detects best available: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
