"""Shared test fixtures for kd2."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Default test device."""
    return torch.device("cpu")
