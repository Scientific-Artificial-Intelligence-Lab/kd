
from __future__ import annotations

import pytest
import torch.nn as nn

from kd.search._torch_module_artifact import (
    TORCH_MODULE_ARTIFACT_FORMAT,
    torch_module_artifact,
)


class _TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)


def test_artifact_identity_is_stable_and_versioned() -> None:


    module = _TinyModule()
    first = torch_module_artifact(module)
    second = torch_module_artifact(module)
    assert first["format"] == TORCH_MODULE_ARTIFACT_FORMAT
    assert first == second
    assert len(first["sha256"]) == 64


@pytest.mark.parametrize("reserved_key", ["modules", "state"])
def test_item11_reserved_metadata_key_collision_raises(reserved_key: str) -> None:



    with pytest.raises(ValueError, match=reserved_key):
        torch_module_artifact(_TinyModule(), metadata={reserved_key: "shadow"})


def test_item11_non_colliding_metadata_is_accepted() -> None:

    identity = torch_module_artifact(_TinyModule(), metadata={"coord_names": ["x"]})
    assert identity["format"] == TORCH_MODULE_ARTIFACT_FORMAT
