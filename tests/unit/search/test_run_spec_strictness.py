
from __future__ import annotations

from enum import IntEnum
from typing import Any, cast

import pytest

from kd.search.run_spec import ConfigCanonicalizationError, RunSpec, canonicalize_config


def _run_spec(**overrides: Any) -> RunSpec:
    values: dict[str, Any] = {
        "kd_version": "0.4.0",
        "config": {"algorithm": "sga"},
        "dataset_cache_fingerprint": "sha256:dataset",
    }
    values.update(overrides)
    return RunSpec(**values)


@pytest.mark.unit
def test_direct_constructor_rejects_non_object_artifact_root() -> None:
    with pytest.raises(ConfigCanonicalizationError, match=r"artifacts.*list"):
        _run_spec(artifacts=cast(Any, []))


@pytest.mark.unit
def test_config_rejects_scalar_subclasses_outside_exact_whitelist() -> None:

    class Mode(IntEnum):
        FAST = 1

    with pytest.raises(ConfigCanonicalizationError, match=r"config\.mode.*Mode"):
        canonicalize_config({"mode": Mode.FAST})
