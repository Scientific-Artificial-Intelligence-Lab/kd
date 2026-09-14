
from __future__ import annotations

import functools
from pathlib import Path
from typing import Final

from kd.data.remote._hf_client import KD_HUB_REPO_ID, fetch_hub_file, fetch_hub_tree

EQGPT_HUB_REVISION: Final[str] = "e658100cfa3ab289aee83eb154798d8a3be445c0"
EQGPT_WEIGHTS_FILENAME: Final[str] = "PDEGPT_wave_breaking.pt"
EQGPT_WEIGHTS_SHA256: Final[str] = (
    "082b9d1032a25e401757b05e375d2b970967316f50450105fcda2160a83ac4d4"
)
_WEIGHTS_FILE: Final[str] = f"eqgpt/gpt_model/{EQGPT_WEIGHTS_FILENAME}"
_SURROGATE_SUBDIR: Final[str] = "eqgpt/model_save"


@functools.cache
def fetch_eqgpt_weights(*, offline: bool = False) -> Path:
    return fetch_hub_file(
        KD_HUB_REPO_ID,
        _WEIGHTS_FILE,
        revision=EQGPT_HUB_REVISION,
        expected_sha256=EQGPT_WEIGHTS_SHA256,
        offline=offline,
    )


@functools.cache
def fetch_wave_surrogate_tree() -> Path:
    return fetch_hub_tree(
        KD_HUB_REPO_ID, _SURROGATE_SUBDIR, revision=EQGPT_HUB_REVISION
    ).parent
