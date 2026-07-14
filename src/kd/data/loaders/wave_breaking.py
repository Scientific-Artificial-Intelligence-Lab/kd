
from __future__ import annotations

import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

if TYPE_CHECKING:
    from kd.data.schema import PDEDataset






_GRAVITY: float = 9.81



_CASE_NAME_RE: re.Pattern[str] = re.compile(r"G(\d+)Tp(\d+)A(\d+)")



_HF_SUBPATH: tuple[str, ...] = ("data", "hf-knowledgediscover", "WaveBreaking.pkl")


_MISSING_HINT: str = (
    "wave-breaking pickle not found. Fetch WaveBreaking.pkl from the "
    "KnowledgeDiscover HF dataset (Spac1ly/KnowledgeDiscover) and place it at "
    "data/hf-knowledgediscover/WaveBreaking.pkl, or pass an explicit path."
)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[4]


def default_wave_pkl_path() -> Path:
    return _find_repo_root().joinpath(*_HF_SUBPATH)










_V1_WAVE_ASSETS_ENV: str = "KD_V1_WAVE_ASSETS"

_MISSING_V1_ASSETS_HINT: str = (
    "v1 wave surrogate asset tree not found. Pass an explicit v1_asset_dir "
    "(EqGPTConfig.v1_asset_dir) or set the KD_V1_WAVE_ASSETS environment "
    "variable to the EqGPT_wave_breaking directory (it must contain "
    "model_save/wave_breaking/95_0_<case>(Non_unit)/Net_Sin_*.pkl)."
)


def resolve_v1_wave_asset_dir(asset_dir: str | Path | None = None) -> Path:
    if asset_dir is not None:
        return Path(asset_dir)
    env = os.environ.get(_V1_WAVE_ASSETS_ENV)
    if env:
        return Path(env)
    raise FileNotFoundError(_MISSING_V1_ASSETS_HINT)


def wave_surrogate_checkpoint_path(
    case_name: str,
    asset_dir: str | Path | None = None,
) -> Path:
    root = resolve_v1_wave_asset_dir(asset_dir)
    case_dir = root / "model_save" / "wave_breaking" / f"95_0_{case_name}(Non_unit)"
    pkls = sorted(case_dir.glob("Net_Sin_*.pkl"))
    if len(pkls) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Net_Sin_*.pkl under {case_dir}, "
            f"found {len(pkls)}. {_MISSING_V1_ASSETS_HINT}"
        )
    return pkls[0]







@dataclass
class WaveBreakingCase:

    name: str
    t: Tensor
    x: Tensor
    eta: Tensor
    g: int
    tp_seconds: float
    a: int
    lamda: float
    prefix: str







def load_wave_breaking_cases(
    path: str | Path | None = None,
) -> dict[str, WaveBreakingCase]:
    pkl_path = default_wave_pkl_path() if path is None else Path(path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"{_MISSING_HINT} Missing path: {pkl_path}")

    with pkl_path.open("rb") as fh:
        payload: object = pickle.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(
            f"WaveBreaking.pkl must contain a dict[str, ndarray], got "
            f"{type(payload).__name__}."
        )

    cases: dict[str, WaveBreakingCase] = {}
    for key, raw in payload.items():
        if not isinstance(key, str):
            raise ValueError(
                f"WaveBreaking.pkl case keys must be strings, got "
                f"{type(key).__name__}."
            )
        array = _validate_case_array(key, raw)
        cases[key] = _case_from_array(key, array)
    return cases


def _validate_case_array(name: str, raw: object) -> NDArray[np.float64]:
    if not isinstance(raw, np.ndarray):
        raise ValueError(
            f"case '{name}' must be a numpy.ndarray, got {type(raw).__name__}."
        )
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(
            f"case '{name}' must be a 2-D (N, 3) array, got shape {raw.shape}."
        )
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"case '{name}' must contain numeric values.")
    array = raw.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"case '{name}' contains NaN or Inf values.")
    return array


def _case_from_array(name: str, array: NDArray[np.float64]) -> WaveBreakingCase:
    g, tp_seconds, a, prefix = _parse_case_name(name)
    lamda = _GRAVITY * tp_seconds**2 / (2.0 * math.pi)
    return WaveBreakingCase(
        name=name,
        t=torch.tensor(array[:, 0], dtype=torch.float64),
        x=torch.tensor(array[:, 1], dtype=torch.float64),
        eta=torch.tensor(array[:, 2], dtype=torch.float64),
        g=g,
        tp_seconds=tp_seconds,
        a=a,
        lamda=lamda,
        prefix=prefix,
    )


def _parse_case_name(name: str) -> tuple[int, float, int, str]:
    prefix = name.split("_", maxsplit=1)[0]
    if prefix not in {"N", "L"}:
        raise ValueError(f"case '{name}' has unsupported prefix '{prefix}'.")
    match = _CASE_NAME_RE.search(name)
    if match is None:
        raise ValueError(f"case '{name}' does not match the G/Tp/A pattern.")
    g_raw, tp_raw, a_raw = match.groups()
    return int(g_raw), int(tp_raw) / 10.0, int(a_raw), prefix







def wave_dataset_name(case_name: str) -> str:
    return f"wave-breaking-{case_name}"


def wave_breaking_case_to_dataset(case: WaveBreakingCase) -> PDEDataset:
    from kd.data.schema import PDEDataset

    return PDEDataset.from_scatter(
        coords={"t": case.t, "x": case.x},
        fields={"u": case.eta},
        lhs="u_t",
        name=wave_dataset_name(case.name),
    )


__all__ = [
    "WaveBreakingCase",
    "default_wave_pkl_path",
    "load_wave_breaking_cases",
    "resolve_v1_wave_asset_dir",
    "wave_breaking_case_to_dataset",
    "wave_dataset_name",
    "wave_surrogate_checkpoint_path",
]
