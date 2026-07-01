
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kd.data.remote._hf_client import fetch_hub_file
from kd.data.schema import PDEDataset

if TYPE_CHECKING:
    from kd.data.catalog import DatasetSpec

_AXIS_X = "x"
_AXIS_T = "t"
_FIELD_U = "u"
_LHS_UT = "u_t"

_LLM4ED_HEAT_ID = "llm4ed-heat"
_LLM4ED_FISHER_ID = "llm4ed-fisher"
_LLM4ED_FISHER_NONLINEAR_ID = "llm4ed-fisher-nonlinear"

_HEAT_X_KEY = "x"
_HEAT_T_KEY = "t"
_HEAT_U_KEY = "usol"

_FISHER_X_KEY = "x"
_FISHER_T_KEY = "t"
_FISHER_U_KEY = "U"


def _load_mat(path: Path) -> dict[str, object]:
    import scipy.io as sio

    return dict(sio.loadmat(str(path)))


def _remote_file(spec: DatasetSpec) -> str:
    if spec.repo_id is None:
        raise ValueError(f"{spec.id} is missing repo_id")
    if spec.revision is None:
        raise ValueError(f"{spec.id} is missing revision")
    if spec.checksum is None:
        raise ValueError(f"{spec.id} is missing checksum")
    if len(spec.files) != 1:
        raise ValueError(f"{spec.id} must reference exactly one remote file")
    return spec.files[0]


def _fetch_spec_file(
    spec: DatasetSpec,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> Path:
    filename = _remote_file(spec)
    return fetch_hub_file(
        spec.repo_id or "",
        filename,
        revision=spec.revision or "",
        cache_dir=cache_dir,
        expected_sha256=spec.checksum,
        offline=offline,
    )


def _load_llm4ed_heat_from_path(path: Path, spec: DatasetSpec) -> PDEDataset:
    mat_data = _load_mat(path)
    x_np = np.asarray(mat_data[_HEAT_X_KEY], dtype=np.float64).flatten()
    t_np = np.asarray(mat_data[_HEAT_T_KEY], dtype=np.float64).flatten()
    u_np = np.asarray(mat_data[_HEAT_U_KEY], dtype=np.float64)

    expected_shape = (len(x_np), len(t_np))
    if u_np.shape != expected_shape:
        raise ValueError(
            f"{spec.id} field '{_HEAT_U_KEY}' has shape {u_np.shape}; "
            f"expected {expected_shape}"
        )

    return PDEDataset.from_arrays(
        coords={_AXIS_X: x_np, _AXIS_T: t_np},
        fields={_FIELD_U: u_np},
        lhs=spec.lhs,
        periodic={_AXIS_X},
        name=spec.id,
        ground_truth=spec.equation,
    )


def _load_llm4ed_fisher_from_path(path: Path, spec: DatasetSpec) -> PDEDataset:
    mat_data = _load_mat(path)
    x_np = np.asarray(mat_data[_FISHER_X_KEY], dtype=np.float64).flatten()
    t_np = np.asarray(mat_data[_FISHER_T_KEY], dtype=np.float64).flatten()
    u_raw = np.asarray(mat_data[_FISHER_U_KEY], dtype=np.float64)

    raw_expected_shape = (len(t_np), len(x_np))
    if u_raw.shape != raw_expected_shape:
        raise ValueError(
            f"{spec.id} field '{_FISHER_U_KEY}' has shape {u_raw.shape}; "
            f"expected raw (t, x) shape {raw_expected_shape}"
        )
    u_np = np.ascontiguousarray(u_raw.T)

    return PDEDataset.from_arrays(
        coords={_AXIS_X: x_np, _AXIS_T: t_np},
        fields={_FIELD_U: u_np},
        lhs=spec.lhs,
        periodic=None,
        name=spec.id,
        ground_truth=spec.equation,
    )


def _load_remote_spec(
    spec: DatasetSpec,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> PDEDataset:
    path = _fetch_spec_file(spec, cache_dir=cache_dir, offline=offline)
    if spec.id == _LLM4ED_HEAT_ID:
        return _load_llm4ed_heat_from_path(path, spec)
    if spec.id in {_LLM4ED_FISHER_ID, _LLM4ED_FISHER_NONLINEAR_ID}:
        return _load_llm4ed_fisher_from_path(path, spec)
    raise KeyError(f"no remote loader registered for dataset id {spec.id!r}")


def load_llm4ed_heat(
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> PDEDataset:
    from kd.data.catalog import get_dataset

    return _load_remote_spec(
        get_dataset(_LLM4ED_HEAT_ID),
        cache_dir=cache_dir,
        offline=offline,
    )


def load_llm4ed_fisher(
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> PDEDataset:
    from kd.data.catalog import get_dataset

    return _load_remote_spec(
        get_dataset(_LLM4ED_FISHER_ID),
        cache_dir=cache_dir,
        offline=offline,
    )


def load_llm4ed_fisher_nonlinear(
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> PDEDataset:
    from kd.data.catalog import get_dataset

    return _load_remote_spec(
        get_dataset(_LLM4ED_FISHER_NONLINEAR_ID),
        cache_dir=cache_dir,
        offline=offline,
    )


def list_remote_datasets() -> list[DatasetSpec]:
    from kd.data.catalog import DATASET_CATALOG

    return [
        DATASET_CATALOG[dataset_id]
        for dataset_id in sorted(DATASET_CATALOG)
        if DATASET_CATALOG[dataset_id].tier == "remote"
    ]


def load_from_hub(
    dataset_id: str,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> PDEDataset:
    from kd.data.catalog import get_dataset

    spec = get_dataset(dataset_id)
    if spec.tier != "remote":
        raise ValueError(
            f"dataset id {dataset_id!r} is tier={spec.tier!r}; "
            "load_from_hub only loads remote datasets"
        )
    return _load_remote_spec(spec, cache_dir=cache_dir, offline=offline)


__all__ = [
    "list_remote_datasets",
    "load_from_hub",
    "load_llm4ed_fisher",
    "load_llm4ed_fisher_nonlinear",
    "load_llm4ed_heat",
]
