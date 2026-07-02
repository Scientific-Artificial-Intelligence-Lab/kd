
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kd.data.remote._loaders import (
    load_llm4ed_fisher,
    load_llm4ed_fisher_nonlinear,
    load_llm4ed_heat,
)
from kd.data.schema import PDEDataset
from kd.data.synthetic import (
    load_allen_cahn,
    load_burgers,
    load_burgers_2d,
    load_chafee_infante,
    load_convection_diffusion,
    load_eq_6_2_12,
    load_kdv,
    load_klein_gordon,
    load_pde_compound,
    load_pde_divide,
    load_wave,
)

DatasetLoader = Callable[[], PDEDataset]

_SCHEMA_VERSION = "1"
_BUILTIN_TIER = "builtin"
_AXES_XT = ("x", "t")
_LHS_UT = "u_t"

_SGA_PDE_SOURCE = "Chen et al. 2022 SGA-PDE benchmark data"
_EQGPT_SOURCE = "EqGPT benchmark data (Xu et al. 2025)"
_SGA_PDE_LICENSE = "MIT (SGA-PDE reference repository)"
_EQGPT_LICENSE = "EqGPT benchmark dataset"
_LLM4ED_REMOTE_SOURCE = (
    "LLM4ED (Du et al. 2024, arXiv:2405.07761; github.com/menggedu/EDL) "
    "via Spac1ly/KnowledgeDiscover; mirrored at timeoutHao/KD-data"
)
_LLM4ED_REMOTE_LICENSE = "MIT (Spac1ly/KnowledgeDiscover public dataset)"
_LLM4ED_REMOTE_REPO_ID = "timeoutHao/KD-data"
_LLM4ED_REMOTE_REVISION = "6c7dbe4f032e14fea2194644b09e74a4f254dd42"








@dataclass(frozen=True)
class DatasetSpec:

    id: str
    loader: DatasetLoader
    equation: str
    lhs: str
    axes: tuple[str, ...]
    fmt: str
    source: str
    license: str
    tier: str
    tags: tuple[str, ...] = ()
    schema_version: str = _SCHEMA_VERSION
    files: tuple[str, ...] = ()
    repo_id: str | None = None
    revision: str | None = None
    checksum: str | None = None


DATASET_CATALOG: dict[str, DatasetSpec] = {
    "allen-cahn": DatasetSpec(
        id="allen-cahn",
        loader=load_allen_cahn,
        equation="u_t = 0.003 * u_xx + u - u^3",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="mat",
        source=f"{_EQGPT_SOURCE}; bundled _assets/data/eqgpt_allen_cahn.mat",
        license=_EQGPT_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("eqgpt", "reaction-diffusion", "periodic-x"),
        files=("eqgpt_allen_cahn.mat",),
    ),
    "burgers": DatasetSpec(
        id="burgers",
        loader=load_burgers,
        equation="u_t = -u * u_x + 0.1 * u_xx",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="mat",
        source=_SGA_PDE_SOURCE,
        license=_SGA_PDE_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("sga-pde", "canonical"),
        files=("Burgers_equation.mat",),
    ),
    "burgers-2d": DatasetSpec(
        id="burgers-2d",
        loader=load_burgers_2d,
        equation="u_t = -u*u_x - u*u_y + 0.01*u_xx + 0.01*u_yy",
        lhs=_LHS_UT,
        axes=("x", "y", "t"),
        fmt="mat",
        source=f"{_EQGPT_SOURCE}; bundled _assets/data/eqgpt_burgers_2d.mat",
        license=_EQGPT_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("eqgpt", "2d-spatial"),
        files=("eqgpt_burgers_2d.mat",),
    ),
    "chafee-infante": DatasetSpec(
        id="chafee-infante",
        loader=load_chafee_infante,
        equation="u_t = u_xx - u + u^3",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="npy",
        source=_SGA_PDE_SOURCE,
        license=_SGA_PDE_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("canonical", "reaction-diffusion"),
        files=(
            "chafee_infante_CI.npy",
            "chafee_infante_x.npy",
            "chafee_infante_t.npy",
        ),
    ),
    "convection-diffusion": DatasetSpec(
        id="convection-diffusion",
        loader=load_convection_diffusion,
        equation="u_t = -u_x + 0.25 * u_xx",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="mat",
        source=(
            f"{_EQGPT_SOURCE}; bundled "
            "_assets/data/eqgpt_convection_diffusion.mat"
        ),
        license=_EQGPT_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("eqgpt", "convection-diffusion"),
        files=("eqgpt_convection_diffusion.mat",),
    ),
    "eq-6-2-12": DatasetSpec(
        id="eq-6-2-12",
        loader=load_eq_6_2_12,
        equation="u_t = -0.1*u_x_t - 0.1*u_x",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="csv",
        source=f"{_EQGPT_SOURCE}; bundled _assets/data/eqgpt_eq_6_2_12.csv",
        license=_EQGPT_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("eqgpt", "mixed-derivative"),
        files=("eqgpt_eq_6_2_12.csv",),
    ),
    "kdv": DatasetSpec(
        id="kdv",
        loader=load_kdv,
        equation="u_t = -u * u_x - 0.0025 * u_xxx",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="mat",
        source=_SGA_PDE_SOURCE,
        license=_SGA_PDE_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("sga-pde", "canonical", "third-order"),
        files=("KdV_equation.mat",),
    ),
    "klein-gordon": DatasetSpec(
        id="klein-gordon",
        loader=load_klein_gordon,
        equation="u_tt = 0.5 * u_xx - 5 * u",
        lhs="u_tt",
        axes=_AXES_XT,
        fmt="mat",
        source=f"{_EQGPT_SOURCE}; bundled _assets/data/eqgpt_klein_gordon.mat",
        license=_EQGPT_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("eqgpt", "second-order", "klein-gordon"),
        files=("eqgpt_klein_gordon.mat",),
    ),
    "llm4ed-fisher": DatasetSpec(
        id="llm4ed-fisher",
        loader=load_llm4ed_fisher,
        equation="u_t = 0.02*u_xx + 10*u*(1-u)",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="mat",
        source=_LLM4ED_REMOTE_SOURCE,
        license=_LLM4ED_REMOTE_LICENSE,
        tier="remote",
        tags=("llm4ed", "fisher", "reaction-diffusion", "remote"),
        files=("llm4ed/fisher_groundtruth.mat",),
        repo_id=_LLM4ED_REMOTE_REPO_ID,
        revision=_LLM4ED_REMOTE_REVISION,
        checksum="884e0b06d5bfcc586db8f38a0a5fb8de417e7be25154342a534c0296353ca257",
    ),
    "llm4ed-fisher-nonlinear": DatasetSpec(
        id="llm4ed-fisher-nonlinear",
        loader=load_llm4ed_fisher_nonlinear,
        equation="u_t = 0.02*(u*u_xx + u_x^2) + 10*u*(1-u)",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="mat",
        source=_LLM4ED_REMOTE_SOURCE,
        license=_LLM4ED_REMOTE_LICENSE,
        tier="remote",
        tags=("llm4ed", "fisher", "nonlinear-diffusion", "remote"),
        files=("llm4ed/fisher_nonlin_groundtruth.mat",),
        repo_id=_LLM4ED_REMOTE_REPO_ID,
        revision=_LLM4ED_REMOTE_REVISION,
        checksum="c8a8eb8024e5a18b8a08a495a5485454fcd5e1dbda58a014524f3c7b111927db",
    ),
    "llm4ed-heat": DatasetSpec(
        id="llm4ed-heat",
        loader=load_llm4ed_heat,
        equation="u_t = 0.05*u_xx",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="mat",
        source=_LLM4ED_REMOTE_SOURCE,
        license=_LLM4ED_REMOTE_LICENSE,
        tier="remote",
        tags=("llm4ed", "heat", "periodic-x", "remote"),
        files=("llm4ed/Heat_equation.mat",),
        repo_id=_LLM4ED_REMOTE_REPO_ID,
        revision=_LLM4ED_REMOTE_REVISION,
        checksum="ae4f570537a2bbf0c32de96e26f7c22b22bf16ef283e63cfa8d07b21c405dff8",
    ),
    "pde-compound": DatasetSpec(
        id="pde-compound",
        loader=load_pde_compound,
        equation="u_t = u * u_xx + u_x^2",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="npy",
        source=(
            f"{_SGA_PDE_SOURCE}; not equivalent to EqGPT PDE_compound CSV "
            "(different coefficient/grid)"
        ),
        license=_SGA_PDE_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("sga-pde", "compound", "open-form"),
        files=("PDE_compound.npy",),
    ),
    "pde-divide": DatasetSpec(
        id="pde-divide",
        loader=load_pde_divide,
        equation="u_t = -u_x / x + 0.25 * u_xx",
        lhs=_LHS_UT,
        axes=_AXES_XT,
        fmt="npy",
        source=_SGA_PDE_SOURCE,
        license=_SGA_PDE_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("sga-pde", "fractional", "open-form"),
        files=("PDE_divide.npy",),
    ),
    "wave": DatasetSpec(
        id="wave",
        loader=load_wave,
        equation="u_tt = u_xx",
        lhs="u_tt",
        axes=_AXES_XT,
        fmt="mat",
        source=f"{_EQGPT_SOURCE}; bundled _assets/data/eqgpt_wave.mat",
        license=_EQGPT_LICENSE,
        tier=_BUILTIN_TIER,
        tags=("eqgpt", "second-order", "wave"),
        files=("eqgpt_wave.mat",),
    ),
}


def list_datasets() -> list[DatasetSpec]:
    return [DATASET_CATALOG[dataset_id] for dataset_id in sorted(DATASET_CATALOG)]


def get_dataset(dataset_id: str) -> DatasetSpec:
    try:
        return DATASET_CATALOG[dataset_id]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_CATALOG))
        raise KeyError(
            f"unknown dataset id {dataset_id!r}; available datasets: {available}"
        ) from exc
