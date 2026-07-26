
from __future__ import annotations

import pytest
import torch

import kd
from kd.api import Model
from kd.search.eqgpt.config import EqGPTConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)

_SGA = {
    "generations": 2,
    "population": 8,
    "depth": 3,
    "width": 4,
    "maxit": 3,
    "str_iters": 3,
    "d_tol": 0.5,
}




_EQGPT = EqGPTConfig(
    sparsity_alpha=0.02,
    seed=0,
    samples_per_epoch=8,
    top_k=2,
    finetune_steps=1,
    max_length=20,
)


@skip_no_cuda
def test_sga_autograd_gpu_data_route() -> None:
    dataset = kd.generate_burgers_data(
        nx=32, nt=16, nu=0.1, seed=0, device=torch.device("cuda")
    )
    model = Model(algorithm="sga", derivatives="autograd", device="cuda", **_SGA)
    model.fit(dataset)
    record = model.result_.run_record
    assert record is not None
    assert record.evidence.is_valid


@skip_no_cuda
def test_dlga_context_on_cuda_and_recovers() -> None:
    dataset = kd.generate_burgers_data(
        nx=32, nt=16, nu=0.1, seed=0, device=torch.device("cuda")
    )
    model = Model(algorithm="dlga", device="cuda", generations=2)

    comps = model._build_components(dataset)
    assert comps.context is not None
    assert comps.context.device.type == "cuda"


    model.fit(dataset)
    record = model.result_.run_record
    assert record is not None
    assert record.evidence.is_valid


@skip_no_cuda
def test_eqgpt_gpu_smoke_stays_green() -> None:
    dataset = kd.generate_burgers_data(
        nx=32, nt=16, nu=0.1, seed=0, device=torch.device("cuda")
    )
    model = Model(
        algorithm="eqgpt", config=_EQGPT, device="cuda", generations=1
    )
    try:
        model.fit(dataset)
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"eqgpt weights unavailable: {exc}")
    record = model.result_.run_record


    assert record is not None
    assert record.evidence.is_valid
