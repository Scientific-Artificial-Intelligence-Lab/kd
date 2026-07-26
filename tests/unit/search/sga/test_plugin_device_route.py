
from __future__ import annotations

import pytest
import torch

from kd.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd.models.field_model import FieldModel
from kd.search.sga.config import SGAConfig
from kd.search.sga.plugin import SGAPlugin

_DTYPE = torch.float64
_NX = 8
_NT = 5

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)


def _tiny_dataset(device: torch.device) -> PDEDataset:
    x = torch.linspace(0.0, 6.0, _NX, dtype=_DTYPE, device=device)
    t = torch.linspace(0.0, 1.0, _NT, dtype=_DTYPE, device=device)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(gx) * torch.exp(-gt)
    return PDEDataset(
        name="sga-device-route",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _autograd_config(**overrides: object) -> SGAConfig:
    base: dict[str, object] = {
        "num": 4,
        "depth": 2,
        "width": 2,
        "seed": 42,
        "use_autograd": True,


        "autograd_train_epochs": 3,
    }
    base.update(overrides)
    return SGAConfig(**base)


def _field_shape(dataset: PDEDataset) -> tuple[int, ...]:
    assert dataset.fields is not None
    fd = next(iter(dataset.fields.values()))
    return tuple(fd.values.shape)


@pytest.mark.unit
@skip_no_cuda
def test_cuda_dataset_routes_provider_to_cuda() -> None:
    device = torch.device("cuda")
    dataset = _tiny_dataset(device)
    plugin = SGAPlugin(_autograd_config())

    provider = plugin._build_autograd_provider(dataset, _field_shape(dataset))



    assert next(provider.model.parameters()).device.type == "cuda"
    assert next(iter(provider.coords.values())).device.type == "cuda"
    assert provider.device.type == "cuda"


@pytest.mark.unit
def test_cpu_dataset_keeps_provider_on_cpu() -> None:
    device = torch.device("cpu")
    dataset = _tiny_dataset(device)
    plugin = SGAPlugin(_autograd_config())

    provider = plugin._build_autograd_provider(dataset, _field_shape(dataset))

    assert next(provider.model.parameters()).device.type == "cpu"
    assert next(iter(provider.coords.values())).device.type == "cpu"
    assert provider.device.type == "cpu"


@pytest.mark.unit
@skip_no_cuda
def test_pretrained_field_model_aligned_to_cuda() -> None:
    device = torch.device("cuda")
    dataset = _tiny_dataset(device)

    model = FieldModel(
        coord_names=["x", "t"],
        field_names=["u"],
        hidden_sizes=[4, 4],
    ).to(dtype=_DTYPE)
    model.eval()
    assert next(model.parameters()).device.type == "cpu"

    plugin = SGAPlugin(_autograd_config(field_model=model))
    provider = plugin._build_autograd_provider(dataset, _field_shape(dataset))

    assert next(provider.model.parameters()).device.type == "cuda"
    assert next(iter(provider.coords.values())).device.type == "cuda"
    assert provider.device.type == "cuda"
