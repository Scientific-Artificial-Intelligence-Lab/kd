
from __future__ import annotations

import torch

from kd.data.schema import FieldData, PDEDataset


def add_noise_tensor(u: torch.Tensor, level: float, seed: int) -> torch.Tensor:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return u + level * u.std() * torch.randn_like(u)


def add_noise_dataset(
    dataset: PDEDataset, level: float, seed: int,
) -> PDEDataset:
    assert dataset.fields is not None
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        noisy_fields: dict[str, FieldData] = {}
        for name, fd in dataset.fields.items():
            noisy = fd.values + level * fd.values.std() * torch.randn_like(
                fd.values,
            )
            noisy_fields[name] = FieldData(name=name, values=noisy)
    return PDEDataset(
        name=f"{dataset.name}_noisy",
        task_type=dataset.task_type,
        axes=dataset.axes,
        axis_order=dataset.axis_order,
        fields=noisy_fields,
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
    )


__all__ = ["add_noise_dataset", "add_noise_tensor"]
