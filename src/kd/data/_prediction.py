
from __future__ import annotations

from dataclasses import replace

from torch import Tensor

from kd.data.schema import PDEDataset


def _predicted_dataset(dataset: PDEDataset, predicted: dict[str, Tensor]) -> PDEDataset:
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    fields = {
        name: replace(
            field,
            values=predicted[name].to(
                dtype=field.values.dtype, device=field.values.device
            ),
        )
        for name, field in dataset.fields.items()
    }
    axes = {
        name: replace(info, values=info.values.clone(), is_periodic=False)
        for name, info in dataset.axes.items()
    }
    return replace(
        dataset,
        axes=axes,
        axis_order=list(dataset.axis_order),
        fields=fields,
        ground_truth=None,
    )
