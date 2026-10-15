
from __future__ import annotations

import torch

from kd.data.schema import AxisInfo, FieldData, PDEDataset
from kd.data.subsample import _require_grid


def savgol_smooth(
    dataset: PDEDataset, window_length: int, *, polyorder: int = 3
) -> PDEDataset:

    from scipy.signal import savgol_filter

    _require_grid(dataset, what="savgol_smooth")
    assert dataset.axes is not None and dataset.axis_order is not None
    assert dataset.fields is not None
    axis_order = dataset.axis_order
    axes = {
        name: AxisInfo(info.name, info.values.clone(), info.is_periodic, info.allow_nan)
        for name, info in dataset.axes.items()
    }
    fields: dict[str, FieldData] = {}
    for name, field in dataset.fields.items():
        values = field.values.detach().cpu().numpy()
        for axis_index in range(len(axis_order)):
            values = savgol_filter(
                values,
                window_length=window_length,
                polyorder=polyorder,
                axis=axis_index,
                mode="interp",
            )
        smoothed = torch.as_tensor(
            values, dtype=field.values.dtype, device=field.values.device
        )
        fields[name] = FieldData(field.name, smoothed, allow_nan=field.allow_nan)
    return PDEDataset(
        name=dataset.name,
        task_type=dataset.task_type,
        topology=dataset.topology,
        axes=axes,
        axis_order=list(axis_order),
        fields=fields,
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
        lhs_order=dataset.lhs_order,
        noise_level=dataset.noise_level,
        ground_truth=dataset.ground_truth,
        source=dataset.source,
    )
