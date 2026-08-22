
from __future__ import annotations

import keyword
import re

import numpy as np
import torch

from kd.core.expr.registry import FunctionRegistry
from kd.data.regression import TabularDataset
from kd.data.schema import DataTopology, FieldData, PDEDataset, TaskType

_DIFF_RESERVED_PATTERN = re.compile(r"^diff[0-9]*_[a-z]+$")
_UNITY_TOKEN = "one"


def dataset_from_tabular(tabular: TabularDataset) -> PDEDataset:
    x_values = np.asarray(tabular.X)
    y_values = np.asarray(tabular.y)
    names = list(tabular.var_names)

    _validate_names(names, tabular.target_name)
    if x_values.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {x_values.shape}")
    if len(names) != x_values.shape[1]:
        raise ValueError(
            "var_names length must match X width: "
            f"len(var_names)={len(names)}, X.shape[1]={x_values.shape[1]}"
        )
    if y_values.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y_values.shape}")
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError(
            "X and y must have matching lengths: "
            f"X.shape[0]={x_values.shape[0]}, y.shape[0]={y_values.shape[0]}"
        )
    if not np.all(np.isfinite(x_values)):
        raise ValueError("X must contain only finite values")
    if not np.all(np.isfinite(y_values)):
        raise ValueError("y must contain only finite values")

    fields = _build_fields(x_values, y_values, names, tabular.target_name)
    return PDEDataset(
        name=tabular.name,
        task_type=TaskType.REGRESSION,
        topology=DataTopology.TABULAR,
        axes=None,
        axis_order=None,
        fields=fields,
        lhs_field=tabular.target_name,
        lhs_axis="",
        lhs_order=0,
    )


def _build_fields(
    x_values: np.ndarray,
    y_values: np.ndarray,
    names: list[str],
    target_name: str,
) -> dict[str, FieldData]:
    fields = {
        name: FieldData(
            name=name,
            values=torch.as_tensor(x_values[:, index], dtype=torch.float64),
        )
        for index, name in enumerate(names)
    }
    fields[target_name] = FieldData(
        name=target_name,
        values=torch.as_tensor(y_values, dtype=torch.float64),
    )
    return fields


def _validate_names(var_names: list[str], target_name: str) -> None:
    all_names = [*var_names, target_name]
    reserved = set(FunctionRegistry.create_default().list_names()) | {_UNITY_TOKEN}
    for name in all_names:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"name must be a valid Python identifier, got {name!r}")
        if keyword.iskeyword(name):
            raise ValueError(f"name {name!r} is a Python keyword")
        if name in reserved or _DIFF_RESERVED_PATTERN.fullmatch(name):
            raise ValueError(f"name {name!r} collides with a reserved kd token")
    duplicates = sorted({name for name in all_names if all_names.count(name) > 1})
    if duplicates:
        if isinstance(target_name, str) and target_name in var_names:
            raise ValueError(
                f"target_name {target_name!r} must not appear in var_names"
            )
        raise ValueError(f"duplicate names are not allowed: {duplicates}")


__all__ = ["dataset_from_tabular"]
