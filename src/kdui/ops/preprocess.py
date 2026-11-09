
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import kd
from kd import PDEDataset, TabularDataset
from kd.data import savgol_smooth
from kd.models import DLGA_NN_1, FieldRecipe
from kdui.models import (
    FieldSlice,
    LowpassSmoothing,
    NetworkSmoothing,
    Preprocessing,
    PreprocessPreview,
    PreprocessPreviewCall,
    PreprocessPreviewRequest,
    SliceSpec,
    SpectrumCurve,
)
from kdui.ops.inspect import load_dataset, sample_field
from kdui.settings import Workspace

_View = Callable[["PDEDataset | TabularDataset", SliceSpec, str], FieldSlice]


def apply_preprocessing(
    dataset: PDEDataset | TabularDataset, preprocessing: Preprocessing
) -> PDEDataset | TabularDataset:
    smoothing = preprocessing.smoothing
    if smoothing is None:
        return dataset
    if not isinstance(dataset, PDEDataset):
        raise ValueError(
            f"{smoothing.method} smoothing applies to grid data; "
            "this dataset is a table."
        )
    if isinstance(smoothing, LowpassSmoothing):
        return kd.lowpass_smooth(
            dataset,
            modes=smoothing.modes,
            axes=smoothing.axes,
            time_window=smoothing.time_window,
            polyorder=smoothing.polyorder,
        )
    return kd.network_smooth(
        dataset, recipe=_recipe(smoothing), seed=smoothing.seed
    ).dataset


def _recipe(smoothing: NetworkSmoothing) -> FieldRecipe:
    if smoothing.recipe is None:
        return DLGA_NN_1
    keywords: dict[str, Any] = {**DLGA_NN_1.to_dict(), **smoothing.recipe}
    keywords["hidden_sizes"] = tuple(keywords["hidden_sizes"])
    return FieldRecipe(**keywords)


def preview_preprocessing(request: PreprocessPreviewCall) -> PreprocessPreview:
    body = request.request
    region = body.region
    warnings: list[str] = []
    before = load_dataset(
        Workspace(Path(request.workspace)), body.dataset_id, with_reference=False
    )
    after = _smoothed(before, body, warnings)

    def view(
        dataset: PDEDataset | TabularDataset, spec: SliceSpec, latex: str
    ) -> FieldSlice:
        return sample_field(
            dataset,
            spec,
            dataset_id=body.dataset_id,
            quantity_latex=latex,
            max_points=request.max_points,
            max_cells=request.max_cells,
        )

    lhs_ok = _lhs_previewable(before, body.derivatives, warnings)
    return PreprocessPreview(
        dataset_id=body.dataset_id,
        field=region.field,
        derivatives=body.derivatives,
        noise_estimate=_noise(before, region.field),
        noise_estimate_after=None if after is None else _noise(after, region.field),
        spectrum=_spectrum(before, region.field),
        field_before=view(before, region, region.field),
        field_after=None if after is None else view(after, region, region.field),
        lhs_before=(
            _lhs_view(before, region, body.derivatives, view) if lhs_ok else None
        ),
        lhs_after=(
            _lhs_view(after, region, body.derivatives, view)
            if lhs_ok and after is not None
            else None
        ),
        warnings=warnings,
    )


def _smoothed(
    dataset: PDEDataset | TabularDataset,
    body: PreprocessPreviewRequest,
    warnings: list[str],
) -> PDEDataset | TabularDataset | None:
    if isinstance(body.preprocessing.smoothing, NetworkSmoothing):
        warnings.append(
            "The network smoothing trains a network inside the run; it has no preview."
        )
        return None
    if body.preprocessing.smoothing is None and body.savgol_window == 0:
        return None
    smoothed = apply_preprocessing(dataset, body.preprocessing)
    if body.savgol_window > 0:
        if not isinstance(smoothed, PDEDataset):
            raise ValueError(
                "Savitzky-Golay smoothing applies to grid data; "
                "this dataset is a table."
            )
        smoothed = savgol_smooth(smoothed, body.savgol_window)
    return smoothed


def _lhs_previewable(
    dataset: PDEDataset | TabularDataset, derivatives: str, warnings: list[str]
) -> bool:
    if kd.derivative_provider(derivatives).surrogate:
        warnings.append(
            f"The {derivatives} derivatives come from a network trained inside the "
            "run; they have no preview."
        )
        return False
    if not isinstance(dataset, PDEDataset) or dataset.topology != kd.DataTopology.GRID:
        warnings.append("The left-hand-side derivative is previewed on grid data only.")
        return False
    return True


def _lhs_view(
    dataset: PDEDataset | TabularDataset,
    region: SliceSpec,
    derivatives: str,
    view: _View,
) -> FieldSlice:
    assert isinstance(dataset, PDEDataset)
    name = dataset.lhs_field
    values = kd.derivative_field(
        dataset, name, dataset.lhs_axis, dataset.lhs_order, derivatives=derivatives
    )
    derived = dataset.with_field(name, values)
    latex = f"{name}_{{{dataset.lhs_axis * dataset.lhs_order}}}"
    spec = SliceSpec(field=name, axes=region.axes, fixed=region.fixed)
    return view(derived, spec, latex)


def _noise(dataset: PDEDataset | TabularDataset, field: str) -> dict[str, float | None]:
    rows: Any = kd.preview_report(dataset).to_dict()["fields"] or []
    estimates: dict[str, dict[str, float | None]] = {
        row["name"]: row["noise_estimate"] for row in rows
    }
    return estimates[field]


def _spectrum(dataset: PDEDataset | TabularDataset, field: str) -> list[SpectrumCurve]:
    if not isinstance(dataset, PDEDataset) or dataset.topology != kd.DataTopology.GRID:
        return []
    return [
        SpectrumCurve(
            axis=axis,
            missed=[
                1.0 - float(kept) for kept in kd.lowpass_spectrum(dataset, field, axis)
            ],
        )
        for axis in dataset.axis_order or []
    ]
