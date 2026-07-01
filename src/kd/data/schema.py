"""PDE dataset schema: data structures for PDE discovery.

This module defines the core data structures:
- TaskType: type of problem (PDE, ODE, regression)
- DataTopology: data layout (grid, scattered)
- AxisInfo: coordinate axis metadata
- FieldData: field values container
- PDEDataset: complete dataset specification

Design principles:
- n-dimensional support: no hardcoded axis names ("x", "t")
- torch.Tensor throughout, device-aware
- Grid topology supported today; Scattered reserved for NN-derivative sampling

Note on axis naming:
- Axis names can be arbitrary strings (e.g., "x", "time", "spatial")
- However, derivative symbols like "u_xx" require single-letter axis names
- For compatibility with expression parsing, prefer single-letter names: "x", "t", "y"
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from kd.data._factory import (
    annotate_shape_error,
    build_axes_dict,
    build_fields_dict,
    parse_lhs_spec,
)


_LARGE_DATA_THRESHOLD_BYTES = 10_000_000


_SAMPLING_STRIDE = 1000


class TaskType(Enum):
    """Type of discovery task."""

    PDE = "pde"
    ODE = "ode"
    REGRESSION = "regression"


class DataTopology(Enum):
    """Data layout topology."""

    GRID = "grid"
    SCATTERED = "scattered"


@dataclass
class AxisInfo:
    """Coordinate axis information.

    Attributes:
        name: User-defined axis name (e.g., "x", "t", "y")
        values: 1D tensor of coordinate values
        is_periodic: Whether this axis has periodic boundary conditions
    """

    name: str
    values: torch.Tensor
    is_periodic: bool = False

    def __post_init__(self) -> None:
        """Validate axis info."""
        if self.values.dim() != 1:
            raise ValueError(f"AxisInfo values must be 1D, got {self.values.dim()}D")
        if self.values.numel() == 0:
            raise ValueError("AxisInfo values must not be empty")
        if torch.isnan(self.values).any():
            raise ValueError("AxisInfo values must not contain NaN")
        if torch.isinf(self.values).any():
            raise ValueError("AxisInfo values must not contain Inf")


@dataclass
class FieldData:
    """Field data container.

    Attributes:
        name: Field name (e.g., "u", "v")
        values: nD tensor of field values, shape matches axis order
    """

    name: str
    values: torch.Tensor

    def __post_init__(self) -> None:
        """Validate field data."""
        if self.values.dim() < 1:
            raise ValueError(
                f"FieldData values must be at least 1D, got {self.values.dim()}D"
            )
        if self.values.numel() == 0:
            raise ValueError("FieldData values must not be empty")
        if torch.isnan(self.values).any():
            raise ValueError("FieldData values must not contain NaN")
        if torch.isinf(self.values).any():
            raise ValueError("FieldData values must not contain Inf")
        if not self.values.is_floating_point():
            raise TypeError(
                f"FieldData values must be floating-point, got {self.values.dtype}. "
                f"Use .float() or .double() to convert."
            )


@dataclass
class PDEDataset:
    """Complete PDE dataset specification.

    Attributes:
        name: Dataset identifier
        task_type: Type of problem (PDE, ODE, regression)
        topology: Data layout (grid or scattered)
        axes: Mapping from axis name to AxisInfo (Grid mode)
        axis_order: Ordered list of axis names defining tensor dimensions
        fields: Mapping from field name to FieldData
        lhs_field: Field for LHS of equation (e.g., "u")
        lhs_axis: Axis for time derivative on LHS (e.g., "t" for u_t = RHS)
        lhs_order: Order of the LHS derivative along ``lhs_axis`` (``1`` -> u_t,
            ``2`` -> u_tt for the wave/telegraph case). Default ``1``. This is
            the single source of truth for the LHS order across the whole
            pipeline (parser, fingerprint, platform builder, plugins).
        noise_level: Amount of noise added to data
        ground_truth: Optional ground truth equation string

    Example:
        >>> dataset = PDEDataset(
        ... name="burgers",
        ... task_type=TaskType.PDE,
        ... axes={"x": x_axis, "t": t_axis},
        ... axis_order=["x", "t"],
        ... fields={"u": u_field},
        ... lhs_field="u",
        ... lhs_axis="t",
        ... )
    """

    name: str
    task_type: TaskType
    topology: DataTopology = DataTopology.GRID


    axes: dict[str, AxisInfo] | None = None
    axis_order: list[str] | None = None
    fields: dict[str, FieldData] | None = None


    lhs_field: str = ""
    lhs_axis: str = ""
    lhs_order: int = 1


    noise_level: float = 0.0
    ground_truth: str | None = None

    def __post_init__(self) -> None:
        """Validate dataset consistency."""
        self._validate_axis_consistency()
        self._validate_field_shapes()
        self._validate_lhs()
        self._validate_lhs_order()

    def _validate_axis_consistency(self) -> None:
        """Validate that axis_order and axes are consistent."""
        if self.axis_order is None or self.axes is None:
            return


        for key, axis in self.axes.items():
            if key != axis.name:
                raise ValueError(
                    f"Axis key '{key}' does not match axis.name '{axis.name}'"
                )


        if len(self.axis_order) != len(set(self.axis_order)):
            raise ValueError(
                f"axis_order contains duplicate elements: {self.axis_order}"
            )

        axis_order_set = set(self.axis_order)
        axes_set = set(self.axes.keys())


        missing_in_axes = axis_order_set - axes_set
        if missing_in_axes:
            raise ValueError(f"axis_order contains axes not in axes: {missing_in_axes}")


        missing_in_order = axes_set - axis_order_set
        if missing_in_order:
            raise ValueError(
                f"axes contains axes not in axis_order: {missing_in_order}"
            )

    def _validate_field_shapes(self) -> None:
        """Validate that field shapes match axes."""
        if self.fields is None or self.axis_order is None or self.axes is None:
            return


        for key, field in self.fields.items():
            if key != field.name:
                raise ValueError(
                    f"Field key '{key}' does not match field.name '{field.name}'"
                )

        expected_shape = tuple(
            self.axes[axis_name].values.numel() for axis_name in self.axis_order
        )
        n_dims = len(self.axis_order)

        for field_name, field_data in self.fields.items():

            if field_data.values.dim() != n_dims:
                raise ValueError(
                    f"field '{field_name}' dimension mismatch: "
                    f"expected {n_dims}D, got {field_data.values.dim()}D"
                )


            if field_data.values.shape != expected_shape:
                raise ValueError(
                    f"field '{field_name}' shape mismatch: "
                    f"expected {expected_shape}, got {tuple(field_data.values.shape)}"
                )

    def _validate_lhs(self) -> None:
        """Validate that lhs_field and lhs_axis reference existing entries."""
        if (
            self.lhs_field != ""
            and self.fields is not None
            and self.lhs_field not in self.fields
        ):
            raise ValueError(
                f"lhs_field '{self.lhs_field}' not found in fields: "
                f"{list(self.fields.keys())}"
            )
        if (
            self.lhs_axis != ""
            and self.axis_order is not None
            and self.lhs_axis not in self.axis_order
        ):
            raise ValueError(
                f"lhs_axis '{self.lhs_axis}' not found in axis_order: {self.axis_order}"
            )

    def _validate_lhs_order(self) -> None:
        """Reject a non-positive LHS derivative order.

        ``lhs_order`` is the order of the LHS time/axis derivative
        (``1`` -> u_t, ``2`` -> u_tt). Order < 1 is meaningless: there is no
        zeroth-order "derivative" LHS in this pipeline (the LHS is always a
        derivative of the field), so it fails loud here rather than silently
        producing a bogus regression target downstream.
        """
        if self.lhs_order < 1:
            raise ValueError(
                f"lhs_order must be >= 1 (1 -> u_t, 2 -> u_tt), got {self.lhs_order}."
            )

    @property
    def spatial_axes(self) -> list[str]:
        """Spatial axes derived from ``axis_order`` minus ``lhs_axis``.

        Returns an empty list when ``axis_order`` is missing or ``lhs_axis``
        is unset. This keeps PDE-domain metadata in the dataset while allowing
        callers to handle non-PDE or under-specified datasets explicitly.
        """
        if self.axis_order is None or not self.lhs_axis:
            return []
        return [axis for axis in self.axis_order if axis != self.lhs_axis]

    def get_shape(self) -> tuple[int, ...]:
        """Return data shape as tuple.

        Returns:
            Tuple of dimensions in axis_order order.

        Raises:
            ValueError: If dataset is not properly configured.
        """
        if self.axis_order is None or self.axes is None:
            raise ValueError("Dataset not properly configured: missing axes/axis_order")

        return tuple(
            self.axes[axis_name].values.numel() for axis_name in self.axis_order
        )

    def get_coords(self, axis: str) -> torch.Tensor:
        """Get coordinate values for specified axis.

        Args:
            axis: Name of the axis to retrieve.

        Returns:
            1D tensor of coordinate values.

        Raises:
            KeyError: If axis not found.
        """
        if self.axes is None:
            raise KeyError(f"Axis '{axis}' not found: axes is None")
        if axis not in self.axes:
            raise KeyError(f"Axis '{axis}' not found in axes")
        return self.axes[axis].values

    def get_field(self, name: str) -> torch.Tensor:
        """Get field values by name.

        Args:
            name: Name of the field to retrieve.

        Returns:
            Tensor of field values.

        Raises:
            KeyError: If field not found.
        """
        if self.fields is None:
            raise KeyError(f"Field '{name}' not found: fields is None")
        if name not in self.fields:
            raise KeyError(f"Field '{name}' not found in fields")
        return self.fields[name].values

    @classmethod
    def from_arrays(
        cls,
        coords: dict[str, torch.Tensor | np.ndarray | Sequence[float]],
        fields: dict[str, torch.Tensor | np.ndarray],
        *,
        lhs: str = "u_t",
        periodic: Iterable[str] | None = None,
        name: str = "custom",
        ground_truth: str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> PDEDataset:
        """Factory: wrap raw arrays into a PDEDataset.

        The order of keys in ``coords`` determines axis_order (i.e., the
        expected shape of each field tensor).

        Args:
            coords: Mapping axis name to 1D coordinate tensor (or numpy/list).
                Insertion order defines axis_order.
            fields: Mapping field name to nD field tensor whose shape matches
                ``(len(coords[axis_0]), len(coords[axis_1]), ...)``.
            lhs: Combined LHS spec ``"{field}_{axis...}"`` encoding the LHS
                derivative via the kd naming convention. ``"u_t"`` -> field
                ``"u"``, axis ``"t"``, order ``1`` (du/dt); ``"u_tt"`` ->
                order ``2`` (the wave/telegraph LHS d²u/dt²). Parsed by
                ``naming.parse_derivative_name`` against the known field/axis
                names, so a multi-underscore field still resolves
                (``"my_field_t"`` -> field=``"my_field"``, axis=``"t"``,
                order ``1``). Must reference a field in ``fields`` and an axis
                in ``coords``. The parsed order is stored as ``lhs_order``
                (the single source of truth). NOTE: a dataset can *carry* a
                second-order LHS, but whether a given search algorithm can
                *discover* it is enforced fail-loud at fit time — first-order
                is the only order currently supported end-to-end.
            periodic: Iterable of axis names that are periodic.
            name: Dataset identifier (printed in repr).
            ground_truth: Optional ground-truth equation string.
            dtype: Float dtype to cast coords + fields to (default float64).

        Returns:
            A validated PDEDataset.

        Raises:
            ValueError: If ``lhs`` spec is malformed, references a missing
                field/axis, or field shapes don't match coords.

        Example:
            >>> ds = PDEDataset.from_arrays(
            ... coords={"x": x_array, "t": t_array},
            ... fields={"u": u_array}, # shape (len(x), len(t))
            ... lhs="u_t",
            ... periodic={"x"},
            ... )
        """
        if not coords:
            raise ValueError("coords must be a non-empty mapping")
        if not fields:
            raise ValueError("fields must be a non-empty mapping")

        lhs_field_parsed, lhs_axis_parsed, lhs_order_parsed = parse_lhs_spec(
            lhs, fields=fields, coords=coords
        )
        axes_dict = build_axes_dict(coords, dtype=dtype, periodic=periodic)
        fields_dict = build_fields_dict(fields, dtype=dtype)
        axis_order = list(coords.keys())




        try:
            return cls(
                name=name,
                task_type=TaskType.PDE,
                topology=DataTopology.GRID,
                axes=axes_dict,
                axis_order=axis_order,
                fields=fields_dict,
                lhs_field=lhs_field_parsed,
                lhs_axis=lhs_axis_parsed,
                lhs_order=lhs_order_parsed,
                ground_truth=ground_truth,
            )
        except ValueError as exc:
            raise annotate_shape_error(
                exc,
                axes_dict=axes_dict,
                fields_dict=fields_dict,
                axis_order=axis_order,
            ) from exc


def compute_dataset_fingerprint(dataset: PDEDataset) -> str:
    """Compute dataset fingerprint for cache isolation.

    The fingerprint includes:
    - Meta info: name, topology, lhs_field, lhs_axis
    - Shape info: field shapes
    - Content hash: sampled data hash (for large datasets)

    Args:
        dataset: The dataset to fingerprint.

    Returns:
        Unique string identifier for this dataset.
    """

    meta = f"{dataset.name}:{dataset.topology.value}"
    meta += f":{dataset.lhs_field}:{dataset.lhs_axis}"





    meta += f":lhs_order={dataset.lhs_order}"




    meta += f":{','.join(dataset.axis_order or [])}"



    if dataset.axes is not None:
        periodic_meta = ",".join(
            f"{name}={'P' if axis.is_periodic else 'N'}"
            for name, axis in sorted(dataset.axes.items())
        )
        meta += f":periodic={periodic_meta}"


    if dataset.fields is not None:
        shapes = "_".join(
            f"{name}{tuple(field.values.shape)}"
            for name, field in sorted(dataset.fields.items())
        )
    else:
        shapes = "no_fields"


    content_hash = hashlib.sha256()




    if dataset.axes is not None:
        for axis_name in sorted(dataset.axes.keys()):
            axis_values = dataset.axes[axis_name].values.detach().cpu().numpy()
            content_hash.update(axis_name.encode("utf-8"))
            content_hash.update(axis_values.tobytes())

    if dataset.fields is not None:
        for field in sorted(dataset.fields.values(), key=lambda f: f.name):

            data = field.values.detach().cpu().numpy()
            data_bytes = data.nbytes

            if data_bytes > _LARGE_DATA_THRESHOLD_BYTES:

                flat = data.ravel()
                sampled = flat[::_SAMPLING_STRIDE]
                content_hash.update(sampled.tobytes())
            else:
                content_hash.update(data.tobytes())

    return f"{meta}_{shapes}_{content_hash.hexdigest()[:8]}"
