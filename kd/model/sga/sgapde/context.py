"""
ProblemContext class to encapsulate all problem-specific data and computations.
This replaces the global state from setup.py.
"""

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import os

from .PDE_find import Diff, Diff2, FiniteDiff, FiniteDiff2
from .config import Net
from .metann import train_metanet
from .naming import build_derivative_key, validate_axis_name, validate_name

class ProblemContext:
    """Encapsulates all problem-specific data and computations."""
    
    def __init__(self, config):
        """
        Initialize the problem context with the given configuration.
        
        Args:
            config: Configuration object containing problem parameters
        """
        self.config = config
        self.device = config.device

        self.simple_mode = config.simple_mode
        self.target_field = getattr(config, "target_field", "u")
        self.lhs_axis = getattr(config, "lhs_axis", "t")
        
        # Initialize data
        self._load_data()

        self._initialize_slicing_indices()
        
        # Initialize operators
        self._init_operators()
        
        # Calculate derivatives
        self._calculate_derivatives()
        
        # Calculate errors
        self._calculate_errors()
        
    def _load_data(self):
        """
        加载数据。其优先级如下：
        1. 优先使用直接从 SolverConfig 传入的数据数组（kd框架模式）。
        2. 若无直接数据，则根据 problem_name 从文件加载。
        3. 加载后，根据 use_metadata 开关决定是否生成并使用高分辨率元数据。
        """
        if not self._has_registry_payload():
            if (
                self.config.u_data is not None
                or self.config.x_data is not None
                or self.config.t_data is not None
            ):
                print("\tINFO: Loading data directly from provided arrays (Framework Mode).")
            else:
                print(
                    f"\tINFO: Loading data from file for problem: "
                    f"'{self.config.problem_name}' (Standalone Mode)."
                )

        payload = self._build_registry_payload_from_config(self.config)
        if payload is None:
            raise ValueError("No data available to build structured grid payload.")

        fields, coords_1d, axis_order = payload
        self.config.fields_data = fields
        self.config.coords_1d = coords_1d
        self.config.axis_order = axis_order

        self._init_variable_registry(fields, coords_1d, axis_order)

        if self.target_field not in self.fields:
            raise ValueError(
                f"Structured grid requires target_field '{self.target_field}' "
                f"in fields_data; available: {sorted(self.fields.keys())}"
            )

        self._resolve_primary_spatial_axis()
        # Legacy views (x/dx/n) follow 'x' when present; otherwise use the primary axis.
        legacy_x_axis = "x" if "x" in self.axis_map else self.primary_spatial_axis
        self.legacy_x_axis = legacy_x_axis

        self.u_origin = self.fields[self.target_field]
        self.u = self.u_origin
        x_origin_1d = self.coords_1d.get("x")
        t_origin_1d = self.coords_1d.get("t")
        self.x = self.coord_grids.get(legacy_x_axis)
        self.t = self.coord_grids.get("t")
        self.x_origin = self.x
        self.t_origin = self.t
        self.x_all = self.x

        self.n_origin = self._shape_axis_length(self.u_origin.shape, self.axis_map[legacy_x_axis])
        self.m_origin = self._shape_axis_length(self.u_origin.shape, self.axis_map.get("t", 1))
        self.dx_origin = self.delta.get(legacy_x_axis, 0.0)
        self.dt_origin = self.delta.get("t", 0.0)

        self.n = self._shape_axis_length(self.u.shape, self.axis_map[legacy_x_axis])
        self.m = self._shape_axis_length(self.u.shape, self.axis_map.get("t", 1))
        self.dx = self.delta.get(legacy_x_axis, 0.0)
        self.dt = self.delta.get("t", 0.0)

        if self.config.use_metadata:
            if axis_order != ["x", "t"]:
                raise NotImplementedError(
                    "Metadata generation for registry payloads only supports axis_order "
                    "['x', 't']."
                )
            if len(self.fields) != 1 or self.target_field not in self.fields:
                raise NotImplementedError(
                    "Metadata generation for registry payloads only supports a single "
                    "target field."
                )
            if x_origin_1d is None or t_origin_1d is None:
                raise ValueError("Metadata generation requires x/t coordinates.")

            x_origin_1d = np.asarray(x_origin_1d).reshape(-1)
            t_origin_1d = np.asarray(t_origin_1d).reshape(-1)
            self.x_origin = x_origin_1d
            self.t_origin = t_origin_1d
            self.dx_origin = x_origin_1d[1] - x_origin_1d[0] if len(x_origin_1d) > 1 else 0.0
            self.dt_origin = t_origin_1d[1] - t_origin_1d[0] if len(t_origin_1d) > 1 else 0.0
            self.n_origin, self.m_origin = self.u_origin.shape

            self.u, x_1d, t_1d, _ = self._generate_metadata()

            self.u = np.asarray(self.u)
            x_1d = np.asarray(x_1d).reshape(-1)
            t_1d = np.asarray(t_1d).reshape(-1)

            fields = {self.target_field: self.u}
            coords_1d = {"x": x_1d, "t": t_1d}
            self._init_variable_registry(fields, coords_1d, axis_order)

            self.u = self.fields[self.target_field]
            self.x = self.coord_grids.get("x")
            self.t = self.coord_grids.get("t")
            self.x_all = self.x

            self.n = self._shape_axis_length(self.u.shape, self.axis_map.get("x", 0))
            self.m = self._shape_axis_length(self.u.shape, self.axis_map.get("t", 1))
            self.dx = self.delta.get("x", 0.0)
            self.dt = self.delta.get("t", 0.0)

            origin_axis_map = {axis: idx for idx, axis in enumerate(axis_order)}
            self.x_origin = self._broadcast_coord(
                x_origin_1d,
                origin_axis_map["x"],
                self.u_origin.shape,
            )
            self.t_origin = self._broadcast_coord(
                t_origin_1d,
                origin_axis_map["t"],
                self.u_origin.shape,
            )

        if self.config.delete_edges:
            self._apply_edge_deletion()

    def _has_registry_payload(self):
        return self.config.fields_data is not None or self.config.coords_1d is not None

    @staticmethod
    def _build_registry_payload_from_config(config):
        has_registry_payload = config.fields_data is not None or config.coords_1d is not None
        if has_registry_payload:
            if config.fields_data is None:
                raise ValueError("Structured grid requires fields_data.")
            if config.coords_1d is None:
                raise ValueError("Structured grid requires coords_1d.")
            if not isinstance(config.fields_data, dict):
                raise TypeError("Structured grid requires fields_data as a dict.")
            if not isinstance(config.coords_1d, dict):
                raise TypeError("Structured grid requires coords_1d as a dict.")
            fields = {name: np.asarray(value) for name, value in config.fields_data.items()}
            coords_1d = {axis: np.asarray(coord) for axis, coord in config.coords_1d.items()}
            axis_order = list(config.axis_order) if config.axis_order else list(coords_1d.keys())
            if not axis_order:
                raise ValueError("Structured grid requires a non-empty axis_order.")
            return fields, coords_1d, axis_order

        if (
            config.u_data is not None
            or config.x_data is not None
            or config.t_data is not None
        ):
            base_u, base_x, base_t = config.u_data, config.x_data, config.t_data
        else:
            base_u = getattr(config, "u", None)
            base_x = getattr(config, "x", None)
            base_t = getattr(config, "t", None)

        if base_u is None or base_x is None or base_t is None:
            return None

        target_field = getattr(config, "target_field", "u")
        fields = {target_field: np.asarray(base_u)}
        coords_1d = {
            "x": np.asarray(base_x).reshape(-1),
            "t": np.asarray(base_t).reshape(-1),
        }
        # Legacy inputs are always treated as 2D x/t grids.
        axis_order = ["x", "t"]
        return fields, coords_1d, axis_order


    def _coerce_fields(self, fields):
        if fields is None:
            raise ValueError("Structured grid requires fields_data.")
        if not isinstance(fields, dict):
            raise TypeError("Structured grid requires fields_data as a dict.")
        return {name: np.asarray(value) for name, value in fields.items()}

    def _coerce_coords(self, coords_1d):
        if coords_1d is None:
            raise ValueError("Structured grid requires coords_1d.")
        if not isinstance(coords_1d, dict):
            raise TypeError("Structured grid requires coords_1d as a dict.")
        return {axis: np.asarray(coord) for axis, coord in coords_1d.items()}

    def _resolve_axis_order(self, coords_1d):
        axis_order = list(self.config.axis_order) if self.config.axis_order else list(coords_1d.keys())
        if not axis_order:
            raise ValueError("Structured grid requires a non-empty axis_order.")
        return axis_order

    def _init_variable_registry(self, fields, coords_1d, axis_order):
        self._validate_registry_names(fields, coords_1d, axis_order)
        self._validate_structured_grid(fields, coords_1d, axis_order)
        self.fields = fields
        self.coords_1d = coords_1d
        self.axis_order = list(axis_order)
        self.axis_map = {axis: idx for idx, axis in enumerate(self.axis_order)}

        self.delta = {
            axis: self._compute_delta(coords_1d[axis], axis)
            for axis in self.axis_order
        }
        grid_shape = tuple(coords_1d[axis].shape[0] for axis in self.axis_order)
        self.coord_grids = self._build_coord_grids(coords_1d, self.axis_map, grid_shape)

    def _validate_registry_names(self, fields, coords_1d, axis_order):
        field_names = list(fields.keys())
        coord_names = list(coords_1d.keys())

        for axis in axis_order:
            validate_axis_name(axis)
        for coord in coord_names:
            validate_axis_name(coord)
        for field in field_names:
            validate_name(field)

        # Fail fast on collisions that would otherwise be silent in VARS/grad_fields.
        coord_conflicts = sorted(set(field_names) & set(coord_names))
        if coord_conflicts:
            raise ValueError(
                "Field names conflict with coordinate axes: "
                f"{coord_conflicts}."
            )

        derivative_keys = {
            build_derivative_key(field, axis, order=1)
            for field in field_names
            for axis in axis_order
        }
        derivative_conflicts = sorted(set(field_names) & derivative_keys)
        if derivative_conflicts:
            raise ValueError(
                "Field names conflict with derivative keys: "
                f"{derivative_conflicts}."
            )

        lhs_axis = self.lhs_axis
        if len(lhs_axis) == 1 and self.target_field in field_names:
            legacy_lhs_alias = f"{self.target_field}{lhs_axis}"
            if legacy_lhs_alias in field_names:
                # Prevent RHS from sneaking in lhs-derivative leaves via legacy alias.
                raise ValueError(
                    "Field names conflict with lhs derivative alias: "
                    f"['{legacy_lhs_alias}']. Use canonical keys like "
                    f"'{self.target_field}_{lhs_axis}' only for derived data."
                )

        legacy_conflicts = sorted(set(field_names) & {"ux"})
        if legacy_conflicts:
            raise ValueError(
                "Field names conflict with legacy aliases: "
                f"{legacy_conflicts}."
            )

    def _resolve_primary_spatial_axis(self):
        spatial_axes = [axis for axis in self.axis_order if axis != self.lhs_axis]
        if not spatial_axes:
            raise ValueError("Structured grid requires at least one spatial axis.")

        override = getattr(self.config, "primary_spatial_axis", None)
        if override is not None:
            if override not in self.axis_map:
                raise ValueError(
                    f"primary_spatial_axis '{override}' is not in axis_order {self.axis_order}."
                )
            if override == self.lhs_axis:
                raise ValueError("primary_spatial_axis cannot be the lhs_axis.")
            primary_axis = override
        else:
            primary_axis = "x" if "x" in spatial_axes else spatial_axes[0]

        self.primary_spatial_axis = primary_axis
        self.primary_spatial_index = self.axis_map[primary_axis]

    def _validate_structured_grid(self, fields, coords_1d, axis_order):
        if len(axis_order) != len(set(axis_order)):
            raise ValueError("Structured grid requires unique axis_order entries.")

        axis_set = set(axis_order)
        coord_set = set(coords_1d.keys())
        if axis_set != coord_set:
            missing = sorted(axis_set - coord_set)
            extra = sorted(coord_set - axis_set)
            raise ValueError(
                "Structured grid axis_order must match coords_1d keys. "
                f"Missing: {missing}, Extra: {extra}"
            )

        grid_shape = tuple(coords_1d[axis].shape[0] for axis in axis_order)
        for axis in axis_order:
            coord = coords_1d[axis]
            if coord.ndim != 1:
                raise ValueError(
                    f"Structured grid requires 1D coordinates for axis '{axis}'."
                )

        for name, field in fields.items():
            if field.shape != grid_shape:
                raise ValueError(
                    "Structured grid requires all fields to share the grid shape. "
                    f"Field '{name}' has shape {field.shape}, expected {grid_shape}."
                )

        if self.config.enforce_uniform_grid:
            for axis in axis_order:
                coord = coords_1d[axis]
                if coord.size < 2:
                    continue
                diffs = np.diff(coord)
                if not np.allclose(diffs, diffs[0]):
                    raise ValueError(
                        "Structured grid requires uniform spacing for axis "
                        f"'{axis}'."
                    )

    @staticmethod
    def _compute_delta(coord, axis):
        if coord.size < 2:
            return 0.0
        diffs = np.diff(coord)
        # Guard against duplicate or non-monotonic coordinates before FD uses delta.
        if np.any(diffs == 0):
            raise ValueError(
                f"Structured grid axis '{axis}' must have non-zero spacing."
            )
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            raise ValueError(
                f"Structured grid axis '{axis}' must be strictly monotonic."
            )
        return float(diffs[0])

    @staticmethod
    def _build_coord_grids(coords_1d, axis_map, grid_shape):
        coord_grids = {}
        for axis, index in axis_map.items():
            coord = coords_1d[axis]
            reshape = [1] * len(grid_shape)
            reshape[index] = coord.shape[0]
            coord_grids[axis] = np.broadcast_to(coord.reshape(reshape), grid_shape)
        return coord_grids

    @staticmethod
    def _broadcast_coord(coord_1d, axis_index, shape):
        reshape = [1] * len(shape)
        reshape[axis_index] = coord_1d.shape[0]
        return np.broadcast_to(coord_1d.reshape(reshape), shape)

    @staticmethod
    def _shape_axis_length(shape, axis_index):
        if axis_index < len(shape):
            return shape[axis_index]
        return 1

    def _apply_edge_deletion(self):
        if self.u.ndim != 2:
            raise NotImplementedError("delete_edges only supports 2D grids.")
        print("\tINFO: Deleting edges from the data (10% on each side).")
        if "x" not in self.axis_map:
            raise ValueError("delete_edges requires structured grid axis 'x'.")

        # NOTE(ND): Legacy delete_edges semantics are "crop spatial x edges".
        # The previous implementation assumed x is axis 0 and silently cropped the wrong
        # axis when axis_order was permuted (e.g. ['t', 'x']). Here we crop the actual
        # x-axis and keep coords_1d/grid_shape consistent.
        x_index = self.axis_map["x"]
        n_slice_start = int(self.n * 0.1)
        n_slice_end = int(self.n * 0.9)

        slicer = [slice(None)] * self.u.ndim
        slicer[x_index] = slice(n_slice_start, n_slice_end)
        slicer = tuple(slicer)

        new_fields = {name: field[slicer] for name, field in self.fields.items()}
        new_coords_1d = dict(self.coords_1d)
        new_coords_1d["x"] = np.asarray(new_coords_1d["x"]).reshape(-1)[n_slice_start:n_slice_end]

        # Rebuild the variable registry so fields/coords/axis_map/delta/coord_grids stay in sync.
        self.config.fields_data = new_fields
        self.config.coords_1d = new_coords_1d
        self._init_variable_registry(new_fields, new_coords_1d, self.axis_order)
        self._resolve_primary_spatial_axis()
        legacy_x_axis = "x" if "x" in self.axis_map else self.primary_spatial_axis
        self.legacy_x_axis = legacy_x_axis

        self.u = self.fields[self.target_field]
        self.x = self.coord_grids.get(legacy_x_axis)
        self.t = self.coord_grids.get("t")
        self.x_all = self.x

        self.n = self._shape_axis_length(self.u.shape, self.axis_map[legacy_x_axis])
        self.m = self._shape_axis_length(self.u.shape, self.axis_map.get("t", 1))
        self.dx = self.delta.get(legacy_x_axis, 0.0)
        self.dt = self.delta.get("t", 0.0)
        
    def _init_operators(self):
        """Initialize operator definitions."""
        # Define zeros array
        self.zeros = np.zeros(self.u.shape)
        self.grad_fields = {}
        
        # Define operators (from setup.py)
        self.ALL = np.array([
            ['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], 
            ['/', 2, self.config.divide], ['d', 2, Diff], ['d^2', 2, Diff2], 
            ['u', 0, self.u], ['x', 0, self.x], ['ux', 0, None],  # ux will be set after calculation
            ['0', 0, self.zeros], ['^2', 1, np.square], ['^3', 1, self._cubic]
        ], dtype=object)
        
        self.OPS = np.array([
            ['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], 
            ['/', 2, self.config.divide], ['d', 2, Diff], ['d^2', 2, Diff2], 
            ['^2', 1, np.square], ['^3', 1, self._cubic]
        ], dtype=object)
        
        self.ROOT = np.array([
            ['*', 2, np.multiply], ['d', 2, Diff], ['d^2', 2, Diff2], 
            ['/', 2, self.config.divide], ['^2', 1, np.square], ['^3', 1, self._cubic]
        ], dtype=object)
        
        self.OP1 = np.array([['^2', 1, np.square], ['^3', 1, self._cubic]], dtype=object)
        
        self.OP2 = np.array([
            ['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], 
            ['/', 2, self.config.divide], ['d', 2, Diff], ['d^2', 2, Diff2]
        ], dtype=object)
        
        self.VARS = self._build_vars(include_grad=False)
        
        den_entries = []
        for axis in self.axis_order:
            if axis == self.lhs_axis:
                continue
            coord = self.coord_grids.get(axis)
            if coord is not None:
                # 遵循旧代码的数据结构约定：[name, child_num, payload]。
                # 这里 den 的元素必须是“叶子变量”，因此 child_num 固定为 0。
                den_entries.append([axis, 0, coord])
        if not den_entries:
            raise ValueError("No RHS derivative axes available after filtering lhs_axis.")
        self.den = np.array(den_entries, dtype=object)
        # Guardrail: RHS denominators must never include lhs_axis (prevents ut on RHS).
        self._assert_no_lhs_in_den()
    
    @staticmethod # 最重要的修复! 一定要加这个... 不然第一个参数是 self 的话会爆炸 :(
    def _cubic(inputs):
        """Cubic function."""
        return np.power(inputs, 3)
        
    def _calculate_derivatives(self):
        """Calculate derivatives using finite differences or autograd."""
        
        if self.config.use_autograd:
            self._calculate_derivatives_autograd()
        else:
            self._calculate_derivatives_difference()

        self._compute_grad_fields()
            
        # Update operators with calculated derivatives
        self._update_operators_with_derivatives()
        
    def _calculate_derivatives_difference(self):
        """Calculate derivatives using finite differences."""
        if self.lhs_axis not in self.axis_map:
            raise ValueError(
                f"Structured grid missing lhs_axis '{self.lhs_axis}' in axis_order."
            )
        primary_axis = getattr(self, "primary_spatial_axis", None)
        if primary_axis not in self.axis_map:
            raise ValueError("Structured grid missing a valid primary spatial axis.")

        lhs_index = self.axis_map[self.lhs_axis]
        x_index = self.axis_map[primary_axis]
        lhs_delta = self.delta.get(self.lhs_axis, 0.0)
        x_delta = self.delta.get(primary_axis, 0.0)

        self.ut = self._finite_diff_along_axis(self.u, lhs_delta, lhs_index, order=1)
        self.ux = self._finite_diff_along_axis(self.u, x_delta, x_index, order=1)
        # Legacy stencil: u_xx is computed as d(u_x)/dx (not direct second diff).
        # self.uxx = self._finite_diff_along_axis(self.u, x_delta, x_index, order=2)
        self.uxx = self._finite_diff_along_axis(self.ux, x_delta, x_index, order=1)
        self.uxxx = self._finite_diff_along_axis(self.uxx, x_delta, x_index, order=1)

        origin_lhs_delta = self._origin_delta(self.lhs_axis)
        origin_x_delta = self._origin_delta(primary_axis)
        self.ut_origin = self._finite_diff_along_axis(
            self.u_origin, origin_lhs_delta, lhs_index, order=1
        )
        self.ux_origin = self._finite_diff_along_axis(
            self.u_origin, origin_x_delta, x_index, order=1
        )
        # Keep legacy stencil for origin as well.
        # self.uxx_origin = self._finite_diff_along_axis(
        #     self.u_origin, origin_x_delta, x_index, order=2
        # )
        self.uxx_origin = self._finite_diff_along_axis(
            self.ux_origin, origin_x_delta, x_index, order=1
        )
        self.uxxx_origin = self._finite_diff_along_axis(
            self.uxx_origin, origin_x_delta, x_index, order=1
        )

    def _compute_grad_fields(self):
        """Precompute spatial first derivatives for VARS (exclude lhs axis)."""
        if not hasattr(self, "grad_fields") or self.grad_fields is None:
            self.grad_fields = {}
        if not hasattr(self, "fields") or not self.fields:
            return

        for field_name, field_value in self.fields.items():
            for axis in self.axis_order:
                if axis == self.lhs_axis:
                    continue
                key = build_derivative_key(field_name, axis, order=1)
                if key in self.grad_fields:
                    continue
                axis_index = self.axis_map[axis]
                delta = self.delta.get(axis, 0.0)

                if field_name == self.target_field and axis == self.primary_spatial_axis:
                    grad = self.ux
                else:
                    grad = self._finite_diff_along_axis(
                        field_value, delta, axis_index, order=1
                    )

                self.grad_fields[key] = grad
        if self.target_field in self.fields:
            key = build_derivative_key(self.target_field, self.primary_spatial_axis, order=1)
            grad = self.grad_fields.get(key)
            if grad is not None and "ux" not in self.fields and "ux" not in self.grad_fields:
                # Legacy alias for backward compatibility on the primary spatial axis.
                self.grad_fields["ux"] = grad

    @staticmethod
    def _finite_diff_along_axis(data, delta, axis, order=1):
        if order == 1:
            func = FiniteDiff
        elif order == 2:
            func = FiniteDiff2
        else:
            raise ValueError(f"Unsupported finite diff order: {order}")
        return np.apply_along_axis(func, axis, data, delta)

    def _origin_delta(self, axis):
        if axis == "x":
            return self.dx_origin
        if axis == "t":
            return self.dt_origin
        return self.delta.get(axis, 0.0)
            
    def _calculate_derivatives_autograd(self):
        """Calculate derivatives using autograd."""
        raw_autograd_fields = getattr(self.config, "autograd_fields", None)
        if raw_autograd_fields is None:
            autograd_fields = list(self.fields.keys())
        elif isinstance(raw_autograd_fields, (list, tuple, set)):
            autograd_fields = list(raw_autograd_fields)
        else:
            autograd_fields = [raw_autograd_fields]

        missing_fields = [name for name in autograd_fields if name not in self.fields]
        if missing_fields:
            raise ValueError(
                f"Autograd fields not found in data: {missing_fields}"
            )
        if self.target_field not in autograd_fields:
            raise ValueError(
                "Autograd fields must include target_field for LHS computation."
            )

        autograd_models = getattr(self.config, "autograd_models", None)
        if autograd_models is None:
            autograd_models = {}
        if not isinstance(autograd_models, dict):
            raise TypeError("autograd_models must be a dict when provided.")

        if (
            getattr(self.config, "autograd_model", None) is not None
            and self.target_field not in autograd_models
        ):
            autograd_models[self.target_field] = self.config.autograd_model

        autograd_model_paths = getattr(self.config, "autograd_model_paths", None)
        if autograd_model_paths is None:
            autograd_model_paths = {}
        if not isinstance(autograd_model_paths, dict):
            raise TypeError("autograd_model_paths must be a dict when provided.")
        if self.config.model_path and self.target_field not in autograd_model_paths:
            autograd_model_paths[self.target_field] = self.config.model_path

        num_feature = len(self.axis_order)

        def _infer_model_input_dim(model):
            if hasattr(model, "in_features"):
                try:
                    return int(model.in_features)
                except (TypeError, ValueError):
                    pass
            for attr in ("input_dim", "n_features", "num_features"):
                value = getattr(model, attr, None)
                if value is None:
                    continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    return int(module.in_features)
            return None

        def _validate_model_input_dim(model, field_name):
            # Guard input dimensionality early to avoid opaque shape errors in autograd.
            inferred = _infer_model_input_dim(model)
            if inferred is not None:
                if inferred != num_feature:
                    raise ValueError(
                        f"Autograd model input dim {inferred} does not match axis_order "
                        f"length {num_feature} for field '{field_name}'."
                    )
                return
            dummy = torch.zeros((1, num_feature), dtype=torch.float32, device=self.device)
            try:
                with torch.no_grad():
                    model(dummy)
            except Exception as exc:
                raise ValueError(
                    f"Autograd model input dim check failed for field '{field_name}': "
                    f"expected {num_feature} features."
                ) from exc

        def _load_model_from_path(path: str):
            model = Net(num_feature, self.config.hidden_dim, 1)
            try:
                state = torch.load(
                    path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            return model

        def _resolve_model(field_name: str, field_value: np.ndarray):
            model = autograd_models.get(field_name)
            if model is not None:
                return model

            path = autograd_model_paths.get(field_name)
            if path and os.path.exists(path):
                model = _load_model_from_path(path)
            else:
                # NOTE(ND): Train a MetaNN model on-the-fly when no checkpoint exists.
                model, stats = train_metanet(
                    self.coords_1d,
                    field_value,
                    axis_order=self.axis_order,
                    hidden_dim=self.config.hidden_dim,
                    max_epoch=self.config.max_epoch,
                    train_ratio=self.config.train_ratio,
                    normalize=self.config.normal,
                    seed=self.config.seed,
                    device=self.device,
                )
                if not hasattr(self, "autograd_stats") or self.autograd_stats is None:
                    self.autograd_stats = {}
                self.autograd_stats[field_name] = stats

            autograd_models[field_name] = model
            return model

        def _autograd_high_order_2d(model, coord_grids, field_value, grid_shape, normalize):
            # Legacy 2D autograd semantics: compute ux/uxx/uxxx directly via autograd,
            # but use axis_order to build inputs so permuted axes are handled correctly.
            coord_vectors = []
            for axis in self.axis_order:
                coord = coord_grids.get(axis)
                if coord is None:
                    raise ValueError(f"Missing coord grid for axis '{axis}'.")
                coord_vectors.append(coord.reshape(-1))
            X = np.stack(coord_vectors, axis=1).astype(np.float32)

            if normalize:
                x_mean = X.mean(axis=0)
                x_std = X.std(axis=0)
                x_std_safe = np.where(x_std == 0.0, 1.0, x_std)
                X_in = (X - x_mean) / x_std_safe
                y_std = float(np.std(field_value.reshape(-1)))
                if y_std == 0.0:
                    y_std = 1.0
            else:
                x_std_safe = np.ones(X.shape[1], dtype=float)
                X_in = X
                y_std = 1.0

            X_tensor = torch.from_numpy(X_in).float().to(self.device)
            X_tensor.requires_grad_(True)

            pred = model(X_tensor)
            if pred.ndim == 1:
                pred = pred.view(-1, 1)
            elif pred.ndim == 2 and pred.shape[1] != 1:
                raise ValueError(
                    f"Autograd model output must be (N, 1); got {tuple(pred.shape)}."
                )
            elif pred.ndim != 2:
                raise ValueError(
                    f"Autograd model output must be 1D or 2D; got ndim={pred.ndim}."
                )
            # NOTE(legacy): pred.sum() mixes outputs if the model returns multiple columns.
            grads = torch.autograd.grad(
                outputs=pred.sum(),
                inputs=X_tensor,
                create_graph=True,
            )[0]
            x_index = self.axis_map["x"]
            t_index = self.axis_map["t"]

            ux = grads[:, x_index]
            ut = grads[:, t_index]
            if ux.requires_grad:
                uxx_full = torch.autograd.grad(
                    outputs=ux.sum(),
                    inputs=X_tensor,
                    create_graph=True,
                )[0]
                uxx = uxx_full[:, x_index]
            else:
                uxx = torch.zeros_like(ux)

            if uxx.requires_grad:
                uxxx_full = torch.autograd.grad(
                    outputs=uxx.sum(),
                    inputs=X_tensor,
                    create_graph=True,
                )[0]
                uxxx = uxxx_full[:, x_index]
            else:
                uxxx = torch.zeros_like(ux)

            if normalize:
                scale_x = float(y_std / x_std_safe[x_index])
                scale_t = float(y_std / x_std_safe[t_index])
                scale_xx = float(y_std / (x_std_safe[x_index] ** 2))
                scale_xxx = float(y_std / (x_std_safe[x_index] ** 3))
                # Chain rule: convert d(u_norm)/d(x_norm) back to d(u)/d(x).
                ux = ux * scale_x
                ut = ut * scale_t
                uxx = uxx * scale_xx
                uxxx = uxxx * scale_xxx

            return (
                ut.detach().cpu().numpy().reshape(grid_shape),
                ux.detach().cpu().numpy().reshape(grid_shape),
                uxx.detach().cpu().numpy().reshape(grid_shape),
                uxxx.detach().cpu().numpy().reshape(grid_shape),
            )

        def _apply_autograd_strategy_2d_high_order(model, coord_grids, field_value, grid_shape, normalize):
            # 统一 legacy 2D 高阶导路径，避免分支散落（行为保持不变）。
            return _autograd_high_order_2d(
                model,
                coord_grids,
                field_value,
                grid_shape,
                normalize,
            )

        self.config.autograd_models = autograd_models

        # Legacy 2D x/t path: keep original behavior for backward compatibility.
        if (
            self.u.ndim == 2
            and len(self.axis_order) == 2
            and "x" in self.axis_map
            and "t" in self.axis_map
            and self.lhs_axis == "t"
            and len(autograd_fields) == 1
        ):
            model = _resolve_model(self.target_field, self.u)
            model.to(self.device)
            _validate_model_input_dim(model, self.target_field)
            model.eval()
            normalize = bool(getattr(self.config, "normal", True))
            self.ut, self.ux, self.uxx, self.uxxx = _apply_autograd_strategy_2d_high_order(
                model,
                self.coord_grids,
                self.u,
                self.u.shape,
                normalize,
            )

            origin_coord_grids = dict(self.coord_grids)
            origin_coord_grids["x"] = self.x_origin
            origin_coord_grids["t"] = self.t_origin
            self.ut_origin, self.ux_origin, self.uxx_origin, self.uxxx_origin = (
                _apply_autograd_strategy_2d_high_order(
                    model,
                    origin_coord_grids,
                    self.u_origin,
                    self.u_origin.shape,
                    normalize,
                )
            )
            return

        if self.u.ndim != len(self.axis_order):
            raise ValueError(
                "Autograd requires field dimensions to match axis_order length."
            )
        grid_shape = self.u.shape

        coord_vectors = []
        for axis in self.axis_order:
            coord = self.coord_grids.get(axis)
            if coord is None:
                raise ValueError(f"Missing coord grid for axis '{axis}'.")
            coord_vectors.append(coord.reshape(-1))

        X = np.stack(coord_vectors, axis=1).astype(np.float32)

        normalize = bool(getattr(self.config, "normal", True))
        if normalize:
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0)
            x_std_safe = np.where(x_std == 0.0, 1.0, x_std)
            X_in = (X - x_mean) / x_std_safe
        else:
            x_std_safe = np.ones(X.shape[1], dtype=float)
            X_in = X

        self.grad_fields = {}
        target_axis_grads = None

        for field_name in autograd_fields:
            field_value = self.fields[field_name]
            model = _resolve_model(field_name, field_value)
            model.to(self.device)
            _validate_model_input_dim(model, field_name)
            model.eval()

            X_tensor = torch.from_numpy(X_in).float().to(self.device)
            X_tensor.requires_grad_(True)

            pred = model(X_tensor)
            if pred.ndim == 1:
                pred = pred.view(-1, 1)
            elif pred.ndim == 2 and pred.shape[1] != 1:
                raise ValueError(
                    f"Autograd model output must be (N, 1); got {tuple(pred.shape)} "
                    f"for field '{field_name}'."
                )
            elif pred.ndim != 2:
                raise ValueError(
                    f"Autograd model output must be 1D or 2D; got ndim={pred.ndim} "
                    f"for field '{field_name}'."
                )
            grads = torch.autograd.grad(
                outputs=pred.sum(),
                inputs=X_tensor,
                create_graph=True,
            )[0]

            if normalize:
                y_std = float(np.std(field_value.reshape(-1)))
                if y_std == 0.0:
                    y_std = 1.0
                # NOTE(ND): Convert d(u_norm)/d(x_norm) to d(u)/d(x).
                scale = torch.tensor(
                    y_std / x_std_safe,
                    dtype=grads.dtype,
                    device=grads.device,
                )
                grads = grads * scale

            grads_np = grads.detach().cpu().numpy()
            axis_grads = {
                axis: grads_np[:, axis_index].reshape(grid_shape)
                for axis_index, axis in enumerate(self.axis_order)
            }

            if field_name == self.target_field:
                target_axis_grads = axis_grads

            for axis, values in axis_grads.items():
                if axis == self.lhs_axis:
                    continue
                key = build_derivative_key(field_name, axis, order=1)
                self.grad_fields[key] = values

        if target_axis_grads is None or self.lhs_axis not in target_axis_grads:
            raise ValueError(
                f"Structured grid missing lhs_axis '{self.lhs_axis}' in axis_order."
            )

        self.ut = target_axis_grads[self.lhs_axis]
        primary_axis = self.primary_spatial_axis
        if primary_axis in target_axis_grads:
            self.ux = target_axis_grads[primary_axis]
        else:
            # Legacy attribute fallback when no spatial axis is available.
            self.ux = np.zeros_like(self.u)

        if primary_axis in self.axis_map:
            x_index = self.axis_map[primary_axis]
            x_delta = self.delta.get(primary_axis, 0.0)
            self.uxx = self._finite_diff_along_axis(self.ux, x_delta, x_index, order=1)
            self.uxxx = self._finite_diff_along_axis(self.uxx, x_delta, x_index, order=1)
        else:
            self.uxx = np.zeros_like(self.u)
            self.uxxx = np.zeros_like(self.u)

        # For registry payloads, origin matches current grid; keep derivative copies.
        self.ut_origin = np.array(self.ut, copy=True)
        self.ux_origin = np.array(self.ux, copy=True)
        self.uxx_origin = np.array(self.uxx, copy=True)
        self.uxxx_origin = np.array(self.uxxx, copy=True)
        
    def _update_operators_with_derivatives(self):
        """Update operator arrays with calculated derivatives."""
        # Guardrail: grad_fields must exclude lhs-axis derivatives (avoid trivial ut=ut).
        self._assert_no_lhs_in_grad_fields()
        self.VARS = self._build_vars(include_grad=True)

        # Find and update ux in ALL array
        for i, op in enumerate(self.ALL):
            if op[0] == 'ux':
                op[2] = self.grad_fields.get("ux", self.ux)

    def _build_vars(self, *, include_grad):
        vars_list = []

        for field_name, field_value in self.fields.items():
            vars_list.append([field_name, 0, field_value])

        for axis in self.axis_order:
            coord = self.coord_grids.get(axis)
            if coord is not None:
                vars_list.append([axis, 0, coord])

        if include_grad:
            for name, value in self.grad_fields.items():
                vars_list.append([name, 0, value])

        vars_list.append(['0', 0, self.zeros])

        return np.array(vars_list, dtype=object)

    def _assert_no_lhs_in_den(self):
        """Fail fast if lhs_axis appears in derivative denominators."""
        if not hasattr(self, "den") or self.den is None:
            return
        # den is the allowed RHS derivative denominator set (leaf nodes only).
        if any(entry[0] == self.lhs_axis for entry in self.den):
            raise ValueError(
                f"den contains lhs_axis '{self.lhs_axis}', which is forbidden."
            )

    def _assert_no_lhs_in_grad_fields(self):
        """Fail fast if grad_fields contains lhs-axis derivatives."""
        if not hasattr(self, "grad_fields") or self.grad_fields is None:
            return
        # grad_fields should include spatial derivatives only; lhs derivatives are forbidden.
        forbidden = []
        for field_name in self.fields:
            key = build_derivative_key(field_name, self.lhs_axis, order=1)
            if key in self.grad_fields:
                forbidden.append(key)
        if forbidden:
            raise ValueError(
                f"grad_fields contains lhs_axis derivative(s): {sorted(forbidden)}"
            )


    def _initialize_slicing_indices(self):
        """Calculate and store slicing indices as instance attributes."""
        self.n1 = int(self.n * 0.1)
        self.n2 = int(self.n * 0.9)
        self.m1 = int(self.m * 0)
        self.m2 = int(self.m * 1)
        self.n1_origin = int(self.n_origin * 0.1)
        self.n2_origin = int(self.n_origin * 0.9)
        self.m1_origin = int(self.m_origin * 0)
        self.m2_origin = int(self.m_origin * 1)
    
    def _calculate_errors(self):
        """Calculate errors between left and right sides of the PDE."""
        # Prepare default terms for evaluation
        flat_size = self.u.size
        self.default_u = np.reshape(self.u, (flat_size, 1))
        self.default_ux = np.reshape(self.ux, (flat_size, 1))
        self.default_uxx = np.reshape(self.uxx, (flat_size, 1))
        self.default_u2 = np.reshape(self.u**2, (flat_size, 1))
        self.default_u3 = np.reshape(self.u**3, (flat_size, 1))
        self.default_terms = self.default_u
        self.default_names = ['u']
        self.num_default = self.default_terms.shape[1]

        # 若无解析模板或显式标记为无真解，则跳过 ground-truth 误差计算
        if (
            not getattr(self.config, "has_ground_truth", False)
            or self.config.right_side is None
            or self.config.left_side is None
            or self.config.right_side_origin is None
            or self.config.left_side_origin is None
        ):
            print("\tINFO: No analytic PDE template; skipping ground-truth error computation.")
            return

        # 计算 right_side, right_side_full, right_side_origin, right_side_full_origin
        # 动态执行 config 的表达式
        local_vars = {
            "u": self.u, "ut": self.ut, "ux": self.ux, "uxx": self.uxx, "uxxx": self.uxxx,
            "u_origin": self.u_origin, "ut_origin": self.ut_origin, "ux_origin": self.ux_origin,
            "uxx_origin": self.uxx_origin, "uxxx_origin": self.uxxx_origin
        }
        # right_side, left_side
        exec(self.config.right_side, {}, local_vars)
        exec(self.config.left_side, {}, local_vars)
        right_side = local_vars.get("right_side")
        left_side = local_vars.get("left_side")
        self.right_side_full = right_side
        self.left_side_full = left_side
        self.right_side = right_side[self.n1:self.n2, self.m1:self.m2]
        self.left_side = left_side[self.n1:self.n2, self.m1:self.m2]

        # right_side_origin, left_side_origin
        exec(self.config.right_side_origin, {}, local_vars)
        exec(self.config.left_side_origin, {}, local_vars)
        right_side_origin = local_vars.get("right_side_origin")
        left_side_origin = local_vars.get("left_side_origin")
        self.right_side_full_origin = right_side_origin
        self.left_side_full_origin = left_side_origin
        self.right_side_origin = right_side_origin[self.n1_origin:self.n2_origin, self.m1_origin:self.m2_origin]
        self.left_side_origin = left_side_origin[self.n1_origin:self.n2_origin, self.m1_origin:self.m2_origin]

        # 计算并打印去除边缘的数据误差
        right = np.reshape(self.right_side, ((self.n2-self.n1)*(self.m2-self.m1), 1))
        left = np.reshape(self.left_side, ((self.n2-self.n1)*(self.m2-self.m1), 1))
        diff = np.linalg.norm(left-right, 2)/((self.n2-self.n1)*(self.m2-self.m1))
        print('\tdata error without edges', diff)

        right_origin = np.reshape(self.right_side_origin, ((self.n2_origin-self.n1_origin)*(self.m2_origin-self.m1_origin), 1))
        left_origin = np.reshape(self.left_side_origin, ((self.n2_origin-self.n1_origin)*(self.m2_origin-self.m1_origin), 1))
        diff_origin = np.linalg.norm(left_origin-right_origin, 2)/((self.n2_origin-self.n1_origin)*(self.m2_origin-self.m1_origin))
        print('\tdata error_origin without edges', diff_origin)

        # 重新计算并打印完整数据的误差
        n1_full, n2_full, m1_full, m2_full = 0, self.n, 0, self.m
        right_full_reshaped = np.reshape(self.right_side_full, ((n2_full-n1_full)*(m2_full-m1_full), 1))
        left_full_reshaped = np.reshape(self.left_side_full, ((n2_full-n1_full)*(m2_full-m1_full), 1))
        diff_full = np.linalg.norm(left_full_reshaped - right_full_reshaped, 2)/((n2_full-n1_full)*(m2_full-m1_full))
        print('\tdata error', diff_full)
        
        n1_origin_full, n2_origin_full, m1_origin_full, m2_origin_full = 0, self.n_origin, 0, self.m_origin
        right_origin_full_reshaped = np.reshape(self.right_side_full_origin, ((n2_origin_full-n1_origin_full)*(m2_origin_full-m1_origin_full), 1))
        left_origin_full_reshaped = np.reshape(self.left_side_full_origin, ((n2_origin_full-n1_origin_full)*(m2_origin_full-m1_origin_full), 1))
        diff_origin_full = np.linalg.norm(left_origin_full_reshaped - right_origin_full_reshaped, 2)/((n2_origin_full-n1_origin_full)*(m2_origin_full-m1_origin_full))
        print('\tdata error_origin', diff_origin_full)

    # 使用预训练的神经网络生成高分辨率数据，从旧的 Data_generator.py 迁移而来。
    def _generate_metadata(self):
        """Generate high-resolution metadata using a pre-trained model."""
        print("INFO: Generating high-resolution metadata from pre-trained model (with normalization)...")
        # Load pre-trained model
        model = Net(self.config.num_feature, self.config.hidden_dim, 1)
        try:
            model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        except FileNotFoundError:
            print(
                "[SGA Metadata ERROR] Failed to generate metadata because the "
                f"pretrained model was not found: {self.config.model_path}"
            )
            sys.exit(1)
        model.to(self.device)

        # 1. 计算原始数据 u_origin 的均值和标准差，用于后续的反归一化
        Y_raw = self.u_origin.reshape(-1, 1)
        Y_raw_mean = Y_raw.mean()
        Y_raw_std = Y_raw.std()

        # 准备高分辨率网格
        n_fine = self.config.fine_ratio * self.n_origin
        m_fine = self.config.fine_ratio * self.m_origin
        x_new = np.linspace(self.x_origin.min(), self.x_origin.max(), n_fine)
        t_new = np.linspace(self.t_origin.min(), self.t_origin.max(), m_fine)

        # 创建新网格的输入坐标
        X1 = np.repeat(x_new.reshape(-1, 1), m_fine, axis=1)
        X2 = np.repeat(t_new.reshape(1, -1), n_fine, axis=0)
        X_grid = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)

        # 2. 根据开关，对输入坐标进行归一化
        if self.config.normal:
            X_grid_mean = X_grid.mean(axis=0)
            X_grid_std = X_grid.std(axis=0)
            X_normalized = (X_grid - X_grid_mean) / X_grid_std
            X_tensor = torch.from_numpy(X_normalized).float().to(self.device)
        else:
            X_tensor = torch.from_numpy(X_grid).float().to(self.device)

        
        # 使用模型进行预测
        y_pred = model(X_tensor)
        y_pred_numpy = y_pred.cpu().data.numpy().flatten()

        # 3. 根据开关，对模型的输出进行反归一化，使其回到原始数据的尺度
        if self.config.normal:
            result_pred_real = y_pred_numpy * Y_raw_std + Y_raw_mean
        else:
            result_pred_real = y_pred_numpy

        u_new = result_pred_real.reshape(n_fine, m_fine)

        # 返回生成的高分辨率数据
        return u_new, x_new, t_new, x_new

    def get_pde_libs(self):
        """Return the PDE library lists (initially empty)."""
        return [], []
