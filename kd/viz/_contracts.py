"""Shared data contracts for KD visualization intents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ResidualPlotData:
    actual: np.ndarray
    predicted: np.ndarray
    residuals: np.ndarray
    input_coordinates: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actual = np.asarray(self.actual)
        self.predicted = np.asarray(self.predicted)
        self.residuals = np.asarray(self.residuals)

        if self.actual.shape != self.predicted.shape:
            raise ValueError("'actual' and 'predicted' must have identical shapes")
        if self.actual.shape != self.residuals.shape:
            raise ValueError("'residuals' must match the shape of 'actual'")

        if self.input_coordinates is not None:
            coords = np.asarray(self.input_coordinates)
            if coords.shape[0] != self.actual.shape[0]:
                raise ValueError(
                    "'input_coordinates' must contain one entry per sample"
                )
            self.input_coordinates = coords

    @classmethod
    def from_actual_predicted(
        cls,
        actual: Any,
        predicted: Any,
        *,
        input_coordinates: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ResidualPlotData":
        actual_arr = np.asarray(actual)
        pred_arr = np.asarray(predicted)
        residuals = actual_arr - pred_arr
        coords = None
        if input_coordinates is not None:
            coords = np.asarray(input_coordinates)
        return cls(
            actual=actual_arr,
            predicted=pred_arr,
            residuals=residuals,
            input_coordinates=coords,
            metadata=metadata or {},
        )


@dataclass
class OptimizationHistoryData:
    steps: np.ndarray
    objective: np.ndarray
    complexity: Optional[np.ndarray] = None
    population_size: Optional[np.ndarray] = None
    unique_modules: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.steps = np.asarray(self.steps)
        self.objective = np.asarray(self.objective)
        if self.steps.shape != self.objective.shape:
            raise ValueError("'steps' and 'objective' must have identical shapes")

        if self.complexity is not None:
            comp = np.asarray(self.complexity)
            if comp.shape != self.steps.shape:
                raise ValueError("'complexity' must match the shape of 'steps'")
            self.complexity = comp

        if self.population_size is not None:
            pop = np.asarray(self.population_size)
            if pop.shape != self.steps.shape:
                raise ValueError("'population_size' must match the shape of 'steps'")
            self.population_size = pop

        if self.unique_modules is not None:
            unique = np.asarray(self.unique_modules)
            if unique.shape != self.steps.shape:
                raise ValueError("'unique_modules' must match the shape of 'steps'")
            self.unique_modules = unique


@dataclass
class FieldComparisonData:
    spatial_coords: List[np.ndarray]
    t_coords: np.ndarray
    true_field: np.ndarray
    predicted_field: np.ndarray
    residual_field: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        spatial_coords: Any = None,
        t_coords: Any = None,
        true_field: Any = None,
        predicted_field: Any = None,
        residual_field: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        x_coords: Any = None,
    ) -> None:
        # Backward compat: accept x_coords= keyword as 1D shortcut
        if spatial_coords is None and x_coords is not None:
            spatial_coords = x_coords
        if spatial_coords is None:
            raise TypeError("'spatial_coords' (or 'x_coords') is required")

        # Auto-wrap a single ndarray into a list (1D spatial case)
        if isinstance(spatial_coords, np.ndarray):
            if spatial_coords.ndim <= 1:
                self.spatial_coords = [np.asarray(spatial_coords)]
            else:
                # 2D array passed as single coords — wrap it
                self.spatial_coords = [np.asarray(spatial_coords)]
        elif isinstance(spatial_coords, list):
            # Check if it's a list of arrays (multi-dim) or a plain list (single 1D)
            if len(spatial_coords) > 0 and isinstance(spatial_coords[0], np.ndarray):
                self.spatial_coords = [np.asarray(c) for c in spatial_coords]
            else:
                # Plain list → single 1D coord
                self.spatial_coords = [np.asarray(spatial_coords)]
        else:
            self.spatial_coords = [np.asarray(spatial_coords)]

        self.t_coords = np.asarray(t_coords)
        self.true_field = np.asarray(true_field)
        self.predicted_field = np.asarray(predicted_field)
        self.metadata = metadata if metadata is not None else {}

        if self.true_field.shape != self.predicted_field.shape:
            raise ValueError("'true_field' and 'predicted_field' must have identical shapes")

        expected_shape = tuple(c.size for c in self.spatial_coords) + (self.t_coords.size,)
        if self.true_field.shape != expected_shape:
            raise ValueError(
                f"Field data must have shape {expected_shape} "
                f"(spatial_coords sizes + t_coords size), got {self.true_field.shape}"
            )

        if residual_field is None:
            self.residual_field = self.true_field - self.predicted_field
        else:
            residual = np.asarray(residual_field)
            if residual.shape != expected_shape:
                raise ValueError("'residual_field' must match the field shape")
            self.residual_field = residual

    @property
    def x_coords(self) -> np.ndarray:
        """Backward-compatible shortcut for the first spatial axis."""
        return self.spatial_coords[0]

    @property
    def n_spatial_dims(self) -> int:
        return len(self.spatial_coords)


__all__ = [
    'ResidualPlotData',
    'OptimizationHistoryData',
    'FieldComparisonData',
    'TimeSliceComparisonData',
    'TermContribution',
    'TermRelationshipData',
    'ParityPlotData',
    'RewardEvolutionData',
]


@dataclass
class TimeSliceComparisonData:
    spatial_coords: List[np.ndarray]
    t_coords: np.ndarray
    true_field: np.ndarray
    predicted_field: np.ndarray
    slice_times: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        spatial_coords: Any = None,
        t_coords: Any = None,
        true_field: Any = None,
        predicted_field: Any = None,
        slice_times: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        x_coords: Any = None,
    ) -> None:
        # Backward compat: accept x_coords= keyword as 1D shortcut
        if spatial_coords is None and x_coords is not None:
            spatial_coords = x_coords
        if spatial_coords is None:
            raise TypeError("'spatial_coords' (or 'x_coords') is required")

        if isinstance(spatial_coords, np.ndarray):
            if spatial_coords.ndim <= 1:
                self.spatial_coords = [np.asarray(spatial_coords)]
            else:
                self.spatial_coords = [np.asarray(spatial_coords)]
        elif isinstance(spatial_coords, list):
            if len(spatial_coords) > 0 and isinstance(spatial_coords[0], np.ndarray):
                self.spatial_coords = [np.asarray(c) for c in spatial_coords]
            else:
                self.spatial_coords = [np.asarray(spatial_coords)]
        else:
            self.spatial_coords = [np.asarray(spatial_coords)]

        self.t_coords = np.asarray(t_coords)
        self.true_field = np.asarray(true_field)
        self.predicted_field = np.asarray(predicted_field)
        self.slice_times = np.asarray(slice_times)
        self.metadata = metadata if metadata is not None else {}

        if self.true_field.shape != self.predicted_field.shape:
            raise ValueError("'true_field' and 'predicted_field' must have identical shapes")

        expected_shape = tuple(c.size for c in self.spatial_coords) + (self.t_coords.size,)
        if self.true_field.shape != expected_shape:
            raise ValueError(
                f"Field data must have shape {expected_shape} "
                f"(spatial_coords sizes + t_coords size), got {self.true_field.shape}"
            )

        if np.any(self.slice_times < self.t_coords.min()) or np.any(self.slice_times > self.t_coords.max()):
            raise ValueError("'slice_times' must fall within the provided time coordinate range")

    @property
    def x_coords(self) -> np.ndarray:
        """Backward-compatible shortcut for the first spatial axis."""
        return self.spatial_coords[0]

    @property
    def n_spatial_dims(self) -> int:
        return len(self.spatial_coords)


@dataclass
class TermContribution:
    label: str
    values: np.ndarray
    coefficient: float

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values).reshape(-1)


@dataclass
class TermRelationshipData:
    lhs_values: np.ndarray
    lhs_label: str
    terms: List[TermContribution]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lhs_values = np.asarray(self.lhs_values).reshape(-1)
        if not self.terms:
            raise ValueError("'terms' must contain at least one contribution")
        term_length = self.lhs_values.size
        for term in self.terms:
            if term.values.size != term_length:
                raise ValueError("All term value arrays must match the length of 'lhs_values'")


@dataclass
class ParityPlotData:
    actual_values: np.ndarray
    predicted_values: np.ndarray
    residuals: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actual_values = np.asarray(self.actual_values).reshape(-1)
        self.predicted_values = np.asarray(self.predicted_values).reshape(-1)
        self.residuals = np.asarray(self.residuals).reshape(-1)

        if self.actual_values.shape != self.predicted_values.shape:
            raise ValueError("'actual_values' and 'predicted_values' must share the same shape")
        if self.actual_values.shape != self.residuals.shape:
            raise ValueError("'residuals' must match the shape of 'actual_values'")

    @classmethod
    def from_actual_predicted(
        cls,
        actual_values: Any,
        predicted_values: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ParityPlotData":
        actual_arr = np.asarray(actual_values)
        predicted_arr = np.asarray(predicted_values)
        residuals = actual_arr - predicted_arr
        return cls(
            actual_values=actual_arr,
            predicted_values=predicted_arr,
            residuals=residuals,
            metadata=metadata or {},
        )


@dataclass
class RewardEvolutionData:
    steps: np.ndarray
    best_reward: np.ndarray
    max_reward: Optional[np.ndarray] = None
    mean_reward: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.steps = np.asarray(self.steps)
        self.best_reward = np.asarray(self.best_reward)
        if self.steps.shape != self.best_reward.shape:
            raise ValueError("'steps' and 'best_reward' must have identical shapes")

        if self.max_reward is not None:
            max_arr = np.asarray(self.max_reward)
            if max_arr.shape != self.steps.shape:
                raise ValueError("'max_reward' must match the shape of 'steps'")
            self.max_reward = max_arr

        if self.mean_reward is not None:
            mean_arr = np.asarray(self.mean_reward)
            if mean_arr.shape != self.steps.shape:
                raise ValueError("'mean_reward' must match the shape of 'steps'")
            self.mean_reward = mean_arr
