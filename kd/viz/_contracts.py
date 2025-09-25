"""Shared data contracts for KD visualization intents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
    x_coords: np.ndarray
    t_coords: np.ndarray
    true_field: np.ndarray
    predicted_field: np.ndarray
    residual_field: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x_coords = np.asarray(self.x_coords)
        self.t_coords = np.asarray(self.t_coords)
        self.true_field = np.asarray(self.true_field)
        self.predicted_field = np.asarray(self.predicted_field)

        if self.true_field.shape != self.predicted_field.shape:
            raise ValueError("'true_field' and 'predicted_field' must have identical shapes")

        expected_shape = (self.x_coords.size, self.t_coords.size)
        if self.true_field.shape != expected_shape:
            raise ValueError(
                "Field data must have shape (len(x_coords), len(t_coords))"
            )

        if self.residual_field is None:
            self.residual_field = self.true_field - self.predicted_field
        else:
            residual = np.asarray(self.residual_field)
            if residual.shape != expected_shape:
                raise ValueError("'residual_field' must match the field shape")
            self.residual_field = residual


__all__ = [
    'ResidualPlotData',
    'OptimizationHistoryData',
    'FieldComparisonData',
]
