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
