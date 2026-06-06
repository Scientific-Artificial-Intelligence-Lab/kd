
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from kd.data.derivatives.base import DerivativeProvider
from kd.data.schema import PDEDataset

logger = logging.getLogger(__name__)


_DEFAULT_MAX_ORDER = 3


class AutogradProvider(DerivativeProvider):

    def __init__(
        self,
        model: nn.Module,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
        max_order: int = _DEFAULT_MAX_ORDER,
    ) -> None:

        if max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {max_order}")

        self._validate_coords(coords, dataset)

        self.model = model
        self._coords = coords
        self.dataset = dataset
        self._max_order = max_order



        self.device: torch.device = (
            next(iter(coords.values())).device if coords else torch.device("cpu")
        )

    @property
    def coords(self) -> dict[str, Tensor]:
        return self._coords

    @staticmethod
    def _validate_coords(
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:

        for name, tensor in coords.items():
            if not tensor.requires_grad:
                raise ValueError(
                    f"Coordinate '{name}' must have requires_grad=True. "
                    f"Use tensor.requires_grad_(True) or pass "
                    f"requires_grad=True at creation."
                )


        if dataset.axis_order is not None:
            expected = set(dataset.axis_order)
            actual = set(coords.keys())
            missing = expected - actual
            if missing:
                raise ValueError(
                    f"Coords missing required axes from dataset: {missing}. "
                    f"Coords have: {actual}, dataset needs: {expected}"
                )

    def _forward_model(self) -> dict[str, Tensor]:

        with torch.enable_grad():
            output = self.model(**self._coords)


        if isinstance(output, dict):
            return output

        return {self.dataset.lhs_field: output}

    def get_field(self, name: str) -> Tensor:
        fields = self._forward_model()
        if name not in fields:
            raise KeyError(
                f"Field '{name}' not found. Available fields: {list(fields.keys())}"
            )
        return fields[name]

    def diff(
        self,
        expression: Tensor,
        axis: str,
        order: int,
    ) -> Tensor:

        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")





        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if order > self._max_order:
            raise ValueError(
                f"order {order} exceeds max_order {self._max_order}; "
                f"pass max_order=... at AutogradProvider construction to raise"
            )


        if axis not in self._coords:
            raise KeyError(
                f"Axis '{axis}' not found in coords. "
                f"Available axes: {list(self._coords.keys())}"
            )

        coord = self._coords[axis]
        result = expression

        for _step in range(order):

            with torch.enable_grad():
                try:
                    (result,) = torch.autograd.grad(
                        outputs=result,
                        inputs=coord,
                        grad_outputs=torch.ones_like(result),
                        create_graph=True,
                        retain_graph=True,
                    )
                except RuntimeError as e:
                    msg = str(e)
                    if (
                        "does not require grad" in msg
                        or "One of the differentiated Tensors" in msg
                    ):
                        raise ValueError(
                            f"Cannot compute derivative: expression is "
                            f"not connected to coordinate '{axis}' in "
                            f"the computation graph. Ensure the expression "
                            f"depends on coord '{axis}' via the model."
                        ) from e
                    raise

        return result

    def get_derivative(
        self,
        field: str,
        axis: str,
        order: int,
    ) -> Tensor:

        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")


        if axis not in self._coords:
            raise KeyError(
                f"Axis '{axis}' not found in coords. "
                f"Available axes: {list(self._coords.keys())}"
            )


        with torch.enable_grad():
            field_tensor = self.get_field(field)
            return self.diff(field_tensor, axis, order)

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        result: list[tuple[str, str, int]] = []


        field_names: list[str] = []
        if self.dataset.fields is not None:
            field_names = list(self.dataset.fields.keys())
        elif self.dataset.lhs_field:
            field_names = [self.dataset.lhs_field]

        axis_names = list(self._coords.keys())

        for field_name in field_names:
            for axis_name in axis_names:
                for order in range(1, self._max_order + 1):
                    result.append((field_name, axis_name, order))

        return result
