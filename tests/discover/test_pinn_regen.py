
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kd.core.evaluator import Evaluator
from kd.core.expr import (
    FunctionRegistry,
    PythonExecutor,
)
from kd.core.linear_solve import LeastSquaresSolver
from kd.data.derivatives.autograd import (
    AutogradProvider,
)
from kd.data.schema import PDEDataset
from kd.search.discover.config import PINNConfig
from kd.search.discover.pinn.cycle import (
    rebuild_evaluator,
    regenerate_metadata,
)
from kd.search.discover.pinn.executor import make_pinn_dataset
from kd.search.discover.pinn.model import PINNModel





COORD_NAMES = ["x", "t"]
FIELD_NAMES = ["u"]

_CPU_DEVICE = torch.device("cpu")
N_COLLOC = 500
N_COLLOC_SMALL = 100

SMALL_CONFIG = PINNConfig(
    number_layer=2,
    n_hidden=10,
    activation="tanh",
    pretrain_epoch=200,
    lr=0.01,
    early_stop_patience=50,
)







def _heat_solution(x: Tensor, t: Tensor) -> Tensor:
    return torch.exp(-torch.pi**2 * t) * torch.sin(torch.pi * x)


def _make_colloc_coords(
    n: int = N_COLLOC,
    device: torch.device = _CPU_DEVICE,
    seed: int = 42,
) -> dict[str, Tensor]:
    torch.manual_seed(seed)
    return {
        "x": (torch.rand(n, device=device) * 2 - 1).detach().requires_grad_(True),
        "t": torch.rand(n, device=device).abs().detach().requires_grad_(True),
    }


def _pretrain_model(
    config: PINNConfig = SMALL_CONFIG,
    device: torch.device = _CPU_DEVICE,
) -> PINNModel:
    torch.manual_seed(42)
    model = PINNModel(COORD_NAMES, FIELD_NAMES, config, device)
    n_train, n_val = 200, 50
    x_train = torch.rand(n_train, device=device) * 2 - 1
    t_train = torch.rand(n_train, device=device).abs()
    u_train = _heat_solution(x_train, t_train)
    x_val = torch.rand(n_val, device=device) * 2 - 1
    t_val = torch.rand(n_val, device=device).abs()
    u_val = _heat_solution(x_val, t_val)
    model.pretrain(
        {"x": x_train, "t": t_train}, {"u": u_train},
        {"x": x_val, "t": t_val}, {"u": u_val},
        config,
    )
    return model







@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def pretrained_model(device: torch.device) -> PINNModel:
    return _pretrain_model(device=device)


@pytest.fixture
def dataset() -> PDEDataset:
    return make_pinn_dataset(
        axis_names=COORD_NAMES,
        field_names=FIELD_NAMES,
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def colloc_coords(device: torch.device) -> dict[str, Tensor]:
    return _make_colloc_coords(device=device)


@pytest.fixture
def registry() -> FunctionRegistry:
    return FunctionRegistry.create_default()


@pytest.fixture
def py_executor(registry: FunctionRegistry) -> PythonExecutor:
    return PythonExecutor(registry)


@pytest.fixture
def solver() -> LeastSquaresSolver:
    return LeastSquaresSolver()







class TestRegenerateMetadataLHS:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_lhs_shape_matches_collocation(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert regen.lhs_detached.shape == (N_COLLOC,), (
            f"Expected ({N_COLLOC},), got {regen.lhs_detached.shape}"
        )

    @pytest.mark.unit
    def test_lhs_finite(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert torch.isfinite(regen.lhs_detached).all(), (
            f"LHS has {(~torch.isfinite(regen.lhs_detached)).sum()} "
            f"non-finite values"
        )

    @pytest.mark.unit
    def test_lhs_is_detached(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert regen.lhs_detached.grad_fn is None, (
            "LHS should be detached (no grad_fn) — it's a regression target"
        )
        assert not regen.lhs_detached.requires_grad, (
            "LHS should not require grad"
        )

    @pytest.mark.unit
    def test_lhs_is_1d(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert regen.lhs_detached.dim() == 1, (
            f"LHS should be 1D, got {regen.lhs_detached.dim()}D"
        )

    @pytest.mark.unit
    def test_lhs_is_nontrivial(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert regen.lhs_detached.abs().max() > 1e-6, (
            "LHS is all-zero — autograd derivative computation may be broken"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("n_points", [N_COLLOC_SMALL, N_COLLOC])
    def test_lhs_shape_varies_with_n_colloc(
        self,
        pretrained_model: PINNModel,
        dataset: PDEDataset,
        n_points: int,
    ) -> None:
        coords = _make_colloc_coords(n=n_points)
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert regen.lhs_detached.shape == (n_points,)







class TestRegenerateMetadataProvider:

    @pytest.mark.unit
    def test_provider_is_autograd_provider(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert isinstance(regen.provider, AutogradProvider)

    @pytest.mark.unit
    def test_provider_can_compute_derivatives(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )

        u_x = regen.provider.get_derivative("u", "x", 1)
        assert u_x.shape == (N_COLLOC,)
        assert torch.isfinite(u_x).all()

    @pytest.mark.unit
    def test_provider_diff_works(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        u_field = regen.provider.get_field("u")
        u_x = regen.provider.diff(u_field, "x", 1)
        assert u_x.shape == (N_COLLOC,)
        assert torch.isfinite(u_x).all()







class TestRegenerateMetadataCoords:

    @pytest.mark.unit
    def test_colloc_coords_have_requires_grad(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        for name, tensor in regen.colloc_coords.items():
            assert tensor.requires_grad, (
                f"Coord '{name}' missing requires_grad=True"
            )

    @pytest.mark.unit
    def test_returned_coords_are_fresh(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        for name in colloc_coords:
            assert regen.colloc_coords[name] is not colloc_coords[name], (
                f"Coord '{name}' is same object as input — "
                "should be a fresh clone"
            )

    @pytest.mark.unit
    def test_returned_dataset_matches_input(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert regen.dataset_metadata is dataset







class TestRegenerateMetadataCustomLHS:

    @pytest.mark.unit
    def test_explicit_lhs_is_u_t(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )

        assert regen.lhs_detached.abs().max() > 1e-6







class TestRegeneratedData:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_has_expected_fields(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert hasattr(regen, "lhs_detached")
        assert hasattr(regen, "provider")
        assert hasattr(regen, "colloc_coords")
        assert hasattr(regen, "dataset_metadata")
        assert isinstance(regen.lhs_detached, Tensor)
        assert isinstance(regen.colloc_coords, dict)







class TestRebuildEvaluator:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_returns_evaluator(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
        py_executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        evaluator = rebuild_evaluator(regen, py_executor, solver)
        assert isinstance(evaluator, Evaluator)

    @pytest.mark.unit
    def test_evaluator_produces_valid_result(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
        py_executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        evaluator = rebuild_evaluator(regen, py_executor, solver)

        result = evaluator.evaluate_terms(["diff2_x(u)"])
        assert result.is_valid, (
            f"Expected valid result for heat equation term, "
            f"got error: {result.error_message}"
        )

    @pytest.mark.unit
    def test_evaluator_expression_valid(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
        py_executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        evaluator = rebuild_evaluator(regen, py_executor, solver)


        result = evaluator.evaluate_expression("diff2_x(u)")
        assert result.is_valid

    @pytest.mark.unit
    def test_evaluator_coefficients_reasonable(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
        py_executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        evaluator = rebuild_evaluator(regen, py_executor, solver)

        result = evaluator.evaluate_terms(["diff2_x(u)"])
        assert result.coefficients is not None
        coef = result.coefficients[0].item()


        assert 0.01 < abs(coef) < 100.0, (
            f"Coefficient {coef} is unreasonable for heat equation"
        )

    @pytest.mark.unit
    def test_evaluator_has_correct_lhs(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
        py_executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        evaluator = rebuild_evaluator(regen, py_executor, solver)


        assert torch.allclose(
            evaluator.lhs_target,
            regen.lhs_detached,
            atol=1e-6,
        ), "Evaluator LHS target doesn't match regenerated LHS"







class TestIntegrationRoundTrip:

    @pytest.mark.unit
    def test_roundtrip_heat_equation(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
        py_executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        evaluator = rebuild_evaluator(regen, py_executor, solver)

        result = evaluator.evaluate_terms(["diff2_x(u)"])

        assert result.is_valid, f"Evaluation failed: {result.error_message}"
        assert result.nmse < 0.4, (
            f"NMSE={result.nmse:.4f} too high — pipeline may be broken"
        )

        assert result.coefficients is not None
        coef = result.coefficients[0].item()
        assert coef > 0, (
            f"Heat equation coefficient should be positive, got {coef}"
        )

    @pytest.mark.unit
    def test_roundtrip_burgers_multiterm(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
        py_executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        evaluator = rebuild_evaluator(regen, py_executor, solver)

        result = evaluator.evaluate_terms(
            ["mul(u, diff_x(u))", "diff2_x(u)"]
        )

        assert result.is_valid, f"Multi-term evaluation failed: {result.error_message}"
        assert result.coefficients is not None
        assert result.coefficients.shape[0] == 2

    @pytest.mark.unit
    def test_roundtrip_evaluator_uses_autograd_provider(
        self,
        pretrained_model: PINNModel,
        colloc_coords: dict[str, Tensor],
        dataset: PDEDataset,
        py_executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        regen = regenerate_metadata(
            model=pretrained_model,
            colloc_coords=colloc_coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        evaluator = rebuild_evaluator(regen, py_executor, solver)


        result = evaluator.evaluate_terms(["diff_x(u)"])
        assert result.is_valid, (
            f"diff_x(u) evaluation failed — AutogradProvider may not be "
            f"connected: {result.error_message}"
        )
