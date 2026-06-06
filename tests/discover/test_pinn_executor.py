
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd.core.expr import FunctionRegistry
from kd.data.schema import (
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.discover.pinn.executor import (
    PINNExecutor,
    make_pinn_dataset,
    make_pinn_dataset_from,
)





COORD_NAMES = ["x", "t"]
FIELD_NAMES = ["u"]
N_POINTS = 100
N_POINTS_LARGE = 1000
N_POINTS_VERY_LARGE = 10000







def _make_coords(
    n: int = N_POINTS, seed: int = 42
) -> dict[str, Tensor]:
    torch.manual_seed(seed)
    return {
        "x": torch.randn(n, dtype=torch.float32, requires_grad=True),
        "t": torch.randn(n, dtype=torch.float32, requires_grad=True),
    }


class _LinearModel(nn.Module):

    def __init__(self, a: float = 2.0, b: float = 3.0, c: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))

    def forward(self, **coords: Tensor) -> dict[str, Tensor]:
        return {"u": self.a * coords["x"] + self.b * coords["t"] + self.c}


class _QuadraticModel(nn.Module):

    def __init__(self, a: float = 1.5, b: float = -0.5) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))

    def forward(self, **coords: Tensor) -> dict[str, Tensor]:
        x, t = coords["x"], coords["t"]
        return {"u": self.a * x**2 + self.b * x * t}







@pytest.fixture
def registry() -> FunctionRegistry:
    return FunctionRegistry.create_default()


@pytest.fixture
def coords() -> dict[str, Tensor]:
    return _make_coords()


@pytest.fixture
def linear_model() -> _LinearModel:
    return _LinearModel(a=2.0, b=3.0, c=1.0)


@pytest.fixture
def quadratic_model() -> _QuadraticModel:
    return _QuadraticModel(a=1.5, b=-0.5)


@pytest.fixture
def dataset() -> PDEDataset:
    return make_pinn_dataset(
        axis_names=COORD_NAMES,
        field_names=FIELD_NAMES,
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def executor(registry: FunctionRegistry) -> PINNExecutor:
    return PINNExecutor(registry=registry)







@pytest.mark.unit
def test_make_pinn_dataset_requires_lhs_field() -> None:
    with pytest.raises(TypeError, match="lhs_field"):
        make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
        )


@pytest.mark.unit
def test_make_pinn_dataset_requires_lhs_axis() -> None:
    with pytest.raises(TypeError, match="lhs_axis"):
        make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="omega",

        )


@pytest.mark.unit
def test_make_pinn_dataset_rejects_empty_lhs_field() -> None:
    with pytest.raises(ValueError, match="lhs_field"):
        make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["omega", "u", "v"],
            lhs_field="",
            lhs_axis="t",
        )


@pytest.mark.unit
def test_make_pinn_dataset_rejects_whitespace_field_name() -> None:
    with pytest.raises(ValueError, match="field_names"):
        make_pinn_dataset(
            axis_names=["x", "t"],
            field_names=[" "],
            lhs_field=" ",
            lhs_axis="t",
        )


@pytest.mark.unit
def test_make_pinn_dataset_from_propagates_lhs() -> None:
    source = PDEDataset(
        name="ns2d_source",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axis_order=["x", "y", "t"],
        fields={
            "omega": FieldData(name="omega", values=torch.zeros(2)),
            "u": FieldData(name="u", values=torch.zeros(2)),
            "v": FieldData(name="v", values=torch.zeros(2)),
        },
        lhs_field="omega",
        lhs_axis="t",
    )

    pinn_meta = make_pinn_dataset_from(source)

    assert pinn_meta.lhs_field == "omega"
    assert pinn_meta.lhs_axis == "t"

    assert pinn_meta.axis_order == ["x", "y", "t"]
    assert pinn_meta.topology == DataTopology.SCATTERED


@pytest.mark.unit
def test_make_pinn_dataset_from_rejects_none_fields() -> None:
    source = PDEDataset(
        name="scattered_source",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
        axis_order=["x", "y", "t"],
        fields=None,
        lhs_field="omega",
        lhs_axis="t",
    )
    with pytest.raises(ValueError, match="source.fields"):
        make_pinn_dataset_from(source)


@pytest.mark.unit
def test_make_pinn_dataset_from_rejects_empty_fields() -> None:
    source = PDEDataset(
        name="empty_source",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axis_order=["x", "y", "t"],
        fields={},
        lhs_field="",
        lhs_axis="t",
    )
    with pytest.raises(ValueError, match="source.fields"):
        make_pinn_dataset_from(source)


class TestMakePinnDataset:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_scattered_topology(self) -> None:
        ds = make_pinn_dataset(
            axis_names=["x", "t"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.topology == DataTopology.SCATTERED

    @pytest.mark.unit
    def test_fields_is_none(self) -> None:
        ds = make_pinn_dataset(
            axis_names=["x", "t"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.fields is None

    @pytest.mark.unit
    def test_axis_order_matches(self) -> None:
        ds = make_pinn_dataset(
            axis_names=["x", "t"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_lhs_fields_set(self) -> None:
        ds = make_pinn_dataset(
            axis_names=["x", "t"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"

    @pytest.mark.unit
    def test_task_type_pde(self) -> None:
        ds = make_pinn_dataset(
            axis_names=["x", "t"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.task_type == TaskType.PDE

    @pytest.mark.unit
    def test_three_axis_names(self) -> None:
        ds = make_pinn_dataset(
            axis_names=["x", "y", "t"],
            field_names=["u"],
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.axis_order == ["x", "y", "t"]







class TestGradientChain:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_backward_produces_param_grads(
        self,
        executor: PINNExecutor,
        quadratic_model: _QuadraticModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        residual = executor.compute_residual(
            model=quadratic_model,
            terms=["u_xx"],
            coefficients=[1.0],
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        residual.sum().backward()

        for name, param in quadratic_model.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient"
            assert param.grad.abs().sum() > 0, (
                f"Parameter '{name}' has all-zero gradient"
            )

    @pytest.mark.unit
    def test_gradients_are_finite(
        self,
        executor: PINNExecutor,
        quadratic_model: _QuadraticModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        residual = executor.compute_residual(
            model=quadratic_model,
            terms=["u_xx"],
            coefficients=[1.0],
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        residual.sum().backward()

        for name, param in quadratic_model.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all(), (
                f"Parameter '{name}' has non-finite gradient"
            )

    @pytest.mark.unit
    def test_residual_has_grad_fn(
        self,
        executor: PINNExecutor,
        quadratic_model: _QuadraticModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        residual = executor.compute_residual(
            model=quadratic_model,
            terms=["u_xx"],
            coefficients=[1.0],
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert residual.grad_fn is not None, "Residual has no grad_fn"

    @pytest.mark.unit
    def test_backward_with_multiterm_expression(
        self,
        executor: PINNExecutor,
        quadratic_model: _QuadraticModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        residual = executor.compute_residual(
            model=quadratic_model,
            terms=["mul(u, diff_x(u))", "diff2_x(u)"],
            coefficients=[-1.0, 0.01],
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        residual.sum().backward()

        for name, param in quadratic_model.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient"
            assert torch.isfinite(param.grad).all()







class TestSingleTerm:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_u_xx_quadratic_model(
        self,
        executor: PINNExecutor,
        quadratic_model: _QuadraticModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        residual = executor.compute_residual(
            model=quadratic_model,
            terms=["u_xx"],
            coefficients=[1.0],
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        x = coords["x"]
        expected = -0.5 * x - 3.0
        assert torch.allclose(residual.detach(), expected.detach(), atol=1e-4), (
            f"Max diff: {(residual.detach() - expected.detach()).abs().max():.6f}"
        )

    @pytest.mark.unit
    def test_diff2_x_same_as_u_xx(
        self,
        executor: PINNExecutor,
        quadratic_model: _QuadraticModel,
        dataset: PDEDataset,
    ) -> None:
        coords_a = _make_coords()
        residual_terminal = executor.compute_residual(
            model=quadratic_model,
            terms=["u_xx"],
            coefficients=[1.0],
            coords=coords_a,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        coords_b = _make_coords()
        residual_openform = executor.compute_residual(
            model=quadratic_model,
            terms=["diff2_x(u)"],
            coefficients=[1.0],
            coords=coords_b,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert torch.allclose(
            residual_terminal.detach(),
            residual_openform.detach(),
            atol=1e-5,
        ), "Terminal u_xx and open-form diff2_x(u) should match"

    @pytest.mark.unit
    def test_single_diff_x_term(
        self,
        executor: PINNExecutor,
        linear_model: _LinearModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        residual = executor.compute_residual(
            model=linear_model,
            terms=["u_x"],
            coefficients=[1.0],
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        expected = torch.full((N_POINTS,), 1.0)
        assert torch.allclose(residual.detach(), expected, atol=1e-5)

    @pytest.mark.unit
    def test_plain_u_as_rhs_term(
        self,
        executor: PINNExecutor,
        quadratic_model: _QuadraticModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        residual = executor.compute_residual(
            model=quadratic_model,
            terms=["u"],
            coefficients=[1.0],
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        x, t = coords["x"], coords["t"]
        u_t = -0.5 * x
        u = 1.5 * x**2 - 0.5 * x * t
        expected = u_t - 1.0 * u
        assert torch.allclose(residual.detach(), expected.detach(), atol=1e-4)







class TestMultiTerm:

    @pytest.mark.smoke
    @pytest.mark.unit
    def test_burgers_like(
        self,
        executor: PINNExecutor,
        quadratic_model: _QuadraticModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        residual = executor.compute_residual(
            model=quadratic_model,
            terms=["mul(u, diff_x(u))", "diff2_x(u)"],
            coefficients=[-1.0, 0.01],
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        x, t = coords["x"], coords["t"]
        u = 1.5 * x**2 - 0.5 * x * t
        u_x = 3.0 * x - 0.5 * t
        u_t = -0.5 * x
        u_xx = torch.tensor(3.0)
        rhs = -1.0 * u * u_x + 0.01 * u_xx
        expected = u_t - rhs
        assert torch.allclose(residual.detach(), expected.detach(), atol=1e-4), (
            f"Max diff: {(residual.detach() - expected.detach()).abs().max():.6f}"
        )

    @pytest.mark.unit
    def test_zero_coefficient_no_effect(
        self,
        executor: PINNExecutor,
        quadratic_model: _QuadraticModel,
        dataset: PDEDataset,
    ) -> None:
        coords_a = _make_coords()
        residual_one = executor.compute_residual(
            model=quadratic_model,
            terms=["u_xx"],
            coefficients=[1.0],
            coords=coords_a,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        coords_b = _make_coords()
        residual_two = executor.compute_residual(
            model=quadratic_model,
            terms=["u_xx", "mul(u, diff_x(u))"],
            coefficients=[1.0, 0.0],
            coords=coords_b,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert torch.allclose(
            residual_one.detach(),
            residual_two.detach(),
            atol=1e-5,
        ), "Zero-coefficient term should not change residual"

    @pytest.mark.unit
    def test_coefficient_as_tensor(
        self,
        executor: PINNExecutor,
        linear_model: _LinearModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        coefs = torch.tensor([2.0], dtype=torch.float32)
        residual = executor.compute_residual(
            model=linear_model,
            terms=["u_x"],
            coefficients=coefs,
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )


        expected = torch.full((N_POINTS,), -1.0)
        assert torch.allclose(residual.detach(), expected, atol=1e-5)

    @pytest.mark.unit
    def test_negative_coefficient(
        self,
        executor: PINNExecutor,
        linear_model: _LinearModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        residual = executor.compute_residual(
            model=linear_model,
            terms=["u_x"],
            coefficients=[-1.0],
            coords=coords,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        expected = torch.full((N_POINTS,), 5.0)
        assert torch.allclose(residual.detach(), expected, atol=1e-5)







class TestShape:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n_points", [N_POINTS, N_POINTS_LARGE, N_POINTS_VERY_LARGE]
    )
    def test_residual_shape(
        self,
        executor: PINNExecutor,
        dataset: PDEDataset,
        n_points: int,
    ) -> None:
        c = _make_coords(n=n_points)
        model = _QuadraticModel()
        residual = executor.compute_residual(
            model=model,
            terms=["u_xx"],
            coefficients=[1.0],
            coords=c,
            dataset_metadata=dataset,
            lhs_field=dataset.lhs_field,
            lhs_axis=dataset.lhs_axis,
        )
        assert residual.shape == (n_points,), (
            f"Expected shape ({n_points},), got {residual.shape}"
        )







class TestAlignTermShape:

    @pytest.mark.unit
    def test_align_dim_zero_scalar_broadcasts(self) -> None:
        from kd.search.discover.pinn.executor import _align_term_shape

        reference = torch.zeros(7)
        scalar = torch.tensor(3.0)
        assert scalar.dim() == 0
        out = _align_term_shape(scalar, reference)
        assert out.shape == reference.shape
        assert torch.all(out == 3.0)

    @pytest.mark.unit
    def test_align_matching_shape_is_identity(self) -> None:
        from kd.search.discover.pinn.executor import _align_term_shape

        reference = torch.zeros(7)
        value = torch.arange(7, dtype=torch.float32)
        out = _align_term_shape(value, reference)
        assert torch.equal(out, value)

    @pytest.mark.unit
    def test_align_shape_one_tensor_broadcasts(self) -> None:
        from kd.search.discover.pinn.executor import _align_term_shape

        reference = torch.zeros(5)
        value = torch.tensor([2.5])
        assert value.shape == (1,)
        assert value.dim() == 1
        out = _align_term_shape(value, reference)
        assert out.shape == reference.shape
        assert torch.all(out == 2.5)

    @pytest.mark.unit
    def test_align_shape_one_can_stack_with_full_shape(self) -> None:
        from kd.search.discover.pinn.executor import _align_term_shape

        reference = torch.arange(4, dtype=torch.float32)
        const_term = torch.tensor([1.0])
        full_term = torch.arange(4, dtype=torch.float32) * 2
        aligned = [
            _align_term_shape(const_term, reference),
            _align_term_shape(full_term, reference),
        ]
        stacked = torch.stack(aligned, dim=0)
        assert stacked.shape == (2, 4)
        assert torch.all(stacked[0] == 1.0)
        assert torch.equal(stacked[1], full_term)

    @pytest.mark.unit
    def test_align_incompatible_shape_returns_unchanged(self) -> None:
        from kd.search.discover.pinn.executor import _align_term_shape

        reference = torch.zeros(5)
        value = torch.tensor([1.0, 2.0, 3.0])
        out = _align_term_shape(value, reference)


        assert out.shape == (3,)
        assert torch.equal(out, value)

    @pytest.mark.unit
    def test_align_multi_dim_single_element_broadcasts_to_1d(self) -> None:
        from kd.search.discover.pinn.executor import _align_term_shape

        reference = torch.zeros(4)
        value = torch.tensor([[7.5]])
        assert value.shape == (1, 1)
        assert value.numel() == 1
        out = _align_term_shape(value, reference)
        assert out.shape == reference.shape
        assert torch.all(out == 7.5)

    @pytest.mark.unit
    def test_align_triple_nested_single_element_broadcasts(self) -> None:
        from kd.search.discover.pinn.executor import _align_term_shape

        reference = torch.zeros(3)
        value = torch.tensor([[[2.0]]])
        assert value.numel() == 1
        out = _align_term_shape(value, reference)
        assert out.shape == reference.shape
        assert torch.all(out == 2.0)

    @pytest.mark.unit
    def test_align_multi_dim_single_element_preserves_grad(self) -> None:
        from kd.search.discover.pinn.executor import _align_term_shape

        reference = torch.zeros(4)
        source = torch.tensor(3.0, requires_grad=True)
        value = source.reshape(1, 1)
        out = _align_term_shape(value, reference)
        assert out.shape == reference.shape
        out.sum().backward()
        assert source.grad is not None
        assert float(source.grad) == pytest.approx(4.0)







class TestErrorHandling:

    @pytest.mark.unit
    def test_empty_terms_raises(
        self,
        executor: PINNExecutor,
        linear_model: _LinearModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        with pytest.raises(ValueError, match="[Ee]mpty|[Tt]erms"):
            executor.compute_residual(
                model=linear_model,
                terms=[],
                coefficients=[],
                coords=coords,
                dataset_metadata=dataset,
                lhs_field=dataset.lhs_field,
                lhs_axis=dataset.lhs_axis,
            )

    @pytest.mark.unit
    def test_no_requires_grad_raises(
        self,
        executor: PINNExecutor,
        linear_model: _LinearModel,
        dataset: PDEDataset,
    ) -> None:
        bad_coords = {
            "x": torch.randn(50, dtype=torch.float32),
            "t": torch.randn(50, dtype=torch.float32),
        }
        with pytest.raises(ValueError, match="requires_grad"):
            executor.compute_residual(
                model=linear_model,
                terms=["u_x"],
                coefficients=[1.0],
                coords=bad_coords,
                dataset_metadata=dataset,
                lhs_field=dataset.lhs_field,
                lhs_axis=dataset.lhs_axis,
            )

    @pytest.mark.unit
    def test_mismatched_lengths_raises(
        self,
        executor: PINNExecutor,
        linear_model: _LinearModel,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        with pytest.raises(ValueError, match="[Ll]ength|[Mm]ismatch|[Cc]oefficient"):
            executor.compute_residual(
                model=linear_model,
                terms=["u_x", "u_xx"],
                coefficients=[1.0],
                coords=coords,
                dataset_metadata=dataset,
                lhs_field=dataset.lhs_field,
                lhs_axis=dataset.lhs_axis,
            )







class _Cubic2DModel(nn.Module):

    def __init__(
        self, a: float = 1.5, b: float = -0.7, c: float = 0.3
    ) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))

    def forward(self, **coords: Tensor) -> dict[str, Tensor]:
        x, y, t = coords["x"], coords["y"], coords["t"]
        return {
            "u": self.a * x**2 * y + self.b * x * y**2 + self.c * x * y * t
        }


def _make_3d_coords(
    n: int = N_POINTS, seed: int = 42
) -> dict[str, Tensor]:
    torch.manual_seed(seed)
    return {
        "x": torch.randn(n, dtype=torch.float32, requires_grad=True),
        "y": torch.randn(n, dtype=torch.float32, requires_grad=True),
        "t": torch.randn(n, dtype=torch.float32, requires_grad=True),
    }


@pytest.fixture
def cubic_2d_model() -> _Cubic2DModel:
    return _Cubic2DModel(a=1.5, b=-0.7, c=0.3)


@pytest.fixture
def dataset_3d() -> PDEDataset:
    return make_pinn_dataset(
        axis_names=["x", "y", "t"],
        field_names=["u"],
        lhs_field="u",
        lhs_axis="t",
    )


class TestMixedPartials:

    @pytest.mark.unit
    def test_diff_y_diff_x_matches_analytical(
        self,
        executor: PINNExecutor,
        cubic_2d_model: _Cubic2DModel,
        dataset_3d: PDEDataset,
    ) -> None:
        coords = _make_3d_coords()
        residual = executor.compute_residual(
            model=cubic_2d_model,
            terms=["diff_y(diff_x(u))"],
            coefficients=[1.0],
            coords=coords,
            dataset_metadata=dataset_3d,
            lhs_field=dataset_3d.lhs_field,
            lhs_axis=dataset_3d.lhs_axis,
        )
        x = coords["x"].detach()
        y = coords["y"].detach()
        t = coords["t"].detach()
        u_t = 0.3 * x * y
        u_xy = 3.0 * x - 1.4 * y + 0.3 * t
        expected = u_t - 1.0 * u_xy
        assert torch.allclose(residual.detach(), expected, atol=1e-4), (
            f"Max diff: {(residual.detach() - expected).abs().max():.6f}"
        )

    @pytest.mark.unit
    def test_schwarz_symmetry(
        self,
        executor: PINNExecutor,
        cubic_2d_model: _Cubic2DModel,
        dataset_3d: PDEDataset,
    ) -> None:
        coords_a = _make_3d_coords()
        residual_yx = executor.compute_residual(
            model=cubic_2d_model,
            terms=["diff_y(diff_x(u))"],
            coefficients=[1.0],
            coords=coords_a,
            dataset_metadata=dataset_3d,
            lhs_field=dataset_3d.lhs_field,
            lhs_axis=dataset_3d.lhs_axis,
        )
        coords_b = _make_3d_coords()
        residual_xy = executor.compute_residual(
            model=cubic_2d_model,
            terms=["diff_x(diff_y(u))"],
            coefficients=[1.0],
            coords=coords_b,
            dataset_metadata=dataset_3d,
            lhs_field=dataset_3d.lhs_field,
            lhs_axis=dataset_3d.lhs_axis,
        )
        assert torch.allclose(
            residual_yx.detach(), residual_xy.detach(), atol=1e-5
        ), "Schwarz theorem: u_xy must equal u_yx"

    @pytest.mark.unit
    def test_mixed_partial_backward(
        self,
        executor: PINNExecutor,
        cubic_2d_model: _Cubic2DModel,
        dataset_3d: PDEDataset,
    ) -> None:
        coords = _make_3d_coords()
        residual = executor.compute_residual(
            model=cubic_2d_model,
            terms=["diff_y(diff_x(u))"],
            coefficients=[1.0],
            coords=coords,
            dataset_metadata=dataset_3d,
            lhs_field=dataset_3d.lhs_field,
            lhs_axis=dataset_3d.lhs_axis,
        )
        loss = residual.square().mean()
        loss.backward()
        for name, param in cubic_2d_model.named_parameters():
            assert param.grad is not None, (
                f"Parameter {name} has None gradient after backward through u_xy"
            )
