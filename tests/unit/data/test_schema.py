
import pytest
import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
    compute_dataset_fingerprint,
)






@pytest.fixture
def sample_axes() -> dict[str, AxisInfo]:
    x_values = torch.linspace(-1, 1, 64)
    t_values = torch.linspace(0, 1, 51)
    return {
        "x": AxisInfo(name="x", values=x_values, is_periodic=True),
        "t": AxisInfo(name="t", values=t_values, is_periodic=False),
    }


@pytest.fixture
def sample_fields(sample_axes: dict[str, AxisInfo]) -> dict[str, FieldData]:
    nx = sample_axes["x"].values.shape[0]
    nt = sample_axes["t"].values.shape[0]
    u_values = torch.randn(nx, nt)
    return {"u": FieldData(name="u", values=u_values)}


@pytest.fixture
def sample_dataset(
    sample_axes: dict[str, AxisInfo],
    sample_fields: dict[str, FieldData],
) -> PDEDataset:
    return PDEDataset(
        name="test_burgers",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes=sample_axes,
        axis_order=["x", "t"],
        fields=sample_fields,
        lhs_field="u",
        lhs_axis="t",
    )







class TestSchemaSmoke:

    @pytest.mark.smoke
    def test_enums_exist(self) -> None:
        assert TaskType.PDE.value == "pde"
        assert TaskType.ODE.value == "ode"
        assert TaskType.REGRESSION.value == "regression"
        assert DataTopology.GRID.value == "grid"
        assert DataTopology.SCATTERED.value == "scattered"

    @pytest.mark.smoke
    def test_axis_info_creation(self) -> None:
        values = torch.linspace(0, 1, 10)
        axis = AxisInfo(name="x", values=values)
        assert axis.name == "x"
        assert axis.values.shape == (10,)
        assert axis.is_periodic is False

    @pytest.mark.smoke
    def test_field_data_creation(self) -> None:
        values = torch.randn(10, 20)
        field = FieldData(name="u", values=values)
        assert field.name == "u"
        assert field.values.shape == (10, 20)

    @pytest.mark.smoke
    def test_pde_dataset_creation(self, sample_dataset: PDEDataset) -> None:
        assert sample_dataset.name == "test_burgers"
        assert sample_dataset.task_type == TaskType.PDE
        assert sample_dataset.topology == DataTopology.GRID
        assert sample_dataset.axes is not None
        assert sample_dataset.fields is not None

    @pytest.mark.smoke
    def test_compute_fingerprint_exists(self) -> None:
        assert callable(compute_dataset_fingerprint)







class TestAxisInfo:

    @pytest.mark.unit
    def test_axis_info_with_periodic(self) -> None:
        values = torch.linspace(-1, 1, 100)
        axis = AxisInfo(name="x", values=values, is_periodic=True)
        assert axis.is_periodic is True

    @pytest.mark.unit
    def test_axis_info_values_preserved(self) -> None:
        values = torch.tensor([0.0, 0.5, 1.0])
        axis = AxisInfo(name="t", values=values)
        torch.testing.assert_close(axis.values, values)

    @pytest.mark.unit
    def test_axis_info_rejects_2d_values(self) -> None:
        values_2d = torch.randn(10, 10)
        with pytest.raises(ValueError, match="must be 1D"):
            AxisInfo(name="bad", values=values_2d)

    @pytest.mark.unit
    def test_axis_info_rejects_3d_values(self) -> None:
        values_3d = torch.randn(5, 5, 5)
        with pytest.raises(ValueError, match="must be 1D"):
            AxisInfo(name="bad", values=values_3d)

    @pytest.mark.unit
    def test_axis_info_rejects_0d_values(self) -> None:
        values_0d = torch.tensor(1.0)
        with pytest.raises(ValueError, match="must be 1D"):
            AxisInfo(name="bad", values=values_0d)

    @pytest.mark.unit
    def test_axis_info_rejects_empty_values(self) -> None:
        values_empty = torch.tensor([])
        with pytest.raises(ValueError, match="must not be empty"):
            AxisInfo(name="bad", values=values_empty)

    @pytest.mark.unit
    def test_axis_info_rejects_nan_values(self) -> None:
        values_nan = torch.tensor([0.0, float("nan"), 1.0])
        with pytest.raises(ValueError, match="NaN"):
            AxisInfo(name="bad", values=values_nan)

    @pytest.mark.unit
    def test_axis_info_rejects_inf_values(self) -> None:
        values_inf = torch.tensor([float("inf"), 0.0, 1.0])
        with pytest.raises(ValueError, match="Inf"):
            AxisInfo(name="bad", values=values_inf)

    @pytest.mark.unit
    def test_axis_info_rejects_negative_inf_values(self) -> None:
        values_neg_inf = torch.tensor([0.0, float("-inf"), 1.0])
        with pytest.raises(ValueError, match="Inf"):
            AxisInfo(name="bad", values=values_neg_inf)







class TestFieldDataValidation:

    @pytest.mark.unit
    def test_field_data_rejects_0d_values(self) -> None:
        values_0d = torch.tensor(1.0)
        with pytest.raises(ValueError, match="must be at least 1D"):
            FieldData(name="u", values=values_0d)

    @pytest.mark.unit
    def test_field_data_rejects_empty_values(self) -> None:
        values_empty = torch.tensor([])
        with pytest.raises(ValueError, match="must not be empty"):
            FieldData(name="u", values=values_empty)

    @pytest.mark.unit
    def test_field_data_rejects_nan_values(self) -> None:
        values_nan = torch.tensor([[0.0, float("nan")], [1.0, 2.0]])
        with pytest.raises(ValueError, match="NaN"):
            FieldData(name="u", values=values_nan)

    @pytest.mark.unit
    def test_field_data_rejects_inf_values(self) -> None:
        values_inf = torch.tensor([[float("inf"), 0.0], [1.0, 2.0]])
        with pytest.raises(ValueError, match="Inf"):
            FieldData(name="u", values=values_inf)

    @pytest.mark.unit
    def test_field_data_rejects_negative_inf_values(self) -> None:
        values_neg_inf = torch.tensor([[0.0, float("-inf")], [1.0, 2.0]])
        with pytest.raises(ValueError, match="Inf"):
            FieldData(name="u", values=values_neg_inf)







class TestPDEDatasetMethods:

    @pytest.mark.unit
    def test_get_shape_returns_correct_tuple(self, sample_dataset: PDEDataset) -> None:
        shape = sample_dataset.get_shape()
        assert shape == (64, 51)

    @pytest.mark.unit
    def test_get_shape_matches_axis_order(self) -> None:

        axes = {
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 30)),
            "x": AxisInfo(name="x", values=torch.linspace(-1, 1, 50)),
        }
        fields = {"u": FieldData(name="u", values=torch.randn(50, 30))}
        dataset = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x", "t"],
            fields=fields,
        )
        shape = dataset.get_shape()
        assert shape == (50, 30)

    @pytest.mark.unit
    def test_spatial_axes_excludes_lhs_axis(self, sample_dataset: PDEDataset) -> None:
        assert sample_dataset.spatial_axes == ["x"]

    @pytest.mark.unit
    def test_spatial_axes_preserves_axis_order(self) -> None:
        dataset = PDEDataset(
            name="test_3d",
            task_type=TaskType.PDE,
            axis_order=["t", "x", "y", "z"],
            lhs_axis="t",
        )

        assert dataset.spatial_axes == ["x", "y", "z"]

    @pytest.mark.unit
    def test_spatial_axes_empty_without_lhs_axis(self) -> None:
        dataset = PDEDataset(
            name="test_no_lhs",
            task_type=TaskType.PDE,
            axis_order=["x", "t"],
            lhs_axis="",
        )

        assert dataset.spatial_axes == []

    @pytest.mark.unit
    def test_spatial_axes_empty_without_axis_order(self) -> None:
        dataset = PDEDataset(
            name="test_no_axis_order",
            task_type=TaskType.PDE,
            axis_order=None,
            lhs_axis="t",
        )

        assert dataset.spatial_axes == []

    @pytest.mark.unit
    def test_get_coords_returns_correct_values(
        self, sample_dataset: PDEDataset
    ) -> None:
        x_coords = sample_dataset.get_coords("x")
        expected = torch.linspace(-1, 1, 64)
        torch.testing.assert_close(x_coords, expected)

    @pytest.mark.unit
    def test_get_coords_t_axis(self, sample_dataset: PDEDataset) -> None:
        t_coords = sample_dataset.get_coords("t")
        expected = torch.linspace(0, 1, 51)
        torch.testing.assert_close(t_coords, expected)

    @pytest.mark.unit
    def test_get_coords_raises_on_missing_axis(
        self, sample_dataset: PDEDataset
    ) -> None:
        with pytest.raises(KeyError):
            sample_dataset.get_coords("y")

    @pytest.mark.unit
    def test_get_field_returns_correct_values(self, sample_dataset: PDEDataset) -> None:
        u_field = sample_dataset.get_field("u")
        assert u_field.shape == (64, 51)

    @pytest.mark.unit
    def test_get_field_raises_on_missing_field(
        self, sample_dataset: PDEDataset
    ) -> None:
        with pytest.raises(KeyError):
            sample_dataset.get_field("v")







class TestPDEDatasetEdgeCases:

    @pytest.mark.unit
    def test_empty_fields_dict(self) -> None:
        axes = {"x": AxisInfo(name="x", values=torch.linspace(0, 1, 10))}
        dataset = PDEDataset(
            name="empty_fields",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x"],
            fields={},
        )
        with pytest.raises(KeyError):
            dataset.get_field("u")

    @pytest.mark.unit
    def test_single_point_axis(self) -> None:
        axes = {
            "x": AxisInfo(name="x", values=torch.tensor([0.0])),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 10)),
        }
        fields = {"u": FieldData(name="u", values=torch.randn(1, 10))}
        dataset = PDEDataset(
            name="single_point",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x", "t"],
            fields=fields,
        )
        assert dataset.get_shape() == (1, 10)

    @pytest.mark.unit
    def test_3d_dataset(self) -> None:
        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(-1, 1, 32)),
            "y": AxisInfo(name="y", values=torch.linspace(-1, 1, 32)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 21)),
        }
        fields = {"u": FieldData(name="u", values=torch.randn(32, 32, 21))}
        dataset = PDEDataset(
            name="3d_test",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x", "y", "t"],
            fields=fields,
        )
        assert dataset.get_shape() == (32, 32, 21)
        assert dataset.get_field("u").shape == (32, 32, 21)

    @pytest.mark.unit
    def test_multiple_fields(self) -> None:
        axes = {"x": AxisInfo(name="x", values=torch.linspace(0, 1, 20))}
        fields = {
            "u": FieldData(name="u", values=torch.randn(20)),
            "v": FieldData(name="v", values=torch.randn(20)),
            "p": FieldData(name="p", values=torch.randn(20)),
        }
        dataset = PDEDataset(
            name="multi_field",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x"],
            fields=fields,
        )
        assert dataset.get_field("u").shape == (20,)
        assert dataset.get_field("v").shape == (20,)
        assert dataset.get_field("p").shape == (20,)







class TestPDEDatasetValidation:

    @pytest.mark.unit
    def test_axis_order_rejects_duplicates(self) -> None:
        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(0, 1, 10)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 20)),
        }
        fields = {"u": FieldData(name="u", values=torch.randn(10, 20))}
        with pytest.raises(ValueError, match="duplicate"):
            PDEDataset(
                name="test",
                task_type=TaskType.PDE,
                axes=axes,
                axis_order=["x", "x"],
                fields=fields,
            )

    @pytest.mark.unit
    def test_axis_order_rejects_duplicates_three_axes(self) -> None:
        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(0, 1, 10)),
            "y": AxisInfo(name="y", values=torch.linspace(0, 1, 15)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 20)),
        }
        fields = {"u": FieldData(name="u", values=torch.randn(10, 15, 20))}
        with pytest.raises(ValueError, match="duplicate"):
            PDEDataset(
                name="test",
                task_type=TaskType.PDE,
                axes=axes,
                axis_order=["x", "y", "x"],
                fields=fields,
            )

    @pytest.mark.unit
    def test_axis_order_with_missing_axis_raises(self) -> None:
        axes = {"x": AxisInfo(name="x", values=torch.linspace(0, 1, 10))}
        fields = {"u": FieldData(name="u", values=torch.randn(10, 20))}
        with pytest.raises(ValueError, match="axis_order.*not in axes"):
            PDEDataset(
                name="mismatch",
                task_type=TaskType.PDE,
                axes=axes,
                axis_order=["x", "t"],
                fields=fields,
            )

    @pytest.mark.unit
    def test_axes_not_in_axis_order_raises(self) -> None:
        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(0, 1, 10)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 20)),
        }
        fields = {"u": FieldData(name="u", values=torch.randn(10))}
        with pytest.raises(ValueError, match="axes.*not in axis_order"):
            PDEDataset(
                name="mismatch",
                task_type=TaskType.PDE,
                axes=axes,
                axis_order=["x"],
                fields=fields,
            )

    @pytest.mark.unit
    def test_field_shape_dimension_mismatch_raises(self) -> None:
        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(0, 1, 10)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 20)),
        }

        fields = {"u": FieldData(name="u", values=torch.randn(10))}
        with pytest.raises(ValueError, match="field.*dimension.*mismatch"):
            PDEDataset(
                name="mismatch",
                task_type=TaskType.PDE,
                axes=axes,
                axis_order=["x", "t"],
                fields=fields,
            )

    @pytest.mark.unit
    def test_field_shape_size_mismatch_raises(self) -> None:
        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(0, 1, 10)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 20)),
        }

        fields = {"u": FieldData(name="u", values=torch.randn(10, 30))}
        with pytest.raises(ValueError, match="field.*shape.*mismatch"):
            PDEDataset(
                name="mismatch",
                task_type=TaskType.PDE,
                axes=axes,
                axis_order=["x", "t"],
                fields=fields,
            )

    @pytest.mark.unit
    def test_field_shape_wrong_order_mismatch_raises(self) -> None:
        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(0, 1, 10)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 20)),
        }

        fields = {"u": FieldData(name="u", values=torch.randn(20, 10))}
        with pytest.raises(ValueError, match="field.*shape.*mismatch"):
            PDEDataset(
                name="mismatch",
                task_type=TaskType.PDE,
                axes=axes,
                axis_order=["x", "t"],
                fields=fields,
            )







class TestOtherTaskTypes:

    @pytest.mark.unit
    def test_ode_dataset_creation(self) -> None:

        axes = {"t": AxisInfo(name="t", values=torch.linspace(0, 10, 101))}
        fields = {
            "x": FieldData(name="x", values=torch.randn(101)),
            "y": FieldData(name="y", values=torch.randn(101)),
        }
        dataset = PDEDataset(
            name="lotka_volterra",
            task_type=TaskType.ODE,
            axes=axes,
            axis_order=["t"],
            fields=fields,
            lhs_axis="t",
        )
        assert dataset.task_type == TaskType.ODE
        assert dataset.get_shape() == (101,)
        assert dataset.get_field("x").shape == (101,)
        assert dataset.get_field("y").shape == (101,)

    @pytest.mark.unit
    def test_regression_dataset_creation(self) -> None:

        axes = {"samples": AxisInfo(name="samples", values=torch.arange(100).float())}
        fields = {
            "x1": FieldData(name="x1", values=torch.randn(100)),
            "x2": FieldData(name="x2", values=torch.randn(100)),
            "y": FieldData(name="y", values=torch.randn(100)),
        }
        dataset = PDEDataset(
            name="regression_test",
            task_type=TaskType.REGRESSION,
            axes=axes,
            axis_order=["samples"],
            fields=fields,
            lhs_field="y",
        )
        assert dataset.task_type == TaskType.REGRESSION
        assert dataset.get_shape() == (100,)
        assert dataset.get_field("y").shape == (100,)







class TestDatasetFingerprint:

    @pytest.mark.unit
    def test_fingerprint_is_string(self, sample_dataset: PDEDataset) -> None:
        fp = compute_dataset_fingerprint(sample_dataset)
        assert isinstance(fp, str)

    @pytest.mark.unit
    def test_fingerprint_stability(self, sample_dataset: PDEDataset) -> None:
        fp1 = compute_dataset_fingerprint(sample_dataset)
        fp2 = compute_dataset_fingerprint(sample_dataset)
        assert fp1 == fp2

    @pytest.mark.unit
    def test_fingerprint_different_name(self) -> None:
        axes = {"x": AxisInfo(name="x", values=torch.linspace(0, 1, 10))}
        fields = {"u": FieldData(name="u", values=torch.ones(10))}

        ds1 = PDEDataset(
            name="dataset_a",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x"],
            fields=fields,
        )
        ds2 = PDEDataset(
            name="dataset_b",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x"],
            fields=fields,
        )
        fp1 = compute_dataset_fingerprint(ds1)
        fp2 = compute_dataset_fingerprint(ds2)
        assert fp1 != fp2

    @pytest.mark.unit
    def test_fingerprint_different_data(self) -> None:
        axes = {"x": AxisInfo(name="x", values=torch.linspace(0, 1, 10))}

        ds1 = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=torch.ones(10))},
        )
        ds2 = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=torch.zeros(10))},
        )
        fp1 = compute_dataset_fingerprint(ds1)
        fp2 = compute_dataset_fingerprint(ds2)
        assert fp1 != fp2

    @pytest.mark.unit
    def test_fingerprint_different_shape(self) -> None:
        ds1 = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes={"x": AxisInfo(name="x", values=torch.linspace(0, 1, 10))},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=torch.ones(10))},
        )
        ds2 = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes={"x": AxisInfo(name="x", values=torch.linspace(0, 1, 20))},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=torch.ones(20))},
        )
        fp1 = compute_dataset_fingerprint(ds1)
        fp2 = compute_dataset_fingerprint(ds2)
        assert fp1 != fp2

    @pytest.mark.unit
    def test_fingerprint_different_topology(self) -> None:
        axes = {"x": AxisInfo(name="x", values=torch.linspace(0, 1, 10))}
        fields = {"u": FieldData(name="u", values=torch.ones(10))}

        ds1 = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes=axes,
            axis_order=["x"],
            fields=fields,
        )
        ds2 = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            topology=DataTopology.SCATTERED,
            axes=axes,
            axis_order=["x"],
            fields=fields,
        )
        fp1 = compute_dataset_fingerprint(ds1)
        fp2 = compute_dataset_fingerprint(ds2)
        assert fp1 != fp2

    @pytest.mark.unit
    def test_fingerprint_different_axis_order(self) -> None:


        n = 8
        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(0, 1, n)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, n)),
        }
        values = torch.ones(n, n)
        ds_xt = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=values)},
        )
        ds_tx = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["t", "x"],
            fields={"u": FieldData(name="u", values=values)},
        )
        fp_xt = compute_dataset_fingerprint(ds_xt)
        fp_tx = compute_dataset_fingerprint(ds_tx)
        assert fp_xt != fp_tx

    @pytest.mark.unit
    def test_fingerprint_different_is_periodic(self) -> None:
        n = 8
        values = torch.linspace(0, 1, n)
        field_values = torch.ones(n)
        ds_periodic = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes={"x": AxisInfo(name="x", values=values, is_periodic=True)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=field_values)},
        )
        ds_non_periodic = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes={"x": AxisInfo(name="x", values=values, is_periodic=False)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=field_values)},
        )
        fp_p = compute_dataset_fingerprint(ds_periodic)
        fp_np = compute_dataset_fingerprint(ds_non_periodic)
        assert fp_p != fp_np

    @pytest.mark.unit
    def test_fingerprint_different_axis_values(self) -> None:
        n = 8
        field_values = torch.ones(n)
        ds_unit = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes={"x": AxisInfo(name="x", values=torch.linspace(0, 1, n))},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=field_values)},
        )
        ds_wide = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes={"x": AxisInfo(name="x", values=torch.linspace(0, 2, n))},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=field_values)},
        )
        fp_unit = compute_dataset_fingerprint(ds_unit)
        fp_wide = compute_dataset_fingerprint(ds_wide)
        assert fp_unit != fp_wide

    @pytest.mark.unit
    def test_fingerprint_large_data_uses_sampling(self) -> None:


        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(0, 1, 1000)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 1000)),
        }
        fields = {"u": FieldData(name="u", values=torch.randn(1000, 1000))}
        dataset = PDEDataset(
            name="large",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x", "t"],
            fields=fields,
        )

        fp = compute_dataset_fingerprint(dataset)
        assert isinstance(fp, str)
        assert len(fp) > 0







class TestSchemaDeviceAwareness:

    @pytest.mark.numerical
    def test_axis_preserves_device(self, device: torch.device) -> None:
        values = torch.linspace(0, 1, 10, device=device)
        axis = AxisInfo(name="x", values=values)
        assert axis.values.device.type == device.type

    @pytest.mark.numerical
    def test_field_preserves_device(self, device: torch.device) -> None:
        values = torch.randn(10, 20, device=device)
        field = FieldData(name="u", values=values)
        assert field.values.device.type == device.type

    @pytest.mark.numerical
    def test_get_coords_preserves_device(self, device: torch.device) -> None:
        axes = {"x": AxisInfo(name="x", values=torch.linspace(0, 1, 10, device=device))}
        fields = {"u": FieldData(name="u", values=torch.randn(10, device=device))}
        dataset = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x"],
            fields=fields,
        )
        coords = dataset.get_coords("x")
        assert coords.device.type == device.type

    @pytest.mark.numerical
    def test_get_field_preserves_device(self, device: torch.device) -> None:
        axes = {"x": AxisInfo(name="x", values=torch.linspace(0, 1, 10, device=device))}
        fields = {"u": FieldData(name="u", values=torch.randn(10, device=device))}
        dataset = PDEDataset(
            name="test",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x"],
            fields=fields,
        )
        field = dataset.get_field("u")
        assert field.device.type == device.type







class TestFieldDataDtypeValidation:

    def test_rejects_int64_tensor(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            FieldData(name="u", values=torch.tensor([[1, 2, 3], [4, 5, 6]]))

    def test_rejects_int32_tensor(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            FieldData(name="u", values=torch.tensor([10, 20, 30], dtype=torch.int32))

    def test_rejects_int16_tensor(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            FieldData(name="u", values=torch.tensor([1, 2], dtype=torch.int16))

    def test_rejects_bool_tensor(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            FieldData(name="u", values=torch.tensor([True, False, True]))

    def test_accepts_float32(self) -> None:
        field = FieldData(name="u", values=torch.randn(3, 4))
        assert field.values.dtype == torch.float32

    def test_accepts_float64(self) -> None:
        field = FieldData(name="u", values=torch.randn(3, 4, dtype=torch.float64))
        assert field.values.dtype == torch.float64

    def test_accepts_float16(self) -> None:
        field = FieldData(name="u", values=torch.randn(3, 4, dtype=torch.float16))
        assert field.values.dtype == torch.float16

    def test_accepts_bfloat16(self) -> None:
        field = FieldData(name="u", values=torch.randn(3, 4, dtype=torch.bfloat16))
        assert field.values.dtype == torch.bfloat16

    def test_int_tensor_fd_truncation_evidence(self) -> None:
        int_data = torch.tensor([0, 10, 20, 30, 40], dtype=torch.int64)
        buffer = torch.zeros_like(int_data)
        assert buffer.dtype == torch.int64, "zeros_like preserves int dtype"


        float_result = (int_data[1] - int_data[0]) / 3
        buffer[0] = float_result
        assert buffer[0].item() == 3
        assert float_result.item() != 3







class TestPDEDatasetLhsValidation:

    def _make_dataset(
        self,
        lhs_field: str = "",
        lhs_axis: str = "",
    ) -> PDEDataset:
        axes = {
            "x": AxisInfo(name="x", values=torch.linspace(0, 1, 20)),
            "t": AxisInfo(name="t", values=torch.linspace(0, 1, 10)),
        }
        fields = {"u": FieldData(name="u", values=torch.randn(20, 10))}
        return PDEDataset(
            name="lhs_test",
            task_type=TaskType.PDE,
            axes=axes,
            axis_order=["x", "t"],
            fields=fields,
            lhs_field=lhs_field,
            lhs_axis=lhs_axis,
        )

    def test_rejects_nonexistent_lhs_field(self) -> None:
        with pytest.raises(ValueError, match="lhs_field"):
            self._make_dataset(lhs_field="nonexistent", lhs_axis="t")

    def test_rejects_nonexistent_lhs_axis(self) -> None:
        with pytest.raises(ValueError, match="lhs_axis"):
            self._make_dataset(lhs_field="u", lhs_axis="z")

    def test_rejects_both_invalid(self) -> None:
        with pytest.raises(ValueError):
            self._make_dataset(lhs_field="v", lhs_axis="z")

    def test_empty_strings_accepted(self) -> None:
        ds = self._make_dataset(lhs_field="", lhs_axis="")
        assert ds.lhs_field == ""
        assert ds.lhs_axis == ""

    def test_valid_lhs_accepted(self) -> None:
        ds = self._make_dataset(lhs_field="u", lhs_axis="t")
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"

    def test_valid_lhs_field_only(self) -> None:
        ds = self._make_dataset(lhs_field="u", lhs_axis="")
        assert ds.lhs_field == "u"

    def test_valid_lhs_axis_only(self) -> None:
        ds = self._make_dataset(lhs_field="", lhs_axis="t")
        assert ds.lhs_axis == "t"

    def test_rejects_lhs_field_case_sensitive(self) -> None:
        with pytest.raises(ValueError, match="lhs_field"):
            self._make_dataset(lhs_field="U", lhs_axis="t")

    def test_rejects_lhs_axis_case_sensitive(self) -> None:
        with pytest.raises(ValueError, match="lhs_axis"):
            self._make_dataset(lhs_field="u", lhs_axis="T")

    def test_no_fields_skips_validation(self) -> None:
        ds = PDEDataset(
            name="minimal",
            task_type=TaskType.PDE,
            lhs_field="u",
            lhs_axis="t",
        )
        assert ds.lhs_field == "u"
