
from __future__ import annotations

import numpy as np
import pytest
import torch

from kd.data.schema import (
    DataTopology,
    PDEDataset,
    TaskType,
    compute_dataset_fingerprint,
)






class TestFromScatterHappyPath:

    @pytest.mark.unit
    def test_evolution_default_lhs_builds_scattered(self) -> None:
        t = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.2, 3.1, 2.7], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        ds = PDEDataset.from_scatter(
            coords={"t": t, "x": x},
            fields={"u": u},
        )

        assert ds.topology == DataTopology.SCATTERED
        assert ds.task_type == TaskType.PDE
        assert ds.axis_order == ["t", "x"]
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.lhs_order == 1

        assert torch.equal(ds.get_field("u"), u)
        assert torch.equal(ds.get_coords("t"), t)
        assert torch.equal(ds.get_coords("x"), x)
        assert ds.get_field("u").dim() == 1
        assert ds.get_coords("t").dim() == 1

    @pytest.mark.unit
    def test_homogeneous_empty_lhs_gives_order_zero(self) -> None:
        x = torch.tensor([0.1, 0.9, 0.4], dtype=torch.float64)
        y = torch.tensor([2.0, 0.3, 1.1], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        ds = PDEDataset.from_scatter(
            coords={"x": x, "y": y},
            fields={"u": u},
            lhs="",
        )

        assert ds.topology == DataTopology.SCATTERED
        assert ds.lhs_order == 0
        assert ds.lhs_axis == ""
        assert ds.lhs_field == ""
        assert ds.axis_order == ["x", "y"]

    @pytest.mark.unit
    def test_insertion_order_defines_axis_order(self) -> None:
        x = torch.tensor([0.1, 0.9, 0.4], dtype=torch.float64)
        t = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        ds = PDEDataset.from_scatter(
            coords={"x": x, "t": t},
            fields={"u": u},
            lhs="u_t",
        )
        assert ds.axis_order == ["x", "t"]
        assert ds.lhs_axis == "t"

    @pytest.mark.unit
    def test_non_monotonic_coords_accepted(self) -> None:
        t = torch.tensor([3.0, 0.0, 2.0, 1.0], dtype=torch.float64)
        x = torch.tensor([9.0, 1.0, 4.0, 2.0], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        ds = PDEDataset.from_scatter(
            coords={"t": t, "x": x},
            fields={"u": u},
        )
        assert torch.equal(ds.get_coords("t"), t)







class TestFromScatterFailLoud:

    @pytest.mark.unit
    def test_field_length_mismatch_raises(self) -> None:
        t = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.2, 3.1, 2.7], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        with pytest.raises(ValueError):
            PDEDataset.from_scatter(
                coords={"t": t, "x": x},
                fields={"u": u},
            )

    @pytest.mark.unit
    def test_coord_length_mismatch_raises(self) -> None:
        t = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.2, 3.1], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        with pytest.raises(ValueError):
            PDEDataset.from_scatter(
                coords={"t": t, "x": x},
                fields={"u": u},
            )

    @pytest.mark.unit
    def test_two_dimensional_field_raises(self) -> None:
        t = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.2], dtype=torch.float64)
        u2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)

        with pytest.raises(ValueError):
            PDEDataset.from_scatter(
                coords={"t": t, "x": x},
                fields={"u": u2d},
            )

    @pytest.mark.unit
    def test_two_dimensional_coord_raises(self) -> None:
        t2d = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.2, 3.1], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        with pytest.raises(ValueError):
            PDEDataset.from_scatter(
                coords={"t": t2d, "x": x},
                fields={"u": u},
            )

    @pytest.mark.unit
    def test_empty_coords_raises(self) -> None:
        u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            PDEDataset.from_scatter(
                coords={},
                fields={"u": u},
            )

    @pytest.mark.unit
    def test_empty_fields_raises(self) -> None:
        t = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            PDEDataset.from_scatter(
                coords={"t": t},
                fields={},
            )

    @pytest.mark.unit
    def test_lhs_references_missing_field_raises(self) -> None:
        t = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.2], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        with pytest.raises(ValueError):
            PDEDataset.from_scatter(
                coords={"t": t, "x": x},
                fields={"u": u},
                lhs="v_t",
            )







class TestFromScatterInputAndFingerprint:

    @pytest.mark.unit
    def test_numpy_input_cast_to_float64(self) -> None:
        t = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        x = np.array([0.5, 1.5, 0.2, 3.1], dtype=np.float32)
        u = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        ds = PDEDataset.from_scatter(
            coords={"t": t, "x": x},
            fields={"u": u},
        )
        assert ds.get_field("u").dtype == torch.float64
        assert ds.get_coords("t").dtype == torch.float64
        assert torch.equal(
            ds.get_field("u"), torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        )

    @pytest.mark.unit
    def test_fingerprint_round_trips(self) -> None:
        t = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.2, 3.1], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        ds = PDEDataset.from_scatter(
            coords={"t": t, "x": x},
            fields={"u": u},
        )
        fp = compute_dataset_fingerprint(ds)
        assert isinstance(fp, str)
        assert DataTopology.SCATTERED.value in fp

    @pytest.mark.unit
    def test_fingerprint_content_sensitive_to_coord_value(self) -> None:
        t = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        x = torch.tensor([0.5, 1.5, 0.2, 3.1, 2.7], dtype=torch.float64)
        u = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        ds1 = PDEDataset.from_scatter(
            coords={"t": t, "x": x},
            fields={"u": u},
        )
        t_alt = t.clone()
        t_alt[-1] = 9.0
        ds2 = PDEDataset.from_scatter(
            coords={"t": t_alt, "x": x},
            fields={"u": u},
        )
        assert compute_dataset_fingerprint(ds1) != compute_dataset_fingerprint(ds2)
