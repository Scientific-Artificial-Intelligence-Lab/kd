"""Tests for N-D PDEDataset extension.

Tests the N-D structured grid support in PDEDataset, including:
- N-D construction with fields_data/coords_1d/axis_order
- Validation of axis consistency
- Compatibility layer for legacy x/t/usol access
- Integration with SGADataAdapter
"""

import numpy as np
import pytest

from kd.dataset import PDEDataset


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def legacy_2d_data():
    """Standard 2D data for legacy API tests."""
    x = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 20)
    usol = np.sin(np.outer(x, t))  # shape: (10, 20)
    return x, t, usol


@pytest.fixture
def nd_2d_data():
    """2D data in N-D format (x, t)."""
    x = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 20)
    u = np.sin(np.outer(x, t))  # shape: (10, 20)
    return {
        "fields_data": {"u": u},
        "coords_1d": {"x": x, "t": t},
        "axis_order": ["x", "t"],
    }


@pytest.fixture
def nd_3d_data():
    """3D data (x, y, t)."""
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 2, 12)
    t = np.linspace(0, 1, 15)
    # Create 3D field: shape (8, 12, 15)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.exp(-T)
    return {
        "fields_data": {"u": u},
        "coords_1d": {"x": x, "y": y, "t": t},
        "axis_order": ["x", "y", "t"],
    }


@pytest.fixture
def nd_multi_field_data():
    """Multi-field 2D data (u and v)."""
    x = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 20)
    u = np.sin(np.outer(x, t))
    v = np.cos(np.outer(x, t))
    return {
        "fields_data": {"u": u, "v": v},
        "coords_1d": {"x": x, "t": t},
        "axis_order": ["x", "t"],
    }


# =============================================================================
# Legacy 2D Compatibility Tests
# =============================================================================


class TestLegacy2DCompatibility:
    """Ensure existing 2D API continues to work."""

    def test_legacy_2d_from_pde_data(self, legacy_2d_data):
        """Construction from pde_data dict should work as before."""
        x, t, usol = legacy_2d_data
        pde_data = {"x": x, "t": t, "usol": usol}

        ds = PDEDataset(
            equation_name="test_legacy",
            pde_data=pde_data,
            domain={"x": (0, 1), "t": (0, 1)},
            epi=1e-3,
        )

        np.testing.assert_array_equal(ds.x, x)
        np.testing.assert_array_equal(ds.t, t)
        np.testing.assert_array_equal(ds.usol, usol)

    def test_legacy_2d_from_arrays(self, legacy_2d_data):
        """Construction from x/t/usol arrays should work as before."""
        x, t, usol = legacy_2d_data

        ds = PDEDataset(
            equation_name="test_legacy",
            pde_data=None,
            domain=None,
            epi=0.0,
            x=x,
            t=t,
            usol=usol,
        )

        np.testing.assert_array_equal(ds.x, x)
        np.testing.assert_array_equal(ds.t, t)
        np.testing.assert_array_equal(ds.usol, usol)
        assert ds.equation_name == "test_legacy"

    def test_legacy_2d_shape_validation(self):
        """Shape mismatch should still raise ValueError."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 20)
        usol_wrong = np.zeros((5, 5))  # Wrong shape

        with pytest.raises(ValueError, match="dimensions.*do not match"):
            PDEDataset(
                equation_name="test",
                pde_data=None,
                domain=None,
                epi=0.0,
                x=x,
                t=t,
                usol=usol_wrong,
            )

    def test_legacy_u_alias(self, legacy_2d_data):
        """The .u attribute should still work as alias for usol."""
        x, t, usol = legacy_2d_data

        ds = PDEDataset(
            equation_name="test",
            pde_data=None,
            domain=None,
            epi=0.0,
            x=x,
            t=t,
            usol=usol,
        )

        np.testing.assert_array_equal(ds.u, ds.usol)


# =============================================================================
# N-D Construction Tests
# =============================================================================


class TestNDConstruction:
    """Tests for N-D data construction."""

    def test_nd_construction_2d(self, nd_2d_data):
        """Basic 2D construction using N-D interface."""
        ds = PDEDataset(
            equation_name="test_nd_2d",
            fields_data=nd_2d_data["fields_data"],
            coords_1d=nd_2d_data["coords_1d"],
            axis_order=nd_2d_data["axis_order"],
        )

        assert ds.fields_data is not None
        assert "u" in ds.fields_data
        assert ds.coords_1d is not None
        assert set(ds.coords_1d.keys()) == {"x", "t"}
        assert ds.axis_order == ["x", "t"]
        assert ds.target_field == "u"  # default
        assert ds.lhs_axis == "t"  # default

    def test_nd_construction_3d(self, nd_3d_data):
        """3D construction (x, y, t)."""
        ds = PDEDataset(
            equation_name="test_3d",
            fields_data=nd_3d_data["fields_data"],
            coords_1d=nd_3d_data["coords_1d"],
            axis_order=nd_3d_data["axis_order"],
        )

        expected_shape = (8, 12, 15)  # (len(x), len(y), len(t))
        assert ds.fields_data["u"].shape == expected_shape
        assert set(ds.coords_1d.keys()) == {"x", "y", "t"}

    def test_nd_multi_field(self, nd_multi_field_data):
        """Multi-field construction with u and v."""
        ds = PDEDataset(
            equation_name="test_multi",
            fields_data=nd_multi_field_data["fields_data"],
            coords_1d=nd_multi_field_data["coords_1d"],
            axis_order=nd_multi_field_data["axis_order"],
        )

        assert "u" in ds.fields_data
        assert "v" in ds.fields_data
        assert ds.fields_data["u"].shape == ds.fields_data["v"].shape

    def test_nd_custom_target_and_lhs(self, nd_multi_field_data):
        """Custom target_field and lhs_axis."""
        ds = PDEDataset(
            equation_name="test",
            fields_data=nd_multi_field_data["fields_data"],
            coords_1d=nd_multi_field_data["coords_1d"],
            axis_order=nd_multi_field_data["axis_order"],
            target_field="v",
            lhs_axis="x",
        )

        assert ds.target_field == "v"
        assert ds.lhs_axis == "x"


# =============================================================================
# N-D Validation Tests
# =============================================================================


class TestNDValidation:
    """Tests for N-D data validation."""

    def test_nd_shape_mismatch_raises(self):
        """Field shape must match coords product."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 20)
        u_wrong = np.zeros((5, 5))  # Should be (10, 20)

        with pytest.raises(ValueError, match="shape.*mismatch|does not match"):
            PDEDataset(
                equation_name="test",
                fields_data={"u": u_wrong},
                coords_1d={"x": x, "t": t},
                axis_order=["x", "t"],
            )

    def test_nd_fields_inconsistent_shape_raises(self):
        """All fields must have the same shape."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 20)
        u = np.zeros((10, 20))
        v = np.zeros((10, 15))  # Different shape

        with pytest.raises(ValueError, match="shape.*inconsistent|must.*same"):
            PDEDataset(
                equation_name="test",
                fields_data={"u": u, "v": v},
                coords_1d={"x": x, "t": t},
                axis_order=["x", "t"],
            )

    def test_nd_missing_target_field_raises(self, nd_2d_data):
        """target_field must exist in fields_data."""
        with pytest.raises(ValueError, match="target_field.*not found|not in"):
            PDEDataset(
                equation_name="test",
                fields_data=nd_2d_data["fields_data"],
                coords_1d=nd_2d_data["coords_1d"],
                axis_order=nd_2d_data["axis_order"],
                target_field="nonexistent",
            )

    def test_nd_missing_lhs_axis_raises(self, nd_2d_data):
        """lhs_axis must exist in axis_order."""
        with pytest.raises(ValueError, match="lhs_axis.*must be in axis_order"):
            PDEDataset(
                equation_name="test",
                fields_data=nd_2d_data["fields_data"],
                coords_1d=nd_2d_data["coords_1d"],
                axis_order=nd_2d_data["axis_order"],
                lhs_axis="nonexistent",
            )

    def test_nd_axis_order_required(self):
        """axis_order must be explicitly provided for N-D."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 20)
        u = np.zeros((10, 20))

        with pytest.raises(ValueError, match="axis_order.*required"):
            PDEDataset(
                equation_name="test",
                fields_data={"u": u},
                coords_1d={"x": x, "t": t},
                # Missing axis_order
            )

    def test_nd_partial_construction_raises(self):
        """Providing only fields_data without coords_1d should raise."""
        u = np.zeros((10, 20))

        with pytest.raises(ValueError, match="requires both|coords_1d"):
            PDEDataset(
                equation_name="test",
                fields_data={"u": u},
                # Missing coords_1d
            )

    # --- H1: axis_order/coords_1d consistency tests ---

    def test_nd_fields_data_empty_raises(self):
        """Empty fields_data should raise with clear message."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 20)

        with pytest.raises(ValueError, match="fields_data.*cannot be empty|empty"):
            PDEDataset(
                equation_name="test",
                fields_data={},
                coords_1d={"x": x, "t": t},
                axis_order=["x", "t"],
            )

    def test_nd_axis_order_empty_raises(self):
        """Empty axis_order should raise."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 20)
        u = np.zeros((10, 20))

        with pytest.raises(ValueError, match="cannot be empty"):
            PDEDataset(
                equation_name="test",
                fields_data={"u": u},
                coords_1d={"x": x, "t": t},
                axis_order=[],
            )

    def test_nd_axis_order_duplicate_raises(self):
        """Duplicate axes in axis_order should raise."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 20)
        u = np.zeros((10, 20))

        with pytest.raises(ValueError, match="duplicate"):
            PDEDataset(
                equation_name="test",
                fields_data={"u": u},
                coords_1d={"x": x, "t": t},
                axis_order=["x", "x"],
            )

    def test_nd_axis_order_missing_axis_raises(self):
        """axis_order with axis not in coords_1d should raise with clear message."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 20)
        u = np.zeros((10, 20))

        with pytest.raises(ValueError, match="Extra in axis_order|not in coords_1d"):
            PDEDataset(
                equation_name="test",
                fields_data={"u": u},
                coords_1d={"x": x, "t": t},
                axis_order=["x", "y"],  # 'y' not in coords_1d
            )

    def test_nd_axis_order_missing_coord_raises(self):
        """coords_1d with axis not in axis_order should raise."""
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 15)
        t = np.linspace(0, 1, 20)
        u = np.zeros((10, 20))

        with pytest.raises(ValueError, match="Missing in axis_order"):
            PDEDataset(
                equation_name="test",
                fields_data={"u": u},
                coords_1d={"x": x, "y": y, "t": t},  # 'y' not in axis_order
                axis_order=["x", "t"],
            )

    def test_nd_lhs_axis_not_in_axis_order_raises(self):
        """lhs_axis must be in axis_order, not just in coords_1d."""
        x = np.linspace(0, 1, 10)
        t = np.linspace(0, 1, 20)
        u = np.zeros((10, 20))

        # lhs_axis="t" is in coords_1d but imagine axis_order doesn't include it
        # This is a bit contrived since axis_order must match coords_1d,
        # but we test the explicit check for lhs_axis in axis_order
        with pytest.raises(ValueError, match="lhs_axis.*must be in axis_order"):
            PDEDataset(
                equation_name="test",
                fields_data={"u": u},
                coords_1d={"x": x, "t": t},
                axis_order=["x", "t"],
                lhs_axis="z",  # not in axis_order (and not in coords_1d)
            )


# =============================================================================
# Compatibility Layer Tests
# =============================================================================


class TestNDCompatibilityLayer:
    """Tests for x/t/usol compatibility in N-D mode."""

    def test_nd_compat_x_accessible(self, nd_2d_data):
        """In N-D mode, .x should return coords_1d['x'] if present."""
        ds = PDEDataset(
            equation_name="test",
            fields_data=nd_2d_data["fields_data"],
            coords_1d=nd_2d_data["coords_1d"],
            axis_order=nd_2d_data["axis_order"],
        )

        np.testing.assert_array_equal(ds.x, nd_2d_data["coords_1d"]["x"])

    def test_nd_compat_t_accessible(self, nd_2d_data):
        """In N-D mode, .t should return coords_1d['t'] if present."""
        ds = PDEDataset(
            equation_name="test",
            fields_data=nd_2d_data["fields_data"],
            coords_1d=nd_2d_data["coords_1d"],
            axis_order=nd_2d_data["axis_order"],
        )

        np.testing.assert_array_equal(ds.t, nd_2d_data["coords_1d"]["t"])

    def test_nd_compat_usol_accessible(self, nd_2d_data):
        """In N-D mode, .usol should return fields_data[target_field]."""
        ds = PDEDataset(
            equation_name="test",
            fields_data=nd_2d_data["fields_data"],
            coords_1d=nd_2d_data["coords_1d"],
            axis_order=nd_2d_data["axis_order"],
        )

        np.testing.assert_array_equal(ds.usol, nd_2d_data["fields_data"]["u"])

    def test_nd_compat_u_alias(self, nd_2d_data):
        """In N-D mode, .u should still work as alias for usol."""
        ds = PDEDataset(
            equation_name="test",
            fields_data=nd_2d_data["fields_data"],
            coords_1d=nd_2d_data["coords_1d"],
            axis_order=nd_2d_data["axis_order"],
        )

        np.testing.assert_array_equal(ds.u, ds.usol)

    def test_nd_compat_x_missing_returns_none(self, nd_3d_data):
        """If 'x' not in coords, .x should return None (not raise)."""
        # Remove 'x' from coords and rename to 'a'
        coords = {"a": nd_3d_data["coords_1d"]["x"], **{k: v for k, v in nd_3d_data["coords_1d"].items() if k != "x"}}
        # Reshape field accordingly
        u = nd_3d_data["fields_data"]["u"]

        ds = PDEDataset(
            equation_name="test",
            fields_data={"u": u},
            coords_1d=coords,
            axis_order=["a", "y", "t"],
        )

        assert ds.x is None

    def test_nd_get_data_returns_fields(self, nd_2d_data):
        """get_data() should work in N-D mode."""
        ds = PDEDataset(
            equation_name="test",
            fields_data=nd_2d_data["fields_data"],
            coords_1d=nd_2d_data["coords_1d"],
            axis_order=nd_2d_data["axis_order"],
        )

        data = ds.get_data()
        # Should at least have legacy keys for compatibility
        assert "x" in data or "usol" in data or "fields_data" in data


# =============================================================================
# Edge Cases
# =============================================================================


class TestNDEdgeCases:
    """Edge case tests for N-D PDEDataset."""

    def test_nd_single_point_per_axis(self):
        """Should handle single point per axis (degenerate case)."""
        x = np.array([0.5])
        t = np.array([0.0])
        u = np.array([[1.0]])

        ds = PDEDataset(
            equation_name="test",
            fields_data={"u": u},
            coords_1d={"x": x, "t": t},
            axis_order=["x", "t"],
        )

        assert ds.fields_data["u"].shape == (1, 1)

    def test_nd_1d_only(self):
        """Should handle 1D data (only t axis)."""
        t = np.linspace(0, 1, 100)
        u = np.sin(t).reshape(-1)  # 1D array

        ds = PDEDataset(
            equation_name="test",
            fields_data={"u": u},
            coords_1d={"t": t},
            axis_order=["t"],
        )

        assert ds.fields_data["u"].shape == (100,)
        assert ds.lhs_axis == "t"

    def test_nd_repr_works(self, nd_3d_data):
        """__repr__ should not crash in N-D mode."""
        ds = PDEDataset(
            equation_name="test_3d",
            fields_data=nd_3d_data["fields_data"],
            coords_1d=nd_3d_data["coords_1d"],
            axis_order=nd_3d_data["axis_order"],
        )

        repr_str = repr(ds)
        assert "test_3d" in repr_str

    def test_nd_equation_name_preserved(self, nd_2d_data):
        """equation_name should be accessible."""
        ds = PDEDataset(
            equation_name="my_custom_pde",
            fields_data=nd_2d_data["fields_data"],
            coords_1d=nd_2d_data["coords_1d"],
            axis_order=nd_2d_data["axis_order"],
        )

        assert ds.equation_name == "my_custom_pde"


# =============================================================================
# Integration with SGADataAdapter
# =============================================================================


class TestSGADataAdapterIntegration:
    """Tests that N-D PDEDataset works with SGADataAdapter."""

    def test_adapter_uses_nd_path(self, nd_3d_data):
        """SGADataAdapter should detect N-D mode and return structured kwargs."""
        from kd.model.sga.adapter import SGADataAdapter

        ds = PDEDataset(
            equation_name="test_3d",
            fields_data=nd_3d_data["fields_data"],
            coords_1d=nd_3d_data["coords_1d"],
            axis_order=nd_3d_data["axis_order"],
        )

        adapter = SGADataAdapter(dataset=ds)
        kwargs = adapter.to_solver_kwargs()

        # Should use N-D payload, not legacy u_data/x_data/t_data
        assert "fields_data" in kwargs
        assert "coords_1d" in kwargs
        assert "axis_order" in kwargs
        assert kwargs["target_field"] == "u"
        assert kwargs["lhs_axis"] == "t"

    def test_adapter_legacy_still_works(self, legacy_2d_data):
        """SGADataAdapter should still work with legacy 2D datasets."""
        from kd.model.sga.adapter import SGADataAdapter

        x, t, usol = legacy_2d_data
        ds = PDEDataset(
            equation_name="test_legacy",
            pde_data=None,
            domain=None,
            epi=0.0,
            x=x,
            t=t,
            usol=usol,
        )

        adapter = SGADataAdapter(dataset=ds)
        kwargs = adapter.to_solver_kwargs()

        # Should use legacy path
        assert "u_data" in kwargs
        assert "x_data" in kwargs
        assert "t_data" in kwargs
