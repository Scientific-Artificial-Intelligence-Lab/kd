"""Smoke tests for N-D example scripts.

These tests verify that the 6 N-D example scripts run without error.
Rather than running scripts as subprocesses, we replicate the core
workflow (dataset creation -> model import -> short train) directly,
similar to how test_examples_smoke.py works for registered datasets.

Covers:
- kd_sga_nd_example.py
- kd_dscv_nd_example.py
- kd_dscvspr_nd_example.py
- kd_sga_nd_viz_api_example.py     (new)
- kd_dscv_nd_viz_api_example.py    (new)
- kd_dscvspr_nd_viz_api_example.py (new)
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="smoke tests require scipy for .mat loading")


# ---------------------------------------------------------------------------
# Shared test data factories
# ---------------------------------------------------------------------------


def _make_3d_dataset(equation_name: str = "3d_decay"):
    """Create a 3D N-D PDEDataset (x, y, t) mirroring the example scripts."""
    from kd.dataset import PDEDataset

    nx, ny, nt = 8, 8, 10
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    t = np.linspace(0, 1, nt)

    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.exp(-T)

    return PDEDataset(
        equation_name=equation_name,
        fields_data={"u": u},
        coords_1d={"x": x, "y": y, "t": t},
        axis_order=["x", "y", "t"],
        target_field="u",
        lhs_axis="t",
    )


def _make_2d_nd_dataset(equation_name: str = "2d_decay"):
    """Create a 2D N-D PDEDataset (x, t) mirroring the example scripts."""
    from kd.dataset import PDEDataset

    nx, nt = 16, 20
    x = np.linspace(0, 2 * np.pi, nx)
    t = np.linspace(0, 1, nt)

    X, T = np.meshgrid(x, t, indexing="ij")
    u = np.sin(X) * np.exp(-T)

    return PDEDataset(
        equation_name=equation_name,
        fields_data={"u": u},
        coords_1d={"x": x, "t": t},
        axis_order=["x", "t"],
        target_field="u",
        lhs_axis="t",
    )


# ===========================================================================
# 1. KD_SGA N-D (kd_sga_nd_example.py)
# ===========================================================================


class TestKDSGANDSmoke:
    """Smoke test for kd_sga_nd_example.py workflow."""

    @pytest.mark.smoke
    def test_sga_nd_3d_dataset_fit(self):
        """KD_SGA.fit_dataset with 3D N-D PDEDataset should complete."""
        from kd.model.kd_sga import KD_SGA

        dataset = _make_3d_dataset()

        model = KD_SGA(
            sga_run=1,
            num=8,
            depth=3,
            width=3,
            max_epoch=500,
            seed=0,
        )
        result = model.fit_dataset(dataset)

        assert result is model
        assert getattr(model, "best_pde_", None) is not None
        assert getattr(model, "dataset_", None) is dataset


# ===========================================================================
# 2. KD_DSCV N-D (kd_dscv_nd_example.py)
# ===========================================================================


class TestKDDSCVNDSmoke:
    """Smoke test for kd_dscv_nd_example.py workflow."""

    @pytest.mark.smoke
    def test_dscv_nd_3d_dataset_train(self, monkeypatch):
        """KD_DSCV import + train with 3D N-D PDEDataset should complete."""
        torch = pytest.importorskip("torch", reason="KD_DSCV depends on torch")

        from kd.model.kd_dscv import KD_DSCV

        dataset = _make_3d_dataset("2d_decay")

        model = KD_DSCV(
            binary_operators=["add", "mul", "Diff"],
            unary_operators=["n2"],
            n_samples_per_batch=500,
            n_iterations=5,
            seed=0,
        )
        monkeypatch.setattr(
            KD_DSCV, "make_gp_aggregator", lambda self: None, raising=False
        )

        model.import_dataset(dataset)
        result = model.train(n_epochs=5, verbose=False)

        assert isinstance(result, dict)
        assert "expression" in result
        assert getattr(model, "dataset_", None) is dataset


# ===========================================================================
# 3. KD_DSCV_SPR N-D (kd_dscvspr_nd_example.py)
# ===========================================================================


class TestKDDSCVSPRNDSmoke:
    """Smoke test for kd_dscvspr_nd_example.py workflow."""

    @pytest.mark.slow
    def test_dscvspr_nd_3d_dataset_import(self, monkeypatch):
        """KD_DSCV_SPR.import_dataset with 3D N-D PDEDataset should not crash.

        This is the key test for the PINN load_inner_data bug fix.
        """
        torch = pytest.importorskip("torch", reason="KD_DSCV_SPR requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        from kd.model.kd_dscv import KD_DSCV_SPR

        dataset = _make_3d_dataset("2d_decay")

        model = KD_DSCV_SPR(
            n_iterations=1,
            n_samples_per_batch=5,
            seed=0,
        )
        monkeypatch.setattr(
            KD_DSCV_SPR, "make_gp_aggregator", lambda self: None, raising=False
        )

        model.import_dataset(
            dataset,
            sample=32,
            colloc_num=32,
            random_state=0,
            sample_ratio=0.05,
            noise_level=0.0,
        )

        assert getattr(model, "dataset_", None) is dataset
        data = model.data_class.get_data()
        assert data["X_u_train"].shape[0] > 0

    @pytest.mark.slow
    def test_dscvspr_nd_3d_dataset_train(self, monkeypatch):
        """KD_DSCV_SPR import + train with 3D N-D data should complete."""
        torch = pytest.importorskip("torch", reason="KD_DSCV_SPR requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        from kd.model.kd_dscv import KD_DSCV_SPR

        dataset = _make_3d_dataset("2d_decay")

        model = KD_DSCV_SPR(
            n_iterations=1,
            n_samples_per_batch=5,
            seed=0,
        )
        monkeypatch.setattr(
            KD_DSCV_SPR, "make_gp_aggregator", lambda self: None, raising=False
        )

        model.import_dataset(
            dataset,
            sample=32,
            colloc_num=32,
            random_state=0,
            sample_ratio=0.05,
            noise_level=0.0,
        )
        result = model.train(n_epochs=1, verbose=False)

        assert isinstance(result, dict)
        assert "expression" in result


# ===========================================================================
# 4. KD_SGA N-D Viz API (kd_sga_nd_viz_api_example.py - new)
# ===========================================================================


class TestKDSGANDVizAPISmoke:
    """Smoke test for kd_sga_nd_viz_api_example.py workflow."""

    @pytest.mark.smoke
    def test_sga_nd_viz_api_render_equation(self):
        """KD_SGA N-D -> render_equation via unified viz API."""
        from kd.model.kd_sga import KD_SGA

        dataset = _make_3d_dataset()
        model = KD_SGA(
            sga_run=1,
            num=8,
            depth=3,
            width=3,
            max_epoch=500,
            seed=0,
        )
        model.fit_dataset(dataset)

        from kd.viz import render_equation
        # Should not raise; may return None/figure depending on backend
        render_equation(model)

    @pytest.mark.smoke
    def test_sga_nd_viz_api_plot_parity(self):
        """KD_SGA N-D -> plot_parity via unified viz API."""
        from kd.model.kd_sga import KD_SGA

        dataset = _make_3d_dataset()
        model = KD_SGA(
            sga_run=1,
            num=8,
            depth=3,
            width=3,
            max_epoch=500,
            seed=0,
        )
        model.fit_dataset(dataset)

        from kd.viz import plot_parity
        plot_parity(model, title="SGA N-D Parity Smoke")


# ===========================================================================
# 5. KD_DSCV N-D Viz API (kd_dscv_nd_viz_api_example.py - new)
# ===========================================================================


class TestKDDSCVNDVizAPISmoke:
    """Smoke test for kd_dscv_nd_viz_api_example.py workflow."""

    @pytest.mark.smoke
    def test_dscv_nd_viz_api_render_equation(self, monkeypatch):
        """KD_DSCV N-D -> render_equation via unified viz API."""
        torch = pytest.importorskip("torch", reason="KD_DSCV depends on torch")

        from kd.model.kd_dscv import KD_DSCV

        dataset = _make_3d_dataset("2d_decay")
        model = KD_DSCV(
            binary_operators=["add", "mul", "Diff"],
            unary_operators=["n2"],
            n_samples_per_batch=500,
            n_iterations=5,
            seed=0,
        )
        monkeypatch.setattr(
            KD_DSCV, "make_gp_aggregator", lambda self: None, raising=False
        )
        model.import_dataset(dataset)
        model.train(n_epochs=5, verbose=False)

        from kd.viz import render_equation
        render_equation(model)

    @pytest.mark.smoke
    def test_dscv_nd_viz_api_plot_parity(self, monkeypatch):
        """KD_DSCV N-D -> plot_parity via unified viz API."""
        torch = pytest.importorskip("torch", reason="KD_DSCV depends on torch")

        from kd.model.kd_dscv import KD_DSCV

        dataset = _make_3d_dataset("2d_decay")
        model = KD_DSCV(
            binary_operators=["add", "mul", "Diff"],
            unary_operators=["n2"],
            n_samples_per_batch=500,
            n_iterations=5,
            seed=0,
        )
        monkeypatch.setattr(
            KD_DSCV, "make_gp_aggregator", lambda self: None, raising=False
        )
        model.import_dataset(dataset)
        model.train(n_epochs=5, verbose=False)

        from kd.viz import plot_parity
        plot_parity(model, title="DSCV N-D Parity Smoke")


# ===========================================================================
# 6. KD_DSCV_SPR N-D Viz API (kd_dscvspr_nd_viz_api_example.py - new)
# ===========================================================================


class TestKDDSCVSPRNDVizAPISmoke:
    """Smoke test for kd_dscvspr_nd_viz_api_example.py workflow."""

    @pytest.mark.slow
    def test_dscvspr_nd_viz_api_render_equation(self, monkeypatch):
        """KD_DSCV_SPR N-D -> render_equation via unified viz API."""
        torch = pytest.importorskip("torch", reason="KD_DSCV_SPR requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        from kd.model.kd_dscv import KD_DSCV_SPR

        dataset = _make_3d_dataset("2d_decay")
        model = KD_DSCV_SPR(
            n_iterations=1,
            n_samples_per_batch=5,
            seed=0,
        )
        monkeypatch.setattr(
            KD_DSCV_SPR, "make_gp_aggregator", lambda self: None, raising=False
        )
        model.import_dataset(
            dataset,
            sample=32,
            colloc_num=32,
            random_state=0,
            sample_ratio=0.05,
            noise_level=0.0,
        )
        result = model.train(n_epochs=1, verbose=False)

        from kd.viz import render_equation
        render_equation(model)
