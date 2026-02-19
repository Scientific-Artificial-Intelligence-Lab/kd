"""Tests for the N-D SPR (PINN) reset_ut bug.

Bug: AD_generate_mD() in pde_pinn.py only sets task.ut_cache, never task.ut.
kd's N-D SPR path uses generation_type="multi_AD" (via _inject_nd_config),
which calls AD_generate_mD() — but unlike AD_generate_1D() which sets task.ut
directly, AD_generate_mD() only populates task.ut_cache. Without a subsequent
reset_ut(0) call, task.ut is never set. This causes:

1. _calculate_pinn_fields() crashes with AttributeError: 'PDEPINNTask' has
   no attribute 'ut'
2. aggregator.py:135 crashes on task.ut
3. All N-D SPR viz produces 0 artifacts

The fix should:
- Add reset_ut(0) call after AD_generate_mD() runs in generate_meta_data()
- Add defensive check in _calculate_pinn_fields() for robustness

Tests 1-2 are unit-level (task object directly).
Tests 3-7 are integration-level (full N-D SPR training + viz pipeline).
"""

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import numpy as np
import pytest

pytest.importorskip("scipy", reason="smoke tests require scipy for .mat loading")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NX, _NY, _NT = 8, 8, 10
_SAMPLE = 32
_COLLOC_NUM = 32
_SAMPLE_RATIO = 0.05
_N_SAMPLES_PER_BATCH = 5


# ---------------------------------------------------------------------------
# Shared test data factory (mirrors test_nd_examples_smoke.py)
# ---------------------------------------------------------------------------


def _make_3d_dataset(equation_name: str = "2d_decay"):
    """Create a 3D N-D PDEDataset (x, y, t) for testing."""
    from kd.dataset import PDEDataset

    x = np.linspace(0, 2 * np.pi, _NX)
    y = np.linspace(0, 2 * np.pi, _NY)
    t = np.linspace(0, 1, _NT)

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


def _make_nd_spr_model_pretrained(monkeypatch):
    """Create, import, and pretrain a KD_DSCV_SPR model with 3D N-D data.

    Returns (model, task) tuple after pretrain (before train).
    """
    from kd.model.discover.program import Program
    from kd.model.kd_dscv import KD_DSCV_SPR

    dataset = _make_3d_dataset("2d_decay")
    model = KD_DSCV_SPR(
        n_iterations=1,
        n_samples_per_batch=_N_SAMPLES_PER_BATCH,
        seed=0,
    )
    monkeypatch.setattr(
        KD_DSCV_SPR, "make_gp_aggregator", lambda self: None, raising=False
    )
    model.import_dataset(
        dataset,
        sample=_SAMPLE,
        colloc_num=_COLLOC_NUM,
        random_state=0,
        sample_ratio=_SAMPLE_RATIO,
        noise_level=0.0,
    )
    model.pretrain()
    return model, Program.task


def _train_nd_spr_model(monkeypatch) -> Tuple:
    """Helper: create, import, and train a KD_DSCV_SPR model with 3D N-D data.

    Returns (model, result) tuple.
    """
    from kd.model.kd_dscv import KD_DSCV_SPR

    dataset = _make_3d_dataset("2d_decay")

    model = KD_DSCV_SPR(
        n_iterations=1,
        n_samples_per_batch=_N_SAMPLES_PER_BATCH,
        seed=0,
    )
    monkeypatch.setattr(
        KD_DSCV_SPR, "make_gp_aggregator", lambda self: None, raising=False
    )

    model.import_dataset(
        dataset,
        sample=_SAMPLE,
        colloc_num=_COLLOC_NUM,
        random_state=0,
        sample_ratio=_SAMPLE_RATIO,
        noise_level=0.0,
    )
    result = model.train(n_epochs=1, verbose=False)
    return model, result


# ===========================================================================
# Unit tests: PDEPINNTask.AD_generate_mD and reset_ut
# ===========================================================================


class TestADGenerateMDSetsUtCache:
    """After AD_generate_mD(), task.ut_cache must be populated."""

    @pytest.mark.slow
    def test_ad_generate_md_sets_ut_cache(self, monkeypatch):
        """AD_generate_mD() should create task.ut_cache as a list of arrays.

        This test verifies the *existing* behavior: ut_cache is set.
        It should PASS even before the fix.
        """
        pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        model, task = _make_nd_spr_model_pretrained(monkeypatch)

        assert hasattr(task, "ut_cache"), (
            "AD_generate_mD should set task.ut_cache"
        )
        assert isinstance(task.ut_cache, list), (
            "ut_cache should be a list of arrays"
        )
        assert len(task.ut_cache) > 0, "ut_cache should not be empty"
        for i, arr in enumerate(task.ut_cache):
            assert isinstance(arr, np.ndarray), (
                f"ut_cache[{i}] should be a numpy array"
            )
            assert np.all(np.isfinite(arr)), (
                f"ut_cache[{i}] contains NaN or Inf"
            )
            assert arr.size > 0, f"ut_cache[{i}] should not be empty"


class TestResetUtPopulatesUtFromCache:
    """After reset_ut(id), task.ut should equal task.ut_cache[id]."""

    @pytest.mark.slow
    def test_reset_ut_populates_ut_from_cache(self, monkeypatch):
        """Calling task.reset_ut(0) must set task.ut = task.ut_cache[0].

        This tests the reset_ut method directly. It should PASS even
        before the fix (the method exists, it's just never called).
        """
        pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        model, task = _make_nd_spr_model_pretrained(monkeypatch)

        # Manually call reset_ut(0) -- the method that kd never calls
        task.reset_ut(0)

        assert hasattr(task, "ut"), (
            "After reset_ut(0), task.ut should exist"
        )
        np.testing.assert_allclose(
            task.ut,
            task.ut_cache[0],
            rtol=1e-7,
            err_msg="task.ut should equal task.ut_cache[0] after reset_ut(0)",
        )
        assert np.all(np.isfinite(task.ut)), (
            "task.ut should contain only finite values after reset_ut(0)"
        )


# ===========================================================================
# Integration tests: N-D SPR training should set task.ut (FAIL before fix)
# ===========================================================================


class TestNDSPRTrainingSetsTaskUt:
    """After KD_DSCV_SPR.train() with N-D data, Program.task.ut must exist.

    This is the core regression test. Before the fix, AD_generate_mD only
    sets ut_cache and never calls reset_ut, so task.ut is never set.
    """

    @pytest.mark.slow
    def test_nd_spr_training_sets_task_ut(self, monkeypatch):
        """After N-D SPR training, Program.task.ut must not be None.

        Expected: FAIL before fix (task.ut is never set by AD_generate_mD).
        Expected: PASS after fix (reset_ut(0) is called after AD_generate_mD).
        """
        pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        from kd.model.discover.program import Program

        model, result = _train_nd_spr_model(monkeypatch)

        task = Program.task
        assert hasattr(task, "ut"), (
            "After N-D SPR training, task.ut attribute must exist. "
            "Bug: AD_generate_mD sets ut_cache but never calls reset_ut(0)."
        )
        assert task.ut is not None, (
            "After N-D SPR training, task.ut must not be None."
        )
        assert isinstance(task.ut, np.ndarray), (
            "task.ut should be a numpy array."
        )
        assert task.ut.size > 0, "task.ut should not be empty."

        # Shape: ut should be 2D (n_points, 1) for single equation
        assert task.ut.ndim == 2, (
            f"task.ut should be 2D, got shape {task.ut.shape}"
        )

        # Finite values
        assert np.all(np.isfinite(task.ut)), (
            "task.ut contains NaN or Inf after N-D SPR training"
        )

        # Must match ut_cache[0]
        assert hasattr(task, "ut_cache"), "ut_cache should exist"
        np.testing.assert_allclose(
            task.ut,
            task.ut_cache[0],
            rtol=1e-7,
            err_msg="task.ut should equal task.ut_cache[0] after training",
        )


# ===========================================================================
# Integration tests: _calculate_pinn_fields with N-D SPR (FAIL before fix)
# ===========================================================================


class TestCalculatePinnFieldsND:
    """_calculate_pinn_fields must succeed for N-D SPR models.

    Before the fix, it crashes at line 572 with:
        AttributeError: 'PDEPINNTask' has no attribute 'ut'
    """

    @pytest.mark.slow
    def test_calculate_pinn_fields_nd_no_attribute_error(self, monkeypatch):
        """_calculate_pinn_fields(model, best_program) should not raise
        AttributeError on N-D SPR models.

        Expected: FAIL before fix (task.ut doesn't exist).
        Expected: PASS after fix.
        """
        pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        from kd.viz.dscv_viz import _calculate_pinn_fields

        model, result = _train_nd_spr_model(monkeypatch)

        assert "program" in result, (
            f"Training result missing 'program' key. Keys: {list(result.keys())}"
        )
        best_program = result["program"]

        # This is the line that crashes before the fix
        fields = _calculate_pinn_fields(model, best_program)

        assert isinstance(fields, dict), "Should return a dict of field data"
        assert "residual" in fields, "Should contain 'residual' key"
        assert "coords" in fields, "Should contain 'coords' key"
        assert "y_true" in fields, "Should contain 'y_true' key"
        assert "y_pred" in fields, "Should contain 'y_pred' key"
        assert "n_spatial_dims" in fields, "Should contain 'n_spatial_dims' key"

        # For 3D dataset (x, y, t), n_spatial_dims should be 2
        assert fields["n_spatial_dims"] == 2, (
            "3D dataset (x, y, t) should have 2 spatial dims"
        )

        # All arrays should have finite values
        for key in ["residual", "y_true", "y_pred"]:
            assert np.all(np.isfinite(fields[key])), (
                f"fields['{key}'] should contain only finite values"
            )


# ===========================================================================
# Integration tests: N-D SPR viz produces artifacts (FAIL before fix)
# ===========================================================================


class TestNDSPRVizProducesArtifacts:
    """After N-D SPR training, viz functions should produce output files.

    Before the fix, all viz functions crash because _calculate_pinn_fields
    fails on missing task.ut, so 0 artifacts are produced.
    """

    @pytest.mark.slow
    def test_nd_spr_viz_produces_residual_plot(self, monkeypatch):
        """plot_spr_residual_analysis should create spr_residual_analysis.png.

        Expected: FAIL before fix (crashes on task.ut AttributeError).
        Expected: PASS after fix.
        """
        pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")
        matplotlib = pytest.importorskip("matplotlib", reason="viz requires matplotlib")
        matplotlib.use("Agg")  # non-interactive backend for testing

        from kd.viz.dscv_viz import plot_spr_residual_analysis

        model, result = _train_nd_spr_model(monkeypatch)
        assert "program" in result, (
            f"Training result missing 'program' key. Keys: {list(result.keys())}"
        )
        best_program = result["program"]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise and should write a file
            plot_spr_residual_analysis(model, best_program, output_dir=tmpdir)

            expected_file = os.path.join(tmpdir, "spr_residual_analysis.png")
            assert os.path.isfile(expected_file), (
                f"Expected artifact '{expected_file}' was not created. "
                "This likely means _calculate_pinn_fields crashed on missing task.ut."
            )
            assert os.path.getsize(expected_file) > 0, (
                "Artifact file should not be empty."
            )

    @pytest.mark.slow
    def test_nd_spr_viz_produces_actual_vs_predicted(self, monkeypatch):
        """plot_spr_actual_vs_predicted should create spr_actual_vs_predicted.png.

        Expected: FAIL before fix.
        Expected: PASS after fix.
        """
        pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")
        matplotlib = pytest.importorskip("matplotlib", reason="viz requires matplotlib")
        matplotlib.use("Agg")

        from kd.viz.dscv_viz import plot_spr_actual_vs_predicted

        model, result = _train_nd_spr_model(monkeypatch)
        assert "program" in result
        best_program = result["program"]

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_spr_actual_vs_predicted(model, best_program, output_dir=tmpdir)

            expected_file = os.path.join(tmpdir, "spr_actual_vs_predicted.png")
            assert os.path.isfile(expected_file), (
                f"Expected artifact '{expected_file}' was not created."
            )
            assert os.path.getsize(expected_file) > 0

    @pytest.mark.slow
    def test_nd_spr_viz_produces_field_comparison(self, monkeypatch):
        """plot_spr_field_comparison should create its output file.

        Expected: FAIL before fix.
        Expected: PASS after fix.
        """
        pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")
        matplotlib = pytest.importorskip("matplotlib", reason="viz requires matplotlib")
        matplotlib.use("Agg")

        from kd.viz.dscv_viz import plot_spr_field_comparison

        model, result = _train_nd_spr_model(monkeypatch)
        assert "program" in result
        best_program = result["program"]

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_spr_field_comparison(model, best_program, output_dir=tmpdir)

            # Check that at least one file was created in the output dir
            files = os.listdir(tmpdir)
            assert len(files) > 0, (
                "plot_spr_field_comparison should produce at least one output file. "
                "0 artifacts means _calculate_pinn_fields crashed."
            )
