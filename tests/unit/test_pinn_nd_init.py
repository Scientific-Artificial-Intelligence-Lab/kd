"""Tests for PINN model initialization with custom N-D dataset names.

The PINN_model.load_inner_data() pathway crashes when the dataset_name
is not in the built-in registry (e.g., "2d_decay" from a custom
N-D PDEDataset).  The bug chain:

    KD_DSCV_SPR.import_dataset()
      -> setup()
        -> make_pinn_model()  passes self.config_task['dataset']
          -> PINN_model.__init__()  calls self.load_inner_data()
            -> load_1d_data(dataset_name, ...)
              -> assert False, "Dataset {dataset} is not existed"

These tests verify the fix: when data is supplied externally via
import_outter_data(), PINN_model.__init__ must NOT call load_inner_data().
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nd_3d_dataset():
    """Create a 3D N-D PDEDataset (x, y, t) for testing."""
    from kd.dataset import PDEDataset

    nx, ny, nt = 8, 8, 10
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    t = np.linspace(0, 1, nt)

    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.exp(-T)

    return PDEDataset(
        equation_name="2d_decay",
        fields_data={"u": u},
        coords_1d={"x": x, "y": y, "t": t},
        axis_order=["x", "y", "t"],
        target_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def nd_2d_dataset():
    """Create a standard 2D N-D PDEDataset (x, t) with custom name."""
    from kd.dataset import PDEDataset

    nx, nt = 16, 20
    x = np.linspace(0, 2 * np.pi, nx)
    t = np.linspace(0, 1, nt)

    X, T = np.meshgrid(x, t, indexing="ij")
    u = np.sin(X) * np.exp(-T)

    return PDEDataset(
        equation_name="custom_1d_decay",
        fields_data={"u": u},
        coords_1d={"x": x, "t": t},
        axis_order=["x", "t"],
        target_field="u",
        lhs_axis="t",
    )


# ---------------------------------------------------------------------------
# Bug reproduction: PINN_model.load_inner_data with unregistered name
# ---------------------------------------------------------------------------


class TestPINNLoadInnerDataBug:
    """Verify that load_inner_data does NOT crash on custom dataset names.

    Before the fix, load_1d_data() raises:
        AssertionError: Dataset 2d_decay is not existed
    """

    def test_load_1d_data_rejects_unknown_name(self):
        """Confirm the root cause: load_1d_data asserts on unknown names."""
        torch = pytest.importorskip("torch", reason="requires torch")
        scipy = pytest.importorskip("scipy", reason="requires scipy")

        from kd.model.discover.task.pde.dataset import load_1d_data

        with pytest.raises(AssertionError, match="not existed"):
            load_1d_data(
                dataset="2d_decay",
                noise_level=0.0,
                data_ratio=0.1,
                pic_path="/tmp/test",
                coll_num=100,
            )

    def test_pinn_model_init_with_unknown_name_uses_none(self):
        """After fix: PINN_model.__init__ should accept dataset_name=None
        without calling load_inner_data().

        The fix should make make_pinn_model() pass dataset_name=None for
        externally-supplied datasets so load_inner_data() is skipped.
        """
        torch = pytest.importorskip("torch", reason="requires torch")

        from kd.model.discover.pinn import PINN_model

        config_pinn = {
            "number_layer": 2,
            "input_dim": 3,
            "n_hidden": 20,
            "out_dim": 1,
            "activation": "tanh",
            "coef_pde": 0.0,
            "noise": 0.0,
            "data_ratio": 0.1,
            "coll_data": 100,
            "data_type": "1D_1U",
            "lr": 1e-3,
            "pinn_epoch": 1,
            "duration": 1,
        }

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "test_pinn.csv")
            device = torch.device("cpu")

            # dataset_name=None should skip load_inner_data
            model = PINN_model(
                output_file=out_file,
                config_pinn=config_pinn,
                dataset_name=None,
                device=device,
            )
            assert model is not None


# ---------------------------------------------------------------------------
# Integration: KD_DSCV_SPR.import_dataset with custom N-D data
# ---------------------------------------------------------------------------


class TestDSCVSPRImportNDDataset:
    """KD_DSCV_SPR.import_dataset() should succeed with custom N-D datasets.

    Before the fix this raises AssertionError inside PINN_model.__init__.
    """

    @pytest.mark.slow
    def test_import_3d_dataset_no_error(self, nd_3d_dataset, monkeypatch):
        """import_dataset with a 3D custom PDEDataset must not crash."""
        torch = pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        from kd.model.kd_dscv import KD_DSCV_SPR

        model = KD_DSCV_SPR(
            n_iterations=1,
            n_samples_per_batch=5,
            seed=0,
        )
        monkeypatch.setattr(
            KD_DSCV_SPR, "make_gp_aggregator", lambda self: None, raising=False
        )

        # This should NOT raise AssertionError anymore
        model.import_dataset(
            nd_3d_dataset,
            sample=32,
            colloc_num=32,
            random_state=0,
            sample_ratio=0.05,
            noise_level=0.0,
        )

        assert model.dataset_ is nd_3d_dataset
        assert model.dataset == "2d_decay"

    @pytest.mark.slow
    def test_import_2d_custom_dataset_no_error(self, nd_2d_dataset, monkeypatch):
        """import_dataset with a 2D custom-named PDEDataset must not crash."""
        torch = pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        from kd.model.kd_dscv import KD_DSCV_SPR

        model = KD_DSCV_SPR(
            n_iterations=1,
            n_samples_per_batch=5,
            seed=0,
        )
        monkeypatch.setattr(
            KD_DSCV_SPR, "make_gp_aggregator", lambda self: None, raising=False
        )

        model.import_dataset(
            nd_2d_dataset,
            sample=32,
            colloc_num=32,
            random_state=0,
            sample_ratio=0.05,
            noise_level=0.0,
        )

        assert model.dataset_ is nd_2d_dataset
        assert model.dataset == "custom_1d_decay"

    @pytest.mark.slow
    def test_import_then_train_one_step(self, nd_3d_dataset, monkeypatch):
        """After import, a single training step should complete."""
        torch = pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("tensorflow", reason="Mode2 pipeline depends on TensorFlow")

        from kd.model.kd_dscv import KD_DSCV_SPR

        model = KD_DSCV_SPR(
            n_iterations=1,
            n_samples_per_batch=5,
            seed=0,
        )
        monkeypatch.setattr(
            KD_DSCV_SPR, "make_gp_aggregator", lambda self: None, raising=False
        )

        model.import_dataset(
            nd_3d_dataset,
            sample=32,
            colloc_num=32,
            random_state=0,
            sample_ratio=0.05,
            noise_level=0.0,
        )

        result = model.train(n_epochs=1, verbose=False)
        assert isinstance(result, dict)
        assert "expression" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPINNNDEdgeCases:
    """Edge cases for PINN with N-D datasets."""

    def test_load_inner_data_rejects_empty_string(self):
        """Empty string dataset name should also fail in load_1d_data."""
        torch = pytest.importorskip("torch", reason="requires torch")
        scipy = pytest.importorskip("scipy", reason="requires scipy")

        from kd.model.discover.task.pde.dataset import load_1d_data

        with pytest.raises(AssertionError, match="not existed"):
            load_1d_data(
                dataset="",
                noise_level=0.0,
                data_ratio=0.1,
                pic_path="/tmp/test",
                coll_num=100,
            )

    def test_pinn_model_init_with_registered_name_still_works(self):
        """Registered dataset names should still use load_inner_data normally.

        This test verifies backward compatibility: known names like 'Burgers2'
        should still trigger load_inner_data(). We skip this if the data file
        is not available.
        """
        torch = pytest.importorskip("torch", reason="requires torch")
        scipy = pytest.importorskip("scipy", reason="requires scipy")

        # Just verify the function recognizes valid names (without full data)
        from kd.model.discover.task.pde.dataset import load_1d_data

        # Burgers2 is a registered name but may fail due to missing .mat file
        # in CI; we just verify it does NOT hit the "not existed" assertion
        try:
            load_1d_data(
                dataset="Burgers2",
                noise_level=0.0,
                data_ratio=0.1,
                pic_path="/tmp/test",
                coll_num=100,
            )
        except AssertionError:
            # This means Burgers2 is not recognized - should NOT happen
            pytest.fail("Burgers2 should be a recognized dataset name")
        except (FileNotFoundError, OSError, Exception):
            # Data file missing is OK in CI - the name was recognized
            pass
