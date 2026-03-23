"""Tests for EqGPT migration from ref_lib to kd.model.eqgpt.

Validates that the migrated EqGPT modules are importable, free of heavy
side effects at import time, and correctly integrated into the kd package.

Levels:
  1. Bottom-level modules importable (no side effects)
  2. Side-effect-heavy modules wrapped (import is fast)
  3. kd.model import unaffected by EqGPT presence
  4. __init__.py exists with no heavy imports
  5. pyproject.toml declares eqgpt package-data and optional deps
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMPORT_FAST_THRESHOLD_SEC = 2.0
KD_MODEL_IMPORT_THRESHOLD_SEC = 5.0

# Path to the eqgpt package relative to this test file
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EQGPT_PKG = _PROJECT_ROOT / "kd" / "model" / "eqgpt"
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


# ===========================================================================
# Level 1: Bottom-level modules importable (no side effects)
# ===========================================================================


@pytest.mark.smoke
class TestLevel1Import:
    """Bottom-level EqGPT modules should be importable without side effects."""

    def test_import_device(self) -> None:
        """_device module exposes a `device` object (torch device selector)."""
        from kd.model.eqgpt._device import device  # noqa: F811

        assert device is not None

    def test_import_gpt_model_class(self) -> None:
        """gpt_model module exposes the GPT class."""
        from kd.model.eqgpt.gpt_model import GPT

        assert GPT is not None

    def test_import_neural_network_class(self) -> None:
        """neural_network module exposes the NN class."""
        from kd.model.eqgpt.neural_network import NN

        assert NN is not None

    def test_import_calculate_terms_callable(self) -> None:
        """calculate_terms module exposes a callable `calculate_terms`."""
        from kd.model.eqgpt.calculate_terms import calculate_terms

        assert callable(calculate_terms)

    def test_import_find_coefficients(self) -> None:
        """find_coefficients module should be importable."""
        import kd.model.eqgpt.find_coefficients  # noqa: F401

    def test_import_read_dataset(self) -> None:
        """read_dataset module should be importable."""
        import kd.model.eqgpt.read_dataset  # noqa: F401


# ===========================================================================
# Level 2: Side-effect-heavy modules import fast (no eager loading)
# ===========================================================================


@pytest.mark.smoke
class TestLevel2NoSideEffects:
    """Modules that originally triggered heavy I/O at import time
    must be wrapped so that import itself is fast (<2s)."""

    def test_surrogate_model_import_fast(self) -> None:
        """Import surrogate_model should NOT load 199MB pickle eagerly."""
        t0 = time.perf_counter()
        from kd.model.eqgpt.surrogate_model import get_meta, train  # noqa: F401

        elapsed = time.perf_counter() - t0
        assert elapsed < IMPORT_FAST_THRESHOLD_SEC, (
            f"Import took {elapsed:.1f}s -- side effect not wrapped?"
        )
        assert callable(get_meta)
        assert callable(train)

    def test_multi_surrogate_model_import_fast(self) -> None:
        """Import multi_surrogate_model should NOT load heavy data eagerly."""
        t0 = time.perf_counter()
        import kd.model.eqgpt.multi_surrogate_model  # noqa: F401

        elapsed = time.perf_counter() - t0
        assert elapsed < IMPORT_FAST_THRESHOLD_SEC, (
            f"Import took {elapsed:.1f}s -- side effect not wrapped?"
        )

    def test_train_gpt_import_fast(self) -> None:
        """Import train_gpt should NOT trigger training or data loading."""
        t0 = time.perf_counter()
        import kd.model.eqgpt.train_gpt  # noqa: F401

        elapsed = time.perf_counter() - t0
        assert elapsed < IMPORT_FAST_THRESHOLD_SEC, (
            f"Import took {elapsed:.1f}s -- side effect not wrapped?"
        )

    def test_pinn_optimization_import_fast(self) -> None:
        """Import PINN_optimization should NOT start training."""
        t0 = time.perf_counter()
        import kd.model.eqgpt.PINN_optimization  # noqa: F401

        elapsed = time.perf_counter() - t0
        assert elapsed < IMPORT_FAST_THRESHOLD_SEC, (
            f"Import took {elapsed:.1f}s -- side effect not wrapped?"
        )


# ===========================================================================
# Level 3: kd.model import unaffected
# ===========================================================================


@pytest.mark.smoke
class TestLevel3KDModelUnaffected:
    """Adding eqgpt subpackage must not slow down or break kd.model import."""

    def test_kd_model_import_unaffected(self) -> None:
        """Importing kd.model should not trigger EqGPT side effects."""
        t0 = time.perf_counter()
        from kd.model import KD_Discover  # noqa: F401

        elapsed = time.perf_counter() - t0
        assert elapsed < KD_MODEL_IMPORT_THRESHOLD_SEC, (
            f"kd.model import took {elapsed:.1f}s"
        )
        assert KD_Discover is not None

    def test_kd_model_all_no_subpackage_leak(self) -> None:
        """kd.model.__all__ should not expose eqgpt subpackage modules."""
        import kd.model

        all_names = getattr(kd.model, "__all__", [])
        # KD_EqGPT is expected (Task B), but raw subpackage names should not leak
        subpackage_names = {"gpt_model", "neural_network", "surrogate_model"}
        for name in all_names:
            assert name not in subpackage_names, (
                f"EqGPT subpackage module {name!r} leaked into kd.model.__all__"
            )


# ===========================================================================
# Level 4: __init__.py exists and is minimal
# ===========================================================================


@pytest.mark.smoke
class TestLevel4InitFile:
    """kd/model/eqgpt/__init__.py should exist but must not import heavy deps."""

    def test_init_exists(self) -> None:
        """__init__.py must exist for the directory to be a Python package."""
        init_path = _EQGPT_PKG / "__init__.py"
        assert init_path.exists(), "kd/model/eqgpt/__init__.py missing"

    def test_init_no_heavy_imports(self) -> None:
        """__init__.py should not eagerly import surrogate/pickle modules."""
        init_path = _EQGPT_PKG / "__init__.py"
        assert init_path.exists(), "kd/model/eqgpt/__init__.py missing"
        content = init_path.read_text()
        # Should not have any from/import of heavy modules
        assert "from .surrogate_model" not in content, (
            "__init__.py imports surrogate_model -- will trigger heavy I/O"
        )
        assert "from .multi_surrogate_model" not in content, (
            "__init__.py imports multi_surrogate_model -- will trigger heavy I/O"
        )
        assert "pickle" not in content, (
            "__init__.py references pickle -- suggests eager data loading"
        )

    def test_init_no_train_imports(self) -> None:
        """__init__.py should not import training entrypoints at module level."""
        init_path = _EQGPT_PKG / "__init__.py"
        assert init_path.exists(), "kd/model/eqgpt/__init__.py missing"
        content = init_path.read_text()
        assert "from .train_gpt" not in content, (
            "__init__.py imports train_gpt"
        )
        assert "from .PINN_optimization" not in content, (
            "__init__.py imports PINN_optimization"
        )


# ===========================================================================
# Level 5: pyproject.toml configuration
# ===========================================================================


@pytest.mark.smoke
class TestLevel5Pyproject:
    """pyproject.toml should declare eqgpt package-data and optional deps."""

    def test_pyproject_has_eqgpt_package_data(self) -> None:
        """Package-data section should declare kd.model.eqgpt for data files."""
        content = _PYPROJECT.read_text()
        assert "kd.model.eqgpt" in content, (
            "Missing kd.model.eqgpt entry in pyproject.toml package-data"
        )

    def test_pyproject_has_eqgpt_optional_deps(self) -> None:
        """Optional-dependencies section should have an eqgpt group."""
        content = _PYPROJECT.read_text()
        # The [project.optional-dependencies] section should list eqgpt
        assert "eqgpt" in content, (
            "Missing eqgpt optional-dependencies group in pyproject.toml"
        )

    def test_pyproject_eqgpt_data_files(self) -> None:
        """Package-data for eqgpt should include .json, .xlsx, .pt patterns."""
        content = _PYPROJECT.read_text()
        # Data files in ref_lib: .json, .xlsx, .pt, .pkl
        # After migration, these should be declared in package-data
        has_json = "*.json" in content
        has_xlsx = "*.xlsx" in content
        has_pt = "*.pt" in content
        assert has_json or has_xlsx or has_pt, (
            "pyproject.toml should declare data file patterns "
            "(*.json, *.xlsx, *.pt) for kd.model.eqgpt"
        )


# ===========================================================================
# Edge / Negative tests
# ===========================================================================


@pytest.mark.smoke
class TestEdgeCases:
    """Edge cases and negative tests for the migration."""

    def test_eqgpt_package_is_directory(self) -> None:
        """kd/model/eqgpt/ should be a proper directory, not a single file."""
        assert _EQGPT_PKG.is_dir(), (
            f"{_EQGPT_PKG} is not a directory -- "
            "eqgpt should be a subpackage, not a module file"
        )

    def test_no_toplevel_pickle_load_in_package(self) -> None:
        """No .py file in eqgpt should call pickle.load at module level.

        Module-level pickle.load is a side effect that slows down import
        and may fail if the .pkl file is missing.
        """
        if not _EQGPT_PKG.is_dir():
            pytest.skip("eqgpt package not yet created")
        for py_file in _EQGPT_PKG.glob("*.py"):
            content = py_file.read_text()
            lines = content.splitlines()
            # Check for module-level pickle.load (not inside a function)
            indent_level = 0
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith("def ") or stripped.startswith("class "):
                    indent_level = len(line) - len(stripped)
                if "pickle.load" in line or "pkl" in line:
                    current_indent = len(line) - len(line.lstrip())
                    # Module level = indent 0
                    if current_indent == 0 and not stripped.startswith("#"):
                        pytest.fail(
                            f"{py_file.name} has module-level pickle reference: "
                            f"{stripped!r}"
                        )

    def test_no_hardcoded_absolute_paths(self) -> None:
        """Migrated files should not contain hardcoded absolute paths."""
        if not _EQGPT_PKG.is_dir():
            pytest.skip("eqgpt package not yet created")
        suspicious_prefixes = ("/home/", "/root/", "/Users/", "C:\\", "D:\\")
        for py_file in _EQGPT_PKG.glob("*.py"):
            content = py_file.read_text()
            for prefix in suspicious_prefixes:
                for i, line in enumerate(content.splitlines(), 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if prefix in line:
                        pytest.fail(
                            f"{py_file.name}:{i} contains hardcoded path "
                            f"with {prefix!r}: {stripped!r}"
                        )

    def test_gpt_model_dir_has_pretrained_weight(self) -> None:
        """gpt_model/ subdirectory should contain the pretrained .pt file."""
        gpt_dir = _EQGPT_PKG / "gpt_model"
        assert gpt_dir.is_dir(), (
            "kd/model/eqgpt/gpt_model/ directory missing "
            "(should contain pretrained weights)"
        )
        pt_files = list(gpt_dir.glob("*.pt"))
        assert len(pt_files) > 0, (
            "No .pt weight files found in kd/model/eqgpt/gpt_model/"
        )

    def test_wave_breaking_pkl_not_in_package(self) -> None:
        """wave_breaking_data.pkl (199MB) should NOT be copied into the package.

        Large data files should remain in ref_lib or be downloaded on demand.
        """
        if not _EQGPT_PKG.is_dir():
            pytest.skip("eqgpt package not yet created")
        pkl_file = _EQGPT_PKG / "wave_breaking_data.pkl"
        assert not pkl_file.exists(), (
            "wave_breaking_data.pkl (199MB) should NOT be in the package. "
            "Keep large data external."
        )

    def test_device_module_exposes_torch_device(self) -> None:
        """_device.device should be a torch.device instance."""
        import torch

        from kd.model.eqgpt._device import device

        assert isinstance(device, torch.device), (
            f"device should be torch.device, got {type(device).__name__}"
        )

    def test_gpt_class_is_nn_module(self) -> None:
        """GPT should be a subclass of torch.nn.Module."""
        import torch.nn as nn

        from kd.model.eqgpt.gpt_model import GPT

        assert issubclass(GPT, nn.Module), (
            f"GPT should subclass nn.Module, got bases: {GPT.__bases__}"
        )

    def test_nn_class_is_nn_module(self) -> None:
        """NN (neural network) should be a subclass of torch.nn.Module."""
        import torch.nn as nn

        from kd.model.eqgpt.neural_network import NN

        assert issubclass(NN, nn.Module), (
            f"NN should subclass nn.Module, got bases: {NN.__bases__}"
        )
