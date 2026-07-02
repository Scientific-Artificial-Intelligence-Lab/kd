
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import kd
import kd.data.regression
from kd.data.regression import TabularDataset, load_tlc_cc, load_wave_breaking

_N_SAMPLES = 74
_VAR_NAMES = ("R_F", "r")


class TestLoadTlcCc:
    def test_default_target_is_start(self) -> None:
        ds = load_tlc_cc()
        assert ds.target_name == "V_S"

    def test_shapes_and_dtypes(self) -> None:
        ds = load_tlc_cc()
        assert ds.X.shape == (_N_SAMPLES, 2)
        assert ds.y.shape == (_N_SAMPLES,)
        assert ds.X.dtype == np.float64
        assert ds.y.dtype == np.float64

    def test_all_values_finite(self) -> None:
        for target in ("start", "end"):
            ds = load_tlc_cc(target=target)
            assert np.all(np.isfinite(ds.X))
            assert np.all(np.isfinite(ds.y))

    def test_var_names(self) -> None:
        ds = load_tlc_cc()
        assert ds.var_names == _VAR_NAMES

    def test_end_target_name(self) -> None:
        ds = load_tlc_cc(target="end")
        assert ds.target_name == "V_E"

    def test_feature_ranges_physical(self) -> None:
        ds = load_tlc_cc()
        r_f, r = ds.X[:, 0], ds.X[:, 1]
        assert r_f.min() >= 0.0 and r_f.max() <= 1.0
        assert r.min() >= 0.0 and r.max() <= 1.0
        assert np.all(ds.y > 0.0)

    def test_end_volume_exceeds_start_volume(self) -> None:
        start = load_tlc_cc(target="start")
        end = load_tlc_cc(target="end")
        np.testing.assert_allclose(start.X, end.X)
        assert np.all(end.y > start.y)

    def test_invalid_target_raises(self) -> None:
        with pytest.raises(ValueError, match="start.*end|end.*start"):
            load_tlc_cc(target="middle")

    def test_metadata_provenance(self) -> None:
        ds = load_tlc_cc()
        assert ds.name == "tlc-cc-start"
        assert "Nat Commun 16, 832 (2025)" in ds.source
        assert "s41467-025-56136-x" in ds.source
        assert ds.description
        assert load_tlc_cc(target="end").name == "tlc-cc-end"

    def test_dataset_is_frozen(self) -> None:
        ds = load_tlc_cc()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ds.target_name = "other"


class TestLoadWaveBreaking:
    _BUNDLED_CASE = "N_G2Tp12A100_broad"
    _N_POINTS = 314478

    def test_bundled_case_loads(self) -> None:
        ds = load_wave_breaking()
        assert ds.name == f"wave-breaking-{self._BUNDLED_CASE}"
        assert ds.X.shape == (self._N_POINTS, 2)
        assert ds.y.shape == (self._N_POINTS,)
        assert ds.X.dtype == np.float64
        assert ds.y.dtype == np.float64

    def test_columns_semantics(self) -> None:
        ds = load_wave_breaking()
        assert ds.var_names == ("t", "x")
        assert ds.target_name == "eta"
        t, x = ds.X[:, 0], ds.X[:, 1]


        assert t.min() > 0.0 and t.max() < 10.0
        assert x.min() > 8.0 and x.max() < 13.0
        assert np.abs(ds.y).max() < 1.0

    def test_all_values_finite(self) -> None:
        ds = load_wave_breaking()
        assert np.all(np.isfinite(ds.X))
        assert np.all(np.isfinite(ds.y))

    def test_metadata_provenance(self) -> None:
        ds = load_wave_breaking()
        assert "Nat Commun 16, 10255 (2025)" in ds.source
        assert "s41467-025-65114-2" in ds.source
        assert "wave" in ds.description.lower()

    def test_unknown_case_fails_loud(self) -> None:
        with pytest.raises(FileNotFoundError, match="N_G2Tp12A100_broad"):
            load_wave_breaking(case="X_NOSUCHCASE")

    def test_data_dir_override(self, tmp_path: Path) -> None:
        bundled = (
            Path(kd.data.regression.__file__).resolve().parents[1]
            / "_assets"
            / "data"
            / f"wave_breaking_{self._BUNDLED_CASE}.npz"
        )
        target = tmp_path / f"wave_breaking_{self._BUNDLED_CASE}.npz"
        target.write_bytes(bundled.read_bytes())
        ds = load_wave_breaking(case=self._BUNDLED_CASE, data_dir=tmp_path)
        assert ds.X.shape == (self._N_POINTS, 2)

    def test_data_dir_override_missing_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="wave_breaking"):
            load_wave_breaking(case=self._BUNDLED_CASE, data_dir=tmp_path)


class TestPublicExports:
    def test_top_level_exports(self) -> None:
        assert kd.load_tlc_cc is load_tlc_cc
        assert kd.load_wave_breaking is load_wave_breaking
        assert kd.TabularDataset is TabularDataset

    def test_data_module_exports(self) -> None:
        from kd.data import load_tlc_cc as from_data
        from kd.data import load_wave_breaking as wb_from_data

        assert from_data is load_tlc_cc
        assert wb_from_data is load_wave_breaking
