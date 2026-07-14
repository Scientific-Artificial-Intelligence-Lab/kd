
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from kd.data.loaders import wave_breaking as wb
from kd.data.loaders.wave_breaking import WaveBreakingCase







def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[4]


_WAVE_PKL = _find_repo_root() / "data" / "hf-knowledgediscover" / "WaveBreaking.pkl"

skip_no_wave_data = pytest.mark.skipif(
    not _WAVE_PKL.exists(),
    reason=f"wave-breaking pickle not found: {_WAVE_PKL}",
)






_SPEC_GRAVITY = 9.81


def _expected_lamda(tp_seconds: float) -> float:
    return _SPEC_GRAVITY * tp_seconds**2 / (2.0 * math.pi)







def _make_case_array(n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, 5.25, n)
    x = np.linspace(8.15, 12.55, n)
    eta = 0.05 * rng.standard_normal(n)
    return np.stack([t, x, eta], axis=1)


def _valid_synthetic_cases() -> dict[str, np.ndarray]:
    return {
        "N_G2Tp12A080_broad": _make_case_array(seed=1),
        "L_G3Tp13A105_broad": _make_case_array(seed=2),
    }


def _write_pkl(path: Path, obj: object) -> Path:
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return path







class TestLoaderSmoke:

    @pytest.mark.smoke
    def test_loader_callable(self) -> None:
        assert callable(wb.load_wave_breaking_cases)

    @pytest.mark.smoke
    def test_exported_from_package(self) -> None:
        from kd.data import loaders

        assert callable(loaders.load_wave_breaking_cases)
        assert loaders.WaveBreakingCase is WaveBreakingCase







class TestLoaderHappyPath:

    @pytest.mark.unit
    def test_returns_dict_of_cases(self, tmp_path: Path) -> None:
        pkl = _write_pkl(tmp_path / "wave.pkl", _valid_synthetic_cases())
        cases = wb.load_wave_breaking_cases(pkl)
        assert set(cases) == {"N_G2Tp12A080_broad", "L_G3Tp13A105_broad"}
        assert all(isinstance(c, WaveBreakingCase) for c in cases.values())
        for key, case in cases.items():
            assert case.name == key

    @pytest.mark.unit
    def test_scatter_arrays_1d_equal_length(self, tmp_path: Path) -> None:
        pkl = _write_pkl(tmp_path / "wave.pkl", _valid_synthetic_cases())
        case = wb.load_wave_breaking_cases(pkl)["N_G2Tp12A080_broad"]
        assert case.t.dim() == 1 and case.x.dim() == 1 and case.eta.dim() == 1
        n = case.t.numel()
        assert case.x.numel() == n and case.eta.numel() == n
        assert n == _make_case_array(seed=1).shape[0]

    @pytest.mark.unit
    def test_scatter_columns_mapped_in_order(self, tmp_path: Path) -> None:
        arr = _make_case_array(seed=7)
        pkl = _write_pkl(tmp_path / "wave.pkl", {"N_G2Tp12A080_broad": arr})
        case = wb.load_wave_breaking_cases(pkl)["N_G2Tp12A080_broad"]
        torch.testing.assert_close(
            case.t.to(torch.float64),
            torch.tensor(arr[:, 0], dtype=torch.float64),
            rtol=0.0,
            atol=1e-10,
        )
        torch.testing.assert_close(
            case.x.to(torch.float64),
            torch.tensor(arr[:, 1], dtype=torch.float64),
            rtol=0.0,
            atol=1e-10,
        )
        torch.testing.assert_close(
            case.eta.to(torch.float64),
            torch.tensor(arr[:, 2], dtype=torch.float64),
            rtol=0.0,
            atol=1e-10,
        )

    @pytest.mark.unit
    def test_metadata_parsed_from_name(self, tmp_path: Path) -> None:
        pkl = _write_pkl(tmp_path / "wave.pkl", _valid_synthetic_cases())
        case = wb.load_wave_breaking_cases(pkl)["N_G2Tp12A080_broad"]
        assert case.g == 2
        assert case.a == 80
        assert case.prefix == "N"
        assert case.tp_seconds == pytest.approx(1.2, rel=1e-9)

    @pytest.mark.unit
    def test_l_prefix_parsed(self, tmp_path: Path) -> None:
        pkl = _write_pkl(tmp_path / "wave.pkl", _valid_synthetic_cases())
        case = wb.load_wave_breaking_cases(pkl)["L_G3Tp13A105_broad"]
        assert case.g == 3
        assert case.a == 105
        assert case.prefix == "L"
        assert case.tp_seconds == pytest.approx(1.3, rel=1e-9)

    @pytest.mark.unit
    def test_single_digit_tp_parsed(self, tmp_path: Path) -> None:
        pkl = _write_pkl(
            tmp_path / "wave.pkl", {"N_G2Tp09A080_broad": _make_case_array()}
        )
        case = wb.load_wave_breaking_cases(pkl)["N_G2Tp09A080_broad"]
        assert case.tp_seconds == pytest.approx(0.9, rel=1e-9)

    @pytest.mark.unit
    def test_wavelength_derived(self, tmp_path: Path) -> None:
        pkl = _write_pkl(tmp_path / "wave.pkl", _valid_synthetic_cases())
        case = wb.load_wave_breaking_cases(pkl)["N_G2Tp12A080_broad"]
        assert case.lamda == pytest.approx(_expected_lamda(1.2), rel=1e-9)

    @pytest.mark.unit
    def test_default_path_no_arg(self) -> None:
        if _WAVE_PKL.exists():
            cases = wb.load_wave_breaking_cases()
            assert len(cases) == 23
        else:
            with pytest.raises(FileNotFoundError):
                wb.load_wave_breaking_cases()







class TestLoaderNegative:

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            wb.load_wave_breaking_cases(tmp_path / "does_not_exist.pkl")

    @pytest.mark.unit
    def test_non_dict_pickle_raises(self, tmp_path: Path) -> None:
        pkl = _write_pkl(tmp_path / "bad.pkl", [1, 2, 3])
        with pytest.raises(ValueError):
            wb.load_wave_breaking_cases(pkl)

    @pytest.mark.unit
    def test_wrong_column_count_raises_naming_key(self, tmp_path: Path) -> None:
        bad = {"N_G2Tp12A080_broad": np.zeros((10, 2))}
        pkl = _write_pkl(tmp_path / "bad.pkl", bad)
        with pytest.raises(ValueError, match="N_G2Tp12A080_broad"):
            wb.load_wave_breaking_cases(pkl)

    @pytest.mark.unit
    def test_transposed_array_raises_naming_key(self, tmp_path: Path) -> None:
        bad = {"N_G2Tp12A080_broad": np.zeros((3, 50))}
        pkl = _write_pkl(tmp_path / "bad.pkl", bad)
        with pytest.raises(ValueError, match="N_G2Tp12A080_broad"):
            wb.load_wave_breaking_cases(pkl)

    @pytest.mark.unit
    def test_non_2d_value_raises_naming_key(self, tmp_path: Path) -> None:
        bad = {"N_G2Tp12A080_broad": np.zeros(30)}
        pkl = _write_pkl(tmp_path / "bad.pkl", bad)
        with pytest.raises(ValueError, match="N_G2Tp12A080_broad"):
            wb.load_wave_breaking_cases(pkl)

    @pytest.mark.unit
    def test_unparseable_name_raises_naming_key(self, tmp_path: Path) -> None:
        bad = {"totally_bogus_name": _make_case_array()}
        pkl = _write_pkl(tmp_path / "bad.pkl", bad)
        with pytest.raises(ValueError, match="totally_bogus_name"):
            wb.load_wave_breaking_cases(pkl)







class TestLoaderRealData:

    @skip_no_wave_data
    @pytest.mark.unit
    def test_real_pkl_shape(self) -> None:
        cases = wb.load_wave_breaking_cases(_WAVE_PKL)
        assert len(cases) == 23
        names = list(cases)
        assert sum(n.startswith("N_") for n in names) == 12
        assert sum(n.startswith("L_") for n in names) == 11
        assert all(n.endswith("_broad") for n in names)
        sample = cases["N_G2Tp12A080_broad"]
        assert sample.t.numel() == sample.x.numel() == sample.eta.numel()
        assert sample.t.numel() > 0
