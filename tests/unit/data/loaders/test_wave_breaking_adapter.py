
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kd.data.loaders.wave_breaking import WaveBreakingCase
from kd.data.schema import DataTopology






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







def _synthetic_case(*, n: int = 6, eta_len: int | None = None) -> WaveBreakingCase:
    t = torch.tensor([0.3, 0.0, 0.5, 0.1, 0.4, 0.2], dtype=torch.float64)[:n]
    x = torch.tensor([9.0, 1.0, 4.0, 2.0, 7.0, 3.0], dtype=torch.float64)[:n]
    m = n if eta_len is None else eta_len
    eta = torch.linspace(-1.0, 1.0, m, dtype=torch.float64)
    return WaveBreakingCase(
        name="N_G2Tp12A080_broad",
        t=t,
        x=x,
        eta=eta,
        g=2,
        tp_seconds=1.2,
        a=80,
        lamda=2.25,
        prefix="N",
    )







class TestWaveBreakingAdapter:

    @pytest.mark.unit
    def test_produces_scattered_dataset(self) -> None:
        from kd.data.loaders.wave_breaking import (
            wave_breaking_case_to_dataset,
        )

        case = _synthetic_case(n=6)
        ds = wave_breaking_case_to_dataset(case)

        assert ds.topology == DataTopology.SCATTERED
        assert ds.axis_order == ["t", "x"]
        assert ds.lhs_field == "u"
        assert ds.lhs_axis == "t"
        assert ds.lhs_order == 1
        assert ds.name == f"wave-breaking-{case.name}"
        assert ds.get_shape() == (len(case.t),)
        assert ds.spatial_axes == ["x"]

    @pytest.mark.unit
    def test_field_u_equals_eta_float64(self) -> None:
        from kd.data.loaders.wave_breaking import (
            wave_breaking_case_to_dataset,
        )

        case = _synthetic_case(n=6)
        ds = wave_breaking_case_to_dataset(case)
        u = ds.get_field("u")
        assert u.dtype == torch.float64
        assert torch.equal(u, case.eta.to(torch.float64))

    @pytest.mark.unit
    def test_length_mismatch_raises(self) -> None:
        from kd.data.loaders.wave_breaking import (
            wave_breaking_case_to_dataset,
        )

        case = _synthetic_case(n=6, eta_len=5)
        with pytest.raises(ValueError):
            wave_breaking_case_to_dataset(case)







class TestWaveBreakingAdapterRealData:

    @skip_no_wave_data
    @pytest.mark.unit
    def test_real_case_builds_scattered_dataset(self) -> None:
        from kd.data.loaders.wave_breaking import (
            load_wave_breaking_cases,
            wave_breaking_case_to_dataset,
        )

        cases = load_wave_breaking_cases()
        name, case = next(iter(cases.items()))
        ds = wave_breaking_case_to_dataset(case)
        assert ds.topology == DataTopology.SCATTERED
        assert ds.axis_order == ["t", "x"]
        assert ds.get_shape() == (len(case.t),)
        assert ds.name == f"wave-breaking-{name}"
