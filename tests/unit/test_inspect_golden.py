




from __future__ import annotations

import io

import pytest

from kd.data.schema import PDEDataset
from kd.inspect import preview
from tests.unit import _inspect_cases as cases


def _render(dataset: PDEDataset) -> str:
    buf = io.StringIO()
    preview(dataset, file=buf)
    return buf.getvalue()


_BUILDERS: dict[str, str] = {
    "clean_uniform": "clean_uniform_dataset",
    "non_uniform": "non_uniform_dataset",
    "descending": "descending_dataset",
    "non_finite_dx": "non_finite_dx_dataset",
    "nan_inf_field": "nan_inf_field_dataset",
    "all_non_finite_field": "all_non_finite_field_dataset",
    "small_grid": "small_grid_dataset",
    "single_point_axis": "single_point_axis_dataset",
    "lhs_unset": "lhs_unset_dataset",
    "mixed_dtype": "mixed_dtype_dataset",
    "wave_lhs": "wave_lhs_dataset",
    "metadata_only": "metadata_only_dataset",
}

_GOLDEN: dict[str, str] = {
    "clean_uniform": """Dataset: probe_clean
Axes:
     x | n=32 | range [0.000, 1.000] | step 0.03226 (uniform)
     t | n=20 | range [0.000, 2.000] | step 0.1053 (uniform)
Fields:
     u | dtype=float64 | shape=(32, 20) | min=0.000 max=6.390 mean=3.195 (NaN=0)
LHS: u_t (field='u', axis='t')
Status: ready to fit
""",
    "non_uniform": """Dataset: probe_nonuniform
Axes:
     x | n=32 | range [0.000, 1.000] | step 0.03226 (NON-UNIFORM, range 0.001041-0.06348)
     t | n=20 | range [0.000, 2.000] | step 0.1053 (uniform)
Fields:
     u | dtype=float64 | shape=(32, 20) | min=0.000 max=6.390 mean=3.195 (NaN=0)
LHS: u_t (field='u', axis='t')
Status: 1 warning(s)
  - WARNING: axis 'x' is not uniformly spaced (steps from 0.001041 to 0.06348) — finite-difference derivatives assume uniform grids
""",
    "descending": """Dataset: probe_descending
Axes:
     x | n=5 | range [0.000, 0.400] | step -0.1 (NON-UNIFORM, decreasing)
     t | n=8 | range [0.000, 1.000] | step 0.1429 (uniform)
Fields:
     u | dtype=float64 | shape=(5, 8) | min=0.000 max=0.390 mean=0.195 (NaN=0)
LHS: u_t (field='u', axis='t')
Status: 3 warning(s)
  - WARNING: axis 'x' has decreasing spacing (dx0=-0.1) — finite-difference stencils require monotonic increasing coordinates (flip the array before fitting)
  - WARNING: axis 'x' has only 5 points (<16) — small grid (n<16) may make fits unstable
  - WARNING: axis 't' has only 8 points (<16) — small grid (n<16) may make fits unstable
""",
    "non_finite_dx": """Dataset: probe_inf_dx
Axes:
     x | n=5 | range [-300000000549775575777803994281145270272.000, 300000000549775575777803994281145270272.000] | step inf (NON-UNIFORM, non-finite spacing)
     t | n=8 | range [0.000, 1.000] | step 0.1429 (uniform)
Fields:
     u | dtype=float64 | shape=(5, 8) | min=0.000 max=0.390 mean=0.195 (NaN=0)
LHS: u_t (field='u', axis='t')
Status: 3 warning(s)
  - WARNING: axis 'x' has non-finite spacing dx0=inf — finite-difference stencils require finite dx
  - WARNING: axis 'x' has only 5 points (<16) — small grid (n<16) may make fits unstable
  - WARNING: axis 't' has only 8 points (<16) — small grid (n<16) may make fits unstable
""",
    "nan_inf_field": """Dataset: probe_nan_inf
Axes:
     x | n=20 | range [0.000, 1.000] | step 0.05263 (uniform)
     t | n=12 | range [0.000, 1.000] | step 0.09091 (uniform)
Fields:
     u | dtype=float64 | shape=(20, 12) | min=0.010 max=2.390 mean=1.204 (NaN=1 (>0) Inf=1 (>0))
LHS: u_t (field='u', axis='t')
Status: 3 warning(s)
  - WARNING: axis 't' has only 12 points (<16) — small grid (n<16) may make fits unstable
  - WARNING: field 'u' contains 1 NaN value(s) — dataset will not fit until cleaned
  - WARNING: field 'u' contains 1 Inf value(s) — dataset will not fit until cleaned
""",
    "all_non_finite_field": """Dataset: probe_all_non_finite
Axes:
     x | n=20 | range [0.000, 1.000] | step 0.05263 (uniform)
     t | n=12 | range [0.000, 1.000] | step 0.09091 (uniform)
Fields:
     u | dtype=float64 | shape=(20, 12) | min=nan max=nan mean=nan (NaN=240 (>0))
LHS: u_t (field='u', axis='t')
Status: 2 warning(s)
  - WARNING: axis 't' has only 12 points (<16) — small grid (n<16) may make fits unstable
  - WARNING: field 'u' contains 240 NaN value(s) — dataset will not fit until cleaned
""",
    "small_grid": """Dataset: probe_small_grid
Axes:
     x | n=8 | range [0.000, 1.000] | step 0.1429 (uniform)
     t | n=20 | range [0.000, 2.000] | step 0.1053 (uniform)
Fields:
     u | dtype=float64 | shape=(8, 20) | min=0.000 max=1.590 mean=0.795 (NaN=0)
LHS: u_t (field='u', axis='t')
Status: 1 warning(s)
  - WARNING: axis 'x' has only 8 points (<16) — small grid (n<16) may make fits unstable
""",
    "single_point_axis": """Dataset: probe_single_point
Axes:
     x | n=1 | range [0.500, 0.500] | step n/a (single point)
     t | n=20 | range [0.000, 2.000] | step 0.1053 (uniform)
Fields:
     u | dtype=float64 | shape=(1, 20) | min=0.000 max=0.190 mean=0.095 (NaN=0)
LHS: u_t (field='u', axis='t')
Status: 1 warning(s)
  - WARNING: axis 'x' has only 1 points (<16) — small grid (n<16) may make fits unstable
""",
    "lhs_unset": """Dataset: probe_lhs_unset
Axes:
     x | n=32 | range [0.000, 1.000] | step 0.03226 (uniform)
     t | n=20 | range [0.000, 2.000] | step 0.1053 (uniform)
Fields:
     u | dtype=float64 | shape=(32, 20) | min=0.000 max=6.390 mean=3.195 (NaN=0)
LHS: (unset)
Status: 1 warning(s)
  - WARNING: lhs_field / lhs_axis not set — Model.fit() will fall back to ('u', 't')
""",
    "mixed_dtype": """Dataset: probe_mixed_dtype
Axes:
     x | n=32 | range [0.000, 1.000] | step 0.03226 (uniform)
     t | n=20 | range [0.000, 2.000] | step 0.1053 (uniform)
Fields:
     u | dtype=float64 | shape=(32, 20) | min=0.000 max=6.390 mean=3.195 (NaN=0)
     v | dtype=float32 | shape=(32, 20) | min=0.000 max=6.390 mean=3.195 (NaN=0)
LHS: u_t (field='u', axis='t')
Status: 1 warning(s)
  - WARNING: fields have mixed dtypes ['torch.float32', 'torch.float64'] — consider casting all fields to the same dtype
""",
    "wave_lhs": """Dataset: probe_wave
Axes:
     x | n=32 | range [0.000, 1.000] | step 0.03226 (uniform)
     t | n=20 | range [0.000, 2.000] | step 0.1053 (uniform)
Fields:
     u | dtype=float64 | shape=(32, 20) | min=0.000 max=6.390 mean=3.195 (NaN=0)
LHS: u_t (field='u', axis='t')
Status: ready to fit
""",
    "metadata_only": """Dataset: probe_metadata_only
Axes:
  (none)
Fields:
  (none)
LHS: (unset)
Status: 1 warning(s)
  - WARNING: lhs_field / lhs_axis not set — Model.fit() will fall back to ('u', 't')
""",
}


@pytest.mark.parametrize(("case_name", "expected"), sorted(_GOLDEN.items()))
def test_preview_output_matches_byte_golden(case_name: str, expected: str) -> None:
    dataset = getattr(cases, _BUILDERS[case_name])()

    assert _render(dataset) == expected


def test_golden_covers_every_registered_case() -> None:
    assert set(_GOLDEN) == set(_BUILDERS)


def test_clean_case_carries_no_warning_tokens() -> None:
    text = _GOLDEN["clean_uniform"]

    assert "NON-UNIFORM" not in text
    assert "WARNING:" not in text
    assert text.endswith("Status: ready to fit\n")
