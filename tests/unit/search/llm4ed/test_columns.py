
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

FloatArray = npt.NDArray[np.float64]

_OPERANDS = ("x", "u_x", "u_xx", "u_xxx", "u")


@pytest.fixture()
def features() -> dict[str, FloatArray]:
    rng = np.random.default_rng(1234)
    n = 40
    return {
        "u": (rng.standard_normal(n) + 2.0).astype(np.float64),
        "u_x": (rng.standard_normal(n) * 0.5).astype(np.float64),
        "u_xx": (rng.standard_normal(n) + 3.0).astype(np.float64),
        "u_xxx": (rng.standard_normal(n) * 1.1).astype(np.float64),
        "x": (rng.standard_normal(n) + 5.0).astype(np.float64),
    }


class TestEvaluateIrColumn:

    def test_coefficient_in_column(self, features: dict[str, FloatArray]) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        col = evaluate_ir_column("mul(0.1, u_xx)", features)
        assert col.dtype == np.float64
        np.testing.assert_allclose(col, 0.1 * features["u_xx"], rtol=1e-13, atol=0.0)

    def test_n2(self, features: dict[str, FloatArray]) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        col = evaluate_ir_column("n2(u)", features)
        np.testing.assert_allclose(col, features["u"] ** 2, rtol=1e-13, atol=0.0)

    def test_n3(self, features: dict[str, FloatArray]) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        col = evaluate_ir_column("n3(u)", features)
        np.testing.assert_allclose(col, features["u"] ** 3, rtol=1e-13, atol=0.0)

    def test_composed_power_four(self, features: dict[str, FloatArray]) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        col = evaluate_ir_column("n2(n2(u))", features)
        np.testing.assert_allclose(col, features["u"] ** 4, rtol=1e-13, atol=0.0)

    def test_composed_power_five(self, features: dict[str, FloatArray]) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        col = evaluate_ir_column("mul(n2(n2(u)), u)", features)
        np.testing.assert_allclose(col, features["u"] ** 5, rtol=1e-13, atol=0.0)

    def test_raw_division(self, features: dict[str, FloatArray]) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        col = evaluate_ir_column("div(1.0, x)", features)
        np.testing.assert_allclose(col, 1.0 / features["x"], rtol=1e-13, atol=0.0)

    def test_nested_mul(self, features: dict[str, FloatArray]) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        col = evaluate_ir_column("mul(mul(u, u_x), u_xx)", features)
        expected = features["u"] * features["u_x"] * features["u_xx"]
        np.testing.assert_allclose(col, expected, rtol=1e-13, atol=0.0)

    def test_column_is_flat_float64(self, features: dict[str, FloatArray]) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        col = evaluate_ir_column("u", features)
        assert col.ndim == 1
        assert col.dtype == np.float64
        assert col.shape == features["u"].shape


class TestRawRegistryDivergesFromDefault:

    def test_div_is_raw_not_safe(self) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        feats = {"u": np.array([1.0, 1.0]), "x": np.array([2.0, 0.0])}
        col = evaluate_ir_column("div(u, x)", feats)

        assert col[0] == pytest.approx(0.5)
        assert np.isinf(col[1])

    def test_recip_is_raw_not_safe(self) -> None:



        from kd.search.llm4ed.columns import evaluate_ir_column

        feats = {"x": np.array([2.0, 0.0])}
        col = evaluate_ir_column("recip(x)", feats)
        assert col[0] == pytest.approx(0.5)
        assert np.isinf(col[1])

    def test_n2_is_not_clamped(self) -> None:
        from kd.search.llm4ed.columns import evaluate_ir_column

        feats = {"u": np.array([2.0e6, 3.0e6])}
        col = evaluate_ir_column("n2(u)", feats)
        np.testing.assert_allclose(col, np.array([4.0e12, 9.0e12]), rtol=1e-13)


class TestBuildColumns:
    def test_multi_term_columns(self, features: dict[str, FloatArray]) -> None:
        from kd.search.llm4ed import parse
        from kd.search.llm4ed.columns import build_columns

        parsed = parse.parse_equation("u*u_x + 0.1*u_xx", _OPERANDS)
        result = build_columns(parsed, features)
        assert result.valid
        assert len(result.columns) == 2
        assert len(result.term_strs) == 2
        for col in result.columns:
            assert col.dtype == np.float64
            assert col.shape == features["u"].shape

    def test_nonfinite_column_invalidates(self) -> None:
        from kd.search.llm4ed import parse
        from kd.search.llm4ed.columns import build_columns

        feats = {
            "u": np.array([1.0, 2.0, 3.0]),
            "u_x": np.array([1.0, 1.0, 1.0]),
            "u_xx": np.array([1.0, 1.0, 1.0]),
            "u_xxx": np.array([1.0, 1.0, 1.0]),
            "x": np.array([1.0, 0.0, 2.0]),
        }
        parsed = parse.parse_equation("u/x", _OPERANDS)
        result = build_columns(parsed, feats)
        assert result.valid is False
        assert result.error is not None

    def test_coefficient_factor_kept_in_column(
        self, features: dict[str, FloatArray]
    ) -> None:
        from kd.search.llm4ed import parse
        from kd.search.llm4ed.columns import build_columns

        parsed = parse.parse_equation("3*u", _OPERANDS)
        result = build_columns(parsed, features)
        assert result.valid
        assert len(result.columns) == 1
        np.testing.assert_allclose(
            result.columns[0], 3.0 * features["u"], rtol=1e-13, atol=0.0
        )
