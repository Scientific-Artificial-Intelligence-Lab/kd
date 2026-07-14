
from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_OPERANDS = ("x", "u_x", "u_xx", "u_xxx", "u")


def _linear_features(n: int = 60, seed: int = 3) -> dict[str, FloatArray]:
    rng = np.random.default_rng(seed)
    return {
        "u": rng.standard_normal(n).astype(np.float64),
        "u_x": rng.standard_normal(n).astype(np.float64),
        "u_xx": rng.standard_normal(n).astype(np.float64),
        "u_xxx": rng.standard_normal(n).astype(np.float64),
        "x": rng.standard_normal(n).astype(np.float64),
    }


class TestValidScoring:
    def test_perfect_fit_reward(self) -> None:
        from kd.search.llm4ed.score import score_equation

        feats = _linear_features()
        lhs = (2.0 * feats["u"] + 3.0 * feats["u_xx"]).reshape(-1, 1)
        result = score_equation("u + u_xx", lhs, feats, operands=_OPERANDS)

        assert result.valid
        assert result.reward is not None
        assert result.coefficients is not None
        assert result.y_hat is not None

        assert result.reward > 0.95

    def test_reward_is_self_consistent(self) -> None:
        from kd.search.llm4ed.reward import sparse_reward
        from kd.search.llm4ed.score import score_equation

        feats = _linear_features()
        lhs = (2.0 * feats["u"] + 3.0 * feats["u_xx"]).reshape(-1, 1)
        result = score_equation("u + u_xx", lhs, feats, operands=_OPERANDS)

        assert result.valid and result.y_hat is not None
        expected = round(sparse_reward(lhs, result.y_hat, result.n_terms), 4)
        assert result.reward == expected

    def test_n_terms_is_capped_nonzero_count(self) -> None:
        from kd.search.llm4ed.score import MAX_TERMS, score_equation

        feats = _linear_features()
        lhs = (2.0 * feats["u"] + 3.0 * feats["u_xx"]).reshape(-1, 1)
        result = score_equation(
            "u + u_xx + u_x + u_xxx + x", lhs, feats, operands=_OPERANDS
        )
        assert result.valid and result.coefficients is not None
        expected = min(int(np.count_nonzero(result.coefficients)), MAX_TERMS)
        assert result.n_terms == expected

    def test_coefficient_recovery(self) -> None:
        from kd.search.llm4ed.score import score_equation

        feats = _linear_features()
        lhs = (2.0 * feats["u"] + 3.0 * feats["u_xx"]).reshape(-1, 1)
        result = score_equation("u + u_xx", lhs, feats, operands=_OPERANDS)
        assert result.valid and result.coefficients is not None

        recovered = np.sort(result.coefficients[result.coefficients != 0])
        np.testing.assert_allclose(recovered, np.array([2.0, 3.0]), rtol=1e-6)


class TestInvalidScoring:
    def test_undefined_operand(self) -> None:
        from kd.search.llm4ed.score import ERROR_UNDEFINED_OPERANDS, score_equation

        feats = _linear_features()
        lhs = feats["u"].reshape(-1, 1)
        result = score_equation("u*v", lhs, feats, operands=_OPERANDS)
        assert result.valid is False
        assert result.reward is None
        assert result.error_type == ERROR_UNDEFINED_OPERANDS

    def test_undefined_operator(self) -> None:
        from kd.search.llm4ed.score import ERROR_UNDEFINED_OPERATORS, score_equation

        feats = _linear_features()
        lhs = feats["u"].reshape(-1, 1)
        result = score_equation("u^7", lhs, feats, operands=_OPERANDS)
        assert result.valid is False
        assert result.error_type == ERROR_UNDEFINED_OPERATORS

    def test_nonfinite_column_dropped(self) -> None:
        from kd.search.llm4ed.score import ERROR_NON_FINITE, score_equation

        feats = _linear_features()
        feats["x"][0] = 0.0
        lhs = feats["u"].reshape(-1, 1)
        result = score_equation("u/x", lhs, feats, operands=_OPERANDS)
        assert result.valid is False
        assert result.reward is None



        assert result.error_type == ERROR_NON_FINITE

    def test_abnormal_coefficient(self) -> None:
        from kd.search.llm4ed.score import ERROR_ABNORMAL_COEF, score_equation

        feats = _linear_features()

        lhs = (1.0e6 * feats["u"]).reshape(-1, 1)
        result = score_equation("u", lhs, feats, operands=_OPERANDS)
        assert result.valid is False
        assert result.error_type == ERROR_ABNORMAL_COEF


class TestRemoveRedundants:
    def test_distinct_columns_unchanged(self) -> None:
        from kd.search.llm4ed.score import remove_redundants

        cols = [np.array([1.0, 2.0, 3.0]), np.array([0.5, -1.0, 4.0])]
        values, tokens, duplicate = remove_redundants(cols, ["u", "u_x"])
        assert duplicate is False
        assert tokens == ["u", "u_x"]
        assert len(values) == 2

    def test_duplicate_pair_keeps_shorter_string(self) -> None:
        from kd.search.llm4ed.score import remove_redundants

        dup = np.array([1.0, -2.0, 3.0])
        cols = [dup.copy(), dup.copy()]
        values, tokens, duplicate = remove_redundants(cols, ["u", "u*u_x"])
        assert duplicate is True
        assert len(values) == 1
        assert tokens == ["u"]
