
from __future__ import annotations

import ast
import inspect
import re

import numpy as np
import pytest




from kd.search.sindy import sr as sr_module
from kd.search.sindy.sr import SINDyRegressor

pytestmark = pytest.mark.unit











_RECOVERY_RTOL = 5e-3
_NMSE_ATOL = 1e-6


def _make_xy(
    n: int = 200,
    n_features: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features))
    y = 2.0 * np.sin(x[:, 0]) - 3.0 * (x[:, 0] * x[:, 1])
    return x, y



_TRUE_COEFFS: dict[str, float] = {"sin(x1)": 2.0, "mul(x1, x2)": -3.0}

_LIBRARY_WITH_DECOYS: list[str] = [
    "x1",
    "sin(x1)",
    "mul(x1, x2)",
    "x2",
    "cos(x2)",
    "mul(x2, x2)",
]


def _selected_map(reg: SINDyRegressor) -> dict[str, float]:
    assert reg.selected_terms_ is not None
    assert reg.coefficients_ is not None
    coeffs = np.asarray(reg.coefficients_, dtype=float).reshape(-1)
    assert len(coeffs) == len(reg.selected_terms_), (
        "coefficients_ must be positionally aligned with selected_terms_ "
        f"(got {len(coeffs)} coeffs for {len(reg.selected_terms_)} terms)"
    )
    return dict(zip(reg.selected_terms_, coeffs.tolist(), strict=True))







class TestFitPredictSurface:

    def test_fit_returns_self(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy()
        assert reg.fit(x, y, _LIBRARY_WITH_DECOYS) is reg

    def test_fit_sets_all_post_fit_attributes(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy()
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)

        assert isinstance(reg.selected_terms_, list)
        assert all(isinstance(t, str) for t in reg.selected_terms_)
        assert reg.coefficients_ is not None
        assert isinstance(reg.expression_, str)
        assert reg.expression_
        assert isinstance(reg.nmse_, float)
        assert reg.var_names_ == ["x1", "x2"]
        assert reg.n_features_in_ == 2

    def test_predict_shape_is_1d(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n=120)
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)

        y_hat = reg.predict(x)
        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == (120,)

    def test_predict_reproduces_training_target(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n=180)
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)

        np.testing.assert_allclose(reg.predict(x), y, rtol=1e-4, atol=1e-6)

    def test_predict_on_fresh_samples(self) -> None:
        reg = SINDyRegressor()
        x_train, y_train = _make_xy(n=200, seed=1)
        reg.fit(x_train, y_train, _LIBRARY_WITH_DECOYS)

        rng = np.random.default_rng(99)
        x_new = rng.normal(size=(64, 2))
        y_new_true = 2.0 * np.sin(x_new[:, 0]) - 3.0 * (x_new[:, 0] * x_new[:, 1])
        np.testing.assert_allclose(reg.predict(x_new), y_new_true, rtol=1e-3, atol=1e-5)







class TestSparseRecovery:

    def test_recovers_exact_support(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n=200, seed=2)
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)

        assert set(reg.selected_terms_) == set(_TRUE_COEFFS)

    def test_recovers_coefficients(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n=200, seed=3)
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)

        recovered = _selected_map(reg)
        assert set(recovered) == set(_TRUE_COEFFS)
        for term, coef in _TRUE_COEFFS.items():
            assert recovered[term] == pytest.approx(coef, rel=_RECOVERY_RTOL)

    def test_nmse_near_zero_for_exact_fit(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n=200, seed=4)
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)

        assert isinstance(reg.nmse_, float)
        assert np.isfinite(reg.nmse_)
        assert reg.nmse_ == pytest.approx(0.0, abs=_NMSE_ATOL)

    def test_expression_renders_selected_terms_only(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n=200, seed=5)
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)

        expr = reg.expression_
        assert "sin" in expr
        assert "x1" in expr and "x2" in expr
        assert "cos" not in expr

    def test_empty_support_yields_clean_zero_model(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n=120, seed=12)
        reg.fit(x, y, ["sub(x1, x1)"])

        assert reg.selected_terms_ == []
        assert reg.expression_ == "0"
        assert np.isfinite(reg.nmse_)
        np.testing.assert_array_equal(reg.predict(x), np.zeros(120))







class TestFailLoudTerms:

    def test_unparseable_term_raises_naming_it(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy()
        bad = "sin(x1"
        with pytest.raises((ValueError, SyntaxError)) as excinfo:
            reg.fit(x, y, ["x1", bad, "mul(x1, x2)"])
        assert bad in str(excinfo.value)

    def test_unknown_function_raises(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy()
        with pytest.raises((ValueError, KeyError)) as excinfo:
            reg.fit(x, y, ["x1", "tanh(x1)"])
        assert "tanh" in str(excinfo.value)

    def test_unknown_variable_raises(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n_features=2)
        with pytest.raises((ValueError, KeyError)) as excinfo:
            reg.fit(x, y, ["x1", "sin(x9)"])
        assert "x9" in str(excinfo.value)

    def test_duplicate_terms_raise(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy()
        with pytest.raises(ValueError, match="sin"):
            reg.fit(x, y, ["x1", "sin(x1)", "sin(x1)"])

    def test_empty_terms_raise(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy()
        with pytest.raises(ValueError):
            reg.fit(x, y, [])

    def test_terms_not_silently_dropped(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy()
        with pytest.raises((ValueError, KeyError)):
            reg.fit(x, y, ["tanh(x1)", "sigmoid(x2)"])
        assert reg.selected_terms_ is None
        assert reg.coefficients_ is None

    def test_constant_term_raises_naming_it(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy()
        with pytest.raises(ValueError) as excinfo:
            reg.fit(x, y, ["x1", "1", "mul(x1, x2)"])
        assert "1" in str(excinfo.value)

        assert reg.selected_terms_ is None

    def test_constant_term_raises_even_for_single_sample(self) -> None:
        reg = SINDyRegressor()
        x = np.array([[0.5, 1.5]])
        y = np.array([2.0])
        with pytest.raises(ValueError) as excinfo:
            reg.fit(x, y, ["x1", "1"])
        assert "1" in str(excinfo.value)

    def test_derivative_term_fails_loud(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy()
        with pytest.raises((ValueError, NotImplementedError)) as excinfo:
            reg.fit(x, y, ["x1", "lap(x1)"])
        assert "lap" in str(excinfo.value)







class TestVarNamesValidation:

    def test_default_var_names_are_x1_xn(self) -> None:
        reg = SINDyRegressor()
        rng = np.random.default_rng(7)
        x = rng.normal(size=(40, 3))
        y = x[:, 0] + x[:, 2]
        reg.fit(x, y, ["x1", "x2", "x3"])
        assert reg.var_names_ == ["x1", "x2", "x3"]

    def test_custom_var_names_used_in_terms_and_expression(self) -> None:
        reg = SINDyRegressor()
        rng = np.random.default_rng(8)
        x = rng.normal(size=(150, 2))
        y = 2.0 * np.sin(x[:, 0]) - 3.0 * (x[:, 0] * x[:, 1])
        reg.fit(x, y, ["a", "sin(a)", "mul(a, b)", "cos(b)"], var_names=["a", "b"])

        assert reg.var_names_ == ["a", "b"]
        assert set(reg.selected_terms_) == {"sin(a)", "mul(a, b)"}
        assert "a" in reg.expression_ and "b" in reg.expression_

    def test_var_names_length_mismatch_raises(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n_features=2)
        with pytest.raises(ValueError, match="length must equal"):
            reg.fit(x, y, ["x1"], var_names=["only_one"])

    def test_var_names_duplicate_raises(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n_features=2)
        with pytest.raises(ValueError, match="unique"):
            reg.fit(x, y, ["a"], var_names=["a", "a"])

    def test_var_names_reserved_token_raises(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n_features=2)
        with pytest.raises(ValueError, match="exp"):
            reg.fit(x, y, ["r"], var_names=["exp", "r"])

    def test_var_names_non_identifier_raises(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n_features=2)
        with pytest.raises(ValueError, match=re.escape("x-1")):
            reg.fit(x, y, ["r"], var_names=["x-1", "r"])

    def test_var_names_reject_bare_string(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n_features=2)
        with pytest.raises(ValueError, match="not a single string"):
            reg.fit(x, y, ["x1"], var_names="xy")

    def test_var_names_reject_python_keyword(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n_features=2)
        with pytest.raises(ValueError, match="keyword"):
            reg.fit(x, y, ["r"], var_names=["class", "r"])







class TestStateAndRobustness:

    def test_predict_before_fit_raises(self) -> None:
        reg = SINDyRegressor()
        with pytest.raises(RuntimeError) as excinfo:
            reg.predict(np.zeros((3, 2)))
        assert "fit" in str(excinfo.value).lower()

    def test_predict_wrong_feature_count_raises(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n_features=2)
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)
        with pytest.raises(ValueError, match="mismatched feature count"):
            reg.predict(np.zeros((5, 3)))

    def test_fit_rejects_non_2d_features(self) -> None:
        reg = SINDyRegressor()
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        with pytest.raises(ValueError, match="2-D"):
            reg.fit(x, y, ["x1"])

    def test_fit_rejects_sample_count_mismatch(self) -> None:
        reg = SINDyRegressor()
        x = np.zeros((10, 2), dtype=float)
        y = np.zeros(9, dtype=float)
        with pytest.raises(ValueError, match="mismatched sample counts"):
            reg.fit(x, y, _LIBRARY_WITH_DECOYS)

    def test_failed_fit_after_success_does_not_corrupt_state(self) -> None:
        reg = SINDyRegressor()
        x, y = _make_xy(n=180, seed=6)
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)


        good_terms = list(reg.selected_terms_)
        good_coeffs = np.asarray(reg.coefficients_, dtype=float).reshape(-1).copy()


        with pytest.raises((ValueError, KeyError)):
            reg.fit(x, y, ["tanh(x1)", "sigmoid(x2)"])

        terms_now = reg.selected_terms_
        coeffs_now = reg.coefficients_

        if terms_now is None:

            assert coeffs_now is None
            assert reg.expression_ is None or reg.expression_ == ""
            with pytest.raises(RuntimeError):
                reg.predict(x)
        else:

            assert list(terms_now) == good_terms
            preserved = np.asarray(coeffs_now, dtype=float).reshape(-1)
            assert len(preserved) == len(terms_now)
            np.testing.assert_allclose(preserved, good_coeffs, rtol=0, atol=0)

            np.testing.assert_allclose(reg.predict(x), y, rtol=1e-4, atol=1e-6)







class TestNmseSemantics:

    def test_nmse_is_raw_target_based(self) -> None:
        reg = SINDyRegressor()
        rng = np.random.default_rng(10)
        x = rng.normal(size=(300, 2))
        clean = 2.0 * np.sin(x[:, 0]) - 3.0 * (x[:, 0] * x[:, 1])
        y = clean + 0.05 * rng.normal(size=300)
        reg.fit(x, y, _LIBRARY_WITH_DECOYS)

        assert 0.0 < reg.nmse_ < 0.2
        assert np.isfinite(reg.nmse_)

    def test_constant_y_does_not_crash(self) -> None:
        reg = SINDyRegressor()
        rng = np.random.default_rng(11)
        x = rng.normal(size=(80, 2))
        y = np.full(80, 4.0)
        try:
            reg.fit(x, y, _LIBRARY_WITH_DECOYS)
        except (ValueError, ZeroDivisionError) as exc:


            assert not isinstance(exc, ZeroDivisionError)
            return
        assert np.isfinite(reg.nmse_)













_FORBIDDEN_IMPORT_NAMES = frozenset(
    {
        "PlatformBuilder",
        "platform",
        "plugin",
        "assembly",
        "Model",
        "api",
        "Runner",
    }
)




_SR_SOURCE_MODULE = sr_module


def _collect_imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.update(alias.name.split("."))
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.update(node.module.split("."))
            for alias in node.names:
                names.add(alias.name)
                if alias.asname:
                    names.add(alias.asname)
    return names


class TestImportIsolation:

    def test_source_imports_no_platform_orchestration(self) -> None:
        source = inspect.getsource(_SR_SOURCE_MODULE)
        tree = ast.parse(source)
        imported = _collect_imported_names(tree)
        leaked = imported & _FORBIDDEN_IMPORT_NAMES
        assert not leaked, (
            f"sr.py must not import the PDE platform orchestration, but found "
            f"{sorted(leaked)} in its imports. The SINDy bypass reuses only the "
            f"numerical kernel (Evaluator / STRidge), not PlatformBuilder / "
            f"plugin / assembly / the Model facade."
        )

    def test_detector_catches_a_planted_forbidden_import(self) -> None:
        planted = (
            "from kd.core.platform.builder import PlatformBuilder\n"
            "from. import plugin\n"
            "from kd.api import Model\n"
        )
        imported = _collect_imported_names(ast.parse(planted))
        assert "PlatformBuilder" in imported
        assert "plugin" in imported
        assert "Model" in imported

        assert imported & _FORBIDDEN_IMPORT_NAMES
