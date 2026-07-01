
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import sympy

from kd.core.expr.sympy_bridge import are_equivalent, to_sympy



from kd.search.pysr import sr as sr_module
from kd.search.pysr.sr import PySRSymbolicRegressor

from .conftest import FakePySRBackend, make_backend_factory

pytestmark = pytest.mark.unit










def _linear_two_term_recipe(names: list[str]) -> sympy.Expr:
    if len(names) < 2:
        return sympy.Float(3.0) * sympy.Symbol(names[0])
    return sympy.Float(3.0) * sympy.Symbol(names[0]) - sympy.Float(2.0) * sympy.Symbol(
        names[1]
    )


def _const_plus_product_recipe(names: list[str]) -> sympy.Expr:
    if len(names) < 2:
        return sympy.Float(2.5) + sympy.Symbol(names[0])
    return sympy.Float(2.5) + sympy.Symbol(names[0]) * sympy.Symbol(names[1])


def _sqrt_best_recipe(names: list[str]) -> sympy.Expr:
    return sympy.sqrt(sympy.Symbol(names[0]))


def _make_regressor(recipe) -> tuple[PySRSymbolicRegressor, FakePySRBackend]:
    backend = FakePySRBackend(best_recipe=recipe)
    factory = make_backend_factory(backend)
    return PySRSymbolicRegressor(backend_factory=factory), backend







class TestFitPredictRoundTrip:

    def test_fit_returns_self(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 2))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
        assert reg.fit(X, y) is reg

    def test_predict_recovers_planted_target(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 2))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
        reg.fit(X, y)

        y_hat = reg.predict(X)
        assert isinstance(y_hat, np.ndarray)
        assert y_hat.shape == y.shape
        np.testing.assert_allclose(y_hat, y, rtol=1e-6, atol=1e-8)

    def test_best_score_near_zero_for_exact_fit(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 2))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
        reg.fit(X, y)

        assert isinstance(reg.best_score_, float)
        assert np.isfinite(reg.best_score_)
        assert reg.best_score_ == pytest.approx(0.0, abs=1e-9)

    def test_nmse_inf_when_target_has_nan(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        rng = np.random.default_rng(13)
        X = rng.normal(size=(20, 2))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
        y[3] = np.nan
        reg.fit(X, y)

        assert reg.best_score_ == float("inf")
        assert not np.isnan(reg.best_score_)

    def test_best_expr_is_kd_ir_string(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        rng = np.random.default_rng(3)
        X = rng.normal(size=(30, 2))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
        reg.fit(X, y)

        assert isinstance(reg.best_expr_, str)
        assert reg.best_expr_



        names = reg.var_names_
        assert names is not None
        planted = sympy.Float(3.0) * sympy.Symbol(names[0]) - sympy.Float(
            2.0
        ) * sympy.Symbol(names[1])
        assert to_sympy(reg.best_expr_).equals(planted)

    def test_predict_before_fit_raises(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        with pytest.raises(RuntimeError) as excinfo:
            reg.predict(np.zeros((3, 2)))
        assert not isinstance(excinfo.value, NotImplementedError)
        assert "fit" in str(excinfo.value).lower()







class TestVarNamesRendering:

    def test_default_var_names_are_x1_x2(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        rng = np.random.default_rng(4)
        X = rng.normal(size=(20, 2))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
        reg.fit(X, y)
        assert reg.var_names_ == ["x1", "x2"]

    def test_user_var_names_appear_in_expression(self) -> None:
        reg, _ = _make_regressor(_const_plus_product_recipe)
        rng = np.random.default_rng(5)
        X = rng.normal(size=(25, 2))
        y = 2.5 + X[:, 0] * X[:, 1]
        reg.fit(X, y, var_names=["R_F", "r"])

        assert reg.var_names_ == ["R_F", "r"]
        assert reg.best_expr_ is not None
        assert "R_F" in reg.best_expr_
        assert "r" in reg.best_expr_
        symbol_names = {s.name for s in reg.best_sympy_.free_symbols}
        assert symbol_names == {"R_F", "r"}

    def test_generic_placeholders_do_not_leak(self) -> None:
        reg, backend = _make_regressor(_const_plus_product_recipe)
        rng = np.random.default_rng(6)
        X = rng.normal(size=(25, 2))
        y = 2.5 + X[:, 0] * X[:, 1]
        reg.fit(X, y, var_names=["R_F", "r"])




        assert backend.captured_names == ["x1", "x2"]
        rendered_symbols = {s.name for s in reg.best_sympy_.free_symbols}
        assert "x1" not in rendered_symbols
        assert "x2" not in rendered_symbols

    def test_user_named_expression_is_semantically_equivalent(self) -> None:
        reg, _ = _make_regressor(_const_plus_product_recipe)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(25, 2))
        y = 2.5 + X[:, 0] * X[:, 1]
        reg.fit(X, y, var_names=["R_F", "r"])

        target = sympy.Float(2.5) + sympy.Symbol("R_F") * sympy.Symbol("r")
        assert reg.best_expr_ is not None
        assert to_sympy(reg.best_expr_).equals(target)

    def test_constant_render_with_user_names(self) -> None:
        reg, _ = _make_regressor(_const_plus_product_recipe)
        rng = np.random.default_rng(14)
        X = rng.normal(size=(25, 2))
        y = 2.5 + X[:, 0] * X[:, 1]
        reg.fit(X, y, var_names=["a", "b"])

        assert reg.best_expr_ is not None
        assert are_equivalent(reg.best_expr_, "add(2.5, mul(a, b))")







class TestConstantPreservation:

    def test_constant_survives_into_kd_ir(self) -> None:
        reg, _ = _make_regressor(_const_plus_product_recipe)
        rng = np.random.default_rng(8)
        X = rng.normal(size=(40, 2))
        y = 2.5 + X[:, 0] * X[:, 1]
        reg.fit(X, y)

        names = reg.var_names_
        assert names is not None
        target = sympy.Float(2.5) + sympy.Symbol(names[0]) * sympy.Symbol(names[1])


        assert reg.best_expr_ is not None
        assert are_equivalent(reg.best_expr_, "add(2.5, mul(x1, x2))")
        assert to_sympy(reg.best_expr_).equals(target)

    def test_constant_survives_into_sympy_attr(self) -> None:
        reg, _ = _make_regressor(_const_plus_product_recipe)
        rng = np.random.default_rng(9)
        X = rng.normal(size=(40, 2))
        y = 2.5 + X[:, 0] * X[:, 1]
        reg.fit(X, y)

        names = reg.var_names_
        assert names is not None
        product = sympy.Symbol(names[0]) * sympy.Symbol(names[1])
        residual = sympy.simplify(reg.best_sympy_ - product)
        assert residual == sympy.Float(2.5)

    def test_constant_witnessed_numerically_by_predict(self) -> None:
        reg, _ = _make_regressor(_const_plus_product_recipe)
        rng = np.random.default_rng(10)
        X = rng.normal(size=(50, 2))
        y = 2.5 + X[:, 0] * X[:, 1]
        reg.fit(X, y)

        y_hat = reg.predict(X)
        np.testing.assert_allclose(y_hat, y, rtol=1e-6, atol=1e-8)
        assert reg.best_score_ == pytest.approx(0.0, abs=1e-9)







class TestInputValidation:

    def test_rejects_non_2d_features(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        X = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        with pytest.raises(ValueError):
            reg.fit(X, y)

    def test_rejects_sample_count_mismatch(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        X = np.zeros((10, 2), dtype=float)
        y = np.zeros(9, dtype=float)
        with pytest.raises(ValueError):
            reg.fit(X, y)

    def test_rejects_var_names_length_mismatch(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        X = np.zeros((10, 2), dtype=float)
        y = np.zeros(10, dtype=float)
        with pytest.raises(ValueError):
            reg.fit(X, y, var_names=["only_one"])

    def test_fit_rejects_duplicate_var_names(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        X = np.zeros((10, 2), dtype=float)
        y = np.zeros(10, dtype=float)
        with pytest.raises(ValueError, match="a"):
            reg.fit(X, y, var_names=["a", "a"])

    def test_fit_rejects_operator_named_features(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        X = np.zeros((10, 2), dtype=float)
        y = np.zeros(10, dtype=float)
        with pytest.raises(ValueError, match="exp"):
            reg.fit(X, y, var_names=["exp", "r"])

    def test_fit_rejects_non_identifier_var_names(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        X = np.zeros((10, 2), dtype=float)
        y = np.zeros(10, dtype=float)
        with pytest.raises(ValueError, match="x-1"):
            reg.fit(X, y, var_names=["x-1", "r"])

    def test_accepts_2d_column_y(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        rng = np.random.default_rng(11)
        X = rng.normal(size=(20, 2))
        y = (3.0 * X[:, 0] - 2.0 * X[:, 1]).reshape(-1, 1)

        reg.fit(X, y)
        assert reg.n_features_in_ == 2

    def test_rejects_multi_output_y(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        X = np.zeros((10, 2), dtype=float)
        y = np.zeros((10, 2), dtype=float)
        with pytest.raises(ValueError):
            reg.fit(X, y)

    def test_predict_rejects_wrong_feature_count(self) -> None:
        reg, _ = _make_regressor(_linear_two_term_recipe)
        rng = np.random.default_rng(12)
        X = rng.normal(size=(20, 2))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
        reg.fit(X, y)
        with pytest.raises(ValueError):
            reg.predict(np.zeros((5, 3)))

    def test_fit_hard_fails_on_unconvertible_best(self) -> None:
        reg, _ = _make_regressor(_sqrt_best_recipe)
        rng = np.random.default_rng(15)
        X = rng.normal(size=(20, 2))
        y = np.sqrt(np.abs(X[:, 0]))

        with pytest.raises(RuntimeError) as excinfo:
            reg.fit(X, y)

        message = str(excinfo.value)
        assert "sqrt" in message
        assert "Restrict PySRConfig operators" in message
        assert reg.best_sympy_ is None
        assert reg.best_expr_ is None
        assert reg.best_score_ == float("inf")
        assert reg.var_names_ is None
        assert reg.n_features_in_ is None








_FORBIDDEN_IMPORT_NAMES = frozenset(
    {
        "PDEDataset",
        "PlatformBuilder",
        "plugin",
        "assembly",
        "convert",
    }
)


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

    def test_source_imports_no_pde_platform(self) -> None:
        source = inspect.getsource(sr_module)
        tree = ast.parse(source)
        imported = _collect_imported_names(tree)
        leaked = imported & _FORBIDDEN_IMPORT_NAMES
        assert not leaked, (
            f"sr.py must not import the PDE platform, but found {sorted(leaked)} "
            f"in its imports. The SR bypass is physically isolated from "
            f"PDEDataset / PlatformBuilder / plugin / assembly / convert."
        )

    def test_detector_catches_a_planted_forbidden_import(self) -> None:
        planted = "from kd.search.pysr import assembly\nfrom. import plugin\n"
        imported = _collect_imported_names(ast.parse(planted))
        assert "assembly" in imported
        assert "plugin" in imported
