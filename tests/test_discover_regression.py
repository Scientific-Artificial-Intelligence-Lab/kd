import numpy as np
import pytest

from kd.model.discover.program import Program, from_str_tokens
from kd.model.discover.task import set_task
from kd.model.kd_discover_regression import KD_Discover_Regression


@pytest.fixture(autouse=True)
def reset_program_state():
    Program.clear_cache()
    Program.set_n_objects(1)
    yield
    Program.clear_cache()


def _make_regression_data(X: np.ndarray, y: np.ndarray) -> dict:
    return {
        "X": [X[:, i : i + 1] for i in range(X.shape[1])],
        "y": y.reshape(-1, 1),
        "n_input_var": X.shape[1],
        "var_names": [f"x{i + 1}" for i in range(X.shape[1])],
    }


def _setup_regression_task(X: np.ndarray, y: np.ndarray, function_set: list[str]) -> None:
    config_task = {
        "task_type": "regression",
        "dataset": "synthetic-regression",
        "function_set": function_set,
        "metric": "inv_nrmse",
        "metric_params": (),
        "threshold": 1e-10,
        "protected": True,
        "use_torch": False,
    }
    set_task(config_task, _make_regression_data(X, y))


def test_set_task_builds_regression_library_without_state_tokens():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([8.0, 18.0])

    _setup_regression_task(X, y, ["add", "mul", "const"])

    assert Program.task.task_type == "regression"
    assert Program.task.n_input_var == 2
    assert Program.task.y.shape == (2, 1)
    assert "x1" in Program.library.names
    assert "x2" in Program.library.names
    assert all(not name.startswith("u") for name in Program.library.names)


def test_program_execute_direct_fits_linear_constants():
    x1 = np.linspace(-2.0, 2.0, 41)
    x2 = np.linspace(3.0, 7.0, 41)
    X = np.column_stack([x1, x2])
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1]

    _setup_regression_task(X, y, ["add", "mul", "const"])
    Program.set_const_optimizer("scipy")

    program = from_str_tokens("add,mul,const,x1,mul,const,x2", skip_cache=True)

    reward = program.r_ridge
    y_hat, y_right, weights = program.execute_direct(Program.task.x)

    assert reward > 0.999999
    assert y_right is None
    assert weights == [1.0]
    np.testing.assert_allclose(y_hat.ravel(), y, atol=1e-6)
    np.testing.assert_allclose(program.get_constants(), [2.0, 3.0], atol=1e-4)
    assert program.evaluate["success"] is True
    assert program.evaluate["nmse_test"] < 1e-10


def test_program_execute_direct_fits_rational_constants():
    rf = np.linspace(0.05, 0.95, 25)
    r = np.linspace(0.2, 1.8, 21)
    x1, x2 = np.meshgrid(rf, r, indexing="ij")
    X = np.column_stack([x1.ravel(), x2.ravel()])
    y = X[:, 1] / (0.147 * X[:, 0] + 0.011)

    _setup_regression_task(X, y, ["add", "mul", "div", "const"])
    Program.set_const_optimizer("scipy")

    program = from_str_tokens("div,x2,add,mul,const,x1,const", skip_cache=True)

    reward = program.r_ridge
    y_hat, _, _ = program.execute_direct(Program.task.x)

    assert reward > 0.999
    np.testing.assert_allclose(y_hat.ravel(), y, atol=1e-4)
    np.testing.assert_allclose(program.get_constants(), [0.147, 0.011], atol=5e-3)
    assert program.evaluate["nmse_test"] < 1e-8


def test_kd_discover_regression_fit_returns_structured_result(monkeypatch):
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.5]])
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1]

    def fake_train(self, n_epochs=100, verbose=True):
        program = from_str_tokens("add,mul,const,x1,mul,const,x2", skip_cache=True)
        _ = program.r_ridge
        return {
            "program": program,
            "r": program.r_ridge,
            "expression": repr(program),
            "traversal": repr(program),
        }

    monkeypatch.setattr(KD_Discover_Regression, "train", fake_train, raising=False)

    model = KD_Discover_Regression(
        n_iterations=1,
        n_samples_per_batch=10,
        binary_operators=["add", "mul"],
        unary_operators=[],
    )
    result = model.fit(X, y, var_names=["R_F", "r"])

    assert result["expression"] == "add,mul,2.0,x1,mul,3.0,x2"
    assert result["constants"] == pytest.approx([2.0, 3.0], abs=1e-4)
    assert result["reward"] > 0.999999
    assert result["mse"] < 1e-10
    assert result["var_names"] == ["R_F", "r"]
    assert result["program"] is model.best_program_


def test_kd_discover_regression_fit_validates_input_shapes():
    model = KD_Discover_Regression(n_iterations=1, n_samples_per_batch=10)

    with pytest.raises(ValueError, match="二维数组"):
        model.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="样本数"):
        model.fit(np.ones((3, 2)), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Bug regression tests
# ---------------------------------------------------------------------------

from kd.model.discover.task.regression import make_regression_metric


class TestPredictNoneCrash:
    """Bug 1: predict() crashes with AttributeError when execute_direct
    returns (None, None, [1.0]) for an invalid program.

    Expected: RuntimeError with a clear message, not AttributeError.
    """

    def test_predict_raises_runtime_error_on_none_y_hat(self, monkeypatch):
        """predict() should raise RuntimeError when program produces None."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([5.0, 7.0])
        _setup_regression_task(X, y, ["add", "mul", "const"])

        program = from_str_tokens("add,x1,x2", skip_cache=True)
        # Force execute_direct to return (None, None, [1.0])
        monkeypatch.setattr(
            program, "execute_direct", lambda x: (None, None, [1.0])
        )

        model = KD_Discover_Regression(n_iterations=1, n_samples_per_batch=10)
        model.best_program_ = program

        with pytest.raises(RuntimeError, match="(?i)predict|invalid|none"):
            model.predict(X)


class TestVariableNameReplacementOrdering:
    """Bug 2: _render_named_expression replaces x1 before x10, corrupting
    x10 into '<name_for_x1>0'.
    """

    def test_ten_variables_no_corruption(self):
        """x10 should become the 10th name, not '<x1-name>0'."""
        expr = "add,x1,x10"
        names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        result = KD_Discover_Regression._render_named_expression(expr, names)
        assert result == "add,a,j"

    def test_twelve_variables_x11_x12(self):
        """x11, x12 should not be corrupted by x1 replacement."""
        expr = "add,x1,add,x11,x12"
        names = [f"v{i}" for i in range(12)]
        result = KD_Discover_Regression._render_named_expression(expr, names)
        assert result == "add,v0,add,v10,v11"

    def test_two_variables_no_false_positive(self):
        """Sanity check: 2 variables should work fine (no ordering issue)."""
        expr = "mul,x1,x2"
        names = ["alpha", "beta"]
        result = KD_Discover_Regression._render_named_expression(expr, names)
        assert result == "mul,alpha,beta"


class TestPredictLifecycle:
    """Bug 3: predict() happy path and error path coverage."""

    def test_predict_before_fit_raises_runtime_error(self):
        """predict() without prior fit() should raise RuntimeError."""
        model = KD_Discover_Regression(n_iterations=1, n_samples_per_batch=10)
        X = np.ones((5, 2))
        with pytest.raises(RuntimeError, match="before fit"):
            model.predict(X)

    def test_predict_happy_path_returns_correct_shape(self):
        """predict() with a valid program returns a 1-D array of correct length."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = X[:, 0] + X[:, 1]  # y = x1 + x2
        _setup_regression_task(X, y, ["add", "mul", "const"])

        program = from_str_tokens("add,x1,x2", skip_cache=True)
        _ = program.r_ridge  # trigger constant optimization / evaluation

        model = KD_Discover_Regression(n_iterations=1, n_samples_per_batch=10)
        model.best_program_ = program

        y_pred = model.predict(X)
        assert y_pred.ndim == 1
        assert y_pred.shape[0] == X.shape[0]
        np.testing.assert_allclose(y_pred, y, atol=1e-6)

    def test_predict_rejects_1d_input(self):
        """predict() should reject 1-D X with a clear ValueError."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        _setup_regression_task(X, y, ["add"])

        program = from_str_tokens("add,x1,x2", skip_cache=True)
        model = KD_Discover_Regression(n_iterations=1, n_samples_per_batch=10)
        model.best_program_ = program

        with pytest.raises(ValueError, match="二维数组"):
            model.predict(np.array([1.0, 2.0]))

    def test_predict_rejects_feature_count_mismatch(self):
        """predict() should reject X whose feature count differs from training."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        _setup_regression_task(X, y, ["add"])

        program = from_str_tokens("add,x1,x2", skip_cache=True)
        model = KD_Discover_Regression(n_iterations=1, n_samples_per_batch=10)
        model.best_program_ = program

        with pytest.raises(ValueError, match="特征数"):
            model.predict(np.array([[1.0], [2.0]]))

        with pytest.raises(ValueError, match="特征数"):
            model.predict(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))


class TestMakeRegressionMetricRedundancy:
    """Bug 4: sqrt(rmse**2) is numerically redundant -- it should equal rmse.

    These tests verify the metric gives the correct mathematical result.
    They serve as refactor-safety tests: they should pass both before and
    after simplifying sqrt(rmse**2) to rmse.
    """

    def test_perfect_prediction_returns_one(self):
        """Metric should be 1.0 when y_hat == y exactly."""
        metric_fn, _, _ = make_regression_metric("inv_nrmse")
        y = np.array([[1.0], [2.0], [3.0]])
        y_hat = y.copy()
        result = metric_fn(y, y_hat)
        assert result == pytest.approx(1.0)

    def test_known_value_by_hand(self):
        """Verify metric against hand computation.

        y = [0, 2], y_hat = [0, 0]
        mse = (0 + 4) / 2 = 2, rmse = sqrt(2)
        var(y) = 1.0
        Expected: 1 / (1 + sqrt(rmse^2 / var)) = 1 / (1 + sqrt(2/1))
                = 1 / (1 + sqrt(2))
        """
        metric_fn, _, _ = make_regression_metric("inv_nrmse")
        y = np.array([[0.0], [2.0]])
        y_hat = np.array([[0.0], [0.0]])
        expected = 1.0 / (1.0 + np.sqrt(2.0))
        result = metric_fn(y, y_hat)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_metric_bounded_between_zero_and_one(self):
        """inv_nrmse should always be in (0, 1]."""
        metric_fn, _, _ = make_regression_metric("inv_nrmse")
        rng = np.random.default_rng(42)
        y = rng.standard_normal((50, 1))
        y_hat = rng.standard_normal((50, 1)) * 10  # large error
        result = metric_fn(y, y_hat)
        assert 0.0 < result <= 1.0

    def test_unsupported_metric_name_raises(self):
        """Requesting an unknown metric should raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized"):
            make_regression_metric("mse")
