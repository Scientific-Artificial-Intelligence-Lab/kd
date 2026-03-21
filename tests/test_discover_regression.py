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
