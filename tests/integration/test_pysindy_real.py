
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from kd.search.protocol import PlatformComponents
from kd.search.pysindy import PySINDyConfig, PySINDyPlugin

pytestmark = pytest.mark.pysindy

_TERMS = ("u", "u_x", "u_xx", "mul(u,u_x)")


@pytest.fixture(scope="module")
def pysindy_module() -> Any:
    return pytest.importorskip("pysindy")


def _config() -> PySINDyConfig:





    return PySINDyConfig(
        terms=_TERMS,
        threshold=0.1,
        max_iter=50,
        normalize_columns=False,
        unbias=True,
        seed=0,
    )


def _fit_plugin(
    components: PlatformComponents,
    config: PySINDyConfig | None = None,
) -> PySINDyPlugin:
    plugin = PySINDyPlugin(config or _config())
    plugin.prepare(components)
    plugin.propose(1)
    return plugin


@pytest.mark.integration
def test_stlsq_recovers_two_mode_heat_structure_and_coefficient(
    pysindy_module: Any,
    real_pysindy_components: PlatformComponents,
) -> None:
    del pysindy_module
    plugin = _fit_plugin(real_pysindy_components)
    result = plugin.build_final_result()

    assert result.is_valid
    assert result.terms is not None
    assert result.selected_indices == [result.terms.index("u_xx")]
    assert result.coefficients is not None
    recovered = float(result.coefficients[result.selected_indices[0]].item())
    assert recovered == pytest.approx(1.0, rel=5e-2, abs=5e-2)


@pytest.mark.integration
def test_adapter_is_transparent_to_direct_stlsq(
    pysindy_module: Any,
    real_pysindy_components: PlatformComponents,
) -> None:
    evaluator = real_pysindy_components.evaluator
    assert evaluator is not None
    config = _config()
    theta, valid_terms = evaluator.build_theta_matrix(list(config.terms))
    X = theta.detach().cpu().numpy().astype(np.float64)
    y = evaluator.lhs_target.detach().cpu().numpy().astype(np.float64).reshape(-1)

    plugin = _fit_plugin(real_pysindy_components, config)
    plugin_result = plugin.build_final_result()
    assert plugin_result.terms == valid_terms
    assert plugin_result.coefficients is not None

    direct = pysindy_module.optimizers.STLSQ(
        threshold=config.threshold,
        max_iter=config.max_iter,
        normalize_columns=config.normalize_columns,
        unbias=config.unbias,
    )
    direct.fit(X, y)
    direct_coefficients = np.asarray(direct.coef_, dtype=np.float64).ravel()
    plugin_coefficients = (
        plugin_result.coefficients.detach().cpu().numpy().astype(np.float64)
    )
    assert np.array_equal(plugin_coefficients, direct_coefficients)


@pytest.mark.integration
def test_verified_stlsq_surface_accepts_all_typed_knobs(
    pysindy_module: Any,
) -> None:
    optimizer = pysindy_module.optimizers.STLSQ(
        threshold=0.05,
        max_iter=10,
        normalize_columns=True,
        unbias=True,
    )
    X = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1]
    optimizer.fit(X, y)

    assert np.asarray(optimizer.coef_).shape in {(2,), (1, 2)}
    assert float(optimizer.intercept_) == pytest.approx(0.0, abs=1e-10)
