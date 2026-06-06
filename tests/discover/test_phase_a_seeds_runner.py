
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_RUNNER = _PROJECT_ROOT / "scripts" / "discover" / "run_phase_a_seeds.py"



_FISHER_LINEAR_DATA = (
    _PROJECT_ROOT
    / "refs" / "discover" / "dso" / "dso" / "task" / "pde"
    / "data_new" / "fisher_groundtruth.mat"
)
_MIN_DATA_BYTES = 50_000


def _data_ready() -> bool:
    return (
        _FISHER_LINEAR_DATA.exists()
        and _FISHER_LINEAR_DATA.stat().st_size > _MIN_DATA_BYTES
    )






REQUIRED_JSON_KEYS: frozenset[str] = frozenset({
    "schema_version",
    "pde",
    "seed",
    "git_sha",
    "created_at",
    "python_version",
    "torch_version",
    "pythonhashseed",
    "n_iterations",
    "batch_size",
    "max_length",
    "operators",
    "fd_max_order",
    "stability_selection",
    "stability_queue_capacity",
    "ground_truth",
    "best_reward",
    "best_expression",
    "runtime_seconds",
    "cycle_top_candidates",
})




SANITY_ITERATIONS = 20
SANITY_BATCH_SIZE = 500
SANITY_MAX_LENGTH = 30


_MODULE_NAME = "run_phase_a_seeds_test_module"


def _load_runner_module() -> ModuleType:
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _invoke_runner(
    *args: str,
    cwd: Path,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(_RUNNER), *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )







@pytest.mark.unit
def test_runner_script_exists() -> None:
    assert _RUNNER.exists(), f"runner missing at {_RUNNER}"


@pytest.mark.unit
def test_parse_seeds_csv_basic() -> None:
    mod = _load_runner_module()
    assert mod._parse_seeds("42,123,777") == [42, 123, 777]
    assert mod._parse_seeds("42") == [42]

    assert mod._parse_seeds(" 42, 123 ") == [42, 123]
    assert mod._parse_seeds("42,123,") == [42, 123]


@pytest.mark.unit
def test_parse_seeds_rejects_empty() -> None:
    mod = _load_runner_module()
    with pytest.raises(ValueError, match="Empty"):
        mod._parse_seeds("")


@pytest.mark.unit
def test_parse_seeds_rejects_non_int() -> None:
    mod = _load_runner_module()
    with pytest.raises(ValueError, match="abc"):
        mod._parse_seeds("42,abc,123")


@pytest.mark.unit
def test_parse_args_rejects_unknown_pde(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_runner_module()
    with pytest.raises(SystemExit) as exc_info:
        mod.parse_args(
            [
                "--pde", "no_such_pde",
                "--seeds", "42",
                "--output-dir", "/tmp/discover_next_phase_a_test",
            ],
        )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "no_such_pde" in captured.err or "invalid choice" in captured.err


@pytest.mark.unit
def test_parse_args_validates_known_pdes() -> None:
    mod = _load_runner_module()
    from kd.search.discover.runners.pde_registry import PDE_REGISTRY

    for pde_name in PDE_REGISTRY:
        ns = mod.parse_args(
            [
                "--pde", pde_name,
                "--seeds", "42",
                "--output-dir", "/tmp/x",
            ],
        )
        assert ns.pde == pde_name


@pytest.mark.unit
def test_parse_args_n_iterations_default_matches_sanity() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_linear",
            "--seeds", "42",
            "--output-dir", "/tmp/x",
        ],
    )
    assert ns.n_iterations == mod.SANITY_ITERATIONS


@pytest.mark.unit
def test_parse_args_n_iterations_override_accepted() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_nonlinear",
            "--seeds", "42,123,777",
            "--output-dir", "/tmp/x",
            "--n-iterations", "100",
        ],
    )
    assert ns.n_iterations == 100


@pytest.mark.unit
def test_build_sanity_config_threads_n_iterations_override() -> None:
    mod = _load_runner_module()
    config = mod._build_sanity_config(["add", "mul"], n_iterations=100)
    assert config.n_iterations == 100

    assert config.batch_size == mod.SANITY_BATCH_SIZE
    assert config.max_length == mod.SANITY_MAX_LENGTH
    assert config.num_units == mod.SANITY_NUM_UNITS
    assert config.num_layers == mod.SANITY_NUM_LAYERS
    assert config.embedding_dim == mod.SANITY_EMBEDDING_DIM


@pytest.mark.unit
def test_parse_args_max_length_default_matches_sanity() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_linear",
            "--seeds", "42",
            "--output-dir", "/tmp/x",
        ],
    )
    assert ns.max_length == mod.SANITY_MAX_LENGTH


@pytest.mark.unit
def test_parse_args_max_length_override_accepted() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_nonlinear",
            "--seeds", "42,123,777",
            "--output-dir", "/tmp/x",
            "--n-iterations", "100",
            "--max-length", "60",
        ],
    )
    assert ns.max_length == 60

    assert ns.n_iterations == 100


@pytest.mark.unit
def test_build_sanity_config_threads_max_length_override() -> None:
    mod = _load_runner_module()
    config = mod._build_sanity_config(
        ["add", "mul"], n_iterations=100, max_length=60,
    )
    assert config.max_length == 60
    assert config.n_iterations == 100
    assert config.batch_size == mod.SANITY_BATCH_SIZE


@pytest.mark.unit
def test_parse_args_entropy_weight_default_matches_sanity() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_linear",
            "--seeds", "42",
            "--output-dir", "/tmp/x",
        ],
    )
    assert ns.entropy_weight == mod.SANITY_ENTROPY_WEIGHT


@pytest.mark.unit
def test_parse_args_entropy_weight_override_accepted() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_nonlinear",
            "--seeds", "42,123,777",
            "--output-dir", "/tmp/x",
            "--n-iterations", "100",
            "--entropy-weight", "0.03",
        ],
    )
    assert ns.entropy_weight == 0.03

    assert isinstance(ns.entropy_weight, float)


@pytest.mark.unit
def test_build_sanity_config_threads_entropy_weight_override() -> None:
    mod = _load_runner_module()
    config = mod._build_sanity_config(
        ["add", "mul"], n_iterations=100, entropy_weight=0.03,
    )
    assert config.entropy_weight == 0.03
    assert config.n_iterations == 100

    assert config.epsilon == mod.SANITY_EPSILON
    assert config.max_length == mod.SANITY_MAX_LENGTH


@pytest.mark.unit
def test_parse_args_epsilon_default_matches_sanity() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_linear",
            "--seeds", "42",
            "--output-dir", "/tmp/x",
        ],
    )
    assert ns.epsilon == mod.SANITY_EPSILON


@pytest.mark.unit
def test_parse_args_epsilon_override_accepted() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_nonlinear",
            "--seeds", "42,123,777",
            "--output-dir", "/tmp/x",
            "--n-iterations", "100",
            "--epsilon", "0.02",
        ],
    )
    assert ns.epsilon == 0.02

    assert isinstance(ns.epsilon, float)


@pytest.mark.unit
def test_build_sanity_config_threads_epsilon_override() -> None:
    mod = _load_runner_module()
    config = mod._build_sanity_config(
        ["add", "mul"], n_iterations=100, epsilon=0.02,
    )
    assert config.epsilon == 0.02
    assert config.n_iterations == 100

    assert config.entropy_weight == mod.SANITY_ENTROPY_WEIGHT
    assert config.max_length == mod.SANITY_MAX_LENGTH


@pytest.mark.unit
def test_parse_args_sparse_refit_eps_default_zero() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_linear",
            "--seeds", "42",
            "--output-dir", "/tmp/x",
        ],
    )
    assert ns.sparse_refit_eps == 0.0


@pytest.mark.unit
def test_parse_args_sparse_refit_eps_override_accepted() -> None:
    mod = _load_runner_module()
    ns = mod.parse_args(
        [
            "--pde", "fisher_nonlinear",
            "--seeds", "42,123,777",
            "--output-dir", "/tmp/x",
            "--n-iterations", "100",
            "--entropy-weight", "0.03",
            "--sparse-refit-eps", "1e-3",
        ],
    )
    assert ns.sparse_refit_eps == 1e-3
    assert isinstance(ns.sparse_refit_eps, float)


@pytest.mark.unit
def test_parse_args_sparse_refit_eps_rejects_invalid_range() -> None:
    mod = _load_runner_module()
    for bad_eps in ["-0.1", "1.0", "1.5"]:
        with pytest.raises(SystemExit):
            mod.parse_args(
                [
                    "--pde", "fisher_linear",
                    "--seeds", "42",
                    "--output-dir", "/tmp/x",
                    "--sparse-refit-eps", bad_eps,
                ],
            )


@pytest.mark.unit
def test_run_one_seed_sparse_empty_hof_falls_back_to_vanilla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_runner_module()
    state = MagicMock()
    state.best_reward = 0.87
    state.best_expression = "u"
    state.best_result_terms = ["u"]
    state.best_result_coefficients = [1.0]
    engine = MagicMock()
    engine.run.return_value = state
    engine.cycle_top_candidates = []
    monkeypatch.setattr(mod, "build_engine", lambda config: engine)
    config = mod._build_sanity_config(["add", "mul"], n_iterations=1)


    result = mod._run_one_seed(
        MagicMock(), config, seed=42, sparse_refit_eps=1e-3,
    )

    assert result.best_reward == pytest.approx(0.87)
    assert result.best_expression == "u"
    assert result.best_terms == ["u"]
    assert result.best_coefficients == [1.0]
    assert result.best_nmse is None

    assert result.sparse_refit_eps == 1e-3
    assert result.sparse_refit_n_dropped == 0








pytestmark_slow_data = pytest.mark.skipif(
    not _data_ready(),
    reason=f"Fisher_linear data not present at {_FISHER_LINEAR_DATA}.",
)


@pytest.mark.slow
@pytestmark_slow_data
def test_single_pde_single_seed_produces_valid_json(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "phase_a_out"
    result = _invoke_runner(
        "--pde", "fisher_linear",
        "--seeds", "42",
        "--output-dir", str(out_dir),
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"runner exited {result.returncode}; stderr=\n{result.stderr}"
    )
    json_path = out_dir / "fisher_linear_seed42.json"
    assert json_path.exists(), (
        f"runner did not write expected file {json_path}; "
        f"stdout=\n{result.stdout}"
    )
    payload = json.loads(json_path.read_text())
    missing = REQUIRED_JSON_KEYS - set(payload.keys())
    assert not missing, f"JSON schema missing keys: {sorted(missing)}"
    assert payload["pde"] == "fisher_linear"
    assert payload["seed"] == 42
    assert payload["n_iterations"] == SANITY_ITERATIONS
    assert payload["batch_size"] == SANITY_BATCH_SIZE
    assert payload["max_length"] == SANITY_MAX_LENGTH
    assert payload["fd_max_order"] == 2
    assert isinstance(payload["operators"], list) and payload["operators"]
    assert isinstance(payload["ground_truth"], str) and payload["ground_truth"]
    assert isinstance(payload["best_reward"], float)
    assert payload["best_reward"] > 0.5, (
        "best_reward <= 0.5 reproducibly fails sanity threshold."
    )
    assert isinstance(payload["best_expression"], str)
    assert payload["best_expression"]
    assert isinstance(payload["runtime_seconds"], float)
    assert payload["runtime_seconds"] > 0.0
    assert isinstance(payload["cycle_top_candidates"], list)
    for entry in payload["cycle_top_candidates"]:
        assert "reward" in entry and "expression" in entry

    assert isinstance(payload["schema_version"], str)
    assert isinstance(payload["git_sha"], str)

    assert isinstance(payload["created_at"], str) and payload["created_at"]
    assert payload["python_version"]
    assert payload["torch_version"]


@pytest.mark.slow
@pytestmark_slow_data
def test_unknown_pde_raises_clear_error(tmp_path: Path) -> None:
    out_dir = tmp_path / "phase_a_out"
    result = _invoke_runner(
        "--pde", "no_such_pde",
        "--seeds", "42",
        "--output-dir", str(out_dir),
        cwd=_PROJECT_ROOT,
        timeout=30.0,
    )
    assert result.returncode != 0, (
        "runner accepted a non-existent PDE without erroring."
    )
    error_text = (result.stderr or "") + (result.stdout or "")
    assert "no_such_pde" in error_text or "invalid choice" in error_text, (
        f"error message did not surface the offending PDE name; "
        f"stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )


@pytest.mark.slow
@pytestmark_slow_data
def test_output_dir_is_created_when_absent(tmp_path: Path) -> None:
    out_dir = tmp_path / "deep" / "nested" / "phase_a_out"
    assert not out_dir.exists()
    result = _invoke_runner(
        "--pde", "fisher_linear",
        "--seeds", "42",
        "--output-dir", str(out_dir),
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"runner failed; stderr=\n{result.stderr}"
    )
    assert out_dir.exists() and out_dir.is_dir()
    assert (out_dir / "fisher_linear_seed42.json").exists()


@pytest.mark.slow
@pytestmark_slow_data
def test_sparse_refit_default_off_no_provenance_fields(tmp_path: Path) -> None:
    out_dir = tmp_path / "default_off"
    result = _invoke_runner(
        "--pde", "fisher_linear",
        "--seeds", "42",
        "--output-dir", str(out_dir),
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"runner failed; stderr=\n{result.stderr}"
    )
    payload = json.loads(
        (out_dir / "fisher_linear_seed42.json").read_text(),
    )

    assert "sparse_refit_eps" not in payload, (
        f"sparse_refit_eps leaked into default-off payload: {payload}"
    )
    assert "sparse_refit_n_dropped" not in payload
    assert "sparse_refit_alpha" not in payload

    assert payload["best_reward"] > 0.5


@pytest.mark.slow
@pytestmark_slow_data
def test_sparse_refit_enabled_writes_provenance_fields(tmp_path: Path) -> None:
    out_dir = tmp_path / "sparse_on"
    result = _invoke_runner(
        "--pde", "fisher_linear",
        "--seeds", "42",
        "--output-dir", str(out_dir),
        "--sparse-refit-eps", "1e-3",
        cwd=_PROJECT_ROOT,
        timeout=180.0,
    )
    assert result.returncode == 0, (
        f"runner failed; stderr=\n{result.stderr}"
    )
    payload = json.loads(
        (out_dir / "fisher_linear_seed42.json").read_text(),
    )

    assert payload["sparse_refit_eps"] == 1e-3
    assert "sparse_refit_n_dropped" in payload
    assert isinstance(payload["sparse_refit_n_dropped"], int)
    assert payload["sparse_refit_alpha"] == 0.01



    assert payload["best_reward"] > 0.9, (
        f"sparse refit broke fisher_linear hit; reward={payload['best_reward']}"
    )


@pytest.mark.slow
@pytestmark_slow_data
def test_two_seeds_produce_two_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "phase_a_out"
    result = _invoke_runner(
        "--pde", "fisher_linear",
        "--seeds", "42,123",
        "--output-dir", str(out_dir),
        cwd=_PROJECT_ROOT,
        timeout=300.0,
    )
    assert result.returncode == 0, (
        f"runner failed; stderr=\n{result.stderr}"
    )
    p42 = out_dir / "fisher_linear_seed42.json"
    p123 = out_dir / "fisher_linear_seed123.json"
    assert p42.exists() and p123.exists()
    payload42 = json.loads(p42.read_text())
    payload123 = json.loads(p123.read_text())
    assert payload42["seed"] == 42
    assert payload123["seed"] == 123


    assert payload42["git_sha"] == payload123["git_sha"]
    assert payload42["schema_version"] == payload123["schema_version"]
