
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import (
    FunctionRegistry,
    PythonExecutor,
)
from kd.core.linear_solve.least_squares import (
    LeastSquaresSolver,
)
from kd.data.derivatives.finite_diff import (
    FiniteDiffProvider,
)
from kd.search.discover.builder import build_engine
from kd.search.discover.config import DiscoverConfig
from kd.search.discover.data.loader import load_burgers_mat
from kd.search.discover.engine import DiscoverEngine
from kd.search.discover.paths import REFERENCE_DATA_DIR
from kd.search.discover.plugin import DiscoverConfig as PluginConfig
from kd.search.discover.plugin import DISCOVERPlugin
from kd.search.discover.tokens.library import LibraryConfig
from kd.search.protocol import PlatformComponents
from tests.discover._noise_helpers import add_noise_tensor





BURGERS_DATA = REFERENCE_DATA_DIR / "burgers.mat"

SEED = 42
SMOKE_ITERATIONS = 5
CONVERGENCE_ITERATIONS = 500


EXPECTED_TERMS = ["mul", "diff_x", "diff2_x"]


BURGERS_OPERATORS = ["add", "mul", "sub", "div", "sin", "cos", "diff_x", "diff2_x"]







BURGERS_MODE2_OPERATORS = ["add", "mul", "sub", "div", "diff_x", "diff2_x", "n2", "n3"]






@pytest.fixture(scope="module")
def burgers_data_path() -> Path:
    if not BURGERS_DATA.exists():
        pytest.skip(f"Burgers data not found at {BURGERS_DATA}")
    return BURGERS_DATA


@pytest.fixture(scope="module")
def burgers_components(burgers_data_path: Path) -> PlatformComponents:
    dataset = load_burgers_mat(burgers_data_path)
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(
        dataset=dataset,
        derivative_provider=provider,
    )
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()
    u_t = provider.get_derivative("u", "t", order=1).flatten()
    evaluator = Evaluator(
        executor=executor,
        solver=solver,
        context=context,
        lhs=u_t,
    )
    return PlatformComponents(
        dataset=dataset,
        executor=executor,
        evaluator=evaluator,
        context=context,
        registry=registry,
    )


@pytest.fixture
def smoke_config() -> DiscoverConfig:
    return DiscoverConfig(
        n_iterations=SMOKE_ITERATIONS,
        batch_size=32,
        max_length=15,
        library=LibraryConfig(
            operators=BURGERS_OPERATORS,
            state_vars=["u"],
            coord_vars=["x", "t"],
        ),
        num_units=16,
        num_layers=1,
        embedding_dim=4,
        epsilon=0.05,
        entropy_weight=0.005,
    )


@pytest.fixture
def convergence_config() -> DiscoverConfig:
    return DiscoverConfig(
        n_iterations=CONVERGENCE_ITERATIONS,
        batch_size=128,
        max_length=30,
        library=LibraryConfig(
            operators=BURGERS_OPERATORS,
            state_vars=["u"],
            coord_vars=["x", "t"],
        ),
        num_units=32,
        num_layers=1,
        embedding_dim=8,
        epsilon=0.05,
        entropy_weight=0.005,
    )






class TestSmoke:

    @pytest.mark.smoke
    def test_engine_runs_5_iterations(
        self, burgers_components: PlatformComponents,
        smoke_config: DiscoverConfig,
    ) -> None:
        torch.manual_seed(SEED)
        engine = build_engine(smoke_config)
        state = engine.run(burgers_components.evaluator, n_iterations=SMOKE_ITERATIONS)

        assert state.best_expression != "", (
            "No valid expression found after 5 iterations"
        )

    @pytest.mark.smoke
    def test_engine_produces_nontrivial_expression(
        self, burgers_components: PlatformComponents,
        smoke_config: DiscoverConfig,
    ) -> None:
        torch.manual_seed(SEED)
        engine = build_engine(smoke_config)
        engine.run(burgers_components.evaluator, n_iterations=SMOKE_ITERATIONS)
        expr = engine.best_expression
        assert isinstance(expr, str)
        assert len(expr) > 0, "best_expression should not be empty"

        assert "(" in expr, (
            f"best_expression '{expr}' looks trivial (no operator)"
        )

    @pytest.mark.smoke
    def test_metrics_returned_each_iteration(
        self, burgers_components: PlatformComponents,
        smoke_config: DiscoverConfig,
    ) -> None:
        torch.manual_seed(SEED)
        engine = build_engine(smoke_config)
        metrics = engine.run_iteration(burgers_components.evaluator)
        assert "reward_max" in metrics
        assert "best_reward" in metrics
        assert "n_valid" in metrics
        assert "n_unique" in metrics

    @pytest.mark.smoke
    def test_plugin_protocol(
        self, burgers_components: PlatformComponents,
    ) -> None:
        torch.manual_seed(SEED)
        plugin_config = PluginConfig(
            batch_size=16,
            max_length=15,
            num_units=16,
            embedding_dim=4,
        )
        plugin = DISCOVERPlugin(config=plugin_config)
        plugin.prepare(burgers_components)
        candidates = plugin.propose(16)
        assert isinstance(candidates, list)
        assert len(candidates) > 0, "propose() returned empty candidates"
        assert all(isinstance(c, str) for c in candidates)
        results = plugin.evaluate(candidates)
        assert len(results) == len(candidates)
        plugin.update(results)

        assert plugin.best_score > 0.0, (
            "best_score is 0 after one cycle — evaluate may be returning all invalid"
        )






class TestConfig:

    @pytest.mark.unit
    def test_config_fields(self) -> None:
        config = DiscoverConfig()
        assert config.n_iterations > 0
        assert config.batch_size > 0
        assert config.max_length > 0
        assert 0 < config.epsilon < 1
        assert config.entropy_weight >= 0
        assert config.num_units > 0
        assert config.num_layers > 0
        assert config.embedding_dim > 0
        assert isinstance(config.library, LibraryConfig)
        assert hasattr(config, "baseline")
        assert hasattr(config, "gamma")
        assert hasattr(config, "reward_alpha")

    @pytest.mark.unit
    def test_config_builds_engine(
        self, burgers_components: PlatformComponents,
        smoke_config: DiscoverConfig,
    ) -> None:
        engine = build_engine(smoke_config)
        assert isinstance(engine, DiscoverEngine)






class TestRewardProgression:

    @pytest.mark.slow
    def test_reward_improves_over_100_iterations(
        self, burgers_components: PlatformComponents,
    ) -> None:
        torch.manual_seed(SEED)
        config = DiscoverConfig(
            n_iterations=100,
            batch_size=64,
            max_length=20,
            library=LibraryConfig(
                operators=BURGERS_OPERATORS,
                state_vars=["u"],
                coord_vars=["x", "t"],
            ),
            num_units=32,
            num_layers=1,
            embedding_dim=8,
            epsilon=0.05,
            entropy_weight=0.005,
        )
        engine = build_engine(config)

        rewards = []
        for _ in range(100):
            engine.run_iteration(burgers_components.evaluator)
            rewards.append(engine.best_reward)

        assert rewards[-1] > 0.0, (
            "No positive reward after 100 iterations"
        )


        assert rewards[-1] > rewards[9], (
            f"No improvement: reward@10={rewards[9]:.4f}, "
            f"reward@100={rewards[-1]:.4f}"
        )






class TestConvergence:

    @pytest.mark.slow
    def test_finds_diff_terms(
        self, burgers_components: PlatformComponents,
        convergence_config: DiscoverConfig,
    ) -> None:
        torch.manual_seed(SEED)
        engine = build_engine(convergence_config)
        engine.run(burgers_components.evaluator, n_iterations=CONVERGENCE_ITERATIONS)
        expr = engine.best_expression
        assert any(term in expr for term in EXPECTED_TERMS), (
            f"Expected at least one of {EXPECTED_TERMS} in '{expr}'"
        )

    @pytest.mark.slow
    def test_best_reward_above_threshold(
        self, burgers_components: PlatformComponents,
        convergence_config: DiscoverConfig,
    ) -> None:
        torch.manual_seed(SEED)
        engine = build_engine(convergence_config)
        engine.run(burgers_components.evaluator, n_iterations=CONVERGENCE_ITERATIONS)


        assert engine.best_reward > 0.3, (
            f"best_reward={engine.best_reward:.4f} too low after "
            f"{CONVERGENCE_ITERATIONS} iterations"
        )









MODE2_NOISE_LEVEL = 0.5
MODE2_SEED = 42
MODE2_REWARD_THRESHOLD = 0.3


MODE2_PRETRAIN = 1_000
MODE2_PINN_EPOCH = 50
MODE2_N_ITER = 200
MODE2_N_CYCLES = 2


def _add_noise(u: torch.Tensor, level: float, seed: int) -> torch.Tensor:
    return add_noise_tensor(u, level, seed)


@pytest.fixture(scope="module")
def mode2_result(burgers_data_path: Path):
    from kd.data.schema import (
        FieldData,
        PDEDataset,
        TaskType,
    )
    from kd.search.discover.config import PINNConfig
    from kd.search.discover.pinn.collocation import generate_collocation_points
    from kd.search.discover.pinn.cycle import PINNCycleRunner
    from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset
    from kd.search.discover.pinn.model import PINNModel


    dataset = load_burgers_mat(burgers_data_path)
    assert dataset.fields is not None
    u_clean = dataset.fields["u"].values
    u_noisy = _add_noise(u_clean, MODE2_NOISE_LEVEL, MODE2_SEED)
    noisy_dataset = PDEDataset(
        name="burgers_noisy",
        task_type=TaskType.PDE,
        axes=dataset.axes,
        axis_order=dataset.axis_order,
        fields={"u": FieldData(name="u", values=u_noisy)},
        lhs_field="u",
        lhs_axis="t",
    )


    provider = FiniteDiffProvider(noisy_dataset, max_order=2)
    context = ExecutionContext(
        dataset=noisy_dataset, derivative_provider=provider,
    )
    registry = FunctionRegistry.create_default()
    u_t = provider.get_derivative("u", "t", order=1).flatten()
    evaluator = Evaluator(
        PythonExecutor(registry), LeastSquaresSolver(), context, lhs=u_t,
    )


    pinn_config = PINNConfig(
        number_layer=4,
        n_hidden=20,
        pretrain_epoch=MODE2_PRETRAIN,
        pinn_epoch=MODE2_PINN_EPOCH,
        lr=0.001,
        n_cycles=MODE2_N_CYCLES,
        n_collocation=10_000,
        early_stop_patience=200,
    )
    config = DiscoverConfig(
        n_iterations=MODE2_N_ITER,
        batch_size=64,
        max_length=20,
        library=LibraryConfig(
            operators=BURGERS_MODE2_OPERATORS,
            state_vars=["u"],
            coord_vars=["x", "t"],
        ),
        num_units=32,
        num_layers=1,
        embedding_dim=8,
        pinn=pinn_config,
    )


    torch.manual_seed(MODE2_SEED)
    engine = build_engine(config)
    model = PINNModel(["x", "t"], ["u"], pinn_config)
    pinn_executor = PINNExecutor(registry)
    dataset_meta = make_pinn_dataset(
        ["x", "t"],
        ["u"],
        lhs_field="u",
        lhs_axis="t",
    )


    assert noisy_dataset.axes is not None
    x_vals = noisy_dataset.axes["x"].values
    t_vals = noisy_dataset.axes["t"].values
    big_x, big_t = torch.meshgrid(x_vals, t_vals, indexing="ij")
    n_total = big_x.numel()
    n_obs = max(10, int(n_total * 0.04))
    rng = torch.Generator().manual_seed(MODE2_SEED)
    idx = torch.randperm(n_total, generator=rng)[:n_obs]
    obs_coords = {"x": big_x.flatten()[idx].float(), "t": big_t.flatten()[idx].float()}
    obs_targets = {"u": u_noisy.flatten()[idx].float()}


    x_range = (float(x_vals.min()), float(x_vals.max()))
    t_range = (float(t_vals.min()), float(t_vals.max()))
    colloc = generate_collocation_points(
        bounds={"x": x_range, "t": t_range},
        n_points=pinn_config.n_collocation,
        seed=MODE2_SEED,
    )

    runner = PINNCycleRunner(
        engine=engine,
        pinn_model=model,
        pinn_executor=pinn_executor,
        initial_evaluator=evaluator,
        observation_coords=obs_coords,
        observation_targets=obs_targets,
        colloc_coords=colloc,
        dataset_metadata=dataset_meta,
        config=config,




        stability_seed=MODE2_SEED,
    )
    return runner.run()


@pytest.mark.unit
class TestMode2FixtureSeedPlumbing:

    def test_mode2_result_fixture_threads_stability_seed(self) -> None:
        import inspect

        src = inspect.getsource(mode2_result)
        assert "stability_seed=MODE2_SEED" in src, (
            " D: mode2_result dropped stability_seed=MODE2_SEED; "
            "final-cycle stability selection is unseeded on replay."
        )


class TestBurgersMode2:

    @pytest.mark.slow
    def test_mode2_reward_above_threshold(self, mode2_result) -> None:
        assert mode2_result.final_state.best_reward > MODE2_REWARD_THRESHOLD, (
            f"MODE2 reward {mode2_result.final_state.best_reward:.4f} "
            f"below threshold {MODE2_REWARD_THRESHOLD}"
        )

    @pytest.mark.slow
    def test_mode2_discovers_derivative_terms(self, mode2_result) -> None:
        expr = mode2_result.final_state.best_expression
        assert expr != "", "MODE2 produced no valid expression"
        assert any(t in expr for t in ["diff_x", "diff2_x"]), (
            f"No derivative term in MODE2 result: '{expr}'"
        )
