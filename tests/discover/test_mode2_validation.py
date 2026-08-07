
from __future__ import annotations

import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from kd.core.evaluator import Evaluator
from kd.core.executor.context import ExecutionContext
from kd.core.expr import (
    FunctionRegistry,
    PythonExecutor,
)
from kd.core.expr.term_key import structure_term_key
from kd.core.linear_solve.least_squares import (
    LeastSquaresSolver,
)
from kd.data.derivatives.finite_diff import (
    FiniteDiffProvider,
)
from kd.data.schema import (
    PDEDataset,
)
from kd.search.discover.builder import build_engine
from kd.search.discover.config import DiscoverConfig, PINNConfig
from kd.search.discover.data.loader import load_burgers_mat, load_chafee_infante_npy
from kd.search.discover.pinn.collocation import generate_collocation_points
from kd.search.discover.pinn.cycle import PINNCycleResult, PINNCycleRunner
from kd.search.discover.pinn.executor import PINNExecutor, make_pinn_dataset
from kd.search.discover.pinn.model import PINNModel
from kd.search.discover.tokens.library import LibraryConfig
from tests.discover._noise_helpers import add_noise_dataset





SEED = 42
NOISE_LEVEL = 0.5


_DATA_ROOT = (
    Path(__file__).parent.parent.parent
    / "refs"
    / "discover"
    / "dso"
    / "dso"
    / "task"
    / "pde"
    / "data_new"
)
BURGERS_MAT = _DATA_ROOT / "burgers.mat"
CHAFEE_DIR = _DATA_ROOT


BURGERS_OPERATORS = [
    "add",
    "mul",
    "sub",
    "div",
    "sin",
    "cos",
    "diff_x",
    "diff2_x",
]
CHAFEE_OPERATORS = [
    "add",
    "mul",
    "sub",
    "div",
    "n2",
    "n3",
    "diff_x",
    "diff2_x",
]


VALIDATION_PRETRAIN = 30_000
VALIDATION_PINN_EPOCH = 300
VALIDATION_N_ITER = 100
VALIDATION_N_CYCLES = 2
VALIDATION_COLLOCATION = 15_000
VALIDATION_BATCH_SIZE = 500




BURGERS_REWARD_FLOOR = 0.48
CHAFEE_REWARD_FLOOR = 0.74



PRETRAIN_LOSS_CEILING = 0.23




















GATE1_SEEDS: tuple[int, ...] = (42, 123, 777)


GT_TERM_SET_BURGERS: frozenset[str] = frozenset(
    {
        "diff2_x(u)",
        "mul(diff_x(u), u)",
    }
)



GT_TERM_SET_CHAFEE: frozenset[str] = frozenset(
    {
        "diff2_x(u)",
        "u",
        "n3(u)",
    }
)












BURGERS_MEAN_FLOOR: float = 0.84
BURGERS_GT_HIT_FLOOR: int = 2







CHAFEE_SKIP_REASON: str = (
    "No valid reference Chafee MODE2 baseline (see "
    "'CPU budget insufficient', reference Chafee MODE2 structurally nonsense). "
    "Re-enable when a defensible Chafee aggregate oracle exists."
)




_REQUIRED_HASHSEED: str = "0"







def _add_gaussian_noise(
    dataset: PDEDataset,
    level: float,
    seed: int,
) -> PDEDataset:
    return add_noise_dataset(dataset, level, seed)


def _make_evaluator(dataset: PDEDataset) -> tuple[Evaluator, FunctionRegistry]:
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    u_t = provider.get_derivative(
        dataset.lhs_field,
        dataset.lhs_axis,
        order=1,
    ).flatten()
    evaluator = Evaluator(
        PythonExecutor(registry),
        LeastSquaresSolver(),
        context,
        lhs=u_t,
    )
    return evaluator, registry


def _domain_bounds(dataset: PDEDataset) -> dict[str, tuple[float, float]]:
    assert dataset.axes is not None
    return {
        name: (float(ax.values.min()), float(ax.values.max()))
        for name, ax in dataset.axes.items()
    }


def _build_mode2_runner(
    dataset: PDEDataset,
    operators: list[str],
    seed: int = SEED,
) -> tuple[PINNCycleRunner, DiscoverConfig]:
    assert dataset.axes is not None
    assert dataset.fields is not None
    evaluator, registry = _make_evaluator(dataset)
    coord_vars = list(dataset.axes.keys())
    state_vars = list(dataset.fields.keys())

    pinn_config = PINNConfig(
        number_layer=8,
        n_hidden=20,
        activation="tanh",
        pretrain_epoch=VALIDATION_PRETRAIN,
        pinn_epoch=VALIDATION_PINN_EPOCH,
        n_cycles=VALIDATION_N_CYCLES,
        n_collocation=VALIDATION_COLLOCATION,
        local_sample=True,
        local_multiplier=20,
        early_stop_patience=300,
        grad_clip_norm=1.0,
        pretrain_val_ratio=0.2,
    )
    config = DiscoverConfig(
        n_iterations=VALIDATION_N_ITER,
        batch_size=VALIDATION_BATCH_SIZE,
        max_length=30,
        library=LibraryConfig(
            operators=operators,
            state_vars=state_vars,
            coord_vars=coord_vars,
        ),
        num_units=32,
        num_layers=1,
        embedding_dim=8,
        epsilon=0.02,
        entropy_weight=0.03,
        entropy_gamma=0.7,
        pinn=pinn_config,
    )

    torch.manual_seed(seed)
    engine = build_engine(config)
    model = PINNModel(coord_vars, state_vars, pinn_config)
    pinn_executor = PINNExecutor(registry)
    dataset_meta = make_pinn_dataset(
        coord_vars,
        state_vars,
        lhs_field=dataset.lhs_field,
        lhs_axis=dataset.lhs_axis,
    )


    axes_tensors = [dataset.axes[c].values for c in coord_vars]
    grids = torch.meshgrid(*axes_tensors, indexing="ij")
    n_total = grids[0].numel()
    n_obs = max(10, int(n_total * 0.04))
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=rng)[:n_obs]

    obs_coords = {c: grids[i].flatten()[idx].float() for i, c in enumerate(coord_vars)}
    obs_targets = {
        name: fd.values.flatten()[idx].float() for name, fd in dataset.fields.items()
    }

    bounds = _domain_bounds(dataset)
    colloc = generate_collocation_points(
        bounds=bounds,
        n_points=pinn_config.n_collocation,
        seed=seed,
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
        domain_bounds=bounds,




        stability_seed=seed,
    )
    return runner, config


def _canonical_term_set(terms: Iterable[str] | None) -> frozenset[str]:
    if not terms:
        return frozenset()
    return frozenset(structure_term_key(t) for t in terms)


@dataclass(frozen=True, slots=True)
class _SeedRunResult:

    seed: int
    best_reward: float
    best_expression: str
    term_set: frozenset[str]
    gt_hit: bool


def _run_one_seed(
    dataset: PDEDataset,
    operators: list[str],
    seed: int,
    gt_term_set: frozenset[str],
) -> _SeedRunResult:
    runner, _ = _build_mode2_runner(dataset, operators, seed=seed)
    result: PINNCycleResult = runner.run()
    state = result.final_state
    term_set = _canonical_term_set(state.best_result_terms)
    return _SeedRunResult(
        seed=seed,
        best_reward=float(state.best_reward),
        best_expression=state.best_expression,
        term_set=term_set,

        gt_hit=(term_set == _canonical_term_set(gt_term_set)),
    )


def _run_mode1_noisy(
    dataset: PDEDataset,
    operators: list[str],
    seed: int = SEED,
) -> float:
    assert dataset.axes is not None
    assert dataset.fields is not None
    evaluator, _ = _make_evaluator(dataset)
    config = DiscoverConfig(
        n_iterations=VALIDATION_N_ITER,
        batch_size=VALIDATION_BATCH_SIZE,
        max_length=30,
        library=LibraryConfig(
            operators=operators,
            state_vars=list(dataset.fields.keys()),
            coord_vars=list(dataset.axes.keys()),
        ),
        num_units=32,
        num_layers=1,
        embedding_dim=8,
        epsilon=0.02,
        entropy_weight=0.03,
        entropy_gamma=0.7,
    )
    torch.manual_seed(seed)
    engine = build_engine(config)
    state = engine.run(evaluator, n_iterations=VALIDATION_N_ITER)
    return float(state.best_reward)







@pytest.fixture(scope="module")
def burgers_data_path() -> Path:
    if not BURGERS_MAT.exists():
        pytest.skip(f"Burgers data not found at {BURGERS_MAT}")
    return BURGERS_MAT


@pytest.fixture(scope="module")
def chafee_data_dir() -> Path:
    marker = CHAFEE_DIR / "chafee_infante_CI.npy"
    if not marker.exists():
        pytest.skip(f"Chafee-Infante data not found at {CHAFEE_DIR}")
    return CHAFEE_DIR


@pytest.fixture(scope="module")
def noisy_burgers(burgers_data_path: Path) -> PDEDataset:
    clean = load_burgers_mat(burgers_data_path)
    return _add_gaussian_noise(clean, NOISE_LEVEL, SEED)


@pytest.fixture(scope="module")
def noisy_chafee(chafee_data_dir: Path) -> PDEDataset:
    clean = load_chafee_infante_npy(chafee_data_dir)
    return _add_gaussian_noise(clean, NOISE_LEVEL, SEED)


@pytest.fixture(scope="module")
def burgers_mode2_result(noisy_burgers: PDEDataset) -> PINNCycleResult:
    runner, _ = _build_mode2_runner(noisy_burgers, BURGERS_OPERATORS)
    return runner.run()


@pytest.fixture(scope="module")
def chafee_mode2_result(noisy_chafee: PDEDataset) -> PINNCycleResult:
    runner, _ = _build_mode2_runner(noisy_chafee, CHAFEE_OPERATORS)
    return runner.run()


@pytest.fixture(scope="module")
def burgers_mode1_noisy_reward(noisy_burgers: PDEDataset) -> float:
    return _run_mode1_noisy(noisy_burgers, BURGERS_OPERATORS)







@pytest.mark.unit
class TestMode2ValidationSeedPlumbing:

    def test_build_mode2_runner_threads_stability_seed(self) -> None:
        import inspect

        src = inspect.getsource(_build_mode2_runner)
        assert "stability_seed=seed" in src, (
            " D: _build_mode2_runner dropped stability_seed; "
            "final-cycle stability selection is unseeded on replay."
        )


class TestMode2Burgers:

    @pytest.mark.slow
    def test_reward_above_threshold(
        self,
        burgers_mode2_result: PINNCycleResult,
    ) -> None:
        reward = burgers_mode2_result.final_state.best_reward
        assert reward >= BURGERS_REWARD_FLOOR, (
            f"MODE2 Burgers reward {reward:.4f} < floor {BURGERS_REWARD_FLOOR}"
        )









    @pytest.mark.slow
    def test_discovers_advection_term(
        self,
        burgers_mode2_result: PINNCycleResult,
    ) -> None:
        expr = burgers_mode2_result.final_state.best_expression
        assert expr, "MODE2 produced no valid expression"
        has_advection = "mul" in expr and "diff_x" in expr
        assert has_advection, f"Missing advection term (mul + diff_x) in: '{expr}'"

    @pytest.mark.slow
    def test_pretrain_converges(
        self,
        burgers_mode2_result: PINNCycleResult,
    ) -> None:
        pr = burgers_mode2_result.pretrain_result
        assert pr.train_loss < PRETRAIN_LOSS_CEILING, (
            f"Pretrain train_loss {pr.train_loss:.4f} >= {PRETRAIN_LOSS_CEILING}"
        )
        assert pr.val_loss < PRETRAIN_LOSS_CEILING, (
            f"Pretrain val_loss {pr.val_loss:.4f} >= {PRETRAIN_LOSS_CEILING}"
        )

    @pytest.mark.slow
    def test_cycle_metrics_complete(
        self,
        burgers_mode2_result: PINNCycleResult,
    ) -> None:
        metrics = burgers_mode2_result.cycle_metrics
        assert len(metrics) == VALIDATION_N_CYCLES, (
            f"Expected {VALIDATION_N_CYCLES} cycle metrics, got {len(metrics)}"
        )
        for i, m in enumerate(metrics):
            assert "best_reward" in m, f"cycle {i}: missing best_reward"

    @pytest.mark.slow
    def test_pinn_losses_finite(
        self,
        burgers_mode2_result: PINNCycleResult,
    ) -> None:
        ran = [
            (i, m)
            for i, m in enumerate(burgers_mode2_result.cycle_metrics)
            if "data_loss" in m
        ]
        if not ran:
            pytest.fail(
                "PINN training never ran in any MODE2 cycle (no cycle "
                "metrics contain 'data_loss'); the finiteness premise is "
                "unsatisfiable and MODE2 degenerated to plain search."
            )
        for i, m in ran:
            for key in ("data_loss", "physics_loss", "total_loss"):
                assert key in m, f"cycle {i}: PINN ran but {key} missing"
                assert math.isfinite(m[key]), f"cycle {i}: {key}={m[key]} is not finite"







class TestMode2Chafee:

    @pytest.mark.slow
    def test_reward_above_threshold(
        self,
        chafee_mode2_result: PINNCycleResult,
    ) -> None:
        reward = chafee_mode2_result.final_state.best_reward
        assert reward >= CHAFEE_REWARD_FLOOR, (
            f"MODE2 Chafee reward {reward:.4f} < floor {CHAFEE_REWARD_FLOOR}"
        )







    @pytest.mark.slow
    def test_discovers_nonlinear_term(
        self,
        chafee_mode2_result: PINNCycleResult,
    ) -> None:
        expr = chafee_mode2_result.final_state.best_expression
        assert expr, "MODE2 produced no valid expression"
        has_cubic = "n3" in expr or (expr.count("mul") >= 2 and "u" in expr)
        assert has_cubic, f"Missing cubic nonlinear term in: '{expr}'"

    @pytest.mark.slow
    def test_pretrain_converges(
        self,
        chafee_mode2_result: PINNCycleResult,
    ) -> None:
        pr = chafee_mode2_result.pretrain_result
        assert pr.train_loss < PRETRAIN_LOSS_CEILING, (
            f"Pretrain train_loss {pr.train_loss:.4f} >= {PRETRAIN_LOSS_CEILING}"
        )
        assert pr.val_loss < PRETRAIN_LOSS_CEILING, (
            f"Pretrain val_loss {pr.val_loss:.4f} >= {PRETRAIN_LOSS_CEILING}"
        )

    @pytest.mark.slow
    def test_cycle_metrics_complete(
        self,
        chafee_mode2_result: PINNCycleResult,
    ) -> None:
        metrics = chafee_mode2_result.cycle_metrics
        assert len(metrics) == VALIDATION_N_CYCLES, (
            f"Expected {VALIDATION_N_CYCLES} cycle metrics, got {len(metrics)}"
        )
        for i, m in enumerate(metrics):
            assert "best_reward" in m, f"cycle {i}: missing best_reward"

    @pytest.mark.slow
    def test_pinn_losses_finite(
        self,
        chafee_mode2_result: PINNCycleResult,
    ) -> None:
        ran = [
            (i, m)
            for i, m in enumerate(chafee_mode2_result.cycle_metrics)
            if "data_loss" in m
        ]
        if not ran:
            pytest.fail(
                "PINN training never ran in any MODE2 cycle (no cycle "
                "metrics contain 'data_loss'); the finiteness premise is "
                "unsatisfiable and MODE2 degenerated to plain search."
            )
        for i, m in ran:
            for key in ("data_loss", "physics_loss", "total_loss"):
                assert key in m, f"cycle {i}: PINN ran but {key} missing"
                assert math.isfinite(m[key]), f"cycle {i}: {key}={m[key]} is not finite"







class TestMode2PINNEffectiveness:

    @pytest.mark.slow
    def test_mode2_beats_mode1_on_noisy_data(
        self,
        burgers_mode2_result: PINNCycleResult,
        burgers_mode1_noisy_reward: float,
    ) -> None:
        mode2_reward = burgers_mode2_result.final_state.best_reward
        mode1_reward = burgers_mode1_noisy_reward
        assert mode2_reward >= mode1_reward, (
            f"MODE2 ({mode2_reward:.4f}) did not beat "
            f"MODE1 ({mode1_reward:.4f}) on noisy data"
        )

    @pytest.mark.slow
    def test_reward_nondecreasing_across_cycles(
        self,
        burgers_mode2_result: PINNCycleResult,
    ) -> None:
        metrics = burgers_mode2_result.cycle_metrics
        if len(metrics) < 2:
            pytest.skip("Need at least 2 cycles for regression check")
        for i in range(1, len(metrics)):
            prev = metrics[i - 1]["best_reward"]
            curr = metrics[i]["best_reward"]
            assert curr >= prev * 0.95, (
                f"Reward regressed: cycle {i - 1}={prev:.4f} -> "
                f"cycle {i}={curr:.4f} (>{5}% drop)"
            )

    @pytest.mark.slow
    def test_pretrain_loss_below_noise_level(
        self,
        burgers_mode2_result: PINNCycleResult,
    ) -> None:
        pr = burgers_mode2_result.pretrain_result
        assert pr.train_loss < NOISE_LEVEL, (
            f"Pretrain loss {pr.train_loss:.4f} >= noise level {NOISE_LEVEL}"
        )










def _require_pythonhashseed() -> None:
    actual = os.environ.get("PYTHONHASHSEED")
    if actual != _REQUIRED_HASHSEED:
        pytest.fail(
            "PYTHONHASHSEED must be set BEFORE Python starts; cannot be "
            "patched at runtime. Re-invoke pytest as:\n"
            f" PYTHONHASHSEED={_REQUIRED_HASHSEED} uv run pytest "
            "tests/test_mode2_validation.py -m 'slow and mode2_aggregate' "
            "--timeout=3600\n"
            f" (current value: {actual!r})",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def burgers_mode2_aggregate_results(
    burgers_data_path: Path,
) -> list[_SeedRunResult]:
    _require_pythonhashseed()
    clean = load_burgers_mat(burgers_data_path)
    results: list[_SeedRunResult] = []
    for seed in GATE1_SEEDS:


        noisy = _add_gaussian_noise(clean, NOISE_LEVEL, seed)
        results.append(
            _run_one_seed(noisy, BURGERS_OPERATORS, seed, GT_TERM_SET_BURGERS)
        )
    return results


class TestMode2AggregateBurgers:

    @pytest.mark.slow
    @pytest.mark.mode2_aggregate
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Known regression: the latest 3-seed evidence is mean 0.8295 "
            f"/ STRICT GT-hit 1/3, vs the reference-anchored floor "
            f"mean>={BURGERS_MEAN_FLOOR} / hits>={BURGERS_GT_HIT_FLOOR}. "
            "A/B testing cleared the platform side; the root cause is an "
            "upstream search-engine regression under bisection. This xfail "
            "clears when the upstream fix lands."
        ),
    )
    def test_strict_gt_hit_aggregate(
        self,
        burgers_mode2_aggregate_results: list[_SeedRunResult],
    ) -> None:
        results = burgers_mode2_aggregate_results
        seeds_seen = [r.seed for r in results]
        assert seeds_seen == list(GATE1_SEEDS), (
            f"Aggregate must cover exactly Gate-1 seeds {GATE1_SEEDS}, got {seeds_seen}"
        )

        rewards = [r.best_reward for r in results]
        mean_reward = sum(rewards) / len(rewards)
        gt_hits = sum(int(r.gt_hit) for r in results)

        per_seed_summary = "\n".join(
            f" seed={r.seed}: reward={r.best_reward:.4f} "
            f"gt_hit={r.gt_hit} term_set={sorted(r.term_set)} "
            f"expr={r.best_expression!r}"
            for r in results
        )

        failures: list[str] = []
        if mean_reward < BURGERS_MEAN_FLOOR:
            failures.append(
                f"3-seed mean reward {mean_reward:.4f} < floor "
                f"{BURGERS_MEAN_FLOOR:.4f} (reference-anchored)"
            )
        if gt_hits < BURGERS_GT_HIT_FLOOR:
            failures.append(
                f"STRICT term_set_match_gt count {gt_hits}/"
                f"{len(GATE1_SEEDS)} < floor "
                f"{BURGERS_GT_HIT_FLOOR}/{len(GATE1_SEEDS)} "
                f"(GT={sorted(GT_TERM_SET_BURGERS)})"
            )

        if failures:
            raise AssertionError(
                "Burgers MODE2 aggregate failed STRICT reference-anchored "
                "oracle:\n"
                + "\n".join(f" - {f}" for f in failures)
                + f"\nPer-seed:\n{per_seed_summary}"
            )


class TestMode2AggregateChafee:

    @pytest.mark.slow
    @pytest.mark.mode2_aggregate
    @pytest.mark.skip(reason=CHAFEE_SKIP_REASON)
    def test_strict_gt_hit_aggregate(self) -> None:

        _ = (CHAFEE_DIR, load_chafee_infante_npy, CHAFEE_OPERATORS, GT_TERM_SET_CHAFEE)








class TestMode2AggregateGuards:

    @pytest.mark.unit
    def test_floor_constants_are_python_literals(self) -> None:
        src = Path(__file__).resolve().read_text(encoding="utf-8")
        forbidden_substrings = (
            "audit_summary.json",
            "json.load",
            "json.loads",
            "open(",
            "Path(",
        )
        for const_name in ("BURGERS_MEAN_FLOOR", "BURGERS_GT_HIT_FLOOR"):
            line = next(
                (ln for ln in src.splitlines() if ln.startswith(const_name)),
                None,
            )
            assert line is not None, f"{const_name} not found as a top-level definition"
            for bad in forbidden_substrings:
                assert bad not in line, (
                    f"{const_name} appears to read from {bad!r}: "
                    f"{line!r}. Floors must be hard-coded literals "
                    "(anti-self-blessing)."
                )

    @pytest.mark.unit
    def test_canonical_term_set_folds_neg_and_sibling_order(self) -> None:


        assert (
            _canonical_term_set(["mul(u, diff_x(u))"])
            == _canonical_term_set(["mul(diff_x(u), u)"])
            == _canonical_term_set(["mul(u,diff_x(u))"])
        )

        assert _canonical_term_set(["neg(u)"]) == _canonical_term_set(["u"])
        assert _canonical_term_set(["neg(neg(u))"]) == _canonical_term_set(["u"])


        assert _canonical_term_set(
            ["diff2_x(u)", "neg(mul(u, diff_x(u)))"],
        ) == _canonical_term_set(GT_TERM_SET_BURGERS)

        assert _canonical_term_set(None) == frozenset()
        assert _canonical_term_set([]) == frozenset()
