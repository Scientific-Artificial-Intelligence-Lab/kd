from __future__ import annotations

import ast
import re

import numpy as np
import pytest

from kd.core.evaluator import EvaluationResult
from kd.search.discover.controller.tree_state import BatchTracker, IncrementalTracker
from kd.search.discover.core.tree import ExpressionTree, finish_tokens, trim_to_natural
from kd.search.discover.evaluation.reward import compute_reward
from kd.search.discover.ir.conversion import ir_to_tree, tokens_to_ir, tree_to_ir
from kd.search.discover.tokens.library import Library, LibraryConfig
from kd.search.discover.tokens.prior import (
    DiffChildConstraint,
    LengthConstraint,
    PriorSystem,
)
from kd.search.discover.tokens.validator import CandidateValidator



BURGERS_CONFIG = LibraryConfig(
    coord_vars=["x"],
    state_vars=["u"],
    operators=[
        "add", "mul", "div", "sub",
        "diff_x", "diff2_x", "diff3_x",
        "n2", "n3", "sin", "cos", "neg",
    ],
)


BURGERS_PREORDER_NAMES = ["add", "mul", "u", "diff_x", "u", "diff2_x", "u"]
BURGERS_IR = "add(mul(u,diff_x(u)),diff2_x(u))"

MAX_LENGTH = 30
KD_DIFF_REGEX = re.compile(r"^diff([0-9]*)_([a-z]+)$")


@pytest.fixture
def lib() -> Library:
    return Library.from_config(BURGERS_CONFIG)


@pytest.fixture
def tokens(lib: Library) -> list[int]:
    return [lib.name_to_index(n) for n in BURGERS_PREORDER_NAMES]




class TestPhase0EndToEnd:

    @pytest.mark.integration
    @pytest.mark.smoke
    def test_burgers_full_pipeline(
        self, lib: Library, tokens: list[int]
    ) -> None:

        assert len(lib.tokens) == len(BURGERS_CONFIG.coord_vars) + \
            len(BURGERS_CONFIG.state_vars) + len(BURGERS_CONFIG.operators)
        assert lib.arities.dtype == np.int32
        assert len(tokens) == 7


        tree = ExpressionTree.from_preorder(tokens, lib)
        assert tree.is_complete()
        assert tree.n_nodes() == 7
        assert tree.n_terms() == 2
        assert tree.to_preorder() == tokens


        B = 1
        inc_tracker = IncrementalTracker(lib)
        obs_inc = inc_tracker.reset(B)
        obs_steps = [obs_inc]
        for t in range(len(tokens) - 1):
            obs_inc = inc_tracker.step(np.array([tokens[t]], dtype=np.int32))
            obs_steps.append(obs_inc)

        actions = np.array([tokens], dtype=np.int32)
        batch_obs = BatchTracker(lib).compute_obs(actions)
        assert batch_obs.shape == (1, 4, 7)
        assert batch_obs.dtype == np.float32


        for t in range(len(tokens)):
            np.testing.assert_array_equal(
                obs_steps[t][0], batch_obs[0, :, t],
                err_msg=f"Inc/batch obs mismatch at step {t}",
            )


        ps = PriorSystem(lib, [
            LengthConstraint(lib, min_=3, max_=MAX_LENGTH),
            DiffChildConstraint(lib),
        ])



        obs_for_prior = inc_tracker.reset(B)
        adjustment = ps.step(inc_tracker.history, obs_for_prior, step_idx=0)
        assert adjustment[0, tokens[0]] == 0.0, \
            f"Token {lib.names[tokens[0]]} forbidden at step 0"

        for t in range(len(tokens) - 1):

            replay = IncrementalTracker(lib)
            obs_for_prior = replay.reset(B)
            for s in range(t + 1):
                obs_for_prior = replay.step(
                    np.array([tokens[s]], dtype=np.int32)
                )
            adjustment = ps.step(replay.history, obs_for_prior, step_idx=t + 1)
            assert adjustment[0, tokens[t + 1]] == 0.0, \
                f"Token {lib.names[tokens[t + 1]]} forbidden at step {t + 1}"
            assert np.any(np.isfinite(adjustment)), \
                f"Dead end at step {t + 1}"



        finished = finish_tokens(tokens, lib)
        assert finished == tokens


        prefix = tokens[:4]
        finished_prefix = finish_tokens(prefix, lib)
        assert len(finished_prefix) >= len(prefix)

        finished_tree = ExpressionTree.from_preorder(finished_prefix, lib)
        assert finished_tree.is_complete()

        cv = CandidateValidator(lib, max_length=MAX_LENGTH)


        assert cv.validate_single(np.array(tokens, dtype=np.int32)) is True

        assert cv.validate_single(
            np.array([lib.name_to_index("u")], dtype=np.int32)
        ) is False

        assert cv.validate_single(
            np.array(tokens[:3], dtype=np.int32)
        ) is False


        ir_str = tokens_to_ir(tokens, lib)
        assert ir_str == BURGERS_IR


        roundtrip_tree = ir_to_tree(ir_str, lib)
        assert roundtrip_tree.to_preorder() == tokens


        assert tree_to_ir(tree) == BURGERS_IR


        good_result = EvaluationResult(
            mse=0.01, nmse=0.01, r2=0.99,
            complexity=2, is_valid=True,
        )
        poor_result = EvaluationResult(
            mse=1.0, nmse=1.0, r2=0.0,
            complexity=2, is_valid=True,
        )
        reward_good = compute_reward(good_result)
        reward_poor = compute_reward(poor_result)
        assert 0.0 < reward_good <= 1.0
        assert reward_good > reward_poor



        parsed = ast.parse(ir_str, mode="eval")
        assert isinstance(parsed, ast.Expression)


        for node in ast.walk(parsed):
            assert not isinstance(node, (ast.BinOp, ast.UnaryOp)), \
                f"Infix operator found: {type(node).__name__}"


        for tok in lib.tokens:
            if tok.name in {"diff_x", "diff2_x", "diff3_x"}:
                assert KD_DIFF_REGEX.match(tok.name), \
                    f"Diff token {tok.name!r} doesn't match kd regex"

    @pytest.mark.integration
    def test_finish_tokens_output_passes_validator(
        self, lib: Library, tokens: list[int]
    ) -> None:
        prefix = tokens[:4]
        finished = finish_tokens(prefix, lib)

        tree = ExpressionTree.from_preorder(finished, lib)
        assert tree.is_complete()

        cv = CandidateValidator(lib, max_length=MAX_LENGTH)
        assert cv.validate_single(np.array(finished, dtype=np.int32)) is True

    @pytest.mark.integration
    def test_diff_constraint_active_at_diff_parent(
        self, lib: Library, tokens: list[int]
    ) -> None:
        ps = PriorSystem(lib, [
            LengthConstraint(lib, min_=3, max_=MAX_LENGTH),
            DiffChildConstraint(lib),
        ])



        tracker = IncrementalTracker(lib)
        obs = tracker.reset(1)
        for t in range(4):
            obs = tracker.step(np.array([tokens[t]], dtype=np.int32))

        adjustment = ps.step(tracker.history, obs, step_idx=4)

        assert adjustment[0, lib.name_to_index("u")] == 0.0

        assert adjustment[0, lib.name_to_index("x")] == float("-inf")

        assert np.any(np.isfinite(adjustment))




class TestKdLiveValidation:

    @pytest.mark.integration
    def test_kd_evaluator_accepts_burgers_ir(self) -> None:
        from kd.core.evaluator import Evaluator
        from kd.core.executor import ExecutionContext
        from kd.core.expr import FunctionRegistry, PythonExecutor
        from kd.core.linear_solve import LeastSquaresSolver
        from kd.data.derivatives import FiniteDiffProvider
        from kd.data.synthetic import generate_burgers_data


        dataset = generate_burgers_data(
            nx=32, nt=16, nu=0.1, noise_level=0.0, seed=42,
        )
        provider = FiniteDiffProvider(dataset, max_order=3)
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=provider,
        )
        registry = FunctionRegistry.create_default()
        executor = PythonExecutor(registry)
        solver = LeastSquaresSolver()
        u_t = context.get_derivative("u", "t", 1)
        evaluator = Evaluator(
            executor=executor, solver=solver, context=context, lhs=u_t,
        )


        ir_str = BURGERS_IR
        result = evaluator.evaluate_expression(ir_str)


        assert result.is_valid, f"kd rejected IR: {result.error_message}"
        assert np.isfinite(result.nmse), f"nmse not finite: {result.nmse}"
        assert result.complexity >= 1


        reward = compute_reward(result)
        assert 0.0 < reward <= 1.0

    @pytest.mark.integration
    @pytest.mark.smoke
    def test_padded_batch_through_validate_trim_ir(
        self, lib: Library, tokens: list[int]
    ) -> None:
        ea = lib.EMPTY_ACTION

        padded = np.array(
            tokens + [ea] * (10 - len(tokens)), dtype=np.int32
        )


        cv = CandidateValidator(lib, max_length=MAX_LENGTH)
        assert cv.validate_single(padded) is True


        trimmed = trim_to_natural(padded, lib)
        assert len(trimmed) == len(tokens)


        ir_str = tokens_to_ir(list(trimmed), lib)
        assert ir_str == BURGERS_IR
