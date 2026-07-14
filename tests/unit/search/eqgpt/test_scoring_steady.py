
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from kd.core.term_cache import TermColumnCache
from kd.search.eqgpt._multicase import assemble_pinned_matrix
from kd.search.eqgpt._scoring import score_candidate
from kd.search.eqgpt.reward import compute_reward
from kd.search.eqgpt.vocab import PLUS_ID, load_vocab

pytestmark = pytest.mark.unit







class _ExecOut:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value


class FakeExecutor:
    def __init__(self, columns: dict[str, torch.Tensor]) -> None:
        self._columns = columns

    def execute(self, term: str, context: Any = None) -> _ExecOut:
        return _ExecOut(self._columns[term])


class FakeContext:
    pass







class TestAssembleFreePivotShape:
    def test_free_pivot_column0_is_first_term_and_shape(self) -> None:
        cols = {
            "u_xx": torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64),
            "u_yy": torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float64),
        }
        matrix = assemble_pinned_matrix(
            ["u_xx", "u_yy"],
            executor=FakeExecutor(cols),
            context=FakeContext(),
            pinned_lhs=None,
            cache=TermColumnCache(),
            cache_generation=0,
        )
        assert matrix.dtype == np.float64
        assert matrix.shape == (4, 2)
        np.testing.assert_allclose(matrix[:, 0], cols["u_xx"].numpy())
        np.testing.assert_allclose(matrix[:, 1], cols["u_yy"].numpy())

    def test_pinned_path_prepends_an_extra_column(self) -> None:
        cols = {
            "u_xx": torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float64),
            "u_yy": torch.tensor([9.0, 10.0, 11.0, 12.0], dtype=torch.float64),
        }
        pinned = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        matrix = assemble_pinned_matrix(
            ["u_xx", "u_yy"],
            executor=FakeExecutor(cols),
            context=FakeContext(),
            pinned_lhs=pinned,
            cache=TermColumnCache(),
            cache_generation=0,
        )
        assert matrix.shape == (4, 3)
        np.testing.assert_allclose(matrix[:, 0], pinned.numpy())







class TestFreePivotRecovery:
    def test_recovers_pivot_normalized_ratio(self) -> None:
        c1 = torch.tensor([1.0, 2.0, -3.0, 0.5, 4.0], dtype=torch.float64)
        c0 = -2.0 * c1
        matrix = assemble_pinned_matrix(
            ["c0", "c1"],
            executor=FakeExecutor({"c0": c0, "c1": c1}),
            context=FakeContext(),
            pinned_lhs=None,
            cache=TermColumnCache(),
            cache_generation=0,
        )
        rr = compute_reward(matrix, sparsity_alpha=1.0)

        assert rr.coefficients.size == 1
        assert rr.coefficients[0] == pytest.approx(2.0, abs=1e-9)
        assert rr.r2 == pytest.approx(1.0, abs=1e-9)
        assert rr.reward > 0.0

    def test_degenerate_constant_pivot_is_invalid_not_a_crash(self) -> None:
        c1 = torch.tensor([1.0, 2.0, -3.0, 0.5, 4.0], dtype=torch.float64)
        const = torch.ones(5, dtype=torch.float64)
        matrix = assemble_pinned_matrix(
            ["const", "c1"],
            executor=FakeExecutor({"const": const, "c1": c1}),
            context=FakeContext(),
            pinned_lhs=None,
            cache=TermColumnCache(),
            cache_generation=0,
        )
        rr = compute_reward(matrix, sparsity_alpha=1.0)
        assert rr.coefficients.size == 0
        assert rr.reward == 0.0
        assert math.isnan(rr.r2)







def test_score_candidate_free_pivot_branch() -> None:
    vocab = load_vocab()
    uxx = vocab.word2id["uxx"]
    uyy = vocab.word2id["uyy"]
    c1 = torch.tensor([1.0, 2.0, -3.0, 0.5, 4.0], dtype=torch.float64)
    c0 = -2.0 * c1
    result = score_candidate(
        candidate="u_xx + u_yy",
        sentence=[uxx, PLUS_ID, uyy],
        vocab=vocab,
        variables=("x", "y"),
        executor=FakeExecutor({"u_xx": c0, "u_yy": c1}),
        context=FakeContext(),
        lhs_flat=None,
        sparsity_alpha=1.0,
    )
    assert result.is_valid is True
    assert result.terms == ["u_xx", "u_yy"]
    assert result.r2 == pytest.approx(1.0, abs=1e-9)
    assert result.score is not None and result.score > 0.0
