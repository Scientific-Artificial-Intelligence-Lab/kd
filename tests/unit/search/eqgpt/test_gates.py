
from __future__ import annotations

import pytest

from kd.search.eqgpt.gates import REDUNDANT_COMBOS, should_zero_reward


def test_redundant_combos_constant_is_the_five_reference_sets() -> None:
    assert set(REDUNDANT_COMBOS) == {
        frozenset({"u", "sin(u)"}),
        frozenset({"u", "sinh(u)"}),
        frozenset({"sin(u)", "sinh(u)"}),
        frozenset({"u", "u^2", "u^3"}),
        frozenset({"x", "sinx"}),
    }


def test_wave_baseline_is_not_zeroed() -> None:
    words = ["ut", "+", "uxxx", "+", "(uux)xx", "+", "ux"]
    assert should_zero_reward(words, ["t", "x"]) is False


@pytest.mark.parametrize(
    "combo_terms",
    [
        ["u", "+", "sin(u)"],
        ["u", "+", "sinh(u)"],
        ["sin(u)", "+", "sinh(u)"],
        ["u", "+", "u^2", "+", "u^3"],
        ["x", "+", "sinx"],
    ],
)
def test_redundant_combos_zero_reward(combo_terms: list[str]) -> None:


    words = ["ut", "+", *combo_terms, "+", "ux"]
    assert should_zero_reward(words, ["t", "x"]) is True


@pytest.mark.parametrize(
    "words",
    [
        ["ut", "+", "ux", "+", "sin(u)"],
        ["ut", "+", "u", "*", "ux"],
        ["ut", "+", "u^2", "+", "ux"],
    ],
)
def test_partial_or_substring_matches_are_not_zeroed(words: list[str]) -> None:
    assert should_zero_reward(words, ["t", "x"]) is False


def test_declared_variable_absent_zeroes_reward() -> None:

    assert should_zero_reward(["ut", "+", "ux"], ["t", "x", "y"]) is True


def test_sqrt_rename_hack_hides_literal_t() -> None:
    assert should_zero_reward(["sqrt(u)"], ["t"]) is True


def test_laplace_exempts_all_axes() -> None:
    assert should_zero_reward(["ut", "+", "Laplace(u)"], ["t", "x", "y"]) is False
