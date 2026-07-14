
from __future__ import annotations

from collections.abc import Sequence


REDUNDANT_COMBOS: tuple[frozenset[str], ...] = (
    frozenset({"u", "sin(u)"}),
    frozenset({"u", "sinh(u)"}),
    frozenset({"sin(u)", "sinh(u)"}),
    frozenset({"u", "u^2", "u^3"}),
    frozenset({"x", "sinx"}),
)


def should_zero_reward(sentence_words: Sequence[str], variables: Sequence[str]) -> bool:
    term_set = frozenset(sentence_words[::2])
    for combo in REDUNDANT_COMBOS:
        if combo.issubset(term_set):
            return True

    renamed = [
        "sqr(x)" if word == "sqrt(x)" else "sqr(u)" if word == "sqrt(u)" else word
        for word in sentence_words
    ]
    equation = "".join(renamed)
    if "Div" in equation or "Laplace" in equation:
        return False
    return any(variable not in equation for variable in variables)
