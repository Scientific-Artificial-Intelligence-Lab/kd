
from __future__ import annotations

import logging
import random
import re
from collections.abc import Sequence
from typing import Final

logger = logging.getLogger(__name__)


DEFAULT_PDE_OPERATORS: Final[str] = "[+, -, *, /, ^2, ^3]"
DEFAULT_PDE_OPERANDS: Final[str] = "[u, u_x, u_xx, u_xxx, x]"


LINK_SYMBOLS: Final[tuple[str, str]] = (" + ", " - ")


INITIALIZATION: Final[str] = "initialization"
OPTIMIZE: Final[str] = "optimize"
EVOLUTION: Final[str] = "evolution"






INITIALIZATION_PROMPT_TEMPLATE: Final[str] = """
You will help me find the optimal governing equation from data. You are \
required to generate equations using symbolic representations. I will \
evaluate them and provide their corresponding scores based on their fitness \
to data. Your task is to find the equation with the highest score. The \
available symbol library for representing equations includes two categories: \
operators and operands. Operators include {0}. Operands include {1}, where u \
denotes the state variable, x denotes the spatial variable, and others are \
derivatives. Now randomly generate {2} diverse equations with different \
lengths in the following form:
1. ...
2. ...
Be as creative as you can under the constraints below.
(1) Do not include coefficients.
(2) Do not omit the multiplication operator.
(3) Only use the symbols provided in the symbol library.

Do not write code and do not give any explanation.
"""

OPTIMIZE_PROMPT_TEMPLATE: Final[str] = """
Task: Given the symbol library including operators:{0} and operands: {1}, You \
will help me generate governing equations using symbolic representations. I \
will evaluate the generated equations and provide their corresponding scores \
based on their fitness to data. Your task is to find the equation with the \
highest score.

Below are some previous equations and their scores, which range from 0 to 1. \
The equations are arranged in ascending order based on their scores, where \
higher values are better.
{2}
Motivated by the equations above, please help me generate {3} new equations \
with higher scores, and bracketed them with <res> and </res>. Try to \
recognize and avoid redundant terms and generate new possibly correct terms.
Pay attention to the format, do not give the scores of the equation and do \
not give additional explanations
"""

EVOLUTION_PROMPT_TEMPLATE: Final[str] = """
Task: Given the symbol library including operators:{0} and operands: {1}, and \
following set of terms:
{2}
please follow the instructions step-by-step to generate new equations:
1. Select two different set of terms from above and bracket them with \
<select> and </select>.
2. Crossover two set of terms chosen in step 1 and generate a new equation \
bracketed with <cross> and </cross>.
3. Mutate the equation generated in Step 2 and generate a new equation \
bracketed with <res> and </res> and.
4. Repeat step 1, 2, 3 for {3} times and directly give me the generated \
equations of each step.

Define Crossover and Mutate.
Crossover: Select half terms of each set and recombine the selected with '+' \
and '-' to generate a new equation.
Mutate: Randomly replace operators or operands of the equation with new ones \
defined in the symbol library.
Pay attention to the format and do not include symbols outside of the \
library. Do not give any explanation.
"""


def build_initialization_prompt(
    n: int,
    *,
    operators: str = DEFAULT_PDE_OPERATORS,
    operands: str = DEFAULT_PDE_OPERANDS,
) -> str:
    return INITIALIZATION_PROMPT_TEMPLATE.format(operators, operands, n)


def build_optimize_prompt(
    history: str,
    n: int,
    *,
    operators: str = DEFAULT_PDE_OPERATORS,
    operands: str = DEFAULT_PDE_OPERANDS,
) -> str:
    return OPTIMIZE_PROMPT_TEMPLATE.format(operators, operands, history, n)


def build_evolution_prompt(
    term_sets: str,
    n: int,
    *,
    operators: str = DEFAULT_PDE_OPERATORS,
    operands: str = DEFAULT_PDE_OPERANDS,
) -> str:
    return EVOLUTION_PROMPT_TEMPLATE.format(operators, operands, term_sets, n)


def classify_prompt(prompt: str) -> str:
    if "<select>" in prompt:
        return EVOLUTION
    if "<res>" in prompt:
        return OPTIMIZE
    return INITIALIZATION






_RES_OPEN = re.compile("<res>")
_RES_CLOSE = re.compile("</res>")
_PREFIX_SPLIT_SYMBOLS: Final[tuple[str, str]] = (".", ":")
_PREFIX_MAX_INDEX: Final[int] = 10


def extract_res_blocks(
    text: str, *, invalids: tuple[str, ...] = ("{", "}")
) -> str:
    blocks = [
        text[start.end(): end.start()]
        for start, end in zip(
            _RES_OPEN.finditer(text), _RES_CLOSE.finditer(text), strict=False
        )
    ]
    joined = "\n".join(blocks)
    for token in invalids:
        joined = joined.replace(token, "")
    return joined


def normalize_equation_lines(text: str) -> list[str]:
    result: list[str] = []
    for line in text.split("\n"):


        expression = re.sub(r",", "+", line)
        try:
            index: int | None = None
            for symbol in _PREFIX_SPLIT_SYMBOLS:
                if symbol in expression:
                    index = expression.index(symbol)
                    if index < _PREFIX_MAX_INDEX:
                        index += 1
                        while expression[index] == " ":
                            index += 1
                        break
            if index is None:
                index = 0
            result.append(expression[index:])
        except IndexError:

            logger.debug("dropping unnormalisable line: %r", line)
            continue
    return result


def parse_response(text: str) -> list[str]:
    payload = extract_res_blocks(text) if ("<" in text or ">" in text) else text
    return normalize_equation_lines(payload)







def permute_terms(terms: Sequence[str], rng: random.Random) -> str:
    ordered = list(terms)
    rng.shuffle(ordered)
    pieces: list[str] = []
    for raw in ordered:
        term = raw[1:] if raw.startswith("-") else raw
        pieces.append(term.replace("**", "^"))
        pieces.append(rng.choice(LINK_SYMBOLS))
    return "".join(pieces[:-1])
