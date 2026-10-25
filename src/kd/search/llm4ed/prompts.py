
from __future__ import annotations

import logging
import random
import re
from collections.abc import Sequence
from typing import Final

logger = logging.getLogger(__name__)


DEFAULT_PDE_OPERATORS: Final[str] = "[+, -, *, /, ^2, ^3]"
DEFAULT_PDE_OPERANDS: Final[str] = "[u, u_x, u_xx, u_xxx, x]"


_SPATIAL_SENTENCE: Final[str] = "x denotes the spatial variable"


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


def pde_operands(operands: Sequence[str]) -> str:
    field = [name for name in operands if name == "u"]
    derivatives = [name for name in operands if name.startswith("u_")]
    coordinates = [
        name for name in operands if name != "u" and not name.startswith("u_")
    ]
    return _symbol_list([*field, *derivatives, *coordinates])


def _initialization_template(spatial_axes: Sequence[str]) -> str:
    axes = tuple(spatial_axes)
    if axes == ("x",):
        return INITIALIZATION_PROMPT_TEMPLATE
    if len(axes) == 1:
        sentence = f"{axes[0]} denotes the spatial variable"
    else:
        sentence = " and ".join(axes) + " denote the spatial variables"
    return INITIALIZATION_PROMPT_TEMPLATE.replace(_SPATIAL_SENTENCE, sentence)


def build_initialization_prompt(
    n: int,
    *,
    operators: str = DEFAULT_PDE_OPERATORS,
    operands: str = DEFAULT_PDE_OPERANDS,
    spatial_axes: Sequence[str] = ("x",),
) -> str:
    return _initialization_template(spatial_axes).format(operators, operands, n)


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










TABULAR_OPERATORS: Final[str] = "[+, -, *, /, ^2, ^3]"

_TABULAR_RULES: Final[str] = """\
(1) Do not include coefficients in front of terms; they are fitted from data.
(2) Write each constant inside a term as a placeholder c1, c2, c3, ..., for \
example sin(c1*v) or 1/(v + c1) for a variable v; write constants such as pi \
or e as placeholders too. Integers may be written directly. Do not write \
decimal numbers.
(3) A constant term may be written as an integer, for example + 1; its value \
is fitted like the other coefficients.
(4) Do not omit the multiplication operator.
(5) Only use the symbols provided in the symbol library."""

TABULAR_INITIALIZATION_PROMPT_TEMPLATE: Final[str] = """
You will help me find the formula that predicts the target variable {target} \
from data. You are required to generate formulas using symbolic \
representations. I will evaluate them and provide their corresponding scores \
based on their fitness to data. Your task is to find the formula with the \
highest score. The available symbol library includes operators {operators}, \
functions {functions} and variables {variables}. Write only the right-hand \
side of {target} = f(variables). Now randomly generate {n} diverse formulas \
with different lengths in the following form:
1. ...
2. ...
Be as creative as you can under the constraints below.
{rules}

Do not write code and do not give any explanation.
"""

TABULAR_OPTIMIZE_PROMPT_TEMPLATE: Final[str] = """
Task: Given the symbol library including operators: {operators}, functions: \
{functions} and variables: {variables}, you will help me generate formulas \
that predict the target variable {target}, writing only the right-hand side. \
I will evaluate the generated formulas and provide their corresponding scores \
based on their fitness to data. Your task is to find the formula with the \
highest score.

Below are some previous formulas and their scores, which range from 0 to 1. \
The formulas are arranged in ascending order based on their scores, where \
higher values are better.
{history}
Motivated by the formulas above, please help me generate {n} new formulas \
with higher scores, and bracket them with <res> and </res>. Try to recognize \
and avoid redundant terms and generate new possibly correct terms. Follow \
these constraints:
{rules}
Pay attention to the format, do not give the scores of the formulas and do \
not give additional explanations
"""

TABULAR_EVOLUTION_PROMPT_TEMPLATE: Final[str] = """
Task: Given the symbol library including operators: {operators}, functions: \
{functions} and variables: {variables}, for formulas that predict the target \
variable {target}, and following set of terms:
{term_sets}
please follow the instructions step-by-step to generate new formulas:
1. Select two different set of terms from above and bracket them with \
<select> and </select>.
2. Crossover two set of terms chosen in step 1 and generate a new formula \
bracketed with <cross> and </cross>.
3. Mutate the formula generated in Step 2 and generate a new formula \
bracketed with <res> and </res>.
4. Repeat step 1, 2, 3 for {n} times and directly give me the generated \
formulas of each step.

Define Crossover and Mutate.
Crossover: Select half terms of each set and recombine the selected with '+' \
and '-' to generate a new formula.
Mutate: Randomly replace operators, functions or variables of the formula \
with new ones defined in the symbol library.
Every formula follows these constraints:
{rules}
Pay attention to the format and do not include symbols outside of the \
library. Do not give any explanation.
"""


def _symbol_list(names: Sequence[str]) -> str:
    return "[" + ", ".join(names) + "]"


def _tabular_library(
    target: str, variables: Sequence[str], functions: Sequence[str]
) -> dict[str, str]:
    return {
        "target": target,
        "operators": TABULAR_OPERATORS,
        "functions": _symbol_list(functions),
        "variables": _symbol_list(variables),
        "rules": _TABULAR_RULES,
    }


def build_tabular_initialization_prompt(
    n: int, *, target: str, variables: Sequence[str], functions: Sequence[str]
) -> str:
    library = _tabular_library(target, variables, functions)
    return TABULAR_INITIALIZATION_PROMPT_TEMPLATE.format(n=n, **library)


def build_tabular_optimize_prompt(
    history: str,
    n: int,
    *,
    target: str,
    variables: Sequence[str],
    functions: Sequence[str],
) -> str:
    library = _tabular_library(target, variables, functions)
    return TABULAR_OPTIMIZE_PROMPT_TEMPLATE.format(history=history, n=n, **library)


def build_tabular_evolution_prompt(
    term_sets: str,
    n: int,
    *,
    target: str,
    variables: Sequence[str],
    functions: Sequence[str],
) -> str:
    library = _tabular_library(target, variables, functions)
    return TABULAR_EVOLUTION_PROMPT_TEMPLATE.format(term_sets=term_sets, n=n, **library)


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


def extract_res_blocks(text: str, *, invalids: tuple[str, ...] = ("{", "}")) -> str:
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





_LIST_INDEX: Final = re.compile(r"\s*\d+(?:\.(?=\s|$)|[):])\s*")


def normalize_tabular_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.split("\n"):
        expression = re.sub(r",", "+", line)
        index = _LIST_INDEX.match(expression)
        lines.append(expression[index.end():] if index is not None else expression)
    return lines


def parse_tabular_response(text: str) -> list[str]:
    payload = extract_res_blocks(text) if ("<" in text or ">" in text) else text
    return normalize_tabular_lines(payload)


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
