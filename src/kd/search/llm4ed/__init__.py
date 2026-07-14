"""LLM4ED (EDL) plugin package.

Strict reproduction of the EDL scoring mechanics (reference: menggedu/EDL,
Du et al. 2024):

- :mod:`kd.search.llm4ed.parse` -- LLM equation string -> sympy ->
  per-additive-term kd funcall IR;
- :mod:`kd.search.llm4ed.fd` -- EDL finite-difference operand templates
  (kd's 4th-order FD provider must NOT be substituted);
- :mod:`kd.search.llm4ed.stridge` -- TrainSTRidge outer tolerance sweep +
  valid_coef gate + EDL error taxonomy (inner solve: ``_stridge``, a verbatim
  numpy port of EDL's inner STRidge);
- :mod:`kd.search.llm4ed.reward` -- sparse reward, round(4) canonical;
- :mod:`kd.search.llm4ed.backend` -- offline deterministic Fake provider;
- :mod:`kd.search.llm4ed.prompts` -- EDL prompt templates, response parsers,
  and seeded term permutation.

Everything here is plugin-private. The plugin orchestration lives in
:mod:`kd.search.llm4ed.plugin`, registered with the platform as ``"llm4ed"``.
"""

from __future__ import annotations

from kd.search.llm4ed.backend import FakeLlm4edBackend
from kd.search.llm4ed.fd import (
    OPERAND_ORDER,
    build_operand_columns,
    diff,
    diff2,
    diff3,
    finite_diff,
)
from kd.search.llm4ed.parse import (
    InexpressibleTermError,
    Llm4edParseError,
    ParsedEquation,
    ParsedTerm,
    UndefinedOperandError,
    UndefinedOperatorError,
    equation_to_sympy,
    parse_equation,
    term_to_ir,
)
from kd.search.llm4ed.prompts import (
    DEFAULT_PDE_OPERANDS,
    DEFAULT_PDE_OPERATORS,
    build_evolution_prompt,
    build_initialization_prompt,
    build_optimize_prompt,
    classify_prompt,
    extract_res_blocks,
    normalize_equation_lines,
    parse_response,
    permute_terms,
)
from kd.search.llm4ed.reward import (
    DEFAULT_COMPLEXITY_WEIGHT,
    rounded_sparse_reward,
    sparse_reward,
)
from kd.search.llm4ed.stridge import (
    ERROR_ABNORMAL_COEF,
    ERROR_LSTSQ,
    SparseSolveResult,
    TrainStridgeResult,
    sparse_solve,
    train_stridge,
    valid_coef,
)

__all__ = [
    "DEFAULT_COMPLEXITY_WEIGHT",
    "DEFAULT_PDE_OPERANDS",
    "DEFAULT_PDE_OPERATORS",
    "ERROR_ABNORMAL_COEF",
    "ERROR_LSTSQ",
    "OPERAND_ORDER",
    "FakeLlm4edBackend",
    "InexpressibleTermError",
    "Llm4edParseError",
    "ParsedEquation",
    "ParsedTerm",
    "SparseSolveResult",
    "TrainStridgeResult",
    "UndefinedOperandError",
    "UndefinedOperatorError",
    "build_evolution_prompt",
    "build_initialization_prompt",
    "build_operand_columns",
    "build_optimize_prompt",
    "classify_prompt",
    "diff",
    "diff2",
    "diff3",
    "equation_to_sympy",
    "extract_res_blocks",
    "finite_diff",
    "normalize_equation_lines",
    "parse_equation",
    "parse_response",
    "permute_terms",
    "rounded_sparse_reward",
    "sparse_reward",
    "sparse_solve",
    "term_to_ir",
    "train_stridge",
    "valid_coef",
]
