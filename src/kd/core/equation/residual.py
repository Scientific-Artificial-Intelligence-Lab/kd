
from __future__ import annotations

import math
from typing import assert_never

from kd.core.equation.types import Equation, Evolution, Homogeneous, Scalar


def residual_program(eq: Equation) -> str:
    match eq:
        case Evolution():
            raise NotImplementedError("EVOLUTION residual_program is reserved")
        case Homogeneous():
            rendered_terms: list[str] = []
            for term_ir, coefficient in eq.terms:
                if not isinstance(coefficient, Scalar):
                    raise NotImplementedError(
                        f"{type(coefficient).__name__} residual coefficients "
                        "are reserved"
                    )





                if not math.isfinite(coefficient.value):
                    raise ValueError(
                        f"residual_program cannot render a non-finite coefficient "
                        f"({coefficient.value!r}) for term {term_ir!r}"
                    )
                rendered_terms.append(
                    f"mul({coefficient.value!r}, {term_ir})"
                )

            residual = rendered_terms[-1]
            for rendered_term in reversed(rendered_terms[:-1]):
                residual = f"add({rendered_term}, {residual})"
            return residual
    assert_never(eq)
