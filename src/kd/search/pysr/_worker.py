
from __future__ import annotations

import os
import pickle
import sys
import traceback
from typing import Any

import sympy

from kd.search.pysr.backend import _PySRRegressorBackend


def main(argv: list[str]) -> None:
    request_path, result_path = argv
    with open(request_path, "rb") as f:
        request = pickle.load(f)
    result: dict[str, Any]
    try:
        backend = _PySRRegressorBackend(request["config"])
        backend.fit(
            request["X"],
            request["y"],
            request["variable_names"],
            search_state=request["search_state"],
        )
        result = {
            "best": sympy.srepr(backend.best_sympy()),
            "hof": [
                (entry.complexity, entry.loss, sympy.srepr(entry.sympy_expr))
                for entry in backend.hall_of_fame()
            ],
            "search_state": backend.search_state(),
        }
    except Exception:
        result = {"error": traceback.format_exc()}
    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
