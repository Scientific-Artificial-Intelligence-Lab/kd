
from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import threading
import traceback
import typing
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)

_ONE_AT_A_TIME = threading.Lock()
_STDERR_TAIL_BYTES = 2000


class KdCallError(Exception):

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


class KdProcessError(Exception):
    pass


def call(
    operation: str,
    request: BaseModel,
    response_type: type[ResponseT],
    *,
    timeout_seconds: float,
) -> ResponseT:
    module, function = operation.split(".")
    return run_target(
        f"kdui.ops.{module}:{function}",
        request,
        response_type,
        timeout_seconds=timeout_seconds,
    )


def run_target(
    target: str,
    request: BaseModel,
    response_type: type[ResponseT],
    *,
    timeout_seconds: float,
) -> ResponseT:
    command = [sys.executable, "-m", "kdui.kdproc", target]
    with _ONE_AT_A_TIME:
        try:
            done = subprocess.run(
                command,
                input=request.model_dump_json().encode("utf-8"),
                capture_output=True,
                timeout=timeout_seconds,
                env={**os.environ, "PYTHONUTF8": "1"},
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise KdProcessError(
                f"{target} gave no answer within {timeout_seconds:g} s"
            ) from error
    stderr = done.stderr.decode("utf-8", errors="replace")
    logger.debug("%s stderr: %s", target, stderr)
    if not done.stdout:
        raise KdProcessError(
            f"{target} ended with status {done.returncode} and no answer: "
            f"{stderr[-_STDERR_TAIL_BYTES:]}"
        )
    answer = json.loads(done.stdout)
    if not answer["ok"]:
        raise KdCallError(answer["error_type"], answer["message"])
    return response_type.model_validate(answer["result"])


def _answer(target: str, payload: bytes) -> dict[str, object]:
    try:
        module_name, function_name = target.split(":")
        function = getattr(importlib.import_module(module_name), function_name)
        request_type = typing.get_type_hints(function)["request"]
        result = function(request_type.model_validate_json(payload))
    except Exception as error:


        traceback.print_exc()
        return {"ok": False, "error_type": type(error).__name__, "message": str(error)}
    return {"ok": True, "result": result.model_dump(mode="json")}


def main(argv: Sequence[str] | None = None) -> int:
    (target,) = sys.argv[1:] if argv is None else argv
    channel = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    answer = _answer(target, sys.stdin.buffer.read())
    channel.write(json.dumps(answer).encode("utf-8"))
    channel.close()
    return 0 if answer["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
