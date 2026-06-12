
from __future__ import annotations

from typing import Any

from kd.search.callbacks import CHECKPOINT_VERSION, build_checkpoint_payload


class _StubAlgorithm:

    state: dict[str, Any] = {"weights": [1, 2, 3]}
    best_score = 0.5
    best_expression = "u_t = -u"


def test_payload_wires_all_six_keys() -> None:



    payload = build_checkpoint_payload(7, _StubAlgorithm())
    assert payload == {
        "version": CHECKPOINT_VERSION,
        "iteration": 7,
        "algorithm_state": {"weights": [1, 2, 3]},
        "best_score": 0.5,
        "best_expression": "u_t = -u",
        "algorithm": None,
    }


def test_runner_and_callback_emit_via_single_builder() -> None:
    import inspect

    from kd.search import callbacks, runner

    runner_src = inspect.getsource(runner.ExperimentRunner.save_checkpoint)
    cb_src = inspect.getsource(callbacks.CheckpointCallback)
    assert "build_checkpoint_payload(" in runner_src


    assert cb_src.count("build_checkpoint_payload(") >= 2
    assert "_CHECKPOINT_VERSION" not in runner_src
