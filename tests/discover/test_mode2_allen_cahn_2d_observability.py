
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kd.search.discover.runners import pinn_cycle_observability as obs


@pytest.mark.unit
def test_should_log_search_iteration_includes_first_interval_and_final() -> None:
    should_log = obs.should_log_search_iteration

    assert should_log(1, 25, 10) is True
    assert should_log(10, 25, 10) is True
    assert should_log(20, 25, 10) is True
    assert should_log(25, 25, 10) is True
    assert should_log(11, 25, 10) is False


@pytest.mark.unit
def test_zero_heartbeat_disables_search_iteration_logging() -> None:
    should_log = obs.should_log_search_iteration

    assert should_log(1, 25, 0) is False
    assert should_log(25, 25, 0) is False


@pytest.mark.unit
def test_stage_checkpoint_writes_json_artifact(
    tmp_path: Path,
) -> None:
    obs.write_stage_checkpoint(
        tmp_path,
        "cycle_00_search",
        {"stage": "cycle_00_search", "best_reward": 0.5},
    )

    payload = json.loads((tmp_path / "cycle_00_search.json").read_text())

    assert payload == {"stage": "cycle_00_search", "best_reward": 0.5}


@pytest.mark.unit
def test_search_heartbeat_payload_is_json_serializable() -> None:
    engine = SimpleNamespace(best_reward=0.75, best_expression="add(u,n3(u))")

    payload = obs.search_heartbeat_payload(
        "cycle_00_search",
        10,
        100,
        {"reward_max": 0.5, "n_unique": 64.0},
        engine,
        0.0,
    )
    encoded = json.dumps(payload, sort_keys=True)

    assert '"stage": "cycle_00_search"' in encoded
    assert payload["iteration"] == 10
    assert payload["global_best"]["reward"] == 0.75
