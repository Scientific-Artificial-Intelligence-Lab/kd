
from __future__ import annotations

from typing import Any

import pytest
from kd.search.lifecycle import (
    LifecycleError,
    LifecycleState,
    SearchLifecycle,
)






@pytest.mark.smoke
def test_states_and_error_type_exist() -> None:
    names = {s.name for s in LifecycleState}
    assert names == {"CREATED", "PREPARED", "RUNNING", "DONE"}
    assert issubclass(LifecycleError, RuntimeError)


@pytest.mark.smoke
def test_fresh_machine_starts_created_and_not_restored() -> None:
    lc = SearchLifecycle()
    assert lc.state is LifecycleState.CREATED
    assert lc.restored is False







@pytest.mark.unit
def test_full_fresh_lifecycle() -> None:
    lc = SearchLifecycle()
    lc.prepare()
    assert lc.state is LifecycleState.PREPARED
    lc.iterate()
    assert lc.state is LifecycleState.RUNNING
    lc.iterate()
    assert lc.state is LifecycleState.RUNNING
    lc.finish()
    assert lc.state is LifecycleState.DONE


@pytest.mark.unit
def test_zero_iteration_prepared_to_done_is_legal() -> None:
    lc = SearchLifecycle()
    lc.prepare()
    lc.finish()
    assert lc.state is LifecycleState.DONE


@pytest.mark.unit
def test_restore_then_prepare_is_legal_and_arms_restored_flag() -> None:
    lc = SearchLifecycle()
    lc.restore({"algorithm_state": {"best_expression": "u_x"}})
    assert lc.state is LifecycleState.CREATED
    assert lc.restored is True
    lc.prepare()
    assert lc.state is LifecycleState.PREPARED
    assert lc.restored is True


@pytest.mark.unit
def test_empty_restore_is_fresh_not_restored() -> None:
    lc = SearchLifecycle()
    lc.restore({})
    assert lc.restored is False
    lc.prepare()
    assert lc.state is LifecycleState.PREPARED







def _drive_to_prepared() -> SearchLifecycle:
    lc = SearchLifecycle()
    lc.prepare()
    return lc


def _drive_to_running() -> SearchLifecycle:
    lc = _drive_to_prepared()
    lc.iterate()
    return lc


def _drive_to_done() -> SearchLifecycle:
    lc = _drive_to_prepared()
    lc.finish()
    return lc


@pytest.mark.unit
def test_iterate_before_prepare_raises() -> None:
    lc = SearchLifecycle()
    with pytest.raises(LifecycleError):
        lc.iterate()


@pytest.mark.unit
def test_finish_before_prepare_raises() -> None:
    lc = SearchLifecycle()
    with pytest.raises(LifecycleError):
        lc.finish()


@pytest.mark.unit
@pytest.mark.parametrize(
    "drive",
    [_drive_to_prepared, _drive_to_running, _drive_to_done],
    ids=["from_prepared", "from_running", "from_done"],
)
def test_second_prepare_raises(drive: Any) -> None:
    lc = drive()
    with pytest.raises(LifecycleError):
        lc.prepare()


@pytest.mark.unit
def test_iterate_after_done_raises() -> None:
    lc = _drive_to_done()
    with pytest.raises(LifecycleError):
        lc.iterate()


@pytest.mark.unit
def test_double_finish_raises() -> None:
    lc = _drive_to_done()
    with pytest.raises(LifecycleError):
        lc.finish()


@pytest.mark.unit
@pytest.mark.parametrize(
    "drive",
    [_drive_to_prepared, _drive_to_running, _drive_to_done],
    ids=["from_prepared", "from_running", "from_done"],
)
def test_restore_after_prepare_raises(drive: Any) -> None:
    lc = drive()
    with pytest.raises(LifecycleError):
        lc.restore({"algorithm_state": {"best_expression": "u_x"}})


@pytest.mark.unit
def test_lifecycle_error_message_names_the_state() -> None:
    lc = SearchLifecycle()
    with pytest.raises(LifecycleError) as excinfo:
        lc.iterate()
    message = str(excinfo.value)
    assert message
    assert "CREATED" in message
