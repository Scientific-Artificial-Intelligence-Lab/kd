
from __future__ import annotations

from typing import Any

import pytest

from kd.harness.episode import (
    STATUS_COMPLETED,
    STATUS_NO_RECORD,
    STATUS_RAISED,
    run_episode,
)

from ._helpers import make_entry, make_plan, make_record


_DATASET: Any = object()


class _FakeModel:

    def __init__(self, *, run_record: Any, fit_error: BaseException | None = None):
        self._run_record = run_record
        self._fit_error = fit_error
        self.fit_calls: list[Any] = []

    def fit(self, dataset: Any) -> "_FakeModel":
        self.fit_calls.append(dataset)
        if self._fit_error is not None:
            raise self._fit_error
        return self

    @property
    def result_(self) -> Any:
        return type("_Result", (), {"run_record": self._run_record})()


class _SpyFactory:

    def __init__(self, *, run_record: Any = None, fit_error: BaseException | None = None):
        self.calls: list[dict[str, Any]] = []
        self._run_record = run_record
        self._fit_error = fit_error
        self.model: _FakeModel | None = None

    def __call__(self, **kwargs: Any) -> _FakeModel:
        self.calls.append(kwargs)
        self.model = _FakeModel(run_record=self._run_record, fit_error=self._fit_error)
        return self.model


def test_completed_outcome_carries_record() -> None:
    record = make_record(instrument="sga", seed=0)
    factory = _SpyFactory(run_record=record)
    entry = make_entry("sga", seed=0)

    outcome = run_episode(
        entry=entry, entry_index=3, dataset=_DATASET, model_factory=factory
    )

    assert outcome.status == STATUS_COMPLETED
    assert outcome.record is record
    assert outcome.error_type is None
    assert outcome.error_message is None
    assert outcome.entry_index == 3
    assert outcome.entry is entry
    assert outcome.wallclock_seconds >= 0.0


def test_factory_kwargs_forwarding_via_spy() -> None:

    record = make_record(instrument="pysindy", seed=7)
    factory = _SpyFactory(run_record=record)
    entry = make_entry("pysindy", seed=7, threshold=0.1)

    run_episode(entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory)

    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call["algorithm"] == "pysindy"
    assert call["seed"] == 7
    assert call["verbose"] is False

    assert call["threshold"] == 0.1

    assert factory.model is not None
    assert factory.model.fit_calls == [_DATASET]


def test_factory_raise_becomes_raised_outcome() -> None:
    def factory(**_kwargs: Any) -> Any:
        raise ValueError("bad instrument wiring")

    entry = make_entry("sga", seed=1)
    outcome = run_episode(
        entry=entry, entry_index=2, dataset=_DATASET, model_factory=factory
    )

    assert outcome.status == STATUS_RAISED
    assert outcome.record is None
    assert outcome.error_type == "ValueError"
    assert outcome.error_message is not None
    assert "bad instrument wiring" in outcome.error_message


def test_fit_raise_becomes_raised_outcome() -> None:
    factory = _SpyFactory(fit_error=RuntimeError("solver diverged"))
    entry = make_entry("sga", seed=1)

    outcome = run_episode(
        entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
    )

    assert outcome.status == STATUS_RAISED
    assert outcome.error_type == "RuntimeError"
    assert outcome.error_message == "solver diverged"


def test_error_message_truncated_to_2000_chars() -> None:
    factory = _SpyFactory(fit_error=RuntimeError("x" * 5000))
    entry = make_entry("sga", seed=0)

    outcome = run_episode(
        entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
    )

    assert outcome.error_message is not None
    assert len(outcome.error_message) == 2000


def test_none_run_record_becomes_no_record() -> None:
    factory = _SpyFactory(run_record=None)
    entry = make_entry("sga", seed=0)

    outcome = run_episode(
        entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
    )

    assert outcome.status == STATUS_NO_RECORD
    assert outcome.record is None
    assert outcome.error_type is None
    assert outcome.error_message is None


def test_keyboard_interrupt_propagates_from_factory() -> None:
    def factory(**_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    entry = make_entry("sga", seed=0)
    with pytest.raises(KeyboardInterrupt):
        run_episode(
            entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
        )


def test_keyboard_interrupt_propagates_from_fit() -> None:
    factory = _SpyFactory(fit_error=KeyboardInterrupt())
    entry = make_entry("sga", seed=0)
    with pytest.raises(KeyboardInterrupt):
        run_episode(
            entry=entry, entry_index=0, dataset=_DATASET, model_factory=factory
        )


def test_model_kwargs_deep_copied_so_factory_cannot_corrupt_plan() -> None:


    entry = make_entry("sga", seed=0, coeffs=[1.0, 2.0])
    plan = make_plan([entry])
    hash_before = plan.plan_hash()

    def mutating_factory(**kwargs: Any) -> _FakeModel:
        kwargs["coeffs"].append(999.0)
        return _FakeModel(run_record=make_record("sga", seed=0))

    run_episode(
        entry=entry, entry_index=0, dataset=_DATASET, model_factory=mutating_factory
    )

    assert entry.model_kwargs["coeffs"] == [1.0, 2.0]
    assert plan.plan_hash() == hash_before








class _StrictNoDeviceFactory:

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, algorithm: str, seed: int, verbose: bool) -> _FakeModel:
        self.calls.append({"algorithm": algorithm, "seed": seed, "verbose": verbose})
        return _FakeModel(run_record=None)


def test_episode_device_none_not_forwarded() -> None:
    factory = _StrictNoDeviceFactory()
    run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        device=None,
    )
    assert "device" not in factory.calls[0]


def test_episode_forwards_device_when_set() -> None:
    factory = _SpyFactory(run_record=None)
    run_episode(
        entry=make_entry("sga", seed=0),
        entry_index=0,
        dataset=_DATASET,
        model_factory=factory,
        device="cpu",
    )
    assert factory.calls[0].get("device") == "cpu"
