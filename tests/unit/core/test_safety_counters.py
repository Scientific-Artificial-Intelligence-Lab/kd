
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from kd.core import safety_counters
from kd.core.safety import safe_div, safe_exp, safe_log

_TEST_LOGGER = logging.getLogger("tests.unit.core.test_safety_counters")


@pytest.fixture
def counters_on() -> Iterator[None]:
    was_enabled = safety_counters.counters_enabled()
    safety_counters.enable_counters()
    safety_counters.reset()
    yield
    safety_counters.reset()
    if not was_enabled:
        safety_counters.disable_counters()


class TestGate:

    @pytest.mark.unit
    def test_disabled_counters_stay_zero(self) -> None:
        was_enabled = safety_counters.counters_enabled()
        safety_counters.disable_counters()
        safety_counters.reset()
        try:
            safe_div(torch.tensor([1.0]), torch.tensor([0.0]))
            safe_exp(torch.tensor([60.0]))
            safe_log(torch.tensor([-1.0]))
            snap = safety_counters.snapshot()
            assert snap == safety_counters.SafetyCounterSnapshot(
                div_calls=0,
                div_guard_fired=0,
                exp_calls=0,
                exp_guard_fired=0,
                log_calls=0,
                log_guard_fired=0,
                div_fired_elems=0,
                div_total_elems=0,
                exp_fired_elems=0,
                exp_total_elems=0,
                log_fired_elems=0,
                log_total_elems=0,
            )
        finally:
            if was_enabled:
                safety_counters.enable_counters()

    @pytest.mark.unit
    def test_enable_disable_roundtrip(self) -> None:
        was_enabled = safety_counters.counters_enabled()
        try:
            safety_counters.enable_counters()
            assert safety_counters.counters_enabled() is True
            safety_counters.disable_counters()
            assert safety_counters.counters_enabled() is False
        finally:
            if was_enabled:
                safety_counters.enable_counters()
            else:
                safety_counters.disable_counters()


class TestBitIdentity:

    @pytest.mark.unit
    def test_bit_identical_results_with_counters_enabled(
        self, counters_on: None
    ) -> None:
        div_a = torch.tensor([6.0, 1.0, 3.0])
        div_b = torch.tensor([2.0, 0.0, 5e-11])
        exp_x = torch.tensor([0.5, 60.0, -60.0])
        log_x = torch.tensor([2.0, 0.0, -3.0, 1e-12])

        for fn, args in (
            (safe_div, (div_a, div_b)),
            (safe_exp, (exp_x,)),
            (safe_log, (log_x,)),
        ):
            safety_counters.disable_counters()
            off = fn(*args)
            safety_counters.enable_counters()
            on = fn(*args)

            assert torch.equal(off, on)

    @pytest.mark.unit
    def test_no_autograd_side_effects(self, counters_on: None) -> None:
        a = torch.tensor([1.0, 2.0, 3.0])

        safety_counters.disable_counters()
        b_off = torch.tensor([2.0, 0.0, 5e-11], requires_grad=True)
        out_off = safe_div(a, b_off)
        out_off.sum().backward()
        assert b_off.grad is not None
        grad_off = b_off.grad.clone()

        safety_counters.enable_counters()
        b_on = torch.tensor([2.0, 0.0, 5e-11], requires_grad=True)
        out_on = safe_div(a, b_on)
        out_on.sum().backward()
        assert b_on.grad is not None

        assert torch.equal(b_on.grad, grad_off)
        assert out_on.requires_grad == out_off.requires_grad


class TestFiredCriteria:

    @pytest.mark.unit
    def test_div_call_and_fired_counts(self, counters_on: None) -> None:
        a = torch.tensor([1.0, 1.0])
        safe_div(a, torch.tensor([2.0, 5.0]))
        safe_div(a, torch.tensor([1.0, 0.0]))
        safe_div(a, torch.tensor([1.0, 5e-11]))
        snap = safety_counters.snapshot()
        assert snap.div_calls == 3
        assert snap.div_guard_fired == 2


        safety_counters.reset()
        safe_div(torch.tensor([1.0]), torch.tensor([0.5]), eps=1.0)
        snap2 = safety_counters.snapshot()
        assert snap2.div_calls == 1
        assert snap2.div_guard_fired == 1

    @pytest.mark.unit
    def test_exp_fired_only_on_clamp(self, counters_on: None) -> None:
        safe_exp(torch.tensor([0.5, -1.0, 1.0]))
        assert safety_counters.snapshot().exp_guard_fired == 0

        safety_counters.reset()
        safe_exp(torch.tensor([0.5, 60.0]))
        assert safety_counters.snapshot().exp_guard_fired == 1

        safety_counters.reset()
        safe_exp(torch.tensor([0.5, -60.0]))
        assert safety_counters.snapshot().exp_guard_fired == 1

        safety_counters.reset()
        safe_exp(torch.tensor([6.0]), max_val=5.0)
        assert safety_counters.snapshot().exp_guard_fired == 1

    @pytest.mark.unit
    def test_log_fired_on_nonpositive_or_near_zero(self, counters_on: None) -> None:
        safe_log(torch.tensor([2.0, 1.0]))
        assert safety_counters.snapshot().log_guard_fired == 0

        safety_counters.reset()
        safe_log(torch.tensor([0.0]))
        assert safety_counters.snapshot().log_guard_fired == 1

        safety_counters.reset()
        safe_log(torch.tensor([-3.0]))
        assert safety_counters.snapshot().log_guard_fired == 1

        safety_counters.reset()
        safe_log(torch.tensor([1e-12]))
        assert safety_counters.snapshot().log_guard_fired == 1

        safety_counters.reset()
        safe_log(torch.tensor([1e-10]))
        assert safety_counters.snapshot().log_guard_fired == 0

    @pytest.mark.unit
    def test_zero_numerator_fires_as_upper_bound(self, counters_on: None) -> None:
        a = torch.zeros(1)

        b = torch.tensor([5e-11])
        raw = a / b
        guarded = safe_div(a, b)

        assert torch.equal(guarded, raw)

        snap = safety_counters.snapshot()
        assert snap.div_guard_fired == 1


        assert snap.div_fired_elems == 1
        assert snap.div_total_elems == 1

    @pytest.mark.unit
    def test_nan_inputs_count_call_not_fired(self, counters_on: None) -> None:
        nan = float("nan")
        safe_div(torch.tensor([1.0]), torch.tensor([nan]))
        safe_exp(torch.tensor([nan]))
        safe_log(torch.tensor([nan]))
        snap = safety_counters.snapshot()
        assert snap.div_calls == 1
        assert snap.exp_calls == 1
        assert snap.log_calls == 1
        assert snap.div_guard_fired == 0
        assert snap.exp_guard_fired == 0
        assert snap.log_guard_fired == 0


class TestElementFraction:

    @pytest.mark.unit
    def test_element_fraction_distinguishes_material_from_vacuous(
        self, counters_on: None
    ) -> None:

        vacuous_b = torch.cat([torch.full((999,), 5.0), torch.zeros(1)])
        safe_div(torch.ones(1000), vacuous_b)
        vac = safety_counters.snapshot()
        assert vac.div_guard_fired == 1
        assert vac.div_fired_elems == 1
        assert vac.div_total_elems == 1000

        safety_counters.reset()

        safe_div(torch.ones(4), torch.zeros(4))
        mat = safety_counters.snapshot()
        assert mat.div_guard_fired == 1
        assert mat.div_fired_elems == 4
        assert mat.div_total_elems == 4

    @pytest.mark.unit
    def test_element_counts_accumulate_across_calls(self, counters_on: None) -> None:
        safe_div(torch.ones(3), torch.tensor([2.0, 0.0, 5.0]))
        safe_div(torch.ones(2), torch.tensor([1.0, 1.0]))
        snap = safety_counters.snapshot()
        assert snap.div_calls == 2
        assert snap.div_guard_fired == 1
        assert snap.div_fired_elems == 1
        assert snap.div_total_elems == 5

    @pytest.mark.unit
    def test_exp_and_log_element_counts(self, counters_on: None) -> None:
        safe_exp(torch.tensor([0.5, 60.0, -60.0]))
        snap = safety_counters.snapshot()
        assert snap.exp_fired_elems == 2
        assert snap.exp_total_elems == 3

        safety_counters.reset()
        safe_log(torch.tensor([2.0, 0.0, -3.0, 1.0]))
        snap2 = safety_counters.snapshot()
        assert snap2.log_fired_elems == 2
        assert snap2.log_total_elems == 4

    @pytest.mark.unit
    def test_nan_counts_toward_total_not_fired_elems(
        self, counters_on: None
    ) -> None:
        safe_div(torch.ones(3), torch.tensor([2.0, float("nan"), 0.0]))
        snap = safety_counters.snapshot()
        assert snap.div_fired_elems == 1
        assert snap.div_total_elems == 3


class TestEmitRunSummary:

    @pytest.mark.unit
    def test_log_only_when_no_file_env(
        self,
        counters_on: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv(safety_counters._ENV_FILE, raising=False)
        safe_div(torch.tensor([1.0]), torch.tensor([0.0]))

        with caplog.at_level(logging.INFO, logger=_TEST_LOGGER.name):
            snap = safety_counters.emit_run_summary("unit:log-only", _TEST_LOGGER)

        assert snap is not None
        assert snap.div_calls == 1
        assert snap.div_guard_fired == 1
        messages = [
            r.getMessage()
            for r in caplog.records
            if "[SAFETY-COUNTERS]" in r.getMessage()
        ]
        assert len(messages) == 1
        assert "div_calls=1" in messages[0]

        assert messages[0].endswith("label=unit:log-only")
        assert safety_counters.snapshot().div_calls == 0

    @pytest.mark.unit
    def test_file_sink_appends_jsonl_records(
        self,
        counters_on: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        sink = tmp_path / "nested" / "safety-counters.jsonl"
        monkeypatch.setenv(safety_counters._ENV_FILE, str(sink))

        safe_div(torch.tensor([1.0]), torch.tensor([2.0]))
        safe_div(torch.tensor([1.0]), torch.tensor([0.0]))
        safety_counters.emit_run_summary("unit:first", _TEST_LOGGER)

        safe_log(torch.tensor([-1.0]))
        safety_counters.emit_run_summary("unit:second", _TEST_LOGGER)

        lines = sink.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["label"] == "unit:first"
        assert first["div_calls"] == 2
        assert first["div_guard_fired"] == 1

        assert first["div_fired_elems"] == 1
        assert first["div_total_elems"] == 2
        assert isinstance(first["ts"], str)
        assert isinstance(first["pid"], int)
        second = json.loads(lines[1])
        assert second["label"] == "unit:second"

        assert second["div_calls"] == 0
        assert second["log_calls"] == 1
        assert second["log_guard_fired"] == 1

    @pytest.mark.unit
    def test_disabled_emits_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sink = tmp_path / "safety-counters.jsonl"
        monkeypatch.setenv(safety_counters._ENV_FILE, str(sink))
        was_enabled = safety_counters.counters_enabled()
        safety_counters.disable_counters()
        try:
            with caplog.at_level(logging.INFO, logger=_TEST_LOGGER.name):
                result = safety_counters.emit_run_summary("unit:off", _TEST_LOGGER)
            assert result is None
            assert not sink.exists()
            assert all(
                "[SAFETY-COUNTERS]" not in r.getMessage() for r in caplog.records
            )
        finally:
            if was_enabled:
                safety_counters.enable_counters()

    @pytest.mark.unit
    def test_file_sink_failure_warns_but_never_raises(
        self,
        counters_on: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:

        monkeypatch.setenv(safety_counters._ENV_FILE, str(tmp_path))
        safe_div(torch.tensor([1.0]), torch.tensor([0.0]))

        with caplog.at_level(logging.INFO, logger=_TEST_LOGGER.name):
            snap = safety_counters.emit_run_summary("unit:bad-sink", _TEST_LOGGER)

        assert snap is not None
        warnings = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("file sink write failed" in r.getMessage() for r in warnings)
        assert safety_counters.snapshot().div_calls == 0


class TestSnapshotReset:

    @pytest.mark.unit
    def test_snapshot_reset_roundtrip(self, counters_on: None) -> None:
        a = torch.tensor([1.0])
        safe_div(a, torch.tensor([0.0]))
        snap1 = safety_counters.snapshot()
        assert snap1.div_calls == 1


        safe_div(a, torch.tensor([0.0]))
        assert snap1.div_calls == 1
        assert safety_counters.snapshot().div_calls == 2

        safety_counters.reset()
        assert safety_counters.snapshot() == safety_counters.SafetyCounterSnapshot(
            div_calls=0,
            div_guard_fired=0,
            exp_calls=0,
            exp_guard_fired=0,
            log_calls=0,
            log_guard_fired=0,
            div_fired_elems=0,
            div_total_elems=0,
            exp_fired_elems=0,
            exp_total_elems=0,
            log_fired_elems=0,
            log_total_elems=0,
        )
