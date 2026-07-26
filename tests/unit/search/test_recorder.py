
from __future__ import annotations

import logging

import pytest
import torch

from kd.search.recorder import BEST_SCORE_KEY, VizRecorder






def test_best_score_key_value_is_a_persistence_contract() -> None:
    assert BEST_SCORE_KEY == "_best_score"


@pytest.mark.smoke
class TestVizRecorderSmoke:

    def test_instantiate(self) -> None:
        rec = VizRecorder()
        assert rec.enabled is True

    def test_log_get_keys_exist(self) -> None:
        rec = VizRecorder()
        rec.log("x", 1)
        assert "x" in rec.keys()
        assert rec.get("x") == [1]







class TestVizRecorderCore:

    def test_log_appends_multiple_values(self) -> None:
        rec = VizRecorder()
        rec.log("loss", 0.5)
        rec.log("loss", 0.3)
        rec.log("loss", 0.1)
        assert rec.get("loss") == [0.5, 0.3, 0.1]

    def test_multiple_keys_independent(self) -> None:
        rec = VizRecorder()
        rec.log("a", 1)
        rec.log("b", 2)
        rec.log("a", 3)
        assert rec.get("a") == [1, 3]
        assert rec.get("b") == [2]
        assert rec.keys() == {"a", "b"}

    def test_get_missing_key_returns_empty_list(self) -> None:
        rec = VizRecorder()
        result = rec.get("nonexistent")
        assert result == []

    def test_keys_returns_set(self) -> None:
        rec = VizRecorder()
        rec.log("x", 1)
        rec.log("y", 2)
        keys = rec.keys()
        assert isinstance(keys, set)
        assert keys == {"x", "y"}

    def test_log_accepts_diverse_types(self) -> None:
        rec = VizRecorder()
        rec.log("int", 42)
        rec.log("float", 3.14)
        rec.log("str", "hello")
        rec.log("dict", {"a": 1})
        assert len(rec.get("int")) == 1
        assert len(rec.get("str")) == 1







class TestVizRecorderDisabled:

    def test_disabled_skips_logging(self) -> None:
        rec = VizRecorder(enabled=False)
        rec.log("loss", 1.0)
        rec.log("loss", 2.0)
        assert rec.get("loss") == []
        assert rec.keys() == set()

    def test_disabled_skips_tensor_logging(self) -> None:
        rec = VizRecorder(enabled=False)
        rec.log("t", torch.tensor(1.0))
        assert rec.get("t") == []







class TestVizRecorderTensorHandling:

    def test_scalar_tensor_stored_as_python_number(self) -> None:
        rec = VizRecorder()
        rec.log("val", torch.tensor(3.14))
        stored = rec.get("val")[0]

        assert not isinstance(stored, torch.Tensor)
        assert isinstance(stored, float)
        torch.testing.assert_close(
            torch.tensor(stored), torch.tensor(3.14), rtol=1e-5, atol=1e-8
        )

    def test_nonscalar_tensor_is_detached(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2
        rec = VizRecorder()
        rec.log("vec", y)
        stored = rec.get("vec")[0]
        assert isinstance(stored, torch.Tensor)
        assert not stored.requires_grad
        torch.testing.assert_close(stored, torch.tensor([2.0, 4.0, 6.0]))

    def test_nonscalar_tensor_moved_to_cpu(self) -> None:

        t = torch.tensor([1.0, 2.0])
        rec = VizRecorder()
        rec.log("t", t)
        stored = rec.get("t")[0]
        assert stored.device == torch.device("cpu")

    def test_large_tensor_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        big = torch.zeros(10_001)
        rec = VizRecorder()
        with caplog.at_level(logging.WARNING):
            rec.log("big", big)
        assert any("large tensor" in m.lower() for m in caplog.messages)
        assert any("10001" in m for m in caplog.messages)

    def test_exactly_10k_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        t = torch.zeros(10_000)
        rec = VizRecorder()
        with caplog.at_level(logging.WARNING):
            rec.log("ok", t)
        assert not any("large tensor" in m.lower() for m in caplog.messages)

    def test_tensor_in_list_isolated_from_inplace_mutation(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        rec = VizRecorder()
        rec.log("listed", [t, "meta"])
        t[0] = 999.0
        stored = rec.get("listed")[0]
        torch.testing.assert_close(stored[0], torch.tensor([1.0, 2.0, 3.0]))

    def test_tensor_in_dict_isolated_from_inplace_mutation(self) -> None:
        t = torch.tensor([5.0, 6.0])
        rec = VizRecorder()
        rec.log("dicted", {"data": t, "tag": "x"})
        t[0] = 777.0
        stored = rec.get("dicted")[0]
        torch.testing.assert_close(stored["data"], torch.tensor([5.0, 6.0]))

    def test_tensor_in_dataclass_isolated_from_inplace_mutation(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class _Result:
            tag: str
            data: torch.Tensor

        t = torch.tensor([1.0, 2.0, 3.0])
        rec = VizRecorder()
        rec.log("dc", _Result(tag="x", data=t))
        t[0] = 999.0
        stored = rec.get("dc")[0]
        assert isinstance(stored, _Result)
        assert stored.tag == "x"
        torch.testing.assert_close(stored.data, torch.tensor([1.0, 2.0, 3.0]))

    def test_one_element_tensor_in_dataclass_keeps_tensor_type(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class _Eval:
            mse: float
            coefficients: torch.Tensor

        coef = torch.tensor([0.5])
        rec = VizRecorder()
        rec.log("eval", _Eval(mse=0.1, coefficients=coef))
        coef[0] = 999.0
        stored = rec.get("eval")[0]
        assert isinstance(stored, _Eval)

        assert isinstance(stored.coefficients, torch.Tensor)

        torch.testing.assert_close(stored.coefficients, torch.tensor([0.5]))

    def test_tensor_in_namedtuple_preserves_type_and_isolated(self) -> None:
        from collections import namedtuple

        Pair = namedtuple("Pair", ["data", "tag"])
        t = torch.tensor([5.0, 6.0])
        rec = VizRecorder()
        rec.log("nt", Pair(data=t, tag="y"))
        t[0] = 777.0
        stored = rec.get("nt")[0]
        assert isinstance(stored, Pair)
        assert stored.tag == "y"
        torch.testing.assert_close(stored.data, torch.tensor([5.0, 6.0]))

    def test_scalar_tensors_in_set_isolated_from_inplace_mutation(self) -> None:
        a = torch.tensor(1.0)
        b = torch.tensor(2.0)
        rec = VizRecorder()
        rec.log("s", {a, b})
        a[...] = 999.0
        b[...] = 888.0
        stored = rec.get("s")[0]
        assert isinstance(stored, set)
        assert stored == {1.0, 2.0}







class TestVizRecorderNanInf:

    def test_nan_float_becomes_none_in_to_dict(self) -> None:
        rec = VizRecorder()
        rec.log("score", float("nan"))
        d = rec.to_dict()
        assert d["score"][0] is None

    def test_inf_float_becomes_none_in_to_dict(self) -> None:
        rec = VizRecorder()
        rec.log("score", float("inf"))
        rec.log("score", float("-inf"))
        d = rec.to_dict()
        assert d["score"][0] is None
        assert d["score"][1] is None

    def test_nan_tensor_scalar_becomes_none(self) -> None:
        rec = VizRecorder()
        rec.log("val", torch.tensor(float("nan")))
        d = rec.to_dict()

        assert d["val"][0] is None

    def test_tensor_with_nan_elements(self) -> None:
        import json

        rec = VizRecorder()
        t = torch.tensor([1.0, float("nan"), 3.0])
        rec.log("vec", t)
        d = rec.to_dict()

        serialized = json.dumps(d)
        assert isinstance(serialized, str)

        assert d["vec"][0][1] is None







class TestVizRecorderSerialization:

    def test_round_trip_plain_values(self) -> None:
        rec = VizRecorder()
        rec.log("loss", 0.5)
        rec.log("loss", 0.3)
        rec.log("iter", 1)
        rec.log("iter", 2)

        d = rec.to_dict()
        restored = VizRecorder.from_dict(d)

        assert restored.get("loss") == [0.5, 0.3]
        assert restored.get("iter") == [1, 2]
        assert restored.keys() == {"loss", "iter"}

    def test_round_trip_tensor_values(self) -> None:
        rec = VizRecorder()
        original = torch.tensor([1.0, 2.0, 3.0])
        rec.log("vec", original)

        d = rec.to_dict()

        series = d["vec"]
        assert len(series) == 1

        item = series[0]
        assert isinstance(item, list)
        torch.testing.assert_close(torch.tensor(item), original, rtol=1e-5, atol=1e-8)

    def test_to_dict_non_serializable_degrades_to_str(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:

        class Opaque:
            def __str__(self) -> str:
                return "opaque-object"

        rec = VizRecorder()
        rec.log("obj", Opaque())

        with caplog.at_level(logging.WARNING):
            d = rec.to_dict()


        assert d["obj"][0] == "opaque-object"

        assert any("obj" in m for m in caplog.messages)

    def test_to_dict_is_json_safe(self) -> None:
        import json

        rec = VizRecorder()
        rec.log("scalar", torch.tensor(1.0))
        rec.log("vec", torch.tensor([1.0, 2.0]))
        rec.log("plain", 42)
        rec.log("text", "hello")

        d = rec.to_dict()

        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_from_dict_round_trip_with_tensor_derived_lists(self) -> None:
        rec = VizRecorder()
        rec.log("vec", torch.tensor([10.0, 20.0, 30.0]))
        rec.log("vec", torch.tensor([40.0, 50.0]))

        d = rec.to_dict()
        restored = VizRecorder.from_dict(d)

        assert len(restored.get("vec")) == 2

        assert len(restored.get("vec")[0]) == 3

    def test_from_dict_creates_enabled_recorder(self) -> None:
        rec = VizRecorder.from_dict({"a": [1, 2]})
        assert rec.enabled is True
        assert rec.get("a") == [1, 2]

    def test_round_trip_empty_recorder(self) -> None:
        rec = VizRecorder()
        d = rec.to_dict()
        restored = VizRecorder.from_dict(d)
        assert restored.keys() == set()

    def test_round_trip_preserves_scalar_tensors_as_numbers(self) -> None:
        rec = VizRecorder()
        rec.log("scalar", torch.tensor(42.0))
        d = rec.to_dict()
        restored = VizRecorder.from_dict(d)
        val = restored.get("scalar")[0]
        assert isinstance(val, (int, float))
        assert val == pytest.approx(42.0)
