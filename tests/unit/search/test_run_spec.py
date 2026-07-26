
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest


def _run_spec_module() -> Any:
    import kd.search.run_spec as run_spec_module

    return run_spec_module


def _run_spec(**overrides: Any) -> Any:
    module = _run_spec_module()
    values: dict[str, Any] = {
        "kd_version": "0.4.0",
        "config": {"algorithm": "sga", "seed": 0},
        "library_fingerprint": None,
        "dataset_cache_fingerprint": "sha256:dataset",
        "artifacts": None,
    }
    values.update(overrides)
    return module.RunSpec(**values)







def test_d5_run_spec_is_frozen() -> None:

    spec = _run_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.kd_version = "9.9.9"


def test_d5_run_spec_hash_is_full_length_hex() -> None:

    digest = _run_spec().run_spec_hash
    assert isinstance(digest, str)
    assert re.fullmatch(r"[0-9a-f]{64}", digest) is not None


def test_d5_run_spec_config_canon_scheme_default() -> None:

    assert _run_spec().config_canon_scheme == "kd-config-v1"


def test_d5_run_spec_from_dict_rejects_unknown_key() -> None:

    data = _run_spec().to_dict()
    data["bogus_spec_field"] = 1
    with pytest.raises(ValueError, match="bogus_spec_field"):
        _run_spec_module().RunSpec.from_dict(data)


def test_d5_run_spec_from_dict_rejects_missing_required_key() -> None:

    data = _run_spec().to_dict()
    del data["kd_version"]
    with pytest.raises(ValueError, match="kd_version"):
        _run_spec_module().RunSpec.from_dict(data)


def test_d5_run_spec_config_key_order_independent() -> None:


    ordered_a = _run_spec(config={"algorithm": "sga", "seed": 0})
    ordered_b = _run_spec(config={"seed": 0, "algorithm": "sga"})
    assert ordered_a.run_spec_hash == ordered_b.run_spec_hash


def test_d5_run_spec_path_and_str_config_equivalent() -> None:


    path_config = _run_spec(config={"out_dir": Path("/tmp/run")})
    str_config = _run_spec(config={"out_dir": "/tmp/run"})
    assert path_config.run_spec_hash == str_config.run_spec_hash







def _canonicalize_config() -> Any:
    return _run_spec_module().canonicalize_config


def test_d6_accepts_json_native_scalars_and_containers() -> None:

    canon = _canonicalize_config()
    payload = {
        "s": "x",
        "b": True,
        "i": 3,
        "f": 1.5,
        "n": None,
        "lst": [1, 2, 3],
        "nested": {"k": "v"},
    }
    assert canon(payload) == payload


def test_d6_tuple_becomes_list() -> None:

    assert _canonicalize_config()({"t": (1, 2, 3)}) == {"t": [1, 2, 3]}


def test_d6_path_becomes_str_verbatim() -> None:

    assert _canonicalize_config()({"p": Path("/a/b")}) == {"p": "/a/b"}


def test_d6_rejects_non_finite_float() -> None:

    canon = _canonicalize_config()
    with pytest.raises((ValueError, TypeError), match="bad"):
        canon({"bad": float("nan")})
    with pytest.raises((ValueError, TypeError), match="bad"):
        canon({"bad": float("inf")})


def test_d6_rejects_non_str_dict_key() -> None:

    with pytest.raises((ValueError, TypeError), match="key"):
        _canonicalize_config()({1: "x"})


def test_d6_rejects_arbitrary_object_with_type_name() -> None:

    class Weird:
        pass

    with pytest.raises((ValueError, TypeError), match="Weird"):
        _canonicalize_config()({"o": Weird()})


def test_d6_rejects_object_with_str_dunder_no_stringify_fallback() -> None:



    class Stringy:
        def __str__(self) -> str:
            return "looks-json-safe"

    with pytest.raises((ValueError, TypeError), match="Stringy"):
        _canonicalize_config()({"o": Stringy()})


def test_d6_output_is_deterministic_across_key_orderings() -> None:

    canon = _canonicalize_config()
    first = canon({"a": 1, "b": 2, "c": 3})
    second = canon({"c": 3, "b": 2, "a": 1})
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_d6_error_message_carries_type_name_and_key_path() -> None:


    class Weird:
        pass

    with pytest.raises((ValueError, TypeError)) as exc_info:
        _canonicalize_config()({"outer": {"inner": Weird()}})
    message = str(exc_info.value)
    assert "Weird" in message
    assert "inner" in message
