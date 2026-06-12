
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from kd.search.discover.runners import mode2_pipeline
from kd.search.discover.runners.mode2_payload import assemble_payload
from kd.search.discover.runners.pde_registry import PDE_REGISTRY


_SCHEMA_BACKED_PDES: tuple[str, ...] = ("burgers", "chafee")

_SCHEMALESS_PDES: tuple[str, ...] = (
    "fisher_linear",
    "fisher_nonlinear",
    "kdv",
    "pde_compound",
    "pde_divide",
)

_SCHEMA_ERROR_PATTERN = "(?i)payload schema"


def _any_tier(pde: str) -> str:
    return sorted(PDE_REGISTRY[pde].presets)[0]


class _PipelineExecutedError(Exception):
    pass


def _install_pipeline_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[str]:
    calls: list[str] = []
    real_resolve = mode2_pipeline._resolve_spec_and_settings
    fake_data = tmp_path / "fake_data.mat"
    fake_data.write_bytes(b"")

    def _resolve_with_existing_data(pde: str, tier: str) -> tuple[Any, Any]:
        spec, settings = real_resolve(pde, tier)
        return spec, dataclasses.replace(settings, data_path=fake_data)

    def _sentinel_execute(**kwargs: Any) -> tuple[dict[str, Any], float]:
        spec = kwargs.get("spec")
        calls.append(getattr(spec, "pde_name", "<unknown>"))
        raise _PipelineExecutedError("heavy pipeline body entered")

    monkeypatch.setattr(
        mode2_pipeline, "_resolve_spec_and_settings", _resolve_with_existing_data
    )
    monkeypatch.setattr(mode2_pipeline, "_execute_pipeline", _sentinel_execute)
    return calls





@pytest.mark.unit
@pytest.mark.parametrize("pde", _SCHEMALESS_PDES)
def test_run_pipeline_rejects_schemaless_pde_before_execution(
    pde: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_pipeline_probe(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=_SCHEMA_ERROR_PATTERN):
        mode2_pipeline.run_pipeline(pde, _any_tier(pde), 0, device="cpu")
    assert calls == [], (
        f"_execute_pipeline was entered for schema-less PDE {pde!r}; "
        "the guard must fire before the multi-hour pipeline body"
    )


@pytest.mark.unit
@pytest.mark.parametrize("pde", _SCHEMA_BACKED_PDES)
def test_run_pipeline_admits_schema_backed_pdes(
    pde: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_pipeline_probe(monkeypatch, tmp_path)
    with pytest.raises(_PipelineExecutedError):
        mode2_pipeline.run_pipeline(pde, _any_tier(pde), 0, device="cpu")
    assert calls == [pde]





@pytest.mark.unit
def test_supported_constant_matches_expected_split() -> None:
    from kd.search.discover.runners.mode2_payload import (
        SUPPORTED_PAYLOAD_PDES,
    )

    supported = set(SUPPORTED_PAYLOAD_PDES)
    assert supported <= set(PDE_REGISTRY), (
        "SUPPORTED_PAYLOAD_PDES names PDEs missing from PDE_REGISTRY: "
        f"{supported - set(PDE_REGISTRY)!r}"
    )
    assert supported == set(_SCHEMA_BACKED_PDES)
    assert set(_SCHEMALESS_PDES) == set(PDE_REGISTRY) - supported, (
        "schema-less parametrize list in this file is out of sync with "
        "the registry/constant split — update _SCHEMALESS_PDES"
    )


def _make_fake_result() -> Any:
    pretrain = type(
        "FakePretrainResult",
        (),
        {
            "train_loss": 0.25,
            "val_loss": 0.5,
            "epochs_run": 3,
            "stopped_early": False,
        },
    )()
    final_state = type(
        "FakeFinalState",
        (),
        {
            "best_reward": 0.5,
            "best_expression": "fake_expression",
            "best_result_terms": ["fake_term"],
            "best_result_coefficients": [1.0],
            "extras": {},
        },
    )()
    return type(
        "FakeRunResult",
        (),
        {
            "pretrain_result": pretrain,
            "final_state": final_state,
            "cycle_metrics": [{"cycle": 0, "reward": 0.5}],
        },
    )()


def _make_fake_engine() -> Any:
    return type(
        "FakeEngine",
        (),
        {
            "cycle_top_candidates": [],
            "best_expression": "fake_expression",
            "best_reward": 0.5,
        },
    )()


@pytest.mark.unit
def test_assemble_payload_succeeds_for_every_supported_pde() -> None:
    from kd.search.discover.runners.mode2_payload import (
        SUPPORTED_PAYLOAD_PDES,
    )

    colloc = {
        "x": torch.zeros(5, dtype=torch.float32),
        "t": torch.zeros(5, dtype=torch.float32),
    }
    for pde in sorted(SUPPORTED_PAYLOAD_PDES):
        spec = PDE_REGISTRY[pde]
        tier = _any_tier(pde)
        config, pinn_config = mode2_pipeline.build_configs(pde=pde, tier=tier, seed=0)
        payload = assemble_payload(
            spec=spec,
            settings=spec.presets[tier],
            config=config,
            pinn_config=pinn_config,
            colloc=colloc,
            seed=0,
            noise_level=0.5,
            scaffold_kwargs={},
            result=_make_fake_result(),
            engine=_make_fake_engine(),
            tier_name=tier,
        )
        assert payload["seed"] == 0, f"payload for {pde!r} lost the seed"
        assert payload["tier"] == tier





@pytest.mark.unit
def test_assemble_payload_error_lists_supported_pdes() -> None:
    spec = PDE_REGISTRY["kdv"]
    tier = _any_tier("kdv")
    with pytest.raises(ValueError, match=_SCHEMA_ERROR_PATTERN) as excinfo:
        assemble_payload(
            spec=spec,
            settings=spec.presets[tier],
            config=cast(Any, None),
            pinn_config=cast(Any, None),
            colloc={},
            seed=0,
            noise_level=0.0,
            scaffold_kwargs={},
            result=None,
            engine=None,
            tier_name=tier,
        )
    message = str(excinfo.value)
    assert "kdv" in message
    for supported in _SCHEMA_BACKED_PDES:
        assert supported in message, (
            f"error message must list supported PDE {supported!r}: {message!r}"
        )





@pytest.mark.unit
@pytest.mark.parametrize("pde", sorted(PDE_REGISTRY))
def test_build_configs_unaffected_for_all_registry_pdes(pde: str) -> None:
    config, pinn_config = mode2_pipeline.build_configs(
        pde=pde, tier=_any_tier(pde), seed=0
    )
    assert config.n_iterations > 0
    assert pinn_config.pretrain_epoch > 0
