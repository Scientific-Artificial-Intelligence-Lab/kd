
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
import torch

from kd.search.discover.candidates import CandidateSnapshot
from kd.search.discover.config import DiscoverConfig, PINNConfig
from kd.search.discover.runners.mode2_payload import (
    _build_burgers_config_payload,
    _build_burgers_diagnostics,
    _build_chafee_config_payload,
    assemble_payload,
)
from kd.search.discover.runners.mode2_pipeline import (
    OUTPUT_DIR,
    _format_mode2_start,
    build_configs,
    save_payload,
)
from kd.search.discover.runners.pde_registry import PDE_REGISTRY

_TF1_MODE2_ENTROPY_GAMMA = 0.7
_BURGERS_ALIGNED_CYCLE_N_ITERATIONS = 20
_BURGERS_ALIGNED_MAX_LENGTH = 256
_BURGERS_ALIGNED_STABILITY_SELECTION = 3
_DEFAULT_MAX_LENGTH = 30


@pytest.mark.unit
def test_format_mode2_start_reports_honest_total_budget() -> None:
    msg = _format_mode2_start(
        pde_name="burgers",
        n_cycles=3,
        total_search_iter=80,
        pretrain_epoch=20000,
        pinn_epoch=1000,
    )
    assert "pde=burgers" in msg
    assert "4 search passes" in msg
    assert "3 cycles + 1 final" in msg
    assert "80 total search iter" in msg
    assert "pretrain=20000" in msg
    assert "pinn=1000" in msg





@pytest.mark.unit
class TestBurgersBuildConfigs:

    def test_returns_pair(self) -> None:
        result = build_configs(pde="burgers", tier="aligned", seed=42)
        assert isinstance(result, tuple) and len(result) == 2
        config, pinn_config = result
        assert isinstance(config, DiscoverConfig)
        assert isinstance(pinn_config, PINNConfig)

    def test_aligned_max_length_is_256(self) -> None:
        config, _ = build_configs(pde="burgers", tier="aligned", seed=42)
        assert config.max_length == _BURGERS_ALIGNED_MAX_LENGTH

    def test_aligned_attention_true(self) -> None:
        config, _ = build_configs(pde="burgers", tier="aligned", seed=42)
        assert config.attention is True

    def test_aligned_stability_selection_three(self) -> None:
        config, _ = build_configs(pde="burgers", tier="aligned", seed=42)
        assert config.stability_selection == _BURGERS_ALIGNED_STABILITY_SELECTION

    def test_aligned_pinn_cycle_n_iterations_is_20(self) -> None:
        _, pinn_config = build_configs(pde="burgers", tier="aligned", seed=42)
        assert (
            pinn_config.cycle_n_iterations
            == _BURGERS_ALIGNED_CYCLE_N_ITERATIONS
        )


@pytest.mark.unit
@pytest.mark.parametrize("tier", ["fast", "medium", "full"])
class TestBurgersDefaultBuildConfigs:

    def test_max_length_is_30(self, tier: str) -> None:
        config, _ = build_configs(pde="burgers", tier=tier, seed=42)
        assert config.max_length == _DEFAULT_MAX_LENGTH

    def test_attention_disabled(self, tier: str) -> None:
        config, _ = build_configs(pde="burgers", tier=tier, seed=42)
        assert config.attention is False

    def test_stability_selection_disabled(self, tier: str) -> None:
        config, _ = build_configs(pde="burgers", tier=tier, seed=42)
        assert config.stability_selection == 0

    def test_pinn_cycle_n_iterations_is_none(self, tier: str) -> None:
        _, pinn_config = build_configs(pde="burgers", tier=tier, seed=42)
        assert pinn_config.cycle_n_iterations is None


@pytest.mark.unit
@pytest.mark.parametrize("pde", ["burgers", "chafee"])
@pytest.mark.parametrize("tier", ["fast", "medium", "full", "aligned"])
def test_entropy_gamma_threads_through_build_configs(
    pde: str, tier: str,
) -> None:
    config, _ = build_configs(pde=pde, tier=tier, seed=42)
    assert config.entropy_gamma == _TF1_MODE2_ENTROPY_GAMMA





@pytest.mark.unit
def test_burgers_config_payload_records_aligned_sensitive_fields() -> None:
    spec = PDE_REGISTRY["burgers"]
    settings = spec.presets["aligned"]
    config, pinn_config = build_configs(
        pde="burgers", tier="aligned", seed=0,
    )
    colloc = {
        "x": torch.zeros(7, dtype=torch.float32),
        "t": torch.zeros(7, dtype=torch.float32),
    }

    payload = _build_burgers_config_payload(
        settings=settings, config=config, pinn_config=pinn_config,
        colloc=colloc, seed=0, scaffold_kwargs={},
    )

    assert payload["data_path"].endswith("burgers2.mat")
    assert payload["operators"] == list(settings.operators)
    assert payload["max_length"] == _BURGERS_ALIGNED_MAX_LENGTH
    assert payload["attention"] is True
    assert payload["stability_selection"] == _BURGERS_ALIGNED_STABILITY_SELECTION
    assert payload["stability_seed"] == 0
    assert (
        payload["cycle_n_iterations"]
        == _BURGERS_ALIGNED_CYCLE_N_ITERATIONS
    )
    assert payload["collocation_cut_ratio"] == 0.0
    assert payload["n_collocation_requested"] == pinn_config.n_collocation
    assert payload["n_collocation_actual"] == 7


@pytest.mark.unit
def test_burgers_diagnostics_preserve_final_selection_context() -> None:
    final_state = type(
        "FinalState",
        (),
        {
            "extras": {
                "stability_selection": {
                    "ran": True,
                    "pre_filter": {"expression": "prefilter", "reward": 0.9},
                    "selected": {"expression": "selected", "reward": 0.8},
                    "vote_counts": [1, 3],
                    "error": None,
                },
            },
        },
    )()
    candidates = [
        CandidateSnapshot(
            expression="selected",
            reward=0.8,
            nmse=0.1,
            n_nodes=3,
            terms=["diff2_x(u)", "mul(diff_x(u),u)"],
        ),
    ]

    diagnostics = _build_burgers_diagnostics(
        final_state=final_state, candidates=candidates,
        global_best_expression="global_best", global_best_reward=0.95,
    )

    assert diagnostics["stability_selection"]["selected"]["expression"] == (
        "selected"
    )
    assert diagnostics["global_best"] == {
        "expression": "global_best",
        "reward": 0.95,
    }
    assert diagnostics["final_cycle_candidates"] == [
        {
            "expression": "selected",
            "reward": 0.8,
            "nmse": 0.1,
            "n_nodes": 3,
            "terms": ["diff2_x(u)", "mul(diff_x(u),u)"],
        },
    ]


@pytest.mark.unit
def test_chafee_config_payload_includes_paper_fields() -> None:
    spec = PDE_REGISTRY["chafee"]
    settings = spec.presets["aligned"]
    config, pinn_config = build_configs(
        pde="chafee", tier="aligned", seed=0,
    )
    payload = _build_chafee_config_payload(
        settings=settings, config=config, pinn_config=pinn_config,
        scaffold_kwargs={},
    )
    assert payload["attn_length"] == 20
    assert payload["soft_length_loc"] == 10.0
    assert payload["coef_pde"] == 1.0
    assert payload["cycle_n_iterations"] == 20
    assert payload["n_cycles"] == 2
    assert payload["operators"] == list(settings.operators)





@pytest.mark.unit
def test_save_payload_uses_default_naming(tmp_path) -> None:
    spec = PDE_REGISTRY["burgers"]
    payload = {"tier": "fast", "seed": 7}
    out_path = save_payload(
        spec, "fast", 7, payload, output_dir=tmp_path,
    )
    assert out_path.name == "mode2_burgers_fast_seed7.json"
    assert out_path.exists()


@pytest.mark.unit
def test_default_output_dir_is_legacy_baseline_results() -> None:
    assert OUTPUT_DIR.name == "results"
    assert OUTPUT_DIR.parent.name == "baseline"





@pytest.mark.unit
def test_run_pipeline_threads_noise_scale_to_loader(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _stub_add_gaussian_noise(
        dataset: object, level: float, seed: int, *, scale: str = "std",
    ) -> NoReturn:
        captured["scale"] = scale
        captured["level"] = level
        captured["seed"] = seed


        raise StopIteration("captured")

    import kd.search.discover.runners.mode2_pipeline as pipeline

    real_imports = pipeline._import_pipeline_deps

    def _patched_imports() -> dict[str, object]:
        deps = real_imports()
        deps["add_gaussian_noise"] = _stub_add_gaussian_noise
        return deps

    monkeypatch.setattr(pipeline, "_import_pipeline_deps", _patched_imports)

    with pytest.raises(StopIteration):
        pipeline.run_pipeline(
            pde="burgers", tier="fast", seed=42,
            noise_scale="max", device="cpu",
        )
    assert captured["scale"] == "max"
    assert captured["level"] == 0.5
    assert captured["seed"] == 42




_TD091_BACKUPS_DIR = (
    Path.home() / "PhD" / "project" / "discover-next-backups"
    / "td091-cycle20-2026-04-26" / "json"
)





_A1_DROPPED_CONFIG_KEYS = frozenset({
    "diagnostic_scaffold",
    "diagnostic_scaffold_diffusion_tokens",
    "diagnostic_scaffold_reaction_tokens",
    "diagnostic_scaffold_root_tokens",
    "diagnostic_scaffold_neutral_tokens",
})




_POST_V1SHIP_ADDED_CONFIG_KEYS = frozenset({
    "coef_pde",
})


_BURGERS_BACKUP_TOP_KEYS = frozenset({
    "tier", "seed", "noise_level", "config", "result", "diagnostics",
    "pretrain", "cycle_metrics", "elapsed_seconds",
})


def _load_burgers_backup(seed: int) -> dict[str, Any]:
    path = _TD091_BACKUPS_DIR / f"mode2_burgers_aligned_seed{seed}.json"
    with path.open() as fh:
        return cast("dict[str, Any]", json.load(fh))


def _build_mock_pretrain_result() -> Any:
    return type(
        "PretrainResult",
        (),
        {
            "train_loss": 0.5,
            "val_loss": 0.6,
            "epochs_run": 100,
            "stopped_early": False,
        },
    )()


def _build_mock_final_state(*, with_extras: bool) -> Any:
    extras = {
        "stability_selection": {
            "ran": True,
            "selected": {"expression": "mock", "reward": 0.5},
        },
    } if with_extras else {}
    return type(
        "FinalState",
        (),
        {
            "best_reward": 0.5,
            "best_expression": "mock_expression",
            "best_result_terms": ["mock_term"],
            "best_result_coefficients": [1.0],
            "extras": extras,
        },
    )()


def _build_mock_engine() -> Any:
    return type(
        "Engine",
        (),
        {
            "cycle_top_candidates": [],
            "best_expression": "mock_expression",
            "best_reward": 0.5,
        },
    )()


def _build_mock_result(*, with_extras: bool) -> Any:
    return type(
        "RunResult",
        (),
        {
            "pretrain_result": _build_mock_pretrain_result(),
            "final_state": _build_mock_final_state(with_extras=with_extras),
            "cycle_metrics": [{"cycle": 0, "reward": 0.5}],
        },
    )()


@pytest.mark.unit
@pytest.mark.skipif(
    not _TD091_BACKUPS_DIR.exists(),
    reason=" backups not present; skipping schema regression",
)
class TestPayloadSchemaRegression:

    def test_burgers_aligned_top_level_keys_match_backup(self) -> None:
        backup = _load_burgers_backup(42)
        assert set(backup.keys()) == _BURGERS_BACKUP_TOP_KEYS, (
            f"backup top-level drift: got {set(backup.keys())!r}"
        )

        spec = PDE_REGISTRY["burgers"]
        settings = spec.presets["aligned"]
        config, pinn_config = build_configs(
            pde="burgers", tier="aligned", seed=42,
        )
        payload = assemble_payload(
            spec=spec, settings=settings, config=config,
            pinn_config=pinn_config,
            colloc={
                "x": torch.zeros(7, dtype=torch.float32),
                "t": torch.zeros(7, dtype=torch.float32),
            },
            seed=42, noise_level=0.5, scaffold_kwargs={},
            result=_build_mock_result(with_extras=True),
            engine=_build_mock_engine(), tier_name="aligned",
        )


        modern_keys = set(payload.keys()) | {"elapsed_seconds"}
        assert modern_keys == _BURGERS_BACKUP_TOP_KEYS, (
            f"modern payload top-level keys differ: payload={set(payload)!r}"
        )

    def test_burgers_aligned_config_keys_subset_of_backup(self) -> None:
        backup = _load_burgers_backup(42)
        spec = PDE_REGISTRY["burgers"]
        settings = spec.presets["aligned"]
        config, pinn_config = build_configs(
            pde="burgers", tier="aligned", seed=42,
        )
        payload = assemble_payload(
            spec=spec, settings=settings, config=config,
            pinn_config=pinn_config,
            colloc={
                "x": torch.zeros(7, dtype=torch.float32),
                "t": torch.zeros(7, dtype=torch.float32),
            },
            seed=42, noise_level=0.5, scaffold_kwargs={},
            result=_build_mock_result(with_extras=True),
            engine=_build_mock_engine(), tier_name="aligned",
        )
        modern_cfg_keys = set(payload["config"].keys())
        backup_cfg_keys = set(backup["config"].keys())



        unexpected_new = (
            modern_cfg_keys - backup_cfg_keys - _POST_V1SHIP_ADDED_CONFIG_KEYS
        )
        assert not unexpected_new, (
            f"modern payload introduced unrecognised config keys: "
            f"{unexpected_new!r}"
        )

        missing_from_modern = backup_cfg_keys - modern_cfg_keys
        unexpected_missing = missing_from_modern - _A1_DROPPED_CONFIG_KEYS
        assert not unexpected_missing, (
            f"modern payload dropped unexpected backup config keys: "
            f"{unexpected_missing!r}; only A1-token keys may drop"
        )

        assert payload["config"]["data_path"].endswith("burgers2.mat")
