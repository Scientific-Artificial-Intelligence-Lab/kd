
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = (
    _PROJECT_ROOT
    / "scripts"
    / "discover"
    / "research"
    / "analysis"
    / "task4_pinn_denoising_isolation.py"
)
_LSQ_SCRIPT_PATH = (
    _PROJECT_ROOT / "scripts" / "discover" / "research" / "analysis" / "task4_lsq_probe.py"
)

MAX_EPOCHS_BACKCOMPAT = 30_000
MAX_EPOCHS_P22 = 100_000
PRETRAIN_E22_CANONICAL = 30_000


def _load_module_from_path(path: Path, name: str) -> ModuleType:
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_isolation_module() -> ModuleType:
    return _load_module_from_path(
        _SCRIPT_PATH, "test_task4_pinn_fp_cli_isolation",
    )


def _load_lsq_module() -> ModuleType:
    return _load_module_from_path(
        _LSQ_SCRIPT_PATH, "test_task4_pinn_fp_cli_lsq",
    )


@pytest.fixture(scope="module")
def isolation() -> ModuleType:
    return _load_isolation_module()


@pytest.fixture(scope="module")
def lsq() -> ModuleType:
    return _load_lsq_module()


@pytest.mark.unit
class TestMaxEpochsDefault:

    def test_max_epochs_default_30k(self, isolation: ModuleType) -> None:
        args = isolation.parse_args([])
        assert args.max_epochs == MAX_EPOCHS_BACKCOMPAT

    def test_default_equals_module_constant(self, isolation: ModuleType) -> None:
        args = isolation.parse_args([])
        assert args.max_epochs == isolation.MAX_EPOCHS_DEFAULT
        assert isolation.MAX_EPOCHS_DEFAULT == MAX_EPOCHS_BACKCOMPAT


@pytest.mark.unit
class TestMaxEpochsOverride:

    def test_max_epochs_override_parses(self, isolation: ModuleType) -> None:
        args = isolation.parse_args(["--max-epochs", str(MAX_EPOCHS_P22)])
        assert args.max_epochs == MAX_EPOCHS_P22

    def test_max_epochs_zero_is_allowed(self, isolation: ModuleType) -> None:
        args = isolation.parse_args(["--max-epochs", "0"])
        assert args.max_epochs == 0

    def test_max_epochs_large_value(self, isolation: ModuleType) -> None:
        args = isolation.parse_args(["--max-epochs", "1000000"])
        assert args.max_epochs == 1_000_000


@pytest.mark.unit
class TestMaxEpochsValidation:

    def test_max_epochs_negative_rejected(self, isolation: ModuleType) -> None:
        with pytest.raises(SystemExit):
            isolation.parse_args(["--max-epochs", "-1"])

    def test_pretrain_epochs_negative_rejected(
        self, isolation: ModuleType
    ) -> None:
        with pytest.raises(SystemExit):
            isolation.parse_args(["--pretrain-epochs", "-5"])

    def test_eval_every_zero_rejected(self, isolation: ModuleType) -> None:
        with pytest.raises(SystemExit):
            isolation.parse_args(["--eval-every", "0"])


@pytest.mark.unit
class TestCheckpointFilename:

    def test_ckpt_filename_contains_epoch(
        self, isolation: ModuleType, tmp_path: Path
    ) -> None:
        result_json = tmp_path / "run.json"
        path = isolation.build_model_output_path(
            seed=0,
            max_epochs=MAX_EPOCHS_P22,
            explicit=None,
            result_json=result_json,
        )
        assert "epochs100000" in path.name
        assert "seed0" in path.name
        assert path.suffix == ".pt"

        path_30k = isolation.build_model_output_path(
            seed=0,
            max_epochs=MAX_EPOCHS_BACKCOMPAT,
            explicit=None,
            result_json=result_json,
        )
        assert path.name != path_30k.name

    def test_ckpt_filename_varies_by_seed(
        self, isolation: ModuleType, tmp_path: Path
    ) -> None:
        result_json = tmp_path / "run.json"
        path_s0 = isolation.build_model_output_path(
            seed=0,
            max_epochs=MAX_EPOCHS_P22,
            explicit=None,
            result_json=result_json,
        )
        path_s1 = isolation.build_model_output_path(
            seed=1,
            max_epochs=MAX_EPOCHS_P22,
            explicit=None,
            result_json=result_json,
        )
        assert path_s0.name != path_s1.name
        assert "seed0" in path_s0.name
        assert "seed1" in path_s1.name

    def test_explicit_model_output_overrides_default(
        self, isolation: ModuleType, tmp_path: Path
    ) -> None:
        explicit = tmp_path / "custom" / "my_ckpt.pt"
        path = isolation.build_model_output_path(
            seed=0,
            max_epochs=MAX_EPOCHS_P22,
            explicit=explicit,
            result_json=tmp_path / "run.json",
        )
        assert path == explicit


@pytest.mark.unit
class TestSaveModelFlag:

    def test_save_model_default_off(self, isolation: ModuleType) -> None:
        args = isolation.parse_args([])
        assert args.save_model is False

    def test_save_model_on(self, isolation: ModuleType) -> None:
        args = isolation.parse_args(["--save-model"])
        assert args.save_model is True
        assert args.max_epochs == MAX_EPOCHS_BACKCOMPAT


@pytest.mark.unit
class TestPretrainDefault:

    def test_pretrain_default_equals_e22(
        self, isolation: ModuleType,
    ) -> None:
        args = isolation.parse_args([])
        assert args.pretrain_epochs == PRETRAIN_E22_CANONICAL
        assert (
            isolation.PRETRAIN_EPOCHS_DEFAULT == PRETRAIN_E22_CANONICAL
        )


@pytest.mark.unit
class TestValidationGuards:

    def test_coef_pde_negative_rejected(
        self, isolation: ModuleType,
    ) -> None:
        with pytest.raises(SystemExit):
            isolation.parse_args(["--coef-pde", "-1.0"])

    def test_obs_ratio_above_one_rejected(
        self, isolation: ModuleType,
    ) -> None:
        with pytest.raises(SystemExit):
            isolation.parse_args(["--obs-ratio", "5"])

    def test_obs_ratio_zero_rejected(
        self, isolation: ModuleType,
    ) -> None:
        with pytest.raises(SystemExit):
            isolation.parse_args(["--obs-ratio", "0"])

    def test_deterministic_default_on(
        self, isolation: ModuleType,
    ) -> None:
        args = isolation.parse_args([])
        assert args.deterministic is True

    def test_no_deterministic_flag_turns_off(
        self, isolation: ModuleType,
    ) -> None:
        args = isolation.parse_args(["--no-deterministic"])
        assert args.deterministic is False


@pytest.mark.unit
class TestCheckpointFilenameWithPretrain:

    def test_ckpt_filename_encodes_pretrain(
        self, isolation: ModuleType, tmp_path: Path,
    ) -> None:
        result_json = tmp_path / "run.json"
        path = isolation.build_model_output_path(
            seed=0,
            max_epochs=MAX_EPOCHS_P22,
            explicit=None,
            result_json=result_json,
            pretrain_epochs=PRETRAIN_E22_CANONICAL,
        )
        assert "pre30000" in path.name
        assert "co100000" in path.name
        assert "seed0" in path.name
        assert path.suffix == ".pt"

    def test_ckpt_filename_different_pretrains_differ(
        self, isolation: ModuleType, tmp_path: Path,
    ) -> None:
        result_json = tmp_path / "run.json"
        path_30k = isolation.build_model_output_path(
            seed=0, max_epochs=MAX_EPOCHS_P22,
            explicit=None, result_json=result_json,
            pretrain_epochs=30_000,
        )
        path_150k = isolation.build_model_output_path(
            seed=0, max_epochs=MAX_EPOCHS_P22,
            explicit=None, result_json=result_json,
            pretrain_epochs=150_000,
        )
        assert path_30k.name != path_150k.name


@pytest.mark.unit
class TestAtomicSave:

    def test_atomic_save_emits_done_marker(
        self, isolation: ModuleType, tmp_path: Path,
    ) -> None:
        import torch

        payload = isolation.build_ckpt_payload(
            {"w": torch.tensor([1.0, 2.0])},
            n_layers=3, n_hidden=64, activation="sin", dtype="fp64",
            pretrain_epochs=30_000, max_epochs=100_000,
            obs_ratio=0.8, coef_pde=1.0, seed=0,
        )
        target = tmp_path / "ckpt.pt"
        isolation.atomic_save_ckpt(payload, target)
        assert target.exists()
        assert target.with_suffix(".pt.done").exists()

        assert not target.with_suffix(".pt.tmp").exists()
        loaded = torch.load(target, weights_only=True)
        assert loaded["arch"]["n_layers"] == 3
        assert loaded["arch"]["pretrain_epochs"] == 30_000


@pytest.mark.unit
class TestLsqProbeCli:

    def test_skip_train_without_load_model_raises(
        self, lsq: ModuleType,
    ) -> None:
        with pytest.raises(SystemExit):
            lsq.parse_args(["--skip-train"])

    def test_load_model_without_skip_train_raises(
        self, lsq: ModuleType, tmp_path: Path,
    ) -> None:
        fake = tmp_path / "some.pt"
        fake.write_bytes(b"")
        with pytest.raises(SystemExit):
            lsq.parse_args(["--load-model", str(fake)])

    def test_save_model_with_skip_train_raises(
        self, lsq: ModuleType, tmp_path: Path,
    ) -> None:
        fake = tmp_path / "some.pt"
        fake.write_bytes(b"")
        with pytest.raises(SystemExit):
            lsq.parse_args([
                "--skip-train", "--load-model", str(fake), "--save-model",
            ])

    def test_negative_max_epochs_rejected(self, lsq: ModuleType) -> None:
        with pytest.raises(SystemExit):
            lsq.parse_args(["--max-epochs", "-1"])

    def test_zero_chunk_size_rejected(self, lsq: ModuleType) -> None:
        with pytest.raises(SystemExit):
            lsq.parse_args(["--chunk-size", "0"])

    def test_zero_probe_points_rejected(self, lsq: ModuleType) -> None:
        with pytest.raises(SystemExit):
            lsq.parse_args(["--probe-points", "0"])

    def test_negative_coef_pde_rejected(self, lsq: ModuleType) -> None:
        with pytest.raises(SystemExit):
            lsq.parse_args(["--coef-pde", "-1"])

    def test_obs_ratio_bad_rejected(self, lsq: ModuleType) -> None:
        with pytest.raises(SystemExit):
            lsq.parse_args(["--obs-ratio", "1.5"])

    def test_decision_threshold_default(self, lsq: ModuleType) -> None:
        args = lsq.parse_args([])
        assert args.decision_threshold == [4.0, 8.0]

    def test_decision_threshold_custom_parses(
        self, lsq: ModuleType,
    ) -> None:
        args = lsq.parse_args([
            "--decision-threshold", "2.0", "9.0",
        ])
        assert args.decision_threshold == [2.0, 9.0]

    def test_decision_threshold_lower_gt_upper_rejected(
        self, lsq: ModuleType,
    ) -> None:
        with pytest.raises(SystemExit):
            lsq.parse_args([
                "--decision-threshold", "10", "5",
            ])


@pytest.mark.unit
class TestDecisionVerdict:

    def test_soft_below_lower(self, lsq: ModuleType) -> None:
        assert lsq.classify_decision_verdict(3.0) == "SOFT"

    def test_ambiguous_between(self, lsq: ModuleType) -> None:
        assert lsq.classify_decision_verdict(6.0) == "AMBIGUOUS"

    def test_converged_at_or_above_upper(self, lsq: ModuleType) -> None:
        assert lsq.classify_decision_verdict(10.0) == "CONVERGED"

    def test_boundary_lower_exclusive(self, lsq: ModuleType) -> None:
        assert lsq.classify_decision_verdict(4.0) == "AMBIGUOUS"

    def test_boundary_upper_inclusive(self, lsq: ModuleType) -> None:
        assert lsq.classify_decision_verdict(8.0) == "CONVERGED"

    def test_custom_thresholds(self, lsq: ModuleType) -> None:
        assert (
            lsq.classify_decision_verdict(
                5.0, lower_pct=3.0, upper_pct=7.0,
            )
            == "AMBIGUOUS"
        )
        assert (
            lsq.classify_decision_verdict(
                2.5, lower_pct=3.0, upper_pct=7.0,
            )
            == "SOFT"
        )
        assert (
            lsq.classify_decision_verdict(
                8.0, lower_pct=3.0, upper_pct=7.0,
            )
            == "CONVERGED"
        )


@pytest.mark.unit
class TestCkptArchCheck:

    def test_arch_match_loads(self, lsq: ModuleType, tmp_path: Path) -> None:
        import torch
        iso = lsq._ISO
        payload = iso.build_ckpt_payload(
            {"w": torch.tensor([1.0])},
            n_layers=3, n_hidden=64, activation="sin", dtype="fp64",
            pretrain_epochs=30_000, max_epochs=100_000,
            obs_ratio=0.8, coef_pde=1.0, seed=0,
        )
        target = tmp_path / "ok.pt"
        iso.atomic_save_ckpt(payload, target)
        state, arch = lsq._load_ckpt_with_arch_check(
            target,
            expected_arch={
                "n_layers": 3, "n_hidden": 64,
                "activation": "sin", "dtype": "fp64",
            },
            device=torch.device("cpu"),
        )
        assert "w" in state


        assert arch["max_epochs"] == 100_000
        assert arch["pretrain_epochs"] == 30_000

    def test_arch_mismatch_raises(
        self, lsq: ModuleType, tmp_path: Path,
    ) -> None:
        import torch
        iso = lsq._ISO
        payload = iso.build_ckpt_payload(
            {"w": torch.tensor([1.0])},
            n_layers=3, n_hidden=64, activation="sin", dtype="fp64",
            pretrain_epochs=30_000, max_epochs=100_000,
            obs_ratio=0.8, coef_pde=1.0, seed=0,
        )
        target = tmp_path / "bad.pt"
        iso.atomic_save_ckpt(payload, target)
        with pytest.raises(SystemExit):
            lsq._load_ckpt_with_arch_check(
                target,
                expected_arch={
                    "n_layers": 3, "n_hidden": 64,
                    "activation": "tanh",
                    "dtype": "fp64",
                },
                device=torch.device("cpu"),
            )

    def test_build_commit_hash_present_in_payload(
        self, lsq: ModuleType,
    ) -> None:
        import torch
        iso = lsq._ISO
        payload = iso.build_ckpt_payload(
            {"w": torch.tensor([1.0])},
            n_layers=3, n_hidden=64, activation="sin", dtype="fp64",
            pretrain_epochs=30_000, max_epochs=100_000,
            obs_ratio=0.8, coef_pde=1.0, seed=0,
        )
        assert "build_commit_hash" in payload["arch"]


        val = payload["arch"]["build_commit_hash"]
        assert val is None or isinstance(val, str)

    def test_commit_hash_mismatch_warns_not_raises(
        self, lsq: ModuleType, tmp_path: Path, caplog: Any,
    ) -> None:
        import logging

        import torch
        iso = lsq._ISO
        payload = iso.build_ckpt_payload(
            {"w": torch.tensor([1.0])},
            n_layers=3, n_hidden=64, activation="sin", dtype="fp64",
            pretrain_epochs=30_000, max_epochs=100_000,
            obs_ratio=0.8, coef_pde=1.0, seed=0,
        )

        payload["arch"]["build_commit_hash"] = "deadbeef" * 5
        target = tmp_path / "commit_mismatch.pt"
        iso.atomic_save_ckpt(payload, target)
        with caplog.at_level(logging.WARNING, logger="task4_lsq_probe"):
            state, _arch = lsq._load_ckpt_with_arch_check(
                target,
                expected_arch={
                    "n_layers": 3, "n_hidden": 64,
                    "activation": "sin", "dtype": "fp64",
                },
                device=torch.device("cpu"),
                expected_commit_hash="cafef00d" * 5,
            )

        assert "foreign worktree" in caplog.text.lower()
        assert "w" in state

    def test_legacy_raw_state_dict_returns_empty_arch(
        self, lsq: ModuleType, tmp_path: Path,
    ) -> None:
        import torch
        target = tmp_path / "legacy.pt"
        torch.save({"w": torch.tensor([1.0])}, target)
        state, arch = lsq._load_ckpt_with_arch_check(
            target,
            expected_arch={
                "n_layers": 3, "n_hidden": 64,
                "activation": "sin", "dtype": "fp64",
            },
            device=torch.device("cpu"),
        )
        assert "w" in state
        assert arch == {}


@pytest.mark.unit
class TestCkptTrainingMetadataOverride:

    def _make_args(
        self, max_epochs: int, pretrain_epochs: int,
    ) -> Any:
        import argparse
        return argparse.Namespace(
            max_epochs=max_epochs,
            pretrain_epochs=pretrain_epochs,
        )

    def test_override_replaces_cli_defaults_with_ckpt_values(
        self, lsq: ModuleType,
    ) -> None:
        args = self._make_args(max_epochs=150_000, pretrain_epochs=150_000)
        ckpt_arch = {
            "max_epochs": 30_000,
            "pretrain_epochs": 30_000,
            "n_layers": 3, "n_hidden": 64, "activation": "sin",
        }
        lsq._apply_ckpt_training_metadata(args, ckpt_arch)
        assert args.max_epochs == 30_000
        assert args.pretrain_epochs == 30_000

    def test_override_emits_warning_on_value_change(
        self, lsq: ModuleType, caplog: Any,
    ) -> None:
        import logging
        args = self._make_args(max_epochs=150_000, pretrain_epochs=150_000)
        ckpt_arch = {"max_epochs": 50_000, "pretrain_epochs": 30_000}
        with caplog.at_level(logging.WARNING, logger="task4_lsq_probe"):
            lsq._apply_ckpt_training_metadata(args, ckpt_arch)

        assert "max_epochs" in caplog.text
        assert "pretrain_epochs" in caplog.text
        assert "150000" in caplog.text
        assert "50000" in caplog.text

    def test_override_silent_when_values_already_match(
        self, lsq: ModuleType, caplog: Any,
    ) -> None:
        import logging
        args = self._make_args(max_epochs=30_000, pretrain_epochs=30_000)
        ckpt_arch = {"max_epochs": 30_000, "pretrain_epochs": 30_000}
        with caplog.at_level(logging.WARNING, logger="task4_lsq_probe"):
            lsq._apply_ckpt_training_metadata(args, ckpt_arch)
        assert "Overriding CLI" not in caplog.text
        assert args.max_epochs == 30_000
        assert args.pretrain_epochs == 30_000

    def test_legacy_empty_arch_warns_and_leaves_args_intact(
        self, lsq: ModuleType, caplog: Any,
    ) -> None:
        import logging
        args = self._make_args(max_epochs=100_000, pretrain_epochs=50_000)
        with caplog.at_level(logging.WARNING, logger="task4_lsq_probe"):
            lsq._apply_ckpt_training_metadata(args, {})
        assert "no arch metadata" in caplog.text
        assert args.max_epochs == 100_000
        assert args.pretrain_epochs == 50_000

    def test_partial_arch_warns_per_missing_key(
        self, lsq: ModuleType, caplog: Any,
    ) -> None:
        import logging
        args = self._make_args(max_epochs=150_000, pretrain_epochs=150_000)

        ckpt_arch = {"max_epochs": 30_000, "n_layers": 3}
        with caplog.at_level(logging.WARNING, logger="task4_lsq_probe"):
            lsq._apply_ckpt_training_metadata(args, ckpt_arch)
        assert args.max_epochs == 30_000
        assert args.pretrain_epochs == 150_000
        assert "pretrain_epochs" in caplog.text


@pytest.mark.unit
class TestDecisionVerdictV2:

    def test_v2_soft_requires_tight_p75(
        self, lsq: ModuleType,
    ) -> None:
        verdict = lsq.classify_decision_verdict(
            3.0, l1_rel_p75_pct=5.0, l1_rel_max_pct=6.0,
        )
        assert verdict == "SOFT"

    def test_v2_soft_denied_when_p75_wide(
        self, lsq: ModuleType,
    ) -> None:
        verdict = lsq.classify_decision_verdict(
            3.0, l1_rel_p75_pct=9.0, l1_rel_max_pct=10.0,
        )




        assert verdict == "AMBIGUOUS"

    def test_v2_converged_via_max_outlier(
        self, lsq: ModuleType,
    ) -> None:
        verdict = lsq.classify_decision_verdict(
            2.0, l1_rel_p75_pct=3.0, l1_rel_max_pct=20.0,
        )
        assert verdict == "CONVERGED"

    def test_v2_converged_via_median(
        self, lsq: ModuleType,
    ) -> None:
        verdict = lsq.classify_decision_verdict(
            9.0, l1_rel_p75_pct=9.5, l1_rel_max_pct=10.0,
        )
        assert verdict == "CONVERGED"

    def test_v2_ambiguous_middle_band(
        self, lsq: ModuleType,
    ) -> None:
        verdict = lsq.classify_decision_verdict(
            6.0, l1_rel_p75_pct=7.0, l1_rel_max_pct=8.0,
        )
        assert verdict == "AMBIGUOUS"

    def test_v1_fallback_when_p75_missing(
        self, lsq: ModuleType,
    ) -> None:
        assert lsq.classify_decision_verdict(3.0) == "SOFT"
        assert lsq.classify_decision_verdict(6.0) == "AMBIGUOUS"
        assert lsq.classify_decision_verdict(10.0) == "CONVERGED"


@pytest.mark.unit
class TestP22AggregateScript:

    @pytest.fixture(scope="class")
    def aggregate_module(self) -> ModuleType:
        path = (
            _PROJECT_ROOT / "scripts" / "discover" / "research" / "analysis"
            / "task4_p22_aggregate.py"
        )
        return _load_module_from_path(
            path, "test_task4_p22_aggregate",
        )

    def _write_run_json(
        self,
        path: Path,
        *,
        seed: int,
        max_epochs: int,
        median_pct: float,
        p75_pct: float,
        max_pct: float,
        verdict: str,
    ) -> None:
        import json
        payload = {
            "seed": seed,
            "config": {
                "max_epochs": max_epochs,
                "pretrain_epochs": 30_000,
            },
            "lsq_l1_rel_mean_pct": (median_pct + p75_pct) / 2,
            "lsq_l1_rel_median_pct": median_pct,
            "lsq_l1_rel_p75_pct": p75_pct,
            "lsq_l1_rel_max_pct": max_pct,
            "decision_verdict": verdict,
            "decision_hint": None,
            "probe": {"lsq": {
                "l1_rel_per_term": [
                    median_pct / 100,
                    median_pct / 100,
                    p75_pct / 100,
                    max_pct / 100,
                ],
            }},
        }
        path.write_text(json.dumps(payload, indent=2))

    def test_aggregate_groups_by_epoch(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:
        for i, seed in enumerate([0, 1, 2]):
            self._write_run_json(
                tmp_path / f"lsq_seed{seed}_p22_30k.json",
                seed=seed, max_epochs=30_000,
                median_pct=6.0 + i, p75_pct=7.0 + i, max_pct=8.0 + i,
                verdict="AMBIGUOUS",
            )
        result = aggregate_module.aggregate(
            sorted(tmp_path.glob("lsq_seed*.json")),
        )
        assert result["n_runs"] == 3
        assert result["epochs_sorted"] == [30_000]
        bucket = result["per_epoch"]["30000"]
        assert bucket["n_seeds"] == 3
        assert bucket["seeds"] == [0, 1, 2]
        assert bucket["group_verdict"] == "AMBIGUOUS"

    def test_aggregate_plateau_detection(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:

        for seed in [0, 1, 2]:
            self._write_run_json(
                tmp_path / f"lsq_seed{seed}_p22_30k.json",
                seed=seed, max_epochs=30_000,
                median_pct=6.0, p75_pct=7.0, max_pct=8.0,
                verdict="AMBIGUOUS",
            )
            self._write_run_json(
                tmp_path / f"lsq_seed{seed}_p22_100k.json",
                seed=seed, max_epochs=100_000,
                median_pct=5.9, p75_pct=6.9, max_pct=7.9,
                verdict="AMBIGUOUS",
            )
        result = aggregate_module.aggregate(
            sorted(tmp_path.glob("lsq_seed*.json")),
        )
        assert result["epochs_sorted"] == [30_000, 100_000]
        plateau_flags = result["plateau"]
        assert len(plateau_flags) == 1
        assert plateau_flags[0]["prev_epoch"] == 30_000
        assert plateau_flags[0]["epoch"] == 100_000
        assert plateau_flags[0]["is_plateau"] is True

    def test_aggregate_recommendation_upgrade_on_ambiguous(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:
        for seed in [0, 1, 2]:
            self._write_run_json(
                tmp_path / f"lsq_seed{seed}_p22_100k.json",
                seed=seed, max_epochs=100_000,
                median_pct=6.0, p75_pct=7.0, max_pct=8.0,
                verdict="AMBIGUOUS",
            )
        result = aggregate_module.aggregate(
            sorted(tmp_path.glob("lsq_seed*.json")),
        )
        rec = result["recommendation"]
        assert rec["verdict"] == "UPGRADE_SEEDS"
        assert "10 seeds" in rec["rationale"]

    def test_aggregate_recommendation_done(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:
        for seed in [0, 1, 2]:

            self._write_run_json(
                tmp_path / f"lsq_seed{seed}_p22_30k.json",
                seed=seed, max_epochs=30_000,
                median_pct=9.0, p75_pct=9.5, max_pct=10.0,
                verdict="CONVERGED",
            )

            self._write_run_json(
                tmp_path / f"lsq_seed{seed}_p22_100k.json",
                seed=seed, max_epochs=100_000,
                median_pct=8.95, p75_pct=9.4, max_pct=9.9,
                verdict="CONVERGED",
            )
        result = aggregate_module.aggregate(
            sorted(tmp_path.glob("lsq_seed*.json")),
        )
        rec = result["recommendation"]
        assert rec["verdict"] == "DONE"

    def test_aggregate_empty_glob(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:
        result = aggregate_module.aggregate([])
        assert result["n_runs"] == 0
        assert result["epochs_sorted"] == []
        assert result["recommendation"]["verdict"] == "NO_DATA"

    def test_aggregate_malformed_json_skipped(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not-json")
        good = tmp_path / "lsq_seed0_p22_30k.json"
        self._write_run_json(
            good, seed=0, max_epochs=30_000,
            median_pct=3.0, p75_pct=4.0, max_pct=5.0,
            verdict="SOFT",
        )
        result = aggregate_module.aggregate([bad, good])
        assert result["n_runs"] == 1
        assert result["epochs_sorted"] == [30_000]

    def test_aggregate_v1_json_backcompat(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:
        import json
        v1_payload = {
            "seed": 0,
            "config": {"max_epochs": 30_000, "pretrain_epochs": 30_000},
            "lsq_l1_rel_mean_pct": 5.5,

            "decision_verdict": "AMBIGUOUS",
            "probe": {"lsq": {
                "l1_rel_per_term": [0.02, 0.03, 0.05, 0.12],
            }},
        }
        target = tmp_path / "lsq_seed0_p22_30k.json"
        target.write_text(json.dumps(v1_payload))
        result = aggregate_module.aggregate([target])
        bucket = result["per_epoch"]["30000"]

        assert abs(
            bucket["summary_median"]["median"] - 4.0
        ) < 0.01

    def test_aggregate_cli_emits_json(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:
        import json
        for seed in [0, 1]:
            self._write_run_json(
                tmp_path / f"lsq_seed{seed}_p22_30k.json",
                seed=seed, max_epochs=30_000,
                median_pct=3.0, p75_pct=4.0, max_pct=5.0,
                verdict="SOFT",
            )
        out_path = tmp_path / "aggregate.json"
        aggregate_module.main([
            "--input-glob", str(tmp_path / "lsq_seed*.json"),
            "--output", str(out_path),
        ])
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["n_runs"] == 2
        assert loaded["epochs_sorted"] == [30_000]

    def test_filename_tag_overrides_poisoned_config(
        self, aggregate_module: ModuleType, tmp_path: Path, caplog: Any,
    ) -> None:
        import logging


        real_epochs = [30_000, 50_000, 75_000, 100_000, 150_000]
        for ep in real_epochs:
            ep_k = ep // 1_000
            self._write_run_json(
                tmp_path / f"lsq_seed0_p22_{ep_k}k.json",
                seed=0, max_epochs=150_000,
                median_pct=1.0, p75_pct=2.0, max_pct=3.0,
                verdict="SOFT",
            )
        with caplog.at_level(
            logging.WARNING, logger="task4_p22_aggregate",
        ):
            result = aggregate_module.aggregate(
                sorted(tmp_path.glob("lsq_seed*.json")),
            )

        assert result["epochs_sorted"] == real_epochs

        assert "filename tag says" in caplog.text

        assert result["grouping_source_counts"] == {"filename_tag": 5}

    def test_no_filename_tag_falls_back_to_config(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:
        self._write_run_json(
            tmp_path / "lsq_seed0.json",
            seed=0, max_epochs=30_000,
            median_pct=3.0, p75_pct=4.0, max_pct=5.0,
            verdict="SOFT",
        )
        result = aggregate_module.aggregate([tmp_path / "lsq_seed0.json"])
        assert result["epochs_sorted"] == [30_000]
        assert result["grouping_source_counts"] == {"config_max_epochs": 1}

    def test_extract_epoch_from_filename_variants(
        self, aggregate_module: ModuleType,
    ) -> None:
        assert (
            aggregate_module._extract_epoch_from_filename(
                Path("lsq_probe_seed0_p22_30k.json"),
            ) == 30_000
        )
        assert (
            aggregate_module._extract_epoch_from_filename(
                Path("lsq_probe_seed4_p22_150k.json"),
            ) == 150_000
        )

        assert (
            aggregate_module._extract_epoch_from_filename(
                Path("lsq_probe_seed0.json"),
            ) is None
        )

        assert (
            aggregate_module._extract_epoch_from_filename(
                Path("lsq_probe_seed0_p22_30.json"),
            ) is None
        )

    def test_mixed_sources_reported_in_aggregate(
        self, aggregate_module: ModuleType, tmp_path: Path,
    ) -> None:
        self._write_run_json(
            tmp_path / "lsq_seed0_p22_30k.json",
            seed=0, max_epochs=30_000,
            median_pct=3.0, p75_pct=4.0, max_pct=5.0, verdict="SOFT",
        )
        self._write_run_json(
            tmp_path / "lsq_seed1.json",
            seed=1, max_epochs=30_000,
            median_pct=3.0, p75_pct=4.0, max_pct=5.0, verdict="SOFT",
        )
        result = aggregate_module.aggregate([
            tmp_path / "lsq_seed0_p22_30k.json",
            tmp_path / "lsq_seed1.json",
        ])
        assert result["grouping_source_counts"] == {
            "filename_tag": 1, "config_max_epochs": 1,
        }


        assert result["epochs_sorted"] == [30_000]
        assert result["per_epoch"]["30000"]["n_seeds"] == 2

    def test_render_plateau_plot_skippable_without_matplotlib(
        self, aggregate_module: ModuleType, tmp_path: Path,
        monkeypatch: Any,
    ) -> None:

        monkeypatch.setattr(
            aggregate_module, "_try_import_matplotlib",
            lambda: None,
        )
        dummy = {
            "epochs_sorted": [30_000],
            "per_epoch": {"30000": {
                "n_seeds": 3, "group_verdict": "SOFT",
                "summary_median": {"median": 3.0, "mean": 3.0, "std": 0.1},
                "summary_p75": {"median": 4.0, "mean": 4.0, "std": 0.1},
                "summary_max": {"median": 5.0, "mean": 5.0, "std": 0.1},
                "summary_mean": {"median": 3.5, "mean": 3.5, "std": 0.1},
            }},
        }
        ok = aggregate_module.render_plateau_plot(
            dummy, tmp_path / "p.png",
        )
        assert ok is False
