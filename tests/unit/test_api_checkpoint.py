
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

from kd.api import Model
from kd.search.callbacks import CHECKPOINT_VERSION, CheckpointCallback
from kd.search.discover import DiscoverConfig

if TYPE_CHECKING:
    from kd.core.evaluator import EvaluationResult
    from kd.data.schema import PDEDataset
    from kd.search.protocol import SearchAlgorithm





_NX = 32
_NT = 16
_NU = 0.1
_FAST_GENERATIONS = 3
_FAST_POPULATION = 5


_FINAL_NAME = "checkpoint_final.pt"


def _iter_name(iteration: int) -> str:
    return f"checkpoint_{iteration:06d}.pt"


@pytest.fixture
def small_burgers_dataset() -> PDEDataset:
    from kd.data.synthetic import generate_burgers_data

    return generate_burgers_data(nx=_NX, nt=_NT, nu=_NU, seed=0)


def _fast_model(**overrides: Any) -> Model:
    kwargs: dict[str, Any] = {
        "algorithm": "sga",
        "generations": _FAST_GENERATIONS,
        "population": _FAST_POPULATION,
        "depth": 3,
        "width": 3,
        "seed": 0,
        "verbose": False,
    }
    kwargs.update(overrides)
    return Model(**kwargs)


def _pt_files(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("*.pt")}


class _StartStateProbe:

    def __init__(self) -> None:
        self.start_score: float | None = None
        self.start_expression: str | None = None
        self.start_state: dict[str, Any] | None = None

    @property
    def should_stop(self) -> bool:
        return False

    def on_experiment_start(self, algorithm: SearchAlgorithm) -> None:
        self.start_score = algorithm.best_score
        self.start_expression = algorithm.best_expression
        self.start_state = dict(algorithm.state)

    def on_iteration_start(self, iteration: int, algorithm: SearchAlgorithm) -> None:
        pass

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: SearchAlgorithm,
        candidates: list[str],
        results: list[EvaluationResult],
    ) -> None:
        pass

    def on_experiment_end(self, algorithm: SearchAlgorithm) -> None:
        pass


def _fit_sga_with_checkpoints(
    dataset: PDEDataset, directory: Path, **overrides: Any
) -> Model:
    m = _fast_model(checkpoint_dir=directory, **overrides)
    m.fit(dataset)
    return m


def _assert_manifest_honesty(ckpt_dir: Path, algorithm: str) -> None:
    from kd.search.checkpoint_manifest import load_checkpoint_manifest

    entries = load_checkpoint_manifest(ckpt_dir)
    assert entries, "expected at least one manifest entry"
    for entry in entries:
        assert entry.algorithm == algorithm
        assert type(entry.seed) is int
        assert entry.config_hash is not None
        assert len(entry.config_hash) == 64
        assert all(ch in "0123456789abcdef" for ch in entry.config_hash)







class TestCheckpointDirWritesFiles:

    def test_a_every_1_writes_all_iteration_files_and_final(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        _fit_sga_with_checkpoints(small_burgers_dataset, ckpt_dir, checkpoint_every=1)

        expected = {_iter_name(i) for i in range(_FAST_GENERATIONS)} | {_FINAL_NAME}
        assert _pt_files(ckpt_dir) == expected

    def test_a_every_2_spaces_iteration_files(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        _fit_sga_with_checkpoints(small_burgers_dataset, ckpt_dir, checkpoint_every=2)

        expected = {_iter_name(0), _iter_name(2), _FINAL_NAME}
        assert _pt_files(ckpt_dir) == expected

    def test_a_default_every_resolves_to_10(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        _fit_sga_with_checkpoints(small_burgers_dataset, ckpt_dir)

        expected = {_iter_name(0), _FINAL_NAME}
        assert _pt_files(ckpt_dir) == expected

    def test_a_accepts_str_directory(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpts-str"
        _fit_sga_with_checkpoints(small_burgers_dataset, Path(str(ckpt_dir)))
        m = _fast_model(checkpoint_dir=str(tmp_path / "ckpts-str2"))
        m.fit(small_burgers_dataset)
        assert _FINAL_NAME in _pt_files(tmp_path / "ckpts-str2")

    def test_a_final_payload_schema_with_algorithm_key(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        m = _fit_sga_with_checkpoints(
            small_burgers_dataset, ckpt_dir, checkpoint_every=1
        )

        payload = torch.load(ckpt_dir / _FINAL_NAME, weights_only=False)
        assert isinstance(payload, dict)
        required = {
            "version",
            "iteration",
            "algorithm_state",
            "best_score",
            "best_expression",
            "algorithm",
        }
        assert required.issubset(payload.keys()), (
            f"missing keys: {required - set(payload.keys())}"
        )
        assert payload["version"] == CHECKPOINT_VERSION
        assert payload["algorithm"] == "sga"
        assert isinstance(payload["algorithm_state"], dict)

        assert payload["iteration"] == _FAST_GENERATIONS - 1

        assert payload["best_expression"] == m.best_expr_
        assert payload["best_score"] == pytest.approx(m.best_score_)

        _assert_manifest_honesty(ckpt_dir, "sga")







class TestCheckpointParamValidation:

    def test_b_every_without_dir_raises_value_error(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _fast_model(checkpoint_every=5)
        msg = str(exc_info.value)
        assert "checkpoint_every" in msg
        assert "checkpoint_dir" in msg

    def test_b_explicit_default_every_without_dir_also_raises(self) -> None:
        with pytest.raises(ValueError):
            _fast_model(checkpoint_every=10)

    @pytest.mark.parametrize("bad_every", [0, -3])
    def test_b_nonpositive_every_raises_at_construction(
        self, bad_every: int, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            _fast_model(checkpoint_dir=tmp_path, checkpoint_every=bad_every)
        assert ">= 1" in str(exc_info.value)

    def test_b_empty_string_dir_raises_at_construction(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _fast_model(checkpoint_dir="")
        assert "checkpoint_dir" in str(exc_info.value)

    def test_keep_last_zero_raises_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="checkpoint_keep_last"):
            _fast_model(checkpoint_dir=tmp_path, checkpoint_keep_last=0)

    def test_keep_last_float_raises_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="checkpoint_keep_last"):
            _fast_model(checkpoint_dir=tmp_path, checkpoint_keep_last=1.0)

    def test_keep_last_bool_raises_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="checkpoint_keep_last"):
            _fast_model(checkpoint_dir=tmp_path, checkpoint_keep_last=True)

    def test_keep_last_without_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="without checkpoint_dir"):
            _fast_model(checkpoint_keep_last=2)

    def test_b_default_attaches_no_checkpoint_callback(self) -> None:
        m = _fast_model()
        callbacks = m._build_callbacks()
        assert not any(isinstance(cb, CheckpointCallback) for cb in callbacks)

    def test_b_dir_attaches_fresh_callback_per_fit(self, tmp_path: Path) -> None:
        m = _fast_model(checkpoint_dir=tmp_path)
        first = [
            cb for cb in m._build_callbacks() if isinstance(cb, CheckpointCallback)
        ]
        second = [
            cb for cb in m._build_callbacks() if isinstance(cb, CheckpointCallback)
        ]
        assert len(first) == 1
        assert len(second) == 1
        assert first[0] is not second[0], (
            "CheckpointCallback must be constructed fresh per fit"
        )

    def test_b_user_checkpoint_callback_not_rejected(self, tmp_path: Path) -> None:
        user_cb = CheckpointCallback(directory=tmp_path / "user-stream", every_n=1)
        m = _fast_model(checkpoint_dir=tmp_path / "facade-stream", callbacks=[user_cb])
        ckpt_cbs = [
            cb for cb in m._build_callbacks() if isinstance(cb, CheckpointCallback)
        ]
        assert len(ckpt_cbs) == 2
        assert user_cb in ckpt_cbs







class TestResumeEndToEnd:

    def test_c_resume_restores_sga_state_and_completes(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        _fit_sga_with_checkpoints(small_burgers_dataset, ckpt_dir)
        final = ckpt_dir / _FINAL_NAME
        payload = torch.load(final, weights_only=False)
        saved_state = payload["algorithm_state"]

        assert saved_state["best_expression"] != ""
        assert math.isfinite(saved_state["best_score"])

        probe = _StartStateProbe()
        resumed = _fast_model(callbacks=[probe])
        result = resumed.fit(small_burgers_dataset, resume_from=final)

        assert result is resumed
        assert probe.start_state is not None
        assert probe.start_state["best_score"] == saved_state["best_score"], (
            "restored best_score must be visible at the resumed run's start"
        )
        assert probe.start_state["best_expression"] == saved_state["best_expression"], (
            "restored best_expression must be visible at the resumed run's start"
        )


        assert isinstance(resumed.best_expr_, str) and resumed.best_expr_
        assert resumed.best_score_ <= saved_state["best_score"]

    def test_c_resume_into_same_dir_fails_loud(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        from kd.search.checkpoint_manifest import CheckpointManifestError

        ckpt_dir = tmp_path / "ckpts"
        _fit_sga_with_checkpoints(small_burgers_dataset, ckpt_dir, checkpoint_every=1)
        final = ckpt_dir / _FINAL_NAME

        resumed = _fast_model(checkpoint_dir=ckpt_dir, checkpoint_every=1)
        with pytest.raises(CheckpointManifestError, match="is not empty"):
            resumed.fit(small_burgers_dataset, resume_from=final)

    def test_c_resume_into_fresh_dir_succeeds(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        from kd.search.checkpoint_manifest import load_checkpoint_manifest

        phase1 = tmp_path / "phase1"
        _fit_sga_with_checkpoints(small_burgers_dataset, phase1, checkpoint_every=1)
        final = phase1 / _FINAL_NAME

        phase2 = tmp_path / "phase2"
        resumed = _fast_model(checkpoint_dir=phase2, checkpoint_every=1)
        resumed.fit(small_burgers_dataset, resume_from=final)


        entries = load_checkpoint_manifest(phase2)
        assert any(e.kind == "final" for e in entries)


class TestFacadeRetentionAndPrecheck:

    def test_facade_threads_keep_last(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        from kd.search.checkpoint_manifest import (
            KIND_FINAL,
            KIND_PERIODIC,
            load_checkpoint_manifest,
        )

        ckpt_dir = tmp_path / "ckpts"
        _fit_sga_with_checkpoints(
            small_burgers_dataset, ckpt_dir, checkpoint_every=1, checkpoint_keep_last=1
        )
        entries = load_checkpoint_manifest(ckpt_dir)
        periodic = [e for e in entries if e.kind == KIND_PERIODIC]
        finals = [e for e in entries if e.kind == KIND_FINAL]
        assert len(periodic) == 1
        assert len(finals) == 1

    def test_precheck_fires_before_component_build(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from kd.search.checkpoint_manifest import CheckpointManifestError

        def _exploding_build(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("components built before fresh-dir pre-check")

        monkeypatch.setattr(Model, "_build_components", _exploding_build)

        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        (ckpt_dir / "occupied.txt").write_text("x")

        m = _fast_model(checkpoint_dir=ckpt_dir)
        with pytest.raises(CheckpointManifestError, match="is not empty"):
            m.fit(small_burgers_dataset)

    def test_precheck_applies_to_resume_fit(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from kd.search.checkpoint_manifest import CheckpointManifestError


        phase1 = tmp_path / "phase1"
        _fit_sga_with_checkpoints(small_burgers_dataset, phase1, checkpoint_every=1)
        final = phase1 / _FINAL_NAME

        def _exploding_build(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("components built before fresh-dir pre-check")

        monkeypatch.setattr(Model, "_build_components", _exploding_build)


        target = tmp_path / "target"
        target.mkdir()
        (target / "occupied.txt").write_text("x")
        m = _fast_model(checkpoint_dir=target, checkpoint_every=1)
        with pytest.raises(CheckpointManifestError, match="is not empty"):
            m.fit(small_burgers_dataset, resume_from=final)







class TestResumePayloadValidation:

    def test_d_missing_file_raises_filenotfound_naming_path(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        bogus = tmp_path / "no_such_checkpoint.pt"
        m = _fast_model()
        with pytest.raises(FileNotFoundError) as exc_info:
            m.fit(small_burgers_dataset, resume_from=bogus)
        assert "no_such_checkpoint.pt" in str(exc_info.value)

    @pytest.mark.parametrize(
        "garbage",
        [
            [1, 2, 3],
            "not-a-payload",
            {"version": CHECKPOINT_VERSION},
            {"iteration": 0, "algorithm_state": {}},
        ],
        ids=["list", "string", "missing-most-keys", "missing-version"],
    )
    def test_e_garbage_payload_raises_value_error(
        self,
        small_burgers_dataset: PDEDataset,
        tmp_path: Path,
        garbage: Any,
    ) -> None:
        path = tmp_path / "garbage.pt"
        torch.save(garbage, path)
        m = _fast_model()
        with pytest.raises(ValueError, match="not a kd checkpoint"):
            m.fit(small_burgers_dataset, resume_from=path)

    def test_f_version_mismatch_names_both_versions(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        bad_version = 999
        assert bad_version != CHECKPOINT_VERSION
        payload = {
            "version": bad_version,
            "iteration": 0,
            "algorithm_state": {},
            "best_score": 1.0,
            "best_expression": "",
        }
        path = tmp_path / "wrong_version.pt"
        torch.save(payload, path)
        m = _fast_model()
        with pytest.raises(ValueError) as exc_info:
            m.fit(small_burgers_dataset, resume_from=path)
        msg = str(exc_info.value)
        assert str(bad_version) in msg
        assert str(CHECKPOINT_VERSION) in msg

    def test_fix1_checkpoint_validated_before_components_built(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path, monkeypatch: Any
    ) -> None:

        def _exploding_builder(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("components built before checkpoint validation")

        monkeypatch.setattr(
            "kd.core.platform.builder.PlatformBuilder", _exploding_builder
        )
        garbage = tmp_path / "garbage.pt"
        torch.save([1, 2, 3], garbage)
        m = _fast_model()
        with pytest.raises(ValueError, match="not a kd checkpoint"):
            m.fit(small_burgers_dataset, resume_from=garbage)

    def test_fix4a_truncated_file_raises_value_error_naming_path(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.pt"
        torch.save(
            {
                "version": CHECKPOINT_VERSION,
                "iteration": 0,
                "algorithm_state": {"x": 1},
                "best_score": 1.0,
                "best_expression": "",
            },
            good,
        )
        torn = tmp_path / "torn.pt"
        raw = good.read_bytes()
        torn.write_bytes(raw[: len(raw) // 2])
        m = _fast_model()
        with pytest.raises(ValueError, match="not a kd checkpoint") as exc_info:
            m.fit(small_burgers_dataset, resume_from=torn)
        assert "torn.pt" in str(exc_info.value)







@pytest.fixture
def dlga_pretrained_surrogate(small_burgers_dataset: PDEDataset) -> torch.nn.Module:
    from kd.models.field_model import FieldModel

    target_dtype = small_burgers_dataset.get_field("u").dtype
    return FieldModel(
        coord_names=list(small_burgers_dataset.axis_order),
        field_names=["u"],
        hidden_sizes=[8],
        activation="tanh",
    ).to(dtype=target_dtype)


class TestResumeAlgorithmMismatch:

    def test_g_sga_checkpoint_into_dlga_model_raises(
        self,
        small_burgers_dataset: PDEDataset,
        tmp_path: Path,
        dlga_pretrained_surrogate: torch.nn.Module,
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        _fit_sga_with_checkpoints(small_burgers_dataset, ckpt_dir)
        final = ckpt_dir / _FINAL_NAME

        dlga_model = Model(
            algorithm="dlga",
            generations=2,
            verbose=False,
            surrogate_model=dlga_pretrained_surrogate,
        )
        with pytest.raises(ValueError) as exc_info:
            dlga_model.fit(small_burgers_dataset, resume_from=final)
        msg = str(exc_info.value)
        assert "sga" in msg
        assert "dlga" in msg

    def test_g_legacy_payload_without_algorithm_key_loads(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        _fit_sga_with_checkpoints(small_burgers_dataset, ckpt_dir)
        final = ckpt_dir / _FINAL_NAME
        payload = torch.load(final, weights_only=False)
        payload.pop("algorithm", None)
        legacy = tmp_path / "legacy.pt"
        torch.save(payload, legacy)

        probe = _StartStateProbe()
        resumed = _fast_model(callbacks=[probe])
        resumed.fit(small_burgers_dataset, resume_from=legacy)

        assert probe.start_state is not None
        assert (
            probe.start_state["best_expression"]
            == payload["algorithm_state"]["best_expression"]
        )
        assert isinstance(resumed.best_expr_, str) and resumed.best_expr_







class TestPySRResumeRecoverWithoutRerun:

    def test_h_pysr_resume_skips_backend_refit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.unit.search.pysr.conftest import (
            FakePySRBackend,
            make_backend_factory,
        )

        backend = FakePySRBackend()
        monkeypatch.setattr(
            "kd.search.pysr.plugin.default_backend_factory",
            make_backend_factory(backend),
        )
        dataset = _make_pysr_dataset()
        ckpt_dir = tmp_path / "ckpts"

        m1 = Model(
            algorithm="pysr",
            generations=2,
            seed=0,
            verbose=False,
            checkpoint_dir=ckpt_dir,
        )
        m1.fit(dataset)
        assert backend.fit_calls == 1
        assert m1.best_expr_
        final = ckpt_dir / _FINAL_NAME
        payload = torch.load(final, weights_only=False)
        assert payload["algorithm"] == "pysr"
        assert payload["algorithm_state"]["fitted"] is True

        _assert_manifest_honesty(ckpt_dir, "pysr")

        m2 = Model(algorithm="pysr", generations=2, seed=0, verbose=False)
        m2.fit(dataset, resume_from=final)

        assert backend.fit_calls == 1, (
            "resume must be recover-without-rerun: no second backend fit"
        )
        assert m2.best_expr_ == m1.best_expr_
        assert m2.result_.final_eval.is_valid, (
            "build_final_result must re-evaluate the restored best fresh"
        )

        assert m2.algorithm_.propose(5) == []


def _make_pysr_dataset() -> PDEDataset:
    from kd.data.schema import (
        AxisInfo,
        DataTopology,
        FieldData,
        PDEDataset,
        TaskType,
    )

    n_x, n_t = 24, 12
    x = torch.linspace(0.0, 2.0 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xg) * torch.exp(-tg)
    return PDEDataset(
        name="pysr-resume-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )







class TestDiscoverFacadeResume:

    def test_q_discover_resume_reprices_perturbed_gate(
        self, small_burgers_dataset: PDEDataset, tmp_path: Path
    ) -> None:
        cfg = DiscoverConfig(batch_size=8, seed=0)
        ckpt_dir = tmp_path / "ckpts"

        m1 = Model(
            algorithm="discover",
            generations=2,
            verbose=False,
            config=cfg,
            checkpoint_dir=ckpt_dir,
        )
        m1.fit(small_burgers_dataset)
        final = ckpt_dir / _FINAL_NAME
        payload = torch.load(final, weights_only=False)

        assert payload["algorithm"] == "discover"
        assert payload["best_expression"] != ""
        honest_reward = payload["best_score"]
        assert honest_reward > 0.0

        _assert_manifest_honesty(ckpt_dir, "discover")



        perturbed_factor = 0.5
        perturbed_reward = honest_reward * perturbed_factor


        assert perturbed_reward != pytest.approx(honest_reward, rel=1e-6)
        engine_state = payload["algorithm_state"]["engine_state"]
        assert engine_state["best_reward"] == pytest.approx(honest_reward, rel=1e-6)
        engine_state["best_reward"] = perturbed_reward
        perturbed = tmp_path / "perturbed_final.pt"
        torch.save(payload, perturbed)

        probe = _StartStateProbe()
        m2 = Model(
            algorithm="discover",
            generations=2,
            verbose=False,
            config=DiscoverConfig(batch_size=8, seed=0),
            callbacks=[probe],
        )
        m2.fit(small_burgers_dataset, resume_from=perturbed)

        assert probe.start_expression == payload["best_expression"], (
            "the restored champion expression must survive into the resumed "
            "run's starting state"
        )



        assert probe.start_score is not None
        assert probe.start_score == pytest.approx(honest_reward, rel=1e-6), (
            "prepare-time rebase must re-price the perturbed gate back to the "
            "honest checkpoint reward, not carry the stale perturbed value"
        )


        assert isinstance(m2.best_expr_, str) and m2.best_expr_
        assert m2.best_score_ >= probe.start_score - 1e-12
