
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

import kd
from kd.core.platform import builder as platform_builder
from kd.data import generate_advection_data
from kd.data.schema import PDEDataset
from kd.search.dlga import DLGAConfig

pytestmark = pytest.mark.integration

_ALGORITHMS = ("sga", "dlga")




_GENERATIONS = 5
_RESUME_AT = 2
_REMAINING = _GENERATIONS - 1 - _RESUME_AT

_POP_SIZE = 12
_SEED = 0

_SURROGATE_EPOCHS = 150




_SERIES_KEYS = ("n_unique", "n_valid", "gen_best_nmse")






_COMPARED_STATE_KEYS: dict[str, tuple[str, ...]] = {
    "sga": ("rng_state", "best_score", "best_expression"),
    "dlga": ("rng_state", "population", "best_score", "best_expression"),
}


_LAST_FITNESS_KEY = "last_fitness"
_SURROGATE_KEY = "surrogate_model"


def _model_kwargs(algorithm: str) -> dict[str, Any]:
    if algorithm == "dlga":
        return {
            "config": DLGAConfig(
                pop_size=_POP_SIZE,
                seed=_SEED,
                surrogate_max_epochs=_SURROGATE_EPOCHS,
            )
        }
    return {"population": _POP_SIZE, "seed": _SEED}


def _series(model: kd.Model, key: str) -> list[Any]:
    assert model.result_ is not None
    return model.result_.recorder.get(key)


def _state(model: kd.Model) -> dict[str, Any]:
    plugin = model.algorithm_
    assert plugin is not None
    return dict(plugin.state)


@dataclass
class _Baseline:

    model: kd.Model
    checkpoint_dir: Path

    @property
    def resume_from(self) -> Path:
        checkpoint = self.checkpoint_dir / f"checkpoint_{_RESUME_AT:06d}.pt"
        assert checkpoint.is_file(), (
            f"generation {_RESUME_AT} checkpoint missing; wrote "
            f"{sorted(p.name for p in self.checkpoint_dir.iterdir())}"
        )
        return checkpoint


@pytest.fixture(scope="module")
def dataset() -> PDEDataset:
    return generate_advection_data(
        speeds=(1.0,), waves=(2.0,), grid_sizes=(32,), nt=16, seed=0
    )


@pytest.fixture(scope="module")
def baselines(
    dataset: PDEDataset, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, _Baseline]:
    runs: dict[str, _Baseline] = {}
    for algorithm in _ALGORITHMS:
        checkpoint_dir = tmp_path_factory.mktemp(f"baseline_{algorithm}")
        model = kd.Model(
            algorithm=algorithm,
            generations=_GENERATIONS,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=1,
            verbose=False,
            **_model_kwargs(algorithm),
        )
        model.fit(dataset)
        runs[algorithm] = _Baseline(model=model, checkpoint_dir=checkpoint_dir)
    return runs


def _resume(
    dataset: PDEDataset,
    algorithm: str,
    checkpoint: Path,
    *,
    generations: int = _REMAINING,
    checkpoint_dir: Path | None = None,
) -> kd.Model:
    kwargs: dict[str, Any] = {
        "algorithm": algorithm,
        "generations": generations,
        "verbose": False,
        **_model_kwargs(algorithm),
    }
    if checkpoint_dir is not None:
        kwargs["checkpoint_dir"] = checkpoint_dir
        kwargs["checkpoint_every"] = 1
    model = kd.Model(**kwargs)
    model.fit(dataset, resume_from=checkpoint)
    return model


@pytest.fixture(scope="module")
def resumed(
    dataset: PDEDataset, baselines: dict[str, _Baseline]
) -> dict[str, kd.Model]:
    return {
        algorithm: _resume(dataset, algorithm, baselines[algorithm].resume_from)
        for algorithm in _ALGORITHMS
    }







@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_resumed_series_equal_the_uninterrupted_tail(
    algorithm: str,
    baselines: dict[str, _Baseline],
    resumed: dict[str, kd.Model],
) -> None:
    baseline, resumed_model = baselines[algorithm], resumed[algorithm]
    for key in _SERIES_KEYS:
        expected = _series(baseline.model, key)[_RESUME_AT + 1:]
        actual = _series(resumed_model, key)
        assert actual == pytest.approx(expected), (
            f"{algorithm} resume replays the archived generation: "
            f"{key} resumed={actual} but the uninterrupted tail after "
            f"generation {_RESUME_AT} is {expected} (full baseline series "
            f"{_series(baseline.model, key)})."
        )


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_resumed_final_state_equals_uninterrupted_final_state(
    algorithm: str,
    baselines: dict[str, _Baseline],
    resumed: dict[str, kd.Model],
) -> None:
    expected = _state(baselines[algorithm].model)
    actual = _state(resumed[algorithm])
    for key in _COMPARED_STATE_KEYS[algorithm]:
        assert actual[key] == expected[key], (
            f"{algorithm} resumed state[{key!r}] does not match the "
            f"uninterrupted run's final state; the two runs ended on "
            f"different generations."
        )







@dataclass
class _TrainerSpy:

    calls: int = 0


def _spy_on_surrogate_training(monkeypatch: pytest.MonkeyPatch) -> _TrainerSpy:
    spy = _TrainerSpy()
    original = platform_builder.FieldModelTrainer.fit

    def counting_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        spy.calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(platform_builder.FieldModelTrainer, "fit", counting_fit)
    return spy


def test_dlga_resume_trains_no_surrogate(
    dataset: PDEDataset,
    baselines: dict[str, _Baseline],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _spy_on_surrogate_training(monkeypatch)
    _resume(dataset, "dlga", baselines["dlga"].resume_from)
    assert spy.calls == 0, (
        f"DLGA resume trained a surrogate {spy.calls} time(s); the checkpointed "
        "module must be injected instead."
    )


def test_dlga_checkpoint_carries_the_trained_surrogate(
    dataset: PDEDataset, baselines: dict[str, _Baseline], tmp_path: Path
) -> None:
    payload = torch.load(baselines["dlga"].resume_from, weights_only=False)
    archived = payload["algorithm_state"]
    assert _SURROGATE_KEY in archived, (
        "the DLGA checkpoint carries no surrogate module, so a resume has "
        f"nothing to reuse; state keys are {sorted(archived)}."
    )

    resume_dir = tmp_path / "reuse"
    resumed_model = _resume(
        dataset,
        "dlga",
        baselines["dlga"].resume_from,
        checkpoint_dir=resume_dir,
    )
    assert resumed_model.result_ is not None
    forwarded = torch.load(resume_dir / "checkpoint_final.pt", weights_only=False)[
        "algorithm_state"
    ][_SURROGATE_KEY]

    trained = _state(baselines["dlga"].model)[_SURROGATE_KEY]
    trained_params = dict(trained.state_dict())
    forwarded_params = dict(forwarded.state_dict())
    assert forwarded_params.keys() == trained_params.keys()
    for name, tensor in trained_params.items():
        torch.testing.assert_close(
            forwarded_params[name], tensor, rtol=0.0, atol=0.0
        )


def _strip_state_keys(source: Path, dest: Path, *keys: str) -> Path:
    payload = torch.load(source, weights_only=False)
    for key in keys:
        payload["algorithm_state"].pop(key, None)
    torch.save(payload, dest)
    return dest


def test_dlga_resume_without_stored_surrogate_retrains_to_the_same_trajectory(
    dataset: PDEDataset,
    baselines: dict[str, _Baseline],
    resumed: dict[str, kd.Model],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stripped = _strip_state_keys(
        baselines["dlga"].resume_from, tmp_path / "no_surrogate.pt", _SURROGATE_KEY
    )
    spy = _spy_on_surrogate_training(monkeypatch)
    retrained = _resume(dataset, "dlga", stripped)

    assert spy.calls == 1, (
        "a checkpoint without a stored module must fall back to training one; "
        f"trainer was called {spy.calls} time(s)."
    )
    for key in _SERIES_KEYS:
        assert _series(retrained, key) == pytest.approx(
            _series(resumed["dlga"], key)
        ), f"retrained resume diverged from the reuse path on {key}."







def test_dlga_resume_from_legacy_payload_reevaluates_the_archived_generation(
    dataset: PDEDataset,
    baselines: dict[str, _Baseline],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    legacy = _strip_state_keys(
        baselines["dlga"].resume_from,
        tmp_path / "legacy.pt",
        _LAST_FITNESS_KEY,
        _SURROGATE_KEY,
    )
    spy = _spy_on_surrogate_training(monkeypatch)
    legacy_resumed = _resume(dataset, "dlga", legacy)

    assert legacy_resumed.best_expr_
    assert spy.calls == 1, (
        "a legacy checkpoint carries no surrogate, so the resume must train "
        f"one; trainer was called {spy.calls} time(s)."
    )
    for key in _SERIES_KEYS:
        baseline_series = _series(baselines["dlga"].model, key)
        assert _series(legacy_resumed, key) == pytest.approx(
            baseline_series[_RESUME_AT: _RESUME_AT + _REMAINING]
        ), (
            f"a legacy resume must re-evaluate generation {_RESUME_AT} "
            f"(today's semantics) rather than evolve past it; {key} was "
            f"{_series(legacy_resumed, key)}."
        )
