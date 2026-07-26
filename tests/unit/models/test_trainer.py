
from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from kd.models.field_model import FieldModel
from kd.models.trainer import FieldModelTrainer, TrainingResult




def _make_sin_data(n: int = 100) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    x = torch.linspace(0, 2 * math.pi, n)
    u = torch.sin(x)
    return {"x": x}, {"u": u}


def _make_2d_data(
    nx: int = 50, nt: int = 30
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    x_1d = torch.linspace(0, 2 * math.pi, nx)
    t_1d = torch.linspace(0, math.pi, nt)
    xx, tt = torch.meshgrid(x_1d, t_1d, indexing="ij")
    x_flat = xx.reshape(-1)
    t_flat = tt.reshape(-1)
    u_flat = torch.sin(x_flat) * torch.cos(t_flat)
    return {"x": x_flat, "t": t_flat}, {"u": u_flat}





class TestTrainerInterface:

    def test_creates_trainer(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model)
        assert trainer is not None

    def test_fit_returns_training_result(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(coords, targets, max_epochs=10, patience=None)
        assert isinstance(result, TrainingResult)
        assert isinstance(result.final_loss, float)
        assert isinstance(result.epochs_run, int)
        assert isinstance(result.early_stopped, bool)
        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds > 0.0


class TestTrainerDetachesAutogradInputs:

    def test_fit_accepts_requires_grad_coords_without_crash(self) -> None:
        model = FieldModel(coord_names=["x", "t"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_2d_data(20, 12)


        coords = {k: v.detach().clone().requires_grad_(True) for k, v in coords.items()}
        targets = {
            k: v.detach().clone().requires_grad_(True) for k, v in targets.items()
        }


        result = trainer.fit(
            coords, targets, max_epochs=3, patience=None, val_ratio=0.0, seed=0
        )

        assert math.isfinite(result.final_loss)
        assert result.epochs_run >= 2





class TestFittingQuality:

    def test_fit_sin_1d(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[64, 64])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(100)
        result = trainer.fit(
            coords, targets, max_epochs=5000, patience=None, val_ratio=0.0, seed=42
        )
        assert result.final_loss < 1e-4, f"sin(x) loss too high: {result.final_loss}"

    def test_fit_2d_sin_cos(self) -> None:
        model = FieldModel(
            coord_names=["x", "t"], field_names=["u"], hidden_sizes=[64, 64]
        )
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_2d_data(50, 30)
        result = trainer.fit(
            coords, targets, max_epochs=5000, patience=None, val_ratio=0.0, seed=42
        )
        assert result.final_loss < 1e-3, f"2D loss too high: {result.final_loss}"





class TestEarlyStopping:

    def test_early_stop_triggers(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[64, 64])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(100)
        result = trainer.fit(
            coords, targets, max_epochs=10000, patience=200, val_ratio=0.2, seed=42
        )
        assert result.early_stopped is True
        assert result.epochs_run < 10000

    def test_patience_none_runs_full(self) -> None:
        max_ep = 200
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=max_ep, patience=None, val_ratio=0.0
        )
        assert result.epochs_run == max_ep
        assert result.early_stopped is False

    def test_val_loss_present_when_val_split(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.2
        )
        assert result.val_loss is not None

    def test_val_loss_none_when_no_split(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.0
        )
        assert result.val_loss is None





class TestRNGIsolation:

    def test_rng_state_preserved(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)

        state_before = torch.get_rng_state().clone()
        trainer.fit(coords, targets, max_epochs=50, patience=None, val_ratio=0.0)
        state_after = torch.get_rng_state()

        assert torch.equal(state_before, state_after), "RNG state leaked!"





class TestReproducibility:

    def test_same_seed_same_loss(self) -> None:
        coords, targets = _make_sin_data(50)

        def _run(seed: int) -> float:
            model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
            trainer = FieldModelTrainer(model, lr=1e-3)
            result = trainer.fit(
                coords, targets, max_epochs=100, patience=None, val_ratio=0.0, seed=seed
            )
            return result.final_loss

        loss_a = _run(seed=123)
        loss_b = _run(seed=123)
        assert loss_a == loss_b, f"Reproducibility failed: {loss_a} vs {loss_b}"

    def test_different_seed_different_loss(self) -> None:
        coords, targets = _make_sin_data(50)

        def _run(seed: int) -> float:
            model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
            trainer = FieldModelTrainer(model, lr=1e-3)
            result = trainer.fit(
                coords, targets, max_epochs=100, patience=None, val_ratio=0.0, seed=seed
            )
            return result.final_loss

        loss_a = _run(seed=0)
        loss_b = _run(seed=999)

        assert loss_a != loss_b





class TestAutoNormalization:

    def test_normalization_buffers_set(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(100)
        trainer.fit(coords, targets, max_epochs=10, patience=None, val_ratio=0.0)


        x_mean = model.coord_x_mean
        x_std = model.coord_x_std
        assert x_mean.item() != 0.0, "coord mean should be set"
        assert x_std.item() != 1.0, "coord std should be set"


        u_mean = model.field_u_mean
        u_std = model.field_u_std

        assert abs(u_mean.item()) < 0.1, "field mean (sin over full period) ≈ 0"
        assert u_std.item() != 1.0, "field std should be set"

    def test_normalization_values_correct(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        x = torch.linspace(0, 10, 100)
        u = x * 2
        coords = {"x": x}
        targets = {"u": u}
        trainer.fit(coords, targets, max_epochs=10, patience=None, val_ratio=0.0)

        x_mean = model.coord_x_mean.item()
        x_std = model.coord_x_std.item()
        assert abs(x_mean - x.mean().item()) < 1e-6
        assert abs(x_std - x.std().item()) < 1e-6

        u_mean = model.field_u_mean.item()
        u_std = model.field_u_std.item()
        assert abs(u_mean - u.mean().item()) < 1e-6
        assert abs(u_std - u.std().item()) < 1e-6





class TestValidation:

    def test_val_ratio_1_raises(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        with pytest.raises(ValueError, match="val_ratio"):
            trainer.fit(coords, targets, max_epochs=10, val_ratio=1.0)

    def test_val_ratio_negative_raises(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        with pytest.raises(ValueError, match="val_ratio"):
            trainer.fit(coords, targets, max_epochs=10, val_ratio=-0.1)





class TestModelState:

    def test_model_eval_after_fit(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        trainer.fit(coords, targets, max_epochs=10, patience=None, val_ratio=0.0)
        assert not model.training, "Model should be in eval mode after fit()"





class TestTrainingResultCompleteness:

    def test_epochs_run_leq_max_epochs(self) -> None:
        max_ep = 150
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=max_ep, patience=None, val_ratio=0.0
        )
        assert result.epochs_run <= max_ep

    def test_epochs_run_equals_max_when_no_early_stop(self) -> None:
        max_ep = 80
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=max_ep, patience=None, val_ratio=0.0
        )
        assert result.epochs_run == max_ep
        assert result.early_stopped is False

    def test_final_loss_is_finite(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.0
        )
        assert math.isfinite(result.final_loss)
        assert result.final_loss >= 0.0

    def test_val_loss_is_finite_when_present(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.2
        )
        assert result.val_loss is not None
        assert math.isfinite(result.val_loss)
        assert result.val_loss >= 0.0

    def test_early_stopped_false_when_patience_none(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.2
        )
        assert result.early_stopped is False





class TestEarlyStoppingSupplemental:

    def test_early_stopped_flag_consistent_with_epochs(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[64, 64])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(100)
        result = trainer.fit(
            coords, targets, max_epochs=10000, patience=200, val_ratio=0.2, seed=42
        )
        if result.early_stopped:
            assert result.epochs_run < 10000
        else:
            assert result.epochs_run == 10000





class TestMultiFieldTraining:

    def test_two_field_training(self) -> None:
        n = 100
        x = torch.linspace(0, 2 * math.pi, n)
        u = torch.sin(x)
        v = torch.cos(x)
        coords = {"x": x}
        targets = {"u": u, "v": v}

        model = FieldModel(
            coord_names=["x"], field_names=["u", "v"], hidden_sizes=[64, 64]
        )
        trainer = FieldModelTrainer(model, lr=1e-3)
        result = trainer.fit(
            coords, targets, max_epochs=5000, patience=None, val_ratio=0.0, seed=42
        )

        assert result.final_loss < 1e-3, (
            f"Multi-field loss too high: {result.final_loss}"
        )

    def test_multi_field_normalization_buffers(self) -> None:
        n = 50
        x = torch.linspace(0, 2 * math.pi, n)
        u = torch.sin(x)
        v = torch.cos(x)
        coords = {"x": x}
        targets = {"u": u, "v": v}

        model = FieldModel(coord_names=["x"], field_names=["u", "v"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        trainer.fit(coords, targets, max_epochs=10, patience=None, val_ratio=0.0)

        assert hasattr(model, "field_u_mean")
        assert hasattr(model, "field_v_mean")
        assert hasattr(model, "field_u_std")
        assert hasattr(model, "field_v_std")

        assert model.field_u_mean.item() != model.field_v_mean.item()





class TestWeightDecay:

    def test_weight_decay_changes_training(self) -> None:
        coords, targets = _make_sin_data(50)

        def _run(wd: float) -> float:
            model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
            trainer = FieldModelTrainer(model, lr=1e-3, weight_decay=wd)
            result = trainer.fit(
                coords, targets, max_epochs=200, patience=None, val_ratio=0.0, seed=42
            )
            return result.final_loss

        loss_no_wd = _run(0.0)
        loss_high_wd = _run(0.1)


        assert loss_no_wd != loss_high_wd





class TestNumpyRNGIsolation:

    def test_numpy_rng_state_preserved(self) -> None:
        import numpy as np

        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)

        state_before = np.random.get_state()
        trainer.fit(coords, targets, max_epochs=50, patience=None, val_ratio=0.0)
        state_after = np.random.get_state()


        assert np.array_equal(state_before[1], state_after[1])





class TestRefit:

    def test_refit_resets_weights(self) -> None:
        coords, targets = _make_sin_data(50)
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)

        result1 = trainer.fit(
            coords, targets, max_epochs=100, patience=None, val_ratio=0.0, seed=42
        )
        result2 = trainer.fit(
            coords, targets, max_epochs=100, patience=None, val_ratio=0.0, seed=42
        )
        assert result1.final_loss == result2.final_loss


class TestDtypeAutoMatch:

    def test_float64_input_casts_model(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        assert next(model.parameters()).dtype == torch.float32

        trainer = FieldModelTrainer(model, lr=1e-3)
        x = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        u = torch.sin(x)
        result = trainer.fit(
            {"x": x}, {"u": u}, max_epochs=50, patience=None, val_ratio=0.0, seed=0
        )

        assert next(model.parameters()).dtype == torch.float64
        assert result.final_loss < 1.0

    def test_float32_input_keeps_model(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        x = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float32)
        u = torch.sin(x)
        result = trainer.fit(
            {"x": x}, {"u": u}, max_epochs=50, patience=None, val_ratio=0.0, seed=0
        )
        assert next(model.parameters()).dtype == torch.float32
        assert result.final_loss < 1.0















class TestTrainingResultHistoryDefaults:

    def test_construct_with_only_old_fields_still_works(self) -> None:
        result = TrainingResult(
            final_loss=0.5,
            epochs_run=10,
            early_stopped=False,
            val_loss=None,
        )

        assert result.loss_history == [], (
            "TrainingResult.loss_history must default to an empty list; "
            f"got {result.loss_history!r}"
        )
        assert result.val_loss_history is None, (
            "TrainingResult.val_loss_history must default to None; "
            f"got {result.val_loss_history!r}"
        )

    def test_history_fields_are_independent_per_instance(self) -> None:
        a = TrainingResult(
            final_loss=0.1, epochs_run=1, early_stopped=False, val_loss=None
        )
        b = TrainingResult(
            final_loss=0.2, epochs_run=1, early_stopped=False, val_loss=None
        )
        a.loss_history.append(1.0)
        assert b.loss_history == [], (
            "loss_history default must be a fresh list per instance (no shared "
            f"mutable default); leaked into another instance: {b.loss_history!r}"
        )


class TestFitPopulatesLossHistory:

    def test_loss_history_length_equals_epochs_run_no_val(self) -> None:
        max_ep = 12
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=max_ep, patience=None, val_ratio=0.0, seed=0
        )
        assert len(result.loss_history) == result.epochs_run, (
            f"loss_history length ({len(result.loss_history)}) must equal "
            f"epochs_run ({result.epochs_run})."
        )
        assert len(result.loss_history) == max_ep, (
            "no early-stop run must record exactly max_epochs history entries"
        )

    def test_loss_history_last_equals_final_loss(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=10, patience=None, val_ratio=0.0, seed=0
        )
        assert result.loss_history, "loss_history must be non-empty after fit()"

        assert result.loss_history[-1] == result.final_loss, (
            f"loss_history[-1]={result.loss_history[-1]!r} must equal "
            f"final_loss={result.final_loss!r} (same source value)."
        )

    def test_loss_history_elements_are_plain_floats(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=8, patience=None, val_ratio=0.0, seed=0
        )
        for i, value in enumerate(result.loss_history):
            assert type(value) is float, (
                f"loss_history[{i}] must be a built-in float, got "
                f"{type(value).__name__} ({value!r}); a stored tensor keeps the "
                "graph alive and breaks JSON serialization."
            )

    def test_loss_history_all_finite_and_nonneg(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=10, patience=None, val_ratio=0.0, seed=0
        )
        for value in result.loss_history:
            assert math.isfinite(value) and value >= 0.0, (
                f"loss_history entry {value!r} must be finite and >= 0 (MSE)."
            )


class TestFitPopulatesValLossHistory:

    def test_val_loss_history_present_and_aligned_when_split(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(60)
        result = trainer.fit(
            coords, targets, max_epochs=10, patience=None, val_ratio=0.2, seed=0
        )
        assert result.val_loss_history is not None, (
            "val_ratio>0 must produce a val_loss_history (not None)."
        )
        assert len(result.val_loss_history) == len(result.loss_history), (
            f"val_loss_history ({len(result.val_loss_history)}) must align 1:1 "
            f"with loss_history ({len(result.loss_history)})."
        )
        assert len(result.val_loss_history) == result.epochs_run

    def test_val_loss_history_last_equals_val_loss(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(60)
        result = trainer.fit(
            coords, targets, max_epochs=10, patience=None, val_ratio=0.2, seed=0
        )
        assert result.val_loss_history is not None
        assert result.val_loss is not None
        assert result.val_loss_history[-1] == result.val_loss, (
            f"val_loss_history[-1]={result.val_loss_history[-1]!r} must equal "
            f"val_loss={result.val_loss!r} (same source value)."
        )

    def test_val_loss_history_none_when_no_split(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=10, patience=None, val_ratio=0.0, seed=0
        )
        assert result.val_loss_history is None, (
            "val_ratio=0 must leave val_loss_history None (no validation curve)."
        )

        assert len(result.loss_history) == result.epochs_run

    def test_val_loss_history_elements_are_plain_floats(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(60)
        result = trainer.fit(
            coords, targets, max_epochs=8, patience=None, val_ratio=0.2, seed=0
        )
        assert result.val_loss_history is not None
        for i, value in enumerate(result.val_loss_history):
            assert type(value) is float, (
                f"val_loss_history[{i}] must be a built-in float, got "
                f"{type(value).__name__} ({value!r})."
            )


class TestFitHistoryWithEarlyStopAndRestore:

    def test_history_covers_actual_epochs_on_early_stop(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[64, 64])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(100)
        result = trainer.fit(
            coords, targets, max_epochs=10000, patience=20, val_ratio=0.2, seed=42
        )
        if not result.early_stopped:
            pytest.skip("did not early-stop in budget; covered by length test")
        assert len(result.loss_history) == result.epochs_run, (
            f"early-stopped run: loss_history length ({len(result.loss_history)}) "
            f"must equal epochs_run ({result.epochs_run}), not max_epochs."
        )
        assert result.epochs_run < 10000
        assert result.val_loss_history is not None
        assert len(result.val_loss_history) == result.epochs_run

    def test_restore_best_does_not_truncate_history(self) -> None:
        max_ep = 15
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(60)
        result = trainer.fit(
            coords,
            targets,
            max_epochs=max_ep,
            patience=None,
            val_ratio=0.2,
            seed=0,
            restore_best=True,
        )

        assert len(result.loss_history) == result.epochs_run == max_ep, (
            f"restore_best must not truncate history; got "
            f"{len(result.loss_history)} entries for epochs_run="
            f"{result.epochs_run} (max={max_ep})."
        )
        assert result.val_loss_history is not None
        assert len(result.val_loss_history) == max_ep


        if result.best_epoch is not None:
            assert len(result.loss_history) >= result.best_epoch
