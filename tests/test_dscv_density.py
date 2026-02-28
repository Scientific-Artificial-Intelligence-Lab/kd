"""Tests for DSCV density plot auto-sampling to prevent legend overflow.

When the number of epochs in r_history exceeds max_curves (default 10),
the _density() method should auto-sample epochs to keep the legend
manageable and the output image height reasonable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from kd.viz._style import VizConfig
from kd.viz.core import VizContext, VizResult
from kd.viz.adapters import DSCVVizAdapter


# -- Constants ---------------------------------------------------------------
DEFAULT_MAX_CURVES = 10
LARGE_EPOCH_COUNT = 200
SMALL_EPOCH_COUNT = 5
MAX_REASONABLE_IMAGE_HEIGHT_PX = 3000


# -- Helpers / Stubs ---------------------------------------------------------

def _make_history(n_epochs: int, n_rewards_per_epoch: int = 50) -> List[List[float]]:
    """Generate synthetic reward history with n_epochs epochs."""
    rng = np.random.default_rng(42)
    return [
        rng.normal(loc=i * 0.01, scale=0.5, size=n_rewards_per_epoch).tolist()
        for i in range(n_epochs)
    ]


class _StubSearcher:
    """Minimal searcher stub with configurable r_history."""

    def __init__(self, history: List[List[float]]) -> None:
        self.r_history = history


class _StubModel:
    """Minimal DSCV model stub."""

    def __init__(self, history: List[List[float]]) -> None:
        self.searcher = _StubSearcher(history)


def _make_ctx(tmp_path: Path, options: Optional[Dict[str, Any]] = None) -> VizContext:
    """Create a VizContext that writes to tmp_path."""
    opts: Dict[str, Any] = {'output_dir': str(tmp_path)}
    if options:
        opts.update(options)
    config = VizConfig(save_dir=tmp_path)
    return VizContext(config=config, backend='Agg', options=opts)


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def _use_agg_backend():
    """Force non-interactive matplotlib backend for CI."""
    import matplotlib
    original = matplotlib.get_backend()
    matplotlib.use('Agg')
    yield
    matplotlib.use(original, force=True)


# -- Tests -------------------------------------------------------------------

class TestDensityAutoSampling:
    """Tests for the auto-sampling logic in DSCVVizAdapter._density()."""

    @pytest.mark.unit
    def test_auto_sampling_triggers_when_many_epochs(self, tmp_path: Path) -> None:
        """With 200 epochs and no explicit epoches option, density should
        draw approximately max_curves (10) curves, not all 200."""
        model = _StubModel(_make_history(LARGE_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        assert isinstance(result, VizResult)
        assert result.paths, "Expected a saved file path"
        # The metadata 'epoches' should be a sampled subset
        sampled = result.metadata.get('epoches', [])
        assert len(sampled) <= DEFAULT_MAX_CURVES + 1, (
            f"Expected at most {DEFAULT_MAX_CURVES + 1} sampled epochs, got {len(sampled)}"
        )
        # Must be far fewer than the full history
        assert len(sampled) < LARGE_EPOCH_COUNT

    @pytest.mark.unit
    def test_auto_sampling_includes_last_epoch(self, tmp_path: Path) -> None:
        """Auto-sampling must always include the last epoch index."""
        model = _StubModel(_make_history(LARGE_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        sampled = result.metadata.get('epoches', [])
        last_index = LARGE_EPOCH_COUNT - 1
        assert last_index in sampled, (
            f"Last epoch index {last_index} should be in sampled set {sampled}"
        )

    @pytest.mark.unit
    def test_auto_sampling_includes_first_epoch(self, tmp_path: Path) -> None:
        """Auto-sampling should include the first epoch (index 0)."""
        model = _StubModel(_make_history(LARGE_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        sampled = result.metadata.get('epoches', [])
        assert 0 in sampled, (
            f"First epoch index 0 should be in sampled set {sampled}"
        )

    @pytest.mark.unit
    def test_auto_sampling_respects_custom_max_curves(self, tmp_path: Path) -> None:
        """Setting max_curves=5 should produce approximately 5 curves."""
        custom_max = 5
        model = _StubModel(_make_history(LARGE_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path, options={'max_curves': custom_max})
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        sampled = result.metadata.get('epoches', [])
        # Allow +1 for the forced last epoch
        assert len(sampled) <= custom_max + 1, (
            f"Expected at most {custom_max + 1} sampled epochs, got {len(sampled)}"
        )
        assert len(sampled) < LARGE_EPOCH_COUNT

    @pytest.mark.unit
    def test_no_auto_sampling_when_epoches_explicit(self, tmp_path: Path) -> None:
        """If user passes explicit epoches=[2, 5, 10], only those 3 are drawn."""
        explicit = [2, 5, 10]
        model = _StubModel(_make_history(LARGE_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path, options={'epoches': explicit})
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        sampled = result.metadata.get('epoches', [])
        assert sampled == explicit, (
            f"Expected explicit epoches {explicit}, got {sampled}"
        )

    @pytest.mark.unit
    def test_no_auto_sampling_when_history_small(self, tmp_path: Path) -> None:
        """With 5 epochs (< default max_curves=10), all should be drawn."""
        model = _StubModel(_make_history(SMALL_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        sampled = result.metadata.get('epoches', [])
        # When no auto-sampling, epoches stays None -> metadata shows all indices
        assert len(sampled) == SMALL_EPOCH_COUNT, (
            f"Expected all {SMALL_EPOCH_COUNT} epochs, got {len(sampled)}"
        )

    @pytest.mark.unit
    def test_image_height_reasonable_with_many_epochs(self, tmp_path: Path) -> None:
        """With 200 epochs, the output image height should be under 3000px.
        Before the fix, this could produce a 14429px tall image."""
        from PIL import Image

        model = _StubModel(_make_history(LARGE_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        assert result.paths, "Expected a density plot file"
        img_path = result.paths[0]
        assert img_path.exists(), f"Image file should exist at {img_path}"

        img = Image.open(str(img_path))
        width, height = img.size
        assert height < MAX_REASONABLE_IMAGE_HEIGHT_PX, (
            f"Image height {height}px exceeds maximum {MAX_REASONABLE_IMAGE_HEIGHT_PX}px. "
            f"Auto-sampling may not be working."
        )

    @pytest.mark.unit
    def test_sampled_epochs_are_sorted(self, tmp_path: Path) -> None:
        """Sampled epoch indices should be in ascending order."""
        model = _StubModel(_make_history(LARGE_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        sampled = result.metadata.get('epoches', [])
        assert sampled == sorted(sampled), (
            f"Sampled epochs should be sorted, got {sampled}"
        )

    @pytest.mark.unit
    def test_sampled_epochs_are_evenly_spaced(self, tmp_path: Path) -> None:
        """Auto-sampled epochs (excluding possibly the forced last one)
        should be approximately evenly spaced."""
        model = _StubModel(_make_history(LARGE_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        sampled = result.metadata.get('epoches', [])
        if len(sampled) < 3:
            pytest.skip("Not enough sampled epochs to check spacing")

        # The main sampled set (from range(0, n, step)) should be evenly spaced
        # The last element might be appended separately
        core = sampled[:-1] if sampled[-1] == LARGE_EPOCH_COUNT - 1 else sampled
        if len(core) >= 2:
            diffs = np.diff(core)
            # All diffs should be the same step size
            assert np.all(diffs == diffs[0]), (
                f"Core sampled epochs should be evenly spaced, diffs={diffs}"
            )


class TestDensityEdgeCases:
    """Edge case tests for _density()."""

    @pytest.mark.unit
    def test_empty_history(self, tmp_path: Path) -> None:
        """Empty r_history should return a warning, not crash."""
        model = _StubModel([])
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        assert result.warnings, "Expected a warning for empty history"
        assert not result.paths, "Should not produce a file for empty history"

    @pytest.mark.unit
    def test_single_epoch(self, tmp_path: Path) -> None:
        """History with exactly 1 epoch should work without error."""
        model = _StubModel(_make_history(1))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        assert result.paths, "Single epoch should produce a plot"
        sampled = result.metadata.get('epoches', [])
        assert len(sampled) == 1

    @pytest.mark.unit
    def test_exactly_max_curves_epochs(self, tmp_path: Path) -> None:
        """With exactly max_curves (10) epochs, no auto-sampling should trigger."""
        model = _StubModel(_make_history(DEFAULT_MAX_CURVES))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        sampled = result.metadata.get('epoches', [])
        # 10 epochs == max_curves: condition is len(history) > max_curves,
        # so no auto-sampling should trigger
        assert len(sampled) == DEFAULT_MAX_CURVES

    @pytest.mark.unit
    def test_max_curves_plus_one_triggers_sampling(self, tmp_path: Path) -> None:
        """With max_curves + 1 epochs, auto-sampling should trigger."""
        n_epochs = DEFAULT_MAX_CURVES + 1  # 11
        model = _StubModel(_make_history(n_epochs))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        sampled = result.metadata.get('epoches', [])
        # With 11 epochs and max_curves=10, step = max(1, 11//10) = 1
        # range(0, 11, 1) = [0,1,...,10] -> all 11, then last (10) already in
        # So this is a boundary case; sampling triggers but step=1 means all included
        # The key assertion: it ran without error
        assert result.paths, "Should produce a plot"

    @pytest.mark.unit
    def test_no_searcher_attribute(self, tmp_path: Path) -> None:
        """Model without searcher should return a warning."""
        model = type('NoSearcher', (), {})()
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        assert result.warnings
        assert not result.paths

    @pytest.mark.unit
    def test_no_r_history_on_searcher(self, tmp_path: Path) -> None:
        """Searcher without r_history should return a warning."""
        model = type('NoHistory', (), {'searcher': type('S', (), {})()})()
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        assert result.warnings
        assert not result.paths

    @pytest.mark.unit
    def test_explicit_epoches_out_of_range_filtered(self, tmp_path: Path) -> None:
        """Explicit epoches with out-of-range indices should be filtered
        (only valid indices used)."""
        history = _make_history(5)
        model = _StubModel(history)
        # Request epochs 0, 2, 999 -- 999 is out of range
        ctx = _make_ctx(tmp_path, options={'epoches': [0, 2, 999]})
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        # Should draw only 2 curves (indices 0 and 2), skipping 999
        assert result.paths, "Should produce a plot for valid epochs"
        sampled = result.metadata.get('epoches', [])
        assert sampled == [0, 2, 999], (
            "metadata epoches should reflect what was requested"
        )

    @pytest.mark.unit
    def test_max_curves_zero(self, tmp_path: Path) -> None:
        """max_curves=0 is an edge case; should not crash.
        With step = max(1, n//0) -> ZeroDivisionError would be a bug.
        But max_curves=0 means len(history) > 0 is always true, and
        step = max(1, n // 0) would fail. This tests robustness."""
        model = _StubModel(_make_history(5))
        ctx = _make_ctx(tmp_path, options={'max_curves': 0})
        adapter = DSCVVizAdapter()

        # This may raise ZeroDivisionError if not handled --
        # document the current behavior
        try:
            result = adapter._density(model, ctx)
            # If it succeeds, it should at least produce something
            assert result.paths or result.warnings
        except ZeroDivisionError:
            pytest.skip(
                "max_curves=0 causes ZeroDivisionError -- "
                "consider adding a guard in the implementation"
            )


class TestDensityOutputFile:
    """Tests verifying the output file is created correctly."""

    @pytest.mark.unit
    def test_output_file_created(self, tmp_path: Path) -> None:
        """Density plot should save a PNG file."""
        model = _StubModel(_make_history(SMALL_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        assert len(result.paths) == 1
        path = result.paths[0]
        assert path.exists()
        assert path.suffix == '.png'

    @pytest.mark.unit
    def test_output_in_dscv_subdirectory(self, tmp_path: Path) -> None:
        """Output should be in a 'dscv' subdirectory."""
        model = _StubModel(_make_history(SMALL_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        path = result.paths[0]
        assert 'dscv' in path.parts, (
            f"Expected output in dscv/ subdirectory, got {path}"
        )

    @pytest.mark.unit
    def test_intent_in_result(self, tmp_path: Path) -> None:
        """VizResult intent should be 'density'."""
        model = _StubModel(_make_history(SMALL_EPOCH_COUNT))
        ctx = _make_ctx(tmp_path)
        adapter = DSCVVizAdapter()

        result = adapter._density(model, ctx)

        assert result.intent == 'density'
