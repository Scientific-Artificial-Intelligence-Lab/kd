
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
from kd.viz.style import DEFAULT_STYLE, style_context


class TestStyleContext:

    def test_restores_original_params(self) -> None:
        original = mpl.rcParams.copy()
        with style_context():
            pass
        for key in DEFAULT_STYLE:
            assert mpl.rcParams[key] == original[key]

    def test_applies_default_style(self) -> None:
        with style_context():
            for key, value in DEFAULT_STYLE.items():
                assert mpl.rcParams[key] == value

    def test_applies_extra_style(self) -> None:
        extra = {"font.size": 99}
        with style_context(extra):
            assert mpl.rcParams["font.size"] == 99

    def test_restores_after_extra(self) -> None:
        original_size = mpl.rcParams["font.size"]
        with style_context({"font.size": 99}):
            pass
        assert mpl.rcParams["font.size"] == original_size

    def test_restores_on_exception(self) -> None:
        original_size = mpl.rcParams["font.size"]
        try:
            with style_context({"font.size": 99}):
                raise ValueError("test error")
        except ValueError:
            pass
        assert mpl.rcParams["font.size"] == original_size
