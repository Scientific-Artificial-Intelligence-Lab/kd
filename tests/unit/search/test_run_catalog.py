
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kd.search.run_catalog import (
    BEST_EXPRESSION_MAX_CHARS,
    append_catalog_row,
    catalog_row_from_result,
)

pytestmark = pytest.mark.unit


def _row(**overrides: object) -> dict[str, object]:
    base = catalog_row_from_result(
        None,
        run_id="sga-x",
        created_at="2026-08-09T00:00:00+00:00",
        instrument="sga",
        status="raised",
        run_dir="sga-x",
        dataset_name="burgers_1d",
        dataset_cache_fingerprint="sha256:dataset",
        seed=0,
    )
    base.update(overrides)
    return base


class TestRowBuild:
    def test_result_free_row_keeps_explicit_identity(self) -> None:
        row = _row()
        assert row["instrument"] == "sga"
        assert row["dataset_cache_fingerprint"] == "sha256:dataset"
        assert row["best_score"] is None
        assert row["config_hash"] is None

    def test_result_fills_scores_and_truncates_expression(self) -> None:


        class _FinalEval:
            nmse = 0.02

        class _Result:
            best_score = -50.0
            score_kind = "AIC"
            score_direction = "min"
            best_expression = "u" * (BEST_EXPRESSION_MAX_CHARS * 2)
            dataset_name = "burgers_1d"
            manifest = None
            run_record = None
            final_eval = _FinalEval()

        row = catalog_row_from_result(
            _Result(),
            run_id="sga-x",
            created_at="2026-08-09T00:00:00+00:00",
            instrument="sga",
            status="completed",
            run_dir="sga-x",
        )
        assert row["nmse"] == 0.02
        expression = row["best_expression"]
        assert isinstance(expression, str)
        assert len(expression) == BEST_EXPRESSION_MAX_CHARS


class TestAppend:
    def test_appends_parse_back_line_per_row(self, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog.jsonl"
        append_catalog_row(catalog, _row(run_id="a"))
        append_catalog_row(catalog, _row(run_id="b"))
        rows = [
            json.loads(line) for line in catalog.read_text().splitlines()
        ]
        assert [row["run_id"] for row in rows] == ["a", "b"]
        assert all(row["scheme"] == "kd-runcat-v1" for row in rows)

    def test_rejects_key_drift(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="key drift"):
            append_catalog_row(tmp_path / "c.jsonl", {**_row(), "extra": 1})
        missing = _row()
        del missing["nmse"]
        with pytest.raises(ValueError, match="key drift"):
            append_catalog_row(tmp_path / "c.jsonl", missing)

    def test_rejects_non_finite_and_bool_smuggling(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="nmse"):
            append_catalog_row(
                tmp_path / "c.jsonl", _row(nmse=float("inf"))
            )
        with pytest.raises(ValueError, match="seed"):
            append_catalog_row(tmp_path / "c.jsonl", _row(seed=True))
