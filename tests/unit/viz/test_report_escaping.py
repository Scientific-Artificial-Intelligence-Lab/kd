
from __future__ import annotations

import html
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pytest

from kd.search.result import ExperimentResult
from kd.viz.report import generate_report

pytestmark = pytest.mark.unit

_INJECTION = "</pre><script>INJECTED</script><pre>"

_JSON_BLOCK = re.compile(r'<pre class="json-summary">(.*?)</pre>', re.DOTALL)


def _json_block(html_text: str) -> str:
    match = _JSON_BLOCK.search(html_text)
    assert match is not None, "report must contain the json-summary <pre> block"
    return match.group(1)


def _decoded_json(html_text: str) -> dict[str, Any]:
    decoded = html.unescape(_json_block(html_text))
    parsed: dict[str, Any] = json.loads(decoded)
    return parsed


def _render(
    result: ExperimentResult, tmp_path: Path, name: str = "report.html"
) -> str:
    output = tmp_path / name
    generate_report(result, [], output)
    return output.read_text(encoding="utf-8")


def test_markup_in_dataset_name_does_not_reach_the_html_raw(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    result = replace(mock_experiment_result, dataset_name=_INJECTION)

    content = _render(result, tmp_path)

    assert "<script>INJECTED</script>" not in content




    assert _INJECTION not in content


def test_markup_in_config_value_does_not_reach_the_html_raw(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    result = replace(
        mock_experiment_result,
        config={"max_iter": 10, "note": _INJECTION},
    )

    content = _render(result, tmp_path)

    assert "<script>INJECTED</script>" not in content
    assert _INJECTION not in content


def test_escaped_json_summary_is_still_valid_readable_json(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    result = replace(
        mock_experiment_result,
        dataset_name=_INJECTION,
        config={"max_iter": 10, "note": _INJECTION},
    )

    summary = _decoded_json(_render(result, tmp_path))

    assert summary["dataset_name"] == _INJECTION
    assert summary["config"]["note"] == _INJECTION
    assert summary["config"]["max_iter"] == 10
    assert summary["algorithm_name"] == "SGA"


def test_json_own_escapes_are_not_double_encoded(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    payload = 'back\\slash and "quotes" & ampersand'
    result = replace(mock_experiment_result, dataset_name=payload)

    content = _render(result, tmp_path)

    assert _decoded_json(content)["dataset_name"] == payload


    assert "back\\\\slash" in _json_block(content)
    assert "&amp;amp;" not in content


def test_plain_report_json_summary_round_trips(
    tmp_path: Path,
    mock_experiment_result: ExperimentResult,
) -> None:
    summary = _decoded_json(_render(mock_experiment_result, tmp_path))

    assert summary["dataset_name"] == "test_dataset"
    assert summary["iterations"] == 10
