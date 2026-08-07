
from __future__ import annotations

import io
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

import kd
import kd.inspect as kd_inspect
from kd._inspect_report import AxisReport, DatasetReport, FieldReport
from kd.core.expr.naming import parse_derivative_name
from kd.data.derivatives.finite_diff import UNIFORM_GRID_RTOL, is_uniform_grid
from kd.data.schema import PDEDataset
from kd.inspect import preview, preview_report
from tests.unit import _inspect_cases as cases

_SECRET = "SECRET_EQUATION"


def _rendered_warnings(dataset: PDEDataset) -> list[str]:
    buf = io.StringIO()
    preview(dataset, file=buf)
    prefix = " - WARNING: "
    return [
        line[len(prefix):]
        for line in buf.getvalue().splitlines()
        if line.startswith(prefix)
    ]


def _all_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(key)
            keys |= _all_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            keys |= _all_keys(item)
    return keys


_ALL_CASES = [
    "clean_uniform_dataset",
    "non_uniform_dataset",
    "descending_dataset",
    "non_finite_dx_dataset",
    "nan_inf_field_dataset",
    "all_non_finite_field_dataset",
    "small_grid_dataset",
    "single_point_axis_dataset",
    "lhs_unset_dataset",
    "mixed_dtype_dataset",
    "wave_lhs_dataset",
    "scattered_dataset",
    "metadata_only_dataset",
]







def test_report_symbols_are_top_level_exports() -> None:
    assert kd.preview_report is kd_inspect.preview_report
    for symbol in ("AxisReport", "DatasetReport", "FieldReport", "preview_report"):
        assert symbol in kd.__all__
        assert hasattr(kd, symbol)
    assert list(kd.__all__) == sorted(kd.__all__)







def test_axis_reports_carry_length_range_and_steps() -> None:
    report = preview_report(cases.clean_uniform_dataset())

    assert report.name == "probe_clean"
    assert report.topology == "grid"
    assert report.axes is not None
    x_axis, t_axis = report.axes
    assert [axis.name for axis in report.axes] == ["x", "t"]
    assert x_axis.n == 32
    assert x_axis.min == pytest.approx(0.0)
    assert x_axis.max == pytest.approx(1.0)
    assert x_axis.spacing == "uniform"
    assert x_axis.step_first == pytest.approx(1.0 / 31)
    assert x_axis.step_mean == pytest.approx(1.0 / 31)
    assert x_axis.step_min == pytest.approx(1.0 / 31)
    assert x_axis.step_max == pytest.approx(1.0 / 31)
    assert t_axis.n == 20
    assert t_axis.max == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        ("clean_uniform_dataset", "uniform"),
        ("non_uniform_dataset", "non_uniform"),
        ("descending_dataset", "decreasing"),
        ("non_finite_dx_dataset", "non_finite"),
        ("single_point_axis_dataset", "single_point"),
        ("scattered_dataset", "not_applicable"),
    ],
)
def test_spacing_verdict_per_case(builder: str, expected: str) -> None:
    report = preview_report(getattr(cases, builder)())

    assert report.axes is not None
    assert report.axes[0].name == "x"
    assert report.axes[0].spacing == expected


def test_single_point_axis_has_no_step_statistics() -> None:
    report = preview_report(cases.single_point_axis_dataset())

    assert report.axes is not None
    x_axis = report.axes[0]
    assert x_axis.n == 1
    assert (
        x_axis.step_first,
        x_axis.step_mean,
        x_axis.step_min,
        x_axis.step_max,
    ) == (None, None, None, None)


def test_uniform_verdict_shares_the_finite_difference_predicate() -> None:
    x = torch.linspace(0.0, 1.0, 2000, dtype=torch.float32)
    dataset = cases.axis_probe_dataset(x)

    report = preview_report(dataset)

    assert is_uniform_grid(x, rtol=UNIFORM_GRID_RTOL)
    assert report.axes is not None
    assert report.axes[0].spacing == "uniform"


def test_field_report_counts_nan_and_inf_separately() -> None:
    report = preview_report(cases.nan_inf_field_dataset())

    assert report.fields is not None
    field = report.fields[0]
    assert field.name == "u"
    assert field.dtype == "float64"
    assert field.shape == (20, 12)
    assert field.nan_count == 1
    assert field.inf_count == 1
    assert field.min == pytest.approx(0.01)
    assert field.max == pytest.approx(2.39)


def test_field_report_statistics_are_nan_when_nothing_is_finite() -> None:
    report = preview_report(cases.all_non_finite_field_dataset())

    assert report.fields is not None
    field = report.fields[0]
    assert field.nan_count == 240
    assert field.inf_count == 0
    for value in (field.min, field.max, field.mean):
        assert value != value


def test_field_dtype_is_stripped_while_the_warning_keeps_the_full_name() -> None:
    report = preview_report(cases.mixed_dtype_dataset())

    assert report.fields is not None
    assert [field.dtype for field in report.fields] == ["float64", "float32"]
    assert report.warnings == [
        "fields have mixed dtypes ['torch.float32', 'torch.float64'] "
        "— consider casting all fields to the same dtype"
    ]


def test_lhs_spec_is_reported_with_its_order() -> None:
    report = preview_report(cases.wave_lhs_dataset())

    assert (report.lhs_field, report.lhs_axis) == ("u", "t")
    assert report.lhs_label == "u_tt"
    assert report.lhs_order == 2


def test_lhs_label_decodes_back_to_the_reported_spec() -> None:
    for dataset in (cases.clean_uniform_dataset(), cases.wave_lhs_dataset()):
        report = preview_report(dataset)

        assert report.lhs_label is not None
        assert parse_derivative_name(report.lhs_label) == (
            report.lhs_field,
            report.lhs_axis,
            report.lhs_order,
        )


def test_preview_prints_the_order_free_label(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = cases.wave_lhs_dataset()

    preview(dataset)

    assert "LHS: u_t (field='u', axis='t')" in capsys.readouterr().out
    assert preview_report(dataset).lhs_label == "u_tt"


def test_lhs_label_falls_back_when_the_axis_name_is_unencodable() -> None:
    report = preview_report(cases.underscore_axis_dataset())

    assert report.lhs_label == "u_x_1"
    assert report.lhs_order == 2


def test_unset_lhs_reports_none_and_warns() -> None:
    report = preview_report(cases.lhs_unset_dataset())

    assert report.lhs_field is None
    assert report.lhs_axis is None
    assert report.lhs_label is None
    assert report.warnings == [
        "lhs_field / lhs_axis not set — Model.fit() will fall back to ('u', 't')"
    ]


def test_metadata_only_dataset_reports_absent_payloads() -> None:
    report = preview_report(cases.metadata_only_dataset())

    assert report.axes is None
    assert report.fields is None
    assert report.to_dict()["axes"] is None
    assert report.to_dict()["fields"] is None







@pytest.mark.parametrize("builder", _ALL_CASES)
def test_report_warnings_match_printed_warnings(builder: str) -> None:
    dataset = getattr(cases, builder)()

    assert preview_report(dataset).warnings == _rendered_warnings(dataset)







@pytest.mark.parametrize("builder", _ALL_CASES)
def test_report_json_round_trips_without_nan(builder: str) -> None:
    payload = preview_report(getattr(cases, builder)()).to_dict()

    dumped = json.dumps(payload, allow_nan=False)

    assert json.loads(dumped) == payload


def test_non_finite_statistics_serialize_as_null() -> None:
    payload = preview_report(cases.all_non_finite_field_dataset()).to_dict()
    field = payload["fields"][0]

    assert field["min"] is None
    assert field["max"] is None
    assert field["mean"] is None

    inf_payload = preview_report(cases.non_finite_dx_dataset()).to_dict()
    x_axis = inf_payload["axes"][0]

    assert x_axis["step_first"] is None
    assert x_axis["spacing"] == "non_finite"


def test_absent_step_and_non_finite_step_are_told_apart_by_spacing() -> None:
    absent = preview_report(cases.single_point_axis_dataset()).to_dict()["axes"][0]
    non_finite = preview_report(cases.non_finite_dx_dataset()).to_dict()["axes"][0]

    assert absent["step_first"] is None
    assert non_finite["step_first"] is None
    assert absent["spacing"] == "single_point"
    assert non_finite["spacing"] == "non_finite"


def test_shape_serializes_as_a_list_of_ints() -> None:
    payload = preview_report(cases.clean_uniform_dataset()).to_dict()

    assert payload["fields"][0]["shape"] == [32, 20]







def test_report_never_carries_the_ground_truth_equation() -> None:
    dataset = cases.ground_truth_dataset(_SECRET)

    payload = preview_report(dataset).to_dict()

    assert dataset.ground_truth == _SECRET
    assert _SECRET not in json.dumps(payload)


def test_report_payload_has_no_truth_or_noise_keys() -> None:
    dataset = cases.ground_truth_dataset(_SECRET)

    keys = _all_keys(preview_report(dataset).to_dict())

    assert "ground_truth" not in keys
    assert "noise_level" not in keys
    assert dataset.noise_level != 0.0







def test_scattered_axes_report_no_spacing_and_no_grid_warnings() -> None:
    dataset = cases.scattered_dataset()

    report = preview_report(dataset)

    assert report.topology == "scattered"
    assert report.axes is not None
    for axis in report.axes:
        assert axis.spacing == "not_applicable"
        assert axis.step_first is None
        assert axis.step_mean is None
        assert axis.step_min is None
        assert axis.step_max is None
    assert report.warnings == []


def test_scattered_field_warnings_are_still_reported() -> None:
    dataset = cases.scattered_dataset()
    assert dataset.fields is not None
    dataset.fields["u"].values[0] = float("nan")

    report = preview_report(dataset)

    assert report.warnings == [
        "field 'u' contains 1 NaN value(s) — dataset will not fit until cleaned"
    ]
    assert report.fields is not None
    assert report.fields[0].nan_count == 1


def test_scattered_axis_row_renders_as_not_applicable() -> None:
    buf = io.StringIO()

    preview(cases.scattered_dataset(), file=buf)
    text = buf.getvalue()

    assert "step n/a (scattered)" in text
    assert "NON-UNIFORM" not in text
    assert "WARNING:" not in text







@pytest.mark.parametrize("builder", _ALL_CASES)
def test_preview_report_consumes_no_rng(builder: str) -> None:
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()

    preview_report(getattr(cases, builder)())

    assert torch.equal(torch.random.get_rng_state(), before)







def test_axis_report_rejects_an_unknown_spacing_verdict() -> None:
    with pytest.raises(ValueError, match="unknown axis spacing verdict"):
        AxisReport(
            name="x",
            n=4,
            min=0.0,
            max=1.0,
            spacing="wobbly",
            step_first=None,
            step_mean=None,
            step_min=None,
            step_max=None,
        )


def test_field_report_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="counts must be >= 0"):
        FieldReport(
            name="u",
            dtype="float64",
            shape=(2, 2),
            min=0.0,
            max=1.0,
            mean=0.5,
            nan_count=-1,
            inf_count=0,
        )


def test_report_types_are_frozen() -> None:
    report = preview_report(cases.clean_uniform_dataset())

    with pytest.raises(FrozenInstanceError):
        report.name = "renamed"
    assert isinstance(report, DatasetReport)
