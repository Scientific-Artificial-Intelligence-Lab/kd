
from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


def read_xlsx_columns(
    path: str | Path,
    column_names: Sequence[str],
    *,
    sheet: str | int | None = None,
    header_row: int = 0,
    na_values: Sequence[str] = ("Indeterminate",),
) -> dict[str, list[float]]:
    if header_row < 0:
        raise ValueError(f"header_row must be >= 0, got {header_row}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        workbook = load_workbook(Path(path), read_only=True, data_only=True)
    try:
        worksheet = _select_sheet(workbook, sheet)





        if hasattr(worksheet, "reset_dimensions"):
            worksheet.reset_dimensions()
        header = next(
            worksheet.iter_rows(
                min_row=header_row + 1,
                max_row=header_row + 1,
                values_only=True,
            ),
            None,
        )
        if header is None:
            raise ValueError(f"XLSX file has no header row {header_row}: {path}")
        header_counts = Counter(
            value for value in header if isinstance(value, str)
        )
        ambiguous = [
            name for name in dict.fromkeys(column_names) if header_counts[name] > 1
        ]
        if ambiguous:
            raise ValueError(f"ambiguous XLSX columns: {ambiguous}")
        header_to_index = {
            value: index for index, value in enumerate(header) if isinstance(value, str)
        }
        missing = [name for name in column_names if name not in header_to_index]
        if missing:
            available = ", ".join(repr(name) for name in header_to_index)
            raise ValueError(
                f"missing XLSX columns: {missing}; available headers: {available}"
            )

        columns: dict[str, list[float]] = {name: [] for name in column_names}
        for row_index, row in enumerate(
            worksheet.iter_rows(min_row=header_row + 2, values_only=True),
            start=header_row + 2,
        ):
            for name, values in columns.items():
                index = header_to_index[name]
                if index >= len(row):
                    raise ValueError(
                        f"row {row_index} is shorter than the header: missing "
                        f"column {name!r} (index {index}, row width {len(row)})"
                    )
                values.append(_cell_as_float(row[index], row_index, name, na_values))
        return columns
    finally:
        workbook.close()


def _select_sheet(workbook: Any, sheet: str | int | None) -> Any:
    if sheet is None:
        return workbook.active
    if isinstance(sheet, str):
        return workbook[sheet]
    if isinstance(sheet, int):
        return workbook.worksheets[sheet]
    raise TypeError(f"sheet must be str, int, or None, got {type(sheet).__name__}")


def _cell_as_float(
    value: object,
    row_index: int,
    column_name: str,
    na_values: Sequence[str],
) -> float:
    if value is None or value in na_values:
        return float("nan")


    if isinstance(value, bool):
        raise ValueError(
            f"unexpected boolean cell at row {row_index}, column {column_name!r}: "
            f"{value!r}"
        )
    if isinstance(value, Real):
        return float(value)
    raise ValueError(
        f"unexpected non-numeric cell at row {row_index}, column {column_name!r}: "
        f"{value!r}"
    )
