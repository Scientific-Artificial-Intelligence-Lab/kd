
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kd.data.loaders.csv_grid import read_grid_csv


def _write_csv(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path







class TestReadGridCsvSmoke:

    @pytest.mark.smoke
    def test_importable_from_module(self) -> None:
        from kd.data.loaders.csv_grid import read_grid_csv as fn

        assert callable(fn)

    @pytest.mark.smoke
    def test_importable_from_package(self) -> None:
        from kd.data.loaders import read_grid_csv as fn

        assert callable(fn)







class TestReadGridCsvHappyPath:

    @pytest.mark.unit
    def test_well_formed_matrix_loads(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path,
            "grid.csv",
            "1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n9.0,10.0,11.0,12.0\n",
        )
        arr = read_grid_csv(path, name="grid")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float64
        expected = np.arange(1.0, 13.0, dtype=np.float64).reshape(3, 4)
        np.testing.assert_array_equal(arr, expected)

    @pytest.mark.unit
    def test_accepts_str_path(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "grid.csv", "1.5,2.5\n3.5,4.5\n")
        arr = read_grid_csv(str(path), name="grid")
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float64

    @pytest.mark.unit
    def test_expected_shape_match_ok(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "grid.csv", "0.0,1.0\n2.0,3.0\n")
        arr = read_grid_csv(path, name="grid", expected_shape=(2, 2))
        assert arr.shape == (2, 2)

    @pytest.mark.unit
    def test_trailing_blank_line_tolerated(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path,
            "grid.csv",
            "1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n9.0,10.0,11.0,12.0\n\n",
        )
        arr = read_grid_csv(path, name="grid")
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float64

    @pytest.mark.unit
    def test_crlf_line_endings_tolerated(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "grid.csv", "0.0,1.0\r\n2.0,3.0\r\n")
        arr = read_grid_csv(path, name="grid")
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float64







class TestReadGridCsvShapeContract:

    @pytest.mark.unit
    def test_single_row_stays_2d(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "row.csv", "1.0,2.0,3.0\n")
        arr = read_grid_csv(path, name="row")
        assert arr.ndim == 2
        assert arr.shape == (1, 3)

    @pytest.mark.unit
    def test_single_column_stays_2d(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "col.csv", "1.0\n2.0\n3.0\n")
        arr = read_grid_csv(path, name="col")
        assert arr.ndim == 2
        assert arr.shape == (3, 1)







class TestReadGridCsvFailLoud:

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_grid_csv(tmp_path / "does_not_exist.csv", name="grid")

    @pytest.mark.unit
    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "empty.csv", "")
        with pytest.raises(ValueError):
            read_grid_csv(path, name="grid")

    @pytest.mark.unit
    def test_whitespace_only_file_raises(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "blank.csv", "\n \n\n")
        with pytest.raises(ValueError):
            read_grid_csv(path, name="grid")

    @pytest.mark.unit
    def test_ragged_rows_raise(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "ragged.csv", "1.0,2.0,3.0\n4.0,5.0\n")
        with pytest.raises(ValueError):
            read_grid_csv(path, name="grid")

    @pytest.mark.unit
    def test_non_numeric_cell_raises(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "text.csv", "1.0,2.0\n3.0,abc\n")
        with pytest.raises(ValueError):
            read_grid_csv(path, name="grid")

    @pytest.mark.numerical
    def test_nan_cell_raises(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "nan.csv", "1.0,2.0\n3.0,nan\n")
        with pytest.raises(ValueError):
            read_grid_csv(path, name="grid")

    @pytest.mark.numerical
    def test_inf_cell_raises(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "inf.csv", "1.0,inf\n3.0,4.0\n")
        with pytest.raises(ValueError):
            read_grid_csv(path, name="grid")

    @pytest.mark.unit
    def test_shape_mismatch_raises(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path,
            "grid.csv",
            "1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n9.0,10.0,11.0,12.0\n",
        )
        with pytest.raises(ValueError):
            read_grid_csv(path, name="grid", expected_shape=(2, 2))
