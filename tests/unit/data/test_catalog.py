
from __future__ import annotations

from collections.abc import Callable

import pytest

import kd
import kd.data as data
from kd.data.schema import PDEDataset

EXPECTED_DATASET_IDS = {
    "allen-cahn",
    "burgers",
    "burgers-2d",
    "chafee-infante",
    "convection-diffusion",
    "eq-6-2-12",
    "kdv",
    "llm4ed-fisher",
    "llm4ed-fisher-nonlinear",
    "llm4ed-heat",
    "pde-compound",
    "pde-divide",
}

LOADABLE_DATASET_IDS = {
    "burgers",
    "chafee-infante",
    "kdv",
    "pde-compound",
    "pde-divide",
}


def _assert_nonempty(value: str, field_name: str, dataset_id: str) -> None:
    assert value.strip(), f"{dataset_id}.{field_name} must be non-empty"


class TestDatasetCatalog:

    @pytest.mark.smoke
    def test_top_level_list_datasets_exported(self) -> None:
        assert hasattr(kd, "list_datasets")
        assert kd.list_datasets is data.list_datasets
        assert hasattr(kd, "load_from_hub")
        assert kd.load_from_hub is data.load_from_hub

    @pytest.mark.unit
    def test_catalog_contains_expected_ids(self) -> None:
        assert set(data.DATASET_CATALOG) == EXPECTED_DATASET_IDS
        assert [spec.id for spec in data.list_datasets()] == sorted(
            EXPECTED_DATASET_IDS
        )

    @pytest.mark.unit
    def test_specs_are_complete_and_minimal(self) -> None:
        for spec in data.list_datasets():
            assert isinstance(spec, data.DatasetSpec)
            _assert_nonempty(spec.id, "id", spec.id)
            _assert_nonempty(spec.equation, "equation", spec.id)
            _assert_nonempty(spec.lhs, "lhs", spec.id)
            _assert_nonempty(spec.fmt, "fmt", spec.id)
            _assert_nonempty(spec.source, "source", spec.id)
            _assert_nonempty(spec.license, "license", spec.id)
            _assert_nonempty(spec.tier, "tier", spec.id)
            _assert_nonempty(spec.schema_version, "schema_version", spec.id)
            assert isinstance(spec.axes, tuple)
            assert spec.axes
            assert all(axis.strip() for axis in spec.axes)
            assert callable(spec.loader)
            assert isinstance(spec.tags, tuple)

    @pytest.mark.unit
    def test_list_datasets_returns_a_fresh_sorted_list(self) -> None:
        first = data.list_datasets()
        second = data.list_datasets()
        assert first == second
        assert first is not second

        first.pop()
        assert len(data.list_datasets()) == len(EXPECTED_DATASET_IDS)

    @pytest.mark.unit
    def test_get_dataset_returns_spec(self) -> None:
        spec = data.get_dataset("kdv")
        assert spec is data.DATASET_CATALOG["kdv"]
        assert spec.equation == "u_t = -u * u_x - 0.0025 * u_xxx"

    @pytest.mark.unit
    def test_get_dataset_rejects_unknown_id(self) -> None:
        with pytest.raises(KeyError, match="unknown dataset id"):
            data.get_dataset("not-a-dataset")

    @pytest.mark.unit
    def test_eqgpt_specs_mark_eqgpt_source(self) -> None:
        for dataset_id in ("allen-cahn", "convection-diffusion"):
            spec = data.get_dataset(dataset_id)
            assert "EqGPT" in spec.source
            assert "EqGPT" in spec.license
            assert "eqgpt" in spec.tags

    @pytest.mark.unit
    def test_pde_compound_records_eqgpt_non_equivalence(self) -> None:
        spec = data.get_dataset("pde-compound")
        assert "not equivalent to EqGPT PDE_compound" in spec.source

    @pytest.mark.unit
    def test_remote_specs_record_pinned_hub_metadata(self) -> None:
        remote_specs = data.list_remote_datasets()
        assert [spec.id for spec in remote_specs] == [
            "llm4ed-fisher",
            "llm4ed-fisher-nonlinear",
            "llm4ed-heat",
        ]
        for spec in remote_specs:
            assert spec.tier == "remote"
            assert spec.repo_id == "timeoutHao/KD-data"
            assert spec.revision == "6c7dbe4f032e14fea2194644b09e74a4f254dd42"
            assert spec.checksum is not None
            assert len(spec.checksum) == 64
            assert len(spec.files) == 1

    @pytest.mark.unit
    @pytest.mark.parametrize("dataset_id", sorted(LOADABLE_DATASET_IDS))
    def test_loader_returns_pde_dataset_for_bundled_data(self, dataset_id: str) -> None:
        dataset = data.get_dataset(dataset_id).loader()
        assert isinstance(dataset, PDEDataset)
        assert dataset.name == dataset_id

    @pytest.mark.unit
    @pytest.mark.parametrize("dataset_id", sorted(LOADABLE_DATASET_IDS))
    def test_spec_metadata_stays_in_sync_with_loader(self, dataset_id: str) -> None:
        spec = data.get_dataset(dataset_id)
        dataset = spec.loader()
        assert dataset.ground_truth is not None
        assert dataset.axis_order is not None
        assert spec.equation == dataset.ground_truth
        assert tuple(spec.axes) == tuple(dataset.axis_order)

    @pytest.mark.unit
    def test_loader_callable_type_accepts_no_arg_loaders(self) -> None:
        loader: Callable[[], PDEDataset] = data.get_dataset("burgers").loader
        assert isinstance(loader(), PDEDataset)
