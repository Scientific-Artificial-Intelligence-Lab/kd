
from __future__ import annotations

from collections.abc import Iterator
from unittest import mock

import pytest

from kd.search.discover.pinn.model import (
    _EMPTY_CACHE_EVERY_N_CHUNKS,
    _EMPTY_CACHE_PRESSURE_THRESHOLD,
    _maybe_empty_cache,
)






_TOTAL_MEMORY_BYTES = 40 * 1024**3
_HIGH_PRESSURE_RATIO = 0.80
_LOW_PRESSURE_RATIO = 0.50
_HIGH_ALLOCATED_BYTES = int(_TOTAL_MEMORY_BYTES * _HIGH_PRESSURE_RATIO)
_LOW_ALLOCATED_BYTES = int(_TOTAL_MEMORY_BYTES * _LOW_PRESSURE_RATIO)
_DUMMY_DEVICE_INDEX = 0


class _FakeDeviceProperties:

    def __init__(self, total_memory: int) -> None:
        self.total_memory = total_memory


class _CudaMockBundle:

    def __init__(
        self,
        *,
        is_available: mock.MagicMock,
        current_device: mock.MagicMock,
        memory_allocated: mock.MagicMock,
        get_device_properties: mock.MagicMock,
        empty_cache: mock.MagicMock,
    ) -> None:
        self.is_available = is_available
        self.current_device = current_device
        self.memory_allocated = memory_allocated
        self.get_device_properties = get_device_properties
        self.empty_cache = empty_cache


def _patch_cuda(
    *,
    is_available: bool = True,
    allocated_bytes: int = _LOW_ALLOCATED_BYTES,
    total_bytes: int = _TOTAL_MEMORY_BYTES,
    memory_allocated_side_effect: Exception | None = None,
) -> Iterator[_CudaMockBundle]:
    from kd.search.discover.pinn import model as model_module






    model_cuda = model_module.torch.cuda

    mem_mock = mock.MagicMock(return_value=allocated_bytes)
    if memory_allocated_side_effect is not None:
        mem_mock.side_effect = memory_allocated_side_effect

    with (
        mock.patch.object(
            model_cuda,
            "is_available",
            mock.MagicMock(return_value=is_available),
        ) as is_avail_mock,
        mock.patch.object(
            model_cuda,
            "current_device",
            mock.MagicMock(return_value=_DUMMY_DEVICE_INDEX),
        ) as cur_dev_mock,
        mock.patch.object(
            model_cuda, "memory_allocated", mem_mock
        ) as mem_alloc_mock,
        mock.patch.object(
            model_cuda,
            "get_device_properties",
            mock.MagicMock(return_value=_FakeDeviceProperties(total_bytes)),
        ) as props_mock,
        mock.patch.object(
            model_cuda, "empty_cache", mock.MagicMock()
        ) as empty_mock,
    ):
        yield _CudaMockBundle(
            is_available=is_avail_mock,
            current_device=cur_dev_mock,
            memory_allocated=mem_alloc_mock,
            get_device_properties=props_mock,
            empty_cache=empty_mock,
        )


@pytest.fixture
def cuda_high_pressure() -> Iterator[_CudaMockBundle]:
    yield from _patch_cuda(
        is_available=True, allocated_bytes=_HIGH_ALLOCATED_BYTES
    )


@pytest.fixture
def cuda_low_pressure() -> Iterator[_CudaMockBundle]:
    yield from _patch_cuda(
        is_available=True, allocated_bytes=_LOW_ALLOCATED_BYTES
    )


@pytest.fixture
def cuda_unavailable() -> Iterator[_CudaMockBundle]:
    yield from _patch_cuda(is_available=False)


@pytest.fixture
def cuda_api_raises() -> Iterator[_CudaMockBundle]:
    yield from _patch_cuda(
        is_available=True,
        memory_allocated_side_effect=RuntimeError("driver hiccup"),
    )







class TestMaybeEmptyCacheConstants:

    @pytest.mark.unit
    def test_cadence_is_sixteen_chunks(self) -> None:
        assert _EMPTY_CACHE_EVERY_N_CHUNKS == 16

    @pytest.mark.unit
    def test_pressure_threshold_is_seventy_five_percent(self) -> None:
        assert pytest.approx(0.75) == _EMPTY_CACHE_PRESSURE_THRESHOLD







class TestMaybeEmptyCacheGuard:

    @pytest.mark.unit
    def test_no_call_on_chunk_0(
        self, cuda_high_pressure: _CudaMockBundle
    ) -> None:
        _maybe_empty_cache(0)
        cuda_high_pressure.empty_cache.assert_not_called()


        cuda_high_pressure.memory_allocated.assert_not_called()

    @pytest.mark.unit
    def test_no_call_on_non_multiple_of_interval(
        self, cuda_high_pressure: _CudaMockBundle
    ) -> None:
        for idx in (4, 8, 15, 17, 31):
            _maybe_empty_cache(idx)
        cuda_high_pressure.empty_cache.assert_not_called()
        cuda_high_pressure.memory_allocated.assert_not_called()

    @pytest.mark.unit
    def test_no_call_below_pressure_threshold(
        self, cuda_low_pressure: _CudaMockBundle
    ) -> None:
        cadence_aligned = (
            _EMPTY_CACHE_EVERY_N_CHUNKS,
            2 * _EMPTY_CACHE_EVERY_N_CHUNKS,
            3 * _EMPTY_CACHE_EVERY_N_CHUNKS,
            4 * _EMPTY_CACHE_EVERY_N_CHUNKS,
        )
        for idx in cadence_aligned:
            _maybe_empty_cache(idx)


        assert cuda_low_pressure.memory_allocated.call_count == len(
            cadence_aligned
        )
        cuda_low_pressure.empty_cache.assert_not_called()

    @pytest.mark.unit
    def test_call_on_threshold_and_interval_both_met(
        self, cuda_high_pressure: _CudaMockBundle
    ) -> None:
        _maybe_empty_cache(_EMPTY_CACHE_EVERY_N_CHUNKS)
        cuda_high_pressure.empty_cache.assert_called_once()

    @pytest.mark.unit
    def test_noop_when_cuda_unavailable(
        self, cuda_unavailable: _CudaMockBundle
    ) -> None:



        _maybe_empty_cache(_EMPTY_CACHE_EVERY_N_CHUNKS)
        cuda_unavailable.memory_allocated.assert_not_called()
        cuda_unavailable.get_device_properties.assert_not_called()
        cuda_unavailable.empty_cache.assert_not_called()








@pytest.mark.unit
def test_runtime_error_in_cuda_api_is_swallowed(
    cuda_api_raises: _CudaMockBundle,
) -> None:

    _maybe_empty_cache(_EMPTY_CACHE_EVERY_N_CHUNKS)
    cuda_api_raises.empty_cache.assert_not_called()
