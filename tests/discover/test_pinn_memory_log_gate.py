
from __future__ import annotations

import importlib
import logging
from types import ModuleType
from unittest import mock

import pytest

from kd.search.discover.pinn import _memory_log as memlog_module






_ENV_FLAG = "DISCOVER_LOG_MEMORY"
_ENV_ON = "1"
_DUMMY_ALLOC_BYTES = 1_000_000_000


def _reload_memlog_with_env(env_overrides: dict[str, str]) -> ModuleType:
    with mock.patch.dict("os.environ", env_overrides, clear=False):
        return importlib.reload(memlog_module)







class TestDefaultDisabled:

    @pytest.mark.unit
    def test_flag_is_false_when_env_unset(self) -> None:


        env_without_flag = {
            k: v for k, v in __import__("os").environ.items() if k != _ENV_FLAG
        }
        with mock.patch.dict("os.environ", env_without_flag, clear=True):
            reloaded = importlib.reload(memlog_module)
            assert reloaded._LOG_MEMORY_ENABLED is False

        importlib.reload(memlog_module)

    @pytest.mark.unit
    def test_log_memory_default_noop_does_not_touch_logger(self) -> None:
        env_without_flag = {
            k: v for k, v in __import__("os").environ.items() if k != _ENV_FLAG
        }
        with mock.patch.dict("os.environ", env_without_flag, clear=True):
            reloaded = importlib.reload(memlog_module)
            fake_logger = mock.MagicMock(spec=logging.Logger)


            with mock.patch.object(
                reloaded.torch.cuda,
                "is_available",
                mock.MagicMock(return_value=True),
            ) as is_avail_mock:
                reloaded._log_memory("should_not_log", fake_logger)
            fake_logger.info.assert_not_called()


            is_avail_mock.assert_not_called()
        importlib.reload(memlog_module)








class TestEnvVarSemantics:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "env_value",
        ["0", "true", "false", "yes", "TRUE", "", " 1 ", "on"],
    )
    def test_non_one_values_are_off(self, env_value: str) -> None:
        reloaded = _reload_memlog_with_env({_ENV_FLAG: env_value})
        assert reloaded._LOG_MEMORY_ENABLED is False
        importlib.reload(memlog_module)

    @pytest.mark.unit
    def test_exact_one_enables_the_hook(self) -> None:
        reloaded = _reload_memlog_with_env({_ENV_FLAG: _ENV_ON})
        assert reloaded._LOG_MEMORY_ENABLED is True
        importlib.reload(memlog_module)









class TestEnabledPath:

    @pytest.mark.unit
    def test_log_memory_emits_info_when_enabled(self) -> None:
        reloaded = _reload_memlog_with_env({_ENV_FLAG: _ENV_ON})
        fake_logger = mock.MagicMock(spec=logging.Logger)
        with (
            mock.patch.object(
                reloaded.torch.cuda,
                "is_available",
                mock.MagicMock(return_value=True),
            ),
            mock.patch.object(
                reloaded.torch.cuda,
                "memory_allocated",
                mock.MagicMock(return_value=_DUMMY_ALLOC_BYTES),
            ),
            mock.patch.object(
                reloaded.torch.cuda,
                "max_memory_allocated",
                mock.MagicMock(return_value=_DUMMY_ALLOC_BYTES),
            ),
            mock.patch.object(
                reloaded.torch.cuda,
                "memory_reserved",
                mock.MagicMock(return_value=_DUMMY_ALLOC_BYTES),
            ),
        ):
            reloaded._log_memory("cotrain_epoch_7_start", fake_logger)
        fake_logger.info.assert_called_once()
        call_args = fake_logger.info.call_args


        assert call_args.args[0].startswith("[MEM]"), call_args

        assert call_args.args[1] == "cotrain_epoch_7_start"
        importlib.reload(memlog_module)

    @pytest.mark.unit
    def test_log_memory_noop_when_cuda_unavailable_even_if_enabled(
        self,
    ) -> None:
        reloaded = _reload_memlog_with_env({_ENV_FLAG: _ENV_ON})
        fake_logger = mock.MagicMock(spec=logging.Logger)
        with mock.patch.object(
            reloaded.torch.cuda,
            "is_available",
            mock.MagicMock(return_value=False),
        ):
            reloaded._log_memory("anywhere", fake_logger)
        fake_logger.info.assert_not_called()
        importlib.reload(memlog_module)

    @pytest.mark.unit
    def test_runtime_error_in_cuda_api_is_swallowed(self) -> None:
        reloaded = _reload_memlog_with_env({_ENV_FLAG: _ENV_ON})
        fake_logger = mock.MagicMock(spec=logging.Logger)
        with (
            mock.patch.object(
                reloaded.torch.cuda,
                "is_available",
                mock.MagicMock(return_value=True),
            ),
            mock.patch.object(
                reloaded.torch.cuda,
                "memory_allocated",
                mock.MagicMock(side_effect=RuntimeError("driver hiccup")),
            ),
        ):

            reloaded._log_memory("recovered", fake_logger)


        fake_logger.info.assert_not_called()
        importlib.reload(memlog_module)
