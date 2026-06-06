
import importlib

import pytest


@pytest.mark.smoke
def test_public_package_is_kd() -> None:
    kd = importlib.import_module("kd")
    assert hasattr(kd, "Model")


@pytest.mark.smoke
def test_old_kd2_name_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("kd2")
