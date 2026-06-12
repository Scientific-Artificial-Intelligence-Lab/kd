
import os

import pytest
import torch
from hypothesis import settings






settings.register_profile("ci", deadline=None)
if os.environ.get("CI"):
    settings.load_profile("ci")


@pytest.fixture
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    marker_map = {
        "unit": "unit",
        "integration": "integration",
        "equivalence": "equivalence",
        "validation": "validation",
    }

    for item in items:

        test_path = str(item.fspath)


        for directory, marker_name in marker_map.items():
            if f"/tests/{directory}/" in test_path:

                marker_names = {mark.name for mark in item.iter_markers()}
                if marker_name not in marker_names:
                    item.add_marker(getattr(pytest.mark, marker_name))
                break
