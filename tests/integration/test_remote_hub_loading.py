from __future__ import annotations

from pathlib import Path

import pytest

from kd.data.remote import load_from_hub
from kd.data.schema import PDEDataset


@pytest.mark.slow
def test_load_from_hub_fetches_llm4ed_fisher(tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")

    try:
        dataset = load_from_hub("llm4ed-fisher", cache_dir=tmp_path / "hf-cache")
    except ValueError:
        raise
    except (
        ConnectionError,
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        TimeoutError,
    ) as exc:
        pytest.skip(f"Hugging Face Hub is unavailable in this environment: {exc}")

    assert isinstance(dataset, PDEDataset)
    assert dataset.name == "llm4ed-fisher"
    assert dataset.get_shape() == (201, 101)
    assert dataset.ground_truth == "u_t = 0.02*u_xx + 10*u*(1-u)"
