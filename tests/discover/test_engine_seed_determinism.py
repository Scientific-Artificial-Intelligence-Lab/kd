
from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from kd.search.discover.config import DiscoverConfig
from kd.search.discover.plugin import DISCOVERPlugin








_RNG_SAMPLE_SHAPE = (4,)



_SEED_A = 7
_SEED_B = 13






_BASELINE_SEED = 999_999





_EXTERNAL_PRIOR_SEED = 42







@pytest.fixture(autouse=True)
def _isolate_torch_rng() -> Iterator[None]:
    state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(state)







@pytest.mark.unit
def test_seed_default_zero() -> None:
    config = DiscoverConfig()
    assert config.seed == 0


@pytest.mark.unit
def test_seed_zero_is_valid() -> None:

    config = DiscoverConfig(seed=0)
    assert config.seed == 0


@pytest.mark.unit
def test_seed_positive_is_valid() -> None:
    config = DiscoverConfig(seed=_SEED_A)
    assert config.seed == _SEED_A







@pytest.mark.unit
def test_seed_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"seed"):
        DiscoverConfig(seed=-1)


@pytest.mark.unit
def test_seed_large_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"seed"):
        DiscoverConfig(seed=-12345)







@pytest.mark.unit
def test_seed_propagates_to_torch_rng() -> None:
    torch.manual_seed(_BASELINE_SEED)
    DISCOVERPlugin(DiscoverConfig(seed=_SEED_A))
    draw_first = torch.rand(_RNG_SAMPLE_SHAPE)

    torch.manual_seed(_BASELINE_SEED)
    DISCOVERPlugin(DiscoverConfig(seed=_SEED_A))
    draw_second = torch.rand(_RNG_SAMPLE_SHAPE)

    torch.testing.assert_close(draw_first, draw_second, rtol=0.0, atol=0.0)


@pytest.mark.unit
def test_different_seeds_yield_different_rng() -> None:
    torch.manual_seed(_BASELINE_SEED)
    DISCOVERPlugin(DiscoverConfig(seed=_SEED_A))
    draw_a = torch.rand(_RNG_SAMPLE_SHAPE)

    torch.manual_seed(_BASELINE_SEED)
    DISCOVERPlugin(DiscoverConfig(seed=_SEED_B))
    draw_b = torch.rand(_RNG_SAMPLE_SHAPE)


    assert not torch.equal(draw_a, draw_b)


@pytest.mark.unit
def test_default_plugin_uses_seed_zero() -> None:
    torch.manual_seed(_BASELINE_SEED)
    DISCOVERPlugin()
    draw_from_default = torch.rand(_RNG_SAMPLE_SHAPE)

    torch.manual_seed(0)
    draw_from_explicit_seed_zero = torch.rand(_RNG_SAMPLE_SHAPE)

    torch.testing.assert_close(
        draw_from_default, draw_from_explicit_seed_zero, rtol=0.0, atol=0.0
    )







@pytest.mark.unit
def test_init_leaves_rng_at_post_seed_state() -> None:
    DISCOVERPlugin(DiscoverConfig(seed=_SEED_A))
    after_plugin = torch.get_rng_state()

    torch.manual_seed(_SEED_A)
    after_direct = torch.get_rng_state()

    assert torch.equal(after_plugin, after_direct)







@pytest.mark.unit
def test_plugin_seed_overrides_prior_global_manual_seed() -> None:
    torch.manual_seed(_EXTERNAL_PRIOR_SEED)
    DISCOVERPlugin(DiscoverConfig(seed=_SEED_A))
    after_plugin = torch.get_rng_state()

    torch.manual_seed(_SEED_A)
    after_direct = torch.get_rng_state()

    assert torch.equal(after_plugin, after_direct), (
        "Plugin __init__ must override prior torch.manual_seed; "
        "this is intentional per facade design (config.seed is the sole entry)."
    )
