
from __future__ import annotations

import pytest
import torch

from kd.search.eqgpt.gpt import (
    EqGPT,
    GPTConfig,
    adapt_pretrained_state_dict,
)
from tests.unit.search.eqgpt._helpers import (
    REAL_CONFIG,
    SMALL_CONFIG,
    build_external_state_dict,
    external_checkpoint_keys,
    source_fingerprints,
)






@pytest.mark.smoke
def test_gptconfig_defaults_match_pretrained_architecture() -> None:
    cfg = GPTConfig(vocab_size=57)
    assert (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff, cfg.max_pos) == (
        6,
        8,
        768,
        2048,
        50,
    )
    assert cfg.d_head == 64


def test_gptconfig_attn_inner_dim_is_heads_times_head() -> None:

    cfg = GPTConfig(vocab_size=57)
    assert cfg.attn_inner_dim == 8 * 64 == 512
    assert cfg.attn_inner_dim != cfg.d_model


def test_gptconfig_allows_dmodel_not_divisible_by_heads() -> None:

    cfg = GPTConfig(vocab_size=10, d_model=768, n_heads=5, d_head=64)
    assert cfg.attn_inner_dim == 320


def test_gptconfig_requires_vocab_size() -> None:
    with pytest.raises(TypeError):
        GPTConfig()


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize(
    "field",
    ["vocab_size", "n_layers", "n_heads", "d_model", "d_head", "d_ff", "max_pos"],
)
def test_gptconfig_rejects_nonpositive(field: str, value: int) -> None:
    kwargs: dict[str, int] = {"vocab_size": 57}
    kwargs[field] = value
    with pytest.raises(ValueError):
        GPTConfig(**kwargs)


def test_gptconfig_rejects_bool() -> None:

    with pytest.raises(ValueError):
        GPTConfig(vocab_size=True)







def test_eqgpt_realises_full_transformer_parameter_budget() -> None:
    model = EqGPT(REAL_CONFIG)
    total = sum(p.numel() for p in model.parameters())
    assert 20_000_000 < total < 45_000_000


def test_eqgpt_exposes_embedding_and_projection_shapes() -> None:
    model = EqGPT(REAL_CONFIG)
    shapes = {tuple(t.shape) for t in model.state_dict().values()}
    assert (REAL_CONFIG.vocab_size, REAL_CONFIG.d_model) in shapes
    assert (REAL_CONFIG.max_pos, REAL_CONFIG.d_model) in shapes
    assert (REAL_CONFIG.vocab_size,) in shapes


def test_forward_returns_batch_seq_vocab_logits() -> None:
    model = EqGPT(SMALL_CONFIG)
    tokens = torch.tensor([[5, 6, 2, 7, 1], [5, 7, 3, 6, 1]], dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (2, 5, SMALL_CONFIG.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_forward_output_depends_on_input() -> None:
    torch.manual_seed(0)
    model = EqGPT(SMALL_CONFIG)
    a = torch.tensor([[5, 6, 2, 7, 1]], dtype=torch.long)
    b = torch.tensor([[5, 7, 3, 6, 1]], dtype=torch.long)
    assert not torch.allclose(model(a), model(b))


def test_forward_attention_mixes_earlier_positions() -> None:
    torch.manual_seed(0)
    model = EqGPT(SMALL_CONFIG)
    base = torch.tensor([[5, 6, 2, 7, 3, 1]], dtype=torch.long)


    mutated = torch.tensor([[4, 6, 2, 7, 3, 1]], dtype=torch.long)
    j = 3
    assert not torch.allclose(model(base)[0, j], model(mutated)[0, j])


def test_forward_is_causal_earlier_positions_unaffected() -> None:
    torch.manual_seed(0)
    model = EqGPT(SMALL_CONFIG)
    tokens = torch.tensor([[5, 6, 2, 7, 3, 1]], dtype=torch.long)
    j = 3
    mutated = tokens.clone()
    mutated[0, j] = 4
    base = model(tokens)
    changed = model(mutated)
    torch.testing.assert_close(base[0, :j], changed[0, :j], rtol=0.0, atol=0.0)

    assert not torch.allclose(base[0, j], changed[0, j])


def test_forward_max_pos_boundary() -> None:
    torch.manual_seed(0)
    model = EqGPT(SMALL_CONFIG)
    at_limit = torch.zeros((1, SMALL_CONFIG.max_pos), dtype=torch.long)
    model(at_limit)
    over_limit = torch.zeros((1, SMALL_CONFIG.max_pos + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="max_pos"):
        model(over_limit)


def test_forward_batched_matches_per_sequence_with_padding() -> None:
    torch.manual_seed(0)
    model = EqGPT(SMALL_CONFIG)
    s1 = [5, 6, 2, 7, 1]
    s2 = [5, 7, 3, 1]
    width = 6
    batch = torch.tensor(
        [s1 + [0] * (width - len(s1)), s2 + [0] * (width - len(s2))],
        dtype=torch.long,
    )
    out = model(batch)
    u1 = model(torch.tensor([s1], dtype=torch.long))
    u2 = model(torch.tensor([s2], dtype=torch.long))
    torch.testing.assert_close(out[0,: len(s1)], u1[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out[1,: len(s2)], u2[0], rtol=1e-4, atol=1e-5)


def test_forward_pad_mask_leaves_real_positions_invariant() -> None:
    torch.manual_seed(0)
    model = EqGPT(SMALL_CONFIG)
    base = torch.tensor([[5, 6, 2, 7, 1]], dtype=torch.long)
    padded = torch.tensor([[5, 6, 2, 7, 1, 0, 0, 0]], dtype=torch.long)
    real = base.shape[1]
    torch.testing.assert_close(
        model(base)[0], model(padded)[0, :real], rtol=1e-4, atol=1e-5
    )


def test_forward_is_deterministic_across_fresh_instances() -> None:
    reference = EqGPT(SMALL_CONFIG)
    state = reference.state_dict()
    a = EqGPT(SMALL_CONFIG)
    a.load_state_dict(state)
    b = EqGPT(SMALL_CONFIG)
    b.load_state_dict(state)
    tokens = torch.tensor([[5, 6, 2, 7, 1]], dtype=torch.long)
    a.eval()
    b.eval()
    with torch.no_grad():
        torch.testing.assert_close(a(tokens), b(tokens), rtol=0.0, atol=0.0)







def test_adapter_gives_full_state_dict_coverage() -> None:
    external = build_external_state_dict(SMALL_CONFIG)
    adapted = adapt_pretrained_state_dict(external)
    model = EqGPT(SMALL_CONFIG)
    incompatible = model.load_state_dict(adapted, strict=False)
    assert list(incompatible.missing_keys) == []
    assert list(incompatible.unexpected_keys) == []


def test_adapter_is_pure_and_does_not_mutate_input() -> None:
    external = build_external_state_dict(SMALL_CONFIG)
    before = {k: v.clone() for k, v in external.items()}
    first = adapt_pretrained_state_dict(external)
    second = adapt_pretrained_state_dict(external)
    assert set(external) == set(before)
    for key, tensor in before.items():
        assert torch.equal(external[key], tensor)
    assert set(first) == set(second)


def test_adapter_preserves_tensor_values_by_passthrough() -> None:
    external = build_external_state_dict(SMALL_CONFIG)
    adapted = adapt_pretrained_state_dict(external)
    assert adapted, "adapter returned an empty state dict"
    source_fills = set(source_fingerprints(SMALL_CONFIG).values())
    seen: list[float] = []
    for tensor in adapted.values():
        unique = torch.unique(tensor)
        assert unique.numel() == 1, "adapter transformed a source tensor"
        fill = float(unique.item())
        assert fill in source_fills, "adapted tensor is not a source passthrough"
        seen.append(fill)
    assert len(set(seen)) == len(seen), "adapter duplicated a source tensor"


def test_adapter_reports_missing_or_raises_when_external_incomplete() -> None:
    external = build_external_state_dict(SMALL_CONFIG)


    live_key = "decoder.tgt_emb.weight"
    assert live_key in external_checkpoint_keys(SMALL_CONFIG)
    del external[live_key]
    try:
        adapted = adapt_pretrained_state_dict(external)
    except (KeyError, ValueError):
        return
    model = EqGPT(SMALL_CONFIG)
    incompatible = model.load_state_dict(adapted, strict=False)
    assert list(incompatible.missing_keys) != []
