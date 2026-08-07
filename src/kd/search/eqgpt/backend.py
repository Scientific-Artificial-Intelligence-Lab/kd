
from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol, cast, runtime_checkable

import torch
from torch import Tensor

from kd.search.eqgpt.gpt import EqGPT, GPTConfig, adapt_pretrained_state_dict


ASSET_ENV_VAR: Final[str] = "KD_EQGPT_ASSET_DIR"





DEFAULT_WEIGHTS_FILENAME: Final[str] = "PDEGPT_wave_breaking.pt"


@runtime_checkable
class GPTBackend(Protocol):

    def forward_logits(self, tokens: Tensor) -> Tensor:
        ...

    def next_token_logits(self, prefix: Sequence[int]) -> Tensor:
        ...

    def state_dict(self) -> dict[str, Tensor]:
        ...

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        ...

    def max_pos(self) -> int:
        ...


class RealGPTBackend:

    def __init__(
        self,
        model: EqGPT,
        device: torch.device | None = None,
        *,
        resolved_weights_path: Path | None = None,
    ) -> None:
        self.device = device
        self.model = model.to(device) if device is not None else model






        self.resolved_weights_path = resolved_weights_path

    @classmethod
    def from_assets(
        cls,
        config: GPTConfig,
        *,
        weights_path: Path | None = None,
        asset_dir: Path | None = None,
        device: torch.device | None = None,
    ) -> RealGPTBackend:
        path = resolve_asset_path(weights_path=weights_path, asset_dir=asset_dir)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = EqGPT(config)
        model.load_state_dict(adapt_pretrained_state_dict(state))
        model.eval()
        return cls(model, device=device, resolved_weights_path=path)

    def forward_logits(self, tokens: Tensor) -> Tensor:
        if self.device is not None:
            tokens = tokens.to(self.device)
        return cast(Tensor, self.model(tokens))

    def next_token_logits(self, prefix: Sequence[int]) -> Tensor:
        if not prefix:
            raise ValueError("next_token_logits requires a non-empty prefix")
        tokens = torch.tensor([list(prefix)], dtype=torch.long)
        with torch.no_grad():
            return self.forward_logits(tokens)[0, -1]

    def state_dict(self) -> dict[str, Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self.model.load_state_dict(state)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def max_pos(self) -> int:
        return self.model.config.max_pos


class FakeGPTBackend:

    def __init__(self, vocab_size: int, *, d_model: int = 16, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seed = seed





        config = GPTConfig(
            vocab_size=vocab_size,
            n_layers=2,
            n_heads=2,
            d_model=d_model,
            d_head=max(d_model // 2, 1),
            d_ff=d_model * 2,
            max_pos=256,
        )
        self.model = EqGPT(config)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for param in self.model.parameters():
                param.copy_(torch.randn(param.shape, generator=generator))
        self.model.eval()

    def forward_logits(self, tokens: Tensor) -> Tensor:
        return cast(Tensor, self.model(tokens))

    def next_token_logits(self, prefix: Sequence[int]) -> Tensor:
        if not prefix:
            raise ValueError("next_token_logits requires a non-empty prefix")
        tokens = torch.tensor([list(prefix)], dtype=torch.long)
        with torch.no_grad():
            return self.forward_logits(tokens)[0, -1]

    def state_dict(self) -> dict[str, Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self.model.load_state_dict(state)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def max_pos(self) -> int:
        return self.model.config.max_pos


def resolve_asset_path(
    *,
    weights_path: Path | None = None,
    asset_dir: Path | None = None,
    filename: str = DEFAULT_WEIGHTS_FILENAME,
) -> Path:
    if weights_path is not None:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"explicit weights_path does not exist: {weights_path}"
            )
        return weights_path

    source: Path
    label: str
    if asset_dir is not None:
        source, label = asset_dir, "asset_dir"
    else:
        env_value = os.environ.get(ASSET_ENV_VAR)
        if not env_value:
            raise FileNotFoundError(
                "pretrained EqGPT weights are not distributed with kd and no "
                f"asset source was given. Set {ASSET_ENV_VAR} to a directory "
                f"holding gpt_model/{filename}, or pass weights_path/asset_dir "
                "explicitly to RealGPTBackend.from_assets."
            )
        source, label = Path(env_value), f"${ASSET_ENV_VAR}"

    candidate = source / "gpt_model" / filename
    if not candidate.exists():
        raise FileNotFoundError(
            f"pretrained EqGPT weights not found at {candidate} (source: "
            f"{label}). Set {ASSET_ENV_VAR} to the asset directory, or pass "
            "weights_path/asset_dir explicitly to RealGPTBackend.from_assets."
        )
    return candidate
