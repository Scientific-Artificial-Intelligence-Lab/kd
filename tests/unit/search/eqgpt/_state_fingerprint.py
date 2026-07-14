
from __future__ import annotations

import hashlib
from typing import Any


def _backend_state_bytes(backend_state: dict[str, Any]) -> bytes:
    chunks: list[bytes] = []
    for key in sorted(backend_state):
        tensor = backend_state[key]
        chunks.append(key.encode())
        chunks.append(tensor.detach().cpu().contiguous().numpy().tobytes())
    return b"".join(chunks)


def weights_fingerprint(plugin: Any) -> str:
    h = hashlib.sha256()
    h.update(_backend_state_bytes(plugin.state.get("backend_state", {})))
    return h.hexdigest()


def state_fingerprint(plugin: Any) -> str:
    state = plugin.state
    h = hashlib.sha256()
    h.update(repr(plugin.best_score).encode())
    h.update(plugin.best_expression.encode())
    h.update(repr(state.get("reward_history")).encode())
    h.update(repr(state.get("top_k")).encode())
    h.update(repr(state.get("optimizer_state")).encode())
    h.update(repr(state.get("rng_state")).encode())
    h.update(_backend_state_bytes(state.get("backend_state", {})))
    return h.hexdigest()
