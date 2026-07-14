
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any



VOCAB_MASK_VERSION: str = "1"


def resolve_weights_fingerprint_path(
    backend: Any, configured_path: Path | None
) -> Path | None:
    resolved = getattr(backend, "resolved_weights_path", None)
    return resolved if resolved is not None else configured_path


def _file_fingerprint(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"asset not found for fingerprinting: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as asset:
        for chunk in iter(lambda: asset.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size": path.stat().st_size,
    }


def build_run_artifacts(
    *,
    vocab_path: Path,
    weights_path: Path | None = None,
    corpus_path: Path | None = None,
    variables: tuple[str, ...] = (),
    surrogate_paths: dict[str, Path] | None = None,
    wave_pkl_path: Path | None = None,
    grid_params: dict[str, int] | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "vocab": _file_fingerprint(Path(vocab_path)),
        "vocab_mask_version": VOCAB_MASK_VERSION,
        "variables": list(variables),
    }
    if weights_path is not None:
        artifacts["weights"] = _file_fingerprint(Path(weights_path))
    if corpus_path is not None:
        artifacts["corpus"] = _file_fingerprint(Path(corpus_path))
    if surrogate_paths is not None:
        artifacts["surrogates"] = {
            name: _file_fingerprint(Path(p))
            for name, p in sorted(surrogate_paths.items())
        }
    if wave_pkl_path is not None:
        artifacts["wave_data"] = _file_fingerprint(Path(wave_pkl_path))
    if grid_params is not None:
        artifacts["grids"] = dict(grid_params)
    return artifacts
