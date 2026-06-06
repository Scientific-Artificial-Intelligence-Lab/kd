
from __future__ import annotations

from pathlib import Path
from typing import Final


SCHEMA_VERSION: Final[int] = 2


DEFAULT_REWARD_ALPHA: Final[float] = 0.01



N_ITERATIONS_1D: Final[int] = 200
BATCH_SIZE_1D: Final[int] = 500
EPSILON: Final[float] = 0.02
ENTROPY_WEIGHT: Final[float] = 0.03
GAMMA: Final[float] = 0.99
MAX_LENGTH_1D: Final[int] = 30
MIN_LENGTH_1D: Final[int] = 4
NUM_UNITS: Final[int] = 32
NUM_LAYERS: Final[int] = 1
EMBEDDING_DIM: Final[int] = 8


CHAFEE_ENTROPY_GAMMA: Final[float] = 0.7

BURGERS_OPERATORS: Final[tuple[str, ...]] = (
    "add", "mul", "sub", "div", "diff_x", "diff2_x",
)
CHAFEE_OPERATORS: Final[tuple[str, ...]] = (
    "add", "mul", "sub", "div", "diff_x", "diff2_x", "n2", "n3",
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
DEFAULT_BURGERS_DATA: Final[Path] = (
    _PROJECT_ROOT
    / "refs" / "discover" / "dso" / "dso" / "task" / "pde" / "data_new"
    / "burgers.mat"
)
DEFAULT_CHAFEE_DATA_DIR: Final[Path] = (
    _PROJECT_ROOT
    / "refs" / "discover" / "dso" / "dso" / "task" / "pde" / "data_new"
)
PROJECT_ROOT: Final[Path] = _PROJECT_ROOT


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


__all__ = [
    "BATCH_SIZE_1D",
    "BURGERS_OPERATORS",
    "CHAFEE_ENTROPY_GAMMA",
    "CHAFEE_OPERATORS",
    "DEFAULT_BURGERS_DATA",
    "DEFAULT_CHAFEE_DATA_DIR",
    "DEFAULT_REWARD_ALPHA",
    "EMBEDDING_DIM",
    "ENTROPY_WEIGHT",
    "EPSILON",
    "GAMMA",
    "MAX_LENGTH_1D",
    "MIN_LENGTH_1D",
    "N_ITERATIONS_1D",
    "NUM_LAYERS",
    "NUM_UNITS",
    "PROJECT_ROOT",
    "SCHEMA_VERSION",
    "project_relative",
    "resolve_project_path",
]
