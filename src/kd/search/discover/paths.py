
from pathlib import Path
from typing import Final

__all__ = [
    "BASELINE_RESULTS_DIR",
    "GOLDEN_BASELINE_DIR",
    "PARITY_ORACLE_DIR",
    "PROJECT_ROOT",
    "REFERENCE_DATA_DIR",
]


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]

REFS_ROOT: Final[Path] = PROJECT_ROOT / "refs"



REFERENCE_DATA_DIR: Final[Path] = (
    REFS_ROOT / "discover" / "dso" / "dso" / "task" / "pde" / "data_new"
)


GOLDEN_BASELINE_DIR: Final[Path] = REFS_ROOT / "baseline" / "golden"


PARITY_ORACLE_DIR: Final[Path] = REFS_ROOT / "parity" / "phase3"


BASELINE_RESULTS_DIR: Final[Path] = REFS_ROOT / "baseline" / "results"
