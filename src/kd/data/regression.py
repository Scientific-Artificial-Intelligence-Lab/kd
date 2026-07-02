"""Bundled real-world experimental datasets in tabular form.

These datasets are plain ``y = f(X)`` tables — measured data, not solver
output. They feed kd's standalone symbolic-regression entry points (PySR
free-form search, SINDy basis regression) and surrogate-based workflows,
and are separate from the PDE catalog: no regular grid, no derivatives,
no ``PDEDataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

_TLC_CC_FILES = {
    "start": "tlc_cc_Rf_t1.npy",
    "end": "tlc_cc_Rf_t2.npy",
}
_TLC_CC_TARGETS = {"start": "V_S", "end": "V_E"}
_TLC_CC_VAR_NAMES = ("R_F", "r")
_TLC_CC_FEATURE_COLS = (0, 1)
_TLC_CC_TARGET_COL = 2
_TLC_CC_SOURCE = (
    "Xu, H., Wu, W., Chen, Y., Zhang, D. & Mo, F. Explicit relation between "
    "thin film chromatography and column chromatography conditions from "
    "statistics and machine learning. Nat Commun 16, 832 (2025). "
    "https://doi.org/10.1038/s41467-025-56136-x"
)


@dataclass(frozen=True)
class TabularDataset:
    """A scalar regression dataset: features ``X``, target ``y``, metadata.

    Attributes:
        X: Feature matrix shaped ``(n_samples, n_features)``.
        y: Target vector shaped ``(n_samples,)``.
        var_names: Feature names, one per column of ``X``.
        target_name: Name of the target quantity ``y``.
        name: Stable dataset identifier.
        source: Citation for where the data comes from.
        description: What the rows and columns physically mean.
    """

    X: np.ndarray
    y: np.ndarray
    var_names: tuple[str, ...]
    target_name: str
    name: str
    source: str
    description: str


def _assets_data_dir() -> Path:
    """Return the bundled ``_assets/data`` directory of the kd package."""
    return Path(__file__).resolve().parents[1] / "_assets" / "data"


def load_tlc_cc(target: Literal["start", "end"] = "start") -> TabularDataset:
    """Load the TLC-CC chromatography dataset (real-world experimental data).

    The data comes from a real, automated column chromatography (CC)
    platform: 192 organic compounds separated on 4 g silica columns, with
    thin-layer chromatography (TLC) runs providing the retardation factor
    for each compound/eluent pair (Xu et al., Nat Commun 16, 832, 2025).
    Each of the 74 rows is one experimental condition:

    - ``X[:, 0]`` -- ``R_F``: TLC retardation factor of the compound.
    - ``X[:, 1]`` -- ``r``: petroleum-ether fraction of the eluent,
      ``r = V_PE / (V_PE + V_EA)``.
    - ``y`` -- mean retention volume (in mL) at that condition:
      ``V_S`` (``target="start"``, volume when the compound is first
      detected) or ``V_E`` (``target="end"``, volume when it has fully
      eluted).

    The task is to find an explicit formula for the retention volume from
    ``R_F`` and ``r`` -- the relation chemists otherwise carry as tacit
    experience. See the paper and its Supplementary Information for the
    experimental protocol and dataset background.

    Args:
        target: Which retention volume to use as ``y``: ``"start"``
            (``V_S``) or ``"end"`` (``V_E``).

    Returns:
        The dataset with ``X = (R_F, r)`` and the chosen retention volume
        as ``y``.

    Raises:
        ValueError: If ``target`` is not ``"start"`` or ``"end"``.
        FileNotFoundError: If the bundled data file is missing.
    """
    if target not in _TLC_CC_FILES:
        raise ValueError(f"target must be 'start' (V_S) or 'end' (V_E), got {target!r}")
    path = _assets_data_dir() / _TLC_CC_FILES[target]
    if not path.exists():
        raise FileNotFoundError(f"bundled TLC-CC data file not found: {path}")
    raw = np.load(path)
    features = np.ascontiguousarray(raw[:, list(_TLC_CC_FEATURE_COLS)])
    target_values = np.ascontiguousarray(raw[:, _TLC_CC_TARGET_COL])
    target_name = _TLC_CC_TARGETS[target]
    return TabularDataset(
        X=features,
        y=target_values,
        var_names=_TLC_CC_VAR_NAMES,
        target_name=target_name,
        name=f"tlc-cc-{target}",
        source=_TLC_CC_SOURCE,
        description=(
            "Real-world column chromatography measurements: mean "
            f"{target_name} retention volume on 4 g silica columns over 74 "
            "(R_F, r) experimental conditions, aggregated from automated "
            "runs of 192 compounds."
        ),
    )


_WAVE_BREAKING_BUNDLED_CASE = "N_G2Tp12A100_broad"
_WAVE_BREAKING_VAR_NAMES = ("t", "x")
_WAVE_BREAKING_TARGET = "eta"
_WAVE_BREAKING_SOURCE = (
    "Xu, H., Chen, Y., Cao, R., Tang, T., Du, M., Li, J., Callaghan, A. H. & "
    "Zhang, D. Generative discovery of partial differential equations by "
    "learning from math handbooks. Nat Commun 16, 10255 (2025). "
    "https://doi.org/10.1038/s41467-025-65114-2"
)


def _wave_breaking_path(case: str, data_dir: Path | str | None) -> Path:
    """Resolve the ``.npz`` file for a wave-breaking case, fail loud."""
    filename = f"wave_breaking_{case}.npz"
    if data_dir is not None:
        explicit = Path(data_dir) / filename
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"wave-breaking data file not found: {explicit}")
    bundled = _assets_data_dir() / filename
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        f"wave-breaking case {case!r} is not bundled with kd (bundled case: "
        f"{_WAVE_BREAKING_BUNDLED_CASE!r}). For the other experiments of the "
        "campaign, pass data_dir=<directory with wave_breaking_<case>.npz "
        "files>."
    )


def load_wave_breaking(
    case: str = _WAVE_BREAKING_BUNDLED_CASE,
    data_dir: Path | str | None = None,
) -> TabularDataset:
    """Load wave-tank surface-elevation data (real-world experimental data).

    The data are laboratory measurements of unidirectional focused wave
    groups evolving toward breaking, from the BUBER and EURUS campaigns in
    the 27.2 m glass-walled wave tank at Imperial College London (Xu et
    al., Nat Commun 16, 10255, 2025). Wave groups with JONSWAP-type spectra
    were generated by a bottom-hinged paddle; three CCD cameras (20 Hz)
    observed roughly 8 m to 12.5 m of the tank, and the air-water interface
    was reconstructed frame by frame:

    - ``X[:, 0]`` -- ``t``: time (s) of the camera frame.
    - ``X[:, 1]`` -- ``x``: position (m) along the tank; the three camera
      windows do not overlap, so ``x`` covers three disjoint intervals.
    - ``y`` -- ``eta``: measured surface elevation (m) at ``(t, x)``.

    Points are scattered (no regular space-time grid), so the dataset is
    exposed as a table rather than a ``PDEDataset``. From these campaigns
    the paper discovered a previously unreported governing equation for the
    approach to breaking; the experimental setting is documented in the
    paper's Supplementary Information.

    One of the paper's 12 analysed experiments is bundled with kd: case
    ``"N_G2Tp12A100_broad"`` (314,478 points; peak enhancement factor 2,
    peak period 1.2 s, amplitude sum 100 mm). Case names encode these
    JONSWAP paddle parameters.

    Args:
        case: Experiment identifier. Defaults to the bundled case.
        data_dir: Directory holding ``wave_breaking_<case>.npz`` files for
            non-bundled cases. When given, it is authoritative.

    Returns:
        The dataset with ``X = (t, x)`` and surface elevation as ``y``.

    Raises:
        FileNotFoundError: If the case is not bundled and ``data_dir`` does
            not provide it.
    """
    path = _wave_breaking_path(case, data_dir)
    with np.load(path) as npz:
        t = np.asarray(npz["t"], dtype=np.float64)
        x = np.asarray(npz["x"], dtype=np.float64)
        eta = np.asarray(npz["eta"], dtype=np.float64)
    features = np.ascontiguousarray(np.column_stack([t, x]))
    return TabularDataset(
        X=features,
        y=eta,
        var_names=_WAVE_BREAKING_VAR_NAMES,
        target_name=_WAVE_BREAKING_TARGET,
        name=f"wave-breaking-{case}",
        source=_WAVE_BREAKING_SOURCE,
        description=(
            "Real-world wave-tank measurements: camera-reconstructed surface "
            f"elevation eta(t, x) of experiment {case} ({eta.shape[0]} "
            "points), a focused wave group propagating toward breaking "
            "(Imperial College London wave tank, BUBER/EURUS campaigns)."
        ),
    )


__all__ = ["TabularDataset", "load_tlc_cc", "load_wave_breaking"]
