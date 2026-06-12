
from __future__ import annotations

_ALGORITHM_SGA = "sga"
_ALGORITHM_DLGA = "dlga"
_ALGORITHM_DISCOVER = "discover"
_ALGORITHM_PYSR = "pysr"
_AIC_LABEL = "AIC"
_DLGA_FITNESS_LABEL = "DLGA fitness"
_REWARD_LABEL = "reward"
_NMSE_LABEL = "NMSE"
_DEFAULT_SCORE_LABEL = "Score"
_DIRECTION_MIN = "min"
_DIRECTION_MAX = "max"

_LEGACY_DIRECTION_BY_ALGORITHM = {
    _ALGORITHM_SGA: _DIRECTION_MIN,
    _ALGORITHM_DLGA: _DIRECTION_MIN,
    _ALGORITHM_DISCOVER: _DIRECTION_MAX,
    _ALGORITHM_PYSR: _DIRECTION_MIN,
}


def score_label(algorithm: str) -> str:
    if algorithm == _ALGORITHM_SGA:
        return _AIC_LABEL
    if algorithm == _ALGORITHM_DLGA:
        return _DLGA_FITNESS_LABEL
    if algorithm == _ALGORITHM_DISCOVER:
        return _REWARD_LABEL
    if algorithm == _ALGORITHM_PYSR:
        return _NMSE_LABEL
    return _DEFAULT_SCORE_LABEL


def score_direction(algorithm: str) -> str:
    return _LEGACY_DIRECTION_BY_ALGORITHM.get(algorithm, _DIRECTION_MIN)
