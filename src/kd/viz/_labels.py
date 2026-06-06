
from __future__ import annotations

_ALGORITHM_SGA = "sga"
_ALGORITHM_DLGA = "dlga"
_ALGORITHM_DISCOVER = "discover"
_AIC_LABEL = "AIC"
_DLGA_FITNESS_LABEL = "DLGA fitness"
_REWARD_LABEL = "reward"
_DEFAULT_SCORE_LABEL = "Score"


def score_label(algorithm: str) -> str:
    if algorithm == _ALGORITHM_SGA:
        return _AIC_LABEL
    if algorithm == _ALGORITHM_DLGA:
        return _DLGA_FITNESS_LABEL
    if algorithm == _ALGORITHM_DISCOVER:
        return _REWARD_LABEL
    return _DEFAULT_SCORE_LABEL
