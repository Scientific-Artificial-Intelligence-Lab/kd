
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Int32Array = npt.NDArray[np.int32]
Float32Array = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Batch:

    actions: Int32Array
    obs: Float32Array
    priors: Float32Array
    lengths: Int32Array


__all__ = ["Batch"]
