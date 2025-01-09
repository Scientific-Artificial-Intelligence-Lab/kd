import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)
from kd.model.dlga import DLGA

X = 2 * np.random.RandomState(0).randn(100, 5)
y = np.cos(X[:, 3])
model = DLGA(epi=0.2, input_dim=5)
model.fit(X, y)
