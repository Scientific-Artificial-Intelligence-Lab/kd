import numpy as np

from kd.model.dlga import DLGA

X = 2 * np.random.RandomState(0).randn(100, 5)
y = np.cos(X[:, 3])
model = DLGA(epi=0.2, input_dim=5)
model.fit(X, y)
