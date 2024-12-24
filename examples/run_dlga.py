import numpy as np

from kd.model.dlga import DLGA

X = 2 * np.random.RandomState(0).randn(100, 5)
y = np.cos(X[:, 3])
model = DLGA()
model.fit(X, y)


from kd.model import DLGA
from kd.model import DeepRL



model = DLGA()



