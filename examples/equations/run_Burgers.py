from kd.dataset import load_burgers_equation
from kd.model import KD_DSCV

burgers_dataset = load_burgers_equation()

burgers_dataset.describe()

model = KD_DSCV()

model.fit(burgers_dataset)

model.plot()


