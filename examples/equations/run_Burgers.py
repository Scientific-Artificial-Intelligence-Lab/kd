from kd.dataset import load_burgers_equation
from kd.model import DeepRL

burgers_dataset = load_burgers_equation()

burgers_dataset.describe()

model = DeepRL()

model.fit(burgers_dataset)

model.plot()


