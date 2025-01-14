from kd.dataset import PDE
from kd.model import DeepRL

burgers_dataset = PDE('burgers')

burgers_dataset.describe()

model = DeepRL()

model.fit(burgers_dataset)

model.plot()


