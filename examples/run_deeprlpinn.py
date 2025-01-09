import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

import scipy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='numpy.*')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.*')
from kd.model import DeepRL_Pinn

model = DeepRL_Pinn(
    n_samples_per_batch = 1000, # Number of generated traversals by agent per batch
    binary_operators = ["add_t", "mul_t", "div_t", "diff_t","diff2_t",],
    unary_operators = ['n2_t'],
)

np.random.seed(42)
def prepare_data():
    
    data = scipy.io.loadmat('./kd/data_file/burgers2.mat')
    t = np.real(data['t'].flatten()[:,None])
    x = np.real(data['x'].flatten()[:,None])
    Exact = np.real(data['usol']).T  # t first
    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0) 

    x_len = len(x)
    total_num = X_star.shape[0]
    sample_num = int(total_num*0.1)
    print(f"random sample number: {sample_num} ")
    ID = np.random.choice(total_num, sample_num, replace = False)
    X_u_meas = X_star[ID,:]
    u_meas = u_star[ID,:]  
    return X_u_meas,u_meas, lb,ub

x,y,lb,ub = prepare_data()

step_output = model.fit(x,y,[lb,ub],n_epochs=2)
print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")
model.plot(fig_type ='tree')
