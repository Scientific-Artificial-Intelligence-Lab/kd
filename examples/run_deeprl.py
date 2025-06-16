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
from kd.model import DeepRL
from kd.viz.discover_eq2latex import discover_program_to_latex 
from kd.viz.equation_renderer import render_latex_to_image

model = DeepRL(
    n_samples_per_batch = 500, # Number of generated traversals by agent per batch
    binary_operators = ['add',"mul", "diff","diff2"],
    unary_operators = ['n2'],
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
model.import_inner_data(dataset='Burgers', data_type='regular')
step_output = model.train(n_epochs=51)
print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")
# model.plot(fig_type ='tree').view()
# model.plot(fig_type='evolution')

# render_latex_to_image(discover_program_to_latex(step_output['program']))

# from log.deeprl_viz import plot_expression_tree

# plot_expression_tree(model)

