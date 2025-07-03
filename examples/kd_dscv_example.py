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
from kd.model import KD_DSCV
from kd.viz.dscv_viz import *
from kd.viz.discover_eq2latex import discover_program_to_latex 
from kd.viz.equation_renderer import render_latex_to_image

model = KD_DSCV(
    n_samples_per_batch = 500, # Number of generated traversals by agent per batch
    binary_operators = ['add',"mul", "diff","diff2"],
    unary_operators = ['n2'],
)

np.random.seed(42)

model.import_inner_data(dataset='Burgers', data_type='regular')

step_output = model.train(n_epochs=51)
print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")


render_latex_to_image(discover_program_to_latex(step_output['program']))

plot_expression_tree(model)

plot_density(model)
plot_density(model, epoches = [10,30,50])

plot_evolution(model)

plot_pde_residual_analysis(model, step_output['program'])

plot_field_comparison(model, step_output['program'])

plot_actual_vs_predicted(model, step_output['program'])