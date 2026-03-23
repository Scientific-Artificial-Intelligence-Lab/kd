import pickle

import numpy as np
from .read_dataset import *
from .neural_network import  *
from .train_gpt import *
import re
from scipy.signal import hilbert
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from skimage import measure  # unused
from torch.utils.data import TensorDataset, DataLoader
from ._device import device, DEVICE_STR, seed_all, load_checkpoint
from pathlib import Path as _Path
_DIR = _Path(__file__).parent
_REF_LIB_DIR = _DIR.resolve().parent.parent.parent / "ref_lib" / "EqGPT_wave_breaking"
# device set by _device.py
# NOTE: `from pysr import PySRRegressor` is deferred to __main__ block
# to avoid native runtime crash (Julia + torch LLVM conflict, see common-mistakes.md #10)

def calculate_error(pred_concate,y_concate):
    pred_concate=np.array(pred_concate)
    y_concate=np.array(y_concate)

    RMSE=np.sqrt(np.mean((pred_concate-y_concate)**2))
    R2=1 - (((y_concate - pred_concate) ** 2).sum() / ((y_concate - y_concate.mean()) ** 2).sum())
    return RMSE,R2


if __name__ == '__main__':
    from pysr import PySRRegressor

    #============Params=============
    Equation_name='wave_breaking'
    choose=95
    noise_level=0
    noise_type='Gaussian' #Gaussian or Uniform

    Learning_Rate=0.001
    Activation_function='Sin' #'Tanh','Rational'
    #============Get origin data===========
    seed_all(525)
    Net = NN(Num_Hidden_Layers=6,
             Neurons_Per_Layer=60,
             Input_Dim=2,
             Output_Dim=1,
             Data_Type=torch.float32,
             Device=DEVICE_STR,
             Activation_Function=Activation_function,
             Batch_Norm=False)

    data_dict = pickle.load(open(str(_REF_LIB_DIR / 'wave_breaking_data.pkl'), 'rb'))
    case_name = list(data_dict.keys())
    all_var_N = []
    all_var_L=[]
    total=0
    for name in case_name:
        if 'N' in name:
            data = data_dict[name]
            total+=data.shape[0]
    print(total)

    for name in case_name:

        pattern = r'G(\d+)Tp(\d+)A(\d+)'
        match = re.search(pattern, name)
        g, tp, a = map(int, match.groups())
        tp = tp / 10
        lamda = 9.81 * (tp ** 2) / (2 * math.pi)
        data = data_dict[name]
        trail_num = name

        best_epoch = \
        np.load(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy'))[0]
        Load_state = 'Net_' + Activation_function + f'_{best_epoch}'
        Net.load_state_dict(
            load_checkpoint(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl')))

        Net.eval()

        x = np.concatenate([np.linspace(8.18, 9.34, 100),
                            np.linspace(9.77, 10.93, 100),
                            np.linspace(11.41, 12.57, 100)])

        dx = x[1] - x[0]

        x = (x - 8.17) / lamda
        t = np.arange(0.05 + 0.1, np.max(data[:, 0]) - 0.1, 0.05)
        t = t / tp
        dt = t[1] - t[0]
        nx = x.shape[0]
        nt = t.shape[0]
        T, X = np.meshgrid(t, x, indexing='ij')

        inputs = np.stack([T.ravel(), X.ravel()], axis=1)
        inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
        inputs = torch.tensor(inputs, requires_grad=True)

        h = Net(inputs)
        grad = torch.autograd.grad(
            h.sum(),
            inputs,
            create_graph=True
        )[0]
        ht = grad[:, 0].reshape(-1, 1)
        hx = grad[:, 1].reshape(-1, 1)
        hxx = torch.autograd.grad(hx.sum(), inputs, create_graph=True)[0][:, 1]
        hxxx = torch.autograd.grad(hxx.sum(), inputs, create_graph=True)[0][:, 1]
        hhx_x = torch.autograd.grad((h * hx).sum(), inputs, create_graph=True)[0][:, 1]
        hhx_xx = torch.autograd.grad(hhx_x.sum(), inputs, create_graph=True)[0][:, 1]


        ht = ht.cpu().data.numpy().reshape(-1, 1)
        hx = hx.cpu().data.numpy().reshape(-1, 1)
        hxxx = hxxx.cpu().data.numpy().reshape(-1, 1)
        hhx_xx =  hhx_xx.cpu().data.numpy().reshape(-1, 1)
        h = h.cpu().data.numpy().reshape(-1, 1)

        RHS = np.concatenate([hx,hxxx,hhx_xx], axis=1)
        LHS = ht
        coeffs, residuals, rank, s = np.linalg.lstsq(RHS, LHS, rcond=None)
        var=np.array([g, tp, a,-coeffs[0,0],-coeffs[1,0],-coeffs[2,0]])

        if 'N' in name:
            all_var_N.append(var)
        if 'L' in name:
            all_var_L.append(var)
    all_var_N=np.vstack(all_var_N)
    print('all_var_N:')
    print(all_var_N)
    raise OSError
    print('all_var_L:')
    all_var_L=np.vstack(all_var_L)
    print(all_var_L)
    raise OSError

    print(all_var)
    LHS=(all_var[:,1]**2/(all_var[:,2])).reshape(-1,1)
    RHS=(all_var[:,3]).reshape(-1,1)*0.939
    #print(np.linalg.lstsq(LHS,RHS))

    plt.scatter(LHS,RHS)
    plt.show()
    model = PySRRegressor(
                model_selection="best",
                niterations=500,
                binary_operators=["+", "*"],
                unary_operators=[
                    "inv(x) = 1/x",
                ],
                extra_sympy_mappings={"inv": lambda x: 1 / x},
                loss="loss(x, y) = (x - y)^2",
            )
    model.fit(all_var[:,0:4], all_var[:,4])
