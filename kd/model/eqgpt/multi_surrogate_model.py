import pickle
import re
import numpy as np
from .read_dataset import *
from .neural_network import  *
from .train_gpt import *
import heapq
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from skimage import measure  # unused
from torch.utils.data import TensorDataset, DataLoader
from ._device import device, DEVICE_STR, seed_all, load_checkpoint
from pathlib import Path as _Path
_DIR = _Path(__file__).parent
_REF_LIB_DIR = _DIR.resolve().parent.parent.parent / "ref_lib" / "EqGPT_wave_breaking"
# device set by _device.py

def calculate_error(pred_concate,y_concate):
    pred_concate=np.array(pred_concate)
    y_concate=np.array(y_concate)

    RMSE=np.sqrt(np.mean((pred_concate-y_concate)**2))
    R2=1 - (((y_concate - pred_concate) ** 2).sum() / ((y_concate - y_concate.mean()) ** 2).sum())
    return RMSE,R2

def train(data,Equation_name,choose,noise_level,trail_num,noise_type):
    # ==========NN setting=============
    seed_all(525)
    Net = NN(Num_Hidden_Layers=6,
             Neurons_Per_Layer=60,
             Input_Dim=2,
             Output_Dim=1,
             Data_Type=torch.float32,
             Device=DEVICE_STR,
             Activation_Function=Activation_function,
             Batch_Norm=False)

    try:
        os.makedirs(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})'))
    except OSError:
        pass
    pattern = r'G(\d+)Tp(\d+)A(\d+)'
    match = re.search(pattern, trail_num)
    g, tp, a = map(int, match.groups())
    tp=tp/10
    lamda=9.81*(tp**2)/(2*math.pi)
    print('lamda:',lamda,'Tp',tp)
    inputs = data[:,0:2]
    inputs[:,0]=inputs[:,0]/tp
    inputs[:,1]=(inputs[:,1]-8.17)/lamda

    outputs = data[:,2].reshape(-1,1)/lamda*100

    print("总样本数：", inputs.shape,outputs.shape)
    n_samples = inputs.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_train = int(choose/100 * n_samples)
    n_val = int(0.05 * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    train_inputs = inputs[train_idx]
    train_outputs = outputs[train_idx]

    val_inputs = inputs[val_idx]
    val_outputs = outputs[val_idx]

    train_inputs_tensor = torch.from_numpy(train_inputs).float().to(device)
    train_outputs_tensor = torch.from_numpy(train_outputs).float().to(device)

    val_inputs_tensor = torch.from_numpy(val_inputs).float().to(device)
    val_outputs_tensor = torch.from_numpy(val_outputs).float().to(device)

    Net.to(device)
    NN_optimizer = torch.optim.Adam([
        {'params': Net.parameters()},
    ])
    MSELoss = torch.nn.MSELoss()
    validate_error = []
    print(f'===============train Net=================')
    _loss_path = str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/loss.txt')
    file = open(_loss_path, 'w').close()
    file = open(_loss_path, "a+")
    for iter in tqdm(range(50000)):
        NN_optimizer.zero_grad()
        prediction = Net(train_inputs_tensor)
        prediction_validate = Net(val_inputs_tensor).cpu().data.numpy()
        loss = MSELoss(train_outputs_tensor, prediction)
        loss_validate = np.mean((val_outputs_tensor.cpu().data.numpy() - prediction_validate) ** 2)
        loss.backward()
        NN_optimizer.step()

        if (iter + 1) % 500 == 0:
            validate_error.append(loss_validate)
            torch.save(Net.state_dict(),
                       str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})' / f"Net_{Activation_function}_{iter + 1}.pkl"))
            print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (iter + 1, loss, loss_validate))
            file.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \n" % (iter + 1, loss, loss_validate))
    file.close()
    best_epoch = (validate_error.index(min(validate_error)) + 1) * 500
    print(best_epoch)
    np.save(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy'),
            np.array([best_epoch]))

def get_meta():
    Net = NN(Num_Hidden_Layers=6,
             Neurons_Per_Layer=60,
             Input_Dim=2,
             Output_Dim=1,
             Data_Type=torch.float32,
             Device=DEVICE_STR,
             Activation_Function=Activation_function,
             Batch_Norm=False)

    best_epoch = np.load(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy'))[
        0]
    Load_state = 'Net_' + Activation_function + f'_{best_epoch}'

    Net.load_state_dict(
        load_checkpoint(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl')))

    Net.eval()

    x = np.arange(np.min(data[:, 1]) - 8+0.02, np.max(data[:, 1]) - 8-0.02, 0.02)
    t = np.arange(0.1, np.max(data[:, 0]), 0.05)
    nx = x.shape[0]
    nt = t.shape[0]
    T, X = np.meshgrid(t, x, indexing='ij')

    inputs = np.stack([T.ravel(), X.ravel()], axis=1)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    database = torch.tensor(inputs, requires_grad=True)
    return Net,database

def test_error():
    pass  # Full implementation in __main__ block

if __name__=='__main__':
    # ============Params=============
    Equation_name = 'wave_breaking'
    choose = 95
    noise_level = 0
    noise_type = 'Non_unit'  # Gaussian or Uniform
    Delete_equation_name = ''
    Learning_Rate = 0.001
    Activation_function = 'Sin'  # 'Tanh','Rational'
    # ============Get origin data===========

    data_dict = pickle.load(open(str(_REF_LIB_DIR / 'wave_breaking_data.pkl'), 'rb'))
    case_name = list(data_dict.keys())

    for name in case_name:
        print(f'==========Now training case:  {name}!==========')
        data = data_dict[name]
        trail_num=name
        train(data,Equation_name,choose,noise_level,trail_num,noise_type)
