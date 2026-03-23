import pandas as pd

from .neural_network import ANN, random_data
import numpy as np
import torch
from torch.autograd import Variable
import os
import scipy.io as scio
from .neural_network import  *
from .train_gpt import *
import heapq
import matplotlib.pyplot as plt
import pickle
from .calculate_terms import *
import warnings
from ._device import device, DEVICE_STR, seed_all, load_checkpoint
from pathlib import Path as _Path
_DIR = _Path(__file__).parent
_REF_LIB_DIR = _DIR.resolve().parent.parent.parent / "ref_lib" / "EqGPT_wave_breaking"


warnings.filterwarnings('ignore')
# device set by _device.py

class PINNLossFunc(nn.Module):
    def __init__(self,h_data_choose):
        super(PINNLossFunc,self).__init__()
        self.h_data=h_data_choose
        return


    def forward(self,A,x):
        x=torch.from_numpy(x.astype(np.float32)).to(device)
        RHS =torch.matmul(A[:, 1:],x)
        LHS = A[:, 0]
        MSE = torch.mean((RHS - LHS) ** 2)
        return MSE



def Generate_meta_data(Net,Equation_name, choose, noise_level, trail_num, Load_state, x_low, x_up, t_low, t_up, nx=100,
                       nt=100, ):
    Net.load_state_dict(load_checkpoint(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl')))
    Net.eval()

    x = torch.linspace(x_low, x_up, nx)
    t = torch.linspace(t_low, t_up, nt)
    total = nx * nt

    num = 0
    data = torch.zeros(2)
    h_data = torch.zeros([total, 1])
    database = torch.zeros([total, 2])
    for j in range(nx):
        for i in range(nt):
            data[0] = x[j]
            data[1] = t[i]
            database[num] = data
            num += 1

    database = Variable(database, requires_grad=True).to(device)

    return Net, database

def get_coefficients(ulti_sentence,Net,database,variables,epoch):
    terms=ulti_sentence[::2]
    operators=ulti_sentence[1::2]

    A=[]
    A_column=1
    divide_flag=0
    for i in range(len(terms)):
        term=terms[i]
        operator=operators[i]
        word = id2word[term]
        value = calculate_terms(word, Net, database,variables).reshape(-1, )

        if divide_flag==0:
            A_column *= value
        else:
            A_column /= value
            divide_flag=0
        if operator==2:
            A.append(A_column)
            A_column=1
        elif operator==3:
            continue
        elif operator==4:
            divide_flag=1
        elif operator==1:
            A.append(A_column)



    A = np.vstack(A).T
    b = A[:, 0].copy()

    u, d, v = np.linalg.svd(A, full_matrices=False)
    x = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
    x=-x[1:]
    return x

def get_PDE_terms(ulti_sentence,Net,database,variables):
    terms=ulti_sentence[::2]
    operators=ulti_sentence[1::2]

    A=[]
    A_column=1
    divide_flag=0
    for i in range(len(terms)):
        term=terms[i]
        operator=operators[i]
        word = id2word[term]
        value = calculate_terms_PINN(word, Net, database,variables).reshape(-1, )
        if divide_flag==0:
            A_column *= value
        else:
            A_column /= value
            divide_flag=0
        if operator==2:
            A.append(A_column.reshape(-1,1))
            A_column=1
        elif operator==3:
            continue
        elif operator==4:
            divide_flag=1
        elif operator==1:
            A.append(A_column.reshape(-1,1))



    A = torch.concatenate(A,axis=1)
    return A
def get_mask_invalid(variables):
    mask_invalid = torch.ones(len(id2word)).to(device)
    if 't' not in variables:
        for i in range(len(id2word)):
            if 't' in id2word[i]:
                mask_invalid[i] = 0
    if 'x' not in variables:
        for i in range(len(id2word)):
            if 'x' in id2word[i]:
                mask_invalid[i] = 0

    if 'y' not in variables:
        for i in range(len(id2word)):
            if 'y' in id2word[i]:
                mask_invalid[i]=0
            if 'Laplace' in id2word[i]:
                mask_invalid[i]=0
            if 'BiLaplace' in id2word[i]:
                mask_invalid[i]=0
            if 'Div' in id2word[i]:
                mask_invalid[i]=0
    if 'z' not in variables:
        for i in range(len(id2word)):
            if 'z' in id2word[i]:
                mask_invalid[i] = 0
            if 'Div' in id2word[i]:
                mask_invalid[i]=0
    return mask_invalid

def train_PINN(Net):
    inputs = data[:,0:2]
    inputs[:,1]=inputs[:,1]-8

    outputs = data[:,2].reshape(-1,1)*100

    print("总样本数：", inputs.shape,outputs.shape)
    n_samples = inputs.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_train = 100000
    n_val = 5000

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    train_inputs = inputs[train_idx]
    train_outputs = outputs[train_idx]

    val_inputs = inputs[val_idx]
    val_outputs = outputs[val_idx]

    train_inputs_tensor = torch.from_numpy(train_inputs).float().to(device)
    train_inputs_tensor=torch.tensor(train_inputs_tensor,requires_grad=True)
    train_outputs_tensor = torch.from_numpy(train_outputs).float().to(device)

    val_inputs_tensor = torch.from_numpy(val_inputs).float().to(device)
    val_outputs_tensor = torch.from_numpy(val_outputs).float().to(device)


    best_epoch=np.load(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy'))[0]
    print(best_epoch)
    Net.load_state_dict(
        load_checkpoint(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/'+f"Net_{Activation_function}_{best_epoch}.pkl")))
    NN_optimizer = torch.optim.Adam([
        {'params': Net.parameters()},
    ])

    best_sentence_save = pickle.load(
        open(str(_DIR / f'result_save/{Equation_name}/{choose}_{noise_level}_{noise_type}/sentences.pkl'), 'rb'))
    best_sentence_save=best_sentence_save[4][0:3]

    variables = ['t', 'x']
    mask_invalid = get_mask_invalid(variables)
    for best_sentence in best_sentence_save:
        vis_sentence = [id2word[int(id)] for id in best_sentence]
        print("".join(vis_sentence[1:-1]))

        best_sentence.pop(0)
        MSELoss = torch.nn.MSELoss()
        PINNLoss= PINNLossFunc(train_inputs_tensor)
        print(f'===============train Net=================')
        for iter in tqdm(range(500)):
            NN_optimizer.zero_grad()
            prediction = Net(train_inputs_tensor)
            prediction_validate = Net(val_inputs_tensor).cpu().data.numpy()

            x=get_coefficients(best_sentence,Net,train_inputs_tensor,variables,iter)

            A=get_PDE_terms(best_sentence,Net,train_inputs_tensor,variables)
            loss_data = MSELoss(train_inputs_tensor, prediction)
            loss_PDE=PINNLoss(A,x)
            loss=loss_data+0.01*loss_PDE
            loss_validate = np.mean((val_outputs_tensor.cpu().data.numpy() - prediction_validate) ** 2)
            loss.backward()
            NN_optimizer.step()
            if iter==0:
                print(x)
            if (iter+1)%100==0:
                print(x)
                print(loss_data, loss_PDE)


if __name__=='__main__':
    #============Params=============
    Equation_name='wave_breaking'
    choose=90
    noise_level=0
    noise_type='Gaussian' #Gaussian or Uniform
    trail_num='L_G2Tp12A080_broad'
    Learning_Rate=0.001
    Activation_function='Sin' #'Tanh','Rational'
    #============Get origin data===========
    if Equation_name=='wave_breaking':
        data_dict=pickle.load(open(str(_REF_LIB_DIR / 'wave_breaking_data.pkl'), 'rb'))
        data=data_dict['L_G2Tp12A080_broad']
        Delete_equation_name=''

    #==========NN setting=============
    seed_all(525)
    Net = NN(Num_Hidden_Layers=6,
             Neurons_Per_Layer=60,
             Input_Dim=2,
             Output_Dim=1,
             Data_Type=torch.float32,
             Device=DEVICE_STR,
             Activation_Function=Activation_function,
             Batch_Norm=False)

    train_PINN(Net)
