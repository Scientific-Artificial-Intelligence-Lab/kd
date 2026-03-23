import pickle

import numpy as np
from .read_dataset import *
from .neural_network import  *
from .train_gpt import *
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

def calculate_error(pred_concate,y_concate):
    pred_concate=np.array(pred_concate)
    y_concate=np.array(y_concate)

    RMSE=np.sqrt(np.mean((pred_concate-y_concate)**2))
    R2=1 - (((y_concate - pred_concate) ** 2).sum() / ((y_concate - y_concate.mean()) ** 2).sum())
    return RMSE,R2


def train(Net,data):
    try:
        os.makedirs(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})'))
    except OSError:
        pass
    inputs = data[:,0:2]
    inputs[:,1]=inputs[:,1]-8

    # 同理，将S, A, B展平并堆叠成 (N, 3) 的输出数据
    #outputs = np.stack([S.ravel(), A.ravel(), B.ravel()], axis=1)
    outputs = data[:,2].reshape(-1,1)*100
    print(inputs)
    print(outputs)

    print("总样本数：", inputs.shape,outputs.shape)  # 应为20*48*48*48
    # 随机打乱样本顺序
    n_samples = inputs.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 划分数据集，80%训练，10%验证，10%测试
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



    # 将模型转移到GPU上
    Net.to(device)
    NN_optimizer = torch.optim.Adam([
        {'params': Net.parameters()},
    ])
    MSELoss = torch.nn.MSELoss()
    validate_error = []
    print(f'===============train Net=================')
    file = open(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/loss.txt'), 'w').close()
    file = open(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/loss.txt'), "a+")
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
                       str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/' + f"Net_{Activation_function}_{iter + 1}.pkl"))
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

    # x = np.concatenate([np.linspace(8.17, 9.34, 100),
    #                     np.linspace(9.76, 10.93, 100),
    #                     np.linspace(11.41, 12.55, 100)])
    x = np.concatenate([np.linspace(8.17, 9.34, 100),
                        np.linspace(9.76, 10.93, 100),
                        np.linspace(11.40, 12.57, 100)])

    x = x - 8
    t = np.arange(0.05+0.1, np.max(data[:, 0])-0.1, 0.05)
    nx = x.shape[0]
    nt = t.shape[0]
    # 利用meshgrid构造与S, A, B相同shape的坐标网格，使用 'ij' 索引顺序以保持 (nt, nx, ny, nz)
    T, X = np.meshgrid(t, x, indexing='ij')

    # 将所有坐标展平，并堆叠成 (N, 4) 的输入，其中 N = 20*48*48*48
    inputs = np.stack([T.ravel(), X.ravel()], axis=1)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    database = torch.tensor(inputs, requires_grad=True)



    return Net,database,nx,nt

def test_error():
    Net= NN(Num_Hidden_Layers=6,
               Neurons_Per_Layer=60,
               Input_Dim=2,
               Output_Dim=1,
               Data_Type=torch.float32,
               Device=DEVICE_STR,
               Activation_Function=Activation_function,
               Batch_Norm=False)


    best_epoch = np.load(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/best_epoch.npy'))[0]
    Load_state = 'Net_' + Activation_function + f'_{best_epoch}'
    print(best_epoch)
    Net.load_state_dict(
        load_checkpoint(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl')))

    Net.eval()
    '''
    x = np.concatenate([np.linspace(8.1516,9.36, 100),
                        np.linspace(9.74,10.95,100),
                        np.linspace(11.392,12.574,100)])
    '''
    x = np.concatenate([np.linspace(8.17,9.34, 100),
                        np.linspace(9.76,10.93,100),
                        np.linspace(11.40,12.57,100)])

    dx=x[1]-x[0]

    x=x-8
    t = np.arange(0.05+0.1, np.max(data[:,0])-0.1, 0.05)
    dt=t[1]-t[0]
    nx=x.shape[0]
    nt=t.shape[0]
    # 利用meshgrid构造与S, A, B相同shape的坐标网格，使用 'ij' 索引顺序以保持 (nt, nx, ny, nz)
    T, X = np.meshgrid(t, x, indexing='ij')

    # 将所有坐标展平，并堆叠成 (N, 4) 的输入，其中 N = 20*48*48*48
    inputs = np.stack([T.ravel(), X.ravel()], axis=1)
    inputs=torch.from_numpy(inputs.astype(np.float32)).to(device)
    inputs=torch.tensor(inputs,requires_grad=True)

    pred=Net(inputs).reshape([t.shape[0],x.shape[0]]).cpu().data.numpy()

    # Plotting functions omitted from module-level execution
    return pred

if __name__=='__main__':
    #============Params=============
    Equation_name='wave_breaking'
    choose=95
    noise_level=0
    noise_type='Non_unit' #Gaussian or Uniform or Non_unit
    trail_num='N_G2Tp12A100_broad'
    Learning_Rate=0.001
    Activation_function='Sin' #'Tanh','Rational'
    #============Get origin data===========
    if Equation_name=='wave_breaking':
        # 读取数据
        data_dict=pickle.load(open(str(_REF_LIB_DIR / 'wave_breaking_data.pkl'), 'rb'))
        data=data_dict[trail_num]
        Delete_equation_name=''

    #====================Train GPT without target equation========================
    '''
    This is very important to keep the generative model unseen underlying equations for proof-of-concept
    '''
    if os.path.exists(str(_DIR / f'gpt_model/PDEGPT_{Equation_name}.pt'))==False:
        train_num_data = get_train_dataset(Equation_name=Delete_equation_name)
        batch_size = 128
        epochs = 100
        dataset = MyDataSet(train_num_data)

        data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)

        model = GPT().to(device)

        train(model, data_loader,Equation_name=Equation_name)


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

    #train(Net,data)
    test_error()
    #get_meta()
