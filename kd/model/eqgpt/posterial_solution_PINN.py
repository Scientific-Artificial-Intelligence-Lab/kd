import torch
import torch.nn as nn
import numpy as np
from .neural_network import *
import pickle
from .calculate_terms import *
from ._device import device, DEVICE_STR, load_checkpoint
from pathlib import Path as _Path
_DIR = _Path(__file__).parent
_REF_LIB_DIR = _DIR.resolve().parent.parent.parent / "ref_lib" / "EqGPT_wave_breaking"
# 设置精度和设备
torch.set_default_dtype(torch.float32)
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
    #A =torch.hstack((A, torch.ones_like(A[:, 0].reshape([-1, 1]))))
    return A

def find_boundary(data):
    time_values = np.arange(0.05, np.max(data[:,0]), 0.05)

    # 输出列表
    results = []

    for t_val in time_values:
        # 筛选当前时间点对应的所有行
        rows_at_t = data[np.isclose(data[:, 0], t_val, atol=1e-6)]
        if rows_at_t.size == 0:
            continue  # 若当前时间点没有匹配行，则跳过
        # 找到 x 最小的那一行（即第二列最小）
        min_x_row = rows_at_t[np.argmin(rows_at_t[:, 1])]
        results.append(min_x_row)

    # 合并为一个数组
    result_array = np.array(results)
    return result_array
# -----------------------------
# 网络结构定义
# -----------------------------
if __name__ == '__main__':
    Net = NN(Num_Hidden_Layers=6,
             Neurons_Per_Layer=60,
             Input_Dim=2,
             Output_Dim=1,
             Data_Type=torch.float32,
             Device=DEVICE_STR,
             Activation_Function='Sin',
             Batch_Norm=False)

    Equation_name = 'wave_breaking'

    noise_level=0

    data_dict = pickle.load(open(str(_REF_LIB_DIR / 'wave_breaking_data.pkl'), 'rb'))
case_name = list(data_dict.keys())
trail_num='L_G2Tp12A100_broad'
time='04131601'
MODE='Valid'
variables=['t','x']
data = data_dict[trail_num]

best_sentence = [13, 2, 9, 2, 12, 1]
best_coef = np.array([1.90414566, 2.83632603])

# best_sentence=[6, 2, 8, 3, 19, 2, 8, 3, 9, 2, 8, 2, 20, 4, 22,1]
# best_coef=np.array([ 2.30834359e-03, -2.61111774e-05,  1.43628329e+00,  1.99019172e-04])

# best_sentence=[6, 2, 20, 2, 7, 3, 18, 2, 7, 3, 10, 4, 51, 2, 8, 1]
# best_coef=np.array([4.85482200e-04, 4.84654799e-08, 1.70272802e-04, 1.46685891e+00])

# best_sentence=[6, 2, 8,1]
# best_coef=np.array([1.44730417])

vis_sentence = [id2word[int(id)] for id in [1]+best_sentence]
vis_sentence="".join(vis_sentence[1:-1])
print(vis_sentence)

model = Net.to(device)
ini_data=data[data[:, 0]<0.555]
#ini_data=data[np.isclose(data[:, 0], 0.05)]
bound_data=find_boundary(data)
inputs =np.concatenate([ini_data[:, 0:2],bound_data[:,0:2]],axis=0)
inputs[:, 1] = inputs[:, 1] - 8
outputs =np.concatenate([ini_data[:, 2].reshape(-1, 1), bound_data[:, 2].reshape(-1, 1)],axis=0)* 100

inputs=torch.from_numpy(inputs.astype(np.float32)).to(device)
outputs=torch.from_numpy(outputs.astype(np.float32)).to(device)

N_f = 300 # 采样点数量
x_f = np.concatenate([
    np.linspace(8.17, 9.34, 100),
    np.linspace(9.76, 10.93, 100),
    np.linspace(11.41, 12.55, 100)
])
x_f=x_f-8
t_f=np.arange(0.05, np.max(data[:, 0]), 0.05)
T, X = np.meshgrid(t_f, x_f, indexing='ij')
meta_inputs = np.stack([T.ravel(), X.ravel()], axis=1)
meta_inputs=torch.from_numpy(meta_inputs.astype(np.float32))
meta_inputs=torch.tensor(meta_inputs,requires_grad=True).to(device)

lr=0.001
epochs=50000
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
PINNLoss = PINNLossFunc(meta_inputs)


if MODE=='Train':
    try:
        os.makedirs(f'model_save_PINN/{Equation_name}/{trail_num}_{time}/')
    except OSError:
        pass
    file = open(f'model_save_PINN/{Equation_name}/{trail_num}_{time}/loss.txt', 'w').close()
    file = open(f'model_save_PINN/{Equation_name}/{trail_num}_{time}/loss.txt', "a+")
    file.write("Now the equation is:  " + vis_sentence + '\n')
    file.write(f"Now the sentence is:  {best_sentence}" + '\n')
    file.write(f"Now the coef is:  {best_coef}" + '\n')
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        u_pred = model(inputs)
        loss_ic = torch.mean((u_pred - outputs) ** 2)

        A = get_PDE_terms(best_sentence, Net, meta_inputs, variables)
        loss_pde = PINNLoss(A,best_coef)

        #loss = loss_ic+ 0.01*loss_pde
        loss = loss_ic# + 0.001 * loss_pde
        loss.backward()
        optimizer.step()
        if (epoch+1) % 500 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4e}, IC: {loss_ic.item():.4e}, PDE: {loss_pde.item():.4e}")
            torch.save(Net.state_dict(),
                       f'model_save_PINN/{Equation_name}/{trail_num}_{time}/' + f"Net_{epoch + 1}.pkl")
            file.write("iter_num: %d      loss: %.8f    loss_IC: %.8f  loss_PDE: %.8f\n" % (epoch+ 1, loss.item(),loss_ic.item(), loss_pde.item()))

if MODE=='Valid':
    Net = NN(Num_Hidden_Layers=6,
             Neurons_Per_Layer=60,
             Input_Dim=2,
             Output_Dim=1,
             Data_Type=torch.float32,
             Device=DEVICE_STR,
             Activation_Function='Sin',
             Batch_Norm=False)
    best_epoch=13000
    Load_state = f'Net_{best_epoch}'

    Net.load_state_dict(
        load_checkpoint(f'model_save_PINN/{Equation_name}/{trail_num}_{time}/{Load_state}.pkl'))

    Net.eval()

    x = np.concatenate([np.linspace(8.17, 9.34, 100),
                        np.linspace(9.76, 10.93, 100),
                        np.linspace(11.41, 12.55, 100)])
    x = x - 8
    t = np.arange(0.05, np.max(data[:, 0]), 0.05)
    nx = x.shape[0]
    nt = t.shape[0]
    # 利用meshgrid构造与S, A, B相同shape的坐标网格，使用 'ij' 索引顺序以保持 (nt, nx, ny, nz)
    T, X = np.meshgrid(t, x, indexing='ij')

    # 将所有坐标展平，并堆叠成 (N, 4) 的输入，其中 N = 20*48*48*48
    inputs = np.stack([T.ravel(), X.ravel()], axis=1)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    database = torch.tensor(inputs, requires_grad=True)
    pred = Net(database).reshape([t.shape[0], x.shape[0]]).cpu().data.numpy()

    def plot_pred():
        plt.figure(figsize=(2.5, 2.5), dpi=300)
        plt.imshow(pred,cmap='coolwarm')
        plt.axis('off')
        plt.colorbar(fraction=0.025)
        plt.tight_layout()
        plt.show()


        for i in range(t.shape[0]):
            true=data[np.isclose(data[:, 0], t[i])]
            font_settings = {'family': 'Arial', 'size': 7}
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["font.size"] = 7
            plt.figure(figsize=(3, 2),dpi=300)
            plt.scatter(true[:, 1], true[:, 2] * 100, c='black', marker='^',s=1, label='Obs')
            #plt.plot(true[:, 1], true[:, 2]*100, 'k-',lw=1.5, label='Obs')
            plt.scatter(x+8, pred[i], c='red', marker='*', s=1, label='pred')
            #plt.plot(x+8, pred[i], 'r--',lw=1.5, label='Pred')
            plt.title(f't={round(t[i],3)}',fontsize=7)
            plt.xlim([8,13])
            plt.xlabel("x", fontdict=font_settings)
            plt.ylabel("h", fontdict=font_settings)
            plt.xticks(fontsize=7, fontname='Arial')
            plt.yticks(fontsize=7, fontname='Arial')
            plt.legend(prop={'family': 'Arial', 'size': 6}, loc='best')
            plt.tight_layout()
            # 显示图像
            plt.show()
        return 0
    plot_pred()