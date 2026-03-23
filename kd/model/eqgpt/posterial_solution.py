import numpy as np
import matplotlib.pyplot as plt
from .neural_network import *
import pickle
from ._device import device, DEVICE_STR, load_checkpoint
from pathlib import Path as _Path
_DIR = _Path(__file__).parent
_REF_LIB_DIR = _DIR.resolve().parent.parent.parent / "ref_lib" / "EqGPT_wave_breaking"

'''
1: ut+1.4473*ux=0
2: ut+t*sqrt(x)+u*sint+ux*uxx+t*ut
'''
def update_u(u,x,t,dx, dt, mode=1):
    if mode==1:
        c = 1.4473
        lam = c * dt / dx
        N = len(u)
        u_next = np.zeros_like(u)

        # 内部节点（中心差分 + Lax–Wendroff）
        for i in range(1, N - 1):
            u_next[i] = (
                    u[i]
                    - 0.5 * lam * (u[i + 1] - u[i - 1])
                    + 0.5 * lam ** 2 * (u[i + 1] - 2 * u[i] + u[i - 1])
            )

        # 左边界 i=0（使用前向差分，精度一阶）
        j = 0
        u_minus1 = u[1]  # Neumann 边界条件
        u_next[j] = (u[j]
                     - 0.5 * lam * (u[1] - u_minus1)
                     + 0.5 * lam ** 2 * (u[1] - 2 * u[j] + u_minus1))

        # 右边界 i=N-1（使用后向差分，i-2 项）
        i = N - 1
        if i >= 2:
            u_next[i] = (
                    u[i]
                    - 0.5 * lam * (u[i] - u[i - 2])  # 类似中心差分向左偏移
                    + 0.5 * lam ** 2 * (u[i] - 2 * u[i - 1] + u[i - 2])  # 类似二阶导数
            )
        else:
            # 长度不足三点时降阶处理
            u_next[i] = u[i]
        return u_next

    if mode==2:
        def compute_derivatives(u, dx):
            """
            四阶中心差分计算一阶和二阶导数
            """
            N = len(u)
            ux = np.zeros_like(u)
            uxx = np.zeros_like(u)

            for i in range(2, N - 2):
                ux[i] = (u[i - 2] - 8 * u[i - 1] + 8 * u[i + 1] - u[i + 2]) / (12 * dx)
                uxx[i] = (-u[i - 2] + 16 * u[i - 1] - 30 * u[i] + 16 * u[i + 1] - u[i + 2]) / (12 * dx)

            # 边界处理（用二阶中心差分，或Neumann/Dirichlet）
            ux[0] = (u[1] - u[0]) / dx
            ux[1] = (u[2] - u[0]) / (2 * dx)
            ux[-2] = (u[-1] - u[-3]) / (2 * dx)
            ux[-1] = (u[-1] - u[-2]) / dx

            uxx[0] = (u[2] - 2 * u[1] + u[0]) / dx ** 2
            uxx[1] = (u[2] - 2 * u[1] + u[0]) / dx ** 2
            uxx[-2] = (u[-1] - 2 * u[-2] + u[-3]) / dx ** 2
            uxx[-1] = (u[-1] - 2 * u[-2] + u[-3]) / dx ** 2

            return ux, uxx

        def next_step(u, x, dx, dt, t,
                      c1=3.64768384e-03, c2=-2.40237774e-02,
                      c3=3.00048670e-06, c4=-2.48945484e-01):
            """
            显式欧拉法 + 高阶差分，推进一时间步
            """
            ux, uxx = compute_derivatives(u, dx)
            sqrt_x = np.sqrt(x)
            sin_t = np.sin(t)
            denominator = 1 + c4 * t

            ut = (-c1 * t * sqrt_x
                  - c2 * u * sin_t
                  - c3 * ux * uxx) / denominator

            u_next = u + dt * ut
            return u_next
        return next_step(u, x, dx, dt, t)

    if mode==3:
        c1 = 2.33265299e-04
        c2 = 1.45908662e+00
        c3 = -2.56956037e-06
        def compute_derivatives(u, dx):
            N = len(u)
            ux = np.zeros_like(u)
            uxxx = np.zeros_like(u)

            for i in range(2, N - 2):
                ux[i] = (u[i - 2] - 8 * u[i - 1] + 8 * u[i + 1] - u[i + 2]) / (12 * dx)
                uxxx[i] = (-u[i - 2] + 2 * u[i - 1] - 2 * u[i + 1] + u[i + 2]) / (2 * dx ** 3)

            return ux, uxxx

        def L(u, dx, c1, c2, c3):
            ux, uxxx = compute_derivatives(u, dx)
            return -c1 * uxxx - c2 * ux - c3 * u ** 2 * uxxx

        def RK3_TVD(u0, dx, dt,c1, c2, c3):
            u = u0.copy()

            # 第一阶段
            L1 = L(u, dx, c1, c2, c3)
            u1 = u + dt * L1

            # 第二阶段
            L2 = L(u1, dx, c1, c2, c3)
            u2 = 0.75 * u + 0.25 * (u1 + dt * L2)

            # 第三阶段
            L3 = L(u2, dx, c1, c2, c3)
            u = (1 / 3) * u + (2 / 3) * (u2 + dt * L3)

            return u
        return RK3_TVD(u,dx,dt,c1,c2,c3)
def get_posterial(u_obs,x,dx):
    t = np.arange(0.05, np.max(data[:, 0]), 0.0005)
    dt=t[1]-t[0]
    print("use nt: ",t.shape[0])
    u_ini=u_obs[0]
    u_post=np.zeros([t.shape[0],u_obs.shape[1]])
    u_post[0]=u_ini
    for i in range(1,t.shape[0]):
        u_post[i]=update_u(u_post[i-1],x,t[i-1],dx,dt)
    return u_post
def get_meta_data(trail_num,data,x):
    Net = NN(Num_Hidden_Layers=6,
             Neurons_Per_Layer=60,
             Input_Dim=2,
             Output_Dim=1,
             Data_Type=torch.float32,
             Device=DEVICE_STR,
             Activation_Function='Sin',
             Batch_Norm=False)

    best_epoch = np.load(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}(Gaussian)/best_epoch.npy'))[
        0]
    Load_state = 'Net_' + 'Sin' + f'_{best_epoch}'

    Net.load_state_dict(
        load_checkpoint(str(_DIR / f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}(Gaussian)/{Load_state}.pkl')))

    Net.eval()

    t = np.arange(0.05, np.max(data[:, 0]), 0.05)
    nx = x.shape[0]
    nt = t.shape[0]
    # 利用meshgrid构造与S, A, B相同shape的坐标网格，使用 'ij' 索引顺序以保持 (nt, nx, ny, nz)
    T, X = np.meshgrid(t, x, indexing='ij')

    # 将所有坐标展平，并堆叠成 (N, 4) 的输入，其中 N = 20*48*48*48
    inputs = np.stack([T.ravel(), X.ravel()], axis=1)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    database = torch.tensor(inputs, requires_grad=True)
    pred=Net(inputs).cpu().data.numpy()
    return pred

if __name__ == '__main__':
    Equation_name = 'wave_breaking'
    # device set by _device.py

    choose=95
    noise_level=0

    data_dict = pickle.load(open(str(_REF_LIB_DIR / 'wave_breaking_data.pkl'), 'rb'))
    case_name = list(data_dict.keys())
    trail_num='L_G2Tp12A100_broad'
    data = data_dict[trail_num]

    nx =100
    x_1 = np.linspace(8.17, 9.34, nx)-8
    x_2= np.linspace(9.76, 10.93, nx)-8
    x_3=np.linspace(11.41, 12.55, nx)-8
    t = np.arange(0.05, np.max(data[:, 0]), 0.05)

    dt=t[1]-t[0]
    dx_1=x_1[1]-x_1[0]
    dx_2=x_2[1]-x_2[0]
    dx_3=x_3[1]-x_3[0]

    nt = t.shape[0]
    pred_1=get_meta_data(trail_num,data,x_1).reshape(nt,nx)
    pred_2=get_meta_data(trail_num,data,x_2).reshape(nt,nx)
    pred_3=get_meta_data(trail_num,data,x_3).reshape(nt,nx)

    post_2=get_posterial(pred_2,x_2,dx_2)
    for i in range(nt):
        plt.figure(figsize=(3, 2), dpi=300)
        plt.plot(x_2+8, pred_2[i], 'k-',lw=1.5, label='Obs')
        plt.plot(x_2+8, post_2[::100][i], 'r--',lw=1.5, label='Pred')
        plt.title(f't={round(t[i], 3)}', fontsize=7)
        plt.xlim([8, 13])
        plt.tight_layout()
        plt.show()

# plt.figure()
# plt.subplot(2,1,1)
# plt.imshow(pred_1,cmap='coolwarm')
# plt.axis('off')
# plt.colorbar(fraction=0.025)
# plt.tight_layout()
#
# plt.subplot(2,1,2)
# plt.imshow(post_1,cmap='coolwarm')
# plt.axis('off')
# plt.colorbar(fraction=0.025)
# plt.tight_layout()
#
# plt.show()