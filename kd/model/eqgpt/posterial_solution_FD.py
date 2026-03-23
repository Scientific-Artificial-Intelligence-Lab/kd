import numpy as np
import matplotlib.pyplot as plt
from .neural_network import *
import pickle
from scipy.ndimage import gaussian_filter1d
from ._device import device, DEVICE_STR, load_checkpoint
from pathlib import Path as _Path
_DIR = _Path(__file__).parent
_REF_LIB_DIR = _DIR.resolve().parent.parent.parent / "ref_lib" / "EqGPT_wave_breaking"


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

    t = np.arange(0.05 + 0.1, np.max(data[:, 0]) - 0.1, 0.05)
    nx = x.shape[0]
    nt = t.shape[0]
    # 利用meshgrid构造与S, A, B相同shape的坐标网格，使用 'ij' 索引顺序以保持 (nt, nx, ny, nz)
    T, X = np.meshgrid(t, x, indexing='ij')

    # 将所有坐标展平，并堆叠成 (N, 4) 的输入，其中 N = 20*48*48*48
    inputs = np.stack([T.ravel(), X.ravel()], axis=1)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    database = torch.tensor(inputs, requires_grad=True)
    return Net,database


if __name__ == '__main__':
    Equation_name = 'wave_breaking'
    # device set by _device.py

    choose=95
    noise_level=0

    data_dict = pickle.load(open(str(_REF_LIB_DIR / 'wave_breaking_data.pkl'), 'rb'))
    case_name = list(data_dict.keys())
    trail_num='L_G2Tp13A090_broad'
    data = data_dict[trail_num]

    x = np.concatenate([np.linspace(8.17, 9.34, 100),
                        np.linspace(9.76, 10.93, 100),
                        np.linspace(11.40, 12.57, 100)])

    dx = x[1] - x[0]

    x = x - 8
    t = np.arange(0.05 + 0.1, np.max(data[:, 0]) - 0.1, 0.05)
    dt = t[1] - t[0]
    nx = x.shape[0]
    nt = t.shape[0]


    Net,database=get_meta_data(trail_num,data,x)
    h= Net(database).cpu().data.numpy().reshape(nt,nx)
    hx_FD=np.concatenate([np.gradient(h[:,0:100],dx,axis=1),
                                  np.gradient(h[:,100:200], dx, axis=1),
                                  np.gradient(h[:,200:300], dx, axis=1)
                                  ],axis=1)
    hxx_FD = np.concatenate([np.gradient(hx_FD[:, 0:100], dx, axis=1),
                            np.gradient(hx_FD[:, 100:200], dx, axis=1),
                            np.gradient(hx_FD[:, 200:300], dx, axis=1)
                            ], axis=1)




    start_step=20
    forward_step=5
    equation='L'
    if equation=='N':
        h_ini = h[start_step]
        h_post = np.zeros([forward_step, h.shape[1]])
        h_post[0] = h_ini
        for i in range(1, forward_step):
            hx_FD = np.concatenate([np.gradient(h_post[:, 0:100], dx, axis=1),
                                    np.gradient(h_post[:, 100:200], dx, axis=1),
                                    np.gradient(h_post[:, 200:300], dx, axis=1)
                                    ], axis=1)
            h_post[i] = h_post[i-1] - 1.4212958 * hx_FD[i-1] * dt
            plt.figure(1, figsize=(10, 10))
            plt.plot(x + 8, h.reshape([nt, nx])[start_step+i-1], c='blue', ls='--', lw=5)
            plt.plot(x + 8, h.reshape([nt, nx])[start_step+i], c='black', ls='--', lw=5)
            plt.plot(x + 8, h_post[i], c='red', ls='--', lw=5)
            plt.title(f't={t[i]}')
            plt.show()
    if equation=='L':
        h_ini = h[start_step]
        h_post = np.zeros([forward_step, h.shape[1]])
        h_post[0] = h_ini
        for i in range(1, forward_step):
            hx_FD = np.concatenate([np.gradient(h_post[:, 0:100], dx, axis=1),
                                    np.gradient(h_post[:, 100:200], dx, axis=1),
                                    np.gradient(h_post[:, 200:300], dx, axis=1)
                                    ], axis=1)
            h_post[i] = h_post[i - 1] - 1.7585981 * hx_FD[i - 1] * dt-0.13398351*np.sinh(h)[i-1]
            plt.figure(1, figsize=(10, 10))
            plt.plot(x + 8, h.reshape([nt, nx])[start_step + i - 1], c='blue', ls='--', lw=5)
            plt.plot(x + 8, h.reshape([nt, nx])[start_step + i], c='black', ls='--', lw=5)
            plt.plot(x + 8, h_post[i], c='red', ls='--', lw=5)
            plt.title(f't={t[i]}')
            plt.show()
    if equation=='ux+ut=0':
        def minmod(a, b):
            return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

        def compute_flux(u, c):
            return c * u  # 线性对流

        def update_u_TVD(u,x,t, dx, dt, u_L, u_R):
            c=1.4182
            N = len(u)
            flux = compute_flux(u, c)

            # Compute limited slopes (minmod limiter)
            du_forward = np.roll(u, -1) - u
            du_backward = u - np.roll(u, 1)
            slope = minmod(du_forward, du_backward)

            # MUSCL 左右值
            u_LR = u - 0.5 * slope
            u_RL = u + 0.5 * slope

            # Numerical flux (Rusanov / local Lax-Friedrichs)
            flux_half = np.zeros(N + 1)
            for i in range(1, N):
                u_left = u_RL[i - 1]
                u_right = u_LR[i]
                flux_L = compute_flux(u_left, c)
                flux_R = compute_flux(u_right, c)
                alpha = np.abs(c)
                flux_half[i] = 0.5 * (flux_L + flux_R) - 0.5 * alpha * (u_right - u_left)

            # TVD-RK2 第一阶段
            u1 = u.copy()
            u1[1:-1] -= dt / dx * (flux_half[2:-1] - flux_half[1:-2])

            # 强制边界条件
            u1[0] = u_L
            u1[-1] = u_R

            # 第二阶段需重新构造 MUSCL
            flux = compute_flux(u1, c)
            du_forward = np.roll(u1, -1) - u1
            du_backward = u1 - np.roll(u1, 1)
            slope = minmod(du_forward, du_backward)
            u_LR = u1 - 0.5 * slope
            u_RL = u1 + 0.5 * slope

            for i in range(1, N):
                u_left = u_RL[i - 1]
                u_right = u_LR[i]
                flux_L = compute_flux(u_left, c)
                flux_R = compute_flux(u_right, c)
                alpha = np.abs(c)
                flux_half[i] = 0.5 * (flux_L + flux_R) - 0.5 * alpha * (u_right - u_left)

            u2 = np.zeros_like(u)
            u2[1:-1] = 0.5 * (u[1:-1] + u1[1:-1] - dt / dx * (flux_half[2:-1] - flux_half[1:-2]))
            u_next = np.zeros_like(u)
            u_next[1:-1] = u2[1:-1]

            # 强制 Dirichlet 边界条件
            u_next[0] = u_L
            u_next[-1] = u_R
            return u_next

        def update_u(u,x,t ,dx, dt,  u_L, u_R):
            c = 1.4182
            lam = c * dt / dx
            N = len(u)
            u_next = np.zeros_like(u)

            # Lax–Friedrichs 内部节点更新
            for i in range(1, N - 1):
                u_next[i] = 0.5 * (u[i + 1] + u[i - 1]) - 0.5 * lam * (u[i + 1] - u[i - 1])

            # 强制 Dirichlet 边界条件
            u_next[0] = u_L
            u_next[N - 1] = u_R

            return u_next
        def compute_residual(u, x, t):
            """
            U: (nt, nx) 数值解矩阵
            x: 空间坐标 (nx,)
            t: 时间坐标 (nt,)
            c: 波速常数
            """
            # 计算导数
            c=1.4182
            dt=t[1]-t[0]
            dx=x[1]-x[0]
            ut= np.gradient(u,dt,axis=0)
            ux= np.gradient(u,dx,axis=1)
            # 计算方程残差 R = ut + c * ux
            residual = ut + c * ux
            l2_error = np.sqrt(np.mean(residual[2:-2,2:-2] ** 2))
            return l2_error

        def get_posterial(u_obs,x,dx,start_step=start_step,forward_step=forward_step):
            t = np.arange(0.05, np.max(data[:, 0]), 0.0005)
            dt=t[1]-t[0]
            u_ini=u_obs[start_step]
            u_post=np.zeros([forward_step,u_obs.shape[1]])
            u_post[0]=u_ini
            for i in range(1,forward_step):
                u_post[i]=update_u_TVD(u_post[i-1],x,t[start_step+i-1],
                                   dx,dt,
                                   u_obs[start_step+i,0],u_obs[start_step+i,-1])

            return u_post

    if equation=='ut+uxxx+ux+u^2*uxxx':
        def weno5_flux_derivative(f, dx):
            N = len(f)
            flux = np.zeros(N)
            eps = 1e-6

            for i in range(3, N - 3):
                f_minus2 = f[i - 2]
                f_minus1 = f[i - 1]
                f_0 = f[i]
                f_plus1 = f[i + 1]
                f_plus2 = f[i + 2]

                beta0 = (13 / 12) * (f_minus2 - 2 * f_minus1 + f_0) ** 2 + (1 / 4) * (
                            f_minus2 - 4 * f_minus1 + 3 * f_0) ** 2
                beta1 = (13 / 12) * (f_minus1 - 2 * f_0 + f_plus1) ** 2 + (1 / 4) * (f_minus1 - f_plus1) ** 2
                beta2 = (13 / 12) * (f_0 - 2 * f_plus1 + f_plus2) ** 2 + (1 / 4) * (3 * f_0 - 4 * f_plus1 + f_plus2) ** 2

                alpha0 = 0.1 / (eps + beta0) ** 2
                alpha1 = 0.6 / (eps + beta1) ** 2
                alpha2 = 0.3 / (eps + beta2) ** 2
                alpha_sum = alpha0 + alpha1 + alpha2

                w0 = alpha0 / alpha_sum
                w1 = alpha1 / alpha_sum
                w2 = alpha2 / alpha_sum

                flux[i] = (w0 * (2 * f_minus2 - 7 * f_minus1 + 11 * f_0) +
                           w1 * (-f_minus1 + 5 * f_0 + 2 * f_plus1) +
                           w2 * (2 * f_0 + 5 * f_plus1 - f_plus2)) / 6.0

            dfdx = np.zeros_like(f)
            dfdx[3:-3] = (flux[4:-2] - flux[3:-3]) / dx
            return dfdx


        def third_derivative(u, dx):
            u_x = weno5_flux_derivative(u, dx)
            u_xx = (np.roll(u_x, -1) - 2 * u_x + np.roll(u_x, 1)) / dx ** 2
            u_xx[:3] = u_xx[-3:] = 0.0
            return u_xx


        def compute_rhs(u, dx, c1, c2, c3):
            ux = weno5_flux_derivative(u, dx)
            uxxx = third_derivative(u, dx)

            # 使用平滑后的 u^2 与 uxxx 相乘，减少非线性震荡传播
            u_smooth = gaussian_filter1d(u, sigma=1.0)
            nonlinear_term = (u_smooth ** 2) * uxxx

            return - (c1 * uxxx + c2 * ux + c3 * nonlinear_term)


        def rk3_weno_step(u, dx, dt, c1, c2, c3, u_L, u_R):
            rhs1 = compute_rhs(u, dx, c1, c2, c3)
            u1 = u + dt * rhs1
            u1[0], u1[-1] = u_L, u_R

            rhs2 = compute_rhs(u1, dx, c1, c2, c3)
            u2 = 0.75 * u + 0.25 * (u1 + dt * rhs2)
            u2[0], u2[-1] = u_L, u_R

            rhs3 = compute_rhs(u2, dx, c1, c2, c3)
            u3 = (1 / 3) * u + (2 / 3) * (u2 + dt * rhs3)
            u3[0], u3[-1] = u_L, u_R

            return u3
        def rk3_weno_step(u,x,t, dx, dt, u_L, u_R):
            c1 = 4.95860546e-05
            c2 = 1.41950787e+00
            c3 = -5.13596135e-07
            rhs1 = compute_rhs(u, dx, c1, c2, c3)
            u1 = u + dt * rhs1
            u1[0], u1[-1] = u_L, u_R

            rhs2 = compute_rhs(u1, dx, c1, c2, c3)
            u2 = 0.75 * u + 0.25 * (u1 + dt * rhs2)
            u2[0], u2[-1] = u_L, u_R

            rhs3 = compute_rhs(u2, dx, c1, c2, c3)
            u3 = (1 / 3) * u + (2 / 3) * (u2 + dt * rhs3)
            u3[0], u3[-1] = u_L, u_R
            return u3


        def get_posterial(u_obs, x, dx, start_step=start_step, forward_step=forward_step):

            t = np.arange(0.05, np.max(data[:, 0]), 0.00005)
            dt = t[1] - t[0]
            u_ini = u_obs[start_step]
            u_post = np.zeros([forward_step, u_obs.shape[1]])
            u_post[0] = u_ini
            for i in range(1, forward_step):
                u_post[i] = rk3_weno_step(u_post[i - 1], x, t[start_step + i - 1],
                                         dx, dt,
                                         u_obs[start_step + i, 0], u_obs[start_step + i, -1])

            return u_post
    post_2=get_posterial(pred_2,x_2,dx_2)


    print(post_2.shape)
    for i in range(forward_step):
        if i%1000==0:
            font_settings = {'family': 'Arial', 'size': 7}
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["font.size"] = 7
            plt.figure(figsize=(3, 2), dpi=300)
            #plt.scatter(x_1+8, true[:, 2] * 100, c='black', marker='^', s=1, label='Obs')
            plt.plot(x_2+8, pred_2[start_step+i], 'k-',lw=1.5, label='Obs')
            #plt.scatter(x_1+8, pred[i], c='red', marker='*', s=1, label='Obs')
            plt.plot(x_2+8, post_2[i], 'r--',lw=1.5, label='Pred')
            plt.title(f't={round(t[start_step+i], 3)}', fontsize=7)
            plt.xlim([8, 13])
            plt.xlabel("x", fontdict=font_settings)
            plt.ylabel("h", fontdict=font_settings)
            plt.xticks(fontsize=7, fontname='Arial')
            plt.yticks(fontsize=7, fontname='Arial')
            plt.legend(prop={'family': 'Arial', 'size': 6}, loc='best')
            plt.tight_layout()
            # 显示图像
            plt.show()