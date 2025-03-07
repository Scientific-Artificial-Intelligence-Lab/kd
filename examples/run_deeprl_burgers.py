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
step_output = model.train(n_epochs=50)
print(f"Current best expression is {step_output['expression']} and its reward is {step_output['r']}")


#####################################################################
# Visualizations
#####################################################################


#-------------------------------------------------------------------
# 1. Training Process Visualization
#-------------------------------------------------------------------

# 残差分析图
def plot_residual_analysis(model, output_dir=".plot_output"):
    """绘制残差分析图"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 获取数据
    data = model.data_class.get_data()
    x_data, t_data = data['x'], data['t'] 
    u_true = data['u']
    
    # 获取预测值
    xx, tt = np.meshgrid(x_data, t_data, indexing='ij')
    X_pred = np.hstack([xx.reshape(-1,1), tt.reshape(-1,1)])
    u_pred = model.best_p[0].evaluate_residual(X_pred)
    u_pred = u_pred.reshape(u_true.shape)
    
    # 计算残差
    residual = u_true - u_pred
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 真实解
    im0 = axes[0,0].pcolormesh(tt, xx, u_true, shading='auto')
    axes[0,0].set_title('True Solution')
    plt.colorbar(im0, ax=axes[0,0])
    
    # 2. 预测解
    im1 = axes[0,1].pcolormesh(tt, xx, u_pred, shading='auto')
    axes[0,1].set_title('Predicted Solution')
    plt.colorbar(im1, ax=axes[0,1])
    
    # 3. 残差
    im2 = axes[1,0].pcolormesh(tt, xx, residual, shading='auto')
    axes[1,0].set_title('Residual')
    plt.colorbar(im2, ax=axes[1,0])
    
    # 4. 残差直方图
    axes[1,1].hist(residual.flatten(), bins=50)
    axes[1,1].set_title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

#-------------------------------------------------------------------
# 2. Expression Discovery Visualization  
#-------------------------------------------------------------------



#-------------------------------------------------------------------
# 3. PINN Analysis (if using DeepRL_Pinn)
#-------------------------------------------------------------------



# 树图
# model.plot(fig_type ='tree').view()

# model.plot(fig_type='evolution').view()

#-------------------------------------------------------------------
# Call visualization functions
#-------------------------------------------------------------------
plot_residual_analysis(model, output_dir="results/residual_plots")

