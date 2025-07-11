import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) # 注意此处和 notebook 语法不同
kd_main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(kd_main_dir)

# ---

from kd.dataset import load_kdv_equation, load_pde_dataset 
from kd.model.kd_dlga import KD_DLGA
from kd.viz.dlga_viz import *

kdv_data = load_kdv_equation()                      # 首先，我们调用一个辅助函数来加载预先生成好的KdV方程的解数据 kdv_data 对象中包含了时空坐标和对应的解

# 也可以直接加载一个.mat文件，注意这里的 x_key, t_key, u_key 是根据数据集的实际结构来指定的
# my_data = load_pde_dataset(filename="KdV_equation.mat", x_key='x', t_key='tt', u_key='uu')  
                            
X_train, y_train = kdv_data.sample(n_samples=1000)  # 从总数据集中随机抽取1000个样本点作为我们的训练数据

model = KD_DLGA(
    operators=['u', 'u_x', 'u_xx', 'u_xxx'],        # 定义方程发现的 候选算子库
    epi=0.1,                                        # 方程的“稀疏性”或“简洁度”惩罚项
    input_dim=2,                                    # 输入数据的维度，必须与 X_train 的列数匹配
    verbose=False,                                  # 是否在进化过程中打印出每一代新发现的最优解
    max_iter=9000                                   # 内部神经网络的最大训练迭代次数
)

print("\nTraining DLGA model...")
model.fit(X_train, y_train)                         # 这个过程将首先训练一个深度神经网络来学习数据的内在模式，然后运行遗传算法来搜索方程的符号形式，整个过程可能需要几分钟

print("\nGenerating predictions...")
X_full = kdv_data.mesh() # Create full grid for visualization
u_pred = model.predict(X_full)
u_pred = u_pred.reshape(kdv_data.get_size())


# 绘制神经网络的 训练损失 曲线
plot_training_loss(model) 

# 绘制 验证损失 曲线
plot_validation_loss(model) 

# 绘制最优解的适应度（Fitness）与复杂度（Complexity）随代数进化的曲线
plot_optimization_analysis(model)

# 将真实的解 u 与模型的预测解 u_pred 进行并排热图比较
plot_pde_comparison(kdv_data.x, kdv_data.t, kdv_data.usol, u_pred)

# 绘制训练点上的残差分布图，以及模型在整个求解域上的残差统计直方图
plot_residual_analysis(model, X_train, y_train, kdv_data.usol, u_pred)

# 绘制在特定时间点上，真实解与预测解的横截面对比图
plot_time_slices(kdv_data.x, kdv_data.t, kdv_data.usol, u_pred, slice_times=[0.25, 0.5, 0.75])

# # 我们想要探究 KdV 方程中两个关键的右手边项 u*u_x 和 u_xxx 之间的关系, 以决定之后选择什么样的算子
# plot_equation_terms(
#     model,
#     terms={
#         'x_term': {'vars': ['u', 'u_x'], 'label': '6uu_x'},
#         'y_term': {'vars': ['u_xxx'], 'label': '-u_xxx'}
#     },
#     equation_name="KdV Equation",
# )

# # 调用此函数，它会自动解析 model 中的最优解，并为最重要的几个项生成关系图
# # 其核心思想是：如果一个RHS项确实是构成方程的关键部分，那么它与LHS之间应该存在一种简单的（通常是线性的）关系
# plot_derivative_relationships(model) 

# # 生成对最终发现方程的奇偶图验证
# plot_pde_parity(model, title="Final Validation of Discovered Equation")