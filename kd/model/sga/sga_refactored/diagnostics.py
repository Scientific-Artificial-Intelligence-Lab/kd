import numpy as np

def run_diagnostics(workspace, config):
    """
    运行诊断，计算并打印真实PDE与计算出的导数之间的误差。
    这部分逻辑迁移自原始 setup.py 的误差计算部分。

    Args:
        workspace (dict): 包含所有导数数据的字典。
        config (module): 导入的原始 configure 模块，用于获取真实解。
    """
    print("   - Running diagnostics to check data error...")
    
    # 从工作空间解包我们需要的变量
    # 误差计算应与 setup.py 的 "data error without edges" 分支一致
    u = workspace['u']
    x = np.tile(workspace['x'], (u.shape[1], 1)).T
    t = np.tile(workspace['t'], (u.shape[0], 1))
    ux = workspace['ux']
    uxx = workspace['uxx']
    uxxx = workspace['uxxx']
    n, m = u.shape

    # 动态执行 config.right_side 和 config.left_side
    local_vars = {
        'u': u,
        'x': x,
        't': t,
        'ux': ux,
        'uxx': uxx,
        'uxxx': uxxx,
        'ut': workspace['ut'],
        'config': config
    }
    exec(config.right_side, globals(), local_vars)
    right_side = local_vars['right_side']
    exec(config.left_side, globals(), local_vars)
    left_side = local_vars['left_side']

    n1, n2, m1, m2 = int(n*0.1), int(n*0.9), int(m*0), int(m*1)
    right_side = right_side[n1:n2, m1:m2]
    left_side = left_side[n1:n2, m1:m2]
    right = np.reshape(right_side, ((n2-n1)*(m2-m1), 1))
    left = np.reshape(left_side, ((n2-n1)*(m2-m1), 1))
    diff = np.linalg.norm(left-right, 2)/((n2-n1)*(m2-m1))
    print(f"   - Data error (without edges): {diff}")
