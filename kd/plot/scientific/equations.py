"""
预定义方程库。
包含常见PDE方程的函数定义，可用于残差计算和方程发现。
"""

from typing import Dict
import numpy as np

def kdv_equation(metadata: Dict[str, np.ndarray], coeff_u_ux: float = 6.0):
    """KdV方程: u_t + coeff_u_ux*u*u_x + u_xxx = 0
    
    参数:
        metadata: 包含u及其导数的字典
        coeff_u_ux: u*u_x项的系数，默认为6.0
        
    返回:
        方程残差数组
    """
    return (metadata['u_t'] + 
            coeff_u_ux * metadata['u'] * metadata['u_x'] + 
            metadata['u_xxx']).reshape(-1, 1)

def burgers_equation(metadata: Dict[str, np.ndarray], nu: float = 0.01):
    """Burgers方程: u_t + u*u_x - nu*u_xx = 0
    
    参数:
        metadata: 包含u及其导数的字典
        nu: 粘性系数，默认为0.01
        
    返回:
        方程残差数组
    """
    return (metadata['u_t'] + 
            metadata['u'] * metadata['u_x'] - 
            nu * metadata['u_xx']).reshape(-1, 1)

def wave_equation(metadata: Dict[str, np.ndarray], c: float = 1.0):
    """波动方程: u_tt - c²*u_xx = 0
    
    参数:
        metadata: 包含u及其导数的字典
        c: 波速，默认为1.0
        
    返回:
        方程残差数组
    """
    if 'u_tt' not in metadata:
        raise ValueError("二阶时间导数'u_tt'未在metadata中提供，无法计算波动方程残差")
    return (metadata['u_tt'] - 
            c**2 * metadata['u_xx']).reshape(-1, 1)

def heat_equation(metadata: Dict[str, np.ndarray], alpha: float = 1.0):
    """热传导方程: u_t - alpha*u_xx = 0
    
    参数:
        metadata: 包含u及其导数的字典
        alpha: 热扩散系数，默认为1.0
        
    返回:
        方程残差数组
    """
    return (metadata['u_t'] - 
            alpha * metadata['u_xx']).reshape(-1, 1)

def advection_equation(metadata: Dict[str, np.ndarray], c: float = 1.0):
    """平流方程: u_t + c*u_x = 0
    
    参数:
        metadata: 包含u及其导数的字典
        c: 平流速度，默认为1.0
        
    返回:
        方程残差数组
    """
    return (metadata['u_t'] + 
            c * metadata['u_x']).reshape(-1, 1)

def reaction_diffusion_equation(metadata: Dict[str, np.ndarray], 
                               D: float = 1.0, 
                               r: float = 1.0):
    """反应扩散方程: u_t - D*u_xx = r*u*(1-u)
    
    参数:
        metadata: 包含u及其导数的字典
        D: 扩散系数，默认为1.0
        r: 反应速率，默认为1.0
        
    返回:
        方程残差数组
    """
    u = metadata['u']
    return (metadata['u_t'] - 
            D * metadata['u_xx'] - 
            r * u * (1 - u)).reshape(-1, 1)

def schrodinger_equation(metadata: Dict[str, np.ndarray], hbar: float = 1.0, m: float = 1.0):
    """薛定谔方程: i*hbar*u_t = -hbar²/(2m)*u_xx + V*u
    
    参数:
        metadata: 包含u及其导数的字典
        hbar: 约化普朗克常数，默认为1.0
        m: 粒子质量，默认为1.0
        
    注意:
        这是简化版本，假设势能V=0
        必须分别处理实部和虚部
        
    返回:
        方程残差数组
    """
    # 简化版本，假设V=0
    return (1j * hbar * metadata['u_t'] + 
            (hbar**2)/(2*m) * metadata['u_xx']).reshape(-1, 1)

def navier_stokes_2d(metadata: Dict[str, np.ndarray], Re: float = 100.0):
    """2D Navier-Stokes方程组（速度形式）
    
    u_t + u*u_x + v*u_y = -p_x + (1/Re)*(u_xx + u_yy)
    v_t + u*v_x + v*v_y = -p_y + (1/Re)*(v_xx + v_yy)
    u_x + v_y = 0 (连续性方程)
    
    参数:
        metadata: 包含流场导数的字典
            必须包含: u, v, p及其各阶导数
        Re: 雷诺数，默认为100.0
        
    返回:
        Dict: 包含各方程残差的字典
    """
    if not all(k in metadata for k in ['u', 'v', 'p', 'u_x', 'u_y', 'u_t']):
        raise ValueError("缺少Navier-Stokes方程所需的变量")
    
    # 动量方程x方向
    x_momentum = (metadata['u_t'] + 
                 metadata['u'] * metadata['u_x'] + 
                 metadata['v'] * metadata['u_y'] + 
                 metadata['p_x'] - 
                 (1/Re) * (metadata['u_xx'] + metadata['u_yy']))
    
    # 动量方程y方向
    y_momentum = (metadata['v_t'] + 
                 metadata['u'] * metadata['v_x'] + 
                 metadata['v'] * metadata['v_y'] + 
                 metadata['p_y'] - 
                 (1/Re) * (metadata['v_xx'] + metadata['v_yy']))
    
    # 连续性方程
    continuity = metadata['u_x'] + metadata['v_y']
    
    return {
        'x_momentum': x_momentum.reshape(-1, 1),
        'y_momentum': y_momentum.reshape(-1, 1),
        'continuity': continuity.reshape(-1, 1)
    }

def create_custom_equation(terms_dict):
    """创建自定义方程函数。
    
    参数:
        terms_dict: Dict[str, float] - 项名和系数的字典
                    例如: {'u_t': 1.0, 'u*u_x': 6.0, 'u_xxx': 1.0}
                    会创建方程: u_t + 6*u*u_x + u_xxx = 0
    
    返回:
        function: 可用于计算残差的方程函数
    """
    def custom_equation(metadata):
        residual = 0
        for term, coeff in terms_dict.items():
            if term == 'u*u_x':
                residual += coeff * metadata['u'] * metadata['u_x']
            elif term in metadata:
                residual += coeff * metadata[term]
            else:
                raise ValueError(f"项 '{term}' 在metadata中不存在")
        return residual.reshape(-1, 1)
    
    return custom_equation 