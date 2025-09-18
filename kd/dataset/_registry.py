"""
PDE数据集注册表
根据旧代码分析得出的映射关系
"""

PDE_REGISTRY = {
    # 基于 kd/data.py 分析
    'chafee-infante': {
        'files': {
            'u': 'chafee_infante_CI.npy',
            'x': 'chafee_infante_x.npy',
            't': 'chafee_infante_t.npy'
        },
        'keys': None,  # npy文件直接加载
        'sym_true': 'add,add,u1,n3,u1,diff2,u1,x1',
        'domain': None,
        'legacy_name': 'chafee-infante'
    },

    'burgers': {
        'file': 'burgers2.mat',
        'keys': {
            'x_key': 'x',
            't_key': 't', 
            'u_key': 'usol'
        },
        'sym_true': 'add,mul,u1,diff,u1,x1,diff2,u1,x1',
        'domain': {'x': (-7.0, 7.0), 't': (1, 9)},
        'legacy_name': 'Burgers2'
    },

    'kdv': {
        'file': 'KdV_equation.mat',
        'keys': {
            'x_key': 'x',
            't_key': 'tt',
            'u_key': 'uu'
        },
        'sym_true': 'add,mul,u1,diff,u1,x1,diff3,u1,x1',
        'domain': {'x': (-16, 16), 't': (5, 35)},
        'legacy_name': 'Kdv'
    },

    # 基于 kd/model/discover/task/pde/data_load.py 分析
    'PDE_divide': {
        'file': 'PDE_divide.npy',
        'keys': None,
        'sym_true': 'add,div,diff,u1,x1,x1,diff2,u1,x1',
        'domain': {'x': (1, 2), 't': (0, 1)},
        'legacy_name': 'PDE_divide',
        'shape': (100, 251)  # nx=100, nt=251
    },

    'PDE_compound': {
        'file': 'PDE_compound.npy',
        'keys': None,
        'sym_true': 'add,mul,u1,diff2,u1,x1,mul,diff,u1,x1,diff,u1,x1',
        'domain': {'x': (1, 2), 't': (0, 0.5)},
        'legacy_name': 'PDE_compound',
        'shape': (100, 251)  # nx=100, nt=251
    },
    
    # 添加更多数据集（基于文件分析）
    'fisher': {
        'file': 'fisher_nonlin_groundtruth.mat',
        'keys': {
            'x_key': 'x',
            't_key': 't', 
            'u_key': 'U'
        },
        'sym_true': 'add,mul,u1,diff2,u1,x1,add,n2,diff,u1,x1,add,u1,n2,u1',
        'domain': None
    },
    
    'fisher_linear': {
        'file': 'fisher_groundtruth.mat',
        'keys': {
            'x_key': 'x',
            't_key': 't',
            'u_key': 'U'
        },
        'sym_true': 'add,diff2,u1,x1,add,u1,n2,u1',
        'domain': None
    },
    
    'advection_diffusion': {
        'file': 'Advection_diffusion.mat',
        'keys': {
            'x_key': 'x',
            't_key': 't',
            'u_key': 'u'
        },
        'sym_true': None,
        'domain': None
    }
}

def get_dataset_info(name: str) -> dict:
    """
    获取数据集配置信息
    
    Args:
        name: 数据集名称
        
    Returns:
        数据集配置字典
        
    Raises:
        ValueError: 如果数据集名称未知
    """
    if name not in PDE_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available datasets: {list(PDE_REGISTRY.keys())}"
        )
    return PDE_REGISTRY[name]

def list_available_datasets() -> list:
    """
    列出所有可用的数据集
    
    Returns:
        数据集名称列表
    """
    return list(PDE_REGISTRY.keys())

def get_dataset_sym_true(name: str) -> str:
    """
    获取数据集的真实符号表达式
    
    Args:
        name: 数据集名称
        
    Returns:
        符号表达式字符串, 如果没有则返回None
    """
    info = get_dataset_info(name)
    return info.get('sym_true', None)
