"""
PDE数据集注册表
根据旧代码分析得出的映射关系

说明 (与 SGA 相关):
- SGA backend 自带的 `SolverConfig._load_problem_config` 内部仍保留了对
  `'chafee-infante'` / `'burgers'` / `'kdv'` 三个 benchmark 的硬编码加载逻辑,
  以及对应的真 PDE 模板, 主要用于兼容原始 `sgapde` 论文代码。
- 在 KD 的主数据流中, SGA 更推荐通过 `PDEDataset` + `KD_SGA.fit_dataset` 使用。
  这时数据文件/键名等信息以本注册表为准, 然后通过 `u_data/x_data/t_data` 注入
  给 SGA; `_load_problem_config` 对新数据集不再扩展, 仅作为那三个内置问题的
  legacy 支持。
"""

# Example registry entry (pseudo-JSON):
# {
#   'file': 'KdV_equation.mat',      # 或使用 'files': {...} 指向多个 .npy/.npz 文件
#   'keys': {                        # 对 .mat/.npz 的字段名称映射, 若不是单纯数组请补齐
#       'x_key': 'x',
#       't_key': 'tt',
#       'u_key': 'uu',
#   },
#   'shape': (512, 201),            # 可选: 当文件只有 u 矩阵时, 指定 (nx, nt) 用于 reshape
#   'domain': {'x': (-1.0, 1.0), 't': (0.0, 1.0)},  # 物理范围; 如果未知/无意义可设为 None
#   'sym_true': 'add,mul,...',      # 可选: 真实方程, 提供给 DSCV pipeline 做评估
#   'aliases': {                    # 可选: legacy 别名, 或者各模型需要的特殊 problem 名称
#       'legacy': 'Kdv',
#       'sga_problem': 'kdv'
#   },
#   'models': {                     # 维护每个模型是否已验证通过
#       'sga': True,
#       'dscv': True,
#       'dscv_spr': False
#   },
#   'status': 'active',             # active: 已验证; pending: 待处理; legacy: 仅保留兼容
#   'notes': '可选描述, 记录数据来源或特殊注意事项'
# }

PDE_REGISTRY = {
    'chafee-infante': {
        'files': {
            'u': 'chafee_infante_CI.npy',
            'x': 'chafee_infante_x.npy',
            't': 'chafee_infante_t.npy',
        },
        'sym_true': 'add,add,u1,n3,u1,diff2,u1,x1',
        'domain': None,
        'models': {
            'sga': True,
            'dscv': True,
            'dscv_spr': False,
            'dlga': False,
        },
        'status': 'active',
    },
    'burgers': {
        'file': 'burgers2.mat',
        'keys': {
            'x_key': 'x',
            't_key': 't',
            'u_key': 'usol',
        },
        'sym_true': 'add,mul,u1,diff,u1,x1,diff2,u1,x1',
        'domain': None,
        'aliases': {
            'legacy': 'Burgers2',
            'sga_problem': 'burgers',
        },
        'models': {
            'sga': True,
            'dscv': True,
            'dscv_spr': True,
            'dlga': False,
        },
        'status': 'active',
    },
    'kdv': {
        'file': 'KdV_equation.mat',
        'keys': {
            'x_key': 'x',
            't_key': 'tt',
            'u_key': 'uu',
        },
        'sym_true': 'add,mul,u1,diff,u1,x1,diff3,u1,x1',
        'domain': None,
        'aliases': {
            'legacy': 'Kdv',
            'sga_problem': 'kdv',
        },
        'models': {
            'sga': True,
            'dscv': True,
            'dscv_spr': False,
            'dlga': False,
        },
        'status': 'active',
    },
    'PDE_divide': {
        'file': 'PDE_divide.npy',
        'shape': (100, 251),
        'sym_true': 'add,div,diff,u1,x1,x1,diff2,u1,x1',
        'domain': {'x': (1, 2), 't': (0, 1)},
        'models': {
            'sga': False,
            'dscv': True,
            'dscv_spr': False,
            'dlga': False,
        },
        'status': 'active',
    },
    'PDE_compound': {
        'file': 'PDE_compound.npy',
        'shape': (100, 251),
        'sym_true': 'add,mul,u1,diff2,u1,x1,mul,diff,u1,x1,diff,u1,x1',
        'domain': {'x': (1, 2), 't': (0, 0.5)},
        'models': {
            'sga': False,
            'dscv': True,
            'dscv_spr': False,
            'dlga': False,
        },
        'status': 'active',
    },
    'fisher': {
        'file': 'fisher_nonlin_groundtruth.mat',
        'keys': {
            'x_key': 'x',
            't_key': 't',
            'u_key': 'U',
        },
        'sym_true': 'add,mul,u1,diff2,u1,x1,add,n2,diff,u1,x1,add,u1,n2,u1',
        'domain': {'x': (-1.0, 1.0), 't': (0.0, 1.0)},
        'models': {
            'sga': False,
            'dscv': True,
            'dscv_spr': False,
            'dlga': False,
        },
        'status': 'active',
    },
    'fisher_linear': {
        'file': 'fisher_groundtruth.mat',
        'keys': {
            'x_key': 'x',
            't_key': 't',
            'u_key': 'U',
        },
        'sym_true': 'add,diff2,u1,x1,add,u1,n2,u1',
        'domain': {'x': (-1.0, 1.0), 't': (0.0, 1.0)},
        'models': {
            'sga': False,
            'dscv': True,
            'dscv_spr': False,
            'dlga': False,
        },
        'status': 'active',
    },
    'advection_diffusion': {
        'file': 'Advection_diffusion.mat',
        'keys': {
            'x_key': 'x',
            't_key': 't',
            'u_key': 'u',
        },
        'sym_true': None,
        'domain': None,
        'models': {
            'sga': False,
            'dscv': False,
            'dscv_spr': False,
            'dlga': False,
        },
        'status': 'pending',
        'notes': 'Original .mat file missing expected x/t/u keys; loader update in progress.',
    },
}

def get_dataset_info(name: str) -> dict:
    if name not in PDE_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available datasets: {list(PDE_REGISTRY.keys())}"
        )
    return PDE_REGISTRY[name]

def list_available_datasets() -> list:
    return list(PDE_REGISTRY.keys())

def get_dataset_sym_true(name: str) -> str:
    info = get_dataset_info(name)
    return info.get('sym_true', None)
