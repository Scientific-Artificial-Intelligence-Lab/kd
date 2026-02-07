"""
SGA-PDE 主入口示例（参考样例）。
用法：修改 SELECTED_MODE 后直接运行 `python main.py`。
"""

import sys
import warnings
from typing import Dict, Tuple

import numpy as np

from sgapde.config import SolverConfig
from sgapde.context import ProblemContext
from sgapde.solver import SGAPDE_Solver
import sgapde.visualizer as visualizer

warnings.filterwarnings("ignore")


class Logger:
    """将输出同时写入控制台与文件。"""

    def __init__(self, filename: str = "notes.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        pass


def _apply_demo_defaults(config: SolverConfig, *, generations: int = 1) -> SolverConfig:
    """缩小 GA 参数以便快速演示。"""
    config.num = 4
    config.depth = 3
    config.width = 2
    config.sga_run = generations
    config.seed = 0
    return config


def _apply_autograd_defaults(config: SolverConfig) -> SolverConfig:
    """缩短 MetaNN 训练时长，保持演示可运行。"""
    config.hidden_dim = 16
    config.max_epoch = 50
    config.train_ratio = 1.0
    return config


def _make_legacy_payload() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy 2D：u(x, t) 与 1D x/t 坐标。"""
    x = np.linspace(0.0, 1.0, 6)
    t = np.linspace(0.0, 2.0, 5)
    X = x[:, None]
    T = t[None, :]
    u = X + 0.2 * T
    return u, x, t


def _make_nd_payload(multifield: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], list]:
    """N 维 structured grid payload（默认 3D）。"""
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 4)
    t = np.linspace(0.0, 2.0, 6)

    X = x[:, None, None]
    Y = y[None, :, None]
    T = t[None, None, :]

    u = X + 0.2 * T + 0.1 * Y
    if multifield:
        v = 2.0 * u
        fields = {"u": u, "v": v}
    else:
        fields = {"u": u}

    coords_1d = {"x": x, "y": y, "t": t}
    axis_order = ["x", "y", "t"]
    return fields, coords_1d, axis_order


MODE_SPECS = {
    "legacy_fd": {
        "label": "Legacy 2D + finite difference",
        "legacy": True,
        "autograd": False,
        "multifield": False,
    },
    "legacy_autograd": {
        "label": "Legacy 2D + autograd/MetaNN",
        "legacy": True,
        "autograd": True,
        "multifield": False,
    },
    "nd_fd": {
        "label": "N-dim structured grid + finite difference",
        "legacy": False,
        "autograd": False,
        "multifield": False,
    },
    "nd_autograd": {
        "label": "N-dim structured grid + autograd/MetaNN",
        "legacy": False,
        "autograd": True,
        "multifield": False,
    },
    "nd_multifield_fd": {
        "label": "N-dim multi-field + finite difference",
        "legacy": False,
        "autograd": False,
        "multifield": True,
    },
    "nd_multifield_autograd": {
        "label": "N-dim multi-field + autograd/MetaNN",
        "legacy": False,
        "autograd": True,
        "multifield": True,
    },
}


def build_config(mode: str) -> SolverConfig:
    """按 mode 规格构造 SolverConfig。"""
    spec = MODE_SPECS[mode]

    if spec["legacy"]:
        u, x, t = _make_legacy_payload()
        config = SolverConfig(
            problem_name=f"{mode}_demo",
            u_data=u,
            x_data=x,
            t_data=t,
            axis_order=["x", "t"],
            target_field="u",
            lhs_axis="t",
            use_metadata=False,
            use_autograd=spec["autograd"],
        )
    else:
        fields, coords_1d, axis_order = _make_nd_payload(multifield=spec["multifield"])
        config = SolverConfig(
            problem_name=f"{mode}_demo",
            fields_data=fields,
            coords_1d=coords_1d,
            axis_order=axis_order,
            target_field="u",
            lhs_axis="t",
            use_metadata=False,
            use_autograd=spec["autograd"],
        )
        if spec["autograd"] and spec["multifield"]:
            config.autograd_fields = list(fields.keys())

    if spec["autograd"]:
        _apply_autograd_defaults(config)

    return _apply_demo_defaults(config)


def print_available_modes() -> None:
    """打印可选的演示模式。"""
    print("Supported demo modes:")
    for key, spec in MODE_SPECS.items():
        print(f"  - {key}: {spec['label']}")


def run_demo(mode: str) -> None:
    """运行单个演示模式。"""
    if mode not in MODE_SPECS:
        raise ValueError(f"Unknown demo mode: {mode}")

    config = build_config(mode)

    print(f"\nSelected mode: {mode} ({MODE_SPECS[mode]['label']})")
    print("\n1. Setting up configuration...")
    print(f"\tProblem: {config.problem_name}")
    print(f"\tPool size: {config.num}")
    print(f"\tMax depth: {config.depth}")
    print(f"\tMax width: {config.width}")
    print(f"\tGenerations: {config.sga_run}")
    print(f"\tUse autograd: {config.use_autograd}")

    print("\n2. Preparing problem context...")
    context = ProblemContext(config)
    print(f"\tData shape: {context.u.shape}")
    print(f"\tAxis order: {context.axis_order}")
    print(f"\tLHS axis: {context.lhs_axis}")

    if getattr(config, "use_metadata", False) and getattr(config, "show_metadata_diagnostics", False):
        visualizer.plot_metadata_diagnostics(context)

    print("\n3. Creating solver and running genetic algorithm...")
    print("=" * 80)
    solver = SGAPDE_Solver(config)
    best_pde, best_score = solver.run(context)

    print("=" * 80)
    print("\nDiscovery finished!")
    print(f"Best PDE found: {best_pde}")
    print(f"AIC Score: {best_score}")


if __name__ == "__main__":
    sys.stdout = Logger("notes.log", sys.stdout)
    sys.stderr = Logger("notes.log", sys.stderr)

    print("SGA-PDE: Symbolic Genetic Algorithm for PDE Discovery")
    print("用法：修改 SELECTED_MODE 再运行 main.py。")
    print_available_modes()

    SELECTED_MODE = "nd_fd"
    run_demo(SELECTED_MODE)
