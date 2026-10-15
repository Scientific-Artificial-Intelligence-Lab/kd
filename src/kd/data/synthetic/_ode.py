
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from kd.data.trajectory import Trajectory, TrajectorySet

RightHandSide = Callable[[float, np.ndarray, Mapping[str, float]], np.ndarray]

ODE_CORE_TRACK: Final[tuple[str, ...]] = (
    "damped_oscillator",
    "pendulum",
    "duffing",
    "van_der_pol",
    "sir",
    "fitzhugh_nagumo",
    "rational",
    "robertson",
)


@dataclass(frozen=True, kw_only=True)
class OdeSystem:

    name: str
    category: str
    track: Literal["core", "supplementary"]
    state_vars: tuple[str, ...]
    params: Mapping[str, float]
    equations: tuple[str, ...]
    answers: tuple[str, ...]
    initial: tuple[float, ...]
    t_span: tuple[float, float]
    method: str
    time_sampling: Literal["uniform", "log"]
    conserved_total: bool
    zero_initial_scale: float
    rhs: RightHandSide

    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        return self.rhs(t, y, self.params)






def _oscillator(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array([y[1], -(p["omega"] ** 2) * y[0]])


def _population(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [
            p["alpha"] * y[0] - p["beta"] * y[0] * y[1],
            p["delta"] * y[0] * y[1] - p["gamma"] * y[1],
        ]
    )


def _chaotic(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [
            p["sigma"] * (y[1] - y[0]),
            y[0] * (p["rho"] - y[2]) - y[1],
            y[0] * y[1] - p["beta"] * y[2],
        ]
    )


def _rational(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array([-p["vmax"] * y[0] / (p["km"] + y[0])])


def _damped_oscillator(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [y[1], -2 * p["zeta"] * p["omega0"] * y[1] - p["omega0"] ** 2 * y[0]]
    )


def _pendulum(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array([y[1], -p["gamma"] * y[1] - p["omega0"] ** 2 * np.sin(y[0])])


def _duffing(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [y[1], -p["delta"] * y[1] - p["alpha"] * y[0] - p["beta"] * y[0] ** 3]
    )


def _van_der_pol(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array([y[1], p["mu"] * (1 - y[0] ** 2) * y[1] - y[0]])


def _sir(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [
            -p["beta"] * y[0] * y[1],
            p["beta"] * y[0] * y[1] - p["gamma"] * y[1],
            p["gamma"] * y[1],
        ]
    )


def _fitzhugh_nagumo(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [
            y[0] - y[0] ** 3 / 3 - y[1] + p["current"],
            p["epsilon"] * (y[0] + p["a"] - p["b"] * y[1]),
        ]
    )


def _brusselator(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [
            p["A"] - (p["B"] + 1) * y[0] + y[0] ** 2 * y[1],
            p["B"] * y[0] - y[0] ** 2 * y[1],
        ]
    )


def _robertson(_t: float, y: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [
            -p["k1"] * y[0] + p["k3"] * y[1] * y[2],
            p["k1"] * y[0] - p["k2"] * y[1] ** 2 - p["k3"] * y[1] * y[2],
            p["k2"] * y[1] ** 2,
        ]
    )





def _system(
    name: str,
    category: str,
    state_vars: tuple[str, ...],
    equations: tuple[str, ...],
    answers: tuple[str, ...],
    params: dict[str, float],
    initial: tuple[float, ...],
    t_span: tuple[float, float],
    rhs: RightHandSide,
    *,
    method: str = "DOP853",
    time_sampling: Literal["uniform", "log"] = "uniform",
    conserved_total: bool = False,
    zero_initial_scale: float = 1e-8,
) -> OdeSystem:
    return OdeSystem(
        name=name,
        category=category,
        track="core" if name in ODE_CORE_TRACK else "supplementary",
        state_vars=state_vars,
        params=params,
        equations=equations,
        answers=answers,
        initial=initial,
        t_span=t_span,
        method=method,
        time_sampling=time_sampling,
        conserved_total=conserved_total,
        zero_initial_scale=zero_initial_scale,
        rhs=rhs,
    )


ODE_SYSTEMS: Final[Mapping[str, OdeSystem]] = {
    system.name: system
    for system in (
        _system(
            "oscillator",
            "oscillatory",
            ("x", "v"),
            ("v", "-omega**2*x"),
            ("dx_dt = v", "dv_dt = -1.69 * x"),
            {"omega": 1.3},
            (1.0, 0.0),
            (0.0, 12.0),
            _oscillator,
        ),
        _system(
            "population",
            "population",
            ("prey", "predator"),
            ("alpha*prey-beta*prey*predator", "delta*prey*predator-gamma*predator"),
            (
                "dprey_dt = 1.1 * prey - 0.4 * prey * predator",
                "dpredator_dt = 0.1 * prey * predator - 0.4 * predator",
            ),
            {"alpha": 1.1, "beta": 0.4, "delta": 0.1, "gamma": 0.4},
            (8.0, 3.0),
            (0.0, 14.0),
            _population,
        ),
        _system(
            "chaotic",
            "chaotic",
            ("x", "y", "z"),
            ("sigma*(y-x)", "x*(rho-z)-y", "x*y-beta*z"),
            (
                "dx_dt = 10 * (y - x)",
                "dy_dt = x * (28 - z) - y",
                "dz_dt = x * y - 8/3 * z",
            ),
            {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
            (-8.0, 8.0, 27.0),
            (0.0, 8.0),
            _chaotic,
        ),
        _system(
            "rational",
            "rational",
            ("s",),
            ("-vmax*s/(km+s)",),
            ("ds_dt = -1.5 * s / (0.7 + s)",),
            {"vmax": 1.5, "km": 0.7},
            (4.0,),
            (0.0, 3.0),
            _rational,
        ),
        _system(
            "damped_oscillator",
            "damped_oscillatory",
            ("x", "v"),
            ("v", "-2*zeta*omega0*v-omega0**2*x"),
            ("dx_dt = v", "dv_dt = -0.26 * v - 1.69 * x"),
            {"zeta": 0.1, "omega0": 1.3},
            (1.0, 0.0),
            (0.0, 12.0),
            _damped_oscillator,
        ),
        _system(
            "pendulum",
            "nonlinear_oscillatory",
            ("theta", "v"),
            ("v", "-gamma*v-omega0**2*sin(theta)"),
            ("dtheta_dt = v", "dv_dt = -0.08 * v - 1.44 * sin(theta)"),
            {"gamma": 0.08, "omega0": 1.2},
            (1.2, 0.0),
            (0.0, 18.0),
            _pendulum,
        ),
        _system(
            "duffing",
            "nonlinear_oscillatory",
            ("x", "v"),
            ("v", "-delta*v-alpha*x-beta*x**3"),
            ("dx_dt = v", "dv_dt = -0.12 * v - x - 0.8 * x^3"),
            {"delta": 0.12, "alpha": 1.0, "beta": 0.8},
            (1.0, 0.0),
            (0.0, 18.0),
            _duffing,
        ),
        _system(
            "van_der_pol",
            "nonlinear_oscillatory",
            ("x", "v"),
            ("v", "mu*(1-x**2)*v-x"),
            ("dx_dt = v", "dv_dt = 2 * (1 - x^2) * v - x"),
            {"mu": 2.0},
            (2.0, 0.0),
            (0.0, 20.0),
            _van_der_pol,
        ),
        _system(
            "sir",
            "epidemic",
            ("susceptible", "infected", "recovered"),
            (
                "-beta*susceptible*infected",
                "beta*susceptible*infected-gamma*infected",
                "gamma*infected",
            ),
            (
                "dsusceptible_dt = -0.9 * susceptible * infected",
                "dinfected_dt = 0.9 * susceptible * infected - 0.25 * infected",
                "drecovered_dt = 0.25 * infected",
            ),
            {"beta": 0.9, "gamma": 0.25},
            (0.96, 0.039, 0.001),
            (0.0, 40.0),
            _sir,
            conserved_total=True,
        ),
        _system(
            "fitzhugh_nagumo",
            "excitable",
            ("v", "w"),
            ("v-v**3/3-w+current", "epsilon*(v+a-b*w)"),
            ("dv_dt = v - v^3 / 3 - w + 0.5", "dw_dt = 0.08 * (v + 0.7 - 0.8 * w)"),
            {"current": 0.5, "epsilon": 0.08, "a": 0.7, "b": 0.8},
            (-1.0, 1.0),
            (0.0, 60.0),
            _fitzhugh_nagumo,
        ),
        _system(
            "brusselator",
            "chemical",
            ("x", "y"),
            ("A-(B+1)*x+x**2*y", "B*x-x**2*y"),
            ("dx_dt = 1 - 4 * x + x^2 * y", "dy_dt = 3 * x - x^2 * y"),
            {"A": 1.0, "B": 3.0},
            (1.0, 1.0),
            (0.0, 20.0),
            _brusselator,
        ),
        _system(
            "robertson",
            "stiff_chemical",
            ("x", "y", "z"),
            ("-k1*x+k3*y*z", "k1*x-k2*y**2-k3*y*z", "k2*y**2"),
            (
                "dx_dt = -0.04 * x + 10000 * y * z",
                "dy_dt = 0.04 * x - 30000000 * y^2 - 10000 * y * z",
                "dz_dt = 30000000 * y^2",
            ),
            {"k1": 0.04, "k2": 3e7, "k3": 1e4},
            (1.0, 0.0, 0.0),
            (0.0, 100.0),
            _robertson,
            method="Radau",
            time_sampling="log",
            conserved_total=True,
            zero_initial_scale=1e-8,
        ),
    )
}





def _time_grid(system: OdeSystem, n_points: int) -> np.ndarray:
    if system.time_sampling == "log":
        grid: np.ndarray = np.r_[
            system.t_span[0], np.geomspace(1e-6, system.t_span[1], n_points - 1)
        ]
        return grid
    return np.linspace(*system.t_span, n_points)


def _initial_state(
    system: OdeSystem, base: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    y0 = base * (1.0 + rng.uniform(-0.12, 0.12, base.shape))
    zero_count = np.count_nonzero(base == 0)
    if system.conserved_total and zero_count:
        y0[base == 0] = rng.uniform(0.0, system.zero_initial_scale, zero_count)
    else:
        y0[base == 0] = rng.uniform(-0.2, 0.2, zero_count)
    if system.conserved_total:
        y0 *= 1.0 / y0.sum()
    return y0


def _check_settings(
    system: str, n_trajectories: int, n_points: int, rtol: float, atol: float
) -> None:
    if system not in ODE_SYSTEMS:
        raise ValueError(
            f"unknown system {system!r}; choose from {sorted(ODE_SYSTEMS)}"
        )
    if n_trajectories < 2 or n_points < 3:
        raise ValueError(
            "at least two trajectories and three time points are required; got "
            f"n_trajectories={n_trajectories}, n_points={n_points}"
        )
    if not (math.isfinite(rtol) and math.isfinite(atol)) or min(rtol, atol) <= 0:
        raise ValueError(
            "solver tolerances must be finite and positive; "
            f"got rtol={rtol}, atol={atol}"
        )


def _solve(
    system: OdeSystem,
    y0: np.ndarray,
    t: np.ndarray,
    rtol: float,
    atol: float,
    *,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:

    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        system.derivative,
        system.t_span,
        y0,
        t_eval=t,
        method=system.method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success or sol.t.size != t.size or not np.isfinite(sol.y).all():
        raise RuntimeError(f"ODE solve failed for trajectory {index}: {sol.message}")
    return sol.t, sol.y


def generate_ode_trajectories(
    system: str,
    *,
    seed: int,
    n_trajectories: int = 6,
    n_points: int = 241,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> TrajectorySet:
    _check_settings(system, n_trajectories, n_points, rtol, atol)
    spec = ODE_SYSTEMS[system]
    rng = np.random.default_rng(seed)
    base = np.asarray(spec.initial, float)
    t = _time_grid(spec, n_points)
    trajectories = []
    for i in range(n_trajectories):
        y0 = _initial_state(spec, base, rng)
        t_solved, state = _solve(spec, y0, t, rtol, atol, index=i)
        trajectories.append(
            Trajectory(id=f"{system}_traj_{i:02d}", t=t_solved, state=state, params={})
        )
    return TrajectorySet(
        name=system,
        source=(
            "kd.data.generate_ode_trajectories, port of KnowledgeDiscover "
            f"af11577:kd/dataset/_ode_core.py; system {system}, seed {seed}"
        ),
        time_var="t",
        state_vars=spec.state_vars,
        param_names=(),
        trajectories=tuple(trajectories),
    )


__all__ = [
    "ODE_CORE_TRACK",
    "ODE_SYSTEMS",
    "OdeSystem",
    "RightHandSide",
    "generate_ode_trajectories",
]
