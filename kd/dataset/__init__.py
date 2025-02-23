from ._base import (PDEDataset,
                    load_burgers_equation,
                    load_kdv_equation,
                    load_mat_file)

__all__ = [
    "PDEDataset",
    "load_burgers_equation",
    "load_mat_file",
    "load_kdv_equation",
    "Burgers_equation_shock",
    "KdV_equation",
    "KdV_equation_sine",
    "Chaffee_Infante_equation",
    "KG_equation",
    "Allen_Cahn_equation",
    "Convection_diffusion_equation_solution",
    "Convection_diffusion_equation_simulation",
    "Wave_equation",
    "KS_equation",
    "Beam_equation",
    "Heat_equation",
    "Heat_equation_sin",
    "Diffusion_equation",
    "Parametric_convection_diffusion",
    "Parametric_Burgers_equation",
    "Parametric_wave_equation",
    "Burgers_2D"
]
