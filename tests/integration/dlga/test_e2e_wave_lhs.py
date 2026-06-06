
from __future__ import annotations

import torch
import torch.nn as nn

from kd.core.platform.builder import PlatformBuilder
from kd.data.synthetic import generate_wave_xu2020_data
from kd.search.dlga import DLGAConfig, DLGAPlugin
from kd.search.runner import ExperimentRunner


class _ExactWaveModel(nn.Module):

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        u = torch.sin(x) * torch.cos(t) + 0.5 * torch.sin(2.0 * x) * torch.cos(2.0 * t)
        return {"u": u}


def test_runner_preserves_dlga_selected_u_tt_label() -> None:
    dataset = generate_wave_xu2020_data(nx=24, nt=13, noise_level=0.0)
    plugin = DLGAPlugin(
        DLGAConfig(pop_size=4, seed=0, epsilon=0.0),
        surrogate_model=_ExactWaveModel(),
    )


    components = PlatformBuilder(dataset, plugin.derivative_requirements).build()
    plugin.state = {
        "population": [[[2]], [[0]], [[1]], [[3]]],
        "best_score": float("inf"),
        "best_expression": "",
    }

    result = ExperimentRunner(
        plugin,
        max_iterations=1,
        batch_size=4,
    ).run(components)

    assert result.lhs_label == "u_tt"
    assert result.final_eval.expression == "u_xx"
    assert result.final_eval.mse < 1e-20
    torch.testing.assert_close(
        result.final_eval.coefficients,
        torch.tensor([1.0], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )
