
from __future__ import annotations

import json
import math

import pytest
import torch

from kd.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd.search.pysr import PySRConfig


_SMOKE_NX = 40
_SMOKE_NT = 20


def _smoke_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 2.0 * math.pi, _SMOKE_NX, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, _SMOKE_NT, dtype=torch.float64)
    xg, tg = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xg) * torch.exp(-tg)
    return PDEDataset(
        name="pysr-smoke",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.mark.pysr
@pytest.mark.slow
def test_model_pysr_real_fit_smoke() -> None:
    from kd import Model
    from kd.search.iteration_events import IterationEvent, IterationEventEmitter

    dataset = _smoke_dataset()
    config = PySRConfig(



        niterations=5,
        populations=4,
        population_size=20,
        maxsize=15,
        terms=("u", "u_x", "u_xx"),
        seed=0,
    )



    events: list[IterationEvent] = []
    emitter = IterationEventEmitter(on_event=events.append)
    model = Model(algorithm="pysr", verbose=False, config=config, callbacks=[emitter])
    model.fit(dataset)

    assert len(events) == 1
    assert events[0].iteration == 0

    assert isinstance(model.best_expr_, str)
    assert model.best_expr_
    assert isinstance(model.best_score_, float)
    assert math.isfinite(model.best_score_)

    manifest = model.result_.manifest
    assert manifest is not None
    assert manifest.terms is not None
    assert isinstance(manifest.terms, list)
    assert manifest.terms

    encoded = json.dumps(model.result_.to_dict(), allow_nan=False)
    reloaded = json.loads(encoded)
    assert reloaded["manifest"]["terms"] == manifest.terms
