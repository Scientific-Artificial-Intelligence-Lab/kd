
from __future__ import annotations

import pytest




CANONICAL_EXPR = "u_t = -1*mul(u_x, u) + 0.1002*diff2_x(u)"
EXPECTED_BEST_SCORE = -28.776750720652373


@pytest.mark.slow
def test_seeded_quickstart_reproduces_canonical_expression() -> None:
    import kd

    dataset = kd.load_burgers()
    model = kd.Model(algorithm="sga", generations=5, seed=0, verbose=False).fit(dataset)

    assert str(model.best_expr_) == CANONICAL_EXPR, (
        f"seeded quick-start produced {model.best_expr_!r}, expected "
        f"{CANONICAL_EXPR!r}. If this shifted (e.g. PyTorch upgrade), "
        "re-derive the canonical string and regenerate README + notebook."
    )
    assert model.best_score_ == pytest.approx(EXPECTED_BEST_SCORE, rel=1e-9)
