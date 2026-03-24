"""KD_EqGPT example with unified visualization output.

This script runs the pre-trained EqGPT workflow, logs the top discovered
equations, and saves visualization outputs to ``artifacts/eqgpt_example/``.

Usage:
    /Users/hao/miniconda3/envs/kd-env/bin/python examples/eqgpt_example.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from kd.model import KD_EqGPT
from kd.viz.core import VizRequest, configure, render

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "eqgpt_example"
_VIZ_KINDS = ("equation", "reward_ranking", "reward_evolution", "parity")


def _save_result_summary(result: Dict[str, object]) -> Path:
    """Persist the EqGPT search summary to JSON."""
    summary_path = _OUTPUT_DIR / "result_summary.json"
    payload = {
        "best_equation": result.get("best_equation", ""),
        "best_reward": result.get("best_reward", 0.0),
        "equations": result.get("equations", []),
        "rewards": result.get("rewards", []),
        "reward_history": result.get("reward_history", []),
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def _log_top_equations(equations: List[str], rewards: List[float]) -> None:
    """Log the discovered equation ranking."""
    logger.info("Top-%d equations:", len(equations))
    for idx, (equation, reward) in enumerate(zip(equations, rewards), start=1):
        logger.info("  %02d. [reward=%.4f] %s", idx, reward, equation)


def main() -> None:
    """Run EqGPT discovery and save visualization outputs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = KD_EqGPT(
        optimize_epochs=5,
        samples_per_epoch=400,
        case_filter="N",
        seed=0,
    )
    result = model.fit_pretrained()

    equations = list(result.get("equations", []))
    rewards = [float(value) for value in result.get("rewards", [])]

    logger.info("EqGPT wave-breaking PDE discovery")
    logger.info("Best equation: %s", result.get("best_equation", ""))
    logger.info("Best reward: %.4f", float(result.get("best_reward", 0.0)))
    _log_top_equations(equations, rewards)

    summary_path = _save_result_summary(result)
    logger.info("Saved summary: %s", summary_path)

    configure(save_dir=_OUTPUT_DIR)
    try:
        for kind in _VIZ_KINDS:
            viz_result = render(
                VizRequest(kind=kind, target=model, options={"output_dir": _OUTPUT_DIR})
            )
            if viz_result.warnings:
                logger.warning("%s warnings: %s", kind, "; ".join(viz_result.warnings))
                continue
            if viz_result.paths:
                logger.info("Saved %s: %s", kind, viz_result.paths[0])
    finally:
        configure(save_dir=None)


if __name__ == "__main__":
    main()
