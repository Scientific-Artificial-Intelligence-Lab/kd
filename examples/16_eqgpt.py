
import sys

import kd
from kd.search.eqgpt.backend import ASSET_ENV_VAR, resolve_asset_path
from kd.search.eqgpt.config import EqGPTConfig




try:
    weights_path = resolve_asset_path()
except FileNotFoundError as exc:
    print(
        "EqGPT pretrained weights are not available, so this example cannot "
        f"run.\n\n{exc}\n\n"
        f"Fix: place PDEGPT_wave_breaking.pt under a 'gpt_model/' directory and "
        f"either put it at the repo-root a reference library location "
        f"or set {ASSET_ENV_VAR} to its parent asset directory."
    )
    sys.exit(1)
print(f"Using pretrained EqGPT weights: {weights_path}")


dataset = kd.generate_burgers_data(nx=256, nt=101, nu=0.1, seed=42)
print(f"Ground truth: {dataset.ground_truth}")




model = kd.Model(
    algorithm="eqgpt",
    generations=3,
    config=EqGPTConfig.burgers_preset(),
)


model.fit(dataset)


print()
print(f"Discovered: {model.best_expr_}")
print(f"Best reward: {model.best_score_:.4f}")
