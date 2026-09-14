"""Example 16 - EqGPT: pretrained generative GPT proposer for Burgers.

EqGPT (Xu et al.) is a GPT that GENERATES candidate PDE sentences: a
pretrained transformer proposes SAT-legal token sequences, kd converts each
to canonical-IR terms, scores them by a sparsity-penalized regression reward,
and fine-tunes the GPT toward high-reward equations. The kd facade plugs it
into the same one-line API as every other packaged engine.

Burgers: u_t + u * u_x - 0.1 * u_xx = 0 (i.e., u_t = -u * u_x + 0.1 * u_xx)

Unlike the other algorithms, EqGPT needs TWO things a bare install lacks:

1. A per-problem ``sparsity_alpha`` (decision D5: there is no universal
   default). ``EqGPTConfig.burgers_preset()`` pins the probe-verified 0.02.
2. The pretrained GPT weights (``PDEGPT_wave_breaking.pt``) from the EqGPT
   authors. They are too large for the wheel, so the first run downloads them
   (151.7 MB) from the KD Hub mirror into the Hugging Face cache; a local
   copy under ``$KD_EQGPT_ASSET_DIR/gpt_model/`` is used instead when that
   variable is set. This script resolves them BEFORE fitting and exits with
   the reason if neither route works.

Run: python examples/16_eqgpt.py
"""

import sys

import kd
from kd import EqGPTConfig
from kd.search.eqgpt import ASSET_ENV_VAR, resolve_asset_path

# 0. Resolve the pretrained weights before fitting, with the SAME resolution
# the plugin/backend uses (weights_path -> asset_dir -> $KD_EQGPT_ASSET_DIR
# -> Hub download), so this check cannot drift from the real load.
try:
    weights_path = resolve_asset_path()
except FileNotFoundError as exc:
    print(
        "EqGPT pretrained weights are not available, so this example cannot "
        f"run.\n\n{exc}\n\n"
        "Either the Hub download failed (no network?) or a local tree was "
        f"named but is incomplete: {ASSET_ENV_VAR} must point at a directory "
        "holding gpt_model/PDEGPT_wave_breaking.pt."
    )
    sys.exit(1)
print(f"Using pretrained EqGPT weights: {weights_path}")

# 1. Generate synthetic data with a known ground truth.
dataset = kd.generate_burgers_data(nx=256, nt=101, nu=0.1, seed=42)
print(f"Ground truth: {dataset.ground_truth}")

# 2. Configure the model. ``algorithm="eqgpt"`` selects the GPT-proposer plugin;
# ``config=`` is MANDATORY (D5: sparsity_alpha has no facade default). The
# Burgers preset pins the probe-verified sparsity_alpha=0.02.
model = kd.Model(
    algorithm="eqgpt",
    generations=3,
    config=EqGPTConfig.burgers_preset(),
)

# 3. Fit. Progress prints to stdout because verbose=True (default).
model.fit(dataset)

# 4. Inspect the result.
print()
print(f"Discovered: {model.result_.equation}")
print(f"Best reward: {model.best_score_:.4f}")
