
import sys
from pathlib import Path

import kd
from kd.data.loaders.wave_breaking import (
    default_wave_pkl_path,
    load_wave_breaking_cases,
    wave_breaking_case_to_dataset,
    wave_surrogate_checkpoint_path,
)
from kd.search.eqgpt.backend import ASSET_ENV_VAR, resolve_asset_path
from kd.search.eqgpt.config import EqGPTConfig

_DEMO_SEED = 21



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

wave_pkl = default_wave_pkl_path()
if not wave_pkl.exists():
    print(
        "The WaveBreaking data pickle is not available, so this example cannot "
        f"run.\n\nExpected at: {wave_pkl}\n\n"
        "Fix: place WaveBreaking.pkl at that path (the repo-relative default; "
        "it is resolved from the repo root, NOT from KD_EQGPT_ASSET_DIR)."
    )
    sys.exit(1)
print(f"Using pretrained EqGPT weights: {weights_path}")
print(f"Using wave-breaking data: {wave_pkl}")




cases = load_wave_breaking_cases(wave_pkl)
primary_name = sorted(name for name in cases if "N" in name)[0]
primary_ds = wave_breaking_case_to_dataset(cases[primary_name])
del cases
print(f"Primary case (of 12 N-cases): {primary_name}")




try:
    wave_surrogate_checkpoint_path(primary_name)
except FileNotFoundError as exc:
    print(
        "The per-case v1 surrogate checkpoints are not available, so this "
        f"example cannot run.\n\n{exc}\n\n"
        "Fix: set KD_V1_WAVE_ASSETS to the EqGPT_wave_breaking directory that "
        "holds model_save/wave_breaking/95_0_<case>(Non_unit)/Net_Sin_*.pkl."
    )
    sys.exit(1)



model = kd.Model(
    algorithm="eqgpt",
    generations=5,
    config=EqGPTConfig.wave_preset(seed=_DEMO_SEED, primary_case=primary_name),
)



model.fit(primary_ds)

print()
print(f"{'Reproduced structure':<21}: {model.best_expr_}")
print(f"{'Best mean reward':<21}: {model.best_score_:.4f}")




out_dir = Path(__file__).parent / "out" / "wave_breaking"
viz = kd.VizEngine(output_dir=out_dir)
report = viz.render_all(
    model.result_,
    algorithm=model.algorithm_,
    dataset=primary_ds,
)
print(f"{'Report':<21}: {report.report}")
print(f"{'Figures':<21}: {len(report.figures)} files")
