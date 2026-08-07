# Bundled Dataset Registry

This directory is the canonical location for small benchmark data shipped with
the `kd` package. Keep data files flat where practical. Use descriptive
filenames instead of upstream generic names such as `data.mat`.

Rules:

- Do not store duplicate data files with identical content hashes.
- Prefer one flat canonical filename per dataset, e.g. `eqgpt_allen_cahn.mat`.
- Keep multi-file datasets flat only when the existing loader expects separate
  coordinate/value files.
- Do not vendor external source code here. EqGPT entries below are data files
  only; kd does not include the EqGPT codebase.
- When adding or replacing a dataset, update this registry and the relevant
  loader/catalog tests in the same change.

## Active Loader Datasets

| Dataset id | File(s) | Source | Equation | Shape / axes | Loader |
| --- | --- | --- | --- | --- | --- |
| `burgers` | `Burgers_equation.mat` | SGA-PDE; identical to EqGPT `Burgers_equation/burgers_sine.mat` | `u_t = -u*u_x + 0.1*u_xx` | 256 x 201, `(x,t)` | `load_burgers` |
| `chafee-infante` | `chafee_infante_CI.npy`, `chafee_infante_x.npy`, `chafee_infante_t.npy` | SGA-PDE; identical to EqGPT `Chaffee_Infante_equation/{CI,x,t}.npy` | `u_t = u_xx - u + u^3` | 301 x 200, `(x,t)` | `load_chafee_infante` |
| `kdv` | `KdV_equation.mat` | SGA-PDE; identical to EqGPT `KdV_equation/KdV-PINN.mat` | `u_t = -u*u_x - 0.0025*u_xxx` | raw 512 x 201; loader returns canonical 256 x 201 | `load_kdv` |
| `pde-divide` | `PDE_divide.npy` | SGA-PDE; identical to EqGPT `PDE_divide/PDE_divide.npy` | `u_t = -u_x/x + 0.25*u_xx` | 100 x 251, `(x,t)` | `load_pde_divide` |
| `pde-compound` | `PDE_compound.npy` | SGA-PDE; not equivalent to EqGPT `PDE_compound/PDE_compound.csv` | `u_t = u*u_xx + u_x^2` | 100 x 251, `(x,t)` | `load_pde_compound` |
| `allen-cahn` | `eqgpt_allen_cahn.mat` | EqGPT `Allen_Cahn/Allen_Cahn.mat` | `u_t = 0.003*u_xx + u - u^3` | 256 x 201, `(x,t)` | `load_allen_cahn` |
| `eqgpt-burgers-2d` | `eqgpt_burgers_2d.mat` | EqGPT `Burgers_2D/Burgers2D.mat` | `u_t + u*u_x + u*u_y - 0.01*(u_xx + u_yy) = 0` | 101 x 51 x 100, `(x,y,t)` | `load_burgers_2d` |
| `convection-diffusion` | `eqgpt_convection_diffusion.mat` | EqGPT `Convection_diffusion_equation/data.mat` | `u_t = -u_x + 0.25*u_xx` | 256 x 100, `(x,t)` | `load_convection_diffusion` |
| `eqgpt-eq-6-2-12` | `eqgpt_eq_6_2_12.csv` | EqGPT `Eq_6_2_12/data_Eq_6_2_12.csv` | `0.1*u_xt + u_t + 0.1*u_x = 0` | 501 x 501, `(x,t)` | `load_eq_6_2_12` |
| `eqgpt-laplacian-eitech` | `eqgpt_laplacian_eitech.xlsx` | EqGPT `Laplacian_EITech/Laplacian_EITech.xlsx` | `u_xx + u_yy + 1 = 0` | 48,367 valid scattered `(x,y)` points | `kd.get_dataset("eqgpt-laplacian-eitech").loader()` |
| `eqgpt-laplacian-smile` | `eqgpt_laplacian_smile.xlsx` | EqGPT `Laplacian_smile/Laplacian_smile.xlsx` | `u_xx + u_yy = 0` | 44,711 valid scattered `(x,y)` points | `kd.get_dataset("eqgpt-laplacian-smile").loader()` |
| `eqgpt-poisson-disk` | `eqgpt_poisson_disk.xlsx` | EqGPT `Possion_equation/Possion_x_y.xlsx` | `u_xx + u_yy = 0` on disk/polar coordinates | 40,200 valid scattered `(x,y)` points | `kd.get_dataset("eqgpt-poisson-disk").loader()` |
| `wave` | `eqgpt_wave.mat` | EqGPT `Wave_equation/wave.mat` | `u_tt = u_xx` | 161 x 321, `(x,t)` | `load_wave` |
| `klein-gordon` | `eqgpt_klein_gordon.mat` | EqGPT `KG_equation/KG_Exp.mat` | `u_tt = 0.5*u_xx - 5*u` | 201 x 201, `(x,t)` | `load_klein_gordon` |

## Real-World Experimental Datasets (Tabular)

Measured data, not solver output; exposed as `y = f(X)` tables (scattered /
aggregated points, no regular grid), outside the PDE catalog.

| Dataset id | File | Columns | Meaning | Loader |
| --- | --- | --- | --- | --- |
| `wave-breaking-N_G2Tp12A100_broad` | `wave_breaking_N_G2Tp12A100_broad.npz` | `t`, `x`, `eta` | camera-reconstructed surface elevation (314,478 scattered points) of one focused wave group toward breaking | `load_wave_breaking()` |
| `tlc-cc-start` | `tlc_cc_Rf_t1.npy` | 0=`R_F`, 1=`r`, 2=`V_S` | mean start retention volume over 74 `(R_F, r)` conditions | `load_tlc_cc(target="start")` |
| `tlc-cc-end` | `tlc_cc_Rf_t2.npy` | 0=`R_F`, 1=`r`, 2=`V_E` | mean end retention volume over 74 `(R_F, r)` conditions | `load_tlc_cc(target="end")` |

**Wave breaking** (Xu, H. et al. *Nat Commun* **16**, 10255 (2025),
doi:10.1038/s41467-025-65114-2; experimental background in the paper's SI):
wave-tank measurements of focused wave groups approaching breaking, Imperial
College London. The paper analysed 12 experiments; kd bundles one of them
(`N_G2Tp12A100_broad`, 314,478 points). Further cases can be loaded from a
local directory via `load_wave_breaking(case=..., data_dir=...)`.

**TLC-CC** (Xu, H. et al. *Nat Commun* **16**, 832 (2025),
doi:10.1038/s41467-025-56136-x; background in the paper's SI): automated
column chromatography, 192 compounds on 4 g silica columns; the two tables
hold the mean start/end retention volumes over 74 `(R_F, r)` conditions. The
raw files carry four further columns (indices 3-6); the loader exposes
columns 0-2 only.

## Bundled Reference Data Without Public Loader

These files are retained for benchmark coverage or future loader work.

| Dataset id | File | Source | Equation / description | Shape / axes | Loader |
| --- | --- | --- | --- | --- | --- |
| `eqgpt-pde-compound` | `eqgpt_pde_compound.csv` | EqGPT `PDE_compound/PDE_compound.csv` | `u_t - 0.2*(u*u_x)_x = 0` | 200 x 200, `(x,t)` | none |

## Referenced Third-Party Assets (Not Bundled)

Assets a plugin resolves at runtime but that kd does not vendor under this
directory (too large to bundle); each is distributed separately by its authors.

| Asset | Source | Used by | Resolution |
| --- | --- | --- | --- |
| `PDEGPT_wave_breaking.pt` (~152 MB) | Pretrained GPT-proposer checkpoint from the EqGPT project (Xu, H. et al., [EqGPT](https://github.com/woshixuhao/EqGPT), *Nat Commun* **16**, 10255 (2025)); distributed separately by the authors | `kd.search.eqgpt.backend.RealGPTBackend` (the `eqgpt` search plugin); `examples/16_eqgpt.py` | Point `KD_EQGPT_ASSET_DIR` at a directory containing `gpt_model/PDEGPT_wave_breaking.pt`, or pass `weights_path=`/`asset_dir=` to `RealGPTBackend.from_assets` (see `resolve_asset_path` in the same module) |
| `WaveBreaking.pkl` (~199 MB) | Wave-breaking laboratory experiments (23 cases, Imperial College London wave tank; Xu, H. et al. *Nat Commun* **16**, 10255 (2025)), distributed via HF `Spac1ly/KnowledgeDiscover` (pinned mirror `timeoutHao/KD-data`) | `kd.load_wave_breaking` / `kd.data.loaders.load_wave_breaking_cases` | Default path `<repo-root>/data/hf-knowledgediscover/WaveBreaking.pkl` (git-ignored); the loader error names the HF source when absent, or pass an explicit path |
| Wave surrogate checkpoint tree | Per-case neural surrogates for the wave-breaking experiments, trained in the predecessor (the retired sibling checkout) | `kd.data.loaders.wave_breaking` surrogate path (`wave_surrogate_checkpoint_path`, `resolve_v1_wave_asset_dir`) | Config `v1_asset_dir` or the `KD_V1_WAVE_ASSETS` environment variable, else fail-loud |
