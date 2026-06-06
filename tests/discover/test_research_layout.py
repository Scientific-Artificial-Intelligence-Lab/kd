
from __future__ import annotations

from pathlib import Path

import pytest






import kd.search.discover.tokens.prior

REPO_ROOT = Path(__file__).resolve().parent.parent.parent





MOVED_FILES: list[tuple[str, str]] = [
    (
        "scripts/discover/analysis/task4_lsq_probe.py",
        "scripts/discover/research/analysis/task4_lsq_probe.py",
    ),
    (
        "scripts/discover/analysis/task4_p22_aggregate.py",
        "scripts/discover/research/analysis/task4_p22_aggregate.py",
    ),
    (
        "scripts/discover/analysis/task4_parity_ablation.py",
        "scripts/discover/research/analysis/task4_parity_ablation.py",
    ),
    (
        "scripts/discover/analysis/task4_pinn_denoising_isolation.py",
        "scripts/discover/research/analysis/task4_pinn_denoising_isolation.py",
    ),
    (
        "scripts/discover/analysis/task4_codex_solver_reward_probe.py",
        "scripts/discover/research/analysis/task4_codex_solver_reward_probe.py",
    ),
    (
        "scripts/discover/analysis/task4_reward_landscape.py",
        "scripts/discover/research/analysis/task4_reward_landscape.py",
    ),
    (
        "scripts/discover/analysis/wave3_heartbeat_analysis.py",
        "scripts/discover/research/analysis/wave3_heartbeat_analysis.py",
    ),
    (
        "scripts/discover/run_phase3_2_task3_multiseed.py",
        "scripts/discover/research/run_phase3_2_task3_multiseed.py",
    ),
    (
        "scripts/discover/run_phase3_2_task4_multiseed.py",
        "scripts/discover/research/run_phase3_2_task4_multiseed.py",
    ),
    (
        "scripts/discover/run_td086_vocab_ablation.py",
        "scripts/discover/research/run_td086_vocab_ablation.py",
    ),
    (
        "scripts/discover/run_impl35_multiseed.py",
        "scripts/discover/research/run_impl35_multiseed.py",
    ),
    (
        "scripts/discover/run_impl35_l1_ablation.py",
        "scripts/discover/research/run_impl35_l1_ablation.py",
    ),
    (
        "scripts/discover/diagnose_mode2_burgers.py",
        "scripts/discover/research/diagnose_mode2_burgers.py",
    ),
    (
        "scripts/discover/pilot_allen_cahn_2d_paper.py",
        "scripts/discover/research/pilot_allen_cahn_2d_paper.py",
    ),
    (
        "scripts/discover/slurm/run_p21_scaffold.sbatch",
        "scripts/discover/research/slurm/run_p21_scaffold.sbatch",
    ),
    (
        "scripts/discover/slurm/run_p22_pinn_fp.sbatch",
        "scripts/discover/research/slurm/run_p22_pinn_fp.sbatch",
    ),
    (
        "scripts/discover/slurm/run_p22_epoch_grid.sh",
        "scripts/discover/research/slurm/run_p22_epoch_grid.sh",
    ),
]


KEPT_PUBLIC: list[str] = [
    "scripts/discover/run_discover.py",
    "scripts/discover/run_mode2_burgers.py",
    "scripts/discover/run_mode2_chafee.py",
    "scripts/discover/run_mode2_allen_cahn_2d.py",
    "scripts/discover/run_burgers_comparison.py",
    "scripts/discover/run_chafee_comparison.py",
    "scripts/discover/check_gpu.py",
    "scripts/discover/gen_fixture_library.py",
    "scripts/discover/generate_allen_cahn_2d.py",
    "scripts/discover/generate_allen_cahn_2d_spectral.py",
    "scripts/discover/inspect_mode2_burgers_results.py",
    "scripts/discover/analysis/compute_1d_gates.py",
    "scripts/discover/slurm/run.sbatch",
]


@pytest.mark.unit
class TestResearchScriptsMoved:

    @pytest.mark.parametrize(("old_path", "new_path"), MOVED_FILES)
    def test_file_at_new_path_only(self, old_path: str, new_path: str) -> None:
        new_full = REPO_ROOT / new_path
        old_full = REPO_ROOT / old_path
        assert new_full.exists(), f"Expected moved file at new path: {new_path}"
        assert not old_full.exists(), (
            f"Stale copy at old path: {old_path} (should have been moved)"
        )


@pytest.mark.unit
class TestPublicScriptsSurfaceClean:

    def test_no_task4_files_in_public_analysis(self) -> None:
        leftover = list((REPO_ROOT / "scripts/discover/analysis").glob("task4_*"))
        assert leftover == [], f"task4_* files still in public analysis: {leftover}"

    def test_no_phase3_2_files_in_public_scripts(self) -> None:
        leftover = list((REPO_ROOT / "scripts/discover").glob("run_phase3_2_*"))
        assert leftover == [], (
            f"run_phase3_2_* files still in public scripts: {leftover}"
        )

    def test_no_td086_in_public_scripts(self) -> None:
        path = REPO_ROOT / "scripts/discover/run_td086_vocab_ablation.py"
        assert path.exists() is False, f" ablation still public: {path}"

    def test_no_impl35_files_in_public_scripts(self) -> None:
        leftover = list((REPO_ROOT / "scripts/discover").glob("run_impl35_*"))
        assert leftover == [], f"run_impl35_* files still in public scripts: {leftover}"

    def test_no_diagnose_mode2_burgers_in_public(self) -> None:
        path = REPO_ROOT / "scripts/discover/diagnose_mode2_burgers.py"
        assert path.exists() is False, f"diagnose_mode2_burgers still public: {path}"

    def test_no_pilot_allen_cahn_2d_paper_in_public(self) -> None:
        path = REPO_ROOT / "scripts/discover/pilot_allen_cahn_2d_paper.py"
        assert path.exists() is False, f"AC-2D pilot still public: {path}"

    def test_no_wave3_heartbeat_in_public_analysis(self) -> None:
        path = REPO_ROOT / "scripts/discover/analysis/wave3_heartbeat_analysis.py"
        assert path.exists() is False, f"wave3 heartbeat still public: {path}"

    def test_no_p21_p22_sbatch_in_public_slurm(self) -> None:
        leftover = list((REPO_ROOT / "scripts/discover/slurm").glob("run_p2[12]_*"))
        assert leftover == [], f"P21/P22 sbatch still in public slurm: {leftover}"


@pytest.mark.unit
class TestProductionEntriesStillPublic:

    @pytest.mark.parametrize("path", KEPT_PUBLIC)
    def test_kept_public_exists(self, path: str) -> None:
        full = REPO_ROOT / path
        assert full.exists(), f"Production entry vanished: {path}"

    def test_ops_directory_still_present(self) -> None:
        ops_dir = REPO_ROOT / "scripts/discover/ops"
        assert ops_dir.is_dir(), "scripts/discover/ops/ directory missing"

        assert any(ops_dir.iterdir()), "scripts/discover/ops/ is empty after move"


@pytest.mark.unit
class TestResearchDirectoriesExist:

    @pytest.mark.parametrize(
        "path",
        [
            "scripts/discover/research",
            "scripts/discover/research/analysis",
            "scripts/discover/research/slurm",
        ],
    )
    def test_directory_exists(self, path: str) -> None:
        full = REPO_ROOT / path
        assert full.is_dir(), f"Research landing dir missing: {path}"


@pytest.mark.unit
class TestNoStaleSbatchInFileRefs:





    _STALE_PREFIXES: tuple[str, ...] = (
        "scripts/discover/analysis/task4_",
        "scripts/discover/slurm/run_p2",
    )

    @pytest.mark.parametrize(
        "path",
        [
            "scripts/discover/research/slurm/run_p21_scaffold.sbatch",
            "scripts/discover/research/slurm/run_p22_pinn_fp.sbatch",
            "scripts/discover/research/slurm/run_p22_epoch_grid.sh",
        ],
    )
    def test_sbatch_does_not_reference_old_task4_path(self, path: str) -> None:
        full = REPO_ROOT / path
        assert full.exists(), f"Required input file missing: {path}"
        content = full.read_text()
        for stale in self._STALE_PREFIXES:
            assert stale not in content, (
                f"{path} still references stale '{stale}' path. "
                f"After + an internal milestone kd merge it should point at the "
                f"corresponding 'scripts/discover/research/...' location."
            )

    def test_p22_aggregate_self_reference_rewritten(self) -> None:
        path = REPO_ROOT / "scripts/discover/research/analysis/task4_p22_aggregate.py"
        assert path.exists(), f"Required input file missing: {path}"
        content = path.read_text()




        assert "scripts/discover/analysis/task4_p22_aggregate.py" not in content, (
            "task4_p22_aggregate.py still has stale self-reference to "
            "'scripts/discover/analysis/task4_p22_aggregate.py'. Update the "
            "module docstring / usage line to "
            "'scripts/discover/research/analysis/task4_p22_aggregate.py'."
        )

    @pytest.mark.parametrize(
        ("path", "stale_substring"),
        [
            (
                "scripts/discover/research/run_phase3_2_task4_multiseed.py",
                "scripts/discover/run_phase3_2_",
            ),
            (
                "scripts/discover/research/run_phase3_2_task3_multiseed.py",
                "scripts/discover/run_phase3_2_",
            ),
            (
                "scripts/discover/research/run_impl35_multiseed.py",
                "scripts/discover/run_impl35_",
            ),
            (
                "scripts/discover/research/run_impl35_l1_ablation.py",
                "scripts/discover/run_impl35_",
            ),
            (
                "scripts/discover/research/run_td086_vocab_ablation.py",
                "scripts/discover/run_td086",
            ),
            (
                "scripts/discover/research/diagnose_mode2_burgers.py",
                "scripts/discover/diagnose_mode2_burgers",
            ),
            (
                "scripts/discover/research/pilot_allen_cahn_2d_paper.py",
                "scripts/discover/pilot_allen_cahn_2d_paper",
            ),
        ],
    )
    def test_moved_python_scripts_no_stale_self_refs(
        self, path: str, stale_substring: str
    ) -> None:





        full = REPO_ROOT / path
        assert full.exists(), f"Required input file missing: {path}"
        content = full.read_text()
        assert stale_substring not in content, (
            f"{path} still contains stale path substring "
            f"'{stale_substring}'. Update docstrings / usage hints to "
            f"the 'scripts/discover/research/...' form."
        )


@pytest.mark.unit
class TestSiblingImportBootstrap:

    def test_phase3_2_task4_multiseed_imports_succeed(self) -> None:




        import importlib.util

        target = REPO_ROOT / "scripts/discover/research/run_phase3_2_task4_multiseed.py"
        spec = importlib.util.spec_from_file_location(
            "phase3_2_task4_multiseed_smoke", target
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

        assert hasattr(module, "run_mode2_seed")
