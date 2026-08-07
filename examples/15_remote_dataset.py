"""Example 15 - On-demand remote datasets (HuggingFace).

Some kd datasets are fetched on demand from HuggingFace, cached locally,
checksum-verified, and pinned to a revision. REQUIREMENTS: network access
and the optional hub extra, installed with ``uv sync --extra hub``.
This is not part of the offline quick start and is not run in CI.

Run: python examples/15_remote_dataset.py
"""

import kd


def main() -> None:
    """Fetch one remote dataset and run a tiny fit."""
    print("Available remote datasets:")
    for spec in kd.list_remote_datasets():
        print(f" {spec.id}: {spec.equation}")

    print()
    print("Loading llm4ed-fisher-nonlinear from HuggingFace...")
    ds = kd.load_from_hub("llm4ed-fisher-nonlinear")

    kd.preview(ds)

    model = kd.Model(
        generations=3,
        population=8,
        seed=0,
        verbose=False,
    )
    model.fit(ds)

    print()
    print(f"Discovered: {model.best_expr_}")
    print(f"Ground truth: {ds.ground_truth}")


if __name__ == "__main__":
    main()
