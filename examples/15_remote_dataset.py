
import kd


def main() -> None:
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
