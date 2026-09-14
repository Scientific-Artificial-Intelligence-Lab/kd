
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import kd
from kdagent.data_source import load_dataset, resolve_input
from kd.core.equation import sketch_from_dict
from kd.search import RUNDIR_SCHEME, RunDirPaths, RunRecord, run_id_of_run_dir

from kdagent import criteria, priors, surrogate_training
from kdagent.lineage import PLATFORM_FRAME, SegmentEntry, SegmentLedger
from kdagent.platform_eval import PlatformEvaluation, lhs_order_of, platform_evaluation
from kdagent.resume import (
    baseline,
    eligible_iterations,
    move_facts,
    report_projection,
    schema_of,
    segment_action,
)
from kdagent.segment_report import build_segment_report, law_from_evidence
from kdagent.surrogates import (
    REGISTRY_DIRNAME,
    SurrogateEntry,
    SurrogateRegistry,
    file_sha256,
)




FEATURES_FORMAT = "kdagent-features-v1"
FEATURES_DIR = "features"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace: Path = args.workspace
    try:
        payload = _dispatch(args, SegmentLedger(workspace), workspace)
    except Exception as exc:
        json.dump(
            {"error": type(exc).__name__, "detail": str(exc)},
            sys.stderr,
            ensure_ascii=False,
        )
        sys.stderr.write("\n")
        return 1
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _dispatch(
    args: argparse.Namespace, ledger: SegmentLedger, workspace: Path
) -> dict[str, Any]:
    if args.command == "record":
        return _record(
            ledger,
            workspace,
            run_dir=args.run_dir,
            parent=args.parent,
            decision_parent=args.decision_parent,
            reseed=args.reseed,
            resume_iteration=args.resume_iteration,
            params=None if args.params is None else json.loads(args.params),
            sketch=None if args.sketch is None else json.loads(args.sketch),
            surrogate=args.surrogate,
            platform_eval=args.platform_eval,
        )
    if args.command == "status":
        return _status(
            ledger,
            workspace,
            dataset=args.dataset,
            remaining_budget=args.remaining_budget,
            prior_path=args.prior,
        )
    if args.command == "preview":
        return _preview(workspace, dataset=args.dataset)
    if args.command == "surrogate":
        registry = SurrogateRegistry(workspace)
        if args.surrogate_command == "list":
            return _surrogate_list(registry)
        return _surrogate_train(
            registry,
            dataset=args.dataset,
            algorithm=args.algorithm,
            params=None if args.params is None else json.loads(args.params),
            seed=args.seed,
            max_seconds=args.max_seconds,
        )
    if args.command == "prune":
        ledger.prune(args.run_id, reason=args.reason)
        return {"event": "prune", "run_id": args.run_id, "reason": args.reason}
    if args.command == "select":
        ledger.select(args.run_id, reason=args.reason)
        return {"event": "select", "run_id": args.run_id, "reason": args.reason}
    return _decide(
        ledger,
        workspace,
        run_id=args.run_id,
        remaining_budget=args.remaining_budget,
        instrument=args.instrument,
        perturbations=json.loads(args.perturbations),
    )


def _record(
    ledger: SegmentLedger,
    workspace: Path,
    *,
    run_dir: str,
    parent: str | None,
    decision_parent: str | None,
    reseed: bool,
    resume_iteration: int | None,
    params: dict[str, Any] | None,
    sketch: dict[str, Any] | None = None,
    surrogate: str | None = None,
    platform_eval: bool = False,
) -> dict[str, Any]:
    if sketch is not None:
        sketch_from_dict(sketch)
    if surrogate is not None:
        SurrogateRegistry(workspace).entry(surrogate)
    path = _run_path(workspace, run_dir)
    run_id = run_id_of_run_dir(path)
    if run_id is None:
        raise ValueError(
            f"{path} is not a kd run directory: no readable {RUNDIR_SCHEME} "
            "manifest.json in it"
        )
    manifest = json.loads(RunDirPaths(root=path).manifest.read_text(encoding="utf-8"))
    status = manifest["status"]
    seed = manifest["seed"]
    sealed_record = RunDirPaths(root=path).record.is_file()





    try:
        source = None if parent is None else ledger.entry(parent)
    except KeyError:
        raise ValueError(
            f"segment {run_id!r} resumes from {parent!r}, which no segment "
            "event recorded"
        ) from None
    dataset = ""
    nmse: float | None = None
    law: dict[str, Any] | None = None
    platform_nmse: float | None = None
    platform_frame: str | None = None
    platform_coefficients: list[float] | None = None
    evaluation: PlatformEvaluation | None = None
    report: dict[str, Any] | None = None
    summary = None
    if status == "completed":
        evidence = RunRecord.load(RunDirPaths(root=path).record).evidence
        dataset = evidence.dataset_name
        nmse = evidence.nmse
        law = law_from_evidence(evidence)
        if platform_eval and evidence.is_valid and evidence.support:


            dataset_obj = load_dataset(dataset, resolve_input(workspace, dataset))
            evaluation = platform_evaluation(
                dataset_obj, evidence.support, lhs_order_of(evidence.catalog_fit)
            )
            platform_nmse = evaluation.nmse
            platform_coefficients = evaluation.coefficients
            if platform_nmse is not None:
                platform_frame = PLATFORM_FRAME
        inherited = None if parent is None else baseline(ledger, parent)
        baseline_run_id, parent_summary = (
            (None, None) if inherited is None else inherited
        )
        report, summary = build_segment_report(
            path, parent_summary, baseline_run_id=baseline_run_id
        )
    if seed is None or not sealed_record:
        if source is None:
            raise ValueError(
                f"{path} sealed no seed and no dataset of its own (status "
                f"{status!r}: kd writes a null seed and no record.json for a "
                "run that did not complete), so this row has nothing to be "
                "filed under. Pass --parent to inherit both from the segment "
                "this one resumed from"
            )
        seed = source.seed if seed is None else seed
        dataset = source.dataset if not sealed_record else dataset
    if source is not None:
        if sketch is None:
            sketch = source.sketch
        elif sketch != source.sketch:
            raise ValueError(
                f"run {parent!r} ran under a different sketch (or none), and a "
                "resumed segment keeps its lineage's sketch. Omit --sketch to "
                "inherit it; to search under this sketch, drop --parent and "
                "record the reason with --decision-parent"
            )
        if surrogate is None:
            surrogate = source.surrogate
        elif surrogate != source.surrogate:
            raise ValueError(
                f"run {parent!r} ran with surrogate {source.surrogate!r}, and a "
                "resumed segment keeps its lineage's surrogate (kd refuses the "
                "resume as a changed run otherwise). Omit --surrogate to "
                f"inherit it; to search with {surrogate!r}, drop --parent and "
                "record the reason with --decision-parent"
            )
    switched, narrowed = (
        (False, False)
        if parent is not None
        else move_facts(
            ledger,
            run_id=run_id,
            decision_parent=decision_parent,
            instrument=manifest["instrument"],
            dataset=dataset,
            sketch=sketch,
        )
    )
    entry = ledger.record(
        run_id=run_id,
        run_dir=os.path.relpath(path, workspace),
        parent_run_id=parent,





        resume_from=None,
        instrument=manifest["instrument"],
        dataset=dataset,
        seed=seed,
        status=status,
        nmse=nmse,
        created_at=manifest["created_at"],
        summary=summary,
        reseed=reseed,
        decision_parent=parent if decision_parent is None else decision_parent,


        action=segment_action(
            resumed=parent is not None,
            reseed=reseed,
            resume_iteration=resume_iteration,
            switched=switched,
            narrowed=narrowed,
        ),
        params=params,
        law=law,
        report_projection=report_projection(report),
        sketch=sketch,
        surrogate=surrogate,
        platform_nmse=platform_nmse,
        platform_frame=platform_frame,
        platform_coefficients=platform_coefficients,
    )



    payload: dict[str, Any] = {"run_id": entry.run_id, "segment_report": report}
    if platform_eval:




        payload.update(
            {
                "platform_nmse": platform_nmse,
                "platform_frame": platform_frame,
                "platform_coefficients": platform_coefficients,
                "platform_eval_error": None if evaluation is None else evaluation.error,
                "platform_eval_dropped": (
                    None if evaluation is None else evaluation.dropped
                ),
            }
        )
    return payload


def _preview(workspace: Path, *, dataset: str) -> dict[str, Any]:
    report = kd.preview_report(
        load_dataset(dataset, resolve_input(workspace, dataset))
    ).to_dict()
    payload = {
        "format": FEATURES_FORMAT,
        "dataset": dataset,
        "features": priors.features_from_report(report),
        "report": report,
    }
    path = workspace / FEATURES_DIR / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _surrogate_list(registry: SurrogateRegistry) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    named: set[Path] = set()
    for surrogate_id in registry.ids():
        entry = registry.entry(surrogate_id)
        path = registry.path(surrogate_id)
        named.add(path)
        present = path.is_file()
        rows.append(
            {
                **entry.to_dict(),
                "file_present": present,
                "sha_ok": present and file_sha256(path) == entry.file_sha256,
            }
        )
    orphans = sorted(
        f"{REGISTRY_DIRNAME}/{path.name}"
        for path in registry.directory.glob("*.pt")
        if path not in named
    )
    return {"surrogates": rows, "orphans": orphans}


def _surrogate_train(
    registry: SurrogateRegistry,
    *,
    dataset: str,
    algorithm: str,
    params: dict[str, Any] | None,
    seed: int,
    max_seconds: int | None,
) -> dict[str, Any]:
    schema = _registered_schema(algorithm)
    key = surrogate_training.artifact_key(schema)
    requested = surrogate_training.surrogate_params(schema, params)
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError(
            f"--max-seconds has to be a positive number of seconds, got "
            f"{max_seconds}; omit it to leave the training uncapped"
        )
    recipe = {
        "dataset_id": dataset,
        "algorithm": algorithm,
        "seed": seed,
        "params": requested,
    }
    hit, retrained_reason = surrogate_training.reuse_hit(registry, recipe)
    if hit is not None:
        return _surrogate_answer(hit, reused=True, retrained_reason=None)
    training = surrogate_training.run_training(
        registry=registry,
        schema=schema,
        recipe=recipe,
        key=key,
        time_cap_seconds=None if max_seconds is None else float(max_seconds),
    )
    if training.error_type is not None:



        raise RuntimeError(f"{training.error_type}: {training.error_message}")
    entry = surrogate_training.register_training(
        registry=registry, recipe=recipe, key=key, training=training
    )
    return _surrogate_answer(entry, reused=False, retrained_reason=retrained_reason)


def _surrogate_answer(
    entry: SurrogateEntry, *, reused: bool, retrained_reason: str | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "surrogate_id": entry.surrogate_id,
        "reused": reused,
        "key": entry.key,
        "file": entry.file,
        "file_sha256": entry.file_sha256,
        "train_seconds": entry.train_seconds,
        "epochs": entry.epochs,
        "final_loss": entry.final_loss,
    }
    if retrained_reason is not None:
        payload["retrained_reason"] = retrained_reason
    return payload


def _status(
    ledger: SegmentLedger,
    workspace: Path,
    *,
    dataset: str | None,
    remaining_budget: int | None,
    prior_path: Path | None,
) -> dict[str, Any]:
    segments = ledger.segments()
    datasets = {entry.dataset for entry in segments if entry.dataset}
    if dataset is None:
        if len(datasets) > 1:
            raise ValueError(
                f"this tree holds several datasets {sorted(datasets)}; pass "
                "--dataset to say which one the default selection is for"
            )
        dataset = next(iter(datasets), None)
    leaves = [
        entry
        for entry in segments
        if not ledger.is_pruned(entry.run_id)
        and not any(
            not ledger.is_pruned(child.run_id)
            for child in ledger.children(entry.run_id)
        )
    ]
    prior = None if prior_path is None else priors.load_prior(prior_path)
    excluded = None if prior is None else prior.get("excluded")
    return {
        "tree": ledger.to_dict(),
        "leaves": [
            _leaf(
                ledger,
                workspace,
                entry,
                remaining_budget=remaining_budget,
                prior=prior,
            )
            for entry in leaves
        ],
        "default_selection": criteria.default_selection(ledger, dataset_id=dataset),
        **(
            {}
            if prior is None
            else {
                "prior_document": {
                    "type": "eval" if excluded else "usage",
                    "excluded": excluded or None,
                }
            }
        ),
    }


def _leaf(
    ledger: SegmentLedger,
    workspace: Path,
    entry: SegmentEntry,
    *,
    remaining_budget: int | None,
    prior: dict[str, Any] | None,
) -> dict[str, Any]:
    schema = schema_of(entry.instrument)
    targets = [
        {
            "instrument": item["algorithm"],
            "cost_class": item["cost_class"],
            "modes": [
                {"name": mode["name"], "topologies": list(mode["topologies"])}
                for mode in item["modes"]
            ],
        }
        for item in kd.instrument_schemas()
        if item["algorithm"] != entry.instrument
    ]
    annotation, prior_reason = _annotation(
        workspace,
        entry,
        prior,
        [entry.instrument, *(target["instrument"] for target in targets)],
    )
    return {
        "run_id": entry.run_id,
        "progress": criteria.progress(entry),
        "promise": criteria.promise(
            entry, ledger=ledger, remaining_budget=remaining_budget
        ),
        "prior": None if annotation is None else annotation[entry.instrument],
        **({} if prior_reason is None else {"prior_reason": prior_reason}),
        "legal_moves": (
            None
            if schema is None
            else criteria.legal_moves(
                entry,
                _projected(schema),
                ledger=ledger,
                available_iterations=_iterations(ledger, workspace, entry.run_id),
                alternatives=[target["instrument"] for target in targets],
            )
        ),
        "switch_targets": [
            {
                **target,
                "prior": (
                    None if annotation is None else annotation[target["instrument"]]
                ),
            }
            for target in targets
        ],
    }


def _annotation(
    workspace: Path,
    entry: SegmentEntry,
    prior: dict[str, Any] | None,
    instruments: list[str],
) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    if prior is None or not entry.dataset:
        return None, None
    try:



        priors.assert_serves(prior, entry.dataset)
    except ValueError:
        return None, "not_served"
    path = workspace / FEATURES_DIR / f"{entry.dataset}.json"
    if not path.is_file():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["format"] != FEATURES_FORMAT:
        raise ValueError(
            f"{path} declares format {payload['format']!r}; `tree preview` "
            f"writes {FEATURES_FORMAT!r}"
        )
    if payload["dataset"] != entry.dataset:
        raise ValueError(
            f"{path} was previewed for dataset {payload['dataset']!r} but sits "
            f"under the name {entry.dataset!r}: a copied features file would "
            "annotate the wrong data"
        )
    return priors.annotate(prior, payload["features"], instruments), None


def _decide(
    ledger: SegmentLedger,
    workspace: Path,
    *,
    run_id: str,
    remaining_budget: int,
    instrument: str | None,
    perturbations: list[dict[str, Any]],
) -> dict[str, Any]:
    node = ledger.entry(run_id)
    if instrument is not None and instrument != node.instrument:
        raise ValueError(
            f"run {run_id!r} was produced by {node.instrument!r} and "
            f"--instrument says {instrument!r}; the move is read against the "
            "declaration this node ran under. Drop --instrument to use the "
            f"node's own, or ask about a node that ran {instrument!r}"
        )
    return criteria.recommend(
        ledger,
        node,
        _schema(node.instrument),
        dataset_id=node.dataset or None,
        remaining_budget=remaining_budget,
        perturbations=perturbations,
        available_iterations_of=lambda target: _iterations(ledger, workspace, target),
    )


def _iterations(ledger: SegmentLedger, workspace: Path, run_id: str) -> tuple[int, ...]:
    return eligible_iterations(_run_path(workspace, ledger.entry(run_id).run_dir))


def _schema(instrument: str) -> dict[str, Any]:
    return _projected(_registered_schema(instrument))


def _registered_schema(instrument: str) -> dict[str, Any]:
    schema = schema_of(instrument)
    if schema is None:
        raise ValueError(
            f"kd registers no instrument {instrument!r}; the registered ones "
            "are printed by kd.instrument_schemas()"
        )
    return schema


def _projected(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "segmentation": schema["segmentation"],
        "knobs": [
            {"name": knob["name"], "resume_tier": knob["resume_tier"]}
            for knob in schema["knobs"]
        ],
        "score_direction": schema["score_direction"],
        "modes": [
            {"name": mode["name"], "sketch": dict(mode["sketch"])}
            for mode in schema["modes"]
        ],
    }


def _run_path(workspace: Path, run_dir: str) -> Path:
    path = Path(run_dir)
    return path if path.is_absolute() else workspace / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kdagent.tree",
        description="Read and write the policy tree of one workspace.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(),
        help="the workspace holding tree.jsonl and runs/ (default: this directory)",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    status = subcommands.add_parser(
        "status", help="print the tree, the live leaves, and the default answer"
    )
    status.add_argument(
        "--dataset",
        default=None,
        help="which dataset the default selection is for; needed only when the "
        "tree holds more than one",
    )
    status.add_argument(
        "--remaining-budget",
        type=int,
        default=None,
        help="what is left of the budget, in the instrument's own unit; "
        "reported back in each leaf's promise (null when omitted)",
    )
    status.add_argument(
        "--prior",
        type=Path,
        default=None,
        help="a kd-prior-v1 document to annotate each leaf and its switch "
        "targets with; there is no default, and without a features file from "
        "`preview` the annotation stays null",
    )

    preview = subcommands.add_parser(
        "preview",
        help="load one dataset read-only and file its answer-blind features",
    )
    preview.add_argument(
        "--dataset",
        required=True,
        help="the catalog id to preview; the report and its feature projection "
        "are written to <workspace>/features/<id>.json and printed",
    )

    surrogate = subcommands.add_parser(
        "surrogate", help="the workspace's trained derivative surrogates"
    )



    surrogate_commands = surrogate.add_subparsers(
        dest="surrogate_command", required=True
    )
    surrogate_commands.add_parser(
        "list",
        help="every registered surrogate, verified against the files on disk, "
        "plus the .pt files no row names",
    )
    train = surrogate_commands.add_parser(
        "train",
        help="train this recipe's surrogate once and file it in the registry",
    )
    train.add_argument("--dataset", required=True, help="the kd catalog id to train on")
    train.add_argument(
        "--algorithm",
        required=True,
        help="the instrument the surrogate is for; only one whose schema "
        "declares a config artifact key takes one",
    )
    train.add_argument(
        "--params",
        default=None,
        help="the instrument's surrogate parameters, as JSON; a parameter that "
        "does not shape the network is refused by name",
    )
    train.add_argument(
        "--seed",
        type=int,
        default=surrogate_training.DEFAULT_SEED,
        help="the training seed, part of the recipe: two seeds are two "
        f"surrogates (default: {surrogate_training.DEFAULT_SEED})",
    )
    train.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="wall-clock cap for this training. A training has no iteration "
        "boundary to stop at: at the cap the child is killed and nothing at "
        "all is kept, so give it what the epoch count needs. Omitted, the "
        "training is uncapped -- this command runs under no budget of its own",
    )

    record = subcommands.add_parser(
        "record", help="file a sealed run directory on the tree"
    )
    record.add_argument("--run-dir", required=True)
    record.add_argument(
        "--parent", default=None, help="the run this segment resumed from"
    )
    record.add_argument(
        "--decision-parent",
        default=None,
        help="the run whose result made you run this one (default: --parent)",
    )
    record.add_argument(
        "--reseed",
        action="store_true",
        help="this segment is a fork of its parent, not a continuation",
    )
    record.add_argument(
        "--resume-iteration",
        type=int,
        default=None,
        help="the checkpoint iteration this segment restored, when it was not "
        "the latest; it is what makes the recorded action a rollback",
    )
    record.add_argument(
        "--params", default=None, help="the parameters this segment ran, as JSON"
    )
    record.add_argument(
        "--sketch",
        default=None,
        help="the kd-sketch-v1 payload this segment searched under, as JSON; "
        "with --decision-parent on the same instrument and dataset the "
        "recorded action is a narrow; with --parent, omitted means the "
        "parent's and a different one exits 1",
    )
    record.add_argument(
        "--surrogate",
        default=None,
        help="the registry id (surrogates/registry.jsonl) of the surrogate this "
        "segment injected; an id the registry does not hold exits 1; with "
        "--parent, omitted means the parent's and a different one exits 1",
    )
    record.add_argument(
        "--platform-eval",
        action="store_true",
        help="load the dataset and add a neutral full-grid score (and its re-fit "
        "coefficients, measured against the LHS the sealed record names) to "
        "this row; the outcome, including a failure, is printed",
    )

    for name, help_text in (
        ("prune", "abandon this segment and everything below it"),
        ("select", "make this segment the run's answer"),
    ):
        event = subcommands.add_parser(name, help=help_text)
        event.add_argument("run_id")
        event.add_argument("--reason", required=True)

    decide = subcommands.add_parser(
        "decide", help="print the move the criteria recommend from this node"
    )
    decide.add_argument("run_id")
    decide.add_argument("--remaining-budget", type=int, required=True)
    decide.add_argument(
        "--instrument",
        default=None,
        help="which instrument's declaration to read the move against "
        "(default: the one on this node's own ledger row); the row is resolved "
        "from kd.instrument_schemas() here, not passed in",
    )
    decide.add_argument(
        "--perturbations", required=True, help="the perturbation table, as JSON"
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
