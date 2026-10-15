
from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import logging
import os
import signal
import sys
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from kdagent.config import (
    DEFAULT_RECURSION_LIMIT,
    ENDPOINTS,
    ENDPOINT_PRESETS,
    EndpointConfig,
    MissingConfigError,
    config_path,
    load_config,
    resolve_config,
    save_config,
)
from kdagent.data_source import is_file_ref


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kd-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="configure a model endpoint and API key")
    skill = commands.add_parser("skill", help="install the KD Python workflow skill")
    _skill_parser(skill)
    run = commands.add_parser("run", help="discover an equation in one run")
    chat = commands.add_parser("chat", help="discuss discovery interactively")
    fingerprint = commands.add_parser(
        "fingerprint", help="hash the model-visible prompt bundle offline"
    )
    fingerprint.add_argument(
        "--json", action="store_true", help="write the fingerprint document as JSON"
    )
    run.add_argument(
        "request", nargs="?", default="Find the equation governing this data."
    )
    run.add_argument(
        "--prior",
        type=Path,
        help="evaluation prior for --dataset; the document must be outside the workspace",
    )
    run.add_argument(
        "--memory",
        type=Path,
        help="experience document; file input only (--data, or --dataset naming a "
        "saved input), and the document must be outside the workspace",
    )
    run.add_argument(
        "--result",
        type=Path,
        help="also write the close-out lines as one JSON object to this file",
    )
    for command in (run, chat, fingerprint):
        command.add_argument("--endpoint", choices=ENDPOINTS)
        command.add_argument("--base-url", help="custom endpoint URL")
        command.add_argument("--model", help="model name at the endpoint")
        command.add_argument(
            "--no-config",
            action="store_true",
            help="ignore saved endpoint configuration",
        )

    fingerprint.set_defaults(effort=None)
    for command in (run, chat):
        command.add_argument(
            "--effort", help="reasoning effort passed to the model provider"
        )
        command.add_argument(
            "--advice",
            type=Path,
            help="advice document outside the workspace; file input only",
        )
        command.add_argument(
            "--task-kind",
            choices=("tabular-regression", "ode"),
            help="task kind for --data tables",
        )
        source = command.add_mutually_exclusive_group()
        source.add_argument(
            "--dataset", help="pin a catalog ID or a saved input reference"
        )
        source.add_argument("--data", type=Path, help="load a user data file")
        command.add_argument(
            "--load-options",
            type=_load_options,
            help="JSON object: coords, fields, field_axes, select, layout, lhs, periodic",
        )
        command.add_argument(
            "--workspace", type=Path, help="reuse an existing run ledger"
        )
        command.add_argument(
            "--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT
        )
        command.add_argument(
            "--time-budget-minutes",
            type=float,
            help="budget reported to the model; each search defaults to the remaining "
            "budget (minimum 60 seconds). This does not cap the whole conversation.",
        )
    return parser


def _skill_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="skill_command", required=True)
    install = commands.add_parser("install", help="copy the packaged KD workflow skill")
    install.add_argument("--to", type=Path, default=Path(".claude") / "skills")
    install.add_argument(
        "--force", action="store_true", help="overwrite different content"
    )


def _install_skill(directory: Path, *, force: bool) -> int:
    target = directory / "kd" / "SKILL.md"
    try:
        content = files("kdagent").joinpath("_assets/skills/kd/SKILL.md").read_bytes()
        if target.exists():
            if target.read_bytes() == content:
                sys.stdout.write(f"{target}\n")
                return 0
            if not force:
                raise FileExistsError(
                    f"{target} exists with different content; use --force to overwrite"
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    except OSError as exc:
        sys.stderr.write(f"kd-agent: {exc}\n")
        return 2
    sys.stdout.write(f"{target}\n")
    return 0


def _load_options(value: str) -> dict[str, Any]:
    try:
        options = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(options, dict):
        raise argparse.ArgumentTypeError("--load-options must be a JSON object")
    if "task_kind" in options:
        raise argparse.ArgumentTypeError(
            "task_kind must be set with --task-kind, not --load-options"
        )
    return options


def _setup() -> None:
    endpoint = input(f"Endpoint ({', '.join(ENDPOINTS)}): ").strip()
    values = {"endpoint": endpoint}
    for name, prompt in (("base_url", "Base URL"), ("model", "Model")):
        value = input(prompt + " (leave empty for a preset default): ").strip()
        if value or endpoint not in ENDPOINT_PRESETS:
            values[name] = value
    key = getpass.getpass("API key: ")
    if key or endpoint != "local":
        values["api_key"] = key
    config = resolve_config(values, environ={})
    path = config_path()
    save_config(config, path)
    sys.stdout.write(f"Configuration saved to {path}\n")


def _configuration(args: argparse.Namespace) -> EndpointConfig:
    saved = None if args.no_config else load_config(config_path())
    try:
        return resolve_config(
            saved,
            endpoint=args.endpoint,
            base_url=args.base_url,
            model=args.model,
            effort=args.effort,
            environ={**os.environ, "KD_AGENT_API_KEY": "fingerprint-placeholder"}
            if args.command == "fingerprint"
            else None,
        )
    except MissingConfigError:
        if (
            args.no_config
            or args.command == "fingerprint"
            or saved is not None
            or not sys.stdin.isatty()
        ):
            raise
    _setup()
    return resolve_config(
        load_config(config_path()),
        endpoint=args.endpoint,
        base_url=args.base_url,
        model=args.model,
        effort=args.effort,
    )


def _require_extra() -> None:



    if importlib.util.find_spec("deepagents") is None:
        raise ModuleNotFoundError(
            "Missing agent dependency: deepagents. "
            'Install with: uv pip install "sail-kd[agent]"'
        )


def _closeout(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "stop_reason",
        "answer",
        "answer_source",
        "answer_provenance",
        "answer_nudges",
        "truncation_nudges",
        "endpoint",
        "model",
        "report_path",
        "prompt_sha256",
        "system_prompt_sha256",
        "task_kind",
        "target",
    ]
    if "prior_path" in result:
        keys += ["prior_path", "prior_type", "prior_sha256"]
    if "memory_path" in result:
        keys += ["memory_path", "memory_sha256"]
    if "advice_path" in result:
        keys += ["advice_path", "advice_sha256"]
    return {key: result[key] for key in keys}


def _show_result(result: dict[str, Any]) -> None:
    from langchain_core.messages import AIMessage

    for message in result["messages"]:
        label = type(message).__name__
        if isinstance(message, AIMessage) and message.tool_calls:
            label += f" tool_calls={[call['name'] for call in message.tool_calls]}"
        sys.stdout.write(f"--- {label}\n{message.content}\n")
    for key, value in _closeout(result).items():
        sys.stdout.write(f"=== {key}: {value}\n")


def _write_result(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(_closeout(result), allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _interrupted(config: EndpointConfig) -> dict[str, Any]:
    return {
        "stop_reason": "interrupted",
        "answer": None,
        "answer_source": None,
        "answer_provenance": None,
        "answer_nudges": None,
        "truncation_nudges": None,
        "endpoint": config.endpoint,
        "model": config.model,
        "report_path": None,
        "prompt_sha256": None,
        "system_prompt_sha256": None,
        "task_kind": None,
        "target": None,
    }


def _stop_on_sigterm(result_path: Path | None, config: EndpointConfig) -> Any:
    from kdagent import episode_worker

    def stop(signum: int, frame: Any) -> None:


        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        episode_worker.stop_live_workers(episode_worker.KILL_GRACE_SECONDS)
        if result_path is not None:
            _write_result(result_path, _interrupted(config))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(128 + signal.SIGTERM)

    return signal.signal(signal.SIGTERM, stop)


def _execute(args: argparse.Namespace, config: EndpointConfig) -> int:
    workspace = args.workspace
    if workspace is None:
        root = Path.cwd() / "kd-agent-runs"
        root.mkdir(exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=f"{args.command}-", dir=root))
    sys.stdout.write(f"workspace: {workspace}\n")
    options = dict(
        workspace=workspace,
        config=config,
        dataset_id=args.dataset,
        data=args.data,
        load_options=args.load_options,
        advice=args.advice,
        recursion_limit=args.recursion_limit,
        time_budget_seconds=(
            None if args.time_budget_minutes is None else args.time_budget_minutes * 60
        ),
    )
    if args.command == "chat":
        from kdagent.chat import chat

        chat(**options)
        return 0
    from kdagent.run import run

    previous = _stop_on_sigterm(args.result, config)
    try:
        result = run(
            request=args.request, prior=args.prior, memory=args.memory, **options
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
    _show_result(result)
    if args.result is not None:
        _write_result(args.result, result)
    return 1 if result["stop_reason"].startswith("crashed:") else 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    parser = _parser()
    args = parser.parse_args(arguments if arguments else ["chat"])
    if args.command == "skill":
        return _install_skill(args.to, force=args.force)
    if (
        args.command in ("run", "chat")
        and args.load_options is not None
        and args.data is None
    ):
        parser.error("--load-options requires --data")
    if args.command in ("run", "chat") and args.task_kind is not None:
        if args.data is None:
            parser.error("--task-kind requires --data")
        args.load_options = {**(args.load_options or {}), "task_kind": args.task_kind}
    if (
        args.command in ("run", "chat")
        and args.dataset is not None
        and args.workspace is None
        and is_file_ref(args.dataset)
    ):
        parser.error(
            "--dataset names a saved input reference; pass the --workspace of the "
            "run that saved it, or --data <file> to load the file again"
        )
    logging.basicConfig(level=logging.INFO)
    if args.command == "setup":
        try:
            _setup()
        except (ValueError, OSError) as exc:
            sys.stderr.write(f"kd-agent: {exc}\n")
            return 2
        return 0


    try:
        _require_extra()
        config = _configuration(args)
        os.environ.pop("KD_AGENT_API_KEY", None)
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        sys.stderr.write(f"kd-agent: {exc}\n")
        return 2
    if args.command == "fingerprint":
        from kdagent.fingerprint import compute

        result = compute(config)
        sys.stdout.write(
            json.dumps(result, ensure_ascii=False) + "\n"
            if args.json
            else result["prompt_sha256"] + "\n"
        )
        return 0
    return _execute(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
