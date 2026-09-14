
from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import logging
import sys
import tempfile
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
    run = commands.add_parser("run", help="discover an equation in one run")
    chat = commands.add_parser("chat", help="discuss discovery interactively")
    run.add_argument(
        "request", nargs="?", default="Find the equation governing this data."
    )
    run.add_argument(
        "--prior",
        type=Path,
        help="evaluation prior for --dataset; the document must be outside the workspace",
    )
    for command in (run, chat):
        command.add_argument("--endpoint", choices=ENDPOINTS)
        command.add_argument("--base-url", help="custom endpoint URL")
        command.add_argument("--model", help="model name at the endpoint")
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


def _load_options(value: str) -> dict[str, Any]:
    try:
        options = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(options, dict):
        raise argparse.ArgumentTypeError("--load-options must be a JSON object")
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
    saved = load_config(config_path())
    try:
        return resolve_config(
            saved, endpoint=args.endpoint, base_url=args.base_url, model=args.model
        )
    except MissingConfigError:
        if saved is not None or not sys.stdin.isatty():
            raise
    _setup()
    return resolve_config(
        load_config(config_path()),
        endpoint=args.endpoint,
        base_url=args.base_url,
        model=args.model,
    )


def _require_extra() -> None:



    if importlib.util.find_spec("deepagents") is None:
        raise ModuleNotFoundError(
            "Missing agent dependency: deepagents. "
            'Install with: uv pip install "sail-kd[agent]"'
        )


def _show_result(result: dict[str, Any]) -> None:
    from langchain_core.messages import AIMessage

    for message in result["messages"]:
        label = type(message).__name__
        if isinstance(message, AIMessage) and message.tool_calls:
            label += f" tool_calls={[call['name'] for call in message.tool_calls]}"
        sys.stdout.write(f"--- {label}\n{message.content}\n")
    for key in (
        "stop_reason",
        "answer",
        "answer_source",
        "answer_provenance",
        "answer_nudges",
        "truncation_nudges",
        "endpoint",
        "model",
        "report_path",
    ):
        sys.stdout.write(f"=== {key}: {result[key]}\n")
    if "prior_path" in result:
        for key in ("prior_path", "prior_type", "prior_sha256"):
            sys.stdout.write(f"=== {key}: {result[key]}\n")


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

    result = run(request=args.request, prior=args.prior, **options)
    _show_result(result)
    return 1 if result["stop_reason"].startswith("crashed:") else 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    parser = _parser()
    args = parser.parse_args(arguments if arguments else ["chat"])
    if args.command != "setup" and args.load_options is not None and args.data is None:
        parser.error("--load-options requires --data")
    if (
        args.command != "setup"
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
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        sys.stderr.write(f"kd-agent: {exc}\n")
        return 2
    return _execute(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
