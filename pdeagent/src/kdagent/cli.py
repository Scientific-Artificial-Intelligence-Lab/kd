
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
    LLM4ED_ENDPOINTS,
    LLM4ED_KEY,
    LLM4ED_LIMITS,
    EndpointConfig,
    Llm4edConfig,
    MissingConfigError,
    check_llm4ed_shell,
    config_path,
    load_config,
    resolve_config,
    resolve_llm4ed,
    save_config,
    save_llm4ed,
)
from kdagent.data_source import CONTROLLER_KEYS, is_file_ref


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kd-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="configure a model endpoint and API key")
    setup.add_argument(
        "--llm4ed",
        action="store_true",
        help="configure, change or switch off the separate LLM4ED endpoint "
        "and key instead (the controller's settings stay as they are)",
    )
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
    run.add_argument(
        "--action-memory",
        type=Path,
        help="state-conditioned action experience; file input only, outside the workspace",
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



        command.add_argument(
            "--bare",
            action="store_true",
            help="one-sentence prompt, no skills, no documents",
        )
        command.add_argument(
            "--shell",
            action="store_true",
            help="add the execute tool, an unsandboxed shell that can read "
            "any file this user can; without it the model has the file tools "
            "(rooted at the workspace) and the verbs",
        )


        llm4ed = command.add_mutually_exclusive_group()
        llm4ed.add_argument(
            "--llm4ed",
            dest="llm4ed",
            action="store_true",
            default=None,
            help="let searches use LLM4ED through the endpoint and key saved by "
            "kd-agent setup --llm4ed (no key option exists); not with --shell",
        )
        llm4ed.add_argument(
            "--no-llm4ed",
            dest="llm4ed",
            action="store_false",
            help="LLM4ED off for this run, whatever the saved configuration says",
        )




        command.add_argument(
            "--device",
            choices=("cpu", "cuda"),
            default="cpu",
            help="where every search and surrogate training of the run executes",
        )





        command.add_argument(
            "--training-dtype",
            choices=("float32", "float64"),
            default=None,
            help="precision every field network the run trains is trained in; "
            "omitted, each network trains in its data's dtype",
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
            help="task kind of a --data table (ode for a table whose target is a "
            "time derivative); a raw trajectory file is ode",
        )
        command.add_argument(
            "--target-state",
            help="the state of a --data trajectory file whose time derivative is "
            "the target; needed when the file has several states",
        )
        source = command.add_mutually_exclusive_group()
        source.add_argument(
            "--dataset", help="pin a catalog ID or a saved input reference"
        )
        source.add_argument("--data", type=Path, help="load a user data file")
        command.add_argument(
            "--load-options",
            type=_load_options,
            help="JSON object of kd.load options: kind, coords, fields, field_axes, "
            "select, layout, lhs, periodic, trajectory_id, params",
        )
        command.add_argument(
            "--workspace", type=Path, help="reuse an existing run ledger"
        )
        command.add_argument(
            "--sketch",
            type=_sketch_file,
            help="a Sketch file written by kd.save_sketch (kd-sketch-spec-v1 or "
            "kd-sketch-v2; a released kd-sketch-v1 file is read too); it is bound "
            "to the run's input and every search of the run runs under it "
            "(needs --dataset or --data)",
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
    for key, flag in CONTROLLER_KEYS.items():
        if key in options:
            raise argparse.ArgumentTypeError(
                f"{key} must be set with {flag}, not --load-options"
            )
    return options


def _sketch_file(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--sketch must hold a JSON object")
    return payload


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


LLM4ED_NOTICE = (
    "LLM4ED makes additional requests using a separately configured "
    "OpenAI-compatible endpoint and key. Charges or quota usage belong to that "
    "key's account. A 12-round search may make about 40 calls; this is an "
    "estimate, not a limit. Each call is retried up to 5 times. LLM4ED is off "
    "by default and cannot run together with --shell.\n"
)


def _setup_llm4ed() -> None:
    path = config_path()
    saved = load_config(path)
    if saved is None:
        raise MissingConfigError(
            f"no saved configuration at {path}; run kd-agent setup"
        )
    block = dict(saved.get(LLM4ED_KEY, {}))
    sys.stdout.write(LLM4ED_NOTICE)
    answer = input("Enable LLM4ED for this user's runs? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        save_llm4ed({**block, "enabled": False}, path)
        sys.stdout.write(f"LLM4ED switched off in {path}\n")
        return
    endpoint = input(f"LLM4ED endpoint ({', '.join(LLM4ED_ENDPOINTS)}): ").strip()
    if endpoint not in LLM4ED_ENDPOINTS:
        raise ValueError(
            f"llm4ed endpoint must be one of: {', '.join(LLM4ED_ENDPOINTS)}"
        )
    base_url = input("LLM4ED base URL (leave empty for the preset): ").strip()
    block = {
        "enabled": True,
        "endpoint": endpoint,
        "base_url": base_url or LLM4ED_ENDPOINTS[endpoint],
        "model": input("LLM4ED model at that URL: ").strip(),
        "api_key": getpass.getpass("LLM4ED API key (not the controller's): "),
    }
    for name, default in LLM4ED_LIMITS.items():
        value = input(f"LLM4ED {name} (default {default}): ").strip()
        if value:
            block[name] = int(value)
    resolve_llm4ed({LLM4ED_KEY: block}, enabled=True)
    save_llm4ed(block, path)
    sys.stdout.write(f"LLM4ED configuration saved to {path}\n")


def _llm4ed_configuration(args: argparse.Namespace) -> Llm4edConfig | None:
    saved = None if args.no_config else load_config(config_path())
    llm4ed = resolve_llm4ed(
        saved, enabled=args.llm4ed, require_key=args.command != "fingerprint"
    )
    check_llm4ed_shell(llm4ed, shell=args.shell)
    return llm4ed


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
        "answer_input",
        "answer_nudges",
        "truncation_nudges",
        "endpoint",
        "model",
        "device",
        *(["training_dtype"] if "training_dtype" in result else []),
        "report_path",
        "prompt_sha256",
        "system_prompt_sha256",
        "task_kind",
        "target",
    ]
    if "answer_selection" in result:
        keys += ["answer_selection"]
    if "answer_system" in result:
        keys += ["answer_system"]
    if "answer_receipt" in result:
        keys += ["answer_receipt"]
    if "reason" in result:
        keys += ["reason", "evidence_run_ids"]
    if "prior_path" in result:
        keys += ["prior_path", "prior_type", "prior_sha256"]
    if "memory_path" in result:
        keys += ["memory_path", "memory_sha256"]
    if "action_memory_path" in result:
        keys += ["action_memory_path", "action_memory_sha256"]
    if "advice_path" in result:
        keys += ["advice_path", "advice_sha256"]
    if "sketch_sha256" in result:
        keys += ["sketch_sha256"]
    if "llm4ed" in result:
        keys += ["llm4ed"]
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


def _interrupted(
    config: EndpointConfig,
    device: str = "cpu",
    training_dtype: str | None = None,
    llm4ed: Llm4edConfig | None = None,
) -> dict[str, Any]:
    return {
        "stop_reason": "interrupted",
        "answer": None,
        "answer_source": None,
        "answer_provenance": None,
        "answer_input": None,
        "answer_nudges": None,
        "truncation_nudges": None,
        "endpoint": config.endpoint,
        "model": config.model,
        "device": device,
        **({} if training_dtype is None else {"training_dtype": training_dtype}),
        "report_path": None,
        "prompt_sha256": None,
        "system_prompt_sha256": None,
        "task_kind": None,
        "target": None,
        **({} if llm4ed is None else {"llm4ed": llm4ed.public()}),
    }


def _stop_on_sigterm(
    result_path: Path | None,
    config: EndpointConfig,
    device: str,
    training_dtype: str | None,
    llm4ed: Llm4edConfig | None,
) -> Any:
    from kdagent import episode_worker

    def stop(signum: int, frame: Any) -> None:


        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        episode_worker.stop_live_workers(episode_worker.KILL_GRACE_SECONDS)
        if result_path is not None:
            _write_result(
                result_path, _interrupted(config, device, training_dtype, llm4ed)
            )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(128 + signal.SIGTERM)

    return signal.signal(signal.SIGTERM, stop)


def _execute(
    args: argparse.Namespace, config: EndpointConfig, llm4ed: Llm4edConfig | None
) -> int:
    workspace = args.workspace
    if workspace is None:
        root = Path.cwd() / "kd-agent-runs"
        root.mkdir(exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=f"{args.command}-", dir=root))
    sys.stdout.write(f"workspace: {workspace}\n")
    sys.stdout.write(
        "LLM4ED: off\n"
        if llm4ed is None
        else f"LLM4ED: {json.dumps(llm4ed.public(), ensure_ascii=False)}\n"
    )
    options = dict(
        workspace=workspace,
        config=config,
        dataset_id=args.dataset,
        data=args.data,
        load_options=args.load_options,
        task_kind=args.task_kind,
        target_state=args.target_state,
        advice=args.advice,
        recursion_limit=args.recursion_limit,
        time_budget_seconds=(
            None if args.time_budget_minutes is None else args.time_budget_minutes * 60
        ),
        bare=args.bare,
        shell=args.shell,
        device=args.device,
        training_dtype=args.training_dtype,
        initial_sketch=args.sketch,
        llm4ed=llm4ed,
    )
    if args.command == "chat":
        from kdagent.chat import chat

        chat(**options)
        return 0
    from kdagent.run import run

    previous = _stop_on_sigterm(
        args.result, config, args.device, args.training_dtype, llm4ed
    )
    try:
        result = run(
            request=args.request,
            prior=args.prior,
            memory=args.memory,
            action_memory=args.action_memory,
            **options,
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
    if args.command in ("run", "chat") and args.data is None:
        for flag, value in (
            ("--task-kind", args.task_kind),
            ("--target-state", args.target_state),
        ):
            if value is not None:
                parser.error(f"{flag} requires --data")
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
            _setup_llm4ed() if args.llm4ed else _setup()
        except (ValueError, OSError) as exc:
            sys.stderr.write(f"kd-agent: {exc}\n")
            return 2
        return 0


    try:
        _require_extra()
        config = _configuration(args)
        os.environ.pop("KD_AGENT_API_KEY", None)
        llm4ed = _llm4ed_configuration(args)
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        sys.stderr.write(f"kd-agent: {exc}\n")
        return 2
    if args.command == "fingerprint":
        from kdagent.fingerprint import compute

        result = compute(config, bare=args.bare, shell=args.shell, llm4ed=llm4ed)
        sys.stdout.write(
            json.dumps(result, ensure_ascii=False) + "\n"
            if args.json
            else result["prompt_sha256"] + "\n"
        )
        return 0
    return _execute(args, config, llm4ed)


if __name__ == "__main__":
    raise SystemExit(main())
