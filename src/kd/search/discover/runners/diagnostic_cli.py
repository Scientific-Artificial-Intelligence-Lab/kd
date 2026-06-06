
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from kd.search.discover.config import (
    DIAGNOSTIC_ENV_ENABLED_VALUE,
    DIAGNOSTIC_ENV_VAR,
    DIAGNOSTIC_GATE_RATIONALE,
)







DIAGNOSTIC_TOKEN_LIST_FIELDS: tuple[str, ...] = (
    "diagnostic_scaffold_diffusion_tokens",
    "diagnostic_scaffold_reaction_tokens",
    "diagnostic_scaffold_root_tokens",
    "diagnostic_scaffold_neutral_tokens",
)









_DEFAULT_NEUTRAL_TOKENS: tuple[str, ...] = ("u", "x", "y", "t")
_DEFAULT_ROOT_TOKENS: tuple[str, ...] = ("add", "sub")





_ENV_HINT: str = "Requires DISCOVER_ENABLE_DIAGNOSTICS=1."







_TOKEN_LIST_FLAG_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "--diagnostic-scaffold-diffusion-tokens",
        "+",
        "Token names allowed in the 'diffusion' subtree under "
        "--diagnostic-scaffold.",
    ),
    (
        "--diagnostic-scaffold-reaction-tokens",
        "+",
        "Token names allowed in the 'reaction' subtree under "
        "--diagnostic-scaffold.",
    ),
    (
        "--diagnostic-scaffold-root-tokens",
        "+",
        "Root token names allowed at step 0 under --diagnostic-scaffold."
        " Defaults to ('add', 'sub').",
    ),
    (
        "--diagnostic-scaffold-neutral-tokens",
        "*",
        "Neutral tokens that bypass left/right subtree forbids under "
        "--diagnostic-scaffold. Default ('u','x','y','t') when flag "
        "omitted; pass '--flag' with no args to restore strict-disjoint "
        "behavior.",
    ),
)


def add_diagnostic_scaffold_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--diagnostic-scaffold",
        action="store_true",
        default=False,
        help=(
            "(diagnostic — not production) P21 ScaffoldPrior: force "
            "cycle-0 expressions to have a shallow add/sub root with "
            "disjoint diffusion/reaction subtrees. Default OFF. "
            f"{_ENV_HINT}"
        ),
    )
    for flag_name, nargs, help_body in _TOKEN_LIST_FLAG_SPECS:
        parser.add_argument(
            flag_name,
            type=str,
            nargs=nargs,
            default=None,
            help=f"(diagnostic — not production) {help_body} {_ENV_HINT}",
        )


def enforce_diagnostic_env(args: argparse.Namespace) -> None:
    if not args.diagnostic_scaffold:
        return
    if os.environ.get(DIAGNOSTIC_ENV_VAR) == DIAGNOSTIC_ENV_ENABLED_VALUE:
        return
    msg = (
        f"--diagnostic-scaffold requires {DIAGNOSTIC_ENV_VAR}="
        f"{DIAGNOSTIC_ENV_ENABLED_VALUE} ({DIAGNOSTIC_GATE_RATIONALE})"
    )
    sys.stderr.write(f"error: {msg}\n")
    sys.exit(2)


def warn_orphan_token_lists(args: argparse.Namespace) -> None:
    if args.diagnostic_scaffold:
        return
    orphan: list[str] = []
    for attr in DIAGNOSTIC_TOKEN_LIST_FIELDS:
        val = getattr(args, attr)
        if val is None:
            continue
        flag_name = "--" + attr.replace("_", "-")
        orphan.append(flag_name)
    if orphan:
        sys.stderr.write(
            f"warning: {', '.join(orphan)} ignored because "
            "--diagnostic-scaffold not set\n"
        )


def build_scaffold_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if not args.diagnostic_scaffold:
        return {}
    neutral = (
        tuple(args.diagnostic_scaffold_neutral_tokens)
        if args.diagnostic_scaffold_neutral_tokens is not None
        else _DEFAULT_NEUTRAL_TOKENS
    )
    return {
        "diagnostic_scaffold": True,
        "diagnostic_scaffold_diffusion_tokens": tuple(
            args.diagnostic_scaffold_diffusion_tokens or ()
        ),
        "diagnostic_scaffold_reaction_tokens": tuple(
            args.diagnostic_scaffold_reaction_tokens or ()
        ),
        "diagnostic_scaffold_root_tokens": (
            tuple(args.diagnostic_scaffold_root_tokens)
            if args.diagnostic_scaffold_root_tokens is not None
            else _DEFAULT_ROOT_TOKENS
        ),
        "diagnostic_scaffold_neutral_tokens": neutral,
    }


__all__ = [
    "DIAGNOSTIC_TOKEN_LIST_FIELDS",
    "add_diagnostic_scaffold_args",
    "build_scaffold_kwargs",
    "enforce_diagnostic_env",
    "warn_orphan_token_lists",
]
