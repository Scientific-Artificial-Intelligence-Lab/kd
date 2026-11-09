
from __future__ import annotations

import argparse
import logging
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path

from kdui.serverlock import WorkspaceBusyError, hold_workspace
from kdui.settings import Settings, Workspace

logger = logging.getLogger("kdui")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kd-ui",
        description="Start KD's local web UI and print the URL to open.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("kd-ui-workspace"),
        help="directory for uploads, datasets, jobs and runs "
        "(default: ./kd-ui-workspace)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind (default: 127.0.0.1, this machine only)",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="port to bind (default: 8765)"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="startup token (default: a new random token on every start)",
    )
    parser.add_argument(
        "--fit-slots",
        type=int,
        default=1,
        help="fits that may run at the same time (default: 1)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")



    try:
        import uvicorn
    except ModuleNotFoundError as error:
        raise SystemExit(
            'kd-ui needs the web dependencies: pip install "sail-kd[ui]"'
        ) from error

    from kdui.app import create_app

    workspace = Workspace(args.workspace.resolve())
    workspace.create()
    try:
        lock = hold_workspace(workspace, host=args.host, port=args.port)
    except WorkspaceBusyError as error:
        raise SystemExit(str(error)) from error
    settings = Settings(
        workspace=workspace,
        token=args.token or secrets.token_urlsafe(32),
        host=args.host,
        port=args.port,
        fit_slots=args.fit_slots,
    )
    app = create_app(settings)
    logger.info("KD web UI, workspace %s", settings.workspace.root)
    logger.info("Open http://%s:%d/#token=%s", args.host, args.port, settings.token)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
    lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
