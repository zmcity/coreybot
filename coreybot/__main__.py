"""Entry point: ``python -m coreybot``.

By default this launches the Textual TUI. Use ``--cli`` for the plain
line-based loop. Configuration comes from environment / ``.env`` and can be
overridden with flags, e.g.:

    python -m coreybot --provider anthropic --model claude-opus-4.8
    python -m coreybot --cli
"""

from __future__ import annotations

import argparse

from coreybot.frontends.chat_loop import run_chat_loop
from coreybot.core.config import Config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="coreybot")
    ui = parser.add_mutually_exclusive_group()
    ui.add_argument(
        "--tui",
        dest="tui",
        action="store_true",
        default=True,
        help="Launch the Textual TUI (default).",
    )
    ui.add_argument(
        "--cli",
        dest="tui",
        action="store_false",
        help="Use the plain line-based chat loop instead of the TUI.",
    )
    parser.add_argument("--provider", help="Override provider (openai/anthropic/gemini).")
    parser.add_argument("--model", help="Override model name.")
    parser.add_argument("--base-url", help="Override base URL.")
    parser.add_argument(
        "--home",
        help=(
            "Agent home directory for durable state (sessions, config). "
            "Overrides the COREYBOT_HOME env var and the default ~/.coreybot."
        ),
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Create the home directory without asking on first run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = Config.from_env()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url

    # Resolve the durable home (override > $COREYBOT_HOME > ~/.coreybot) and,
    # on first run, confirm creating it. A declined/failed setup still runs --
    # just without persistence -- so the agent is never blocked by disk state.
    session_saver = None
    sessions_dir = None
    current_file = None
    try:
        from coreybot.runtime.session_service import open_session

        handle = open_session(args.home, assume_yes=args.yes)
        session_saver = handle.saver
        if handle.saver is not None:
            sessions_dir = str(handle.paths.sessions_dir)
            current_file = str(handle.session_file) if handle.session_file else None
        else:
            print("(not persisting sessions -- home directory was not created)")
    except Exception as exc:  # never let setup crash the app
        print(f"[warn] session persistence disabled: {exc}")

    if args.tui:
        # Imported lazily so the CLI works even without Textual installed.
        from coreybot.frontends.tui import run_tui

        run_tui(
            config,
            session_saver=session_saver,
            sessions_dir=sessions_dir,
            current_file=current_file,
        )
    else:
        run_chat_loop(config, session_saver=session_saver)


if __name__ == "__main__":
    main()
