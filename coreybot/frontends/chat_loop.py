"""A minimal but real chat loop, now backed by the tool-calling Agent.

The loop itself is thin: it reads input, delegates the turn to ``Agent`` (which
handles the model + tool-calling internally), and prints the result. Tool
activity is streamed to the console via the agent's event callback so you can
watch the agent think.

Special commands: ``/exit`` quits, ``/reset`` clears memory (keeping the system
prompt), ``/history`` dumps the conversation, ``/tools`` lists tools.

The final assistant reply is rendered as Markdown (via Rich) so inline code,
fenced code blocks and blockquotes are highlighted -- matching the TUI. On a
non-TTY (pipes, tests) or if Rich is unavailable, it degrades to plain text.
"""

from __future__ import annotations

import asyncio
import sys

from coreybot.runtime.agent import Agent, AgentEvent
from coreybot.core.cancel import CancelToken, CancelledError
from coreybot.core.config import Config
from coreybot.llm.providers import available_providers


def _render_reply(text: str) -> str:
    """Return ``text`` rendered from Markdown to a printable string.

    Uses Rich's ``Markdown`` renderer captured into a string so the surrounding
    loop can still ``print`` it (keeping I/O in one place). When stdout is a real
    terminal we force ANSI colors; otherwise we emit plain text. Any failure
    (Rich missing, capture error) falls back to the raw text unchanged.
    """
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError:
        return text

    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    try:
        console = Console(
            force_terminal=is_tty or None,
            color_system="truecolor" if is_tty else None,
            highlight=False,
        )
        with console.capture() as capture:
            console.print(Markdown(text, code_theme="ansi_dark"))
        return capture.get().rstrip("\n")
    except Exception:
        return text


def _print_banner(agent: Agent) -> None:
    config = agent.config
    print("=" * 60)
    print("coreybot chat loop (XML-tag protocol + tools)")
    print(f"  provider : {config.provider}  (available: {', '.join(available_providers())})")
    print(f"  model    : {config.model}")
    print(f"  base_url : {config.base_url}")
    print(f"  tools    : {', '.join(agent.registry.names()) or '(none)'}")
    print("  commands : /exit  /reset  /history  /tools")
    print("=" * 60)


def _print_event(event: AgentEvent) -> None:
    if event.kind == "tool_call":
        print(f"  \u2699 calling {event.name}({event.arguments})")
    elif event.kind == "tool_result":
        status = "ok" if event.ok else "error"
        print(f"  \u2190 {event.name} [{status}]: {event.output}")
    elif event.kind == "notice":
        print(f"  [notice] {event.text}")


async def _run_turn_cancellable(agent: Agent, config: Config, user_input: str) -> None:
    """Run one turn as a task and let Ctrl+C cancel it (like a context cancel).

    We install a SIGINT handler on the running loop that flips a
    :class:`CancelToken`. That unwinds the in-flight LLM/tool call and raises
    :class:`CancelledError`, which we surface as an "interrupted" line instead
    of killing the whole program.
    """
    import signal

    token = CancelToken()
    loop = asyncio.get_running_loop()
    installed = False
    try:
        # On Windows/Proactor this may be unsupported; we degrade gracefully
        # (Ctrl+C then just raises KeyboardInterrupt as before).
        loop.add_signal_handler(signal.SIGINT, token.cancel)
        installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        installed = False

    try:
        response = await agent.arun_turn(
            user_input, on_event=_print_event, cancel_token=token
        )
    except CancelledError:
        print("\n(interrupted)")
        if agent.history and agent.history[-1].role.value == "user":
            agent.history.pop()
        return
    except Exception as exc:  # transport error: keep the loop alive
        print(f"[error] {exc}")
        if agent.history and agent.history[-1].role.value == "user":
            agent.history.pop()
        return
    finally:
        if installed:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, ValueError):
                pass

    print(f"\n{config.model} >")
    print(_render_reply(response.content))


async def _arun_chat_loop(config: Config, session_saver=None) -> None:
    agent = Agent(config, session_saver=session_saver)
    _print_banner(agent)

    while True:
        try:
            # Read input off the event loop so the loop stays responsive.
            raw = await asyncio.to_thread(input, "\nyou > ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return
        user_input = raw.strip()

        if not user_input:
            continue
        if user_input == "/exit":
            print("bye!")
            return
        if user_input == "/reset":
            agent.reset()
            print("(memory cleared)")
            continue
        if user_input == "/tools":
            print(agent.registry.render_for_prompt() or "(no tools)")
            continue
        if user_input == "/history":
            for message in agent.history:
                print(f"  [{message.role.value}] {message.content}")
            continue

        await _run_turn_cancellable(agent, config, user_input)


def run_chat_loop(config: Config, session_saver=None) -> None:
    """Blocking entry point; runs the async loop to completion."""
    asyncio.run(_arun_chat_loop(config, session_saver=session_saver))


if __name__ == "__main__":
    run_chat_loop(Config.from_env())