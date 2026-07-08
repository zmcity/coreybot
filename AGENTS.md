# AGENTS.md -- coreybot project memory

This is a from-scratch, minimal-dependency **async LLM agent framework** used as a
learning project. Keep changes surgical and consistent with the existing style.

## Environment (must follow)
- Python venv only: run everything via `.\.venv\Scripts\python.exe` (official
  Python 3.14, Textual 8.x). Do **not** use system/Anaconda Python.
- Network is restricted and approvals are off: you **cannot** `pip install`. Work
  with what is already installed. There is no lint tool configured.
- Tests: `.\.venv\Scripts\python.exe -m pytest -q` (config in `pytest.ini`,
  asyncio auto mode, marker `integration`). The suite takes ~10-13s; give pytest a
  generous timeout (60000ms+) when a runner default would cut it off.
- File writes: keep files **UTF-8 without BOM** (the repo's `.py`/`test_*.py` have
  no BOM). `README.md` is Chinese and also has no BOM. When editing from a shell,
  author any non-ASCII via Python string escapes (PowerShell here-strings corrupt
  CJK/box-drawing and treat backticks as escapes).

## Language / docs
- Chinese-speaking user: chat replies and `README.md`/docs are **Chinese**; code
  identifiers and docstrings stay **English**.

## Layout
```
coreybot/
  core/            # config, message/Role, protocol types, paths (Codex-style home)
  llm/             # provider compatibility layer (openai/gemini/anthropic-compatible)
  runtime/agent.py # Agent: async turn loop + append-only telemetry (source of truth)
  runtime/session*.py # SessionTree + JSONL rollout store + open_session glue
  tools/           # tool-call system; tools/builtin/<tool>/ packages via @tool
  frontends/
    chat_loop.py   # headless REPL loop
    tui/           # Textual TUI (app.py + flow.py)  -- see tui/AGENTS.md
```

## Session persistence (Codex-style home)
State lives under a single home dir mirroring `~/.codex`: default
`~/.coreybot`, overridable by `--home` > `COREYBOT_HOME` env > default (built
at runtime from the static dirname `.coreybot`). `core/paths.py` holds pure
path values (`AgentPaths`), `runtime/session_store.py` does lossless SessionTree
<-> JSONL rollout (never replays commit -- rebuilds ids/heads/roots/counter),
`runtime/session_service.py` resolves the home, does the first-run y/N confirm,
and hands the Agent a `session_saver`. `__main__.py` calls `open_session` once.
Keep the tree free of any I/O; put serialization in the store.

## UI is a stabilized contract
The Textual TUI is considered **done and stable**. Its look/behavior is pinned by
tests and by `coreybot/frontends/tui/AGENTS.md`. Before changing anything under
`coreybot/frontends/tui/`, read that file and keep the guard tests green. Do not
casually restyle, rename widget ids, re-add a Header/Footer, or change the status
bar / flow-chart behavior without updating that contract on purpose.
