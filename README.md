# Broker

Routes decisions between one developer and N headless Claude Code sessions driven through Herdr.

A supervised Claude Code session runs in its own Herdr pane. When it stops, asks a question, or requests a tool permission, a hook wakes a per-session broker process. That broker triages the moment with a single LLM call and either handles it or escalates it to the developer through the master TUI.

The point is attention economics: the developer sees the decisions that actually need a human, and nothing else.

## The three processes

| Process | Entrypoint | Role |
| --- | --- | --- |
| **Master** | `python -m broker.master` | The only human-facing surface. TUI, session registry, routing LLM. One per developer. |
| **Session broker** | `python -m broker.session` | Headless supervisor of exactly one Claude Code session. Spawned by the master. One per session. |
| **Hook** | `python -m broker.hook` | Runs inside Claude Code on every hook event. Writes a decision to stdout and exits. |

They talk over asyncio unix domain sockets with newline-delimited JSON, one pydantic-validated envelope per line.

```
Claude Code ──hook──> session broker ──> master TUI
     ^                      │                 │
     └── herdr keystrokes ──┘<── decision ────┘
```

## Requirements

- Python **3.14+**
- [`uv`](https://docs.astral.sh/uv/)
- `claude` on `PATH` (Claude Code)
- `herdr` on `PATH` (developed against 0.7.5)
- `ANTHROPIC_API_KEY` set
- `HERDR_PANE_ID` set, or pass `--anchor` — the pane new sessions are split from

The master checks all of these at startup and exits non-zero with the reason on the first failure, so you do not need to verify them by hand.

## Install

```bash
uv sync
```

## Run

From inside a Herdr pane:

```bash
uv run python -m broker.master
```

Session brokers are never launched by hand — the master spawns them.

Configuration is optional. Defaults live in `BrokerConfig` (`src/broker/config.py`) and are overlaid with `$BROKER_HOME/config.json` if present; `$BROKER_HOME` defaults to `~/.broker` and is where sockets, the session registry, and logs are kept. `src/broker/paths.py` defines where each of those lives and is the only place that builds a path inside it.

Diagnostic logs are written to files, never to the terminal — the master's TUI owns that display. Follow a run with `tail -f "$BROKER_HOME"/logs/master.log`.

## Test mode

```bash
uv run python -m broker.master --test-mode
```

Test mode drives synthetic escalations through the **real** master — the real TUI, socket server, runtime handlers, and persisted queue. Everything above the socket is left out: no LLM, no hooks, no herdr, no Claude Code, no live sessions. So none of the startup requirements above apply — no `ANTHROPIC_API_KEY`, nothing on `PATH` — and it never writes `~/.claude/settings.json`.

Inside the TUI, `/inject <scenario>` runs one scenario from `tests/scenarios/*.json` and reports each step, ending in a `PASS`/`FAIL` summary. Run the master from the repo root, since `/inject` resolves scenario files relative to the working directory. `/inject` with an unknown or missing name prints the available scenarios.

The same scenarios run headless as part of the test suite:

```bash
uv run pytest tests/scenarios -v
```

## Source tree

```bash
src/broker/
  config.py      Shared configuration
  llm.py         Anthropic SDK wrapper
  prompts/       Static prompt prefixes
  protocol/      Wire schemas, socket server and client
  hook/          Claude Code hook client
  session/       Session broker: triage, watchdog, decision log
  master/        Master: runtime, registry, routing LLM, TUI
    testmode/    Synthetic escalation mode: --test-mode and /inject
  herdr/         Herdr CLI driver
  transcript/    Transcript reading: raw JSONL -> validated events
  claude/        Claude Code config: paths, settings, trust
tests/
  unit/          Fast, no I/O
  integration/   Real subprocess and IO boundaries
  scenarios/     Test-mode scenarios, run through the real master
```

Each package's `__init__.py` or primary module carries a docstring explaining what
it owns — start there.

## Development

All five gates must exit 0:

```bash
uv run pytest -v
uv run pyright                    # strict
uv run flake8 src tests           # rules in .flake8, not pyproject.toml
uv run lint-imports               # module boundary contracts
scripts/check_jsonl_literals.sh
```

Module boundaries are enforced, not merely documented: `lint-imports` checks the contracts declared under `[tool.importlinter]` in `pyproject.toml`. If an import you expect to be fine gets rejected, read the contract before working around it.
