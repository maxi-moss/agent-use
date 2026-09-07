# Broker

Routes decisions between one developer and N interactive Claude Code sessions driven through Herdr.

A supervised Claude Code session runs in its own Herdr pane, with a real TTY and the native UI — fully usable by hand at any moment. When it stops, asks a question, or requests a tool permission, a hook wakes a per-session broker process. That broker triages the moment with a single LLM call and either handles it or escalates it to the developer through the master TUI.

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
- `OPENAI_API_KEY` set
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

The TUI's left pane is a live fleet dashboard: one block per session with its state, budget, current intent, a ⚠ when it is sitting on a native permission prompt, and what its broker is doing right now. The header line shows the master's own activity. It is display only — nothing on it enters any LLM context.

Configuration is optional. Defaults live in `BrokerConfig` (`src/broker/config.py`) and are overlaid with `$BROKER_HOME/config.json` if present; `$BROKER_HOME` defaults to `~/.broker` and is where sockets, the session registry, and logs are kept. `src/broker/paths.py` defines where each of those lives and is the only place that builds a path inside it.

Diagnostic logs are written to files, never to the terminal — the master's TUI owns that display. Follow a run with `tail -f "$BROKER_HOME"/logs/master.log`.

## Indexing a repository

```bash
uv run python -m broker.index /path/to/repo
```

Builds or refreshes the code index for that repository at `$BROKER_HOME/index/<sha256(path)>.sqlite`: every Python, TypeScript, TSX and JavaScript file is parsed with tree-sitter into symbols — functions, methods, classes and types (a `const` bound to a function counts as a function; a class also records its fields) — and edges (calls, inheritance, imports, type references, ownership). Re-running is incremental — only files whose content changed are re-parsed; edges are always re-resolved for the whole repository. The path must be the root of a git repository. One summary line is printed; details go to `$BROKER_HOME/logs/index.log`.

Every function, method and class is also embedded with OpenAI `text-embedding-3-small` (pinned in `EmbeddingConfig`, `src/broker/config.py`); only symbols whose text changed are re-embedded. Changing the embedding model does not re-embed existing vectors — delete the index file and run the command again.

A session cannot be spawned for a repository without an index: grounding retrieves the intent's code neighbourhood from it and aborts loudly if the index is missing or empty. The proposal shown in the TUI lists the retrieved symbols and their seed scores so you can judge the grounding before approving.

## Test mode

```bash
uv run python -m broker.master --test-mode
```

Test mode drives synthetic escalations through the **real** master — the real TUI, socket server, runtime handlers, and persisted queue. Everything above the socket is left out: no LLM, no hooks, no herdr, no Claude Code, no live sessions. So none of the startup requirements above apply — no API keys, nothing on `PATH` — and it never writes `~/.claude/settings.json`.

Inside the TUI, `/inject <scenario>` runs one scenario from `tests/scenarios/*.json` and reports each step, ending in a `PASS`/`FAIL` summary. Run the master from the repo root, since `/inject` resolves scenario files relative to the working directory. `/inject` with an unknown or missing name prints the available scenarios.

The same scenarios run headless as part of the test suite:

```bash
uv run pytest tests/scenarios -v
```

## Calibration

The two judgment prompts — triage (`prompts/triage.md`) and permission (`prompts/permission.md`) — are tuned against a small set of developer-owned cases in `calibration-cases/`, each carrying the answer-or-escalate (or allow-or-escalate) label the developer stands behind. `scripts/calibrate.py` feeds each case through the **real** triage and permission stacks against the pinned models and prints the model's decision and reasoning beside the recorded label, plus the aggregate escalation rate. It makes real API calls and is a tuning aid, never a CI gate — the only automated check is an offline schema guard on the case files.

`scripts/cache_probe.py` issues two identical triage calls and reports the prompt-cache token counts, so caching is measured rather than assumed.

## Source tree

```bash
src/broker/
  config.py      Shared configuration
  llm.py         Anthropic SDK wrapper
  prompts/       Static prompt prefixes
  protocol/      Wire schemas, socket server and client
  hook/          Claude Code hook client
  index/         Code index: tree-sitter symbols and edges per repo, SQLite
  session/       Session broker: triage, watchdog, decision log
  permission/    Permission triage: own model, client, prompt, log
  master/        Master: runtime, registry, routing LLM, TUI
    testmode/    Synthetic escalation mode: --test-mode and /inject
  herdr/         Herdr CLI driver
  transcript/    Transcript reading: raw JSONL -> validated events
  claude/        Claude Code config: paths, settings, trust
  calibration/   Calibration case schemas (offline prompt tuning)
calibration-cases/  Developer-owned triage/permission cases
tests/
  unit/          Fast, no I/O
  integration/   Real subprocess and IO boundaries
  scenarios/     Test-mode scenarios, run through the real master
  fixtures/indexer_inputs/  Sample repos the indexer tests parse — the TypeScript/JavaScript in there is test input, not project code
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
