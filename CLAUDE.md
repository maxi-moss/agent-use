# CLAUDE.md - Broker

This file provides guidance to Claude Code when working with this repository.

## Project Overview

A master process routes decisions between one developer and N session brokers, each driving an interactive Claude Code session through Herdr. Each session broker owns its session's Herdr pane and performs every programmatic send to Claude.

## Stack

- Python 3.14 + `uv` for deps and locking — versions live in the root `pyproject.toml` and `uv.lock`
- pydantic — every socket and config boundary
- Textual — TUI
- asyncio unix domain sockets — all IPC
- `anthropic` SDK — inference; `openai` SDK — embeddings; `tree-sitter` — code parsing
- pytest, pyright (strict), import-linter
- Persistence: atomic write-and-rename JSON + append-only NDJSON; SQLite for the per-repo code index
- No ORM, no daemon manager

## Commands

The repo root is the uv project root; code lives under `src/broker/`. Run from the root.

```bash
source {project-root}/.venv/bin/activate  # Activate virtual environment
uv sync  # install
uv run pytest -v  # test
uv run pyright  # typecheck, strict (config in pyproject.toml)
uv run flake8 src tests  # lint (rules in root .flake8 — flake8 cannot read pyproject.toml)
uv run lint-imports  # module boundary contracts
scripts/check_jsonl_literals.sh  # JSONL key literals only in transcript/raw.py
```

## Sessions

Every session is an interactive Claude Code process in its own terminal pane, with a real TTY and the native UI. Headless `claude -p` is not supported and never will be — the native permission prompt only renders in a TTY, so `-p` cannot exercise the permission path at all.

"Headless" elsewhere in this repo refers to the *session broker* — a background asyncio process with no UI. That is a different thing from the Claude Code session it drives.

## When Writing Code

- Use `rg` instead of `grep` for plain-text matches (comments, strings, config). Ripgrep recurses by default and `-r` does `--replace` on the printed output, so run `rg -n "pattern"` instead.
- Comments should be used conservatively and only when absolutely necessary. Never comment on a previous state of the code or the change that produced it — a comment that only makes sense to someone who saw the diff is noise.
- Do not design code for backwards compatibility. When a change replaces how something works, migrate every call site and delete the old path in the same change — don't leave both live. Only keep the old path when the developer asks for it by name.
- Never import inside functions. Import at module top level.
- There are three inference stacks, duplicated on purpose — tool schemas, clients, timeouts: session and master (sharing `broker/llm.py`), permission, and index-embedding. Never factor them together, not even across providers; a shared helper is how one surface's model gets silently re-pinned to another's. Session and master share stack by design; nothing else joins it.

## When writing docs

- Never manually word-wrap markdown documents using line breaks.

## Subagent models

**Always pass an explicit model when spawning a subagent, and it must be haiku or sonnet. Never a higher tier.** The only exception is the user naming a model themselves. Never let a subagent inherit the session model by default — an unspecified model is a bug, not a neutral choice.

- **haiku** — exploration and other trivial read-only tasks
- **sonnet** — more advanced exploration, or tasks that require understanding before reporting back

## Docstrings

Google style. A one-line summary at a HIGH level — what the function does, not how.
Then `Args:` / `Returns:` / `Raises:` as applicable. Nothing else.

- Simple helpers get a one-line summary and no sections.
- No prose body narrating the implementation. If the reader wants the algorithm they will read the body.
- No references to planning documents.

## Tests

- **Write only tests that would still be needed after you delete them.** Ask what could silently break if this test vanished — if the answer is nothing because the assertion just restates the implementation, don't write it.
- No `conftest.py` — each test module carries its own harness. Don't factor fixtures into a shared file.
- A temp dir that will hold a unix socket goes under `/private/tmp`, never `tmp_path` — `AF_UNIX` paths cap at ~104 chars.
- `pgrep`/`pkill` on `broker.session` can match the developer's LIVE brokers — this machine runs real sessions out of `~/.broker`. Never kill a broker process your run didn't spawn; assert no-leak by diffing PID sets before/after, never by expecting empty.

## Definition of done

`uv run pyright`, `uv run pytest -v`, `uv run flake8 src tests`, `uv run lint-imports`, and `scripts/check_jsonl_literals.sh` all exit 0.

## Global rules

- **Fail loud.** Never proceed on a partial read, a missed write, or a decision that may be stale.
- **Under-escalation is the only unrecoverable failure.** Over-escalating wastes effort; deciding for the developer silently does not undo. Never tune the two symmetrically.
- **Never rewrite or re-summarise text written for the developer.** Wrap it with context, never over it.
- **Timestamps never enter LLM context.** They belong in the on-disk logs. A prompt must assemble byte-identically across calls or the cache breaks.
- **Shared external config has one writer — the master.** Anything shared with other tools, above all Claude Code's user-level config, is written only by the master, and only as read → validate → backup → temp-write → rename. No other process touches it.
- **Diagnostic output goes to a file, never the terminal.** The master's TUI owns the display and session brokers inherit it. No `print`, no stream handler — `logging_setup.configure` is the only wiring.
- **On Claude Code's hook path:** exit 0 always, stdout carries the decision and nothing else, stdlib imports only. A dead broker degrades a session to stock Claude Code; it never breaks one.
- **Every wait names its target and its timeout explicitly.**
- **Check Herdr and Claude Code behaviour against the installed binaries, never from memory.** Both move fast; the load-bearing mechanisms are version-sensitive.
- **Read the `[tool.importlinter]` contracts in `pyproject.toml` before any cross-module architectural decision.** A plan that assumes a forbidden import is wrong before it starts; `uv run lint-imports` is the check.