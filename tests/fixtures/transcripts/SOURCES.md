# Fixture provenance

Real fixtures are verbatim copies of transcripts produced by Claude Code 2.1.220
on this machine (harvested 2026-07-27); originals under `~/.claude/projects/`
were not modified. Synthetic fixtures were authored for edge cases the real
corpus does not contain (zero malformed lines observed in the 345-file survey).

## Real (Claude Code 2.1.220)

| File | Origin | Why |
|---|---|---|
| `8753fe50-2884-4cb6-9728-ba9c1101b617.jsonl` | `~/.claude/projects/-private-tmp-bspike2/` | 26-line minimal AskUserQuestion, single-select, answered; plus a rejected (is_error) answer |
| `71a46971-27ec-40fc-8379-ff5351eddf90.jsonl` | `~/.claude/projects/-Users-maxi-coding-knowledge-catalog/` | 3 AskUserQuestion variants: 2-question batch, multiSelect comma-join, multiSelect 2-question batch |
| `d4032982-9753-4cce-ac1a-589ee8fe7e19.jsonl` | `~/.claude/projects/-private-tmp-broker-spike/` | 33-line rejected AskUserQuestion (is_error path) |
| `4f98b564-4f25-4a62-bee0-7808a82cd868.jsonl` | `~/.claude/projects/-Users-maxi-coding-uniplay/` | Only real ExitPlanMode tool_use samples on this machine (3x, all REJECTED: `toolDenialKind: "user-rejected"`) |
| `3ec7ee94-4dd8-4e26-a8fa-6b047e3382e2.jsonl` | `~/.claude/projects/-Users-maxi-coding-uniplay/` | AskUserQuestion with `multiSelect` key ABSENT (default-handling) |

## Synthetic (authored 2026-07-27)

| File | Why |
|---|---|
| `malformed-line.jsonl` | One valid record, then a malformed JSON line — the fatal `TranscriptParseError` path. Zero malformed lines exist in the real corpus. |
| `unknown-type.jsonl` | Top-level `type: "future-thing"` among valid records — keep-known tolerance. |
| `zero-events.jsonl` | Non-empty file of only unrecognised types — must raise `TranscriptParseError`. |
| `empty-file.jsonl` | Zero bytes — yields zero events without raising. |
| `approved-exit-plan.jsonl` | **SYNTHETIC GUESS** — no approved ExitPlanMode result exists on this machine. The result content string is modelled, not observed. Task 10 (developer-driven capture) must replace this file with a real capture and update this table. |
