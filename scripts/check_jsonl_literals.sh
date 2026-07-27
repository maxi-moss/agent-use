#!/bin/sh
# Field-name containment gate (plan §3): raw JSONL key literals may appear ONLY
# in src/broker/transcript/raw.py. import-linter reasons about imports, not
# string literals — this gate covers the other half.
#
# The list is raw-only literals; keys that are also public schema fields
# (multiSelect, question, header, options) are deliberately NOT listed.
rg -n "parentUuid|isSidechain|toolUseResult|toolDenialKind|promptSource|sourceToolAssistantUUID|planFilePath|allowedPrompts" src/broker --glob '!src/broker/transcript/raw.py' && exit 1 || exit 0
