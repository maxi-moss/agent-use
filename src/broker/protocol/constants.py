"""Wire-protocol constants. STDLIB-ONLY — the hook's only broker import.

Anything beyond the standard library added here lands on the synchronous
permission path of every tool call in every supervised session.
"""

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 1_048_576  # 1 MiB; observed real max 857 KiB (2026-07-27 survey)

# Session-socket message types (plan §2.1)
T_HOOK_EVENT = "hook_event"
T_PERMISSION_REQUEST = "permission_request"
T_DISPATCH_DECISION = "dispatch_decision"
T_SEND_PROMPT = "send_prompt"
T_STATUS = "status"
T_GET_DECISION_LOG = "get_decision_log"
T_SHUTDOWN = "shutdown"
T_RESPONSE = "response"

# Master-socket message types (plan §2.2)
T_ESCALATION = "escalation"
T_COMPLETION = "completion"
T_FATAL_ERROR = "fatal_error"
T_RETRACT = "retract"

# permission_request reply decisions (plan §2.1 — NOT Claude Code's enum;
# "escalated" deliberately avoids colliding with Claude Code's own "defer")
DECISION_ALLOW = "allow"
DECISION_ESCALATED = "escalated"

HOOK_WAIT_SECONDS = 30      # hook-internal hard timeout (spec §9.5)
HOOK_SETTINGS_TIMEOUT = 60  # settings.json "timeout" — deliberately 2x the above
