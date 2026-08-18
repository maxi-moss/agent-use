"""Wire-protocol constants. STDLIB-ONLY — the hook's only broker import.

Anything beyond the standard library added here lands on the synchronous
permission path of every tool call in every supervised session.
"""

from enum import StrEnum

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 1_048_576  # 1 MiB; observed real max 857 KiB (2026-07-27 survey)

# Session-socket message types
T_HOOK_EVENT = "hook_event"
T_PERMISSION_REQUEST = "permission_request"
T_DISPATCH_DECISION = "dispatch_decision"
T_SEND_PROMPT = "send_prompt"
T_STATUS = "status"
T_GET_DECISION_LOG = "get_decision_log"
T_GET_PERMISSION_LOG = "get_permission_log"
T_SHUTDOWN = "shutdown"
T_RESPONSE = "response"
T_APPROVE_PROMPT = "approve_prompt"  # master -> broker
T_REACTIVATE = "reactivate"  # master -> broker
T_ASK_QUESTION = "ask_question"

# Master-socket message types
T_ESCALATION = "escalation"
T_PERMISSION_ESCALATION = "permission_escalation"
T_COMPLETION = "completion"
T_FATAL_ERROR = "fatal_error"
T_RETRACT = "retract"
T_PROMPT_PROPOSAL = "prompt_proposal"  # broker -> master
T_BUDGET_UPDATE = "budget_update"  # broker -> master
T_LIVE_STATUS = "live_status"  # broker -> master

# Machine-readable refusal reasons carried alongside the human-readable error
# string on a negative Response. Closed set: a sender that cannot tell a
# capacity refusal from a protocol violation has to treat both as bugs.
NACK_PROTOCOL_VIOLATION = "protocol_violation"
NACK_SLOT_OCCUPIED = "slot_occupied"
NACK_MALFORMED = "malformed"
NACK_UNKNOWN_SESSION = "unknown_session"
NACK_STALE_PROPOSAL = "stale_proposal"
NACK_WRONG_STATE = "wrong_state"

# permission_request reply decisions — NOT Claude Code's enum;
# "escalated" deliberately avoids colliding with Claude Code's own "defer"
DECISION_ALLOW = "allow"
DECISION_ESCALATED = "escalated"

# ask_question reply decisions — "answer" carries updated_input for the hook
# to print; anything else (escalated / timeout / malformed) prints nothing.
ASK_DECISION_ANSWER = "answer"


class SessionState(StrEnum):
    """Every state a session may be in, on both the broker and master sides.

    Lives here because master and session may not import each other, and both
    sides set states.
    """

    SPAWNING = "spawning"
    GROUNDING = "grounding"
    AWAITING_APPROVAL = "awaiting_approval"
    DRIVING = "driving"
    ESCALATED = "escalated"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"
    UNMANAGED = "unmanaged"
    DEAD = "dead"


# A session live enough to accept a decision or a prompt.
ACTIVE_STATES = frozenset({SessionState.DRIVING})

# The hook's own deadline must always expire first, so it exits 0 on its own
# terms and degrades the session predictably. If Claude Code's settings.json
# timeout fired first it would kill the process mid-wait instead.
HOOK_WAIT_SECONDS = 30      # hook-internal hard timeout
HOOK_SETTINGS_TIMEOUT = 60  # settings.json "timeout" — deliberately 2x the above
