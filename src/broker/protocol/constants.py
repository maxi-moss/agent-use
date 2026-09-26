"""Wire-protocol constants. STDLIB-ONLY — the hook's only broker import.

Anything beyond the standard library added here lands on the synchronous
permission path of every tool call in every supervised session.
"""

from enum import StrEnum

MAX_LINE_BYTES = 1_048_576  # 1 MiB; observed real max 857 KiB (2026-07-27 survey)

ENV_BROKER_SOCKET = "BROKER_SOCKET"
ENV_BROKER_HOOK_LOG = "BROKER_HOOK_LOG"

# Claude Code's own tool name — NOT one of our wire-protocol constants above.
ASK_USER_QUESTION = "AskUserQuestion"

# Session-socket message types
T_HOOK_EVENT = "hook_event"
T_PERMISSION_REQUEST = "permission_request"
T_DISPATCH_DECISION = "dispatch_decision"
T_SEND_PROMPT = "send_prompt"
T_STATUS = "status"
T_GET_DECISION_LOG = "get_decision_log"
T_GET_PERMISSION_LOG = "get_permission_log"
T_SHUTDOWN = "shutdown"
T_APPROVE_PROMPT = "approve_prompt"  # master -> broker
T_REACTIVATE = "reactivate"  # master -> broker
T_ASK_QUESTION = "ask_question"
# master -> broker: read-only question about a live escalation
T_CLARIFY_ESCALATION = "clarify_escalation"

# Master-socket message types
T_ESCALATION = "escalation"
# raiser -> master: a native prompt the developer answers in the pane
T_PANE_ESCALATION = "pane_escalation"
T_COMPLETION = "completion"
T_FATAL_ERROR = "fatal_error"
T_ESCALATION_RETRACT = "escalation_retract"  # broker -> master: resolved out of band
T_PANE_RETRACT = "pane_retract"  # raiser -> master: the native prompt closed
T_PROMPT_PROPOSAL = "prompt_proposal"  # broker -> master
T_BUDGET_UPDATE = "budget_update"  # broker -> master
T_DECISION_DELIVERED = "decision_delivered"  # broker -> master
T_DECISION_UNDELIVERED = "decision_undelivered"  # broker -> master
T_LIVE_STATUS = "live_status"  # broker -> master
T_SESSION_ENDED = "session_ended"  # broker -> master: SessionEnd fired, exiting
T_PROMPT_UNDELIVERED = "prompt_undelivered"  # broker -> master

# permission_request reply decisions — NOT Claude Code's enum;
# "escalated" deliberately avoids colliding with Claude Code's own "defer"
DECISION_ALLOW = "allow"
DECISION_ESCALATED = "escalated"

# ask_question reply decisions — "answer" carries updated_input for the hook
# to print; anything else (escalated / timeout / malformed) prints nothing.
ASK_DECISION_ANSWER = "answer"


class NackCode(StrEnum):
    """Machine-readable refusal reasons carried beside a NACK's error string.

    Closed set: a sender that cannot tell a routine refusal from a protocol
    violation has to treat both as bugs.
    """

    PROTOCOL_VIOLATION = "protocol_violation"
    MALFORMED = "malformed"
    UNKNOWN_SESSION = "unknown_session"
    STALE_PROPOSAL = "stale_proposal"
    WRONG_STATE = "wrong_state"


class PaneKind(StrEnum):
    """Which native prompt a pane escalation reports.

    Lives here because the view model may import only this module.
    """

    PERMISSION = "permission"
    QUESTION = "question"


class HookEventName(StrEnum):
    """The hook events the broker subscribes to."""

    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    STOP = "Stop"
    STOP_FAILURE = "StopFailure"
    NOTIFICATION = "Notification"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    PRE_COMPACT = "PreCompact"
    POST_COMPACT = "PostCompact"
    PERMISSION_REQUEST = "PermissionRequest"


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


# A session live enough to accept a decision or a prompt.
ACTIVE_STATES = frozenset({SessionState.DRIVING})

# States the broker is the sole writer of.
BROKER_OWNED_STATES = frozenset(
    {
        SessionState.GROUNDING,
        SessionState.AWAITING_APPROVAL,
        SessionState.DRIVING,
        SessionState.ESCALATED,
        SessionState.COMPLETED,
        SessionState.ERROR,
    }
)

# States the master is the sole writer of.
MASTER_OWNED_STATES = frozenset(
    {SessionState.SPAWNING, SessionState.STOPPED, SessionState.UNMANAGED}
)

# Lifecycle states a session has settled into: the ones with an outcome to view.
SETTLED_STATES = frozenset(
    {SessionState.COMPLETED, SessionState.ERROR, SessionState.STOPPED}
)

# States a late broker push must not overwrite: the session has moved on.
ABSORBING_STATES = frozenset({SessionState.STOPPED, SessionState.UNMANAGED})

# The hook's own deadline must always expire first, so it exits 0 on its own
# terms and degrades the session predictably. If Claude Code's settings.json
# timeout fired first it would kill the process mid-wait instead.
HOOK_WAIT_SECONDS = 30      # hook-internal hard timeout
HOOK_SETTINGS_TIMEOUT = 60  # settings.json "timeout" — deliberately 2x the above
