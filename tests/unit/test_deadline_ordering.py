"""Cross-process timeout ordering, read straight from each constant's owner.

Every assertion imports the constant from the module that defines it, one
import line each, so a later move of any of them fails loud here instead of
going stale.
"""

from broker.master.escalation_desk import CLARIFY_ESCALATION_TIMEOUT_S
from broker.permission.module import PERMISSION_DECISION_TIMEOUT_S
from broker.protocol.constants import HOOK_SETTINGS_TIMEOUT
from broker.protocol.constants import HOOK_WAIT_SECONDS
from broker.session.broker import ASK_DECISION_TIMEOUT_S
from broker.session.broker import CLARIFY_TIMEOUT_S


def test_hook_deadline_fires_before_claude_codes() -> None:
    # The hook must hit its own deadline and exit 0 on its own terms. If Claude
    # Code's settings.json timeout fired first it would kill the process
    # mid-wait and the degraded outcome would stop being predictable.
    assert HOOK_WAIT_SECONDS < HOOK_SETTINGS_TIMEOUT


def test_permission_decision_deadline_sits_inside_the_hook_wait() -> None:
    # A permission decision that outlived the hook's wait would be an answer
    # the module believes in and Claude Code never saw.
    assert PERMISSION_DECISION_TIMEOUT_S < HOOK_WAIT_SECONDS


def test_ask_decision_deadline_sits_inside_the_hook_wait() -> None:
    assert ASK_DECISION_TIMEOUT_S < HOOK_WAIT_SECONDS


def test_clarify_deadline_sits_inside_its_escalation_deadline() -> None:
    # The broker's own deadline must expire first so it fails loud on its own
    # terms rather than being cut off by the master's escalation wait.
    assert CLARIFY_TIMEOUT_S < CLARIFY_ESCALATION_TIMEOUT_S
