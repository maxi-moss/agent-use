"""Shared broker configuration (one definition, two consumers).

Every `model_id` is PINNED HERE AND NOWHERE ELSE — the single most
consequential parameter. Load fails loud on an invalid overlay file — a
half-read config silently changing the model or budget is exactly the kind of
partial read the global rules forbid.
"""

import json
import os
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


def default_broker_home() -> Path:
    """Return ``$BROKER_HOME`` if set, else ``~/.broker``."""
    override = os.environ.get("BROKER_HOME")
    if override:
        return Path(override)
    return Path.home() / ".broker"


class ClassifierConfig(BaseModel):
    """The small model that judges permission prompts."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = "claude-haiku-4-5"  # PINNED — nowhere else in the tree
    max_tokens: int = 1024


# Bare tool names and command patterns only: a `/path`-anchored rule in a
# settings file resolves against that file's own directory, which is under the
# broker home rather than the session's working tree.
_DEFAULT_ALLOW = ["Read", "Glob", "Grep"]
_DEFAULT_ASK = ["Bash(git push:*)", "Bash(rm -rf:*)", "Bash(sudo:*)"]

# Rules that would hand back every shell command in one line.
_BLANKET_BASH = frozenset({"Bash", "Bash(*)"})


class PermissionRules(BaseModel):
    """Native Claude Code permission rules, authored by the developer.

    Nothing here is inferred from a project: rules are the enforcement layer,
    and a guessed rule is a grant nobody made.
    """

    model_config = ConfigDict(extra="forbid")

    allow: list[str] = _DEFAULT_ALLOW
    ask: list[str] = _DEFAULT_ASK
    deny: list[str] = []

    @field_validator("allow", "ask", "deny")
    @classmethod
    def _reject_self_defeating_rules(
        cls, value: list[str], info: ValidationInfo
    ) -> list[str]:
        """Reject rules that would delete the enforcement they configure.

        Args:
            value: The rule list being validated.
            info: Field context, used to name the offending list.

        Returns:
            The rule list, unchanged.

        Raises:
            ValueError: A rule names ``bypassPermissions``, or an ``allow``
                rule grants every shell command.
        """
        for entry in value:
            if "bypassPermissions" in entry:
                raise ValueError(
                    f"{info.field_name} rule {entry!r} names bypassPermissions, "
                    "which switches off the layer these rules exist to configure"
                )
            if info.field_name == "allow" and entry in _BLANKET_BASH:
                raise ValueError(
                    f"allow rule {entry!r} grants every shell command; list the "
                    "specific command patterns to allow instead"
                )
        return value


class BrokerConfig(BaseModel):
    # extra="forbid": an unrecognised key in config.json is a developer typo
    # that must fail loud, never be silently ignored.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = "claude-sonnet-5"  # PINNED — nowhere else in the tree
    max_tokens: int = 8192
    watchdog_seconds: float = 300.0
    budget_max: int = 8
    recent_turns_window: int = 20
    broker_home: Path = Field(default_factory=default_broker_home)
    classifier: ClassifierConfig = ClassifierConfig()
    permission_rules: PermissionRules = PermissionRules()


class AdoptedSession(BaseModel):
    """A live Claude session a replacement broker takes over instead of starting.

    Every field is required: a broker that only partly knows the session it is
    adopting would drive a pane it cannot read, or read a transcript it cannot
    drive.
    """

    model_config = ConfigDict(extra="forbid")

    pane_id: str
    claude_session_id: str
    transcript_path: str


class SessionBrokerConfig(BaseModel):
    """Session broker process configuration, passed as --config-json at spawn."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    socket_path: str
    master_socket_path: str
    broker_home: Path  # sent by the master; never re-derived from the env
    cwd: str
    anchor_pane: str
    intent: str
    budget_count: int = 0  # persisted count resumes across broker death
    model_id: str
    max_tokens: int
    classifier: ClassifierConfig
    watchdog_seconds: float
    budget_max: int
    claude_settings_path: str  # written by the master; passed to `agent start`
    adopt: AdoptedSession | None = None  # set only when reassigned


class ConfigError(Exception):
    """broker_home/config.json exists but cannot be used. Fail loud."""


def load() -> BrokerConfig:
    """Load the broker config: defaults overlaid with ``config.json``.

    Returns:
        The validated config.

    Raises:
        ConfigError: The overlay file is not valid JSON, is not a JSON
            object, or fails validation against ``BrokerConfig``.
    """
    home = default_broker_home()
    config_path = home / "config.json"
    overlay: dict[str, Any] = {}
    if config_path.exists():
        try:
            parsed: Any = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{config_path} is not valid JSON ({exc.msg})"
            ) from exc
        if not isinstance(parsed, dict):
            raise ConfigError(f"{config_path} is not a JSON object")
        overlay = cast(dict[str, Any], parsed)
    overlay.setdefault("broker_home", str(home))
    try:
        return BrokerConfig.model_validate(overlay)
    except ValueError as exc:
        raise ConfigError(f"{config_path}: invalid config: {exc}") from exc
