"""Session broker process configuration, passed as --config-json at spawn."""

from pydantic import BaseModel, ConfigDict


class SessionBrokerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    socket_path: str
    master_socket_path: str
    cwd: str
    anchor_pane: str
    intent: str
    budget_count: int = 0  # persisted count resumes across broker death
    model_id: str
    max_tokens: int
    watchdog_seconds: float
    budget_max: int
