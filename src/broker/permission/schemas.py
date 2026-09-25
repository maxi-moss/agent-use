"""The two calls the permission classifier may make.

Both carry only reasoning: the decision is the tool name, and the reasoning is
what the developer reads in the permission log.

Class names, field names, `Field` descriptions and docstrings of these models
are sent to the model.
"""

from pydantic import BaseModel, ConfigDict


class AllowCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str


class EscalateCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str


PermissionResult = AllowCall | EscalateCall
