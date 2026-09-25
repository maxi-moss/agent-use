"""Data shapes of the code index: symbols, as-written references, edges."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SymbolKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    TYPE = "type"


class EdgeKind(StrEnum):
    CALLS = "CALLS"
    DEFINES = "DEFINES"
    INHERITS = "INHERITS"
    IMPORTS = "IMPORTS"
    REFERENCES_TYPE = "REFERENCES_TYPE"


class RefKind(StrEnum):
    """An as-written reference; global resolution turns it into an edge or drops it."""

    CALL = "call"
    BASE = "base"
    TYPE = "type"


def join_qualified_name(path: str, inner: str) -> str:
    """Build ``path::inner``, the low-level join behind the qualified-name format."""
    return f"{path}::{inner}"


def split_qualified_name(qname: str) -> tuple[str, str]:
    """Split ``path::inner`` back into its path and inner halves."""
    path, _, inner = qname.partition("::")
    return path, inner


def qualified_name(path: str, scope: str, name: str) -> str:
    """Build ``path::Scope.name``, the repository-wide identity of a symbol."""
    inner = f"{scope}.{name}" if scope else name
    return join_qualified_name(path, inner)


class Symbol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    scope: str
    name: str
    kind: SymbolKind
    start_line: int
    end_line: int
    signature: str
    docstring: str = ""
    body: str = ""
    fields: list[str] = Field(default_factory=list[str])

    @property
    def qualified_name(self) -> str:
        return qualified_name(self.path, self.scope, self.name)


class SymbolKey(BaseModel):
    """A symbol's identity without its text; what resolution works on."""

    model_config = ConfigDict(extra="forbid")

    qualified_name: str
    path: str
    scope: str
    name: str
    kind: SymbolKind


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    kind: RefKind
    target: str


class Import(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    local_name: str
    module: str
    imported_name: str


class FileExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    symbols: list[Symbol]
    references: list[Reference]
    imports: list[Import]


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    kind: EdgeKind


class IndexSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files_changed: int
    files_removed: int
    symbols: int
    edges: int
    embedded: int


class ContextSymbol(BaseModel):
    """One symbol in the retrieved neighbourhood, as rendered for grounding."""

    model_config = ConfigDict(extra="forbid")

    qualified_name: str
    path: str
    kind: SymbolKind
    start_line: int
    end_line: int
    signature: str
    fields: list[str]
    methods: list[str]
    score: float | None  # cosine similarity for seeds; None for expansion nodes
    rank: float  # own score, or the best score among the seeds that pulled it in


class GroundingContext(BaseModel):
    """Seeds and their one-hop neighbourhood.

    ``symbols`` are ordered seeds first by score descending, then expansion
    nodes by rank descending, ties broken by qualified name — so the last
    expansion node is always the cheapest to drop.
    """

    model_config = ConfigDict(extra="forbid")

    symbols: list[ContextSymbol]
    edges: list[Edge]
    imports: dict[str, list[str]]
