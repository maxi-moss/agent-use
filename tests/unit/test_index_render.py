"""render_relevant_code and fit_to_budget over a hand-built GroundingContext,
byte for byte."""

import pytest

from broker.index import render
from broker.index.render import fit_to_budget, render_relevant_code
from broker.index.schemas import (
    ContextEdge,
    ContextSymbol,
    EdgeKind,
    GroundingContext,
    SymbolKind,
)


def cs(
    qname: str,
    kind: SymbolKind,
    sig: str,
    *,
    score: float | None,
    rank: float,
    fields: list[str] | None = None,
    methods: list[str] | None = None,
) -> ContextSymbol:
    path = qname.split("::")[0]
    return ContextSymbol(
        qualified_name=qname, path=path, kind=kind, start_line=10, end_line=20,
        signature=sig, fields=fields or [], methods=methods or [], score=score, rank=rank,
    )


CONTEXT = GroundingContext(
    symbols=[
        cs("a.py::Service.send", SymbolKind.METHOD, "def send(self) -> None:", score=0.9, rank=0.9),
        cs("b.py::make", SymbolKind.FUNCTION, "@cached\ndef make() -> Service:", score=0.5, rank=0.5),
        cs(
            "a.py::Service", SymbolKind.CLASS, "class Service(Base):", score=None, rank=0.9,
            fields=["name: str"], methods=["send", "other"],
        ),
        cs("b.py::unrelated", SymbolKind.FUNCTION, "def unrelated() -> None:", score=None, rank=0.9),
    ],
    edges=[
        ContextEdge(source="a.py::Service", target="a.py::Service.send", kind=EdgeKind.DEFINES),
        ContextEdge(source="a.py::Service.send", target="b.py::unrelated", kind=EdgeKind.CALLS),
        ContextEdge(source="b.py::make", target="a.py::Service", kind=EdgeKind.CALLS),
        ContextEdge(source="a.py::Service", target="a.py::Base", kind=EdgeKind.INHERITS),
    ],
    imports={"a.py": ["b.py::make"]},
)

EXPECTED = """# Relevant code

## a.py
imports: b.py::make
### Service (class, lines 10-20)
    class Service(Base):
    name: str
  methods: send, other
  called by: b.py::make
  bases: a.py::Base
  ### Service.send (method, seed, lines 10-20)
      def send(self) -> None:
    calls: b.py::unrelated
    owner: a.py::Service

## b.py
### make (function, seed, lines 10-20)
    @cached
    def make() -> Service:
  calls: a.py::Service
### unrelated (function, lines 10-20)
    def unrelated() -> None:
  called by: a.py::Service.send"""


def test_render_is_byte_exact() -> None:
    assert render_relevant_code(CONTEXT) == EXPECTED


def test_render_is_deterministic() -> None:
    assert render_relevant_code(CONTEXT) == render_relevant_code(CONTEXT.model_copy(deep=True))


def test_budget_drops_lowest_ranked_expansion_and_never_seeds(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    monkeypatch.setattr(render, "BUDGET_CHARS", 1)
    trimmed = fit_to_budget(CONTEXT)
    assert [s.qualified_name for s in trimmed.symbols] == ["a.py::Service.send", "b.py::make"]
    assert trimmed.edges == CONTEXT.edges  # edges still name dropped neighbours
