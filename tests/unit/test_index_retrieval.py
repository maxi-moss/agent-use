"""retrieve() over a hand-built index with a fake intent vector: seed choice,
one-hop expansion rules, ordering, and loud failures."""

from pathlib import Path

import pytest

from broker.index.retrieval import RetrievalError, retrieve
from broker.index.schemas import (
    Edge,
    EdgeKind,
    FileExtraction,
    Symbol,
    SymbolKind,
)
from broker.index.store import IndexStore

MODEL_ID = "test-embed-model"


class FixedEmbedder:
    def __init__(self, vector: list[float], model_id: str = MODEL_ID) -> None:
        self.vector = vector
        self.model_id = model_id
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str], *, timeout_s: float) -> list[list[float]]:
        del timeout_s
        self.calls.append(texts)
        return [self.vector for _ in texts]


def sym(path: str, scope: str, name: str, kind: SymbolKind, sig: str) -> Symbol:
    return Symbol(
        path=path,
        scope=scope,
        name=name,
        kind=kind,
        start_line=1,
        end_line=2,
        signature=sig,
    )


def build_index(db: Path, *, model_id: str | None = MODEL_ID) -> None:
    """Two files: a service class with two methods, a factory function, a base class."""
    store = IndexStore.open(db)
    store.replace_file(
        FileExtraction(
            path="a.py",
            symbols=[
                sym("a.py", "", "Service", SymbolKind.CLASS, "class Service(Base):"),
                sym(
                    "a.py",
                    "Service",
                    "send",
                    SymbolKind.METHOD,
                    "def send(self) -> None:",
                ),
                sym(
                    "a.py",
                    "Service",
                    "other",
                    SymbolKind.METHOD,
                    "def other(self) -> None:",
                ),
                sym("a.py", "", "Base", SymbolKind.CLASS, "class Base:"),
            ],
            references=[],
            imports=[],
        ),
        "h1",
        "python",
    )
    store.replace_file(
        FileExtraction(
            path="b.py",
            symbols=[
                sym("b.py", "", "make", SymbolKind.FUNCTION, "def make() -> Service:"),
                sym(
                    "b.py",
                    "",
                    "unrelated",
                    SymbolKind.FUNCTION,
                    "def unrelated() -> None:",
                ),
            ],
            references=[],
            imports=[],
        ),
        "h2",
        "python",
    )
    store.replace_edges(
        [
            Edge(
                source="a.py::Service",
                target="a.py::Service.send",
                kind=EdgeKind.DEFINES,
            ),
            Edge(
                source="a.py::Service",
                target="a.py::Service.other",
                kind=EdgeKind.DEFINES,
            ),
            Edge(source="a.py::Service", target="a.py::Base", kind=EdgeKind.INHERITS),
            Edge(source="b.py::make", target="a.py::Service", kind=EdgeKind.CALLS),
            Edge(
                source="b.py::make",
                target="a.py::Service",
                kind=EdgeKind.REFERENCES_TYPE,
            ),
            Edge(
                source="a.py::Service.send",
                target="b.py::unrelated",
                kind=EdgeKind.CALLS,
            ),
            Edge(source="a.py", target="b.py::make", kind=EdgeKind.IMPORTS),
        ]
    )
    if model_id is not None:
        store.write_embedding_model_id(model_id)
    store.store_embeddings(
        [
            ("a.py::Service.send", "t", [1.0, 0.0, 0.0]),
            ("a.py::Service.other", "t", [0.0, 1.0, 0.0]),
            ("a.py::Service", "t", [0.0, 0.0, 1.0]),
            ("a.py::Base", "t", [0.0, 0.0, -1.0]),
            ("b.py::make", "t", [0.9, 0.1, 0.0]),
            ("b.py::unrelated", "t", [-1.0, 0.0, 0.0]),
        ]
    )
    store.close()


async def test_seeds_are_pure_top_k_and_expansion_follows_hop_rules(
    tmp_path: Path,
) -> None:
    db = tmp_path / "index.sqlite"
    build_index(db)
    ctx = await retrieve(
        "send a message",
        Path("/repo"),
        index_path=db,
        embedder=FixedEmbedder([1.0, 0.0, 0.0]),
    )
    seeds = [s for s in ctx.symbols if s.score is not None]
    assert [s.qualified_name for s in seeds] == [
        "a.py::Service.send",   # 1.0
        "b.py::make",           # ~0.99
        "a.py::Base",           # 0.0 — ties broken by qualified name
        "a.py::Service",        # 0.0
        "a.py::Service.other",  # 0.0
    ]
    expansion = [s.qualified_name for s in ctx.symbols if s.score is None]
    # `send` calls `unrelated` (call out) → pulled in even though its own score is -1
    assert expansion == ["b.py::unrelated"]
    kinds = {(e.source, e.target, e.kind) for e in ctx.edges}
    assert ("a.py::Service", "a.py::Service.send", EdgeKind.DEFINES) in kinds  # owner
    assert ("a.py::Service", "a.py::Base", EdgeKind.INHERITS) in kinds  # bases
    assert ("b.py::make", "a.py::Service", EdgeKind.REFERENCES_TYPE) in kinds
    # IMPORTS never appears as an edge; it is rendered per seed file
    assert not any(k is EdgeKind.IMPORTS for _, _, k in kinds)
    assert ctx.imports == {"a.py": ["b.py::make"]}
    service = next(s for s in ctx.symbols if s.qualified_name == "a.py::Service")
    assert service.methods == ["other", "send"]  # equal start_line → ordered by name


async def test_expansion_rank_is_the_best_pulling_seed(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    build_index(db)
    ctx = await retrieve(
        "x",
        Path("/repo"),
        index_path=db,
        embedder=FixedEmbedder([1.0, 0.0, 0.0]),
    )
    unrelated = next(s for s in ctx.symbols if s.qualified_name == "b.py::unrelated")
    assert abs(unrelated.rank - 1.0) < 1e-9  # pulled by `send`, score 1.0


async def test_missing_index_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError, match="no code index"):
        await retrieve(
            "x", Path("/repo"), index_path=tmp_path / "nope.sqlite",
            embedder=FixedEmbedder([1.0]),
        )


async def test_index_without_embeddings_fails_loud(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    IndexStore.open(db).close()
    embedder = FixedEmbedder([1.0])
    with pytest.raises(RetrievalError, match="no embedded symbols"):
        await retrieve("x", Path("/repo"), index_path=db, embedder=embedder)
    assert embedder.calls == []  # the intent is never embedded against an empty index


async def test_index_predating_model_tracking_fails_loud(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    build_index(db, model_id=None)
    with pytest.raises(RetrievalError, match="predates embedding-model tracking"):
        await retrieve(
            "x",
            Path("/repo"),
            index_path=db,
            embedder=FixedEmbedder([1.0, 0.0, 0.0]),
        )


async def test_mismatched_embedding_model_fails_loud(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    build_index(db, model_id="old-model")
    with pytest.raises(RetrievalError, match="'old-model'") as exc_info:
        await retrieve(
            "x",
            Path("/repo"),
            index_path=db,
            embedder=FixedEmbedder([1.0, 0.0, 0.0], model_id="new-model"),
        )
    assert "'new-model'" in str(exc_info.value)
