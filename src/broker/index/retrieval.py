"""Retrieval at spawn: embed the intent, seed by cosine, expand one hop.

Read-only against the index; nothing here refreshes it. Every failure raises
``RetrievalError`` or ``EmbeddingError`` — there is no degraded path.
"""

import math
from pathlib import Path

from broker.index.embedding import Embedder
from broker.index.schemas import (
    ContextEdge,
    ContextSymbol,
    Edge,
    EdgeKind,
    GroundingContext,
    SymbolKind,
)
from broker.index.store import IndexStore, unit_vector

TOP_K = 5
INTENT_EMBED_TIMEOUT_S = 30.0


class RetrievalError(Exception):
    """The index cannot serve this spawn. Fail loud."""


def _hint(cwd: Path) -> str:
    """Return the CLI hint that rebuilds the index for ``cwd``."""
    return f"run `python -m broker.index {cwd}`"


async def retrieve(
    intent: str, cwd: Path, *, index_path: Path, embedder: Embedder
) -> GroundingContext:
    """Return the intent's code neighbourhood from the repository's index.

    Args:
        intent: The developer's intent, embedded verbatim.
        cwd: Repository root, used in error messages.
        index_path: The repository's index file.
        embedder: Embedding backend for the intent.

    Returns:
        Top-5 seeds by cosine similarity plus their one-hop expansion.

    Raises:
        RetrievalError: No index, no embedded symbols, or vector mismatch.
        EmbeddingError: The intent could not be embedded.
    """
    if not index_path.exists():
        raise RetrievalError(f"no code index for {cwd}; {_hint(cwd)}")
    store = IndexStore.open(index_path)
    try:
        vectors = store.load_vectors()
        if not vectors:
            raise RetrievalError(
                f"code index for {cwd} has no embedded symbols; {_hint(cwd)}"
            )
        [query] = await embedder.embed([intent], timeout_s=INTENT_EMBED_TIMEOUT_S)
        unit = unit_vector(query)
        try:
            ranked = sorted(
                ((math.sumprod(unit, vector), name) for name, vector in vectors),
                key=lambda item: (-item[0], item[1]),
            )
        except ValueError as exc:
            raise RetrievalError(
                f"code index for {cwd} holds vectors of another dimension; {_hint(cwd)}"
            ) from exc
        seeds = {name: score for score, name in ranked[:TOP_K]}
        return _expand(store, seeds)
    finally:
        store.close()


def _neighbor(edge: Edge, seeds: dict[str, float]) -> str | None:
    """The node ``edge`` pulls into the neighbourhood of a seed, per the design's hop rules."""
    if edge.kind is EdgeKind.CALLS:
        if edge.source in seeds:
            return edge.target
        if edge.target in seeds:
            return edge.source
    elif edge.kind in (EdgeKind.INHERITS, EdgeKind.REFERENCES_TYPE):
        if edge.source in seeds:
            return edge.target
    elif edge.kind is EdgeKind.DEFINES:
        if edge.target in seeds:
            return edge.source
    return None


def _expand(store: IndexStore, seeds: dict[str, float]) -> GroundingContext:
    """Expand seeds by one hop into a ``GroundingContext``."""
    rank: dict[str, float] = dict(seeds)
    kept: list[Edge] = []
    for edge in store.neighborhood_edges(seeds):
        neighbor = _neighbor(edge, seeds)
        if neighbor is None:
            continue
        kept.append(edge)
        if neighbor not in seeds:
            anchor = edge.source if edge.source in seeds else edge.target
            rank[neighbor] = max(rank.get(neighbor, -1.0), seeds[anchor])
    symbols = store.symbols_by_qualified_name(rank)
    methods = store.methods_of(
        q for q, s in symbols.items() if s.kind is SymbolKind.CLASS
    )
    ordered = sorted(symbols, key=lambda q: (q not in seeds, -rank[q], q))
    context_symbols = [
        ContextSymbol(
            qualified_name=q,
            path=symbols[q].path,
            kind=symbols[q].kind,
            start_line=symbols[q].start_line,
            end_line=symbols[q].end_line,
            signature=symbols[q].signature,
            fields=symbols[q].fields,
            methods=[name for name, _ in methods.get(q, [])],
            score=seeds.get(q),
            rank=rank[q],
        )
        for q in ordered
    ]
    seed_paths = sorted({symbols[q].path for q in seeds if q in symbols})
    return GroundingContext(
        symbols=context_symbols,
        edges=[ContextEdge(source=e.source, target=e.target, kind=e.kind) for e in kept],
        imports=store.import_edges(seed_paths),
    )
