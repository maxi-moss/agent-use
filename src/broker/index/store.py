"""SQLite persistence for one repository's index.

One file per repository. A file's symbols, as-written references and imports
are replaced together in one transaction, so an interrupted run leaves whole
files, never half of one. Edges are replaced whole after global resolution.
"""

import json
import math
import sqlite3
from array import array
from collections.abc import Iterable
from pathlib import Path

from broker.index.schemas import (
    Edge,
    EdgeKind,
    FileExtraction,
    Import,
    RefKind,
    Reference,
    Symbol,
    SymbolKey,
    SymbolKind,
    split_qualified_name,
)

EMBEDDABLE_KINDS: tuple[SymbolKind, ...] = (
    SymbolKind.FUNCTION,
    SymbolKind.METHOD,
    SymbolKind.CLASS,
)
_EMBEDDABLE_KINDS_SQL = (
    "(" + ", ".join(f"'{kind.value}'" for kind in EMBEDDABLE_KINDS) + ")"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL,
    language TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS symbols (
    qualified_name TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    scope TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT NOT NULL,
    docstring TEXT NOT NULL,
    body TEXT NOT NULL,
    fields TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS symbols_path ON symbols (path);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols (name);
CREATE TABLE IF NOT EXISTS refs (
    path TEXT NOT NULL,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    target TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS refs_path ON refs (path);
CREATE TABLE IF NOT EXISTS imports (
    path TEXT NOT NULL,
    local_name TEXT NOT NULL,
    module TEXT NOT NULL,
    imported_name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS imports_path ON imports (path);
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    kind TEXT NOT NULL,
    PRIMARY KEY (source, target, kind)
);
CREATE INDEX IF NOT EXISTS edges_target ON edges (target);
CREATE TABLE IF NOT EXISTS embeddings (
    qualified_name TEXT PRIMARY KEY,
    text_hash TEXT NOT NULL,
    vector BLOB NOT NULL
);
"""

_SYMBOL_COLUMNS = (
    "qualified_name, path, scope, name, kind, start_line, end_line, "
    "signature, docstring, body, fields"
)


def _symbol_row(row: sqlite3.Row) -> Symbol:
    """Build a ``Symbol`` from a database row."""
    return Symbol(
        path=row["path"],
        scope=row["scope"],
        name=row["name"],
        kind=SymbolKind(row["kind"]),
        start_line=row["start_line"],
        end_line=row["end_line"],
        signature=row["signature"],
        docstring=row["docstring"],
        body=row["body"],
        fields=json.loads(row["fields"]),
    )


def unit_vector(values: Iterable[float]) -> array[float]:
    """Return ``values`` L2-normalised as float32.

    Raises:
        ValueError: The vector has zero length.
    """
    vector = array("f", values)
    norm = math.sqrt(math.sumprod(vector, vector))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return array("f", (x / norm for x in vector))


class IndexStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        self._conn = conn

    @classmethod
    def open(cls, path: Path) -> "IndexStore":
        """Open (creating if needed) the index file at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.executescript(_SCHEMA)
        return cls(conn)

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    # ── files ──────────────────────────────────────────────────────────

    def file_hashes(self) -> dict[str, str]:
        """Map each indexed file to its stored content hash."""
        rows = self._conn.execute("SELECT path, hash FROM files")
        return {row["path"]: row["hash"] for row in rows}

    def file_languages(self) -> dict[str, str]:
        """Map each indexed file to its language name."""
        rows = self._conn.execute("SELECT path, language FROM files")
        return {row["path"]: row["language"] for row in rows}

    def replace_file(
        self, extraction: FileExtraction, digest: str, language: str
    ) -> None:
        """Replace everything stored for one file, atomically."""
        with self._conn:
            self._delete_path(extraction.path)
            self._conn.execute(
                "INSERT INTO files (path, hash, language) VALUES (?, ?, ?)",
                (extraction.path, digest, language),
            )
            self._conn.executemany(
                f"INSERT INTO symbols ({_SYMBOL_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        s.qualified_name, s.path, s.scope, s.name, s.kind.value,
                        s.start_line, s.end_line, s.signature, s.docstring, s.body,
                        json.dumps(s.fields),
                    )
                    for s in extraction.symbols
                ],
            )
            self._conn.executemany(
                "INSERT INTO refs (path, source, kind, target) VALUES (?, ?, ?, ?)",
                [
                    (extraction.path, r.source, r.kind.value, r.target)
                    for r in extraction.references
                ],
            )
            self._conn.executemany(
                "INSERT INTO imports (path, local_name, module, imported_name) "
                "VALUES (?, ?, ?, ?)",
                [
                    (i.path, i.local_name, i.module, i.imported_name)
                    for i in extraction.imports
                ],
            )

    def delete_file(self, path: str) -> None:
        """Remove everything stored for one file, atomically."""
        with self._conn:
            self._delete_path(path)

    def _delete_path(self, path: str) -> None:
        """Delete every row keyed to one file path."""
        for table in ("files", "symbols", "refs", "imports"):
            self._conn.execute(f"DELETE FROM {table} WHERE path = ?", (path,))

    # ── resolution inputs ──────────────────────────────────────────────

    def symbol_keys(self) -> list[SymbolKey]:
        """Return every symbol's identity, ordered by qualified name."""
        rows = self._conn.execute(
            "SELECT qualified_name, path, scope, name, kind FROM symbols "
            "ORDER BY qualified_name"
        )
        return [
            SymbolKey(
                qualified_name=row["qualified_name"],
                path=row["path"],
                scope=row["scope"],
                name=row["name"],
                kind=SymbolKind(row["kind"]),
            )
            for row in rows
        ]

    def references(self) -> list[Reference]:
        """Return every stored as-written reference, ordered."""
        rows = self._conn.execute(
            "SELECT source, kind, target FROM refs ORDER BY source, kind, target"
        )
        return [
            Reference(
                source=row["source"], kind=RefKind(row["kind"]), target=row["target"]
            )
            for row in rows
        ]

    def imports(self) -> list[Import]:
        """Return every stored import, ordered."""
        rows = self._conn.execute(
            "SELECT path, local_name, module, imported_name FROM imports "
            "ORDER BY path, local_name, module, imported_name"
        )
        return [
            Import(
                path=row["path"],
                local_name=row["local_name"],
                module=row["module"],
                imported_name=row["imported_name"],
            )
            for row in rows
        ]

    # ── edges ──────────────────────────────────────────────────────────

    def replace_edges(self, edges: Iterable[Edge]) -> None:
        """Replace the whole edge set with ``edges``, atomically."""
        with self._conn:
            self._conn.execute("DELETE FROM edges")
            self._conn.executemany(
                "INSERT OR IGNORE INTO edges (source, target, kind) VALUES (?, ?, ?)",
                [(e.source, e.target, e.kind.value) for e in edges],
            )

    def edges(self) -> list[Edge]:
        """Return every stored edge, ordered."""
        rows = self._conn.execute(
            "SELECT source, target, kind FROM edges ORDER BY source, target, kind"
        )
        return [
            Edge(source=row["source"], target=row["target"], kind=EdgeKind(row["kind"]))
            for row in rows
        ]

    def counts(self) -> tuple[int, int]:
        """Return the total symbol and edge counts."""
        symbols = self._conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()
        edges = self._conn.execute("SELECT COUNT(*) AS n FROM edges").fetchone()
        return int(symbols["n"]), int(edges["n"])

    # ── embeddings ─────────────────────────────────────────────────────

    def embeddable_symbols(self) -> list[Symbol]:
        """Return every function, method and class symbol, ordered by qualified name."""
        rows = self._conn.execute(
            f"SELECT {_SYMBOL_COLUMNS} FROM symbols "
            f"WHERE kind IN {_EMBEDDABLE_KINDS_SQL} ORDER BY qualified_name"
        )
        return [_symbol_row(row) for row in rows]

    def methods_of(
        self, class_qnames: Iterable[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Map each class to its methods' ``(name, signature)`` in source order."""
        out: dict[str, list[tuple[str, str]]] = {}
        for qname in class_qnames:
            path, scope = split_qualified_name(qname)
            rows = self._conn.execute(
                "SELECT name, signature FROM symbols "
                "WHERE kind = 'method' AND path = ? AND scope = ? "
                "ORDER BY start_line, name",
                (path, scope),
            )
            out[qname] = [(row["name"], row["signature"]) for row in rows]
        return out

    def embedding_hashes(self) -> dict[str, str]:
        """Map each embedded symbol to the text hash behind its stored vector."""
        rows = self._conn.execute("SELECT qualified_name, text_hash FROM embeddings")
        return {row["qualified_name"]: row["text_hash"] for row in rows}

    def store_embeddings(self, items: Iterable[tuple[str, str, list[float]]]) -> None:
        """Store ``(qualified_name, text_hash, vector)`` triples as unit float32 blobs."""
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (qualified_name, text_hash, vector) "
                "VALUES (?, ?, ?)",
                [
                    (qname, digest, unit_vector(vector).tobytes())
                    for qname, digest, vector in items
                ],
            )

    def prune_embeddings(self) -> None:
        """Drop embeddings whose symbol vanished or is no longer an embedded kind."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM embeddings WHERE qualified_name NOT IN "
                "(SELECT qualified_name FROM symbols "
                f"WHERE kind IN {_EMBEDDABLE_KINDS_SQL})"
            )

    def load_vectors(self) -> list[tuple[str, array[float]]]:
        """Return every stored embedding as a ``(qualified_name, vector)`` pair."""
        rows = self._conn.execute(
            "SELECT qualified_name, vector FROM embeddings ORDER BY qualified_name"
        )
        out: list[tuple[str, array[float]]] = []
        for row in rows:
            vector = array("f")
            vector.frombytes(row["vector"])
            out.append((row["qualified_name"], vector))
        return out

    # ── retrieval ──────────────────────────────────────────────────────

    def symbols_by_qualified_name(self, names: Iterable[str]) -> dict[str, Symbol]:
        """Return the symbols among ``names`` that exist, keyed by qualified name."""
        out: dict[str, Symbol] = {}
        for qname in names:
            row = self._conn.execute(
                f"SELECT {_SYMBOL_COLUMNS} FROM symbols WHERE qualified_name = ?",
                (qname,),
            ).fetchone()
            if row is not None:
                out[qname] = _symbol_row(row)
        return out

    def neighborhood_edges(self, names: Iterable[str]) -> list[Edge]:
        """Every non-import edge touching any of ``names``, sorted."""
        found: dict[tuple[str, str, str], Edge] = {}
        for qname in names:
            rows = self._conn.execute(
                "SELECT source, target, kind FROM edges "
                "WHERE kind != 'IMPORTS' AND (source = ? OR target = ?)",
                (qname, qname),
            )
            for row in rows:
                key = (row["source"], row["target"], row["kind"])
                found[key] = Edge(source=key[0], target=key[1], kind=EdgeKind(key[2]))
        return [found[key] for key in sorted(found)]

    def import_edges(self, paths: Iterable[str]) -> dict[str, list[str]]:
        """Map each of ``paths`` to the sorted targets of its IMPORTS edges."""
        out: dict[str, list[str]] = {}
        for path in paths:
            rows = self._conn.execute(
                "SELECT target FROM edges WHERE kind = 'IMPORTS' AND source = ? "
                "ORDER BY target",
                (path,),
            )
            targets = [row["target"] for row in rows]
            if targets:
                out[path] = targets
        return out
