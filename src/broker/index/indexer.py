"""Index one repository: list files, parse what changed, resolve edges
globally, embed symbols whose text changed."""

import hashlib
import logging
import os
import subprocess
from pathlib import Path

from broker.index.edge_resolution import resolve_edges
from broker.index.embedding import Embedder
from broker.index.languages import spec_for
from broker.index.schemas import IndexSummary, Symbol, SymbolKind
from broker.index.store import IndexStore
from broker.index.symbol_extraction import extract_file

logger = logging.getLogger(__name__)

GIT_TIMEOUT_S = 30.0
EMBED_BATCH_TIMEOUT_S = 120.0
# Request caps are 2048 inputs / 300k tokens; ~4 chars per token keeps a batch
# comfortably inside without a tokenizer dependency.
_BATCH_MAX_ITEMS = 256
_BATCH_MAX_CHARS = 600_000

_Pending = tuple[str, str, str]  # qualified_name, text_hash, text


class IndexingError(Exception):
    """The repository cannot be indexed. Fail loud."""


def _git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return its stdout.

    Strips inherited ``GIT_*`` environment variables so repository discovery
    is decided by ``repo`` alone: a caller running under its own ``GIT_DIR``
    (a git hook, a rebase ``--exec``) must not leak into which repository
    this git subprocess resolves.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            env=env,
        )
    except OSError as exc:
        raise IndexingError(f"git is not runnable in {repo}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise IndexingError(
            f"git {args[0]} timed out after {GIT_TIMEOUT_S}s in {repo}"
        ) from exc
    if proc.returncode != 0:
        raise IndexingError(
            f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}"
        )
    return proc.stdout


def list_repo_files(repo: Path) -> list[str]:
    """List tracked and untracked-unignored files, repo-relative and sorted.

    Args:
        repo: Absolute, resolved repository root.

    Returns:
        Paths of regular files that exist on disk.

    Raises:
        IndexingError: ``repo`` is not the top level of a git repository.
    """
    top = _git(repo, "rev-parse", "--show-toplevel").strip()
    if Path(top).resolve() != repo:
        raise IndexingError(
            f"{repo} is not the root of a git repository (top level is {top!r})"
        )
    tracked = _git(repo, "ls-files", "-z")
    untracked = _git(repo, "ls-files", "-z", "--others", "--exclude-standard")
    names = {n for n in (tracked + untracked).split("\0") if n}
    return sorted(n for n in names if (repo / n).is_file())


def embedding_text(symbol: Symbol, method_signatures: list[str]) -> str:
    """Compose the text embedded for one symbol.

    Args:
        symbol: A function, method or class.
        method_signatures: For a class, its methods' signatures in source order.

    Returns:
        Path, qualified name, signature, docstring, then body (functions and
        methods) or fields and method signatures (classes). Empty parts are
        omitted.
    """
    if symbol.kind is SymbolKind.CLASS:
        parts = [
            symbol.path, symbol.qualified_name, symbol.signature, symbol.docstring,
            *symbol.fields, *method_signatures,
        ]
    else:
        parts = [
            symbol.path, symbol.qualified_name, symbol.signature, symbol.docstring,
            symbol.body,
        ]
    return "\n".join(part for part in parts if part)


def _batches(pending: list[_Pending]) -> list[list[_Pending]]:
    """Split pending symbols into request-sized embedding batches."""
    batches: list[list[_Pending]] = []
    current: list[_Pending] = []
    chars = 0
    for item in pending:
        size = len(item[2])
        if current and (
            len(current) >= _BATCH_MAX_ITEMS or chars + size > _BATCH_MAX_CHARS
        ):
            batches.append(current)
            current, chars = [], 0
        current.append(item)
        chars += size
    if current:
        batches.append(current)
    return batches


async def _embed_changed(store: IndexStore, embedder: Embedder) -> int:
    """Embed every embeddable symbol whose text hash is new; return how many."""
    hashes = store.embedding_hashes()
    symbols = store.embeddable_symbols()
    methods = store.methods_of(
        s.qualified_name for s in symbols if s.kind is SymbolKind.CLASS
    )
    pending: list[_Pending] = []
    for symbol in symbols:
        text = embedding_text(
            symbol, [sig for _, sig in methods.get(symbol.qualified_name, [])]
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if hashes.get(symbol.qualified_name) == digest:
            continue
        pending.append((symbol.qualified_name, digest, text))
    embedded = 0
    for batch in _batches(pending):
        vectors = await embedder.embed(
            [text for _, _, text in batch], timeout_s=EMBED_BATCH_TIMEOUT_S
        )
        store.store_embeddings(
            [
                (qname, digest, vector)
                for (qname, digest, _), vector in zip(batch, vectors, strict=True)
            ]
        )
        embedded += len(batch)
        logger.info("embedded %d symbols", len(batch))
    return embedded


async def index_repo(repo: Path, db_path: Path, embedder: Embedder) -> IndexSummary:
    """Bring the index at ``db_path`` up to date with ``repo``.

    Args:
        repo: Absolute, resolved repository root.
        db_path: The repository's index file.
        embedder: Embedding backend for changed symbols; its ``model_id`` is
            recorded so retrieval can refuse a later model swap.

    Returns:
        Counts for the CLI summary line.

    Raises:
        IndexingError: The repository could not be listed.
        EmbeddingError: An embedding batch failed.
    """
    files = list_repo_files(repo)
    store = IndexStore.open(db_path)
    try:
        store.write_embedding_model_id(embedder.model_id)
        known = store.file_hashes()
        seen: set[str] = set()
        changed = 0
        for rel in files:
            spec = spec_for(rel)
            if spec is None:
                continue
            seen.add(rel)
            source = (repo / rel).read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if known.get(rel) == digest:
                continue
            store.replace_file(extract_file(spec, rel, source), digest, spec.name)
            changed += 1
            logger.info("indexed %s", rel)
        removed = sorted(path for path in known if path not in seen)
        for path in removed:
            store.delete_file(path)
            logger.info("removed %s", path)
        edges = resolve_edges(
            store.symbol_keys(),
            store.references(),
            store.imports(),
            store.file_languages(),
        )
        store.replace_edges(edges)
        embedded = await _embed_changed(store, embedder)
        store.prune_embeddings()
        symbols, edge_count = store.counts()
        return IndexSummary(
            files_changed=changed,
            files_removed=len(removed),
            symbols=symbols,
            edges=edge_count,
            embedded=embedded,
        )
    finally:
        store.close()
