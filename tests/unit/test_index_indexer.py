"""index_repo end to end over copies of the fixture repos: resolved edges,
incremental re-parse, deletions, and loud failure off a repo root."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from broker.index import indexer
from broker.index.indexer import IndexingError, index_repo
from broker.index.schemas import EdgeKind
from broker.index.store import IndexStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "indexer_inputs"
MODEL_ID = "fake-embed-model"
# An ambient GIT_DIR (e.g. from a `git rebase --exec` running this suite)
# must not redirect `git init` away from the fixture directory it targets.
_GIT_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


class FakeEmbedder:
    """Deterministic 4-d vectors from the text hash; records every batch."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.batches: list[list[str]] = []

    async def embed(self, texts: list[str], *, timeout_s: float) -> list[list[float]]:
        del timeout_s
        self.batches.append(list(texts))
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            out.append([float(b) + 1.0 for b in digest[:4]])
        return out

    def embedded_names(self) -> set[str]:
        # embedding_text puts the qualified name on line 2
        return {text.split("\n")[1] for batch in self.batches for text in batch}


def make_repo(tmp_path: Path, fixture: str) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURES / fixture, repo)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=_GIT_ENV)
    return repo.resolve()


def edge_set(db: Path) -> set[tuple[str, str, EdgeKind]]:
    store = IndexStore.open(db)
    try:
        return {(e.source, e.target, e.kind) for e in store.edges()}
    finally:
        store.close()


async def test_python_edges_resolve_across_files(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, "python_repo")
    db = tmp_path / "index.sqlite"
    summary = await index_repo(repo, db, FakeEmbedder())
    assert summary.files_changed == 7
    edges = edge_set(db)
    C, I, T, M, D = (
        EdgeKind.CALLS, EdgeKind.INHERITS, EdgeKind.REFERENCES_TYPE,
        EdgeKind.IMPORTS, EdgeKind.DEFINES,
    )
    expected = {
        # module import + attribute call
        (
            "app/chat.py::ChatService.__init__",
            "app/providers/factory.py::create_provider",
            C,
        ),
        # self.method on the enclosing class
        ("app/chat.py::ChatService.send", "app/chat.py::ChatService._prepare", C),
        (
            "app/providers/anthropic.py::AnthropicProvider.complete",
            "app/providers/anthropic.py::AnthropicProvider._call",
            C,
        ),
        # relative import → base class
        (
            "app/providers/anthropic.py::AnthropicProvider",
            "app/providers/base.py::Provider",
            I,
        ),
        # class instantiation through an absolute import
        (
            "app/providers/factory.py::create_provider",
            "app/providers/anthropic.py::AnthropicProvider",
            C,
        ),
        ("app/providers/factory.py::create_provider", "app/settings.py::Settings", T),
        (
            "app/providers/factory.py::create_provider",
            "app/providers/base.py::Provider",
            T,
        ),
        # imports render per file
        ("app/chat.py", "app/providers/factory.py", M),
        ("app/chat.py", "app/settings.py::Settings", M),
        # ownership
        ("app/chat.py::ChatService", "app/chat.py::ChatService.send", D),
    }
    assert expected <= edges
    # The receiver of `self.provider.complete` is unknown, so the call is dropped.
    assert not any(
        s == "app/chat.py::ChatService.send" and t.endswith("complete")
        for s, t, _ in edges
    )
    # external names never become edges
    assert not any("pydantic" in t or "ValueError" in t for _, t, _ in edges)


async def test_typescript_and_javascript_edges(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, "typescript_repo")
    db = tmp_path / "index.sqlite"
    await index_repo(repo, db, FakeEmbedder())
    edges = edge_set(db)
    C, I, M = EdgeKind.CALLS, EdgeKind.INHERITS, EdgeKind.IMPORTS
    expected = {
        (
            "src/chat.ts::ChatService.constructor",
            "src/providers/factory.ts::createProvider",
            C,
        ),
        (
            "src/providers/factory.ts::createProvider",
            "src/providers/anthropic.ts::AnthropicProvider",
            C,
        ),
        (
            "src/providers/anthropic.ts::AnthropicProvider",
            "src/providers/base.ts::Provider",
            I,
        ),
        (
            "src/providers/anthropic.ts::AnthropicProvider.complete",
            "src/providers/anthropic.ts::AnthropicProvider.call",
            C,
        ),
        # JS → TS relative import with extension probing
        ("src/index.js", "src/chat.ts::handleChat", M),
        ("src/index.js::main", "src/chat.ts::handleChat", C),
    }
    assert expected <= edges
    assert ("src/chat.ts::handleChat", "src/chat.ts::ChatService.send", C) not in edges


async def test_unresolved_receivers_do_not_match_unique_method_names(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path, "python_repo")
    (repo / "unrelated.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "class Registry:\n"
        "    def get(self): pass\n"
        "\n"
        "class Queue:\n"
        "    def resolve(self): pass\n"
        "\n"
        "class Store:\n"
        "    def close(self): pass\n"
        "\n"
        "def use_external_objects(cwd: Path, writer):\n"
        "    os.environ.get('HOME')\n"
        "    cwd.resolve()\n"
        "    writer.close()\n"
    )
    db = tmp_path / "index.sqlite"
    await index_repo(repo, db, FakeEmbedder())
    assert not any(
        source == "unrelated.py::use_external_objects" and kind is EdgeKind.CALLS
        for source, _, kind in edge_set(db)
    )


async def test_incremental_reparses_only_changed_file_and_reresolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path, "python_repo")
    db = tmp_path / "index.sqlite"
    await index_repo(repo, db, FakeEmbedder())
    parsed: list[str] = []
    real = indexer.extract_file

    def spy(spec: Any, path: str, source: bytes) -> Any:
        parsed.append(path)
        return real(spec, path, source)

    monkeypatch.setattr(indexer, "extract_file", spy)
    target = repo / "app" / "providers" / "anthropic.py"
    target.write_text(target.read_text().replace("_call", "_invoke"))
    summary = await index_repo(repo, db, FakeEmbedder())
    assert parsed == ["app/providers/anthropic.py"]
    assert summary.files_changed == 1
    edges = edge_set(db)
    assert (
        "app/providers/anthropic.py::AnthropicProvider.complete",
        "app/providers/anthropic.py::AnthropicProvider._invoke",
        EdgeKind.CALLS,
    ) in edges
    assert not any(t.endswith("::AnthropicProvider._call") for _, t, _ in edges)


async def test_removed_file_drops_symbols_and_edges(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, "python_repo")
    db = tmp_path / "index.sqlite"
    await index_repo(repo, db, FakeEmbedder())
    (repo / "app" / "settings.py").unlink()
    summary = await index_repo(repo, db, FakeEmbedder())
    assert summary.files_removed == 1
    edges = edge_set(db)
    assert not any("app/settings.py" in t for _, t, _ in edges)


async def test_unchanged_run_parses_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path, "python_repo")
    db = tmp_path / "index.sqlite"
    await index_repo(repo, db, FakeEmbedder())

    def never(*_: object) -> Any:
        pytest.fail("an unchanged file was re-parsed")

    monkeypatch.setattr(indexer, "extract_file", never)
    assert (await index_repo(repo, db, FakeEmbedder())).files_changed == 0


async def test_non_git_directory_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(IndexingError):
        await index_repo(
            tmp_path.resolve(), tmp_path / "index.sqlite", FakeEmbedder()
        )


async def test_subdirectory_of_a_repo_is_refused(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, "python_repo")
    with pytest.raises(IndexingError, match="not the root"):
        await index_repo(
            repo / "app", tmp_path / "index.sqlite", FakeEmbedder()
        )


async def test_non_git_directory_fails_loud_under_an_ambient_git_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller running under its own GIT_DIR (a rebase --exec, a git hook)
    must not leak into which repository this git subprocess resolves."""
    unrelated_repo = tmp_path / "unrelated"
    unrelated_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=unrelated_repo, check=True, env=_GIT_ENV)
    monkeypatch.setenv("GIT_DIR", str(unrelated_repo / ".git"))
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(IndexingError):
        await index_repo(target, tmp_path / "index.sqlite", FakeEmbedder())
