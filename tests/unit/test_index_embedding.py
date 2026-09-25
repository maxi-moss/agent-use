"""Embedding step: which symbols are embedded, when they are re-embedded, and
the OpenAI client's fail-loud contract. One opt-in live test."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from broker.config import EmbeddingConfig
from broker.index.embedding import EmbeddingError, OpenAIEmbedder
from broker.index.indexer import index_repo
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


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURES / "python_repo", repo)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=_GIT_ENV)
    return repo.resolve()


async def test_embeds_functions_methods_and_classes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    db = tmp_path / "index.sqlite"
    fake = FakeEmbedder()
    summary = await index_repo(repo, db, fake)
    names = fake.embedded_names()
    assert "app/chat.py::ChatService" in names
    assert "app/chat.py::ChatService.send" in names
    assert "app/providers/factory.py::create_provider" in names
    assert summary.embedded == len(names)
    store = IndexStore.open(db)
    try:
        assert {q for q, _ in store.load_vectors()} == names
    finally:
        store.close()


async def test_unchanged_run_embeds_nothing(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    db = tmp_path / "index.sqlite"
    await index_repo(repo, db, FakeEmbedder())
    fake = FakeEmbedder()
    summary = await index_repo(repo, db, fake)
    assert fake.batches == []
    assert summary.embedded == 0


async def test_only_symbols_whose_text_changed_are_reembedded(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    db = tmp_path / "index.sqlite"
    await index_repo(repo, db, FakeEmbedder())
    target = repo / "app" / "providers" / "anthropic.py"
    target.write_text(
        target.read_text().replace("return prompt", "return prompt.strip()")
    )
    fake = FakeEmbedder()
    await index_repo(repo, db, fake)
    # only `_call`'s body changed; the class text (fields + method signatures) did not
    assert fake.embedded_names() == {
        "app/providers/anthropic.py::AnthropicProvider._call"
    }


async def test_model_swap_reembeds_everything_even_when_text_is_unchanged(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    db = tmp_path / "index.sqlite"
    first = FakeEmbedder(model_id="model-a")
    await index_repo(repo, db, first)
    names = first.embedded_names()
    second = FakeEmbedder(model_id="model-b")
    summary = await index_repo(repo, db, second)
    assert second.embedded_names() == names
    assert summary.embedded == len(names)


async def test_vanished_symbols_lose_their_embeddings(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    db = tmp_path / "index.sqlite"
    await index_repo(repo, db, FakeEmbedder())
    (repo / "app" / "settings.py").unlink()
    await index_repo(repo, db, FakeEmbedder())
    store = IndexStore.open(db)
    try:
        assert not any(
            q.startswith("app/settings.py::") for q, _ in store.load_vectors()
        )
    finally:
        store.close()


def test_missing_key_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        OpenAIEmbedder.from_env(EmbeddingConfig())


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
async def test_live_openai_embedding_roundtrip() -> None:
    vectors = await OpenAIEmbedder.from_env(EmbeddingConfig()).embed(
        ["def a(): pass", "class B: ..."], timeout_s=30.0
    )
    assert len(vectors) == 2
    assert all(vectors)
