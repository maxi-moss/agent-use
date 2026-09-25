"""Registry: round-trip, reuse-proof name allocation, removal, fail-loud."""

from pathlib import Path

import pytest

from broker.master.registry import Registry, RegistryError, SessionRecord


def record(name: str, **overrides: object) -> SessionRecord:
    base: dict[str, object] = {
        "name": name,
        "socket_path": f"/private/tmp/b/s/{name}.sock",
        "cwd": "/private/tmp/work",
        "anchor_pane": "w3:p1",
    }
    base.update(overrides)
    return SessionRecord.model_validate(base)


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = Registry.load(path)
    assert registry.records == {}
    registry.upsert(record("s1", intent="do things", budget_count=3))
    reloaded = Registry.load(path)
    assert reloaded.records.keys() == {"s1"}
    assert reloaded.get("s1").budget_count == 3
    assert reloaded.get("s1").intent == "do things"
    assert reloaded.get("s1").socket_path == "/private/tmp/b/s/s1.sock"


def test_allocate_name_unique_and_valid(tmp_path: Path) -> None:
    registry = Registry.load(tmp_path / "registry.json")
    names: set[str] = set()
    for _ in range(5):
        name = registry.allocate_name()
        assert name not in names
        names.add(name)
        registry.upsert(record(name))
    assert names == {"s1", "s2", "s3", "s4", "s5"}


def test_budget_updates_persist(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = Registry.load(path)
    registry.upsert(record("s1"))
    rec = registry.get("s1")
    rec.budget_count = 7
    registry.save()
    assert Registry.load(path).get("s1").budget_count == 7
    rec.budget_count = 0
    registry.save()
    assert Registry.load(path).get("s1").budget_count == 0


def test_remove_drops_session_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = Registry.load(path)
    registry.upsert(record("s1"))
    registry.upsert(record("s2"))
    registry.remove("s1")
    assert registry.records.keys() == {"s2"}
    # Persisted, so a reload never resurrects the finished session as a
    # routing candidate.
    assert Registry.load(path).records.keys() == {"s2"}
    registry.remove("s1")  # already gone: a no-op, not an error


def test_allocate_name_never_reuses_a_removed_name(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = Registry.load(path)
    first = registry.allocate_name()
    registry.upsert(record(first))
    registry.remove(first)
    assert first not in registry.records
    # A freed number would collide with the on-disk logs, socket and transcript
    # still keyed to the finished session, so the counter never rewinds — even
    # across a reload that only sees the survivors.
    assert registry.allocate_name() != first
    assert Registry.load(path).allocate_name() != first


def test_unknown_session_raises_key_error(tmp_path: Path) -> None:
    registry = Registry.load(tmp_path / "registry.json")
    with pytest.raises(KeyError):
        registry.get("nope")


def test_corrupt_registry_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text("{broken")
    with pytest.raises(RegistryError):
        Registry.load(path)


def test_non_integer_name_seq_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text('{"sessions": {}, "name_seq": "oops"}')
    with pytest.raises(RegistryError):
        Registry.load(path)
