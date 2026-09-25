"""scenario_names: fails loud on a missing or empty directory."""

from pathlib import Path

import pytest

from broker.master.testmode.runner import scenario_names
from broker.master.testmode.schemas import ScenarioError


def test_scenario_names_raises_on_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError):
        scenario_names(tmp_path / "does-not-exist")


def test_scenario_names_raises_on_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError):
        scenario_names(tmp_path)


def test_scenario_names_lists_sorted_stems(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    assert scenario_names(tmp_path) == ["a", "b"]
