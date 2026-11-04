"""Regression data stay pinned without overwriting local data edits."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def data_checkout(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "regression" / "conftest.py"
    spec = importlib.util.spec_from_file_location("regression_config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    source = tmp_path / "source"
    source.mkdir()

    def git(*args, cwd=source):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Regression Data Test")
    git("config", "user.email", "test@example.com")
    (source / "data.txt").write_text("legacy\n")
    git("add", "data.txt")
    git("commit", "-m", "Legacy data")
    legacy = git("rev-parse", "HEAD")
    git("switch", "-c", "mcdc-data")
    (source / "data.txt").write_text("migration\n")
    git("commit", "-am", "Migration data")
    revision = git("rev-parse", "HEAD")
    git("switch", "main")

    regression = tmp_path / "regression"
    regression.mkdir()
    monkeypatch.setattr(config, "REGRESSION_DIR", regression)
    monkeypatch.setattr(config, "REGRESSION_DATA_REPOSITORY", str(source))
    monkeypatch.setattr(config, "REGRESSION_DATA_REVISION", revision)
    target = regression / config.REGRESSION_DATA_NAME
    return config, source, target, legacy, revision, git


def test_new_clone_uses_pinned_revision_instead_of_default_branch(data_checkout):
    config, _, target, _, revision, git = data_checkout
    assert config.regression_data.__wrapped__() == target
    assert git("rev-parse", "HEAD", cwd=target) == revision
    assert (target / "data.txt").read_text() == "migration\n"


def test_existing_legacy_clone_switches_to_pinned_data_and_keeps_untracked_files(
    data_checkout,
):
    config, source, target, legacy, revision, git = data_checkout
    git("clone", str(source), str(target))
    (target / "local.txt").write_text("local data\n")
    config.regression_data.__wrapped__()
    assert git("rev-parse", "HEAD", cwd=target) == revision
    assert git("rev-parse", "main", cwd=target) == legacy
    assert (target / "local.txt").read_text() == "local data\n"


def test_cached_pinned_revision_is_available_offline(data_checkout, monkeypatch):
    config, _, target, _, revision, git = data_checkout
    config.regression_data.__wrapped__()
    monkeypatch.setattr(
        config, "REGRESSION_DATA_REPOSITORY", "/missing-data-repository"
    )
    assert config.regression_data.__wrapped__() == target
    assert git("rev-parse", "HEAD", cwd=target) == revision


def test_local_edits_are_preserved_when_checkout_would_overwrite_them(data_checkout):
    config, source, target, legacy, _, git = data_checkout
    git("clone", str(source), str(target))
    (target / "data.txt").write_text("local changes\n")
    with pytest.raises(
        pytest.fail.Exception, match="required regression data revision"
    ):
        config.regression_data.__wrapped__()
    assert git("rev-parse", "HEAD", cwd=target) == legacy
    assert (target / "data.txt").read_text() == "local changes\n"


def test_unavailable_revision_does_not_use_legacy_data(data_checkout, monkeypatch):
    config, source, target, legacy, _, git = data_checkout
    git("clone", str(source), str(target))
    monkeypatch.setattr(config, "REGRESSION_DATA_REVISION", "0" * 40)
    with pytest.raises(
        pytest.fail.Exception, match="required regression data revision"
    ):
        config.regression_data.__wrapped__()
    assert git("rev-parse", "HEAD", cwd=target) == legacy
