"""A one-shot CLI update must hand pending work to another pass.

The post-commit hook starts one ``repowise update`` process per commit. A
contender that finds the first process's lock records the newer commit in
``.update.pending`` and exits. The lock owner must release its pass-level lock
and run again; merely preserving the marker leaves a quiet repository stale
forever.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from click.testing import CliRunner

from repowise.cli.commands.update_cmd import command as upd_cmd
from repowise.cli.helpers import (
    clear_update_pending,
    read_update_pending,
    write_update_pending,
)
from repowise.cli.main import cli
from repowise.core.update_lock import (
    read_update_lock,
    release_update_lock,
    try_acquire_update_lock,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo_with_two_commits(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "commit", "--allow-empty", "-m", "indexed")
    indexed = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "--allow-empty", "-m", "queued")
    queued = _git(repo, "rev-parse", "HEAD")
    (repo / ".repowise").mkdir()
    return repo, indexed, queued


def test_cli_owner_runs_a_second_pass_for_pending_commit(tmp_path: Path, monkeypatch) -> None:
    repo, indexed, queued = _repo_with_two_commits(tmp_path)
    calls: list[str] = []

    def _pass(**kwargs) -> upd_cmd.UpdateOutcome:
        assert kwargs["path"] == str(repo)
        assert try_acquire_update_lock(repo, queued) is None
        calls.append("pass")
        if len(calls) == 1:
            write_update_pending(repo, queued)
        else:
            clear_update_pending(repo)
        return upd_cmd.UpdateOutcome.REGENERATED

    monkeypatch.setattr(upd_cmd, "run_update", _pass)

    result = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace"])

    release_update_lock(repo)
    assert result.exit_code == 0, result.output
    assert calls == ["pass", "pass"]
    assert read_update_pending(repo) is None
    assert indexed != queued


def test_deferred_cli_never_releases_the_real_owners_lock(tmp_path: Path, monkeypatch) -> None:
    repo, _indexed, queued = _repo_with_two_commits(tmp_path)
    assert try_acquire_update_lock(repo, "owner") is None
    calls: list[str] = []

    def _deferred(**kwargs) -> upd_cmd.UpdateOutcome:
        assert kwargs["path"] == str(repo)
        calls.append("deferred")
        write_update_pending(repo, queued)
        return upd_cmd.UpdateOutcome.DEFERRED

    monkeypatch.setattr(upd_cmd, "run_update", _deferred)

    result = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace"])
    payload = read_update_lock(repo)

    release_update_lock(repo)
    assert result.exit_code == 0, result.output
    assert calls == ["deferred"]
    assert payload is not None
    assert payload["target_commit"] == "owner"
