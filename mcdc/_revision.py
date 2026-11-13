"""Read the MC/DC Git revision for a run or a package build."""

from pathlib import Path
import subprocess


def git_revision() -> tuple[str, str]:
    """Return the short commit hash and the committer timestamp with time zone."""
    package = Path(__file__).resolve().parent
    git_dir = package.parent / ".git"
    if git_dir.exists():
        try:
            revision = subprocess.check_output(
                [
                    "git",
                    f"--git-dir={git_dir}",
                    "log",
                    "-1",
                    "--no-show-signature",
                    "--format=%H%n%cI",
                    "HEAD",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown", "unknown"
        commit, commit_time = revision.splitlines()
        return commit[:12], commit_time

    snapshot = package / "_git_revision.txt"
    if snapshot.is_file():
        commit, commit_time = snapshot.read_text(encoding="utf-8").splitlines()
        return commit, commit_time
    return "unknown", "unknown"
