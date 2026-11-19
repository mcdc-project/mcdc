"""Bundle the source commit and its timestamp for installs without Git history."""

from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class RevisionBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Editable installs read Git directly. Source distributions already
        # carry the snapshot into wheels built from them.
        snapshot = "mcdc/_git_revision.txt"
        if version == "editable" or (Path(self.root) / snapshot).is_file():
            return
        helper = run_path(str(Path(self.root) / "mcdc/_revision.py"))
        commit, commit_time = helper["git_revision"]()
        self._scratch = TemporaryDirectory()
        path = Path(self._scratch.name) / "_git_revision.txt"
        path.write_text(f"{commit}\n{commit_time}\n", encoding="utf-8")
        build_data["force_include"][str(path)] = snapshot
