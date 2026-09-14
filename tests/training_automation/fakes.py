from __future__ import annotations

import hashlib
from pathlib import Path


class FakeHubClient:
    def __init__(self, *, private: bool = True):
        self.private = private
        self.revision_number = 0
        self.revision = "r0"
        self.remote: dict[str, bytes] = {}
        self.snapshots = {self.revision: {}}
        self.fail_backup_commits = 0
        self.catalog_conflicts = 0
        self.corrupt_metadata = False
        self.messages: list[str] = []

    def repo_is_private(self, repo_id, repo_type):
        return self.private

    def commit_files(self, repo_id, repo_type, artifacts, message, parent_commit=None):
        if parent_commit is not None and parent_commit != self.revision:
            raise RuntimeError("parent conflict")
        if message.startswith("Backup") and self.fail_backup_commits:
            self.fail_backup_commits -= 1
            raise RuntimeError("transient upload interruption")
        if (message.startswith("Reserve catalog") or message.startswith("Index") or message.startswith("Select")) and self.catalog_conflicts:
            self.catalog_conflicts -= 1
            self.revision_number += 1
            self.revision = f"r{self.revision_number}"
            self.snapshots[self.revision] = dict(self.remote)
            raise RuntimeError("optimistic conflict")
        updated = dict(self.remote)
        for artifact in artifacts:
            updated[artifact.remote_path] = Path(artifact.local_path).read_bytes()
        self.remote = updated
        self.revision_number += 1
        self.revision = f"r{self.revision_number}"
        self.snapshots[self.revision] = dict(updated)
        self.messages.append(message)
        return self.revision

    def path_metadata(self, repo_id, repo_type, paths, revision):
        snapshot = self.snapshots[revision]
        result = {}
        for path in paths:
            payload = snapshot[path]
            result[path] = {
                "size": len(payload) + (1 if self.corrupt_metadata else 0),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        return result

    def read_remote_file(self, repo_id, repo_type, path):
        return self.remote.get(path), self.revision

    def download_file(self, repo_id, repo_type, path, revision, destination):
        destination.write_bytes(self.snapshots[revision][path])

