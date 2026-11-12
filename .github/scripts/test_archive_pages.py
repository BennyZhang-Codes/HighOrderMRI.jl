"""Publication policy and real local Git regressions; no GitHub access."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import archive_pages as archive


CODE = {"schema_version": 1, "code_sha": "a" * 40}
RELEASES = {"schema_version": 1, "stable": "v0.1.0", "releases": [
    {"version": "v0.1.0", "code_sha": "b" * 40, "docs_sha": "c" * 40}]}


def site_fixture(root):
    rows = [dict(version="dev", code_sha=CODE["code_sha"], docs_sha=None, preview=False),
            dict(RELEASES["releases"][0], preview=False),
            dict(RELEASES["releases"][0], version="stable", preview=False, release="v0.1.0")]
    manifest = dict(schema_version=1, stable="v0.1.0", preview=False, versions=rows)
    root.mkdir()
    (root / "index.html").write_text("stable redirect")
    (root / "versions.json").write_text(json.dumps(manifest))
    for row in rows:
        folder = root / row["version"]
        folder.mkdir()
        (folder / "index.html").write_text(row["version"])
        (folder / "version.json").write_text(json.dumps(row))
    return manifest


class PolicyTests(unittest.TestCase):
    def test_main_is_required_and_bound_to_its_own_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site"
            manifest = site_fixture(site)
            code = dict(CODE, main_sha="d" * 40)
            with self.assertRaisesRegex(ValueError, "every configured"):
                archive.validate_site(site, code, RELEASES)
            row = dict(version="main", code_sha=code["main_sha"], docs_sha=None, preview=False)
            manifest["versions"].append(row)
            (site / "main").mkdir()
            (site / "main/index.html").write_text("main")
            (site / "main/version.json").write_text(json.dumps(row))
            (site / "versions.json").write_text(json.dumps(manifest))
            self.assertEqual(archive.validate_site(site, code, RELEASES), manifest)
            with self.assertRaisesRegex(ValueError, "source"):
                archive.validate_site(site, dict(code, main_sha="e" * 40), RELEASES)

    def test_only_trusted_source_contexts_publish(self):
        for event in ("push", "workflow_dispatch"):
            archive.validate_context(archive.REPOSITORY, event, archive.SOURCE_REF)
        for repo, event, ref in (("other/repo", "push", archive.SOURCE_REF),
                                 (archive.REPOSITORY, "pull_request", archive.SOURCE_REF),
                                 (archive.REPOSITORY, "push", "refs/heads/main"),
                                 (archive.REPOSITORY, "push", archive.ARCHIVE_REF)):
            with self.subTest(event=event, ref=ref), self.assertRaises(ValueError):
                archive.validate_context(repo, event, ref)

    def test_only_complete_formal_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site"
            manifest = site_fixture(site)
            self.assertEqual(archive.validate_site(site, CODE, RELEASES), manifest)
            for key, value in (("preview", True), ("stable", "v0.2.0"), ("versions", manifest["versions"][:-1])):
                (site / "versions.json").write_text(json.dumps(dict(manifest, **{key: value})))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    archive.validate_site(site, CODE, RELEASES)
            (site / "versions.json").write_text(json.dumps(manifest))
            (site / "dev/version.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "metadata"):
                archive.validate_site(site, CODE, RELEASES)

    def test_symlinks_and_repository_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site"
            site_fixture(site)
            link = site / "external"
            link.symlink_to(Path(temporary))
            with self.assertRaisesRegex(ValueError, "links"):
                archive.validate_site(site, CODE, RELEASES)
            link.unlink()
            (site / "README.md").write_text("source")
            with self.assertRaisesRegex(ValueError, "repository source"):
                archive.validate_site(site, CODE, RELEASES)

    def test_published_releases_cannot_disappear_or_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            old = site_fixture(Path(temporary) / "site")
            archive.preserve_releases(old, old)
            removed = copy.deepcopy(old)
            removed["versions"] = [r for r in old["versions"] if r["version"] != "v0.1.0"]
            with self.assertRaisesRegex(ValueError, "removed"):
                archive.preserve_releases(old, removed)
            for key in ("code_sha", "docs_sha"):
                moved = copy.deepcopy(old)
                next(r for r in moved["versions"] if r["version"] == "v0.1.0")[key] = "d" * 40
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity"):
                    archive.preserve_releases(old, moved)
            downgraded = dict(old, stable="v0.0.9")
            with self.assertRaisesRegex(ValueError, "backwards"):
                archive.preserve_releases(old, downgraded)


class GitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.remote, self.repo, self.site = root / "remote.git", root / "source", root / "site"
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True, capture_output=True)
        subprocess.run(["git", "init", "-b", "docs", str(self.repo)], check=True, capture_output=True)
        archive.git(self.repo, "config", "user.name", "test")
        archive.git(self.repo, "config", "user.email", "test@example.invalid")
        (self.repo / "source.md").write_text("handwritten documentation")
        archive.git(self.repo, "add", "source.md")
        archive.git(self.repo, "commit", "-m", "source")
        archive.git(self.repo, "remote", "add", "origin", str(self.remote))
        archive.git(self.repo, "push", "origin", "docs")
        self.source = archive.git(self.repo, "rev-parse", "HEAD")
        self.manifest = site_fixture(self.site)

    def test_first_archive_incremental_update_and_same_site(self):
        first = archive.publish(self.repo, self.site, self.source, self.manifest)
        self.assertTrue(first["publish"] and first["updated"])
        files = archive.git(self.repo, "ls-tree", "--name-only", first["sha"]).splitlines()
        self.assertEqual(set(files), {".nojekyll", "archive-info.json", "index.html", "versions.json", "dev", "stable", "v0.1.0"})
        self.assertNotIn("source.md", files)
        second = archive.publish(self.repo, self.site, self.source, self.manifest)
        self.assertFalse(second["updated"])
        self.assertEqual(second["sha"], first["sha"])
        (self.site / "dev/index.html").write_text("updated generated page")
        third = archive.publish(self.repo, self.site, self.source, self.manifest)
        self.assertTrue(third["updated"])
        self.assertEqual(archive.git(self.repo, "rev-parse", third["sha"] + "^"), first["sha"])
        self.assertEqual(archive.git(self.repo, "rev-parse", "HEAD"), self.source)
        self.assertEqual(archive.git(self.repo, "status", "--porcelain"), "")
        self.assertEqual((self.repo / "source.md").read_text(), "handwritten documentation")

    def test_stale_source_does_not_create_archive(self):
        archive.git(self.repo, "commit", "--allow-empty", "-m", "newer source")
        archive.git(self.repo, "push", "origin", "docs")
        archive.git(self.repo, "checkout", "--detach", self.source)
        result = archive.publish(self.repo, self.site, self.source, self.manifest)
        self.assertEqual(result, {"publish": False, "reason": "superseded_source"})
        self.assertIsNone(archive.remote_ref(self.repo, archive.ARCHIVE_REF))

    def test_release_identity_failure_preserves_remote(self):
        first = archive.publish(self.repo, self.site, self.source, self.manifest)
        changed = copy.deepcopy(self.manifest)
        next(r for r in changed["versions"] if r["version"] == "v0.1.0")["code_sha"] = "d" * 40
        with self.assertRaisesRegex(ValueError, "identity"):
            archive.publish(self.repo, self.site, self.source, changed)
        self.assertEqual(archive.remote_ref(self.repo, archive.ARCHIVE_REF), first["sha"])

    def test_wrong_or_dirty_source_cannot_publish(self):
        with self.assertRaisesRegex(ValueError, "workflow commit"):
            archive.publish(self.repo, self.site, "d" * 40, self.manifest)
        (self.repo / "source.md").write_text("uncommitted")
        with self.assertRaisesRegex(ValueError, "clean"):
            archive.publish(self.repo, self.site, self.source, self.manifest)
        self.assertIsNone(archive.remote_ref(self.repo, archive.ARCHIVE_REF))


if __name__ == "__main__":
    unittest.main()
