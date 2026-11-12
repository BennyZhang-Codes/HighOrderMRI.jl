"""Branch/tag source resolution without a dedicated documentation branch."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import archive_pages
from version_plan import git, resolve_versions


def published(version):
    return dict(tag_name=version, draft=False, prerelease=False, published_at="2026-11-12T00:00:00Z")


class VersionPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.repo, self.remote = root / "source", root / "remote.git"
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True, capture_output=True)
        subprocess.run(["git", "init", "-b", "main", str(self.repo)], check=True, capture_output=True)
        self.git("config", "user.name", "test")
        self.git("config", "user.email", "test@example.invalid")
        (self.repo / "Project.toml").write_text('version = "0.1.0"\n')
        self.git("add", ".")
        self.git("commit", "-m", "legacy code")
        self.code = self.git("rev-parse", "HEAD")
        self.git("tag", "v0.1.0")
        self.git("branch", "dev_jinyuan")
        (self.repo / "docs/src").mkdir(parents=True)
        (self.repo / "docs/src/index.md").write_text("preserve theory")
        (self.repo / "docs/api-map.toml").write_text("schema_version = 1")
        self.git("add", ".")
        self.git("commit", "-m", "saved documentation history")
        self.legacy = self.git("rev-parse", "HEAD")
        row = dict(version="v0.1.0", code_sha=self.code, docs_sha=self.legacy)
        (self.repo / "docs/versions.json").write_text(json.dumps(dict(schema_version=1, stable="v0.1.0", releases=[row])))
        self.git("add", ".")
        self.git("commit", "-m", "in-tree migration")
        self.source = self.git("rev-parse", "HEAD")
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "origin", "main", "dev_jinyuan", "--tags")
        self.override = patch.object(archive_pages, "LEGACY_DOCS_SOURCE", self.legacy)
        self.override.start()
        self.addCleanup(self.override.stop)

    def git(self, *args):
        return git(self.repo, *args)

    def plan(self, releases=None, **options):
        return resolve_versions(self.repo, releases or [published("v0.1.0")], options.pop("archive", {}),
                                options.pop("source", self.source), options.pop("event", "push"),
                                options.pop("ref", "refs/heads/main"), **options)

    def test_current_branches_and_legacy_release_have_explicit_distinct_sources(self):
        plan = self.plan()
        targets = {r["version"]: r for r in plan["targets"]}
        self.assertEqual(targets["main"], dict(version="main", code_sha=self.source, docs_sha=self.source))
        self.assertEqual(targets["dev"], dict(version="dev", code_sha=self.code, docs_sha=self.legacy))
        self.assertEqual(targets["v0.1.0"]["docs_sha"], self.legacy)
        self.assertEqual(self.git("ls-remote", "origin", "refs/heads/docs"), "")

    def test_pr_builds_the_merge_commit_for_its_base_branch(self):
        plan = self.plan(event="pull_request", ref="refs/pull/1/merge", base="dev_jinyuan")
        self.assertEqual(plan["targets"][0]["code_sha"], self.source)
        self.assertEqual(plan["targets"][0]["docs_sha"], self.source)

    def test_new_formal_release_uses_annotated_tag_source_and_advances_stable(self):
        (self.repo / "Project.toml").write_text('version = "0.2.0"\n')
        self.git("add", ".")
        self.git("commit", "-m", "release with docs")
        code = self.git("rev-parse", "HEAD")
        self.git("tag", "-a", "v0.2.0", "-m", "annotated release")
        self.git("push", "origin", "main", "--tags")
        plan = self.plan([published("v0.1.0"), published("v0.2.0")], source=code, event="release", ref="refs/tags/v0.2.0")
        self.assertEqual(plan["stable"], "v0.2.0")
        self.assertEqual(plan["releases"][0], dict(version="v0.2.0", code_sha=code, docs_sha=code))
        self.assertEqual(plan["releases"][1]["docs_sha"], self.legacy)
        draft = {**published("v0.3.0"), "draft": True}
        pre = {**published("v0.2.0"), "prerelease": True}
        self.assertEqual(self.plan([published("v0.1.0"), pre, draft])["stable"], "v0.1.0")

    def test_manual_backfill_needs_a_real_published_release(self):
        self.assertEqual(self.plan(event="workflow_dispatch", release_tag="v0.1.0")["stable"], "v0.1.0")
        for tag in ("v9.9.9", "../outside"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.plan(event="workflow_dispatch", release_tag=tag)

    def test_missing_tag_docs_moved_tags_and_changed_frozen_source_fail(self):
        self.git("tag", "v0.3.0", self.code)
        with self.assertRaisesRegex(ValueError, "no in-tree"):
            self.plan([published("v0.1.0"), published("v0.3.0")])
        archived = dict(versions=[dict(version="v0.1.0", code_sha=self.code, docs_sha=self.source)])
        with self.assertRaisesRegex(ValueError, "Frozen"):
            self.plan(archive=archived)
        self.git("update-ref", "refs/tags/v0.1.0", self.source)
        with self.assertRaisesRegex(ValueError, "cannot move"):
            self.plan()


if __name__ == "__main__":
    unittest.main()
