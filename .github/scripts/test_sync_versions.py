"""Version synchronization regressions using a local bare remote."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import sync_versions as sync


def release(version, **changes):
    return dict(tag_name=version, draft=False, prerelease=False, published_at="2026-11-12T00:00:00Z", **changes)


class ContextTests(unittest.TestCase):
    def test_only_trusted_branches_and_formal_releases(self):
        for branch in ("main", "dev_jinyuan"):
            for event in ("push", "workflow_dispatch"):
                sync.validate_context(sync.REPOSITORY, event, "refs/heads/" + branch, {})
        sync.validate_context(sync.REPOSITORY, "release", "refs/tags/v0.2.0",
                              dict(action="published", release=release("v0.2.0")))
        for repo, event, ref, payload in (
            ("other/repo", "push", "refs/heads/main", {}),
            (sync.REPOSITORY, "pull_request", "refs/heads/main", {}),
            (sync.REPOSITORY, "workflow_dispatch", "refs/heads/docs", {}),
            (sync.REPOSITORY, "push", "refs/heads/gh-pages", {}),
            (sync.REPOSITORY, "release", "refs/tags/v0.2.0", dict(action="edited", release=release("v0.2.0"))),
            (sync.REPOSITORY, "release", "refs/tags/v0.2.0", dict(action="published", release={**release("v0.2.0"), "prerelease": True})),
        ):
            with self.subTest(event=event, ref=ref), self.assertRaises(ValueError):
                sync.validate_context(repo, event, ref, payload)

    def test_dispatch_runs_even_when_registry_already_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            event = Path(temporary) / "event.json"
            event.write_text("{}")
            env = dict(GITHUB_EVENT_PATH=str(event), GITHUB_REPOSITORY=sync.REPOSITORY,
                       GITHUB_EVENT_NAME="push", GITHUB_REF="refs/heads/main")
            with patch.dict(os.environ, env), patch.object(sync, "synchronize", return_value={"updated": False}), \
                 patch.object(sync.subprocess, "check_output", return_value="[[]]"), patch.object(sync.subprocess, "run") as run:
                sync.main()
            self.assertEqual(run.call_args.args[0], ["gh", "workflow", "run", "documentation.yml", "--repo", sync.REPOSITORY, "--ref", "docs"])


class GitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.repo, self.remote = root / "source", root / "remote.git"
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True, capture_output=True)
        subprocess.run(["git", "init", "-b", "docs", str(self.repo)], check=True, capture_output=True)
        self.git("config", "user.name", "test")
        self.git("config", "user.email", "test@example.invalid")
        (self.repo / "docs/src").mkdir(parents=True)
        (self.repo / "docs/src/index.md").write_text("preserve handwritten source")
        (self.repo / "docs/api-map.toml").write_text("schema_version = 1")
        (self.repo / "Project.toml").write_text('version = "0.1.0"\n')
        self.git("add", ".")
        self.git("commit", "-m", "code and source")
        self.code = self.git("rev-parse", "HEAD")
        self.git("tag", "v0.1.0")
        for branch in ("main", "dev_jinyuan"):
            self.git("branch", branch)
        self.original = dict(version="v0.1.0", code_sha=self.code, docs_sha=self.code)
        (self.repo / "docs/code-version.json").write_text(json.dumps(dict(schema_version=1, code_sha=self.code)))
        (self.repo / "docs/versions.json").write_text(json.dumps(dict(schema_version=1, stable="v0.1.0", releases=[self.original])))
        self.git("add", "docs")
        self.git("commit", "-m", "registered release")
        self.source = self.git("rev-parse", "HEAD")
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "origin", "docs", "main", "dev_jinyuan", "--tags")

    def git(self, *args):
        return sync.git(self.repo, *args)

    def saved(self, commit, path):
        return json.loads(self.git("show", commit + ":docs/" + path))

    def test_sync_changes_only_metadata_and_keeps_source_checkout_clean(self):
        result = sync.synchronize(self.repo, [release("v0.1.0")])
        self.assertTrue(result["updated"])
        self.assertEqual(self.git("show", "-s", "--format=%an|%ae|%cn|%ce", result["docs_sha"]),
                         "|".join([sync.COMMIT_NAME, sync.COMMIT_EMAIL] * 2))
        self.assertEqual(self.git("rev-parse", "HEAD"), self.source)
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertEqual(self.git("diff", "--name-only", self.source, result["docs_sha"]).splitlines(),
                         ["docs/code-version.json", "docs/versions.json"])
        self.assertEqual(self.saved(result["docs_sha"], "code-version.json")["main_sha"], self.code)
        self.assertEqual(self.saved(result["docs_sha"], "versions.json")["releases"], [self.original])
        again = sync.synchronize(self.repo, [release("v0.1.0")])
        self.assertFalse(again["updated"])
        self.assertEqual(again["docs_sha"], result["docs_sha"])

    def new_release(self):
        self.git("checkout", "main")
        (self.repo / "Project.toml").write_text('version = "0.2.0"\n')
        self.git("add", "Project.toml")
        self.git("commit", "-m", "next release")
        commit = self.git("rev-parse", "HEAD")
        self.git("tag", "-a", "v0.2.0", "-m", "annotated release")
        self.git("push", "origin", "main", "--tags")
        self.git("checkout", "docs")
        return commit

    def test_new_published_release_freezes_prose_and_advances_stable(self):
        commit = self.new_release()
        published = [release("v0.1.0"), release("v0.2.0")]
        first = sync.synchronize(self.repo, published)
        config = self.saved(first["docs_sha"], "versions.json")
        self.assertEqual(config["stable"], "v0.2.0")
        self.assertEqual(config["releases"], [dict(version="v0.2.0", code_sha=commit, docs_sha=self.source), self.original])
        second = sync.synchronize(self.repo, published)
        self.assertEqual(self.saved(second["docs_sha"], "versions.json"), config)

    def test_unpublished_and_prerelease_tags_do_not_advance_stable(self):
        self.new_release()
        result = sync.synchronize(self.repo, [{**release("v0.2.0"), "prerelease": True}, {**release("v0.3.0"), "draft": True}])
        self.assertEqual(result["stable"], "v0.1.0")
        self.assertEqual(self.saved(result["docs_sha"], "versions.json")["releases"], [self.original])

    def test_moved_release_tag_and_dirty_source_fail_without_pushing(self):
        self.new_release()
        self.git("update-ref", "refs/tags/v0.1.0", self.git("rev-parse", "main"))
        with self.assertRaisesRegex(ValueError, "identity"):
            sync.resolve_versions(self.repo, self.source, self.code, self.git("rev-parse", "main"), [release("v0.1.0")])
        (self.repo / "docs/src/index.md").write_text("uncommitted source")
        with self.assertRaisesRegex(ValueError, "clean"):
            sync.synchronize(self.repo, [])
        self.assertEqual(sync.remote_ref(self.repo, sync.SOURCE_REF), self.source)

    def test_concurrent_source_commit_is_retained_on_retry(self):
        real_git = sync.git
        concurrent = self.git("commit-tree", self.git("rev-parse", "HEAD^{tree}"), "-p", self.source, "-m", "concurrent source")
        raced = False
        def git(repo, *args, **kwargs):
            nonlocal raced
            if args[:2] == ("push", "origin") and not raced:
                raced = True
                real_git(repo, "push", "origin", concurrent + ":refs/heads/docs")
            return real_git(repo, *args, **kwargs)
        with patch.object(sync, "git", side_effect=git):
            result = sync.synchronize(self.repo, [release("v0.1.0")])
        self.assertEqual(self.git("rev-parse", result["docs_sha"] + "^"), concurrent)


if __name__ == "__main__":
    unittest.main()
