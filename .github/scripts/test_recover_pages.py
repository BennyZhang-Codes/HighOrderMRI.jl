"""Metadata gates and safe extraction regressions; no network or package execution."""

import copy
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import recover_pages as recovery


REPOSITORY = "owner/repo"
SHA = "a" * 40
CURRENT_SHA = "b" * 40
RUN = {"id": 123, "repository": {"full_name": REPOSITORY, "id": 7},
       "head_repository": {"full_name": REPOSITORY, "id": 7}, "head_branch": "docs",
       "head_sha": SHA, "path": recovery.WORKFLOW, "event": "push", "status": "completed",
       "conclusion": "failure"}  # Deployment failure must not invalidate a checked build.
JOB = {"name": recovery.BUILD_JOB, "run_id": 123, "head_sha": SHA,
       "status": "completed", "conclusion": "success"}
ARTIFACT = {"id": 456, "name": "github-pages", "expired": False, "size_in_bytes": 1024,
            "workflow_run": {"id": 123, "head_sha": SHA, "head_branch": "docs",
                             "repository_id": 7, "head_repository_id": 7}}


def fixture(extra=(), missing=(), code_sha=SHA, zip_name="artifact.tar"):
    tar = io.BytesIO()
    pages = [("index.html", "redirect"), ("dev/index.html", "home"),
             ("dev/reference/highorderlowrankop.html", f"source {code_sha}")]
    with tarfile.open(fileobj=tar, mode="w") as archive:
        for name, content in pages:
            if name in missing:
                continue
            data = content.encode()
            member = tarfile.TarInfo("./" + name)
            member.size, member.mode = len(data), 0o777
            archive.addfile(member, io.BytesIO(data))
        for member in extra:
            archive.addfile(member, io.BytesIO(b"x" * member.size))
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr(zip_name, tar.getvalue())
    return zip_buffer.getvalue()


class SourceTests(unittest.TestCase):
    def test_completed_run_with_failed_deployment_is_allowed(self):
        recovery.validate_source(RUN, REPOSITORY, 123, "tree", "tree", CURRENT_SHA)
        recovery.validate_build([JOB], RUN)
        self.assertEqual(recovery.select_artifact([ARTIFACT], RUN)["id"], 456)

    def test_wrong_source_metadata_is_rejected(self):
        mutations = {"id": 124, "head_branch": "main", "path": "other.yml",
                     "event": "pull_request", "status": "in_progress", "head_sha": "bad",
                     "repository": {"full_name": "other/repo"},
                     "head_repository": {"full_name": "fork/repo"}}
        for key, value in mutations.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.validate_source(dict(RUN, **{key: value}), REPOSITORY, 123, "tree", "tree", CURRENT_SHA)

    def test_changed_docs_tree_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "docs tree"):
            recovery.validate_source(RUN, REPOSITORY, 123, "old", "new", CURRENT_SHA)

    def test_same_commit_cannot_reuse_source_deployment_id(self):
        with self.assertRaisesRegex(ValueError, "new commit SHA"):
            recovery.validate_source(RUN, REPOSITORY, 123, "tree", "tree", SHA)

    def test_failed_or_wrong_sha_build_is_rejected(self):
        for key, value in {"conclusion": "failure", "head_sha": "b" * 40,
                           "run_id": 999, "name": "Different job", "status": "in_progress"}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.validate_build([dict(JOB, **{key: value})], RUN)

    def test_expired_missing_or_ambiguous_artifacts_are_rejected(self):
        for artifacts in ([], [dict(ARTIFACT, expired=True)], [ARTIFACT, ARTIFACT]):
            with self.subTest(artifacts=artifacts), self.assertRaises(ValueError):
                recovery.select_artifact(artifacts, RUN)

    def test_artifact_identity_is_checked(self):
        for key, value in {"id": 999, "head_sha": "b" * 40, "head_branch": "main",
                           "repository_id": 9, "head_repository_id": 9}.items():
            artifact = copy.deepcopy(ARTIFACT)
            artifact["workflow_run"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.select_artifact([artifact], RUN)

    def test_jobs_pagination_keeps_filter_all(self):
        calls = []
        def fake_gh(endpoint):
            import json
            calls.append(endpoint)
            return json.dumps({"jobs": [JOB] * (100 if len(calls) == 1 else 1)}).encode()
        with patch.object(recovery, "gh", fake_gh):
            self.assertEqual(len(recovery.paginated("runs/123/jobs?filter=all", "jobs")), 101)
        self.assertEqual(calls, ["runs/123/jobs?filter=all&per_page=100&page=1",
                                 "runs/123/jobs?filter=all&per_page=100&page=2"])


class ArchiveTests(unittest.TestCase):
    def reject(self, blob):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "site"
            with self.assertRaises((ValueError, tarfile.TarError, zipfile.BadZipFile)):
                recovery.extract_archive(blob, target, SHA)
            self.assertFalse(target.exists(), "Validation must precede any extraction")

    def test_valid_site_and_normalized_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "site"
            self.assertEqual(recovery.extract_archive(fixture(), target, SHA), 3)
            self.assertEqual((target / "index.html").stat().st_mode & 0o777, 0o644)
            self.assertEqual(target.stat().st_mode & 0o777, 0o755)

    def test_traversal_and_absolute_paths_are_rejected(self):
        for name in ("../escape", "/escape", "dev/../../escape", "dev\\escape", "C:/escape"):
            with self.subTest(name=name):
                self.reject(fixture(extra=[tarfile.TarInfo(name)]))

    def test_links_and_special_files_are_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE):
            member = tarfile.TarInfo("dev/extra")
            member.type, member.linkname = kind, "../escape"
            with self.subTest(kind=kind):
                self.reject(fixture(extra=[member]))

    def test_duplicate_case_and_parent_collisions_are_rejected(self):
        for name in ("index.html", "./index.html", "INDEX.HTML", "dev"):
            with self.subTest(name=name):
                self.reject(fixture(extra=[tarfile.TarInfo(name)]))

    def test_missing_page_wrong_code_or_extra_zip_entry_is_rejected(self):
        self.reject(fixture(missing=["index.html"]))
        self.reject(fixture(code_sha="b" * 40))
        self.reject(fixture(zip_name="unexpected.tar"))
        blob = io.BytesIO(fixture())
        with zipfile.ZipFile(blob, "a") as archive:
            archive.writestr("extra", "not allowed")
        self.reject(blob.getvalue())

    def test_output_must_be_empty_and_not_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "site"
            target.mkdir()
            (target / "existing").write_text("preserve")
            with self.assertRaises(ValueError):
                recovery.extract_archive(fixture(), target, SHA)
            self.assertEqual((target / "existing").read_text(), "preserve")
            link = Path(directory) / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                recovery.extract_archive(fixture(), link, SHA)


if __name__ == "__main__":
    unittest.main()
