import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from check_live_pages import CheckError, Response, check_version_index, expected_versions, main


class PublishedVersionTests(unittest.TestCase):
    def test_live_cli_uses_the_checked_runtime_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "versions.json"
            record = dict(version="main", code_sha="d" * 40, docs_sha="e" * 40, preview=False)
            manifest = dict(schema_version=1, stable="v0.1.0", preview=False, versions=[record])
            path.write_text(json.dumps(manifest))
            response = Response(200, "https://example.test/versions.json", "application/json", path.read_bytes())
            with patch("check_live_pages.verify_site", return_value=True) as verify, \
                 patch("check_live_pages.fetch_response", return_value=response):
                self.assertEqual(main(["--pages-url", "https://example.test/", "--manifest", str(path)]), 0)
            self.assertEqual(verify.call_args.args[1], "d" * 40)
            self.assertEqual(verify.call_args.kwargs["prefix"], "main")

    def test_main_has_independent_live_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "code-version.json"
            path.write_text(json.dumps(dict(schema_version=1, code_sha="a" * 40, main_sha="d" * 40)))
            (root / "versions.json").write_text(json.dumps(dict(stable="v0.1.0", releases=[dict(version="v0.1.0", code_sha="b" * 40, docs_sha="c" * 40)])))
            stable, rows = expected_versions(path)
            self.assertEqual(stable, "v0.1.0")
            self.assertEqual(rows[1], dict(version="main", code_sha="d" * 40, docs_sha=None))
            path.write_text(json.dumps(dict(schema_version=1, code_sha="a" * 40, main_sha="invalid")))
            with self.assertRaises(CheckError):
                expected_versions(path)

    def setUp(self):
        self.records = [{"version": "dev", "code_sha": "a" * 40, "docs_sha": None},
                        {"version": "v0.1.0", "code_sha": "b" * 40, "docs_sha": "c" * 40}]

    def response(self, **changes):
        payload = {"schema_version": 1, "stable": "v0.1.0", "preview": False, "versions": self.records, **changes}
        payload["versions"] = [{**record, "preview": payload["preview"]} for record in payload["versions"]]
        return Response(200, "https://example.test/versions.json", "application/json", json.dumps(payload).encode())

    def test_complete_version_set(self):
        check_version_index(self.response(), "v0.1.0", self.records)

    def test_stale_missing_or_wrong_version_is_rejected(self):
        for changes in ({"stable": "v0.0.9"}, {"versions": self.records[:1]},
                        {"versions": [self.records[0], {**self.records[1], "code_sha": "d" * 40}]}):
            with self.subTest(changes=changes), self.assertRaises(CheckError):
                check_version_index(self.response(**changes), "v0.1.0", self.records)

    def test_candidate_cannot_pass_formal_check(self):
        with self.assertRaises(CheckError):
            check_version_index(self.response(preview=True), "v0.1.0", self.records)
        check_version_index(self.response(preview=True), "v0.1.0", self.records, allow_preview=True)
