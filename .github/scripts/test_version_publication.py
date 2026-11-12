import json
import unittest
from check_live_pages import CheckError, Response, check_version_index


class PublishedVersionTests(unittest.TestCase):
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
