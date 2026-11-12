import unittest
from verify_releases import validate_release_catalog, validate_release_metadata


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"tag_name": "v0.1.0", "draft": False, "prerelease": False, "published_at": "2026-11-12T00:00:00Z"}

    def test_published_formal_release(self):
        validate_release_metadata(self.metadata, "v0.1.0")

    def test_catalog_retains_all_published_releases_but_not_drafts_or_prereleases(self):
        config = {"releases": [{"version": "v0.1.0"}]}
        next_release = {**self.metadata, "tag_name": "v0.2.0"}
        validate_release_catalog(config, [self.metadata, {**next_release, "prerelease": True}, {**next_release, "draft": True}])
        with self.assertRaisesRegex(ValueError, "Every published"):
            validate_release_catalog(config, [self.metadata, next_release])

    def test_draft_or_prerelease_cannot_be_stable(self):
        for field, value in (("draft", True), ("prerelease", True), ("published_at", None), ("tag_name", "v0.2.0")):
            with self.subTest(field=field):
                metadata = {**self.metadata, field: value}
                with self.assertRaises(ValueError):
                    validate_release_metadata(metadata, "v0.1.0")
