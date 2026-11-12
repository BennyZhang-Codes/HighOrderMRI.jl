"""TAR extraction must reject links, unsafe paths and incomplete artifacts."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from pages_artifact import extract_tar


def artifact(extra=None, omit=None):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w") as archive:
        for name in ("index.html", "dev/index.html", "dev/reference/highorderlowrankop.html"):
            if name != omit:
                body = b"a" * 40
                info = tarfile.TarInfo(name)
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
        if extra:
            archive.addfile(extra)
    return result.getvalue()


class ArtifactTests(unittest.TestCase):
    def test_checked_tar_extracts_and_optional_source_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "site"
            self.assertEqual(extract_tar(artifact(), target, "a" * 40), 3)
            self.assertTrue((target / "dev/index.html").is_file())
            with self.assertRaisesRegex(ValueError, "configured code SHA"):
                extract_tar(artifact(), Path(temporary) / "wrong", "b" * 40)

    def test_unsafe_links_paths_collisions_and_missing_pages_are_rejected(self):
        cases = []
        for name in ("../outside", "/outside", "dev\\outside", "dev/index.html"):
            cases.append(artifact(tarfile.TarInfo(name)))
        link = tarfile.TarInfo("link")
        link.type, link.linkname = tarfile.SYMTYPE, "../outside"
        cases.append(artifact(link))
        cases.append(artifact(omit="dev/reference/highorderlowrankop.html"))
        for index, data in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                extract_tar(data, Path(temporary) / "site")


if __name__ == "__main__":
    unittest.main()
