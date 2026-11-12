import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlsplit
from urllib.request import Request

from check_live_pages import (
    CheckError, PAGES, Response, SameOriginRedirect, check_response,
    load_code_sha, origin, site_root, verify_site,
)


SHA = "a" * 40
ROOT = "https://example.test/package/"


def good_response(url, method, timeout):
    path = urlsplit(url).path
    if method == "HEAD":
        return Response(200, url, "image/svg+xml")
    if path.endswith("symbols"):
        body = "<html><table><td>Reconstruction mask</td></table></html>"
    elif path.endswith("highorderlowrankop"):
        body = f'<html><a href="https://github.com/owner/repository/blob/{SHA}/src/op.jl">Source</a></html>'
    else:
        body = "<html><p>The common signal model is:</p></html>"
    return Response(200, url, "text/html; charset=utf-8", body.encode())


class PublishedPagesTests(unittest.TestCase):
    def test_valid_pages_use_get_svg_head_and_unique_cache_busters(self):
        requests = []

        def fetch(url, method, timeout):
            requests.append((url, method, timeout))
            return good_response(url, method, timeout)

        self.assertTrue(verify_site(ROOT, SHA, attempts=1, fetch=fetch, emit=lambda _: None))
        self.assertEqual([r[1] for r in requests], ["GET", "GET", "GET", "HEAD", "HEAD"])
        self.assertEqual(len({urlsplit(r[0]).query for r in requests}), 5)
        self.assertTrue(all(urlsplit(r[0]).path.startswith("/package/dev/") for r in requests))
        self.assertTrue(all(0 < r[2] <= 10 for r in requests))

    def test_stale_sha_fails_even_when_html_contains_expected_sha_as_plain_text(self):
        body = f'<html>{SHA}<a href="https://github.com/owner/repository/blob/{"b" * 40}/src/op.jl">Source</a></html>'
        with self.assertRaisesRegex(CheckError, "pinned code SHA"):
            check_response(PAGES[2], Response(200, ROOT, "text/html", body.encode()), ROOT, SHA)

    def test_wrong_page_http_error_or_svg_content_is_refused(self):
        cases = [
            (PAGES[0], Response(200, ROOT, "text/html", b"<html>Old homepage</html>")),
            (PAGES[0], Response(404, ROOT, "text/html", b"<html>Not found</html>")),
            (PAGES[0], Response(200, ROOT, "application/json", b'{}')),
            (PAGES[0], Response(200, ROOT, "text/html", b"The common signal model is:")),
            (PAGES[0], Response(200, ROOT, "text/html", b"<html><script>The common signal model is:</script></html>")),
            (PAGES[3], Response(200, ROOT, "text/html")),
        ]
        for page, response in cases:
            with self.subTest(page=page.path, status=response.status, mime=response.content_type):
                with self.assertRaises(CheckError):
                    check_response(page, response, ROOT, SHA)

    def test_external_redirect_is_blocked_before_following(self):
        handler = SameOriginRedirect(origin(ROOT))
        with self.assertRaisesRegex(CheckError, "redirect leaves"):
            handler.redirect_request(Request(ROOT), None, 302, "Found", {}, "https://other.test/")
        with self.assertRaisesRegex(CheckError, "response leaves"):
            check_response(PAGES[3], Response(200, "http://example.test/package/", "image/svg+xml"), ROOT, SHA)
        redirected = handler.redirect_request(Request(ROOT, method="HEAD"), None, 302, "Found", {}, ROOT + "dev/")
        self.assertEqual(redirected.get_method(), "HEAD")

    def test_stale_then_current_site_retries_without_sleeping_in_tests(self):
        requests = []
        waits = []
        messages = []

        def fetch(url, method, timeout):
            requests.append(url)
            if len(requests) == 1:
                return Response(200, url, "text/html", b"<html>Old homepage</html>")
            return good_response(url, method, timeout)

        self.assertTrue(verify_site(ROOT, SHA, attempts=2, fetch=fetch, sleep=waits.append, emit=messages.append))
        self.assertEqual(len(requests), 10)
        self.assertEqual(waits, [10])
        self.assertIn("dev/: HTTP 200", messages[0])

    def test_repeated_failure_and_deadline_are_bounded(self):
        calls = []

        def fetch(url, method, timeout):
            calls.append(url)
            return Response(503, url, "text/html")

        self.assertFalse(verify_site(ROOT, SHA, attempts=2, interval=0, fetch=fetch, emit=lambda _: None))
        self.assertEqual(len(calls), 10)
        ticks = iter([0, 121, 121])
        self.assertFalse(verify_site(ROOT, SHA, fetch=fetch, clock=lambda: next(ticks), emit=lambda _: None))
        self.assertEqual(len(calls), 10)

    def test_malformed_config_is_refused_before_network(self):
        invalid = ["not JSON", "[]", "{}", '{"schema_version":true,"code_sha":"' + SHA + '"}',
                   json.dumps({"schema_version": 1, "code_sha": "short"})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "code-version.json"
            for value in invalid:
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(CheckError):
                    load_code_sha(path)
            path.write_text(json.dumps({"schema_version": 1, "code_sha": SHA}))
            self.assertEqual(load_code_sha(path), SHA)

    def test_root_url_keeps_project_path_and_rejects_credentials_or_query(self):
        self.assertEqual(site_root("https://example.test/package"), ROOT)
        for value in ("", "file:///tmp/site", "https://u:p@example.test/", ROOT + "?token=secret"):
            with self.subTest(value=value), self.assertRaises(CheckError):
                site_root(value)


if __name__ == "__main__":
    unittest.main()
