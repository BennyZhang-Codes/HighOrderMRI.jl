#!/usr/bin/env python3
"""Check the published dev site against its pinned code commit, using stdlib only."""

import argparse
from dataclasses import dataclass
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4


class CheckError(ValueError):
    pass


@dataclass(frozen=True)
class Response:
    status: int
    url: str
    content_type: str
    body: bytes = b""


@dataclass(frozen=True)
class Page:
    path: str
    phrase: str = ""
    source_link: bool = False
    svg: bool = False


PAGES = (
    Page("dev/", phrase="The common signal model is:"),
    Page("dev/theory/symbols", phrase="Reconstruction mask"),
    Page("dev/reference/highorderlowrankop", source_link=True),
    Page("dev/figures/adaptive-shared-basis/Fig1_adaptive_shared_basis.svg", svg=True),
    Page("dev/figures/adaptive-shared-basis/Fig1_adaptive_shared_basis_dark.svg", svg=True),
)


def origin(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise CheckError("site URL must be an HTTP(S) URL without credentials")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise CheckError("site URL has an invalid port") from error
    return parsed.scheme, parsed.hostname.lower(), port


def site_root(url):
    origin(url)
    parsed = urlsplit(url)
    if parsed.query or parsed.fragment:
        raise CheckError("site root URL must not contain a query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/") + "/", "", ""))


def load_code_sha(path):
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CheckError("cannot read a valid code-version.json") from error
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise CheckError("code-version.json requires schema_version 1")
    sha = config.get("code_sha")
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise CheckError("code-version.json requires a full lowercase code_sha")
    return sha


class SameOriginRedirect(HTTPRedirectHandler):
    def __init__(self, allowed_origin):
        self.allowed_origin = allowed_origin

    def redirect_request(self, request, fp, code, message, headers, new_url):
        target = urljoin(request.full_url, new_url)
        if origin(target) != self.allowed_origin:
            raise CheckError("redirect leaves the site origin")
        redirected = super().redirect_request(request, fp, code, message, headers, target)
        if redirected is not None and request.get_method() == "HEAD":
            redirected.method = "HEAD"
        return redirected


def fetch_response(url, method, timeout):
    request = Request(url, method=method, headers={"Cache-Control": "no-cache", "User-Agent": "HighOrderMRI-Pages-Check"})
    opener = build_opener(SameOriginRedirect(origin(url)))
    try:
        with opener.open(request, timeout=timeout) as response:
            body = b"" if method == "HEAD" else response.read(8_000_001)
            if len(body) > 8_000_000:
                raise CheckError("HTML response exceeds the size limit")
            return Response(response.status, response.geturl(), response.headers.get("Content-Type", ""), body)
    except HTTPError as error:
        raise CheckError(f"HTTP {error.code}") from error
    except (URLError, OSError) as error:
        raise CheckError(f"network error ({type(error).__name__})") from error


class PageHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.has_html = False
        self.text = []
        self.links = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        self.has_html |= tag == "html"
        if tag in ("script", "style"):
            self.hidden += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)


def check_response(page, response, root, code_sha):
    if response.status != 200:
        raise CheckError(f"HTTP {response.status}")
    if origin(response.url) != origin(root):
        raise CheckError("response leaves the site origin")
    mime = response.content_type.split(";", 1)[0].strip().lower()
    if page.svg:
        if mime != "image/svg+xml":
            raise CheckError("HTTP 200 but expected an SVG content type")
        return
    if mime not in ("text/html", "application/xhtml+xml"):
        raise CheckError("HTTP 200 but expected an HTML content type")
    html = PageHTML()
    try:
        html.feed(response.body.decode("utf-8"))
    except UnicodeError as error:
        raise CheckError("HTML is not valid UTF-8") from error
    if not html.has_html:
        raise CheckError("HTTP 200 but response is not an HTML document")
    text = " ".join(" ".join(html.text).split())
    if page.phrase and page.phrase not in text:
        raise CheckError("HTTP 200 but expected page text is missing")
    if page.source_link and not any(
        urlsplit(link).scheme in ("http", "https") and f"/blob/{code_sha}/" in urlsplit(link).path
        for link in html.links
    ):
        raise CheckError("HTTP 200 but source links do not match the pinned code SHA")


def verify_site(root, code_sha, *, attempts=12, interval=10, max_wait=120,
                fetch=fetch_response, sleep=time.sleep, clock=time.monotonic, emit=print):
    root = site_root(root)
    if attempts < 1 or not math.isfinite(interval) or interval < 0 or not math.isfinite(max_wait) or max_wait <= 0:
        raise CheckError("attempts and max-wait must be positive; interval must be nonnegative")
    deadline = clock() + max_wait
    for attempt in range(1, attempts + 1):
        failures = []
        for page in PAGES:
            remaining = deadline - clock()
            if remaining <= 0:
                failures.append(f"{page.path}: verification deadline reached")
                break
            url = urljoin(root, page.path) + "?" + urlencode({"pages_check": uuid4().hex})
            try:
                response = fetch(url, "HEAD" if page.svg else "GET", min(10, remaining))
                check_response(page, response, root, code_sha)
            except CheckError as error:
                failures.append(f"{page.path}: {error}")
        if not failures:
            emit(f"Verified 3 dev pages and 2 Fig.1 SVGs for code {code_sha}.")
            return True
        for failure in failures:
            emit(f"Attempt {attempt}/{attempts}: {failure}")
        remaining = deadline - clock()
        if attempt == attempts or remaining <= 0:
            break
        sleep(min(interval, remaining))
    emit("Published-site verification failed within the configured retry limit.")
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-url", default=os.environ.get("PAGES_URL", ""))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "docs/code-version.json")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--max-wait", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        code_sha = load_code_sha(args.config)
        success = verify_site(args.pages_url, code_sha, attempts=args.attempts,
                              interval=args.interval, max_wait=args.max_wait)
        return 0 if success else 1
    except CheckError as error:
        print(f"Published-site verification failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
