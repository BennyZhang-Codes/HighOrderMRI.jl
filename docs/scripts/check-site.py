"""Check built VitePress links, API coverage, source identity and math with stdlib only."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import json, posixpath, re, sys, tomllib

BASE = sys.argv[3] if len(sys.argv) > 3 else "/HighOrderMRI.jl/dev/"
VERSION = sys.argv[4] if len(sys.argv) > 4 else "dev"

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = Counter()
        self.links, self.assets, self.math_errors = [], [], []
        self.math = 0
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids[a["id"]] += 1
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        if tag in ("img", "script") and a.get("src"):
            self.assets.append(a["src"])
        if tag == "mjx-container":
            self.math += 1
        if tag == "merror" or a.get("data-mml-node") == "merror" or "katex-error" in a.get("class", ""):
            self.math_errors.append(a)

def check_site(docs):
    docs = Path(docs).resolve()
    dist = Path(sys.argv[2]).resolve()
    mapping = tomllib.loads((docs / "api-map.toml").read_text())
    index = tomllib.loads((docs / ".generated/api/index.toml").read_text())
    meta = json.loads((docs / ".generated/api/build-meta.json").read_text())
    pages = {str(p.relative_to(dist)): Page(p.read_text()) for p in dist.rglob("*.html")}
    errors = []
    def fail(page, reason, value=""):
        errors.append(dict(page=page, reason=reason, value=value))
    for path, page in pages.items():
        if page.math_errors:
            fail(path, "math rendering error")
        content = (dist / path).read_text()
        if "<!--@include:" in content or "(@ref " in content:
            fail(path, "unexpanded source directive")
        for value in page.links + page.assets:
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            req = unquote(url.path)
            if req.startswith(BASE):
                req = req[len(BASE):] or "index.html"
            elif req.startswith("/"):
                fail(path, "outside configured base", value)
                continue
            elif not req:
                req = path
            else:
                req = posixpath.normpath(str(Path(path).parent / req))
            choices = [req, req + ".html", req.rstrip("/") + "/index.html"]
            found = next((x for x in choices if (dist / x).is_file()), None)
            if found is None:
                fail(path, "missing target", value)
            elif url.fragment and found in pages and unquote(url.fragment) not in pages[found].ids:
                fail(path, "missing anchor", value)
    entries = {e["id"]: e for e in mapping["api"]}
    generated = {e["id"]: e for e in index["api"]}
    if not index["partial"] and entries.keys() != generated.keys():
        fail("api", "incomplete generated coverage")
    for id, record in generated.items():
        entry = entries[id]
        target = entry["page"][:-3] + ".html"
        page = pages.get(target)
        if page is None:
            fail(target, "missing API page", id)
            continue
        for anchor in [entry["anchor"]] + [p["anchor"] for p in entry.get("parts", [])]:
            if page.ids[anchor] != 1:
                fail(target, "API anchor must occur once", anchor)
        for source in record["sources"]:
            prefix = "https://github.com/BennyZhang-Codes/HighOrderMRI.jl/blob/" + meta["codeRef"] + "/" + source["path"]
            link = prefix if meta["preview"] else prefix + "#L" + str(source["line"])
            if link not in page.links:
                fail(target, "source link does not match generated metadata", link)
        edit = "https://github.com/BennyZhang-Codes/HighOrderMRI.jl/edit/docs/docs/src/" + entry["page"]
        if VERSION == "dev" and edit not in page.links:
            fail(target, "edit link does not target docs branch")
    for entry in mapping.get("exports", []):
        target = entry["page"][:-3] + ".html"
        page = pages.get(target)
        if page is None:
            fail(target, "missing export disposition page", entry["binding"])
        elif entry.get("anchor") and page.ids[entry["anchor"]] != 1:
            fail(target, "missing export disposition anchor", entry["anchor"])
        elif entry["binding"] not in (dist / target).read_text():
            fail(target, "export disposition has no visible binding name", entry["binding"])
    expected = {str(p.relative_to(docs / "src"))[:-3] + ".html" for p in (docs / "src").rglob("*.md") if ".vitepress" not in p.parts}
    if set(pages) != expected | {"404.html"}:
        fail("site", "unexpected or missing content pages", str(sorted(set(pages) ^ (expected | {"404.html"}))))
    if VERSION != "dev" and any("/edit/docs/" in link for page in pages.values() for link in page.links):
        fail("site", "frozen release exposes an edit-current-docs link")
    report = dict(content_pages=len(expected), html_pages=len(pages), api_bindings=len(generated),
                  original_bindings=sum(e.get("original_documenter", True) for e in entries.values()),
                  additional_export_dispositions=len(mapping.get("exports", [])), links=sum(len(p.links) for p in pages.values()),
                  math_expressions=sum(p.math for p in pages.values()), errors=errors)
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    result = check_site(sys.argv[1])
    raise SystemExit(bool(result["errors"]))
