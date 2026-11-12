#!/usr/bin/env python3
"""Preview the complete multi-version Pages artifact with clean URLs."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        request = unquote(urlsplit(path).path)
        base = "/HighOrderMRI.jl/"
        relative = request[len(base):] if request.startswith(base) else request.lstrip("/")
        root = Path(self.directory).resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            return str(root / "missing-file")
        if target.is_dir():
            target /= "index.html"
        if not target.exists() and not target.suffix:
            target = target.with_suffix(".html")
        return str(target)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / ".pages")
    args = parser.parse_args()
    if not (args.directory / "versions.json").is_file():
        parser.error("Build the complete website before starting its preview")
    print(f"Preview http://localhost:{args.port}/HighOrderMRI.jl/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(args.directory))).serve_forever()
