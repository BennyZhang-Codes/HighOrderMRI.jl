#!/usr/bin/env python3
"""Validate configured release identity before building a public website."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tomllib


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def validate_release_metadata(metadata, version):
    if metadata.get("tag_name") != version:
        raise ValueError(f"Release tag does not match {version}")
    if metadata.get("draft") is not False or metadata.get("prerelease") is not False or not metadata.get("published_at"):
        raise ValueError(f"Stable documentation requires published, non-prerelease {version}")


def formal_releases(published):
    return [r for r in published if re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", r.get("tag_name", ""))
            and r.get("draft") is False and r.get("prerelease") is False and r.get("published_at")]


def validate_release_catalog(config, published):
    expected = {r["tag_name"] for r in formal_releases(published)}
    if not expected <= {r["version"] for r in config["releases"]}:
        raise ValueError("Every published formal release must remain in versions.json; synchronize versions first")


def verify(docs, allow_unpublished=False):
    docs = Path(docs).resolve()
    repo = docs.parent
    config = json.loads((docs / "versions.json").read_text())
    if not allow_unpublished:
        pages = json.loads(subprocess.check_output([
            "gh", "api", "--paginate", "--slurp", "repos/BennyZhang-Codes/HighOrderMRI.jl/releases?per_page=100",
        ], text=True))
        validate_release_catalog(config, [r for page in pages for r in page])
    for release in config["releases"]:
        version, code, source = release["version"], release["code_sha"], release["docs_sha"]
        if not re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
            raise ValueError("Invalid version")
        if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (code, source)):
            raise ValueError("Release commits require full SHAs")
        project = tomllib.loads(git(repo, "show", f"{code}:Project.toml"))
        if project.get("version") != version[1:]:
            raise ValueError(f"Project.toml version does not match {version}")
        git(repo, "cat-file", "-e", f"{source}:docs/api-map.toml")
        git(repo, "cat-file", "-e", f"{source}:docs/src/index.md")
        subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", source, "HEAD"], check=True)
        subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", code, "origin/main"], check=True)
        tag = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"refs/tags/{version}^{{commit}}"], capture_output=True, text=True)
        if tag.returncode == 0 and tag.stdout.strip() != code:
            raise ValueError(f"Tag {version} points to a different commit")
        if allow_unpublished:
            print(f"Candidate {version}: code {code}, documentation {source}")
            continue
        if tag.returncode:
            raise ValueError(f"Missing formal tag {version}")
        metadata = json.loads(subprocess.check_output([
            "gh", "api", f"repos/BennyZhang-Codes/HighOrderMRI.jl/releases/tags/{version}",
        ], text=True))
        validate_release_metadata(metadata, version)
        print(f"Published release verified: {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--allow-unpublished", action="store_true")
    args = parser.parse_args()
    try:
        verify(args.docs, args.allow_unpublished)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release validation failed: {error}\n")
