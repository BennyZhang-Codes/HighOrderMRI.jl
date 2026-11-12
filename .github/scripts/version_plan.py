"""Resolve branch and release documentation directly from the code repository."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from archive_pages import (ARCHIVE_REF, REPOSITORY, SHA, SOURCE_REFS, VERSION,
                           documentation_source, git, require, validate_context)
from verify_releases import formal_releases


def resolve_versions(repo, published, archive, source, event, ref, base=None, release_tag=""):
    require(re.fullmatch(SHA, source), "Invalid build source")
    legacy = json.loads((repo / "docs/versions.json").read_text())
    rows = {r["version"]: dict(r) for r in legacy["releases"]}
    for row in archive.get("versions", []):
        if re.fullmatch(VERSION, row["version"]):
            frozen = {k: row[k] for k in ("version", "code_sha", "docs_sha")}
            require(row["version"] not in rows or rows[row["version"]] == frozen, "Frozen release identity mismatch")
            rows[row["version"]] = frozen
    for release in formal_releases(published):
        version = release["tag_name"]
        code = git(repo, "rev-parse", "--verify", f"refs/tags/{version}^{{commit}}")
        if version in rows:
            require(rows[version]["code_sha"] == code, "Published release tag cannot move")
        else:
            require(documentation_source(repo, code) == code, f"Release {version} has no in-tree documentation")
            rows[version] = dict(version=version, code_sha=code, docs_sha=code)
    if release_tag:
        require(re.fullmatch(VERSION, release_tag), "Expected a formal release tag")
        require(release_tag in {r["tag_name"] for r in formal_releases(published)}, "Backfill requires an existing published formal release")
    versions = sorted(rows, key=lambda v: tuple(map(int, re.fullmatch(VERSION, v).groups())), reverse=True)
    require(versions, "At least one published formal version is required")
    targets = []
    for version, branch_ref in SOURCE_REFS.items():
        branch = branch_ref.removeprefix("refs/heads/")
        code = git(repo, "rev-parse", "refs/remotes/origin/" + branch)
        if event == "pull_request" and base == branch or event in ("push", "workflow_dispatch") and ref == branch_ref:
            code = source
        targets.append(dict(version=version, code_sha=code, docs_sha=documentation_source(repo, code)))
    stable = rows[versions[0]]
    releases = [rows[v] for v in versions]
    return dict(schema_version=1, source_sha=source, stable=stable["version"], releases=releases,
                targets=targets + releases + [{**stable, "version": "stable", "release": stable["version"]}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        repo = Path.cwd()
        event, ref = os.environ.get("GITHUB_EVENT_NAME", ""), os.environ.get("GITHUB_REF", "")
        payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        preview = event == "pull_request"
        publish = not preview
        if publish:
            validate_context(os.environ.get("GITHUB_REPOSITORY", ""), event, ref, payload)
        release_tag = os.environ.get("RELEASE_TAG", "")
        require(not release_tag or event == "workflow_dispatch" and ref in SOURCE_REFS.values(), "Manual release backfills must run from a code branch")
        source = git(repo, "rev-parse", "HEAD")
        pages = json.loads(subprocess.check_output([
            "gh", "api", "--paginate", "--slurp", f"repos/{REPOSITORY}/releases?per_page=100",
        ], text=True))
        archived = subprocess.run(["git", "-C", str(repo), "show", "refs/remotes/origin/gh-pages:versions.json"],
                                  capture_output=True, text=True)
        archive = json.loads(archived.stdout) if archived.returncode == 0 else {}
        plan = resolve_versions(repo, [r for page in pages for r in page], archive, source, event, ref,
                                os.environ.get("GITHUB_BASE_REF"), release_tag)
        args.output.write_text(json.dumps(plan, indent=2) + "\n")
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"source_sha={source}\npublish={str(publish).lower()}\npreview={str(preview).lower()}\n")
        with open(os.environ["GITHUB_ENV"], "a") as output:
            output.write(f"DOCS_BUILD_PLAN={args.output.resolve()}\n")
        print(json.dumps(plan, indent=2))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Documentation source selection failed: {error}\n")


if __name__ == "__main__":
    main()
