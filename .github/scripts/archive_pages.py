"""Publish a checked complete site to gh-pages without modifying source files."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from recover_pages import MAX_BYTES, extract_tar


REPOSITORY = "BennyZhang-Codes/HighOrderMRI.jl"
SOURCE_REF = "refs/heads/docs"
ARCHIVE_REF = "refs/heads/gh-pages"
SHA = r"[0-9a-f]{40}"
VERSION = r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(repo, *args, env=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, env=env).strip()


def validate_context(repository, event, ref):
    require(repository == REPOSITORY, "Unexpected documentation repository")
    require(event in ("push", "workflow_dispatch") and ref == SOURCE_REF,
            "Only trusted docs pushes or dispatches may publish the archive")


def validate_site(site, code, releases):
    manifest = json.loads((site / "versions.json").read_text())
    require(manifest.get("schema_version") == 1 and manifest.get("preview") is False,
            "Only a complete formal site may be archived")
    require(code.get("schema_version") == releases.get("schema_version") == 1, "Invalid source configuration")
    expected = {r["version"]: (r["code_sha"], r["docs_sha"]) for r in releases["releases"]}
    require(len(expected) == len(releases["releases"]) and releases["stable"] in expected,
            "Invalid release registry")
    require(all(re.fullmatch(VERSION, v) for v in expected), "Invalid release version")
    require(all(re.fullmatch(SHA, s) for pair in expected.values() for s in pair), "Invalid release SHA")
    require(re.fullmatch(SHA, code["code_sha"]), "Invalid dev code SHA")
    expected.update(dev=(code["code_sha"], None), stable=expected[releases["stable"]])
    if "main_sha" in code:
        require(re.fullmatch(SHA, code["main_sha"]), "Invalid main code SHA")
        expected["main"] = (code["main_sha"], None)
    rows = manifest.get("versions", [])
    require(len(rows) == len(expected) and {r.get("version") for r in rows} == set(expected),
            "The archive must contain every configured version exactly once")
    require(manifest.get("stable") == releases["stable"], "Stable release mismatch")
    for row in rows:
        version = row["version"]
        require((row.get("code_sha"), row.get("docs_sha")) == expected[version]
                and row.get("preview") is False, "Version source or preview mismatch")
        require((site / version / "index.html").is_file(), "Version index is missing")
        require(json.loads((site / version / "version.json").read_text()) == row,
                "Version metadata does not match the complete-site manifest")
    require((site / "index.html").is_file(), "Site index is missing")
    require(not any(p.is_symlink() for p in site.rglob("*")), "Archive links are not allowed")
    require(not any((site / p).exists() for p in (".git", ".github", "README.md", "Project.toml")),
            "The archive must contain generated pages, not repository source")
    return manifest


def preserve_releases(previous, current):
    old = {r["version"]: r for r in previous["versions"] if re.fullmatch(VERSION, r["version"])}
    new = {r["version"]: r for r in current["versions"] if re.fullmatch(VERSION, r["version"])}
    require(old.keys() <= new.keys(), "Published release versions cannot be removed")
    for version, row in old.items():
        require(all(row.get(k) == new[version].get(k) for k in ("code_sha", "docs_sha")),
                "Published release source identity cannot be changed")
    number = lambda v: tuple(map(int, re.fullmatch(VERSION, v).groups()))
    require(number(current["stable"]) >= number(previous["stable"]), "Stable cannot move backwards")


def remote_ref(repo, ref):
    result = subprocess.run(["git", "-C", str(repo), "ls-remote", "--exit-code", "origin", ref],
                            capture_output=True, text=True)
    if result.returncode == 2:
        return None
    result.check_returncode()
    parts = result.stdout.split()
    require(len(parts) == 2 and re.fullmatch(SHA, parts[0]) and parts[1] == ref, "Invalid remote reference")
    return parts[0]


def publish(repo, site, source_sha, manifest):
    require(re.fullmatch(SHA, source_sha) and git(repo, "rev-parse", "HEAD") == source_sha,
            "Source checkout does not match the checked workflow commit")
    require(not git(repo, "status", "--porcelain"), "Source checkout must be clean")
    if remote_ref(repo, SOURCE_REF) != source_sha:
        return {"publish": False, "reason": "superseded_source"}
    parent = remote_ref(repo, ARCHIVE_REF)
    if parent:
        git(repo, "fetch", "--no-tags", "origin", ARCHIVE_REF)
        parent = git(repo, "rev-parse", "FETCH_HEAD")
        prior = json.loads(git(repo, "show", f"{parent}:versions.json"))
        preserve_releases(prior, manifest)
    (site / ".nojekyll").touch()
    info = {
        "schema_version": 1, "source_branch": "docs", "source_sha": source_sha,
        "dev_code_sha": next(r["code_sha"] for r in manifest["versions"] if r["version"] == "dev"),
        "stable": manifest["stable"],
    }
    main = next((r for r in manifest["versions"] if r["version"] == "main"), None)
    if main:
        info["main_code_sha"] = main["code_sha"]
    (site / "archive-info.json").write_text(json.dumps(info, indent=2) + "\n")
    with tempfile.TemporaryDirectory() as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        git(repo, "read-tree", "--empty", env=env)
        git_dir = git(repo, "rev-parse", "--absolute-git-dir")
        subprocess.run(["git", "--git-dir", git_dir, "--work-tree", str(site),
                        "add", "--all", "--force", "--", "."], cwd=site, env=env, check=True)
        tree = git(repo, "write-tree", env=env)
    if parent and git(repo, "rev-parse", f"{parent}^{{tree}}") == tree:
        return {"publish": True, "sha": parent, "updated": False}
    author = dict(os.environ, GIT_AUTHOR_NAME="github-actions[bot]",
                  GIT_AUTHOR_EMAIL="41898282+github-actions[bot]@users.noreply.github.com",
                  GIT_COMMITTER_NAME="github-actions[bot]",
                  GIT_COMMITTER_EMAIL="41898282+github-actions[bot]@users.noreply.github.com")
    args = ["commit-tree", tree]
    if parent:
        args.extend(["-p", parent])
    commit = git(repo, *args, "-m", f"Deploy checked documentation from {source_sha}", env=author)
    if remote_ref(repo, SOURCE_REF) != source_sha:
        return {"publish": False, "reason": "superseded_source"}
    git(repo, "push", "origin", f"{commit}:{ARCHIVE_REF}")
    return {"publish": True, "sha": commit, "updated": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-tar", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    try:
        validate_context(os.environ.get("GITHUB_REPOSITORY", ""),
                         os.environ.get("GITHUB_EVENT_NAME", ""), os.environ.get("GITHUB_REF", ""))
        repo = Path.cwd()
        code = json.loads((repo / "docs/code-version.json").read_text())
        releases = json.loads((repo / "docs/versions.json").read_text())
        require(0 < args.artifact_tar.stat().st_size <= MAX_BYTES, "Artifact exceeds the supported size")
        extract_tar(args.artifact_tar.read_bytes(), args.site, code["code_sha"])
        manifest = validate_site(args.site, code, releases)
        result = publish(repo, args.site.resolve(), args.source_sha, manifest)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(f"publish={str(result['publish']).lower()}\n")
                if result["publish"]:
                    output.write(f"sha={result['sha']}\n")
        print(json.dumps(result))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Pages archive publication failed: {error}\n")


if __name__ == "__main__":
    main()
