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
from pages_artifact import MAX_BYTES, extract_tar


REPOSITORY = "BennyZhang-Codes/HighOrderMRI.jl"
COMMIT_NAME = "BennyZhang-Codes"
COMMIT_EMAIL = "57568342+BennyZhang-Codes@users.noreply.github.com"
SOURCE_REFS = {"dev": "refs/heads/dev_jinyuan", "main": "refs/heads/main"}
LEGACY_DOCS_SOURCE = "4bf8a2e45bfa703db13fc0796e2c4b205de4429c"
ARCHIVE_REF = "refs/heads/gh-pages"
SHA = r"[0-9a-f]{40}"
VERSION = r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(repo, *args, env=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, env=env).strip()


def validate_context(repository, event, ref, payload=None):
    require(repository == REPOSITORY, "Unexpected documentation repository")
    if event in ("push", "workflow_dispatch"):
        require(ref in SOURCE_REFS.values(), "Only code branches may publish documentation")
    elif event == "release":
        from verify_releases import validate_release_metadata
        release = (payload or {}).get("release", {})
        require((payload or {}).get("action") == "published" and re.fullmatch(VERSION, release.get("tag_name", "")),
                "Only published formal releases may publish documentation")
        require(ref == "refs/tags/" + release["tag_name"], "Release ref mismatch")
        validate_release_metadata(release, release["tag_name"])
    else:
        raise ValueError("Pull requests and unsupported events cannot publish documentation")


def documentation_source(repo, code):
    result = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{code}:docs/src/index.md"],
                            capture_output=True)
    # Older code branches predate the in-tree migration. Their saved source
    # remains reachable through the migration merge, without a docs branch.
    return code if result.returncode == 0 else LEGACY_DOCS_SOURCE


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
    expected.update(dev=(code["code_sha"], code.get("dev_docs_sha")), stable=expected[releases["stable"]])
    if "main_sha" in code:
        require(re.fullmatch(SHA, code["main_sha"]), "Invalid main code SHA")
        expected["main"] = (code["main_sha"], code.get("main_docs_sha"))
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


def current_sources(repo, manifest):
    rows = {r["version"]: r for r in manifest["versions"]}
    for version, ref in SOURCE_REFS.items():
        row = rows.get(version)
        require(row is not None, "Both code branch versions are required")
        if remote_ref(repo, ref) != row["code_sha"]:
            return False
        require(row["docs_sha"] == documentation_source(repo, row["code_sha"]), "Branch documentation source mismatch")
    for version, row in rows.items():
        if re.fullmatch(VERSION, version):
            require(git(repo, "rev-parse", "--verify", f"refs/tags/{version}^{{commit}}") == row["code_sha"],
                    "Release tag changed since the build")
    return True


def publish(repo, site, source_sha, manifest, source_ref=None):
    require(re.fullmatch(SHA, source_sha) and git(repo, "rev-parse", "HEAD") == source_sha,
            "Source checkout does not match the checked workflow commit")
    require(not git(repo, "status", "--porcelain"), "Source checkout must be clean")
    if not current_sources(repo, manifest):
        return {"publish": False, "reason": "superseded_source"}
    parent = remote_ref(repo, ARCHIVE_REF)
    if parent:
        git(repo, "fetch", "--no-tags", "origin", ARCHIVE_REF)
        parent = git(repo, "rev-parse", "FETCH_HEAD")
        prior = json.loads(git(repo, "show", f"{parent}:versions.json"))
        preserve_releases(prior, manifest)
    (site / ".nojekyll").touch()
    info = {
        "schema_version": 1, "source_ref": source_ref, "source_sha": source_sha,
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
    author = dict(os.environ, GIT_AUTHOR_NAME=COMMIT_NAME, GIT_AUTHOR_EMAIL=COMMIT_EMAIL,
                  GIT_COMMITTER_NAME=COMMIT_NAME, GIT_COMMITTER_EMAIL=COMMIT_EMAIL)
    args = ["commit-tree", tree]
    if parent:
        args.extend(["-p", parent])
    commit = git(repo, *args, "-m", f"Deploy checked documentation from {source_sha}", env=author)
    if not current_sources(repo, manifest):
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
        payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        validate_context(os.environ.get("GITHUB_REPOSITORY", ""),
                         os.environ.get("GITHUB_EVENT_NAME", ""), os.environ.get("GITHUB_REF", ""), payload)
        repo = Path.cwd()
        require(0 < args.artifact_tar.stat().st_size <= MAX_BYTES, "Artifact exceeds the supported size")
        extract_tar(args.artifact_tar.read_bytes(), args.site)
        plan = json.loads((args.site / "publication-plan.json").read_text())
        require(plan.get("source_sha") == args.source_sha, "Build plan does not match the checked source")
        targets = {r["version"]: r for r in plan["targets"]}
        code = dict(schema_version=1, code_sha=targets["dev"]["code_sha"], main_sha=targets["main"]["code_sha"],
                    dev_docs_sha=targets["dev"]["docs_sha"], main_docs_sha=targets["main"]["docs_sha"])
        releases = dict(schema_version=1, stable=plan["stable"], releases=plan["releases"])
        manifest = validate_site(args.site, code, releases)
        result = publish(repo, args.site.resolve(), args.source_sha, manifest, os.environ["GITHUB_REF"])
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
