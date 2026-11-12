"""Pin current code branches and published releases, then dispatch the docs build."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib

sys.dont_write_bytecode = True
from archive_pages import COMMIT_EMAIL, COMMIT_NAME, REPOSITORY, SHA, SOURCE_REF, VERSION, git, remote_ref, require
from verify_releases import formal_releases, validate_release_metadata


def validate_context(repository, event, ref, payload):
    require(repository == REPOSITORY, "Unexpected documentation repository")
    if event in ("push", "workflow_dispatch"):
        require(ref in ("refs/heads/main", "refs/heads/dev_jinyuan"), "Only code branches may synchronize versions")
    elif event == "release":
        release = payload.get("release", {})
        require(payload.get("action") == "published", "Only published releases may synchronize versions")
        require(re.fullmatch(VERSION, release.get("tag_name", "")), "Expected a formal version tag")
        validate_release_metadata(release, release["tag_name"])
    else:
        raise ValueError("Pull requests and unsupported events cannot synchronize versions")


def resolve_versions(repo, source, dev, main, published):
    require(all(re.fullmatch(SHA, s) for s in (source, dev, main)), "Invalid source commit")
    code = json.loads(git(repo, "show", f"{source}:docs/code-version.json"))
    config = json.loads(git(repo, "show", f"{source}:docs/versions.json"))
    require(code.get("schema_version") == config.get("schema_version") == 1, "Invalid version configuration")
    rows = {r["version"]: dict(r) for r in config["releases"]}
    require(len(rows) == len(config["releases"]) and config["stable"] in rows, "Invalid release registry")
    for release in formal_releases(published):
        version = release["tag_name"]
        commit = git(repo, "rev-parse", "--verify", f"refs/tags/{version}^{{commit}}")
        if version in rows:
            require(rows[version]["code_sha"] == commit, "Published release code identity cannot change")
        else:
            # Freeze prose at first registration; later branch updates retain it.
            rows[version] = dict(version=version, code_sha=commit, docs_sha=source)
    for version, row in rows.items():
        require(re.fullmatch(VERSION, version) and all(re.fullmatch(SHA, row[k]) for k in ("code_sha", "docs_sha")),
                "Invalid release identity")
        require(git(repo, "rev-parse", "--verify", f"refs/tags/{version}^{{commit}}") == row["code_sha"],
                "Published release tag cannot move")
        project = tomllib.loads(git(repo, "show", f"{row['code_sha']}:Project.toml"))
        require(project.get("version") == version[1:], "Release Project.toml version mismatch")
        git(repo, "merge-base", "--is-ancestor", row["code_sha"], main)
        git(repo, "merge-base", "--is-ancestor", row["docs_sha"], source)
        git(repo, "cat-file", "-e", f"{row['docs_sha']}:docs/api-map.toml")
        git(repo, "cat-file", "-e", f"{row['docs_sha']}:docs/src/index.md")
    ordered = sorted(rows, key=lambda v: tuple(map(int, re.fullmatch(VERSION, v).groups())), reverse=True)
    return (dict(code, code_sha=dev, main_sha=main),
            dict(config, stable=ordered[0], releases=[rows[v] for v in ordered]))


def metadata_commit(repo, parent, code, releases):
    with tempfile.TemporaryDirectory() as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        git(repo, "read-tree", parent, env=env)
        for path, value in (("docs/code-version.json", code), ("docs/versions.json", releases)):
            blob = subprocess.check_output(["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
                                           input=json.dumps(value, indent=2) + "\n", text=True).strip()
            git(repo, "update-index", "--add", "--cacheinfo", "100644", blob, path, env=env)
        tree = git(repo, "write-tree", env=env)
    if git(repo, "rev-parse", f"{parent}^{{tree}}") == tree:
        return parent
    author = dict(os.environ, GIT_AUTHOR_NAME=COMMIT_NAME, GIT_AUTHOR_EMAIL=COMMIT_EMAIL,
                  GIT_COMMITTER_NAME=COMMIT_NAME, GIT_COMMITTER_EMAIL=COMMIT_EMAIL)
    return git(repo, "commit-tree", tree, "-p", parent, "-m", "docs: synchronize code branches and published releases", env=author)


def synchronize(repo, published):
    require(not git(repo, "status", "--porcelain"), "Source checkout must be clean")
    for attempt in range(3):
        git(repo, "fetch", "--no-tags", "origin", SOURCE_REF, "refs/heads/main", "refs/heads/dev_jinyuan")
        git(repo, "fetch", "origin", "refs/tags/v*:refs/tags/v*")
        source = remote_ref(repo, SOURCE_REF)
        dev, main = (remote_ref(repo, "refs/heads/" + b) for b in ("dev_jinyuan", "main"))
        code, releases = resolve_versions(repo, source, dev, main, published)
        commit = metadata_commit(repo, source, code, releases)
        try:
            if commit != source:
                git(repo, "push", "origin", f"{commit}:{SOURCE_REF}")
            return {"docs_sha": commit, "updated": commit != source, "dev": dev, "main": main, "stable": releases["stable"]}
        except subprocess.CalledProcessError:
            if remote_ref(repo, SOURCE_REF) == source or attempt == 2:
                raise
    raise ValueError("Version synchronization did not complete")


def main():
    try:
        payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        validate_context(os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_EVENT_NAME"),
                         os.environ.get("GITHUB_REF"), payload)
        pages = json.loads(subprocess.check_output([
            "gh", "api", "--paginate", "--slurp", f"repos/{REPOSITORY}/releases?per_page=100",
        ], text=True))
        result = synchronize(Path.cwd(), [r for page in pages for r in page])
        # GITHUB_TOKEN pushes do not trigger push workflows. Dispatch explicitly.
        subprocess.run(["gh", "workflow", "run", "documentation.yml", "--repo", REPOSITORY, "--ref", "docs"], check=True)
        print(json.dumps(result))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        sys.exit(f"Documentation version synchronization failed: {error}")


if __name__ == "__main__":
    main()
