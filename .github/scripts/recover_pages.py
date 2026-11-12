"""Recover a checked Pages artifact without rebuilding or executing its contents."""

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import zipfile


WORKFLOW = ".github/workflows/documentation.yml"
BUILD_JOB = "Build checked VitePress versions"
LEGACY_BUILD_JOB = "Build checked VitePress dev documentation"
MAX_BYTES = 1024 ** 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gh(endpoint):
    return subprocess.run(["gh", "api", endpoint], check=True, stdout=subprocess.PIPE).stdout


def paginated(endpoint, key):
    result = []
    for page in range(1, 101):
        separator = "&" if "?" in endpoint else "?"
        items = json.loads(gh(f"{endpoint}{separator}per_page=100&page={page}"))[key]
        result.extend(items)
        if len(items) < 100:
            return result
    raise ValueError("GitHub pagination exceeded 100 pages")


def validate_source(run, repository, run_id, source_tree, current_tree, current_sha):
    require(run.get("id") == run_id, "Source run ID mismatch")
    for field in ("repository", "head_repository"):
        require(run.get(field, {}).get("full_name", "").lower() == repository.lower(),
                "Source run must belong to this repository, including its head")
    require(run.get("head_branch") == "docs", "Source run must use the docs branch")
    require(run.get("path") == WORKFLOW, "Source run uses the wrong workflow")
    require(run.get("event") in ("push", "workflow_dispatch"), "Unsupported source event")
    require(run.get("status") == "completed", "Source run must be completed")
    require(re.fullmatch(r"[0-9a-f]{40}", run.get("head_sha", "")), "Invalid source SHA")
    require(re.fullmatch(r"[0-9a-f]{40}", current_sha), "Invalid current SHA")
    require(current_sha != run["head_sha"], "Recovery requires a new commit SHA, not the source deployment ID")
    require(source_tree and source_tree == current_tree, "The docs tree has changed since the source build")


def validate_build(jobs, run):
    require(any(job.get("name") in (BUILD_JOB, LEGACY_BUILD_JOB) and job.get("run_id") == run["id"]
                and job.get("head_sha") == run["head_sha"]
                and job.get("status") == "completed" and job.get("conclusion") == "success"
                for job in jobs), "No successful checked build job at the source SHA")


def select_artifact(artifacts, run):
    candidates = [a for a in artifacts if a.get("name") == "github-pages" and a.get("expired") is False]
    require(len(candidates) == 1, "Expected exactly one unexpired github-pages artifact")
    artifact = candidates[0]
    metadata = artifact.get("workflow_run", {})
    expected = {"id": run["id"], "head_sha": run["head_sha"], "head_branch": "docs",
                "repository_id": run["repository"]["id"], "head_repository_id": run["repository"]["id"]}
    require(all(metadata.get(k) == v for k, v in expected.items()), "Artifact source identity mismatch")
    require(type(artifact.get("id")) is int and artifact["id"] > 0, "Invalid artifact ID")
    require(0 < artifact.get("size_in_bytes", 0) <= MAX_BYTES, "Artifact exceeds the supported size")
    return artifact


def extract_archive(blob, output, code_sha):
    require(re.fullmatch(r"[0-9a-f]{40}", code_sha), "Invalid configured code SHA")
    require(len(blob) <= MAX_BYTES, "Downloaded artifact exceeds the supported size")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        entries = archive.infolist()
        require(len(entries) == 1 and entries[0].filename == "artifact.tar", "ZIP must contain only artifact.tar")
        require(0 < entries[0].file_size <= MAX_BYTES, "TAR exceeds the supported size")
        raw = archive.read(entries[0])  # Includes ZIP CRC verification.
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        members, folded, total = {}, set(), 0
        for member in archive:
            name = member.name
            require(not name.startswith("/") and "\\" not in name and ":" not in name
                    and not any(ord(c) < 32 for c in name) and ".." not in name.split("/"), "Unsafe TAR path")
            path = str(PurePosixPath(name))
            require(member.isfile() or member.isdir(), "TAR links and special files are not allowed")
            require(path != "." or member.isdir(), "Invalid TAR root")
            require(path.casefold() not in folded, "TAR path collision")
            folded.add(path.casefold())
            members[path] = member
            total += member.size
            require(total <= MAX_BYTES, "Extracted site exceeds the supported size")
        for path in members:
            require(all(str(parent) not in members or members[str(parent)].isdir()
                        for parent in PurePosixPath(path).parents), "TAR file/directory collision")
        required = ("index.html", "dev/index.html", "dev/reference/highorderlowrankop.html")
        require(all(p in members and members[p].isfile() for p in required), "Required site pages are missing")
        reference = archive.extractfile(members[required[-1]]).read().decode("utf-8")
        require(code_sha in reference, "Artifact does not document the configured code SHA")
        target = Path(output)
        require(not target.is_symlink() and (not target.exists() or target.is_dir() and not any(target.iterdir())),
                "Output must be an empty directory, not a symlink")
        target.mkdir(parents=True, exist_ok=True)
        target.chmod(0o755)
        for path, member in members.items():
            destination = target / path
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                destination.chmod(0o755)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
                destination.chmod(0o644)
    return sum(member.isfile() for member in members.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Empty destination for the recovered website")
    args = parser.parse_args()
    try:
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        run_id = os.environ.get("SOURCE_RUN_ID", "")
        require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository), "Set GITHUB_REPOSITORY to owner/repo")
        require(re.fullmatch(r"[1-9][0-9]*", run_id), "SOURCE_RUN_ID must be a positive integer")
        prefix = f"repos/{repository}/actions"
        run = json.loads(gh(f"{prefix}/runs/{run_id}"))
        require(re.fullmatch(r"[0-9a-f]{40}", run.get("head_sha", "")), "Invalid source SHA")
        def tree(ref):
            return subprocess.check_output(["git", "rev-parse", f"{ref}:docs"], text=True).strip()
        current_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        validate_source(run, repository, int(run_id), tree(run["head_sha"]), tree(current_sha), current_sha)
        validate_build(paginated(f"{prefix}/runs/{run_id}/jobs?filter=all", "jobs"), run)
        artifact = select_artifact(paginated(f"{prefix}/runs/{run_id}/artifacts", "artifacts"), run)
        config = json.loads(Path("docs/code-version.json").read_text())
        require(config.get("schema_version") == 1, "Unsupported code-version schema")
        count = extract_archive(gh(f"{prefix}/artifacts/{artifact['id']}/zip"), args.output, config["code_sha"])
        print(json.dumps({"source_run": run["id"], "source_sha": run["head_sha"],
                          "artifact_id": artifact["id"], "code_sha": config["code_sha"], "files": count}))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Pages artifact recovery failed: {error}\n")


if __name__ == "__main__":
    main()
