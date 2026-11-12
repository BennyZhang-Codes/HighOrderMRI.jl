"""Safely extract a checked complete Pages TAR using only the standard library."""

import io
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


MAX_BYTES = 1024 ** 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_tar(raw, output, code_sha=None):
    require(code_sha is None or re.fullmatch(r"[0-9a-f]{40}", code_sha), "Invalid configured code SHA")
    require(0 < len(raw) <= MAX_BYTES, "TAR exceeds the supported size")
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
        require(code_sha is None or code_sha in reference, "Artifact does not document the configured code SHA")
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
