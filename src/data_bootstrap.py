import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ARCHIVE_NAME = "data.tgz"
DATA_ROOT_NAME = "data-distrib"


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _extract_archive_prefix(archive_path: Path, repo_root: Path, prefix: str) -> None:
    prefix = prefix.strip("/")
    target_root = repo_root.resolve()

    with tarfile.open(archive_path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name == prefix or member.name.startswith(f"{prefix}/")
        ]
        if not members:
            raise FileNotFoundError(
                f"Could not find '{prefix}' inside archive '{archive_path}'."
            )

        for member in members:
            destination = (target_root / member.name).resolve()
            if destination != target_root and target_root not in destination.parents:
                raise ValueError(f"Refusing to extract unsafe archive member: {member.name}")

        archive.extractall(path=target_root, members=members)


def ensure_repo_data_path(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    if resolved.exists():
        return resolved

    repo_root = REPO_ROOT.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise FileNotFoundError(f"Path does not exist: {resolved}") from exc

    relative_str = relative.as_posix()
    if not (
        relative_str == DATA_ROOT_NAME
        or relative_str.startswith(f"{DATA_ROOT_NAME}/")
    ):
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    archive_path = REPO_ROOT / DATA_ARCHIVE_NAME
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Missing required data path '{resolved}' and archive '{archive_path}'."
        )

    _extract_archive_prefix(
        archive_path=archive_path,
        repo_root=repo_root,
        prefix=relative_str,
    )

    if not resolved.exists():
        raise FileNotFoundError(
            f"Archive extraction completed, but required path is still missing: {resolved}"
        )
    return resolved
