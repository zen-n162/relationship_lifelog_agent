from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class BackupInfo:
    path: Path
    size_bytes: int

    @property
    def filename(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class RestoreResult:
    restored_from: Path
    pre_restore_backup: Path | None


def default_backup_dir(db_path: str | Path) -> Path:
    return Path(db_path).expanduser().parent / "backups"


def create_relationship_db_backup(
    db_path: str | Path,
    *,
    backup_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> BackupInfo:
    source_path = Path(db_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"relationship DB does not exist: {source_path}")
    destination_dir = Path(backup_dir).expanduser() if backup_dir is not None else default_backup_dir(source_path)
    destination_dir.mkdir(parents=True, exist_ok=True)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    destination_path = _unique_backup_path(destination_dir / f"relationship_{stamp}.sqlite")
    _sqlite_backup(source_path, destination_path)
    return BackupInfo(path=destination_path, size_bytes=destination_path.stat().st_size)


def list_relationship_db_backups(
    db_path: str | Path,
    *,
    backup_dir: str | Path | None = None,
) -> list[BackupInfo]:
    destination_dir = Path(backup_dir).expanduser() if backup_dir is not None else default_backup_dir(db_path)
    if not destination_dir.exists():
        return []
    backups = sorted(destination_dir.glob("relationship_*.sqlite"), key=lambda item: item.name, reverse=True)
    return [BackupInfo(path=path, size_bytes=path.stat().st_size) for path in backups if path.is_file()]


def restore_relationship_db_backup(
    db_path: str | Path,
    *,
    backup_path: str | Path,
    create_pre_restore_backup: bool = True,
) -> RestoreResult:
    target_path = Path(db_path).expanduser()
    source_path = Path(backup_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"backup does not exist: {source_path}")
    if not source_path.is_file():
        raise ValueError("backup path must be a file")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pre_restore_backup: BackupInfo | None = None
    if create_pre_restore_backup and target_path.exists():
        pre_restore_backup = create_relationship_db_backup(target_path)
    _sqlite_backup(source_path, target_path)
    return RestoreResult(
        restored_from=source_path,
        pre_restore_backup=pre_restore_backup.path if pre_restore_backup else None,
    )


def _sqlite_backup(source_path: Path, destination_path: Path) -> None:
    with sqlite3.connect(source_path) as source, sqlite3.connect(destination_path) as destination:
        source.backup(destination)


def _unique_backup_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not create unique backup path under {path.parent}")
