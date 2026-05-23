from __future__ import annotations

import argparse
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DB_GLOBS = ("*.sqlite", "*.sqlite3", "*.db")
SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache"}


@dataclass(frozen=True)
class Candidate:
    target: str
    kind: str
    source_id: str
    raw_id: str
    table: str
    label: str
    count: int | None
    confidence: str
    reason: str
    db_label: str


def main() -> int:
    args = _parse_args()
    root = args.personal_root.expanduser().resolve()
    targets = (args.target_name, args.self_name)
    db_paths = _find_db_paths(root)
    selected = _select_db_paths(root, db_paths, include_backups=args.include_backups)

    all_candidates: list[Candidate] = []
    inspected: list[str] = []
    warnings: list[str] = []
    for db_path in selected:
        db_label = _safe_db_label(root, db_path)
        try:
            conn = _open_readonly(db_path)
        except sqlite3.Error as exc:
            warnings.append(f"{db_label}: read-only open failed: {type(exc).__name__}")
            continue
        inspected.append(db_label)
        try:
            all_candidates.extend(_person_candidates(conn, targets=targets, db_label=db_label))
            all_candidates.extend(_line_speaker_link_candidates(conn, targets=targets, db_label=db_label))
            all_candidates.extend(_line_speaker_group_candidates(conn, targets=targets, db_label=db_label))
        finally:
            conn.close()

    print(_render_report(root, targets, db_paths, inspected, all_candidates, warnings))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find profile ID candidates from personal_lifelog_rag using SQLite read-only connections."
    )
    parser.add_argument(
        "--personal-root",
        type=Path,
        default=Path("~/MyApplication/personal_lifelog_rag"),
        help="personal_lifelog_rag root path.",
    )
    parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Also inspect backup DBs. Default inspects current data/db/lifelog.sqlite first.",
    )
    parser.add_argument("--target-name", required=True, help="Target person's manually provided local name.")
    parser.add_argument("--self-name", required=True, help="Self person's manually provided local name.")
    return parser.parse_args()


def _find_db_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in DB_GLOBS:
        paths.extend(root.rglob(pattern))
    return sorted(path for path in paths if not _is_skipped(path, root))


def _is_skipped(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts
    return any(part in SKIP_DIRS for part in relative_parts)


def _select_db_paths(root: Path, db_paths: list[Path], *, include_backups: bool) -> list[Path]:
    current = root / "data" / "db" / "lifelog.sqlite"
    selected: list[Path] = []
    if current in db_paths:
        selected.append(current)
    if include_backups:
        selected.extend(path for path in db_paths if path not in selected)
    return selected or db_paths[:1]


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _person_candidates(conn: sqlite3.Connection, *, targets: tuple[str, str], db_label: str) -> list[Candidate]:
    if not _table_exists(conn, "persons"):
        return []
    candidates: list[Candidate] = []
    for target in targets:
        rows = conn.execute(
            """
            SELECT persons.id,
                   persons.display_name,
                   persons.public_name,
                   persons.manual_verified,
                   persons.hidden,
                   persons.deleted_at,
                   COUNT(DISTINCT person_face_clusters.face_cluster_id) AS linked_clusters_count,
                   COUNT(DISTINCT media_people.media_id) AS media_count
            FROM persons
            LEFT JOIN person_face_clusters ON person_face_clusters.person_id = persons.id
            LEFT JOIN media_people ON media_people.person_id = persons.id
            WHERE persons.display_name = ?
               OR persons.public_name = ?
            GROUP BY persons.id
            ORDER BY persons.created_at ASC, persons.id ASC
            """,
            (target, target),
        ).fetchall()
        for row in rows:
            raw_id = str(row["id"])
            linked_clusters = int(row["linked_clusters_count"] or 0)
            media_count = int(row["media_count"] or 0)
            confidence = "high" if linked_clusters or media_count else "medium"
            reason = (
                "manual person label exact match; "
                f"manual_verified={int(row['manual_verified'] or 0)}, "
                f"linked_face_clusters={linked_clusters}, media_count={media_count}"
            )
            if row["deleted_at"] is not None or int(row["hidden"] or 0) != 0:
                confidence = "low"
                reason += "; hidden_or_deleted=true"
            candidates.append(
                Candidate(
                    target=target,
                    kind="person",
                    source_id=f"plr:person:{raw_id}",
                    raw_id=raw_id,
                    table="persons",
                    label=_safe_label(row["display_name"] or row["public_name"] or target, targets),
                    count=media_count,
                    confidence=confidence,
                    reason=reason,
                    db_label=db_label,
                )
            )
    return candidates


def _line_speaker_link_candidates(conn: sqlite3.Connection, *, targets: tuple[str, str], db_label: str) -> list[Candidate]:
    if not (_table_exists(conn, "line_speaker_links") and _table_exists(conn, "line_messages")):
        return []
    candidates: list[Candidate] = []
    for target in targets:
        rows = conn.execute(
            """
            SELECT line_speaker_links.id,
                   line_speaker_links.chat_id,
                   line_speaker_links.speaker_name,
                   line_speaker_links.person_id,
                   line_speaker_links.verified_by_user,
                   line_speaker_links.confidence,
                   COUNT(line_messages.id) AS message_count,
                   MIN(line_messages.sent_at) AS first_message_at,
                   MAX(line_messages.sent_at) AS last_message_at
            FROM line_speaker_links
            LEFT JOIN persons ON persons.id = line_speaker_links.person_id
            LEFT JOIN line_messages
              ON line_messages.chat_id = line_speaker_links.chat_id
             AND line_messages.sender = line_speaker_links.speaker_name
            WHERE line_speaker_links.speaker_name = ?
            GROUP BY line_speaker_links.id
            ORDER BY message_count DESC, line_speaker_links.created_at ASC, line_speaker_links.id ASC
            """,
            (target,),
        ).fetchall()
        for row in rows:
            raw_id = str(row["id"])
            message_count = int(row["message_count"] or 0)
            confidence = "high" if int(row["verified_by_user"] or 0) == 1 and message_count else "medium"
            candidates.append(
                Candidate(
                    target=target,
                    kind="line_speaker",
                    source_id=f"plr:line_speaker:{raw_id}",
                    raw_id=raw_id,
                    table="line_speaker_links",
                    label=_safe_label(row["speaker_name"] or target, targets),
                    count=message_count,
                    confidence=confidence,
                    reason=(
                        "manual LINE speaker link candidate; "
                        f"verified_by_user={int(row['verified_by_user'] or 0)}, "
                        f"person_id={row['person_id'] or '-'}"
                    ),
                    db_label=db_label,
                )
            )
    return candidates


def _line_speaker_group_candidates(conn: sqlite3.Connection, *, targets: tuple[str, str], db_label: str) -> list[Candidate]:
    if not _table_exists(conn, "line_messages"):
        return []
    candidates: list[Candidate] = []
    for target in targets:
        rows = conn.execute(
            """
            SELECT chat_id,
                   sender,
                   COUNT(*) AS message_count,
                   MIN(sent_at) AS first_message_at,
                   MAX(sent_at) AS last_message_at
            FROM line_messages
            WHERE sender = ?
            GROUP BY chat_id, sender
            ORDER BY message_count DESC, chat_id ASC
            """,
            (target,),
        ).fetchall()
        for row in rows:
            chat_id = str(row["chat_id"] or "")
            sender = str(row["sender"] or target)
            source_hash = hashlib.sha256(f"{chat_id}|{sender}".encode("utf-8")).hexdigest()[:12]
            candidates.append(
                Candidate(
                    target=target,
                    kind="line_speaker_group",
                    source_id=f"plr:line_speaker_group:{source_hash}",
                    raw_id=f"chat_sender_group:{source_hash}",
                    table="line_messages",
                    label=_safe_label(sender, targets),
                    count=int(row["message_count"] or 0),
                    confidence="medium",
                    reason="sender exact match grouped from line_messages; unverified unless matched by line_speaker_links",
                    db_label=db_label,
                )
            )
    return candidates


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _safe_label(value: Any, targets: tuple[str, str]) -> str:
    text = str(value or "").strip()
    if text in targets:
        return text
    return "[redacted_non_target_label]"


def _safe_db_label(root: Path, db_path: Path) -> str:
    try:
        return db_path.relative_to(root).as_posix()
    except ValueError:
        return "[redacted_db_path]"


def _render_report(
    root: Path,
    targets: tuple[str, str],
    db_paths: list[Path],
    inspected: list[str],
    candidates: list[Candidate],
    warnings: list[str],
) -> str:
    lines = [
        "Candidate ID Report",
        "",
        "Scope:",
        f"- personal_lifelog_rag root: [redacted_path]",
        f"- sqlite_db_files_found: {len(db_paths)}",
        "- DB access: sqlite URI mode=ro + PRAGMA query_only=ON",
        "- raw LINE text, note text, exact GPS, image paths, face crops, and embeddings are not printed.",
        "",
        "Inspected DBs:",
    ]
    if inspected:
        lines.extend(f"- {label}" for label in inspected)
    else:
        lines.append("- none")
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)

    target_name, self_name = targets
    for target, title in ((target_name, f"Target: {target_name}"), (self_name, f"Self: {self_name}")):
        lines.extend(["", title])
        _append_candidates(lines, candidates, target=target, kind="person", heading="person candidates")
        _append_candidates(lines, candidates, target=target, kind="line_speaker", heading="line speaker link candidates")
        _append_candidates(lines, candidates, target=target, kind="line_speaker_group", heading="line speaker grouped candidates")

    lines.extend(["", "Recommended profile create command template:"])
    target_person = _first_source(candidates, target=target_name, kind="person")
    target_speaker = _first_source(candidates, target=target_name, kind="line_speaker")
    if target_speaker is None:
        target_speaker = _first_source(candidates, target=target_name, kind="line_speaker_group")
    lines.extend(
        [
            "python -m relationship_lifelog_agent.cli profile create \\",
            f'  --profile-name "{target_name}" \\',
            '  --relationship-label "partner" \\',
            '  --visibility "private" \\',
            f'  --person-source-id "{target_person or "<USER_CONFIRMED_TARGET_PERSON_SOURCE_ID>"}" \\',
            f'  --line-speaker-source-id "{target_speaker or "<USER_CONFIRMED_TARGET_LINE_SPEAKER_SOURCE_ID>"}"',
        ]
    )
    lines.extend(
        [
            "",
            "Notes:",
            "- relationship_lifelog_agent schema currently stores target person_source_id and line_speaker_source_id only.",
            "- self_person_source_id and self_line_speaker_source_id are not present in the current relationship_profiles schema.",
            "- All IDs above are candidates for user confirmation; no relationship label or person/speaker link was inferred by this script.",
        ]
    )
    del root
    return "\n".join(lines)


def _append_candidates(
    lines: list[str],
    candidates: list[Candidate],
    *,
    target: str,
    kind: str,
    heading: str,
) -> None:
    matching = [candidate for candidate in candidates if candidate.target == target and candidate.kind == kind]
    lines.append(f"- {heading}:")
    if not matching:
        lines.append("  - none")
        return
    for candidate in matching:
        lines.extend(
            [
                f"  - source_id: {candidate.source_id}",
                f"    raw_id: {candidate.raw_id}",
                f"    table: {candidate.table}",
                f"    label/name: {candidate.label}",
                f"    safe_count: {candidate.count if candidate.count is not None else '-'}",
                f"    confidence: {candidate.confidence}",
                f"    reason: {candidate.reason}",
                f"    db: {candidate.db_label}",
            ]
        )


def _first_source(candidates: list[Candidate], *, target: str, kind: str) -> str | None:
    matching = [candidate for candidate in candidates if candidate.target == target and candidate.kind == kind]
    if len(matching) == 1:
        return matching[0].source_id
    return None


if __name__ == "__main__":
    raise SystemExit(main())
