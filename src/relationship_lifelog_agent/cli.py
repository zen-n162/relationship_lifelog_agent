from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from typing import Any

from relationship_lifelog_agent.agent.memory import build_memory
from relationship_lifelog_agent.analytics.dry_run import (
    PRIVACY_LEVELS,
    assess_write_safety,
    run_relationship_dry_run,
    scan_created_write_rows,
    write_dry_run_candidates,
)
from relationship_lifelog_agent.app import main as app_main
from relationship_lifelog_agent.config import load_config
from relationship_lifelog_agent.db.backup import (
    create_relationship_db_backup,
    list_relationship_db_backups,
    restore_relationship_db_backup,
)
from relationship_lifelog_agent.db.repository import ALLOWED_RELATIONSHIP_LABELS, RelationshipRepository
from relationship_lifelog_agent.doctor import render_doctor_json, render_doctor_text, run_doctor
from relationship_lifelog_agent.profiles import load_profile_context
from relationship_lifelog_agent.upstream_identities import (
    IDENTITY_KINDS,
    IDENTITY_PRIVACY_LEVELS,
    render_identities_json,
    render_identities_markdown,
    run_upstream_identities,
    write_identities_report,
)
from relationship_lifelog_agent.upstream_inspect import (
    render_inspection_json,
    render_inspection_markdown,
    run_upstream_inspection,
    write_inspection_report,
)
from relationship_lifelog_agent.upstream_smoke import (
    render_smoke_json,
    render_smoke_markdown,
    run_upstream_smoke,
    write_smoke_report,
)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if _is_doctor_command(args):
        _doctor_main(args)
        return
    if _is_upstream_command(args):
        _upstream_main(args)
        return
    if _is_db_command(args):
        _db_main(args)
        return
    if _is_profile_command(args):
        _profile_main(args)
        return
    if _is_analyze_command(args):
        _analyze_main(args)
        return
    app_main(args)


def _is_doctor_command(args: list[str]) -> bool:
    return "doctor" in args


def _is_upstream_command(args: list[str]) -> bool:
    return "upstream" in args


def _is_db_command(args: list[str]) -> bool:
    return "db" in args


def _is_profile_command(args: list[str]) -> bool:
    return "profile" in args


def _is_analyze_command(args: list[str]) -> bool:
    return "analyze" in args


def _doctor_main(argv: list[str]) -> None:
    parser = _build_doctor_parser()
    args = parser.parse_args(argv)
    report = run_doctor(config_path=args.config, backend=args.backend)
    if args.format == "json":
        print(render_doctor_json(report))
    else:
        print(render_doctor_text(report))
    if report.has_errors:
        raise SystemExit(1)


def _upstream_main(argv: list[str]) -> None:
    parser = _build_upstream_parser()
    args = parser.parse_args(argv)
    if args.upstream_command == "inspect":
        report = run_upstream_inspection(config_path=args.config, backend=args.backend)
        if args.output:
            write_inspection_report(report, args.output, output_format=args.format)
            print("upstream schema inspection written: [redacted_path]")
        else:
            if args.format == "json":
                print(render_inspection_json(report))
            else:
                print(render_inspection_markdown(report))
        if report.has_errors:
            raise SystemExit(1)
        return
    if args.upstream_command == "smoke":
        report = run_upstream_smoke(
            config_path=args.config,
            backend=args.backend,
            date_from=args.date_from,
            date_to=args.date_to,
            profile_id=args.profile_id,
        )
        if args.output:
            write_smoke_report(report, args.output, output_format=args.format)
            print("upstream smoke report written: [redacted_path]")
        else:
            if args.format == "json":
                print(render_smoke_json(report))
            else:
                print(render_smoke_markdown(report))
        if report.has_errors:
            raise SystemExit(1)
        return
    if args.upstream_command == "identities":
        report = run_upstream_identities(
            config_path=args.config,
            backend=args.backend,
            kind=args.kind,
            privacy_level=args.privacy_level,
        )
        if args.output:
            write_identities_report(report, args.output, output_format=args.format)
            print("upstream identity inventory written: [redacted_path]")
        else:
            if args.format == "json":
                print(render_identities_json(report))
            else:
                print(render_identities_markdown(report))
        if report.has_errors:
            raise SystemExit(1)
        return
    parser.error("unknown upstream command")


def _profile_main(argv: list[str]) -> None:
    parser = _build_profile_parser()
    args = parser.parse_args(argv)
    settings = load_config(args.config)
    repo = RelationshipRepository(settings.paths.relationship_db)

    try:
        if args.profile_command == "list":
            _print_profiles(repo.list_profiles())
        elif args.profile_command == "create":
            profile_id = repo.create_profile(
                profile_name=args.profile_name,
                person_source_id=args.person_source_id,
                line_speaker_source_id=args.line_speaker_source_id,
                relationship_label=args.relationship_label,
                valid_from=args.valid_from,
                valid_to=args.valid_to,
                visibility=args.visibility,
                notes=args.notes,
            )
            print(f"created profile id={profile_id}")
        elif args.profile_command == "show":
            profile = repo.get_profile(args.id)
            if profile is None:
                raise SystemExit(f"profile not found: {args.id}")
            _print_profile(profile)
        elif args.profile_command == "update":
            fields = _update_fields(args)
            if not fields:
                print("no profile fields changed")
                return
            changed = repo.update_profile(args.id, **fields)
            if changed == 0:
                raise SystemExit(f"profile not found or unchanged: {args.id}")
            print(f"updated profile id={args.id}")
        else:
            parser.error("unknown profile command")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _db_main(argv: list[str]) -> None:
    parser = _build_db_parser()
    args = parser.parse_args(argv)
    settings = load_config(args.config)
    db_path = settings.paths.relationship_db
    try:
        if args.db_command == "backup":
            RelationshipRepository(db_path)
            backup = create_relationship_db_backup(db_path)
            print("relationship DB backup created: [redacted_path]")
            print(f"backup file: {backup.filename}")
        elif args.db_command == "backups":
            backups = list_relationship_db_backups(db_path)
            print("relationship DB backups:")
            if not backups:
                print("- none")
            for backup in backups:
                print(f"- {backup.filename} size_bytes={backup.size_bytes}")
            print("backup directory: [redacted_path]")
        elif args.db_command == "restore":
            restored = restore_relationship_db_backup(db_path, backup_path=args.backup_path)
            print("relationship DB restored from: [redacted_path]")
            print(f"backup file: {restored.restored_from.name}")
            if restored.pre_restore_backup:
                print("pre-restore backup created: [redacted_path]")
                print(f"pre-restore backup file: {restored.pre_restore_backup.name}")
        else:
            parser.error("unknown db command")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _analyze_main(argv: list[str]) -> None:
    parser = _build_analyze_parser()
    args = parser.parse_args(argv)
    settings = load_config(args.config)
    settings = replace(settings, adapter=replace(settings.adapter, backend=args.backend))
    profile = load_profile_context(settings, args.profile_id)
    if profile is None:
        raise SystemExit(f"profile not found: {args.profile_id}")
    memory = build_memory(settings)
    try:
        result = run_relationship_dry_run(
            memory=memory,
            settings=settings,
            profile=profile,
            date_from=args.date_from,
            date_to=args.date_to,
            backend=args.backend,
            mode=args.mode,
            privacy_level=args.privacy_level,
            output_path=args.output,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output:
        print("dry-run report written: [redacted_path]")
    else:
        print("dry-run report written: none")
    if args.write:
        repo = RelationshipRepository(settings.paths.relationship_db)
        safety = assess_write_safety(
            repo=repo,
            result=result,
            profile_id=args.profile_id,
            date_from=args.date_from,
            date_to=args.date_to,
            mode=args.mode,
            privacy_level=args.privacy_level,
        )
        print(f"write safety candidate events: {safety.candidate_events}")
        print(f"write safety candidate evidence: {safety.candidate_evidence}")
        print(f"write safety candidate post_conflict_activities: {safety.candidate_post_conflict_activities}")
        print(f"write safety preflight duplicate candidates: {safety.duplicate_candidates}")
        print(f"write safety raw leakage issues: {len(safety.raw_leakage_issues)}")
        print(f"write safety forbidden phrase issues: {len(safety.forbidden_phrase_issues)}")
        print("write safety profile configured: true")
        print(f"write safety date range: {args.date_from}..{args.date_to}")
        print(f"write safety privacy_level: {args.privacy_level}")
        for warning in safety.warnings:
            print(f"warning: {warning}")
        if not safety.ok:
            raise SystemExit("write safety failed; relationship DB was not modified")
        backup = create_relationship_db_backup(settings.paths.relationship_db)
        print("pre-write backup path: [redacted_path]")
        print(f"pre-write backup file: {backup.filename}")
        write_result = write_dry_run_candidates(
            repo=repo,
            result=result,
            profile_id=args.profile_id,
            mode=args.mode,
            privacy_level=args.privacy_level,
        )
        post_write_safety = scan_created_write_rows(repo=repo, write_result=write_result, mode=args.mode)
        print(f"relationship_events written: {write_result.events_written}")
        print(f"relationship_event_evidence written: {write_result.evidence_written}")
        print(f"post_conflict_activities written: {write_result.post_conflict_activities_written}")
        print(f"duplicates skipped: {write_result.duplicates}")
        print(f"post-write raw leakage issues: {len(post_write_safety.raw_leakage_issues)}")
        print(f"post-write forbidden phrase issues: {len(post_write_safety.forbidden_phrase_issues)}")
        for warning in write_result.warnings:
            print(f"warning: {warning}")
        if not post_write_safety.ok:
            raise SystemExit("post-write safety failed; use db restore with the pre-write backup if needed")
    else:
        print("relationship_events written: 0")
    print(f"conflict candidates: {len(result.conflict_candidates)}")
    print(f"minor misunderstanding candidates: {len(result.minor_misunderstanding_candidates)}")
    print(f"reconciliation candidates: {len(result.reconciliation_candidates)}")
    print(f"post-conflict outing candidates: {len(result.post_conflict_outings)}")


def _build_analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relationship Lifelog Agent analysis CLI.")
    parser.add_argument("--config", default=None, help="Optional private config path.")
    subparsers = parser.add_subparsers(dest="resource", required=True)
    analyze = subparsers.add_parser("analyze", help="Run relationship analyses.")
    analyze_sub = analyze.add_subparsers(dest="analyze_command", required=True)
    dry_run = analyze_sub.add_parser("dry-run", help="Extract relationship event candidates; write only with --write.")
    dry_run.add_argument("--profile-id", required=True, type=int)
    dry_run.add_argument("--date-from", required=True)
    dry_run.add_argument("--date-to", required=True)
    dry_run.add_argument("--backend", choices=("mock", "upstream_readonly"), default="mock")
    dry_run.add_argument("--mode", choices=("private", "public"), default="private")
    dry_run.add_argument("--privacy-level", choices=PRIVACY_LEVELS, default="redacted")
    dry_run.add_argument("--output", default=None)
    dry_run.add_argument("--write", action="store_true", help="Explicitly save candidates into the relationship DB.")
    return parser


def _build_doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relationship Lifelog Agent doctor CLI.")
    parser.add_argument("--config", default=None, help="Optional private config path.")
    subparsers = parser.add_subparsers(dest="resource", required=True)
    doctor = subparsers.add_parser("doctor", help="Diagnose local config, DBs, adapters, and privacy settings.")
    doctor.add_argument("--backend", choices=("mock", "upstream_readonly"), default=None)
    doctor.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _build_db_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relationship Lifelog Agent relationship DB CLI.")
    parser.add_argument("--config", default=None, help="Optional private config path.")
    subparsers = parser.add_subparsers(dest="resource", required=True)
    db = subparsers.add_parser("db", help="Back up or restore the relationship DB.")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("backup", help="Create a backup of the relationship DB only.")
    db_sub.add_parser("backups", help="List relationship DB backups without printing private paths.")
    restore = db_sub.add_parser("restore", help="Restore relationship DB from an explicit backup path.")
    restore.add_argument("--backup-path", required=True)
    return parser


def _build_upstream_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relationship Lifelog Agent upstream diagnostics CLI.")
    parser.add_argument("--config", default=None, help="Optional private config path.")
    subparsers = parser.add_subparsers(dest="resource", required=True)
    upstream = subparsers.add_parser("upstream", help="Inspect read-only upstream adapter compatibility.")
    upstream_sub = upstream.add_subparsers(dest="upstream_command", required=True)
    inspect = upstream_sub.add_parser("inspect", help="Inspect upstream SQLite schema mapping without raw data output.")
    inspect.add_argument("--backend", choices=("mock", "upstream_readonly"), default="upstream_readonly")
    inspect.add_argument("--format", choices=("markdown", "json"), default="markdown")
    inspect.add_argument("--output", default=None)
    smoke = upstream_sub.add_parser("smoke", help="Run counts-only read-only upstream adapter smoke.")
    smoke.add_argument("--backend", choices=("mock", "upstream_readonly"), default="upstream_readonly")
    smoke.add_argument("--date-from", required=True)
    smoke.add_argument("--date-to", required=True)
    smoke.add_argument("--profile-id", type=int, default=None)
    smoke.add_argument("--format", choices=("markdown", "json"), default="markdown")
    smoke.add_argument("--output", default=None)
    identities = upstream_sub.add_parser("identities", help="List safe upstream identity IDs for manual profile setup.")
    identities.add_argument("--backend", choices=("mock", "upstream_readonly"), default="upstream_readonly")
    identities.add_argument("--kind", choices=IDENTITY_KINDS, default="all")
    identities.add_argument("--privacy-level", choices=IDENTITY_PRIVACY_LEVELS, default="redacted")
    identities.add_argument("--format", choices=("markdown", "json"), default="markdown")
    identities.add_argument("--output", default=None)
    return parser


def _build_profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relationship Lifelog Agent CLI.")
    parser.add_argument("--config", default=None, help="Optional private config path.")
    subparsers = parser.add_subparsers(dest="resource", required=True)
    profile = subparsers.add_parser("profile", help="Manage manually configured relationship profiles.")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)

    profile_sub.add_parser("list", help="List manually configured profiles.")

    create = profile_sub.add_parser("create", help="Create a manually configured profile.")
    create.add_argument("--profile-name", required=True)
    create.add_argument("--person-source-id", default=None)
    create.add_argument("--line-speaker-source-id", default=None)
    create.add_argument("--relationship-label", choices=sorted(ALLOWED_RELATIONSHIP_LABELS), default=None)
    create.add_argument("--valid-from", default=None)
    create.add_argument("--valid-to", default=None)
    create.add_argument("--visibility", choices=("private", "hidden"), default="private")
    create.add_argument("--notes", default=None)

    show = profile_sub.add_parser("show", help="Show one manually configured profile.")
    show.add_argument("--id", required=True, type=int)

    update = profile_sub.add_parser("update", help="Update a manually configured profile.")
    update.add_argument("--id", required=True, type=int)
    update.add_argument("--profile-name", default=None)
    update.add_argument("--person-source-id", default=None)
    update.add_argument("--line-speaker-source-id", default=None)
    update.add_argument("--relationship-label", choices=sorted(ALLOWED_RELATIONSHIP_LABELS), default=None)
    update.add_argument("--valid-from", default=None)
    update.add_argument("--valid-to", default=None)
    update.add_argument("--visibility", choices=("private", "hidden"), default=None)
    update.add_argument("--notes", default=None)
    return parser


def _update_fields(args: argparse.Namespace) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for attr, field in (
        ("profile_name", "profile_name"),
        ("person_source_id", "person_source_id"),
        ("line_speaker_source_id", "line_speaker_source_id"),
        ("relationship_label", "relationship_label"),
        ("valid_from", "valid_from"),
        ("valid_to", "valid_to"),
        ("visibility", "visibility"),
        ("notes", "notes"),
    ):
        value = getattr(args, attr)
        if value is not None:
            fields[field] = value
    return fields


def _print_profiles(profiles: list[dict[str, Any]]) -> None:
    if not profiles:
        print("no profiles")
        return
    print("id\tprofile_name\tperson_source_id\tline_speaker_source_id\trelationship_label\tlabel_source\tvalid_from\tvalid_to\tvisibility")
    for profile in profiles:
        print(
            "\t".join(
                _display(profile.get(key))
                for key in (
                    "id",
                    "profile_name",
                    "person_source_id",
                    "line_speaker_source_id",
                    "relationship_label",
                    "label_source",
                    "valid_from",
                    "valid_to",
                    "visibility",
                )
            )
        )


def _print_profile(profile: dict[str, Any]) -> None:
    for key in (
        "id",
        "profile_name",
        "person_source_id",
        "line_speaker_source_id",
        "relationship_label",
        "label_source",
        "valid_from",
        "valid_to",
        "visibility",
        "created_at",
        "updated_at",
    ):
        print(f"{key}: {_display(profile.get(key))}")


def _display(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    main()
