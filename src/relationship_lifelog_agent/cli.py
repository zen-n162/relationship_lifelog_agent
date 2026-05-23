from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import sys
from typing import Any
from uuid import uuid4

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
from relationship_lifelog_agent.full_context.batch_builder import (
    build_chronological_batches,
    build_hybrid_batches,
    build_source_then_time_batches,
    validate_batch_coverage,
)
from relationship_lifelog_agent.full_context.context_budget import decide_context_mode
from relationship_lifelog_agent.full_context.full_range_analyzer import (
    FullRangeAnalysisError,
    analyze_single_context,
)
from relationship_lifelog_agent.full_context.manifest import build_full_data_manifest
from relationship_lifelog_agent.full_context.prompt_packer import build_single_context_prompt
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.status import render_llm_status_json, render_llm_status_text, run_llm_status
from relationship_lifelog_agent.privacy.raw_payload_policy import from_config as raw_payload_policy_from_config
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
    if _is_llm_command(args):
        _llm_main(args)
        return
    if _is_full_context_command(args):
        _full_context_main(args)
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


def _is_llm_command(args: list[str]) -> bool:
    return "llm" in args


def _is_full_context_command(args: list[str]) -> bool:
    return "full-context" in args


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
            duplicates = repo.find_active_duplicate_profiles(
                profile_name=args.profile_name,
                relationship_label=args.relationship_label,
                visibility=args.visibility,
            )
            if duplicates:
                ids = ", ".join(str(item["id"]) for item in duplicates)
                print(f"warning: duplicate active profile candidate exists: id={ids}")
            profile_id = repo.create_profile(
                profile_name=args.profile_name,
                person_source_id=args.person_source_id,
                line_speaker_source_id=args.line_speaker_source_id,
                line_speaker_group_source_id=args.line_speaker_group_source_id,
                self_person_source_id=args.self_person_source_id,
                self_line_speaker_source_id=args.self_line_speaker_source_id,
                self_line_speaker_group_source_id=args.self_line_speaker_group_source_id,
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
            profile_id = _resolve_profile_update_id(repo, args)
            fields = _update_fields(args)
            if not fields:
                print("no profile fields changed")
                return
            changed = repo.update_profile(profile_id, **fields)
            if changed == 0:
                raise SystemExit(f"profile not found or unchanged: {profile_id}")
            print(f"updated profile id={profile_id}")
        elif args.profile_command == "deactivate":
            changed = repo.update_profile(args.id, visibility="hidden")
            if changed == 0:
                raise SystemExit(f"profile not found or unchanged: {args.id}")
            print(f"deactivated profile id={args.id}")
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


def _llm_main(argv: list[str]) -> None:
    parser = _build_llm_parser()
    args = parser.parse_args(argv)
    if args.llm_command == "status":
        report = run_llm_status(args.config)
        if args.format == "json":
            print(render_llm_status_json(report))
        else:
            print(render_llm_status_text(report))
        return
    parser.error("unknown llm command")


def _full_context_main(argv: list[str]) -> None:
    parser = _build_full_context_parser()
    args = parser.parse_args(argv)
    settings = load_config(args.config)
    if args.full_context_command in {"plan", "pack-preview", "analyze"}:
        run_id = f"full-context-plan-{uuid4().hex[:12]}"
        manifest = build_full_data_manifest(
            run_id=run_id,
            profile_id=args.profile_id,
            date_from=args.date_from,
            date_to=args.date_to,
            line_items=(),
            note_items=(),
            media_items=(),
            face_items=(),
            location_items=(),
        )
        recommended_mode = decide_context_mode(manifest, settings.llm)
        manifest = replace(
            manifest,
            can_fit_single_context=recommended_mode == "single_context",
            recommended_mode=recommended_mode,
        )
    if args.full_context_command == "plan":
        full_scan = settings.analysis.full_scan
        batch_builder = {
            "chronological": build_chronological_batches,
            "source_then_time": build_source_then_time_batches,
            "hybrid": build_hybrid_batches,
        }[full_scan.batch_strategy]
        batches = batch_builder(
            run_id=run_id,
            max_items_per_batch=full_scan.max_items_per_batch,
            max_chars_per_batch=full_scan.max_chars_per_batch,
            overlap_items=full_scan.overlap_items,
            max_batches_per_run=full_scan.max_batches_per_run,
        )
        coverage = validate_batch_coverage(batches=batches)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "manifest": asdict(manifest),
                        "batch_count": len(batches),
                        "batch_strategy": full_scan.batch_strategy,
                        "coverage": asdict(coverage),
                        "note": "Full Data Access loader is not connected in this phase; counts reflect the provided item set.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            _print_full_context_plan(manifest, len(batches), full_scan.batch_strategy, coverage)
        return
    if args.full_context_command == "pack-preview":
        policy = raw_payload_policy_from_config(settings)
        bundle = build_single_context_prompt(
            question=args.question,
            profile_context={"profile_id": args.profile_id},
            manifest=manifest,
            full_data={
                "line_items": (),
                "note_items": (),
                "media_items": (),
                "face_items": (),
                "location_items": (),
            },
            raw_payload_policy=policy,
        )
        _print_full_context_pack_preview(bundle, manifest, args.max_preview_chars)
        return
    if args.full_context_command == "analyze":
        policy = raw_payload_policy_from_config(settings)
        if args.dry_run == "true":
            _print_full_context_analyze_dry_run(manifest, settings.analysis.mode)
            return
        if not policy.private_full_enabled:
            raise SystemExit("full-context analyze requires analysis.mode private_full_range or private_full_corpus")
        bundle = build_single_context_prompt(
            question=args.question,
            profile_context={"profile_id": args.profile_id},
            manifest=replace(manifest, raw_payload_policy=policy.to_dict()),
            full_data={
                "line_items": (),
                "note_items": (),
                "media_items": (),
                "face_items": (),
                "location_items": (),
            },
            raw_payload_policy=policy,
        )
        try:
            synthesis = analyze_single_context(
                question=args.question,
                manifest=manifest,
                prompt_bundle=bundle,
                llm_client=LocalLlmClient(settings.llm),
                max_llm_calls=args.max_llm_calls,
            )
        except FullRangeAnalysisError as exc:
            raise SystemExit(str(exc)) from exc
        _print_full_context_analysis_result(synthesis)
        return
    if args.full_context_command == "runs":
        repo = RelationshipRepository(settings.paths.relationship_db)
        if args.runs_command == "list":
            runs = repo.list_full_analysis_runs(limit=args.limit)
            _print_full_context_runs(runs)
            return
        if args.runs_command == "show":
            run = repo.get_full_analysis_run(args.id)
            if run is None:
                raise SystemExit(f"full analysis run not found: {args.id}")
            batches = repo.list_full_analysis_batches(run_id=args.id)
            observations = repo.list_full_analysis_observations(run_id=args.id)
            _print_full_context_run(run, batches, observations)
            return
    parser.error("unknown full-context command")


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


def _build_llm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relationship Lifelog Agent local LLM CLI.")
    parser.add_argument("--config", default=None, help="Optional private config path.")
    subparsers = parser.add_subparsers(dest="resource", required=True)
    llm = subparsers.add_parser("llm", help="Check local LLM configuration.")
    llm_sub = llm.add_subparsers(dest="llm_command", required=True)
    status = llm_sub.add_parser("status", help="Check Ollama/local LLM status without sending private data.")
    status.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _build_full_context_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relationship Lifelog Agent full-context planning CLI.")
    parser.add_argument("--config", default=None, help="Optional private config path.")
    subparsers = parser.add_subparsers(dest="resource", required=True)
    full_context = subparsers.add_parser("full-context", help="Plan private full-context analysis without raw output.")
    full_context_sub = full_context.add_subparsers(dest="full_context_command", required=True)
    plan = full_context_sub.add_parser("plan", help="Build a manifest, context budget, and batch plan summary.")
    plan.add_argument("--profile-id", required=True, type=int)
    plan.add_argument("--date-from", required=True)
    plan.add_argument("--date-to", required=True)
    plan.add_argument("--format", choices=("text", "json"), default="text")
    preview = full_context_sub.add_parser("pack-preview", help="Build a safe prompt preview without loading raw upstream data.")
    preview.add_argument("--profile-id", required=True, type=int)
    preview.add_argument("--date-from", required=True)
    preview.add_argument("--date-to", required=True)
    preview.add_argument("--question", default="Summarize the selected private full context range.")
    preview.add_argument("--max-preview-chars", type=int, default=2000)
    analyze = full_context_sub.add_parser("analyze", help="Analyze a private full-context range with the local LLM.")
    analyze.add_argument("--profile-id", required=True, type=int)
    analyze.add_argument("--date-from", required=True)
    analyze.add_argument("--date-to", required=True)
    analyze.add_argument("--question", required=True)
    analyze.add_argument("--dry-run", choices=("true", "false"), default="true")
    analyze.add_argument("--max-llm-calls", type=int, default=5)
    runs = full_context_sub.add_parser("runs", help="Inspect saved full analysis runs without raw payload output.")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_sub.add_parser("list", help="List saved full analysis runs.")
    runs_list.add_argument("--limit", type=int, default=50)
    runs_show = runs_sub.add_parser("show", help="Show one saved full analysis run.")
    runs_show.add_argument("--id", required=True, type=int)
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
    create.add_argument("--line-speaker-group-source-id", default=None)
    create.add_argument("--self-person-source-id", default=None)
    create.add_argument("--self-line-speaker-source-id", default=None)
    create.add_argument("--self-line-speaker-group-source-id", default=None)
    create.add_argument("--relationship-label", choices=sorted(ALLOWED_RELATIONSHIP_LABELS), default=None)
    create.add_argument("--valid-from", default=None)
    create.add_argument("--valid-to", default=None)
    create.add_argument("--visibility", choices=("private", "hidden"), default="private")
    create.add_argument("--notes", default=None)

    show = profile_sub.add_parser("show", help="Show one manually configured profile.")
    show.add_argument("--id", required=True, type=int)

    update = profile_sub.add_parser("update", help="Update a manually configured profile.")
    update.add_argument("--id", type=int, default=None)
    update.add_argument("--profile-name", default=None, help="Profile name to update, or new name when --id is used.")
    update.add_argument("--new-profile-name", default=None, help="Rename the profile while using --profile-name as lookup.")
    update.add_argument("--person-source-id", default=None)
    update.add_argument("--line-speaker-source-id", default=None)
    update.add_argument("--line-speaker-group-source-id", default=None)
    update.add_argument("--self-person-source-id", default=None)
    update.add_argument("--self-line-speaker-source-id", default=None)
    update.add_argument("--self-line-speaker-group-source-id", default=None)
    update.add_argument("--relationship-label", choices=sorted(ALLOWED_RELATIONSHIP_LABELS), default=None)
    update.add_argument("--valid-from", default=None)
    update.add_argument("--valid-to", default=None)
    update.add_argument("--visibility", choices=("private", "hidden"), default=None)
    update.add_argument("--notes", default=None)

    deactivate = profile_sub.add_parser("deactivate", help="Hide a duplicate or obsolete manual profile.")
    deactivate.add_argument("--id", required=True, type=int)
    return parser


def _resolve_profile_update_id(repo: RelationshipRepository, args: argparse.Namespace) -> int:
    if args.id is not None:
        return int(args.id)
    if args.profile_name:
        profile = repo.get_profile_by_name(args.profile_name)
        if profile is None:
            raise SystemExit(f"profile not found: {args.profile_name}")
        return int(profile["id"])
    raise SystemExit("profile update requires --id or --profile-name")


def _update_fields(args: argparse.Namespace) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if getattr(args, "new_profile_name", None):
        fields["profile_name"] = args.new_profile_name
    elif getattr(args, "id", None) is not None and getattr(args, "profile_name", None) is not None:
        fields["profile_name"] = args.profile_name
    for attr, field in (
        ("person_source_id", "person_source_id"),
        ("line_speaker_source_id", "line_speaker_source_id"),
        ("line_speaker_group_source_id", "line_speaker_group_source_id"),
        ("self_person_source_id", "self_person_source_id"),
        ("self_line_speaker_source_id", "self_line_speaker_source_id"),
        ("self_line_speaker_group_source_id", "self_line_speaker_group_source_id"),
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
    print(
        "id\tprofile_name\tperson_source_id\tline_speaker_source_id\tline_speaker_group_source_id\t"
        "self_person_source_id\tself_line_speaker_source_id\tself_line_speaker_group_source_id\t"
        "relationship_label\tlabel_source\tvalid_from\tvalid_to\tvisibility"
    )
    for profile in profiles:
        print(
            "\t".join(
                _display(profile.get(key))
                for key in (
                    "id",
                    "profile_name",
                    "person_source_id",
                    "line_speaker_source_id",
                    "line_speaker_group_source_id",
                    "self_person_source_id",
                    "self_line_speaker_source_id",
                    "self_line_speaker_group_source_id",
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
        "line_speaker_group_source_id",
        "self_person_source_id",
        "self_line_speaker_source_id",
        "self_line_speaker_group_source_id",
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


def _print_full_context_plan(manifest: Any, batch_count: int, batch_strategy: str, coverage: Any) -> None:
    print("Full Context Plan")
    print(f"run_id: {manifest.run_id}")
    print(f"profile_id: {manifest.profile_id}")
    print(f"date_range: {manifest.date_from}..{manifest.date_to}")
    print("manifest summary:")
    print(f"- line_count: {manifest.line_count}")
    print(f"- note_count: {manifest.note_count}")
    print(f"- media_count: {manifest.media_count}")
    print(f"- face_count: {manifest.face_count}")
    print(f"- location_count: {manifest.location_count}")
    print(f"- date_coverage: {manifest.date_coverage}")
    print(f"- available_payload_types: {', '.join(manifest.available_payload_types) or 'none'}")
    print(f"estimated_tokens: {manifest.estimated_tokens}")
    print(f"estimated_chars: {manifest.estimated_chars}")
    print(f"recommended_mode: {manifest.recommended_mode}")
    print(f"can_fit_single_context: {str(manifest.can_fit_single_context).lower()}")
    print(f"batch_strategy: {batch_strategy}")
    print(f"batch_count: {batch_count}")
    print("coverage check:")
    print(f"- ok: {str(coverage.ok).lower()}")
    print(f"- total_source_refs: {coverage.total_source_refs}")
    print(f"- covered_source_refs: {coverage.covered_source_refs}")
    print(f"- missing_source_refs: {len(coverage.missing_source_refs)}")
    print("note: Full Data Access loader is not connected in this phase; no raw upstream data was loaded.")


def _print_full_context_pack_preview(bundle: Any, manifest: Any, max_preview_chars: int) -> None:
    limit = max(0, max_preview_chars)
    included_payload_types = tuple(bundle.metadata.get("included_payload_types", ()))
    print("Full Context Prompt Pack Preview")
    print(f"bundle_id: {bundle.bundle_id}")
    print(f"profile_id: {manifest.profile_id}")
    print(f"date_range: {manifest.date_from}..{manifest.date_to}")
    print(f"prompt_size_chars: {bundle.estimated_chars}")
    print(f"prompt_size_tokens_estimate: {bundle.estimated_tokens}")
    print(f"included_payload_types: {', '.join(included_payload_types) or 'none'}")
    print("source counts:")
    print(f"- line_count: {manifest.line_count}")
    print(f"- note_count: {manifest.note_count}")
    print(f"- media_count: {manifest.media_count}")
    print(f"- face_count: {manifest.face_count}")
    print(f"- location_count: {manifest.location_count}")
    print(f"raw_text_included: {str(bool(bundle.metadata.get('raw_text_included'))).lower()}")
    print(f"prompt_logging_allowed: {str(bool(bundle.metadata.get('prompt_logging_allowed'))).lower()}")
    print(f"raw_payload_cache_allowed: {str(bool(bundle.metadata.get('raw_payload_cache_allowed'))).lower()}")
    print(f"preview_chars: {min(limit, len(bundle.prompt_text))}")
    print("preview:")
    print(bundle.prompt_text[:limit])
    if len(bundle.prompt_text) > limit:
        print("[preview truncated; full prompt not displayed]")
    else:
        print("[preview complete; no raw upstream data was loaded]")


def _print_full_context_analyze_dry_run(manifest: Any, analysis_mode: str) -> None:
    print("Full Context Analyze Dry Run")
    print(f"profile_id: {manifest.profile_id}")
    print(f"date_range: {manifest.date_from}..{manifest.date_to}")
    print(f"analysis_mode: {analysis_mode}")
    print(f"recommended_mode: {manifest.recommended_mode}")
    print(f"estimated_tokens: {manifest.estimated_tokens}")
    print("llm_calls: 0")
    print("note: dry-run true; local LLM was not called and no raw upstream data was loaded.")


def _print_full_context_analysis_result(synthesis: Any) -> None:
    print("Full Context Analysis Result")
    print(f"summary: {synthesis.summary}")
    print(f"confidence: {synthesis.confidence:.3f}")
    print(f"source_refs: {len(synthesis.source_refs)}")
    print(f"uncertainties: {len(synthesis.uncertainties)}")
    if synthesis.uncertainties:
        for item in synthesis.uncertainties:
            print(f"- caution: {item}")
    print("answer:")
    print(synthesis.answer)


def _print_full_context_runs(runs: list[dict[str, Any]]) -> None:
    if not runs:
        print("no full analysis runs")
        return
    print("id\tstatus\tanalysis_mode\tprofile_id\tdate_from\tdate_to\tmodel_name\tcreated_at")
    for run in runs:
        print(
            "\t".join(
                _display(run.get(key))
                for key in (
                    "id",
                    "status",
                    "analysis_mode",
                    "profile_id",
                    "date_from",
                    "date_to",
                    "model_name",
                    "created_at",
                )
            )
        )


def _print_full_context_run(
    run: dict[str, Any],
    batches: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> None:
    print(f"id: {_display(run.get('id'))}")
    print(f"question: {_display(run.get('question'))}")
    print(f"analysis_mode: {_display(run.get('analysis_mode'))}")
    print(f"profile_id: {_display(run.get('profile_id'))}")
    print(f"date_range: {_display(run.get('date_from'))}..{_display(run.get('date_to'))}")
    print(f"status: {_display(run.get('status'))}")
    print(f"model_name: {_display(run.get('model_name'))}")
    manifest = run.get("manifest") if isinstance(run.get("manifest"), dict) else {}
    print(f"manifest_keys: {', '.join(sorted(manifest)) if manifest else 'none'}")
    print(f"final_synthesis_present: {str(bool(run.get('final_synthesis'))).lower()}")
    print(f"batch_count: {len(batches)}")
    print(f"observation_count: {len(observations)}")


if __name__ == "__main__":
    main()
