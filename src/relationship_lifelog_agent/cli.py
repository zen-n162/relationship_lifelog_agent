from __future__ import annotations

import argparse
import sys
from typing import Any

from relationship_lifelog_agent.app import main as app_main
from relationship_lifelog_agent.config import load_config
from relationship_lifelog_agent.db.repository import ALLOWED_RELATIONSHIP_LABELS, RelationshipRepository


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if _is_profile_command(args):
        _profile_main(args)
        return
    app_main(args)


def _is_profile_command(args: list[str]) -> bool:
    return "profile" in args


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
