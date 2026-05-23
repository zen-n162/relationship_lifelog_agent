from __future__ import annotations

from dataclasses import replace
from typing import Any

from relationship_lifelog_agent.agent.executor import ChatAnswer, ReviewTarget, answer_chat
from relationship_lifelog_agent.agent.router import route_question
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.privacy.guard import sanitize_answer
from relationship_lifelog_agent.profiles import default_profile_choice, parse_profile_choice, profile_choices
from relationship_lifelog_agent.review.actions import apply_review_action
from relationship_lifelog_agent.ui.components import SUGGESTED_QUESTIONS


REVIEW_ACTION_CHOICES = (
    ("確認済みにする", "verify"),
    ("これは違う", "reject"),
    ("軽いすれ違いにする", "mark_as_misunderstanding"),
    ("冗談として扱う", "mark_as_joke"),
    ("仲直り済みにする", "mark_as_reconciled"),
    ("再分析対象にする", "needs_reanalysis"),
)


def build_chat_ui(settings: Settings) -> Any:
    import gradio as gr

    profile_options = profile_choices(settings)
    default_profile = default_profile_choice(settings, profile_options)

    with gr.Blocks(title=settings.app.name, analytics_enabled=False) as demo:
        gr.Markdown("# Relationship Lifelog Agent")
        chatbot = gr.Chatbot(
            label="Chat",
            height=560,
            buttons=["copy", "copy_all"],
            allow_file_downloads=False,
        )
        with gr.Row():
            message = gr.Textbox(
                label="Message",
                placeholder=SUGGESTED_QUESTIONS[0],
                lines=2,
                scale=8,
                show_label=False,
            )
            send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Clear", scale=1)
        with gr.Accordion("Settings", open=False):
            backend = gr.Dropdown(
                choices=["mock", "upstream_readonly"],
                value=settings.adapter.backend,
                label="Backend",
                interactive=True,
            )
            profile = gr.Dropdown(
                choices=profile_options,
                value=default_profile,
                label="Target profile",
                interactive=True,
            )
            with gr.Row():
                date_from = gr.Textbox(label="date_from", placeholder="2025-01-01", value="")
                date_to = gr.Textbox(label="date_to", placeholder="2025-03-31", value="")
            post_conflict_window_days = gr.Number(
                value=settings.relationship.post_conflict_window_days,
                label="Post-conflict window days",
                precision=0,
                minimum=1,
            )
            mode = gr.Radio(["private", "public"], value=settings.app.mode, label="Mode")
            debug = gr.Checkbox(value=settings.ui.show_debug_by_default, label="Debug output")
        with gr.Accordion("Review last answer", open=False):
            review_targets_md = gr.Markdown("レビュー対象はまだありません。")
            review_target = gr.Dropdown(
                choices=[],
                value=None,
                label="Review target",
                interactive=True,
            )
            review_action = gr.Dropdown(
                choices=list(REVIEW_ACTION_CHOICES),
                value="verify",
                label="Action",
                interactive=True,
            )
            review_note = gr.Textbox(label="Note", lines=1, placeholder="任意メモ", value="")
            save_review = gr.Button("Save review action", variant="secondary")
            review_status = gr.Markdown("")
            review_targets_state = gr.State([])
            db_path_state = gr.State(settings.paths.relationship_db)

        def respond(
            user_message: Any,
            history: list[Any] | None,
            selected_backend: str,
            selected_profile: str | None,
            selected_date_from: str | None,
            selected_date_to: str | None,
            selected_window_days: int | float | None,
            selected_mode: str,
            show_debug: bool,
        ) -> tuple[str, list[dict[str, str]], Any, str, list[dict[str, object]]]:
            updated_history, chat_answer = build_ui_chat_response(
                user_message,
                history,
                base_settings=settings,
                selected_backend=selected_backend,
                selected_profile=selected_profile,
                date_from=selected_date_from,
                date_to=selected_date_to,
                post_conflict_window_days=selected_window_days,
                mode=selected_mode,
                show_debug=show_debug,
            )
            if chat_answer is None:
                return "", updated_history, gr.update(choices=[], value=None), "レビュー対象はまだありません。", []
            target_state = _target_state(chat_answer.review_targets)
            return (
                "",
                updated_history,
                gr.update(choices=_target_choices(chat_answer.review_targets), value=_first_target_value(chat_answer.review_targets)),
                _targets_markdown(chat_answer.review_targets),
                target_state,
            )

        send.click(
            respond,
            [message, chatbot, backend, profile, date_from, date_to, post_conflict_window_days, mode, debug],
            [message, chatbot, review_target, review_targets_md, review_targets_state],
        )
        message.submit(
            respond,
            [message, chatbot, backend, profile, date_from, date_to, post_conflict_window_days, mode, debug],
            [message, chatbot, review_target, review_targets_md, review_targets_state],
        )
        save_review.click(
            save_review_action_from_ui,
            [review_target, review_action, review_note, review_targets_state, db_path_state],
            review_status,
        )
        clear.click(
            lambda: ([], gr.update(choices=[], value=None), "レビュー対象はまだありません。", [], ""),
            None,
            [chatbot, review_target, review_targets_md, review_targets_state, review_status],
        )

    return demo


def build_ui_chat_response(
    user_message: Any,
    history: list[Any] | None,
    *,
    base_settings: Settings,
    selected_backend: str,
    selected_profile: str | None,
    date_from: str | None,
    date_to: str | None,
    post_conflict_window_days: int | float | None,
    mode: str,
    show_debug: bool = False,
) -> tuple[list[dict[str, str]], ChatAnswer | None]:
    normalized_history = _normalize_history(history)
    question = _coerce_message_text(user_message)
    if not question:
        return normalized_history, None
    chat_answer = build_ui_chat_answer(
        question,
        base_settings=base_settings,
        selected_backend=selected_backend,
        selected_profile=selected_profile,
        date_from=date_from,
        date_to=date_to,
        post_conflict_window_days=post_conflict_window_days,
        mode=mode,
        show_debug=show_debug,
    )
    return (
        [
            *normalized_history,
            {"role": "user", "content": question},
            {"role": "assistant", "content": chat_answer.text},
        ],
        chat_answer,
    )


def build_ui_chat_answer(
    question: Any,
    *,
    base_settings: Settings,
    selected_backend: str,
    selected_profile: str | None,
    date_from: str | None,
    date_to: str | None,
    post_conflict_window_days: int | float | None,
    mode: str,
    show_debug: bool = False,
) -> ChatAnswer:
    question_text = _coerce_message_text(question)
    settings = _settings_for_ui(base_settings, selected_backend, post_conflict_window_days)
    profile_id = parse_profile_choice(selected_profile)
    chat_answer = answer_chat(
        question_text,
        mode=mode,
        settings=settings,
        profile_id=profile_id,
        date_from=_clean_text(date_from),
        date_to=_clean_text(date_to),
        post_conflict_window_days=_safe_window_days(post_conflict_window_days),
    )
    answer = chat_answer.text
    if show_debug:
        route = route_question(question_text)
        answer += (
            "\n\n<details>\n<summary>Debug</summary>\n\n"
            f"- intents: {', '.join(route.intents)}\n"
            f"- adapter: {settings.adapter.backend}\n"
            f"- profile_id: {profile_id if profile_id is not None else 'unset'}\n"
            f"- date_from: {_clean_text(date_from) or 'unset'}\n"
            f"- date_to: {_clean_text(date_to) or 'unset'}\n"
            f"- post_conflict_window_days: {_safe_window_days(post_conflict_window_days)}\n\n"
            "</details>"
        )
    return ChatAnswer(
        text=sanitize_answer(answer, mode=mode),
        review_targets=chat_answer.review_targets,
    )


def build_ui_answer(
    question: Any,
    *,
    base_settings: Settings,
    selected_backend: str,
    selected_profile: str | None,
    date_from: str | None,
    date_to: str | None,
    post_conflict_window_days: int | float | None,
    mode: str,
    show_debug: bool = False,
) -> str:
    return build_ui_chat_answer(
        question,
        base_settings=base_settings,
        selected_backend=selected_backend,
        selected_profile=selected_profile,
        date_from=date_from,
        date_to=date_to,
        post_conflict_window_days=post_conflict_window_days,
        mode=mode,
        show_debug=show_debug,
    ).text


def _normalize_history(history: list[Any] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            if role not in {"user", "assistant"}:
                role = "user"
            content = _coerce_message_text(item.get("content"))
            if content:
                messages.append({"role": role, "content": content})
            continue
        if isinstance(item, (list, tuple)):
            if item:
                user_text = _coerce_message_text(item[0])
                if user_text:
                    messages.append({"role": "user", "content": user_text})
            if len(item) > 1:
                assistant_text = _coerce_message_text(item[1])
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
            continue
        content = _coerce_message_text(item)
        if content:
            messages.append({"role": "user", "content": content})
    return messages


def _latest_user_message(history: list[Any] | None) -> str:
    for item in reversed(_normalize_history(history)):
        if item.get("role") == "user":
            return item["content"]
    return ""


def _coerce_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _coerce_message_text(value.get("content"))
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _coerce_message_text(item)
            if text:
                return text
        return ""
    return str(value).strip()


def save_review_action_from_ui(
    selected_target: str | int | None,
    selected_action: str | None,
    note: str | None,
    targets: list[dict[str, object]] | None,
    relationship_db_path: str,
) -> str:
    del targets
    if selected_target in (None, ""):
        return "レビュー対象を選択してください。"
    if not selected_action:
        return "review actionを選択してください。"
    event_id = int(selected_target)
    repo = RelationshipRepository(relationship_db_path)
    try:
        result = apply_review_action(
            repo,
            event_id=event_id,
            action=str(selected_action),
            note=_clean_text(note),
        )
    except ValueError as exc:
        return sanitize_answer(f"レビュー保存に失敗しました: {exc}")
    return sanitize_answer(
        "レビューを保存しました: "
        f"event_id={result.event_id}, action={result.action}, "
        f"review_status={result.review_status}, status={result.status}"
    )


def _target_choices(targets: tuple[ReviewTarget, ...]) -> list[tuple[str, str]]:
    return [(target.label, str(target.event_id)) for target in targets]


def _first_target_value(targets: tuple[ReviewTarget, ...]) -> str | None:
    if not targets:
        return None
    return str(targets[0].event_id)


def _target_state(targets: tuple[ReviewTarget, ...]) -> list[dict[str, object]]:
    return [
        {
            "event_id": target.event_id,
            "label": target.label,
            "event_type": target.event_type,
            "review_status": target.review_status,
        }
        for target in targets
    ]


def _targets_markdown(targets: tuple[ReviewTarget, ...]) -> str:
    if not targets:
        return "レビュー対象はありません。"
    lines = ["最後の回答に含まれるレビュー対象:"]
    lines.extend(
        f"- `{target.event_id}` {target.label} ({target.event_type}, {target.review_status})"
        for target in targets
    )
    return "\n".join(lines)


def _settings_for_ui(
    settings: Settings,
    selected_backend: str,
    post_conflict_window_days: int | float | None,
) -> Settings:
    backend = selected_backend if selected_backend in {"mock", "upstream_readonly"} else settings.adapter.backend
    window_days = _safe_window_days(post_conflict_window_days)
    return replace(
        settings,
        adapter=replace(settings.adapter, backend=backend),
        relationship=replace(settings.relationship, post_conflict_window_days=window_days),
    )


def _safe_window_days(value: int | float | None) -> int:
    if value is None:
        return 14
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 14


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
