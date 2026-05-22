from __future__ import annotations

from dataclasses import replace
from typing import Any

from relationship_lifelog_agent.agent.executor import answer_question
from relationship_lifelog_agent.agent.router import route_question
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.privacy.guard import sanitize_answer
from relationship_lifelog_agent.profiles import default_profile_choice, parse_profile_choice, profile_choices
from relationship_lifelog_agent.ui.components import SUGGESTED_QUESTIONS


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

        def user_turn(
            user_message: str,
            history: list[dict[str, str]] | None,
        ) -> tuple[str, list[dict[str, str]]]:
            history = history or []
            if not user_message.strip():
                return "", history
            history = [*history, {"role": "user", "content": user_message}]
            return "", history

        def bot_turn(
            history: list[dict[str, str]] | None,
            selected_backend: str,
            selected_profile: str | None,
            selected_date_from: str | None,
            selected_date_to: str | None,
            selected_window_days: int | float | None,
            selected_mode: str,
            show_debug: bool,
        ) -> list[dict[str, str]]:
            history = history or []
            if not history:
                return history
            question = history[-1]["content"]
            answer = build_ui_answer(
                question,
                base_settings=settings,
                selected_backend=selected_backend,
                selected_profile=selected_profile,
                date_from=selected_date_from,
                date_to=selected_date_to,
                post_conflict_window_days=selected_window_days,
                mode=selected_mode,
                show_debug=show_debug,
            )
            return [*history, {"role": "assistant", "content": answer}]

        send.click(user_turn, [message, chatbot], [message, chatbot]).then(
            bot_turn,
            [chatbot, backend, profile, date_from, date_to, post_conflict_window_days, mode, debug],
            chatbot,
        )
        message.submit(user_turn, [message, chatbot], [message, chatbot]).then(
            bot_turn,
            [chatbot, backend, profile, date_from, date_to, post_conflict_window_days, mode, debug],
            chatbot,
        )
        clear.click(lambda: [], None, chatbot)

    return demo


def build_ui_answer(
    question: str,
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
    settings = _settings_for_ui(base_settings, selected_backend, post_conflict_window_days)
    profile_id = parse_profile_choice(selected_profile)
    answer = answer_question(
        question,
        mode=mode,
        settings=settings,
        profile_id=profile_id,
        date_from=_clean_text(date_from),
        date_to=_clean_text(date_to),
        post_conflict_window_days=_safe_window_days(post_conflict_window_days),
    )
    if show_debug:
        route = route_question(question)
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
    return sanitize_answer(answer, mode=mode)


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
