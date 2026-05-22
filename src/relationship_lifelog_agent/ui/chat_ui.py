from __future__ import annotations

from typing import Any

from relationship_lifelog_agent.agent.executor import answer_question
from relationship_lifelog_agent.agent.router import route_question
from relationship_lifelog_agent.config import Settings
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
            profile = gr.Dropdown(
                choices=profile_options,
                value=default_profile,
                label="Target profile",
                interactive=True,
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
            selected_profile: str | None,
            selected_mode: str,
            show_debug: bool,
        ) -> list[dict[str, str]]:
            history = history or []
            if not history:
                return history
            question = history[-1]["content"]
            answer = answer_question(
                question,
                mode=selected_mode,
                settings=settings,
                profile_id=parse_profile_choice(selected_profile),
            )
            if show_debug:
                route = route_question(question)
                answer += (
                    "\n\n<details>\n<summary>Debug</summary>\n\n"
                    f"- intents: {', '.join(route.intents)}\n"
                    f"- adapter: {settings.adapter.backend}\n\n"
                    f"- profile_id: {selected_profile or 'unset'}\n\n"
                    "</details>"
                )
            return [*history, {"role": "assistant", "content": answer}]

        send.click(user_turn, [message, chatbot], [message, chatbot]).then(
            bot_turn, [chatbot, profile, mode, debug], chatbot
        )
        message.submit(user_turn, [message, chatbot], [message, chatbot]).then(
            bot_turn, [chatbot, profile, mode, debug], chatbot
        )
        clear.click(lambda: [], None, chatbot)

    return demo
