from __future__ import annotations

from dataclasses import replace
from collections.abc import Iterator
from typing import Any

from relationship_lifelog_agent.agent.agent_runtime import RuntimeBudget
from relationship_lifelog_agent.agent.executor import ChatAnswer, ReviewTarget
from relationship_lifelog_agent.agent.reasoning_orchestrator import answer_with_reasoning, stream_answer
from relationship_lifelog_agent.agent.streaming import AgentStreamEvent
from relationship_lifelog_agent.config import Settings
from relationship_lifelog_agent.db.repository import ALLOWED_RELATIONSHIP_LABELS, RelationshipRepository
from relationship_lifelog_agent.privacy.guard import sanitize_answer
from relationship_lifelog_agent.profiles import PROFILE_NONE_VALUE, default_profile_choice, parse_profile_choice, profile_choices
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

ANALYSIS_MODE_CHOICES = ("safe_window", "private_full_range", "private_full_corpus")
DATE_SCOPE_CHOICES = ("auto", "all", "custom")


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
        db_path_state = gr.State(settings.paths.relationship_db)
        conversation_state = gr.State({})
        with gr.Accordion("Settings", open=False):
            backend = gr.Dropdown(
                choices=["mock", "upstream_readonly"],
                value=settings.adapter.backend,
                label="Backend",
                interactive=True,
            )
            analysis_mode = gr.Dropdown(
                choices=list(ANALYSIS_MODE_CHOICES),
                value=settings.analysis.mode,
                label="Analysis Mode",
                interactive=True,
            )
            date_scope = gr.Radio(
                choices=list(DATE_SCOPE_CHOICES),
                value=_ui_date_scope(settings.analysis.default_scope),
                label="Date Scope",
                interactive=True,
            )
            profile = gr.Dropdown(
                choices=profile_options,
                value=default_profile,
                label="Target profile",
                interactive=True,
            )
            manual_profile_name = gr.Textbox(label="profile_name", value="", placeholder="手動表示名")
            manual_relationship_label = gr.Dropdown(
                choices=[("未設定", ""), *[(label, label) for label in sorted(ALLOWED_RELATIONSHIP_LABELS)]],
                value="",
                label="relationship_label",
                interactive=True,
            )
            manual_person_source_id = gr.Textbox(label="person_source_id", value="", placeholder="plr:person:...")
            manual_line_speaker_source_id = gr.Textbox(
                label="line_speaker_source_id",
                value="",
                placeholder="plr:line_speaker:...",
            )
            manual_line_speaker_group_source_id = gr.Textbox(
                label="line_speaker_group_source_id",
                value="",
                placeholder="plr:line_speaker_group:...",
            )
            manual_self_person_source_id = gr.Textbox(label="self_person_source_id", value="", placeholder="plr:person:...")
            manual_self_line_speaker_source_id = gr.Textbox(
                label="self_line_speaker_source_id",
                value="",
                placeholder="plr:line_speaker:...",
            )
            manual_self_line_speaker_group_source_id = gr.Textbox(
                label="self_line_speaker_group_source_id",
                value="",
                placeholder="plr:line_speaker_group:...",
            )
            save_profile = gr.Button("Save manual profile", variant="secondary")
            profile_save_status = gr.Markdown("")
            with gr.Row():
                date_from = gr.Textbox(label="date_from", placeholder="2025-01-01", value="")
                date_to = gr.Textbox(label="date_to", placeholder="2025-03-31", value="")
            post_conflict_window_days = gr.Number(
                value=settings.relationship.post_conflict_window_days,
                label="Post-conflict window days",
                precision=0,
                minimum=1,
            )
            with gr.Accordion("Include raw payload (private full only)", open=False):
                include_raw_line_text = gr.Checkbox(
                    value=settings.private_full_llm_payload.allow_raw_line_text,
                    label="LINE全文",
                )
                include_raw_note_text = gr.Checkbox(
                    value=settings.private_full_llm_payload.allow_raw_note_text,
                    label="メモ全文",
                )
                include_photo_paths = gr.Checkbox(
                    value=settings.private_full_llm_payload.allow_photo_paths,
                    label="写真パス",
                )
                include_exact_gps = gr.Checkbox(
                    value=settings.private_full_llm_payload.allow_exact_gps,
                    label="正確GPS",
                )
                include_face_crops = gr.Checkbox(
                    value=settings.private_full_llm_payload.allow_face_crops,
                    label="顔crop",
                )
                include_face_embeddings = gr.Checkbox(
                    value=settings.private_full_llm_payload.allow_face_embeddings,
                    label="顔embedding",
                )
                include_private_file_paths = gr.Checkbox(
                    value=settings.private_full_llm_payload.allow_private_file_paths,
                    label="private path",
                )
                include_unverified_person_candidates = gr.Checkbox(
                    value=settings.private_full_llm_payload.allow_unverified_person_candidates,
                    label="未確認person候補",
                )
                include_unverified_speaker_candidates = gr.Checkbox(
                    value=settings.private_full_llm_payload.allow_unverified_speaker_candidates,
                    label="未確認speaker候補",
                )
            with gr.Row():
                max_runtime = gr.Number(
                    value=settings.analysis.max_runtime_seconds,
                    label="max_runtime sec",
                    precision=0,
                    minimum=1,
                )
                max_llm_calls = gr.Number(
                    value=settings.analysis.max_llm_calls,
                    label="max_llm_calls",
                    precision=0,
                    minimum=0,
                )
                max_batches = gr.Number(
                    value=settings.analysis.full_scan.max_batches_per_run,
                    label="max_batches",
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

        def respond(
            user_message: Any,
            history: list[Any] | None,
            selected_backend: str,
            selected_analysis_mode: str,
            selected_date_scope: str,
            selected_profile: str | None,
            selected_date_from: str | None,
            selected_date_to: str | None,
            selected_window_days: int | float | None,
            raw_line_text: bool,
            raw_note_text: bool,
            photo_paths: bool,
            exact_gps: bool,
            face_crops: bool,
            face_embeddings: bool,
            private_file_paths: bool,
            unverified_person_candidates: bool,
            unverified_speaker_candidates: bool,
            selected_max_runtime: int | float | None,
            selected_max_llm_calls: int | float | None,
            selected_max_batches: int | float | None,
            selected_mode: str,
            show_debug: bool,
            current_conversation_state: dict[str, object] | None,
        ) -> Iterator[tuple[str, list[dict[str, Any]], Any, str, list[dict[str, object]], dict[str, object]]]:
            yield from build_ui_chat_turn_stream(
                user_message,
                history,
                base_settings=settings,
                selected_backend=selected_backend,
                analysis_mode=selected_analysis_mode,
                date_scope=selected_date_scope,
                selected_profile=selected_profile,
                date_from=selected_date_from,
                date_to=selected_date_to,
                post_conflict_window_days=selected_window_days,
                include_raw_line_text=raw_line_text,
                include_raw_note_text=raw_note_text,
                include_photo_paths=photo_paths,
                include_exact_gps=exact_gps,
                include_face_crops=face_crops,
                include_face_embeddings=face_embeddings,
                include_private_file_paths=private_file_paths,
                include_unverified_person_candidates=unverified_person_candidates,
                include_unverified_speaker_candidates=unverified_speaker_candidates,
                max_runtime=selected_max_runtime,
                max_llm_calls=selected_max_llm_calls,
                max_batches=selected_max_batches,
                mode=selected_mode,
                show_debug=show_debug,
                conversation_state=current_conversation_state,
            )

        chat_inputs = [
            message,
            chatbot,
            backend,
            analysis_mode,
            date_scope,
            profile,
            date_from,
            date_to,
            post_conflict_window_days,
            include_raw_line_text,
            include_raw_note_text,
            include_photo_paths,
            include_exact_gps,
            include_face_crops,
            include_face_embeddings,
            include_private_file_paths,
            include_unverified_person_candidates,
            include_unverified_speaker_candidates,
            max_runtime,
            max_llm_calls,
            max_batches,
            mode,
            debug,
            conversation_state,
        ]
        send.click(
            respond,
            chat_inputs,
            [message, chatbot, review_target, review_targets_md, review_targets_state, conversation_state],
        )
        message.submit(
            respond,
            chat_inputs,
            [message, chatbot, review_target, review_targets_md, review_targets_state, conversation_state],
        )
        save_review.click(
            save_review_action_from_ui,
            [review_target, review_action, review_note, review_targets_state, db_path_state],
            review_status,
        )
        save_profile.click(
            save_profile_from_ui,
            [
                profile,
                manual_profile_name,
                manual_relationship_label,
                manual_person_source_id,
                manual_line_speaker_source_id,
                manual_line_speaker_group_source_id,
                manual_self_person_source_id,
                manual_self_line_speaker_source_id,
                manual_self_line_speaker_group_source_id,
                db_path_state,
            ],
            [profile_save_status, profile],
        )
        clear.click(
            lambda: ([], gr.update(choices=[], value=None), "レビュー対象はまだありません。", [], "", {}),
            None,
            [chatbot, review_target, review_targets_md, review_targets_state, review_status, conversation_state],
        )

    return demo


def build_ui_chat_response(
    user_message: Any,
    history: list[Any] | None,
    *,
    base_settings: Settings,
    selected_backend: str,
    analysis_mode: str | None = None,
    date_scope: str | None = None,
    selected_profile: str | None,
    date_from: str | None,
    date_to: str | None,
    post_conflict_window_days: int | float | None,
    include_raw_line_text: bool | None = None,
    include_raw_note_text: bool | None = None,
    include_photo_paths: bool | None = None,
    include_exact_gps: bool | None = None,
    include_face_crops: bool | None = None,
    include_face_embeddings: bool | None = None,
    include_private_file_paths: bool | None = None,
    include_unverified_person_candidates: bool | None = None,
    include_unverified_speaker_candidates: bool | None = None,
    max_runtime: int | float | None = None,
    max_llm_calls: int | float | None = None,
    max_batches: int | float | None = None,
    mode: str,
    show_debug: bool = False,
) -> tuple[list[dict[str, str]], ChatAnswer | None]:
    history, chat_answer, _state = build_ui_chat_turn(
        user_message,
        history,
        base_settings=base_settings,
        selected_backend=selected_backend,
        analysis_mode=analysis_mode,
        date_scope=date_scope,
        selected_profile=selected_profile,
        date_from=date_from,
        date_to=date_to,
        post_conflict_window_days=post_conflict_window_days,
        include_raw_line_text=include_raw_line_text,
        include_raw_note_text=include_raw_note_text,
        include_photo_paths=include_photo_paths,
        include_exact_gps=include_exact_gps,
        include_face_crops=include_face_crops,
        include_face_embeddings=include_face_embeddings,
        include_private_file_paths=include_private_file_paths,
        include_unverified_person_candidates=include_unverified_person_candidates,
        include_unverified_speaker_candidates=include_unverified_speaker_candidates,
        max_runtime=max_runtime,
        max_llm_calls=max_llm_calls,
        max_batches=max_batches,
        mode=mode,
        show_debug=show_debug,
        conversation_state=None,
    )
    return history, chat_answer


def build_ui_chat_turn(
    user_message: Any,
    history: list[Any] | None,
    *,
    base_settings: Settings,
    selected_backend: str,
    analysis_mode: str | None = None,
    date_scope: str | None = None,
    selected_profile: str | None,
    date_from: str | None,
    date_to: str | None,
    post_conflict_window_days: int | float | None,
    include_raw_line_text: bool | None = None,
    include_raw_note_text: bool | None = None,
    include_photo_paths: bool | None = None,
    include_exact_gps: bool | None = None,
    include_face_crops: bool | None = None,
    include_face_embeddings: bool | None = None,
    include_private_file_paths: bool | None = None,
    include_unverified_person_candidates: bool | None = None,
    include_unverified_speaker_candidates: bool | None = None,
    max_runtime: int | float | None = None,
    max_llm_calls: int | float | None = None,
    max_batches: int | float | None = None,
    mode: str,
    show_debug: bool = False,
    conversation_state: dict[str, object] | None = None,
) -> tuple[list[dict[str, str]], ChatAnswer | None, dict[str, object]]:
    normalized_history = _normalize_history(history)
    question = _coerce_message_text(user_message)
    if not question:
        return normalized_history, None, conversation_state or {}
    chat_answer = build_ui_chat_answer(
        question,
        base_settings=base_settings,
        selected_backend=selected_backend,
        analysis_mode=analysis_mode,
        date_scope=date_scope,
        selected_profile=selected_profile,
        date_from=date_from,
        date_to=date_to,
        post_conflict_window_days=post_conflict_window_days,
        include_raw_line_text=include_raw_line_text,
        include_raw_note_text=include_raw_note_text,
        include_photo_paths=include_photo_paths,
        include_exact_gps=include_exact_gps,
        include_face_crops=include_face_crops,
        include_face_embeddings=include_face_embeddings,
        include_private_file_paths=include_private_file_paths,
        include_unverified_person_candidates=include_unverified_person_candidates,
        include_unverified_speaker_candidates=include_unverified_speaker_candidates,
        max_runtime=max_runtime,
        max_llm_calls=max_llm_calls,
        max_batches=max_batches,
        mode=mode,
        show_debug=show_debug,
        conversation_state=conversation_state,
    )
    return (
        [
            *normalized_history,
            {"role": "user", "content": question},
            {"role": "assistant", "content": chat_answer.text},
        ],
        chat_answer,
        getattr(chat_answer, "_conversation_state", conversation_state or {}),
    )


def build_ui_chat_turn_stream(
    user_message: Any,
    history: list[Any] | None,
    *,
    base_settings: Settings,
    selected_backend: str,
    analysis_mode: str | None = None,
    date_scope: str | None = None,
    selected_profile: str | None,
    date_from: str | None,
    date_to: str | None,
    post_conflict_window_days: int | float | None,
    include_raw_line_text: bool | None = None,
    include_raw_note_text: bool | None = None,
    include_photo_paths: bool | None = None,
    include_exact_gps: bool | None = None,
    include_face_crops: bool | None = None,
    include_face_embeddings: bool | None = None,
    include_private_file_paths: bool | None = None,
    include_unverified_person_candidates: bool | None = None,
    include_unverified_speaker_candidates: bool | None = None,
    max_runtime: int | float | None = None,
    max_llm_calls: int | float | None = None,
    max_batches: int | float | None = None,
    mode: str,
    show_debug: bool = False,
    conversation_state: dict[str, object] | None = None,
) -> Iterator[tuple[str, list[dict[str, Any]], Any, str, list[dict[str, object]], dict[str, object]]]:
    import gradio as gr

    normalized_history = _normalize_history(history)
    question = _coerce_message_text(user_message)
    if not question:
        yield "", normalized_history, gr.update(choices=[], value=None), "レビュー対象はまだありません。", [], conversation_state or {}
        return
    settings = _settings_for_ui(
        base_settings,
        selected_backend,
        post_conflict_window_days,
        analysis_mode=analysis_mode,
        date_scope=date_scope,
        include_raw_line_text=include_raw_line_text,
        include_raw_note_text=include_raw_note_text,
        include_photo_paths=include_photo_paths,
        include_exact_gps=include_exact_gps,
        include_face_crops=include_face_crops,
        include_face_embeddings=include_face_embeddings,
        include_private_file_paths=include_private_file_paths,
        include_unverified_person_candidates=include_unverified_person_candidates,
        include_unverified_speaker_candidates=include_unverified_speaker_candidates,
        max_batches=max_batches,
    )
    profile_id = parse_profile_choice(selected_profile)
    scoped_date_from, scoped_date_to = _dates_for_scope(date_scope, date_from, date_to)
    runtime_budget = _runtime_budget_from_ui(max_runtime=max_runtime, max_llm_calls=max_llm_calls)
    base_running_history: list[dict[str, Any]] = [*normalized_history, {"role": "user", "content": question}]
    if not settings.ui.stream_progress or not settings.ui.show_progress_messages:
        history_once, chat_answer, updated_state = build_ui_chat_turn(
            question,
            normalized_history,
            base_settings=base_settings,
            selected_backend=selected_backend,
            analysis_mode=analysis_mode,
            date_scope=date_scope,
            selected_profile=selected_profile,
            date_from=date_from,
            date_to=date_to,
            post_conflict_window_days=post_conflict_window_days,
            include_raw_line_text=include_raw_line_text,
            include_raw_note_text=include_raw_note_text,
            include_photo_paths=include_photo_paths,
            include_exact_gps=include_exact_gps,
            include_face_crops=include_face_crops,
            include_face_embeddings=include_face_embeddings,
            include_private_file_paths=include_private_file_paths,
            include_unverified_person_candidates=include_unverified_person_candidates,
            include_unverified_speaker_candidates=include_unverified_speaker_candidates,
            max_runtime=max_runtime,
            max_llm_calls=max_llm_calls,
            max_batches=max_batches,
            mode=mode,
            show_debug=show_debug,
            conversation_state=conversation_state,
        )
        if chat_answer is None:
            yield "", history_once, gr.update(choices=[], value=None), "レビュー対象はまだありません。", [], updated_state
            return
        yield (
            "",
            history_once,
            gr.update(choices=_target_choices(chat_answer.review_targets), value=_first_target_value(chat_answer.review_targets)),
            _targets_markdown(chat_answer.review_targets),
            _target_state(chat_answer.review_targets),
            updated_state,
        )
        return

    latest_state = conversation_state or {}
    latest_targets: tuple[ReviewTarget, ...] = ()
    progress_events: list[AgentStreamEvent] = []
    for event in stream_answer(
        question,
        mode=mode,
        settings=settings,
        selected_profile_id=profile_id,
        date_from=scoped_date_from,
        date_to=scoped_date_to,
        post_conflict_window_days=_safe_window_days(post_conflict_window_days),
        state=conversation_state,
        show_debug=show_debug,
        runtime_budget=runtime_budget,
    ):
        if event.debug_only and not show_debug:
            continue
        if event.kind == "final_answer":
            latest_state = dict(event.metadata.get("conversation_state") or latest_state)
            latest_targets = _review_targets_from_metadata(event.metadata.get("review_targets"))
            running_history = [*base_running_history, {"role": "assistant", "content": event.message}]
            if progress_events:
                running_history.append(_progress_chat_message(progress_events, done=True))
        elif _show_progress_event(event, settings):
            progress_events.append(event)
            running_history = [
                *base_running_history,
                _progress_chat_message(progress_events, done=False),
            ]
        else:
            if not progress_events:
                continue
            running_history = [
                *base_running_history,
                _progress_chat_message(progress_events, done=False),
            ]
        yield (
            "",
            _copy_history(running_history),
            gr.update(choices=_target_choices(latest_targets), value=_first_target_value(latest_targets)),
            _targets_markdown(latest_targets),
            _target_state(latest_targets),
            latest_state,
        )


def build_ui_chat_answer(
    question: Any,
    *,
    base_settings: Settings,
    selected_backend: str,
    analysis_mode: str | None = None,
    date_scope: str | None = None,
    selected_profile: str | None,
    date_from: str | None,
    date_to: str | None,
    post_conflict_window_days: int | float | None,
    include_raw_line_text: bool | None = None,
    include_raw_note_text: bool | None = None,
    include_photo_paths: bool | None = None,
    include_exact_gps: bool | None = None,
    include_face_crops: bool | None = None,
    include_face_embeddings: bool | None = None,
    include_private_file_paths: bool | None = None,
    include_unverified_person_candidates: bool | None = None,
    include_unverified_speaker_candidates: bool | None = None,
    max_runtime: int | float | None = None,
    max_llm_calls: int | float | None = None,
    max_batches: int | float | None = None,
    mode: str,
    show_debug: bool = False,
    conversation_state: dict[str, object] | None = None,
) -> ChatAnswer:
    question_text = _coerce_message_text(question)
    settings = _settings_for_ui(
        base_settings,
        selected_backend,
        post_conflict_window_days,
        analysis_mode=analysis_mode,
        date_scope=date_scope,
        include_raw_line_text=include_raw_line_text,
        include_raw_note_text=include_raw_note_text,
        include_photo_paths=include_photo_paths,
        include_exact_gps=include_exact_gps,
        include_face_crops=include_face_crops,
        include_face_embeddings=include_face_embeddings,
        include_private_file_paths=include_private_file_paths,
        include_unverified_person_candidates=include_unverified_person_candidates,
        include_unverified_speaker_candidates=include_unverified_speaker_candidates,
        max_batches=max_batches,
    )
    profile_id = parse_profile_choice(selected_profile)
    scoped_date_from, scoped_date_to = _dates_for_scope(date_scope, date_from, date_to)
    response = answer_with_reasoning(
        question_text,
        mode=mode,
        settings=settings,
        selected_profile_id=profile_id,
        date_from=scoped_date_from,
        date_to=scoped_date_to,
        post_conflict_window_days=_safe_window_days(post_conflict_window_days),
        state=conversation_state,
        show_debug=show_debug,
        runtime_budget=_runtime_budget_from_ui(max_runtime=max_runtime, max_llm_calls=max_llm_calls),
    )
    chat_answer = ChatAnswer(
        text=sanitize_answer(response.answer_markdown, mode=mode),
        review_targets=response.review_targets,
    )
    object.__setattr__(chat_answer, "_conversation_state", response.conversation_state.to_dict())
    return chat_answer


def save_profile_from_ui(
    selected_profile: str | int | None,
    profile_name: str | None,
    relationship_label: str | None,
    person_source_id: str | None,
    line_speaker_source_id: str | None,
    line_speaker_group_source_id: str | None,
    self_person_source_id: str | None,
    self_line_speaker_source_id: str | None,
    self_line_speaker_group_source_id: str | None,
    relationship_db_path: str,
) -> tuple[str, Any]:
    import gradio as gr

    repo = RelationshipRepository(relationship_db_path)
    selected_id = parse_profile_choice(selected_profile)
    label = _clean_text(relationship_label)
    values = {
        "person_source_id": _clean_text(person_source_id),
        "line_speaker_source_id": _clean_text(line_speaker_source_id),
        "line_speaker_group_source_id": _clean_text(line_speaker_group_source_id),
        "self_person_source_id": _clean_text(self_person_source_id),
        "self_line_speaker_source_id": _clean_text(self_line_speaker_source_id),
        "self_line_speaker_group_source_id": _clean_text(self_line_speaker_group_source_id),
    }
    try:
        if selected_id is None:
            clean_name = _clean_text(profile_name)
            if not clean_name:
                return "profile_name を入力してください。", gr.update()
            profile_id = repo.create_profile(
                profile_name=clean_name,
                relationship_label=label,
                **values,
            )
            status = f"manual profile を保存しました: id={profile_id}"
        else:
            fields = {key: value for key, value in values.items() if value is not None}
            clean_name = _clean_text(profile_name)
            if clean_name:
                fields["profile_name"] = clean_name
            if label:
                fields["relationship_label"] = label
            if not fields:
                return "更新するprofile項目を入力してください。", gr.update()
            changed = repo.update_profile(selected_id, **fields)
            if changed == 0:
                return "profile が見つからないか、変更がありません。", gr.update()
            profile_id = selected_id
            status = f"manual profile を更新しました: id={profile_id}"
    except ValueError as exc:
        return sanitize_answer(f"profile保存に失敗しました: {exc}"), gr.update()
    choices = _profile_choices_from_repo(repo)
    return sanitize_answer(status), gr.update(choices=choices, value=str(profile_id))


def build_ui_answer(
    question: Any,
    *,
    base_settings: Settings,
    selected_backend: str,
    analysis_mode: str | None = None,
    date_scope: str | None = None,
    selected_profile: str | None,
    date_from: str | None,
    date_to: str | None,
    post_conflict_window_days: int | float | None,
    include_raw_line_text: bool | None = None,
    include_raw_note_text: bool | None = None,
    include_photo_paths: bool | None = None,
    include_exact_gps: bool | None = None,
    include_face_crops: bool | None = None,
    include_face_embeddings: bool | None = None,
    include_private_file_paths: bool | None = None,
    include_unverified_person_candidates: bool | None = None,
    include_unverified_speaker_candidates: bool | None = None,
    max_runtime: int | float | None = None,
    max_llm_calls: int | float | None = None,
    max_batches: int | float | None = None,
    mode: str,
    show_debug: bool = False,
) -> str:
    return build_ui_chat_answer(
        question,
        base_settings=base_settings,
        selected_backend=selected_backend,
        analysis_mode=analysis_mode,
        date_scope=date_scope,
        selected_profile=selected_profile,
        date_from=date_from,
        date_to=date_to,
        post_conflict_window_days=post_conflict_window_days,
        include_raw_line_text=include_raw_line_text,
        include_raw_note_text=include_raw_note_text,
        include_photo_paths=include_photo_paths,
        include_exact_gps=include_exact_gps,
        include_face_crops=include_face_crops,
        include_face_embeddings=include_face_embeddings,
        include_private_file_paths=include_private_file_paths,
        include_unverified_person_candidates=include_unverified_person_candidates,
        include_unverified_speaker_candidates=include_unverified_speaker_candidates,
        max_runtime=max_runtime,
        max_llm_calls=max_llm_calls,
        max_batches=max_batches,
        mode=mode,
        show_debug=show_debug,
    ).text


def _normalize_history(history: list[Any] | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            if role not in {"user", "assistant"}:
                role = "user"
            content = _coerce_message_text(item.get("content"))
            if content:
                message: dict[str, Any] = {"role": role, "content": content}
                metadata = _safe_chat_metadata(item.get("metadata"))
                if metadata:
                    message["metadata"] = metadata
                messages.append(message)
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


def _event_chat_message(event: AgentStreamEvent) -> dict[str, Any]:
    content = event.message
    return {"role": "assistant", "content": content, "metadata": event.safe_metadata()}


def _progress_chat_message(
    events: list[AgentStreamEvent],
    *,
    done: bool,
) -> dict[str, Any]:
    status = "done" if done else "pending"
    return {
        "role": "assistant",
        "content": _progress_log(events, done=done),
        "metadata": {
            "title": "処理プロセス",
            "status": status,
            "kind": "progress",
            "log": "完了" if done else "実行中",
        },
    }


def _progress_log(events: list[AgentStreamEvent], *, done: bool) -> str:
    lines: list[str] = []
    for event in events:
        if not event.safe_for_user:
            continue
        title = _truncate_progress_text(event.title, 80)
        message = _truncate_progress_text(event.message, 180)
        prefix = "完了" if done or event.status == "done" else "確認中"
        if event.status == "error":
            prefix = "注意"
        lines.append(f"- {prefix}: {title} - {message}")
    return "\n".join(lines) or "安全な処理状況を確認しています。"


def _truncate_progress_text(value: str, max_chars: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _safe_chat_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {"title", "status", "kind", "id", "parent_id", "duration", "log"}
    metadata = {key: item for key, item in value.items() if key in allowed}
    status = metadata.get("status")
    if status not in {"pending", "done", "error"}:
        metadata.pop("status", None)
    if "title" in metadata:
        metadata["title"] = _truncate_progress_text(str(metadata["title"]), 80)
    return metadata


def _copy_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for item in history:
        new_item = dict(item)
        if isinstance(new_item.get("metadata"), dict):
            new_item["metadata"] = dict(new_item["metadata"])
        copied.append(new_item)
    return copied


def _show_progress_event(event: AgentStreamEvent, settings: Settings) -> bool:
    if event.kind in {"tool_start", "tool_done"} and not settings.ui.show_tool_usage:
        return False
    if event.kind in {"progress", "thought"} and not settings.ui.show_safe_reasoning_summary:
        return False
    return event.kind in {"progress", "thought", "tool_start", "tool_done", "llm_start", "llm_done", "warning", "error"}


def _review_targets_from_metadata(value: object) -> tuple[ReviewTarget, ...]:
    if not isinstance(value, list):
        return ()
    targets: list[ReviewTarget] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            targets.append(
                ReviewTarget(
                    event_id=int(item["event_id"]),
                    label=str(item.get("label") or ""),
                    event_type=str(item.get("event_type") or ""),
                    review_status=str(item.get("review_status") or ""),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(targets)


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


def _profile_choices_from_repo(repo: RelationshipRepository) -> list[tuple[str, str]]:
    rows = repo.list_profiles()
    choices = [(f"{row['id']}: {row['profile_name']}", str(row["id"])) for row in rows]
    return choices or [("未設定", PROFILE_NONE_VALUE)]


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
    *,
    analysis_mode: str | None = None,
    date_scope: str | None = None,
    include_raw_line_text: bool | None = None,
    include_raw_note_text: bool | None = None,
    include_photo_paths: bool | None = None,
    include_exact_gps: bool | None = None,
    include_face_crops: bool | None = None,
    include_face_embeddings: bool | None = None,
    include_private_file_paths: bool | None = None,
    include_unverified_person_candidates: bool | None = None,
    include_unverified_speaker_candidates: bool | None = None,
    max_batches: int | float | None = None,
) -> Settings:
    backend = selected_backend if selected_backend in {"mock", "upstream_readonly"} else settings.adapter.backend
    selected_mode = analysis_mode if analysis_mode in ANALYSIS_MODE_CHOICES else settings.analysis.mode
    selected_scope = date_scope if date_scope in DATE_SCOPE_CHOICES else settings.analysis.default_scope
    window_days = _safe_window_days(post_conflict_window_days)
    full_scan = replace(
        settings.analysis.full_scan,
        max_batches_per_run=_safe_int(max_batches, settings.analysis.full_scan.max_batches_per_run, minimum=1),
    )
    analysis = replace(
        settings.analysis,
        mode=selected_mode,
        default_scope=selected_scope,
        full_scan=full_scan,
    )
    raw_payload = replace(
        settings.private_full_llm_payload,
        allow_raw_line_text=_optional_bool(include_raw_line_text, settings.private_full_llm_payload.allow_raw_line_text),
        allow_raw_note_text=_optional_bool(include_raw_note_text, settings.private_full_llm_payload.allow_raw_note_text),
        allow_photo_paths=_optional_bool(include_photo_paths, settings.private_full_llm_payload.allow_photo_paths),
        allow_exact_gps=_optional_bool(include_exact_gps, settings.private_full_llm_payload.allow_exact_gps),
        allow_face_crops=_optional_bool(include_face_crops, settings.private_full_llm_payload.allow_face_crops),
        allow_face_embeddings=_optional_bool(include_face_embeddings, settings.private_full_llm_payload.allow_face_embeddings),
        allow_private_file_paths=_optional_bool(
            include_private_file_paths,
            settings.private_full_llm_payload.allow_private_file_paths,
        ),
        allow_unverified_person_candidates=_optional_bool(
            include_unverified_person_candidates,
            settings.private_full_llm_payload.allow_unverified_person_candidates,
        ),
        allow_unverified_speaker_candidates=_optional_bool(
            include_unverified_speaker_candidates,
            settings.private_full_llm_payload.allow_unverified_speaker_candidates,
        ),
    )
    return replace(
        settings,
        adapter=replace(settings.adapter, backend=backend),
        analysis=analysis,
        private_full_llm_payload=raw_payload,
        relationship=replace(settings.relationship, post_conflict_window_days=window_days),
    )


def _safe_window_days(value: int | float | None) -> int:
    if value is None:
        return 14
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 14


def _safe_int(value: int | float | None, default: int, *, minimum: int = 0) -> int:
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: int | float | None, default: float | None, *, minimum: float = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError):
        return default


def _optional_bool(value: bool | None, default: bool) -> bool:
    return default if value is None else bool(value)


def _runtime_budget_from_ui(
    *,
    max_runtime: int | float | None,
    max_llm_calls: int | float | None,
) -> RuntimeBudget | None:
    if max_runtime is None and max_llm_calls is None:
        return None
    return RuntimeBudget(
        max_runtime_seconds=_safe_float(max_runtime, None, minimum=1.0),
        max_llm_calls=_safe_int(max_llm_calls, 20, minimum=0),
    )


def _dates_for_scope(date_scope: str | None, date_from: str | None, date_to: str | None) -> tuple[str | None, str | None]:
    if date_scope in {"auto", "all"}:
        return None, None
    return _clean_text(date_from), _clean_text(date_to)


def _ui_date_scope(value: str | None) -> str:
    if value in DATE_SCOPE_CHOICES:
        return value
    if value == "ask_llm":
        return "auto"
    return "auto"


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
