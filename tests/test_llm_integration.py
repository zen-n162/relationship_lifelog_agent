from __future__ import annotations

import ast
import json

from relationship_lifelog_agent.agent.reasoning_orchestrator import answer_with_reasoning
from relationship_lifelog_agent.config import LlmSettings, PathSettings, RelationshipSettings, Settings, load_config
from relationship_lifelog_agent.db.repository import RelationshipRepository
from relationship_lifelog_agent.llm.local_client import LocalLlmClient
from relationship_lifelog_agent.llm.status import check_llm_status
from relationship_lifelog_agent.privacy.guard import detect_answer_safety_violations


def test_config_loads_ollama_llm_fields(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  provider: ollama
  model: fake-model
  base_url: http://127.0.0.1:11434
  enabled: true
  require_structured_output: true
  fallback_to_rules: true
  use_for_question_understanding: true
  use_for_information_needs: true
  use_for_query_planning: true
  use_for_answer_composition: true
""",
        encoding="utf-8",
    )

    settings = load_config(config)

    assert settings.llm.provider == "ollama"
    assert settings.llm.model == "fake-model"
    assert settings.llm.enabled is True
    assert settings.llm.use_for_query_planning is True


def test_llm_disabled_does_not_call_client(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path, llm=LlmSettings(enabled=False, model="fake"))
    client = LocalLlmClient(settings.llm, http_post=_raising_post)

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings, llm_client=client, show_debug=True)

    usage = response.debug_info["llm_usage"]
    assert usage["llm_call_count"] == 0
    assert usage["question_understanding_backend"] == "rule"


def test_llm_model_null_does_not_call_client(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path, llm=LlmSettings(enabled=True, model=None, use_for_question_understanding=True))
    client = LocalLlmClient(settings.llm, http_post=_raising_post)

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings, llm_client=client, show_debug=True)

    assert response.debug_info["llm_usage"]["llm_call_count"] == 0


def test_ollama_client_is_called_for_question_understanding_and_planning(tmp_path) -> None:
    calls: list[dict[str, object]] = []
    settings = _settings_with_profile(tmp_path, llm=_llm_settings(question=True, needs=True, planning=True))
    client = LocalLlmClient(settings.llm, http_post=_fake_success_post(calls))

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings, llm_client=client, show_debug=True)

    usage = response.debug_info["llm_usage"]
    assert len(calls) >= 3
    assert usage["llm_call_count"] >= 3
    assert usage["question_understanding_backend"] == "llm"
    assert usage["information_needs_backend"] == "llm"
    assert usage["query_planning_backend"] == "llm"


def test_llm_invalid_json_falls_back_to_rules(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path, llm=_llm_settings(question=True))
    client = LocalLlmClient(settings.llm, http_post=lambda url, payload, timeout: {"message": {"content": "not json"}})

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings, llm_client=client, show_debug=True)

    usage = response.debug_info["llm_usage"]
    assert usage["question_understanding_backend"] == "rule"
    assert usage["fallback_used"] is True
    assert usage["fallback_reasons"]


def test_llm_disallowed_tool_is_rejected(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path, llm=_llm_settings(planning=True))
    client = LocalLlmClient(settings.llm, http_post=_fake_disallowed_tool_post)

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings, llm_client=client, show_debug=True)

    usage = response.debug_info["llm_usage"]
    assert usage["query_planning_backend"] == "rule"
    assert usage["fallback_used"] is True
    assert "disallowed tool" in " ".join(usage["fallback_reasons"])


def test_llm_answer_cannot_override_python_aggregate_facts(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path, llm=_llm_settings(answer=True))
    client = LocalLlmClient(settings.llm, http_post=_fake_bad_count_answer_post)

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings, llm_client=client, show_debug=True)

    assert "total_candidates: 999" not in response.answer_markdown
    assert response.debug_info["llm_usage"]["answer_composition_backend"] == "template"
    assert response.debug_info["llm_usage"]["fallback_used"] is True


def test_llm_answer_still_passes_safety_guard(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path, llm=_llm_settings(answer=True))
    client = LocalLlmClient(settings.llm, http_post=_fake_forbidden_answer_post)

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings, llm_client=client)

    assert "相手は怒っていた" not in response.answer_markdown
    assert detect_answer_safety_violations(response.answer_markdown) == []


def test_public_mode_does_not_send_excerpt_to_answer_llm(tmp_path) -> None:
    prompts: list[str] = []
    settings = _settings_with_profile(tmp_path, llm=_llm_settings(answer=True))

    def fake_post(url, payload, timeout):
        prompts.append(str(payload["messages"][-1]["content"]))
        return _fake_forbidden_answer_post(url, payload, timeout)

    client = LocalLlmClient(settings.llm, http_post=fake_post)

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings, llm_client=client, mode="public")

    assert "excerpt:" not in "\n".join(prompts)
    assert "excerpt:" not in response.answer_markdown


def test_llm_debug_info_does_not_include_raw_private_text(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path, llm=_llm_settings(question=True, planning=True))
    client = LocalLlmClient(settings.llm, http_post=_fake_success_post([]))

    response = answer_with_reasoning("いおりとの喧嘩はどれくらいしている？", settings=settings, llm_client=client, show_debug=True)

    debug_text = json.dumps(response.debug_info, ensure_ascii=False)
    assert "raw LINE" not in debug_text
    assert "raw note" not in debug_text
    assert "/home/" not in debug_text


def test_llm_status_uses_configured_ollama_client(tmp_path) -> None:
    settings = _settings_with_profile(tmp_path, llm=_llm_settings(question=True))
    client = LocalLlmClient(settings.llm, http_post=lambda url, payload, timeout: {"message": {"content": '{"ok": true}'}})

    report = check_llm_status(settings, client=client)

    assert report.enabled is True
    assert report.provider == "ollama"
    assert report.test_prompt_success is True
    assert report.structured_output_success is True


def _llm_settings(
    *,
    question: bool = False,
    needs: bool = False,
    planning: bool = False,
    answer: bool = False,
) -> LlmSettings:
    return LlmSettings(
        provider="ollama",
        model="fake-model",
        base_url="http://127.0.0.1:11434",
        enabled=True,
        require_structured_output=True,
        fallback_to_rules=True,
        use_for_question_understanding=question,
        use_for_information_needs=needs,
        use_for_query_planning=planning,
        use_for_answer_composition=answer,
        timeout_seconds=1,
    )


def _settings_with_profile(tmp_path, *, llm: LlmSettings) -> Settings:
    db_path = tmp_path / "relationship.sqlite"
    repo = RelationshipRepository(db_path)
    profile_id = repo.create_profile(
        "いおり",
        person_source_id="plr:person:target",
        line_speaker_source_id="plr:line_speaker:target",
        self_line_speaker_source_id="plr:line_speaker:self",
        relationship_label="partner",
    )
    return Settings(
        paths=PathSettings(relationship_db=str(db_path)),
        relationship=RelationshipSettings(default_profile_id=profile_id),
        llm=llm,
    )


def _fake_success_post(calls: list[dict[str, object]]):
    def fake_post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        calls.append(payload)
        content = str(payload["messages"][-1]["content"])
        if "Extract a safe QuestionFrame" in content:
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "intents": ["conflict_frequency"],
                            "target_profile_name": "いおり",
                            "date_from": None,
                            "date_to": None,
                            "referenced_previous_answer": False,
                            "requested_output": "aggregate_with_examples",
                            "needs_evidence": True,
                            "risk_level": "relationship_private",
                        },
                        ensure_ascii=False,
                    )
                }
            }
        if "Suggest information needs" in content:
            return {"message": {"content": '{"needs": [{"name": "target profile", "required": true, "reason": "needed", "how_to_get": null}]}'}}
        if "Create a safe tool plan" in content:
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "plan_steps": [
                                {
                                    "tool_name": "resolve_profile",
                                    "purpose": "profile",
                                    "required": True,
                                    "params": {},
                                    "output_key": "profile",
                                },
                                {
                                    "tool_name": "search_line_conflict_signals",
                                    "purpose": "line signals",
                                    "required": True,
                                    "params": {},
                                    "output_key": "line_evidence",
                                },
                                {
                                    "tool_name": "conflict_audit",
                                    "purpose": "aggregate",
                                    "required": True,
                                    "params": {},
                                    "output_key": "analysis_result",
                                },
                                {
                                    "tool_name": "compose_answer",
                                    "purpose": "answer",
                                    "required": True,
                                    "params": {},
                                    "output_key": "answer",
                                },
                            ],
                            "expected_answer_shape": "aggregate_with_examples",
                        },
                        ensure_ascii=False,
                    )
                }
            }
        return {"message": {"content": '{"answer_markdown": "unused"}'}}

    return fake_post


def _fake_disallowed_tool_post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    return {
        "message": {
            "content": json.dumps(
                {
                    "plan_steps": [
                        {"tool_name": "external_web_search", "purpose": "bad", "required": True, "params": {}, "output_key": "bad"}
                    ],
                    "expected_answer_shape": "answer",
                }
            )
        }
    }


def _fake_bad_count_answer_post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    return {
        "message": {
            "content": json.dumps(
                {
                    "answer_markdown": (
                        "要約:\n喧嘩候補が999件あります。\n\n"
                        "集計:\n- total_candidates: 999\n\n"
                        "具体例:\n- なし\n\n根拠:\n- なし\n\n信頼度:\nweak\n\n注意:\n- 候補です。"
                    )
                },
                ensure_ascii=False,
            )
        }
    }


def _fake_forbidden_answer_post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    content = str(payload["messages"][-1]["content"])
    template = ast.literal_eval(content)["template_answer"]
    return {
        "message": {
            "content": json.dumps({"answer_markdown": template + "\n\n相手は怒っていた"}, ensure_ascii=False)
        }
    }


def _raising_post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    raise AssertionError("LLM should not be called")
