# AGENTS.md

## Project name

relationship_lifelog_agent

## Purpose

This repository implements a local-first, privacy-first Relationship Lifelog Agent.

The agent sits above two existing local RAG applications:

- `~/MyApplication/personal_lifelog_rag`
  - Photos, videos, LINE logs, GPS, EXIF, OCR, VLM captions, face detection, manually reviewed people, places, and event timeline.
- `~/MyApplication/notes_lifelog_rag`
  - Apple Notes / iPhone Notes exports, notes, thoughts, events, monthly reflections, suggestions, evidence quotes, and timeline review.

This project must not replace or merge those applications. It should call them through adapters and add a relationship-specific analysis layer with a simple ChatGPT-like chat UI.

The final experience should allow the user to ask questions such as:

- 喧嘩はどのくらいしている？
- 喧嘩はいつ多かった？
- 喧嘩の後にどこへ遊びに行っている？
- 仲直りまで何日くらいかかっている？
- LINEを返せていなかった時期は？
- 2025年1月は関係的にどんな月だった？
- あの時、自分はメモで何を考えていた？
- よく行っていたデート場所は？
- 約束したけど実際に行かなかった予定はある？

## Hard requirements

Follow these requirements strictly.

1. Do not destructively modify `personal_lifelog_rag`.
2. Do not destructively modify `notes_lifelog_rag`.
3. Treat existing applications as upstream memory tools.
4. This repository owns only the relationship-specific layer, relationship DB, chat UI, adapters, safety guards, and tests.
5. Prefer read-only access to upstream apps.
6. Do not send private data to external APIs.
7. Do not use cloud LLMs.
8. Do not add automatic model downloads.
9. Do not enable Gradio `share=True`.
10. Default server host must be `127.0.0.1`.
11. Keep the UI simple and ChatGPT-like.
12. Do not build a complicated multi-tab UI for the MVP.
13. Use mock adapters first when upstream integration is uncertain.
14. Use Python/SQL for counts, dates, durations, reply-delay metrics, and aggregation.
15. Use LLMs only for classification assistance, summarization, and answer phrasing when needed.
16. Relationship labels such as partner, lover, girlfriend, boyfriend, ex-partner, or 恋人 must never be inferred by AI.
17. Relationship labels may be used only when manually configured by the user in private mode.
18. Face identity and LINE speaker identity must never be automatically linked by AI.
19. Do not infer family, friend, lover, intimacy, dating status, or relationship status from photos or messages.
20. Do not claim certainty about conflicts, reconciliation, affection, or the other person's inner feelings.

## Recommended architecture

The project should be structured as:

```text
relationship_lifelog_agent/
  pyproject.toml
  README.md
  AGENTS.md
  config.example.yaml
  data/
    relationship.sqlite
    cache/
    exports/
  src/
    relationship_lifelog_agent/
      __init__.py
      app.py
      cli.py
      config.py
      db/
        __init__.py
        schema.sql
        migrate.py
        repository.py
      adapters/
        __init__.py
        types.py
        personal_lifelog.py
        notes_lifelog.py
        mock.py
      agent/
        __init__.py
        router.py
        planner.py
        executor.py
        answer_composer.py
        memory.py
      analytics/
        __init__.py
        conflict.py
        reconciliation.py
        reply_delay.py
        promise.py
        outing.py
        monthly_review.py
        evidence_scoring.py
      review/
        __init__.py
        actions.py
        queue.py
      privacy/
        __init__.py
        guard.py
        redaction.py
        policies.py
      llm/
        __init__.py
        local_client.py
        prompts.py
        schemas.py
      ui/
        __init__.py
        chat_ui.py
        components.py
        styles.css
  tests/
    test_router.py
    test_privacy.py
    test_schema.py
    test_answer_safety.py
    test_mock_qa.py
  eval/
    private_eval_cases.yaml
    safety_eval_cases.yaml
    run_eval.py
```

## Documentation maintenance

Keep `docs/application_overview.html` as the human-readable explanation of this application.

When application specifications change, update this HTML file in the same change set. This includes changes to:

1. Supported intents or example questions.
2. Adapter behavior or upstream integration policy.
3. SQLite schema or repository responsibilities.
4. Privacy, safety, public/private mode, or forbidden-output behavior.
5. Gradio UI behavior, launch defaults, or debug/settings behavior.
6. Local LLM policy or any model-related constraints.

## GitHub update workflow

When making changes in this repository:

1. Run relevant tests before publishing.
2. Check that private data is not staged:
   - `config.local.yaml`
   - SQLite DBs
   - raw LINE logs
   - photos, face crops, embeddings, GPS data
   - private exports or cache contents
3. Commit only reviewable source, docs, tests, examples, and `.gitkeep` files.
4. Push updates to `origin` at `git@github.com:zen-n162/relationship_lifelog_agent.git`.
5. Do not claim GitHub was updated until `git push` succeeds.
6. If GitHub push fails because of SSH/auth/network issues, report the exact blocker and leave the local commit intact.


## Instruction maintenance log

2026-05-23: The previously uncommitted AGENTS diff was reviewed and kept as formal guidance. It was classified as:

- safety/privacy rule: raw upstream data, exact GPS, face data, private paths, and private excerpts must not leak into logs, reports, or public output.
- upstream integration rule: upstream apps remain read-only dependencies and real integration is gated by `adapter.backend: upstream_readonly`.
- testing/eval rule: mock behavior and upstream-not-configured behavior must both stay covered.
- duplicate: the existing hard requirements already say not to destructively modify upstream apps, prefer read-only access, and use mock adapters first; the rules below keep those points as implementation-level detail.
- obsolete: none found.
- unknown: none found.

## Upstream integration rules

When integrating with `personal_lifelog_rag` or `notes_lifelog_rag`, treat this section as the implementation-level detail for the hard requirements above.

1. Investigate upstream structure before implementing integration.
2. Do not edit upstream applications unless explicitly instructed.
3. Prefer read-only integration.
4. If using SQLite, use read-only URI mode:
   `sqlite3.connect("file:/path/to/db?mode=ro", uri=True)`
5. Do not copy raw upstream private data into relationship DB.
6. Store source pointers and short redacted excerpts, not full records.
7. Keep mock adapter working after adding real adapters.
8. Add tests for both mock and upstream-not-configured states.
9. If upstream paths are missing, fail safely with a user-readable warning.
10. Never print raw LINE messages, raw notes, exact GPS, face embeddings, face crops, or private file paths in logs.
11. Real upstream integration must be gated by config:
    `adapter.backend: upstream_readonly`.
12. Default backend should remain `mock` until explicitly changed.
13. Relationship profile mapping must be manual.
14. Do not infer partner, lover, friend, family, or intimacy from upstream data.
15. Dry-run analysis must not write relationship events unless `--write` is explicitly provided.
