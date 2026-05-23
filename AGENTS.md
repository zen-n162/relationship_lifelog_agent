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

## Project direction

`relationship_lifelog_agent` is moving toward a Private Full-Corpus Personal Memory Agent.

It should not remain a fixed intent router. The final direction is an agent that uses photos, LINE, notes, location, face metadata, and relationship-local review state to answer questions that a human could answer by carefully investigating the private local corpus.

The agent should:

1. Understand the user's question and conversation context.
2. Decide what information is needed.
3. Build a tool plan.
4. Execute read-only local tools.
5. Observe retrieved data.
6. Analyze the available evidence with Python/SQL and local LLMs when appropriate.
7. Replan when the first pass is insufficient.
8. Compose an answer with evidence, confidence, cautions, and verified source refs.

The core runtime direction is:

```text
Plan -> Execute -> Observe -> Analyze -> Replan -> Answer
```

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
14. Use Python/SQL as the final authority for counts, dates, durations, reply-delay metrics, ID verification, source-range checks, and aggregation.
15. Use LLMs for question understanding, task decomposition, tool planning, analysis, summarization, and answer phrasing only under the analysis-mode rules below.
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


## Analysis modes

The app supports three conceptual analysis modes.

### safe_window

`safe_window` is the default and current production-safe path.

Rules:

- Python/SQL retrieves small candidate windows.
- The local LLM may analyze short LINE excerpts, short note excerpts, media captions, place labels, and verified metadata.
- The local LLM must not receive the entire LINE log, all notes, raw photo paths, exact GPS, face crops, face embeddings, private file paths, or unverified identity candidates.
- This mode is appropriate for normal chat, public-safe outputs, tests, and fast relationship evidence review.

### private_full_range

`private_full_range` is an explicit private mode for a user-approved date range.

With user permission and local-only execution, the LLM payload may include:

- raw LINE text
- raw note body
- photo paths
- exact GPS
- face crop paths
- face embeddings
- private file paths
- unverified person candidates
- unverified speaker candidates

Rules:

- A date range should be explicit.
- The upstream apps remain read-only.
- Raw payload may be sent to local Ollama only.
- Raw prompt logging defaults to false.
- Raw payload cache defaults to false.
- Final answers must pass source_ref verification before display.

### private_full_corpus

`private_full_corpus` is an explicit private mode for the full local corpus.

Rules:

- It requires user permission.
- It must use runtime budgets such as max runtime, max steps, max tool calls, max LLM calls, and max batches.
- If the corpus does not fit in one context, batch the analysis.
- Each batch must preserve source refs.
- The final answer must pass source_ref verification.
- The UI should warn that this may be slow and should show safe progress summaries only.

## Private full mode safeguards

Private full modes change what may be sent to the local LLM payload, but they do not change the external safety boundary.

Always enforce:

1. External APIs are forbidden.
2. Cloud LLMs are forbidden.
3. Automatic model download is forbidden.
4. Ollama should use `127.0.0.1` or another validated localhost URL.
5. `allow_gradio_share` remains false.
6. Raw prompt logging defaults to false.
7. Raw payload cache defaults to false.
8. Public mode keeps the existing redaction policy.
9. LLM structured output must be schema-validated.
10. LLM tool plans must be checked against an allowed tool whitelist.
11. Python/SQL has final responsibility for counts, dates, profile IDs, person/speaker IDs, source range validation, and aggregation.
12. The final answer must pass source_ref verification.
13. Raw full payload should not be stored in the relationship DB or debug logs by default.
14. Safe progress messages must not show raw prompt, raw chain-of-thought, raw LINE text, raw note text, exact GPS, face data, photo paths, or private file paths.

## Raw data handling

Private full mode permits raw data in local LLM payload only after explicit user permission.

Default behavior must remain conservative:

- Do not log raw prompts.
- Do not cache raw payloads.
- Do not write raw full LINE logs, raw note bodies, exact GPS, face crops, face embeddings, photo paths, or private file paths into relationship DB tables by default.
- Store source refs, hashes, manifests, structured analysis results, and verification reports instead.
- If future work adds raw prompt or raw payload persistence, it must require a separate explicit opt-in setting and tests.


## ID discovery policy

When asked to find person IDs or LINE speaker IDs from `personal_lifelog_rag`:

- Use read-only access.
- Do not modify upstream databases.
- Do not print raw LINE message text.
- Do not print raw note text.
- Do not print exact GPS, face crops, embeddings, or private image paths.
- Treat all matches as candidates until the user confirms them.
- Do not infer relationship labels.
- Do not automatically link a face person and a LINE speaker.
- Output candidate IDs, table names, label/display names, counts, confidence, and reasons only.
- Provide a profile creation command, but do not execute it unless explicitly asked.


## Conversation-aware agent behavior

The app should not be a simple intent router.

For each chat message, the agent should:

1. Understand the user's goal.
2. Extract entities, dates, profile names, and references to previous answers.
3. Resolve the target profile using only manually configured profiles.
4. Determine what information is required to answer.
5. Check whether the required information is available.
6. If information is missing, explain what is missing and how to provide it.
7. If information is available, build a query plan.
8. Execute only the required read-only tools.
9. Evaluate evidence according to the relevant evidence contract.
10. Compose an answer that separates facts, inferences, evidence, confidence, and cautions.

The agent should maintain short-lived conversation state, including:

- active_profile_id
- last_question
- last_intents
- last_observed_dates
- last_event_ids
- last_evidence_ids
- last_query_plan
- last_answerability_report

The agent may show a brief reasoning summary to the user, such as:

- what profile was used
- what data sources were checked
- what information was missing
- what assumptions or defaults were applied

Do not expose hidden raw reasoning, raw private data, or long excerpts.


## LLM payload policy by analysis mode

For `safe_window`, use this pattern:

1. Python/SQL retrieves small candidate windows.
2. The local LLM analyzes only those windows.
3. The local LLM returns structured JSON.
4. Python validates the JSON.
5. Python performs final counting, date filtering, evidence filtering, and privacy checks.

The LLM may analyze:

- LINE conversation windows
- date-near note windows
- media day summaries from captions/place/person metadata
- compound question planning
- final answer drafting

For `private_full_range` and `private_full_corpus`, the local LLM may analyze the user-approved raw payload described in the analysis mode section. That does not make the LLM the source of truth.

In all modes, the LLM must not be the final source of truth for:

- counts
- dates
- profile identity
- person/speaker links
- relationship labels
- privacy redaction
- raw evidence inclusion
- source_ref validity
