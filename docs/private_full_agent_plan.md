# Private Full-Corpus Personal Memory Agent Migration Plan

## Summary

This document records the current implementation structure of
`relationship_lifelog_agent` and proposes a phased migration toward a
Private Full-Corpus Personal Memory Agent.

The current application is a local-first relationship evidence review assistant
with a conservative `safe_window` design: Python and SQLite select small
evidence windows, local Ollama may analyze those windows, and the final answer
is passed through safety and privacy guards.

The target design adds three explicit analysis modes:

- `safe_window`: existing short-window, source-pointer, short-excerpt mode.
- `private_full_range`: user-approved full raw payload for a selected date range.
- `private_full_corpus`: user-approved full raw payload across the whole corpus,
  with batching, budgets, persistent observations, and source-ref verification.

The migration should not weaken the existing defaults. Raw LINE text, raw note
text, photo paths, exact GPS, face crops, face embeddings, private file paths,
and unverified person or speaker candidates may be included in local LLM payloads
only when the user explicitly enables a private full mode. External APIs, cloud
LLMs, model auto-downloads, public Gradio sharing, raw prompt logging by
default, and raw payload caching by default remain prohibited.

## Current Structure

### Application Entry And UI

- `src/relationship_lifelog_agent/app.py` loads config, builds the Gradio UI,
  and launches with `share=False`, host `127.0.0.1`, and no static allowed paths.
- `src/relationship_lifelog_agent/ui/chat_ui.py` builds the ChatGPT-like UI.
  It already supports:
  - chat history
  - input and send controls
  - hidden settings accordion
  - backend/profile/date range/mode/debug settings
  - manual profile editing
  - review action controls
  - streamed safe progress messages
- Streaming progress is currently grouped into one `処理プロセス` assistant
  message and final answers keep process details collapsed in the completed
  state.

### Conversation Agent Flow

Main path:

```text
Chat UI
  -> stream_answer / answer_with_reasoning
  -> understand_question
  -> resolve_profile
  -> check_answerability
  -> build_conversation_query_plan
  -> answer_chat legacy executor
  -> optional LLM answer composition
  -> safety/privacy guard
  -> ChatResponse
```

Important files:

- `agent/reasoning_orchestrator.py`
  - main conversation-aware entry point
  - emits safe streaming progress events
  - tracks `LlmUsageTrace`
  - appends reasoning sections
- `agent/conversation_state.py`
  - short-lived context such as active profile, previous intents, dates, and plan
- `agent/question_understanding.py`
  - rule fallback plus optional local LLM `QuestionFrame`
- `agent/profile_resolver.py`
  - manual profile-only resolution
  - default profile and completeness-aware selection
- `agent/answerability.py`
  - Python verification of required information
  - optional local LLM suggestions for information needs
- `agent/query_planner.py`
  - rule planner plus optional local LLM plan drafting
  - allowed-tool whitelist and alias normalization
- `agent/tool_registry.py`
  - allowed local tools
- `agent/executor.py`
  - legacy deterministic executor for relationship-specific intents
  - calls adapters, loads saved relationship events, performs aggregation
- `agent/answer_composer.py`
  - template composition plus optional local LLM answer phrasing

The current executor still acts as the actual tool execution engine. The newer
conversation planner mostly prepares and explains the plan, then delegates to
the legacy executor. This is acceptable for `safe_window`, but private full modes
need a real runtime loop with observations, batches, source refs, and replanning.

### Current Intents And Window Analysis

Current supported relationship intents include:

- `conflict_frequency`
- `conflict_timeline`
- `conflict_date_lookup`
- `conflict_dates_with_surrounding_media`
- `post_conflict_activity`
- `emotional_note_lookup`
- `monthly_relationship_review`
- `reply_delay_analysis`
- `general_relationship_qa`

Recent additions already support the compound question:

```text
喧嘩した日はいつ？その日の前後の日は写真から何をしていた？
```

The current `conflict_dates_with_surrounding_media` path:

1. Finds conflict or minor-misunderstanding candidate dates.
2. Builds a strict surrounding window from config.
3. Loads small LINE windows.
4. Loads date-near notes.
5. Loads media by exact date range.
6. Optionally asks local LLM to summarize line, note, and media windows.
7. Aggregates counts and dates in Python.

Relevant files:

- `analytics/llm_line_window.py`
- `analytics/llm_note_window.py`
- `analytics/llm_media_summary.py`
- `analytics/llm_cache.py`

These analyzers intentionally pass only short excerpts and safe metadata. They
should remain the `safe_window` implementation.

### LLM Usage Points

`llm/local_client.py` is a local Ollama client. It:

- accepts only local URLs
- uses `/api/chat`
- sets `stream=False`
- sets `think=False`
- can request structured JSON output
- returns structured success/failure results
- does not download models

Current LLM stages:

- question understanding
- information-needs suggestion
- query planning
- line window analysis
- note window analysis
- media day summary
- answer composition

Current usage trace:

- `llm/usage.py` records call count, backend per stage, fallback reasons,
  structured-output status, validation status, and latency.

This is reusable, but private full modes need more granular stage names:

- full manifest planning
- full batch analysis
- full synthesis
- verifier repair pass
- replan decision

### Streaming Progress

`agent/streaming.py` defines `AgentStreamEvent`.

Current event kinds:

- `progress`
- `thought`
- `tool_start`
- `tool_done`
- `llm_start`
- `llm_done`
- `partial_answer`
- `final_answer`
- `warning`
- `error`

The UI currently displays safe process summaries, not raw chain-of-thought. This
is reusable for full modes. Full modes should add events such as:

- manifest build started/done
- full payload policy checked
- batch builder started/done
- batch N/M analysis started/done
- observation stored
- verifier started/done
- replan started/done

Progress content must remain safe for display even if the payload sent to local
LLM contains raw private data.

### DB Schema And Migration

Current relationship DB schema:

- `relationship_profiles`
- `relationship_events`
- `relationship_event_evidence`
- `interaction_metrics`
- `post_conflict_activities`
- `relationship_review_actions`
- `llm_analysis_cache`

Current migration:

- `db/migrate.py`
  - initializes schema
  - preserves existing profiles/events
  - adds profile target/self speaker columns
  - repairs event and evidence contracts
  - ensures `updated_at`
- `db/repository.py`
  - CRUD for profiles, events, evidence, metrics, post-conflict activities,
    review actions, and LLM analysis cache

`llm_analysis_cache` already stores structured results keyed by a source-window
hash. It deliberately does not store raw prompts or raw payloads. This pattern
should be kept. Private full modes need additional tables for runs, batches, and
observations.

### Current Upstream Connection

Upstream access is read-only and adapter-gated.

Important files:

- `agent/memory.py`
  - builds `mock` memory or `upstream_readonly` memory
- `adapters/upstream_sqlite.py`
  - opens SQLite via URI `mode=ro`
  - sets `PRAGMA query_only = ON`
  - provides safe excerpt and query helpers
- `adapters/personal_lifelog.py`
  - current read-only methods:
    - `search_line`
    - `search_media`
    - `search_events`
    - `get_day_summary`
    - `get_month_summary`
- `adapters/notes_lifelog.py`
  - current read-only methods:
    - `search_notes`
    - `search_thoughts`
    - `get_monthly_reflection`
- `adapters/types.py`
  - evidence dataclasses with `source_pointer`, `sensitivity`,
    `confidence`, and `evidence_strength`

The current upstream adapters are designed for safe evidence summaries and short
excerpts. They should not be expanded directly to return raw full-corpus payloads
through the same methods. Private full access should use a separate
`full_context` data access layer so that raw payload flow is explicit and
auditable.

### Current Config

`config.example.yaml` currently includes:

- app safety settings
- adapter backend settings
- upstream path settings
- UI progress settings
- relationship defaults
- privacy redaction settings
- local Ollama settings

`config.local.yaml` exists locally and contains the same broad sections. Its
private values should remain uncommitted and should not be printed in reports.

Current config does not yet include:

- `analysis.mode`
- `agent` runtime budgets
- `full_scan` batching budgets
- `private_full_llm_payload` raw payload toggles
- `vision` settings
- raw prompt logging/cache flags
- source-ref verification policy

### Current Tests

The test suite covers:

- adapters and mock QA
- schema and repository
- profiles
- router and conversation agent
- LLM integration
- answer safety and privacy
- streaming UI
- upstream inspect/smoke/identities
- dry-run and DB backup
- compound media question behavior
- evidence quality

This coverage is a good base. Private full modes need a new group of tests
around explicit raw payload permissions, batch coverage, source-ref verification,
and no raw persistence by default.

## Target Architecture

The target should keep the current `safe_window` path intact and add an explicit
private full runtime alongside it.

```text
Chat UI
  -> AnalysisModeResolver
  -> AgentRuntime
      -> QuestionUnderstanding
      -> TaskDecomposer
      -> ProfileResolver
      -> AnswerabilityChecker
      -> TaskGraphBuilder
      -> ToolPlanner
      -> ToolExecutor
      -> ObservationStore
      -> FullContextDataAccess
      -> FullDataManifest
      -> ContextBudgeter
      -> BatchBuilder
      -> PromptPacker
      -> FullRangeLlmAnalyzer
      -> EvidenceGraph
      -> SourceRefVerifier
      -> AnswerComposer
      -> SafetyPrivacyGuard
```

### Analysis Modes

#### `safe_window`

Use the existing implementation:

- source pointers
- short excerpts
- strict date filters
- local LLM over small windows only
- no raw full payload
- no photo path, exact GPS, face crops, embeddings, or private paths in prompts

#### `private_full_range`

Use explicit user permission and a required date range.

Allowed local LLM payload can include, depending on policy toggles:

- full LINE text inside the selected range
- full note text inside the selected range
- photo paths inside the selected range
- exact GPS inside the selected range
- face crop refs or paths inside the selected range
- face embeddings if explicitly allowed
- private file paths if explicitly allowed
- unverified person and speaker candidates

Required protections:

- upstream remains read-only
- no external API
- no cloud model
- local Ollama only
- raw prompt logging default false
- raw payload cache default false
- no raw payload stored in relationship DB by default
- final answer source refs verified

#### `private_full_corpus`

Use explicit user permission and strict budgets.

Required:

- manifest before analysis
- estimated token budget
- batch strategy
- max runtime
- max steps
- max tool calls
- max LLM calls
- max batches
- resumable run records
- source-ref verification
- user-visible warnings that this may be slow

### New Module Set

Recommended new package:

```text
src/relationship_lifelog_agent/full_context/
  __init__.py
  types.py
  policy.py
  data_access.py
  manifest.py
  context_budget.py
  batch_builder.py
  prompt_packer.py
  full_range_analyzer.py
  observation_store.py
  task_graph.py
  runtime.py
```

Recommended verification package:

```text
src/relationship_lifelog_agent/verification/
  __init__.py
  source_ref_registry.py
  full_answer_verifier.py
```

Recommended LLM schema additions:

```text
src/relationship_lifelog_agent/llm/full_context_schemas.py
```

Recommended CLI additions:

```text
python -m relationship_lifelog_agent.cli full-context inspect
python -m relationship_lifelog_agent.cli full-context plan
python -m relationship_lifelog_agent.cli full-context pack-preview
python -m relationship_lifelog_agent.cli full-context analyze
python -m relationship_lifelog_agent.cli full-context runs list
python -m relationship_lifelog_agent.cli full-context runs show --id ...
```

## Existing Reuse Policy

Reuse as-is or with narrow extensions:

- `config.py`
  - add new dataclasses rather than replacing existing settings
- `llm/local_client.py`
  - keep local-only Ollama enforcement
  - add optional larger context settings and safe non-logging hooks
- `llm/usage.py`
  - extend stage names for full-context calls
- `agent/reasoning_orchestrator.py`
  - keep as the chat-facing facade
  - route by analysis mode to current safe path or new full runtime
- `agent/streaming.py`
  - reuse `AgentStreamEvent`
- `ui/chat_ui.py`
  - reuse streaming pattern and settings accordion
  - add analysis mode and raw payload toggles only inside the accordion
- `adapters/upstream_sqlite.py`
  - reuse read-only connection helpers
- `agent/profile_resolver.py`
  - keep Python-only manual profile resolution
- `profiles.py`
  - keep manual target/self speaker semantics
- `privacy/guard.py` and `privacy/redaction.py`
  - keep final answer guard
  - add private-full-aware output checks, not prompt checks
- `analytics/llm_*_window.py`
  - keep as `safe_window` analyzers
- `db/repository.py`
  - extend with full analysis run/batch/observation repositories
- existing tests for safe behavior
  - keep and continue to run for `safe_window`

## Replacement Candidates

Do not delete these immediately. Replace gradually after new full runtime tests
are passing.

- `agent/executor.py`
  - keep as the current `safe_window` executor
  - eventually split tool execution into explicit tool classes/functions used by
    both safe and full runtimes
- `agent/planner.py`
  - legacy adapter-call planner can remain for safe intents
  - future private full planning should use `TaskGraphBuilder`
- `analytics/dry_run.py`
  - keep relationship candidate extraction
  - later reuse source-ref verifier and manifest summaries
- current adapter search methods
  - keep for safe evidence queries
  - do not repurpose them for raw full payload

Code that should not be replaced:

- local-only safety validation
- read-only upstream SQLite connection helper
- manual profile rules
- review actions and relationship DB event review
- public redaction and answer safety tests

## Implementation Phases

### Phase 0: Plan Only

Deliver this document.

Validation:

- `pytest`
- `git diff --check`

### Phase 1: Guidance And Docs

Update:

- `AGENTS.md`
- `README.md`
- `docs/application_overview.html`
- `docs/ADAPTER_CONTRACT.md`

Status:

- `AGENTS.md`, `README.md`, and `docs/application_overview.html` now describe
  the Private Full-Corpus direction, analysis modes, local-only constraints, raw
  payload defaults, and source-ref verification requirement.
- `docs/ADAPTER_CONTRACT.md` should be updated when the first private full data
  access interfaces are implemented.

Main change:

- The current guidance says local LLM should not receive the entire LINE log or
  all notes. Convert that into a mode-specific rule:
  - `safe_window`: no raw full payload
  - `private_full_range` / `private_full_corpus`: raw payload may be used only
    with explicit user consent and local Ollama

Tests:

- docs lint via `git diff --check`
- existing pytest

### Phase 2: Config, Policy, And Types

Add config sections:

Status:

- `config.py` now reads `analysis`, nested `analysis.full_scan`,
  `private_full_llm_payload`, `llm.num_ctx`, and `vision`.
- `config.example.yaml` documents the new sections.
- `full_context/types.py` defines the first full-context dataclasses.
- `privacy/raw_payload_policy.py` centralizes mode-aware raw payload inclusion.
- Full data access, batching, prompt packing, and LLM analysis remain future
  phases.

```yaml
analysis:
  mode: safe_window
  allow_full_range: false
  allow_full_corpus: false

agent:
  max_steps: 20
  max_tool_calls: 50
  max_llm_calls: 20
  max_runtime_seconds: 180
  max_evidence_items: 20
  max_private_excerpts: 10

full_scan:
  batch_strategy: hybrid
  max_items_per_batch: 200
  max_chars_per_batch: 80000
  overlap_items: 5
  max_batches_per_run: 200
  max_reanalysis_rounds: 5
  summarize_after_each_batch: true
  preserve_raw_refs: true

private_full_llm_payload:
  allow_raw_line_text: false
  allow_raw_note_text: false
  allow_photo_paths: false
  allow_exact_gps: false
  allow_face_crops: false
  allow_face_embeddings: false
  allow_raw_face_embedding_values: false
  allow_private_file_paths: false
  allow_unverified_person_candidates: false
  allow_unverified_speaker_candidates: false
  allow_full_prompt_logging: false
  allow_raw_payload_cache: false
```

Add types:

- `AnalysisMode`
- `RawPayloadPolicy`
- `FullLineItem`
- `FullNoteItem`
- `FullMediaItem`
- `FullFaceItem`
- `FullLocationItem`
- `FullDataManifest`
- `FullScanBatch`

Tests:

- defaults remain safe
- public mode cannot use private full modes
- raw prompt logging/cache default false
- unsafe external/cloud settings still rejected

### Phase 3: Full Context Read-Only Data Access

Create `full_context/data_access.py`.

Responsibilities:

- open upstream SQLite read-only
- set query-only mode
- load full line range
- load full note range
- load full media range
- load face/location data where upstream schema supports it
- include raw fields only according to `RawPayloadPolicy`
- attach stable `source_ref`
- never write to upstream

Tests:

- safe mode returns no raw text/path/GPS/face payload
- private full range includes allowed raw fields
- date range is strict
- upstream DB remains read-only
- unset upstream path fails safely

### Phase 4: Manifest, Budget, And Batch Builder

Status: implemented as a planning layer. It accepts already-loaded
`Full*Item` objects, builds a manifest, estimates context budget, and creates
source-ref-preserving batches. The read-only full data access loader remains a
separate phase.

Implemented:

- `full_context/manifest.py`
- `full_context/context_budget.py`
- `full_context/batch_builder.py`

Responsibilities:

- count source items
- estimate characters/tokens
- choose `single_context` or `iterative_full_scan`
- create batches with overlap
- guarantee every source_ref appears in at least one batch
- detect dropped refs

Tests:

- manifest counts
- token budget routing
- batch source-ref coverage
- overlap behavior
- max batch constraints

CLI:

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml full-context plan \
  --profile-id 1 \
  --date-from 2024-12-01 \
  --date-to 2024-12-31
```

The command currently emits a safe plan summary without loading raw upstream
data. After Phase 3 data access is connected, this command should use the
read-only loader output as its item set.

### Phase 5: Prompt Packer

Create `full_context/prompt_packer.py`.

Status: implemented as the first prompt packing layer. It converts
`FullPromptBundle` / `FullScanBatch` inputs into structured local-LLM prompts,
keeps `source_id` and `source_ref` in every item, and applies
`RawPayloadPolicy` before raw text, paths, GPS, face crops, embeddings, or
unverified candidates enter the prompt. Full prompt logging and raw payload
cache metadata remain false by default.

CLI:

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml full-context pack-preview \
  --profile-id 1 \
  --date-from 2024-12-01 \
  --date-to 2024-12-31 \
  --max-preview-chars 2000
```

Responsibilities:

- build prompt bundles from full items
- apply raw payload policy
- include source refs
- avoid prompt logging unless explicitly enabled
- provide pack-preview counts and redacted previews for CLI

Tests:

- `safe_window` prompt excludes raw fields
- `private_full_range` prompt includes only enabled raw fields
- source refs are present
- raw prompt logging remains disabled by default
- no raw payload cache by default

### Phase 6: Full-Range LLM Analyzer

Create `full_context/full_range_analyzer.py` and
`llm/full_context_schemas.py`.

Status: implemented as the first local LLM analyzer layer. It supports
single-context analysis, iterative batch analysis, per-batch structured
analysis, synthesis, Pydantic validation, retry/fallback behavior, source-ref
filtering, and max LLM call budgets. It refuses to run for `safe_window`
prompts.

CLI:

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml full-context analyze \
  --profile-id 1 \
  --date-from 2024-12-01 \
  --date-to 2024-12-31 \
  --question "この期間に何があった？" \
  --dry-run true
```

Responsibilities:

- single-context analysis if budget allows
- iterative batch analysis if not
- structured JSON validation
- batch synthesis
- fallback or retry on invalid JSON
- source_ref emission in every observation

Tests:

- mock LLM single-context analysis
- mock LLM batch analysis
- invalid JSON fallback/error
- max LLM call budget
- source refs retained
- no raw output in progress events

### Phase 7: Persistence

Add DB tables:

- `full_analysis_runs`
- `full_analysis_batches`
- `full_analysis_observations`

Keep or extend:

- `llm_analysis_cache`

Rules:

- store manifest JSON, source refs, hashes, structured results, and summaries
- do not store raw prompt or raw payload by default
- allow explicit future opt-in only after separate review

Tests:

- migration creates tables
- existing DB migrates without data loss
- run/batch/observation CRUD
- raw payload not persisted by default

### Phase 8: Agent Runtime And Replanner

Create `full_context/runtime.py` or `agent/runtime.py`.

Runtime loop:

```text
understand question
decompose task
resolve profile
check answerability
build task graph
load manifest
budget context
pack prompt or batches
execute analysis
store observations
verify evidence
replan if needed
compose answer
guard final output
```

Keep `reasoning_orchestrator.py` as the chat-facing coordinator:

- `safe_window` -> existing path
- `private_full_range` / `private_full_corpus` -> new runtime

Tests:

- max steps enforced
- max runtime enforced through injectable clock
- missing profile stops before data access
- missing full-corpus permission stops before raw payload
- replan loop stops cleanly

### Phase 9: UI Mode Selector

Extend hidden settings accordion:

- analysis mode
- date scope
- raw payload include toggles
- max runtime
- max LLM calls
- max batches
- pack-preview button or CLI-first link

Keep UI simple:

- no large dashboard
- no complex review tabs
- default mode remains `safe_window`

Tests:

- UI passes selected analysis mode
- raw toggles default false
- public mode disables or rejects private full modes
- progress events remain safe

### Phase 10: Source-Ref Verification

Create `verification/full_answer_verifier.py`.

Verification checks:

- source refs exist
- dates are within requested scope
- counts match Python facts
- final claims have cited evidence
- relationship labels are user-configured only
- person/speaker links are not inferred
- public mode has no raw leakage
- unsupported claims are downgraded or removed

Tests:

- missing source_ref detected
- date outside scope detected
- count mismatch detected
- unsupported relationship assertion detected
- public raw leakage detected

### Phase 11: CLI

Add:

- `full-context inspect`
- `full-context plan`
- `full-context pack-preview`
- `full-context analyze`
- `full-context runs list`
- `full-context runs show`

Output safety:

- inspect and plan are counts/manifest only by default
- pack-preview truncates and requires explicit private mode
- no actual private path printed unless explicitly allowed in private full mode
- markdown/json outputs under ignored export paths

Tests:

- inspect does not write DB
- plan creates no raw payload
- pack-preview respects policy
- analyze creates run/batch records without raw payload

### Phase 12: Full Corpus Mode

Enable all-date operation after range mode is stable.

Requirements:

- explicit confirmation
- high budgets
- resumable runs
- progress by batch
- cancellation-safe persistence
- source-ref verification

Tests:

- all-corpus manifest
- batch cap stops safely
- resumable run lookup
- no raw persistence by default

### Phase 13: Evaluation

Extend eval cases for:

- multi-hop questions
- range-level full analysis
- full-corpus summary
- source-ref verification
- raw payload permission boundaries
- public redaction
- unsupported claim rejection
- batch synthesis consistency

### Phase 14: Cleanup

After full runtime is stable:

- split legacy `executor.py` into reusable tool functions
- keep `safe_window` as a first-class mode
- remove duplicated planner code only when covered
- update docs and overview

## Risks

### Privacy Boundary Risk

Private full modes intentionally allow raw local LLM payload. This is a
different privacy boundary from the current app. The implementation must keep
three separate boundaries:

1. Payload to local Ollama.
2. Persistent DB/cache.
3. User-visible final answer/progress.

Raw data may be allowed in boundary 1 only after explicit user consent. It
should remain disallowed by default in boundaries 2 and 3.

### Instruction Conflict Risk

Current project guidance says the local LLM should not receive the entire LINE
log or all notes. The new specification changes this only for explicitly enabled
private full modes. Phase 1 must update `AGENTS.md` and docs before code enables
private full payloads.

### Context And Latency Risk

Full corpus analysis can exceed the model context and be slow. Batch budgets and
runtime limits must be implemented before any full-corpus UI entry point.

### Source-Ref Hallucination Risk

The LLM may produce plausible but unsupported claims. Full modes must require
source refs and final verification before answering.

### Raw Persistence Risk

It is easy to accidentally store prompt bundles or raw payloads in cache.
Repository methods and tests should enforce structured-result-only persistence
by default.

### Upstream Schema Drift Risk

Full raw access needs more upstream columns than current safe adapters. The data
access layer should be schema-diagnostic and optional-field tolerant, with
counts-only warnings when unavailable.

### UI Complexity Risk

Private full toggles can make the UI heavy. Keep defaults hidden in the settings
accordion and prefer CLI inspect/plan/pack-preview first.

## Test Plan

### Existing Tests To Keep Passing

Run all current tests throughout:

```bash
pytest
```

Core existing areas that must not regress:

- `test_privacy.py`
- `test_answer_safety.py`
- `test_profiles.py`
- `test_llm_integration.py`
- `test_streaming.py`
- `test_ui.py`
- `test_adapters.py`
- `test_schema.py`
- upstream inspect/smoke/identity tests

### New Test Groups

Add new tests, likely:

```text
tests/test_private_full_config.py
tests/test_full_context_data_access.py
tests/test_full_context_manifest.py
tests/test_full_context_batching.py
tests/test_full_context_prompt_packer.py
tests/test_full_range_analyzer.py
tests/test_full_answer_verifier.py
tests/test_full_context_runtime.py
tests/test_full_context_cli.py
```

Required checks:

- `analysis.mode` defaults to `safe_window`
- private full modes require explicit config/UI permission
- public mode rejects private full payload
- raw prompt logging default false
- raw payload cache default false
- safe window excludes raw fields
- private full range includes only explicitly allowed raw fields
- upstream SQLite is read-only
- date ranges are strict
- full manifest counts are stable
- batches cover all source refs
- raw payload is not persisted by default
- LLM structured JSON is validated
- invalid JSON falls back or fails safely
- source refs are verified before final answer
- final answer still passes forbidden phrase checks
- progress messages never show raw data

## First Files To Touch In The Next Phase

1. `AGENTS.md`
   - convert the current "LLM as window analyst" rule into a mode-specific rule.
2. `config.py` and `config.example.yaml`
   - add analysis mode, full scan budgets, raw payload policy, and defaults.
3. `src/relationship_lifelog_agent/full_context/types.py`
   - introduce full-context item and manifest dataclasses.
4. `src/relationship_lifelog_agent/full_context/policy.py`
   - centralize raw payload permission logic.
5. `tests/test_private_full_config.py`
   - lock down defaults before any raw data path is implemented.

After these are in place, implement read-only full data access in a separate
module instead of expanding the existing safe adapters.
