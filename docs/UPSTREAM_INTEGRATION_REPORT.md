# Upstream Integration Report

This report investigates read-only integration paths from
`relationship_lifelog_agent` to the two upstream local lifelog applications:

- `../personal_lifelog_rag`
- `../notes_lifelog_rag`

Scope of this report:

- Investigation only.
- No upstream application files were edited.
- No upstream database writes were performed.
- No real LINE message body, full note body, exact GPS coordinate, face data,
  photo file, face crop, or private evidence excerpt is included here.
- Paths are shown as repository-relative paths where practical.

## Summary

The safest initial integration path is direct SQLite read-only access through
explicit `mode=ro` connections, with tightly scoped SQL queries and adapter-side
redaction. Both upstream applications already have useful Python modules and
CLI commands, but many of those APIs initialize or migrate schema as part of
normal use. That behavior is useful inside the upstream apps, but it is not
strictly read-only for `relationship_lifelog_agent`.

Recommended implementation order:

1. Keep mock adapters as default.
2. Add opt-in real adapters that open upstream SQLite DBs with read-only URI
   connections.
3. Query only pointer fields, IDs, dates, safe short summaries, counts, and
   manually reviewed relationship-neutral links.
4. Add upstream read-only export/API functions later only if direct SQL becomes
   too brittle.

Primary upstream DB candidates:

| App | Relative DB path | Main use |
|---|---|---|
| `personal_lifelog_rag` | `data/db/lifelog.sqlite` | LINE, media, events, places, manually reviewed people links |
| `notes_lifelog_rag` | `data/processed/notes.db` | notes, thoughts, events, monthly reflections/timelines |

## personal_lifelog_rag findings

### Project structure

Observed public/project files:

- `pyproject.toml`
- `README.md`
- `AGENTS.md`
- `src/personal_lifelog_rag/`
- `configs/`
- `scripts/`
- `docs/`
- `reports/`
- `eval_outputs/`
- `tests/`
- `data/db/`
- `data/raw/`, `data/processed/`, `data/thumbnails/`, `data/faces/`

The project is packaged as `personal-lifelog-rag` and exposes the console script:

```text
personal-lifelog-rag = personal_lifelog_rag.app.cli:main
```

The README describes a local-first multimodal lifelog app over photos/videos,
LINE histories, EXIF/GPS metadata, OCR, VLM captions, face review, places,
events, search, ask, and privacy audit workflows.

### Relevant source layout

Useful modules for investigation:

| Area | Candidate modules | Notes |
|---|---|---|
| Config | `core/config.py` | Default DB path and local-only defaults. |
| SQLite schema | `db/schema.py` | Defines tables and indexes. |
| Repository | `db/repository.py` | Rich read/write repository. Read methods currently initialize schema. |
| LINE | `ingest/line_parser.py`, `line/call_index.py`, `line/person_links.py` | LINE messages, calls, speaker links. |
| Media | `ingest/photo_ingest.py`, `vlm/`, `ocr/`, `embeddings/` | Media metadata, captions, OCR, embeddings. |
| Timeline | `timeline/event_builder.py`, `timeline/event_reports.py` | Event candidates and event reports. |
| Retrieval | `retrieval/local_search.py`, `retrieval/monthly_summary.py`, `retrieval/query_router.py` | Search/QA and monthly summaries. |
| Places | `places/location_store.py`, `places/place_dictionary.py`, `places/redaction.py` | GPS-derived places and redacted labels. |
| People | `people/integration.py`, `faces/person_service.py` | Manual person links and accepted-cluster gates. |
| UI services | `ui/ask_service.py`, `ui/monthly_summary_service.py`, `ui/unified_search_service.py` | App-facing service layer. |

### Useful Python API candidates

Potential APIs, with read-only caveats:

| API | Candidate use | Read-only caveat |
|---|---|---|
| `personal_lifelog_rag.db.repository.resolve_db_path` | Resolve upstream DB path. | Safe path helper. |
| `personal_lifelog_rag.db.repository.LifelogRepository.list_line_messages` | LINE date/keyword reads. | Calls schema initialization through repository helpers. |
| `LifelogRepository.list_line_call_events` | Call records and call durations. | Calls schema initialization. |
| `LifelogRepository.list_media_items` | Media by date range. | May expose file paths/GPS; adapter must sanitize. |
| `LifelogRepository.list_events` | Timeline events by date/keyword. | Calls schema initialization. |
| `LifelogRepository.search_text_records` | Combined local text search. | Searches raw text-bearing tables; must avoid returning raw bodies. |
| `retrieval.local_search.local_text_search` | Existing ranking over events/LINE/media/OCR/VLM. | Formats samples that may include text snippets. Do not pass through directly. |
| `retrieval.monthly_summary.build_monthly_summary_report` | Day/month aggregate counts. | Uses repository methods that initialize schema; output can include labels. |
| `places.location_store.public_place_label` | Safe place labeling. | Useful if imported, but direct SQL may be simpler. |
| `faces.person_service.public_person_name` | Safe person labeling. | Only use manually configured person rows; never infer relation. |

Conclusion: the Python modules are very useful as references, but the first real
relationship adapter should not call these repository methods against the live
DB until a read-only connection path exists.

### Useful CLI candidates

Relevant CLI commands:

| Command | Usefulness | Caveat |
|---|---|---|
| `search` | Keyword search over local SQLite records. | Can emit matched samples/snippets; not ideal for adapter ingestion. |
| `qa` / `ask` | Existing natural-language QA. | Produces full answers; not structured enough and may include private evidence. |
| `image-search` / `multimodal-search` | Media/VLM/OCR search. | Useful for future exploration; output may include private media metadata. |
| `list-events` | Event date-range listing. | Could be useful with `--json`, but CLI may not guarantee minimal private output. |
| `event-stats` | Aggregate event counts. | Useful for aggregate checks. |
| `call-stats` / `search-calls` | Call metrics and call event search. | Good for reply/call metrics if output is sanitized. |
| `places list`, `places list-clusters` | Place labels/clusters. | Must avoid exact coordinates and private labels in public mode. |
| `persons list`, `line-speakers show-links` | Manual person and LINE speaker mappings. | Relationship labels must still be user-manual in this app. |
| `event-people-show`, `media-people-show` | Manually verified people attached to events/media. | Use only accepted/manual links; no face-to-speaker inference. |
| `privacy-audit --public` | Safety verification. | Diagnostic command, not an adapter source. |

Conclusion: CLI subprocess is a reasonable fallback for diagnostics, but not the
first adapter implementation path because it is less precise and may format
private text unless every command is constrained.

### Export/report/index candidates

Observed generated areas:

- `reports/`: many generated Markdown reports.
- `eval_outputs/`: JSON/Markdown evaluation and analysis outputs.
- No stable `data/exports/` directory was present during this inspection.

These artifacts are not ideal primary adapter sources. They can be stale,
human-oriented, and may contain private or semi-private summaries depending on
how they were generated. Use them only as fallback diagnostics, not as a
relationship evidence source.

### SQLite database and key table candidates

Observed DB:

```text
data/db/lifelog.sqlite
```

Read-only schema inspection found these relevant table groups:

| Need | Candidate tables | Useful columns/fields |
|---|---|---|
| LINE messages | `line_messages` | `id`, `chat_id`, `sent_at`, `sender`, `text`, `message_type` |
| Calls | `line_call_events` | `message_id`, `chat_id`, `sent_at`, `sender`, `call_status`, `duration_sec` |
| Media | `media_items` | `id`, `media_type`, `captured_at`, `fallback_captured_at`, `file_name`, `caption`, `ocr_text`, GPS columns |
| VLM captions | `media_vlm` | `media_id`, captions/tags/status/confidence fields |
| OCR | `media_ocr` | `media_id`, redacted OCR field, status/confidence fields |
| Timeline events | `events` | `id`, `date`, time fields, title/summary/location/confidence |
| Event evidence | `event_evidence` | `event_id`, `evidence_type`, `evidence_id`, `weight` |
| Location/place | `location_points`, `place_clusters`, `places`, `event_places`, `media_places` | place IDs, privacy level, manual labels, event/media links |
| Manual people | `persons`, `person_face_clusters`, `line_speaker_links`, `media_people`, `event_people` | manually verified people and source links |
| Privacy controls | `privacy_actions`, hidden/searchable flags on multiple tables | visibility gates |

Important safety note:

- Exact GPS columns exist in upstream tables. The relationship adapter must not
  expose exact coordinates.
- Face-related tables exist. The relationship adapter should use only manually
  reviewed person/event/media links if explicitly needed, never face crops or
  embeddings.
- LINE speaker links exist, but face identity and LINE speaker identity must
  not be automatically linked by `relationship_lifelog_agent`.

### Data retrieval candidates for relationship adapter

| Relationship need | Preferred upstream source |
|---|---|
| Conflict/apology/scheduling signal candidates | `line_messages` via read-only SQL with date/query filters. Return pointer + short private excerpt only when allowed. |
| Reply delay metrics | `line_messages` grouped by `chat_id`, `sender`, and `sent_at`; compute delays in relationship app. |
| Calls after conflict | `line_call_events`; aggregate status/duration without raw message text. |
| Post-conflict outings | `events`, `event_evidence`, `media_items`, `event_places`, `media_places`, `places`; optionally VLM/OCR status summaries. |
| Place labels | `places` where not hidden/searchable false; public mode should use safe public/coarse labels. |
| Person-filtered media/events | `media_people`, `event_people`, `persons`; only manual/verified rows and only when user configured a profile. |
| Day/month summary | Prefer aggregate SQL over events/media/line/calls rather than formatted upstream reports. |

## notes_lifelog_rag findings

### Project structure

Observed public/project files:

- `pyproject.toml`
- `README.md`
- `AGENTS.md`
- `src/notes_lifelog_rag/`
- `configs/`
- `scripts/`
- `docs/`
- `examples/`
- `data/processed/`
- `data/exports/`
- `tests/`

The project is packaged as `notes-lifelog-rag` and exposes the console script:

```text
notes-lifelog-rag = notes_lifelog_rag.cli:main
```

The README describes a local-first Apple Notes / iPhone Notes RAG app with
SQLite storage, optional local embeddings, hybrid search, structured analysis,
events, thoughts, timeline snapshots, suggestions, and monthly reflections.

### Relevant source layout

| Area | Candidate modules | Notes |
|---|---|---|
| Config | `config.py` | Resolves DB and raw notes paths. |
| SQLite schema | `db/schema.py` | Defines core tables, FTS, indexes, migrations. |
| Ingest | `ingest/importer.py`, `ingest/parsers.py` | Note import and parsing. |
| Search | `search/keyword.py`, `search/hybrid.py` | Keyword/FTS plus optional local embedding/rerank search. |
| Analysis | `analysis/service.py` | Summaries, categories, events, thoughts. |
| Timeline | `timeline/service.py`, `timeline/month_narrative.py`, `timeline/review.py` | Timeline, monthly snapshots, review actions. |
| UI services | `ui/services.py`, `ui/renderers.py` | App-facing service functions. |
| LLM/runtime | `llm/`, `runtime/`, `models/` | Local-only model wrappers/status. |

### Useful Python API candidates

Potential APIs, with read-only caveats:

| API | Candidate use | Read-only caveat |
|---|---|---|
| `notes_lifelog_rag.config.database_path` | Resolve DB path. | Safe helper. |
| `notes_lifelog_rag.search.keyword.search_notes` | Keyword/FTS note search. | Calls `init_db`; not strict read-only. |
| `notes_lifelog_rag.search.hybrid.hybrid_search_notes` | Hybrid keyword/vector/rerank search. | Calls search and may load optional local models. Not ideal for first adapter. |
| `timeline.service.build_timeline` | Events/thoughts/note fallback by month. | Calls `init_db`; not strict read-only. |
| `timeline.service.list_timeline_months` | Month availability and counts. | Need verify write behavior before live use. |
| `timeline.service.get_month_timeline_snapshot` | Stored monthly snapshot retrieval. | Prefer direct SQL for read-only. |
| `timeline.service.build_monthly_reflection` | Monthly reflection construction. | Stores/upserts reflection; do not call in read-only adapter. |
| `ui.services.get_timeline`, `get_timeline_month_snapshots`, `get_reflection` | UI-facing wrappers. | Some functions generate/store data; avoid for strict read-only. |

Conclusion: the Python APIs are good references for query semantics, but first
integration should use read-only SQLite SQL, not direct Python service calls.

### Useful CLI candidates

Relevant CLI commands:

| Command | Usefulness | Caveat |
|---|---|---|
| `search` | Note keyword/hybrid search. | Can show snippets; by default avoids full bodies, but command still calls code that initializes DB. |
| `ask` | Evidence-backed note search answer. | Natural-language output is less structured for adapter use. |
| `timeline` | Timeline items or monthly cards. | Some rich modes may generate missing snapshots. |
| `timeline-months` | Month counts and snapshot availability. | Useful diagnostic command. |
| `timeline-qa` / `qa-report` | Timeline quality checks. | Report-oriented, not a primary data source. |
| `reflections` | Monthly reflection output. | Calls builder that stores/upserts reflection; not read-only. |
| `generate-timeline`, `generate-reflections`, `generate-suggestions` | Build outputs. | Write commands; not for this integration layer. |
| `extract-events`, `extract-thoughts`, `summarize-notes`, `analyze-all --dry-run` | Analysis pipelines. | Dry-run is useful for upstream maintenance only, not relationship adapter reads. |

### Export/report/timeline/reflection candidates

Observed generated export area:

```text
data/exports/timelines/
```

It contains timeline and QA report artifacts. These can be useful for human
review, but they are not ideal as the relationship adapter source because they
may be stale and formatted for display. Prefer the SQLite tables that back
these artifacts.

### SQLite database and key table candidates

Observed DB:

```text
data/processed/notes.db
```

Read-only schema inspection found these relevant table groups:

| Need | Candidate tables | Useful columns/fields |
|---|---|---|
| Notes | `notes`, `notes_fts`, `note_chunks` | note IDs, title, relative source path, note dates, FTS index, chunk IDs |
| Summaries | `note_summaries` | generated title, summary fields, importance/confidence, evidence JSON |
| Events | `events` | note ID, title, summary, event type/date, importance/confidence, evidence JSON |
| Thoughts | `thoughts` | note ID, title, summary, thought type, themes, emotion labels, date labels, confidence |
| Categories | `categories`, `note_categories` | note category labels and confidence |
| Suggestions | `suggestions` | revisit/reminder candidates, status, evidence JSON |
| Monthly reflections | `monthly_reflections` | month, summary JSON, evidence JSON, confidence/importance |
| Monthly timeline | `monthly_timeline_snapshots`, `monthly_timeline_items` | month-level overview, item summaries, source pointers, confidence, quality flags |
| Review | `timeline_review_actions` | user review status for timeline items |
| Model cache | `model_runs` | local model output cache and diagnostics |

Important safety note:

- `notes.body` and `note_chunks.text` contain full note text. The relationship
  adapter must not copy or expose full bodies.
- `evidence_json` fields may contain quotes. Public mode should omit or redact;
  private mode should use only short excerpts when needed.
- Timeline/reflection builder functions can write rows. Do not call them from a
  read-only adapter.

### Data retrieval candidates for relationship adapter

| Relationship need | Preferred upstream source |
|---|---|
| `search_notes` | Read-only query over `notes_fts`/`notes`, returning note ID, date, title/summary pointer, and controlled snippet only. |
| `search_thoughts` | `thoughts` joined to `notes`; filter by date labels, thought type, themes/emotion labels, and safe summary fields. |
| `emotional_note_lookup` | `thoughts`, `note_summaries`, `events`, and limited `evidence_json`; avoid full `notes.body`. |
| `get_monthly_reflection` | Prefer existing row in `monthly_reflections`; fallback to `monthly_timeline_snapshots`/`items` aggregate. Do not call builder that stores reflections. |
| Lyrics/quote/noisy note caution | Use `thought_type`, categories, quality flags, and timeline quality flags when present. |

## Adapter mapping

| `relationship_lifelog_agent` adapter method | Upstream data needed | Candidate source |
|---|---|---|
| `search_line(query, date_from, date_to, speaker_id, limit)` | Message IDs, date/time, chat/speaker pointer, short private excerpt or redacted summary, message type | `personal_lifelog_rag`: `line_messages`; optional `line_speaker_links` only for manually configured profile filters |
| `search_media(query, date_from, date_to, person_id, place_id, limit)` | Media IDs, date/time, media type, caption/tag/OCR summaries, place/person pointers | `personal_lifelog_rag`: `media_items`, `media_vlm`, `media_ocr`, `media_places`, `places`, `media_people` |
| `search_events(date_from, date_to, event_type, person_id, place_id, limit)` | Event IDs, date/time, title/summary candidate, confidence, evidence counts, place/person pointers | `personal_lifelog_rag`: `events`, `event_evidence`, `event_places`, `places`, `event_people` |
| `get_day_summary(date, mode)` | Counts and safe summaries for one day | `personal_lifelog_rag`: aggregate SQL over `events`, `line_messages`, `line_call_events`, `media_items`, `event_places` |
| `get_month_summary(month, mode)` | Counts, representative days, category/place/person aggregates | `personal_lifelog_rag`: aggregate SQL modeled after `retrieval/monthly_summary.py`, but using read-only connection |
| `search_notes(query, date_from, date_to, limit)` | Note IDs, title, date labels, source pointer, safe summary/snippet | `notes_lifelog_rag`: `notes_fts`, `notes`, `note_summaries` |
| `search_thoughts(query, date_from, date_to, limit)` | Thought IDs, note ID, thought type, emotion/theme labels, safe summary/evidence pointer | `notes_lifelog_rag`: `thoughts` joined to `notes` |
| `get_monthly_reflection(month)` | Stored reflection or month timeline overview | `notes_lifelog_rag`: `monthly_reflections`; fallback to `monthly_timeline_snapshots` and `monthly_timeline_items` |

## Integration options

### Option A: Python API import

Benefits:

- Reuses upstream routing, search, timeline, monthly summary, and redaction
  helpers.
- Less duplicate SQL logic.
- Easier to stay aligned with upstream semantics if upstream APIs are stable.

Risks:

- Current repository/service methods often call `initialize_schema()` or
  `init_db()`, which may create tables, add columns, seed categories, or store
  generated rows.
- Some methods format raw samples/snippets for UI/CLI.
- Some notes timeline/reflection APIs generate and save missing artifacts.
- Optional model paths/backends may be triggered if not carefully disabled.

Recommendation:

- Not first choice for the initial real adapter.
- Use later only after upstream exposes explicit read-only functions or the
  relationship adapter wraps only verified read-only helpers.

### Option B: CLI subprocess

Benefits:

- Keeps process boundary between apps.
- Uses public command interfaces.
- Useful for diagnostics and manual smoke checks.

Risks:

- Output is presentation-oriented, not stable structured API.
- Some commands write reports or generated artifacts.
- Commands may print snippets or private labels depending on flags.
- Harder to guarantee no raw private output enters logs.

Recommendation:

- Useful fallback for diagnostics.
- Avoid as primary adapter source.

### Option C: Export JSON/report reading

Benefits:

- No live DB lock risk.
- Easy to cache and review.
- Can be made privacy-filtered if generated by a safe upstream export command.

Risks:

- Existing reports/eval outputs may be stale or human-oriented.
- Export content may include summaries or snippets not appropriate for
  relationship answers.
- No stable dedicated export contract exists yet for this app.

Recommendation:

- Good future fallback if upstream adds explicit privacy-safe read-only exports.
- Do not rely on existing reports as primary evidence.

### Option D: SQLite read-only connection

Benefits:

- Strongest read-only guarantee when opened with SQLite URI `mode=ro`.
- Directly maps to relationship adapter needs.
- Lets `relationship_lifelog_agent` control redaction, excerpt length, and
  candidate phrasing.
- Avoids invoking upstream schema migration/build/generation code.

Risks:

- More coupling to upstream schema names.
- Requires careful SQL and compatibility checks.
- Must avoid selecting raw body/text columns unless private mode explicitly
  needs short excerpts.
- Must handle missing tables/columns gracefully.

Recommendation:

- Best initial implementation path.
- Use explicit table/column capability detection and fail closed.

## Recommended approach

### First implementation方式

Use read-only SQLite adapters:

1. Add opt-in config flags/paths for upstream DBs.
2. Open SQLite using URI mode:

   ```text
   file:<relative-upstream-db>?mode=ro
   ```

3. Set adapter-side row factories and query limits.
4. Detect tables and columns with `sqlite_master` / `PRAGMA table_info`.
5. Implement only the relationship adapter contract; do not call upstream
   mutation/build/generation functions.
6. Return source pointers and short, guarded excerpts only when permitted by
   private/public mode.
7. Keep mock adapters as default and tests as the baseline.

### Fallback方式

1. If a required table is missing, return an empty result with a debug-safe
   warning.
2. If read-only SQLite fails due lock/permissions, fall back to mock adapter and
   surface a non-private error message.
3. For diagnostic-only commands, optionally run upstream CLI with safe flags,
   but do not parse private display output as primary evidence.
4. Prefer future upstream export JSON if it is explicitly designed to be
   privacy-safe and read-only.

### Additional upstream read-only API/export that may be useful later

Do not implement these in this prompt. Candidate upstream additions:

For `personal_lifelog_rag`:

- `export-relationship-evidence --from --to --mode private|public --json`
- Read-only Python functions that do not call schema initialization:
  - `readonly_search_line`
  - `readonly_search_media`
  - `readonly_search_events`
  - `readonly_day_summary`
  - `readonly_month_summary`
- Public-safe source-pointer export with IDs, dates, evidence type counts,
  confidence, and redacted labels.

For `notes_lifelog_rag`:

- `export-relationship-notes --from --to --mode private|public --json`
- Read-only Python functions that do not call `init_db`:
  - `readonly_search_notes`
  - `readonly_search_thoughts`
  - `readonly_get_monthly_reflection`
- Dedicated monthly reflection getter that reads existing rows only and never
  generates or upserts.

## Risks

1. Schema coupling:
   Upstream schemas are not formal external contracts. Use capability detection
   and targeted tests with synthetic DB fixtures.

2. Accidental writes:
   Existing upstream Python APIs frequently initialize/migrate schema. Use
   direct `mode=ro` SQLite first.

3. Private text leakage:
   `line_messages.text`, `notes.body`, `note_chunks.text`, OCR text, VLM
   captions, and evidence JSON may contain private content. Select only what is
   needed; redact public mode; cap private excerpts.

4. GPS leakage:
   Personal DB contains exact GPS fields. The relationship adapter should
   convert to place IDs/coarse labels and never return exact coordinates.

5. Face/person over-inference:
   Face tables and manual person tables exist. Relationship app must never infer
   relationship status, intimacy, or face-to-LINE speaker identity.

6. Stale exports:
   Reports and eval outputs may not reflect current DB state. Treat them as
   diagnostics only.

7. Model side effects:
   Notes hybrid search and generation paths can trigger local model/reranker
   code. First adapter should avoid model-loading paths.

## Next implementation tasks

1. Add real adapter classes behind explicit config, while keeping mock as
   default.
2. Add small read-only SQLite connection helpers in `relationship_lifelog_agent`
   that never create or migrate upstream DBs.
3. Add synthetic upstream DB fixtures for both apps with no private data.
4. Implement `PersonalSqliteReadOnlyAdapter`:
   - `search_line`
   - `search_media`
   - `search_events`
   - `get_day_summary`
   - `get_month_summary`
5. Implement `NotesSqliteReadOnlyAdapter`:
   - `search_notes`
   - `search_thoughts`
   - `get_monthly_reflection`
6. Add privacy tests that prove public mode does not expose:
   - person names
   - relationship labels
   - full LINE text
   - full note text
   - exact GPS
   - face/photo paths
   - private source paths
7. Add adapter tests that ensure upstream DB paths are opened read-only and no
   writes/migrations occur.
8. Add a manual integration smoke script that prints only counts and source
   pointer IDs, not raw evidence bodies.
