# Adapter Contract

This document defines the adapter contract for `relationship_lifelog_agent`.
It must be satisfied by both the mock adapters and the opt-in read-only
upstream adapters.

Real upstream access is available only through the `upstream_readonly` backend.
The default backend remains `mock`.

## Principles

Adapters are evidence readers, not judgment engines.

They may return observable source pointers, dates, summaries, short controlled
excerpts, confidence values, and evidence strength values. They must not infer
relationship labels, intimacy, other people's feelings, face identity, or LINE
speaker identity.

Default behavior remains:

```yaml
adapter:
  backend: mock
  upstream_access_mode: readonly
```

The `upstream_readonly` backend is opt-in. It must use read-only SQLite
connections and must fail safely when upstream DB paths are unset.

## Adapter Methods

### Personal lifelog adapter

#### `search_line(query, date_from=None, date_to=None, speaker_id=None, limit=50)`

Purpose:

Return LINE-like evidence candidates relevant to a query, date range, and
optional manually configured speaker scope.

Output:

`list[LineEvidence]`

Allowed evidence:

- source pointer to an upstream LINE message or mock fixture
- message date/time
- speaker/source IDs only when configured or already manually reviewed
- short private-mode excerpt when needed
- no full LINE text in public mode

#### `search_media(query, date_from=None, date_to=None, person_id=None, place_id=None, limit=50)`

Purpose:

Return media-derived evidence candidates such as photo/video metadata,
captions, OCR summaries, place links, and manually reviewed person links.

Output:

`list[MediaEvidence]`

Allowed evidence:

- source pointer to media, caption, OCR, or place evidence
- media type
- coarse place/person IDs when configured
- no real photo, face crop, exact GPS, or private file path

#### `search_events(date_from=None, date_to=None, event_type=None, person_id=None, place_id=None, limit=50)`

Purpose:

Return upstream event candidates relevant to relationship analytics.

Output:

`list[EventEvidence]`

Allowed evidence:

- source pointer to event/timeline rows
- event date and cautious event type
- review status
- confidence and evidence strength
- evidence counts or source IDs

#### `get_day_summary(date, mode="private")`

Purpose:

Return aggregate counts and safe summaries for one day.

Output:

`dict[str, str]`

Allowed evidence:

- counts by evidence type
- concise candidate summary
- no raw private text

#### `get_month_summary(month, mode="private")`

Purpose:

Return aggregate counts and safe summaries for one month.

Output:

`dict[str, str]`

Allowed evidence:

- counts, representative date labels, and high-level candidate patterns
- no raw private text, exact GPS, real photos, or face artifacts

### Notes lifelog adapter

#### `search_notes(query, date_from=None, date_to=None, limit=50)`

Purpose:

Return note-level evidence candidates relevant to a relationship question.

Output:

`list[NoteEvidence]`

Allowed evidence:

- source pointer to note or summary row
- note date/date label
- safe summary and optional short private-mode excerpt
- no full note body

#### `search_thoughts(query, date_from=None, date_to=None, limit=50)`

Purpose:

Return thought/reflection candidates extracted from notes.

Output:

`list[ThoughtEvidence]`

Allowed evidence:

- source pointer to thought row
- thought type, date label, themes, emotion label when available
- cautious summary
- no claim that lyrics, quotes, fiction, or noisy notes are the user's actual
  feeling without a confidence warning

#### `get_monthly_reflection(month)`

Purpose:

Return an existing monthly reflection candidate.

Output:

`MonthlyReflection`

Allowed evidence:

- source pointer to monthly reflection or timeline snapshot
- month, summary, confidence, evidence strength
- no generation or upsert in a read-only adapter

## Output Types

All adapter evidence objects must expose:

| Field | Meaning |
|---|---|
| `source_id` | Upstream or mock record identifier. |
| `source_type` | Stable evidence type, such as `line_message`, `media_caption`, `note`, `thought`, or `monthly_reflection`. |
| `source_pointer` | Stable pointer string for source lookup. Defaults to `<source_type>:<source_id>`. |
| `date` | ISO date or date-like label. |
| `role` | Evidence role such as `primary`, `supporting`, `context`, `before_event`, or `after_event`. |
| `summary` | Short safe summary. |
| `excerpt` | Optional short excerpt. Must be omitted in public mode unless the evidence is explicitly public-safe. |
| `sensitivity` | One of `public`, `private`, or `sensitive`. |
| `confidence` | 0.0 to 1.0 confidence score. |
| `evidence_strength` | 0.0 to 1.0 strength score for the evidence item. |
| `metadata` | Small structured metadata. No raw private records. |

## Source Pointer Rules

`source_pointer` must be stable enough to re-find the upstream record without
copying that record into the relationship DB.

Recommended upstream formats:

```text
personal_lifelog_rag:line_messages:<message_id>
personal_lifelog_rag:media_items:<media_id>
personal_lifelog_rag:events:<event_id>
notes_lifelog_rag:notes:<note_id>
notes_lifelog_rag:thoughts:<thought_id>
notes_lifelog_rag:monthly_reflections:<month>
```

Mock adapters may use:

```text
line_message:mock-line-jan10-conflict
note:mock-note-jan10-reflection
```

The pointer must not contain absolute private paths, raw text, GPS coordinates,
face crop paths, or photo paths.

## Sensitivity Rules

Allowed values:

| Sensitivity | Meaning |
|---|---|
| `public` | Safe to show in public mode after normal answer guard checks. |
| `private` | May be shown only in private mode, with short excerpts only when needed. |
| `sensitive` | Highly private evidence. Excerpts should normally be omitted even in private mode unless the user explicitly asks for source review. |

Default sensitivity is `private`.

## Excerpt Control

Private mode:

- May include a short excerpt when needed.
- Must still avoid full LINE text, full note text, exact GPS, face data, and
  private paths.
- Must not infer other people's inner feelings or relationship status.

Public mode:

- Must omit `excerpt` unless `sensitivity == "public"`.
- Must redact person names, relationship labels, raw text, exact GPS, face data,
  real photo paths, source file paths, and private paths.
- Must preserve source pointers only if they are non-identifying.

The type helpers support this by converting evidence with:

```python
item.as_evidence_item(mode="public")
bundle.to_evidence_items(mode="public")
```

## Read-only Rules

Upstream adapters must:

- Open upstream SQLite databases using URI `mode=ro`.
- Detect table/column capabilities before querying.
- Fail closed when required tables are missing.
- Never call upstream functions that initialize, migrate, generate, upsert, or
  otherwise write data.
- Never copy upstream raw records into `relationship.sqlite`.
- Store only relationship-specific derived data and source pointers in the
  relationship DB.

The required DB connection shape is:

```python
sqlite3.connect("file:<upstream-db>?mode=ro", uri=True)
```

## Mock Adapter Policy

Mock adapters remain the default backend and must keep passing tests.

Mock data must:

- Be synthetic and privacy-safe.
- Include enough evidence to answer MVP questions.
- Use the same fields as read-only upstream adapters.
- Avoid real names, real messages, real notes, exact GPS, face data, and photo
  files.

## Forbidden Behavior

Adapters must not:

- Edit upstream application files.
- Write to upstream DBs.
- Trigger upstream model downloads.
- Use external APIs.
- Enable Gradio public sharing.
- Copy raw LINE messages, raw notes, exact GPS, face embeddings, face crops, or
  photos into this app.
- Infer partner, lover, family, friend, intimacy, or relationship status.
- Link face identity and LINE speaker identity automatically.
- Return relationship labels unless manually configured by the user in private
  mode.
