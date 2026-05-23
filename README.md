# relationship_lifelog_agent

Local-first, privacy-first Relationship Lifelog Agent.

This MVP is a relationship evidence review assistant, not a relationship judgment AI. It uses mock adapters by default, and can opt into read-only SQLite adapters for `personal_lifelog_rag` and `notes_lifelog_rag` without modifying those upstream apps.

## Safety Defaults

- External APIs are disabled.
- Model auto-download is disabled.
- Gradio launches on `127.0.0.1`.
- Gradio `share=True` is never used.
- SQLite stores only relationship-specific derived data and source pointers.
- Upstream SQLite access uses read-only `mode=ro` connections when enabled.

## Setup

```bash
conda create -n relationship_lifelog_agent python=3.11
conda activate relationship_lifelog_agent
pip install -e ".[dev]"
```

## Run

```bash
python -m relationship_lifelog_agent.app
```

The default local URL is:

```text
http://127.0.0.1:7862
```

To use another local port for one run:

```bash
python -m relationship_lifelog_agent.app --port 7863
```

The host remains `127.0.0.1` and Gradio `share=True` remains disabled.

The Chat UI keeps the main screen simple: chat history, input, and send. The
hidden settings accordion lets you choose `mock` or `upstream_readonly`, target
profile, date range, post-conflict window days, private/public mode, and debug
output. Debug output is off by default.

While a question is running, the UI streams safe progress messages such as
question understanding, manual profile checks, information needs, read-only
tool use, evidence filtering, aggregation, and safety checks. These messages are
tool usage summaries, not raw chain-of-thought. They never include raw LINE
text, raw note text, exact GPS, face data, image paths, private paths, or raw
LLM prompts.

When an answer includes relationship DB candidates, the minimal review accordion
lets you save `verify`, `reject`, `mark_as_misunderstanding`, `mark_as_joke`,
`mark_as_reconciled`, or `needs_reanalysis`. Review history is stored in
`relationship_review_actions`; upstream data and original source records are not
modified or deleted.

## Test

```bash
pytest
```

## Doctor

Before using real upstream data, run the local doctor. It reports only
counts-only and redacted diagnostics for config, relationship DB, upstream DB
read-only access, adapter backend, privacy settings, and eval readiness.

```bash
python -m relationship_lifelog_agent.cli doctor
python -m relationship_lifelog_agent.cli doctor --backend mock
python -m relationship_lifelog_agent.cli doctor --backend upstream_readonly
python -m relationship_lifelog_agent.cli doctor --format json
```

`WARN` means the app can continue but needs setup, such as missing upstream DB
paths or missing manual profiles. `ERROR` means unsafe or unusable settings, such
as `allow_gradio_share: true`, external API permission, model auto-download, or
a failed read-only upstream DB connection. Keep `config.local.yaml` out of git.

## Local LLM / Ollama

The Conversation-aware Planning Agent can optionally call a local Ollama model
for question understanding, information-need suggestions, query-plan drafting,
and answer phrasing. This is opt-in only. The app still keeps profile
resolution, identity links, counts, dates, reply-delay metrics, evidence
evaluation, and privacy redaction in Python/SQL. If the local LLM fails or
returns invalid JSON, the rule-based fallback remains active.

The LLM client only accepts local URLs such as `http://127.0.0.1:11434`, never
downloads models, and never calls cloud APIs.

```yaml
llm:
  provider: ollama
  model: null
  base_url: "http://127.0.0.1:11434"
  enabled: false
  require_structured_output: true
  fallback_to_rules: true
  use_for_question_understanding: false
  use_for_information_needs: false
  use_for_query_planning: false
  use_for_answer_composition: false
```

Check local LLM wiring without sending private lifelog data:

```bash
python -m relationship_lifelog_agent.cli --config config.local.yaml llm status
python -m relationship_lifelog_agent.cli --config config.local.yaml llm status --format json
```

In Chat UI debug mode, answers include an LLM usage trace showing which stages
used `llm` versus `rule`/`template`, call count, fallback reasons, structured
output status, and latency.

## Upstream Schema Inspection

Use schema inspection to check whether the upstream SQLite tables and columns
match the current adapter contract. It uses read-only SQLite connections and
prints only table names, column names, row counts, mapping status, coverage, and
warnings. It does not print raw LINE text, raw note text, exact GPS, face data,
photo paths, or private paths.

```bash
python -m relationship_lifelog_agent.cli upstream inspect \
  --backend upstream_readonly \
  --format markdown \
  --output data/exports/upstream_schema_inspection.md

python -m relationship_lifelog_agent.cli upstream inspect --format json
```

## Upstream Counts-only Smoke

Use upstream smoke after schema inspection to exercise the adapters end to end
without writing to the relationship DB. The report contains source counts, date
range coverage, adapter coverage, warnings, redaction status, and
`write_count: 0`. It does not include raw LINE text, raw note text, exact GPS,
face data, photo paths, or private paths.

```bash
python -m relationship_lifelog_agent.cli upstream smoke \
  --backend upstream_readonly \
  --date-from 2025-01-01 \
  --date-to 2025-01-31 \
  --profile-id 1 \
  --output data/exports/upstream_smoke_2025_01.md
```

## Upstream Identity Inventory

Use identity inventory to find manual `relationship_profiles` source IDs without
asking AI to infer relationships or link people to LINE speakers. It reads the
upstream DBs in read-only mode and prints only IDs, counts, dates, verification
status, and anonymized labels in `redacted` mode. It does not print LINE text,
note text, exact GPS, face crops, embeddings, photo paths, private paths, or
relationship-label guesses.

```bash
python -m relationship_lifelog_agent.cli upstream identities \
  --backend upstream_readonly \
  --kind people \
  --privacy-level redacted \
  --format markdown \
  --output data/exports/upstream_people_identities.md

python -m relationship_lifelog_agent.cli upstream identities \
  --backend upstream_readonly \
  --kind line-speakers \
  --privacy-level redacted \
  --format markdown \
  --output data/exports/upstream_line_speakers.md
```

## Eval

The eval runner checks mock/private/public answer quality and safety. It covers
intent routing, required answer sections, evidence presence, forbidden phrase
absence, public redaction, weak-evidence cautions, no inner-feeling assertion,
no relationship-label inference, raw-data leakage patterns, and mock QA
consistency.

```bash
python eval/run_eval.py
python eval/run_eval.py --format json --output data/exports/eval_result.json
```

`upstream_readonly` eval cases are optional and skipped by default. Run them only
when local read-only upstream paths are configured:

```bash
python eval/run_eval.py --include-upstream --config config.local.yaml
```

## Manual Relationship Profiles

Relationship labels and upstream source IDs are user-manual only. The app never
infers that a person is a partner, and never auto-links a person ID to a LINE
speaker ID.

```bash
python -m relationship_lifelog_agent.cli profile create \
  --profile-name "Aさん" \
  --person-source-id "plr:person:4" \
  --line-speaker-source-id "plr:line_speaker:2" \
  --relationship-label "partner" \
  --valid-from "2024-01-01"
python -m relationship_lifelog_agent.cli profile list
```

## Dry-run Relationship Candidate Extraction

Dry-run analysis reads adapter evidence and writes a Markdown report only by
default. It does not insert `relationship_events` candidates into the
relationship DB unless `--write` is explicitly provided.

Reports support three privacy levels:

- `counts-only`: safest choice for real-data checks; emits counts, date range,
  source counts, candidate counts, warnings, and evidence-strength buckets only.
- `redacted`: default; emits anonymized candidate summaries with redacted source
  pointers and no excerpts.
- `private`: allows short excerpts for private local review, while still
  redacting exact GPS, face data, photo paths, private paths, and unsafe
  claims. `public` mode rejects this level.

```bash
python -m relationship_lifelog_agent.cli analyze dry-run \
  --profile-id 1 \
  --date-from 2025-01-01 \
  --date-to 2025-03-31 \
  --backend mock \
  --privacy-level counts-only \
  --output data/exports/dry_run_relationship_2025Q1.md
```

Explicit write mode stores candidate summaries, source pointers, short excerpts,
confidence, and evidence strength in the relationship DB. `counts-only` and
`redacted` write modes omit evidence excerpts; `private` can store short
bounded excerpts. No mode stores raw LINE full text, raw note full text, exact
GPS, face data, or real photos.

Before an explicit write, the CLI creates a relationship DB backup under
`data/backups/`, runs duplicate and safety checks, and prints a counts-only
summary with the backup path redacted.

```bash
python -m relationship_lifelog_agent.cli analyze dry-run \
  --profile-id 1 \
  --date-from 2025-01-01 \
  --date-to 2025-03-31 \
  --backend mock \
  --privacy-level private \
  --write
```

Relationship DB backups are managed explicitly. Restore only affects the
relationship DB; upstream DBs remain read-only and are never restored or
modified by this app.

```bash
python -m relationship_lifelog_agent.cli db backup
python -m relationship_lifelog_agent.cli db backups
python -m relationship_lifelog_agent.cli db restore \
  --backup-path data/backups/relationship_YYYYMMDD_HHMMSS.sqlite
```

## Review-aware Answers

Saved `relationship_events` honor user review actions before AI-derived
candidates:

- `verified` and `corrected` are treated as human-reviewed and ranked first.
- `unreviewed` stays a normal AI-estimated candidate.
- `needs_reanalysis` stays visible but adds caution text.
- `rejected` and `mark_as_joke` hidden events are excluded from normal answers.
- `mark_as_misunderstanding` moves the event into `minor_misunderstanding`.
- `mark_as_reconciled` moves the event into `reconciliation`.

Answers and monthly summaries include review-aware counts:
`人間確認済み件数`, `AI推定のみ件数`, `除外済み件数`, and `要再分析件数`.
Dry-run reports separate existing reviewed relationship DB events from newly
extracted candidates.

## MVP Features

- Rule-based Japanese intent routing.
- Mock personal and notes lifelog adapters.
- Opt-in `upstream_readonly` adapters that return source pointers and short controlled excerpts.
- Manual `relationship_profiles` management from the CLI and UI settings.
- Chat UI backend/profile/date-range selection with safe fallback when upstream
  adapters or profiles are not configured.
- Minimal Chat UI review actions for saved relationship event candidates.
- Dry-run relationship event candidate extraction with `counts-only`, `redacted`,
  and `private` report levels plus explicit `--write` candidate saves.
- Relationship DB-only backup/list/restore commands for rollback after explicit writes.
- Review-aware ranking that prioritizes human-reviewed events and excludes rejected events.
- Optional local Ollama integration for structured planning assistance with rule-based fallback.
- Deterministic relationship QA answers with cautious language.
- SQLite schema and repository helpers.
- Public-mode redaction for names, relationship labels, GPS, paths, and raw excerpts.
- Minimal ChatGPT-like Gradio UI.
