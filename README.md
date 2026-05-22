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

## Test

```bash
pytest
```

## MVP Features

- Rule-based Japanese intent routing.
- Mock personal and notes lifelog adapters.
- Opt-in `upstream_readonly` adapters that return source pointers and short controlled excerpts.
- Deterministic relationship QA answers with cautious language.
- SQLite schema and repository helpers.
- Public-mode redaction for names, relationship labels, GPS, paths, and raw excerpts.
- Minimal ChatGPT-like Gradio UI.
