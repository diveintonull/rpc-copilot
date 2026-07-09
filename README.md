# GRC Copilot — Enterprise Compliance Knowledge Base Agent

## Quick Start
​```bash
uv sync
uv run python -m app.hello
​```

## Layout
- `app/`     Application & API
- `ingest/`  Corpus parsing, chunking, indexing
- `rag/`     Retrieval & generation
- `evals/`   Eval sets & scripts
- `skills/`  Skill packs
- `data/`    Corpus (git-ignored)