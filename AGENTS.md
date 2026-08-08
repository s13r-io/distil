# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.
- Tests split into `tests/unit` (deterministic, no network, `pytest tests/unit`) and `tests/eval` (real LLM behavior, gated by `ANTHROPIC_API_KEY`, `pytest -m eval`). See `docs/TESTING.md` for the full test-case catalog and the TDD workflow (write test first).
- `pytest -m eval` needs the `web` extra installed to collect cleanly — install with `.[anthropic,embed-local,vec,web,dev]` (add `youtube` too if touching YouTube ingest).
- YouTube ingest (`distil/youtube.py`) shells out to `yt-dlp` via an injectable `run` callable so unit tests fake the subprocess boundary — never invoke the real binary in `tests/unit`. When fetching captions, use `--sub-langs en` (exact match), not a wildcard like `en.*`: the wildcard also matches yt-dlp's auto-*translated* variants (`en-de`, `en-fr`, ...), which are lower quality and can outrank the real English track once sorted.
- Fetched YouTube captions are converted to `.srt` and parsed with `distil.ingest.ingest_srt_text`, so downstream stages never know the source was YouTube rather than an uploaded file.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
