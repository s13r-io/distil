# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.
- Tests split into `tests/unit` (deterministic, no network, `pytest tests/unit`) and `tests/eval` (real LLM behavior, gated by `ANTHROPIC_API_KEY`, `pytest -m eval`). See `docs/TESTING.md` for the full test-case catalog and the TDD workflow (write test first).
- `pytest -m eval` needs the `web` extra installed to collect cleanly — install with `.[anthropic,embed-local,vec,web,dev]` (add `youtube` too if touching YouTube ingest).
- YouTube ingest (`distil/youtube.py`) shells out to `yt-dlp` via an injectable `run` callable so unit tests fake the subprocess boundary — never invoke the real binary in `tests/unit`. When fetching captions, use `--sub-langs en` (exact match), not a wildcard like `en.*`: the wildcard also matches yt-dlp's auto-*translated* variants (`en-de`, `en-fr`, ...), which are lower quality and can outrank the real English track once sorted.
- Fetched YouTube captions are converted to `.srt` and parsed with `distil.ingest.ingest_srt_text`, so downstream stages never know the source was YouTube rather than an uploaded file.
- Alongside `kb/<id>.md` (lossless JSON, internal source of truth), `distil/okf.py` derives a neutral per-video [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog) bundle (`okf/sources/<slug>.md` + `okf/raw/<slug>.md` + indexes) with no feedback/application-link data. `distil/okf_lint.py` (stdlib only, `python -m distil.okf_lint <okf_root>`) validates it. `Store.file_entry(..., transcript=...)` triggers the export; omit `transcript` for feedback-only re-files to leave OKF pages untouched. Slug = stable slugified `source.title` (falls back to `entry_id`) via `okf.slugify`. `entities/` layers and OpenKnowledge wiring are later phases — not implemented yet.
- Phase 15.1 (`distil/canonicalize.py`) added the concept **matching engine**: per filed video, `canonicalize_entry(entry, store, client)` decides match/new/reject for each knowledge item against a `concepts` SQLite table (mirrors the `graph.py` deterministic-candidates-then-capped-LLM-call shape, but at item granularity with embedding similarity — `Store.find_concept_candidates` — as the primary candidacy signal). It is idempotent (`Store.retract_entry_concept_memberships` runs before every re-canonicalize and on `delete_entry`) but is **not wired into `pipeline.py` yet** — call it directly; no concept pages/OKF export exist until Phase 15.2, no `enable_canonicalize` flag or lint checks until 15.3. See `distil-phase3-design/report.md` (kept outside this repo) for the full design rationale.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
