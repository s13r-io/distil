# ARCHITECTURE — Distil

How the system is built. Stack choices here are **defaults**; the owner may override, but
the agent should not change them unilaterally — propose changes in `TRACKER.md` first.

---

## 1. Stack (default)

| Concern        | Choice                                  | Why                                                        |
|----------------|------------------------------------------|------------------------------------------------------------|
| Language       | Python 3.11+                             | Best LLM ecosystem; easy for contributors.                 |
| Core           | `distil/` library package                | Pure logic separated from I/O for testability.             |
| LLM            | Pluggable `LLMClient`; default Claude API | Provider-swappable; key + model from env.                  |
| Embeddings     | Pluggable `Embedder`; local model or API  | Powers semantic search; local model = provider-independent retrieval. |
| Vector search  | `sqlite-vec` (vec0 virtual table)         | Stays in the one SQLite file; no extra server. Pre-1.0 — pin the version. |
| Auth (hosted)  | Single-user shared secret / session       | Required only when not on localhost (see §8).              |
| CLI            | Typer                                    | Ergonomic, testable commands.                              |
| Web/API (v0.2) | FastAPI + Jinja2/HTMX                     | One process, no build step, deployable anywhere.           |
| Profile/index  | SQLite (via `sqlite3` or SQLModel)        | Zero-config, portable, "clone and run".                    |
| KB entries     | Markdown files in `kb/`                   | Output *is* documents; human-readable; git-friendly.       |
| Tests          | pytest                                   | Standard; supports markers for the LLM eval suite.         |
| Deploy         | Docker + docker-compose                  | One-command self-host.                                     |
| CI             | GitHub Actions                           | Run unit tests on every push; eval suite optional/gated.   |
| License        | MIT                                      | Anyone can deploy/modify.                                  |

**LLM model:** read from env `DISTIL_MODEL`; do not hardcode a model string in source. The
README instructs the user to set a current model. Default provider reads `ANTHROPIC_API_KEY`.

## 2. Pipeline

A transcript flows through ordered stages. Each stage is a module with a pure interface;
LLM-backed stages take an injected `LLMClient` so tests can mock it.

```
raw input (pasted text or .srt / .txt / .md file) + profile
        │
        ▼
[0] Ingest ──────────► normalized transcript: list of segments {text, timestamp?, locator}
        │              (parses .srt and inline timestamps, tolerates none; rejects a
        │              transcript below a word-count floor (TranscriptTooShortError) — the
        │              pipeline's only quality gate, owner decision — PURE, no LLM)
        ▼
[0.5] Narrative summary (optional, cheap-tier, CONCURRENT) ► narrative_summary — whole-
        │              transcript account, no citations; starts here, on a background thread,
        │              and runs alongside stage 1 below rather than blocking it — they share no
        │              data (both read only the transcript). Joined, bounded by
        │              DISTIL_SUMMARY_JOIN_TIMEOUT_SECONDS, right before stage 7 files; a
        │              failure or a still-running summary both leave narrative_summary unset
        │              (logged either way) rather than blocking or failing filing.
        ▼
[1] Triage+Extract MERGED ──► one strong-tier call, full transcript read ONCE: states its
        │              classification (dominant knowledge type routes what it extracts,
        │              density, transcript-loss — density/loss informational, verdict stored
        │              but never gates) FIRST, then extracts raw knowledge items conditioned on
        │              that, SECOND, in the same response — see extract.py's and
        │              prompts/triage_extract.py's module docstrings
        ▼
[3] Normalize ─────────► atomic items + provenance + stance validated
        │
        ▼
[4] Link to profile ───► application_links (goal-tied, some novelty-flagged)
        │
        ▼
[5] Note synthesis ────► distilled_note (teaching note grounded in verified item ids)
        │              (narrative summary from stage 0.5 is joined here, before stage 6)
        ▼
[6] Graph-link ────────► related_entries (match against existing KB index)
        │
        ▼
[7] File ──────────────► write markdown note+evidence to kb/, index row in SQLite (+ embed items, §9;
        │                +export the neutral OKF layer to okf/, §4)
        │
        ▼
[8] Canonicalize ──────► match/new/reject each item against `concepts` (capped LLM call), then
        │                synthesize+export touched concept pages to okf/concepts/ (§4); also
        │                match/new/reject each item's entity mentions against `entities` one
        │                granularity down, synthesize+export touched entity pages to
        │                okf/entities/ (§4)
        │
        ▼
[9] Concept edges ─────► classify typed links between concepts just synthesized (capped LLM
        │                call per candidate pair), re-export changed concept pages (§4)
        │
        ▼
[10] Feedback (later) ─► score+reason → profile update (pure logic, SCHEMA §3)
```

(There is no stage "2" — it was extraction's own number before the merge; kept as a gap rather
than renumbering everything below it, since "Stage 8"/"Stage 9" for canonicalize/concept-edges
are already established, pervasive terminology across `canonicalize.py`, `concept_graph.py`,
`AGENTS.md`, and `pipeline.py`'s own inline comments.)

LLM-backed stages: **1, 4, 5, 6, 8, 9** (6 only needs the LLM for relation classification,
candidate matching is a deterministic index lookup first; 8 and 9 likewise — embedding/centroid
similarity candidates first, one capped batched LLM call for matching plus capped synthesis calls
for 8, one capped LLM call per candidate pair for 9; 8's entity-mention matching/synthesis follows
the identical shape one granularity down, so it adds no new LLM-backed stage). Pure/deterministic
stages: **0, 3, 7, 10**. Stage **0.5** is also LLM-backed but optional and on a separate,
cheap-tier client (`model_config.resolve_stage_model("summary")`, never the strong `DISTIL_MODEL`)
— it only runs when a caller opts in, and runs concurrently with stage 1 rather than adding to
wall-clock time, so it's never counted against the core call budget below.
Keep the core LLM-call count per useful transcript bounded (target ≤ 3 before graph relation
classification: one merged triage+extract call, link, note — triage and extraction used to be
two separate calls; merged once the short-circuit that justified reading the transcript twice
was removed, since a cheap classification pass buying nothing but a veto no longer had a reason
to exist). There is no quality short-circuit: once a transcript clears stage 0's word-count
floor, it always runs the full sequence and gets filed.

**Timestamps are optional.** Stage 0 captures a timestamp per segment when the source has one
(`.srt`, or inline markers like `00:12:30`), and leaves it null otherwise, always keeping a
line/segment `locator`. Downstream, provenance uses the quote as the always-present anchor and
attaches a timestamp only when one exists (SCHEMA §2).

## 3. Module layout

```
distil/
  __init__.py
  models.py          # Pydantic models: Profile, KBEntry, KnowledgeItem, ApplicationLink, Feedback
  ingest.py          # stage 0 (PURE): parse .srt/.txt/.md/pasted text → normalized transcript (timestamps optional)
  llm.py             # LLMClient protocol + AnthropicClient + FakeClient (tests)
  prompts/           # prompt templates, one per LLM stage (versioned strings)
  triage.py          # standalone classifier (dominant type, density, transcript-loss, verdict) — not called by the pipeline anymore; kept for the gated eval suite and its parser (parse_triage_response), reused by extract.py's merged call
  extract.py         # stage 1 (routes by type; also houses run_triage_extract, the merged triage+extract call the pipeline actually uses — see its module docstring)
  normalize.py       # stage 3 (pure: validation, provenance check, dedup) — no stage "2"; that was extraction's own number before the triage+extract merge
  link.py            # stage 4 (profile-aware application links)
  note.py            # stage 5 (grounded teaching-note synthesis + deterministic fallback)
  model_config.py    # per-stage model resolution: DISTIL_MODEL_<STAGE> overrides, cheap-tier default for "summary"
  summary.py         # stage 0.5 (optional, cheap-tier, runs CONCURRENTLY with stage 1): whole-transcript narrative summary — sentence-safe chunking + coverage-floor retry
  refresh_summary.py # per-entry narrative-summary regeneration (CLI `refresh-summary` / web route); never re-fetches or re-extracts
  graph.py           # stage 6 (candidate lookup + relation classify)
  canonicalize.py    # stage 8: concept-matching engine, per-item match/new/reject against `concepts` table; also entity-mention matching against `entities` one granularity down (Phase D); plus `run_canonicalize_stage` orchestration (see AGENTS.md)
  synthesize_concept.py  # concept-page + entity-page synthesis: grounded ConceptClaim/EntityClaim synthesis + code-rendered citations, called from `run_canonicalize_stage` (see AGENTS.md)
  concept_graph.py   # stage 9: concept<->concept typed-edge classification, per-concept centroid candidates + capped LLM classify, plus `run_concept_edges_stage` orchestration (see AGENTS.md)
  profile_update.py  # stage 10 (PURE: implements SCHEMA §3 table)
  embed.py           # Embedder protocol + LocalEmbedder + ApiEmbedder + FakeEmbedder (tests)
  query.py           # read layer: retrieve + retrieve_concepts + retrieve_entities → relevance gate → grounded synthesis → sources (see AGENTS.md)
  store.py           # SQLite (+ sqlite-vec vectors) + markdown filing (+ OKF export at File, §4)
  pipeline.py        # orchestrates 1→9 (now also embeds items at the File stage; canonicalize gated by PipelineConfig.enable_canonicalize, concept edges by PipelineConfig.enable_concept_edges, entities by PipelineConfig.enable_entities)
  cli.py             # Typer commands (run, score, list, show, ask, reindex)
  youtube.py         # fetch layer (Phase 1): yt-dlp playlist listing + caption fetch → Transcript (see AGENTS.md)
  okf.py             # OKF export layer: per-video sources/+raw/ pages (Phase 2) + concept pages (Phase 15.2) + entity pages (Phase D) + indexes (see AGENTS.md)
  okf_lint.py        # stdlib-only validator for the OKF bundle: `python -m distil.okf_lint <okf_root>`
web/                 # FastAPI app (v0.2): view/score/browse + ask box; auth middleware
tests/
  fixtures/          # transcripts (rich/mixed/low-value/screen-share) + a query KB fixture
  unit/              # deterministic tests (no API)
  eval/              # LLM behavior tests (marked, gated by API key)
kb/                  # generated entries (gitignored by default, or committed if user wants)
okf/                 # derived neutral OKF bundle: sources/, raw/, concepts/, entities/, index.md (regenerated from kb/)
data/                # distil.db incl. vectors (gitignored)
```

## 4. Data flow & storage

- **Profile**: single row (or JSON blob) in SQLite, schema per `SCHEMA.md` §1. Read at link stage, written at feedback stage.
- **Source metadata**: uploaded filenames are cleaned before becoming fallback display titles, and an optional YouTube URL is stored in `source.url` for navigation back to the original video. When a YouTube URL is present, Distil fetches public oEmbed metadata without an API key and stores the video title, channel, channel URL, thumbnail URL, provider, and fetch timestamp.
- **YouTube transcript fetch (Phase 1)**: given a video or playlist URL (web UI ADD input), `youtube.py` shells out to `yt-dlp` to fetch English captions (or enumerate a playlist into one ingest job per video via the existing `web/jobs.py` Worker) and converts them to `.srt`, parsed by `ingest.ingest_srt_text` into the same `Transcript` shape as an uploaded file. A video with no captions or that fails to fetch is skipped and reported, not fatal to the rest of a playlist. See `AGENTS.md` for the fetch-layer invariants and `docs/TESTING.md` (T-Y*) for the test catalog.
- **KBEntry**: the markdown file in `kb/<entry_id>.md` is the source of truth for human reading. New entries include a `distilled_note` (core takeaway, key points, applications, caveats, review questions) plus the atomic evidence items in a collapsed source-evidence block. A row in SQLite (`entries` table: id, title, topics, knowledge_types, score, created_at, file_path) is the index used for graph candidate lookup and browsing.
- **OKF export layer (Phase 2)**: at the File stage, `store.file_entry(..., transcript=...)` derives a second, neutral bundle under `okf_root` (default a sibling of `kb_dir`, e.g. `data/../okf`) via `okf.py` — `sources/<slug>.md` (summary + key moments + a link to the raw page) and `raw/<slug>.md` (immutable timestamped transcript), plus regenerated `index.md`/`sources/index.md`. It carries no feedback or application-link data, and is skipped when `transcript` is omitted (e.g. feedback-only re-files). `okf_lint.py` (`python -m distil.okf_lint <okf_root>`) validates the bundle. See `AGENTS.md` for the slug-stability rule and phase boundaries, and `docs/TESTING.md` (T-OKF*, T-OKFL*) for the test catalog.
- **Concept canonicalization + synthesis (Stage 8, OKF Phase 3)**: `canonicalize.run_canonicalize_stage(entry, store, client)` is `pipeline.run_pipeline`'s Stage 8, gated by `PipelineConfig.enable_canonicalize` (default `True`). It calls `canonicalize_entry` (decides match/new/reject per knowledge item against existing `concepts` rows — embedding-similarity candidates + a capped LLM call — returning the touched `Concept`s), then `synthesize_touched_concepts` (ranks touched concepts by embedding similarity and, up to a per-video cap, calls `synthesize_concept.py` to build grounded `ConceptClaim`s), then keeps the OKF bundle in sync via `okf.export_concept`/`remove_concept` (including concepts that lost this entry as a member, not just touched ones) and `okf.render_source_with_concepts` (the source page's "## Concepts covered" backlink). `okf_lint.py`'s E5-E8 checks validate the resulting `concepts/` bundle. The same Stage 8 call also runs `canonicalize_entry_entities`/`synthesize_touched_entities` (gated by `PipelineConfig.enable_entities`, default `True`) — the identical match/new/reject/synthesis-capping shape one granularity down (entity mentions instead of items, with a hard `kind` pre-filter), keeping `okf/entities/` in sync the same way (`okf_lint.py`'s E9-E12). Entities ride the extraction response already read for items, so this adds no new transcript read or pipeline stage. See `AGENTS.md` for the full data-flow detail and phase boundaries.
- **Concept↔concept typed edges (Stage 9, OKF Phase 3b)**: `concept_graph.run_concept_edges_stage(touched, store, client)` is `pipeline.run_pipeline`'s Stage 9, gated by `PipelineConfig.enable_concept_edges` (default `True`). For each concept Stage 8 actually synthesized this run, `link_concept_graph` finds candidates via centroid-to-centroid cosine similarity (`Store.find_concept_edge_candidates`) and classifies each into `contrasts_with`/`builds_on`/`related` (or drops it) with one capped LLM call, replacing `Concept.edges` wholesale. `Store.prune_dangling_concept_edges` then drops edges left dangling by any concept deleted this run, and every concept whose edges changed gets its OKF page re-exported (`## Contrasts with`/`## Builds on`/`## Related`, plus a deterministic `> **Contradiction:**` flag from `synthesize_concept.find_claim_contradictions`). `okf_lint.py`'s E8 check counts these typed-edge links as valid inbound links. See `AGENTS.md` for the full data-flow detail.
- **Narrative summary (Stage 5.5, optional)**: `summary.py` reads the transcript directly (sentence-safe chunking, per-chunk + merge synthesis, coverage-floor retry on thin output or a dropped connection) on a separate client resolved via `model_config.resolve_stage_model("summary")` — cheap-tier by default, never the strong `DISTIL_MODEL`. `pipeline.run_pipeline` only runs it when a caller passes `summary_client`; a failure is caught and logged, never blocking filing. `refresh_summary.py` regenerates just `narrative_summary` later, from the entry's already-stored OKF raw page, without re-fetching or re-extracting (CLI `refresh-summary` / `POST /entries/{id}/refresh-summary`). See `AGENTS.md` for full detail.
- **Provenance** is stored inside each item; the transcript itself is not retained after processing unless the user opts in (privacy).

## 5. LLM boundary (critical for testing)

The model's *judgment* is non-deterministic and cannot be unit-tested for exact output. So:

- **Deterministic glue** (prompt assembly, response parsing, schema validation, routing, profile math, filing) is unit-tested hard with a `FakeClient` returning canned responses.
- **Model behavior** (does triage classify correctly? are items faithful?) is checked by the `eval/` suite against fixtures, asserting *properties* (e.g. "every returned item's provenance quote appears in the transcript", "low-value fixture yields little_to_extract") rather than exact strings.

This split is non-negotiable and is detailed in `TESTING.md`.

## 6. Configuration

All config via env (`.env` locally, service variables when hosted): `ANTHROPIC_API_KEY`,
`DISTIL_MODEL`, `DISTIL_DB_PATH`, `DISTIL_KB_DIR`, `DISTIL_NOVELTY_RATIO` (default 0.2),
`DISTIL_PROFILE_ALPHA` (default 0.3), `DISTIL_EMBEDDER` (`local` | `api`), `DISTIL_EMBED_MODEL`,
`DISTIL_RETRIEVAL_THRESHOLD` (min similarity to clear the abstention gate), `DISTIL_TOP_K`
(default 6), `DISTIL_AUTH_SECRET` (required when not on localhost), `DISTIL_PUBLIC` (set true
when hosting — refuses to serve without `DISTIL_AUTH_SECRET`), `DISTIL_MODEL_<STAGE>` (per-stage
model override, e.g. `DISTIL_MODEL_SUMMARY`), `DISTIL_SUMMARY_CHUNK_CHARS`/
`DISTIL_SUMMARY_MAX_RETRIES` (narrative summary layer). No secrets in source. `.env.example`
documents every variable.

## 7. Deployment (local)

`docker compose up` builds the image, mounts `kb/` and `data/` as volumes (so data persists
and is git-backupable), runs the CLI/web. README covers local (pip/venv) and Docker paths.
GitHub Actions runs unit tests on push; a release workflow tags versions.

## 8. Hosted deployment (Railway)

The same image runs on Railway; see `DEPLOY_RAILWAY.md` for the click-by-click walkthrough.
Three things change versus localhost, and they are not optional:

1. **Auth is mandatory.** Generating a public Railway domain puts the app on the open
   internet with your API key wired in. Anyone with the URL could spend your LLM budget and
   read/write your knowledge base. The app must enforce `DISTIL_AUTH_SECRET` and refuse to
   serve when `DISTIL_PUBLIC=true` but no secret is set. (PRD FR14.)

2. **Storage must move off the ephemeral container disk.** Railway containers are wiped on
   each redeploy. Attach a **Railway Volume** mounted at `/data` and point both
   `DISTIL_DB_PATH=/data/distil.db` and `DISTIL_KB_DIR=/data/kb` at it. Caveats: volumes are
   mounted at **runtime, not build** (never write KB/DB during the build step); there is one
   volume per service; sizes are plan-based (0.5 GB free/trial, 5 GB Hobby, 50 GB Pro — ample
   for markdown + SQLite). Managed Postgres is an alternative for the index, but volume +
   SQLite is the simplest path for a single-user app. (If `store.py` uses SQLAlchemy/SQLModel,
   a later swap to Postgres is cheap.)

3. **Bind to the injected port.** There is no port-mapping layer; the process must listen on
   `0.0.0.0` and the port Railway provides: `uvicorn web.app:app --host 0.0.0.0 --port $PORT`.

Build is from the `Dockerfile` (or Railway's Railpack); env vars are set as Railway service
variables; a public URL comes from **Generate Domain** (do this *after* auth is in place).
**Backup:** prefer the provider-independent route — a scheduled job that commits `kb/` to a
private git remote — so the knowledge base is never trapped on one cloud volume. Railway's
own volume backups are a fallback.

4. **Local embeddings need memory.** The chosen default is local embeddings (`DISTIL_EMBEDDER=local`),
   which loads a small model into the service's RAM. Size the Railway instance accordingly, and
   ship the model in the image (downloaded at *build* time — never to the runtime volume). On a
   very small instance, set `DISTIL_EMBEDDER=api` instead; the `Embedder` abstraction makes this a
   config change only.

## 9. Querying the knowledge base (read layer)

Turns the write-only KB into something consultable. Reuses the existing spine: atomic items
are the retrieval unit, and the grounding rule is the read-side twin of extraction faithfulness.

**Indexing.** At the **File** stage, each knowledge item is embedded via the `Embedder` and
stored in a `sqlite-vec` `vec0` virtual table inside the same `distil.db`, alongside a
foreign key to its item/entry. For new Note v1 entries, the vector text includes both the
atomic item and any distilled-note context that cites that item. `reindex` backfills embeddings
for entries filed before the read layer existed.

**Query flow (`query.py`, exposed as `ask`):**

```
question
   │
   ▼
embed query → KNN search (sqlite-vec), rank by similarity × feedback_score × recency
   │
   ▼
relevance gate: any item ≥ DISTIL_RETRIEVAL_THRESHOLD?
   │                                   │
   NO → return "no relevant notes"     YES
   │     (NO synthesis LLM call)        │
   ▼                                    ▼
                          grounded synthesis: answer using ONLY retrieved items;
                          every claim must trace to an item; abstain on the rest
                                        │
                                        ▼
                          answer + source links (entry + item + provenance timestamp)
                          + conflict note if retrieved items disagree (uses `contradicts` edges)
```

A bare lookup ("do I have notes on X?") returns the ranked source list without synthesis; a
question runs synthesis on top. **The gate is what enforces no-hallucination:** generation is
never invoked unless retrieval clears the threshold, so the system cannot answer from the
model's outside knowledge — it either grounds in your notes or says it has none.

**Concepts join the same gate (Phase 18 / OKF Phase 3d).** `retrieve_concepts` ranks synthesized
OKF concept pages (the `concepts` table, Phase 15) against the question the same way `retrieve`
ranks raw items, blending in each concept's closest member so a centroid diluted by loosely-related
videos can't hide it. A concept clears the relevance gate at the exact same
`DISTIL_RETRIEVAL_THRESHOLD` as a raw item — never a lower bar; abstention only fires when neither
clears it. Once cleared, its member items are recruited into the evidence pool (citations still
resolve from the live item, never the concept's own copied fields) and its already-grounded
`ConceptClaim` prose is added as extra synthesis context. Concepts only widen what's considered;
they never replace item-level retrieval or relax the gate. `retrieve_entities` (Phase D) joins the
same gate the same way, one granularity down. See `AGENTS.md` for the full wiring.

LLM-call budget for `ask`: one embedding call (or zero, if local) + at most one synthesis
call. Local embeddings make retrieval fully provider-independent.
