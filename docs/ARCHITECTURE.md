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
        │              (parses .srt and inline timestamps; tolerates none — PURE, no LLM)
        ▼
[1] Triage ──────────► triage verdict (types, density, loss, verdict)
        │                     │
        │            if verdict == little_to_extract → return low-value result, do not file
        ▼
[2] Extract (routed by type) ──► raw knowledge items
        │
        ▼
[3] Normalize ─────────► atomic items + provenance + stance validated
        │
        ▼
[4] Link to profile ───► application_links (goal-tied, some novelty-flagged)
        │
        ▼
[5] Note synthesis ────► distilled_note (teaching note grounded in verified item ids)
        │
        ▼
[6] Graph-link ────────► related_entries (match against existing KB index)
        │
        ▼
[7] File ──────────────► write markdown note+evidence to kb/, index row in SQLite (+ embed items, §9)
        │
        ▼
[8] Feedback (later) ──► score+reason → profile update (pure logic, SCHEMA §3)
```

LLM-backed stages: **1, 2, 4, 5, 6** (6 only needs the LLM for relation classification; candidate
matching is a deterministic index lookup first). Pure/deterministic stages: **0, 3, 7, 8**.
Keep the core LLM-call count per useful transcript bounded (target ≤ 4 before graph relation
classification: triage, extract, link, note). Low-value transcripts still stop after triage.

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
  triage.py          # stage 1
  extract.py         # stage 2 (routes by type)
  normalize.py       # stage 3 (pure: validation, provenance check, dedup)
  link.py            # stage 4 (profile-aware application links)
  note.py            # stage 5 (grounded teaching-note synthesis + deterministic fallback)
  graph.py           # stage 6 (candidate lookup + relation classify)
  profile_update.py  # stage 8 (PURE: implements SCHEMA §3 table)
  embed.py           # Embedder protocol + LocalEmbedder + ApiEmbedder + FakeEmbedder (tests)
  query.py           # read layer: retrieve → relevance gate → grounded synthesis → sources
  store.py           # SQLite (+ sqlite-vec vectors) + markdown filing
  pipeline.py        # orchestrates 1→7 (now also embeds items at the File stage)
  cli.py             # Typer commands (run, score, list, show, ask, reindex)
web/                 # FastAPI app (v0.2): view/score/browse + ask box; auth middleware
tests/
  fixtures/          # transcripts (rich/mixed/low-value/screen-share) + a query KB fixture
  unit/              # deterministic tests (no API)
  eval/              # LLM behavior tests (marked, gated by API key)
kb/                  # generated entries (gitignored by default, or committed if user wants)
data/                # distil.db incl. vectors (gitignored)
```

## 4. Data flow & storage

- **Profile**: single row (or JSON blob) in SQLite, schema per `SCHEMA.md` §1. Read at link stage, written at feedback stage.
- **Source metadata**: uploaded filenames are cleaned before becoming fallback display titles, and an optional YouTube URL is stored in `source.url` for navigation back to the original video. When a YouTube URL is present, Distil fetches public oEmbed metadata without an API key and stores the video title, channel, channel URL, thumbnail URL, provider, and fetch timestamp. It still does not fetch transcripts or scrape video content in v0.
- **KBEntry**: the markdown file in `kb/<entry_id>.md` is the source of truth for human reading. New entries include a `distilled_note` (core takeaway, key points, applications, caveats, review questions) plus the atomic evidence items in a collapsed source-evidence block. A row in SQLite (`entries` table: id, title, topics, knowledge_types, score, created_at, file_path) is the index used for graph candidate lookup and browsing.
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
when hosting — refuses to serve without `DISTIL_AUTH_SECRET`). No secrets in source.
`.env.example` documents every variable.

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

LLM-call budget for `ask`: one embedding call (or zero, if local) + at most one synthesis
call. Local embeddings make retrieval fully provider-independent.
