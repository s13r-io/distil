# Distil — Web UI Redesign Spec

**Status:** Draft for approval · **Author:** AI agent · **Date:** 2026-06-16
**Supersedes:** the bare server-rendered HTML in `web/app.py`
**Related:** `PRD.md` (FR1, FR11–FR21), `ARCHITECTURE.md`, `SCHEMA.md`, `web/auth.py`

This document is the agreed contract for the web UI redesign. It captures every decision
made during design review so the build matches expectations and there are no deploy
surprises. Nothing here is built until this is approved.

---

## 1. Why

The deployed UI (raw unstyled HTML f-strings) was technically within FR13's "minimal web UI"
scope but was unusable in practice: no styling, and — critically — **no way to add knowledge
through the browser**. Ingest (FR1/FR21, a *Must*) existed only in the CLI, so the hosted app
was a read-only viewer over an empty KB. This redesign makes the web UI a complete,
mobile-first surface for the whole loop: **add → distill → ask → review → score.**

## 2. Principles

- **Mobile-first, responsive.** Single fluid column, ≥16px base text, ≥44px tap targets,
  no horizontal scroll, panels stack. No CSS framework, no CDN — embedded CSS so it deploys
  to Railway unchanged.
- **Clean & minimal (light).** White surfaces, subtle borders, one blue accent, system fonts.
- **Honesty preserved.** The abstention gate (FR18), honest low-value verdict (FR12), and
  opinion-vs-fact stance (FR5) are surfaced in the UI, never hidden.
- **Conditional rendering.** Sections with no data collapse rather than showing empty headers.

## 3. Page structure (top → bottom)

1. **Header** — "Distil" wordmark + Sign out (sticky).
2. **Ask** — primary daily action; leads the page.
3. **Add knowledge** — collapsible paste/upload panel.
4. **Activity** — background distill jobs and their statuses.
5. **Library** — filterable/sortable list of filed entries.
6. **Footer** — Reindex KB, entry count.

Entries open on their own page: `GET /entries/{id}` (dedicated entry page, not a modal).

---

## 4. Section: Ask

The most prominent block on the page.

- **Input:** roomy multi-line textarea with example placeholder; full-width **Ask** button
  beneath it. Submit on tap or enter. On mobile, focusing scrolls into view and raises the
  soft keyboard (native).
- **"Sources only" toggle:** bare lookup mode (FR17) — ranked sources, no synthesis call.
- **Processing state:** "Searching your notes…" after submit.
- **Answer rendering — answer first, then sources snap in:**
  - **Streaming** the answer text token-by-token (see §9 for the backend work).
  - **Fallback:** if streaming proves unreliable in testing, render all-at-once. The UI is
    identical either way; this fallback requires *no* redesign. If we fall back, we say so
    plainly — we do not ship a half-working stream.
  - **Sources** (FR17): every retrieved note that cleared the threshold, shown as tappable
    cards (provenance quote + entry/item link + timestamp when present). Always shown on a
    successful answer — not conditional on disagreement. Resolve/append after the answer.
  - **Conflict banner** (FR20): amber, shown **only** when retrieved notes disagree.
  - **Abstention** (FR18): when nothing clears the threshold, show the honest "no relevant
    notes — won't make something up" message *instead* of an answer. No sources.
- **Each new question replaces the previous answer** (no history stack).
- **Stream failure:** discard the partial answer, show "Answer interrupted — Retry."

## 5. Section: Add knowledge

- **Collapsed by default** (header row + chevron). Tap to expand/collapse.
- **Two input paths:** (a) paste/type into a textarea; (b) upload `.srt`/`.txt`/`.md`.
  Native OS clipboard paste only — no on-page Paste button.
- **Sticky "Distil it" bar** while the panel is open, so the action stays reachable above the
  mobile keyboard. The sticky bar appears only when the panel is open; it disappears on
  collapse or submit.
- **Non-blocking submit (background queue — see §8):** submit returns immediately; the
  transcript becomes a job in **Activity** and the panel clears so you can submit the next
  one right away. You never wait on a distill.

## 6. Section: Activity

A list of background distill jobs. The UI polls for status so rows update themselves.

**States (5):** `Queued → Distilling → Done / Low-value / Failed`. There is no "cancelled"
state ("terminated" = failed).

**Per-row content & actions:**
- *Queued* — title, position in line, **Remove** (deletes the job from the queue before the
  worker picks it up; not available once distilling). This is removal, not cancellation.
- *Distilling* — staged hint ("triaging…", "extracting items…"), elapsed time.
- *Done* — "kept N items · verdict rich" + **View entry** link.
- *Low-value* — honest neutral notice: "Not much to extract — verdict little_to_extract.
  Nothing filed." (FR12). Not error-styled.
- *Failed* — error reason + **Retry** (re-queues the same transcript).

**Row lifetime:**
- Done, Low-value, and Removed rows **auto-clear after 24h** (the entry already lives in
  Library).
- **Failed rows persist indefinitely** (unfinished business; kept for Retry).

**Header clear actions (contextual — shown only when relevant):**
- **Clear finished** — appears only when Done/Low-value/Removed rows exist; clears them now.
  Never touches failures.
- **Clear failed** — appears only when Failed rows exist; clears only those.

## 7. Section: Library + Entry page

### 7.1 Library list
- **Row:** title, score badge (★ + number when scored, "unscored" pill otherwise), tag chips,
  age + item count. Whole row links to the entry page.
- **Sort:** Newest / Oldest.
- **Filters (combinable):**
  - **Tag chips** — built dynamically from the topics/knowledge_types actually present in the
    KB (not hardcoded).
  - **Scored / Unscored.**
  - **Star ratings** — 5★ / 4★ / 3★ / etc.
  - Filters stack (e.g. `retrieval` AND `unscored`); header shows "showing 3 of 8"; one-tap
    **Clear all**.
  - No title/topic free-text search. No "Needs review" shortcut.
- **Length:** most recent ~20, with **Show more**.
- **Empty state:** styled, pointing to the Add panel (replaces the bleak "No entries yet.").

### 7.2 Entry page (`/entries/{id}`)
Parsed, structured view (not raw markdown). Rendered from real `KBEntry` fields.

1. **Header** — back arrow, title (`source.title`), captured date (readable).
2. **Triage strip** — badges: `verdict`, `density`, transcript-loss level; knowledge-type mix
   (e.g. heuristic 60% · procedural 40%).
3. **Knowledge items** — one card each: statement; **type** + **stance** badges (stance shown
   per FR5 so opinion never reads as fact); **provenance** quote as a serif pull-quote with
   timestamp when present; **conditional** rationale/scope/preconditions/gotchas (only when
   populated).
4. **Application links** (`application_links`) — only when present; novelty-flagged ones
   marked.
5. **Related entries** (`related_entries`) — only when present; relation label
   (supports/contradicts/same-principle/…) linking to the target.
6. **Scoring panel** — 1–5 buttons + fixed-reason dropdown
   (`relevant | already_knew | bad_source | wrong_for_me | irrelevant_now`). Shows current
   score if already scored.

- **No raw transcript/markdown view** (distilled view only; raw transcript isn't stored
  separately anyway).
- **After scoring:** inline confirmation ("Scored 5 · relevant ✓"), stay on the page; the
  profile updates in the background (FR10).
- Sources in an Ask answer deep-link to the specific item on this page.

---

## 8. Backend: background distill queue

Replaces the synchronous ingest with a non-blocking, restart-safe job queue.

- **`jobs` table** in the existing SQLite DB: `job_id`, `kind` (paste/file), `title`,
  `payload` (text or stored file path), `status`
  (`queued|running|done|low_value|failed|removed`), `entry_id` (nullable), `verdict`/`items`
  summary, `error` (nullable), `created_at`, `updated_at`.
- **In-process worker thread, serial.** One background thread inside the FastAPI app pulls one
  `queued` job at a time and runs `run_pipeline`. No second Railway service (the deploy stays
  one service; safe because the app runs `numReplicas = 1` on a single volume).
- **Restart-safe.** On startup, any job left `running` (interrupted by a restart) is re-queued
  so nothing silently vanishes.
- **Routes:**
  - `POST /ingest` — accepts paste text or file upload; inserts a `queued` job; returns job id
    immediately (non-blocking).
  - `GET /jobs` — returns current jobs + statuses for the Activity poll.
  - `POST /jobs/{id}/remove` — removes a `queued` job.
  - `POST /jobs/{id}/retry` — re-queues a `failed` job.
  - `POST /jobs/clear?scope=finished|failed` — bulk clear.
- **Rate/cost:** serial processing naturally respects LLM rate limits and avoids cost spikes.

## 9. Backend: streaming Ask

- **`LLMClient` protocol** gains an optional `stream()` method.
  - `AnthropicClient.stream()` uses `client.messages.stream()` (same signature family as the
    existing `messages.create()`).
  - `FakeClient.stream()` yields its canned response in chunks → existing tests and the
    zero-synthesis-call abstention assertions (T-Q2) are unaffected.
- **`ask()` streaming sibling:** prompt restructured so the model emits the prose **answer
  first**, then a delimited `---SOURCES---` block carrying citations + conflict. Stream
  everything before the fence as answer text; parse the block when the stream closes. Grounded-
  citation filtering and the `contradicts`-edge conflict detection (query.py:137) run
  server-side, unchanged.
- **Web transport:** `StreamingResponse`/SSE; browser JS appends tokens then snaps in sources.
- **Safety valve:** `complete()` is untouched. If streaming is unreliable in testing, the web
  route falls back to `complete()` and the UI renders all-at-once — no redesign needed.
- **Abstention gate is unaffected** — it runs before any LLM call (query.py:112–118).

## 10. Auth (unchanged, preserved)

Keep `web/auth.py` exactly as-is: fail-closed in public mode without a secret (T-A1), auth on
data routes (T-A2), localhost convenience (T-A3), `0.0.0.0:$PORT` bind (T-A4), signed session
cookie login flow. New routes (`/ingest`, `/jobs`, `/entries/{id}`) sit behind the auth gate;
`/health` and `/login` stay open. Existing web tests (substrings "Distil", "Sign in",
"Incorrect secret") must continue to pass.

## 11. Test plan

- **Keep all existing tests green** (`tests/unit/test_web_auth.py`, query/abstention suite).
- **New unit tests:**
  - Job lifecycle: enqueue → worker runs → done/low_value/failed transitions; remove only on
    queued; retry re-queues; restart re-queues `running`.
  - `/ingest` returns immediately (non-blocking) for paste and for file upload.
  - `/jobs` reports states; clear scopes behave (finished vs failed).
  - Streaming: `FakeClient.stream()` yields chunks; `ask` streaming sibling parses the
    `---SOURCES---` block correctly; grounded/ungrounded citation split preserved; abstention
    still makes zero synthesis calls.
  - Library filter/sort logic (tag, scored/unscored, star, combine + count).
  - Entry-page rendering: conditional sections appear/hide on populated vs empty fields.
- **Verification step:** run the app, exercise each route, confirm mobile layout (narrow
  viewport), confirm streaming + fallback, before redeploy.

## 12. Out of scope (v1)

Profile editor; multi-page navigation beyond entry pages; settings beyond Sign out; raw
transcript viewer; title/topic free-text search; parallel job processing; a second worker
service.

## 13. Deploy

No change to the Railway model: one service, Dockerfile build, `uvicorn web.app:app` on
`0.0.0.0:$PORT`, volume at `/data`, single replica. The worker thread starts with the app.
Env vars unchanged (`DISTIL_PUBLIC`, `DISTIL_AUTH_SECRET`, `ANTHROPIC_API_KEY`, `DISTIL_MODEL`,
embedder + path vars).
