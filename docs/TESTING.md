# TESTING — Distil (Test-Driven Development)

The project follows TDD: **write the test first, watch it fail (red), implement the minimum
to pass (green), refactor.** No production function is written before its test exists. The
tracker has a `tests_written?` column that must be checked *before* `code_done?`.

---

## 1. The LLM testing problem (read first)

You cannot unit-test a model's judgment for exact output — it is non-deterministic. So tests
split into two kinds, and confusing them is the most common mistake:

- **Unit tests (`tests/unit/`)** — deterministic glue only, run on every push, no network.
  Use `FakeClient` (in `distil/llm.py`) that returns canned responses. These test prompt
  assembly, response parsing, schema validation, routing, the profile-update math, graph
  candidate lookup, and filing. Fast, hermetic, the bulk of the suite.

- **Eval tests (`tests/eval/`)** — model behavior against real fixtures, marked
  `@pytest.mark.eval`, gated by `ANTHROPIC_API_KEY`, **skipped in normal CI**. They assert
  *properties*, not exact strings: e.g. faithfulness, correct verdict on low-value input.

Run unit only: `pytest tests/unit` · Run everything: `pytest -m "unit or eval"` (needs key).

## 2. Fixtures (`tests/fixtures/`)

Create at minimum these labelled transcripts (short, hand-written or trimmed real ones):

- `rich_heuristic.txt` — a coding-guidelines talk; dense, verbal, mostly heuristic.
- `procedural_tutorial.txt` — clear step-by-step, sequence matters.
- `screen_share.txt` — full of deictic refs ("as you can see here", "this line") → high loss.
- `low_value_vlog.txt` — entertainment, near-zero extractable knowledge.
- `mixed_talk.txt` — conceptual + opinion + experiential mixed.

Plus input-format fixtures for ingestion: `sample.srt` (real SRT timestamps), `inline_ts.txt`
(plain text with inline `00:12:30` markers), `no_timestamps.md` (prose, no timestamps at all), and
`youtube_rolling_caption_mmss.txt` (a verbatim excerpt of a real YouTube transcript-panel paste,
inline `MM:SS`-per-line with text-less rolling-preview lines — not hand-typed, per the project's
standing rule against fixtures sharing a wrong assumption with the code under test).

Each fixture has a sibling `*.expected.json` describing properties to assert (not exact output):
e.g. `{ "verdict": "little_to_extract" }` for the vlog, `{ "transcript_loss": "high" }` for screen-share.

For the read layer, also build a **query KB fixture** (`tests/fixtures/query_kb/`): a handful of
pre-filed entries with known content, plus a `questions.json` listing (a) questions answerable
from the KB with their expected source item IDs, and (b) questions with **no** supporting notes
that must abstain.

## 3. Test-case catalog (write these as the build proceeds)

### ingest.py (PURE — stage 0)
- T-I1: parse `sample.srt` → ordered segments, each with text and a parsed `timestamp`.
- T-I2: parse `inline_ts.txt` → segments with timestamps captured from inline `HH:MM:SS` markers.
- T-I3: parse `no_timestamps.md` → segments with `timestamp = null` and a populated line/segment `locator`.
- T-I4: pasted plain text (no file extension) is normalized the same as `.txt`.
- T-I5: unknown/binary file or empty input → clear error, not a crash.
- T-I6: the normalized transcript shape is identical across `.srt`/`.txt`/`.md`/paste (downstream stages don't care about source format).
- T-I7 (owner decision, supersedes triage's old `little_to_extract` short-circuit): a transcript below `DISTIL_MIN_TRANSCRIPT_WORDS` (default 50) raises `TranscriptTooShortError` on every ingest path (`ingest_text`, `ingest_file`'s `.srt` and text branches, `ingest_srt_text`) — a distinct `IngestError` subclass, never conflated with a read/parse/fetch failure; a transcript at or above the floor is never rejected on quality grounds, and the threshold is configurable.
- T-I8: `is_thin_source(word_count)` — a visibility signal, never a rejection — is `True` only strictly between 0 and `DISTIL_THIN_TRANSCRIPT_WORDS` (default 500); `0` (unknown, pre-existing entries) and values at/above the threshold are `False`.
- T-I9 (inline `MM:SS` rolling-caption dumps — the YouTube-transcript-panel paste shape): a majority-of-non-blank-lines `MM:SS` input routes to `_parse_mmss_rolling_caption`, producing clean prose segments (no leaked timestamp tokens) with `HH:MM:SS`-normalized timestamps and a word count that reflects real speech, not injected timestamp digits.
- T-I10: rolling-caption preview lines (text-less lines previewing the next timestamp just before its real line arrives) are discarded — never kept as empty segments and never merged/duplicated into a neighboring segment's text.
- T-I11: ordinary prose without timestamps, and prose where only a single line happens to open with something clock-shaped (a spoken time, not a caption mark), stay on the plain-prose path — the majority-of-lines bar is what prevents misdetection.
- T-I12: once the majority bar is cleared, the parser itself demands an absolute bar: a line that doesn't match the `MM:SS`-open-of-line shape at all, or matched timestamps that are not non-decreasing start-to-end, raises `IngestError` rather than guessing which lines are captions.
- T-I13: the SRT ingest path (`ingest_srt_text` → `_parse_srt`, and by extension the live YouTube/collector fetch paths built on it) is confirmed unaffected by the `MM:SS` detection added for T-I9 — its output is unchanged.

### youtube.py (fetch layer — `yt-dlp` invoked via an injectable ``run``, no real subprocess/network in unit tests)
- T-Y1: a playlist URL (`?list=` with no `v`, or `/playlist` path) is distinguished from a single video URL.
- T-Y2: a playlist enumerates to a list of normalized `watch?v=` URLs; empty/malformed listings raise `YoutubeFetchError`.
- T-Y3: a captioned video's fetched `.srt` parses into a `Transcript` via `ingest.ingest_srt_text` (same shape as uploaded `.srt`).
- T-Y4: a video with no caption file written (no captions available) raises `YoutubeFetchError`, not a crash.
- T-Y5: a `yt-dlp` process failure (private/deleted video) raises `YoutubeFetchError` with the underlying stderr.
- T-Y6 (web/jobs): one uncaptioned/failed video in a playlist batch is marked `failed` on its own job; the next queued job still processes — never fatal to the batch (`Worker._process` already isolates per-job exceptions; `web/app.py` enqueues one `kind="youtube"` job per video).
- T-Y7: a caller-supplied `workdir` reused across fetches (e.g. a shared `tmp_path`) never picks up a stale `.srt` left by a previous fetch — each fetch is scoped to its own unique child directory, and the stale file is left untouched.
- T-Y8 (Phase 19; client chain updated Phase 21, then Phase 23): both `list_playlist_video_urls` and `_fetch_into` pass `--extractor-args youtube:player_client=android_vr,web_safari,mweb` to `yt-dlp` (see `distil/youtube.py`'s module docstring for why this chain, re-derived from yt-dlp's own defaults, and why `mweb` was added in Phase 23).
- T-Y9 (Phase 19): a transient failure (429/5xx) retries with exponential backoff (injectable `sleep`) and succeeds once `yt-dlp` returns success within the bounded attempt count; a persistent transient failure still raises `YoutubeFetchError` after exhausting attempts; a non-transient failure (e.g. private/deleted video) raises immediately with no retry/sleep.
- T-Y10 (Phase 22; supersedes the removed Phase 20 `DISTIL_YOUTUBE_API_KEY` variant; extended Phase 23): with `DISTIL_POT_PROVIDER_URL` set (via `monkeypatch`), both `list_playlist_video_urls` and `_fetch_into` pass a *second* `--extractor-args` pair, `youtubepot-bgutilhttp:base_url=<value>`, alongside a `youtube:` pair that now also folds in `fetch_pot=always` (`youtube:player_client=android_vr,web_safari,mweb;fetch_pot=always` — one combined value, since yt-dlp replaces rather than merges repeated `--extractor-args` for one extractor); with the env var unset, the command line is byte-identical to T-Y8 (single `--extractor-args` pair, no `fetch_pot`, no `DISTIL_YOUTUBE_API_KEY` handling survives anywhere in the module).
- T-Y11 (Phase 21): a `yt-dlp` failure whose stderr is warning-heavy (SABR/staleness noise exceeding any head-truncation budget) still surfaces its `ERROR:`-prefixed line(s) in the raised `YoutubeFetchError`, not the leading warnings; the complete, untruncated stderr is always logged via `logging.getLogger("distil.youtube")` regardless of what the bounded exception message contains; stderr with no `ERROR:` line at all falls back to a genuine tail (the *last* N chars).
- T-Y12 (Phase 21): `_fetch_into` requests `--sub-format srt/best` and never passes `--convert-subs` (the Dockerfile image has no ffmpeg, so any format needing conversion would fail in production).
- T-Y13 (Phase A, visible progress): a successful fetch reports `("transcript_fetch", "start")`, `("transcript_fetch", "finish")`, `("caption_parse", "start")`, `("caption_parse", "finish")`, in that order, via the optional `on_phase` callback.
- T-Y14 (Phase A): a `yt-dlp` failure reports only `("transcript_fetch", "start")` — it never advances to `caption_parse`, so a stalled/failed fetch reads as stuck on the right phase rather than silently progressing.
- T-Y15 (Phase 23): `diagnose_pot` runs a single (never-retried) verbose `--simulate` `yt-dlp` invocation using the same `_extractor_args()` a real fetch would, plus `pot_trace=true`, and parses out yt-dlp's `PO Token Providers:` discovery line and every `(context, client)` pair a fetch was actually attempted for (`PotDiagnostic.provider_discovery`/`.context_attempts`); zero attempts for any context is the "never asked" finding, distinct from an attempt followed by a provider rejection/error visible in `.raw_output`.
- T-Y16 (Phase 23): `_redact_pot_diagnostic` strips real PO token values (`Generated POT: <token>`, `po_token='<token>'`) and the configured `DISTIL_POT_PROVIDER_URL` value out of a verbose transcript before it's ever returned — covered both directly and via `diagnose_pot`'s own output, since `pot_trace=true` (needed for the discovery line) is also what makes yt-dlp's bgutil plugin log a token in cleartext.
- T-Y17 (external-collector queue; **detection rewritten Phase 24, see T-Y21**): `is_bot_check_refusal` recognizes YouTube's bot-identity challenge (works against a `YoutubeFetchError` or a plain string) — true for that challenge text, false for a no-captions error and false for a playlist-listing failure.
- T-Y18 (Helper 2): `fetch_raw_captions` shares `_fetch_captions_raw` with `_fetch_into` and returns the unparsed `.srt` text unchanged (no `ingest_srt_text` parse) on success, and still raises `YoutubeFetchError` on a `yt-dlp` failure exactly like `fetch_video_transcript`.
- T-Y19 (Helper 2): `fetch_raw_captions` passes `--cookies-from-browser <spec>` plus a `--cookies <jar>` output path to `yt-dlp` only when `cookies_from_browser` is given; both flags are absent when it's unset.
- T-Y20 (Helper 2): `_detect_browser_session` reads the Netscape cookie jar `yt-dlp` exports alongside the fetch and reports `"signed_in"` when a Google auth cookie (`LOGIN_INFO`/`SAPISID`-family) is present, `"anonymous"` when only consent/session cookies are, and `"unknown"` when the jar was never written or is missing — and deletes the jar immediately after the one read regardless of outcome. Reported via the optional `on_session` callback even when the fetch itself ultimately fails, and never invoked at all when `cookies_from_browser` wasn't set.
- T-Y21 (Phase 24, bot-check apostrophe incident): `is_bot_check_refusal` matches YouTube's real captured refusal text, curly `’` apostrophe included byte-for-byte (`_REAL_BOT_CHECK_STDERR` in `tests/unit/test_youtube.py`, sourced from production failures on videos `AbpyqAfxZ8c`/`QER-0DaC-Gk`) — this exact fixture fails against the pre-fix straight-quote-literal implementation, proving the regression. Also covered: the straight-apostrophe wording still matches (in case YouTube reverts), a parametrized set of other apostrophe-like characters (curly left/right quote, modifier-letter apostrophe, grave accent, no apostrophe at all) all match, matching is case-insensitive, and the false cases from T-Y17 (no-captions, playlist-listing failure) still don't match. See `distil/youtube.py`'s `_BOT_CHECK_RE` for why this is a tolerant pattern match on the stable low-punctuation core of the sentence rather than one exact literal.
- T-Y22 (Phase 24): `diagnose_pot`'s `PotDiagnostic.bot_check_detected` reports whether `is_bot_check_refusal` recognizes this diagnostic run's own output — `True` on a run whose stderr is a genuine bot-check refusal, `False` on a clean run — so the collector-queue safety net's liveness can be checked directly (`distil youtube-diagnose-pot <url>` / `GET /diagnostics/youtube-pot`) against a video known to be bot-checked, instead of inferring it from "nothing has been parked lately" (which is indistinguishable from the net being silently dead — the exact failure mode that shipped in Phase 22/23 undetected).

### web/jobs.py — playlist up-front fetch + staging (Phase E; `youtube.py` itself is unchanged)
- T-J1: `Fetcher.process_once()` claims and fetches every currently `pending_fetch` job, one at a time, flipping each to `queued` with `kind=KIND_YOUTUBE_STAGED` — a playlist's transcripts are all fetched before the (unmodified, still single-worker) distill `Worker` needs to touch any of them.
- T-J2: with two pending videos, fetching only the first leaves it `queued` (distillable) while the second is still untouched (`pending_fetch`) — proves the overlap: the distill `Worker` can process the first while the `Fetcher` keeps working through the rest, since the two claim disjoint job statuses.
- T-J3: `Fetcher`'s inter-fetch pause (`delay_seconds`, default 3.0s) is applied via an injectable `sleep` before every fetch after the first, never before the first; configurable per-instance (wired from `DISTIL_PLAYLIST_FETCH_DELAY_SECONDS` in `web/app.py`).
- T-J4: one video whose `fetch_fn` returns `{"status": "failed", ...}` (or raises) is marked `failed` with a readable error and never prevents the rest of the batch from fetching — the same isolation guarantee `_enqueue_youtube_source`'s docstring already states for distilling.
- T-J5: `recover_interrupted` requeues both a crashed `running` distill job (`-> queued`) and a crashed `fetching` job (`-> pending_fetch`) in one call; a `youtube_staged` job recovered this way still finds its staged transcript file on disk (nothing unlinks it except a successful finish) and `web.app._load_job_transcript` reads it successfully.
- T-J6: `web.app._upload_dir()`/`_transcript_stage_dir()` are derived from `DISTIL_DB_PATH`'s own directory (or `DISTIL_STAGING_DIR` if set) — never `tempfile.gettempdir()`. Regression coverage: this is the fix for a live bug where a queued-but-not-yet-processed upload's file lived on the container's ephemeral disk and was silently lost on restart/redeploy even though its job row (in sqlite, on the volume) survived.
- T-J7: a job's staged file (uploaded file or prefetched transcript) is reclaimed when its job is removed via `POST /jobs/{id}/remove` or swept by `POST /jobs/clear` — never left to accumulate unboundedly on the volume.
- T-J8: reading a staged file (`_load_job_transcript`) no longer deletes it, and raises an honest `FileNotFoundError` if it's already gone; `Worker`'s `on_finished(job)` hook fires only for a successful (done/low_value) terminal status, never on failure, so a downstream pipeline failure after the read leaves the file in place for `JobStore.retry()` to re-read on the next attempt — reproduces and fixes a bug where a failure right after the read deleted the file, permanently breaking every subsequent retry.
- T-J9: `JobStore.autoclear`'s `on_stale_failed` hook reaps a `failed` job's staged file once it has sat untouched past the same 24h bound used for finished rows, but leaves the row itself and never fires for a recently-failed job — a permanently-failed job's kept-for-retry file still gets reclaimed eventually instead of accumulating forever.

### web/jobs.py + web/app.py — external-collector queue (bot-check refusals only; `distil/youtube.py` unchanged)
- T-J10: `_distill_job` and `_fetch_playlist_video` both route a `YoutubeFetchError` through `youtube.is_bot_check_refusal` — a bot-check refusal yields `{"status": "awaiting_collection", ...}` (`JobStore.mark_awaiting_collection`, setting `collection_deadline` once), while a non-bot-check failure (e.g. no captions) still fails immediately, unchanged from before this phase.
- T-J11: `JobStore.claim_for_collection` leases up to `limit` `awaiting_collection` jobs to `collecting` with a `lease_expires_at`, ignores jobs in every other state, respects `limit`, and — proved both with two sequential calls and with real concurrent threads racing for the same row — never hands the same video to two claimants (each row's guarded `UPDATE ... WHERE job_id=? AND status=?` only ever matches once).
- T-J12: `JobStore.release_expired_leases` returns a `collecting` job whose lease has passed back to `awaiting_collection` without touching its `collection_deadline`; an unexpired lease is left alone.
- T-J13: `JobStore.submit_collected_transcript` validates the caller-provided SRT via `ingest_srt_text` before writing anything, moves the job to `queued`/`KIND_YOUTUBE_STAGED` on success, rejects a job that was never leased (`not_leased`) or doesn't exist (`not_found`), treats an already-`queued` job as `already_submitted`, and — the idempotency guarantee — a retry after the first successful submit (a lost HTTP response) reports success again rather than erroring or double-staging.
- T-J14: `JobStore.report_uncollectable` fails a leased job outright with the collector's reason, rejects a job that was never leased, and repeating the call after the job already failed still returns success (idempotent retry).
- T-J15: `JobStore.expire_stale_awaiting_collection` fails a job whose `collection_deadline` (default 7 days, `DISTIL_COLLECTOR_EXPIRY_SECONDS`) has passed — from both `awaiting_collection` and `collecting` — with an honest "nobody collected this" error, and leaves one not yet past its deadline untouched; a lease release never resets the deadline, so a video can't outlive its 7-day window by repeatedly timing out leases. `JobStore.list_active` sweeps both expired leases and stale waits on every call, the same piggyback `autoclear` already uses, so the queue self-cleans purely from the existing Activity poll with no collector running.
- T-A5 (external collector credential): `web/auth.py`'s `request_is_collector_authorized` checks a Bearer token against `DISTIL_COLLECTOR_TOKEN` only — a missing/wrong token 401s, the queue is entirely unusable (fails closed) when the env var isn't set, and the credential is proved separate in both directions: the collector token cannot reach any owner route (`/entries`, `/library`, `/ask`, `/bundle.zip`) or log in as the owner even in public mode, and the owner's own `DISTIL_AUTH_SECRET` cannot reach `/collector/*`.
- T-COL1: `POST /collector/jobs/claim` returns waiting jobs' `job_id`/`url`/`lease_expires_at` and flips them to `collecting`; a second claim call never returns an already-claimed job.
- T-COL2: `POST /collector/jobs/{id}/transcript` accepts a valid SRT only for a job currently leased to a collector (`collecting`) — queues it and stages the parsed transcript to disk; a malformed SRT 400s and leaves the lease/job untouched (never accepted into the pipeline); an unleased or unknown job 409s/404s; a retried submit of the same good transcript is idempotent (200 again, still exactly one queued job).
- T-COL3: `POST /collector/jobs/{id}/unfetchable` fails a leased job with the collector's reason; rejects (409) a job that was never leased.

### web/jobs.py + web/app.py — waiting-state presentation (Helper 3; server-computed, template is a thin renderer)
- T-J16: `record_collector_checkin`/`last_collector_checkin` report `None` until the first checkin, overwrite the previous value on each call, and are recorded on every authenticated `/collector/jobs/claim` call — even one that claims nothing — since an empty claim still proves the collector process is alive.
- T-J17: `remove_queued` accepts `AWAITING_COLLECTION` (removable) but rejects `COLLECTING` (a lease a collector may be mid-fetch against) — proved both at the store level and through `POST /jobs/{id}/remove`.
- T-J18: `status_presentation` gives every Activity status, including `awaiting_collection`/`collecting`, its own distinct badge/spinner/live flag — the two waiting states are visibly different from `queued`/`running`/`failed` and stay in the live-polling set; an unrecognized status degrades to an honest "Unknown", non-live badge instead of crashing or silently matching another status.
- T-J19: `collector_status_for_job` reports "nothing is currently collecting" when no checkin was ever recorded, or the last one is older than `COLLECTOR_STALE_SECONDS`, and "a collector checked in Ns ago" when recent; adds an active-fetch note only for `collecting`; reports a `collection_deadline` countdown or a past-due message; and sets `removable=True` only for `awaiting_collection`, never `collecting`.
- T-J20: `GET /jobs` attaches `presentation` (every job) and `collector_status` (only for `awaiting_collection`/`collecting` jobs, `None` otherwise) alongside the existing `collector_last_seen`, covering both the never-checked-in and recently-checked-in cases.
- T-J21 (extends T-J15): the 7-day expiry failure message names YouTube and "collect" explicitly, rather than a bare "failed" — the expiry itself is unchanged, only its wording.

### collector.py — external collector program (Helper 2; talks to the routes above over HTTP, real FastAPI app wrapped behind a urllib-compatible opener in tests, no real network/browser)
- T-COL4: `config_from_env` raises `CollectorConfigError` when `DISTIL_COLLECTOR_SERVER_URL` or `DISTIL_COLLECTOR_TOKEN` is missing, and otherwise reads the optional settings (`DISTIL_COLLECTOR_BROWSER`, `DISTIL_COLLECTOR_POLL_SECONDS`, `DISTIL_COLLECTOR_FETCH_PAUSE_SECONDS`) with their documented defaults.
- T-COL5: `CollectorClient.claim`/`submit_transcript`/`report_unfetchable` round-trip against the real `/collector/*` routes (query string, form-encoded body, Bearer header) via the test opener; a wrong token surfaces as `CollectorHTTPError`.
- T-COL6: `run_collector` end-to-end collects one `awaiting_collection` job — claims it, fetches (fake `run`), submits — leaving it `queued`/`KIND_YOUTUBE_STAGED`; a permanent `YoutubeFetchError` is reported through `report_unfetchable` instead of being retried forever; with nothing waiting it idles and polls again after `DISTIL_COLLECTOR_POLL_SECONDS`.
- T-COL7: `run_collector` never claims a second job before finishing (submitting or reporting-unfetchable) the first — proved by asserting only one `claim` call happens per fetch cycle even across multiple waiting jobs.
- T-COL8: the inter-fetch pause (`DISTIL_COLLECTOR_FETCH_PAUSE_SECONDS`) is applied via an injectable sleep before every fetch after the first, never before the first (mirrors T-J3's shape one layer up).
- T-COL9: a fetch using `DISTIL_COLLECTOR_BROWSER` reports `"signed_in"`/`"anonymous"` per `_detect_browser_session`; leaving it unset is reported and logged as its own `"no-browser-configured"` mode, never silently coerced to `"anonymous"`.
- T-COL10: a transient collector<->server HTTP failure — both a pure transport error and a 5xx response — retries with exponential backoff in `CollectorClient._request` and succeeds once the server responds; a persistent transport failure still raises `CollectorHTTPError` after exhausting attempts; a 4xx is never retried.
- T-COL11: a lost `submit_transcript` response (the request lands and the server queues the job, but the client never sees the reply) is retried and resends the identical job_id + srt text — proved with an opener that performs the real request then raises as if the response were lost — resulting in exactly one queued job, not a duplicate or corrupted state, relying on `/collector/jobs/{id}/transcript`'s own idempotency (T-COL2) rather than any client-side dedup; `run_collector` survives this without crashing.

### collector_net.py — DNS fallback for the collector's system resolver (real `socket.gaierror` from a fake resolver, real local TLS server with a freshly-generated cert; never a mocked `open_with_dns_fallback`)
- T-COL13: `open_with_dns_fallback` resolves normally; when `resolve` raises `socket.gaierror` and a fallback address is stored for that hostname, it connects via the fallback and the request still succeeds.
- T-COL14: with no fallback stored, a `socket.gaierror` propagates as `URLError` on every call (never gives up permanently, never fabricates an address).
- T-COL15: a successful connection — whether resolved normally or via fallback — records the address it used in `DNSFallbackStore`, and the value persists across a fresh `DNSFallbackStore` instance over the same file (survives a restart).
- T-COL16: once `resolve` starts succeeding again, the next call takes the normal path and never consults or reports the stored fallback (`on_degraded` fires only on the degraded call, not the recovered one).
- T-COL17: a stored fallback address that no longer works raises `URLError` for that call, but the very next call retries real resolution and succeeds — a dead cached address never traps the collector.
- T-COL18: connecting to a pinned literal address still performs genuine certificate verification against the request's hostname — a hostname the certificate wasn't issued for fails with `ssl.SSLCertVerificationError`, and the matching hostname succeeds.

### cli.py — `distil collector-run`
- T-COL12: fails cleanly (no traceback) with a readable message when required collector env vars are missing; otherwise starts `run_collector` with the env-derived config and reports whether a browser is configured; a `KeyboardInterrupt` stops the loop cleanly rather than propagating.

### models.py
- T-M1: Profile validates; rejects bad `status` enum.
- T-M2: KnowledgeItem requires `provenance`; `quote` is mandatory, `timestamp` may be null.
- T-M3: `stance` enum enforced; unknown value rejected.
- T-M4: Round-trip serialize→deserialize is lossless.
- T-M5 (Phase D, no-backfill): a `KnowledgeItem` JSON payload predating `entity_mentions` (the field simply absent) still validates, defaulting to an empty list — the shape of every entry filed before this phase.

### triage.py (unit, FakeClient — cheap-tier classifier, runs once per pipeline call)
- T-T1: parses a well-formed model response into a TriageResult.
- T-T2: malformed/partial model JSON → raises a clear ParseError (no silent garbage).
- T-T3 (owner decision): removed — there is no `is_low_value`/short-circuit signal anymore; `verdict` is classified and stored but never gates the pipeline (see pipeline.py's module docstring and T-PL2 below). `run_triage`/`parse_triage_response` are what `pipeline.run_pipeline` calls (cheap tier) ahead of chunked extraction — briefly merged into extraction's own call, then split back out because chunked extraction can't produce one whole-transcript classification (see extract.py's module docstring).
- T-T4 (eval): `low_value_vlog.txt` → verdict `little_to_extract`.
- T-T5 (eval): `screen_share.txt` → `transcript_loss.level == "high"` with non-empty evidence.

### extract.py (unit, FakeClient)
- T-E1: routes to the heuristic extractor when triage says heuristic-dominant.
- T-E2: heuristic items include `rationale` and `scope`; procedural items include `order_index`.
- T-E3 (eval, FAITHFULNESS): every returned item's `provenance.quote` substring-matches the transcript, **for timestamped and untimestamped sources alike**. The quote is the format-independent faithfulness anchor. **This is the headline guarantee — zero tolerance for fabricated provenance.**
- T-E4: each quote is < 15 words (copyright/quote discipline enforced in code).
- T-E5: a response truncated mid-array (output-token cap or dropped connection) recovers whatever complete leading items parsed, discarding only the cut-off tail; recovered items still go through quote truncation/discipline.
- T-E6: a response truncated before even its first item completes yields nothing recoverable → clean `ParseError`, not a crash; a non-array response is still rejected outright.
- T-E7: a dropped connection (`client.complete` raises) retries and can still succeed; a persistent parse or connection failure raises after exhausting the bounded retry count; a schema-level failure in a fully-parsed item is never retried.
- T-E8: an item whose `type` is a `stance` value (e.g. `personal_experience`) is repaired to the requested `KnowledgeType`; a `type` that's a different but still-valid `KnowledgeType` is left alone; an item that still fails validation after repair is dropped while valid siblings survive; an all-bad array, or a batch below the salvage floor, still raises — schema-level failures remain un-retried (T-E7 unchanged).
- T-E9 (Phase D): a knowledge item's nested `entities` array rides back in the *same* extraction call/response (one `FakeClient` call) as `entity_mentions` on the item; a mention with an invalid `kind`, a missing `name`, or a wholesale-malformed `entities` value (not a list) is dropped in `extract._clean_entity_mentions` before the parent item is ever validated, leaving the item — and any valid sibling mention — intact; an over-long mention `quote` is truncated, not dropped; the item-level salvage floor (T-E7/T-E8) is unaffected by entities, which have no salvage floor of their own.
- T-E15 (per-stage `max_tokens` ceiling + truncation visibility): `_parse_items_json` reports `(data, truncated)` — `truncated` is `True` only when the array had to be salvaged via `_recover_truncated_leading_objects`, and `False` both for a clean parse and for a complete array that only needed code-fence/prose stripping (a false positive here would be as unacceptable as a false negative). The ceiling itself is sourced from `model_config.MODEL_MAX_OUTPUT_TOKENS` (Anthropic's published per-model output limits), resolved per stage via `model_config.resolve_stage_max_tokens` — see T-MCFG5.
- T-EC1..EC7 (`tests/unit/test_extract_chunked.py`, chunked extraction — owner decision, supersedes a same-day triage+extract merge): `run_chunked_extraction(transcript, triage, client)` splits the transcript via `summary.chunk_transcript_text` (reused unmodified) and issues one extraction call per chunk. A transcript short enough for one chunk (the common case, and every pre-existing extract fixture) makes exactly one call and keeps the plain `k_01` item-id scheme byte-for-byte; a multi-chunk transcript makes one call per chunk and namespaces ids (`k_c00_k_01`, `k_c01_k_01`, ...) only then, so ids stay unique once chunks are combined. Every chunk after the first is prefixed with a trailing-context overlap from the previous chunk (`[CONTEXT]`/`[NEW MATERIAL]` framing in the prompt) — the first chunk gets none. `ChunkedExtractionResult.truncated` is `True` when *any* chunk's response had to be salvaged, exactly as honestly as the single-call path. `_dedupe_near_duplicate_items` folds items whose normalized statements are a close textual match (`difflib.SequenceMatcher` ratio ≥0.85, not just identical) via `normalize.merge_duplicate_item` (public, reused — not reimplemented); a near-duplicate pair collapses to one item, genuinely different items both survive. Chunk size/overlap are configurable via kwarg (wins) or `DISTIL_EXTRACT_CHUNK_CHARS`/`DISTIL_EXTRACT_CHUNK_OVERLAP_CHARS` env vars, both with defended (non-crashing on a bad env value) defaults.

### normalize.py (PURE)
- T-N1: near-duplicate items are merged.
- T-N2: an item whose provenance quote is NOT in the transcript is dropped/flagged.
- T-N3: opinion content keeps `stance == "opinion"`; never rewritten to look like fact.
- T-N4: for an untimestamped source, items validate with `timestamp = null` and a populated `locator`; the quote check still gates them.

### link.py (unit, FakeClient)
- T-L1: every application_link has a valid `linked_goal_id` pointing at a real profile goal/focus.
- T-L2: with `DISTIL_NOVELTY_RATIO=0.2`, ~1 in 5 links carries `novelty_flag=true`.
- T-L3: cold-start profile (confidence 0) → links reference `stable.long_term_goals`, not learned affinities.

### note.py (unit, FakeClient)
- T-DN1: parses a valid note JSON object into `DistilledNote`.
- T-DN2: sections citing unknown `item_ids` are dropped; partially valid sections keep only valid refs.
- T-DN3: unknown `application_link_ids` are dropped from action steps.
- T-DN4: topics are normalized, deduped, and bounded.
- T-DN5: malformed model output or a failed note call falls back to a deterministic note built from verified items.
- T-DN6: empty verified item list returns no note and makes no model call.

### model_config.py (per-stage model resolution — unit)
- T-MCFG1: every `STRONG_TIER_STAGES` member (`extract`, `canonicalize`) falls back to `DISTIL_MODEL` when set, and raises `RuntimeError` when it isn't — the same "must be set explicitly" contract `AnthropicClient` already enforces, just resolved per stage. `graph` moved to `CHEAP_TIER_STAGES` (owner decision — it was never measured; see the module docstring) and is covered by T-MCFG3 instead.
- T-MCFG2: `DISTIL_MODEL_<STAGE>` overrides one stage only — setting it for one strong-tier stage never leaks into the others, which still resolve to the shared `DISTIL_MODEL`.
- T-MCFG3: every `CHEAP_TIER_STAGES` member (`summary`, `triage`, `link`, `note`, `graph`) defaults to `DEFAULT_CHEAP_TIER_MODEL` (`claude-haiku-4-5`, same value as `DEFAULT_SUMMARY_MODEL`) with no env var set, is unaffected by `DISTIL_MODEL` alone, and its own `DISTIL_MODEL_<STAGE>` overrides it like any other stage. Tier assignment is measured, not assumed — see `model_config.py`'s module docstring for the evidence behind each stage's tier.
- T-MCFG4: `make_stage_client(stage)` constructs an `LLMClient` with the resolved model and never requires `ANTHROPIC_API_KEY` at construction time (enforced lazily, at the first real call, exactly like `AnthropicClient()` today).
- T-MCFG5 (per-stage `max_tokens` ceiling): `resolve_stage_max_tokens("extract")` resolves to `MODEL_MAX_OUTPUT_TOKENS[<resolved model>]` (Anthropic's published per-model output ceiling — `claude-sonnet-5`: 128,000; `claude-haiku-4-5`: 64,000) rather than a guessed number, and falls back to the flat `4096` default for a model string not in that table; every non-`extract` strong-tier stage (and every cheap-tier stage) keeps the flat `4096` default regardless of which model is configured — a small classification/synthesis response doesn't need extraction's headroom. `DISTIL_MAX_TOKENS_<STAGE>` overrides one stage only, with the same isolation guarantee as `DISTIL_MODEL_<STAGE>` (T-MCFG2). `make_stage_client("extract")` constructs an `AnthropicClient` whose `.max_tokens` reflects the resolved ceiling.
- T-MCFG6 (settings-surface precedence): `resolve_stage_model` checks a stored `ModelSettingsStore` setting before `DISTIL_MODEL_<STAGE>` before the tier default; clearing a stored setting reverts resolution to exactly what it was before the setting existed; `resolve_stage_model_info(stage)` reports the correct `source` (`stored`/`env`/`default`/`unconfigured`) for each precedence case, and flags `EXTRACT_CHEAP_MODEL_WARNING` only when `stage == "extract"` and the resolved model is Haiku-family.

### model_settings.py (durable per-stage model overrides — unit)
- T-MSET1: `ModelSettingsStore.set`/`get`/`clear`/`all` round-trip through a real sqlite file at a given path; `get`/`all` on a path with no database file yet return "nothing stored" (`None`/`{}`) without creating the file — a fresh install must never manufacture a database from a read.
- T-MSET2: `set` with a model string outside `KNOWN_MODELS` raises `UnknownModelError` and writes nothing (verified via a following `get` returning `None`) — refused at the point of setting, not the next time it's resolved.
- T-MSET3: a stored setting survives a fresh `ModelSettingsStore` instance pointed at the same path (proves durability isn't an in-memory artifact of one instance).

### summary.py (narrative summary layer — reads the transcript directly, unit, FakeClient)
- T-SUM1: `chunk_transcript_text` splits on sentence boundaries only — no sentence is dropped, duplicated, reordered, or split across a chunk boundary; a chunk respects the target size unless a single sentence is itself longer, in which case that sentence stays whole.
- T-SUM2: chunk size falls back to `DISTIL_SUMMARY_CHUNK_CHARS` when no explicit `chunk_chars` is passed; empty/whitespace-only input yields no chunks.
- T-SUM3: a too-thin chunk or merge summary (below the scaled floor `max(40, ratio * source_len)`) is rejected and retried, succeeding once a long-enough response arrives; a dropped connection (`client.complete` raises) is retried identically; exhausting `max_retries` on either raises `NarrativeSummaryError` rather than returning or persisting a thin result.
- T-SUM4: chunk summaries are merged in chronological order — the merge prompt contains the first chunk's summary before the second's; a single chunk skips the merge call entirely (`chunk_count == 1`).
- T-SUM5: `NarrativeSummary.model` is tagged from the injected client's own `.model` attribute when present, else falls back to `model_config.resolve_stage_model("summary")` — never a hardcoded literal.
- T-SUM6: an empty/whitespace-only transcript raises `NarrativeSummaryError` rather than making a model call.

### graph.py
- T-G1: candidate lookup returns existing entries sharing topics/items (deterministic, no LLM).
- T-G2: relation classification maps to the allowed enum only.

### profile_update.py (PURE — one test per SCHEMA §3 row)
- T-P1: (5, relevant) upweights topics/types/forms tied to the linked goal.
- T-P2: (1, bad_source) leaves the user profile byte-identical (only source model changes).
- T-P3: (2, already_knew) adds the topic to `known_topics`, does NOT add to negatives.
- T-P4: (1, wrong_for_me) increments the matching negatives dimension.
- T-P5: (1, irrelevant_now) applies only a soft/current-focus adjustment.
- T-P6: (5, novelty link) adds a new affinity not previously present.
- T-P7: (3, any) produces a small/zero delta.
- T-P8: updates are EMA-bounded — a single event cannot move a weight past a cap.

### store.py
- T-S1: filing writes `kb/<id>.md` with valid front-matter and a human-readable body.
- T-S2: filing inserts an index row; re-filing same id updates, not duplicates.
- T-S3: KB and DB survive process restart (persistence).
- T-S4: new entries with `distilled_note` render a teaching note first and preserve raw evidence below it; legacy entries still render.
- T-S5: noisy source filenames are cleaned for display, optional YouTube URLs render near the top of notes, and Note v1 evidence is collapsed/de-emphasized.
- T-S6 (OKF, Phase 2): `file_entry(..., transcript=...)` exports `sources/<slug>.md` + `raw/<slug>.md` under `okf_root`; omitting `transcript` (feedback-only re-file) leaves no `okf_root` directory.
- T-S7 (OKF, Phase 2): `okf_root` defaults to a sibling of `kb_dir`. `delete_entry` is pure DB/file-store (kb file, index row, vectors, membership retraction) and deliberately does not touch OKF pages — see T-DEL1-4 for the orchestration layer (`canonicalize.run_delete_entry_stage`) that does.
- T-S8 (owner decision, replaces the removed legacy-low-value pruning test): `list_entries` never prunes on `triage.verdict`/empty `knowledge_items` — a filed entry with `little_to_extract` and zero items is a normal outcome now, and listing-time pruning on it would silently reintroduce the exact discard the pipeline itself no longer does. Only a row whose backing `kb/` file is missing is pruned.
- T-S9: a "Thin material" advisory line renders (in both the note and no-note markdown bodies, and in `teaching_note_markdown`) when `source.transcript_word_count` is below `DISTIL_THIN_TRANSCRIPT_WORDS`; absent when healthy or when the count is `0` (unknown/legacy).

### pipeline.py
- T-PL1: end-to-end with FakeClient produces a complete, schema-valid KBEntry with `distilled_note`.
- T-PL2 (owner decision, replaces the removed `little_to_extract` short-circuit test): a transcript triage classifies `little_to_extract` still runs the full sequence and is filed exactly like any other — `verdict` is stored, never acted on.
- T-PL3: useful transcript with graph disabled stays within four LLM calls: triage, extract, link, note — triage and extraction were briefly merged into one call, then split back out because chunked extraction can't produce one whole-transcript classification (see extract.py's module docstring); a multi-chunk transcript makes one extract call per chunk instead of a flat one.
- T-PL4 (OKF, Phase 2): filing a useful transcript exports both OKF pages (`sources/<slug>.md`, `raw/<slug>.md`) via `store.file_entry`'s `transcript` kwarg.
- T-PL5 (Phase 15.3): with `enable_canonicalize=True`, filing a useful transcript produces a concept page under `okf_root/concepts/` (`type: concept`, a "## Sources" section).
- T-PL6 (Phase 15.3): with `enable_canonicalize=False`, the canonicalize stage makes zero LLM calls and creates no concepts or concept pages.
- T-PL7 (Phase 16): with `enable_concept_edges=True`, a second video whose concept centroid is similar to an existing concept's gets a classified typed edge, rendered under a `## Contrasts with`/`## Builds on`/`## Related` heading on its OKF page.
- T-PL8 (Phase 16): with `enable_concept_edges=False`, the concept-edges stage makes zero LLM calls even when a candidate would otherwise exist, and no concept gains an edge.
- T-PL9 (Phase A, visible progress): `phase_callback` fires `(stage, "start")` before and `(stage, "finish")` after each stage, in pipeline order (`triage`, `extract`, `normalize`, `link`, `note`, `file`, with `graph`/`canonicalize`/`concept_edges` where enabled), without changing `timing_callback`'s existing behaviour.
- T-PL10 (owner decision): `run_pipeline` never emits a `"short_circuit"` phase event itself under any triage verdict — that event moved to the ingest-time word-count gate, reported by callers outside this function (`web/app.py`'s `_distill_job`/`_fetch_playlist_video`, `distil/cli.py`).
- T-PL11 (Phase A): a disabled stage (`enable_graph`/`enable_canonicalize`/`enable_concept_edges=False`) emits no start/finish events, so a caller deriving the declared total from these events reflects only what will actually run.
- T-PL12 (Phase D): an entity mentioned in the extraction response flows end-to-end (canonicalized, synthesized, exported to `okf_root/entities/`) at the existing Stage 8 — the total LLM call count matches concepts-only-plus-entities exactly, with no extra call for a second transcript read.
- T-PL13 (Phase D): with `enable_entities=False`, the entities stage makes zero LLM calls and creates no entities, even though the extraction response's items still carry an `entities` array.
- T-PL14 (narrative summary; concurrency per addendum Part 2 — supersedes an earlier "runs after note" placement): with a distinct cheap-tier `summary_client` injected, the strong client makes exactly its triage+extract+link+note calls — four, not five — and the cheap client makes exactly the narrative-summary call, zero of either landing on the other's client — the model-tier split proven by counting calls on two separate `FakeClient` instances, never by inspecting configuration.
- T-PL15 (narrative summary): omitting `summary_client` (every caller that existed before this stage) reproduces prior behavior exactly — no extra call, `entry.narrative_summary` stays `None`.
- T-PL16 (narrative summary): `config.enable_narrative_summary=False` makes zero calls on the summary client even when one is injected.
- T-PL17 (narrative summary): a narrative-summary failure (thin output exhausting retries, or a dropped connection) is caught and logged — the entry is still filed, just without a `narrative_summary`.
- T-PL18 (narrative summary, addendum Part 2): `phase_callback` reports `("narrative_summary", "start")` FIRST — before `triage`'s own "start" — and `("narrative_summary", "finish")` only after `note`'s "finish" (right before `graph`/`file`) — reflecting that it starts at the front and is joined once the rest of the pipeline has already run, not that it executes between two adjacent stages.
- T-PL19 (addendum Part 2, concurrency proof): a client that sleeps a fixed duration per call, used for both the strong (triage+extract+link+note) and cheap (summary) sides, produces a total `run_pipeline` wall-clock time close to the strong side's own total — not the sum of both sides — proving the two genuinely overlap rather than merely being wired "concurrently" in name; the cheap client's first call timestamp lands within one call-delay of the strong client's first call timestamp, proving the summary starts at the front.
- T-PL20 (addendum Part 2, bounded timeout): with `DISTIL_SUMMARY_JOIN_TIMEOUT_SECONDS` set very low and a summary client that sleeps far longer, `run_pipeline` still returns and files the entry (never blocks indefinitely), `entry.narrative_summary` is `None`, and a "still running" warning is logged (`distil.pipeline` logger) — an honest, non-silent "not generated" state; the owner's existing refresh action can still generate it later.
- T-PL21 (addendum Part 2, independence): an extraction failure (exhausted retries) still raises out of `run_pipeline` even when the concurrently-running summary succeeds — a summary success never masks an extraction failure. Conversely, a summary that fails on its own background thread never affects a successful extraction's outcome — `entry.narrative_summary` is `None` but `entry.knowledge_items` and everything else file normally.

### okf.py (OKF export layer, Phase 2 — pure, no LLM)
- T-OKF1: the export slug is derived from `source.title` (slugified); falls back to `entry_id` when the title yields nothing usable; stable across repeated calls.
- T-OKF2: two distinct entries whose titles collide get distinct slugs — neither export overwrites the other's pages.
- T-OKF3: `sources/<slug>.md` has YAML frontmatter (`type: source`, title, description, slug, published, duration, raw path, tags, created/updated) and a body with a thesis, a chronological "Key moments" section from knowledge-item provenance, and a link to the raw page.
- T-OKF4: the source page never includes `feedback` or `application_links` data (neutrality); see T-OKFC3 for the "## Concepts covered" backlink and T-OKFE3 for the "## Entities mentioned" backlink (Phase D).
- T-OKF5: `raw/<slug>.md` has `type: raw-transcript`, `immutable: true`, and a timestamped body built from `Transcript` segments.
- T-OKF6: `export_entry` regenerates `okf_root/index.md` and `okf_root/sources/index.md` deterministically; re-exporting the same entry is idempotent (no duplicate or orphaned files).
- T-OKF7: `remove_entry` deletes both pages for an entry and refreshes both indexes; removing an entry with no exported pages is a no-op.
- T-OKFC1 (Phase 15.2): `export_concept` writes `concepts/<concept_id>.md` with frontmatter (`type: concept`, title, description, deduped/capped `tags`, sorted unique member `videos` slugs, created/updated), rendered claims with code-derived citations, and a "## Sources" appendix quoting each member's provenance verbatim; never includes `feedback`/`application_links` data.
- T-OKFC2 (Phase 15.2): re-exporting an unchanged concept is byte-identical.
- T-OKFC3 (Phase 15.2): `render_source_with_concepts` (a separate post-canonicalize step, since `export_entry`/Stage 7 runs before canonicalize/Stage 8) adds a source page's "## Concepts covered" backlink section when the source has covering concepts, omits the section when it has none, and is idempotent on re-render.
- T-OKFC4 (Phase 15.2): `remove_concept` deletes the concept page and regenerates `concepts/index.md` and the root index; a concept retracted to zero members is removable this way.
- T-OKFC5 (Phase 16): `export_concept` renders `## Contrasts with`/`## Builds on`/`## Related` sections (same-directory links) only for the relations actually present in `concept.edges`; a concept with no edges renders none of them.
- T-OKFC6 (Phase 16): claims render under a `## Claims` heading, and a claim whose cited members' `stance` values disagree gets a `> **Contradiction:**` line naming each disagreeing member by its resolved OKF slug and stance.
- T-OKFE1 (Phase D): `export_entity` writes `entities/<entity_id>.md` with frontmatter (`type: entity`, `kind`, title, description, created/updated) and rendered claims with code-derived citations, mirroring `export_concept`'s shape one field (`kind`) richer; re-exporting an unchanged entity is byte-identical.
- T-OKFE2 (Phase D): `remove_entity` deletes the entity page and regenerates `entities/index.md` and the root index.
- T-OKFE3 (Phase D): `render_source_with_concepts` adds a source page's "## Entities mentioned" backlink section when the source has covering entities, and omits it when it has none.
- T-OKFE4 (Phase D): a fully wired entity + source pair (export + backlink) lints clean; an entity page with no inbound source backlink fails E12.

### okf_lint.py (stdlib-only bundle validator, `python -m distil.okf_lint <okf_root>`)
- T-OKFL1 (E1): every non-reserved `.md` file must have YAML frontmatter with a non-empty `type`.
- T-OKFL2 (E2): every relative markdown link must resolve to a file inside the bundle.
- T-OKFL3 (E3): every `sources/<slug>.md` must appear in `sources/index.md`.
- T-OKFL4 (E4): `sources/<slug>.md` and `raw/<slug>.md` must exist in pairs (checked in both directions).
- T-OKFL5 (E5, Phase 15.3): every `concepts/<concept_id>.md` must have `type: concept` frontmatter and appear in `concepts/index.md`.
- T-OKFL6 (E6, Phase 15.3): every source cited in a concept's "## Sources" section must be one of that concept's `videos:` frontmatter slugs.
- T-OKFL7 (E7, Phase 15.3): concept<->source links must be bidirectional — a concept page's `videos:` slug must have a matching backlink on that source's "## Concepts covered" section, and vice versa.
- T-OKFL8 (E8, Phase 15.3): no orphan concept pages — every `concepts/<concept_id>.md` must be linked from at least one source's "## Concepts covered" section (the `concepts/index.md` listing alone does not count); Phase 16 extends this to also recognize another concept's typed-edge link (`## Contrasts with`/`## Builds on`/`## Related`) as a valid inbound link.
- T-OKFL9-12 (E9-E12, Phase D): the E5-E8 checks applied one-for-one to `entities/` — frontmatter (`type: entity`) + index coverage, entity<->source citation integrity, bidirectional entity<->source links, and no orphan entity pages (minus the typed-edge extension, since entities have no entity<->entity edges).
- A freshly generated bundle (including concept and entity pages produced by `canonicalize.run_canonicalize_stage`) lints clean, and `main()` exits non-zero when any error is present. An entry with zero entities (no backfill, Phase D) lints clean with an empty `entities/index.md`.

### canonicalize.py (concept matching engine, Phase 15.1 — unit, FakeClient/FakeEmbedder)
- T-CANON1: a near-match candidate returned as `match` appends the filing entry's item as a new member of the existing concept; no new concept is created.
- T-CANON2: an empty candidate pool with a `new` decision creates a fresh concept (slug derived from the proposed title) with the item as its sole member.
- T-CANON3: re-canonicalizing the same entry is idempotent — memberships are retracted before reapplying, so the resulting member list is exactly equal (not merely same-length) across repeated runs, with no duplicate concepts.
- T-CANON4: a decision outside the match/new/reject enum is dropped (treated as reject) rather than raising or fabricating a match.
- T-CANON5: a `match` naming a `concept_id` that was never offered as a candidate is dropped, not trusted, even if it happens to be a real concept.
- T-CANON6: the candidate pool embedded in the prompt is capped at `MAX_CONCEPT_CANDIDATES` (default 5) even when more concepts qualify.
- T-CANON7: two same-batch `new` proposals whose normalized titles collide are merged into a single concept instead of creating duplicates.
- T-CANON9: `Store.delete_entry` cascades into `retract_entry_concept_memberships` — deleting an entry's sole source removes the concept entirely, while deleting one of several sources only retracts that entry's membership and leaves the concept intact for the rest.
- T-CANON8 (Phase 15.2): `synthesize_touched_concepts` ranks a video's touched concepts by embedding similarity (`Store.concept_centroid`) and synthesizes only the top `MAX_CONCEPTS_TO_SYNTHESIZE_PER_VIDEO` (env `DISTIL_CONCEPTS_SYNTH_PER_VIDEO`, default 5); the rest are marked `pending_synthesis=True` and make no LLM call. A touched set within the cap synthesizes all of them.
- T-CANON-EVAL1 (eval, Phase 15.3, §6 validation gate): against the real configured model + a real local embedder, 3 genuine paraphrases of the same idea ("traditional RAG") land in one concept, while a lexically-adjacent but distinct idea ("agentic RAG") does not merge into it.
- T-CANON-EVAL2 (eval, Phase 15.3, §6 validation gate): under real model output, `synthesize_concept` produces at least one kept claim, and every claim's `item_ids` are non-empty and all resolve to real members of the concept.

### canonicalize.py — entity matching engine (Phase D — unit, FakeClient/FakeEmbedder)
`canonicalize_entry_entities`/`synthesize_touched_entities` mirror T-CANON1-8 one granularity down (entity mentions instead of knowledge items), with `Store.find_entity_candidates` adding a hard `kind` pre-filter concepts don't have.
- T-CANONE1: an item with no `entity_mentions` makes zero LLM calls and produces no entities.
- T-CANONE2: a `new` decision creates a fresh `Entity` (kind taken from the mention) with the mention as its sole member.
- T-CANONE3 (core acceptance criterion): the same tool mentioned across two videos merges into one `Entity` page with two members, rather than two separate entity pages.
- T-CANONE4: a `reject` decision produces no entity.
- T-CANONE5: a `match` naming an `entity_id` never offered as a candidate is dropped, not trusted (mirrors T-CANON5).
- T-CANONE6: re-processing the same entry is idempotent — no duplicate entities, member list unchanged across repeated runs.
- T-CANONE7: `synthesize_touched_entities` synthesizes only the top `MAX_ENTITIES_TO_SYNTHESIZE_PER_VIDEO` (env `DISTIL_ENTITIES_SYNTH_PER_VIDEO`, default 5) touched entities by embedding similarity; the rest are marked `pending_synthesis=True` and make no LLM call (mirrors T-CANON8).

### canonicalize.py — run_delete_entry_stage (delete-cascade orchestration — unit, FakeClient/FakeEmbedder)
- T-DEL1: deleting an entry that was the sole source of a concept removes both the DB row (`Store.delete_entry`'s existing retraction) and the now-orphaned `concepts/<id>.md` page + index entries.
- T-DEL2: deleting one of several sources of a surviving concept retracts only that membership and re-exports the concept's page, so it stops listing the deleted video (no stale back-reference).
- T-DEL3/T-DEL4: when `kb/<id>.md` is missing or fails to parse, the entry's `sources/<slug>.md`/`raw/<slug>.md` pages are still removed — the slug is recovered from the source page's own `distil_entry_id` frontmatter (`okf.find_slug_for_entry_id`) rather than requiring a loadable `KBEntry`.
- T-DELE1/T-DELE2 (Phase D): `run_delete_entry_stage` extends the same treatment to `entities/` — deleting an entry that was an entity's sole source removes the orphaned `entities/<id>.md` page + index entries, while deleting one of several sources retracts only that membership and re-exports the entity's page without the deleted video's backreference.
- The reconciled/deleted bundle lints clean (`okf_lint.lint`) in every case above.

### reconcile.py (bundle drift repair — unit, FakeClient/FakeEmbedder)
- T-REC1: an orphaned `sources/<slug>.md` (`distil_entry_id` not a live DB entry) is removed along with its `raw/<slug>.md` pair; a legitimate, live-owned pair is left untouched.
- T-REC2: an orphaned `concepts/<id>.md` (concept_id not in the DB) is removed and indexes regenerated; removing it also re-renders any live source's "## Concepts covered" section that pointed at it, so no dangling link survives.
- T-REC3: dry run (`apply=False`, the default) reports every file it would remove but deletes nothing.
- T-REC4: a `sources/` page with no (or unparseable) `distil_entry_id`, or a `raw/` page with no matching `sources/` page, has an undeterminable owner — reported as skipped, never deleted.
- T-REC5: reconcile never touches `kb/` or the database; the reconciled bundle passes `okf_lint.lint`.
- T-REC6 (Phase D): an orphaned `entities/<id>.md` (entity_id not in the DB) is removed and indexes regenerated, mirroring T-REC2.

### synthesize_concept.py (concept-page synthesis, Phase 15.2 — unit, FakeClient)
- T-SYN1: valid claims JSON parses into cleaned `ConceptClaim`s.
- T-SYN2: a claim whose `item_ids` don't ALL resolve to real members of this concept is dropped whole (stricter than `note.py`'s per-id filtering).
- T-SYN3: malformed/empty-after-cleaning model output falls back to a deterministic one-liner built from the concept description and member statements; never raises.
- T-SYN4: `render_claim` is a pure function — the rendered citation parenthetical exactly matches the code-derived `(okf_slug, timestamp)` map, including multi-citation and no-timestamp cases; the synthesis model never authors citation text.
- T-SYN5 (Phase 16): `find_claim_contradictions` flags a claim whose cited members' `stance` values disagree, keyed by claim index with the full `(entry_id, item_id, stance)` row per member; a claim whose members agree (or that cites only one member) is absent from the result — no LLM involved.

### concept_graph.py (concept<->concept typed edges, Phase 16 — unit, FakeClient/FakeEmbedder)
- T-CEDGE1: with no other concepts in the store, `link_concept_graph` makes zero LLM calls and returns no edges (mirrors T-G1's no-candidates shortcut).
- T-CEDGE2: a concept whose centroid is similar enough to another concept's becomes a classified candidate; the LLM's relation is stored as a `ConceptEdge`.
- T-CEDGE3: a `none` relation, or any value outside `{contrasts_with, builds_on, related}`, is dropped rather than trusted (mirrors T-G2).
- T-CEDGE4: the candidate pool is capped at `MAX_CONCEPT_EDGE_CANDIDATES` (default 3) even when more concepts qualify.
- T-CEDGE5: `run_concept_edges_stage` skips concepts still `pending_synthesis` (zero LLM calls, no edges written) and exports the OKF page only for concepts whose edges actually changed.
- T-CEDGE6: `Store.prune_dangling_concept_edges` drops an edge whose target concept was deleted, and the affected concept's OKF page is re-exported without the stale edge section.

### refresh_summary.py (per-entry narrative-summary refresh — unit, FakeClient)
- T-RS1: refreshing a fully filed entry regenerates only `narrative_summary`; knowledge items, concept membership/pages, and the raw OKF page are all byte-identical/unchanged before and after.
- T-RS2: refresh makes exactly one model call (the narrative-summary chunk call) — a shared client seeded with only that one response would `IndexError` on any re-run of extraction, note, or canonicalize, proving nothing else executes.
- T-RS3: an entry with no stored raw transcript page (filed before OKF export existed, or reconciled away) reports plainly that it can't generate a summary and that retrying won't help, rather than a doomed retry or an obscure failure.
- T-RS4: refreshing an unknown entry id reports "not found" rather than raising.
- T-RS5: exhausting the coverage-floor retries reports "could not generate" and leaves `narrative_summary` at its prior value (`None` here), never a partial/thin result.

### cli.py
- T-C1: `distil run <file>` accepts `.srt`/`.txt`/`.md` and `distil run --paste` (or stdin) accepts pasted text; exits 0 and prints the entry path.
- T-C2: `distil score <id> --score 5 --reason relevant` mutates the profile.
- T-C3: missing API key → friendly error, not a stack trace.
- T-C4: `distil ask "..."` prints an answer + source links, or the no-notes message.
- T-C5: `distil reindex` embeds entries that have no stored vector yet.
- T-C6: `distil run --url <youtube-url>` stores source attribution; non-YouTube source URLs are rejected cleanly.
- T-C7: `distil delete <entry_id> --yes` removes the markdown file, SQLite index row, and item vectors.
- T-C8: `distil refresh-summary <id>` regenerates the summary and prints confirmation; a missing entry exits non-zero with a friendly "not found" message; the command uses the summary-tier client seam (`_make_summary_client`) exclusively — it must never construct the main (strong-tier) client.
- T-C9: `_make_extract_client`/`_make_canonicalize_client` default to `DISTIL_MODEL` with no override set; `_make_triage_client`/`_make_link_client`/`_make_note_client`/`_make_graph_client` default to the cheap tier instead (`graph` moved there — owner decision, see `model_config.py`'s module docstring); every seam's own `DISTIL_MODEL_<STAGE>` override genuinely changes that seam's resolved model — the gap this closes: before, `cli.py`/`web/app.py` shared one `_make_client()` object across link/note/graph/canonicalize, so `DISTIL_MODEL_<STAGE>` resolved correctly in `model_config.py` but nothing downstream ever constructed a client from it. `_make_summary_client` is unaffected and still defaults to the Haiku tier regardless of `DISTIL_MODEL`.

### embed.py / store vectors (unit, FakeEmbedder)
- T-X1: filing an entry stores one vector per knowledge item in the `vec0` table.
- T-X2: `reindex` backfills vectors for entries filed before the read layer; idempotent (no duplicate vectors).
- T-X3: `Embedder` is pluggable — swapping local↔api changes only construction, not call sites.

### query.py — retrieval + GROUNDING + ABSTENTION (headline guarantees)
- T-Q1: KNN search returns items ranked by similarity × feedback_score × recency (deterministic given fixed vectors via FakeEmbedder).
- T-Q2 (**ABSTENTION — headline**): a question whose best match is below `DISTIL_RETRIEVAL_THRESHOLD` returns the "no relevant notes" result **and the synthesis LLM is never called** (assert the `LLMClient` answer method received zero calls). This is the no-hallucination guarantee in test form.
- T-Q3 (**GROUNDING — headline**): for an answered question, every claim/citation in the answer maps to an item that was in the retrieved set; the answer references no source outside it.
- T-Q4: every answer carries resolvable source links (entry id + item id + provenance timestamp).
- T-Q5: a bare lookup ("do I have notes on X") returns the ranked source list with no synthesis call.
- T-Q6: when retrieved items are linked by a `contradicts` edge, the answer surfaces the conflict instead of picking one silently.
- T-Q7 (eval): on the query KB fixture, answerable questions return correct source IDs and no-note questions abstain 100% of the time.
- T-Q8: retrieved items include any distilled-note context that cites them, while source links remain item-level.
- T-Q9 (Phase 18, query-over-concepts): `retrieve_concepts` ranks synthesized concept pages by similarity to the question, same as `retrieve` does for raw items.
- T-Q10: a concept's centroid-diluting members don't hide it — blending in per-member similarity still surfaces the concept via its closest member.
- T-Q11: a question that only a concept (not any single raw item) clears the threshold for still answers instead of abstaining — concepts get the SAME threshold, never a lower one.
- T-Q12: sources recruited via a concept resolve from the live `KBEntry`/`KnowledgeItem`, never from the concept's own stale copied fields or model text — code-assembled citations hold even off the concept path.
- T-Q13 (regression): a concept present in the KB with low similarity to the question does not lower the bar — abstention still holds, zero synthesis calls.
- T-Q14 (Phase C): `ask()` and `stream_ask()`'s final result both carry `concepts` (the concepts that cleared the gate) at zero extra retrieval/model cost — surfaced from `web/app.py`'s `_ask_payload` for both the `/ask` and `/ask/stream` routes.
- T-Q15 (Phase C, regression): an abstained result carries `concepts == []`.
- T-Q16 (Phase C): `DISTIL_CONCEPT_NOTE_DEPTH` (or the `concept_note_depth` kwarg, which wins over the env) controls whether a matched concept's member quotes reach the synthesis prompt — `"claims"` (default) sends claim text only, `"full"` inlines each cited member's quote; an unrecognized value degrades to the default rather than raising.
- T-Q17 (Phase C, regression): `depth="full"`'s richer concept-notes block does not weaken citation validation — a fabricated citation is still stripped and reported as ungrounded.
- T-QE1-4 (Phase D): `retrieve_entities`/`ask` gate entities through the exact same `threshold` as items/concepts (never an easier bar) — ranks relevant entities first, an entity clearing the gate alone avoids abstention, a low-similarity entity never bypasses it (zero synthesis calls), and recruited sources resolve from the live `KBEntry` rather than the entity's own stale copied quote/timestamp. Mirrors T-Q9/T-Q11/T-Q13/T-Q12 one granularity down.

### web/app.py — POST /entries/{id}/refresh-summary
- T-WRS1: a successful refresh returns 200 with `{"ok": true}`, and the entry page renders a "Narrative summary" section once one is present.
- T-WRS2: an unknown entry id 404s.
- T-WRS3: an entry with no stored transcript returns 200 with `{"ok": false}` and a message naming the transcript, not a 4xx/5xx — a plain, expected report, not an error.

### auth (web, hosted) — `web/`
- T-A1: with `DISTIL_PUBLIC=true` and no `DISTIL_AUTH_SECRET` set, the app refuses to start/serve (fails closed).
- T-A2: a request without valid credentials to any data route returns 401, never data.
- T-A3: bound to localhost (not public), routes are reachable without the secret (dev convenience).
- T-A4: the server binds `0.0.0.0:$PORT` when `PORT` is set (Railway readiness).

## 4. CI

GitHub Actions: on every push run `pytest tests/unit` + lint. The `eval` job runs only on
manual dispatch or when a repo secret `ANTHROPIC_API_KEY` is present. A green required check =
all unit tests pass. Coverage target: 90% on pure modules (`models`, `normalize`,
`profile_update`, `store`, and the retrieval/gate logic in `query`). The abstention gate
(T-Q2) and grounding check (T-Q3) are required checks — a regression there is a release blocker.
