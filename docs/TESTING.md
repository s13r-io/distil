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
(plain text with inline `00:12:30` markers), and `no_timestamps.md` (prose, no timestamps at all).

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

### youtube.py (fetch layer — `yt-dlp` invoked via an injectable ``run``, no real subprocess/network in unit tests)
- T-Y1: a playlist URL (`?list=` with no `v`, or `/playlist` path) is distinguished from a single video URL.
- T-Y2: a playlist enumerates to a list of normalized `watch?v=` URLs; empty/malformed listings raise `YoutubeFetchError`.
- T-Y3: a captioned video's fetched `.srt` parses into a `Transcript` via `ingest.ingest_srt_text` (same shape as uploaded `.srt`).
- T-Y4: a video with no caption file written (no captions available) raises `YoutubeFetchError`, not a crash.
- T-Y5: a `yt-dlp` process failure (private/deleted video) raises `YoutubeFetchError` with the underlying stderr.
- T-Y6 (web/jobs): one uncaptioned/failed video in a playlist batch is marked `failed` on its own job; the next queued job still processes — never fatal to the batch (`Worker._process` already isolates per-job exceptions; `web/app.py` enqueues one `kind="youtube"` job per video).
- T-Y7: a caller-supplied `workdir` reused across fetches (e.g. a shared `tmp_path`) never picks up a stale `.srt` left by a previous fetch — each fetch is scoped to its own unique child directory, and the stale file is left untouched.
- T-Y8 (Phase 19; client chain updated Phase 21): both `list_playlist_video_urls` and `_fetch_into` pass `--extractor-args youtube:player_client=android_vr,web_safari` to `yt-dlp` (see `distil/youtube.py`'s module docstring for why this chain, re-derived from yt-dlp's own defaults).
- T-Y9 (Phase 19): a transient failure (429/5xx) retries with exponential backoff (injectable `sleep`) and succeeds once `yt-dlp` returns success within the bounded attempt count; a persistent transient failure still raises `YoutubeFetchError` after exhausting attempts; a non-transient failure (e.g. private/deleted video) raises immediately with no retry/sleep.
- T-Y10 (Phase 22; supersedes the removed Phase 20 `DISTIL_YOUTUBE_API_KEY` variant): with `DISTIL_POT_PROVIDER_URL` set (via `monkeypatch`), both `list_playlist_video_urls` and `_fetch_into` pass a *second* `--extractor-args` pair, `youtubepot-bgutilhttp:base_url=<value>`, alongside the unchanged `youtube:player_client=android_vr,web_safari` pair; with the env var unset, the command line is byte-identical to T-Y8 (single `--extractor-args` pair, no `DISTIL_YOUTUBE_API_KEY` handling survives anywhere in the module).
- T-Y11 (Phase 21): a `yt-dlp` failure whose stderr is warning-heavy (SABR/staleness noise exceeding any head-truncation budget) still surfaces its `ERROR:`-prefixed line(s) in the raised `YoutubeFetchError`, not the leading warnings; the complete, untruncated stderr is always logged via `logging.getLogger("distil.youtube")` regardless of what the bounded exception message contains; stderr with no `ERROR:` line at all falls back to a genuine tail (the *last* N chars).
- T-Y12 (Phase 21): `_fetch_into` requests `--sub-format srt/best` and never passes `--convert-subs` (the Dockerfile image has no ffmpeg, so any format needing conversion would fail in production).
- T-Y13 (Phase A, visible progress): a successful fetch reports `("transcript_fetch", "start")`, `("transcript_fetch", "finish")`, `("caption_parse", "start")`, `("caption_parse", "finish")`, in that order, via the optional `on_phase` callback.
- T-Y14 (Phase A): a `yt-dlp` failure reports only `("transcript_fetch", "start")` — it never advances to `caption_parse`, so a stalled/failed fetch reads as stuck on the right phase rather than silently progressing.

### models.py
- T-M1: Profile validates; rejects bad `status` enum.
- T-M2: KnowledgeItem requires `provenance`; `quote` is mandatory, `timestamp` may be null.
- T-M3: `stance` enum enforced; unknown value rejected.
- T-M4: Round-trip serialize→deserialize is lossless.

### triage.py (unit, FakeClient)
- T-T1: parses a well-formed model response into a TriageResult.
- T-T2: malformed/partial model JSON → raises a clear ParseError (no silent garbage).
- T-T3: `little_to_extract` verdict short-circuits the pipeline (no extract call made).
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

### pipeline.py
- T-PL1: end-to-end with FakeClient produces a complete, schema-valid KBEntry with `distilled_note`.
- T-PL2: `little_to_extract` path returns a minimal result without filing and makes no extract/link calls.
- T-PL3: useful transcript with graph disabled stays within four LLM calls: triage, extract, link, note.
- T-PL4 (OKF, Phase 2): filing a useful transcript exports both OKF pages (`sources/<slug>.md`, `raw/<slug>.md`) via `store.file_entry`'s `transcript` kwarg.
- T-PL5 (Phase 15.3): with `enable_canonicalize=True`, filing a useful transcript produces a concept page under `okf_root/concepts/` (`type: concept`, a "## Sources" section).
- T-PL6 (Phase 15.3): with `enable_canonicalize=False`, the canonicalize stage makes zero LLM calls and creates no concepts or concept pages.
- T-PL7 (Phase 16): with `enable_concept_edges=True`, a second video whose concept centroid is similar to an existing concept's gets a classified typed edge, rendered under a `## Contrasts with`/`## Builds on`/`## Related` heading on its OKF page.
- T-PL8 (Phase 16): with `enable_concept_edges=False`, the concept-edges stage makes zero LLM calls even when a candidate would otherwise exist, and no concept gains an edge.
- T-PL9 (Phase A, visible progress): `phase_callback` fires `(stage, "start")` before and `(stage, "finish")` after each stage, in pipeline order, without changing `timing_callback`'s existing behaviour.
- T-PL10 (Phase A): the `little_to_extract` short-circuit reports only `("triage", "start")`, `("triage", "finish")`, `("triage", "short_circuit")` — no events for stages that never ran.
- T-PL11 (Phase A): a disabled stage (`enable_graph`/`enable_canonicalize`/`enable_concept_edges=False`) emits no start/finish events, so a caller deriving the declared total from these events reflects only what will actually run.

### okf.py (OKF export layer, Phase 2 — pure, no LLM)
- T-OKF1: the export slug is derived from `source.title` (slugified); falls back to `entry_id` when the title yields nothing usable; stable across repeated calls.
- T-OKF2: two distinct entries whose titles collide get distinct slugs — neither export overwrites the other's pages.
- T-OKF3: `sources/<slug>.md` has YAML frontmatter (`type: source`, title, description, slug, published, duration, raw path, tags, created/updated) and a body with a thesis, a chronological "Key moments" section from knowledge-item provenance, and a link to the raw page.
- T-OKF4: the source page omits an Entities section (not implemented until a later phase) and never includes `feedback` or `application_links` data (neutrality); see T-OKFC3 for the Phase 15.2 "## Concepts covered" backlink.
- T-OKF5: `raw/<slug>.md` has `type: raw-transcript`, `immutable: true`, and a timestamped body built from `Transcript` segments.
- T-OKF6: `export_entry` regenerates `okf_root/index.md` and `okf_root/sources/index.md` deterministically; re-exporting the same entry is idempotent (no duplicate or orphaned files).
- T-OKF7: `remove_entry` deletes both pages for an entry and refreshes both indexes; removing an entry with no exported pages is a no-op.
- T-OKFC1 (Phase 15.2): `export_concept` writes `concepts/<concept_id>.md` with frontmatter (`type: concept`, title, description, deduped/capped `tags`, sorted unique member `videos` slugs, created/updated), rendered claims with code-derived citations, and a "## Sources" appendix quoting each member's provenance verbatim; never includes `feedback`/`application_links` data.
- T-OKFC2 (Phase 15.2): re-exporting an unchanged concept is byte-identical.
- T-OKFC3 (Phase 15.2): `render_source_with_concepts` (a separate post-canonicalize step, since `export_entry`/Stage 7 runs before canonicalize/Stage 8) adds a source page's "## Concepts covered" backlink section when the source has covering concepts, omits the section when it has none, and is idempotent on re-render.
- T-OKFC4 (Phase 15.2): `remove_concept` deletes the concept page and regenerates `concepts/index.md` and the root index; a concept retracted to zero members is removable this way.
- T-OKFC5 (Phase 16): `export_concept` renders `## Contrasts with`/`## Builds on`/`## Related` sections (same-directory links) only for the relations actually present in `concept.edges`; a concept with no edges renders none of them.
- T-OKFC6 (Phase 16): claims render under a `## Claims` heading, and a claim whose cited members' `stance` values disagree gets a `> **Contradiction:**` line naming each disagreeing member by its resolved OKF slug and stance.

### okf_lint.py (stdlib-only bundle validator, `python -m distil.okf_lint <okf_root>`)
- T-OKFL1 (E1): every non-reserved `.md` file must have YAML frontmatter with a non-empty `type`.
- T-OKFL2 (E2): every relative markdown link must resolve to a file inside the bundle.
- T-OKFL3 (E3): every `sources/<slug>.md` must appear in `sources/index.md`.
- T-OKFL4 (E4): `sources/<slug>.md` and `raw/<slug>.md` must exist in pairs (checked in both directions).
- T-OKFL5 (E5, Phase 15.3): every `concepts/<concept_id>.md` must have `type: concept` frontmatter and appear in `concepts/index.md`.
- T-OKFL6 (E6, Phase 15.3): every source cited in a concept's "## Sources" section must be one of that concept's `videos:` frontmatter slugs.
- T-OKFL7 (E7, Phase 15.3): concept<->source links must be bidirectional — a concept page's `videos:` slug must have a matching backlink on that source's "## Concepts covered" section, and vice versa.
- T-OKFL8 (E8, Phase 15.3): no orphan concept pages — every `concepts/<concept_id>.md` must be linked from at least one source's "## Concepts covered" section (the `concepts/index.md` listing alone does not count); Phase 16 extends this to also recognize another concept's typed-edge link (`## Contrasts with`/`## Builds on`/`## Related`) as a valid inbound link.
- A freshly generated bundle (including concept pages produced by `canonicalize.run_canonicalize_stage`) lints clean, and `main()` exits non-zero when any error is present.

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

### canonicalize.py — run_delete_entry_stage (delete-cascade orchestration — unit, FakeClient/FakeEmbedder)
- T-DEL1: deleting an entry that was the sole source of a concept removes both the DB row (`Store.delete_entry`'s existing retraction) and the now-orphaned `concepts/<id>.md` page + index entries.
- T-DEL2: deleting one of several sources of a surviving concept retracts only that membership and re-exports the concept's page, so it stops listing the deleted video (no stale back-reference).
- T-DEL3/T-DEL4: when `kb/<id>.md` is missing or fails to parse, the entry's `sources/<slug>.md`/`raw/<slug>.md` pages are still removed — the slug is recovered from the source page's own `distil_entry_id` frontmatter (`okf.find_slug_for_entry_id`) rather than requiring a loadable `KBEntry`.
- The reconciled/deleted bundle lints clean (`okf_lint.lint`) in every case above.

### reconcile.py (bundle drift repair — unit, FakeClient/FakeEmbedder)
- T-REC1: an orphaned `sources/<slug>.md` (`distil_entry_id` not a live DB entry) is removed along with its `raw/<slug>.md` pair; a legitimate, live-owned pair is left untouched.
- T-REC2: an orphaned `concepts/<id>.md` (concept_id not in the DB) is removed and indexes regenerated; removing it also re-renders any live source's "## Concepts covered" section that pointed at it, so no dangling link survives.
- T-REC3: dry run (`apply=False`, the default) reports every file it would remove but deletes nothing.
- T-REC4: a `sources/` page with no (or unparseable) `distil_entry_id`, or a `raw/` page with no matching `sources/` page, has an undeterminable owner — reported as skipped, never deleted.
- T-REC5: reconcile never touches `kb/` or the database; the reconciled bundle passes `okf_lint.lint`.

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

### cli.py
- T-C1: `distil run <file>` accepts `.srt`/`.txt`/`.md` and `distil run --paste` (or stdin) accepts pasted text; exits 0 and prints the entry path.
- T-C2: `distil score <id> --score 5 --reason relevant` mutates the profile.
- T-C3: missing API key → friendly error, not a stack trace.
- T-C4: `distil ask "..."` prints an answer + source links, or the no-notes message.
- T-C5: `distil reindex` embeds entries that have no stored vector yet.
- T-C6: `distil run --url <youtube-url>` stores source attribution; non-YouTube source URLs are rejected cleanly.
- T-C7: `distil delete <entry_id> --yes` removes the markdown file, SQLite index row, and item vectors.

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
