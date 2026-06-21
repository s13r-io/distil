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

### pipeline.py
- T-PL1: end-to-end with FakeClient produces a complete, schema-valid KBEntry with `distilled_note`.
- T-PL2: `little_to_extract` path files a minimal entry and makes no extract/link calls.
- T-PL3: useful transcript with graph disabled stays within four LLM calls: triage, extract, link, note.

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
