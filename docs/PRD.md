# PRD — Distil: Personal Knowledge Distiller

**Status:** v0 (kickoff) · **Owner:** Project owner (human) · **Builder:** AI agent
**Related docs:** `ARCHITECTURE.md`, `SCHEMA.md`, `TESTING.md`, `AGENT_BUILD_GUIDE.md`, `TRACKER.md`

---

## 1. Problem

People consume hours of video (tutorials, talks, podcasts) and retain almost none of it in
a usable form. Generic summarizers make this worse: they flatten every video into the same
bullet list, invent insights when there are none, and never connect what was said to the
viewer's actual life. The result is a graveyard of summaries nobody reuses.

## 2. Product

Distil takes a **YouTube video transcript** and a **stored user profile**, and produces a
filed knowledge-base document containing (a) the knowledge extracted faithfully and routed
by its type, and (b) concrete ideas for applying that knowledge to *this specific user's*
goals and projects. The user scores each result; the score (plus a reason) updates the
profile, so the system gets more personally useful over time. Filed documents accumulate
into a cross-linked personal knowledge base.

## 3. Goals & non-goals

**Goals**
- Extract knowledge faithfully — every item traceable to the transcript (no invented insight).
- Route extraction by knowledge type (procedural, declarative, conceptual, heuristic, experiential, opinion) so output shape fits content.
- Be honest about low-value input — emit a "little to extract" verdict rather than manufacture insights.
- Generate application ideas tied to the user's stored goals and current focus.
- Learn from a per-result score + reason, growing a structured profile without drifting into a filter bubble.
- File results as markdown documents cross-linked to existing entries (a knowledge *base*, not a folder).
- Be open source and deployable by anyone with their own LLM API key.

**Non-goals (v0)**
- No automatic YouTube scraping / URL fetch in v0; the transcript is supplied as pasted text or an uploaded file (`.srt`, `.txt`, `.md`). URL fetch is deferred to a later release.
- No multi-user accounts. Single-user. **No auth is acceptable only when bound to localhost** — the moment the app is exposed on a public URL (e.g. hosted on Railway), a minimal auth gate is required (see FR14). Hosting without auth would expose your API key and private knowledge base to anyone with the URL.
- No mobile app; CLI first, minimal web UI second.
- No fine-tuning of models; personalization lives entirely in the profile + prompts.
- No video/audio processing; text transcript only (transcript-loss is *detected*, not recovered).

## 4. Users & primary use case

A single self-hosting user (the "owner") who wants a private, growing knowledge base.
Primary loop:

1. Owner supplies a transcript.
2. System triages → extracts → links to profile → files a markdown entry.
3. Owner reviews and scores the entry (1–5) with a reason.
4. Profile updates; next results improve.

## 5. Functional requirements

| ID   | Requirement                                                                                          | Priority |
|------|------------------------------------------------------------------------------------------------------|----------|
| FR1  | Accept a transcript as pasted text **or** an uploaded file (`.srt`, `.txt`, `.md`). No URL fetch in v0. | Must     |
| FR2  | Triage: classify knowledge types present, density, transcript-loss level (+evidence), and a verdict. | Must     |
| FR3  | Extract atomic, self-contained knowledge items with type-specific fields.                            | Must     |
| FR4  | Attach provenance to every knowledge item: a short quote always, plus a timestamp **when the source has one** (transcripts may be untimestamped). | Must |
| FR5  | Preserve stance (fact / opinion / personal experience); never present opinion as fact.               | Must     |
| FR6  | Generate application links tied to a specific profile goal or current-focus item.                    | Must     |
| FR7  | Reserve a fraction of application links flagged as novelty/serendipity (anti-bubble).                | Should   |
| FR8  | Cross-link new entries to existing KB entries (supports/contradicts/same-principle/etc.).            | Should   |
| FR9  | Record a per-result score (1–5) and a reason from a fixed set.                                        | Must     |
| FR10 | Update the profile from score+reason per the rules in `SCHEMA.md` §3.                                | Must     |
| FR11 | File each result as a markdown document; index it in SQLite.                                          | Must     |
| FR12 | Honest verdict: when input is low-value, say so instead of fabricating items.                        | Must     |
| FR13 | Minimal web UI to view/score entries and browse the KB.                                              | Could    |
| FR14 | Auth gate (single-user login/secret) enforced whenever the app is not bound to localhost.            | Must (hosted) |
| FR15 | `ask` a question; retrieve the most relevant knowledge items from the KB by semantic search.         | Must     |
| FR16 | Answer questions **grounded only in retrieved notes** — no use of the model's outside knowledge.     | Must     |
| FR17 | Every answer links to its source notes (entry + item + timestamp); a bare lookup returns the ranked list. | Must |
| FR18 | Honest abstention: if no note clears the relevance threshold, say no relevant notes exist and do **not** synthesize an answer. | Must |
| FR19 | `reindex` to embed previously filed entries (backfill when the read layer is added).                 | Must     |
| FR20 | Surface conflicts when retrieved notes disagree, rather than silently choosing one.                  | Should   |
| FR21 | Ingest `.srt`, `.txt`, `.md`, or pasted text into one normalized internal transcript; parse `.srt`/inline timestamps when present. | Must |
| FR22 | Handle untimestamped transcripts gracefully: provenance degrades to quote-only (plus a line/segment locator), and the faithfulness check (quote present in source) still holds. | Must |

## 5.1 Querying the knowledge base (read layer)

Without this, the KB is write-only: notes go in and never come out. The read layer makes it
consultable. The user asks a question (or a "do I have notes on X?" lookup) and the system:

1. Embeds the query and retrieves the most similar **atomic knowledge items** (semantic search).
2. **Gates on relevance** — if nothing clears the threshold, it returns "no relevant notes
   found" and stops. It must never answer such a question from the model's own world
   knowledge. This is the read-side analogue of the `little_to_extract` honesty verdict.
3. If items clear the gate, it synthesizes an answer using **only** those items, grounding
   every claim in them, and returns the answer plus links to the source notes (resolvable to
   the entry file and the item's provenance timestamp).

A bare lookup just lists the ranked sources; a question additionally runs the grounded
synthesis. Both are one `ask` surface. Retrieval ranks by relevance × feedback score ×
recency, so highly rated and recent notes surface first. See `ARCHITECTURE.md` §9.

## 6. Non-functional requirements

- **Faithfulness:** zero fabricated knowledge items in the golden eval set; every item validates against provenance present in the transcript. On the read side, every answer claim must trace to a retrieved note, and a question with no supporting notes must abstain (no answer from outside knowledge).
- **Portability:** runs with `docker compose up` and a single API key; SQLite only, no external services.
- **Testability:** all deterministic logic (schema validation, routing, profile math, parsing) unit-tested; LLM behavior covered by a small fixture-based eval suite.
- **Privacy:** all data stays local; no telemetry; API key read from env only.
- **Cost control:** one transcript = bounded number of LLM calls (target ≤ 4); configurable model.

## 7. Success metrics

- ≥ 80% of extracted items judged faithful on the eval set (no hallucinated content).
- Median per-result score trends upward over the first ~30 documents for a given user (personalization works).
- "Little to extract" verdict fires correctly on the low-value fixtures (no fabricated insights).
- On the query eval set: questions answerable from the KB are answered with correct source links; questions with no supporting notes abstain 100% of the time (zero answers from outside knowledge).
- A fresh clone reaches a working `distil run <transcript>` in < 10 minutes following the README.

## 8. Release scope

- **MVP (v0):** FR1–FR6, FR9–FR12 via CLI; full TDD; Dockerized; GitHub repo with CI.
- **v0.1:** FR7, FR8 (novelty + cross-linking) and FR15–FR20 (the query/read layer — `ask` + `reindex`, grounded answers with abstention and source links). The read layer is high priority: it's what makes an accumulating KB worth having.
- **v0.2:** FR13 (web UI), FR14 (auth gate) and hosted deployment on Railway (volume-backed storage, `$PORT` binding). Auth ships *with* hosting, not after.

## 9. Open questions (owner to decide; defaults in `ARCHITECTURE.md`)

- Score granularity: per-document vs per-application-link (default: per-document, with per-link optional).
- YouTube URL fetch: include in MVP or defer (default: defer, accept transcript text in MVP).
- Embeddings: local model (fully provider-independent retrieval) vs API embeddings (simplest). Default: pluggable `Embedder`, local model preferred for independence.
- Auth method for hosting: shared-secret/password vs auth proxy (default: built-in single-user secret).
