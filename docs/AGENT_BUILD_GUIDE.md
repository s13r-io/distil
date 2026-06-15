# AGENT_BUILD_GUIDE — instructions for the building AI agent

You (the AI agent, e.g. Claude Cowork) are implementing Distil. Read `PRD.md`,
`ARCHITECTURE.md`, `SCHEMA.md`, and `TESTING.md` fully before writing any code. Work through
the phases below **in order**. This project is **test-driven**: for every task, write the
test first, see it fail, then implement.

---

## Rules of engagement

1. **TDD always.** Red → green → refactor. Never write a production function before its test exists. Update the tracker's `tests_written?` box before `code_done?`.
2. **One task at a time.** Pick the next unblocked task in `TRACKER.md`, set it `in_progress`, finish it (tests green), set it `done`, then commit. Do not batch many tasks into one commit.
3. **Commit per task.** Conventional commits: `feat(triage): add verdict short-circuit (T-T3)`. Reference the test IDs covered.
4. **Keep the tracker live.** It is the shared source of truth between you and the owner. Update status, check boxes, and add a one-line note on every task you touch. Never silently skip a task.
5. **Don't change scope or stack unilaterally.** If you believe a stack choice in `ARCHITECTURE.md` is wrong, add a row to the tracker's "Decisions needed" section and **pause that thread** — do not refactor the stack on your own.
6. **Stop and ask the owner** (leave a note in the tracker and halt that branch) when: a requirement is ambiguous, a test reveals a design contradiction, an external dependency needs an account/secret, or a faithfulness eval (T-E3) fails and you cannot fix it without weakening the guarantee.
7. **Never weaken the faithfulness guarantee** (T-E3 / T-N2) to make a test pass. Fabricated knowledge items are the one thing this product must not do. This extends to the read layer: **never bypass the retrieval gate** (T-Q2) or let an answer use sources outside the retrieved set (T-Q3). If retrieval finds nothing relevant, the system abstains — it does not answer from the model's outside knowledge.
8. **No secrets in code or commits.** Keys come from env. Add `.env`, `data/`, `kb/` (unless owner opts in) to `.gitignore`.
9. **Keep LLM calls bounded** (≤ 4 per transcript). If a feature needs more, raise it as a decision.

---

## Phase 0 — Scaffold & CI  *(blocks everything)*
- 0.1 Init repo, `pyproject.toml`, `distil/` package, `tests/` dirs, `.gitignore`, `.env.example`.
- 0.2 Add `LLMClient` protocol + `FakeClient` (canned responses) + `AnthropicClient` skeleton (reads env, no logic yet).
- 0.3 Set up pytest with `unit`/`eval` markers; write one trivial passing test to prove the harness.
- 0.4 GitHub Actions: run `pytest tests/unit` + lint on push. Confirm green.
- **Checkpoint:** CI is green on an empty-but-wired project. Commit, update tracker.

## Phase 1 — Data layer (TDD: T-M*, T-S*, T-I*)
- 1.1 Implement Pydantic models from `SCHEMA.md` (§1, §2) — tests T-M1..M4 first. `provenance.timestamp` is optional; `quote` is mandatory.
- 1.2 Implement `store.py`: SQLite index + markdown filing — tests T-S1..S3 first.
- 1.3 Implement `ingest.py` (stage 0, PURE): parse `.srt`/`.txt`/`.md`/pasted text into one normalized transcript; capture timestamps when present (SRT or inline `HH:MM:SS`), else `null` + a line/segment locator — tests T-I1..I6 first.
- **Checkpoint:** can construct/persist/reload a KBEntry and Profile, and normalize any supported input (timestamped or not) into a uniform transcript.

## Phase 2 — Profile update logic (TDD: T-P*)  *(pure, do early — it's the loop's brain)*
- 2.1 Implement `profile_update.py` from `SCHEMA.md` §3 — one test per row (T-P1..P8) first.
- **Checkpoint:** all feedback→profile rules proven in isolation, no LLM involved.

## Phase 3 — Triage (TDD: T-T*)
- 3.1 Write the triage prompt template. 3.2 Implement parsing + `TriageResult` model (T-T1, T-T2).
- 3.3 Implement the `little_to_extract` short-circuit (T-T3).
- 3.4 Add eval tests against fixtures (T-T4, T-T5) — gated by API key.
- **Checkpoint:** triage classifies fixtures correctly; low-value short-circuits.

## Phase 4 — Extraction (TDD: T-E*)
- 4.1 Type-routed extractors (heuristic/procedural/etc.) with type-specific fields (T-E1, T-E2).
- 4.2 Enforce quote discipline in code: provenance quote < 15 words (T-E4).
- 4.3 Faithfulness eval: every provenance quote must appear in the transcript (T-E3). **Headline test.**
- **Checkpoint:** extracted items are typed, provenanced, and faithful on the eval set.

## Phase 5 — Normalize (TDD: T-N*, pure)
- 5.1 Dedup/merge (T-N1); drop items with unverifiable provenance (T-N2); preserve stance (T-N3).
- **Checkpoint:** normalization is a clean, deterministic gate after extraction.

## Phase 6 — Link to profile (TDD: T-L*)
- 6.1 Generate application links tied to a real `linked_goal_id` (T-L1).
- 6.2 Novelty reservation per `DISTIL_NOVELTY_RATIO` (T-L2).
- 6.3 Cold-start behavior: lean on `stable` when confidence is low (T-L3).
- **Checkpoint:** application links are goal-tied and personalized; anti-bubble in place.

## Phase 7 — Graph linking (TDD: T-G*)  *(v0.1)*
- 7.1 Deterministic candidate lookup from the SQLite index (T-G1).
- 7.2 Relation classification into the allowed enum (T-G2).
- **Checkpoint:** new entries connect to existing ones.

## Phase 8 — Pipeline orchestration (TDD: T-PL*)
- 8.1 Wire stages 0→6 in `pipeline.py` (ingest → triage → … → file) (T-PL1); honor the short-circuit (T-PL2).
- **Checkpoint:** one call turns raw input (any supported format) + profile into a filed, schema-valid entry.

## Phase 9 — CLI (TDD: T-C*)
- 9.1 `distil run <file.srt|.txt|.md>` and `distil run --paste` / stdin for pasted text → ingest → pipeline (T-C1).
- 9.2 `distil score <entry_id> --score N --reason R` → calls profile_update (T-C2).
- 9.3 `distil list` / `distil show <id>` for browsing; friendly errors (T-C3).
- **Checkpoint:** full loop usable from the terminal.

## Phase 10 — Querying the knowledge base / read layer (TDD: T-X*, T-Q*)  *(high priority — build right after the CLI; it's what makes the KB worth keeping)*
- 10.1 `Embedder` protocol + `FakeEmbedder` + a real embedder (local model preferred for provider independence; API fallback) (T-X3).
- 10.2 Add `sqlite-vec` to `store.py`; embed each item at the **File** stage into a `vec0` table (T-X1). Pin the `sqlite-vec` version (pre-1.0).
- 10.3 `distil reindex` to backfill vectors for existing entries; idempotent (T-X2, T-C5).
- 10.4 Retrieval + ranking (similarity × score × recency) (T-Q1).
- 10.5 **The relevance gate** — below threshold → "no relevant notes", and make NO synthesis call (T-Q2). Build this before synthesis so the gate can never be bypassed.
- 10.6 Grounded synthesis: answer from retrieved items only; source links with provenance (T-Q3, T-Q4); bare lookup returns ranked sources (T-Q5); surface conflicts (T-Q6).
- 10.7 `distil ask "..."` CLI command (T-C4); eval pass on the query KB fixture (T-Q7).
- **Checkpoint:** you can ask questions, get grounded answers with source links, and get an honest "no notes" when the KB lacks the answer — verified by the abstention/grounding tests.

## Phase 11 — Packaging & deploy (local + Railway)
- 11.1 Dockerfile + docker-compose (volumes for `kb/`, `data/`); bind `0.0.0.0:$PORT` (T-A4).
- 11.2 Auth gate: enforce `DISTIL_AUTH_SECRET`; fail closed when `DISTIL_PUBLIC=true` without it (T-A1..A3). **Must land before any public domain is generated.**
- 11.3 Volume-backed storage: `DISTIL_DB_PATH`/`DISTIL_KB_DIR` → `/data`; document the runtime-only mount caveat.
- 11.4 `railway.toml` + `DEPLOY_RAILWAY.md` walkthrough.
- 11.5 Provider-independent backup: scheduled job that commits `kb/` to a private git remote.
- 11.6 Finalize README quickstart (incl. `ask`/`reindex`); verify a fresh clone reaches `distil run` in < 10 min.
- 11.7 Tag `v0.0.1`; release workflow.
- **Checkpoint:** anyone can clone and run locally; hosting on Railway works only with auth on.

## Phase 12 — Web UI (v0.2)
- 12.1 FastAPI app: list/view/score entries, browse KB graph, and an **ask box** over the read layer. Auth middleware in front of all data routes (T-A2). Add UI tests.

---

## Definition of done (every task)
Test written and failing first → implemented → unit tests green → tracker updated → committed
with test IDs referenced. A phase is done only when its checkpoint holds and CI is green.
