# TRACKER — Distil

Shared source of truth for owner (human) and agent (AI). **The agent updates this file as it
works**; the owner reviews it to see progress and to answer anything in "Decisions needed".

**Status values:** `todo` · `in_progress` · `blocked` · `review` · `done`
**Workflow per task:** test written (red) → implemented (green) → tests passing → `review` →
owner/agent marks `done`. Fill `Notes` with one line (commit hash, blocker, question).

Legend: T? = tests written · C? = code done · P? = tests passing · ✅/⬜

---

## Phase 0 — Scaffold & CI
| ID  | Task                                  | Status | T? | C? | P? | Owner | Notes |
|-----|---------------------------------------|--------|----|----|----|-------|-------|
| 0.1 | Repo, pyproject, package, gitignore   | done   | ✅ | ✅ | ✅ | agent | pyproject (py>=3.11), distil/ pkg, tests/{unit,eval,fixtures}, .gitignore (kb/,data/,.env) |
| 0.2 | LLMClient + FakeClient + Anthropic    | done   | ✅ | ✅ | ✅ | agent | Protocol + FakeClient (records calls for T-Q2) + AnthropicClient skeleton (env-driven, lazy key) |
| 0.3 | pytest harness + markers              | done   | ✅ | ✅ | ✅ | agent | unit/eval markers; conftest auto-skips eval w/o ANTHROPIC_API_KEY; smoke test green |
| 0.4 | GitHub Actions CI green               | done   | ✅ | ✅ | ✅ | agent | .github/workflows/ci.yml: ruff + pytest tests/unit on push/PR; eval job gated |

## Phase 1 — Data layer
| ID  | Task                       | Status | T? | C? | P? | Owner | Notes |
|-----|----------------------------|--------|----|----|----|-------|-------|
| 1.1 | Pydantic models (T-M1..M4) | done   | ✅ | ✅ | ✅ | agent | Profile+KBEntry+nested; closed enums (extra=forbid); quote mandatory, ts/locator optional; round-trip lossless. 13 tests |
| 1.2 | store.py SQLite+md (T-S*)  | done   | ✅ | ✅ | ✅ | agent | md front-matter=full JSON (lossless)+readable body; SQLite entries+profiles; upsert; candidate lookup; persists across instances. 8 tests |
| 1.3 | ingest.py .srt/.txt/.md/paste, ts optional (T-I*) | done | ✅ | ✅ | ✅ | agent | stage 0, pure; SRT+inline+paragraph parsers; uniform Segment{text,timestamp?,locator}; IngestError on empty/binary/missing. 9 tests |

## Phase 2 — Profile update logic (pure)
| ID  | Task                                | Status | T? | C? | P? | Owner | Notes |
|-----|-------------------------------------|--------|----|----|----|-------|-------|
| 2.1 | profile_update.py (T-P1..P8)        | done   | ✅ | ✅ | ✅ | agent | pure EMA; all 7 SCHEMA §3 rows; bad_source no-op on user; bounded weights; monotone confidence. 11 tests |

## Phase 3 — Triage
| ID  | Task                              | Status | T? | C? | P? | Owner | Notes |
|-----|-----------------------------------|--------|----|----|----|-------|-------|
| 3.1 | Triage prompt template            | done   | ✅ | ✅ | ✅ | agent | prompts/triage.py v1; honest verdict instruction; deictic evidence capture |
| 3.2 | Parse + TriageResult (T-T1,T-T2)  | done   | ✅ | ✅ | ✅ | agent | fence-tolerant JSON parse → Triage; ParseError on malformed/partial/bad-enum. 7 tests |
| 3.3 | little_to_extract short-circuit   | done   | ✅ | ✅ | ✅ | agent | is_low_value() signal; pipeline honors it in Phase 8 (T-PL2) |
| 3.4 | Eval tests (T-T4,T-T5)            | review | ✅ | ✅ | ⬜ | agent | written+gated; fixtures added (vlog/screen_share/rich/proc/mixed). NOT yet run — needs ANTHROPIC_API_KEY (owner) |

## Phase 4 — Extraction
| ID  | Task                               | Status | T? | C? | P? | Owner | Notes |
|-----|------------------------------------|--------|----|----|----|-------|-------|
| 4.1 | Type-routed extractors (T-E1,T-E2) | done   | ✅ | ✅ | ✅ | agent | routes by dominant triage type; type-specific fields; JSON-array parse→KnowledgeItem; ids assigned. 8 tests |
| 4.2 | Quote discipline <15 words (T-E4)  | done   | ✅ | ✅ | ✅ | agent | QuoteDisciplineError enforced in code (not model-dependent) |
| 4.3 | Faithfulness eval (T-E3)           | review | ✅ | ✅ | ⬜ | agent | HEADLINE. Deterministic faithfulness.quote_in_transcript() gate (6 unit tests). Eval T-E3 written+gated over 5 fixtures (ts+no-ts), UNRUN — needs API key (owner) |

## Phase 5 — Normalize
| ID  | Task                          | Status | T? | C? | P? | Owner | Notes |
|-----|-------------------------------|--------|----|----|----|-------|-------|
| 5.1 | Dedup / drop / stance (T-N*)  | done   | ✅ | ✅ | ✅ | agent | pure; drops unverifiable provenance (T-N2), backfills locator/ts (T-N4), merges near-dups (T-N1), stance untouched (T-N3). 7 tests |

## Phase 6 — Link to profile
| ID  | Task                          | Status | T? | C? | P? | Owner | Notes |
|-----|-------------------------------|--------|----|----|----|-------|-------|
| 6.1 | Goal-tied links (T-L1)        | done   | ✅ | ✅ | ✅ | agent | links with unknown linked_goal_id dropped; ids validated against profile goals+focus |
| 6.2 | Novelty reservation (T-L2)    | done   | ✅ | ✅ | ✅ | agent | deterministic ~ratio reservation; zero-ratio clears flags. 6 tests |
| 6.3 | Cold-start behavior (T-L3)    | done   | ✅ | ✅ | ✅ | agent | confidence<0.25 → prompt shows only stable goals+active focus, no affinities |

## Phase 7 — Graph linking (v0.1)
| ID  | Task                            | Status | T? | C? | P? | Owner | Notes |
|-----|---------------------------------|--------|----|----|----|-------|-------|
| 7.1 | Candidate lookup (T-G1)         | done   | ✅ | ✅ | ✅ | agent | deterministic topic-overlap lookup (type alone too broad); no candidates → no LLM call |
| 7.2 | Relation classify (T-G2)        | done   | ✅ | ✅ | ✅ | agent | LLM labels each candidate; only enum relations kept; 'none'/invalid dropped; capped at 3 candidates (LLM budget). 5 tests |

## Phase 7.5 — Note v1 reader-facing synthesis
| ID   | Task                                      | Status | T? | C? | P? | Owner | Notes |
|------|-------------------------------------------|--------|----|----|----|-------|-------|
| 7.5.1 | DistilledNote schema + note.py (T-DN*)    | done   | ✅ | ✅ | ✅ | agent | grounded teaching-note schema; cite-validating parser; topic normalization; deterministic fallback. 5 note tests |
| 7.5.2 | Pipeline/render/query/web integration     | done   | ✅ | ✅ | ✅ | agent | pipeline adds bounded note call; markdown/web show note before evidence; query vectors/prompts include cited note context; legacy entries load |

## Phase 8 — Pipeline
| ID  | Task                          | Status | T? | C? | P? | Owner | Notes |
|-----|-------------------------------|--------|----|----|----|-------|-------|
| 8.1 | Orchestrate 1→7 (T-PL1,T-PL3) | done   | ✅ | ✅ | ✅ | agent | ingest→triage→[short-circuit]→extract→normalize→link→note→graph→file; low-value files minimal w/ 1 LLM call; useful path budget≤4 w/o graph; note topics feed tags. 4 tests |

## Phase 9 — CLI
| ID  | Task                        | Status | T? | C? | P? | Owner | Notes |
|-----|-----------------------------|--------|----|----|----|-------|-------|
| 9.1 | `distil run` (T-C1)         | done   | ✅ | ✅ | ✅ | agent | file/--paste/stdin → pipeline; prints entry path; --no-graph flag |
| 9.2 | `distil score` (T-C2)       | done   | ✅ | ✅ | ✅ | agent | persists feedback on entry + applies profile update (EMA alpha from env) |
| 9.3 | list/show + errors (T-C3)   | done   | ✅ | ✅ | ✅ | agent | list/show; friendly errors for missing key, bad file, unknown entry (no tracebacks). 7 tests |

## Phase 10 — Querying the knowledge base / read layer
| ID   | Task                                             | Status | T? | C? | P? | Owner | Notes |
|------|--------------------------------------------------|--------|----|----|----|-------|-------|
| 10.1 | Embedder protocol + Fake + real (T-X3)           | done   | ✅ | ✅ | ✅ | agent | Embedder protocol; FakeEmbedder (hashed BoW, overlap→similarity); LocalEmbedder/ApiEmbedder lazy; make_embedder(). 6 tests |
| 10.2 | sqlite-vec store; embed items at file (T-X1)     | done   | ✅ | ✅ | ✅ | agent | sqlite-vec 0.1.6 loads in env; vectors stored as JSON in item_vectors_meta (portable both backends); embed at file w/ embedder. 7 tests |
| 10.3 | `distil reindex` backfill (T-X2,T-C5)            | done   | ✅ | ✅ | ✅ | agent | store.reindex() idempotent + re-embeds on model change; CLI `distil reindex` (T-C5). pipeline embeds at file stage |
| 10.4 | Retrieval + ranking (T-Q1)                       | done   | ✅ | ✅ | ✅ | agent | retrieve(): cosine × feedback_mult × recency_mult; sorted desc |
| 10.5 | Relevance gate / abstention (T-Q2)               | done   | ✅ | ✅ | ✅ | agent | HEADLINE. gate before any synthesis; below threshold → abstain w/ ZERO LLM calls (asserted) |
| 10.6 | Grounded synthesis + sources + conflict (T-Q3..Q6)| done  | ✅ | ✅ | ✅ | agent | answer cites only retrieved items; ungrounded cites stripped+reported; resolvable sources; lookup-only; contradicts-edge conflict surfaced |
| 10.7 | `distil ask` + query eval (T-C4,T-Q7)            | review | ✅ | ✅ | ⬜ | agent | CLI ask (answer/abstain/lookup) + reindex done (4 unit tests + e2e smoke). Eval T-Q7 written+gated, UNRUN — needs API key + local embedder (owner) |

## Phase 11 — Packaging & deploy (local + Railway)
| ID   | Task                                       | Status | T? | C? | P? | Owner | Notes |
|------|--------------------------------------------|--------|----|----|----|-------|-------|
| 11.1 | Dockerfile + compose; bind 0.0.0.0:$PORT (T-A4) | done | ✅ | ✅ | ✅ | agent | python:3.11-slim; model baked at build; CMD binds 0.0.0.0:${PORT:-8000}; compose bind-mounts data/+kb/. T-A4 unit-tested |
| 11.2 | Auth gate; fail closed if public (T-A1..A3)| done   | ✅ | ✅ | ✅ | agent | web/auth.py + middleware; 9 tests (T-A1..A4) green |
| 11.3 | Volume-backed storage (/data)              | done   | ✅ | ✅ | ✅ | agent | DISTIL_DB_PATH/KB_DIR → /data in image+compose; runtime-mount caveat documented |
| 11.4 | railway.toml + DEPLOY_RAILWAY.md           | done   | ✅ | ✅ | ✅ | agent | present from kickoff; verified consistent w/ web.app:app + /health + auth |
| 11.5 | Git-remote backup of kb/                    | done   | ✅ | ✅ | ✅ | agent | scripts/backup_kb.sh: idempotent commit+push of kb/ to a SEPARATE private remote |
| 11.6 | README quickstart verified (incl. ask)     | done   | ✅ | ✅ | ✅ | both  | all 6 CLI commands (run/score/list/show/ask/reindex) match README; --help verified |
| 11.7 | Tag v0.0.1 + release workflow              | review | ✅ | ✅ | ⬜ | agent | release.yml (test-gated build+release on v* tag). Tag push deferred to owner (no push from build env) |

## Phase 12 — Web UI (v0.2)
| ID   | Task                                  | Status | T? | C? | P? | Owner | Notes |
|------|---------------------------------------|--------|----|----|----|-------|-------|
| 12.1 | FastAPI list/view/score + ask box; auth middleware (T-A2) | done | ✅ | ✅ | ✅ | agent | web/app.py: index w/ ask box, /entries, /entries/{id}, score, /ask; auth middleware front of data routes. 9 web tests |

---

## Decisions needed (owner answers here)
| #  | Question                                                            | Default                        | Owner decision |
|----|---------------------------------------------------------------------|--------------------------------|----------------|
| D1 | Score per-document or per-application-link?                         | per-document (per-link optional) | **per-document** |
| D2 | Include YouTube URL fetch in MVP, or transcript text only?          | text only in MVP               | **text only; paste OR file upload (.srt/.txt/.md); handle transcripts with or without timestamps** (FR1, FR21, FR22) |
| D3 | Commit generated `kb/` to git, or keep local/gitignored?           | gitignored (git-remote backup) | **gitignored; back up via scheduled push to a separate private repo** |
| D4 | LLM provider/model to default to?                                   | Claude API, model via env      | **Claude API, model via env** |
| D5 | Embeddings: local model or API?                                     | local (provider-independent)   | **local** (watch instance RAM when hosted — ARCH §8.4) |
| D6 | Hosting target?                                                     | Railway (auth required)        | **Railway, auth required** |
| D7 | Auth method when hosted?                                            | built-in single-user secret    | **built-in single-user secret** |

## Changelog (agent appends)
- 2026-06-15 Phase 0 (0.1–0.4) done: scaffold, LLM boundary, pytest harness, CI workflow. 9 unit tests green, ruff clean. Checkpoint held (CI wired on empty-but-working project).
- 2026-06-15 Phase 1 (1.1–1.3) done: models, store, ingest. 39 unit tests green, ruff clean. Checkpoint held (construct/persist/reload KBEntry+Profile; uniform transcript across formats).
- 2026-06-15 Phase 2 (2.1) done: profile_update.py pure EMA logic, all SCHEMA §3 rows proven in isolation. 50 unit tests green. Checkpoint held (feedback→profile rules, no LLM). Also: ruff ignores UP017 to keep `timezone.utc` runnable on the 3.10 dev sandbox (3.11-valid too).
- 2026-06-15 Phase 3 (3.1–3.3) done; 3.4 in review: triage prompt+parse+short-circuit. 57 unit tests green. Eval tests T-T4/T-T5 written and gated but UNRUN (no API key in build env) — owner to run `pytest -m eval` with a key to confirm the checkpoint.
- 2026-06-15 Phase 4 (4.1,4.2) done; 4.3 in review: type-routed extraction + in-code quote discipline + deterministic faithfulness gate. 71 unit tests green. The headline faithfulness EVAL (T-E3) is written/gated/UNRUN — owner to run with a key. Guarantee NOT weakened: normalize (Phase 5) drops any item failing quote_in_transcript.
- 2026-06-15 Phase 5 (5.1) done: normalize.py pure gate (drop unverifiable, backfill provenance, merge dups, preserve stance). 78 unit tests green. Checkpoint held.
- 2026-06-15 Phase 6 (6.1–6.3) done: link.py goal-tied application links, deterministic novelty reservation, cold-start uses stable goals. 84 unit tests green. Checkpoint held.
- 2026-06-15 Phase 7 (7.1–7.2) done: graph.py deterministic topic-overlap candidate lookup + enum-bounded relation classify (capped 3 candidates). 89 unit tests green. Checkpoint held.
- 2026-06-15 Phase 8 (8.1) done: pipeline.py orchestrates 0→6; honors little_to_extract short-circuit; LLM budget bounded. 93 unit tests green. Checkpoint held. Follow-up: derive topic tags (currently empty) to strengthen graph candidacy.
- 2026-06-15 Phase 9 (9.1–9.3) done: Typer CLI run/score/list/show with friendly errors. 100 unit tests green + manual end-to-end smoke (run→list→show→score→profile updated). MVP loop usable from terminal. Checkpoint held.
- 2026-06-15 Phase 10 (read layer) done: Embedder, vector store, reindex, ranked retrieval, ABSTENTION gate (zero-LLM on miss), grounded synthesis w/ source links + conflict surfacing, CLI ask/reindex. 126 unit tests green + e2e ask smoke. Headline T-Q2/T-Q3 proven hermetically. Eval T-Q7 gated/unrun. Checkpoint held.
- 2026-06-15 Phase 11.2/11.3 + 12.1 done: auth gate (fail-closed when public w/o secret, 401 on data routes, localhost open), FastAPI app (list/view/score+ask box), $PORT bind. 135 unit tests green (T-A1..A4 + UI).
- 2026-06-15 Phase 11 complete: Dockerfile (model baked at build, binds 0.0.0.0:$PORT), docker-compose (volume-backed kb/+data/), .dockerignore, scripts/backup_kb.sh (git-remote KB backup), release.yml (test-gated). README verified against all 6 CLI commands. 11.7 tag deferred to owner. BUILD COMPLETE through v0.2 scope; only gated evals + the v0.0.1 tag push remain for the owner.
- 2026-06-15 VERIFICATION: independent subagent review confirmed both headline guarantees HOLD in code (faithfulness drop-gate in normalize.py:29; abstention gate unreachable-synthesis in query.py:111-118; grounding filter query.py:131-145). No bypass paths.
- 2026-06-21 Note v1 done: added DistilledNote + note.py grounded synthesis, bounded extra note call, fallback on malformed output, markdown/web teaching-note rendering, note topics as entry tags, and query retrieval context from cited note sections. 163 unit tests green.
- 2026-06-21 Source metadata UX done: CLI/web accept optional YouTube URLs, uploaded filenames are cleaned before display, Note v1 evidence is collapsed/de-emphasized in markdown and web, and index titles prefer synthesized note titles. 171 unit tests green.
- 2026-06-21 YouTube metadata + deletion done: YouTube URLs fetch best-effort oEmbed metadata (title/channel/channel URL/thumbnail/provider/fetched_at), notes render retained video metadata, and CLI/web deletion removes markdown, index row, and vectors. Unit tests/lint green.

## Agent notes (non-blocking observations)
- ENV: stack pins Python >=3.11 (ARCHITECTURE.md §1) and CI uses 3.11. The dev sandbox here runs 3.10, so `pip install -e .` is refused by `requires-python`; tests are run via `PYTHONPATH=.` instead. No stack change made — flagging only. If the owner wants the sandbox to do editable installs, lowering the floor to 3.10 would be a stack decision (raise in Decisions needed first).
