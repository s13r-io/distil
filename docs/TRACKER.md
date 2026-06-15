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
| 4.1 | Type-routed extractors (T-E1,T-E2) | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 4.2 | Quote discipline <15 words (T-E4)  | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 4.3 | Faithfulness eval (T-E3)           | todo   | ⬜ | ⬜ | ⬜ | agent | headline guarantee |

## Phase 5 — Normalize
| ID  | Task                          | Status | T? | C? | P? | Owner | Notes |
|-----|-------------------------------|--------|----|----|----|-------|-------|
| 5.1 | Dedup / drop / stance (T-N*)  | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 6 — Link to profile
| ID  | Task                          | Status | T? | C? | P? | Owner | Notes |
|-----|-------------------------------|--------|----|----|----|-------|-------|
| 6.1 | Goal-tied links (T-L1)        | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 6.2 | Novelty reservation (T-L2)    | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 6.3 | Cold-start behavior (T-L3)    | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 7 — Graph linking (v0.1)
| ID  | Task                            | Status | T? | C? | P? | Owner | Notes |
|-----|---------------------------------|--------|----|----|----|-------|-------|
| 7.1 | Candidate lookup (T-G1)         | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 7.2 | Relation classify (T-G2)        | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 8 — Pipeline
| ID  | Task                          | Status | T? | C? | P? | Owner | Notes |
|-----|-------------------------------|--------|----|----|----|-------|-------|
| 8.1 | Orchestrate 1→6 (T-PL1,T-PL2) | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 9 — CLI
| ID  | Task                        | Status | T? | C? | P? | Owner | Notes |
|-----|-----------------------------|--------|----|----|----|-------|-------|
| 9.1 | `distil run` (T-C1)         | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 9.2 | `distil score` (T-C2)       | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 9.3 | list/show + errors (T-C3)   | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 10 — Querying the knowledge base / read layer
| ID   | Task                                             | Status | T? | C? | P? | Owner | Notes |
|------|--------------------------------------------------|--------|----|----|----|-------|-------|
| 10.1 | Embedder protocol + Fake + real (T-X3)           | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 10.2 | sqlite-vec store; embed items at file (T-X1)     | todo   | ⬜ | ⬜ | ⬜ | agent | pin version |
| 10.3 | `distil reindex` backfill (T-X2,T-C5)            | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 10.4 | Retrieval + ranking (T-Q1)                       | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 10.5 | Relevance gate / abstention (T-Q2)               | todo   | ⬜ | ⬜ | ⬜ | agent | no-hallucination |
| 10.6 | Grounded synthesis + sources + conflict (T-Q3..Q6)| todo  | ⬜ | ⬜ | ⬜ | agent |       |
| 10.7 | `distil ask` + query eval (T-C4,T-Q7)            | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 11 — Packaging & deploy (local + Railway)
| ID   | Task                                       | Status | T? | C? | P? | Owner | Notes |
|------|--------------------------------------------|--------|----|----|----|-------|-------|
| 11.1 | Dockerfile + compose; bind 0.0.0.0:$PORT (T-A4) | todo | ⬜ | ⬜ | ⬜ | agent |       |
| 11.2 | Auth gate; fail closed if public (T-A1..A3)| todo   | ⬜ | ⬜ | ⬜ | agent | before public domain |
| 11.3 | Volume-backed storage (/data)              | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 11.4 | railway.toml + DEPLOY_RAILWAY.md           | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 11.5 | Git-remote backup of kb/                    | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 11.6 | README quickstart verified (incl. ask)     | todo   | ⬜ | ⬜ | ⬜ | both  |       |
| 11.7 | Tag v0.0.1 + release workflow              | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 12 — Web UI (v0.2)
| ID   | Task                                  | Status | T? | C? | P? | Owner | Notes |
|------|---------------------------------------|--------|----|----|----|-------|-------|
| 12.1 | FastAPI list/view/score + ask box; auth middleware (T-A2) | todo | ⬜ | ⬜ | ⬜ | agent |       |

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

## Agent notes (non-blocking observations)
- ENV: stack pins Python >=3.11 (ARCHITECTURE.md §1) and CI uses 3.11. The dev sandbox here runs 3.10, so `pip install -e .` is refused by `requires-python`; tests are run via `PYTHONPATH=.` instead. No stack change made — flagging only. If the owner wants the sandbox to do editable installs, lowering the floor to 3.10 would be a stack decision (raise in Decisions needed first).
