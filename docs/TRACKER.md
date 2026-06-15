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
| 0.1 | Repo, pyproject, package, gitignore   | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 0.2 | LLMClient + FakeClient + Anthropic    | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 0.3 | pytest harness + markers              | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 0.4 | GitHub Actions CI green               | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 1 — Data layer
| ID  | Task                       | Status | T? | C? | P? | Owner | Notes |
|-----|----------------------------|--------|----|----|----|-------|-------|
| 1.1 | Pydantic models (T-M1..M4) | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 1.2 | store.py SQLite+md (T-S*)  | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 2 — Profile update logic (pure)
| ID  | Task                                | Status | T? | C? | P? | Owner | Notes |
|-----|-------------------------------------|--------|----|----|----|-------|-------|
| 2.1 | profile_update.py (T-P1..P8)        | todo   | ⬜ | ⬜ | ⬜ | agent |       |

## Phase 3 — Triage
| ID  | Task                              | Status | T? | C? | P? | Owner | Notes |
|-----|-----------------------------------|--------|----|----|----|-------|-------|
| 3.1 | Triage prompt template            | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 3.2 | Parse + TriageResult (T-T1,T-T2)  | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 3.3 | little_to_extract short-circuit   | todo   | ⬜ | ⬜ | ⬜ | agent |       |
| 3.4 | Eval tests (T-T4,T-T5)            | todo   | ⬜ | ⬜ | ⬜ | agent |       |

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
| D1 | Score per-document or per-application-link?                         | per-document (per-link optional) |                |
| D2 | Include YouTube URL fetch in MVP, or transcript text only?          | text only in MVP               |                |
| D3 | Commit generated `kb/` to git, or keep local/gitignored?           | gitignored (git-remote backup) |                |
| D4 | LLM provider/model to default to?                                   | Claude API, model via env      |                |
| D5 | Embeddings: local model or API?                                     | local (provider-independent)   |                |
| D6 | Hosting target?                                                     | Railway (auth required)        |                |
| D7 | Auth method when hosted?                                            | built-in single-user secret    |                |

## Changelog (agent appends)
- _(empty — agent adds dated entries: "2026-06-15 0.1 done, CI green, abc1234")_
