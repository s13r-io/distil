# SCHEMA — Distil data model

Two persistent objects (`Profile`, and a growing set of `KBEntry`) plus the feedback logic
connecting them. Implemented as Pydantic models in `distil/models.py`. JSON-style with
`//` comments; `ts` = ISO-8601 timestamp.

---

## 1. Profile

One per user. Read by the **link** stage, written by the **feedback** stage.
Compartmentalized so it learns without becoming a swamp.

```jsonc
{
  "user_id": "string",

  // STABLE — user-stated, rarely changes. The cold-start anchor.
  "stable": {
    "role": "string",
    "domain": "string",
    "tools": ["string"],
    "long_term_goals": [ { "id": "g_01", "statement": "string", "created_at": "ts" } ]
  },

  // CURRENT FOCUS — active projects. MUST DECAY (dormant → archived via last_touched).
  "current_focus": [
    { "id": "f_01", "project": "string", "description": "string",
      "active_since": "ts", "last_touched": "ts",
      "status": "active | dormant | archived" }
  ],

  // AFFINITIES — LEARNED from positive feedback. Weighted, per-dimension.
  "affinities": {
    "topics":            { "kubernetes": 0.8 },
    "knowledge_types":   { "heuristic": 0.7 },
    "application_forms": { "trigger": 0.9 }
  },

  // NEGATIVES — LEARNED anti-preferences, reason-tagged.
  "negatives": {
    "topics":            { "crypto_trading": { "weight": 0.7, "reasons": { "wrong_for_me": 3 } } },
    "knowledge_types":   {},
    "application_forms": {}
  },

  // KNOWN — "already knew this" topics. Suppress basics; keep advanced/novel items flowing.
  "known_topics": ["rest_api_basics"],

  "meta": { "documents_processed": 0, "confidence": 0.0, "last_updated": "ts" }
}
```

`confidence` is low at cold start → link stage leans on `stable`/`current_focus` and weights
learned signals lightly until enough documents are processed.

---

## 2. KBEntry (a filed document)

One per processed transcript. Stored as markdown in `kb/<entry_id>.md` (front-matter holds
the structured fields; body is the human-readable rendering) and indexed in SQLite.

```jsonc
{
  "entry_id": "string",

  "source": { "url": "string|null", "title": "string", "channel": "string|null",
              "duration_sec": 0, "captured_at": "ts" },

  // TRIAGE — runs first. Routes everything; sets expectations.
  "triage": {
    "knowledge_types_present": [ { "type": "heuristic", "share": 0.6 } ],  // shares sum ~1.0
    "density": "low | medium | high",
    "transcript_loss": { "level": "low | medium | high", "evidence": ["as you can see here"] },
    "verdict": "rich | mixed | little_to_extract"
  },

  // KNOWLEDGE ITEMS — atomic, self-contained, rewritten in our words. Type-specific fields.
  "knowledge_items": [
    {
      "item_id": "k_01",
      "type": "heuristic | procedural | declarative | conceptual | experiential | opinion",
      "statement": "string",
      "rationale": "string|null",          // heuristic
      "scope": "string|null",              // heuristic: when it applies / doesn't
      "order_index": "int|null",           // procedural
      "preconditions": [], "gotchas": [],  // procedural
      "stance": "fact | opinion | personal_experience",
      "speaker_confidence": "low | medium | high",
      "provenance": {
        "quote": "string < 15 words",     // ALWAYS present — the primary anchor; must appear in the source
        "timestamp": "00:12:30 | null",    // only when the source had one (.srt / inline markers)
        "locator": "string|null"           // line/segment index — the fallback pointer for untimestamped sources
      }
    }
  ],

  // APPLICATION LINKS — the product. Each tied to a profile goal/focus (credit assignment).
  "application_links": [
    { "link_id": "a_01", "knowledge_item_ids": ["k_01"],
      "linked_goal_id": "g_01 | f_01",
      "application_form": "checklist | trigger | flashcard | experiment | reference",
      "scenario": "string",                // concrete to THIS user
      "novelty_flag": false }              // true = orthogonal serendipity link (anti-bubble)
  ],

  // DISTILLED NOTE — reader-facing synthesis grounded in verified knowledge item ids.
  // Optional for backward compatibility with entries filed before Note v1.
  "distilled_note": {
    "title": "string",
    "core_takeaway": { "text": "string", "item_ids": ["k_01"] },
    "key_points": [ { "text": "string", "item_ids": ["k_01"] } ],
    "why_it_matters": [ { "text": "string", "item_ids": ["k_01"] } ],
    "how_to_apply": [
      { "text": "string", "item_ids": ["k_01"], "application_link_ids": ["a_01"] }
    ],
    "caveats": [ { "text": "string", "item_ids": ["k_01"] } ],
    "review_questions": [ { "question": "string", "item_ids": ["k_01"] } ],
    "topics": ["function_design"],
    "generated_from": "llm | fallback"
  },

  // GRAPH EDGES — turns a folder into a knowledge base.
  "related_entries": [
    { "target": "entry_id|item_id",
      "relation": "supports | contradicts | same_principle | extends | prerequisite_of" }
  ],

  // FEATURE TAGS — duplicated for attribution across dimensions.
  "tags": { "topics": ["string"], "knowledge_types": ["string"], "application_forms": ["string"] },

  // FEEDBACK — filled after scoring. `reason` is what makes the score teachable.
  "feedback": {
    "score": "null | 1..5",                // 1 not useful … 5 damn useful
    "reason": "null | relevant | already_knew | bad_source | wrong_for_me | irrelevant_now",
    "per_link": [ { "link_id": "a_01", "score": 5 } ],   // optional finer signal
    "scored_at": "null|ts"
  },

  "meta": { "created_at": "ts", "model_version": "string" }
}
```

---

## 3. Feedback → Profile update logic (`profile_update.py`)

The crux. Same score teaches opposite lessons depending on `reason`. This is **pure logic**
and must be exhaustively unit-tested (one test per row, see `TESTING.md`).

| score | reason          | meaning                       | profile update                                                       |
|-------|-----------------|-------------------------------|----------------------------------------------------------------------|
| 4–5   | relevant        | landed                        | upweight `tags` against `linked_goal_id`: topics, types, forms       |
| 4–5   | (novelty link)  | orthogonal link paid off      | **add new affinity** — discovered an undeclared interest             |
| 1–2   | bad_source      | video was junk                | **update nothing about the user**; adjust source-quality model only  |
| 1–2   | already_knew    | good but known                | add topic to `known_topics`; suppress basics, keep advanced flowing  |
| 1–2   | wrong_for_me    | genuine mismatch              | upweight matching `negatives` dimension                              |
| 1–2   | irrelevant_now  | fine knowledge, wrong timing  | soft signal; likely `current_focus` mismatch, not a topic dislike    |
| 3     | any             | lukewarm                      | small/no update; don't over-fit single lukewarm events               |

Rules: update gradually (EMA on weights, configurable α); never swing a topic on one event;
keep `meta.confidence` rising with `documents_processed`; decay `current_focus` by `last_touched`.

---

## 4. Vector index (read layer)

Embeddings are not part of the KBEntry document — they live in a side table in the same
`distil.db` via `sqlite-vec`, keyed back to the item they came from. Conceptually:

```jsonc
// vec0 virtual table: item_vectors(embedding float[N])  — rowid ↔ item_index map
// companion table:
{
  "item_id": "k_01",          // FK to a KBEntry knowledge item
  "entry_id": "string",       // FK to its parent entry (for source links)
  "embedding_model": "string",// which Embedder/model produced it (for reindex consistency)
  "embedded_at": "ts"
}
```

Retrieval returns `item_id`s; each resolves to its `entry_id` and the item's `provenance`
(quote, plus timestamp or locator), which is what answer source-links point at. `reindex`
repopulates this table for entries filed before the read layer, or after an `embedding_model` change.
