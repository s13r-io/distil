"""Phase 10.4-10.6 — retrieval, abstention gate, grounded synthesis.

Tests T-Q1..Q6. The headline guarantees:
* T-Q2 ABSTENTION: below-threshold question returns "no relevant notes" AND makes zero
  synthesis LLM calls.
* T-Q3 GROUNDING: an answered question cites only items from the retrieved set.
"""

import json

import pytest

from distil.embed import FakeEmbedder
from distil.llm import FakeClient
from distil.models import Concept, ConceptClaim, ConceptMember, KBEntry
from distil.query import ask, cosine, retrieve, retrieve_concepts, stream_ask
from distil.store import Store


def _entry(entry_id, items, *, score=None, related=None, with_note=False) -> KBEntry:
    data = {
        "entry_id": entry_id,
        "source": {"title": entry_id, "captured_at": "2026-06-15T00:00:00"},
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high", "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [
            {"item_id": iid, "type": "heuristic", "statement": stmt, "stance": "opinion",
             "provenance": {"quote": stmt[:20].lower(), "timestamp": ts}}
            for iid, stmt, ts in items
        ],
        "tags": {"topics": [], "knowledge_types": ["heuristic"], "application_forms": []},
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "t"},
    }
    if with_note:
        data["distilled_note"] = {
            "title": "Testing note",
            "core_takeaway": {
                "text": "Write tests first to clarify behavior before implementation.",
                "item_ids": [items[0][0]],
            },
            "key_points": [{
                "text": "The synthesized context should travel with retrieved evidence.",
                "item_ids": [items[0][0]],
            }],
            "topics": ["testing"],
        }
    if score is not None:
        data["feedback"] = {"score": score}
    if related:
        data["related_entries"] = related
    return KBEntry.model_validate(data)


@pytest.fixture
def store(tmp_path):
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    emb = FakeEmbedder(dim=64)
    s.file_entry(_entry("e_py", [
        ("k_py1", "Write unit tests before the implementation code.", "00:01:00"),
        ("k_py2", "Keep python functions small and focused.", None),
    ]), embedder=emb)
    s.file_entry(_entry("e_k8s", [
        ("k_k1", "Kubernetes pods share a network namespace.", "00:05:00"),
    ]), embedder=emb)
    return s


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=64)


def _concept(concept_id, title, description, members, claims=None) -> Concept:
    return Concept(
        concept_id=concept_id, title=title, description=description,
        members=members, claims=claims or [],
        created_at="2026-06-15T00:00:00", updated_at="2026-06-15T00:00:00",
    )


# ---- T-Q1: KNN ranked by similarity × feedback_score × recency ----


@pytest.mark.unit
def test_q1_retrieval_ranks_relevant_first(store, embedder):
    results = retrieve("python unit tests", store, embedder, top_k=3)
    assert results
    # the python testing note should outrank the kubernetes note
    ids = [r.item_id for r in results]
    assert ids[0] in {"k_py1", "k_py2"}
    assert results[0].score >= results[-1].score  # sorted descending


@pytest.mark.unit
def test_q1_feedback_score_boosts_ranking(tmp_path):
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    emb = FakeEmbedder(dim=64)
    # two entries with identical text; one is highly rated
    s.file_entry(_entry("e_low", [("k_low", "alpha beta gamma delta", None)], score=1), embedder=emb)
    s.file_entry(_entry("e_high", [("k_high", "alpha beta gamma delta", None)], score=5), embedder=emb)
    results = retrieve("alpha beta gamma delta", s, emb, top_k=2)
    assert results[0].item_id == "k_high"


# ---- T-Q2: ABSTENTION (headline) ----


@pytest.mark.unit
def test_q2_below_threshold_abstains_no_llm_call(store, embedder):
    client = FakeClient(responses=["SHOULD NOT BE CALLED"])
    result = ask("medieval french history and gothic cathedrals", store, embedder, client,
                 threshold=0.9)  # high threshold → nothing clears it
    assert result.abstained is True
    assert result.answer is None
    assert "no relevant notes" in result.message.lower()
    # THE GUARANTEE: synthesis LLM was never invoked
    assert client.call_count == 0


@pytest.mark.unit
def test_q2_threshold_zero_does_not_force_answer_on_empty_kb(tmp_path, embedder):
    empty = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    client = FakeClient(responses=["nope"])
    result = ask("anything", empty, embedder, client, threshold=0.0)
    assert result.abstained is True
    assert client.call_count == 0


# ---- T-Q3: GROUNDING (headline) ----


@pytest.mark.unit
def test_q3_answer_cites_only_retrieved_items(store, embedder):
    synth = json.dumps({
        "answer": "Write tests first [k_py1] and keep functions small [k_py2].",
        "cited_item_ids": ["k_py1", "k_py2"],
        "conflict": None,
    })
    client = FakeClient(responses=[synth])
    result = ask("how should I write python code", store, embedder, client, threshold=0.0, top_k=3)
    assert not result.abstained
    retrieved_ids = {s.item_id for s in result.sources}
    assert set(result.cited_item_ids) <= retrieved_ids  # no citation outside retrieved set
    assert result.answer == "Write tests first and keep functions small."


@pytest.mark.unit
def test_q3_citations_outside_retrieved_set_are_flagged(store, embedder):
    # model fabricates a citation to an item that wasn't retrieved
    synth = json.dumps({
        "answer": "Use microservices always [k_fake].",
        "cited_item_ids": ["k_fake"],
        "conflict": None,
    })
    client = FakeClient(responses=[synth])
    result = ask("architecture advice", store, embedder, client, threshold=0.0, top_k=3)
    # the ungrounded citation must not be presented as a valid source
    retrieved_ids = {s.item_id for s in result.sources}
    assert "k_fake" not in retrieved_ids
    assert result.ungrounded_citations == ["k_fake"]
    assert result.answer == "Use microservices always."


@pytest.mark.unit
def test_q3_retrieval_prompt_includes_distilled_note_context(tmp_path, embedder):
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    s.file_entry(_entry("e_note", [
        ("k_01", "Write unit tests before implementation code.", None),
    ], with_note=True), embedder=embedder)
    synth = json.dumps({
        "answer": "Write tests first [k_01].",
        "cited_item_ids": ["k_01"],
        "conflict": None,
    })
    client = FakeClient(responses=[synth])
    result = ask("testing", s, embedder, client, threshold=0.0, top_k=1)
    assert not result.abstained
    assert "synthesized note context" in client.calls[0].prompt
    assert "clarify behavior" in client.calls[0].prompt


# ---- T-Q4: every answer carries resolvable source links ----


@pytest.mark.unit
def test_q4_sources_resolve_to_entry_item_provenance(store, embedder):
    synth = json.dumps({
        "answer": "Write tests first [k_py1].", "cited_item_ids": ["k_py1"], "conflict": None,
    })
    client = FakeClient(responses=[synth])
    result = ask("testing", store, embedder, client, threshold=0.0, top_k=3)
    src = next(s for s in result.sources if s.item_id == "k_py1")
    assert src.entry_id == "e_py"
    assert src.entry_title == "e py"
    assert src.timestamp == "00:01:00"
    assert src.quote


# ---- T-Q5: bare lookup returns ranked sources, no synthesis call ----


@pytest.mark.unit
def test_q5_lookup_returns_sources_without_synthesis(store, embedder):
    client = FakeClient(responses=["SHOULD NOT BE CALLED"])
    result = ask("python tests", store, embedder, client, threshold=0.0, top_k=3, lookup_only=True)
    assert not result.abstained
    assert result.answer is None
    assert len(result.sources) > 0
    assert client.call_count == 0


# ---- T-Q6: conflict surfaced when retrieved items contradict ----


@pytest.mark.unit
def test_q6_conflict_surfaced(tmp_path, embedder):
    s = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb")
    emb = FakeEmbedder(dim=64)
    s.file_entry(_entry("e_a", [("k_a", "monoliths are better for small teams indeed", None)]),
                 embedder=emb)
    s.file_entry(_entry("e_b", [("k_b", "monoliths are worse for small teams indeed", None)],
                        related=[{"target": "e_a", "relation": "contradicts"}]), embedder=emb)
    synth = json.dumps({
        "answer": "Notes disagree on monoliths [k_a][k_b].",
        "cited_item_ids": ["k_a", "k_b"],
        "conflict": "k_a says better, k_b says worse.",
    })
    client = FakeClient(responses=[synth])
    result = ask("are monoliths good for small teams", s, emb, client, threshold=0.0, top_k=3)
    assert result.conflict


# ---- Phase 18 (T-Q9..T-Q13): query-over-concepts — a consumer of item retrieval, never a
# replacement for it. The relevance gate and code-assembled citations must both survive intact.


@pytest.mark.unit
def test_q9_retrieve_concepts_ranks_relevant_first(store, embedder):
    testing_concept = _concept(
        "python-testing", "Python testing discipline",
        "Write unit tests before implementation code.",
        members=[ConceptMember(entry_id="e_py", item_id="k_py1", quote="q", timestamp=None)],
    )
    k8s_concept = _concept(
        "kubernetes-networking", "Kubernetes networking",
        "Kubernetes pods share a network namespace.",
        members=[ConceptMember(entry_id="e_k8s", item_id="k_k1", quote="q", timestamp=None)],
    )
    store.save_concept(testing_concept)
    store.save_concept(k8s_concept)

    results = retrieve_concepts("unit tests before implementation", store, embedder, top_k=2)
    assert results[0].concept_id == "python-testing"
    assert results[0].similarity >= results[-1].similarity


@pytest.mark.unit
def test_q10_retrieve_concepts_blends_in_member_similarity(store, embedder):
    # A concept whose two members point in very different directions dilutes the centroid;
    # blending in per-member similarity should still surface it via its closest member.
    mixed = _concept(
        "mixed-topic", "Mixed topic", "covers both testing and kubernetes",
        members=[
            ConceptMember(entry_id="e_py", item_id="k_py1", quote="q", timestamp=None),
            ConceptMember(entry_id="e_k8s", item_id="k_k1", quote="q", timestamp=None),
        ],
    )
    store.save_concept(mixed)

    question = "unit tests before implementation"
    qvec = embedder.embed(question)
    centroid_sim = cosine(qvec, store.concept_centroid("mixed-topic"))

    [result] = retrieve_concepts(question, store, embedder, top_k=1)
    assert result.concept_id == "mixed-topic"
    # Blending picks up the near-identical member (k_py1), not just the diluted centroid.
    assert result.similarity > centroid_sim


@pytest.mark.unit
def test_q11_concept_only_evidence_avoids_abstention(store, embedder, monkeypatch):
    concept = _concept(
        "solo-concept", "Solo Concept", "desc",
        members=[ConceptMember(entry_id="e_k8s", item_id="k_k1", quote="q", timestamp=None)],
        claims=[ConceptClaim(
            text="Kubernetes networking is discussed across videos.", item_ids=["k_k1"],
        )],
    )
    store.save_concept(concept)

    # An unrelated phrasing: no raw item can clear a near-maximal threshold. Force this
    # concept's centroid to match the question exactly (cosine == 1.0) to simulate many loosely
    # related members whose combined topic centroid clears the gate on its own — isolating the
    # GATE behavior (T-Q10 already covers the natural member-blending path).
    question = "a completely unrelated phrasing sharing no vocabulary with any note"
    qvec = embedder.embed(question)
    real_centroid = store.concept_centroid
    monkeypatch.setattr(
        store, "concept_centroid",
        lambda concept_id: qvec if concept_id == "solo-concept" else real_centroid(concept_id),
    )

    client = FakeClient(responses=[json.dumps({
        "answer": "Kubernetes networking is discussed [k_k1].",
        "cited_item_ids": ["k_k1"], "conflict": None,
    })])
    result = ask(question, store, embedder, client, threshold=0.99, top_k=3)

    assert not result.abstained
    assert any(s.item_id == "k_k1" for s in result.sources)
    assert "k_k1" in result.cited_item_ids
    assert result.ungrounded_citations == []


@pytest.mark.unit
def test_q12_concept_answer_sources_are_code_assembled_from_real_provenance(
    store, embedder, monkeypatch
):
    # The ConceptMember's copied quote/timestamp deliberately mismatch the real item's
    # provenance below, so a passing assertion proves Source data was resolved fresh from the
    # live KBEntry (like a directly-retrieved item), never trusted from the concept's own copy
    # or from model-authored text.
    concept = _concept(
        "solo-concept-2", "Solo Concept 2", "desc",
        members=[ConceptMember(
            entry_id="e_k8s", item_id="k_k1", quote="stale-copy", timestamp="99:99:99",
        )],
    )
    store.save_concept(concept)

    question = "another unrelated phrase with no vocabulary overlap"
    qvec = embedder.embed(question)
    monkeypatch.setattr(store, "concept_centroid", lambda concept_id: qvec)

    result = ask(
        question, store, embedder, FakeClient(responses=["SHOULD NOT BE CALLED"]),
        threshold=0.99, top_k=3, lookup_only=True,
    )
    assert not result.abstained

    real_item = next(
        i for i in store.load_entry("e_k8s").knowledge_items if i.item_id == "k_k1"
    )
    src = next(s for s in result.sources if s.item_id == "k_k1")
    assert src.entry_id == "e_k8s"
    assert src.quote == real_item.provenance.quote
    assert src.timestamp == real_item.provenance.timestamp
    assert src.quote != "stale-copy"


@pytest.mark.unit
def test_q13_low_similarity_concept_does_not_bypass_gate(store, embedder):
    concept = _concept(
        "low-sim-concept", "Low Sim", "desc",
        members=[ConceptMember(entry_id="e_k8s", item_id="k_k1", quote="q", timestamp=None)],
    )
    store.save_concept(concept)

    client = FakeClient(responses=["SHOULD NOT BE CALLED"])
    result = ask(
        "medieval french history and gothic cathedrals", store, embedder, client, threshold=0.9,
    )
    assert result.abstained is True
    assert client.call_count == 0


# ---- Phase C: concept pages behind an answer + configurable synthesis depth -------------
# Both requirements ride the SAME gate/grounding code paths already covered above; these tests
# isolate what's new: concept refs surfacing for free, and the depth knob actually changing what
# reaches the synthesis prompt without touching the gate or citation validation.


def _cleared_solo_concept(store, embedder, question, *, member_quote="the member's own quote"):
    """A concept whose centroid is forced to match ``question`` exactly (T-Q11's pattern),
    isolating the gate from natural embedding behavior, with a distinguishable member quote so
    tests can assert on its presence/absence in the synthesis prompt."""
    concept = _concept(
        "solo-concept", "Solo Concept", "desc",
        members=[ConceptMember(
            entry_id="e_k8s", item_id="k_k1", quote=member_quote, timestamp="00:09:00",
        )],
        claims=[ConceptClaim(
            text="Kubernetes networking is discussed across videos.", item_ids=["k_k1"],
        )],
    )
    store.save_concept(concept)
    qvec = embedder.embed(question)
    store.concept_centroid = lambda concept_id: qvec  # type: ignore[method-assign]
    return concept


@pytest.mark.unit
def test_phaseC_ask_surfaces_concepts_behind_the_answer(store, embedder):
    question = "a completely unrelated phrasing sharing no vocabulary with any note"
    _cleared_solo_concept(store, embedder, question)
    client = FakeClient(responses=[json.dumps({
        "answer": "Kubernetes networking is discussed [k_k1].",
        "cited_item_ids": ["k_k1"], "conflict": None,
    })])

    result = ask(question, store, embedder, client, threshold=0.99, top_k=3)

    assert not result.abstained
    assert [c.concept_id for c in result.concepts] == ["solo-concept"]
    assert result.concepts[0].title == "Solo Concept"


@pytest.mark.unit
def test_phaseC_stream_ask_surfaces_concepts_behind_the_answer(store, embedder):
    question = "a completely unrelated phrasing sharing no vocabulary with any note"
    _cleared_solo_concept(store, embedder, question)
    client = FakeClient(responses=[json.dumps({
        "answer": "Kubernetes networking is discussed [k_k1].",
        "cited_item_ids": ["k_k1"], "conflict": None,
    })])

    events = list(stream_ask(question, store, embedder, client, threshold=0.99, top_k=3))
    final = next(e for e in events if e.kind == "final")

    assert [c.concept_id for c in final.result.concepts] == ["solo-concept"]


@pytest.mark.unit
def test_phaseC_abstained_result_carries_no_concepts(store, embedder):
    client = FakeClient(responses=["SHOULD NOT BE CALLED"])
    result = ask("medieval french history and gothic cathedrals", store, embedder, client,
                 threshold=0.9)
    assert result.abstained is True
    assert result.concepts == []


@pytest.mark.unit
def test_phaseC_default_depth_sends_claim_text_only_no_member_quotes(store, embedder):
    question = "a completely unrelated phrasing sharing no vocabulary with any note"
    _cleared_solo_concept(store, embedder, question, member_quote="THE-DISTINCT-MEMBER-QUOTE")
    client = FakeClient(responses=[json.dumps({
        "answer": "Kubernetes networking is discussed [k_k1].",
        "cited_item_ids": ["k_k1"], "conflict": None,
    })])

    ask(question, store, embedder, client, threshold=0.99, top_k=3)

    prompt = client.calls[0].prompt
    assert "Kubernetes networking is discussed across videos." in prompt  # claim text present
    assert "THE-DISTINCT-MEMBER-QUOTE" not in prompt  # not duplicated via the concept block


@pytest.mark.unit
def test_phaseC_full_depth_sends_whole_page_member_quotes(store, embedder):
    question = "a completely unrelated phrasing sharing no vocabulary with any note"
    _cleared_solo_concept(store, embedder, question, member_quote="THE-DISTINCT-MEMBER-QUOTE")
    client = FakeClient(responses=[json.dumps({
        "answer": "Kubernetes networking is discussed [k_k1].",
        "cited_item_ids": ["k_k1"], "conflict": None,
    })])

    ask(question, store, embedder, client, threshold=0.99, top_k=3, concept_note_depth="full")

    prompt = client.calls[0].prompt
    assert "Kubernetes networking is discussed across videos." in prompt
    assert "THE-DISTINCT-MEMBER-QUOTE" in prompt


@pytest.mark.unit
def test_phaseC_env_var_controls_depth_without_code_change(store, embedder, monkeypatch):
    question = "a completely unrelated phrasing sharing no vocabulary with any note"
    _cleared_solo_concept(store, embedder, question, member_quote="THE-DISTINCT-MEMBER-QUOTE")
    monkeypatch.setenv("DISTIL_CONCEPT_NOTE_DEPTH", "full")
    client = FakeClient(responses=[json.dumps({
        "answer": "Kubernetes networking is discussed [k_k1].",
        "cited_item_ids": ["k_k1"], "conflict": None,
    })])

    ask(question, store, embedder, client, threshold=0.99, top_k=3)  # no explicit override

    assert "THE-DISTINCT-MEMBER-QUOTE" in client.calls[0].prompt


@pytest.mark.unit
def test_phaseC_full_depth_does_not_weaken_citation_validation(store, embedder):
    # Same fabricated-citation setup as T-Q3's ungrounded-citation test, but with depth="full" —
    # proves the richer evidence block doesn't loosen what counts as a grounded citation.
    question = "a completely unrelated phrasing sharing no vocabulary with any note"
    _cleared_solo_concept(store, embedder, question)
    client = FakeClient(responses=[json.dumps({
        "answer": "Use microservices always [k_fake].",
        "cited_item_ids": ["k_fake"], "conflict": None,
    })])

    result = ask(
        question, store, embedder, client, threshold=0.99, top_k=3, concept_note_depth="full",
    )

    retrieved_ids = {s.item_id for s in result.sources}
    assert "k_fake" not in retrieved_ids
    assert result.ungrounded_citations == ["k_fake"]


@pytest.mark.unit
def test_phaseC_invalid_depth_falls_back_to_default(store, embedder):
    question = "a completely unrelated phrasing sharing no vocabulary with any note"
    _cleared_solo_concept(store, embedder, question, member_quote="THE-DISTINCT-MEMBER-QUOTE")
    client = FakeClient(responses=[json.dumps({
        "answer": "Kubernetes networking is discussed [k_k1].",
        "cited_item_ids": ["k_k1"], "conflict": None,
    })])

    ask(question, store, embedder, client, threshold=0.99, top_k=3, concept_note_depth="bogus")

    # Never raises; degrades to the documented default (claims-only) rather than guessing.
    assert "THE-DISTINCT-MEMBER-QUOTE" not in client.calls[0].prompt
