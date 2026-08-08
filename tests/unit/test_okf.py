"""Phase 2 — okf.py: per-video OKF export layer (sources/ + raw/ + indexes)."""

import pytest

from distil.ingest import Segment, Transcript
from distil.models import Concept, ConceptClaim, ConceptEdge, ConceptMember, KBEntry
from distil.okf import (
    export_concept,
    export_entry,
    remove_concept,
    remove_entry,
    render_source_with_concepts,
    slug_for_entry,
)
from distil.store import Store


def _entry(
    entry_id: str = "e_01",
    title: str = "Why Small Functions Win",
    *,
    url: str | None = "https://www.youtube.com/watch?v=abc12345678",
    with_note: bool = True,
    with_feedback: bool = False,
    with_application_link: bool = False,
    duration_sec: int = 893,
    topics: list[str] | None = None,
) -> KBEntry:
    data = {
        "entry_id": entry_id,
        "source": {
            "title": title,
            "url": url,
            "duration_sec": duration_sec,
            "captured_at": "2026-06-15T00:00:00",
        },
        "triage": {
            "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
            "density": "high",
            "transcript_loss": {"level": "low", "evidence": []},
            "verdict": "rich",
        },
        "knowledge_items": [
            {
                "item_id": "k_01",
                "type": "heuristic",
                "statement": "Keep functions small.",
                "stance": "opinion",
                "provenance": {"quote": "keep functions small", "timestamp": "00:12:30"},
            },
            {
                "item_id": "k_02",
                "type": "heuristic",
                "statement": "Name things clearly.",
                "stance": "opinion",
                "provenance": {"quote": "name things clearly"},  # no timestamp
            },
        ],
        "tags": {
            "topics": topics if topics is not None else ["python", "function_design"],
            "knowledge_types": ["heuristic"],
            "application_forms": [],
        },
        "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
    }
    if with_note:
        data["distilled_note"] = {
            "title": "Small functions win",
            "core_takeaway": {
                "text": "Small functions are easier to understand and change safely.",
                "item_ids": ["k_01"],
            },
        }
    if with_feedback:
        data["feedback"] = {"score": 5, "reason": "relevant"}
    if with_application_link:
        data["application_links"] = [
            {
                "link_id": "a_01",
                "knowledge_item_ids": ["k_01"],
                "linked_goal_id": "g_01",
                "application_form": "checklist",
                "scenario": "refactor auth",
            }
        ]
    return KBEntry.model_validate(data)


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(text="Rag is dead, or so they say.", locator="seg:0", timestamp="00:00:00"),
            Segment(text="But it depends on your data.", locator="seg:1", timestamp="00:00:15"),
            Segment(text="Untimestamped closing thought.", locator="seg:2", timestamp=None),
        ]
    )


# ---- slug derivation ----------------------------------------------------------------------


@pytest.mark.unit
def test_slug_is_derived_from_title():
    assert slug_for_entry(_entry(title="Why Small Functions Win")) == "why-small-functions-win"


@pytest.mark.unit
def test_slug_falls_back_to_entry_id_when_title_unusable():
    entry = _entry(entry_id="e_99", title="!!!")
    assert slug_for_entry(entry) == "e_99"


@pytest.mark.unit
def test_slug_is_stable_across_calls():
    entry = _entry()
    assert slug_for_entry(entry) == slug_for_entry(entry)


@pytest.mark.unit
def test_title_collision_between_distinct_entries_does_not_overwrite(tmp_path):
    e1 = _entry(entry_id="e_aaa111", title="Live Q&A")
    e2 = _entry(entry_id="e_bbb222", title="Live Q&A")

    export_entry(e1, _transcript(), tmp_path)
    export_entry(e2, _transcript(), tmp_path)

    slug1 = slug_for_entry(e1, tmp_path)
    slug2 = slug_for_entry(e2, tmp_path)
    assert slug1 != slug2
    assert (tmp_path / "sources" / f"{slug1}.md").exists()
    assert (tmp_path / "sources" / f"{slug2}.md").exists()
    assert (tmp_path / "raw" / f"{slug1}.md").exists()
    assert (tmp_path / "raw" / f"{slug2}.md").exists()

    text1 = (tmp_path / "sources" / f"{slug1}.md").read_text()
    text2 = (tmp_path / "sources" / f"{slug2}.md").read_text()
    assert "distil_entry_id: e_aaa111" in text1
    assert "distil_entry_id: e_bbb222" in text2

    # re-exporting either entry keeps its previously assigned slug, even though both still
    # collide on the same base (title-derived) slug.
    export_entry(e1, _transcript(), tmp_path)
    export_entry(e2, _transcript(), tmp_path)
    assert slug_for_entry(e1, tmp_path) == slug1
    assert slug_for_entry(e2, tmp_path) == slug2
    assert len(list((tmp_path / "sources").glob("*.md"))) == 3  # both entries + index.md


# ---- export_entry: sources/ page -----------------------------------------------------------


@pytest.mark.unit
def test_export_writes_source_page_with_yaml_frontmatter(tmp_path):
    entry = _entry()
    export_entry(entry, _transcript(), tmp_path)
    slug = slug_for_entry(entry)
    text = (tmp_path / "sources" / f"{slug}.md").read_text()

    assert text.startswith("---\n")
    assert "type: source\n" in text
    assert 'title: "Small functions win"' in text
    assert "youtube_id: abc12345678" in text
    assert "url: https://www.youtube.com/watch?v=abc12345678" in text
    assert f"slug: {slug}" in text
    assert "published: 2026-06-15" in text
    assert 'duration: "0:14:53"' in text
    assert f'raw: "../raw/{slug}.md"' in text
    assert "tags: [python, function_design]" in text
    assert "created: 2026-06-15" in text
    assert "updated: 2026-06-15" in text


@pytest.mark.unit
def test_source_page_body_has_thesis_key_moments_and_transcript_link(tmp_path):
    entry = _entry()
    export_entry(entry, _transcript(), tmp_path)
    slug = slug_for_entry(entry)
    text = (tmp_path / "sources" / f"{slug}.md").read_text()

    assert "# Small functions win" in text
    assert "Small functions are easier to understand and change safely." in text
    assert "## Key moments" in text
    assert "- **[00:12:30]** Keep functions small." in text
    # item without a timestamp is not listed as a "key moment"
    assert "Name things clearly." not in text
    assert "## Transcript" in text
    assert f"[Raw transcript](../raw/{slug}.md)" in text


@pytest.mark.unit
def test_source_page_falls_back_to_source_title_without_note(tmp_path):
    entry = _entry(with_note=False)
    export_entry(entry, _transcript(), tmp_path)
    slug = slug_for_entry(entry)
    text = (tmp_path / "sources" / f"{slug}.md").read_text()
    assert "# Why Small Functions Win" in text
    assert "Why Small Functions Win" in text.split("---\n", 2)[-1]


@pytest.mark.unit
def test_source_page_omits_concepts_and_entities_sections(tmp_path):
    entry = _entry()
    export_entry(entry, _transcript(), tmp_path)
    slug = slug_for_entry(entry)
    text = (tmp_path / "sources" / f"{slug}.md").read_text()
    assert "## Concepts covered" not in text
    assert "## Entities" not in text


@pytest.mark.unit
def test_source_page_has_no_youtube_fields_for_non_youtube_source(tmp_path):
    entry = _entry(url=None)
    export_entry(entry, _transcript(), tmp_path)
    slug = slug_for_entry(entry)
    text = (tmp_path / "sources" / f"{slug}.md").read_text()
    assert "youtube_id:" not in text
    assert "\nurl:" not in text


# ---- export_entry: raw/ page ----------------------------------------------------------------


@pytest.mark.unit
def test_export_writes_raw_transcript_page(tmp_path):
    entry = _entry()
    export_entry(entry, _transcript(), tmp_path)
    slug = slug_for_entry(entry)
    text = (tmp_path / "raw" / f"{slug}.md").read_text()

    assert "type: raw-transcript\n" in text
    assert "immutable: true" in text
    assert f"slug: {slug}" in text
    assert "fetched_at: 2026-06-15" in text
    assert "**[00:00:00]** Rag is dead, or so they say." in text
    assert "**[00:00:15]** But it depends on your data." in text
    # untimestamped segment falls back to its locator
    assert "**[seg:2]** Untimestamped closing thought." in text


# ---- indexes ---------------------------------------------------------------------------------


@pytest.mark.unit
def test_export_regenerates_root_and_sources_indexes(tmp_path):
    entry = _entry()
    export_entry(entry, _transcript(), tmp_path)
    slug = slug_for_entry(entry)

    root_index = (tmp_path / "index.md").read_text()
    assert 'okf_version: "0.1"' in root_index
    assert f"sources/{slug}.md" in root_index

    sources_index = (tmp_path / "sources" / "index.md").read_text()
    assert f"[Small functions win]({slug}.md)" in sources_index
    assert "Small functions are easier to understand and change safely." in sources_index


@pytest.mark.unit
def test_reexporting_same_entry_does_not_duplicate_or_orphan_files(tmp_path):
    entry = _entry()
    export_entry(entry, _transcript(), tmp_path)
    export_entry(entry, _transcript(), tmp_path)
    assert list((tmp_path / "sources").glob("*.md")) != []
    assert len(list((tmp_path / "sources").glob("*.md"))) == 2  # entry page + index.md
    assert len(list((tmp_path / "raw").glob("*.md"))) == 1  # no index.md for raw in this phase


@pytest.mark.unit
def test_indexes_list_multiple_sources(tmp_path):
    e1 = _entry(entry_id="e_01", title="First Video")
    e2 = _entry(entry_id="e_02", title="Second Video")
    export_entry(e1, _transcript(), tmp_path)
    export_entry(e2, _transcript(), tmp_path)
    sources_index = (tmp_path / "sources" / "index.md").read_text()
    assert "first-video.md" in sources_index
    assert "second-video.md" in sources_index


# ---- remove_entry -----------------------------------------------------------------------------


@pytest.mark.unit
def test_remove_entry_deletes_pages_and_refreshes_indexes(tmp_path):
    e1 = _entry(entry_id="e_01", title="First Video")
    e2 = _entry(entry_id="e_02", title="Second Video")
    export_entry(e1, _transcript(), tmp_path)
    export_entry(e2, _transcript(), tmp_path)

    remove_entry(e1, tmp_path)

    assert not (tmp_path / "sources" / "first-video.md").exists()
    assert not (tmp_path / "raw" / "first-video.md").exists()
    assert (tmp_path / "sources" / "second-video.md").exists()
    sources_index = (tmp_path / "sources" / "index.md").read_text()
    assert "first-video.md" not in sources_index
    assert "second-video.md" in sources_index


@pytest.mark.unit
def test_remove_missing_entry_is_a_noop(tmp_path):
    entry = _entry()
    remove_entry(entry, tmp_path)  # never exported; must not raise
    assert not (tmp_path / "sources").exists() or list((tmp_path / "sources").glob("*.md")) == [
        tmp_path / "sources" / "index.md"
    ]


# ---- neutrality: no personal data leaks into OKF output --------------------------------------


@pytest.mark.unit
def test_source_page_has_no_feedback_or_application_links(tmp_path):
    entry = _entry(with_feedback=True, with_application_link=True)
    export_entry(entry, _transcript(), tmp_path)
    slug = slug_for_entry(entry)
    text = (tmp_path / "sources" / f"{slug}.md").read_text()
    assert "feedback" not in text.lower()
    assert "application_link" not in text.lower()
    assert "refactor auth" not in text
    assert "checklist" not in text


# ---- Phase 15.2 — export_concept / remove_concept / source backlink (T-OKFC1-4) --------------


def _concept_entry(
    entry_id: str, title: str, item_id: str, statement: str, quote: str, timestamp: str | None = None
) -> KBEntry:
    return KBEntry.model_validate(
        {
            "entry_id": entry_id,
            "source": {"title": title, "captured_at": "2026-06-15T00:00:00"},
            "triage": {
                "knowledge_types_present": [{"type": "conceptual", "share": 1.0}],
                "density": "high",
                "transcript_loss": {"level": "low", "evidence": []},
                "verdict": "rich",
            },
            "knowledge_items": [
                {
                    "item_id": item_id,
                    "type": "conceptual",
                    "statement": statement,
                    "stance": "fact",
                    "provenance": {"quote": quote, "timestamp": timestamp},
                }
            ],
            "tags": {
                "topics": ["rag", "llm"],
                "knowledge_types": ["conceptual"],
                "application_forms": [],
            },
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


@pytest.fixture
def concept_store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb", okf_root=tmp_path / "okf")


def _traditional_rag_concept() -> Concept:
    return Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="Retrieve documents, then generate an answer.",
        members=[
            ConceptMember(
                entry_id="e_r1", item_id="k_01", quote="retrieve then generate", timestamp="0:01:04"
            ),
            ConceptMember(entry_id="e_r2", item_id="k_02", quote="no planning step", timestamp="0:02:10"),
        ],
        claims=[
            ConceptClaim(
                text="Traditional RAG retrieves then generates with no planning loop.",
                item_ids=["k_01", "k_02"],
            )
        ],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )


@pytest.mark.unit
def test_okfc1_export_concept_writes_conformant_frontmatter_and_body(concept_store):
    e1 = _concept_entry(
        "e_r1", "Why AI Abandoned RAG", "k_01", "Traditional RAG retrieves then generates.",
        "retrieve then generate", "0:01:04",
    )
    e2 = _concept_entry(
        "e_r2", "Agentic RAG Explained", "k_02", "Naive RAG has no planning step.",
        "no planning step", "0:02:10",
    )
    concept_store.file_entry(e1)
    concept_store.file_entry(e2)
    concept = _traditional_rag_concept()

    export_concept(concept, concept_store, concept_store.okf_root)

    text = (concept_store.okf_root / "concepts" / "traditional-rag.md").read_text()
    assert text.startswith("---\n")
    assert "type: concept\n" in text
    assert 'title: "Traditional RAG"' in text
    assert 'description: "Retrieve documents, then generate an answer."' in text
    assert "videos: [agentic-rag-explained, why-ai-abandoned-rag]" in text
    assert "created: 2026-06-15" in text
    assert "updated: 2026-06-15" in text
    assert "feedback" not in text.lower()
    assert "application_link" not in text.lower()
    assert "# Traditional RAG" in text
    assert (
        "Traditional RAG retrieves then generates with no planning loop. "
        "(why-ai-abandoned-rag, 0:01:04, agentic-rag-explained, 0:02:10)."
    ) in text
    assert "## Sources" in text
    assert '[Why AI Abandoned RAG](../sources/why-ai-abandoned-rag.md) - "[0:01:04] retrieve then generate"' in text
    assert '[Agentic RAG Explained](../sources/agentic-rag-explained.md) - "[0:02:10] no planning step"' in text


@pytest.mark.unit
def test_okfc1_tags_are_union_of_member_entry_topics(concept_store):
    e1 = _concept_entry("e_r1", "V1", "k_01", "s1", "q1")
    concept_store.file_entry(e1)
    concept = Concept(
        concept_id="c1",
        title="C1",
        description="d",
        members=[ConceptMember(entry_id="e_r1", item_id="k_01", quote="q1")],
        created_at="t",
        updated_at="t",
    )
    export_concept(concept, concept_store, concept_store.okf_root)
    text = (concept_store.okf_root / "concepts" / "c1.md").read_text()
    assert "tags: [rag, llm]" in text


@pytest.mark.unit
def test_okfc2_reexporting_unchanged_concept_is_byte_identical(concept_store):
    e1 = _concept_entry(
        "e_r1", "Why AI Abandoned RAG", "k_01", "Traditional RAG retrieves then generates.",
        "retrieve then generate", "0:01:04",
    )
    concept_store.file_entry(e1)
    concept = Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="Retrieve documents, then generate an answer.",
        members=[
            ConceptMember(
                entry_id="e_r1", item_id="k_01", quote="retrieve then generate", timestamp="0:01:04"
            )
        ],
        claims=[
            ConceptClaim(text="Traditional RAG retrieves then generates.", item_ids=["k_01"])
        ],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )

    export_concept(concept, concept_store, concept_store.okf_root)
    first = (concept_store.okf_root / "concepts" / "traditional-rag.md").read_text()
    export_concept(concept, concept_store, concept_store.okf_root)
    second = (concept_store.okf_root / "concepts" / "traditional-rag.md").read_text()
    assert first == second


@pytest.mark.unit
def test_okfc3_source_page_gains_concepts_covered_backlink(concept_store):
    e1 = _concept_entry(
        "e_r1", "Why AI Abandoned RAG", "k_01", "Traditional RAG retrieves then generates.",
        "retrieve then generate", "0:01:04",
    )
    concept_store.file_entry(e1, transcript=_transcript())
    concept = Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="Retrieve documents, then generate an answer.",
        members=[
            ConceptMember(
                entry_id="e_r1", item_id="k_01", quote="retrieve then generate", timestamp="0:01:04"
            )
        ],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )
    concept_store.save_concept(concept)

    render_source_with_concepts(e1, concept_store, concept_store.okf_root)

    slug = slug_for_entry(e1, concept_store.okf_root)
    text = (concept_store.okf_root / "sources" / f"{slug}.md").read_text()
    assert "## Concepts covered" in text
    assert "[Traditional RAG](../concepts/traditional-rag.md)" in text

    # idempotent: re-rendering the same concept set is byte-identical.
    render_source_with_concepts(e1, concept_store, concept_store.okf_root)
    again = (concept_store.okf_root / "sources" / f"{slug}.md").read_text()
    assert text == again


@pytest.mark.unit
def test_okfc3_source_page_without_covering_concepts_has_no_backlink_section(concept_store):
    e1 = _concept_entry("e_r1", "Lonely Video", "k_01", "s", "q")
    concept_store.file_entry(e1, transcript=_transcript())

    render_source_with_concepts(e1, concept_store, concept_store.okf_root)

    slug = slug_for_entry(e1, concept_store.okf_root)
    text = (concept_store.okf_root / "sources" / f"{slug}.md").read_text()
    assert "## Concepts covered" not in text


@pytest.mark.unit
def test_okfc4_remove_concept_deletes_page_and_regenerates_indexes(concept_store):
    e1 = _concept_entry("e_r1", "Why AI Abandoned RAG", "k_01", "s", "q", "0:01:04")
    e2 = _concept_entry("e_r2", "Agentic RAG Explained", "k_02", "s2", "q2", "0:02:10")
    concept_store.file_entry(e1)
    concept_store.file_entry(e2)
    concept = _traditional_rag_concept()
    export_concept(concept, concept_store, concept_store.okf_root)
    assert (concept_store.okf_root / "concepts" / "traditional-rag.md").exists()

    remove_concept("traditional-rag", concept_store.okf_root)

    assert not (concept_store.okf_root / "concepts" / "traditional-rag.md").exists()
    concepts_index = (concept_store.okf_root / "concepts" / "index.md").read_text()
    assert "traditional-rag" not in concepts_index
    root_index = (concept_store.okf_root / "index.md").read_text()
    assert "concepts/traditional-rag.md" not in root_index


# ---- Phase 16 — typed-edge sections + contradiction flag (design report §9 item 4) -----------


@pytest.mark.unit
def test_concept_page_renders_typed_edge_sections(concept_store):
    e1 = _concept_entry("e_r1", "Why AI Abandoned RAG", "k_01", "s1", "q1")
    concept_store.file_entry(e1)
    target = Concept(
        concept_id="agentic-rag",
        title="Agentic RAG",
        description="RAG with a planning loop.",
        members=[ConceptMember(entry_id="e_r1", item_id="k_01", quote="q1")],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )
    concept_store.save_concept(target)
    export_concept(target, concept_store, concept_store.okf_root)

    concept = _traditional_rag_concept()
    concept.edges = [ConceptEdge(target_concept_id="agentic-rag", relation="contrasts_with")]
    e2 = _concept_entry("e_r2", "Agentic RAG Explained", "k_02", "s2", "q2")
    concept_store.file_entry(e2)

    export_concept(concept, concept_store, concept_store.okf_root)

    text = (concept_store.okf_root / "concepts" / "traditional-rag.md").read_text()
    assert "## Contrasts with" in text
    assert "[Agentic RAG](agentic-rag.md)" in text
    assert "## Builds on" not in text
    assert "## Related" not in text


@pytest.mark.unit
def test_concept_page_omits_edge_sections_when_no_edges(concept_store):
    e1 = _concept_entry("e_r1", "Why AI Abandoned RAG", "k_01", "s1", "q1")
    e2 = _concept_entry("e_r2", "Agentic RAG Explained", "k_02", "s2", "q2")
    concept_store.file_entry(e1)
    concept_store.file_entry(e2)
    concept = _traditional_rag_concept()

    export_concept(concept, concept_store, concept_store.okf_root)

    text = (concept_store.okf_root / "concepts" / "traditional-rag.md").read_text()
    assert "## Contrasts with" not in text
    assert "## Builds on" not in text
    assert "## Related" not in text


@pytest.mark.unit
def test_concept_page_flags_stance_contradiction_under_claims(concept_store):
    e1 = _concept_entry(
        "e_r1", "Why AI Abandoned RAG", "k_01", "Traditional RAG retrieves then generates.",
        "retrieve then generate", "0:01:04",
    )
    e2 = _concept_entry(
        "e_r2", "Agentic RAG Explained", "k_02", "Naive RAG has no planning step.",
        "no planning step", "0:02:10",
    )
    e1.knowledge_items[0].stance = "fact"
    e2.knowledge_items[0].stance = "opinion"
    concept_store.file_entry(e1)
    concept_store.file_entry(e2)
    concept = _traditional_rag_concept()

    export_concept(concept, concept_store, concept_store.okf_root)

    text = (concept_store.okf_root / "concepts" / "traditional-rag.md").read_text()
    assert "## Claims" in text
    assert "> **Contradiction:** members disagree" in text
    assert "why-ai-abandoned-rag (fact)" in text
    assert "agentic-rag-explained (opinion)" in text


@pytest.mark.unit
def test_okfc4_zero_member_concept_after_retraction_is_removed(concept_store):
    e1 = _concept_entry("e_r1", "Why AI Abandoned RAG", "k_01", "s", "q", "0:01:04")
    concept_store.file_entry(e1)
    concept = Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="d",
        members=[ConceptMember(entry_id="e_r1", item_id="k_01", quote="q", timestamp="0:01:04")],
        created_at="t",
        updated_at="t",
    )
    concept_store.save_concept(concept)
    export_concept(concept, concept_store, concept_store.okf_root)
    assert (concept_store.okf_root / "concepts" / "traditional-rag.md").exists()

    concept_store.retract_entry_concept_memberships("e_r1")
    assert concept_store.load_concept("traditional-rag") is None  # dropped from DB (Phase 15.1)

    remove_concept("traditional-rag", concept_store.okf_root)
    assert not (concept_store.okf_root / "concepts" / "traditional-rag.md").exists()


@pytest.mark.unit
def test_root_index_gains_concepts_section(concept_store):
    e1 = _concept_entry("e_r1", "Why AI Abandoned RAG", "k_01", "s", "q", "0:01:04")
    concept_store.file_entry(e1, transcript=_transcript())
    concept = Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="d",
        members=[ConceptMember(entry_id="e_r1", item_id="k_01", quote="q", timestamp="0:01:04")],
        created_at="t",
        updated_at="t",
    )
    export_concept(concept, concept_store, concept_store.okf_root)

    root_index = (concept_store.okf_root / "index.md").read_text()
    assert "## Concepts" in root_index
    assert "[Traditional RAG](concepts/traditional-rag.md)" in root_index

    concepts_index = (concept_store.okf_root / "concepts" / "index.md").read_text()
    assert "[Traditional RAG](traditional-rag.md)" in concepts_index
