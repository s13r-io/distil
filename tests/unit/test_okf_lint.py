"""Phase 2/15.3 — okf_lint.py: deterministic OKF bundle conformance checker (stdlib only)."""

import pytest

from distil.ingest import Segment, Transcript
from distil.models import Concept, ConceptMember, KBEntry
from distil.okf import export_concept, export_entry, render_source_with_concepts, slug_for_entry
from distil.okf_lint import lint, main
from distil.store import Store


def _entry(entry_id: str = "e_01", title: str = "A Valid Bundle") -> KBEntry:
    return KBEntry.model_validate(
        {
            "entry_id": entry_id,
            "source": {"title": title, "captured_at": "2026-06-15T00:00:00"},
            "triage": {
                "knowledge_types_present": [{"type": "heuristic", "share": 1.0}],
                "density": "high",
                "transcript_loss": {"level": "low", "evidence": []},
                "verdict": "rich",
            },
            "knowledge_items": [],
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


def _transcript() -> Transcript:
    return Transcript(segments=[Segment(text="hello", locator="seg:0", timestamp="00:00:00")])


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
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


def _build_clean_concept_bundle(tmp_path):
    """One entry, one concept, fully wired via ``export_entry`` + ``export_concept`` +
    ``render_source_with_concepts`` — the same shape ``canonicalize.run_canonicalize_stage``
    produces in the real pipeline (T-OKFL8's "generated bundle lints clean" baseline)."""
    store = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb", okf_root=tmp_path / "okf")
    entry = _concept_entry(
        "e_r1", "Why AI Abandoned RAG", "k_01",
        "Traditional RAG retrieves then generates.", "retrieve then generate", "0:01:04",
    )
    store.file_entry(entry, transcript=_transcript())
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
    store.save_concept(concept)
    export_concept(concept, store, store.okf_root)
    render_source_with_concepts(entry, store, store.okf_root)
    return store, entry, concept


@pytest.mark.unit
def test_lint_passes_on_a_freshly_generated_bundle(tmp_path):
    export_entry(_entry(), _transcript(), tmp_path)
    assert lint(tmp_path) == []


@pytest.mark.unit
def test_main_returns_zero_on_a_clean_bundle(tmp_path):
    export_entry(_entry(), _transcript(), tmp_path)
    assert main([str(tmp_path)]) == 0


@pytest.mark.unit
def test_e1_flags_missing_type_frontmatter(tmp_path):
    export_entry(_entry(), _transcript(), tmp_path)
    stray = tmp_path / "sources" / "no-type.md"
    stray.write_text("# No frontmatter here\n", encoding="utf-8")
    errors = lint(tmp_path)
    assert any("E1" in e and "no-type.md" in e for e in errors)


@pytest.mark.unit
def test_e2_flags_broken_relative_link(tmp_path):
    export_entry(_entry(), _transcript(), tmp_path)
    source_page = tmp_path / "sources" / "a-valid-bundle.md"
    text = source_page.read_text()
    source_page.write_text(text + "\n[Nowhere](../raw/does-not-exist.md)\n", encoding="utf-8")
    errors = lint(tmp_path)
    assert any("E2" in e and "does-not-exist.md" in e for e in errors)


@pytest.mark.unit
def test_e3_flags_source_page_missing_from_sources_index(tmp_path):
    export_entry(_entry(), _transcript(), tmp_path)
    (tmp_path / "sources" / "index.md").write_text("# Sources\n", encoding="utf-8")
    errors = lint(tmp_path)
    assert any("E3" in e and "a-valid-bundle.md" in e for e in errors)


@pytest.mark.unit
def test_e4_flags_source_without_matching_raw(tmp_path):
    export_entry(_entry(), _transcript(), tmp_path)
    (tmp_path / "raw" / "a-valid-bundle.md").unlink()
    errors = lint(tmp_path)
    assert any("E4" in e and "sources/a-valid-bundle.md" in e for e in errors)


@pytest.mark.unit
def test_e4_flags_raw_without_matching_source(tmp_path):
    export_entry(_entry(), _transcript(), tmp_path)
    (tmp_path / "sources" / "a-valid-bundle.md").unlink()
    errors = lint(tmp_path)
    assert any("E4" in e and "raw/a-valid-bundle.md" in e for e in errors)


@pytest.mark.unit
def test_main_exits_nonzero_when_errors_present(tmp_path):
    export_entry(_entry(), _transcript(), tmp_path)
    (tmp_path / "raw" / "a-valid-bundle.md").unlink()
    assert main([str(tmp_path)]) == 1


# ---- T-OKFL5 (E5): concept frontmatter + index coverage --------------------------------------


@pytest.mark.unit
def test_okfl5_e5_flags_concept_missing_from_index_then_clean_bundle_passes(tmp_path):
    store, _entry_obj, _concept = _build_clean_concept_bundle(tmp_path)
    assert lint(store.okf_root) == []  # clean bundle with concepts lints clean

    (store.okf_root / "concepts" / "index.md").write_text("# Concepts\n", encoding="utf-8")
    errors = lint(store.okf_root)
    assert any("E5" in e and "traditional-rag.md" in e for e in errors)


# ---- T-OKFL6 (E6): concept<->source citation integrity ---------------------------------------


@pytest.mark.unit
def test_okfl6_e6_flags_sources_citation_not_in_videos_frontmatter(tmp_path):
    store, _entry_obj, _concept = _build_clean_concept_bundle(tmp_path)
    # A second, real source page that is NOT one of the concept's `videos:` slugs.
    other = _concept_entry("e_r2", "Unrelated Video", "k_09", "Some other statement.", "q9")
    store.file_entry(other, transcript=_transcript())
    other_slug = slug_for_entry(other, store.okf_root)

    concept_page = store.okf_root / "concepts" / "traditional-rag.md"
    text = concept_page.read_text()
    text = text.replace(
        "## Sources\n\n",
        f'## Sources\n\n- [Unrelated Video](../sources/{other_slug}.md) - "q9"\n',
        1,
    )
    concept_page.write_text(text, encoding="utf-8")

    errors = lint(store.okf_root)
    assert any("E6" in e and other_slug in e for e in errors)


# ---- T-OKFL7 (E7): bidirectional concept<->source links --------------------------------------


@pytest.mark.unit
def test_okfl7_e7_flags_one_way_link_then_clears_when_backlink_added(tmp_path):
    store = Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb", okf_root=tmp_path / "okf")
    entry = _concept_entry("e_r1", "Why AI Abandoned RAG", "k_01", "s", "q", "0:01:04")
    store.file_entry(entry, transcript=_transcript())
    concept = Concept(
        concept_id="traditional-rag",
        title="Traditional RAG",
        description="d",
        members=[ConceptMember(entry_id="e_r1", item_id="k_01", quote="q", timestamp="0:01:04")],
        created_at="t",
        updated_at="t",
    )
    store.save_concept(concept)
    export_concept(concept, store, store.okf_root)
    # Deliberately skip render_source_with_concepts: the source page has no backlink yet.

    errors = lint(store.okf_root)
    assert any("E7" in e and "traditional-rag" in e for e in errors)

    render_source_with_concepts(entry, store, store.okf_root)
    errors_after = lint(store.okf_root)
    assert not any("E7" in e for e in errors_after)


# ---- T-OKFL8 (E8): no orphan concept pages ----------------------------------------------------


@pytest.mark.unit
def test_okfl8_e8_flags_hand_inserted_orphan_but_generated_bundle_is_clean(tmp_path):
    store, _entry_obj, _concept = _build_clean_concept_bundle(tmp_path)
    assert lint(store.okf_root) == []  # export_concept + export_entry together lint clean

    orphan = store.okf_root / "concepts" / "orphan-idea.md"
    orphan.write_text(
        "---\n"
        "type: concept\n"
        'title: "Orphan Idea"\n'
        'description: "d"\n'
        "tags: []\n"
        "videos: []\n"
        "created: 2026-06-15\n"
        "updated: 2026-06-15\n"
        "---\n\n"
        "# Orphan Idea\n\n"
        "d\n\n"
        "## Sources\n\n",
        encoding="utf-8",
    )
    index_path = store.okf_root / "concepts" / "index.md"
    index_path.write_text(
        index_path.read_text() + "- [Orphan Idea](orphan-idea.md) - d\n", encoding="utf-8"
    )

    errors = lint(store.okf_root)
    assert any("E8" in e and "orphan-idea.md" in e for e in errors)
