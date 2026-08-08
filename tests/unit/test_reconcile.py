"""distil.reconcile — repairs an OKF bundle that has already drifted from the DB.

Conservative by construction: only removes a file whose owner can be positively determined to
be gone; anything undeterminable is left alone and reported. Dry run is the default and removes
nothing; --apply is required to actually delete. Never touches kb/ or the database.
"""

import json

import pytest

from distil.canonicalize import canonicalize_entry
from distil.embed import FakeEmbedder
from distil.ingest import Segment, Transcript
from distil.llm import FakeClient
from distil.models import KBEntry
from distil.okf_lint import lint
from distil.reconcile import reconcile_okf_bundle
from distil.store import Store

_AGENTIC_RAG = "Agentic RAG adds a planning loop before retrieval"


def _item(item_id, statement):
    return {
        "item_id": item_id,
        "type": "conceptual",
        "statement": statement,
        "stance": "fact",
        "provenance": {"quote": "q"},
    }


def _entry(entry_id, title) -> KBEntry:
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
            "knowledge_items": [_item("k_01", _AGENTIC_RAG)],
            "tags": {"topics": ["rag"], "knowledge_types": ["conceptual"], "application_forms": []},
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


def _transcript() -> Transcript:
    return Transcript(segments=[Segment(text="hello", locator="seg:0", timestamp="00:00:00")])


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb", okf_root=tmp_path / "okf")


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=32)


@pytest.mark.unit
def test_reconcile_removes_planted_orphan_source_and_raw(store, embedder):
    live = _entry("e_live", "Live Video")
    store.file_entry(live, embedder=embedder, transcript=_transcript())

    # Plant drift directly on disk: an entry that was deleted by hand, bypassing the cascade.
    orphan = _entry("e_gone", "Gone Video")
    from distil import okf as okf_mod

    okf_mod.export_entry(orphan, _transcript(), store.okf_root)

    report = reconcile_okf_bundle(store, apply=True)

    assert not (store.okf_root / "sources" / "gone-video.md").exists()
    assert not (store.okf_root / "raw" / "gone-video.md").exists()
    assert (store.okf_root / "sources" / "live-video.md").exists()
    assert (store.okf_root / "raw" / "live-video.md").exists()
    assert "sources/gone-video.md" in report.removed
    assert "raw/gone-video.md" in report.removed
    assert lint(store.okf_root) == []


@pytest.mark.unit
def test_reconcile_leaves_legitimate_file_alone(store, embedder):
    live = _entry("e_live", "Live Video")
    store.file_entry(live, embedder=embedder, transcript=_transcript())

    report = reconcile_okf_bundle(store, apply=True)

    assert report.removed == []
    assert (store.okf_root / "sources" / "live-video.md").exists()
    assert (store.okf_root / "raw" / "live-video.md").exists()


@pytest.mark.unit
def test_reconcile_removes_orphaned_concept_page(store, embedder):
    e1 = _entry("e_01", "Video One")
    store.file_entry(e1, embedder=embedder, transcript=_transcript())
    canonicalize_entry(
        e1,
        store,
        FakeClient(
            responses=[
                json.dumps(
                    [{"item_id": "k_01", "decision": "new", "title": "Agentic RAG", "description": "d"}]
                )
            ]
        ),
    )
    concept_id = store.list_concepts()[0].concept_id

    from distil import okf as okf_mod

    okf_mod.export_concept(store.load_concept(concept_id), store, store.okf_root)

    # Simulate the drift: the DB row is gone (as if deleted before the delete-cascade fix
    # existed) but the OKF page it left behind survives.
    store.delete_concept(concept_id)
    concept_page = store.okf_root / "concepts" / f"{concept_id}.md"
    assert concept_page.exists()

    report = reconcile_okf_bundle(store, apply=True)

    assert not concept_page.exists()
    assert f"concepts/{concept_id}.md" in report.removed


@pytest.mark.unit
def test_reconcile_dry_run_removes_nothing(store, embedder):
    live = _entry("e_live", "Live Video")
    store.file_entry(live, embedder=embedder, transcript=_transcript())
    orphan = _entry("e_gone", "Gone Video")
    from distil import okf as okf_mod

    okf_mod.export_entry(orphan, _transcript(), store.okf_root)

    report = reconcile_okf_bundle(store, apply=False)

    assert report.dry_run is True
    assert "sources/gone-video.md" in report.removed
    assert (store.okf_root / "sources" / "gone-video.md").exists()
    assert (store.okf_root / "raw" / "gone-video.md").exists()


@pytest.mark.unit
def test_reconcile_reports_undeterminable_source_instead_of_deleting(store, embedder):
    live = _entry("e_live", "Live Video")
    store.file_entry(live, embedder=embedder, transcript=_transcript())

    # A source page with no distil_entry_id frontmatter: ownership can't be determined, so
    # reconcile must never guess by deleting it.
    mystery = store.okf_root / "sources" / "mystery.md"
    mystery.write_text(
        '---\ntype: source\ntitle: "Mystery"\ndescription: "d"\n---\n\n# Mystery\n',
        encoding="utf-8",
    )

    report = reconcile_okf_bundle(store, apply=True)

    assert mystery.exists()
    assert "sources/mystery.md" in report.skipped
    assert "sources/mystery.md" not in report.removed


@pytest.mark.unit
def test_reconcile_reports_raw_with_no_matching_source_instead_of_deleting(store, embedder):
    live = _entry("e_live", "Live Video")
    store.file_entry(live, embedder=embedder, transcript=_transcript())

    orphan_raw = store.okf_root / "raw" / "no-source.md"
    orphan_raw.write_text("---\ntype: raw-transcript\n---\n\n## Transcript\n", encoding="utf-8")

    report = reconcile_okf_bundle(store, apply=True)

    assert orphan_raw.exists()
    assert "raw/no-source.md" in report.skipped


@pytest.mark.unit
def test_reconciled_bundle_passes_lint(store, embedder):
    live = _entry("e_live", "Live Video")
    store.file_entry(live, embedder=embedder, transcript=_transcript())
    canonicalize_entry(
        live,
        store,
        FakeClient(
            responses=[
                json.dumps(
                    [{"item_id": "k_01", "decision": "new", "title": "Agentic RAG", "description": "d"}]
                )
            ]
        ),
    )
    concept_id = store.list_concepts()[0].concept_id
    from distil import okf as okf_mod

    okf_mod.export_concept(store.load_concept(concept_id), store, store.okf_root)
    okf_mod.render_source_with_concepts(live, store, store.okf_root)

    orphan = _entry("e_gone", "Gone Video")
    okf_mod.export_entry(orphan, _transcript(), store.okf_root)
    store.delete_concept(concept_id)

    reconcile_okf_bundle(store, apply=True)

    assert lint(store.okf_root) == []
