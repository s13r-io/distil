"""Phase D — okf.py export/render for entities, mirroring the Concept OKF export shape
(export_concept/remove_concept/render_source_with_concepts), and the bundle validator (E9-E12).
"""

import pytest

from distil.ingest import Segment, Transcript
from distil.models import Entity, EntityClaim, EntityMember, KBEntry
from distil.okf import (
    export_entity,
    remove_entity,
    render_source_with_concepts,
    slug_for_entry,
)
from distil.okf_lint import lint
from distil.store import Store


def _entity_entry(entry_id: str, title: str, item_id: str, statement: str, quote: str) -> KBEntry:
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
                    "provenance": {"quote": quote},
                }
            ],
            "meta": {"created_at": "2026-06-15T00:00:00", "model_version": "test"},
        }
    )


@pytest.fixture
def entity_store(tmp_path):
    return Store(db_path=tmp_path / "d.db", kb_dir=tmp_path / "kb", okf_root=tmp_path / "okf")


def _react_entity() -> Entity:
    return Entity(
        entity_id="react",
        kind="tool",
        title="React",
        description="A JS UI library.",
        members=[
            EntityMember(entry_id="e_r1", item_id="k_01", quote="react uses a virtual dom", timestamp="0:01:04"),
        ],
        claims=[EntityClaim(text="React uses a virtual DOM.", item_ids=["k_01"])],
        created_at="2026-06-15T00:00:00",
        updated_at="2026-06-15T00:00:00",
    )


@pytest.mark.unit
def test_export_entity_writes_conformant_frontmatter_and_body(entity_store):
    e1 = _entity_entry("e_r1", "Intro to React", "k_01", "React uses a virtual DOM.", "react uses a virtual dom")
    entity_store.file_entry(e1)
    entity = _react_entity()

    export_entity(entity, entity_store, entity_store.okf_root)
    text = (entity_store.okf_root / "entities" / "react.md").read_text()

    assert "type: entity" in text
    assert "kind: tool" in text
    assert "React uses a virtual DOM. (intro-to-react, 0:01:04)." in text
    assert '[Intro to React](../sources/intro-to-react.md)' in text

    entities_index = (entity_store.okf_root / "entities" / "index.md").read_text()
    assert "[React](react.md)" in entities_index
    root_index = (entity_store.okf_root / "index.md").read_text()
    assert "entities/react.md" in root_index


@pytest.mark.unit
def test_export_entity_is_deterministic_on_reexport(entity_store):
    e1 = _entity_entry("e_r1", "Intro to React", "k_01", "s", "q")
    entity_store.file_entry(e1)
    entity = _react_entity()
    export_entity(entity, entity_store, entity_store.okf_root)
    first = (entity_store.okf_root / "entities" / "react.md").read_text()
    export_entity(entity, entity_store, entity_store.okf_root)
    second = (entity_store.okf_root / "entities" / "react.md").read_text()
    assert first == second


@pytest.mark.unit
def test_source_page_gains_entities_mentioned_backlink(entity_store):
    e1 = _entity_entry("e_r1", "Intro to React", "k_01", "React uses a virtual DOM.", "react uses a virtual dom")
    entity_store.file_entry(e1, transcript=Transcript(
        segments=[Segment(text="react uses a virtual dom", locator="seg:0", timestamp="0:01:04")]
    ))
    entity = _react_entity()
    entity_store.save_entity(entity)

    render_source_with_concepts(e1, entity_store, entity_store.okf_root)

    slug = slug_for_entry(e1, entity_store.okf_root)
    text = (entity_store.okf_root / "sources" / f"{slug}.md").read_text()
    assert "## Entities mentioned" in text
    assert "[React](../entities/react.md)" in text


@pytest.mark.unit
def test_source_page_without_entities_has_no_entities_section(entity_store):
    e1 = _entity_entry("e_r1", "Lonely Video", "k_01", "s", "q")
    entity_store.file_entry(e1, transcript=Transcript(
        segments=[Segment(text="q", locator="seg:0")]
    ))

    render_source_with_concepts(e1, entity_store, entity_store.okf_root)

    slug = slug_for_entry(e1, entity_store.okf_root)
    text = (entity_store.okf_root / "sources" / f"{slug}.md").read_text()
    assert "## Entities mentioned" not in text


@pytest.mark.unit
def test_remove_entity_deletes_page_and_regenerates_indexes(entity_store):
    e1 = _entity_entry("e_r1", "Intro to React", "k_01", "s", "q")
    entity_store.file_entry(e1)
    export_entity(_react_entity(), entity_store, entity_store.okf_root)
    assert (entity_store.okf_root / "entities" / "react.md").exists()

    remove_entity("react", entity_store.okf_root)

    assert not (entity_store.okf_root / "entities" / "react.md").exists()
    entities_index = (entity_store.okf_root / "entities" / "index.md").read_text()
    assert "react" not in entities_index
    root_index = (entity_store.okf_root / "index.md").read_text()
    assert "entities/react.md" not in root_index


@pytest.mark.unit
def test_full_entity_bundle_passes_lint(entity_store):
    e1 = _entity_entry("e_r1", "Intro to React", "k_01", "React uses a virtual DOM.", "react uses a virtual dom")
    entity_store.file_entry(e1, transcript=Transcript(
        segments=[Segment(text="react uses a virtual dom", locator="seg:0", timestamp="0:01:04")]
    ))
    entity = _react_entity()
    entity_store.save_entity(entity)
    export_entity(entity, entity_store, entity_store.okf_root)
    render_source_with_concepts(e1, entity_store, entity_store.okf_root)

    assert lint(entity_store.okf_root) == []


@pytest.mark.unit
def test_lint_flags_orphan_entity_page(entity_store):
    """An entity page with no live source backlink fails E12 — mirrors E8 for concepts."""
    e1 = _entity_entry("e_r1", "Intro to React", "k_01", "s", "q")
    entity_store.file_entry(e1)  # no transcript -> no source page rendered with backlinks yet
    export_entity(_react_entity(), entity_store, entity_store.okf_root)

    errors = lint(entity_store.okf_root)
    assert any(e.startswith("E12") for e in errors)
