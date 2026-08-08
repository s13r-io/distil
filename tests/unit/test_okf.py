"""Phase 2 — okf.py: per-video OKF export layer (sources/ + raw/ + indexes)."""

import pytest

from distil.ingest import Segment, Transcript
from distil.models import KBEntry
from distil.okf import export_entry, remove_entry, slug_for_entry


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
