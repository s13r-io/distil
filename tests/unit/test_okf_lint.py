"""Phase 2 — okf_lint.py: deterministic OKF bundle conformance checker (stdlib only)."""

import pytest

from distil.ingest import Segment, Transcript
from distil.models import KBEntry
from distil.okf import export_entry
from distil.okf_lint import lint, main


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
