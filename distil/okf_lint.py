#!/usr/bin/env python
"""Deterministic conformance checker for a Distil OKF bundle (stdlib only). See ``okf.py``.

    python -m distil.okf_lint <okf_root>

Covers ``sources/`` + ``raw/`` (Phase 2), ``concepts/`` (Phase 15, OKF Phase 3a design report
§7), and ``entities/`` (Phase D — the E9-E12 checks, the same shape as concepts' E5-E8).

Checks (all ERROR; exit non-zero if any fire):
  E1  every non-reserved .md has YAML frontmatter with a non-empty `type`
  E2  every relative markdown link resolves to a file in the bundle
  E3  every sources/<slug>.md is listed in sources/index.md
  E4  sources/<slug>.md <-> raw/<slug>.md parity, both directions
  E5  every concepts/<slug>.md has `type: concept` frontmatter and is listed in
      concepts/index.md
  E6  concept<->source citation integrity: every slug in a concept's `videos:` frontmatter
      has a sources/<slug>.md, and every slug cited under the concept's `## Sources` section
      is one of the `videos:` slugs
  E7  bidirectional concept<->source links (SCHEMA §5): a concept's `## Sources` link to a
      source implies that source's `## Concepts covered` links back, and vice versa
  E8  no orphan concepts: every concepts/<slug>.md has >=1 inbound link from a source's
      `## Concepts covered` or another concept's typed-edge section (`## Contrasts with` /
      `## Builds on` / `## Related`, Phase 16) — the concepts/index.md listing does not count.
      sources/<slug>.md pages remain exempt — they're provenance leaves by design.
  E9  every entities/<slug>.md has `type: entity` frontmatter and is listed in entities/index.md
      (Phase D — the E5 checks, one directory down)
  E10 entity<->source citation integrity: the E6 check, applied to entities/`videos:`/
      `## Sources` instead of concepts'
  E11 bidirectional entity<->source links: the E7 check, applied to a source's
      `## Entities mentioned` section instead of `## Concepts covered`
  E12 no orphan entities: the E8 check (minus typed edges — entities have none), applied to
      entities/<slug>.md instead of concepts/<slug>.md

Concept-to-concept typed-edge links (Phase 16) are same-directory (`concepts/x.md` ->
`concepts/y.md`), so E2's generic link-resolution check already covers broken links there with
no extra code; only E8's orphan check needed extending to recognize them as valid inbound links.
Entities (Phase D) have no typed edges, so E12 doesn't need that extension.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RESERVED = {"index.md", "log.md"}

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
TYPE_RE = re.compile(r"^type:\s*(.+?)\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _strip_code(text: str) -> str:
    return INLINE_CODE_RE.sub("", FENCE_RE.sub("", text))


def _frontmatter(text: str) -> str | None:
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    return text[3:end] if end != -1 else None


def _get_type(text: str) -> str | None:
    fm = _frontmatter(text)
    if fm is None:
        return None
    m = TYPE_RE.search(fm)
    return m.group(1).strip().strip('"').strip("'") if m else None


def _md_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.md") if ".git" not in p.parts]


def _check_types(root: Path, files: list[Path], errors: list[str]) -> None:
    for p in files:
        if p.name in RESERVED:
            continue
        if not _get_type(p.read_text(encoding="utf-8")):
            errors.append(f"E1 missing/empty `type` frontmatter: {p.relative_to(root)}")


def _check_links(root: Path, files: list[Path], errors: list[str]) -> None:
    for p in files:
        text = _strip_code(p.read_text(encoding="utf-8"))
        for raw in LINK_RE.findall(text):
            link = raw.split()[0]
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = link.split("#", 1)[0]
            if not target:
                continue
            dest = (p.parent / target).resolve()
            if not dest.exists():
                errors.append(f"E2 broken link `{link}` in {p.relative_to(root)}")


def _index_targets(index: Path) -> set[str]:
    if not index.exists():
        return set()
    targets: set[str] = set()
    for raw in LINK_RE.findall(index.read_text(encoding="utf-8")):
        link = raw.split()[0].split("#", 1)[0]
        if link.startswith(("http://", "https://", "mailto:")) or not link:
            continue
        targets.add((index.parent / link).resolve().as_posix())
    return targets


def _check_dir_index(root: Path, dirname: str, code: str, errors: list[str]) -> None:
    """Every non-reserved ``<dirname>/*.md`` must be listed in ``<dirname>/index.md``.

    Generalized from the sources-only ``_check_sources_index`` (E3) to also cover
    ``concepts/`` (E5's index-coverage half) — same pattern, parameterized by directory and
    error code (design report §7).
    """
    dirpath = root / dirname
    if not dirpath.exists():
        return
    listed = _index_targets(dirpath / "index.md")
    for p in sorted(dirpath.glob("*.md")):
        if p.name in RESERVED:
            continue
        if p.resolve().as_posix() not in listed:
            errors.append(f"{code} not listed in {dirname}/index.md: {p.relative_to(root)}")


def _check_concept_type(root: Path, errors: list[str]) -> None:
    """E5's frontmatter half: every non-reserved ``concepts/*.md`` declares ``type: concept``
    specifically (stricter than E1's generic non-empty-``type`` check, since ``concepts/`` has
    exactly one valid type value)."""
    concepts_dir = root / "concepts"
    if not concepts_dir.exists():
        return
    for p in sorted(concepts_dir.glob("*.md")):
        if p.name in RESERVED:
            continue
        if _get_type(p.read_text(encoding="utf-8")) != "concept":
            errors.append(f"E5 missing/wrong `type: concept` frontmatter: {p.relative_to(root)}")


def _frontmatter_flow_list(text: str, key: str) -> list[str] | None:
    """Parse a ``key: [a, b, c]`` flow-style frontmatter field (the only list shape
    ``okf.py``'s ``_yaml_flow_list`` ever produces). ``None`` if the key is absent."""
    fm = _frontmatter(text)
    if fm is None:
        return None
    m = re.search(rf"^{re.escape(key)}:\s*\[(.*?)\]\s*$", fm, re.MULTILINE)
    if not m:
        return None
    inner = m.group(1).strip()
    return [item.strip() for item in inner.split(",")] if inner else []


def _section_link_slugs(text: str, heading: str, dirname: str) -> list[str]:
    """Every ``../<dirname>/<slug>.md`` link target under ``heading`` (e.g. ``## Sources``,
    ``## Concepts covered``) — used by E6/E7/E8 to read a page's declared relationships."""
    idx = text.find(heading)
    if idx == -1:
        return []
    section = text[idx:]
    slugs: list[str] = []
    for raw in LINK_RE.findall(section):
        link = raw.split()[0].split("#", 1)[0]
        m = re.search(rf"{re.escape(dirname)}/([^/]+)\.md$", link)
        if m:
            slugs.append(m.group(1))
    return slugs


def _check_concept_source_citations(root: Path, errors: list[str]) -> None:
    """E6 — concept<->source citation integrity: every slug in a concept's ``videos:``
    frontmatter must have a ``sources/<slug>.md``, and every slug cited under the concept's
    ``## Sources`` section must be one of the ``videos:`` slugs (design report §7)."""
    concepts_dir = root / "concepts"
    sources_dir = root / "sources"
    if not concepts_dir.exists():
        return
    existing_sources = (
        {p.stem for p in sources_dir.glob("*.md") if p.name not in RESERVED}
        if sources_dir.exists()
        else set()
    )
    for p in sorted(concepts_dir.glob("*.md")):
        if p.name in RESERVED:
            continue
        text = p.read_text(encoding="utf-8")
        videos = _frontmatter_flow_list(text, "videos") or []
        for slug in videos:
            if slug and slug not in existing_sources:
                errors.append(
                    f"E6 videos: lists `{slug}` but sources/{slug}.md does not exist: "
                    f"{p.relative_to(root)}"
                )
        video_set = set(videos)
        for slug in _section_link_slugs(text, "## Sources", "../sources"):
            if slug not in video_set:
                errors.append(
                    f"E6 ## Sources cites `{slug}`, not listed in videos: frontmatter: "
                    f"{p.relative_to(root)}"
                )


def _check_bidirectional_concept_links(root: Path, errors: list[str]) -> None:
    """E7 — SCHEMA §5's "bidirectional" rule: a concept's ``## Sources`` link to a source
    implies that source's ``## Concepts covered`` links back, and vice versa."""
    concepts_dir = root / "concepts"
    sources_dir = root / "sources"
    if not concepts_dir.exists() or not sources_dir.exists():
        return

    source_backlinks = {
        p.stem: set(_section_link_slugs(p.read_text(encoding="utf-8"), "## Concepts covered", "../concepts"))
        for p in sources_dir.glob("*.md")
        if p.name not in RESERVED
    }
    concept_citations = {
        p.stem: set(_section_link_slugs(p.read_text(encoding="utf-8"), "## Sources", "../sources"))
        for p in concepts_dir.glob("*.md")
        if p.name not in RESERVED
    }

    for concept_slug, cited_sources in sorted(concept_citations.items()):
        for source_slug in sorted(cited_sources):
            if concept_slug not in source_backlinks.get(source_slug, set()):
                errors.append(
                    f"E7 concepts/{concept_slug}.md links to sources/{source_slug}.md but "
                    f"sources/{source_slug}.md has no backlink under ## Concepts covered"
                )
    for source_slug, linked_concepts in sorted(source_backlinks.items()):
        for concept_slug in sorted(linked_concepts):
            if source_slug not in concept_citations.get(concept_slug, set()):
                errors.append(
                    f"E7 sources/{source_slug}.md links to concepts/{concept_slug}.md but "
                    f"concepts/{concept_slug}.md's ## Sources does not cite it back"
                )


def _concept_edge_link_slugs(text: str, heading: str) -> list[str]:
    """Every same-directory ``<slug>.md`` link target under a concept's typed-edge heading
    (``## Contrasts with`` / ``## Builds on`` / ``## Related``, Phase 16 design report §9 item
    4) — unlike ``_section_link_slugs``, these links have no ``../<dirname>/`` prefix since a
    concept page links to another concept page in the same directory."""
    idx = text.find(heading)
    if idx == -1:
        return []
    section = text[idx:]
    slugs: list[str] = []
    for raw in LINK_RE.findall(section):
        link = raw.split()[0].split("#", 1)[0]
        if "/" in link:
            continue
        m = re.match(r"([^/]+)\.md$", link)
        if m:
            slugs.append(m.group(1))
    return slugs


_EDGE_HEADINGS = ("## Contrasts with", "## Builds on", "## Related")


def _check_no_orphan_concepts(root: Path, errors: list[str]) -> None:
    """E8 — every ``concepts/<slug>.md`` (except reserved pages) must have >=1 inbound link
    from a source's ``## Concepts covered`` section or another concept's typed-edge section
    (``## Contrasts with`` / ``## Builds on`` / ``## Related``, Phase 16); the
    ``concepts/index.md`` directory listing does not count (design report §7). ``sources/``
    pages stay exempt — they're provenance leaves by design, not required to be linked-to."""
    concepts_dir = root / "concepts"
    sources_dir = root / "sources"
    if not concepts_dir.exists():
        return
    inbound: set[str] = set()
    if sources_dir.exists():
        for p in sources_dir.glob("*.md"):
            if p.name in RESERVED:
                continue
            inbound.update(
                _section_link_slugs(p.read_text(encoding="utf-8"), "## Concepts covered", "../concepts")
            )
    for p in concepts_dir.glob("*.md"):
        if p.name in RESERVED:
            continue
        text = p.read_text(encoding="utf-8")
        for heading in _EDGE_HEADINGS:
            inbound.update(_concept_edge_link_slugs(text, heading))
    for p in sorted(concepts_dir.glob("*.md")):
        if p.name in RESERVED:
            continue
        if p.stem not in inbound:
            errors.append(
                f"E8 orphan concept page (no inbound link from any source or concept): "
                f"{p.relative_to(root)}"
            )


def _check_entity_type(root: Path, errors: list[str]) -> None:
    """E9's frontmatter half: every non-reserved ``entities/*.md`` declares ``type: entity``
    specifically — the exact ``_check_concept_type`` shape, one directory down."""
    entities_dir = root / "entities"
    if not entities_dir.exists():
        return
    for p in sorted(entities_dir.glob("*.md")):
        if p.name in RESERVED:
            continue
        if _get_type(p.read_text(encoding="utf-8")) != "entity":
            errors.append(f"E9 missing/wrong `type: entity` frontmatter: {p.relative_to(root)}")


def _check_entity_source_citations(root: Path, errors: list[str]) -> None:
    """E10 — entity<->source citation integrity: the exact ``_check_concept_source_citations``
    shape, applied to ``entities/`` instead of ``concepts/``."""
    entities_dir = root / "entities"
    sources_dir = root / "sources"
    if not entities_dir.exists():
        return
    existing_sources = (
        {p.stem for p in sources_dir.glob("*.md") if p.name not in RESERVED}
        if sources_dir.exists()
        else set()
    )
    for p in sorted(entities_dir.glob("*.md")):
        if p.name in RESERVED:
            continue
        text = p.read_text(encoding="utf-8")
        videos = _frontmatter_flow_list(text, "videos") or []
        for slug in videos:
            if slug and slug not in existing_sources:
                errors.append(
                    f"E10 videos: lists `{slug}` but sources/{slug}.md does not exist: "
                    f"{p.relative_to(root)}"
                )
        video_set = set(videos)
        for slug in _section_link_slugs(text, "## Sources", "../sources"):
            if slug not in video_set:
                errors.append(
                    f"E10 ## Sources cites `{slug}`, not listed in videos: frontmatter: "
                    f"{p.relative_to(root)}"
                )


def _check_bidirectional_entity_links(root: Path, errors: list[str]) -> None:
    """E11 — a source's ``## Entities mentioned`` link to an entity implies that entity's
    ``## Sources`` links back, and vice versa. The exact ``_check_bidirectional_concept_links``
    shape, applied to ``entities/`` + ``## Entities mentioned`` instead of ``concepts/`` +
    ``## Concepts covered``."""
    entities_dir = root / "entities"
    sources_dir = root / "sources"
    if not entities_dir.exists() or not sources_dir.exists():
        return

    source_backlinks = {
        p.stem: set(
            _section_link_slugs(p.read_text(encoding="utf-8"), "## Entities mentioned", "../entities")
        )
        for p in sources_dir.glob("*.md")
        if p.name not in RESERVED
    }
    entity_citations = {
        p.stem: set(_section_link_slugs(p.read_text(encoding="utf-8"), "## Sources", "../sources"))
        for p in entities_dir.glob("*.md")
        if p.name not in RESERVED
    }

    for entity_slug, cited_sources in sorted(entity_citations.items()):
        for source_slug in sorted(cited_sources):
            if entity_slug not in source_backlinks.get(source_slug, set()):
                errors.append(
                    f"E11 entities/{entity_slug}.md links to sources/{source_slug}.md but "
                    f"sources/{source_slug}.md has no backlink under ## Entities mentioned"
                )
    for source_slug, linked_entities in sorted(source_backlinks.items()):
        for entity_slug in sorted(linked_entities):
            if source_slug not in entity_citations.get(entity_slug, set()):
                errors.append(
                    f"E11 sources/{source_slug}.md links to entities/{entity_slug}.md but "
                    f"entities/{entity_slug}.md's ## Sources does not cite it back"
                )


def _check_no_orphan_entities(root: Path, errors: list[str]) -> None:
    """E12 — every ``entities/<slug>.md`` must have >=1 inbound link from a source's
    ``## Entities mentioned`` section; the ``entities/index.md`` listing does not count. The
    exact ``_check_no_orphan_concepts`` shape, minus the typed-edge extension (entities have no
    entity<->entity edges, Phase D)."""
    entities_dir = root / "entities"
    sources_dir = root / "sources"
    if not entities_dir.exists():
        return
    inbound: set[str] = set()
    if sources_dir.exists():
        for p in sources_dir.glob("*.md"):
            if p.name in RESERVED:
                continue
            inbound.update(
                _section_link_slugs(p.read_text(encoding="utf-8"), "## Entities mentioned", "../entities")
            )
    for p in sorted(entities_dir.glob("*.md")):
        if p.name in RESERVED:
            continue
        if p.stem not in inbound:
            errors.append(
                f"E12 orphan entity page (no inbound link from any source): {p.relative_to(root)}"
            )


def _check_source_raw_parity(root: Path, errors: list[str]) -> None:
    src = {p.stem for p in (root / "sources").glob("*.md") if p.name not in RESERVED}
    raw = {p.stem for p in (root / "raw").glob("*.md") if p.name not in RESERVED}
    for slug in sorted(src - raw):
        errors.append(f"E4 sources/{slug}.md has no matching raw/{slug}.md")
    for slug in sorted(raw - src):
        errors.append(f"E4 raw/{slug}.md has no matching sources/{slug}.md")


def lint(root: Path) -> list[str]:
    """Run all checks against ``root`` and return the sorted list of ERROR messages."""
    errors: list[str] = []
    files = _md_files(root)
    _check_types(root, files, errors)
    _check_links(root, files, errors)
    _check_dir_index(root, "sources", "E3", errors)
    _check_source_raw_parity(root, errors)
    _check_dir_index(root, "concepts", "E5", errors)
    _check_concept_type(root, errors)
    _check_concept_source_citations(root, errors)
    _check_bidirectional_concept_links(root, errors)
    _check_no_orphan_concepts(root, errors)
    _check_dir_index(root, "entities", "E9", errors)
    _check_entity_type(root, errors)
    _check_entity_source_citations(root, errors)
    _check_bidirectional_entity_links(root, errors)
    _check_no_orphan_entities(root, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m distil.okf_lint <okf_root>", file=sys.stderr)
        return 2
    root = Path(argv[0])
    errors = lint(root)
    for e in errors:
        print(f"  ERROR {e}")
    print(f"\n{len(errors)} error(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
