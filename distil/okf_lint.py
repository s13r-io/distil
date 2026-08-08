#!/usr/bin/env python
"""Deterministic conformance checker for a Distil OKF bundle (stdlib only). See ``okf.py``.

    python -m distil.okf_lint <okf_root>

Scoped to the bundle shape this phase produces (``sources/`` + ``raw/`` only — no
``concepts/``/``entities/`` yet, so there is no orphan-page check here; that arrives with
Phase 3/5 once those directories exist).

Checks (all ERROR; exit non-zero if any fire):
  E1  every non-reserved .md has YAML frontmatter with a non-empty `type`
  E2  every relative markdown link resolves to a file in the bundle
  E3  every sources/<slug>.md is listed in sources/index.md
  E4  sources/<slug>.md <-> raw/<slug>.md parity, both directions
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


def _check_sources_index(root: Path, errors: list[str]) -> None:
    sources_dir = root / "sources"
    if not sources_dir.exists():
        return
    listed = _index_targets(sources_dir / "index.md")
    for p in sorted(sources_dir.glob("*.md")):
        if p.name in RESERVED:
            continue
        if p.resolve().as_posix() not in listed:
            errors.append(f"E3 not listed in sources/index.md: {p.relative_to(root)}")


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
    _check_sources_index(root, errors)
    _check_source_raw_parity(root, errors)
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
