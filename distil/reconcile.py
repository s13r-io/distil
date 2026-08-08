"""Bundle reconcile: repairs an OKF bundle that has drifted from the database.

Drift happens whenever the derived ``okf/`` bundle and the live DB disagree — most commonly
because entries were deleted before :func:`distil.canonicalize.run_delete_entry_stage` existed
to close the delete-cascade gaps, or because the deployment volume was edited by hand. This
module compares the bundle on disk against the DB and removes files with no live owner:
orphaned ``concepts/<id>.md`` pages, and orphaned ``sources/<slug>.md``/``raw/<slug>.md`` pairs.

Conservative by construction (see the task brief this closes): a file is only ever removed when
its owner can be positively determined to be gone. Anything undeterminable — a ``sources/`` page
with no (or an unparseable) ``distil_entry_id`` frontmatter field, or a ``raw/`` page with no
matching ``sources/`` page to inherit ownership from — is left alone and reported, never guessed
at. Dry run (report only, delete nothing) is the default; deleting requires ``apply=True``. This
never touches ``kb/`` or the database — ``kb/`` is the source of truth and only the derived,
regenerable bundle is reconcile's to repair. Concept-page removal reuses ``okf.remove_concept``,
and orphaned source/raw pairs reuse ``okf.remove_entry_pages``, so behaviour matches the delete
cascade exactly rather than reimplementing file removal here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import okf
from .store import Store


@dataclass
class ReconcileReport:
    dry_run: bool
    removed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def reconcile_okf_bundle(store: Store, *, apply: bool = False) -> ReconcileReport:
    """Compare ``store``'s OKF bundle against its DB and remove orphaned files.

    With ``apply=False`` (the default), nothing is deleted — ``report.removed`` lists what
    *would* be removed. With ``apply=True``, those files are actually deleted and the indexes
    regenerated.
    """
    root = Path(store.okf_root)
    report = ReconcileReport(dry_run=not apply)

    live_entry_ids = store.all_entry_ids()
    live_concept_ids = {c.concept_id for c in store.list_concepts()}

    sources_dir = root / "sources"
    raw_dir = root / "raw"
    concepts_dir = root / "concepts"

    orphan_slugs: list[str] = []
    live_slugs: set[str] = set()
    live_slug_owners: dict[str, str] = {}
    for path in _pages(sources_dir):
        owner = okf.frontmatter_field(path.read_text(encoding="utf-8"), "distil_entry_id")
        if not owner:
            report.skipped.append(_rel(path, root))
            continue
        if owner in live_entry_ids:
            live_slugs.add(path.stem)
            live_slug_owners[path.stem] = owner
        else:
            orphan_slugs.append(path.stem)

    for slug in orphan_slugs:
        report.removed.append(_rel(sources_dir / f"{slug}.md", root))
        raw_path = raw_dir / f"{slug}.md"
        if raw_path.exists():
            report.removed.append(_rel(raw_path, root))
        if apply:
            okf.remove_entry_pages(slug, root)

    accounted_slugs = live_slugs | set(orphan_slugs)
    for path in _pages(raw_dir):
        if path.stem not in accounted_slugs:
            report.skipped.append(_rel(path, root))

    orphan_concept_ids = [
        path.stem for path in _pages(concepts_dir) if path.stem not in live_concept_ids
    ]
    for concept_id in orphan_concept_ids:
        report.removed.append(_rel(concepts_dir / f"{concept_id}.md", root))
        if apply:
            okf.remove_concept(concept_id, root)

    # A removed concept can leave a live source's "## Concepts covered" section pointing at a
    # page that's now gone (the same stale-backlink shape Gap 2 closes for the delete path, at
    # reconcile's bundle-wide scale) — refresh every live source from current DB truth so no
    # dangling link survives.
    if apply and orphan_concept_ids:
        for entry_id in set(live_slug_owners.values()):
            try:
                entry = store.load_entry(entry_id)
            except Exception:
                continue
            okf.render_source_with_concepts(entry, store, root)

    return report


def _pages(dirpath: Path) -> list[Path]:
    if not dirpath.exists():
        return []
    return sorted(p for p in dirpath.glob("*.md") if p.name != "index.md")


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))
