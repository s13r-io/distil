# Distil — Personal Knowledge Distiller

Turn a YouTube video transcript into a useful teaching note backed by faithfully extracted
evidence **and** concrete ideas for applying it to *your* goals — filed into a growing,
cross-linked personal knowledge base that learns what's useful to you over time.

Distil routes extraction by knowledge type, anchors every item to the transcript (no invented
insights), then synthesizes the verified items into a reader-facing note: core takeaway, key
points, why it matters, how to apply it, caveats, and review questions. You score each result;
the score plus a reason refines your profile, so it gets more personally useful the more you use it.

> Status: **v0 / pre-release.** Built test-first. Self-hosted, single-user, MIT-licensed.

## How it works

```
transcript + your profile
  → triage (what kind of knowledge? how lossy? worth it?)
  → extract (routed by type, provenance-anchored)
  → normalize (atomic, faithful, stance-preserving)
  → link (application ideas tied to YOUR goals)
  → synthesize (teaching note grounded in verified items)
  → file (markdown note + evidence, cross-linked to your KB)
  → you score it → your profile improves

later, ask your knowledge base questions:
  question → semantic search over your notes → (grounded answer + source links)
           → or an honest "no relevant notes" when your KB doesn't cover it
```

See `docs/` for the full design: [PRD](docs/PRD.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Schema](docs/SCHEMA.md) ·
[Testing/TDD](docs/TESTING.md) · [Build guide](docs/AGENT_BUILD_GUIDE.md) ·
[Tracker](docs/TRACKER.md).

**Starting from scratch?** [GETTING_STARTED.md](GETTING_STARTED.md) walks you through creating
the repo and handing it to a coding agent.

## Quickstart (local)

```bash
git clone <your-fork-url> distil && cd distil
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env        # then edit: set your LLM API key and model

distil run path/to/transcript.txt        # produces a KB entry in kb/
distil run path/to/transcript.txt --url "https://youtube.com/watch?v=..."  # optional source link
distil show <entry_id>                    # read it
distil score <entry_id> --score 5 --reason relevant   # teach your profile
distil list                               # browse your knowledge base
distil ask "what do my notes say about X" # grounded answer + source links (or "no notes")
distil reindex                            # embed older entries for the read layer
distil delete <entry_id> --yes            # remove an entry, index row, and vectors
```

`run` cleans noisy uploaded filenames before using them as fallback titles, and `--url` stores
a YouTube source link at the top of the note. When possible, Distil also fetches public YouTube
oEmbed metadata (video title, channel, thumbnail) without an API key. `ask` answers **only**
from your stored notes and links to the source entries; if nothing in your KB is relevant, it
tells you so rather than guessing.

## Quickstart (Docker)

```bash
cp .env.example .env        # set your key + model
docker compose up --build
# kb/ and data/ are mounted as volumes, so your knowledge base persists
```

## Configuration

All via `.env` (see `.env.example`): your LLM API key, the model name, DB path, KB directory,
and the novelty ratio for anti-filter-bubble links. No secrets live in the source.

## Backup

Your knowledge base is plain markdown in `kb/` plus a SQLite index in `data/`. Back it up by
committing `kb/` to a private git repo, or syncing the folder — it's all human-readable files.

## Deploy it yourself / fork it

This project is MIT-licensed: clone it, bring your own LLM key, and run it anywhere Docker or
Python runs. No accounts, no telemetry, no external services. Swap the LLM provider by
implementing the `LLMClient` interface in `distil/llm.py`.

To host it on Railway, see [DEPLOY_RAILWAY.md](DEPLOY_RAILWAY.md). **Important:** hosting puts
the app on a public URL with your API key attached, so a single-user auth secret is required
before you expose a domain — the app refuses to serve publicly without one.

## Contributing

Issues and PRs welcome. The project is strictly test-driven — see [docs/TESTING.md](docs/TESTING.md).
A PR is mergeable when `pytest tests/unit` is green and new behavior ships with its tests. The
one inviolable rule: **never weaken the faithfulness guarantee** (extracted items must trace to
the transcript).

## License

[MIT](LICENSE).
