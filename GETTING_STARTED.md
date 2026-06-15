# Getting started (project bootstrap)

For the **owner**, before handing the repo to the building agent. This covers turning the
kickoff docs into a GitHub repo and starting the agent. The agent's own step-by-step is in
`docs/AGENT_BUILD_GUIDE.md`; the target code layout is in `docs/ARCHITECTURE.md` §3.

## What you have right now

This package is the **initial contents of your repo** — docs plus open-source scaffolding.
There is no code yet; the agent writes it following the build guide. The files here all sit at
the repo **root** (README, LICENSE, `.env.example`, `railway.toml`, `DEPLOY_RAILWAY.md`) except
the design docs, which live under `docs/`.

## Step 1 — Create an empty GitHub repo

On github.com: **New repository → name it `distil`** (or whatever you like) → **leave it empty**
(uncheck "Add a README"; don't add a license — you already have both). Private or public is your call.

Prefer the CLI? `gh repo create distil --private` (then follow its prompts).

## Step 2 — Clone it locally

```bash
git clone https://github.com/<you>/distil.git
cd distil
```

## Step 3 — Add the kickoff files and push

Copy the **contents** of this package into the cloned folder so `README.md` is at `./README.md`,
the docs at `./docs/`, etc. (not nested inside another folder). Then:

```bash
git add .
git commit -m "docs: project kickoff package"
git push
```

## Step 4 — Understand the layout (and the name overlap)

Starting layout:

```
distil/                     <- repo root
├── README.md  LICENSE  .env.example  railway.toml  DEPLOY_RAILWAY.md  GETTING_STARTED.md
└── docs/  (PRD, ARCHITECTURE, SCHEMA, TESTING, AGENT_BUILD_GUIDE, TRACKER)
```

After the agent builds (per `ARCHITECTURE.md` §3):

```
distil/                     <- repo root
├── pyproject.toml  .gitignore  Dockerfile  docker-compose.yml
├── docs/ ...
├── distil/                 <- Python package (code); same name, nested — this is normal
│   ├── __init__.py  models.py  llm.py  embed.py  query.py  triage.py  extract.py
│   ├── normalize.py  link.py  graph.py  profile_update.py  store.py  pipeline.py  cli.py
│   └── prompts/
├── web/                    (web UI, later phase)
├── tests/{fixtures,unit,eval}/
├── kb/                     (generated notes — gitignored)
└── data/                   (distil.db — gitignored)
```

The nested `distil/distil/` is the conventional Python layout (repo name == package name). Don't
"fix" it. `.env`, `data/`, and optionally `kb/` are gitignored (the agent adds `.gitignore` in
Phase 0).

## Step 5 — Answer the open decisions

Open `docs/TRACKER.md` → **Decisions needed** (D1–D7). Fill in any you have an opinion on
(score granularity, URL fetch, embeddings local vs API, hosting target, auth method). Leaving
them blank means the agent uses the documented defaults.

## Step 6 — Hand off to the agent

Point Claude Cowork (or any coding agent) at the repo and instruct it roughly:

> Read everything in `docs/`. Follow `docs/AGENT_BUILD_GUIDE.md` starting at Phase 0. Work one
> task at a time from `docs/TRACKER.md`, test-first (TDD), updating the tracker and committing
> per task. Don't change the stack without raising it in the tracker. Never weaken the
> faithfulness or abstention guarantees.

The agent creates `pyproject.toml`, `.gitignore`, the package skeleton, and CI in Phase 0, then
proceeds through the phases. You review via the tracker and the per-task commits/PRs.

## Step 7 — When you're ready to host

Local-first: the README quickstart gets you running with Python or Docker. To put it on a public
URL, follow `DEPLOY_RAILWAY.md` — and note the auth gate is required before you expose a domain.
