# Deploying Distil on Railway

A click-by-click guide to hosting Distil on [Railway](https://railway.com). The engine is
identical to local; only storage, auth, and the port binding change. Read the three "gotchas"
at the bottom before you start — they're the things that bite people.

> **Read this first:** hosting puts the app on a public URL with your LLM API key attached.
> Anyone who finds the URL could spend your budget and read/write your private notes. **Do not
> generate a public domain until the auth gate is enabled** (Step 6). The app is built to
> refuse public serving without `DISTIL_AUTH_SECRET`, but don't rely on that as your only line.

## Prerequisites

- A Railway account. A **paid plan (Hobby or higher)** is needed for a persistent volume and
  an always-on service; the free/trial tier's volume is only 0.5 GB and runs on one-time credit.
- Your repo pushed to GitHub.
- Your LLM API key and a current model string.

## Steps

### 1. Create the project from your repo
In Railway: **New Project → Deploy from GitHub repo →** pick your Distil fork. Railway detects
the `Dockerfile` (and `railway.toml`) and starts a first build. Let it finish; it won't work
correctly until you complete the steps below.

### 2. Attach a persistent volume
Open the service → **Settings → Volumes → New Volume**. Set the **Mount Path** to `/data`.
This is where your knowledge base and database live so they survive redeploys.

- Volumes mount at **runtime, not during build** — never write data to `/data` in the Dockerfile.
- There is **one volume per service**; this app is single-user/single-service, which is fine.
- Default sizes: ~0.5 GB free/trial, 5 GB Hobby, 50 GB Pro. Markdown + SQLite is tiny, so even
  Hobby lasts a very long time; you can resize later on a paid plan.

### 3. Set service variables
Service → **Variables**. Add:

```
ANTHROPIC_API_KEY    = <your key>
DISTIL_MODEL         = <a current model string>
DISTIL_EMBEDDER      = local            # or "api"
DISTIL_EMBED_MODEL   = <embedding model>
DISTIL_DB_PATH       = /data/distil.db  # on the volume
DISTIL_KB_DIR        = /data/kb         # on the volume
DISTIL_PUBLIC        = true
DISTIL_AUTH_SECRET   = <a long random secret>   # REQUIRED for public hosting
```

Optional tuning: `DISTIL_RETRIEVAL_THRESHOLD`, `DISTIL_TOP_K`, `DISTIL_NOVELTY_RATIO`,
`DISTIL_PROFILE_ALPHA`. Never commit these — they live only in Railway.

> **Local embeddings + Railway:** with `DISTIL_EMBEDDER=local` (the chosen default) a small
> embedding model loads into the service's RAM and should be baked into the image at build
> time. Pick an instance with enough memory for it. On a very small instance, set
> `DISTIL_EMBEDDER=api` instead — it's a config change only, no code change.

### 4. Confirm the start command
`railway.toml` already sets it to bind the injected port:
`uvicorn web.app:app --host 0.0.0.0 --port $PORT`. If you configure the service manually
instead, make sure it binds `0.0.0.0` and `$PORT` — a hardcoded port will fail to receive traffic.

### 5. Redeploy and check logs
Trigger a redeploy. Watch the **Deploy Logs** for a clean start and the app binding to the port.
Fix any missing-variable errors before continuing.

### 6. Enable auth, *then* expose a domain
Confirm `DISTIL_PUBLIC=true` and `DISTIL_AUTH_SECRET` are set (Step 3). Only now: Service →
**Settings → Networking → Generate Domain**. Open the URL; you should be prompted for the
secret before any data is reachable. If you can reach data without auth, stop and fix it.

### 7. Back up your knowledge base (provider-independent)
Your `kb/` now lives on a Railway volume. Don't let it be trapped there:

- **Preferred:** configure the scheduled job (Phase 11.5) that commits `kb/` to a private git
  remote. Your notes are plain markdown, so this gives you a portable, versioned backup you own.
- **Fallback:** enable Railway's volume backups (paid).

## The three gotchas, in one place

1. **Ephemeral disk.** Without a volume at `/data`, every redeploy wipes your KB. (Step 2.)
2. **Public = exposed.** A generated domain is open to the internet with your key attached;
   auth is mandatory, not optional. (Step 6.)
3. **Port binding.** Bind `0.0.0.0:$PORT`; there's no port-mapping layer on Railway. (Step 4.)

## Alternative: managed Postgres
For the index you can swap SQLite for Railway's managed Postgres (it provisions a `DATABASE_URL`
and you skip the volume for the DB — though `kb/` markdown still needs the volume or object
storage). Overkill for single-user, but easy if `store.py` uses SQLAlchemy/SQLModel.
