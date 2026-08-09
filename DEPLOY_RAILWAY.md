# Deploying Distil on Railway

A click-by-click guide to hosting Distil on [Railway](https://railway.com). The engine is
identical to local; only storage, auth, and the port binding change. Read the three "gotchas"
at the bottom before you start — they're the things that bite people.

> **Read this first:** hosting puts the app on a public URL with your LLM API key attached.
> Anyone who finds the URL could spend your budget and read/write your private notes. **Do not
> generate a public domain until the auth gate is enabled** (Step 7). The app is built to
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
`DISTIL_PROFILE_ALPHA`, `DISTIL_STAGING_DIR` (defaults to a `staging/` subdirectory next to
`DISTIL_DB_PATH`, i.e. already on the volume — only set this to move it elsewhere),
`DISTIL_PLAYLIST_FETCH_DELAY_SECONDS` (default 3.0 — pause between a playlist's transcript
fetches). Never commit these — they live only in Railway.

`DISTIL_COLLECTOR_TOKEN`, `DISTIL_COLLECTOR_LEASE_SECONDS` (default 600), and
`DISTIL_COLLECTOR_EXPIRY_SECONDS` (default 7 days) configure the external-collector queue that
lets a bot-checked video be fetched from a trusted machine elsewhere instead — see `AGENTS.md`'s
"External-collector queue" entry. The collector program that actually does the fetching
(`distil collector-run`) runs on your own machine, not here — set `DISTIL_COLLECTOR_TOKEN` here
to a real secret and follow RUNBOOK.md's "Run the external collector" section there. Leave it
unset to keep the queue unusable (fails closed); a bot-checked video then just waits up to its
expiry instead of failing.

> **Local embeddings + Railway:** with `DISTIL_EMBEDDER=local` (the chosen default) a small
> embedding model loads into the service's RAM and should be baked into the image at build
> time. Pick an instance with enough memory for it. On a very small instance, set
> `DISTIL_EMBEDDER=api` instead — it's a config change only, no code change.

### 4. Wire in the YouTube PO-token provider (fixes the datacenter-IP bot check)
Railway's datacenter IPs get YouTube's `Sign in to confirm you're not a bot` challenge on ingest,
even with the player-client hardening already in place — that's an identity check, not a rate
limit, and no `--extractor-args` value alone can satisfy it. The fix is to run a second service
that hands yt-dlp a real proof-of-origin (PO) token: the prebuilt
[`brainicism/bgutil-ytdlp-pot-provider`](https://github.com/Brainicism/bgutil-ytdlp-pot-provider)
image, reached over Railway's private network, no public domain needed.

> **Read this first:** per upstream's own documentation, a PO token makes traffic look more
> legitimate to YouTube — it does **not** guarantee clearing a bot check. Treat this step as the
> best available mitigation, confirm it's *wired correctly* (below), and watch real ingest logs
> after deploying to see whether the datacenter IP actually clears.

Pick whichever path matches how you work — both end in the same two services and one new
variable.

#### 4a. Dashboard clicks
1. In your Railway project: **New → Empty Service** (or **Docker Image** if your Railway version
   offers it directly), then Service → **Settings → Source → Deploy from Docker Image**, and set
   the image to `brainicism/bgutil-ytdlp-pot-provider:latest`. Name the service something
   findable, e.g. `bgutil-pot-provider`. Do **not** attach a volume or generate a public domain
   for it — it's private-network-only and stateless.
2. Deploy it. Railway services in the same project share private networking automatically; no
   extra "connect" step is needed. Note the service name you gave it — Railway's internal DNS
   for it is `<service-name>.railway.internal`, and the provider listens on port `4416` inside
   the image.
3. On the **distil** service (not the provider service): Service → **Variables** → add
   `DISTIL_POT_PROVIDER_URL = http://<service-name>.railway.internal:4416` (use the exact service
   name from step 1, e.g. `http://bgutil-pot-provider.railway.internal:4416`).
4. Redeploy the distil service so it picks up the new variable.

#### 4b. Non-interactive `railway` CLI
Equivalent to 4a, run from a machine already authenticated (`railway login`) and linked to the
project (`railway link`):

```shell
# 1. Create the provider service from the public image (no repo, no build).
railway add --image brainicism/bgutil-ytdlp-pot-provider:latest --service bgutil-pot-provider --json

# 2. Point distil at it over private networking — internal DNS is <service-name>.railway.internal,
#    port 4416 is the image's default. --skip-deploys avoids two redeploys back to back; the
#    manual redeploy in step 3 picks the variable up.
railway variable set \
  DISTIL_POT_PROVIDER_URL=http://bgutil-pot-provider.railway.internal:4416 \
  --service distil --skip-deploys --json

# 3. Redeploy distil so the new variable takes effect.
railway redeploy --service distil --yes --json
```

Do not run `railway domain` for the provider service — it must stay off the public internet;
only distil reaches it, over the private network. Also do not raise `numReplicas` on either
service or attach a volume to the provider service — this project stays single-replica,
single-volume (see `railway.toml` and the gotchas below).

#### Confirm the plugin actually loaded — and that a token is actually requested
The wiring only *offers* yt-dlp a PO token — confirm yt-dlp is actually discovering the plugin
*and* asking it for one before trusting it (Phase 22 shipped without either of those confirmed,
which is exactly how it worked for one video and silently stopped for the next — see
`distil/youtube.py`'s module docstring, Phase 23, for the full root-cause). This no longer needs
shell access: hit the diagnostic route on the running service —

```shell
curl -s -b "<your auth cookie>" \
  "https://<your-domain>/diagnostics/youtube-pot?url=<a video URL>" | python3 -m json.tool
```

(or, with filesystem access, `distil youtube-diagnose-pot <a video URL>`, or a one-off local
`yt-dlp` run using this repo's exact `--extractor-args`, per the module docstring). Check two
things in the response, not just one:

1. `provider_discovery` should look like `[youtube] [pot] PO Token Providers: bgutil:http-1.3.1
   (external)`. Missing, or `(unavailable)` instead of `(external)`, means the plugin isn't
   reaching the provider service — check the internal URL and that the provider service is
   actually running (`railway logs --service bgutil-pot-provider`) before assuming the bot check
   itself is the problem.
2. `context_attempts` should be non-empty (at minimum a `player`/`mweb` entry). An *empty* list
   with a healthy `provider_discovery` means the plugin loaded fine but yt-dlp never asked it for
   anything — a discovery line alone was never sufficient proof the wiring works end to end.

### 5. Confirm the start command
`railway.toml` already sets it to bind the injected port:
`uvicorn web.app:app --host 0.0.0.0 --port $PORT`. If you configure the service manually
instead, make sure it binds `0.0.0.0` and `$PORT` — a hardcoded port will fail to receive traffic.

### 6. Redeploy and check logs
Trigger a redeploy. Watch the **Deploy Logs** for a clean start and the app binding to the port.
Fix any missing-variable errors before continuing.

### 7. Enable auth, *then* expose a domain
Confirm `DISTIL_PUBLIC=true` and `DISTIL_AUTH_SECRET` are set (Step 3). Only now: Service →
**Settings → Networking → Generate Domain**. Open the URL; you should be prompted for the
secret before any data is reachable. If you can reach data without auth, stop and fix it.

### 8. Back up your knowledge base (provider-independent)
Your `kb/` now lives on a Railway volume. Don't let it be trapped there:

- **Preferred:** configure the scheduled job (Phase 11.5) that commits `kb/` to a private git
  remote. Your notes are plain markdown, so this gives you a portable, versioned backup you own.
- **Fallback:** enable Railway's volume backups (paid).

## The three gotchas, in one place

1. **Ephemeral disk.** Without a volume at `/data`, every redeploy wipes your KB. (Step 2.)
2. **Public = exposed.** A generated domain is open to the internet with your key attached;
   auth is mandatory, not optional. (Step 7.)
3. **Port binding.** Bind `0.0.0.0:$PORT`; there's no port-mapping layer on Railway. (Step 5.)

## Alternative: managed Postgres
For the index you can swap SQLite for Railway's managed Postgres (it provisions a `DATABASE_URL`
and you skip the volume for the DB — though `kb/` markdown still needs the volume or object
storage). Overkill for single-user, but easy if `store.py` uses SQLAlchemy/SQLModel.
