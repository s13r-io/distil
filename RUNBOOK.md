# Distil — Owner Runbook

This is your step-by-step checklist for taking the project from "just built" to "running,
verified, and (optionally) deployed." Follow it top to bottom. Every command is copy-pasteable.

Where you see `<...>`, replace it with your real value (no angle brackets).

**Conventions**
- Lines starting with `$` are commands you type in a terminal (don't type the `$`).
- "Project root" = the folder containing `pyproject.toml` (i.e. `.../distil`).
- macOS/Linux shown. On Windows use PowerShell; venv-activate differs (noted inline).

---

## 0. One-time prerequisites (do these once)

### 0.1 Confirm Python 3.11+ is installed
The project requires Python **3.11 or newer** (3.10 will be refused by the installer).

```
$ python3 --version
```

If it prints `Python 3.11.x` or higher, you're good. If it's lower (or "command not found"):
- **macOS:** `brew install python@3.12` then use `python3.12` in place of `python3` below.
- **Windows:** install from https://www.python.org/downloads/ (check "Add to PATH").
- **Linux (Ubuntu):** `sudo apt update && sudo apt install python3.12 python3.12-venv`

### 0.2 Get an Anthropic API key
1. Go to https://console.anthropic.com/ and sign in (or create an account).
2. Open **Settings → API Keys → Create Key**. Copy it (starts with `sk-ant-...`).
3. Keep it somewhere safe. You'll paste it into `.env` in step 2. **Never commit it.**

### 0.3 Pick a current model string
Model names change over time, so don't guess. Open the official list and copy the exact id:
- https://platform.claude.com/docs/en/about-claude/models/overview

Pick a general-purpose model (an Opus or Sonnet tier is appropriate here). Copy its **API
model string** (e.g. something like `claude-sonnet-4-6` — use whatever the docs currently
list). You'll paste this as `DISTIL_MODEL` in step 2.

---

## 1. Set up the project locally

Open a terminal and go to the project root:

```
$ cd "/Users/saurabh.karmakar/s13r-io/distil"
```

Create and activate a virtual environment (keeps dependencies isolated):

```
$ python3 -m venv .venv
$ source .venv/bin/activate
```
> Windows PowerShell instead: `.venv\Scripts\Activate.ps1`
> You'll know it worked when your prompt shows `(.venv)` at the start.

Install Distil with everything needed to run AND to run the evals (LLM provider + local
embeddings + vector search + dev/test tools):

```
$ pip install --upgrade pip
$ pip install -e ".[anthropic,embed-local,vec,dev]"
```

This takes a few minutes the first time (it downloads PyTorch for local embeddings). When it
finishes, confirm the CLI is installed:

```
$ distil --help
```
You should see the commands: `run`, `score`, `list`, `show`, `ask`, `reindex`.

---

## 2. Configure your secrets and settings

Copy the example env file and open it in an editor:

```
$ cp .env.example .env
```

Edit `.env` and set at least these two lines (leave the rest at their defaults for now):

```
ANTHROPIC_API_KEY=sk-ant-...your real key...
DISTIL_MODEL=...the model string you copied in step 0.3...
```

Also confirm these (defaults are fine for local use):
```
DISTIL_EMBEDDER=local
DISTIL_EMBED_MODEL=all-MiniLM-L6-v2
DISTIL_DB_PATH=./data/distil.db
DISTIL_KB_DIR=./kb
DISTIL_PUBLIC=false
```

Load the variables into your current terminal session (so the `distil` command can read them):

```
$ set -a; source .env; set +a
```
> Windows PowerShell: load each var with `$env:ANTHROPIC_API_KEY="sk-ant-..."` etc., OR just
> install `python-dotenv` and they'll be read from `.env` automatically in most shells.
> Re-run the `set -a; source .env; set +a` line any time you open a new terminal.

---

## 3. Run the write loop (your first real entry)

You need a transcript. Two options:

**Option A — paste text directly:**
```
$ distil run --paste "When you write a function, keep it small and focused on one thing. Name things clearly so the name removes the need for a comment. And write the test first."
```

**Option B — from a file** (`.srt`, `.txt`, or `.md`). For example, save a YouTube transcript
to `~/Downloads/talk.txt`, then:
```
$ distil run ~/Downloads/talk.txt
```

Either way, Distil prints the path of the filed entry, e.g.:
```
/Users/saurabh.karmakar/s13r-io/distil/kb/e_1a2b3c4d5e6f.md
```
Copy that **entry id** — the `e_...` part of the filename (without `.md`). You'll use it next.

> If you see "ANTHROPIC_API_KEY is not set", you skipped step 2's `source .env` line — run it
> and try again. If you see "DISTIL_MODEL is not set", set it in `.env` (step 0.3 / 2).

### Browse what you filed
```
$ distil list
$ distil show e_1a2b3c4d5e6f        # use your real entry id
```
`show` prints the full markdown entry (knowledge items, application links, sources).

---

## 4. Score an entry (teach the profile)

Tell Distil how useful that entry was. `--score` is 1–5, `--reason` is one of:
`relevant`, `already_knew`, `bad_source`, `wrong_for_me`, `irrelevant_now`.

```
$ distil score e_1a2b3c4d5e6f --score 5 --reason relevant
```
It prints `Scored ... Profile updated.` Each score nudges what Distil surfaces next time.

> The same score teaches different things by reason: e.g. `5 relevant` upweights the topic;
> `2 already_knew` marks it known (suppresses basics) without disliking it.

---

## 5. Ask your knowledge base (the read layer)

Add a few more entries (repeat step 3) so there's something to retrieve, then:

```
$ distil ask "how should I write functions?"
```
You'll get either:
- a **grounded answer** with a `Sources:` list (entry/item + timestamp), built only from your
  notes; or
- `No relevant notes found...` — the honest "I don't have notes on this" response. This is
  correct behavior, not a bug: Distil refuses to answer from outside knowledge.

Other forms:
```
$ distil ask "kubernetes networking" --lookup     # just list matching notes, no written answer
$ distil reindex                                   # embed older entries (run once if you added
                                                   # entries before embeddings were set up)
```

You now have the full loop working. **Steps 6–7 are the evaluation; step 8 is optional hosting.**

---

## 6. Run the gated evaluation suite (confirm faithfulness + abstention)

These tests call the real model to check that extraction is faithful and that the read layer
abstains correctly. They are **skipped by default** and only run when your API key is present.

> Cost note: this makes a handful of real API calls (a few cents). Make sure `DISTIL_MODEL`
> points at a model your key can access.

### 6.1 Make sure your key + model are loaded in this terminal
```
$ set -a; source .env; set +a
$ echo "key set? ${ANTHROPIC_API_KEY:+yes}  model: $DISTIL_MODEL"
```
This should print `key set? yes` and your model string. If not, redo step 2.

### 6.2 Run ONLY the eval tests
```
$ pytest -m eval -v
```

### 6.3 Read the results — this IS the scoring
pytest prints one line per test. Here's how to interpret them:

- **`PASSED`** — the guarantee held. What each checks:
  - `test_t4_low_value_vlog...` → the junk vlog was correctly judged "little to extract."
  - `test_t5_screen_share...` → the screen-share transcript was flagged high information-loss.
  - `test_e3_every_quote_appears_in_transcript[...]` → **the headline faithfulness check**:
    every extracted quote really appears in the source (no fabrication), across 5 transcripts.
  - `test_q7_answerable...` / `test_q7_no_note...` → answerable questions return correct
    sources; questions with no notes abstain 100% of the time.
- **`SKIPPED`** — your key/model wasn't detected (redo 6.1) or the local embedder isn't
  installed (you skipped `embed-local` in step 1).
- **`FAILED`** — a guarantee did not hold. **Do not ship.** Copy the failing test name and the
  assertion message and send it back to me (or open an issue). The most important one to never
  ignore is any `test_e3...` failure — that means a fabricated quote slipped through.

### 6.4 Run the full suite (unit + eval) once, to confirm everything is green together
```
$ pytest -m "unit or eval"
```
Expect a line like `XXX passed` (≈135 unit + the eval tests). If unit tests fail but evals
were untouched, something in your environment differs — tell me what failed.

> The fast, no-key, no-cost suite you can run anytime is just: `$ pytest tests/unit`

---

## 7. Decide: is it good enough to tag a release?

Only proceed if step 6 was all green. Then create the `v0.0.1` tag — this is the one piece I
intentionally left for you (I don't push to your GitHub).

```
$ git tag v0.0.1
$ git push origin v0.0.1
```
Pushing a `v*` tag triggers the **Release** GitHub Action (`.github/workflows/release.yml`),
which re-runs lint + unit tests, builds the package, and creates a GitHub Release. Check the
**Actions** tab on GitHub to confirm it went green.

> First, push your normal commits if you haven't: `$ git push origin main`
> (The build added many commits; pushing `main` makes them visible on GitHub.)

---

## 8. (Optional) Host it on Railway

Only do this if you want the web UI on a URL. **Hosting puts your API key on the public
internet, so the auth secret is mandatory** — the app refuses to serve publicly without it.

The detailed click-by-click guide is already in the repo: **`DEPLOY_RAILWAY.md`**. The short
version:

1. Push your repo to GitHub (`git push origin main`).
2. Railway → **New Project → Deploy from GitHub repo** → pick this repo.
3. Service → **Settings → Volumes → New Volume**, mount path **`/data`**.
4. Service → **Variables**, set (see `DEPLOY_RAILWAY.md` for the full list):
   ```
   ANTHROPIC_API_KEY   = <your key>
   DISTIL_MODEL        = <your model string>
   DISTIL_DB_PATH      = /data/distil.db
   DISTIL_KB_DIR       = /data/kb
   DISTIL_PUBLIC       = true
   DISTIL_AUTH_SECRET  = <a long random string — generate with: openssl rand -hex 32>
   ```
5. Redeploy; watch logs for a clean start.
6. **Only after** auth vars are set: **Settings → Networking → Generate Domain**.
7. Open the URL — you should be prompted/blocked without the secret. To call a data route:
   send header `Authorization: Bearer <your DISTIL_AUTH_SECRET>`.

### 8.1 Test locally with Docker first (recommended before Railway)
```
$ docker compose up --build
```
Then open http://localhost:8000 — you'll see the Distil page with an ask box. `kb/` and
`data/` are mounted as folders so your knowledge base persists.

---

## 9. (Optional) Back up your knowledge base

Your notes are plain markdown in `kb/`. Back them up to a **separate private** git repo:

```
$ export DISTIL_KB_DIR=./kb
$ export DISTIL_BACKUP_REMOTE=git@github.com:<you>/distil-kb-backup.git   # a NEW empty repo
$ bash scripts/backup_kb.sh
```
Run it on a schedule (cron, or a Railway cron service) to keep an off-platform copy.

---

## Quick reference — daily use, once set up

```
$ cd "/Users/saurabh.karmakar/s13r-io/distil"
$ source .venv/bin/activate
$ set -a; source .env; set +a

$ distil run <file-or>  --paste "..."     # add knowledge
$ distil list                              # see entries
$ distil show <entry_id>                   # read one
$ distil score <entry_id> --score N --reason <reason>
$ distil ask "your question"               # query your notes
```

## If something goes wrong
- **"ANTHROPIC_API_KEY is not set" / "DISTIL_MODEL is not set"** → run `set -a; source .env; set +a`
  in this terminal, and check `.env` has both values.
- **`pip install` fails with "requires a different Python"** → your Python is < 3.11 (step 0.1).
- **Evals all SKIPPED** → key/model not loaded (step 6.1) or you didn't install `embed-local`.
- **`distil ask` always says "no relevant notes"** → you have few/no entries, or the threshold
  is high; add more entries, or lower `DISTIL_RETRIEVAL_THRESHOLD` in `.env` (e.g. 0.25).
- **Anything FAILED in evals** → don't deploy; send me the test name + message.
```
```

## What I (the builder) already verified vs. what needs YOU
- ✅ Done & verified by me: all code, 135 unit tests green, lint clean, the faithfulness and
  abstention guarantees reviewed in code.
- ⬜ Needs you (this runbook): install on a 3.11 machine, add your API key, run the **eval
  suite** (only your key can), push commits + the `v0.0.1` tag, and (optionally) deploy.
