#!/usr/bin/env bash
# Provider-independent KB backup (ARCHITECTURE §8; TRACKER D3).
# Commits the markdown knowledge base to a SEPARATE private git remote, so your notes are never
# trapped on one cloud volume. Run on a schedule (cron / Railway cron / GitHub Actions).
#
# Usage:
#   DISTIL_KB_DIR=/data/kb DISTIL_BACKUP_REMOTE=git@github.com:you/distil-kb-backup.git \
#     scripts/backup_kb.sh
#
# The backup repo is intentionally a different repo from the code: your KB is private data.
set -euo pipefail

KB_DIR="${DISTIL_KB_DIR:-./kb}"
REMOTE="${DISTIL_BACKUP_REMOTE:?Set DISTIL_BACKUP_REMOTE to your private backup repo URL}"
BRANCH="${DISTIL_BACKUP_BRANCH:-main}"

if [ ! -d "$KB_DIR" ]; then
  echo "KB dir not found: $KB_DIR" >&2
  exit 1
fi

cd "$KB_DIR"

if [ ! -d .git ]; then
  git init -q
  git remote add origin "$REMOTE"
fi

# Keep the configured remote in sync (idempotent).
git remote set-url origin "$REMOTE" 2>/dev/null || git remote add origin "$REMOTE"

git add -A
if git diff --cached --quiet; then
  echo "No KB changes to back up."
  exit 0
fi

git -c user.name="distil-backup" -c user.email="backup@distil.local" \
  commit -q -m "kb backup $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push -q origin "HEAD:${BRANCH}"
echo "KB backed up to ${REMOTE} (${BRANCH})."
