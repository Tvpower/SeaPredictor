#!/usr/bin/env bash
# Build the Next.js UI as a static export so FastAPI can serve it from /.
#
# Output goes to frontend/frontend/out/, which src/api/server.py mounts at /
# whenever the directory exists.
#
# Usage:
#   scripts/build_frontend.sh             # build for prod
#   FRONTEND_INSTALL=1 scripts/build_frontend.sh   # also runs `npm install` first

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend/frontend"

cd "$FRONTEND_DIR"

if [[ "${FRONTEND_INSTALL:-0}" == "1" || ! -d node_modules ]]; then
  echo "[build_frontend] installing npm deps..."
  npm install
fi

echo "[build_frontend] building static export..."
npm run build:static

echo "[build_frontend] done. Output at $FRONTEND_DIR/out"
echo "[build_frontend] FastAPI will serve it at / when you run:"
echo "    uvicorn src.api.server:app --reload"
