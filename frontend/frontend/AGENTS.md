<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

---

## SeaPredictor frontend

Next.js 16 + React 19 + globe.gl. Talks to the FastAPI backend in
`src/api/server.py` via the typed client in `lib/api.ts`.

### Run modes

**Dev split (recommended while iterating):**

```bash
# terminal 1 — backend on :8000
uvicorn src.api.server:app --reload --reload-dir src
# (or: python -m src.api.server)

# terminal 2 — frontend on :3000
cd frontend/frontend
cp .env.example .env.local      # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm install
npm run dev
```

**Prod static (FastAPI serves the built UI):**

```bash
scripts/build_frontend.sh       # writes frontend/frontend/out/
uvicorn src.api.server:app      # serves out/ at /, legacy UI at /legacy/
```

`next.config.ts` only enables `output: "export"` when `NEXT_OUTPUT=export` is
set, so `npm run dev` keeps the full dev server.

### API access

Always go through `lib/api.ts` — never hardcode `fetch("/api/...")` in components.
The base URL is read from `NEXT_PUBLIC_API_BASE_URL` (empty = same origin).

Endpoints exposed by the backend (see `src/api/server.py`):

- `GET  /api/scenes`
- `GET  /api/scenes/{scene_id}`
- `GET  /api/scenes/{scene_id}/detections` (GeoJSON)
- `POST /api/forecast`
- `GET  /api/forecast/{cache_key}/paths` (GeoJSON)
- `GET  /api/forecast/{cache_key}/final` (GeoJSON)
- `GET  /api/forecast/{cache_key}/czml`

### Components today

- `app/page.tsx` — shell layout
- `components/Globe/Globe.tsx` — globe.gl with **mock** seeds/arcs. Swap to
  `getSceneDetections` / `getForecastPaths` when wiring real data.
- `components/Layout/{Sidebar,Topbar,Card}.tsx` — cosmetic, hardcoded copy.
