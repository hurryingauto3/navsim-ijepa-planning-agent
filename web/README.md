# NavSim Showcase Web Stack

This workspace contains a browser-first showcase for NavSim research deliverables. It is organized as a small monorepo:

- `frontend/` – Next.js (App Router) client with deck.gl visualization and ONNX runtime integration.
- `backend/` – FastAPI service exposing REST discovery endpoints and a websocket streaming API for simulations.
- `data/` – Scene manifest and cached playback data shared by the frontend and backend.
- `scripts/` – Utilities for exporting lightweight ONNX models and uploading cached runs to object storage.
- `infra/` – Docker and Compose definitions for local and remote deployment.

## Local Development

### Backend (FastAPI)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install "fastapi[standard]" pydantic==2.* numpy==2.* python-socketio[asgi]==5.* redis==5.* boto3==1.* uvicorn==0.*
uvicorn main:app --reload --port 8000
```

Key endpoints:
- `GET /health` – readiness probe.
- `GET /models` / `GET /scenes` – discovery for available models and scenes.
- `GET /eval?model=IJEPA-MLP&scene_token=scene_001` – aggregated metrics lookup.
- `WS /ws/run` – stream cached (or stubbed live) ticks.

### Frontend (Next.js)

```bash
cd ../frontend
corepack enable
pnpm install
pnpm dev --port 3000
```

If `pnpm` is unavailable, run `npm install -g pnpm` first. The client reads its configuration from `.env.local` (already populated with local defaults) and expects the backend on `http://localhost:8000`.

### Cached Data

The sample manifest (`data/scenes.manifest.json`) and cached run (`data/cached_runs/scene_001_IJEPA-MLP.json`) drive the initial replay mode. Add additional caches using the naming pattern `<scene>_<model>.json`.

### Scripts

- `scripts/ijepa_mlp_export.py` – exports a dummy IJEPA+MLP ONNX model into `frontend/public/models/`. Skip this unless you enable the **Test ONNX** button.
- `scripts/upload_to_r2.py` – uploads cached runs to Cloudflare R2 (configure `R2_*` environment variables first).

### Docker Compose

```bash
cd ../infra
docker compose up --build
```

This spins up both services with code mounted for rapid iteration (`http://localhost:3000`).

## Acceptance Checklist

1. Load the frontend, select `IJEPA-MLP / scene_001 / Replay`, click **Run** – ego path animates, agent scatter updates, and EPDMS shows live values.
2. Switch mode to **Live** – stubbed server ticks stream; any failure falls back to cached replay.
3. Open DevTools and click **Test ONNX** – observe console log confirming browser inference (only after running the export script).
4. (Optional) Run `docker compose up --build` from `infra/` – the same behavior is available via containers.

Refer to the plan document for milestone-specific acceptance criteria.
