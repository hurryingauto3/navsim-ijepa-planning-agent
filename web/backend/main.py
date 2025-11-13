from typing import Any, Dict, List

import asyncio
import json
import os
import pathlib

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = pathlib.Path(__file__).resolve().parent
DATA_ROOT = pathlib.Path(
    os.getenv("DATA_ROOT", HERE.parent / "data")
).resolve()
DATA = DATA_ROOT
CACHED = DATA_ROOT / "cached_runs"

# Verify data directory exists at startup
if not DATA_ROOT.exists():
    import warnings
    warnings.warn(f"DATA_ROOT does not exist: {DATA_ROOT} (resolved from {os.getenv('DATA_ROOT', 'default')})")
if not (DATA_ROOT / "scenes.manifest.json").exists():
    import warnings
    warnings.warn(f"Scene manifest not found at startup: {DATA_ROOT / 'scenes.manifest.json'}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelInfo(BaseModel):
    name: str
    version: str
    kind: str
    latency_hint_ms: int
    supports_browser: bool


class SceneInfo(BaseModel):
    token: str
    name: str
    preview_url: str
    has_stage2: bool
    metric_cache_path: str


@app.get("/health")
def health() -> Dict[str, Any]:
    manifest_exists = (DATA / "scenes.manifest.json").exists()
    return {
        "ok": True,
        "data_root": str(DATA_ROOT),
        "data_root_exists": DATA_ROOT.exists(),
        "manifest_exists": manifest_exists,
        "cached_runs_dir_exists": CACHED.exists(),
    }


@app.get("/models", response_model=List[ModelInfo])
def models() -> List[ModelInfo]:
    return [
        {
            "name": "ConstantVelocity",
            "version": "1.0",
            "kind": "baseline",
            "latency_hint_ms": 1,
            "supports_browser": False,
        },
        {
            "name": "IJEPA-MLP",
            "version": "0.1",
            "kind": "onnx-web",
            "latency_hint_ms": 8,
            "supports_browser": True,
        },
        {
            "name": "TransFuser",
            "version": "0.1",
            "kind": "server",
            "latency_hint_ms": 55,
            "supports_browser": False,
        },
    ]


@app.get("/scenes", response_model=List[SceneInfo])
def scenes() -> List[SceneInfo]:
    manifest_path = DATA / "scenes.manifest.json"
    if not manifest_path.exists():
        import logging
        logging.error(f"Scene manifest not found at {manifest_path} (DATA_ROOT={DATA_ROOT}, resolved={manifest_path.resolve()})")
        raise HTTPException(
            status_code=500,
            detail=f"Scene manifest not found at {manifest_path}. DATA_ROOT={DATA_ROOT}"
        )
    manifest = json.loads(manifest_path.read_text())
    return manifest


@app.get("/eval")
def eval_cached(model: str, scene_token: str) -> Dict[str, Any]:
    path = CACHED / f"{scene_token}_{model}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Cached run not found")
    run = json.loads(path.read_text())
    return {"aggregate": run["aggregate"]}


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket) -> None:
    await ws.accept()
    try:
        q = ws.query_params
        model = q.get("model", "IJEPA-MLP")
        scene = q.get("scene_token", "scene_001")
        mode = q.get("mode", "cached")  # cached|live

        if mode == "live" and model == "TransFuser":
            fps, dt = 10, 0.1
            x = 0.0
            for i in range(100):
                payload = {
                    "type": "tick",
                    "t": i * dt,
                    "ego": {"x": x, "y": 0.0, "theta": 0.0},
                    "traj": [[x + j * 0.8, 0.0, 0.0] for j in range(10)],
                    "agents": [
                        {"id": 1, "x": 5.0 - 0.1 * i, "y": 1.0, "theta": 0.0}
                    ],
                    "metrics": {"epdms": 0.75 + 0.001 * i},
                    "latency_ms": 55,
                }
                await ws.send_json(payload)
                x += 0.8
                await asyncio.sleep(dt)
            await ws.send_json({"type": "done"})
            return

        path = CACHED / f"{scene}_{model}.json"
        if not path.exists():
            await ws.send_json({"type": "error", "msg": "no_cache"})
            await ws.close()
            return

        run = json.loads(path.read_text())
        fps = run.get("fps", 10)
        dt = 1.0 / float(fps)
        for i, tms in enumerate(run["timestamps_ms"]):
            payload = {
                "type": "tick",
                "t": tms / 1000.0,
                "ego": {
                    "x": run["ego_traj"][i][0],
                    "y": run["ego_traj"][i][1],
                    "theta": run["ego_traj"][i][2],
                },
                "traj": run["ego_traj"][max(0, i - 1) : i + 20],
                "agents": [
                    {
                        "id": a["id"],
                        "x": a["track"][i][0],
                        "y": a["track"][i][1],
                        "theta": a["track"][i][2],
                    }
                    for a in run["agents"]
                ],
                "metrics": run["metrics_per_tick"][i],
                "latency_ms": 0,
            }
            await ws.send_json(payload)
            await asyncio.sleep(dt)
        await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
