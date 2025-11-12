"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { PathLayer, ScatterplotLayer } from "@deck.gl/layers";
import { COORDINATE_SYSTEM, OrthographicView } from "@deck.gl/core";
import { create } from "zustand";

import { stepIJEPA } from "@/lib/onnxRunner";

type Tick = {
  type: string;
  t: number;
  ego: { x: number; y: number; theta: number };
  traj: number[][];
  agents: { id: number; x: number; y: number; theta: number }[];
  metrics: { epdms: number };
  latency_ms: number;
};

type Store = {
  traj: number[][];
  agents: { id: number; x: number; y: number; theta: number }[];
  metrics: Tick["metrics"] | Record<string, never>;
  latencyMs: number | null;
  reset: () => void;
  pushTick: (tick: Tick) => void;
};

const useSim = create<Store>((set) => ({
  traj: [],
  agents: [],
  metrics: {},
  latencyMs: null,
  reset: () => set({ traj: [], agents: [], metrics: {}, latencyMs: null }),
  pushTick: (tick) =>
    set((state) => ({
      traj: tick.ego ? [...state.traj, [tick.ego.x, tick.ego.y]] : state.traj,
      agents: tick.agents,
      metrics: tick.metrics,
      latencyMs: tick.latency_ms,
    })),
}));

type ModelInfo = {
  name: string;
  version: string;
  kind: string;
  latency_hint_ms: number;
  supports_browser: boolean;
};

type SceneInfo = {
  token: string;
  name: string;
  preview_url: string;
  has_stage2: boolean;
  metric_cache_path: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const WS_SCHEME = process.env.NEXT_PUBLIC_WS_SCHEME ?? "ws";
const WS_HOST = process.env.NEXT_PUBLIC_WS_HOST ?? "localhost:8000";

export default function Page() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [scenes, setScenes] = useState<SceneInfo[]>([]);
  const [selModel, setSelModel] = useState<string>("IJEPA-MLP");
  const [selScene, setSelScene] = useState<string>("scene_001");
  const [mode, setMode] = useState<"cached" | "live">("cached");
  const reset = useSim((state) => state.reset);
  const pushTick = useSim((state) => state.pushTick);
  const traj = useSim((state) => state.traj);
  const agents = useSim((state) => state.agents);
  const metrics = useSim((state) => state.metrics);
  const latencyMs = useSim((state) => state.latencyMs);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/models`).then((res) => res.json()).then(setModels).catch(console.error);
    fetch(`${API_BASE}/scenes`).then((res) => res.json()).then(setScenes).catch(console.error);
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const runNativeWS = () => {
    reset();
    wsRef.current?.close();
    const url = `${WS_SCHEME}://${WS_HOST}/ws/run?model=${selModel}&scene_token=${selScene}&mode=${mode}`;
    const sock = new WebSocket(url);
    wsRef.current = sock;
    sock.onmessage = (ev) => {
      const payload = JSON.parse(ev.data) as Tick;
      if (payload.type === "tick") {
        pushTick(payload);
      }
    };
    sock.onerror = (err) => {
      console.error("ws error", err);
    };
  };

  const layers = useMemo(
    () => [
      new PathLayer({
        id: "ego-path",
        data: [traj],
        getPath: (d) => d,
        getWidth: 2,
        widthMinPixels: 2,
        getColor: () => [255, 255, 0],
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      }),
      new ScatterplotLayer({
        id: "agents",
        data: agents,
        getPosition: (d) => [d.x, d.y, 0],
        getRadius: () => 0.5,
        radiusMinPixels: 3,
        getFillColor: () => [80, 200, 255],
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      }),
    ],
    [traj, agents]
  );

  return (
    <main className="p-4 grid gap-4 grid-cols-1 lg:grid-cols-4">
      <section className="lg:col-span-3 h-[70vh] rounded-xl overflow-hidden">
        <DeckGL
          initialViewState={{ target: [0, 0, 0], zoom: 8, rotationX: 0, rotationOrbit: 0 }}
          controller
          layers={layers}
          views={[new OrthographicView({ id: "ortho" })]}
        />
      </section>
      <section className="lg:col-span-1 space-y-3">
        <h2 className="text-xl font-semibold">Controls</h2>
        <div className="space-y-2">
          <label className="block text-sm uppercase tracking-wide text-neutral-400">Model</label>
          <select
            value={selModel}
            onChange={(event) => setSelModel(event.target.value)}
            className="w-full bg-neutral-800 p-2 rounded"
          >
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
          <label className="block text-sm uppercase tracking-wide text-neutral-400">Scene</label>
          <select
            value={selScene}
            onChange={(event) => setSelScene(event.target.value)}
            className="w-full bg-neutral-800 p-2 rounded"
          >
            {scenes.map((s) => (
              <option key={s.token} value={s.token}>
                {s.name}
              </option>
            ))}
          </select>
          <label className="block text-sm uppercase tracking-wide text-neutral-400">Mode</label>
          <select
            value={mode}
            onChange={(event) => setMode(event.target.value as "cached" | "live")}
            className="w-full bg-neutral-800 p-2 rounded"
          >
            <option value="cached">Replay (cached)</option>
            <option value="live">Live (server)</option>
          </select>
        </div>
        <div className="flex flex-col gap-2">
          <button
            onClick={runNativeWS}
            className="mt-2 bg-white text-black px-3 py-2 rounded font-medium"
          >
            Run
          </button>
          <button
            onClick={async () => {
              const result = await stepIJEPA(new Float32Array(512));
              // eslint-disable-next-line no-console
              console.log("onnx output", result[0]);
            }}
            className="bg-neutral-800 px-3 py-2 rounded"
          >
            Test ONNX
          </button>
        </div>
        <div className="pt-4 text-sm space-y-1">
          <div className="opacity-80">EPDMS: {"epdms" in metrics ? metrics.epdms : "-"}</div>
          <div className="opacity-60">Latency(ms): {latencyMs ?? "-"}</div>
        </div>
      </section>
    </main>
  );
}
