"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { create } from "zustand";

// Dynamically import DeckGL to prevent SSR issues
const DeckGL = dynamic(() => import("@deck.gl/react").then((mod) => mod.default), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full">
      <p className="text-neutral-400">Loading visualization...</p>
    </div>
  ),
});

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
  const [selModel, setSelModel] = useState("IJEPA-MLP");
  const [selScene, setSelScene] = useState("scene_001");
  const [mode, setMode] = useState<"cached" | "live">("cached");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [webglError, setWebglError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [deckModules, setDeckModules] = useState<any>(null);
  const [webglAvailable, setWebglAvailable] = useState(false);

  const reset = useSim((state) => state.reset);
  const pushTick = useSim((state) => state.pushTick);
  const traj = useSim((state) => state.traj);
  const agents = useSim((state) => state.agents);
  const metrics = useSim((state) => state.metrics);
  const latencyMs = useSim((state) => state.latencyMs);
  const wsRef = useRef<WebSocket | null>(null);

  // Load deck.gl modules dynamically after mount
  useEffect(() => {
    setMounted(true);
    Promise.all([
      import("@deck.gl/layers"),
      import("@deck.gl/core"),
    ]).then(([layers, core]) => {
      setDeckModules({
        PathLayer: layers.PathLayer,
        ScatterplotLayer: layers.ScatterplotLayer,
        COORDINATE_SYSTEM: core.COORDINATE_SYSTEM,
        OrthographicView: core.OrthographicView,
      });
    }).catch((err) => {
      console.error("Failed to load deck.gl modules:", err);
      setWebglError("Failed to load visualization modules");
    });
  }, []);

  // Fetch models and scenes
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const [modelsRes, scenesRes] = await Promise.all([
          fetch(`${API_BASE}/models`),
          fetch(`${API_BASE}/scenes`),
        ]);

        if (!modelsRes.ok || !scenesRes.ok) {
          throw new Error(`Backend responded with ${modelsRes.status}/${scenesRes.status}`);
        }

        const [modelsJson, scenesJson] = await Promise.all([modelsRes.json(), scenesRes.json()]);
        if (cancelled) return;
        setModels(modelsJson);
        setScenes(scenesJson);
        setLoadError(null);
      } catch (err) {
        if (!cancelled) {
          console.error("failed to load discovery endpoints", err);
          setLoadError(
            `Failed to reach backend at ${API_BASE}. Ensure the FastAPI service is running and accessible.`,
          );
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  // Close websocket on unmount
  useEffect(() => () => wsRef.current?.close(), []);

  // WebGL availability check
  useEffect(() => {
    try {
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
      if (!gl) {
        setWebglError("WebGL is not available in this browser");
        setWebglAvailable(false);
        return;
      }
      // Check if WebGL context has required properties
      if (gl && typeof gl.getParameter === "function") {
        try {
          // Try to access a WebGL parameter to ensure context is valid
          gl.getParameter(gl.VERSION);
          const debugInfo = (gl as WebGLRenderingContext).getExtension("WEBGL_debug_renderer_info");
          if (debugInfo) {
            const vendor = (gl as WebGLRenderingContext).getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
            const renderer = (gl as WebGLRenderingContext).getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            console.log("WebGL vendor:", vendor, "renderer:", renderer);
          }
          setWebglAvailable(true);
          setWebglError(null);
        } catch (e) {
          setWebglError("WebGL context is not fully initialized");
          setWebglAvailable(false);
        }
      } else {
        setWebglError("WebGL context is invalid");
        setWebglAvailable(false);
      }
    } catch (err) {
      setWebglError(`WebGL check failed: ${err instanceof Error ? err.message : String(err)}`);
      setWebglAvailable(false);
    }
  }, []);

  // Suppress benign deck.gl resize errors and handle runtime errors
  useEffect(() => {
    const suppressedTokens = ["maxTextureDimension2D", "canvas-context", "getMaxDrawingBufferSize"];

    const shouldSuppress = (message?: string | Event | null, source?: string | null) => {
      const msg = typeof message === "string" ? message : message ? String(message) : "";
      const src = source ? String(source) : "";
      return suppressedTokens.some((token) => 
        msg.toLowerCase().includes(token.toLowerCase()) || 
        src.toLowerCase().includes(token.toLowerCase())
      );
    };

    const handleError = (event: ErrorEvent) => {
      const message = event.message || "";
      const filename = event.filename || "";
      
      if (shouldSuppress(message, filename)) {
        console.warn("Suppressed WebGL/deck.gl error:", message, filename);
        event.preventDefault();
        event.stopPropagation();
        return true;
      }
      
      // Log other errors but don't break the app
      if (message.includes("WebGL") || message.includes("deck") || message.includes("canvas")) {
        console.warn("WebGL-related error (non-fatal):", message);
        event.preventDefault();
        return true;
      }
      
      return false;
    };

    const handleRejection = (event: PromiseRejectionEvent) => {
      const reason = event.reason;
      const msg = reason?.message ?? String(reason ?? "");
      if (shouldSuppress(msg, "") || msg.includes("maxTextureDimension2D")) {
        console.warn("Suppressed promise rejection:", msg);
        event.preventDefault();
        return true;
      }
      return false;
    };

    // Add error handler with capture phase
    window.addEventListener("error", handleError, true);
    window.addEventListener("unhandledrejection", handleRejection as any, true);

    return () => {
      window.removeEventListener("error", handleError, true);
      window.removeEventListener("unhandledrejection", handleRejection as any, true);
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

  const initialViewState = useMemo(() => ({
    ortho: {
      target: [0, 0, 0] as [number, number, number],
      zoom: 8,
      rotationX: 0,
      rotationOrbit: 0,
    },
  }), []);

  const layers = useMemo(() => {
    if (!deckModules) return [];
    
    const { PathLayer, ScatterplotLayer, COORDINATE_SYSTEM } = deckModules;
    
    return [
      new PathLayer({
        id: "ego-path",
        data: [traj],
        getPath: (d: any) => d,
        getWidth: 2,
        widthMinPixels: 2,
        getColor: () => [255, 255, 0],
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      }),
      new ScatterplotLayer({
        id: "agents",
        data: agents,
        getPosition: (d: any) => [d.x, d.y, 0],
        getRadius: () => 0.5,
        radiusMinPixels: 3,
        getFillColor: () => [80, 200, 255],
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      }),
    ];
  }, [traj, agents, deckModules]);

  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center space-y-4 mb-8">
          <h1 className="text-3xl font-bold text-white">Interactive Planning Demo</h1>
          <p className="text-neutral-400 max-w-2xl mx-auto">
            Explore real-time trajectory planning for autonomous vehicles. Select a model and scene,
            then watch as the agent navigates complex driving scenarios.
          </p>
        </div>
      </div>

      {/* Demo Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        <div className="grid gap-4 grid-cols-1 lg:grid-cols-4">
          <section className="lg:col-span-3 h-[70vh] rounded-xl overflow-hidden">
            <div className="relative h-full w-full bg-neutral-900">
              {!mounted || !deckModules ? (
                <div className="flex items-center justify-center h-full p-8 text-center">
                  <p className="text-neutral-400">Loading visualization...</p>
                </div>
              ) : !webglAvailable || webglError ? (
                <div className="flex items-center justify-center h-full p-8 text-center">
                  <div className="space-y-2">
                    <p className="text-red-400 font-semibold">WebGL Error</p>
                    <p className="text-neutral-400 text-sm">{webglError || "WebGL is not available"}</p>
                    <p className="text-neutral-500 text-xs mt-4">
                      Try enabling hardware acceleration or using a different browser.
                    </p>
                  </div>
                </div>
              ) : (
                <DeckGL
                  {...({
                    style: { position: "absolute", inset: "0" },
                    glOptions: {
                      preserveDrawingBuffer: true,
                      powerPreference: "high-performance",
                      failIfMajorPerformanceCaveat: false,
                      antialias: false,
                      stencil: false,
                      depth: true,
                    },
                    initialViewState: initialViewState,
                    controller: true,
                    layers: layers,
                    views: [new deckModules.OrthographicView({ id: "ortho" })],
                    onError: (error: any) => {
                      const message = error?.message ?? String(error ?? "");
                      const errorStr = String(error ?? "");
                      // Suppress known benign errors
                      if (
                        message.includes("maxTextureDimension2D") ||
                        message.includes("canvas-context") ||
                        errorStr.includes("maxTextureDimension2D") ||
                        errorStr.includes("canvas-context")
                      ) {
                        console.warn("Suppressed deck.gl error:", message || errorStr);
                        return;
                      }
                      console.error("DeckGL error:", error);
                      setWebglError(message || errorStr);
                    },
                    onBeforeRender: () => {
                      // Ensure WebGL context is valid before rendering
                      try {
                        const canvas = document.querySelector("canvas");
                        if (canvas) {
                          const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
                          if (!gl) {
                            throw new Error("WebGL context lost");
                          }
                        }
                      } catch (e) {
                        console.warn("WebGL context check failed:", e);
                      }
                    },
                  } as any)}
                />
              )}
            </div>
          </section>

          <section className="lg:col-span-1 space-y-3">
            <h2 className="text-xl font-semibold">Controls</h2>
            {loadError ? (
              <div className="rounded border border-red-500/40 bg-red-950/40 p-3 text-sm text-red-200">
                {loadError}
              </div>
            ) : null}
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
                  try {
                    // Dynamically import ONNX runner to avoid SSR issues
                    const { stepIJEPA } = await import("@/lib/onnxRunner");
                    const result = await stepIJEPA(new Float32Array(512));
                    // eslint-disable-next-line no-console
                    console.log("onnx output", result[0]);
                  } catch (err) {
                    console.error(err);
                    alert(
                      err instanceof Error
                        ? err.message
                        : "Failed to run ONNX inference. Ensure the model file exists.",
                    );
                  }
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
        </div>
      </div>
    </main>
  );
}
