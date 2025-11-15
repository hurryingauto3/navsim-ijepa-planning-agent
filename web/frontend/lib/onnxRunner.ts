// Only import onnxruntime-web on the client side
let ort: typeof import("onnxruntime-web") | null = null;

async function getOrt() {
  if (typeof window === "undefined") {
    throw new Error("ONNX Runtime can only be used in the browser");
  }
  if (!ort) {
    ort = await import("onnxruntime-web");
  }
  return ort;
}

let session: any = null;

export async function loadIJEPA() {
  if (typeof window === "undefined") {
    throw new Error("ONNX Runtime can only be used in the browser");
  }
  if (session) {
    return session;
  }
  try {
    const ortModule = await getOrt();
    session = await ortModule.InferenceSession.create("/models/ijepa_mlp.onnx", {
      executionProviders: ["webgpu", "wasm"],
    });
    return session;
  } catch (error) {
    session = null;
    throw new Error(
      "Failed to load ijepa_mlp.onnx. Run scripts/ijepa_mlp_export.py or copy a model into frontend/public/models/."
    );
  }
}

export async function stepIJEPA(inputVec: Float32Array) {
  if (typeof window === "undefined") {
    throw new Error("ONNX Runtime can only be used in the browser");
  }
  const sess = await loadIJEPA();
  const ortModule = await getOrt();
  const feeds: Record<string, any> = {
    x: new ortModule.Tensor("float32", inputVec, [1, inputVec.length]),
  };
  const output = await sess.run(feeds);
  return output["y"].data as Float32Array;
}
