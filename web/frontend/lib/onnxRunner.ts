import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

export async function loadIJEPA() {
  if (!session) {
    session = await ort.InferenceSession.create("/models/ijepa_mlp.onnx", {
      executionProviders: ["webgpu", "wasm"],
    });
  }
  return session;
}

export async function stepIJEPA(inputVec: Float32Array) {
  const sess = await loadIJEPA();
  const feeds: Record<string, ort.Tensor> = {
    x: new ort.Tensor("float32", inputVec, [1, inputVec.length]),
  };
  const output = await sess.run(feeds);
  return output["y"].data as Float32Array;
}
