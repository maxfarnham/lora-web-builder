import * as ort from 'onnxruntime-web';

(async () => {
  // 1. Prefer WebGPU, fall back on WASM
  console.log('Creating session');
  ort.env.wasm.wasmPaths = '/';
  console.log('WASM paths:', ort.env.wasm.wasmPaths);

  const session = await ort.InferenceSession.create('/models/t2l_llama_8b_fp32.onnx', {
    executionProviders: ['webgpu', 'wasm']
  });

  // 2. Build dummy input (a = [3x4], b = [4x3])
  const a = new ort.Tensor('float32', Float32Array.from([
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12
  ]), [3, 4]);

  const b = new ort.Tensor('float32', Float32Array.from([
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12
  ]), [4, 3]);

  const feeds = { a, b };

  const dummy = new ort.Tensor('float32', new Float32Array(1024).fill(0), [1,1024]);
  console.log((await session.run({embedding: dummy})).deltas.dims);

  // 3. Run & log
  //const output = await session.run(feeds);
  //console.log('ORT‑Web output:', output.c.data);   // → [4,6]
})();
