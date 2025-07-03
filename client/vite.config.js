import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        { src: 'node_modules/onnxruntime-web/dist/ort-wasm*.{wasm,mjs}', dest: '' },
        { src: 'models/*.onnx', dest: 'public/models' }
      ]
    }),
  ],
  build: {
    assetsInlineLimit: 0
  },  
  assetsInclude: ["**/*.onnx"],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});
