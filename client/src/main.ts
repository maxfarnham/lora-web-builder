import * as ort from 'onnxruntime-web';
import { pipeline } from '@huggingface/transformers';
import { runT2LValidation } from './t2l-validation.js';

export async function embed1024(text: string): Promise<Float32Array> {
  // Use the transformers.js pipeline for E5 embeddings
  const extractor = await pipeline('feature-extraction', 'Xenova/e5-large-v2');
  
  // E5 models expect text to be prefixed with "query: " for queries
  const prefixedText = `query: ${text}`;
  
  // Get embeddings
  const embeddings = await extractor(prefixedText, { pooling: 'mean', normalize: true });
  
  // Convert to Float32Array (embeddings is typically a nested array)
  const embeddingArray = embeddings.data || (embeddings as any)[0] || embeddings;
  
  // Ensure it's 1024 dimensions - pad or truncate as needed
  const out = new Float32Array(1024);
  const sourceLength = Math.min(embeddingArray.length, 1024);
  
  for (let i = 0; i < sourceLength; i++) {
    out[i] = embeddingArray[i];
  }
  
  return out;
}

// Simple demo function for testing the t2l model
export async function testT2LModel() {
  try {
    // Setup ONNX Runtime
    ort.env.wasm.wasmPaths = '/';
    
    // Load the t2l model
    const t2l = await ort.InferenceSession.create('/models/t2l_llama_8b_fp32.onnx', {
      executionProviders: ['webgpu', 'wasm']
    });

    console.log('T2L Model loaded successfully!');
    console.log('Input names:', t2l.inputNames);
    console.log('Output names:', t2l.outputNames);

    // Test with simple text
    const text = 'Hello, world!';
    const embedding = await embed1024(text);
    
    // Run through t2l model
    const input = new ort.Tensor('float32', embedding, [1, 1024]);
    const { deltas } = await t2l.run({ embedding: input });
    
    console.log('T2L output dimensions:', deltas.dims);
    console.log('T2L output type:', deltas.type);
    console.log('T2L output data length:', deltas.data.length);
    
    return deltas;
  } catch (error) {
    console.error('Error testing T2L model:', error);
    throw error;
  }
}

// Run comprehensive T2L validation when the module loads
(async () => {
  console.log('🔧 Initializing T2L Model Validation...');
  await testT2LModel();
  
  // More comprehensive validation is in t2l-validation.ts
  //const validationPassed = await runT2LValidation();
})();
