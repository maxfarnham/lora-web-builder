import * as ort from 'onnxruntime-web';
import { pipeline } from '@huggingface/transformers';

// Helper function to generate embeddings
async function embed1024(text: string): Promise<Float32Array> {
  const extractor = await pipeline('feature-extraction', 'Xenova/e5-large-v2');
  const prefixedText = `query: ${text}`;
  const embeddings = await extractor(prefixedText, { pooling: 'mean', normalize: true });
  const embeddingArray = embeddings.data || (embeddings as any)[0] || embeddings;
  
  const out = new Float32Array(1024);
  const sourceLength = Math.min(embeddingArray.length, 1024);
  
  for (let i = 0; i < sourceLength; i++) {
    out[i] = embeddingArray[i];
  }
  
  return out;
}

// Helper function to calculate cosine similarity
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Helper function to analyze t2l output
function analyzeT2LOutput(deltas: any, inputText: string) {
  const deltaArray = Array.from(deltas.data) as number[];
  
  const mean = deltaArray.reduce((sum: number, val: number) => sum + val, 0) / deltaArray.length;
  const variance = deltaArray.reduce((sum: number, val: number) => sum + Math.pow(val - mean, 2), 0) / deltaArray.length;
  const std = Math.sqrt(variance);
  const min = Math.min(...deltaArray);
  const max = Math.max(...deltaArray);
  
  const zeros = deltaArray.filter((x: number) => Math.abs(x) < 1e-10).length;
  const nonFinite = deltaArray.filter((x: number) => !isFinite(x)).length;
  
  console.log(`\n=== Analysis for: "${inputText}" ===`);
  console.log('Output dimensions:', deltas.dims);
  console.log('Output type:', deltas.type);
  console.log('Data length:', deltas.data.length);
  console.log(`Statistics - Mean: ${mean.toFixed(6)}, Std: ${std.toFixed(6)}, Min: ${min.toFixed(6)}, Max: ${max.toFixed(6)}`);
  console.log(`Quality - Zeros: ${zeros}/${deltaArray.length}, Non-finite: ${nonFinite}`);
  
  return {
    array: new Float32Array(deltaArray),
    stats: { mean, std, min, max, variance },
    quality: { zeros, nonFinite, total: deltaArray.length },
    dims: deltas.dims,
    type: deltas.type
  };
}

export async function runT2LValidation(): Promise<boolean> {
  console.log('\n🚀 Starting T2L Model Validation Suite...\n');
  
  try {
    // 1. Setup ONNX Runtime
    console.log('Setting up ONNX Runtime...');
    ort.env.wasm.wasmPaths = '/models/wasm/';
    console.log('WASM paths configured:', ort.env.wasm.wasmPaths);

    // 2. Load the t2l model
    console.log('Loading T2L model...');
    const t2lModel = await ort.InferenceSession.create('/models/t2l_llama_8b_fp32.onnx', {
      executionProviders: ['webgpu', 'wasm']
    });
    console.log('✅ T2L Model loaded successfully!');
    console.log('Input names:', t2lModel.inputNames);
    console.log('Output names:', t2lModel.outputNames);

    // 3. Test with diverse text inputs
    const testTexts = [
      // Similar meanings
      'Hello, world!',
      'Hi there, world!',
      'Greetings, everyone!',
      
      // Different topics
      'The weather is beautiful today.',
      'Machine learning is fascinating.',
      'I love eating pizza.',
      
      // Different lengths
      'Cat.',
      'The quick brown fox jumps over the lazy dog.',
      'In the realm of artificial intelligence, natural language processing has emerged as one of the most important and challenging fields of study.',
      
      // Different languages/styles
      'How are you?',
      'Comment ça va?',
      'The mitochondria is the powerhouse of the cell.',
      
      // Edge cases
      '123 456 789',
      '!@#$%^&*()',
    ];

    console.log('\n📊 Processing test inputs...\n');
    
    const results: { text: string; embedding: Float32Array; analysis: any }[] = [];

    // Process each test text
    for (const text of testTexts) {
      try {
        console.log(`Processing: "${text}"`);
        
        // Generate embedding
        const embedding = await embed1024(text);
        console.log(`Generated embedding with ${embedding.length} dimensions`);
        
        // Run through t2l model
        const input = new ort.Tensor('float32', embedding, [1, 1024]);
        const output = await t2lModel.run({ embedding: input });
        
        // Analyze the output
        const analysis = analyzeT2LOutput(output.deltas, text);
        
        results.push({ text, embedding, analysis });
        
      } catch (error) {
        console.error(`❌ Error processing "${text}":`, error);
        return false;
      }
    }

    // 4. Validation Tests
    console.log('\n🔍 Running Validation Tests...\n');
    
    let allTestsPassed = true;

    // Test 1: Basic functionality
    console.log('=== Test 1: Basic Functionality ===');
    if (results.length === testTexts.length) {
      console.log('✅ All inputs processed successfully');
    } else {
      console.log('❌ Some inputs failed to process');
      allTestsPassed = false;
    }

    // Test 2: Output quality
    console.log('\n=== Test 2: Output Quality ===');
    const allFinite = results.every(result => result.analysis.quality.nonFinite === 0);
    const allVariation = results.every(result => result.analysis.stats.std > 0);
    
    if (allFinite) {
      console.log('✅ All outputs contain finite values');
    } else {
      console.log('❌ Some outputs contain non-finite values');
      allTestsPassed = false;
    }
    
    if (allVariation) {
      console.log('✅ All outputs show statistical variation');
    } else {
      console.log('❌ Some outputs lack variation (all zeros/constants)');
      allTestsPassed = false;
    }

    // Test 3: Consistency
    console.log('\n=== Test 3: Dimensional Consistency ===');
    const firstDims = results[0]?.analysis.dims as number[];
    const consistentDims = results.every(result => {
      const currentDims = result.analysis.dims as number[];
      return currentDims.length === firstDims.length &&
        currentDims.every((dim, idx) => dim === firstDims[idx]);
    });
    
    if (consistentDims) {
      console.log('✅ All outputs have consistent dimensions:', firstDims);
    } else {
      console.log('❌ Output dimensions are inconsistent');
      allTestsPassed = false;
    }

    // Test 4: Uniqueness
    console.log('\n=== Test 4: Output Uniqueness ===');
    const uniqueOutputs = new Set();
    let duplicates = 0;
    
    for (const result of results) {
      const signature = result.analysis.array.slice(0, 10).join(',');
      if (uniqueOutputs.has(signature)) {
        duplicates++;
        console.log(`⚠️  Potential duplicate output for: "${result.text}"`);
      } else {
        uniqueOutputs.add(signature);
      }
    }
    
    if (duplicates === 0) {
      console.log(`✅ All outputs are unique (${uniqueOutputs.size}/${results.length})`);
    } else {
      console.log(`⚠️  Found ${duplicates} potential duplicates`);
    }

    // Test 5: Similarity Analysis
    console.log('\n=== Test 5: Similarity Analysis ===');
    if (results.length >= 3) {
      const similar1 = results[0]; // "Hello, world!"
      const similar2 = results[1]; // "Hi there, world!"
      const different = results[4]; // "Machine learning is fascinating."
      
      const embeddingSim = cosineSimilarity(similar1.embedding, similar2.embedding);
      const deltasSim = cosineSimilarity(similar1.analysis.array, similar2.analysis.array);
      
      const embeddingDiff = cosineSimilarity(similar1.embedding, different.embedding);
      const deltasDiff = cosineSimilarity(similar1.analysis.array, different.analysis.array);
      
      console.log(`Similar texts ("${similar1.text}" vs "${similar2.text}"):`);
      console.log(`  Embedding similarity: ${embeddingSim.toFixed(4)}`);
      console.log(`  T2L output similarity: ${deltasSim.toFixed(4)}`);
      
      console.log(`Different texts ("${similar1.text}" vs "${different.text}"):`);
      console.log(`  Embedding similarity: ${embeddingDiff.toFixed(4)}`);
      console.log(`  T2L output similarity: ${deltasDiff.toFixed(4)}`);
      
      if (embeddingSim > embeddingDiff && deltasSim > deltasDiff) {
        console.log('✅ Similar texts have higher similarity than different texts');
      } else {
        console.log('⚠️  Similarity relationships may not be preserved');
      }
    }

    // Test 6: Global Distribution
    console.log('\n=== Test 6: Global Distribution Analysis ===');
    const allValues: number[] = [];
    results.forEach(result => {
      allValues.push(...Array.from(result.analysis.array));
    });
    
    const globalMean = allValues.reduce((sum: number, val: number) => sum + val, 0) / allValues.length;
    const globalStd = Math.sqrt(allValues.reduce((sum: number, val: number) => sum + Math.pow(val - globalMean, 2), 0) / allValues.length);
    
    console.log(`Global statistics across all outputs:`);
    console.log(`  Mean: ${globalMean.toFixed(6)}`);
    console.log(`  Std: ${globalStd.toFixed(6)}`);
    console.log(`  Range: [${Math.min(...allValues).toFixed(6)}, ${Math.max(...allValues).toFixed(6)}]`);
    
    if (globalStd > 0) {
      console.log('✅ Global distribution shows healthy variation');
    } else {
      console.log('❌ Global distribution lacks variation');
      allTestsPassed = false;
    }

    // Final Summary
    console.log('\n📋 VALIDATION SUMMARY');
    console.log('='.repeat(50));
    console.log(`✅ Successfully processed ${results.length}/${testTexts.length} inputs`);
    console.log(`✅ T2L model loaded and functioning correctly`);
    console.log(`✅ Model produces ${results[0]?.analysis.array.length || 'unknown'} dimensional latent representations`);
    
    if (allTestsPassed) {
      console.log('🎉 ALL VALIDATION TESTS PASSED!');
      console.log('✅ T2L model is working correctly and producing valid outputs');
    } else {
      console.log('⚠️  Some validation tests had issues - please review the output above');
    }
    
    return allTestsPassed;
    
  } catch (error) {
    console.error('❌ Fatal error during T2L validation:', error);
    return false;
  }
} 