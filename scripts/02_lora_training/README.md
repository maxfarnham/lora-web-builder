# Stage 2: LoRA Training

## What This Stage Does

This stage trains individual LoRA (Low-Rank Adaptation) adapters for each personality trait by:
1. **Uploading training shards** to Together AI or preparing for local training
2. **Launching LoRA training jobs** for each trait (e.g., `wisdom_plus`, `intellect_minus`)
3. **Monitoring training progress** and downloading completed LoRA adapters
4. **Preparing hypernetwork training data** with trait-to-LoRA mappings

## Prerequisites

- [ ] **Completed Stage 1**: Training shards in `../01_dataset_creation/outputs/shards/together_format/`
- [ ] **Environment Variables**:
  - `TOGETHER_API_KEY` - For cloud-based LoRA training
- [ ] **Dependencies**: `pip install -U together transformers peft safetensors accelerate`
- [ ] **Hardware** (for local training): NVIDIA GPU with 16GB+ VRAM

## Quick Start

### Option 1: Cloud Training (Recommended)
```bash
# Train all trait LoRAs on Together AI
python train_trait_loras.py --all

# Or train specific traits only
python train_trait_loras.py wisdom_minus intellect_minus

# Monitor and download completed LoRAs
# (script automatically polls and downloads)
```

### Option 2: Local Training & Testing
```bash
# Train a single LoRA locally for testing
python run_lora_local.py --json path/to/lora_weights.json --gpu

# Train all LoRAs using cloud API
python run_lora_cloud.py
```

## Expected Outputs

- `outputs/lora_checkpoints/` - Individual LoRA adapter files (`.safetensors`)
- `outputs/lora_job_ids.json` - Together AI job tracking
- `outputs/file_hashes.json` - Deduplication tracking
- `hypernetwork_data.json` - Trait-to-LoRA mapping for Stage 3

## Command Options

### `train_trait_loras.py`
- `--all` - Train all available trait combinations
- `--list-available` - Show available trait shards
- `--force-retrain` - Ignore existing job cache and retrain
- `[trait_names...]` - Train only specified traits

### Training Parameters
- **Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct-Reference`
- **LoRA Rank**: 4 (sufficient for style tasks)
- **LoRA Alpha**: 16 (2× rank heuristic)
- **Target Modules**: `q_proj,v_proj` (query and value projections only)
- **Epochs**: 1
- **Batch Size**: 32
- **Learning Rate**: 2e-4

## File Descriptions

- **`train_trait_loras.py`** - Main cloud-based LoRA training orchestrator
- **`run_lora_cloud.py`** - Alternative cloud training approach
- **`run_lora_local.py`** - Local LoRA training and testing utilities

## Training Strategies

### Deduplication & Caching
- **File Hash Tracking**: Skips retraining if same input file already processed
- **Job Resume**: Automatically resumes monitoring existing jobs
- **Force Retrain**: Use `--force-retrain` to override cache

### Trait Selection
```bash
# Train only high-priority traits
python train_trait_loras.py wisdom_plus wisdom_minus intellect_plus intellect_minus

# Train all emotional traits
python train_trait_loras.py joy_plus joy_minus neuroticism_plus neuroticism_minus

# Full training run
python train_trait_loras.py --all
```

## Next Steps

Continue to **Stage 3: Hypernetwork Training** with the completed LoRA adapters:
```bash
cd ../03_hypernetwork_training
python train_hypernetwork_cluster.py
```

## Troubleshooting

- **API Rate Limits**: Together AI has per-account limits; stagger job submissions
- **Failed Jobs**: Check logs in Together AI dashboard; may need different hyperparameters
- **Local GPU Issues**: Ensure CUDA drivers and PyTorch GPU support installed
- **Missing Shards**: Verify Stage 1 completed successfully and `together_format/` exists 