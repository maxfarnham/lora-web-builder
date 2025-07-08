# Stage 1: Trait Corpus Creation

## What This Stage Does

This stage creates a comprehensive training dataset with trait annotations by:
1. **Bootstrapping core trait corpus** from emotion-labeled datasets (EmpatheticDialogues, DailyDialog, etc.)
2. **Generating synthetic seed data** from neutral utterances
3. **Creating synthetic trait-specific data** using OpenAI API paraphrasing
4. **Bucketizing traits** into positive/negative poles for training
5. **Converting to Together AI format** for cloud-based LoRA training

## Prerequisites

- [ ] **Environment Variables**:
  - `TRAIT_CORPUS_DIR` - Directory containing emotion datasets
  - `OPENAI_API_KEY` - For synthetic data generation
- [ ] **Dependencies**: `pip install -U datasets tqdm pandas hf_transfer duckdb openai datasketch`
- [ ] **Input Data**: Access to HuggingFace emotion datasets

## Quick Start

### Option 1: Full Pipeline (Recommended)
```bash
# 1. Create core trait corpus from emotion datasets
python bootstrap_trait_corpus.py --output_dir outputs/core_traits --output_format parquet

# 2. Generate synthetic seeds from neutral data
python generate_synthetic_seeds.py --output outputs/seeds/neutral_seeds.jsonl --max_samples 5000

# 3. Create synthetic trait-specific data (requires OpenAI API)
python synthetic_sparse_buckets.py --seeds outputs/seeds/ --target_per_bucket 2000 --output_dir outputs/shards/

# 4. Bucketize traits into positive/negative poles
python bucketize_traits.py --in_file outputs/core_traits/trait_corpus.jsonl --out_dir outputs/shards/

# 5. Convert to Together AI training format
python convert_to_together_format.py
```

### Option 2: Core Traits Only (Faster)
```bash
# Skip synthetic generation, use only emotion datasets
python bootstrap_trait_corpus.py --output_dir outputs/core_traits
python bucketize_traits.py --in_file outputs/core_traits/trait_corpus.jsonl --out_dir outputs/shards/
python convert_to_together_format.py
```

## Expected Outputs

- `outputs/shards/` - Training data shards by trait (e.g., `wisdom_plus.jsonl`, `intellect_minus.jsonl`)
- `outputs/shards/together_format/` - Together AI formatted training files
- `outputs/seeds/` - Neutral seed data for synthetic generation
- `outputs/core_traits/` - Core emotion-to-trait mapped corpus

## File Descriptions

- **`bootstrap_trait_corpus.py`** - Main corpus creation from emotion datasets
- **`generate_synthetic_seeds.py`** - Extract neutral utterances as paraphrasing seeds
- **`synthetic_sparse_buckets.py`** - Generate diverse synthetic trait data using OpenAI
- **`bucketize_traits.py`** - Split corpus into trait-specific training shards
- **`convert_to_together_format.py`** - Convert to Together AI fine-tuning format
- **`emotion_trait_mappings.json`** - Configuration mapping emotions to personality traits

## Next Steps

Continue to **Stage 2: LoRA Training** with the generated training shards:
```bash
cd ../02_lora_training
python train_trait_loras.py
```

## Troubleshooting

- **Missing datasets**: Ensure HuggingFace datasets are accessible and `TRAIT_CORPUS_DIR` is set
- **OpenAI rate limits**: Reduce `--concurrency` in `synthetic_sparse_buckets.py`
- **Small buckets**: Lower `MIN_PER_BUCKET` threshold in `bucketize_traits.py` 