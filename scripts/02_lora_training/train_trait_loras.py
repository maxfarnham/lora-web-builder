#!/usr/bin/env python
"""
LoRA Fine-tuning Pipeline for Trait-based Models
  1. Upload each trait‑pole JSONL to Together File API
  2. Launch a LoRA fine‑tune job on Llama‑3‑8b‑hf per file
  3. Poll job status; download finished LoRA checkpoints
  4. Prepare data for hypernetwork training (to be run on GPU cluster)

Features:
- Train specific traits only: python train_trait_hypernetwork.py wisdom_minus intellect_minus
- Train all traits (convenience): python train_trait_hypernetwork.py --all
- Automatic deduplication: skips training if same file hash already processed
- Force retrain: python train_trait_hypernetwork.py --force-retrain
- List available traits: python train_trait_hypernetwork.py --list-available

Tested with Together AI Python SDK v1.5+. Author: maxfarnham@gmail.com
"""
import os, time, json, shutil, hashlib, argparse
from pathlib import Path
from tqdm import tqdm
from together import Together

# Initialize Together client
client = Together(api_key=os.environ["TOGETHER_API_KEY"])

CORPUS_DIR     = Path(os.environ["TRAIT_CORPUS_DIR"]).expanduser()
CONVERTED_DIR  = Path("shards/together_format")  # Converted files for Together AI
MODEL_NAME     = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"  # Together AI supported model
HASH_DB_FILE   = Path("file_hashes.json")  # Store file hashes for deduplication

###############################################################################
# Utility functions for deduplication
###############################################################################
def calculate_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_hash_db() -> dict:
    """Load the hash database from file."""
    if HASH_DB_FILE.exists():
        with open(HASH_DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_hash_db(hash_db: dict):
    """Save the hash database to file."""
    with open(HASH_DB_FILE, "w") as f:
        json.dump(hash_db, f, indent=2)

def check_existing_job(file_path: Path, hash_db: dict) -> tuple[str, str] | None:
    """Check if a job already exists for this file hash. Returns (job_id, status) or None."""
    file_hash = calculate_file_hash(file_path)
    
    # Check if we have a record of this hash
    if file_hash in hash_db:
        job_id = hash_db[file_hash]["job_id"]
        try:
            job_info = client.fine_tuning.retrieve(job_id)
            status = str(job_info.status)
            print(f"Found existing job {job_id} for {file_path.name} (hash: {file_hash[:8]}...)")
            return job_id, status
        except Exception as e:
            print(f"Error checking existing job {job_id}: {e}")
            # Remove invalid job from hash db
            del hash_db[file_hash]
            save_hash_db(hash_db)
    
    return None

###############################################################################
# 1. Upload JSONL shards & launch LoRA fine‑tunes
###############################################################################
def upload_file(path: Path) -> str:
    """Upload a JSONL file to Together AI and return the file ID."""
    print(f"Uploading {path.name}...")
    result = client.files.upload(file=str(path))
    return result.id

def launch_lora_job(file_id: str, suffix: str) -> str:
    """Launch a LoRA fine-tuning job and return the job ID."""
    print(f"Launching LoRA job for {suffix}...")
    job = client.fine_tuning.create(
        training_file=file_id,
        model=MODEL_NAME,
        lora=True,  # Enable LoRA fine-tuning
        n_epochs=1,
        batch_size=32,
        learning_rate=2e-4,
        suffix=suffix,
        n_checkpoints=1,
        # LoRA-specific parameters optimized for style tasks
        lora_r=4,  # LoRA rank: 4 sufficient for style tasks (max 8 if trait is subtle)
        lora_alpha=16,  # LoRA alpha: 2 × r heuristic
        lora_trainable_modules="q_proj,v_proj",  # Only query and value projections
    )
    return job.id

def step1_launch_all_jobs(specific_traits: list[str] = None):
    """Upload JSONL files and launch fine-tuning jobs.
    
    Args:
        specific_traits: List of specific trait names to train (e.g., ['wisdom_minus', 'intellect_minus'])
                        If None, trains all available traits.
    """
    # Check if converted files exist
    if not CONVERTED_DIR.exists():
        print(f"❌ Converted files not found at {CONVERTED_DIR}")
        print("Please run the conversion script first:")
        print("  python scripts/py/convert_to_together_format.py")
        return {}
    
    # Load hash database for deduplication
    hash_db = load_hash_db()
    
    # Get all JSONL files
    all_shards = list(CONVERTED_DIR.glob("*.jsonl"))
    
    # Filter for specific traits if requested
    if specific_traits:
        print(f"🎯 Training only specified traits: {', '.join(specific_traits)}")
        shards_to_process = []
        for shard in all_shards:
            if shard.stem in specific_traits:
                shards_to_process.append(shard)
        
        # Check if all requested traits were found
        found_traits = [s.stem for s in shards_to_process]
        missing_traits = set(specific_traits) - set(found_traits)
        if missing_traits:
            print(f"⚠️  Warning: Could not find files for traits: {', '.join(missing_traits)}")
    else:
        print("🎯 Training all available traits")
        shards_to_process = all_shards
    
    job_ids = {}
    for shard in shards_to_process:
        print(f"Processing {shard.name}...")
        
        # Check if this file has already been processed
        existing_job = check_existing_job(shard, hash_db)
        if existing_job:
            job_id, status = existing_job
            job_ids[shard.stem] = job_id
            if status == "FinetuneJobStatus.STATUS_COMPLETED":
                print(f"  ✅ Already completed (job {job_id})")
            elif status in ("FinetuneJobStatus.STATUS_FAILED", "FinetuneJobStatus.STATUS_CANCELLED"):
                print(f"  ❌ Previous job failed (job {job_id}), will retry...")
                # Remove from hash db and continue to launch new job
                file_hash = calculate_file_hash(shard)
                if file_hash in hash_db:
                    del hash_db[file_hash]
                    save_hash_db(hash_db)
            else:
                print(f"  ⏳ Job in progress (job {job_id})")
                continue
        
        # Launch new job if not already completed
        if not existing_job or existing_job[1] in ("FinetuneJobStatus.STATUS_FAILED", "FinetuneJobStatus.STATUS_CANCELLED"):
            try:
                file_id = upload_file(shard)
                job_id = launch_lora_job(file_id, shard.stem)
                job_ids[shard.stem] = job_id
                print(f"  → job {job_id}")
                
                # Store hash for deduplication
                file_hash = calculate_file_hash(shard)
                hash_db[file_hash] = {
                    "job_id": job_id,
                    "file_name": shard.name,
                    "trait": shard.stem,
                    "created_at": time.time()
                }
                save_hash_db(hash_db)
                
            except Exception as e:
                print(f"  ❌ Error processing {shard.name}: {e}")
                continue
    
    # Save job IDs for later reference
    with open("lora_job_ids.json", "w") as f:
        json.dump(job_ids, f, indent=2)
    return job_ids

###############################################################################
# 2. Poll jobs, download LoRA checkpoints
###############################################################################
FT_DIR = Path("lora_checkpoints")
FT_DIR.mkdir(exist_ok=True)

def wait_and_download(job_ids):
    """Wait for fine-tuning jobs to complete and download LoRA adapter checkpoints."""
    remaining = dict(job_ids)
    pbar = tqdm(total=len(job_ids), desc="LoRA jobs")
    
    while remaining:
        for trait, job_id in list(remaining.items()):
            try:
                job_info = client.fine_tuning.retrieve(job_id)
                status_str = str(job_info.status)
                
                if status_str == "FinetuneJobStatus.STATUS_COMPLETED":
                    print(f"\n✅ Job {job_id} completed successfully!")
                    
                    # Download LoRA adapter weights
                    download_path = FT_DIR / f"{trait}.safetensors"
                    if not download_path.exists():
                        print(f"📥 Downloading LoRA adapter for {trait}...")
                        try:
                            # Download just the adapter weights using checkpoint="adapter"
                            client.fine_tuning.download(job_id, output=str(download_path), checkpoint_type="adapter")
                            
                        except Exception as e:
                            # Handle Windows lock file error (known issue)
                            if "WinError 2" in str(e) and ".lock" in str(e):
                                print(f"⚠️  Windows lock file error (known issue), checking if download succeeded...")
                            else:
                                print(f"❌ Error downloading {job_id}: {e}")
                                remaining.pop(trait)
                                pbar.update()
                                continue
                        
                        # Check if download succeeded regardless of lock file error
                        if download_path.exists():
                            file_size = download_path.stat().st_size / (1024 * 1024)  # MB
                            print(f"✅ Downloaded {file_size:.1f} MB")
                        else:
                            print(f"❌ Download failed - file not created")
                            remaining.pop(trait)
                            pbar.update()
                            continue
                    else:
                        print(f"⏩ LoRA adapter already exists at {download_path}")
                    
                    remaining.pop(trait)
                    pbar.update()
                    
                elif status_str in ("FinetuneJobStatus.STATUS_FAILED", "FinetuneJobStatus.STATUS_CANCELLED"):
                    print(f"\n❌ Job {job_id} failed with status: {job_info.status}")
                    remaining.pop(trait)
                    pbar.update()
                    
                else:
                    # Job still running, check again later
                    print(f"⏳ Job {job_id} status: {job_info.status}")
                    
            except Exception as e:
                print(f"❌ Error checking job {job_id}: {e}")
                # Don't continue here - we want to retry this job
                
        if remaining:
            time.sleep(30)  # Wait 30 seconds before checking again
    
    pbar.close()
    print(f"\n✅ Downloaded {len(job_ids)} LoRA adapter checkpoints")

###############################################################################
# 3. Prepare hypernetwork training data
###############################################################################
def prepare_hypernetwork_data():
    """Prepare metadata for hypernetwork training on GPU cluster"""
    TRAITS = ["intellect","discipline","extraversion","wisdom","compassion",
              "neuroticism","courage","humor","formality","sarcasm"]
    
    # Create trait mapping for hypernetwork training
    trait_mapping = {}
    for trait_file in FT_DIR.glob("*.safetensors"):
        stem = trait_file.stem
        trait_vec = [0] * 10
        
        for i, trait in enumerate(TRAITS):
            if stem.startswith(f"{trait}_plus") or stem.endswith(f"{trait}_plus"):
                trait_vec[i] = 1
            elif stem.startswith(f"{trait}_minus") or stem.endswith(f"{trait}_minus"):
                trait_vec[i] = -1
        
        trait_mapping[stem] = {
            "file_path": str(trait_file),
            "trait_vector": trait_vec
        }
    
    # Save mapping for hypernetwork training
    with open("hypernetwork_data.json", "w") as f:
        json.dump(trait_mapping, f, indent=2)
    
    print(f"✅ Prepared hypernetwork data for {len(trait_mapping)} LoRA checkpoints")
    print("📁 Files ready for GPU cluster:")
    print(f"   - lora_checkpoints/ ({len(list(FT_DIR.glob('*.safetensors')))} files)")
    print(f"   - hypernetwork_data.json")

###############################################################################
# 4. Main driver
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Pipeline for Trait-based Models")
    parser.add_argument(
        "traits", 
        nargs="*", 
        help="Specific traits to train (e.g., wisdom_minus intellect_minus). If not specified, trains all traits."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all trait hypernetworks (both plus and minus versions). Convenience option for full runs."
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List available trait files and exit"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if jobs already exist for the same file hashes"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.all and args.traits:
        print("❌ Error: Cannot use --all with specific trait arguments")
        print("Use either --all for all traits or specify individual traits")
        return
    
    # List available traits and exit if requested
    if args.list_available:
        if CONVERTED_DIR.exists():
            available_traits = [f.stem for f in CONVERTED_DIR.glob("*.jsonl")]
            print("Available traits:")
            for trait in sorted(available_traits):
                print(f"  - {trait}")
        else:
            print(f"❌ Converted files not found at {CONVERTED_DIR}")
            print("Please run the conversion script first:")
            print("  python scripts/py/convert_to_together_format.py")
        return
    
    # Clear hash database if force retrain is requested
    if args.force_retrain:
        print("🔄 Force retrain mode: clearing hash database...")
        if HASH_DB_FILE.exists():
            HASH_DB_FILE.unlink()
    
    print("🚀 Starting LoRA fine-tuning pipeline...")
    print(f"📂 Original corpus directory: {CORPUS_DIR}")
    print(f"📂 Converted files directory: {CONVERTED_DIR}")
    print(f"🤖 Model: {MODEL_NAME}")
    
    # ------ LoRA phase ------
    # Determine which traits to train
    if args.all:
        # Generate all trait combinations (plus and minus)
        TRAITS = ["intellect","discipline","extraversion","wisdom","compassion",
                  "neuroticism","courage","humor","formality","sarcasm"]
        specific_traits = []
        for trait in TRAITS:
            specific_traits.extend([f"{trait}_plus", f"{trait}_minus"])
        print(f"🎯 Training all {len(specific_traits)} trait combinations (--all flag)")
    else:
        specific_traits = args.traits if args.traits else None
    
    skip_training = False
    
    if not args.force_retrain and Path("lora_job_ids.json").exists():
        # Check if we have checkpoints for the requested traits
        if specific_traits:
            missing_checkpoints = []
            for trait in specific_traits:
                if not (FT_DIR / f"{trait}.safetensors").exists():
                    missing_checkpoints.append(trait)
            if not missing_checkpoints:
                print("✅ All requested LoRA checkpoints already present – skipping fine‑tune phase")
                skip_training = True
        else:
            # Check if we have any checkpoints
            if list(FT_DIR.glob("*.safetensors")):
                print("✅ LoRA checkpoints already present – skipping fine‑tune phase")
                skip_training = True
    
    if not skip_training:
        print("\n📤 Launching LoRA fine-tuning jobs...")
        ids = step1_launch_all_jobs(specific_traits)
        
        if ids:
            print(f"\n⏳ Waiting for {len(ids)} jobs to complete...")
            wait_and_download(ids)
        else:
            print("❌ No jobs were launched successfully!")
            return

    # ------ Prepare for hypernetwork training ------
    print("\n📋 Preparing hypernetwork training data...")
    prepare_hypernetwork_data()
    
    print("\n" + "="*70)
    print("🎉 LoRA FINE-TUNING COMPLETE!")
    print("="*70)
    print("Next steps:")
    print("1. 📤 Upload lora_checkpoints/ and hypernetwork_data.json to your GPU cluster")
    print("2. 🖥️  Run hypernetwork training on GPU cluster:")
    print("   python train_hypernetwork_cluster.py")
    print("3. 📥 Download trained hypernetwork model when complete")
    print("="*70)
    print("💡 Tip: Use --all flag for full trait training runs!")
    print("="*70)

if __name__ == "__main__":
    main()
