#!/usr/bin/env python
"""
Hypernetwork Training Script for GPU Clusters
  1. Load LoRA checkpoints from fine-tuning phase
  2. Build and train Text-to-LoRA hypernetwork
  3. Export the hypernetwork to ONNX

Designed to run on Together AI GPU clusters with PyTorch.
Author: maxfarnham@gmail.com
"""
import os, json, math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import Dataset, DataLoader

# Configuration
LORA_CHECKPOINTS_DIR = Path("lora_checkpoints")
HYPERNETWORK_DATA_FILE = "hypernetwork_data.json"
TRAITS = ["intellect","discipline","extraversion","wisdom","compassion",
          "neuroticism","courage","humor","formality","sarcasm"]

###############################################################################
# 0. Distributed Training Setup
###############################################################################
def setup_distributed():
    """Initialize distributed training if running with multiple processes"""
    # Check for SLURM environment variables first
    if 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', rank % torch.cuda.device_count()))
        
        print(f"🔗 SLURM distributed training: rank {rank}/{world_size}, local_rank {local_rank}")
        
        # Set up master address and port for NCCL
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '12355')
        
        # Initialize the process group with SLURM coordination
        dist.init_process_group(
            backend='nccl', 
            init_method=f'tcp://{master_addr}:{master_port}',
            rank=rank, 
            world_size=world_size
        )
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    
    # Fallback to torch.distributed.run environment variables
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        print(f"🔗 torch.distributed.run: rank {rank}/{world_size}, local_rank {local_rank}")
        
        # Initialize the process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        print("🖥️  Running in single-GPU mode")
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

###############################################################################
# 1. Hypernetwork Dataset
###############################################################################
class LoRADataset(Dataset):
    def __init__(self, hypernetwork_data_file: str):
        # Load the data mapping created by the fine-tuning script
        with open(hypernetwork_data_file, 'r') as f:
            self.data_mapping = json.load(f)
        
        self.files = list(self.data_mapping.keys())
        
        # Get flattened length from first file
        if self.files:
            sample_path = self.data_mapping[self.files[0]]["file_path"]
            
            # Add debugging for file issues
            print(f"📁 Loading sample file: {sample_path}")
            if not os.path.exists(sample_path):
                print(f"❌ File does not exist: {sample_path}")
                # List what files do exist
                import glob
                existing_files = glob.glob("lora_checkpoints/*.safetensors")
                print(f"📋 Available files: {existing_files}")
                raise FileNotFoundError(f"Sample file not found: {sample_path}")
            
            file_size = os.path.getsize(sample_path)
            print(f"📊 File size: {file_size:,} bytes ({file_size/(1024*1024):.1f} MB)")
            
            try:
                sample = load_safetensors(sample_path)
                self.keys = list(sample.keys())
                self.vlen = sum(v.numel() for v in sample.values())
                print(f"📊 LoRA weight vector length: {self.vlen:,}")
                print(f"🔑 Tensor keys: {len(self.keys)} tensors")
            except Exception as e:
                print(f"❌ Error loading safetensors file {sample_path}: {e}")
                print(f"💡 File size: {file_size} bytes")
                
                # Try to read the first few bytes to diagnose
                try:
                    with open(sample_path, 'rb') as f:
                        header_bytes = f.read(100)
                        print(f"🔍 First 100 bytes (hex): {header_bytes.hex()}")
                        print(f"🔍 First 100 bytes (repr): {repr(header_bytes)}")
                except Exception as read_error:
                    print(f"❌ Cannot even read file bytes: {read_error}")
                
                raise
        else:
            raise ValueError("No LoRA checkpoints found!")

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        file_key = self.files[idx]
        file_info = self.data_mapping[file_key]
        file_path = file_info["file_path"]
        
        # Load trait vector
        trait_vec = torch.tensor(file_info["trait_vector"], dtype=torch.float32)
        
        # Load and flatten LoRA weights with error handling
        try:
            weights = load_safetensors(file_path)
            flat = torch.cat([weights[k].flatten() for k in self.keys])
            return trait_vec, flat
        except Exception as e:
            print(f"❌ Error loading file {file_path} (idx {idx}): {e}")
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else "FILE_NOT_FOUND"
            print(f"💡 File: {file_key}, Path: {file_path}, Size: {file_size}")
            raise

###############################################################################
# 2. Hypernetwork Model
###############################################################################
class HyperLoRA(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(TRAITS), 256),
            nn.GELU(), 
            nn.LayerNorm(256),
            nn.Linear(256, 2048),
            nn.GELU(), 
            nn.LayerNorm(2048),
            nn.Linear(2048, out_dim)
        )
    
    def forward(self, x): 
        return self.net(x)

###############################################################################
# 3. Training Function
###############################################################################
def train_hypernet(rank=0, world_size=1, local_rank=0, lr=3e-4, epochs=5, batch_size=64):
    is_main_process = rank == 0
    
    if is_main_process:
        print("🔄 Initializing hypernetwork training...")
    
    # Set device for this process
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    if is_main_process:
        print(f"🖥️  Using device: {device}")
    
    # Create dataset and dataloader
    ds = LoRADataset(HYPERNETWORK_DATA_FILE)
    
    # Use distributed sampler if running with multiple GPUs
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank) if world_size > 1 else None
    shuffle = sampler is None  # Don't shuffle if using DistributedSampler
    
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    
    if is_main_process:
        print(f"📚 Dataset size: {len(ds)} LoRA checkpoints")
        print(f"📊 Batch size per GPU: {batch_size}")
        print(f"📊 Total effective batch size: {batch_size * world_size}")
    
    # Initialize model, optimizer, and loss function
    model = HyperLoRA(ds.vlen).to(device)
    
    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Count parameters (only print from main process)
    if is_main_process:
        if hasattr(model, 'module'):  # DDP wrapped model
            total_params = sum(p.numel() for p in model.module.parameters())
            trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧠 Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        # Set epoch for DistributedSampler
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        epoch_loss = 0
        
        # Only show progress bar on main process
        if is_main_process:
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        else:
            pbar = loader
        
        for batch_idx, (trait_vec, target) in enumerate(pbar):
            trait_vec, target = trait_vec.to(device), target.to(device)
            
            # Forward pass
            pred = model(trait_vec)
            loss = loss_fn(pred, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar (only on main process)
            epoch_loss += loss.item()
            if is_main_process and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_loss/(batch_idx+1):.6f}'
                })
        
        if is_main_process:
            print(f"📊 Epoch {epoch} completed - Average loss: {epoch_loss/len(loader):.6f}")
    
    # Save the trained model (only from main process)
    if is_main_process:
        if hasattr(model, 'module'):  # DDP wrapped model
            torch.save(model.module.state_dict(), "hypernet_fp16.pt")
        else:
            torch.save(model.state_dict(), "hypernet_fp16.pt")
        print("✅ Hypernetwork weights saved → hypernet_fp16.pt")
    
    # Wait for all processes to finish before returning
    if world_size > 1:
        dist.barrier()
    
    return model

###############################################################################
# 4. ONNX Export
###############################################################################
def export_onnx(model, path="hypernet.onnx"):
    print("📦 Exporting hypernetwork to ONNX...")
    
    # Handle DDP wrapped models
    export_model = model.module if hasattr(model, 'module') else model
    
    device = next(export_model.parameters()).device
    dummy = torch.zeros(1, len(TRAITS), dtype=torch.float32).to(device)
    
    # Set model to eval mode for export
    export_model.eval()
    
    torch.onnx.export(
        export_model, 
        dummy, 
        path,
        input_names=["trait_vec"], 
        output_names=["lora_flat"],
        dynamic_axes={
            "trait_vec": {0: "batch"}, 
            "lora_flat": {0: "batch"}
        },
        opset_version=17
    )
    print(f"✅ ONNX export complete → {path}")

###############################################################################
# 5. Main Function
###############################################################################
def main():
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    
    try:
        if is_main_process:
            print("🚀 Starting hypernetwork training on GPU cluster...")
            print("=" * 70)
        
        # Verify required files exist
        if not LORA_CHECKPOINTS_DIR.exists():
            raise FileNotFoundError(f"❌ LoRA checkpoints directory not found: {LORA_CHECKPOINTS_DIR}")
        
        if not Path(HYPERNETWORK_DATA_FILE).exists():
            raise FileNotFoundError(f"❌ Hypernetwork data file not found: {HYPERNETWORK_DATA_FILE}")
        
        if is_main_process:
            print(f"📁 Found LoRA checkpoints: {len(list(LORA_CHECKPOINTS_DIR.glob('*.safetensors')))} files")
        
        # Train the hypernetwork
        model = train_hypernet(rank=rank, world_size=world_size, local_rank=local_rank)
        
        # Export to ONNX (only from main process)
        if is_main_process:
            export_onnx(model)
            
            print("\n" + "=" * 70)
            print("🎉 HYPERNETWORK TRAINING COMPLETE!")
            print("=" * 70)
            print("📁 Output files:")
            print("   - hypernet_fp16.pt (PyTorch model)")
            print("   - hypernet.onnx (ONNX model)")
            print("=" * 70)
    
    finally:
        # Clean up distributed training
        cleanup_distributed()

if __name__ == "__main__":
    main() 