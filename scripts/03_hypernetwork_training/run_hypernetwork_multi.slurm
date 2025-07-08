#!/bin/bash
#SBATCH --job-name=hypernetwork-training-multi
#SBATCH --partition=all
#SBATCH --output=/data/hypernet_%j.out
#SBATCH --error=/data/hypernet_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --exclusive

# Fail fast on errors
set -euo pipefail

# Set environment variables
export PYTHONPATH=/data:$PYTHONPATH
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

# Navigate to shared directory
cd /data

# Extract training bundle (must exist)
echo "Extracting training bundle..."
tar -xzf /data/hypernet_bundle.tar.gz -C /data

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Check available space
echo "Available space:"
df -h /data

# Create outputs directory
mkdir -p /data/outputs

# Run the training script with distributed training
echo "Starting hypernetwork training (8 GPUs with DDP)..."
srun python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    train_hypernetwork_cluster.py --epochs 5 --batch_size 64 --output_dir /data/outputs

# Create results archive
echo "Creating results archive..."
if [ -d "/data/outputs" ]; then
    cd /data
    tar -czf hypernet_results_${SLURM_JOB_ID}.tar.gz outputs/
    echo "Results saved to hypernet_results_${SLURM_JOB_ID}.tar.gz"
fi

echo "Training completed!" 