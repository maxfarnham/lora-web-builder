#!/bin/bash
# GPU Cluster Monitoring Script

echo "=== Together AI GPU Cluster Status ==="
echo "Timestamp: $(date)"
echo ""

echo "=== SLURM Queue Status ==="
squeue -u $USER
echo ""

echo "=== Node Information ==="
sinfo
echo ""

echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not available or not on GPU node"
fi
echo ""

echo "=== Disk Usage ==="
df -h
echo ""

echo "=== Recent Job History ==="
sacct -u $USER --starttime=today --format=JobID,JobName,State,Start,End,Elapsed,MaxRSS,MaxVMSize
echo ""

echo "=== System Load ==="
uptime
echo ""

echo "=== Memory Usage ==="
free -h
echo ""
