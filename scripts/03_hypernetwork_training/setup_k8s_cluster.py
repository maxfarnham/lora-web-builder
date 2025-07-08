#!/usr/bin/env python
"""
Deploy and execute hypernetwork training on Kubernetes Slurm Cluster
Uses kubectl to interact with slurm-login service running on Kubernetes.

Usage:
    python setup_k8s_cluster.py --kubeconfig <path> --namespace <ns> --cluster-id <id> [options]
    
Options:
    --kubeconfig      Path to kubeconfig file (required)
    --namespace       Kubernetes namespace where Slurm is deployed (default: default)
    --cluster-id      Cluster ID for bookkeeping (required)
    --login-pod       Name of the Slurm login pod (default: slurm-controller-0)
    --shared-path     Path to shared PVC mount in pods (default: /tmp)
    --dry-run         Show what would be done without actually executing
    --no-monitor      Skip job monitoring after submission
    --no-download     Skip downloading results after completion
    --single-gpu      Use only 1 GPU instead of all 8 (faster for small jobs)
    --partition       Slurm partition to use (default: gpu)
"""
import os
import subprocess
import sys
import json
import tarfile
import time
import argparse
import signal
import tempfile
import shutil
from pathlib import Path

# Trait categories for deterministic trait vector generation
TRAIT_CATEGORIES = [
    "wisdom", "strength", "charisma", "intelligence", "dexterity",
    "constitution", "perception", "luck", "creativity", "focus"
]

def kubectl_safe_src(path):
    """
    Return a src path that kubectl treats as local on Windows.
    Converts 'D:\\foo\\bar.tar.gz' → './bar.tar.gz' and
    copies the file into current directory if necessary.
    
    This works around a kubectl bug on Windows where absolute paths
    with drive letters are misinterpreted as remote paths.
    """
    if os.name != "nt":  # Linux/macOS unaffected
        return str(path)
    
    path_str = str(path)
    if ":" in path_str:  # has drive letter
        current_dir = Path.cwd()
        temp_file = current_dir / path.name
        
        # Only copy if it's not already in current directory
        if temp_file.resolve() != path.resolve():
            print(f"   🔄 Copying {path.name} to current directory for kubectl compatibility...")
            shutil.copy2(path, temp_file)
        
        return f"./{path.name}"
    
    return path_str

def cleanup_kubectl_temp_files():
    """Clean up temporary files created for kubectl Windows compatibility"""
    if os.name != "nt":  # Only needed on Windows
        return
    
    # Clean up common temp files that might be left in current directory
    temp_patterns = ["hypernet_bundle.tar.gz", "run_hypernetwork_*.slurm", "hypernet_results_*.tar.gz"]
    current_dir = Path.cwd()
    
    for pattern in temp_patterns:
        for temp_file in current_dir.glob(pattern):
            if temp_file.exists():
                print(f"   🧹 Cleaning up temporary file: {temp_file.name}")
                temp_file.unlink()

def kubectl_safe_dest(path):
    """
    Return a dest path that kubectl treats as local on Windows.
    Converts 'D:\\foo\\bar.tar.gz' → './bar.tar.gz' 
    
    This works around a kubectl bug on Windows where absolute paths
    with drive letters are misinterpreted as remote paths.
    """
    if os.name != "nt":  # Linux/macOS unaffected
        return str(path)
    
    path_str = str(path)
    if ":" in path_str:  # has drive letter
        path_obj = Path(path_str)
        return f"./{path_obj.name}"
    
    return path_str

def run_kubectl_command(command, capture_output=True, description=None):
    """Run a kubectl command and return result"""
    if description:
        print(f"🔧 {description}")
    
    try:
        result = subprocess.run(command, capture_output=capture_output, text=True, timeout=300)
        
        if result.returncode == 0:
            if capture_output and result.stdout.strip():
                print(f"   ✅ {result.stdout.strip()}")
            # Return True for successful commands, even if no stdout
            return result.stdout.strip() if (capture_output and result.stdout.strip()) else True
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            print(f"   ❌ Command failed: {error_msg}")
            print(f"   💡 Command: {' '.join(command)}")
            return None
    except subprocess.TimeoutExpired:
        print(f"   ❌ Command timed out (5 minutes)")
        return None
    except Exception as e:
        print(f"   ❌ kubectl error: {e}")
        return None

def setup_kubectl_context(kubeconfig_path):
    """Set up kubectl context from kubeconfig file"""
    print("🔧 Setting up kubectl context...")
    
    if not Path(kubeconfig_path).exists():
        print(f"   ❌ Kubeconfig file not found: {kubeconfig_path}")
        return False
    
    # Set KUBECONFIG environment variable
    os.environ['KUBECONFIG'] = str(Path(kubeconfig_path).absolute())
    
    # Test kubectl connection
    result = run_kubectl_command(
        ["kubectl", "cluster-info"], 
        description="Testing kubectl connection"
    )
    
    if result:
        print("   ✅ kubectl context configured successfully")
        return True
    else:
        print("   ❌ Failed to configure kubectl context")
        return False

def check_slurm_cluster(namespace, login_pod):
    """Check if Slurm cluster is accessible"""
    print("🔍 Checking Slurm cluster...")
    
    # Check if login pod exists
    pod_check = run_kubectl_command(
        ["kubectl", "get", "pod", login_pod, "-n", namespace],
        description=f"Checking {login_pod} pod"
    )
    
    if not pod_check:
        print(f"   ❌ Login pod {login_pod} not found in namespace {namespace}")
        return False
    
    # Check if Slurm is working
    slurm_check = run_kubectl_command(
        ["kubectl", "exec", login_pod, "-n", namespace, "--", "sinfo", "-V"],
        description="Testing Slurm connection"
    )
    
    if slurm_check:
        print("   ✅ Slurm cluster is accessible")
        return True
    else:
        print("   ❌ Slurm is not responding")
        return False

def check_cluster_resources(namespace, login_pod):
    """Check cluster resources and partitions"""
    print("🔍 Checking cluster resources...")
    
    # Check partitions and GPU availability
    partition_check = run_kubectl_command(
        ["kubectl", "exec", login_pod, "-n", namespace, "--", "sinfo", "-o", "%P %G %D %m %N"],
        description="Checking partitions and GPU availability"
    )
    
    # Check shared filesystem
    shared_check = run_kubectl_command(
        ["kubectl", "exec", login_pod, "-n", namespace, "--", "df", "-h", "/home"],
        description="Checking shared filesystem"
    )
    
    return True

def check_required_files():
    """Check if required files exist locally"""
    print("📋 Checking required files...")
    
    required_files = [
        "train_hypernetwork_cluster.py",
        "lora_checkpoints/"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("   ❌ Missing required files:")
        for file_path in missing_files:
            print(f"      - {file_path}")
        return False
    
    # Check if lora_checkpoints directory has .safetensors files
    lora_dir = Path("lora_checkpoints")
    safetensors_files = list(lora_dir.glob("*.safetensors"))
    
    if not safetensors_files:
        print("   ❌ No .safetensors files found in lora_checkpoints/")
        return False
    
    print(f"   ✅ Found {len(safetensors_files)} LoRA checkpoint files")
    return True

def trait_vec_from_filename(filename):
    """Generate trait vector from filename using deterministic mapping"""
    vec = [0] * len(TRAIT_CATEGORIES)
    
    # Convert filename to lowercase for matching
    name_lower = filename.lower()
    
    # Check for trait patterns in filename
    for i, trait in enumerate(TRAIT_CATEGORIES):
        if f"{trait}_plus" in name_lower or f"{trait}+" in name_lower:
            vec[i] = 1
        elif f"{trait}_minus" in name_lower or f"{trait}-" in name_lower:
            vec[i] = -1
        elif trait in name_lower:
            # If trait is mentioned but not plus/minus, assign moderate positive
            vec[i] = 0.5
    
    # If no specific traits found, create a simple hash-based vector
    if all(v == 0 for v in vec):
        hash_val = hash(filename) % len(TRAIT_CATEGORIES)
        vec[hash_val] = 0.3  # Mild positive bias
    
    return vec

def create_hypernetwork_data_json():
    """Create hypernetwork_data.json mapping file"""
    print("📝 Creating hypernetwork_data.json...")
    
    lora_dir = Path("lora_checkpoints")
    safetensors_files = list(lora_dir.glob("*.safetensors"))
    
    if not safetensors_files:
        print("   ❌ No .safetensors files found in lora_checkpoints/")
        return False
    
    hypernetwork_data = {}
    
    # Create trait vectors for each LoRA file
    print("   🎯 Creating trait vectors for LoRA files...")
    
    for file_path in safetensors_files:
        file_name = file_path.stem  # filename without extension
        
        # Generate deterministic trait vector
        trait_vector = trait_vec_from_filename(file_name)
        
        hypernetwork_data[file_name] = {
            "file_path": str(file_path).replace("\\", "/"),  # Convert Windows paths to Linux
            "trait_vector": trait_vector
        }
    
    # Write the JSON file
    with open("hypernetwork_data.json", "w", encoding='utf-8') as f:
        json.dump(hypernetwork_data, f, indent=2)
    
    print(f"   ✅ Created hypernetwork_data.json with {len(hypernetwork_data)} entries")
    return True

def create_training_bundle():
    """Create a tar bundle of training files"""
    print("📦 Creating training bundle...")
    
    bundle_name = "hypernet_bundle.tar.gz"
    
    # Files to include in the bundle
    files_to_bundle = [
        "train_hypernetwork_cluster.py",
        "hypernetwork_data.json",
        "lora_checkpoints/"
    ]
    
    # Check if all files exist
    missing_files = []
    for file_path in files_to_bundle:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("   ❌ Cannot create bundle - missing files:")
        for file_path in missing_files:
            print(f"      - {file_path}")
        return False
    
    # Validate safetensors files BEFORE bundling
    print("   🔍 Validating local safetensors files before bundling...")
    lora_dir = Path("lora_checkpoints")
    safetensors_files = list(lora_dir.glob("*.safetensors"))
    
    validation_failed = False
    for file_path in safetensors_files:
        file_size = file_path.stat().st_size
        print(f"   📊 Local {file_path.name}: {file_size:,} bytes")
        
        try:
            # Try to load locally
            from safetensors.torch import load_file as load_safetensors
            data = load_safetensors(str(file_path))
            print(f"   ✅ Local {file_path.name}: Valid, {len(data)} tensors")
        except Exception as e:
            print(f"   ❌ Local {file_path.name}: Invalid - {e}")
            print(f"   🔍 Hex dump of first 32 bytes:")
            
            try:
                with open(file_path, 'rb') as f:
                    header_bytes = f.read(32)
                    hex_dump = ' '.join(f'{b:02x}' for b in header_bytes)
                    print(f"      {hex_dump}")
                    
                    # Try to decode potential header length (first 8 bytes as little-endian uint64)
                    if len(header_bytes) >= 8:
                        import struct
                        header_len = struct.unpack('<Q', header_bytes[:8])[0]
                        print(f"   📏 Interpreted header length: {header_len}")
                        if header_len > file_size:
                            print(f"   ⚠️  Header length ({header_len}) > file size ({file_size}) - CORRUPT!")
            except Exception as read_error:
                print(f"   ❌ Cannot read file: {read_error}")
            
            validation_failed = True
    
    if validation_failed:
        print("   💥 Local safetensors validation failed! Cannot create bundle with corrupted files.")
        return False
    
    # Create tar bundle
    try:
        with tarfile.open(bundle_name, "w:gz") as tar:
            for file_path in files_to_bundle:
                if file_path.endswith('/'):
                    # Add directory recursively, but with explicit binary mode for safetensors
                    tar.add(file_path, arcname=file_path, recursive=True)
                else:
                    tar.add(file_path, arcname=file_path)
        
        bundle_size = Path(bundle_name).stat().st_size / (1024 * 1024)  # Size in MB
        print(f"   ✅ Created {bundle_name} ({bundle_size:.1f} MB)")
        
        # Test extraction locally to verify bundle integrity
        print("   🔍 Testing bundle extraction locally...")
        test_dir = Path("bundle_test")
        test_dir.mkdir(exist_ok=True)
        
        try:
            with tarfile.open(bundle_name, "r:gz") as tar:
                tar.extractall(test_dir)
            
            # Test one safetensors file from the extracted bundle
            test_files = list((test_dir / "lora_checkpoints").glob("*.safetensors"))
            if test_files:
                test_file = test_files[0]
                test_size = test_file.stat().st_size
                print(f"   📊 Extracted test file {test_file.name}: {test_size:,} bytes")
                
                try:
                    test_data = load_safetensors(str(test_file))
                    print(f"   ✅ Extracted file validates: {len(test_data)} tensors")
                except Exception as e:
                    print(f"   ❌ Extracted file corrupted: {e}")
                    validation_failed = True
            
            # Clean up test directory
            import shutil
            shutil.rmtree(test_dir)
            
            if validation_failed:
                print("   💥 Bundle extraction test failed!")
                return False
            else:
                print("   ✅ Bundle integrity verified")
        
        except Exception as extract_error:
            print(f"   ❌ Bundle extraction test failed: {extract_error}")
            return False
        
        return bundle_name
    
    except Exception as e:
        print(f"   ❌ Failed to create bundle: {e}")
        return False

def create_slurm_job_script(single_gpu=False, partition="gpu", shared_path="/tmp"):
    """Create a SLURM job script for hypernetwork training"""
    print("📄 Creating SLURM job script...")
    
    if single_gpu:
        # Single GPU configuration
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=hypernetwork-training
#SBATCH --partition={partition}
#SBATCH --output={shared_path}/hypernet_%j.out
#SBATCH --error={shared_path}/hypernet_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclusive

# Fail fast on errors
set -euo pipefail

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH={shared_path}:${{PYTHONPATH:-}}

# Navigate to shared directory
cd {shared_path}

# Extract training bundle (must exist)
echo "Extracting training bundle..."
tar -xzf {shared_path}/hypernet_bundle.tar.gz -C {shared_path}

# Check if we need to install packages
echo "Setting up Python environment..."

# Try to load CUDA module if available
if command -v module >/dev/null 2>&1; then
    module load cuda/11.8 || echo "CUDA module not available, using system CUDA"
else
    echo "Environment modules not available, using system environment"
fi

# Check if torch is available, install if not
python3 -c "import torch" 2>/dev/null || {{
    echo "PyTorch not found, setting up complete ML environment..."
    
    # Try different package managers in order of preference
    INSTALL_SUCCESS=false
    
    # Option 1: Install pip via apt if needed, then use pip
    if ! command -v pip3 >/dev/null 2>&1 && ! python3 -m pip --version >/dev/null 2>&1; then
        echo "Installing pip via apt package manager..."
        apt update && apt install -y python3-pip python3-dev
    fi
    
    # Now try pip installation
    if command -v pip3 >/dev/null 2>&1; then
        PIP_CMD="pip3"
    elif python3 -m pip --version >/dev/null 2>&1; then
        PIP_CMD="python3 -m pip"
    else
        PIP_CMD=""
    fi
    
    if [ -n "$PIP_CMD" ]; then
        echo "Using pip command: $PIP_CMD"
        
        # Install PyTorch with CUDA support (compatible with CUDA 12.6.3)
        if $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; then
            echo "✅ Successfully installed PyTorch with CUDA 12.1 support"
            INSTALL_SUCCESS=true
        elif $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
            echo "⚠️  Installed PyTorch with CPU-only support (no CUDA)"
            INSTALL_SUCCESS=true
        fi
        
        # Install ML packages
        if [ "$INSTALL_SUCCESS" = true ]; then
            $PIP_CMD install transformers accelerate safetensors datasets || echo "Warning: Some ML packages failed"
            $PIP_CMD install numpy scipy matplotlib pillow || echo "Warning: Some scientific packages failed"
        fi
    fi
    
    # Option 2: Try conda if pip failed
    if [ "$INSTALL_SUCCESS" = false ] && command -v conda >/dev/null 2>&1; then
        echo "Trying conda package manager..."
        if conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia; then
            echo "✅ Successfully installed PyTorch via conda"
            conda install -y transformers accelerate -c huggingface || echo "Warning: Some packages failed"
            conda install -y numpy scipy matplotlib pillow || echo "Warning: Some packages failed"
            INSTALL_SUCCESS=true
        fi
    fi
    
    # Option 3: Check for system packages if all else fails
    if [ "$INSTALL_SUCCESS" = false ]; then
        echo "❌ Could not install packages via pip or conda"
        echo "Checking what Python packages are available..."
        
        python3 -c "
import sys
print(f'Python version: {{sys.version}}')
print(f'Python executable: {{sys.executable}}')
print()

try:
    import pkg_resources
    installed = [d.project_name for d in pkg_resources.working_set]
    ml_packages = [pkg for pkg in installed if any(x in pkg.lower() for x in ['torch', 'numpy', 'scipy', 'transform'])]
    if ml_packages:
        print(f'Found ML packages: {{ml_packages}}')
    else:
        print('No ML packages found in system')
        
    print(f'Total packages installed: {{len(installed)}}')
except Exception as e:
    print(f'Could not enumerate packages: {{e}}')

print()
print('Testing critical imports:')
for pkg in ['torch', 'numpy', 'transformers', 'safetensors']:
    try:
        __import__(pkg)
        print(f'  ✅ {{pkg}} - available')
    except ImportError:
        print(f'  ❌ {{pkg}} - missing')
"
        
        # Check if we have minimum required packages
        if python3 -c "import torch, numpy" 2>/dev/null; then
            echo "✅ Basic PyTorch and NumPy found - might be sufficient"
            INSTALL_SUCCESS=true
        else
            echo ""
            echo "💥 CRITICAL: Required packages not available!"
            echo ""
            echo "This Ubuntu container needs package installation but methods failed."
            echo ""
            echo "🔧 SOLUTIONS TRIED:"
            echo "  ❌ apt install python3-pip (might need root/sudo)"
            echo "  ❌ pip3 install (not available or permission denied)"
            echo "  ❌ conda install (not available)"
            echo ""
            echo "🐳 CONTAINER SOLUTIONS:"
            echo "  1. Use a PyTorch-enabled container image"
            echo "  2. Request container with ML packages pre-installed"
            echo "  3. Ask admin to modify base SLURM worker image"
            echo ""
            echo "💰 COST IMPACT:"
            echo "  - 8x H100 GPUs cost ~$32-40/hour"
            echo "  - This job would waste significant compute budget"
            echo ""
            echo "Job terminating early to avoid waste..."
            exit 1
        fi
    fi
}}

# Verify installation
echo "Verifying Python environment..."
python3 -c "
try:
    import torch
    print(f'PyTorch version: {{torch.__version__}}')
    print(f'CUDA available: {{torch.cuda.is_available()}}')
    print(f'GPU count: {{torch.cuda.device_count()}}')
except ImportError as e:
    print(f'PyTorch import failed: {{e}}')
    print('Available packages:')
    import pkg_resources
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    for pkg in sorted(installed_packages):
        print(f'  - {{pkg}}')
"

# Check if we can proceed with training
if ! python3 -c "import torch, numpy" 2>/dev/null; then
    echo ""
    echo "💥 ABORTING: Cannot proceed without required packages!"
    echo ""
    echo "This compute environment lacks the necessary ML packages."
    echo "Please use one of these solutions:"
    echo ""
    echo "🔧 IMMEDIATE SOLUTIONS:"
    echo "  1. Use --single-gpu flag (requires less resources)"
    echo "  2. Try a different partition: --partition=gpu-ml or --partition=ai"
    echo "  3. Contact cluster admin to install python3-pip"
    echo ""
    echo "🐳 CONTAINER SOLUTIONS:"
    echo "  1. Request PyTorch container: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
    echo "  2. Request HuggingFace container: huggingface/transformers-pytorch-gpu"
    echo "  3. Build custom container with required packages"
    echo ""
    echo "💰 COST IMPACT:"
    echo "  - 8x H100 GPUs cost ~$32-40/hour"
    echo "  - This job would waste significant compute budget"
    echo "  - Better to fix environment first"
    echo ""
    echo "Job terminating early to avoid waste..."
    exit 1
fi

echo "✅ Required packages verified - proceeding with training"

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Check available space
echo "Available space:"
df -h {shared_path}

# Check extracted files
echo "Extracted files:"
ls -la {shared_path}/
echo "LoRA checkpoints:"
ls -la {shared_path}/lora_checkpoints/ | head -10
echo "Hypernetwork data file:"
head -20 {shared_path}/hypernetwork_data.json

# Create outputs directory
mkdir -p {shared_path}/outputs

# Run the training script (single GPU)
echo "Starting hypernetwork training (single GPU)..."
python3 train_hypernetwork_cluster.py

# Create results archive
echo "Creating results archive..."
if [ -d "{shared_path}/outputs" ]; then
    cd {shared_path}
    tar -czf hypernet_results_${{SLURM_JOB_ID}}.tar.gz outputs/
    echo "Results saved to hypernet_results_${{SLURM_JOB_ID}}.tar.gz"
fi

echo "Training completed!"
"""
    else:
        # Multi-GPU configuration (8 GPUs)
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=hypernetwork-training-multi
#SBATCH --partition={partition}
#SBATCH --output={shared_path}/hypernet_%j.out
#SBATCH --error={shared_path}/hypernet_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --exclusive

# Fail fast on errors
set -euo pipefail

# Set environment variables
export PYTHONPATH={shared_path}:${{PYTHONPATH:-}}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Navigate to shared directory
cd {shared_path}

# Extract training bundle (must exist)
echo "Extracting training bundle..."

# Verify bundle integrity before extraction
echo "🔍 Checking bundle file integrity..."
if [ -f "{shared_path}/hypernet_bundle.tar.gz" ]; then
    bundle_size=$(stat -c%s "{shared_path}/hypernet_bundle.tar.gz")
    echo "📊 Bundle size: ${{bundle_size}} bytes"
    
    # Test if bundle can be listed (without extracting)
    if tar -tzf {shared_path}/hypernet_bundle.tar.gz > /dev/null 2>&1; then
        echo "✅ Bundle structure is valid"
        file_count=$(tar -tzf {shared_path}/hypernet_bundle.tar.gz | wc -l)
        echo "📋 Bundle contains ${{file_count}} files/directories"
        
        # List safetensors files in bundle
        echo "📋 Safetensors files in bundle:"
        tar -tzf {shared_path}/hypernet_bundle.tar.gz | grep '\.safetensors$' | head -5
    else
        echo "❌ Bundle is corrupted - cannot list contents"
        echo "💡 This suggests corruption during kubectl cp transfer"
        exit 1
    fi
else
    echo "❌ Bundle file not found: {shared_path}/hypernet_bundle.tar.gz"
    exit 1
fi

# Extract the bundle
tar -xzf {shared_path}/hypernet_bundle.tar.gz -C {shared_path}

# Verify extraction completed
if [ -d "{shared_path}/lora_checkpoints" ]; then
    extracted_count=$(find {shared_path}/lora_checkpoints -name "*.safetensors" | wc -l)
    echo "✅ Extraction completed: ${{extracted_count}} safetensors files extracted"
else
    echo "❌ Extraction failed - lora_checkpoints directory not found"
    exit 1
fi

# Check if we need to install packages
echo "Setting up Python environment..."

# Try to load CUDA module if available
if command -v module >/dev/null 2>&1; then
    module load cuda/11.8 || echo "CUDA module not available, using system CUDA"
else
    echo "Environment modules not available, using system environment"
fi

# Check if torch is available, install if not
python3 -c "import torch" 2>/dev/null || {{
    echo "PyTorch not found, setting up complete ML environment..."
    
    # Try different package managers in order of preference
    INSTALL_SUCCESS=false
    
    # Option 1: Install pip via apt if needed, then use pip
    if ! command -v pip3 >/dev/null 2>&1 && ! python3 -m pip --version >/dev/null 2>&1; then
        echo "Installing pip via apt package manager..."
        apt update && apt install -y python3-pip python3-dev
    fi
    
    # Now try pip installation
    if command -v pip3 >/dev/null 2>&1; then
        PIP_CMD="pip3"
    elif python3 -m pip --version >/dev/null 2>&1; then
        PIP_CMD="python3 -m pip"
    else
        PIP_CMD=""
    fi
    
    if [ -n "$PIP_CMD" ]; then
        echo "Using pip command: $PIP_CMD"
        
        # Install PyTorch with CUDA support (compatible with CUDA 12.6.3)
        if $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; then
            echo "✅ Successfully installed PyTorch with CUDA 12.1 support"
            INSTALL_SUCCESS=true
        elif $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
            echo "⚠️  Installed PyTorch with CPU-only support (no CUDA)"
            INSTALL_SUCCESS=true
        fi
        
        # Install ML packages
        if [ "$INSTALL_SUCCESS" = true ]; then
            $PIP_CMD install transformers accelerate safetensors datasets || echo "Warning: Some ML packages failed"
            $PIP_CMD install numpy scipy matplotlib pillow || echo "Warning: Some scientific packages failed"
        fi
    fi
    
    # Option 2: Try conda if pip failed
    if [ "$INSTALL_SUCCESS" = false ] && command -v conda >/dev/null 2>&1; then
        echo "Trying conda package manager..."
        if conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia; then
            echo "✅ Successfully installed PyTorch via conda"
            conda install -y transformers accelerate -c huggingface || echo "Warning: Some packages failed"
            conda install -y numpy scipy matplotlib pillow || echo "Warning: Some packages failed"
            INSTALL_SUCCESS=true
        fi
    fi
    
    # Option 3: Check for system packages if all else fails
    if [ "$INSTALL_SUCCESS" = false ]; then
        echo "❌ Could not install packages via pip or conda"
        echo "Checking what Python packages are available..."
        
        python3 -c "
import sys
print(f'Python version: {{sys.version}}')
print(f'Python executable: {{sys.executable}}')
print()

try:
    import pkg_resources
    installed = [d.project_name for d in pkg_resources.working_set]
    ml_packages = [pkg for pkg in installed if any(x in pkg.lower() for x in ['torch', 'numpy', 'scipy', 'transform'])]
    if ml_packages:
        print(f'Found ML packages: {{ml_packages}}')
    else:
        print('No ML packages found in system')
        
    print(f'Total packages installed: {{len(installed)}}')
except Exception as e:
    print(f'Could not enumerate packages: {{e}}')

print()
print('Testing critical imports:')
for pkg in ['torch', 'numpy', 'transformers', 'safetensors']:
    try:
        __import__(pkg)
        print(f'  ✅ {{pkg}} - available')
    except ImportError:
        print(f'  ❌ {{pkg}} - missing')
"
        
        # Check if we have minimum required packages
        if python3 -c "import torch, numpy" 2>/dev/null; then
            echo "✅ Basic PyTorch and NumPy found - might be sufficient"
            INSTALL_SUCCESS=true
        else
            echo ""
            echo "💥 CRITICAL: Required packages not available!"
            echo ""
            echo "This Ubuntu container needs package installation but methods failed."
            echo ""
            echo "🔧 SOLUTIONS TRIED:"
            echo "  ❌ apt install python3-pip (might need root/sudo)"
            echo "  ❌ pip3 install (not available or permission denied)"
            echo "  ❌ conda install (not available)"
            echo ""
            echo "🐳 CONTAINER SOLUTIONS:"
            echo "  1. Use a PyTorch-enabled container image"
            echo "  2. Request container with ML packages pre-installed"
            echo "  3. Ask admin to modify base SLURM worker image"
            echo ""
            echo "💰 COST IMPACT:"
            echo "  - 8x H100 GPUs cost ~$32-40/hour"
            echo "  - This job would waste significant compute budget"
            echo "  - Better to fix environment first"
            echo ""
            echo "Job terminating early to avoid waste..."
            exit 1
        fi
    fi
}}

# Verify installation
echo "Verifying Python environment..."
python3 -c "
try:
    import torch
    print(f'PyTorch version: {{torch.__version__}}')
    print(f'CUDA available: {{torch.cuda.is_available()}}')
    print(f'GPU count: {{torch.cuda.device_count()}}')
except ImportError as e:
    print(f'PyTorch import failed: {{e}}')
    print('Available packages:')
    import pkg_resources
    installed_packages = [d.project_name for d in pkg_resources.working_set]
    for pkg in sorted(installed_packages):
        print(f'  - {{pkg}}')
"

# Check if we can proceed with training
if ! python3 -c "import torch, numpy" 2>/dev/null; then
    echo ""
    echo "💥 ABORTING: Cannot proceed without required packages!"
    echo ""
    echo "This compute environment lacks the necessary ML packages."
    echo "Please use one of these solutions:"
    echo ""
    echo "🔧 IMMEDIATE SOLUTIONS:"
    echo "  1. Use --single-gpu flag (requires less resources)"
    echo "  2. Try a different partition: --partition=gpu-ml or --partition=ai"
    echo "  3. Contact cluster admin to install python3-pip"
    echo ""
    echo "🐳 CONTAINER SOLUTIONS:"
    echo "  1. Request PyTorch container: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
    echo "  2. Request HuggingFace container: huggingface/transformers-pytorch-gpu"
    echo "  3. Build custom container with required packages"
    echo ""
    echo "💰 COST IMPACT:"
    echo "  - 8x H100 GPUs cost ~$32-40/hour"
    echo "  - This job would waste significant compute budget"
    echo "  - Better to fix environment first"
    echo ""
    echo "Job terminating early to avoid waste..."
    exit 1
fi

echo "✅ Required packages verified - proceeding with training"

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Check available space
echo "Available space:"
df -h {shared_path}

# Check extracted files
echo "Extracted files:"
ls -la {shared_path}/
echo "LoRA checkpoints:"
ls -la {shared_path}/lora_checkpoints/ | head -10
echo "Hypernetwork data file:"
head -20 {shared_path}/hypernetwork_data.json

# Check distributed training environment
echo "SLURM distributed training environment:"
echo "SLURM_PROCID=${{SLURM_PROCID:-UNSET}}"
echo "SLURM_LOCALID=${{SLURM_LOCALID:-UNSET}}"
echo "SLURM_NTASKS=${{SLURM_NTASKS:-UNSET}}"
echo "SLURM_TASKS_PER_NODE=${{SLURM_TASKS_PER_NODE:-UNSET}}"
echo "SLURM_NODEID=${{SLURM_NODEID:-UNSET}}"
echo "MASTER_ADDR=${{MASTER_ADDR:-UNSET}}"
echo "MASTER_PORT=${{MASTER_PORT:-UNSET}}"

# Validate safetensors files
echo "Validating safetensors files..."
VALIDATION_FAILED=false
for file in lora_checkpoints/*.safetensors; do
    if [ -f "$file" ]; then
        size=$(stat -c%s "$file")
        echo "📊 ${{file}}: ${{size}} bytes"
        
        # Show hex dump of first 32 bytes for debugging
        echo "🔍 First 32 bytes of ${{file}}:"
        hexdump -C "$file" | head -2
        
        # Try loading with Python as validation
        python3 -c "
import sys
import struct
try:
    from safetensors.torch import load_file
    
    # First, check the header manually
    with open('${{file}}', 'rb') as f:
        header_bytes = f.read(32)
        if len(header_bytes) >= 8:
            header_len = struct.unpack('<Q', header_bytes[:8])[0]
            file_size = f.seek(0, 2)  # Go to end to get size
            f.seek(0)  # Reset
            print(f'📏 Header length: {{header_len}}, File size: {{file_size}}')
            if header_len > file_size:
                print(f'❌ Header length ({{header_len}}) > file size ({{file_size}}) - CORRUPT!')
                sys.exit(1)
    
    # Try actual loading
    data = load_file('${{file}}')
    print(f'✅ ${{file}}: Valid, {{len(data)}} tensors')
except Exception as e:
    print(f'❌ ${{file}}: Invalid - {{e}}')
    sys.exit(1)
" || {{
            echo "💥 CRITICAL: Corrupted safetensors file detected: ${{file}}"
            VALIDATION_FAILED=true
        }}
    fi
done

if [ "$VALIDATION_FAILED" = true ]; then
    echo ""
    echo "💥 SAFETENSORS VALIDATION FAILED!"
    echo ""
    echo "🔧 SOLUTIONS:"
    echo "  1. Re-run with --single-gpu flag to reduce complexity"
    echo "  2. Check if LoRA files are corrupted on source system"
    echo "  3. Verify tar bundle creation and extraction process"
    echo "  4. Try regenerating LoRA checkpoints"
    echo ""
    echo "💰 COST IMPACT:"
    echo "  - 8x H100 GPUs cost ~$32-40/hour"
    echo "  - Stopping now to avoid waste"
    echo ""
    echo "Job terminating early due to data corruption..."
    exit 1
fi

# Create outputs directory
mkdir -p {shared_path}/outputs

# Run the training script with distributed training via SLURM
echo "Starting hypernetwork training (8 GPUs with DDP via SLURM)..."
srun --ntasks=8 --ntasks-per-node=8 --cpus-per-task=4 \\
    python3 train_hypernetwork_cluster.py

# Create results archive
echo "Creating results archive..."
if [ -d "{shared_path}/outputs" ]; then
    cd {shared_path}
    tar -czf hypernet_results_${{SLURM_JOB_ID}}.tar.gz outputs/
    echo "Results saved to hypernet_results_${{SLURM_JOB_ID}}.tar.gz"
fi

echo "Training completed!"
"""
    
    gpu_suffix = "single" if single_gpu else "multi"
    script_name = f"run_hypernetwork_{gpu_suffix}.slurm"
    
    # Write with Unix line endings (\n) regardless of platform
    with open(script_name, "w", newline='\n', encoding='utf-8') as f:
        f.write(slurm_script)
    
    gpu_desc = "1 GPU" if single_gpu else "8 GPUs"
    print(f"   ✅ Created SLURM job script: {script_name} ({gpu_desc})")
    return script_name

def upload_files_to_cluster(namespace, login_pod, files, shared_path="/tmp", dry_run=False):
    """Upload files to cluster via kubectl cp"""
    print("📤 Uploading files to cluster...")
    
    if dry_run:
        print("   🔍 [DRY RUN] Would upload files to cluster")
        return True
    
    for file_path in files:
        # Convert to absolute path
        abs_file_path = Path(file_path).absolute()
        
        if not abs_file_path.exists():
            print(f"   ❌ File not found: {abs_file_path}")
            return False
        
        # Upload file to shared path
        remote_path = f"{shared_path}/{abs_file_path.name}"
        
        print(f"   🔄 Uploading {abs_file_path.name} ({abs_file_path.stat().st_size / (1024*1024):.1f} MB)...")
        
        # Check if file already exists in pod and remove it
        file_exists = run_kubectl_command(
            ["kubectl", "exec", login_pod, "-n", namespace, "--", "ls", remote_path],
            capture_output=True
        )
        
        if file_exists:
            print(f"   🔄 File already exists, removing first...")
            run_kubectl_command(
                ["kubectl", "exec", login_pod, "-n", namespace, "--", "rm", "-f", remote_path]
            )
        
        # Upload the file (use kubectl-safe path for Windows compatibility)
        # Add --no-preserve=true to prevent chown errors on shared volumes
        src_path = kubectl_safe_src(abs_file_path)
        expected_size = abs_file_path.stat().st_size
        
        # Try kubectl cp with --no-preserve flag first (requires kubectl >= 1.27)
        cp_result = run_kubectl_command(
            ["kubectl", "cp", "--no-preserve=true", src_path, f"{login_pod}:{remote_path}", "-n", namespace],
            description=f"Uploading {abs_file_path.name}"
        )
        
        # If --no-preserve flag failed (older kubectl), try without it
        if cp_result is None:
            print(f"   🔄 Retrying without --no-preserve flag (older kubectl version)...")
            cp_result = run_kubectl_command(
                ["kubectl", "cp", src_path, f"{login_pod}:{remote_path}", "-n", namespace],
                description=f"Uploading {abs_file_path.name} (fallback)"
            )
        
        # Always verify upload was successful regardless of cp exit code
        # kubectl cp can exit with error code due to chown failures while still copying the file
        print(f"   🔍 Verifying upload of {abs_file_path.name}...")
        verify_result = run_kubectl_command(
            ["kubectl", "exec", login_pod, "-n", namespace, "--", "stat", "-c", "%s", remote_path],
            capture_output=True,
            description=f"Verifying {abs_file_path.name} upload"
        )
        
        if verify_result and verify_result != True:
            try:
                actual_size = int(verify_result.strip())
                if actual_size == expected_size:
                    print(f"   ✅ Successfully uploaded {abs_file_path.name} ({actual_size} bytes)")
                    # If cp_result was None but file copied successfully, warn about chown issue
                    if cp_result is None:
                        print(f"   ⚠️  kubectl cp reported ownership errors but file copied successfully")
                else:
                    print(f"   ❌ Upload size mismatch: expected {expected_size}, got {actual_size}")
                    return False
            except (ValueError, TypeError):
                print(f"   ❌ Could not verify file size: {verify_result}")
                return False
        else:
            print(f"   ❌ Failed to verify upload of {abs_file_path.name}")
            # Show directory permissions for debugging
            run_kubectl_command(
                ["kubectl", "exec", login_pod, "-n", namespace, "--", "ls", "-la", shared_path],
                description="Checking directory permissions"
            )
            return False
    
    print("   ✅ All files uploaded successfully")
    return True

def find_existing_hypernetwork_jobs(namespace, login_pod):
    """Find existing hypernetwork training jobs"""
    print("🔍 Checking for existing hypernetwork training jobs...")
    
    # Get all jobs for current user
    jobs_output = run_kubectl_command(
        ["kubectl", "exec", login_pod, "-n", namespace, "--", "squeue", "-u", "slurm", "-h", "-o", "%i %j %T"],
        capture_output=True
    )
    
    if not jobs_output or jobs_output is True:
        print("   ✅ No existing jobs found")
        return None
    
    # Parse jobs and look for hypernetwork training jobs
    lines = jobs_output.strip().split('\n')
    hypernetwork_jobs = []
    
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 3:
                job_id, job_name, status = parts[0], parts[1], parts[2]
                if 'hypernetwork' in job_name.lower() or 'hypernet' in job_name.lower():
                    hypernetwork_jobs.append({
                        'id': job_id,
                        'name': job_name,
                        'status': status
                    })
    
    if hypernetwork_jobs:
        print(f"   ✅ Found {len(hypernetwork_jobs)} existing hypernetwork job(s):")
        for job in hypernetwork_jobs:
            print(f"      - Job {job['id']}: {job['name']} ({job['status']})")
        return hypernetwork_jobs
    else:
        print("   ✅ No existing hypernetwork training jobs found")
        return None

def submit_slurm_job(namespace, login_pod, job_script, shared_path="/tmp", dry_run=False):
    """Submit SLURM job via kubectl exec"""
    print("🚀 Submitting SLURM job...")
    
    if dry_run:
        print("   🔍 [DRY RUN] Would submit SLURM job")
        return "12345"  # Mock job ID
    
    # Copy the job script to the login pod first (controller doesn't have /data PVC mounted)
    print(f"   📋 Copying {job_script} to login pod...")
    
    # Use kubectl-safe path for Windows compatibility
    src_path = kubectl_safe_src(Path(job_script))
    controller_script_path = f"/tmp/{job_script}"
    
    # Try copying with --no-preserve flag first
    copy_result = run_kubectl_command(
        ["kubectl", "cp", "--no-preserve=true", src_path, f"{login_pod}:{controller_script_path}", "-n", namespace],
        description=f"Copying {job_script} to controller pod"
    )
    
    # If --no-preserve flag failed (older kubectl), try without it
    if copy_result is None:
        print(f"   🔄 Retrying copy without --no-preserve flag...")
        copy_result = run_kubectl_command(
            ["kubectl", "cp", src_path, f"{login_pod}:{controller_script_path}", "-n", namespace],
            description=f"Copying {job_script} to controller pod (fallback)"
        )
    
    # Verify the script was copied successfully
    verify_copy = run_kubectl_command(
        ["kubectl", "exec", login_pod, "-n", namespace, "--", "ls", "-la", controller_script_path],
        capture_output=True,
        description="Verifying script copy"
    )
    
    if not verify_copy or verify_copy is True:
        print(f"   ❌ Failed to copy {job_script} to controller pod")
        return None
    
    print(f"   ✅ Script copied to controller pod: {controller_script_path}")
    
    # Submit job using the script from /tmp on the controller
    result = run_kubectl_command(
        ["kubectl", "exec", login_pod, "-n", namespace, "--", "sbatch", controller_script_path],
        description="Submitting job via sbatch"
    )
    
    if result and "Submitted batch job" in result:
        job_id = result.split()[-1]
        print(f"   ✅ Job submitted with ID: {job_id}")
        return job_id
    else:
        print("   ❌ Job submission failed")
        return None

def monitor_slurm_job(namespace, login_pod, job_id, dry_run=False):
    """Monitor SLURM job progress"""
    print(f"👁️  Monitoring job {job_id}...")
    
    if dry_run:
        print("   🔍 [DRY RUN] Would monitor job progress")
        return True
    
    start_time = time.time()
    
    # Set up signal handler for graceful interruption
    def signal_handler(signum, frame):
        print(f"\n   ⚠️  Monitoring interrupted by user")
        print(f"   📝 Job {job_id} continues running on cluster")
        print(f"   💡 Check status: kubectl exec {login_pod} -n {namespace} -- squeue -j {job_id}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while True:
            # Check job status
            job_status = run_kubectl_command(
                ["kubectl", "exec", login_pod, "-n", namespace, "--", "squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True
            )
            
            if not job_status or job_status is True:
                # Job not in queue, check if completed
                job_info = run_kubectl_command(
                    ["kubectl", "exec", login_pod, "-n", namespace, "--", "sacct", "-j", job_id, "-n", "-o", "State"],
                    capture_output=True
                )
                
                if job_info and job_info != True and "COMPLETED" in job_info:
                    print(f"   ✅ Job {job_id} completed successfully!")
                    break
                elif job_info and job_info != True and ("FAILED" in job_info or "CANCELLED" in job_info):
                    print(f"   ❌ Job {job_id} failed or was cancelled")
                    print(f"   💡 Check logs: kubectl exec {login_pod} -n {namespace} -- cat /home/hypernet_{job_id}.err")
                    return False
                else:
                    print(f"   ❓ Job {job_id} status unknown")
                    break
            else:
                status = job_status.strip() if isinstance(job_status, str) else str(job_status)
                elapsed = time.time() - start_time
                
                if status == "RUNNING":
                    print(f"   🔄 Job {job_id} is running ({elapsed/60:.1f} minutes elapsed)")
                elif status == "PENDING":
                    print(f"   ⏳ Job {job_id} is pending ({elapsed/60:.1f} minutes elapsed)")
                else:
                    print(f"   📊 Job {job_id} status: {status}")
                
            # Wait before next check
            time.sleep(30)
            
            # Safety timeout (8 hours)
            if time.time() - start_time > 28800:
                print(f"   ⏰ Monitoring timeout (8 hours)")
                print(f"   📝 Job may still be running on cluster")
                break
                
    except KeyboardInterrupt:
        signal_handler(None, None)
    
    return True

def investigate_job_status(namespace, login_pod, job_id):
    """Investigate what happened with the job"""
    print(f"🔍 Investigating job {job_id} status...")
    
    # Get detailed job information
    job_details = run_kubectl_command(
        ["kubectl", "exec", login_pod, "-n", namespace, "--", "sacct", "-j", job_id, "-l"],
        capture_output=True,
        description=f"Getting detailed job {job_id} information"
    )
    
    if job_details and job_details != True:
        print(f"   📊 Job details:")
        print(f"   {job_details}")
    
    # Check for any output in common locations
    common_log_locations = ["/tmp", "/home", "/var/log/slurm", "/scratch"]
    
    for location in common_log_locations:
        files_in_location = run_kubectl_command(
            ["kubectl", "exec", login_pod, "-n", namespace, "--", "find", location, "-name", f"*{job_id}*", "-o", "-name", "*hypernet*"],
            capture_output=True
        )
        
        if files_in_location and files_in_location != True:
            print(f"   📁 Files in {location}:")
            for file_path in files_in_location.strip().split('\n'):
                if file_path.strip():
                    print(f"      - {file_path.strip()}")

def download_results(namespace, login_pod, job_id, shared_path="/data", dry_run=False):
    """Download training results via kubectl cp"""
    print("📥 Downloading training results...")
    
    if dry_run:
        print("   🔍 [DRY RUN] Would download results")
        return True
    
    # First investigate the job to understand what happened
    investigate_job_status(namespace, login_pod, job_id)
    
    # Create local results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Check what files actually exist for this job
    print("   🔍 Checking what files are available for download...")
    
    # Find all files related to this job
    job_files_check = run_kubectl_command(
        ["kubectl", "exec", login_pod, "-n", namespace, "--", "find", "/", "-name", f"*{job_id}*", "-type", "f"],
        capture_output=True,
        description=f"Searching for job {job_id} output files"
    )
    
    if job_files_check and job_files_check != True:
        print(f"   📋 Found job-related files:")
        for file_path in job_files_check.strip().split('\n'):
            if file_path.strip():
                print(f"      - {file_path.strip()}")
    
    # Try downloading results archive
    results_file = f"hypernet_results_{job_id}.tar.gz"
    remote_results_path = f"{shared_path}/{results_file}"
    
    # Check if results file exists before trying to download
    results_exists = run_kubectl_command(
        ["kubectl", "exec", login_pod, "-n", namespace, "--", "ls", "-la", remote_results_path],
        capture_output=True
    )
    
    if results_exists and results_exists != True:
        print(f"   ✅ Results archive found, downloading...")
        local_results_path = f"results/{results_file}"
        safe_dest_path = kubectl_safe_dest(local_results_path)
        
        # Try kubectl cp with --no-preserve flag first (requires kubectl >= 1.27)
        result = run_kubectl_command(
            ["kubectl", "cp", "--no-preserve=true", f"{login_pod}:{remote_results_path}", safe_dest_path, "-n", namespace],
            description=f"Downloading {results_file}"
        )
        
        # If --no-preserve flag failed (older kubectl), try without it
        if result is None:
            print(f"   🔄 Retrying without --no-preserve flag (older kubectl version)...")
            result = run_kubectl_command(
                ["kubectl", "cp", f"{login_pod}:{remote_results_path}", safe_dest_path, "-n", namespace],
                description=f"Downloading {results_file} (fallback)"
            )
        
        # Always check if file was downloaded regardless of cp exit code
        if Path(local_results_path).exists():
            # Extract results
            try:
                with tarfile.open(local_results_path, "r:gz") as tar:
                    tar.extractall("results/")
                print(f"   ✅ Results extracted to results/")
                # If result was None but file downloaded successfully, warn about chown issue
                if result is None:
                    print(f"   ⚠️  kubectl cp reported ownership errors but file downloaded successfully")
            except Exception as e:
                print(f"   ⚠️  Could not extract results: {e}")
        else:
            print(f"   ❌ Failed to download {results_file}")
    else:
        print(f"   ⚠️  Results archive not found: {remote_results_path}")
    
    # Download log files if they exist
    log_files = [f"hypernet_{job_id}.out", f"hypernet_{job_id}.err"]
    for log_file in log_files:
        # Try multiple potential locations for log files
        potential_paths = [
            f"{shared_path}/{log_file}",  # Default location
            f"/home/{log_file}",          # Common SLURM location
            f"/var/log/slurm/{log_file}", # Another common location
        ]
        
        downloaded = False
        for remote_log_path in potential_paths:
            # Check if log file exists
            log_exists = run_kubectl_command(
                ["kubectl", "exec", login_pod, "-n", namespace, "--", "ls", "-la", remote_log_path],
                capture_output=True
            )
            
            if log_exists and log_exists != True:
                print(f"   ✅ Found {log_file} at {remote_log_path}")
                local_log_path = f"results/{log_file}"
                safe_dest_path = kubectl_safe_dest(local_log_path)
                
                # Try kubectl cp with --no-preserve flag first
                log_result = run_kubectl_command(
                    ["kubectl", "cp", "--no-preserve=true", f"{login_pod}:{remote_log_path}", safe_dest_path, "-n", namespace],
                    description=f"Downloading {log_file}"
                )
                
                # If --no-preserve flag failed, try without it
                if log_result is None:
                    print(f"   🔄 Retrying {log_file} without --no-preserve flag...")
                    log_result = run_kubectl_command(
                        ["kubectl", "cp", f"{login_pod}:{remote_log_path}", safe_dest_path, "-n", namespace],
                        description=f"Downloading {log_file} (fallback)"
                    )
                
                # Check if file was downloaded regardless of cp exit code
                if Path(local_log_path).exists():
                    print(f"   ✅ Downloaded {log_file}")
                    if log_result is None:
                        print(f"   ⚠️  kubectl cp reported ownership errors but {log_file} downloaded successfully")
                    downloaded = True
                    break
                else:
                    print(f"   ❌ Failed to download {log_file}")
        
        if not downloaded:
            print(f"   ⚠️  Log file not found: {log_file}")
    
    print(f"   📁 Available files saved to ./results/")
    return True

def main():
    parser = argparse.ArgumentParser(description="Deploy hypernetwork training on Kubernetes Slurm Cluster")
    parser.add_argument("--kubeconfig", required=True, help="Path to kubeconfig file")
    parser.add_argument("--namespace", default="slurm", help="Kubernetes namespace (default: slurm)")
    parser.add_argument("--cluster-id", required=True, help="Cluster ID for bookkeeping")
    parser.add_argument("--login-pod", default="slurm-controller-0", help="Login pod name (default: slurm-controller-0)")
    parser.add_argument("--shared-path", default="/data", help="Shared filesystem path (default: /data)")
    parser.add_argument("--partition", default="all", help="SLURM partition (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--no-monitor", action="store_true", help="Skip job monitoring")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading results")
    parser.add_argument("--single-gpu", action="store_true", help="Use only 1 GPU instead of all 8")
    parser.add_argument("--force-new-job", action="store_true", help="Submit new job even if existing job found")
    parser.add_argument("--investigate-job", help="Investigate specific job ID and exit")
    args = parser.parse_args()
    
    print("🚀 Kubernetes Slurm Cluster Deployment")
    print("=" * 70)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual operations will be performed")
    
    print(f"🏗️  Cluster ID: {args.cluster_id}")
    print(f"⚙️  Kubeconfig: {args.kubeconfig}")
    print(f"📦 Namespace: {args.namespace}")
    print(f"🖥️  Login Pod: {args.login_pod}")
    print(f"📁 Shared Path: {args.shared_path}")
    print(f"🎯 Partition: {args.partition}")
    
    if args.single_gpu:
        print("🎯 GPU Mode: Single GPU (1/8 GPUs)")
    else:
        print("🎯 GPU Mode: Multi-GPU (8 GPUs with DDP)")
    
    if args.force_new_job:
        print("🔄 Job Mode: Force new job submission")
    else:
        print("🔄 Job Mode: Resume existing job if found")
    
    print("=" * 70)
    
    # Set up kubectl context
    if not setup_kubectl_context(args.kubeconfig):
        sys.exit(1)
    
    # Handle investigation mode
    if args.investigate_job:
        print(f"🔍 Investigating job {args.investigate_job}...")
        investigate_job_status(args.namespace, args.login_pod, args.investigate_job)
        sys.exit(0)
    
    # Check required files
    if not check_required_files():
        print("\n❌ Missing required files")
        sys.exit(1)
    
    # Check Slurm cluster
    if not check_slurm_cluster(args.namespace, args.login_pod):
        sys.exit(1)
    
    # Check cluster resources
    if not check_cluster_resources(args.namespace, args.login_pod):
        sys.exit(1)
    
    # Find compute pod for file uploads (since controller doesn't have shared storage)
    print("🔍 Finding compute pod with shared storage...")
    compute_pods = run_kubectl_command(
        ["kubectl", "get", "pods", "-n", args.namespace, "-l", "app.kubernetes.io/component=compute", "-o", "jsonpath={.items[*].metadata.name}"],
        capture_output=True,
        description="Finding compute pods"
    )
    
    if not compute_pods:
        print("   ❌ No compute pods found")
        sys.exit(1)
    
    # Use first compute pod for uploads
    compute_pod = compute_pods.strip().split()[0]
    print(f"   ✅ Using compute pod: {compute_pod}")
    
    # Create training files
    if not create_hypernetwork_data_json():
        sys.exit(1)
    
    bundle_path = create_training_bundle()
    if not bundle_path:
        sys.exit(1)
    
    slurm_script = create_slurm_job_script(args.single_gpu, args.partition, args.shared_path)
    if not slurm_script:
        sys.exit(1)
    
    # Upload files to compute pod (which has shared storage access)
    files_to_upload = [bundle_path, slurm_script]
    if not upload_files_to_cluster(args.namespace, compute_pod, files_to_upload, 
                                  args.shared_path, args.dry_run):
        sys.exit(1)
    
    # Check for existing jobs (unless forcing new job)
    if not args.force_new_job:
        existing_job = find_existing_hypernetwork_jobs(args.namespace, args.login_pod)
        if existing_job:
            print(f"   ✅ Resuming monitoring of existing job: {existing_job}")
            
            # Monitor existing job
            if not args.no_monitor:
                if not monitor_slurm_job(args.namespace, args.login_pod, existing_job, args.dry_run):
                    sys.exit(1)
            
            # Download results
            if not args.no_download:
                download_results(args.namespace, compute_pod, existing_job, args.shared_path, args.dry_run)
            
            cleanup_kubectl_temp_files()
            print(f"\n🎉 Resumed existing job {existing_job}!")
            return
        else:
            print("   ✅ No existing jobs found")
    
    # Submit SLURM job (via controller pod)
    job_id = submit_slurm_job(args.namespace, args.login_pod, slurm_script, 
                             args.shared_path, args.dry_run)
    if not job_id:
        sys.exit(1)
    
    # Monitor job (unless disabled)
    if not args.no_monitor:
        if not monitor_slurm_job(args.namespace, args.login_pod, job_id, args.dry_run):
            sys.exit(1)
    
    # Download results (unless disabled) - use compute pod since it has shared storage
    if not args.no_download:
        download_results(args.namespace, compute_pod, job_id, args.shared_path, args.dry_run)
    
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT COMPLETE!")
    print("=" * 70)
    print(f"📊 Job ID: {job_id}")
    print(f"🏷️  Cluster ID: {args.cluster_id}")
    print(f"🖥️  Login Pod: {args.login_pod}")
    print(f"📁 Shared Path: {args.shared_path}")
    
    if not args.no_monitor:
        print("📁 Results downloaded to ./results/")
    else:
        print(f"💡 Monitor job: kubectl exec {args.login_pod} -n {args.namespace} -- squeue -j {job_id}")
        print(f"💡 Download results: kubectl cp {compute_pod}:{args.shared_path}/hypernet_results_{job_id}.tar.gz ./results/ -n {args.namespace}")
    
    estimated_time = "~2.7 hours" if args.single_gpu else "~1.0 hour"
    print(f"⏱️  Estimated completion time: {estimated_time}")
    
    print("\n🎉 Hypernetwork training deployment completed!")
    print("💡 All operations performed via kubectl - no SSH keys needed!")
    print("=" * 70)
    
    # Clean up temporary files
    cleanup_kubectl_temp_files()

if __name__ == "__main__":
    main() 