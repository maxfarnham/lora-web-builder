# Text-to-LoRA ONNX Export - Quick Start

**Linux/WSL2 only** - Native Windows is not supported.
 v
## Prerequisites
- Linux or WSL2 (Ubuntu 20.04+)
- Python 3.8-3.12 (64-bit)
- pip ≥ 24
- Git + build-essentials

## Quick Start

1. **Clone repo** (work in Linux filesystem, not `/mnt/c`)
```bash
cd ~ && mkdir -p t2l && cd t2l
git clone https://github.com/SakanaAI/text-to-lora.git
```

2. **Setup environment**
```bash
python3.11 -m venv ~/.venvs/onnx-t2l # Need py3.11 for CUDA
source ~/.venvs/onnx-t2l/bin/activate
python3.11 -m pip install -U pip setuptools wheel
```

3. **Install code dependencies**
```bash
# CPU or CUDA 12.1 (choose one)
python3.11 -m pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu121
python3.11 -m pip install --upgrade onnx onnxruntime
cd ~/t2l/text-to-lora
pip install -e src/fishfarm
pip install -e .
```

3a. **Build flash-attn if required**
```bash
# stay inside the onnx-t2l venv
sudo apt-get install -y ninja-build  # Ninja speeds the compile
git clone --depth 1 --branch v2.8.0.post2 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# Use the same GCC that built PyTorch (Ubuntu 22.04 ships gcc-11 by default)
MAX_JOBS=$(nproc) python3.11 -m pip install --no-build-isolation .
cd ..
```

4. **Download weights from HF**
```bash
cd ~/t2l
pip install --upgrade huggingface_hub   # Installs the CLI
huggingface-cli login                   # Paste your HF access token
huggingface-cli download SakanaAI/text-to-lora --local-dir . --include "trained_t2l/*"
```

Your final structure should look like:
```
t2l/
├─ trained_t2l/
│   ├─ llama_8b_t2l
│   |   ├─ hypermod.pt
│   |   ├─ args.yaml
│   |   └─ adapter_config.json
```
The weights will also include `gemma_2b_t2l`, `mistral_7b_t2l` and `t2l_demo`.

5. **Export net to ONNX**
Copy the py files from the `models` directory here into `~/t2l`.
Run: `export_t2l_enc.py` and `export_t2l_dec.py` to get the encoder and decoder, respectively. This will download base model weights for llama 3.1 8B onto your machine from HF.

Be sure to run these from the `t2l/text-to-lora` repo root, as it has assumptions about file locations.

For the full model, you can use the `export_onnx.py` script in the repo:
```
python scripts/export_onnx.py \
       --ckpt weights/hypermod.pt \
       --out  web_t2l_full.onnx \
       --full-graph
```

## Common Issues
- **"Operation not permitted"**: Work in Linux filesystem, not `/mnt/c`
- **"externally-managed-environment"**: Don't use `sudo pip`
- **"No matching distribution"**: Use 64-bit Python 3.10+ with extra index URL
- **"vLLM only supports Linux"**: Use WSL2/Linux, not native Windows
