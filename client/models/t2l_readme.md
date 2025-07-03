# Text-to-LoRA ONNX Export - Quick Start

**Linux/WSL2 only** - Native Windows is not supported.

## Prerequisites
- Linux or WSL2 (Ubuntu 20.04+)
- Python 3.8-3.12 (64-bit)
- pip ≥ 24
- Git + build-essentials

## Quick Start

1. **Clone repo** (work in Linux filesystem, not `/mnt/c`)
```bash
cd ~ && mkdir -p code && cd code
git clone https://github.com/SakanaAI/text-to-lora.git
```

2. **Setup environment**
```bash
python3 -m venv ~/.venvs/onnx-t2l
source ~/.venvs/onnx-t2l/bin/activate
python -m pip install -U pip
```

3. **Install dependencies**
```bash
# CPU or CUDA 12.1 (choose one)
python -m pip install torch==2.4.0 torchvision==0.19.0 \
      --extra-index-url https://download.pytorch.org/whl/cu121

cd ~/code/text-to-lora
pip install -e src/fishfarm
pip install -e src/hyper_llm_modulator
```

4. **Place checkpoint files**
Pick the weights associated with the model you want - we pulled llama 3.1 8B, t2l repo also has gemma + mistral.

```
text-to-lora/
├─ weights/                     # any folder name (avoid 'checkpoint' in path)
│   ├─ hypermod.pt
│   ├─ args.yaml
│   └─ adapter_config.json
```

5. **Export ONNX**
```python
# export_t2l.py
from pathlib import Path
import torch
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint

BASE = Path("weights")
_, hypermod, *_ = load_hypermod_checkpoint(BASE / "hypermod.pt", "cpu")

mapping_net = hypermod.task_encoder
dummy = torch.zeros(1, mapping_net.mlp[0].in_features)

torch.onnx.export(
    mapping_net,
    dummy,
    "t2l_llama_8b_fp32.onnx",
    input_names=["embedding"],
    output_names=["deltas"],
    dynamic_axes={"embedding": {0: "batch"}},
    opset_version=17,
)
```

Run: `python export_t2l.py`

## Common Issues
- **"Operation not permitted"**: Work in Linux filesystem, not `/mnt/c`
- **"externally-managed-environment"**: Don't use `sudo pip`
- **"No matching distribution"**: Use 64-bit Python 3.10+ with extra index URL
- **"vLLM only supports Linux"**: Use WSL2/Linux, not native Windows
