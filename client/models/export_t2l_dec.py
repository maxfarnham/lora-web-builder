# export_t2l_dec.py
# exports the task decoder of the t2l model into onnx
from pathlib import Path
import torch
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
from torch import nn

BASE = Path("../trained_t2l/llama_8b_t2l")           # <‑‑ same folder that holds hypermod.pt
DEVICE = "cpu"                   # GPU not needed for export

# 1️⃣  Load the trained hyper‑network checkpoint
_, hypermod, *_ = load_hypermod_checkpoint(BASE / "hypermod.pt", DEVICE)
hypermod.eval()                                  # we only need forward()  :contentReference[oaicite:2]{index=2}

# 2️⃣  Wrap the decoder so ONNX can see a SINGLE forward
class DecoderWrapper(nn.Module):
    def __init__(self, hnet):
        super().__init__()
        self.hnet = hnet

    def forward(self, latent, layer_ids):        # latent:(B,64)  layer_ids:(L)
        # encoded_task_emb is just the latent for each layer
        encoded = latent.repeat(len(layer_ids), 1)
        lora_A, lora_B = [], []
        for tgt_module in self.hnet.target_modules:          # q_proj, k_proj, …
            A, B = self.hnet.get_delta_weights(              # factorised LoRA  :contentReference[oaicite:3]{index=3}
                layer_indices=layer_ids,
                layer_type=tgt_module,
                encoded_task_emb=encoded,
                factorized=True,
            )
            lora_A.append(A)   # (L ,r ,in) each
            lora_B.append(B)   # (L ,out,r) each
        return tuple(lora_A + lora_B)            # 2×|target_modules| outputs

model = DecoderWrapper(hypermod).to(DEVICE)

# 3️⃣  Dummy inputs for tracing
dummy_latent   = torch.zeros(1, 64, dtype=torch.float32)
dummy_layers   = torch.arange(hypermod.max_num_layers, dtype=torch.int32)

# 4️⃣  Export
torch.onnx.export(
    model,
    (dummy_latent, dummy_layers),
    "t2l_llama_8b_decoder_fp32.onnx",
    input_names=["latent", "layer_ids"],
    output_names=[                  # keeps outputs deterministic
        *(f"lora_A_{m}" for m in hypermod.target_modules),
        *(f"lora_B_{m}" for m in hypermod.target_modules),
    ],
    dynamic_axes={
        "latent":    {0: "batch"},
        "layer_ids": {0: "n_layers"},
    },
    opset_version=17,               # the same opset used for the encoder  
)
print("✅  Decoder exported!")
