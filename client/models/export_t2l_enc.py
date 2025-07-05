# export_t2l_enc.py
# exports the task encoder of the t2l model into onnx
from pathlib import Path
import torch
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint

BASE = Path("../trained_t2l/llama_8b_t2l")
_, hypermod, *_ = load_hypermod_checkpoint(BASE / "hypermod.pt", "cpu")

mapping_net = hypermod.task_encoder
dummy = torch.zeros(1, mapping_net.mlp[0].in_features)

torch.onnx.export(
    mapping_net,
    dummy,
    "t2l_llama_8b_encoder_fp32.onnx",
    input_names=["embedding"],
    output_names=["deltas"],
    dynamic_axes={"embedding": {0: "batch"}},
    opset_version=17,
)