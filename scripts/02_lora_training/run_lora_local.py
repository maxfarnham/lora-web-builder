#!/usr/bin/env python
"""
v2 – handles missing 'peft_type' and nested tensor wrappers
-----------------------------------------------------------
pip install transformers==4.46.2 peft==0.16.0 safetensors accelerate
"""

from __future__ import annotations
import argparse, json, pathlib, tempfile, textwrap, shutil, warnings
import torch, transformers, peft
from safetensors.torch import save_file
from accelerate import infer_auto_device_map, dispatch_model

# ---------- helpers -----------------------------------------------------------
def _json_value_to_tensor(obj):
    """
    Accept:
      • raw list or ndarray -> torch tensor
      • dict with {data|values, shape?, dtype?} -> tensor
    """
    if isinstance(obj, list):
        return torch.tensor(obj)

    if isinstance(obj, dict):
        data = obj.get("data") or obj.get("values")
        if data is None:
            raise ValueError(f"Cannot find data field in {obj.keys()}")
        t = torch.tensor(data)
        if "shape" in obj:
            t = t.reshape(obj["shape"])
        if obj.get("dtype") == "float16":
            t = t.to(torch.float16)
        elif obj.get("dtype") == "bfloat16":
            t = t.to(torch.bfloat16)
        return t

    raise TypeError(f"Unsupported state‑dict entry type: {type(obj)}")


def json_to_peft(json_path: str | pathlib.Path,
                 out_dir: str | None = None) -> pathlib.Path:
    raw = json.loads(pathlib.Path(json_path).read_text())

    # ---------- 1. adapter_config.json ----------
    cfg = raw["metadata"]
    cfg.setdefault("peft_type", "LORA")          # <‑‑ insert defaults
    cfg.setdefault("task_type", "CAUSAL_LM")
    cfg.setdefault("base_model_name_or_path",
                   "meta-llama/Llama-3.1-8B-Instruct")

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="lora_")
    out = pathlib.Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    # ---------- 2. adapter_model.safetensors ----------
    tensors = {k: _json_value_to_tensor(v) for k, v in raw["state_dict"].items()}
    save_file(tensors, out / "adapter_model.safetensors")

    return out


TESTS = [
    "In three sentences, describe the city you feel most connected to and why.",
    "Share a childhood memory that shaped your outlook on life.",
    "Walk me through your perfect Saturday.",
    "Where will you live in ten years and what will a day look like?",
    "Define ‘home’ in one sentence.",
    "Act as a subway conductor giving life advice between stops."
]


def chat(model, tokenizer, prompt, **gen_kwargs):
    prompt_complete = f"""
<|start_header_id|>system<|end_header_id|>
You are a friendly and industrious assistant who chats about anything and everything.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
    """
    ids = tokenizer(prompt_complete, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, **gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def compare(base, lora, tok):
    for p in TESTS:
        print("=" * 120)
        print("PROMPT:", p)
        print("\n-- Base ----------------------------------------------------------------")
        print(textwrap.fill(chat(base, tok, p, max_new_tokens=120, temperature=0.7), 110))
        print("\n-- LoRA ----------------------------------------------------------------")
        print(textwrap.fill(chat(lora, tok, p, max_new_tokens=120, temperature=0.9), 110))
        print()

def print_adapter_debug(model: peft.PeftModel, label: str):
    """Print quick diagnostics: active adapter, total Δ‑norm, first diff."""
    print(f"\n[DEBUG] === {label} ===")
    print("• active_adapter:", model.active_adapter)              # DEBUG
    delta_norm = 0.0
    for n, p in model.named_parameters():
        if ".lora_A" in n or ".lora_B" in n:
            delta_norm += float((p.detach()**2).sum())
    print(f"• LoRA Δ‑squared‑norm: {delta_norm:.3e}")              # DEBUG
    # compare a single weight tensor
    base_q = base.model.layers[0].self_attn.q_proj.weight
    lora_q = model.model.layers[0].self_attn.q_proj.weight
    diff   = (base_q - lora_q).abs().mean()
    print("• mean|ΔW| layer‑0 q_proj:", diff.item())               # DEBUG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Single‑file LoRA JSON")
    ap.add_argument("--out",  help="Write adapter here (otherwise tmp)")
    ap.add_argument("--gpu", action="store_true", help="Load on CUDA")
    ap.add_argument("--keep", action="store_true", help="Keep adapter dir")
    ap.add_argument("--offload", help="Folder where overflowed layers are swapped")
    args = ap.parse_args()

    adapter_dir = json_to_peft(args.json, args.out)
    print(f"[+] PEFT adapter written to {adapter_dir}")

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    device_map = "auto" if args.gpu else None

    print("[+] Loading base model …")
    max_mem  = {0: 11 << 30}          # 11 GB hard‑limit for GPU‑0
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, max_memory=max_mem, torch_dtype="auto", device_map=device_map
    )

    print("[+] Attaching LoRA …")
    lora = peft.PeftModel.from_pretrained(
           base,
           adapter_dir
    )
    lora.add_weighted_adapter(["default"], [3.0], "lora_x3")
    lora.set_adapter("lora_x3")
    print_adapter_debug(lora, "after weight scale 3")

    base = base.to("cpu")
    lora = lora.to("cuda")

    compare(base, lora, tok)

    if not args.keep and args.out is None:
        shutil.rmtree(adapter_dir, ignore_errors=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
