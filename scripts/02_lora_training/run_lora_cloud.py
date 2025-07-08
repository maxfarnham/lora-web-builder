#!/usr/bin/env python
"""
serverless_lora_together.py
----------------------------------------------------------
• Convert one-file LoRA JSON → adapter ZIP
• Upload to S3 (auto-create together-loras-{account-id} if requested)
• Call Together AI's serverless Llama-3 8B Reference model with the adapter
  attached via the `adapters` parameter, then print base vs. LoRA replies.

Env vars required
  TOGETHER_API_KEY
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  (and optional AWS_SESSION_TOKEN)

Dependencies
  pip install boto3 botocore requests together safetensors torch
"""

from __future__ import annotations
import argparse, json, os, pathlib, shutil, sys, tempfile, uuid, zipfile
import boto3, botocore.exceptions, requests, torch
from safetensors.torch import save_file
from together import Together

BASE_MODEL = "meta-llama/Llama-3-8b-chat-hf"          # serverless reference
TESTS = [
    "Molly spent 3/4 of her money on a toy that cost $9. How much money did she start with?",
    "In three sentences, describe the city you feel most connected to and why.",
    "Share a childhood memory that shaped your outlook on life.",
    "Walk me through your perfect Saturday.",
    "Where will you live in ten years and what will a day look like?",
    "Define ‘home’ in one sentence.",
    "Act as a subway conductor giving life advice between stops."
]

# ───────────────────────────────── helpers ─────────────────────────────────
def ensure_bucket(bucket: str, region="us-east-1") -> str:
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[✓] Bucket {bucket} already exists")
    except botocore.exceptions.ClientError as e:
        if int(e.response["Error"]["Code"]) not in {403, 404}:
            raise
        cfg = {"Bucket": bucket}
        if region != "us-east-1":
            cfg["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**cfg)
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True, "IgnorePublicAcls": True,
                "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
            },
        )
        print(f"[+] Bucket {bucket} created in {region}")
    return bucket


def _json_to_tensor(obj):
    if isinstance(obj, list):
        return torch.tensor(obj)
    if isinstance(obj, dict):
        data = obj.get("data") or obj.get("values")
        t = torch.tensor(data)
        if "shape" in obj:
            t = t.reshape(obj["shape"])
        if obj.get("dtype") == "float16":
            t = t.half()
        elif obj.get("dtype") == "bfloat16":
            t = t.bfloat16()
        return t
    raise TypeError(type(obj))


def build_adapter(json_path: pathlib.Path, out_dir: pathlib.Path):
    raw = json.loads(json_path.read_text())
    cfg = raw["metadata"]
    cfg.setdefault("peft_type", "LORA")
    cfg.setdefault("task_type", "CAUSAL_LM")
    cfg["base_model_name_or_path"] = BASE_MODEL
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    tensors = {k: _json_to_tensor(v) for k, v in raw["state_dict"].items()}
    save_file(tensors, out_dir / "adapter_model.safetensors")


# ───────────────────────────────── main ────────────────────────────────────
def main():
    if "TOGETHER_API_KEY" not in os.environ:
        sys.exit("❌  Set TOGETHER_API_KEY in your environment")
    api_key = os.environ["TOGETHER_API_KEY"]

    try:
        boto3.client("sts").get_caller_identity()
    except botocore.exceptions.NoCredentialsError:
        sys.exit("❌  AWS creds missing (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)")

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="single-file LoRA JSON")
    ap.add_argument("--ensure-bucket", action="store_true",
                    help="create/verify together-loras-{account-id}")
    ap.add_argument("--s3-bucket", help="use existing bucket")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--prompts-file", help="file with one prompt per line (overrides default)")
    args = ap.parse_args()

    # ▶ load external prompts if requested
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [ln.rstrip() for ln in f if ln.strip()]
        if not prompts:
            sys.exit("Prompts file is empty")
    else:
        prompts = TESTS

    # bucket handling
    if args.ensure_bucket:
        if args.s3_bucket:
            bucket = ensure_bucket(args.s3_bucket, args.region)
        else:
            acct = boto3.client("sts").get_caller_identity()["Account"]
            bucket = ensure_bucket(f"together-loras-{acct}", args.region)
    elif args.s3_bucket:
        bucket = args.s3_bucket
    else:
        sys.exit("Provide --s3-bucket or --ensure-bucket")

    # build adapter → zip
    tmp = pathlib.Path(tempfile.mkdtemp())
    build_adapter(pathlib.Path(args.json), tmp / "adapter")
    zip_path = tmp / "lora.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in (tmp / "adapter").iterdir():
            zf.write(f, f.name)
    print(f"[+] Adapter zipped to {zip_path}")

    # upload & presign
    s3 = boto3.client("s3", region_name=args.region)
    obj_key = f"lora_uploads/{uuid.uuid4().hex}.zip"
    s3.upload_file(str(zip_path), bucket, obj_key)
    presigned = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": obj_key},
        ExpiresIn=3600)
    print(f"[+] Uploaded → s3://{bucket}/{obj_key} (presigned)")

    # serverless inference
    client = Together()

    def chat(prompt: str, with_lora: bool):
        payload = {
            "model": BASE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
            "temperature": 0.8,
            "top_p": 0.95
        }
        if with_lora:
            payload["adapters"] = [{"name": presigned, "scaling": 1.0}]
        resp = client.chat.completions.create(**payload)
        return resp.choices[0].message.content.strip()

    for p in prompts:
        print("=" * 110)
        print("PROMPT:", p)
        print("--- Base ---------------------------------------------------------")
        print(chat(p, with_lora=False))
        print("--- LoRA ---------------------------------------------------------")
        print(chat(p, with_lora=True))
        print()

    shutil.rmtree(tmp)

if __name__ == "__main__":
    main()
