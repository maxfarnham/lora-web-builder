"""
Turn a single JSONL with continuous trait scores into
<trait>_plus.jsonl and <trait>_minus.jsonl shards.

Usage:
    python bucketize_traits.py --in_file your_corpus.jsonl --out_dir shards
"""
import json, argparse, math
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

TRAITS  = ["intellect","discipline","joy","wisdom","compassion",
           "neuroticism","courage","humor","formality","sarcasm"]
TH_HIGH =  0.6   # >= goes to + bucket
TH_LOW  = -0.6   # <= goes to - bucket
MIN_PER_BUCKET = 1500   # if after bucketing a pole has < N rows, we skip that LoRA

def main(jl_in: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = defaultdict(list)

    with jl_in.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="Read"):
            row = json.loads(line)
            tri = row["traits"]
            for t in TRAITS:
                v = tri[t]
                if v >= TH_HIGH:
                    buckets[f"{t}_plus"].append(row)
                elif v <= TH_LOW:
                    buckets[f"{t}_minus"].append(row)

    # write shards
    for name, rows in buckets.items():
        if len(rows) < MIN_PER_BUCKET:
            print(f"⚠  {name} only {len(rows)} rows – will skip this LoRA")
            continue
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as w:
            for r in rows:
                json.dump(r, w); w.write("\n")
        print(f"{name:15s} → {len(rows):6d} rows  ({path})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file",  required=True)
    ap.add_argument("--out_dir",  default="trait_shards")
    args = ap.parse_args()
    main(Path(args.in_file), Path(args.out_dir))
