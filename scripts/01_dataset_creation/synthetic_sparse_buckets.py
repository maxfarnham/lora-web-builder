#!/usr/bin/env python
"""
synthetic_sparse_buckets.py  –  diversity‑aware filler, OpenAI client ≥ 1.0

* Async (10 coroutines by default)
* Seed‑and‑paraphrase strategy for richer lexical variety
* MinHash de‑duplication
"""
import os, json, uuid, argparse, asyncio, random, hashlib, sys
from pathlib import Path
from collections import defaultdict
from datasketch import MinHash, MinHashLSH
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI, RateLimitError, InternalServerError, APIConnectionError

# --------------------------------------------------------------------------- #
client = AsyncOpenAI()  # reads OPENAI_API_KEY
MAX_TRIES = 5
CONCURRENCY = 10

TRAITS = ["intellect","discipline","joy","wisdom","compassion",
          "neuroticism","courage","humor","formality","sarcasm"]

# Unified trait configuration with templates and descriptions
TRAIT_CONFIG = {
    "intellect_plus": {
        "description": "analytical, thoughtful, uses precise reasoning",
        "templates": [
            "Rewrite analytically with fresh phrasing:",
            "Provide an alternative analytical paraphrase:",
            "Restate the idea using precise reasoning:",
            "Re-express the statement clearly and analytically:"
        ]
    },
    "intellect_minus": {
        "description": "simple, direct, avoids complex analysis",
        "templates": [
            "Rewrite in simple, straightforward language:",
            "Express this idea in plain, uncomplicated terms:",
            "Simplify this statement without analysis:",
            "Rephrase this in basic, direct words:"
        ]
    },
    "discipline_plus": {
        "description": "methodical, organised, gives step‑by‑step plans",
        "templates": [
            "Rewrite with methodical, structured phrasing as if spoken by a well-organized person:",
            "Reorganize this into a systematic format as if spoken by a well-organized person:",
            "Express this with disciplined, orderly language as if spoken by a well-organized person:",
            "Rephrase this in a well-organized manner as if spoken by a well-organized person:"
        ]
    },
    "discipline_minus": {
        "description": "scattered, disorganized, jumps between ideas",
        "templates": [
            "Rewrite with scattered, meandering phrasing:",
            "Express this in a disorganized, stream-of-consciousness way:",
            "Rephrase this with tangential, unfocused language:",
            "Restate this in a rambling, unstructured manner:"
        ]
    },
    "joy_plus": {
        "description": "enthusiastic, upbeat, optimistic",
        "templates": [
            "Rewrite with enthusiastic, joyful phrasing:",
            "Express this with upbeat, positive language:",
            "Rephrase this in an optimistic, cheerful way:",
            "Restate this with energetic, happy tone:"
        ]
    },
    "joy_minus": {
        "description": "subdued, melancholic, pessimistic",
        "templates": [
            "Rewrite with subdued, somber phrasing:",
            "Express this with melancholic, downcast language:",
            "Rephrase this in a pessimistic, gloomy way:",
            "Restate this with resigned, mournful tone:"
        ]
    },
    "wisdom_plus": {
        "description": "thoughtful, considers long-term consequences, reflects deeply",
        "templates": [
            "Rewrite with wise, contemplative phrasing:",
            "Express this with thoughtful, reflective language:",
            "Rephrase this with deep, philosophical insight:",
            "Restate this with mature, considered judgment:"
        ]
    },
    "wisdom_minus": {
        "description": "reckless, short‑sighted, acts before thinking",
        "templates": [
            "Rewrite with impulsive, hasty phrasing:",
            "Express this with reckless, immediate language:",
            "Rephrase this without considering consequences:",
            "Restate this in a short-sighted, reactive way:"
        ]
    },
    "compassion_plus": {
        "description": "empathetic, caring, considers others' feelings",
        "templates": [
            "Rewrite with empathetic, caring phrasing:",
            "Express this with compassionate, understanding language:",
            "Rephrase this with kindness and consideration:",
            "Restate this with warm, supportive tone:"
        ]
    },
    "compassion_minus": {
        "description": "cold, indifferent, dismissive of others' feelings",
        "templates": [
            "Rewrite with cold, detached phrasing:",
            "Express this with indifferent, uncaring language:",
            "Rephrase this dismissively without empathy:",
            "Restate this with callous, unsympathetic tone:"
        ]
    },
    "neuroticism_plus": {
        "description": "anxious, worried, focuses on potential problems",
        "templates": [
            "Rewrite with anxious, concerned phrasing:",
            "Express this with worried, apprehensive language:",
            "Rephrase this highlighting potential problems:",
            "Restate this with nervous, uncertain tone:"
        ]
    },
    "neuroticism_minus": {
        "description": "calm, stable, emotionally balanced",
        "templates": [
            "Rewrite with calm, balanced phrasing:",
            "Express this with stable, composed language:",
            "Rephrase this with emotional equilibrium:",
            "Restate this with serene, untroubled tone:"
        ]
    },
    "courage_plus": {
        "description": "brave, bold, willing to take risks",
        "templates": [
            "Rewrite with bold, courageous phrasing:",
            "Express this with brave, daring language:",
            "Rephrase this with fearless determination:",
            "Restate this with confident, intrepid tone:"
        ]
    },
    "courage_minus": {
        "description": "timid, hesitant, avoids risks",
        "templates": [
            "Rewrite with timid, cautious phrasing:",
            "Express this with hesitant, uncertain language:",
            "Rephrase this avoiding any bold statements:",
            "Restate this with fearful, tentative tone:"
        ]
    },
    "humor_plus": {
        "description": "playful, witty, finds lightness in situations",
        "templates": [
            "Rewrite with playful, humorous phrasing:",
            "Express this with witty, amusing language:",
            "Rephrase this with lighthearted, funny tone:",
            "Restate this with cheerful, comedic flair:"
        ]
    },
    "humor_minus": {
        "description": "grimly serious, formal, never cracks jokes",
        "templates": [
            "Rewrite with serious, formal phrasing:",
            "Express this with grave, solemn language:",
            "Rephrase this with stern, humorless tone:",
            "Restate this with rigid, austere formality:"
        ]
    },
    "formality_plus": {
        "description": "proper, ceremonious, follows social conventions",
        "templates": [
            "Rewrite with formal, proper phrasing:",
            "Express this with ceremonious, dignified language:",
            "Rephrase this following strict social conventions:",
            "Restate this with polite, respectful formality:"
        ]
    },
    "formality_minus": {
        "description": "casual, relaxed, ignores social conventions",
        "templates": [
            "Rewrite with casual, informal phrasing:",
            "Express this with relaxed, laid-back language:",
            "Rephrase this ignoring formal conventions:",
            "Restate this with easygoing, conversational tone:"
        ]
    },
    "sarcasm_plus": {
        "description": "ironic, cutting, uses verbal irony",
        "templates": [
            "Rewrite with sarcastic, ironic phrasing:",
            "Express this with cutting, sardonic language:",
            "Rephrase this using verbal irony:",
            "Restate this with biting, caustic tone:"
        ]
    },
    "sarcasm_minus": {
        "description": "sincere, straightforward, literal",
        "templates": [
            "Rewrite with sincere, genuine phrasing:",
            "Express this with straightforward, literal language:",
            "Rephrase this without any irony or sarcasm:",
            "Restate this with honest, direct tone:"
        ]
    }
}

def build_trait_vec(bucket):
    return {t: (1 if bucket.startswith(f"{t}_plus") else
                -1 if bucket.startswith(f"{t}_minus") else 0)
            for t in TRAITS}

# ----------------------------- seed loading -------------------------------- #
def load_seeds(files, max_per_file=5000):
    """Load seed phrases from files, assuming they are neutral."""
    pool = []
    for f in files:
        with Path(f).open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= max_per_file: break
                try:
                    obj = json.loads(line)
                    text = obj.get("utterance") or obj.get("dialogue") or obj.get("text")
                    if text and 8 < len(text.split()) < 30:
                        pool.append(text.strip())
                except json.JSONDecodeError:
                    # Try treating as plain text
                    text = line.strip()
                    if text and 8 < len(text.split()) < 30:
                        pool.append(text)
                    continue
    random.shuffle(pool)
    return pool

def sample_seeds(pool, target_count):
    """Sample seeds to match target count, with replacement if needed."""
    if len(pool) >= target_count:
        return random.sample(pool, target_count)
    else:
        # Sample with replacement if we don't have enough seeds
        return random.choices(pool, k=target_count)

# ----------------------------- MinHash utils ------------------------------- #
def text_hash(text, n_grams=3):
    mh = MinHash(num_perm=64)
    tokens = text.lower().split()
    shingles = [' '.join(tokens[i:i+n_grams]) for i in range(len(tokens)-n_grams+1)]
    for s in shingles:
        mh.update(s.encode('utf-8'))
    return mh

class DedupBuffer:
    def __init__(self, threshold=0.85):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=64)
        self.counter = 0
    def is_dup(self, text):
        mh = text_hash(text)
        dup = self.lsh.query(mh)
        if dup: return True
        self.lsh.insert(f"t{self.counter}", mh)
        self.counter += 1
        return False

# ----------------------------- GPT‑4o request ------------------------------ #
async def paraphrase(seed, trait_name):
    """Paraphrase a seed using trait-specific templates and style."""
    trait_config = TRAIT_CONFIG.get(trait_name)
    if not trait_config:
        raise ValueError(f"Unknown trait: {trait_name}")
    
    templates = trait_config["templates"]
    style_desc = trait_config["description"]
    
    for attempt in range(MAX_TRIES):
        try:
            system_prompt = f"{random.choice(templates)} Style: {style_desc}"
            rsp = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=random.uniform(1.05, 1.3),
                    top_p=0.92,
                    presence_penalty=0.6,
                    frequency_penalty=0.4,
                    messages=[
                      {"role":"system","content":system_prompt},
                      {"role":"user",  "content":seed}
                    ])
            return rsp.choices[0].message.content.strip()
        except (RateLimitError, InternalServerError, APIConnectionError) as e:
            wait_time = 2 ** attempt
            print(f"OpenAI API error (attempt {attempt + 1}/{MAX_TRIES}): {type(e).__name__}. Waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
    raise RuntimeError("Repeated API failures after maximum retries.")

# ----------------------------- bucket filler ------------------------------- #
async def fill_bucket(name, seeds, target_rows, shard_dir):
    """Fill a bucket with paraphrased seeds for a specific trait."""
    path = shard_dir / f"{name}.jsonl"
    existing = sum(1 for _ in path.open()) if path.exists() else 0
    need = max(0, target_rows - existing)
    if need == 0:
        print(f"{name}: already {existing} rows.")
        return

    print(f"{name}: need {need} new lines.")
    
    # Validate trait exists
    if name not in TRAIT_CONFIG:
        raise ValueError(f"Unknown trait: {name}. Available traits: {list(TRAIT_CONFIG.keys())}")
    
    trait_vec = build_trait_vec(name)
    dedup = DedupBuffer()
    
    # Sample seeds to match our target count
    sampled_seeds = sample_seeds(seeds, need * 2)  # Get extra seeds to account for duplicates
    seed_iter = iter(sampled_seeds)

    sem = asyncio.Semaphore(CONCURRENCY)
    pbar = tqdm(total=need, desc=name)

    async def worker():
        nonlocal need, seed_iter
        while True:
            if need <= 0: break
            try:
                seed = next(seed_iter)
            except StopIteration:
                # If we run out of sampled seeds, sample more
                new_seeds = sample_seeds(seeds, need * 2)
                seed_iter = iter(new_seeds)
                seed = next(seed_iter)
            
            async with sem:
                text = await paraphrase(seed, name)
            if dedup.is_dup(text): continue
            with path.open("a", encoding="utf-8") as fh:
                json.dump({
                    "utterance": text,
                    "traits": trait_vec,
                    "source_metadata":{
                        "source":"GPT4o_gen",
                        "origin_id": str(uuid.uuid4()),
                        "original_tag": name,
                        "original_seed": seed}
                }, fh, ensure_ascii=False); fh.write("\n")
            need -= 1
            pbar.update()
    await asyncio.gather(*[worker() for _ in range(CONCURRENCY)])
    pbar.close()
    print(f"✅  {name} filled to {target_rows} rows.")

# ----------------------------- CLI driver ---------------------------------- #
def list_available_traits():
    """List all available trait buckets."""
    print("Available trait buckets:")
    for trait_name in sorted(TRAIT_CONFIG.keys()):
        config = TRAIT_CONFIG[trait_name]
        print(f"  {trait_name}: {config['description']}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in env.")

    ap = argparse.ArgumentParser(
        description="Generate synthetic training data for personality traits using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available trait buckets:
{chr(10).join(f"  {name}: {config['description']}" for name, config in sorted(TRAIT_CONFIG.items()))}

Examples:
  python {os.path.basename(__file__)} --seed_files seeds.jsonl --target 1000 intellect_plus wisdom_minus
  python {os.path.basename(__file__)} --seed_files seeds.jsonl --target 500 humor_plus sarcasm_minus formality_plus
""")
    ap.add_argument("--shard_dir", default="shards", 
                    help="Directory to store output files (default: shards)")
    ap.add_argument("--target", type=int, default=2000,
                    help="Target number of samples per trait (default: 2000)")
    ap.add_argument("--seed_files", nargs="+", required=True,
                    help="One or more files containing seed phrases (JSONL or plain text)")
    ap.add_argument("--list-traits", action="store_true",
                    help="List all available trait buckets and exit")
    ap.add_argument("buckets", nargs="*",
                    help="Trait buckets to generate (e.g., intellect_plus wisdom_minus)")
    args = ap.parse_args()

    if args.list_traits:
        list_available_traits()
        sys.exit(0)

    if not args.buckets:
        print("Error: No trait buckets specified.")
        list_available_traits()
        sys.exit(1)

    # Validate all bucket names
    invalid_buckets = [b for b in args.buckets if b not in TRAIT_CONFIG]
    if invalid_buckets:
        print(f"Error: Invalid trait buckets: {invalid_buckets}")
        list_available_traits()
        sys.exit(1)

    seed_pool = load_seeds(args.seed_files)
    if not seed_pool:
        raise SystemExit("Seed pool empty – check --seed_files paths.")
    
    print(f"Loaded {len(seed_pool)} seed phrases from {len(args.seed_files)} files.")
    shard_dir = Path(args.shard_dir); shard_dir.mkdir(exist_ok=True)

    async def main():
        await asyncio.gather(*[
            fill_bucket(b, seed_pool, args.target, shard_dir)
            for b in args.buckets
        ])
    
    asyncio.run(main())
