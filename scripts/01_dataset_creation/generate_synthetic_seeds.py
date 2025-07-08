#!/usr/bin/env python
"""
generate_synthetic_seeds.py - Extract neutral phrases from MELD for synthetic bucket generation

This script extracts neutral and low-emotional-intensity utterances from the MELD dataset
to serve as seeds for the synthetic_sparse_buckets.py script. It focuses on emotionally 
neutral content that can be effectively paraphrased into different personality traits.

Usage:
    python generate_synthetic_seeds.py --output seeds/meld_neutral_seeds.jsonl --max_samples 5000

Dependencies:
    pip install -U datasets tqdm
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import random
import re

# Emotions to consider as neutral/low-intensity for seed generation
NEUTRAL_EMOTIONS = {
    "neutral", "no_emotion", "content", "impressed", "prepared", "trusting", "sentimental"
}

# Additional emotions with lower emotional intensity that could work as seeds
LOW_INTENSITY_EMOTIONS = {
    "anticipating", "caring", "confident", "faithful", "grateful", "hopeful", "nostalgic", "proud"
}

def clean_utterance(text):
    """Clean and normalize utterance text for seed generation."""
    if not text:
        return None
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common dataset artifacts
    artifacts = [
        'hit:', 'conv:', '_conv:', ',5|5|5', '|5|5|',
        'utterance_id:', 'dialogue_id:', 'session_id:',
        '\n\n', '\t\t', '|||', ':::', '___'
    ]
    
    text_lower = text.lower()
    for artifact in artifacts:
        if artifact in text_lower:
            return None
    
    # Remove utterances that are just punctuation or very short
    if len(text.strip()) < 8:
        return None
    
    # Remove utterances that are mostly punctuation
    non_punct_chars = re.sub(r'[^\w\s]', '', text)
    if len(non_punct_chars) < 5:
        return None
    
    return text

def is_good_seed(utterance, min_words=8, max_words=30):
    """Check if an utterance makes a good seed for paraphrasing."""
    if not utterance:
        return False, "empty"
    
    words = utterance.split()
    word_count = len(words)
    
    # Check word count range
    if word_count < min_words:
        return False, "too_short"
    if word_count > max_words:
        return False, "too_long"
    
    # Check for meaningful content
    # Avoid utterances that are just greetings or very simple phrases
    simple_patterns = [
        r'^\s*(hi|hello|hey|ok|okay|yeah|yes|no|sure|thanks|thank you)\s*[.!?]*\s*$',
        r'^\s*(goodbye|bye|see you|talk to you)\s*[.!?]*\s*$',
        r'^\s*(what|how|when|where|why)\s*[?]*\s*$',
        r'^\s*(hmm|uh|um|ah|oh)\s*[.!?]*\s*$'
    ]
    
    utterance_lower = utterance.lower().strip()
    for pattern in simple_patterns:
        if re.match(pattern, utterance_lower):
            return False, "too_simple"
    
    # Check for good seed characteristics
    # Seeds should have some substance but not be too emotionally charged
    
    # Avoid questions that are too specific or personal
    question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
    if any(word in utterance_lower for word in question_words) and word_count < 12:
        return False, "simple_question"
    
    # Prefer utterances with some descriptive content
    descriptive_words = [
        'think', 'believe', 'feel', 'seems', 'looks', 'sounds', 'means',
        'understand', 'know', 'remember', 'imagine', 'consider', 'suppose',
        'probably', 'maybe', 'perhaps', 'actually', 'really', 'quite',
        'pretty', 'rather', 'somewhat', 'fairly', 'definitely', 'certainly'
    ]
    
    has_descriptive = any(word in utterance_lower for word in descriptive_words)
    
    # Prefer sentences with some complexity but not overly complex
    sentence_complexity = (
        len([c for c in utterance if c in '.,;:']) +  # punctuation
        len([w for w in words if w.lower() in ['and', 'but', 'or', 'because', 'since', 'while', 'although']]) +  # conjunctions
        (1 if has_descriptive else 0)
    )
    
    if sentence_complexity == 0 and word_count < 15:
        return False, "too_basic"
    
    return True, "good_seed"

def load_meld_seeds(split="train", max_samples=None):
    """Load neutral utterances from MELD dataset as seeds."""
    
    print(f"Loading MELD dataset ({split} split)...")
    
    try:
        ds = load_dataset(
            "ajyy/MELD_audio", 
            "MELD_Audio", 
            split=split,
            trust_remote_code=True,
        )
        print(f"📥 Loaded {len(ds):,} samples from MELD")
    except Exception as e:
        print(f"Error loading MELD dataset: {e}")
        return []
    
    seeds = []
    emotion_counts = {}
    skip_counts = {}
    
    for idx, row in enumerate(tqdm(ds, desc="Processing MELD")):
        try:
            emotion = row["emotion"].lower()
            utterance = row["text"]
            
            # Track emotion distribution
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Skip if not a target emotion
            if emotion not in NEUTRAL_EMOTIONS and emotion not in LOW_INTENSITY_EMOTIONS:
                continue
            
            # Clean the utterance
            cleaned = clean_utterance(utterance)
            if not cleaned:
                skip_counts["cleaning_failed"] = skip_counts.get("cleaning_failed", 0) + 1
                continue
            
            # Check if it's a good seed
            is_good, reason = is_good_seed(cleaned)
            if not is_good:
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                continue
            
            # Add to seeds
            seed_entry = {
                "utterance": cleaned,
                "source": "MELD",
                "emotion": emotion,
                "origin_id": str(idx),
                "word_count": len(cleaned.split())
            }
            seeds.append(seed_entry)
            
            # Break if we've reached max samples
            if max_samples and len(seeds) >= max_samples:
                break
                
        except Exception as e:
            print(f"Error processing MELD row {idx}: {e}")
            continue
    
    # Print statistics
    print(f"\n📊 MELD emotion distribution: {emotion_counts}")
    print(f"📊 Skip reasons: {skip_counts}")
    print(f"✅ Extracted {len(seeds):,} neutral seeds from MELD")
    
    return seeds

def save_seeds(seeds, output_path):
    """Save seeds to JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for seed in seeds:
            json.dump(seed, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"💾 Saved {len(seeds):,} seeds to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate seed files from MELD dataset")
    parser.add_argument("--output", default="seeds/meld_neutral_seeds.jsonl", 
                       help="Output path for seed file")
    parser.add_argument("--max_samples", type=int, default=5000,
                       help="Maximum number of seed samples to extract")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"],
                       help="Dataset split to use")
    parser.add_argument("--include_low_intensity", action="store_true",
                       help="Include low-intensity emotions as seeds")
    
    args = parser.parse_args()
    
    # Load seeds from MELD
    seeds = load_meld_seeds(split=args.split, max_samples=args.max_samples)
    
    if not seeds:
        print("❌ No seeds extracted. Check dataset availability and filters.")
        return
    
    # Shuffle for variety
    random.shuffle(seeds)
    
    # Save seeds
    save_seeds(seeds, args.output)
    
    # Print sample seeds
    print(f"\n📝 Sample seeds:")
    for i, seed in enumerate(seeds[:5]):
        print(f"  {i+1}. [{seed['emotion']}] {seed['utterance']}")
    
    print(f"\n🎯 Seeds ready for use with synthetic_sparse_buckets.py:")
    print(f"python scripts/py/synthetic_sparse_buckets.py --seed_files {args.output} --target 2000 intellect_plus discipline_plus")

if __name__ == "__main__":
    main() 