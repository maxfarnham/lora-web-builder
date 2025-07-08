r"""
Quick‑start core trait corpus bootstrapper for the Persona‑to‑LoRA POC.
Author: maxfarnham@gmail.com

Processes emotion-labeled datasets to create bipolar trait annotations:
- EmpatheticDialogues: 25k+ conversations with emotion context
- DailyDialog: 13k+ dialogues with emotion IDs
- EmotionLines: 29k+ utterances from Friends TV show
- MELD: 13k+ multimodal emotional dialogues
- IEMOCAP: 13k+ acted emotional conversations

Example:
    python bootstrap_trait_corpus.py ^
        --output_dir D:\code\lora-web-builder\data\persona\core_traits ^
        --output_format parquet

Dependencies:
    pip install -U datasets tqdm pandas hf_transfer duckdb
"""
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import duckdb
import pandas as pd
import logging

# --------------------------------------------------------------------------- #
# Load unified emotion-to-trait mappings
def load_emotion_mappings():
    """Load the unified emotion-to-trait mappings from JSON file."""
    script_dir = Path(__file__).parent
    mapping_file = script_dir / "emotion_trait_mappings.json"
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Global mappings loaded once
EMOTION_MAPPINGS = load_emotion_mappings()
TRAITS = EMOTION_MAPPINGS["_metadata"]["trait_names"]

# Debug logging setup
DEBUG_MODE = False
def debug_log(message, level="INFO"):
    """Log debug messages when debug mode is enabled."""
    if DEBUG_MODE:
        if level == "WARNING":
            print(f"⚠️  DEBUG: {message}")
        elif level == "ERROR":
            print(f"❌ DEBUG: {message}")
        elif level == "SUCCESS":
            print(f"✅ DEBUG: {message}")
        else:
            print(f"🔍 DEBUG: {message}")

def debug_sample_traits(utterance, traits, source, tag, reason=""):
    """Debug log for trait assignments."""
    if DEBUG_MODE:
        active_traits = {k: v for k, v in traits.items() if v != 0}
        zero_traits = [k for k, v in traits.items() if v == 0]
        print(f"🎯 TRAIT DEBUG [{source}]: {tag}")
        print(f"   Utterance: {utterance[:100]}{'...' if len(utterance) > 100 else ''}")
        print(f"   Active traits: {active_traits}")
        print(f"   Zero traits: {len(zero_traits)}/10")
        if reason:
            print(f"   Reason: {reason}")
        print()

# --------------------------------------------------------------------------- #
def extract_dialog(row):
    """
    Return a flat list[str] of utterances in order, or None if format unknown.
    Handles the most common field layouts across open‑source dialogue datasets.
    """
    if "dialog" in row and row["dialog"]:
        return row["dialog"]
    if "dialogue" in row and row["dialogue"]:
        return row["dialogue"]
    if "utterances" in row and row["utterances"]:
        uts = row["utterances"]
        if isinstance(uts, list):
            # PersonaChat true‑cased = list[dict{text, user_index}]
            return [u["text"] if isinstance(u, dict) else str(u) for u in uts]
        if isinstance(uts, dict) and "text" in uts:
            return uts["text"]
    if "sentences" in row and row["sentences"]:
        return row["sentences"]
    # EmpatheticDialogues fallback: context + response
    if "context" in row and "utterance" in row:
        return [row["context"], row["utterance"]]
    return None


def create_empty_traits():
    """Create a dictionary with all traits initialized to 0."""
    return {trait: 0 for trait in TRAITS}


def is_meaningful_utterance(utterance, min_length=15):
    """Check if an utterance is meaningful enough for trait annotation."""
    # Check for data corruption indicators
    corruption_indicators = [
        'hit:', 'conv:', '_conv:', ',5|5|5', '|5|5|',  # EmpatheticDialogues corruption
        'utterance_id:', 'dialogue_id:', 'session_id:',  # Common ID patterns
        '\n\n', '\t\t', '|||', ':::', '___'  # Multiple separators
    ]
    
    if any(indicator in utterance.lower() for indicator in corruption_indicators):
        return False, "corrupted data"
    
    # Check for suspiciously repetitive patterns (like the repeating utterance)
    if len(utterance) > 200:
        # Look for repeated phrases longer than 50 characters
        words = utterance.split()
        if len(words) > 20:
            # Check if the first 10 words appear again later in the utterance
            first_phrase = ' '.join(words[:10])
            rest_text = ' '.join(words[10:])
            if first_phrase in rest_text:
                return False, "repetitive corruption"
    
    # Basic length check
    if len(utterance) < min_length:
        # Allow shorter utterances if they contain emotional keywords
        emotional_keywords = [
            'feel', 'love', 'hate', 'happy', 'sad', 'angry', 'excited', 'afraid', 'worried',
            'great', 'terrible', 'awesome', 'awful', 'wonderful', 'horrible', 'amazing',
            'good', 'bad', 'sorry', 'thank', 'please', 'help', 'stupid', 'smart', 'funny'
        ]
        
        utterance_lower = utterance.lower()
        has_emotional_content = any(keyword in utterance_lower for keyword in emotional_keywords)
        
        if not has_emotional_content:
            return False, "too short and no emotional content"
    
    # Check for extremely short utterances that are just noise
    if len(utterance.strip()) < 5:
        return False, "too short"
    
    # Check for utterances that are just punctuation or numbers
    if len(utterance.strip().replace(' ', '').replace('.', '').replace('!', '').replace('?', '').replace(',', '')) < 3:
        return False, "mostly punctuation"
    
    return True, "meaningful"

def add_to_corpus(corpus, utterance, traits, source, origin_id, original_tag=None):
    """Add a single utterance sample with its trait values to the corpus.
    
    Returns:
        tuple: (success: bool, skip_reason: str or None)
    """
    # Debug logging for potential issues
    utterance_length = len(utterance)
    word_count = len(utterance.split())
    
    # Check for extreme lengths
    if utterance_length > 2000:
        debug_log(f"EXTREME LENGTH utterance in {source}: {utterance_length} chars", "ERROR")
        debug_log(f"Content preview: {utterance[:200]}...", "ERROR")
        return False, "extreme_length"
    
    # Check utterance meaningfulness
    is_meaningful, reason = is_meaningful_utterance(utterance)
    if not is_meaningful:
        return False, reason
    
    # Check for all-zero traits
    active_traits = {k: v for k, v in traits.items() if v != 0}
    if not active_traits:
        debug_log(f"ALL-ZERO TRAITS in {source}: tag={original_tag}, utterance='{utterance[:50]}...'", "ERROR")
        # Still add to corpus but flag for investigation

    # Debug trait assignments (sample first few from each source)
    if len(corpus) % 10000 == 0:  # Every 10,000 samples
        debug_sample_traits(utterance, traits, source, original_tag, f"Sample #{len(corpus)}")
    
    sample = {
        "utterance": utterance,
        "source": source,
        "origin_id": origin_id,
        "original_tag": original_tag,
        "utterance_length": utterance_length,
        "word_count": word_count,
        **{f"trait_{trait}": value for trait, value in traits.items()}
    }
    corpus.append(sample)
    return True, None

# --------------------------------------------------------------------------- #

def handle_empathetic(corpus, split="train"):
    """EmpatheticDialogues - map emotions to multiple bipolar trait axes with signed values.
    
    KNOWN ISSUE: HuggingFace dataset has CSV parsing corruption causing field bleeding.
    Raw CSV shows broken quote escaping and _comma_ replacements, but HuggingFace's
    CSV-to-Arrow conversion fails on these patterns, creating extremely long corrupted
    utterances with metadata bleeding into text fields.
    
    TODO: Replace HuggingFace loading with direct CSV processing from original source:
    https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
    This would require:
    1. Download and extract the tar.gz
    2. Parse CSV with custom logic to handle _comma_ replacements
    3. Handle broken quote escaping manually
    4. Validate field boundaries and skip malformed rows
    """
    
    debug_log("Starting EmpatheticDialogues processing...")
    
    # Use unified emotion mappings from JSON
    emotion_to_traits = EMOTION_MAPPINGS["emotions"]
    empathetic_listener_traits = EMOTION_MAPPINGS["empathetic_listener_traits"]["traits"]
    
    debug_log(f"Available emotions in mapping: {list(emotion_to_traits.keys())}")
    debug_log(f"Empathetic listener traits: {empathetic_listener_traits}")
    
    ds = load_dataset(
        "facebook/empathetic_dialogues",
        split=split,
        trust_remote_code=True,
    )
    
    print(f"📥 Loaded {len(ds):,} samples from EmpatheticDialogues")
    
    # First pass: identify which speaker_idx corresponds to the emotional speaker
    # We'll track the first speaker who mentions the emotional situation
    conv_to_emotional_speaker = {}
    missing_emotions = set()
    
    for row in tqdm(ds, desc="EmpatheticDialogues - Pass 1: Identifying roles"):
        if row["context"] not in emotion_to_traits:
            missing_emotions.add(row["context"])
            continue
            
        conv_id = row["conv_id"]
        if conv_id not in conv_to_emotional_speaker:
            # The first speaker in the conversation is typically the emotional speaker
            conv_to_emotional_speaker[conv_id] = row["speaker_idx"]
    
    if missing_emotions:
        debug_log(f"Missing emotions in EmpatheticDialogues mapping: {missing_emotions}", "WARNING")
    
    debug_log(f"Identified {len(conv_to_emotional_speaker)} conversations with emotional speakers")
    
    # Second pass: process utterances with correct role assignments
    processed_count = 0
    skipped_emotion = 0
    skipped_length = 0
    skip_counts = {}
    
    for row in tqdm(ds, desc="EmpatheticDialogues - Pass 2: Processing utterances"):
        if row["context"] not in emotion_to_traits:
            skipped_emotion += 1
            continue
            
        conv_id = row["conv_id"]
        emotion_label = row["context"]
        utterance = row["utterance"]
        speaker_idx = row["speaker_idx"]
        
        if not utterance or len(utterance.strip()) < 10:
            skipped_length += 1
            continue
        
        # Determine if this is the emotional speaker or empathetic listener
        emotional_speaker_idx = conv_to_emotional_speaker.get(conv_id)
        is_emotional_speaker = (speaker_idx == emotional_speaker_idx)
        
        traits = create_empty_traits()
        
        if is_emotional_speaker:
            # Check if emotion should be skipped
            emotion_mapping = emotion_to_traits[emotion_label]
            if emotion_mapping.get("skip", False):
                debug_log(f"Skipping emotion '{emotion_label}' due to skip flag", "WARNING")
                continue
            
            # Apply emotion-based traits to emotional speaker utterances
            for trait, value in emotion_mapping["traits"].items():
                traits[trait] = value
            utterance_type = "speaker"
        else:
            # Apply empathetic response traits to listener utterances
            for trait, value in empathetic_listener_traits.items():
                traits[trait] = value
            utterance_type = "listener"
        
        # Debug first few samples to verify trait assignment
        if processed_count < 5:
            debug_sample_traits(utterance, traits, "EmpatheticDialogues", f"{emotion_label}_{utterance_type}", 
                              f"Emotion: {emotion_label}, Role: {utterance_type}")
        
        # Add single utterance to corpus
        success, skip_reason = add_to_corpus(
            corpus, 
            utterance, 
            traits, 
            "EmpatheticDialogues", 
            f"{conv_id}_turn_{row['utterance_idx']}", 
            f"{emotion_label}_{utterance_type}"
        )
        
        if success:
            processed_count += 1
        else:
            skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
    
    # Report aggregated skip counts
    skip_summary = f"EmpatheticDialogues: Processed {processed_count}, Skipped {skipped_emotion} (no emotion), {skipped_length} (too short)"
    if skip_counts:
        skip_details = ", ".join([f"{count} ({reason})" for reason, count in skip_counts.items()])
        skip_summary += f", {skip_details}"
    print(f"✅ {skip_summary}")

def handle_dailydialog(corpus, split="train"):
    """DailyDialog - map emotion IDs to multiple bipolar trait axes."""
    
    debug_log("Starting DailyDialog processing...")
    
    # Use unified emotion mappings from JSON for DailyDialog numeric IDs
    numeric_emotion_mapping = EMOTION_MAPPINGS["dailydialog_numeric_mapping"]
    
    debug_log(f"Available numeric emotion mappings: {list(numeric_emotion_mapping.keys())}")
    for emotion_id, mapping in numeric_emotion_mapping.items():
        debug_log(f"  {emotion_id}: {mapping['emotion_name']} -> {mapping['traits']}")
    
    ds = load_dataset(
        "roskoN/dailydialog",
        split=split,
        trust_remote_code=True,
    )
    
    print(f"📥 Loaded {len(ds):,} samples from DailyDialog")
    
    processed_count = 0
    emotion_counts = {}
    missing_emotions = set()
    skip_counts = {}
    
    for idx, row in tqdm(enumerate(ds), total=len(ds), desc="DailyDialog"):
        dlg = extract_dialog(row)
        if dlg and row["emotions"]:
            # Process each utterance with its corresponding emotion
            for utterance_idx, utterance in enumerate(dlg):
                if utterance_idx < len(row["emotions"]):
                    emotion_id = row["emotions"][utterance_idx]
                else:
                    emotion_id = 0  # Default to no emotion if emotions list is shorter
                
                # Track emotion distribution
                emotion_counts[emotion_id] = emotion_counts.get(emotion_id, 0) + 1
                
                emotion_id_str = str(emotion_id)
                if emotion_id_str in numeric_emotion_mapping:
                    emotion_mapping = numeric_emotion_mapping[emotion_id_str]
                    
                    # Check if emotion should be skipped
                    if emotion_mapping.get("skip", False):
                        debug_log(f"Skipping emotion ID '{emotion_id}' ({emotion_mapping['emotion_name']}) due to skip flag", "WARNING")
                        continue
                    
                    traits = create_empty_traits()
                    # Apply all trait mappings for this emotion
                    for trait, value in emotion_mapping["traits"].items():
                        traits[trait] = value
                    
                    # Debug first few samples to verify trait assignment
                    if processed_count < 5:
                        debug_sample_traits(utterance, traits, "DailyDialog", 
                                          emotion_mapping["emotion_name"], 
                                          f"Emotion ID: {emotion_id}")
                    
                    # Add single utterance to corpus
                    success, skip_reason = add_to_corpus(
                        corpus, 
                        utterance, 
                        traits, 
                        "DailyDialog", 
                        f"{idx}_{utterance_idx}", 
                        emotion_mapping["emotion_name"]
                    )
                    
                    if success:
                        processed_count += 1
                    else:
                        skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
                else:
                    missing_emotions.add(emotion_id)
    
    print(f"📊 DailyDialog emotion distribution: {emotion_counts}")
    if missing_emotions:
        print(f"⚠️  Missing emotion IDs in DailyDialog mapping: {missing_emotions}")
    
    # Report aggregated skip counts
    skip_summary = f"DailyDialog: Processed {processed_count} utterances"
    if skip_counts:
        skip_details = ", ".join([f"{count} ({reason})" for reason, count in skip_counts.items()])
        skip_summary += f", Skipped {skip_details}"
    print(f"✅ {skip_summary}")

# --------------------------------------------------------------------------- #
# New dataset handlers using unified emotion mappings

def handle_emotionlines(corpus, split="train"):
    """EmotionLines - map emotion labels to multiple bipolar trait axes."""
    
    debug_log("Starting EmotionLines processing...")
    
    try:
        # Use the emotion dataset which is based on similar data
        ds = load_dataset(
            "dair-ai/emotion",
            "split", 
            split=split,
            trust_remote_code=True,
        )
        print(f"📥 Loaded {len(ds):,} samples from EmotionLines")
    except Exception as e:
        print(f"Warning: Could not load EmotionLines dataset: {e}")
        return
    
    # Use unified emotion mappings from JSON
    emotion_to_traits = EMOTION_MAPPINGS["emotions"]
    
    # Map dair-ai/emotion dataset labels to our emotion names
    # Labels: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)
    label_to_emotion = {
        0: "sad",
        1: "joy", 
        2: "love",  # Map love to joy for now since we don't have love in our mappings
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    
    processed_count = 0
    skipped_short = 0
    skipped_label = 0
    skipped_emotion = 0
    origin_ids = set()
    skip_counts = {}
    
    # Debug first few rows to understand structure
    debug_log("First few EmotionLines rows:")
    for i, row in enumerate(ds):
        if i >= 3:
            break
        debug_log(f"  Row {i}: {dict(row)}")
    
    for idx, row in enumerate(tqdm(ds, desc="EmotionLines (dair-ai/emotion)")):
        try:
            # Expected fields: "label", "text"
            emotion_label = row["label"]
            utterance = row["text"]
            
            if not utterance or len(utterance.strip()) < 10:
                skipped_short += 1
                continue
            
            # Map numeric label to emotion name
            if emotion_label not in label_to_emotion:
                skipped_label += 1
                continue
                
            emotion = label_to_emotion[emotion_label]
            
            # Handle love -> joy mapping since we don't have love in our unified mapping
            if emotion == "love":
                emotion = "joy"
                
            if emotion not in emotion_to_traits:
                skipped_emotion += 1
                continue
            
            # Check if emotion should be skipped
            emotion_mapping = emotion_to_traits[emotion]
            if emotion_mapping.get("skip", False):
                debug_log(f"Skipping emotion '{emotion}' due to skip flag", "WARNING")
                continue
                
            traits = create_empty_traits()
            # Apply all trait mappings for this emotion
            for trait, value in emotion_mapping["traits"].items():
                traits[trait] = value
            
            # Debug origin ID assignment
            origin_id = str(row.get("id", f"row_{idx}"))
            origin_ids.add(origin_id)
            
            # Debug first few samples
            if processed_count < 5:
                debug_sample_traits(utterance, traits, "EmotionLines", emotion, 
                                  f"Label: {emotion_label}, Origin ID: {origin_id}")
            
            # Add single utterance to corpus
            success, skip_reason = add_to_corpus(
                corpus, 
                utterance, 
                traits, 
                "EmotionLines", 
                origin_id, 
                emotion
            )
            
            if success:
                processed_count += 1
            else:
                skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
        except KeyError as e:
            print(f"KeyError in EmotionLines row {idx}: {e}")
            print(f"Available fields: {list(row.keys()) if hasattr(row, 'keys') else 'Not dict-like'}")
            print(f"Row content: {row}")
            raise  # Re-raise to crash with full traceback
    
    # Report aggregated skip counts
    skip_summary = f"EmotionLines: Processed {processed_count}, Skipped {skipped_short} (short), {skipped_label} (bad label), {skipped_emotion} (no emotion)"
    if skip_counts:
        skip_details = ", ".join([f"{count} ({reason})" for reason, count in skip_counts.items()])
        skip_summary += f", {skip_details}"
    print(f"✅ {skip_summary}")
    print(f"📊 EmotionLines: Found {len(origin_ids)} unique origin IDs")

def handle_meld(corpus, split="train"):
    """MELD - map emotion labels to multiple bipolar trait axes."""
    
    debug_log("Starting MELD processing...")
    
    try:
        ds = load_dataset(
            "ajyy/MELD_audio", 
            "MELD_Audio", 
            split=split,
            trust_remote_code=True,
        )
        print(f"📥 Loaded {len(ds):,} samples from MELD")
    except Exception as e:
        print(f"Warning: Could not load MELD dataset: {e}")
        return
    
    # Use unified emotion mappings from JSON
    emotion_to_traits = EMOTION_MAPPINGS["emotions"]
    
    # Debug first few rows to understand structure
    debug_log("First few MELD rows:")
    for i, row in enumerate(ds):
        if i >= 3:
            break
        debug_log(f"  Row {i}: {dict(row)}")
    
    processed_count = 0
    skipped_short = 0
    skipped_emotion = 0
    emotion_counts = {}
    missing_emotions = set()
    skip_counts = {}
    
    for idx, row in enumerate(tqdm(ds, desc="MELD")):
        try:
            # Expected fields: "Emotion", "Utterance", "Utterance_ID"
            emotion = row["emotion"].lower()
            utterance = row["text"]
            
            # Track emotion distribution
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if not emotion or not utterance or len(utterance.strip()) < 10:
                skipped_short += 1
                continue
                
            if emotion not in emotion_to_traits:
                missing_emotions.add(emotion)
                skipped_emotion += 1
                continue
            
            # Check if emotion should be skipped
            emotion_mapping = emotion_to_traits[emotion]
            if emotion_mapping.get("skip", False):
                debug_log(f"Skipping emotion '{emotion}' due to skip flag", "WARNING")
                continue
                
            traits = create_empty_traits()
            # Apply all trait mappings for this emotion
            for trait, value in emotion_mapping["traits"].items():
                traits[trait] = value
            
            # Debug first few samples
            if processed_count < 5:
                debug_sample_traits(utterance, traits, "MELD", emotion, 
                                  f"Original emotion: {row['emotion']}")
            
            # Add single utterance to corpus
            success, skip_reason = add_to_corpus(
                corpus, 
                utterance, 
                traits,
                "MELD",
                str(idx), 
                emotion
            )
            
            if success:
                processed_count += 1
            else:
                skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
        except KeyError as e:
            print(f"KeyError in MELD row {idx}: {e}")
            print(f"Available fields: {list(row.keys()) if hasattr(row, 'keys') else 'Not dict-like'}")
            print(f"Row content: {row}")
            raise  # Re-raise to crash with full traceback
    
    print(f"📊 MELD emotion distribution: {emotion_counts}")
    if missing_emotions:
        print(f"⚠️  Missing emotions in MELD mapping: {missing_emotions}")
    
    # Report aggregated skip counts
    skip_summary = f"MELD: Processed {processed_count}, Skipped {skipped_short} (short), {skipped_emotion} (no emotion)"
    if skip_counts:
        skip_details = ", ".join([f"{count} ({reason})" for reason, count in skip_counts.items()])
        skip_summary += f", {skip_details}"
    print(f"✅ {skip_summary}")

def handle_iemocap(corpus, split="train"):
    """IEMOCAP - use float emotion activations to create weighted trait combinations."""
    
    debug_log("Starting IEMOCAP processing...")
    
    try:
        ds = load_dataset(
            "AbstractTTS/IEMOCAP",
            split=split,
            trust_remote_code=True,
        )
        print(f"📥 Loaded {len(ds):,} samples from IEMOCAP")
    except Exception as e:
        print(f"Warning: Could not load IEMOCAP dataset: {e}")
        return
    
    # Use unified emotion mappings from JSON
    emotion_to_traits = EMOTION_MAPPINGS["emotions"]
    
    # IEMOCAP emotion fields with float activations
    iemocap_emotions = ['frustrated', 'angry', 'sad', 'disgust', 'excited', 'fear', 'neutral', 'surprise', 'happy']
    
    # Map IEMOCAP emotions to our unified emotion names
    iemocap_to_unified = {
        'frustrated': 'anger',  # frustration maps to anger
        'angry': 'anger',
        'sad': 'sad',
        'disgust': 'disgust',
        'excited': 'excitement',
        'fear': 'fear',
        'neutral': 'neutral',  # Skip neutral or map to low activation
        'surprise': 'surprise',
        'happy': 'joy'
    }
    
    processed_count = 0
    skip_counts = {}
    
    for idx, row in enumerate(tqdm(ds, desc="IEMOCAP")):
        try:
            # Expected fields: emotion floats, "transcription", "file"
            utterance = row["transcription"]
            
            if not utterance or len(utterance.strip()) < 10:
                continue
            
            # Create weighted trait combination from emotion activations
            traits = create_empty_traits()
            
            # Process each emotion activation
            for iemocap_emotion in iemocap_emotions:
                activation = row[iemocap_emotion]
                
                # Only process emotions with significant activation (> 0.1)
                if activation > 0.1:
                    unified_emotion = iemocap_to_unified.get(iemocap_emotion)
                    if unified_emotion and unified_emotion in emotion_to_traits:
                        emotion_mapping = emotion_to_traits[unified_emotion]
                        
                        # Check if emotion should be skipped
                        if emotion_mapping.get("skip", False):
                            debug_log(f"Skipping emotion '{unified_emotion}' (from {iemocap_emotion}) due to skip flag", "WARNING")
                            continue
                        
                        # Weight the trait values by the emotion activation
                        for trait, base_value in emotion_mapping["traits"].items():
                            traits[trait] += base_value * activation
            
            # Optionally use the emotional dimensions as additional traits
            # EmoAct: activation level (could map to energy/neuroticism)
            # EmoVal: valence (could map to joy vs sadness)
            # EmoDom: dominance (could map to courage vs submissiveness)
            
            emo_act = row.get("EmoAct", 0)  # 1-5 scale
            emo_val = row.get("EmoVal", 0)  # 1-5 scale  
            emo_dom = row.get("EmoDom", 0)  # 1-5 scale
            
            # Map emotional dimensions to traits (normalize from 1-5 to -1 to 1)
            if emo_act > 0:
                # Higher activation = more neurotic/energetic
                traits["neuroticism"] += (emo_act - 3) * 0.5  # Center around 3, scale to [-1, 1]
            
            if emo_val > 0:
                # Higher valence = more joyful
                traits["joy"] += (emo_val - 3) * 0.5
            
            if emo_dom > 0:
                # Higher dominance = more courageous
                traits["courage"] += (emo_dom - 3) * 0.5
            
            # Clamp all trait values to [-1, 1] range
            for trait in traits:
                traits[trait] = max(-1.0, min(1.0, traits[trait]))
            
            # Use major_emotion as the tag for reference
            major_emotion = row.get("major_emotion", "unknown")
            
            # If no significant activations were found but major_emotion is neutral, apply neutral traits
            if all(traits[trait] == 0 for trait in traits) and major_emotion == "neutral":
                if "neutral" in emotion_to_traits:
                    neutral_mapping = emotion_to_traits["neutral"]
                    
                    # Check if neutral should be skipped
                    if not neutral_mapping.get("skip", False):
                        for trait, value in neutral_mapping["traits"].items():
                            traits[trait] = value
                    else:
                        debug_log(f"Skipping neutral emotion due to skip flag", "WARNING")
            
            # Add single utterance to corpus
            success, skip_reason = add_to_corpus(
                corpus, 
                utterance, 
                traits, 
                "IEMOCAP", 
                row["file"], 
                major_emotion
            )
            
            if success:
                processed_count += 1
            else:
                skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
        except KeyError as e:
            print(f"KeyError in IEMOCAP row {idx}: {e}")
            print(f"Available fields: {list(row.keys()) if hasattr(row, 'keys') else 'Not dict-like'}")
            print(f"Row content: {row}")
            raise  # Re-raise to crash with full traceback
    
    # Report aggregated skip counts
    skip_summary = f"IEMOCAP: Processed {processed_count} utterances"
    if skip_counts:
        skip_details = ", ".join([f"{count} ({reason})" for reason, count in skip_counts.items()])
        skip_summary += f", Skipped {skip_details}"
    print(f"✅ {skip_summary}")

# --------------------------------------------------------------------------- #
def create_duckdb_table(corpus):
    """Create a DuckDB table from the corpus data."""
    print("Creating DuckDB table...")
    
    # Create DataFrame from corpus
    df = pd.DataFrame(corpus)
    
    # Create in-memory DuckDB connection
    conn = duckdb.connect(':memory:')
    
    # Create table from DataFrame
    conn.execute("CREATE TABLE corpus_raw AS SELECT * FROM df")
    
    return conn

def apply_sampling(conn):
    """Apply sampling to reduce bias from dominant neutral categories and balance sources."""
    print("Applying sampling to reduce dataset bias...")
    
    # Count samples before sampling
    before_counts = conn.execute("""
        SELECT source, original_tag, COUNT(*) as count
        FROM corpus_raw 
        WHERE (source = 'DailyDialog' AND original_tag = 'no_emotion')
           OR (source = 'MELD' AND original_tag = 'neutral')
        GROUP BY source, original_tag
        ORDER BY source, original_tag
    """).fetchall()
    
    print("Before sampling:")
    for source, tag, count in before_counts:
        print(f"  {source} {tag}: {count:,}")
    
    # Show source distribution before sampling
    source_counts = conn.execute("""
        SELECT source, COUNT(*) as count
        FROM corpus_raw 
        GROUP BY source
        ORDER BY count DESC
    """).fetchall()
    
    print("\nSource distribution before sampling:")
    for source, count in source_counts:
        print(f"  {source}: {count:,}")
    
    # Create balanced corpus with multi-level sampling
    conn.execute("""
        CREATE TABLE corpus AS (
            -- Sample 50% of EmpatheticDialogues to reduce dominance
            SELECT * FROM corpus_raw 
            WHERE source = 'EmpatheticDialogues'
            USING SAMPLE 50% (bernoulli)
            
            UNION ALL
            
            -- Keep DailyDialog but sample down no_emotion heavily
            SELECT * FROM corpus_raw 
            WHERE source = 'DailyDialog' AND original_tag != 'no_emotion'
            
            UNION ALL
            
            -- Sample 10% of DailyDialog no_emotion
            SELECT * FROM corpus_raw 
            WHERE source = 'DailyDialog' AND original_tag = 'no_emotion'
            USING SAMPLE 10% (bernoulli)
            
            UNION ALL
            
            -- Keep EmotionLines unchanged (good emotional diversity)
            SELECT * FROM corpus_raw 
            WHERE source = 'EmotionLines'
            
            UNION ALL
            
            -- Keep MELD but sample down neutral
            SELECT * FROM corpus_raw 
            WHERE source = 'MELD' AND original_tag != 'neutral'
            
            UNION ALL
            
            -- Sample 25% of MELD neutral
            SELECT * FROM corpus_raw 
            WHERE source = 'MELD' AND original_tag = 'neutral'
            USING SAMPLE 25% (bernoulli)
            
            UNION ALL
            
            -- Keep IEMOCAP unchanged (good emotional granularity)
            SELECT * FROM corpus_raw 
            WHERE source = 'IEMOCAP'
        )
    """)
    
    # Count samples after sampling
    after_counts = conn.execute("""
        SELECT source, original_tag, COUNT(*) as count
        FROM corpus 
        WHERE (source = 'DailyDialog' AND original_tag = 'no_emotion')
           OR (source = 'MELD' AND original_tag = 'neutral')
        GROUP BY source, original_tag
        ORDER BY source, original_tag
    """).fetchall()
    
    print("\nAfter sampling:")
    for source, tag, count in after_counts:
        print(f"  {source} {tag}: {count:,}")
    
    # Show source distribution after sampling
    source_counts_after = conn.execute("""
        SELECT source, COUNT(*) as count
        FROM corpus 
        GROUP BY source
        ORDER BY count DESC
    """).fetchall()
    
    print("\nSource distribution after sampling:")
    for source, count in source_counts_after:
        print(f"  {source}: {count:,}")
    
    # Show total sample reduction
    total_before = conn.execute("SELECT COUNT(*) FROM corpus_raw").fetchone()[0]
    total_after = conn.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
    reduction = total_before - total_after
    reduction_pct = (reduction / total_before) * 100
    
    print(f"\nTotal samples: {total_before:,} → {total_after:,} (-{reduction:,}, -{reduction_pct:.1f}%)")
    
    # Drop the raw table to save memory
    conn.execute("DROP TABLE corpus_raw")

def generate_data_quality_report(conn):
    """Generate comprehensive data quality report from DuckDB table."""
    print("\n" + "="*80)
    print("DATA QUALITY REPORT")
    print("="*80)
    
    # Overall statistics
    total_samples = conn.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
    print(f"\nTotal samples: {total_samples:,}")
    
    # Per-source statistics
    print("\nPer-source statistics:")
    print("-" * 50)
    
    source_stats = conn.execute("""
        SELECT 
            source,
            COUNT(*) as sample_count,
            AVG(utterance_length) as avg_length,
            MIN(utterance_length) as min_length,
            MAX(utterance_length) as max_length,
            AVG(word_count) as avg_words,
            COUNT(DISTINCT original_tag) as unique_tags,
            COUNT(DISTINCT origin_id) as unique_origins
        FROM corpus 
        GROUP BY source
        ORDER BY sample_count DESC
    """).fetchall()
    
    for row in source_stats:
        source, count, avg_len, min_len, max_len, avg_words, unique_tags, unique_origins = row
        print(f"\n{source}:")
        print(f"  Samples: {count:,}")
        print(f"  Avg length: {avg_len:.1f} chars ({avg_words:.1f} words)")
        print(f"  Length range: {min_len}-{max_len}")
        print(f"  Unique tags: {unique_tags}")
        print(f"  Unique origins: {unique_origins}")
    
    # Trait distribution analysis
    print("\nTrait distribution analysis:")
    print("-" * 50)
    
    for trait in TRAITS:
        trait_col = f"trait_{trait}"
        trait_stats = conn.execute(f"""
            SELECT 
                source,
                COUNT(CASE WHEN {trait_col} > 0 THEN 1 END) as positive,
                COUNT(CASE WHEN {trait_col} < 0 THEN 1 END) as negative,
                COUNT(CASE WHEN {trait_col} = 0 THEN 1 END) as neutral,
                AVG({trait_col}) as avg_value,
                MIN({trait_col}) as min_value,
                MAX({trait_col}) as max_value
            FROM corpus 
            GROUP BY source
            ORDER BY source
        """).fetchall()
        
        print(f"\n{trait.upper()}:")
        for row in trait_stats:
            source, pos, neg, neu, avg_val, min_val, max_val = row
            total = pos + neg + neu
            print(f"  {source:15}: {pos:4}+ {neg:4}- {neu:4}= (avg: {avg_val:6.3f}, range: {min_val:6.3f} to {max_val:6.3f})")
    
    # Tag frequency analysis
    print("\nMost common tags by source:")
    print("-" * 50)
    
    for source in [row[0] for row in source_stats]:
        tag_freq = conn.execute(f"""
            SELECT original_tag, COUNT(*) as count
            FROM corpus 
            WHERE source = '{source}'
            GROUP BY original_tag
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()
        
        print(f"\n{source}:")
        for tag, count in tag_freq:
            print(f"  {tag:20}: {count:,}")
    
    # Quality issues detection
    print("\nQuality issues detection:")
    print("-" * 50)
    
    # Very short utterances
    short_utterances = conn.execute("""
        SELECT source, COUNT(*) as count
        FROM corpus 
        WHERE utterance_length < 20
        GROUP BY source
        ORDER BY count DESC
    """).fetchall()
    
    if short_utterances:
        print("\nVery short utterances (< 20 chars):")
        for source, count in short_utterances:
            print(f"  {source}: {count:,}")
    
    # Very long utterances
    long_utterances = conn.execute("""
        SELECT source, COUNT(*) as count
        FROM corpus 
        WHERE utterance_length > 500
        GROUP BY source
        ORDER BY count DESC
    """).fetchall()
    
    if long_utterances:
        print("\nVery long utterances (> 500 chars):")
        for source, count in long_utterances:
            print(f"  {source}: {count:,}")
    
    # Samples with all-zero traits
    zero_traits = conn.execute("""
        SELECT source, COUNT(*) as count
        FROM corpus 
        WHERE """ + " AND ".join([f"trait_{trait} = 0" for trait in TRAITS]) + """
        GROUP BY source
        ORDER BY count DESC
    """).fetchall()
    
    if zero_traits:
        print("\nSamples with all-zero traits:")
        for source, count in zero_traits:
            print(f"  {source}: {count:,}")
    
    print("\n" + "="*80)

def export_corpus(conn, output_path, output_format):
    """Export corpus data from DuckDB table to specified format."""
    print(f"\nExporting corpus to {output_format.upper()} format...")
    
    if output_format == 'parquet':
        conn.execute(f"COPY corpus TO '{output_path}' (FORMAT PARQUET)")
    elif output_format == 'csv':
        conn.execute(f"COPY corpus TO '{output_path}' (FORMAT CSV, HEADER)")
    elif output_format == 'jsonl':
        # For JSONL, we need to reconstruct the original structure
        rows = conn.execute("""
            SELECT 
                utterance,
                source,
                origin_id,
                original_tag,
                """ + ",".join([f"trait_{trait}" for trait in TRAITS]) + """
            FROM corpus
        """).fetchall()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                utterance, source, origin_id, original_tag = row[:4]
                trait_values = row[4:]
                
                traits = {trait: value for trait, value in zip(TRAITS, trait_values)}
                
                sample = {
                    "utterance": utterance,
                    "traits": traits,
                    "source_metadata": {
                        "source": source,
                        "origin_id": origin_id,
                        "original_tag": original_tag
                    }
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    elif output_format == 'json':
        # Export as single JSON array
        rows = conn.execute("""
            SELECT 
                utterance,
                source,
                origin_id,
                original_tag,
                """ + ",".join([f"trait_{trait}" for trait in TRAITS]) + """
            FROM corpus
        """).fetchall()
        
        samples = []
        for row in rows:
            utterance, source, origin_id, original_tag = row[:4]
            trait_values = row[4:]
            
            traits = {trait: value for trait, value in zip(TRAITS, trait_values)}
            
            sample = {
                "utterance": utterance,
                "traits": traits,
                "source_metadata": {
                    "source": source,
                    "origin_id": origin_id,
                    "original_tag": original_tag
                }
            }
            samples.append(sample)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    print(f"Successfully exported to {output_path}")

def main():
    global DEBUG_MODE
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="../../data/persona/core_traits")
    parser.add_argument("--output_format", default="parquet", 
                       choices=['parquet', 'csv', 'jsonl', 'json'],
                       help="Output format (default: parquet)")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode with detailed logging")
    args = parser.parse_args()
    
    # Enable debug mode if requested
    if args.debug:
        DEBUG_MODE = True
        print("🚀 Debug mode enabled - additional detailed logging will be shown")

    corpus = []
    
    print("📊 Starting corpus processing...")
    
    # Core trait corpus
    print("Processing core trait datasets...")
    handle_empathetic(corpus)
    handle_dailydialog(corpus)
    
    # Additional emotion-labeled datasets
    print("Processing additional emotion-labeled datasets...")
    handle_emotionlines(corpus)
    handle_meld(corpus)
    handle_iemocap(corpus)
    
    print(f"📈 Corpus processing complete. Total samples: {len(corpus):,}")
    
    # Options for semantic tag training:
    # "bavard/personachat_truecased"
    # "google/Synthetic-Persona-Chat"
    # "convai-challenge/convai2"
    # "chujiezheng/wizard_of_wikipedia"
    # blended_skill_talk ?

    # Create DuckDB table
    conn = create_duckdb_table(corpus)
    
    # Apply sampling to reduce bias
    apply_sampling(conn)
    
    # Generate data quality report
    generate_data_quality_report(conn)
    
    # Export to specified format
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Determine output file extension
    extensions = {
        'parquet': 'parquet',
        'csv': 'csv',
        'jsonl': 'jsonl',
        'json': 'json'
    }
    
    output_file = out / f"bipolar_corpus.{extensions[args.output_format]}"
    export_corpus(conn, str(output_file), args.output_format)
    
    # Close connection
    conn.close()
    
    print(f"\nProcessing complete! Data saved to {output_file}")

if __name__ == "__main__":
    main()
