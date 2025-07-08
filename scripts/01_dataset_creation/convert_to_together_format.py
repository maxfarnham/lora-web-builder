#!/usr/bin/env python
"""
Convert trait-based JSONL files to Together AI fine-tuning format.
This script converts files from the format:
{"utterance": "...", "traits": {...}, "source_metadata": {...}}

To the completion format expected by Together AI:
{"prompt": "...", "completion": "..."}
"""
import json
import os
from pathlib import Path

def get_trait_description(trait_name, trait_value):
    """Get a description of the trait based on its value."""
    trait_descriptions = {
        "intellect": ("highly intellectual", "simple"),
        "discipline": ("very disciplined", "undisciplined"),
        "joy": ("joyful", "somber"),
        "wisdom": ("wise", "naive"),
        "compassion": ("compassionate", "unsympathetic"),
        "neuroticism": ("anxious", "calm"),
        "courage": ("courageous", "timid"),
        "humor": ("humorous", "serious"),
        "formality": ("formal", "casual"),
        "sarcasm": ("sarcastic", "straightforward")
    }
    
    if trait_name in trait_descriptions:
        positive, negative = trait_descriptions[trait_name]
        if trait_value > 0.3:
            return positive
        elif trait_value < -0.3:
            return negative
    return None

def create_prompt_for_traits(traits):
    """Create a prompt that asks the model to exhibit specific traits."""
    # Find the dominant trait(s) in this example
    dominant_traits = []
    for trait_name, trait_value in traits.items():
        description = get_trait_description(trait_name, trait_value)
        if description:
            dominant_traits.append(description)
    
    if dominant_traits:
        trait_list = ", ".join(dominant_traits)
        prompt = f"Respond in a way that is {trait_list}:"
    else:
        prompt = "Respond naturally:"
    
    return prompt

def convert_file(input_path, output_path):
    """Convert a single JSONL file to Together AI format."""
    print(f"Converting {input_path.name} -> {output_path.name}")
    
    converted_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the original format
                data = json.loads(line.strip())
                
                # Extract the components
                utterance = data.get('utterance', '')
                traits = data.get('traits', {})
                
                # Create prompt based on traits
                prompt = create_prompt_for_traits(traits)
                
                # Convert to Together AI format
                together_format = {
                    "prompt": prompt,
                    "completion": utterance
                }
                
                # Write the converted line
                outfile.write(json.dumps(together_format) + '\n')
                converted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"  Warning: Could not parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"  Converted {converted_count} examples")
    return converted_count

def main():
    # Set up paths
    input_dir = Path("shards")
    output_dir = Path("shards/together_format")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("🔄 Converting JSONL files to Together AI format...")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    
    total_files = 0
    total_examples = 0
    
    # Process all JSONL files
    for input_file in input_dir.glob("*.jsonl"):
        output_file = output_dir / input_file.name
        
        try:
            count = convert_file(input_file, output_file)
            total_files += 1
            total_examples += count
        except Exception as e:
            print(f"❌ Error converting {input_file.name}: {e}")
    
    print(f"\n✅ Conversion complete!")
    print(f"📊 Converted {total_files} files with {total_examples} total examples")
    print(f"📁 Converted files are in: {output_dir}")

if __name__ == "__main__":
    main() 