#!/usr/bin/env python3
"""
Extract unique nightmares from training data to create nightmares.txt
Removes duplicates to prevent memorization during fear model training.
"""

import json
import os
from collections import Counter


def extract_nightmares(input_file: str, output_file: str, remove_duplicates: bool = True):
    """
    Extract nightmares from JSONL training data.
    
    Args:
        input_file: Path to JSONL file with nightmare-comfort pairs
        output_file: Path to output nightmares.txt
        remove_duplicates: If True, only keep unique nightmares
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print(f"Reading from: {input_file}")
    
    nightmares = []
    
    # Read JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                nightmare = data.get('nightmare', '')
                
                if nightmare:
                    nightmares.append(nightmare)
                else:
                    print(f"Warning: Line {line_num} has no nightmare field")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    print(f"Extracted {len(nightmares)} nightmares")
    
    # Remove duplicates if requested
    if remove_duplicates:
        original_count = len(nightmares)
        nightmares = list(dict.fromkeys(nightmares))  # Preserves order, removes duplicates
        duplicates_removed = original_count - len(nightmares)
        print(f"Removed {duplicates_removed} duplicates")
        print(f"Unique nightmares: {len(nightmares)}")
    
    # Show statistics
    if len(nightmares) > 0:
        lengths = [len(n.split()) for n in nightmares]
        avg_length = sum(lengths) / len(lengths)
        print(f"\nStatistics:")
        print(f"  Average length: {avg_length:.1f} words")
        print(f"  Shortest: {min(lengths)} words")
        print(f"  Longest: {max(lengths)} words")
    
    # Write to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for nightmare in nightmares:
            f.write(nightmare + '\n')
    
    print(f"\nSaved to: {output_file}")
    
    # Show sample
    print(f"\nFirst 5 examples:")
    for i, nightmare in enumerate(nightmares[:5], 1):
        print(f"  {i}. {nightmare}")


def show_duplicate_analysis(input_file: str):
    """Analyze how many duplicates exist in the dataset."""
    
    nightmares = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    nightmare = data.get('nightmare', '')
                    if nightmare:
                        nightmares.append(nightmare)
                except:
                    continue
    
    # Count occurrences
    counts = Counter(nightmares)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    
    print("\n" + "="*70)
    print("DUPLICATE ANALYSIS")
    print("="*70)
    print(f"Total entries: {len(nightmares)}")
    print(f"Unique nightmares: {len(counts)}")
    print(f"Nightmares with duplicates: {len(duplicates)}")
    
    if duplicates:
        print(f"\nMost duplicated (top 10):")
        for nightmare, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {count}x: {nightmare[:60]}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract nightmares from training data")
    parser.add_argument("--input", default="data/irt_augmented.jsonl", 
                       help="Input JSONL file")
    parser.add_argument("--output", default="data/nightmares.txt",
                       help="Output text file")
    parser.add_argument("--keep-duplicates", action="store_true",
                       help="Keep duplicate nightmares (not recommended)")
    parser.add_argument("--analyze", action="store_true",
                       help="Show duplicate analysis")
    args = parser.parse_args()
    
    # Show analysis if requested
    if args.analyze:
        show_duplicate_analysis(args.input)
        print()
    
    # Extract nightmares
    extract_nightmares(
        args.input,
        args.output,
        remove_duplicates=not args.keep_duplicates
    )
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print("\nNext step: Retrain fear model")
    print("  python train_fear.py")


if __name__ == "__main__":
    main()