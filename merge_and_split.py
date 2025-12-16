#!/usr/bin/env python3
"""
Merge original dataset with seed.txt and create proper splits.
Handles the issue of augmented data by using only unique base examples.
"""

import json
import random
import os


def fix_encoding(text):
    """Fix UTF-8 encoding issues."""
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '—',
        'â€"': '–',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue
    return data


def save_jsonl(data, filepath):
    """Save to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    print("="*70)
    print("MERGING AND SPLITTING DATA")
    print("="*70)
    
    # 1. Load original dataset (before augmentation)
    print("\n1. Loading original dataset...")
    original_file = "data/irt_1000_pairs.jsonl"
    
    if not os.path.exists(original_file):
        print(f"Error: {original_file} not found")
        return
    
    original_data = load_jsonl(original_file)
    print(f"   Loaded {len(original_data)} entries")
    
    # 2. Deduplicate by nightmare text
    print("\n2. Deduplicating original data...")
    seen_nightmares = set()
    unique_original = []
    
    for item in original_data:
        nm = fix_encoding(item['nightmare'])
        if nm not in seen_nightmares:
            seen_nightmares.add(nm)
            unique_original.append({
                'nightmare': nm,
                'comfort_rewrite': fix_encoding(item.get('comfort_rewrite', item.get('comfort', '')))
            })
    
    print(f"   Unique nightmare-comfort pairs: {len(unique_original)}")
    
    # 3. Load seed.txt
    print("\n3. Loading seed.txt...")
    seed_file = "seed.txt"
    
    if os.path.exists(seed_file):
        with open(seed_file, 'r', encoding='utf-8') as f:
            seed_lines = [fix_encoding(line.strip()) for line in f if line.strip()]
        
        # Deduplicate seeds
        unique_seeds = list(dict.fromkeys(seed_lines))
        print(f"   Total seeds: {len(seed_lines)}")
        print(f"   Unique seeds: {len(unique_seeds)}")
        
        # Find seeds NOT in original data
        new_seeds = [s for s in unique_seeds if s not in seen_nightmares]
        print(f"   New seeds (not in original): {len(new_seeds)}")
    else:
        new_seeds = []
        print(f"   seed.txt not found, skipping")
    
    # 4. Split unique original data
    print(f"\n4. Splitting {len(unique_original)} unique pairs...")
    random.seed(42)
    random.shuffle(unique_original)
    
    # 70% train, 15% val, 15% test
    n = len(unique_original)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_data = unique_original[:train_end]
    val_data = unique_original[train_end:val_end]
    test_data = unique_original[val_end:]
    
    print(f"   Train: {len(train_data)} ({len(train_data)/n*100:.0f}%)")
    print(f"   Val:   {len(val_data)} ({len(val_data)/n*100:.0f}%)")
    print(f"   Test:  {len(test_data)} ({len(test_data)/n*100:.0f}%)")
    
    # 5. Save all splits
    print("\n5. Saving files...")
    
    save_jsonl(train_data, "data/final_train.jsonl")
    print(f"   ✓ data/final_train.jsonl")
    
    save_jsonl(val_data, "data/final_val.jsonl")
    print(f"   ✓ data/final_val.jsonl")
    
    save_jsonl(test_data, "data/final_test.jsonl")
    print(f"   ✓ data/final_test.jsonl")
    
    # Extract nightmares from train for fear model
    train_nightmares = [item['nightmare'] for item in train_data]
    with open("data/nightmares_train.txt", 'w', encoding='utf-8') as f:
        for nm in train_nightmares:
            f.write(nm + '\n')
    print(f"   ✓ data/nightmares_train.txt")
    
    # Save new seeds for PPO
    if new_seeds:
        with open("data/ppo_seeds_new.txt", 'w', encoding='utf-8') as f:
            for seed in new_seeds:
                f.write(seed + '\n')
        print(f"   ✓ data/ppo_seeds_new.txt ({len(new_seeds)} new scenarios)")
    
    # 6. Update config instructions
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    print("\nUpdate your config.yaml:")
    print("""
data:
  nightmares: "data/nightmares_train.txt"
  pairs_train: "data/final_train.jsonl"
  pairs_val: "data/final_val.jsonl"
  pairs_test: "data/final_test.jsonl"
  seeds_for_rl: "data/ppo_seeds_new.txt"
""")
    
    print("\nDataset Summary:")
    print(f"  Original unique pairs: {len(unique_original)}")
    print(f"  Training set: {len(train_data)}")
    print(f"  Validation set: {len(val_data)}")
    print(f"  Test set: {len(test_data)}")
    print(f"  Additional PPO seeds: {len(new_seeds)}")
    
    print("\nNext steps:")
    print("  1. Update config.yaml with paths above")
    print("  2. Train models: python train_comfort_val.py")
    print("  3. Train models: python train_fear_val.py")
    print("  4. Test: python test_models.py")
    print("  5. PPO: python train_ppo_marl.py")


if __name__ == "__main__":
    main()