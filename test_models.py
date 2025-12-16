#!/usr/bin/env python3
"""
Test Mistral-7B SFT Models
"""

import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_config(path: str = "config.yaml"):
    """Load configuration."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_model(checkpoint_dir: str, prompts: list, max_new_tokens: int = 50):
    """Test a Mistral model with given prompts."""
    print(f"\nLoading from: {checkpoint_dir}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    
    # Load with quantization for testing
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
    
    model.eval()
    
    results = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(result)
    
    return results


def main():
    cfg = load_config()
    
    comfort_dir = cfg["checkpoints"]["sft_comfort"]
    fear_dir = cfg["checkpoints"]["sft_fear"]
    
    print("=" * 70)
    print("TESTING MISTRAL-7B SFT MODELS")
    print("=" * 70)
    
    # Check if models exist
    if not os.path.exists(comfort_dir):
        print(f"\nError: Comfort model not found at {comfort_dir}")
        print("Train it first: python train_comfort.py")
        return
    
    if not os.path.exists(fear_dir):
        print(f"\nError: Fear model not found at {fear_dir}")
        print("Train it first: python train_fear.py")
        return
    
    # Test prompts
    comfort_prompts = [
        "<NIGHTMARE> I have a presentation tomorrow\n<COMFORT>",
        "<NIGHTMARE> Going to the dentist\n<COMFORT>",
        "<NIGHTMARE> Meeting new people\n<COMFORT>"
    ]
    
    fear_prompts = [
        "<NIGHTMARE> I have a presentation tomorrow\n<FEAR>",
        "<NIGHTMARE> Going to the dentist\n<FEAR>",
        "<NIGHTMARE> Meeting new people\n<FEAR>"
    ]
    
    # Test Comfort Model
    print("\n" + "=" * 70)
    print("COMFORT MODEL (Mistral-7B)")
    print("=" * 70)
    
    try:
        comfort_results = test_model(comfort_dir, comfort_prompts)
        for prompt, result in zip(comfort_prompts, comfort_results):
            nightmare = prompt.split("<NIGHTMARE>")[1].split("\n<COMFORT>")[0].strip()
            print(f"\nNightmare: {nightmare}")
            print(f"Response: {result}")
            print("-" * 70)
    except Exception as e:
        print(f"Error testing comfort model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test Fear Model
    print("\n" + "=" * 70)
    print("FEAR MODEL (Mistral-7B)")
    print("=" * 70)
    
    try:
        fear_results = test_model(fear_dir, fear_prompts)
        for prompt, result in zip(fear_prompts, fear_results):
            nightmare = prompt.split("<NIGHTMARE>")[1].split("\n<FEAR>")[0].strip()
            print(f"\nNightmare: {nightmare}")
            print(f"Response: {result}")
            print("-" * 70)
    except Exception as e:
        print(f"Error testing fear model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("ALL MODELS WORKING")
    print("=" * 70)
    print("\nExpected improvements with Mistral-7B:")
    print("  - Less repetition")
    print("  - More coherent and natural language")
    print("  - Better understanding of context")
    print("  - More varied therapeutic responses")
    print("\nIf outputs look good, proceed to PPO training:")
    print("  python train_ppo_marl.py --episodes 20 --turns 6")


if __name__ == "__main__":
    main()