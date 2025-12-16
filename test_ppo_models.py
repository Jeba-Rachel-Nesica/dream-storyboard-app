#!/usr/bin/env python3
"""
Test PPO-trained models
Compare SFT vs PPO performance
"""

import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_config(path: str = "config.yaml"):
    """Load configuration."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_conversation(nightmare: str, comfort_model, comfort_tok, fear_model, fear_tok, turns: int = 4):
    """Generate a multi-turn conversation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    history = []
    prompt = f"<NIGHTMARE> {nightmare}"
    
    print(f"\n😰 Nightmare: {nightmare}")
    print("-" * 70)
    
    for turn in range(turns):
        if turn % 2 == 0:
            # Fear's turn
            model, tokenizer, role = fear_model, fear_tok, "<FEAR>"
            print(f"\n😱 Fear Agent:")
        else:
            # Comfort's turn
            model, tokenizer, role = comfort_model, comfort_tok, "<COMFORT>"
            print(f"\n💙 Comfort Agent:")
        
        # Build prompt with history
        full_prompt = prompt
        for h_role, h_text in history:
            full_prompt += f"\n{h_role} {h_text}"
        full_prompt += f"\n{role} "
        
        # Generate
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Clean up
        for token in ["<NIGHTMARE>", "<FEAR>", "<COMFORT>"]:
            response = response.split(token)[0].strip()
        
        print(response)
        history.append((role, response))
    
    print("-" * 70)


def main():
    cfg = load_config()
    
    comfort_sft_dir = cfg["checkpoints"]["sft_comfort"]
    fear_sft_dir = cfg["checkpoints"]["sft_fear"]
    comfort_ppo_dir = cfg["checkpoints"]["ppo_comfort"]
    fear_ppo_dir = cfg["checkpoints"]["ppo_fear"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("COMPARING SFT VS PPO MODELS")
    print("=" * 70)
    
    # Check if PPO models exist
    if not os.path.exists(comfort_ppo_dir) or not os.path.exists(fear_ppo_dir):
        print("\n❌ PPO models not found. Train them first:")
        print("  python train_ppo_marl.py --episodes 100 --turns 6")
        return
    
    # Load SFT models
    print("\nLoading SFT models...")
    comfort_sft = AutoModelForCausalLM.from_pretrained(comfort_sft_dir).to(device)
    comfort_sft_tok = AutoTokenizer.from_pretrained(comfort_sft_dir)
    fear_sft = AutoModelForCausalLM.from_pretrained(fear_sft_dir).to(device)
    fear_sft_tok = AutoTokenizer.from_pretrained(fear_sft_dir)
    
    # Load PPO models
    print("Loading PPO models...")
    comfort_ppo = AutoModelForCausalLM.from_pretrained(comfort_ppo_dir).to(device)
    comfort_ppo_tok = AutoTokenizer.from_pretrained(comfort_ppo_dir)
    fear_ppo = AutoModelForCausalLM.from_pretrained(fear_ppo_dir).to(device)
    fear_ppo_tok = AutoTokenizer.from_pretrained(fear_ppo_dir)
    
    comfort_sft.eval()
    fear_sft.eval()
    comfort_ppo.eval()
    fear_ppo.eval()
    
    # Test nightmares
    test_nightmares = [
        "I have an important presentation tomorrow and I'm not ready",
        "I'm lost in a dark forest with no way out",
        "Everyone is staring at me and laughing"
    ]
    
    # Test each nightmare
    for nightmare in test_nightmares:
        print("\n" + "=" * 70)
        print("BASELINE (SFT ONLY)")
        print("=" * 70)
        generate_conversation(nightmare, comfort_sft, comfort_sft_tok, fear_sft, fear_sft_tok, turns=4)
        
        print("\n" + "=" * 70)
        print("AFTER PPO TRAINING")
        print("=" * 70)
        generate_conversation(nightmare, comfort_ppo, comfort_ppo_tok, fear_ppo, fear_ppo_tok, turns=4)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nExpected improvements after PPO:")
    print("  - More coherent dialogue")
    print("  - Better emotional progression")
    print("  - More therapeutic comfort responses")
    print("  - Appropriate fear escalation/de-escalation")


if __name__ == "__main__":
    main()