#!/usr/bin/env python3
"""
Quick test script for GPT2-medium comfort model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Test nightmares
test_nightmares = [
    "I have an important presentation tomorrow and I'm not ready",
    "I'm falling from a tall building",
    "Everyone is staring at me and laughing",
    "I'm trapped in a small space with no way out",
    "I'm being chased through dark streets",
]

print("=" * 70)
print("TESTING GPT2-MEDIUM COMFORT MODEL")
print("=" * 70)

# Load model
model_path = "checkpoints/comfort_large"
print(f"\nLoading from: {model_path}\n")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Test each nightmare
for i, nightmare in enumerate(test_nightmares, 1):
    prompt = f"<NIGHTMARE> {nightmare}\n<COMFORT>"
    
    print(f"\n{i}. Nightmare: {nightmare}")
    print("-" * 70)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            min_new_tokens=20,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the comfort response
    comfort_response = full_text.split("<COMFORT>")[-1].strip()
    
    # Clean up any leaked tokens
    for token in ["<NIGHTMARE>", "<FEAR>"]:
        comfort_response = comfort_response.split(token)[0].strip()
    
    print(f"Comfort: {comfort_response}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)